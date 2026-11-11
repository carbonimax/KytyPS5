#include "graphics/host_gpu/renderer/cpPipeline.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace Libs::Graphics::CpPipeline {

namespace {

constexpr uint64_t Slots      = 32;
constexpr uint32_t MaxServers = 8;
// Spin this long before sleeping on the other side (a draw's back part takes a few microseconds).
constexpr auto SpinTime = std::chrono::microseconds(50);

struct State {
	std::unique_ptr<std::array<DrawRecord, Slots>> slots;
	void*                                          context  = nullptr;
	ExecuteFn                                      execute  = nullptr;
	RelookupFn                                     relookup = nullptr;
	FeedbackFn                                     feedback = nullptr;

	// Records published by the front / finished by the back (monotonic counts).
	std::atomic<uint64_t> head {0};
	std::atomic<uint64_t> tail {0};
	// Front only: records whose guess feedback was applied.
	uint64_t reclaimed = 0;

	std::atomic<DrawRecord*> request {nullptr};
	// Bumped by the back after each record and each request; the front sleeps on it.
	std::atomic<uint32_t> back_event {0};
	std::atomic<bool>     back_sleeping {false};
	std::atomic<bool>     front_sleeping {false};

	std::array<std::atomic<uint64_t>, static_cast<size_t>(Stat::Count)> stats {};
	std::atomic<uint64_t> drains_waited {0};
	std::atomic<uint64_t> drain_wait_ns {0};
	std::atomic<uint64_t> full_waits {0};
	std::atomic<uint64_t> full_wait_ns {0};
	std::atomic<uint64_t> relookups {0};
	std::atomic<uint64_t> back_busy_ns {0};

	// KYTY_CP_LOOKUP_STAGE: each server's queue of record indices (a record in flight is in at most
	// one, so Slots entries are enough).
	struct Server {
		std::array<uint64_t, Slots> items {};
		std::atomic<uint64_t>       head {0};
		std::atomic<uint64_t>       tail {0};
		std::atomic<bool>           sleeping {false};
	};
	std::unique_ptr<std::array<Server, MaxServers>> servers;
	LookupFn                                        lookup = nullptr;
	VerifyFn                                        verify = nullptr;
	std::atomic<bool>                               back_waits_lookup {false};
	// Back only: records below this index are looked up again (a staged draw wrote memory).
	uint64_t                                        relook_until = 0;
	std::atomic<uint64_t>                           lookup_wait_ns {0};
	std::atomic<uint64_t>                           server_busy_ns {0};
	// A server's guest memory fault, resolved by the front (one at a time).
	struct FaultRequest {
		bool (*fn)(void*) = nullptr;
		void* arg         = nullptr;
		bool  result      = false;
	};
	std::mutex                  fault_mutex;
	std::atomic<FaultRequest*>  fault_request {nullptr};
	std::atomic<uint64_t>       server_faults {0};
};

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

State& S() noexcept {
	static State state;
	return state;
}

thread_local bool t_back        = false;
thread_local bool t_in_lookup   = false;
thread_local bool t_heap_inline = false;
thread_local bool t_off_front   = false;
thread_local bool t_aborted     = false;
thread_local bool t_server      = false;
// Front: inside ServeFault (a back parked on a staged lookup then counts as idle).
thread_local bool t_serving_fault = false;
HeapFn            g_heap_hook   = nullptr;

DrawRecord& Slot(uint64_t index) noexcept {
	return (*S().slots)[index % Slots];
}

void Serve() noexcept {
	auto& s   = S();
	auto* rec = s.request.load(std::memory_order_acquire);
	if (rec == nullptr) return;
	s.relookup(s.context, *rec);
	s.request.store(nullptr, std::memory_order_seq_cst);
	s.request.notify_one();
}

void ApplyFeedback() noexcept {
	auto&      s    = S();
	const auto tail = s.tail.load(std::memory_order_acquire);
	while (s.reclaimed < tail) {
		s.feedback(s.context, Slot(s.reclaimed));
		s.reclaimed++;
	}
}

void NotifyFront() noexcept {
	auto& s = S();
	s.back_event.fetch_add(1, std::memory_order_seq_cst);
	if (s.front_sleeping.load(std::memory_order_seq_cst)) {
		s.back_event.notify_one();
	}
}

// Front: resolves a lookup server's guest memory fault on the GPU thread (KYTY_CP_LOOKUP_STAGE).
void ServeFault() noexcept {
	auto& s   = S();
	auto* req = s.fault_request.load(std::memory_order_acquire);
	if (req == nullptr) return;
	t_serving_fault = true;
	req->result     = req->fn(req->arg);
	t_serving_fault = false;
	s.fault_request.store(nullptr, std::memory_order_seq_cst);
	s.fault_request.notify_all();
}

// Front: wait until `done()` holds, serving re-lookup requests unless inside a lookup, and server faults.
template <typename Done>
void FrontWait(Done&& done) noexcept {
	auto&      s     = S();
	const auto start = std::chrono::steady_clock::now();
	for (;;) {
		if (!t_serving_fault && s.fault_request.load(std::memory_order_acquire) != nullptr) {
			ServeFault();
			continue;
		}
		if (!t_in_lookup && s.request.load(std::memory_order_acquire) != nullptr) {
			Serve();
			continue;
		}
		const auto event = s.back_event.load(std::memory_order_seq_cst);
		if (done()) return;
		if (std::chrono::steady_clock::now() - start < SpinTime) {
			std::this_thread::yield();
			continue;
		}
		s.front_sleeping.store(true, std::memory_order_seq_cst);
		if (!done() && (t_in_lookup || s.request.load(std::memory_order_seq_cst) == nullptr) &&
		    (t_serving_fault || s.fault_request.load(std::memory_order_seq_cst) == nullptr)) {
			s.back_event.wait(event, std::memory_order_seq_cst);
		}
		s.front_sleeping.store(false, std::memory_order_seq_cst);
	}
}

// Front serving a server fault: a back parked on a staged lookup touches nothing until it is done.
bool ParkedForFault() noexcept {
	return t_serving_fault && S().back_waits_lookup.load(std::memory_order_acquire);
}

void ServerThread(State::Server& server) {
	KYTY_PROFILER_THREAD("Thread_GpuLookup");
	t_off_front = true;
	t_server    = true;
	auto& s     = S();
	for (;;) {
		const auto tail = server.tail.load(std::memory_order_relaxed);
		if (server.head.load(std::memory_order_acquire) == tail) {
			const auto start = std::chrono::steady_clock::now();
			while (server.head.load(std::memory_order_acquire) == tail &&
			       std::chrono::steady_clock::now() - start < SpinTime) {
				std::this_thread::yield();
			}
			while (server.head.load(std::memory_order_seq_cst) == tail) {
				server.sleeping.store(true, std::memory_order_seq_cst);
				if (server.head.load(std::memory_order_seq_cst) == tail) {
					server.head.wait(tail, std::memory_order_seq_cst);
				}
				server.sleeping.store(false, std::memory_order_seq_cst);
			}
		}
		auto&      rec   = Slot(server.items[tail % Slots]);
		const auto start = NowNs();
		s.lookup(s.context, rec);
		const bool aborted = std::exchange(t_aborted, false);
		s.server_busy_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
		server.tail.store(tail + 1, std::memory_order_relaxed);
		rec.lookup.store(aborted ? LookupAborted : LookupDone, std::memory_order_seq_cst);
		if (s.back_waits_lookup.load(std::memory_order_seq_cst)) {
			rec.lookup.notify_all();
		}
	}
}

// Back: waits for the staged record's lookup, then has the front redo it if it cannot be used.
void FinishStaged(DrawRecord& rec, uint64_t index) noexcept {
	auto& s = S();
	if (rec.lookup.load(std::memory_order_acquire) == LookupPending) {
		const auto start = NowNs();
		const auto spin  = std::chrono::steady_clock::now();
		while (rec.lookup.load(std::memory_order_acquire) == LookupPending &&
		       std::chrono::steady_clock::now() - spin < SpinTime) {
			std::this_thread::yield();
		}
		s.back_waits_lookup.store(true, std::memory_order_seq_cst);
		// A front draining inside a server fault it serves waits for this (ParkedForFault).
		NotifyFront();
		while (rec.lookup.load(std::memory_order_seq_cst) == LookupPending) {
			rec.lookup.wait(LookupPending, std::memory_order_seq_cst);
		}
		s.back_waits_lookup.store(false, std::memory_order_seq_cst);
		s.lookup_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	}
	const bool aborted = rec.lookup.load(std::memory_order_acquire) == LookupAborted;
	if (aborted || index < s.relook_until) {
		Note(aborted ? Stat::StagedAborts : Stat::StagedStale);
		// The draw starts before the front's lookup acquires image heap views; the back runs it.
		g_heap_hook(s.context);
		rec.heap_begun             = true;
		rec.relookup_single_sample = rec.guessed_single_sample;
		RequestRelookup(rec);
		s.relookups.fetch_sub(1, std::memory_order_relaxed);
	}
	if (s.verify != nullptr && LookupVerify()) {
		t_off_front   = true;
		const bool eq = s.verify(s.context, rec);
		t_off_front   = false;
		if (!std::exchange(t_aborted, false)) {
			Note(Stat::LookupVerified);
			if (!eq) Note(Stat::LookupMismatch);
		}
	}
}

void BackThread() {
	KYTY_PROFILER_THREAD("Thread_GpuBack");
	t_back = true;
	// The back part runs GPU-thread code paths (caches, recording, inline fault resolution).
	GuestGpu::MarkHostGpuThread();
	PerfStats::MarkGpuThread();
	auto& s = S();
	for (;;) {
		const auto tail = s.tail.load(std::memory_order_relaxed);
		if (s.head.load(std::memory_order_acquire) == tail) {
			const auto start = std::chrono::steady_clock::now();
			while (s.head.load(std::memory_order_acquire) == tail &&
			       std::chrono::steady_clock::now() - start < SpinTime) {
				std::this_thread::yield();
			}
			while (s.head.load(std::memory_order_seq_cst) == tail) {
				s.back_sleeping.store(true, std::memory_order_seq_cst);
				if (s.head.load(std::memory_order_seq_cst) == tail) {
					s.head.wait(tail, std::memory_order_seq_cst);
				}
				s.back_sleeping.store(false, std::memory_order_seq_cst);
			}
		}
		auto& rec = Slot(tail);
		if (rec.staged) FinishStaged(rec, tail);
		const auto start = NowNs();
		s.execute(s.context, rec);
		s.back_busy_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
		if (rec.staged && rec.writes) {
			// Records published so far may have read what it writes before it ran.
			Note(Stat::StagedWrites);
			s.relook_until = s.head.load(std::memory_order_acquire);
		}
		s.tail.store(tail + 1, std::memory_order_seq_cst);
		NotifyFront();
	}
}

} // namespace

int Mode() noexcept {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_CP_PIPELINE");
		const int   m     = value != nullptr ? std::atoi(value) : 1;
		return m == 1 || m == 2 ? m : 0;
	}();
	return mode;
}

bool Engaged() noexcept {
	static const uint64_t min_frame_us = [] {
		const char* value = std::getenv("KYTY_CP_PIPELINE_MIN_FRAME_MS");
		return value != nullptr ? std::strtoull(value, nullptr, 10) * 1000ull : 25000ull;
	}();
	return Mode() == 2 || PerfStats::RecentFrameUs() >= min_frame_us;
}

void Init(void* context, ExecuteFn execute, RelookupFn relookup, FeedbackFn feedback) {
	auto& s = S();
	if (s.slots != nullptr || Mode() == 0) return;
	s.context  = context;
	s.execute  = execute;
	s.relookup = relookup;
	s.feedback = feedback;
	s.slots    = std::make_unique<std::array<DrawRecord, Slots>>();
	if (Mode() == 1) {
		// Detached: it only ever waits for work or runs a record, and dies with the process.
		std::thread(BackThread).detach();
	}
}

uint32_t LookupServers() noexcept {
	static const uint32_t servers = [] {
		const char* value = std::getenv("KYTY_CP_LOOKUP_STAGE");
		const int   k     = value != nullptr ? std::atoi(value) : 0;
		return Mode() == 1 ? static_cast<uint32_t>(std::clamp(k, 0, static_cast<int>(MaxServers))) : 0u;
	}();
	return servers;
}

bool LookupVerify() noexcept {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_CP_LOOKUP_STAGE_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return verify;
}

void SetLookupHooks(LookupFn lookup, VerifyFn verify) noexcept {
	auto& s = S();
	if (s.servers != nullptr || LookupServers() == 0 || s.slots == nullptr) return;
	s.lookup  = lookup;
	s.verify  = verify;
	s.servers = std::make_unique<std::array<State::Server, MaxServers>>();
	for (uint32_t i = 0; i < LookupServers(); i++) {
		// Detached like the back thread.
		std::thread(ServerThread, std::ref((*s.servers)[i])).detach();
	}
}

void PublishStaged(uint64_t route) noexcept {
	auto&      s      = S();
	const auto head   = s.head.load(std::memory_order_relaxed);
	auto&      rec    = Slot(head);
	auto&      server = (*s.servers)[route % LookupServers()];
	rec.staged        = true;
	rec.writes        = false;
	rec.heap_begun    = false;
	rec.lookup.store(LookupPending, std::memory_order_relaxed);
	const auto at            = server.head.load(std::memory_order_relaxed);
	server.items[at % Slots] = head;
	server.head.store(at + 1, std::memory_order_seq_cst);
	if (server.sleeping.load(std::memory_order_seq_cst)) {
		server.head.notify_one();
	}
	Publish();
}

bool OffFrontLookup() noexcept {
	return t_off_front;
}

void AbortLookup() noexcept {
	t_aborted = true;
}

bool ResolveOnFront(bool (*fn)(void*), void* arg) noexcept {
	auto& s = S();
	s.server_faults.fetch_add(1, std::memory_order_relaxed);
	std::scoped_lock    lock(s.fault_mutex);
	State::FaultRequest req {.fn = fn, .arg = arg};
	s.fault_request.store(&req, std::memory_order_seq_cst);
	NotifyFront();
	while (s.fault_request.load(std::memory_order_acquire) != nullptr) {
		s.fault_request.wait(&req, std::memory_order_acquire);
	}
	return req.result;
}

bool IsLookupServer() noexcept {
	return t_server;
}

bool LookupWasAborted() noexcept {
	return t_aborted;
}

bool IsBackThread() noexcept {
	return t_back;
}

bool FrontActive() noexcept {
	return Mode() != 0 && !t_back && GuestGpu::IsGpuThread();
}

bool BackIdle() noexcept {
	auto& s = S();
	return s.tail.load(std::memory_order_acquire) == s.head.load(std::memory_order_acquire);
}

uint64_t Finished() noexcept {
	return S().tail.load(std::memory_order_acquire);
}

uint64_t Published() noexcept {
	return S().head.load(std::memory_order_acquire);
}

const DrawRecord& Queued(uint64_t index) noexcept {
	return Slot(index);
}

void Drain() noexcept {
	if (!FrontActive() || S().slots == nullptr) return;
	auto& s = S();
	if (s.tail.load(std::memory_order_acquire) != s.head.load(std::memory_order_relaxed)) {
		const auto start = NowNs();
		FrontWait([&] {
			if (s.tail.load(std::memory_order_acquire) == s.head.load(std::memory_order_relaxed)) {
				return true;
			}
			return (t_in_lookup && s.request.load(std::memory_order_acquire) != nullptr) || ParkedForFault();
		});
		s.drains_waited.fetch_add(1, std::memory_order_relaxed);
		s.drain_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	}
	FrontWait([&] {
		if (s.tail.load(std::memory_order_acquire) == s.head.load(std::memory_order_relaxed)) {
			return true;
		}
		// Parked on a request this lookup cannot serve: it touches nothing until served.
		return (t_in_lookup && s.request.load(std::memory_order_acquire) != nullptr) || ParkedForFault();
	});
	if (!t_in_lookup) ApplyFeedback();
}

void Poll() noexcept {
	if (S().slots == nullptr || t_in_lookup || !FrontActive()) return;
	if (S().fault_request.load(std::memory_order_acquire) != nullptr) ServeFault();
	if (S().request.load(std::memory_order_acquire) != nullptr) Serve();
}

void SetInLookup(bool on) noexcept {
	t_in_lookup = on;
}

bool InLookup() noexcept {
	return t_in_lookup;
}

DrawRecord& Acquire() noexcept {
	auto&      s    = S();
	const auto head = s.head.load(std::memory_order_relaxed);
	if (head - s.tail.load(std::memory_order_acquire) >= Slots) {
		const auto start = NowNs();
		FrontWait([&] { return head - s.tail.load(std::memory_order_acquire) < Slots; });
		s.full_waits.fetch_add(1, std::memory_order_relaxed);
		s.full_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	}
	ApplyFeedback();
	auto& rec          = Slot(head);
	rec.feedback_valid = false;
	rec.staged         = false;
	return rec;
}

void Publish() noexcept {
	auto&      s    = S();
	const auto head = s.head.load(std::memory_order_relaxed);
	if (Mode() == 2) {
		s.execute(s.context, Slot(head));
		s.head.store(head + 1, std::memory_order_release);
		s.tail.store(head + 1, std::memory_order_release);
		return;
	}
	s.head.store(head + 1, std::memory_order_seq_cst);
	if (s.back_sleeping.load(std::memory_order_seq_cst)) {
		s.head.notify_one();
	}
}

void RequestRelookup(DrawRecord& record) noexcept {
	auto& s = S();
	if (!t_back) {
		// Inline mode: the GPU thread runs the record outside any lookup.
		s.relookup(s.context, record);
		return;
	}
	s.relookups.fetch_add(1, std::memory_order_relaxed);
	s.request.store(&record, std::memory_order_seq_cst);
	NotifyFront();
	while (s.request.load(std::memory_order_acquire) != nullptr) {
		s.request.wait(&record, std::memory_order_acquire);
	}
}

void SetHeapHook(HeapFn hook) noexcept {
	g_heap_hook = hook;
}

void NoteHeapAcquire() noexcept {
	if (!t_in_lookup || t_heap_inline || S().slots == nullptr || !FrontActive()) return;
	Drain();
	g_heap_hook(S().context);
	t_heap_inline = true;
}

bool TakeHeapInline() noexcept {
	return std::exchange(t_heap_inline, false);
}

void Note(Stat stat) noexcept {
	S().stats[static_cast<size_t>(stat)].fetch_add(1, std::memory_order_relaxed);
}

void Report(double seconds) noexcept {
	if (Mode() == 0) return;
	auto&      s    = S();
	const auto take = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
	const auto ms   = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	uint64_t   st[static_cast<size_t>(Stat::Count)];
	for (size_t i = 0; i < std::size(st); i++) st[i] = take(s.stats[i]);
	const auto drains = take(s.drains_waited);
	const auto drain  = take(s.drain_wait_ns);
	const auto fulls  = take(s.full_waits);
	const auto full   = take(s.full_wait_ns);
	std::printf("PERF cppipe %.1fs: queued=%" PRIu64 " inline_heap=%" PRIu64 " inline_writes=%" PRIu64
	            " fallback_args=%" PRIu64 " fallback_targets=%" PRIu64 " fallback_shaders=%" PRIu64 " fallback_fast=%" PRIu64
	            " drain_waits=%" PRIu64 "(%.0fms) full_waits=%" PRIu64 "(%.0fms) relookups=%" PRIu64
	            " back_busy=%.0fms",
	            seconds, st[0], st[1], st[2], st[3], st[4], st[5], st[6], drains, ms(drain), fulls, ms(full),
	            take(s.relookups), ms(take(s.back_busy_ns)));
	if (LookupServers() != 0) {
		std::printf(" servers=%u lookup_wait=%.0fms server_busy=%.0fms staged_aborts=%" PRIu64
		            " staged_stale=%" PRIu64 " staged_writes=%" PRIu64 " lookup_verified=%" PRIu64
		            " lookup_mismatch=%" PRIu64 " server_faults=%" PRIu64,
		            LookupServers(), ms(take(s.lookup_wait_ns)), ms(take(s.server_busy_ns)), st[7], st[8],
		            st[9], st[11], st[10], take(s.server_faults));
	}
	std::printf("\n");
}

} // namespace Libs::Graphics::CpPipeline
