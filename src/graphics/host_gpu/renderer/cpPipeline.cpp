#include "graphics/host_gpu/renderer/cpPipeline.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>

namespace Libs::Graphics::CpPipeline {

namespace {

constexpr uint64_t Slots = 32;
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

// Front: wait until `done()` holds, serving re-lookup requests unless inside a lookup.
template <typename Done>
void FrontWait(Done&& done) noexcept {
	auto&      s     = S();
	const auto start = std::chrono::steady_clock::now();
	for (;;) {
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
		if (!done() && (t_in_lookup || s.request.load(std::memory_order_seq_cst) == nullptr)) {
			s.back_event.wait(event, std::memory_order_seq_cst);
		}
		s.front_sleeping.store(false, std::memory_order_seq_cst);
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
		const auto start = NowNs();
		s.execute(s.context, Slot(tail));
		s.back_busy_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
		s.tail.store(tail + 1, std::memory_order_seq_cst);
		NotifyFront();
	}
}

} // namespace

int Mode() noexcept {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_CP_PIPELINE");
		const int   m     = value != nullptr ? std::atoi(value) : 0;
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

void Drain() noexcept {
	if (!FrontActive() || S().slots == nullptr) return;
	auto& s = S();
	if (s.tail.load(std::memory_order_acquire) != s.head.load(std::memory_order_relaxed)) {
		const auto start = NowNs();
		FrontWait([&] {
			if (s.tail.load(std::memory_order_acquire) == s.head.load(std::memory_order_relaxed)) {
				return true;
			}
			return t_in_lookup && s.request.load(std::memory_order_acquire) != nullptr;
		});
		s.drains_waited.fetch_add(1, std::memory_order_relaxed);
		s.drain_wait_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	}
	FrontWait([&] {
		if (s.tail.load(std::memory_order_acquire) == s.head.load(std::memory_order_relaxed)) {
			return true;
		}
		// Parked on a request this lookup cannot serve: it touches nothing until served.
		return t_in_lookup && s.request.load(std::memory_order_acquire) != nullptr;
	});
	if (!t_in_lookup) ApplyFeedback();
}

void Poll() noexcept {
	if (S().slots == nullptr || t_in_lookup || !FrontActive()) return;
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
	            " back_busy=%.0fms\n",
	            seconds, st[0], st[1], st[2], st[3], st[4], st[5], st[6], drains, ms(drain), fulls, ms(full),
	            take(s.relookups), ms(take(s.back_busy_ns)));
}

} // namespace Libs::Graphics::CpPipeline
