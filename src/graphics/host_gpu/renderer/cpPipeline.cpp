#include "graphics/host_gpu/renderer/cpPipeline.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cpOrderLedger.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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
thread_local bool t_serving     = false;
HeapFn            g_heap_hook   = nullptr;

// KYTY_CP_ORDER_CHECK state. The mutex serializes the front's reads and the back's writes.
struct OrderHazard {
	CpOrderLedger::Hit hit;
	OrderWrite         kind;
	bool               window;
	uint64_t           reader_hash, writer_hash;
	bool               printed;
};
struct Order {
	std::mutex                  mutex;
	CpOrderLedger               ledger;
	std::atomic<uint64_t>       serial {0}; // ledger.Serial(), for lookups to snapshot
	std::vector<OrderHazard>    first;      // the first 20
	std::array<std::pair<uint64_t, uint64_t>, Slots * 2> reader_hash {}; // reader + 1, hash
	enum Counter : uint32_t {
		Lookups, LookupsQueued, DepthSum, Reads, DirtySet, BufferWrites, ImageWrites, HzBuffer, HzImage,
		WinBuffer, WinImage, FrontNs, BackNs, ReadsEmpty, WinOverflow, LedgerSum, CounterCount
	};
	std::array<std::atomic<uint64_t>, CounterCount> n {};
	std::array<uint64_t, CounterCount>              total {};
	std::atomic<uint64_t>                           max_depth {0}; // since launch
	uint64_t                                        reported_meets = 0; // under mutex
};
Order&                O() noexcept {
	static Order order;
	return order;
}
thread_local uint64_t t_lookup_serial = 0;
thread_local bool     t_order_queued  = false; // records were queued when the lookup began
thread_local uint64_t t_back_hash     = 0;

DrawRecord& Slot(uint64_t index) noexcept {
	return (*S().slots)[index % Slots];
}

void Serve() noexcept {
	auto& s   = S();
	auto* rec = s.request.load(std::memory_order_acquire);
	if (rec == nullptr) return;
	t_serving = true; // its reads belong to the parked record, not to the next one
	s.relookup(s.context, *rec);
	t_serving = false;
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
	t_in_lookup    = on;
	t_order_queued = false;
	if (on && OrderCheck() && !t_serving && !t_back && S().slots != nullptr) [[unlikely]] {
		auto& o = O();
		// Serial first: a write noted before the depth is read is then still in the window.
		t_lookup_serial  = o.serial.load(std::memory_order_acquire);
		const auto depth = Published() - Finished();
		t_order_queued   = depth != 0;
		o.n[Order::Lookups].fetch_add(1, std::memory_order_relaxed);
		if (depth != 0) {
			o.n[Order::LookupsQueued].fetch_add(1, std::memory_order_relaxed);
			o.n[Order::DepthSum].fetch_add(depth, std::memory_order_relaxed);
			if (depth > o.max_depth.load(std::memory_order_relaxed)) o.max_depth.store(depth, std::memory_order_relaxed);
		}
	}
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

bool OrderCheck() noexcept {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_CP_ORDER_CHECK");
		const bool  set   = value != nullptr && std::strcmp(value, "1") == 0 && Mode() == 1;
		if (set) {
			std::printf("KYTY_CP_ORDER_CHECK=1: pipelined lookup reads paired with queued draw writes\n");
		}
		return set;
	}();
	return on;
}

bool OrderActive() noexcept {
	// The queue at lookup start: the back may empty it during the walk, and its writes then are the
	// window hits.
	return OrderCheck() && t_in_lookup && !t_serving && !t_back && t_order_queued;
}

namespace {

uint64_t OrderReaderHash(const Order& o, uint64_t reader) noexcept {
	const auto& entry = o.reader_hash[reader % o.reader_hash.size()];
	return entry.first == reader + 1 ? entry.second : 0;
}

// Under o.mutex.
void OrderHazardHit(Order& o, const CpOrderLedger::Hit& hit, OrderWrite kind, bool window,
                    uint64_t writer_hash) noexcept {
	const bool image = kind == OrderWrite::Image;
	o.n[window ? (image ? Order::WinImage : Order::WinBuffer) : (image ? Order::HzImage : Order::HzBuffer)]
	    .fetch_add(1, std::memory_order_relaxed);
	if (o.first.size() < 20) {
		o.first.push_back({hit, kind, window, OrderReaderHash(o, hit.reader), writer_hash, false});
	}
}

} // namespace

void OrderNoteRead(uint64_t vaddr, uint64_t size, bool dirty_set) noexcept {
	auto&      o     = O();
	const auto start = NowNs();
	o.n[Order::Reads].fetch_add(1, std::memory_order_relaxed);
	if (dirty_set) {
		o.n[Order::DirtySet].fetch_add(1, std::memory_order_relaxed);
	}
	{
		std::lock_guard lock(o.mutex);
		const auto finished = Finished();
		if (finished >= Published()) {
			o.n[Order::ReadsEmpty].fetch_add(1, std::memory_order_relaxed);
		}
		if (o.ledger.WindowOverflow(t_lookup_serial)) {
			o.n[Order::WinOverflow].fetch_add(1, std::memory_order_relaxed);
		}
		o.ledger.Prune(finished);
		o.ledger.AddRead(vaddr, vaddr + size, Published(), finished, t_lookup_serial,
		                 [&](const CpOrderLedger::Hit& hit) {
			                 OrderHazardHit(o, hit, static_cast<OrderWrite>(hit.tag), true, 0);
		                 });
	}
	o.n[Order::FrontNs].fetch_add(NowNs() - start, std::memory_order_relaxed);
}

void OrderLookupDone(uint64_t shader_hash) noexcept {
	if (!OrderCheck() || t_serving || S().slots == nullptr) return;
	auto&           o      = O();
	const auto      reader = Published();
	std::lock_guard lock(o.mutex);
	o.reader_hash[reader % o.reader_hash.size()] = {reader + 1, shader_hash};
	for (auto& hazard: o.first) {
		if (hazard.hit.reader == reader && hazard.reader_hash == 0) hazard.reader_hash = shader_hash;
	}
}

void OrderSetBackDraw(uint64_t shader_hash) noexcept {
	t_back_hash = shader_hash;
}

void OrderNoteWrite(uint64_t vaddr, uint64_t size, OrderWrite kind) noexcept {
	if (!OrderCheck() || !t_back) return;
	auto&      o     = O();
	const auto start = NowNs();
	o.n[kind == OrderWrite::Image ? Order::ImageWrites : Order::BufferWrites].fetch_add(1, std::memory_order_relaxed);
	{
		std::lock_guard lock(o.mutex);
		// On the back thread, the tail is the index of the record it runs.
		o.ledger.Write(vaddr, vaddr + size, S().tail.load(std::memory_order_relaxed), static_cast<uint32_t>(kind),
		               [&](const CpOrderLedger::Hit& hit) { OrderHazardHit(o, hit, kind, false, t_back_hash); });
		o.serial.store(o.ledger.Serial(), std::memory_order_release);
		o.n[Order::LedgerSum].fetch_add(o.ledger.Reads(), std::memory_order_relaxed);
	}
	o.n[Order::BackNs].fetch_add(NowNs() - start, std::memory_order_relaxed);
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
	if (!OrderCheck()) return;
	auto&    o = O();
	uint64_t n[Order::CounterCount];
	for (uint32_t i = 0; i < Order::CounterCount; i++) {
		n[i] = take(o.n[i]);
		o.total[i] += n[i];
	}
	size_t   ledger = 0;
	uint64_t meets  = 0;
	{
		std::lock_guard lock(o.mutex);
		ledger           = o.ledger.Reads();
		meets            = o.ledger.GranuleMeets() - o.reported_meets;
		o.reported_meets = o.ledger.GranuleMeets();
		for (size_t i = 0; i < o.first.size(); i++) {
			auto& h = o.first[i];
			if (h.printed) continue;
			h.printed = true;
			std::printf("CP-ORDER hazard %zu: %s%s addr=0x%" PRIx64 " size=0x%" PRIx64 " reader=%" PRIu64
			            " writer=%" PRIu64 " distance=%" PRIu64 " reader_hash=0x%016" PRIx64 " writer_hash=0x%016" PRIx64 "\n",
			            i + 1, h.kind == OrderWrite::Image ? "image" : "buffer", h.window ? " window" : "", h.hit.address,
			            h.hit.size, h.hit.reader, h.hit.writer, h.hit.reader - h.hit.writer, h.reader_hash, h.writer_hash);
		}
	}
	const auto& t = o.total;
	std::printf("PERF cporder %.1fs: lookups=%" PRIu64 " queued_lookups=%" PRIu64 " avg_depth=%.1f max_depth=%" PRIu64
	            " reads=%" PRIu64 " buffer_writes=%" PRIu64 " image_writes=%" PRIu64 " hz_buffer=%" PRIu64
	            " hz_image=%" PRIu64 " hz_dirty_set=%" PRIu64 " win_buffer=%" PRIu64 " win_image=%" PRIu64
	            " ledger=%zu reads_empty=%" PRIu64 " win_overflow=%" PRIu64 " avg_ledger_at_write=%.1f granule_meets=%" PRIu64
	            " front=%.2fms back=%.2fms | total hz_buffer=%" PRIu64 " hz_image=%" PRIu64
	            " hz_dirty_set=%" PRIu64 " win_buffer=%" PRIu64 " win_image=%" PRIu64 " queued_lookups=%" PRIu64 "\n",
	            seconds, n[Order::Lookups], n[Order::LookupsQueued],
	            n[Order::LookupsQueued] != 0 ? static_cast<double>(n[Order::DepthSum]) / n[Order::LookupsQueued] : 0.0,
	            o.max_depth.load(std::memory_order_relaxed), n[Order::Reads], n[Order::BufferWrites], n[Order::ImageWrites], n[Order::HzBuffer],
	            n[Order::HzImage], n[Order::DirtySet], n[Order::WinBuffer], n[Order::WinImage], ledger, n[Order::ReadsEmpty], n[Order::WinOverflow],
	            n[Order::BufferWrites] + n[Order::ImageWrites] != 0
	                ? static_cast<double>(n[Order::LedgerSum]) / (n[Order::BufferWrites] + n[Order::ImageWrites])
	                : 0.0,
	            meets, ms(n[Order::FrontNs]),
	            ms(n[Order::BackNs]), t[Order::HzBuffer], t[Order::HzImage], t[Order::DirtySet], t[Order::WinBuffer],
	            t[Order::WinImage], t[Order::LookupsQueued]);
}

} // namespace Libs::Graphics::CpPipeline
