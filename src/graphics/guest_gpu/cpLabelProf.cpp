#include "graphics/guest_gpu/cpLabelProf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Libs::Graphics::CpLabelProf {

namespace {

constexpr uint32_t NoQueue   = UINT32_MAX;
constexpr uint32_t MaxQueues = 64;
constexpr uint32_t MaxEvents = 0x60;
constexpr int      Buckets   = 9;
// Histogram edges: 0.05, 0.2, 0.5, 1, 2, 5, 10, 20 ms.
constexpr std::array<uint64_t, Buckets - 1> BucketEdgesNs = {50'000,    200'000,    500'000,
                                                             1'000'000, 2'000'000,  5'000'000,
                                                             10'000'000, 20'000'000};

struct Hist {
	std::array<uint64_t, Buckets> n {};
	void Add(uint64_t ns) {
		int b = 0;
		while (b < Buckets - 1 && ns >= BucketEdgesNs[static_cast<size_t>(b)]) {
			b++;
		}
		n[static_cast<size_t>(b)]++;
	}
	[[nodiscard]] std::string Str() const {
		std::string s;
		for (size_t i = 0; i < n.size(); i++) {
			s += (i == 0 ? "" : ",") + std::to_string(n[i]);
		}
		return s;
	}
};

uint64_t Sat(uint64_t a, uint64_t b) {
	return a > b ? a - b : 0;
}

uint32_t ClassOf(uint32_t queue) {
	return queue == 0 ? 0u : 1u;
}

thread_local uint32_t t_queue   = NoQueue;
thread_local uint64_t t_enqueue = 0;
thread_local uint64_t t_start   = 0;

struct PendingWrite {
	uint32_t queue = NoQueue;
	uint64_t enq   = 0;
	uint64_t start = 0;
	uint64_t dec   = 0;
	uint64_t value = 0;
};

struct LabelRec {
	uint32_t                 queue = NoQueue;
	uint64_t                 enq   = 0;
	uint64_t                 start = 0;
	uint64_t                 dec   = 0;
	uint64_t                 vis   = 0;
	uint64_t                 value = 0;
	std::deque<PendingWrite> pending;
};

struct LabelAgg {
	uint64_t n = 0, cpu = 0, gpu_owned = 0, deferred = 0, enq_start = 0, start_dec = 0, dec_vis = 0,
	         delivered = 0;
	Hist enq_dec;
};

struct IrqPending {
	uint64_t enq = 0, start = 0, dec = 0;
};

struct IrqLast {
	uint64_t enq = 0, start = 0, dec = 0, trig = 0;
};

struct IrqAgg {
	uint64_t n = 0, enq_start = 0, start_dec = 0, dec_trig = 0, max_dec_trig = 0;
	Hist     enq_trig;
};

struct WaitState {
	bool     active = false;
	uint64_t addr   = 0;
	uint64_t start  = 0;
	uint64_t polls  = 0;
	uint32_t pred   = 3;
};

// Producer classes: 0 = graphics ring, 1 = a compute ring, 2 = the waiting ring itself, 3 = unknown (guest
// CPU, DMA or an untracked write).
struct WaitAgg {
	uint64_t n = 0, imm = 0, dur = 0, unsub = 0, pick = 0, proc = 0, gpu = 0, recheck = 0, max = 0, polls = 0;
	Hist     h;
};

struct CpuAgg {
	uint64_t n = 0, imm = 0, unknown = 0, dur = 0, unsub = 0, pick = 0, proc = 0, gpu = 0, wake = 0, max = 0;
	Hist     h;
};

struct State {
	std::mutex                                 mutex;
	std::unordered_map<uint64_t, LabelRec>     labels;
	std::array<std::array<LabelAgg, 2>, 2>     label_agg {}; // [writer class][kind]
	std::array<std::deque<IrqPending>, MaxEvents> irq_fifo;
	std::array<IrqLast, MaxEvents>             irq_last {};
	std::array<IrqAgg, 2>                      irq_agg {};
	std::array<WaitState, MaxQueues>           waits {};
	std::array<std::array<WaitAgg, 4>, 2>      wait_agg {}; // [waiter class][producer class]
	std::array<CpuAgg, 2>                      cpu_agg {};  // [producer class]
	std::array<uint64_t, 2>                    idle_ns {};
	std::array<uint64_t, 2>                    idle_gw_ns {};
	uint64_t                                   lost_waits = 0;
};

State& St() {
	static State* state = new State; // never destroyed: hooks may run during shutdown
	return *state;
}

std::atomic<uint64_t>               g_frames {0};
std::array<std::atomic<uint64_t>, 2> g_idle_begin {};

uint32_t ProducerClass(const LabelRec* rec, uint32_t waiter) {
	if (rec == nullptr || rec->queue == NoQueue) {
		return 3;
	}
	if (rec->queue == waiter) {
		return 2;
	}
	return ClassOf(rec->queue);
}

bool TestValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: return false;
	}
}

int EventClass(uint64_t event_id) {
	if (event_id == 0) {
		return 0;
	}
	if (event_id >= 0x20 && event_id < MaxEvents) {
		return 1;
	}
	return -1;
}

} // namespace

bool EnabledSlow() noexcept {
	const char* value = std::getenv("KYTY_CP_LABEL_PROF");
	return value != nullptr && std::atoi(value) != 0;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void SliceBegin(uint32_t queue_id, uint64_t enqueue_ns, uint64_t start_ns) noexcept {
	t_queue   = queue_id < MaxQueues ? queue_id : NoQueue;
	t_enqueue = enqueue_ns;
	t_start   = start_ns;
}

void SliceEnd() noexcept {
	t_queue = NoQueue;
}

void Frame() noexcept {
	g_frames.fetch_add(1, std::memory_order_relaxed);
}

void Label(Kind kind, const void* address, uint32_t size, uint64_t value, Path path) noexcept {
	if (t_queue == NoQueue || address == nullptr) {
		return;
	}
	(void)size;
	const auto now  = NowNs();
	const auto addr = reinterpret_cast<uint64_t>(address);
	auto&      st   = St();
	std::lock_guard lock(st.mutex);
	auto& a = st.label_agg[ClassOf(t_queue)][static_cast<size_t>(kind)];
	a.n++;
	a.enq_start += Sat(t_start, t_enqueue);
	a.start_dec += Sat(now, t_start);
	a.enq_dec.Add(Sat(now, t_enqueue));
	auto& rec = st.labels[addr];
	if (path == Path::Deferred) {
		a.deferred++;
		rec.pending.push_back({t_queue, t_enqueue, t_start, now, value});
		return;
	}
	(path == Path::GpuOwned ? a.gpu_owned : a.cpu)++;
	rec.queue = t_queue;
	rec.enq   = t_enqueue;
	rec.start = t_start;
	rec.dec   = now;
	rec.vis   = now;
	rec.value = value;
}

void LabelDelivered(uint64_t address) noexcept {
	const auto now = NowNs();
	auto&      st  = St();
	std::lock_guard lock(st.mutex);
	auto it = st.labels.find(address);
	if (it == st.labels.end() || it->second.pending.empty()) {
		return;
	}
	auto& rec = it->second;
	const auto w = rec.pending.front();
	rec.pending.pop_front();
	rec.queue = w.queue;
	rec.enq   = w.enq;
	rec.start = w.start;
	rec.dec   = w.dec;
	rec.vis   = now;
	rec.value = w.value;
	auto& a   = st.label_agg[ClassOf(w.queue)][static_cast<size_t>(Kind::ReleaseMem)];
	a.dec_vis += Sat(now, w.dec);
	a.delivered++;
}

void InterruptPacket(int event_id) noexcept {
	if (t_queue == NoQueue || event_id < 0 || static_cast<uint32_t>(event_id) >= MaxEvents) {
		return;
	}
	const auto now = NowNs();
	auto&      st  = St();
	std::lock_guard lock(st.mutex);
	st.irq_fifo[static_cast<size_t>(event_id)].push_back({t_enqueue, t_start, now});
}

void InterruptDelivered(int event_id) noexcept {
	if (event_id < 0 || static_cast<uint32_t>(event_id) >= MaxEvents) {
		return;
	}
	const auto now = NowNs();
	auto&      st  = St();
	std::lock_guard lock(st.mutex);
	auto& fifo = st.irq_fifo[static_cast<size_t>(event_id)];
	if (fifo.empty()) {
		return;
	}
	const auto p = fifo.front();
	fifo.pop_front();
	const int cls = EventClass(static_cast<uint64_t>(event_id));
	if (cls >= 0) {
		auto& a = st.irq_agg[static_cast<size_t>(cls)];
		a.n++;
		a.enq_start += Sat(p.start, p.enq);
		a.start_dec += Sat(p.dec, p.start);
		a.dec_trig += Sat(now, p.dec);
		a.max_dec_trig = std::max(a.max_dec_trig, Sat(now, p.dec));
		a.enq_trig.Add(Sat(now, p.enq));
	}
	st.irq_last[static_cast<size_t>(event_id)] = {p.enq, p.start, p.dec, now};
}

void WaitBlocked(const void* address) noexcept {
	const auto q = t_queue;
	if (q == NoQueue) {
		return;
	}
	const auto addr = reinterpret_cast<uint64_t>(address);
	const auto now  = NowNs();
	auto&      st   = St();
	std::lock_guard lock(st.mutex);
	auto& ws = st.waits[q];
	if (ws.active && ws.addr == addr) {
		ws.polls++;
		return;
	}
	if (ws.active) {
		st.lost_waits++;
	}
	const auto it = st.labels.find(addr);
	ws            = {true, addr, now, 1, ProducerClass(it == st.labels.end() ? nullptr : &it->second, q)};
}

void WaitPassed(const void* address, uint64_t ref, uint64_t mask, uint32_t func) noexcept {
	const auto q = t_queue;
	if (q == NoQueue) {
		return;
	}
	const auto addr = reinterpret_cast<uint64_t>(address);
	const auto now  = NowNs();
	auto&      st   = St();
	std::lock_guard lock(st.mutex);
	auto&           ws  = st.waits[q];
	const auto      it  = st.labels.find(addr);
	const LabelRec* rec = it == st.labels.end() ? nullptr : &it->second;
	const auto      wc  = ClassOf(q);
	if (!ws.active || ws.addr != addr) {
		st.wait_agg[wc][ProducerClass(rec, q)].imm++;
		return;
	}
	ws.active      = false;
	const auto s   = ws.start;
	const auto dur = Sat(now, s);
	const bool attributed =
	    rec != nullptr && rec->queue != NoQueue && rec->vis >= s && TestValue(rec->value, ref, mask, func);
	auto& a = st.wait_agg[wc][attributed ? ProducerClass(rec, q) : 3u];
	a.n++;
	a.dur += dur;
	a.max = std::max(a.max, dur);
	a.polls += ws.polls;
	a.h.Add(dur);
	if (attributed) {
		a.unsub += Sat(rec->enq, s);
		a.pick += Sat(rec->start, std::max(s, rec->enq));
		a.proc += Sat(rec->dec, std::max(s, rec->start));
		a.gpu += Sat(rec->vis, std::max(s, rec->dec));
		a.recheck += Sat(now, std::max(s, rec->vis));
	}
}

void CpuWaitDone(uint64_t event_id, uint64_t start_ns, bool blocked) noexcept {
	const int cls = EventClass(event_id);
	if (cls < 0) {
		return;
	}
	const auto now = NowNs();
	auto&      st  = St();
	std::lock_guard lock(st.mutex);
	auto& a = st.cpu_agg[static_cast<size_t>(cls)];
	if (!blocked) {
		a.imm++;
		return;
	}
	const auto dur = Sat(now, start_ns);
	a.n++;
	a.dur += dur;
	a.max = std::max(a.max, dur);
	a.h.Add(dur);
	const auto& last = st.irq_last[static_cast<size_t>(event_id)];
	if (last.trig < start_ns) {
		a.unknown++;
		return;
	}
	a.unsub += Sat(last.enq, start_ns);
	a.pick += Sat(last.start, std::max(start_ns, last.enq));
	a.proc += Sat(last.dec, std::max(start_ns, last.start));
	a.gpu += Sat(last.trig, std::max(start_ns, last.dec));
	a.wake += Sat(now, std::max(start_ns, last.trig));
}

void IdleBegin(uint32_t lane) noexcept {
	if (lane < 2) {
		g_idle_begin[lane].store(NowNs(), std::memory_order_relaxed);
	}
}

void IdleEnd(uint32_t lane) noexcept {
	if (lane >= 2) {
		return;
	}
	const auto now = NowNs();
	const auto d   = Sat(now, g_idle_begin[lane].load(std::memory_order_relaxed));
	auto&      st  = St();
	std::lock_guard lock(st.mutex);
	st.idle_ns[lane] += d;
	const auto& gw = st.waits[0];
	if (gw.active && gw.pred == 1) {
		st.idle_gw_ns[lane] += std::min(d, Sat(now, gw.start));
	}
}

void Report(double seconds) {
	if (!Enabled()) {
		return;
	}
	auto& st = St();
	std::array<std::array<LabelAgg, 2>, 2> label_agg {};
	std::array<IrqAgg, 2>                  irq_agg {};
	std::array<std::array<WaitAgg, 4>, 2>  wait_agg {};
	std::array<CpuAgg, 2>                  cpu_agg {};
	std::array<uint64_t, 2>                idle_ns {};
	std::array<uint64_t, 2>                idle_gw_ns {};
	uint64_t                               lost_waits = 0;
	size_t                                 map_size   = 0;
	{
		std::lock_guard lock(st.mutex);
		std::swap(label_agg, st.label_agg);
		std::swap(irq_agg, st.irq_agg);
		std::swap(wait_agg, st.wait_agg);
		std::swap(cpu_agg, st.cpu_agg);
		std::swap(idle_ns, st.idle_ns);
		std::swap(idle_gw_ns, st.idle_gw_ns);
		std::swap(lost_waits, st.lost_waits);
		// Labels not written for 30 s are forgotten (data writes to many addresses would grow the table).
		const auto horizon = Sat(NowNs(), 30'000'000'000ull);
		for (auto it = st.labels.begin(); it != st.labels.end();) {
			if (it->second.pending.empty() && it->second.vis < horizon) {
				it = st.labels.erase(it);
			} else {
				++it;
			}
		}
		map_size = st.labels.size();
	}
	const auto ms     = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	const char* cls[] = {"gfx", "cmp"};
	const char* kinds[] = {"rm", "wd"};
	const char* prod[]  = {"gfx", "cmp", "self", "unk"};
	std::printf("PERF cplabel %.1fs: frames=%" PRIu64 " idle0=%.1f idle1=%.1f idle0_gw=%.1f idle1_gw=%.1f lost=%" PRIu64
	            " map=%zu\n",
	            seconds, g_frames.exchange(0, std::memory_order_relaxed), ms(idle_ns[0]), ms(idle_ns[1]),
	            ms(idle_gw_ns[0]), ms(idle_gw_ns[1]), lost_waits, map_size);
	for (int c = 0; c < 2; c++) {
		for (int k = 0; k < 2; k++) {
			const auto& a = label_agg[static_cast<size_t>(c)][static_cast<size_t>(k)];
			if (a.n == 0) {
				continue;
			}
			std::printf("PERF cplabel lbl c=%s k=%s n=%" PRIu64 " cpu=%" PRIu64 " gpuown=%" PRIu64 " def=%" PRIu64
			            " enq_start=%.1f start_dec=%.1f dec_vis=%.1f delivered=%" PRIu64 " h=%s\n",
			            cls[c], kinds[k], a.n, a.cpu, a.gpu_owned, a.deferred, ms(a.enq_start), ms(a.start_dec),
			            ms(a.dec_vis), a.delivered, a.enq_dec.Str().c_str());
		}
	}
	for (int c = 0; c < 2; c++) {
		const auto& a = irq_agg[static_cast<size_t>(c)];
		if (a.n == 0) {
			continue;
		}
		std::printf("PERF cplabel irq c=%s n=%" PRIu64 " enq_start=%.1f start_dec=%.1f dec_trig=%.1f max_dec_trig=%.2f"
		            " h=%s\n",
		            cls[c], a.n, ms(a.enq_start), ms(a.start_dec), ms(a.dec_trig), ms(a.max_dec_trig),
		            a.enq_trig.Str().c_str());
	}
	for (int w = 0; w < 2; w++) {
		for (int p = 0; p < 4; p++) {
			const auto& a = wait_agg[static_cast<size_t>(w)][static_cast<size_t>(p)];
			if (a.n == 0 && a.imm == 0) {
				continue;
			}
			std::printf("PERF cplabel wait w=%s p=%s n=%" PRIu64 " imm=%" PRIu64 " dur=%.1f unsub=%.1f pick=%.1f"
			            " proc=%.1f gpu=%.1f recheck=%.1f max=%.2f polls=%" PRIu64 " h=%s\n",
			            cls[w], prod[p], a.n, a.imm, ms(a.dur), ms(a.unsub), ms(a.pick), ms(a.proc), ms(a.gpu),
			            ms(a.recheck), ms(a.max), a.polls, a.h.Str().c_str());
		}
	}
	for (int c = 0; c < 2; c++) {
		const auto& a = cpu_agg[static_cast<size_t>(c)];
		if (a.n == 0 && a.imm == 0) {
			continue;
		}
		std::printf("PERF cplabel cpu p=%s n=%" PRIu64 " imm=%" PRIu64 " unk=%" PRIu64 " dur=%.1f unsub=%.1f"
		            " pick=%.1f proc=%.1f gpu=%.1f wake=%.1f max=%.2f h=%s\n",
		            cls[c], a.n, a.imm, a.unknown, ms(a.dur), ms(a.unsub), ms(a.pick), ms(a.proc), ms(a.gpu),
		            ms(a.wake), ms(a.max), a.h.Str().c_str());
	}
}

} // namespace Libs::Graphics::CpLabelProf
