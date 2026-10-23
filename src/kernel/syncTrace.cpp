#include "kernel/syncTrace.h"

#include "common/threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::LibKernel::SyncTrace {

namespace {

constexpr int      MaxSlots     = 1024;
constexpr uint64_t ReportUs     = 2000000;
constexpr uint64_t BlockedUs    = 1000000;
constexpr const char* KindNames[] = {"sema", "psem", "evf", "equeue", "cond", "mutex", "addr"};

using Clock = std::chrono::steady_clock;

struct Slot {
	std::atomic<int> active {0};
	int              tid     = 0;
	Kind             kind    = Kind::Sema;
	const void*      object  = nullptr;
	char             name[48] {};
	uint64_t         arg     = 0;
	int64_t          timeout = -1;
	uint64_t         start   = 0;
};

struct Stat {
	Kind        kind     = Kind::Sema;
	std::string name;
	uint64_t    blocks   = 0;
	uint64_t    timeouts = 0;
	uint64_t    wakes    = 0;
	uint64_t    block_us = 0;
};

struct State {
	FILE*                                    file = nullptr;
	Clock::time_point                        t0;
	uint64_t                                 min_us = 2000;
	std::mutex                               mutex;
	std::unordered_map<const void*, Stat>    stats;
	Slot                                     slots[MaxSlots];
	std::atomic<int>                         next_slot {0};
};

State* Init() {
	const char* path = std::getenv("KYTY_SYNC_TRACE");
	if (path == nullptr || *path == '\0' || std::strcmp(path, "0") == 0) {
		return nullptr;
	}
	auto* state = new State;
	state->file = std::fopen(path, "w");
	if (state->file == nullptr) {
		delete state;
		return nullptr;
	}
	std::setvbuf(state->file, nullptr, _IOFBF, 1 << 20);
	state->t0 = Clock::now();
	if (const char* min = std::getenv("KYTY_SYNC_TRACE_MIN_US")) {
		state->min_us = std::strtoull(min, nullptr, 10);
	}
	const auto now = std::time(nullptr);
	char       wall[32];
	std::strftime(wall, sizeof(wall), "%H:%M:%S", std::localtime(&now));
	std::fprintf(state->file, "START wall=%s min_us=%llu\n", wall,
	             static_cast<unsigned long long>(state->min_us));
	std::thread([state] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::microseconds(ReportUs));
			const auto now_us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - state->t0)
			        .count());
			std::lock_guard lock(state->mutex);
			std::fprintf(state->file, "%10.3f REPORT\n", static_cast<double>(now_us) / 1e6);
			const int count = std::min(state->next_slot.load(), MaxSlots);
			for (int i = 0; i < count; i++) {
				auto& slot = state->slots[i];
				if (slot.active.load(std::memory_order_acquire) == 0) continue;
				const auto start = slot.start;
				if (now_us < start || now_us - start < BlockedUs) continue;
				std::fprintf(state->file,
				             "%10.3f   BLOCKED tid=%d %s obj=%p name=%s arg=0x%llx timeout=%lld "
				             "for=%.1fs\n",
				             static_cast<double>(now_us) / 1e6, slot.tid,
				             KindNames[static_cast<int>(slot.kind)], slot.object, slot.name,
				             static_cast<unsigned long long>(slot.arg),
				             static_cast<long long>(slot.timeout),
				             static_cast<double>(now_us - start) / 1e6);
			}
			std::vector<std::pair<const void*, Stat>> busy(state->stats.begin(),
			                                               state->stats.end());
			std::sort(busy.begin(), busy.end(), [](const auto& a, const auto& b) {
				return a.second.blocks + a.second.timeouts + a.second.wakes >
				       b.second.blocks + b.second.timeouts + b.second.wakes;
			});
			for (size_t i = 0; i < busy.size() && i < 12; i++) {
				const auto& s = busy[i].second;
				std::fprintf(state->file,
				             "%10.3f   STAT %s obj=%p name=%s blocks=%llu timeouts=%llu wakes=%llu "
				             "block_ms=%.1f\n",
				             static_cast<double>(now_us) / 1e6, KindNames[static_cast<int>(s.kind)],
				             busy[i].first, s.name.c_str(), static_cast<unsigned long long>(s.blocks),
				             static_cast<unsigned long long>(s.timeouts),
				             static_cast<unsigned long long>(s.wakes),
				             static_cast<double>(s.block_us) / 1e3);
			}
			state->stats.clear();
			std::fflush(state->file);
		}
	}).detach();
	return state;
}

State* Get() {
	static State* state = Init();
	return state;
}

uint64_t NowUs(const State* state) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - state->t0).count());
}

int ThreadSlot(State* state) {
	thread_local int slot = -1;
	if (slot < 0) {
		slot = state->next_slot.fetch_add(1);
		if (slot >= MaxSlots) slot = MaxSlots;
	}
	return slot < MaxSlots ? slot : -1;
}

} // namespace

bool Enabled() {
	return Get() != nullptr;
}

Wait::Wait(Kind kind, const void* object, const char* name, uint64_t arg, int64_t timeout_us) {
	auto* state = Get();
	if (state == nullptr) return;
	m_slot  = ThreadSlot(state);
	m_start = NowUs(state);
	if (m_slot < 0) return;
	auto& slot  = state->slots[m_slot];
	slot.tid    = Common::Thread::GetThreadIdUnique();
	slot.kind   = kind;
	slot.object = object;
	std::snprintf(slot.name, sizeof(slot.name), "%s", name != nullptr ? name : "");
	slot.arg     = arg;
	slot.timeout = timeout_us;
	slot.start   = m_start;
	slot.active.store(1, std::memory_order_release);
}

Wait::~Wait() {
	auto* state = Get();
	if (state == nullptr || m_slot < 0) return;
	auto& slot = state->slots[m_slot];
	slot.active.store(0, std::memory_order_release);
	const auto now = NowUs(state);
	const auto dur = now - m_start;
	std::lock_guard lock(state->mutex);
	auto& stat = state->stats[slot.object];
	stat.kind  = slot.kind;
	if (stat.name.empty()) stat.name = slot.name;
	stat.blocks++;
	stat.block_us += dur;
	const bool timed_out = m_result != 0;
	if (timed_out) stat.timeouts++;
	if (dur >= state->min_us) {
		std::fprintf(state->file, "%10.3f WAIT tid=%d %s obj=%p name=%s arg=0x%llx timeout=%lld "
		                          "dur_ms=%.2f result=%d\n",
		             static_cast<double>(now) / 1e6, slot.tid,
		             KindNames[static_cast<int>(slot.kind)], slot.object, slot.name,
		             static_cast<unsigned long long>(slot.arg),
		             static_cast<long long>(slot.timeout), static_cast<double>(dur) / 1e3,
		             m_result);
	}
}

void Wake(Kind kind, const void* object, const char* name, uint64_t arg, int waiters) {
	auto* state = Get();
	if (state == nullptr) return;
	const auto now = NowUs(state);
	const int  tid = Common::Thread::GetThreadIdUnique();
	std::lock_guard lock(state->mutex);
	auto& stat = state->stats[object];
	stat.kind  = kind;
	if (stat.name.empty() && name != nullptr) stat.name = name;
	stat.wakes++;
	if (waiters > 0) {
		std::fprintf(state->file, "%10.3f WAKE tid=%d %s obj=%p name=%s arg=0x%llx waiters=%d\n",
		             static_cast<double>(now) / 1e6, tid, KindNames[static_cast<int>(kind)],
		             object, name != nullptr ? name : "", static_cast<unsigned long long>(arg),
		             waiters);
	}
}

void Note(const char* format, ...) {
	auto* state = Get();
	if (state == nullptr) return;
	char    buf[600];
	va_list args;
	va_start(args, format);
	std::vsnprintf(buf, sizeof(buf), format, args);
	va_end(args);
	const auto now = NowUs(state);
	const int  tid = Common::Thread::GetThreadIdUnique();
	std::lock_guard lock(state->mutex);
	std::fprintf(state->file, "%10.3f NOTE tid=%d %s\n", static_cast<double>(now) / 1e6, tid, buf);
}

} // namespace Libs::LibKernel::SyncTrace
