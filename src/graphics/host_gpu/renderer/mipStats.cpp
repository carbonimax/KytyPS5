#include "graphics/host_gpu/renderer/mipStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics::MipStats {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t HeaderSize   = 64;
constexpr uint64_t SampledMip0  = 0;
constexpr uint64_t NotSampled   = uint64_t {0xf} << 56u;
constexpr uint64_t InUseReports = 300;
constexpr auto     ReliefTime   = std::chrono::seconds(15);
constexpr auto     Cooldown     = std::chrono::seconds(20);

// Counters sampled since the last report, one bit each (marked by the draw thread).
std::array<std::atomic<uint64_t>, 4> g_counters {};
// The last NoteStreamingThrash, in steady_clock ticks; 0: none yet.
std::atomic<int64_t> g_thrash_time {0};

// Report state (one reporting thread: the command processor).
struct ReportState {
	uint64_t                  reports         = 0;
	std::array<uint64_t, 256> last_marked {};
	int64_t                   relief_end  = 0;
	int64_t                   rearm_time  = 0;
	std::mutex                mutex;
};
ReportState g_report;

bool DebugEnabled() {
	static const bool enabled = std::getenv("KYTY_DEBUG_MIP_STATS") != nullptr;
	return enabled;
}

int64_t NowTicks() {
	return Clock::now().time_since_epoch().count();
}

} // namespace

bool ReliefEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MIP_STATS_RELIEF");
		return value == nullptr || *value == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void Mark(uint32_t id) noexcept {
	g_counters[(id / 64u) & 3u].fetch_or(uint64_t {1} << (id % 64u), std::memory_order_relaxed);
}

void Report(void* dst, uint32_t size, bool reset) {
	if (dst == nullptr || size == 0) {
		return;
	}
	std::memset(dst, 0, size);
	if (size < sizeof(uint32_t)) {
		return;
	}
	*static_cast<uint32_t*>(dst) = 1;
	if (!ReliefEnabled()) {
		return;
	}

	std::lock_guard lock(g_report.mutex);
	auto&           s = g_report;
	s.reports++;
	for (uint32_t word = 0; word < g_counters.size(); word++) {
		const auto bits = reset ? g_counters[word].exchange(0, std::memory_order_relaxed)
		                        : g_counters[word].load(std::memory_order_relaxed);
		for (uint32_t bit = 0; bit < 64; bit++) {
			if (((bits >> bit) & 1u) != 0) {
				s.last_marked[word * 64u + bit] = s.reports;
			}
		}
	}
	const auto in_use = [&](uint64_t last) {
		return last != 0 && s.reports - last <= InUseReports;
	};
	const auto now         = NowTicks();
	const auto thrash_time = g_thrash_time.load(std::memory_order_relaxed);
	if (s.relief_end == 0 && thrash_time != 0 && thrash_time > s.rearm_time) {
		s.relief_end = now + std::chrono::duration_cast<Clock::duration>(ReliefTime).count();
		if (DebugEnabled()) {
			std::printf("mip-stats: relief starts at report %llu\n",
			            static_cast<unsigned long long>(s.reports));
		}
	}
	if (s.relief_end != 0 && now >= s.relief_end) {
		s.relief_end = 0;
		s.rearm_time = now + std::chrono::duration_cast<Clock::duration>(Cooldown).count();
	}
	const bool relief = s.relief_end != 0;
	// The original relief waited for draws to mark some counter before reporting any as unused. A
	// loading screen can end on a plain fade that samples no tracked texture while the streamer still
	// thrashes: then every counter is unused, which a relief may report for its 15 s.
	if (!relief) {
		return;
	}
	const auto counters = std::min<uint32_t>(size > HeaderSize ? (size - HeaderSize) / 8u : 0u,
	                                         static_cast<uint32_t>(s.last_marked.size()));
	auto*    values = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(dst) + HeaderSize);
	uint32_t unused = 0;
	for (uint32_t id = 0; id < counters; id++) {
		const bool sampled = in_use(s.last_marked[id]);
		values[id]         = sampled ? SampledMip0 : NotSampled;
		unused += sampled ? 0u : 1u;
	}
	if (DebugEnabled() && s.reports % 30 == 0) {
		std::printf("mip-stats: report %llu relief unused=%u of %u\n",
		            static_cast<unsigned long long>(s.reports), unused, counters);
		std::fflush(stdout);
	}
}

void NoteRepeatedFileRead(uint64_t file_key, uint64_t offset, uint64_t size) {
	constexpr uint64_t MinSize = 1024 * 1024;
	constexpr uint32_t Repeats = 4;
	constexpr auto     Window  = std::chrono::seconds(10);
	if (size < MinSize || !ReliefEnabled()) {
		return;
	}
	struct Read {
		uint64_t          key   = 0;
		uint32_t          count = 0;
		Clock::time_point first;
	};
	static std::mutex            mutex;
	static std::array<Read, 256> reads;
	const uint64_t key = (file_key * 0x9E3779B97F4A7C15ull) ^
	                     (offset * 0xC2B2AE3D27D4EB4Full) ^ size;
	const auto     now  = Clock::now();
	bool           note = false;
	{
		std::scoped_lock lock(mutex);
		auto&            read = reads[(key ^ (key >> 29u)) % reads.size()];
		if (read.key != key || now - read.first > Window) {
			read = {key, 1, now};
		} else if (++read.count >= Repeats) {
			read = {key, 0, now};
			note = true;
		}
	}
	if (note) {
		g_thrash_time.store(NowTicks(), std::memory_order_relaxed);
		if (DebugEnabled()) {
			std::printf("mip-stats: repeated read of %llu bytes at %llu: streaming thrash\n",
			            static_cast<unsigned long long>(size), static_cast<unsigned long long>(offset));
			std::fflush(stdout);
		}
	}
}

} // namespace Libs::Graphics::MipStats
