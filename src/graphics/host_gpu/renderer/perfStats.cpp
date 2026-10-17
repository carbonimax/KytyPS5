#include "graphics/host_gpu/renderer/perfStats.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::PerfStats {

namespace {

constexpr auto ReportInterval = std::chrono::seconds(5);
constexpr auto CounterCount   = static_cast<size_t>(Counter::Count);

std::array<std::atomic<uint64_t>, CounterCount> g_counters {};
std::atomic<uint64_t>                           g_programs {0};
std::atomic<uint64_t>                           g_permutations {0};
thread_local bool                               g_gpu_thread = false;

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

} // namespace

void Add(Counter counter, uint64_t value) noexcept {
	if (counter >= Counter::Count) {
		return;
	}
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void SetProgramTotals(uint64_t programs, uint64_t permutations) noexcept {
	g_programs.store(programs, std::memory_order_relaxed);
	g_permutations.store(permutations, std::memory_order_relaxed);
}

void MarkGpuThread() noexcept {
	g_gpu_thread = true;
}

bool IsGpuThread() noexcept {
	return g_gpu_thread;
}

void MaybeReport() noexcept {
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < ReportInterval) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - last).count();
	last                 = now;
	std::array<uint64_t, CounterCount> v {};
	for (size_t i = 0; i < CounterCount; i++) {
		v[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	if (!Enabled()) {
		return;
	}
	const auto ms = [&](Counter c) { return static_cast<double>(v[static_cast<size_t>(c)]) / 1e6; };
	const auto n  = [&](Counter c) { return v[static_cast<size_t>(c)]; };
	const double wall_ms = seconds * 1000.0;
	std::printf("PERF stats %.1fs: gpu_busy=%.0f%% recompile=%.0fms(%" PRIu64
	            ") module=%.0fms gfx_pipe=%.0fms(%" PRIu64 ") cs_pipe=%.0fms(%" PRIu64
	            ") wait_gpu_thread=%.0fms(%" PRIu64 ") wait_other=%.0fms(%" PRIu64
	            ") readback_sync=%.0fms(%" PRIu64 ") submits=%.0f/s programs=%" PRIu64
	            " permutations=%" PRIu64 "\n",
	            seconds, 100.0 * ms(Counter::GpuBusyNs) / wall_ms, ms(Counter::RecompileNs),
	            n(Counter::RecompileCount), ms(Counter::ShaderModuleNs), ms(Counter::GfxPipelineNs),
	            n(Counter::GfxPipelineCount), ms(Counter::CsPipelineNs), n(Counter::CsPipelineCount),
	            ms(Counter::WaitGpuThreadNs), n(Counter::WaitGpuThreadCount),
	            ms(Counter::WaitOtherNs), n(Counter::WaitOtherCount), ms(Counter::ReadbackSyncNs),
	            n(Counter::ReadbackSyncCount), static_cast<double>(n(Counter::Submits)) / seconds,
	            g_programs.load(std::memory_order_relaxed),
	            g_permutations.load(std::memory_order_relaxed));
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PerfStats
