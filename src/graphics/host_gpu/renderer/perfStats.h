#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_

#include <chrono>
#include <cstdint>

// Lightweight counters reported to the console every few seconds ("PERF stats").
// Off by default; set KYTY_PERF_STATS=1 to enable the report.
namespace Libs::Graphics::PerfStats {

enum class Counter : uint32_t {
	GpuBusyNs,         // GPU thread processing guest submissions and commands
	RecompileNs,       // guest shader translation + SPIR-V emission (permutation miss)
	RecompileCount,
	ShaderModuleNs,    // vkCreateShaderModule
	GfxPipelineNs,     // vkCreateGraphicsPipelines (+ layouts)
	GfxPipelineCount,
	CsPipelineNs,      // vkCreateComputePipelines (+ layouts)
	CsPipelineCount,
	WaitGpuThreadNs,   // GPU-thread CPU blocked on the host GPU timeline
	WaitGpuThreadCount,
	WaitOtherNs,       // other threads (priority ops, guest CPU) blocked on the timeline
	WaitOtherCount,
	ReadbackSyncNs,    // guest CPU thread blocked on a synchronous buffer readback
	ReadbackSyncCount,
	Submits,           // vkQueueSubmit calls
	Count
};

void Add(Counter counter, uint64_t value) noexcept;
void SetProgramTotals(uint64_t programs, uint64_t permutations) noexcept;
// Called from the GPU thread; prints a report when the interval has elapsed.
void MaybeReport() noexcept;
void MarkGpuThread() noexcept;
[[nodiscard]] bool IsGpuThread() noexcept;

class ScopedTimer {
public:
	ScopedTimer(Counter time, Counter count) noexcept
	    : m_time(time), m_count(count), m_start(std::chrono::steady_clock::now()) {}
	explicit ScopedTimer(Counter time) noexcept: ScopedTimer(time, Counter::Count) {}
	~ScopedTimer() {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                    std::chrono::steady_clock::now() - m_start)
		                    .count();
		Add(m_time, static_cast<uint64_t>(ns));
		if (m_count != Counter::Count) {
			Add(m_count, 1);
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_time;
	Counter                               m_count;
	std::chrono::steady_clock::time_point m_start;
};

} // namespace Libs::Graphics::PerfStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
