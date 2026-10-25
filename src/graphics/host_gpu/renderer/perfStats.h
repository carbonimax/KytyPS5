#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_

#include <chrono>
#include <cstdint>

// Measurement counters of the command processor (GPU thread), shader translation, pipeline creation
// and host timeline waits, reported as one "PERF stats" console line every 5 seconds.
//
// KYTY_PERF_STATS=1 enables the report (default 0: counters are not even timed).
// KYTY_FRAME_TIMES=<file> writes one line per presented frame (independent of KYTY_PERF_STATS).
//
// Portable (std::chrono and relaxed atomics only); safe on any host thread.
namespace Libs::Graphics::PerfStats {

enum class Counter : uint32_t {
	Frames,            // guest frames (suspend points) processed by the GPU thread
	GpuBusyNs,         // GPU thread processing guest submissions and commands
	DrawNs,            // draw packets (preparation + recording)
	DrawCount,
	DispatchNs,        // dispatch packets (preparation + recording)
	DispatchCount,
	ProgramNs,         // program lookup per stage (resource materialization + permutation search)
	ProgramCount,
	SrtNs,             // resource materialization alone (SRT walks)
	RecompileNs,       // guest shader translation + SPIR-V emission (permutation miss)
	RecompileCount,
	GfxPipelineNs,     // vkCreateGraphicsPipelines
	GfxPipelineCount,
	GfxPipelineHits,   // driver pipeline-cache hits (pipeline creation feedback)
	GfxPipelineHitNs,
	GfxPipelineMisses,
	GfxPipelineMissNs,
	CsPipelineNs,      // vkCreateComputePipelines
	CsPipelineCount,
	WaitGpuThreadNs,   // GPU thread blocked on the host GPU timeline
	WaitGpuThreadCount,
	WaitOtherNs,       // other threads blocked on the host GPU timeline
	WaitOtherCount,
	Submits,           // vkQueueSubmit calls
	Count
};

[[nodiscard]] bool Enabled() noexcept;
void               Add(Counter counter, uint64_t value) noexcept;
// GPU thread: marks the calling thread (wait attribution) and prints the report when due.
void               MarkGpuThread() noexcept;
[[nodiscard]] bool IsGpuThread() noexcept;
void               MaybeReport() noexcept;
// A pipeline creation reported by VK_EXT_pipeline_creation_feedback (core in Vulkan 1.3).
void NoteGraphicsPipelineFeedback(bool driver_cache_hit, uint64_t duration_ns) noexcept;

// KYTY_FRAME_TIMES=<file> (measurement, default off): one line per presented frame with its
// steady-clock time in microseconds and the flip argument of the latest guest flip completed before
// it ("us arg"), after a "# start HH:MM:SS steady_us=N" header that ties the clock to wall time.
// Frame-time percentiles and stalls are computed offline; a game whose flip argument counts its
// simulation ticks gives windows in game progress instead of wall time.
void NotePresentedFrame() noexcept;
void NoteFlipArg(int64_t arg) noexcept;
// Smoothed interval between presented frames in microseconds (0 before the second frame).
[[nodiscard]] uint64_t RecentFrameUs() noexcept;

class ScopedTimer {
public:
	ScopedTimer(Counter time, Counter count) noexcept: m_time(time), m_count(count) {
		if (Enabled()) {
			m_start = std::chrono::steady_clock::now();
			m_on    = true;
		}
	}
	explicit ScopedTimer(Counter time) noexcept: ScopedTimer(time, Counter::Count) {}
	~ScopedTimer() {
		if (!m_on) {
			return;
		}
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
	std::chrono::steady_clock::time_point m_start {};
	bool                                  m_on = false;
};

} // namespace Libs::Graphics::PerfStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
