#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_

#include <atomic>
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
	IdleSubmits,       // of which KYTY_IDLE_SUBMIT submissions
	SideReadNs,        // KYTY_SYNC_SIDE_READS: GPU thread waiting for readback-queue copies
	SideReadCount,
	// KYTY_BIND_PROF=1 (with KYTY_PERF_STATS=1): resource binding of draws and dispatches.
	BindPrepareNs,     // PrepareBindings: texture resolution, samplers, shader data
	BindFindBuffersNs, // FindBuffers: buffer cache lookups
	BindHeapNs,        // ImageHeap::Rebind
	BindImagesNs,      // RebindImages: image views
	BindBuffersNs,     // RebindBuffers: storage buffer views, uploads
	BindCommitNs,      // CommitBindings, whole
	BindCheckNs,       // of which scalar-read overlap checks + image heap commit
	BindTransitNs,     // of which image layout transitions
	BindPushNs,        // of which push descriptors / set update + bind (record hook or driver)
	BindCommitCount,   // CommitBindings calls
	BindWriteCount,    // vk::WriteDescriptorSet entries
	BindDescriptorCount, // descriptors in those writes
	// KYTY_SRT_PROF=1 (with KYTY_PERF_STATS=1): program lookups that found a cached source (GPU thread).
	SrtLookups,        // lookups timed
	SrtRefreshNs,      // call-source refresh (GPU-selected call targets)
	SrtWalkNs,         // WalkSrt: x86 walker or walk reuse
	SrtEvalNs,         // descriptor evaluation, indirect tables, sampler checks
	SrtSpecNs,         // BuildResourceSpecialization
	SrtMatchNs,        // permutation search and the rest of the lookup
	SrtSamePerm,       // chose the entry's previous permutation
	SrtMultiPerm,      // entry has more than one permutation
	SrtMiss,           // no permutation matched (translation follows)
	SrtUserSame,       // user data equal to the entry's previous lookup
	SrtDescSame,       // buffer/image/sampler descriptors equal to the previous lookup
	SrtFlatSame,       // flattened SRT equal to the previous lookup
	SrtNoLoad,         // walk loaded no guest memory (descriptors from user data only)
	SrtLoads,          // guest loads of the walks (memo count)
	SrtFlatWords,      // flattened SRT dwords (scalar values the shader reads)
	SrtFlatNeeded,     // of the plan's flattened SRT reads, those descriptors/selectors/conditions depend on
	SrtFlatPlan,       // the plan's flattened SRT reads
	SrtBuffers,        // buffer resources
	SrtImages,         // image resources, of which
	SrtHeapImages,     //   resident image heap roots/kinds
	SrtConvImages,     //   images needing a format conversion in the shader
	SrtSamplers,       // sampler resources
	SrtNoResources,    // lookups without buffers, images or samplers
	SrtHeapOnly,       // lookups whose resources are all heap images
	SrtWithBuffers,    // lookups with at least one buffer
	SrtWithImages,     // lookups with at least one non-heap image
	SrtWithSamplers,   // lookups with at least one sampler
	SrtFormatSpec,     // lookups whose specialization depends on a descriptor format/stride/type field
	BindFindImageNs,   // of prepare/images: TextureCache::FindImage (resolution + tracking)
	BindSamplerNs,     // of prepare: sampler cache
	BindObtainNs,      // of buffers: NativeStorageBuffer (buffer cache, uploads)
	BindUploadNs,      // of buffers: flattened SRT + shader data stream uploads
	BindViewNs,        // of images: TextureCache::FindTexture (views)
	Count
};

[[nodiscard]] bool Enabled() noexcept;
// KYTY_BIND_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): times the binding phases.
[[nodiscard]] bool BindProfEnabled() noexcept;
// KYTY_SRT_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): splits program lookups (walk,
// descriptor evaluation, specialization, permutation search), counts what changes between lookups and
// which resources they bind, and splits texture/sampler/buffer resolution ("PERF srtsplit" line).
[[nodiscard]] bool SrtProfEnabled() noexcept;
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

// KYTY_COMPILE_LOG=<file> (diagnostic, default off): one line per shader registration, program
// translation miss and pipeline creation, each starting with a kind letter and its steady-clock time
// in microseconds (same clock and "# start" header as KYTY_FRAME_TIMES). Safe on any host thread.
[[nodiscard]] bool     CompileLogEnabled() noexcept;
[[nodiscard]] uint64_t SteadyUs() noexcept;
void                   CompileLogLine(const char* format, ...) noexcept;

// Draws and dispatches recorded (GPU thread, and the KYTY_CP_PIPELINE back thread), always counted:
// KYTY_IDLE_SUBMIT reads it (CommandScheduler::IdleSubmitDue).
inline std::atomic<uint64_t> t_work_recorded {0};
inline void                  NoteWork() noexcept {
    t_work_recorded.fetch_add(1, std::memory_order_relaxed);
}

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

	// Timer that only runs when `on` (e.g. BindProfEnabled()).
	ScopedTimer(Counter time, bool on) noexcept: m_time(time), m_count(Counter::Count) {
		if (on) {
			m_start = std::chrono::steady_clock::now();
			m_on    = true;
		}
	}

private:
	Counter                               m_time;
	Counter                               m_count;
	std::chrono::steady_clock::time_point m_start {};
	bool                                  m_on = false;
};

} // namespace Libs::Graphics::PerfStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
