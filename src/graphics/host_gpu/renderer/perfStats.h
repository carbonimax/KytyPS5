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
	BindCheckNs,       // of which stage checks + image heap commit
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
	// KYTY_ASYNC_PIPELINE=1: draws skipped while their pipeline is created, guest frames with a
	// skipped draw, and the misses that waited, by why the draw could not be skipped.
	AsyncSkipped,
	AsyncSkipFrames,
	AsyncBlockWrites,    // writes buffers, images or memory by address
	AsyncBlockDepthOnly, // no color attachment (depth-only: later passes test or sample it)
	AsyncBlockQuery,     // an occlusion query counts its samples
	AsyncBlockPasses,    // aliased-target passes
	AsyncBlockCap,       // skipped for ASYNC_PIPELINE_SKIP_FRAMES guest frames, still queued
	StagingWriteNs,      // guest bytes written into the upload staging ring (any thread)
	StagingWriteBytes,
	StagingWriteCount,
	// KYTY_SYNC_CMP: synchronized buffer-backing reads, [caller walk/other][bytes unchanged/changed] x
	// {reads, sync ns, GPU-written bytes downloaded, reads whose 4 KiB page is unchanged too}.
	SyncCmpBase,
	SyncCmpEnd = SyncCmpBase + 16,
	SyncNoDownloadWalk, // synchronized reads that downloaded no byte (walk / other)
	SyncNoDownloadOther,
	SyncStillDirtyWalk, // ... after which the read range was still GPU-dirty
	SyncStillDirtyOther,
	// KYTY_WDWORD: synchronized walker reads seen by the probe after a program lookup's walk, and
	// those outside it; per read: changed by the synchronization (backing before != after), first of
	// its (program, lookup in the frame, read in the walk) key, different from that key's previous value.
	WdwordReads,
	WdwordOutside,
	WdwordChanged,
	WdwordFirst,
	WdwordDiffPrev,
	// Re-walks with the read dword replaced (value + 1, then 0): override applied, walk failed, and
	// what differed from the unperturbed re-walk: flattened SRT, descriptor dwords, resource or image
	// activity, walked outputs (indirect keys, uniform fill), guest load addresses. Baseline: the
	// unperturbed re-walk did not reproduce the lookup's flattened SRT (must stay 0).
	WdwordProbes,
	WdwordApplied,
	WdwordFailed,
	WdwordFlat,
	WdwordDesc,
	WdwordActive,
	WdwordOutputs,
	WdwordAddrs,
	WdwordNone,
	WdwordBaseline,
	// KYTY_BACK_SAME: CP back records, and those whose resolved draw state equals the previous one.
	BackRecords,
	BackRecordNs,
	BackShaped,    // single-pass draws compared with the previous one
	BackSamePipe,  // same pipeline
	BackSameRt,    // same render targets
	BackSameBind,  // same descriptors and buffers of every stage
	BackSameVtx,   // same vertex and index buffers
	BackSameDyn,   // dynamic state unchanged
	BackSame,      // all of the above
	BackSameNs,
	BackSameClean, // ... with no staging copy during the record
	BackSameCleanNs,
	// What differs from the previous adjacent record: dynamic state groups, then binding parts.
	BackDiffViewport,  // viewports, scissors
	BackDiffDepth,     // depth test/write/compare/bounds/bias
	BackDiffStencil,
	BackDiffRaster,    // cull mode, front face, line width
	BackDiffBlend,     // blend constants, feedback loop aspects
	BackDiffSnapshot,  // descriptor values of a stage
	BackDiffUserData,  // user data of a stage
	BackDiffBuffers,   // host buffers bound
	BackDiffImages,    // image views or layouts
	BackDiffSamplers,
	BackDiffShaderData,
	BackDiffFlat,      // flattened SRT of a stage
	BackSameNoData,    // same except shader data, user data and flattened SRT
	BackCompareNs,     // time of the comparison itself (inside the record time)
	Count
};

[[nodiscard]] bool Enabled() noexcept;
// KYTY_BIND_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): times the binding phases.
[[nodiscard]] bool BindProfEnabled() noexcept;
// KYTY_SRT_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): splits program lookups (walk,
// descriptor evaluation, specialization, permutation search), counts what changes between lookups and
// which resources they bind, and splits texture/sampler/buffer resolution ("PERF srtsplit" line).
[[nodiscard]] bool SrtProfEnabled() noexcept;
// KYTY_SYNC_CMP=1 / KYTY_BACK_SAME=1 (measurement, default off; need KYTY_PERF_STATS=1): "PERF synccmp"
// (backing bytes before and after each synchronized read) and "PERF backsame" (CP back records equal to
// the previous one) lines.
[[nodiscard]] bool SyncCmpEnabled() noexcept;
// KYTY_WDWORD=1 (step 2.13 measurement, default off; needs KYTY_SYNC_CMP=1): "PERF wdword" line and
// "WDWORD0" log lines (pipelineCache.cpp ProbeWalkerReads).
[[nodiscard]] bool WdwordEnabled() noexcept;
[[nodiscard]] bool BackSameEnabled() noexcept;
void               Add(Counter counter, uint64_t value) noexcept;
[[nodiscard]] uint64_t Peek(Counter counter) noexcept;
// GPU thread: marks the calling thread (wait attribution) and prints the report when due.
void               MarkGpuThread() noexcept;
[[nodiscard]] bool IsGpuThread() noexcept;
void               MaybeReport() noexcept;
// KYTY_BOUNDS_MODE=2 counters (Spirv::StorageBoundsCounter, host-mapped), reported with the stats.
void SetStorageBoundsCounters(const volatile uint32_t* counters) noexcept;
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
