#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/cpPipeline.h"
#include "graphics/host_gpu/renderer/cpSplitProf.h"
#include "graphics/host_gpu/guestStacks.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <vector>

namespace Libs::Graphics::CmdRecord {
void Report(double seconds); // cmdRecord.h (KYTY_CP_RECORD_THREAD)
} // namespace Libs::Graphics::CmdRecord

namespace Libs::Graphics::PerfStats {

namespace {

constexpr auto ReportInterval = std::chrono::seconds(5);
constexpr auto CounterCount   = static_cast<size_t>(Counter::Count);

std::array<std::atomic<uint64_t>, CounterCount> g_counters {};
thread_local bool                               g_gpu_thread = false;
const auto                                      g_epoch      = std::chrono::steady_clock::now();
std::atomic<int64_t>                            g_last_flip_arg {0};
std::atomic<uint64_t>                           g_recent_frame_us {0};

uint64_t NowUs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                 std::chrono::steady_clock::now() - g_epoch)
	                                 .count());
}

std::tm LocalNow() noexcept {
	const auto wall = std::time(nullptr);
	std::tm    local {};
#if defined(_WIN32)
	localtime_s(&local, &wall);
#else
	localtime_r(&wall, &local);
#endif
	return local;
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

bool BindProfEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BIND_PROF");
		return Enabled() && value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

bool SrtProfEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_PROF");
		return Enabled() && value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

void Add(Counter counter, uint64_t value) noexcept {
	if (Enabled()) {
		g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
	}
}

void MarkGpuThread() noexcept {
	g_gpu_thread = true;
	if (Enabled()) {
		ShaderRecompiler::IR::EnableWalkStats();
	}
}

bool IsGpuThread() noexcept {
	return g_gpu_thread;
}

void NoteGraphicsPipelineFeedback(bool driver_cache_hit, uint64_t duration_ns) noexcept {
	Add(driver_cache_hit ? Counter::GfxPipelineHits : Counter::GfxPipelineMisses, 1);
	Add(driver_cache_hit ? Counter::GfxPipelineHitNs : Counter::GfxPipelineMissNs, duration_ns);
}

void NoteFlipArg(int64_t arg) noexcept {
	g_last_flip_arg.store(arg, std::memory_order_relaxed);
}

uint64_t RecentFrameUs() noexcept {
	return g_recent_frame_us.load(std::memory_order_relaxed);
}

void NotePresentedFrame() noexcept {
	{
		// Smoothed presentation interval (1/8 of each new interval), always kept: the GPU thread reads
		// it (KYTY_CP_RECORD_MIN_FRAME_MS, KYTY_IDLE_SUBMIT_MIN_FRAME_MS). Presentation runs on one thread.
		static uint64_t last_us = 0;
		const auto      now_us  = NowUs();
		if (last_us != 0) {
			const auto interval = now_us - last_us;
			const auto recent   = g_recent_frame_us.load(std::memory_order_relaxed);
			g_recent_frame_us.store(recent == 0 ? interval : recent - recent / 8 + interval / 8,
			                        std::memory_order_relaxed);
		}
		last_us = now_us;
	}
	struct Recorder {
		std::FILE*                                file = nullptr;
		std::vector<std::pair<uint64_t, int64_t>> pending;
	};
	static Recorder* recorder = [] {
		auto*       r    = new Recorder;
		const char* path = std::getenv("KYTY_FRAME_TIMES");
		if (path != nullptr && path[0] != '\0') {
			r->file = std::fopen(path, "w");
		}
		if (r->file != nullptr) {
			const auto local = LocalNow();
			std::fprintf(r->file, "# start %02d:%02d:%02d steady_us=%" PRIu64 "\n", local.tm_hour,
			             local.tm_min, local.tm_sec, NowUs());
			std::fflush(r->file);
		}
		return r;
	}();
	if (recorder->file == nullptr) {
		return;
	}
	// Presentation runs on one thread.
	recorder->pending.emplace_back(NowUs(), g_last_flip_arg.load(std::memory_order_relaxed));
	if (recorder->pending.size() >= 32) {
		for (const auto& [us, arg]: recorder->pending) {
			std::fprintf(recorder->file, "%" PRIu64 " %" PRId64 "\n", us, arg);
		}
		std::fflush(recorder->file);
		recorder->pending.clear();
	}
}

void MaybeReport() noexcept {
	if (!Enabled()) {
		return;
	}
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
	const auto   ms      = [&](Counter c) { return static_cast<double>(v[static_cast<size_t>(c)]) / 1e6; };
	const auto   n       = [&](Counter c) { return v[static_cast<size_t>(c)]; };
	const double wall_ms = seconds * 1000.0;
	const auto   local   = LocalNow();
	std::printf("PERF stats%02d:%02d:%02d %.1fs: fps=%.2f gpu_busy=%.0f%% draw=%.0fms(%" PRIu64
	            ") dispatch=%.0fms(%" PRIu64 ") programs=%.0fms(%" PRIu64 ") srt=%.0fms recompile=%.0fms(%" PRIu64
	            ") gfx_pipe=%.0fms(%" PRIu64 ",hit=%" PRIu64 "/%.0fms,miss=%" PRIu64 "/%.0fms) cs_pipe=%.0fms(%" PRIu64
	            ") wait_gpu_thread=%.0fms(%" PRIu64 ") wait_other=%.0fms(%" PRIu64 ") side_reads=%.0fms(%" PRIu64
	            ") submits=%.0f/s idle_submits=%.0f/s\n",
	            local.tm_hour, local.tm_min, local.tm_sec, seconds,
	            static_cast<double>(n(Counter::Frames)) / seconds, 100.0 * ms(Counter::GpuBusyNs) / wall_ms,
	            ms(Counter::DrawNs), n(Counter::DrawCount), ms(Counter::DispatchNs), n(Counter::DispatchCount),
	            ms(Counter::ProgramNs), n(Counter::ProgramCount), ms(Counter::SrtNs), ms(Counter::RecompileNs),
	            n(Counter::RecompileCount), ms(Counter::GfxPipelineNs), n(Counter::GfxPipelineCount),
	            n(Counter::GfxPipelineHits), ms(Counter::GfxPipelineHitNs), n(Counter::GfxPipelineMisses),
	            ms(Counter::GfxPipelineMissNs), ms(Counter::CsPipelineNs), n(Counter::CsPipelineCount),
	            ms(Counter::WaitGpuThreadNs), n(Counter::WaitGpuThreadCount), ms(Counter::WaitOtherNs),
	            n(Counter::WaitOtherCount), ms(Counter::SideReadNs), n(Counter::SideReadCount),
	            static_cast<double>(n(Counter::Submits)) / seconds,
	            static_cast<double>(n(Counter::IdleSubmits)) / seconds);
	if (BindProfEnabled()) {
		std::printf("PERF bind %.1fs: prepare=%.0fms find_buffers=%.0fms heap=%.0fms images=%.0fms buffers=%.0fms"
		            " commit=%.0fms(%" PRIu64 ") check=%.0fms transit=%.0fms push=%.0fms writes=%" PRIu64
		            " descriptors=%" PRIu64 "\n",
		            seconds, ms(Counter::BindPrepareNs), ms(Counter::BindFindBuffersNs), ms(Counter::BindHeapNs),
		            ms(Counter::BindImagesNs), ms(Counter::BindBuffersNs), ms(Counter::BindCommitNs),
		            n(Counter::BindCommitCount), ms(Counter::BindCheckNs), ms(Counter::BindTransitNs),
		            ms(Counter::BindPushNs), n(Counter::BindWriteCount), n(Counter::BindDescriptorCount));
	}
	if (SrtProfEnabled()) {
		std::printf("PERF srtsplit %.1fs: lookups=%" PRIu64 " refresh=%.0fms walk=%.0fms eval=%.0fms spec=%.0fms match=%.0fms"
		            " same_perm=%" PRIu64 " multi_perm=%" PRIu64 " miss=%" PRIu64 " user_same=%" PRIu64
		            " desc_same=%" PRIu64 " flat_same=%" PRIu64 " no_load=%" PRIu64 " loads=%" PRIu64
		            " flat_words=%" PRIu64 " flat_needed=%" PRIu64 "/%" PRIu64 " buffers=%" PRIu64 " images=%" PRIu64 " heap_images=%" PRIu64
		            " conv_images=%" PRIu64 " samplers=%" PRIu64 " no_res=%" PRIu64 " heap_only=%" PRIu64
		            " with_buf=%" PRIu64 " with_img=%" PRIu64 " with_samp=%" PRIu64 " format_spec=%" PRIu64
		            " find_image=%.0fms sampler=%.0fms obtain=%.0fms upload=%.0fms view=%.0fms\n",
		            seconds, n(Counter::SrtLookups), ms(Counter::SrtRefreshNs), ms(Counter::SrtWalkNs),
		            ms(Counter::SrtEvalNs), ms(Counter::SrtSpecNs), ms(Counter::SrtMatchNs), n(Counter::SrtSamePerm),
		            n(Counter::SrtMultiPerm), n(Counter::SrtMiss), n(Counter::SrtUserSame), n(Counter::SrtDescSame),
		            n(Counter::SrtFlatSame), n(Counter::SrtNoLoad), n(Counter::SrtLoads), n(Counter::SrtFlatWords), n(Counter::SrtFlatNeeded), n(Counter::SrtFlatPlan),
		            n(Counter::SrtBuffers), n(Counter::SrtImages), n(Counter::SrtHeapImages), n(Counter::SrtConvImages),
		            n(Counter::SrtSamplers), n(Counter::SrtNoResources), n(Counter::SrtHeapOnly),
		            n(Counter::SrtWithBuffers), n(Counter::SrtWithImages), n(Counter::SrtWithSamplers),
		            n(Counter::SrtFormatSpec), ms(Counter::BindFindImageNs), ms(Counter::BindSamplerNs),
		            ms(Counter::BindObtainNs), ms(Counter::BindUploadNs), ms(Counter::BindViewNs));
	}
	if (const auto walk = ShaderRecompiler::IR::TakeWalkStats(); walk.walks != 0 || walk.reuse_checks != 0) {
		std::printf("PERF srtwalk %.1fs: walk=%.0fms(%" PRIu64 ") backing_reads=%" PRIu64
		            " fault_reads=%" PRIu64 " protected_calls=%" PRIu64 "(%.0fms,synced=%" PRIu64 ") evictions=%" PRIu64
		            " reuse=%" PRIu64 "/%" PRIu64 "(%.0fms,diffs=%" PRIu64 ")\n",
		            seconds, static_cast<double>(walk.ns) / 1e6, walk.walks, walk.backing_reads,
		            walk.fault_reads, walk.protected_calls, static_cast<double>(walk.protected_ns) / 1e6,
		            walk.protected_synced, walk.protected_evictions, walk.reuse_hits, walk.reuse_checks,
		            static_cast<double>(walk.reuse_ns) / 1e6, walk.reuse_diffs);
	}
	CmdRecord::Report(seconds);
	CpSplitProf::Report(seconds);
	CpPipeline::Report(seconds);
	GuestStacks::Report();
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PerfStats
