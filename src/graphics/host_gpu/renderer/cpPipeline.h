#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <array>
#include <cstdint>

// KYTY_CP_PIPELINE (default 1; 0 = off): pipelined command processor, slice 2.1 (direct graphics
// draws).
// The GPU thread (front) decodes PM4, keeps the registers and looks up the draw's programs (SRT
// walks); it then copies everything the rest of the draw reads into a DrawRecord: the registers, the
// draw arguments, the looked-up state and a copy of each stage's resource snapshot (the program
// cache rewrites its own copy at the next lookup). A host back thread executes the records in order:
// render targets, caches, bindings, pipelines, recording, i.e. everything that follows the program
// lookup. The front drains the queue (waits until the back is idle) before every packet that is not
// a register write or a direct draw, before running commands from other threads, before any submit
// or timeline wait of its own, before reading GPU-written memory and before faults it takes, so
// labels, EOP/RELEASE_MEM, waits and flips keep their order and their point in the stream.
// Dispatches, indirect draws, draws that write memory from shaders and draws using the image heap
// run on the GPU thread as before, after a drain.
// Only the back holds the render mutex for a queued draw; the front's lookups use the program
// cache, which only the GPU thread touches, and the buffer tracker's own lock. A pixel permutation
// that the resolved targets reject is looked up again by the front (the back parks meanwhile).
// KYTY_CP_RECORD_THREAD stays a third stage: front and back are its producers in turn, never at once
// (the front drains before it records anything itself).
// The back thread is host-only: it never runs guest code nor touches guest TLS. Hand-over uses
// explicit acquire/release atomics (no reliance on x86 ordering).
// Modes: 0 = off (every draw runs on the GPU thread); 1 = back thread (default when the variable is
// unset); 2 = records executed inline on the GPU thread (checks that a record holds everything its
// draw reads, no concurrency). Any other value is off.
namespace Libs::Graphics::CpPipeline {

struct DrawRecord {
	bool             indexed   = false;
	uint64_t         submit_id = 0;
	DrawIndexArgs    index_args;
	DrawAutoArgs     auto_args;
	HW::Context      registers;
	HW::UserConfig   user_config;
	HW::Shader       shaders;
	DrawRenderState  state;
	bool             guessed_single_sample = true;
	// Vertex stages 0-2, then the pixel stage.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 4> snapshots;
	// Back -> front: the single-sample result of the draw's targets (guess feedback).
	bool     feedback_valid         = false;
	uint64_t feedback_ps_address    = 0;
	bool     feedback_single_sample = true;
	// Back -> front: program lookup again with this single-sample value.
	bool relookup_single_sample = true;
};

using ExecuteFn  = void (*)(void* context, DrawRecord& record);
using RelookupFn = void (*)(void* context, DrawRecord& record);
using FeedbackFn = void (*)(void* context, const DrawRecord& record);

[[nodiscard]] int Mode() noexcept;
// KYTY_CP_PIPELINE_MIN_FRAME_MS=M (default 25, 0 = always): mode 1 queues draws only while presented
// frames are at least M ms apart on average (PerfStats::RecentFrameUs), as the record thread does. At
// the display rate the GPU thread has time to spare and each hand-off only delays the submission.
[[nodiscard]] bool Engaged() noexcept;
// Registered once by the renderer before the first record.
void Init(void* context, ExecuteFn execute, RelookupFn relookup, FeedbackFn feedback);

[[nodiscard]] bool IsBackThread() noexcept;
// GPU thread with the pipeline on (records may be queued).
[[nodiscard]] bool FrontActive() noexcept;
// No queued or running record (any thread; acquire).
[[nodiscard]] bool BackIdle() noexcept;

// Why the front drains (KYTY_CP_DRAIN_PROF). Values below PacketReasons are PM4 packets, built by
// PacketReason(); the named reasons follow.
enum class Reason : uint32_t {
	PacketReasons = 0x20000,
	Other         = PacketReasons,
	Command,       // command sent by another thread, run between packets
	GpuCommand,    // command run by the GPU thread loop
	SubmitBegin,   // start of a guest submission slice
	SchedBegin,    // scheduler Begin
	EndRendering,
	Flush,
	FlushAndWait,
	Finish,
	SchedWait,     // scheduler Wait(tick)
	Submit,
	TimelineWait,  // host timeline semaphore wait on the GPU thread
	Fault,         // GPU-thread page fault
	Readback,      // front read of GPU-written memory
	ImageHeap,     // image heap acquired inside a pipelined lookup
	InlineHeap,    // looked-up draw run inline: image heap
	InlineWrites,  // looked-up draw run inline: writes memory from shaders (or no vertex program)
	FallbackArgs,  // the Stat::Fallback* draws, in the same order
	FallbackTargets,
	FallbackShaders,
	FallbackFast,
	QueueFull,     // Acquire waited for a free slot
	// Commands from other threads by GpuCommandKind (Command = Other): CPU read of GPU-written
	// bytes, fault window published asynchronously / synchronously, label repair, unmap.
	CommandReadMemory,
	CommandFaultAsync,
	CommandFaultSync,
	CommandLabelRepair,
	CommandUnmap,
	Count
};
[[nodiscard]] Reason CommandReason(uint32_t gpu_command_kind) noexcept;
// Packet key: opcode, a sub-reason byte (IT_NOP's R, the event type of EVENT_WRITE/RELEASE_MEM) and
// the queue (async compute or graphics).
[[nodiscard]] constexpr Reason PacketReason(uint32_t opcode, uint32_t sub, bool compute) noexcept {
	return static_cast<Reason>((opcode & 0xffu) | ((sub & 0xffu) << 8u) | (compute ? 0x10000u : 0u));
}
// The drain reason of a draw that PipelineDraw did not queue (its last Fallback* stat).
[[nodiscard]] Reason FallbackReason() noexcept;
// KYTY_CP_DRAIN_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): every front drain per
// reason, whether it waited for a busy back thread and how long ("PERF cpdrain" line).
[[nodiscard]] bool DrainProfEnabled() noexcept;
// KYTY_CP_DRAIN_PROF=2: also the GPU thread time of the work it runs itself ("PERF cpfront" line):
// handlers of draining packets, the indirect draw paths, draws run on the GPU thread outside an
// indirect packet (no-ES draws by their stages and primitive type) and commands from other threads.
// Packet times include the indirect paths of their packet, which include their draws.
[[nodiscard]] bool FrontProfEnabled() noexcept;
enum class Front : uint32_t {
	Packet = 1,  // value: the packet's Reason
	Indirect,    // value: IndirectPath | indexed << 4
	Inline,      // value: Reason (fallback or inline draw)
	NoEs,        // value: ls | hs << 1 | gs << 2 | ps << 3 | primitive type << 4
	Command,     // value: Reason (CommandReason)
	Dispatch,    // value: DispatchPhase | async compute ring << 8 (attribute rows: count only, no time)
};
enum class IndirectPath : uint32_t { GpuRecords, GpuMesh, CpuRecords };
// RenderExecutor::DispatchDirect, in order. Attributes count dispatches (buffers/images: their sum).
enum class DispatchPhase : uint32_t {
	Pop, Lock, Lookup, Meta, EndRendering, Pipeline, Bindings, HeapBda, Rebind, Commit, Barrier, Record,
	AttrHeap = 0x20, AttrDma, AttrWrites, AttrChained, AttrBuffers, AttrImages, AttrMeta,
};
void NoteFront(Front family, uint32_t value, uint64_t ns, uint64_t count = 1) noexcept;
[[nodiscard]] uint64_t NowNs() noexcept;
// Times its scope when FrontProfEnabled() (one flag test otherwise).
class FrontScope {
public:
	FrontScope(Front family, uint32_t value, bool on = true) noexcept: m_family(family), m_value(value) {
		if (on && FrontProfEnabled()) m_start = NowNs();
	}
	~FrontScope() {
		if (m_start != 0) NoteFront(m_family, m_value, NowNs() - m_start);
	}
	FrontScope(const FrontScope&)            = delete;
	FrontScope& operator=(const FrontScope&) = delete;

private:
	Front    m_family;
	uint32_t m_value;
	uint64_t m_start = 0;
};

// Front: wait until the back thread is idle. Inside a program lookup a back thread parked on a
// re-lookup request counts as idle (it touches nothing until the front serves it after the lookup).
void Drain(Reason reason = Reason::Other) noexcept;
// Front: serves a pending re-lookup request (between packets, outside lookups).
void Poll() noexcept;
// Front: program lookups of pipelined draws (re-lookup requests are not served inside).
void SetInLookup(bool on) noexcept;
[[nodiscard]] bool InLookup() noexcept;

// Front: a free record (applies the guess feedback of finished ones), then publish it.
DrawRecord& Acquire() noexcept;
void        Publish() noexcept;

// Back thread (or inline mode): the record needs another program lookup; returns once done.
void RequestRelookup(DrawRecord& record) noexcept;

// ImageHeap::Acquire: inside a pipelined lookup, drains and calls the hook once (the renderer then
// runs the draw on the GPU thread); TakeHeapInline() tells and clears it.
using HeapFn = void (*)(void* context);
void               SetHeapHook(HeapFn hook) noexcept;
void               NoteHeapAcquire() noexcept;
[[nodiscard]] bool TakeHeapInline() noexcept;

// KYTY_PERF_STATS: what happened to the draws the front saw ("PERF cppipe" line).
enum class Stat : uint32_t {
	Queued,
	InlineHeap,
	InlineWrites,
	FallbackArgs,     // offset source, indirect, empty or debug dump
	FallbackTargets,  // metadata color operation, resolve or depth copy
	FallbackShaders,  // no valid vertex shader or topology
	FallbackFast,     // frames faster than KYTY_CP_PIPELINE_MIN_FRAME_MS
	Count
};
void Note(Stat stat) noexcept;
void Report(double seconds, uint64_t frames) noexcept;

} // namespace Libs::Graphics::CpPipeline

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_
