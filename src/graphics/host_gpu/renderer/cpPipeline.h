#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <array>
#include <cstdint>

// KYTY_CP_PIPELINE (default 0): pipelined command processor, slice 2.1 (direct graphics draws).
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
// Modes: 1 = back thread; 2 = records executed inline on the GPU thread (checks that a record holds
// everything its draw reads, no concurrency).
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

// Front: wait until the back thread is idle. Inside a program lookup a back thread parked on a
// re-lookup request counts as idle (it touches nothing until the front serves it after the lookup).
void Drain() noexcept;
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
void Report(double seconds) noexcept;

} // namespace Libs::Graphics::CpPipeline

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPPIPELINE_H_
