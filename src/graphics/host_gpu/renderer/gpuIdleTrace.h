#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUIDLETRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUIDLETRACE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// KYTY_GPU_IDLE_TRACE=<file> (measurement, default off): a timeline on the steady clock of
// KYTY_FRAME_TIMES, to attribute host GPU idle gaps offline.
//   G <sem> <tick> <waits> <submit_us> <done_us>  every main-queue submission, in queue order; the
//                                                 completion is observed by a waiter thread
//   Q <tick> <us>                                 the GPU thread handed a submission to the recorder
//   S <tag> <thread> <start_us> <end_us>          a state of an emulator thread (spans of one tag on
//                                                 one thread closer than 100 us are merged)
// Timer-based tags (busy, translate, pipelines, waits, side reads) need KYTY_PERF_STATS=1.
namespace Libs::Graphics::GpuIdleTrace {

enum class Tag : uint8_t {
	Busy,        // GPU thread processing a guest submission slice or command
	Command,     // of which a queued command (flip, ...)
	IdleEmpty,   // GPU thread asleep: no guest submission queued
	IdleBlocked, // GPU thread asleep: every queued submission blocked (WAIT_REG_MEM, label)
	Translate,   // shader translation
	GfxPipeline, // vkCreateGraphicsPipelines
	CsPipeline,  // vkCreateComputePipelines
	WaitGpu,     // GPU thread blocked on the host GPU timeline
	WaitOther,   // other thread blocked on the host GPU timeline
	SideRead,    // readback-queue copy wait
	Drain,       // KYTY_CP_PIPELINE front waiting for the back
	BackBusy,    // KYTY_CP_PIPELINE back executing a record
	RecBusy,     // KYTY_CP_RECORD_THREAD recorder replaying calls
	Reprotect,   // batched uploads / reprotection flush
	Fault,       // GPU fault buffer processing
	Readback,    // color metadata readback
	Acquire,     // vkAcquireNextImageKHR (with the slot wait before it)
	Present,     // vkQueuePresentKHR
	Count
};

[[nodiscard]] bool     On() noexcept;
[[nodiscard]] uint64_t NowUs() noexcept;
void                   SetThreadName(const char* name) noexcept;
void                   Span(Tag tag, uint64_t start_us, uint64_t end_us) noexcept;
// Right after vkQueueSubmit on the main queue, under the queue lock (keeps queue order).
void NoteSubmit(vk::Device device, vk::Semaphore semaphore, uint64_t tick, uint32_t waits) noexcept;
void NoteQueued(uint64_t tick) noexcept;

class Scope {
public:
	explicit Scope(Tag tag) noexcept: m_tag(tag), m_on(On()) {
		if (m_on) m_start = NowUs();
	}
	~Scope() {
		if (m_on) Span(m_tag, m_start, NowUs());
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Tag      m_tag;
	bool     m_on;
	uint64_t m_start = 0;
};

} // namespace Libs::Graphics::GpuIdleTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUIDLETRACE_H_
