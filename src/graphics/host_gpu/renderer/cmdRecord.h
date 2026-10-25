#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORD_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstdint>

// KYTY_CP_RECORD_THREAD=N (default 0 = off): Vulkan recording of the main command buffer on a
// second thread (pipelined command processor, first stage).
//
// The GPU thread (producer) keeps doing everything it does today: PM4 parsing, register state,
// programs, SRT, caches, page protection, descriptor contents, pipelines, layout tracking. Only the
// vkCmd* driver calls on the main scheduler's command buffer change thread: the dispatcher entry
// points are replaced, the producer deep-copies each call (arguments and the arrays they point to)
// into an in-order ring, and a recorder thread replays the calls on the same command buffer in the
// same order. The command stream is identical; it is only recorded later, and always before the
// buffer ends and is submitted: without KYTY_CP_RECORD_SUBMIT every vkEndCommandBuffer first drains
// the ring; with it, end and submission are themselves queued behind the buffer's calls (see
// below), so anything that depends on GPU execution (waits, readbacks, labels, EOP interrupts,
// flips) still waits for a tick whose work was recorded and submitted in order. Calls without a
// deep-copy implementation, calls from other threads on that buffer, command pool operations and
// object destruction from the GPU thread drain first and then run directly, so the ring never holds
// a call that could see a different Vulkan state than it would have seen when it was made.
//
// Modes: 1 = recorder thread; 2 = capture and replay inline on the GPU thread (deep-copy check, no
// concurrency); 3 = timing only (all calls direct, driver time per entry point in PERF cmdrec).
namespace Libs::Graphics::CmdRecord {

[[nodiscard]] int Mode() noexcept;
// After VULKAN_HPP_DEFAULT_DISPATCHER.init(device). No-op when the mode is 0.
void Install(vk::Device device);
// The main scheduler's command buffer being recorded (nullptr: none).
void SetTarget(vk::CommandBuffer buffer) noexcept;
// GPU thread: wait until the recorder thread has recorded every captured call.
void Drain() noexcept;
// Any thread with no Vulkan queue lock held (shutdown): wait until every published call ran.
void DrainAll() noexcept;
// GPU thread, with the PERF report.
void Report(double seconds);

// Modes 1 and 2 with KYTY_CP_RECORD_SUBMIT (default 1): vkBeginCommandBuffer, vkEndCommandBuffer
// and the main queue submission (Call below) are captured too, so the GPU thread never waits for
// the recorder at a command buffer boundary. Ticks are still assigned by the GPU thread in
// submission order; host waits and other queues' waits on a tick whose submission is still in the
// ring simply wait longer (timeline semaphores).
// Only on the GPU thread: another thread (a test driving the command processor) submits directly.
[[nodiscard]] bool DefersSubmit() noexcept;
// Runs fn(copy of data) on the recorder thread after every call captured before it (mode 2:
// inline). `data` must be trivially copyable; `bytes` at most 64 KiB.
void Call(void (*fn)(void* data), const void* data, size_t bytes);
// Crash reports: ring positions and recorder state (any thread, lock-free reads).
void Describe() noexcept;

} // namespace Libs::Graphics::CmdRecord

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORD_H_
