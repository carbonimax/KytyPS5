#ifndef GRAPHICS_GUEST_GPU_CP_LABEL_PROF_H
#define GRAPHICS_GUEST_GPU_CP_LABEL_PROF_H

#include <cstdint>

// KYTY_CP_LABEL_PROF=1 (default 0, measurement only): label latency between the command-processor
// rings and their waiters. Needs KYTY_PERF_STATS=1 for the report (`PERF cplabel` lines, every 5 s).
// - Labels: every RELEASE_MEM and every WRITE_DATA of at most two dwords, by writer class (graphics ring
//   or compute ring): guest enqueue of the submission -> first slice of the submission -> packet decode
//   -> guest-visible write (decode itself, except for KYTY_EOP_AT_GPU=1 deferred labels).
// - Interrupts: RELEASE_MEM interrupt packets, decode -> TriggerInterrupt (host GPU completion).
// - WAIT_REG_MEM that blocks at least once, by waiter ring and by the ring whose label satisfied it. The
//   wait is split into: producer submission not enqueued yet, producer enqueued but not started, producer
//   started but its packet not decoded yet, label written but not visible yet (GPU), and visible but the
//   waiter not re-run yet (scheduler / lane).
// - Guest CPU waits in the event queue that return a graphics interrupt, split the same way.
// - Command-processor thread idle time, and the part of it while the graphics ring waits on a compute label.
// Off, every hook is one predictable branch on a cached flag; nothing changes the emulator's behaviour.
namespace Libs::Graphics::CpLabelProf {

bool EnabledSlow() noexcept;
inline bool Enabled() noexcept {
	static const bool enabled = EnabledSlow();
	return enabled;
}

uint64_t NowNs() noexcept;

enum class Kind : uint8_t { ReleaseMem, WriteData };
enum class Path : uint8_t { CpuStore, GpuOwned, Deferred };

// The calling thread processes a slice of the submission of ring `queue_id` (0 = graphics,
// 1.. = compute rings), enqueued by the guest at `enqueue_ns` and first started at `start_ns`.
void SliceBegin(uint32_t queue_id, uint64_t enqueue_ns, uint64_t start_ns) noexcept;
void SliceEnd() noexcept;
void Frame() noexcept;

void Label(Kind kind, const void* address, uint32_t size, uint64_t value, Path path) noexcept;
// KYTY_EOP_AT_GPU=1: a deferred label reached guest memory (completion thread).
void LabelDelivered(uint64_t address) noexcept;

void InterruptPacket(int event_id) noexcept;
void InterruptDelivered(int event_id) noexcept;

// WAIT_REG_MEM of the ring in the current slice: the test failed / passed.
void WaitBlocked(const void* address) noexcept;
void WaitPassed(const void* address, uint64_t ref, uint64_t mask, uint32_t func) noexcept;

// A guest event-queue wait that returned graphics interrupt `event_id`; `blocked` = it had to sleep.
void CpuWaitDone(uint64_t event_id, uint64_t start_ns, bool blocked) noexcept;

// Command-processor thread `lane` (0 = GPU thread / graphics lane, 1 = compute lane) sleeps for work.
void IdleBegin(uint32_t lane) noexcept;
void IdleEnd(uint32_t lane) noexcept;

void Report(double seconds);

} // namespace Libs::Graphics::CpLabelProf

#endif // GRAPHICS_GUEST_GPU_CP_LABEL_PROF_H
