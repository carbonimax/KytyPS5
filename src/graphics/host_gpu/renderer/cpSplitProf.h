#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPSPLITPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPSPLITPROF_H_

#include <cstdint>

// KYTY_CP_SPLIT_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): how much of the GPU
// thread a pipelined command processor could overlap. The GPU thread is cut into segments at the
// points where a pipeline would have to drain its queue (synchronization packets, submissions, flips,
// commands from other threads, synchronous reads of GPU-written memory). Inside a segment every
// pipelinable item (a draw, or a dispatch for the full model) has a front part (everything on the GPU
// thread since the previous item, plus its program lookup) and a back part (the rest of its handler:
// caches, bindings, pipelines, recording). A two-stage flow shop is simulated per segment: the back of
// item i starts when its front is done and the back of item i-1 is done. Saved time = serial time -
// simulated time. Two drain models are reported side by side ("PERF cpsplit" line):
// - s21: slice 2.1, only direct graphics draws are pipelined; dispatches and indirect draws drain;
// - full: draws and dispatches are pipelined, only synchronization drains.
// In both, a lookup after an item that writes memory from shaders drains first (the front may read
// what the queued item writes). GPU thread only; nothing changes when off.
namespace Libs::Graphics::CpSplitProf {

enum class Cause : uint32_t {
	WaitRegMem,
	ReleaseMem,
	EventWrite,
	AcquireMem,
	WriteData,
	DmaData,
	CopyData,
	Predication,
	Flip,
	OtherPacket,
	Submit,    // command buffer submitted or GPU timeline waited outside any item
	Command,   // command sent by another thread (SendCommand), run between packets
	SyncRead,  // front read of GPU-written memory (walker/backing readback)
	Hazard,    // lookup after a queued item that writes memory from shaders
	Idle,      // end of a busy period of the GPU thread
	Dispatch,  // s21 only
	Indirect,  // s21 only: indirect draws (CPU-read or GPU-read records)
	Count
};

enum class ItemKind : uint32_t { Draw, Indirect, Dispatch };

[[nodiscard]] bool Enabled() noexcept;

// PM4 loop, before the handler of each packet (custom = IT_NOP's R field).
void Packet(uint32_t opcode, uint32_t custom) noexcept;
void Drain(Cause cause) noexcept;
// Draw and dispatch handlers of the renderer.
void ItemBegin(ItemKind kind) noexcept;
void ItemEnd() noexcept;
void NoteItemWrites() noexcept;
// Program lookup (front work) inside an item.
void LookupBegin() noexcept;
void LookupEnd() noexcept;
// Indirect draws issue their draws through the direct handlers.
void SetIndirect(bool on) noexcept;
// Busy periods of the GPU thread (idle time is excluded from the segments).
void BusyBegin() noexcept;
void BusyEnd() noexcept;
// Submissions and timeline waits: a drain when not inside an item.
void NoteSubmitOrWait() noexcept;

void Report(double seconds) noexcept;

class ItemScope {
public:
	explicit ItemScope(ItemKind kind) noexcept: m_on(Enabled()) {
		if (m_on) ItemBegin(kind);
	}
	~ItemScope() {
		if (m_on) ItemEnd();
	}
	ItemScope(const ItemScope&)            = delete;
	ItemScope& operator=(const ItemScope&) = delete;

private:
	bool m_on;
};

class LookupScope {
public:
	LookupScope() noexcept: m_on(Enabled()) {
		if (m_on) LookupBegin();
	}
	~LookupScope() {
		if (m_on) LookupEnd();
	}
	LookupScope(const LookupScope&)            = delete;
	LookupScope& operator=(const LookupScope&) = delete;

private:
	bool m_on;
};

} // namespace Libs::Graphics::CpSplitProf

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPSPLITPROF_H_
