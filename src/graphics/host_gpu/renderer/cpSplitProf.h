#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPSPLITPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPSPLITPROF_H_

#include <cstddef>
#include <cstdint>

// KYTY_CP_SPLIT_PROF=1 (measurement, default off; needs KYTY_PERF_STATS=1): how much of the GPU
// thread a pipelined command processor could overlap. The GPU thread is cut into segments at the
// points where a pipeline would have to drain its queue (synchronization packets, submissions, flips,
// commands from other threads, synchronous reads of GPU-written memory). Inside a segment every
// pipelinable item (a draw, or a dispatch for the full model) has a front part (everything on the GPU
// thread since the previous item, plus its program lookup) and a back part (the rest of its handler:
// caches, bindings, pipelines, recording). A two-stage flow shop is simulated per segment: the back of
// item i starts when its front is done and the back of item i-1 is done. Saved time = serial time -
// simulated time. Drain models reported side by side ("PERF cpsplit" lines):
// - s21: slice 2.1, only direct graphics draws are pipelined; dispatches and indirect draws drain;
// - full: draws and dispatches are pipelined, only synchronization drains;
// - s21r: what KYTY_CP_PIPELINE really queues today: s21 with its packet rule (every packet but
//   register writes, fetcher and direct draws drains), draws it does not queue (offset source,
//   empty, metadata/resolve/depth copy, no valid vertex shader or topology) draining before their
//   lookup, draws that write memory, use the image heap or have no compiled vertex program draining
//   after it, and GPU-thread faults draining;
// - s24d: s21r plus direct dispatches queued, buffer-writing ones included, with a per-range hazard:
//   a queued dispatch records the guest ranges of the storage buffers and storage images it binds as
//   written, and the front drains only when one of its reads (SRT walker loads, backing reads,
//   indirect buffers, faults, memory waits) overlaps a range written by a queued item. Dispatches
//   whose writes the bindings do not describe (address writes, device addresses, image heap, GDS),
//   consumed fills/clears and indirect dispatches stay on the front;
// - s24r: s24d plus RELEASE_MEM/EVENT_WRITE queued in order (their handler is back work; GDS
//   readbacks drain) and ACQUIRE_MEM (a no-op) not draining; every memory wait drains;
// - s24w: s24r where a memory wait drains only when it overlaps a range written by a queued item, or
//   when it is not satisfied (the front stops anyway); a wait on a label of a queued RELEASE_MEM is
//   resolved on the front with the value the packet carries;
// - s24c: s24w with the coarse hazard (any lookup or wait after a queued writer drains);
// - s24x: s24w where front reads never drain (ceiling, incorrect).
// In s21 and full, a lookup after an item that writes memory from shaders drains first (the front
// may read what the queued item writes). GPU thread only; nothing changes when off.
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
	Submit,      // command buffer submitted or GPU timeline waited outside any item
	Command,     // command sent by another thread (SendCommand), run between packets
	SyncRead,    // front read of GPU-written memory (walker/backing readback)
	Hazard,      // coarse: lookup after a queued item that writes memory from shaders
	Idle,        // end of a busy period of the GPU thread
	Dispatch,    // s21/s21r: dispatches are not queued
	Indirect,    // indirect draws (CPU-read or GPU-read records)
	Inline,      // the item runs on the front after its lookup (writes, image heap, ...)
	Fallback,    // s21r/s24: the item drains before its lookup (draw not queued, indirect dispatch)
	Fault,       // s21r/s24: GPU-thread fault outside a back part
	HazardSrt,   // s24: SRT walker load in a range a queued item writes
	HazardRead,  // s24: backing read (synchronizing or not) in such a range
	HazardPm4,   // s24: indirect buffer in such a range
	HazardFault, // s24: front fault in such a range
	HazardWait,  // s24w: memory wait in such a range
	Count
};

// Sync: RELEASE_MEM / EVENT_WRITE handlers (queued in order by s24r/s24w/s24c/s24x, drains elsewhere).
enum class ItemKind : uint32_t { Draw, Indirect, Dispatch, Sync };

// Where a front read comes from (s24 hazard causes).
enum class ReadSource : uint32_t { Srt, Backing, Pm4 };

[[nodiscard]] bool Enabled() noexcept;

// PM4 loop, before the handler of each packet (custom = IT_NOP's R field).
void Packet(uint32_t opcode, uint32_t custom) noexcept;
void Drain(Cause cause) noexcept;
// Draw and dispatch handlers of the renderer.
void ItemBegin(ItemKind kind) noexcept;
void ItemEnd() noexcept;
void NoteItemWrites() noexcept;
// s21r/s24: the item runs on the front after its lookup (writes or accesses a queue cannot describe,
// image heap, consumed fill/clear, no compiled vertex program).
void NoteItemInline() noexcept;
// s21r/s24: the item drains before its lookup and runs on the front (draw the pipeline does not
// queue, indirect dispatch; `cause` = ReleaseMem for a GDS readback).
void NoteItemFallback(Cause cause = Cause::Fallback) noexcept;
// s24: guest range [address, address + size) that the item writes (storage buffer or image).
void NoteItemWriteRange(uint64_t address, uint64_t size) noexcept;
// Guest memory a RELEASE_MEM/EVENT_WRITE item writes (s24w statistics).
void NoteLabel(uint64_t address, uint64_t size) noexcept;
// WAIT_REG_MEM reads this guest memory; `satisfied`: the wait passes now.
void WaitMemory(uint64_t address, uint64_t size, bool satisfied) noexcept;
// Front reads of guest memory (s24 hazard). Inside a lookup they are checked when it ends, so the
// check stays out of the measured front time; in a back part they are ignored.
void FrontRead(uint64_t address, uint64_t size, ReadSource source) noexcept;
// SRT walker loads: `count` addresses (4-byte loads) every `stride` entries.
void FrontReadLoads(const uint64_t* addresses, size_t count, size_t stride) noexcept;
// Synchronous read of GPU-written memory (drains; classified by the s24 hazard).
void SyncRead(uint64_t address, uint64_t size) noexcept;
// GPU-thread page fault at address (drains in s21r/s24 outside back parts).
void Fault(uint64_t address) noexcept;
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
