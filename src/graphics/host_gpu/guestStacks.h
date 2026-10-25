#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTSTACKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTSTACKS_H_

#include "common/common.h"

#include <cstdint>

// Guest thread stacks and fiber contexts, for page-protection forensics.
//
// On Windows a fault is delivered by pushing the exception record and context on the faulting
// thread's own stack. Guest threads run on guest stacks, and a guest stack can live in guest memory
// that the GPU tracking protects (read-only for watched uploads and images, no-access for GPU-dirty
// pages). When the thread then faults while the pages just below its stack pointer are protected
// (typically a large frame `sub rsp` lands in them and its first store faults), the kernel cannot
// write the exception frame and terminates the process directly: no vectored handler runs, nothing
// is printed, only Windows Error Reporting sees the fault (and writes its dump if LocalDumps is set).
//
// Registered ranges: every guest thread stack (user-supplied or allocated by the emulator) for the
// life of the thread, and every fiber context. PageManager reports each protection that touches a
// registered range (first ones with a host backtrace, then periodically), and the guest fault
// handler describes the faulting thread's stack.
//
// KYTY_STACK_INVALIDATE (default 1): a user-supplied stack or fiber context is handed to the GPU
// tracking as written by the CPU before the thread or fiber first runs on it (the same invalidation
// a guest file read into that memory does: GPU-written bytes are published first, watches dropped,
// cached copies re-uploaded on their next use). Memory recycled from textures or buffers then
// carries no stale watch into the stack. 0 disables it.
namespace Libs::Graphics::GuestStacks {

[[nodiscard]] bool InvalidateEnabled() noexcept;
// Registers [begin, begin + size) (replaces an entry with the same base).
void Register(uint64_t begin, uint64_t size, const char* owner);
void Unregister(uint64_t begin);
void NoteInvalidated() noexcept;
// Fast path for every protection: false unless a registered range shares a 4 MiB region.
[[nodiscard]] bool MaybeHit(uint64_t vaddr, uint64_t size) noexcept;
// A protection (read-only or no-access) of [vaddr, vaddr + size) that may touch a registered range.
void OnProtect(uint64_t vaddr, uint64_t size, bool no_access);
// Guest fault handler: the registered range holding `rsp` and the host protection of the pages
// around the stack pointer (the exception frame needs the pages below it writable).
void Describe(uint64_t rsp);
// PERF stacks line (GPU thread, with the PERF report) when a counter changed.
void Report();

} // namespace Libs::Graphics::GuestStacks

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTSTACKS_H_
