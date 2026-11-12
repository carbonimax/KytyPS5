#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUMARKERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUMARKERS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {
struct GraphicContext;
}

// KYTY_GPU_MARKERS=1 (diagnostic, default off; needs VK_AMD_buffer_marker): each dispatch and draw
// of the main queue writes its sequence number into a host-coherent buffer when the CP reaches it
// (top of pipe, "begun") and when it and everything before it is done (bottom of pipe, "finished").
// On a device loss, the commands after the last finished one are written with their shader, groups
// and bound guest ranges to KYTY_GPU_MARKERS_FILE (default gpu_markers.txt).
namespace Libs::Graphics::GpuMarkers {

[[nodiscard]] bool Requested() noexcept;
// After the allocator exists; no-op unless requested and the extension is enabled.
void Init(GraphicContext& graphics);
[[nodiscard]] bool Active() noexcept;

// Thread-local: the shader whose bindings follow (always kept, also used by the import net log).
void               SetShader(uint64_t address) noexcept;
[[nodiscard]] uint64_t Shader() noexcept;
// kind: 0 device memory, 1 imported guest memory, 2 stream copy.
// buffer/buffer_size: the cache buffer bound (its guest base), 0 for a stream copy.
// handle/offset: the host buffer and byte offset the shader reads (KYTY_GPU_MARKERS_CAP copies).
void NoteBinding(uint64_t vaddr, uint64_t size, bool written, int kind, uint64_t buffer = 0,
                 uint64_t buffer_size = 0, VkBuffer handle = VK_NULL_HANDLE, uint64_t offset = 0) noexcept;

// KYTY_GPU_MARKERS_CAP=addr[,addr...] (hex compute shaders, diagnostic): each of their dispatches copies
// every binding (up to 128 KiB each) on the GPU right before it runs and right after, into a 128 MiB
// host ring, with its first 16 user SGPRs. Written to <markers file>.cap at the loss, and every
// KYTY_GPU_MARKERS_CAP_PERIOD seconds (default 0: never) to <markers file>.cap<n> (at most 6).
void SetUserData(const uint32_t* sgprs) noexcept;

// A command-processor write of guest memory (DMA fill/copy, WRITE_DATA): kind 'F'/'C'/'W' on the CPU
// at record time, 'f'/'c' through the command stream. Kept in a ring, printed against the hung reads.
void NoteCpWrite(uint64_t vaddr, uint64_t size, char kind) noexcept;

// type: 'C' dispatch, 'I' indirect dispatch, 'D' draw, 'X' indirect draw. Returns 0 when inactive.
uint32_t Begin(vk::CommandBuffer cmd, char type, uint64_t shader, uint64_t shader2, uint32_t x,
               uint32_t y, uint32_t z, uint64_t tick, uint32_t flags = 0) noexcept;
void     End(vk::CommandBuffer cmd, uint32_t seq) noexcept;

// PERF gpumark: recorded, begun and finished sequence numbers (positive control).
void Report(double seconds);
// On the first device loss (ReportDeviceLost).
void Dump(const char* site) noexcept;

} // namespace Libs::Graphics::GpuMarkers

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUMARKERS_H_
