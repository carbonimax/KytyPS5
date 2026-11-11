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
void NoteBinding(uint64_t vaddr, uint64_t size, bool written, int kind) noexcept;

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
