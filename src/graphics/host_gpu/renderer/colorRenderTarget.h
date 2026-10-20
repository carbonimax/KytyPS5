#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>

namespace Libs::Graphics {

struct PreparedBindings;
struct RenderDepthInfo;

struct RenderColorInfo {
	// Discovery keeps guest image information but can remap the view into a larger cache image.
	TextureCache::ImageDesc         desc;
	ImageId                         image_id;
	uint32_t                        target_slot      = 0;
	uint32_t                        guest_mip_level   = 0;
	uint32_t                        guest_array_layer = 0;
	Prospero::ColorComponentMapping export_mapping;
	// Other aliased target slots whose components this attachment also carries.
	uint32_t merged_slots = 0;
	// Aliased target slots that draw one pass each on this image.
	uint32_t pass_slots = 0;

	[[nodiscard]] vk::Extent2D Extent() const {
		return {std::max(desc.info.extent.width >> guest_mip_level, 1u),
		        std::max(desc.info.extent.height >> guest_mip_level, 1u)};
	}
};

// Per-target passes repeat every shader stage of a draw on one attachment. Returns why that would
// change the result of the original single draw (V1-V4), or an empty string.
std::string CheckAliasedColorPasses(std::span<const RenderColorInfo>   colors,
                                    const RenderDepthInfo&             depth,
                                    std::span<PreparedBindings* const> stages,
                                    std::span<const GuestRange>        fixed_reads);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
