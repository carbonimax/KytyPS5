#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCSTUBPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCSTUBPOLICY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

// KYTY_ASYNC_STUB_PS=1 (with KYTY_ASYNC_PIPELINE=1): a skipped draw whose pixel stage only writes
// its color targets is drawn with its real vertex stage and a stub pixel stage writing one
// constant per target format while its pipeline is created, for this many guest frames at most
// (the stub hides the hole longer than a skip may).
inline constexpr uint64_t ASYNC_STUB_SKIP_FRAMES = 300;

// The constant a stub pixel stage writes to a target of `format`; none for a format without a
// chosen value (the draw is then skipped, not stubbed). Swizzle-invariant on purpose: an export
// mapping may reorder the components.
struct AsyncStubValue {
	bool                    uint_type = false;
	std::array<uint32_t, 4> bits {}; // float bits, or the uint components
};

[[nodiscard]] constexpr std::optional<AsyncStubValue> AsyncStubTargetValue(vk::Format format) {
	constexpr uint32_t grey = 0x3e3851ecu; // 0.18f: a finite, non-negative HDR mid grey
	constexpr uint32_t one  = 0x3f800000u;
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32: return AsyncStubValue {false, {grey, grey, grey, one}};
		// Motion vectors: zero (expect temporal trails on a moving object).
		case vk::Format::eR16G16Sfloat: return AsyncStubValue {false, {0, 0, 0, 0}};
		// Normals: +Z, a unit vector both as xyz and as an octahedral xy (0, 0); never zero.
		case vk::Format::eR16G16B16A16Snorm: return AsyncStubValue {false, {0, 0, one, 0}};
		// An id of unknown meaning: 0, the value of a cleared target.
		case vk::Format::eR32Uint: return AsyncStubValue {true, {0, 0, 0, 0}};
		default: return std::nullopt;
	}
}

// What decides whether a draw may be stubbed (the 5.22 "safe" rule).
struct AsyncStubDraw {
	bool                        pixel_kill        = false;
	bool                        depth_export      = false;
	bool                        sample_mask       = false;
	bool                        alpha_to_coverage = false;
	bool                        merged_targets    = false; // aliased targets merged into one
	std::span<const vk::Format> formats;                   // per slot, eUndefined when unused
	std::span<const uint32_t>   color_mask;                // per slot
	std::span<const bool>       blend;                     // per slot
};

// True when a stub pixel stage leaves the draw's depth, coverage and blending as the real one
// would, and every written target has a stub value.
[[nodiscard]] constexpr bool AsyncStubEligible(const AsyncStubDraw& draw) {
	if (draw.pixel_kill || draw.depth_export || draw.sample_mask || draw.alpha_to_coverage ||
	    draw.merged_targets) {
		return false;
	}
	bool any = false;
	for (size_t slot = 0; slot < draw.formats.size(); slot++) {
		if (draw.formats[slot] == vk::Format::eUndefined) continue;
		if (!AsyncStubTargetValue(draw.formats[slot])) return false;
		if (draw.blend[slot] && draw.color_mask[slot] != 0) return false;
		any = true;
	}
	return any;
}

// A pixel stage with no inputs writing AsyncStubTargetValue to each slot's target (Location = slot,
// eUndefined slots left out); empty when a format has no value.
std::vector<uint32_t> BuildAsyncStubPixel(std::span<const vk::Format> formats);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCSTUBPOLICY_H_
