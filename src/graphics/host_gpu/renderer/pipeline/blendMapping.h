#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_

#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics {

enum class BlendMappingSupport {
	Direct,
	SourceAlpha, // Requires logical alpha in the second blend source.
	SourceAlphaOne, // Source alpha for RGB, one for alpha, with a shared destination equation.
	SourceAlphaZero, // Source alpha for RGB, zero for alpha, with a shared destination equation.
	Unsupported,
};

bool                BlendFactorIsDualSource(uint8_t factor);
BlendMappingSupport ClassifyBlendMapping(const HW::BlendControl&                blend,
                                         const Prospero::ColorComponentMapping& mapping);

// Color targets bound to identical CB registers share one attachment at the lowest slot of
// their group. A target with a nonzero component_mask is merged.
struct AliasedColorTargets {
	uint8_t slot[8]           = {0, 1, 2, 3, 4, 5, 6, 7};
	uint8_t component_mask[8] = {}; // physical components each merged export stores
	uint8_t write_mask[8]     = {}; // physical write mask of each shared attachment
	// The shared attachment 0 takes each component's source alpha from the second blend source.
	bool             alpha_source    = false;
	HW::BlendControl blend[8]        = {};
	bool             blend_enable[8] = {};

	[[nodiscard]] bool Active() const;
};

// Plans the merge for active_mask slots. Returns an empty string, or why the per-target blending
// of an aliased group has no exact single-attachment form (EQUIVALENCE.md E1-E10).
std::string PlanAliasedColorTargets(const HW::Context& ctx, uint32_t active_mask,
                                    AliasedColorTargets& plan);

// When aliased targets cannot share an attachment, each can still draw in its own pass, in slot
// order, on the same attachment. Returns an empty string and the group of slots, or why the
// separate passes would change the result (P1-P4).
std::string PlanAliasedColorPasses(const HW::Context& ctx, uint32_t active_mask, uint32_t& group);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_
