#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <bit>
#include <fmt/format.h>
#include <initializer_list>

namespace Libs::Graphics {

bool BlendFactorIsDualSource(uint8_t factor) {
	return factor >= static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color) &&
	       factor <= static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
}

namespace {

bool BlendFactorIsConstantColor(uint8_t factor) {
	return factor == static_cast<uint8_t>(Prospero::BlendFactor::kConstantColor) ||
	       factor == static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusConstantColor);
}

} // namespace

BlendMappingSupport ClassifyBlendMapping(const HW::BlendControl&                blend,
                                         const Prospero::ColorComponentMapping& mapping) {
	// Color constants are not swizzled with the exports; scalar constant alpha is unaffected.
	if (!mapping.IsIdentity() && (BlendFactorIsConstantColor(blend.color_srcblend) ||
	                              BlendFactorIsConstantColor(blend.color_destblend))) {
		return BlendMappingSupport::Unsupported;
	}
	if (mapping.Map(3) == 3u) {
		return BlendMappingSupport::Direct;
	}
	if (blend.separate_alpha_blend &&
	    blend.color_srcblend == static_cast<uint8_t>(Prospero::BlendFactor::kSrcAlpha) &&
	    blend.color_destblend == static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrcAlpha) &&
	    blend.alpha_destblend == blend.color_destblend &&
	    blend.color_comb_fcn == static_cast<uint8_t>(Prospero::BlendOp::kAdd) &&
	    blend.alpha_comb_fcn == blend.color_comb_fcn) {
		switch (static_cast<Prospero::BlendFactor>(blend.alpha_srcblend)) {
			case Prospero::BlendFactor::kZero: return BlendMappingSupport::SourceAlphaZero;
			case Prospero::BlendFactor::kOne: return BlendMappingSupport::SourceAlphaOne;
			default: break;
		}
	}
	// Moving alpha requires the same equation for all channels.
	if (blend.separate_alpha_blend && (blend.alpha_srcblend != blend.color_srcblend ||
	                                   blend.alpha_destblend != blend.color_destblend ||
	                                   blend.alpha_comb_fcn != blend.color_comb_fcn)) {
		return BlendMappingSupport::Unsupported;
	}
	auto support = BlendMappingSupport::Direct;
	for (const auto factor: {blend.color_srcblend, blend.color_destblend}) {
		if (BlendFactorIsDualSource(factor)) {
			return BlendMappingSupport::Unsupported;
		}
		switch (static_cast<Prospero::BlendFactor>(factor)) {
			case Prospero::BlendFactor::kSrcAlpha:
			case Prospero::BlendFactor::kOneMinusSrcAlpha:
				support = BlendMappingSupport::SourceAlpha;
				break;
			case Prospero::BlendFactor::kDstAlpha:
			case Prospero::BlendFactor::kOneMinusDstAlpha:
			case Prospero::BlendFactor::kSrcAlphaSaturate: return BlendMappingSupport::Unsupported;
			default: break;
		}
	}
	return support;
}

bool AliasedColorTargets::Active() const {
	return std::ranges::any_of(component_mask, [](uint8_t mask) { return mask != 0; });
}

namespace {

struct BlendEquation {
	uint8_t src = 0;
	uint8_t dst = 0;
	uint8_t op  = 0;

	bool operator==(const BlendEquation&) const = default;

	[[nodiscard]] bool Uses(std::initializer_list<Prospero::BlendFactor> factors) const {
		return std::ranges::any_of(factors, [&](Prospero::BlendFactor factor) {
			return src == static_cast<uint8_t>(factor) || dst == static_cast<uint8_t>(factor);
		});
	}
};

BlendEquation ColorEquation(const HW::BlendControl& blend) {
	return {blend.color_srcblend, blend.color_destblend, blend.color_comb_fcn};
}

BlendEquation AlphaEquation(const HW::BlendControl& blend) {
	return blend.separate_alpha_blend
	           ? BlendEquation {blend.alpha_srcblend, blend.alpha_destblend, blend.alpha_comb_fcn}
	           : ColorEquation(blend);
}

bool UsesDualSource(const HW::BlendControl& blend) {
	return BlendFactorIsDualSource(blend.color_srcblend) ||
	       BlendFactorIsDualSource(blend.color_destblend) ||
	       (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
	                                       BlendFactorIsDualSource(blend.alpha_destblend)));
}

uint8_t TargetOutputMode(const HW::Context& ctx, uint32_t slot) {
	const auto& sh = ctx.GetShaderRegisters();
	for (uint32_t index = 0; index < 8u; index++) {
		if (ShaderPixelExportTarget(sh.m_cbShaderMask, index) == slot) return sh.target_output_mode[index];
	}
	return 0;
}

std::string DescribeGroup(const HW::Context& ctx, uint32_t group) {
	std::string text;
	for (uint32_t slot = 0; slot < 8u; slot++) {
		if ((group & (1u << slot)) == 0) {
			continue;
		}
		const auto& rt      = ctx.GetRenderTarget(slot);
		const auto& blend   = ctx.GetBlendControl(slot);
		const auto  mapping = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
		                                                   rt.info.channel_order)
		                          .export_mapping;
		const auto  logical = render_target_mask_slot(ctx.GetRenderTargetMask(), slot);
		text += fmt::format(
		    " [slot={} base=0x{:x} fmt={}/{}/{} samples={} mask={:x}/{:x} mode={} mapping=0x{:02x}"
		    " blend enable={} bypass={} color={}/{}/{} alpha={}/{}/{} separate={}]",
		    slot, rt.base.addr, static_cast<uint32_t>(rt.info.format),
		    static_cast<uint32_t>(rt.info.channel_type),
		    static_cast<uint32_t>(rt.info.channel_order), rt.attrib.num_samples, logical,
		    mapping.ApplyMask(logical), TargetOutputMode(ctx, slot),
		    mapping.packed, blend.enable ? 1 : 0, rt.info.blend_bypass ? 1 : 0,
		    blend.color_srcblend, blend.color_destblend, blend.color_comb_fcn, blend.alpha_srcblend,
		    blend.alpha_destblend, blend.alpha_comb_fcn, blend.separate_alpha_blend ? 1 : 0);
	}
	return text;
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::string PlanAliasedColorTargets(const HW::Context& ctx, uint32_t active_mask,
                                    AliasedColorTargets& plan) {
	using F          = Prospero::BlendFactor;
	plan             = {};
	uint32_t grouped = 0;
	for (uint32_t first = 0; first < 8u; first++) {
		if ((active_mask & (1u << first)) == 0 || (grouped & (1u << first)) != 0) {
			continue;
		}
		const auto& target = ctx.GetRenderTarget(first);
		uint32_t    group  = 1u << first;
		for (uint32_t slot = first + 1; slot < 8u; slot++) {
			if ((active_mask & (1u << slot)) != 0 && ctx.GetRenderTarget(slot) == target) {
				group |= 1u << slot;
			}
		}
		grouped |= group;
		if (std::popcount(group) < 2) {
			continue;
		}
		const auto fail = [&](const char* rule, const char* reason) {
			return fmt::format("{} {} (attachment slot {}):{}", rule, reason, first,
			                   DescribeGroup(ctx, group));
		};
		const auto mapping =
		    TextureGetRenderTargetFormat(target.info.format, target.info.channel_type,
		                                 target.info.channel_order)
		        .export_mapping;
		const auto& first_blend  = ctx.GetBlendControl(first);
		const bool  blend_enable = first_blend.enable && !target.info.blend_bypass;
		uint32_t    written      = 0;
		uint32_t    alpha_owner  = 8;
		uint32_t    rgb_owners   = 0;
		uint8_t     owned[8]     = {};
		for (uint32_t slot = first; slot < 8u; slot++) {
			if ((group & (1u << slot)) == 0) {
				continue;
			}
			const auto& blend = ctx.GetBlendControl(slot);
			owned[slot]       = static_cast<uint8_t>(
			    mapping.ApplyMask(render_target_mask_slot(ctx.GetRenderTargetMask(), slot)));
			if ((written & owned[slot]) != 0) {
				return fail("E1", "aliased targets write the same component");
			}
			written |= owned[slot];
			if (TargetOutputMode(ctx, slot) != TargetOutputMode(ctx, first)) {
				return fail("E2", "aliased targets use different export formats");
			}
			if ((blend.enable && !target.info.blend_bypass) != blend_enable) {
				return fail("E4", "aliased targets differ in blend enable");
			}
			if (blend_enable && UsesDualSource(blend)) {
				return fail("E10", "an aliased target already uses dual-source blending");
			}
			if (blend_enable &&
			    ClassifyBlendMapping(blend, mapping) != BlendMappingSupport::Direct) {
				return fail("E2", "the target format moves alpha or swizzles constant colors");
			}
			if ((owned[slot] & 0x8u) != 0) {
				alpha_owner = slot;
			}
			if ((owned[slot] & 0x7u) != 0) {
				rgb_owners |= 1u << slot;
			}
		}
		bool remap        = false;
		auto alpha_source = alpha_owner;
		if (blend_enable) {
			const auto color_slot =
			    rgb_owners != 0 ? static_cast<uint32_t>(std::countr_zero(rgb_owners)) : alpha_owner;
			const auto color = ColorEquation(ctx.GetBlendControl(color_slot));
			for (uint32_t slot = 0; slot < 8u; slot++) {
				if ((rgb_owners & (1u << slot)) != 0 &&
				    ColorEquation(ctx.GetBlendControl(slot)) != color) {
					return fail("E5", "RGB owners use different color equations");
				}
			}
			if (alpha_source == 8) {
				alpha_source = color_slot;
			}
			const bool cross = (rgb_owners & ~(1u << alpha_source)) != 0;
			if (cross && alpha_owner != 8 &&
			    color.Uses({F::kDstAlpha, F::kOneMinusDstAlpha, F::kSrcAlphaSaturate})) {
				return fail("E8", "an RGB owner reads destination alpha written by another target");
			}
			if (cross && color.Uses({F::kSrcAlphaSaturate})) {
				return fail("E7", "alpha saturation needs another target's source alpha");
			}
			remap = cross && color.Uses({F::kSrcAlpha, F::kOneMinusSrcAlpha});
			if (remap && (first != 0 || (active_mask & ~group) != 0)) {
				return fail("E6", "per-component source alpha needs attachment 0 alone");
			}
			auto& merged           = plan.blend[first];
			merged.color_srcblend  = color.src;
			merged.color_destblend = color.dst;
			merged.color_comb_fcn  = color.op;
			const auto alpha =
			    alpha_owner != 8 ? AlphaEquation(ctx.GetBlendControl(alpha_owner)) : color;
			merged.alpha_srcblend       = alpha.src;
			merged.alpha_destblend      = alpha.dst;
			merged.alpha_comb_fcn       = alpha.op;
			merged.separate_alpha_blend = true;
			merged.enable               = true;
		}
		for (uint32_t slot = first; slot < 8u; slot++) {
			if ((group & (1u << slot)) != 0) {
				plan.slot[slot]           = static_cast<uint8_t>(first);
				plan.component_mask[slot] = owned[slot];
			}
		}
		plan.write_mask[first]   = static_cast<uint8_t>(written);
		plan.blend_enable[first] = blend_enable;
		plan.alpha_source |= remap;
	}
	return {};
}

std::string PlanAliasedColorPasses(const HW::Context& ctx, uint32_t active_mask, uint32_t& group) {
	using F = Prospero::BlendFactor;
	group   = 0;
	for (uint32_t first = 0; first < 8u && group == 0; first++) {
		for (uint32_t slot = first + 1; slot < 8u; slot++) {
			if ((active_mask & (1u << first)) != 0 && (active_mask & (1u << slot)) != 0 &&
			    ctx.GetRenderTarget(slot) == ctx.GetRenderTarget(first)) {
				group |= (1u << first) | (1u << slot);
			}
		}
	}
	const auto fail = [&](const char* rule, const char* reason) {
		return fmt::format("{} {}:{}", rule, reason, DescribeGroup(ctx, active_mask));
	};
	if (group == 0 || group != active_mask) {
		return fail("P1", "every color target must belong to the one aliased group");
	}
	const auto& target = ctx.GetRenderTarget(static_cast<uint32_t>(std::countr_zero(group)));
	if (target.attrib.num_samples != 0 || target.attrib.num_fragments != 0) {
		return fail("P3", "per-target passes are limited to single-sample targets");
	}
	uint32_t written     = 0;
	uint32_t alpha_owner = 8;
	for (uint32_t slot = 0; slot < 8u; slot++) {
		if ((group & (1u << slot)) == 0) {
			continue;
		}
		const auto owned = render_target_mask_slot(ctx.GetRenderTargetMask(), slot);
		if ((written & owned) != 0) {
			return fail("P2", "aliased targets write the same component");
		}
		written |= owned;
		alpha_owner = (owned & 0x8u) != 0 ? slot : alpha_owner;
	}
	for (uint32_t slot = 0; slot < 8u; slot++) {
		if ((group & (1u << slot)) == 0) {
			continue;
		}
		const auto& blend = ctx.GetBlendControl(slot);
		if (!blend.enable || target.info.blend_bypass) {
			continue;
		}
		if (UsesDualSource(blend)) {
			return fail("P4", "a target blends with the second source of another export");
		}
		// Each pass sees the alpha written by earlier passes; the original targets have no order.
		if (alpha_owner != 8 && alpha_owner != slot &&
		    ColorEquation(blend).Uses({F::kDstAlpha, F::kOneMinusDstAlpha, F::kSrcAlphaSaturate})) {
			return fail("P4", "a target reads destination alpha written by another pass");
		}
	}
	return {};
}

} // namespace Libs::Graphics
