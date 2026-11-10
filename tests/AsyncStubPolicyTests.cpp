#include "graphics/host_gpu/renderer/pipeline/asyncStubPolicy.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::AsyncStubDraw;
using Libs::Graphics::AsyncStubEligible;
using Libs::Graphics::BuildAsyncStubPixel;
using F = vk::Format;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AsyncStubPolicyTests: failed: %s\n", message);
		std::abort();
	}
}

bool Valid(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		std::fprintf(stderr, "  spirv-val: %s\n", message);
	});
	return !spirv.empty() && tools.Validate(spirv.data(), spirv.size(), spvtools::ValidatorOptions {});
}

} // namespace

int main() {
	// The two G-buffer layouts of 5.22: HDR + motion, and HDR + motion + id + gap + normal.
	const F two[]   = {F::eB10G11R11UfloatPack32, F::eR16G16Sfloat};
	const F five[]  = {F::eB10G11R11UfloatPack32, F::eR16G16Sfloat, F::eR32Uint, F::eUndefined,
	                   F::eR16G16B16A16Snorm};
	const uint32_t mask[5]     = {0xf, 0xf, 0xf, 0, 0xf};
	const bool     no_blend[5] = {};
	AsyncStubDraw  draw {.formats = five, .color_mask = mask, .blend = no_blend};
	Check(AsyncStubEligible(draw), "the five-target layout was refused");
	draw.formats = two;
	Check(AsyncStubEligible(draw), "the two-target layout was refused");

	auto refused = [&](auto change, const char* message) {
		AsyncStubDraw d {.formats = five, .color_mask = mask, .blend = no_blend};
		change(d);
		Check(!AsyncStubEligible(d), message);
	};
	refused([](AsyncStubDraw& d) { d.pixel_kill = true; }, "a pixel kill was stubbed");
	refused([](AsyncStubDraw& d) { d.depth_export = true; }, "a depth export was stubbed");
	refused([](AsyncStubDraw& d) { d.sample_mask = true; }, "a sample mask export was stubbed");
	refused([](AsyncStubDraw& d) { d.alpha_to_coverage = true; }, "alpha to coverage was stubbed");
	refused([](AsyncStubDraw& d) { d.merged_targets = true; }, "merged targets were stubbed");
	const bool blend[5] = {false, true};
	refused([&](AsyncStubDraw& d) { d.blend = blend; }, "a blended target was stubbed");
	const F rgba8[] = {F::eR8G8B8A8Unorm};
	refused([&](AsyncStubDraw& d) { d.formats = rgba8; }, "a format without a stub value was stubbed");
	const F none[] = {F::eUndefined};
	refused([&](AsyncStubDraw& d) { d.formats = none; }, "a draw without color targets was stubbed");
	// Blending on a target the draw does not write does not change it.
	const uint32_t masked[2] = {0xf, 0};
	Check(AsyncStubEligible({.formats = two, .color_mask = masked, .blend = blend}),
	      "a blend on a masked target refused the draw");

	Check(Valid(BuildAsyncStubPixel(two)), "the two-target stub is not valid SPIR-V");
	Check(Valid(BuildAsyncStubPixel(five)), "the five-target stub (uint id, gap) is not valid SPIR-V");
	Check(BuildAsyncStubPixel(rgba8).empty(), "a stub was built for a format without a value");
	std::printf("AsyncStubPolicyTests: ok\n");
	return 0;
}
