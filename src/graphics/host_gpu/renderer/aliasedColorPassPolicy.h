#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ALIASEDCOLORPASSPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ALIASEDCOLORPASSPOLICY_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics {

// Per-target passes run a stage once per aliased target, so a stage that writes memory or GDS
// would repeat that write. Address reads alone (uses_dma, e.g. the index fetch of every mesh
// shader) are repeated with the same result: no stage writes, and attachment writes do not reach
// the buffer cache between passes. Scratch, local FLAT and LDS stay private to one invocation;
// BDA fault bits are OR-ed, so a repeated fault sets the same bit. Returns the first write found,
// or an empty string.
inline std::string AliasedPassStageWrite(const ShaderRecompiler::IR::ShaderInfo& info, const ShaderRecompiler::IR::Bindings& bindings, bool has_address_writes) {
	if (has_address_writes) {
		return "address stores";
	}
	for (uint32_t i = 0; i < info.buffers.size(); i++) {
		if (info.buffers[i].written || info.buffers[i].atomic) {
			return "buffer " + std::to_string(i) + (info.buffers[i].atomic ? " atomics" : " writes");
		}
	}
	for (uint32_t i = 0; i < info.images.size(); i++) {
		if (info.images[i].written || info.images[i].atomic) {
			return "image " + std::to_string(i) + (info.images[i].atomic ? " atomics" : " writes");
		}
	}
	if (bindings.descriptor_counts[static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::Gds)] != 0) {
		return "GDS";
	}
	return {};
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ALIASEDCOLORPASSPOLICY_H_
