#include "gpu_blit_shaders/gpu_mesh_indirect_spv.h"

#include <spirv/unified1/spirv.hpp>

#include <cstdint>
#include <cstdio>
#include <iterator>

namespace {

// At SPIR-V 1.6 glslang declares a task shader's workgroup size with LocalSizeId, which needs
// maintenance4: the build targets SPIR-V 1.5, which keeps LocalSize.
bool MeshIndirectTaskDeclaresLocalSize() {
	bool local_size    = false;
	bool local_size_id = false;
	for (size_t i = 5; i < std::size(GPU_MESH_INDIRECT_SPV);) {
		const uint32_t word  = GPU_MESH_INDIRECT_SPV[i];
		const uint32_t count = word >> spv::WordCountShift;
		if (count == 0 || i + count > std::size(GPU_MESH_INDIRECT_SPV)) {
			std::printf("gpu_mesh_indirect: malformed instruction at word %zu\n", i);
			return false;
		}
		const auto op = static_cast<spv::Op>(word & spv::OpCodeMask);
		if (count > 2 && (op == spv::OpExecutionMode || op == spv::OpExecutionModeId)) {
			const auto mode = static_cast<spv::ExecutionMode>(GPU_MESH_INDIRECT_SPV[i + 2]);
			local_size |= mode == spv::ExecutionModeLocalSize;
			local_size_id |= mode == spv::ExecutionModeLocalSizeId;
		}
		i += count;
	}
	if (!local_size || local_size_id) {
		std::printf("gpu_mesh_indirect: LocalSize=%d LocalSizeId=%d\n", local_size, local_size_id);
		return false;
	}
	return true;
}

} // namespace

int main() {
	if (!MeshIndirectTaskDeclaresLocalSize()) {
		std::printf("FAIL GpuBlitShaders.MeshIndirectTaskDeclaresLocalSize\n");
		return 1;
	}
	std::printf("PASS GpuBlitShaders.MeshIndirectTaskDeclaresLocalSize\n");
	return 0;
}
