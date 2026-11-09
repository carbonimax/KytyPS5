#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVOPTIMIZER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVOPTIMIZER_H_

#include <atomic>
#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// KYTY_SPIRV_OPT: 0 (default) off; 1 runs SPIRV-Tools passes that keep float results (CSE, dead
// branches and dead code) on each emitted module; 2 also validates the result and keeps the input
// when it fails.
[[nodiscard]] uint32_t SpirvOptLevelFromEnvironment();

struct SpirvOptStats {
	std::atomic<uint64_t> modules {0};
	std::atomic<uint64_t> words_in {0};
	std::atomic<uint64_t> words_out {0};
	std::atomic<uint64_t> optimize_us {0};
	std::atomic<uint64_t> optimizer_failures {0};
	std::atomic<uint64_t> validation_failures {0};
};
SpirvOptStats& GetSpirvOptStats();

// Optimizes `spirv` in place at `level` (1 or 2, see above). Returns false and leaves `spirv`
// unchanged when the optimizer fails or, at level 2, when its result does not validate.
// `allow_offset_texture_operand`: VK_KHR_maintenance8, as for ValidateShaderSpirv.
bool OptimizeSpirv(std::vector<uint32_t>& spirv, uint32_t level, bool allow_offset_texture_operand);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVOPTIMIZER_H_ */
