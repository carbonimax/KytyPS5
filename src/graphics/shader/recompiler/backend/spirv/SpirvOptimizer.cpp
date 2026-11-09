#include "graphics/shader/recompiler/backend/spirv/SpirvOptimizer.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <chrono>
#include <cstdlib>

// The idea of running SPIRV-Tools on each recompiled module comes from AnyPS5's SpirvOptimizer.
// Of the -O recipe, only value-numbering CSE shrinks our modules (the emitter already writes SSA
// without dead code or local variables), and inlining grows them: the list is CSE and the
// clean-up after it. Nothing folds or reorders float math.

namespace Libs::Graphics::ShaderRecompiler::Spirv {

uint32_t SpirvOptLevelFromEnvironment() {
	static const uint32_t level = [] {
		const char* value = std::getenv("KYTY_SPIRV_OPT");
		if (value == nullptr || *value == '\0') {
			return 0u;
		}
		const auto parsed = std::strtoul(value, nullptr, 10);
		return parsed > 2 ? 2u : static_cast<uint32_t>(parsed);
	}();
	return level;
}

SpirvOptStats& GetSpirvOptStats() {
	static SpirvOptStats stats;
	return stats;
}

bool OptimizeSpirv(std::vector<uint32_t>& spirv, uint32_t level, bool allow_offset_texture_operand) {
	using namespace spvtools;
	const auto begin = std::chrono::steady_clock::now();
	auto&      stats = GetSpirvOptStats();
	Optimizer  optimizer(SPV_ENV_VULKAN_1_3);
	optimizer.RegisterPass(CreateRedundancyEliminationPass())
	    .RegisterPass(CreateDeadBranchElimPass())
	    .RegisterPass(CreateBlockMergePass())
	    .RegisterPass(CreateVectorDCEPass())
	    .RegisterPass(CreateDeadInsertElimPass())
	    .RegisterPass(CreateAggressiveDCEPass())
	    .RegisterPass(CreateCFGCleanupPass());
	OptimizerOptions options;
	options.set_run_validator(false);
	options.set_preserve_bindings(true);
	options.set_preserve_spec_constants(true);
	std::vector<uint32_t> result;
	bool                  ok = optimizer.Run(spirv.data(), spirv.size(), &result, options);
	if (!ok) {
		stats.optimizer_failures++;
	} else if (level >= 2) {
		SpirvTools       tools(SPV_ENV_VULKAN_1_3);
		ValidatorOptions validator;
		validator.SetAllowOffsetTextureOperand(allow_offset_texture_operand);
		ok = tools.Validate(result.data(), result.size(), validator);
		if (!ok) {
			stats.validation_failures++;
		}
	}
	stats.modules++;
	stats.words_in += spirv.size();
	if (ok) {
		spirv = std::move(result);
	}
	stats.words_out += spirv.size();
	stats.optimize_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                               std::chrono::steady_clock::now() - begin)
	                                               .count());
	return ok;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
