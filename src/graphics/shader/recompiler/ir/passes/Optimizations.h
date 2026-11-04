#pragma once

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Optional IR optimization passes (step 5.6). Each has its own switch so that an A/B can turn a
// single pass off: KYTY_SHADER_OPT_<NAME>=0/1 (default on), and KYTY_SHADER_OPT=0 turns them all
// off. The shader store keys its records on these switches.
enum ShaderOptimization : uint32_t {
	ShaderOptLaunchedLanes = 1u << 0u, // KYTY_SHADER_OPT_LANES
};

inline constexpr uint32_t AllShaderOptimizations = ShaderOptLaunchedLanes;

// Read once per process.
[[nodiscard]] uint32_t ShaderOptimizationsFromEnvironment();

} // namespace Libs::Graphics::ShaderRecompiler::IR
