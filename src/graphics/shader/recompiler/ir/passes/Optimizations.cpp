#include "graphics/shader/recompiler/ir/passes/Optimizations.h"

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool EnvSwitch(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return std::strcmp(value, "0") != 0;
}

} // namespace

uint32_t ShaderOptimizationsFromEnvironment() {
	static const uint32_t mask = [] {
		if (!EnvSwitch("KYTY_SHADER_OPT", true)) {
			return 0u;
		}
		uint32_t result = 0;
		if (EnvSwitch("KYTY_SHADER_OPT_LANES", true)) result |= ShaderOptLaunchedLanes;
		return result;
	}();
	return mask;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
