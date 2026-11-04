#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Vertex shaders build their entry EXEC from ESVertCount (s3), which the translator sets to the
// number of launched lanes: popcount(Ballot(true)). The guest then tests its lane in the low
// ESVertCount bits, (~0 >> -count) bit lane. On a host whose vertex subgroup holds one guest wave64
// (64-lane subgroup, real subgroup builtins), every launched invocation is one of those vertices,
// so the test is true; this folds it, and the EXEC selects it guards, to their written values.
//
// The idea follows AnyPS5's starting-lane bit test (github.com/boykopovar/AnyPS5,
// core/shader/recompiler/Optimization/src/ConstantFolder.cpp @4f41f4c375, GPL-2.0), which folds the
// pixel-shader form; the vertex-count form here is ours.
// Returns the number of folded tests; run constant propagation and dead-code elimination after.
[[nodiscard]] uint32_t FoldLaunchedLaneTests(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
