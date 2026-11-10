#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_STUBSTAGESHADER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_STUBSTAGESHADER_H_

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Stub stages that pair with one real stage so the driver compiles that stage alone. Each writes
// zero to its outputs and does nothing else; capabilities and extensions are copied from the
// source module. Empty when the source has an interface type the stub does not copy.

// A vertex shader writing the position and every Location input of `pixel`.
std::vector<uint32_t> BuildStubVertexForPixel(std::span<const uint32_t> pixel);
// A vertex shader writing the position and every Location output of `vertex`.
std::vector<uint32_t> BuildStubVertexLike(std::span<const uint32_t> vertex);
// A pixel shader with no inputs writing every Location output of `pixel`.
std::vector<uint32_t> BuildStubPixelLike(std::span<const uint32_t> pixel);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_STUBSTAGESHADER_H_ */
