#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_

#include "graphics/shader/shader.h"

#include <array>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
} // namespace HW

struct ShaderParams {
	std::span<const uint32_t> code;
	std::array<uint32_t, 40>  user_data {}; // 32 user SGPRs plus the merged-stage s0:s7 prefix.
	uint32_t                  user_data_count = 0;
	uint64_t                  hash = 0;
	std::span<const uint32_t> back_code;

	[[nodiscard]] uint64_t Base() const {
		return reinterpret_cast<uint64_t>(code.data());
	}
};

void BuildStageStaticKey(const ShaderVertexInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderPixelInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderComputeInputInfo& input_info, std::vector<uint32_t>& key);
// The part of the static key that ShaderRecompiler::TranslateProgram reads (the rest is read by
// the SPIR-V emission only): programs with equal code and translation keys translate identically.
void BuildStageTranslationKey(const ShaderVertexInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageTranslationKey(const ShaderPixelInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageTranslationKey(const ShaderComputeInputInfo& input_info, std::vector<uint32_t>& key);
// Program-level inputs of the translation, appended after BuildStageTranslationKey.
void AppendProgramTranslationKey(std::vector<uint32_t>& key, ShaderType stage,
                                 uint32_t user_data_count, size_t code_words, size_t back_code_words,
                                 uint32_t wave_size, uint32_t user_data_base);
// The translation inputs of a registered pixel shader predicted from the register defaults of its
// binary (KYTY_SHADER_PRETRANSLATE). False when they are missing or unsupported.
bool ShaderPredictPixelProgram(const ShaderMappedData& data, ShaderPixelInputInfo& info,
                               uint32_t& user_data_count);

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& input_info);
std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info);
ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               input_info);
ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_ */
