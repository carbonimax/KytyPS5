#include "graphics/host_gpu/renderer/pipeline/asyncStubPolicy.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"

#include <utility>

namespace Libs::Graphics {

std::vector<uint32_t> BuildAsyncStubPixel(std::span<const vk::Format> formats) {
	using ShaderRecompiler::Spirv::Builder;
	Builder builder(0x00010000u);
	builder.RequireCapability(spv::CapabilityShader);
	builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	const auto void_type = builder.Type(spv::OpTypeVoid);
	const auto fn_type   = builder.Type(spv::OpTypeFunction, void_type);
	const auto f32       = builder.Type(spv::OpTypeFloat, 32u);
	const auto u32       = builder.Type(spv::OpTypeInt, 32u, 0u);
	const auto vec4      = builder.Type(spv::OpTypeVector, f32, 4u);
	const auto uvec4     = builder.Type(spv::OpTypeVector, u32, 4u);

	std::vector<uint32_t>                      interfaces;
	std::vector<std::pair<uint32_t, uint32_t>> stores; // variable, constant
	for (uint32_t slot = 0; slot < formats.size(); slot++) {
		if (formats[slot] == vk::Format::eUndefined) continue;
		const auto value = AsyncStubTargetValue(formats[slot]);
		if (!value) return {};
		const auto scalar = value->uint_type ? u32 : f32;
		const auto vector = value->uint_type ? uvec4 : vec4;
		std::array<uint32_t, 4> components {};
		for (size_t i = 0; i < 4; i++) {
			components[i] = builder.Constant(spv::OpConstant, scalar, value->bits[i]);
		}
		const auto constant = builder.Constant(spv::OpConstantComposite, vector,
		                                       std::span<const uint32_t>(components));
		const auto variable = builder.DefineGlobalVariable(
		    builder.Type(spv::OpTypePointer, spv::StorageClassOutput, vector), spv::StorageClassOutput);
		builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationLocation, slot);
		interfaces.push_back(variable);
		stores.emplace_back(variable, constant);
	}
	const auto main = builder.AllocateId();
	builder.AddEntryPoint(spv::ExecutionModelFragment, main, "main", interfaces);
	builder.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
	builder.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone, fn_type);
	builder.AddFunction(spv::OpLabel, builder.AllocateId());
	for (const auto& [variable, constant]: stores) {
		builder.AddFunction(spv::OpStore, variable, constant);
	}
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	return builder.Build();
}

} // namespace Libs::Graphics
