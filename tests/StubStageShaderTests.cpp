#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"
#include "graphics/shader/stubStageShader.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <map>
#include <vector>

namespace {

using Libs::Graphics::ShaderRecompiler::Spirv::Builder;

int g_failures = 0;

void Check(bool ok, const char* what) {
	std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
	g_failures += ok ? 0 : 1;
}

// A pixel shader shaped like the translator's: vec4 parameters (one flat), a builtin input, and
// float / uint color outputs (location 1 with a component offset on a scalar).
std::vector<uint32_t> PixelShader(bool with_struct_input) {
	Builder b(0x00010500u);
	b.RequireCapability(spv::CapabilityShader);
	b.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	const auto void_type = b.Type(spv::OpTypeVoid);
	const auto fn        = b.Type(spv::OpTypeFunction, void_type);
	const auto f32       = b.Type(spv::OpTypeFloat, 32u);
	const auto u32       = b.Type(spv::OpTypeInt, 32u, 0u);
	const auto vec4      = b.Type(spv::OpTypeVector, f32, 4u);
	const auto uvec4     = b.Type(spv::OpTypeVector, u32, 4u);
	std::vector<uint32_t> interfaces;
	const auto variable  = [&](spv::StorageClass storage, uint32_t type) {
        const auto v = b.DefineGlobalVariable(b.Type(spv::OpTypePointer, storage, type), storage);
        interfaces.push_back(v);
        return v;
	};
	const auto in0 = variable(spv::StorageClassInput, vec4);
	b.AddAnnotation(spv::OpDecorate, in0, spv::DecorationLocation, 0u);
	const auto in3 = variable(spv::StorageClassInput, vec4);
	b.AddAnnotation(spv::OpDecorate, in3, spv::DecorationFlat);
	b.AddAnnotation(spv::OpDecorate, in3, spv::DecorationLocation, 3u);
	const auto coord = variable(spv::StorageClassInput, vec4);
	b.AddAnnotation(spv::OpDecorate, coord, spv::DecorationBuiltIn, spv::BuiltInFragCoord);
	if (with_struct_input) {
		const auto block = variable(spv::StorageClassInput, b.Type(spv::OpTypeStruct, vec4));
		b.AddAnnotation(spv::OpDecorate, block, spv::DecorationLocation, 5u);
	}
	const auto out0 = variable(spv::StorageClassOutput, vec4);
	b.AddAnnotation(spv::OpDecorate, out0, spv::DecorationLocation, 0u);
	const auto out1 = variable(spv::StorageClassOutput, uvec4);
	b.AddAnnotation(spv::OpDecorate, out1, spv::DecorationLocation, 1u);
	const auto value = b.Constant(spv::OpConstantNull, vec4);
	const auto main  = b.AllocateId();
	b.AddEntryPoint(spv::ExecutionModelFragment, main, "main", interfaces);
	b.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
	b.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone, fn);
	b.AddFunction(spv::OpLabel, b.AllocateId());
	b.AddFunction(spv::OpStore, out0, value);
	b.AddFunction(spv::OpReturn);
	b.AddFunction(spv::OpFunctionEnd);
	return b.Build();
}

bool Valid(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		std::printf("  spirv-val: %s\n", message);
	});
	return !spirv.empty() && tools.Validate(spirv.data(), spirv.size(), spvtools::ValidatorOptions {});
}

// Execution model of the entry point, and location -> storage class of the Location variables.
struct Interface {
	uint32_t                     model = ~0u;
	std::map<uint32_t, uint32_t> locations;
	int                          positions = 0;
};

Interface Read(const std::vector<uint32_t>& spirv) {
	Interface                    out;
	std::map<uint32_t, uint32_t> location_of, storage_of;
	for (size_t at = 5; at < spirv.size(); at += spirv[at] >> 16u) {
		const auto op = spirv[at] & 0xffffu;
		if (op == spv::OpEntryPoint) out.model = spirv[at + 1];
		if (op == spv::OpDecorate && spirv[at + 2] == spv::DecorationLocation) location_of[spirv[at + 1]] = spirv[at + 3];
		if (op == spv::OpDecorate && spirv[at + 2] == spv::DecorationBuiltIn && spirv[at + 3] == spv::BuiltInPosition)
			out.positions++;
		if (op == spv::OpVariable) storage_of[spirv[at + 2]] = spirv[at + 3];
	}
	for (const auto& [id, location]: location_of) out.locations[location] = storage_of[id];
	return out;
}

} // namespace

int main() {
	const auto pixel = PixelShader(false);
	Check(Valid(pixel), "StubStageShader.SourceIsValid");

	const auto vs = Libs::Graphics::BuildStubVertexForPixel(pixel);
	const auto vi = Read(vs);
	Check(Valid(vs), "StubStageShader.VertexForPixelIsValid");
	Check(vi.model == spv::ExecutionModelVertex && vi.positions == 1 &&
	          vi.locations == std::map<uint32_t, uint32_t> {{0u, spv::StorageClassOutput}, {3u, spv::StorageClassOutput}},
	      "StubStageShader.VertexForPixelWritesThePixelInputs");

	const auto like = Libs::Graphics::BuildStubVertexLike(vs);
	Check(Valid(like) && Read(like).locations == vi.locations, "StubStageShader.VertexLikeCopiesOutputs");

	const auto ps = Libs::Graphics::BuildStubPixelLike(pixel);
	const auto pi = Read(ps);
	Check(Valid(ps), "StubStageShader.PixelLikeIsValid");
	Check(pi.model == spv::ExecutionModelFragment && pi.positions == 0 &&
	          pi.locations == std::map<uint32_t, uint32_t> {{0u, spv::StorageClassOutput}, {1u, spv::StorageClassOutput}},
	      "StubStageShader.PixelLikeWritesThePixelOutputs");

	Check(Libs::Graphics::BuildStubVertexForPixel(PixelShader(true)).empty(),
	      "StubStageShader.StructInputIsUnsupported");
	return g_failures == 0 ? 0 : 1;
}
