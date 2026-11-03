#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info, ShaderHostFeatures host_features) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	state.resource_min_lod = host_features.resource_min_lod;
	if (!host_features.buffer_int64_atomics && program.info.buffer_int64_atomics) {
		Fail(program,
		     "shader requires shaderBufferInt64Atomics, which is not enabled on the host GPU");
	}
	if (!host_features.cull_distance &&
	    std::ranges::any_of(program.info.outputs, [](const auto& output) {
		    return output.kind == IR::StageOutputKind::CullDistance;
	    })) {
		Fail(program, "shader requires shaderCullDistance, which is not enabled on the host GPU");
	}
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	// A wave32 pixel shader cannot require a 32-lane subgroup on drivers without fragment
	// subgroup size control (amdvlk: compute only). Packing is also correct in a 32-lane subgroup.
	state.packed_wave32 =
	    (workgroup != nullptr && program.wave_size == 32u && workgroup->host_subgroup_size == 64u) ||
	    (program.stage == ShaderType::Pixel && program.wave_size == 32u &&
	     Emitter::PackedPixelWave32Enabled()) ||
	    (workgroup == nullptr && program.wave_size == 32u &&
	     (program.stage == ShaderType::Vertex || program.stage == ShaderType::Local ||
	      program.stage == ShaderType::TessellationEvaluation) &&
	     host_features.subgroup_size == 64u && Emitter::PackedVertexWave32Enabled());
	state.emulated_lane_id =
	    !host_features.vertex_subgroups &&
	    (program.stage == ShaderType::Vertex || program.stage == ShaderType::Local ||
	     program.stage == ShaderType::TessellationEvaluation);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
