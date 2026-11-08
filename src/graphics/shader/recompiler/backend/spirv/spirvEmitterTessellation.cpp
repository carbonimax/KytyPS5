#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool IsTessellationInput(IR::TessellationAttribute kind) {
	using Attribute = IR::TessellationAttribute;
	return kind == Attribute::ControlInput || kind == Attribute::EvaluationInput ||
	       kind == Attribute::PatchInput;
}

bool IsTessellationPatch(IR::TessellationAttribute kind) {
	using Attribute = IR::TessellationAttribute;
	return kind == Attribute::PatchOutput || kind == Attribute::PatchInput;
}

uint32_t ControlPointStride(const ShaderTessellationInputInfo& tess, IR::TessellationAttribute kind) {
	using Attribute = IR::TessellationAttribute;
	return kind == Attribute::LocalOutput || kind == Attribute::ControlInput ? tess.ls_stride
	                                                                          : tess.hs_stride;
}

// One variable per location for per-control-point interfaces (KYTY_TESS_HS_MODEL, default 1):
// with one uvec4[points][12] array of arrays, the AMD Windows driver stores the HS control
// points 25 dwords apart, so they overlap. 0 keeps the array (of arrays).
bool SplitLocations() {
	const char* value = std::getenv("KYTY_TESS_HS_MODEL");
	return value == nullptr || std::strcmp(value, "0") != 0;
}

// The address modulo the control point stride, when known at compile time.
// shortcut: ignores 32-bit wrap-around; ring addresses stay far below 2^32.
std::optional<uint32_t> ConstantOffset(IR::Value value, uint32_t stride) {
	using Opcode = IR::ValueOpcode;
	value        = value.Resolve();
	if (value.IsImmediate()) return value.U32() % stride;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return std::nullopt;
	const auto opcode = inst->GetOpcode();
	if (opcode != Opcode::IAdd32 && opcode != Opcode::IMul32 &&
	    opcode != Opcode::ShiftLeftLogical32)
		return std::nullopt;
	const auto lhs = ConstantOffset(inst->Arg(0), stride);
	std::optional<uint32_t> rhs;
	if (opcode == Opcode::ShiftLeftLogical32) {
		const auto shift = inst->Arg(1).Resolve();
		if (!shift.IsImmediate() || shift.U32() >= 32u) return std::nullopt;
		rhs = static_cast<uint32_t>((uint64_t {1} << shift.U32()) % stride);
	} else {
		rhs = ConstantOffset(inst->Arg(1), stride);
	}
	if (opcode != Opcode::IAdd32 && ((lhs && *lhs == 0u) || (rhs && *rhs == 0u))) return 0u;
	if (!lhs || !rhs) return std::nullopt;
	return static_cast<uint32_t>(opcode == Opcode::IAdd32 ? (uint64_t {*lhs} + *rhs) % stride
	                                                      : (uint64_t {*lhs} * *rhs) % stride);
}

// `location`: the split variable to address (UINT32_MAX for the array layout).
uint32_t TessellationPointer(ValueEmitContext& ctx, const IR::Inst& inst,
                             uint32_t location = UINT32_MAX) {
	auto& state          = ctx.state;
	using Attribute      = IR::TessellationAttribute;
	const auto  kind     = static_cast<Attribute>(inst.Arg(0).U32());
	const auto& tess     = state.input_info.vertex->tess;
	const auto  variable = state.tess_variables.at(static_cast<uint32_t>(kind));
	EXIT_IF(variable == 0);
	const bool patch = IsTessellationPatch(kind);
	const auto storage =
	    IsTessellationInput(kind) ? spv::StorageClassInput : spv::StorageClassOutput;
	const auto pointer = state.builder.AllocateId();
	if (kind == Attribute::Factor) {
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		const auto index       = inst.Arg(1).U32() / 4u;
		const bool quad        = tess.domain == 2u;
		const auto outer_count = quad ? 4u : 3u;
		const bool outer       = index < outer_count;
		EXIT_NOT_IMPLEMENTED(index >= (quad ? 6u : 4u));
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeF32(state)),
		                          pointer, outer ? variable : state.tess_inner_variable,
		                          ConstantU32(state, outer ? index : index - outer_count));
		return pointer;
	}
	auto address = ctx.Arg(inst, 1);
	if (patch) {
		address = EmitBinaryU32(state, spv::OpISub, address, ConstantU32(state, tess.patch_base));
	}
	const auto stride = ControlPointStride(tess, kind);
	const auto offset =
	    patch ? address : EmitBinaryU32(state, spv::OpUMod, address, ConstantU32(state, stride));
	const auto attribute =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 4));
	const auto component =
	    EmitBinaryU32(state, spv::OpBitwiseAnd,
	                  EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 2)),
	                  ConstantU32(state, 3));
	const auto type = TypePointer(state, storage, TypeU32(state));
	const auto split =
	    location != UINT32_MAX
	        ? state.tess_location_variables.at(static_cast<uint32_t>(kind)).at(location)
	        : 0u;
	if (kind == Attribute::LocalOutput || patch) {
		if (split != 0u) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, split, component);
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute,
			                          component);
		}
	} else {
		// An HS writes only its own control point but may read any other.
		const auto vertex =
		    kind == Attribute::ControlOutput &&
		            inst.GetOpcode() == IR::ValueOpcode::SetTessellationAttribute
		        ? EmitLaneId(state)
		        : EmitBinaryU32(state, spv::OpUDiv, address, ConstantU32(state, stride));
		if (split != 0u) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, split, vertex, component);
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, vertex,
			                          attribute, component);
		}
	}
	return pointer;
}

// Split layout: the location of a compile-time offset, else UINT32_MAX and the run-time
// location in `dynamic`.
uint32_t SplitLocation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t& dynamic) {
	auto&      state  = ctx.state;
	const auto stride = ControlPointStride(state.input_info.vertex->tess,
	                                       static_cast<IR::TessellationAttribute>(inst.Arg(0).U32()));
	if (const auto offset = ConstantOffset(inst.Arg(1), stride)) return *offset / 16u;
	dynamic = EmitBinaryU32(
	    state, spv::OpShiftRightLogical,
	    EmitBinaryU32(state, spv::OpUMod, ctx.Arg(inst, 1), ConstantU32(state, stride)),
	    ConstantU32(state, 4));
	return UINT32_MAX;
}

uint32_t IsLocation(EmitterState& state, uint32_t dynamic, uint32_t location) {
	return Binary(state, spv::OpIEqual, TypeBool(state), dynamic, ConstantU32(state, location));
}

} // namespace

void DefineTessellationInterfaces(EmitterState& state) {
	using Attribute = IR::TessellationAttribute;
	// Tessellation lowering emits attribute accesses only in these stages.
	const auto stage = state.program.stage;
	if (stage != ShaderType::Local && stage != ShaderType::TessellationControl &&
	    stage != ShaderType::TessellationEvaluation) {
		return;
	}
	std::array<bool, IR::NumTessellationAttributes> used {};
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::GetTessellationAttribute ||
			    inst.GetOpcode() == IR::ValueOpcode::SetTessellationAttribute)
				used.at(inst.Arg(0).U32()) = true;
		}
	}
	if (std::ranges::none_of(used, [](bool value) { return value; })) return;
	const auto& tess  = state.input_info.vertex->tess;
	const auto  array = [&](uint32_t type, uint32_t count) {
		return state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	};
	const bool split = SplitLocations();
	for (uint32_t index = 0; index < used.size(); index++) {
		if (!used[index]) continue;
		const auto kind  = static_cast<Attribute>(index);
		const bool patch = IsTessellationPatch(kind);
		const auto storage =
		    IsTessellationInput(kind) ? spv::StorageClassInput : spv::StorageClassOutput;
		uint32_t type;
		if (kind == Attribute::Factor) {
			type = array(TypeF32(state), 4u);
		} else if (patch) {
			// HS and TES share the reflected patch layout so their interfaces match.
			type = array(TypeU32Vector(state, 4), (tess.patch_size + 15u) / 16u);
		} else {
			const auto locations = (ControlPointStride(tess, kind) + 15u) / 16u;
			const auto points    = kind == Attribute::ControlInput ? tess.input_control_points
			                                                       : tess.output_control_points;
			if (split) {
				// uvec4 (LS) or uvec4[points] at each location.
				const auto element = kind == Attribute::LocalOutput
				                         ? TypeU32Vector(state, 4)
				                         : array(TypeU32Vector(state, 4), points);
				auto& variables = state.tess_location_variables[index];
				for (uint32_t location = 0; location < locations; location++) {
					variables.push_back(
					    DefineInterfaceVariable(state, element, storage, "tess_location"));
					state.builder.AddAnnotation(spv::OpDecorate, variables.back(),
					                            spv::DecorationLocation, location);
				}
				state.tess_variables[index] = variables.front();
				continue;
			}
			type = array(TypeU32Vector(state, 4), locations);
			if (kind != Attribute::LocalOutput) type = array(type, points);
		}
		auto& variable = state.tess_variables[index];
		variable       = DefineInterfaceVariable(state, type, storage, "tess_attributes");
		if (kind == Attribute::Factor) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
			                            spv::BuiltInTessLevelOuter);
			state.tess_inner_variable =
			    DefineInterfaceVariable(state, array(TypeF32(state), 2u), storage, "tess_inner");
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationPatch);
		} else {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationLocation,
			                            patch ? (tess.hs_stride + 15u) / 16u : 0u);
		}
		if (kind == Attribute::Factor || patch) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPatch);
		}
	}
}

void DefineTessellationExecutionModes(EmitterState& state) {
	const auto& tess = state.input_info.vertex->tess;
	// Triangle or quad domain, integer or fractional partitioning, triangle output.
	EXIT_NOT_IMPLEMENTED((tess.domain != 1u && tess.domain != 2u) ||
	                     (tess.partitioning != 0u && tess.partitioning != 2u && tess.partitioning != 3u) ||
	                     (tess.output_topology != 2u && tess.output_topology != 3u));
	state.builder.RequireCapability(spv::CapabilityTessellation);
	if (state.program.stage == ShaderType::TessellationControl) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               tess.output_control_points);
	} else {
		state.builder.AddExecutionMode(state.main_func, tess.domain == 1u
		                                                   ? spv::ExecutionModeTriangles
		                                                   : spv::ExecutionModeQuads);
		state.builder.AddExecutionMode(
		    state.main_func, tess.partitioning == 0u ? spv::ExecutionModeSpacingEqual
		                     : tess.partitioning == 2u ? spv::ExecutionModeSpacingFractionalOdd
		                                                : spv::ExecutionModeSpacingFractionalEven);
		state.builder.AddExecutionMode(state.main_func, tess.output_topology == 2u
		                                                   ? spv::ExecutionModeVertexOrderCw
		                                                   : spv::ExecutionModeVertexOrderCcw);
	}
}

uint32_t EmitGetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, 2), [&] {
		auto&       state = ctx.state;
		const auto  load  = [&](uint32_t location) {
            return Unary(state, spv::OpLoad, TypeU32(state),
                         TessellationPointer(ctx, inst, location));
		};
		const auto& split = state.tess_location_variables.at(inst.Arg(0).U32());
		if (split.empty()) return load(UINT32_MAX);
		uint32_t dynamic = 0;
		if (const auto location = SplitLocation(ctx, inst, dynamic); location != UINT32_MAX)
			return load(location);
		// The offset is below the stride, so exactly one location matches.
		auto value = load(0);
		for (uint32_t location = 1; location < split.size(); location++) {
			value = Select(state, TypeU32(state), IsLocation(state, dynamic, location),
			               load(location), value);
		}
		return value;
	});
}

void EmitSetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, 3), [&] {
		auto value = ctx.Arg(inst, 2);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::Factor)) {
			const auto floating = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpBitcast, TypeF32(ctx.state), floating, value);
			value = floating;
		}
		const auto store = [&](uint32_t location) {
			ctx.state.builder.AddFunction(spv::OpStore, TessellationPointer(ctx, inst, location),
			                              value);
		};
		const auto& split = ctx.state.tess_location_variables.at(inst.Arg(0).U32());
		uint32_t    dynamic  = 0;
		const auto  location = split.empty() ? UINT32_MAX : SplitLocation(ctx, inst, dynamic);
		if (split.empty() || location != UINT32_MAX) {
			store(location);
			return;
		}
		for (uint32_t index = 0; index < split.size(); index++) {
			EmitIfCondition(ctx.state, IsLocation(ctx.state, dynamic, index),
			                [&] { store(index); });
		}
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
