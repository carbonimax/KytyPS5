#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <cstdlib>
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

// KYTY_TESS_HS_MODEL=1: HS control point outputs and the matching TES inputs as one arrayed
// variable per location. With one array-of-arrays variable of 12 locations per control point,
// the AMD Windows driver stores control points 25 dwords apart, so they overlap.
bool SplitControlPoints(IR::TessellationAttribute kind) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TESS_HS_MODEL");
		return value != nullptr && value[0] == '1';
	}();
	return enabled && (kind == IR::TessellationAttribute::ControlOutput ||
	                   kind == IR::TessellationAttribute::EvaluationInput);
}

// The byte offset within a control point when the address is c or c + stride * k * x.
std::optional<uint32_t> ConstantControlPointOffset(IR::Value address, uint32_t stride) {
	address = address.Resolve();
	if (address.IsImmediate()) return address.U32() % stride;
	const auto* sum = address.TryInstruction();
	if (sum == nullptr || sum->GetOpcode() != IR::ValueOpcode::IAdd32) return std::nullopt;
	for (size_t index = 0; index < 2u; index++) {
		const auto constant = sum->Arg(index).Resolve();
		const auto* product = sum->Arg(1u - index).Resolve().TryInstruction();
		if (!constant.IsImmediate() || product == nullptr ||
		    product->GetOpcode() != IR::ValueOpcode::IMul32)
			continue;
		for (size_t factor = 0; factor < 2u; factor++) {
			const auto value = product->Arg(factor).Resolve();
			if (value.IsImmediate() && value.U32() % stride == 0u) return constant.U32() % stride;
		}
	}
	return std::nullopt;
}

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
	const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
	const auto stride = local ? tess.ls_stride : tess.hs_stride;
	const auto offset =
	    patch ? address : EmitBinaryU32(state, spv::OpUMod, address, ConstantU32(state, stride));
	const auto attribute =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 4));
	const auto component =
	    EmitBinaryU32(state, spv::OpBitwiseAnd,
	                  EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 2)),
	                  ConstantU32(state, 3));
	const auto type = TypePointer(state, storage, TypeU32(state));
	if (kind == Attribute::LocalOutput || patch) {
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute,
		                          component);
	} else {
		// An HS writes only its own control point but may read any other.
		const auto vertex =
		    kind == Attribute::ControlOutput &&
		            inst.GetOpcode() == IR::ValueOpcode::SetTessellationAttribute
		        ? EmitLaneId(state)
		        : EmitBinaryU32(state, spv::OpUDiv, address, ConstantU32(state, stride));
		if (location != UINT32_MAX) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer,
			                          state.tess_split_variables.at(location), vertex, component);
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, vertex,
			                          attribute, component);
		}
	}
	return pointer;
}

// Split control points: the location of a constant offset, or UINT32_MAX and the dynamic one.
uint32_t SplitLocation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t& dynamic) {
	const auto stride = ctx.state.input_info.vertex->tess.hs_stride;
	if (const auto offset = ConstantControlPointOffset(inst.Arg(1), stride)) return *offset / 16u;
	dynamic = EmitBinaryU32(
	    ctx.state, spv::OpShiftRightLogical,
	    EmitBinaryU32(ctx.state, spv::OpUMod, ctx.Arg(inst, 1), ConstantU32(ctx.state, stride)),
	    ConstantU32(ctx.state, 4));
	return UINT32_MAX;
}

uint32_t IsLocation(EmitterState& state, uint32_t dynamic, uint32_t location) {
	return Binary(state, spv::OpIEqual, TypeBool(state), dynamic, ConstantU32(state, location));
}

// KYTY_TESS_PROBE: records the word at the access's address. LS outputs repeat every LS stride.
void ProbeTessellationWord(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t base,
                           uint32_t last, uint32_t value, uint32_t first = 4u,
                           uint32_t modulus = 0u) {
	auto& state = ctx.state;
	if (state.tess_probe_won == 0) return;
	auto address = ctx.Arg(inst, 1);
	if (base != 0u) address = EmitBinaryU32(state, spv::OpISub, address, ConstantU32(state, base));
	if (state.program.stage == ShaderType::Local) {
		address = EmitBinaryU32(state, spv::OpUMod, address,
		                        ConstantU32(state, state.input_info.vertex->tess.ls_stride));
	}
	if (modulus != 0u)
		address = EmitBinaryU32(state, spv::OpUMod, address, ConstantU32(state, modulus));
	const auto word = EmitUMin32(
	    state, EmitBinaryU32(state, spv::OpShiftRightLogical, address, ConstantU32(state, 2)),
	    ConstantU32(state, last));
	EmitTessProbeStore(state, EmitAddU32(state, word, ConstantU32(state, first)), value);
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
			const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
			const auto stride = local ? tess.ls_stride : tess.hs_stride;
			type              = array(TypeU32Vector(state, 4), (stride + 15u) / 16u);
			if (kind != Attribute::LocalOutput) {
				type = array(type, local ? tess.input_control_points : tess.output_control_points);
			}
		}
		auto& variable = state.tess_variables[index];
		if (SplitControlPoints(kind)) {
			const auto element = array(TypeU32Vector(state, 4), tess.output_control_points);
			for (uint32_t location = 0; location < (tess.hs_stride + 15u) / 16u; location++) {
				const auto split = DefineInterfaceVariable(state, element, storage, "tess_points");
				state.builder.AddAnnotation(spv::OpDecorate, split, spv::DecorationLocation,
				                            location);
				state.tess_split_variables.push_back(split);
			}
			variable = state.tess_split_variables.front();
			continue;
		}
		variable = DefineInterfaceVariable(state, type, storage, "tess_attributes");
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
		auto& state = ctx.state;
		const auto kind = static_cast<IR::TessellationAttribute>(inst.Arg(0).U32());
		uint32_t   value;
		if (uint32_t dynamic = 0; SplitControlPoints(kind)) {
			const auto location = SplitLocation(ctx, inst, dynamic);
			if (location != UINT32_MAX) {
				value = Unary(state, spv::OpLoad, TypeU32(state),
				              TessellationPointer(ctx, inst, location));
			} else {
				value = ConstantU32(state, 0);
				for (uint32_t index = 0; index < state.tess_split_variables.size(); index++) {
					value = Select(state, TypeU32(state), IsLocation(state, dynamic, index),
					               Unary(state, spv::OpLoad, TypeU32(state),
					                     TessellationPointer(ctx, inst, index)),
					               value);
				}
			}
		} else {
			value = Unary(state, spv::OpLoad, TypeU32(state), TessellationPointer(ctx, inst));
		}
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::EvaluationInput))
			ProbeTessellationWord(ctx, inst, 0u, 191u, value);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::ControlInput))
			ProbeTessellationWord(ctx, inst, 0u, 143u, value);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::PatchInput))
			ProbeTessellationWord(ctx, inst, ctx.state.input_info.vertex->tess.patch_base, 23u,
			                      value, 196u);
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
		const auto kind = static_cast<IR::TessellationAttribute>(inst.Arg(0).U32());
		if (uint32_t dynamic = 0; SplitControlPoints(kind)) {
			const auto location = SplitLocation(ctx, inst, dynamic);
			if (location != UINT32_MAX) {
				ctx.state.builder.AddFunction(spv::OpStore,
				                              TessellationPointer(ctx, inst, location), value);
			} else {
				for (uint32_t index = 0; index < ctx.state.tess_split_variables.size(); index++) {
					EmitIfCondition(ctx.state, IsLocation(ctx.state, dynamic, index), [&] {
						ctx.state.builder.AddFunction(spv::OpStore,
						                              TessellationPointer(ctx, inst, index), value);
					});
				}
			}
		} else {
			ctx.state.builder.AddFunction(spv::OpStore, TessellationPointer(ctx, inst), value);
		}
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::LocalOutput))
			ProbeTessellationWord(ctx, inst, 0u, 31u, value);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::ControlOutput)) {
			auto& state = ctx.state;
			ProbeTessellationWord(ctx, inst, 0u, 47u, value, 160u,
			                      state.input_info.vertex->tess.hs_stride);
			// Word 148: writes in the high half, writes to another control point's block in the
			// low half; word 149: the last such address.
			if (state.tess_probe_won != 0) {
				const auto address = ctx.Arg(inst, 1);
				const auto other   = Binary(
                    state, spv::OpINotEqual, TypeBool(state),
                    EmitBinaryU32(state, spv::OpUDiv, address,
                                  ConstantU32(state, state.input_info.vertex->tess.hs_stride)),
                    EmitLaneId(state));
				EmitTessProbeAdd(state, 148u,
				                 Select(state, TypeU32(state), other, ConstantU32(state, 0x10001u),
				                        ConstantU32(state, 0x10000u)));
				EmitIfCondition(state, other,
				                [&] { EmitTessProbeStore(state, ConstantU32(state, 149u), address); });
			}
		}
	});
}

void EmitTessProbeOutputs(EmitterState& state) {
	const auto variable =
	    state.tess_variables[static_cast<uint32_t>(IR::TessellationAttribute::ControlOutput)];
	if (state.program.stage != ShaderType::TessellationControl || state.tess_probe_won == 0 ||
	    variable == 0)
		return;
	const auto words = std::min(state.input_info.vertex->tess.hs_stride / 4u, 48u);
	const auto type  = TypePointer(state, spv::StorageClassOutput, TypeU32(state));
	const auto point = EmitLaneId(state);
	for (uint32_t word = 0; word < words; word++) {
		const auto pointer = state.builder.AllocateId();
		if (SplitControlPoints(IR::TessellationAttribute::ControlOutput)) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer,
			                          state.tess_split_variables.at(word / 4u), point,
			                          ConstantU32(state, word % 4u));
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, point,
			                          ConstantU32(state, word / 4u), ConstantU32(state, word % 4u));
		}
		EmitTessProbeStore(state, ConstantU32(state, 208u + word),
		                   Unary(state, spv::OpLoad, TypeU32(state), pointer));
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
