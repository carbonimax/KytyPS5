#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

// KYTY_F16_ROUNDING (spirvEmitterAlu.cpp): 0 keeps the f32 fma rounded twice.
static bool F16RoundNearestEven() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_F16_ROUNDING");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

IR::F32 Translator::Float16Fma(IR::F32 a, IR::F32 b, IR::F32 c) {
	if (!F16RoundNearestEven()) {
		return IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a, b, c}));
	}
	// f16 operands: the product is exact in f32 and every nonzero sum is at least 2^-48. TwoSum
	// gives the exact error of the f32 sum, which then rounds to odd so that the later f16
	// rounding to nearest even rounds only once.
	const auto add = [&](IR::Value lhs, IR::Value rhs) {
		return IR::F32(ir.Emit(IR::ValueOpcode::FPAdd32, {lhs, rhs}));
	};
	const auto sub = [&](IR::Value lhs, IR::Value rhs) {
		return IR::F32(ir.Emit(IR::ValueOpcode::FPSub32, {lhs, rhs}));
	};
	const auto product   = IR::F32(ir.Emit(IR::ValueOpcode::FPMul32, {a, b}));
	const auto sum       = IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a, b, c})); // product + c
	const auto virtual_c = sub(sum, product);
	const auto error     = add(sub(product, sub(sum, virtual_c)), sub(c, virtual_c));
	// A NaN error (infinite or NaN sum) compares unordered and keeps the sum.
	const auto inexact =
	    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdNotEqual32, {error, IR::Value::F32(0.0f)}));
	const auto bits    = ir.BitCastU32(sum);
	const auto even    = ir.IEqual(ir.BitwiseAnd(bits, IR::U32(IR::Value(1u))), IR::U32(IR::Value(0u)));
	// The odd neighbour lies on the side of the error: one ulp away from zero when the error has
	// the sign of the sum, toward zero otherwise.
	const auto away    = ir.IEqual(ir.ShiftRightLogical(ir.BitwiseXor(bits, ir.BitCastU32(error)),
	                                                    IR::U32(IR::Value(31u))),
	                               IR::U32(IR::Value(0u)));
	const auto odd     = ir.Select(away, ir.IAdd(bits, IR::U32(IR::Value(1u))),
	                               ir.ISub(bits, IR::U32(IR::Value(1u))));
	return ir.BitCastF32(ir.Select(ir.LogicalAnd(inexact, even), odd, bits));
}

void Translator::PackedFloat16(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool accumulator, bool quiet_snan) {
	const auto translate_lane = [&](bool high) {
		const auto lhs = ReadF16LaneAsF32(inst.src0, high);
		const auto rhs = ReadF16LaneAsF32(inst.src1, high);
		IR::F32    result;
		if (accumulator || inst.src_count == 3u) {
			const auto addend = ReadF16LaneAsF32(accumulator ? inst.dst : inst.src2, high);
			result = opcode == IR::ValueOpcode::FPFma32 ? Float16Fma(lhs, rhs, addend)
			                                            : IR::F32(ir.Emit(opcode, {lhs, rhs, addend}));
		} else {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs}));
		}
		return ApplyF32ResultModifiers(inst.dst, result);
	};
	auto raw    = DestinationOperand(inst);
	raw.omod    = 0u;
	raw.clamp   = false;
	auto result = PackHalf2x16(translate_lane(false), translate_lane(true));
	if (quiet_snan) {
		const auto quiet_snan_lane = [&](const Decoder::Operand& operand, bool high) {
			const auto bits     = ReadU16LaneAsU32(operand, high, false);
			const auto exponent = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x7e00u)));
			const auto payload  = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x01ffu)));
			const auto snan     = ir.LogicalAnd(ir.IEqual(exponent, IR::U32(IR::Value(0x7c00u))),
			                                    ir.INotEqual(payload, IR::U32(IR::Value(0u))));
			return std::pair {snan, ir.BitwiseOr(bits, IR::U32(IR::Value(0x0200u)))};
		};
		const auto override_lane = [&](bool high) {
			const auto [lhs_snan, lhs_quiet] = quiet_snan_lane(inst.src0, high);
			const auto [rhs_snan, rhs_quiet] = quiet_snan_lane(inst.src1, high);
			const auto normal = high ? ir.ShiftRightLogical(result, IR::U32(IR::Value(16u)))
			                         : ir.BitwiseAnd(result, IR::U32(IR::Value(0xffffu)));
			return ir.Select(lhs_snan, lhs_quiet, ir.Select(rhs_snan, rhs_quiet, normal));
		};
		result = PackU16Lanes(override_lane(false), override_lane(true));
	}
	WriteOperand(raw, result);
}

void Translator::Float16Unary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool invalid_negative) {
	const auto argument = ReadF16AsF32(inst.src0);
	auto       result   = IR::F32(ir.Emit(opcode, {argument}));
	if (!invalid_negative) {
		WriteF16(DestinationOperand(inst), result);
		return;
	}
	const auto negative =
	    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {argument, IR::Value::F32(0.0f)}));
	result             = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(negative, invalid, bits));
}

void Translator::Float16Trig(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto argument = ReadF16AsF32(inst.src0);
	const auto magnitude =
	    ir.BitwiseAnd(ir.BitCastU32(argument), IR::U32(IR::Value(0x7fffffffu)));
	auto result = IR::F32(ir.Emit(opcode, {argument}));
	if (opcode == IR::ValueOpcode::FPSin) {
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto whole = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.0f)}));
		const auto half = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.5f)}));
		const auto nonzero = ir.INotEqual(magnitude, IR::U32(IR::Value(0u)));
		const auto cardinal = ir.LogicalOr(ir.LogicalAnd(whole, nonzero), half);
		result = IR::F32(ir.Emit(
		    IR::ValueOpcode::SelectF32, {cardinal, IR::Value::F32(0.0f), result}));
	} else {
		// The architectural quarter-cycle result is exactly zero. Evaluating cos(pi/2) with a
		// rounded F32 PI can otherwise produce a half-precision subnormal instead.
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto quarter = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.25f)}));
		const auto three_quarters = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.75f)}));
		result = IR::F32(
		    ir.Emit(IR::ValueOpcode::SelectF32,
		            {ir.LogicalOr(quarter, three_quarters), IR::Value::F32(0.0f), result}));
	}

	const auto infinite = ir.IEqual(magnitude, IR::U32(IR::Value(0x7f800000u)));
	result              = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(infinite, invalid, bits));
}

void Translator::Float16Binary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool reverse) {
	const auto lhs = ReadF16AsF32(reverse ? inst.src1 : inst.src0);
	const auto rhs = ReadF16AsF32(reverse ? inst.src0 : inst.src1);
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {lhs, rhs})));
}

void Translator::Float16Ternary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		args[index] = mix ? IR::Value(ReadMixF32(operand)) : IR::Value(ReadF16AsF32(operand));
	}
	// Mixed f32 sources leave the product inexact: the f32 fma then rounds twice.
	if (opcode == IR::ValueOpcode::FPFma32 && !mix) {
		WriteF16(DestinationOperand(inst),
		         Float16Fma(IR::F32(args[0]), IR::F32(args[1]), IR::F32(args[2])));
		return;
	}
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {args[0], args[1], args[2]})));
}

void Translator::FloatUnary(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto type = IR::ArgTypeOf(opcode, 0);
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {ReadOperand(inst.src0, type)}));
}

void Translator::V_FRACT(const Decoder::Instruction& inst, IR::Type type) {
	// x - floor(x) rounds to 1.0 for a tiny negative x; v_fract returns the largest value below
	// one instead. The ordered compare lets NaN through.
	const auto destination = DestinationOperand(inst);
	if (type == IR::Type::F64) {
		const auto result    = ir.Emit(IR::ValueOpcode::FPFract64, {ReadOperand(inst.src0, type)});
		const auto below_one = ir.Emit(IR::ValueOpcode::BitCastF64U64,
		                               {IR::Value(uint64_t {0x3fefffffffffffff})});
		const auto above     = ir.Emit(IR::ValueOpcode::FPOrdGreaterThan64, {result, below_one});
		WriteOperand(destination, ir.Emit(IR::ValueOpcode::SelectF64, {above, below_one, result}));
		return;
	}
	const bool half   = type == IR::Type::F16;
	const auto source = half ? ReadF16AsF32(inst.src0) : IR::F32(ReadOperand(inst.src0, type));
	const auto result = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {source}));
	// Clamp in the destination precision: 0x3f7fe000 is the half 0x3bff.
	const auto below_one =
	    IR::F32(IR::Value::F32(std::bit_cast<float>(half ? 0x3f7fe000u : 0x3f7fffffu)));
	const auto above   = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThan32, {result, below_one}));
	const auto clamped = SelectF32(above, below_one, result);
	if (half) {
		WriteF16(destination, clamped);
	} else {
		WriteOperand(destination, clamped);
	}
}

void Translator::FloatBinary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                             bool reverse) {
	std::array<IR::Value, 2> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = SourceAt(inst, reverse ? 1u - index : index);
		args[index]        = ReadOperand(operand, IR::ArgTypeOf(opcode, index));
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1]}));
}

void Translator::FloatTernary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		const auto type    = IR::ArgTypeOf(opcode, index);
		args[index]        = type == IR::Type::F32 && mix ? IR::Value(ReadMixF32(operand))
		                                                  : ReadOperand(operand, type);
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1], args[2]}));
}

void Translator::V_FREXP_MANT_F32(const Decoder::Instruction& inst) {
	const auto bits = ReadU32(inst.src0);
	const auto exponent =
	    IR::U32(ir.Emit(IR::ValueOpcode::BitFieldUExtract, {bits, IR::Value(23u), IR::Value(8u)}));
	const auto mantissa = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x007fffffu)));
	const auto sign     = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x80000000u)));
	const auto base     = ir.BitwiseOr(sign, IR::U32(IR::Value(0x3f000000u)));
	const auto normal   = ir.BitwiseOr(base, mantissa);
	const auto msb      = IR::U32(ir.Emit(IR::ValueOpcode::FindUMsb32, {mantissa}));
	const auto shift    = ir.ISub(IR::U32(IR::Value(23u)), msb);
	const auto fraction =
	    ir.BitwiseAnd(ir.ShiftLeftLogical(mantissa, shift), IR::U32(IR::Value(0x007fffffu)));
	const auto subnormal = ir.BitwiseOr(base, fraction);
	const auto zero      = ir.IEqual(mantissa, IR::U32(IR::Value(0u)));
	const auto finite    = ir.Select(ir.INotEqual(exponent, IR::U32(IR::Value(0u))), normal,
	                                 ir.Select(zero, bits, subnormal));
	const auto result    = ir.Select(ir.IEqual(exponent, IR::U32(IR::Value(0xffu))), bits, finite);
	WriteOperand(DestinationOperand(inst), ir.BitCastF32(result));
}

void Translator::V_DOT2C_F32_F16(const Decoder::Instruction& inst) {
	auto a          = inst.src0;
	a.op_sel        = false;
	a.op_sel_hi     = true;
	auto b          = inst.src1;
	b.op_sel        = false;
	b.op_sel_hi     = true;
	const auto a_lo = ReadF16LaneAsF32(a, false);
	const auto a_hi = ReadF16LaneAsF32(a, true);
	const auto b_lo = ReadF16LaneAsF32(b, false);
	const auto b_hi = ReadF16LaneAsF32(b, true);
	const auto acc  = ReadMixF32(inst.dst);
	const auto lo   = IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a_lo, b_lo, acc}));
	WriteOperand(DestinationOperand(inst), ir.Emit(IR::ValueOpcode::FPFma32, {a_hi, b_hi, lo}));
}

void Translator::V_CUBEID_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 0u);
}

void Translator::V_CUBESC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 1u);
}

void Translator::V_CUBETC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 2u);
}

void Translator::V_CUBEMA_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 3u);
}

void Translator::FloatCube(const Decoder::Instruction& inst, uint32_t result_kind) {
	const auto x  = ReadMixF32(inst.src0);
	const auto y  = ReadMixF32(inst.src1);
	const auto z  = ReadMixF32(inst.src2);
	const auto nx = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {x}));
	const auto ny = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {y}));
	const auto nz = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {z}));
	const auto ax = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {x}));
	const auto ay = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {y}));
	const auto az = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {z}));
	const auto z_face =
	    ir.LogicalAnd(IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ax})),
	                  IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ay})));
	const auto y_face = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {ay, ax}));
	const auto x_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {x, IR::Value::F32(0.0f)}));
	const auto y_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {y, IR::Value::F32(0.0f)}));
	const auto z_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {z, IR::Value::F32(0.0f)}));
	const auto select_face = [&](IR::F32 x_value, IR::F32 y_value, IR::F32 z_value) {
		return SelectF32(z_face, z_value, SelectF32(y_face, y_value, x_value));
	};
	IR::F32 result;
	switch (result_kind) {
		case 0u:
			result = select_face(
			    SelectF32(x_neg, IR::F32(IR::Value::F32(1.0f)), IR::F32(IR::Value::F32(0.0f))),
			    SelectF32(y_neg, IR::F32(IR::Value::F32(3.0f)), IR::F32(IR::Value::F32(2.0f))),
			    SelectF32(z_neg, IR::F32(IR::Value::F32(5.0f)), IR::F32(IR::Value::F32(4.0f))));
			break;
		case 1u: result = select_face(SelectF32(x_neg, z, nz), x, SelectF32(z_neg, nx, x)); break;
		case 2u: result = select_face(ny, SelectF32(y_neg, nz, z), ny); break;
		case 3u:
			result = IR::F32(
			    ir.Emit(IR::ValueOpcode::FPMul32, {select_face(x, y, z), IR::Value::F32(2.0f)}));
			break;
		default: EXIT("invalid cube result kind");
	}
	WriteOperand(DestinationOperand(inst), result);
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
