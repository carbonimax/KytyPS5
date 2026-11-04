#include "graphics/shader/recompiler/ir/passes/LaunchedLaneFolding.h"

#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

const Inst* Op(Value value, ValueOpcode opcode) {
	const auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

bool IsU32(Value value, uint32_t immediate) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == immediate;
}

// The other operand of a commutative binary op whose one operand is the given immediate.
std::optional<Value> OtherThan(const Inst& inst, uint32_t immediate) {
	if (IsU32(inst.Arg(1), immediate)) return inst.Arg(0).Resolve();
	if (IsU32(inst.Arg(0), immediate)) return inst.Arg(1).Resolve();
	return std::nullopt;
}

bool IsLane(Value value) {
	return Op(value, ValueOpcode::LaneId) != nullptr;
}

// Word `index` of Ballot(true) evaluated in the entry block (every launched lane runs it).
bool IsLaunchedBallotWord(const Program& program, Value value, uint32_t index) {
	const auto* extract = Op(value, ValueOpcode::CompositeExtractU32x4);
	if (extract == nullptr || !IsU32(extract->Arg(1), index)) return false;
	const auto* ballot = Op(extract->Arg(0), ValueOpcode::Ballot);
	if (ballot == nullptr || ballot->Parent() != program.blocks.front()) return false;
	const auto predicate = ballot->Arg(0).Resolve();
	return predicate.IsImmediate() && predicate.GetType() == Type::U1 && predicate.U1();
}

// popcount(word 0) + popcount(word 1) of the launched-lane ballot.
bool IsLaunchedCount(const Program& program, Value value) {
	const auto* add = Op(value, ValueOpcode::IAdd32);
	if (add == nullptr) return false;
	for (uint32_t order = 0; order < 2u; order++) {
		const auto* low  = Op(add->Arg(order), ValueOpcode::BitCount32);
		const auto* high = Op(add->Arg(order ^ 1u), ValueOpcode::BitCount32);
		if (low != nullptr && high != nullptr && IsLaunchedBallotWord(program, low->Arg(0), 0u) &&
		    IsLaunchedBallotWord(program, high->Arg(0), 1u))
			return true;
	}
	return false;
}

// ~0 >> ((0 - (count | K)) & 63) with K & 63 == 0: the low `count` lanes (all 64 for count 64).
bool IsLaunchedMask(const Program& program, Value value) {
	const auto* shift = Op(value, ValueOpcode::ShiftRightLogical64);
	if (shift == nullptr) return false;
	const auto ones = shift->Arg(0).Resolve();
	if (!ones.IsImmediate() || ones.GetType() != Type::U64 || ones.U64() != ~0ull) return false;
	const auto* negate = Op(shift->Arg(1), ValueOpcode::ISub32);
	if (negate == nullptr || !IsU32(negate->Arg(0), 0u)) return false;
	auto count = negate->Arg(1).Resolve();
	if (const auto* merged = Op(count, ValueOpcode::BitwiseOr32); merged != nullptr) {
		for (uint32_t side = 0; side < 2u; side++) {
			const auto other = merged->Arg(side ^ 1u).Resolve();
			if (other.IsImmediate() && other.GetType() == Type::U32 && (other.U32() & 63u) == 0u) {
				count = merged->Arg(side).Resolve();
				break;
			}
		}
	}
	return IsLaunchedCount(program, count);
}

// INotEqual32((word >> (LaneId & 31)) & 1, 0) with word = LaneId < 32 ? mask.lo : mask.hi.
bool IsLaunchedLaneTest(const Program& program, const Inst& inst) {
	const auto bit = OtherThan(inst, 0u);
	const auto* one = bit ? Op(*bit, ValueOpcode::BitwiseAnd32) : nullptr;
	const auto shifted_value = one != nullptr ? OtherThan(*one, 1u) : std::nullopt;
	const auto* shifted = shifted_value ? Op(*shifted_value, ValueOpcode::ShiftRightLogical32) : nullptr;
	if (shifted == nullptr) return false;
	const auto* amount = Op(shifted->Arg(1), ValueOpcode::BitwiseAnd32);
	const auto lane = amount != nullptr ? OtherThan(*amount, 31u) : std::nullopt;
	if (!lane || !IsLane(*lane)) return false;
	const auto* word = Op(shifted->Arg(0), ValueOpcode::SelectU32);
	if (word == nullptr) return false;
	const auto* half = Op(word->Arg(0), ValueOpcode::ULessThan32);
	if (half == nullptr || !IsLane(half->Arg(0)) || !IsU32(half->Arg(1), 32u)) return false;
	const auto* low  = Op(word->Arg(1), ValueOpcode::CompositeExtractU64);
	const auto* high = Op(word->Arg(2), ValueOpcode::CompositeExtractU64);
	if (low == nullptr || high == nullptr || !IsU32(low->Arg(1), 0u) || !IsU32(high->Arg(1), 1u) ||
	    !(low->Arg(0).Resolve() == high->Arg(0).Resolve()))
		return false;
	return IsLaunchedMask(program, low->Arg(0));
}

} // namespace

uint32_t FoldLaunchedLaneTests(Program& program) {
	if (program.stage != ShaderType::Vertex || program.wave_size != 64u || program.blocks.empty()) {
		return 0;
	}
	uint32_t folded = 0;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::INotEqual32 || !inst.HasUses() ||
			    !IsLaunchedLaneTest(program, inst))
				continue;
			inst.ReplaceUsesWith(Value(true));
			folded++;
		}
	}
	return folded;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
