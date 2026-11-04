// IR-level tests of the optional shader optimization passes (step 5.6): every rewrite each pass
// makes, and the cases it must leave alone.
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/LaunchedLaneFolding.h"
#include "graphics/shader/recompiler/ir/passes/Optimizations.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

namespace {

void Check(bool condition, const char* message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
}

struct Fixture {
	Program program;
	Block*  block = nullptr;

	Fixture() {
		program.stage           = Libs::Graphics::ShaderType::Compute;
		program.user_data_count = 64;
		block                   = AddBlock();
	}

	Block* AddBlock() {
		auto  storage = std::make_unique<Block>();
		auto* result  = storage.get();
		program.block_storage.push_back(std::move(storage));
		program.blocks.push_back(result);
		result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
		return result;
	}

	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {}, uint64_t flags = 0,
	           Block* destination = nullptr) {
		auto& inst = (destination != nullptr ? destination : block)->AppendNewInst(opcode, args, flags);
		return Value(&inst);
	}

	// Keeps a value alive the way a store does.
	void Use(Value value, Block* destination = nullptr) {
		Emit(ValueOpcode::ReferenceU32, {value}, 0, destination);
	}

	void Simplify() {
		RemoveIdentities(program);
		EliminateDeadCode(program.blocks);
	}

	[[nodiscard]] size_t Count(ValueOpcode opcode) const {
		size_t count = 0;
		for (const auto* each: program.blocks)
			for (const auto& inst: *each)
				count += inst.GetOpcode() == opcode ? 1u : 0u;
		return count;
	}
};

Value UserData(Fixture& fixture, uint32_t index, Block* block = nullptr) {
	return fixture.Emit(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))}, 0, block);
}

// Launched-lane folding: the vertex-shader entry test lane < popcount(Ballot(true)), built from
// ESVertCount as ~0 >> ((0 - (count | 1 << 28)) & 63) and a per-lane bit test.
struct LaneTestShape {
	ShaderType stage          = ShaderType::Vertex;
	uint32_t   wave_size      = 64;
	bool       ballot_true    = true;
	bool       ballot_late    = false; // ballot evaluated after a branch instead of in the entry block
	uint32_t   merged_bits    = 1u << 28u;
	bool       merge          = true;
	bool       swap_halves    = false;
	bool       commuted       = false;
};

struct LaneTestResult {
	uint32_t folded;
	size_t   compares;
	size_t   wide_shifts;
	size_t   ballots;
};

LaneTestResult RunLaneTest(const LaneTestShape& shape) {
	Fixture f;
	f.program.stage     = shape.stage;
	f.program.wave_size = shape.wave_size;
	auto* entry         = f.block;
	auto* body          = f.AddBlock();
	entry->AddBranch(body);
	auto*      ballot_block = shape.ballot_late ? body : entry;
	const auto predicate    = shape.ballot_true ? Value(true) : f.Emit(ValueOpcode::SLessThan32, {UserData(f, 1), Value(0u)});
	const auto ballot       = f.Emit(ValueOpcode::Ballot, {predicate}, 0, ballot_block);
	const auto lo_count = f.Emit(ValueOpcode::BitCount32, {f.Emit(ValueOpcode::CompositeExtractU32x4, {ballot, Value(0u)}, 0, ballot_block)}, 0, ballot_block);
	const auto hi_count = f.Emit(ValueOpcode::BitCount32, {f.Emit(ValueOpcode::CompositeExtractU32x4, {ballot, Value(1u)}, 0, ballot_block)}, 0, ballot_block);
	auto count = f.Emit(ValueOpcode::IAdd32, {shape.commuted ? hi_count : lo_count, shape.commuted ? lo_count : hi_count}, 0, ballot_block);
	if (shape.merge) {
		count = shape.commuted ? f.Emit(ValueOpcode::BitwiseOr32, {Value(shape.merged_bits), count}, 0, ballot_block)
		                       : f.Emit(ValueOpcode::BitwiseOr32, {count, Value(shape.merged_bits)}, 0, ballot_block);
	}
	// The guest computes the mask after the branch, from the value it kept in s3.
	const auto negative = f.Emit(ValueOpcode::ISub32, {Value(0u), count}, 0, body);
	const auto mask     = f.Emit(ValueOpcode::ShiftRightLogical64, {Value(~uint64_t {0}), negative}, 0, body);
	const auto lo       = f.Emit(ValueOpcode::CompositeExtractU64, {mask, Value(shape.swap_halves ? 1u : 0u)}, 0, body);
	const auto hi       = f.Emit(ValueOpcode::CompositeExtractU64, {mask, Value(shape.swap_halves ? 0u : 1u)}, 0, body);
	const auto lane     = f.Emit(ValueOpcode::LaneId, {}, 0, body);
	const auto low_half = f.Emit(ValueOpcode::ULessThan32, {lane, Value(32u)}, 0, body);
	const auto word     = f.Emit(ValueOpcode::SelectU32, {low_half, lo, hi}, 0, body);
	const auto amount   = f.Emit(ValueOpcode::BitwiseAnd32, {shape.commuted ? Value(31u) : lane, shape.commuted ? lane : Value(31u)}, 0, body);
	const auto shifted  = f.Emit(ValueOpcode::ShiftRightLogical32, {word, amount}, 0, body);
	const auto bit      = f.Emit(ValueOpcode::BitwiseAnd32, {shape.commuted ? Value(1u) : shifted, shape.commuted ? shifted : Value(1u)}, 0, body);
	const auto active   = f.Emit(ValueOpcode::INotEqual32, {shape.commuted ? Value(0u) : bit, shape.commuted ? bit : Value(0u)}, 0, body);
	// An EXEC-masked VGPR write: the select keeps the old value in lanes the test rejects.
	const auto written  = f.Emit(ValueOpcode::SelectU32, {active, UserData(f, 2, body), UserData(f, 3, body)}, 0, body);
	f.Use(written, body);
	const auto folded = FoldLaunchedLaneTests(f.program);
	f.Simplify();
	return {folded, f.Count(ValueOpcode::INotEqual32), f.Count(ValueOpcode::ShiftRightLogical64), f.Count(ValueOpcode::Ballot)};
}

// The entry test folds to true, and the mask arithmetic and the ballot become dead.
void TestLaunchedLaneFold() {
	const auto result = RunLaneTest({});
	Check(result.folded == 1u, "lanes: test folded");
	Check(result.compares == 0u && result.wide_shifts == 0u && result.ballots == 0u, "lanes: mask arithmetic removed");
	LaneTestShape commuted;
	commuted.commuted = true;
	Check(RunLaneTest(commuted).folded == 1u, "lanes: commuted operands folded");
	LaneTestShape plain;
	plain.merge = false;
	Check(RunLaneTest(plain).folded == 1u, "lanes: count without merged bits folded");
	LaneTestShape aligned;
	aligned.merged_bits = 0x40u;
	Check(RunLaneTest(aligned).folded == 1u, "lanes: merged bits above the shift range folded");
}

// Anything that is not that exact test, or not a wave64 vertex shader, is left alone.
void TestLaunchedLaneKept() {
	const auto kept = [](const LaneTestShape& shape, const char* message) {
		const auto result = RunLaneTest(shape);
		Check(result.folded == 0u && result.compares == 1u && result.ballots == 1u, message);
	};
	LaneTestShape shape;
	shape.stage = ShaderType::Pixel;
	kept(shape, "lanes: pixel shader kept");
	shape           = {};
	shape.wave_size = 32;
	kept(shape, "lanes: wave32 kept");
	shape             = {};
	shape.ballot_true = false;
	kept(shape, "lanes: ballot of a predicate kept");
	shape             = {};
	shape.ballot_late = true;
	kept(shape, "lanes: ballot outside the entry block kept");
	shape             = {};
	shape.merged_bits = (1u << 28u) | 1u;
	kept(shape, "lanes: merged bits inside the shift range kept");
	shape             = {};
	shape.swap_halves = true;
	kept(shape, "lanes: swapped mask halves kept");
}

} // namespace

// The IR asserts report through these; the passes under test never reach them.
namespace Common {

int DbgExitIfHandler(const char*, const char*, int) {
	return 1;
}

int DbgExitHandler(const char*, int, std::string_view text) {
	throw std::runtime_error(std::string(text));
}

void DbgExit(int) {
	std::abort();
}

} // namespace Common

int main(int argc, char** argv) {
	(void)argc;
	(void)argv;
	try {
		const auto Run = [](const char* name, auto test) {
			try {
				test();
			} catch (const std::exception& exception) {
				throw std::runtime_error(std::string(name) + ": " + exception.what());
			}
		};
		Run("lanes fold", TestLaunchedLaneFold);
		Run("lanes kept", TestLaunchedLaneKept);
	} catch (const std::exception& exception) {
		std::fprintf(stderr, "FAILED: %s\n", exception.what());
		return 1;
	}
	std::printf("shader optimization tests passed\n");
	return 0;
}

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.cpp"
#include "graphics/shader/recompiler/ir/passes/LaunchedLaneFolding.cpp"
#include "graphics/shader/recompiler/ir/passes/Optimizations.cpp"
