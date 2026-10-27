#include "graphics/shader/recompiler/ir/ProgramPack.h"

#include "graphics/shader/recompiler/ir/Block.h"

#include <list>
#include <memory>
#include <span>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Stream layout (LEB128 numbers):
//   per block of block_storage, then value_storage: instruction count
//   section A, per instruction: opcode, a bit set of the default fields, then the others: argument
//     count, flags, walk output, definitions, parent block (index + 1, 0 = none)
//   section B, per instruction: arguments (phi: count, then predecessor block + 1 and value),
//     then its use list in order (user index, operand)
// A value is its type, then its payload: instruction index (opaque), register or immediate bits.
// Instruction indices are relative to the instruction being written (zigzag), mostly one byte.
class ProgramPacker {
public:
	static bool Pack(PackedProgram& out, Program& program);
	static Program Unpack(PackedProgram& packed);

private:
	struct Writer {
		std::vector<uint8_t>& out;

		void PutSigned(int64_t value) {
			Put((static_cast<uint64_t>(value) << 1u) ^ static_cast<uint64_t>(value >> 63));
		}

		void Put(uint64_t value) {
			while (value >= 0x80u) {
				out.push_back(static_cast<uint8_t>(value | 0x80u));
				value >>= 7u;
			}
			out.push_back(static_cast<uint8_t>(value));
		}
	};

	struct Reader {
		std::span<const uint8_t> in;
		size_t                   pos = 0;

		uint64_t Get() {
			uint64_t value = 0;
			for (uint32_t shift = 0;; shift += 7u) {
				const uint8_t byte = in[pos++];
				value |= static_cast<uint64_t>(byte & 0x7fu) << shift;
				if ((byte & 0x80u) == 0) {
					return value;
				}
			}
		}

		int64_t GetSigned() {
			const uint64_t value = Get();
			return static_cast<int64_t>(value >> 1u) ^ -static_cast<int64_t>(value & 1u);
		}
	};

	struct Numbering {
		std::unordered_map<const Inst*, uint32_t>  insts;
		std::unordered_map<const Block*, uint32_t> blocks;
		bool                                       ok = true;

		uint64_t InstIndex(const IR::Inst* inst) {
			const auto it = insts.find(inst);
			if (it == insts.end()) {
				ok = false;
				return 0;
			}
			return it->second;
		}
		uint64_t BlockIndex(const IR::Block* block) {
			if (block == nullptr) {
				return 0;
			}
			const auto it = blocks.find(block);
			if (it == blocks.end()) {
				ok = false;
				return 0;
			}
			return it->second + 1u;
		}
	};

	enum HeaderBits : uint32_t {
		Live           = 1u << 0u,
		OwnerParent    = 1u << 1u, // parent is the block (or value storage: none) holding it
		NoWalkOutput   = 1u << 2u,
		NoDefinitions  = 1u << 3u,
		NoFlags        = 1u << 4u,
		DefaultArgs    = 1u << 5u, // argument count of a new instruction with that opcode
	};

	static uint8_t DefaultArgCount(ValueOpcode opcode) {
		return opcode == ValueOpcode::Phi ? Inst::PhiArity : static_cast<uint8_t>(NumArgsOf(opcode));
	}

	static void PutValue(Writer& w, Numbering& numbers, const Value& value, int64_t base) {
		w.Put(static_cast<uint32_t>(value.type));
		switch (value.type) {
			case Type::Void: break;
			case Type::Opaque: w.PutSigned(static_cast<int64_t>(numbers.InstIndex(value.inst)) - base); break;
			case Type::ScalarReg: w.Put(static_cast<uint16_t>(value.scalar_reg)); break;
			case Type::VectorReg: w.Put(static_cast<uint16_t>(value.vector_reg)); break;
			case Type::U1: w.Put(value.imm_u1 ? 1u : 0u); break;
			case Type::U8: w.Put(value.imm_u8); break;
			case Type::U16:
			case Type::F16: w.Put(value.imm_u16); break;
			case Type::U32:
			case Type::F32: w.Put(value.imm_u32); break;
			default: w.Put(value.imm_u64); break;
		}
	}

	static Value GetValue(Reader& r, const std::vector<Inst*>& insts, int64_t base) {
		Value value;
		value.type = static_cast<Type>(r.Get());
		switch (value.type) {
			case Type::Void: break;
			case Type::Opaque: value.inst = insts[static_cast<size_t>(base + r.GetSigned())]; break;
			case Type::ScalarReg: value.scalar_reg = static_cast<ScalarReg>(r.Get()); break;
			case Type::VectorReg: value.vector_reg = static_cast<VectorReg>(r.Get()); break;
			case Type::U1: value.imm_u1 = r.Get() != 0; break;
			case Type::U8: value.imm_u8 = static_cast<uint8_t>(r.Get()); break;
			case Type::U16:
			case Type::F16: value.imm_u16 = static_cast<uint16_t>(r.Get()); break;
			case Type::U32:
			case Type::F32: value.imm_u32 = static_cast<uint32_t>(r.Get()); break;
			default: value.imm_u64 = r.Get(); break;
		}
		return value;
	}

	// Values held outside the instructions (blocks, resource plan): instruction pointer <-> index.
	template <typename Fn>
	static void ForEachOuterValue(Program& program, Fn&& fn) {
		for (auto& block: program.block_storage) {
			fn(block->condition);
			fn(block->indirect_target);
		}
		for (auto& source: program.descriptor_sources) {
			for (auto& dword: source.dwords) {
				fn(dword);
			}
			if (source.indirect_descriptor) {
				fn(source.indirect_descriptor->key_count);
				fn(source.indirect_descriptor->selector_first);
				fn(source.indirect_descriptor->selector_mask);
			}
		}
		for (auto& read: program.srt_reads) {
			fn(read.value);
		}
		for (auto& block: program.control_flow) {
			fn(block.condition);
		}
		for (auto& value: program.uniform_fill.values) {
			fn(value);
		}
		fn(program.uniform_fill.pattern_size);
		fn(program.uniform_fill.element_count);
		for (auto& write: program.scalar_writes) {
			fn(write.value);
		}
	}

	// fn(inst, owner block or nullptr for the value storage).
	template <typename Fn>
	static void ForEachInst(Program& program, Fn&& fn) {
		for (auto& block: program.block_storage) {
			for (auto& inst: block->Instructions()) {
				fn(inst, block.get());
			}
		}
		for (auto& inst: program.value_storage) {
			fn(inst, nullptr);
		}
	}

	static void SetArgStorage(Inst& inst, uint8_t num_args) {
		if (inst.num_args == Inst::PhiArity) {
			std::destroy_at(&inst.phi_args);
		} else if (inst.num_args > Inst::InlineArity) {
			std::destroy_at(&inst.large_args);
		} else {
			std::destroy_at(&inst.fixed_args);
		}
		if (num_args == Inst::PhiArity) {
			std::construct_at(&inst.phi_args);
		} else if (num_args > Inst::InlineArity) {
			std::construct_at(&inst.large_args, num_args);
		} else {
			std::construct_at(&inst.fixed_args);
		}
		inst.num_args = num_args;
	}
};

bool ProgramPacker::Pack(PackedProgram& out, Program& program) {
	if (out.packed || program.srt_code != nullptr) {
		return false;
	}
	Numbering numbers;
	uint32_t  count = 0;
	ForEachInst(program, [&](const Inst& inst, const Block*) { numbers.insts.emplace(&inst, count++); });
	for (uint32_t index = 0; index < program.block_storage.size(); index++) {
		numbers.blocks.emplace(program.block_storage[index].get(), index);
	}

	std::vector<uint8_t> stream;
	stream.reserve(static_cast<size_t>(count) * 24u + 64u);
	Writer w {stream};
	for (const auto& block: program.block_storage) {
		w.Put(block->Instructions().size());
	}
	w.Put(program.value_storage.size());
	ForEachInst(program, [&](const Inst& inst, const Block* owner) {
		uint32_t bits = inst.live ? Live : 0u;
		bits |= inst.parent == owner ? OwnerParent : 0u;
		bits |= inst.walk_output == UINT32_MAX ? NoWalkOutput : 0u;
		bits |= inst.definitions[0] == 0 && inst.definitions[1] == 0 ? NoDefinitions : 0u;
		bits |= inst.flags == 0 ? NoFlags : 0u;
		bits |= inst.num_args == DefaultArgCount(inst.opcode) ? DefaultArgs : 0u;
		w.Put(static_cast<uint16_t>(inst.opcode));
		w.Put(bits);
		if ((bits & DefaultArgs) == 0) w.Put(inst.num_args);
		if ((bits & NoFlags) == 0) w.Put(inst.flags);
		if ((bits & NoWalkOutput) == 0) w.Put(inst.walk_output);
		if ((bits & NoDefinitions) == 0) {
			w.Put(inst.definitions[0]);
			w.Put(inst.definitions[1]);
		}
		if ((bits & OwnerParent) == 0) w.Put(numbers.BlockIndex(inst.parent));
	});
	int64_t self = 0;
	ForEachInst(program, [&](const Inst& inst, const Block*) {
		if (inst.num_args == Inst::PhiArity) {
			w.Put(inst.phi_args.size());
			for (const auto& [block, value]: inst.phi_args) {
				w.Put(numbers.BlockIndex(block));
				PutValue(w, numbers, value, self);
			}
		} else {
			for (size_t index = 0; index < inst.num_args; index++) {
				PutValue(w, numbers, inst.Arg(index), self);
			}
		}
		w.Put(inst.uses.size());
		for (const auto& use: inst.uses) {
			w.PutSigned(static_cast<int64_t>(numbers.InstIndex(use.user)) - self);
			w.Put(use.operand);
		}
		self++;
	});
	ForEachOuterValue(program, [&](Value& value) {
		if (value.type == Type::Opaque) {
			(void)numbers.InstIndex(value.inst);
		}
	});
	std::vector<std::pair<uint32_t, std::vector<uint32_t>>> phi_predecessors;
	phi_predecessors.reserve(program.control_phi_predecessors.size());
	for (const auto& [inst, predecessors]: program.control_phi_predecessors) {
		phi_predecessors.emplace_back(static_cast<uint32_t>(numbers.InstIndex(inst)), predecessors);
	}
	if (!numbers.ok) {
		return false;
	}

	out.shell = std::move(program);
	auto& shell = out.shell;
	ForEachOuterValue(shell, [&](Value& value) {
		if (value.type == Type::Opaque) {
			value.inst = reinterpret_cast<Inst*>(static_cast<uintptr_t>(numbers.insts.at(value.inst)) + 1u);
		}
	});
	shell.control_phi_predecessors.clear();
	// Same teardown as ~Program: detach every argument while all definitions are alive.
	for (auto& inst: shell.value_storage) {
		inst.Invalidate();
	}
	for (auto& block: shell.block_storage) {
		for (auto& inst: block->Instructions()) {
			inst.Invalidate();
		}
	}
	shell.value_storage.clear();
	for (auto& block: shell.block_storage) {
		block->Instructions().clear();
	}
	stream.shrink_to_fit();
	out.stream                   = std::move(stream);
	out.control_phi_predecessors = std::move(phi_predecessors);
	out.inst_count               = count;
	out.packed                   = true;
	return true;
}

Program ProgramPacker::Unpack(PackedProgram& packed) {
	auto&  shell = packed.shell;
	Reader r {packed.stream};
	std::vector<uint64_t> counts(shell.block_storage.size() + 1u);
	for (auto& count: counts) {
		count = r.Get();
	}
	std::vector<Inst*> insts;
	insts.reserve(packed.inst_count);
	const auto create = [&](std::list<Inst>& list, Block* owner) {
		const auto opcode   = static_cast<ValueOpcode>(r.Get());
		const auto bits     = static_cast<uint32_t>(r.Get());
		const auto num_args = (bits & DefaultArgs) != 0 ? DefaultArgCount(opcode) : static_cast<uint8_t>(r.Get());
		const auto flags    = (bits & NoFlags) != 0 ? 0u : r.Get();
		auto&      inst     = list.emplace_back(opcode, flags);
		if (inst.num_args != num_args) {
			SetArgStorage(inst, num_args);
		}
		inst.live        = (bits & Live) != 0;
		inst.walk_output = (bits & NoWalkOutput) != 0 ? UINT32_MAX : static_cast<uint32_t>(r.Get());
		if ((bits & NoDefinitions) == 0) {
			inst.definitions[0] = static_cast<uint32_t>(r.Get());
			inst.definitions[1] = static_cast<uint32_t>(r.Get());
		}
		if ((bits & OwnerParent) != 0) {
			inst.parent = owner;
		} else {
			const auto parent = r.Get();
			inst.parent       = parent == 0 ? nullptr : shell.block_storage[parent - 1u].get();
		}
		insts.push_back(&inst);
	};
	for (size_t block = 0; block < shell.block_storage.size(); block++) {
		auto* owner = shell.block_storage[block].get();
		for (uint64_t index = 0; index < counts[block]; index++) {
			create(owner->Instructions(), owner);
		}
	}
	for (uint64_t index = 0; index < counts.back(); index++) {
		create(shell.value_storage, nullptr);
	}
	for (size_t self = 0; self < insts.size(); self++) {
		auto*      inst = insts[self];
		const auto base = static_cast<int64_t>(self);
		if (inst->num_args == Inst::PhiArity) {
			const auto phi_count = r.Get();
			inst->phi_args.reserve(phi_count);
			for (uint64_t index = 0; index < phi_count; index++) {
				const auto block = r.Get();
				auto*      pred  = block == 0 ? nullptr : shell.block_storage[block - 1u].get();
				inst->phi_args.emplace_back(pred, GetValue(r, insts, base));
			}
		} else if (inst->num_args > Inst::InlineArity) {
			for (auto& arg: inst->large_args) {
				arg = GetValue(r, insts, base);
			}
		} else {
			for (size_t index = 0; index < inst->num_args; index++) {
				inst->fixed_args[index] = GetValue(r, insts, base);
			}
		}
		const auto use_count = r.Get();
		inst->uses.reserve(use_count);
		for (uint64_t index = 0; index < use_count; index++) {
			auto* user = insts[static_cast<size_t>(base + r.GetSigned())];
			inst->uses.push_back({user, static_cast<size_t>(r.Get())});
		}
	}
	ForEachOuterValue(shell, [&](Value& value) {
		if (value.type == Type::Opaque) {
			value.inst = insts[reinterpret_cast<uintptr_t>(value.inst) - 1u];
		}
	});
	for (auto& [inst, predecessors]: packed.control_phi_predecessors) {
		shell.control_phi_predecessors.emplace(insts[inst], std::move(predecessors));
	}
	packed.control_phi_predecessors.clear();
	packed.stream = {};
	packed.packed = false;
	return std::move(shell);
}

bool PackedProgram::Pack(Program& program) {
	return ProgramPacker::Pack(*this, program);
}

Program PackedProgram::Unpack() {
	return ProgramPacker::Unpack(*this);
}

size_t PackedProgram::Bytes() const {
	size_t bytes = sizeof(*this) + stream.capacity() + shell.block_storage.size() * (sizeof(Block) + 96u) +
	               shell.descriptor_sources.size() * sizeof(DescriptorSource) +
	               shell.memory_info.size() * sizeof(MemoryInfo) + shell.export_info.size() * sizeof(ExportInfo) +
	               shell.control_flow.size() * sizeof(ResourceBlock) + 4096u;
	for (const auto& [inst, predecessors]: control_phi_predecessors) {
		bytes += sizeof(inst) + 32u + predecessors.size() * sizeof(uint32_t);
	}
	return bytes;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
