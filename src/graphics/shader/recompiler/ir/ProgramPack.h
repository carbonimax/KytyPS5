#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// A translated program kept for later, with its instructions packed into a byte stream (about a
// tenth of their in-memory size). The blocks and the resource plan stay C++ objects; their values
// refer to instructions by index while packed. Unpack restores an identical program: same
// instructions, arguments, use lists (in order), flags and block structure.
class PackedProgram {
public:
	PackedProgram() = default;

	PackedProgram(const PackedProgram&)            = delete;
	PackedProgram& operator=(const PackedProgram&) = delete;
	PackedProgram(PackedProgram&&) noexcept            = default;
	PackedProgram& operator=(PackedProgram&&) noexcept = default;

	// Moves `program` into this object and packs it. False (program untouched) when the program
	// holds state that cannot be packed (a compiled walker, or a value outside the program).
	[[nodiscard]] bool Pack(Program& program);
	// Once, after a successful Pack.
	[[nodiscard]] Program Unpack();

	[[nodiscard]] bool   Packed() const { return packed; }
	[[nodiscard]] size_t Bytes() const;

private:
	friend class ProgramPacker;

	Program                                             shell;
	std::vector<uint8_t>                                stream;
	std::vector<std::pair<uint32_t, std::vector<uint32_t>>> control_phi_predecessors;
	uint32_t                                            inst_count = 0;
	bool                                                packed     = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR
