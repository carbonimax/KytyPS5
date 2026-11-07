#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

// GPU capture file: PM4 submissions, the guest memory and shaders they use, and expected output
// hashes, replayed without the guest CPU or HLE (memory is mapped back at the same addresses).
//
// Layout (little-endian, fixed-width fields): a 16-byte header {char magic[8], u32 version,
// u32 reserved}, then records {u32 type, u32 flags, u64 payload_size, payload, zero padding to
// 8 bytes}. A reader rejects an unknown version, record type or flag.
namespace Libs::Graphics::GpuCapture {

constexpr uint32_t Version = 2;

// Guest bytes at an address, restored before the first submission.
struct Memory {
	uint64_t             address = 0;
	std::vector<uint8_t> bytes;
};

// A shader registration (ShaderMapUserData). user_data and input_semantics are guest pointers.
struct Shader {
	struct Register {
		uint32_t offset = 0;
		uint32_t value  = 0;
	};
	uint64_t              address             = 0;
	uint64_t              user_data           = 0;
	uint64_t              input_semantics     = 0;
	uint32_t              type                = 0;
	uint32_t              code_size_bytes     = 0;
	uint32_t              scratch_size_dwords = 0;
	uint32_t              num_input_semantics = 0;
	std::vector<Register> cx_registers;
	std::vector<Register> sh_registers;
};

// Commands in guest memory; queue 0 is graphics, 0x20.. are the compute queues.
struct Submit {
	uint64_t address     = 0;
	uint32_t queue       = 0;
	uint32_t dword_count = 0;
};

// Frame boundary (GuestGpu::SuspendPoint).
struct FrameEnd {};

// Zero-filled guest bytes (zero pages are not stored as Memory).
struct Zero {
	uint64_t address = 0;
	uint64_t size    = 0;
};

// Guest bytes the CPU rewrote during the frame (reused rings), seen while the GPU processed
// submission `submit` (0-based index of its Submit record in the frame). Replay writes them, GPU
// idle, right before that Submit.
struct Update {
	uint64_t             address = 0;
	uint32_t             submit  = 0;
	std::vector<uint8_t> bytes;
};

// XXH3-64 of guest memory after the replay.
struct Expect {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0;
};

// A submission (Submit index) the game's GPU had started but not finished at the frame end (a
// wait nobody released yet): replay leaves it blocked as captured instead of failing.
struct InFlight {
	uint32_t submit = 0;
};

// Order-independent hash (UnorderedHash) of the range of the Expect just before it: a replay whose
// bytes are a permutation of the game's records (atomic append lists) keeps it.
struct Unordered {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0;
};

// The command processor state a queue (as in Submit) had when the recording started: compute queues
// keep theirs across frames, and a scope starting after submission 0 inherits every queue's. Raw bytes
// of the recording build's layout (CommandProcessor::SaveState); replay restores it before the first Submit.
struct CpState {
	uint32_t             queue = 0;
	std::vector<uint8_t> bytes;
};

using Record = std::variant<Memory, Shader, Submit, FrameEnd, Expect, Zero, Update, InFlight, Unordered, CpState>;

// Sum of a 64-bit mix of each dword (a zero-padded tail counts as one): equal for any permutation
// of records whose size is a multiple of 4 bytes.
[[nodiscard]] uint64_t UnorderedHash(std::span<const uint8_t> bytes);

struct Capture {
	std::vector<Record> records;
};

[[nodiscard]] std::vector<uint8_t> Serialize(const Capture& capture);
// Streaming form of Serialize: the header, then each record appended to `out`.
void SerializeHeader(std::vector<uint8_t>& out);
void SerializeRecord(std::vector<uint8_t>& out, const Record& record);
// nullopt and an error message on a malformed or unsupported file.
[[nodiscard]] std::optional<Capture> Parse(std::span<const uint8_t> file, std::string* error);

} // namespace Libs::Graphics::GpuCapture

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_ */
