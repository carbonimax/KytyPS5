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

constexpr uint32_t Version = 1;

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

// XXH3-64 of guest memory after the replay.
struct Expect {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0;
};

using Record = std::variant<Memory, Shader, Submit, FrameEnd, Expect>;

struct Capture {
	std::vector<Record> records;
};

[[nodiscard]] std::vector<uint8_t> Serialize(const Capture& capture);
// nullopt and an error message on a malformed or unsupported file.
[[nodiscard]] std::optional<Capture> Parse(std::span<const uint8_t> file, std::string* error);

} // namespace Libs::Graphics::GpuCapture

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_ */
