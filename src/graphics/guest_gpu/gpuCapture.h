#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_

#include <atomic>
#include <cstdint>
#include <functional>
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

// Zero-filled guest bytes (zero pages are not stored as Memory).
struct Zero {
	uint64_t address = 0;
	uint64_t size    = 0;
};

// Guest bytes the CPU rewrote during the frame (reused rings). The Update records that follow a
// Submit are written once the GPU is idle, before that Submit.
struct Update {
	uint64_t             address = 0;
	std::vector<uint8_t> bytes;
};

// XXH3-64 of guest memory after the replay.
struct Expect {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0;
};

using Record = std::variant<Memory, Shader, Submit, FrameEnd, Expect, Zero, Update>;

struct Capture {
	std::vector<Record> records;
};

[[nodiscard]] std::vector<uint8_t> Serialize(const Capture& capture);
// Streaming form of Serialize: the header, then each record appended to `out`.
void SerializeHeader(std::vector<uint8_t>& out);
void SerializeRecord(std::vector<uint8_t>& out, const Record& record);
// nullopt and an error message on a malformed or unsupported file.
[[nodiscard]] std::optional<Capture> Parse(std::span<const uint8_t> file, std::string* error);

// Recording in the emulator. KYTY_GPU_CAPTURE=<dir> enables it (off by default); it records GPU
// frame KYTY_GPU_CAPTURE_FRAME=N (suspend points processed before it) and the frame after each
// <dir>/trigger file appears, to <dir>/frame_<N>.kygpucap. Guest bytes are recorded at their first use
// in the frame; GPU-resident data is published to guest memory when recording starts.
extern const bool       g_enabled;
extern std::atomic_bool g_recording;
inline bool             Enabled() noexcept { return g_enabled; }
inline bool             Recording() noexcept { return g_recording.load(std::memory_order_relaxed); }
void                    TouchSlow(uint64_t address, uint64_t size, bool output);
// Guest bytes the GPU side is about to use; `output` ranges are hashed at the frame end (Expect).
inline void Touch(uint64_t address, uint64_t size, bool output = false) {
	if (Recording()) TouchSlow(address, size, output);
}
// While Recording(): a submission's first slice (queue as in Submit).
void RecordSubmit(std::span<const uint32_t> commands, uint32_t queue);
// While Enabled(): ShaderMapUserData keeps the register defaults, only valid during that call.
void KeepShader(Shader shader);
// While Recording(): a shader-map lookup; records the shader and its guest data once per frame.
void UseShader(uint64_t address);
// GPU thread, at each processed suspend point while Enabled(). `publish_gpu_data` writes
// GPU-resident data back to guest memory.
void FrameBoundary(const std::function<void()>& publish_gpu_data);

} // namespace Libs::Graphics::GpuCapture

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_GPUCAPTURE_H_ */
