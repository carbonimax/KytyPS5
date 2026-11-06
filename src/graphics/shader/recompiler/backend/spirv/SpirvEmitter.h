#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// How storage buffer accesses past the bound range read zero and drop writes (KYTY_BOUNDS_MODE).
enum class StorageBounds : uint8_t {
	Shader, // A branch around every access.
	Select, // Loads read a clamped element and select zero past the end; stores branch.
	// robustBufferAccess2 and nullDescriptor bound the access. Loads and plain stores of dword-only
	// bindings have no check: their range is cut to whole dwords (StorageDeviceRange). Bindings
	// with 8/16-bit views keep the exact range, which the device rounds up to
	// storage_bounds_alignment, so their loads select zero past it and their stores branch.
	Device,
};

// Counters of KYTY_BOUNDS_MODE=2, at Profile::storage_bounds_verify_address.
enum class StorageBoundsCounter : uint32_t {
	OutOfRangeLoads,   // Loads without a branch that the shader check would read as zero.
	NonZeroLoads,      // Of those, unchecked device loads that did not read zero (must stay 0).
	OutOfRangeStores,  // Unchecked stores that the shader check would drop.
	TailReads,         // Selected loads whose device value past the range was not zero.
	Count,
};

// Host device capabilities that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
	StorageBounds storage_bounds = StorageBounds::Shader;
	// robustStorageBufferAccessSizeAlignment (1 or 4) with StorageBounds::Device.
	uint32_t storage_bounds_alignment = 4;
	// KYTY_BOUNDS_MODE=2: device address of the StorageBoundsCounter words, 0 otherwise.
	uint64_t storage_bounds_verify_address = 0;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	// KYTY_SPV_NONWRITABLE=1: NonWritable on the storage buffers a program only reads, so the
	// driver can load them through the scalar cache.
	bool nonwritable_buffers = false;
};

// True when no buffer or image of `info` is written or atomically updated and no access asks for
// GLC/DLC coherence: then a guest-memory read cannot alias a write of this shader, nor poll for one
// from another agent, and the buffer arrays can be NonWritable. Writes by other draws or dispatches
// are ordered by the barriers between them, which also invalidate the scalar cache.
[[nodiscard]] inline bool StorageBuffersReadOnly(const IR::ShaderInfo& info) {
	const auto writes = [](const auto& resource) { return resource.written || resource.atomic; };
	return !info.coherent_buffers && std::ranges::none_of(info.buffers, writes) &&
	       std::ranges::none_of(info.images, writes);
}

// Descriptor range of a storage binding of `range` bytes whose shader leaves loads and plain stores
// to the device. Without 8/16-bit views every access is a whole dword, so cutting the range to
// whole dwords makes the device drop exactly the dwords the shader check drops, whatever the
// robustness alignment. 0 means a null descriptor.
[[nodiscard]] constexpr uint64_t StorageDeviceRange(const Profile& profile, uint64_t range,
                                                   bool subword) {
	if (profile.storage_bounds != StorageBounds::Device || subword ||
	    profile.storage_bounds_alignment == 1u) {
		return range;
	}
	return range & ~uint64_t {3};
}

inline constexpr Profile DefaultProfile {};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t           push_data_start_dword = 0,
                                  const Profile&     profile               = DefaultProfile,
                                  bool               emit_debug_info       = false,
                                  ShaderHostFeatures host_features         = {});

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
