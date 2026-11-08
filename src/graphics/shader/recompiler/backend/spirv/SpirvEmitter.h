#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

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

// KYTY_TESS_PROBE records of TessProbe::RecordDwords words. Word 0 is the claim (first writer
// wins), words 1-3 identify the invocation. LS: by VertexIndex modulo LsRecords (1 VertexIndex,
// 2 InstanceIndex),
// words 4-12 the LS outputs. TES: for each of the first patches, its 4 domain corners and the
// first interior point of each domain quadrant (1 PrimitiveId, 2-3 TessCoord bits), words 4-195
// the control point words read (address / 4), 196-219 the patch constant words read, 240-243
// the exported position, 244-247 the same after the clip-space conversion. One spare record after each range
// takes the invocations that record nothing.
struct TessProbe {
	static constexpr uint32_t RecordDwords = 256;
	static constexpr uint32_t LsRecords    = 512;
	static constexpr uint32_t TesPatches   = 64;
	static constexpr uint32_t TesBase      = LsRecords + 1u;
	static constexpr uint32_t TesRecords   = TesPatches * 8u;
	// HS: by PrimitiveId * 4 + InvocationId (1 PrimitiveId, 2 InvocationId), words 4-147 the
	// input words read (address / 4), 160-207 the own output words written.
	static constexpr uint32_t HsBase       = TesBase + TesRecords + 1u;
	static constexpr uint32_t HsRecords    = TesPatches * 4u;
	static constexpr uint32_t Bytes        = (HsBase + HsRecords + 1u) * RecordDwords * 4u;
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
	// KYTY_TESS_PROBE: device address of the TessProbe records, 0 otherwise.
	uint64_t tess_probe_address = 0;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
};

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
