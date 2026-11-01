#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
};

inline constexpr Profile DefaultProfile {};

// KYTY_GPU_TIMING (local measurement build): what the emitter added to the program last emitted
// on this thread (reset by EmitProgram).
struct EmitStats {
	uint32_t storage_bounds         = 0; // element bounds checks on storage and scalar buffers
	uint32_t storage_bounds_skipped = 0; // the same, left to the device (KYTY_PROF_STORAGE_BOUNDS=0)
	uint32_t other_bounds           = 0; // LDS, GDS and scratch bounds checks
};
EmitStats& ThreadEmitStats() noexcept;
// KYTY_PROF_STORAGE_BOUNDS=0 (local measurement build only, never a default): storage and scalar
// buffer loads and plain stores skip their shader bounds check and rely on robustBufferAccess2
// (enabled on the device when supported), as upstream ee034be4 does. Elements past the descriptor
// range but inside its robustness alignment are no longer zero: rendering parity is not
// guaranteed; this only bounds the GPU time of the checks.
bool ProfSkipStorageBounds() noexcept;

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t           push_data_start_dword = 0,
                                  const Profile&     profile               = DefaultProfile,
                                  bool               emit_debug_info       = false,
                                  ShaderHostFeatures host_features         = {});

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
