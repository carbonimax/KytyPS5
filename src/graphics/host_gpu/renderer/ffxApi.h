// The parts of the AMD FSR API (FidelityFX SDK 2.4.0: ffx_api.h, ffx_api_types.h,
// vk/ffx_api_vk.h, ffx_upscale.h) that the PSSR upscaler uses, copied from the SDK headers.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FFXAPI_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FFXAPI_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics::Fsr::Ffx {

constexpr uint32_t RETURN_OK = 0;

using Context    = void*;
using StructType = uint64_t;

struct Header {
	StructType type;
	Header*    next;
};

using Message = void (*)(uint32_t type, const wchar_t* message);

struct Allocation;
using PfnCreateContext  = uint32_t (*)(Context* context, Header* desc, const Allocation* memory);
using PfnDestroyContext = uint32_t (*)(Context* context, const Allocation* memory);
using PfnConfigure      = uint32_t (*)(Context* context, const Header* desc);
using PfnQuery          = uint32_t (*)(Context* context, Header* desc);
using PfnDispatch       = uint32_t (*)(Context* context, const Header* desc);

constexpr uint64_t EFFECT_MASK       = 0x00ff0000u;
constexpr uint64_t BACKEND_MASK      = 0xff000000u;
constexpr uint64_t EFFECT_ID_UPSCALE = 0x00010000u;
constexpr uint64_t BACKEND_ID_VK     = 0x01000000u;

constexpr uint64_t MakeEffectSubId(uint64_t effect, uint64_t sub) {
	return (effect & EFFECT_MASK) | (sub & ~EFFECT_MASK);
}
constexpr uint64_t MakeBackendSubId(uint64_t backend, uint64_t sub) {
	return (backend & BACKEND_MASK) | (sub & ~BACKEND_MASK);
}

constexpr StructType CONFIGURE_DESC_TYPE_GLOBALDEBUG  = 7u;
constexpr StructType QUERY_DESC_TYPE_GET_VERSIONS     = 4u;
constexpr StructType DESC_TYPE_OVERRIDE_VERSION       = 5u;
constexpr StructType QUERY_DESC_TYPE_PROVIDER_VERSION = 6u;
constexpr StructType CREATE_CONTEXT_DESC_TYPE_BACKEND_VK = MakeBackendSubId(BACKEND_ID_VK, 0x03);
constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE = MakeEffectSubId(EFFECT_ID_UPSCALE, 0x00);
constexpr StructType DISPATCH_DESC_TYPE_UPSCALE       = MakeEffectSubId(EFFECT_ID_UPSCALE, 0x01);
constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION =
    MakeEffectSubId(EFFECT_ID_UPSCALE, 0x0b);

constexpr uint32_t DEBUG_LEVEL_WARNINGS = 2;

struct ConfigureDescGlobalDebug {
	Header   header;
	uint64_t effect_id;
	Message  message;
	uint32_t debug_level;
};

struct QueryDescGetVersions {
	Header       header;
	uint64_t     create_desc_type;
	void*        device; // a DeviceContext
	uint64_t*    output_count;
	uint64_t*    version_ids;
	const char** version_names;
};

struct OverrideVersion {
	Header   header;
	uint64_t version_id;
};

struct QueryGetProviderVersion {
	Header      header;
	uint64_t    version_id;
	const char* version_name;
};

struct DeviceContext {
	VkDevice                device;
	VkPhysicalDevice        physical_device;
	PFN_vkGetDeviceProcAddr get_device_proc_addr;
};

struct CreateBackendVkDesc {
	Header        header;
	DeviceContext device;
};

// FfxApiSurfaceFormat values.
constexpr uint32_t SURFACE_FORMAT_UNKNOWN             = 0;
constexpr uint32_t SURFACE_FORMAT_R32G32B32A32_FLOAT  = 3;
constexpr uint32_t SURFACE_FORMAT_R16G16B16A16_FLOAT  = 4;
constexpr uint32_t SURFACE_FORMAT_R11G11B10_FLOAT     = 16;
constexpr uint32_t SURFACE_FORMAT_R16G16_FLOAT        = 18;
constexpr uint32_t SURFACE_FORMAT_R16_FLOAT           = 21;
constexpr uint32_t SURFACE_FORMAT_R32_FLOAT           = 28;
constexpr uint32_t SURFACE_FORMAT_R32G32_FLOAT        = 6;

constexpr uint32_t RESOURCE_USAGE_READ_ONLY   = 0;
constexpr uint32_t RESOURCE_USAGE_UAV         = 1u << 1;
constexpr uint32_t RESOURCE_USAGE_DEPTHTARGET = 1u << 2;

constexpr uint32_t RESOURCE_STATE_UNORDERED_ACCESS = 1u << 1;

constexpr uint32_t RESOURCE_TYPE_TEXTURE2D = 2;

struct ResourceDescription {
	uint32_t type;
	uint32_t format;
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t mip_count;
	uint32_t flags;
	uint32_t usage;
};

struct Resource {
	void*               resource; // a VkImage
	ResourceDescription description;
	uint32_t            state;
};

struct FloatCoords2D {
	float x;
	float y;
};

struct Dimensions2D {
	uint32_t width;
	uint32_t height;
};

// FfxApiCreateContextUpscaleFlags.
constexpr uint32_t UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE  = 1u << 0;
constexpr uint32_t UPSCALE_ENABLE_DEPTH_INVERTED      = 1u << 3;
constexpr uint32_t UPSCALE_ENABLE_AUTO_EXPOSURE       = 1u << 5;
constexpr uint32_t UPSCALE_ENABLE_DYNAMIC_RESOLUTION  = 1u << 6;

constexpr uint32_t UPSCALER_VERSION = (4u << 22) | (1u << 12) | 2u; // 4.1.2

struct CreateContextDescUpscale {
	Header       header;
	uint32_t     flags;
	Dimensions2D max_render_size;
	Dimensions2D max_upscale_size;
	Message      message;
};

struct CreateContextDescUpscaleVersion {
	Header   header;
	uint32_t version;
};

struct DispatchDescUpscale {
	Header        header;
	void*         command_list; // a VkCommandBuffer
	Resource      color;
	Resource      depth;
	Resource      motion_vectors;
	Resource      exposure;
	Resource      reactive;
	Resource      transparency_and_composition;
	Resource      output;
	FloatCoords2D jitter_offset;
	FloatCoords2D motion_vector_scale;
	Dimensions2D  render_size;
	Dimensions2D  upscale_size;
	bool          enable_sharpening;
	float         sharpness;
	float         frame_time_delta; // milliseconds
	float         pre_exposure;
	bool          reset;
	float         camera_near;
	float         camera_far;
	float         camera_fov_angle_vertical; // radians
	float         view_space_to_meters_factor;
	uint32_t      flags;
};
static_assert(sizeof(Resource) == 48 && sizeof(DispatchDescUpscale) == 432);

} // namespace Libs::Graphics::Fsr::Ffx

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FFXAPI_H_
