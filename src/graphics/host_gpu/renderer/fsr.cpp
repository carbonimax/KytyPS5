#include "graphics/host_gpu/renderer/fsr.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/ffxApi.h"
#include "graphics/shader/shaderBindings.h"

#include <SDL3/SDL_loadso.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <string_view>

namespace Libs::Graphics::Fsr {

struct Context {
	Ffx::Context                          handle = nullptr;
	Ffx::CreateBackendVkDesc              backend {};
	Ffx::CreateContextDescUpscaleVersion  version {};
	Ffx::CreateContextDescUpscale         create {};
	uint32_t                              flags = 0; // Pssr::Flags fixed at creation
	std::chrono::steady_clock::time_point last;
};

namespace {

// The request flags a context fixes at creation.
constexpr uint32_t FIXED_FLAGS = Pssr::kReverseDepth | Pssr::kAutoExposure;

struct Api {
	Ffx::PfnCreateContext  create;
	Ffx::PfnDestroyContext destroy;
	Ffx::PfnConfigure      configure;
	Ffx::PfnQuery          query;
	Ffx::PfnDispatch       dispatch;
};

SDL_SharedObject*  g_module = nullptr;
Api                g_api {};
Ffx::DeviceContext g_device {};
bool               g_enabled = false;

void Warn(std::string_view text) {
	Log::WriteToConsoleAndLog(fmt::format("WARNING: FSR: {}; PSSR scales linearly.\n", text));
}

void Close() {
	SDL_UnloadObject(g_module);
	g_module = nullptr;
}

template <typename Function>
bool Bind(Function& function, const char* name) {
	function = reinterpret_cast<Function>(SDL_LoadFunction(g_module, name));
	return function != nullptr;
}

void OnMessage(uint32_t type, const wchar_t* message) {
	std::string text;
	for (const wchar_t* c = message; c != nullptr && *c != 0; ++c) {
		text.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
	}
	Log::WriteToConsoleAndLog(fmt::format("FSR {}: {}\n", type == 0 ? "error" : "warning", text));
}

uint32_t SurfaceFormat(VkFormat format) {
	switch (format) {
		case VK_FORMAT_R16G16B16A16_SFLOAT: return Ffx::SURFACE_FORMAT_R16G16B16A16_FLOAT;
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return Ffx::SURFACE_FORMAT_R11G11B10_FLOAT;
		case VK_FORMAT_R16G16_SFLOAT: return Ffx::SURFACE_FORMAT_R16G16_FLOAT;
		case VK_FORMAT_R32G32_SFLOAT: return Ffx::SURFACE_FORMAT_R32G32_FLOAT;
		case VK_FORMAT_R16_SFLOAT: return Ffx::SURFACE_FORMAT_R16_FLOAT;
		case VK_FORMAT_R32_SFLOAT:
		case VK_FORMAT_D32_SFLOAT: return Ffx::SURFACE_FORMAT_R32_FLOAT;
		default: return Ffx::SURFACE_FORMAT_UNKNOWN;
	}
}

bool Whole(const Dlss::Resource& resource) {
	return resource.range.baseMipLevel == 0 && resource.range.baseArrayLayer == 0 &&
	       SurfaceFormat(resource.format) != Ffx::SURFACE_FORMAT_UNKNOWN;
}

// Every image is in the general layout, which the API calls the unordered access state.
Ffx::Resource MakeResource(const Dlss::Resource* resource) {
	if (resource == nullptr) {
		return {};
	}
	uint32_t usage = resource->read_write != 0 ? Ffx::RESOURCE_USAGE_UAV
	                                           : Ffx::RESOURCE_USAGE_READ_ONLY;
	if ((resource->range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
		usage |= Ffx::RESOURCE_USAGE_DEPTHTARGET;
	}
	return {.resource    = resource->image,
	        .description = {.type      = Ffx::RESOURCE_TYPE_TEXTURE2D,
	                        .format    = SurfaceFormat(resource->format),
	                        .width     = resource->width,
	                        .height    = resource->height,
	                        .depth     = 1,
	                        .mip_count = 1,
	                        .flags     = 0,
	                        .usage     = usage},
	        .state       = Ffx::RESOURCE_STATE_UNORDERED_ACCESS};
}

} // namespace

bool Accepts(const Pssr::Request& request) {
	ShaderTextureResource depth {}, motion {};
	std::copy_n(request.depth, 8, depth.fields);
	std::copy_n(request.motion, 8, motion.fields);
	return (request.flags & Pssr::kSpatialOnly) == 0 && !depth.IsNull() && !motion.IsNull() &&
	       request.x == 0 && request.y == 0;
}

bool Fits(const Dlss::Inputs& inputs) {
	return Whole(inputs.color) && Whole(inputs.depth) && Whole(inputs.motion) &&
	       Whole(inputs.output) && (inputs.exposure == nullptr || Whole(*inputs.exposure));
}

bool Takes(const Context* context, const Pssr::Request& request, const Dlss::Inputs& inputs) {
	return context->create.max_upscale_size.width == inputs.output.width &&
	       context->create.max_upscale_size.height == inputs.output.height &&
	       context->flags == (request.flags & FIXED_FLAGS) &&
	       request.width <= context->create.max_render_size.width &&
	       request.height <= context->create.max_render_size.height;
}

void Load() {
	if (!Config::Ps5ProEnabled() || Config::GetFsrFolder().empty()) {
		return;
	}
	// LoadLibrary wants backslashes.
	const auto path = (Config::GetFsrFolder() / "amd_fidelityfx_loader_vk.dll").make_preferred();
	g_module        = SDL_LoadObject(Common::PathToString(path).c_str());
	if (g_module == nullptr) {
		Warn(fmt::format("{} did not load", Common::PathToString(path)));
		return;
	}
	if (!Bind(g_api.create, "ffxCreateContext") || !Bind(g_api.destroy, "ffxDestroyContext") ||
	    !Bind(g_api.configure, "ffxConfigure") || !Bind(g_api.query, "ffxQuery") ||
	    !Bind(g_api.dispatch, "ffxDispatch")) {
		Warn("the FSR loader lacks an entry point");
		Close();
	}
}

bool Loaded() {
	return g_module != nullptr;
}

void AppendExtensions(std::vector<const char*>&                   out,
                      const std::vector<vk::ExtensionProperties>& available, bool device) {
	if (g_module == nullptr || !device) {
		return;
	}
	// shortcut: VK_EXT_shader_float8 (FSR 4 on RDNA 4) is left out, untested without such a GPU.
	const auto same = [](std::string_view other) {
		return other == VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
	};
	if (std::ranges::any_of(available, same,
	                        [](const auto& properties) {
		                        return std::string_view(properties.extensionName.data());
	                        }) &&
	    std::ranges::none_of(out, same)) {
		out.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	}
}

void Init(const GraphicContext& graphics, PFN_vkGetDeviceProcAddr get_device_proc_addr) {
	if (g_module == nullptr) {
		return;
	}
	g_device = {static_cast<VkDevice>(graphics.device),
	            static_cast<VkPhysicalDevice>(graphics.physical_device), get_device_proc_addr};
	Ffx::ConfigureDescGlobalDebug debug {{Ffx::CONFIGURE_DESC_TYPE_GLOBALDEBUG, nullptr},
	                                     Ffx::EFFECT_ID_UPSCALE, OnMessage,
	                                     Ffx::DEBUG_LEVEL_WARNINGS};
	(void)g_api.configure(nullptr, &debug.header);
	uint64_t                       count = 0;
	Ffx::QueryDescGetVersions      versions {{Ffx::QUERY_DESC_TYPE_GET_VERSIONS, nullptr},
	                                         Ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE, &g_device,
	                                         &count, nullptr, nullptr};
	if (g_api.query(nullptr, &versions.header) != Ffx::RETURN_OK || count == 0) {
		Warn("the FSR DLLs offer no upscaler for this GPU");
		Close();
		return;
	}
	std::vector<uint64_t>    ids(count);
	std::vector<const char*> names(count);
	versions.version_ids   = ids.data();
	versions.version_names = names.data();
	(void)g_api.query(nullptr, &versions.header);
	std::string list;
	for (uint64_t i = 0; i < count; ++i) {
		list += fmt::format("{}{}", i == 0 ? "" : ", ", names[i] != nullptr ? names[i] : "?");
	}
	g_enabled = true;
	Log::WriteToConsoleAndLog(fmt::format("FSR: upscaling PSSR with the DLLs in {}; versions: {}\n",
	                                      Common::PathToString(Config::GetFsrFolder()), list));
}

void Shutdown() {
	g_enabled = false;
	g_device  = {};
	if (g_module != nullptr) {
		Close();
	}
}

bool Enabled() {
	return g_enabled;
}

Context* CreateContext(const Pssr::Request& request, const Dlss::Inputs& inputs) {
	auto* context  = new Context {};
	context->flags = request.flags & FIXED_FLAGS;
	// One context takes every render size up to the output's, so dynamic resolution keeps its
	// history.
	context->backend = {{Ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, nullptr}, g_device};
	context->version = {{Ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION, &context->backend.header},
	                    Ffx::UPSCALER_VERSION};
	uint32_t flags = Ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | Ffx::UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
	if ((context->flags & Pssr::kReverseDepth) != 0) {
		flags |= Ffx::UPSCALE_ENABLE_DEPTH_INVERTED;
	}
	if ((context->flags & Pssr::kAutoExposure) != 0) {
		flags |= Ffx::UPSCALE_ENABLE_AUTO_EXPOSURE;
	}
	context->create = {{Ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE, &context->version.header},
	                   flags,
	                   {inputs.output.width, inputs.output.height},
	                   {inputs.output.width, inputs.output.height},
	                   OnMessage};
	if (const auto result = g_api.create(&context->handle, &context->create.header, nullptr);
	    result != Ffx::RETURN_OK) {
		Warn(fmt::format("creating a context to {}x{} failed ({})", inputs.output.width,
		                 inputs.output.height, result));
		delete context;
		return nullptr;
	}
	Ffx::QueryGetProviderVersion provider {{Ffx::QUERY_DESC_TYPE_PROVIDER_VERSION, nullptr}, 0,
	                                       nullptr};
	(void)g_api.query(&context->handle, &provider.header);
	Log::WriteToConsoleAndLog(fmt::format("FSR: {} context to {}x{}\n",
	                                      provider.version_name != nullptr ? provider.version_name
	                                                                       : "unknown",
	                                      inputs.output.width, inputs.output.height));
	return context;
}

bool Evaluate(vk::CommandBuffer command, Context* context, const Pssr::Request& request,
              Dlss::Inputs& inputs, bool reset) {
	const auto now = std::chrono::steady_clock::now();
	const auto delta =
	    context->last.time_since_epoch().count() == 0
	        ? 16.7f
	        : std::clamp(std::chrono::duration<float, std::milli>(now - context->last).count(),
	                     1.0f, 100.0f);
	context->last = now;
	Ffx::DispatchDescUpscale dispatch {};
	dispatch.header         = {Ffx::DISPATCH_DESC_TYPE_UPSCALE, nullptr};
	dispatch.command_list   = static_cast<VkCommandBuffer>(command);
	dispatch.color          = MakeResource(&inputs.color);
	dispatch.depth          = MakeResource(&inputs.depth);
	dispatch.motion_vectors = MakeResource(&inputs.motion);
	dispatch.exposure       = MakeResource(inputs.exposure);
	dispatch.output         = MakeResource(&inputs.output);
	dispatch.jitter_offset  = {request.jitter[0], request.jitter[1]};
	dispatch.motion_vector_scale = {request.motion_scale[0], request.motion_scale[1]};
	dispatch.render_size         = {request.width, request.height};
	dispatch.upscale_size        = {inputs.output.width, inputs.output.height};
	dispatch.frame_time_delta    = delta;
	dispatch.pre_exposure        = request.pre_exposure > 0.0f ? request.pre_exposure : 1.0f;
	dispatch.reset               = reset;
	// shortcut: PSSR passes no camera; these are typical values, read them from the guest if
	// FSR 3 shows depth-dependent artefacts.
	dispatch.camera_near                 = 0.1f;
	dispatch.camera_far                  = 1000.0f;
	dispatch.camera_fov_angle_vertical   = 1.0f;
	dispatch.view_space_to_meters_factor = 1.0f;
	if (const auto result = g_api.dispatch(&context->handle, &dispatch.header);
	    result != Ffx::RETURN_OK) {
		Warn(fmt::format("dispatch failed ({})", result));
		return false;
	}
	return true;
}

void ReleaseContext(Context* context) {
	(void)g_api.destroy(&context->handle, nullptr);
	delete context;
}

} // namespace Libs::Graphics::Fsr
