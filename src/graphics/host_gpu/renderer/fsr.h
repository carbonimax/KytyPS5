#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_

#include "graphics/host_gpu/renderer/dlss.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <vector>

// AMD FSR upscaling through the user's signed FSR SDK DLLs (amd_fidelityfx_loader_vk.dll and
// amd_fidelityfx_upscaler_vk.dll). The API picks FSR 4 where the GPU runs it, else FSR 3.
namespace Libs::Graphics::Fsr {

struct Context;

// Whether FSR can take the request: temporal inputs and no render offset.
[[nodiscard]] bool Accepts(const Pssr::Request& request);
// Whether FSR can read the inputs: whole-image views at level and layer 0 in formats it knows.
[[nodiscard]] bool Fits(const Dlss::Inputs& inputs);
// Whether a context made for an earlier request takes this one's output and flags.
[[nodiscard]] bool Takes(const Context* context, const Pssr::Request& request,
                         const Dlss::Inputs& inputs);

// Loads the FSR loader DLL in Pro mode with an FSR folder. Call before the instance.
void Load();
[[nodiscard]] bool Loaded();
// Adds the device extensions FSR 4 uses that `available` has.
void AppendExtensions(std::vector<const char*>&                   out,
                      const std::vector<vk::ExtensionProperties>& available, bool device);
// Starts FSR on the device. It stays off when the DLLs offer no upscaler for it.
void Init(const GraphicContext& graphics, PFN_vkGetDeviceProcAddr get_device_proc_addr);
// Call after every context is released and before the device is destroyed.
void               Shutdown();
[[nodiscard]] bool Enabled(); // false while FSR is off

// A null context or false means FSR can't take the request.
[[nodiscard]] Context* CreateContext(const Pssr::Request& request, const Dlss::Inputs& inputs);
[[nodiscard]] bool     Evaluate(vk::CommandBuffer command, Context* context,
                                const Pssr::Request& request, Dlss::Inputs& inputs, bool reset);
void                   ReleaseContext(Context* context);

} // namespace Libs::Graphics::Fsr

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FSR_H_
