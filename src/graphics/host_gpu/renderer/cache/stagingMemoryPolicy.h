#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STAGINGMEMORYPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STAGINGMEMORYPOLICY_H_

#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

// Device-local host-visible memory larger than this is a Resizable BAR heap (a plain BAR is 256 MiB).
inline constexpr uint64_t REBAR_MIN_HEAP_SIZE = 1024ull * 1024 * 1024;

// True when the CPU can write device-local memory of a large heap (Resizable BAR on a discrete GPU),
// so that a host-written ring can live in VRAM. False on integrated or unified-memory GPUs (MoltenVK),
// where host-visible memory is already the device's and moving the ring changes nothing.
[[nodiscard]] inline bool HasResizableBarHeap(vk::PhysicalDeviceType                    type,
                                              const vk::PhysicalDeviceMemoryProperties& memory) {
	if (type != vk::PhysicalDeviceType::eDiscreteGpu) {
		return false;
	}
	constexpr vk::MemoryPropertyFlags bar =
	    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible;
	for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
		const auto& memory_type = memory.memoryTypes[i];
		if ((memory_type.propertyFlags & bar) == bar &&
		    memory.memoryHeaps[memory_type.heapIndex].size > REBAR_MIN_HEAP_SIZE) {
			return true;
		}
	}
	return false;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STAGINGMEMORYPOLICY_H_ */
