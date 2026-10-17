#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

// KYTY_IMAGE_BACKING_POOL (default on, =0 turns it off): the Vulkan backing of a deleted image is
// kept, keyed by its creation parameters, and handed to the next image created with the same
// parameters instead of going through vmaDestroyImage / vmaCreateImage. Titles that alias the
// same guest memory for different transient render targets destroy and create several
// full-screen targets per frame (a 3840x2160 creation alone can take milliseconds). The texture
// cache destroys an image only after the GPU finished with it, and a recycled backing is used as
// a new image: layout Undefined, contents undefined. KYTY_IMAGE_BACKING_POOL_MIB bounds the
// memory kept (default 512).
struct ImageKey {
	uint32_t format, image_type, width, height, depth, layers, mips, samples;
	uint64_t usage, flags;
	bool     operator==(const ImageKey&) const = default;
};

struct ImageKeyHash {
	size_t operator()(const ImageKey& k) const noexcept {
		uint64_t h = 1469598103934665603ull;
		for (const uint64_t v: {uint64_t {k.format}, uint64_t {k.image_type}, uint64_t {k.width},
		                        uint64_t {k.height}, uint64_t {k.depth}, uint64_t {k.layers},
		                        uint64_t {k.mips}, uint64_t {k.samples}, k.usage, k.flags}) {
			h = (h ^ v) * 1099511628211ull;
		}
		return static_cast<size_t>(h);
	}
};

struct PooledImage {
	ImageKey      key;
	VkImage       image;
	VmaAllocation allocation;
	uint64_t      bytes;
};

struct ImagePool {
	std::mutex                                                           mutex;
	std::unordered_map<ImageKey, std::vector<PooledImage>, ImageKeyHash> by_key;
	std::deque<PooledImage>                                              order; // oldest first
	uint64_t                                                             bytes = 0;
};

ImagePool g_image_pool;

bool ImagePoolEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_BACKING_POOL");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t ImagePoolLimit() {
	static const uint64_t limit = [] {
		const char* value = std::getenv("KYTY_IMAGE_BACKING_POOL_MIB");
		const long  mib   = value != nullptr ? std::strtol(value, nullptr, 10) : 512;
		return static_cast<uint64_t>(mib > 0 ? mib : 512) << 20u;
	}();
	return limit;
}

ImageKey MakeImageKey(const VulkanImage& image) {
	return {static_cast<uint32_t>(image.format),
	        static_cast<uint32_t>(image.image_type),
	        image.extent.width,
	        image.extent.height,
	        image.extent.depth,
	        image.layers,
	        image.mip_levels,
	        image.samples,
	        static_cast<uint64_t>(static_cast<VkImageUsageFlags>(image.usage)),
	        static_cast<uint64_t>(static_cast<VkImageCreateFlags>(image.flags))};
}

ImageKey MakeImageKey(const vk::ImageCreateInfo& info) {
	return {static_cast<uint32_t>(info.format),
	        static_cast<uint32_t>(info.imageType),
	        info.extent.width,
	        info.extent.height,
	        info.extent.depth,
	        info.arrayLayers,
	        info.mipLevels,
	        static_cast<uint32_t>(info.samples),
	        static_cast<uint64_t>(static_cast<VkImageUsageFlags>(info.usage)),
	        static_cast<uint64_t>(static_cast<VkImageCreateFlags>(info.flags))};
}

// Only plain images are pooled: the key does not describe extension structures, sharing or tiling.
bool IsPoolable(const vk::ImageCreateInfo& info) {
	return info.pNext == nullptr && info.tiling == vk::ImageTiling::eOptimal &&
	       info.sharingMode == vk::SharingMode::eExclusive &&
	       info.initialLayout == vk::ImageLayout::eUndefined;
}

// Destroys the oldest pooled images until the pool fits under `limit`. Caller holds the mutex.
void TrimImagePoolLocked(VmaAllocator allocator, uint64_t limit) {
	while (g_image_pool.bytes > limit && !g_image_pool.order.empty()) {
		const PooledImage victim = g_image_pool.order.front();
		g_image_pool.order.pop_front();
		auto&      list = g_image_pool.by_key[victim.key];
		const auto it   = std::find_if(list.begin(), list.end(),
		                              [&](const PooledImage& p) { return p.image == victim.image; });
		if (it != list.end()) {
			list.erase(it);
		}
		g_image_pool.bytes -= victim.bytes;
		vmaDestroyImage(allocator, victim.image, victim.allocation);
	}
}

} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	{
		std::lock_guard lock(g_image_pool.mutex);
		TrimImagePoolLocked(allocator, 0);
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	const bool       use_pool     = ImagePoolEnabled() && IsPoolable(image_info);
	vk::Image::CType native_image = VK_NULL_HANDLE;
	bool             recycled     = false;
	if (use_pool) {
		std::lock_guard lock(g_image_pool.mutex);
		const auto      found = g_image_pool.by_key.find(MakeImageKey(image_info));
		if (found != g_image_pool.by_key.end() && !found->second.empty()) {
			const PooledImage hit = found->second.back();
			found->second.pop_back();
			const auto it = std::find_if(g_image_pool.order.begin(), g_image_pool.order.end(),
			                             [&](const PooledImage& p) { return p.image == hit.image; });
			if (it != g_image_pool.order.end()) {
				g_image_pool.order.erase(it);
			}
			g_image_pool.bytes -= hit.bytes;
			native_image     = hit.image;
			image.allocation = hit.allocation;
			recycled         = true;
		}
	}
	vk::Result result = vk::Result::eSuccess;
	if (!recycled) {
		VmaAllocationCreateInfo alloc_info {};
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		result = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		if (result != vk::Result::eSuccess && ImagePoolEnabled()) {
			// Out of device memory: give the pooled backings back and try again.
			{
				std::lock_guard lock(g_image_pool.mutex);
				TrimImagePoolLocked(allocator, 0);
			}
			result = static_cast<vk::Result>(vmaCreateImage(
			    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
			    &alloc_info, &native_image, &image.allocation, nullptr));
		}
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.poolable   = use_pool;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (ImagePoolEnabled() && image.poolable) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		const uint64_t bytes = allocation_info.size;
		if (bytes != 0 && bytes <= ImagePoolLimit() / 2) {
			std::lock_guard   lock(g_image_pool.mutex);
			const ImageKey    key = MakeImageKey(image);
			const PooledImage entry {key, static_cast<VkImage>(image.image), image.allocation, bytes};
			g_image_pool.by_key[key].push_back(entry);
			g_image_pool.order.push_back(entry);
			g_image_pool.bytes += bytes;
			TrimImagePoolLocked(allocator, ImagePoolLimit());
			image.image      = nullptr;
			image.allocation = nullptr;
			return;
		}
	}
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
