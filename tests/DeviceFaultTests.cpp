// VK_EXT_device_fault reporting (graphics/host_gpu/deviceFault.cpp).
//
// Default run: the device-address registry lookup and the record formatting, no GPU (exit 0, 1 on
// failure).
// --force-device-lost: a compute shader writes through a device address 64 GiB past a small buffer,
// which loses the device; the driver's fault record must name an address range holding that write
// (exit 0 found, 1 not found, 2 inconclusive: no driver with deviceFault, or the write did not fault).
// It resets the GPU context, so it is opt-in (KYTY_DEVICE_FAULT_GPU_TEST) and never runs by default.
#include "graphics/host_gpu/deviceFault.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using Libs::Graphics::DeviceFault::AddressRegistry;
using Libs::Graphics::DeviceFault::Format;
using Libs::Graphics::DeviceFault::Query;
using Libs::Graphics::DeviceFault::Record;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAIL: %s\n", what);
		g_failures++;
	}
}

bool Contains(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

void TestRegistry() {
	AddressRegistry registry;
	Check(!registry.Find(0x1000).has_value(), "empty registry finds nothing");
	registry.Add({0x1000, 0x1000, 0x5000, "a"});
	registry.Add({0x10000, 0x100, 0, "b"});

	auto found = registry.Find(0x1800);
	Check(found && std::strcmp(found->kind, "a") == 0 && !found->freed, "address inside a");
	found = registry.Find(0x1fff);
	Check(found && std::strcmp(found->kind, "a") == 0, "last byte of a");
	found = registry.Find(0x2100);
	Check(found && std::strcmp(found->kind, "a") == 0 && 0x2100 - found->begin >= found->size,
	      "overrun of a reports a");
	found = registry.Find(0x10080);
	Check(found && std::strcmp(found->kind, "b") == 0, "address inside b");
	Check(!registry.Find(0x800).has_value(), "address below every range");

	registry.Remove(0x1000);
	found = registry.Find(0x1800);
	Check(found && std::strcmp(found->kind, "a") == 0 && found->freed, "freed a still named");
	registry.Add({0x1000, 0x2000, 0, "c"});
	found = registry.Find(0x1800);
	Check(found && std::strcmp(found->kind, "c") == 0 && !found->freed, "live range wins over freed");
	registry.Remove(0x4242); // Unknown: no-op.
}

void TestFormat() {
	AddressRegistry registry;
	registry.Add({0x10000, 0x100, 0x7000, "stream buffer"});
	Record record;
	record.description = "page fault";
	VkDeviceFaultAddressInfoEXT address {};
	address.addressType      = VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT;
	address.reportedAddress  = 0x10050;
	address.addressPrecision = 0x40;
	record.addresses.push_back(address);
	address.reportedAddress  = 0x20000;
	address.addressPrecision = 0;
	record.addresses.push_back(address);
	address.reportedAddress = 0x100;
	record.addresses.push_back(address);
	VkDeviceFaultVendorInfoEXT vendor {};
	std::snprintf(vendor.description, sizeof(vendor.description), "vendor");
	vendor.vendorFaultCode = 0x12;
	vendor.vendorFaultData = 0x34;
	record.vendors.push_back(vendor);

	const auto text = Format(record, registry);
	std::printf("%s", text.c_str());
	Check(Contains(text, "Device fault: page fault (addresses=3, vendor=1)"), "header");
	Check(Contains(text, "WriteInvalid va=0x0000000000010050 in [0x0000000000010040, 0x000000000001007f]"),
	      "aligned precision block");
	Check(Contains(text, "stream buffer guest=0x0000000000007000 +0x50 of 0x100"), "match inside");
	Check(Contains(text, "0xff00 bytes past the end of stream buffer"), "match past the end");
	Check(Contains(text, "va=0x0000000000000100 in [0x0000000000000100, 0x0000000000000100]: no device-address buffer"),
	      "no match");
	Check(Contains(text, "vendor: \"vendor\" code=0x0000000000000012 data=0x0000000000000034"), "vendor");
}

// glslangValidator -V --target-env vulkan1.2:
// #version 460
// #extension GL_EXT_buffer_reference : require
// #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// layout(local_size_x = 64) in;
// layout(buffer_reference, std430) buffer Words { uint v[]; };
// layout(push_constant) uniform Push { uint64_t target; } pc;
// void main() { Words(pc.target).v[gl_GlobalInvocationID.x] = 0xdeadbeefu; }
constexpr uint32_t kOutOfBoundsWrite[] = {
	0x07230203,0x00010500,0x0008000b,0x00000021,0x00000000,0x00020011,0x00000001,0x00020011,
	0x0000000b,0x00020011,0x000014e3,0x0006000b,0x00000001,0x4c534c47,0x6474732e,0x3035342e,
	0x00000000,0x0003000e,0x000014e4,0x00000001,0x0007000f,0x00000005,0x00000004,0x6e69616d,
	0x00000000,0x00000009,0x00000016,0x00060010,0x00000004,0x00000011,0x00000040,0x00000001,
	0x00000001,0x00030003,0x00000002,0x000001cc,0x00070004,0x455f4c47,0x625f5458,0x65666675,
	0x65725f72,0x65726566,0x0065636e,0x000d0004,0x455f4c47,0x735f5458,0x65646168,0x78655f72,
	0x63696c70,0x615f7469,0x68746972,0x6974656d,0x79745f63,0x5f736570,0x36746e69,0x00000034,
	0x00040005,0x00000004,0x6e69616d,0x00000000,0x00040005,0x00000007,0x68737550,0x00000000,
	0x00050006,0x00000007,0x00000000,0x67726174,0x00007465,0x00030005,0x00000009,0x00006370,
	0x00040005,0x00000012,0x64726f57,0x00000073,0x00040006,0x00000012,0x00000000,0x00000076,
	0x00080005,0x00000016,0x475f6c67,0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,
	0x00030047,0x00000007,0x00000002,0x00050048,0x00000007,0x00000000,0x00000023,0x00000000,
	0x00040047,0x00000011,0x00000006,0x00000004,0x00030047,0x00000012,0x00000002,0x00050048,
	0x00000012,0x00000000,0x00000023,0x00000000,0x00040047,0x00000016,0x0000000b,0x0000001c,
	0x00040047,0x00000020,0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,0x00000003,
	0x00000002,0x00040015,0x00000006,0x00000040,0x00000000,0x0003001e,0x00000007,0x00000006,
	0x00040020,0x00000008,0x00000009,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000009,
	0x00040015,0x0000000a,0x00000020,0x00000001,0x0004002b,0x0000000a,0x0000000b,0x00000000,
	0x00040020,0x0000000c,0x00000009,0x00000006,0x00030027,0x0000000f,0x000014e5,0x00040015,
	0x00000010,0x00000020,0x00000000,0x0003001d,0x00000011,0x00000010,0x0003001e,0x00000012,
	0x00000011,0x00040020,0x0000000f,0x000014e5,0x00000012,0x00040017,0x00000014,0x00000010,
	0x00000003,0x00040020,0x00000015,0x00000001,0x00000014,0x0004003b,0x00000015,0x00000016,
	0x00000001,0x0004002b,0x00000010,0x00000017,0x00000000,0x00040020,0x00000018,0x00000001,
	0x00000010,0x0004002b,0x00000010,0x0000001b,0xdeadbeef,0x00040020,0x0000001c,0x000014e5,
	0x00000010,0x0004002b,0x00000010,0x0000001e,0x00000040,0x0004002b,0x00000010,0x0000001f,
	0x00000001,0x0006002c,0x00000014,0x00000020,0x0000001e,0x0000001f,0x0000001f,0x00050036,
	0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,0x0000000c,
	0x0000000d,0x00000009,0x0000000b,0x0004003d,0x00000006,0x0000000e,0x0000000d,0x00040078,
	0x0000000f,0x00000013,0x0000000e,0x00050041,0x00000018,0x00000019,0x00000016,0x00000017,
	0x0004003d,0x00000010,0x0000001a,0x00000019,0x00060041,0x0000001c,0x0000001d,0x00000013,
	0x0000000b,0x0000001a,0x0005003e,0x0000001d,0x0000001b,0x00000002,0x00000004,0x000100fd,
	0x00010038};

int Inconclusive(const char* why) {
	std::printf("INCONCLUSIVE: %s\n", why);
	return 2;
}

int ForceDeviceLost() {
	static vk::detail::DynamicLoader loader;
	const auto get_instance_proc_addr =
	    loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	if (get_instance_proc_addr == nullptr) {
		return Inconclusive("no Vulkan loader");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);
	vk::ApplicationInfo app {};
	app.pApplicationName = "device_fault_tests";
	app.apiVersion       = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info {};
	instance_info.pApplicationInfo = &app;
	vk::Instance instance;
	if (vk::createInstance(&instance_info, nullptr, &instance) != vk::Result::eSuccess) {
		return Inconclusive("vkCreateInstance failed");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

	uint32_t count = 0;
	(void)instance.enumeratePhysicalDevices(&count, nullptr);
	std::vector<vk::PhysicalDevice> physical_devices(count);
	(void)instance.enumeratePhysicalDevices(&count, physical_devices.data());
	vk::PhysicalDevice physical;
	uint32_t           family = 0;
	for (const auto candidate: physical_devices) {
		uint32_t extension_count = 0;
		(void)candidate.enumerateDeviceExtensionProperties(nullptr, &extension_count, nullptr);
		std::vector<vk::ExtensionProperties> extensions(extension_count);
		(void)candidate.enumerateDeviceExtensionProperties(nullptr, &extension_count, extensions.data());
		bool has_fault = false;
		for (const auto& extension: extensions) {
			has_fault |= std::strcmp(extension.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0;
		}
		vk::PhysicalDeviceFaultFeaturesEXT  fault {};
		vk::PhysicalDeviceVulkan12Features  features12 {};
		vk::PhysicalDeviceFeatures2         features {};
		features.pNext   = &features12;
		features12.pNext = has_fault ? &fault : nullptr;
		candidate.getFeatures2(&features);
		uint32_t family_count = 0;
		candidate.getQueueFamilyProperties(&family_count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(family_count);
		candidate.getQueueFamilyProperties(&family_count, families.data());
		for (uint32_t i = 0; i < family_count; i++) {
			if (has_fault && fault.deviceFault && features12.bufferDeviceAddress &&
			    (families[i].queueFlags & vk::QueueFlagBits::eCompute)) {
				physical = candidate;
				family   = i;
				break;
			}
		}
		if (physical) {
			break;
		}
	}
	if (!physical) {
		return Inconclusive("no device with VK_EXT_device_fault deviceFault and bufferDeviceAddress");
	}
	vk::PhysicalDeviceProperties properties {};
	physical.getProperties(&properties);
	std::printf("Device: %s\n", properties.deviceName.data());

	const float                priority = 1.0f;
	vk::DeviceQueueCreateInfo  queue_info {};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::PhysicalDeviceFaultFeaturesEXT fault {};
	fault.deviceFault = VK_TRUE;
	vk::PhysicalDeviceVulkan12Features features12 {};
	features12.pNext               = &fault;
	features12.bufferDeviceAddress = VK_TRUE;
	const char*          extension = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
	vk::DeviceCreateInfo device_info {};
	device_info.pNext                   = &features12;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = 1;
	device_info.ppEnabledExtensionNames = &extension;
	vk::Device device;
	if (physical.createDevice(&device_info, nullptr, &device) != vk::Result::eSuccess) {
		return Inconclusive("vkCreateDevice failed");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
	vk::Queue queue;
	device.getQueue(family, 0, &queue);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = 4096;
	buffer_info.usage = vk::BufferUsageFlagBits::eStorageBuffer |
	                    vk::BufferUsageFlagBits::eShaderDeviceAddress;
	vk::Buffer buffer;
	if (device.createBuffer(&buffer_info, nullptr, &buffer) != vk::Result::eSuccess) {
		return Inconclusive("vkCreateBuffer failed");
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(buffer, &requirements);
	vk::PhysicalDeviceMemoryProperties memory {};
	physical.getMemoryProperties(&memory);
	uint32_t type = 0;
	while (type < memory.memoryTypeCount &&
	       ((requirements.memoryTypeBits & (1u << type)) == 0 ||
	        !(memory.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))) {
		type++;
	}
	vk::MemoryAllocateFlagsInfo flags_info {};
	flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.pNext           = &flags_info;
	allocate_info.allocationSize  = requirements.size;
	allocate_info.memoryTypeIndex = type;
	vk::DeviceMemory buffer_memory;
	if (type == memory.memoryTypeCount ||
	    device.allocateMemory(&allocate_info, nullptr, &buffer_memory) != vk::Result::eSuccess ||
	    device.bindBufferMemory(buffer, buffer_memory, 0) != vk::Result::eSuccess) {
		return Inconclusive("buffer memory");
	}
	vk::BufferDeviceAddressInfo address_info {};
	address_info.buffer          = buffer;
	const uint64_t buffer_address = device.getBufferAddress(&address_info);
	const uint64_t target         = buffer_address + (uint64_t {1} << 36u);
	AddressRegistry registry;
	registry.Add({buffer_address, 4096, 0, "test buffer"});
	std::printf("Buffer at 0x%016" PRIx64 ", writing 256 bytes at 0x%016" PRIx64 "\n", buffer_address,
	            target);

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(kOutOfBoundsWrite);
	module_info.pCode    = kOutOfBoundsWrite;
	vk::ShaderModule module;
	vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint64_t)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	vk::PipelineLayout layout;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess ||
	    device.createPipelineLayout(&layout_info, nullptr, &layout) != vk::Result::eSuccess) {
		return Inconclusive("shader module or layout");
	}
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = layout;
	vk::Pipeline pipeline;
	if (device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline) !=
	    vk::Result::eSuccess) {
		return Inconclusive("compute pipeline");
	}

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = family;
	vk::CommandPool pool;
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	vk::CommandBufferBeginInfo begin {};
	if (device.createCommandPool(&pool_info, nullptr, &pool) != vk::Result::eSuccess) {
		return Inconclusive("command pool");
	}
	command_info.commandPool = pool;
	if (device.allocateCommandBuffers(&command_info, &command) != vk::Result::eSuccess ||
	    command.begin(&begin) != vk::Result::eSuccess) {
		return Inconclusive("command buffer");
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(target), &target);
	command.dispatch(1, 1, 1);
	if (command.end() != vk::Result::eSuccess) {
		return Inconclusive("vkEndCommandBuffer failed");
	}
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence;
	(void)device.createFence(&fence_info, nullptr, &fence);
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	auto result               = queue.submit(1, &submit, fence);
	if (result == vk::Result::eSuccess) {
		result = device.waitForFences(1, &fence, VK_TRUE, 10'000'000'000ull);
	}
	std::printf("Result: %s\n", vk::to_string(result).c_str());
	if (result != vk::Result::eErrorDeviceLost) {
		return Inconclusive("the out-of-bounds write did not lose the device");
	}

	Record record;
	if (!Query(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT, device, record)) {
		std::printf("FAIL: device lost but no readable fault record\n");
		return 1;
	}
	std::printf("%s", Format(record, registry).c_str());
	bool found = false;
	for (const auto& address: record.addresses) {
		const uint64_t mask = address.addressPrecision > 1 ? address.addressPrecision - 1 : 0;
		const uint64_t low  = address.reportedAddress & ~mask;
		const uint64_t high = address.reportedAddress | mask;
		found |= low < target + 256 && high >= target;
	}
	std::printf("%s: fault address %s the written range\n", found ? "OK" : "FAIL",
	            found ? "covers" : "misses");
	return found ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "--force-device-lost") == 0) {
		return ForceDeviceLost();
	}
	TestRegistry();
	TestFormat();
	std::printf("%s (%d failures)\n", g_failures == 0 ? "OK" : "FAIL", g_failures);
	return g_failures == 0 ? 0 : 1;
}
