#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// VK_EXT_device_fault: after a device loss, what the driver recorded (faulting GPU virtual
// addresses, vendor codes), with each address named after the device-address buffer around it.
namespace Libs::Graphics::DeviceFault {

// Device-address ranges of live and recently destroyed buffers.
class AddressRegistry {
public:
	struct Range {
		uint64_t    begin = 0;
		uint64_t    size  = 0;
		uint64_t    guest = 0; // Guest address the buffer mirrors, 0 for emulator-owned buffers.
		const char* kind  = "";
		bool        freed = false;
	};

	void Add(const Range& range);
	// Keeps the range as freed: a fault on a destroyed buffer is a likely cause of device loss.
	void Remove(uint64_t begin);
	// The live range holding address, else the most recently freed one holding it, else the
	// closest live range below it (an overrun).
	[[nodiscard]] std::optional<Range> Find(uint64_t address) const;

private:
	static constexpr size_t FREED_KEPT = 64;

	mutable std::mutex              m_mutex;
	std::map<uint64_t, Range>       m_live;
	std::array<Range, FREED_KEPT>   m_freed {};
	size_t                          m_freed_next = 0;
};

AddressRegistry& Registry();

struct Record {
	std::string                              description;
	std::vector<VkDeviceFaultAddressInfoEXT> addresses;
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors;
};

// Reads the fault record of a lost device; false when the driver recorded nothing readable.
[[nodiscard]] bool Query(PFN_vkGetDeviceFaultInfoEXT query, VkDevice device, Record& record);

// One line per record entry, addresses matched against registry.
[[nodiscard]] std::string Format(const Record& record, const AddressRegistry& registry);

// KYTY_DEVICE_FAULT: "0" turns the extension off; on by default (it costs nothing until a loss).
[[nodiscard]] bool RequestedByEnvironment();

} // namespace Libs::Graphics::DeviceFault

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICEFAULT_H_ */
