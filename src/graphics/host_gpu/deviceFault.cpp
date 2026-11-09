#include "graphics/host_gpu/deviceFault.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::DeviceFault {

void AddressRegistry::Add(const Range& range) {
	std::lock_guard lock(m_mutex);
	m_live[range.begin] = range;
}

void AddressRegistry::Remove(uint64_t begin) {
	std::lock_guard lock(m_mutex);
	const auto      it = m_live.find(begin);
	if (it == m_live.end()) {
		return;
	}
	it->second.freed           = true;
	m_freed[m_freed_next]      = it->second;
	m_freed_next               = (m_freed_next + 1) % FREED_KEPT;
	m_live.erase(it);
}

std::optional<AddressRegistry::Range> AddressRegistry::Find(uint64_t address) const {
	std::lock_guard lock(m_mutex);
	auto            it = m_live.upper_bound(address);
	if (it != m_live.begin()) {
		--it;
		if (address - it->first < it->second.size) {
			return it->second;
		}
	}
	for (size_t i = 1; i <= FREED_KEPT; i++) {
		const auto& freed = m_freed[(m_freed_next + FREED_KEPT - i) % FREED_KEPT];
		if (freed.size != 0 && address >= freed.begin && address - freed.begin < freed.size) {
			return freed;
		}
	}
	if (it != m_live.end() && it->first <= address) {
		return it->second;
	}
	return std::nullopt;
}

AddressRegistry& Registry() {
	static AddressRegistry registry;
	return registry;
}

bool Query(PFN_vkGetDeviceFaultInfoEXT query, VkDevice device, Record& record) {
	if (query == nullptr || device == VK_NULL_HANDLE) {
		return false;
	}
	VkDeviceFaultCountsEXT counts {};
	counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
	if (query(device, &counts, nullptr) != VK_SUCCESS) {
		return false;
	}
	record.addresses.resize(counts.addressInfoCount);
	record.vendors.resize(counts.vendorInfoCount);
	// Vendor binary data is not requested: the feature is never enabled (the AMD driver lacks it).
	counts.vendorBinarySize = 0;
	VkDeviceFaultInfoEXT info {};
	info.sType         = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
	info.pAddressInfos = record.addresses.empty() ? nullptr : record.addresses.data();
	info.pVendorInfos  = record.vendors.empty() ? nullptr : record.vendors.data();
	const auto result  = query(device, &counts, &info);
	if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
		return false;
	}
	record.addresses.resize(counts.addressInfoCount);
	record.vendors.resize(counts.vendorInfoCount);
	record.description.assign(info.description, strnlen(info.description, VK_MAX_DESCRIPTION_SIZE));
	return true;
}

namespace {

std::string Match(uint64_t address, const AddressRegistry& registry) {
	const auto range = registry.Find(address);
	if (!range) {
		return "no device-address buffer";
	}
	char text[192];
	const auto offset = address - range->begin;
	if (offset < range->size) {
		std::snprintf(text, sizeof(text), "%s%s guest=0x%016" PRIx64 " +0x%" PRIx64 " of 0x%" PRIx64,
		              range->freed ? "freed " : "", range->kind, range->guest, offset, range->size);
	} else {
		std::snprintf(text, sizeof(text), "0x%" PRIx64 " bytes past the end of %s guest=0x%016" PRIx64
		              " size 0x%" PRIx64, offset - range->size, range->kind, range->guest, range->size);
	}
	return text;
}

} // namespace

std::string Format(const Record& record, const AddressRegistry& registry) {
	std::string text = "Device fault: " + record.description + " (addresses=" +
	                   std::to_string(record.addresses.size()) +
	                   ", vendor=" + std::to_string(record.vendors.size()) + ")\n";
	char line[512];
	for (const auto& address: record.addresses) {
		// The fault lies in the naturally aligned power-of-two block around the reported address.
		const uint64_t mask = address.addressPrecision > 1 ? address.addressPrecision - 1 : 0;
		std::snprintf(line, sizeof(line),
		              "\tdevice fault address: %s va=0x%016" PRIx64 " in [0x%016" PRIx64
		              ", 0x%016" PRIx64 "]: %s\n",
		              vk::to_string(static_cast<vk::DeviceFaultAddressTypeEXT>(address.addressType)).c_str(),
		              address.reportedAddress, address.reportedAddress & ~mask,
		              address.reportedAddress | mask, Match(address.reportedAddress, registry).c_str());
		text += line;
	}
	for (const auto& vendor: record.vendors) {
		std::snprintf(line, sizeof(line),
		              "\tdevice fault vendor: \"%.*s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n",
		              static_cast<int>(strnlen(vendor.description, VK_MAX_DESCRIPTION_SIZE)),
		              vendor.description, vendor.vendorFaultCode, vendor.vendorFaultData);
		text += line;
	}
	return text;
}

bool RequestedByEnvironment() {
	const char* value = std::getenv("KYTY_DEVICE_FAULT");
	return value == nullptr || std::strcmp(value, "0") != 0;
}

} // namespace Libs::Graphics::DeviceFault
