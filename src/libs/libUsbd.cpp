#include "common/abi.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/usbHost.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs {

LIB_VERSION("Usbd", 1, "Usbd", 1, 1);

namespace Usbd {

namespace Host = UsbHost;
using Host::Device;

constexpr uint32_t POLL_INTERVAL_MS = 16;

// KYTY_USB_PASSTHRU=VVVV:PPPP[,VVVV:PPPP]: exposes those WinUSB devices to the guest (libusb
// semantics; the guest's device and handle pointers are both the host Device). Unset = no device.
struct Options {
	std::vector<std::pair<uint16_t, uint16_t>> ids;
	bool                                       log = false;
};

static const Options& Opts() {
	static const Options opts = [] {
		Options     o;
		const char* text = std::getenv("KYTY_USB_PASSTHRU");
		const char* log  = std::getenv("KYTY_USB_LOG");
		o.log            = log != nullptr && log[0] == '1';
		if (text != nullptr && !Host::ParseIds(text, &o.ids)) {
			LOGF("KYTY_USB_PASSTHRU: bad list '%s' (want VVVV:PPPP[,VVVV:PPPP])\n", text);
		}
		return o;
	}();
	return opts;
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define USB_LOG(...)                                                                               \
	do {                                                                                           \
		if (Opts().log) {                                                                          \
			LOGF("KYTY_USB " __VA_ARGS__);                                                         \
		}                                                                                          \
	} while (false)

static bool g_have_devices = false;

// sceUsbd errors are 0x802400xx with xx = -(libusb error); 0xff for the rest.
static int Sce(int result) {
	if (result >= 0) {
		return result;
	}
	const uint32_t code = (-result <= 0x0c) ? static_cast<uint32_t>(-result) : 0xffu;
	return static_cast<int>(0x80240000u | code);
}

// libusb_transfer, x86-64 layout.
struct Transfer;
using TransferCallback = KYTY_SYSV_ABI void (*)(Transfer*);
struct Transfer {
	Device*          dev_handle;
	uint8_t          flags;
	uint8_t          endpoint;
	uint8_t          type;
	uint32_t         timeout;
	int32_t          status;
	int32_t          length;
	int32_t          actual_length;
	TransferCallback callback;
	void*            user_data;
	uint8_t*         buffer;
	int32_t          num_iso_packets;
};
static_assert(offsetof(Transfer, timeout) == 12 && offsetof(Transfer, callback) == 32 &&
              offsetof(Transfer, num_iso_packets) == 56);
constexpr size_t  TRANSFER_HEADER_SIZE        = 60; // iso_packet_desc[] starts here, 12 bytes each
constexpr uint8_t TRANSFER_TYPE_CONTROL       = 0;
constexpr uint8_t TRANSFER_TYPE_ISO           = 1;
constexpr uint8_t TRANSFER_TYPE_INTERRUPT     = 3;
constexpr uint8_t TRANSFER_FLAG_SHORT_NOT_OK  = 1;
constexpr uint8_t TRANSFER_FLAG_FREE_TRANSFER = 4;

static int KYTY_SYSV_ABI UsbdInit() {
	PRINT_NAME();

	return OK;
}

static int KYTY_SYSV_ABI UsbdExit() {
	PRINT_NAME();

	return OK;
}

struct TimeVal {
	int64_t tv_sec;
	int64_t tv_usec;
};

static int KYTY_SYSV_ABI UsbdHandleEventsTimeout(const TimeVal* tv) {
	PRINT_NAME();

	if (!g_have_devices) {
		// Avoid a busy loop while no device is passed through.
		Common::Thread::SleepMicro(POLL_INTERVAL_MS * 1000);
		return OK;
	}
	const uint32_t timeout_ms = tv != nullptr
	                                ? static_cast<uint32_t>(tv->tv_sec * 1000 + tv->tv_usec / 1000)
	                                : POLL_INTERVAL_MS;
	std::vector<Host::Completion> done;
	Host::Wait(timeout_ms, &done);
	for (const auto& c: done) {
		auto* t          = static_cast<Transfer*>(c.tag);
		t->actual_length = c.actual;
		t->status        = (c.status == Host::USB_TRANSFER_COMPLETED &&
		                    (t->flags & TRANSFER_FLAG_SHORT_NOT_OK) != 0 && c.actual < t->length)
		                       ? Host::USB_TRANSFER_ERROR
		                       : c.status;
		USB_LOG("transfer done %p ep=%02x status=%d actual=%d/%d\n", static_cast<void*>(t),
		        t->endpoint, t->status, t->actual_length, t->length);
		const bool free_after = (t->flags & TRANSFER_FLAG_FREE_TRANSFER) != 0;
		if (t->callback != nullptr) {
			t->callback(t);
		}
		if (free_after) {
			std::free(t);
		}
	}
	return OK;
}

static int KYTY_SYSV_ABI UsbdEventHandlingOk() {
	PRINT_NAME();

	return 1;
}

static int64_t KYTY_SYSV_ABI UsbdGetDeviceList(Device*** list) {
	PRINT_NAME();

	if (Opts().ids.empty()) {
		return 0;
	}
	// Enumerated once: walking every USB devnode per call could hitch a polling game (no hot-plug).
	static const std::vector<Device*> devices = [] {
		std::string diagnostics;
		auto        found = Host::List(Opts().ids, &diagnostics);
		LOGF("KYTY_USB_PASSTHRU: %zu device(s)\n%s", found.size(), diagnostics.c_str());
		g_have_devices = !found.empty();
		return found;
	}();
	auto* out = new Device*[devices.size() + 1];
	std::copy(devices.begin(), devices.end(), out);
	out[devices.size()] = nullptr;
	if (list != nullptr) {
		*list = out;
	}
	USB_LOG("GetDeviceList -> %zu\n", devices.size());
	return static_cast<int64_t>(devices.size());
}

static void KYTY_SYSV_ABI UsbdFreeDeviceList(Device** list, int /*unref_devices*/) {
	PRINT_NAME();

	if (!Opts().ids.empty()) {
		delete[] list; // option off: GetDeviceList never wrote the guest's list
	}
}

// Devices live for the whole process, so references are not counted.
static Device* KYTY_SYSV_ABI UsbdRefDevice(Device* dev) {
	USB_LOG("RefDevice %p\n", static_cast<void*>(dev));
	return dev;
}

static void KYTY_SYSV_ABI UsbdUnrefDevice(Device* dev) {
	USB_LOG("UnrefDevice %p\n", static_cast<void*>(dev));
}

static int KYTY_SYSV_ABI UsbdGetDeviceDescriptor(Device* dev, uint8_t* desc) {
	if (dev == nullptr || desc == nullptr) {
		return Sce(Host::USB_ERROR_INVALID_PARAM);
	}
	std::memcpy(desc, Host::DeviceDescriptor(dev), 18);
	USB_LOG("GetDeviceDescriptor %p -> %02x%02x:%02x%02x\n", static_cast<void*>(dev), desc[9],
	        desc[8], desc[11], desc[10]);
	return OK;
}

static int KYTY_SYSV_ABI UsbdGetActiveConfigDescriptor(Device*                  dev,
                                                       Host::ConfigDescriptor** config) {
	if (dev == nullptr || config == nullptr) {
		return Sce(Host::USB_ERROR_INVALID_PARAM);
	}
	const auto& raw = Host::RawConfig(dev);
	*config         = Host::ParseConfig(raw.data(), raw.size());
	USB_LOG("GetActiveConfigDescriptor %p -> %p (%zu bytes)\n", static_cast<void*>(dev),
	        static_cast<void*>(*config), raw.size());
	return *config != nullptr ? OK : Sce(Host::USB_ERROR_IO);
}

// shortcut: only configuration index 0 (the active one under WinUSB) is known.
static int KYTY_SYSV_ABI UsbdGetConfigDescriptor(Device* dev, uint8_t index,
                                                 Host::ConfigDescriptor** config) {
	USB_LOG("GetConfigDescriptor %p index=%u\n", static_cast<void*>(dev), index);
	return index == 0 ? UsbdGetActiveConfigDescriptor(dev, config) : Sce(Host::USB_ERROR_NOT_FOUND);
}

static void KYTY_SYSV_ABI UsbdFreeConfigDescriptor(Host::ConfigDescriptor* config) {
	USB_LOG("FreeConfigDescriptor %p\n", static_cast<void*>(config));
	Host::FreeConfig(config);
}

static uint8_t KYTY_SYSV_ABI UsbdGetBusNumber(Device* dev) {
	USB_LOG("GetBusNumber %p\n", static_cast<void*>(dev));
	return 1;
}

static uint8_t KYTY_SYSV_ABI UsbdGetDeviceAddress(Device* dev) {
	USB_LOG("GetDeviceAddress %p\n", static_cast<void*>(dev));
	return 1;
}

static int KYTY_SYSV_ABI UsbdOpen(Device* dev, Device** handle) {
	USB_LOG("Open %p\n", static_cast<void*>(dev));
	if (dev == nullptr || handle == nullptr) {
		return Sce(Host::USB_ERROR_INVALID_PARAM);
	}
	*handle = dev;
	return OK;
}

static void KYTY_SYSV_ABI UsbdClose(Device* handle) {
	USB_LOG("Close %p\n", static_cast<void*>(handle));
}

static int KYTY_SYSV_ABI UsbdCheckConnected(Device* handle) {
	const bool connected = handle != nullptr && Host::Connected(handle);
	USB_LOG("CheckConnected %p -> %d\n", static_cast<void*>(handle), connected ? 1 : 0);
	return connected ? OK : Sce(Host::USB_ERROR_NO_DEVICE);
}

static int KYTY_SYSV_ABI UsbdSetConfiguration(Device* handle, int config) {
	USB_LOG("SetConfiguration %p %d\n", static_cast<void*>(handle), config);
	const auto& raw = Host::RawConfig(handle);
	return raw.size() > 5 && (config == raw[5] || config == -1)
	           ? OK
	           : Sce(Host::USB_ERROR_NOT_SUPPORTED);
}

// Only interfaces bound to WinUSB (Zadig) can be claimed; the others stay with their Windows
// driver.
static int KYTY_SYSV_ABI UsbdClaimInterface(Device* handle, int number) {
	const bool ok = handle != nullptr && Host::HasInterface(handle, number);
	if (!ok) {
		LOGF("KYTY_USB ClaimInterface %d: interface not on WinUSB\n", number);
	}
	USB_LOG("ClaimInterface %p %d -> %d\n", static_cast<void*>(handle), number, ok ? 1 : 0);
	return ok ? OK : Sce(Host::USB_ERROR_BUSY);
}

static int KYTY_SYSV_ABI UsbdReleaseInterface(Device* handle, int number) {
	USB_LOG("ReleaseInterface %p %d\n", static_cast<void*>(handle), number);
	return OK;
}

static int KYTY_SYSV_ABI UsbdKernelDriverActive(Device* handle, int number) {
	USB_LOG("KernelDriverActive %p %d\n", static_cast<void*>(handle), number);
	return 0;
}

static int KYTY_SYSV_ABI UsbdAttachKernelDriver(Device* handle, int number) {
	USB_LOG("AttachKernelDriver %p %d\n", static_cast<void*>(handle), number);
	return OK;
}

static int KYTY_SYSV_ABI UsbdResetDevice(Device* handle) {
	USB_LOG("ResetDevice %p (ignored)\n", static_cast<void*>(handle));
	return OK;
}

static int KYTY_SYSV_ABI UsbdGetStringDescriptor(Device* handle, uint8_t index, uint16_t lang,
                                                 uint8_t* data, int length) {
	const int r = Host::GetStringDescriptor(handle, index, lang, data, length);
	USB_LOG("GetStringDescriptor %p index=%u lang=%04x len=%d -> %d\n", static_cast<void*>(handle),
	        index, lang, length, r);
	return Sce(r);
}

static int KYTY_SYSV_ABI UsbdControlTransfer(Device* handle, uint8_t request_type, uint8_t request,
                                             uint16_t value, uint16_t index, uint8_t* data,
                                             uint16_t length, uint32_t timeout) {
	const int r = Host::Control(handle, request_type, request, value, index, data, length, timeout);
	USB_LOG("ControlTransfer %p rt=%02x req=%02x val=%04x idx=%04x len=%u to=%u -> %d\n",
	        static_cast<void*>(handle), request_type, request, value, index, length, timeout, r);
	return Sce(r);
}

static Transfer* KYTY_SYSV_ABI UsbdAllocTransfer(int iso_packets) {
	const size_t size =
	    TRANSFER_HEADER_SIZE + 12u * static_cast<size_t>(iso_packets < 0 ? 0 : iso_packets);
	auto* t = static_cast<Transfer*>(std::calloc(1, std::max(size, sizeof(Transfer))));
	if (t != nullptr) {
		t->num_iso_packets = iso_packets;
	}
	USB_LOG("AllocTransfer %d -> %p\n", iso_packets, static_cast<void*>(t));
	return t;
}

static void KYTY_SYSV_ABI UsbdFreeTransfer(Transfer* t) {
	USB_LOG("FreeTransfer %p\n", static_cast<void*>(t));
	std::free(t);
}

static void KYTY_SYSV_ABI UsbdFillInterruptTransfer(Transfer* t, Device* handle, uint8_t endpoint,
                                                    uint8_t* buffer, int length,
                                                    TransferCallback callback, void* user_data,
                                                    uint32_t timeout) {
	USB_LOG("FillInterruptTransfer %p ep=%02x len=%d to=%u\n", static_cast<void*>(t), endpoint,
	        length, timeout);
	t->dev_handle = handle;
	t->endpoint   = endpoint;
	t->type       = TRANSFER_TYPE_INTERRUPT;
	t->timeout    = timeout;
	t->buffer     = buffer;
	t->length     = length;
	t->user_data  = user_data;
	t->callback   = callback;
}

// shortcut: interrupt and bulk only (the guest library fills nothing else); control/iso refused.
static int KYTY_SYSV_ABI UsbdSubmitTransfer(Transfer* t) {
	int r = Host::USB_ERROR_NOT_SUPPORTED;
	if (t != nullptr && t->dev_handle != nullptr && t->type != TRANSFER_TYPE_CONTROL &&
	    t->type != TRANSFER_TYPE_ISO) {
		r = Host::Submit(t->dev_handle, t->endpoint, t->buffer, t->length, t->timeout, t);
	}
	USB_LOG("SubmitTransfer %p type=%u ep=%02x len=%d -> %d\n", static_cast<void*>(t),
	        t != nullptr ? t->type : 0, t != nullptr ? t->endpoint : 0,
	        t != nullptr ? t->length : 0, r);
	return Sce(r);
}

static int KYTY_SYSV_ABI UsbdCancelTransfer(Transfer* t) {
	USB_LOG("CancelTransfer %p\n", static_cast<void*>(t));
	if (t == nullptr || t->dev_handle == nullptr) {
		return Sce(Host::USB_ERROR_NOT_FOUND);
	}
	Host::Cancel(t->dev_handle, t->endpoint);
	return OK;
}

} // namespace Usbd

LIB_DEFINE(InitUsbd_1) {
	LIB_FUNC("TOhg7P6kTH4", Usbd::UsbdInit);
	LIB_FUNC("Fq6+0Fm55xU", Usbd::UsbdExit);
	LIB_FUNC("+wU6CGuZcWk", Usbd::UsbdHandleEventsTimeout);
	LIB_FUNC("8qB9Ar4P5nc", Usbd::UsbdGetDeviceList);
	LIB_FUNC("EQ6SCLMqzkM", Usbd::UsbdFreeDeviceList);
	if (Usbd::Opts().ids.empty()) {
		return; // option off: the rest stays stubbed, as before KYTY_USB_PASSTHRU
	}
	LIB_FUNC("e7gp1xhu6RI", Usbd::UsbdEventHandlingOk);
	LIB_FUNC("U1t1SoJvV-A", Usbd::UsbdRefDevice);
	LIB_FUNC("OULgIo1zAsA", Usbd::UsbdUnrefDevice);
	LIB_FUNC("bhomgbiQgeo", Usbd::UsbdGetDeviceDescriptor);
	LIB_FUNC("S1o1C6yOt5g", Usbd::UsbdGetActiveConfigDescriptor);
	LIB_FUNC("Dkm5qe8j3XE", Usbd::UsbdGetConfigDescriptor);
	LIB_FUNC("Hvd3S--n25w", Usbd::UsbdFreeConfigDescriptor);
	LIB_FUNC("t7WE9mb1TB8", Usbd::UsbdGetBusNumber);
	LIB_FUNC("GjlCrU4GcIY", Usbd::UsbdGetDeviceAddress);
	LIB_FUNC("VJ6oMq-Di2U", Usbd::UsbdOpen);
	LIB_FUNC("HarYYlaFGJY", Usbd::UsbdClose);
	LIB_FUNC("MlW6deWfPp0", Usbd::UsbdCheckConnected);
	LIB_FUNC("FhU9oYrbXoA", Usbd::UsbdSetConfiguration);
	LIB_FUNC("AE+mHBHneyk", Usbd::UsbdClaimInterface);
	LIB_FUNC("REfUTmTchMw", Usbd::UsbdReleaseInterface);
	LIB_FUNC("RLf56F-WjKQ", Usbd::UsbdKernelDriverActive);
	LIB_FUNC("BKMEGvfCPyU", Usbd::UsbdAttachKernelDriver);
	LIB_FUNC("hvMn0QJXj5g", Usbd::UsbdResetDevice);
	LIB_FUNC("g2oYm1DitDg", Usbd::UsbdGetStringDescriptor);
	LIB_FUNC("RRKFcKQ1Ka4", Usbd::UsbdControlTransfer);
	LIB_FUNC("0ktE1PhzGFU", Usbd::UsbdAllocTransfer);
	LIB_FUNC("-sgi7EeLSO8", Usbd::UsbdFreeTransfer);
	LIB_FUNC("t3J5pXxhJlI", Usbd::UsbdFillInterruptTransfer);
	LIB_FUNC("L0EHgZZNVas", Usbd::UsbdSubmitTransfer);
	LIB_FUNC("-KNh1VFIzlM", Usbd::UsbdCancelTransfer);
}

} // namespace Libs
