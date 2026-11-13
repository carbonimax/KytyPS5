#include "common/abi.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/controller.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/usbHost.h"
#include "libs/usbPad.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

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
			o.ids.clear(); // a half-parsed list must not turn the option on
		}
#if !defined(_WIN32)
		if (!o.ids.empty()) {
			LOGF("KYTY_USB_PASSTHRU: not supported on this platform (WinUSB only), option off\n");
			o.ids.clear();
		}
#endif
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
	USB_LOG("Init\n");

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
	const uint32_t  timeout_ms = tv != nullptr
	                                 ? static_cast<uint32_t>(tv->tv_sec * 1000 + tv->tv_usec / 1000)
	                                 : POLL_INTERVAL_MS;
	static uint32_t last_timeout_ms = UINT32_MAX; // logged on change: a 0 here spins the caller
	if (timeout_ms != last_timeout_ms) {
		last_timeout_ms = timeout_ms;
		USB_LOG("HandleEventsTimeout timeout=%u ms\n", timeout_ms);
	}
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
	USB_LOG("EventHandlingOk\n");

	return 1;
}

// Enumerated once: walking every USB devnode per call could hitch a polling game (no hot-plug).
static const std::vector<Device*>& Devices() {
	static const std::vector<Device*> devices = [] {
		std::string diagnostics;
		auto        found = Host::List(Opts().ids, &diagnostics);
		LOGF("KYTY_USB_PASSTHRU: %zu device(s)\n%s", found.size(), diagnostics.c_str());
		g_have_devices = !found.empty();
		return found;
	}();
	return devices;
}

static int64_t KYTY_SYSV_ABI UsbdGetDeviceList(Device*** list) {
	PRINT_NAME();

	if (Opts().ids.empty()) {
		return 0;
	}
	const auto& devices = Devices();
	auto*       out     = new Device*[devices.size() + 1];
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

namespace UsbPad {

namespace Host = UsbHost;
using Host::Device;
using Usbd::Opts;

constexpr int PAD_DEVICE_CLASS_STEERING_WHEEL = 6;

// The ext pad: the first passed-through device matching an scePadOpenExt filter. Its input comes
// from the interrupt IN endpoint of the device's first interface (its main HID one).
// shortcut: one ext pad; a second wheel would need a table of these.
struct WheelPad {
	Device*    dev              = nullptr;
	uint16_t   vid              = 0;
	uint16_t   pid              = 0;
	uint8_t    interface_number = 0;
	uint8_t    in_endpoint      = 0;
	uint8_t    out_endpoint     = 0;
	int        in_size          = 64;
	std::mutex mutex;
	uint8_t    report[64] {}; // latest input report (the device only reports changes)
};
static WheelPad g_pad;

bool Enabled() {
	return !Opts().ids.empty();
}

bool IsHandle(int handle) {
	return handle == HANDLE && g_pad.dev != nullptr;
}

static void ReadReports() {
	uint8_t buf[64] {};
	for (;;) {
		int       actual = 0;
		const int r =
		    Host::Transfer(g_pad.dev, g_pad.in_endpoint, buf, g_pad.in_size, 100, &actual);
		if (r == Host::USB_ERROR_NO_DEVICE) {
			LOGF("KYTY_USB pad: device lost\n");
			return;
		}
		if (r == Host::USB_OK && actual > 0) {
			std::lock_guard lock(g_pad.mutex);
			std::memcpy(g_pad.report, buf, static_cast<size_t>(std::min(actual, 64)));
		} else if (r != Host::USB_ERROR_TIMEOUT) {
			Common::Thread::SleepMicro(10000); // a failing pipe must not spin
		}
	}
}

static bool Attach(Device* dev) {
	const auto& raw    = Host::RawConfig(dev);
	auto*       config = Host::ParseConfig(raw.data(), raw.size());
	if (config == nullptr || config->bNumInterfaces == 0 ||
	    config->interface[0].num_altsetting == 0) {
		Host::FreeConfig(config);
		return false;
	}
	const auto& itf        = config->interface[0].altsetting[0];
	g_pad.interface_number = itf.bInterfaceNumber;
	for (int e = 0; e < itf.bNumEndpoints; e++) {
		const auto& ep = itf.endpoint[e];
		if ((ep.bmAttributes & 3u) != 3) {
			continue;
		}
		if ((ep.bEndpointAddress & 0x80u) != 0) {
			g_pad.in_endpoint = ep.bEndpointAddress;
			g_pad.in_size     = std::min<int>(ep.wMaxPacketSize, 64);
		} else {
			g_pad.out_endpoint = ep.bEndpointAddress;
		}
	}
	Host::FreeConfig(config);
	if (g_pad.in_endpoint == 0) {
		return false;
	}
	// Until the first report: sticks and steering centred, d-pad neutral, pedals released.
	std::memset(g_pad.report + 1, 0x80, 4);
	g_pad.report[8]  = 0x08;
	g_pad.report[41] = 0x80;
	std::memset(g_pad.report + 42, 0xff, 6);
	g_pad.dev = dev;
	std::thread(ReadReports).detach();
	return true;
}

int OpenExt(int type, const void* param) {
	if (!Enabled() || type != 2 || param == nullptr) {
		return Controller::PAD_ERROR_DEVICE_NOT_CONNECTED;
	}
	uint16_t filter[3]; // vendor id, first and last product id
	std::memcpy(filter, param, sizeof(filter));
	auto matches = [&](uint16_t vid, uint16_t pid) {
		return vid == filter[0] && pid >= filter[1] && pid <= filter[2];
	};
	static std::mutex open_mutex;
	std::lock_guard   lock(open_mutex);
	for (Device* dev: Usbd::Devices()) {
		if (g_pad.dev != nullptr) {
			break;
		}
		const uint8_t* dd  = Host::DeviceDescriptor(dev);
		const auto     vid = static_cast<uint16_t>(dd[8] | (dd[9] << 8u));
		const auto     pid = static_cast<uint16_t>(dd[10] | (dd[11] << 8u));
		if (matches(vid, pid) && Attach(dev)) {
			g_pad.vid = vid;
			g_pad.pid = pid;
			LOGF("KYTY_USB pad: %04x:%04x as ext pad, interface %u, in %02x out %02x\n", vid, pid,
			     g_pad.interface_number, g_pad.in_endpoint, g_pad.out_endpoint);
			break;
		}
	}
	const bool ok     = g_pad.dev != nullptr && matches(g_pad.vid, g_pad.pid);
	static int logged = 0;
	if (ok || logged++ < 4) {
		USB_LOG("PadOpenExt type=%d filter=%04x:%04x-%04x -> %d\n", type, filter[0], filter[1],
		        filter[2], ok ? HANDLE : -1);
	}
	return ok ? HANDLE : Controller::PAD_ERROR_DEVICE_NOT_CONNECTED;
}

int Close(int handle) {
	USB_LOG("PadClose %d\n", handle);
	return OK;
}

// shortcut: DualSense USB layout assumed for bytes 1-10 (only the neutral hat byte is confirmed);
// check the reports logged under KYTY_USB_LOG if a button is wrong.
static uint32_t Buttons(const uint8_t* r) {
	namespace C                  = Controller;
	static const uint32_t hat[8] = {C::PAD_BUTTON_UP,    C::PAD_BUTTON_UP | C::PAD_BUTTON_RIGHT,
	                                C::PAD_BUTTON_RIGHT, C::PAD_BUTTON_RIGHT | C::PAD_BUTTON_DOWN,
	                                C::PAD_BUTTON_DOWN,  C::PAD_BUTTON_DOWN | C::PAD_BUTTON_LEFT,
	                                C::PAD_BUTTON_LEFT,  C::PAD_BUTTON_LEFT | C::PAD_BUTTON_UP};
	static const uint32_t bits[3][8] = {{0, 0, 0, 0, C::PAD_BUTTON_SQUARE, C::PAD_BUTTON_CROSS,
	                                     C::PAD_BUTTON_CIRCLE, C::PAD_BUTTON_TRIANGLE},
	                                    {C::PAD_BUTTON_L1, C::PAD_BUTTON_R1, C::PAD_BUTTON_L2,
	                                     C::PAD_BUTTON_R2, 0, C::PAD_BUTTON_OPTIONS,
	                                     C::PAD_BUTTON_L3, C::PAD_BUTTON_R3},
	                                    {0, C::PAD_BUTTON_TOUCH_PAD, 0, 0, 0, 0, 0, 0}};
	uint32_t              buttons    = (r[8] & 0x0fu) < 8 ? hat[r[8] & 0x0fu] : 0;
	for (int byte = 0; byte < 3; byte++) {
		for (int bit = 0; bit < 8; bit++) {
			buttons |= (r[8 + byte] >> bit & 1u) != 0 ? bits[byte][bit] : 0;
		}
	}
	return buttons;
}

int Read(int handle, Controller::PadData* data, int num) {
	uint8_t r[64];
	{
		std::lock_guard lock(g_pad.mutex);
		std::memcpy(r, g_pad.report, sizeof(r));
	}
	std::memset(data, 0, sizeof(*data) * static_cast<size_t>(num < 1 ? 1 : num));
	auto& d             = data[0];
	d.buttons           = Buttons(r);
	d.left_stick_x      = r[1];
	d.left_stick_y      = r[2];
	d.right_stick_x     = r[3];
	d.right_stick_y     = r[4];
	d.analog_buttons_l2 = r[5];
	d.analog_buttons_r2 = r[6];
	d.orientation_w     = 1.0f;
	d.connected         = true;
	d.connected_count   = 1;
	d.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                                        std::chrono::steady_clock::now().time_since_epoch())
	                                        .count());
	// Steering, accelerator, brake, clutch: what DeviceClassParseData decodes.
	d.device_unique_data_len = 12;
	std::memcpy(d.device_unique_data, r + 40, 12);
	static uint16_t last_steer = 0;
	const auto      steer      = static_cast<uint16_t>(r[40] | (r[41] << 8u));
	if (Opts().log && steer / 1024 != last_steer / 1024) {
		last_steer = steer;
		LOGF("KYTY_USB PadRead %d: buttons=%08x steer=%u pedals=%02x%02x %02x%02x %02x%02x\n",
		     handle, d.buttons, steer, r[43], r[42], r[45], r[44], r[47], r[46]);
	}
	return 1;
}

int GetExtControllerInformation(int handle, void* info) {
	auto* p = static_cast<uint8_t*>(info);
	if (p == nullptr) {
		return Controller::PAD_ERROR_INVALID_ARG;
	}
	// ScePadControllerInformation (0x1c bytes), then the device's vendor and product ids.
	const int32_t device_class = PAD_DEVICE_CLASS_STEERING_WHEEL;
	p[0x0b]                    = 1; // connected count
	p[0x0c]                    = 1; // connected
	std::memcpy(p + 0x10, &device_class, 4);
	std::memcpy(p + 0x1c, &g_pad.vid, 2);
	std::memcpy(p + 0x1e, &g_pad.pid, 2);
	static bool logged = false;
	if (!logged) {
		logged = true;
		USB_LOG("PadGetExtControllerInformation %d -> %04x:%04x class %d\n", handle, g_pad.vid,
		        g_pad.pid, device_class);
	}
	return OK;
}

// shortcut: a 900-degree lock is assumed for the angle; pedals read 0 released .. 65535 pressed
// (the device sends them inverted).
int ParseData(int /*handle*/, const Controller::PadData* data, void* class_data) {
	if (data == nullptr || class_data == nullptr) {
		return Controller::PAD_ERROR_INVALID_ARG;
	}
	auto*          out = static_cast<uint8_t*>(class_data);
	const uint8_t* u   = data->device_unique_data;
	auto           u16 = [&](int i) { return static_cast<uint16_t>(u[i] | (u[i + 1] << 8u)); };
	std::memset(out, 0, 24);
	const int32_t  device_class = PAD_DEVICE_CLASS_STEERING_WHEEL;
	const uint16_t steer        = u16(0);
	const float    angle        = (static_cast<float>(steer) - 32768.0f) / 32768.0f * 450.0f;
	const uint16_t pedals[3]    = {static_cast<uint16_t>(0xffffu - u16(2)),
	                               static_cast<uint16_t>(0xffffu - u16(4)),
	                               static_cast<uint16_t>(0xffffu - u16(6))};
	std::memcpy(out, &device_class, 4);
	out[4] = data->connected ? 1 : 0;
	std::memcpy(out + 8, &angle, 4);
	std::memcpy(out + 12, &steer, 2);
	std::memcpy(out + 14, pedals, sizeof(pedals)); // accelerator, brake, clutch
	return OK;
}

int GetFeatureReport(int handle, int report_id, uint8_t* data, int length) {
	const int r =
	    Host::Control(g_pad.dev, 0xa1, 0x01, static_cast<uint16_t>(0x0300u | (report_id & 0xffu)),
	                  g_pad.interface_number, data, static_cast<uint16_t>(length), 500);
	USB_LOG("PadGetFeatureReport %d id=%02x len=%d -> %d first=%02x\n", handle, report_id, length,
	        r, r > 0 ? data[0] : 0);
	return r > 0 ? OK : Controller::PAD_ERROR_DEVICE_NOT_CONNECTED;
}

// The data may or may not start with the report id; the device expects it first.
int OutputReport(int handle, int report_id, const uint8_t* data, int length) {
	uint8_t buf[64] {};
	int     n = 0;
	if (length <= 0 || data[0] != report_id) {
		buf[n++] = static_cast<uint8_t>(report_id);
	}
	const int copy = std::min(length, static_cast<int>(sizeof(buf)) - n);
	std::memcpy(buf + n, data, static_cast<size_t>(std::max(copy, 0)));
	n += std::max(copy, 0);
	int r      = 0;
	int actual = 0;
	if (g_pad.out_endpoint != 0) {
		r = Host::Transfer(g_pad.dev, g_pad.out_endpoint, buf, n, 100, &actual);
	} else {
		r = Host::Control(g_pad.dev, 0x21, 0x09, static_cast<uint16_t>(0x0200u | buf[0]),
		                  g_pad.interface_number, buf, static_cast<uint16_t>(n), 100);
	}
	USB_LOG("PadOutputReport %d id=%02x len=%d -> %d [%02x %02x %02x %02x %02x %02x %02x %02x]\n",
	        handle, report_id, length, r, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6],
	        buf[7]);
	return r >= 0 ? OK : Controller::PAD_ERROR_DEVICE_NOT_CONNECTED;
}

} // namespace UsbPad

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
