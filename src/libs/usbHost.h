#ifndef EMULATOR_INCLUDE_LIBS_USB_HOST_H_
#define EMULATOR_INCLUDE_LIBS_USB_HOST_H_

// KYTY_USB_PASSTHRU host side: USB devices bound to WinUSB, with libusb-1.0 semantics (the
// guest's sceUsbd is libusb). Windows only; elsewhere List() finds nothing.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifdef interface
#undef interface // combaseapi.h macro; ConfigDescriptor has a field of that name
#endif

namespace Libs::UsbHost {

// libusb error codes and transfer statuses (same values as libusb-1.0).
enum : int {
	USB_OK                  = 0,
	USB_ERROR_IO            = -1,
	USB_ERROR_INVALID_PARAM = -2,
	USB_ERROR_ACCESS        = -3,
	USB_ERROR_NO_DEVICE     = -4,
	USB_ERROR_NOT_FOUND     = -5,
	USB_ERROR_BUSY          = -6,
	USB_ERROR_TIMEOUT       = -7,
	USB_ERROR_OVERFLOW      = -8,
	USB_ERROR_PIPE          = -9,
	USB_ERROR_NO_MEM        = -11,
	USB_ERROR_NOT_SUPPORTED = -12,
	USB_ERROR_OTHER         = -99,
};
enum : int {
	USB_TRANSFER_COMPLETED = 0,
	USB_TRANSFER_ERROR     = 1,
	USB_TRANSFER_TIMED_OUT = 2,
	USB_TRANSFER_CANCELLED = 3,
	USB_TRANSFER_STALL     = 4,
	USB_TRANSFER_NO_DEVICE = 5,
	USB_TRANSFER_OVERFLOW  = 6,
};

// Guest ABI: libusb-1.0 descriptor structs (x86-64 layout, identical under SysV and MSVC).
struct EndpointDescriptor {
	uint8_t        bLength;
	uint8_t        bDescriptorType;
	uint8_t        bEndpointAddress;
	uint8_t        bmAttributes;
	uint16_t       wMaxPacketSize;
	uint8_t        bInterval;
	uint8_t        bRefresh;
	uint8_t        bSynchAddress;
	const uint8_t* extra;
	int            extra_length;
};
struct InterfaceDescriptor {
	uint8_t                   bLength;
	uint8_t                   bDescriptorType;
	uint8_t                   bInterfaceNumber;
	uint8_t                   bAlternateSetting;
	uint8_t                   bNumEndpoints;
	uint8_t                   bInterfaceClass;
	uint8_t                   bInterfaceSubClass;
	uint8_t                   bInterfaceProtocol;
	uint8_t                   iInterface;
	const EndpointDescriptor* endpoint;
	const uint8_t*            extra;
	int                       extra_length;
};
struct Interface {
	const InterfaceDescriptor* altsetting;
	int                        num_altsetting;
};
struct ConfigDescriptor {
	uint8_t          bLength;
	uint8_t          bDescriptorType;
	uint16_t         wTotalLength;
	uint8_t          bNumInterfaces;
	uint8_t          bConfigurationValue;
	uint8_t          iConfiguration;
	uint8_t          bmAttributes;
	uint8_t          MaxPower;
	const Interface* interface;
	const uint8_t*   extra;
	int              extra_length;
};
static_assert(sizeof(EndpointDescriptor) == 32 && sizeof(InterfaceDescriptor) == 40 &&
              sizeof(Interface) == 16 && sizeof(ConfigDescriptor) == 40);

// Builds the libusb tree from a raw configuration descriptor (null if malformed). One allocation
// owned by the result; free it with FreeConfig.
ConfigDescriptor* ParseConfig(const uint8_t* raw, size_t size);
void              FreeConfig(ConfigDescriptor* config);

struct Device;

// "VVVV:PPPP[,VVVV:PPPP]" -> ids; false on a malformed list.
bool ParseIds(const char* text, std::vector<std::pair<uint16_t, uint16_t>>* ids);

// Present WinUSB devices with one of the ids, opened once and kept for the process lifetime. The
// WinUSB functions of a composite device (its MI_xx children) form one Device.
// shortcut: no hot-plug, the device must be connected before the first call; reconnect = restart.
std::vector<Device*> List(const std::vector<std::pair<uint16_t, uint16_t>>& ids,
                          std::string*                                      diagnostics);

const uint8_t*              DeviceDescriptor(const Device* dev); // 18 bytes
const std::vector<uint8_t>& RawConfig(const Device* dev);        // active configuration
bool HasInterface(const Device* dev, int number);                // bound to WinUSB and opened
bool Connected(const Device* dev);

int GetStringDescriptor(Device* dev, uint8_t index, uint16_t lang, uint8_t* data, int length);
// Returns the bytes transferred or a USB_ERROR_*.
int Control(Device* dev, uint8_t request_type, uint8_t request, uint16_t value, uint16_t index,
            uint8_t* data, uint16_t length, uint32_t timeout_ms);
// Synchronous interrupt/bulk transfer: returns USB_OK or a USB_ERROR_*.
int Transfer(Device* dev, uint8_t endpoint, uint8_t* data, int length, uint32_t timeout_ms,
             int* actual);

// Asynchronous interrupt/bulk transfers: Submit starts one, Wait reports finished ones.
struct Completion {
	void* tag;
	int   status; // USB_TRANSFER_*
	int   actual;
};
int  Submit(Device* dev, uint8_t endpoint, uint8_t* data, int length, uint32_t timeout_ms,
            void* tag);
void Cancel(Device* dev, uint8_t endpoint); // pending transfers on it finish as CANCELLED
void Wait(uint32_t timeout_ms, std::vector<Completion>* done);

} // namespace Libs::UsbHost

#endif /* EMULATOR_INCLUDE_LIBS_USB_HOST_H_ */
