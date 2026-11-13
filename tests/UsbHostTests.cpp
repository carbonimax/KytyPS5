// KYTY_USB_PASSTHRU: id list parser and the libusb configuration tree built for the guest.
#include "libs/usbHost.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace Libs::UsbHost;

namespace {
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "UsbHostTests: %s\n", message);
		std::abort();
	}
}
} // namespace

int main() {
	std::vector<std::pair<uint16_t, uint16_t>> ids;
	Check(ParseIds("046d:c269", &ids) && ids.size() == 1 && ids[0].first == 0x046d &&
	          ids[0].second == 0xc269,
	      "one id");
	Check(ParseIds("046D:C269,1234:abcd", &ids) && ids.size() == 2 && ids[1].second == 0xabcd,
	      "two ids");
	Check(!ParseIds("", &ids) && !ParseIds("046d", &ids) && !ParseIds(":c269", &ids), "bad lists");
	Check(!ParseIds("12345:1", &ids) && !ParseIds("046d:c269x", &ids), "bad values");

	// Config with an IAD (config extra), interface 0 (HID descriptor extra, IN + OUT endpoints),
	// interface 1 alt 0, interface 0 alt 1 after it (must be grouped with interface 0).
	const uint8_t raw[] = {
	    9, 2,  64,   0, 2,    1,    0,  0x80, 250, // config
	    8, 11, 0,    2, 3,    0,    0,  0,         // IAD -> config extra
	    9, 4,  0,    0, 2,    3,    0,  0,    0,   // interface 0 alt 0
	    9, 33, 0x11, 1, 0,    1,    34, 0x20, 0,   // HID -> interface extra
	    7, 5,  0x81, 3, 64,   0,    1,             // EP 81 interrupt IN
	    7, 5,  0x01, 3, 64,   0,    1,             // EP 01 interrupt OUT
	    9, 4,  1,    0, 0,    0xff, 0,  0,    0,   // interface 1 alt 0
	    9, 4,  0,    1, 1,    3,    0,  0,    0,   // interface 0 alt 1
	    7, 5,  0x82, 3, 0x00, 2,    4,             // EP 82, 512 bytes
	};
	ConfigDescriptor* c = ParseConfig(raw, sizeof(raw));
	Check(c != nullptr, "parse");
	Check(c->bNumInterfaces == 2 && c->wTotalLength == 64 && c->MaxPower == 250 &&
	          c->extra_length == 8 && c->extra != nullptr && c->extra[1] == 11,
	      "config fields");
	const auto& i0 = c->interface[0];
	const auto& i1 = c->interface[1];
	Check(i0.num_altsetting == 2 && i1.num_altsetting == 1, "altsetting grouping");
	Check(i0.altsetting[0].bInterfaceNumber == 0 && i0.altsetting[1].bAlternateSetting == 1 &&
	          i1.altsetting[0].bInterfaceNumber == 1,
	      "interface numbers");
	const auto& a0 = i0.altsetting[0];
	Check(a0.bNumEndpoints == 2 && a0.extra_length == 9 && a0.extra[1] == 33, "interface extra");
	Check(a0.endpoint[0].bEndpointAddress == 0x81 && a0.endpoint[1].bEndpointAddress == 0x01 &&
	          a0.endpoint[0].wMaxPacketSize == 64 && a0.endpoint[0].extra == nullptr,
	      "endpoints");
	Check(i1.altsetting[0].bNumEndpoints == 0 && i1.altsetting[0].endpoint == nullptr,
	      "no endpoints");
	Check(i0.altsetting[1].endpoint[0].bEndpointAddress == 0x82 &&
	          i0.altsetting[1].endpoint[0].wMaxPacketSize == 512 &&
	          i0.altsetting[1].endpoint[0].bInterval == 4,
	      "alt 1 endpoint");
	FreeConfig(c);

	Check(ParseConfig(raw, 5) == nullptr, "short");
	uint8_t bad[sizeof(raw)];
	std::copy(raw, raw + sizeof(raw), bad);
	bad[17] = 0; // zero-length descriptor
	Check(ParseConfig(bad, sizeof(bad)) == nullptr, "zero length");
	std::printf("UsbHostTests: OK\n");
	return 0;
}
