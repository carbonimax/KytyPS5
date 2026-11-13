// Opens a KYTY_USB_PASSTHRU device through the emulator's WinUSB backend, prints its descriptors,
// then reads its interrupt IN endpoint asynchronously (the sceUsbd path) and prints each report
// that differs from the previous one. Usage: usb_check VVVV:PPPP [seconds] (default 20).
#include "libs/usbHost.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace Libs::UsbHost;

static void PrintConfig(const ConfigDescriptor* c) {
	std::printf("config value=%u interfaces=%u total=%u attributes=%02x power=%u extra=%d\n",
	            c->bConfigurationValue, c->bNumInterfaces, c->wTotalLength, c->bmAttributes,
	            c->MaxPower, c->extra_length);
	for (int i = 0; i < c->bNumInterfaces; i++) {
		const auto& itf = c->interface[i];
		for (int a = 0; a < itf.num_altsetting; a++) {
			const auto& d = itf.altsetting[a];
			std::printf("  interface %u alt %u class=%02x/%02x/%02x endpoints=%u extra=%d\n",
			            d.bInterfaceNumber, d.bAlternateSetting, d.bInterfaceClass,
			            d.bInterfaceSubClass, d.bInterfaceProtocol, d.bNumEndpoints,
			            d.extra_length);
			for (int e = 0; e < d.bNumEndpoints; e++) {
				const auto& ep = d.endpoint[e];
				std::printf("    endpoint %02x attributes=%02x max_packet=%u interval=%u\n",
				            ep.bEndpointAddress, ep.bmAttributes, ep.wMaxPacketSize, ep.bInterval);
			}
		}
	}
}

static std::string Ascii(Device* dev, uint8_t index) {
	uint8_t     buf[256] {};
	const int   n = index != 0 ? GetStringDescriptor(dev, index, 0x0409, buf, sizeof(buf)) : 0;
	std::string out;
	for (int i = 2; i + 1 < n; i += 2) {
		out += static_cast<char>(buf[i]);
	}
	return out;
}

int main(int argc, char** argv) {
	std::vector<std::pair<uint16_t, uint16_t>> ids;
	if (argc < 2 || !ParseIds(argv[1], &ids)) {
		std::fprintf(stderr, "usage: usb_check VVVV:PPPP [seconds]\n");
		return 2;
	}
	const int   seconds = argc > 2 ? std::atoi(argv[2]) : 20;
	std::string diagnostics;
	auto        devices = List(ids, &diagnostics);
	std::printf("%s%zu device(s)\n", diagnostics.c_str(), devices.size());
	if (devices.empty()) {
		return 1;
	}
	Device*        dev = devices[0];
	const uint8_t* dd  = DeviceDescriptor(dev);
	std::printf(
	    "device %02x%02x:%02x%02x usb=%x.%02x class=%02x max_packet0=%u configs=%u '%s' '%s'\n",
	    dd[9], dd[8], dd[11], dd[10], dd[3], dd[2], dd[4], dd[7], dd[17],
	    Ascii(dev, dd[14]).c_str(), Ascii(dev, dd[15]).c_str());
	const auto& raw    = RawConfig(dev);
	auto*       config = ParseConfig(raw.data(), raw.size());
	if (config == nullptr) {
		std::printf("config descriptor malformed (%zu bytes)\n", raw.size());
		return 1;
	}
	PrintConfig(config);
	std::printf("raw config:");
	for (uint8_t b: raw) {
		std::printf(" %02x", b);
	}
	std::printf("\n");
	struct Reader {
		uint8_t              endpoint = 0;
		int                  size     = 0;
		uint8_t              buf[1024] {};
		std::vector<uint8_t> last;
		int                  reports = 0;
		int                  changes = 0;
	};
	std::vector<Reader> readers;
	for (int i = 0; i < config->bNumInterfaces; i++) {
		for (int a = 0; a < config->interface[i].num_altsetting; a++) {
			const auto& d     = config->interface[i].altsetting[a];
			const bool  owned = HasInterface(dev, d.bInterfaceNumber);
			std::printf("interface %u alt %u: %s\n", d.bInterfaceNumber, d.bAlternateSetting,
			            owned ? "WinUSB" : "not on WinUSB");
			for (int e = 0; a == 0 && owned && e < d.bNumEndpoints; e++) {
				const auto& ep = d.endpoint[e];
				if ((ep.bEndpointAddress & 0x80u) != 0 && (ep.bmAttributes & 3u) == 3) {
					Reader r;
					r.endpoint = ep.bEndpointAddress;
					r.size     = std::min<int>(ep.wMaxPacketSize, 1024);
					readers.push_back(std::move(r));
				}
			}
		}
	}
	FreeConfig(config);
	if (readers.empty()) {
		std::printf("no interrupt IN endpoint on a WinUSB interface\n");
		return 1;
	}
	for (auto& r: readers) { // readers no longer grows: &r stays valid as the tag
		std::printf("reading endpoint %02x (%d bytes) for %d s\n", r.endpoint, r.size, seconds);
		if (Submit(dev, r.endpoint, r.buf, r.size, 0, &r) != USB_OK) {
			std::printf("submit %02x failed\n", r.endpoint);
			return 1;
		}
	}
	std::fflush(stdout);

	const auto start = std::chrono::steady_clock::now();
	const auto end   = start + std::chrono::seconds(seconds);
	while (std::chrono::steady_clock::now() < end) {
		std::vector<Completion> done;
		Wait(100, &done);
		for (const auto& c: done) {
			auto& r = *static_cast<Reader*>(c.tag);
			if (c.status != USB_TRANSFER_COMPLETED) {
				std::printf("endpoint %02x: transfer status %d\n", r.endpoint, c.status);
				return 1;
			}
			r.reports++;
			if (r.last.size() != static_cast<size_t>(c.actual) ||
			    std::memcmp(r.last.data(), r.buf, r.last.size()) != 0) {
				r.last.assign(r.buf, r.buf + c.actual);
				r.changes++;
				const double t =
				    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
				std::printf("%8.3f %02x:", t, r.endpoint);
				for (int i = 0; i < c.actual; i++) {
					std::printf(" %02x", r.buf[i]);
				}
				std::printf("\n");
				std::fflush(stdout);
			}
			if (Submit(dev, r.endpoint, r.buf, r.size, 0, &r) != USB_OK) {
				std::printf("resubmit %02x failed\n", r.endpoint);
				return 1;
			}
		}
	}
	bool changed = false;
	for (auto& r: readers) {
		Cancel(dev, r.endpoint);
		std::printf("endpoint %02x: reports=%d changes=%d\n", r.endpoint, r.reports, r.changes);
		changed = changed || r.changes > 1;
	}
	return changed ? 0 : 1;
}
