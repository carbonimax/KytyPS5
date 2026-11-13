#include "libs/usbHost.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cfgmgr32.h>
#include <cwchar>
#include <map>
#include <mutex>
#include <setupapi.h>
#include <winusb.h>
#undef interface // combaseapi.h macro; ConfigDescriptor has a field of that name
#endif

namespace Libs::UsbHost {

namespace {

struct ConfigHolder {
	ConfigDescriptor                 cfg {}; // first: the guest pointer is the holder's address
	std::vector<uint8_t>             raw;
	std::vector<Interface>           interfaces;
	std::vector<InterfaceDescriptor> alts;
	std::vector<EndpointDescriptor>  endpoints;
};
static_assert(std::is_standard_layout_v<ConfigHolder>);

struct Span {
	size_t off = 0;
	size_t len = 0;
};

void Extend(Span* span, size_t off, size_t len) {
	if (span->len == 0) {
		span->off = off;
	}
	span->len = off + len - span->off;
}

} // namespace

ConfigDescriptor* ParseConfig(const uint8_t* raw, size_t size) {
	if (raw == nullptr || size < 9 || raw[0] < 9 || raw[0] > size || raw[1] != 2) {
		return nullptr;
	}
	auto* h = new ConfigHolder;
	h->raw.assign(raw, raw + size);
	const uint8_t* p = h->raw.data();

	struct Alt {
		InterfaceDescriptor d {};
		size_t              ep_first = 0;
		Span                extra;
		size_t              number_index = 0;
	};
	std::vector<Alt>     alts;
	std::vector<Span>    ep_extra;
	std::vector<uint8_t> numbers; // interface numbers in order of appearance
	Span                 cfg_extra;
	enum { OWNER_CONFIG, OWNER_ALT, OWNER_ENDPOINT } owner = OWNER_CONFIG;

	for (size_t off = p[0]; off + 2 <= size;) {
		const size_t  len  = p[off];
		const uint8_t type = p[off + 1];
		if (len < 2 || off + len > size) {
			delete h;
			return nullptr;
		}
		if (type == 4 && len >= 9) {
			Alt a;
			std::memcpy(&a.d, p + off, 9);
			a.d.bNumEndpoints = 0; // counted from the endpoints actually present
			a.ep_first        = h->endpoints.size();
			auto it           = std::find(numbers.begin(), numbers.end(), a.d.bInterfaceNumber);
			a.number_index    = static_cast<size_t>(it - numbers.begin());
			if (it == numbers.end()) {
				numbers.push_back(a.d.bInterfaceNumber);
			}
			alts.push_back(a);
			owner = OWNER_ALT;
		} else if (type == 5 && len >= 7 && !alts.empty()) {
			EndpointDescriptor e {};
			std::memcpy(&e, p + off, 4);
			e.wMaxPacketSize = static_cast<uint16_t>(p[off + 4] | (p[off + 5] << 8u));
			e.bInterval      = p[off + 6];
			if (len >= 9) {
				e.bRefresh      = p[off + 7];
				e.bSynchAddress = p[off + 8];
			}
			h->endpoints.push_back(e);
			ep_extra.emplace_back();
			alts.back().d.bNumEndpoints =
			    static_cast<uint8_t>(h->endpoints.size() - alts.back().ep_first);
			owner = OWNER_ENDPOINT;
		} else {
			Extend(owner == OWNER_CONFIG ? &cfg_extra
			       : owner == OWNER_ALT  ? &alts.back().extra
			                             : &ep_extra.back(),
			       off, len);
		}
		off += len;
	}

	// Endpoints get their pointers only now: the vectors no longer grow.
	for (size_t i = 0; i < h->endpoints.size(); i++) {
		h->endpoints[i].extra        = ep_extra[i].len != 0 ? p + ep_extra[i].off : nullptr;
		h->endpoints[i].extra_length = static_cast<int>(ep_extra[i].len);
	}
	std::stable_sort(alts.begin(), alts.end(),
	                 [](const Alt& a, const Alt& b) { return a.number_index < b.number_index; });
	for (auto& a: alts) {
		a.d.endpoint     = a.d.bNumEndpoints != 0 ? &h->endpoints[a.ep_first] : nullptr;
		a.d.extra        = a.extra.len != 0 ? p + a.extra.off : nullptr;
		a.d.extra_length = static_cast<int>(a.extra.len);
		h->alts.push_back(a.d);
	}
	// bNumInterfaces sizes the guest's walk of interface[]: pad it with empty entries.
	h->interfaces.resize(std::max<size_t>(numbers.size(), p[4]));
	for (size_t i = 0; i < alts.size(); i++) {
		auto& itf = h->interfaces[alts[i].number_index];
		if (itf.num_altsetting++ == 0) {
			itf.altsetting = &h->alts[i];
		}
	}

	std::memcpy(&h->cfg, p, 9);
	h->cfg.wTotalLength   = static_cast<uint16_t>(p[2] | (p[3] << 8u));
	h->cfg.bNumInterfaces = static_cast<uint8_t>(h->interfaces.size());
	h->cfg.interface      = h->interfaces.data();
	h->cfg.extra          = cfg_extra.len != 0 ? p + cfg_extra.off : nullptr;
	h->cfg.extra_length   = static_cast<int>(cfg_extra.len);
	return &h->cfg;
}

void FreeConfig(ConfigDescriptor* config) {
	delete reinterpret_cast<ConfigHolder*>(config);
}

bool ParseIds(const char* text, std::vector<std::pair<uint16_t, uint16_t>>* ids) {
	ids->clear();
	while (text != nullptr && *text != '\0') {
		char*               end = nullptr;
		const unsigned long vid = std::strtoul(text, &end, 16);
		if (end == text || *end != ':' || vid > 0xffff) {
			return false;
		}
		text                    = end + 1;
		const unsigned long pid = std::strtoul(text, &end, 16);
		if (end == text || (*end != ',' && *end != '\0') || pid > 0xffff) {
			return false;
		}
		ids->emplace_back(static_cast<uint16_t>(vid), static_cast<uint16_t>(pid));
		text = *end == ',' ? end + 1 : end;
	}
	return !ids->empty();
}

#if defined(_WIN32)

// One WinUSB-bound interface: a composite device has one per MI_xx child.
struct Function {
	HANDLE                  file   = INVALID_HANDLE_VALUE;
	WINUSB_INTERFACE_HANDLE usb    = nullptr;
	uint8_t                 number = 0;
};

struct Device {
	DEVINST                                    key = 0;   // composite parent, or the device itself
	std::vector<Function>                      functions; // by interface number; [0] carries ep 0
	std::map<uint8_t, WINUSB_INTERFACE_HANDLE> pipes;     // endpoint address -> owning interface
	uint8_t                                    device_descriptor[18] {};
	std::vector<uint8_t>                       config;
	bool                                       lost = false;
	std::mutex                                 policy_mutex;
	std::map<std::pair<WINUSB_INTERFACE_HANDLE, uint8_t>, ULONG> timeouts; // per handle and pipe
};

namespace {

std::mutex           g_list_mutex;
std::vector<Device*> g_devices;

struct Pending {
	OVERLAPPED              ov {};
	Device*                 dev = nullptr;
	WINUSB_INTERFACE_HANDLE usb = nullptr;
	void*                   tag = nullptr;
};
std::mutex            g_pending_mutex;
std::vector<Pending*> g_pending;

HANDLE WakeEvent() {
	static HANDLE wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	return wake;
}

WINUSB_INTERFACE_HANDLE Pipe(const Device* dev, uint8_t endpoint) {
	if ((endpoint & 0x7fu) == 0) {
		return dev->functions[0].usb;
	}
	auto it = dev->pipes.find(endpoint);
	return it != dev->pipes.end() ? it->second : nullptr;
}

bool IsGone(DWORD err) {
	return err == ERROR_DEVICE_NOT_CONNECTED || err == ERROR_BAD_COMMAND ||
	       err == ERROR_FILE_NOT_FOUND || err == ERROR_NO_SUCH_DEVICE;
}

int MapError(Device* dev, DWORD err) {
	switch (err) {
		case 0: return USB_OK;
		case ERROR_SEM_TIMEOUT: return USB_ERROR_TIMEOUT;
		case ERROR_GEN_FAILURE: return USB_ERROR_PIPE; // WinUSB reports a stall this way
		case ERROR_INVALID_PARAMETER: return USB_ERROR_INVALID_PARAM;
		case ERROR_ACCESS_DENIED: return USB_ERROR_ACCESS;
		case ERROR_NOT_ENOUGH_MEMORY: return USB_ERROR_NO_MEM;
		case ERROR_NOT_SUPPORTED: return USB_ERROR_NOT_SUPPORTED;
		default: break;
	}
	if (IsGone(err)) {
		dev->lost = true;
		return USB_ERROR_NO_DEVICE;
	}
	return USB_ERROR_IO;
}

int MapStatus(Device* dev, DWORD err) {
	switch (MapError(dev, err)) {
		case USB_OK: return USB_TRANSFER_COMPLETED;
		case USB_ERROR_TIMEOUT: return USB_TRANSFER_TIMED_OUT;
		case USB_ERROR_PIPE: return USB_TRANSFER_STALL;
		case USB_ERROR_NO_DEVICE: return USB_TRANSFER_NO_DEVICE;
		default:
			return err == ERROR_OPERATION_ABORTED ? USB_TRANSFER_CANCELLED : USB_TRANSFER_ERROR;
	}
}

// libusb and WinUSB both read 0 as "no timeout".
void SetTimeout(Device* dev, WINUSB_INTERFACE_HANDLE usb, uint8_t endpoint, uint32_t timeout_ms) {
	std::lock_guard lock(dev->policy_mutex);
	auto            it = dev->timeouts.find({usb, endpoint});
	if (it != dev->timeouts.end() && it->second == timeout_ms) {
		return;
	}
	ULONG value = timeout_ms;
	WinUsb_SetPipePolicy(usb, endpoint, PIPE_TRANSFER_TIMEOUT, sizeof(value), &value);
	dev->timeouts[{usb, endpoint}] = value;
}

// Runs one overlapped WinUSB call to completion; returns the Win32 error (0 on success).
template <class F>
DWORD RunSync(WINUSB_INTERFACE_HANDLE usb, F start, ULONG* got) {
	OVERLAPPED ov {};
	ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
	BOOL  ok  = start(&ov);
	DWORD err = ok ? 0 : GetLastError();
	if (ok || err == ERROR_IO_PENDING) {
		ok  = WinUsb_GetOverlappedResult(usb, &ov, got, TRUE);
		err = ok ? 0 : GetLastError();
	}
	CloseHandle(ov.hEvent);
	return err;
}

std::string Narrow(const std::wstring& w) {
	std::string out;
	for (wchar_t c: w) {
		out += static_cast<char>(c); // ids and paths are ASCII
	}
	return out;
}

struct Found {
	DEVINST      key;
	std::wstring path;
};

// Device-interface paths of the WinUSB-bound functions whose instance id has VID_v&PID_p.
std::vector<Found> FindPaths(uint16_t vid, uint16_t pid, std::string* diagnostics) {
	std::vector<Found> found;
	wchar_t            want[32];
	std::swprintf(want, 32, L"\\VID_%04X&PID_%04X", vid, pid);
	HDEVINFO set = SetupDiGetClassDevsW(nullptr, L"USB", nullptr, DIGCF_PRESENT | DIGCF_ALLCLASSES);
	if (set == INVALID_HANDLE_VALUE) {
		return found;
	}
	SP_DEVINFO_DATA info {sizeof(info)};
	for (DWORD i = 0; SetupDiEnumDeviceInfo(set, i, &info); i++) {
		wchar_t id[512] {};
		if (!SetupDiGetDeviceInstanceIdW(set, &info, id, 512, nullptr)) {
			continue;
		}
		_wcsupr_s(id);
		if (std::wcsstr(id, want) == nullptr) {
			continue;
		}
		wchar_t guids[512] {};
		DWORD   size = sizeof(guids) - sizeof(wchar_t);
		HKEY    key  = SetupDiOpenDevRegKey(set, &info, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);
		bool    have = false;
		if (key != INVALID_HANDLE_VALUE) {
			have = RegQueryValueExW(key, L"DeviceInterfaceGUIDs", nullptr, nullptr,
			                        reinterpret_cast<BYTE*>(guids), &size) == ERROR_SUCCESS;
			if (!have) {
				size = sizeof(guids) - sizeof(wchar_t);
				have = RegQueryValueExW(key, L"DeviceInterfaceGUID", nullptr, nullptr,
				                        reinterpret_cast<BYTE*>(guids), &size) == ERROR_SUCCESS;
			}
			RegCloseKey(key);
		}
		GUID     g {};
		unsigned b[8] {};
		if (!have ||
		    std::swscanf(guids, L"{%8lx-%4hx-%4hx-%2x%2x-%2x%2x%2x%2x%2x%2x}", &g.Data1, &g.Data2,
		                 &g.Data3, &b[0], &b[1], &b[2], &b[3], &b[4], &b[5], &b[6], &b[7]) != 11) {
			*diagnostics += Narrow(id) + ": not on WinUSB (no DeviceInterfaceGUIDs)\n";
			continue;
		}
		for (int k = 0; k < 8; k++) {
			g.Data4[k] = static_cast<unsigned char>(b[k]);
		}
		DEVINST owner = info.DevInst;
		if (std::wcsstr(id, L"&MI_") != nullptr) {
			CM_Get_Parent(&owner, info.DevInst, 0); // the MI_xx children share their parent
		}
		HDEVINFO ifs = SetupDiGetClassDevsW(&g, id, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
		if (ifs == INVALID_HANDLE_VALUE) {
			continue;
		}
		SP_DEVICE_INTERFACE_DATA data {sizeof(data)};
		if (SetupDiEnumDeviceInterfaces(ifs, nullptr, &g, 0, &data)) {
			DWORD need = 0;
			SetupDiGetDeviceInterfaceDetailW(ifs, &data, nullptr, 0, &need, nullptr);
			std::vector<uint8_t> buf(
			    std::max<DWORD>(need, sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W)));
			auto* detail   = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(buf.data());
			detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
			if (SetupDiGetDeviceInterfaceDetailW(ifs, &data, detail, need, nullptr, nullptr)) {
				found.push_back({owner, detail->DevicePath});
				*diagnostics += Narrow(id) + ": WinUSB\n";
			}
		}
		SetupDiDestroyDeviceInfoList(ifs);
	}
	SetupDiDestroyDeviceInfoList(set);
	return found;
}

bool OpenFunction(Device* dev, const std::wstring& path, std::string* diagnostics) {
	Function f;
	f.file =
	    CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
	                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
	USB_INTERFACE_DESCRIPTOR itf {};
	if (f.file == INVALID_HANDLE_VALUE || !WinUsb_Initialize(f.file, &f.usb) ||
	    !WinUsb_QueryInterfaceSettings(f.usb, 0, &itf)) {
		*diagnostics += "open failed, error " + std::to_string(GetLastError()) + "\n";
		if (f.usb != nullptr) {
			WinUsb_Free(f.usb);
		}
		if (f.file != INVALID_HANDLE_VALUE) {
			CloseHandle(f.file);
		}
		return false;
	}
	f.number = itf.bInterfaceNumber;
	for (UCHAR i = 0; i < itf.bNumEndpoints; i++) {
		WINUSB_PIPE_INFORMATION pipe {};
		if (WinUsb_QueryPipe(f.usb, 0, i, &pipe)) {
			dev->pipes[pipe.PipeId] = f.usb;
		}
	}
	dev->functions.push_back(f);
	return true;
}

// Configuration descriptor 0 through WinUsb_GetDescriptor, or a raw GET_DESCRIPTOR request.
bool ReadConfig(WINUSB_INTERFACE_HANDLE usb, bool control, std::vector<uint8_t>* out) {
	auto read = [&](uint8_t* data, uint16_t length) {
		ULONG got = 0;
		if (!control) {
			return WinUsb_GetDescriptor(usb, 2, 0, 0, data, length, &got) ? got : 0;
		}
		WINUSB_SETUP_PACKET setup {0x80, 6, 0x0200, 0, length};
		return RunSync(
		           usb,
		           [&](OVERLAPPED* ov) {
			           return WinUsb_ControlTransfer(usb, setup, data, length, &got, ov);
		           },
		           &got) == 0
		           ? got
		           : 0;
	};
	uint8_t head[9] {};
	if (read(head, 9) != 9 || head[1] != 2) {
		return false;
	}
	out->resize(static_cast<size_t>(head[2] | (head[3] << 8u)));
	if (out->size() < 9 || read(out->data(), static_cast<uint16_t>(out->size())) != out->size()) {
		out->clear();
		return false;
	}
	return true;
}

// Opens every WinUSB function of one device; descriptors come through the lowest interface.
Device* OpenDevice(DEVINST key, const std::vector<std::wstring>& paths, std::string* diagnostics) {
	auto* dev = new Device;
	dev->key  = key;
	for (const auto& path: paths) {
		OpenFunction(dev, path, diagnostics);
	}
	std::sort(dev->functions.begin(), dev->functions.end(),
	          [](const Function& a, const Function& b) { return a.number < b.number; });
	ULONG got = 0;
	if (!dev->functions.empty() &&
	    WinUsb_GetDescriptor(dev->functions[0].usb, 1, 0, 0, dev->device_descriptor, 18, &got) &&
	    got == 18) {
		// A composite child's handle reports only its own function's configuration: prefer the
		// device's own descriptor (control request), else rebuild it from every function's part.
		std::vector<uint8_t> merged;
		for (const auto& f: dev->functions) {
			std::vector<uint8_t> part;
			if (ReadConfig(f.usb, false, &part)) {
				*diagnostics += "interface " + std::to_string(f.number) + ": config part " +
				                std::to_string(part.size()) + " bytes\n";
				merged.insert(merged.end(), part.begin() + (merged.empty() ? 0 : 9), part.end());
			}
		}
		if (!merged.empty()) {
			merged[2] = static_cast<uint8_t>(merged.size());
			merged[3] = static_cast<uint8_t>(merged.size() >> 8u);
			merged[4] = static_cast<uint8_t>(dev->functions.size());
		}
		const bool control = ReadConfig(dev->functions[0].usb, true, &dev->config) &&
		                     (merged.empty() || dev->config[4] >= merged[4]);
		*diagnostics +=
		    control
		        ? "config from GET_DESCRIPTOR: " + std::to_string(dev->config.size()) + " bytes\n"
		        : "config merged from the functions: " + std::to_string(merged.size()) + " bytes\n";
		if (!control) {
			dev->config = std::move(merged);
		}
		if (!dev->config.empty()) {
			return dev;
		}
	}
	*diagnostics += "descriptors failed, error " + std::to_string(GetLastError()) + "\n";
	for (auto& f: dev->functions) {
		WinUsb_Free(f.usb);
		CloseHandle(f.file);
	}
	delete dev;
	return nullptr;
}

} // namespace

std::vector<Device*> List(const std::vector<std::pair<uint16_t, uint16_t>>& ids,
                          std::string*                                      diagnostics) {
	std::lock_guard      lock(g_list_mutex);
	std::vector<Device*> out;
	for (const auto& [vid, pid]: ids) {
		std::map<DEVINST, std::vector<std::wstring>> by_device;
		for (auto& f: FindPaths(vid, pid, diagnostics)) {
			by_device[f.key].push_back(std::move(f.path));
		}
		for (const auto& [key, paths]: by_device) {
			auto    it  = std::find_if(g_devices.begin(), g_devices.end(),
			                           [&](const Device* d) { return d->key == key; });
			Device* dev = it != g_devices.end() ? *it : OpenDevice(key, paths, diagnostics);
			if (dev != nullptr) {
				if (it == g_devices.end()) {
					g_devices.push_back(dev);
				}
				out.push_back(dev);
			}
		}
	}
	return out;
}

const uint8_t* DeviceDescriptor(const Device* dev) {
	return dev->device_descriptor;
}
const std::vector<uint8_t>& RawConfig(const Device* dev) {
	return dev->config;
}
bool HasInterface(const Device* dev, int number) {
	return std::any_of(dev->functions.begin(), dev->functions.end(),
	                   [&](const Function& f) { return f.number == number; });
}
bool Connected(const Device* dev) {
	return !dev->lost;
}

int GetStringDescriptor(Device* dev, uint8_t index, uint16_t lang, uint8_t* data, int length) {
	ULONG got = 0;
	auto  get = [&](uint16_t id) {
		return WinUsb_GetDescriptor(dev->functions[0].usb, 3, index, id, data,
		                            static_cast<ULONG>(length), &got) != FALSE;
	};
	// Guests ask with langid 0 (libusb style); a device may only answer its own language.
	if (!get(lang) && !(lang == 0 && index != 0 && get(0x0409))) {
		return MapError(dev, GetLastError());
	}
	return static_cast<int>(got);
}

int Control(Device* dev, uint8_t request_type, uint8_t request, uint16_t value, uint16_t index,
            uint8_t* data, uint16_t length, uint32_t timeout_ms) {
	// Each MI_xx handle only reaches its own interface: route interface/endpoint requests to it.
	WINUSB_INTERFACE_HANDLE usb       = dev->functions[0].usb;
	const uint8_t           recipient = request_type & 0x1fu;
	if (recipient == 1) {
		for (const auto& f: dev->functions) {
			usb = f.number == (index & 0xffu) ? f.usb : usb;
		}
	} else if (recipient == 2 && Pipe(dev, static_cast<uint8_t>(index)) != nullptr) {
		usb = Pipe(dev, static_cast<uint8_t>(index));
	}
	SetTimeout(dev, usb, 0, timeout_ms);
	WINUSB_SETUP_PACKET setup {request_type, request, value, index, length};
	ULONG               got = 0;
	const DWORD         err = RunSync(
	    usb,
	    [&](OVERLAPPED* ov) { return WinUsb_ControlTransfer(usb, setup, data, length, &got, ov); },
	    &got);
	return err == 0 ? static_cast<int>(got) : MapError(dev, err);
}

int Transfer(Device* dev, uint8_t endpoint, uint8_t* data, int length, uint32_t timeout_ms,
             int* actual) {
	*actual  = 0;
	auto usb = Pipe(dev, endpoint);
	if (usb == nullptr) {
		return USB_ERROR_NOT_FOUND;
	}
	SetTimeout(dev, usb, endpoint, timeout_ms);
	ULONG       got = 0;
	const DWORD err = RunSync(
	    usb,
	    [&](OVERLAPPED* ov) {
		    return (endpoint & 0x80u) != 0
		               ? WinUsb_ReadPipe(usb, endpoint, data, static_cast<ULONG>(length), &got, ov)
		               : WinUsb_WritePipe(usb, endpoint, data, static_cast<ULONG>(length), &got,
		                                  ov);
	    },
	    &got);
	*actual = static_cast<int>(got);
	return MapError(dev, err);
}

int Submit(Device* dev, uint8_t endpoint, uint8_t* data, int length, uint32_t timeout_ms,
           void* tag) {
	auto usb = Pipe(dev, endpoint);
	if (usb == nullptr) {
		return USB_ERROR_NOT_FOUND;
	}
	SetTimeout(dev, usb, endpoint, timeout_ms);
	auto* p      = new Pending;
	p->dev       = dev;
	p->usb       = usb;
	p->tag       = tag;
	p->ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
	const BOOL ok =
	    (endpoint & 0x80u) != 0
	        ? WinUsb_ReadPipe(usb, endpoint, data, static_cast<ULONG>(length), nullptr, &p->ov)
	        : WinUsb_WritePipe(usb, endpoint, data, static_cast<ULONG>(length), nullptr, &p->ov);
	const DWORD err = ok ? 0 : GetLastError();
	if (!ok && err != ERROR_IO_PENDING) {
		CloseHandle(p->ov.hEvent);
		delete p;
		return MapError(dev, err);
	}
	std::lock_guard lock(g_pending_mutex);
	g_pending.push_back(p);
	SetEvent(WakeEvent());
	return USB_OK;
}

// shortcut: WinUSB aborts every transfer of the pipe, not just one; fine for one reader per pipe.
void Cancel(Device* dev, uint8_t endpoint) {
	if (auto usb = Pipe(dev, endpoint); usb != nullptr) {
		WinUsb_AbortPipe(usb, endpoint);
	}
}

void Wait(uint32_t timeout_ms, std::vector<Completion>* done) {
	std::vector<HANDLE> events {WakeEvent()};
	{
		std::lock_guard lock(g_pending_mutex);
		for (size_t i = 0; i < g_pending.size() && events.size() < MAXIMUM_WAIT_OBJECTS; i++) {
			events.push_back(g_pending[i]->ov.hEvent);
		}
	}
	WaitForMultipleObjects(static_cast<DWORD>(events.size()), events.data(), FALSE, timeout_ms);
	std::lock_guard lock(g_pending_mutex);
	for (size_t i = 0; i < g_pending.size();) {
		Pending* p   = g_pending[i];
		ULONG    got = 0;
		DWORD    err = 0;
		if (!WinUsb_GetOverlappedResult(p->usb, &p->ov, &got, FALSE)) {
			err = GetLastError();
			if (err == ERROR_IO_INCOMPLETE) {
				i++;
				continue;
			}
		}
		done->push_back({p->tag, MapStatus(p->dev, err), static_cast<int>(got)});
		CloseHandle(p->ov.hEvent);
		delete p;
		g_pending.erase(g_pending.begin() + static_cast<std::ptrdiff_t>(i));
	}
}

#else // shortcut: WinUSB only; another host backend would replace this block.

struct Device {
	uint8_t              device_descriptor[18] {};
	std::vector<uint8_t> config;
};

std::vector<Device*> List(const std::vector<std::pair<uint16_t, uint16_t>>& /*ids*/,
                          std::string* diagnostics) {
	*diagnostics += "USB passthrough is Windows only\n";
	return {};
}
const uint8_t* DeviceDescriptor(const Device* dev) {
	return dev->device_descriptor;
}
const std::vector<uint8_t>& RawConfig(const Device* dev) {
	return dev->config;
}
bool HasInterface(const Device* /*dev*/, int /*number*/) {
	return false;
}
bool Connected(const Device* /*dev*/) {
	return false;
}
int GetStringDescriptor(Device*, uint8_t, uint16_t, uint8_t*, int) {
	return USB_ERROR_NOT_SUPPORTED;
}
int Control(Device*, uint8_t, uint8_t, uint16_t, uint16_t, uint8_t*, uint16_t, uint32_t) {
	return USB_ERROR_NOT_SUPPORTED;
}
int Transfer(Device*, uint8_t, uint8_t*, int, uint32_t, int* actual) {
	*actual = 0;
	return USB_ERROR_NOT_SUPPORTED;
}
int Submit(Device*, uint8_t, uint8_t*, int, uint32_t, void*) {
	return USB_ERROR_NOT_SUPPORTED;
}
void Cancel(Device*, uint8_t) {}
void Wait(uint32_t /*timeout_ms*/, std::vector<Completion>* /*done*/) {}

#endif

} // namespace Libs::UsbHost
