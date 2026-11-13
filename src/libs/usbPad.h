#ifndef EMULATOR_INCLUDE_LIBS_USB_PAD_H_
#define EMULATOR_INCLUDE_LIBS_USB_PAD_H_

// KYTY_USB_PASSTHRU, pad side: on a PS5 the system's pad driver also presents a racing wheel as an
// "ext" pad (scePadOpenExt, device class 6 = steering wheel) built from its main HID interface,
// next to the raw sceUsbd access. This emulates that driver for a passed-through device.

#include "libs/padData.h"

#include <cstdint>

namespace Libs::UsbPad {

constexpr int HANDLE = 0x20; // after the player, special and VR handles

bool Enabled(); // KYTY_USB_PASSTHRU set
bool IsHandle(int handle);
int  OpenExt(int type, const void* param); // handle, or a pad error when no device matches
int  Close(int handle);
int  Read(int handle, Controller::PadData* data, int num); // number of entries written
int  GetExtControllerInformation(int handle, void* info);
int  ParseData(int handle, const Controller::PadData* data, void* class_data);
int  GetFeatureReport(int handle, int report_id, uint8_t* data, int length);
int  OutputReport(int handle, int report_id, const uint8_t* data, int length);

} // namespace Libs::UsbPad

#endif /* EMULATOR_INCLUDE_LIBS_USB_PAD_H_ */
