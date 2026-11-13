// KYTY_WHEEL map parser and axis/trigger conversion.
#include "graphics/presentation/window/wheel.h"
#include "libs/controller.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Libs::Graphics;
namespace Pad = Libs::Controller;

namespace {
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "WheelTests: %s\n", message);
		std::abort();
	}
}
} // namespace

int main() {
	WheelMap    map;
	std::string error;

	Check(ParseWheelMap("", &map, &error), "empty map");
	Check(map.steer_axis == 0 && map.throttle_axis == 3 && map.brake_axis == 4 &&
	          map.clutch_axis == 5 && map.clutch_button == 0 && map.gain == 1.0f &&
	          map.deadzone == 0.0f,
	      "RS50 defaults");
	Check(map.buttons[0] == Pad::PAD_BUTTON_CROSS && map.buttons[11] == Pad::PAD_BUTTON_TOUCH_PAD &&
	          map.buttons[7] == 0 && map.buttons[12] == 0,
	      "default buttons");

	Check(ParseWheelMap("throttle=4, brake=3 clutch=none clutchto=l1 gain=2.5 dz=0.1 b0=none "
	                    "b21=triangle pdz=0",
	                    &map, &error),
	      "custom map");
	Check(map.throttle_axis == 4 && map.brake_axis == 3 && map.clutch_axis == -1 &&
	          map.clutch_button == Pad::PAD_BUTTON_L1 && map.gain == 2.5f && map.deadzone == 0.1f &&
	          map.buttons[0] == 0 && map.buttons[21] == Pad::PAD_BUTTON_TRIANGLE &&
	          map.pedal_threshold == 0,
	      "custom values");
	Check(DescribeWheelMap(map).find("b21=triangle") != std::string::npos, "describe");

	Check(!ParseWheelMap("b0=jump", &map, &error) && error == "b0=jump", "bad button name");
	Check(!ParseWheelMap("gain", &map, &error), "missing value");
	Check(!ParseWheelMap("throttle=x", &map, &error), "bad number");
	Check(!ParseWheelMap("speed=1", &map, &error), "unknown key");
	Check(!ParseWheelMap("dz=0.95", &map, &error), "deadzone range");

	WheelMap plain;
	Check(WheelSteer(plain, -32768) == 0 && WheelSteer(plain, 0) == 128 &&
	          WheelSteer(plain, 32767) == 255,
	      "steer range");
	plain.gain = 2.0f;
	Check(WheelSteer(plain, 16384) == 255 && WheelSteer(plain, -8192) == 64, "steer gain");
	plain.gain     = 1.0f;
	plain.deadzone = 0.5f;
	Check(WheelSteer(plain, 16000) == 128 && WheelSteer(plain, 32767) == 255 &&
	          WheelSteer(plain, -24576) == 64,
	      "steer deadzone");

	Check(WheelPedal(-32768, 0) == 0 && WheelPedal(32767, 0) == 255 && WheelPedal(0, 0) == 127,
	      "pedal range");
	Check(WheelPedal(-32000, 8) == 0 && WheelPedal(-30000, 8) > 8, "pedal threshold");

	Check(WheelMatches("1", "Logitech RS50 Base for PlayStation/PC", 0x046d, 0xc276), "name");
	Check(!WheelMatches("1", "DualSense Wireless Controller", 0x054c, 0x0ce6), "not a wheel");
	Check(WheelMatches("046d:c276", "Base", 0x046d, 0xc276), "vid:pid");
	Check(!WheelMatches("046d:c294", "Logitech G29", 0x046d, 0xc24f), "other vid:pid");

	std::puts("WheelTests: all cases passed");
	return 0;
}
