// Opens the KYTY_WHEEL device through WheelInput, the emulator's code path, and prints what it
// feeds to the pad. No window and no game. Usage: wheel_check [seconds] (default 20).
#include "graphics/presentation/window/wheel.h"

#include <SDL3/SDL.h>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Libs::Graphics;

// Feeds SDL events to the wheel as the emulator's window loop does, printing each change.
static bool Run(WheelInput& wheel, int seconds) {
	bool         seen = false;
	const Uint64 end  = SDL_GetTicks() + static_cast<Uint64>(seconds) * 1000;
	for (Uint64 now = SDL_GetTicks(); now < end; now = SDL_GetTicks()) {
		SDL_Event event;
		if (!SDL_WaitEventTimeout(&event, static_cast<Sint32>(end - now))) {
			continue;
		}
		if (wheel.OnEvent(event) || (!seen && wheel.Connected())) {
			seen            = seen || wheel.Connected();
			const auto& pad = wheel.Pad();
			std::printf("%8.3f %s buttons=%08x left_x=%3d l2=%3d r2=%3d\n",
			            static_cast<double>(now) / 1000.0, wheel.Connected() ? "on " : "off",
			            pad.buttons, pad.left_x, pad.l2, pad.r2);
			std::fflush(stdout);
		}
	}
	return seen;
}

int main(int argc, char** argv) {
	const int   seconds = argc > 1 ? std::atoi(argv[1]) : 20;
	const char* option  = std::getenv("KYTY_WHEEL");
	const char* text    = std::getenv("KYTY_WHEEL_MAP");
	const char* log     = std::getenv("KYTY_WHEEL_LOG");
	WheelMap    map;
	std::string error;
	if (!ParseWheelMap(text != nullptr ? text : "", &map, &error)) {
		std::printf("KYTY_WHEEL_MAP: invalid item '%s'\n", error.c_str());
		return 1;
	}
	std::printf("map: %s\n", DescribeWheelMap(map).c_str());

	// Without a window there is no focus; the emulator gets events while its window has it.
	SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
	if (!SDL_Init(SDL_INIT_GAMEPAD)) {
		std::printf("SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}
	bool seen = false;
	{
		WheelInput wheel(option != nullptr && *option != '\0' ? option : "1", map,
		                 log != nullptr && std::string(log) == "1");
		seen = Run(wheel, seconds);
	}
	std::printf("%s\n", seen ? "wheel found" : "no wheel found");
	SDL_Quit();
	return seen ? 0 : 2;
}
