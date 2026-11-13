#include "graphics/presentation/window/wheel.h"

#include "common/logging/log.h"
#include "libs/controller.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr std::array<std::pair<std::string_view, uint32_t>, 20> BUTTON_NAMES = {{
    {"none", 0},
    {"cross", Controller::PAD_BUTTON_CROSS},
    {"circle", Controller::PAD_BUTTON_CIRCLE},
    {"square", Controller::PAD_BUTTON_SQUARE},
    {"triangle", Controller::PAD_BUTTON_TRIANGLE},
    {"l1", Controller::PAD_BUTTON_L1},
    {"r1", Controller::PAD_BUTTON_R1},
    {"l2", Controller::PAD_BUTTON_L2},
    {"r2", Controller::PAD_BUTTON_R2},
    {"l3", Controller::PAD_BUTTON_L3},
    {"r3", Controller::PAD_BUTTON_R3},
    {"options", Controller::PAD_BUTTON_OPTIONS},
    {"touchpad", Controller::PAD_BUTTON_TOUCH_PAD},
    {"up", Controller::PAD_BUTTON_UP},
    {"down", Controller::PAD_BUTTON_DOWN},
    {"left", Controller::PAD_BUTTON_LEFT},
    {"right", Controller::PAD_BUTTON_RIGHT},
    // The guest pad data has no bit for these; they are accepted so a map can name them.
    {"share", 0},
    {"create", 0},
    {"ps", 0},
}};

constexpr std::string_view DEFAULT_BUTTONS =
    "b0=cross b1=circle b2=square b3=triangle b4=l1 b5=r1 b6=options b7=share b8=l3 b9=r3 b10=ps "
    "b11=touchpad";

bool ButtonFromName(std::string_view name, uint32_t* button) {
	for (const auto& [n, b]: BUTTON_NAMES) {
		if (n == name) {
			*button = b;
			return true;
		}
	}
	return false;
}

std::string_view ButtonName(uint32_t button) {
	for (const auto& [n, b]: BUTTON_NAMES) {
		if (b == button) {
			return n;
		}
	}
	return "?";
}

bool ParseInt(std::string_view s, int min, int max, int* out) {
	if (s == "none") {
		*out = -1;
		return min <= -1;
	}
	const std::string str(s);
	char*             end = nullptr;
	const long        v   = std::strtol(str.c_str(), &end, 10);
	if (str.empty() || *end != '\0' || v < min || v > max) {
		return false;
	}
	*out = static_cast<int>(v);
	return true;
}

bool ParseFloat(std::string_view s, float min, float max, float* out) {
	const std::string str(s);
	char*             end = nullptr;
	const float       v   = std::strtof(str.c_str(), &end);
	if (str.empty() || *end != '\0' || !std::isfinite(v) || v < min || v > max) {
		return false;
	}
	*out = v;
	return true;
}

bool ParseItems(std::string_view text, WheelMap* map, std::string* error) {
	while (!text.empty()) {
		const auto start = text.find_first_not_of(", ");
		if (start == std::string_view::npos) {
			break;
		}
		text            = text.substr(start);
		const auto end  = text.find_first_of(", ");
		const auto item = text.substr(0, end);
		text            = end == std::string_view::npos ? std::string_view() : text.substr(end);

		const auto split = item.find('=');
		const auto key   = item.substr(0, split);
		const auto value = split == std::string_view::npos ? std::string_view() : item.substr(split + 1);
		int        n     = 0;
		bool       ok    = false;
		if (key == "steer") {
			ok = ParseInt(value, 0, 63, &map->steer_axis);
		} else if (key == "gain") {
			ok = ParseFloat(value, -100.0f, 100.0f, &map->gain);
		} else if (key == "dz") {
			ok = ParseFloat(value, 0.0f, 0.9f, &map->deadzone);
		} else if (key == "adz") {
			ok = ParseFloat(value, 0.0f, 0.9f, &map->anti_deadzone);
		} else if (key == "throttle") {
			ok = ParseInt(value, -1, 63, &map->throttle_axis);
		} else if (key == "brake") {
			ok = ParseInt(value, -1, 63, &map->brake_axis);
		} else if (key == "clutch") {
			ok = ParseInt(value, -1, 63, &map->clutch_axis);
		} else if (key == "clutchto") {
			ok = ButtonFromName(value, &map->clutch_button);
		} else if (key == "pdz") {
			ok = ParseInt(value, 0, 254, &map->pedal_threshold);
		} else if (key.size() > 1 && key[0] == 'b' && ParseInt(key.substr(1), 0, 127, &n)) {
			ok = ButtonFromName(value, &map->buttons[n]);
		}
		if (!ok) {
			*error = std::string(item);
			return false;
		}
	}
	return true;
}

} // namespace

bool ParseWheelMap(std::string_view text, WheelMap* map, std::string* error) {
	*map = {};
	return ParseItems(DEFAULT_BUTTONS, map, error) && ParseItems(text, map, error);
}

std::string DescribeWheelMap(const WheelMap& map) {
	auto text = "steer=" + std::to_string(map.steer_axis) + " gain=" + std::to_string(map.gain) +
	            " dz=" + std::to_string(map.deadzone) + " adz=" + std::to_string(map.anti_deadzone) +
	            " throttle=" +
	            std::to_string(map.throttle_axis) + " brake=" + std::to_string(map.brake_axis) +
	            " clutch=" + std::to_string(map.clutch_axis) + " clutchto=" +
	            std::string(ButtonName(map.clutch_button)) +
	            " pdz=" + std::to_string(map.pedal_threshold);
	for (std::size_t i = 0; i < map.buttons.size(); i++) {
		if (map.buttons[i] != 0) {
			text += " b" + std::to_string(i) + "=" + std::string(ButtonName(map.buttons[i]));
		}
	}
	return text;
}

int WheelSteer(const WheelMap& map, int raw) {
	float x = std::clamp(static_cast<float>(raw) / 32767.0f, -1.0f, 1.0f);
	const float a = std::abs(x);
	if (a <= map.deadzone) {
		return 128;
	}
	x = std::clamp(std::copysign((a - map.deadzone) / (1.0f - map.deadzone), x) * map.gain, -1.0f,
	               1.0f);
	// adz: start just past the game's own stick deadzone, so a small turn already steers.
	if (map.anti_deadzone > 0.0f && x != 0.0f) {
		x = std::copysign(map.anti_deadzone + (1.0f - map.anti_deadzone) * std::abs(x), x);
	}
	// 0 is full left and 255 full right, as on a pad stick.
	return static_cast<int>(std::lround(128.0f + x * (x < 0.0f ? 128.0f : 127.0f)));
}

int WheelPedal(int raw, int threshold) {
	const int v = Controller::controller_get_axis(SDL_JOYSTICK_AXIS_MIN, SDL_JOYSTICK_AXIS_MAX, raw);
	return v <= threshold ? 0 : v;
}

bool WheelMatches(std::string_view option, std::string_view name, uint16_t vid, uint16_t pid) {
	if (const auto colon = option.find(':'); colon != std::string_view::npos) {
		const std::string v(option.substr(0, colon));
		const std::string p(option.substr(colon + 1));
		return std::strtoul(v.c_str(), nullptr, 16) == vid &&
		       std::strtoul(p.c_str(), nullptr, 16) == pid;
	}
	constexpr std::array<std::string_view, 5> KEYS = {"Wheel", "RS50", "G29", "G923", "Racing"};
	return std::any_of(KEYS.begin(), KEYS.end(),
	                   [name](std::string_view key) { return name.find(key) != std::string_view::npos; });
}

WheelInput::WheelInput(std::string option, const WheelMap& map, bool log)
    : m_option(std::move(option)), m_map(map), m_log(log) {}

WheelInput::~WheelInput() {
	if (m_joystick != nullptr) {
		SDL_CloseJoystick(m_joystick);
	}
}

bool WheelInput::OnEvent(const SDL_Event& event) {
	switch (event.type) {
		case SDL_EVENT_JOYSTICK_ADDED: {
			const auto id = event.jdevice.which;
			if (m_joystick != nullptr || SDL_IsGamepad(id)) {
				return false;
			}
			const char* name = SDL_GetJoystickNameForID(id);
			if (!WheelMatches(m_option, name != nullptr ? name : "", SDL_GetJoystickVendorForID(id),
			                  SDL_GetJoystickProductForID(id))) {
				return false;
			}
			m_joystick = SDL_OpenJoystick(id);
			if (m_joystick == nullptr) {
				LOGF("KYTY_WHEEL: cannot open %s: %s\n", name, SDL_GetError());
				return false;
			}
			m_id          = id;
			m_steer       = SDL_GetJoystickAxis(m_joystick, m_map.steer_axis);
			m_pedal       = {-32768, -32768, -32768};
			m_button_down = {};
			m_hat         = 0;
			LOGF("KYTY_WHEEL: opened '%s' (%04x:%04x, %d axes, %d buttons)\n", name,
			     SDL_GetJoystickVendor(m_joystick), SDL_GetJoystickProduct(m_joystick),
			     SDL_GetNumJoystickAxes(m_joystick), SDL_GetNumJoystickButtons(m_joystick));
			return Update();
		}
		case SDL_EVENT_JOYSTICK_REMOVED:
			if (m_joystick == nullptr || event.jdevice.which != m_id) {
				return false;
			}
			LOGF("KYTY_WHEEL: removed\n");
			SDL_CloseJoystick(m_joystick);
			m_joystick = nullptr;
			m_pad      = {};
			return true;
		case SDL_EVENT_JOYSTICK_AXIS_MOTION: {
			if (m_joystick == nullptr || event.jaxis.which != m_id) {
				return false;
			}
			const int axis = event.jaxis.axis;
			const int axes[3] = {m_map.throttle_axis, m_map.brake_axis, m_map.clutch_axis};
			if (axis == m_map.steer_axis) {
				m_steer = event.jaxis.value;
			}
			for (int i = 0; i < 3; i++) {
				if (axis == axes[i]) {
					m_pedal[i] = event.jaxis.value;
				}
			}
			return Update();
		}
		case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
		case SDL_EVENT_JOYSTICK_BUTTON_UP:
			if (m_joystick == nullptr || event.jbutton.which != m_id ||
			    event.jbutton.button >= m_button_down.size()) {
				return false;
			}
			if (m_log && event.jbutton.down) {
				LOGF("KYTY_WHEEL button %d\n", event.jbutton.button);
			}
			m_button_down[event.jbutton.button] = event.jbutton.down;
			return Update();
		case SDL_EVENT_JOYSTICK_HAT_MOTION:
			if (m_joystick == nullptr || event.jhat.which != m_id || event.jhat.hat != 0) {
				return false;
			}
			m_hat = event.jhat.value;
			return Update();
		default: return false;
	}
}

bool WheelInput::Update() {
	WheelPad pad;
	for (std::size_t i = 0; i < m_button_down.size(); i++) {
		if (m_button_down[i]) {
			pad.buttons |= m_map.buttons[i];
		}
	}
	pad.buttons |= ((m_hat & SDL_HAT_UP) != 0 ? Controller::PAD_BUTTON_UP : 0) |
	               ((m_hat & SDL_HAT_DOWN) != 0 ? Controller::PAD_BUTTON_DOWN : 0) |
	               ((m_hat & SDL_HAT_LEFT) != 0 ? Controller::PAD_BUTTON_LEFT : 0) |
	               ((m_hat & SDL_HAT_RIGHT) != 0 ? Controller::PAD_BUTTON_RIGHT : 0);
	if (m_map.clutch_axis >= 0 && WheelPedal(m_pedal[2], 0) > 127) {
		pad.buttons |= m_map.clutch_button;
	}
	pad.left_x = WheelSteer(m_map, m_steer);
	pad.r2     = m_map.throttle_axis >= 0 ? WheelPedal(m_pedal[0], m_map.pedal_threshold) : 0;
	pad.l2     = m_map.brake_axis >= 0 ? WheelPedal(m_pedal[1], m_map.pedal_threshold) : 0;
	if ((pad.buttons & Controller::PAD_BUTTON_R2) != 0) {
		pad.r2 = 255;
	}
	if ((pad.buttons & Controller::PAD_BUTTON_L2) != 0) {
		pad.l2 = 255;
	}
	pad.buttons &= ~(Controller::PAD_BUTTON_L2 | Controller::PAD_BUTTON_R2);
	const bool changed = pad != m_pad;
	m_pad              = pad;
	return changed;
}

} // namespace Libs::Graphics
