#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_WHEEL_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_WHEEL_H_

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

union SDL_Event;
struct SDL_Joystick;

namespace Libs::Graphics {

// KYTY_WHEEL: a racing wheel read as a plain SDL joystick and fed to the first player's pad.
struct WheelMap {
	int   steer_axis    = 0;
	float gain          = 1.0f; // steering: guest stick = clamp(raw * gain)
	float deadzone      = 0.0f; // steering, fraction of half travel
	int   throttle_axis = 3;    // -> R2, -1 for none
	int   brake_axis    = 4;    // -> L2, -1 for none
	int   clutch_axis   = 5;
	uint32_t clutch_button   = 0; // pad button pressed past half travel, 0 for none
	int      pedal_threshold = 8; // trigger values at or below it read 0, so R2/L2 bits stay off
	// Joystick button -> pad button; 0 for none.
	std::array<uint32_t, 128> buttons {};
};

// What the wheel contributes to the pad. L2/R2 travel through the triggers only.
struct WheelPad {
	uint32_t buttons = 0;
	int      left_x  = 128;
	int      l2      = 0;
	int      r2      = 0;

	bool operator==(const WheelPad&) const = default;
};

// Parses "key=value" items separated by commas or spaces over the RS50 defaults.
[[nodiscard]] bool        ParseWheelMap(std::string_view text, WheelMap* map, std::string* error);
[[nodiscard]] std::string DescribeWheelMap(const WheelMap& map);
[[nodiscard]] int         WheelSteer(const WheelMap& map, int raw);
[[nodiscard]] int         WheelPedal(int raw, int threshold);
// option is "1" (match the name) or "VVVV:PPPP" in hex.
[[nodiscard]] bool WheelMatches(std::string_view option, std::string_view name, uint16_t vid,
                                uint16_t pid);

class WheelInput {
public:
	WheelInput(std::string option, const WheelMap& map, bool log);
	~WheelInput();
	WheelInput(const WheelInput&)            = delete;
	WheelInput& operator=(const WheelInput&) = delete;

	// True when Pad() changed.
	bool OnEvent(const SDL_Event& event);

	[[nodiscard]] bool            Connected() const { return m_joystick != nullptr; }
	[[nodiscard]] const WheelPad& Pad() const { return m_pad; }

private:
	bool Update();

	std::string   m_option;
	WheelMap      m_map;
	bool          m_log      = false;
	SDL_Joystick* m_joystick = nullptr;
	uint32_t      m_id       = 0;
	int           m_steer    = 0;
	// Pedals read as released until their first event: an unread axis is 0, i.e. half travel.
	std::array<int, 3>         m_pedal {-32768, -32768, -32768};
	std::array<bool, 128>      m_button_down {};
	uint8_t                    m_hat = 0;
	WheelPad                   m_pad;
};

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_PRESENTATION_WINDOW_WHEEL_H_ */
