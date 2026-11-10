#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// What-if experiment, off by default (measurement only).
// KYTY_WHATIF_CP_SPIN_CLASS=<letters> turns on per-class counts ("PERF whatif spin" lines) and
// selects the classes that busy-wait KYTY_WHATIF_CP_SPIN_NS=N ns on the GPU thread (front):
//   q = per draw queued to the CP back thread (after its record is published),
//   d = per compute dispatch (direct or indirect, any ring),
//   i = per draw run on the GPU thread itself (fallbacks and inline draws).
// It dilates front time of one class to measure frame time per ms of front time.
namespace Libs::Graphics::WhatIf {

enum class SpinClass : int { Queued = 0, Dispatch = 1, Inline = 2, Count = 3 };

struct SpinState {
	bool                  counting = false;
	bool                  spin[3]  = {false, false, false};
	uint32_t              ns       = 0;
	std::atomic<uint64_t> items[3] {};
	std::atomic<uint64_t> spun_ns[3] {};

	SpinState() {
		const char* cls = std::getenv("KYTY_WHATIF_CP_SPIN_CLASS");
		const char* n   = std::getenv("KYTY_WHATIF_CP_SPIN_NS");
		counting        = cls != nullptr && *cls != '\0';
		ns              = n != nullptr ? static_cast<uint32_t>(std::strtoul(n, nullptr, 0)) : 0u;
		if (counting) {
			spin[0] = std::strchr(cls, 'q') != nullptr;
			spin[1] = std::strchr(cls, 'd') != nullptr;
			spin[2] = std::strchr(cls, 'i') != nullptr;
		}
	}
};

inline SpinState& Spins() {
	static SpinState s;
	return s;
}

inline void Spin(SpinClass c) {
	auto&      s = Spins();
	const auto i = static_cast<int>(c);
	if (!s.counting) {
		return;
	}
	s.items[i].fetch_add(1, std::memory_order_relaxed);
	if (!s.spin[i] || s.ns == 0) {
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	const auto end   = start + std::chrono::nanoseconds(s.ns);
	auto       now   = start;
	while (now < end) {
		now = std::chrono::steady_clock::now();
	}
	s.spun_ns[i].fetch_add(static_cast<uint64_t>(std::chrono::nanoseconds(now - start).count()),
	                       std::memory_order_relaxed);
}

inline void Report(double seconds) {
	auto& s = Spins();
	if (!s.counting) {
		return;
	}
	uint64_t n[3];
	uint64_t t[3];
	for (int i = 0; i < 3; i++) {
		n[i] = s.items[i].exchange(0, std::memory_order_relaxed);
		t[i] = s.spun_ns[i].exchange(0, std::memory_order_relaxed);
	}
	std::printf("PERF whatif spin %.1fs: ns=%u queued=%" PRIu64 "(%.1fms) dispatch=%" PRIu64 "(%.1fms) inline=%" PRIu64
	            "(%.1fms)\n",
	            seconds, s.ns, n[0], static_cast<double>(t[0]) / 1e6, n[1], static_cast<double>(t[1]) / 1e6, n[2],
	            static_cast<double>(t[2]) / 1e6);
}

} // namespace Libs::Graphics::WhatIf

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WHATIF_H_ */
