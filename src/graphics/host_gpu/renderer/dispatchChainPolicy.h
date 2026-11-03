#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHCHAINPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHCHAINPOLICY_H_

#include <cstdint>
#include <vector>

// Barrier choice for direct dispatches recorded back to back (KYTY_DISPATCH_CHAIN).
//
// Without chaining, a dispatch that writes memory waits for all earlier work (hazard barrier,
// ALL_COMMANDS -> COMPUTE) and every dispatch makes its accesses visible to all later work
// (COMPUTE -> ALL_COMMANDS). A chain keeps that outer pair but drops what lies between dispatches
// that follow each other with nothing else recorded in between:
// - the trailing barrier is deferred and recorded once, before the next command that is not a
//   chained dispatch (CommandBuffer::Handle) or at the end of the command buffer;
// - a dispatch of the chain waits for the previous ones (COMPUTE -> COMPUTE) when a guest sync
//   point separates them (cache flushes, acquires, releases, waits: the guest orders them there),
//   when it touches a resource that an earlier dispatch since the last barrier writes, or writes
//   one that such a dispatch reads (RAW, WAR, WAW on the host buffer range or image, or on the
//   guest memory behind them, as an image and a buffer may cover the same bytes), or when
//   either side reaches memory the bindings do not describe (address loads and stores, GDS
//   ordering, the image heap);
// - a writing dispatch still waits for earlier non-dispatch work once per chain (hazard barrier);
//   the first one already orders everything recorded before it ahead of all later dispatches.
namespace Libs::Graphics::DispatchChain {

// One resource range a dispatch reads or writes. Buffers: host buffer handle and byte range.
// Images: image identity (with ImageKey) over the whole image. Guest memory behind either:
// GuestMemory and the guest address range.
struct Access {
	uint64_t resource = 0;
	uint64_t begin    = 0;
	uint64_t end      = 0; // exclusive
	bool     write    = false;
};

inline constexpr uint64_t GuestMemory = 0;

[[nodiscard]] constexpr uint64_t ImageKey(uint32_t image_index) noexcept {
	return (uint64_t {1} << 63u) | image_index;
}

struct Accesses {
	std::vector<Access> ranges;
	bool                writes  = false; // the dispatch writes memory (it needs the hazard barrier)
	bool                unknown = false; // it reaches memory outside the described ranges
	void                Clear() {
		               ranges.clear();
		               writes  = false;
		               unknown = false;
	}
};

enum class Barrier : uint8_t {
	None,             // chained: no barrier before the dispatch
	ComputeToCompute, // ordered after the dispatches of the chain
	Hazard,           // ordered after all earlier work (as without chaining)
};

enum class Reason : uint8_t {
	Chained,  // same sync epoch, no overlap: no barrier
	Fresh,    // no chain open (something else was recorded since the last dispatch)
	Epoch,    // a guest sync point separates it from the chain
	Conflict, // RAW, WAR or WAW with a dispatch since the last barrier
	Unknown,  // address accesses, GDS or image heap on either side
	Cap,      // too many tracked ranges since the last barrier
	Hazard,   // a writer whose chain has no hazard barrier yet
};

// Ranges tracked since the last barrier; above this, the next dispatch waits.
inline constexpr size_t MaxTrackedRanges = 512;

struct State {
	bool                pending     = false; // deferred trailing barrier owed
	bool                hazard_done = false; // the chain's hazard barrier is recorded
	bool                unknown     = false; // a dispatch since the last barrier is unknown
	uint64_t            epoch       = 0;
	std::vector<Access> window; // accesses of the dispatches since the last barrier
	// KYTY_DISPATCH_CHAIN=2: guest address ranges of the same dispatches (verification only).
	std::vector<Access> guest_window;
	void                Reset() {
		               pending     = false;
		               hazard_done = false;
		               unknown     = false;
		               epoch       = 0;
		               window.clear();
		               guest_window.clear();
	}
};

[[nodiscard]] inline bool Overlaps(const Access& a, const Access& b) noexcept {
	return a.resource == b.resource && a.begin < b.end && b.begin < a.end && (a.write || b.write);
}

[[nodiscard]] inline bool Conflicts(const std::vector<Access>& window, const Accesses& next) noexcept {
	for (const auto& a: next.ranges) {
		for (const auto& b: window) {
			if (Overlaps(a, b)) {
				return true;
			}
		}
	}
	return false;
}

struct Decision {
	Barrier barrier = Barrier::None;
	Reason  reason  = Reason::Chained;
};

// The barrier a dispatch needs before it, given the chain state and the current guest sync epoch.
[[nodiscard]] inline Decision Decide(const State& state, const Accesses& next, uint64_t epoch) noexcept {
	if (!state.pending) {
		return next.writes ? Decision {Barrier::Hazard, Reason::Fresh} : Decision {Barrier::None, Reason::Fresh};
	}
	if (next.writes && !state.hazard_done) {
		return {Barrier::Hazard, Reason::Hazard};
	}
	if (state.epoch != epoch) {
		return {Barrier::ComputeToCompute, Reason::Epoch};
	}
	if (state.unknown || next.unknown) {
		return {Barrier::ComputeToCompute, Reason::Unknown};
	}
	if (state.window.size() + next.ranges.size() > MaxTrackedRanges) {
		return {Barrier::ComputeToCompute, Reason::Cap};
	}
	if (Conflicts(state.window, next)) {
		return {Barrier::ComputeToCompute, Reason::Conflict};
	}
	return {Barrier::None, Reason::Chained};
}

// State after recording the dispatch with the barrier Decide chose.
inline void Commit(State& state, const Accesses& next, uint64_t epoch, Barrier barrier) {
	if (!state.pending || barrier != Barrier::None) {
		// Every barrier orders all earlier commands: the window restarts with this dispatch.
		state.window.clear();
		state.unknown = false;
	}
	if (!state.pending) {
		state.hazard_done = false;
	}
	state.pending = true;
	state.hazard_done |= barrier == Barrier::Hazard;
	state.unknown |= next.unknown;
	state.epoch = epoch;
	state.window.insert(state.window.end(), next.ranges.begin(), next.ranges.end());
}

} // namespace Libs::Graphics::DispatchChain

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHCHAINPOLICY_H_
