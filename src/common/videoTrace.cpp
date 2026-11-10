#include "common/videoTrace.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Common::VideoTrace {

namespace {

struct Slot {
	uint64_t base      = 0;
	uint64_t size      = 0;
	int64_t  content   = -1;
	int64_t  delivered = -1;
	uint32_t gen       = 0;
	uint32_t delivered_gen = 0;
	bool     writing   = false;
	bool     held      = false;
};

struct State {
	std::mutex             mutex;
	std::FILE*             file = nullptr;
	std::array<Slot, 64>   slots {};
	std::atomic<int>       active {0};
	const std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
};

State& Get() noexcept {
	static State* state = [] {
		auto*       s    = new State;
		const char* path = std::getenv("KYTY_AVPLAYER_TRACE");
		if (path != nullptr && path[0] != '\0') {
			s->file = std::fopen(path, "w");
		}
		return s;
	}();
	return *state;
}

Slot* FindLocked(State& s, uint64_t base) noexcept {
	for (auto& slot: s.slots) {
		if (slot.size != 0 && slot.base == base) {
			return &slot;
		}
	}
	return nullptr;
}

void LineLocked(State& s, const char* format, std::va_list args) noexcept {
	std::vfprintf(s.file, format, args);
	std::fputc('\n', s.file);
	std::fflush(s.file);
}

void LineLocked(State& s, const char* format, ...) noexcept {
	std::va_list args;
	va_start(args, format);
	LineLocked(s, format, args);
	va_end(args);
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = Get().file != nullptr;
	return enabled;
}

uint64_t NowUs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - Get().epoch).count());
}

void Line(const char* format, ...) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::va_list    args;
	va_start(args, format);
	std::lock_guard lock(s.mutex);
	LineLocked(s, format, args);
	va_end(args);
}

void Register(uint64_t base, uint64_t size) noexcept {
	if (!Enabled() || base == 0) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	for (auto& slot: s.slots) {
		if (slot.size == 0) {
			slot = {.base = base, .size = size};
			s.active.fetch_add(1, std::memory_order_relaxed);
			LineLocked(s, "R %" PRIu64 " 0x%" PRIx64 " 0x%" PRIx64, NowUs(), base, size);
			return;
		}
	}
	LineLocked(s, "R %" PRIu64 " 0x%" PRIx64 " full", NowUs(), base);
}

void Unregister(uint64_t base) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (auto* slot = FindLocked(s, base)) {
		*slot = {};
		s.active.fetch_sub(1, std::memory_order_relaxed);
		LineLocked(s, "X %" PRIu64 " 0x%" PRIx64, NowUs(), base);
	}
}

void BeginWrite(uint64_t base) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (auto* slot = FindLocked(s, base)) {
		slot->gen++;
		slot->writing = true;
	}
}

void EndWrite(uint64_t base, int64_t pts, uint64_t t0) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (auto* slot = FindLocked(s, base)) {
		slot->writing = false;
		slot->content = pts;
	}
	LineLocked(s, "W %" PRIu64 " %" PRIu64 " 0x%" PRIx64 " %" PRId64, t0, NowUs(), base, pts);
}

void Deliver(uint64_t base, int64_t pts, int64_t clock, uint64_t prev) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (auto* slot = FindLocked(s, prev)) {
		slot->held = false;
	}
	if (auto* slot = FindLocked(s, base)) {
		slot->held      = true;
		slot->delivered = pts;
		slot->delivered_gen = slot->gen;
	}
	LineLocked(s, "V %" PRIu64 " 0x%" PRIx64 " %" PRId64 " %" PRId64 " 0x%" PRIx64, NowUs(), base, pts, clock, prev);
}

Mark BeginRead(uint64_t address, uint64_t size) noexcept {
	Mark mark;
	if (!Enabled()) {
		return mark;
	}
	auto& s = Get();
	if (s.active.load(std::memory_order_relaxed) == 0) {
		return mark;
	}
	std::lock_guard lock(s.mutex);
	for (size_t i = 0; i < s.slots.size(); i++) {
		const auto& slot = s.slots[i];
		if (slot.size != 0 && address < slot.base + slot.size && slot.base < address + size) {
			mark = {.slot = static_cast<int>(i), .t0 = NowUs(), .gen = slot.gen, .writing = slot.writing};
			break;
		}
	}
	return mark;
}

void EndRead(const Mark& mark, char kind, uint64_t address, uint64_t size, const char* extra) noexcept {
	if (mark.slot < 0) {
		return;
	}
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	const auto&     slot = s.slots[mark.slot];
	LineLocked(s, "%c %" PRIu64 " %" PRIu64 " 0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 " %" PRId64 " %" PRId64 " %u %u %u %d %d %d %s",
	           kind, mark.t0, NowUs(), slot.base, address - slot.base, size, slot.content, slot.delivered, slot.delivered_gen, mark.gen,
	           slot.gen, mark.writing ? 1 : 0, slot.writing ? 1 : 0, slot.held ? 1 : 0, extra);
}

} // namespace Common::VideoTrace
