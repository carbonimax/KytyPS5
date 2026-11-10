#ifndef KYTY_COMMON_VIDEO_TRACE_H_
#define KYTY_COMMON_VIDEO_TRACE_H_

#include <cstdint>

// KYTY_AVPLAYER_TRACE=<file> (default off): measurement of the avPlayer video frame buffers. One text
// line per event, times in microseconds of the steady clock:
//   W t0 t1 base pts            decoder wrote a frame into a buffer (t0..t1)
//   V t base pts clock prev     Video() delivered a frame; `prev` went back to the decoder
//   v t clock head              Video() delivered nothing (`head`: next ready frame, -1 none)
//   A t pts before after        Audio() re-synced the video clock from `before` to `after` (ms)
//   S|T|B t0 t1 base off size content delivered dgen gen0 gen1 w0 w1 held [extra]
//                               GPU-side read of guest bytes inside a video buffer: texture lookup
//                               (S), texture upload (T) or buffer cache upload (B). `content` is the
//                               pts last written into the buffer, `delivered` the pts it had when last
//                               handed to the game, dgen the write generation at that delivery
//                               (gen1 != dgen: rewritten since), gen0/gen1 the generation at t0/t1,
//                               w0/w1 a decoder write in progress at t0/t1, held the game's frame.
namespace Common::VideoTrace {

[[nodiscard]] bool     Enabled() noexcept;
[[nodiscard]] uint64_t NowUs() noexcept;
void                   Line(const char* format, ...) noexcept;

void Register(uint64_t base, uint64_t size) noexcept;
void Unregister(uint64_t base) noexcept;
void BeginWrite(uint64_t base) noexcept;
void EndWrite(uint64_t base, int64_t pts, uint64_t t0) noexcept;
void Deliver(uint64_t base, int64_t pts, int64_t clock, uint64_t prev) noexcept;

struct Mark {
	int      slot    = -1;
	uint64_t t0      = 0;
	uint32_t gen     = 0;
	bool     writing = false;
};
// slot -1: the range touches no video buffer (EndRead then does nothing).
[[nodiscard]] Mark BeginRead(uint64_t address, uint64_t size) noexcept;
void EndRead(const Mark& mark, char kind, uint64_t address, uint64_t size, const char* extra = "") noexcept;

} // namespace Common::VideoTrace

#endif /* KYTY_COMMON_VIDEO_TRACE_H_ */
