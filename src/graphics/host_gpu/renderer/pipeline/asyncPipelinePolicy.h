#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_

#include <cstdint>

namespace Libs::Graphics {

// KYTY_ASYNC_PIPELINE: guest frames a draw may be skipped while its pipeline is created; then it
// waits like any miss. Bounds a hole (missing geometry) to ~1-2 s even when a creation backlog
// would keep a pipeline queued for much longer.
inline constexpr uint64_t ASYNC_PIPELINE_SKIP_FRAMES = 30;
inline constexpr uint64_t ASYNC_PIPELINE_NOT_SKIPPED = UINT64_MAX;

// `first_skip`: guest frame of the key's first skip (ASYNC_PIPELINE_NOT_SKIPPED: none yet, set to
// `frame`). True while a draw of the key may still be skipped at `frame`.
[[nodiscard]] constexpr bool AsyncPipelineMaySkip(uint64_t& first_skip, uint64_t frame) {
	if (first_skip == ASYNC_PIPELINE_NOT_SKIPPED) {
		first_skip = frame;
	}
	return frame - first_skip < ASYNC_PIPELINE_SKIP_FRAMES;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_ */
