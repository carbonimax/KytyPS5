#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_

#include <cstdint>
#include <utility>

namespace Libs::Graphics {

// KYTY_ASYNC_PIPELINE: guest frames a draw may be skipped while its pipeline is created; then it
// waits like any miss. Bounds a hole (missing geometry) to ~1-2 s even when a creation backlog
// would keep a pipeline queued for much longer.
inline constexpr uint64_t ASYNC_PIPELINE_SKIP_FRAMES = 30;
inline constexpr uint64_t ASYNC_PIPELINE_NOT_SKIPPED = UINT64_MAX;

// `first_skip`: guest frame of the key's first skip (ASYNC_PIPELINE_NOT_SKIPPED: none yet, set to
// `frame`). True while a draw of the key may still be skipped at `frame` (`cap` frames).
[[nodiscard]] constexpr bool AsyncPipelineMaySkip(uint64_t& first_skip, uint64_t frame,
                                                  uint64_t cap = ASYNC_PIPELINE_SKIP_FRAMES) {
	if (first_skip == ASYNC_PIPELINE_NOT_SKIPPED) {
		first_skip = frame;
	}
	return frame - first_skip < cap;
}

// The next job for a pipeline worker: the skipped keys first, in first-skip order (the nearest cap
// first), then the predicted ones. LIFO among skipped keys left the oldest queued until their cap,
// and the draw then created it inline. One of the queues is not empty.
template <class Queue>
[[nodiscard]] typename Queue::value_type AsyncPipelinePop(Queue& skipped, Queue& predicted) {
	auto& from = skipped.empty() ? predicted : skipped;
	auto  job  = std::move(from.front());
	from.pop_front();
	return job;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_ */
