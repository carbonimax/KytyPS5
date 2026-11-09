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
// `frame`). True while a draw of the key may still be skipped at `frame`.
[[nodiscard]] constexpr bool AsyncPipelineMaySkip(uint64_t& first_skip, uint64_t frame) {
	if (first_skip == ASYNC_PIPELINE_NOT_SKIPPED) {
		first_skip = frame;
	}
	return frame - first_skip < ASYNC_PIPELINE_SKIP_FRAMES;
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

// KYTY_PIPELINE_FAST_FIRST: optimised replacements of unoptimised pipelines run only while no miss
// is queued, on at most `cap` workers at once, so a new miss still finds a free worker.
template <class Queue>
[[nodiscard]] bool AsyncPipelineHasWork(const Queue& skipped, const Queue& predicted,
                                        const Queue& upgrades, uint32_t upgrading, uint32_t cap) {
	return !skipped.empty() || !predicted.empty() || (!upgrades.empty() && upgrading < cap);
}

// The next job once AsyncPipelineHasWork: a miss (AsyncPipelinePop), else the oldest replacement
// (`upgrade` set).
template <class Queue>
[[nodiscard]] typename Queue::value_type AsyncPipelinePopAny(Queue& skipped, Queue& predicted,
                                                             Queue& upgrades, bool& upgrade) {
	upgrade = skipped.empty() && predicted.empty();
	return upgrade ? AsyncPipelinePop(upgrades, upgrades) : AsyncPipelinePop(skipped, predicted);
}

// A created replacement: `current` takes it and the handle to retire (destroyed once the command
// buffers that may use it completed) is returned; an empty handle while it is not ready.
template <class Handle>
[[nodiscard]] Handle PipelineSwapReplacement(Handle& current, Handle& replacement, bool ready) {
	if (!ready || replacement == Handle {}) {
		return Handle {};
	}
	return std::exchange(current, std::exchange(replacement, Handle {}));
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCPIPELINEPOLICY_H_ */
