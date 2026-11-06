#include "graphics/host_gpu/renderer/pipeline/asyncPipelinePolicy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::ASYNC_PIPELINE_NOT_SKIPPED;
using Libs::Graphics::ASYNC_PIPELINE_SKIP_FRAMES;
using Libs::Graphics::AsyncPipelineMaySkip;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AsyncPipelinePolicyTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	uint64_t first = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(first, 100) && first == 100, "the first skip did not start the cap");
	Check(AsyncPipelineMaySkip(first, 100), "a second draw in the same frame was refused");
	Check(AsyncPipelineMaySkip(first, 100 + ASYNC_PIPELINE_SKIP_FRAMES - 1),
	      "the last frame of the cap was refused");
	Check(!AsyncPipelineMaySkip(first, 100 + ASYNC_PIPELINE_SKIP_FRAMES), "the cap was not applied");
	Check(!AsyncPipelineMaySkip(first, 100 + 10 * ASYNC_PIPELINE_SKIP_FRAMES),
	      "a draw long after the cap was skipped");
	Check(first == 100, "a later skip moved the cap start");

	uint64_t from_zero = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(from_zero, 0) && from_zero == 0, "frame 0 was not a valid start");
	Check(!AsyncPipelineMaySkip(from_zero, ASYNC_PIPELINE_SKIP_FRAMES), "the cap from frame 0 was not applied");

	std::puts("AsyncPipelinePolicyTests: all cases passed");
	return 0;
}
