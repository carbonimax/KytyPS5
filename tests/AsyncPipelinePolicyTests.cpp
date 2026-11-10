#include "graphics/host_gpu/renderer/pipeline/asyncPipelinePolicy.h"

#include <cstdio>
#include <cstdlib>
#include <deque>

namespace {

using Libs::Graphics::ASYNC_PIPELINE_NOT_SKIPPED;
using Libs::Graphics::ASYNC_PIPELINE_SKIP_FRAMES;
using Libs::Graphics::AsyncPipelineMaySkip;
using Libs::Graphics::AsyncPipelinePop;

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
	Check(AsyncPipelineMaySkip(first, 100 + 299, 300) && !AsyncPipelineMaySkip(first, 100 + 300, 300),
	      "a longer cap was not applied from the same start");

	uint64_t from_zero = ASYNC_PIPELINE_NOT_SKIPPED;
	Check(AsyncPipelineMaySkip(from_zero, 0) && from_zero == 0, "frame 0 was not a valid start");
	Check(!AsyncPipelineMaySkip(from_zero, ASYNC_PIPELINE_SKIP_FRAMES), "the cap from frame 0 was not applied");

	std::deque<int> skipped {1, 2};
	std::deque<int> predicted {10, 11};
	Check(AsyncPipelinePop(skipped, predicted) == 1, "the oldest skipped key was not first");
	skipped.push_back(3);
	Check(AsyncPipelinePop(skipped, predicted) == 2, "a newer skipped key went ahead of an older one");
	Check(AsyncPipelinePop(skipped, predicted) == 3, "a skipped key went after a predicted one");
	Check(AsyncPipelinePop(skipped, predicted) == 10 && AsyncPipelinePop(skipped, predicted) == 11,
	      "the predicted keys were not taken in order");
	Check(skipped.empty() && predicted.empty(), "a pop left a job behind");

	std::puts("AsyncPipelinePolicyTests: all cases passed");
	return 0;
}
