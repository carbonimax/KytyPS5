#include "graphics/host_gpu/renderer/pipeline/asyncPipelinePolicy.h"

#include <cstdio>
#include <cstdlib>
#include <deque>

namespace {

using Libs::Graphics::ASYNC_PIPELINE_NOT_SKIPPED;
using Libs::Graphics::ASYNC_PIPELINE_SKIP_FRAMES;
using Libs::Graphics::AsyncPipelineMaySkip;
using Libs::Graphics::AsyncPipelineHasWork;
using Libs::Graphics::AsyncPipelinePop;
using Libs::Graphics::AsyncPipelinePopAny;
using Libs::Graphics::PipelineSwapReplacement;

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

	std::deque<int> skipped {1, 2};
	std::deque<int> predicted {10, 11};
	Check(AsyncPipelinePop(skipped, predicted) == 1, "the oldest skipped key was not first");
	skipped.push_back(3);
	Check(AsyncPipelinePop(skipped, predicted) == 2, "a newer skipped key went ahead of an older one");
	Check(AsyncPipelinePop(skipped, predicted) == 3, "a skipped key went after a predicted one");
	Check(AsyncPipelinePop(skipped, predicted) == 10 && AsyncPipelinePop(skipped, predicted) == 11,
	      "the predicted keys were not taken in order");
	Check(skipped.empty() && predicted.empty(), "a pop left a job behind");

	// KYTY_PIPELINE_FAST_FIRST: replacements after every miss, at most `cap` at once.
	std::deque<int> upgrades {20, 21};
	Check(!AsyncPipelineHasWork(skipped, predicted, upgrades, 2, 2), "a replacement ran past the cap");
	Check(AsyncPipelineHasWork(skipped, predicted, upgrades, 1, 2), "a free replacement slot was not used");
	predicted.push_back(12);
	Check(AsyncPipelineHasWork(skipped, predicted, upgrades, 2, 2), "a miss waited for the replacement cap");
	bool upgrade = true;
	Check(AsyncPipelinePopAny(skipped, predicted, upgrades, upgrade) == 12 && !upgrade,
	      "a replacement went ahead of a miss");
	Check(AsyncPipelinePopAny(skipped, predicted, upgrades, upgrade) == 20 && upgrade,
	      "the oldest replacement was not taken once no miss was queued");
	std::deque<int> none;
	Check(!AsyncPipelineHasWork(none, none, none, 0, 2), "an empty pool had work");

	// Swap bookkeeping (0 = no handle): nothing before the replacement is created; then the
	// unoptimised handle comes back once, to be retired, and the key keeps the replacement.
	int current = 7, replacement = 0;
	Check(PipelineSwapReplacement(current, replacement, false) == 0 && current == 7,
	      "a replacement still being created was swapped in");
	replacement = 9;
	Check(PipelineSwapReplacement(current, replacement, false) == 0 && current == 7 && replacement == 9,
	      "a replacement not marked created was swapped in");
	Check(PipelineSwapReplacement(current, replacement, true) == 7, "the replaced handle was not retired");
	Check(current == 9 && replacement == 0, "the key did not keep its replacement");
	Check(PipelineSwapReplacement(current, replacement, true) == 0 && current == 9,
	      "a second swap retired the replacement");

	std::puts("AsyncPipelinePolicyTests: all cases passed");
	return 0;
}
