#include "graphics/host_gpu/renderer/cache/lateReadSub.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace Libs::Graphics::LateReadSub;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LateReadSubTests: failed: %s\n", message);
		std::abort();
	}
}

void TestDiffAndIntersection() {
	constexpr uint64_t   size = Block * 70 + 17; // 71 blocks, the last one partial, two words
	std::vector<uint8_t> before(size, 0), now(size, 0);
	now[0]              = 1;            // block 0
	now[Block * 3 + 5]  = 1;            // block 3
	now[Block * 65]     = 1;            // block 65 (second word)
	now[size - 1]       = 1;            // block 70 (partial)
	Bits changed, read;
	Reset(changed, size);
	Reset(read, size);
	Check(changed.size() == 2, "two words for 71 blocks");
	Diff(before.data(), now.data(), size, changed);
	Check(Count(changed) == 4, "four rewritten blocks");
	// Bytes [Block * 3 - 1, Block * 66): blocks 2..65.
	Mark(read, Block * 3 - 1, Block * 66);
	Check(Count(read) == 64, "read blocks 2..65");
	Check(CountAnd(read, changed) == 2, "blocks 3 and 65 are read and rewritten");
	const auto runs = Runs(changed);
	Check(runs.size() == 4 && runs[0].first == 0 && runs[1].first == 3 && runs[2].first == 65 &&
	          runs[3].first == 70 && runs[3].second == 1,
	      "runs of rewritten blocks");
	Mark(read, Block * 69, size); // the partial last block counts
	Check(Count(read) == 66, "mark up to the range end");
	Bits all;
	Reset(all, size);
	Or(all, changed);
	Or(all, read);
	Check(Count(all) == 66 + 1, "union adds block 0");
}

void TestParseRanges() {
	const auto ranges = ParseRanges("0x55aaf0000+0x118000,bad,0x10+0,4096+256");
	Check(ranges.size() == 2, "two valid ranges");
	Check(ranges[0].first == 0x55aaf0000ull && ranges[0].second == 0x118000ull, "first range");
	Check(ranges[1].first == 4096 && ranges[1].second == 256, "decimal range");
	Check(ParseRanges(nullptr).empty() && ParseRanges("").empty(), "unset is empty");
}

} // namespace

int main() {
	TestDiffAndIntersection();
	TestParseRanges();
	std::printf("LateReadSubTests: ok\n");
	return 0;
}
