#include "graphics/host_gpu/renderer/cache/refreshProf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::RefreshProf::DirtyBelow;
using Libs::Graphics::RefreshProf::Locate;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "RefreshProfTests: failed: %s\n", text);
		std::abort();
	}
}

// Guest mip chains run backwards: the smallest mips first, mip0 last.
constexpr uint64_t kOffsets[] = {0x1500, 0x500, 0x100, 0x0};
constexpr uint64_t kSizes[]   = {0x4000, 0x1000, 0x400, 0x100};
constexpr uint64_t kSize      = 0x5500;

void TestDirtyBelow() {
	Check(DirtyBelow(kOffsets, kSizes, 4, 0x500, kSize, 2), "dirty bytes only in mips 0-1, view samples from mip 2");
	Check(!DirtyBelow(kOffsets, kSizes, 4, 0x4ff, kSize, 2), "one dirty byte in sampled mip 2");
	Check(DirtyBelow(kOffsets, kSizes, 4, 0x1500, kSize, 1), "only mip 0 dirty, view samples from mip 1");
	Check(!DirtyBelow(kOffsets, kSizes, 4, 0x1500, kSize, 0), "view samples every mip");
	Check(!DirtyBelow(kOffsets, kSizes, 4, 0x1500, kSize, 4), "view samples no mip");
	Check(!DirtyBelow(kOffsets, kSizes, 4, kSize, kSize, 2), "nothing past the watch: no deferral");
	Check(!DirtyBelow(kOffsets, kSizes, 4, 0x500, 0x5000, 2), "a mip past the image size is not trusted");
	constexpr uint64_t empty[] = {0x1500, 0x500, 0x100, 0};
	constexpr uint64_t empty_sizes[] = {0x4000, 0x1000, 0x400, 0};
	Check(!DirtyBelow(empty, empty_sizes, 4, 0x500, kSize, 2), "an empty mip is not trusted");
}

void TestTailSharedBlock() {
	// Mips 2 and 3 share the tail block [0, 0x400).
	constexpr uint64_t offsets[] = {0x1400, 0x400, 0x0, 0x0};
	constexpr uint64_t sizes[]   = {0x4000, 0x1000, 0x400, 0x400};
	Check(DirtyBelow(offsets, sizes, 4, 0x400, 0x5400, 3), "tail untouched, view samples mip 3");
	Check(!DirtyBelow(offsets, sizes, 4, 0x300, 0x5400, 3), "write into the shared tail reaches sampled mip 3");
	Check(Locate(offsets, sizes, 4, 1, 0x50).mip == 2, "shared tail byte names the tail's first level");
}

void TestLocate() {
	Check(Locate(kOffsets, kSizes, 4, 1, 0x1500).mip == 0, "mip 0 start");
	Check(Locate(kOffsets, kSizes, 4, 1, 0x54ff).mip == 0, "mip 0 end");
	Check(Locate(kOffsets, kSizes, 4, 1, 0x4ff).mip == 2, "mip 2 end");
	const auto layered = Locate(kOffsets, kSizes, 4, 2, 0x1500 + 0x2000);
	Check(layered.mip == 0 && layered.layer == 1, "second layer of mip 0");
	Check(Locate(kOffsets, kSizes, 4, 2, 0x1500 + 0x1fff).layer == 0, "last byte of the first layer");
}

} // namespace

int main() {
	TestDirtyBelow();
	TestTailSharedBlock();
	TestLocate();
	std::printf("RefreshProfTests: all cases passed\n");
	return 0;
}
