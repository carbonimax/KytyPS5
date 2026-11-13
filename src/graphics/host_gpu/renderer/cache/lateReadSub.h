#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_LATE_READ_SUB_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_LATE_READ_SUB_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// KYTY_LATE_READ_SUB (measurement): 256-byte block maps of a watched guest range. Bit i covers bytes
// [i * Block, (i + 1) * Block) of the range.
namespace Libs::Graphics::LateReadSub {

constexpr uint64_t Block = 256;

using Bits = std::vector<uint64_t>;

inline void Reset(Bits& bits, uint64_t size) {
	bits.assign((size + Block * 64 - 1) / (Block * 64), 0);
}

// Marks the blocks touching the byte offsets [begin, end); `end` must not pass the range size (the map
// only clamps to whole 64-block words).
inline void Mark(Bits& bits, uint64_t begin, uint64_t end) {
	const uint64_t blocks = bits.size() * 64;
	const uint64_t last   = std::min((end + Block - 1) / Block, blocks);
	for (uint64_t b = begin / Block; b < last; b++) {
		bits[b / 64] |= 1ull << (b % 64);
	}
}

// Marks the blocks of `size` bytes where `now` differs from `before`.
inline void Diff(const uint8_t* before, const uint8_t* now, uint64_t size, Bits& bits) {
	for (uint64_t off = 0; off < size; off += Block) {
		const auto n = std::min(Block, size - off);
		if (std::memcmp(before + off, now + off, n) != 0) {
			bits[off / Block / 64] |= 1ull << (off / Block % 64);
		}
	}
}

inline uint64_t Count(const Bits& bits) {
	uint64_t n = 0;
	for (const auto w: bits) {
		n += static_cast<uint64_t>(std::popcount(w));
	}
	return n;
}

inline uint64_t CountAnd(const Bits& a, const Bits& b) {
	uint64_t n = 0;
	for (size_t i = 0; i < a.size() && i < b.size(); i++) {
		n += static_cast<uint64_t>(std::popcount(a[i] & b[i]));
	}
	return n;
}

inline void Or(Bits& into, const Bits& from) {
	for (size_t i = 0; i < into.size() && i < from.size(); i++) {
		into[i] |= from[i];
	}
}

// Runs of set blocks as (first block, block count).
inline std::vector<std::pair<uint64_t, uint64_t>> Runs(const Bits& bits) {
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	const uint64_t                             blocks = bits.size() * 64;
	for (uint64_t b = 0; b < blocks;) {
		if ((bits[b / 64] >> (b % 64) & 1u) == 0) {
			b++;
			continue;
		}
		const auto first = b;
		while (b < blocks && (bits[b / 64] >> (b % 64) & 1u) != 0) {
			b++;
		}
		runs.emplace_back(first, b - first);
	}
	return runs;
}

// "0xA+0xS,0xB+0xT" (hex or decimal); malformed or empty items are skipped.
inline std::vector<std::pair<uint64_t, uint64_t>> ParseRanges(const char* text) {
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (const char* p = text; p != nullptr && *p != '\0';) {
		char*          end   = nullptr;
		const uint64_t vaddr = std::strtoull(p, &end, 0);
		uint64_t       size  = 0;
		if (end != p && *end == '+') {
			const char* s = end + 1;
			size          = std::strtoull(s, &end, 0);
			if (end == s) {
				size = 0;
			}
		}
		if (size != 0) {
			ranges.emplace_back(vaddr, size);
		}
		while (*end != '\0' && *end != ',') {
			end++;
		}
		p = *end == ',' ? end + 1 : end;
	}
	return ranges;
}

} // namespace Libs::Graphics::LateReadSub

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_LATE_READ_SUB_H_ */
