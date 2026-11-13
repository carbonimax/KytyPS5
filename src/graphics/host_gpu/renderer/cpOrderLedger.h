#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPORDERLEDGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPORDERLEDGER_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics {

// KYTY_CP_ORDER_CHECK (measurement): pairs the guest bytes that pipelined program lookups read
// without a drain with the bytes queued draws write afterwards. Draws are numbered as the CP
// pipeline publishes them. A hazard: draw `writer` writes bytes that the lookup of a later draw
// `reader` (> writer) read while `writer` was still queued, so the lookup saw the bytes from before
// that write. Not thread-safe: the caller serializes.
class CpOrderLedger {
public:
	struct Hit {
		uint64_t address = 0; // first overlapping byte
		uint64_t size    = 0; // overlap with that read
		uint64_t reader  = 0;
		uint64_t writer  = 0;
		uint32_t tag     = 0; // the write's, as given to Write()
	};

	// Writes kept to pair with reads inserted after them (ring).
	static constexpr uint32_t RecentWrites = 256;

	// Front: draw `reader` read [begin, end). `finished`: records the back has finished (reads of a
	// reader <= finished can no longer be hazards). `since`: Serial() when the lookup began; writes
	// noted after it, by a draw before `reader`, are reported through `window` (the read may have
	// happened before or after them: the walker loads before the read is inserted). One window hit per
	// write and reader.
	template <typename Window>
	void AddRead(uint64_t begin, uint64_t end, uint64_t reader, uint64_t finished, uint64_t since,
	             Window&& window) {
		if (end <= begin || reader <= finished) return;
		for (uint64_t serial = std::max(since, m_serial > RecentWrites ? m_serial - RecentWrites : 0);
		     serial < m_serial; serial++) {
			auto& write = m_recent[serial % RecentWrites];
			if (write.writer < reader && write.last_window_reader != reader + 1 && write.begin < end &&
			    begin < write.end) {
				write.last_window_reader = reader + 1;
				const uint64_t from = std::max(begin, write.begin);
				window(Hit {from, std::min(end, write.end) - from, reader, write.writer, write.tag});
			}
		}
		m_reads.push_back({begin, end, reader});
		ForGranules(begin, end, [&](uint64_t granule) { m_granules[granule]++; });
	}

	// Back: draw `writer` (the record it runs) writes [begin, end). Calls hit() once per later reader
	// whose reads overlap (with the first overlapping read); returns the number of hits. `tag`: the
	// caller's kind of write, returned in its hits.
	template <typename HitFn>
	uint32_t Write(uint64_t begin, uint64_t end, uint64_t writer, uint32_t tag, HitFn&& hit) {
		if (end <= begin) return 0;
		Prune(writer);
		m_recent[m_serial % RecentWrites] = {begin, end, writer, 0, tag};
		m_serial++;
		bool any = false;
		ForGranules(begin, end, [&](uint64_t granule) { any = any || m_granules.contains(granule); });
		if (!any) return 0;
		uint32_t hits        = 0;
		uint64_t last_reader = UINT64_MAX;
		for (const auto& read: m_reads) {
			if (read.reader <= writer || read.reader == last_reader || read.end <= begin || end <= read.begin) {
				continue;
			}
			last_reader       = read.reader;
			const uint64_t from = std::max(begin, read.begin);
			hit(Hit {from, std::min(end, read.end) - from, read.reader, writer, tag});
			hits++;
		}
		return hits;
	}

	// Drops the reads of readers <= finished (every draw before them is done).
	void Prune(uint64_t finished) {
		while (!m_reads.empty() && m_reads.front().reader <= finished) {
			const auto read = m_reads.front();
			m_reads.pop_front();
			ForGranules(read.begin, read.end, [&](uint64_t granule) {
				if (auto found = m_granules.find(granule); found != m_granules.end() && --found->second == 0) {
					m_granules.erase(found);
				}
			});
		}
	}

	[[nodiscard]] uint64_t Serial() const noexcept { return m_serial; }
	[[nodiscard]] size_t   Reads() const noexcept { return m_reads.size(); }

private:
	static constexpr uint32_t GranuleBits = 16;

	struct Read {
		uint64_t begin, end, reader;
	};
	struct RecentWrite {
		uint64_t begin = 0, end = 0, writer = 0, last_window_reader = 0;
		uint32_t tag = 0;
	};

	template <typename F>
	static void ForGranules(uint64_t begin, uint64_t end, F&& f) {
		for (uint64_t granule = begin >> GranuleBits; granule <= (end - 1) >> GranuleBits; granule++) f(granule);
	}

	// Readers never decrease (the front numbers lookups with the next record to publish).
	std::deque<Read>                       m_reads;
	std::unordered_map<uint64_t, uint32_t> m_granules;
	std::array<RecentWrite, RecentWrites>  m_recent {};
	uint64_t                               m_serial = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPORDERLEDGER_H_
