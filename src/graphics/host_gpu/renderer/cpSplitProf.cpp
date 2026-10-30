#include "graphics/host_gpu/renderer/cpSplitProf.h"

#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::CpSplitProf {

namespace {

constexpr auto     CauseCount  = static_cast<size_t>(Cause::Count);
constexpr uint32_t BucketCount = 7; // items per segment: 1, 2-3, 4-7, 8-15, 16-63, 64-255, 256+

const char* const g_cause_names[CauseCount] = {"wrm",   "release", "event", "acquire", "write",
                                               "dma",   "copy",    "pred",  "flip",    "other",
                                               "submit", "command", "sync_read", "hazard", "idle",
                                               "dispatch", "indirect"};

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint32_t Bucket(uint32_t items) noexcept {
	if (items <= 1) return 0;
	if (items <= 3) return 1;
	if (items <= 7) return 2;
	if (items <= 15) return 3;
	if (items <= 63) return 4;
	if (items <= 255) return 5;
	return 6;
}

// One drain model: which items it pipelines, the current segment and the report totals.
struct Model {
	bool full = false;

	// Current segment (times relative to its start are not needed: the flow shop only adds).
	uint64_t last_mark   = 0; // front accounting resumes here
	uint64_t f_end       = 0; // simulated end of the front stage
	uint64_t b_end       = 0; // simulated end of the back stage
	uint64_t seg_front   = 0;
	uint64_t seg_back    = 0;
	uint32_t seg_items   = 0;
	bool     open        = false;
	uint64_t item_start  = 0;
	uint64_t item_lookup = 0;
	bool     hazard      = false;

	// Report totals.
	uint64_t segments = 0, items = 0, front_ns = 0, back_ns = 0, saved_ns = 0, minsum_ns = 0;
	std::array<uint64_t, CauseCount>  causes {};
	std::array<uint64_t, BucketCount> seg_hist {};
	std::array<uint64_t, BucketCount> back_hist {};
	std::array<uint64_t, BucketCount> saved_hist {};

	[[nodiscard]] bool Pipelines(ItemKind kind) const noexcept { return full || kind == ItemKind::Draw; }

	void EndSegment(uint64_t at, Cause cause) noexcept {
		const uint64_t tail = at > last_mark ? at - last_mark : 0;
		last_mark           = at;
		if (seg_items == 0) {
			f_end = b_end = seg_front = seg_back = 0;
			return;
		}
		seg_front += tail;
		f_end += tail;
		const uint64_t serial   = seg_front + seg_back;
		const uint64_t pipeline = std::max(f_end, b_end);
		const uint64_t saved    = serial > pipeline ? serial - pipeline : 0;
		const uint32_t bucket   = Bucket(seg_items);
		segments++;
		items += seg_items;
		front_ns += seg_front;
		back_ns += seg_back;
		saved_ns += saved;
		minsum_ns += std::min(seg_front, seg_back);
		causes[static_cast<size_t>(cause)]++;
		seg_hist[bucket]++;
		back_hist[bucket] += seg_back;
		saved_hist[bucket] += saved;
		f_end = b_end = seg_front = seg_back = 0;
		seg_items = 0;
	}

	void AddItem(uint64_t front, uint64_t back) noexcept {
		f_end += front;
		b_end = std::max(b_end, f_end) + back;
		seg_front += front;
		seg_back += back;
		seg_items++;
	}

	void Drain(uint64_t now, Cause cause) noexcept {
		// A drain inside an open item happens before its back part: cut at the item start.
		EndSegment(open ? std::max(item_start, last_mark) : now, cause);
		if (open) item_start = last_mark;
		hazard = false;
	}
};

struct State {
	std::array<Model, 2> models {};
	bool                 in_item      = false;
	bool                 in_lookup    = false;
	bool                 item_writes  = false;
	bool                 indirect     = false;
	uint64_t             lookup_start = 0;
	// Totals over both models.
	std::array<uint64_t, 3> kind_items {};
	uint64_t                lookup_ns    = 0;
	uint64_t                writer_items = 0;
	uint64_t                raw_drains   = 0;
};

State& S() noexcept {
	static State state = [] {
		State s;
		s.models[1].full = true;
		return s;
	}();
	return state;
}

bool Active() noexcept {
	return Enabled() && PerfStats::IsGpuThread();
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CP_SPLIT_PROF");
		return PerfStats::Enabled() && value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

void Drain(Cause cause) noexcept {
	if (!Active()) return;
	auto& s = S();
	// A back part's own reads and submissions are back work, not drains.
	if (s.in_item && !s.in_lookup &&
	    (cause == Cause::SyncRead || cause == Cause::Submit || cause == Cause::Command)) {
		return;
	}
	const auto now = NowNs();
	s.raw_drains++;
	for (auto& m: s.models) m.Drain(now, cause);
}

void Packet(uint32_t opcode, uint32_t custom) noexcept {
	if (!Active()) return;
	Cause cause = Cause::Count;
	switch (opcode) {
		case Pm4::IT_NOP:
			switch (custom) {
				case Pm4::R_FLIP:
				case Pm4::R_WAIT_FLIP_DONE: cause = Cause::Flip; break;
				case Pm4::R_WRITE_DATA: cause = Cause::WriteData; break;
				case Pm4::R_DMA_DATA: cause = Cause::DmaData; break;
				default: break;
			}
			break;
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_GET_LOD_STATS:
		// Items: the draw and dispatch handlers decide per model.
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: break;
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64: cause = Cause::WaitRegMem; break;
		case Pm4::IT_RELEASE_MEM: cause = Cause::ReleaseMem; break;
		case Pm4::IT_EVENT_WRITE: cause = Cause::EventWrite; break;
		case Pm4::IT_ACQUIRE_MEM: cause = Cause::AcquireMem; break;
		case Pm4::IT_WRITE_DATA: cause = Cause::WriteData; break;
		case Pm4::IT_DMA_DATA: cause = Cause::DmaData; break;
		case Pm4::IT_COPY_DATA: cause = Cause::CopyData; break;
		case Pm4::IT_COND_EXEC:
		case Pm4::IT_SET_PREDICATION: cause = Cause::Predication; break;
		default: cause = Cause::OtherPacket; break;
	}
	if (cause != Cause::Count) Drain(cause);
}

void ItemBegin(ItemKind kind) noexcept {
	if (!Active()) return;
	auto& s = S();
	if (s.in_item) return; // not nested
	if (s.indirect && kind == ItemKind::Draw) kind = ItemKind::Indirect;
	const auto now = NowNs();
	s.in_item      = true;
	s.item_writes  = false;
	s.kind_items[static_cast<size_t>(kind)]++;
	for (auto& m: s.models) {
		if (m.Pipelines(kind)) {
			m.open        = true;
			m.item_start  = now;
			m.item_lookup = 0;
		} else {
			m.Drain(now, kind == ItemKind::Dispatch ? Cause::Dispatch : Cause::Indirect);
		}
	}
}

void ItemEnd() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_item) return;
	const auto now = NowNs();
	s.in_item      = false;
	if (s.item_writes) s.writer_items++;
	for (auto& m: s.models) {
		if (!m.open) continue;
		const uint64_t lead  = m.item_start > m.last_mark ? m.item_start - m.last_mark : 0;
		const uint64_t whole = now > m.item_start ? now - m.item_start : 0;
		const uint64_t look  = std::min(m.item_lookup, whole);
		m.AddItem(lead + look, whole - look);
		m.last_mark = now;
		m.open      = false;
		m.hazard |= s.item_writes;
	}
}

void NoteItemWrites() noexcept {
	if (!Active()) return;
	S().item_writes = true;
}

void LookupBegin() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_item || s.in_lookup) return;
	s.in_lookup    = true;
	const auto now = NowNs();
	for (auto& m: s.models) {
		if (m.open && m.hazard) m.Drain(now, Cause::Hazard);
	}
	s.lookup_start = NowNs();
}

void LookupEnd() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_lookup) return;
	s.in_lookup        = false;
	const auto elapsed = NowNs() - s.lookup_start;
	s.lookup_ns += elapsed;
	for (auto& m: s.models) {
		if (m.open) m.item_lookup += elapsed;
	}
}

void SetIndirect(bool on) noexcept {
	if (!Active()) return;
	S().indirect = on;
}

void BusyBegin() noexcept {
	if (!Active()) return;
	const auto now = NowNs();
	for (auto& m: S().models) {
		m.EndSegment(now, Cause::Idle);
		m.hazard = false;
	}
}

void BusyEnd() noexcept {
	Drain(Cause::Idle);
}

void NoteSubmitOrWait() noexcept {
	Drain(Cause::Submit);
}

void Report(double seconds) noexcept {
	if (!Enabled()) return;
	auto&      s  = S();
	const auto ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	std::printf("PERF cpsplit %.1fs: draws=%" PRIu64 " indirect=%" PRIu64 " dispatches=%" PRIu64
	            " writers=%" PRIu64 " lookups=%.0fms raw_drains=%" PRIu64 "\n",
	            seconds, s.kind_items[0], s.kind_items[1], s.kind_items[2], s.writer_items,
	            ms(s.lookup_ns), s.raw_drains);
	s.kind_items   = {};
	s.lookup_ns    = 0;
	s.writer_items = 0;
	s.raw_drains   = 0;
	for (auto& m: s.models) {
		std::printf("PERF cpsplit %s: segs=%" PRIu64 " items=%" PRIu64
		            " front=%.0fms back=%.0fms saved=%.0fms minsum=%.0fms",
		            m.full ? "full" : "s21", m.segments, m.items, ms(m.front_ns), ms(m.back_ns),
		            ms(m.saved_ns), ms(m.minsum_ns));
		std::printf(" hist=");
		for (uint32_t b = 0; b < BucketCount; b++) {
			std::printf("%s%" PRIu64, b ? "/" : "", m.seg_hist[b]);
		}
		std::printf(" back_by_len=");
		for (uint32_t b = 0; b < BucketCount; b++) {
			std::printf("%s%.0f", b ? "/" : "", ms(m.back_hist[b]));
		}
		std::printf(" saved_by_len=");
		for (uint32_t b = 0; b < BucketCount; b++) {
			std::printf("%s%.0f", b ? "/" : "", ms(m.saved_hist[b]));
		}
		std::printf(" causes:");
		for (size_t c = 0; c < CauseCount; c++) {
			if (m.causes[c] != 0) std::printf(" %s=%" PRIu64, g_cause_names[c], m.causes[c]);
		}
		std::printf("\n");
		m.segments = m.items = m.front_ns = m.back_ns = m.saved_ns = m.minsum_ns = 0;
		m.causes                                                                  = {};
		m.seg_hist                                                                = {};
		m.back_hist                                                               = {};
		m.saved_hist                                                              = {};
	}
}

} // namespace Libs::Graphics::CpSplitProf
