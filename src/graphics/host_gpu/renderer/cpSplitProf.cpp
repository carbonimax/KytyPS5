#include "graphics/host_gpu/renderer/cpSplitProf.h"

#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace Libs::Graphics::CpSplitProf {

namespace {

constexpr auto     CauseCount  = static_cast<size_t>(Cause::Count);
constexpr uint32_t BucketCount = 7; // items per segment: 1, 2-3, 4-7, 8-15, 16-63, 64-255, 256+

const char* const g_cause_names[CauseCount] = {
    "wrm",      "release",   "event",      "acquire",  "write",      "dma",
    "copy",     "pred",      "flip",       "other",    "submit",     "command",
    "sync_read", "hazard",   "idle",       "dispatch", "indirect",   "inline",
    "fallback", "fault",     "hazard_srt", "hazard_read", "hazard_pm4", "hazard_fault",
    "hazard_wait"};

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

// Mirrors IsPipelinedPacket (graphicsRun.cpp): packets KYTY_CP_PIPELINE runs without draining.
bool RealPipelinedPacket(uint32_t opcode, uint32_t custom) noexcept {
	switch (opcode) {
		case Pm4::IT_NOP: return custom == Pm4::R_ZERO || custom == Pm4::R_CONTEXT_STATE;
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
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return true;
		// Items: their handlers decide per model.
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

// Disjoint, sorted, merged [begin, end) ranges.
class RangeSet {
public:
	[[nodiscard]] bool   Empty() const noexcept { return m_ranges.empty(); }
	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }
	void                 Clear() noexcept { m_ranges.clear(); }
	void                 Add(uint64_t begin, uint64_t end) {
        if (begin >= end) return;
        // First range ending at or after begin (adjacent ranges merge).
        auto first = std::lower_bound(m_ranges.begin(), m_ranges.end(), begin,
		                                                  [](const auto& r, uint64_t v) { return r.second < v; });
        auto last  = first;
        while (last != m_ranges.end() && last->first <= end) {
            begin = std::min(begin, last->first);
            end   = std::max(end, last->second);
            ++last;
        }
        first = m_ranges.erase(first, last);
        m_ranges.insert(first, {begin, end});
	}
	[[nodiscard]] bool Overlaps(uint64_t begin, uint64_t end) const noexcept {
		if (m_ranges.empty() || begin >= end) return false;
		// First range ending after begin.
		const auto it = std::upper_bound(m_ranges.begin(), m_ranges.end(), begin,
		                                 [](uint64_t v, const auto& r) { return v < r.second; });
		return it != m_ranges.end() && it->first < end;
	}

private:
	std::vector<std::pair<uint64_t, uint64_t>> m_ranges;
};

enum class Kind : uint32_t { S21, Full, S21R, S24D, S24R, S24W, S24C, S24X, Count };
constexpr auto    KindCount            = static_cast<size_t>(Kind::Count);
const char* const g_kind_names[KindCount] = {"s21", "full", "s21r", "s24d", "s24r", "s24w", "s24c", "s24x"};

enum class Rule : uint32_t { Coarse, Fine, None };
enum class Mode : uint32_t { Queue, Inline, Fallback };

struct Read {
	uint64_t   address;
	uint64_t   size;
	ReadSource source;
};

Cause HazardCause(ReadSource source) noexcept {
	switch (source) {
		case ReadSource::Srt: return Cause::HazardSrt;
		case ReadSource::Backing: return Cause::HazardRead;
		default: return Cause::HazardPm4;
	}
}

// One drain model: which items it pipelines, the current segment and the report totals.
struct Model {
	Kind kind = Kind::S21;

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
	bool     hazard      = false; // coarse: a queued item of the segment writes memory
	RangeSet writes;              // fine: guest ranges written by queued items of the segment
	RangeSet labels;              // labels written by queued RELEASE_MEM/EVENT_WRITE of the segment

	// Report totals.
	uint64_t segments = 0, items = 0, front_ns = 0, back_ns = 0, saved_ns = 0, minsum_ns = 0;
	uint64_t inline_items = 0, fallback_items = 0, queued_writers = 0, queued_syncs = 0;
	uint64_t waits_label = 0, waits_memory = 0; // memory waits resolved on the front
	std::array<uint64_t, CauseCount>  causes {};
	std::array<uint64_t, BucketCount> seg_hist {};
	std::array<uint64_t, BucketCount> back_hist {};
	std::array<uint64_t, BucketCount> saved_hist {};

	[[nodiscard]] bool Real() const noexcept { return kind >= Kind::S21R; }
	[[nodiscard]] bool QueuesSync() const noexcept { return kind >= Kind::S24R; }
	[[nodiscard]] Rule HazardRule() const noexcept {
		switch (kind) {
			case Kind::S24D:
			case Kind::S24R:
			case Kind::S24W: return Rule::Fine;
			case Kind::S24X: return Rule::None;
			default: return Rule::Coarse;
		}
	}
	[[nodiscard]] bool Fine() const noexcept { return HazardRule() == Rule::Fine; }
	[[nodiscard]] bool Pipelines(ItemKind item) const noexcept {
		switch (item) {
			case ItemKind::Draw: return true;
			case ItemKind::Indirect: return kind == Kind::Full;
			case ItemKind::Dispatch: return kind == Kind::Full || kind >= Kind::S24D;
			case ItemKind::Sync: return QueuesSync();
		}
		return false;
	}
	// Packet drains this model ignores (s24r and later queue the synchronization packets; memory
	// waits are decided by WaitMemory once the address and the value are known).
	[[nodiscard]] bool Ignores(Cause cause) const noexcept {
		return QueuesSync() && (cause == Cause::ReleaseMem || cause == Cause::EventWrite ||
		                        cause == Cause::AcquireMem || cause == Cause::WaitRegMem);
	}

	void EndSegment(uint64_t at, Cause cause) noexcept {
		const uint64_t tail = at > last_mark ? at - last_mark : 0;
		last_mark           = at;
		writes.Clear();
		labels.Clear();
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
	std::array<Model, KindCount> models {};
	bool                         in_item       = false;
	bool                         in_lookup     = false;
	bool                         item_writes   = false;
	bool                         item_inline   = false;
	bool                         item_fallback = false;
	Cause                        fallback_cause = Cause::Fallback;
	bool                         indirect      = false;
	ItemKind                     item_kind     = ItemKind::Draw;
	std::vector<std::pair<uint64_t, uint64_t>> item_ranges; // [begin, end) written by the item
	std::vector<std::pair<uint64_t, uint64_t>> item_labels;
	std::vector<Read>                          pending_reads; // front reads of the current lookup
	uint64_t                                   lookup_start  = 0;
	uint64_t                                   lookup_end    = 0;
	uint64_t                                   item_overhead = 0; // hazard checks inside the item
	// Totals over all models.
	std::array<uint64_t, 4> kind_items {};
	uint64_t                lookup_ns    = 0;
	uint64_t                writer_items = 0;
	uint64_t                raw_drains   = 0;
	uint64_t                write_ranges = 0;
	uint64_t                reads_checked = 0;
	uint64_t                check_ns     = 0;
};

State& S() noexcept {
	static State state = [] {
		State s;
		for (size_t i = 0; i < s.models.size(); i++) s.models[i].kind = static_cast<Kind>(i);
		s.pending_reads.reserve(4096);
		return s;
	}();
	return state;
}

bool Active() noexcept {
	return Enabled() && PerfStats::IsGpuThread();
}

bool AnyFineWrites(const State& s) noexcept {
	return std::any_of(s.models.begin(), s.models.end(),
	                   [](const Model& m) { return m.Fine() && !m.writes.Empty(); });
}

// Fine models: drain when a read overlaps a range written by a queued item.
void CheckRead(State& s, uint64_t now, uint64_t address, uint64_t size, Cause cause) noexcept {
	s.reads_checked++;
	for (auto& m: s.models) {
		if (m.Fine() && m.writes.Overlaps(address, address + size)) m.Drain(now, cause);
	}
}

Mode Decide(const State& s, const Model& m) noexcept {
	if (!m.Real()) return Mode::Queue;
	if (s.item_fallback) return Mode::Fallback;
	switch (s.item_kind) {
		case ItemKind::Draw: return s.item_inline || s.item_writes ? Mode::Inline : Mode::Queue;
		case ItemKind::Dispatch: return s.item_inline ? Mode::Inline : Mode::Queue;
		default: return Mode::Queue;
	}
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CP_SPLIT_PROF");
		const bool  on    = PerfStats::Enabled() && value != nullptr && std::atoi(value) != 0;
		if (on) ShaderRecompiler::IR::SetWalkLoadsHook(&FrontReadLoads);
		return on;
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
	for (auto& m: s.models) {
		if (!m.Ignores(cause)) m.Drain(now, cause);
	}
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
	// The real pipeline drains on every packet it does not run while draws are queued.
	Cause real = cause;
	if (RealPipelinedPacket(opcode, custom)) {
		real = Cause::Count;
	} else if (real == Cause::Count) {
		real = Cause::OtherPacket;
	}
	if (cause == Cause::Count && real == Cause::Count) return;
	auto&      s   = S();
	const auto now = NowNs();
	s.raw_drains++;
	for (auto& m: s.models) {
		const Cause c = m.Real() ? real : cause;
		if (c != Cause::Count && !m.Ignores(c)) m.Drain(now, c);
	}
}

void ItemBegin(ItemKind kind) noexcept {
	if (!Active()) return;
	auto& s = S();
	if (s.in_item) return; // not nested
	if (s.indirect && kind == ItemKind::Draw) kind = ItemKind::Indirect;
	const auto now  = NowNs();
	s.in_item       = true;
	s.item_kind     = kind;
	s.item_writes   = false;
	s.item_inline   = false;
	s.item_fallback = false;
	s.fallback_cause = Cause::Fallback;
	s.item_ranges.clear();
	s.item_labels.clear();
	s.pending_reads.clear();
	s.lookup_end    = 0;
	s.item_overhead = 0;
	s.kind_items[static_cast<size_t>(kind)]++;
	for (auto& m: s.models) {
		if (m.Pipelines(kind)) {
			m.open        = true;
			m.item_start  = now;
			m.item_lookup = 0;
		} else {
			m.Drain(now, kind == ItemKind::Dispatch ? Cause::Dispatch
			             : kind == ItemKind::Sync   ? Cause::ReleaseMem
			                                        : Cause::Indirect);
		}
	}
}

void ItemEnd() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_item) return;
	const auto now = NowNs();
	s.in_item      = false;
	s.in_lookup    = false;
	if (s.item_writes) s.writer_items++;
	for (auto& m: s.models) {
		if (!m.open) continue;
		m.open          = false;
		const auto mode = Decide(s, m);
		if (mode == Mode::Fallback) {
			// Drains before the item, which then runs serially on the front.
			m.EndSegment(std::max(m.item_start, m.last_mark), s.fallback_cause);
			m.hazard    = false;
			m.last_mark = now;
			m.fallback_items++;
			continue;
		}
		const uint64_t lead = m.item_start > m.last_mark ? m.item_start - m.last_mark : 0;
		if (mode == Mode::Inline) {
			// Front part up to the end of the lookup, then a drain, then the rest runs serially.
			const uint64_t cut = std::clamp(s.lookup_end, m.item_start, now);
			m.AddItem(lead + (cut - m.item_start), 0);
			m.last_mark = cut;
			m.EndSegment(cut, Cause::Inline);
			m.hazard    = false;
			m.last_mark = now;
			m.inline_items++;
			continue;
		}
		uint64_t whole = now > m.item_start ? now - m.item_start : 0;
		whole -= std::min(whole, s.item_overhead);
		const uint64_t look = std::min(m.item_lookup, whole);
		m.AddItem(lead + look, whole - look);
		m.last_mark = now;
		if (s.item_writes) {
			m.hazard = true;
			m.queued_writers++;
		}
		if (m.Fine()) {
			for (const auto& [b, e]: s.item_ranges) m.writes.Add(b, e);
		}
		if (s.item_kind == ItemKind::Sync) {
			m.queued_syncs++;
			for (const auto& [b, e]: s.item_labels) m.labels.Add(b, e);
		}
	}
}

void NoteItemWrites() noexcept {
	if (!Active()) return;
	S().item_writes = true;
}

void NoteItemInline() noexcept {
	if (!Active()) return;
	S().item_inline = true;
}

void NoteItemFallback(Cause cause) noexcept {
	if (!Active()) return;
	auto& s          = S();
	s.item_fallback  = true;
	s.fallback_cause = cause;
}

void NoteItemWriteRange(uint64_t address, uint64_t size) noexcept {
	if (!Active() || size == 0) return;
	auto& s = S();
	if (!s.in_item) return;
	s.item_ranges.emplace_back(address, address + size);
	s.write_ranges++;
}

void NoteLabel(uint64_t address, uint64_t size) noexcept {
	if (!Active() || size == 0) return;
	auto& s = S();
	if (!s.in_item) return;
	s.item_labels.emplace_back(address, address + size);
}

void WaitMemory(uint64_t address, uint64_t size, bool satisfied) noexcept {
	if (!Active()) return;
	auto&      s   = S();
	const auto now = NowNs();
	const auto end = address + size;
	for (auto& m: s.models) {
		if (!m.QueuesSync()) continue; // the packet already drained
		Cause cause = Cause::Count;
		if (m.kind == Kind::S24R) {
			cause = Cause::WaitRegMem;
		} else if (m.Fine() && m.writes.Overlaps(address, end)) {
			cause = Cause::HazardWait;
		} else if (m.HazardRule() == Rule::Coarse && m.hazard) {
			cause = Cause::Hazard;
		} else if (!satisfied) {
			cause = Cause::WaitRegMem;
		}
		if (cause != Cause::Count) {
			m.Drain(now, cause);
		} else if (m.labels.Overlaps(address, end)) {
			m.waits_label++;
		} else {
			m.waits_memory++;
		}
	}
}

void FrontRead(uint64_t address, uint64_t size, ReadSource source) noexcept {
	if (!Active()) return;
	auto& s = S();
	if (s.in_item && !s.in_lookup) return; // back work
	if (!AnyFineWrites(s)) return;
	if (s.in_lookup) {
		s.pending_reads.push_back({address, size, source});
		return;
	}
	const auto start = NowNs();
	CheckRead(s, start, address, size, HazardCause(source));
	s.check_ns += NowNs() - start;
}

void FrontReadLoads(const uint64_t* addresses, size_t count, size_t stride) noexcept {
	if (!Active() || count == 0) return;
	auto& s = S();
	if (s.in_item && !s.in_lookup) return; // back work
	if (!AnyFineWrites(s)) return;
	if (s.in_lookup) {
		for (size_t i = 0; i < count; i++) {
			s.pending_reads.push_back({addresses[i * stride], sizeof(uint32_t), ReadSource::Srt});
		}
		return;
	}
	const auto start = NowNs();
	for (size_t i = 0; i < count; i++) {
		CheckRead(s, start, addresses[i * stride], sizeof(uint32_t), Cause::HazardSrt);
	}
	s.check_ns += NowNs() - start;
}

void SyncRead(uint64_t address, uint64_t size) noexcept {
	if (!Active()) return;
	auto& s = S();
	if (s.in_item && !s.in_lookup) return; // the back part's own read
	const auto now = NowNs();
	s.raw_drains++;
	for (auto& m: s.models) {
		m.Drain(now, m.Fine() && m.writes.Overlaps(address, address + size) ? Cause::HazardRead
		                                                                     : Cause::SyncRead);
	}
}

void Fault(uint64_t address) noexcept {
	if (!Active()) return;
	auto& s = S();
	if (s.in_item && !s.in_lookup) return; // the back part's own fault
	const auto now = NowNs();
	s.raw_drains++;
	for (auto& m: s.models) {
		if (!m.Real()) continue;
		m.Drain(now, m.Fine() && m.writes.Overlaps(address, address + 1) ? Cause::HazardFault
		                                                                  : Cause::Fault);
	}
}

void LookupBegin() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_item || s.in_lookup) return;
	s.in_lookup    = true;
	const auto now = NowNs();
	for (auto& m: s.models) {
		if (m.open && m.hazard && m.HazardRule() == Rule::Coarse) m.Drain(now, Cause::Hazard);
	}
	s.lookup_start = NowNs();
}

void LookupEnd() noexcept {
	if (!Active()) return;
	auto& s = S();
	if (!s.in_lookup) return;
	s.in_lookup        = false;
	const auto end     = NowNs();
	const auto elapsed = end - s.lookup_start;
	s.lookup_end       = end;
	s.lookup_ns += elapsed;
	for (auto& m: s.models) {
		if (m.open) m.item_lookup += elapsed;
	}
	if (s.pending_reads.empty()) return;
	// The reads of the lookup, checked now so the check is not front time; it is taken out of the
	// item's back part too. A hazard drains before the item (Model::Drain cuts at its start).
	for (const auto& read: s.pending_reads) {
		CheckRead(s, end, read.address, read.size, HazardCause(read.source));
		if (!AnyFineWrites(s)) break;
	}
	s.pending_reads.clear();
	const auto checked = NowNs() - end;
	s.item_overhead += checked;
	s.check_ns += checked;
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
	            " syncs=%" PRIu64 " writers=%" PRIu64 " lookups=%.0fms raw_drains=%" PRIu64
	            " write_ranges=%" PRIu64 " reads_checked=%" PRIu64 " check=%.0fms\n",
	            seconds, s.kind_items[0], s.kind_items[1], s.kind_items[2], s.kind_items[3],
	            s.writer_items, ms(s.lookup_ns), s.raw_drains, s.write_ranges, s.reads_checked,
	            ms(s.check_ns));
	s.kind_items    = {};
	s.lookup_ns     = 0;
	s.writer_items  = 0;
	s.raw_drains    = 0;
	s.write_ranges  = 0;
	s.reads_checked = 0;
	s.check_ns      = 0;
	for (auto& m: s.models) {
		std::printf("PERF cpsplit %s: segs=%" PRIu64 " items=%" PRIu64
		            " front=%.0fms back=%.0fms saved=%.0fms minsum=%.0fms inline=%" PRIu64
		            " fallback=%" PRIu64 " qwriters=%" PRIu64 " qsyncs=%" PRIu64 " wlabel=%" PRIu64
		            " wmem=%" PRIu64,
		            g_kind_names[static_cast<size_t>(m.kind)], m.segments, m.items, ms(m.front_ns),
		            ms(m.back_ns), ms(m.saved_ns), ms(m.minsum_ns), m.inline_items, m.fallback_items,
		            m.queued_writers, m.queued_syncs, m.waits_label, m.waits_memory);
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
		m.inline_items = m.fallback_items = m.queued_writers = m.queued_syncs = 0;
		m.waits_label = m.waits_memory = 0;
		m.causes                       = {};
		m.seg_hist                     = {};
		m.back_hist                    = {};
		m.saved_hist                   = {};
	}
}

} // namespace Libs::Graphics::CpSplitProf
