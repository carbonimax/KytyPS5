#include "graphics/host_gpu/renderer/cache/refreshProf.h"

#include "graphics/host_gpu/renderer/image/image.h"
#include "kernel/memory.h"

#include <xxhash.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::RefreshProf {

namespace {

constexpr const char* ReasonNames[ReasonCount] = {"new",   "new_gpu",  "cpu_fault", "host",    "label",  "gpu_fill",
                                                  "gpu_copy", "gpu_storage", "img_copy", "bc_alias", "edge"};

struct Bucket {
	uint64_t n     = 0;
	uint64_t bytes = 0;
	void     Add(uint64_t b) noexcept {
		n++;
		bytes += b;
	}
};

struct ImageEntry {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint32_t width = 0, height = 0, levels = 0, layers = 0, format = 0;
	Bucket   all, same, diff, gpu, skipped;
	uint64_t marks       = 0;
	uint32_t reason_mask = 0;
	uint64_t last_serial = 0;
	uint32_t serials     = 0;
	uint64_t span_lo = UINT64_MAX, span_hi = 0;
	MipLayer lo_at, hi_at;
	uint32_t view_levels = 0, view_layers = 0, view_base_level = 0, view_base_layer = 0;
	uint32_t partial_views = 0;
};

struct State {
	std::mutex                                 mutex;
	std::chrono::steady_clock::time_point      last = std::chrono::steady_clock::now();
	Bucket                                     all, fresh, refresh, gpu_src, same, diff, unknown, skipped;
	std::array<Bucket, ReasonCount>            by_reason {};
	std::array<uint64_t, ReasonCount>          marks {};
	uint64_t                                   record_ns = 0;
	uint64_t                                   span_bytes = 0;
	uint64_t                                   partial_view = 0;
	uint64_t                                   mismatches   = 0;
	uint64_t                                   mismatches_total = 0;
	std::unordered_map<uint64_t, ImageEntry>   images;
};

State& GetState() {
	static auto* state = new State;
	return *state;
}

struct View {
	uint32_t base_level = 0, levels = 0, base_layer = 0, layers = 0;
	bool     valid      = false;
};
thread_local View t_view;

uint32_t EnvU32(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	return value == nullptr ? fallback : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

void Report(State& s, double seconds) {
	const auto mb = [](uint64_t b) { return static_cast<double>(b) / 1048576.0; };
	std::printf("REFRESH prof %.1fs: uploads=%" PRIu64 "/%.1fMB new=%" PRIu64 "/%.1fMB refresh=%" PRIu64
	            "/%.1fMB gpu_src=%" PRIu64 "/%.1fMB same=%" PRIu64 "/%.1fMB diff=%" PRIu64 "/%.1fMB unknown=%" PRIu64
	            "/%.1fMB skipped=%" PRIu64 "/%.1fMB record=%.1fms span=%.1fMB partial_view=%" PRIu64 " mismatches=%" PRIu64
	            " (total %" PRIu64 ")\n",
	            seconds, s.all.n, mb(s.all.bytes), s.fresh.n, mb(s.fresh.bytes), s.refresh.n, mb(s.refresh.bytes),
	            s.gpu_src.n, mb(s.gpu_src.bytes), s.same.n, mb(s.same.bytes), s.diff.n, mb(s.diff.bytes), s.unknown.n,
	            mb(s.unknown.bytes), s.skipped.n, mb(s.skipped.bytes), static_cast<double>(s.record_ns) / 1e6,
	            mb(s.span_bytes), s.partial_view, s.mismatches, s.mismatches_total);
	std::printf("REFRESH reasons %.1fs:", seconds);
	for (uint32_t r = 0; r < ReasonCount; r++) {
		std::printf(" %s=%" PRIu64 "/%.1fMB(marks %" PRIu64 ")", ReasonNames[r], s.by_reason[r].n,
		            mb(s.by_reason[r].bytes), s.marks[r]);
	}
	std::printf("\n");
	std::vector<const ImageEntry*> top;
	top.reserve(s.images.size());
	for (const auto& [key, entry]: s.images) {
		top.push_back(&entry);
	}
	std::sort(top.begin(), top.end(), [](const ImageEntry* a, const ImageEntry* b) { return a->all.bytes > b->all.bytes; });
	const size_t count = std::min<size_t>(top.size(), EnvU32("KYTY_REFRESH_PROF_TOP", 8));
	for (size_t i = 0; i < count; i++) {
		const auto& e = *top[i];
		std::printf("REFRESH top%zu addr=0x%" PRIx64 " size=0x%" PRIx64 " fmt=%u %ux%u m%u l%u n=%" PRIu64
		            " MB=%.1f same=%" PRIu64 " diff=%" PRIu64 " gpu=%" PRIu64 " skipped=%" PRIu64 " marks=%" PRIu64
		            " serials=%u reasons=",
		            i, e.address, e.size, e.format, e.width, e.height, e.levels, e.layers, e.all.n, mb(e.all.bytes),
		            e.same.n, e.diff.n, e.gpu.n, e.skipped.n, e.marks, e.serials);
		bool first = true;
		for (uint32_t r = 0; r < ReasonCount; r++) {
			if ((e.reason_mask & (1u << r)) != 0) {
				std::printf("%s%s", first ? "" : ",", ReasonNames[r]);
				first = false;
			}
		}
		if (e.span_lo <= e.span_hi) {
			std::printf(" span=0x%" PRIx64 "..0x%" PRIx64 " mip%u/l%u..mip%u/l%u", e.span_lo, e.span_hi, e.lo_at.mip,
			            e.lo_at.layer, e.hi_at.mip, e.hi_at.layer);
		}
		std::printf(" view=m%u+%u/l%u+%u partial_views=%u\n", e.view_base_level, e.view_levels, e.view_base_layer,
		            e.view_layers, e.partial_views);
	}
	std::fflush(stdout);
	s.all = s.fresh = s.refresh = s.gpu_src = s.same = s.diff = s.unknown = s.skipped = {};
	s.by_reason  = {};
	s.marks      = {};
	s.record_ns  = 0;
	s.span_bytes = 0;
	s.partial_view = 0;
	s.mismatches   = 0;
	s.images.clear();
}

void MaybeReport(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last >= std::chrono::seconds(5)) {
		Report(s, std::chrono::duration<double>(now - s.last).count());
		s.last = now;
	}
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = EnvU32("KYTY_REFRESH_PROF", 0) != 0;
	return enabled;
}

uint32_t SkipSameMode() noexcept {
	static const uint32_t mode = std::min(EnvU32("KYTY_REFRESH_SKIP_SAME", 0), 2u);
	return mode;
}

void NoteDirty(Image& image, uint8_t reason, uint64_t address, uint64_t size) noexcept {
	auto& note = image.refresh_note;
	note.reason = reason;
	note.reasons |= 1u << reason;
	note.marks++;
	const auto begin = image.info.data.address;
	const auto end   = image.info.data.End();
	const auto lo    = std::max(address, begin);
	const auto hi    = std::min(address + size, end);
	if (lo < hi) {
		note.lo = std::min(note.lo, lo - begin);
		note.hi = std::max(note.hi, hi - begin);
	}
	auto&           s = GetState();
	std::scoped_lock lock {s.mutex};
	s.marks[reason]++;
}

void NoteView(uint32_t base_level, uint32_t levels, uint32_t base_layer, uint32_t layers) noexcept {
	t_view = {base_level, levels, base_layer, layers, true};
}

void ClearView() noexcept {
	t_view.valid = false;
}

void NoteMismatch() noexcept {
	auto&           s = GetState();
	std::scoped_lock lock {s.mutex};
	s.mismatches++;
	s.mismatches_total++;
}

void NoteUpload(Image& image, bool gpu_source, uint64_t record_ns, int same, bool skipped) noexcept {
	auto&       note  = image.refresh_note;
	const auto& info  = image.info;
	const auto  bytes = info.data.size;
	const bool  fresh = note.uploads == 0;
	uint8_t     reason = note.reasons == 0 ? (fresh ? New : note.reason) : note.reason;
	if (fresh && reason != NewGpu) {
		reason = New;
	}
	auto&           s = GetState();
	std::scoped_lock lock {s.mutex};
	s.all.Add(bytes);
	(fresh ? s.fresh : s.refresh).Add(bytes);
	if (skipped) {
		s.skipped.Add(bytes);
	}
	if (gpu_source) {
		s.gpu_src.Add(bytes);
	} else if (!fresh) {
		(same > 0 ? s.same : same == 0 ? s.diff : s.unknown).Add(bytes);
	}
	s.by_reason[reason].Add(bytes);
	s.record_ns += record_ns;
	auto& e = s.images[info.data.address ^ (info.data.size << 20u) ^ static_cast<uint64_t>(info.pixel_format) << 56u];
	if (e.all.n == 0) {
		e.address = info.data.address;
		e.size    = info.data.size;
		e.width   = info.extent.width;
		e.height  = info.extent.height;
		e.levels  = info.resources.levels;
		e.layers  = info.resources.layers;
		e.format  = static_cast<uint32_t>(info.pixel_format);
	}
	e.all.Add(bytes);
	if (skipped) {
		e.skipped.Add(bytes);
	}
	if (gpu_source) {
		e.gpu.Add(bytes);
	} else if (!fresh) {
		(same > 0 ? e.same : e.diff).Add(bytes);
	}
	e.marks += note.marks;
	e.reason_mask |= note.reasons == 0 ? (1u << reason) : note.reasons;
	if (e.last_serial != image.serial) {
		e.last_serial = image.serial;
		e.serials++;
	}
	if (!fresh && note.lo < note.hi) {
		s.span_bytes += note.hi - note.lo;
		std::array<uint64_t, 16> offsets {};
		std::array<uint64_t, 16> sizes {};
		const auto levels = std::min<uint32_t>(info.resources.levels, 16);
		for (uint32_t m = 0; m < levels; m++) {
			offsets[m] = info.mip_layout[m].offset;
			sizes[m]   = info.mip_layout[m].size;
		}
		if (note.lo < e.span_lo) {
			e.span_lo = note.lo;
			e.lo_at   = Locate(offsets.data(), sizes.data(), levels, info.resources.layers, note.lo);
		}
		if (note.hi > e.span_hi || e.span_hi == 0) {
			e.span_hi = note.hi;
			e.hi_at   = Locate(offsets.data(), sizes.data(), levels, info.resources.layers, note.hi - 1);
		}
	}
	if (t_view.valid) {
		e.view_base_level = t_view.base_level;
		e.view_levels     = t_view.levels;
		e.view_base_layer = t_view.base_layer;
		e.view_layers     = t_view.layers;
		if (t_view.levels < info.resources.levels || t_view.layers < info.resources.layers) {
			e.partial_views++;
			s.partial_view++;
		}
	}
	note.uploads++;
	note.reasons = 0;
	note.marks   = 0;
	note.lo      = UINT64_MAX;
	note.hi      = 0;
	MaybeReport(s);
}

uint64_t HashGuest(uint64_t address, uint64_t size) noexcept {
	thread_local std::vector<uint8_t> chunk(1u << 20u);
	thread_local XXH3_state_t*        state = XXH3_createState();
	XXH3_64bits_reset(state);
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min<uint64_t>(size - done, chunk.size());
		if (!LibKernel::Memory::TryReadBacking(address + done, chunk.data(), part)) {
			return 0;
		}
		XXH3_64bits_update(state, chunk.data(), static_cast<size_t>(part));
		done += part;
	}
	return XXH3_64bits_digest(state) | 1u; // 0 means "unreadable"
}

} // namespace Libs::Graphics::RefreshProf
