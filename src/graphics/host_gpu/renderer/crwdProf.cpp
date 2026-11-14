#include "graphics/host_gpu/renderer/crwdProf.h"

#include "graphics/host_gpu/renderer/perfStats.h"

#include <xxhash.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace Libs::Graphics::CrwdProf {

namespace {

struct Tables {
	std::mutex  mutex;
	ToggleTable draw {true};  // direct draws by (VS, PS): draws and instance sum
	ToggleTable drwi {true};  // indirect draws by (VS, PS): draws (instances are in GPU records)
	// Indirect packets by their order in the frame (their arguments sit in rings: new addresses each frame).
	ToggleTable icpu {true}; // read by the CPU: count and records
	ToggleTable igpu {true}; // read by the GPU: count only (records not observed)
	uint64_t    icpu_next = 0, igpu_next = 0;
	ToggleTable side {false}; // (address, size) of GPU-written bytes read back: content
	uint64_t    frames = 0;
	uint64_t    side_bytes = 0;
};

Tables& Get() {
	static Tables tables;
	return tables;
}

uint64_t Mix(uint64_t a, uint64_t b) {
	return (a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6u) + (a >> 2u))) * 0xBF58476D1CE4E5B9ull;
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CRWD_PROF");
		return PerfStats::Enabled() && value != nullptr && std::atoi(value) >= 2;
	}();
	return enabled;
}

void NoteDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t instances, bool indirect) noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	// Draw count in the high bits, instance sum in the low 40.
	(indirect ? t.drwi : t.draw)
	    .AddSum(Mix(vs_hash, ps_hash), vs_hash, ps_hash, (uint64_t {1} << 40u) + (indirect ? 0 : instances));
}

void NoteIndirect(uint64_t args_addr, uint32_t count, bool cpu, uint64_t records_hash) noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	if (cpu) {
		t.icpu.AddSeq(t.icpu_next++, args_addr, count, Mix(count, records_hash));
	} else {
		t.igpu.AddSeq(t.igpu_next++, args_addr, count, count);
	}
}

void NoteSideRead(uint64_t address, uint64_t size, const void* data) noexcept {
	const auto      hash = XXH3_64bits(data, static_cast<size_t>(size));
	auto&           t    = Get();
	std::lock_guard lock(t.mutex);
	t.side.AddSeq(Mix(address, size), address, size, hash);
	t.side_bytes += size;
}

void EndFrame() noexcept {
	auto&           t = Get();
	std::lock_guard lock(t.mutex);
	t.draw.EndFrame();
	t.drwi.EndFrame();
	t.icpu.EndFrame();
	t.igpu.EndFrame();
	t.side.EndFrame();
	t.icpu_next = t.igpu_next = 0;
	t.frames++;
}

void Report(std::string& out) noexcept {
	struct Top {
		char     tag;
		uint64_t a, b;
		uint32_t evaluated, toggles;
	};
	auto&            t = Get();
	std::lock_guard  lock(t.mutex);
	std::vector<Top> top;
	char             text[256];
	std::snprintf(text, sizeof(text), " frames=%" PRIu64 " side_bytes=%" PRIu64, t.frames, t.side_bytes);
	out += text;
	const auto table = [&](const char* name, char tag, ToggleTable& table) {
		uint64_t evaluated = 0, changes = 0, toggles = 0, hot = 0, max_pct = 0;
		for (const auto& [key, e]: table.Entries()) {
			evaluated += e.evaluated;
			changes += e.changes;
			toggles += e.toggles;
			// Gate keys: >= 20 % of at least 20 frame pairs.
			if (e.evaluated >= 20) {
				hot += e.toggles * 5 >= e.evaluated ? 1 : 0;
				max_pct = std::max<uint64_t>(max_pct, uint64_t {e.toggles} * 100 / e.evaluated);
			}
			if (e.toggles != 0) {
				top.push_back({tag, e.a, e.b, e.evaluated, e.toggles});
			}
		}
		std::snprintf(text, sizeof(text),
		              " %s=keys:%zu,eval:%" PRIu64 ",chg:%" PRIu64 ",tog:%" PRIu64 ",hot20:%" PRIu64 ",max:%" PRIu64 "%%",
		              name, table.Entries().size(), evaluated, changes, toggles, hot, max_pct);
		out += text;
		table.Clear();
	};
	table("draw", 'D', t.draw);
	table("drwi", 'd', t.drwi);
	table("icpu", 'I', t.icpu);
	table("igpu", 'G', t.igpu);
	table("side", 'S', t.side);
	const auto count = std::min<size_t>(top.size(), 5);
	std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(count), top.end(),
	                  [](const Top& l, const Top& r) {
		                  return l.toggles != r.toggles ? l.toggles > r.toggles : l.evaluated < r.evaluated;
	                  });
	out += " top:";
	for (size_t i = 0; i < count; i++) {
		const auto& e = top[i];
		// D, d: vs/ps (d = indirect draws); I, G: last packet address/count at that rank; S: address/size.
		std::snprintf(text, sizeof(text), " %c:%016" PRIx64 "/%" PRIx64 ":%u/%u=%u%%", e.tag, e.a, e.b, e.toggles,
		              e.evaluated, e.toggles * 100 / std::max(e.evaluated, 1u));
		out += text;
	}
	t.frames     = 0;
	t.side_bytes = 0;
}

} // namespace Libs::Graphics::CrwdProf
