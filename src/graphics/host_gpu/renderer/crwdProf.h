#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CRWDPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CRWDPROF_H_

#include <cstdint>
#include <string>
#include <unordered_map>

// KYTY_CRWD_PROF=2 (diagnostic, default off; needs KYTY_PERF_STATS=1): does the draw list change from
// frame to frame? Per frame it keys draws by (VS, PS), indirect packets by rank and GPU-written bytes
// the GPU thread reads back by (address, size), and counts the keys whose per-frame value alternates
// A, B, A. The "PERF crwd2" line prints the window totals and the top toggling keys.
namespace Libs::Graphics::CrwdProf {

[[nodiscard]] bool Enabled() noexcept;
// One draw. `indirect`: the instance count comes from GPU records and is not known here.
void NoteDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t instances, bool indirect) noexcept;
// One indirect draw packet: the count read and, when the CPU read the records (`cpu`), a hash of them.
// The GPU reads the other packets' records itself: guest memory does not show them.
void NoteIndirect(uint64_t args_addr, uint32_t count, bool cpu, uint64_t records_hash) noexcept;
// GPU-written bytes the GPU thread just read back from guest memory.
void NoteSideRead(uint64_t address, uint64_t size, const void* data) noexcept;
// GPU thread, at each frame boundary.
void EndFrame() noexcept;
// Appends the window's tables and top toggling keys to `out`, one "PERF crwd2 hot" line per table with
// keys at >= 20 % to `hot`, and starts a new window.
void Report(std::string& out, std::string& hot) noexcept;
// Buffer joins made by this thread (#2224: a draw input acquisition joining a bound buffer).
inline thread_local uint64_t g_joins = 0;

// Per-key values of consecutive frames. A key toggles at frame N when its value at N equals the one
// at N-2 and differs from N-1. `absent_counts`: a frame without the key is a value of its own (a draw
// that comes and goes); otherwise only frames that all have the key are compared (ring addresses).
class ToggleTable {
public:
	struct Entry {
		uint64_t a = 0, b = 0; // printed identity
		uint64_t cur       = 0;
		bool     seen      = false;
		uint64_t v1 = 0, v2 = 0; // values at N-1 and N-2
		uint8_t  have      = 0;  // bit 0: present at N-1, bit 1: present at N-2
		uint32_t idle      = 0;
		uint32_t evaluated = 0, changes = 0, toggles = 0;
		uint64_t tog_v0 = 0, tog_v1 = 0; // values at N and N-1 of the latest toggle
		uint8_t  tog_have = 0;           // bit 0: present at N, bit 1: present at N-1
	};

	explicit ToggleTable(bool absent_counts): m_absent_counts(absent_counts) {}

	// Adds `value` to the frame's sum for the key.
	void AddSum(uint64_t key, uint64_t a, uint64_t b, uint64_t value) {
		auto& e = Touch(key, a, b);
		e.cur += value;
	}
	// Folds `value` into the frame's ordered combination for the key.
	void AddSeq(uint64_t key, uint64_t a, uint64_t b, uint64_t value) {
		auto& e = Touch(key, a, b);
		e.cur   = e.cur * 0x9E3779B97F4A7C15ull + value + 1;
	}

	void EndFrame() {
		for (auto it = m_map.begin(); it != m_map.end();) {
			auto&      e  = it->second;
			const bool p0 = e.seen, p1 = (e.have & 1u) != 0, p2 = (e.have & 2u) != 0;
			const bool eval = m_absent_counts ? (p0 || p1 || p2) : (p0 && p1 && p2);
			if (eval) {
				const auto same = [](bool pa, uint64_t va, bool pb, uint64_t vb) { return pa == pb && (!pa || va == vb); };
				e.evaluated++;
				e.changes += same(p0, e.cur, p1, e.v1) ? 0u : 1u;
				if (same(p0, e.cur, p2, e.v2) && !same(p0, e.cur, p1, e.v1)) {
					e.toggles++;
					e.tog_v0   = e.cur;
					e.tog_v1   = e.v1;
					e.tog_have = static_cast<uint8_t>((p0 ? 1u : 0u) | (p1 ? 2u : 0u));
				}
			}
			e.v2   = e.v1;
			e.v1   = e.cur;
			e.have = static_cast<uint8_t>(((e.have << 1u) | (p0 ? 1u : 0u)) & 3u);
			e.idle = p0 ? 0 : e.idle + 1;
			e.cur  = 0;
			e.seen = false;
			// shortcut: an idle key's window counts are dropped once reported (Clear), not kept for later windows.
			if (e.idle >= 3 && e.evaluated == 0) {
				it = m_map.erase(it);
			} else {
				++it;
			}
		}
	}

	// Starts a new window (history is kept).
	void Clear() {
		for (auto& [key, e]: m_map) {
			e.evaluated = e.changes = e.toggles = 0;
		}
	}

	[[nodiscard]] const std::unordered_map<uint64_t, Entry>& Entries() const { return m_map; }

private:
	Entry& Touch(uint64_t key, uint64_t a, uint64_t b) {
		auto& e = m_map[key];
		e.a     = a;
		e.b     = b;
		e.seen  = true;
		return e;
	}

	bool                                m_absent_counts;
	std::unordered_map<uint64_t, Entry> m_map;
};

} // namespace Libs::Graphics::CrwdProf

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CRWDPROF_H_ */
