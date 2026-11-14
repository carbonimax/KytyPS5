#include "graphics/host_gpu/renderer/crwdProf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using Libs::Graphics::CrwdProf::ToggleTable;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CrwdProfTests: failed: %s\n", text);
		std::abort();
	}
}

// One value per frame for key 1; 0 = the key is absent that frame.
const ToggleTable::Entry& Run(ToggleTable& table, std::initializer_list<uint64_t> frames) {
	for (const auto v: frames) {
		if (v != 0) table.AddSum(1, 0, 0, v);
		table.EndFrame();
	}
	return table.Entries().at(1);
}

void TestToggles() {
	ToggleTable stable(false);
	const auto& s = Run(stable, {5, 5, 5, 5, 5});
	Check(s.evaluated == 3 && s.changes == 0 && s.toggles == 0, "stable value: no change, no toggle");

	ToggleTable flip(false);
	const auto& f = Run(flip, {5, 7, 5, 7, 5});
	Check(f.evaluated == 3 && f.changes == 3 && f.toggles == 3, "A,B,A,B,A: every pair toggles");

	ToggleTable drift(false);
	const auto& d = Run(drift, {1, 2, 3, 4, 5});
	Check(d.changes == 3 && d.toggles == 0, "a value that keeps changing is not a toggle");

	ToggleTable ring(false);
	const auto& r = Run(ring, {5, 0, 5, 0, 5, 5, 5});
	Check(r.evaluated == 1 && r.toggles == 0, "addresses: frames without the key are not compared");

	ToggleTable draws(true);
	const auto& w = Run(draws, {5, 0, 5, 0, 5});
	Check(w.evaluated == 5 && w.toggles == 4, "draws: a key present every other frame toggles");

	ToggleTable sum(false);
	for (int i = 0; i < 4; i++) {
		sum.AddSum(1, 0, 0, 2);
		if (i % 2 == 0) sum.AddSum(1, 0, 0, 2); // two draws, then one
		sum.EndFrame();
	}
	const auto& m = sum.Entries().at(1);
	Check(m.toggles == 2, "sums per frame: 4,2,4,2");
	sum.Clear();
	Check(sum.Entries().at(1).toggles == 0 && sum.Entries().at(1).evaluated == 0, "Clear starts a new window");

	ToggleTable seq(false);
	for (int i = 0; i < 4; i++) {
		seq.AddSeq(1, 0, 0, i % 2 == 0 ? 10 : 20);
		seq.AddSeq(1, 0, 0, i % 2 == 0 ? 20 : 10);
		seq.EndFrame();
	}
	Check(seq.Entries().at(1).toggles == 2, "ordered combination: (10,20),(20,10) differ");

	ToggleTable gone(true);
	gone.AddSum(1, 0, 0, 1);
	for (int i = 0; i < 4; i++) gone.EndFrame();
	Check(gone.Entries().size() == 1, "kept while its window has counts to report");
	gone.Clear();
	gone.EndFrame();
	Check(gone.Entries().empty(), "a key absent for 3 frames with nothing to report is dropped");
}

} // namespace

int main() {
	TestToggles();
	std::printf("CrwdProfTests: ok\n");
	return 0;
}
