#include "graphics/host_gpu/renderer/cpOrderLedger.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::CpOrderLedger;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "CpOrderLedgerTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	std::vector<CpOrderLedger::Hit> hits;
	const auto keep  = [&](const CpOrderLedger::Hit& hit) { hits.push_back(hit); };
	const auto never = [](const CpOrderLedger::Hit&) { Check(false, "unexpected window hit"); };

	// Draws 1..4 queued (0 finished); the lookup of draw 5 reads two ranges.
	CpOrderLedger ledger;
	ledger.AddRead(0x10000, 0x10040, 5, 0, ledger.Serial(), never);
	ledger.AddRead(0x20000, 0x20010, 5, 0, ledger.Serial(), never);
	Check(ledger.Write(0x30000, 0x40000, 2, 0, keep) == 0 && hits.empty(), "a write elsewhere was a hazard");
	Check(ledger.Write(0x10020, 0x10100, 2, 0, keep) == 1 && hits.size() == 1, "an overlapping earlier writer was missed");
	Check(hits[0].address == 0x10020 && hits[0].size == 0x20 && hits[0].reader == 5 && hits[0].writer == 2,
	      "wrong overlap");
	// One hit per reader even when the write covers both of its reads.
	hits.clear();
	Check(ledger.Write(0x10000, 0x30000, 3, 0, keep) == 1, "one write counted a reader twice");

	// A writer at or after the reader is the normal order.
	CpOrderLedger later;
	later.AddRead(0x10000, 0x10040, 5, 0, later.Serial(), never);
	hits.clear();
	Check(later.Write(0x10000, 0x10040, 5, 0, keep) == 0 && hits.empty(), "the reader's own draw was a hazard");
	Check(later.Reads() == 0, "reads of a reader <= the running writer were kept");

	// Readers already reached by the back are dropped.
	CpOrderLedger pruned;
	pruned.AddRead(0x10000, 0x10040, 3, 0, pruned.Serial(), never);
	pruned.AddRead(0x50000, 0x50040, 7, 0, pruned.Serial(), never);
	pruned.Prune(3);
	Check(pruned.Reads() == 1, "prune kept a finished reader");
	hits.clear();
	Check(pruned.Write(0x10000, 0x10040, 4, 0, keep) == 0, "a pruned read was a hazard");
	Check(pruned.Write(0x50000, 0x50004, 4, 0, keep) == 1 && hits.back().reader == 7, "a live read was lost");
	// A read with no draw left before it is not recorded.
	pruned.AddRead(0x60000, 0x60004, 6, 6, pruned.Serial(), never);
	Check(pruned.Reads() == 1, "a read with nothing queued before it was recorded");

	// Window: a write noted after the lookup began, before the read is inserted.
	CpOrderLedger window;
	const auto since = window.Serial();
	Check(window.Write(0x10000, 0x10040, 2, 7, keep) == 0, "a write before any read hit");
	std::vector<CpOrderLedger::Hit> windows;
	const auto note = [&](const CpOrderLedger::Hit& hit) { windows.push_back(hit); };
	window.AddRead(0x10010, 0x10020, 5, 1, since, note);
	window.AddRead(0x10020, 0x10030, 5, 1, since, note);
	Check(windows.size() == 1 && windows[0].writer == 2 && windows[0].reader == 5 &&
	          windows[0].tag == 7, "window hit missed or doubled");
	// A write before the lookup began is not a window hit.
	window.AddRead(0x10000, 0x10004, 6, 1, window.Serial(), never);

	// The back finished every draw while the lookup ran: no read is kept, the window still pairs.
	CpOrderLedger drained;
	const auto start = drained.Serial();
	Check(drained.Write(0x10000, 0x10040, 3, 1, keep) == 0, "a write before any read hit (drained)");
	windows.clear();
	drained.AddRead(0x10000, 0x10004, 4, 4, start, note);
	Check(windows.size() == 1 && windows[0].writer == 3 && drained.Reads() == 0,
	      "a write during a lookup that ended with an empty queue was missed");
	Check(!drained.WindowOverflow(start), "a short window overflowed");
	for (uint32_t i = 0; i <= CpOrderLedger::RecentWrites; i++) (void)drained.Write(0x90000, 0x90004, 9, 0, keep);
	Check(drained.WindowOverflow(start), "a full ring did not overflow");
	// Granule meets: a write in the 64 KiB granule of a kept read, even without overlap.
	CpOrderLedger meets;
	meets.AddRead(0x10000, 0x10004, 5, 0, meets.Serial(), never);
	Check(meets.Write(0x10100, 0x10200, 2, 0, keep) == 0 && meets.GranuleMeets() == 1, "a granule meet was not counted");

	std::printf("CpOrderLedgerTests: all cases passed\n");
	return 0;
}
