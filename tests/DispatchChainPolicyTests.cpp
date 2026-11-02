#include "graphics/host_gpu/renderer/dispatchChainPolicy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::DispatchChain;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DispatchChainPolicyTests: failed: %s\n", message);
		std::abort();
	}
}

Accesses Make(std::initializer_list<Access> ranges, bool writes, bool unknown = false) {
	Accesses result;
	result.ranges.assign(ranges.begin(), ranges.end());
	result.writes  = writes;
	result.unknown = unknown;
	return result;
}

constexpr uint64_t BufferA = 0x1000;
constexpr uint64_t BufferB = 0x2000;

// Decides, then commits, as DispatchDirect does.
Decision Record(State& state, const Accesses& next, uint64_t epoch) {
	const auto decision = Decide(state, next, epoch);
	Commit(state, next, epoch, decision.barrier);
	return decision;
}

} // namespace

int main() {
	const auto read_a  = Make({{BufferA, 0, 256, false}}, false);
	const auto write_a = Make({{BufferA, 0, 256, true}}, true);
	const auto write_a_hi = Make({{BufferA, 256, 512, true}}, true);
	const auto write_b = Make({{BufferB, 0, 64, true}}, true);
	const auto read_b  = Make({{BufferB, 0, 64, false}}, false);

	{
		// A chain opens with the hazard barrier of a writer, as without chaining.
		State state;
		const auto first = Record(state, write_a, 1);
		Check(first.barrier == Barrier::Hazard && first.reason == Reason::Fresh, "a writer opening a chain waits for earlier work");
		Check(state.pending && state.hazard_done, "the chain owes its trailing barrier");
		// Disjoint writers of the same epoch share the barriers.
		const auto second = Record(state, write_b, 1);
		Check(second.barrier == Barrier::None && second.reason == Reason::Chained, "disjoint writers of one epoch chain");
		const auto third = Record(state, write_a_hi, 1);
		Check(third.barrier == Barrier::None, "adjacent ranges of one buffer do not overlap");
		// RAW on a buffer written since the last barrier keeps an ordering barrier.
		const auto raw = Record(state, read_b, 1);
		Check(raw.barrier == Barrier::ComputeToCompute && raw.reason == Reason::Conflict, "read after write in one epoch waits");
		// After that barrier, the window restarts with the reader only.
		Check(state.window.size() == 1, "a barrier restarts the tracked window");
		const auto war = Record(state, write_b, 1);
		Check(war.barrier == Barrier::ComputeToCompute && war.reason == Reason::Conflict, "write after read waits");
		const auto waw = Record(state, write_b, 1);
		Check(waw.barrier == Barrier::ComputeToCompute && waw.reason == Reason::Conflict, "write after write waits");
	}
	{
		// Readers never conflict with readers.
		State state;
		Check(Record(state, read_a, 1).barrier == Barrier::None, "a reader opening a chain needs no barrier");
		Check(!state.hazard_done, "a reader records no hazard barrier");
		Check(Record(state, read_a, 1).barrier == Barrier::None, "readers of one range chain");
		// The first writer of a chain opened by readers still waits for the earlier work.
		const auto writer = Record(state, write_b, 1);
		Check(writer.barrier == Barrier::Hazard && writer.reason == Reason::Hazard, "the first writer of a chain records the hazard barrier");
		Check(state.hazard_done, "one hazard barrier per chain");
		Check(Record(state, Make({{BufferA, 512, 600, true}}, true), 1).barrier == Barrier::None,
		      "later disjoint writers skip the hazard barrier");
	}
	{
		// A guest sync point orders the next dispatch after the chain.
		State state;
		Record(state, write_a, 1);
		const auto next = Record(state, write_b, 2);
		Check(next.barrier == Barrier::ComputeToCompute && next.reason == Reason::Epoch, "a new sync epoch orders the dispatches");
	}
	{
		// Unknown accesses on either side keep an ordering barrier.
		State state;
		Record(state, write_a, 1);
		const auto unknown = Record(state, Make({}, true, true), 1);
		Check(unknown.barrier == Barrier::ComputeToCompute && unknown.reason == Reason::Unknown, "an address writer waits for the chain");
		const auto after = Record(state, write_b, 1);
		Check(after.barrier == Barrier::ComputeToCompute && after.reason == Reason::Unknown, "the dispatch after an address writer waits for it");
		Check(Record(state, Make({{BufferB, 128, 192, true}}, true), 1).barrier == Barrier::None,
		      "once ordered after it, the chain continues");
	}
	{
		// Images are tracked as a whole.
		State state;
		Record(state, Make({{ImageKey(7), 0, UINT64_MAX, true}}, true), 1);
		Check(Record(state, Make({{ImageKey(8), 0, UINT64_MAX, false}}, false), 1).barrier == Barrier::None,
		      "another image chains");
		Check(Record(state, Make({{ImageKey(7), 0, UINT64_MAX, false}}, false), 1).barrier == Barrier::ComputeToCompute,
		      "sampling an image written in the chain waits");
		Check(ImageKey(7) != 7 && ImageKey(7) != BufferA, "image keys never alias buffer handles");
	}
	{
		// An image and a buffer over the same guest memory: different host resources, same bytes.
		State state;
		Record(state, Make({{ImageKey(3), 0, UINT64_MAX, false}, {GuestMemory, 0x10000, 0x90000, false}}, false), 1);
		const auto alias = Record(state, Make({{BufferA, 0, 0x4000, true}, {GuestMemory, 0x50000, 0x54000, true}}, true), 1);
		Check(alias.barrier == Barrier::Hazard, "the first writer still records the hazard barrier");
		const auto again = Record(state, Make({{BufferB, 0, 0x4000, true}, {GuestMemory, 0x50000, 0x54000, true}}, true), 1);
		Check(again.barrier == Barrier::ComputeToCompute && again.reason == Reason::Conflict,
		      "a write to guest memory written in the chain through another resource waits");
		Check(Record(state, Make({{BufferA, 0x8000, 0x9000, true}, {GuestMemory, 0x58000, 0x59000, true}}, true), 1).barrier == Barrier::None,
		      "disjoint guest memory chains");
	}
	{
		// The tracked window is bounded.
		State state;
		Record(state, read_a, 1);
		uint64_t ordered = 0;
		for (uint64_t i = 0; i < MaxTrackedRanges + 8; i++) {
			const auto d = Record(state, Make({{BufferB + 1 + i, 0, 16, false}}, false), 1);
			ordered += d.barrier != Barrier::None;
			Check(d.barrier == Barrier::None || d.reason == Reason::Cap, "only the cap orders disjoint readers");
		}
		Check(ordered == 1 && state.window.size() <= MaxTrackedRanges, "the cap restarts the window once");
	}
	{
		// Once the trailing barrier is recorded (anything else recorded), the next writer waits
		// for everything again.
		State state;
		Record(state, write_a, 1);
		state.Reset();
		const auto fresh = Record(state, write_b, 1);
		Check(fresh.barrier == Barrier::Hazard && fresh.reason == Reason::Fresh, "a flushed chain starts again with a hazard barrier");
	}
	std::printf("DispatchChainPolicyTests: ok\n");
	return 0;
}
