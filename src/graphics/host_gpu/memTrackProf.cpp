#include "graphics/host_gpu/memTrackProf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::MemTrackProf {

namespace {

constexpr size_t FaultCount   = static_cast<size_t>(Fault::Count);
constexpr size_t KindCount    = static_cast<size_t>(Kind::Count);
constexpr size_t ProtCount    = static_cast<size_t>(Prot::Count);
constexpr size_t MapCount     = static_cast<size_t>(Mapping::Count);
constexpr size_t CounterCount = static_cast<size_t>(Counter::Count);
constexpr size_t MaxSlots     = 256;

using Atomic = std::atomic<uint64_t>;

// Per-thread counters. One writer (the owning thread); the report reads and resets them.
struct Slot {
	std::atomic<bool>                  used {false};
	ThreadKind                         kind = ThreadKind::Host;
	char                               name[32] {};
	std::array<Atomic, FaultCount>     faults {};
	std::array<Atomic, FaultCount>     fault_ns {};
	std::array<Atomic, KindCount>      kinds {};
	std::array<Atomic, KindCount>      kind_ns {};
	std::array<Atomic, ProtCount>      prot_calls {};
	std::array<Atomic, ProtCount>      prot_bytes {};
	std::array<Atomic, ProtCount>      prot_ns {};
	Atomic                             prot_fault_ns {}; // protection changes made inside a fault
	Atomic                             wait_ns {};       // FaultWaitNs of this thread
	std::array<Atomic, MapCount>       maps {};
};

std::array<Slot, MaxSlots>          g_slots;
std::atomic<size_t>                 g_slot_count {0};
std::array<Atomic, CounterCount>    g_counters {};
std::atomic<Namer>                  g_namer {nullptr};
std::atomic<Classifier>             g_classifier {nullptr};
std::atomic<MapStatsFn>             g_map_stats {nullptr};
Slot                                g_overflow; // threads beyond MaxSlots share it

struct ThreadState {
	Slot* slot         = nullptr;
	bool  in_fault     = false;
	bool  buffer_hit   = false;
	bool  texture_hit  = false;
};
thread_local ThreadState t_state;

Slot& NewSlot(ThreadKind kind, const char* name) noexcept {
	const auto index = g_slot_count.fetch_add(1, std::memory_order_relaxed);
	Slot&      slot  = index < MaxSlots ? g_slots[index] : g_overflow;
	if (index < MaxSlots) {
		slot.kind = kind;
		std::snprintf(slot.name, sizeof(slot.name), "%s", name);
		slot.used.store(true, std::memory_order_release);
	}
	return slot;
}

Slot& ThisSlot() noexcept {
	if (t_state.slot != nullptr) [[likely]] {
		return *t_state.slot;
	}
	char name[32] = "host";
	auto kind     = ThreadKind::Host;
	if (auto namer = g_namer.load(std::memory_order_acquire); namer != nullptr && namer(name, sizeof(name))) {
		kind = ThreadKind::Guest;
	}
	t_state.slot = &NewSlot(kind, name);
	return *t_state.slot;
}

void Bump(Atomic& counter, uint64_t value) noexcept {
	counter.fetch_add(value, std::memory_order_relaxed);
}

uint64_t Take(Atomic& counter) noexcept {
	return counter.exchange(0, std::memory_order_relaxed);
}

const char* KindName(ThreadKind kind) {
	switch (kind) {
		case ThreadKind::Gpu: return "gpu";
		case ThreadKind::Deferred: return "deferred";
		case ThreadKind::Guest: return "guest";
		default: return "host";
	}
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MEMTRACK_PROF");
		return value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

void Add(Counter counter, uint64_t value) noexcept {
	if (Enabled()) {
		Bump(g_counters[static_cast<size_t>(counter)], value);
	}
}

void SetThreadKind(ThreadKind kind, const char* name) noexcept {
	if (Enabled() && t_state.slot == nullptr) {
		t_state.slot = &NewSlot(kind, name);
	}
}

void SetNamer(Namer namer) noexcept {
	g_namer.store(namer, std::memory_order_release);
}

void SetClassifier(Classifier classifier) noexcept {
	g_classifier.store(classifier, std::memory_order_release);
}

void SetMapStats(MapStatsFn fn) noexcept {
	g_map_stats.store(fn, std::memory_order_release);
}

void BeginFault() noexcept {
	if (Enabled()) {
		t_state.in_fault    = true;
		t_state.buffer_hit  = false;
		t_state.texture_hit = false;
	}
}

void NoteBufferHit() noexcept {
	t_state.buffer_hit = true;
}

void NoteTextureHit() noexcept {
	t_state.texture_hit = true;
}

bool InFault() noexcept {
	return t_state.in_fault;
}

void NoteFault(Fault path, uint64_t vaddr, uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&      slot = ThisSlot();
	const auto f    = static_cast<size_t>(path);
	Bump(slot.faults[f], 1);
	Bump(slot.fault_ns[f], ns);
	if (path == Fault::Write) {
		const auto kind = t_state.buffer_hit ? (t_state.texture_hit ? Kind::Both : Kind::Buffer)
		                                     : (t_state.texture_hit ? Kind::Texture : Kind::Other);
		Bump(slot.kinds[static_cast<size_t>(kind)], 1);
		Bump(slot.kind_ns[static_cast<size_t>(kind)], ns);
	}
	if (path == Fault::Write || path == Fault::Read) {
		const auto classify = g_classifier.load(std::memory_order_acquire);
		const auto mapping  = classify != nullptr ? classify(vaddr) : Mapping::Unknown;
		Bump(slot.maps[static_cast<size_t>(mapping)], 1);
	}
	t_state.in_fault = false;
}

void NoteFaultWait(uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	Bump(ThisSlot().wait_ns, ns);
	Bump(g_counters[static_cast<size_t>(Counter::FaultWaitNs)], ns);
	Bump(g_counters[static_cast<size_t>(Counter::FaultWaitCount)], 1);
}

void NoteProtect(Prot mode, uint64_t bytes, uint64_t ns) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&      slot = ThisSlot();
	const auto m    = static_cast<size_t>(mode);
	Bump(slot.prot_calls[m], 1);
	Bump(slot.prot_bytes[m], bytes);
	Bump(slot.prot_ns[m], ns);
	if (t_state.in_fault) {
		Bump(slot.prot_fault_ns, ns);
	}
}

void Report(double seconds, uint64_t frames) noexcept {
	if (!Enabled() || seconds <= 0.0) {
		return;
	}
	const double fr = frames != 0 ? static_cast<double>(frames) : 1.0;
	// Per thread class, then the busiest threads by fault time.
	struct Totals {
		std::array<uint64_t, FaultCount> faults {}, fault_ns {};
		std::array<uint64_t, KindCount>  kinds {}, kind_ns {};
		std::array<uint64_t, ProtCount>  prot_calls {}, prot_bytes {}, prot_ns {};
		std::array<uint64_t, MapCount>   maps {};
		uint64_t                         prot_fault_ns = 0, wait_ns = 0;
	};
	std::array<Totals, static_cast<size_t>(ThreadKind::Count)> by_kind {};
	struct Top {
		const Slot* slot;
		uint64_t    faults, fault_ns, prot_ns, wait_ns;
	};
	std::array<Top, MaxSlots + 1> tops {};
	size_t                        top_count = 0;
	const auto                    used      = std::min(g_slot_count.load(std::memory_order_relaxed), MaxSlots);
	for (size_t i = 0; i <= used; i++) {
		Slot& slot = i < used ? g_slots[i] : g_overflow;
		if (i < used && !slot.used.load(std::memory_order_acquire)) {
			continue;
		}
		auto& t      = by_kind[static_cast<size_t>(slot.kind)];
		Top   top {&slot, 0, 0, 0, 0};
		for (size_t f = 0; f < FaultCount; f++) {
			const auto n  = Take(slot.faults[f]);
			const auto ns = Take(slot.fault_ns[f]);
			t.faults[f] += n;
			t.fault_ns[f] += ns;
			top.faults += n;
			top.fault_ns += ns;
		}
		for (size_t k = 0; k < KindCount; k++) {
			t.kinds[k] += Take(slot.kinds[k]);
			t.kind_ns[k] += Take(slot.kind_ns[k]);
		}
		for (size_t m = 0; m < ProtCount; m++) {
			t.prot_calls[m] += Take(slot.prot_calls[m]);
			t.prot_bytes[m] += Take(slot.prot_bytes[m]);
			const auto ns = Take(slot.prot_ns[m]);
			t.prot_ns[m] += ns;
			top.prot_ns += ns;
		}
		for (size_t m = 0; m < MapCount; m++) {
			t.maps[m] += Take(slot.maps[m]);
		}
		t.prot_fault_ns += Take(slot.prot_fault_ns);
		top.wait_ns = Take(slot.wait_ns);
		t.wait_ns += top.wait_ns;
		if (top.faults != 0 || top.prot_ns != 0) {
			tops[top_count++] = top;
		}
	}
	std::array<uint64_t, CounterCount> c {};
	for (size_t i = 0; i < CounterCount; i++) {
		c[i] = Take(g_counters[i]);
	}
	const auto ms  = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / fr; };
	const auto ps  = [&](uint64_t n) { return static_cast<double>(n) / seconds; };
	const auto cn  = [&](Counter k) { return c[static_cast<size_t>(k)]; };
	const auto mib = [&](uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0) / fr; };
	// Units: counts per second, times in ms per frame, bytes in MiB per frame.
	std::printf("PERF memtrack %.1fs frames=%" PRIu64 " (counts /s, ms and MiB /frame): sync=%.2fms(%.0f) upload=%.2fMiB(%.0f copies,"
	            " copy=%.2fms) download=%.2fMiB(%.0f) dl_sync_wait=%.2fms(%.0f) fault_cmd=%.2fms(%.0f) fault_send=%.2fms"
	            " fault_wait=%.2fms(%.0f) write_flush=%.0f invalidate=%.0f(%.2fMiB) readahead_pages=%.0f\n",
	            seconds, frames, ms(cn(Counter::SyncNs)), ps(cn(Counter::SyncCount)), mib(cn(Counter::UploadBytes)),
	            ps(cn(Counter::UploadCopies)), ms(cn(Counter::UploadCopyNs)), mib(cn(Counter::DownloadBytes)),
	            ps(cn(Counter::DownloadCount)), ms(cn(Counter::DownloadSyncNs)), ps(cn(Counter::DownloadSyncCount)),
	            ms(cn(Counter::FaultCmdNs)), ps(cn(Counter::FaultCmdCount)), ms(cn(Counter::FaultSendNs)),
	            ms(cn(Counter::FaultWaitNs)), ps(cn(Counter::FaultWaitCount)), ps(cn(Counter::WriteFlushCount)),
	            ps(cn(Counter::InvalidateCount)), mib(cn(Counter::InvalidateBytes)), ps(cn(Counter::ReadAheadPages)));
	for (size_t k = 0; k < by_kind.size(); k++) {
		const auto& t = by_kind[k];
		uint64_t    any = t.prot_calls[0] + t.prot_calls[1] + t.prot_calls[2] + t.wait_ns;
		for (auto n: t.faults) any += n;
		if (any == 0) {
			continue;
		}
		std::printf("PERF memtrack %s: faults w=%.0f(%.2fms) r=%.0f(%.2fms) remap=%.0f(%.2fms) walker=%.0f(%.2fms)"
		            " skip=%.0f | wkind buf=%.0f(%.2fms) tex=%.0f(%.2fms) both=%.0f other=%.0f(%.2fms)"
		            " | map backing=%.0f private=%.0f unknown=%.0f | prot rw=%.0f(%.1fMiB,%.2fms) ro=%.0f(%.1fMiB,%.2fms)"
		            " na=%.0f(%.1fMiB,%.2fms) in_fault=%.2fms | wait=%.2fms\n",
		            KindName(static_cast<ThreadKind>(k)), ps(t.faults[0]), ms(t.fault_ns[0]), ps(t.faults[1]),
		            ms(t.fault_ns[1]), ps(t.faults[2]), ms(t.fault_ns[2]), ps(t.faults[3]), ms(t.fault_ns[3]),
		            ps(t.faults[4]), ps(t.kinds[0]), ms(t.kind_ns[0]), ps(t.kinds[1]), ms(t.kind_ns[1]),
		            ps(t.kinds[2]), ps(t.kinds[3]), ms(t.kind_ns[3]), ps(t.maps[0]), ps(t.maps[1]), ps(t.maps[2]),
		            ps(t.prot_calls[0]), mib(t.prot_bytes[0]), ms(t.prot_ns[0]), ps(t.prot_calls[1]),
		            mib(t.prot_bytes[1]), ms(t.prot_ns[1]), ps(t.prot_calls[2]), mib(t.prot_bytes[2]),
		            ms(t.prot_ns[2]), ms(t.prot_fault_ns), ms(t.wait_ns));
	}
	std::sort(tops.begin(), tops.begin() + static_cast<std::ptrdiff_t>(top_count), [](const Top& a, const Top& b) {
		return a.fault_ns + a.prot_ns + a.wait_ns > b.fault_ns + b.prot_ns + b.wait_ns;
	});
	std::printf("PERF memtrack top:");
	for (size_t i = 0; i < std::min<size_t>(top_count, 8); i++) {
		const auto& top = tops[i];
		std::printf(" [%s %s f=%.0f %.2fms p=%.2fms w=%.2fms]", KindName(top.slot->kind), top.slot->name,
		            ps(top.faults), ms(top.fault_ns), ms(top.prot_ns), ms(top.wait_ns));
	}
	std::printf("\n");
	if (const auto fn = g_map_stats.load(std::memory_order_acquire); fn != nullptr) {
		const auto m = fn();
		std::printf("PERF memtrack map: backing=%.1fMiB(%" PRIu64 " views) aliased=%.1fMiB private=%.1fMiB\n",
		            static_cast<double>(m.backing_bytes) / 1048576.0, m.backing_views,
		            static_cast<double>(m.aliased_bytes) / 1048576.0, static_cast<double>(m.private_bytes) / 1048576.0);
	}
}

} // namespace Libs::Graphics::MemTrackProf
