#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/guestStacks.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

struct DeferredSpan {
	void*                       impl; // PageManager::Impl
	uint64_t                    begin;
	uint64_t                    end;
	Common::VirtualMemory::Mode mode;
};

// Per thread: whether this thread defers the protections of its capture scopes, and the flush state.
struct DeferredProtect {
	bool                      active   = false;
	bool                      capture  = false;
	bool                      applying = false; // inside FlushDeferredProtect: protect at once
	std::vector<DeferredSpan> scratch;
};

thread_local DeferredProtect t_deferred;

// Recorded spans, shared by the deferring threads (the GPU thread and the command-processor back
// thread take turns; either may flush what the other recorded). Taken inside a region lock, never
// the other way round.
std::atomic_flag          g_deferred_lock = ATOMIC_FLAG_INIT;
std::vector<DeferredSpan> g_deferred_spans;
std::atomic<bool>         g_deferred_any {false};

// KYTY_REPROTECT_BATCH=2: after a flush, the host protection of every page it was asked to protect
// is read back (VirtualQuery) and compared with the page state.
std::atomic<bool>     g_verify_deferred {false};
std::atomic<uint64_t> g_verify_checked {0};
std::atomic<uint64_t> g_verify_mismatch {0};

// KYTY_NOOBS_PROF (measurement, step 4.7): see PageManager::NoObsEntry.
namespace NoObs {

using Source                    = PageManager::NoObsSource;
constexpr uint8_t  NONE         = 0xff;
constexpr uint32_t SOURCES      = static_cast<uint32_t>(Source::Count) + 1; // + none
constexpr uint32_t AGES         = 7; // none, after, <10us, <100us, <1ms, <10ms, >=10ms
constexpr uint32_t LAT          = 4; // <10us, <100us, <1ms, >=1ms
constexpr uint32_t REPROT       = 5; // <100us, <1ms, <10ms, <100ms, >=100ms
constexpr uint32_t SLOTS        = 1u << 14u;
enum Host : uint8_t { HostUnknown, HostWritable, HostRead, HostNoAccess };
enum Lost : uint32_t { LostHandling, LostDispatch, LostCount };
enum Kind : uint32_t { KindSame, KindOther, KindGpu, KindCount };

bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_NOOBS_PROF");
		return PerfStats::Enabled() && value != nullptr && std::atoi(value) != 0;
	}();
	return on;
}

thread_local uint8_t  t_source = NONE;
thread_local uint32_t t_tid    = 0;
std::atomic<uint32_t> g_next_tid {1};
uint32_t              Tid() {
    if (t_tid == 0) {
        t_tid = g_next_tid.fetch_add(1, std::memory_order_relaxed);
    }
    return t_tid;
}

// Last protection change of a page (region lock).
struct PageLog {
	uint64_t release_us  = 0;
	uint64_t protect_us  = 0;
	uint32_t release_tid = 0;
	uint8_t  release_src = NONE;
	uint8_t  protect_src = NONE;
	bool     release_gpu = false;
};

struct Stash {
	uint64_t page     = UINT64_MAX;
	uint64_t entry_us = 0;
	uint32_t watchers = 0;
	Host     host     = HostUnknown;
};
thread_local Stash t_stash;

using Counter = std::atomic<uint64_t>;
struct Totals {
	Counter faults, untracked, noentry, entry0_watched, unknown, stale;
	Counter lost[LostCount];
	Counter rel[LostCount][SOURCES][KindCount];
	Counter age[LostCount][AGES];
	Counter stale_src[SOURCES];
	Counter stale_age[AGES];
	Counter stale_noaccess;
	Counter stale_last_release[SOURCES];
	Counter lat_all_us, lat_all_n, lat_unt_us;
	Counter lat[LAT];
	Counter reprot_in, reprot_over;
	Counter reprot[REPROT];
};
Totals g_totals;

// Untracked fault pages waiting for their next protection edge (direct-mapped, page + 1).
struct Slot {
	std::atomic<uint64_t> page {0};
	std::atomic<uint64_t> us {0};
};
std::array<Slot, SLOTS> g_slots;

struct RegionStat {
	uint64_t                     faults = 0;
	uint64_t                     stale  = 0;
	uint64_t                     lost[LostCount] {};
	uint64_t                     first  = UINT64_MAX;
	uint64_t                     last   = 0;
	std::unordered_set<uint64_t> pages;
};
std::mutex                               g_region_mutex;
std::unordered_map<uint64_t, RegionStat> g_regions;

uint32_t AgeBucket(uint64_t at, uint64_t since) {
	if (since == 0) return 0;
	if (since > at) return 1;
	const auto d = at - since;
	return d < 10 ? 2 : d < 100 ? 3 : d < 1000 ? 4 : d < 10000 ? 5 : 6;
}

uint32_t SourceIndex(uint8_t source) {
	return source == NONE ? SOURCES - 1 : source;
}

// Region lock held: a protection edge (RW <-> protected) of `page`.
void NoteEdge(PageLog& log, uint64_t page, bool release, uint64_t now) {
	const auto source = t_source;
	if (release) {
		log.release_us  = now;
		log.release_src = source;
		log.release_tid = Tid();
		log.release_gpu = GuestGpu::IsGpuThread();
		return;
	}
	log.protect_us  = now;
	log.protect_src = source;
	auto& slot      = g_slots[page % SLOTS];
	if (slot.page.load(std::memory_order_relaxed) == page + 1) {
		const auto d = now - std::min(now, slot.us.load(std::memory_order_relaxed));
		slot.page.store(0, std::memory_order_relaxed);
		g_totals.reprot[d < 100 ? 0 : d < 1000 ? 1 : d < 10000 ? 2 : d < 100000 ? 3 : 4].fetch_add(
		    1, std::memory_order_relaxed);
	}
}

// Sets the source of the protection changes of a public PageManager call unless a caller tagged it.
struct DefaultSource {
	explicit DefaultSource(Source source) noexcept: previous(t_source) {
		if (previous == NONE) {
			t_source = static_cast<uint8_t>(source);
		}
	}
	~DefaultSource() { t_source = previous; }
	uint8_t previous;
};

Host QueryHost(uint64_t vaddr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<void*>(vaddr), &info, sizeof(info)) == 0) {
		return HostUnknown;
	}
	const auto protect = info.Protect & 0xffu;
	if (protect == PAGE_READWRITE || protect == PAGE_WRITECOPY || protect == PAGE_EXECUTE_READWRITE ||
	    protect == PAGE_EXECUTE_WRITECOPY) {
		return HostWritable;
	}
	return protect == PAGE_NOACCESS ? HostNoAccess : HostRead;
#else
	(void)vaddr;
	return HostUnknown;
#endif
}

} // namespace NoObs

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  = 0;
		uint8_t access_watchers = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers == UINT8_MAX) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == UINT8_MAX) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 2);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		std::unique_ptr<NoObs::PageLog[]>   log; // KYTY_NOOBS_PROF only
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (uint64_t index = 0; index < REGION_COUNT; index++) {
			auto* region = regions[index].load(std::memory_order_relaxed);
			if (region == nullptr) continue;
			SpinGuard lock(region->lock);
			for (uint64_t i = 0; i < REGION_PAGES; i++) {
				const auto& page = region->pages[i];
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					Fatal("destroyed with live page state at 0x%016" PRIx64 " (write %u, access %u)",
					      index * REGION_SIZE + i * PAGE_SIZE, static_cast<unsigned>(page.write_watchers),
					      static_cast<unsigned>(page.access_watchers));
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return regions[vaddr / REGION_SIZE].load(std::memory_order_acquire);
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		if (NoObs::On()) {
			region->log = std::make_unique<NoObs::PageLog[]>(REGION_PAGES);
		}
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		if (t_deferred.capture && t_deferred.active && !t_deferred.applying &&
		    mode != Common::VirtualMemory::Mode::ReadWrite) [[unlikely]] {
			SpinGuard lock(g_deferred_lock);
			auto&     spans = g_deferred_spans;
			if (!spans.empty() && spans.back().impl == this && spans.back().mode == mode &&
			    spans.back().end == vaddr) {
				spans.back().end = vaddr + size;
			} else {
				spans.push_back({this, vaddr, vaddr + size, mode});
			}
			g_deferred_any.store(true, std::memory_order_relaxed);
			return;
		}
		ForEachOutsideStacks(vaddr, size, mode, [](uint64_t begin, uint64_t bytes, Common::VirtualMemory::Mode m) {
			if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(begin, bytes, m)) {
				Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, begin,
				      static_cast<uint32_t>(m));
			}
		});
	}

	// Applies `protect` to [vaddr, vaddr + size) except the pages of live guest stacks, which keep
	// their access (GuestStacks): Windows writes a fault's exception frame on the faulting stack.
	template <typename F>
	static void ForEachOutsideStacks(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode, F&& protect) {
		if (mode == Common::VirtualMemory::Mode::ReadWrite || !GuestStacks::MaybeHit(vaddr, size)) [[likely]] {
			protect(vaddr, size, mode);
			return;
		}
		const auto end = vaddr + size;
		for (auto at = vaddr; at < end;) {
			uint64_t skip_begin = end, skip_end = end;
			if (!GuestStacks::SkipLive(at, end, mode == Common::VirtualMemory::Mode::NoAccess, &skip_begin,
			                           &skip_end)) {
				skip_begin = skip_end = end;
			}
			if (skip_begin > at) {
				protect(at, skip_begin - at, mode);
			}
			at = skip_end;
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		const auto noobs_now            = region.log ? PerfStats::SteadyUs() : 0;
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		constexpr int delta = track ? 1 : -1;
		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<delta, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}
			if (region.log && old_perms != new_perms &&
			    (old_perms == Common::VirtualMemory::Mode::ReadWrite ||
			     new_perms == Common::VirtualMemory::Mode::ReadWrite)) [[unlikely]] {
				NoObs::NoteEdge(region.log[page_index], address / PAGE_SIZE,
				                new_perms == Common::VirtualMemory::Mode::ReadWrite, noobs_now);
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	// Applies `mode` to the pages of [begin, end) whose state asks for it (KYTY_PROTECT_COALESCE).
	void ApplyDeferred(uint64_t begin, uint64_t end, Common::VirtualMemory::Mode mode) noexcept {
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region      = FindRegion(chunk_begin);
			if (region != nullptr) {
				const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				SpinGuard  lock(region->lock);
				size_t     run = last;
				for (size_t page = first; page <= last; page++) {
					const bool wanted = page < last && region->pages[page].Perms() == mode;
					if (wanted && run == last) {
						run = page;
					} else if (!wanted && run != last) {
						Protect(region_base + run * PAGE_SIZE, (page - run) * PAGE_SIZE, mode);
						run = last;
					}
				}
				if (g_verify_deferred.load(std::memory_order_relaxed)) [[unlikely]] {
					VerifyDeferred(*region, region_base, first, last);
				}
			}
			chunk_begin = chunk_end;
		}
	}

	// KYTY_REPROTECT_BATCH=2 (region lock held): every page of [first, last) whose state asks for a
	// protection must not be writable on the host once the flush applied it.
	static void VerifyDeferred(const Region& region, uint64_t region_base, size_t first,
	                           size_t last) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (size_t page = first; page < last; page++) {
			const auto perms = region.pages[page].Perms();
			if (perms == Common::VirtualMemory::Mode::ReadWrite) {
				continue;
			}
			const auto               address = region_base + page * PAGE_SIZE;
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) == 0) {
				continue;
			}
			g_verify_checked.fetch_add(1, std::memory_order_relaxed);
			const auto protect  = info.Protect & 0xffu;
			const bool writable = protect == PAGE_READWRITE || protect == PAGE_WRITECOPY ||
			                      protect == PAGE_EXECUTE_READWRITE ||
			                      protect == PAGE_EXECUTE_WRITECOPY;
			const bool readable = protect != PAGE_NOACCESS;
			const bool bad = writable || (perms == Common::VirtualMemory::Mode::NoAccess && readable);
			if (bad && g_verify_mismatch.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr,
				             "PageManager: deferred protection mismatch at 0x%016" PRIx64
				             " (state %u, host 0x%08lx)\n",
				             address, static_cast<unsigned>(perms), info.Protect);
			}
		}
#else
		(void)region;
		(void)region_base;
		(void)first;
		(void)last;
#endif
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

template <bool track, bool is_read>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	const NoObs::DefaultSource source(NoObsSource::Texture);
	m_impl->UpdatePageWatchers<track, is_read>(vaddr, size);
}

uint32_t PageManager::Watchers(uint64_t vaddr) const noexcept {
	if (vaddr >= ADDRESS_SIZE) {
		return 0;
	}
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return 0;
	}
	SpinGuard   lock(region->lock);
	const auto& page = region->pages[(vaddr % REGION_SIZE) / PAGE_SIZE];
	return static_cast<uint32_t>(page.write_watchers) + page.access_watchers;
}

PageManager::NoObsTag::NoObsTag(NoObsSource source) noexcept: m_previous(NoObs::t_source) {
	NoObs::t_source = static_cast<uint8_t>(source);
}

PageManager::NoObsTag::~NoObsTag() {
	NoObs::t_source = m_previous;
}

bool PageManager::NoObsEnabled() noexcept {
	return NoObs::On();
}

void PageManager::NoObsEntry(uint64_t vaddr) const noexcept {
	if (!NoObs::On() || vaddr >= ADDRESS_SIZE) {
		return;
	}
	auto& stash    = NoObs::t_stash;
	stash.page     = vaddr / PAGE_SIZE;
	stash.entry_us = PerfStats::SteadyUs();
	stash.watchers = 0;
	stash.host     = NoObs::HostUnknown;
	auto* region   = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		stash.host = NoObs::QueryHost(Common::AlignDown(vaddr, PAGE_SIZE));
		return;
	}
	// Under the region lock the page state and its host protection agree, unless the protection
	// was left behind (or set outside the page manager).
	SpinGuard   lock(region->lock);
	const auto& page = region->pages[(vaddr % REGION_SIZE) / PAGE_SIZE];
	stash.watchers   = static_cast<uint32_t>(page.write_watchers) + page.access_watchers;
	if (stash.watchers == 0) {
		stash.host = NoObs::QueryHost(Common::AlignDown(vaddr, PAGE_SIZE));
	}
}

void PageManager::NoObsFault(uint64_t vaddr, bool untracked) const noexcept {
	using namespace NoObs;
	if (!On()) {
		return;
	}
	auto&      t     = g_totals;
	const auto page  = vaddr / PAGE_SIZE;
	auto&      stash = t_stash;
	const bool have  = stash.page == page;
	stash.page       = UINT64_MAX;
	const auto now   = PerfStats::SteadyUs();
	const auto lat   = have ? now - std::min(now, stash.entry_us) : 0;
	t.faults.fetch_add(1, std::memory_order_relaxed);
	if (have) {
		t.lat_all_us.fetch_add(lat, std::memory_order_relaxed);
		t.lat_all_n.fetch_add(1, std::memory_order_relaxed);
	}
	if (!untracked) {
		if (have && stash.watchers == 0) {
			t.entry0_watched.fetch_add(1, std::memory_order_relaxed);
		}
		return;
	}
	t.untracked.fetch_add(1, std::memory_order_relaxed);
	if (!have) {
		t.noentry.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	t.lat_unt_us.fetch_add(lat, std::memory_order_relaxed);
	t.lat[lat < 10 ? 0 : lat < 100 ? 1 : lat < 1000 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
	PageLog log;
	if (auto* region = m_impl->FindRegion(vaddr); region != nullptr && region->log) {
		SpinGuard lock(region->lock);
		log = region->log[(vaddr % REGION_SIZE) / PAGE_SIZE];
	}
	int cls = -1; // stale
	if (stash.watchers > 0) {
		cls = LostHandling;
	} else if (stash.host == HostWritable) {
		cls = LostDispatch;
	} else if (stash.host == HostUnknown) {
		t.unknown.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (cls >= 0) {
		const auto kind = log.release_us == 0 ? KindOther
		                  : log.release_gpu   ? KindGpu
		                  : log.release_tid == Tid() ? KindSame
		                                             : KindOther;
		t.lost[cls].fetch_add(1, std::memory_order_relaxed);
		t.rel[cls][SourceIndex(log.release_src)][kind].fetch_add(1, std::memory_order_relaxed);
		t.age[cls][AgeBucket(cls == LostHandling ? now : stash.entry_us, log.release_us)].fetch_add(
		    1, std::memory_order_relaxed);
	} else {
		t.stale.fetch_add(1, std::memory_order_relaxed);
		t.stale_src[SourceIndex(log.protect_src)].fetch_add(1, std::memory_order_relaxed);
		t.stale_age[AgeBucket(now, log.protect_us)].fetch_add(1, std::memory_order_relaxed);
		t.stale_last_release[SourceIndex(log.release_src)].fetch_add(1, std::memory_order_relaxed);
		if (stash.host == HostNoAccess) {
			t.stale_noaccess.fetch_add(1, std::memory_order_relaxed);
		}
	}
	auto& slot = g_slots[page % SLOTS];
	if (slot.page.exchange(page + 1, std::memory_order_relaxed) != 0) {
		t.reprot_over.fetch_add(1, std::memory_order_relaxed);
	}
	slot.us.store(now, std::memory_order_relaxed);
	t.reprot_in.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(g_region_mutex);
	auto&           r = g_regions[page * PAGE_SIZE / REGION_SIZE];
	r.faults++;
	if (cls < 0) {
		r.stale++;
	} else {
		r.lost[cls]++;
	}
	r.first = std::min(r.first, page);
	r.last  = std::max(r.last, page);
	r.pages.insert(page);
}

void PageManager::NoObsReport(double seconds) noexcept {
	using namespace NoObs;
	if (!On()) {
		return;
	}
	auto&              t    = g_totals;
	const auto         take = [](Counter& v) { return v.exchange(0, std::memory_order_relaxed); };
	static const char* src[SOURCES] = {"bo", "bf", "br", "bi", "tx", "hp", "na"};
	std::printf("PERF noobs %.1fs: faults=%" PRIu64 " untracked=%" PRIu64 " noentry=%" PRIu64
	            " entry0_watched=%" PRIu64 " unknown=%" PRIu64 " lost_handling=%" PRIu64
	            " lost_dispatch=%" PRIu64 " stale=%" PRIu64 " stale_noaccess=%" PRIu64,
	            seconds, take(t.faults), take(t.untracked), take(t.noentry), take(t.entry0_watched),
	            take(t.unknown), take(t.lost[LostHandling]), take(t.lost[LostDispatch]), take(t.stale),
	            take(t.stale_noaccess));
	std::printf(" lat_all=%" PRIu64 "/%" PRIu64 " lat_unt=%" PRIu64 " lat=", take(t.lat_all_us),
	            take(t.lat_all_n), take(t.lat_unt_us));
	for (uint32_t i = 0; i < LAT; i++) {
		std::printf("%s%" PRIu64, i == 0 ? "" : ",", take(t.lat[i]));
	}
	std::printf(" reprot_in=%" PRIu64 " reprot_over=%" PRIu64 " reprot=", take(t.reprot_in),
	            take(t.reprot_over));
	for (uint32_t i = 0; i < REPROT; i++) {
		std::printf("%s%" PRIu64, i == 0 ? "" : ",", take(t.reprot[i]));
	}
	std::printf("\n");
	// Last releaser of the lost pages, source:same/other/GPU thread; age none,after,<10us..>=10ms.
	for (uint32_t c = 0; c < LostCount; c++) {
		std::printf("PERF noobs_%s %.1fs: rel=", c == LostHandling ? "handling" : "dispatch", seconds);
		for (uint32_t s = 0; s < SOURCES; s++) {
			std::printf("%s%s:%" PRIu64 "/%" PRIu64 "/%" PRIu64, s == 0 ? "" : ",", src[s],
			            take(t.rel[c][s][KindSame]), take(t.rel[c][s][KindOther]), take(t.rel[c][s][KindGpu]));
		}
		std::printf(" age=");
		for (uint32_t i = 0; i < AGES; i++) {
			std::printf("%s%" PRIu64, i == 0 ? "" : ",", take(t.age[c][i]));
		}
		std::printf("\n");
	}
	std::printf("PERF noobs_stale %.1fs: protect_src=", seconds);
	for (uint32_t s = 0; s < SOURCES; s++) {
		std::printf("%s%s:%" PRIu64, s == 0 ? "" : ",", src[s], take(t.stale_src[s]));
	}
	std::printf(" last_release=");
	for (uint32_t s = 0; s < SOURCES; s++) {
		std::printf("%s%s:%" PRIu64, s == 0 ? "" : ",", src[s], take(t.stale_last_release[s]));
	}
	std::printf(" protect_age=");
	for (uint32_t i = 0; i < AGES; i++) {
		std::printf("%s%" PRIu64, i == 0 ? "" : ",", take(t.stale_age[i]));
	}
	std::printf("\n");
	std::vector<std::pair<uint64_t, RegionStat>> regions;
	{
		std::lock_guard lock(g_region_mutex);
		regions.reserve(g_regions.size());
		for (auto& [base, r]: g_regions) {
			regions.emplace_back(base, std::move(r));
		}
		g_regions.clear();
	}
	std::sort(regions.begin(), regions.end(),
	          [](const auto& a, const auto& b) { return a.second.faults > b.second.faults; });
	// base=faults/distinct pages/stale/lost handling/lost dispatch/span pages
	std::printf("PERF noobs_regions %.1fs: regions=%zu", seconds, regions.size());
	for (size_t i = 0; i < std::min<size_t>(regions.size(), 8); i++) {
		const auto& [base, r] = regions[i];
		std::printf(" 0x%" PRIx64 "=%" PRIu64 "/%zu/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64,
		            base * REGION_SIZE, r.faults, r.pages.size(), r.stale, r.lost[LostHandling],
		            r.lost[LostDispatch], r.last - r.first + 1);
	}
	std::printf("\n");
}

void PageManager::RestoreProtection(uint64_t vaddr, uint64_t size) noexcept {
	using Mode       = Common::VirtualMemory::Mode;
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
	uint64_t   run_begin = 0;
	Mode       run_mode  = Mode::ReadWrite;
	bool       in_run    = false;
	const auto flush     = [&](uint64_t at) {
        if (in_run) {
            Impl::ForEachOutsideStacks(run_begin, at - run_begin, run_mode, [](uint64_t b, uint64_t n, Mode m) {
                if (!Common::VirtualMemory::Protect(b, n, m)) {
                    Fatal("protection restore failed at 0x%016" PRIx64, b);
                }
            });
            in_run = false;
        }
	};
	for (auto address = begin; address < end; address += PAGE_SIZE) {
		const auto* region = m_impl->FindRegion(address);
		const auto  mode   = region != nullptr
		                         ? region->pages[(address % REGION_SIZE) / PAGE_SIZE].Perms()
		                         : Mode::ReadWrite;
		if (mode == Mode::ReadWrite) {
			flush(address);
			continue;
		}
		if (in_run && mode == run_mode) {
			continue;
		}
		flush(address);
		run_begin = address;
		run_mode  = mode;
		in_run    = true;
	}
	flush(end);
}

template void PageManager::UpdatePageWatchers<true, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<true, false>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, false>(uint64_t, uint64_t);

void PageManager::BeginDeferredProtect() noexcept {
	t_deferred.active = true;
}

PageManager::DeferCapture::DeferCapture() noexcept: m_previous(t_deferred.capture) {
	t_deferred.capture = true;
}

PageManager::DeferCapture::~DeferCapture() {
	t_deferred.capture = m_previous;
}

bool PageManager::DeferringProtect() noexcept {
	return t_deferred.active;
}

void PageManager::FlushDeferredProtect() noexcept {
	if (!g_deferred_any.load(std::memory_order_acquire)) {
		return;
	}
	constexpr uint64_t MaxGap = 16 * PAGE_SIZE;
	auto&              spans  = t_deferred.scratch;
	spans.clear();
	{
		SpinGuard lock(g_deferred_lock);
		std::swap(spans, g_deferred_spans);
		g_deferred_any.store(false, std::memory_order_relaxed);
	}
	// ApplyDeferred's protections are made at once.
	t_deferred.applying = true;
	std::sort(spans.begin(), spans.end(), [](const DeferredSpan& a, const DeferredSpan& b) {
		return a.impl != b.impl ? a.impl < b.impl : a.begin < b.begin;
	});
	for (size_t i = 0; i < spans.size();) {
		auto   merged = spans[i];
		size_t j      = i + 1;
		for (; j < spans.size(); j++) {
			const auto& next = spans[j];
			if (next.impl != merged.impl || next.mode != merged.mode ||
			    next.begin > merged.end + MaxGap) {
				break;
			}
			merged.end = std::max(merged.end, next.end);
		}
		static_cast<Impl*>(merged.impl)->ApplyDeferred(merged.begin, merged.end, merged.mode);
		i = j;
	}
	spans.clear();
	t_deferred.applying = false;
}

void PageManager::SetDeferredVerify(bool on) noexcept {
	g_verify_deferred.store(on, std::memory_order_relaxed);
}

PageManager::DeferredVerifyStats PageManager::GetDeferredVerifyStats() noexcept {
	return {g_verify_checked.load(std::memory_order_relaxed),
	        g_verify_mismatch.load(std::memory_order_relaxed)};
}

void PageManager::EndDeferredProtect() noexcept {
	FlushDeferredProtect();
	t_deferred.active = false;
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	const NoObs::DefaultSource source(NoObsSource::BufOther);
	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	const auto first       = start_range.first;
	const auto last        = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
