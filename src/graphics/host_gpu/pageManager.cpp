#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/guestStacks.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
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

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

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
					if (access_watchers != 0) {
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
					if (write_watchers == 0x7f) {
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
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
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

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
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

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

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
