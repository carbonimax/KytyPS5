#include "graphics/host_gpu/guestStacks.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics::GuestStacks {
namespace {

struct Entry {
	uint64_t begin = 0;
	uint64_t end   = 0;
	char     owner[48] {};
};

inline constexpr uint64_t REGION_SHIFT = 22u; // 4 MiB, the tracker region size
inline constexpr uint64_t REGION_COUNT = (uint64_t {1} << 44u) >> REGION_SHIFT;

struct State {
	std::shared_mutex                         mutex;
	std::vector<Entry>                        entries;
	std::unique_ptr<std::atomic<uint16_t>[]> region_refs;
	std::atomic<bool>                         any {false};
	std::atomic<uint64_t>                     hits_read {0};
	std::atomic<uint64_t>                     hits_noaccess {0};
	std::atomic<uint64_t>                     reports {0};
	std::atomic<uint64_t>                     registered {0};
	std::atomic<uint64_t>                     invalidated {0};
	std::atomic<uint64_t>                     at_register {0};
};

State& S() {
	static State state;
	return state;
}

} // namespace

bool InvalidateEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STACK_INVALIDATE");
		return value == nullptr || std::atoi(value) != 0;
	}();
	return enabled;
}

static void AdjustRegions(uint64_t begin, uint64_t end, int delta) {
	auto& s = S();
	for (auto region = begin >> REGION_SHIFT; region <= ((end - 1) >> REGION_SHIFT) && region < REGION_COUNT;
	     region++) {
		s.region_refs[region].fetch_add(static_cast<uint16_t>(delta), std::memory_order_relaxed);
	}
}

void Register(uint64_t begin, uint64_t size, const char* owner) {
	if (size == 0 || begin + size > (uint64_t {1} << 44u)) {
		return;
	}
	auto&             s = S();
	std::unique_lock lock(s.mutex);
	if (!s.region_refs) {
		s.region_refs = std::make_unique<std::atomic<uint16_t>[]>(REGION_COUNT);
	}
	auto it = std::find_if(s.entries.begin(), s.entries.end(),
	                       [begin](const Entry& e) { return e.begin == begin; });
	if (it != s.entries.end()) {
		AdjustRegions(it->begin, it->end, -1);
		s.entries.erase(it);
	}
	Entry entry {begin, begin + size, {}};
	std::snprintf(entry.owner, sizeof(entry.owner), "%s", owner != nullptr ? owner : "");
	s.entries.push_back(entry);
	AdjustRegions(entry.begin, entry.end, 1);
	s.any.store(true, std::memory_order_release);
	s.registered.fetch_add(1, std::memory_order_relaxed);
	lock.unlock();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Pages already protected when the stack is handed out: watches or GPU-dirty state left by
	// earlier users of the memory (the guest recycled it from textures or buffers).
	uint64_t read_only = 0, no_access = 0, first = 0;
	for (auto address = begin & ~uint64_t {0xfff}; address < begin + size;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<LPCVOID>(address), &info, sizeof(info)) == 0) {
			break;
		}
		const auto region_end = std::min(begin + size, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
		const auto pages      = (region_end - address + 0xfff) / 0x1000;
		const auto protect    = info.Protect & 0xffu;
		if (info.State == MEM_COMMIT && (protect == PAGE_READONLY || protect == PAGE_NOACCESS)) {
			(protect == PAGE_READONLY ? read_only : no_access) += pages;
			if (first == 0) first = address;
		}
		address = region_end;
	}
	if (read_only + no_access != 0) {
		s.at_register.fetch_add(1, std::memory_order_relaxed);
		const auto r = s.reports.fetch_add(1, std::memory_order_relaxed);
		if (r < 32 || (r & 0xfffu) == 0) {
			std::printf("STACK-PROTECT at register: %s 0x%" PRIx64 "..0x%" PRIx64 " has %" PRIu64
			            " read-only and %" PRIu64 " no-access pages (first 0x%" PRIx64 ", depth 0x%" PRIx64 ")\n",
			            entry.owner, begin, begin + size, read_only, no_access, first, begin + size - first);
			std::fflush(stdout);
		}
	}
#endif
}

void NoteInvalidated() noexcept {
	S().invalidated.fetch_add(1, std::memory_order_relaxed);
}

void Unregister(uint64_t begin) {
	auto&             s = S();
	std::unique_lock lock(s.mutex);
	auto it = std::find_if(s.entries.begin(), s.entries.end(),
	                       [begin](const Entry& e) { return e.begin == begin; });
	if (it != s.entries.end()) {
		AdjustRegions(it->begin, it->end, -1);
		s.entries.erase(it);
	}
}

bool MaybeHit(uint64_t vaddr, uint64_t size) noexcept {
	auto& s = S();
	if (!s.any.load(std::memory_order_acquire) || size == 0) {
		return false;
	}
	const auto first = vaddr >> REGION_SHIFT;
	const auto last  = (vaddr + size - 1) >> REGION_SHIFT;
	for (auto region = first; region <= last && region < REGION_COUNT; region++) {
		if (s.region_refs[region].load(std::memory_order_relaxed) != 0) {
			return true;
		}
	}
	return false;
}

static bool Find(uint64_t vaddr, Entry* out) {
	auto&               s = S();
	std::shared_lock lock(s.mutex);
	for (const auto& e: s.entries) {
		if (vaddr >= e.begin && vaddr < e.end) {
			*out = e;
			return true;
		}
	}
	return false;
}

static void PrintBacktrace() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void*      frames[12] {};
	const auto count = RtlCaptureStackBackTrace(2, 12, frames, nullptr);
	std::printf("  bt:");
	for (USHORT i = 0; i < count; i++) {
		HMODULE module = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       static_cast<LPCSTR>(frames[i]), &module) &&
		    module == GetModuleHandleA(nullptr)) {
			std::printf(" exe+0x%" PRIx64, reinterpret_cast<uint64_t>(frames[i]) - reinterpret_cast<uint64_t>(module));
		} else {
			std::printf(" %p", frames[i]);
		}
	}
	std::printf("\n");
#endif
}

void OnProtect(uint64_t vaddr, uint64_t size, bool no_access) {
	auto&    s = S();
	Entry    hit {};
	uint64_t overlap_begin = 0, overlap_end = 0;
	{
		std::shared_lock lock(s.mutex);
		for (const auto& e: s.entries) {
			if (vaddr < e.end && vaddr + size > e.begin) {
				hit           = e;
				overlap_begin = std::max(vaddr, e.begin);
				overlap_end   = std::min(vaddr + size, e.end);
				break;
			}
		}
	}
	if (hit.end == 0) {
		return;
	}
	const auto n = (no_access ? s.hits_noaccess : s.hits_read).fetch_add(1, std::memory_order_relaxed) + 1;
	const auto r = s.reports.fetch_add(1, std::memory_order_relaxed);
	if (r < 32 || (r & 0xfffu) == 0) {
		std::printf("STACK-PROTECT %s #%" PRIu64 " pages 0x%" PRIx64 "..0x%" PRIx64 " in stack %s 0x%" PRIx64
		            "..0x%" PRIx64 " (depth 0x%" PRIx64 "..0x%" PRIx64 " below top), total read=%" PRIu64
		            " noaccess=%" PRIu64 "\n",
		            no_access ? "noaccess" : "read-only", n, overlap_begin, overlap_end, hit.owner, hit.begin, hit.end,
		            hit.end - overlap_end, hit.end - overlap_begin, s.hits_read.load(std::memory_order_relaxed),
		            s.hits_noaccess.load(std::memory_order_relaxed));
		PrintBacktrace();
		std::fflush(stdout);
	}
}

// Guest fault handler: the registered range holding `rsp` and the host protection of its pages
// around the stack pointer (the exception frame needs the pages below it writable).
void Describe(uint64_t rsp) {
	Entry e {};
	if (Find(rsp, &e)) {
		std::printf("stack: %s 0x%" PRIx64 "..0x%" PRIx64 ", rsp depth 0x%" PRIx64 "\n", e.owner, e.begin, e.end,
		            e.end - rsp);
	} else {
		std::printf("stack: rsp 0x%" PRIx64 " not in a registered guest stack (%zu registered)\n", rsp,
		            S().entries.size());
	}
	std::printf("stack protections: read-only %" PRIu64 ", no-access %" PRIu64 " (registered %" PRIu64
	            ", invalidated %" PRIu64 ")\n",
	            S().hits_read.load(), S().hits_noaccess.load(), S().registered.load(), S().invalidated.load());
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto page = rsp & ~uint64_t {0xfff};
	for (auto address = page - 0x4000; address <= page + 0x1000; address += 0x1000) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<LPCVOID>(address), &info, sizeof(info)) != 0) {
			std::printf("  page 0x%" PRIx64 ": state 0x%lx protect 0x%lx\n", address, info.State, info.Protect);
		}
	}
#endif
}

void Report() {
	auto&         s = S();
	static uint64_t last_registered = 0, last_hits = ~uint64_t {0};
	const auto      registered = s.registered.load(std::memory_order_relaxed);
	const auto      hits       = s.hits_read.load(std::memory_order_relaxed) +
	                  s.hits_noaccess.load(std::memory_order_relaxed) + s.at_register.load(std::memory_order_relaxed);
	if (registered == last_registered && hits == last_hits) {
		return;
	}
	last_registered = registered;
	last_hits       = hits;
	size_t live     = 0;
	{
		std::shared_lock lock(s.mutex);
		live = s.entries.size();
	}
	std::printf("PERF stacks: live=%zu registered=%" PRIu64 " invalidated=%" PRIu64 " protect read-only=%" PRIu64
	            " no-access=%" PRIu64 " protected at register=%" PRIu64 "\n",
	            live, registered, s.invalidated.load(std::memory_order_relaxed),
	            s.hits_read.load(std::memory_order_relaxed), s.hits_noaccess.load(std::memory_order_relaxed),
	            s.at_register.load(std::memory_order_relaxed));
}

} // namespace Libs::Graphics::GuestStacks
