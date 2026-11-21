// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                   MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

template <typename Regions, typename Region, typename Gap>
static void ForEachSparseRange(const Regions& regions, uint64_t vaddr, uint64_t size,
                               Region&& region, Gap&& gap) {
	const auto end     = vaddr + size;
	auto       mapping = regions.lower_bound(vaddr);
	if (mapping != regions.begin()) {
		auto previous = std::prev(mapping);
		if (previous->second.vaddr + previous->second.size > vaddr) {
			mapping = previous;
		}
	}

	auto cursor = vaddr;
	for (; mapping != regions.end() && mapping->second.vaddr < end; ++mapping) {
		const auto& entry = mapping->second;
		const auto  start = std::max(vaddr, entry.vaddr);
		const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
		if (cursor < start) {
			gap(cursor, start - cursor);
		}
		region(entry, start, bytes);
		cursor = start + bytes;
	}
	if (cursor < end) {
		gap(cursor, end - cursor);
	}
}

template <typename Regions, typename Source, typename Gap>
static void CopySparseRange(const Regions& regions, uint64_t vaddr, void* data, uint64_t size,
                            Source&& source, Gap&& gap) {
	auto* destination = static_cast<uint8_t*>(data);
	ForEachSparseRange(
	    regions, vaddr, size,
	    [&](const auto& entry, uint64_t start, uint64_t bytes) {
		    std::memcpy(destination + (start - vaddr), source(entry, start), bytes);
	    },
	    [&](uint64_t start, uint64_t bytes) { gap(start, destination + (start - vaddr), bytes); });
}

// Mapping of a thread's last backing read (GuestBackingStore::TryReadBackingCached).
struct BackingReadCache {
	const void* store          = nullptr;
	uint64_t    generation     = 0;
	uint64_t    vaddr          = 0;
	uint64_t    size           = 0;
	uint64_t    backing_offset = 0;
};
static thread_local BackingReadCache t_backing_read_cache;

// Backing mappings of a thread's recent sparse copies (GuestAddressSpace::CopySparseMemory). Kept
// apart from t_backing_read_cache so the CP thread's SRT walks and its buffer uploads do not evict
// each other's mapping.
struct SparseCopyCache {
	static constexpr uint32_t Ways = 4;
	BackingReadCache          entries[Ways];
	uint64_t                  map_bumps[Ways] {}; // m_map_bumps at the fill (mode 2 stale-miss cause)
	uint32_t                  next = 0;
};
static thread_local SparseCopyCache t_sparse_copy_cache;

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		AddressRangeUnavailable,
	};

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr) {
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory: 0x%08" PRIx32
			     ". Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     m_size / (1024U * 1024U), static_cast<uint32_t>(GetLastError()));
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			EXIT("direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(GetLastError()));
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#if defined(__APPLE__)
		{
			char shm_name[64];
			snprintf(shm_name, sizeof(shm_name), "/kyty-dm-%d-%llx", static_cast<int>(getpid()),
			         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
			m_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (m_fd >= 0) {
				shm_unlink(shm_name);
			}
		}
#else
		m_fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
#endif
		if (m_fd < 0) {
			EXIT("direct-memory backing: shared fd creation failed: %d\n", errno);
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			EXIT("direct-memory backing: ftruncate to %" PRIu64 " MB failed: %d\n",
			     m_size / (1024U * 1024U), errno);
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			EXIT("direct-memory backing: writable alias failed: %d\n", errno);
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#endif
	}

	~GuestBackingStore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [_, entry]: m_maps) {
			UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size);
		}
		UnmapViewOfFile(m_backing_base);
		CloseHandle(m_handle);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		for (const auto& [_, entry]: m_maps) {
			EXIT_IF(
			    !UnmapViewPreservePlaceholder(reinterpret_cast<void*>(entry.vaddr), entry.size));
		}
		munmap(m_backing_base, m_size);
		close(m_fd);
#endif
	}

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			default: return "unknown";
		}
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	void ZeroRange(uint64_t backing_offset, uint64_t size) {
		std::lock_guard lock(m_mutex);
		BumpMapGeneration(false);
		std::memset(m_backing_base + backing_offset, 0, size);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		g_test_backing_bytes_zeroed += size;
#endif
	}

	void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		// Skip unbacked gaps, mirroring sparse reads that zero-fill them.
		std::lock_guard lock(m_mutex);
		BumpMapGeneration(false);
		const auto*     source = static_cast<const uint8_t*>(data);
		ForEachSparseRange(
		    m_maps, vaddr, size,
		    [&](const auto& entry, uint64_t start, uint64_t bytes) {
			    std::memcpy(m_backing_base + (entry.backing_offset + start - entry.vaddr),
			                source + (start - vaddr), bytes);
		    },
		    [](uint64_t, uint64_t) {});
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		if (TryReadBackingCached(vaddr, data, size)) {
			return true;
		}

		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		const auto end   = vaddr + size;
		auto       first = m_maps.upper_bound(vaddr);
		if (first == m_maps.begin()) {
			return false;
		}
		--first;
		auto current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			if (mapping == m_maps.end() || current < mapping->second.vaddr ||
			    current >= mapping->second.vaddr + mapping->second.size) {
				return false;
			}
			current = std::min(end, mapping->second.vaddr + mapping->second.size);
		}

		auto* destination = static_cast<uint8_t*>(data);
		current = vaddr;
		for (auto mapping = first; current < end; ++mapping) {
			const auto& entry  = mapping->second;
			const auto  bytes  = std::min(end, entry.vaddr + entry.size) - current;
			const auto  offset = entry.backing_offset + current - entry.vaddr;
			std::memcpy(destination + (current - vaddr), m_backing_base + offset, bytes);
			current += bytes;
		}
		if (BackingReadCacheEnabled()) {
			// Writers bump the generation under the exclusive lock: it is stable here.
			auto& cache          = t_backing_read_cache;
			cache.store          = this;
			cache.generation     = m_map_generation.load(std::memory_order_relaxed);
			cache.vaddr          = first->second.vaddr;
			cache.size           = first->second.size;
			cache.backing_offset = first->second.backing_offset;
		}
		return true;
	}

	// KYTY_MEM_IMPORT: backing offset of [vaddr, vaddr + size) when the whole range is mapped from one
	// contiguous run of the backing (one view, or adjacent views of adjacent backing ranges).
	bool ContiguousOffset(uint64_t vaddr, uint64_t size, uint64_t* offset) {
		if (m_backing_base == nullptr || offset == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		auto it = m_maps.upper_bound(vaddr);
		if (it == m_maps.begin()) {
			return false;
		}
		--it;
		if (vaddr >= it->second.vaddr + it->second.size) {
			return false;
		}
		const auto base    = it->second.backing_offset + (vaddr - it->second.vaddr);
		const auto end     = vaddr + size;
		auto       current = vaddr;
		for (;;) {
			const auto& entry = it->second;
			if (current < entry.vaddr || current >= entry.vaddr + entry.size ||
			    entry.backing_offset + (current - entry.vaddr) != base + (current - vaddr)) {
				return false;
			}
			current = std::min(end, entry.vaddr + entry.size);
			if (current >= end) {
				break;
			}
			if (++it == m_maps.end()) {
				return false;
			}
		}
		if (base >= m_size || size > m_size - base) {
			return false;
		}
		*offset = base;
		return true;
	}

	template <typename Gap>
	void ReadSparseBacking(uint64_t vaddr, void* data, uint64_t size, Gap&& gap) {
		std::shared_lock lock(m_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		CopySparseRange(m_maps, vaddr, data, size,
		                [this](const auto& entry, uint64_t start) {
			                return m_backing_base + (entry.backing_offset + start - entry.vaddr);
		                },
		                gap);
		if (SparseCopyCacheMode() != 0) {
			// Remember the mapping when it holds the whole range. Writers bump the generation
			// under the exclusive lock, so it is stable here.
			auto next = m_maps.upper_bound(vaddr);
			if (next != m_maps.begin()) {
				const auto& entry = std::prev(next)->second;
				if (vaddr - entry.vaddr < entry.size && size <= entry.size - (vaddr - entry.vaddr)) {
					auto&             cache = t_sparse_copy_cache;
					BackingReadCache* found = nullptr;
					for (auto& slot: cache.entries) {
						if (slot.store == this && slot.vaddr == entry.vaddr) {
							found = &slot;
						}
					}
					auto& slot =
					    found != nullptr ? *found : cache.entries[cache.next++ % SparseCopyCache::Ways];
					slot.store          = this;
					slot.generation     = m_map_generation.load(std::memory_order_relaxed);
					slot.vaddr          = entry.vaddr;
					slot.size           = entry.size;
					slot.backing_offset = entry.backing_offset;
					cache.map_bumps[&slot - cache.entries] = m_map_bumps.load(std::memory_order_relaxed);
				}
			}
		}
	}

	// KYTY_SPARSE_COPY_CACHE: 0 always takes the locks, 1 (default) copies a range that lies inside
	// one recently used backing mapping without any lock (TryCopyCached), 2 also re-checks every
	// lock-free copy's translation under the locks and exits on a difference. Read once, at the
	// first copy. With 1 or 2, a copy that misses the cache takes one address-space lock, not two.
	static int SparseCopyCacheMode() {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_sparse_copy_cache_mode >= 0) {
			return g_test_sparse_copy_cache_mode;
		}
#endif
		static const int mode = [] {
			const char* value = std::getenv("KYTY_SPARSE_COPY_CACHE");
			const int   m     = value == nullptr ? 1 : static_cast<int>(std::strtol(value, nullptr, 10));
			if (value != nullptr) {
				std::printf("KYTY_SPARSE_COPY_CACHE=%d: %s sparse guest copies\n", m,
				            m != 0 ? "lock-free" : "locked");
				std::fflush(stdout);
			}
			return m;
		}();
		return mode;
	}

	// Lock-free copy for GuestAddressSpace::CopySparseMemory. Same seqlock as
	// TryReadBackingCached, over the thread's sparse-copy mappings. On success, *generation and
	// *backing_offset tell what the copy was validated against (for the mode 2 check). In mode 2,
	// a miss on a slot that covers the range sets *stale: 1 if only byte writes (WriteBacking,
	// ZeroRange) bumped the generation since its fill, 2 if a mapping change did.
	bool TryCopyCached(uint64_t vaddr, void* data, uint64_t size, uint64_t* generation,
	                   uint64_t* backing_offset, int* stale) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			return false;
		}
#endif
		const auto& cache       = t_sparse_copy_cache;
		const auto  stale_cause = [&](uint32_t way) {
			if (SparseCopyCacheMode() == 2) {
				*stale = m_map_bumps.load(std::memory_order_relaxed) == cache.map_bumps[way] ? 1 : 2;
			}
		};
		for (uint32_t way = 0; way < SparseCopyCache::Ways; ++way) {
			const auto& slot = cache.entries[way];
			if (slot.store != this || vaddr < slot.vaddr || size > slot.size ||
			    vaddr - slot.vaddr > slot.size - size) {
				continue;
			}
			const auto current = m_map_generation.load(std::memory_order_acquire);
			const auto offset  = slot.backing_offset + (vaddr - slot.vaddr);
			// The slot was a store mapping at this generation, so the offset is inside the store.
			if (current != slot.generation) {
				stale_cause(way);
				continue;
			}
			std::memcpy(data, m_backing_base + offset, size);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (m_map_generation.load(std::memory_order_relaxed) != current) {
				stale_cause(way);
				return false;
			}
			*generation     = current;
			*backing_offset = offset;
			return true;
		}
		return false;
	}

	// Mode 2 check of a lock-free copy: 1 if the mapping still translates the range to
	// backing_offset, 0 if not, 2 if a writer ran since (nothing left to check: the copy was
	// validated before it).
	int CheckCopyTranslation(uint64_t vaddr, uint64_t size, uint64_t generation,
	                         uint64_t backing_offset) {
		std::shared_lock lock(m_mutex);
		if (m_map_generation.load(std::memory_order_relaxed) != generation) {
			return 2;
		}
		auto next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return 0;
		}
		const auto& entry = std::prev(next)->second;
		return vaddr - entry.vaddr < entry.size && size <= entry.size - (vaddr - entry.vaddr) &&
		               entry.backing_offset + (vaddr - entry.vaddr) == backing_offset
		           ? 1
		           : 0;
	}

	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		BumpMapGeneration(true);
#else
		std::lock_guard lock(m_mutex);
		BumpMapGeneration(true);
		if (!MapView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
		RemoveMappingsUnlocked(vaddr, size);
#endif
		m_maps.emplace(vaddr, MapEntry {vaddr, size, backing_offset, mode});
		return true;
	}

	bool Unmap(uint64_t vaddr, uint64_t size) {
		const auto            end = vaddr + size;
		std::vector<MapEntry> targets;
		{
			std::lock_guard lock(m_mutex);
			BumpMapGeneration(true);
			auto            current = vaddr;
			while (current < end) {
				const auto* entry = FindContainingUnlocked(current);
				if (entry == nullptr) {
					return false;
				}
				const auto part_size = std::min(end, entry->vaddr + entry->size) - current;
				targets.push_back({current, part_size});
				current += part_size;
			}
		}

		// Collecting the views validated the span; failing after it is an emulator fault.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& entry: targets) {
			UnmapOne(entry.vaddr, entry.size);
		}
		// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
		// records this contiguous guest span as one free range, so normalize fragmented backing
		// views into one placeholder before publishing it there.
		EXIT_IF(targets.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size));

		std::lock_guard lock(m_mutex);
		BumpMapGeneration(true);
		RemoveMappingsUnlocked(vaddr, size);
#endif
		return true;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void UnmapOne(uint64_t vaddr, uint64_t size) {
		MapEntry old {};
		{
			std::lock_guard lock(m_mutex);
			BumpMapGeneration(true);
			old = m_maps.extract(std::prev(m_maps.upper_bound(vaddr))).mapped();
		}
		const uint64_t left_size  = vaddr - old.vaddr;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = old.vaddr + old.size - right_addr;
		// Re-mapping resets the survivors to the view's mode, so the protections guest mprotect and
		// write tracking set on them are re-applied afterwards.
		std::vector<MEMORY_BASIC_INFORMATION> protections;
		QueryProtections(old.vaddr, left_size, old.mode, &protections);
		QueryProtections(right_addr, right_size, old.mode, &protections);

		// A partial unmap remaps the surviving parts: guest accesses wait for them
		// (WaitBackingRemap) and their GPU watch protections are restored.
		const bool partial = left_size != 0 || right_size != 0;
		if (partial) {
			BeginBackingRemap(old.vaddr, old.size);
		}
		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(old.vaddr), old.size));
		EXIT_IF(left_size != 0 && !SplitPlaceholder(old.vaddr, left_size));
		EXIT_IF(right_size != 0 && !SplitPlaceholder(vaddr, size));
		EXIT_IF(left_size != 0 && !MapFixed(old.vaddr, left_size, old.backing_offset, old.mode));
		EXIT_IF(right_size != 0 &&
		        !MapFixed(right_addr, right_size, old.backing_offset + (right_addr - old.vaddr),
		                  old.mode));
		for (const auto& info: protections) {
			EXIT_IF(!ApplyProtection(info.BaseAddress, info.RegionSize, info.Protect));
		}
		if (partial) {
			RestoreBackingProtection(old.vaddr, left_size);
			RestoreBackingProtection(right_addr, right_size);
			EndBackingRemap();
		}
	}
#endif

private:
	struct MapEntry {
		uint64_t            vaddr          = 0;
		uint64_t            size           = 0;
		uint64_t            backing_offset = 0;
		VirtualMemory::Mode mode           = VirtualMemory::Mode::NoAccess;
	};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	void RemoveMappingsUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end = vaddr + size;
		auto       it = m_maps.upper_bound(vaddr);
		if (it != m_maps.begin()) {
			--it;
		}
		while (it != m_maps.end() && it->first < end) {
			const auto old     = it->second;
			const auto old_end = old.vaddr + old.size;
			if (old_end <= vaddr) {
				++it;
				continue;
			}
			it = m_maps.erase(it);
			if (old.vaddr < vaddr) {
				m_maps.emplace(old.vaddr,
				               MapEntry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.mode});
			}
			if (end < old_end) {
				m_maps.emplace(end, MapEntry {end, old_end - end,
				                             old.backing_offset + end - old.vaddr, old.mode});
			}
		}
	}
#endif

	const MapEntry* FindContainingUnlocked(uint64_t vaddr) const {
		auto next = m_maps.upper_bound(vaddr);
		if (next == m_maps.begin()) {
			return nullptr;
		}
		const auto& entry = std::prev(next)->second;
		return vaddr < entry.vaddr + entry.size ? &entry : nullptr;
	}

	static void SetFailure(FailureReason* out, FailureReason reason) {
		if (out != nullptr) {
			*out = reason;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}

	// Collects the page runs of the span whose protection differs from the view's mode.
	static void QueryProtections(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
	                             std::vector<MEMORY_BASIC_INFORMATION>* out) {
		const auto end = vaddr + size;
		for (auto addr = vaddr; addr < end;) {
			MEMORY_BASIC_INFORMATION info {};
			EXIT_IF(VirtualQuery(reinterpret_cast<void*>(addr), &info, sizeof(info)) == 0);
			info.RegionSize = std::min<SIZE_T>(info.RegionSize, end - addr);
			if (info.Protect != GetProtect(mode)) {
				out->push_back(info);
			}
			addr += info.RegionSize;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, DWORD protect) {
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, protect, &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		if (VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0) {
			return true;
		}
		const auto error = static_cast<uint32_t>(GetLastError());
		LOGF_COLOR(Log::Color::Red,
		           "\t guest placeholder split failed: base=0x%016" PRIx64 " size=0x%016" PRIx64
		           " error=0x%08" PRIx32 "\n",
		           vaddr, size, error);
		return false;
	}

	bool ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                            VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr =
		    MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr),
		                   backing_offset, size, MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return false;
		}

		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, PAGE_NOACCESS)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return false;
		}
		return true;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static int GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PROT_READ;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::Execute: return PROT_EXEC;
			case VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PROT_NONE;
		}
	}

	bool MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	             FailureReason* failure_reason) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
		if (mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode), MAP_SHARED | MAP_FIXED,
		         m_fd, static_cast<off_t>(backing_offset)) == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return false;
		}
		return true;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		return mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) !=
		       MAP_FAILED;
	}

	int m_fd = -1;
#endif

	// KYTY_BACKING_READ_CACHE=0 disables. Backing reads (SRT descriptor walks: ~0.4 M block reads/s
	// in one title) took the shared lock and searched the mapping tree each time. Each thread now
	// remembers the mapping of its last read with the map generation of that time. A read inside
	// that mapping copies from the backing view without the lock when no writer started since
	// (seqlock: the generation is read before and after the copy). The backing view itself stays
	// mapped for the store's lifetime, so a copy racing a remap is harmless and is retried locked.
	static bool BackingReadCacheEnabled() {
		static std::atomic<int64_t> enabled_once {Common::OnceUnset};
		const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
			const char* value = std::getenv("KYTY_BACKING_READ_CACHE");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}));
		return enabled;
	}

	// The fence keeps the writer's later stores (mapping changes, ZeroRange/WriteBacking bytes)
	// after the bump for lock-free readers on weakly ordered hosts (seqlock writer side).
	// maps: an m_maps change (false: a locked byte write). Counted per kind with the cache on.
	void BumpMapGeneration(bool maps) {
		m_map_generation.fetch_add(1, std::memory_order_release);
		std::atomic_thread_fence(std::memory_order_release);
		if (SparseCopyCacheMode() != 0) {
			(maps ? m_map_bumps : m_byte_bumps).fetch_add(1, std::memory_order_relaxed);
		}
	}

public:
	void BumpCounts(uint64_t* maps, uint64_t* bytes) const {
		*maps  = m_map_bumps.load(std::memory_order_relaxed);
		*bytes = m_byte_bumps.load(std::memory_order_relaxed);
	}

private:

	bool TryReadBackingCached(uint64_t vaddr, void* data, uint64_t size) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			return false;
		}
#endif
		const auto& cache = t_backing_read_cache;
		if (cache.store != this || vaddr < cache.vaddr || size > cache.size ||
		    vaddr - cache.vaddr > cache.size - size) {
			return false;
		}
		const auto generation = m_map_generation.load(std::memory_order_acquire);
		if (generation != cache.generation) {
			return false;
		}
		const auto offset = cache.backing_offset + (vaddr - cache.vaddr);
		std::memcpy(data, m_backing_base + offset, size);
		std::atomic_thread_fence(std::memory_order_acquire);
		return m_map_generation.load(std::memory_order_relaxed) == generation;
	}

public:
	// KYTY_MEMTRACK_PROF: bytes and count of the guest views, and backing bytes mapped by more
	// than one view (the whole-backing writable alias is not counted).
	void ViewStats(uint64_t* bytes, uint64_t* views, uint64_t* aliased) {
		std::vector<std::pair<uint64_t, int>> events;
		{
			std::shared_lock lock(m_mutex);
			for (const auto& [vaddr, entry]: m_maps) {
				(void)vaddr;
				*bytes += entry.size;
				++*views;
				events.emplace_back(entry.backing_offset, 1);
				events.emplace_back(entry.backing_offset + entry.size, -1);
			}
		}
		std::sort(events.begin(), events.end());
		int      depth = 0;
		uint64_t at    = 0;
		for (const auto& [position, delta]: events) {
			if (depth >= 2) {
				*aliased += position - at;
			}
			depth += delta;
			at = position;
		}
	}

private:
	uint8_t*                     m_backing_base = nullptr;
	uint64_t                     m_size         = 0;
	std::shared_mutex            m_mutex;
	std::map<uint64_t, MapEntry> m_maps;
	std::atomic<uint64_t>        m_map_generation {0};
	std::atomic<uint64_t>        m_map_bumps {0};  // KYTY_SPARSE_COPY_CACHE != 0 only
	std::atomic<uint64_t>        m_byte_bumps {0}; // KYTY_SPARSE_COPY_CACHE != 0 only
};

class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
	}

	~GuestAddressSpace() {
		delete m_backing;
		m_backing = nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [start, size]: m_owned) {
			const auto end     = start + size;
			auto       current = start;
			while (current < end) {
				MEMORY_BASIC_INFORMATION info {};
				EXIT_IF(VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) ==
				        0);
				const auto region_start = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end   = region_start + static_cast<uint64_t>(info.RegionSize);
				if (info.State == MEM_COMMIT) {
					EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_start), info.RegionSize,
					                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_RESERVE) {
					EXIT_IF(VirtualFree(info.AllocationBase, 0, MEM_RELEASE) == 0);
				}
				current = region_end;
			}
		}
#else
		for (const auto& [start, size]: m_owned) {
			munmap(reinterpret_cast<void*>(start), size);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestAddressSpace);

	[[nodiscard]] uint64_t GetGranularity() const { return m_granularity; }
	[[nodiscard]] uint64_t GetBackingBase() const { return m_backing->GetBackingBase(); }
	[[nodiscard]] uint64_t GetBackingSize() const { return m_backing->GetBackingSize(); }

	void ZeroBacking(uint64_t backing_offset, uint64_t size) {
		m_backing->ZeroRange(backing_offset, size);
	}

	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return OwnsUnlocked(vaddr, size);
	}

	// m_owned is only written by ReserveGuestRegions, from the constructor: it needs no lock.
	[[nodiscard]] bool OwnsUnlocked(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

	// Nothing (all zero) with KYTY_SPARSE_COPY_CACHE=0.
	[[nodiscard]] SparseCopyStats GetSparseCopyStats() const {
		SparseCopyStats stats {};
		stats.mode        = GuestBackingStore::SparseCopyCacheMode();
		stats.checked     = m_sparse_checked.load(std::memory_order_relaxed);
		stats.locked      = m_sparse_locked.load(std::memory_order_relaxed);
		stats.stale_bytes = m_sparse_stale_bytes.load(std::memory_order_relaxed);
		stats.stale_maps  = m_sparse_stale_maps.load(std::memory_order_relaxed);
		m_backing->BumpCounts(&stats.map_bumps, &stats.byte_bumps);
		return stats;
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}
#endif

	// Writes the backed bytes of a range that holds no private mappings.
	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		if (OverlapsPrivateUnlocked(vaddr, size)) {
			return false;
		}
		m_backing->WriteBacking(vaddr, data, size);
		return true;
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}

	// KYTY_MEM_IMPORT: see GuestBackingStore::ContiguousOffset; never for ranges with private mappings.
	bool BackingOffsetOf(uint64_t vaddr, uint64_t size, uint64_t* offset) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		return !OverlapsPrivateUnlocked(vaddr, size) && m_backing->ContiguousOffset(vaddr, size, offset);
	}

	// KYTY_MEMTRACK_PROF: 0 = view of the backing, 1 = private mapping, 2 = unknown (or the
	// address-space lock is held exclusively: called from fault handlers, never waits).
	int MappingKindOf(uint64_t vaddr) {
		std::shared_lock lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()) {
			return 2;
		}
		const auto it = FindMappedContainingUnlocked(vaddr);
		if (it == m_mapped.end()) {
			return 2;
		}
		return it->second.type == MappingType::Backing ? 0 : 1;
	}

	void MapStats(uint64_t* backing, uint64_t* views, uint64_t* aliased, uint64_t* private_bytes) {
		m_backing->ViewStats(backing, views, aliased);
		std::shared_lock lock(m_mutex);
		for (const auto& [vaddr, region]: m_mapped) {
			(void)vaddr;
			if (region.type == MappingType::Private) {
				*private_bytes += region.size;
			}
		}
	}

	template <bool copy_private>
	bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
		// Lock-free path (KYTY_SPARSE_COPY_CACHE, on by default). Invariants it relies on:
		// - Memory safety: the copy only dereferences the store's writable alias
		//   [m_backing_base, +m_size), never a guest view. The alias is mapped once by the store's
		//   constructor, never protected, resized or unmapped before ~GuestBackingStore, and the
		//   locked path reads the same bytes. A slot only covers alias bytes the locked path
		//   dereferenced for that same entry, so a stale slot can only read other alias bytes,
		//   never freed or unmapped memory, whatever guest munmap or mmap does to the views.
		// - Slots are keyed by `this`. Requirement: one store per process (Memory::Initialize),
		//   and Memory::Shutdown destroys it only after every thread that copies has stopped.
		// - Translation: every m_maps change and every locked byte write (store MapFixed, Unmap,
		//   UnmapOne, ZeroRange, WriteBacking) bumps the 64-bit generation first, inside the same
		//   exclusive section of the store lock. Slot generations are only captured under the shared
		//   lock, when no writer is inside, so one bump per writer is enough (no odd/even): a copy
		//   that read the slot's generation before and after it ran entirely before the next writer
		//   acted, at a valid translation. Otherwise it is discarded and redone under the locks.
		// - What can unmap or remap a range meanwhile: UnmapBacking and MapBacking (guest munmap,
		//   mmap, MAP_FIXED over a view) go through those store calls. The Windows partial unmap
		//   bumps at extract and at each survivor's MapFixed; the survivors' new entries refill the
		//   cache on their next locked copy. Commit and ReleaseCommitted touch only free or
		//   Private ranges. Protect and ProtectTransient may change a backing view's host
		//   protection, but never m_maps or the alias, so they need no bump: the copy does not
		//   go through the view.
		// - A range inside one backing view is owned (MapBacking consumes owned free ranges only)
		//   and holds no Private mapping (the two types are disjoint), so skipping Owns and
		//   OverlapsPrivateUnlocked returns what the locked path returns. Mode 2 checks this.
		// - Bytes: the locks never guarded them. Guest CPU stores and tracker protections take no
		//   lock, and the alias ignores view protections, so a racing guest store is unordered with
		//   the copy in both paths. Tracker protect-then-copy stays program order on this thread.
		// Each thread (CP thread, #69 back thread) has its own slots; only the generation is shared,
		// so any map change or WriteBacking anywhere invalidates every thread's slots.
		if (data != nullptr && size != 0 && UINT64_MAX - vaddr >= size &&
		    GuestBackingStore::SparseCopyCacheMode() != 0) {
			uint64_t generation     = 0;
			uint64_t backing_offset = 0;
			int      stale          = 0;
			if (m_backing->TryCopyCached(vaddr, data, size, &generation, &backing_offset, &stale)) {
				if (GuestBackingStore::SparseCopyCacheMode() == 2) {
					// Same lock order as the locked path: address space, then backing store.
					EXIT_IF(!Owns(vaddr, size));
					std::shared_lock lock(m_mutex);
					const auto check = m_backing->CheckCopyTranslation(vaddr, size, generation,
					                                                   backing_offset);
					EXIT_IF(check == 0 || (check == 1 && OverlapsPrivateUnlocked(vaddr, size)));
					m_sparse_checked.fetch_add(1, std::memory_order_relaxed);
				}
				return true;
			}
			if (GuestBackingStore::SparseCopyCacheMode() == 2) {
				m_sparse_locked.fetch_add(1, std::memory_order_relaxed);
				if (stale != 0) {
					(stale == 1 ? m_sparse_stale_bytes : m_sparse_stale_maps).fetch_add(1, std::memory_order_relaxed);
				}
			}
		}
		// With the cache on, a miss also skips the lock Owns takes (m_owned is constant).
		if (data == nullptr || !(GuestBackingStore::SparseCopyCacheMode() != 0
		                             ? OwnsUnlocked(vaddr, size)
		                             : Owns(vaddr, size))) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		if constexpr (!copy_private) {
			if (OverlapsPrivateUnlocked(vaddr, size)) {
				return false;
			}
		}
		m_backing->ReadSparseBacking(
		    vaddr, data, size, [&](uint64_t start, uint8_t* destination, uint64_t bytes) {
			    const auto zero = [](uint64_t, uint8_t* output, uint64_t length) {
				    std::memset(output, 0, length);
			    };
			    if constexpr (copy_private) {
				    // Backing gaps can contain private mappings or reservations only.
				    CopySparseRange(
				        m_mapped, start, destination, bytes,
				        [](const auto&, uint64_t address) {
					        return reinterpret_cast<const void*>(address);
				        },
				        zero);
			    } else {
				    zero(start, destination, bytes);
			    }
		    });
		return true;
	}

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::lock_guard lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// Tracker protections only read m_mapped; the page tracker orders them per page.
	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::shared_lock lock(m_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
		std::lock_guard lock(m_mutex);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const bool replacing = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
#else
		constexpr bool replacing = false;
#endif
		if (!replacing && !ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		if (m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason)) {
			if (!replacing) {
				AddMappedUnlocked(vaddr, size, MappingType::Backing);
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
			return false;
		}
		if (!m_backing->Unmap(vaddr, size)) {
			return false;
		}
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		alignment = std::max<uint64_t>(alignment, PageSize());
		std::shared_lock lock(m_mutex);
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ContainsFree(uint64_t vaddr, uint64_t size) {
		std::shared_lock lock(m_mutex);
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::lock_guard lock(m_mutex);
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto* ptr = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
		                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
		                          GetProtect(host_mode), nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		AddMappedUnlocked(vaddr, size, MappingType::Private);
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			AddMappedUnlocked(vaddr, size, MappingType::Private);
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (!IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Private)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		while (current < end) {
			const auto it       = FindMappedContainingUnlocked(current);
			const auto part_end = std::min(end, it->second.vaddr + it->second.size);
			regions.emplace_back(current, part_end - current);
			current = part_end;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [region_vaddr, region_size]: regions) {
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_vaddr), region_size,
			                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
		}
		// ReleaseCommitted also publishes the aggregate span as one logical free range.
		EXIT_IF(regions.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		for (const auto& [region_vaddr, region_size]: regions) {
			RestorePlaceholder(region_vaddr, region_size);
		}
#endif
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		auto addr = FindFreeAligned(HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_MANAGED_MAX + 1u,
		                            PageSize(), PageSize());
		if (addr == 0) {
			addr = FindFreeAligned(HOST_USER_MIN, HOST_USER_MAX + 1u, PageSize(), PageSize());
		}
		if (addr == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: 16 KiB range unavailable\n");
			return false;
		}
		if (!Commit(addr, PageSize(), VirtualMemory::Mode::ReadWrite)) {
			LOGF_COLOR(Log::Color::Red, "\t guest address-space self-test: commit failed\n");
			return false;
		}

		auto* ptr = reinterpret_cast<uint64_t*>(addr);
		*ptr      = 0x4b59545950484f4cull; // "KYTYPHOL"

		if (!ReleaseCommitted(addr, PageSize())) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: release-committed failed\n");
			return false;
		}

		LOGF_COLOR(Log::Color::Green, "\t guest address-space self-test: ok\n");
		return true;
	}

private:
	enum class MappingType { Backing, Private };

	struct MappedRegion {
		uint64_t    vaddr = 0;
		uint64_t    size  = 0;
		MappingType type  = MappingType::Private;
	};

	static constexpr uint64_t PageSize() { return 0x4000; }

	static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
		const auto remainder = value % alignment;
		return remainder != 0 ? value + (alignment - remainder) : value;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start >= range_end) {
				continue;
			}
			if (!VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		const auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		m_free.erase(vaddr);
		return true;
	}

	std::map<uint64_t, MappedRegion>::iterator FindMappedContainingUnlocked(uint64_t vaddr) {
		auto next = m_mapped.upper_bound(vaddr);
		if (next == m_mapped.begin()) {
			return m_mapped.end();
		}
		auto it = std::prev(next);
		return vaddr < it->second.vaddr + it->second.size ? it : m_mapped.end();
	}

	bool ContainsMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it = FindMappedContainingUnlocked(current);
			if (it == m_mapped.end() || it->second.type != type) {
				return false;
			}
			current = std::min(end, it->second.vaddr + it->second.size);
		}
		return true;
	}

	[[nodiscard]] bool OverlapsPrivateUnlocked(uint64_t vaddr, uint64_t size) const {
		auto mapping = m_mapped.upper_bound(vaddr);
		if (mapping != m_mapped.begin()) {
			--mapping;
		}
		for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
			const auto& region = mapping->second;
			if (region.type == MappingType::Private && region.vaddr + region.size > vaddr) {
				return true;
			}
		}
		return false;
	}

	void AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type});
	}

	void RemoveMappedUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it           = FindMappedContainingUnlocked(current);
			const auto original     = it->second;
			const auto original_end = original.vaddr + original.size;
			const auto part_end     = std::min(end, original_end);
			m_mapped.erase(it);
			if (original.vaddr < current) {
				AddMappedUnlocked(original.vaddr, current - original.vaddr, original.type);
			}
			if (part_end < original_end) {
				AddMappedUnlocked(part_end, original_end - part_end, original.type);
			}
			current = part_end;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}
#endif

	void ReserveGuestRegions() {
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
		    {kExtendedMemoryBase, kExtendedMemoryBase + kExtendedMemorySize},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		for (const auto& [guest_start, guest_end]: regions) {
			auto next_addr = guest_start;
			while (next_addr < guest_end) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
					EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
				}

				const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end = std::min<uint64_t>(
				    guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
				const auto reserve_start =
				    AlignUp(std::max(region_base, guest_start), m_granularity);
				const auto reserve_end = AlignDown(region_end, m_granularity);

				if (info.State == MEM_FREE && reserve_end > reserve_start &&
				    reserve_end - reserve_start > MinimumRegionSize) {
					auto* ptr = VirtualAlloc2(
					    process, reinterpret_cast<void*>(reserve_start), reserve_end - reserve_start,
					    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
					if (reinterpret_cast<uint64_t>(ptr) != reserve_start) {
						EXIT("failed to reserve guest address space at 0x%016" PRIx64
						     ", size 0x%016" PRIx64 "\n",
						     reserve_start, reserve_end - reserve_start);
					}
					AddFreeUnlocked(reserve_start, reserve_end - reserve_start);
					m_owned.emplace_back(reserve_start, reserve_end - reserve_start);
				}

				next_addr = region_end;
			}
		}
#else
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		std::vector<std::pair<uint64_t, uint64_t>> occupied;
		FILE*                                      maps = fopen("/proc/self/maps", "r");
		EXIT_IF(maps == nullptr);

		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long mapping_start = 0;
			unsigned long long mapping_end   = 0;
			if (sscanf(line, "%llx-%llx", &mapping_start, &mapping_end) == 2) {
				occupied.emplace_back(static_cast<uint64_t>(mapping_start),
				                      static_cast<uint64_t>(mapping_end));
			}
		}
		fclose(maps);

		auto reserve_range = [this](uint64_t start, uint64_t end) {
			start = AlignUp(start, PageSize());
			end   = AlignDown(end, PageSize());
			if (end <= start) {
				return;
			}

			const auto size = end - start;
			int        flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif

			void* ptr = mmap(reinterpret_cast<void*>(start), size, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, size);
				}
				return;
			}

			AddFreeUnlocked(start, size);
			// Free ranges merge across region boundaries, so owned ranges do as well.
			if (!m_owned.empty() && m_owned.back().first + m_owned.back().second == start) {
				m_owned.back().second += size;
			} else {
				m_owned.emplace_back(start, size);
			}
		};

		for (const auto& [region_start, region_end]: regions) {
			auto current = region_start;

			for (const auto& [mapping_start, mapping_end]: occupied) {
				if (mapping_end <= current) {
					continue;
				}
				if (mapping_start >= region_end) {
					break;
				}

				if (mapping_start > current) {
					reserve_range(current, std::min(mapping_start, region_end));
				}

				current = std::max(current, mapping_end);
				if (current >= region_end) {
					break;
				}
			}

			if (current < region_end) {
				reserve_range(current, region_end);
			}
		}
#else
		for (const auto& [start, end]: regions) {
			int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif
			void* ptr = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, end - start);
				}
				EXIT("failed to reserve guest address space at 0x%016" PRIx64
				     ", size 0x%016" PRIx64 "\n",
				     start, end - start);
			}
			AddFreeUnlocked(start, end - start);
			m_owned.emplace_back(start, end - start);
		}
#endif
#endif
	}

	uint64_t FindFreeAlignedUnlocked(uint64_t search_start, uint64_t search_end, uint64_t size,
	                                 uint64_t alignment) const {
		// Include the predecessor because search_start can lie inside a free range.
		auto it = m_free.upper_bound(search_start);
		if (it != m_free.begin()) {
			--it;
		}
		for (; it != m_free.end() && it->first < search_end; ++it) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
			g_test_free_range_visits.fetch_add(1, std::memory_order_relaxed);
#endif
			const auto& [base, range_size] = *it;
			const auto candidate           = AlignUp(std::max(base, search_start), alignment);
			if (candidate < search_end && size <= search_end - candidate &&
			    GuestFreeRangeContains(base, range_size, candidate, size)) {
				return candidate;
			}
		}
		return 0;
	}

	std::map<uint64_t, uint64_t>::iterator FindContaining(uint64_t vaddr, uint64_t size) {
		auto next = m_free.upper_bound(vaddr);
		if (next == m_free.begin()) {
			return m_free.end();
		}
		auto it = std::prev(next);
		return GuestFreeRangeContains(it->first, it->second, vaddr, size) ? it : m_free.end();
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(base), left_size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           base, end - base, vaddr, size, static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			it->second = left_size;
			it         = m_free.emplace(vaddr, end - vaddr).first;
		}

		if (it->second != size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(it->first), size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           it->first, it->second, vaddr, size,
				           static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			const auto right_start = it->first + size;
			const auto right_size  = it->second - size;
			it->second             = size;
			m_free.emplace(right_start, right_size);
		}

		return true;
	}

	void AddFreeUnlocked(uint64_t vaddr, uint64_t size) {
		auto [it, inserted] = m_free.emplace(vaddr, size);
		if (!inserted) {
			it->second = size;
		}

		auto base = it->first;
		auto end  = it->first + it->second;

		auto first = it;
		while (first != m_free.begin()) {
			auto prev = std::prev(first);
			if (prev->first + prev->second != base) {
				break;
			}
			base  = prev->first;
			first = prev;
		}

		auto last = std::next(it);
		while (last != m_free.end() && end == last->first) {
			end  = last->first + last->second;
			last = std::next(last);
		}

		if (base != it->first || end != it->first + it->second) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (CoalescePlaceholders(base, end - base)) {
				m_free.erase(first, last);
				m_free.emplace(base, end - base);
			}
#else
			m_free.erase(first, last);
			m_free.emplace(base, end - base);
#endif
		}
	}

	void RestorePlaceholder(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// The address remains owned for the process lifetime; MAP_FIXED only replaces one of our
		// mappings and never acquires a new guest address.
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, PROT_NONE,
		                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		EXIT_IF(ptr == MAP_FAILED);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	std::shared_mutex                          m_mutex;
	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
	std::atomic<uint64_t>                      m_sparse_checked {0}; // KYTY_SPARSE_COPY_CACHE=2 only
	std::atomic<uint64_t>                      m_sparse_locked {0};
	std::atomic<uint64_t>                      m_sparse_stale_bytes {0}; // part of locked, mode 2 only
	std::atomic<uint64_t>                      m_sparse_stale_maps {0};
};
