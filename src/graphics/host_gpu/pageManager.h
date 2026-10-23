#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);
	// Re-applies the watch protections of [vaddr, vaddr+size) to the host pages, after the
	// mapping was recreated with plain guest permissions. Takes no tracker lock: the caller
	// holds the address-space lock that concurrent watcher updates wait for before protecting.
	void RestoreProtection(uint64_t vaddr, uint64_t size) noexcept;

	// KYTY_PROTECT_COALESCE (BufferCache::BeginDeferredUploads). Between Begin and Flush, the
	// protections this thread would add (read-only / no-access) inside a DeferCapture scope are
	// recorded instead of applied; the watcher counts change at once. Flush applies them in
	// merged runs: under each region's lock, every page of a recorded span (plus gaps of up to
	// 16 pages between spans) whose current state still asks for that protection. A page whose
	// state changed meanwhile (another thread made it writable again) is skipped: like every
	// other protection change, the host protection is set to the page's state under the region
	// lock, never to an older value. Until Flush, recorded pages stay writable, so their bytes
	// must be read after Flush.
	static void BeginDeferredProtect() noexcept;
	static void FlushDeferredProtect() noexcept;
	static void EndDeferredProtect() noexcept;
	[[nodiscard]] static bool DeferringProtect() noexcept;
	// Only protections made inside a capture scope (upload re-protections of a read-only
	// synchronization, MemoryTracker::ForEachUploadRange) are deferred; all others apply at once.
	class DeferCapture final {
	public:
		DeferCapture() noexcept;
		~DeferCapture();
		DeferCapture(const DeferCapture&)            = delete;
		DeferCapture& operator=(const DeferCapture&) = delete;

	private:
		bool m_previous = false;
	};

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
