#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;
class ReadbackQueue;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	// KYTY_ASYNC_READBACK (default 1): guest-thread fault readbacks do not stall the GPU thread.
	[[nodiscard]] static bool AsyncFaultReadbacks();

	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Guest read and write faults on tracked memory: like ReadMemory and InvalidateMemory for
	// the faulting byte, but the GPU-written bytes of the whole readback window around it
	// (KYTY_READBACK_WINDOW_KIB, clamped to the faulting buffer) are published in one sync.
	void                   ReadFaultMemory(uint64_t vaddr);
	void                   InvalidateFaultMemory(uint64_t vaddr);
	// GPU thread: like ReadMemory, but publishes the GPU-written bytes of the whole readback window
	// around the range, as guest faults do, so neighbouring reads no longer need the GPU.
	// KYTY_BACKING_READ_WINDOW=0 reads only the range.
	void                   ReadMemoryWindow(uint64_t vaddr, uint64_t size);
	// GPU thread: a command-processor write (WRITE_DATA) into GPU-owned memory. Instead of
	// draining the GPU to publish the page, the bytes go to the backing and to the cached buffer
	// at this point of the command stream. Returns false when the regular CPU write must be used
	// (KYTY_GPU_OWNED_WRITE_DATA=0 always does).
	[[nodiscard]] bool     WriteGpuOwned(uint64_t vaddr, const void* data, uint64_t size);
	// Any thread (KYTY_EOP_AT_GPU label delivery): the bytes were just written through the backing.
	// Marks them CPU-modified, as a guest write fault would, and returns true; returns false and
	// changes nothing when a byte is GPU-modified (a readback would overwrite them).
	[[nodiscard]] bool     InvalidateCompletedWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// GPU thread: while a batch is open, the CPU-data uploads into cache buffers share one leading
	// and one trailing barrier. The trailing barrier is recorded at the latest when the batch
	// closes, or before anything else is recorded (CommandBuffer::Handle).
	class UploadBatch {
	public:
		explicit UploadBatch(BufferCache& cache) noexcept: m_cache(cache) {
			m_cache.BeginUploadBatch();
		}
		~UploadBatch() { m_cache.EndUploadBatch(); }
		UploadBatch(const UploadBatch&)            = delete;
		UploadBatch& operator=(const UploadBatch&) = delete;

	private:
		BufferCache& m_cache;
	};
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// GPU thread: the tracker page of vaddr may hold GPU-written bytes (page bits only).
	[[nodiscard]] bool MaybeGpuModifiedPage(uint64_t vaddr) {
		return m_memory_tracker.MaybeGpuModifiedOnGpuThread(vaddr, sizeof(uint32_t));
	}
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	// Synchronizes the queued BDA ranges that are mapped, with deferred uploads (see
	// BeginDeferredUploads).
	void               SynchronizeBda(const RangeSet& mapped);
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	void PublishFaultWindow(uint64_t vaddr, bool is_write);

	// A guest-fault readback whose copy was submitted without waiting on the GPU thread: the
	// priority-operation thread publishes it and releases the page protection.
	struct ReadbackCompletion {
		std::mutex              mutex;
		std::condition_variable signal;
		bool                    done = false;
		// Readback-queue copy: the faulting thread waits for `side_tick` and runs `side_publish`.
		uint64_t                     side_tick = 0;
		Common::UniqueFunction<void> side_publish;
		void Signal() {
			{
				std::lock_guard lock(mutex);
				done = true;
			}
			signal.notify_all();
		}
		void Wait() {
			std::unique_lock lock(mutex);
			signal.wait(lock, [this] { return done; });
		}
	};
	struct PendingPublish {
		uint64_t                            begin = 0;
		uint64_t                            end   = 0;
		uint64_t                            tick  = 0;
		std::shared_ptr<ReadbackCompletion> done;
		bool                                side = false; // published by the faulting thread
	};
	[[nodiscard]] std::shared_ptr<ReadbackCompletion> FindPendingPublish(uint64_t vaddr,
	                                                                     uint64_t size);
	// GPU thread: waits until no queued fault readback covers the range (its publication
	// releases the page protection, which must not race new GPU ownership of the pages).
	void WaitPendingPublishes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool RecordDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                  Common::UniqueFunction<void>& publish);
	[[nodiscard]] std::shared_ptr<ReadbackCompletion> DownloadForFault(Buffer& buffer, uint64_t vaddr,
	                                                                   uint64_t size,
	                                                                   bool side_ok = false);
	// KYTY_READBACK_QUEUE (GPU thread): makes a write recorded now wait for readback-queue copies
	// of the same bytes, and notes the command-buffer tick that may write each 64 KiB of guest
	// memory through a cached buffer (not for CPU-data uploads, see the definition).
	void NoteBufferWrite(uint64_t vaddr, uint64_t size, bool cpu_upload = false);
	// Submitted main-queue command buffers not yet complete.
	[[nodiscard]] uint64_t MainQueueBacklog();
	// True when every such write to the range has completed on the GPU.
	[[nodiscard]] bool WritesComplete(uint64_t vaddr, uint64_t size);
	// Latest main-queue tick that wrote GPU data to the range (0: none known); UINT64_MAX when the
	// range is not tracked or was written by the command buffer being recorded.
	[[nodiscard]] uint64_t LatestWriteTick(uint64_t vaddr, uint64_t size);
	// KYTY_SYNC_SIDE_READS: ReadMemoryWindow through the readback queue; false when not taken.
	bool ReadWindowOnSideQueue(Buffer& buffer, uint64_t begin, uint64_t size);
	void CollectDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                     std::vector<vk::BufferCopy>& copies, uint64_t& total_size);
	void RecordCollected(Buffer& buffer, std::vector<vk::BufferCopy> copies, uint64_t total_size,
	                     Common::UniqueFunction<void>& publish);
	[[nodiscard]] std::shared_ptr<ReadbackCompletion>
	DownloadCollected(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                  std::vector<vk::BufferCopy> copies, uint64_t total_size,
	                  std::shared_ptr<ReadbackCompletion> done);

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void BeginUploadBatch() noexcept;
	void EndUploadBatch();
	// `may_defer`: a read-only synchronization, whose bytes may be copied later (see
	// BeginDeferredUploads).
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size, bool may_defer = false);
	// KYTY_PROTECT_COALESCE (default 1, 0 disables; GPU thread, SynchronizeBda): between Begin
	// and End, the upload re-protections of all synchronized buffers are applied together in
	// merged runs (PageManager::BeginDeferredProtect), and the guest bytes of those uploads are
	// copied into the (coherent) staging memory only after them: at End, or before any
	// submission. Begin returns false (nothing deferred) when the switch is off or staging is
	// not coherent.
	[[nodiscard]] bool BeginDeferredUploads();
	void               FlushDeferredUploads();
	void               EndDeferredUploads();
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// KYTY_CP_PIPELINE: the GPU thread's program lookups test the ranges while the back thread runs.
	std::mutex                                        m_gpu_ranges_mutex;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	std::mutex                                        m_pending_mutex;
	std::vector<PendingPublish>                       m_pending_publishes;
	std::unique_ptr<ReadbackQueue>                    m_readback_queue;
	MultiLevelPageTable<uint64_t, 16, 44, 16>         m_write_ticks;
	struct SideCopy {
		uint64_t begin = 0, end = 0, tick = 0;
	};
	std::vector<SideCopy> m_side_copies; // readback-queue copies that may still run (GPU thread)
	uint64_t              m_side_completed = 0;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	StreamBuffer                                      m_staging_buffer;
	struct BatchRange {
		vk::Buffer buffer;
		uint64_t   begin = 0;
		uint64_t   end   = 0;
	};
	uint32_t                m_upload_batch_depth = 0;
	std::vector<BatchRange> m_batch_ranges; // destinations copied since the batch's leading barrier
	struct DeferredCopy {
		uint64_t address;
		uint8_t* destination;
		uint64_t size;
	};
	bool                                              m_deferring_uploads = false;
	std::vector<DeferredCopy>                         m_deferred_copies;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
