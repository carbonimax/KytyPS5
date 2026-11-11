#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/memTrackProf.h"

#include "common/alignment.h"
#include "graphics/guest_gpu/gpuCapture.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/gpuMarkers.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpPipeline.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <optional>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// Second queue (KYTY_READBACK_QUEUE) for guest-fault readbacks of GPU writes that already
// completed: the copy does not wait behind the work queued on the main queue. GPU thread:
// Allocate, Begin, Submit. Faulting threads: Wait, Release.
class ReadbackQueue {
public:
	static constexpr uint64_t RingSize = 32ull * 1024 * 1024;

	ReadbackQueue(GraphicContext& graphics, CommandScheduler& scheduler)
	    : m_graphics(graphics),
	      m_ring(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	             RingSize) {
		EXIT_IF(m_ring.Mapped().empty());
		vk::CommandPoolCreateInfo pool {};
		pool.queueFamilyIndex = graphics.readback_family;
		pool.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		             vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		EXIT_IF(graphics.device.createCommandPool(&pool, nullptr, &m_pool) != vk::Result::eSuccess);
		vk::SemaphoreTypeCreateInfo type {};
		type.semaphoreType = vk::SemaphoreType::eTimeline;
		type.initialValue  = 0;
		vk::SemaphoreCreateInfo create {};
		create.pNext = &type;
		EXIT_IF(graphics.device.createSemaphore(&create, nullptr, &m_semaphore) != vk::Result::eSuccess);
	}
	~ReadbackQueue() {
		Wait(m_next - 1);
		m_graphics.device.destroyCommandPool(m_pool, nullptr);
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
	KYTY_CLASS_NO_COPY(ReadbackQueue);

	[[nodiscard]] vk::Semaphore Semaphore() const noexcept { return m_semaphore; }
	[[nodiscard]] uint64_t      LastTick() const noexcept { return m_next - 1; }
	[[nodiscard]] uint64_t      CompletedTick() {
		uint64_t value = 0;
		EXIT_IF(m_graphics.device.getSemaphoreCounterValue(m_semaphore, &value) != vk::Result::eSuccess);
		return value;
	}
	[[nodiscard]] Buffer&       Ring() noexcept { return m_ring; }

	// Ring space for `size` bytes, released by Release(seq) once published.
	[[nodiscard]] bool Allocate(uint64_t size, uint64_t& offset, uint64_t& seq) {
		size = Common::AlignUp(size, 256);
		std::lock_guard lock(m_mutex);
		while (!m_live.empty() && m_live.front().released) {
			m_live.pop_front();
		}
		if (size > RingSize / 2) {
			return false;
		}
		uint64_t at = m_head;
		if (m_live.empty()) {
			if (at + size > RingSize) {
				at = 0;
			}
		} else {
			const auto tail = m_live.front().begin;
			if (at >= tail) {
				if (at + size > RingSize) {
					if (size > tail) {
						return false;
					}
					at = 0;
				}
			} else if (at + size > tail) {
				return false;
			}
		}
		m_head = at + size;
		seq    = ++m_seq;
		m_live.push_back({seq, at, false});
		offset = at;
		return true;
	}
	void Release(uint64_t seq) {
		std::lock_guard lock(m_mutex);
		for (auto& live: m_live) {
			if (live.seq == seq) {
				live.released = true;
				break;
			}
		}
	}

	[[nodiscard]] vk::CommandBuffer Begin() {
		uint64_t completed = 0;
		EXIT_IF(m_graphics.device.getSemaphoreCounterValue(m_semaphore, &completed) !=
		        vk::Result::eSuccess);
		size_t index = m_commands.size();
		for (size_t i = 0; i < m_commands.size(); i++) {
			if (m_commands[i].second <= completed) {
				index = i;
				break;
			}
		}
		if (index == m_commands.size()) {
			vk::CommandBufferAllocateInfo allocate {};
			allocate.commandPool        = m_pool;
			allocate.level              = vk::CommandBufferLevel::ePrimary;
			allocate.commandBufferCount = 1;
			vk::CommandBuffer command;
			EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, &command) != vk::Result::eSuccess);
			m_commands.emplace_back(command, 0);
		}
		m_current = index;
		auto command = m_commands[index].first;
		command.reset({});
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		EXIT_IF(command.begin(&begin) != vk::Result::eSuccess);
		return command;
	}
	// Submits the command buffer of Begin after `main_tick` of the main timeline (complete
	// already: the wait only orders its memory). Returns the readback-queue tick.
	uint64_t Submit(vk::Semaphore main_semaphore, uint64_t main_tick) {
		auto command = m_commands[m_current].first;
		EXIT_IF(command.end() != vk::Result::eSuccess);
		const uint64_t                    tick       = m_next++;
		const vk::PipelineStageFlags      wait_stage = vk::PipelineStageFlagBits::eTransfer;
		vk::TimelineSemaphoreSubmitInfo   timeline {};
		timeline.waitSemaphoreValueCount   = 1;
		timeline.pWaitSemaphoreValues      = &main_tick;
		timeline.signalSemaphoreValueCount = 1;
		timeline.pSignalSemaphoreValues    = &tick;
		vk::SubmitInfo submit {};
		submit.pNext                = &timeline;
		submit.waitSemaphoreCount   = 1;
		submit.pWaitSemaphores      = &main_semaphore;
		submit.pWaitDstStageMask    = &wait_stage;
		submit.commandBufferCount   = 1;
		submit.pCommandBuffers      = &command;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores    = &m_semaphore;
		vk::Result result;
		if (m_graphics.readback_family == m_graphics.queue_family) {
			// Same family: the driver may serialize queue access across queues.
			Common::LockGuard lock(m_graphics.queue_mutex);
			result = m_graphics.readback_queue.submit(1, &submit, nullptr);
		} else {
			result = m_graphics.readback_queue.submit(1, &submit, nullptr);
		}
		m_graphics.ReportDeviceLost(result, "readback vkQueueSubmit");
		EXIT_IF(result != vk::Result::eSuccess);
		m_commands[m_current].second = tick;
		return tick;
	}
	void Wait(uint64_t tick) {
		if (tick == 0) {
			return;
		}
		vk::SemaphoreWaitInfo wait {};
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &m_semaphore;
		wait.pValues        = &tick;
		const auto result = m_graphics.device.waitSemaphores(&wait, UINT64_MAX);
		m_graphics.ReportDeviceLost(result, "readback vkWaitSemaphores");
		EXIT_IF(result != vk::Result::eSuccess);
	}

private:
	struct Live {
		uint64_t seq      = 0;
		uint64_t begin    = 0;
		bool     released = false;
	};

	GraphicContext&                                       m_graphics;
	Buffer                                                m_ring;
	vk::CommandPool                                       m_pool      = nullptr;
	vk::Semaphore                                         m_semaphore = nullptr;
	std::vector<std::pair<vk::CommandBuffer, uint64_t>>   m_commands;
	size_t                                                m_current = 0;
	uint64_t                                              m_next    = 1;
	std::mutex                                            m_mutex;
	std::deque<Live>                                      m_live;
	uint64_t                                              m_head = 0;
	uint64_t                                              m_seq  = 0;
};

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		{
			PerfStats::ScopedTimer timer(PerfStats::Counter::StagingWriteNs,
			                             PerfStats::Counter::StagingWriteCount);
			write(mapped, offset);
		}
		PerfStats::Add(PerfStats::Counter::StagingWriteBytes, size);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferRelease([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

// KYTY_MEM_IMPORT counters ("PERF import" lines, KYTY_PERF_STATS=1).
namespace {

struct ImportStats {
	std::atomic<uint64_t> imported {0};     // cache buffers created on imported memory
	std::atomic<uint64_t> device {0};       // ... in device memory: range not one backing run
	std::atomic<uint64_t> failed {0};       // ... in device memory: import or binding refused
	std::atomic<uint64_t> chunks {0};       // 64 MiB backing chunks imported
	std::atomic<uint64_t> own {0};          // large buffers importing their own range
	std::atomic<uint64_t> import_ns {0};    // time in vkAllocateMemory (imports)
	std::atomic<uint64_t> read_syncs {0};   // read synchronizations: nothing to upload
	std::atomic<uint64_t> write_syncs {0};  // write synchronizations: ownership only, no copy
	std::atomic<uint64_t> published {0};    // GPU writes published without a copy, already complete
	std::atomic<uint64_t> waited {0};       // ... after the GPU thread waited for the writer
	std::atomic<uint64_t> wait_ns {0};
	std::atomic<uint64_t> replayed {0};     // guest faults that waited for the writer themselves
	std::atomic<uint64_t> stream_skips {0}; // small reads bound directly instead of stream copies
	std::atomic<uint64_t> join_copies {0};  // GPU-written ranges copied into a new imported buffer
	std::atomic<uint64_t> dropped {0};      // imported buffers deleted at unmap
	std::atomic<uint64_t> image_syncs {0};  // texel buffers filled from an image (guest memory written)
	std::atomic<uint64_t> owned_labels {0}; // GPU-owned labels written in the stream only
	std::atomic<uint64_t> cp_writes {0};    // command-processor CPU writes into imported memory
	std::atomic<uint64_t> cp_waits {0};     // ... that waited for earlier GPU work
	std::atomic<uint64_t> cp_wait_ns {0};
	std::atomic<uint64_t> write_demotes {0}; // imported buffers sent back to device memory by a GPU write
};

ImportStats g_import_stats;

// KYTY_MEM_IMPORT_LOG=1 (diagnostic): log every imported buffer and the GPU-written ranges of each.
static bool ImportLogEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MEM_IMPORT_LOG");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// First GPU write of each 16 KiB page of imported memory (KYTY_MEM_IMPORT_LOG, GPU thread).
std::unordered_set<uint64_t> g_import_logged_writes;

uint64_t ImportNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

// Imports the writable backing alias of guest direct memory (never protected, so always importable)
// in 64 MiB chunks every 32 MiB: a range of up to 32 MiB always fits in one chunk. Chunks stay
// imported until shutdown. Larger buffers import their own range.
class BufferCache::HostImport {
public:
	static constexpr uint64_t ChunkStride = 32ull * 1024 * 1024;
	static constexpr uint64_t ChunkSize   = 2 * ChunkStride;
	static constexpr uint64_t ImportSlack = 64ull * 1024; // > one caching page (16 KiB)

	explicit HostImport(GraphicContext& graphics)
	    : m_graphics(graphics), m_base(LibKernel::Memory::BackingAliasBase()),
	      m_size(LibKernel::Memory::BackingAliasSize()) {}
	~HostImport() {
		for (auto& [index, memory]: m_chunks) {
			(void)index;
			if (memory != nullptr) {
				m_graphics.device.freeMemory(memory);
			}
		}
	}
	KYTY_CLASS_NO_COPY(HostImport);

	[[nodiscard]] bool Valid() const noexcept { return m_base != 0 && m_size != 0; }

	// Memory for a buffer of `size` bytes whose guest bytes start at host address `host` (inside the
	// alias); nullopt when it cannot be imported.
	std::optional<ImportedMemory> Acquire(uint64_t host, uint64_t size, vk::BufferUsageFlags flags) {
		if (host < m_base || size == 0 || host - m_base > m_size || size > m_size - (host - m_base)) {
			return std::nullopt;
		}
		const auto requirements = Buffer::ImportedRequirements(m_graphics, flags, size);
		const auto offset       = host - m_base;
		std::lock_guard lock(m_mutex);
		if (m_type == UINT32_MAX) {
			m_type = PickType(requirements.memoryTypeBits);
		}
		if (m_type == UINT32_MAX || (requirements.memoryTypeBits & (1u << m_type)) == 0) {
			return std::nullopt;
		}
		// BDA shaders address whole caching pages from a buffer's device address: keep the imported
		// memory mapped past the buffer's end, or a read of its last page leaves the allocation.
		const auto needed      = requirements.size + ImportSlack;
		const auto index       = offset / ChunkStride;
		const auto chunk_begin = index * ChunkStride;
		const auto chunk_size  = std::min(ChunkSize, m_size - chunk_begin);
		const auto in_chunk    = offset - chunk_begin;
		if (needed <= chunk_size && in_chunk <= chunk_size - needed &&
		    in_chunk % std::max<uint64_t>(requirements.alignment, 1) == 0) {
			auto [it, inserted] = m_chunks.try_emplace(index, nullptr);
			if (inserted) {
				it->second = Import(m_base + chunk_begin, chunk_size);
				if (it->second != nullptr) {
					g_import_stats.chunks.fetch_add(1, std::memory_order_relaxed);
				}
			}
			if (it->second == nullptr) {
				return std::nullopt;
			}
			return ImportedMemory {it->second, in_chunk, false};
		}
		// Larger than a chunk stride (or past the last chunk): import exactly this range.
		const auto alignment = std::max<uint64_t>(m_graphics.host_import_alignment, 1);
		const auto bytes     = Common::AlignUp(needed, alignment);
		if (host % alignment != 0 || bytes > m_size - offset) {
			return std::nullopt;
		}
		auto memory = Import(host, bytes);
		if (memory == nullptr) {
			return std::nullopt;
		}
		g_import_stats.own.fetch_add(1, std::memory_order_relaxed);
		return ImportedMemory {memory, 0, true};
	}

private:
	[[nodiscard]] uint32_t PickType(uint32_t buffer_bits) {
		vk::MemoryHostPointerPropertiesEXT properties {};
		const auto result = m_graphics.device.getMemoryHostPointerPropertiesEXT(
		    vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
		    reinterpret_cast<const void*>(m_base), &properties);
		if (result != vk::Result::eSuccess) {
			return UINT32_MAX;
		}
		const auto  bits  = properties.memoryTypeBits & buffer_bits;
		const auto& types = m_graphics.physical_device_memory_properties;
		uint32_t    any   = UINT32_MAX;
		for (uint32_t i = 0; i < types.memoryTypeCount; i++) {
			if ((bits & (1u << i)) == 0) {
				continue;
			}
			if (types.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent) {
				return i;
			}
			any = std::min(any, i);
		}
		return any;
	}

	[[nodiscard]] vk::DeviceMemory Import(uint64_t host, uint64_t size) {
		vk::ImportMemoryHostPointerInfoEXT import {};
		import.handleType   = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
		import.pHostPointer = reinterpret_cast<void*>(host);
		vk::MemoryAllocateFlagsInfo flags {};
		flags.pNext = &import;
		flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
		vk::MemoryAllocateInfo allocate {};
		allocate.pNext           = &flags;
		allocate.allocationSize  = size;
		allocate.memoryTypeIndex = m_type;
		vk::DeviceMemory memory  = nullptr;
		const auto       start   = ImportNowNs();
		const auto       result  = m_graphics.device.allocateMemory(&allocate, nullptr, &memory);
		g_import_stats.import_ns.fetch_add(ImportNowNs() - start, std::memory_order_relaxed);
		if (result != vk::Result::eSuccess) {
			LOGF("BufferCache: host memory import refused (%d) host=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
			     static_cast<int>(result), host, size);
			return nullptr;
		}
		return memory;
	}

	GraphicContext&                                m_graphics;
	uint64_t                                       m_base = 0;
	uint64_t                                       m_size = 0;
	uint32_t                                       m_type = UINT32_MAX;
	std::mutex                                     m_mutex;
	std::unordered_map<uint64_t, vk::DeviceMemory> m_chunks;
};

void BufferCache::ReportImport(double seconds) {
	if (seconds <= 0.0) {
		return;
	}
	auto&      st           = g_import_stats;
	const auto take         = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const auto imported     = take(st.imported);
	const auto device       = take(st.device);
	const auto failed       = take(st.failed);
	const auto chunks       = take(st.chunks);
	const auto own          = take(st.own);
	const auto import_ns    = take(st.import_ns);
	const auto read_syncs   = take(st.read_syncs);
	const auto write_syncs  = take(st.write_syncs);
	const auto published    = take(st.published);
	const auto waited       = take(st.waited);
	const auto wait_ns      = take(st.wait_ns);
	const auto replayed     = take(st.replayed);
	const auto stream_skips = take(st.stream_skips);
	const auto join_copies  = take(st.join_copies);
	const auto dropped      = take(st.dropped);
	const auto image_syncs  = take(st.image_syncs);
	const auto owned_labels = take(st.owned_labels);
	const auto cp_writes    = take(st.cp_writes);
	const auto cp_waits     = take(st.cp_waits);
	const auto cp_wait_ns   = take(st.cp_wait_ns);
	const auto demotes      = take(st.write_demotes);
	if (imported + device + failed + read_syncs + write_syncs + published + waited == 0) {
		return;
	}
	std::printf("PERF import %.1fs: buffers imported=%" PRIu64 " device=%" PRIu64 " failed=%" PRIu64
	            " chunks=%" PRIu64 " own=%" PRIu64 " import=%.1fms | syncs read=%.0f/s write=%.0f/s"
	            " stream_skips=%.0f/s | publish now=%.0f/s waited=%.0f/s(%.2fms/s) replayed=%.0f/s"
	            " | join_copies=%" PRIu64 " dropped=%" PRIu64 " image_syncs=%.0f/s owned_labels=%.0f/s"
	            " | cp_writes=%.0f/s cp_waits=%.0f/s(%.2fms/s) write_demotes=%" PRIu64 "\n",
	            seconds, imported, device, failed, chunks, own, static_cast<double>(import_ns) / 1e6,
	            static_cast<double>(read_syncs) / seconds, static_cast<double>(write_syncs) / seconds,
	            static_cast<double>(stream_skips) / seconds, static_cast<double>(published) / seconds,
	            static_cast<double>(waited) / seconds, static_cast<double>(wait_ns) / 1e6 / seconds,
	            static_cast<double>(replayed) / seconds, join_copies, dropped,
	            static_cast<double>(image_syncs) / seconds, static_cast<double>(owned_labels) / seconds,
	            static_cast<double>(cp_writes) / seconds, static_cast<double>(cp_waits) / seconds,
	            static_cast<double>(cp_wait_ns) / 1e6 / seconds, demotes);
}

void BufferCache::BeforeCpuWrite(uint64_t vaddr, uint64_t size) {
	if (!m_import || size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	uint64_t latest = 0;
	bool     any    = false;
	for (auto chunk = vaddr >> 16; chunk <= (vaddr + size - 1) >> 16; chunk++) {
		if (!decltype(m_read_ticks)::IsValidPage(chunk)) {
			return;
		}
		if (const auto* tick = m_read_ticks.Find(chunk); tick != nullptr && *tick != 0) {
			latest = std::max(latest, *tick);
			any    = true;
		}
		if (const auto* tick = m_write_ticks.Find(chunk); tick != nullptr) {
			latest = std::max(latest, *tick);
		}
	}
	if (!any) {
		return; // never bound from imported memory
	}
	g_import_stats.cp_writes.fetch_add(1, std::memory_order_relaxed);
	if (latest == 0 || m_scheduler.IsFree(latest)) {
		return;
	}
	const auto start = ImportNowNs();
	if (latest >= m_scheduler.CurrentTick()) {
		latest = m_scheduler.CurrentTick();
		m_scheduler.Flush();
	}
	m_scheduler.Wait(latest);
	g_import_stats.cp_wait_ns.fetch_add(ImportNowNs() - start, std::memory_order_relaxed);
	g_import_stats.cp_waits.fetch_add(1, std::memory_order_relaxed);
}

bool BufferCache::IsImportedRange(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return false;
	}
	const auto& buffer = m_slot_buffers[*owner];
	return buffer.IsImported() && buffer.IsInBounds(vaddr, size);
}

bool BufferCache::PublishImported(uint64_t vaddr, uint64_t size, bool may_wait, uint64_t* wait_tick) {
	auto tick = LatestWriteTick(vaddr, size);
	if (tick == UINT64_MAX) {
		// Written by the command buffer being recorded: submit it first.
		tick = m_scheduler.CurrentTick();
		m_scheduler.Flush();
	}
	if (tick != 0 && !m_scheduler.IsFree(tick)) {
		if (!may_wait) {
			*wait_tick = tick;
			return false;
		}
		const auto start = ImportNowNs();
		m_scheduler.Wait(tick);
		g_import_stats.wait_ns.fetch_add(ImportNowNs() - start, std::memory_order_relaxed);
		g_import_stats.waited.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_import_stats.published.fetch_add(1, std::memory_order_relaxed);
	}
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    {
			    std::lock_guard ranges_lock(m_gpu_ranges_mutex);
			    m_gpu_modified_ranges.Subtract(address, bytes);
		    }
		    ForgetColorMetadataChecks(address, bytes);
	    });
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	return true;
}

void BufferCache::DropImported(uint64_t vaddr, uint64_t size) {
	if (!m_import) {
		return;
	}
	std::vector<BufferId> drop;
	auto                  it = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < vaddr + size; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.IsImported() && buffer.CpuAddress() + buffer.Size() > vaddr) {
			drop.push_back(it->second);
		}
	}
	for (const auto id: drop) {
		auto& buffer = m_slot_buffers[id];
		WaitPendingPublishes(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			(void)PublishImported(buffer.CpuAddress(), buffer.Size(), true, nullptr);
		}
		DeleteBuffer(id);
		g_import_stats.dropped.fetch_add(1, std::memory_order_relaxed);
	}
}

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	// Readback-queue copies may still read the buffer.
	const auto side_tick = m_readback_queue != nullptr ? m_readback_queue->LastTick() : 0;
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id, side_tick] {
			if (side_tick != 0) {
				m_readback_queue->Wait(side_tick);
			}
			if (m_graphics.device_address_destruction_waits_for_queue) {
				// Every submitted command buffer may reference this address until Metal completes.
				// ponytail: drains the queue per deletion batch; a per-buffer retire tick would
				// need the driver to stop referencing buffers the work does not use.
				m_scheduler.WaitRetired(m_scheduler.CurrentTick() - 1);
			}
			m_slot_buffers.erase(id);
		});
	} else {
		if (side_tick != 0) {
			m_readback_queue->Wait(side_tick);
		}
		m_slot_buffers.erase(id);
	}
}

std::shared_ptr<BufferCache::ReadbackCompletion> BufferCache::FindPendingPublish(uint64_t vaddr,
                                                                                uint64_t size) {
	std::lock_guard lock(m_pending_mutex);
	for (const auto& pending: m_pending_publishes) {
		if (pending.begin < vaddr + size && vaddr < pending.end) {
			return pending.done;
		}
	}
	return nullptr;
}

void BufferCache::WaitPendingPublishes(uint64_t vaddr, uint64_t size) {
	// Readbacks are only left pending in asynchronous mode.
	if (!AsyncFaultReadbacks()) {
		return;
	}
	uint64_t tick = 0;
	std::vector<std::shared_ptr<ReadbackCompletion>> side;
	{
		std::lock_guard lock(m_pending_mutex);
		for (const auto& pending: m_pending_publishes) {
			if (pending.begin < vaddr + size && vaddr < pending.end) {
				if (pending.side) {
					side.push_back(pending.done);
				} else {
					tick = std::max(tick, pending.tick);
				}
			}
		}
	}
	if (tick != 0) {
		m_scheduler.WaitPriorityOperations(tick);
	}
	// Readback-queue publications are run by the faulting thread, which only waits on that
	// queue's copy.
	for (const auto& done: side) {
		done->Wait();
	}
}

void BufferCache::NoteBufferWrite(uint64_t vaddr, uint64_t size, bool cpu_upload) {
	if ((m_readback_queue == nullptr && !m_import) || size == 0) {
		return;
	}
	if (m_readback_queue != nullptr && !m_side_copies.empty()) {
		// A write recorded now runs after an earlier readback-queue copy of the same bytes only
		// if the main queue waits for it (write after read across queues).
		uint64_t wait = 0;
		for (const auto& copy: m_side_copies) {
			if (copy.begin < vaddr + size && vaddr < copy.end && copy.tick > m_side_completed) {
				wait = std::max(wait, copy.tick);
			}
		}
		if (wait != 0 || m_side_copies.size() > 256) {
			m_side_completed = m_readback_queue->CompletedTick();
			std::erase_if(m_side_copies,
			              [this](const SideCopy& copy) { return copy.tick <= m_side_completed; });
		}
		if (wait > m_side_completed) {
			m_scheduler.WaitBeforeNextSubmit(m_readback_queue->Semaphore(), wait);
		}
	}
	// The ticks only decide whether a readback-queue copy, which copies the GPU-modified bytes of
	// GPU-dirty pages, may run without waiting for the main queue. CPU-data uploads only write
	// CPU-dirty pages, which are never GPU-dirty at the same time, so no such copy reads what they
	// write; a later write binding over those bytes notes its own tick.
	if (cpu_upload) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	for (auto chunk = vaddr >> 16; chunk <= (vaddr + size - 1) >> 16; chunk++) {
		if (decltype(m_write_ticks)::IsValidPage(chunk)) {
			m_write_ticks[chunk] = tick;
		}
	}
}

uint64_t BufferCache::MainQueueBacklog() {
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto known   = master.KnownGpuTick();
	const auto current = m_scheduler.CurrentTick();
	return current - 1 > known ? current - 1 - known : 0;
}

uint64_t BufferCache::LatestWriteTick(uint64_t vaddr, uint64_t size) {
	uint64_t latest = 0;
	for (auto chunk = vaddr >> 16; chunk <= (vaddr + size - 1) >> 16; chunk++) {
		if (!decltype(m_write_ticks)::IsValidPage(chunk)) {
			return UINT64_MAX;
		}
		if (const auto* tick = m_write_ticks.Find(chunk); tick != nullptr) {
			latest = std::max(latest, *tick);
		}
	}
	return latest >= m_scheduler.CurrentTick() ? UINT64_MAX : latest;
}

// KYTY_SYNC_SIDE_READS=N (default 2): a synchronous read of GPU-written bytes by the GPU thread
// (backing reads of the SRT walker, shader and vertex-table reads: ReadMemoryWindow; fault windows
// published on the GPU thread, e.g. indirect arguments read by the command processor) whose writers
// were all submitted copies them on the readback queue, as guest fault readbacks do, instead of
// recording the copy in the command buffer being recorded, submitting it and waiting for the whole
// main-queue backlog. 0: off; 1: only when every write already completed; 2: also when the writers are
// submitted but still running (waits for the latest writer's tick first). The copied bytes, their
// publication and the memory-tracker state are the same; later main-queue writes to them wait for
// the copy (NoteBufferWrite). Interrupts, labels and command order are untouched.
static uint32_t SyncSideReads() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_SYNC_SIDE_READS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u;
	}();
	return mode;
}

bool BufferCache::ReadWindowOnSideQueue(Buffer& buffer, uint64_t begin, uint64_t size) {
	if (buffer.IsImported()) {
		WaitPendingPublishes(begin, size);
		(void)PublishImported(begin, size, true, nullptr);
		return true;
	}
	const auto mode = SyncSideReads();
	if (mode == 0 || m_readback_queue == nullptr) {
		return false;
	}
	const auto latest = LatestWriteTick(begin, size);
	if (latest == UINT64_MAX) {
		return false;
	}
	if (latest != 0 && !m_scheduler.IsFree(latest)) {
		if (mode < 2) {
			return false;
		}
		m_scheduler.Wait(latest);
	}
	WaitPendingPublishes(begin, size);
	auto done = DownloadForFault(buffer, begin, size, true);
	if (done == nullptr) {
		// No GPU-written byte left in the window: nothing to publish.
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, size);
		return true;
	}
	if (done->side_tick != 0) {
		PerfStats::ScopedTimer timer(PerfStats::Counter::SideReadNs, PerfStats::Counter::SideReadCount);
		m_readback_queue->Wait(done->side_tick);
		done->side_publish();
		return true;
	}
	// No ring space: the main queue copies the window, published by a priority operation.
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	done->Wait();
	return true;
}

bool BufferCache::WritesComplete(uint64_t vaddr, uint64_t size) {
	uint64_t latest = 0;
	for (auto chunk = vaddr >> 16; chunk <= (vaddr + size - 1) >> 16; chunk++) {
		if (!decltype(m_write_ticks)::IsValidPage(chunk)) {
			return false;
		}
		if (const auto* tick = m_write_ticks.Find(chunk); tick != nullptr) {
			latest = std::max(latest, *tick);
		}
	}
	if (latest >= m_scheduler.CurrentTick()) {
		return false; // written by the command buffer being recorded
	}
	return latest == 0 || m_scheduler.IsFree(latest);
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	WaitPendingPublishes(vaddr, size);
	if (buffer.IsImported()) {
		return PublishImported(vaddr, size, true, nullptr);
	}
	Common::UniqueFunction<void> publish;
	if (!RecordDownload(buffer, vaddr, size, publish)) {
		return false;
	}
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		{
			MemTrackProf::Timer timer(MemTrackProf::Counter::DownloadSyncNs, MemTrackProf::Counter::DownloadSyncCount);
			m_scheduler.Wait(tick);
			m_scheduler.WaitPriorityOperations(tick);
		}
		publish();
	}
	return true;
}

std::shared_ptr<BufferCache::ReadbackCompletion> BufferCache::DownloadForFault(Buffer&  buffer,
                                                                              uint64_t vaddr,
                                                                              uint64_t size,
                                                                              bool     side_ok) {
	if (side_ok && m_readback_queue != nullptr) {
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total_size = 0;
		CollectDownload(buffer, vaddr, size, copies, total_size);
		if (copies.empty()) {
			return nullptr;
		}
		auto     done   = std::make_shared<ReadbackCompletion>();
		uint64_t offset = 0;
		uint64_t seq    = 0;
		if (m_readback_queue->Allocate(total_size, offset, seq)) {
			for (auto& copy: copies) {
				copy.dstOffset += offset;
			}
			auto&      ring    = m_readback_queue->Ring();
			const auto command = m_readback_queue->Begin();
			command.copyBuffer(buffer.Handle(), ring.Handle(), static_cast<uint32_t>(copies.size()),
			                   copies.data());
			vk::BufferMemoryBarrier after {};
			after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
			after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			after.buffer              = ring.Handle();
			after.offset              = offset;
			after.size                = Common::AlignUp(total_size, 256);
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
			                        nullptr);
			auto& master    = m_scheduler.GetMasterSemaphore();
			done->side_tick = m_readback_queue->Submit(master.Handle(), master.KnownGpuTick());
			// Later main-queue writes to these bytes wait for the copy (NoteBufferWrite).
			m_side_copies.push_back({vaddr, vaddr + size, done->side_tick});
			{
				std::lock_guard lock(m_pending_mutex);
				m_pending_publishes.push_back({vaddr, vaddr + size, 0, done, true});
			}
			const auto  buffer_address = buffer.CpuAddress();
			auto* const mapped         = ring.Mapped().data() + offset;
			auto* const raw_done       = done.get();
			done->side_publish = [this, vaddr, size, raw_done, seq, offset, total_size, buffer_address,
			                      mapped, copies = std::move(copies)] {
				auto& ring = m_readback_queue->Ring();
				ring.Invalidate(offset, Common::AlignUp(total_size, 256));
				for (const auto& copy: copies) {
					Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
					                                      mapped + (copy.dstOffset - offset), copy.size);
				}
				m_readback_queue->Release(seq);
				m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
				{
					std::lock_guard lock(m_pending_mutex);
					std::erase_if(m_pending_publishes, [&](const PendingPublish& pending) {
						return pending.done.get() == raw_done;
					});
				}
				raw_done->Signal();
			};
			return done;
		}
		// No ring space: the main queue copies these ranges.
		return DownloadCollected(buffer, vaddr, size, std::move(copies), total_size, std::move(done));
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	CollectDownload(buffer, vaddr, size, copies, total_size);
	if (copies.empty()) {
		return nullptr;
	}
	return DownloadCollected(buffer, vaddr, size, std::move(copies), total_size,
	                         std::make_shared<ReadbackCompletion>());
}

std::shared_ptr<BufferCache::ReadbackCompletion>
BufferCache::DownloadCollected(Buffer& buffer, uint64_t vaddr, uint64_t size,
                               std::vector<vk::BufferCopy> copies, uint64_t total_size,
                               std::shared_ptr<ReadbackCompletion> done) {
	Common::UniqueFunction<void> publish;
	RecordCollected(buffer, std::move(copies), total_size, publish);
	{
		std::lock_guard lock(m_pending_mutex);
		m_pending_publishes.push_back({vaddr, vaddr + size, m_scheduler.CurrentTick(), done});
	}
	// Runs once the copy's command buffer completed. The pages keep their GPU ownership (and
	// protection) until the bytes are published; GPU-thread users of the range wait for this.
	m_scheduler.DeferPriorityOperation(
	    [this, vaddr, size, done, publish = std::move(publish)]() mutable {
		    publish();
		    m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		    {
			    std::lock_guard lock(m_pending_mutex);
			    std::erase_if(m_pending_publishes,
			                  [&](const PendingPublish& pending) { return pending.done == done; });
		    }
		    done->Signal();
	    });
	m_scheduler.Flush();
	return done;
}

bool BufferCache::RecordDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                 Common::UniqueFunction<void>& publish_out) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	CollectDownload(buffer, vaddr, size, copies, total_size);
	if (copies.empty()) {
		return false;
	}
	RecordCollected(buffer, std::move(copies), total_size, publish_out);
	return true;
}

void BufferCache::CollectDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                  std::vector<vk::BufferCopy>& copies, uint64_t& total_size) {
	const auto buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    std::lock_guard ranges_lock(m_gpu_ranges_mutex);
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (!copies.empty()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
		MemTrackProf::Add(MemTrackProf::Counter::DownloadBytes, total_size);
		MemTrackProf::Add(MemTrackProf::Counter::DownloadCount, 1);
	}
}

void BufferCache::RecordCollected(Buffer& buffer, std::vector<vk::BufferCopy> copies,
                                  uint64_t total_size, Common::UniqueFunction<void>& publish_out) {
	const auto buffer_address = buffer.CpuAddress();
	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	if (temporary) {
		m_scheduler.DeferRelease([owner = std::move(temporary)]() mutable { owner.reset(); });
	}
	publish_out = [mapped, offset, total_size, buffer_address, download = &download, copies = std::move(copies)] {
		download->Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
}

// KYTY_REPROTECT_BATCH: 1 (default, also when unset) = on, 0 = off, 2 = on with checks; other
// values = off.
static int ReprotectBatchMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_REPROTECT_BATCH");
		const auto  v     = value != nullptr ? std::strtol(value, nullptr, 10) : 1;
		return v == 1 || v == 2 ? static_cast<int>(v) : 0;
	}();
	return mode;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, m_graphics.null_descriptor_enabled ? MemoryUsage::DeviceLocal : MemoryUsage::Stream, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	if (!m_graphics.null_descriptor_enabled) {
		auto& buffer = GetBuffer(null_id);
		std::memset(buffer.Mapped().data(), 0, static_cast<size_t>(buffer.Size()));
		buffer.Flush(0, buffer.Size());
	}
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (graphics.readback_queue != nullptr && AsyncFaultReadbacks()) {
		m_readback_queue = std::make_unique<ReadbackQueue>(graphics, scheduler);
	}
	m_reprotect_batch = ReprotectBatchEnabled() && m_staging_buffer.IsCoherent();
	if (ReprotectBatchMode() == 2) {
		PageManager::SetDeferredVerify(true);
	}
	if (graphics.host_memory_import_enabled) {
		m_import = std::make_unique<HostImport>(graphics);
		if (!m_import->Valid()) {
			m_import.reset();
		}
	}
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	MemTrackProf::Add(MemTrackProf::Counter::InvalidateCount, 1);
	MemTrackProf::Add(MemTrackProf::Counter::InvalidateBytes, size);
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		MemTrackProf::Timer timer(MemTrackProf::Counter::FaultCmdNs, MemTrackProf::Counter::FaultCmdCount);
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			// Cache buffers cover whole tracker pages, so this range has nothing to download.
			const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
			{
				std::lock_guard ranges_lock(m_gpu_ranges_mutex);
				m_gpu_modified_ranges.Subtract(begin, Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - begin);
			}
			m_memory_tracker.ReleaseUnownedRegion(vaddr, size);
			QueueBdaSync(vaddr, size);
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

bool BufferCache::AsyncFaultReadbacks() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_READBACK");
		// Default on; KYTY_ASYNC_READBACK=0 restores synchronous readbacks (and, by default, turns
		// compute preemption and the readback queue off with them).
		return value == nullptr || std::strtol(value, nullptr, 10) != 0;
	}();
	return enabled;
}

namespace {

// KYTY_READBACK_WINDOW_KIB (default 256, rounded down to a power of two, at most 64 MiB; 0
// restores page-sized readbacks): bytes published around a guest memory fault.
uint64_t ReadbackWindowBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_READBACK_WINDOW_KIB");
		const auto  kib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull;
		const auto  size  = std::bit_floor(std::min<uint64_t>(kib, 64ull * 1024ull) * 1024ull);
		return std::max<uint64_t>(size, TRACKER_PAGE_SIZE);
	}();
	return bytes;
}

} // namespace

namespace LateRead {
namespace {

// KYTY_LATE_READ_PROF=N checks one batch in N. Unset with KYTY_MEM_IMPORT on, the safety net of the
// imported buffers samples one batch in KYTY_MEM_IMPORT_NET (default 16, 0 = off).
int SamplePeriod() {
	static const int period = [] {
		const auto number = [](const char* value) {
			return std::max(0, static_cast<int>(std::strtol(value, nullptr, 10)));
		};
		const char* value = std::getenv("KYTY_LATE_READ_PROF");
		if (value != nullptr) {
			return number(value);
		}
		const char* import = std::getenv("KYTY_MEM_IMPORT");
		if (import == nullptr || number(import) == 0) {
			return 0;
		}
		const char* net = std::getenv("KYTY_MEM_IMPORT_NET");
		return net != nullptr ? number(net) : 16;
	}();
	return period;
}

// The detector runs only as the import safety net (KYTY_LATE_READ_PROF unset).
bool NetOnly() {
	static const bool net = std::getenv("KYTY_LATE_READ_PROF") == nullptr;
	return net;
}

uint64_t WfaultDelayNs() {
	static const uint64_t ns = [] {
		const char* value = std::getenv("KYTY_WFAULT_DELAY_US");
		return value != nullptr ? std::strtoull(value, nullptr, 10) * 1000ull : 0ull;
	}();
	return ns;
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Stream: small CPU-modified read copied into the stream buffer. Ring: cache buffer created or grown
// by a stream leap (the CPU-written rings of 13.90). Fixed: any other cache buffer.
enum Path : uint8_t { PathStream, PathRing, PathFixed, PathCount };
constexpr std::array<const char*, PathCount> PathNames {"stream", "ring", "fixed"};
constexpr std::array<const char*, static_cast<size_t>(Site::Count)> SiteNames {"-", "vtx", "idx", "desc",
                                                                               "ind"};

// Ranges up to this size keep a copy of their bytes: a change is then split into differing runs, each
// attributed to the logged writes covering it. Larger ones keep a hash only.
constexpr uint64_t MaxCopy = 1ull << 20u;

struct Entry {
	uint64_t             vaddr = 0, size = 0, hash = 0, seq = 0; // hash 0: unreadable
	Path                 path = PathFixed;
	Site                 site = Site::None;
	std::vector<uint8_t> bytes;                            // size <= MaxCopy
	uint64_t             buffer_vaddr = 0, buffer_size = 0; // imported buffer holding it, else 0
	uint64_t             shader = 0;                        // binding shader (GpuMarkers::Shader)
};

struct Key {
	uint64_t vaddr = 0, size = 0;
	bool     operator==(const Key&) const = default;
};
struct KeyHash {
	size_t operator()(const Key& key) const noexcept {
		return std::hash<uint64_t> {}(key.vaddr * 0x9E3779B97F4A7C15ull ^ key.size);
	}
};

// guest: some changed byte is covered by no logged write. cp: every changed byte is covered, some by
// a command-processor write. backing: covered by backing writes only. big: a hashed-only range changed
// under a logged write (not split).
struct Cell {
	uint64_t n = 0, bytes = 0, guest = 0, guest_bytes = 0, guest_diff = 0, cp = 0, cp_bytes = 0,
	         backing = 0, big = 0, unread = 0;
};

struct Changed {
	uint64_t guest = 0, cp = 0;
	Path     path = PathFixed;
	Site     site = Site::None;
};

struct Write {
	uint64_t seq = 0, vaddr = 0, size = 0;
	bool     cp = false;
};

struct State {
	std::mutex                       mutex; // everything but the write log
	std::vector<Entry>               pending;
	std::unordered_set<Key, KeyHash> seen; // ranges of the open batch
	uint64_t                         batch_index = 0;
	bool                             sampled     = true;
	std::array<std::array<Cell, static_cast<size_t>(Site::Count)>, PathCount> cells {};
	std::unordered_map<Key, Changed, KeyHash> changed;
	uint64_t batches = 0, checked = 0, bind_ns = 0, check_ns = 0, lost = 0, latency_ns = 0;
	std::vector<Key> demote; // imported buffers the guest rewrote under the GPU (back to copies)
	uint64_t         demoted = 0;

	std::mutex            log_mutex;
	std::deque<Write>     log;
	std::atomic<uint64_t> seq {0};
	uint64_t              dropped_seq = 0; // writes up to it left the log
};

State& S() {
	static State state;
	return state;
}

thread_local Site     t_site = Site::None;
std::atomic<uint64_t> g_wfault_delayed {0};

// XXH3 of guest bytes read from the backing (never faults); `overlay`: with the pending labels, as an
// upload recorded now sees them. 0 when unreadable.
uint64_t Hash(uint64_t vaddr, uint64_t size, bool overlay) {
	thread_local std::vector<uint8_t> chunk(1u << 20u);
	thread_local XXH3_state_t*        state = XXH3_createState();
	XXH3_64bits_reset(state);
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min<uint64_t>(size - done, chunk.size());
		if (!LibKernel::Memory::TryReadBacking(vaddr + done, chunk.data(), part)) {
			return 0;
		}
		if (overlay) {
			(void)EopAtGpu::OverlayPending(vaddr + done, chunk.data(), part);
		}
		XXH3_64bits_update(state, chunk.data(), static_cast<size_t>(part));
		done += part;
	}
	return XXH3_64bits_digest(state) | 1u;
}

// GPU thread or CP back thread, inside ObtainBuffer of a read.
void Bind(uint64_t vaddr, uint64_t size, Path path, uint64_t buffer_vaddr = 0, uint64_t buffer_size = 0) {
	if (t_site == Site::None) {
		return;
	}
	auto& st = S();
	{
		std::lock_guard lock(st.mutex);
		if (!st.sampled || !st.seen.insert({vaddr, size}).second) {
			return;
		}
	}
	// Taken before reading: a logged write racing with the read is attributed to its writer.
	Entry entry {vaddr, size, 0, st.seq.load(std::memory_order_acquire), path, t_site, {}, buffer_vaddr,
	             buffer_size, GpuMarkers::Shader()};
	const auto start = NowNs();
	if (size <= MaxCopy) {
		entry.bytes.resize(size);
		if (LibKernel::Memory::TryReadBacking(vaddr, entry.bytes.data(), size)) {
			if (EopAtGpu::Enabled()) {
				(void)EopAtGpu::OverlayPending(vaddr, entry.bytes.data(), size);
			}
			entry.hash = 1;
		}
	} else {
		entry.hash = Hash(vaddr, size, EopAtGpu::Enabled());
	}
	std::lock_guard lock(st.mutex);
	st.bind_ns += NowNs() - start;
	st.pending.push_back(std::move(entry));
}

// The logged writes overlapping the entry since its bind read started; false when the log lost some.
bool WritesSince(const Entry& entry, std::vector<Write>& writes) {
	auto&           st = S();
	std::lock_guard lock(st.log_mutex);
	writes.clear();
	for (auto it = st.log.rbegin(); it != st.log.rend() && it->seq > entry.seq; ++it) {
		if (it->vaddr < entry.vaddr + entry.size && entry.vaddr < it->vaddr + it->size) {
			writes.push_back(*it);
		}
	}
	return entry.seq >= st.dropped_seq;
}

enum Kind : uint8_t { Same, Guest, Cp, Backing, Big, Unreadable };

// A copied entry whose bytes changed: runs of differing bytes (closer than 16 bytes are merged) not
// covered by a logged write are guest writes.
Kind Attribute(const Entry& entry, const uint8_t* now, uint64_t& guest_diff, bool& lost) {
	thread_local std::vector<Write> writes;
	lost       = !WritesSince(entry, writes);
	bool cp    = false;
	guest_diff = 0;
	for (uint64_t i = 0; i < entry.size;) {
		if (now[i] == entry.bytes[i]) {
			i++;
			continue;
		}
		auto end = i + 1;
		for (auto gap = end; gap < entry.size && gap < end + 16; gap++) {
			if (now[gap] != entry.bytes[gap]) {
				end = gap + 1;
			}
		}
		const auto a       = entry.vaddr + i;
		const auto b       = entry.vaddr + end;
		bool       covered = false;
		for (const auto& w : writes) {
			if (w.vaddr < b && a < w.vaddr + w.size) {
				covered = true;
				cp |= w.cp;
			}
		}
		guest_diff += covered ? 0 : end - i;
		i = end;
	}
	return guest_diff != 0 || lost ? Guest : cp ? Cp : Backing;
}

// Priority runner: the GPU completed the batch's work; no label released after it was delivered yet.
void Check(const std::vector<Entry>& batch, uint64_t sealed_ns) {
	thread_local std::vector<uint8_t> now;
	thread_local std::vector<Write>   writes;
	const auto                        start = NowNs();
	std::vector<Kind>                 kinds(batch.size());
	std::vector<uint64_t>             diffs(batch.size());
	uint64_t                          lost_count = 0;
	for (size_t i = 0; i < batch.size(); i++) {
		const auto& entry = batch[i];
		bool        lost  = false;
		if (entry.hash == 0) {
			kinds[i] = Unreadable;
		} else if (entry.bytes.empty()) {
			const auto hash = Hash(entry.vaddr, entry.size, false);
			if (hash == 0) {
				kinds[i] = Unreadable;
			} else if (hash == entry.hash) {
				kinds[i] = Same;
			} else {
				lost     = !WritesSince(entry, writes);
				kinds[i] = !lost && !writes.empty() ? Big : Guest;
				diffs[i] = kinds[i] == Guest ? entry.size : 0;
			}
		} else {
			now.resize(entry.size);
			if (!LibKernel::Memory::TryReadBacking(entry.vaddr, now.data(), entry.size)) {
				kinds[i] = Unreadable;
			} else if (std::memcmp(now.data(), entry.bytes.data(), entry.size) == 0) {
				kinds[i] = Same;
			} else {
				kinds[i] = Attribute(entry, now.data(), diffs[i], lost);
			}
		}
		lost_count += lost ? 1 : 0;
	}
	auto&           st = S();
	std::lock_guard lock(st.mutex);
	st.checked++;
	st.lost += lost_count;
	st.latency_ns += start - sealed_ns;
	for (size_t i = 0; i < batch.size(); i++) {
		const auto& entry = batch[i];
		auto&       cell  = st.cells[entry.path][static_cast<size_t>(entry.site)];
		cell.n++;
		cell.bytes += entry.size;
		switch (kinds[i]) {
			case Guest:
				cell.guest++;
				cell.guest_bytes += entry.size;
				cell.guest_diff += diffs[i];
				break;
			case Cp:
				cell.cp++;
				cell.cp_bytes += entry.size;
				break;
			case Backing: cell.backing++; break;
			case Big: cell.big++; break;
			case Unreadable: cell.unread++; break;
			default: break;
		}
		if (kinds[i] == Guest && entry.buffer_size != 0) {
			st.demote.push_back({entry.buffer_vaddr, entry.buffer_size});
			LOGF("BufferCache: KYTY_MEM_IMPORT net: bound 0x%016" PRIx64 "+0x%" PRIx64 " %s/%s shader=0x%016" PRIx64
			     " changed=%" PRIu64 " bytes, seal->check %.2f ms\n",
			     entry.vaddr, entry.size, PathNames[entry.path], SiteNames[static_cast<size_t>(entry.site)],
			     entry.shader, diffs[i], static_cast<double>(start - sealed_ns) / 1e6);
		}
		if (kinds[i] == Guest || kinds[i] == Cp) {
			auto& changed = st.changed[{entry.vaddr, entry.size}];
			(kinds[i] == Guest ? changed.guest : changed.cp)++;
			changed.path = entry.path;
			changed.site = entry.site;
		}
	}
	st.check_ns += NowNs() - start;
}

void DelayWriteFault() {
	const auto ns = WfaultDelayNs();
	if (ns == 0) {
		return;
	}
	for (const auto end = NowNs() + ns; NowNs() < end;) {
	}
	g_wfault_delayed.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

bool Enabled() noexcept {
	return SamplePeriod() != 0;
}

Scope::Scope(Site site) noexcept: m_previous(t_site) {
	t_site = site;
}

Scope::~Scope() {
	t_site = m_previous;
}

void NoteBackingWrite(uint64_t vaddr, uint64_t size, bool cp) noexcept {
	if (!Enabled() || size == 0) {
		return;
	}
	static constexpr size_t MaxLog = 1u << 20u;
	auto&                   st     = S();
	std::lock_guard         lock(st.log_mutex);
	const auto              seq = st.seq.fetch_add(1, std::memory_order_acq_rel) + 1;
	st.log.push_back({seq, vaddr, size, cp});
	if (st.log.size() > MaxLog) {
		st.dropped_seq = st.log.front().seq;
		st.log.pop_front();
	}
}

void Report(double seconds, uint64_t frames) {
	if (seconds <= 0.0) {
		return;
	}
	const double f       = static_cast<double>(std::max<uint64_t>(frames, 1));
	const auto   delayed = g_wfault_delayed.exchange(0, std::memory_order_relaxed);
	if (WfaultDelayNs() != 0) {
		std::printf("PERF wfault %.1fs: delay=%" PRIu64 "us delayed=%.0f/frame (%.2f ms/frame) frames=%" PRIu64
		            "\n",
		            seconds, WfaultDelayNs() / 1000, static_cast<double>(delayed) / f,
		            static_cast<double>(delayed * WfaultDelayNs()) / 1e6 / f, frames);
	}
	if (!Enabled()) {
		return;
	}
	auto&           st  = S();
	std::lock_guard lock(st.mutex);
	const char*     eop = std::getenv("KYTY_EOP_AT_GPU");
	std::printf("PERF lateread %.1fs period=%d eop=%s: frames=%" PRIu64 " batches=%" PRIu64 " checked=%" PRIu64
	            " bind=%.2fms/frame check=%.2fms/frame latency=%.2fms lost=%" PRIu64 " demoted=%" PRIu64 "\n",
	            seconds, SamplePeriod(), eop != nullptr ? eop : "0", frames, st.batches, st.checked,
	            static_cast<double>(st.bind_ns) / 1e6 / f, static_cast<double>(st.check_ns) / 1e6 / f,
	            st.checked != 0 ? static_cast<double>(st.latency_ns) / 1e6 / static_cast<double>(st.checked)
	                            : 0.0,
	            st.lost, st.demoted);
	for (size_t p = 0; p < PathCount; p++) {
		for (size_t s = 0; s < SiteNames.size(); s++) {
			const auto& c = st.cells[p][s];
			if (c.n == 0) {
				continue;
			}
			std::printf("PERF lateread_cell %s/%s: n=%" PRIu64 " bytes=%" PRIu64 " guest=%" PRIu64
			            " guest_bytes=%" PRIu64 " guest_diff=%" PRIu64 " cp=%" PRIu64 " cp_bytes=%" PRIu64
			            " backing=%" PRIu64 " big=%" PRIu64 " unread=%" PRIu64 "\n",
			            PathNames[p], SiteNames[s], c.n, c.bytes, c.guest, c.guest_bytes, c.guest_diff, c.cp,
			            c.cp_bytes, c.backing, c.big, c.unread);
		}
	}
	std::vector<std::pair<Key, Changed>> top(st.changed.begin(), st.changed.end());
	for (const bool by_guest : {true, false}) {
		const auto count = [by_guest](const Changed& c) { return by_guest ? c.guest : c.cp; };
		std::sort(top.begin(), top.end(),
		          [&](const auto& a, const auto& b) { return count(a.second) > count(b.second); });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 8) && count(top[i].second) != 0; i++) {
			std::printf("PERF lateread_top %s 0x%016" PRIx64 "+0x%" PRIx64 " %s/%s count=%" PRIu64 "\n",
			            by_guest ? "guest" : "cp", top[i].first.vaddr, top[i].first.size,
			            PathNames[top[i].second.path], SiteNames[static_cast<size_t>(top[i].second.site)],
			            count(top[i].second));
		}
	}
	std::fflush(stdout);
	st.cells = {};
	st.changed.clear();
	st.batches = st.checked = st.bind_ns = st.check_ns = st.lost = st.latency_ns = st.demoted = 0;
}

} // namespace LateRead

void BufferCache::SealLateReads() {
	if (!LateRead::Enabled() || (LateRead::NetOnly() && !m_import)) {
		return;
	}
	auto&                        st = LateRead::S();
	std::vector<LateRead::Entry> batch;
	std::vector<LateRead::Key>   demote;
	bool                         sampled = false;
	{
		std::lock_guard lock(st.mutex);
		sampled = st.sampled;
		demote.swap(st.demote);
	}
	// Draws queued on the CP back before this point bind their ranges first; the back is idle for
	// the demotions too.
	if (sampled || !demote.empty()) {
		CpPipeline::Drain();
	}
	for (const auto& key: demote) {
		// Safety net (KYTY_MEM_IMPORT): the guest rewrote bytes of this imported buffer while the
		// GPU could still read them. Its range goes back to device memory and uploads.
		if (m_import && !ImportExcludedRange(key.vaddr, key.size)) {
			LOGF("BufferCache: KYTY_MEM_IMPORT guest rewrote 0x%016" PRIx64 "+0x%" PRIx64
			     " under the GPU: back to device memory\n",
			     key.vaddr, key.size);
			DemoteImported(key.vaddr, key.size);
			std::lock_guard lock(st.mutex);
			st.demoted++;
		}
	}
	{
		std::lock_guard lock(st.mutex);
		batch.swap(st.pending);
		st.seen.clear();
		st.batches++;
		st.sampled = ++st.batch_index % static_cast<uint64_t>(LateRead::SamplePeriod()) == 0;
	}
	if (batch.empty()) {
		return;
	}
	m_scheduler.DeferPriorityOperation(
	    [batch = std::move(batch), sealed = LateRead::NowNs()] { LateRead::Check(batch, sealed); });
}

void BufferCache::ReadFaultMemory(uint64_t vaddr) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		ReadMemory(vaddr, 1);
		return;
	}
	PublishFaultWindow(vaddr, false);
}

void BufferCache::InvalidateFaultMemory(uint64_t vaddr) {
	const bool gpu_thread = GuestGpu::IsGpuThread();
	if (!gpu_thread && CommandScheduler::InDeferredOperation()) {
		InvalidateMemory(vaddr, 1);
		return;
	}
	if (!gpu_thread) {
		LateRead::DelayWriteFault();
	}
	if (!GuestRange {vaddr, 1}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	if (MemTrackProf::Enabled() && MemTrackProf::InFault() && m_memory_tracker.IsPageWatched(vaddr)) {
		MemTrackProf::NoteBufferHit();
	}
	bool flushed = false;
	m_memory_tracker.InvalidateRegion(vaddr, 1, [this, vaddr, &flushed] {
		flushed = true;
		PublishFaultWindow(vaddr, true);
	});
	QueueBdaSync(vaddr, 1);
	if (flushed) {
		MemTrackProf::Add(MemTrackProf::Counter::WriteFlushCount, 1);
	}
	// KYTY_WRITE_FAULT_READAHEAD=N (pages, default 15; 0 disables): a guest thread that write-
	// faults on the page after its previous write fault (a sequential writer, e.g. a constant
	// ring) also gets the next N CPU-clean pages unprotected and marked CPU-modified, up to the
	// first GPU-modified page. Writes that reach them before the next synchronization do not
	// fault, and the run is unprotected and re-protected with one call each.
	static const uint64_t readahead = [] {
		const char* value = std::getenv("KYTY_WRITE_FAULT_READAHEAD");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : 15ull;
	}();
	if (readahead == 0 || gpu_thread || flushed) {
		return;
	}
	static thread_local uint64_t last_page  = 0;
	static thread_local uint64_t next_page  = 0;
	const auto                   page       = vaddr / TRACKER_PAGE_SIZE;
	const bool                   sequential = page == last_page + 1 || page == next_page;
	last_page                               = page;
	if (!sequential) {
		return;
	}
	const auto marked = m_memory_tracker.ExtendCpuModified((page + 1) * TRACKER_PAGE_SIZE,
	                                                       readahead * TRACKER_PAGE_SIZE);
	if (marked != 0) {
		QueueBdaSync((page + 1) * TRACKER_PAGE_SIZE, marked);
		MemTrackProf::Add(MemTrackProf::Counter::ReadAheadPages, marked / TRACKER_PAGE_SIZE);
	}
	next_page         = page + 1 + marked / TRACKER_PAGE_SIZE;
}

void BufferCache::PublishFaultWindow(uint64_t vaddr, bool is_write) {
	if (!GuestGpu::IsGpuThread() && AsyncFaultReadbacks()) {
		// The GPU thread only records and submits the copy; this thread waits for the
		// publication, so other GPU work is not stalled behind the readback.
		std::shared_ptr<ReadbackCompletion> wait_for;
		bool                                own_side    = false;
		uint64_t                            import_wait = 0;
		const auto                          send_start = MemTrackProf::Enabled() ? MemTrackProf::NowNs() : 0;
		const auto                          publish    = [&] {
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, is_write, &wait_for,
		                                                &own_side, &import_wait] {
			MemTrackProf::Timer timer(MemTrackProf::Counter::FaultCmdNs, MemTrackProf::Counter::FaultCmdCount);
			if (is_write && !IsRegionRegistered(vaddr, 1)) {
				// Cache buffers cover whole tracker pages, so this page has nothing to download.
				m_gpu_modified_ranges.Subtract(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
				                               TRACKER_PAGE_SIZE);
				m_memory_tracker.ReleaseUnownedRegion(vaddr, 1);
				QueueBdaSync(vaddr, 1);
				return;
			}
			auto&      buffer = m_slot_buffers[FindBuffer(vaddr, 1)];
			const auto base   = buffer.CpuAddress();
			const auto window = ReadbackWindowBytes();
			const auto begin  = std::max(base, Common::AlignDown(vaddr, window));
			const auto last   = std::min(base + buffer.Size(), Common::AlignDown(vaddr, window) + window);
			if (begin >= last) {
				return;
			}
			// A readback of the same window is already queued: wait for it instead.
			if ((wait_for = FindPendingPublish(begin, last - begin)) != nullptr) {
				return;
			}
			if (buffer.IsImported()) {
				// KYTY_MEM_IMPORT: no copy, the bytes are in guest memory once the writers
				// completed. When they have not, this thread waits for them itself and retries.
				(void)PublishImported(begin, last - begin, false, &import_wait);
				return;
			}
			// Every GPU write to the window already completed: the readback queue copies it
			// at once instead of behind the work queued on the main queue.
			const bool side_ok = m_readback_queue != nullptr && WritesComplete(begin, last - begin);
			const auto backlog = side_ok ? MainQueueBacklog() : 0;
			wait_for           = DownloadForFault(buffer, begin, last - begin, side_ok);
			own_side           = wait_for != nullptr && wait_for->side_tick != 0;
			if (own_side && backlog == 0) {
				// Keep the GPU fed: submit the command buffer being recorded, as the main-queue
				// copy path does.
				m_scheduler.Flush();
			}
			if (wait_for == nullptr) {
				// No GPU-written byte left in the window: nothing to publish.
				m_memory_tracker.UnmarkRegionAsGpuModified(begin, last - begin);
			}
		});
		};
		publish();
		while (import_wait != 0) {
			// The writer was submitted by the command: no lock is held while waiting for it.
			g_import_stats.replayed.fetch_add(1, std::memory_order_relaxed);
			m_scheduler.GetMasterSemaphore().Wait(std::exchange(import_wait, 0));
			publish();
		}
		const auto wait_start = send_start != 0 ? MemTrackProf::NowNs() : 0;
		if (send_start != 0) {
			MemTrackProf::Add(MemTrackProf::Counter::FaultSendNs, wait_start - send_start);
		}
		if (own_side) {
			// Copied on the readback queue: this thread publishes once the copy completed.
			m_readback_queue->Wait(wait_for->side_tick);
			wait_for->side_publish();
		} else if (wait_for != nullptr) {
			wait_for->Wait();
		}
		if (wait_start != 0 && wait_for != nullptr) {
			MemTrackProf::NoteFaultWait(MemTrackProf::NowNs() - wait_start);
		}
		// A write retries against the now write-protected page and takes the CPU-dirty path.
		return;
	}
	const auto send_start = MemTrackProf::Enabled() ? MemTrackProf::NowNs() : 0;
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, is_write] {
		MemTrackProf::Timer timer(MemTrackProf::Counter::FaultCmdNs, MemTrackProf::Counter::FaultCmdCount);
		if (is_write && !IsRegionRegistered(vaddr, 1)) {
			// Cache buffers cover whole tracker pages, so this page has nothing to download.
			{
				std::lock_guard ranges_lock(m_gpu_ranges_mutex);
				m_gpu_modified_ranges.Subtract(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
				                               TRACKER_PAGE_SIZE);
			}
			m_memory_tracker.ReleaseUnownedRegion(vaddr, 1);
			QueueBdaSync(vaddr, 1);
			return;
		}
		auto&      buffer = m_slot_buffers[FindBuffer(vaddr, 1)];
		const auto base   = buffer.CpuAddress();
		// Each fault costs a queue flush and a full timeline wait, and guests walk GPU-written
		// ranges page by page. Publish the GPU-modified bytes of the whole aligned window at
		// once: the neighbouring pages become clean (still write-tracked), so later accesses
		// to them no longer need the GPU. Only the faulting page becomes CPU-modified.
		const auto window = ReadbackWindowBytes();
		const auto begin  = std::max(base, Common::AlignDown(vaddr, window));
		const auto last   = std::min(base + buffer.Size(), Common::AlignDown(vaddr, window) + window);
		// A fault of the GPU thread itself (e.g. indirect arguments read by the command processor)
		// can use the readback queue like ReadMemoryWindow (KYTY_SYNC_SIDE_READS).
		if (begin < last && GuestGpu::IsGpuThread() && ReadWindowOnSideQueue(buffer, begin, last - begin)) {
			// Published (or nothing GPU-written was left in the window).
		} else if (begin < last && DownloadBufferMemory<false>(buffer, begin, last - begin)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(begin, last - begin);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, 1);
			QueueBdaSync(vaddr, 1);
		}
	});
	if (send_start != 0) {
		MemTrackProf::Add(MemTrackProf::Counter::FaultSendNs, MemTrackProf::NowNs() - send_start);
	}
}

bool BufferCache::WriteGpuOwned(uint64_t vaddr, const void* data, uint64_t size) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_OWNED_WRITE_DATA");
		return value == nullptr || std::strtol(value, nullptr, 10) != 0;
	}();
	if (!enabled || size == 0 || !GuestGpu::IsGpuThread() || !GuestRange {vaddr, size}.Valid() ||
	    m_scheduler.Current().IsInvalid() || !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return false;
	}
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return false;
	}
	// A fault readback of these bytes either completes first (they are published and not
	// GPU-owned any more) or has not been recorded yet and copies the buffer with this write.
	WaitPendingPublishes(vaddr, size);
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	LateRead::NoteBackingWrite(vaddr, size, true); // a command-processor write (before and after)
	if (buffer.IsImported()) {
		// The buffer is the guest memory: the bytes land there at this point of the stream, and
		// guest readers wait for it (PublishImported).
		WriteDataBuffer(buffer, vaddr, data, size);
		NoteBufferWrite(vaddr, size);
		g_import_stats.owned_labels.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	const bool written = Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size);
	LateRead::NoteBackingWrite(vaddr, size, true);
	if (!written) {
		return false;
	}
	// The pages stay GPU-modified: a later readback copies the cached buffer, which then holds
	// these bytes in command-stream order with the GPU writes around them.
	WriteDataBuffer(buffer, vaddr, data, size);
	NoteBufferWrite(vaddr, size);
	return true;
}

bool BufferCache::InvalidateCompletedWrite(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	bool gpu_owned = false;
	m_memory_tracker.InvalidateRegion(vaddr, size, [&gpu_owned] { gpu_owned = true; });
	if (gpu_owned) {
		return false;
	}
	QueueBdaSync(vaddr, size);
	return true;
}

void BufferCache::ReadMemoryWindow(uint64_t vaddr, uint64_t size) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BACKING_READ_WINDOW");
		return value == nullptr || std::strtol(value, nullptr, 10) != 0;
	}();
	if (!enabled || !GuestGpu::IsGpuThread()) {
		ReadMemory(vaddr, size);
		return;
	}
	auto&      buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	const auto base   = buffer.CpuAddress();
	const auto window = ReadbackWindowBytes();
	const auto begin  = std::max(base, Common::AlignDown(vaddr, window));
	const auto last   = std::min(base + buffer.Size(), Common::AlignUp(vaddr + size, window));
	if (begin < last && ReadWindowOnSideQueue(buffer, begin, last - begin)) {
		return;
	}
	if (begin < last && DownloadBufferMemory<false>(buffer, begin, last - begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, last - begin);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	GpuCapture::Touch(vaddr, size);
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice
			// versa.
			// KYTY_MEM_IMPORT: a leap stays inside the run of direct memory the range is mapped from,
			// or the buffer could not use the guest's memory; it shrinks (halving) until it fits.
			// KYTY_MEM_IMPORT=2 only: imported rings are read live by work recorded earlier (device
			// lost observed); with 1 the leaped stream buffers stay in device memory.
			static const bool import_leaps = [] {
				const char* value = std::getenv("KYTY_MEM_IMPORT");
				return value != nullptr && std::strtol(value, nullptr, 10) == 2;
			}();
			const bool clamp_leap = m_import != nullptr && import_leaps &&
			                        LibKernel::Memory::BackingHostAddress(begin, end - begin) != 0;
			const auto fitting_leap = [&](uint64_t leap, bool right) {
				if (!clamp_leap) {
					return leap;
				}
				for (; leap >= TRACKER_PAGE_SIZE; leap /= 2) {
					if (LibKernel::Memory::BackingHostAddress(right ? begin - leap : begin,
					                                          end - begin + leap) != 0) {
						return leap;
					}
				}
				return uint64_t {0};
			};
			if (expands_left) {
				end += fitting_leap(std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                               : LibKernel::Memory::kExtendedMemoryBase +
				                                                     LibKernel::Memory::kExtendedMemorySize) -
				                                                  end),
				                    false);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= fitting_leap(std::min(StreamLeapSize, begin - minimum), true);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	if (!new_buffer.IsImported()) {
		new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
		                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	} else if (!overlap.IsImported()) {
		// The new buffer is guest memory: only the GPU-written bytes of the old one are newer (the
		// rest is already there, or older than the guest's own writes).
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		{
			std::lock_guard ranges_lock(m_gpu_ranges_mutex);
			m_gpu_modified_ranges.ForEachInRange(overlap.CpuAddress(), overlap.Size(),
			                                     [&](uint64_t start, uint64_t end) {
				                                     ranges.emplace_back(start, end);
			                                     });
		}
		for (const auto& [start, end]: ranges) {
			new_buffer.CopyFrom(m_scheduler.Current(), overlap, start - overlap.CpuAddress(),
			                    start - new_buffer.CpuAddress(), end - start);
		}
		g_import_stats.join_copies.fetch_add(ranges.size(), std::memory_order_relaxed);
	}
	DeleteBuffer(overlap_id);
}

// KYTY_MEM_IMPORT_EXCLUDE=begin-end[,begin-end...] (hex guest addresses, diagnostic): buffers
// touching these ranges stay in device memory.
static bool ImportExcluded(uint64_t vaddr, uint64_t size) {
	static const std::vector<std::pair<uint64_t, uint64_t>> ranges = [] {
		std::vector<std::pair<uint64_t, uint64_t>> out;
		const char* value = std::getenv("KYTY_MEM_IMPORT_EXCLUDE");
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*      next  = nullptr;
			const auto begin = std::strtoull(at, &next, 16);
			if (next == at || *next != '-') {
				break;
			}
			at             = next + 1;
			const auto end = std::strtoull(at, &next, 16);
			if (next == at) {
				break;
			}
			out.emplace_back(begin, end);
			at = *next == ',' ? next + 1 : next;
		}
		return out;
	}();
	return std::any_of(ranges.begin(), ranges.end(),
	                   [&](const auto& r) { return vaddr < r.second && r.first < vaddr + size; });
}

void BufferCache::DemoteImported(uint64_t vaddr, uint64_t size) {
	m_import_demoted.emplace_back(vaddr, vaddr + size);
	DropImported(vaddr, size);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

// KYTY_MEM_IMPORT_GPU_WRITES=1 keeps GPU-written buffers on imported memory (device lost in game A:
// a compute shader then never completes); by default they go back to device memory at their first
// write binding, and only buffers the GPU reads use guest memory.
static bool ImportGpuWrites() {
	static const bool keep = [] {
		const char* value = std::getenv("KYTY_MEM_IMPORT_GPU_WRITES");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return keep;
}

// KYTY_MEM_IMPORT_SHADERS_ACCESS=r|w (diagnostic): only the listed shaders' reads (r) or writes (w)
// leave imported memory; unset, both do.
static bool ImportShaderAccess(bool is_written) {
	static const int only = [] {
		const char* value = std::getenv("KYTY_MEM_IMPORT_SHADERS_ACCESS");
		return value == nullptr ? 0 : value[0] == 'r' ? 1 : value[0] == 'w' ? 2 : 0;
	}();
	return only == 0 || (only == 1) != is_written;
}

// KYTY_MEM_IMPORT_SHADERS=addr[,addr...] (hex guest shader addresses, local configuration): the
// buffers these compute shaders bind, read or written, leave imported memory for good.
void BufferCache::BeginImportShader(uint64_t cs_address) {
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> out;
		const char*           value = std::getenv("KYTY_MEM_IMPORT_SHADERS");
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*      next    = nullptr;
			const auto address = std::strtoull(at, &next, 16);
			if (next == at) {
				break;
			}
			out.push_back(address);
			at = *next == ',' ? next + 1 : next;
		}
		return out;
	}();
	m_import_shader = m_import && std::find(shaders.begin(), shaders.end(), cs_address) != shaders.end()
	                      ? cs_address
	                      : 0;
}

bool BufferCache::ImportExcludedRange(uint64_t vaddr, uint64_t size) const {
	return ImportExcluded(vaddr, size) ||
	       std::any_of(m_import_demoted.begin(), m_import_demoted.end(),
	                   [&](const auto& r) { return vaddr < r.second && r.first < vaddr + size; });
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);
	const auto flags   = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	const auto bytes   = overlap.end - overlap.begin;

	// KYTY_MEM_IMPORT: a range mapped from one run of direct memory gets the guest's own bytes.
	std::optional<ImportedMemory> imported;
	if (m_import) {
		const auto host = ImportExcludedRange(overlap.begin, bytes)
		                      ? 0
		                      : LibKernel::Memory::BackingHostAddress(overlap.begin, bytes);
		if (host != 0) {
			imported = m_import->Acquire(host, bytes, flags);
		}
		(imported ? g_import_stats.imported : host != 0 ? g_import_stats.failed : g_import_stats.device)
		    .fetch_add(1, std::memory_order_relaxed);
		static std::atomic<uint32_t> logged {0};
		if (!imported && logged.fetch_add(1, std::memory_order_relaxed) < 256) {
			LOGF("BufferCache: KYTY_MEM_IMPORT device buffer guest=0x%016" PRIx64 " size=0x%" PRIx64
			     " request=0x%016" PRIx64 "+0x%" PRIx64 " backed=%d leap=%d joined=%d\n",
			     overlap.begin, bytes, vaddr, size,
			     static_cast<int>(LibKernel::Memory::BackingHostAddress(vaddr, size) != 0),
			     static_cast<int>(overlap.has_stream_leap), static_cast<int>(overlap.first != overlap.last));
		}
		if (imported && ImportLogEnabled()) {
			LOGF("BufferCache: KYTY_MEM_IMPORT imported guest=0x%016" PRIx64 " size=0x%" PRIx64 " leap=%d\n",
			     overlap.begin, bytes, static_cast<int>(overlap.has_stream_leap));
		}
	}
	const auto id =
	    imported ? m_slot_buffers.insert(m_graphics, m_scheduler, overlap.begin, flags, bytes, *imported)
	             : m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal,
	                                     overlap.begin, flags, bytes);
	auto& buffer  = m_slot_buffers[id];
	buffer.leaped = overlap.has_stream_leap;
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer{}[guest=0x{:016x} size=0x{:x}]",
	                     buffer.IsImported() ? ".Imported" : "", overlap.begin, bytes);
	if (overlap.first != overlap.last) {
		NoteBufferWrite(overlap.begin, overlap.end - overlap.begin);
	}
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		buffer.leaped |= m_slot_buffers[old_id].leaped;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

// KYTY_UPLOAD_BARRIER_BATCH=0 restores one barrier pair per upload.
static bool UploadBarrierBatchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UPLOAD_BARRIER_BATCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_UPLOAD_BATCH: 1 (default) = the copies of an upload batch are recorded together when the
// batch flushes, one vkCmdCopyBuffer per source and destination pair, between one leading and one
// trailing barrier, and the draw and dispatch resource uploads run in batches; 0 = one copy per
// upload as recorded, batches only around buffer discovery and BDA synchronization.
int BufferCache::UploadBatchMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_UPLOAD_BATCH");
		return value == nullptr ? 1 : static_cast<int>(std::strtol(value, nullptr, 10) != 0);
	}();
	return UploadBarrierBatchEnabled() ? mode : 0;
}

void BufferCache::BeginUploadBatch() noexcept {
	if (UploadBarrierBatchEnabled()) {
		m_upload_batch_depth++;
	}
}

void BufferCache::EndUploadBatch() {
	if (!UploadBarrierBatchEnabled()) {
		return;
	}
	EXIT_IF(m_upload_batch_depth == 0);
	if (--m_upload_batch_depth == 0) {
		m_batch_ranges.clear();
		if (m_scheduler.Active() && m_scheduler.Current().UploadPostPending()) {
			m_scheduler.Current().FlushUploadPost();
		}
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	MemTrackProf::Timer prof_timer(MemTrackProf::Counter::SyncNs, MemTrackProf::Counter::SyncCount);
	if (buffer.IsImported()) {
		// The GPU reads the guest's bytes at execution: nothing to upload, no re-protection. A
		// write only takes GPU ownership (CPU readers wait for the writer, see PublishImported).
		const auto tick = m_scheduler.CurrentTick();
		for (auto chunk = vaddr >> 16; chunk <= (vaddr + size - 1) >> 16; chunk++) {
			if (decltype(m_read_ticks)::IsValidPage(chunk)) {
				m_read_ticks[chunk] = tick;
			}
		}
		if (is_written) {
			if (ImportLogEnabled() && g_import_logged_writes.insert(vaddr >> 14u).second) {
				LOGF("BufferCache: KYTY_MEM_IMPORT gpu write guest=0x%016" PRIx64 "+0x%" PRIx64
				     " buffer=0x%016" PRIx64 "+0x%" PRIx64 " texel=%d\n",
				     vaddr, size, buffer.CpuAddress(), buffer.Size(), static_cast<int>(is_texel_buffer));
			}
			m_memory_tracker.ForEachUploadRange(
			    vaddr, size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
			g_import_stats.write_syncs.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		g_import_stats.read_syncs.fetch_add(1, std::memory_order_relaxed);
		if (is_texel_buffer) {
			NoteBufferWrite(vaddr, size);
			const bool filled = SynchronizeBufferFromImage(buffer, vaddr, size);
			if (filled) {
				g_import_stats.image_syncs.fetch_add(1, std::memory_order_relaxed);
			}
			return filled;
		}
		return false;
	}
	if (m_reprotect_batch && !is_written) {
		EnsureReprotectBatch();
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size, !is_written); });
	if (source) {
		NoteBufferWrite(vaddr, size, true);
		auto& command = m_scheduler.Current();
		const bool batch = m_upload_batch_depth != 0;
		// Evaluated before any handle use: a pending trailing barrier means the previous recorded
		// work is an upload of this batch, whose leading barrier already covers this one.
		bool joined = batch && command.UploadPostPending();
		command.EndRendering();
		const auto native = batch ? command.HandleForUpload() : command.Handle();
		if (!batch) {
			vk::BufferMemoryBarrier before {};
			before.srcAccessMask = vk::AccessFlagBits::eMemoryRead |
			                       vk::AccessFlagBits::eMemoryWrite |
			                       vk::AccessFlagBits::eTransferRead |
			                       vk::AccessFlagBits::eTransferWrite;
			before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
			before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before.buffer              = buffer.Handle();
			before.offset              = 0;
			before.size                = buffer.Size();
			native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                       vk::PipelineStageFlagBits::eTransfer,
			                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0,
			                       nullptr);
			native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
			                  copies.data());
			auto after          = before;
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                       vk::PipelineStageFlagBits::eAllCommands,
			                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0,
			                       nullptr);
		} else if (UploadBatchMode() != 0) {
			// The copy is recorded with the batch's barriers when the batch flushes.
			command.DeferUploadCopies(source, buffer.Handle(), copies);
		} else {
			// Destination ranges of this batch's copies: a later copy over the same bytes needs an
			// explicit write-after-write dependency (the pending trailing barrier is not one).
			bool overlap = false;
			for (const auto& copy: copies) {
				const uint64_t begin = copy.dstOffset;
				const uint64_t end   = copy.dstOffset + copy.size;
				for (const auto& range: m_batch_ranges) {
					overlap |= range.buffer == buffer.Handle() && begin < range.end &&
					           range.begin < end;
				}
			}
			if (joined && m_batch_ranges.size() > 256) {
				joined = false; // bound the overlap bookkeeping: start a new leading barrier
			}
			if (!joined) {
				m_batch_ranges.clear();
			}
			vk::MemoryBarrier2 shared {};
			if (!joined) {
				// Everything recorded before orders and becomes visible before the batch's copies
				// (global scope: the batch's later copies may target other buffers).
				shared.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
				shared.srcAccessMask =
				    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
				shared.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				shared.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
			} else if (overlap) {
				shared.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				shared.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
				shared.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				shared.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
			}
			if (!joined || overlap) {
				vk::DependencyInfo dependency {};
				dependency.dependencyFlags    = vk::DependencyFlagBits::eByRegion;
				dependency.memoryBarrierCount = 1;
				dependency.pMemoryBarriers    = &shared;
				native.pipelineBarrier2(dependency);
			}
			native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
			                  copies.data());
			for (const auto& copy: copies) {
				m_batch_ranges.push_back(
				    {buffer.Handle(), copy.dstOffset, copy.dstOffset + copy.size});
			}
			command.DeferUploadPost();
		}
	}
	if (is_texel_buffer && !is_written) {
		// May copy an image into the buffer.
		NoteBufferWrite(vaddr, size);
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

static bool ProtectCoalesceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PROTECT_COALESCE");
		return value == nullptr || std::strtol(value, nullptr, 10) != 0;
	}();
	return enabled;
}

bool BufferCache::ReprotectBatchEnabled() noexcept {
	return ReprotectBatchMode() != 0;
}

void BufferCache::EnsureReprotectBatch() {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	if (!m_deferring_uploads) {
		m_deferring_uploads = true;
		m_scheduler.SetPreSubmit(
		    [](void* context) { static_cast<BufferCache*>(context)->FlushDeferredUploads(); }, this);
	}
	if (!PageManager::DeferringProtect()) {
		PageManager::BeginDeferredProtect();
	}
}

void BufferCache::FlushBatchedUploads() {
	if (m_reprotect_batch && m_deferring_uploads) {
		FlushDeferredUploads();
	}
}

bool BufferCache::BeginDeferredUploads() {
	if (m_reprotect_batch) {
		EnsureReprotectBatch();
		return false;
	}
	if (!ProtectCoalesceEnabled() || !m_staging_buffer.IsCoherent() || m_deferring_uploads) {
		return false;
	}
	m_deferring_uploads = true;
	PageManager::BeginDeferredProtect();
	m_scheduler.SetPreSubmit(
	    [](void* context) { static_cast<BufferCache*>(context)->FlushDeferredUploads(); }, this);
	return true;
}

// Order matters: the recorded protections first, then the bytes. A guest write that lands
// before a page's protection is applied is in the copy; a later one faults and is logged.
void BufferCache::FlushDeferredUploads() {
	// One flush at a time: the protections of everything recorded so far, then all their copies.
	std::lock_guard lock(m_deferred_mutex);
	PageManager::FlushDeferredProtect();
	MemTrackProf::Timer prof_timer(MemTrackProf::Counter::UploadCopyNs);
	PerfStats::ScopedTimer timer(PerfStats::Counter::StagingWriteNs);
	for (const auto& copy: m_deferred_copies) {
		PerfStats::Add(PerfStats::Counter::StagingWriteBytes, copy.size);
		PerfStats::Add(PerfStats::Counter::StagingWriteCount, 1);
		if (!LibKernel::Memory::CopySparseMemory(copy.address, copy.destination, copy.size)) {
			EXIT("BufferCache: failed to read sparse guest buffer memory\n");
		}
		(void)EopAtGpu::OverlayPending(copy.address, copy.destination, copy.size);
	}
	if (ReprotectBatchMode() == 2 && !m_deferred_copies.empty()) [[unlikely]] {
		static uint64_t flushes = 0;
		static uint64_t copies  = 0;
		copies += m_deferred_copies.size();
		if ((++flushes & 0x3fffu) == 0) {
			const auto stats = PageManager::GetDeferredVerifyStats();
			std::printf("BufferCache: KYTY_REPROTECT_BATCH=2 flushes=%" PRIu64 " copies=%" PRIu64
			            " pages_checked=%" PRIu64 " mismatches=%" PRIu64 "\n",
			            flushes, copies, stats.checked, stats.mismatch);
			std::fflush(stdout);
		}
	}
	m_deferred_copies.clear();
}

void BufferCache::EndDeferredUploads() {
	FlushDeferredUploads();
	PageManager::EndDeferredProtect();
	m_scheduler.SetPreSubmit(nullptr, nullptr);
	m_deferring_uploads = false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size, bool may_defer) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	MemTrackProf::Add(MemTrackProf::Counter::UploadBytes, total_size);
	MemTrackProf::Add(MemTrackProf::Counter::UploadCopies, copies.size());
	// Only a thread whose re-protections were deferred defers the copy (its protections are
	// applied by the flush, before the copy).
	if (PageManager::DeferringProtect() && m_deferring_uploads && may_defer) {
		// No wait for ring space here (a wait submits): such uploads take the immediate path.
		const auto [mapped, offset] = m_staging_buffer.Map(total_size, 4, false);
		if (mapped != nullptr) {
			std::lock_guard lock(m_deferred_mutex);
			for (auto& copy: copies) {
				m_deferred_copies.push_back(
				    {buffer.CpuAddress() + copy.dstOffset, mapped + copy.srcOffset, copy.size});
				copy.srcOffset += offset;
			}
			m_staging_buffer.Commit(); // coherent: no flush to order after the copies
			return m_staging_buffer.Handle();
		}
		// Immediate path: everything recorded so far is applied first, in order.
		FlushDeferredUploads();
	}

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		MemTrackProf::Timer timer(MemTrackProf::Counter::UploadCopyNs);
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			(void)EopAtGpu::OverlayPending(address, mapped + copy.srcOffset, copy.size);
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	GpuCapture::Touch(vaddr, size, is_written ? GpuCapture::Out::Buffer : GpuCapture::Out::None);
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	const bool imported_range = m_import && !is_written && size <= CACHING_PAGESIZE &&
	                            IsImportedRange(vaddr, size);
	if (imported_range) {
		g_import_stats.stream_skips.fetch_add(1, std::memory_order_relaxed);
	}
	if (!is_written && size <= CACHING_PAGESIZE && !imported_range &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			(void)EopAtGpu::OverlayPending(vaddr, mapped, size);
			m_stream_buffer.Commit();
			if (LateRead::Enabled() && !m_import && !LateRead::NetOnly()) [[unlikely]] {
				LateRead::Bind(vaddr, size, LateRead::PathStream);
			}
			GpuMarkers::NoteBinding(vaddr, size, false, 2);
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	static uint32_t shader_logs = 0;
	if (m_import_shader != 0 && ImportLogEnabled() && (m_slot_buffers[id].IsImported() || shader_logs++ < 64)) {
		LOGF("BufferCache: KYTY_MEM_IMPORT shader 0x%016" PRIx64 " binds 0x%016" PRIx64 "+0x%" PRIx64
		     " written=%d imported=%d\n",
		     m_import_shader, vaddr, size, is_written ? 1 : 0, m_slot_buffers[id].IsImported() ? 1 : 0);
	}
	if (m_slot_buffers[id].IsImported() &&
	    ((m_import_shader != 0 && ImportShaderAccess(is_written)) || (is_written && !ImportGpuWrites()))) {
		const auto& imported = m_slot_buffers[id];
		g_import_stats.write_demotes.fetch_add(1, std::memory_order_relaxed);
		if (ImportLogEnabled()) {
			LOGF("BufferCache: KYTY_MEM_IMPORT gpu %s 0x%016" PRIx64 "+0x%" PRIx64
			     ": buffer 0x%016" PRIx64 "+0x%" PRIx64 " back to device memory\n",
			     is_written ? "write" : "read", vaddr, size, imported.CpuAddress(), imported.Size());
		}
		DemoteImported(imported.CpuAddress(), imported.Size());
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// Imported GPU-written bytes change under the GPU by design: not checked.
	if (!is_written && LateRead::Enabled() &&
	    (m_import ? buffer.IsImported() && !m_memory_tracker.IsRegionGpuModified(vaddr, size)
	              : !LateRead::NetOnly())) [[unlikely]] {
		// With KYTY_MEM_IMPORT only imported ranges are checked (copies are snapshots).
		LateRead::Bind(vaddr, size, buffer.leaped ? LateRead::PathRing : LateRead::PathFixed,
		               buffer.IsImported() ? buffer.CpuAddress() : 0, buffer.IsImported() ? buffer.Size() : 0);
	}
	if (is_written) {
		WaitPendingPublishes(vaddr, size);
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		if (buffer.IsImported()) {
			LateRead::NoteBackingWrite(vaddr, size, true); // the GPU writes guest memory itself
		}
		{
			std::lock_guard ranges_lock(m_gpu_ranges_mutex);
			m_gpu_modified_ranges.Add(vaddr, size);
		}
		ForgetColorMetadataChecks(vaddr, size);
		NoteBufferWrite(vaddr, size);
	}
	GpuMarkers::NoteBinding(vaddr, size, is_written, buffer.IsImported() ? 1 : 0);
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		BeforeCpuWrite(vaddr, size);
		LateRead::NoteBackingWrite(vaddr, size, true);
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		LateRead::NoteBackingWrite(vaddr, size, true);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, RefreshProf::GpuFill);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		BeforeCpuWrite(dst_vaddr, size);
		LateRead::NoteBackingWrite(dst_vaddr, size, true);
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		LateRead::NoteBackingWrite(dst_vaddr, size, true);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, RefreshProf::GpuCopy);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		if (checked.Intersects(vaddr, size)) {
			checked.Subtract(vaddr, size);
			m_color_metadata_epoch.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

void BufferCache::ReadGpuModified() {
	std::vector<std::pair<uint64_t, uint64_t>> dirty;
	for (const auto& [vaddr, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		if (IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			dirty.emplace_back(buffer.CpuAddress(), buffer.Size());
		}
	}
	for (const auto& [vaddr, size]: dirty) {
		ReadMemory(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Dirty bytes only exist on GPU-dirty tracker pages; the page bits are cheaper to test.
	if (GuestGpu::IsGpuThread() && !m_memory_tracker.MaybeGpuModifiedOnGpuThread(vaddr, size)) {
		return false;
	}
	// Bytes of a fault readback still in flight left the dirty set but are not published yet.
	bool intersects = false;
	{
		std::lock_guard ranges_lock(m_gpu_ranges_mutex);
		intersects = m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	return intersects ||
	       (AsyncFaultReadbacks() && FindPendingPublish(vaddr, size) != nullptr);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		WaitPendingPublishes(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitRetired(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	// Take before walking: writes queued during synchronization remain pending.
	const auto pending = TakeBdaSync();
	const bool deferred = BeginDeferredUploads();
	pending.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			SynchronizeBuffersInRange(start, finish - start);
		});
	});
	if (deferred) {
		EndDeferredUploads();
	}
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
