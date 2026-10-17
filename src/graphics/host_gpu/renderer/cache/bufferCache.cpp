#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

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
		EXIT_IF(m_graphics.device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess);
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
		write(mapped, offset);
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
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		m_memory_tracker.QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
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
	if (m_readback_queue == nullptr || size == 0) {
		return;
	}
	if (!m_side_copies.empty()) {
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
	Common::UniqueFunction<void> publish;
	if (!RecordDownload(buffer, vaddr, size, publish)) {
		return false;
	}
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
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
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (!copies.empty()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
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
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			// Cache buffers cover whole tracker pages, so this range has nothing to download.
			const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
			m_gpu_modified_ranges.Subtract(begin, Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - begin);
			m_memory_tracker.ReleaseUnownedRegion(vaddr, size);
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
}

bool BufferCache::AsyncFaultReadbacks() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_READBACK");
		return value != nullptr && std::strtol(value, nullptr, 10) != 0;
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
		return std::max(size, TRACKER_PAGE_SIZE);
	}();
	return bytes;
}

} // namespace

void BufferCache::ReadFaultMemory(uint64_t vaddr) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		ReadMemory(vaddr, 1);
		return;
	}
	PublishFaultWindow(vaddr, false);
}

void BufferCache::InvalidateFaultMemory(uint64_t vaddr) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		InvalidateMemory(vaddr, 1);
		return;
	}
	if (!GuestRange {vaddr, 1}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, 1, [this, vaddr] { PublishFaultWindow(vaddr, true); });
}

void BufferCache::PublishFaultWindow(uint64_t vaddr, bool is_write) {
	if (!GuestGpu::IsGpuThread() && AsyncFaultReadbacks()) {
		// The GPU thread only records and submits the copy; this thread waits for the
		// publication, so other GPU work is not stalled behind the readback.
		std::shared_ptr<ReadbackCompletion> wait_for;
		bool                                own_side = false;
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, is_write, &wait_for,
		                                                &own_side] {
			if (is_write && !IsRegionRegistered(vaddr, 1)) {
				// Cache buffers cover whole tracker pages, so this page has nothing to download.
				m_gpu_modified_ranges.Subtract(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
				                               TRACKER_PAGE_SIZE);
				m_memory_tracker.ReleaseUnownedRegion(vaddr, 1);
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
		if (own_side) {
			// Copied on the readback queue: this thread publishes once the copy completed.
			m_readback_queue->Wait(wait_for->side_tick);
			wait_for->side_publish();
		} else if (wait_for != nullptr) {
			wait_for->Wait();
		}
		// A write retries against the now write-protected page and takes the CPU-dirty path.
		return;
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, 1)) {
			// Cache buffers cover whole tracker pages, so this page has nothing to download.
			m_gpu_modified_ranges.Subtract(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
			                               TRACKER_PAGE_SIZE);
			m_memory_tracker.ReleaseUnownedRegion(vaddr, 1);
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
		if (begin < last && DownloadBufferMemory<false>(buffer, begin, last - begin)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(begin, last - begin);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, 1);
		}
	});
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
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	// The pages stay GPU-modified: a later readback copies the cached buffer, which then holds
	// these bytes in command-stream order with the GPU writes around them.
	WriteDataBuffer(buffer, vaddr, data, size);
	NoteBufferWrite(vaddr, size);
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
	if (begin < last && DownloadBufferMemory<false>(buffer, begin, last - begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(begin, last - begin);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
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
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
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
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	if (overlap.first != overlap.last) {
		NoteBufferWrite(overlap.begin, overlap.end - overlap.begin);
	}
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
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

bool BufferCache::BeginDeferredUploads() {
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
	PageManager::FlushDeferredProtect();
	for (const auto& copy: m_deferred_copies) {
		if (!LibKernel::Memory::CopySparseMemory(copy.address, copy.destination, copy.size)) {
			EXIT("BufferCache: failed to read sparse guest buffer memory\n");
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
	if (m_deferring_uploads && may_defer) {
		// No wait for ring space here (a wait submits): such uploads take the immediate path.
		const auto [mapped, offset] = m_staging_buffer.Map(total_size, 4, false);
		if (mapped != nullptr) {
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
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	if (is_written) {
		WaitPendingPublishes(vaddr, size);
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		NoteBufferWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
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
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
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
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
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
		checked.Subtract(vaddr, size);
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
	return m_gpu_modified_ranges.Intersects(vaddr, size) ||
	       (AsyncFaultReadbacks() && FindPendingPublish(vaddr, size) != nullptr);
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
		EXIT_IF(buffer.is_deleted);
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
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
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
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	// Take before walking: writes queued during synchronization remain pending.
	const auto pending = m_memory_tracker.TakeBdaSync();
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
