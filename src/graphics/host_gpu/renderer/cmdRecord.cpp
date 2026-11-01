#include "graphics/host_gpu/renderer/cmdRecord.h"

#include "common/assert.h"
#include "graphics/host_gpu/renderer/perfStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define KYTY_CMDREC_X86 1
#endif

namespace Libs::Graphics::CmdRecord {

namespace {

inline void CpuRelax() noexcept {
#ifdef KYTY_CMDREC_X86
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

inline uint64_t Ticks() noexcept {
#ifdef KYTY_CMDREC_X86
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

int ReadMode() {
	const char* value = std::getenv("KYTY_CP_RECORD_THREAD");
	if (value == nullptr) {
		return 1;
	}
	const auto mode = std::strtol(value, nullptr, 10);
	return mode >= 0 && mode <= 3 ? static_cast<int>(mode) : 0;
}

// ---------------------------------------------------------------------------------------------
// Entry point table: originals and indices (stats).

enum FuncIndex : uint32_t {
#define KYTY_VK_CMD(name) F_##name,
#include "graphics/host_gpu/renderer/cmdRecordList.inc"
#undef KYTY_VK_CMD
	F_vkBeginCommandBuffer,
	F_vkEndCommandBuffer,
	F_vkResetCommandBuffer,
	F_Device, // device-level calls that drain first (pools, destruction)
	F_Count
};
static_assert(F_Count < 65536);

const char* const g_func_names[F_Count] = {
#define KYTY_VK_CMD(name) #name,
#include "graphics/host_gpu/renderer/cmdRecordList.inc"
#undef KYTY_VK_CMD
    "vkBeginCommandBuffer", "vkEndCommandBuffer", "vkResetCommandBuffer", "device"};

#define KYTY_VK_CMD(name) PFN_##name g_orig_##name = nullptr;
#include "graphics/host_gpu/renderer/cmdRecordList.inc"
#undef KYTY_VK_CMD
PFN_vkBeginCommandBuffer          g_orig_vkBeginCommandBuffer          = nullptr;
PFN_vkEndCommandBuffer            g_orig_vkEndCommandBuffer            = nullptr;
PFN_vkResetCommandBuffer          g_orig_vkResetCommandBuffer          = nullptr;
PFN_vkAllocateCommandBuffers      g_orig_vkAllocateCommandBuffers      = nullptr;
PFN_vkFreeCommandBuffers          g_orig_vkFreeCommandBuffers          = nullptr;
PFN_vkResetCommandPool            g_orig_vkResetCommandPool            = nullptr;
PFN_vkTrimCommandPool             g_orig_vkTrimCommandPool             = nullptr;
PFN_vkDestroyCommandPool          g_orig_vkDestroyCommandPool          = nullptr;
PFN_vkResetQueryPool              g_orig_vkResetQueryPool              = nullptr;
PFN_vkDestroyQueryPool            g_orig_vkDestroyQueryPool            = nullptr;
PFN_vkDestroyImage                g_orig_vkDestroyImage                = nullptr;
PFN_vkDestroyImageView            g_orig_vkDestroyImageView            = nullptr;
PFN_vkDestroyBuffer               g_orig_vkDestroyBuffer               = nullptr;
PFN_vkDestroyBufferView           g_orig_vkDestroyBufferView           = nullptr;
PFN_vkDestroySampler              g_orig_vkDestroySampler              = nullptr;
PFN_vkDestroyPipeline             g_orig_vkDestroyPipeline             = nullptr;
PFN_vkDestroyPipelineLayout       g_orig_vkDestroyPipelineLayout       = nullptr;
PFN_vkDestroyDescriptorSetLayout  g_orig_vkDestroyDescriptorSetLayout  = nullptr;
PFN_vkDestroyDescriptorPool       g_orig_vkDestroyDescriptorPool       = nullptr;
PFN_vkResetDescriptorPool         g_orig_vkResetDescriptorPool         = nullptr;
PFN_vkFreeDescriptorSets          g_orig_vkFreeDescriptorSets          = nullptr;
PFN_vkFreeMemory                  g_orig_vkFreeMemory                  = nullptr;
PFN_vkDestroyEvent                g_orig_vkDestroyEvent                = nullptr;
PFN_vkDestroyShaderModule         g_orig_vkDestroyShaderModule         = nullptr;

// ---------------------------------------------------------------------------------------------
// State.

int  g_mode   = -1;
bool g_submit = false;

std::atomic<VkCommandBuffer> g_target {VK_NULL_HANDLE};

constexpr uint64_t RingBytes = uint64_t {8} << 20u;
constexpr uint64_t MaxCommandBytes = RingBytes / 8;

struct alignas(16) Header {
	void (*run)(Header*) = nullptr; // nullptr: padding up to the end of the ring
	uint32_t size = 0;              // bytes, header included, multiple of 16
	uint16_t func = 0;
	uint16_t tag  = 0;              // KYTY_GPU_TIMING: GpuTag of the producer when published
};
static_assert(sizeof(Header) == 16);

alignas(64) uint8_t* g_ring = nullptr;
alignas(64) std::atomic<uint64_t> g_write {0}; // published end (producer)
alignas(64) std::atomic<uint64_t> g_read {0};  // recorded end (recorder)
alignas(64) std::atomic<bool> g_sleeping {false};
uint64_t                      g_head = 0; // producer: allocated end (== g_write between calls)
uint64_t                      g_read_seen = 0; // producer: last g_read loaded (space checks)
std::mutex                    g_sleep_mutex;
std::condition_variable       g_sleep_cv;

// Stats: producer counters are GPU-thread only; recorder counters are atomics.
struct ProducerStats {
	std::array<uint64_t, F_Count> deferred {};
	std::array<uint64_t, F_Count> trapped {};
	std::array<uint64_t, F_Count> timed_calls {};
	std::array<uint64_t, F_Count> timed_ticks {};
	uint64_t                      bytes        = 0;
	uint64_t                      drains       = 0;
	uint64_t                      drain_waits  = 0;
	uint64_t                      drain_ticks  = 0;
	uint64_t                      space_waits  = 0;
	uint64_t                      space_ticks  = 0;
	uint64_t                      wakes        = 0;
	uint64_t                      foreign      = 0;
};
ProducerStats                g_ps;
std::atomic<uint64_t>        g_rec_busy_ticks {0};
std::atomic<uint64_t>        g_rec_commands {0};
std::atomic<uint64_t>        g_rec_sleeps {0};
std::atomic<uint64_t>        g_foreign_total {0};

// KYTY_GPU_TIMING: replay observer (recorder thread, or the GPU thread in mode 2), set at Install.
void (*g_replay_observer)(Header* header, bool before) = nullptr;

constexpr uint64_t AlignUp(uint64_t value, uint64_t align) noexcept {
	return (value + align - 1) & ~(align - 1);
}

// ---------------------------------------------------------------------------------------------
// Recorder thread.

void Wake() {
	std::lock_guard lock(g_sleep_mutex);
	g_sleep_cv.notify_one();
}

uint64_t Replay(uint64_t from, uint64_t to) {
	uint64_t commands = 0;
	while (from != to) {
		commands++;
		auto* header = reinterpret_cast<Header*>(g_ring + (from % RingBytes));
		if (header->run != nullptr) {
			if (g_replay_observer != nullptr) [[unlikely]] {
				g_replay_observer(header, true);
				header->run(header);
				g_replay_observer(header, false);
			} else {
				header->run(header);
			}
		}
		from += header->size;
	}
	g_rec_commands.fetch_add(commands, std::memory_order_relaxed);
	return from;
}

void RecorderThread() {
	uint64_t read = 0;
	for (;;) {
		uint64_t write = g_write.load(std::memory_order_acquire);
		if (write == read) {
			// Spin ~250 us first: the GPU thread produces calls every few microseconds while busy.
			const auto spin_start = Ticks();
			bool       got        = false;
			for (uint32_t i = 0;; i++) {
				CpuRelax();
				write = g_write.load(std::memory_order_acquire);
				if (write != read) {
					got = true;
					break;
				}
				if ((i & 63u) == 63u && Ticks() - spin_start > 1000000u) {
					break;
				}
			}
			if (!got) {
				std::unique_lock lock(g_sleep_mutex);
				g_sleeping.store(true, std::memory_order_seq_cst);
				write = g_write.load(std::memory_order_seq_cst);
				if (write == read) {
					g_rec_sleeps.fetch_add(1, std::memory_order_relaxed);
					// Publish wakes under the mutex; the timeout is only a safety net.
					g_sleep_cv.wait_for(lock, std::chrono::milliseconds(1), [read] {
						return g_write.load(std::memory_order_acquire) != read;
					});
				}
				g_sleeping.store(false, std::memory_order_relaxed);
				continue;
			}
		}
		const auto t0 = Ticks();
		read          = Replay(read, write);
		g_read.store(read, std::memory_order_release);
		g_rec_busy_ticks.fetch_add(Ticks() - t0, std::memory_order_relaxed);
	}
}

// ---------------------------------------------------------------------------------------------
// Producer side.

inline bool OnGpuThread() noexcept {
	return PerfStats::IsGpuThread();
}

// Calls on the main command buffer from the GPU thread are captured (mode 1, 2).
inline bool Captures(VkCommandBuffer buffer) noexcept {
	return buffer == g_target.load(std::memory_order_relaxed) && buffer != VK_NULL_HANDLE &&
	       OnGpuThread();
}

void DrainFrom(uint64_t until) noexcept {
	if (g_read.load(std::memory_order_acquire) == until) {
		return;
	}
	const auto t0 = Ticks();
	while (g_read.load(std::memory_order_acquire) != until) {
		if (g_sleeping.load(std::memory_order_relaxed)) {
			Wake();
		}
		CpuRelax();
	}
	if (OnGpuThread()) {
		g_ps.drain_waits++;
		g_ps.drain_ticks += Ticks() - t0;
	}
}

// A call that cannot be captured, made on the target buffer or affecting what the ring refers to.
inline void Barrier(VkCommandBuffer buffer, uint32_t func) noexcept {
	if (g_mode != 1 && g_mode != 2) {
		return;
	}
	if (buffer != VK_NULL_HANDLE && buffer != g_target.load(std::memory_order_relaxed)) {
		return;
	}
	if (!OnGpuThread()) {
		if (buffer == VK_NULL_HANDLE) {
			return; // device-level call from another thread: not ordered against the GPU thread
		}
		if (g_foreign_total.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("PERF cmdrec: %s on the main command buffer from another thread\n",
			            g_func_names[func]);
			std::fflush(stdout);
		}
		DrainFrom(g_write.load(std::memory_order_acquire));
		return;
	}
	g_ps.trapped[func]++;
	g_ps.drains++;
	DrainFrom(g_head);
}

// Contiguous space for one command of `bytes` (multiple of 16) in the ring.
uint8_t* Allocate(uint64_t bytes) noexcept {
	const uint64_t offset = g_head % RingBytes;
	const uint64_t pad    = offset + bytes > RingBytes ? RingBytes - offset : 0;
	if (g_head + pad + bytes - g_read_seen > RingBytes &&
	    g_head + pad + bytes - (g_read_seen = g_read.load(std::memory_order_acquire)) > RingBytes) {
		const auto t0 = Ticks();
		while (g_head + pad + bytes - g_read.load(std::memory_order_acquire) > RingBytes) {
			if (g_sleeping.load(std::memory_order_relaxed)) {
				Wake();
			}
			CpuRelax();
		}
		g_read_seen = g_read.load(std::memory_order_acquire);
		g_ps.space_waits++;
		g_ps.space_ticks += Ticks() - t0;
	}
	if (pad != 0) {
		auto* skip = reinterpret_cast<Header*>(g_ring + offset);
		skip->run  = nullptr;
		skip->size = static_cast<uint32_t>(pad);
		g_head += pad;
	}
	return g_ring + (g_head % RingBytes);
}

void Publish(Header* header, uint32_t func) noexcept {
	header->tag = t_gpu_tag;
	g_ps.deferred[func]++;
	g_ps.bytes += header->size;
	g_head += header->size;
	if (g_mode == 2) {
		// Inline replay: the deep copies are used exactly as the recorder thread would.
		const auto read = Replay(g_read.load(std::memory_order_relaxed), g_head);
		g_write.store(g_head, std::memory_order_relaxed);
		g_read.store(read, std::memory_order_relaxed);
		return;
	}
	// Store-load ordering with the recorder going to sleep (it stores g_sleeping, then loads g_write,
	// both seq_cst): either it sees this end, or this load sees it asleep and wakes it under the sleep
	// mutex, so the notification cannot fall between its check and its wait. Explicit, since neither
	// x86 (store buffer) nor ARM64 orders a plain store before a later load.
	g_write.store(g_head, std::memory_order_seq_cst);
	// One notification per sleep.
	if (g_sleeping.load(std::memory_order_seq_cst) &&
	    g_sleeping.exchange(false, std::memory_order_acq_rel)) {
		g_ps.wakes++;
		Wake();
	}
}

// Bump allocation inside one command's block.
class Block {
public:
	explicit Block(uint8_t* base, uint64_t used): m_base(base), m_used(used) {}
	template <typename T>
	T* Copy(const T* source, uint32_t count) noexcept {
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		auto* target = reinterpret_cast<T*>(m_base + m_used);
		std::memcpy(static_cast<void*>(target), source, sizeof(T) * count);
		m_used += AlignUp(sizeof(T) * count, 16);
		return target;
	}

private:
	uint8_t* m_base;
	uint64_t m_used;
};

template <typename T>
constexpr uint64_t Bytes(uint32_t count) noexcept {
	return AlignUp(sizeof(T) * count, 16);
}

// A command object C (Header first) followed by `extra` bytes of arrays.
template <typename C>
C* NewCommand(uint64_t extra, uint32_t func, Block*& block, Block& storage) noexcept {
	const uint64_t head  = AlignUp(sizeof(C), 16);
	const uint64_t total = head + extra;
	auto*          base  = Allocate(total);
	auto*          cmd   = new (base) C {};
	cmd->h.run           = &C::Run;
	cmd->h.size          = static_cast<uint32_t>(total);
	cmd->h.func          = static_cast<uint16_t>(func);
	storage              = Block(base, head);
	block                = &storage;
	return cmd;
}

// ---------------------------------------------------------------------------------------------
// Generic hooks.

template <typename F>
struct Hook;

template <typename R, typename... A>
struct Hook<R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	using Pfn = R(VKAPI_PTR*)(VkCommandBuffer, A...);

	// Not captured: drain first when it concerns the main buffer, then call directly.
	template <Pfn* Orig, uint32_t Func>
	static R VKAPI_CALL Trap(VkCommandBuffer buffer, A... args) {
		Barrier(buffer, Func);
		return (*Orig)(buffer, args...);
	}

	// Mode 3: direct call, timed.
	template <Pfn* Orig, uint32_t Func>
	static R VKAPI_CALL Timed(VkCommandBuffer buffer, A... args) {
		if (!OnGpuThread()) {
			return (*Orig)(buffer, args...);
		}
		const auto t0 = Ticks();
		if constexpr (std::is_void_v<R>) {
			(*Orig)(buffer, args...);
			g_ps.timed_ticks[Func] += Ticks() - t0;
			g_ps.timed_calls[Func]++;
		} else {
			R result = (*Orig)(buffer, args...);
			g_ps.timed_ticks[Func] += Ticks() - t0;
			g_ps.timed_calls[Func]++;
			return result;
		}
	}

	// Captured by value (no pointer arguments).
	template <Pfn* Orig, uint32_t Func>
	struct ByValue {
		Header             h;
		VkCommandBuffer    buffer;
		std::tuple<A...>   args;
		static void Run(Header* header) {
			auto* cmd = reinterpret_cast<ByValue*>(header);
			std::apply([cmd](A... values) { (*Orig)(cmd->buffer, values...); }, cmd->args);
		}
	};

	template <Pfn* Orig, uint32_t Func>
	static void VKAPI_CALL Defer(VkCommandBuffer buffer, A... args) {
		static_assert(std::is_void_v<R>);
		// Handles are pointers to opaque types; input data is always passed as const T*.
		static_assert(((!std::is_pointer_v<A> || !std::is_const_v<std::remove_pointer_t<A>>) && ...),
		              "pointer arguments need a deep copy");
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, args...);
			return;
		}
		using C    = ByValue<Orig, Func>;
		auto* base = Allocate(AlignUp(sizeof(C), 16));
		auto* cmd  = new (base) C {};
		cmd->h.run  = &C::Run;
		cmd->h.size = static_cast<uint32_t>(AlignUp(sizeof(C), 16));
		cmd->h.func = static_cast<uint16_t>(Func);
		cmd->buffer = buffer;
		cmd->args   = std::tuple<A...>(args...);
		Publish(&cmd->h, Func);
	}
};

template <typename F>
struct DeviceHook;

template <typename R, typename D, typename... A>
struct DeviceHook<R(VKAPI_PTR*)(D, A...)> {
	using Pfn = R(VKAPI_PTR*)(D, A...);
	template <Pfn* Orig>
	static R VKAPI_CALL Drain(D device, A... args) {
		Barrier(VK_NULL_HANDLE, F_Device);
		return (*Orig)(device, args...);
	}
};

// ---------------------------------------------------------------------------------------------
// Deep copies.

template <PFN_vkCmdPipelineBarrier* Orig, uint32_t Func>
struct PipelineBarrierCmd {
	Header                 h;
	VkCommandBuffer        buffer;
	VkPipelineStageFlags   src, dst;
	VkDependencyFlags      flags;
	uint32_t               nm, nb, ni;
	VkMemoryBarrier*       m;
	VkBufferMemoryBarrier* b;
	VkImageMemoryBarrier*  i;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<PipelineBarrierCmd*>(header);
		(*Orig)(c->buffer, c->src, c->dst, c->flags, c->nm, c->m, c->nb, c->b, c->ni, c->i);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkPipelineStageFlags src,
	                            VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t nm,
	                            const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b,
	                            uint32_t ni, const VkImageMemoryBarrier* i) {
		bool chained = false;
		for (uint32_t k = 0; k < nm; k++) chained |= m[k].pNext != nullptr;
		for (uint32_t k = 0; k < nb; k++) chained |= b[k].pNext != nullptr;
		for (uint32_t k = 0; k < ni; k++) chained |= i[k].pNext != nullptr;
		if (chained || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, src, dst, flags, nm, m, nb, b, ni, i);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<PipelineBarrierCmd>(Bytes<VkMemoryBarrier>(nm) +
		                                                 Bytes<VkBufferMemoryBarrier>(nb) +
		                                                 Bytes<VkImageMemoryBarrier>(ni),
		                                             Func, block, storage);
		c->buffer = buffer;
		c->src    = src;
		c->dst    = dst;
		c->flags  = flags;
		c->nm     = nm;
		c->nb     = nb;
		c->ni     = ni;
		c->m      = block->Copy(m, nm);
		c->b      = block->Copy(b, nb);
		c->i      = block->Copy(i, ni);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdPipelineBarrier2* Orig, uint32_t Func>
struct PipelineBarrier2Cmd {
	Header           h;
	VkCommandBuffer  buffer;
	VkDependencyInfo info;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<PipelineBarrier2Cmd*>(header);
		(*Orig)(c->buffer, &c->info);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, const VkDependencyInfo* info) {
		bool chained = info == nullptr || info->pNext != nullptr;
		if (!chained) {
			for (uint32_t k = 0; k < info->memoryBarrierCount; k++)
				chained |= info->pMemoryBarriers[k].pNext != nullptr;
			for (uint32_t k = 0; k < info->bufferMemoryBarrierCount; k++)
				chained |= info->pBufferMemoryBarriers[k].pNext != nullptr;
			for (uint32_t k = 0; k < info->imageMemoryBarrierCount; k++)
				chained |= info->pImageMemoryBarriers[k].pNext != nullptr;
		}
		if (chained || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, info);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<PipelineBarrier2Cmd>(
            Bytes<VkMemoryBarrier2>(info->memoryBarrierCount) +
                Bytes<VkBufferMemoryBarrier2>(info->bufferMemoryBarrierCount) +
                Bytes<VkImageMemoryBarrier2>(info->imageMemoryBarrierCount),
            Func, block, storage);
		c->buffer                     = buffer;
		c->info                       = *info;
		c->info.pMemoryBarriers       = block->Copy(info->pMemoryBarriers, info->memoryBarrierCount);
		c->info.pBufferMemoryBarriers =
		    block->Copy(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
		c->info.pImageMemoryBarriers =
		    block->Copy(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdBeginRendering* Orig, uint32_t Func>
struct BeginRenderingCmd {
	Header                    h;
	VkCommandBuffer           buffer;
	VkRenderingInfo           info;
	VkRenderingAttachmentInfo depth;
	VkRenderingAttachmentInfo stencil;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BeginRenderingCmd*>(header);
		(*Orig)(c->buffer, &c->info);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, const VkRenderingInfo* info) {
		bool chained = info == nullptr || info->pNext != nullptr ||
		               (info->pDepthAttachment != nullptr && info->pDepthAttachment->pNext != nullptr) ||
		               (info->pStencilAttachment != nullptr &&
		                info->pStencilAttachment->pNext != nullptr);
		if (!chained) {
			for (uint32_t k = 0; k < info->colorAttachmentCount; k++)
				chained |= info->pColorAttachments[k].pNext != nullptr;
		}
		if (chained || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, info);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BeginRenderingCmd>(
            Bytes<VkRenderingAttachmentInfo>(info->colorAttachmentCount), Func, block, storage);
		c->buffer                 = buffer;
		c->info                   = *info;
		c->info.pColorAttachments = block->Copy(info->pColorAttachments, info->colorAttachmentCount);
		if (info->pDepthAttachment != nullptr) {
			c->depth                 = *info->pDepthAttachment;
			c->info.pDepthAttachment = &c->depth;
		}
		if (info->pStencilAttachment != nullptr) {
			c->stencil                 = *info->pStencilAttachment;
			c->info.pStencilAttachment = &c->stencil;
		}
		Publish(&c->h, Func);
	}
};

bool IsImageDescriptor(VkDescriptorType type) {
	switch (type) {
		case VK_DESCRIPTOR_TYPE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return true;
		default: return false;
	}
}

bool IsBufferDescriptor(VkDescriptorType type) {
	switch (type) {
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return true;
		default: return false;
	}
}

bool IsTexelDescriptor(VkDescriptorType type) {
	return type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
	       type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
}

template <PFN_vkCmdPushDescriptorSetKHR* Orig, uint32_t Func>
struct PushDescriptorSetCmd {
	Header                h;
	VkCommandBuffer       buffer;
	VkPipelineBindPoint   bind_point;
	VkPipelineLayout      layout;
	uint32_t              set;
	uint32_t              count;
	VkWriteDescriptorSet* writes;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<PushDescriptorSetCmd*>(header);
		(*Orig)(c->buffer, c->bind_point, c->layout, c->set, c->count, c->writes);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkPipelineBindPoint bind_point,
	                            VkPipelineLayout layout, uint32_t set, uint32_t count,
	                            const VkWriteDescriptorSet* writes) {
		bool     chained = false;
		uint64_t extra   = Bytes<VkWriteDescriptorSet>(count);
		for (uint32_t k = 0; k < count && !chained; k++) {
			const auto& w = writes[k];
			const auto  n = w.descriptorCount;
			if (w.pNext != nullptr) {
				chained = true;
			} else if (IsImageDescriptor(w.descriptorType)) {
				extra += Bytes<VkDescriptorImageInfo>(n);
			} else if (IsBufferDescriptor(w.descriptorType)) {
				extra += Bytes<VkDescriptorBufferInfo>(n);
			} else if (IsTexelDescriptor(w.descriptorType)) {
				extra += Bytes<VkBufferView>(n);
			} else {
				chained = true;
			}
		}
		if (chained || extra > MaxCommandBytes || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, bind_point, layout, set, count, writes);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<PushDescriptorSetCmd>(extra, Func, block, storage);
		c->buffer     = buffer;
		c->bind_point = bind_point;
		c->layout     = layout;
		c->set        = set;
		c->count      = count;
		c->writes     = block->Copy(writes, count);
		for (uint32_t k = 0; k < count; k++) {
			auto&      w    = c->writes[k];
			const auto n    = w.descriptorCount;
			const auto type = w.descriptorType;
			w.pImageInfo  = IsImageDescriptor(type) ? block->Copy(writes[k].pImageInfo, n) : nullptr;
			w.pBufferInfo = IsBufferDescriptor(type) ? block->Copy(writes[k].pBufferInfo, n) : nullptr;
			w.pTexelBufferView =
			    IsTexelDescriptor(type) ? block->Copy(writes[k].pTexelBufferView, n) : nullptr;
		}
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdBindDescriptorSets* Orig, uint32_t Func>
struct BindDescriptorSetsCmd {
	Header              h;
	VkCommandBuffer     buffer;
	VkPipelineBindPoint bind_point;
	VkPipelineLayout    layout;
	uint32_t            first, count, offset_count;
	VkDescriptorSet*    sets;
	uint32_t*           offsets;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BindDescriptorSetsCmd*>(header);
		(*Orig)(c->buffer, c->bind_point, c->layout, c->first, c->count, c->sets, c->offset_count,
		        c->offsets);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkPipelineBindPoint bind_point,
	                            VkPipelineLayout layout, uint32_t first, uint32_t count,
	                            const VkDescriptorSet* sets, uint32_t offset_count,
	                            const uint32_t* offsets) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, bind_point, layout, first, count, sets, offset_count, offsets);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BindDescriptorSetsCmd>(
            Bytes<VkDescriptorSet>(count) + Bytes<uint32_t>(offset_count), Func, block, storage);
		c->buffer       = buffer;
		c->bind_point   = bind_point;
		c->layout       = layout;
		c->first        = first;
		c->count        = count;
		c->offset_count = offset_count;
		c->sets         = block->Copy(sets, count);
		c->offsets      = block->Copy(offsets, offset_count);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdBindVertexBuffers2* Orig, uint32_t Func>
struct BindVertexBuffers2Cmd {
	Header          h;
	VkCommandBuffer buffer;
	uint32_t        first, count;
	VkBuffer*       buffers;
	VkDeviceSize *  offsets, *sizes, *strides;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BindVertexBuffers2Cmd*>(header);
		(*Orig)(c->buffer, c->first, c->count, c->buffers, c->offsets, c->sizes, c->strides);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, uint32_t first, uint32_t count,
	                            const VkBuffer* buffers, const VkDeviceSize* offsets,
	                            const VkDeviceSize* sizes, const VkDeviceSize* strides) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, first, count, buffers, offsets, sizes, strides);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BindVertexBuffers2Cmd>(
            Bytes<VkBuffer>(count) + 3 * Bytes<VkDeviceSize>(count), Func, block, storage);
		c->buffer  = buffer;
		c->first   = first;
		c->count   = count;
		c->buffers = block->Copy(buffers, count);
		c->offsets = block->Copy(offsets, count);
		c->sizes   = block->Copy(sizes, count);
		c->strides = block->Copy(strides, count);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdBindVertexBuffers* Orig, uint32_t Func>
struct BindVertexBuffersCmd {
	Header          h;
	VkCommandBuffer buffer;
	uint32_t        first, count;
	VkBuffer*       buffers;
	VkDeviceSize*   offsets;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BindVertexBuffersCmd*>(header);
		(*Orig)(c->buffer, c->first, c->count, c->buffers, c->offsets);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, uint32_t first, uint32_t count,
	                            const VkBuffer* buffers, const VkDeviceSize* offsets) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, first, count, buffers, offsets);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BindVertexBuffersCmd>(
            Bytes<VkBuffer>(count) + Bytes<VkDeviceSize>(count), Func, block, storage);
		c->buffer  = buffer;
		c->first   = first;
		c->count   = count;
		c->buffers = block->Copy(buffers, count);
		c->offsets = block->Copy(offsets, count);
		Publish(&c->h, Func);
	}
};

// Calls of the form (buffer, a0, a1, ..., count, const R* regions[, tail]): copies, blits, resolves.
template <typename Pfn, Pfn* Orig, uint32_t Func, typename R, typename... Pre>
struct RegionsCmd {
	Header              h;
	VkCommandBuffer     buffer;
	std::tuple<Pre...>  pre;
	uint32_t            count;
	R*                  regions;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<RegionsCmd*>(header);
		std::apply([c](Pre... values) { (*Orig)(c->buffer, values..., c->count, c->regions); },
		           c->pre);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, Pre... pre, uint32_t count,
	                            const R* regions) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, pre..., count, regions);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<RegionsCmd>(Bytes<R>(count), Func, block, storage);
		c->buffer  = buffer;
		c->pre     = std::tuple<Pre...>(pre...);
		c->count   = count;
		c->regions = block->Copy(regions, count);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdBlitImage* Orig, uint32_t Func>
struct BlitImageCmd {
	Header          h;
	VkCommandBuffer buffer;
	VkImage         src, dst;
	VkImageLayout   src_layout, dst_layout;
	uint32_t        count;
	VkImageBlit*    regions;
	VkFilter        filter;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BlitImageCmd*>(header);
		(*Orig)(c->buffer, c->src, c->src_layout, c->dst, c->dst_layout, c->count, c->regions,
		        c->filter);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout,
	                            VkImage dst, VkImageLayout dst_layout, uint32_t count,
	                            const VkImageBlit* regions, VkFilter filter) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, src, src_layout, dst, dst_layout, count, regions, filter);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BlitImageCmd>(Bytes<VkImageBlit>(count), Func, block, storage);
		c->buffer     = buffer;
		c->src        = src;
		c->dst        = dst;
		c->src_layout = src_layout;
		c->dst_layout = dst_layout;
		c->count      = count;
		c->regions    = block->Copy(regions, count);
		c->filter     = filter;
		Publish(&c->h, Func);
	}
};

// vkCmdClearColorImage / vkCmdClearDepthStencilImage.
template <typename Pfn, Pfn* Orig, uint32_t Func, typename V>
struct ClearImageCmd {
	Header                   h;
	VkCommandBuffer          buffer;
	VkImage                  image;
	VkImageLayout            layout;
	V                        value;
	uint32_t                 count;
	VkImageSubresourceRange* ranges;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<ClearImageCmd*>(header);
		(*Orig)(c->buffer, c->image, c->layout, &c->value, c->count, c->ranges);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
	                            const V* value, uint32_t count,
	                            const VkImageSubresourceRange* ranges) {
		if (value == nullptr || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, image, layout, value, count, ranges);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c = NewCommand<ClearImageCmd>(Bytes<VkImageSubresourceRange>(count), Func, block,
		                                     storage);
		c->buffer = buffer;
		c->image  = image;
		c->layout = layout;
		c->value  = *value;
		c->count  = count;
		c->ranges = block->Copy(ranges, count);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdClearAttachments* Orig, uint32_t Func>
struct ClearAttachmentsCmd {
	Header             h;
	VkCommandBuffer    buffer;
	uint32_t           count, rect_count;
	VkClearAttachment* attachments;
	VkClearRect*       rects;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<ClearAttachmentsCmd*>(header);
		(*Orig)(c->buffer, c->count, c->attachments, c->rect_count, c->rects);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, uint32_t count,
	                            const VkClearAttachment* attachments, uint32_t rect_count,
	                            const VkClearRect* rects) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, count, attachments, rect_count, rects);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<ClearAttachmentsCmd>(
            Bytes<VkClearAttachment>(count) + Bytes<VkClearRect>(rect_count), Func, block, storage);
		c->buffer      = buffer;
		c->count       = count;
		c->rect_count  = rect_count;
		c->attachments = block->Copy(attachments, count);
		c->rects       = block->Copy(rects, rect_count);
		Publish(&c->h, Func);
	}
};

template <PFN_vkCmdPushConstants* Orig, uint32_t Func>
struct PushConstantsCmd {
	Header             h;
	VkCommandBuffer    buffer;
	VkPipelineLayout   layout;
	VkShaderStageFlags stages;
	uint32_t           offset, size;
	uint8_t*           data;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<PushConstantsCmd*>(header);
		(*Orig)(c->buffer, c->layout, c->stages, c->offset, c->size, c->data);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, VkPipelineLayout layout,
	                            VkShaderStageFlags stages, uint32_t offset, uint32_t size,
	                            const void* data) {
		if (size > MaxCommandBytes || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, layout, stages, offset, size, data);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<PushConstantsCmd>(Bytes<uint8_t>(size), Func, block, storage);
		c->buffer = buffer;
		c->layout = layout;
		c->stages = stages;
		c->offset = offset;
		c->size   = size;
		c->data   = block->Copy(static_cast<const uint8_t*>(data), size);
		Publish(&c->h, Func);
	}
};

// (buffer, first, count, const T*) and (buffer, count, const T*).
template <typename Pfn, Pfn* Orig, uint32_t Func, typename T>
struct FirstCountCmd {
	Header          h;
	VkCommandBuffer buffer;
	uint32_t        first, count;
	T*              items;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<FirstCountCmd*>(header);
		(*Orig)(c->buffer, c->first, c->count, c->items);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, uint32_t first, uint32_t count,
	                            const T* items) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, first, count, items);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<FirstCountCmd>(Bytes<T>(count), Func, block, storage);
		c->buffer = buffer;
		c->first  = first;
		c->count  = count;
		c->items  = block->Copy(items, count);
		Publish(&c->h, Func);
	}
};

template <typename Pfn, Pfn* Orig, uint32_t Func, typename T>
struct CountCmd {
	Header          h;
	VkCommandBuffer buffer;
	uint32_t        count;
	T*              items;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<CountCmd*>(header);
		(*Orig)(c->buffer, c->count, c->items);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, uint32_t count, const T* items) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, count, items);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<CountCmd>(Bytes<T>(count), Func, block, storage);
		c->buffer = buffer;
		c->count  = count;
		c->items  = block->Copy(items, count);
		Publish(&c->h, Func);
	}
};

struct BeginBufferCmd {
	Header                   h;
	VkCommandBuffer          buffer;
	VkCommandBufferBeginInfo info;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BeginBufferCmd*>(header);
		EXIT_NOT_IMPLEMENTED(g_orig_vkBeginCommandBuffer(c->buffer, &c->info) != VK_SUCCESS);
	}
	static VkResult VKAPI_CALL Hook(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
		if (info == nullptr || info->pNext != nullptr || info->pInheritanceInfo != nullptr ||
		    !Captures(buffer)) {
			Barrier(buffer, F_vkBeginCommandBuffer);
			return g_orig_vkBeginCommandBuffer(buffer, info);
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BeginBufferCmd>(0, F_vkBeginCommandBuffer, block, storage);
		c->buffer    = buffer;
		c->info      = *info;
		Publish(&c->h, F_vkBeginCommandBuffer);
		return VK_SUCCESS;
	}
};

struct EndBufferCmd {
	Header          h;
	VkCommandBuffer buffer;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<EndBufferCmd*>(header);
		EXIT_NOT_IMPLEMENTED(g_orig_vkEndCommandBuffer(c->buffer) != VK_SUCCESS);
	}
	static VkResult VKAPI_CALL Hook(VkCommandBuffer buffer) {
		if (!Captures(buffer)) {
			Barrier(buffer, F_vkEndCommandBuffer);
			return g_orig_vkEndCommandBuffer(buffer);
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<EndBufferCmd>(0, F_vkEndCommandBuffer, block, storage);
		c->buffer    = buffer;
		Publish(&c->h, F_vkEndCommandBuffer);
		return VK_SUCCESS;
	}
};

template <PFN_vkCmdBeginConditionalRenderingEXT* Orig, uint32_t Func>
struct ConditionalRenderingCmd {
	Header                               h;
	VkCommandBuffer                      buffer;
	VkConditionalRenderingBeginInfoEXT   info;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<ConditionalRenderingCmd*>(header);
		(*Orig)(c->buffer, &c->info);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, const VkConditionalRenderingBeginInfoEXT* info) {
		if (info == nullptr || info->pNext != nullptr || !Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, info);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<ConditionalRenderingCmd>(0, Func, block, storage);
		c->buffer    = buffer;
		c->info      = *info;
		Publish(&c->h, Func);
	}
};

struct CallCmd {
	Header h;
	void (*fn)(void*);
	uint8_t* data;
	static void Run(Header* header) {
		auto* c = reinterpret_cast<CallCmd*>(header);
		c->fn(c->data);
	}
};

template <PFN_vkCmdSetBlendConstants* Orig, uint32_t Func>
struct BlendConstantsCmd {
	Header          h;
	VkCommandBuffer buffer;
	float           constants[4];
	static void Run(Header* header) {
		auto* c = reinterpret_cast<BlendConstantsCmd*>(header);
		(*Orig)(c->buffer, c->constants);
	}
	static void VKAPI_CALL Hook(VkCommandBuffer buffer, const float constants[4]) {
		if (!Captures(buffer)) {
			Barrier(buffer, Func);
			(*Orig)(buffer, constants);
			return;
		}
		Block  storage(nullptr, 0);
		Block* block = nullptr;
		auto*  c     = NewCommand<BlendConstantsCmd>(0, Func, block, storage);
		c->buffer    = buffer;
		std::memcpy(c->constants, constants, sizeof(c->constants));
		Publish(&c->h, Func);
	}
};

// ---------------------------------------------------------------------------------------------
// Installation.

template <typename Pfn>
using HookOf = Hook<Pfn>;

template <typename D>
void InstallTraps(D& d, bool timed) {
#define KYTY_VK_CMD(name)                                                                         \
	g_orig_##name = d.name;                                                                       \
	if (d.name != nullptr) {                                                                      \
		d.name = timed ? &HookOf<PFN_##name>::template Timed<&g_orig_##name, F_##name>            \
		               : &HookOf<PFN_##name>::template Trap<&g_orig_##name, F_##name>;            \
	}
#include "graphics/host_gpu/renderer/cmdRecordList.inc"
#undef KYTY_VK_CMD
#define KYTY_CB_FN(name)                                                                          \
	g_orig_##name = d.name;                                                                       \
	if (d.name != nullptr) {                                                                      \
		d.name = timed ? &HookOf<PFN_##name>::template Timed<&g_orig_##name, F_##name>            \
		               : &HookOf<PFN_##name>::template Trap<&g_orig_##name, F_##name>;            \
	}
	KYTY_CB_FN(vkBeginCommandBuffer)
	KYTY_CB_FN(vkEndCommandBuffer)
	KYTY_CB_FN(vkResetCommandBuffer)
#undef KYTY_CB_FN
	if (timed) {
		return;
	}
#define KYTY_DEV_FN(name)                                                                         \
	g_orig_##name = d.name;                                                                       \
	if (d.name != nullptr) {                                                                      \
		d.name = &DeviceHook<PFN_##name>::template Drain<&g_orig_##name>;                        \
	}
	KYTY_DEV_FN(vkAllocateCommandBuffers)
	KYTY_DEV_FN(vkFreeCommandBuffers)
	KYTY_DEV_FN(vkResetCommandPool)
	KYTY_DEV_FN(vkTrimCommandPool)
	KYTY_DEV_FN(vkDestroyCommandPool)
	KYTY_DEV_FN(vkResetQueryPool)
	KYTY_DEV_FN(vkDestroyQueryPool)
	KYTY_DEV_FN(vkDestroyImage)
	KYTY_DEV_FN(vkDestroyImageView)
	KYTY_DEV_FN(vkDestroyBuffer)
	KYTY_DEV_FN(vkDestroyBufferView)
	KYTY_DEV_FN(vkDestroySampler)
	KYTY_DEV_FN(vkDestroyPipeline)
	KYTY_DEV_FN(vkDestroyPipelineLayout)
	KYTY_DEV_FN(vkDestroyDescriptorSetLayout)
	KYTY_DEV_FN(vkDestroyDescriptorPool)
	KYTY_DEV_FN(vkResetDescriptorPool)
	KYTY_DEV_FN(vkFreeDescriptorSets)
	KYTY_DEV_FN(vkFreeMemory)
	KYTY_DEV_FN(vkDestroyEvent)
	KYTY_DEV_FN(vkDestroyShaderModule)
#undef KYTY_DEV_FN
}

template <typename D>
void InstallCaptures(D& d) {
	// By value.
#define KYTY_BYVAL(name)                                                                          \
	if (d.name != nullptr) {                                                                      \
		d.name = &HookOf<PFN_##name>::template Defer<&g_orig_##name, F_##name>;                   \
	}
	KYTY_BYVAL(vkCmdBindPipeline)
	KYTY_BYVAL(vkCmdBindIndexBuffer)
	KYTY_BYVAL(vkCmdDraw)
	KYTY_BYVAL(vkCmdDrawIndexed)
	KYTY_BYVAL(vkCmdDrawIndirect)
	KYTY_BYVAL(vkCmdDrawIndexedIndirect)
	KYTY_BYVAL(vkCmdDrawIndirectCount)
	KYTY_BYVAL(vkCmdDrawIndexedIndirectCount)
	KYTY_BYVAL(vkCmdDrawMeshTasksEXT)
	KYTY_BYVAL(vkCmdDrawMeshTasksIndirectEXT)
	KYTY_BYVAL(vkCmdDrawMeshTasksIndirectCountEXT)
	KYTY_BYVAL(vkCmdDispatch)
	KYTY_BYVAL(vkCmdDispatchIndirect)
	KYTY_BYVAL(vkCmdDispatchBase)
	KYTY_BYVAL(vkCmdEndRendering)
	KYTY_BYVAL(vkCmdEndRenderingKHR)
	KYTY_BYVAL(vkCmdFillBuffer)
	KYTY_BYVAL(vkCmdResetQueryPool)
	KYTY_BYVAL(vkCmdBeginQuery)
	KYTY_BYVAL(vkCmdEndQuery)
	KYTY_BYVAL(vkCmdWriteTimestamp)
	KYTY_BYVAL(vkCmdWriteTimestamp2)
	KYTY_BYVAL(vkCmdCopyQueryPoolResults)
	KYTY_BYVAL(vkCmdSetDepthBias)
	KYTY_BYVAL(vkCmdSetDepthBiasEnable)
	KYTY_BYVAL(vkCmdSetDepthBiasEnableEXT)
	KYTY_BYVAL(vkCmdSetDepthBounds)
	KYTY_BYVAL(vkCmdSetDepthBoundsTestEnable)
	KYTY_BYVAL(vkCmdSetDepthBoundsTestEnableEXT)
	KYTY_BYVAL(vkCmdSetDepthCompareOp)
	KYTY_BYVAL(vkCmdSetDepthCompareOpEXT)
	KYTY_BYVAL(vkCmdSetDepthTestEnable)
	KYTY_BYVAL(vkCmdSetDepthTestEnableEXT)
	KYTY_BYVAL(vkCmdSetDepthWriteEnable)
	KYTY_BYVAL(vkCmdSetDepthWriteEnableEXT)
	KYTY_BYVAL(vkCmdSetLineWidth)
	KYTY_BYVAL(vkCmdSetStencilCompareMask)
	KYTY_BYVAL(vkCmdSetStencilWriteMask)
	KYTY_BYVAL(vkCmdSetStencilReference)
	KYTY_BYVAL(vkCmdSetStencilOp)
	KYTY_BYVAL(vkCmdSetStencilOpEXT)
	KYTY_BYVAL(vkCmdSetStencilTestEnable)
	KYTY_BYVAL(vkCmdSetStencilTestEnableEXT)
	KYTY_BYVAL(vkCmdSetAttachmentFeedbackLoopEnableEXT)
	KYTY_BYVAL(vkCmdSetCullMode)
	KYTY_BYVAL(vkCmdSetCullModeEXT)
	KYTY_BYVAL(vkCmdSetFrontFace)
	KYTY_BYVAL(vkCmdSetFrontFaceEXT)
	KYTY_BYVAL(vkCmdSetPrimitiveTopology)
	KYTY_BYVAL(vkCmdSetPrimitiveTopologyEXT)
	KYTY_BYVAL(vkCmdSetPrimitiveRestartEnable)
	KYTY_BYVAL(vkCmdSetPrimitiveRestartEnableEXT)
	KYTY_BYVAL(vkCmdSetRasterizerDiscardEnable)
	KYTY_BYVAL(vkCmdSetRasterizerDiscardEnableEXT)
	KYTY_BYVAL(vkCmdEndConditionalRenderingEXT)
#undef KYTY_BYVAL

	if (d.vkCmdPipelineBarrier != nullptr)
		d.vkCmdPipelineBarrier =
		    &PipelineBarrierCmd<&g_orig_vkCmdPipelineBarrier, F_vkCmdPipelineBarrier>::Hook;
	if (d.vkCmdPipelineBarrier2 != nullptr)
		d.vkCmdPipelineBarrier2 =
		    &PipelineBarrier2Cmd<&g_orig_vkCmdPipelineBarrier2, F_vkCmdPipelineBarrier2>::Hook;
	if (d.vkCmdPipelineBarrier2KHR != nullptr)
		d.vkCmdPipelineBarrier2KHR =
		    &PipelineBarrier2Cmd<&g_orig_vkCmdPipelineBarrier2KHR, F_vkCmdPipelineBarrier2KHR>::Hook;
	if (d.vkCmdBeginRendering != nullptr)
		d.vkCmdBeginRendering =
		    &BeginRenderingCmd<&g_orig_vkCmdBeginRendering, F_vkCmdBeginRendering>::Hook;
	if (d.vkCmdBeginRenderingKHR != nullptr)
		d.vkCmdBeginRenderingKHR =
		    &BeginRenderingCmd<&g_orig_vkCmdBeginRenderingKHR, F_vkCmdBeginRenderingKHR>::Hook;
	if (d.vkCmdPushDescriptorSetKHR != nullptr)
		d.vkCmdPushDescriptorSetKHR =
		    &PushDescriptorSetCmd<&g_orig_vkCmdPushDescriptorSetKHR, F_vkCmdPushDescriptorSetKHR>::Hook;
	if (d.vkCmdPushDescriptorSet != nullptr)
		d.vkCmdPushDescriptorSet =
		    &PushDescriptorSetCmd<&g_orig_vkCmdPushDescriptorSet, F_vkCmdPushDescriptorSet>::Hook;
	if (d.vkCmdBindDescriptorSets != nullptr)
		d.vkCmdBindDescriptorSets =
		    &BindDescriptorSetsCmd<&g_orig_vkCmdBindDescriptorSets, F_vkCmdBindDescriptorSets>::Hook;
	if (d.vkCmdBindVertexBuffers2 != nullptr)
		d.vkCmdBindVertexBuffers2 =
		    &BindVertexBuffers2Cmd<&g_orig_vkCmdBindVertexBuffers2, F_vkCmdBindVertexBuffers2>::Hook;
	if (d.vkCmdBindVertexBuffers2EXT != nullptr)
		d.vkCmdBindVertexBuffers2EXT =
		    &BindVertexBuffers2Cmd<&g_orig_vkCmdBindVertexBuffers2EXT,
		                           F_vkCmdBindVertexBuffers2EXT>::Hook;
	if (d.vkCmdBindVertexBuffers != nullptr)
		d.vkCmdBindVertexBuffers =
		    &BindVertexBuffersCmd<&g_orig_vkCmdBindVertexBuffers, F_vkCmdBindVertexBuffers>::Hook;
	if (d.vkCmdCopyBuffer != nullptr)
		d.vkCmdCopyBuffer = &RegionsCmd<PFN_vkCmdCopyBuffer, &g_orig_vkCmdCopyBuffer,
		                                F_vkCmdCopyBuffer, VkBufferCopy, VkBuffer, VkBuffer>::Hook;
	if (d.vkCmdCopyBufferToImage != nullptr)
		d.vkCmdCopyBufferToImage =
		    &RegionsCmd<PFN_vkCmdCopyBufferToImage, &g_orig_vkCmdCopyBufferToImage,
		                F_vkCmdCopyBufferToImage, VkBufferImageCopy, VkBuffer, VkImage,
		                VkImageLayout>::Hook;
	if (d.vkCmdCopyImageToBuffer != nullptr)
		d.vkCmdCopyImageToBuffer =
		    &RegionsCmd<PFN_vkCmdCopyImageToBuffer, &g_orig_vkCmdCopyImageToBuffer,
		                F_vkCmdCopyImageToBuffer, VkBufferImageCopy, VkImage, VkImageLayout,
		                VkBuffer>::Hook;
	if (d.vkCmdCopyImage != nullptr)
		d.vkCmdCopyImage = &RegionsCmd<PFN_vkCmdCopyImage, &g_orig_vkCmdCopyImage, F_vkCmdCopyImage,
		                               VkImageCopy, VkImage, VkImageLayout, VkImage,
		                               VkImageLayout>::Hook;
	if (d.vkCmdResolveImage != nullptr)
		d.vkCmdResolveImage =
		    &RegionsCmd<PFN_vkCmdResolveImage, &g_orig_vkCmdResolveImage, F_vkCmdResolveImage,
		                VkImageResolve, VkImage, VkImageLayout, VkImage, VkImageLayout>::Hook;
	if (d.vkCmdBlitImage != nullptr)
		d.vkCmdBlitImage = &BlitImageCmd<&g_orig_vkCmdBlitImage, F_vkCmdBlitImage>::Hook;
	if (d.vkCmdClearColorImage != nullptr)
		d.vkCmdClearColorImage =
		    &ClearImageCmd<PFN_vkCmdClearColorImage, &g_orig_vkCmdClearColorImage,
		                   F_vkCmdClearColorImage, VkClearColorValue>::Hook;
	if (d.vkCmdClearDepthStencilImage != nullptr)
		d.vkCmdClearDepthStencilImage =
		    &ClearImageCmd<PFN_vkCmdClearDepthStencilImage, &g_orig_vkCmdClearDepthStencilImage,
		                   F_vkCmdClearDepthStencilImage, VkClearDepthStencilValue>::Hook;
	if (d.vkCmdClearAttachments != nullptr)
		d.vkCmdClearAttachments =
		    &ClearAttachmentsCmd<&g_orig_vkCmdClearAttachments, F_vkCmdClearAttachments>::Hook;
	if (d.vkCmdPushConstants != nullptr)
		d.vkCmdPushConstants =
		    &PushConstantsCmd<&g_orig_vkCmdPushConstants, F_vkCmdPushConstants>::Hook;
	if (d.vkCmdSetViewport != nullptr)
		d.vkCmdSetViewport = &FirstCountCmd<PFN_vkCmdSetViewport, &g_orig_vkCmdSetViewport,
		                                    F_vkCmdSetViewport, VkViewport>::Hook;
	if (d.vkCmdSetScissor != nullptr)
		d.vkCmdSetScissor = &FirstCountCmd<PFN_vkCmdSetScissor, &g_orig_vkCmdSetScissor,
		                                   F_vkCmdSetScissor, VkRect2D>::Hook;
	if (d.vkCmdSetViewportWithCount != nullptr)
		d.vkCmdSetViewportWithCount =
		    &CountCmd<PFN_vkCmdSetViewportWithCount, &g_orig_vkCmdSetViewportWithCount,
		              F_vkCmdSetViewportWithCount, VkViewport>::Hook;
	if (d.vkCmdSetViewportWithCountEXT != nullptr)
		d.vkCmdSetViewportWithCountEXT =
		    &CountCmd<PFN_vkCmdSetViewportWithCountEXT, &g_orig_vkCmdSetViewportWithCountEXT,
		              F_vkCmdSetViewportWithCountEXT, VkViewport>::Hook;
	if (d.vkCmdSetScissorWithCount != nullptr)
		d.vkCmdSetScissorWithCount =
		    &CountCmd<PFN_vkCmdSetScissorWithCount, &g_orig_vkCmdSetScissorWithCount,
		              F_vkCmdSetScissorWithCount, VkRect2D>::Hook;
	if (d.vkCmdSetScissorWithCountEXT != nullptr)
		d.vkCmdSetScissorWithCountEXT =
		    &CountCmd<PFN_vkCmdSetScissorWithCountEXT, &g_orig_vkCmdSetScissorWithCountEXT,
		              F_vkCmdSetScissorWithCountEXT, VkRect2D>::Hook;
	if (d.vkCmdSetColorWriteEnableEXT != nullptr)
		d.vkCmdSetColorWriteEnableEXT =
		    &CountCmd<PFN_vkCmdSetColorWriteEnableEXT, &g_orig_vkCmdSetColorWriteEnableEXT,
		              F_vkCmdSetColorWriteEnableEXT, VkBool32>::Hook;
	if (d.vkCmdSetBlendConstants != nullptr)
		d.vkCmdSetBlendConstants =
		    &BlendConstantsCmd<&g_orig_vkCmdSetBlendConstants, F_vkCmdSetBlendConstants>::Hook;
	if (d.vkCmdBeginConditionalRenderingEXT != nullptr)
		d.vkCmdBeginConditionalRenderingEXT =
		    &ConditionalRenderingCmd<&g_orig_vkCmdBeginConditionalRenderingEXT,
		                             F_vkCmdBeginConditionalRenderingEXT>::Hook;
	if (g_submit) {
		d.vkBeginCommandBuffer = &BeginBufferCmd::Hook;
		d.vkEndCommandBuffer   = &EndBufferCmd::Hook;
	}
}

// ---------------------------------------------------------------------------------------------
// KYTY_GPU_TIMING=1 (local measurement build, default off; modes 1 and 2).
// - Every main command buffer gets a TOP_OF_PIPE timestamp after vkBeginCommandBuffer and a
//   BOTTOM_OF_PIPE one before vkEndCommandBuffer: busy time (union of buffer spans) and idle gaps
//   between consecutive buffers on the queue ("PERF gput edges").
// - In 1 of every KYTY_GPU_TIMING_EVERY (4) main buffers, the replay also writes a BOTTOM_OF_PIPE
//   timestamp after each draw, dispatch, barrier, copy, blit, resolve, fill, clear and pass edge.
//   The delta between two consecutive timestamps is charged to the command just replayed: its
//   category, its emulator tag (GpuTagScope), its pipeline, its render pass and, for barriers, its
//   stage masks. This partitions the buffer's GPU execution time.
// - Barrier counts (per tag and stage masks, full-pipeline ones, image layout transitions) are
//   taken on every main buffer.
// Results are read without waiting, at later buffer begins. The recorded commands and their order
// are unchanged; only query resets and timestamp writes are added.
namespace GpuTiming {

constexpr uint32_t Cap      = 16384;
constexpr uint32_t Slots    = 8;
constexpr uint32_t EdgeRing = 4096;
constexpr uint32_t Tags     = static_cast<uint32_t>(GpuTag::Count);

enum Cat : uint8_t {
	C_None,
	C_Draw,
	C_DrawMesh,
	C_DrawIndirect,
	C_Dispatch,
	C_DispatchIndirect,
	C_Barrier,
	C_Copy,
	C_Blit,
	C_Resolve,
	C_Fill,
	C_Clear,
	C_ClearAtt,
	C_Pass,
	C_Count
};
const char* const g_cat_names[C_Count] = {"none",    "draw",  "draw_mesh", "draw_indirect", "dispatch",
                                          "dispatch_indirect", "barrier", "copy", "blit", "resolve",
                                          "fill",    "clear", "clear_att", "pass_edge"};

struct Entry {
	uint16_t func;
	uint16_t tag;
	uint32_t pass;
	uint64_t pipe;  // pipeline handle (draws, dispatches); (src << 32) | dst stages (barriers)
	uint64_t work;  // vertex or group invocations (0 if indirect)
};

struct Slot {
	VkQueryPool           pool    = VK_NULL_HANDLE;
	uint32_t              count   = 0;
	bool                  pending = false;
	uint64_t              last    = 0; // latest timestamp of the previous use (stale results are older)
	std::vector<Entry>    entries;
	std::vector<uint64_t> ts;
};

struct PipeAcc {
	double   ns   = 0;
	uint64_t n    = 0;
	uint64_t work = 0;
	uint16_t tag  = 0;
	uint8_t  cat  = 0;
};
struct BarrierKey {
	uint16_t tag;
	uint32_t src, dst;
	bool     operator<(const BarrierKey& o) const noexcept {
        return std::tie(tag, src, dst) < std::tie(o.tag, o.src, o.dst);
	}
};
struct BarrierAcc {
	double   ns     = 0; // sampled buffers
	uint64_t timed  = 0; // sampled buffers
	uint64_t n      = 0; // all buffers
	uint64_t images = 0; // image layout transitions, all buffers
};

struct Acc {
	std::array<std::array<double, Tags>, C_Count>   cat_ns {};
	std::array<std::array<uint64_t, Tags>, C_Count> cat_n {};
	std::unordered_map<uint64_t, PipeAcc>           pipe;
	std::unordered_map<uint32_t, std::pair<double, uint64_t>> pass; // draws per pass key
	std::map<BarrierKey, BarrierAcc>                barriers;
	uint64_t barriers_all = 0, barriers_full = 0, barriers_src_all = 0, image_transitions = 0;
	double   total_ns = 0;
	uint64_t sampled  = 0;
	uint64_t buffers  = 0;
	uint64_t busy     = 0; // sampled buffer skipped: every slot still pending
	// Edges (all buffers).
	uint64_t edge_buffers = 0, edge_lost = 0;
	double   edge_busy_ns = 0, edge_raw_ns = 0, edge_span_ns = 0;
	std::array<double, 7>   gap_ns {};
	std::array<uint64_t, 7> gap_n {};
	// Gaps before buffers submitted while the queue was idle / still busy / not known.
	std::array<double, 3>   gap_kind_ns {};
	std::array<uint64_t, 3> gap_kind_n {};
	std::array<uint64_t, 3> buffers_kind {};
};
// Gap buckets: <20 us, <100 us, <1 ms, <5 ms, <20 ms, <100 ms, >=100 ms.
constexpr double g_gap_limits_ns[6] = {20e3, 100e3, 1e6, 5e6, 20e6, 100e6};

bool                     g_on        = false;
uint32_t                 g_every     = 4;
double                   g_period_ns = 1.0;
VkDevice                 g_device    = VK_NULL_HANDLE;
std::array<Cat, F_Count> g_cat {};
Slot                     g_slots[Slots];
int                      g_cur        = -1;
VkCommandBuffer          g_cur_buffer = VK_NULL_HANDLE;
uint64_t                 g_gfx_pipe = 0, g_cs_pipe = 0;
uint32_t                 g_cur_pass = 0;
std::mutex               g_mutex; // g_acc and g_pass_names (replay thread vs report)
Acc                      g_acc;
std::vector<std::string> g_pass_names {"none"};
std::unordered_map<std::string, uint32_t> g_pass_ids; // replay thread only
std::mutex                                g_label_mutex;
std::unordered_map<uint64_t, std::string> g_labels;

// Edges.
VkQueryPool g_edge_pool = VK_NULL_HANDLE;
uint64_t    g_edge_next = 0;   // buffers begun
uint64_t    g_edge_done = 0;   // buffers read (or lost)
bool        g_edge_open = false;
std::array<uint8_t, EdgeRing> g_edge_kind {}; // 0 unknown, 1 queue idle at submit, 2 queue busy at submit
uint64_t    g_edge_last_begin = 0, g_edge_last_end = 0, g_edge_first = 0;

void Init(VkDevice device) {
	const char* value = std::getenv("KYTY_GPU_TIMING");
	g_on              = value != nullptr && std::atoi(value) != 0;
	if (!g_on) {
		return;
	}
	if (const char* every = std::getenv("KYTY_GPU_TIMING_EVERY"); every != nullptr && std::atoi(every) > 0) {
		g_every = static_cast<uint32_t>(std::atoi(every));
	}
	g_device = device;
	for (uint32_t i = 0; i < F_Count; i++) {
		const std::string name = g_func_names[i];
		const auto        has  = [&](const char* prefix) { return name.rfind(prefix, 0) == 0; };
		const bool indirect = name.find("Indirect") != std::string::npos;
		g_cat[i] = has("vkCmdDrawMeshTasks")                                 ? C_DrawMesh
		           : has("vkCmdDraw") && indirect                            ? C_DrawIndirect
		           : has("vkCmdDraw")                                        ? C_Draw
		           : has("vkCmdDispatch") && indirect                        ? C_DispatchIndirect
		           : has("vkCmdDispatch")                                    ? C_Dispatch
		           : has("vkCmdPipelineBarrier")                             ? C_Barrier
		           : has("vkCmdCopyQueryPoolResults")                        ? C_Copy
		           : has("vkCmdCopy")                                        ? C_Copy
		           : has("vkCmdBlit")                                        ? C_Blit
		           : has("vkCmdResolve")                                     ? C_Resolve
		           : has("vkCmdFillBuffer") || has("vkCmdUpdateBuffer")      ? C_Fill
		           : has("vkCmdClearAttachments")                            ? C_ClearAtt
		           : has("vkCmdClear")                                       ? C_Clear
		           : has("vkCmdBeginRendering") || has("vkCmdEndRendering")  ? C_Pass
		                                                                     : C_None;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	for (auto& slot: g_slots) {
		VkQueryPoolCreateInfo info {};
		info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
		info.queryCount = Cap;
		EXIT_NOT_IMPLEMENTED(d.vkCreateQueryPool(device, &info, nullptr, &slot.pool) != VK_SUCCESS);
		slot.entries.resize(Cap);
		slot.ts.resize(Cap);
	}
	VkQueryPoolCreateInfo info {};
	info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	info.queryCount = EdgeRing * 2;
	EXIT_NOT_IMPLEMENTED(d.vkCreateQueryPool(device, &info, nullptr, &g_edge_pool) != VK_SUCCESS);
}

void PollEdges() {
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	// The open buffer (begun, not ended) is never read.
	const uint64_t limit = g_edge_next - (g_edge_open ? 1 : 0);
	while (g_edge_done < limit) {
		const auto index = static_cast<uint32_t>(g_edge_done % EdgeRing);
		uint64_t   ts[2] {};
		const auto result = d.vkGetQueryPoolResults(g_device, g_edge_pool, index * 2, 2, sizeof(ts), ts,
		                                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		// Before the GPU runs the buffer's reset, the values of the ring's previous use are still
		// there: older than the last accepted begin.
		const bool fresh = result == VK_SUCCESS && ts[0] >= g_edge_last_begin && ts[1] >= ts[0];
		if (!fresh) {
			if (limit - g_edge_done > EdgeRing / 2) {
				// Never executed (or unreadable): drop it so the ring keeps moving.
				std::lock_guard lock(g_mutex);
				g_acc.edge_lost++;
				g_edge_done++;
				continue;
			}
			return;
		}
		std::lock_guard lock(g_mutex);
		auto&      a     = g_acc;
		const auto begin = ts[0], end = ts[1];
		if (g_edge_first == 0) {
			g_edge_first    = begin;
			g_edge_last_end = begin;
		}
		a.edge_buffers++;
		a.edge_raw_ns += static_cast<double>(end - begin) * g_period_ns;
		const uint32_t kind = g_edge_kind[index] == 1 ? 0u : g_edge_kind[index] == 2 ? 1u : 2u;
		a.buffers_kind[kind]++;
		if (begin > g_edge_last_end) {
			const double gap = static_cast<double>(begin - g_edge_last_end) * g_period_ns;
			if (gap < 100e6) {
				a.gap_kind_ns[kind] += gap;
				a.gap_kind_n[kind]++;
			}
			uint32_t     b   = 0;
			while (b < 6 && gap >= g_gap_limits_ns[b]) b++;
			a.gap_ns[b] += gap;
			a.gap_n[b]++;
		}
		const auto from = std::max(begin, g_edge_last_end);
		if (end > from) {
			a.edge_busy_ns += static_cast<double>(end - from) * g_period_ns;
		}
		if (end > g_edge_last_end) {
			a.edge_span_ns += static_cast<double>(end - std::max(g_edge_last_end, g_edge_first)) * g_period_ns;
			g_edge_last_end = end;
		}
		g_edge_last_begin = begin;
		g_edge_done++;
	}
}

void Poll() {
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	PollEdges();
	for (auto& slot: g_slots) {
		if (!slot.pending) {
			continue;
		}
		const auto result = d.vkGetQueryPoolResults(g_device, slot.pool, 0, slot.count,
		                                            slot.count * sizeof(uint64_t), slot.ts.data(),
		                                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		// Before the GPU runs the buffer's query reset, the previous use's values are still available.
		if (result != VK_SUCCESS || slot.ts[0] <= slot.last) {
			continue;
		}
		slot.pending = false;
		slot.last    = *std::max_element(slot.ts.begin(), slot.ts.begin() + slot.count);
		std::lock_guard lock(g_mutex);
		auto&           a = g_acc;
		for (uint32_t i = 1; i < slot.count; i++) {
			if (slot.ts[i] < slot.ts[i - 1]) {
				continue;
			}
			const double ns  = static_cast<double>(slot.ts[i] - slot.ts[i - 1]) * g_period_ns;
			const auto&  e   = slot.entries[i];
			const auto   cat = g_cat[e.func];
			const auto   tag = std::min<uint32_t>(e.tag, Tags - 1);
			a.cat_ns[cat][tag] += ns;
			a.cat_n[cat][tag]++;
			if (cat == C_Barrier) {
				auto& b = a.barriers[{static_cast<uint16_t>(tag), static_cast<uint32_t>(e.pipe >> 32u),
				                      static_cast<uint32_t>(e.pipe)}];
				b.ns += ns;
				b.timed++;
			} else if (e.pipe != 0) {
				auto& p = a.pipe[e.pipe];
				p.ns += ns;
				p.n++;
				p.work += e.work;
				p.tag = e.tag;
				p.cat = cat;
			}
			if (cat == C_Draw || cat == C_DrawMesh || cat == C_DrawIndirect) {
				auto& p = a.pass[e.pass];
				p.first += ns;
				p.second++;
			}
		}
		if (slot.count > 1 && slot.ts[slot.count - 1] >= slot.ts[0]) {
			a.total_ns += static_cast<double>(slot.ts[slot.count - 1] - slot.ts[0]) * g_period_ns;
		}
		a.sampled++;
	}
}

void Stamp(Slot& slot, uint32_t func, uint16_t tag, uint64_t pipe, uint64_t work) {
	if (slot.count >= Cap) {
		return;
	}
	g_orig_vkCmdWriteTimestamp(g_cur_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.pool, slot.count);
	slot.entries[slot.count] = {static_cast<uint16_t>(func), tag, g_cur_pass, pipe, work};
	slot.count++;
}

// Stage masks of a barrier (lower 32 bits of the 64-bit masks of barrier2) and its image layout
// transitions.
void BarrierMasks(Header* header, uint32_t& src, uint32_t& dst, uint32_t& images) {
	src = dst = images = 0;
	if (header->func == F_vkCmdPipelineBarrier) {
		using C = PipelineBarrierCmd<&g_orig_vkCmdPipelineBarrier, F_vkCmdPipelineBarrier>;
		auto* c = reinterpret_cast<C*>(header);
		src     = c->src;
		dst     = c->dst;
		for (uint32_t k = 0; k < c->ni; k++) images += c->i[k].oldLayout != c->i[k].newLayout ? 1u : 0u;
		return;
	}
	using C = PipelineBarrier2Cmd<&g_orig_vkCmdPipelineBarrier2, F_vkCmdPipelineBarrier2>;
	const auto& info = reinterpret_cast<C*>(header)->info;
	for (uint32_t k = 0; k < info.memoryBarrierCount; k++) {
		src |= static_cast<uint32_t>(info.pMemoryBarriers[k].srcStageMask);
		dst |= static_cast<uint32_t>(info.pMemoryBarriers[k].dstStageMask);
	}
	for (uint32_t k = 0; k < info.bufferMemoryBarrierCount; k++) {
		src |= static_cast<uint32_t>(info.pBufferMemoryBarriers[k].srcStageMask);
		dst |= static_cast<uint32_t>(info.pBufferMemoryBarriers[k].dstStageMask);
	}
	for (uint32_t k = 0; k < info.imageMemoryBarrierCount; k++) {
		const auto& b = info.pImageMemoryBarriers[k];
		src |= static_cast<uint32_t>(b.srcStageMask);
		dst |= static_cast<uint32_t>(b.dstStageMask);
		images += b.oldLayout != b.newLayout ? 1u : 0u;
	}
}

uint32_t PassId(Header* header) {
	using C = BeginRenderingCmd<&g_orig_vkCmdBeginRendering, F_vkCmdBeginRendering>;
	const auto& info    = reinterpret_cast<C*>(header)->info;
	bool        resolve = false;
	uint32_t    clears  = 0;
	for (uint32_t k = 0; k < info.colorAttachmentCount; k++) {
		resolve |= info.pColorAttachments[k].resolveMode != VK_RESOLVE_MODE_NONE;
		clears += info.pColorAttachments[k].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1u : 0u;
	}
	const bool depth = info.pDepthAttachment != nullptr && info.pDepthAttachment->imageView != VK_NULL_HANDLE;
	char       key[96];
	std::snprintf(key, sizeof(key), "%ux%u c%u%s%s%s", info.renderArea.extent.width, info.renderArea.extent.height,
	              info.colorAttachmentCount, depth ? " d" : "", resolve ? " resolve" : "",
	              clears != 0 ? " clear" : "");
	const auto [it, inserted] = g_pass_ids.try_emplace(key, static_cast<uint32_t>(g_pass_ids.size() + 1));
	if (inserted) {
		std::lock_guard lock(g_mutex);
		g_pass_names.emplace_back(key);
	}
	return it->second;
}

void Observe(Header* header, bool before) {
	const uint32_t func = header->func;
	if (func == F_vkBeginCommandBuffer) {
		if (before) {
			return;
		}
		static uint64_t buffers = 0;
		buffers++;
		{
			std::lock_guard lock(g_mutex);
			g_acc.buffers++;
		}
		Poll();
		g_cur        = -1;
		g_cur_pass   = 0;
		g_cur_buffer = reinterpret_cast<BeginBufferCmd*>(header)->buffer;
		// Edge stamps (every buffer).
		const auto edge = static_cast<uint32_t>(g_edge_next % EdgeRing);
		g_orig_vkCmdResetQueryPool(g_cur_buffer, g_edge_pool, edge * 2, 2);
		g_orig_vkCmdWriteTimestamp(g_cur_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_edge_pool, edge * 2);
		g_edge_kind[edge] = 0;
		g_edge_next++;
		g_edge_open = true;
		if (buffers % g_every != 0) {
			return;
		}
		for (int i = 0; i < static_cast<int>(Slots); i++) {
			if (!g_slots[i].pending) {
				g_cur = i;
				break;
			}
		}
		if (g_cur < 0) {
			std::lock_guard lock(g_mutex);
			g_acc.busy++;
			return;
		}
		auto& slot = g_slots[g_cur];
		slot.count = 0;
		g_orig_vkCmdResetQueryPool(g_cur_buffer, slot.pool, 0, Cap);
		Stamp(slot, F_vkBeginCommandBuffer, 0, 0, 0);
		return;
	}
	if (func == F_vkEndCommandBuffer) {
		if (before) {
			if (g_cur >= 0) {
				auto& slot   = g_slots[g_cur];
				slot.pending = slot.count > 1;
				g_cur        = -1;
			}
			if (g_edge_open) {
				const auto edge = static_cast<uint32_t>((g_edge_next - 1) % EdgeRing);
				g_orig_vkCmdWriteTimestamp(g_cur_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_edge_pool,
				                           edge * 2 + 1);
				g_edge_open = false;
			}
		}
		return;
	}
	if (before) {
		return;
	}
	if (func == F_vkCmdBindPipeline) {
		using C          = Hook<PFN_vkCmdBindPipeline>::ByValue<&g_orig_vkCmdBindPipeline, F_vkCmdBindPipeline>;
		auto*      cmd   = reinterpret_cast<C*>(header);
		const auto point = std::get<0>(cmd->args);
		const auto pipe  = reinterpret_cast<uint64_t>(std::get<1>(cmd->args));
		(point == VK_PIPELINE_BIND_POINT_COMPUTE ? g_cs_pipe : g_gfx_pipe) = pipe;
		return;
	}
	if (func >= F_Count || g_cat[func] == C_None) {
		return;
	}
	const auto cat = g_cat[func];
	if (cat == C_Pass && (func == F_vkCmdBeginRendering || func == F_vkCmdBeginRenderingKHR)) {
		g_cur_pass = PassId(header);
	}
	uint64_t pipe = 0;
	if (cat == C_Barrier) {
		uint32_t src = 0, dst = 0, images = 0;
		BarrierMasks(header, src, dst, images);
		pipe = (static_cast<uint64_t>(src) << 32u) | dst;
		const auto      tag = static_cast<uint16_t>(std::min<uint32_t>(header->tag, Tags - 1));
		std::lock_guard lock(g_mutex);
		auto&           b = g_acc.barriers[{tag, src, dst}];
		b.n++;
		b.images += images;
		g_acc.barriers_all++;
		g_acc.image_transitions += images;
		constexpr uint32_t all = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		g_acc.barriers_src_all += (src & all) != 0 ? 1u : 0u;
		g_acc.barriers_full += (src & all) != 0 && (dst & all) != 0 ? 1u : 0u;
	}
	if (g_cur < 0) {
		if (cat == C_Pass && func != F_vkCmdBeginRendering && func != F_vkCmdBeginRenderingKHR) {
			g_cur_pass = 0;
		}
		return;
	}
	auto&    slot = g_slots[g_cur];
	uint64_t work = 0;
	if (func == F_vkCmdDrawIndexed) {
		using C   = Hook<PFN_vkCmdDrawIndexed>::ByValue<&g_orig_vkCmdDrawIndexed, F_vkCmdDrawIndexed>;
		auto* cmd = reinterpret_cast<C*>(header);
		work      = uint64_t {std::get<0>(cmd->args)} * std::get<1>(cmd->args);
	} else if (func == F_vkCmdDraw) {
		using C   = Hook<PFN_vkCmdDraw>::ByValue<&g_orig_vkCmdDraw, F_vkCmdDraw>;
		auto* cmd = reinterpret_cast<C*>(header);
		work      = uint64_t {std::get<0>(cmd->args)} * std::get<1>(cmd->args);
	} else if (func == F_vkCmdDrawMeshTasksEXT) {
		using C   = Hook<PFN_vkCmdDrawMeshTasksEXT>::ByValue<&g_orig_vkCmdDrawMeshTasksEXT, F_vkCmdDrawMeshTasksEXT>;
		auto* cmd = reinterpret_cast<C*>(header);
		work      = uint64_t {std::get<0>(cmd->args)} * std::get<1>(cmd->args) * std::get<2>(cmd->args);
	} else if (func == F_vkCmdDispatch) {
		using C   = Hook<PFN_vkCmdDispatch>::ByValue<&g_orig_vkCmdDispatch, F_vkCmdDispatch>;
		auto* cmd = reinterpret_cast<C*>(header);
		work      = uint64_t {std::get<0>(cmd->args)} * std::get<1>(cmd->args) * std::get<2>(cmd->args);
	}
	if (cat != C_Barrier) {
		const bool gfx = cat == C_Draw || cat == C_DrawMesh || cat == C_DrawIndirect;
		const bool cs  = cat == C_Dispatch || cat == C_DispatchIndirect;
		pipe           = gfx ? g_gfx_pipe : cs ? g_cs_pipe : 0;
	}
	Stamp(slot, func, header->tag, pipe, work);
	if (cat == C_Pass && func != F_vkCmdBeginRendering && func != F_vkCmdBeginRenderingKHR) {
		g_cur_pass = 0;
	}
}

std::string StageNames(uint32_t mask) {
	static const std::pair<uint32_t, const char*> names[] = {
	    {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, "top"},
	    {VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, "indirect"},
	    {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, "vin"},
	    {VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, "vs"},
	    {VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT, "hs"},
	    {VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT, "ds"},
	    {VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, "gs"},
	    {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, "fs"},
	    {VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, "early"},
	    {VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, "late"},
	    {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, "color"},
	    {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "cs"},
	    {VK_PIPELINE_STAGE_TRANSFER_BIT, "xfer"},
	    {VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, "bottom"},
	    {VK_PIPELINE_STAGE_HOST_BIT, "host"},
	    {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, "ALLGFX"},
	    {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, "ALL"},
	    {VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT, "task"},
	    {VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, "mesh"},
	    {VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT, "cond"},
	};
	std::string out;
	for (const auto& [bit, name]: names) {
		if ((mask & bit) != 0) {
			if (!out.empty()) out += '+';
			out += name;
			mask &= ~bit;
		}
	}
	if (mask != 0) {
		char buf[16];
		std::snprintf(buf, sizeof(buf), "%s0x%x", out.empty() ? "" : "+", mask);
		out += buf;
	}
	return out.empty() ? "none" : out;
}

void Report(double seconds) {
	if (!g_on) {
		return;
	}
	Acc                      acc;
	std::vector<std::string> pass_names;
	{
		std::lock_guard lock(g_mutex);
		acc   = std::move(g_acc);
		g_acc = {};
		pass_names = g_pass_names;
	}
	const auto per_s = [seconds](double ns) { return ns / seconds / 1e6; };
	std::printf("PERF gput edges %.1fs: buffers=%" PRIu64 " lost=%" PRIu64 " busy=%.1fms/s raw=%.1fms/s span=%.1fms/s"
	            " gaps=%.1fms/s(%" PRIu64 ") <20us=%.2f(%" PRIu64 ") <100us=%.2f(%" PRIu64 ") <1ms=%.2f(%" PRIu64
	            ") <5ms=%.2f(%" PRIu64 ") <20ms=%.2f(%" PRIu64 ") <100ms=%.2f(%" PRIu64 ") >=100ms=%.2f(%" PRIu64 ")\n",
	            seconds, acc.edge_buffers, acc.edge_lost, per_s(acc.edge_busy_ns), per_s(acc.edge_raw_ns),
	            per_s(acc.edge_span_ns),
	            per_s(acc.gap_ns[0] + acc.gap_ns[1] + acc.gap_ns[2] + acc.gap_ns[3] + acc.gap_ns[4] + acc.gap_ns[5] +
	                  acc.gap_ns[6]),
	            acc.gap_n[0] + acc.gap_n[1] + acc.gap_n[2] + acc.gap_n[3] + acc.gap_n[4] + acc.gap_n[5] + acc.gap_n[6],
	            per_s(acc.gap_ns[0]), acc.gap_n[0], per_s(acc.gap_ns[1]), acc.gap_n[1], per_s(acc.gap_ns[2]),
	            acc.gap_n[2], per_s(acc.gap_ns[3]), acc.gap_n[3], per_s(acc.gap_ns[4]), acc.gap_n[4],
	            per_s(acc.gap_ns[5]), acc.gap_n[5], per_s(acc.gap_ns[6]), acc.gap_n[6]);
	std::printf("PERF gput gapkind %.1fs: starved=%.2fms/s(%" PRIu64 " of %" PRIu64 ") queued=%.2fms/s(%" PRIu64
	            " of %" PRIu64 ") unknown=%.2fms/s(%" PRIu64 " of %" PRIu64 ")\n",
	            seconds, per_s(acc.gap_kind_ns[0]), acc.gap_kind_n[0], acc.buffers_kind[0], per_s(acc.gap_kind_ns[1]),
	            acc.gap_kind_n[1], acc.buffers_kind[1], per_s(acc.gap_kind_ns[2]), acc.gap_kind_n[2], acc.buffers_kind[2]);
	// Barrier counts (all buffers, per second).
	std::printf("PERF gput barriers %.1fs: n=%.0f/s full=%.0f/s src_all=%.0f/s image_transitions=%.0f/s\n", seconds,
	            static_cast<double>(acc.barriers_all) / seconds, static_cast<double>(acc.barriers_full) / seconds,
	            static_cast<double>(acc.barriers_src_all) / seconds,
	            static_cast<double>(acc.image_transitions) / seconds);
	if (acc.sampled == 0) {
		std::printf("PERF gput %.1fs: buffers=%" PRIu64 " sampled=0 busy=%" PRIu64 "\n", seconds, acc.buffers,
		            acc.busy);
		return;
	}
	// Per second of wall time, scaled from the sampled buffers to all captured buffers.
	const double ratio = static_cast<double>(acc.buffers) / static_cast<double>(acc.sampled);
	const double scale = ratio / seconds / 1e6;
	std::string  line;
	char         buf[320];
	for (uint32_t c = 1; c < C_Count; c++) {
		double   ns = 0;
		uint64_t n  = 0;
		for (uint32_t t = 0; t < Tags; t++) {
			ns += acc.cat_ns[c][t];
			n += acc.cat_n[c][t];
		}
		if (n == 0) continue;
		std::snprintf(buf, sizeof(buf), " %s=%.2fms(%.0f)", g_cat_names[c], ns * scale,
		              static_cast<double>(n) * ratio / seconds);
		line += buf;
	}
	std::printf("PERF gput %.1fs: buffers=%" PRIu64 " sampled=%" PRIu64 " busy=%" PRIu64 " gpu=%.1fms/s%s\n",
	            seconds, acc.buffers, acc.sampled, acc.busy, acc.total_ns * scale, line.c_str());
	line.clear();
	for (uint32_t c = 1; c < C_Count; c++) {
		for (uint32_t t = 0; t < Tags; t++) {
			if (acc.cat_n[c][t] == 0) continue;
			std::snprintf(buf, sizeof(buf), " %s/%s=%.2fms(%.0f)", g_cat_names[c], GpuTagName(static_cast<GpuTag>(t)),
			              acc.cat_ns[c][t] * scale, static_cast<double>(acc.cat_n[c][t]) * ratio / seconds);
			line += buf;
		}
	}
	std::printf("PERF gput tags (per s):%s\n", line.c_str());
	// Barriers by tag and stage masks: count (all buffers) and time (sampled, scaled).
	{
		std::vector<std::pair<double, BarrierKey>> v;
		for (const auto& [k, b]: acc.barriers) v.emplace_back(b.ns * scale + 1e-9 * static_cast<double>(b.n), k);
		std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		for (size_t i = 0; i < v.size() && i < 16; i++) {
			const auto& k = v[i].second;
			const auto& b = acc.barriers[k];
			std::printf("PERF gput barrier %2zu: %.2fms/s n=%.0f/s timed=%.0f/s avg=%.1fus images=%.0f/s %s %s -> %s\n", i,
			            b.ns * scale, static_cast<double>(b.n) / seconds,
			            static_cast<double>(b.timed) * ratio / seconds,
			            b.timed != 0 ? b.ns / 1e3 / static_cast<double>(b.timed) : 0.0,
			            static_cast<double>(b.images) / seconds, GpuTagName(static_cast<GpuTag>(k.tag)),
			            StageNames(k.src).c_str(), StageNames(k.dst).c_str());
		}
	}
	// Draw time per render pass key.
	{
		std::vector<std::pair<double, uint32_t>> v;
		for (const auto& [id, p]: acc.pass) v.emplace_back(p.first, id);
		std::sort(v.begin(), v.end(), std::greater<>());
		for (size_t i = 0; i < v.size() && i < 16; i++) {
			const auto& p = acc.pass[v[i].second];
			std::printf("PERF gput pass %2zu: %.2fms/s draws=%.0f/s avg=%.1fus %s\n", i, p.first * scale,
			            static_cast<double>(p.second) * ratio / seconds,
			            p.first / 1e3 / static_cast<double>(p.second),
			            v[i].second < pass_names.size() ? pass_names[v[i].second].c_str() : "?");
		}
	}
	std::vector<std::pair<double, uint64_t>> pipes;
	for (const auto& [pipe, v]: acc.pipe) pipes.emplace_back(v.ns, pipe);
	std::sort(pipes.begin(), pipes.end(), std::greater<>());
	std::lock_guard lock(g_label_mutex);
	for (size_t i = 0; i < pipes.size() && i < 40; i++) {
		const auto  pipe  = pipes[i].second;
		const auto& v     = acc.pipe[pipe];
		const auto  label = g_labels.find(pipe);
		std::printf("PERF gput pipe %2zu: %.2fms/s n=%.0f/s avg=%.1fus work=%.0f/s ns_per_work=%.2f %s tag=%s h=%" PRIx64 " %s\n", i,
		            v.ns * scale, static_cast<double>(v.n) * ratio / seconds, v.ns / 1e3 / static_cast<double>(v.n),
		            static_cast<double>(v.work) * ratio / seconds,
		            v.work > 0 ? v.ns / static_cast<double>(v.work) : 0.0, g_cat_names[v.cat],
		            GpuTagName(static_cast<GpuTag>(v.tag)), pipe, label != g_labels.end() ? label->second.c_str() : "?");
	}
}

} // namespace GpuTiming

} // namespace

int Mode() noexcept {
	if (g_mode < 0) {
		g_mode = ReadMode();
	}
	return g_mode;
}

void Install(vk::Device device) {
	(void)device;
	if (Mode() == 0) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	InstallTraps(d, Mode() == 3);
	if (Mode() == 1 || Mode() == 2) {
		const char* submit = std::getenv("KYTY_CP_RECORD_SUBMIT");
		g_submit           = submit == nullptr || std::strtol(submit, nullptr, 10) != 0;
		g_ring = static_cast<uint8_t*>(::operator new(RingBytes, std::align_val_t {64}));
		InstallCaptures(d);
	}
	if (Mode() == 1 || Mode() == 2) {
		GpuTiming::Init(static_cast<VkDevice>(device));
		if (GpuTiming::g_on) {
			g_replay_observer = &GpuTiming::Observe;
		}
	}
	if (Mode() == 1) {
		std::thread(RecorderThread).detach();
	}
	std::printf("PERF cmdrec: mode %d installed (ring %" PRIu64 " MiB, submit %d)\n", Mode(),
	            RingBytes >> 20u, g_submit ? 1 : 0);
	std::fflush(stdout);
}

void SetTarget(vk::CommandBuffer buffer) noexcept {
	if (g_mode == 1 || g_mode == 2) {
		g_target.store(static_cast<VkCommandBuffer>(buffer), std::memory_order_relaxed);
	}
}

void Drain() noexcept {
	if ((g_mode == 1 || g_mode == 2) && OnGpuThread()) {
		g_ps.drains++;
		DrainFrom(g_head);
	}
}

void DrainAll() noexcept {
	if (g_mode == 1 || g_mode == 2) {
		DrainFrom(g_write.load(std::memory_order_acquire));
	}
}

bool DefersSubmit() noexcept {
	return (g_mode == 1 || g_mode == 2) && g_submit && OnGpuThread();
}

void Call(void (*fn)(void* data), const void* data, size_t bytes) {
	EXIT_IF(bytes > 65536 || !OnGpuThread());
	if (g_mode != 1 && g_mode != 2) {
		std::vector<uint8_t> copy(static_cast<const uint8_t*>(data),
		                          static_cast<const uint8_t*>(data) + bytes);
		fn(copy.data());
		return;
	}
	Block  storage(nullptr, 0);
	Block* block = nullptr;
	auto*  c     = NewCommand<CallCmd>(Bytes<uint8_t>(static_cast<uint32_t>(bytes)), F_Device,
	                                   block, storage);
	c->fn        = fn;
	c->data      = block->Copy(static_cast<const uint8_t*>(data), static_cast<uint32_t>(bytes));
	Publish(&c->h, F_Device);
}

const char* GpuTagName(GpuTag tag) noexcept {
	static const char* const names[] = {"guest",     "buf_upload", "buf_download", "buf_copy",  "stream",
	                                    "img_upload", "img_readback", "img_copy",   "img_resolve", "depth_copy",
	                                    "rt_copy",   "detile",     "fault",        "tex_clear", "blit",
	                                    "present",   "img_transit", "draw_sync",  "dispatch_sync"};
	static_assert(std::size(names) == static_cast<size_t>(GpuTag::Count));
	const auto index = static_cast<size_t>(tag);
	return index < std::size(names) ? names[index] : "?";
}

bool GpuTimingOn() noexcept {
	return GpuTiming::g_on;
}

void NoteSubmitState(bool previous_done) noexcept {
	if (GpuTiming::g_on && !GpuTiming::g_edge_open && GpuTiming::g_edge_next != 0) {
		GpuTiming::g_edge_kind[(GpuTiming::g_edge_next - 1) % GpuTiming::EdgeRing] = previous_done ? 1 : 2;
	}
}

void SetTimestampPeriod(float period_ns) noexcept {
	GpuTiming::g_period_ns = period_ns;
	if (GpuTiming::g_on) {
		std::printf("PERF gput: on, 1 of %u main buffers, period %.3f ns\n", GpuTiming::g_every, period_ns);
	}
}

void NotePipelineLabel(vk::Pipeline pipeline, const char* label) {
	if (!GpuTiming::g_on) {
		return;
	}
	std::lock_guard lock(GpuTiming::g_label_mutex);
	GpuTiming::g_labels[reinterpret_cast<uint64_t>(static_cast<VkPipeline>(pipeline))] = label;
}

void Report(double seconds) {
	if (g_mode <= 0) {
		return;
	}
	GpuTiming::Report(seconds);
	static uint64_t last_tsc = Ticks();
	static auto     last_ns  = std::chrono::steady_clock::now();
	const auto      now_tsc  = Ticks();
	const auto      now_ns   = std::chrono::steady_clock::now();
	const double    ns       = std::chrono::duration<double, std::nano>(now_ns - last_ns).count();
	const double    tick_ns  = now_tsc > last_tsc && ns > 0 ? ns / static_cast<double>(now_tsc - last_tsc) : 0.0;
	last_tsc                 = now_tsc;
	last_ns                  = now_ns;
	const auto ms            = [tick_ns](uint64_t ticks) {
        return static_cast<double>(ticks) * tick_ns / 1e6;
	};
	auto top = [](const std::array<uint64_t, F_Count>& counts, size_t n) {
		std::vector<std::pair<uint64_t, uint32_t>> v;
		for (uint32_t i = 0; i < F_Count; i++) {
			if (counts[i] != 0) v.emplace_back(counts[i], i);
		}
		std::sort(v.begin(), v.end(), std::greater<>());
		if (v.size() > n) v.resize(n);
		return v;
	};
	if (g_mode == 3) {
		uint64_t calls = 0, ticks = 0;
		for (uint32_t i = 0; i < F_Count; i++) {
			calls += g_ps.timed_calls[i];
			ticks += g_ps.timed_ticks[i];
		}
		std::string line;
		for (const auto& [t, i]: top(g_ps.timed_ticks, 12)) {
			char buf[128];
			const char* name = g_func_names[i];
			std::snprintf(buf, sizeof(buf), " %s=%.1fms/%" PRIu64,
			              std::strncmp(name, "vkCmd", 5) == 0 ? name + 5 : name + 2, ms(t),
			              g_ps.timed_calls[i]);
			line += buf;
		}
		std::printf("PERF cmdrec timed %.1fs: calls=%" PRIu64 " driver=%.1fms%s\n", seconds, calls,
		            ms(ticks), line.c_str());
	} else {
		uint64_t deferred = 0, trapped = 0;
		for (uint32_t i = 0; i < F_Count; i++) {
			deferred += g_ps.deferred[i];
			trapped += g_ps.trapped[i];
		}
		std::string line;
		for (const auto& [count, i]: top(g_ps.trapped, 8)) {
			char buf[128];
			std::snprintf(buf, sizeof(buf), " %s=%" PRIu64, g_func_names[i], count);
			line += buf;
		}
		std::printf("PERF cmdrec %.1fs: deferred=%" PRIu64 " (%.1f MiB) recorded=%" PRIu64
		            " recorder_busy=%.1fms drains=%" PRIu64 " drain_waits=%" PRIu64
		            " drain_wait=%.1fms space_waits=%" PRIu64 "/%.1fms wakes=%" PRIu64
		            " sleeps=%" PRIu64 " foreign=%" PRIu64 " trapped=%" PRIu64 ":%s\n",
		            seconds, deferred, static_cast<double>(g_ps.bytes) / (1024.0 * 1024.0),
		            g_rec_commands.exchange(0, std::memory_order_relaxed),
		            ms(g_rec_busy_ticks.exchange(0, std::memory_order_relaxed)), g_ps.drains,
		            g_ps.drain_waits, ms(g_ps.drain_ticks), g_ps.space_waits, ms(g_ps.space_ticks),
		            g_ps.wakes, g_rec_sleeps.exchange(0, std::memory_order_relaxed),
		            g_foreign_total.load(std::memory_order_relaxed), trapped, line.c_str());
	}
	g_ps = {};
}

void Describe() noexcept {
	if (g_mode <= 0) {
		std::printf("cmdrec: off\n");
		return;
	}
	const auto write = g_write.load(std::memory_order_acquire);
	const auto read  = g_read.load(std::memory_order_acquire);
	std::printf("cmdrec: mode %d submit %d published %llu recorded %llu (pending %llu bytes) recorder %s, target %p, "
	            "commands %llu, foreign %llu\n",
	            g_mode, g_submit ? 1 : 0, static_cast<unsigned long long>(write), static_cast<unsigned long long>(read),
	            static_cast<unsigned long long>(write - read), g_sleeping.load() ? "sleeping" : "running",
	            static_cast<void*>(g_target.load()), static_cast<unsigned long long>(g_rec_commands.load()),
	            static_cast<unsigned long long>(g_foreign_total.load()));
}

} // namespace Libs::Graphics::CmdRecord
