#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/cpPipeline.h"
#include "graphics/host_gpu/renderer/cpSplitProf.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// KYTY_EOP_AT_GPU=1 (default 0): RELEASE_MEM labels reach guest memory when the host GPU has
// completed the work recorded before the packet, in stream order, instead of when the command
// processor reads the packet. The label write joins the interrupt in one priority operation of the
// command scheduler, whose single runner takes them in queue order once the master semaphore
// reached the packet's tick: delivery order = release order, one write per release (no merge).
// Until then the label is "pending": GPU-side readers keep today's view (label written at release)
// through the pending table: command-processor reads (WAIT_REG_MEM, COND_EXEC, branch, predication)
// and buffer-cache uploads of guest memory overlay the pending bytes, so the front never waits for
// its own end of pipe and later recorded work still sees the label. Labels on GPU-modified pages
// keep the immediate GPU-owned path (a guest sees them only through a readback, which waits for
// this point of the command stream). Timestamp labels (data_sel 3) keep the reference clock read
// with the packet, as with 0: only their delivery waits for completion.
// =2 (verify): labels are written immediately as with 0; the pending table and the completion
// operations run as with 1 and check the delivery order, that the label bytes reached memory, and
// that overlays agree with memory. Counters: "PERF eop" lines (KYTY_PERF_STATS=1).
namespace {

int EopAtGpuMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_EOP_AT_GPU");
		return value != nullptr ? static_cast<int>(std::strtol(value, nullptr, 10)) : 0;
	}();
	return mode;
}

// KYTY_EOP_AT_GPU_FLUSH=1 (with KYTY_EOP_AT_GPU=1, default 0): submit the command buffer after
// each deferred label, so the guest does not wait for the end of the slice to see it.
bool EopFlushEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EOP_AT_GPU_FLUSH");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint64_t EopNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct EopStats {
	std::atomic<uint64_t> deferred {0};       // labels written at completion
	std::atomic<uint64_t> gpu_owned {0};      // labels kept on the immediate GPU-owned path
	std::atomic<uint64_t> gds {0};            // GDS labels (drained, then written)
	std::atomic<uint64_t> interrupts {0};
	std::atomic<uint64_t> fixups {0};         // completions that found the bytes GPU-owned
	std::atomic<uint64_t> fallback {0};       // completions outside the backing, rewritten by the GPU thread
	std::atomic<uint64_t> latency_ns {0};     // packet to label delivery
	std::atomic<uint64_t> latency_max_ns {0};
	std::atomic<uint64_t> front_reads {0};    // front reads that hit a pending label
	std::atomic<uint64_t> front_passes {0};   // ... WAIT_REG_MEM that passed only because of it
	std::atomic<uint64_t> upload_overlays {0}; // buffer uploads that covered a pending label
	std::atomic<uint64_t> order_errors {0};   // delivery order differs from release order
	std::atomic<uint64_t> value_errors {0};   // label bytes not in memory after delivery
	std::atomic<uint64_t> overlay_diffs {0};  // verify: overlay and memory disagree
	std::atomic<uint64_t> overwritten {0};    // verify: guest rewrote the label before completion
	std::atomic<uint64_t> checked {0};        // completions checked
};

EopStats g_eop_stats;

void NoteLatency(uint64_t released_ns) {
	const auto latency = EopNowNs() - released_ns;
	g_eop_stats.latency_ns.fetch_add(latency, std::memory_order_relaxed);
	auto max = g_eop_stats.latency_max_ns.load(std::memory_order_relaxed);
	while (latency > max && !g_eop_stats.latency_max_ns.compare_exchange_weak(max, latency, std::memory_order_relaxed)) {
	}
}

// Released labels by address: the value of the latest one, how many are pending (released, not yet
// written to guest memory) and the sequence of the latest release, so a GPU-thread fix-up never
// replaces a newer label.
class PendingLabels {
public:
	uint64_t Add(uint64_t address, uint32_t size, uint64_t value, bool pending) {
		std::lock_guard lock(m_mutex);
		if (m_entries.size() > 4096) {
			std::erase_if(m_entries, [](const auto& item) { return item.second.pending == 0; });
		}
		auto& entry  = m_entries[address];
		entry.value  = value;
		entry.size   = size;
		entry.latest = ++m_seq;
		if (pending) {
			entry.pending++;
			m_pending.fetch_add(1, std::memory_order_release);
		}
		return entry.latest;
	}

	void Delivered(uint64_t address) {
		std::lock_guard lock(m_mutex);
		auto            it = m_entries.find(address);
		if (it != m_entries.end() && it->second.pending != 0) {
			it->second.pending--;
			m_pending.fetch_sub(1, std::memory_order_release);
		}
	}

	// No label was released at `address` after the one with sequence `seq`.
	[[nodiscard]] bool IsLatest(uint64_t address, uint64_t seq) {
		std::lock_guard lock(m_mutex);
		auto            it = m_entries.find(address);
		return it == m_entries.end() || it->second.latest == seq;
	}

	[[nodiscard]] bool AnyPending() const noexcept { return m_pending.load(std::memory_order_acquire) != 0; }

	// A pending label overlaps [address, address + size).
	[[nodiscard]] bool HasPending(uint64_t address, uint64_t size) {
		if (!AnyPending()) {
			return false;
		}
		std::lock_guard lock(m_mutex);
		bool            hit = false;
		ForEachPending(address, size, [&hit](uint64_t, const Entry&) { hit = true; });
		return hit;
	}

	// Overlays the pending label bytes on `bytes`, a copy of guest memory [address, address + size).
	[[nodiscard]] bool Overlay(uint64_t address, uint64_t size, uint8_t* bytes) {
		if (!AnyPending()) {
			return false;
		}
		bool            hit = false;
		std::lock_guard lock(m_mutex);
		ForEachPending(address, size, [&](uint64_t base, const Entry& entry) {
			const auto begin = std::max(base, address);
			const auto end   = std::min(base + entry.size, address + size);
			for (auto byte = begin; byte < end; byte++) {
				bytes[byte - address] = static_cast<uint8_t>(entry.value >> (8u * (byte - base)));
			}
			hit = true;
		});
		return hit;
	}

private:
	struct Entry {
		uint64_t value   = 0;
		uint64_t latest  = 0;
		uint32_t size    = 0;
		uint32_t pending = 0;
	};

	template <typename Func>
	void ForEachPending(uint64_t address, uint64_t size, Func&& func) {
		// Labels are at most 8 bytes: one that starts up to 7 bytes before the range can overlap it.
		for (auto it = m_entries.lower_bound(address >= 7 ? address - 7 : 0);
		     it != m_entries.end() && it->first < address + size; ++it) {
			if (it->second.pending != 0 && it->first + it->second.size > address) {
				func(it->first, it->second);
			}
		}
	}

	std::mutex                    m_mutex;
	std::map<uint64_t, Entry>     m_entries;
	std::atomic<uint32_t>         m_pending {0};
	uint64_t                      m_seq = 0;
};

PendingLabels         g_pending_labels;
std::atomic<uint64_t> g_eop_next_order {0}; // release order (GPU thread)
uint64_t              g_eop_delivered_order = 0; // delivery order (priority runner only)

void CheckOrder(uint64_t order) {
	if (order != g_eop_delivered_order) {
		g_eop_stats.order_errors.fetch_add(1, std::memory_order_relaxed);
	}
	g_eop_delivered_order = order + 1;
}

// Front read of a value that may be a pending label (WAIT_REG_MEM, COND_EXEC, branch, predication).
template <typename T>
T ReadFrontValue(const volatile T* address, bool* from_pending) {
	T value       = *address;
	*from_pending = false;
	if (EopAtGpuMode() == 0 || !g_pending_labels.AnyPending()) {
		return value;
	}
	uint8_t bytes[sizeof(T)];
	std::memcpy(bytes, &value, sizeof(T));
	if (g_pending_labels.Overlay(reinterpret_cast<uint64_t>(address), sizeof(T), bytes)) {
		T pending_value {};
		std::memcpy(&pending_value, bytes, sizeof(T));
		*from_pending = true;
		g_eop_stats.front_reads.fetch_add(1, std::memory_order_relaxed);
		if (EopAtGpuMode() == 2) {
			// The label is already in memory: the overlay must agree unless the guest rewrote it.
			if (pending_value != value) {
				g_eop_stats.overlay_diffs.fetch_add(1, std::memory_order_relaxed);
			}
			return value;
		}
		return pending_value;
	}
	return value;
}

} // namespace

namespace EopAtGpu {

bool OverlayPending(uint64_t address, void* bytes, uint64_t size) noexcept {
	if (EopAtGpuMode() == 0 || !g_pending_labels.AnyPending()) {
		return false;
	}
	if (EopAtGpuMode() == 2) {
		if (size <= 64) {
			uint8_t copy[64];
			std::memcpy(copy, bytes, size);
			if (g_pending_labels.Overlay(address, size, copy)) {
				g_eop_stats.upload_overlays.fetch_add(1, std::memory_order_relaxed);
				if (std::memcmp(copy, bytes, size) != 0) {
					g_eop_stats.overlay_diffs.fetch_add(1, std::memory_order_relaxed);
				}
			}
		} else if (g_pending_labels.HasPending(address, size)) {
			g_eop_stats.upload_overlays.fetch_add(1, std::memory_order_relaxed);
		}
		return false;
	}
	if (g_pending_labels.Overlay(address, size, static_cast<uint8_t*>(bytes))) {
		g_eop_stats.upload_overlays.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	return false;
}

uint32_t ReadFront32(const volatile uint32_t* address) noexcept {
	bool from_pending = false;
	return ReadFrontValue<uint32_t>(address, &from_pending);
}

uint64_t ReadFront64(const volatile uint64_t* address) noexcept {
	bool from_pending = false;
	return ReadFrontValue<uint64_t>(address, &from_pending);
}

bool Enabled() noexcept {
	return EopAtGpuMode() != 0;
}

void Report(double seconds) {
	if (EopAtGpuMode() == 0 || seconds <= 0.0) {
		return;
	}
	auto&      st       = g_eop_stats;
	const auto take     = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const auto deferred = take(st.deferred);
	const auto latency  = take(st.latency_ns);
	const auto checked  = take(st.checked);
	std::printf("PERF eop %.1fs mode=%d: deferred=%.0f/s gpu_owned=%.0f/s gds=%" PRIu64 " interrupts=%.0f/s"
	            " latency=%.3fms(max %.2fms) fixups=%" PRIu64 " fallback=%" PRIu64 " front=%" PRIu64
	            "(passes %" PRIu64 ") uploads=%" PRIu64 " check: checked=%" PRIu64 " order_err=%" PRIu64
	            " value_err=%" PRIu64 " overlay_diff=%" PRIu64 " overwritten=%" PRIu64 "\n",
	            seconds, EopAtGpuMode(), static_cast<double>(deferred) / seconds,
	            static_cast<double>(take(st.gpu_owned)) / seconds, take(st.gds),
	            static_cast<double>(take(st.interrupts)) / seconds,
	            checked != 0 ? static_cast<double>(latency) / 1e6 / static_cast<double>(checked) : 0.0,
	            static_cast<double>(take(st.latency_max_ns)) / 1e6, take(st.fixups), take(st.fallback),
	            take(st.front_reads), take(st.front_passes), take(st.upload_overlays), checked,
	            take(st.order_errors), take(st.value_errors), take(st.overlay_diffs), take(st.overwritten));
}

} // namespace EopAtGpu

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::Wake() {
	Common::LockGuard lock(m_queue_mutex);
	m_wake_pending = true;
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	if (m_pending_commands.load(std::memory_order_acquire) != 0) {
		CpSplitProf::Drain(CpSplitProf::Cause::Command);
	}
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		// Per command: one may arrive after the check above (it records and submits like the back).
		CpPipeline::Drain();
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Graphics;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

// KYTY_COMPUTE_PREEMPT: "1" = every compute queue, "0" = off, or a comma-separated list of
// compute queue interrupt ids (0x20 = first queue of the first pipe). Bit i = compute queue i.
// Default: every queue when fault readbacks are asynchronous. With synchronous readbacks the
// guest threads wait far longer on faults when compute runs earlier (in one title -30 % frames).
static uint64_t ComputePreemptMask() {
	static const uint64_t mask = [] {
		const char* value = std::getenv("KYTY_COMPUTE_PREEMPT");
		if (value == nullptr) {
			return BufferCache::AsyncFaultReadbacks() ? ~uint64_t {0} : uint64_t {0};
		}
		uint64_t result = 0;
		for (const char* p = value; *p != '\0';) {
			char*      end = nullptr;
			const auto id  = std::strtoul(p, &end, 0);
			if (end == p) {
				break;
			}
			if (id == 1) {
				result = ~uint64_t {0};
			} else if (id >= 0x20 && id < 0x20 + 64) {
				result |= uint64_t {1} << (id - 0x20);
			}
			p = (*end == ',') ? end + 1 : end;
		}
		return result;
	}();
	return mask;
}

static bool ComputePreemptionEnabled() {
	return ComputePreemptMask() != 0;
}

bool GuestGpu::ShouldYieldToCompute(uint64_t& seen) const {
	// Async compute runs beside graphics on the real GPU. Games may pace their simulation on
	// compute end-of-pipe interrupts (one tick per interrupt), so a compute submission must not
	// wait behind a whole graphics command buffer.
	if (!ComputePreemptionEnabled()) {
		return false;
	}
	const auto enqueued = m_compute_enqueued.load(std::memory_order_acquire);
	if (enqueued == seen) {
		return false;
	}
	seen = enqueued;
	return true;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_addr               = 0;
	m_predicate_skip                   = false;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
	// The occlusion counters reach a guest that polls them without waiting for a later fence.
	m_renderer.GetOcclusionQueries().Publish();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	bool       from_pending = false;
	const auto value        = ReadFrontValue<T>(addr, &from_pending);
	if (!TestWaitRegMemValue(value, ref, mask, func)) {
		SuspendPm4();
	} else if (from_pending && EopAtGpuMode() == 1 && !TestWaitRegMemValue(*addr, ref, mask, func)) {
		g_eop_stats.front_passes.fetch_add(1, std::memory_order_relaxed);
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4();
	} else {
		(*counter)--;
	}
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// Destinations whose bytes the GPU owns: write through the command stream.
	{
		const auto address = reinterpret_cast<uint64_t>(dst);
		const auto bytes   = write_one_address ? uint64_t {sizeof(uint32_t)}
		                                       : uint64_t {dw_num} * sizeof(uint32_t);
		if (m_renderer.GetBufferCache().WriteGpuOwned(
		        address, write_one_address ? &src[dw_num - 1] : src, bytes)) {
			m_renderer.GetTextureCache().InvalidateMemory(address, bytes);
			return;
		}
	}

	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	if (submission.type == SubmissionType::Compute && submission.queue_id - 1 < 64 &&
	    ((ComputePreemptMask() >> (submission.queue_id - 1)) & 1u) != 0) {
		m_compute_enqueued.fetch_add(1, std::memory_order_release);
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	PerfStats::MarkGpuThread();

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					// A wake that came before this wait must not be lost.
					if (!std::exchange(gpu->m_wake_pending, false)) {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					}
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			{
				PerfStats::ScopedTimer busy(PerfStats::Counter::GpuBusyNs);
				CpSplitProf::BusyBegin();
				CpPipeline::Drain();
				command();
				CpSplitProf::BusyEnd();
			}
			PerfStats::MaybeReport();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		bool complete = false;
		{
			PerfStats::ScopedTimer busy(PerfStats::Counter::GpuBusyNs);
			CpSplitProf::BusyBegin();
			complete = gpu->Process(submission);
			CpSplitProf::BusyEnd();
		}
		PerfStats::MaybeReport();

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			// A yielded submission can resume as soon as the compute queues were served.
			submission.blocked = !submission.yielded;
			submission.yielded = false;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
	}

	// KYTY_CP_PIPELINE: the scheduler and caches change hands at submission boundaries.
	CpPipeline::Drain();
	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Compute:
			if (first_slice) {
				const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
				const auto*     buffer = submission.commands.data();
				static uint32_t compute_batch_log_count = 0;
				if (num_dw <= 128 && compute_batch_log_count++ < 32) {
					LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
					     reinterpret_cast<uint64_t>(buffer), num_dw);
					for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
						LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
					}
				}
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			[[fallthrough]];
		case SubmissionType::Graphics:
			complete =
			    cp.Process(submission.execution, submission.commands) == Pm4ProcessResult::Complete;
			// Only graphics yields (before a sync packet) to queued compute work.
			submission.yielded = submission.execution.Yielded();
			if (submission.execution.MadeProgress()) {
				cp.BufferFlush();
			}
			break;
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			PerfStats::Add(PerfStats::Counter::Frames, 1);
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;
	execution.m_yielded       = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

static bool IsYieldPoint(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_MEM_SEMAPHORE:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_ACQUIRE_MEM: return true;
		default: return false;
	}
}

// KYTY_CP_PIPELINE: packets the front runs while draws are queued (register writes, fetcher,
// direct draws). Every other packet drains the queue first.
static bool IsPipelinedPacket(uint32_t opcode, uint32_t custom) {
	switch (opcode) {
		case Pm4::IT_NOP: return custom == Pm4::R_ZERO || custom == Pm4::R_CONTEXT_STATE;
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return true;
		default: return false;
	}
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	static const bool idle_submit = CommandScheduler::IdleSubmitEnabled();
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		// KYTY_IDLE_SUBMIT: an idle host GPU gets the work recorded so far (earlier submission only).
		CpPipeline::Poll();
		if (idle_submit && CpPipeline::BackIdle() && GetScheduler().Active() &&
		    GetScheduler().IdleSubmitDue()) {
			PerfStats::Add(PerfStats::Counter::IdleSubmits, 1);
			BufferFlush();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		// Graphics yields to queued compute work only before synchronization packets, where
		// the guest itself expects other queues to make progress.
		if (m_interrupt_event_id == 0 && execution.m_made_progress && g_gpu_state != nullptr &&
		    IsYieldPoint(opcode) &&
		    g_gpu_state->ShouldYieldToCompute(m_compute_seen)) {
			execution.m_yielded = true;
			return;
		}

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];
		CpSplitProf::Packet(opcode, KYTY_PM4_R(packet_header));
		if (!IsPipelinedPacket(opcode, KYTY_PM4_R(packet_header))) {
			CpPipeline::Drain();
		}

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances      = num_instances;
	m_num_instances_addr = 0;
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_addr != 0) {
		m_num_instances      = *reinterpret_cast<const uint32_t*>(m_num_instances_addr);
		m_num_instances_addr = 0;
	}
	return m_num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			m_renderer.GetOcclusionQueries().Publish();
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
		{
			bool from_pending = false;
			value = ReadFrontValue<uint64_t>(reinterpret_cast<const volatile uint64_t*>(address), &from_pending);
		}
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, DrawBuffer(), args);
}

CommandBuffer& CommandProcessor::DrawBuffer() {
	return CpPipeline::Mode() != 0 ? GetScheduler().CurrentUnchecked() : CurrentBuffer();
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}
	return reinterpret_cast<const void*>(m_index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index, uint32_t draw_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	write(registers.draw_index, draw_index);
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	const auto draw_count = count_addr == nullptr
	                            ? max_count_or_count
	                            : std::min(uint32_t {*count_addr}, max_count_or_count);
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_draw_indirect_args_base_addr + data_offset;
	struct IndirectScope {
		IndirectScope() { CpSplitProf::SetIndirect(true); }
		~IndirectScope() { CpSplitProf::SetIndirect(false); }
	} indirect_scope;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// With no SGPRs to patch, the GPU reads the records if the shaders allow. Shaders apply
	// the records' offsets only through those SGPRs; Vulkan applies them, which matches
	// while they are zero.
	const DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP && registers.draw_index == Pm4::SH_NOP &&
	    (!indexed || m_index_buffer_size != 0) &&
	    (indexed ? executor.DrawIndex(m_submit_id, CurrentBuffer(),
	                                  {.index_count         = m_index_buffer_size,
	                                   .index_addr          = IndexAddress(0),
	                                   .index_type_and_size = m_index_type_and_size,
	                                   .indirect            = records})
	             : executor.DrawAuto(m_submit_id, CurrentBuffer(), {.indirect = records}))) {
		return;
	}

	const uint32_t max_indices = m_index_buffer_size != 0 ? m_index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto addr = args_addr + static_cast<uint64_t>(i) * stride_in_bytes;
		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(addr);
			PatchIndirectDrawOffsets(*this, registers, args.firstVertex, args.firstInstance, 0, i);
			executor.DrawAuto(m_submit_id, CurrentBuffer(),
			                  {.vertex_count   = args.vertexCount,
			                   .instance_count = args.instanceCount,
			                   .first_vertex   = args.firstVertex,
			                   .first_instance = args.firstInstance,
			                   .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(addr);
		PatchIndirectDrawOffsets(*this, registers, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);
		executor.DrawIndex(m_submit_id, CurrentBuffer(),
		                   {.index_count         = std::min(args.indexCount, max_indices),
		                    .index_addr          = IndexAddress(args.firstIndex),
		                    .instance_count      = args.instanceCount,
		                    .index_type_and_size = m_index_type_and_size,
		                    .base_vertex         = args.vertexOffset,
		                    .first_instance      = args.firstInstance,
		                    .offset_source       = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode, IsAsyncComputeQueue());
	}
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, DrawBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	// Stall only this command processor, not the host command lane or other queues.
	if (!m_renderer.GetVideoOut().IsSafeForRendering(static_cast<int>(video_out_handle),
	                                                 static_cast<int>(display_buffer_index))) {
		SuspendPm4();
	}
}

// KYTY_GPU_OWNED_EOP_LABEL=0 disables. A RELEASE_MEM label often shares a tracker page with
// GPU-written bytes (for example a fence next to a small buffer written by compute). The CPU
// store then faulted on the GPU thread and drained the GPU to publish the page. When a cache
// buffer owns that GPU-modified page, write the label like WRITE_DATA does: into the backing
// and into the cached buffer at this point of the command stream.
static bool GpuOwnedEopLabelEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_OWNED_EOP_LABEL");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void CommandProcessor::WriteEopLabel(void* dst, const void* data, uint64_t size) {
	const auto address = reinterpret_cast<uint64_t>(dst);
	if (GpuOwnedEopLabelEnabled() &&
	    m_renderer.GetBufferCache().WriteGpuOwned(address, data, size)) {
		m_renderer.GetTextureCache().InvalidateMemory(address, size);
		return;
	}
	std::memcpy(dst, data, size);
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	// Dumped occlusion counters are written before any end-of-pipe signal reaches the guest.
	m_renderer.GetOcclusionQueries().Publish();

	const int eop_mode = EopAtGpuMode();
	if (eop_mode == 1) {
		ReleaseMemAtGpu(data_sel, interrupt, address, data, interrupt_context_id);
		return;
	}

	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS.
	uint32_t label_size = 0;
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			WriteEopLabel(address, &value, sizeof(value));
			label_size = sizeof(value);
			data       = value;
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2:
			WriteEopLabel(address, &data, sizeof(data));
			label_size = sizeof(data);
			break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	if (eop_mode == 2) {
		// Verify: the label is already in memory. It is also entered in the pending table (so the
		// overlays are exercised and compared with memory), and the completion checks the delivery
		// order and whether the guest rewrote the label before the host GPU got there.
		auto&      st            = g_eop_stats;
		const auto address_value = reinterpret_cast<uint64_t>(address);
		const auto order         = g_eop_next_order.fetch_add(1, std::memory_order_relaxed);
		uint64_t   seq           = 0;
		if (label_size != 0) {
			uint64_t memory = 0;
			if (Libs::LibKernel::Memory::TryReadBacking(address_value, &memory, label_size) &&
			    std::memcmp(&memory, &data, label_size) != 0) {
				st.value_errors.fetch_add(1, std::memory_order_relaxed);
			}
			seq = g_pending_labels.Add(address_value, label_size, data, true);
		}
		const auto released_ns = EopNowNs();
		GetScheduler().DeferPriorityOperation([&renderer = m_renderer, event_id = m_interrupt_event_id,
		                                       interrupt_context_id, interrupt, address_value, label_size,
		                                       value = data, seq, order, released_ns] {
			auto& st = g_eop_stats;
			CheckOrder(order);
			if (label_size != 0) {
				NoteLatency(released_ns);
				st.checked.fetch_add(1, std::memory_order_relaxed);
				uint64_t memory = 0;
				if (g_pending_labels.IsLatest(address_value, seq) &&
				    Libs::LibKernel::Memory::TryReadBacking(address_value, &memory, label_size) &&
				    std::memcmp(&memory, &value, label_size) != 0) {
					st.overwritten.fetch_add(1, std::memory_order_relaxed);
				}
				g_pending_labels.Delivered(address_value);
			}
			if (interrupt) {
				st.interrupts.fetch_add(1, std::memory_order_relaxed);
				renderer.TriggerInterrupt(event_id, interrupt_context_id);
			}
		});
		return;
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
			    renderer.TriggerInterrupt(event_id, interrupt_context_id);
		    });
	}
}

// KYTY_EOP_AT_GPU=1: see EopAtGpuMode. The completion operation captures the render context, not
// this processor: priority operations can still run while the guest GPU shuts down.
void CommandProcessor::ReleaseMemAtGpu(uint32_t data_sel, bool interrupt, void* address, uint64_t data,
                                       uint32_t interrupt_context_id) {
	auto&      st            = g_eop_stats;
	const auto address_value = reinterpret_cast<uint64_t>(address);
	uint32_t   label_size    = 0;
	bool       ref_clock     = false;
	switch (data_sel) {
		case 0: break;
		case 1:
			label_size = sizeof(uint32_t);
			data       = static_cast<uint32_t>(data);
			break;
		case 3: ref_clock = true; [[fallthrough]];
		case 2: label_size = sizeof(uint64_t); break;
		case 5: {
			// The GDS bytes are read once all earlier work completed and every earlier label was
			// delivered (nothing older stays pending), then written at once.
			GetScheduler().Finish();
			GetScheduler().WaitPriorityOperations(GetScheduler().CurrentTick() - 1);
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			st.gds.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	bool     deferred = false;
	uint64_t seq      = 0;
	uint64_t value    = data;
	if (label_size != 0) {
		value = ref_clock ? Sync::ReadReferenceClock() : data;
		// A GPU-modified destination keeps the immediate GPU-owned path: the guest reads it only
		// through a readback, which waits for this point of the command stream. Not while an older
		// label to these bytes is pending: that one would land after this one.
		if (GpuOwnedEopLabelEnabled() && !g_pending_labels.HasPending(address_value, label_size) &&
		    m_renderer.GetBufferCache().WriteGpuOwned(address_value, &value, label_size)) {
			m_renderer.GetTextureCache().InvalidateMemory(address_value, label_size);
			seq = g_pending_labels.Add(address_value, label_size, value, false);
			st.gpu_owned.fetch_add(1, std::memory_order_relaxed);
		} else {
			seq      = g_pending_labels.Add(address_value, label_size, value, true);
			deferred = true;
		}
	}
	if (!deferred && !interrupt) {
		return;
	}
	const auto order       = g_eop_next_order.fetch_add(1, std::memory_order_relaxed);
	const auto released_ns = EopNowNs();
	GetScheduler().DeferPriorityOperation([&renderer = m_renderer, event_id = m_interrupt_event_id,
	                                       interrupt_context_id, interrupt, deferred, address_value,
	                                       label_size, value, seq, order, released_ns] {
		auto& st = g_eop_stats;
		CheckOrder(order);
		if (deferred) {
			// End-of-pipe timestamps keep the clock read with the packet (as with 0). Read now,
			// every packet of a command buffer would get the buffer's end time and the guest's
			// pass timings would collapse; games that size their work from them then add work.
			const uint64_t label  = value;
			const auto     result = renderer.WriteCompletedLabel(address_value, &label, label_size);
			if (result == RenderContext::CompletedWrite::NotBacking &&
			    !renderer.IsMapped(address_value, label_size)) {
				// Untracked memory outside the backing: a plain store cannot need the GPU thread.
				std::memcpy(reinterpret_cast<void*>(address_value), &label, label_size);
			} else if (result != RenderContext::CompletedWrite::Done) {
				// A readback of GPU-written bytes would overwrite the label (or it lies outside the
				// backing): the GPU thread writes it again, in front of any later fault readback of
				// the page, unless a newer label to these bytes was released meanwhile.
				(result == RenderContext::CompletedWrite::GpuOwned ? st.fixups : st.fallback)
				    .fetch_add(1, std::memory_order_relaxed);
				renderer.TrySendGpuCommand([&renderer, address_value, label_size, label, seq] {
					if (!g_pending_labels.IsLatest(address_value, seq)) {
						return;
					}
					if (GpuOwnedEopLabelEnabled() &&
					    renderer.GetBufferCache().WriteGpuOwned(address_value, &label, label_size)) {
						renderer.GetTextureCache().InvalidateMemory(address_value, label_size);
						return;
					}
					std::memcpy(reinterpret_cast<void*>(address_value), &label, label_size);
				});
			} else {
				uint64_t memory = 0;
				if (Libs::LibKernel::Memory::TryReadBacking(address_value, &memory, label_size) &&
				    std::memcmp(&memory, &label, label_size) != 0) {
					st.value_errors.fetch_add(1, std::memory_order_relaxed);
				}
			}
			g_pending_labels.Delivered(address_value);
			st.deferred.fetch_add(1, std::memory_order_relaxed);
			st.checked.fetch_add(1, std::memory_order_relaxed);
			NoteLatency(released_ns);
		}
		if (interrupt) {
			st.interrupts.fetch_add(1, std::memory_order_relaxed);
			renderer.TriggerInterrupt(event_id, interrupt_context_id);
		}
	});
	if (deferred && EopFlushEnabled()) {
		GetScheduler().Flush();
	}
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (auto& queries = m_renderer.GetOcclusionQueries(); queries.Enabled()) {
				queries.Dump(event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}
	m_renderer.GetOcclusionQueries().Publish();

	auto& command = CurrentBuffer();
	auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

void GuestGpu::MarkHostGpuThread() noexcept {
	g_gpu_thread = true;
}

} // namespace Libs::Graphics
