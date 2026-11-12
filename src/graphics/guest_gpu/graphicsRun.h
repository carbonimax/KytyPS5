#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// Retries blocked submissions now instead of after the 100 ms stall timeout.
	void               Wake();

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	// Same, giving up once every queued submission stayed blocked (waiting on memory nobody writes)
	// for timeout_ms; busy work that progresses is waited for. Also true, without the timeout, once
	// every queued submission is blocked and its command memory is in `may_stay` (capture replay).
	bool              WaitForIdle(uint32_t timeout_ms, std::span<const uint32_t* const> may_stay = {});
	// Capture replay, after WaitForIdle: drops the queued (blocked) submissions; returns how many.
	uint32_t          DropQueued();
	// Capture replay, before the first submission: a queue's (0 or a compute queue as in SubmitCompute)
	// command processor state from GpuCapture::CpState; false for another build's layout.
	bool              RestoreCpState(uint32_t queue, std::span<const uint8_t> bytes);
	// Diagnostics: one line per queued submission (queue, commands, depth, next packet dwords).
	[[nodiscard]] std::string DescribeQueued();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	// KYTY_CP_PIPELINE back thread: runs GPU-thread code paths (never guest code).
	static void MarkHostGpuThread() noexcept;
	// GPU thread: KYTY_REPROTECT_BATCH flush at a guest sync point (see BufferCache).
	void FlushBatchedUploads();

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Commands, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Commands;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      started         = false;
		uint32_t                  capture_submit  = UINT32_MAX; // GpuCapture::RecordSubmit index
		bool                      blocked         = false;
		uint64_t                  flip_request_id = 0;
		bool                      yielded         = false;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	// GPU capture: the frame boundary (`frame_end`) or a scope boundary (GpuCapture::ScopeBoundary).
	void              CaptureBoundary(bool frame_end);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);
	// Graphics processing yields at a packet boundary when compute work was queued since `seen`.
	bool ShouldYieldToCompute(uint64_t& seen) const;

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	std::atomic_uint64_t                           m_compute_enqueued {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_wake_pending      = false;
	bool                                           m_stalled           = false; // every front blocked, no progress since
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	std::atomic_int m_done_num = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
};

// KYTY_EOP_AT_GPU (graphicsRun.cpp): RELEASE_MEM labels released by the command processor that
// reach guest memory only at host-GPU completion ("pending" until then).
namespace EopAtGpu {
// KYTY_EOP_AT_GPU is 1 or 2.
[[nodiscard]] bool Enabled() noexcept;
// GPU thread: `bytes` holds a copy of guest memory [address, address + size) about to be used by
// GPU work recorded now; overlays the pending label bytes (as if written at release). Returns true
// when it changed something.
bool OverlayPending(uint64_t address, void* bytes, uint64_t size) noexcept;
// The pending labels overlapping [address, address + size) as (start, size); mode 1 only.
void PendingRanges(uint64_t address, uint64_t size, std::vector<std::pair<uint64_t, uint32_t>>& out) noexcept;
// Command-processor reads of a value that may be a pending label.
[[nodiscard]] uint32_t ReadFront32(const volatile uint32_t* address) noexcept;
[[nodiscard]] uint64_t ReadFront64(const volatile uint64_t* address) noexcept;
} // namespace EopAtGpu
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
