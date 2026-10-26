#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Called at the start of every submission (GPU thread), while set: work recorded earlier
	// may read host memory that is filled late (KYTY_PROTECT_COALESCE deferred uploads).
	void SetPreSubmit(void (*callback)(void*), void* context) noexcept {
		m_pre_submit         = callback;
		m_pre_submit_context = context;
	}
	// The next submission waits for `value` of `semaphore` (readback-queue copies that read
	// buffers later commands may overwrite).
	void WaitBeforeNextSubmit(vk::Semaphore semaphore, uint64_t value) noexcept {
		m_external_wait       = semaphore;
		m_external_wait_value = std::max(m_external_wait_value, value);
	}
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Waits for submitted work through tick, including Metal completion on MoltenVK.
	void                      WaitRetired(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferRelease(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	// The render context's scheduler: its command buffers are the ones KYTY_CP_RECORD_THREAD
	// records on the recorder thread (see cmdRecord.h).
	void MarkMainRecorder() noexcept { m_record_main = true; }
	// KYTY_IDLE_SUBMIT=N (default 8, 0 = off; GPU thread, command processor between packets): true when
	// the command buffer being recorded holds at least N draws or dispatches and the GPU completed
	// everything submitted before it. The caller then submits it (earlier submission only: same
	// commands, same order, same interrupts and labels), so an idle GPU starts on recorded work.
	// KYTY_IDLE_SUBMIT_PASS=0: not while a rendering pass is open (the submission would end it).
	[[nodiscard]] static bool IdleSubmitEnabled();
	[[nodiscard]] bool        IdleSubmitDue();
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	struct SubmittedFence {
		uint64_t  tick = 0;
		vk::Fence fence = nullptr;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(std::queue<PendingOperation>& queue, Common::UniqueFunction<void>&& operation);
	void PopReady(std::queue<PendingOperation>& queue, uint64_t retired_tick);
	vk::Fence AcquireFence();
	uint64_t RetireFences(uint64_t wait_tick);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_pending_releases;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	const bool                   m_release_waits_for_fence;
	bool                         m_record_main = false;
	bool                         m_capturing   = false; // the recorder thread takes the current buffer
	// KYTY_IDLE_SUBMIT: PerfStats::t_work_recorded at BeginCommand, time of the last semaphore query.
	uint64_t m_work_at_begin   = 0;
	uint64_t m_idle_refresh_ns = 0;
	std::mutex                   m_fence_mutex;
	std::deque<SubmittedFence>    m_submitted_fences;
	std::vector<vk::Fence>        m_free_fences;
	uint64_t                     m_retired_tick = 0;
	void (*m_pre_submit)(void*)                         = nullptr;
	void*                        m_pre_submit_context   = nullptr;
	vk::Semaphore                m_external_wait        = nullptr;
	uint64_t                     m_external_wait_value  = 0;
	uint64_t                     m_external_waited      = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
