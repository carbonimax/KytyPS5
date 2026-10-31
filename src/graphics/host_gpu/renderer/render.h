#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;
namespace CpPipeline {
struct DrawRecord;
} // namespace CpPipeline

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t address = 0;
	uint32_t count   = 0;
	uint32_t stride  = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

// KYTY_CP_PIPELINE: the back thread reads a draw record's copy of the registers through the same
// CommandBuffer the GPU thread keeps binding to the live ones.
struct RegisterOverride {
	HW::Context*    registers   = nullptr;
	HW::UserConfig* user_config = nullptr;
	HW::Shader*     shaders     = nullptr;
};
inline thread_local RegisterOverride t_register_override;

class ScopedRegisterOverride {
public:
	ScopedRegisterOverride(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept
	    : m_previous(t_register_override) {
		t_register_override = {&registers, &user_config, &shaders};
	}
	~ScopedRegisterOverride() { t_register_override = m_previous; }
	ScopedRegisterOverride(const ScopedRegisterOverride&)            = delete;
	ScopedRegisterOverride& operator=(const ScopedRegisterOverride&) = delete;

private:
	RegisterOverride m_previous;
};

// Guest synchronization points seen by the command processors: cache flush and invalidate events,
// releases, acquires, waits, semaphores, flips and CP memory reads or writes. Draws between two of
// them are unordered on the guest GPU, so their buffer writes need no barrier between each other.
inline std::atomic<uint64_t> g_guest_sync_epoch {0};
inline void                  NoteGuestSync() noexcept {
    g_guest_sync_epoch.fetch_add(1, std::memory_order_relaxed);
}
[[nodiscard]] inline uint64_t GuestSyncEpoch() noexcept {
	return g_guest_sync_epoch.load(std::memory_order_relaxed);
}

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool Rendering() const noexcept { return m_rendering; }
	// Buffer writes of draws recorded in the open pass, made visible to later work when the pass
	// ends (EndRendering records the barrier). Nothing that runs outside a pass can be recorded
	// before that, so only draws see the writes late, and only draws of the same sync epoch.
	void DeferDrawWrites(vk::PipelineStageFlags stages, uint64_t sync_epoch) const noexcept {
		m_draw_write_stages |= stages;
		m_draw_write_epoch   = sync_epoch;
	}
	// The open pass holds buffer-writing draws of this sync epoch: a later buffer-writing draw of
	// the same epoch needs no hazard barrier (the first one of the pass ordered it after earlier work).
	[[nodiscard]] bool DrawWritesChained(uint64_t sync_epoch) const noexcept {
		return m_rendering && m_draw_write_stages && m_draw_write_epoch == sync_epoch;
	}
	// The open pass holds buffer writes of an earlier sync epoch, which the next draw may read.
	[[nodiscard]] bool DrawWritesStale(uint64_t sync_epoch) const noexcept {
		return m_draw_write_stages && m_draw_write_epoch != sync_epoch;
	}

	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Buffer-cache upload batches: the trailing barrier of an upload is deferred. Every Handle()
	// use first records it, so nothing else is ever recorded between an upload and its barrier;
	// HandleForUpload() is for the next upload of the same batch only.
	[[nodiscard]] vk::CommandBuffer HandleForUpload() const;
	[[nodiscard]] bool UploadPostPending() const noexcept { return m_upload_post_pending; }
	void               DeferUploadPost() const noexcept { m_upload_post_pending = true; }
	void               FlushUploadPost() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context& GetRegisters() const noexcept {
		return t_register_override.registers != nullptr ? *t_register_override.registers : *m_registers;
	}
	[[nodiscard]] HW::UserConfig& GetUserConfig() const noexcept {
		return t_register_override.user_config != nullptr ? *t_register_override.user_config : *m_user_config;
	}
	[[nodiscard]] HW::Shader& GetShaders() const noexcept {
		return t_register_override.shaders != nullptr ? *t_register_override.shaders : *m_shaders;
	}

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable bool        m_upload_post_pending = false;
	mutable vk::PipelineStageFlags m_draw_write_stages {};
	mutable uint64_t               m_draw_write_epoch = 0;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	// KYTY_CP_PIPELINE (cpPipeline.h). How much of PrepareDrawRenderState already ran: nothing, the
	// program lookup, or the lookup after the image heap's BeginDraw.
	enum class DrawPrepared { No, LookedUp, LookedUpHeapBegun };
	bool DrawIndexBody(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args,
	                   DrawRenderState* prepared, CpPipeline::DrawRecord* record, DrawPrepared stage);
	bool DrawAutoBody(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args,
	                  DrawRenderState* prepared, CpPipeline::DrawRecord* record, DrawPrepared stage);
	// GPU thread: looks the draw's programs up and queues it; false when it must run here.
	bool PipelineDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs* index_args,
	                  const DrawAutoArgs* auto_args);
	static void PipelineExecute(void* context, CpPipeline::DrawRecord& record);
	static void PipelineRelookup(void* context, CpPipeline::DrawRecord& record);
	static void PipelineFeedback(void* context, const CpPipeline::DrawRecord& record);
	static void PipelineHeap(void* context);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state,
	                                          CpPipeline::DrawRecord* record = nullptr,
	                                          DrawPrepared prepared = DrawPrepared::No);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, const DrawEmitInfo& emit,
	                         const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	struct WrittenRange {
		uint64_t address = 0;
		uint64_t size    = 0;
		bool     image   = false;
	};
	std::vector<WrittenRange>             m_written_ranges; // CommitBindings scratch
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
