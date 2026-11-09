#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/dispatchChainPolicy.h"
#include "graphics/host_gpu/renderer/dlss.h"
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

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
// KYTY_MESH_INDIRECT_GPU=1: indirect mesh draws read their records from a task shader.
bool MeshIndirectGpuEnabled();

struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	// The user-data slots the packet patches with each record's bases (mesh draws only).
	ShaderIndirectOffsets offsets;
};

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
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

// KYTY_DISPATCH_CHAIN: 0 = a barrier pair around every direct dispatch (previous behavior);
// 1 = direct dispatches recorded back to back share one barrier pair while no guest sync point and
// no resource overlap separates them (dispatchChainPolicy.h; default); 2 = 1 plus verification:
// guest address overlaps between dispatches left without a barrier are counted as missed
// hazards, and "PERF dchain" reports the barrier counts.
[[nodiscard]] int DispatchChainMode();
namespace DispatchChain {
void Report(double seconds, uint64_t frames); // "PERF dchain" line in mode 2
} // namespace DispatchChain

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return m_buffer == nullptr; }

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

	// Direct dispatches recorded back to back defer their trailing barrier (KYTY_DISPATCH_CHAIN).
	// Every Handle() use outside a DispatchRecording scope records it first, so anything else
	// recorded after a chain of dispatches is ordered after all of them, as without chaining.
	[[nodiscard]] DispatchChain::State& DispatchChainState() const noexcept { return m_dispatch_chain; }
	void                                FlushDispatchChain() const;
	class DispatchRecording {
	public:
		DispatchRecording(const CommandBuffer& buffer, bool active) noexcept
		    : m_buffer(buffer), m_active(active) {
			if (m_active) {
				m_buffer.m_dispatch_recording = true;
			}
		}
		~DispatchRecording() {
			if (m_active) {
				m_buffer.m_dispatch_recording = false;
			}
		}
		DispatchRecording(const DispatchRecording&)            = delete;
		DispatchRecording& operator=(const DispatchRecording&) = delete;

	private:
		const CommandBuffer& m_buffer;
		bool                 m_active;
	};

	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Buffer-cache upload batches: the trailing barrier of an upload is deferred. Every Handle()
	// use first records it, so nothing else is ever recorded between an upload and its barrier;
	// HandleForUpload() is for the next upload of the same batch only.
	[[nodiscard]] vk::CommandBuffer HandleForUpload() const;
	[[nodiscard]] bool UploadPostPending() const noexcept { return m_upload_post_pending; }
	void               DeferUploadPost() const noexcept { m_upload_post_pending = true; }
	void               FlushUploadPost() const;
	// KYTY_UPLOAD_BATCH=1: the copies of a batch are recorded with its barriers when it is flushed
	// (leading barrier, one vkCmdCopyBuffer per source and destination pair, trailing barrier).
	// The same rule as the trailing barrier applies: nothing else is recorded before the flush, so
	// the copies keep their place in the command stream. A region over bytes that a pending copy
	// writes flushes the pending ones first (write-after-write order).
	void DeferUploadCopies(vk::Buffer source, vk::Buffer destination,
	                       std::span<const vk::BufferCopy> regions) const;
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
	vk::CommandBuffer   m_buffer = nullptr;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable bool        m_upload_post_pending = false;
	struct PendingUploadCopy {
		vk::Buffer     source;
		vk::Buffer     destination;
		vk::BufferCopy region;
	};
	mutable std::vector<PendingUploadCopy> m_upload_copies;
	mutable std::vector<vk::BufferCopy>    m_upload_regions; // flush scratch
	mutable vk::PipelineStageFlags m_draw_write_stages {};
	mutable uint64_t               m_draw_write_epoch = 0;
	mutable DispatchChain::State   m_dispatch_chain;
	mutable bool                   m_dispatch_recording = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

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
	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	// KYTY_CP_PIPELINE (cpPipeline.h). How much of PrepareDrawRenderState already ran: nothing, the
	// program lookup, or the lookup after the image heap's BeginDraw.
	enum class DrawPrepared { No, LookedUp, LookedUpHeapBegun };
	bool DrawIndexBody(CommandBuffer& buffer, const DrawIndexArgs& args,
	                   DrawRenderState* prepared, CpPipeline::DrawRecord* record, DrawPrepared stage);
	bool DrawAutoBody(CommandBuffer& buffer, const DrawAutoArgs& args,
	                  DrawRenderState* prepared, CpPipeline::DrawRecord* record, DrawPrepared stage);
	// GPU thread: looks the draw's programs up and queues it; false when it must run here.
	bool PipelineDraw(CommandBuffer& buffer, const DrawIndexArgs* index_args,
	                  const DrawAutoArgs* auto_args);
	static void PipelineExecute(void* context, CpPipeline::DrawRecord& record);
	static void PipelineRelookup(void* context, CpPipeline::DrawRecord& record);
	static void PipelineFeedback(void* context, const CpPipeline::DrawRecord& record);
	static void PipelineHeap(void* context);
	// KYTY_PIPELINE_PARALLEL (pipelineCache.h): predicts the pipelines of the queued draws.
	static void PipelinePrefetch(void* context);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false,
	                              TextureCache::FindImageMemo* memo = nullptr);
	// memo (KYTY_TARGET_REUSE, KYTY_BIND_PROF): pass an empty one; FindImage fills it when its search may be kept.
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target,
	                              TextureCache::FindImageMemo* memo = nullptr);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          DrawRenderState& state,
	                                          CpPipeline::DrawRecord* record = nullptr,
	                                          DrawPrepared prepared = DrawPrepared::No);
	void ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, const DrawEmitInfo& emit,
	                         const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool        Upscale(vk::CommandBuffer command, const Pssr::Request& request,
	                                  Dlss::Inputs& inputs);

	// DLSS history of a PSSR context; a null feature is one DLSS refused or failed.
	struct DlssContext {
		uint64_t       context;
		Dlss::Creation creation;
		Dlss::Feature* feature;
	};

	// GET_LOD_STATS counters since their last reset. With --lod-stats a bound T# with counting
	// enabled counts as sampled at its BASE_LEVEL, clamped when its MIN_LOD lies above that level.
	struct MipStatsCounter {
		uint8_t finest_level = 15; // no samples
		bool    clamped      = false;
		bool    unclamped    = false;
	};

	RenderContext&                        m_context;
	bool                                  m_lod_stats = Config::LodStatsEnabled();
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	DispatchChain::Accesses              m_chain_accesses;
	std::vector<DispatchChain::Access>   m_chain_guest_accesses;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	std::vector<DlssContext>              m_dlss_contexts;
	std::array<MipStatsCounter, 256>      m_mip_stats {};

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
