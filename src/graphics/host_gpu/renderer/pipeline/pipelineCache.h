#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct AliasedColorTargets;
struct GraphicContext;
class CommandBuffer;
class ShaderStore;

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint8_t                    patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 115);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
	// The draw samples the depth target it writes and the host has no dynamic loop state.
	bool                                                 depth_feedback = false;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	// A module loaded from the shader store has no handle module until a pipeline needs it.
	explicit operator bool() const { return id != 0; }
};

// A compiled program variant. The handle id is stable across runs: it hashes the program key,
// push data start, specialization and function code, which with the host and the build decide
// the SPIR-V. A pipeline is built from its key and its stages' modules alone.
struct ShaderModule {
	ShaderProgram                                handle;
	ShaderRecompiler::IR::CompiledShaderInfo     program;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// VS or LS: the host format of each vertex input attribute.
	std::vector<vk::Format>                      vertex_formats;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Loaded from the shader store: 1 + the index of its SPIR-V there (ShaderStore::ReadSpirv),
	// from which the handle module is created when a pipeline needs it; 0 otherwise.
	uint32_t                                     store_spirv = 0;
	// Loaded from the shader store and not used yet in this launch.
	bool                                         store_use_pending = false;
	// Loaded with KYTY_SHADER_STORE=2: its first use retranslates it to check the store.
	bool                                         store_verify_pending = false;
};

// A draw's topology, shader stages (from PipelineCache::GetGraphicsPrograms) and targets.
struct DrawRenderState {
	RenderDepthInfo                      depth_info;
	RenderColorInfo                      color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t                             color_count                              = 0;
	// Slots drawn in one pass each (aliased targets that cannot share one attachment).
	uint32_t                             aliased_passes                           = 0;
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask                                 = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable                 = false;
	bool                                 ps_active                                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		// Prewarmed from the shader store and not used yet in this launch: its record id.
		uint64_t                store_id              = 0;
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	// `single_sample` selects the pixel permutation; `aliased` merges aliased color targets.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state,
	                         bool single_sample = false, const AliasedColorTargets* aliased = nullptr);
	// An empty program skips the dispatch.
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state) {
		return GetGraphicsPipeline(command, state, {state.color_info, state.color_count});
	}
	// `colors`: the attachments of one pass (all of the state's, or one aliased-target pass).
	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              std::span<const RenderColorInfo> colors,
	                              bool                             depth_feedback = false);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

private:
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	// The shader store (KYTY_SHADER_STORE, --shader-cache), loaded by m_warm_up. Every other member
	// function waits for the warm-up first.
	std::unique_ptr<ShaderStore>  m_store;
	std::thread                   m_warm_up;
	// Loaded modules whose handle module a new pipeline needed (GPU thread).
	uint64_t                      m_stored_modules_created = 0;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	uint64_t                      m_saved_payload_hash = 0; // payload of the file on disk
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;

	void InitializeDriverCache();
	void CheckpointDriverCache();
	// Loads the shader store: its modules, then its pipelines, created on several threads.
	void WarmUp(const std::filesystem::path& path);
	void WaitWarmUp() {
		if (m_warm_up.joinable()) [[unlikely]] {
			JoinWarmUp();
		}
	}
	void JoinWarmUp();
	// Creates the handle module of a module loaded from the shader store, if it has none.
	void LoadStoredModule(const ShaderModule* module);
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Driver pipeline-cache outcome of the latest CreatePipelineInternal call on this thread, from
// pipeline creation feedback (core in Vulkan 1.3). `valid` is false when the driver did not report.
struct PipelineCacheFeedback {
	bool valid = false;
	bool hit   = false;
};
[[nodiscard]] PipelineCacheFeedback LastPipelineCacheFeedback() noexcept;
// Pipelines created by this thread are left out of the performance counters (shader store warm-up).
void SetPipelineCreationQuiet(bool quiet) noexcept;
// `vertex_stages`: VS, mesh or LS/HS/TES; `pixel` is null without a pixel stage.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&        rendering,
                            const PipelineVertexInputState&      vertex_input,
                            const PipelineStaticParameters&      static_params,
                            std::span<const ShaderModule* const> vertex_stages,
                            const ShaderModule* pixel, vk::PipelineCache driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& compute, vk::PipelineCache driver_cache);
// The host format of each vertex input attribute of a VS or LS program.
std::vector<vk::Format> VertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
