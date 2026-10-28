#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_

#include "common/file.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// The per-title shader store, _PipelineCache/<title>.shaders: a header naming everything the
// records depend on besides their own fields (the build, the host and the switches that change
// the SPIR-V), then checksummed zstd frames, one per record. Records are only appended, so a store
// cut short keeps every complete record; the first damaged one ends it. Any header difference
// starts the store over, so SPIR-V from another build or host is never loaded.
//
// Adapted from upstream ec8b75b2 (store and prewarm), dd5bc26c (background writer) and 83f8bfd3
// (console messages).

enum class ShaderStoreMode : uint8_t { Off, On, Verify };

// KYTY_SHADER_STORE=0/1/2 (off, on, on and retranslate each loaded module at its first use to
// check it), else --shader-cache. Off in non-Release builds and with graphics debug dumps.
[[nodiscard]] ShaderStoreMode GetShaderStoreMode();

struct ShaderStoreModule {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	uint32_t              code_size       = 0;
	std::vector<uint32_t> static_state;
	// SaveResourcePlan of the program, in the first record of each program only.
	std::vector<uint8_t>  plan;
	// The handle module is created from `spirv`; stored programs have no function code.
	ShaderModule          module;
	// Pixel programs: the pretranslation id of their translation key, 0 otherwise.
	uint64_t              translation_id = 0;
	std::vector<uint32_t> spirv;
};

struct ShaderStorePipeline {
	PipelineRenderingState   rendering;
	std::array<uint64_t, 3>  vertex_ids {};
	uint64_t                 ps_id = 0;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
};

struct ShaderStoreContents {
	// Empty when the modules went to a ShaderStoreModuleSink.
	std::vector<ShaderStoreModule>   modules;
	std::vector<ShaderStorePipeline> pipelines;
	uint64_t                         file_bytes   = 0;
	uint64_t                         kept_bytes   = 0;
	bool                             header_match = false;
};

// Takes each module record as it is decoded, so that only one decoded module is held at a time.
using ShaderStoreModuleSink = std::function<void(ShaderStoreModule&&)>;

// Host state that changes the emitted SPIR-V; part of the store header.
std::vector<uint8_t> ShaderStoreHostKey(const GraphicContext& graphics);

class ShaderStore {
public:
	ShaderStore() = default;
	~ShaderStore();
	KYTY_CLASS_NO_COPY(ShaderStore);

	// Reads every intact record of the store at `path` whose header matches `host_key`, then
	// keeps the file open to append; a mismatching or unreadable store starts over. Call once.
	// With a sink, module records go to it in file order instead of into the contents.
	ShaderStoreContents Open(const std::filesystem::path& path, std::span<const uint8_t> host_key,
	                         const ShaderStoreModuleSink& sink = {});
	[[nodiscard]] bool IsOpen() const { return m_thread.joinable(); }

	// Encode on the calling thread; compression and writes run on the store thread, in order.
	void AppendModule(const ShaderStoreModule& module);
	void AppendPipeline(const ShaderStorePipeline& pipeline);

	// Parses `bytes` (a store file) as Open does, for tests.
	static ShaderStoreContents Parse(std::span<const uint8_t> bytes,
	                                 std::span<const uint8_t> host_key,
	                                 const ShaderStoreModuleSink& sink = {});

private:
	void Post(std::vector<uint8_t>&& record);
	void Run();

	Common::File                     m_file;
	std::mutex                       m_mutex;
	std::condition_variable          m_ready;
	std::deque<std::vector<uint8_t>> m_records;
	bool                             m_stop = false;
	std::thread                      m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_
