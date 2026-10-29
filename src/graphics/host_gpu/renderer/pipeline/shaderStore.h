#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_

#include "common/file.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// The per-title shader store, _PipelineCache/<title>.shaders: a header naming everything the
// records depend on besides their own fields (the build, the host and the switches that change
// the SPIR-V), then checksummed zstd frames, one per record. Each frame names its record's kind,
// id, the last launch that wrote or used it and how many launches did. A store cut short keeps
// every complete record; the first damaged one ends it. Any header difference starts the store
// over, so SPIR-V from another build or host is never loaded.
//
// Records are appended while the title runs: programs (key and resource plan), modules (their
// compiled info), the SPIR-V of each module in a frame of its own, graphics pipelines, and lists
// of the loaded modules and pipelines each launch used. A record that no launch used for
// RetainedLaunches launches is dead, as is every duplicate and every use list; once dead records
// are a quarter of the file, the store thread rewrites it next to the old one with the live
// records only and renames it over the old one, which stays complete until then.
//
// Adapted from upstream ec8b75b2 (store and prewarm), dd5bc26c (background writer) and 83f8bfd3
// (console messages).

enum class ShaderStoreMode : uint8_t { Off, On, Verify };

// KYTY_SHADER_STORE=0/1/2 (off, on, on and retranslate each loaded module at its first use to
// check it), else --shader-cache. Off in non-Release builds and with graphics debug dumps.
[[nodiscard]] ShaderStoreMode GetShaderStoreMode();

// A program of the program cache: its key and the SaveResourcePlan of its resources.
struct ShaderStoreProgram {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	uint32_t              code_size       = 0;
	std::vector<uint32_t> static_state;
	std::vector<uint8_t>  plan;
};

struct ShaderStoreModule {
	// The key of the module's program.
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	uint32_t              code_size       = 0;
	std::vector<uint32_t> static_state;
	// Without a handle module; stored programs have no function code.
	ShaderModule          module;
	// Pixel programs: the pretranslation id of their translation key, 0 otherwise.
	uint64_t              translation_id = 0;
	// AppendModule: the SPIR-V. Loaded modules leave it empty; ReadSpirv(spirv_index) reads it.
	std::vector<uint32_t> spirv;
	uint32_t              spirv_index = 0;
	// Loaded modules: how many launches wrote or used it (at most 255).
	uint32_t              uses = 0;
};

struct ShaderStorePipeline {
	PipelineRenderingState   rendering;
	std::array<uint64_t, 3>  vertex_ids {};
	uint64_t                 ps_id = 0;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
	// Loaded pipelines: the record id, for NoteUsedPipeline, and how many launches wrote or used
	// it (at most 255).
	uint64_t                 id   = 0;
	uint32_t                 uses = 0;
};

struct ShaderStoreContents {
	// The live records, in file order. Every module's program is in `programs`, and every
	// pipeline's modules are in `modules`.
	std::vector<ShaderStoreProgram>  programs;
	std::vector<ShaderStoreModule>   modules;
	std::vector<ShaderStorePipeline> pipelines;
	// The number of this launch: one more than the last launch the store saw.
	uint32_t                         launch       = 1;
	uint64_t                         file_bytes   = 0;
	// The intact part of the file, then what of it is live (header included).
	uint64_t                         kept_bytes   = 0;
	uint64_t                         live_bytes   = 0;
	uint32_t                         dead_records = 0;
	bool                             header_match = false;
	// Open: the store thread rewrites the file with the live records before it appends.
	bool                             compacting   = false;
};

// Host state that changes the emitted SPIR-V; part of the store header.
std::vector<uint8_t> ShaderStoreHostKey(const GraphicContext& graphics);

class ShaderStore {
public:
	// A record unused for this many launches is dead.
	static constexpr uint32_t RetainedLaunches = 2;
	// The prewarm creates the pipelines of records that this many launches wrote or used; the
	// others, often variants of one launch only, are created when a draw needs them.
	static constexpr uint32_t PrewarmUses = 2;

	ShaderStore();
	~ShaderStore();
	KYTY_CLASS_NO_COPY(ShaderStore);

	// Reads the live records of the store at `path` whose header matches `host_key`, then keeps
	// the file open to append; a mismatching or unreadable store starts over. Call once.
	ShaderStoreContents Open(const std::filesystem::path& path, std::span<const uint8_t> host_key);
	[[nodiscard]] bool IsOpen() const { return m_thread.joinable(); }

	// Encode on the calling thread; compression and writes run on the store thread, in order.
	// Records the store holds already are not written again, nor pipelines whose modules it does
	// not hold. Call these and the Note functions from one thread at a time.
	void AppendProgram(const ShaderStoreProgram& program);
	void AppendModule(const ShaderStoreModule& module);
	void AppendPipeline(const ShaderStorePipeline& pipeline);
	// A loaded module or pipeline was used in this launch; later calls for it do nothing.
	void NoteUsedModule(uint64_t id);
	void NoteUsedPipeline(uint64_t id);

	// The SPIR-V of a loaded module (ShaderStoreModule::spirv_index, handle id); empty when the
	// store cannot read it back. Any thread.
	[[nodiscard]] std::vector<uint32_t> ReadSpirv(uint32_t index, uint64_t module_id);

	// Waits until the store thread wrote everything posted and noted so far, and the compaction
	// (tests).
	void WaitForWrites();

	// Parses `bytes` (a store file) as Open does, for tests.
	static ShaderStoreContents Parse(std::span<const uint8_t> bytes,
	                                 std::span<const uint8_t> host_key);

	struct Layout;

private:
	struct Pending {
		uint8_t              type = 0;
		uint64_t             id   = 0;
		std::vector<uint8_t> raw;
	};

	void Post(uint8_t type, uint64_t id, std::vector<uint8_t>&& raw);
	void Run();
	void WriteFrame(uint8_t type, uint64_t id, std::span<const uint8_t> raw,
	                std::vector<uint8_t>& frame_bytes);
	bool Compact();

	std::filesystem::path m_path;
	uint32_t              m_launch = 0;
	// The append handle, store thread only once it runs.
	Common::File          m_file;

	// The read handle and where each loaded module's SPIR-V frame is in the file.
	std::mutex            m_read_mutex;
	Common::File          m_read;
	std::vector<uint64_t> m_spirv_offsets;

	// What the file holds, and the loaded modules and pipelines not used yet (Append and Note
	// callers).
	std::unordered_set<uint64_t> m_known_programs;
	std::unordered_set<uint64_t> m_known_modules;
	std::unordered_set<uint64_t> m_known_pipelines;
	std::unordered_set<uint64_t> m_unused_modules;
	std::unordered_set<uint64_t> m_unused_pipelines;

	// The live frames of the file read by Open while the store thread compacts it.
	std::unique_ptr<Layout> m_compaction;

	std::mutex              m_mutex;
	std::condition_variable m_ready;
	std::condition_variable m_idle;
	std::deque<Pending>     m_records;
	std::vector<uint64_t>   m_used_modules;
	std::vector<uint64_t>   m_used_pipelines;
	bool                    m_busy  = true;
	bool                    m_flush = false;
	bool                    m_stop  = false;
	std::atomic<bool>       m_abort {false};
	std::thread             m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORE_H_
