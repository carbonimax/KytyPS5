#include "graphics/host_gpu/renderer/pipeline/shaderStore.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <xxhash.h>
#include <zstd.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

constexpr std::array<char, 8> StoreMagic {'K', 'Y', 'S', 'H', 'S', 'T', 'O', 'R'};
// Bumped when the record encoding changes.
constexpr uint32_t StoreFormatVersion = 1;

enum class Record : uint8_t { Module = 1, GraphicsPipeline = 2 };

// Precedes each record's zstd frame.
struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0; // XXH3 of the compressed bytes.
};
static_assert(sizeof(Frame) == 16);

std::filesystem::path ExecutablePath() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::vector<wchar_t> name(32768);
	const auto           length =
	    GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	if (length == 0 || length >= name.size()) {
		return {};
	}
	return std::filesystem::path(name.data());
#elif defined(__APPLE__)
	std::array<char, 4096> name {};
	auto                   size = static_cast<uint32_t>(name.size());
	if (_NSGetExecutablePath(name.data(), &size) != 0) {
		return {};
	}
	return std::filesystem::path(name.data());
#else
	return std::filesystem::path("/proc/self/exe");
#endif
}

// The content of the running executable, so that any rebuild of the recompiler starts the store
// over. Zero when the executable cannot be read.
uint64_t BuildHash() {
	const auto path = ExecutablePath();
	if (path.empty()) {
		return 0;
	}
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return 0;
	}
	XXH3_state_t* state = XXH3_createState();
	XXH3_64bits_reset(state);
	std::vector<uint8_t> chunk(4u << 20u);
	uint64_t             total = 0;
	for (;;) {
		uint32_t read = 0;
		file.Read(chunk.data(), static_cast<uint32_t>(chunk.size()), &read);
		if (read == 0) {
			break;
		}
		XXH3_64bits_update(state, chunk.data(), read);
		total += read;
	}
	file.Close();
	const auto hash = XXH3_64bits_digest(state);
	XXH3_freeState(state);
	return total != 0 && hash != 0 ? hash : 0;
}

// Switches read while translating or emitting shaders.
constexpr std::array ShaderEnvironment {
    "KYTY_MAD_SEPARATE_ROUNDING", "KYTY_PS_UNLAUNCHED_LANES",       "KYTY_PS_PACKED_WAVE32",
    "KYTY_VS_PACKED_WAVE32",      "KYTY_VERTEX_TABLE_BACKING_READS", "KYTY_SRT_PROTECTED_LOADS",
    "KYTY_PERVERTEX_EXPERIMENTAL",
};

template <class S, class P> void IoPipeline(S& io, P& pipeline) {
	auto& rendering = pipeline.rendering;
	IoAll(io, rendering.color_formats, rendering.depth_format, rendering.stencil_format,
	      rendering.color_count, rendering.depth_feedback, pipeline.vertex_ids, pipeline.ps_id);
	auto& input = pipeline.vertex_input;
	IoAll(io, input.binding_count, input.attribute_count);
	if constexpr (ShaderRecompiler::IR::Reading<S>) {
		io.ok = io.ok && input.binding_count <= input.bindings.size() &&
		        input.attribute_count <= input.attributes.size();
		if (!io.ok) {
			return;
		}
	}
	for (uint32_t i = 0; i < input.binding_count; i++) {
		IoAll(io, input.bindings[i].stride, input.bindings[i].instance);
	}
	for (uint32_t i = 0; i < input.attribute_count; i++) {
		IoAll(io, input.attributes[i].offset, input.attributes[i].binding);
	}
	io.Raw(&pipeline.static_params, sizeof(pipeline.static_params));
}

template <class S, class M> void IoModule(S& io, M& record) {
	auto& module = record.module;
	IoAll(io, record.stage, record.hash, record.user_data_count, record.code_size,
	      record.static_state, record.plan, module.handle.id, module.program, module.specialization,
	      module.vertex_formats, module.rect_parameters, record.translation_id, record.spirv);
}

std::vector<uint8_t> FileHeader(std::span<const uint8_t> host_key) {
	Writer header;
	header.Raw(StoreMagic.data(), StoreMagic.size());
	const auto size = static_cast<uint32_t>(host_key.size());
	IoAll(header, StoreFormatVersion, size);
	header.Raw(host_key.data(), host_key.size());
	return std::move(header.bytes);
}

// Decodes the record at the front of `records` and drops it from the span; false when it is
// incomplete or damaged.
bool ParseRecord(std::span<const uint8_t>& records, ShaderStoreContents& contents,
                 const ShaderStoreModuleSink& sink) {
	Frame frame;
	if (records.size() < sizeof(frame)) {
		return false;
	}
	std::memcpy(&frame, records.data(), sizeof(frame));
	if (records.size() - sizeof(frame) < frame.compressed_size) {
		return false;
	}
	const auto compressed = records.subspan(sizeof(frame), frame.compressed_size);
	if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return false;
	}
	std::vector<uint8_t> raw(frame.raw_size);
	if (ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) !=
	    raw.size()) {
		return false;
	}
	Reader reader {raw};
	auto   type     = Record::Module;
	IoAll(reader, type);
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	switch (type) {
		case Record::Module: {
			ShaderStoreModule module;
			IoModule(reader, module);
			if (!complete() || module.spirv.empty() || module.module.handle.id == 0) {
				return false;
			}
			if (sink) {
				sink(std::move(module));
			} else {
				contents.modules.push_back(std::move(module));
			}
			break;
		}
		case Record::GraphicsPipeline: {
			ShaderStorePipeline pipeline;
			IoPipeline(reader, pipeline);
			if (!complete()) {
				return false;
			}
			contents.pipelines.push_back(pipeline);
			break;
		}
		default: return false;
	}
	records = records.subspan(sizeof(frame) + compressed.size());
	return true;
}

} // namespace

ShaderStoreMode GetShaderStoreMode() {
	static const ShaderStoreMode mode = [] {
		if (KYTY_BUILD != KYTY_BUILD_RELEASE || Config::GraphicsDebugDumpEnabled()) {
			return ShaderStoreMode::Off;
		}
		if (const char* value = std::getenv("KYTY_SHADER_STORE"); value != nullptr && value[0] != 0) {
			switch (value[0]) {
				case '1': return ShaderStoreMode::On;
				case '2': return ShaderStoreMode::Verify;
				default: return ShaderStoreMode::Off;
			}
		}
		return Config::ShaderCacheEnabled() ? ShaderStoreMode::On : ShaderStoreMode::Off;
	}();
	return mode;
}

std::vector<uint8_t> ShaderStoreHostKey(const GraphicContext& graphics) {
	const auto build = BuildHash();
	if (build == 0) {
		return {};
	}
	const auto& properties = graphics.GetPhysicalDeviceProperties();
	Writer      key;
	// The SPIR-V profile and host features of ProgramCache, then what selects compute variants.
	IoAll(key, build, graphics.maintenance8_enabled, properties.vendorID,
	      graphics.device_clock_period_ps, graphics.shader_buffer_int64_atomics_enabled,
	      graphics.shader_cull_distance_enabled, graphics.subgroup_size,
	      graphics.vertex_subgroups_supported, graphics.shader_resource_min_lod_enabled,
	      graphics.null_descriptor_enabled, graphics.image_view_min_lod_enabled,
	      graphics.mesh_shader_enabled, graphics.SupportsComputeWave64(),
	      properties.limits.maxComputeSharedMemorySize, properties.deviceID,
	      properties.driverVersion);
	key.Raw(properties.pipelineCacheUUID.data(), properties.pipelineCacheUUID.size());
	for (const char* name: ShaderEnvironment) {
		const char* value = std::getenv(name);
		std::string entry = std::string(name) + "=" + (value != nullptr ? value : "");
		IoAll(key, entry);
	}
	return std::move(key.bytes);
}

ShaderStoreContents ShaderStore::Parse(std::span<const uint8_t> bytes,
                                       std::span<const uint8_t> host_key,
                                       const ShaderStoreModuleSink& sink) {
	ShaderStoreContents contents;
	contents.file_bytes = bytes.size();
	const auto header   = FileHeader(host_key);
	if (bytes.size() < header.size() || !std::ranges::equal(bytes.first(header.size()), header)) {
		return contents;
	}
	contents.header_match = true;
	auto records          = bytes.subspan(header.size());
	while (ParseRecord(records, contents, sink)) {
	}
	// A damaged record ends the store; the records before it stay.
	contents.kept_bytes = bytes.size() - records.size();
	return contents;
}

ShaderStoreContents ShaderStore::Open(const std::filesystem::path& path,
                                      std::span<const uint8_t>     host_key,
                                      const ShaderStoreModuleSink& sink) {
	EXIT_IF(IsOpen() || host_key.empty());
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		const auto size = file.Size();
		if (size <= std::numeric_limits<uint32_t>::max()) {
			bytes.resize(size);
			uint32_t read = 0;
			file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
			bytes.resize(read);
		}
	}
	auto contents = Parse(bytes, host_key, sink);
	bytes         = {};
	Common::File::CreateDirectories(path.parent_path());
	if (contents.kept_bytes == 0) {
		const auto header = FileHeader(host_key);
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(contents.kept_bytes);
		m_file.Seek(contents.kept_bytes);
	}
	if (!m_file.IsInvalid()) {
		m_thread = std::thread([this] { Run(); });
	}
	return contents;
}

ShaderStore::~ShaderStore() {
	if (m_thread.joinable()) {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
		}
		m_ready.notify_one();
		m_thread.join();
	}
	m_file.Close();
}

void ShaderStore::AppendModule(const ShaderStoreModule& module) {
	if (!IsOpen()) {
		return;
	}
	Writer record;
	IoAll(record, Record::Module);
	IoModule(record, module);
	Post(std::move(record.bytes));
}

void ShaderStore::AppendPipeline(const ShaderStorePipeline& pipeline) {
	if (!IsOpen()) {
		return;
	}
	Writer record;
	IoAll(record, Record::GraphicsPipeline);
	IoPipeline(record, pipeline);
	Post(std::move(record.bytes));
}

void ShaderStore::Post(std::vector<uint8_t>&& record) {
	{
		std::scoped_lock lock(m_mutex);
		m_records.push_back(std::move(record));
	}
	m_ready.notify_one();
}

void ShaderStore::Run() {
	std::vector<uint8_t>             frame_bytes;
	std::deque<std::vector<uint8_t>> batch;
	std::unique_lock                 lock(m_mutex);
	for (;;) {
		m_ready.wait(lock, [this] { return m_stop || !m_records.empty(); });
		if (m_records.empty()) {
			return; // Stopped with nothing left to write.
		}
		batch.swap(m_records);
		lock.unlock();
		for (const auto& record: batch) {
			frame_bytes.resize(sizeof(Frame) + ZSTD_compressBound(record.size()));
			const auto size = ZSTD_compress(frame_bytes.data() + sizeof(Frame),
			                                frame_bytes.size() - sizeof(Frame), record.data(),
			                                record.size(), 3);
			if (ZSTD_isError(size)) {
				continue;
			}
			const Frame frame {.compressed_size = static_cast<uint32_t>(size),
			                   .raw_size        = static_cast<uint32_t>(record.size()),
			                   .checksum = XXH3_64bits(frame_bytes.data() + sizeof(Frame), size)};
			std::memcpy(frame_bytes.data(), &frame, sizeof(frame));
			// A failed write leaves a partial record; the next run drops it and what follows.
			m_file.Write(frame_bytes.data(), static_cast<uint32_t>(sizeof(Frame) + size));
		}
		m_file.Flush();
		batch.clear();
		lock.lock();
	}
}

} // namespace Libs::Graphics
