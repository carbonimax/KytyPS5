#include "graphics/host_gpu/renderer/pipeline/shaderStore.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <string>
#include <unordered_map>
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
// Bumped when the frame or record encoding changes.
constexpr uint32_t StoreFormatVersion = 2;

enum class Record : uint8_t {
	Program          = 1,
	Module           = 2,
	Spirv            = 3,
	GraphicsPipeline = 4,
	// The loaded modules and pipelines a launch used.
	Used             = 5,
	// Written first by each launch.
	Launch           = 6,
};

// Precedes each record's zstd frame.
struct Frame {
	uint32_t               compressed_size = 0;
	uint32_t               raw_size        = 0;
	// XXH3 of the compressed bytes, seeded with the rest of the frame.
	uint64_t               checksum        = 0;
	// Programs: ProgramId; modules and their SPIR-V: the handle id; pipelines: XXH3 of the record.
	uint64_t               id              = 0;
	// The last launch that wrote or used the record.
	uint32_t               launch          = 0;
	Record                 type            = Record::Launch;
	// How many launches wrote or used it as of `launch` (at most 255).
	uint8_t                uses            = 1;
	std::array<uint8_t, 2> reserved {};
};
static_assert(sizeof(Frame) == 32);

// Larger frames are damage.
constexpr uint32_t MaxFrameBytes = 1u << 30u;

// Dead records start a compaction once they are this part of the file.
constexpr uint64_t CompactDeadDivisor = 4;

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

uint64_t FrameChecksum(Frame frame, std::span<const uint8_t> compressed) {
	frame.checksum = 0;
	return XXH3_64bits_withSeed(compressed.data(), compressed.size(),
	                            XXH3_64bits(&frame, sizeof(frame)));
}

bool Decompress(std::span<const uint8_t> compressed, void* raw, size_t raw_size) {
	return ZSTD_getFrameContentSize(compressed.data(), compressed.size()) == raw_size &&
	       ZSTD_decompress(raw, raw_size, compressed.data(), compressed.size()) == raw_size;
}

std::filesystem::path TempPath(const std::filesystem::path& path) {
	auto temp = path;
	temp += ".tmp";
	return temp;
}

template <class S, class P> void IoProgram(S& io, P& program) {
	IoAll(io, program.stage, program.hash, program.user_data_count, program.code_size,
	      program.static_state, program.plan);
}

template <class S, class M> void IoModule(S& io, M& record) {
	auto& module = record.module;
	IoAll(io, record.stage, record.hash, record.user_data_count, record.code_size,
	      record.static_state, module.handle.id, module.program, module.specialization,
	      module.vertex_formats, module.rect_parameters, record.translation_id);
}

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

// The id of the program a program or module record belongs to.
template <class R> uint64_t ProgramId(const R& record) {
	Writer key;
	IoAll(key, record.stage, record.hash, record.user_data_count, record.code_size,
	      record.static_state);
	const auto id = XXH3_64bits(key.bytes.data(), key.bytes.size());
	return id != 0 ? id : 1;
}

std::vector<uint8_t> FileHeader(std::span<const uint8_t> host_key) {
	Writer header;
	header.Raw(StoreMagic.data(), StoreMagic.size());
	const auto size = static_cast<uint32_t>(host_key.size());
	IoAll(header, StoreFormatVersion, size);
	header.Raw(host_key.data(), host_key.size());
	return std::move(header.bytes);
}

struct FrameView {
	Frame                    frame;
	uint64_t                 offset = 0;
	std::span<const uint8_t> compressed;

	[[nodiscard]] uint64_t Size() const { return sizeof(Frame) + compressed.size(); }
};

// The intact frame at `offset`, which then moves past it.
bool NextFrame(std::span<const uint8_t> bytes, uint64_t& offset, FrameView& view) {
	if (bytes.size() - offset < sizeof(Frame)) {
		return false;
	}
	std::memcpy(&view.frame, bytes.data() + offset, sizeof(Frame));
	const auto& frame = view.frame;
	if (frame.compressed_size > MaxFrameBytes ||
	    bytes.size() - offset - sizeof(Frame) < frame.compressed_size) {
		return false;
	}
	view.compressed = bytes.subspan(offset + sizeof(Frame), frame.compressed_size);
	if (FrameChecksum(frame, view.compressed) != frame.checksum) {
		return false;
	}
	view.offset = offset;
	offset += view.Size();
	return true;
}

// Decodes the record of an intact frame with `io`; false unless it takes every byte.
template <class T, class F> bool DecodeFrame(const FrameView& view, T& value, F&& io) {
	std::vector<uint8_t> raw(view.frame.raw_size);
	if (raw.empty() || !Decompress(view.compressed, raw.data(), raw.size())) {
		return false;
	}
	Reader reader {raw};
	io(reader, value);
	return reader.ok && reader.bytes.empty();
}

const auto DecodeProgram = [](Reader& reader, ShaderStoreProgram& value) {
	IoProgram(reader, value);
};
const auto DecodeModule = [](Reader& reader, ShaderStoreModule& value) {
	IoModule(reader, value);
};
const auto DecodePipeline = [](Reader& reader, ShaderStorePipeline& value) {
	IoPipeline(reader, value);
};

using UsedLists = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;
const auto DecodeUsed = [](Reader& reader, UsedLists& lists) {
	IoAll(reader, lists.first, lists.second);
};

} // namespace

// What of a parsed store file compaction keeps.
struct ShaderStore::Layout {
	struct Kept {
		uint64_t offset = 0;
		uint32_t size   = 0;
		uint32_t launch = 0;
		uint8_t  uses   = 1;
	};
	std::vector<uint8_t>  header;
	// In file order.
	std::vector<Kept>     frames;
	// The index in `frames` of each loaded module's SPIR-V (ShaderStoreModule::spirv_index).
	std::vector<uint32_t> spirv_frames;
};

namespace {

// `layout`: what compaction keeps; `spirv_offsets`: where each loaded module's SPIR-V is.
ShaderStoreContents ParseStore(std::span<const uint8_t> bytes, std::span<const uint8_t> host_key,
                               ShaderStore::Layout* layout, std::vector<uint64_t>* spirv_offsets) {
	ShaderStoreContents contents;
	contents.file_bytes = bytes.size();
	const auto header   = FileHeader(host_key);
	if (bytes.size() < header.size() || !std::ranges::equal(bytes.first(header.size()), header)) {
		return contents;
	}
	contents.header_match = true;

	// Every intact frame, and the last launch that wrote or used each module and pipeline.
	std::vector<FrameView>                 frames;
	std::unordered_map<uint64_t, uint32_t> module_use;
	std::unordered_map<uint64_t, uint32_t> pipeline_use;
	// The launches whose use lists name each module and pipeline.
	std::unordered_map<uint64_t, std::vector<uint32_t>> module_launches;
	std::unordered_map<uint64_t, std::vector<uint32_t>> pipeline_launches;
	std::unordered_map<uint64_t, size_t>   program_frames;
	std::unordered_map<uint64_t, size_t>   spirv_frames;
	const auto use = [](std::unordered_map<uint64_t, uint32_t>& uses, uint64_t id, uint32_t launch) {
		auto& last = uses[id];
		last       = std::max(last, launch);
	};
	uint32_t  last_launch = 0;
	uint64_t  offset      = header.size();
	FrameView view;
	while (NextFrame(bytes, offset, view)) {
		const auto& frame = view.frame;
		last_launch       = std::max(last_launch, frame.launch);
		switch (frame.type) {
			case Record::Program: program_frames.try_emplace(frame.id, frames.size()); break;
			case Record::Module: use(module_use, frame.id, frame.launch); break;
			case Record::Spirv: spirv_frames.try_emplace(frame.id, frames.size()); break;
			case Record::GraphicsPipeline: use(pipeline_use, frame.id, frame.launch); break;
			case Record::Used: {
				UsedLists used;
				if (DecodeFrame(view, used, DecodeUsed)) {
					for (const auto id: used.first) {
						use(module_use, id, frame.launch);
						module_launches[id].push_back(frame.launch);
					}
					for (const auto id: used.second) {
						use(pipeline_use, id, frame.launch);
						pipeline_launches[id].push_back(frame.launch);
					}
				}
				break;
			}
			default: break;
		}
		frames.push_back(view);
	}
	contents.kept_bytes = offset;
	contents.launch     = last_launch + 1;
	const auto live     = [&contents](uint32_t used) {
		return used + ShaderStore::RetainedLaunches >= contents.launch;
	};
	// The record's own count, plus the later launches whose use lists name it.
	const auto count_uses = [](std::unordered_map<uint64_t, std::vector<uint32_t>>& launches,
	                           const Frame& frame) {
		uint32_t uses = frame.uses;
		if (const auto found = launches.find(frame.id); found != launches.end()) {
			auto& list = found->second;
			std::ranges::sort(list);
			const auto [first, last] = std::ranges::unique(list);
			list.erase(first, last);
			uses += static_cast<uint32_t>(std::ranges::count_if(
			    list, [&frame](uint32_t launch) { return launch > frame.launch; }));
		}
		return static_cast<uint8_t>(std::min(uses, 255u));
	};
	std::vector<uint8_t> counts(frames.size(), 1);

	// Live modules: the first record of each id, used recently, with its SPIR-V and its program.
	std::vector<uint32_t>                  tags(frames.size(), 0);
	std::vector<bool>                      keep(frames.size(), false);
	std::vector<size_t>                    module_frames;
	std::vector<size_t>                    module_spirv;
	std::vector<uint64_t>                  module_programs;
	std::unordered_map<uint64_t, uint32_t> program_use;
	std::unordered_set<uint64_t>           seen;
	for (size_t i = 0; i < frames.size(); i++) {
		const auto& frame = frames[i].frame;
		if (frame.type != Record::Module || !seen.insert(frame.id).second ||
		    !live(module_use[frame.id])) {
			continue;
		}
		const auto        spirv = spirv_frames.find(frame.id);
		ShaderStoreModule record;
		if (spirv == spirv_frames.end() || !DecodeFrame(frames[i], record, DecodeModule) ||
		    record.module.handle.id != frame.id) {
			continue;
		}
		const auto program = ProgramId(record);
		if (!program_frames.contains(program)) {
			continue;
		}
		const auto last = module_use[frame.id];
		keep[i] = keep[spirv->second] = true;
		tags[i] = tags[spirv->second] = last;
		counts[i] = counts[spirv->second] = count_uses(module_launches, frame);
		record.uses = counts[i];
		use(program_use, program, last);
		module_frames.push_back(i);
		module_spirv.push_back(spirv->second);
		module_programs.push_back(program);
		contents.modules.push_back(std::move(record));
	}

	// Their programs, in file order; a program that does not decode takes its modules with it.
	std::vector<size_t> program_order;
	for (const auto& [program, last]: program_use) {
		program_order.push_back(program_frames.at(program));
	}
	std::ranges::sort(program_order);
	std::unordered_set<uint64_t> failed_programs;
	for (const auto index: program_order) {
		const auto&        frame = frames[index].frame;
		ShaderStoreProgram record;
		if (!DecodeFrame(frames[index], record, DecodeProgram) || ProgramId(record) != frame.id) {
			failed_programs.insert(frame.id);
			continue;
		}
		keep[index] = true;
		tags[index] = program_use.at(frame.id);
		contents.programs.push_back(std::move(record));
	}
	std::unordered_set<uint64_t> live_modules;
	size_t                       kept_modules = 0;
	for (size_t i = 0; i < contents.modules.size(); i++) {
		if (failed_programs.contains(module_programs[i])) {
			keep[module_frames[i]] = keep[module_spirv[i]] = false;
			continue;
		}
		live_modules.insert(contents.modules[i].module.handle.id);
		module_spirv[kept_modules]       = module_spirv[i];
		contents.modules[kept_modules++] = std::move(contents.modules[i]);
	}
	contents.modules.resize(kept_modules);
	module_spirv.resize(kept_modules);

	// Live pipelines: the first record of each id, used recently, whose modules are live.
	const auto module_live = [&live_modules](uint64_t id) {
		return id == 0 || live_modules.contains(id);
	};
	seen.clear();
	for (size_t i = 0; i < frames.size(); i++) {
		const auto& frame = frames[i].frame;
		if (frame.type != Record::GraphicsPipeline || !seen.insert(frame.id).second ||
		    !live(pipeline_use[frame.id])) {
			continue;
		}
		ShaderStorePipeline record;
		if (!DecodeFrame(frames[i], record, DecodePipeline) || record.vertex_ids[0] == 0 ||
		    !std::ranges::all_of(record.vertex_ids, module_live) || !module_live(record.ps_id)) {
			continue;
		}
		keep[i]   = true;
		tags[i]     = pipeline_use[frame.id];
		counts[i]   = count_uses(pipeline_launches, frame);
		record.id   = frame.id;
		record.uses = counts[i];
		contents.pipelines.push_back(record);
	}

	// Launch and use lists, duplicates and expired records are dead.
	contents.live_bytes = header.size();
	std::vector<uint32_t> kept_index(frames.size(), 0);
	for (size_t i = 0; i < frames.size(); i++) {
		if (!keep[i]) {
			contents.dead_records++;
			continue;
		}
		contents.live_bytes += frames[i].Size();
		if (layout != nullptr) {
			kept_index[i] = static_cast<uint32_t>(layout->frames.size());
			layout->frames.push_back({.offset = frames[i].offset,
			                          .size   = static_cast<uint32_t>(frames[i].Size()),
			                          .launch = tags[i],
			                          .uses   = counts[i]});
		}
	}
	for (size_t i = 0; i < contents.modules.size(); i++) {
		contents.modules[i].spirv_index = static_cast<uint32_t>(i);
		if (layout != nullptr) {
			layout->spirv_frames.push_back(kept_index[module_spirv[i]]);
		}
		if (spirv_offsets != nullptr) {
			spirv_offsets->push_back(frames[module_spirv[i]].offset);
		}
	}
	return contents;
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
                                       std::span<const uint8_t> host_key) {
	return ParseStore(bytes, host_key, nullptr, nullptr);
}

ShaderStoreContents ShaderStore::Open(const std::filesystem::path& path,
                                      std::span<const uint8_t>     host_key) {
	EXIT_IF(IsOpen() || host_key.empty());
	m_path = path;
	// A compaction cut short leaves its file; the store is still the old file, whole.
	std::error_code error;
	std::filesystem::remove(TempPath(path), error);
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
	auto layout   = std::make_unique<Layout>();
	auto contents = ParseStore(bytes, host_key, layout.get(), &m_spirv_offsets);
	bytes         = {};
	m_launch      = contents.launch;
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
	if (m_file.IsInvalid() || !m_read.Open(path, Common::File::Mode::Read)) {
		m_file.Close();
		m_read.Close();
		return contents;
	}
	for (const auto& program: contents.programs) {
		m_known_programs.insert(ProgramId(program));
	}
	for (const auto& module: contents.modules) {
		m_known_modules.insert(module.module.handle.id);
	}
	for (const auto& pipeline: contents.pipelines) {
		m_known_pipelines.insert(pipeline.id);
	}
	m_unused_modules   = m_known_modules;
	m_unused_pipelines = m_known_pipelines;
	contents.compacting = contents.kept_bytes != 0 &&
	                      (contents.kept_bytes - contents.live_bytes) * CompactDeadDivisor >
	                          contents.kept_bytes;
	if (contents.compacting) {
		layout->header = FileHeader(host_key);
		m_compaction   = std::move(layout);
	}
	// Each launch writes its number first, so that the next one counts it even if it uses nothing.
	Writer launch;
	IoAll(launch, m_launch);
	Post(static_cast<uint8_t>(Record::Launch), m_launch, std::move(launch.bytes));
	m_thread = std::thread([this] { Run(); });
	return contents;
}

ShaderStore::ShaderStore() = default;

ShaderStore::~ShaderStore() {
	if (m_thread.joinable()) {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
		}
		// A compaction still running gives up; the old file is whole.
		m_abort = true;
		m_ready.notify_one();
		m_thread.join();
	}
	m_file.Close();
	m_read.Close();
}

void ShaderStore::AppendProgram(const ShaderStoreProgram& program) {
	if (!IsOpen()) {
		return;
	}
	const auto id = ProgramId(program);
	if (!m_known_programs.insert(id).second) {
		return;
	}
	Writer record;
	IoProgram(record, program);
	Post(static_cast<uint8_t>(Record::Program), id, std::move(record.bytes));
}

void ShaderStore::AppendModule(const ShaderStoreModule& module) {
	const auto id = module.module.handle.id;
	if (!IsOpen() || module.spirv.empty() || id == 0 ||
	    !m_known_programs.contains(ProgramId(module)) || !m_known_modules.insert(id).second) {
		return;
	}
	Writer record;
	IoModule(record, module);
	Post(static_cast<uint8_t>(Record::Module), id, std::move(record.bytes));
	std::vector<uint8_t> spirv(module.spirv.size() * sizeof(uint32_t));
	std::memcpy(spirv.data(), module.spirv.data(), spirv.size());
	Post(static_cast<uint8_t>(Record::Spirv), id, std::move(spirv));
}

void ShaderStore::AppendPipeline(const ShaderStorePipeline& pipeline) {
	if (!IsOpen()) {
		return;
	}
	// A pipeline with a module the store lacks (a program with calls) could never be prewarmed.
	const auto known = [this](uint64_t id) { return id == 0 || m_known_modules.contains(id); };
	if (pipeline.vertex_ids[0] == 0 || !std::ranges::all_of(pipeline.vertex_ids, known) ||
	    !known(pipeline.ps_id)) {
		return;
	}
	Writer record;
	IoPipeline(record, pipeline);
	const auto id = XXH3_64bits(record.bytes.data(), record.bytes.size());
	if (!m_known_pipelines.insert(id).second) {
		// Loaded but not prewarmed: it stays while it is used.
		NoteUsedPipeline(id);
		return;
	}
	Post(static_cast<uint8_t>(Record::GraphicsPipeline), id, std::move(record.bytes));
}

void ShaderStore::NoteUsedModule(uint64_t id) {
	if (!IsOpen() || m_unused_modules.erase(id) == 0) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	m_used_modules.push_back(id);
}

void ShaderStore::NoteUsedPipeline(uint64_t id) {
	if (!IsOpen() || m_unused_pipelines.erase(id) == 0) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	m_used_pipelines.push_back(id);
}

std::vector<uint32_t> ShaderStore::ReadSpirv(uint32_t index, uint64_t module_id) {
	Frame                frame;
	std::vector<uint8_t> compressed;
	{
		std::scoped_lock lock(m_read_mutex);
		if (index >= m_spirv_offsets.size() || m_read.IsInvalid() ||
		    !m_read.Seek(m_spirv_offsets[index])) {
			return {};
		}
		uint32_t read = 0;
		m_read.Read(&frame, sizeof(frame), &read);
		if (read != sizeof(frame) || frame.compressed_size > MaxFrameBytes) {
			return {};
		}
		compressed.resize(frame.compressed_size);
		m_read.Read(compressed.data(), frame.compressed_size, &read);
		if (read != frame.compressed_size) {
			return {};
		}
	}
	if (frame.type != Record::Spirv || frame.id != module_id || frame.raw_size == 0 ||
	    frame.raw_size % sizeof(uint32_t) != 0 ||
	    FrameChecksum(frame, compressed) != frame.checksum) {
		return {};
	}
	std::vector<uint32_t> spirv(frame.raw_size / sizeof(uint32_t));
	if (!Decompress(compressed, spirv.data(), frame.raw_size)) {
		return {};
	}
	return spirv;
}

void ShaderStore::WaitForWrites() {
	if (!IsOpen()) {
		return;
	}
	std::unique_lock lock(m_mutex);
	m_flush = true;
	m_ready.notify_one();
	m_idle.wait(lock, [this] {
		return !m_busy && m_records.empty() && m_used_modules.empty() && m_used_pipelines.empty();
	});
}

void ShaderStore::Post(uint8_t type, uint64_t id, std::vector<uint8_t>&& raw) {
	{
		std::scoped_lock lock(m_mutex);
		m_records.push_back({.type = type, .id = id, .raw = std::move(raw)});
	}
	m_ready.notify_one();
}

void ShaderStore::WriteFrame(uint8_t type, uint64_t id, std::span<const uint8_t> raw,
                             std::vector<uint8_t>& frame_bytes) {
	if (m_file.IsInvalid()) {
		return;
	}
	frame_bytes.resize(sizeof(Frame) + ZSTD_compressBound(raw.size()));
	const auto size = ZSTD_compress(frame_bytes.data() + sizeof(Frame),
	                                frame_bytes.size() - sizeof(Frame), raw.data(), raw.size(), 3);
	if (ZSTD_isError(size) || size > MaxFrameBytes) {
		return;
	}
	Frame frame {.compressed_size = static_cast<uint32_t>(size),
	             .raw_size        = static_cast<uint32_t>(raw.size()),
	             .id              = id,
	             .launch          = m_launch,
	             .type            = static_cast<Record>(type)};
	frame.checksum = FrameChecksum(frame, std::span(frame_bytes).subspan(sizeof(Frame), size));
	std::memcpy(frame_bytes.data(), &frame, sizeof(frame));
	// A failed write leaves a partial record; the next run drops it and what follows.
	m_file.Write(frame_bytes.data(), static_cast<uint32_t>(sizeof(Frame) + size));
}

// Copies the live frames of the store to a new file, each tagged with the last launch that used
// it, then renames that over the store. The old file stays whole until the rename.
bool ShaderStore::Compact() {
	const auto&  layout = *m_compaction;
	const auto   temp   = TempPath(m_path);
	Common::File source;
	Common::File target;
	bool         ok      = source.Open(m_path, Common::File::Mode::Read) && target.Create(temp);
	uint32_t     written = 0;
	if (ok) {
		target.Write(layout.header.data(), static_cast<uint32_t>(layout.header.size()), &written);
		ok = written == layout.header.size();
	}
	std::vector<uint64_t> offsets(layout.frames.size());
	std::vector<uint8_t>  bytes;
	uint64_t              position = layout.header.size();
	for (size_t i = 0; ok && i < layout.frames.size(); i++) {
		const auto& kept = layout.frames[i];
		bytes.resize(kept.size);
		uint32_t read = 0;
		ok            = !m_abort && source.Seek(kept.offset);
		if (ok) {
			source.Read(bytes.data(), kept.size, &read);
			ok = read == kept.size;
		}
		Frame      frame;
		const auto compressed = std::span<const uint8_t>(bytes).subspan(sizeof(Frame));
		if (ok) {
			std::memcpy(&frame, bytes.data(), sizeof(frame));
			ok = FrameChecksum(frame, compressed) == frame.checksum;
		}
		if (!ok) {
			break;
		}
		frame.launch   = kept.launch;
		frame.uses     = kept.uses;
		frame.checksum = FrameChecksum(frame, compressed);
		std::memcpy(bytes.data(), &frame, sizeof(frame));
		target.Write(bytes.data(), kept.size, &written);
		ok         = written == kept.size;
		offsets[i] = position;
		position += kept.size;
	}
	ok = ok && target.Flush();
	if (!source.IsInvalid()) {
		source.Close();
	}
	if (!target.IsInvalid()) {
		target.Close();
	}
	std::error_code error;
	if (!ok) {
		std::filesystem::remove(temp, error);
		return false;
	}
	bool renamed = false;
	{
		// Both handles are closed for the rename; ReadSpirv waits meanwhile.
		std::scoped_lock lock(m_read_mutex);
		m_read.Close();
		m_file.Close();
		std::filesystem::rename(temp, m_path, error);
		renamed = !error;
		if (renamed) {
			for (size_t i = 0; i < m_spirv_offsets.size(); i++) {
				m_spirv_offsets[i] = offsets[layout.spirv_frames[i]];
			}
		}
		if (!m_read.Open(m_path, Common::File::Mode::Read)) {
			m_read.Close();
		}
		if (m_file.Open(m_path, Common::File::Mode::ReadWrite)) {
			m_file.Seek(m_file.Size());
		} else {
			m_file.Close();
		}
	}
	if (!renamed) {
		std::filesystem::remove(temp, error);
	}
	return renamed;
}

void ShaderStore::Run() {
	if (m_compaction) {
		const auto start = std::chrono::steady_clock::now();
		uint64_t   bytes = m_compaction->header.size();
		for (const auto& frame: m_compaction->frames) {
			bytes += frame.size;
		}
		const bool compacted = Compact();
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader store: {} to {:.1f} MB ({} records) in {} ms\n",
		    compacted ? "compacted" : "left uncompacted", static_cast<double>(bytes) / 1'000'000.0,
		    m_compaction->frames.size(),
		    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
		                                                          start)
		        .count()));
		m_compaction.reset();
	}
	std::vector<uint8_t>  frame_bytes;
	std::deque<Pending>   batch;
	std::vector<uint64_t> used_modules;
	std::vector<uint64_t> used_pipelines;
	std::unique_lock      lock(m_mutex);
	for (;;) {
		m_busy = false;
		m_idle.notify_all();
		// Use lists wait for records, for a second or for a flush, so that they come in few frames.
		m_ready.wait_for(lock, std::chrono::seconds(1), [this] {
			return m_stop || m_flush || !m_records.empty();
		});
		m_flush = false;
		if (m_records.empty() && m_used_modules.empty() && m_used_pipelines.empty()) {
			if (m_stop) {
				return; // Stopped with nothing left to write.
			}
			continue;
		}
		m_busy = true;
		batch.swap(m_records);
		used_modules.swap(m_used_modules);
		used_pipelines.swap(m_used_pipelines);
		lock.unlock();
		for (const auto& record: batch) {
			WriteFrame(record.type, record.id, record.raw, frame_bytes);
		}
		if (!used_modules.empty() || !used_pipelines.empty()) {
			Writer used;
			IoAll(used, used_modules, used_pipelines);
			WriteFrame(static_cast<uint8_t>(Record::Used), 0, used.bytes, frame_bytes);
		}
		if (!m_file.IsInvalid()) {
			m_file.Flush();
		}
		batch.clear();
		used_modules.clear();
		used_pipelines.clear();
		lock.lock();
	}
}

} // namespace Libs::Graphics
