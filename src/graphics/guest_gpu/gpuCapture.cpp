#include "graphics/guest_gpu/gpuCapture.h"

#include "common/logging/log.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <xxhash.h>

namespace Libs::Graphics::GpuCapture {

static_assert(std::endian::native == std::endian::little, "capture fields are stored in host order");

namespace {

constexpr char     Magic[8]   = {'K', 'Y', 'G', 'P', 'U', 'C', 'A', 'P'};
constexpr uint64_t HeaderSize = 16;

class Writer {
public:
	template <typename T>
	void Put(T value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* p = reinterpret_cast<const uint8_t*>(&value);
		out.insert(out.end(), p, p + sizeof(T));
	}
	void PutBytes(std::span<const uint8_t> bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); }

	std::vector<uint8_t> out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}

	template <typename T>
	bool Get(T& value) {
		if (m_data.size() - m_pos < sizeof(T)) return false;
		std::memcpy(&value, m_data.data() + m_pos, sizeof(T));
		m_pos += sizeof(T);
		return true;
	}
	bool GetBytes(std::vector<uint8_t>& bytes, uint64_t size) {
		if (m_data.size() - m_pos < size) return false;
		bytes.assign(m_data.begin() + static_cast<ptrdiff_t>(m_pos),
		             m_data.begin() + static_cast<ptrdiff_t>(m_pos + size));
		m_pos += size;
		return true;
	}
	[[nodiscard]] bool Done() const { return m_pos == m_data.size(); }

private:
	std::span<const uint8_t> m_data;
	uint64_t                 m_pos = 0;
};

void PutPayload(Writer& w, const Memory& m) {
	w.Put<uint64_t>(m.address);
	w.Put<uint64_t>(m.bytes.size());
	w.PutBytes(m.bytes);
}
void PutPayload(Writer& w, const Shader& s) {
	w.Put(s.address);
	w.Put(s.user_data);
	w.Put(s.input_semantics);
	w.Put(s.type);
	w.Put(s.code_size_bytes);
	w.Put(s.scratch_size_dwords);
	w.Put(s.num_input_semantics);
	w.Put(static_cast<uint32_t>(s.cx_registers.size()));
	w.Put(static_cast<uint32_t>(s.sh_registers.size()));
	for (const auto* list: {&s.cx_registers, &s.sh_registers}) {
		for (const auto& r: *list) {
			w.Put(r.offset);
			w.Put(r.value);
		}
	}
}
void PutPayload(Writer& w, const Submit& s) {
	w.Put(s.address);
	w.Put(s.queue);
	w.Put(s.dword_count);
}
void PutPayload(Writer& /*w*/, const FrameEnd& /*f*/) {}
void PutPayload(Writer& w, const Update& u) {
	w.Put(u.submit);
	w.Put<uint32_t>(0);
	PutPayload(w, Memory {u.address, u.bytes});
}
void PutPayload(Writer& w, const Zero& z) {
	w.Put(z.address);
	w.Put(z.size);
}
void PutPayload(Writer& w, const Expect& e) {
	w.Put(e.address);
	w.Put(e.size);
	w.Put(e.hash);
}

bool GetPayload(Reader& r, Memory& m) {
	uint64_t size = 0;
	return r.Get(m.address) && r.Get(size) && size != 0 && m.address + size > m.address &&
	       r.GetBytes(m.bytes, size);
}
bool GetPayload(Reader& r, Shader& s) {
	uint32_t cx = 0, sh = 0;
	if (!(r.Get(s.address) && r.Get(s.user_data) && r.Get(s.input_semantics) && r.Get(s.type) &&
	      r.Get(s.code_size_bytes) && r.Get(s.scratch_size_dwords) && r.Get(s.num_input_semantics) &&
	      r.Get(cx) && r.Get(sh))) {
		return false;
	}
	for (auto [list, count]: {std::pair {&s.cx_registers, cx}, std::pair {&s.sh_registers, sh}}) {
		for (uint32_t i = 0; i < count; i++) {
			Shader::Register reg;
			if (!r.Get(reg.offset) || !r.Get(reg.value)) return false;
			list->push_back(reg);
		}
	}
	return s.address != 0 && s.code_size_bytes != 0;
}
bool GetPayload(Reader& r, Submit& s) {
	return r.Get(s.address) && r.Get(s.queue) && r.Get(s.dword_count) && s.address != 0 && s.dword_count != 0;
}
bool GetPayload(Reader& /*r*/, FrameEnd& /*f*/) { return true; }
bool GetPayload(Reader& r, Update& u) {
	uint32_t reserved = 0;
	Memory   m;
	if (!r.Get(u.submit) || !r.Get(reserved) || reserved != 0 || !GetPayload(r, m)) return false;
	u.address = m.address;
	u.bytes   = std::move(m.bytes);
	return true;
}
bool GetPayload(Reader& r, Zero& z) {
	return r.Get(z.address) && r.Get(z.size) && z.size != 0 && z.address + z.size > z.address;
}
bool GetPayload(Reader& r, Expect& e) {
	return r.Get(e.address) && r.Get(e.size) && r.Get(e.hash) && e.size != 0 && e.address + e.size > e.address;
}

template <size_t I = 0>
bool ParseRecord(uint32_t type, Reader& payload, Record& record) {
	if constexpr (I < std::variant_size_v<Record>) {
		if (type == I + 1) {
			return GetPayload(payload, record.emplace<I>()) && payload.Done();
		}
		return ParseRecord<I + 1>(type, payload, record);
	} else {
		return false;
	}
}

} // namespace

void SerializeHeader(std::vector<uint8_t>& out) {
	Writer w;
	w.PutBytes({reinterpret_cast<const uint8_t*>(Magic), sizeof(Magic)});
	w.Put(Version);
	w.Put<uint32_t>(0);
	out.insert(out.end(), w.out.begin(), w.out.end());
}

void SerializeRecord(std::vector<uint8_t>& out, const Record& record) {
	Writer payload;
	std::visit([&](const auto& r) { PutPayload(payload, r); }, record);
	Writer w;
	w.Put(static_cast<uint32_t>(record.index() + 1));
	w.Put<uint32_t>(0);
	w.Put<uint64_t>(payload.out.size());
	out.insert(out.end(), w.out.begin(), w.out.end());
	out.insert(out.end(), payload.out.begin(), payload.out.end());
	out.resize((out.size() + 7u) & ~size_t {7});
}

std::vector<uint8_t> Serialize(const Capture& capture) {
	std::vector<uint8_t> out;
	SerializeHeader(out);
	for (const auto& record: capture.records) {
		SerializeRecord(out, record);
	}
	return out;
}

std::optional<Capture> Parse(std::span<const uint8_t> file, std::string* error) {
	const auto fail = [&](const std::string& message) {
		if (error != nullptr) *error = message;
		return std::nullopt;
	};
	Reader   header(file.first(std::min<size_t>(file.size(), HeaderSize)));
	char     magic[8] {};
	uint32_t version = 0, reserved = 0;
	if (!header.Get(magic) || !header.Get(version) || !header.Get(reserved) ||
	    std::memcmp(magic, Magic, sizeof(Magic)) != 0) {
		return fail("not a GPU capture");
	}
	if (version != Version || reserved != 0) {
		return fail("unsupported GPU capture version " + std::to_string(version));
	}
	Capture capture;
	uint64_t pos = HeaderSize;
	while (pos < file.size()) {
		Reader   head(file.subspan(pos));
		uint32_t type = 0, flags = 0;
		uint64_t size = 0;
		if (!head.Get(type) || !head.Get(flags) || !head.Get(size) || size > file.size() - pos - 16) {
			return fail("truncated record at offset " + std::to_string(pos));
		}
		Reader payload(file.subspan(pos + 16, size));
		Record record;
		if (flags != 0 || !ParseRecord(type, payload, record)) {
			return fail("bad record type " + std::to_string(type) + " at offset " + std::to_string(pos));
		}
		capture.records.push_back(std::move(record));
		pos = (pos + 16 + size + 7u) & ~uint64_t {7};
	}
	return capture;
}

const bool       g_enabled = std::getenv("KYTY_GPU_CAPTURE") != nullptr;
std::atomic_bool g_recording {false};

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t PageSize = 4096;

struct Recorder {
	std::mutex                                 mutex;
	FILE*                                      file = nullptr;
	std::string                                path;
	std::vector<uint8_t>                       out;
	std::unordered_map<uint64_t, uint64_t>     pages;   // recorded this frame -> XXH3 of the last bytes
	std::unordered_set<uint64_t>               shaders; // recorded this frame
	std::unordered_map<uint64_t, Shader>       kept;    // every registered shader
	std::vector<std::pair<uint64_t, uint64_t>> outputs;
	uint64_t          frame = 0, stored = 0, zero = 0, unreadable = 0, skipped = 0, submits = 0;
	uint64_t          missing_shaders = 0, written = 0, updates = 0;
	uint32_t          current = 0; // Submit index of the slice being processed
	Clock::duration   touch_time {}, publish_time {};
	Clock::time_point start;
};

Recorder& GetRecorder() {
	static Recorder recorder;
	return recorder;
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

// Caller holds the mutex.
void Write(Recorder& r, const Record& record) {
	if (r.file == nullptr) return;
	r.out.clear();
	SerializeRecord(r.out, record);
	r.written += r.out.size();
	if (std::fwrite(r.out.data(), 1, r.out.size(), r.file) != r.out.size()) {
		LOGF("GPUCAPTURE write failed, %s is incomplete\n", r.path.c_str());
		std::fclose(r.file);
		r.file = nullptr;
		g_recording.store(false, std::memory_order_relaxed);
	}
}

bool IsZeroPage(const uint8_t* p) {
	return std::all_of(p, p + PageSize, [](uint8_t b) { return b == 0; });
}

// Page-aligned [begin, end), read as the buffer cache uploads it (backing, private mappings, zero
// holes): zero pages become Zero records, pages outside guest memory are left out.
void RecordRun(Recorder& r, uint64_t begin, uint64_t end) {
	std::vector<uint8_t> bytes(end - begin);
	if (!LibKernel::Memory::CopySparseMemory(begin, bytes.data(), bytes.size())) {
		if (end - begin == PageSize) {
			r.unreadable += PageSize;
			return;
		}
		for (uint64_t page = begin; page < end; page += PageSize) RecordRun(r, page, page + PageSize);
		return;
	}
	for (uint64_t i = 0; i < bytes.size(); i += PageSize) {
		r.pages[begin + i] = XXH3_64bits(bytes.data() + i, PageSize);
	}
	for (uint64_t i = 0; i < bytes.size();) {
		const bool zero = IsZeroPage(bytes.data() + i);
		uint64_t   j    = i + PageSize;
		while (j < bytes.size() && IsZeroPage(bytes.data() + j) == zero) j += PageSize;
		if (zero) {
			Write(r, Zero {begin + i, j - i});
			r.zero += j - i;
		} else {
			Write(r, Memory {begin + i, {bytes.begin() + static_cast<ptrdiff_t>(i),
			                             bytes.begin() + static_cast<ptrdiff_t>(j)}});
			r.stored += j - i;
		}
		i = j;
	}
}

// Caller holds the mutex.
void TouchLocked(Recorder& r, uint64_t address, uint64_t size, bool output) {
	if (!Recording() || size == 0 || address == 0) return;
	// shortcut: ranges over 1 GiB are not recorded (bad descriptors); raise it if a game needs them.
	if (size > (uint64_t {1} << 30u) || address + size < address) {
		r.skipped++;
		return;
	}
	const auto start = Clock::now();
	if (output) r.outputs.emplace_back(address, size);
	// Pages recorded earlier in the frame and changed since (CPU rings) become Updates.
	// shortcut: only accesses up to 1 MiB are compared; raise it if a replay misses larger rewrites.
	const bool           compare = size <= (uint64_t {1} << 20u);
	std::vector<uint8_t> bytes;
	const uint64_t       end    = (address + size + PageSize - 1) & ~(PageSize - 1);
	uint64_t             run    = 0;
	bool                 in_run = false;
	for (uint64_t page = address & ~(PageSize - 1); page <= end; page += PageSize) {
		bool fresh = false;
		if (page < end) {
			const auto [it, inserted] = r.pages.try_emplace(page, 0);
			fresh                     = inserted;
			if (!inserted && compare) {
				bytes.resize(PageSize);
				if (LibKernel::Memory::CopySparseMemory(page, bytes.data(), PageSize)) {
					const auto hash = XXH3_64bits(bytes.data(), PageSize);
					if (hash != it->second) {
						it->second = hash;
						Write(r, Update {page, r.current, bytes});
						r.updates++;
					}
				}
			}
		}
		if (fresh && !in_run) {
			run    = page;
			in_run = true;
		} else if (!fresh && in_run) {
			RecordRun(r, run, page);
			in_run = false;
		}
	}
	r.touch_time += Clock::now() - start;
}

} // namespace

void TouchSlow(uint64_t address, uint64_t size, bool output) {
	auto&            r = GetRecorder();
	std::scoped_lock lock(r.mutex);
	TouchLocked(r, address, size, output);
}

uint32_t RecordSubmit(std::span<const uint32_t> commands, uint32_t queue) {
	const auto       address = reinterpret_cast<uint64_t>(commands.data());
	auto&            r       = GetRecorder();
	std::scoped_lock lock(r.mutex);
	if (!Recording()) return UINT32_MAX;
	Write(r, Submit {address, queue, static_cast<uint32_t>(commands.size())});
	r.current = static_cast<uint32_t>(r.submits++);
	// Updates of reused command memory belong to this Submit.
	TouchLocked(r, address, commands.size_bytes(), false);
	return r.current;
}

void ResumeSubmit(uint32_t index) {
	auto&            r = GetRecorder();
	std::scoped_lock lock(r.mutex);
	// shortcut: an index from an earlier recorded frame is taken as this frame's when it is in range;
	// only back-to-back captured frames can hit it, tag indices with the frame if they are needed.
	if (index < r.submits) r.current = index;
}

void KeepShader(Shader shader) {
	auto&            r = GetRecorder();
	std::scoped_lock lock(r.mutex);
	r.kept[shader.address] = std::move(shader);
}

void UseShader(uint64_t address) {
	auto&  r = GetRecorder();
	Shader shader;
	{
		std::scoped_lock lock(r.mutex);
		if (!Recording() || !r.shaders.insert(address).second) return;
		const auto it = r.kept.find(address);
		if (it == r.kept.end()) {
			r.missing_shaders++;
			return;
		}
		shader = it->second;
		Write(r, shader);
	}
	TouchSlow(address, shader.code_size_bytes, false);
	TouchSlow(shader.input_semantics, uint64_t {shader.num_input_semantics} * sizeof(ShaderSemantic), false);
	if (const auto* user_data = reinterpret_cast<const ShaderUserData*>(shader.user_data)) {
		TouchSlow(shader.user_data, sizeof(ShaderUserData), false);
		TouchSlow(reinterpret_cast<uint64_t>(user_data->direct_resource_offset),
		          uint64_t {user_data->direct_resource_count} * sizeof(uint16_t), false);
		for (uint32_t i = 0; i < 4; i++) {
			TouchSlow(reinterpret_cast<uint64_t>(user_data->sharp_resource_offset[i]),
			          uint64_t {user_data->sharp_resource_count[i]} * sizeof(ShaderSharp), false);
		}
	}
}

void FrameBoundary(const std::function<void()>& publish_gpu_data) {
	static const std::filesystem::path dir    = std::getenv("KYTY_GPU_CAPTURE");
	static const int64_t               target = [] {
		const char* v = std::getenv("KYTY_GPU_CAPTURE_FRAME");
		return v != nullptr ? std::strtoll(v, nullptr, 0) : int64_t {-1};
	}();
	auto&      r     = GetRecorder();
	const auto frame = ++r.frame; // suspend points processed = number of the frame that starts
	if (r.file != nullptr) {
		const auto end_start = Clock::now();
		g_recording.store(false, std::memory_order_relaxed);
		publish_gpu_data();
		std::scoped_lock lock(r.mutex);
		Write(r, FrameEnd {});
		// The outputs are published like the inputs, then hashed.
		auto& outputs = r.outputs;
		std::sort(outputs.begin(), outputs.end());
		std::vector<std::pair<uint64_t, uint64_t>> merged; // [begin, end)
		for (const auto& [address, size]: outputs) {
			if (!merged.empty() && address <= merged.back().second) {
				merged.back().second = std::max(merged.back().second, address + size);
			} else {
				merged.emplace_back(address, address + size);
			}
		}
		std::vector<uint8_t> bytes;
		uint64_t             expects = 0;
		for (const auto& [begin, end]: merged) {
			bytes.resize(end - begin);
			if (LibKernel::Memory::CopySparseMemory(begin, bytes.data(), bytes.size())) {
				Write(r, Expect {begin, end - begin, XXH3_64bits(bytes.data(), bytes.size())});
				expects++;
			}
		}
		const bool complete = r.file != nullptr && std::fclose(r.file) == 0;
		r.file          = nullptr;
		const auto line = fmt::format(
		    "GPUCAPTURE {} frame={} complete={} bytes={} memory={} zero={} unreadable={} skipped={} submits={} updates={} "
		    "shaders={} missing_shaders={} expects={} publish_start_ms={:.1f} record_ms={:.1f} "
		    "end_ms={:.1f} frame_ms={:.1f}",
		    r.path, frame - 1, complete, r.written, r.stored, r.zero, r.unreadable, r.skipped, r.submits, r.updates,
		    r.shaders.size(), r.missing_shaders, expects, Ms(r.publish_time), Ms(r.touch_time),
		    Ms(Clock::now() - end_start), Ms(end_start - r.start));
		LOGF("%s\n", line.c_str());
		if (FILE* log = std::fopen((dir / "capture.log").string().c_str(), "a")) {
			std::fprintf(log, "%s\n", line.c_str());
			std::fclose(log);
		}
	}
	std::error_code error;
	const bool      triggered = std::filesystem::remove(dir / "trigger", error);
	if (static_cast<int64_t>(frame) != target && !triggered) return;
	const auto publish_start = Clock::now();
	publish_gpu_data();
	std::scoped_lock lock(r.mutex);
	r.path = (dir / fmt::format("frame_{:06d}.kygpucap", frame)).string();
	r.file = std::fopen(r.path.c_str(), "wb");
	if (r.file == nullptr) {
		LOGF("GPUCAPTURE cannot create %s\n", r.path.c_str());
		return;
	}
	r.out.clear();
	SerializeHeader(r.out);
	std::fwrite(r.out.data(), 1, r.out.size(), r.file);
	r.written = r.out.size();
	r.pages.clear();
	r.shaders.clear();
	r.outputs.clear();
	r.stored = r.zero = r.unreadable = r.skipped = r.submits = r.missing_shaders = r.updates = 0;
	r.current = 0;
	r.touch_time   = {};
	r.publish_time = Clock::now() - publish_start;
	r.start        = Clock::now();
	g_recording.store(true, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::GpuCapture
