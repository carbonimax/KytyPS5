#include "graphics/guest_gpu/gpuCapture.h"

#include <bit>
#include <cstring>
#include <type_traits>

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

std::vector<uint8_t> Serialize(const Capture& capture) {
	Writer w;
	w.PutBytes({reinterpret_cast<const uint8_t*>(Magic), sizeof(Magic)});
	w.Put(Version);
	w.Put<uint32_t>(0);
	for (const auto& record: capture.records) {
		Writer payload;
		std::visit([&](const auto& r) { PutPayload(payload, r); }, record);
		w.Put(static_cast<uint32_t>(record.index() + 1));
		w.Put<uint32_t>(0);
		w.Put<uint64_t>(payload.out.size());
		w.PutBytes(payload.out);
		w.out.resize((w.out.size() + 7u) & ~size_t {7});
	}
	return std::move(w.out);
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

} // namespace Libs::Graphics::GpuCapture
