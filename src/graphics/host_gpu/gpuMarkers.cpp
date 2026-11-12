#include "graphics/host_gpu/gpuMarkers.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::GpuMarkers {

namespace {

constexpr uint32_t TableSize   = 1u << 16u; // commands kept (power of two): the front runs far ahead
constexpr uint32_t MaxBindings = 16;
constexpr uint32_t HeadBytes   = 32; // first bytes of each imported read, at bind time

struct Binding {
	uint64_t vaddr = 0, size = 0, buffer = 0, buffer_size = 0;
	uint8_t  written = 0, kind = 0, head_ok = 0;
	std::array<uint8_t, HeadBytes> head {};
};

struct Entry {
	uint32_t seq = 0, x = 0, y = 0, z = 0, flags = 0, count = 0;
	char     type = 0;
	uint64_t shader = 0, shader2 = 0, tick = 0;
	std::array<Binding, MaxBindings> bindings {};
};

struct Pending {
	uint32_t                         count = 0;
	std::array<Binding, MaxBindings> bindings {};
};

VkBuffer                g_buffer     = VK_NULL_HANDLE;
VmaAllocation           g_allocation = nullptr;
const volatile uint32_t* g_mapped    = nullptr; // [0] begun, [1] finished
std::atomic<uint32_t>   g_seq {0};
Entry*                  g_table = nullptr;
thread_local uint64_t   t_shader = 0;
thread_local Pending    t_pending;

// KYTY_GPU_MARKERS_SNAP=address (hex, one compute shader): full copies of its imported reads at bind
// time, kept for its last SnapSlots commands and printed next to the bytes at the loss.
constexpr uint64_t SnapMax   = 16384;
constexpr uint32_t SnapSlots = 256;
struct Snap {
	uint32_t                            seq = 0, count = 0;
	std::array<uint64_t, 4>             vaddr {};
	std::array<std::vector<uint8_t>, 4> bytes {};
};
Snap*             g_snaps = nullptr;
thread_local Snap t_snap;

uint64_t SnapShader() {
	static const uint64_t address = [] {
		const char* value = std::getenv("KYTY_GPU_MARKERS_SNAP");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	return address;
}

void Words(std::FILE* f, const char* tag, uint64_t vaddr, const uint8_t* bytes, uint64_t size) {
	std::fprintf(f, "  %s 0x%016" PRIx64 " (%" PRIu64 " bytes):", tag, vaddr, size);
	for (uint64_t k = 0; k < size; k += 4) {
		uint32_t word = 0;
		std::memcpy(&word, bytes + k, std::min<uint64_t>(4, size - k));
		std::fprintf(f, "%s%08x", k % 64 == 0 ? "\n    " : " ", word);
	}
	std::fprintf(f, "\n");
}

void Print(std::FILE* f, const Entry& e, const char* tag) {
	std::fprintf(f, "%s seq=%u type=%c shader=0x%016" PRIx64 " shader2=0x%016" PRIx64
	                " groups=%u,%u,%u flags=0x%x tick=%" PRIu64 " bindings=%u\n",
	             tag, e.seq, e.type != 0 ? e.type : '?', e.shader, e.shader2, e.x, e.y, e.z, e.flags, e.tick,
	             e.count);
	for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
		const auto& b = e.bindings[i];
		static constexpr std::array<const char*, 3> kinds {"vram", "imported", "stream"};
		std::fprintf(f, "  bind 0x%016" PRIx64 "+0x%" PRIx64 " %s %s buffer=0x%" PRIx64 "+0x%" PRIx64, b.vaddr,
		             b.size, b.written != 0 ? "w" : "r", kinds[std::min<uint32_t>(b.kind, 2)], b.buffer,
		             b.buffer_size);
		if (b.head_ok != 0) {
			std::array<uint8_t, HeadBytes> now {};
			const bool now_ok = LibKernel::Memory::TryReadBacking(b.vaddr, now.data(), HeadBytes);
			std::fprintf(f, " head@bind=");
			for (auto c: b.head) {
				std::fprintf(f, "%02x", c);
			}
			std::fprintf(f, " now=%s", now_ok ? (now == b.head ? "same" : "changed:") : "unreadable");
			if (now_ok && now != b.head) {
				for (auto c: now) {
					std::fprintf(f, "%02x", c);
				}
			}
		}
		std::fprintf(f, "\n");
	}
}

} // namespace

bool Requested() noexcept {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_GPU_MARKERS");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return on;
}

void Init(GraphicContext& graphics) {
	if (!Requested() || !graphics.buffer_marker_enabled || graphics.allocator == nullptr) {
		if (Requested()) {
			LOGF("GpuMarkers: KYTY_GPU_MARKERS requested but VK_AMD_buffer_marker is not enabled\n");
		}
		return;
	}
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = 256;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationCreateInfo alloc_info {};
	alloc_info.usage         = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags         = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo allocation_info {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &alloc_info, &g_buffer, &g_allocation,
	                    &allocation_info) != VK_SUCCESS ||
	    allocation_info.pMappedData == nullptr) {
		LOGF("GpuMarkers: could not create the marker buffer\n");
		g_buffer = VK_NULL_HANDLE;
		return;
	}
	std::memset(allocation_info.pMappedData, 0, 256);
	g_table  = new Entry[TableSize];
	g_snaps  = SnapShader() != 0 ? new Snap[SnapSlots] : nullptr;
	g_mapped = static_cast<const volatile uint32_t*>(allocation_info.pMappedData);
	LOGF("GpuMarkers: on (VK_AMD_buffer_marker, %u commands kept)\n", TableSize);
}

bool Active() noexcept {
	return g_mapped != nullptr;
}

void SetShader(uint64_t address) noexcept {
	t_shader        = address;
	t_pending.count = 0; // bindings of a command that wrote no marker (copies, clears)
	t_snap.count    = 0;
}

uint64_t Shader() noexcept {
	return t_shader;
}

void NoteBinding(uint64_t vaddr, uint64_t size, bool written, int kind, uint64_t buffer,
                 uint64_t buffer_size) noexcept {
	if (!Active()) {
		return;
	}
	auto& p = t_pending;
	if (p.count < MaxBindings) {
		auto& b   = p.bindings[p.count];
		b.vaddr       = vaddr;
		b.size        = size;
		b.buffer      = buffer;
		b.buffer_size = buffer_size;
		b.written = written ? 1 : 0;
		b.kind    = static_cast<uint8_t>(kind);
		b.head_ok = kind == 1 && !written && size >= HeadBytes &&
		                    LibKernel::Memory::TryReadBacking(vaddr, b.head.data(), HeadBytes)
		                ? 1
		                : 0;
	}
	p.count++;
	if (g_snaps != nullptr && kind == 1 && !written && size <= SnapMax && t_shader == SnapShader() &&
	    t_snap.count < t_snap.vaddr.size()) {
		auto& bytes = t_snap.bytes[t_snap.count];
		bytes.resize(size);
		if (LibKernel::Memory::TryReadBacking(vaddr, bytes.data(), size)) {
			t_snap.vaddr[t_snap.count++] = vaddr;
		}
	}
}

uint32_t Begin(vk::CommandBuffer cmd, char type, uint64_t shader, uint64_t shader2, uint32_t x, uint32_t y,
               uint32_t z, uint64_t tick, uint32_t flags) noexcept {
	if (!Active() || !cmd) {
		t_pending.count = 0;
		return 0;
	}
	const uint32_t seq = g_seq.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&          e   = g_table[seq & (TableSize - 1)];
	e.seq              = seq;
	e.type             = type;
	e.shader           = shader;
	e.shader2          = shader2;
	e.x                = x;
	e.y                = y;
	e.z                = z;
	e.tick             = tick;
	e.flags            = flags;
	e.count            = t_pending.count;
	std::copy_n(t_pending.bindings.begin(), std::min(t_pending.count, MaxBindings), e.bindings.begin());
	t_pending.count = 0;
	if (t_snap.count != 0) {
		auto& slot = g_snaps[seq % SnapSlots];
		std::swap(slot, t_snap);
		slot.seq     = seq;
		t_snap.count = 0;
	}
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, vk::Buffer(g_buffer), 0, seq);
	return seq;
}

void End(vk::CommandBuffer cmd, uint32_t seq) noexcept {
	if (seq == 0 || !cmd) {
		return;
	}
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, vk::Buffer(g_buffer), 4, seq);
}

void Report(double seconds) {
	if (!Active()) {
		return;
	}
	std::printf("PERF gpumark %.1fs: recorded=%u begun=%u finished=%u\n", seconds,
	            g_seq.load(std::memory_order_relaxed), g_mapped[0], g_mapped[1]);
}

void Dump(const char* site) noexcept {
	if (!Active()) {
		return;
	}
	const uint32_t recorded = g_seq.load(std::memory_order_relaxed);
	const uint32_t begun    = g_mapped[0];
	const uint32_t finished = g_mapped[1];
	const char*    path     = std::getenv("KYTY_GPU_MARKERS_FILE");
	std::FILE*     f        = std::fopen(path != nullptr ? path : "gpu_markers.txt", "w");
	LOGF("GpuMarkers: device lost at %s: recorded=%u begun=%u finished=%u (hung: seq %u)\n", site, recorded,
	     begun, finished, finished + 1);
	if (f == nullptr) {
		return;
	}
	std::fprintf(f, "device lost at %s: recorded=%u begun=%u finished=%u\n", site, recorded, begun, finished);
	// The first command not finished is the hung one; the CP may have begun later ones (no barrier).
	const uint32_t first = finished > 4 ? finished - 4 : 1;
	const uint32_t last  = std::min(std::max(begun, finished + 1) + 4, recorded);
	for (uint32_t seq = first; seq <= last && seq - first < 64; seq++) {
		const auto& e = g_table[seq & (TableSize - 1)];
		if (e.seq != seq) {
			std::fprintf(f, "seq=%u overwritten (table wrapped)\n", seq);
			continue;
		}
		const char* tag = seq <= finished ? "done " : seq == finished + 1 ? "HUNG " : seq <= begun ? "begun" : "queued";
		Print(f, e, tag);
		if (seq != finished + 1) {
			continue;
		}
		// The hung command's guest bytes now (up to 16 KiB of each binding), and at bind time (SNAP).
		std::vector<uint8_t> now(SnapMax);
		for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
			const auto& bind = e.bindings[i];
			const auto  size = std::min<uint64_t>(bind.size, SnapMax);
			if (LibKernel::Memory::TryReadBacking(bind.vaddr, now.data(), size)) {
				Words(f, "now", bind.vaddr, now.data(), size);
			}
		}
		const auto* snap = g_snaps != nullptr ? &g_snaps[seq % SnapSlots] : nullptr;
		for (uint32_t i = 0; snap != nullptr && snap->seq == seq && i < snap->count; i++) {
			Words(f, "bind", snap->vaddr[i], snap->bytes[i].data(), snap->bytes[i].size());
		}
		// Every kept command that bound bytes of the hung command's reads (latest first, 24 per read).
		for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
			const auto& r = e.bindings[i];
			if (r.written != 0) {
				continue;
			}
			std::fprintf(f, "  users of 0x%016" PRIx64 "+0x%" PRIx64 ":\n", r.vaddr, r.size);
			uint32_t shown = 0;
			for (uint32_t back = 1; back < TableSize && shown < 24; back++) {
				const auto& o = g_table[(seq - back) & (TableSize - 1)];
				if (o.seq != seq - back) {
					break;
				}
				for (uint32_t k = 0; k < std::min(o.count, MaxBindings); k++) {
					const auto& w = o.bindings[k];
					if (w.vaddr < r.vaddr + r.size && r.vaddr < w.vaddr + w.size) {
						std::fprintf(f, "    seq=%u (-%u) %c shader=0x%016" PRIx64 " 0x%016" PRIx64 "+0x%" PRIx64
						                " %s kind=%u buffer=0x%" PRIx64 "+0x%" PRIx64 "\n",
						             o.seq, back, o.type, o.shader, w.vaddr, w.size, w.written != 0 ? "w" : "r", w.kind,
						             w.buffer, w.buffer_size);
						shown++;
					}
				}
			}
		}
	}
	std::fflush(f);
	std::fclose(f);
}

} // namespace Libs::Graphics::GpuMarkers
