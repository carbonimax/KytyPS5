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
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::GpuMarkers {

namespace {

constexpr uint32_t TableSize   = 1u << 16u; // commands kept (power of two): the front runs far ahead
constexpr uint32_t MaxBindings = 16;
constexpr uint32_t HeadBytes   = 32; // first bytes of each imported read, at bind time

struct Binding {
	uint64_t vaddr = 0, size = 0, buffer = 0, buffer_size = 0;
	VkBuffer handle = VK_NULL_HANDLE; // host buffer bound, at `offset` (CAP copies)
	uint64_t offset = 0;
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
constexpr uint64_t SnapMax   = 65536;
constexpr uint64_t PrintMax  = 16384; // words printed per hung binding
constexpr uint32_t SnapSlots = 256;
struct Snap {
	uint32_t                            seq = 0, count = 0;
	uint32_t                            finished = 0; // last finished command at bind
	std::array<uint64_t, 4>             vaddr {};
	std::array<std::vector<uint8_t>, 4> bytes {};
};
Snap*             g_snaps = nullptr;
thread_local Snap t_snap;

// KYTY_GPU_MARKERS_SNAP=addr[,addr...]: the shaders whose imported reads are copied at bind.
const std::vector<uint64_t>& SnapShaders() {
	static const std::vector<uint64_t> list = [] {
		std::vector<uint64_t> out;
		const char*           value = std::getenv("KYTY_GPU_MARKERS_SNAP");
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*      next    = nullptr;
			const auto address = std::strtoull(at, &next, 16);
			if (next == at) {
				break;
			}
			out.push_back(address);
			at = *next == ',' ? next + 1 : next;
		}
		return out;
	}();
	return list;
}
bool IsSnapShader(uint64_t address) {
	const auto& list = SnapShaders();
	return address != 0 && std::find(list.begin(), list.end(), address) != list.end();
}

struct CpWrite {
	uint32_t seq = 0; // last command recorded before the write
	char     kind = 0;
	uint64_t vaddr = 0, size = 0;
};
constexpr uint32_t CpWrites = 1u << 14u;
CpWrite*              g_cp_writes = nullptr;
std::atomic<uint32_t> g_cp_count {0};

// KYTY_GPU_MARKERS_CAP: GPU copies of the listed shaders' bindings before and after each dispatch.
constexpr uint64_t CapRing       = 128ull << 20u;
constexpr uint64_t CapBindingMax = 128ull << 10u;
constexpr uint32_t CapMetas      = 4096;
constexpr uint32_t CapUser       = 16;
constexpr uint32_t CapSrt        = 64;
struct SrtRead {
	uint64_t vaddr = 0;
	uint32_t value = 0, state = 0;
};
struct SrtReads {
	uint64_t                         shader = 0;
	uint32_t                         count  = 0;
	std::array<SrtRead, CapSrt>      reads {};
};
struct CapMeta {
	std::atomic<uint32_t> seq {0}; // 0 while being filled
	uint32_t              x = 0, y = 0, z = 0, count = 0;
	uint64_t              shader = 0, start = 0, bytes = 0; // ring cursor of the pre copies, total bytes
	std::array<uint32_t, CapUser>    user {};
	SrtReads                         srt {};
	std::array<Binding, MaxBindings> bindings {};
	std::array<uint32_t, MaxBindings> copied {}, pre {}, post {}; // bytes copied, ring offsets
};
VkBuffer          g_cap_buffer = VK_NULL_HANDLE;
VmaAllocation     g_cap_allocation = nullptr;
const uint8_t*    g_cap_mapped = nullptr;
CapMeta*          g_caps       = nullptr;
std::mutex        g_cap_mutex;
uint64_t          g_cap_cursor = 0; // bytes ever allocated in the ring (g_cap_mutex)
uint32_t          g_cap_count  = 0; // captures ever started (g_cap_mutex)
thread_local std::array<uint32_t, CapUser> t_user {};
thread_local CapMeta* t_cap = nullptr; // capture whose after-copies End records
thread_local SrtReads t_srt;           // walker loads noted since the last dispatch

const std::vector<uint64_t>& CapShaders() {
	static const std::vector<uint64_t> list = [] {
		std::vector<uint64_t> out;
		const char*           value = std::getenv("KYTY_GPU_MARKERS_CAP");
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*      next    = nullptr;
			const auto address = std::strtoull(at, &next, 16);
			if (next == at) {
				break;
			}
			out.push_back(address);
			at = *next == ',' ? next + 1 : next;
		}
		return out;
	}();
	return list;
}

void CapBarrier(vk::CommandBuffer cmd) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
	                    &barrier, 0, nullptr, 0, nullptr);
}

void CapAfterCopies(vk::CommandBuffer cmd) {
	// The dispatch may write what the copies read; the host reads the ring once the marker says finished.
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eHostRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                    vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 1, &barrier,
	                    0, nullptr, 0, nullptr);
}

void CapCopies(vk::CommandBuffer cmd, const CapMeta& m, bool post) {
	CapBarrier(cmd);
	for (uint32_t i = 0; i < std::min(m.count, MaxBindings); i++) {
		const auto& b = m.bindings[i];
		if (b.handle == VK_NULL_HANDLE || m.copied[i] == 0) {
			continue;
		}
		vk::BufferCopy region {b.offset, (post ? m.post[i] : m.pre[i]), m.copied[i]};
		cmd.copyBuffer(vk::Buffer(b.handle), vk::Buffer(g_cap_buffer), 1, &region);
	}
	CapAfterCopies(cmd);
}

// Before the dispatch `e` (its bindings are final): reserves ring space and records the before-copies.
void CapBegin(vk::CommandBuffer cmd, const Entry& e) {
	t_cap = nullptr;
	const uint32_t srt_count = std::exchange(t_srt.count, 0u);
	if (g_caps == nullptr || (e.type != 'C' && e.type != 'I')) {
		return;
	}
	const auto& list = CapShaders();
	if (std::find(list.begin(), list.end(), e.shader) == list.end()) {
		return;
	}
	const auto aligned = [](uint64_t n) { return (n + 255u) & ~uint64_t {255}; };
	uint64_t   half    = 0;
	for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
		half += aligned(std::min(e.bindings[i].size, CapBindingMax));
	}
	CapMeta* m = nullptr;
	{
		std::lock_guard lock(g_cap_mutex);
		if (g_cap_cursor % CapRing + 2 * half > CapRing) {
			g_cap_cursor += CapRing - g_cap_cursor % CapRing;
		}
		m = &g_caps[g_cap_count++ % CapMetas];
		m->seq.store(0);
		m->start = g_cap_cursor;
		g_cap_cursor += 2 * half;
	}
	m->bytes  = 2 * half;
	m->shader = e.shader;
	m->x      = e.x;
	m->y      = e.y;
	m->z      = e.z;
	m->count  = std::min(e.count, MaxBindings);
	m->user   = t_user;
	m->srt    = t_srt;
	m->srt.count = srt_count;
	uint64_t at = m->start % CapRing;
	for (uint32_t i = 0; i < m->count; i++) {
		m->bindings[i] = e.bindings[i];
		m->copied[i]   = static_cast<uint32_t>(std::min(e.bindings[i].size, CapBindingMax));
		m->pre[i]      = static_cast<uint32_t>(at);
		m->post[i]     = static_cast<uint32_t>(at + half);
		at += aligned(m->copied[i]);
	}
	CapCopies(cmd, *m, false);
	m->seq.store(e.seq);
	t_cap = m;
}

// Captures still in the ring, oldest first; `done_only`: their dispatch finished on the GPU.
void WriteCaps(const char* path, uint32_t finished, bool done_only, uint32_t newest) {
	if (g_caps == nullptr) {
		return;
	}
	std::FILE* f = std::fopen(path, "wb");
	if (f == nullptr) {
		return;
	}
	uint64_t cursor = 0;
	uint32_t total  = 0;
	{
		std::lock_guard lock(g_cap_mutex);
		cursor = g_cap_cursor;
		total  = g_cap_count;
	}
	const uint32_t recorded = g_seq.load(std::memory_order_relaxed);
	// The newest GPU write binding and CP write over `vaddr` recorded before command `from` (0: none kept).
	const auto last_writers = [](uint32_t from, uint64_t vaddr, uint32_t& gpu_seq, uint64_t& gpu_shader, uint32_t& cp_seq) {
		gpu_seq = cp_seq = 0;
		gpu_shader       = 0;
		for (uint32_t back = 1; back < TableSize && gpu_seq == 0; back++) {
			const auto& o = g_table[(from - back) & (TableSize - 1)];
			if (o.seq != from - back) {
				break;
			}
			for (uint32_t k = 0; k < std::min(o.count, MaxBindings); k++) {
				const auto& w = o.bindings[k];
				if (w.written != 0 && w.vaddr <= vaddr && vaddr < w.vaddr + w.size) {
					gpu_seq    = o.seq;
					gpu_shader = o.shader;
					break;
				}
			}
		}
		const uint32_t total = g_cp_count.load(std::memory_order_relaxed);
		for (uint32_t back = 1; g_cp_writes != nullptr && back <= std::min(total, CpWrites); back++) {
			const auto& w = g_cp_writes[(total - back) & (CpWrites - 1)];
			if (static_cast<int32_t>(w.seq - from) < 0 && w.vaddr <= vaddr && vaddr < w.vaddr + w.size) {
				cp_seq = w.seq;
				break;
			}
		}
	};
	std::fwrite("KCAP2\0\0\0", 1, 8, f);
	std::fwrite(&recorded, 4, 1, f);
	std::fwrite(&finished, 4, 1, f);
	std::vector<uint8_t> data;
	const uint32_t       first = total > std::min(newest, CapMetas) ? total - std::min(newest, CapMetas) : 0;
	for (uint32_t n = first; n < total; n++) {
		const auto&    m   = g_caps[n % CapMetas];
		const uint32_t seq = m.seq.load();
		if (seq == 0 || cursor > m.start + CapRing || (done_only && static_cast<int32_t>(seq - finished) > 0)) {
			continue;
		}
		const uint32_t done = static_cast<int32_t>(seq - finished) <= 0 ? 1 : 0;
		data.assign(g_cap_mapped + m.start % CapRing, g_cap_mapped + m.start % CapRing + m.bytes);
		if (m.seq.load() != seq) {
			continue; // reused while copying
		}
		std::fwrite(&seq, 4, 1, f);
		std::fwrite(&done, 4, 1, f);
		std::fwrite(&m.shader, 8, 1, f);
		std::fwrite(&m.x, 4, 1, f);
		std::fwrite(&m.y, 4, 1, f);
		std::fwrite(&m.z, 4, 1, f);
		std::fwrite(m.user.data(), 4, CapUser, f);
		// KCAP2: the walker loads noted for this dispatch, the value now, and (when it changed) the last writers.
		const uint32_t srt_count = std::min(m.srt.count, CapSrt);
		std::fwrite(&m.srt.shader, 8, 1, f);
		std::fwrite(&srt_count, 4, 1, f);
		for (uint32_t i = 0; i < srt_count; i++) {
			const auto& r   = m.srt.reads[i];
			uint32_t    now = 0, gpu_seq = 0, cp_seq = 0;
			uint64_t    gpu_shader = 0;
			const uint32_t now_ok = LibKernel::Memory::TryReadBacking(r.vaddr, &now, 4) ? 1u : 0u;
			if (now != r.value || now_ok == 0) {
				last_writers(seq, r.vaddr, gpu_seq, gpu_shader, cp_seq);
			}
			const uint32_t state = r.state | (now_ok << 8u);
			std::fwrite(&r.vaddr, 8, 1, f);
			std::fwrite(&r.value, 4, 1, f);
			std::fwrite(&now, 4, 1, f);
			std::fwrite(&state, 4, 1, f);
			std::fwrite(&gpu_seq, 4, 1, f);
			std::fwrite(&gpu_shader, 8, 1, f);
			std::fwrite(&cp_seq, 4, 1, f);
			std::fwrite(&cp_seq, 4, 1, f); // pad to 40 bytes
		}
		std::fwrite(&m.count, 4, 1, f);
		for (uint32_t i = 0; i < m.count; i++) {
			const auto& b = m.bindings[i];
			const uint8_t flags[4] = {b.written, b.kind, 0, 0};
			std::fwrite(&b.vaddr, 8, 1, f);
			std::fwrite(&b.size, 8, 1, f);
			std::fwrite(&b.buffer, 8, 1, f);
			std::fwrite(&m.copied[i], 4, 1, f);
			std::fwrite(flags, 1, 4, f);
		}
		for (uint32_t i = 0; i < m.count; i++) {
			std::fwrite(data.data() + (m.pre[i] - m.start % CapRing), 1, m.copied[i], f);
		}
		for (uint32_t i = 0; i < m.count; i++) {
			std::fwrite(data.data() + (m.post[i] - m.start % CapRing), 1, m.copied[i], f);
		}
	}
	std::fclose(f);
}

const char* MarkersPath() {
	const char* path = std::getenv("KYTY_GPU_MARKERS_FILE");
	return path != nullptr ? path : "gpu_markers.txt";
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

// Bind-time copies of a command's reads (SNAP) against the bytes now: changed word count and first index.
void SnapDiff(std::FILE* f, uint32_t seq) {
	const auto* snap = g_snaps != nullptr ? &g_snaps[seq % SnapSlots] : nullptr;
	if (snap == nullptr || snap->seq != seq) {
		return;
	}
	std::vector<uint8_t> now(SnapMax);
	for (uint32_t i = 0; i < snap->count; i++) {
		const auto& bytes = snap->bytes[i];
		if (!LibKernel::Memory::TryReadBacking(snap->vaddr[i], now.data(), bytes.size())) {
			continue;
		}
		uint32_t changed = 0, first = UINT32_MAX;
		for (uint64_t k = 0; k + 4 <= bytes.size(); k += 4) {
			if (std::memcmp(bytes.data() + k, now.data() + k, 4) != 0) {
				changed++;
				first = std::min<uint32_t>(first, static_cast<uint32_t>(k / 4));
			}
		}
		std::fprintf(f, "      snap 0x%016" PRIx64 " (%zu bytes): %u words changed bind->now, first %d;"
		                " finished@bind=%u (%u behind)\n",
		             snap->vaddr[i], bytes.size(), changed, first == UINT32_MAX ? -1 : static_cast<int>(first),
		             snap->finished, seq - snap->finished);
	}
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

bool Overlaps(const Binding& a, const Binding& b) {
	return a.vaddr < b.vaddr + b.size && b.vaddr < a.vaddr + a.size;
}

// The latest kept commands before `from` that wrote bytes of `in` (up to 3, with their bindings), and the
// command-processor writes over it before `from`; at depth 2, the same for each read of the latest writer.
void EarlierWriters(std::FILE* f, uint32_t from, const Binding& in, int depth) {
	const char* indent = depth > 1 ? "      " : "          ";
	uint32_t    shown  = 0;
	for (uint32_t back = 1; back < TableSize && shown < 3; back++) {
		const auto& o = g_table[(from - back) & (TableSize - 1)];
		if (o.seq != from - back) {
			break;
		}
		for (uint32_t k = 0; k < std::min(o.count, MaxBindings); k++) {
			const auto& w = o.bindings[k];
			if (w.written == 0 || !Overlaps(w, in)) {
				continue;
			}
			std::fprintf(f, "%searlier GPU write of 0x%016" PRIx64 "+0x%" PRIx64 " (depth %d): seq=%u (-%u) 0x%016" PRIx64
			                "+0x%" PRIx64 " kind=%u buffer=0x%" PRIx64 "+0x%" PRIx64 "\n",
			             indent, in.vaddr, in.size, depth, o.seq, back, w.vaddr, w.size, w.kind, w.buffer, w.buffer_size);
			Print(f, o, depth > 1 ? "      prior" : "          prior2");
			if (depth > 1 && shown == 0) {
				for (uint32_t j = 0; j < std::min(o.count, MaxBindings); j++) {
					if (o.bindings[j].written == 0) {
						EarlierWriters(f, o.seq, o.bindings[j], depth - 1);
					}
				}
			}
			shown++;
			break;
		}
	}
	if (shown == 0) {
		std::fprintf(f, "%sno earlier GPU write of 0x%016" PRIx64 "+0x%" PRIx64 " kept (depth %d)\n", indent, in.vaddr,
		             in.size, depth);
	}
	const uint32_t total = g_cp_count.load(std::memory_order_relaxed);
	uint32_t       cps   = 0;
	for (uint32_t back = 1; g_cp_writes != nullptr && back <= std::min(total, CpWrites) && cps < 4; back++) {
		const auto& w = g_cp_writes[(total - back) & (CpWrites - 1)];
		if (static_cast<int32_t>(w.seq - from) >= 0) {
			continue;
		}
		if (w.vaddr < in.vaddr + in.size && in.vaddr < w.vaddr + w.size) {
			std::fprintf(f, "%searlier cp %c over 0x%016" PRIx64 ": 0x%016" PRIx64 "+0x%" PRIx64 " after seq=%u (%d)\n",
			             indent, w.kind, in.vaddr, w.vaddr, w.size, w.seq, static_cast<int>(w.seq) - static_cast<int>(from));
			cps++;
		}
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
	g_snaps  = !SnapShaders().empty() ? new Snap[SnapSlots] : nullptr;
	g_cp_writes = new CpWrite[CpWrites];
	if (!CapShaders().empty()) {
		buffer_info.size  = CapRing;
		buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationInfo cap_info {};
		if (vmaCreateBuffer(graphics.allocator, &buffer_info, &alloc_info, &g_cap_buffer, &g_cap_allocation,
		                    &cap_info) == VK_SUCCESS &&
		    cap_info.pMappedData != nullptr) {
			g_cap_mapped = static_cast<const uint8_t*>(cap_info.pMappedData);
			g_caps       = new CapMeta[CapMetas];
		}
		LOGF("GpuMarkers: KYTY_GPU_MARKERS_CAP %zu shaders, ring %s\n", CapShaders().size(),
		     g_caps != nullptr ? "on" : "NOT created");
	}
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

void SetUserData(const uint32_t* sgprs) noexcept {
	if (g_caps != nullptr) {
		std::copy_n(sgprs, CapUser, t_user.begin());
	}
}

bool CapListed(uint64_t shader) noexcept {
	if (g_caps == nullptr) {
		return false;
	}
	const auto& list = CapShaders();
	return std::find(list.begin(), list.end(), shader) != list.end();
}

void NoteSrtRead(uint64_t shader, uint64_t vaddr, uint32_t value, uint32_t state) noexcept {
	if (t_srt.shader != shader) {
		t_srt.shader = shader;
		t_srt.count  = 0;
	}
	if (t_srt.count < CapSrt) {
		t_srt.reads[t_srt.count] = {vaddr, value, state};
	}
	t_srt.count++;
}

void NoteBinding(uint64_t vaddr, uint64_t size, bool written, int kind, uint64_t buffer,
                 uint64_t buffer_size, VkBuffer handle, uint64_t offset) noexcept {
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
		b.handle      = handle;
		b.offset      = offset;
		b.written = written ? 1 : 0;
		b.kind    = static_cast<uint8_t>(kind);
		b.head_ok = kind == 1 && !written && size >= HeadBytes &&
		                    LibKernel::Memory::TryReadBacking(vaddr, b.head.data(), HeadBytes)
		                ? 1
		                : 0;
	}
	p.count++;
	if (g_snaps != nullptr && kind == 1 && !written && size <= SnapMax && IsSnapShader(t_shader) &&
	    t_snap.count < t_snap.vaddr.size()) {
		auto& bytes = t_snap.bytes[t_snap.count];
		bytes.resize(size);
		if (LibKernel::Memory::TryReadBacking(vaddr, bytes.data(), size)) {
			t_snap.vaddr[t_snap.count++] = vaddr;
		}
	}
}

void NoteCpWrite(uint64_t vaddr, uint64_t size, char kind) noexcept {
	if (!Active() || g_cp_writes == nullptr) {
		return;
	}
	const uint32_t n = g_cp_count.fetch_add(1, std::memory_order_relaxed);
	auto&          w = g_cp_writes[n & (CpWrites - 1)];
	w.seq            = g_seq.load(std::memory_order_relaxed);
	w.kind           = kind;
	w.vaddr          = vaddr;
	w.size           = size;
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
		slot.finished = g_mapped[1];
		t_snap.count = 0;
	}
	CapBegin(cmd, e);
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, vk::Buffer(g_buffer), 0, seq);
	return seq;
}

void End(vk::CommandBuffer cmd, uint32_t seq) noexcept {
	if (seq == 0 || !cmd) {
		return;
	}
	if (t_cap != nullptr && t_cap->seq.load() == seq) {
		CapCopies(cmd, *t_cap, true); // before the finished marker: a finished capture is complete
	}
	t_cap = nullptr;
	cmd.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, vk::Buffer(g_buffer), 4, seq);
}

void Report(double seconds) {
	if (!Active()) {
		return;
	}
	std::printf("PERF gpumark %.1fs: recorded=%u begun=%u finished=%u\n", seconds,
	            g_seq.load(std::memory_order_relaxed), g_mapped[0], g_mapped[1]);
	static const double period = [] {
		const char* value = std::getenv("KYTY_GPU_MARKERS_CAP_PERIOD");
		return value != nullptr ? std::strtod(value, nullptr) : 0.0;
	}();
	static double   elapsed = 0;
	static uint32_t written = 0;
	if (g_caps != nullptr && period > 0 && written < 6 && (elapsed += seconds) >= period) {
		elapsed = 0;
		char path[512];
		std::snprintf(path, sizeof(path), "%s.cap%u", MarkersPath(), written++);
		WriteCaps(path, g_mapped[1], true, 1200);
		std::printf("PERF gpumark cap: wrote %s\n", path);
	}
}

void Dump(const char* site) noexcept {
	if (!Active()) {
		return;
	}
	const uint32_t recorded = g_seq.load(std::memory_order_relaxed);
	const uint32_t begun    = g_mapped[0];
	const uint32_t finished = g_mapped[1];
	std::FILE*     f        = std::fopen(MarkersPath(), "w");
	LOGF("GpuMarkers: device lost at %s: recorded=%u begun=%u finished=%u (hung: seq %u)\n", site, recorded,
	     begun, finished, finished + 1);
	if (g_caps != nullptr) {
		const std::string cap = std::string(MarkersPath()) + ".cap";
		WriteCaps(cap.c_str(), finished, false, CapMetas);
		LOGF("GpuMarkers: captures written to %s\n", cap.c_str());
	}
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
		std::fflush(f); // keep what is written if the process ends during the dump
		if (seq != finished + 1) {
			continue;
		}
		SnapDiff(f, seq);
		// Every kept command that bound bytes of the hung command's reads (latest first, 24 per read).
		for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
			const auto& r = e.bindings[i];
			if (r.written != 0) {
				continue;
			}
			std::fprintf(f, "  users of 0x%016" PRIx64 "+0x%" PRIx64 ":\n", r.vaddr, r.size);
			uint32_t shown = 0, writers_shown = 0;
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
						if (w.written != 0 && writers_shown < 4) {
							// The writer itself: its reads at bind and now, and its SNAP copies.
							writers_shown++;
							Print(f, o, "    writer");
							SnapDiff(f, o.seq);
							// What wrote the writer's reads after it was recorded: GPU writes landing in
							// stream order, or command-processor writes. Neither: the guest CPU.
							for (uint32_t j = 0; j < std::min(o.count, MaxBindings); j++) {
								const auto& in = o.bindings[j];
								if (in.written != 0) {
									continue;
								}
								uint32_t later = 0;
								for (uint32_t s2 = o.seq + 1; s2 != seq + 1 && later < 8; s2++) {
									const auto& l = g_table[s2 & (TableSize - 1)];
									for (uint32_t m = 0; l.seq == s2 && m < std::min(l.count, MaxBindings); m++) {
										const auto& lw = l.bindings[m];
										if (lw.written != 0 && lw.vaddr < in.vaddr + in.size && in.vaddr < lw.vaddr + lw.size) {
											std::fprintf(f, "      later GPU write of 0x%016" PRIx64 ": seq=%u (+%u) shader=0x%016" PRIx64
											                " 0x%016" PRIx64 "+0x%" PRIx64 "\n",
											             in.vaddr, l.seq, l.seq - o.seq, l.shader, lw.vaddr, lw.size);
											later++;
										}
									}
								}
								const uint32_t total = g_cp_count.load(std::memory_order_relaxed);
								for (uint32_t back = 1; g_cp_writes != nullptr && back <= std::min(total, CpWrites); back++) {
									const auto& w2 = g_cp_writes[(total - back) & (CpWrites - 1)];
									if (static_cast<int32_t>(w2.seq - o.seq) < 0) {
										break; // ring is in record order
									}
									if (w2.vaddr < in.vaddr + in.size && in.vaddr < w2.vaddr + w2.size) {
										std::fprintf(f, "      later cp %c over 0x%016" PRIx64 ": 0x%016" PRIx64 "+0x%" PRIx64
										                " after seq=%u (+%d)\n",
										             w2.kind, in.vaddr, w2.vaddr, w2.size, w2.seq,
										             static_cast<int>(w2.seq) - static_cast<int>(o.seq));
									}
								}
								std::fprintf(f, "      read 0x%016" PRIx64 "+0x%" PRIx64 ": %u later GPU writes shown\n", in.vaddr,
								             in.size, later);
								if (writers_shown == 1) {
									EarlierWriters(f, o.seq, in, 2);
								}
							}
						}
					}
				}
			}
			// Command-processor writes over the read, latest first (the ring keeps the last CpWrites).
			const uint32_t total = g_cp_count.load(std::memory_order_relaxed);
			uint32_t       cp_shown = 0;
			for (uint32_t back = 1; g_cp_writes != nullptr && back <= std::min(total, CpWrites) && cp_shown < 16; back++) {
				const auto& w = g_cp_writes[(total - back) & (CpWrites - 1)];
				if (w.vaddr < r.vaddr + r.size && r.vaddr < w.vaddr + w.size) {
					std::fprintf(f, "    cp %c 0x%016" PRIx64 "+0x%" PRIx64 " after seq=%u (%d)\n", w.kind, w.vaddr,
					             w.size, w.seq, static_cast<int>(w.seq) - static_cast<int>(seq));
					cp_shown++;
				}
			}
			std::fflush(f);
		}
		// The hung command's guest bytes now (up to 16 KiB of each binding), and at bind time (SNAP).
		std::vector<uint8_t> now(SnapMax);
		for (uint32_t i = 0; i < std::min(e.count, MaxBindings); i++) {
			const auto& bind = e.bindings[i];
			const auto  size = std::min<uint64_t>(bind.size, PrintMax);
			if (LibKernel::Memory::TryReadBacking(bind.vaddr, now.data(), size)) {
				Words(f, "now", bind.vaddr, now.data(), size);
			}
		}
		const auto* snap = g_snaps != nullptr ? &g_snaps[seq % SnapSlots] : nullptr;
		for (uint32_t i = 0; snap != nullptr && snap->seq == seq && i < snap->count; i++) {
			Words(f, "bind", snap->vaddr[i], snap->bytes[i].data(), snap->bytes[i].size());
		}
	}
	std::fflush(f);
	std::fclose(f);
}

} // namespace Libs::Graphics::GpuMarkers
