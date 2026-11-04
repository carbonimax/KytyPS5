#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cpPipeline.h"
#include "graphics/host_gpu/renderer/cpSplitProf.h"
#include "graphics/host_gpu/renderer/dispatchChainPolicy.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	// Invocations whose element index reaches the bound do not store.
	const bool covered = fill.bounded_elements != 0
	                         ? std::min<uint64_t>(invocations, fill.bounded_elements) == elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.program->shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.program->bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

int DispatchChainMode() {
	static const int mode = [] {
		const char* env   = std::getenv("KYTY_DISPATCH_CHAIN");
		int         value = 1;
		if (env != nullptr && env[0] != '\0') {
			value = std::atoi(env);
		}
		if (value < 0 || value > 2) {
			value = 0;
		}
		std::printf("Dispatch chain: mode %d\n", value);
		return value;
	}();
	return mode;
}

namespace DispatchChain {

enum class Counter : uint32_t {
	Direct,       // direct dispatches recorded
	DirectWrites, // of which write memory (the previous behavior's hazard barrier)
	Indirect,     // indirect dispatches (never chained)
	Chained,      // no barrier before the dispatch inside a chain
	Hazard,       // hazard barriers (ALL_COMMANDS -> COMPUTE)
	Order,        // COMPUTE -> COMPUTE barriers
	Flush,        // deferred trailing barriers (COMPUTE -> ALL_COMMANDS)
	Fresh,        // dispatches starting a chain
	Epoch,        // ordered: guest sync point
	Conflict,     // ordered: RAW, WAR or WAW on a host range
	Unknown,      // ordered: address, GDS or image heap accesses
	Cap,          // ordered: tracking cap
	HazardReason, // hazard inside a chain (first writer)
	Alias,        // of the conflicts: guest memory only (different host resources)
	Missed,       // verification: guest overlap left without a barrier
	Count
};

namespace {

std::array<std::atomic<uint64_t>, static_cast<size_t>(Counter::Count)> g_counters {};

// Writes after reads need only the execution dependency; writes become visible to later reads
// and writes of the next dispatches.
void ComputeOrderBarrier(vk::CommandBuffer vk_buffer) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eUniformRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                          vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags {}, 1,
	                          &barrier, 0, nullptr, 0, nullptr);
}

uint64_t HandleKey(vk::Buffer buffer) noexcept {
	return reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer));
}

// What a direct dispatch reads and writes, from its final bindings: host buffer ranges and images,
// and the guest address ranges behind them (an image and a buffer can cover the same guest memory).
// Accesses that the bindings do not describe make the dispatch unknown. With guest != nullptr the
// guest ranges also go there (verification).
void CollectAccesses(RenderContext& context, const ShaderComputeInputInfo& input,
                     const PreparedBindings& bindings, bool writes, Accesses& host,
                     std::vector<Access>* guest) {
	using ShaderRecompiler::IR::DescriptorBindingKind;
	const auto& program = *input.stage.program;
	const auto& counts  = program.bindings.descriptor_counts;
	host.Clear();
	host.writes  = writes;
	host.unknown = program.has_address_writes || program.info.uses_dma ||
	               program.info.uses_image_heap || program.info.uses_gds ||
	               bindings.gds.buffer != nullptr ||
	               counts[static_cast<size_t>(DescriptorBindingKind::Gds)] != 0 ||
	               counts[static_cast<size_t>(DescriptorBindingKind::BdaPagetable)] != 0 ||
	               counts[static_cast<size_t>(DescriptorBindingKind::FaultBuffer)] != 0;
	if (guest != nullptr) {
		guest->clear();
	}
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX || index >= bindings.buffers.size() ||
		    index >= bindings.buffer_sources.size()) {
			continue;
		}
		const auto& source = bindings.buffer_sources[index];
		if (source.size == 0) {
			continue; // null binding: no access reaches memory
		}
		const auto& view = bindings.buffers[index];
		if (!view.buffer) {
			// Null descriptor (KYTY_BOUNDS_MODE device path, binding under one dword): the device
			// reads zero and drops writes, so no access reaches memory. Its handle key would also be
			// GuestMemory.
			continue;
		}
		const auto  end  = view.range == VK_WHOLE_SIZE ? UINT64_MAX : view.offset + view.range;
		host.ranges.push_back({HandleKey(view.buffer), view.offset, end, resource.written});
		const Access memory {GuestMemory, source.address, source.address + source.size, resource.written};
		host.ranges.push_back(memory);
		if (guest != nullptr) {
			guest->push_back(memory);
		}
	}
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		const auto id = bindings.images[i].image_id;
		if (!id) {
			continue;
		}
		const bool write = program.info.images[i].written;
		host.ranges.push_back({ImageKey(id.index), 0, UINT64_MAX, write});
		const auto& info = context.GetTextureCache().GetImage(id).info;
		for (const auto range: {info.data, info.stencil}) {
			if (range.size != 0) {
				const Access memory {GuestMemory, range.address, range.address + range.size, write};
				host.ranges.push_back(memory);
				if (guest != nullptr) {
					guest->push_back(memory);
				}
			}
		}
	}
	if (bindings.shared_memory.buffer != nullptr) {
		// LDS backing: one utility buffer range that every such dispatch rewrites.
		const auto&  view = bindings.shared_memory;
		const Access access {HandleKey(view.buffer), view.offset, view.offset + view.range, true};
		host.ranges.push_back(access);
		if (guest != nullptr) {
			guest->push_back(access);
		}
	}
}

} // namespace

void Add(Counter counter, uint64_t value = 1) noexcept {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void Report(double seconds, uint64_t frames) {
	if (DispatchChainMode() != 2) {
		return;
	}
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> v {};
	for (size_t i = 0; i < v.size(); i++) {
		v[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	const auto n = [&](Counter c) { return v[static_cast<size_t>(c)]; };
	std::printf("PERF dchain %.1fs: frames=%" PRIu64 " direct=%" PRIu64 " writers=%" PRIu64
	            " indirect=%" PRIu64 " before=%" PRIu64 " after=%" PRIu64 " hazard=%" PRIu64
	            " order=%" PRIu64 " flush=%" PRIu64 " chained=%" PRIu64 " fresh=%" PRIu64
	            " epoch=%" PRIu64 " conflict=%" PRIu64 " unknown=%" PRIu64 " cap=%" PRIu64
	            " hazard_in_chain=%" PRIu64 " alias=%" PRIu64 " missed=%" PRIu64 "\n",
	            seconds, frames, n(Counter::Direct), n(Counter::DirectWrites), n(Counter::Indirect),
	            n(Counter::DirectWrites) + n(Counter::Direct),
	            n(Counter::Hazard) + n(Counter::Order) + n(Counter::Flush), n(Counter::Hazard),
	            n(Counter::Order), n(Counter::Flush), n(Counter::Chained), n(Counter::Fresh),
	            n(Counter::Epoch), n(Counter::Conflict), n(Counter::Unknown), n(Counter::Cap),
	            n(Counter::HazardReason), n(Counter::Alias), n(Counter::Missed));
}

} // namespace DispatchChain

void CommandBuffer::FlushDispatchChain() const {
	EXIT_IF(!m_dispatch_chain.pending || IsInvalid());
	m_dispatch_chain.Reset();
	// The trailing barrier of the chain, as each dispatch recorded it before chaining.
	ShaderAccessBarrier(m_buffer, vk::PipelineStageFlagBits::eComputeShader);
	if (DispatchChainMode() == 2) {
		DispatchChain::Add(DispatchChain::Counter::Flush);
	}
}

// Records the barrier a direct dispatch of a chain needs before it and updates the chain.
static void ChainDispatchBarrier(RenderContext& context, const CommandBuffer& buffer,
                                 vk::CommandBuffer vk_buffer, const ShaderComputeInputInfo& input,
                                 const PreparedBindings& bindings, bool writes,
                                 DispatchChain::Accesses& accesses,
                                 std::vector<DispatchChain::Access>& guest, uint64_t epoch) {
	using namespace DispatchChain;
	const bool verify = DispatchChainMode() == 2;
	CollectAccesses(context, input, bindings, writes, accesses, verify ? &guest : nullptr);
	auto&      state    = buffer.DispatchChainState();
	const auto decision = Decide(state, accesses, epoch);
	if (verify) {
		Add(Counter::Direct);
		if (writes) {
			Add(Counter::DirectWrites);
		}
		constexpr std::array reasons {Counter::Chained,  Counter::Fresh,   Counter::Epoch,
		                              Counter::Conflict, Counter::Unknown, Counter::Cap,
		                              Counter::HazardReason};
		Add(reasons[static_cast<size_t>(decision.reason)]);
		if (decision.reason == Reason::Conflict) {
			// Same check on the host ranges alone: a guest-only overlap is an image and a buffer
			// (or two images) over the same guest memory.
			const auto host_only = [](const Access& a) { return a.resource != GuestMemory; };
			bool host_conflict = false;
			for (const auto& a: accesses.ranges) {
				for (const auto& b: state.window) {
					host_conflict |= host_only(a) && host_only(b) && Overlaps(a, b);
				}
			}
			if (!host_conflict) {
				Add(Counter::Alias);
			}
		}
		if (decision.barrier == Barrier::None && state.pending) {
			// Independent check on guest addresses: a dispatch left unordered after another one
			// must not touch guest memory that the other writes, nor write what it reads.
			for (const auto& a: guest) {
				const auto hit = std::find_if(state.guest_window.begin(), state.guest_window.end(),
				                              [&](const Access& b) { return Overlaps(a, b); });
				if (hit != state.guest_window.end()) {
					Add(Counter::Missed);
					static std::atomic<uint32_t> logged {0};
					if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
						std::printf("PERF dchain missed: shader=0x%016" PRIx64 " range=0x%" PRIx64
						            "-0x%" PRIx64 "%s against 0x%" PRIx64 "-0x%" PRIx64 "%s\n",
						            input.stage.program->shader_hash, a.begin, a.end,
						            a.write ? " w" : " r", hit->begin, hit->end,
						            hit->write ? " w" : " r");
					}
					break;
				}
			}
		}
	}
	if (decision.barrier == Barrier::Hazard) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	} else if (decision.barrier == Barrier::ComputeToCompute) {
		ComputeOrderBarrier(vk_buffer);
	}
	if (verify) {
		if (decision.barrier == Barrier::Hazard) {
			Add(Counter::Hazard);
		} else if (decision.barrier == Barrier::ComputeToCompute) {
			Add(Counter::Order);
		}
		if (!state.pending || decision.barrier != Barrier::None) {
			state.guest_window.clear();
		}
		state.guest_window.insert(state.guest_window.end(), guest.begin(), guest.end());
	}
	Commit(state, accesses, epoch, decision.barrier);
}

// Direct dispatches with KYTY_CP_PIPELINE_DISPATCH: what the bindings do not describe (writes
// through addresses, device addresses, the image heap, GDS, the BDA page table, the fault buffer)
// cannot enter the front's pending-write set, so such dispatches run on the GPU thread.
static bool DispatchWritesUndescribed(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	using ShaderRecompiler::IR::DescriptorBindingKind;
	const auto& counts = program.bindings.descriptor_counts;
	return program.has_address_writes || program.info.uses_dma || program.info.uses_image_heap ||
	       program.info.uses_gds || counts[static_cast<size_t>(DescriptorBindingKind::Gds)] != 0 ||
	       counts[static_cast<size_t>(DescriptorBindingKind::BdaPagetable)] != 0 ||
	       counts[static_cast<size_t>(DescriptorBindingKind::FaultBuffer)] != 0;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	PerfStats::ScopedTimer perf_timer(PerfStats::Counter::DispatchNs, PerfStats::Counter::DispatchCount);
	PerfStats::NoteWork();
	CpSplitProf::ItemScope split_item(CpSplitProf::ItemKind::Dispatch);
	if (CpPipeline::DispatchEnabled() && CpPipeline::FrontActive() &&
	    PipelineDispatch(submit_id, buffer, thread_group_x, thread_group_y, thread_group_z, mode,
	                     async_compute)) {
		return;
	}
	CpPipeline::Drain();
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	const auto& cs_regs = buffer.GetShaders().GetCs();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    cs_regs.cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		LogZeroDispatch(cs_regs, thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}
	Common::LockGuard  lock(m_context.GetMutex());
	ShaderComputeInputInfo input_info {};
	if (!LookUpDispatch(cs_regs, thread_group_x, thread_group_y, thread_group_z, mode, async_compute,
	                    true, input_info)) {
		return;
	}
	DispatchDirectBody(submit_id, buffer, cs_regs, input_info, thread_group_x, thread_group_y,
	                   thread_group_z, mode, GuestSyncEpoch(), nullptr);
}

void RenderExecutor::LogZeroDispatch(const HW::ComputeShaderInfo& cs_regs, uint32_t thread_group_x,
                                     uint32_t thread_group_y, uint32_t thread_group_z,
                                     uint32_t mode) {
	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
		     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
		     thread_group_x, thread_group_y, thread_group_z, mode, cs_regs.cs_regs.data_addr);
	}
}

// The part of a direct dispatch up to its program lookup; false when nothing is to be recorded.
bool RenderExecutor::LookUpDispatch(const HW::ComputeShaderInfo& cs_regs, uint32_t thread_group_x,
                                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
                                    bool async_compute, bool begin_heap,
                                    ShaderComputeInputInfo& input_info) {
	if (cs_regs.cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return false;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, cs_regs.cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	input_info               = {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	if (begin_heap) {
		m_context.GetImageHeap().BeginDraw();
	}
	bool found_program = false;
	{
		CpSplitProf::LookupScope lookup;
		found_program = static_cast<bool>(m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info));
	}
	if (!found_program) return false;
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}
	return true;
}

// The guest ranges a queued dispatch may write, from its snapshot: written V# as FindBuffers sizes
// them (without the clamps that only shrink), and storage T# as ResolveTexture describes them.
void RenderExecutor::NoteDispatchWrites(CpPipeline::DrawRecord& record) {
	const auto& input     = record.dispatch.input;
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	for (uint32_t i = 0; i < program.info.buffers.size() && i < resources.buffers.size(); i++) {
		const auto& metadata = program.info.buffers[i];
		if (!metadata.written || metadata.descriptor_index == UINT32_MAX) continue;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		auto       size       = descriptor.GetSize();
		if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
		    !metadata.scalar) {
			size = std::max(size, uint64_t {metadata.max_byte_extent});
		}
		if (descriptor.NumRecords() == UINT32_MAX) {
			size = std::min(size, uint64_t {256} * 1024 * 1024);
		}
		if (descriptor.Base48() != 0) {
			CpPipeline::NoteWrite(record, descriptor.Base48(), size);
		}
	}
	for (uint32_t i = 0; i < program.info.images.size() && i < resources.images.size(); i++) {
		GuestRange data;
		GuestRange stencil;
		if (program.info.images[i].written &&
		    StorageImageRanges(program.info.images[i], resources.images[i], data, stencil)) {
			CpPipeline::NoteWrite(record, data.address, data.size);
			CpPipeline::NoteWrite(record, stencil.address, stencil.size);
		}
	}
}

// GPU thread, KYTY_CP_PIPELINE_DISPATCH: looks the dispatch's program up and queues it; false when
// it must run here (pipeline not engaged).
bool RenderExecutor::PipelineDispatch(uint64_t submit_id, CommandBuffer& buffer,
                                      uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	if (!CpPipeline::Engaged()) {
		CpPipeline::Note(CpPipeline::Stat::DispatchFast);
		return false;
	}
	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		LogZeroDispatch(buffer.GetShaders().GetCs(), thread_group_x, thread_group_y, thread_group_z,
		                mode);
		return true;
	}
	static const bool initialized = [this] {
		CpPipeline::Init(this, PipelineExecute, PipelineRelookup, PipelineFeedback);
		CpPipeline::SetHeapHook(PipelineHeap);
		return true;
	}();
	(void)initialized;

	auto& record       = CpPipeline::Acquire();
	auto& dispatch     = record.dispatch;
	record.is_dispatch = true;
	record.submit_id   = submit_id;
	dispatch.cs        = buffer.GetShaders().GetCs();
	dispatch.writes.clear();
	CpPipeline::SetInLookup(true);
	bool found = LookUpDispatch(dispatch.cs, thread_group_x, thread_group_y, thread_group_z, mode,
	                            async_compute, false, dispatch.input);
	bool heap = false;
	if (CpPipeline::TakeWalkHazard()) {
		// The walk read bytes a queued dispatch writes: look up again once they are written (an
		// image heap use begins its draw again through the heap hook).
		CpPipeline::Note(CpPipeline::Stat::HazardWalk);
		CpPipeline::Drain();
		heap  = CpPipeline::TakeHeapInline();
		found = LookUpDispatch(dispatch.cs, thread_group_x, thread_group_y, thread_group_z, mode,
		                       async_compute, false, dispatch.input);
	}
	CpPipeline::SetInLookup(false);
	heap = CpPipeline::TakeHeapInline() || heap;
	if (!found) {
		return true;
	}
	if (heap || DispatchWritesUndescribed(*dispatch.input.stage.program)) {
		CpPipeline::Note(CpPipeline::Stat::DispatchInline);
		CpPipeline::Drain();
		EXIT_IF(buffer.IsInvalid());
		m_context.GetCommandScheduler().PopPendingOperations();
		buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
		                    thread_group_x, thread_group_y, thread_group_z, mode,
		                    dispatch.cs.cs_regs.data_addr);
		Common::LockGuard lock(m_context.GetMutex());
		if (!heap) {
			m_context.GetImageHeap().BeginDraw();
		}
		DispatchDirectBody(submit_id, buffer, dispatch.cs, dispatch.input, thread_group_x,
		                   thread_group_y, thread_group_z, mode, GuestSyncEpoch(), nullptr);
		return true;
	}
	dispatch.groups[0]      = thread_group_x;
	dispatch.groups[1]      = thread_group_y;
	dispatch.groups[2]      = thread_group_z;
	dispatch.mode           = mode;
	dispatch.sync_epoch     = GuestSyncEpoch();
	dispatch.snapshot       = *dispatch.input.stage.resources;
	dispatch.input.stage.resources = &dispatch.snapshot;
	NoteDispatchWrites(record);
	CpPipeline::Note(CpPipeline::Stat::DispatchQueued);
	CpPipeline::Publish();
	return true;
}

// Back thread (or GPU thread): a queued direct dispatch, from the packet's place in the stream.
void RenderExecutor::PipelineExecuteDispatch(CpPipeline::DrawRecord& record) {
	auto& dispatch = record.dispatch;
	auto& buffer   = m_context.GetCommandScheduler().Current();
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect),
	                    record.submit_id, dispatch.groups[0], dispatch.groups[1],
	                    dispatch.groups[2], dispatch.mode, dispatch.cs.cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	m_context.GetImageHeap().BeginDraw();
	DispatchDirectBody(record.submit_id, buffer, dispatch.cs, dispatch.input, dispatch.groups[0],
	                   dispatch.groups[1], dispatch.groups[2], dispatch.mode, dispatch.sync_epoch,
	                   &record);
}

// Everything after the program lookup: consumed fills, bindings, barriers and the dispatch. The
// caller holds the render mutex. `epoch` is the guest sync point the dispatch was decoded under.
void RenderExecutor::DispatchDirectBody(uint64_t submit_id, CommandBuffer& buffer,
                                        const HW::ComputeShaderInfo& cs_regs,
                                        ShaderComputeInputInfo& input_info, uint32_t thread_group_x,
                                        uint32_t thread_group_y, uint32_t thread_group_z,
                                        uint32_t mode, uint64_t epoch,
                                        const CpPipeline::DrawRecord* record) {
	const bool use_thread_dimensions = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
	                           thread_group_z, mode))) {
		ResetBindings();
		return;
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	if (program.info.uses_image_heap) {
		PerfStats::ScopedTimer heap_timer(PerfStats::Counter::BindHeapNs, PerfStats::BindProfEnabled());
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	{
		BufferCache::UploadBatch upload_batch(m_context.GetBufferCache(), true);
		RebindBuffers(bindings);
	}
	if (record != nullptr && CpPipeline::Mode() == 2) {
		// Verify mode: every byte the final bindings let the dispatch write was recorded.
		for (const auto& resource: program.info.buffers) {
			const auto index = resource.descriptor_index;
			if (resource.written && index < bindings.buffers.size() &&
			    index < bindings.buffer_sources.size() && bindings.buffers[index].buffer) {
				CpPipeline::VerifyWrite(*record, bindings.buffer_sources[index].address,
				                        bindings.buffer_sources[index].size, program.shader_hash);
			}
		}
		for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
			if (program.info.images[i].written && bindings.images[i].image_id) {
				const auto& info = bindings.images[i].desc.info;
				CpPipeline::VerifyWrite(*record, info.data.address, info.data.size,
				                        program.shader_hash);
				CpPipeline::VerifyWrite(*record, info.stencil.address, info.stencil.size,
				                        program.shader_hash);
			}
		}
	}

	// From here on, only this dispatch's own bindings, image layout transitions (ordered by their
	// own tracked accesses), its barrier and the dispatch are recorded: a dispatch chain stays open.
	// The image heap commit may record more, so it closes the chain first.
	const int                        chain_mode = DispatchChainMode();
	CommandBuffer::DispatchRecording recording(buffer, chain_mode != 0 && !program.info.uses_image_heap);
	auto              vk_buffer        = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	if (HasShaderBufferWrites(input_info.stage)) {
		CpSplitProf::NoteItemWrites();
	}
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	if (chain_mode != 0) {
		ChainDispatchBarrier(m_context, buffer, vk_buffer, input_info, bindings, has_storage_writes,
		                     m_chain_accesses, m_chain_guest_accesses, epoch);
	} else if (has_storage_writes) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, cs_regs.cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);

	// The removed host fence also ordered read-only dispatches before later writers. A chain
	// defers this barrier to the next command that is not a chained dispatch.
	if (chain_mode == 0) {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute) {
	PerfStats::ScopedTimer perf_timer(PerfStats::Counter::DispatchNs, PerfStats::Counter::DispatchCount);
	PerfStats::NoteWork();
	CpSplitProf::ItemScope split_item(CpSplitProf::ItemKind::Dispatch);
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	m_context.GetImageHeap().BeginDraw();
	bool found_program = false;
	{
		CpSplitProf::LookupScope lookup;
		found_program = static_cast<bool>(m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info));
	}
	if (!found_program) return;
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = *input_info.stage.program;
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		PerfStats::ScopedTimer heap_timer(PerfStats::Counter::BindHeapNs, PerfStats::BindProfEnabled());
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	std::optional<BufferCache::UploadBatch> upload_batch;
	upload_batch.emplace(m_context.GetBufferCache(), true);
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	upload_batch.reset();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	if (HasShaderBufferWrites(input_info.stage)) {
		CpSplitProf::NoteItemWrites();
	}
	const auto vk_buffer = buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (DispatchChainMode() == 2) {
		DispatchChain::Add(DispatchChain::Counter::Indirect);
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
