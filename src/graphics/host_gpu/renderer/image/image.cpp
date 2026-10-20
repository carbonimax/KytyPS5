#include "graphics/host_gpu/renderer/image/image.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// The copies of the D32S8 stencil repro: depth and stencil in one call into mip 1 of a two-mip
// attachment image, and stencil alone into a single-mip one. Returns whether a stencil was lost,
// or nullopt when D32S8 cannot be a transfer attachment image.
[[nodiscard]] std::optional<bool> ProbeStencilCopyLost(GraphicContext& graphics) {
	constexpr auto format = vk::Format::eD32SfloatS8Uint;
	constexpr auto needed = vk::FormatFeatureFlagBits::eDepthStencilAttachment |
	                        vk::FormatFeatureFlagBits::eTransferSrc |
	                        vk::FormatFeatureFlagBits::eTransferDst;
	if ((graphics.physical_device.getFormatProperties(format).optimalTilingFeatures & needed) !=
	    needed) {
		return std::nullopt;
	}
	const auto device      = graphics.device;
	const auto memory_type = [&](uint32_t bits, vk::MemoryPropertyFlags flags) {
		const auto& properties = graphics.physical_device_memory_properties;
		for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
			if ((bits & (1u << i)) != 0 && (properties.memoryTypes[i].propertyFlags & flags) == flags) {
				return i;
			}
		}
		EXIT("stencil copy probe: no memory type\n");
	};
	std::array<vk::DeviceMemory, 4> memories {};
	const auto                      allocate = [&](uint32_t index, vk::MemoryRequirements req,
                                  vk::MemoryPropertyFlags flags) {
		vk::MemoryAllocateInfo info {};
		info.allocationSize  = req.size;
		info.memoryTypeIndex = memory_type(req.memoryTypeBits, flags);
		EXIT_IF(device.allocateMemory(&info, nullptr, &memories[index]) != vk::Result::eSuccess);
		return memories[index];
	};
	// source 2x2, destination 4x4 with two mips, destination 2x2 with one mip.
	std::array<vk::Image, 3> images {};
	for (uint32_t i = 0; i < images.size(); i++) {
		vk::ImageCreateInfo info {};
		info.imageType   = vk::ImageType::e2D;
		info.format      = format;
		info.extent      = vk::Extent3D {i == 1 ? 4u : 2u, i == 1 ? 4u : 2u, 1};
		info.mipLevels   = i == 1 ? 2 : 1;
		info.arrayLayers = 1;
		info.samples     = vk::SampleCountFlagBits::e1;
		info.tiling      = vk::ImageTiling::eOptimal;
		info.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
		             vk::ImageUsageFlagBits::eDepthStencilAttachment |
		             vk::ImageUsageFlagBits::eSampled;
		EXIT_IF(device.createImage(&info, nullptr, &images[i]) != vk::Result::eSuccess);
		vk::MemoryRequirements req {};
		device.getImageMemoryRequirements(images[i], &req);
		EXIT_IF(device.bindImageMemory(
		            images[i], allocate(i, req, vk::MemoryPropertyFlagBits::eDeviceLocal), 0) !=
		        vk::Result::eSuccess);
	}
	vk::Buffer           buffer = nullptr;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = 256;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
	EXIT_IF(device.createBuffer(&buffer_info, nullptr, &buffer) != vk::Result::eSuccess);
	vk::MemoryRequirements buffer_req {};
	device.getBufferMemoryRequirements(buffer, &buffer_req);
	const auto buffer_memory = allocate(3, buffer_req,
	                                    vk::MemoryPropertyFlagBits::eHostVisible |
	                                        vk::MemoryPropertyFlagBits::eHostCoherent);
	EXIT_IF(device.bindBufferMemory(buffer, buffer_memory, 0) != vk::Result::eSuccess);
	void* mapped = nullptr;
	EXIT_IF(device.mapMemory(buffer_memory, 0, 256, {}, &mapped) != vk::Result::eSuccess);
	auto*                             bytes   = static_cast<uint8_t*>(mapped);
	constexpr std::array<float, 4>    depth   = {0.25f, 0.5f, 0.75f, 1.0f};
	constexpr std::array<uint8_t, 4>  stencil = {0x12, 0x34, 0x56, 0x78};
	std::memset(bytes, 0xEE, 256);
	std::memcpy(bytes, depth.data(), sizeof(depth));
	std::memcpy(bytes + 64, stencil.data(), sizeof(stencil));

	vk::CommandPool           pool = nullptr;
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = graphics.queue_family;
	EXIT_IF(device.createCommandPool(&pool_info, nullptr, &pool) != vk::Result::eSuccess);
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command       = nullptr;
	EXIT_IF(device.allocateCommandBuffers(&command_info, &command) != vk::Result::eSuccess);
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	EXIT_IF(command.begin(&begin) != vk::Result::eSuccess);

	constexpr auto aspects = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
	const auto     barrier = [&](uint32_t first, uint32_t count, vk::ImageLayout from,
                             vk::ImageLayout to) {
		std::array<vk::ImageMemoryBarrier2, 3> barriers {};
		for (uint32_t i = 0; i < count; i++) {
			auto& b               = barriers[i];
			b.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
			b.srcAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
			b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
			b.dstAccessMask       = b.srcAccessMask;
			b.oldLayout           = from;
			b.newLayout           = to;
			b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.image               = images[first + i];
			b.subresourceRange    = {aspects, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
		}
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = count;
		dependency.pImageMemoryBarriers    = barriers.data();
		command.pipelineBarrier2(dependency);
	};
	const auto buffer_copy = [](uint64_t offset, vk::ImageAspectFlagBits aspect, uint32_t mip) {
		vk::BufferImageCopy copy {};
		copy.bufferOffset     = offset;
		copy.imageSubresource = {aspect, mip, 0, 1};
		copy.imageExtent      = vk::Extent3D {2, 2, 1};
		return copy;
	};
	barrier(0, 3, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	const std::array uploads = {buffer_copy(0, vk::ImageAspectFlagBits::eDepth, 0),
	                            buffer_copy(64, vk::ImageAspectFlagBits::eStencil, 0)};
	command.copyBufferToImage(buffer, images[0], vk::ImageLayout::eTransferDstOptimal, 2,
	                          uploads.data());
	const vk::ClearDepthStencilValue     clear {0.0f, 0xAA};
	const vk::ImageSubresourceRange all {aspects, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
	command.clearDepthStencilImage(images[1], vk::ImageLayout::eTransferDstOptimal, &clear, 1, &all);
	command.clearDepthStencilImage(images[2], vk::ImageLayout::eTransferDstOptimal, &clear, 1, &all);
	barrier(0, 1, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal);
	barrier(1, 2, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferDstOptimal);
	std::array<vk::ImageCopy, 2> copies {};
	for (uint32_t i = 0; i < 2; i++) {
		const auto aspect        = i == 0 ? vk::ImageAspectFlagBits::eDepth
		                                  : vk::ImageAspectFlagBits::eStencil;
		copies[i].srcSubresource = {aspect, 0, 0, 1};
		copies[i].dstSubresource = {aspect, 1, 0, 1};
		copies[i].extent         = vk::Extent3D {2, 2, 1};
	}
	command.copyImage(images[0], vk::ImageLayout::eTransferSrcOptimal, images[1],
	                  vk::ImageLayout::eTransferDstOptimal, 2, copies.data());
	copies[1].dstSubresource.mipLevel = 0;
	command.copyImage(images[0], vk::ImageLayout::eTransferSrcOptimal, images[2],
	                  vk::ImageLayout::eTransferDstOptimal, 1, &copies[1]);
	barrier(1, 2, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal);
	const auto mip1 = buffer_copy(128, vk::ImageAspectFlagBits::eStencil, 1);
	const auto mip0 = buffer_copy(192, vk::ImageAspectFlagBits::eStencil, 0);
	command.copyImageToBuffer(images[1], vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &mip1);
	command.copyImageToBuffer(images[2], vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &mip0);
	vk::MemoryBarrier2 host {};
	host.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo host_dependency {};
	host_dependency.memoryBarrierCount = 1;
	host_dependency.pMemoryBarriers    = &host;
	command.pipelineBarrier2(host_dependency);
	EXIT_IF(command.end() != vk::Result::eSuccess);

	vk::Fence           fence = nullptr;
	vk::FenceCreateInfo fence_info {};
	EXIT_IF(device.createFence(&fence_info, nullptr, &fence) != vk::Result::eSuccess);
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		EXIT_IF(graphics.queue.submit(1, &submit, fence) != vk::Result::eSuccess);
	}
	EXIT_IF(device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess);
	const bool lost = std::memcmp(bytes + 128, stencil.data(), stencil.size()) != 0 ||
	                  std::memcmp(bytes + 192, stencil.data(), stencil.size()) != 0;
	device.destroyFence(fence, nullptr);
	device.destroyCommandPool(pool, nullptr);
	device.destroyBuffer(buffer, nullptr);
	for (const auto image: images) {
		device.destroyImage(image, nullptr);
	}
	for (const auto memory: memories) {
		device.freeMemory(memory, nullptr);
	}
	return lost;
}

[[nodiscard]] vk::ImageType HostImageType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor3D: return vk::ImageType::e3D;
		case Prospero::ImageType::kColor2D: return vk::ImageType::e2D;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(type));
	}
}

[[nodiscard]] vk::ImageCreateFlags
ImageCreateFlags(const GraphicContext& graphics, const ImageInfo& info, vk::ImageUsageFlags usage) {
	vk::ImageCreateFlags flags {};
	if (DepthAspectTransferFormat(info.pixel_format) == vk::Format::eUndefined) {
		flags |= vk::ImageCreateFlagBits::eMutableFormat;
		if (info.IsBlock() && graphics.supports_block_texel_view) {
			flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
		}
	}
	if (usage & vk::ImageUsageFlagBits::eStorage) {
		flags |= vk::ImageCreateFlagBits::eExtendedUsage;
	}
	if (info.IsVolume()) {
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	return flags;
}

[[nodiscard]] bool HasFormatFeature(vk::FormatProperties      properties,
                                    vk::FormatFeatureFlagBits feature) {
	return static_cast<bool>(properties.optimalTilingFeatures & feature);
}

[[nodiscard]] vk::ImageUsageFlags ImageUsageFlags(GraphicContext& graphics, const ImageInfo& info) {
	auto usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (info.IsBlock()) {
		usage |= vk::ImageUsageFlagBits::eSampled;
		if (graphics.supports_block_texel_view) {
			const auto storage = usage | vk::ImageUsageFlagBits::eStorage;
			if (graphics.GetImageFormatProperties(info.pixel_format, HostImageType(info.type),
			                                      vk::ImageTiling::eOptimal, storage,
			                                      ImageCreateFlags(graphics, info, storage),
			                                      nullptr) == vk::Result::eSuccess) {
				usage = storage;
			} else {
				static std::atomic_flag warned = ATOMIC_FLAG_INIT;
				if (!warned.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Warning: format {} does not support storage access; block-compressed "
					    "textures written by the guest will not render.\n",
					    vk::to_string(info.pixel_format)));
				}
			}
		}
		return usage;
	}
	const auto properties = graphics.GetFormatProperties(info.pixel_format);
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eSampledImage)) {
		usage |= vk::ImageUsageFlagBits::eSampled;
	}
	if (DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined) {
		usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
		if (graphics.attachment_feedback_loop_enabled && (usage & vk::ImageUsageFlagBits::eSampled)) {
			usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
		}
		return usage;
	}
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eColorAttachment)) {
		usage |= vk::ImageUsageFlagBits::eColorAttachment;
	}
	if (graphics.GetPhysicalDeviceProperties().limits.storageImageSampleCounts &
	    static_cast<vk::SampleCountFlagBits>(info.samples)) {
		usage |= vk::ImageUsageFlagBits::eStorage;
	}
	return usage;
}

void ValidateOptionalRange(GuestRange range, const char* name) {
	if (!range.ValidOrEmpty()) {
		EXIT("invalid %s image range: address=0x%016llx size=0x%016llx\n", name,
		     static_cast<unsigned long long>(range.address),
		     static_cast<unsigned long long>(range.size));
	}
}

// KYTY_RO_DEPTH_ELIDE=0 disables. In a read-only depth/stencil layout nothing writes the image:
// depth targets sampled by the draw that also tests against them toggled their access between
// attachment-only and attachment+shader-read every draw, and each toggle recorded a barrier and
// restarted rendering. Without writes since the layout was entered, such accesses need no barrier;
// the state keeps the union of stages and accesses for the next real transition.
bool ElideReadOnlyDepthBarrier(const VulkanImageState& state, vk::ImageLayout destination_layout,
                               vk::AccessFlags2 write_access) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RO_DEPTH_ELIDE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (!enabled || state.layout != destination_layout || (state.access_mask & write_access)) {
		return false;
	}
	switch (destination_layout) {
		case vk::ImageLayout::eDepthReadOnlyOptimal:
		case vk::ImageLayout::eStencilReadOnlyOptimal:
		case vk::ImageLayout::eDepthStencilReadOnlyOptimal: return true;
		default: return false;
	}
}

} // namespace

bool ResolveStencilCopyViaBuffer(GraphicContext& graphics) {
	const char* env = std::getenv("KYTY_STENCIL_COPY_VIA_BUFFER");
	if (env != nullptr && (std::strcmp(env, "0") == 0 || std::strcmp(env, "1") == 0)) {
		std::printf("Stencil copy via buffer: %s (KYTY_STENCIL_COPY_VIA_BUFFER)\n",
		            env[0] == '1' ? "on" : "off");
		return env[0] == '1';
	}
	const auto lost = ProbeStencilCopyLost(graphics);
	std::printf("Stencil copy probe: %s, stencil copies via buffer %s\n",
	            !lost ? "no D32S8 attachment copies" : *lost ? "stencil lost" : "ok",
	            lost.value_or(false) ? "on" : "off");
	return lost.value_or(false);
}

vk::ImageAspectFlags Image::FullAspectMask(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

Image::Barriers Image::GetBarriers(vk::ImageLayout                      destination_layout,
                                   vk::AccessFlags2                     destination_access,
                                   vk::PipelineStageFlags2              destination_stage,
                                   std::optional<ImageSubresourceRange> range) {
	auto& state              = backing.state;
	auto& subresource_states = backing.subresource_states;
	if (range && info.IsVolume()) {
		range->base_layer  = 0;
		range->layer_count = 1;
	}

	const bool partial =
	    range && (range->base_level != 0 || range->level_count != info.resources.levels ||
	              range->base_layer != 0 || range->layer_count != info.resources.layers);
	const bool has_subresource_states = !subresource_states.empty();

	Barriers barriers;
	if (partial || has_subresource_states) {
		if (!has_subresource_states) {
			subresource_states.resize(info.resources.levels * info.resources.layers, state);
		}

		const uint32_t base_level  = partial ? range->base_level : 0;
		const uint32_t level_count = partial ? range->level_count : info.resources.levels;
		const uint32_t base_layer  = partial ? range->base_layer : 0;
		const uint32_t layer_count = partial ? range->layer_count : info.resources.layers;
		for (uint32_t level = base_level; level < base_level + level_count; level++) {
			for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
				const auto index = level * info.resources.layers + layer;
				EXIT_IF(index >= subresource_states.size());
				auto& subresource_state = subresource_states[index];

				constexpr auto write_access = vk::AccessFlagBits2::eTransferWrite |
				                              vk::AccessFlagBits2::eShaderWrite |
				                              vk::AccessFlagBits2::eMemoryWrite;
				const bool     repeated_write =
				    static_cast<bool>(subresource_state.access_mask & write_access);
				if (ElideReadOnlyDepthBarrier(subresource_state, destination_layout, write_access)) {
					subresource_state = {subresource_state.pl_stage | destination_stage,
					                     subresource_state.access_mask | destination_access,
					                     destination_layout};
					continue;
				}
				if (subresource_state.layout != destination_layout ||
				    subresource_state.access_mask != destination_access || repeated_write) {
					vk::ImageMemoryBarrier2 barrier {};
					barrier.srcStageMask                    = subresource_state.pl_stage;
					barrier.srcAccessMask                   = subresource_state.access_mask;
					barrier.dstStageMask                    = destination_stage;
					barrier.dstAccessMask                   = destination_access;
					barrier.oldLayout                       = subresource_state.layout;
					barrier.newLayout                       = destination_layout;
					barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.image                           = backing.image;
					barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
					barrier.subresourceRange.baseMipLevel   = level;
					barrier.subresourceRange.levelCount     = 1;
					barrier.subresourceRange.baseArrayLayer = layer;
					barrier.subresourceRange.layerCount     = 1;
					barriers.push_back(barrier);
					subresource_state = {destination_stage, destination_access, destination_layout};
				}
			}
		}

		if (!partial) {
			subresource_states.clear();
		}
	} else {
		constexpr auto write_access   = vk::AccessFlagBits2::eTransferWrite |
		                                vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eMemoryWrite;
		const bool     repeated_write = static_cast<bool>(state.access_mask & write_access);
		if (state.layout == destination_layout && state.access_mask == destination_access &&
		    !repeated_write) {
			return {};
		}
		if (ElideReadOnlyDepthBarrier(state, destination_layout, write_access)) {
			state = {state.pl_stage | destination_stage, state.access_mask | destination_access,
			         destination_layout};
			return {};
		}

		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask                    = state.pl_stage;
		barrier.srcAccessMask                   = state.access_mask;
		barrier.dstStageMask                    = destination_stage;
		barrier.dstAccessMask                   = destination_access;
		barrier.oldLayout                       = state.layout;
		barrier.newLayout                       = destination_layout;
		barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                           = backing.image;
		barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
		barriers.push_back(barrier);
	}

	// Resident heaps sample their images in the read state; only leaving it invalidates them.
	if (std::ranges::any_of(barriers, [&](const vk::ImageMemoryBarrier2& barrier) {
		    return barrier.oldLayout == ReadLayout() &&
		           barrier.srcAccessMask == vk::AccessFlagBits2::eShaderRead;
	    })) {
		HeapChanged();
	}
	state = {destination_stage, destination_access, destination_layout};
	return barriers;
}

void Image::Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
                    std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer) {
	const auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::PipelineStageFlags2 destination_stage {};
	if (static_cast<bool>(destination_access & transfer_access)) {
		destination_stage |= vk::PipelineStageFlagBits2::eTransfer;
	}
	if (!destination_access ||
	    static_cast<bool>(destination_access & ~vk::AccessFlags2 {transfer_access})) {
		destination_stage |=
		    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	}
	const auto barriers =
	    GetBarriers(destination_layout, destination_access, destination_stage, range);
	if (barriers.empty()) {
		return;
	}
	m_scheduler.EndRendering();
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command_buffer.pipelineBarrier2(dependency);
}

void Image::Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
                   uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyBufferToImage(buffer, backing.image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                     uint64_t offset, uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
}

std::pair<uint32_t, uint32_t> Image::SanitizeCopyLayers(const Image& source,
                                                        const Image& destination, uint32_t depth) {
	const auto source_type        = source.backing.image_type;
	const auto destination_type   = destination.backing.image_type;
	uint32_t   source_layers      = source.backing.layers;
	uint32_t   destination_layers = destination.backing.layers;
	if (source_type == vk::ImageType::e3D) {
		source_layers = 1;
	}
	if (destination_type == vk::ImageType::e3D) {
		destination_layers = 1;
	}
	if (source_type == destination_type) {
		source_layers = destination_layers = std::min(source_layers, destination_layers);
	} else if (source_type == vk::ImageType::e2D) {
		source_layers = depth;
	} else {
		destination_layers = depth;
	}
	return {source_layers, destination_layers};
}

void Image::CopyImage(Image& source) {
	EXIT_IF(source.backing.samples != backing.samples);
	m_scheduler.EndRendering();
	const uint32_t levels     = std::min(source.backing.mip_levels, backing.mip_levels);
	uint32_t       base_depth = source.backing.extent.depth;
	if (source.backing.image_type == backing.image_type) {
		base_depth = std::min(base_depth, backing.extent.depth);
	} else if (backing.image_type == vk::ImageType::e3D) {
		base_depth = backing.extent.depth;
	}
	const auto source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	std::array<vk::ImageCopy, std::tuple_size_v<decltype(ImageInfo::mip_layout)>> copies {};
	for (uint32_t level = 0; level < levels; level++) {
		// Overlapping cache images can differ in size; copying the source extent into a smaller
		// destination writes out of bounds and loses the device.
		const auto width  = std::min(std::max(source.backing.extent.width >> level, 1u),
		                             std::max(backing.extent.width >> level, 1u));
		const auto height = std::min(std::max(source.backing.extent.height >> level, 1u),
		                             std::max(backing.extent.height >> level, 1u));
		const auto depth  = std::max(base_depth >> level, 1u);
		const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);

		auto& copy          = copies[level];
		copy.srcSubresource = {source_aspect, level, 0, 1};
		copy.dstSubresource = {destination_aspect, level, 0, 1};
		if (source.backing.image_type == backing.image_type) {
			if (source.backing.image_type == vk::ImageType::e3D) {
				copy.extent = {width, height, depth};
			} else {
				copy.srcSubresource.layerCount = std::min(source_layers, destination_layers);
				copy.dstSubresource.layerCount = copy.srcSubresource.layerCount;
				copy.extent                    = {width, height, 1};
			}
		} else if (source.backing.image_type == vk::ImageType::e2D) {
			copy.srcSubresource.layerCount = source_layers;
			copy.extent                    = {width, height, source_layers};
		} else {
			copy.dstSubresource.layerCount = destination_layers;
			copy.extent                    = {width, height, destination_layers};
		}
	}
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, levels, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Resolve(Image& source, const ImageSubresourceRange& source_range,
                    const ImageSubresourceRange& destination_range) {
	EXIT_IF(backing.samples != 1 || source.backing.image_type != vk::ImageType::e2D ||
	        backing.image_type != vk::ImageType::e2D || source_range.level_count != 1 ||
	        destination_range.level_count != 1 ||
	        source_range.base_level >= source.backing.mip_levels ||
	        destination_range.base_level >= backing.mip_levels ||
	        source_range.base_layer >= source.backing.layers ||
	        destination_range.base_layer >= backing.layers);
	const auto layers       = std::min({source_range.layer_count, destination_range.layer_count,
	                                    source.backing.layers - source_range.base_layer,
	                                    backing.layers - destination_range.base_layer});
	const auto source_width = std::max(source.backing.extent.width >> source_range.base_level, 1u);
	const auto source_height =
	    std::max(source.backing.extent.height >> source_range.base_level, 1u);
	const auto destination_width =
	    std::max(backing.extent.width >> destination_range.base_level, 1u);
	const auto destination_height =
	    std::max(backing.extent.height >> destination_range.base_level, 1u);
	const bool copy = source.backing.samples == 1;
	EXIT_IF(layers == 0 ||
	        (copy ? !ImageViewOps::FormatsCompatible(source.backing.format, backing.format)
	              : source.backing.format != backing.format));
	auto resolved_source_range             = source_range;
	auto resolved_destination_range        = destination_range;
	resolved_source_range.layer_count      = layers;
	resolved_destination_range.layer_count = layers;
	const vk::Extent3D resolve_extent {std::min(source_width, destination_width),
	                                   std::min(source_height, destination_height), 1};

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               resolved_source_range, command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	        resolved_destination_range, command);
	if (copy) {
		vk::ImageCopy region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, region);
	} else {
		vk::ImageResolve region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.resolveImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                     backing.image, vk::ImageLayout::eTransferDstOptimal, region);
	}
}

uint32_t Image::CopyRows(uint64_t row_size, uint32_t rows, uint64_t capacity) noexcept {
	if (row_size == 0 || rows == 0 || row_size > capacity) {
		return 0;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(rows, capacity / row_size));
}

void Image::CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler) {
	EXIT_IF(buffer.Handle() == nullptr || source.backing.samples != 1 || backing.samples != 1);
	m_scheduler.EndRendering();
	const uint32_t levels = std::min(source.backing.mip_levels, backing.mip_levels);
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto     source_bytes      = DepthAspectTransferBytes(source.backing.format) != 0
	                                       ? DepthAspectTransferBytes(source.backing.format)
	                                       : source.info.bytes_per_block;
	const auto     destination_bytes = DepthAspectTransferBytes(backing.format) != 0
	                                       ? DepthAspectTransferBytes(backing.format)
	                                       : info.bytes_per_block;
	const uint32_t source_block      = source.info.IsBlock() ? 4u : 1u;
	const uint32_t destination_block = info.IsBlock() ? 4u : 1u;
	EXIT_IF(levels == 0 || source_bytes == 0 || source_bytes != destination_bytes ||
	        source_block != destination_block);
	const auto source_transform = source.info.GetColorTransform();
	const auto target_transform = info.GetColorTransform();

	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	auto command                        = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	for (uint32_t level = 0; level < levels; level++) {
		const auto width             = std::max(source.backing.extent.width >> level, 1u);
		const auto height            = std::max(source.backing.extent.height >> level, 1u);
		// The destination may be smaller than the source (e.g. a 60x33x26
		// image copied into a 54x33x26 one). Clamp the copied region like CopyImage
		// does, or copyBufferToImage writes out of bounds.
		const auto copy_width  = std::min(width, std::max(backing.extent.width >> level, 1u));
		const auto copy_height = std::min(height, std::max(backing.extent.height >> level, 1u));
		const auto source_depth      = source.backing.image_type == vk::ImageType::e3D
		                                   ? std::max(source.backing.extent.depth >> level, 1u)
		                                   : source.backing.layers;
		const auto destination_depth = backing.image_type == vk::ImageType::e3D
		                                   ? std::max(backing.extent.depth >> level, 1u)
		                                   : backing.layers;
		const auto slices            = std::min(source_depth, destination_depth);		const auto block_rows = (copy_height + source_block - 1) / source_block;
		const auto row_size =
		    static_cast<uint64_t>((width + source_block - 1) / source_block) * source_bytes;
		const auto rows_per_copy = CopyRows(row_size, block_rows, buffer.Size());
		EXIT_IF(slices == 0 || rows_per_copy == 0);
		for (uint32_t slice = 0; slice < slices; slice++) {
			for (uint32_t block_row = 0; block_row < block_rows; block_row += rows_per_copy) {
				const auto copy_rows = std::min(rows_per_copy, block_rows - block_row);
			const auto y             = block_row * source_block;
			const auto chunk_height  = std::min(copy_rows * source_block, copy_height - y);
			const auto copy_size     = row_size * copy_rows;
				vk::BufferImageCopy source_copy {};
				source_copy.imageSubresource = {
				    source_aspect, level,
				    source.backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				source_copy.imageOffset           = {0, static_cast<int32_t>(y),
			                                     source.backing.image_type == vk::ImageType::e3D
			                                         ? static_cast<int32_t>(slice)
			                                         : 0};
				source_copy.imageExtent           = {width, chunk_height, 1};
				auto destination_copy             = source_copy;
				// The staging buffer holds full source rows, so describe that stride
				// explicitly once the uploaded extent is narrower than the download.
				destination_copy.bufferRowLength   = width;
				destination_copy.bufferImageHeight = chunk_height;
				destination_copy.imageExtent       = {copy_width, chunk_height, 1};
				destination_copy.imageSubresource = {
				    destination_aspect, level,
				    backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				destination_copy.imageOffset.z =
				    backing.image_type == vk::ImageType::e3D ? static_cast<int32_t>(slice) : 0;
				barrier.size          = copy_size;
				barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
				barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				command.pipelineBarrier2(dependency);
				command.copyImageToBuffer(source.backing.image,
				                          vk::ImageLayout::eTransferSrcOptimal, buffer.Handle(),
				                          source_copy);
				if (source_transform != target_transform) {
					const TileManager::Result bytes {buffer.Handle(), 0, copy_size};
					if (source_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, source_transform, false);
					}
					if (target_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, target_transform, true);
					}
				} else {
					barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
					barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
					command.pipelineBarrier2(dependency);
				}
				command.copyBufferToImage(buffer.Handle(), backing.image,
				                          vk::ImageLayout::eTransferDstOptimal, destination_copy);
			}
		}
	}
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyMip(Image& source, uint32_t mip, uint32_t layer, Buffer& scratch) {
	EXIT_IF(source.backing.samples != backing.samples || mip >= backing.mip_levels ||
	        layer >= backing.layers);
	m_scheduler.EndRendering();
	const auto width  = std::max(backing.extent.width >> mip, 1u);
	const auto height = std::max(backing.extent.height >> mip, 1u);
	const auto depth  = std::max(backing.extent.depth >> mip, 1u);
	EXIT_IF(width != source.backing.extent.width || height != source.backing.extent.height);
	const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
	const auto aspects                             = FullAspectMask(source.backing.format);
	EXIT_IF(aspects != FullAspectMask(backing.format));
	std::array<vk::ImageCopy, 2> copies {};
	uint32_t                     copy_count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
	                         vk::ImageAspectFlagBits::eStencil}) {
		if (!static_cast<bool>(aspects & aspect)) {
			continue;
		}
		auto& copy          = copies[copy_count++];
		copy.srcSubresource = {aspect, 0, 0, source_layers};
		copy.dstSubresource = {aspect, mip, layer, destination_layers};
		copy.extent         = {width, height, depth};
	}
	auto command = m_scheduler.Current().Handle();
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	CopyRegions(source, std::span {copies.data(), copy_count}, scratch, command);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyRegions(Image& source, std::span<const vk::ImageCopy> regions, Buffer& scratch,
                        vk::CommandBuffer command) {
	std::array<vk::ImageCopy, 3> direct {};
	uint32_t                     direct_count = 0;
	EXIT_IF(regions.size() > direct.size());
	// On the affected driver only depth and stencil copied in one call into a single-mip image
	// keep the stencil.
	const bool stencil_lost =
	    m_graphics.stencil_copy_via_buffer &&
	    static_cast<bool>(backing.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) &&
	    (backing.mip_levels != 1 || std::ranges::none_of(regions, [](const vk::ImageCopy& r) {
		     return static_cast<bool>(r.dstSubresource.aspectMask & vk::ImageAspectFlagBits::eDepth);
	     }));
	for (const auto& region: regions) {
		if (!stencil_lost || region.dstSubresource.aspectMask != vk::ImageAspectFlagBits::eStencil) {
			direct[direct_count++] = region;
		}
	}
	if (direct_count != 0) {
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, direct_count, direct.data());
	}
	if (direct_count == regions.size()) {
		return;
	}
	for (const auto& region: regions) {
		if (region.dstSubresource.aspectMask == vk::ImageAspectFlagBits::eStencil) {
			CopyStencilWithBuffer(source, region, scratch, command);
		}
	}
}

void Image::CopyStencilWithBuffer(Image& source, const vk::ImageCopy& region, Buffer& scratch,
                                  vk::CommandBuffer command) {
	static std::atomic<uint64_t> copies {0};
	const auto                   count = copies.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((count & (count - 1)) == 0) {
		std::printf("Stencil copy via buffer #%llu (mip %u, %ux%u, %u layers)\n",
		            static_cast<unsigned long long>(count), region.dstSubresource.mipLevel,
		            region.extent.width, region.extent.height, region.dstSubresource.layerCount);
	}
	EXIT_IF(region.srcSubresource.aspectMask != vk::ImageAspectFlagBits::eStencil ||
	        region.extent.depth != 1 ||
	        region.srcSubresource.layerCount != region.dstSubresource.layerCount);
	// The stencil aspect is tightly packed at one byte per texel.
	const auto rows_per_copy = CopyRows(region.extent.width, region.extent.height, scratch.Size());
	EXIT_IF(rows_per_copy == 0);
	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = scratch.Handle();
	barrier.size                = VK_WHOLE_SIZE;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	const auto sync = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
	                      vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
		barrier.srcStageMask  = src_stage;
		barrier.srcAccessMask = src_access;
		barrier.dstStageMask  = dst_stage;
		barrier.dstAccessMask = dst_access;
		command.pipelineBarrier2(dependency);
	};
	for (uint32_t layer = 0; layer < region.dstSubresource.layerCount; layer++) {
		for (uint32_t row = 0; row < region.extent.height; row += rows_per_copy) {
			vk::BufferImageCopy read {};
			read.imageSubresource = {vk::ImageAspectFlagBits::eStencil,
			                         region.srcSubresource.mipLevel,
			                         region.srcSubresource.baseArrayLayer + layer, 1};
			read.imageOffset      = {region.srcOffset.x,
			                         region.srcOffset.y + static_cast<int32_t>(row),
			                         region.srcOffset.z};
			read.imageExtent      = {region.extent.width,
			                         std::min(rows_per_copy, region.extent.height - row), 1};
			auto write            = read;
			write.imageSubresource = {vk::ImageAspectFlagBits::eStencil,
			                          region.dstSubresource.mipLevel,
			                          region.dstSubresource.baseArrayLayer + layer, 1};
			write.imageOffset      = {region.dstOffset.x,
			                          region.dstOffset.y + static_cast<int32_t>(row),
			                          region.dstOffset.z};
			// The scratch buffer is shared with other GPU users (compute LDS, buffered copies).
			sync(vk::PipelineStageFlagBits2::eAllCommands,
			     vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
			     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
			command.copyImageToBuffer(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
			                          scratch.Handle(), read);
			sync(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
			command.copyBufferToImage(scratch.Handle(), backing.image,
			                          vk::ImageLayout::eTransferDstOptimal, write);
		}
	}
	sync(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
	     vk::PipelineStageFlagBits2::eAllCommands,
	     vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

namespace ImageOps {

void Validate(const ImageInfo& info) {
	ValidateOptionalRange(info.data, "data");
	ValidateOptionalRange(info.stencil, "stencil");

	if (info.pixel_format == vk::Format::eUndefined) {
		const bool metadata_empty =
		    info.metadata.range.Empty() && info.metadata.kind == ImageMetadataKind::None &&
		    info.metadata.compression == VideoOutCompression::Uncompressed &&
		    !info.metadata.stencil_compressed;
		if (info.data.Empty() || info.HasStencil() || !metadata_empty || info.extent.width == 0 ||
		    info.extent.height == 0 || info.extent.depth == 0 || info.resources.levels != 1 ||
		    info.resources.layers != 1 || info.samples != 1 || info.pitch != 0 ||
		    info.bytes_per_block != 0) {
			EXIT("invalid stencil association image\n");
		}
		return;
	}

	if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth == 0 ||
	    info.resources.levels == 0 || info.resources.levels > info.mip_layout.size() ||
	    info.resources.layers == 0 || info.samples == 0 ||
	    vulkan_sample_count(info.samples) == vk::SampleCountFlagBits {} ||
	    info.bytes_per_block == 0 || (info.data.address != 0 && info.pitch == 0)) {
		EXIT("invalid image geometry or format\n");
	}

	switch (info.type) {
		case Prospero::ImageType::kColor3D:
			if (info.resources.layers != 1) {
				EXIT("3D images cannot have array layers\n");
			}
			break;
		case Prospero::ImageType::kColor2D:
			if (info.extent.depth != 1) {
				EXIT("invalid 2D image shape\n");
			}
			break;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(info.type));
	}
	if (info.samples > 1 && info.resources.levels != 1) {
		EXIT("multisampled images cannot have mip levels\n");
	}

	if (info.metadata.stencil_compressed && !info.HasStencil()) {
		EXIT("compressed stencil metadata requires a stencil plane\n");
	}
	switch (info.metadata.kind) {
		case ImageMetadataKind::None:
			if (!info.metadata.range.Empty() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed ||
			    info.metadata.stencil_compressed) {
				EXIT("metadata-free image has metadata state\n");
			}
			break;
		case ImageMetadataKind::Htile:
			if (!info.metadata.range.Valid() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("invalid HTILE metadata\n");
			}
			break;
		case ImageMetadataKind::Dcc:
		case ImageMetadataKind::Cmask:
			if (!GuestRange {info.metadata.range.address,
			                 std::max<uint64_t>(info.metadata.range.size, 1)}.Valid() ||
			    info.metadata.compression == VideoOutCompression::Unsupported) {
				EXIT("invalid color metadata\n");
			}
			break;
	}
}

Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element) {
	switch (bytes_per_element) {
		case 1: return Prospero::BufferFormat::k8UNorm;
		case 2: return Prospero::BufferFormat::k16UNorm;
		case 4: return Prospero::BufferFormat::k32Float;
		case 8: return Prospero::BufferFormat::k16_16_16_16Float;
		case 16: return Prospero::BufferFormat::k32_32_32_32Float;
		default: EXIT("unsupported render-target element size: %u\n", bytes_per_element);
	}
}

} // namespace ImageOps

Image::Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& image_info)
    : info(image_info),
      stencil_subresources {0, image_info.resources.levels, 0, image_info.resources.layers},
      m_graphics(graphics), m_scheduler(scheduler) {
	KYTY_PROFILER_FUNCTION();
	ImageOps::Validate(info);
	m_cpu_dirty =
	    !info.data.Empty() && info.metadata.compression == VideoOutCompression::Uncompressed;
	if (info.pixel_format == vk::Format::eUndefined) {
		return;
	}

	vk::ImageCreateInfo create {};
	create.usage         = ImageUsageFlags(graphics, info);
	create.flags         = ImageCreateFlags(graphics, info, create.usage);
	create.imageType     = HostImageType(info.type);
	create.extent        = info.extent;
	create.mipLevels     = info.resources.levels;
	create.arrayLayers   = info.IsVolume() ? 1u : info.resources.layers;
	create.format        = info.pixel_format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.samples       = vulkan_sample_count(info.samples);

	vk::ImageFormatProperties properties {};
	if (graphics.GetImageFormatProperties(create.format, create.imageType, create.tiling,
	                                      create.usage, create.flags,
	                                      &properties) != vk::Result::eSuccess ||
	    !static_cast<bool>(properties.sampleCounts & create.samples)) {
		EXIT("image format does not support required usage: format=%d type=%d usage=0x%x "
		     "flags=0x%x samples=%u\n",
		     static_cast<int>(create.format), static_cast<int>(create.imageType),
		     static_cast<vk::ImageUsageFlags::MaskType>(create.usage),
		     static_cast<vk::ImageCreateFlags::MaskType>(create.flags), info.samples);
	}

	if (!graphics.CreateImage(create, backing)) {
		EXIT("failed to create image: extent=%ux%ux%u format=%d layers=%u levels=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth,
		     static_cast<int>(create.format), create.arrayLayers, create.mipLevels);
	}
	SetVulkanObjectNameF(
	    graphics.device, backing.image,
	    "Kyty.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} samples={}]",
	    info.data.address, info.data.size, info.extent.width, info.extent.height, info.extent.depth,
	    static_cast<uint32_t>(info.pixel_format), info.resources.levels, info.resources.layers,
	    info.samples);
}

uint64_t Image::HashGuestEdges() const {
	std::array<uint8_t, TRACKER_PAGE_SIZE * 2> bytes {};
	const auto                                 range = info.data;
	const uint64_t head_end =
	    std::min(range.End(), Common::AlignUp(range.address, TRACKER_PAGE_SIZE));
	const uint64_t tail_begin =
	    std::max(range.address, Common::AlignDown(range.End(), TRACKER_PAGE_SIZE));
	const uint64_t head_size    = head_end - range.address;
	const uint64_t tail_address = tail_begin < head_end ? head_end : tail_begin;
	const uint64_t tail_size    = range.End() - tail_address;
	if ((head_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(range.address, bytes.data(), head_size)) ||
	    (tail_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(tail_address, bytes.data() + head_size, tail_size))) {
		EXIT("Image: failed to hash guest backing\n");
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(head_size + tail_size));
}

Image::~Image() {
	KYTY_PROFILER_FUNCTION();
	for (const auto& cached: views) {
		m_graphics.device.destroyImageView(cached.view, nullptr);
	}
	if (backing.image != nullptr) {
		m_graphics.DeleteImage(backing);
	}
}

} // namespace Libs::Graphics
