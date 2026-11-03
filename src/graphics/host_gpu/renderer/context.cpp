#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	if (m_upload_post_pending) {
		FlushUploadPost();
	}
	if (m_dispatch_chain.pending && !m_dispatch_recording) {
		FlushDispatchChain();
	}
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::HandleForUpload() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::DeferUploadCopies(vk::Buffer source, vk::Buffer destination,
                                      std::span<const vk::BufferCopy> regions) const {
	// The overlap test is linear in the pending list: bound it.
	constexpr size_t MaxPending = 256;
	bool             flush      = m_upload_copies.size() + regions.size() > MaxPending;
	for (const auto& region: regions) {
		for (const auto& pending: m_upload_copies) {
			flush |= pending.destination == destination &&
			         region.dstOffset < pending.region.dstOffset + pending.region.size &&
			         pending.region.dstOffset < region.dstOffset + region.size;
		}
	}
	if (flush && m_upload_post_pending) {
		FlushUploadPost();
	}
	for (const auto& region: regions) {
		// Contiguous in both buffers with a pending region of the same pair: extend that region.
		const auto adjacent = std::find_if(
		    m_upload_copies.begin(), m_upload_copies.end(), [&](const PendingUploadCopy& pending) {
			    return pending.source == source && pending.destination == destination &&
			           pending.region.srcOffset + pending.region.size == region.srcOffset &&
			           pending.region.dstOffset + pending.region.size == region.dstOffset;
		    });
		if (adjacent != m_upload_copies.end()) {
			adjacent->region.size += region.size;
		} else {
			m_upload_copies.push_back({source, destination, region});
		}
	}
	m_upload_post_pending = true;
}

void CommandBuffer::FlushUploadPost() const {
	m_upload_post_pending = false;
	if (!m_upload_copies.empty()) {
		// Everything recorded before orders and becomes visible before the batch's copies (global
		// scope: the copies may target several buffers). A dispatch chain may still be pending
		// here (uploads record outside Handle()); this barrier then also orders the chained
		// dispatches' reads and writes before the copies, as the chain's trailing barrier would,
		// and the trailing barrier below makes the copies visible to the next dispatch.
		vk::MemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		vk::DependencyInfo leading {};
		leading.dependencyFlags    = vk::DependencyFlagBits::eByRegion;
		leading.memoryBarrierCount = 1;
		leading.pMemoryBarriers    = &before;
		m_buffer.pipelineBarrier2(leading);
		// One copy per source and destination pair, pairs in first-use order. Destination ranges
		// do not overlap (DeferUploadCopies), so the order of the regions does not matter.
		for (size_t i = 0; i < m_upload_copies.size(); i++) {
			const auto source      = m_upload_copies[i].source;
			const auto destination = m_upload_copies[i].destination;
			if (!source) {
				continue; // already recorded with an earlier entry of its pair
			}
			m_upload_regions.clear();
			for (size_t j = i; j < m_upload_copies.size(); j++) {
				auto& pending = m_upload_copies[j];
				if (pending.source == source && pending.destination == destination) {
					m_upload_regions.push_back(pending.region);
					pending.source = nullptr;
				}
			}
			m_buffer.copyBuffer(source, destination, static_cast<uint32_t>(m_upload_regions.size()),
			                    m_upload_regions.data());
		}
		m_upload_copies.clear();
	}
	// All copies of the batch complete and become visible before any later command.
	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags    = vk::DependencyFlagBits::eByRegion;
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &after;
	m_buffer.pipelineBarrier2(dependency);
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	m_dispatch_chain.Reset();
	m_dispatch_recording = false;
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	// The trailing barrier of a dispatch chain is recorded before the end.
	if (m_dispatch_chain.pending) {
		FlushDispatchChain();
	}
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering = false;
	if (m_draw_write_stages) {
		const auto stages   = m_draw_write_stages;
		m_draw_write_stages = {};
		ShaderWriteBarrier(Handle(), stages);
	}
}

} // namespace Libs::Graphics
