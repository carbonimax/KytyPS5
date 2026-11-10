#include "common/assert.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "gpu_blit_shaders/gpu_pssr_motion_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/guest_gpu/pssrProf.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cmdRecord.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

struct MotionParameters {
	uint32_t base[2];
	uint32_t size[2];
	float    scale[2];
	float    offset[2];
};
static_assert(sizeof(MotionParameters) == sizeof(PssrParameters)); // one pipeline layout

constexpr uint32_t PSSR_QUERY_PAIRS = 64;

// KYTY_PSSR_OFFSET_SIGN=<x><y>, each + or - (default ++): the sign of motion_offset per axis.
std::array<float, 2> OffsetSign() {
	static const auto sign = [] {
		const char* value = std::getenv("KYTY_PSSR_OFFSET_SIGN");
		std::array<float, 2> out {1.0f, 1.0f};
		for (size_t i = 0; value != nullptr && i < 2 && value[i] != 0; ++i) {
			out[i] = value[i] == '-' ? -1.0f : 1.0f;
		}
		return out;
	}();
	return sign;
}

// DLSS and FSR read and write with stages and accesses of their own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

} // namespace

RenderExecutor::~RenderExecutor() {
	for (const auto& context: m_dlss_contexts) {
		if (context.feature != nullptr) {
			Dlss::ReleaseFeature(context.feature);
		}
	}
	for (const auto& context: m_fsr_contexts) {
		if (context.fsr != nullptr) {
			Fsr::ReleaseContext(context.fsr);
		}
	}
	auto&      graphics = m_context.GetGraphics();
	const auto device   = graphics.device;
	if (m_pssr_motion != nullptr) {
		device.destroyImageView(m_pssr_motion_view, nullptr);
		graphics.DeleteImage(*m_pssr_motion);
		delete m_pssr_motion;
	}
	device.destroyQueryPool(m_pssr_queries, nullptr);
	device.destroyPipeline(m_pssr_motion_pipeline, nullptr);
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

void RenderExecutor::CreatePssrPipelines() {
	if (m_pssr_pipeline != nullptr) {
		return;
	}
	auto&      graphics = m_context.GetGraphics();
	const std::array bindings {
	    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
	                                    vk::ShaderStageFlagBits::eCompute},
	    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
	                                    vk::ShaderStageFlagBits::eCompute}};
	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = bindings.size();
	descriptor_info.pBindings = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
	                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
	                     "create upscaling descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
	                                  sizeof(PssrParameters)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &m_pssr_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(
	                         &layout_info, nullptr, &m_pssr_pipeline_layout),
	                     "create upscaling pipeline layout");
	const auto create = [&](std::span<const uint32_t> code, vk::Pipeline& pipeline) {
		const auto shader = CompileSPV(code, graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.stage.stage = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = shader;
		create.stage.pName = "main";
		create.layout = m_pssr_pipeline_layout;
		const auto result = graphics.device.createComputePipelines(
		    nullptr, 1, &create, nullptr, &pipeline);
		graphics.device.destroyShaderModule(shader, nullptr);
		RequireVulkanSuccess(result, "create upscaling pipeline");
	};
	create(GPU_PSSR_LINEAR_SPV, m_pssr_pipeline);
	create(GPU_PSSR_MOTION_SPV, m_pssr_motion_pipeline);
}

bool RenderExecutor::FixMotion(vk::CommandBuffer command, const Pssr::Request& request,
                               Dlss::Resource& motion) {
	auto& graphics = m_context.GetGraphics();
	if (m_pssr_motion == nullptr || m_pssr_motion->extent.width != motion.width ||
	    m_pssr_motion->extent.height != motion.height) {
		if (m_pssr_motion != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [&graphics, image = m_pssr_motion, view = m_pssr_motion_view] {
				    graphics.device.destroyImageView(view, nullptr);
				    graphics.DeleteImage(*image);
				    delete image;
			    });
			m_pssr_motion      = nullptr;
			m_pssr_motion_view = nullptr;
		}
		vk::ImageCreateInfo info {};
		info.imageType   = vk::ImageType::e2D;
		info.format      = vk::Format::eR16G16Sfloat;
		info.extent      = vk::Extent3D {motion.width, motion.height, 1};
		info.mipLevels   = 1;
		info.arrayLayers = 1;
		info.samples     = vk::SampleCountFlagBits::e1;
		info.tiling      = vk::ImageTiling::eOptimal;
		info.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
		info.initialLayout = vk::ImageLayout::eUndefined;
		auto* image        = new VulkanImage;
		if (!graphics.CreateImage(info, *image)) {
			delete image;
			return false;
		}
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = image->image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = info.format;
		view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr,
		                                                     &m_pssr_motion_view),
		                     "create motion vector view");
		m_pssr_motion = image;
		const vk::ImageMemoryBarrier2 barrier {
		    .srcStageMask     = vk::PipelineStageFlagBits2::eNone,
		    .dstStageMask     = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask    = vk::AccessFlagBits2::eShaderWrite,
		    .oldLayout        = vk::ImageLayout::eUndefined,
		    .newLayout        = vk::ImageLayout::eGeneral,
		    .image            = image->image,
		    .subresourceRange = view_info.subresourceRange};
		command.pipelineBarrier2(
		    vk::DependencyInfo {.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
	}
	const auto             sign = OffsetSign();
	const MotionParameters parameters {
	    {request.x, request.y},
	    {request.width, request.height},
	    {request.motion_scale[0], request.motion_scale[1]},
	    {sign[0] * request.motion_offset[0], sign[1] * request.motion_offset[1]}};
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, static_cast<vk::ImageView>(motion.view),
	                             vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, m_pssr_motion_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = i == 0 ? vk::DescriptorType::eSampledImage
		                                   : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout, 0,
	                             writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_motion_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(parameters), &parameters);
	command.dispatch((request.width + 7) / 8, (request.height + 7) / 8, 1);
	MemoryBarrier(command);
	motion = Dlss::Resource {
	    .view   = static_cast<VkImageView>(m_pssr_motion_view),
	    .image  = static_cast<VkImage>(m_pssr_motion->image),
	    .range  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	    .format = VK_FORMAT_R16G16_SFLOAT,
	    .width  = motion.width,
	    .height = motion.height};
	return true;
}

bool RenderExecutor::Upscale(vk::CommandBuffer command, const Pssr::Request& request,
                             Dlss::Inputs& inputs) {
	if (Dlss::Capabilities() != nullptr && Dlss::Accepts(request)) {
		return UpscaleDlss(command, request, inputs);
	}
	return Fsr::Enabled() && Fsr::Accepts(request) && UpscaleFsr(command, request, inputs);
}

bool RenderExecutor::UpscaleFsr(vk::CommandBuffer command, const Pssr::Request& request,
                                Dlss::Inputs& inputs) {
	if (!Fsr::Fits(inputs)) {
		return false;
	}
	const auto release = [&](Fsr::Context* fsr) {
		if (fsr != nullptr) {
			m_context.GetCommandScheduler().DeferOperation([fsr] { Fsr::ReleaseContext(fsr); });
		}
	};
	auto context = std::ranges::find(m_fsr_contexts, request.context, &FsrContext::context);
	bool reset   = (request.flags & Pssr::kReset) != 0;
	if (context == m_fsr_contexts.end() ||
	    (context->fsr != nullptr && !Fsr::Takes(context->fsr, request, inputs))) {
		if (context != m_fsr_contexts.end()) {
			release(context->fsr);
		} else {
			// A library has at most four contexts.
			if (m_fsr_contexts.size() == 4) {
				release(m_fsr_contexts.front().fsr);
				m_fsr_contexts.erase(m_fsr_contexts.begin());
			}
			context = m_fsr_contexts.insert(m_fsr_contexts.end(), {request.context, nullptr});
		}
		context->fsr = Fsr::CreateContext(request, inputs);
		reset        = true;
	}
	if (context->fsr == nullptr) {
		return false;
	}
	MemoryBarrier(command);
	CmdRecord::Drain(); // FSR records with the driver's entry points, not the recorder's
	if (!Fsr::Evaluate(command, context->fsr, request, inputs, reset)) {
		release(context->fsr);
		context->fsr = nullptr;
		return false;
	}
	MemoryBarrier(command);
	return true;
}

bool RenderExecutor::UpscaleDlss(vk::CommandBuffer command, const Pssr::Request& request,
                                 Dlss::Inputs& inputs) {
	const auto release = [&](Dlss::Feature* feature) {
		if (feature != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [feature] { Dlss::ReleaseFeature(feature); });
		}
	};
	auto context = std::ranges::find(m_dlss_contexts, request.context, &DlssContext::context);
	bool reset   = (request.flags & Pssr::kReset) != 0;
	if (context == m_dlss_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size no quality mode takes scales linearly.
		const auto creation = Dlss::Choose(*Dlss::Capabilities(), request, inputs);
		if (!creation) {
			return false;
		}
		if (context != m_dlss_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_dlss_contexts.size() == 4) {
				release(m_dlss_contexts.front().feature);
				m_dlss_contexts.erase(m_dlss_contexts.begin());
			}
			context = m_dlss_contexts.insert(m_dlss_contexts.end(), {request.context, {}, nullptr});
		}
		context->creation = *creation;
		CmdRecord::Drain(); // NGX records with the driver's entry points, not the recorder's
		context->feature  = Dlss::CreateFeature(command, *creation);
		reset             = true;
	}
	if (context->feature == nullptr) {
		return false;
	}
	MemoryBarrier(command);
	CmdRecord::Drain();
	if (!Dlss::Evaluate(command, context->feature, request, inputs, reset)) {
		release(context->feature);
		context->feature = nullptr;
		return false;
	}
	MemoryBarrier(command);
	return true;
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS or FSR, which write the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	if (((Dlss::Capabilities() != nullptr && Dlss::Accepts(request)) ||
	     (Fsr::Enabled() && Fsr::Accepts(request))) &&
	    parameters.output_channels[0] == 0 && parameters.output_channels[1] == 1 &&
	    parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	buffer.EndRendering();
	const auto command = buffer.Handle();
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	auto& graphics = m_context.GetGraphics();
	CreatePssrPipelines();
	// KYTY_PSSR_PROF: a timestamp pair around the pass; a pair is read back when its slot comes
	// round again, so each sample is the pass recorded PSSR_QUERY_PAIRS requests earlier.
	const bool timed = Pssr::Prof::Enabled();
	uint64_t   gpu_ns = 0;
	bool       gpu_ready = false;
	const uint32_t query = 2 * (m_pssr_query++ % PSSR_QUERY_PAIRS);
	if (timed) {
		if (m_pssr_queries == nullptr) {
			const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eTimestamp,
			                                    .queryCount = 2 * PSSR_QUERY_PAIRS};
			RequireVulkanSuccess(graphics.device.createQueryPool(&info, nullptr, &m_pssr_queries),
			                     "create upscaling query pool");
			command.resetQueryPool(m_pssr_queries, 0, 2 * PSSR_QUERY_PAIRS);
		} else if (m_pssr_query > PSSR_QUERY_PAIRS) {
			std::array<uint64_t, 2> stamps {};
			gpu_ready = graphics.device.getQueryPoolResults(m_pssr_queries, query, 2,
			                                                sizeof(stamps), stamps.data(),
			                                                sizeof(uint64_t),
			                                                vk::QueryResultFlagBits::e64) ==
			            vk::Result::eSuccess;
			gpu_ns = static_cast<uint64_t>(
			    static_cast<double>(stamps[1] - stamps[0]) *
			    graphics.physical_device_properties.limits.timestampPeriod);
			command.resetQueryPool(m_pssr_queries, query, 2);
		}
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_pssr_queries, query);
	}
	const auto finish = [&](Pssr::Prof::Path path) {
		if (timed) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_pssr_queries,
			                        query + 1);
			Pssr::Prof::Upscaled(path, gpu_ns, gpu_ready);
		}
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (dlss_bindings[0]) {
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  extent = binding.desc.info.MipExtent(view.base_level);
			// DLSS takes a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
				               view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		// Neither upscaler takes a motion offset: fold it into pixel vectors first.
		Pssr::Request upscaled = request;
		bool          ready    = true;
		if (request.motion_offset[0] != 0.0f || request.motion_offset[1] != 0.0f) {
			ready                   = FixMotion(command, request, inputs.motion);
			upscaled.motion_scale[0] = upscaled.motion_scale[1] = 1.0f;
			upscaled.motion_offset[0] = upscaled.motion_offset[1] = 0.0f;
		}
		if (ready && Upscale(command, upscaled, inputs)) {
			finish(Dlss::Capabilities() != nullptr ? Pssr::Prof::Path::Dlss
			                                       : Pssr::Prof::Path::Fsr);
			ResetBindings();
			return;
		}
	}
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, input.image_view, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, output.image_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
		                                 : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout,
	                             0, writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                      0, sizeof(parameters), &parameters);
	command.dispatch((parameters.output_size[0] + 7) / 8,
	                 (parameters.output_size[1] + 7) / 8, 1);
	ShaderAccessBarrier(command, vk::PipelineStageFlagBits::eComputeShader);
	finish(Pssr::Prof::Path::Linear);
	ResetBindings();
}

} // namespace Libs::Graphics
