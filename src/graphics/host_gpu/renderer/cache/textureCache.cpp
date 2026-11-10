#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "graphics/guest_gpu/gpuCapture.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 16;
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

// Clear new multisampled color images to zero (KYTY_MSAA_ZERO_INIT=0 disables). They never
// upload guest memory (InitializeImage returns early), so unless a guest clear is materialized
// they start with whatever the host allocation held, e.g. recycled VRAM in transient targets.
// Zero matches what the guest sees after a CMASK fast clear to zero.
bool MsaaZeroInit() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MSAA_ZERO_INIT");
		return value == nullptr || std::strtol(value, nullptr, 10) != 0;
	}();
	return enabled;
}

// KYTY_IMAGE_CHURN_OPT: 0 = off, 1 = on (default), 2 = on and verify against guest bytes.
uint32_t ImageChurnOpt() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_IMAGE_CHURN_OPT");
		return value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}();
	return mode;
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache), m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()),
      m_readback_storage_images(Config::ReadbackStorageImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	RegisterImage(id);
	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	m_image_index_epoch++;
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
	});
	image.registered = true;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	image.HeapChanged();
	UntrackImage(id);
	m_image_index_epoch++;
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	m_lru_cache.Free(image.lru_id);
	m_total_used_memory -= image.AccountedSize();
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.depth_id) {
		std::erase(m_slot_images[image.depth_id].stencil_ids, id);
	} else {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image.stencil_ids, {})) {
			FreeImage(association);
		}
	}
	m_download_images.erase(id);
	if (image.info.metadata.kind == ImageMetadataKind::Htile) {
		// Only the last image to acquire an HTile range releases its clear state.
		const auto metadata = m_surface_metas.find(image.info.metadata.range.address);
		if (metadata != m_surface_metas.end() && metadata->second.owner == id) {
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferRelease([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

void TextureCache::FreeImage(ImageId id) {
	m_slot_images[id].ClearGpuModified();
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	if (RefreshProf::Enabled()) [[unlikely]] {
		RefreshProf::NoteDirty(image, RefreshProf::Edge, 0, 0);
	}
	image.MarkMaybeCpuDirty();
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageFrom(ImageId id, uint64_t address) {
	auto&      image = m_slot_images[id];
	const auto from  = std::max(address, image.track_addr);
	if (!image.IsTracked() || from >= image.track_addr_end) {
		return;
	}
	m_page_manager.UpdatePageWatchers<false>(from, image.track_addr_end - from);
	image.track_addr_end = from;
	if (image.track_addr_end == image.track_addr) {
		image.track_addr     = 0;
		image.track_addr_end = 0;
	}
	image.HeapChanged();
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	// Small tiled targets are written back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	if (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                   : m_readback_linear_images) {
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			const auto& image = m_slot_images[id];
			if (image.picked || !image.Overlaps(address, size)) {
				return;
			}
			image.picked = true;
			result.push_back(id);
		});
	});
	for (const auto id: result) {
		m_slot_images[id].picked = false;
	}
	return result;
}

// FindImage enumerated the owners of every 1 MiB page of the requested range on each lookup,
// which showed up in GPU-thread profiles. The candidate list of a
// range only changes when an image is registered or unregistered (or a registered record's range
// is rewritten), so recent lists are reused while none of that happened; the candidates are then
// examined exactly as before.
const std::vector<ImageId>& TextureCache::FindImageCandidates(uint64_t address, uint64_t size) {
	const auto hash  = (address >> 8u) * 0x9E3779B97F4A7C15ull ^ size * 0xC2B2AE3D27D4EB4Full;
	auto&      entry = m_candidate_cache[(hash >> 58u) & (m_candidate_cache.size() - 1)];
	if (entry.epoch != m_image_index_epoch || entry.address != address || entry.size != size) {
		entry.ids.clear();
		for (const auto id: FindImagesInRegion(address, size)) {
			entry.ids.push_back(id);
		}
		entry.address = address;
		entry.size    = size;
		entry.epoch   = m_image_index_epoch;
	}
	return entry.ids;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	// Converts guest D16 between a four-byte host depth aspect and two-byte color or D16 depth.
	const bool source_wide = DepthAspectTransferBytes(source.backing.format) == sizeof(uint32_t);
	if (source_wide == (DepthAspectTransferBytes(destination.backing.format) == sizeof(uint32_t))) {
		return false;
	}
	auto& depth  = source_wide ? source : destination;
	auto& narrow = source_wide ? destination : source;
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    narrow.info.bytes_per_block != sizeof(uint16_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * sizeof(uint32_t);
	const uint64_t narrow_slice =
	    static_cast<uint64_t>(narrow.info.pitch) * narrow.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || narrow_slice > UINT64_MAX / layers);
	const std::array depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array narrow_copies {
	    vk::BufferImageCopy {0,
	                         narrow.info.pitch,
	                         narrow.info.extent.height,
	                         {narrow.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
	                                                : vk::ImageAspectFlagBits::eColor,
	                          0, 0, layers},
	                         {},
	                         narrow.info.extent}};

	auto depth_buffer  = m_tiler.GetScratchBuffer(depth_slice * layers);
	auto narrow_buffer = m_tiler.GetScratchBuffer(narrow_slice * layers, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(narrow.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * sizeof(uint32_t),
	    .source_slice_stride = narrow_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_wide) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, narrow_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                   narrow_buffer.size);
	} else {
		source.Download(narrow_copies, narrow_buffer.buffer, narrow_buffer.offset,
		                narrow_buffer.size);
		m_tiler.ConvertD16(narrow_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	destination.refresh_note.hash_valid      = false;
	destination.refresh_note.native_is_guest = false;
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	destination.RefreshComplete();
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			if (RefreshProf::Enabled()) [[unlikely]] {
				RefreshProf::NoteDirty(destination, RefreshProf::ImageCopy, destination.info.data.address,
				                       destination.info.data.size);
			}
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		DropGuestSiblings(destination);
		destination.MarkGpuModified(GpuCapture::StampWrites());
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	destination.refresh_note.hash_valid      = false;
	destination.refresh_note.native_is_guest = false;
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer,
	                    m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal));
	if (source.IsGpuModified()) {
		DropGuestSiblings(destination);
		destination.MarkGpuModified(GpuCapture::StampWrites());
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id, bool& keep_both) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	// Depth bytes sampled through a color format no depth view provides (16_UINT, 8_8_UNORM)
	// need a color copy of the whole single-sample depth layout.
	const bool raw_depth_texture =
	    cached.info.IsDepth() && !requested.IsDepth() && !cached.info.HasStencil() &&
	    !requested.HasMetadata() && cached.backing.samples == 1 && requested.samples == 1 &&
	    requested.pitch == cached.info.pitch &&
	    (requested.resources == cached.info.resources || retain_cached_layout) &&
	    !IsSupportedSampledDepthFormat(cached.info.pixel_format, requested.pixel_format);
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_depth_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	if (KeepBeside(cached, requested, binding)) {
		keep_both = true;
		return {};
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_initial_clear  = false;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

namespace BcAlias {

bool Matches(const ImageInfo& requested, const ImageInfo& cached) {
	return requested.guest_format != cached.guest_format &&
	       requested.IsBlock() != cached.IsBlock() &&
	       requested.tile_mode == cached.tile_mode &&
	       requested.bytes_per_block == cached.bytes_per_block &&
	       requested.samples == 1 && cached.samples == 1 &&
	       (requested.data != cached.data || requested.resources != cached.resources ||
	        requested.BlockExtent() != cached.BlockExtent()) &&
	       !TextureCache::SameBacking(cached, requested, false);
}

bool Publish(TextureCache& cache, Image& image) {
	if (!cache.CanDownload(image)) {
		return false;
	}
	// ObtainBuffer preserves padding and records ownership before the GPU-only transfer.
	auto [buffer, offset] =
	    cache.m_buffer_cache.ObtainBuffer(image.info.data.address, image.info.data.size, true);
	cache.DownloadImage(image, *buffer, offset, image.info.data.size, image.info.resources.levels);
	image.ClearGpuModified();
	if (RefreshProf::Enabled()) [[unlikely]] {
		RefreshProf::NoteDirty(image, RefreshProf::BcAlias, image.info.data.address, image.info.data.size);
	}
	image.MarkBufferModified();
	return true;
}

void Resolve(TextureCache& cache, ImageId cached_id, ImageId merged_id) {
	auto& cached = cache.m_slot_images[cached_id];
	auto* merged = merged_id ? &cache.m_slot_images[merged_id] : nullptr;
	// PS5 raw block images can span several packed BC mips. Transfer their complete
	// guest bytes, including tail padding, before retiring the incompatible backing.
	if (cached.SafeToDownload()) {
		if (merged && merged->SafeToDownload() && !Publish(cache, *merged)) {
			return;
		}
		if (!Publish(cache, cached)) {
			return;
		}
		if (merged) {
			if (RefreshProf::Enabled()) [[unlikely]] {
				RefreshProf::NoteDirty(*merged, RefreshProf::BcAlias, merged->info.data.address,
				                       merged->info.data.size);
			}
			merged->MarkBufferModified();
		}
	}
	cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
	cache.FreeImage(cached_id);
}

} // namespace BcAlias

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto& cached = m_slot_images[cached_id];
	if (!cached.registered) {
		return {merged_id};
	}
	// PS5: image heaps keep BC images and raw block views of their layers resident together, so a
	// raw read of a layer subset with the same levels views the BC image instead of evicting it.
	const bool raw_view = binding == BindingType::Texture && !requested.IsBlock() &&
	                      requested.type == cached.info.type &&
	                      requested.data.address == cached.info.data.address &&
	                      requested.BlockExtent() == cached.info.BlockExtent() &&
	                      requested.resources.levels == cached.info.resources.levels &&
	                      requested.resources.layers < cached.info.resources.layers;
	if (!raw_view && BcAlias::Matches(requested, cached.info) && !cached.IsCpuDirty()) {
		BcAlias::Resolve(*this, cached_id, merged_id);
		return {merged_id};
	}
	const bool safe_to_delete = m_gc_tick - cached.tick_accessed_last > NumFramesBeforeRemoval &&
	                            cached.heap_refs.load(std::memory_order_relaxed) == 0u;

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		bool keep_both = false;
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id, keep_both)) {
			return {depth_id};
		}
		if (keep_both) {
			return {merged_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			if (KeepBeside(cached, requested, binding)) {
				return {merged_id};
			}
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			return {ImageViewOps::FormatsCompatible(m_slot_images[result_id].info.pixel_format,
			                                        requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image, bool dcc_as_plain) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return (dcc_as_plain || info.metadata.compression == VideoOutCompression::Uncompressed) &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
		return;
	}

	if (destination.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		image.RefreshComplete();
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		const bool prof       = RefreshProf::Enabled();
		const bool gpu_source = image.IsBufferModified();
		uint64_t   hash       = 0;
		int        same       = -1;
		auto       start      = std::chrono::steady_clock::time_point {};
		if (prof) [[unlikely]] {
			if (!gpu_source && RefreshProf::HashEnabled()) {
				hash = RefreshProf::HashGuest(image.info.data.address, image.info.data.size);
				if (image.refresh_note.hash_valid && hash != 0) {
					same = hash == image.refresh_note.hash ? 1 : 0;
				}
			}
			start = std::chrono::steady_clock::now();
		}
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
		if (prof) [[unlikely]] {
			const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			                    std::chrono::steady_clock::now() - start)
			                    .count();
			RefreshProf::NoteUpload(image, gpu_source, static_cast<uint64_t>(ns), same, false);
			image.refresh_note.hash       = hash;
			image.refresh_note.hash_valid = hash != 0;
		}
		image.refresh_note.native_is_guest = !gpu_source;
		if (RefreshProf::SkipSameMode() == 2 && !gpu_source) [[unlikely]] {
			const auto& info = image.info;
			for (uint32_t m = 0; m < std::min<uint32_t>(info.resources.levels, 16); m++) {
				image.refresh_note.level_hash[m] =
				    RefreshProf::HashGuest(info.data.address + info.mip_layout[m].offset, info.mip_layout[m].size);
			}
		}
	}
	image.RefreshComplete();
}

bool TextureCache::DeferRefresh(Image& image, const ImageViewInfo& view) {
	const auto  mode = RefreshProf::SkipSameMode();
	const auto& info = image.info;
	const auto& note = image.refresh_note;
	if (mode == 0 || view.min_lod < 256 || !note.fault_only || !note.native_is_guest ||
	    !image.IsDefinitelyCpuDirty() || image.IsMaybeCpuDirty() || image.IsBufferModified() ||
	    image.IsGpuModified() || !image.IsTracked() || image.track_addr != info.data.address ||
	    image.depth_id || info.IsDepth() || info.samples != 1 || info.resources.levels > 16 ||
	    info.metadata.compression != VideoOutCompression::Uncompressed) {
		return false;
	}
	std::array<uint64_t, 16> offsets {};
	std::array<uint64_t, 16> sizes {};
	for (uint32_t m = 0; m < info.resources.levels; m++) {
		offsets[m] = info.mip_layout[m].offset;
		sizes[m]   = info.mip_layout[m].size;
	}
	// Every byte past track_addr_end (or past the lowest fault, if a TrackImage re-armed the watch)
	// may have changed unseen.
	const auto dirty_end     = std::min(image.track_addr_end, std::max(note.fault_from, info.data.address));
	const auto first_sampled = view.base_level + view.min_lod / 256u;
	if (!RefreshProf::DirtyBelow(offsets.data(), sizes.data(), info.resources.levels,
	                             dirty_end - info.data.address, info.data.size, first_sampled)) {
		return false;
	}
	// In-session A/B: arm 0 refreshes as mode 0 would; the fault-side tracking of mode 1 stays on.
	if (PerfStats::AbOff("refresh_skip_same")) [[unlikely]] {
		return false;
	}
	const bool first = !note.deferred;
	image.refresh_note.deferred = true;
	if (mode == 2) {
		for (uint32_t m = first_sampled; m < info.resources.levels; m++) {
			if (RefreshProf::HashGuest(info.data.address + offsets[m], sizes[m]) != note.level_hash[m]) {
				RefreshProf::NoteMismatch();
				break;
			}
		}
	}
	// A deferred image is checked again at every lookup until its refresh: count only when reporting.
	if (mode == 2 || RefreshProf::Enabled()) [[unlikely]] {
		RefreshProf::NoteDefer(info.data.size, first);
	}
	return mode == 1;
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto  range = desc.info.metadata.range;
	const auto& view  = desc.view_info;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || view.level_count != 1) {
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// Color metadata slices consist of complete 4 KiB blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	const bool volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto first          = volume_texture ? 0u : metadata_base_layer;
	const auto image_first    = volume_texture ? 0u : view.base_layer;
	const auto count =
	    volume_texture ? std::max(desc.info.extent.depth >> view.base_level, 1u) : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	const auto slice_size = range.size / layers;
	// Each slice holds the viewed mip's metadata in one run; DCC mips past the first tail mip
	// have none. FindImage moves base_level only for one-mip descriptors, which keep the slice.
	TileSizeOffset mip {.size = static_cast<uint32_t>(slice_size)};
	if (desc.info.metadata.kind == ImageMetadataKind::Dcc && desc.info.resources.levels > 1) {
		std::array<TileSizeOffset, 16> mips {};
		TileSizeAlign                  total {};
		(void)TileGetDccSize(desc.info.extent.width, desc.info.extent.height, layers,
		                     desc.info.bytes_per_block, desc.info.resources.levels,
		                     desc.info.tile_mode, total, 0, mips.data());
		mip = mips[view.base_level];
	}
	if (mip.size == 0) {
		return;
	}
	const GuestRange viewed {range.address + slice_size * first + mip.offset,
	                         slice_size * (count - 1) + mip.size};
	const auto run = [&](uint32_t slice) { return viewed.address + slice_size * slice; };
	if (m_buffer_cache.IsRegionGpuModified(viewed.address, viewed.size)) {
		std::array<vk::ClearAttachment, TileManager::ColorClearSlots> clears {};
		uint32_t keys = 0;
		for (uint32_t key = 0; key < clears.size(); key++) {
			clears[key].aspectMask = vk::ImageAspectFlagBits::eColor;
			if (DecodeColorClear(desc, static_cast<uint8_t>(key << 5u),
			                     clears[key].clearValue.color)) {
				keys |= 1u << key;
			}
		}
		// No selected key to clear, or none left since the last check. A target rebound while no
		// checked metadata was forgotten (mipmapped DCC targets bind each mip every frame) reuses
		// its last result instead of looking up every slice and key again.
		const Image::ColorClearCheck check {viewed.address, slice_size, mip.size, count, keys,
		                                    m_buffer_cache.ColorMetadataEpoch()};
		{
			std::scoped_lock lock {m_lock};
			if (m_slot_images[id].color_clear_check == check) {
				return;
			}
		}
		bool checked = true;
		for (uint32_t slice = 0; slice < count && checked; slice++) {
			checked = m_buffer_cache.IsColorMetadataChecked(run(slice), mip.size, keys);
		}
		if (checked) {
			std::scoped_lock lock {m_lock};
			m_slot_images[id].color_clear_check = check;
			return;
		}
		{
			std::scoped_lock lock {m_lock};
			auto&            image = m_slot_images[id];
			// Compressed images cannot upload, so they must already hold native contents.
			const bool native_current = (image.usage.render_target || image.IsGpuModified()) &&
			                            !image.IsBufferModified() && !image.IsCpuDirty();
			// 2D-array attachment views select 2D layers or, of a volume, depth slices.
			if (m_graphics.conditional_rendering_enabled &&
			    count <= m_graphics.GetPhysicalDeviceProperties().limits.maxFramebufferLayers &&
			    (image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment) &&
			    (m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
			     vk::FormatFeatureFlagBits::eColorAttachment) &&
			    (desc.info.metadata.compression == VideoOutCompression::Uncompressed ||
			     native_current)) {
				// The GPU selects each slice's clear, so the CPU never waits for the metadata.
				Common::DebugCounters::Add(Common::DebugCounters::Counter::ColorMetadataChecks);
				InitializeImage(id);
				const bool expand = desc.type != BindingType::VideoOut;
				const auto [buffer, offset] =
				    m_buffer_cache.ObtainBuffer(viewed.address, viewed.size, expand);
				const auto predicates = m_tiler.CheckColorClear(
				    {buffer->Handle(), offset, viewed.size}, slice_size, count, keys, expand);
				for (uint32_t slice = 0; slice < count && expand; slice++) {
					m_buffer_cache.MarkColorMetadataChecked(run(slice), mip.size, keys);
				}
				const auto command = m_scheduler.Current().Handle();
				const auto area    = BeginColorRendering(image, view.format, view.base_level,
				                                         image_first, count, nullptr, command);
				vk::ClearRect rect {.rect = area, .layerCount = 1};
				for (uint32_t slice = 0; slice < count; slice++) {
					rect.baseArrayLayer = slice;
					for (uint32_t key = 0; key < clears.size(); key++) {
						if ((keys & (1u << key)) != 0) {
							const vk::ConditionalRenderingBeginInfoEXT predicate {
							    .buffer = predicates,
							    .offset = (slice * clears.size() + key) * sizeof(uint32_t)};
							command.beginConditionalRenderingEXT(&predicate);
							command.clearAttachments(1, &clears[key], 1, &rect);
							command.endConditionalRenderingEXT();
						}
					}
				}
				command.endRendering();
				CommitGpuWrite(image);
				return;
			}
		}
		// Finish native metadata writes before reading backing bytes. This can submit the
		// scheduler, so discovery runs before final draw uploads and never holds the texture
		// lock across it.
		Common::DebugCounters::ScopedTimer timer(
		    Common::DebugCounters::Counter::ColorMetadataReadbacks,
		    Common::DebugCounters::Counter::ColorMetadataReadbackNs);
		m_buffer_cache.ReadMemory(viewed.address, viewed.size, false);
	}
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = run(slice);
		uint8_t code = 0;
		GpuCapture::Touch(address, sizeof(code));
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		std::vector<uint8_t> bytes(mip.size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			continue;
		}
		{
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			const bool depth = image.info.IsDepth();
			if (depth) {
				const auto value = clear.color.float32[0];
				clear.depthStencil = {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? image.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, bytes.size());
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
}

void TextureCache::RefreshImage(ImageId id, const ImageViewInfo* view) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	// Keeps the shortened watch: writes past it stay unseen until a refresh re-reads them.
	if (view != nullptr && DeferRefresh(image, *view)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		image.ResolveMaybeCpuHash(image.HashGuestEdges());
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size)) {
		const auto& owner = m_slot_images[id];
		if (owner.info.data == stencil && owner.info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetRenderTargetPitch(plane.extent.width, 1) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		if (record.depth_id != depth_id) {
			if (record.depth_id) {
				std::erase(m_slot_images[record.depth_id].stencil_ids, association);
			}
			depth.stencil_ids.push_back(association);
		}
		m_image_index_epoch++;
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
	}

	return association;
}

uint32_t TextureCache::FindImageMemoMode() noexcept {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_FIND_IMAGE_MEMO");
		return value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}();
	return mode;
}

// KYTY_FIND_IMAGE_MEMO. A search is kept only when its result came straight from the candidate
// scan (same backing, or the stencil plane of a depth image), no candidate could block-alias the
// description, and nothing was inserted, freed or merged. Its outcome then depends only on the
// description (identical, by the caller's contract), the candidate list and, per candidate,
// fields that change only together with m_image_index_epoch (registration, range, stencil
// association, which also rewrites the record's info). While the epoch is unchanged the same
// search gives the same image, so a hit skips the description checks and the scan. The per-bind
// rest (view offset, access tick, LRU touch, metadata clears, video-out check) still runs.
ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, FindImageMemo* memo) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	const auto memo_mode           = memo != nullptr ? FindImageMemoMode() : 0u;
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		bool             memo_hit = false;
		if (memo_mode != 0 && memo->id && memo->epoch == m_image_index_epoch) {
			const auto* kept = m_slot_images.try_get(memo->id);
			memo_hit         = kept != nullptr && kept->registered;
		}
		if (memo_hit && memo_mode == 1) {
			result = memo->id;
		} else {
			const auto kept_id = memo_hit ? memo->id : ImageId {};
			result = FindImageLocked(command, desc, exact_format, memo_mode != 0 ? memo : nullptr);
			if (memo_hit && (kept_id != result || memo->id != result)) {
				EXIT("KYTY_FIND_IMAGE_MEMO=2: kept image %u, search gives %u (keepable %d) "
				     "addr=0x%010" PRIx64 " size=0x%" PRIx64 "\n",
				     static_cast<uint32_t>(kept_id.index), static_cast<uint32_t>(result.index),
				     memo->id ? 1 : 0, desc.info.data.address, desc.info.data.size);
			}
			if (!result) {
				return {};
			}
		}
		auto& image = m_slot_images[result];
		// The descriptor's base level and layer are relative to its own data. When it resolves
		// into a larger image that starts before that data, offset them by where the data starts
		// in the final image (a position taken from an earlier candidate may no longer apply).
		// A smaller mip can also start at the image's own address (mips stored smallest first);
		// only a one-mip descriptor of a mip chain can be one (SliceOf needs both).
		if (desc.info.data.address != image.info.data.address ||
		    (desc.info.resources.levels == 1 && image.info.resources.levels > 1 &&
		     desc.info.extent != image.info.extent)) {
			const int32_t mip   = desc.info.MipOf(image.info);
			const int32_t layer = mip >= 0 ? desc.info.SliceOf(image.info, mip) : -1;
			if (layer >= 0) {
				desc.view_info.base_level += static_cast<uint32_t>(mip);
				desc.view_info.base_layer += static_cast<uint32_t>(layer);
			}
		}
		image.tick_accessed_last = m_gc_tick;
		TouchImage(image);
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

// The search of FindImage, under m_lock. Returns an empty id when an image heap must wait for
// another extent. memo (if any) is filled when the result may be kept, cleared otherwise.
ImageId TextureCache::FindImageLocked(CommandBuffer& command, ImageDesc& desc, bool exact_format,
                                      FindImageMemo* memo) {
	ValidateImageDesc(desc);
	EXIT_IF(desc.info.data.Empty());

	ImageId result {};
	bool    keepable = true;
	{
		// Stays valid below: only FindImage (not reentrant, m_lock) rewrites cached lists.
		const auto& candidates =
		    FindImageCandidates(desc.info.data.address, desc.info.data.size);
		bool block_alias = false;

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			// Stencil descriptors name the associated plane; its depth owner supplies
			// the native image and the canonical guest tiling.
			if ((desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
			    image.depth_id && image.info.data == desc.info.data &&
			    m_slot_images[image.depth_id].info.samples == desc.info.samples &&
			    image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
			    image.info.resources.levels >= desc.info.resources.levels &&
			    image.info.resources.layers >= desc.info.resources.layers &&
			    desc.info.bytes_per_block == 1 &&
			    desc.info.guest_format == Prospero::BufferFormat::k8UInt) {
				result = id;
				break;
			}
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			} else if (BcAlias::Matches(desc.info, image.info)) {
				// SafeToDownload changes without an epoch bump: a possible alias is not kept.
				keepable = false;
				if (image.SafeToDownload()) {
					block_alias = true;
				}
			}
		}

		if (!result) {
			keepable = false;
			for (const auto candidate: candidates) {
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result = overlap.image;
				}
			}
			// PS5: image heaps hold a descriptor per dynamic-resolution extent of a render target,
			// and only one extent at an address holds data, so heap lookups create no second one.
			if (!result && desc.heap &&
			    std::any_of(candidates.begin(), candidates.end(), [&](ImageId id) {
				    const auto* image = m_slot_images.try_get(id);
				    return image != nullptr && image->registered &&
					       image->heap_refs.load(std::memory_order_relaxed) != 0u &&
					       image->info.data.address == desc.info.data.address &&
					       image->info.type == desc.info.type &&
					       image->info.extent != desc.info.extent;
			    })) {
				if (memo != nullptr) {
					*memo = {};
				}
				return {};
			}
		} else if (block_alias) {
			for (const auto candidate: candidates) {
				const auto& owner = m_slot_images[candidate];
				if (candidate != result && owner.registered && !owner.IsCpuDirty() &&
				    BcAlias::Matches(m_slot_images[result].info, owner.info)) {
					BcAlias::Resolve(*this, candidate, result);
				}
			}
		}

		if (result && !m_slot_images[result].depth_id) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			keepable       = false;
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				if (RefreshProf::Enabled()) [[unlikely]] {
					RefreshProf::NoteDirty(inserted, RefreshProf::NewGpu, inserted.info.data.address,
					                       inserted.info.data.size);
				}
				inserted.MarkBufferModified();
			}
			for (const auto id: candidates) {
				const auto* other = m_slot_images.try_get(id);
				if (other != nullptr && other->registered && other->guest_siblings &&
				    other->info.data == inserted.info.data) {
					inserted.guest_siblings = true;
					break;
				}
			}
			if (MsaaZeroInit() && inserted.info.samples > 1 && !inserted.info.IsDepth() &&
			    inserted.backing.image != nullptr && inserted.backing.layers != 0) {
				// Defines the native image only; guest bytes are untouched. ClearImage commits a
				// GPU write, which drops BufferModified (never uploaded for multisampled images).
				vk::ClearValue zero {};
				zero.color = vk::ClearColorValue {};
				ClearImage(command, result, inserted.backing.format,
				           {vk::ImageAspectFlagBits::eColor, 0, inserted.info.resources.levels, 0,
				            inserted.backing.layers},
				           zero);
			}
		}
	}
	if (ImageChurnOpt() == 2 && result && m_slot_images[result].guest_siblings) {
		VerifyGuestMirror(m_slot_images[result], "find");
	}
	if (memo != nullptr) {
		*memo = keepable ? FindImageMemo {result, m_image_index_epoch} : FindImageMemo {};
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	for (const auto id: FindImagesInRegion(address, size)) {
		const auto* owner = &m_slot_images[id];
		if (owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = &m_slot_images[owner->depth_id];
		}
		if (ensure_valid && !owner->SafeToDownload()) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid && m_slot_images[selected].depth_id) {
		selected = m_slot_images[selected].depth_id;
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (GpuCapture::Recording()) {
		GpuCapture::Touch(image.info.data.address, image.info.data.size);
		GpuCapture::Touch(image.info.stencil.address, image.info.stencil.size);
		GpuCapture::Touch(image.info.metadata.range.address, image.info.metadata.range.size);
	}
	TouchImage(image);
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
	}
	// Storage readback also queues the first GPU write since the CPU last wrote the image: an
	// image rewritten every frame would otherwise be downloaded at every GL2 writeback. GPU data
	// in the buffer cache over the whole image is not a CPU write. Read before RefreshImage
	// uploads the CPU write.
	const bool first_write =
	    m_readback_storage_images && (!image.IsGpuModified() || image.IsDefinitelyCpuDirty()) &&
	    !m_buffer_cache.HasOnlyGpuDirtyBytes(image.info.data.address, image.info.data.size);
	if (RefreshProf::Enabled()) [[unlikely]] {
		RefreshProf::NoteView(desc.view_info.base_level, desc.view_info.level_count,
		                      desc.view_info.base_layer, desc.view_info.layer_count, desc.view_info.min_lod);
	}
	RefreshImage(id, desc.type == BindingType::Texture ? &desc.view_info : nullptr);
	if (RefreshProf::Enabled()) [[unlikely]] {
		RefreshProf::ClearView();
	}
	if (image.info.HasStencil() &&
	    desc.info.data.address >= image.info.stencil.address &&
	    desc.info.data.End() <= image.info.stencil.End()) {
		for (const auto stencil_id:
		     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size)) {
			const auto& stencil = m_slot_images[stencil_id];
			if (stencil.depth_id == id && stencil.info.data == image.info.stencil) {
				RefreshImage(stencil_id);
				break;
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			CommitGpuWrite(image);
			TrackImageDownload(id, image);
			if (first_write) {
				m_download_images.insert(id);
			}
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	if (GpuCapture::Recording()) {
		GpuCapture::Touch(image.info.data.address, image.info.data.size, GpuCapture::Out::Color);
		GpuCapture::Touch(image.info.metadata.range.address, image.info.metadata.range.size);
	}
	TouchImage(image);
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	if (GpuCapture::Recording()) {
		GpuCapture::Touch(image.info.data.address, image.info.data.size);
		GpuCapture::Touch(desc.info.stencil.address, desc.info.stencil.size);
		GpuCapture::Touch(image.info.metadata.range.address, image.info.metadata.range.size);
	}
	TouchImage(image);
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto [metadata, inserted] = m_surface_metas.try_emplace(desc.info.metadata.range.address);
		if (inserted && image.info.htile_initial_clear) {
			metadata->second.clear_mask.set();
		}
		metadata->second.owner = id;
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

bool TextureCache::KeepBeside(Image& cached, const ImageInfo& requested, BindingType binding) {
	// Both images then hold the guest bytes: the old copy over a fresh upload rewrote them.
	constexpr uint64_t MaxSiblingBytes = 64 * 1024;
	if (ImageChurnOpt() == 0 || binding != BindingType::Texture || requested.data != cached.info.data ||
	    requested.resources != cached.info.resources || requested.data.size > MaxSiblingBytes ||
	    cached.IsGpuModified() || cached.IsBufferModified() || cached.depth_id ||
	    !cached.stencil_ids.empty() || requested.samples != 1 || cached.info.samples != 1 ||
	    cached.backing.samples != 1 || requested.HasStencil() || cached.info.HasStencil() ||
	    requested.HasMetadata() || cached.info.metadata.kind != ImageMetadataKind::None ||
	    requested.metadata.compression != VideoOutCompression::Uncompressed ||
	    cached.info.metadata.compression != VideoOutCompression::Uncompressed) {
		return false;
	}
	if (ImageChurnOpt() == 2) {
		VerifyGuestMirror(cached, "keep");
	}
	cached.guest_siblings = true;
	return true;
}

void TextureCache::DropGuestSiblings(Image& image) {
	if (!image.guest_siblings) {
		return;
	}
	image.guest_siblings = false;
	for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size)) {
		auto& sibling = m_slot_images[id];
		if (&sibling != &image && sibling.registered && sibling.guest_siblings &&
		    sibling.info.data == image.info.data) {
			sibling.binding.needs_rebind |= sibling.binding.is_bound || sibling.binding.is_target;
			FreeImage(id);
		}
	}
}

void TextureCache::VerifyGuestMirror(Image& image, const char* where) {
	static std::atomic<uint64_t> checked {0}, mismatched {0}, skipped {0};
	// Only an image that claims to hold the guest bytes now (no pending refresh) is compared.
	if (image.IsCpuDirty() || image.IsBufferModified() || image.IsGpuModified() ||
	    !CanDownload(image)) {
		skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	std::vector<uint8_t> guest(range.size);
	if (mapped == nullptr || !LibKernel::Memory::TryReadBacking(range.address, guest.data(), range.size)) {
		skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	download.Commit();
	// Bytes the native image does not cover (padding) read back as the guest's; the first byte
	// (always covered) starts inverted, so a readback that writes nothing mismatches.
	std::memcpy(mapped, guest.data(), range.size);
	static_cast<uint8_t*>(mapped)[0] ^= 0xffu;
	download.Flush(offset, range.size);
	m_scheduler.EndRendering();
	DownloadImage(image, download, offset, range.size, image.info.resources.levels);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                              vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	const auto& info = image.info;
	m_scheduler.DeferPriorityOperation([&download, mapped, offset, range, where, guest = std::move(guest),
	                                    w = info.extent.width, h = info.extent.height,
	                                    type = static_cast<uint32_t>(info.type),
	                                    format = static_cast<uint32_t>(info.pixel_format)] {
		download.Invalidate(offset, range.size);
		const auto n = checked.fetch_add(1, std::memory_order_relaxed) + 1;
		if (std::memcmp(mapped, guest.data(), range.size) != 0) {
			const auto bad = mismatched.fetch_add(1, std::memory_order_relaxed) + 1;
			LOGF_COLOR(Log::Color::BrightRed,
			           "KYTY_IMAGE_CHURN_OPT=2 mismatch (%s) addr=0x%010" PRIx64 " size=%" PRIu64
			           " %ux%u type=%u vk=%u: %" PRIu64 " of %" PRIu64 "\n",
			           where, range.address, range.size, w, h, type, format, bad, n);
		}
		if ((n & (n - 1)) == 0) {
			LOGF("KYTY_IMAGE_CHURN_OPT=2: %" PRIu64 " checked, %" PRIu64 " mismatches, %" PRIu64
			     " skipped\n",
			     n, mismatched.load(std::memory_order_relaxed), skipped.load(std::memory_order_relaxed));
		}
	});
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	DropGuestSiblings(image);
	image.ClearBufferModified();
	image.RefreshComplete();
	image.MarkGpuModified(GpuCapture::StampWrites());
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size)) {
		const auto&          owner = m_slot_images[id];
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner.info.resources.levels, 0, owner.info.TransferLayers()};
		if (owner.depth_id && owner.info.data.address == address &&
		    owner.info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner.depth_id;
			candidate_subresources = owner.stencil_subresources;
			const auto& depth      = m_slot_images[candidate_id];
			if (depth.backing.image == nullptr || !depth.info.HasStencil()) {
				continue;
			}
		} else if (!owner.depth_id && owner.info.data.address == address &&
		           owner.info.data.size == size) {
			candidate = owner.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                 : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		BeginColorRendering(image, format, range.baseMipLevel, range.baseArrayLayer,
		                    range.layerCount, &clear, command.Handle());
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	// A depth/stencil clear as a rendering load op leaves the image in the attachment layout:
	// no TRANSFER_DST round trip before the next depth pass.
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	if (ImageChurnOpt() != 0 && range.aspectMask != vk::ImageAspectFlagBits::eColor &&
	    range.levelCount == 1 && !image.info.IsVolume() &&
	    (image.backing.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) &&
	    range.layerCount <= limits.maxFramebufferLayers &&
	    std::max(image.info.extent.width >> range.baseMipLevel, 1u) <= limits.maxFramebufferWidth &&
	    std::max(image.info.extent.height >> range.baseMipLevel, 1u) <= limits.maxFramebufferHeight) {
		ImageViewInfo view {};
		view.format      = image.backing.format;
		view.type        = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.aspect      = aspects;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		image.Transit(vk::ImageLayout::eDepthStencilAttachmentOptimal,
		              vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                  vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		              {}, command.Handle());
		vk::RenderingAttachmentInfo attachments[2] {};
		for (uint32_t i = 0; i < 2; i++) {
			const auto aspect          = i == 0 ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eStencil;
			attachments[i].imageView   = image.FindView(view);
			attachments[i].imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
			attachments[i].loadOp      = (range.aspectMask & aspect) ? vk::AttachmentLoadOp::eClear
			                                                         : vk::AttachmentLoadOp::eLoad;
			attachments[i].storeOp     = vk::AttachmentStoreOp::eStore;
			attachments[i].clearValue  = clear;
		}
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {std::max(image.info.extent.width >> range.baseMipLevel, 1u),
		                               std::max(image.info.extent.height >> range.baseMipLevel, 1u)};
		rendering.layerCount         = range.layerCount;
		rendering.pDepthAttachment   = (aspects & vk::ImageAspectFlagBits::eDepth) ? &attachments[0] : nullptr;
		rendering.pStencilAttachment = (aspects & vk::ImageAspectFlagBits::eStencil) ? &attachments[1] : nullptr;
		command.Handle().beginRendering(&rendering);
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

vk::Rect2D TextureCache::BeginColorRendering(Image& image, vk::Format format, uint32_t level,
                                             uint32_t layer, uint32_t layers,
                                             const vk::ClearValue* clear,
                                             vk::CommandBuffer     command) {
	ImageViewInfo view {};
	view.format      = format;
	view.type        = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	view.base_level  = level;
	view.base_layer  = layer;
	view.layer_count = layers;
	view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentRead |
	                  vk::AccessFlagBits2::eColorAttachmentWrite,
	              {}, command);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = image.FindView(view);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	if (clear != nullptr) {
		attachment.loadOp     = vk::AttachmentLoadOp::eClear;
		attachment.clearValue = *clear;
	}
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {std::max(image.info.extent.width >> level, 1u),
	                                  std::max(image.info.extent.height >> level, 1u)};
	rendering.layerCount           = layers;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	return rendering.renderArea;
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

bool TextureCache::CanDownloadStencil(const Image& plane) const {
	const auto& depth = m_slot_images[plane.depth_id];
	// Multisampled or compressed stencil is never uploaded either (RefreshImage).
	return depth.info.samples == 1 && !depth.info.metadata.stencil_compressed &&
	       depth.backing.image != nullptr && plane.info.resources.layers != 0 &&
	       plane.info.data.size % plane.info.resources.layers == 0 && plane.info.LinearSize() != 0;
}

void TextureCache::DownloadStencil(Image& plane, Buffer& destination, uint64_t destination_offset) {
	// Reverse of UploadImage for a stencil association: the depth owner holds the aspect.
	const auto& info  = plane.info;
	auto&       depth = m_slot_images[plane.depth_id];
	auto        copies = info.BufferCopies(info.IsTiled() ? 0 : destination_offset);
	for (auto& copy: copies) {
		copy.imageSubresource.baseArrayLayer += plane.stencil_subresources.base_layer;
	}
	if (!info.IsTiled()) {
		depth.Download(copies, destination.Handle(), destination_offset, info.data.size);
		return;
	}
	const auto linear = m_tiler.GetScratchBuffer(info.LinearSize());
	depth.Download(copies, linear.buffer, 0, linear.size);
	m_tiler.Tile(linear.buffer, 0, linear.size, destination.Handle(), destination_offset,
	             info.data.size, info);
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
		                       transform, false);
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	ForgetColorMetadataChecks(image.info.data.address, copy_size);
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, Buffer* staging, bool capture) {
	auto& image = m_slot_images[id];
	if (image.depth_id ? !capture || !CanDownloadStencil(image) : !CanDownload(image, capture)) {
		return false;
	}
	const auto range = image.info.data;
	if (!image.SafeToDownload()) {
		return false;
	}
	// The capture's staging buffer, or the download ring with a temporary buffer past its size.
	auto download = staging != nullptr
	                    ? StreamBuffer::Download {staging, staging->Mapped().data(), 0, nullptr}
	                    : m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download)
	                          .MapDownload(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (!LibKernel::Memory::TryReadBacking(range.address, download.mapped, range.size)) {
		return false;
	}
	download.buffer->Flush(download.offset, range.size);

	if (image.depth_id) {
		DownloadStencil(image, *download.buffer, download.offset);
	} else {
		DownloadImage(image, *download.buffer, download.offset, range.size,
		              image.info.resources.levels);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.buffer->Handle();
	barrier.offset              = download.offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	// With storage readback, guest accesses to its pages wait until the stored data lands.
	if (m_readback_storage_images) {
		{
			std::scoped_lock write_back_lock {m_write_back_mutex};
			m_write_backs.push_back({range, m_scheduler.CurrentTick()});
		}
		m_page_manager.UpdatePageWatchers<true, true>(range.address, range.size);
	}
	m_scheduler.DeferPriorityOperation([this, download = std::move(download), range,
	                                    trapped = m_readback_storage_images] {
		download.buffer->Invalidate(download.offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, download.mapped, range.size);
		if (trapped) {
			m_page_manager.UpdatePageWatchers<false, true>(range.address, range.size);
			// Priority operations run in queue order, so this write-back is the oldest.
			std::scoped_lock write_back_lock {m_write_back_mutex};
			m_write_backs.erase(m_write_backs.begin());
		}
	});
	return true;
}

uint64_t TextureCache::WriteBackTick(uint64_t address, uint64_t size) {
	if (!m_readback_storage_images) {
		return 0;
	}
	const auto begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	uint64_t   tick  = 0;

	std::scoped_lock write_back_lock {m_write_back_mutex};
	for (const auto& write_back: m_write_backs) {
		if (write_back.range.address < end && begin < write_back.range.End()) {
			tick = std::max(tick, write_back.tick);
		}
	}
	return tick;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size, uint8_t reason) {
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (RefreshProf::Enabled()) [[unlikely]] {
			RefreshProf::NoteDirty(image, reason, address, size);
		}
		image.ClearGpuModified();
		image.MarkBufferModified();
	}
}

bool TextureCache::CopyToImage(uint64_t address, uint64_t source, uint64_t size) {
	std::scoped_lock lock {m_lock};
	const auto       targets = FindImagesInRegion(address, size);
	if (targets.size() != 1) {
		return false;
	}
	auto&       image   = m_slot_images[targets.front()];
	const auto& info    = image.info;
	const auto  sources = FindImagesInRegion(source, size);
	if (image.depth_id || !image.SafeToDownload() || !CanDownload(image) || !info.IsTiled() ||
	    info.IsVolume() || info.IsDepth() || info.HasMetadata() || address < info.data.address ||
	    std::any_of(sources.begin(), sources.end(),
	                [&](ImageId id) { return m_slot_images[id].IsGpuModified(); })) {
		return false;
	}
	// Each layer holds its levels from the smallest up, and the tail levels share one block.
	const auto layers  = info.resources.layers;
	const auto stride  = info.data.size / layers;
	const auto offset  = address - info.data.address;
	const auto begin   = offset % stride;
	const auto end     = begin + size;
	uint64_t   covered = begin;
	uint32_t   first   = 0;
	uint32_t   count   = 0;
	for (uint32_t level = info.resources.levels; level-- > 0;) {
		const auto& mip     = info.mip_layout[level];
		const auto  mip_end = mip.offset + mip.size / layers;
		if (mip_end <= begin || mip.offset >= end) {
			continue;
		}
		if (mip.offset < begin) {
			return false;
		}
		covered = mip_end;
		first   = level;
		count++;
	}
	if (covered != end) {
		return false;
	}
	ImageInfo  part   = info;
	const auto extent = info.MipExtent(first);
	part.data.size    = size;
	part.extent       = {extent.width, extent.height, 1};
	part.resources    = {count, 1};
	part.first_tail_level -= std::min(first, info.first_tail_level);
	uint64_t linear = 0;
	for (uint32_t level = 0; level < count; level++) {
		auto& mip = part.mip_layout[level] = info.mip_layout[first + level];
		mip.offset -= begin;
		mip.size /= layers;
		mip.linear_offset = Common::AlignUp<uint64_t>(linear, std::max(info.bytes_per_block, 4u));
		linear            = mip.linear_offset + mip.linear_size;
	}
	const auto [buffer, source_offset] = m_buffer_cache.ObtainBuffer(source, size, false);
	const auto texels = m_tiler.Detile(buffer->Handle(), source_offset, part);
	auto       copies = part.BufferCopies(texels.offset);
	for (auto& copy: copies) {
		copy.imageSubresource.mipLevel += first;
		copy.imageSubresource.baseArrayLayer = static_cast<uint32_t>(offset / stride);
	}
	image.Upload(copies, texels.buffer, texels.offset, texels.size);
	// The per-level hashes of the last refresh no longer describe the image (DeferRefresh).
	image.refresh_note.native_is_guest = false;
	return true;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size, uint8_t reason) {
	std::scoped_lock lock {m_lock};
	const auto       page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto       page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(page_begin, page_end - page_begin)) {
		auto& owner = m_slot_images[id];
		if (owner.Overlaps(address, size)) {
			if (RefreshProf::Enabled()) [[unlikely]] {
				RefreshProf::NoteDirty(owner, reason, address, size);
			}
			// KYTY_REFRESH_SKIP_SAME: a fault keeps the watch below its page, so a later refresh knows
			// that only bytes from there on may have changed (DeferRefresh).
			if (reason == RefreshProf::CpuFault && RefreshProf::SkipSameMode() != 0 && owner.IsTracked() &&
			    owner.track_addr == owner.info.data.address) [[unlikely]] {
				const auto page = Common::AlignDown(address, TRACKER_PAGE_SIZE);
				owner.MarkCpuFaultDirty();
				owner.refresh_note.fault_from = std::min(owner.refresh_note.fault_from, page);
				UntrackImageFrom(id, page);
				continue;
			}
			owner.MarkCpuDirty();
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner.info.data.address;
		const auto image_end   = owner.info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	return found->second.clear_mask.test(slice);
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask.set();
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	found->second.clear_mask.set(slice, is_clear);
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	m_surface_metas.erase(m_surface_metas.lower_bound(address),
	                      m_surface_metas.lower_bound(address + size));
	auto images = FindImagesInRegion(address, size);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool     pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool     aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		uint64_t age        = 16;
		size_t   deletions  = 10;
		if (aggressive) {
			age       = 160;
			deletions = 40;
		} else if (pressured) {
			age       = 80;
			deletions = 20;
		}
		age = std::min(age, tick);
		std::vector<ImageId> candidates;
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) { candidates.push_back(id); });
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			if (owner->heap_refs.load(std::memory_order_relaxed) != 0u) {
				// Resident heaps sample it without touching it.
				TouchImage(*owner);
				continue;
			}
			if (owner->SafeToDownload() &&
			    (!pressured || owner->info.HasStencil() || !DownloadImageMemory(id))) {
				continue;
			}
			--deletions;
			FreeImage(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

void TextureCache::DownloadGpuModified(std::vector<GpuCapture::Unpublished>* skipped,
                                       std::vector<GpuCapture::Published>*   written) {
	std::vector<ImageId> ids;
	const auto           limit = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download).Size();
	{
		std::scoped_lock lock {m_lock};
		m_slot_images.ForEach([&](ImageId id, Image& image) {
			if (!image.registered || !image.IsGpuModified()) return;
			const auto* owner  = image.depth_id ? &m_slot_images[image.depth_id] : nullptr;
			const char* reason = owner != nullptr && owner->info.samples != 1 ? "stencil_msaa"
			                     : owner != nullptr && owner->info.metadata.stencil_compressed ? "stencil_compressed"
			                     : !image.SafeToDownload()    ? "cpu_or_buffer_modified"
			                     : owner != nullptr           ? (CanDownloadStencil(image) ? nullptr : "stencil_format")
			                     : CanDownload(image, true)   ? nullptr
			                     : image.info.samples != 1    ? "msaa"
			                     : UploadBinding(image) == BindingType::DepthTarget ? "depth_format"
			                                                                        : "format";
			if (reason == nullptr) {
				ids.push_back(id);
			} else if (skipped != nullptr) {
				skipped->push_back({image.info.data.address, image.info.data.size, reason});
			}
		});
		// Oldest GPU write first: where a stale registered image overlaps a newer one, the newer bytes land last.
		std::ranges::stable_sort(ids, {}, [&](ImageId id) { return m_slot_images[id].gpu_write_seq; });
	}
	// One image at a time: the download buffer is reused; a larger image gets its own buffer.
	for (const auto id: ids) {
		bool                    downloaded = false;
		std::unique_ptr<Buffer> staging;
		{
			std::scoped_lock lock {m_lock};
			const auto       size = m_slot_images[id].info.data.size;
			if (size > limit) {
				staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
				                                   vk::BufferUsageFlagBits::eTransferDst, size);
			}
			downloaded       = DownloadImageMemory(id, staging.get(), true);
			const auto& image = m_slot_images[id];
			if (!downloaded && skipped != nullptr) {
				skipped->push_back({image.info.data.address, image.info.data.size, "backing"});
			}
			if (downloaded && written != nullptr) {
				written->push_back({image.info.data.address, image.info.data.size, image.gpu_write_seq, id.index,
				                    image.depth_id            ? "stencil"
				                    : image.info.IsDepth()    ? "depth"
				                    : image.usage.render_target ? "color"
				                                                : "other"});
			}
		}
		if (downloaded) {
			m_scheduler.Finish();
			m_scheduler.WaitPriorityOperations(m_scheduler.CurrentTick() - 1);
		}
	}
}

bool TextureCache::Thumbnail(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                             std::vector<uint8_t>& rgba) {
	VulkanImage             converted;
	std::unique_ptr<Buffer> download;
	vk::Extent3D            extent;
	{
		std::scoped_lock lock {m_lock};
		Image*           source = nullptr;
		for (const auto id: FindImagesInRegion(address, size)) {
			auto& image = m_slot_images[id];
			if (image.info.data.address != address || image.info.data.size > size || !image.registered ||
			    !image.IsGpuModified() || !image.usage.render_target || image.info.IsDepth() || image.depth_id ||
			    image.info.samples != 1 || image.backing.image == nullptr) {
				continue;
			}
			if (source == nullptr || image.gpu_write_seq > source->gpu_write_seq) source = &image;
		}
		if (source == nullptr || !(m_graphics.GetFormatProperties(source->backing.format).optimalTilingFeatures &
		                           vk::FormatFeatureFlagBits::eBlitSrc)) {
			return false;
		}
		extent = vk::Extent3D {source->backing.extent.width, source->backing.extent.height, 1};
		vk::ImageCreateInfo create {};
		create.imageType   = vk::ImageType::e2D;
		create.extent      = extent;
		create.mipLevels   = 1;
		create.arrayLayers = 1;
		create.format      = vk::Format::eR8G8B8A8Unorm;
		create.tiling      = vk::ImageTiling::eOptimal;
		create.usage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
		create.samples     = vk::SampleCountFlagBits::e1;
		if (!m_graphics.CreateImage(create, converted)) return false;
		download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                    vk::BufferUsageFlagBits::eTransferDst, uint64_t {extent.width} * extent.height * 4);
		m_scheduler.EndRendering();
		auto command = m_scheduler.Current().Handle();
		source->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, std::nullopt, command);
		const vk::ImageSubresourceRange color {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		vk::ImageMemoryBarrier          barrier {};
		barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.oldLayout           = vk::ImageLayout::eUndefined;
		barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = converted.image;
		barrier.subresourceRange    = color;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, 0,
		                        nullptr, 0, nullptr, 1, &barrier);
		const vk::Offset3D corner {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
		vk::ImageBlit      blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.dstSubresource = blit.srcSubresource;
		blit.srcOffsets[1]  = corner;
		blit.dstOffsets[1]  = corner;
		// Same size: the blit only converts the format (sRGB decoded, floats clamped to [0, 1]).
		command.blitImage(source->backing.image, vk::ImageLayout::eTransferSrcOptimal, converted.image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eNearest);
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 0,
		                        nullptr, 0, nullptr, 1, &barrier);
		vk::BufferImageCopy copy {};
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent      = extent;
		command.copyImageToBuffer(converted.image, vk::ImageLayout::eTransferSrcOptimal, download->Handle(), 1, &copy);
		vk::BufferMemoryBarrier host {};
		host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		host.buffer              = download->Handle();
		host.size                = download->Size();
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
		                        1, &host, 0, nullptr);
	}
	m_scheduler.Finish();
	download->Invalidate(0, download->Size());
	m_graphics.DeleteImage(converted);
	// Box filter: each thumbnail pixel is the rounded mean of the source pixels its cell covers.
	const auto* pixels = download->Mapped().data();
	rgba.assign(size_t {width} * height * 4, 0);
	for (uint32_t ty = 0; ty < height; ty++) {
		for (uint32_t tx = 0; tx < width; tx++) {
			const uint64_t x0 = uint64_t {tx} * extent.width / width, x1 = std::max(x0 + 1, uint64_t {tx + 1} * extent.width / width);
			const uint64_t y0 = uint64_t {ty} * extent.height / height, y1 = std::max(y0 + 1, uint64_t {ty + 1} * extent.height / height);
			uint64_t       sum[4] {};
			for (uint64_t y = y0; y < y1; y++) {
				for (uint64_t x = x0; x < x1; x++) {
					for (int c = 0; c < 4; c++) sum[c] += pixels[(y * extent.width + x) * 4 + c];
				}
			}
			const uint64_t count = (x1 - x0) * (y1 - y0); // >= 1: x0 < x1 <= width, y0 < y1 <= height
			for (int c = 0; c < 4; c++) {
				rgba[(size_t {ty} * width + tx) * 4 + c] = static_cast<uint8_t>((sum[c] + count / 2) / count);
			}
		}
	}
	return true;
}

void TextureCache::ProcessDownloadImages() {
	if (!m_readback_linear_images && !m_readback_tiled_images) {
		return;
	}
	(void)DownloadQueuedImages();
}

bool TextureCache::WriteBackImages() {
	return m_readback_storage_images && DownloadQueuedImages();
}

bool TextureCache::DownloadQueuedImages() {
	std::scoped_lock lock {m_lock};
	bool             downloaded = false;
	for (const auto id: m_download_images) {
		if (m_slot_images[id].IsGpuModified()) {
			downloaded |= DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
	return downloaded;
}

} // namespace Libs::Graphics
