#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDESCMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDESCMEMO_H_

#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>

namespace Libs::Graphics::TargetDescMemo {

// KYTY_TARGET_DESC_MEMO (default 1; 0 = off; 2 = verify): render-target resolution keeps the
// image description it derived from a color target's or the depth target's registers and
// reuses it the next time a draw binds the same register values, skipping the checks, the
// format and footprint derivation and the tiled layout (ImageInfo::UpdateSize). The description
// is a pure function of those registers (and of the device, fixed for the process); the
// texture cache lookup and everything after it still run per draw. 2 = derive anyway and exit
// on any difference with the kept description.
inline uint32_t Mode() noexcept {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_TARGET_DESC_MEMO");
		return value == nullptr ? 1u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}();
	return mode;
}

template <typename Key>
uint64_t HashBytes(const Key& key, uint64_t seed) noexcept {
	static_assert(std::is_trivially_copyable_v<Key>);
	const auto* bytes = reinterpret_cast<const unsigned char*>(&key);
	uint64_t    hash  = 0x9e3779b97f4a7c15ull ^ seed;
	for (size_t i = 0; i < sizeof(Key); ++i) {
		hash = (hash ^ bytes[i]) * 0x100000001b3ull;
	}
	return hash ^ (hash >> 29u);
}

// Direct-mapped table per host thread (the GPU thread, and the KYTY_CP_PIPELINE back thread):
// no lock, no shared state. Keys are compared bytewise, so a padding difference can only miss.
template <typename Key, typename Value, size_t Slots>
class Table {
public:
	struct Entry {
		bool  valid = false;
		Key   key {};
		Value value {};
		// FindImage results for this description (KYTY_FIND_IMAGE_MEMO), per exact_format.
		std::array<TextureCache::FindImageMemo, 2> find {};
	};

	// The entry for key; *hit tells whether it holds key's value. A miss leaves it to Store.
	Entry& Lookup(const Key& key, uint64_t seed, bool* hit) {
		if (!m_entries) {
			m_entries = std::make_unique<std::array<Entry, Slots>>();
		}
		auto& entry = (*m_entries)[HashBytes(key, seed) & (Slots - 1u)];
		*hit = entry.valid && std::memcmp(&entry.key, &key, sizeof(Key)) == 0;
		return entry;
	}

	static void Store(Entry& entry, const Key& key, const Value& value) {
		entry.valid = true;
		std::memcpy(&entry.key, &key, sizeof(Key));
		entry.value = value;
		entry.find  = {};
	}

private:
	std::unique_ptr<std::array<Entry, Slots>> m_entries;
};

inline bool SameImageDesc(const TextureCache::ImageDesc& a, const TextureCache::ImageDesc& b) {
	const auto& x  = a.info;
	const auto& y  = b.info;
	const auto& xm = x.metadata;
	const auto& ym = y.metadata;
	return a.type == b.type && a.heap == b.heap && a.view_info == b.view_info && x.data == y.data &&
	       x.stencil == y.stencil && xm.range == ym.range && xm.kind == ym.kind &&
	       xm.control == ym.control && xm.clear_word == ym.clear_word &&
	       xm.compression == ym.compression && xm.stencil_compressed == ym.stencil_compressed &&
	       xm.clear_register_valid == ym.clear_register_valid &&
	       xm.dcc_alpha_msb == ym.dcc_alpha_msb && x.htile_initial_clear == y.htile_initial_clear &&
	       x.pixel_format == y.pixel_format && x.guest_format == y.guest_format &&
	       x.type == y.type && x.extent == y.extent && x.resources == y.resources &&
	       x.pitch == y.pitch && x.bytes_per_block == y.bytes_per_block &&
	       x.samples == y.samples && x.tile_mode == y.tile_mode && x.bgra16 == y.bgra16 &&
	       x.mip_layout == y.mip_layout &&
	       std::memcmp(&x.tiling, &y.tiling, sizeof(x.tiling)) == 0 &&
	       x.tiled_slice_stride == y.tiled_slice_stride &&
	       x.linear_slice_stride == y.linear_slice_stride &&
	       x.first_tail_level == y.first_tail_level;
}

} // namespace Libs::Graphics::TargetDescMemo

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TARGETDESCMEMO_H_
