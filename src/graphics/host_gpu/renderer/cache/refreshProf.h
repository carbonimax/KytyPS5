#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_REFRESHPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_REFRESHPROF_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

class Image;

// KYTY_REFRESH_PROF=1: why texture cache images are re-uploaded (detiled). Each dirty mark records
// who made it and the byte span; each upload records its bytes, the span's mips and layers, the view
// that asked for it and whether the guest bytes hash the same as at the image's previous upload.
// A "REFRESH" summary and a per-image top list are printed every 5 s. Off, only Enabled() runs.
namespace RefreshProf {

enum Reason : uint8_t {
	New,        // first upload of a new image record
	NewGpu,     // new record over bytes the GPU owns in the buffer cache
	CpuFault,   // guest CPU write fault on a watched page
	HostWrite,  // emulator write into guest memory (file reads, stacks, ...)
	Label,      // command processor label / WRITE_DATA into guest memory
	GpuFill,    // DMA fill of GPU-owned bytes
	GpuCopy,    // DMA copy into guest memory
	GpuStorage, // storage buffer bound as written at the image address
	ImageCopy,  // CopyImage of a source whose bytes are buffer-owned
	BcAlias,    // block/raw alias published through the buffer cache
	Edge,       // head/tail page write outside the image, edge hash changed
	ReasonCount
};

[[nodiscard]] bool Enabled() noexcept;
// KYTY_REFRESH_PROF=2: also hash the guest bytes of every upload ("dirty but same bytes").
[[nodiscard]] bool HashEnabled() noexcept;
// KYTY_REFRESH_SKIP_SAME: 0 off; 1 a texture lookup keeps the native image while every byte that
// may have changed since its last upload lies in mips its view cannot sample (below base_level +
// floor(min_lod)); 2 verify: decide the same, upload anyway, and count a mismatch when a sampled
// mip's guest bytes differ from that upload.
[[nodiscard]] uint32_t SkipSameMode() noexcept;
// A deferred refresh (first of the dirty cycle: `first`), or one mode 2 checked.
void NoteDefer(uint64_t bytes, bool first) noexcept;

void NoteDirty(Image& image, uint8_t reason, uint64_t address, uint64_t size) noexcept;
// The view a texture lookup asks for, kept until the next upload.
void NoteView(uint32_t base_level, uint32_t levels, uint32_t base_layer, uint32_t layers, uint32_t min_lod) noexcept;
void ClearView() noexcept;
// One InitializeImage upload. `gpu_source`: the bytes came from the buffer cache, not the guest.
// `skipped`: KYTY_REFRESH_SKIP_SAME kept the native image. `mismatch`: verify mode found the
// skipped content different.
void NoteUpload(Image& image, bool gpu_source, uint64_t record_ns, int same, bool skipped) noexcept;
void NoteMismatch() noexcept;
// XXH3 of guest bytes read from the backing (never faults); 0 when unreadable.
[[nodiscard]] uint64_t HashGuest(uint64_t address, uint64_t size) noexcept;

// Mip and layer holding byte `offset` of a guest image (tiled colors are mip-major, each mip
// spanning all layers). Exposed for the unit test.
struct MipLayer {
	uint32_t mip   = 0;
	uint32_t layer = 0;
};
[[nodiscard]] inline MipLayer Locate(const uint64_t* mip_offsets, const uint64_t* mip_sizes, uint32_t levels, uint32_t layers,
                uint64_t offset) noexcept {
	MipLayer result {};
	uint64_t best = UINT64_MAX;
	for (uint32_t m = 0; m < levels; m++) {
		// Tail levels share one block: the smallest containing span names the level.
		if (offset >= mip_offsets[m] && offset - mip_offsets[m] < mip_sizes[m] && mip_sizes[m] < best) {
			best         = mip_sizes[m];
			result.mip   = m;
			const auto per_layer = layers > 1 ? mip_sizes[m] / layers : 0;
			result.layer = per_layer != 0 ? static_cast<uint32_t>(std::min<uint64_t>((offset - mip_offsets[m]) / per_layer, layers - 1)) : 0;
		}
	}
	return result;
}

// True when every level whose bytes intersect [dirty_from, size) is below `first_sampled`, and at
// least one level is sampled. Tail levels sharing a block intersect together.
[[nodiscard]] inline bool DirtyBelow(const uint64_t* mip_offsets, const uint64_t* mip_sizes, uint32_t levels,
                                     uint64_t dirty_from, uint64_t size, uint32_t first_sampled) noexcept {
	if (first_sampled == 0 || first_sampled >= levels || dirty_from >= size) {
		return false;
	}
	for (uint32_t m = 0; m < levels; m++) {
		if (mip_sizes[m] == 0 || mip_offsets[m] + mip_sizes[m] > size) {
			return false;
		}
		if (m >= first_sampled && mip_offsets[m] + mip_sizes[m] > dirty_from) {
			return false;
		}
	}
	return true;
}

} // namespace RefreshProf
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_REFRESHPROF_H_
