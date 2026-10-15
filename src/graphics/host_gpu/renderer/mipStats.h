#ifndef GRAPHICS_HOST_GPU_RENDERER_MIP_STATS_H_
#define GRAPHICS_HOST_GPU_RENDERER_MIP_STATS_H_

#include <cstdint>

// GET_LOD_STATS reports and the texture streaming thrash relief (after BryKytyPS5 70fa382e).
//
// A report is a 64-byte header whose first dword marks it complete, then one 64-bit value per mip
// statistics counter (a sampled texture's MIP_STATS_CNT_ID, descriptor dword 6, counted when dword
// 5 bit 25 is set): the finest mip sampled in bits 56-59 (15: none) and a sample count in bits
// 0-23. A game's texture streamer can move each texture's detail to the reported mip.
//
// The emulator cannot tell which mips a draw samples, so every counter reports mip 0. When every
// texture claims full detail, the streamer can overflow its memory while a level loads behind the
// loading screen and reload the same textures in turn forever. When the game
// is seen reading the same large file region again and again (NoteRepeatedFileRead), for a while
// the counters that no draw marked recently report no samples, so their textures drop out of the
// way and the load completes.
//
// KYTY_MIP_STATS_RELIEF=0 restores the old report (all zeros, no relief).
namespace Libs::Graphics::MipStats {

bool ReliefEnabled();
// A draw samples a texture whose descriptor enables mip statistics counter `id`.
void Mark(uint32_t id) noexcept;
// Writes a GET_LOD_STATS report of `size` bytes to `dst`, then starts counting anew if `reset`.
void Report(void* dst, uint32_t size, bool reset);
// A file region was read; four reads of the same large region within 10 s start a relief.
void NoteRepeatedFileRead(uint64_t file_key, uint64_t offset, uint64_t size);

} // namespace Libs::Graphics::MipStats

#endif
