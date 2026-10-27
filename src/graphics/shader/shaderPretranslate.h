#ifndef EMULATOR_SRC_GRAPHICS_SHADER_SHADERPRETRANSLATE_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_SHADERPRETRANSLATE_H_

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <span>

// Shader translation when the game registers a shader, instead of at its first draw.
//
// KYTY_SHADER_PRETRANSLATE=N (default 0 = off): N low-priority host workers translate each registered
// pixel shader (GCN -> IR: decoding, CFG, SSA, simplification and resource tracking, the costly and
// state-independent part of a shader miss) with the translation inputs predicted from the register
// defaults of its binary (ShaderPredictPixelProgram). On a program-cache miss the GPU thread takes the
// translated program when its own translation key (code hash, BuildStageTranslationKey and the program
// inputs) is equal and the code words are unchanged, and only does the per-draw specialization (SPIR-V
// emission for the draw's static state and resources). Otherwise it translates as before.
//   KYTY_SHADER_PRETRANSLATE_MB (default 1024): memory budget of translated programs not yet used;
//     the oldest are dropped first.
//   KYTY_SHADER_PRETRANSLATE_REFILL (default 1): how many times a used program (taken, or translated by
//     the GPU thread) is translated again in the background, for the next static state or resource
//     permutation of the shader. About half of the used pixel programs need a second one, very few a third.
//   KYTY_SHADER_PRETRANSLATE_PACK (default 1): keep the programs packed (IR::PackedProgram, about a
//     tenth of the memory), unpacked by the GPU thread when taken.
//   KYTY_SHADER_PRETRANSLATE_VERIFY=1: the GPU thread also translates and compares the emitted SPIR-V.
// Workers are host-only threads: no guest state, no guest memory access (the code is copied at
// registration). A fatal check during a speculative translation drops that shader: its worker is
// retired without unwinding (Common::SetFatalRetiresThread) and replaced; after 8 retired workers,
// pretranslation stops for the session.
namespace Libs::Graphics::ShaderPretranslate {

[[nodiscard]] bool Enabled() noexcept;
[[nodiscard]] bool VerifyEnabled() noexcept;

// Guest thread registering the shader (ShaderMapUserData); `data` register pointers are valid.
void OnShaderRegistered(uint64_t address, uint64_t hash, const ShaderMappedData& data);

// GPU thread, on a pixel program miss. `key`: BuildStageTranslationKey + AppendProgramTranslationKey.
// True: `out` is the translation of `code` for that key.
[[nodiscard]] bool Take(uint64_t hash, std::span<const uint32_t> key, std::span<const uint32_t> code,
                        ShaderRecompiler::TranslateResult& out);

// GPU thread, after translating a pixel program itself: translate it again in the background for its
// next static state or resource permutation (KYTY_SHADER_PRETRANSLATE_REFILL).
void Remember(uint64_t hash, std::span<const uint32_t> key, std::span<const uint32_t> code,
              const ShaderPixelInputInfo& info, uint32_t user_data_count);

void NoteVerify(bool equal) noexcept;

// "PERF pretranslate" line, printed with the PERF report.
void Report();

} // namespace Libs::Graphics::ShaderPretranslate

#endif // EMULATOR_SRC_GRAPHICS_SHADER_SHADERPRETRANSLATE_H_
