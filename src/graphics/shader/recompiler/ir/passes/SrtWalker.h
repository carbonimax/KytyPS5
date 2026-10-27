#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

// Image descriptors at base + first + k * step, k < count, within size bytes that a sampled
// root selects by GPU-computed byte offset. The host keeps them resident instead of
// materializing them.
struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
};

struct ImageHeapView {
	uint64_t                         slots      = 0; // Device address of one slot per heap dword.
	uint32_t                         slot_count = 0; // Keys past it select a trailing null slot.
	std::span<const DescriptorValue> kinds; // One representative descriptor per kind.
};

class ImageHeapSource {
public:
	// False when the host does not keep the table resident.
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;

protected:
	~ImageHeapSource() = default;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr;
	// False when the host has no null descriptors: null buffers and images get a fallback.
	bool                      null_descriptor_supported = true;
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
// Compiles the plan's x86 walker. Without values it walks the plan's resources; otherwise it
// only evaluates the values, with every block reachable.
void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values = {});
// Runs the walker. flat receives the scalar reads of reachable blocks and reads, when non-null,
// one DWORD range per guest load. False when a reachable read is unknown.
bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads);
// Results of the last walk; false when unknown.
bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result);
bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result);
bool WalkedActive(const ResourcePlan& plan, uint32_t source);
// Measurement (KYTY_PERF_STATS): once enabled, walks count their number and wall time, and the
// fault handler counts walker loads from protected pages (`backing`: completed from the backing,
// else through the page fault path). Taken (and reset) by the PERF report.
struct WalkStats {
	uint64_t walks = 0, ns = 0, backing_reads = 0, fault_reads = 0, protected_calls = 0,
	         protected_evictions = 0, protected_ns = 0, protected_synced = 0;
	// KYTY_SRT_WALK_REUSE: walks whose inputs matched the plan's last walk (checks), of which every
	// guest load still read the same value (hits); validation time; verify-mode mismatches.
	uint64_t reuse_checks = 0, reuse_hits = 0, reuse_ns = 0, reuse_diffs = 0;
};
// KYTY_SRT_WALK_REUSE (default 1; x86-64 walker; 0 = off): a walk whose user data and shader base equal the
// plan's last walk, and whose recorded guest loads all still read the recorded values (loaded the
// way the walker loads them: protected-page list, backing reads, fault handling), is not run again:
// the plan's scratch still holds that walk's results. 1 = reuse; 2 = verify (walk anyway and
// compare the flat buffer and the whole scratch; mismatches counted and logged). Walks that
// restarted are never reused.
[[nodiscard]] uint32_t WalkReuseMode() noexcept;
// KYTY_SRT_PROTECTED_LOADS (default 1): the fault handler lists the page of a walker load it
// completed from the backing (NoteProtectedLoad); later walker loads from a listed page call the
// reader (GPU thread: the backing, synchronizing GPU-written bytes) instead of faulting. The reader
// returns ProtectedLoadValue when it read the value, and ProtectedLoadKeep while the page is still
// protected; without Keep the page leaves the list, and without Value the walker loads directly
// (through the fault path when the page is protected after all).
using ProtectedLoadReader = uint32_t (*)(uint64_t address, uint32_t* value);
constexpr uint32_t ProtectedLoadValue = 1u;
constexpr uint32_t ProtectedLoadKeep  = 2u;
constexpr uint32_t ProtectedLoadSynced = 4u; // measurement: GPU-written bytes were synchronized first
void SetProtectedLoadReader(ProtectedLoadReader reader) noexcept;
void NoteProtectedLoad(uint64_t address) noexcept;
void      EnableWalkStats() noexcept;
[[nodiscard]] bool WalkStatsEnabled() noexcept;
void      NoteWalkerFaultRead(bool backing) noexcept;
WalkStats TakeWalkStats() noexcept;
// Length of the guest load at rip if it belongs to the walker running on this thread, else 0.
// The exception handler completes such a load as a zero read.
uint32_t SrtWalkerLoadLength(uint64_t rip);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
