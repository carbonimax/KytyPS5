// KYTY_SPIRV_OPT (step 5.16). Without arguments: a small module goes through OptimizeSpirv and
// comes out valid, inlined and with the same entry points; a broken binary is left unchanged.
// With arguments: each is a shader store (_PipelineCache/<title>.shaders); every SPIR-V record in
// it goes through OptimizeSpirv at level 2 and the instruction counts, optimizer time and
// failures are printed (offline measurement, no GPU).
#include "graphics/shader/recompiler/backend/spirv/SpirvOptimizer.h"

#include <spirv-tools/libspirv.hpp>
#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace Libs::Graphics::ShaderRecompiler::Spirv;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::printf("FAIL: %s\n", message);
		g_failures++;
	}
}

uint32_t CountInstructions(const std::vector<uint32_t>& spirv, uint32_t opcode = UINT32_MAX) {
	uint32_t count = 0;
	for (size_t i = 5; i < spirv.size();) {
		const uint32_t words = spirv[i] >> 16u;
		count += opcode == UINT32_MAX || (spirv[i] & 0xffffu) == opcode ? 1u : 0u;
		i += words == 0 ? 1 : words;
	}
	return count;
}

// "model:name" of each OpEntryPoint.
std::vector<std::string> EntryPoints(const std::vector<uint32_t>& spirv) {
	std::vector<std::string> result;
	for (size_t i = 5; i < spirv.size();) {
		const uint32_t words = spirv[i] >> 16u;
		if ((spirv[i] & 0xffffu) == 15u && words > 3 && i + words <= spirv.size()) {
			const auto* name = reinterpret_cast<const char*>(&spirv[i + 3]);
			result.push_back(std::to_string(spirv[i + 1]) + ":" +
			                 std::string(name, strnlen(name, (words - 3) * 4)));
		}
		i += words == 0 ? 1 : words;
	}
	return result;
}

constexpr const char* TestModule = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%block = OpTypeStruct %uint
%ptr_sb = OpTypePointer StorageBuffer %block
%out = OpVariable %ptr_sb StorageBuffer
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%ptr_fn_uint = OpTypePointer Function %uint
%fn_uint = OpTypeFunction %uint %uint
%c0 = OpConstant %uint 0
%c7 = OpConstant %uint 7
%add = OpFunction %uint None %fn_uint
%x = OpFunctionParameter %uint
%add_label = OpLabel
%r = OpIAdd %uint %x %c7
OpReturnValue %r
OpFunctionEnd
%main = OpFunction %void None %fn
%label = OpLabel
%v = OpVariable %ptr_fn_uint Function
OpStore %v %c0
%a = OpLoad %uint %v
%b = OpFunctionCall %uint %add %a
%b2 = OpFunctionCall %uint %add %a
%s = OpIAdd %uint %b %b2
%p = OpAccessChain %ptr_sb_uint %out %c0
OpStore %p %s
OpReturn
OpFunctionEnd
)";

int RunUnitTest() {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_1);
	std::vector<uint32_t> spirv;
	Check(tools.Assemble(TestModule, &spirv), "assemble the test module");
	Check(tools.Validate(spirv), "the test module is valid");
	const auto entry_points = EntryPoints(spirv);
	const auto before       = CountInstructions(spirv);

	for (const uint32_t level: {1u, 2u}) {
		auto optimized = spirv;
		Check(OptimizeSpirv(optimized, level, false), "the optimizer accepts the module");
		Check(tools.Validate(optimized), "the optimized module is valid");
		Check(EntryPoints(optimized) == entry_points, "same entry points");
		Check(CountInstructions(optimized, 57u /* OpFunctionCall */) == 0, "calls inlined");
		Check(CountInstructions(optimized, 61u /* OpLoad */) == 0, "the local load is gone");
		Check(CountInstructions(optimized) < before, "fewer instructions");
		Check(optimized[1] == spirv[1], "same SPIR-V version");
	}

	std::vector<uint32_t> broken(spirv.begin(), spirv.begin() + 7);
	const auto            copy = broken;
	Check(!OptimizeSpirv(broken, 2, false), "a broken module is refused");
	Check(broken == copy, "a refused module is left unchanged");

	const auto& stats = GetSpirvOptStats();
	Check(stats.modules == 3 && stats.optimizer_failures + stats.validation_failures == 1,
	      "the counters see three modules and one failure");
	std::printf("%s\n", g_failures == 0 ? "spirv_optimizer: OK" : "spirv_optimizer: FAILED");
	return g_failures == 0 ? 0 : 1;
}

struct ModuleResult {
	uint32_t stage  = 0;
	uint32_t before = 0;
	uint32_t after  = 0;
	double   ms     = 0;
	bool     input_valid = true;
	bool     ok     = true;
	uint64_t id     = 0;
};

const char* StageName(uint32_t model) {
	switch (model) {
		case 0: return "vs";
		case 1: return "hs";
		case 2: return "ds";
		case 3: return "gs";
		case 4: return "ps";
		case 5: return "cs";
		case 5364: return "ts";
		case 5365: return "ms";
		default: return "?";
	}
}

uint32_t Stage(const std::vector<uint32_t>& spirv) {
	for (size_t i = 5; i < spirv.size();) {
		const uint32_t words = spirv[i] >> 16u;
		if ((spirv[i] & 0xffffu) == 15u && words > 1) {
			return spirv[i + 1];
		}
		i += words == 0 ? 1 : words;
	}
	return UINT32_MAX;
}

double Median(std::vector<double> values) {
	if (values.empty()) {
		return 0;
	}
	std::ranges::sort(values);
	return values[values.size() / 2];
}

// The SPIR-V records of a store file: its header (magic, format version, key size, key), then
// 32-byte frame headers (compressed size, raw size, checksum, id, launch, type, uses) each followed
// by a zstd frame; type 3 holds a module's SPIR-V words.
int MeasureStore(const char* path) {
	std::ifstream        file(path, std::ios::binary);
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.size() < 16 || std::memcmp(bytes.data(), "KYSHSTOR", 8) != 0) {
		std::printf("%s: not a shader store\n", path);
		return 1;
	}
	uint32_t key_size = 0;
	std::memcpy(&key_size, bytes.data() + 12, 4);
	size_t                       offset = 16 + size_t {key_size};
	std::unordered_set<uint64_t> seen;
	std::vector<ModuleResult>    results;
	spvtools::SpirvTools         tools(SPV_ENV_VULKAN_1_3);
	spvtools::ValidatorOptions   validator;
	validator.SetAllowOffsetTextureOperand(true);
	while (bytes.size() - offset >= 32) {
		uint32_t compressed = 0, raw = 0;
		uint64_t id = 0;
		std::memcpy(&compressed, bytes.data() + offset, 4);
		std::memcpy(&raw, bytes.data() + offset + 4, 4);
		std::memcpy(&id, bytes.data() + offset + 16, 8);
		const uint8_t type = bytes[offset + 28];
		if (bytes.size() - offset - 32 < compressed) {
			break;
		}
		const uint8_t* data = bytes.data() + offset + 32;
		offset += 32 + size_t {compressed};
		if (type != 3 || raw == 0 || raw % 4 != 0 || !seen.insert(id).second) {
			continue;
		}
		std::vector<uint32_t> spirv(raw / 4);
		if (ZSTD_decompress(spirv.data(), raw, data, compressed) != raw) {
			continue;
		}
		ModuleResult result {.stage = Stage(spirv), .before = CountInstructions(spirv), .id = id};
		result.input_valid = tools.Validate(spirv.data(), spirv.size(), validator);
		const auto begin   = std::chrono::steady_clock::now();
		result.ok          = OptimizeSpirv(spirv, 2, true);
		result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		result.after = CountInstructions(spirv);
		results.push_back(result);
	}

	uint64_t            before = 0, after = 0;
	uint32_t            invalid_inputs = 0, failures = 0;
	double              total_ms = 0, max_ms = 0;
	std::vector<double> ratios, times;
	for (const auto& r: results) {
		before += r.before;
		after += r.after;
		invalid_inputs += r.input_valid ? 0u : 1u;
		failures += r.ok ? 0u : 1u;
		total_ms += r.ms;
		max_ms = std::max(max_ms, r.ms);
		ratios.push_back(r.before != 0 ? static_cast<double>(r.after) / r.before : 1.0);
		times.push_back(r.ms);
	}
	const auto& stats = GetSpirvOptStats();
	std::printf("== %s\nmodules=%zu insts %llu -> %llu (%.2f%%) median_module_ratio=%.4f invalid_inputs=%u "
	            "refused=%u (optimizer=%llu validation=%llu, totals so far)\n",
	            path, results.size(), static_cast<unsigned long long>(before),
	            static_cast<unsigned long long>(after),
	            before != 0 ? 100.0 * (static_cast<double>(after) - before) / before : 0.0, Median(ratios),
	            invalid_inputs, failures, static_cast<unsigned long long>(stats.optimizer_failures.load()),
	            static_cast<unsigned long long>(stats.validation_failures.load()));
	std::printf("optimizer ms: total=%.0f median=%.2f max=%.1f\n", total_ms, Median(times), max_ms);
	for (const uint32_t stage: {0u, 4u, 5u}) {
		uint64_t b = 0, a = 0, n = 0;
		for (const auto& r: results) {
			if (r.stage == stage) {
				b += r.before;
				a += r.after;
				n++;
			}
		}
		std::printf("  %s: modules=%llu insts %llu -> %llu (%.2f%%)\n", StageName(stage),
		            static_cast<unsigned long long>(n), static_cast<unsigned long long>(b),
		            static_cast<unsigned long long>(a), b != 0 ? 100.0 * (static_cast<double>(a) - b) / b : 0.0);
	}
	auto top = results;
	std::ranges::sort(top, [](const auto& x, const auto& y) { return x.before > y.before; });
	top.resize(std::min<size_t>(top.size(), 20));
	std::printf("top 20 by instructions: id stage before after change ms ok\n");
	for (const auto& r: top) {
		std::printf("  %016llx %s %u %u %.2f%% %.1f %d\n", static_cast<unsigned long long>(r.id),
		            StageName(r.stage), r.before, r.after,
		            100.0 * (static_cast<double>(r.after) - r.before) / r.before, r.ms, r.ok ? 1 : 0);
	}
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	if (argc == 1) {
		return RunUnitTest();
	}
	int result = 0;
	for (int i = 1; i < argc; i++) {
		result |= MeasureStore(argv[i]);
	}
	return result;
}
