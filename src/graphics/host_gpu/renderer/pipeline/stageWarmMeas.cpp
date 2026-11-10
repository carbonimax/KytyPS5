// Local measurement (step 5.19 gate), not product code. KYTY_MEAS_STAGE_WARM=<dir>: each graphics
// pipeline whose real creation missed the driver cache while <dir>/from exists (and <dir>/go does
// not) is queued; once <dir>/go exists one thread replays up to KYTY_MEAS_STAGE_WARM_MAX (200) of
// them, spread over the queue, with copies of their modules made unique by an OpSource (the
// driver's caches cannot hit them) and no pipeline cache:
//   a   new VS + new PS (cold)
//   bw  stub VS (writes the PS inputs) + PS_b, then b = new VS + PS_b (same PS handle)
//   b2w stub VS (the real VS outputs) + PS_b2, then b2 = new VS + PS_b2
//   b3w stub VS writing the vertex index (no constant output) + PS_b3, then b3 = new VS + PS_b3
//   cw  VS_c + stub PS, then c = VS_c + new PS (VS known, PS new: the usual game miss)
//   ew  stub VS + PS_e, then e = VS_c + PS_e (VS known, PS warmed by a stub only)
//   d   VS_c + PS_b (both seen with real partners, new pairing)
//   rep VS_c + PS_c again (exact repeat)
// a, b, b2, b3 and c run in a rotating order. One MEASSW line per key in <dir>/meas.txt, then <dir>/done.
// KYTY_MEAS_CCAP=1 (step 5.20): throughput against workers, spinners, priority. KYTY_MEAS_CSHAPE=1 (step 5.21):
// creation cost against module size and content on one thread, and pinned workers on distinct cores vs SMT pairs.
#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/stubStageShader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {

namespace {

namespace fs = std::filesystem;

struct MeasJob {
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
	ShaderModule             vertex;
	ShaderModule             pixel;
	std::vector<uint32_t>    vertex_spirv;
	std::vector<uint32_t>    pixel_spirv;
	std::vector<uint32_t>    vertex_alt; // KYTY_MEAS_CSHAPE: the VS translated without the optional IR passes
	uint64_t                 vertex_hash = 0;
	uint64_t                 pixel_hash  = 0;
	long long                real_us     = 0;
	uint64_t                 at_us       = 0;
};

fs::path& Dir() {
	static fs::path dir;
	return dir;
}

bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_MEAS_STAGE_WARM");
		if (value == nullptr || *value == '\0') {
			return false;
		}
		Dir() = value;
		return true;
	}();
	return on;
}

thread_local bool t_meas_thread = false;

std::mutex                                                                     g_mutex;
std::unordered_map<VkShaderModule, std::pair<uint64_t, std::vector<uint32_t>>> g_spirv;
std::unordered_map<VkShaderModule, std::vector<uint32_t>>                       g_alt;
thread_local std::vector<uint32_t>                                              t_pending_alt;
std::vector<MeasJob>                                                            g_jobs;
bool                                                                           g_started = false;

bool Exists(const char* name) {
	std::error_code error;
	return fs::exists(Dir() / name, error);
}

long long Since(std::chrono::steady_clock::time_point begin) {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin)
	    .count();
}

uint32_t Salt() {
	static std::atomic<uint32_t> next {
	    static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())};
	return next.fetch_add(7919u);
}

// A copy of `spirv` with a unique OpSource after the debug-source section.
vk::ShaderModule Unique(GraphicContext& graphics, std::vector<uint32_t> spirv) {
	size_t at = 5;
	while (at < spirv.size()) {
		const auto op = spirv[at] & 0xffffu;
		if (op != 17 && op != 10 && op != 11 && op != 14 && op != 15 && op != 16 && op != 331 && op != 7 &&
		    op != 4 && op != 3 && op != 2) {
			break;
		}
		at += spirv[at] >> 16u;
	}
	const uint32_t source[3] = {(3u << 16u) | 3u, 0u, Salt()};
	spirv.insert(spirv.begin() + static_cast<std::ptrdiff_t>(at), source, source + 3);
	return CompileSPV(spirv, graphics.device);
}

bool Valid(const std::vector<uint32_t>& spirv, std::string& messages) {
	if (spirv.empty()) {
		messages = "unsupported interface";
		return false;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
	                                     const char* message) {
		if (messages.empty()) messages = message;
	});
	return tools.Validate(spirv.data(), spirv.size(), spvtools::ValidatorOptions {});
}

void MeasureOne(GraphicContext& graphics, MeasJob& job, uint32_t index, uint32_t total, FILE* out) {
	const auto stub_vs      = BuildStubVertexForPixel(job.pixel_spirv);
	const auto stub_vs_like = BuildStubVertexLike(job.vertex_spirv);
	const auto stub_ps      = BuildStubPixelLike(job.pixel_spirv);
	const auto stub_vs_var  = BuildStubVaryingVertexForPixel(job.pixel_spirv);
	std::string messages;
	const char* failed = !Valid(stub_vs, messages) ? "stub_vs" : !Valid(stub_vs_like, messages) ? "stub_vs_like"
	                                                       : !Valid(stub_ps, messages)        ? "stub_ps"
	                                                       : !Valid(stub_vs_var, messages)    ? "stub_vs_var"
	                                                                                         : nullptr;
	if (failed != nullptr) {
		std::fprintf(out, "MEASSW_SKIP i=%u vs=%016" PRIx64 " ps=%016" PRIx64 " %s: %s\n", index, job.vertex_hash,
		             job.pixel_hash, failed, messages.c_str());
		std::fflush(out);
		return;
	}
	ShaderModule      vertex = job.vertex;
	ShaderModule      pixel  = job.pixel;
	const auto        create = [&](vk::ShaderModule vs, vk::ShaderModule ps) {
        vertex.handle.module = vs;
        pixel.handle.module  = ps;
        const ShaderModule*     stages[1] = {&vertex};
        PipelineCache::Pipeline pipeline {};
        const auto              begin = std::chrono::steady_clock::now();
        CreatePipelineInternal(graphics, pipeline, job.rendering, job.vertex_input, job.static_params, stages,
                               &pixel, nullptr);
        const auto us = Since(begin);
        graphics.device.destroyPipeline(pipeline.pipeline);
        graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
        graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
        return us;
	};
	const auto unique  = [&](const std::vector<uint32_t>& spirv) { return Unique(graphics, spirv); };
	const auto destroy = [&](vk::ShaderModule module) { graphics.device.destroyShaderModule(module); };

	long long        a = -1, bw = -1, b = -1, b2w = -1, b2 = -1, b3w = -1, b3 = -1, cw = -1, c = -1;
	vk::ShaderModule ps_b = nullptr, vs_c = nullptr, ps_c = nullptr;
	for (uint32_t step = 0; step < 5; step++) {
		switch ((index + step) % 5u) {
			case 0: {
				const auto vs = unique(job.vertex_spirv), ps = unique(job.pixel_spirv);
				a = create(vs, ps);
				destroy(vs);
				destroy(ps);
				break;
			}
			case 1: {
				ps_b           = unique(job.pixel_spirv);
				const auto sv  = unique(stub_vs);
				bw             = create(sv, ps_b);
				const auto vs  = unique(job.vertex_spirv);
				b              = create(vs, ps_b);
				destroy(sv);
				destroy(vs);
				break;
			}
			case 2: {
				const auto ps = unique(job.pixel_spirv), sl = unique(stub_vs_like);
				b2w           = create(sl, ps);
				const auto vs = unique(job.vertex_spirv);
				b2            = create(vs, ps);
				destroy(ps);
				destroy(sl);
				destroy(vs);
				break;
			}
			case 3: {
				const auto ps = unique(job.pixel_spirv), sv3 = unique(stub_vs_var);
				b3w           = create(sv3, ps);
				const auto vs = unique(job.vertex_spirv);
				b3            = create(vs, ps);
				destroy(ps);
				destroy(sv3);
				destroy(vs);
				break;
			}
			default: {
				vs_c          = unique(job.vertex_spirv);
				const auto sp = unique(stub_ps);
				cw            = create(vs_c, sp);
				ps_c          = unique(job.pixel_spirv);
				c             = create(vs_c, ps_c);
				destroy(sp);
				break;
			}
		}
	}
	const auto ps_e = unique(job.pixel_spirv), sv = unique(stub_vs);
	const auto ew   = create(sv, ps_e);
	const auto e    = create(vs_c, ps_e);
	const auto d    = create(vs_c, ps_b);
	const auto rep  = create(vs_c, ps_c);
	for (const auto module: {ps_e, sv, ps_b, vs_c, ps_c}) {
		destroy(module);
	}
	std::fprintf(out,
	             "MEASSW i=%u n=%u at_us=%" PRIu64 " vs=%016" PRIx64 " ps=%016" PRIx64
	             " real=%lld a=%lld bw=%lld b=%lld b2w=%lld b2=%lld b3w=%lld b3=%lld cw=%lld c=%lld ew=%lld e=%lld d=%lld rep=%lld"
	             " order=%u colors=%u depth=%d samples=%u vs_words=%zu ps_words=%zu\n",
	             index, total, job.at_us, job.vertex_hash, job.pixel_hash, job.real_us, a, bw, b, b2w, b2, b3w, b3, cw, c, ew,
	             e, d, rep, index % 5u, job.rendering.color_count,
	             job.rendering.depth_format != vk::Format::eUndefined ? 1 : 0,
	             static_cast<uint32_t>(job.static_params.samples), job.vertex_spirv.size(), job.pixel_spirv.size());
	std::fflush(out);
}

#if defined(_WIN32)
// Step 5.20 (KYTY_MEAS_CCAP=1): cold creation throughput. Each config creates `count` cold pipelines
// (variant a: new VS + new PS, salted) with `workers` threads at normal or above-normal priority while
// `spinners` threads spin on pause (guest job threads in a stall). Per creation: wall and thread CPU;
// per config: process CPU over the config and over a 3 s idle baseline before it.
long long ThreadCpuUs() {
	FILETIME c, e, k, u;
	GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u);
	return static_cast<long long>(((static_cast<uint64_t>(k.dwHighDateTime) << 32u) | k.dwLowDateTime) +
	                              ((static_cast<uint64_t>(u.dwHighDateTime) << 32u) | u.dwLowDateTime)) /
	       10;
}

long long ProcessCpuUs() {
	FILETIME c, e, k, u;
	GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
	return static_cast<long long>(((static_cast<uint64_t>(k.dwHighDateTime) << 32u) | k.dwLowDateTime) +
	                              ((static_cast<uint64_t>(u.dwHighDateTime) << 32u) | u.dwLowDateTime)) /
	       10;
}

long long ThreadCycles() {
	ULONG64 cycles = 0;
	QueryThreadCycleTime(GetCurrentThread(), &cycles);
	return static_cast<long long>(cycles);
}

// Logical-CPU masks of each physical core of processor group 0, in core order.
const std::vector<std::vector<DWORD_PTR>>& Cores() {
	static const auto cores = [] {
		std::vector<std::vector<DWORD_PTR>> result;
		DWORD                               length = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
		std::vector<uint8_t> buffer(length);
		auto*                info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
		if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
			return result;
		}
		for (DWORD at = 0; at < length;) {
			const auto* item = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
			if (item->Processor.GroupMask[0].Group == 0) {
				std::vector<DWORD_PTR> logical;
				for (DWORD_PTR bit = 1; bit != 0; bit <<= 1u) {
					if ((item->Processor.GroupMask[0].Mask & bit) != 0) logical.push_back(bit);
				}
				result.push_back(logical);
			}
			at += item->Size;
		}
		return result;
	}();
	return cores;
}

struct CapConfig {
	uint32_t workers;
	uint32_t spinners;
	bool     above;
	// 0: no affinity; 'P': worker w alone on core C-n+w; 'S': workers 2j, 2j+1 on both logical CPUs of core
	// C-n/2+j (C cores, n workers).
	char     pin = 0;
};

// Affinity of worker `w` under `cfg`, or 0 (none / not possible).
DWORD_PTR PinMask(const CapConfig& cfg, uint32_t w) {
	const auto& cores = Cores();
	const auto  n     = cfg.workers;
	if (cfg.pin == 'P' && n <= cores.size()) {
		return cores[cores.size() - n + w][0];
	}
	if (cfg.pin == 'S' && n % 2 == 0 && n / 2 <= cores.size()) {
		const auto& core = cores[cores.size() - n / 2 + w / 2];
		return core.size() >= 2 ? core[w % 2] : 0;
	}
	return 0;
}

long long Percentile(std::vector<long long> v, double p) {
	std::sort(v.begin(), v.end());
	return v.empty() ? -1 : v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

void RunCapConfig(GraphicContext& graphics, std::vector<MeasJob>& jobs, const CapConfig& cfg, uint32_t count,
                  uint32_t round, FILE* out) {
	auto p0 = ProcessCpuUs();
	auto t0 = std::chrono::steady_clock::now();
	std::this_thread::sleep_for(std::chrono::seconds(3));
	const double base_cores = static_cast<double>(ProcessCpuUs() - p0) / static_cast<double>(Since(t0));

	std::atomic<bool>        stop {false};
	std::vector<std::thread> spinners;
	for (uint32_t i = 0; i < cfg.spinners; i++) {
		spinners.emplace_back([&stop] {
			while (!stop.load(std::memory_order_relaxed)) {
				YieldProcessor();
			}
		});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	std::vector<long long>   wall(count), cpu(count), cyc(count);
	std::vector<uint32_t>    owner(count);
	std::atomic<uint32_t>    next {0};
	std::vector<std::thread> workers;
	p0 = ProcessCpuUs();
	t0 = std::chrono::steady_clock::now();
	for (uint32_t w = 0; w < cfg.workers; w++) {
		workers.emplace_back([&, w] {
			t_meas_thread = true;
			SetPipelineCreationQuiet(true);
			if (cfg.above) {
				SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
			}
			if (const auto mask = PinMask(cfg, w); mask != 0) {
				SetThreadAffinityMask(GetCurrentThread(), mask);
			}
			for (uint32_t k = next.fetch_add(1); k < count; k = next.fetch_add(1)) {
				auto& job = jobs[static_cast<size_t>(k) * jobs.size() / count];
				ShaderModule vertex = job.vertex;
				ShaderModule pixel  = job.pixel;
				// Module creation stays outside the timed region, as in 5.19.
				vertex.handle.module = Unique(graphics, job.vertex_spirv);
				pixel.handle.module  = Unique(graphics, job.pixel_spirv);
				const ShaderModule*     stages[1] = {&vertex};
				PipelineCache::Pipeline pipeline {};
				const auto              c0    = ThreadCpuUs();
				const auto              y0    = ThreadCycles();
				const auto              begin = std::chrono::steady_clock::now();
				CreatePipelineInternal(graphics, pipeline, job.rendering, job.vertex_input, job.static_params, stages,
				                       &pixel, nullptr);
				wall[k]  = Since(begin);
				cyc[k]   = ThreadCycles() - y0;
				cpu[k]   = ThreadCpuUs() - c0;
				owner[k] = w;
				graphics.device.destroyPipeline(pipeline.pipeline);
				graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
				graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
				graphics.device.destroyShaderModule(vertex.handle.module);
				graphics.device.destroyShaderModule(pixel.handle.module);
			}
		});
	}
	for (auto& t: workers) {
		t.join();
	}
	const auto   span_us = Since(t0);
	const double cores   = static_cast<double>(ProcessCpuUs() - p0) / static_cast<double>(span_us);
	stop = true;
	for (auto& t: spinners) {
		t.join();
	}
	char name[32];
	if (cfg.pin != 0) {
		std::snprintf(name, sizeof(name), "%c%u", cfg.pin, cfg.workers);
	} else {
		std::snprintf(name, sizeof(name), "w%us%u%s", cfg.workers, cfg.spinners, cfg.above ? "a" : "n");
	}
	long long              sum_wall = 0, sum_cpu = 0;
	std::vector<long long> ratio_permille(count);
	for (uint32_t k = 0; k < count; k++) {
		std::fprintf(out, "CCAPJ cfg=%s round=%u k=%u w=%u wall=%lld cpu=%lld kcyc=%lld\n", name, round, k, owner[k],
		             wall[k], cpu[k], cyc[k] / 1000);
		sum_wall += wall[k];
		sum_cpu += cpu[k];
		ratio_permille[k] = wall[k] > 0 ? cpu[k] * 1000 / wall[k] : 0;
	}
	std::fprintf(out,
	             "CCAP cfg=%s round=%u workers=%u spinners=%u above=%d count=%u span_ms=%lld per_s=%.2f wall_med=%lld"
	             " wall_p10=%lld wall_p90=%lld wall_mean=%lld cpu_med=%lld cpu_mean=%lld cpu_wall=%.3f"
	             " ratio_med=%.3f proc_cores=%.2f base_cores=%.2f mcyc_med=%lld\n",
	             name, round, cfg.workers, cfg.spinners, cfg.above ? 1 : 0, count, span_us / 1000,
	             static_cast<double>(count) * 1e6 / static_cast<double>(span_us), Percentile(wall, 0.5) / 1000,
	             Percentile(wall, 0.1) / 1000, Percentile(wall, 0.9) / 1000, sum_wall / count / 1000,
	             Percentile(cpu, 0.5) / 1000, sum_cpu / count / 1000,
	             static_cast<double>(sum_cpu) / static_cast<double>(std::max(1LL, sum_wall)),
	             static_cast<double>(Percentile(ratio_permille, 0.5)) / 1000.0, cores, base_cores,
	             Percentile(cyc, 0.5) / 1000000);
	std::fflush(out);
}

void RunCap(GraphicContext& graphics, std::vector<MeasJob>& jobs, FILE* out) {
	// KYTY_MEAS_CCAP_LIST: workers:spinners:n|a,... (default below), run in order then reversed.
	std::vector<CapConfig> list;
	const char*            text = std::getenv("KYTY_MEAS_CCAP_LIST");
	std::string            spec = text != nullptr && *text != '\0'
	                                  ? text
	                                  : "1:0:n,2:0:n,4:0:n,8:0:n,12:0:n,1:6:n,8:6:n,8:6:a,8:3:n,12:6:n,12:6:a";
	for (size_t at = 0; at < spec.size();) {
		size_t   end = spec.find(',', at);
		unsigned w = 0, s = 0;
		char     p = 'n';
		if (std::sscanf(spec.substr(at, end - at).c_str(), "%u:%u:%c", &w, &s, &p) == 3 && w > 0) {
			list.push_back({w, s, p == 'a'});
		}
		at = end == std::string::npos ? spec.size() : end + 1;
	}
	const char*    count_text = std::getenv("KYTY_MEAS_CCAP_COUNT");
	const uint32_t count      = count_text != nullptr ? static_cast<uint32_t>(std::atoi(count_text)) : 96u;
	std::fprintf(out, "# ccap keys=%zu count=%u configs=%zu\n", jobs.size(), count, list.size());
	std::fflush(out);
	for (uint32_t round = 0; round < 2; round++) {
		for (size_t i = 0; i < list.size(); i++) {
			RunCapConfig(graphics, jobs, list[round == 0 ? i : list.size() - 1 - i], count, round, out);
		}
	}
}

// Step 5.21 (KYTY_MEAS_CSHAPE=1). Opcode histogram of a module, "op:count,...".
std::string Mix(const std::vector<uint32_t>& spirv) {
	std::map<uint32_t, uint32_t> counts;
	for (size_t at = 5; at < spirv.size();) {
		const auto length = spirv[at] >> 16u;
		counts[spirv[at] & 0xffffu]++;
		if (length == 0) break;
		at += length;
	}
	std::string text;
	char        item[24];
	for (const auto& [op, n]: counts) {
		std::snprintf(item, sizeof(item), "%s%u:%u", text.empty() ? "" : ",", op, n);
		text += item;
	}
	return text;
}

// Descriptor (set, binding) pairs, Locations and BuiltIns a module declares.
std::set<uint64_t> InterfaceOf(const std::vector<uint32_t>& spirv) {
	std::unordered_map<uint32_t, uint32_t> sets, bindings;
	std::set<uint64_t>                     result;
	for (size_t at = 5; at < spirv.size();) {
		const auto length = spirv[at] >> 16u;
		const auto op     = spirv[at] & 0xffffu;
		if (length == 0 || at + length > spirv.size()) break;
		if (op == 71 && length >= 4) { // OpDecorate target decoration literal
			const auto target = spirv[at + 1], decoration = spirv[at + 2], literal = spirv[at + 3];
			if (decoration == 34) sets[target] = literal;
			if (decoration == 33) bindings[target] = literal;
			if (decoration == 30) result.insert((1ull << 62u) | literal);
			if (decoration == 11) result.insert((2ull << 62u) | literal);
		} else if (op == 72 && length >= 5 && spirv[at + 3] == 11) { // OpMemberDecorate ... BuiltIn
			result.insert((3ull << 62u) | spirv[at + 4]);
		}
		at += length;
	}
	for (const auto& [target, binding]: bindings) {
		const auto set = sets.find(target);
		result.insert((static_cast<uint64_t>(set != sets.end() ? set->second : 0u) << 32u) | binding);
	}
	return result;
}

// CSE and its clean-up (the 5.16 list) or exhaustive inlining; validated, empty on failure.
std::vector<uint32_t> Transform(const std::vector<uint32_t>& spirv, bool inline_all) {
	spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_3);
	if (inline_all) {
		optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass())
		    .RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
	} else {
		optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass())
		    .RegisterPass(spvtools::CreateDeadBranchElimPass())
		    .RegisterPass(spvtools::CreateBlockMergePass())
		    .RegisterPass(spvtools::CreateVectorDCEPass())
		    .RegisterPass(spvtools::CreateDeadInsertElimPass())
		    .RegisterPass(spvtools::CreateAggressiveDCEPass())
		    .RegisterPass(spvtools::CreateCFGCleanupPass());
	}
	spvtools::OptimizerOptions options;
	options.set_run_validator(false);
	options.set_preserve_bindings(true);
	options.set_preserve_spec_constants(true);
	std::vector<uint32_t> result;
	if (!optimizer.Run(spirv.data(), spirv.size(), &result, options)) {
		return {};
	}
	spvtools::SpirvTools       tools(SPV_ENV_VULKAN_1_3);
	spvtools::ValidatorOptions validator;
	validator.SetAllowOffsetTextureOperand(true);
	return tools.Validate(result.data(), result.size(), validator) ? result : std::vector<uint32_t> {};
}

struct ArmTime {
	long long wall = -1, kcyc = -1, cpu = -1, module_us = -1;
};

// One cold creation (salted modules, no pipeline cache) on this thread.
ArmTime TimeArm(GraphicContext& graphics, const MeasJob& job, const std::vector<uint32_t>& vs,
                const std::vector<uint32_t>& ps) {
	ArmTime      time;
	ShaderModule vertex = job.vertex;
	ShaderModule pixel  = job.pixel;
	const auto   m0     = std::chrono::steady_clock::now();
	vertex.handle.module = Unique(graphics, vs);
	pixel.handle.module  = Unique(graphics, ps);
	time.module_us       = Since(m0);
	const ShaderModule*     stages[1] = {&vertex};
	PipelineCache::Pipeline pipeline {};
	const auto              c0    = ThreadCpuUs();
	const auto              y0    = ThreadCycles();
	const auto              begin = std::chrono::steady_clock::now();
	CreatePipelineInternal(graphics, pipeline, job.rendering, job.vertex_input, job.static_params, stages, &pixel,
	                       nullptr);
	time.wall = Since(begin);
	time.kcyc = (ThreadCycles() - y0) / 1000;
	time.cpu  = ThreadCpuUs() - c0;
	graphics.device.destroyPipeline(pipeline.pipeline);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
	graphics.device.destroyShaderModule(vertex.handle.module);
	graphics.device.destroyShaderModule(pixel.handle.module);
	return time;
}

// Pinned workers (P = distinct physical cores, S = both logical CPUs of a core), in order or reversed.
void RunShapeSmt(GraphicContext& graphics, std::vector<MeasJob>& jobs, uint32_t round, FILE* out) {
	const char* count_text = std::getenv("KYTY_MEAS_CCAP_COUNT");
	const auto  count      = count_text != nullptr ? static_cast<uint32_t>(std::atoi(count_text)) : 96u;
	std::vector<CapConfig> list = {{1, 0, false, 'P'}, {2, 0, false, 'P'}, {2, 0, false, 'S'},
	                               {4, 0, false, 'P'}, {4, 0, false, 'S'}, {8, 0, false, 'P'}};
	for (size_t i = 0; i < list.size(); i++) {
		RunCapConfig(graphics, jobs, list[round == 0 ? i : list.size() - 1 - i], count, round, out);
	}
}

void RunShape(GraphicContext& graphics, std::vector<MeasJob>& jobs, FILE* out) {
	const auto& cores = Cores();
	std::fprintf(out, "# cshape keys=%zu cores=%zu logical0=%zu", jobs.size(), cores.size(),
	             cores.empty() ? 0 : cores[0].size());
	for (const auto& core: cores) {
		DWORD_PTR mask = 0;
		for (const auto bit: core) mask |= bit;
		std::fprintf(out, " %llx", static_cast<unsigned long long>(mask));
	}
	std::fprintf(out, "\n");
	std::fflush(out);
	const char* smt_text = std::getenv("KYTY_MEAS_CSHAPE_SMT");
	const bool  smt      = smt_text == nullptr || *smt_text != '0';
	if (smt) RunShapeSmt(graphics, jobs, 0, out);

	const char* max_text = std::getenv("KYTY_MEAS_CSHAPE_MAX");
	const auto  total    = static_cast<uint32_t>(
        std::min<size_t>(jobs.size(), max_text != nullptr ? static_cast<size_t>(std::atoi(max_text)) : jobs.size()));
	for (uint32_t k = 0; k < total; k++) {
		const auto& job     = jobs[static_cast<size_t>(k) * jobs.size() / total];
		const auto  o0      = std::chrono::steady_clock::now();
		const auto  vs_cse  = Transform(job.vertex_spirv, false);
		const auto  ps_cse  = Transform(job.pixel_spirv, false);
		const auto  vs_inl  = Transform(job.vertex_spirv, true);
		const auto  ps_inl  = Transform(job.pixel_spirv, true);
		const auto  opt_us  = Since(o0);
		const auto  stub_vs = BuildStubVertexForPixel(job.pixel_spirv);
		const auto  stub_ps = BuildStubPixelLike(job.pixel_spirv);
		std::string messages;
		const bool  floor_ok = Valid(stub_vs, messages) && Valid(stub_ps, messages);
		// The alt VS runs with the base VS's layout: only when it declares nothing the base does not.
		const char* alt_state = "none";
		if (!job.vertex_alt.empty()) {
			const auto base_if = InterfaceOf(job.vertex_spirv), alt_if = InterfaceOf(job.vertex_alt);
			alt_state = job.vertex_alt == job.vertex_spirv ? "same"
			            : !std::includes(base_if.begin(), base_if.end(), alt_if.begin(), alt_if.end()) ? "iface"
			                                                                                           : "ok";
		}
		const bool cse_ok = !vs_cse.empty() && !ps_cse.empty();
		const bool inl_ok = !vs_inl.empty() && !ps_inl.empty() &&
		                    (vs_inl.size() != job.vertex_spirv.size() || ps_inl.size() != job.pixel_spirv.size());
		const bool alt_ok = std::strcmp(alt_state, "ok") == 0;

		ArmTime    a0, cse, inl, alt, flr, a1;
		const auto p0   = ProcessCpuUs();
		const auto own0 = ThreadCpuUs();
		const auto t0   = std::chrono::steady_clock::now();
		a0              = TimeArm(graphics, job, job.vertex_spirv, job.pixel_spirv);
		for (uint32_t step = 0; step < 4; step++) {
			switch ((k + step) % 4u) {
				case 0:
					if (cse_ok) cse = TimeArm(graphics, job, vs_cse, ps_cse);
					break;
				case 1:
					if (inl_ok) inl = TimeArm(graphics, job, vs_inl, ps_inl);
					break;
				case 2:
					if (alt_ok) alt = TimeArm(graphics, job, job.vertex_alt, job.pixel_spirv);
					break;
				default:
					if (floor_ok) flr = TimeArm(graphics, job, stub_vs, stub_ps);
					break;
			}
		}
		a1                  = TimeArm(graphics, job, job.vertex_spirv, job.pixel_spirv);
		const auto   span   = Since(t0);
		const double others = static_cast<double>((ProcessCpuUs() - p0) - (ThreadCpuUs() - own0)) /
		                      static_cast<double>(std::max(1LL, span));
		const auto arm = [](const ArmTime& t) {
			char text[96];
			std::snprintf(text, sizeof(text), "%lld,%lld,%lld,%lld", t.wall, t.kcyc, t.cpu, t.module_us);
			return std::string(text);
		};
		const auto words = [](const std::vector<uint32_t>& v) { return v.empty() ? -1LL : static_cast<long long>(v.size()); };
		std::fprintf(out,
		             "MEASCS k=%u n=%u vs=%016" PRIx64 " ps=%016" PRIx64 " real=%lld order=%u colors=%u depth=%d"
		             " vs_w=%zu ps_w=%zu vs_cse=%lld ps_cse=%lld vs_inl=%lld ps_inl=%lld vs_alt=%lld alt=%s"
		             " stub_vs=%zu stub_ps=%zu opt_us=%lld others=%.2f"
		             " a0=%s cse=%s inl=%s altv=%s floor=%s a1=%s\n",
		             k, total, job.vertex_hash, job.pixel_hash, job.real_us, k % 4u, job.rendering.color_count,
		             job.rendering.depth_format != vk::Format::eUndefined ? 1 : 0, job.vertex_spirv.size(),
		             job.pixel_spirv.size(), words(vs_cse), words(ps_cse), words(vs_inl), words(ps_inl),
		             words(job.vertex_alt), alt_state, stub_vs.size(), stub_ps.size(), opt_us, others,
		             arm(a0).c_str(), arm(cse).c_str(), arm(inl).c_str(), arm(alt).c_str(), arm(flr).c_str(),
		             arm(a1).c_str());
		std::fprintf(out, "MIX k=%u st=vs %s\n", k, Mix(job.vertex_spirv).c_str());
		std::fprintf(out, "MIX k=%u st=ps %s\n", k, Mix(job.pixel_spirv).c_str());
		if (cse_ok) std::fprintf(out, "MIX k=%u st=ps_cse %s\n", k, Mix(ps_cse).c_str());
		std::fflush(out);
	}
	if (smt) RunShapeSmt(graphics, jobs, 1, out);
}
#endif

void Run(GraphicContext* graphics) {
	t_meas_thread = true;
	SetPipelineCreationQuiet(true);
	while (!Exists("go")) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	std::vector<MeasJob> jobs;
	{
		std::lock_guard lock(g_mutex);
		jobs.swap(g_jobs);
	}
#if defined(_WIN32)
	const char* cap   = std::getenv("KYTY_MEAS_CCAP");
	const char* shape = std::getenv("KYTY_MEAS_CSHAPE");
	const bool  ccap  = cap != nullptr && *cap == '1';
	const bool  cs    = shape != nullptr && *shape == '1';
	if ((ccap || cs) && !jobs.empty()) {
		FILE* out = std::fopen((Dir() / "meas.txt").string().c_str(), "w");
		EXIT_IF(out == nullptr);
		if (cs) {
			RunShape(*graphics, jobs, out);
		} else {
			RunCap(*graphics, jobs, out);
		}
		std::fprintf(out, "END\n");
		std::fclose(out);
		if (FILE* done = std::fopen((Dir() / "done").string().c_str(), "w")) {
			std::fclose(done);
		}
		return;
	}
#endif
	const char*    max_text = std::getenv("KYTY_MEAS_STAGE_WARM_MAX");
	const uint32_t max      = max_text != nullptr ? static_cast<uint32_t>(std::atoi(max_text)) : 200u;
	const auto     total    = static_cast<uint32_t>(std::min<size_t>(jobs.size(), max));
	FILE*          out      = std::fopen((Dir() / "meas.txt").string().c_str(), "w");
	EXIT_IF(out == nullptr);
	std::fprintf(out, "# queued=%zu measured=%u\n", jobs.size(), total);
	std::fflush(out);
	for (uint32_t i = 0; i < total; i++) {
		// Spread over the queue rather than the first misses of the window.
		MeasureOne(*graphics, jobs[static_cast<size_t>(i) * jobs.size() / total], i, total, out);
	}
	std::fprintf(out, "END\n");
	std::fclose(out);
	if (FILE* done = std::fopen((Dir() / "done").string().c_str(), "w")) {
		std::fclose(done);
	}
}

} // namespace

void MeasNoteSpirv(vk::ShaderModule module, uint64_t hash, const std::vector<uint32_t>& spirv) {
	if (!On()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	g_spirv[static_cast<VkShaderModule>(module)] = {hash, spirv};
	if (!t_pending_alt.empty()) {
		g_alt[static_cast<VkShaderModule>(module)] = std::move(t_pending_alt);
		t_pending_alt.clear();
	}
}

bool MeasShapeAltOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_MEAS_CSHAPE");
		return value != nullptr && *value == '1';
	}();
	return on && On();
}

void MeasSetAltSpirv(std::vector<uint32_t> spirv) {
	t_pending_alt = std::move(spirv);
}

bool MeasStageWarmOn() {
	return On() && !t_meas_thread;
}

void MeasStageWarmEnqueue(GraphicContext& graphics, const PipelineRenderingState& rendering,
                          const PipelineVertexInputState& vertex_input,
                          const PipelineStaticParameters& static_params, const ShaderModule& vertex,
                          const ShaderModule& pixel, long long real_us, uint64_t at_us) {
	if (!MeasStageWarmOn() || !Exists("from") || Exists("go")) {
		return;
	}
	std::lock_guard lock(g_mutex);
	const auto      vs = g_spirv.find(static_cast<VkShaderModule>(vertex.handle.module));
	const auto      ps = g_spirv.find(static_cast<VkShaderModule>(pixel.handle.module));
	if (vs == g_spirv.end() || ps == g_spirv.end()) {
		return;
	}
	const auto alt = g_alt.find(static_cast<VkShaderModule>(vertex.handle.module));
	g_jobs.push_back({rendering, vertex_input, static_params, vertex, pixel, vs->second.second, ps->second.second,
	                  alt != g_alt.end() ? alt->second : std::vector<uint32_t> {}, vs->second.first,
	                  ps->second.first, real_us, at_us});
	if (!g_started) {
		g_started = true;
		std::thread(Run, &graphics).detach();
	}
}

} // namespace Libs::Graphics
