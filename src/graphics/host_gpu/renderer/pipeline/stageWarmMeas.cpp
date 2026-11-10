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
#include <filesystem>
#include <mutex>
#include <spirv-tools/libspirv.hpp>
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

struct CapConfig {
	uint32_t workers;
	uint32_t spinners;
	bool     above;
};

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

	std::vector<long long>   wall(count), cpu(count);
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
				const auto              begin = std::chrono::steady_clock::now();
				CreatePipelineInternal(graphics, pipeline, job.rendering, job.vertex_input, job.static_params, stages,
				                       &pixel, nullptr);
				wall[k]  = Since(begin);
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
	std::snprintf(name, sizeof(name), "w%us%u%s", cfg.workers, cfg.spinners, cfg.above ? "a" : "n");
	long long              sum_wall = 0, sum_cpu = 0;
	std::vector<long long> ratio_permille(count);
	for (uint32_t k = 0; k < count; k++) {
		std::fprintf(out, "CCAPJ cfg=%s round=%u k=%u w=%u wall=%lld cpu=%lld\n", name, round, k, owner[k], wall[k],
		             cpu[k]);
		sum_wall += wall[k];
		sum_cpu += cpu[k];
		ratio_permille[k] = wall[k] > 0 ? cpu[k] * 1000 / wall[k] : 0;
	}
	std::fprintf(out,
	             "CCAP cfg=%s round=%u workers=%u spinners=%u above=%d count=%u span_ms=%lld per_s=%.2f wall_med=%lld"
	             " wall_p10=%lld wall_p90=%lld wall_mean=%lld cpu_med=%lld cpu_mean=%lld cpu_wall=%.3f"
	             " ratio_med=%.3f proc_cores=%.2f base_cores=%.2f\n",
	             name, round, cfg.workers, cfg.spinners, cfg.above ? 1 : 0, count, span_us / 1000,
	             static_cast<double>(count) * 1e6 / static_cast<double>(span_us), Percentile(wall, 0.5) / 1000,
	             Percentile(wall, 0.1) / 1000, Percentile(wall, 0.9) / 1000, sum_wall / count / 1000,
	             Percentile(cpu, 0.5) / 1000, sum_cpu / count / 1000,
	             static_cast<double>(sum_cpu) / static_cast<double>(std::max(1LL, sum_wall)),
	             static_cast<double>(Percentile(ratio_permille, 0.5)) / 1000.0, cores, base_cores);
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
	if (const char* cap = std::getenv("KYTY_MEAS_CCAP"); cap != nullptr && *cap == '1' && !jobs.empty()) {
		FILE* out = std::fopen((Dir() / "meas.txt").string().c_str(), "w");
		EXIT_IF(out == nullptr);
		RunCap(*graphics, jobs, out);
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
	g_jobs.push_back({rendering, vertex_input, static_params, vertex, pixel, vs->second.second, ps->second.second,
	                  vs->second.first, ps->second.first, real_us, at_us});
	if (!g_started) {
		g_started = true;
		std::thread(Run, &graphics).detach();
	}
}

} // namespace Libs::Graphics
