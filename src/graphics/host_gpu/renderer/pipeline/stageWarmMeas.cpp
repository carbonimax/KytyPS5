// Local measurement (step 5.19 gate), not product code. KYTY_MEAS_STAGE_WARM=<dir>: each graphics
// pipeline whose real creation missed the driver cache while <dir>/from exists (and <dir>/go does
// not) is queued; once <dir>/go exists one thread replays up to KYTY_MEAS_STAGE_WARM_MAX (200) of
// them, spread over the queue, with copies of their modules made unique by an OpSource (the
// driver's caches cannot hit them) and no pipeline cache:
//   a   new VS + new PS (cold)
//   bw  stub VS (writes the PS inputs) + PS_b, then b = new VS + PS_b (same PS handle)
//   b2w stub VS (the real VS outputs) + PS_b2, then b2 = new VS + PS_b2
//   cw  VS_c + stub PS, then c = VS_c + new PS (VS known, PS new: the usual game miss)
//   ew  stub VS + PS_e, then e = VS_c + PS_e (VS known, PS warmed by a stub only)
//   d   VS_c + PS_b (both seen with real partners, new pairing)
//   rep VS_c + PS_c again (exact repeat)
// a, b, b2 and c run in a rotating order. One MEASSW line per key in <dir>/meas.txt, then <dir>/done.
#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/stubStageShader.h"

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
	std::string messages;
	const char* failed = !Valid(stub_vs, messages) ? "stub_vs" : !Valid(stub_vs_like, messages) ? "stub_vs_like"
	                                                       : !Valid(stub_ps, messages)        ? "stub_ps"
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

	long long        a = -1, bw = -1, b = -1, b2w = -1, b2 = -1, cw = -1, c = -1;
	vk::ShaderModule ps_b = nullptr, vs_c = nullptr, ps_c = nullptr;
	for (uint32_t step = 0; step < 4; step++) {
		switch ((index + step) % 4u) {
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
	             " real=%lld a=%lld bw=%lld b=%lld b2w=%lld b2=%lld cw=%lld c=%lld ew=%lld e=%lld d=%lld rep=%lld"
	             " order=%u colors=%u depth=%d samples=%u vs_words=%zu ps_words=%zu\n",
	             index, total, job.at_us, job.vertex_hash, job.pixel_hash, job.real_us, a, bw, b, b2w, b2, cw, c, ew,
	             e, d, rep, index % 4u, job.rendering.color_count,
	             job.rendering.depth_format != vk::Format::eUndefined ? 1 : 0,
	             static_cast<uint32_t>(job.static_params.samples), job.vertex_spirv.size(), job.pixel_spirv.size());
	std::fflush(out);
}

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
