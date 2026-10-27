#include "graphics/shader/shaderPretranslate.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/perfStats.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ProgramPack.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace Libs::Graphics::ShaderPretranslate {

namespace {

uint32_t EnvU32(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
	                                             : fallback;
}

// A fatal check during a speculative translation retires its worker (Common::SetFatalRetiresThread);
// after this many, pretranslation stops.
constexpr uint32_t MaxRetiredWorkers = 8;

uint32_t WorkerCount() {
	static const uint32_t count = std::min(EnvU32("KYTY_SHADER_PRETRANSLATE", 4), 16u);
	return count;
}

enum class State { Queued, Running, Ready, Dropped };

struct Job {
	uint64_t              hash  = 0;
	uint64_t              id    = 0; // XXH3 of the key words, seeded by the code hash
	uint64_t              order = 0; // registration order (eviction: oldest first)
	std::vector<uint32_t> key;
	std::vector<uint32_t> code;
	ShaderPixelInputInfo  info {};
	uint32_t              user_data_count = 0;
	State                 state           = State::Queued;
	size_t                bytes           = 0;
	uint32_t              refills         = 0; // background translations of this key after a use
	ShaderRecompiler::TranslateResult result; // result.program is empty when packed
	ShaderRecompiler::IR::PackedProgram packed;
};

struct Stats {
	std::atomic<uint64_t> registered {0}, unpredicted {0}, duplicates {0}, translated {0}, calls {0}, abandoned {0},
	    taken {0}, waited {0}, wait_us {0}, absent {0}, absent_registered {0}, absent_dropped {0}, key_mismatch {0},
	    queued_at_use {0}, remembered {0},
	    code_changed {0}, evicted {0}, refills {0}, worker_us {0}, verified {0}, verify_diffs {0}, packed {0},
	    unpacked_bytes {0}, packed_bytes {0}, pack_us {0}, unpack_us {0};
};

struct Pool {
	std::mutex                                     mutex;
	std::condition_variable                        work;
	std::condition_variable                        done;
	std::unordered_map<uint64_t, std::shared_ptr<Job>> jobs;
	std::deque<std::shared_ptr<Job>>               queue;
	std::map<uint64_t, std::shared_ptr<Job>>       ready; // by order
	std::unordered_set<uint64_t>                   hashes; // predicted registrations
	std::unordered_set<uint64_t>                   ids;    // every job id ever queued
	size_t                                         ready_bytes = 0;
	uint64_t                                       next_order  = 0;
	uint32_t                                       live_workers = 0;
	uint32_t                                       retired      = 0;
	bool                                           stopped      = false; // too many retired workers
	size_t                                         ready_peak  = 0;
	const size_t   budget = static_cast<size_t>(EnvU32("KYTY_SHADER_PRETRANSLATE_MB", 1024)) << 20u;
	const uint32_t refill = EnvU32("KYTY_SHADER_PRETRANSLATE_REFILL", 1);
	const bool     pack   = EnvU32("KYTY_SHADER_PRETRANSLATE_PACK", 1) != 0;
	Stats        stats;
};

Pool& GetPool() {
	static auto* pool = new Pool; // never destroyed: workers may still run at exit
	return *pool;
}

void LowerThreadPriority() {
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#elif defined(__APPLE__)
	pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
}

size_t EstimateBytes(const ShaderRecompiler::TranslateResult& result) {
	const auto& program = result.program;
	size_t      insts   = program.value_storage.size();
	for (const auto* block: program.blocks) {
		insts += block->Instructions().size();
	}
	return program.blocks.size() * sizeof(ShaderRecompiler::IR::Block) +
	       insts * (sizeof(ShaderRecompiler::IR::Inst) + 64u) + result.decoded_dump.size() +
	       result.cfg_dump.size() + 4096u;
}

// Same options as the GPU thread for a pixel program without calls (ProgramCache::Get): only the
// translation inputs matter here, the emission options are the GPU thread's. False: the program has
// calls, linked per draw.
bool TranslateJob(const Job& job, ShaderRecompiler::TranslateResult* out) {
	std::array<uint32_t, HW::UserSgprInfo::SGPRS_MAX> user_data {};
	ShaderRecompiler::CompileOptions options {.wave_size = job.info.wave_size};
	options.stage             = ShaderType::Pixel;
	options.user_data_base    = 0;
	options.shader_hash       = job.hash;
	options.user_data         = std::span<const uint32_t>(user_data).first(job.user_data_count);
	options.input_info.pixel  = &job.info;
	options.dump_ir           = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump        = options.dump_ir;
	options.emit_debug_info   = Config::GraphicsDebugDumpEnabled();
	options.dump_label        = "ShaderPretranslate PS";
	// What PrepareShaderSource does for a pixel shader without calls. Calls are linked per draw
	// (ShaderSource refresh) and only supported in compute shaders: leave them to the GPU thread.
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(job.code, decoded);
	if (decoded.has_swap_pc) {
		return false;
	}
	*out = ShaderRecompiler::TranslateProgram(decoded, options);
	return true;
}

// Pool lock held.
void DropReady(Pool& pool, const std::shared_ptr<Job>& job) {
	pool.ready.erase(job->order);
	pool.ready_bytes -= job->bytes;
	job->state = State::Dropped;
	if (auto it = pool.jobs.find(job->id); it != pool.jobs.end() && it->second == job) {
		pool.jobs.erase(it);
	}
}

void WorkerMain();

// Pool lock held.
void StartWorker(Pool& pool) {
	pool.live_workers++;
	std::thread(WorkerMain).detach();
}

// The job the calling worker translates.
thread_local Job* t_job = nullptr;

// A fatal check fired while this worker translated `t_job` speculatively: drop the shader and
// replace the worker, which blocks forever when this returns (nothing it holds is touched again).
void OnWorkerRetired(std::string_view report) {
	auto&       pool = GetPool();
	auto*       job  = t_job;
	std::string line(report);
	std::ranges::replace(line, '\n', ' ');
	bool stopped = false;
	{
		std::scoped_lock lock(pool.mutex);
		pool.stats.abandoned++;
		job->state = State::Dropped;
		if (auto it = pool.jobs.find(job->id); it != pool.jobs.end() && it->second.get() == job) {
			pool.jobs.erase(it);
		}
		pool.live_workers--;
		if (++pool.retired >= MaxRetiredWorkers) {
			pool.stopped = true;
			for (auto& queued: pool.queue) {
				queued->state = State::Dropped;
			}
		} else if (!pool.queue.empty()) {
			StartWorker(pool);
		}
		stopped = pool.stopped;
	}
	pool.done.notify_all();
	std::printf("PERF pretranslate: PS %016" PRIx64 " dropped, its worker is retired (%s): %s\n",
	            job->hash, stopped ? "pretranslation stopped" : "replaced", line.c_str());
}

void WorkerMain() {
	LowerThreadPriority();
	auto& pool = GetPool();
	for (;;) {
		std::shared_ptr<Job> job;
		{
			std::unique_lock lock(pool.mutex);
			pool.work.wait(lock, [&] { return !pool.queue.empty(); });
			job = std::move(pool.queue.front());
			pool.queue.pop_front();
			if (job->state != State::Queued) {
				continue; // taken over by the GPU thread
			}
			job->state = State::Running;
		}
		const auto begin  = PerfStats::SteadyUs();
		auto       result = std::make_unique<ShaderRecompiler::TranslateResult>();
		t_job             = job.get();
		Common::SetFatalRetiresThread(OnWorkerRetired);
		const bool ok = TranslateJob(*job, result.get());
		Common::SetFatalRetiresThread(nullptr);
		t_job         = nullptr;
		const auto us = PerfStats::SteadyUs() - begin;
		pool.stats.worker_us += us;
		size_t bytes = ok ? EstimateBytes(*result) : 0;
		size_t insts = 0, blocks = 0;
		if (ok && PerfStats::CompileLogEnabled()) {
			insts  = result->program.value_storage.size();
			blocks = result->program.blocks.size();
			for (const auto* block: result->program.blocks) {
				insts += block->Instructions().size();
			}
		}
		ShaderRecompiler::IR::PackedProgram packed;
		uint64_t                            pack_us = 0;
		if (ok && pool.pack) {
			const auto pack_begin = PerfStats::SteadyUs();
			if (packed.Pack(result->program)) {
				pool.stats.packed++;
				pool.stats.unpacked_bytes += bytes;
				bytes = packed.Bytes();
				pool.stats.packed_bytes += bytes;
			}
			pack_us = PerfStats::SteadyUs() - pack_begin;
			pool.stats.pack_us += pack_us;
		}
		if (ok && PerfStats::CompileLogEnabled()) {
			// T time hash job_id worker_us blocks insts kept_bytes pack_us packed
			PerfStats::CompileLogLine("T %" PRIu64 " %016" PRIx64 " %016" PRIx64 " %" PRIu64 " %zu %zu %zu %" PRIu64 " %d",
			                          begin, job->hash, job->id, us, blocks, insts, bytes, pack_us,
			                          packed.Packed() ? 1 : 0);
		}
		{
			std::scoped_lock lock(pool.mutex);
			if (ok) {
				job->result = std::move(*result);
				job->packed = std::move(packed);
				job->bytes  = bytes + job->code.size() * sizeof(uint32_t) + sizeof(Job);
				job->state  = State::Ready;
				pool.ready.emplace(job->order, job);
				pool.ready_bytes += job->bytes;
				pool.ready_peak = std::max(pool.ready_peak, pool.ready_bytes);
				pool.stats.translated++;
				while (pool.ready_bytes > pool.budget && !pool.ready.empty()) {
					auto oldest = pool.ready.begin()->second;
					oldest->result = {};
					oldest->packed = {};
					DropReady(pool, oldest);
					pool.stats.evicted++;
				}
			} else {
				job->state = State::Dropped;
				if (auto it = pool.jobs.find(job->id); it != pool.jobs.end() && it->second == job) {
					pool.jobs.erase(it);
				}
				pool.stats.calls++;
			}
		}
		pool.done.notify_all();
	}
}

// Pool lock held.
void Enqueue(Pool& pool, std::shared_ptr<Job> job) {
	if (pool.stopped) {
		return;
	}
	job->order = pool.next_order++;
	pool.ids.insert(job->id);
	pool.jobs[job->id] = job;
	pool.queue.push_back(std::move(job));
	if (pool.live_workers < WorkerCount()) {
		StartWorker(pool);
	}
	pool.work.notify_one();
}

} // namespace

bool Enabled() noexcept {
	return WorkerCount() != 0;
}

bool VerifyEnabled() noexcept {
	static const bool verify = Enabled() && EnvU32("KYTY_SHADER_PRETRANSLATE_VERIFY", 0) != 0;
	return verify;
}

void OnShaderRegistered(uint64_t address, uint64_t hash, const ShaderMappedData& data) {
	if (!Enabled() || data.type != Prospero::ShaderBinaryType::kPs) {
		return;
	}
	auto& pool = GetPool();
	auto  job  = std::make_shared<Job>();
	if (!ShaderPredictPixelProgram(data, job->info, job->user_data_count)) {
		pool.stats.unpredicted++;
		return;
	}
	const auto words = data.code_size_bytes / sizeof(uint32_t);
	job->hash = hash;
	BuildStageTranslationKey(job->info, job->key);
	AppendProgramTranslationKey(job->key, ShaderType::Pixel, job->user_data_count, words, 0,
	                            job->info.wave_size, 0);
	job->id = XXH3_64bits_withSeed(job->key.data(), job->key.size() * sizeof(uint32_t), hash);
	const auto* code = reinterpret_cast<const uint32_t*>(address);
	job->code.assign(code, code + words);
	std::scoped_lock lock(pool.mutex);
	if (pool.jobs.contains(job->id)) {
		pool.stats.duplicates++;
		return;
	}
	pool.stats.registered++;
	pool.hashes.insert(hash);
	Enqueue(pool, std::move(job));
}

bool Take(uint64_t hash, std::span<const uint32_t> key, std::span<const uint32_t> code,
          ShaderRecompiler::TranslateResult& out) {
	auto&      pool = GetPool();
	const auto id   = XXH3_64bits_withSeed(key.data(), key.size_bytes(), hash);
	std::unique_lock lock(pool.mutex);
	const auto it = pool.jobs.find(id);
	if (it == pool.jobs.end()) {
		pool.stats.absent++;
		if (pool.hashes.contains(hash)) {
			pool.stats.absent_registered++; // other translation key, or taken/evicted/failed
		}
		if (pool.ids.contains(id)) {
			pool.stats.absent_dropped++; // that key was queued: taken, evicted or failed since
		}
		return false;
	}
	auto job = it->second;
	if (job->hash != hash || !std::ranges::equal(job->key, key)) {
		pool.stats.key_mismatch++;
		return false;
	}
	if (job->state == State::Queued) {
		// Not started: the GPU thread translates it now; the worker skips the queue entry.
		job->state = State::Dropped;
		pool.jobs.erase(it);
		pool.stats.queued_at_use++;
		return false;
	}
	if (job->state == State::Running) {
		const auto begin = PerfStats::SteadyUs();
		pool.done.wait(lock, [&] { return job->state != State::Running; });
		pool.stats.waited++;
		pool.stats.wait_us += PerfStats::SteadyUs() - begin;
	}
	if (job->state != State::Ready) {
		return false;
	}
	if (!std::ranges::equal(job->code, code)) {
		pool.stats.code_changed++;
		job->result = {};
		job->packed = {};
		DropReady(pool, job);
		return false;
	}
	out         = std::move(job->result);
	auto packed = std::move(job->packed);
	DropReady(pool, job);
	pool.stats.taken++;
	if (job->refills < pool.refill) {
		// Most used pixel programs need one more static state or resource permutation, few a third.
		auto again             = std::make_shared<Job>();
		again->hash            = job->hash;
		again->id              = job->id;
		again->key             = job->key;
		again->code            = job->code;
		again->info            = job->info;
		again->user_data_count = job->user_data_count;
		again->refills         = job->refills + 1;
		pool.stats.refills++;
		Enqueue(pool, std::move(again));
	}
	lock.unlock();
	if (packed.Packed()) {
		const auto begin = PerfStats::SteadyUs();
		out.program      = packed.Unpack();
		pool.stats.unpack_us += PerfStats::SteadyUs() - begin;
	}
	return true;
}

void Remember(uint64_t hash, std::span<const uint32_t> key, std::span<const uint32_t> code,
              const ShaderPixelInputInfo& info, uint32_t user_data_count) {
	auto& pool = GetPool();
	if (pool.refill == 0) {
		return;
	}
	auto job             = std::make_shared<Job>();
	job->hash            = hash;
	job->id              = XXH3_64bits_withSeed(key.data(), key.size_bytes(), hash);
	job->key.assign(key.begin(), key.end());
	job->code.assign(code.begin(), code.end());
	job->info            = info;
	job->user_data_count = user_data_count;
	job->refills         = 1; // the GPU thread used this key once already
	std::scoped_lock lock(pool.mutex);
	if (pool.jobs.contains(job->id)) {
		return;
	}
	pool.stats.remembered++;
	Enqueue(pool, std::move(job));
}

void NoteVerify(bool equal) noexcept {
	auto& pool = GetPool();
	pool.stats.verified++;
	if (!equal) {
		pool.stats.verify_diffs++;
	}
}

void Report() {
	if (!Enabled()) {
		return;
	}
	auto&  pool = GetPool();
	size_t ready_bytes = 0, ready_peak = 0, ready = 0, queued = 0;
	{
		std::scoped_lock lock(pool.mutex);
		ready_bytes = pool.ready_bytes;
		ready_peak  = pool.ready_peak;
		ready       = pool.ready.size();
		queued      = pool.queue.size();
	}
	const auto& s = pool.stats;
	std::printf("PERF pretranslate registered=%" PRIu64 " unpredicted=%" PRIu64 " dup=%" PRIu64
	            " translated=%" PRIu64 " calls=%" PRIu64 " abandoned=%" PRIu64 " taken=%" PRIu64 " waited=%" PRIu64
	            "(%" PRIu64 "ms) absent=%" PRIu64 "(registered %" PRIu64 " dropped %" PRIu64 ") keymiss=%" PRIu64
	            " queued_at_use=%" PRIu64 " remembered=%" PRIu64
	            " code_changed=%" PRIu64 " evicted=%" PRIu64 " refills=%" PRIu64 " ready=%zu(%zuMB peak %zuMB)"
	            " queue=%zu worker_ms=%" PRIu64 " verify=%" PRIu64 "/%" PRIu64 " packed=%" PRIu64 "(%" PRIu64
	            "MB->%" PRIu64 "MB pack_ms=%" PRIu64 " unpack_ms=%" PRIu64 ")\n",
	            s.registered.load(), s.unpredicted.load(), s.duplicates.load(), s.translated.load(),
	            s.calls.load(), s.abandoned.load(), s.taken.load(), s.waited.load(), s.wait_us.load() / 1000,
	            s.absent.load(), s.absent_registered.load(), s.absent_dropped.load(), s.key_mismatch.load(),
	            s.queued_at_use.load(), s.remembered.load(),
	            s.code_changed.load(), s.evicted.load(), s.refills.load(), ready, ready_bytes >> 20u, ready_peak >> 20u,
	            queued, s.worker_us.load() / 1000, s.verify_diffs.load(), s.verified.load(), s.packed.load(),
	            s.unpacked_bytes.load() >> 20u, s.packed_bytes.load() >> 20u, s.pack_us.load() / 1000,
	            s.unpack_us.load() / 1000);
}

} // namespace Libs::Graphics::ShaderPretranslate
