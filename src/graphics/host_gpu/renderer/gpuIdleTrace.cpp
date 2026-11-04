#include "graphics/host_gpu/renderer/gpuIdleTrace.h"

#include "graphics/host_gpu/renderer/perfStats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics::GpuIdleTrace {

namespace {

constexpr uint64_t MergeUs = 20; // tiling states closer than this are merged
constexpr uint64_t MinUs   = 5;  // other states shorter than this are dropped

const char* const g_tag_names[static_cast<size_t>(Tag::Count)] = {
    "busy",    "command", "idle_empty", "idle_blocked", "translate", "gfx_pipe", "cs_pipe",
    "wait_gpu", "wait_other", "side_read", "drain", "back_busy", "rec_busy", "reprotect",
    "fault",   "readback", "acquire", "present", "program", "draw_slow"};

struct Pending {
	vk::Semaphore semaphore;
	uint64_t      tick;
	uint64_t      submit_us;
	uint32_t      waits;
	uint32_t      sem_id;
};

struct Trace {
	std::FILE*                  file = nullptr;
	std::mutex                  file_mutex;
	std::mutex                  queue_mutex;
	std::condition_variable     queue_cv;
	std::deque<Pending>         queue;
	vk::Device                  device;
	std::vector<vk::Semaphore>  semaphores; // index = sem id (queue lock held by the callers)
	bool                        waiter_started = false;
};

Trace& T() noexcept {
	static Trace* trace = [] {
		auto*       t    = new Trace;
		const char* path = std::getenv("KYTY_GPU_IDLE_TRACE");
		if (path != nullptr && path[0] != '\0') {
			t->file = std::fopen(path, "w");
		}
		if (t->file != nullptr) {
			std::setvbuf(t->file, nullptr, _IOFBF, 1u << 20u);
			const auto wall  = std::chrono::system_clock::now();
			const auto secs  = std::chrono::system_clock::to_time_t(wall);
			const auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(wall.time_since_epoch()).count() % 1000;
			std::tm    local {};
#if defined(_WIN32)
			localtime_s(&local, &secs);
#else
			localtime_r(&secs, &local);
#endif
			std::fprintf(t->file, "# start %02d:%02d:%02d.%03d steady_us=%" PRIu64 "\n", local.tm_hour,
			             local.tm_min, local.tm_sec, static_cast<int>(milli), PerfStats::SteadyUs());
			std::fflush(t->file);
		}
		return t;
	}();
	return *trace;
}

thread_local const char* t_name = nullptr;
thread_local std::array<std::pair<uint64_t, uint64_t>, static_cast<size_t>(Tag::Count)> t_pending {};

void Write(Tag tag, uint64_t start, uint64_t end) noexcept {
	char        other[24];
	const char* name = t_name;
	if (name == nullptr) {
		std::snprintf(other, sizeof(other), "t%zu", std::hash<std::thread::id> {}(std::this_thread::get_id()) % 100000u);
		name = other;
	}
	auto&             t = T();
	std::lock_guard   lock(t.file_mutex);
	std::fprintf(t.file, "S %s %s %" PRIu64 " %" PRIu64 "\n", g_tag_names[static_cast<size_t>(tag)], name, start, end);
}

void Waiter() {
	auto& t          = T();
	auto  last_flush = PerfStats::SteadyUs();
	for (;;) {
		Pending p {};
		{
			std::unique_lock lock(t.queue_mutex);
			t.queue_cv.wait_for(lock, std::chrono::seconds(1), [&] { return !t.queue.empty(); });
			if (!t.queue.empty()) {
				p = t.queue.front();
				t.queue.pop_front();
			}
		}
		if (p.semaphore) {
			vk::SemaphoreWaitInfo info {};
			info.semaphoreCount = 1;
			info.pSemaphores    = &p.semaphore;
			info.pValues        = &p.tick;
			const auto result   = t.device.waitSemaphores(&info, 5'000'000'000ull);
			const auto done     = result == vk::Result::eSuccess ? PerfStats::SteadyUs() : 0;
			std::lock_guard lock(t.file_mutex);
			std::fprintf(t.file, "G %u %" PRIu64 " %u %" PRIu64 " %" PRIu64 "\n", p.sem_id, p.tick, p.waits,
			             p.submit_us, done);
		}
		const auto now = PerfStats::SteadyUs();
		if (now - last_flush > 1'000'000) {
			std::lock_guard lock(t.file_mutex);
			std::fflush(t.file);
			last_flush = now;
		}
	}
}

} // namespace

bool On() noexcept {
	static const bool on = T().file != nullptr;
	return on;
}

uint64_t NowUs() noexcept {
	return PerfStats::SteadyUs();
}

void SetThreadName(const char* name) noexcept {
	t_name = name;
}

void Span(Tag tag, uint64_t start_us, uint64_t end_us) noexcept {
	if (!On()) return;
	// Only states that tile the thread's time are merged; a merged short state (a no-op flush at
	// every slice) would cover the work between its calls.
	const bool tiling = tag == Tag::Busy || tag == Tag::IdleEmpty || tag == Tag::IdleBlocked ||
	                    tag == Tag::BackBusy || tag == Tag::RecBusy;
	if (!tiling) {
		const uint64_t min = tag == Tag::Program ? 200 : tag == Tag::DrawSlow ? 500 : MinUs;
		if (end_us >= start_us + min) Write(tag, start_us, end_us);
		return;
	}
	auto& [s, e] = t_pending[static_cast<size_t>(tag)];
	if (e != 0 && start_us <= e + MergeUs) {
		e = std::max(e, end_us);
		return;
	}
	if (e != 0) Write(tag, s, e);
	s = start_us;
	e = std::max(end_us, start_us + 1);
}

void NoteSubmit(vk::Device device, vk::Semaphore semaphore, uint64_t tick, uint32_t waits) noexcept {
	if (!On()) return;
	auto&    t  = T();
	uint32_t id = 0;
	while (id < t.semaphores.size() && t.semaphores[id] != semaphore) id++;
	if (id == t.semaphores.size()) t.semaphores.push_back(semaphore);
	{
		std::lock_guard lock(t.queue_mutex);
		t.device = device;
		t.queue.push_back({semaphore, tick, PerfStats::SteadyUs(), waits, id});
		if (!t.waiter_started) {
			t.waiter_started = true;
			std::thread(Waiter).detach();
		}
	}
	t.queue_cv.notify_one();
}

void NoteQueued(uint64_t tick) noexcept {
	if (!On()) return;
	auto&           t = T();
	std::lock_guard lock(t.file_mutex);
	std::fprintf(t.file, "Q %" PRIu64 " %" PRIu64 "\n", tick, PerfStats::SteadyUs());
}

} // namespace Libs::Graphics::GpuIdleTrace
