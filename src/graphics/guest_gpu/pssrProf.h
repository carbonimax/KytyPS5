#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSRPROF_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSRPROF_H_

#include "graphics/guest_gpu/pssr.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

// KYTY_PSSR_PROF=1 (measurement, default off, one cached bool test when off): once a second, "PSSRPROF" lines
// with the libPsml entry points called, the ProcessPssr calls, and per guest context the input/output size and
// flags of the requests. Header only so that libPsml and the command processor share the counters.
namespace Libs::Graphics::Pssr::Prof {

enum class Entry : uint32_t {
	Initialize,
	GetSharedRequirements,
	CreateSharedResources,
	GetContextRequirements,
	CreateContext,
	ReleaseContext,
	GetPacketSize,
	Dispatch,
	DispatchOk, // the packet was written
	GetMipmapBias,
	Mfsr2Initialize,
	Mfsr2CreateContext,
	Mfsr2Dispatch,
	Mfsr2DispatchOk,
	// why a request was marked spatial-only (one call per cause, a request may have several)
	WhyMotionOffset,
	WhyViewDepth,
	WhyPassThrough,
	WhyVrLookup,
	WhyMotionGamma,
	WhyMfsr2Encoding,
	WhyMfsr2Dilated,
	Count
};

inline const char* const kEntryNames[] = {"init",       "shared_req",  "shared_create", "ctx_req",     "ctx_create",
                                          "ctx_release", "packet_size", "dispatch",      "dispatch_ok", "mip_bias",
                                          "mfsr2_init",  "mfsr2_ctx",   "mfsr2_disp",    "mfsr2_disp_ok",
                                          "why_motion_offset", "why_view_depth", "why_pass_through", "why_vr_lookup",
                                          "why_motion_gamma", "why_mfsr2_encoding", "why_mfsr2_dilated"};
static_assert(sizeof(kEntryNames) / sizeof(kEntryNames[0]) == static_cast<size_t>(Entry::Count));

struct ContextStats {
	uint64_t count = 0, spatial = 0, reset = 0, reverse = 0, auto_exposure = 0, depth = 0, motion = 0,
	         exposure = 0;
	uint32_t in_w = 0, in_h = 0, render_w = 0, render_h = 0, out_w = 0, out_h = 0;
};

struct State {
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Entry::Count)> entries {};
	std::atomic<uint64_t>                                                 requests {};
	std::mutex                                                            mutex;
	std::map<uint64_t, ContextStats>                                      contexts; // also past sizes: last one kept
	std::once_flag                                                        started;
};

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PSSR_PROF");
		return value != nullptr && std::atoi(value) != 0;
	}();
	return enabled;
}

inline State& Get() {
	static State state;
	return state;
}

inline bool Present(const uint32_t (&descriptor)[8]) {
	for (const auto word: descriptor) {
		if (word != 0) {
			return true;
		}
	}
	return false;
}

inline void Report() {
	auto&    s = Get();
	uint64_t last_entries[static_cast<size_t>(Entry::Count)] {}, last_requests = 0;
	std::map<uint64_t, uint64_t> last_counts;
	for (int second = 1;; ++second) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		uint64_t entries[static_cast<size_t>(Entry::Count)];
		bool     changed = false;
		for (size_t i = 0; i < std::size(entries); ++i) {
			entries[i] = s.entries[i].load(std::memory_order_relaxed);
			changed |= entries[i] != last_entries[i];
		}
		const uint64_t requests = s.requests.load(std::memory_order_relaxed);
		changed |= requests != last_requests;
		if (!changed) {
			continue;
		}
		std::printf("PSSRPROF t=%ds ProcessPssr/s=%llu total=%llu |", second,
		            static_cast<unsigned long long>(requests - last_requests),
		            static_cast<unsigned long long>(requests));
		for (size_t i = 0; i < std::size(entries); ++i) {
			if (entries[i] != 0) {
				std::printf(" %s=%llu(+%llu)", kEntryNames[i], static_cast<unsigned long long>(entries[i]),
				            static_cast<unsigned long long>(entries[i] - last_entries[i]));
			}
		}
		std::printf("\n");
		{
			std::lock_guard lock(s.mutex);
			for (const auto& [context, c]: s.contexts) {
				const auto delta = c.count - last_counts[context];
				std::printf("PSSRPROF   ctx=%llx +%llu/s total=%llu in=%ux%u render=%ux%u out=%ux%u spatial=%llu "
				            "reset=%llu reverse=%llu autoexp=%llu depth=%llu motion=%llu exposure=%llu\n",
				            static_cast<unsigned long long>(context), static_cast<unsigned long long>(delta),
				            static_cast<unsigned long long>(c.count), c.in_w, c.in_h, c.render_w, c.render_h,
				            c.out_w, c.out_h, static_cast<unsigned long long>(c.spatial),
				            static_cast<unsigned long long>(c.reset), static_cast<unsigned long long>(c.reverse),
				            static_cast<unsigned long long>(c.auto_exposure),
				            static_cast<unsigned long long>(c.depth), static_cast<unsigned long long>(c.motion),
				            static_cast<unsigned long long>(c.exposure));
				last_counts[context] = c.count;
			}
		}
		std::fflush(stdout);
		std::copy(std::begin(entries), std::end(entries), std::begin(last_entries));
		last_requests = requests;
	}
}

inline void Start() {
	auto& s = Get();
	std::call_once(s.started, [] { std::thread(Report).detach(); });
}

inline void Call(Entry entry) {
	if (!Enabled()) {
		return;
	}
	Start();
	Get().entries[static_cast<size_t>(entry)].fetch_add(1, std::memory_order_relaxed);
}

inline void Note(const Request& request) {
	if (!Enabled()) {
		return;
	}
	Start();
	auto& s = Get();
	s.requests.fetch_add(1, std::memory_order_relaxed);
	ShaderTextureResource input, output;
	std::copy_n(request.input, 8, input.fields);
	std::copy_n(request.output, 8, output.fields);
	std::lock_guard lock(s.mutex);
	auto&           c = s.contexts[request.context];
	++c.count;
	c.spatial += (request.flags & kSpatialOnly) != 0;
	c.reset += (request.flags & kReset) != 0;
	c.reverse += (request.flags & kReverseDepth) != 0;
	c.auto_exposure += (request.flags & kAutoExposure) != 0;
	c.depth += Present(request.depth);
	c.motion += Present(request.motion);
	c.exposure += Present(request.exposure);
	c.in_w     = input.Width5() + 1u;
	c.in_h     = input.Height5() + 1u;
	c.render_w = request.width;
	c.render_h = request.height;
	c.out_w    = output.Width5() + 1u;
	c.out_h    = output.Height5() + 1u;
}

} // namespace Libs::Graphics::Pssr::Prof

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSRPROF_H_
