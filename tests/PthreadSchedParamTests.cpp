#include <SDL3/SDL_main.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

using namespace Libs::LibKernel;

static void Check(bool value, const char* message) {
	if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main(int, char**) {
	Common::InitializeThreads();
	Common::Subsystems subsystems;
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions options;
	options.printf_direction = Config::LogDirection::Silent;
	Config::Load(options);
	subsystems.Initialize<Log::Lifecycle>();
	Initialize();
	struct Buffer { KernelSchedParam param; uint32_t sentinel; PthreadAttr attr; } buffer {};
	static_assert(offsetof(Buffer, sentinel) == 4);
	Check(PthreadAttrInit(&buffer.attr) == 0, "attribute init");
	const auto original_attr = buffer.attr;
	for (int priority : {478, 700, 733}) {
		KernelSchedParam input {priority};
		Check(PthreadAttrSetschedparam(&buffer.attr, &input) == 0, "set priority");
		buffer.sentinel = 0xdeadbeef;
		Check(PthreadAttrGetschedparam(&buffer.attr, &buffer.param) == 0, "get priority");
		Check(buffer.sentinel == 0xdeadbeef && buffer.attr == original_attr, "four-byte output preserves adjacent guest memory");
		Check(buffer.param.sched_priority == priority, "exact guest priority round trip");
	}
	Check(PthreadAttrGetschedparam(nullptr, &buffer.param) == KERNEL_ERROR_EINVAL, "null attribute");
	PthreadAttr empty = nullptr;
	Check(PthreadAttrGetschedparam(&empty, &buffer.param) == KERNEL_ERROR_EINVAL, "empty attribute");
	Check(PthreadAttrGetschedparam(&buffer.attr, nullptr) == KERNEL_ERROR_EINVAL, "null output");
	Check(PthreadAttrSetschedparam(&buffer.attr, nullptr) == KERNEL_ERROR_EINVAL, "null input");
	Check(PthreadAttrDestroy(&buffer.attr) == 0, "attribute destroy");
	std::puts("PASS: guest sched_param bounds, exact priorities, invalid arguments");
	return 0;
}
