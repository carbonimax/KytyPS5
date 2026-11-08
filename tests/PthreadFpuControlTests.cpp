#include <SDL3/SDL_main.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "common/singleton.h"
#include "kernel/pthread.h"
#include "loader/runtimeLinker.h"

using namespace Libs::LibKernel;

static void Check(bool value, const char* message) {
	if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static uint16_t g_fcw   = 0;
static uint32_t g_mxcsr = 0;

static KYTY_SYSV_ABI void* Entry(void* /*arg*/) {
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("fnstcw %0" : "=m"(g_fcw));
	asm volatile("stmxcsr %0" : "=m"(g_mxcsr));
#endif
	return nullptr;
}

alignas(64) static uint8_t g_stack[256 * 1024];

int main(int, char**) {
#if !defined(__x86_64__) && !defined(_M_X64)
	std::puts("SKIP: no x87 on this host");
	return 0;
#else
	Common::InitializeThreads();
	Common::Subsystems subsystems;
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions options;
	options.printf_direction = Config::LogDirection::Silent;
	Config::Load(options);
	subsystems.Initialize<Log::Lifecycle>();
	Initialize();
	(void)Common::Singleton<Loader::RuntimeLinker>::Instance(); // main thread only; guest thread exit uses it
	// The host default on Windows; set it here too so a thread that inherits it (POSIX hosts) still fails without the fix.
	const uint16_t host = 0x027f;
	asm volatile("fldcw %0" : : "m"(host));
	PthreadAttr attr = nullptr;
	Check(PthreadAttrInit(&attr) == 0, "attribute init");
	Check(PthreadAttrSetstack(&attr, g_stack, sizeof(g_stack)) == 0, "guest-supplied stack");
	Pthread thread = nullptr;
	Check(PthreadCreate(&thread, &attr, Entry, nullptr, "fpu_control") == 0, "create");
	Check(PthreadJoin(thread, nullptr) == 0, "join");
	Check(PthreadAttrDestroy(&attr) == 0, "attribute destroy");
	std::printf("guest thread: x87 control word 0x%04x, mxcsr 0x%08x\n", g_fcw, g_mxcsr);
	Check(g_fcw == 0x037f, "guest thread starts with x87 control word 0x037F");
	Check((g_mxcsr & ~0x3fu) == 0x1f80u, "MXCSR control bits keep the host default 0x1F80");
	std::puts("PASS: guest thread x87 control word");
	return 0;
#endif
}
