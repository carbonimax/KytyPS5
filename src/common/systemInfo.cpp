#include "common/systemInfo.h"

#include "common/assert.h"
#include "cpuinfo.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace Common {

static bool HostTranslatesX86() {
#if defined(__APPLE__)
	int    translated = 0;
	size_t size       = sizeof(translated);
	return sysctlbyname("sysctl.proc_translated", &translated, &size, nullptr, 0) == 0 &&
	       translated != 0;
#elif defined(_WIN32)
	USHORT process_machine = 0;
	USHORT native_machine  = 0;
	return IsWow64Process2(GetCurrentProcess(), &process_machine, &native_machine) != 0 &&
	       native_machine != IMAGE_FILE_MACHINE_AMD64;
#else
	return false;
#endif
}

bool GuestTscHostSupported() {
#if defined(__x86_64__) || defined(_M_X64)
	static const bool supported = [] {
		if (HostTranslatesX86()) return false;
		static const bool initialized = cpuinfo_initialize();
		return initialized && cpuinfo_has_x86_bmi2();
	}();
	return supported;
#else
	return false;
#endif
}

SystemInfo GetSystemInfo() {
	static const bool initialized = cpuinfo_initialize();
	EXIT_IF(!initialized);

	const auto* package = cpuinfo_get_package(0);
	EXIT_IF(package == nullptr);

	return {package->name};
}

} // namespace Common
