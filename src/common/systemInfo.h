#ifndef KYTY_COMMON_SYSTEM_INFO_H_
#define KYTY_COMMON_SYSTEM_INFO_H_

#include <string>

namespace Common {

struct SystemInfo {
	std::string ProcessorName;
};

[[nodiscard]] SystemInfo GetSystemInfo();

// Whether the host can run the guest TSC rewrite (--guest-tsc): a native x86-64 CPU with BMI2. False
// under Rosetta or another x86 translator on an ARM host, and on non-x86 builds.
[[nodiscard]] bool GuestTscHostSupported();

} // namespace Common

#endif /* KYTY_COMMON_SYSTEM_INFO_H_ */
