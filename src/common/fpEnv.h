#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// x87 control word 0x037F (64-bit precision, all exceptions masked, round to nearest). MXCSR 0x9FC0:
// all exceptions masked, round to nearest, flush-to-zero and denormals-are-zero, as on the console.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

inline FpEnv ReadFpEnv() {
	FpEnv env {};
	asm volatile("stmxcsr %0\n\tfnstcw %1" : "=m"(env.mxcsr), "=m"(env.control) : : "memory");
	return env;
}

inline void WriteFpEnv(FpEnv env) {
	asm volatile("ldmxcsr %0\n\tfldcw %1" : : "m"(env.mxcsr), "m"(env.control) : "memory");
}

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
