#ifndef KERNEL_SYNC_TRACE_H_
#define KERNEL_SYNC_TRACE_H_

#include <cstdint>

// Diagnostic wait-object tracer. KYTY_SYNC_TRACE=<file> enables it:
// - every wait that blocks at least KYTY_SYNC_TRACE_MIN_US (default 2000) is logged when it ends,
//   with its object, duration and result; timeouts are always counted;
// - wakes (semaphore signals, event flag set/clear, equeue triggers, cond signals) are logged when
//   a thread waits on the object;
// - every 2 s, the threads blocked for more than 1 s and the busiest objects are listed;
// - Note() adds free-form events (file opens) to the same timeline.
// Disabled, every call is one predictable branch.
namespace Libs::LibKernel::SyncTrace {

enum class Kind : uint8_t { Sema, PosixSem, EventFlag, Equeue, Cond, Mutex, Address, Count };

bool Enabled();

class Wait {
public:
	Wait(Kind kind, const void* object, const char* name, uint64_t arg, int64_t timeout_us);
	~Wait();
	Wait(const Wait&)            = delete;
	Wait& operator=(const Wait&) = delete;

	void SetResult(int result) { m_result = result; }

private:
	int      m_slot   = -1;
	int      m_result = 0;
	uint64_t m_start  = 0;
};

void Wake(Kind kind, const void* object, const char* name, uint64_t arg, int waiters);
void Note(const char* format, ...);

} // namespace Libs::LibKernel::SyncTrace

#endif
