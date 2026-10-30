#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMTRACKPROF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMTRACKPROF_H_

#include <chrono>
#include <cstddef>
#include <cstdint>

// KYTY_MEMTRACK_PROF=1 (measurement, default 0): cost of the protection-fault based GPU memory
// tracking. Counts and times the host faults resolved for the tracker (by path, write-fault kind and
// thread), the host protection changes (by mode, thread and whether a fault handler made them), the
// CPU-to-GPU uploads of dirty ranges and the GPU-to-guest readbacks. Reported as "PERF memtrack"
// lines with the PerfStats report (needs KYTY_PERF_STATS=1 for the report cadence).
//
// Off: every entry point returns after one cached flag test. Portable (std::chrono and relaxed
// atomics only); safe on any host thread, including inside a host exception handler.
namespace Libs::Graphics::MemTrackProf {

// Path that resolved a host fault (KytyExceptionHandler).
enum class Fault : uint8_t {
	Write,      // tracker write fault (buffer/texture/image-heap invalidation)
	Read,       // tracker read fault (publication of GPU-written bytes)
	Remap,      // wait for a backing remap
	WalkerRead, // SRT-walker load emulated from the backing (GPU thread)
	Skip,       // SRT-walker load skipped
	Count
};

// Which tracker a write fault invalidated.
enum class Kind : uint8_t { Buffer, Texture, Both, Other, Count };

// Host protection set by the page manager.
enum class Prot : uint8_t { ReadWrite, Read, NoAccess, Count };

// Thread class of the calling thread.
enum class ThreadKind : uint8_t { Host, Gpu, Deferred, Guest, Count };

enum class Counter : uint32_t {
	SyncNs,           // BufferCache::SynchronizeBuffer (tracker walk, protections, copies, recording)
	SyncCount,
	UploadBytes,      // guest bytes copied to staging for the GPU
	UploadCopies,
	UploadCopyNs,     // CPU copies of guest bytes into staging (immediate and deferred)
	DownloadSyncNs,   // GPU thread waiting for a synchronous download
	DownloadSyncCount,
	DownloadBytes,    // GPU-written bytes copied back to the guest (all paths)
	DownloadCount,
	FaultCmdNs,       // GPU thread running a guest fault's command (readback recording, state)
	FaultCmdCount,
	FaultSendNs,      // faulting threads inside SendCommandSync (queue + GPU-thread command)
	FaultWaitNs,      // faulting threads waiting for a readback's publication
	FaultWaitCount,
	WriteFlushCount,  // write faults on GPU-written pages (readback before the write)
	InvalidateCount,  // non-fault invalidations (HLE writes, file reads, unmaps)
	InvalidateBytes,
	ReadAheadPages,   // KYTY_WRITE_FAULT_READAHEAD pages opened ahead of sequential writers
	Count
};

[[nodiscard]] bool Enabled() noexcept;
void               Add(Counter counter, uint64_t value) noexcept;
[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Names the calling thread's slot (first call wins). Unnamed threads ask the namer, if set.
void SetThreadKind(ThreadKind kind, const char* name) noexcept;
// Returns true and fills `name` for a guest thread; false for a host thread.
using Namer = bool (*)(char* name, size_t size);
void SetNamer(Namer namer) noexcept;

// Guest mapping that holds a faulting address: a view of the shared direct-memory backing (file
// mapping, aliasable) or private memory (flexible/committed). Set by the kernel memory module.
enum class Mapping : uint8_t { Backing, Private, Unknown, Count };
using Classifier = Mapping (*)(uint64_t vaddr);
void SetClassifier(Classifier classifier) noexcept;
struct MapStats {
	uint64_t backing_bytes = 0; // guest views of the backing
	uint64_t backing_views = 0;
	uint64_t aliased_bytes = 0; // backing bytes mapped by more than one guest view
	uint64_t private_bytes = 0;
};
using MapStatsFn = MapStats (*)();
void SetMapStats(MapStatsFn fn) noexcept;

// Host exception handler: one resolved fault at `vaddr` of `path` that took `ns` in the handler.
void NoteFault(Fault path, uint64_t vaddr, uint64_t ns) noexcept;
// Inside the handler: which caches a write fault invalidated.
void BeginFault() noexcept;
void NoteBufferHit() noexcept;
void NoteTextureHit() noexcept;
[[nodiscard]] bool InFault() noexcept;
// A faulting thread waited `ns` for a readback's publication (FaultWaitNs, per thread too).
void NoteFaultWait(uint64_t ns) noexcept;
// A host protection change of `bytes` to `mode` that took `ns`.
void NoteProtect(Prot mode, uint64_t bytes, uint64_t ns) noexcept;

// Called from the PerfStats report: prints the lines for the last `seconds`.
void Report(double seconds, uint64_t frames) noexcept;

class Timer {
public:
	explicit Timer(Counter time, Counter count = Counter::Count) noexcept
	    : m_time(time), m_count(count), m_start(Enabled() ? NowNs() : 0) {}
	~Timer() {
		if (m_start != 0) {
			Add(m_time, NowNs() - m_start);
			if (m_count != Counter::Count) {
				Add(m_count, 1);
			}
		}
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;

private:
	Counter  m_time;
	Counter  m_count;
	uint64_t m_start;
};

} // namespace Libs::Graphics::MemTrackProf

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMTRACKPROF_H_
