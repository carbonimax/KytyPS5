#include "graphics/host_gpu/renderer/cache/stagingMemoryPolicy.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::HasResizableBarHeap;
using Flag = vk::MemoryPropertyFlagBits;

constexpr uint64_t MiB = 1024ull * 1024;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "StagingMemoryPolicyTests: failed: %s\n", message);
		std::abort();
	}
}

// Heap 0 = system RAM (host), heap 1 = VRAM; type 0 host memory, type 1 VRAM, type 2 the CPU window
// into heap `bar_heap` (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT).
vk::PhysicalDeviceMemoryProperties Memory(uint64_t vram, uint64_t bar_heap_size, bool separate_bar_heap) {
	vk::PhysicalDeviceMemoryProperties memory {};
	memory.memoryHeapCount = separate_bar_heap ? 3 : 2;
	memory.memoryHeaps[0]  = vk::MemoryHeap {48 * 1024 * MiB, {}};
	memory.memoryHeaps[1]  = vk::MemoryHeap {vram, vk::MemoryHeapFlagBits::eDeviceLocal};
	if (separate_bar_heap) {
		memory.memoryHeaps[2] = vk::MemoryHeap {bar_heap_size, vk::MemoryHeapFlagBits::eDeviceLocal};
	}
	memory.memoryTypeCount = 3;
	memory.memoryTypes[0]  = vk::MemoryType {Flag::eHostVisible | Flag::eHostCoherent, 0};
	memory.memoryTypes[1]  = vk::MemoryType {Flag::eDeviceLocal, 1};
	memory.memoryTypes[2] = vk::MemoryType {Flag::eDeviceLocal | Flag::eHostVisible | Flag::eHostCoherent,
	                                        separate_bar_heap ? 2u : 1u};
	return memory;
}

} // namespace

int main() {
	constexpr auto discrete   = vk::PhysicalDeviceType::eDiscreteGpu;
	constexpr auto integrated = vk::PhysicalDeviceType::eIntegratedGpu;

	// Resizable BAR: the host-visible device-local type spans the whole 16 GiB VRAM heap.
	Check(HasResizableBarHeap(discrete, Memory(16 * 1024 * MiB, 0, false)), "ReBAR VRAM heap");
	// Legacy BAR: a separate 256 MiB window.
	Check(!HasResizableBarHeap(discrete, Memory(16 * 1024 * MiB, 256 * MiB, true)), "256 MiB BAR window");
	// Exactly the threshold is not enough.
	Check(!HasResizableBarHeap(discrete, Memory(16 * 1024 * MiB, 1024 * MiB, true)), "1 GiB window");
	// No host-visible device-local type at all.
	auto no_bar = Memory(16 * 1024 * MiB, 0, false);
	no_bar.memoryTypeCount = 2;
	Check(!HasResizableBarHeap(discrete, no_bar), "no BAR memory type");
	// Unified memory (integrated / Apple GPU through MoltenVK): every type is device-local and
	// host-visible, moving the ring changes nothing.
	Check(!HasResizableBarHeap(integrated, Memory(16 * 1024 * MiB, 0, false)), "integrated GPU");

	std::printf("StagingMemoryPolicyTests: OK\n");
	return 0;
}
