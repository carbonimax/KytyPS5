#include "graphics/host_gpu/renderer/aliasedColorPassPolicy.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using Libs::Graphics::AliasedPassStageWrite;
using Libs::Graphics::ShaderRecompiler::IR::Bindings;
using Libs::Graphics::ShaderRecompiler::IR::DescriptorBindingKind;
using Libs::Graphics::ShaderRecompiler::IR::ShaderInfo;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AliasedColorPassPolicyTests: failed: %s\n", message);
		std::abort();
	}
}

// What the mesh index fetch leaves behind: one address read (uses_dma) and read-only buffers.
ShaderInfo ReadOnlyDma() {
	ShaderInfo info;
	info.uses_dma = true;
	info.buffers.resize(2);
	info.images.resize(1);
	return info;
}

} // namespace

int main() {
	const Bindings none;
	Check(AliasedPassStageWrite(ReadOnlyDma(), none, false).empty(),
	      "a stage that only reads by address was refused per-target passes");
	Check(AliasedPassStageWrite(ShaderInfo {}, none, false).empty(),
	      "a stage without memory access was refused per-target passes");

	Check(AliasedPassStageWrite(ReadOnlyDma(), none, true) == "address stores",
	      "address stores were accepted");
	auto buffer_write              = ReadOnlyDma();
	buffer_write.buffers[1].written = true;
	Check(AliasedPassStageWrite(buffer_write, none, false) == "buffer 1 writes",
	      "a buffer write was accepted");
	auto buffer_atomic             = ReadOnlyDma();
	buffer_atomic.buffers[0].atomic = true;
	Check(AliasedPassStageWrite(buffer_atomic, none, false) == "buffer 0 atomics",
	      "a buffer atomic was accepted");
	auto image_write              = ReadOnlyDma();
	image_write.images[0].written = true;
	Check(AliasedPassStageWrite(image_write, none, false) == "image 0 writes",
	      "an image write was accepted");
	auto image_atomic             = ReadOnlyDma();
	image_atomic.images[0].atomic = true;
	Check(AliasedPassStageWrite(image_atomic, none, false) == "image 0 atomics",
	      "an image atomic was accepted");
	Bindings gds;
	gds.descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Gds)] = 1;
	gds.descriptor_mask |= uint64_t{1} << static_cast<size_t>(DescriptorBindingKind::Gds);
	Check(AliasedPassStageWrite(ReadOnlyDma(), gds, false) == "GDS", "GDS access was accepted");

	std::puts("AliasedColorPassPolicyTests: all cases passed");
	return 0;
}
