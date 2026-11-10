#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/renderer/fsr.h"

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

// Without Pro mode, an FSR folder, or the loader DLL in it, FSR loads nothing, adds no device
// extension and stays off, so PSSR keeps its linear upscale. Needs no AMD driver or FSR DLL.
TEST(FsrTest, StaysOffWithoutProModeAFolderOrItsDll) {
	vk::ExtensionProperties matrix;
	matrix.extensionName = std::string(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	Config::Initialize();
	for (const auto& [pro, folder]:
	     {std::pair {false, "fsr"}, {true, ""}, {true, "no-such-fsr-folder"}}) {
		Config::ConfigOptions options;
		options.pro_enabled = pro;
		options.fsr_folder  = folder;
		Config::Load(options);
		Fsr::Load();
		std::vector<const char*> extensions;
		Fsr::AppendExtensions(extensions, {matrix}, false);
		Fsr::AppendExtensions(extensions, {matrix}, true);
		EXPECT_FALSE(Fsr::Loaded()) << pro << folder;
		EXPECT_TRUE(extensions.empty()) << pro << folder;
		EXPECT_FALSE(Fsr::Enabled()) << pro << folder;
		Fsr::Shutdown();
	}
	Config::Shutdown();
}

// FSR takes temporal requests without a render offset; the rest scale linearly.
TEST(FsrTest, AcceptsTemporalRequestsAtTheOrigin) {
	Pssr::Request request {};
	request.depth[0]  = 1;
	request.motion[0] = 1;
	EXPECT_TRUE(Fsr::Accepts(request));
	request.motion_offset[0] = 0.5f; // folded into the vectors before FSR runs
	EXPECT_TRUE(Fsr::Accepts(request));
	request.x = 16;
	EXPECT_FALSE(Fsr::Accepts(request));
	request.x     = 0;
	request.flags = Pssr::kSpatialOnly;
	EXPECT_FALSE(Fsr::Accepts(request));
	request.flags     = 0;
	request.motion[0] = 0;
	EXPECT_FALSE(Fsr::Accepts(request));
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
