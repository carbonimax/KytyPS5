#include "emulator.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/window.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kytyGitVersion.h"
#include "libs/agc.h"
#include "libs/audio.h"
#include "libs/controller.h"
#include "libs/libs.h"
#include "libs/network.h"
#include "loader/additionalContent.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>

namespace Emulator {

static void PrintSystemInfo() {
	const Common::SystemInfo info = Common::GetSystemInfo();

#if defined(__APPLE__)
	static constexpr auto platform_name = "macOS";
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr auto platform_name = "Windows";
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static constexpr auto platform_name = "Linux";
#else
	static constexpr auto platform_name = "Unknown";
#endif

	LOGF("Build\n"
	     "  version: %s\n\n"
	     "Host\n"
	     "  os:      %s\n"
	     "  cpu:     %s\n"
	     "  threads: %u\n\n",
	     KYTY_BUILD_LABEL, platform_name, info.ProcessorName.c_str(),
	     std::thread::hardware_concurrency());
}

// KYTY_MEM_IMPORT_TITLES=ID[=MODE][,ID[=MODE]...] (local configuration, default empty): the import
// mode (default 1) of the running title, or nullptr when it is not listed.
static const char* MemImportModeForTitle(const char* list, const std::string& title_id) {
	static thread_local std::string mode;
	if (list == nullptr || title_id.empty()) {
		return nullptr;
	}
	const std::string_view all(list);
	for (size_t at = 0; at <= all.size();) {
		auto end = all.find(',', at);
		end      = end == std::string_view::npos ? all.size() : end;
		auto item = all.substr(at, end - at);
		const auto equal = item.find('=');
		if (item.substr(0, equal) == title_id) {
			mode = equal == std::string_view::npos ? "1" : std::string(item.substr(equal + 1));
			return mode.c_str();
		}
		at = end + 1;
	}
	return nullptr;
}

static void SetEnvIfUnset(const char* name, const char* value) {
	if (std::getenv(name) != nullptr) {
		return;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	_putenv_s(name, value);
#else
	setenv(name, value, 0);
#endif
}

// Per-title guest memory import: an explicit KYTY_MEM_IMPORT wins; a listed title gets its mode and
// the labels/EOP mode it needs (KYTY_EOP_AT_GPU=1), before the GPU device is created.
static void ApplyPerTitleMemImport(const std::string& title_id) {
	EXIT_IF(MemImportModeForTitle("A1,B2=2", "B2") == nullptr || std::string(MemImportModeForTitle("A1,B2=2", "B2")) != "2" ||
	        std::string(MemImportModeForTitle("A1,B2=2", "A1")) != "1" || MemImportModeForTitle("A1", "A") != nullptr ||
	        MemImportModeForTitle("", "A1") != nullptr);
	const char* mode = MemImportModeForTitle(std::getenv("KYTY_MEM_IMPORT_TITLES"), title_id);
	if (mode == nullptr || std::getenv("KYTY_MEM_IMPORT") != nullptr) {
		return;
	}
	SetEnvIfUnset("KYTY_MEM_IMPORT", mode);
	SetEnvIfUnset("KYTY_EOP_AT_GPU", "1");
	Log::WriteToConsoleAndLog(fmt::format("Title {}: KYTY_MEM_IMPORT={} from KYTY_MEM_IMPORT_TITLES\n", title_id, mode));
}

static void KytyClose() {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	rt->Clear();

	LOGF("done!\n");

	Common::Subsystems::EmergencyShutdownActive();
}

static void MountOrCreateDir(const std::filesystem::path& dir, const std::string& point) {
	if (!Common::File::IsDirectoryExisting(dir)) {
		Common::File::CreateDirectories(dir);
	}

	EXIT_NOT_IMPLEMENTED(!Common::File::IsDirectoryExisting(dir));

	Libs::LibKernel::FileSystem::Mount(dir, point);
	auto dir_text = Common::PathToString(dir);
	LOGF("Mounted %s -> %s\n", point.c_str(), dir_text.c_str());
}

static void MountSandboxDirs() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	MountOrCreateDir("_DownloadData/" + title_id, "/download0");
	MountOrCreateDir("_TempData/" + title_id, "/temp0");
	MountOrCreateDir("_TempData/" + title_id, "/temp");
}

static void Init(const Config::ConfigOptions& cfg, const std::filesystem::path& param_json,
                 Common::Subsystems& subsystems) {
	EXIT_IF(!Common::Thread::IsMainThread());

	subsystems.Initialize<Config::Lifecycle>();
	Config::Load(cfg);
	subsystems.Initialize<Log::Lifecycle>();

	if (Common::File::IsFileExisting(param_json)) {
		Loader::SystemContentLoadParamSfo(param_json);
	}
	if (cfg.splash_enabled) {
		Libs::Graphics::InitializeSplashScreen(param_json.parent_path());
	}

	// Initialization order is explicit; destruction is automatic and reversed.
	subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
	subsystems.Initialize<Profiler::Lifecycle>();
	subsystems.Initialize<Libs::Network::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::FileSystem::Lifecycle>();
	subsystems.Initialize<Libs::Controller::Lifecycle>();
	subsystems.Initialize<Libs::Audio::Lifecycle>();
	subsystems.Initialize<Libs::Graphics::Lifecycle>();
}

static void LoadElf(const std::filesystem::path& elf, bool dbg_print_reloc = false,
                    const std::filesystem::path& save_name = {}) {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	auto* program = rt->LoadProgram(
	    Libs::LibKernel::FileSystem::GetRealFilename(Common::PathToGenericString(elf)));

	if (dbg_print_reloc) {
		program->dbg_print_reloc = true;
	}

	if (!save_name.empty()) {
		rt->SaveProgram(program, Libs::LibKernel::FileSystem::GetRealFilename(
		                             Common::PathToGenericString(save_name)));
	}
}

static void Execute(const std::filesystem::path& game_patch) {
	std::thread([game_patch] {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->Execute(game_patch);
	}).detach();
	Libs::Graphics::WindowRun();
	std::quick_exit(0);
}

void Run(const RunOptions& options) {
	if (options.app0_dir.empty()) {
		EXIT("app0 directory is required\n");
	}

	if (options.elf.empty()) {
		EXIT("ELF is required\n");
	}

	const auto         param_json = options.app0_dir / "sce_sys" / "param.json";
	Common::Subsystems subsystems(true);
	Init(options.config, param_json, subsystems);

	PrintSystemInfo();
	std::string title_id;
	Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id);
	if (title_id.empty()) {
		std::string content_id;
		if (Loader::SystemContentParamSfoGetString("CONTENT_ID", &content_id) &&
		    content_id.size() == 36 && content_id[6] == '-' && content_id[16] == '_') {
			title_id = content_id.substr(7, 9);
		}
	}
	if (!title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {}\n", title_id));
	}
	ApplyPerTitleMemImport(title_id);

	int ok = atexit(KytyClose);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	// Guest threads are still running, so skip KytyClose() and only flush emergency state.
	ok = at_quick_exit(Common::Subsystems::EmergencyShutdownActive);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/app0");
	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/hostapp");

	MountSandboxDirs();

	Loader::AddContentLoad(options.addcont_dir, options.addcont, title_id);
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	Libs::InitAll(rt->Symbols());

	LoadElf(options.elf);

	Execute(options.game_patch);
}

} // namespace Emulator
