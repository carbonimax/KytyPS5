#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <thread>

namespace Common {

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

static thread_local RetireHandler t_retire_handler = nullptr;

void SetFatalRetiresThread(RetireHandler on_retire) noexcept {
	t_retire_handler = on_retire;
}

static void RetireThreadIfSpeculative(const char* title, std::string_view text, const char* file,
                                      int line) {
	auto* on_retire = t_retire_handler;
	if (on_retire == nullptr) {
		return;
	}
	t_retire_handler = nullptr;
	on_retire(BuildFatalReport(title, text, file, line));
	for (;;) {
		std::this_thread::sleep_for(std::chrono::hours(24));
	}
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	RetireThreadIfSpeculative(title, text, file, line);
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	RetireThreadIfSpeculative("--- Error ---", text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	RetireThreadIfSpeculative("--- Error ---", text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
