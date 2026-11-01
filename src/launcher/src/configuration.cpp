#include "configuration.h"
#if defined(__APPLE__)
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QStandardPaths>
#endif

QVariantMap Configuration::GameSettings() const {
	QVariantMap values;
#define KYTY_CFG_SET(n) values.insert(QStringLiteral(#n), QVariant::fromValue(n).toString())
	KYTY_CFG_SET(screen_resolution);
	KYTY_CFG_SET(user_name);
	KYTY_CFG_SET(user_id);
	KYTY_CFG_SET(audio_input_device);
	KYTY_CFG_SET(present_mode);
	KYTY_CFG_SET(gpu_index);
	KYTY_CFG_SET(fullscreen_enabled);
	KYTY_CFG_SET(hdr_enabled);
	KYTY_CFG_SET(hide_cursor_enabled);
	KYTY_CFG_SET(readback_linear_images);
	KYTY_CFG_SET(readback_tiled_images);
	KYTY_CFG_SET(sync_raw_image_buffers);
	KYTY_CFG_SET(tessellation_enabled);
	KYTY_CFG_SET(occlusion_queries_enabled);
	KYTY_CFG_SET(shader_cache_enabled);
	KYTY_CFG_SET(trophy_enabled);
	KYTY_CFG_SET(trophy_duration_seconds);
	KYTY_CFG_SET(perf_overlay_enabled);
	KYTY_CFG_SET(skip_notice_screen);
	KYTY_CFG_SET(network_connected);
	KYTY_CFG_SET(vblank_frequency);
	KYTY_CFG_SET(console_language);
	KYTY_CFG_SET(vulkan_validation_enabled);
	KYTY_CFG_SET(shader_validation_enabled);
	KYTY_CFG_SET(shader_optimization_type);
	KYTY_CFG_SET(shader_log_direction);
	KYTY_CFG_SET(shader_log_folder);
	KYTY_CFG_SET(command_buffer_dump_enabled);
	KYTY_CFG_SET(command_buffer_dump_folder);
	KYTY_CFG_SET(printf_direction);
	KYTY_CFG_SET(printf_output_file);
	KYTY_CFG_SET(profiler_enabled);
	KYTY_CFG_SET(renderdoc_enabled);
	KYTY_CFG_SET(amd_cpu_enabled);
#if defined(_WIN32)
	KYTY_CFG_SET(red_zone_protection_enabled);
#endif
	KYTY_CFG_SET(elf);
#if defined(__APPLE__)
	KYTY_CFG_SET(mac_srt_readback_elide);
	KYTY_CFG_SET(mac_materialize_compiled);
	KYTY_CFG_SET(mac_commit_overlap_fast);
	KYTY_CFG_SET(mac_image_lookup_fast);
	KYTY_CFG_SET(mac_pervertex_unique_disabled);
	KYTY_CFG_SET(mac_moltenvk_path);
	KYTY_CFG_SET(mac_extra_environment);
#endif
#undef KYTY_CFG_SET
	return values;
}

template <class Settings>
void Configuration::ReadGameSettingsValues(const Settings& s) {
#define KYTY_CFG_GET(n) n = s.value(QStringLiteral(#n)).template value<decltype(n)>()
	KYTY_CFG_GET(screen_resolution);
	user_name          = s.value("user_name", user_name).toString();
	bool user_id_ok    = false;
	auto saved_user_id = s.value("user_id", user_id).toInt(&user_id_ok);
	user_id            = user_id_ok && Config::IsConfiguredUserIdValid(saved_user_id)
	                         ? saved_user_id
	                         : Config::DEFAULT_USER_ID;
	audio_input_device = s.value("audio_input_device", audio_input_device).toString();
	KYTY_CFG_GET(present_mode);
	gpu_index = s.value("gpu_index", -1).toInt();
	if (EnumToText(present_mode).isEmpty()) {
		present_mode = PresentMode::Mailbox;
	}
	KYTY_CFG_GET(fullscreen_enabled);
	KYTY_CFG_GET(hdr_enabled);
	KYTY_CFG_GET(hide_cursor_enabled);
	KYTY_CFG_GET(readback_linear_images);
	KYTY_CFG_GET(readback_tiled_images);
	KYTY_CFG_GET(sync_raw_image_buffers);
	KYTY_CFG_GET(tessellation_enabled);
	KYTY_CFG_GET(occlusion_queries_enabled);
	shader_cache_enabled = s.value("shader_cache_enabled", shader_cache_enabled).toBool();
	trophy_enabled       = s.value("trophy_enabled", trophy_enabled).toBool();
	KYTY_CFG_GET(trophy_duration_seconds);
	KYTY_CFG_GET(perf_overlay_enabled);
	KYTY_CFG_GET(skip_notice_screen);
	KYTY_CFG_GET(network_connected);
	vblank_frequency = s.value("vblank_frequency", vblank_frequency).toInt();
	console_language = s.value("console_language", console_language).toInt();
	if (console_language < 0 || console_language > MAX_CONSOLE_LANGUAGE) {
		console_language = DEFAULT_CONSOLE_LANGUAGE;
	}
	KYTY_CFG_GET(vulkan_validation_enabled);
	KYTY_CFG_GET(shader_validation_enabled);
	KYTY_CFG_GET(shader_optimization_type);
	KYTY_CFG_GET(shader_log_direction);
	KYTY_CFG_GET(shader_log_folder);
	KYTY_CFG_GET(command_buffer_dump_enabled);
	KYTY_CFG_GET(command_buffer_dump_folder);
	KYTY_CFG_GET(printf_direction);
	KYTY_CFG_GET(printf_output_file);
	KYTY_CFG_GET(profiler_enabled);
	KYTY_CFG_GET(renderdoc_enabled);
	amd_cpu_enabled = s.value("amd_cpu_enabled", false).toBool();
#if defined(_WIN32)
	red_zone_protection_enabled =
	    s.value("red_zone_protection_enabled", red_zone_protection_enabled).toBool();
#endif
	elf = s.value("elf", elf).toString();
#if defined(__APPLE__)
	mac_srt_readback_elide = s.value("mac_srt_readback_elide", mac_srt_readback_elide).toBool();
	mac_materialize_compiled = s.value("mac_materialize_compiled", mac_materialize_compiled).toBool();
	mac_commit_overlap_fast = s.value("mac_commit_overlap_fast", mac_commit_overlap_fast).toBool();
	mac_image_lookup_fast = s.value("mac_image_lookup_fast", mac_image_lookup_fast).toBool();
	mac_pervertex_unique_disabled = s.value("mac_pervertex_unique_disabled", mac_pervertex_unique_disabled).toBool();
	mac_moltenvk_path = s.value("mac_moltenvk_path", mac_moltenvk_path).toString();
	mac_extra_environment = s.value("mac_extra_environment", mac_extra_environment).toString();
#endif
#undef KYTY_CFG_GET
}

void Configuration::WriteSettings(QSettings* s) const {
	const auto values = GameSettings();
	for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
		s->setValue(it.key(), it.value());
	}
	s->setValue("name", name);
	s->setValue("basedir", basedir);
	s->setValue("game_path", game_path);
	s->setValue("custom_settings", QVariant::fromValue(custom_settings).toString());
	s->setValue("host_input_mapping", host_input_mapping);
}

void Configuration::ReadSettings(QSettings* s) {
	ReadGameSettingsValues(*s);
	name               = s->value("name").toString();
	basedir            = s->value("basedir").toString();
	game_path          = s->value("game_path").toString();
	custom_settings    = s->value("custom_settings").toBool();
	host_input_mapping = s->value("host_input_mapping", host_input_mapping).toStringList();
}

bool Configuration::SetGameSettings(const QJsonObject& settings, QString& error) {
	const auto fail = [&error](const QString& message) {
		error = message;
		return false;
	};
	auto values = GameSettings();
	for (auto it = settings.constBegin(); it != settings.constEnd(); ++it) {
		if (!values.contains(it.key())) {
			return fail(tr("Unknown game setting: %1").arg(it.key()));
		}
		if (!it.value().isString()) {
			return fail(tr("Game setting %1 must be a string.").arg(it.key()));
		}
		values.insert(it.key(), it.value().toString());
	}

	ReadGameSettingsValues(values);
	const auto canonical = GameSettings();
	for (auto it = settings.constBegin(); it != settings.constEnd(); ++it) {
		if (canonical.value(it.key()) != values.value(it.key())) {
			return fail(tr("Invalid value for game setting: %1").arg(it.key()));
		}
	}
	if (user_name.trimmed().isEmpty() || user_name.toUtf8().size() > Config::MAX_USER_NAME_LENGTH) {
		return fail(tr("User name must contain 1-16 UTF-8 bytes."));
	}
	if (gpu_index < -1) {
		return fail(tr("GPU index must be -1 or greater."));
	}
	if (vblank_frequency < 30 || vblank_frequency > 360) {
		return fail(tr("Vblank frequency must be between 30 and 360."));
	}
	if (trophy_duration_seconds < 5 || trophy_duration_seconds > 10) {
		return fail(tr("Trophy notification duration must be between 5 and 10 seconds."));
	}
	if ((shader_log_direction == LogDirection::File && shader_log_folder.isEmpty()) ||
	    (printf_direction == LogDirection::File && printf_output_file.isEmpty()) ||
	    (command_buffer_dump_enabled && command_buffer_dump_folder.isEmpty())) {
		return fail(tr("Enabled logging and command buffer dumps require an output path."));
	}
#if defined(__APPLE__)
	auto environment = QProcessEnvironment::systemEnvironment();
	if (!MacEnvironment(environment, error)) {
		return false;
	}
#endif

	return true;
}

#if defined(__APPLE__)
QString Configuration::MacRuntimeDirectory() {
	// A local INI keeps portable installations and test runs self-contained.
	const QFileInfo ini(QStringLiteral("Kyty.ini"));
	return (ini.isFile() ? ini.absolutePath() : QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation)) + "/KytyPS5";
}

QString Configuration::RecommendedMacEnvironment() {
	// Supporting switches from the adopted stack; omit diagnostics and benchmark cache paths.
	return QStringLiteral("KYTY_PERVERTEX_EXPERIMENTAL=1\nKYTY_SKIP_EMPTY_SUBMIT=1\nKYTY_SUBMIT_SPLIT_DRAWS=32\nMVK_CONFIG_PERVERTEX_PREPASS=1\nKYTY_COLOR_CLEAR_NO_READBACK=1\nMVK_CONFIG_PERVERTEX_BATCH_RESIDENCY=1\nMVK_CONFIG_PERVERTEX_SCRATCH_POOL=1\nKYTY_READBACK_CLEAN_BACKING=1\nMVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS=0\nKYTY_AMD_PATCH=1\nKYTY_INDIRECT_GPU=1\nKYTY_INDIRECT_GPU_MESH=1\nMVK_CONFIG_MESH_INDIRECT_GPU=1\nKYTY_READBACK_WINDOW_KIB=512\nMVK_CONFIG_PERVERTEX_UNIQUE_TG_SIMDS=8");
}

bool Configuration::MacEnvironment(QProcessEnvironment& environment, QString& error) const {
	auto result = environment;
	result.insert("KYTY_SRT_READBACK_ELIDE", mac_srt_readback_elide ? "1" : "0");
	result.insert("KYTY_MATERIALIZE_COMPILED", mac_materialize_compiled ? "1" : "0");
	result.insert("KYTY_COMMIT_OVERLAP_FAST", mac_commit_overlap_fast ? "1" : "0");
	result.insert("KYTY_IMAGE_LOOKUP_FAST", mac_image_lookup_fast ? "1" : "0");
	result.insert("MVK_CONFIG_PERVERTEX_CAPTURE_UNIQUE", mac_pervertex_unique_disabled ? "0" : "1");
	const QStringList managed = {"KYTY_SRT_READBACK_ELIDE", "KYTY_MATERIALIZE_COMPILED", "KYTY_COMMIT_OVERLAP_FAST", "KYTY_IMAGE_LOOKUP_FAST", "MVK_CONFIG_PERVERTEX_CAPTURE_UNIQUE", "SDL_VULKAN_LIBRARY"};
	const QRegularExpression key_pattern(QStringLiteral("^[A-Za-z_][A-Za-z0-9_]*$"));
	const auto lines = mac_extra_environment.split('\n');
	for (int index = 0; index < lines.size(); ++index) {
		auto line = lines.at(index);
		if (line.endsWith('\r')) {
			line.chop(1);
		}
		if (line.trimmed().isEmpty()) {
			continue;
		}
		const auto separator = line.indexOf('=');
		const auto key = line.left(separator);
		if (separator < 1 || !key_pattern.match(key).hasMatch() || line.contains(QChar::Null) || managed.contains(key)) {
			error = tr("Invalid environment line %1. Use KEY=VALUE; checkbox and MoltenVK variables are reserved.").arg(index + 1);
			return false;
		}
		result.insert(key, line.mid(separator + 1));
	}
	const auto path = mac_moltenvk_path.trimmed();
	if (path.contains(QChar::Null)) {
		error = tr("Invalid MoltenVK path.");
		return false;
	}
	const auto library = QDir::cleanPath(path.isEmpty() ? QCoreApplication::applicationDirPath() + "/../Frameworks/libMoltenVK.dylib" : QFileInfo(path).absoluteFilePath());
	if (!QFileInfo(library).isFile() || !QFileInfo(library).isReadable()) {
		error = tr("MoltenVK library is not readable: %1").arg(library);
		return false;
	}
	result.insert("SDL_VULKAN_LIBRARY", library);
	environment = result;
	error.clear();
	return true;
}
#endif
