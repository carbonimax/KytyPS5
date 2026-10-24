#include "configuration.h"
#include "configurationEditDialog.h"
#include "launcherTheme.h"
#include "mainDialog.h"
#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QSettings>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <cstdio>
#include <cstdlib>

static void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "MacLauncherTests: %s\n", message);
		std::abort();
	}
}

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);
	QTimer::singleShot(60000, []() { Check(false, "test exceeded 60 seconds"); });
	LauncherTheme::Initialize(app);
	QTemporaryDir temporary;
	Check(temporary.isValid(), "temporary directory");
	const auto evidence_path = [&temporary](const char* key, const char* filename) { const auto path = qEnvironmentVariable(key); return path.isEmpty() ? temporary.filePath(QString::fromLatin1(filename)) : path; };
	Check(QDir::setCurrent(temporary.path()), "isolated current directory");
	// The empty MoltenVK path resolves to ../Frameworks/libMoltenVK.dylib next to this executable; a build tree has none, so
	// provide an empty placeholder (only existence is checked) and remove it at the end. Never replace a real library.
	const auto frameworks = QDir::cleanPath(QCoreApplication::applicationDirPath() + "/../Frameworks");
	QFile bundled(frameworks + "/libMoltenVK.dylib");
	const bool bundled_placeholder = !bundled.exists();
	Check(!bundled_placeholder || (QDir().mkpath(frameworks) && bundled.open(QIODevice::WriteOnly)), "bundled MoltenVK placeholder");
	bundled.close();
	QSettings settings(temporary.filePath("Kyty.ini"), QSettings::IniFormat);
	Configuration defaults;
	Configuration legacy;
	legacy.ReadSettings(&settings);
	Check(legacy.mac_srt_readback_elide && legacy.mac_materialize_compiled && legacy.mac_commit_overlap_fast && legacy.mac_image_lookup_fast && legacy.mac_pervertex_unique_disabled, "old settings retain Mac defaults");
	Configuration changed;
	changed.mac_srt_readback_elide = false;
	changed.mac_materialize_compiled = false;
	changed.mac_commit_overlap_fast = false;
	changed.mac_image_lookup_fast = false;
	changed.mac_pervertex_unique_disabled = false;
	changed.mac_extra_environment = "LAUNCHER_TEST_VALUE= first=last \r\nLAUNCHER_TEST_EMPTY=\nLAUNCHER_TEST_LITERAL=$(echo nope)\nLAUNCHER_TEST_REPEAT=first\nLAUNCHER_TEST_REPEAT=second\n\n";
	changed.WriteSettings(&settings);
	settings.sync();
	Configuration restored;
	restored.ReadSettings(&settings);
	Check(restored.GameSettings() == changed.GameSettings(), "settings roundtrip includes multiline text and false values");
	Configuration copied;
	copied.CopyEmulatorSettingsFrom(restored);
	Check(copied.GameSettings() == restored.GameSettings(), "global-to-game copy");
	Configuration imported;
	QString error;
	Check(imported.SetGameSettings(QJsonObject::fromVariantMap(changed.GameSettings()), error), "JSON import");
	Check(imported.GameSettings() == changed.GameSettings(), "JSON roundtrip");
	Check(!imported.SetGameSettings(QJsonObject {{"mac_extra_environment", "BROKEN"}}, error), "JSON rejects malformed environment");
	QProcessEnvironment environment;
	environment.insert("KYTY_IMAGE_LOOKUP_FAST", "2");
	environment.insert("LAUNCHER_TEST_INHERITED", "yes");
	Check(restored.MacEnvironment(environment, error), "valid environment");
	Check(environment.value("KYTY_IMAGE_LOOKUP_FAST") == "0" && environment.value("KYTY_SRT_READBACK_ELIDE") == "0" && environment.value("KYTY_MATERIALIZE_COMPILED") == "0" && environment.value("KYTY_COMMIT_OVERLAP_FAST") == "0", "unchecked overrides inherited flags");
	Check(environment.value("MVK_CONFIG_PERVERTEX_CAPTURE_UNIQUE") == "1", "unchecked UNIQUE=0 selects 1");
	Check(environment.value("LAUNCHER_TEST_VALUE") == " first=last " && environment.value("LAUNCHER_TEST_EMPTY").isEmpty() && environment.value("LAUNCHER_TEST_LITERAL") == "$(echo nope)" && environment.value("LAUNCHER_TEST_REPEAT") == "second" && environment.value("LAUNCHER_TEST_INHERITED") == "yes", "literal values, empty, equals, spaces, duplicate and inheritance");
	for (const auto& invalid: QStringList {"NO_EQUALS", "BAD-KEY=1", "=empty", "KYTY_IMAGE_LOOKUP_FAST=2", "SDL_VULKAN_LIBRARY=/override", QString("LAUNCHER_TEST_BAD=x") + QChar::Null}) {
		restored.mac_extra_environment = invalid;
		const auto before = environment;
		Check(!restored.MacEnvironment(environment, error) && environment == before && !error.isEmpty(), "invalid environment fails atomically");
	}
	restored.mac_extra_environment.clear();
	restored.mac_moltenvk_path = temporary.filePath("missing.dylib");
	Check(!restored.MacEnvironment(environment, error), "missing custom dylib");
	QFile library(temporary.filePath("custom library.dylib"));
	Check(library.open(QIODevice::WriteOnly), "test library path");
	library.close();
	changed.mac_moltenvk_path = library.fileName();
	Check(changed.MacEnvironment(environment, error) && environment.value("SDL_VULKAN_LIBRARY") == library.fileName(), "custom dylib with spaces");
	ConfigurationEditDialog dialog(changed);
	dialog.SetGlobalSettings({temporary.path()});
	auto* tabs = dialog.findChild<QTabWidget*>("settings_tabs");
	const int mac_tab = tabs != nullptr ? tabs->count() - 1 : -1;
	Check(tabs != nullptr && dialog.findChildren<QTabWidget*>().size() == 1 && tabs->tabText(mac_tab) == "macOS / performance", "Mac tab visible");
	dialog.findChild<QLineEdit*>("lineEdit_user_name")->setText("test user");
	tabs->setCurrentIndex(mac_tab);
	dialog.show();
	QTest::qWait(100);
	Check(!dialog.findChild<QCheckBox*>("mac_elide")->isChecked(), "UI restores false");
	QTest::mouseClick(dialog.findChild<QPushButton*>("mac_recommended"), Qt::LeftButton);
	Check(dialog.findChild<QCheckBox*>("mac_elide")->isChecked() && dialog.findChild<QCheckBox*>("mac_compiled")->isChecked() && dialog.findChild<QCheckBox*>("mac_overlap")->isChecked() && dialog.findChild<QCheckBox*>("mac_findimage")->isChecked() && dialog.findChild<QCheckBox*>("mac_unique")->isChecked(), "recommended checks all five");
	Check(dialog.findChild<QLineEdit*>("mac_moltenvk")->text().isEmpty() && dialog.findChild<QPlainTextEdit*>("mac_environment")->toPlainText() == Configuration::RecommendedMacEnvironment(), "recommended path and supporting stack");
	Check(dialog.grab().save(evidence_path("LAUNCHER_TEST_SCREENSHOT", "mac-performance.png")), "Mac options screenshot");
	QTest::mouseClick(dialog.findChild<QPushButton*>("ok_button"), Qt::LeftButton);
	defaults.user_name = "test user";
	Check(dialog.result() == QDialog::Accepted && changed.GameSettings() == defaults.GameSettings(), "UI save includes recommendations and preserves General edits");

	// Only this test directory gets a fake helper; never replace the delivered emulator.
	const auto fake_path = QCoreApplication::applicationDirPath() + "/kyty_emulator";
	Check(!QFile::exists(fake_path), "fake helper cannot overwrite an existing executable");
	QFile fake(fake_path);
	Check(fake.open(QIODevice::WriteOnly), "create fake helper");
	fake.write("#!/bin/sh\nif [ $# -eq 0 ]; then printf 'fake helper\\nversion fake\\n'; exit 0; fi\nprintf 'ELIDE=%s\\nCOMPILED=%s\\nOVERLAP=%s\\nFINDIMAGE=%s\\nUNIQUE=%s\\nLIBRARY=%s\\nLITERAL=%s\\n' \"$KYTY_SRT_READBACK_ELIDE\" \"$KYTY_MATERIALIZE_COMPILED\" \"$KYTY_COMMIT_OVERLAP_FAST\" \"$KYTY_IMAGE_LOOKUP_FAST\" \"$MVK_CONFIG_PERVERTEX_CAPTURE_UNIQUE\" \"$SDL_VULKAN_LIBRARY\" \"$LAUNCHER_TEST_LITERAL\"\nprintf 'ARG=%s\\n' \"$@\"\n");
	fake.close();
	Check(fake.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner), "fake helper executable");
	settings.setValue("MainDialog/check_updates_on_startup", false);
	settings.setValue("Launcher/game_dirs", QStringList {temporary.path()});
	settings.sync();
	MainDialog launcher;
	launcher.Start();
	QTest::qWait(200);
	launcher.show();
	QTest::qWait(100);
	Check(launcher.grab().save(evidence_path("LAUNCHER_TEST_LAUNCHER_SCREENSHOT", "launcher.png")), "launcher screenshot");
	changed.mac_extra_environment += "\nLAUNCHER_TEST_LITERAL=$(echo nope)";
	changed.basedir = temporary.filePath("fake game directory");
	QProcess process;
	launcher.RunInterpreter(&process, changed, false);
	if (process.state() != QProcess::NotRunning) {
		Check(process.waitForFinished(5000), "fake launch terminates");
	}
	const auto output = process.readAllStandardOutput();
	Check(process.exitCode() == 0 && output.contains("ELIDE=1\nCOMPILED=1\nOVERLAP=1\nFINDIMAGE=1\nUNIQUE=0\n") && output.contains("LITERAL=$(echo nope)\n") && output.contains("fake game directory"), "production QProcess path transmits environment and arguments literally");
	std::printf("Runtime actual=%s expected=%s\n", qPrintable(process.workingDirectory()), qPrintable(temporary.filePath("KytyPS5")));
	Check(QFileInfo(process.workingDirectory()).canonicalFilePath() == QFileInfo(temporary.filePath("KytyPS5")).canonicalFilePath() && QDir(process.workingDirectory()).exists(), "runtime uses portable data folder outside the bundle");
	QFile evidence(evidence_path("LAUNCHER_TEST_ENV_EVIDENCE", "fake-launch.txt"));
	Check(evidence.open(QIODevice::WriteOnly) && evidence.write(output) == output.size(), "environment evidence");
	Check(fake.remove(), "remove fake helper");
	if (bundled_placeholder) {
		Check(bundled.remove(), "remove bundled MoltenVK placeholder");
		QDir().rmdir(frameworks);
	}
	std::puts("PASS: persistence, legacy defaults, JSON/copy, environment validation, Mac UI/recommended/save, production fake launch; no game executed.");
	return 0;
}
