"""Logging and crash reporting utilities for the LFO application."""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import ContextDecorator
import time
import functools

import faulthandler
from logging.handlers import TimedRotatingFileHandler
# Wird für Crash-Logging verwendet (Schreiben in mehrere Streams)
class _MultiStreamWriter:
    def __init__(self, *streams):
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass
from qtpy import QtCore, QtGui, QtWidgets


LOG_RETENTION_DAYS = 3
LOG_DIRECTORY_NAME = "LFO"
_FAULT_HANDLER_FILE = None
_FAULT_HANDLER_STREAM = None

# Einfacher, global steuerbarer Schalter für Performance‑Messungen
# Standard jetzt: Performance-Logging AUS, kann per Umgebungsvariable wieder aktiviert werden.
PERF_ENABLED = os.environ.get("LFO_DEBUG_PERF", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

__all__ = ["configure_logging", "CrashReporter", "PerfTimer", "perf_section", "measure_time"]


class PerfTimer(ContextDecorator):
    """
    Kontextmanager / Dekorator zur Zeitmessung.
    Nutzung:
        with PerfTimer("update_calculations"):
            ...
    oder:
        @measure_time("UiSources.on_x_position_changed")
        def handler(...):
            ...
    """

    def __init__(self, label: str, **context):
        """
        label: Klarer Name des Messpunkts, z.B. 'Main.calculate_spl' oder
               'UiSources.show_sources_tab'.
        context: Beliebige Schlüssel/Werte (speaker_array_id=..., plot_mode=..., usw.),
                 die zur besseren Zuordnung mit ausgegeben werden.
        """
        self.label = label
        self.context = context or {}
        self._start = None

    def __enter__(self):
        if not PERF_ENABLED:
            return self
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not PERF_ENABLED or self._start is None:
            return False
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        ctx = ""
        if self.context:
            # Kleine, einzeilige Kontextdarstellung, eindeutig zuordenbar
            # z.B. [speaker_array_id=4, plot_mode=SPL plot]
            pairs = [f"{k}={v}" for k, v in sorted(self.context.items())]
            ctx = " [" + ", ".join(pairs) + "]"
        print(f"[PERF] {self.label}: {duration_ms:.2f} ms{ctx}")
        return False


def perf_section(label: str, **context) -> PerfTimer:
    """
    Hilfsfunktion für kurze with‑Blöcke:
        with perf_section("Main.update_speaker_array_calculations", step="beamsteering"):
            ...
    """
    return PerfTimer(label, **context)


def measure_time(label: str | None = None):
    """
    Dekorator für Funktionen/Methoden.
    Beispiel:
        @measure_time("UiSources.on_y_position_changed")
        def on_y_position_changed(...):
            ...
    Wenn kein Label angegeben wird, wird qualname der Funktion verwendet.
    """

    def decorator(func):
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PERF_ENABLED:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                print(f"[PERF] {name}: {duration_ms:.2f} ms")

        return wrapper

    return decorator


def _get_log_directory() -> Path:
    base_path = Path(tempfile.gettempdir())
    log_dir = base_path / LOG_DIRECTORY_NAME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _cleanup_old_logs(log_dir: Path, retention_days: int = LOG_RETENTION_DAYS) -> None:
    threshold = datetime.now() - timedelta(days=retention_days)
    for file_path in log_dir.glob("*"):
        try:
            if file_path.is_file() and datetime.fromtimestamp(file_path.stat().st_mtime) < threshold:
                file_path.unlink()
        except OSError:
            logging.getLogger("lfo").warning("Unable to delete old log file: %s", file_path)


def configure_logging() -> Path:
    log_dir = _get_log_directory()
    _cleanup_old_logs(log_dir)

    log_file = log_dir / "lfo_app.log"
    handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    debug_mode = os.environ.get("LFO_DEBUG_PLOT", os.environ.get("LFO_DEBUG"))
    if debug_mode and debug_mode.strip().lower() in {"1", "true", "yes", "on"}:
        root_level = logging.DEBUG
    else:
        root_level = logging.INFO
    root_logger.setLevel(root_level)

    for existing_handler in list(root_logger.handlers):
        if isinstance(existing_handler, TimedRotatingFileHandler) and existing_handler.baseFilename == handler.baseFilename:
            root_logger.removeHandler(existing_handler)

    root_logger.addHandler(handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(root_level)
        root_logger.addHandler(console_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    global _FAULT_HANDLER_FILE, _FAULT_HANDLER_STREAM
    crash_dump_path = log_dir / "crashdump.txt"
    _FAULT_HANDLER_FILE = open(crash_dump_path, "a", encoding="utf-8")
    faulthandler.enable(_FAULT_HANDLER_FILE)
    faulthandler.enable(sys.stderr)

    return log_dir


class CrashReporter:
    """Collects crash information and assists the user in reporting issues."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

    def handle_exception(self, exc_type, exc_value, exc_traceback) -> None:
        logger = logging.getLogger("lfo.crash")
        logger.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

        app = QtWidgets.QApplication.instance()
        if app is None:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit(1)

        detailed_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        summary = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
        print("Unhandled exception:", summary, file=sys.stderr)
        print(detailed_trace, file=sys.stderr)
        self._show_dialog(summary, detailed_trace)

    def _show_dialog(self, summary: str, detailed_trace: str) -> None:
        app = QtWidgets.QApplication.instance()
        parent = app.activeWindow() if app else None

        dialog = QtWidgets.QMessageBox(parent)
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setWindowTitle("Unexpected Error")
        dialog.setText("The application encountered an unexpected error and needs to close.")
        dialog.setInformativeText(
            f"Error: {summary}\n"
            f"Log files were stored temporarily in {self.log_dir}."
        )
        dialog.setDetailedText(detailed_trace)
        send_button = dialog.addButton("Create crash report", QtWidgets.QMessageBox.ActionRole)
        open_button = dialog.addButton("Open log folder", QtWidgets.QMessageBox.ActionRole)
        dialog.addButton(QtWidgets.QMessageBox.Close)
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked == open_button:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.log_dir)))
        elif clicked == send_button:
            report_path = self._package_report()
            info_box = QtWidgets.QMessageBox(parent)
            info_box.setIcon(QtWidgets.QMessageBox.Information)
            info_box.setWindowTitle("Crash report bundle ready")
            info_box.setText("A crash report archive has been created.")
            info_box.setInformativeText(
                f"Archive: {report_path}\n"
                "Please send this file manually to the developer (e.g. via email)."
            )
            info_box.exec_()
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(report_path.parent)))

        QtWidgets.QApplication.quit()
        sys.exit(1)

    def _package_report(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        downloads_dir = Path.home() / "Downloads"
        try:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            downloads_dir = self.log_dir

        zip_path = downloads_dir / f"crash_report_{timestamp}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for log_file in self.log_dir.glob("*.log*"):
                archive.write(log_file, log_file.name)
            crash_dump = self.log_dir / "crashdump.txt"
            if crash_dump.exists():
                archive.write(crash_dump, crash_dump.name)
            system_info = (
                f"Timestamp: {datetime.now().isoformat()}\n"
                f"Platform: {platform.platform()}\n"
                f"Python version: {platform.python_version()}\n"
            )
            archive.writestr("system_info.txt", system_info)

        return zip_path


