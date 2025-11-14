import sys
import cProfile
import logging
import traceback
import os
import hashlib
from time import perf_counter
from typing import Optional

import numpy as np
from qtpy import QtWidgets, QtCore
from PyQt5.QtCore import Qt

import matplotlib as mpl
mpl.set_loglevel('WARNING')  # Reduziert matplotlib Debug-Ausgaben

ENABLE_PERFORMANCE_LOGS = False
if ENABLE_PERFORMANCE_LOGS:
    os.environ.setdefault("LFO_DEBUG_PERF", "1")
else:
    os.environ["LFO_DEBUG_PERF"] = "0"

from Module_LFO.Modules_Init.Logging import configure_logging, CrashReporter
from Module_LFO.Modules_Init.Progress import ProgressManager

from Module_LFO.Modules_Ui.UiSourceManagement import Sources
from Module_LFO.Modules_Ui.UiImpulsemanager import ImpulseManager
from Module_LFO.Modules_Ui.UiSettings import UiSettings
from Module_LFO.Modules_Ui.UiFile import UiFile
from Module_LFO.Modules_Window.Mainwindow import Ui_MainWindow
from Module_LFO.Modules_Ui.UiManageSpeaker import UiManageSpeaker
from Module_LFO.Modules_Ui.UISurfaceManager import UISurfaceManager

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator
from Module_LFO.Modules_Calculate.SoundFieldCalculator_FEM import SoundFieldCalculatorFEM
from Module_LFO.Modules_Calculate.SoundfieldCalculatorXaxis import SoundFieldCalculatorXaxis
from Module_LFO.Modules_Calculate.SoundfieldCalculatorYaxis import SoundFieldCalculatorYaxis
from Module_LFO.Modules_Calculate.WindowingCalculator import WindowingCalculator
from Module_LFO.Modules_Calculate.BeamSteering import BeamSteering
from Module_LFO.Modules_Calculate.PolarPlotCalculator import PolarPlotCalculator
from Module_LFO.Modules_Calculate.BandwidthCalculator import BandwidthCalculator
from Module_LFO.Modules_Calculate.SpeakerPositionCalculator import SpeakerPositionCalculator

from Module_LFO.Modules_Data.data_module import DataContainer
from Module_LFO.Modules_Data.settings_state import Settings

from Module_LFO.Modules_Window.WindowPlotsMainwindow import DrawPlotsMainwindow
from Module_LFO.Modules_Window.WindowWidgets import DrawWidgets
from Module_LFO.Modules_Window.WindowSnapshotWidget import SnapshotWidget

"to Dos"

"Logik"
"Autosafe"

"UI"
"source window: bild generieren"

"Layout"

"Funktionalität"
"Image Source noch nicht fertig"


class _PerfScope:
    def __init__(self, owner: "MainWindow", label: str):
        self._owner = owner
        self._label = label
        self._start = 0.0

    def __enter__(self):
        if getattr(self._owner, "_perf_enabled", False):
            self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self._owner, "_perf_enabled", False):
            duration = perf_counter() - self._start
            self._owner._log_perf(self._label, duration)


class MainWindow(QtWidgets.QMainWindow):
    """
    Hauptfenster der Anwendung.

    Diese Klasse repräsentiert das Hauptfenster der Anwendung und verwaltet
    die Benutzeroberfläche sowie die Interaktionen mit anderen Modulen.

    Attribute:
        settings (Settings): Einstellungsobjekt für die Anwendung.
        data_container (DataContainer): Container für Anwendungsdaten.
    """

    # ---- INIT

    def __init__(self, settings, container, parent=None):
        """
        Initialisiert das MainWind
        Args:
            settings (Settings): Einstellungsobjekt für die Anwendung.
            data_container (DataContainer): Container für Anwendungsdaten.
            parent (QWidget, optional): Übergeordnetes Widget. Standardmäßig None.
        """
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.settings = settings
        self.container = container
        
        # Übergeben Sie settings und data_container an andere Klassen
        # self.sources_instance = Sources(self, self.settings, self.container)
        self.ui_settings = UiSettings(self, self.settings, self.container.data)
        self.draw_plots = DrawPlotsMainwindow(self, self.settings, self.container)
        self.draw_widgets = DrawWidgets(self, self.settings, self.container)
        self.ui_file = UiFile(self, self.settings, self.container)
        self.impulse_manager = ImpulseManager(self, self.settings, self.container)
        self.snapshot_engine = SnapshotWidget(self, self.settings, self.container)
        self.surface_manager = None  # Surface-Dock wird nur bei Bedarf erstellt
        self.progress_manager = ProgressManager(self)
        self._perf_enabled = ENABLE_PERFORMANCE_LOGS and bool(int(os.environ.get("LFO_DEBUG_PERF", "1")))
        self._perf_logger = logging.getLogger("LFO.Performance")
        self._fem_unavailable_notified = False
        
        # Tracking für physische Speaker-Parameter
        self._speaker_position_hashes = {}

        self._initialize_default_plot_flags()

        
        # Verbinde die Aktionen mit den entsprechenden Methoden von UiFile
        self.ui.actionLoad_2.triggered.connect(self.ui_file.load_file)
        self.ui.actionSave_as_2.triggered.connect(self.ui_file.save_file_as)
        self.ui.actionSave_2.triggered.connect(self.ui_file.overwrite_file)
        self.ui.actionNew_2.triggered.connect(self.ui_file.new_pa_file)
        self.ui.actionReset_to_Factory_Settings.triggered.connect(self.ui_file.reset_to_factory)

        # ---- Eingabe aus File
        # Settings Window öffnen
        self.ui.actionSettings_window.triggered.connect(self.ui_settings.open_settings_window)

        # ---- Eingabe aus Window
        # zeige dock widgets
        self.ui.actionSourceLayout_Window.triggered.connect(self.draw_widgets.show_source_layout_widget)
        self.ui.actionImpulse_Window.triggered.connect(self.impulse_manager.show_impulse_input_dock_widget)
        self.ui.actionSurface_Window.triggered.connect(self._show_surface_dock_lazy)

        # zeige Snapshot window
        self.ui.actionCourves_Window.triggered.connect(self.snapshot_engine.show_snapshot_widget)
        
        # zeige speakerspecs window
        self.ui.actionSpeakerSpecs_Window.triggered.connect(self.show_sources_dock_widget)

        # Verbinden Sie den Button mit der starte_skript-Methode
        self.ui.StartSkript.clicked.connect(self.starte_skript)

        # Splitter Positionen
        self.ui.actionDefault_View.triggered.connect(self.draw_plots.set_default_view)
        self.ui.actionFocus_SPL.triggered.connect(self.draw_plots.set_focus_spl)
        self.ui.actionFocus_Polar.triggered.connect(self.draw_plots.set_focus_polar)
        self.ui.actionFocus_Xaxis.triggered.connect(self.draw_plots.set_focus_xaxis)
        self.ui.actionFocus_Yaxis.triggered.connect(self.draw_plots.set_focus_yaxis)

        # Manage Speaker Window initialisieren
        self.manage_speaker_window = None
        
        # Verbinde die Action mit der Methode
        self.ui.actionManage_Speaker.triggered.connect(self.show_manage_speaker)
        
    # ----  PLOTS MAINWINDOW

    def plot_spl(self, update_axes: bool = True, reset_camera: bool = False):
        """Zeichnet den SPL-Plot (Sound Pressure Level)."""
        with self._perf_scope("Plot SPL total"):
            if not self.container.calculation_spl.get("show_in_plot", True):
                self.draw_plots.show_empty_spl()
                if update_axes:
                    self.draw_plots.show_empty_axes()
                    self.draw_plots.show_empty_polar()
                return
            speaker_array_id = self.get_selected_speaker_array_id()
            with self._perf_scope("Plot SPL draw"):
                self.draw_plots.plot_spl(self.settings, speaker_array_id, update_axes=update_axes, reset_camera=reset_camera)
        
    def plot_xaxis(self):
        with self._perf_scope("Plot axis X total"):
            self.draw_plots.plot_xaxis()
        
    def plot_yaxis(self):
        with self._perf_scope("Plot axis Y total"):
            self.draw_plots.plot_yaxis()

    def plot_polar_pattern(self):
        with self._perf_scope("Plot polar total"):
            polar_data = self.container.get_polar_data()
            if not polar_data or not polar_data.get("show_in_plot", True):
                self.draw_plots.show_empty_polar()
                return
            self.draw_plots.plot_polar_pattern()

    def init_plot(self, preserve_camera: bool = True, view: Optional[str] = None):
        print("DEBUG: init_plot called")
        """
        Initialisiert alle Plotbereiche mit einer leeren Darstellung.

        Args:
            preserve_camera (bool): Wenn True, bleibt die aktuelle Kameraperspektive erhalten.
        """
        if hasattr(self, 'draw_plots'):
            QtCore.QTimer.singleShot(
                0,
                lambda: self.draw_plots.show_empty_plots(
                    preserve_camera=preserve_camera,
                    view=view,
                ),
            )

    def _initialize_default_plot_flags(self) -> None:
        axes_data = self.container.calculation_axes.setdefault("aktuelle_simulation", {})
        if "show_in_plot" not in axes_data:
            axes_data["show_in_plot"] = True
        axes_data.setdefault("color", "#6A5ACD")

        if "show_in_plot" not in self.container.calculation_spl:
            self.container.calculation_spl["show_in_plot"] = True
        if "show_in_plot" not in self.container.calculation_polar:
            self.container.calculation_polar["show_in_plot"] = True


    # ---- SPLITTER

    def set_default_view(self):
        self.draw_plots.set_default_view()
        
    def set_focus_spl(self):
        self.draw_plots.set_focus_spl()
        
    def set_focus_polar(self):
        self.draw_plots.set_focus_polar()
        
    def set_focus_xaxis(self):
        self.draw_plots.set_focus_xaxis()
        
    def set_focus_yaxis(self):
        self.draw_plots.set_focus_yaxis()


    # ---- WIDGETS
     
    def show_sources_dock_widget(self):
        if not hasattr(self, 'sources_instance'):
            self.sources_instance = Sources(self, self.settings, self.container)
            self._ensure_surface_manager().bind_to_sources_widget(self.sources_instance)
        else:
            if self.surface_manager is not None:
                self.surface_manager.bind_to_sources_widget(self.sources_instance)

        self.sources_instance.show_sources_dock_widget()

    def _ensure_surface_manager(self):
        if self.surface_manager is None:
            self.surface_manager = UISurfaceManager(self, self.settings, self.container)
            if hasattr(self, 'sources_instance'):
                self.surface_manager.bind_to_sources_widget(self.sources_instance)
        return self.surface_manager

    def show_manage_speaker(self):
        """Zeigt das Manage Speaker Fenster an oder erstellt es, falls es noch nicht existiert."""
        if self.manage_speaker_window is None:
            self.manage_speaker_window = UiManageSpeaker(parent=self, settings=self.settings,
                                container=self.container, main_window=self)
        self.manage_speaker_window.show()


    # ---- CALCULATIONS
        
    def starte_skript(self):
        """
        Startet die Hauptberechnungen und Aktualisierungen.

        Diese Methode führt gezielt die Berechnungen für deaktivierte
        Auto-Updates aus, damit sie manuell nachgezogen werden können.
        """
        update_soundfield = getattr(self.settings, "update_pressure_soundfield", True)
        update_axisplot = getattr(self.settings, "update_pressure_axisplot", True)
        update_polarplot = getattr(self.settings, "update_pressure_polarplot", True)
        update_impulse = getattr(self.settings, "update_pressure_impulse", True)

        tasks = []

        if not update_axisplot:
            tasks.append((
                "Calculating Axes",
                lambda: self._run_with_log(
                    "Manual axes calculation",
                    self.calculate_axes,
                    post=lambda: self._set_axes_show_flag(True),
                ),
            ))

        if not update_polarplot:
            tasks.append((
                "Calculating Polar",
                lambda: self._run_with_log(
                    "Manual polar calculation",
                    self.calculate_polar,
                    post=lambda: self._set_polar_show_flag(True),
                ),
            ))

        if not update_soundfield:
            tasks.append((
                "Calculating SPL",
                lambda: self._run_with_log(
                    "Manual SPL calculation",
                    lambda: self.calculate_spl(
                        show_progress=False,
                        update_axisplots=not update_axisplot,
                        update_polarplots=not update_polarplot,
                    ),
                    post=lambda: self._set_spl_show_flag(True),
                ),
            ))

        if hasattr(self, 'impulse_manager'):
            if not update_impulse and self.settings.impulse_points:
                tasks.append((
                    "Calculating Impulse",
                    lambda: self._run_with_log(
                        "Manual impulse calculation",
                        self.calculate_impulse,
                        post=lambda: self._log_async("Impulse"),
                        args=(True,),
                    ),
                ))
            elif not update_impulse:
                tasks.append((
                    "Clearing impulse plot",
                    lambda: self._run_with_log("Manual impulse clear", self.impulse_manager.show_empty_plot),
                ))

        if not tasks:
            return

        self.run_tasks_with_progress("Running main calculations", tasks)

    def update_widgets(self):
        self.draw_widgets.update_source_layout_widget()

    def update_beamsteering_wdw_plot(self):
        speaker_array_id = self.get_selected_speaker_array_id()
        speakerspecs_instance = self.sources_instance.get_speakerspecs_instance(speaker_array_id)
        if speakerspecs_instance:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                self.beamsteering_calculator(speaker_array, speaker_array_id)
                self.windowing_calculator(speaker_array_id)
                self.sources_instance.update_beamsteering_plot(speaker_array_id)
                self.sources_instance.update_windowing_plot(self.container.calculation_windowing, speaker_array_id)

    def update_axes_spl_plot(self):
        self.update_speaker_array_calculations()


    def update_freq_bandwidth(self):
        lower_freq = self.settings.lower_calculate_frequency
        upper_freq = self.settings.upper_calculate_frequency
        
        # # Erstelle eine Instanz des Calculators und berechne beide
        bandwidth_calculator = BandwidthCalculator(self.settings, self.container)  # 🚀 OPTIMIERT: DataContainer-Objekt
        bandwidth_calculator.calculate_magnitude_average(lower_freq, upper_freq)
        bandwidth_calculator.calculate_phase_average(lower_freq, upper_freq)
        
        # 🎯 TERZBAND-MITTELUNG: Berechne für alle Polar-Frequenzen
        bandwidth_calculator.calculate_polar_frequency_averages(self.settings.polar_frequencies)

        self.update_speaker_array_calculations()


    def update_speaker_array_calculations(self):
        """
        Aktualisiert die Berechnungen für das ausgewählte Lautsprecherarray.
        """
        if not hasattr(self, 'sources_instance'):
            return

        speaker_array_id = self.get_selected_speaker_array_id()
        speakerspecs_instance = self.sources_instance.get_speakerspecs_instance(speaker_array_id)
        
        if speakerspecs_instance:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                snapshot_engine = getattr(self, "snapshot_engine", None)
                if snapshot_engine:
                    # Backups im Snapshot-Widget verwerfen
                    snapshot_engine._current_spl_backup = None
                    snapshot_engine._current_polar_backup = None
                    snapshot_engine._current_axes_backup = None
                    snapshot_engine._current_impulse_backup = None

                    # Container-Daten für aktuelle Simulation zurücksetzen
                    if hasattr(self.container, "calculation_axes"):
                        self.container.calculation_axes["aktuelle_simulation"] = {"show_in_plot": True}

                    if hasattr(self.container, "calculation_impulse"):
                        self.container.calculation_impulse["aktuelle_simulation"] = {}

                    if hasattr(self.container, "calculation_spl"):
                        try:
                            self.container.calculation_spl.clear()
                        except AttributeError:
                            self.container.calculation_spl = {}

                    if hasattr(self.container, "calculation_polar"):
                        try:
                            self.container.calculation_polar.clear()
                        except AttributeError:
                            self.container.calculation_polar = {}

                    snapshot_engine.selected_snapshot_key = "aktuelle_simulation"
                    if snapshot_engine.snapshot_tree_widget:
                        blocker = QtCore.QSignalBlocker(snapshot_engine.snapshot_tree_widget)
                        snapshot_engine.update_snapshot_widgets()
                        del blocker
                update_soundfield = getattr(self.settings, "update_pressure_soundfield", True)
                update_axisplot = getattr(self.settings, "update_pressure_axisplot", True)
                update_polarplot = getattr(self.settings, "update_pressure_polarplot", True)
                update_impulse = getattr(self.settings, "update_pressure_impulse", True)

                def _clear_axes():
                    self._set_axes_show_flag(False)
                    self.draw_plots.show_empty_axes()

                def _clear_polar():
                    self._set_polar_show_flag(False)
                    self.draw_plots.show_empty_polar()

                def _clear_spl():
                    self._set_spl_show_flag(False)
                    self.draw_plots.show_empty_spl()

                run_spl = not getattr(self.settings, "spl_plot_fem", False)
                run_xaxis = not getattr(self.settings, "xaxis_plot_fem", False)
                run_yaxis = not getattr(self.settings, "yaxis_plot_fem", False)
                run_polar = not getattr(self.settings, "polar_plot_fem", False)

                tasks = [
                    ("Updating speaker positions", lambda: self._run_with_log("Auto speaker positions", self.speaker_position_calculator, post=lambda: None, args=(speaker_array,))),
                    ("Calculating beam steering", lambda: self._run_with_log("Auto beam steering", self.beamsteering_calculator, post=lambda: None, args=(speaker_array, speaker_array_id))),
                    ("Calculating windowing", lambda: self._run_with_log("Auto windowing", self.windowing_calculator, post=lambda: None, args=(speaker_array_id,))),
                ]

                if update_axisplot:
                    tasks.append(("Calculating axes", lambda: self._run_with_log(
                        "Auto axes calculation",
                        self.calculate_axes,
                        post=lambda: self._set_axes_show_flag(True),
                    )))
                else:
                    tasks.append(("Clearing axis plots", _clear_axes))

                if update_polarplot:
                    tasks.append(("Calculating polar pattern", lambda: self._run_with_log(
                        "Auto polar calculation",
                        self.calculate_polar,
                        post=lambda: self._set_polar_show_flag(True),
                    )))
                else:
                    tasks.append(("Clearing polar plot", _clear_polar))

                if update_soundfield:
                    tasks.append((
                        "Calculating SPL",
                        lambda: self._run_with_log(
                            "Auto SPL calculation",
                            lambda: self.calculate_spl(
                                show_progress=False,
                                update_axisplots=False,
                                update_polarplots=False,
                            ),
                            post=lambda: self._set_spl_show_flag(True),
                        ),
                    ))
                else:
                    tasks.append(("Clearing SPL plot", _clear_spl))

                if hasattr(self, 'impulse_manager'):
                    if update_impulse and getattr(self.settings, "impulse_points", []):
                        tasks.append((
                            "Calculating impulse",
                            lambda: self._run_with_log(
                                "Auto impulse calculation",
                                self.calculate_impulse,
                                post=lambda: self._log_async("Impulse"),
                                args=(True,),
                            ),
                        ))
                    elif not update_impulse:
                        tasks.append((
                            "Clearing impulse plot",
                            lambda: self._run_with_log("Auto impulse clear", self.impulse_manager.show_empty_plot),
                        ))

                self.run_tasks_with_progress("Updating array calculations", tasks)

                if not update_axisplot:
                    self.draw_plots.show_empty_axes()
                if not update_polarplot:
                    self.draw_plots.show_empty_polar()
                if not update_soundfield:
                    self.draw_plots.show_empty_spl()

                if update_soundfield or update_polarplot or update_impulse:
                    with self._perf_scope("Beamsteering plot update"):
                        self.sources_instance.update_beamsteering_plot(speaker_array_id)
                    self._log_async("Beamsteering plot")
                    with self._perf_scope("Windowing plot update"):
                        self.sources_instance.update_windowing_plot(self.container.calculation_windowing, speaker_array_id)

    def run_tasks_with_progress(self, title, tasks):
        if not tasks:
            return

        with self._perf_scope(f"Progress {title} (total)"):
            with self.progress_manager.start(title, len(tasks)) as progress:
                for label, func in tasks:
                    progress.update(label)
                    with self._perf_scope(f"TASK {label}"):
                        func()
                    progress.advance()
        self._log_async(f"{title}")

    def _perf_scope(self, label: str) -> _PerfScope:
        return _PerfScope(self, label)

    def _log_perf(self, label: str, duration: float) -> None:
        if self._perf_enabled:
            self._perf_logger.info("[PERF] %s: %.3f s", label, duration)

    def _log_async(self, label: str) -> None:
        if not self._perf_enabled:
            return
        start = perf_counter()
        QtCore.QTimer.singleShot(0, lambda: self._log_perf(f"{label} UI idle", perf_counter() - start))

    def _run_with_log(self, label: str, func, post=None, args=()):
        result = func(*args)
        if callable(post):
            post()
        return result

    def _set_axes_show_flag(self, value: bool):
        axes_data = self.container.calculation_axes.setdefault("aktuelle_simulation", {})
        axes_data["show_in_plot"] = bool(value)

    def _set_polar_show_flag(self, value: bool):
        self.container.calculation_polar["show_in_plot"] = bool(value)

    def _set_spl_show_flag(self, value: bool):
        self.container.calculation_spl["show_in_plot"] = bool(value)

    def _is_fem_runtime_error(self, exc: Exception) -> bool:
        """
        Erkennt zur FEM gehörende Laufzeitfehler (insbesondere Kompilierungsfehler),
        bei denen ein Fallback auf die Superpositionsberechnung sinnvoll ist.
        """
        visited = set()
        markers = ("dolfinx", "ffcx", "cffi", "VerificationError", "CompileError", "clang", "ffibuilder", "CalledProcessError")

        current = exc
        while current is not None and current not in visited:
            visited.add(current)
            summary = f"{current.__class__.__name__}: {current}".lower()
            if any(marker.lower() in summary for marker in markers):
                return True
            current = current.__cause__ or current.__context__

        return False

    def _select_soundfield_calculator_class(self):
        use_fem = getattr(self.settings, "spl_plot_fem", False)
        use_superposition = getattr(self.settings, "spl_plot_superposition", False)

        if use_fem and not use_superposition:
            return SoundFieldCalculatorFEM

        if use_fem and use_superposition:
            return SoundFieldCalculatorFEM

        if use_superposition:
            return SoundFieldCalculator

        return SoundFieldCalculator

    def _handle_fem_unavailable(self, exc: Exception) -> None:
        if isinstance(exc, ImportError):
            message = (
                "Die FEM-Berechnung konnte nicht gestartet werden, da notwendige Bibliotheken fehlen.\n\n"
                "Bitte installieren Sie FEniCSx (dolfinx, ufl, mpi4py, petsc4py) oder "
                "wechseln Sie zur Superpositionsberechnung.\n\n"
            )
        else:
            message = (
                "Die FEM-Berechnung ist aufgrund eines Kompilierungs- oder Laufzeitfehlers fehlgeschlagen.\n\n"
                "Die Anwendung schaltet automatisch auf die Superpositionsberechnung um, damit die SPL-Berechnung "
                "fortgeführt werden kann.\n\n"
                "Überprüfen Sie die Installation von FEniCSx und stellen Sie sicher, dass der verwendete Pfad keine "
                "Leerzeichen enthält (z.B. verschieben Sie die FEM-Umgebung aus Cloud-Verzeichnissen mit Leerzeichen "
                "oder verwenden Sie einen symbolischen Link ohne Leerzeichen).\n\n"
            )
        details = str(exc).strip()
        if details:
            message += f"Technische Details:\n{details}"

        if not getattr(self, "_fem_unavailable_notified", False):
            QtWidgets.QMessageBox.warning(
                self,
                "FEM-Berechnung nicht verfügbar",
                message,
            )
        else:
            print("[WARN] FEM-Berechnung deaktiviert – Superposition wird verwendet.")

        self.settings.spl_plot_fem = False
        self.settings.spl_plot_superposition = True
        self._fem_unavailable_notified = True

        if hasattr(self, "ui_settings"):
            if hasattr(self.ui_settings, "spl_plot_fem"):
                with QtCore.QSignalBlocker(self.ui_settings.spl_plot_fem):
                    self.ui_settings.spl_plot_fem.setChecked(False)
            if hasattr(self.ui_settings, "spl_plot_superposition"):
                with QtCore.QSignalBlocker(self.ui_settings.spl_plot_superposition):
                    self.ui_settings.spl_plot_superposition.setChecked(True)

    def calculate_spl(self, show_progress: bool = True, update_axisplots: bool = True, update_polarplots: bool = True):
        """
        Führt die SPL-Berechnung durch und aktualisiert den Plot.

        Diese Methode berechnet das Schalldruckfeld und aktualisiert den SPL-Plot.
        """
        def _compute_spl():
            with self._perf_scope("SPL total"):
                calculator_cls = self._select_soundfield_calculator_class()
                with self._perf_scope("SPL init calculator"):
                    calculator_instance = calculator_cls(self.settings, self.container.data, self.container.calculation_spl)
                    calculator_instance.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!

                def _run_superposition_fallback(error: Exception):
                    self._handle_fem_unavailable(error)
                    fallback_cls = SoundFieldCalculator
                    with self._perf_scope("SPL init calculator (fallback)"):
                        fallback_instance = fallback_cls(self.settings, self.container.data, self.container.calculation_spl)
                        fallback_instance.set_data_container(self.container)
                    with self._perf_scope("SPL sound field calculation (fallback)"):
                        fallback_instance.calculate_soundfield_pressure()
                    return fallback_instance

                try:
                    with self._perf_scope("SPL sound field calculation"):
                        calculator_instance.calculate_soundfield_pressure()
                except ImportError as exc:
                    if calculator_cls is SoundFieldCalculatorFEM:
                        calculator_instance = _run_superposition_fallback(exc)
                    else:
                        raise
                except Exception as exc:
                    if calculator_cls is SoundFieldCalculatorFEM and self._is_fem_runtime_error(exc):
                        calculator_instance = _run_superposition_fallback(exc)
                    else:
                        raise
                with self._perf_scope("SPL store results"):
                    self.container.set_calculation_SPL(calculator_instance.calculation_spl)
                    self._set_spl_show_flag(True)
                with self._perf_scope("SPL plot"):
                    self.plot_spl(update_axes=False)
                if update_axisplots:
                    with self._perf_scope("Axes plot from SPL"):
                        self.plot_xaxis()
                        self.plot_yaxis()
                if update_polarplots:
                    with self._perf_scope("Polar plot from SPL"):
                        self.plot_polar_pattern()
            self._log_async("SPL")

        _compute_spl()

    def calculate_axes(self, include_x: bool = True, include_y: bool = True):
        """
        Berechnet die X- und Y-Achsen und aktualisiert die entsprechenden Plots.
        
        Args:
            include_x (bool): Ob die X-Achse per Superposition berechnet werden soll.
            include_y (bool): Ob die Y-Achse per Superposition berechnet werden soll.
        """
        with self._perf_scope("Calc axes total"):
            if not include_x and not include_y:
                self._set_axes_show_flag(False)
                return

            performed = False

            if include_x:
                with self._perf_scope("Calc axis X"):
                    calculator_x = SoundFieldCalculatorXaxis(self.settings, self.container.data, self.container.calculation_spl)
                    calculator_x.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
                    calculator_x.calculateXAxis()
                    self.container.set_calculation_axes(calculator_x.calculation_spl)
                    performed = True

            if include_y:
                with self._perf_scope("Calc axis Y"):
                    calculator_y = SoundFieldCalculatorYaxis(self.settings, self.container.data, self.container.calculation_spl)
                    calculator_y.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
                    calculator_y.calculateYAxis()
                    self.container.set_calculation_axes(calculator_y.calculation_spl)
                    performed = True

            if not performed:
                self._set_axes_show_flag(False)
                self.draw_plots.show_empty_axes()
                return

            self._set_axes_show_flag(True)

            if include_x:
                self.plot_xaxis()
            if include_y:
                self.plot_yaxis()

    def _get_physical_params_hash(self, speaker_array):
        """
        Erstellt einen Hash der relevanten physischen Parameter eines speaker_array.
        
        Relevante Parameter:
        - Anzahl der Quellen
        - Lautsprecher-Typen
        - Positionen (x, y, z, z_stack, z_flown)
        - Orientierung (azimuth, site, angle)
        - Arc-Konfiguration (arc_angle, arc_shape, arc_scale_factor)
        
        NICHT enthalten:
        - Pegel (gain, source_level)
        - Delay (delay, source_time)
        - Polarität (source_polarity)
        - Windowing-Parameter
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
            
        Returns:
            str: SHA256 Hash der relevanten Parameter
        """
        def _array_to_string(arr):
            """Konvertiert ein Array zu einem konsistenten String mit gerundeten Werten."""
            if isinstance(arr, np.ndarray):
                # Runde auf 6 Dezimalstellen, um Floating-Point-Ungenauigkeiten zu vermeiden
                return ','.join(f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in arr)
            else:
                return ','.join(str(x) for x in arr)
        
        # Sammle alle relevanten physischen Parameter
        params = []
        
        # Anzahl und Typen
        params.append(str(speaker_array.number_of_sources))
        params.append(_array_to_string(speaker_array.source_polar_pattern))
        
        # Positionen
        if hasattr(speaker_array, 'source_position_x'):
            params.append(_array_to_string(speaker_array.source_position_x))
        if hasattr(speaker_array, 'source_position_y'):
            params.append(_array_to_string(speaker_array.source_position_y))
        if hasattr(speaker_array, 'source_position_z'):
            params.append(_array_to_string(speaker_array.source_position_z))
        if hasattr(speaker_array, 'source_position_z_stack'):
            params.append(_array_to_string(speaker_array.source_position_z_stack))
        if hasattr(speaker_array, 'source_position_z_flown'):
            params.append(_array_to_string(speaker_array.source_position_z_flown))
        
        # Orientierung
        if hasattr(speaker_array, 'source_azimuth'):
            params.append(_array_to_string(speaker_array.source_azimuth))
        if hasattr(speaker_array, 'source_site'):
            params.append(_array_to_string(speaker_array.source_site))
        if hasattr(speaker_array, 'source_angle'):
            params.append(_array_to_string(speaker_array.source_angle))
        
        # Arc-Konfiguration (runde auch hier)
        if hasattr(speaker_array, 'arc_angle'):
            params.append(f"{float(speaker_array.arc_angle):.6f}")
        if hasattr(speaker_array, 'arc_shape'):
            params.append(str(speaker_array.arc_shape))
        if hasattr(speaker_array, 'arc_scale_factor'):
            params.append(f"{float(speaker_array.arc_scale_factor):.6f}")
        
        # Erstelle Hash
        param_string = '|'.join(params)
        return hashlib.sha256(param_string.encode('utf-8')).hexdigest()

    def _should_recalculate_speaker_positions(self, speaker_array):
        """
        Prüft, ob die Speaker-Positionen neu berechnet werden müssen.
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
            
        Returns:
            bool: True wenn Neuberechnung nötig, False sonst
        """
        array_id = speaker_array.id
        current_hash = self._get_physical_params_hash(speaker_array)
        
        # Wenn kein gespeicherter Hash existiert, muss berechnet werden
        if array_id not in self._speaker_position_hashes:
            self._speaker_position_hashes[array_id] = current_hash
            return True
        
        # Wenn sich der Hash geändert hat, muss neu berechnet werden
        old_hash = self._speaker_position_hashes[array_id]
        if old_hash != current_hash:
            self._speaker_position_hashes[array_id] = current_hash
            return True
        
        # Keine Änderung erkannt
        return False

    def speaker_position_calculator(self, speaker_array):
        """
        Berechnet den Nullpunkt in der Mitte der Lautsprecherhöhe für Lautsprechersysteme.
        Berücksichtigt die front_height aus den Cabinet-Daten.
        
        Optimiert: Berechnung erfolgt nur, wenn sich physische Parameter geändert haben.
        """
        if not self._should_recalculate_speaker_positions(speaker_array):
            return
        
        config = getattr(speaker_array, 'configuration', 'unknown')
        
        with self._perf_scope(f"Speaker position calc ({config})"):
            speaker_position_calculator = SpeakerPositionCalculator(self.container)
            speaker_position_calculator.calculate_stack_center(speaker_array)


    def beamsteering_calculator(self, speaker_array, speaker_array_id):
        """
        Berechnet die Beamsteering-Daten für ein bestimmtes Lautsprecherarray.
        """
        config = getattr(speaker_array, 'configuration', 'unknown')
        with self._perf_scope(f"Beamsteering calc ({config})"):
            beamsteering = BeamSteering(speaker_array, self.container.data, self.settings)
            beamsteering.calculate(speaker_array_id)


    def windowing_calculator(self, speaker_array_id):
        """
        Führt Fensterberechnungen für ein bestimmtes Lautsprecherarray durch.

        Args:
            speaker_array_id (int): Die ID des Lautsprecherarrays.
        """
        calculator_instance = WindowingCalculator(self.settings, self.container.data, self.container.calculation_windowing, speaker_array_id)
        self.container.set_calculation_windowing(calculator_instance.calculate_windowing())

    def calculate_impulse(self, force: bool = False):
        with self._perf_scope("Calc impulse total"):
            # Prüfe ob ImpulseManager existiert und Impulspunkte vorhanden sind
            if hasattr(self, 'impulse_manager') and self.settings.impulse_points:
                with self._perf_scope("Impulse calc only"):
                    self.impulse_manager.update_calculation_impulse(force=force)
                self.update_plot_impulse()

    def update_plot_impulse(self):
        with self._perf_scope("Plot impulse total"):
            self.impulse_manager.update_plot_impulse()
        
    def calculate_polar(self):
        """
        Berechnet die Polardaten und aktualisiert den Plot.

        Diese Methode berechnet die Polardaten und aktualisiert den Plot basierend auf dem aktuell
        ausgewählten Lautsprecherarray.
        """
        with self._perf_scope("Calc polar total"):
            if getattr(self.settings, "polar_plot_fem", False):
                return

            calculator_instance = PolarPlotCalculator(self.settings, self.container.data, self.container.calculation_polar)
            calculator_instance.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
            with self._perf_scope("Calc polar pressure"):
                calculator_instance.calculate_polar_pressure()
            self._set_polar_show_flag(True)

            self.plot_polar_pattern()

    # ---- VERSCHIEDENE AUFGABEN

    def init_container(self, force_reload: bool = False):
        """
        Initialisiert den Datencontainer.

        Diese Methode lädt Polardaten und initialisiert die Berechnungsachsen.
        """
        self.container.load_polardata(force=force_reload)
        self.container.reset_runtime_state()
        
        # Lösche gespeicherte Positionshashes, damit beim nächsten Update neu berechnet wird
        if force_reload:
            self._speaker_position_hashes.clear()


    def get_selected_speaker_array_id(self):
        """
        Ermittelt die ID des aktuell ausgewählten Lautsprecherarrays.

        Returns:
            int or None: ID des ausgewählten Arrays oder None, wenn keins ausgewählt ist.
        """
        if not hasattr(self, 'sources_instance') or not hasattr(self.sources_instance, 'sources_tree_widget'):
            array_ids = list(self.settings.get_all_speaker_array_ids())
            return array_ids[0] if array_ids else None

        selected_items = self.sources_instance.sources_tree_widget.selectedItems()
        
        if selected_items:
            selected_item = selected_items[0]  # Nehmen Sie das erste ausgewählte Item
            speaker_array_id = selected_item.data(0, Qt.UserRole)  # type: ignore 
            # Holen Sie die gespeicherte speaker_array_id
            
            # Prüfen Sie, ob speaker_array_id gültig ist
            if speaker_array_id is not None:
                speaker_array = self.settings.get_speaker_array(speaker_array_id)
                
                if speaker_array is not None:
                    return speaker_array_id
        
        return None

    def closeEvent(self, event):
        # Schließen Sie alle zusätzlichen Widgets und Ressourcen
        if hasattr(self, 'impulse_manager'):
            self.impulse_manager.close_impulse_input_dock_widget()
        
        if hasattr(self, 'snapshot_engine'):
            self.snapshot_engine.close_capture_dock_widget()
        
        if hasattr(self, 'draw_widgets'):
            self.draw_widgets.close_all_widgets()
        
        if hasattr(self, 'sources_instance'):
            self.sources_instance.close_sources_dock_widget()

        if self.surface_manager is not None:
            self.surface_manager.close_surface_dock_widget()
        
        if hasattr(self, 'ui_settings'):
            self.ui_settings.close()
        
        if hasattr(self, 'manage_speaker_window') and self.manage_speaker_window:
            if hasattr(self.manage_speaker_window, 'transfer_function_viewer') and \
               self.manage_speaker_window.transfer_function_viewer:
                self.manage_speaker_window.transfer_function_viewer.close()
            self.manage_speaker_window.close()
        
        event.accept()
        super().closeEvent(event)

    def _show_surface_dock_lazy(self):
        manager = self._ensure_surface_manager()
        manager.show_surface_dock_widget()


if __name__ == "__main__":  # Befehle zur Erstausführung des Skripts

    log_dir = configure_logging()
    crash_reporter = CrashReporter(log_dir)
    sys.excepthook = crash_reporter.handle_exception

    profiler = cProfile.Profile()
    profiler.enable()
    
    exit_code = 1
    try:
        settings = Settings()
        container = DataContainer()
        app = QtWidgets.QApplication(sys.argv)
        
        window = MainWindow(settings, container)
        window.init_container()
        window.show_sources_dock_widget()
        window.snapshot_engine.show_snapshot_widget()
        window.update_freq_bandwidth()
        window.show()

        if os.environ.get("LFO_CRASH_TEST") == "1":
            QtCore.QTimer.singleShot(
                1000,
                lambda: (_ for _ in ()).throw(RuntimeError("Crash-Test: absichtlich ausgelöster Fehler")),
            )

        exit_code = app.exec_()
    except Exception:
        crash_reporter.handle_exception(*sys.exc_info())
    finally:
        profiler.disable()

    sys.exit(exit_code)
        






