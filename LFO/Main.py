import sys
import cProfile
import logging
import traceback
import os
import hashlib
import time
from typing import Optional

import numpy as np
from qtpy import QtWidgets, QtCore
from PyQt5.QtCore import Qt

import matplotlib as mpl
mpl.set_loglevel('WARNING')  # Reduziert matplotlib Debug-Ausgaben

from Module_LFO.Modules_Init.Logging import configure_logging, CrashReporter
from Module_LFO.Modules_Init.Progress import ProgressManager, ProgressCancelled

from Module_LFO.Modules_Ui.UiSourceManagement import Sources
from Module_LFO.Modules_Ui.UiImpulsemanager import ImpulseManager
from Module_LFO.Modules_Ui.UiSettings import UiSettings
from Module_LFO.Modules_Ui.UiFile import UiFile
from Module_LFO.Modules_Window.Mainwindow import Ui_MainWindow
from Module_LFO.Modules_Ui.UiManageSpeaker import UiManageSpeaker
from Module_LFO.Modules_Ui.UISurfaceManager import UISurfaceManager

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator
from Module_LFO.Modules_Calculate.SoundFieldCalculator_FEM import SoundFieldCalculatorFEM
from Module_LFO.Modules_Calculate.SoundFieldCalculator_FDTD import SoundFieldCalculatorFDTD
from Module_LFO.Modules_Calculate.SoundfieldCalculatorXaxis import SoundFieldCalculatorXaxis
from Module_LFO.Modules_Calculate.SoundfieldCalculatorYaxis import SoundFieldCalculatorYaxis
from Module_LFO.Modules_Calculate.WindowingCalculator import WindowingCalculator
from Module_LFO.Modules_Calculate.BeamSteering import BeamSteering
from Module_LFO.Modules_Calculate.PolarPlotCalculator import PolarPlotCalculator
from Module_LFO.Modules_Calculate.BandwidthCalculator import BandwidthCalculator
from Module_LFO.Modules_Calculate.SpeakerPositionCalculator import SpeakerPositionCalculator
from Module_LFO.Modules_Calculate.CalculationHandler import CalculationHandler

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
        self._task_runtime_stats: dict[str, dict[str, float]] = {}
        self._skip_fem_recalc_once = False
        self._fem_frequency_dirty = True
        
        # Calculation Handler für Berechnungslogik
        self.calculation_handler = CalculationHandler(self.settings)

        self._initialize_default_plot_flags()

        
        # Verbinde die Aktionen mit den entsprechenden Methoden von UiFile
        self.ui.actionLoad.triggered.connect(self.ui_file.load_file)
        self.ui.actionSave_as.triggered.connect(self.ui_file.save_file_as)
        self.ui.actionSave.triggered.connect(self.ui_file.overwrite_file)
        self.ui.actionNew.triggered.connect(self.ui_file.new_pa_file)
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
        

    # ---- PLOT METHODEN ----

    def plot_spl(self, update_axes: bool = True, reset_camera: bool = False):
        """Zeichnet den SPL-Plot (Sound Pressure Level)."""
        if not self.container.calculation_spl.get("show_in_plot", True):
            self.draw_plots.show_empty_spl()
            if update_axes:
                self.draw_plots.show_empty_axes()
                self.draw_plots.show_empty_polar()
            return
        speaker_array_id = self.get_selected_speaker_array_id()
        self.draw_plots.plot_spl(self.settings, speaker_array_id, update_axes=update_axes, reset_camera=reset_camera)

    def plot_xaxis(self):
        """Zeichnet den X-Achsen-Plot."""
        self.draw_plots.plot_xaxis()

    def plot_yaxis(self):
        """Zeichnet den Y-Achsen-Plot."""
        self.draw_plots.plot_yaxis()

    def plot_polar_pattern(self):
        """Zeichnet das Polar-Pattern."""
        polar_data = self.container.get_polar_data()
        if not polar_data or not polar_data.get("show_in_plot", True):
            self.draw_plots.show_empty_polar()
            return
        self.draw_plots.plot_polar_pattern()

    def plot_beamsteering(self, speaker_array_id: int):
        """
        Zeichnet den Beamsteering-Plot für ein Speaker-Array.
        
        Args:
            speaker_array_id: Die ID des Speaker-Arrays
        """
        if hasattr(self, 'sources_instance'):
            self.sources_instance.update_beamsteering_plot(speaker_array_id)

    def plot_windowing(self, speaker_array_id: int):
        """
        Zeichnet den Windowing-Plot für ein Speaker-Array.
        
        Args:
            speaker_array_id: Die ID des Speaker-Arrays
        """
        if hasattr(self, 'sources_instance'):
            self.sources_instance.update_windowing_plot(self.container.calculation_windowing, speaker_array_id)

    def init_plot(self, preserve_camera: bool = True, view: Optional[str] = None):
        """
        Initialisiert alle Plotbereiche mit leerer Darstellung.
        
        Args:
            preserve_camera: Kameraperspektive beibehalten
            view: Optionale View-Einstellung
        """
        if hasattr(self, 'draw_plots'):
            QtCore.QTimer.singleShot(
                0,
                lambda: self.draw_plots.show_empty_plots(
                    preserve_camera=preserve_camera,
                    view=view,
                ),
            )


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
            tasks.append(("Calculating Axes", lambda: self.calculate_axes(update_plot=True)))

        if not update_polarplot:
            tasks.append(("Calculating Polar", lambda: self.calculate_polar(update_plot=True)))

        if not update_soundfield:
            # Calculate Button: Manuelle Berechnung basierend auf Plot-Modus
            plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
            is_fem_mode = bool(getattr(self.settings, 'spl_plot_fem', False))
            
            if plot_mode == "SPL plot" and is_fem_mode:
                # "SPL plot" + FEM: Nur FEM-Analyse
                tasks.append(("Calculating FEM", lambda: self.calculate_spl(
                    show_progress=True,
                    update_plot=True,
                    update_axisplots=not update_axisplot,
                    update_polarplots=not update_polarplot,
                )))
            elif plot_mode == "SPL over time" and is_fem_mode:
                # "SPL over time" + FEM: Nur FDTD-Analyse
                tasks.append(("Calculating FDTD", lambda: self.calculate_fdtd(
                    show_progress=True,
                    update_plot=True,
                )))
            elif plot_mode == "SPL over time":
                # "SPL over time" ohne FEM: Nur FDTD
                tasks.append(("Calculating FDTD", lambda: self.calculate_fdtd(
                    show_progress=True,
                    update_plot=True,
                )))
            else:
                # Normale SPL-Berechnung (inkl. FEM falls aktiv)
                tasks.append(("Calculating SPL", lambda: self.calculate_spl(
                    show_progress=True,
                    update_plot=True,
                    update_axisplots=not update_axisplot,
                    update_polarplots=not update_polarplot,
                )))

        if hasattr(self, 'impulse_manager'):
            if not update_impulse and self.settings.impulse_points:
                tasks.append(("Calculating Impulse", lambda: self.calculate_impulse(force=True, update_plot=True)))
            elif not update_impulse:
                tasks.append(("Clearing impulse plot", self.impulse_manager.show_empty_plot))

        if not tasks:
            return
        try:
            self.run_tasks_with_progress("Running main calculations", tasks)
        except ProgressCancelled:
            return

    def update_freq_bandwidth(self):
        lower_freq = self.settings.lower_calculate_frequency
        upper_freq = self.settings.upper_calculate_frequency
        
        # # Erstelle eine Instanz des Calculators und berechne beide
        bandwidth_calculator = BandwidthCalculator(self.settings, self.container)  # 🚀 OPTIMIERT: DataContainer-Objekt
        bandwidth_calculator.calculate_magnitude_average(lower_freq, upper_freq)
        bandwidth_calculator.calculate_phase_average(lower_freq, upper_freq)
        
        # 🎯 TERZBAND-MITTELUNG: Berechne für alle Polar-Frequenzen
        bandwidth_calculator.calculate_polar_frequency_averages(self.settings.polar_frequencies)

        skip_fem = bool(getattr(self.settings, "spl_plot_fem", False))
        if skip_fem and not self._fem_frequency_dirty:
            self._skip_fem_recalc_once = True
        self.update_speaker_array_calculations(skip_fem_recalc=skip_fem)

    def mark_fem_frequency_dirty(self):
        self._fem_frequency_dirty = True
        self._skip_fem_recalc_once = False

    def speaker_position_calculator(self, speaker_array):
        """
        Berechnet den Nullpunkt in der Mitte der Lautsprecherhöhe für Lautsprechersysteme.
        Berücksichtigt die front_height aus den Cabinet-Daten.
        
        Optimiert: Berechnung erfolgt nur, wenn sich physische Parameter geändert haben.
        """
        # Prüfe über CalculationHandler, ob Neuberechnung nötig ist
        if not self.calculation_handler.should_recalculate_speaker_positions(speaker_array, debug=False):
            return
        
        speaker_position_calculator = SpeakerPositionCalculator(self.container)
        speaker_position_calculator.calculate_stack_center(speaker_array)

    def beamsteering_calculator(self, speaker_array, speaker_array_id, update_plot: bool = False):
        """
        Berechnet die Beamsteering-Daten für ein bestimmtes Lautsprecherarray.
        
        Args:
            speaker_array: Das Lautsprecherarray
            speaker_array_id: Die ID des Arrays
            update_plot: Ob der Plot aktualisiert werden soll
        """
        beamsteering = BeamSteering(speaker_array, self.container.data, self.settings)
        beamsteering.calculate(speaker_array_id)
        
        if update_plot:
            self.plot_beamsteering(speaker_array_id)

    def windowing_calculator(self, speaker_array_id, update_plot: bool = False):
        """
        Führt Fensterberechnungen für ein bestimmtes Lautsprecherarray durch.

        Args:
            speaker_array_id: Die ID des Lautsprecherarrays
            update_plot: Ob der Plot aktualisiert werden soll
        """
        calculator_instance = WindowingCalculator(self.settings, self.container.data, self.container.calculation_windowing, speaker_array_id)
        self.container.set_calculation_windowing(calculator_instance.calculate_windowing())
        
        if update_plot:
            self.plot_windowing(speaker_array_id)

    def update_speaker_array_calculations(self, skip_fem_recalc: bool = False):
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
                    # Prüfe ob snapshot_tree_widget existiert und noch gültig ist
                    if hasattr(snapshot_engine, '_is_widget_valid') and snapshot_engine._is_widget_valid():
                        try:
                            blocker = QtCore.QSignalBlocker(snapshot_engine.snapshot_tree_widget)
                            snapshot_engine.update_snapshot_widgets()
                            del blocker
                        except RuntimeError:
                            # Widget wurde während der Verwendung gelöscht
                            print("[Main] snapshot_tree_widget wurde während Update gelöscht")
                    elif snapshot_engine.snapshot_tree_widget:
                        # Widget existiert, aber ist nicht mehr gültig - update_snapshot_widgets() wird es selbst prüfen
                        snapshot_engine.update_snapshot_widgets()
                skip_fem_recalc = skip_fem_recalc or self._skip_fem_recalc_once
                self._skip_fem_recalc_once = False
                if self._fem_frequency_dirty:
                    skip_fem_recalc = False

                update_soundfield = getattr(self.settings, "update_pressure_soundfield", True)
                update_axisplot = getattr(self.settings, "update_pressure_axisplot", True)
                update_polarplot = getattr(self.settings, "update_pressure_polarplot", True)
                update_impulse = getattr(self.settings, "update_pressure_impulse", True)

                is_fem_mode = bool(getattr(self.settings, "spl_plot_fem", False))
                skipped_fem_run = False
                if skip_fem_recalc and is_fem_mode:
                    update_soundfield = False
                    skipped_fem_run = True

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

                # Plots werden aktualisiert, wenn SPL, Polar oder Impulse berechnet werden
                should_update_plots = update_soundfield or update_polarplot or update_impulse
                
                tasks = [
                    ("Updating speaker positions", lambda: self.speaker_position_calculator(speaker_array)),
                    ("Calculating beam steering", lambda up=should_update_plots: self.beamsteering_calculator(speaker_array, speaker_array_id, update_plot=up)),
                    ("Calculating windowing", lambda up=should_update_plots: self.windowing_calculator(speaker_array_id, update_plot=up)),
                ]

                if update_axisplot:
                    tasks.append(("Calculating axes", lambda: self.calculate_axes(update_plot=True)))
                else:
                    tasks.append(("Clearing axis plots", _clear_axes))

                if update_polarplot:
                    tasks.append(("Calculating polar pattern", lambda: self.calculate_polar(update_plot=True)))
                else:
                    tasks.append(("Clearing polar plot", _clear_polar))

                fem_calculated = False

                # Automatische Berechnung: Prüfe Plot-Modus
                plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
                
                if update_soundfield:
                    # Automatische Berechnung basierend auf aktivem Plot-Modus
                    if plot_mode == "SPL plot" and is_fem_mode:
                        # "SPL plot" + FEM: Nur FEM-Analyse
                        tasks.append(("Calculating FEM", lambda: self.calculate_spl(
                            show_progress=True,
                            update_plot=True,
                            update_axisplots=False,
                            update_polarplots=False,
                        )))
                        fem_calculated = True
                    elif plot_mode == "SPL over time":
                        # "SPL over time": Nur FDTD-Analyse (unabhängig von FEM-Modus)
                        tasks.append(("Calculating FDTD", lambda: self.calculate_fdtd(
                            show_progress=True,
                            update_plot=True,
                        )))
                    else:
                        # Normale SPL-Berechnung (ohne FEM)
                        tasks.append(("Calculating SPL", lambda: self.calculate_spl(
                            show_progress=True,
                            update_plot=True,
                            update_axisplots=False,
                            update_polarplots=False,
                        )))
                elif skipped_fem_run:
                    pass
                else:
                    tasks.append(("Clearing SPL plot", _clear_spl))

                if hasattr(self, 'impulse_manager'):
                    if update_impulse and getattr(self.settings, "impulse_points", []):
                        tasks.append(("Calculating impulse", lambda: self.calculate_impulse(force=True, update_plot=True)))
                    elif not update_impulse:
                        tasks.append(("Clearing impulse plot", self.impulse_manager.show_empty_plot))

                try:
                    self.run_tasks_with_progress("Updating array calculations", tasks)
                except ProgressCancelled:
                    return

                if not update_axisplot:
                    self.draw_plots.show_empty_axes()
                if not update_polarplot:
                    self.draw_plots.show_empty_polar()
                if not update_soundfield and not skipped_fem_run:
                    self.draw_plots.show_empty_spl()

                if fem_calculated:
                    self._fem_frequency_dirty = False

    def calculate_spl(self, show_progress: bool = True, update_plot: bool = True, update_axisplots: bool = True, update_polarplots: bool = True):
        """
        Führt die SPL-Berechnung durch und aktualisiert optional den Plot.

        Args:
            show_progress: Progress-Bar anzeigen
            update_plot: SPL-Plot aktualisieren
            update_axisplots: Axis-Plots aktualisieren
            update_polarplots: Polar-Plot aktualisieren
        """
        calculator_cls = self.calculation_handler.select_soundfield_calculator_class()
        calculator_instance = calculator_cls(self.settings, self.container.data, self.container.calculation_spl)
        calculator_instance.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!

        frequency_progress_ctx = None
        if show_progress and calculator_cls is SoundFieldCalculatorFEM:
            try:
                precomputed_frequencies = calculator_instance._determine_frequencies()
            except Exception:
                precomputed_frequencies = []
            else:
                calculator_instance.set_precomputed_frequencies(precomputed_frequencies)

            if precomputed_frequencies:
                total_freqs = len(precomputed_frequencies)
                if total_freqs >= 3:
                    import math
                    last_third_start = math.floor(total_freqs * (2 / 3)) + 1
                    last_third_start = min(last_third_start, total_freqs)
                    last_third_count = total_freqs - last_third_start + 1
                    extra_steps = last_third_count
                else:
                    last_third_start = None
                    extra_steps = 0

                total_steps = len(precomputed_frequencies) + extra_steps
                frequency_progress_ctx = self.progress_manager.start(
                    "FEM-Frequenzen",
                    total_steps,
                    minimum_duration_ms=0,
                )
                calculator_instance.set_frequency_progress_plan(last_third_start)

        # Progress-Callback auch nutzen, wenn bereits eine Session aktiv ist
        progress_session = getattr(self, '_current_progress_session', None)
        if hasattr(calculator_instance, 'set_progress_callback') and (show_progress or progress_session):
            def _progress_callback(message: str):
                session = getattr(self, '_current_progress_session', None)
                if session:
                    session.update(message)
                    if session.is_cancelled():
                        session.raise_if_cancelled()
            calculator_instance.set_progress_callback(_progress_callback)

        def _run_calculator():
            if frequency_progress_ctx:
                with frequency_progress_ctx as freq_session:
                    calculator_instance.set_frequency_progress_session(freq_session)
                    calculator_instance.calculate_soundfield_pressure()
                    calculator_instance.set_frequency_progress_session(None)
            else:
                calculator_instance.calculate_soundfield_pressure()

        use_internal_progress = show_progress and progress_session is None and frequency_progress_ctx is None

        try:
            if use_internal_progress:
                if calculator_cls is SoundFieldCalculatorFEM:
                    task_title = "FEM-Berechnung"
                    task_label = "FEM-Schallfeld lösen"
                else:
                    task_title = "SPL-Berechnung"
                    task_label = "Schallfeld berechnen"
                self.run_tasks_with_progress(task_title, [(task_label, _run_calculator)])
            else:
                _run_calculator()
        except ProgressCancelled:
            self._show_empty_plots_after_cancel()
            return
        
        self.container.set_calculation_SPL(calculator_instance.calculation_spl)
        self._set_spl_show_flag(True)
        
        if update_plot:
            self.plot_spl(update_axes=False)
        
        if update_plot and update_axisplots:
            self.plot_xaxis()
            self.plot_yaxis()
        if update_plot and update_polarplots:
            self.plot_polar_pattern()

    def calculate_fdtd(self, show_progress: bool = True, update_plot: bool = True):
        """
        Führt die FDTD-Zeitsimulation durch und aktualisiert optional den Plot.
        
        Args:
            show_progress: Progress-Bar anzeigen
            update_plot: SPL-Plot aktualisieren
        """
        print("[FDTD] calculate_fdtd() aufgerufen")
        
        # Erstelle FDTD-Calculator (unabhängig von FEM)
        calculator = SoundFieldCalculatorFDTD(
            self.settings,
            self.container.data,
            self.container.calculation_spl,
        )
        calculator.set_data_container(self.container)
        
        # Ermittle primäre Frequenz aus Settings
        frequency = calculator._select_primary_frequency()
        if frequency is None:
            print("[FDTD] Keine gültige Frequenz gefunden.")
            return
        
        # Frames pro Periode
        frames_per_period = int(getattr(self.settings, "fem_time_frames_per_period", 16) or 16)
        frames_per_period = max(1, frames_per_period)
        
        def _run_fdtd():
            calculator.calculate_fdtd_snapshots(frequency, frames_per_period=frames_per_period)
        
        # Führe Berechnung mit Progress durch
        try:
            if show_progress:
                self.run_tasks_with_progress(
                    "FDTD-Berechnung",
                    [("FDTD-Zeitsimulation", _run_fdtd)]
                )
            else:
                _run_fdtd()
        except ProgressCancelled:
            return
        
        # Aktualisiere calculation_spl (FDTD-Daten werden bereits in calculator.calculation_spl gespeichert,
        # da es das gleiche Objekt wie self.container.calculation_spl ist)
        self.container.set_calculation_SPL(calculator.calculation_spl)
        
        # Aktualisiere Plot
        if update_plot:
            self.plot_spl(update_axes=False)

    def calculate_axes(self, include_x: bool = True, include_y: bool = True, update_plot: bool = True):
        """
        Berechnet die X- und Y-Achsen und aktualisiert optional die Plots.
        
        Args:
            include_x: Ob die X-Achse per Superposition berechnet werden soll
            include_y: Ob die Y-Achse per Superposition berechnet werden soll
            update_plot: Ob die Plots aktualisiert werden sollen
        """
        if not include_x and not include_y:
            self._set_axes_show_flag(False)
            return

        performed = False

        if include_x:
            calculator_x = SoundFieldCalculatorXaxis(self.settings, self.container.data, self.container.calculation_spl)
            calculator_x.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
            calculator_x.calculateXAxis()
            self.container.set_calculation_axes(calculator_x.calculation_spl)
            performed = True

        if include_y:
            calculator_y = SoundFieldCalculatorYaxis(self.settings, self.container.data, self.container.calculation_spl)
            calculator_y.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
            calculator_y.calculateYAxis()
            self.container.set_calculation_axes(calculator_y.calculation_spl)
            performed = True

        if not performed:
            self._set_axes_show_flag(False)
            if update_plot:
                self.draw_plots.show_empty_axes()
            return

        self._set_axes_show_flag(True)

        if update_plot:
            if include_x:
                self.plot_xaxis()
            if include_y:
                self.plot_yaxis()
        
    def calculate_polar(self, update_plot: bool = True):
        """
        Berechnet die Polardaten und aktualisiert optional den Plot.

        Args:
            update_plot: Ob der Plot aktualisiert werden soll
        """
        calculator_instance = PolarPlotCalculator(self.settings, self.container.data, self.container.calculation_polar)
        calculator_instance.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
        calculator_instance.calculate_polar_pressure()
        self._set_polar_show_flag(True)

        if update_plot:
            self.plot_polar_pattern()

    def calculate_impulse(self, force: bool = False, update_plot: bool = True):
        """
        Berechnet Impulse-Response und aktualisiert optional den Plot.
        
        Args:
            force: Erzwinge Neuberechnung
            update_plot: Ob der Plot aktualisiert werden soll
        """
        if hasattr(self, 'impulse_manager') and self.settings.impulse_points:
            self.impulse_manager.update_calculation_impulse(force=force)
            if update_plot:
                self.impulse_manager.update_plot_impulse()


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
            self.calculation_handler.clear_speaker_position_hashes()

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

    def _initialize_default_plot_flags(self) -> None:
        """
        Setzt Plot-Sichtbarkeits-Flags auf True beim App-Start.
        Verhindert leere Plots bei Erststart oder nach Container-Reset.
        """
        axes_data = self.container.calculation_axes.setdefault("aktuelle_simulation", {})
        if "show_in_plot" not in axes_data:
            axes_data["show_in_plot"] = True
        axes_data.setdefault("color", "#6A5ACD")

        if "show_in_plot" not in self.container.calculation_spl:
            self.container.calculation_spl["show_in_plot"] = True
        if "show_in_plot" not in self.container.calculation_polar:
            self.container.calculation_polar["show_in_plot"] = True

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


# ---- HELP METHODEN ----

    def _record_task_runtime(self, label: str, duration: float) -> None:
        if duration <= 0:
            return
        stats = self._task_runtime_stats.setdefault(label, {"avg": duration, "count": 0, "last": duration})
        count = int(stats.get("count", 0)) + 1
        avg = stats.get("avg", duration)
        avg = ((avg * (count - 1)) + duration) / count
        stats.update({"avg": avg, "count": count, "last": duration})

    def _get_task_runtime_estimate(self, label: str) -> Optional[float]:
        stats = self._task_runtime_stats.get(label)
        if not stats:
            return None
        return stats.get("avg")

    def _format_progress_label(self, label: str, estimate: Optional[float] = None) -> str:
        estimate = estimate if estimate is not None else self._get_task_runtime_estimate(label)
        if not estimate:
            return label
        if estimate >= 1.0:
            eta = f"{estimate:.1f}s"
        else:
            eta = f"{estimate * 1000:.0f}ms"
        return f"{label} · ≈{eta}"

    def _show_empty_plots_after_cancel(self) -> None:
        self._set_axes_show_flag(False)
        self._set_polar_show_flag(False)
        self._set_spl_show_flag(False)
        if hasattr(self.draw_plots, "show_empty_axes"):
            self.draw_plots.show_empty_axes()
        if hasattr(self.draw_plots, "show_empty_polar"):
            self.draw_plots.show_empty_polar()
        if hasattr(self.draw_plots, "show_empty_spl"):
            self.draw_plots.show_empty_spl()
        if hasattr(self, "impulse_manager") and hasattr(self.impulse_manager, "show_empty_plot"):
            self.impulse_manager.show_empty_plot()

    def _set_axes_show_flag(self, value: bool):
        axes_data = self.container.calculation_axes.setdefault("aktuelle_simulation", {})
        axes_data["show_in_plot"] = bool(value)

    def _set_polar_show_flag(self, value: bool):
        self.container.calculation_polar["show_in_plot"] = bool(value)

    def _set_spl_show_flag(self, value: bool):
        self.container.calculation_spl["show_in_plot"] = bool(value)


# ----- PROGRESS BAR ----

    def run_tasks_with_progress(self, title, tasks):
        """
        Führt Tasks sequentiell aus mit Progress-Bar Feedback.
        
        Args:
            title: Progress-Bar Titel
            tasks: Liste von (label, function[, cancel_callback]) Tuples
        """
        if not tasks:
            return True

        normalized_tasks = []
        for task in tasks:
            if len(task) == 3:
                label, func, cancel_cb = task
            else:
                label, func = task  # type: ignore[misc]
                cancel_cb = None
            normalized_tasks.append((label, func, cancel_cb))

        weights = []
        for label, _, _ in normalized_tasks:
            estimate = self._get_task_runtime_estimate(label)
            weights.append(estimate if estimate and estimate > 0 else 1.0)

        min_weight = min(weights) if weights else 1.0
        normalized_steps = [
            max(1, int(round(weight / min_weight))) if min_weight > 0 else 1
            for weight in weights
        ]
        total_steps = max(1, sum(normalized_steps)) if normalized_steps else len(normalized_tasks)

        cancelled = False

        with self.progress_manager.start(title, total_steps) as progress:
            self._current_progress_session = progress
            try:
                for (label, func, cancel_cb), step_weight in zip(normalized_tasks, normalized_steps):
                    if progress.is_cancelled():
                        cancelled = True
                        if cancel_cb:
                            cancel_cb()
                        break

                    progress.update(self._format_progress_label(label))
                    if progress.is_cancelled():
                        cancelled = True
                        if cancel_cb:
                            cancel_cb()
                        break

                    start_time = time.perf_counter()
                    try:
                        func()
                    except ProgressCancelled:
                        cancelled = True
                        if cancel_cb:
                            cancel_cb()
                        break
                    else:
                        duration = time.perf_counter() - start_time
                        self._record_task_runtime(label, duration)
                    progress.advance(step_weight)
            finally:
                self._current_progress_session = None

        if cancelled:
            self._show_empty_plots_after_cancel()
            raise ProgressCancelled("Tasks cancelled by user.")

        return True



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
        






