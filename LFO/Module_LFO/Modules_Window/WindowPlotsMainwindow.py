from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import Optional
from functools import partial

from Module_LFO.Modules_Plot.PlotMPLCanvas import MplCanvas
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SoundFieldCalculator_PhaseDiff import SoundFieldCalculatorPhaseDiff
from Module_LFO.Modules_Calculate.SoundFieldCalculator_FDTD import SoundFieldCalculatorFDTD

try:
    from Module_LFO.Modules_Plot.PlotSPL3D import DrawSPLPlot3D
except ImportError:  # PyVista optional
    DrawSPLPlot3D = None
from Module_LFO.Modules_Plot.PlotSPLXaxis import DrawSPLPlot_Xaxis
from Module_LFO.Modules_Plot.PlotSPLYaxis import DrawSPLPlot_Yaxis
from Module_LFO.Modules_Plot.PlotPolarPattern import DrawPolarPattern


class DrawPlotsMainwindow(ModuleBase):
    PLOT_MODE_OPTIONS = ("SPL plot", "Phase alignment", "SPL over time")
    def __init__(self, main_window, settings, container):
        super().__init__(settings)  
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self._time_frame_index = 0
        self._time_frames_per_period = int(getattr(self.settings, "fem_time_frames_per_period", 16) or 16)
        if not hasattr(self.settings, "fem_time_frames_per_period"):
            setattr(self.settings, "fem_time_frames_per_period", self._time_frames_per_period)
        
        # Initialisiere die Splitter-Referenzen
        self.set_splitter_positions()
        
        # Erstellen Sie separate MplCanvas-Instanzen für jeden Plot
        # X- und Y-Achsen mit kleineren Figuren (width, height reduziert)
        self.matplotlib_canvas_xaxis = MplCanvas(parent=self.main_window, width=5, height=2.5)
        self.matplotlib_canvas_yaxis = MplCanvas(parent=self.main_window, width=3, height=4)
        # Layout-Engine wird von den Plot-Klassen selbst gesteuert
        # SPL-Plot: constrained_layout (in PlotSPL3dB/PlotSPLLinear)
        # X/Y-Achsen: subplots_adjust (in PlotSPLXaxis/PlotSPLYaxis)
        # Polar: tight_layout (in PlotPolarPattern)
        
        # Polar Plot erstellen
        self.matplotlib_canvas_polar_pattern = MplCanvas(parent=self.main_window)
        # Ersetze die Standard-Achsen durch polare Achsen
        self.matplotlib_canvas_polar_pattern.fig.clear()
        self.matplotlib_canvas_polar_pattern.ax = self.matplotlib_canvas_polar_pattern.fig.add_subplot(111, projection='polar')
        # Layout wird von PlotPolarPattern gesteuert
        
        # Colorbar-Canvas erstellen
        self.colorbar_canvas = FigureCanvas(Figure(figsize=(0.8, 4)))
        self.colorbar_ax = self.colorbar_canvas.figure.add_subplot(111)

        # Maximale Ausnutzung des verfügbaren Platzes
        self.colorbar_canvas.figure.subplots_adjust(
            left=0.3,    
            right=0.9,   
            top=1.0,     # Wichtig: top=1.0 für keine Lücke oben
            bottom=0.05
        )

        # Layout für Colorbar ohne Margins
        layout = QtWidgets.QVBoxLayout(self.main_window.ui.colorbar_plot)
        layout.setContentsMargins(0, 0, 0, 0)  # Keine Ränder
        layout.setSpacing(0)                    # Kein Abstand
        layout.addWidget(self.colorbar_canvas)

        self.draw_spl_plotter = None
        self.active_spl_widget = None

        self.plot_mode_menu: Optional[QtWidgets.QMenu] = None
        self.plot_mode_context_group: Optional[QtWidgets.QActionGroup] = None
        self.plot_mode_menu_group: Optional[QtWidgets.QActionGroup] = None
        self.plot_mode_menu_actions: dict[str, QtWidgets.QAction] = {}
        self.menu_mode_actions: dict[str, QtWidgets.QAction] = {}
        self._setup_plot_mode_controls()

        self._initialize_spl_plotter()
        initial_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        self.on_plot_mode_changed(initial_mode)
        # Sicherstellen, dass der 3D-Plot keine Mindesthöhe erzwingt
        if self.draw_spl_plotter and hasattr(self.draw_spl_plotter, "widget"):
            self.draw_spl_plotter.widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.draw_spl_plotter.widget.setMinimumSize(QtCore.QSize(0, 0))

        # Übergebe main_window Referenz, damit Plot-Klassen snapshot_engine dynamisch holen können
        self.draw_spl_plot_xaxis = DrawSPLPlot_Xaxis(self.matplotlib_canvas_xaxis.ax, self.settings, self.main_window)
        self.draw_spl_plot_yaxis = DrawSPLPlot_Yaxis(self.matplotlib_canvas_yaxis.ax, self.settings, self.main_window)

        self.matplotlib_canvas_xaxis.mpl_connect('scroll_event', self.zoom)
        self.matplotlib_canvas_yaxis.mpl_connect('scroll_event', self.zoom)
        # Polar Pattern Plot vom Zoomen ausgeschlossen
        
        self.matplotlib_canvas_xaxis.mpl_connect('button_press_event', self.pan)
        self.matplotlib_canvas_yaxis.mpl_connect('button_press_event', self.pan)
        # Polar Pattern Plot vom Pannen ausgeschlossen

        # Erstelle Polar Pattern Plot Handler
        self.draw_polar_pattern = DrawPolarPattern(self.matplotlib_canvas_polar_pattern.ax, self.settings)
        
        # HINWEIS: X/Y-Achsen und Polar-Plots sind bereits durch ihre __init__ Methoden
        # vollständig initialisiert und gezeichnet (initialize_empty_plots() + _apply_layout())
        
        # Füge Canvas-Widgets zu UI-Layouts hinzu (beim Start)
        self._setup_plot_layouts()
        
        # Initialisiere calculation_spl Container mit leeren Arrays
        self._initialize_empty_sound_field()
        
        # Zeichne SPL-Plot mit empty data (erster Plot)
        # X/Y-Achsen und Polar sind bereits initialisiert, daher kein erneuter Aufruf nötig
        self.plot_spl(self.settings, None, update_axes=False)
        
        # Setze die Default-Ansicht für die Splitter
        self.set_default_view()

    def _setup_plot_layouts(self):
        """Fügt Canvas-Widgets zu den UI-Layouts hinzu"""
        # SPL-Plot
        if self.main_window.ui.SPLPlot.layout() is None:
            layout_spl = QtWidgets.QVBoxLayout(self.main_window.ui.SPLPlot)
        else:
            layout_spl = self.main_window.ui.SPLPlot.layout()
        if self.active_spl_widget is not None:
            layout_spl.addWidget(self.active_spl_widget)
        
        # SPL-Plot X-Axis
        if self.main_window.ui.SPLPlot_XAxis.layout() is None:
            layout_xaxis = QtWidgets.QVBoxLayout(self.main_window.ui.SPLPlot_XAxis)
        else:
            layout_xaxis = self.main_window.ui.SPLPlot_XAxis.layout()
        layout_xaxis.addWidget(self.matplotlib_canvas_xaxis)
        
        # SPL-Plot Y-Axis
        if self.main_window.ui.SPLPlot_YAxis.layout() is None:
            layout_yaxis = QtWidgets.QVBoxLayout(self.main_window.ui.SPLPlot_YAxis)
        else:
            layout_yaxis = self.main_window.ui.SPLPlot_YAxis.layout()
        layout_yaxis.addWidget(self.matplotlib_canvas_yaxis)
        
        # Polar Pattern Plot
        if self.main_window.ui.Plot_PolarPattern.layout() is None:
            layout_window = QtWidgets.QVBoxLayout(self.main_window.ui.Plot_PolarPattern)
        else:
            layout_window = self.main_window.ui.Plot_PolarPattern.layout()
        layout_window.addWidget(self.matplotlib_canvas_polar_pattern)

    def _initialize_spl_plotter(self):
        if DrawSPLPlot3D is None:
            raise ImportError(
                "PyVista und PyVistaQt sind nicht verfügbar. Bitte installieren, um den 3D-SPL-Plot zu verwenden."
            )
        self.draw_spl_plotter = DrawSPLPlot3D(self.main_window.ui.SPLPlot, self.settings, self.colorbar_ax)
        self.draw_spl_plotter.set_time_slider_callback(self._on_time_slider_changed)
        # Prüfe ob "SPL over time" aktiv ist
        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        time_mode_active = (plot_mode == "SPL over time")
        if time_mode_active:
            frames = max(1, self._time_frames_per_period)
            self.draw_spl_plotter.update_time_control(True, frames + 1, self._time_frame_index, 0.2)
        else:
            self.draw_spl_plotter.update_time_control(False, self._time_frames_per_period + 1, self._time_frame_index, 0.2)
        self.active_spl_widget = self.draw_spl_plotter.widget

    def _initialize_empty_sound_field(self):
        """Erstellt leere Arrays für das Schallfeld beim Start"""
        # Zugriff auf calculation_spl
        if not hasattr(self.container, 'calculation_spl'):
            self.container.calculation_spl = {}
        
        resolution = self.settings.resolution
        
        # Y-Achse (Höhe des Plots)
        y_start = -self.settings.length / 2
        y_end = self.settings.length / 2
        sound_field_y = np.arange(y_start, y_end + resolution, resolution).tolist()

        # X-Achse (Breite des Plots)
        x_start = -self.settings.width / 2
        x_end = self.settings.width / 2
        sound_field_x = np.arange(x_start, x_end + resolution, resolution).tolist()

        # Pressure-Array mit NaN füllen (empty data)
        sound_field_p = np.full((len(sound_field_y), len(sound_field_x)), np.nan, dtype=float)
        sound_field_phase = np.full((len(sound_field_y), len(sound_field_x)), np.nan, dtype=float)
        sound_field_phase_diff = np.full((len(sound_field_y), len(sound_field_x)), np.nan, dtype=float)

        self.container.calculation_spl['sound_field_x'] = sound_field_x
        self.container.calculation_spl['sound_field_y'] = sound_field_y
        self.container.calculation_spl['sound_field_p'] = sound_field_p.tolist()
        self.container.calculation_spl['sound_field_phase'] = sound_field_phase.tolist()
        self.container.calculation_spl['sound_field_phase_diff'] = sound_field_phase_diff.tolist()

    def _compute_time_mode_field(self):
        calc_spl = getattr(self.container, 'calculation_spl', {})
        
        # Prüfe ob FDTD-Daten vorhanden sind (FDTD ist eigenständig, keine FEM-Abhängigkeit)
        fdtd_results = calc_spl.get("fdtd_simulation")
        has_fdtd = isinstance(fdtd_results, dict) and bool(fdtd_results)
        print(f"[SPL over time] _compute_time_mode_field: FDTD-Daten vorhanden={has_fdtd}")
        
        # WICHTIG: Auch wenn keine FDTD-Daten vorhanden sind, versuchen wir die Simulation zu starten.
        # get_time_snapshot_grid() ruft automatisch calculate_fdtd_snapshots() auf, wenn keine Daten vorhanden sind.
        
        # Hole frames_per_period aus FDTD-Daten (falls vorhanden), sonst aus Settings
        frames = None
        if has_fdtd:
            # Versuche frames_per_period aus den FDTD-Daten zu holen
            for freq_key, sim_data in fdtd_results.items():
                if isinstance(sim_data, dict):
                    stored_frames = sim_data.get("frames_per_period")
                    if stored_frames is not None:
                        frames = int(stored_frames)
                        break
                    # Fallback: Berechne aus total_frames
                    total_frames = sim_data.get("total_frames")
                    if total_frames is not None:
                        frames = int(total_frames) - 1  # -1 weil total_frames = frames_per_period + 1
                        break
        
        # Fallback auf Settings
        if frames is None:
            frames = int(getattr(self.settings, "fem_time_frames_per_period", self._time_frames_per_period) or self._time_frames_per_period)
        
        frames = max(1, frames)
        self._time_frames_per_period = frames
        print(f"[SPL over time] Verwende frames_per_period: {frames} (aus FDTD-Daten: {frames is not None and has_fdtd})")
        
        try:
            provider = SoundFieldCalculatorFDTD(
                self.settings,
                getattr(self.container, 'data', None),
                self.container.calculation_spl,
            )
            provider.set_data_container(self.container)
            
            # Frequenz zu finden: zuerst aus FDTD-Daten, dann aus Settings (KEINE FEM-Abhängigkeit)
            frequency = None
            if has_fdtd:
                # Nimm die erste verfügbare Frequenz aus FDTD-Daten
                try:
                    freq_keys = [float(k) for k in fdtd_results.keys() if isinstance(k, (int, float, str))]
                    if freq_keys:
                        frequency = float(freq_keys[0])
                        print(f"[SPL over time] Frequenz aus FDTD-Daten: {frequency} Hz")
                except (ValueError, TypeError):
                    pass
            
            # Falls keine Frequenz aus FDTD-Daten, versuche aus Settings
            if frequency is None:
                frequency = provider._select_primary_frequency()
            
            if frequency is None:
                print("[SPL over time] Keine gültige Frequenz gefunden")
                return None
            
            print(f"[SPL over time] Verwende Frequenz: {frequency} Hz, Frame: {self._time_frame_index}")
            # total_frames = frames_per_period + 1 (inkl. Frame 0 für t=0)
            total_frames = frames + 1
            frame_index = self._time_frame_index % total_frames
            # get_time_snapshot_grid() ruft automatisch calculate_fdtd_snapshots() auf,
            # wenn keine Daten vorhanden sind (mit Caching)
            sound_field_x, sound_field_y, pressure_grid, _ = provider.get_time_snapshot_grid(
                frequency,
                frame_index,
                frames_per_period=frames,
            )
            print(f"[SPL over time] Daten geladen: Grid {len(sound_field_x)}x{len(sound_field_y)}, Druck min={np.min(pressure_grid):.3e}, max={np.max(pressure_grid):.3e}")
            return frequency, total_frames, sound_field_x, sound_field_y, pressure_grid
        except Exception as exc:  # noqa: BLE001
            print(f"[SPL over time] Snapshot fehlgeschlagen: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def show_empty_plots(self, preserve_camera: bool = True, view: Optional[str] = None):
        """
        Setzt alle Plotbereiche auf ihre Leer-Darstellung zurück.

        Args:
            preserve_camera: Wenn True, bleibt die aktuelle Kameraperspektive erhalten.
        """
        # Aktuelle Splittergrößen sichern, damit das Layout später unverändert bleibt
        top_sizes = self.horizontal_splitter_top.sizes() if self.horizontal_splitter_top else []
        bottom_sizes = self.horizontal_splitter_bottom.sizes() if self.horizontal_splitter_bottom else []
        vertical_sizes = self.vertical_splitter.sizes() if self.vertical_splitter else []

        self._initialize_empty_sound_field()

        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.initialize_empty_scene(preserve_camera=preserve_camera)
            if not preserve_camera and view == "top":
                plotter.set_view_top()
            plotter.update_overlays(self.settings, self.container)
            self.colorbar_canvas.draw()

        self.draw_spl_plot_xaxis.initialize_empty_plots()
        self.matplotlib_canvas_xaxis.draw()

        self.draw_spl_plot_yaxis.initialize_empty_plots()
        self.matplotlib_canvas_yaxis.draw()

        if hasattr(self.draw_polar_pattern, 'initialize_empty_plots'):
            self.draw_polar_pattern.initialize_empty_plots()
            self.matplotlib_canvas_polar_pattern.draw()

        # Gesicherte Splittergrößen wiederherstellen
        if self.horizontal_splitter_top and len(top_sizes) == self.horizontal_splitter_top.count():
            self.horizontal_splitter_top.setSizes(top_sizes)
        if self.horizontal_splitter_bottom and len(bottom_sizes) == self.horizontal_splitter_bottom.count():
            self.horizontal_splitter_bottom.setSizes(bottom_sizes)
        if self.vertical_splitter and len(vertical_sizes) == self.vertical_splitter.count():
            self.vertical_splitter.setSizes(vertical_sizes)

    def _setup_plot_mode_controls(self):
        """Richtet Kontext- und Menüsteuerung für die Plotmodi ein."""
        self.plot_mode_menu = QtWidgets.QMenu(self.main_window)
        self.plot_mode_menu.setObjectName("plot_mode_context_menu")
        self.plot_mode_context_group = QtWidgets.QActionGroup(self.plot_mode_menu)
        self.plot_mode_context_group.setExclusive(True)

        for mode in self.PLOT_MODE_OPTIONS:
            action = self.plot_mode_menu.addAction(mode)
            action.setCheckable(True)
            action.setData(mode)
            self.plot_mode_context_group.addAction(action)
            action.toggled.connect(partial(self._handle_plot_mode_change, mode))
            self.plot_mode_menu_actions[mode] = action
        self.plot_mode_menu.addSeparator()
        self.time_alignment_action = self.plot_mode_menu.addAction("Time alignment")
        self.time_alignment_action.setIcon(QtGui.QIcon.fromTheme("view-time-line"))
        self.time_alignment_action.triggered.connect(self._handle_time_alignment_action)

        self.menu_mode_actions = {}
        ui = getattr(self.main_window, 'ui', None)
        if ui is not None:
            self.plot_mode_menu_group = QtWidgets.QActionGroup(self.main_window)
            self.plot_mode_menu_group.setExclusive(True)
            for mode, action_name in (
                (self.PLOT_MODE_OPTIONS[0], "actionPlotMode_SPL"),
                (self.PLOT_MODE_OPTIONS[1], "actionPlotMode_Phase"),
                (self.PLOT_MODE_OPTIONS[2], "actionPlotMode_Time"),
            ):
                action = getattr(ui, action_name, None)
                if action is None:
                    continue
                action.setCheckable(True)
                self.plot_mode_menu_group.addAction(action)
                action.toggled.connect(partial(self._handle_plot_mode_change, mode))
                self.menu_mode_actions[mode] = action

        colorbar_widget = getattr(self.main_window.ui, 'colorbar_widget', None)
        context_targets = [
            colorbar_widget,
            getattr(self.main_window.ui, 'colorbar_plot', None),
            self.colorbar_canvas,
        ]
        for target in context_targets:
            if target is None:
                continue
            target.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            target.customContextMenuRequested.connect(partial(self._show_plot_mode_menu, target))

        initial_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        setattr(self.settings, 'spl_plot_mode', initial_mode)
        self._sync_plot_mode_actions(initial_mode)
        self._update_plot_mode_availability()

    def _handle_time_alignment_action(self):
        """Kontextmenü-Aktion für schnelle Umschaltung in den Time-Alignment-Modus."""
        if not self._is_mode_allowed("SPL over time"):
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "Time alignment nur im FEM-SPL-Modus verfügbar.",
            )
            return
        setattr(self.settings, 'spl_plot_mode', "SPL over time")
        self._sync_plot_mode_actions("SPL over time")
        self.plot_spl(self.settings, None, update_axes=False, reset_camera=False)

    def _is_mode_allowed(self, mode: str) -> bool:
        fem_active = bool(getattr(self.settings, 'spl_plot_fem', False))
        if mode == "Phase alignment":
            return not fem_active
        if mode == "SPL over time":
            return fem_active
        return True

    def _update_plot_mode_availability(self):
        """Aktualisiert die Aktivierung der Plotmodi je nach FEM-Status."""
        current_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        allowed_default = self.PLOT_MODE_OPTIONS[0]
        for action_source in (self.plot_mode_menu_actions, self.menu_mode_actions):
            if not action_source:
                continue
            for mode, action in action_source.items():
                if action is None:
                    continue
                allowed = self._is_mode_allowed(mode)
                action.setEnabled(allowed)
                if allowed:
                    action.setToolTip("")
                else:
                    tooltip = (
                        "Nur im FEM-Modus verfügbar"
                        if mode == "SPL over time"
                        else "Nur im Superpositionsmodus verfügbar"
                    )
                    action.setToolTip(tooltip)

        if not self._is_mode_allowed(current_mode):
            # fallback auf Standardmodus, der immer erlaubt ist
            new_mode = allowed_default
            if not self._is_mode_allowed(new_mode):
                for mode in self.PLOT_MODE_OPTIONS:
                    if self._is_mode_allowed(mode):
                        new_mode = mode
                        break
            setattr(self.settings, 'spl_plot_mode', new_mode)
            self._sync_plot_mode_actions(new_mode)
            self.show_empty_spl()

    def refresh_plot_mode_availability(self):
        """Öffentliche Schnittstelle für UI-Änderungen (z. B. FEM-Umschaltung)."""
        self._update_plot_mode_availability()

    def _show_plot_mode_menu(self, widget, pos):
        """Zeigt das Kontextmenü für die Plotmodus-Auswahl."""
        if self.plot_mode_menu is None:
            return
        if widget is None:
            return
        self._update_plot_mode_availability()
        current_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        self._sync_plot_mode_actions(current_mode)
        global_pos = widget.mapToGlobal(pos)
        self.plot_mode_menu.popup(global_pos)

    def _sync_plot_mode_actions(self, selection: str):
        """Synchronisiert die Check-States zwischen Menü und Kontextmenü."""
        for mode, action in self.plot_mode_menu_actions.items():
            if action is None:
                continue
            blocked = action.blockSignals(True)
            action.setChecked(mode == selection)
            action.blockSignals(blocked)
        for mode, action in self.menu_mode_actions.items():
            if action is None:
                continue
            blocked = action.blockSignals(True)
            action.setChecked(mode == selection)
            action.blockSignals(blocked)

    def _handle_plot_mode_change(self, selection: str, checked: bool = True):
        """Signal-Handler für Kontext- oder Menüaktionen."""
        if not checked:
            return
        if not self._is_mode_allowed(selection):
            self._update_plot_mode_availability()
            return
        self.on_plot_mode_changed(selection)

    def show_empty_spl(self, preserve_camera: bool = True):
        """Setzt nur den SPL-Plot auf eine leere Darstellung."""
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.initialize_empty_scene(preserve_camera=preserve_camera)
            plotter.update_overlays(self.settings, self.container)
            self.colorbar_canvas.draw()
            # Prüfe ob "SPL over time" aktiv ist
            plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
            time_mode_active = (plot_mode == "SPL over time")
            if time_mode_active:
                frames = max(1, self._time_frames_per_period)
                plotter.update_time_control(True, frames + 1, self._time_frame_index, 0.2)
            else:
                plotter.update_time_control(False, self._time_frames_per_period + 1, self._time_frame_index, 0.2)

    def show_empty_axes(self):
        """Setzt X- und Y-Achsen-Plots zurück."""
        self.draw_spl_plot_xaxis.initialize_empty_plots()
        self.matplotlib_canvas_xaxis.draw()
        self.draw_spl_plot_yaxis.initialize_empty_plots()
        self.matplotlib_canvas_yaxis.draw()

    def show_empty_polar(self):
        """Setzt den Polar-Plot zurück."""
        if hasattr(self.draw_polar_pattern, 'initialize_empty_plots'):
            self.draw_polar_pattern.initialize_empty_plots()
            self.matplotlib_canvas_polar_pattern.draw()

    def plot_spl(self, settings, speaker_array_id, update_axes=True, reset_camera=False):
        """
        Zeichnet den SPL-Plot
        
        Args:
            settings: Settings-Objekt
            speaker_array_id: ID des Lautsprecher-Arrays
            update_axes: Wenn True, werden auch X/Y-Achsen und Polar aktualisiert
                        (False beim Init, da bereits durch __init__ initialisiert)
        """
        self.settings = settings
        self._update_plot_mode_availability()
        plot_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        if not self._is_mode_allowed(plot_mode):
            plot_mode = self.PLOT_MODE_OPTIONS[0]
            setattr(self.settings, 'spl_plot_mode', plot_mode)
            self._sync_plot_mode_actions(plot_mode)
        time_mode_enabled = plot_mode == "SPL over time"
        if plot_mode == "Phase alignment":
            field_key = 'sound_field_phase_diff'
        else:
            field_key = 'sound_field_p'
        
        draw_spl_plotter = self._get_current_spl_plotter()

        if not self.container.calculation_spl.get("show_in_plot", True):
            if draw_spl_plotter is not None:
                draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
                draw_spl_plotter.update_overlays(self.settings, self.container)
                self.colorbar_canvas.draw()
            if update_axes:
                self.draw_spl_plot_xaxis.initialize_empty_plots()
                self.matplotlib_canvas_xaxis.draw()
                self.draw_spl_plot_yaxis.initialize_empty_plots()
                self.matplotlib_canvas_yaxis.draw()
                if hasattr(self.draw_polar_pattern, 'initialize_empty_plots'):
                    self.draw_polar_pattern.initialize_empty_plots()
                    self.matplotlib_canvas_polar_pattern.draw()
            return
        
        calc_spl = self.container.calculation_spl
        allow_zero_phase = plot_mode == "Phase alignment"
        if plot_mode == "Phase alignment":
            if not isinstance(calc_spl, dict):
                calc_spl = {}
            if self._is_empty_data(calc_spl.get('sound_field_phase_diff'), allow_all_zero=True):
                if not self._calculate_phase_alignment_field():
                    calc_spl['sound_field_phase_diff'] = []
        # Prüfe ob gültige Daten vorhanden sind (nicht nur NaN)
        time_snapshot = None
        if time_mode_enabled:
            time_snapshot = self._compute_time_mode_field()
            has_data = time_snapshot is not None
        else:
            has_data = (
                isinstance(calc_spl, dict) and
                len(calc_spl) > 0 and
                'sound_field_x' in calc_spl and
                'sound_field_y' in calc_spl and
                field_key in calc_spl and
                calc_spl.get('sound_field_x') is not None and
                calc_spl.get('sound_field_y') is not None and
                calc_spl.get(field_key) is not None and
                len(calc_spl.get('sound_field_x', [])) > 0 and
                len(calc_spl.get('sound_field_y', [])) > 0 and
                len(calc_spl.get(field_key, [])) > 0 and
                not self._is_empty_data(calc_spl[field_key], allow_all_zero=allow_zero_phase)
            )
        
        if not has_data:
            # Debug: Warum greift der Fallback?
            calc_spl = self.container.calculation_spl
            debug_info = []
            debug_info.append(f"is_dict={isinstance(calc_spl, dict)}")
            debug_info.append(f"len={len(calc_spl) if isinstance(calc_spl, dict) else 0}")
            debug_info.append(f"has_x={'sound_field_x' in calc_spl if isinstance(calc_spl, dict) else False}")
            debug_info.append(f"has_y={'sound_field_y' in calc_spl if isinstance(calc_spl, dict) else False}")
            debug_info.append(f"has_value={field_key in calc_spl if isinstance(calc_spl, dict) else False}")
            if isinstance(calc_spl, dict):
                x_len = len(calc_spl.get('sound_field_x', []))
                y_len = len(calc_spl.get('sound_field_y', []))
                v_len = len(calc_spl.get(field_key, []))
                debug_info.append(f"len_x={x_len}, len_y={y_len}, len_value={v_len}")
                if field_key in calc_spl:
                    is_empty = self._is_empty_data(calc_spl[field_key], allow_all_zero=allow_zero_phase)
                    debug_info.append(f"is_empty={is_empty}")
            print(f"[Plot SPL] Fallback zu leerem Plot aktiviert. Prüfung: {', '.join(debug_info)}")
            
            # Keine Daten - zeige leere Szene
            if draw_spl_plotter is not None:
                draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
                draw_spl_plotter.update_overlays(self.settings, self.container)
                # Zeit-Fader immer anzeigen wenn "SPL over time" aktiv ist
                if time_mode_enabled:
                    # Zeige Zeit-Fader auch ohne Daten
                    frames = max(1, self._time_frames_per_period)
                    draw_spl_plotter.update_time_control(True, frames + 1, self._time_frame_index, 0.2)
                else:
                    draw_spl_plotter.update_time_control(False, self._time_frames_per_period + 1, self._time_frame_index, 0.2)

            self.colorbar_canvas.draw()

            if update_axes:
                self.plot_xaxis()
                self.plot_yaxis()
                self.plot_polar_pattern()
            return
        
        if time_mode_enabled and time_snapshot is not None:
            _, snapshot_total_frames, sound_field_x, sound_field_y, snapshot_values = time_snapshot
            # snapshot_total_frames ist bereits total_frames (frames_per_period + 1)
            # Berechne frames_per_period aus total_frames
            snapshot_frames_per_period = max(1, snapshot_total_frames - 1)  # -1 weil total_frames = frames_per_period + 1
            self._time_frames_per_period = snapshot_frames_per_period
            self.settings.fem_time_frames_per_period = snapshot_frames_per_period
            sound_field_values = snapshot_values
        else:
            sound_field_x = self.container.calculation_spl['sound_field_x']
            sound_field_y = self.container.calculation_spl['sound_field_y']
            sound_field_values = self.container.calculation_spl[field_key]

        # 3D-Plot aktualisieren
        draw_spl_plotter.update_spl_plot(
            sound_field_x,
            sound_field_y,
            sound_field_values,
            self.settings.colorization_mode,
        )
        draw_spl_plotter.update_overlays(self.settings, self.container)
        
        # WICHTIG: update_time_control() NACH update_spl_plot() aufrufen,
        # damit der Fader nicht von initialize_empty_scene() versteckt wird
        if draw_spl_plotter is not None:
            if time_mode_enabled:
                if time_snapshot is not None:
                    # Verwende total_frames aus snapshot
                    simulation_time = 0.2  # Standard
                    fdtd_results = calc_spl.get("fdtd_simulation")
                    if isinstance(fdtd_results, dict) and fdtd_results:
                        # Versuche simulation_time aus den FDTD-Daten zu holen
                        # (wird aktuell nicht gespeichert, daher Standard verwenden)
                        pass
                    draw_spl_plotter.update_time_control(
                        True,
                        snapshot_total_frames,  # Verwende total_frames direkt
                        self._time_frame_index % snapshot_total_frames,
                        simulation_time,
                    )
                else:
                    # Fallback: Verwende frames_per_period
                    frames = max(1, self._time_frames_per_period)
                    draw_spl_plotter.update_time_control(True, frames + 1, self._time_frame_index, 0.2)
            else:
                draw_spl_plotter.update_time_control(False, self._time_frames_per_period + 1, self._time_frame_index, 0.2)
        self.colorbar_canvas.draw()
        if reset_camera:
            self.reset_zoom()

        if update_axes:
            self.plot_xaxis()
            self.plot_yaxis()
            self.plot_polar_pattern()

    def _calculate_phase_alignment_field(self) -> bool:
        """Berechnet Phasendifferenzen für den Phase-Mode."""
        calc_spl = getattr(self.container, 'calculation_spl', {})
        if not isinstance(calc_spl, dict):
            return False
        if self._is_empty_data(calc_spl.get('sound_field_p')):
            return False

        active_arrays = [
            arr for arr in self.settings.speaker_arrays.values()
            if not getattr(arr, 'mute', False) and not getattr(arr, 'hide', False)
        ]
        if len(active_arrays) == 0:
            return False

        calculator = SoundFieldCalculatorPhaseDiff(self.settings, self.container.data, calc_spl)
        calculator.set_data_container(self.container)
        calculator.calculate_phase_alignment()
        return True

    def _is_empty_data(self, sound_field_p, allow_all_zero: bool = False):
        """Prüft ob sound_field_p nur NaN oder Nullen enthält"""
        # Prüfe ob sound_field_p None oder leer ist
        if sound_field_p is None:
            return True
        
        try:
            # Konvertiere zu numpy array falls noch nicht geschehen
            arr = np.array(sound_field_p)
            
            # Prüfe ob Array leer ist
            if arr.size == 0:
                return True
            
            all_nan = np.all(np.isnan(arr))
            all_zero = np.all(arr == 0)
            if allow_all_zero:
                return all_nan
            return all_nan or all_zero
        except Exception:
            return True

    def _get_current_spl_plotter(self):
        """
        Gibt den SPL-Plotter zurück (erstellt ihn lazy beim ersten Aufruf).
        Der Plotter unterstützt beide Colorization-Modi.
        """
        if self.draw_spl_plotter is None:
            self._initialize_spl_plotter()
        return self.draw_spl_plotter

    def plot_xaxis(self):
        # Aufrufen der plot_xaxis Methode von DrawSPLPlot_Xaxis
        self.draw_spl_plot_xaxis.plot_xaxis(self.container.calculation_axes)

        # Aktualisieren des Plots
        self.matplotlib_canvas_xaxis.draw()

    def plot_yaxis(self):
        # Aufrufen der plot_yaxis Methode von DrawSPLPlot_Yaxis
        self.draw_spl_plot_yaxis.plot_yaxis(self.container.calculation_axes)

        # Aktualisieren des Plots
        self.matplotlib_canvas_yaxis.draw()

    def plot_polar_pattern(self):
        polar_data = self.container.get_polar_data()

        # Update Polar-Plot (zeigt Empty Plot wenn keine Daten vorhanden)
        self.draw_polar_pattern.update_polar_pattern(polar_data)

        # Aktualisieren des Plots
        self.matplotlib_canvas_polar_pattern.draw()

    def update_plot_Stacks2SPL(self):
        """Kompatibilitätsmethode – aktualisiert die 3D-Overlays."""
        self._update_overlays()

    def plot_lines(self):
        """Kompatibilitätsmethode – aktualisiert die 3D-Overlays."""
        self._update_overlays()

    def plot_impulse_points(self):
        """Kompatibilitätsmethode – aktualisiert die 3D-Overlays."""
        self._update_overlays()

    def _update_overlays(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.update_overlays(self.settings, self.container)



    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)  # Entfernt das Widget vom Layout, löscht es aber nicht

    def zoom(self, event):
        """Handler für Mausrad-Zoom Events in den Plots"""
        if event.inaxes:
            ax = event.inaxes
            scale_factor = 1.1 if event.button == 'up' else 1/1.1

            # Hole aktuelle Grenzen (x für horizontal, y für vertikal)
            x_min, x_max = ax.get_xlim()  # Horizontale Grenzen
            y_min, y_max = ax.get_ylim()  # Vertikale Grenzen

            # Mausposition als Zentrum des Zooms
            x_center = event.xdata
            y_center = event.ydata

            # Berechne neue Grenzen
            x_new_min = x_center - (x_center - x_min) * scale_factor
            x_new_max = x_center + (x_max - x_center) * scale_factor
            y_new_min = y_center - (y_center - y_min) * scale_factor
            y_new_max = y_center + (y_max - y_center) * scale_factor

            # Setze neue Grenzen
            ax.set_xlim(x_new_min, x_new_max)  # Horizontale Grenzen
            ax.set_ylim(y_new_min, y_new_max)  # Vertikale Grenzen

            ax.figure.canvas.draw_idle()

    def pan(self, event):
        """Handler für Maus-Pan Events (Linke Maustaste)"""
        if event.inaxes and event.button == 1:  # Linke Maustaste
            ax = event.inaxes
            ax._pan_start = (event.xdata, event.ydata)
            ax.figure.canvas.mpl_connect('motion_notify_event', self.on_pan)
            ax.figure.canvas.mpl_connect('button_release_event', self.on_pan_release)


# ----- SIGNAL HANDLER -----

    def on_plot_mode_changed(self, selection: str):
        """Zentrale Reaktion auf einen Moduswechsel."""
        if not self._is_mode_allowed(selection):
            self._update_plot_mode_availability()
            return
        setattr(self.settings, 'spl_plot_mode', selection)
        self._sync_plot_mode_actions(selection)
        calc_spl = getattr(self.container, 'calculation_spl', None)
        
        # Wenn "SPL over time" aktiviert wird, prüfe ob FDTD berechnet werden soll
        if selection == "SPL over time":
            print(f"[SPL over time] Modus aktiviert, autocalc prüfen...")
            # Prüfe ob FDTD-Daten bereits vorhanden sind
            fdtd_results = calc_spl.get("fdtd_simulation") if isinstance(calc_spl, dict) else None
            has_fdtd = isinstance(fdtd_results, dict) and bool(fdtd_results)
            print(f"[SPL over time] FDTD-Daten vorhanden: {has_fdtd}")
            
            if not has_fdtd:
                # Prüfe ob autocalc aktiv ist (show_in_plot)
                autocalc = calc_spl.get("show_in_plot", True) if isinstance(calc_spl, dict) else True
                print(f"[SPL over time] autocalc={autocalc}")
                if autocalc:
                    # FDTD berechnen
                    if hasattr(self.main_window, 'calculate_fdtd'):
                        print("[SPL over time] Starte FDTD-Berechnung...")
                        self.main_window.calculate_fdtd(show_progress=True, update_plot=True)
                        return
                    else:
                        print("[SPL over time] calculate_fdtd nicht verfügbar")
            else:
                # FDTD-Daten vorhanden - nur Plot aktualisieren
                print("[SPL over time] FDTD-Daten vorhanden, aktualisiere Plot")
                if hasattr(self.main_window, 'plot_spl'):
                    self.main_window.plot_spl(update_axes=False)
                return
        
        if selection == "Phase alignment" and isinstance(calc_spl, dict):
            has_data = self._calculate_phase_alignment_field()
            if not has_data:
                self.show_empty_spl()
                return

        if not isinstance(calc_spl, dict):
            plotter = self._get_current_spl_plotter()
            if plotter is not None:
                plotter.render()
            return

        get_id = getattr(self.main_window, 'get_selected_speaker_array_id', None)
        speaker_array_id = get_id() if callable(get_id) else None
        self.plot_spl(self.settings, speaker_array_id, update_axes=False)

    def _on_time_slider_changed(self, value: int):
        # total_frames = frames_per_period + 1 (inkl. Frame 0)
        total_frames = max(1, self._time_frames_per_period + 1)
        new_value = int(value) % total_frames
        if new_value == self._time_frame_index:
            return
        self._time_frame_index = new_value
        self.plot_spl(self.settings, None, update_axes=False)

    def on_pan(self, event):
        """Handler für Mausbewegung während des Pannens"""
        if event.inaxes and hasattr(event.inaxes, '_pan_start'):
            ax = event.inaxes
            dx = event.xdata - ax._pan_start[0]
            dy = event.ydata - ax._pan_start[1]
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.set_xlim(x_min - dx, x_max - dx)
            ax.set_ylim(y_min - dy, y_max - dy)
            ax.figure.canvas.draw_idle()

    def on_pan_release(self, event):
        """Handler für Loslassen der Maustaste nach dem Pannen"""
        if event.inaxes:
            ax = event.inaxes
            ax.figure.canvas.mpl_disconnect(ax.figure.canvas.mpl_connect('motion_notify_event', self.on_pan))
            ax.figure.canvas.mpl_disconnect(ax.figure.canvas.mpl_connect('button_release_event', self.on_pan_release))
            if hasattr(ax, '_pan_start'):
                del ax._pan_start

    def on_resize(self, event):
        # Layout wird von den individuellen Plot-Klassen gesteuert
        # Trigger redraw für alle Plots
        try:
            if self.draw_spl_plotter is not None:
                self.draw_spl_plotter.render()
            self.matplotlib_canvas_xaxis.draw_idle()
            self.matplotlib_canvas_yaxis.draw_idle()
            self.matplotlib_canvas_polar_pattern.draw_idle()
        except Exception:
            pass

    def reset_zoom(self):
        plotter = self._get_current_spl_plotter()
        if plotter is None:
            return
        try:
            plotter.plotter.reset_camera()
            plotter.render()
        except Exception:
            pass


# ----- PARAMETER SETZEN -----

    def set_splitter_positions(self):
        # Hole die Referenzen zu den Splittern
        self.horizontal_splitter_top = self.main_window.ui.horizontal_splitter_top
        self.vertical_splitter = self.main_window.ui.vertical_splitter
        self.horizontal_splitter_bottom = self.main_window.ui.horizontal_splitter_bottom

    def set_default_view(self):
        """Standard-Ansicht mit gleichmäßiger Aufteilung"""
        self.horizontal_splitter_top.setSizes([597, 250])     # SPLPlot und YAxis
        self.vertical_splitter.setSizes([245, 177])           # Oberer und unterer Bereich
        self.horizontal_splitter_bottom.setSizes([597, 250])  # XAxis und PolarPattern

    def set_focus_spl(self):
        """Maximiert den SPL-Plot"""
        self.horizontal_splitter_top.setSizes([10000, 0])    # Maximum links (SPL), Minimum rechts (YAxis)
        self.vertical_splitter.setSizes([10000, 0])          # Maximum oben, Minimum unten
        self.horizontal_splitter_bottom.setSizes([1, 1])     # Beide minimal (XAxis und PolarPattern)
    
    def set_focus_yaxis(self):
        """Minimiert die Achsenplots"""
        self.horizontal_splitter_top.setSizes([0, 10000])   # Minimale YAxis
        self.vertical_splitter.setSizes([10000 , 0])     # Minimale XAxis
        self.horizontal_splitter_bottom.setSizes([1, 1])  # Normal PolarPattern
  
    def set_focus_xaxis(self):
        """Alle Plots gleich groß"""
        self.horizontal_splitter_top.setSizes([10000, 0])
        self.vertical_splitter.setSizes([0, 10000])
        self.horizontal_splitter_bottom.setSizes([10000, 0])

    def set_focus_polar(self):
        """Maximiert den Polar Pattern Plot"""
        self.horizontal_splitter_top.setSizes([0, 10000])
        self.vertical_splitter.setSizes([0, 10000])
        self.horizontal_splitter_bottom.setSizes([0, 10000])  # Gleiche Größe für beide

    def set_view_isometric(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_isometric()

    def set_view_top(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_top()

    def set_view_side_y(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_side_y()

    def set_view_side_x(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_side_x()


    # ========== DEBUG-METHODEN FÜR SPLITTER ==========

    def _debug_splitter_top(self, pos, index):
        """Debug-Ausgabe für horizontal_splitter_top (SPLPlot und YAxis)"""
        sizes = self.horizontal_splitter_top.sizes()
        print(f"🔧 DEBUG horizontal_splitter_top: setSizes({sizes})  # SPLPlot und YAxis")
    
    def _debug_splitter_vertical(self, pos, index):
        """Debug-Ausgabe für vertical_splitter (Oberer und unterer Bereich)"""
        sizes = self.vertical_splitter.sizes()
        print(f"🔧 DEBUG vertical_splitter: setSizes({sizes})  # Oberer und unterer Bereich")
    
    def _debug_splitter_bottom(self, pos, index):
        """Debug-Ausgabe für horizontal_splitter_bottom (XAxis und PolarPattern)"""
        sizes = self.horizontal_splitter_bottom.sizes()
        print(f"🔧 DEBUG horizontal_splitter_bottom: setSizes({sizes})  # XAxis und PolarPattern")

