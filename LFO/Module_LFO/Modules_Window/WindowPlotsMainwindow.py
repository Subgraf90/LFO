import os
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
    from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3D import DrawSPLPlot3D
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
        # Colorbar-Canvas soll expandieren
        self.colorbar_canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.colorbar_canvas, 1)  # Stretch-Faktor 1 für Expansion

        self.draw_spl_plotter = None
        self.active_spl_widget = None

        self.plot_mode_menu: Optional[QtWidgets.QMenu] = None
        self.plot_mode_context_group: Optional[QtWidgets.QActionGroup] = None
        self.plot_mode_menu_group: Optional[QtWidgets.QActionGroup] = None
        self.plot_mode_menu_actions: dict[str, QtWidgets.QAction] = {}
        self.menu_mode_actions: dict[str, QtWidgets.QAction] = {}
        self._setup_plot_mode_controls()
        self._camera_debug_enabled = self._resolve_camera_debug_flag()

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
        
        # Verbinde Mouse-Move-Events für Mauspositions-Anzeige
        self._setup_mouse_position_tracking()
        
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
            # PyVista nicht verfügbar - setze plotter auf None und gib Warnung aus
            import warnings
            warnings.warn(
                "PyVista und PyVistaQt sind nicht verfügbar. 3D-SPL-Plot wird deaktiviert. "
                "Bitte installieren mit: pip install pyvista pyvistaqt",
                ImportWarning
            )
            self.draw_spl_plotter = None
            return
        self.draw_spl_plotter = DrawSPLPlot3D(self.main_window.ui.SPLPlot, self.settings, self.colorbar_ax)
        # Setze main_window-Referenz für Mouse-Position-Tracking
        if self.draw_spl_plotter:
            self.draw_spl_plotter.main_window = self.main_window
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
        print(f"[PLOT] show_empty_plots() aufgerufen (preserve_camera={preserve_camera}, view={view})")
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
            # 🎯 FIX: Setze Signatur zurück VOR update_overlays(), damit draw_surfaces() definitiv neu gezeichnet wird
            # Das Problem: update_overlays() prüft container.calculation_spl (das noch vorhanden ist),
            # setzt create_empty_plot_surfaces=False, und die Signatur-Prüfung verhindert dann das erneute Zeichnen
            if hasattr(plotter, 'overlay_surfaces'):
                plotter.overlay_surfaces._last_surfaces_state = None
            plotter.update_overlays(self.settings, self.container)
            # 🎯 FIX: Setze Signatur erneut zurück NACH update_overlays(), damit draw_surfaces() mit create_empty_plot_surfaces=True definitiv neu gezeichnet wird
            # (update_overlays() könnte die Signatur bereits gesetzt haben)
            if hasattr(plotter, 'overlay_surfaces'):
                plotter.overlay_surfaces._last_surfaces_state = None
                # Erstelle graue Flächen für enabled Surfaces (nur im leeren Plot)
                # Dies stellt sicher, dass bei disabled Checkbox der graue Empty Plot korrekt angezeigt wird
                plotter.overlay_surfaces.draw_surfaces(self.settings, self.container, create_empty_plot_surfaces=True)
            # 🎯 FIX: Render explizit aufrufen, damit Änderungen sofort sichtbar sind
            plotter.render()
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
            # 🎯 FIX: Entferne ALLE SPL-Daten aus Container, damit update_overlays() weiß, dass keine SPL-Daten vorhanden sind
            # Dies verhindert, dass alte SPL-Daten wieder angezeigt werden
            if hasattr(self.container, 'calculation_spl') and isinstance(self.container.calculation_spl, dict):
                # Entferne alle SPL-bezogenen Daten
                self.container.calculation_spl.pop('sound_field_p', None)
                self.container.calculation_spl.pop('sound_field_x', None)
                self.container.calculation_spl.pop('sound_field_y', None)
                # Entferne auch surface_grids und surface_results, damit keine alten Daten verwendet werden
                if 'surface_grids' in self.container.calculation_spl:
                    self.container.calculation_spl['surface_grids'].clear()
                if 'surface_results' in self.container.calculation_spl:
                    self.container.calculation_spl['surface_results'].clear()
            
            plotter.initialize_empty_scene(preserve_camera=preserve_camera)
            # 🎯 FIX: Stelle sicher, dass der Plotter gerendert wird, damit die entfernten Actors wirklich verschwinden
            if hasattr(plotter, 'render'):
                plotter.render()
            
            # Aktualisiere Overlays (ohne graue Flächen für enabled Surfaces)
            plotter.update_overlays(self.settings, self.container)
            # Erstelle graue Flächen für enabled Surfaces (nur im leeren Plot)
            if hasattr(plotter, 'overlay_surfaces'):
                plotter.overlay_surfaces.draw_surfaces(self.settings, self.container, create_empty_plot_surfaces=True)
            
            # 🎯 FIX: Render nochmal, damit alle Änderungen sichtbar werden
            if hasattr(plotter, 'render'):
                plotter.render()
            
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
        """Setzt die X- und Y-Achsen-Plots auf eine leere Darstellung."""
        # 🎯 FIX: Lösche alte Daten aus calculation_axes, damit keine -200 dB Linien mehr geplottet werden
        # Behalte aber Snapshots bei (falls vorhanden)
        if hasattr(self.container, 'calculation_axes') and isinstance(self.container.calculation_axes, dict):
            # Setze aktuelle_simulation auf leere Daten mit show_in_plot=False
            self.container.calculation_axes["aktuelle_simulation"] = {
                "x_data_xaxis": [],
                "y_data_xaxis": [],
                "x_data_yaxis": [],
                "y_data_yaxis": [],
                "show_in_plot": False,
                "color": "#6A5ACD",
                "segment_boundaries_xaxis": [],
                "segment_boundaries_yaxis": []
            }
        
        # 🎯 FIX: Prüfe ob Snapshots vorhanden sind - wenn ja, plotte diese statt leere Plots
        calculation_axes = self.container.calculation_axes
        has_snapshots = False
        if isinstance(calculation_axes, dict):
            # Prüfe ob Snapshots vorhanden sind (alle Keys außer "aktuelle_simulation")
            snapshot_keys = [k for k in calculation_axes.keys() if k != "aktuelle_simulation"]
            for key in snapshot_keys:
                snapshot_data = calculation_axes.get(key, {})
                if isinstance(snapshot_data, dict) and snapshot_data.get("show_in_plot", False):
                    # Prüfe ob tatsächlich Daten vorhanden sind
                    has_x_data = ('x_data_xaxis' in snapshot_data and 'y_data_xaxis' in snapshot_data and
                                 len(snapshot_data.get('x_data_xaxis', [])) > 0 and
                                 len(snapshot_data.get('y_data_xaxis', [])) > 0)
                    has_y_data = ('x_data_yaxis' in snapshot_data and 'y_data_yaxis' in snapshot_data and
                                 len(snapshot_data.get('x_data_yaxis', [])) > 0 and
                                 len(snapshot_data.get('y_data_yaxis', [])) > 0)
                    if has_x_data or has_y_data:
                        has_snapshots = True
                        break
        
        if has_snapshots:
            # Snapshots vorhanden → plotte diese (plot_xaxis/plot_yaxis prüfen selbst, ob Daten vorhanden sind)
            self.plot_xaxis()
            self.plot_yaxis()
        else:
            # Keine Snapshots → zeige leere Plots
            self.draw_spl_plot_xaxis.initialize_empty_plots()
            self.matplotlib_canvas_xaxis.draw()
            self.draw_spl_plot_yaxis.initialize_empty_plots()
            self.matplotlib_canvas_yaxis.draw()

    def show_empty_polar(self):
        """Setzt den Polar-Plot zurück."""
        if hasattr(self.draw_polar_pattern, 'initialize_empty_plots'):
            self.draw_polar_pattern.initialize_empty_plots()
            self.matplotlib_canvas_polar_pattern.draw()

    def update_plots_for_surface_state(self):
        """
        🎯 ZENTRALE METHODE: Analysiert Surface-Status und koordiniert Berechnung + Plotting.
        
        Kategorisiert alle Surfaces nach ihrem Status (enabled/hidden):
        - enabled + nicht-hidden → Für Berechnung (SoundfieldCalculator)
        - disabled + nicht-hidden → Empty Plot (gestrichelt im Overlay)
        - hidden → Nicht geplottet
        
        Triggert entsprechend:
        - Berechnung wenn enabled Surfaces vorhanden
        - Empty Plot wenn keine enabled Surfaces vorhanden
        - Overlay-Update für visuelle Darstellung
        """
        # ============================================================
        # SCHRITT 1: Analysiere Surface-Status
        # ============================================================
        surface_definitions = getattr(self.settings, 'surface_definitions', {})
        if not isinstance(surface_definitions, dict):
            # Keine Surface-Definitionen → Empty Plot
            self.show_empty_plots()
            return
        
        # Kategorisiere Surfaces
        enabled_surfaces = []      # Für Berechnung: enabled + nicht-hidden
        empty_plot_surfaces = []   # Für Empty Plot: disabled + nicht-hidden
        hidden_surfaces = []       # Nicht geplottet: hidden
        
        # #region agent log
        import json
        import time as time_module
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "WindowPlotsMainwindow.py:update_plots_for_surface_state:before_surface_loop",
                    "message": "Start surface categorization",
                    "data": {
                        "total_surfaces": len(surface_definitions),
                        "surface_ids": list(surface_definitions.keys())[:5]  # First 5 for brevity
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 🎯 NEU: Erstelle Mapping von group_id zu Gruppen-Status für schnellen Zugriff
        surface_groups = getattr(self.settings, 'surface_groups', {})
        group_status: dict[str, dict[str, bool]] = {}  # group_id -> {'enabled': bool, 'hidden': bool}
        if isinstance(surface_groups, dict):
            for group_id, group_data in surface_groups.items():
                if hasattr(group_data, 'enabled'):
                    group_status[group_id] = {
                        'enabled': bool(group_data.enabled),
                        'hidden': bool(getattr(group_data, 'hidden', False))
                    }
                elif isinstance(group_data, dict):
                    group_status[group_id] = {
                        'enabled': bool(group_data.get('enabled', True)),
                        'hidden': bool(group_data.get('hidden', False))
                    }
        
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "WindowPlotsMainwindow.py:update_plots_for_surface_state:group_status",
                    "message": "Group status loaded",
                    "data": {
                        "group_status": {k: v for k, v in list(group_status.items())[:5]}  # First 5 for brevity
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        for surface_id, surface_def in surface_definitions.items():
            if hasattr(surface_def, "enabled") and hasattr(surface_def, "hidden"):
                enabled = bool(surface_def.enabled)
                hidden = bool(surface_def.hidden)
                group_id = getattr(surface_def, 'group_id', None)
            else:
                enabled = bool(surface_def.get('enabled', False)) if isinstance(surface_def, dict) else False
                hidden = bool(surface_def.get('hidden', False)) if isinstance(surface_def, dict) else False
                group_id = surface_def.get('group_id') if isinstance(surface_def, dict) else None
            
            # 🎯 NEU: Berücksichtige Gruppen-Status (wie in draw_surfaces)
            if group_id and group_id in group_status:
                group_enabled = group_status[group_id]['enabled']
                group_hidden = group_status[group_id]['hidden']
                
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "WindowPlotsMainwindow.py:update_plots_for_surface_state:group_override",
                            "message": "Surface group status override",
                            "data": {
                                "surface_id": str(surface_id),
                                "group_id": group_id,
                                "surface_enabled": enabled,
                                "surface_hidden": hidden,
                                "group_enabled": group_enabled,
                                "group_hidden": group_hidden
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Wenn Gruppe hidden ist → Surface komplett überspringen
                if group_hidden:
                    hidden = True
                # Wenn Gruppe disabled ist → Surface als disabled behandeln
                elif not group_enabled:
                    enabled = False
            
            if hidden:
                hidden_surfaces.append(surface_id)
            elif enabled:
                enabled_surfaces.append(surface_id)
            else:
                empty_plot_surfaces.append(surface_id)
        
        # ============================================================
        # SCHRITT 2: Entscheide Berechnung oder Empty Plot
        # ============================================================
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "WindowPlotsMainwindow.py:update_plots_for_surface_state:surface_categorization",
                    "message": "Surface categorization complete",
                    "data": {
                        "enabled_count": len(enabled_surfaces),
                        "empty_plot_count": len(empty_plot_surfaces),
                        "hidden_count": len(hidden_surfaces),
                        "enabled_ids": enabled_surfaces[:5],  # First 5 for brevity
                        "empty_plot_ids": empty_plot_surfaces[:5]
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        has_enabled_surface = len(enabled_surfaces) > 0
        has_empty_plot_surface = len(empty_plot_surfaces) > 0
        has_any_visible_surface = has_enabled_surface or has_empty_plot_surface
        
        if not has_any_visible_surface:
            # Alle Surfaces sind versteckt → SPL Plot komplett entfernen
            # (keine Surfaces zum Zeichnen, auch keine Empty Plot Surfaces)
            plotter = self._get_current_spl_plotter()
            if plotter is not None:
                plotter.initialize_empty_scene(preserve_camera=True)
                # Keine Surfaces zeichnen, da alle versteckt sind
                plotter.update_overlays(self.settings, self.container)
                plotter.render()
                self.colorbar_canvas.draw()
            return
        
        if not has_enabled_surface:
            # Keine enabled Surfaces, aber es gibt disabled Surfaces → Empty Plot
            # show_empty_plots() ruft bereits update_overlays() und draw_surfaces() mit create_empty_plot_surfaces=True auf
            self.show_empty_plots()
            return
        
        # ============================================================
        # SCHRITT 3: Enabled Surfaces vorhanden → Trigger Berechnung
        # ============================================================
        # SoundfieldCalculator holt sich enabled Surfaces automatisch via _get_enabled_surfaces()
        # (filtert bereits auf enabled + nicht-hidden)
        
        # Prüfe ob aktiver Speaker vorhanden ist (nicht mute, nicht hide)
        has_active_speaker = False
        speaker_array_id = None
        if hasattr(self.main_window, 'sources_instance') and self.main_window.sources_instance:
            if hasattr(self.main_window, 'get_selected_speaker_array_id'):
                speaker_array_id = self.main_window.get_selected_speaker_array_id()
                if speaker_array_id is not None:
                    speaker_array = self.settings.get_speaker_array(speaker_array_id)
                    if speaker_array is not None:
                        # 🎯 KORREKTUR: Prüfe ob Speaker aktiv ist (nicht mute, nicht hide)
                        is_mute = getattr(speaker_array, 'mute', False)
                        is_hide = getattr(speaker_array, 'hide', False)
                        if not is_mute and not is_hide:
                            has_active_speaker = True
        
        # 🎯 NEU: Prüfe ob für enabled Surfaces SPL-Daten fehlen
        # Wenn ein Surface enabled wird, aber keine SPL-Daten vorhanden sind, sollte neu berechnet werden
        needs_recalculation = False
        if enabled_surfaces:
            calc_spl = getattr(self.container, 'calculation_spl', {}) if hasattr(self.container, 'calculation_spl') else {}
            if isinstance(calc_spl, dict):
                surface_grids_data = calc_spl.get('surface_grids', {})
                surface_results_data = calc_spl.get('surface_results', {})
                
                # Prüfe ob für mindestens ein enabled Surface keine SPL-Daten vorhanden sind
                for surface_id in enabled_surfaces:
                    has_grid_data = isinstance(surface_grids_data, dict) and surface_id in surface_grids_data
                    has_result_data = isinstance(surface_results_data, dict) and surface_id in surface_results_data
                    if not has_grid_data and not has_result_data:
                        needs_recalculation = True
                        break
        
        # Trigger Berechnung oder Plot-Update
        # 🚀 OPTIMIERUNG: Flag setzen, um zu verhindern, dass update_overlays() doppelt aufgerufen wird
        plot_spl_called = False
        
        if has_active_speaker and hasattr(self.main_window, 'update_speaker_array_calculations'):
            # 🎯 KORREKTUR: Neuberechnung wenn:
            # 1. Aktiver Speaker vorhanden ist (nicht mute, nicht hide)
            # 2. ODER wenn enabled Surfaces vorhanden sind, aber keine SPL-Daten (needs_recalculation)
            # Wenn ein Surface von disabled auf enabled gestellt wird und ein aktiver Speaker vorhanden ist,
            # soll dieses Surface berechnet werden
            # update_speaker_array_calculations() ruft intern plot_spl() auf (über calculate_spl()),
            # was wiederum update_overlays() aufruft, daher ist der Aufruf am Ende redundant
            self.main_window.update_speaker_array_calculations()
            plot_spl_called = True  # plot_spl() wird intern aufgerufen
        elif needs_recalculation and hasattr(self.main_window, 'update_speaker_array_calculations'):
            # 🎯 NEU: Auch wenn kein aktiver Speaker, aber enabled Surfaces ohne SPL-Daten → Neuberechnung
            # (kann vorkommen wenn Speaker später aktiviert wird)
            self.main_window.update_speaker_array_calculations()
            plot_spl_called = True
        elif hasattr(self.main_window, 'plot_spl'):
            # Nur Plot-Update (wenn bereits Daten vorhanden, aber kein aktiver Speaker)
            # plot_spl() ruft bereits update_overlays() intern auf,
            # daher müssen wir es am Ende NICHT erneut aufrufen
            self.main_window.plot_spl(update_axes=False)
            plot_spl_called = True
        else:
            # Keine Source vorhanden, aber Surfaces existieren → zeige leeren Plot mit Surfaces
            # Stelle sicher, dass der Plotter initialisiert ist
            print(f"[PLOT] Keine Source vorhanden → initialize_empty_scene()")
            if self.draw_spl_plotter is not None:
                self.draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
        
        # ============================================================
        # SCHRITT 4: Overlays aktualisieren (für visuelle Darstellung)
        # ============================================================
        # PlotSPL3DOverlays.draw_surfaces() zeichnet automatisch:
        # - enabled + nicht-hidden → Normale Linien (SPL Plot)
        # - disabled + nicht-hidden → Gestrichelte Linien (Empty Plot)
        # - hidden → Nicht geplottet
        # 🚀 OPTIMIERUNG: update_overlays() nur aufrufen, wenn plot_spl() NICHT aufgerufen wurde,
        # da plot_spl() bereits update_overlays() intern aufruft (verhindert doppelte Aufrufe)
        if not plot_spl_called and self.draw_spl_plotter and hasattr(self.draw_spl_plotter, 'update_overlays'):
            self.draw_spl_plotter.update_overlays(self.settings, self.container)
            # 🎯 FIX: Render explizit aufrufen, damit Änderungen sofort sichtbar sind
            if hasattr(self.draw_spl_plotter, 'render'):
                self.draw_spl_plotter.render()

    def plot_spl(self, settings, speaker_array_id, update_axes=True, reset_camera=False):
        """
        Zeichnet den SPL-Plot
        
        Args:
            settings: Settings-Objekt
            speaker_array_id: ID des Lautsprecher-Arrays
            update_axes: Wenn True, werden auch X/Y-Achsen und Polar aktualisiert
                        (False beim Init, da bereits durch __init__ initialisiert)
        """
        plot_mode = getattr(settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
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
        
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "G",
                    "location": "WindowPlotsMainwindow.py:plot_spl:mode_detection",
                    "message": "Plot-Modus erkannt",
                    "data": {
                        "plot_mode": plot_mode,
                        "field_key": field_key,
                        "time_mode_enabled": time_mode_enabled,
                        "is_phase_mode": plot_mode == "Phase alignment"
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        draw_spl_plotter = self._get_current_spl_plotter()

        if not self.container.calculation_spl.get("show_in_plot", True):
            if draw_spl_plotter is not None:
                draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
                # 🎯 FIX: update_overlays() entfernt - Overlays (Lautsprecher) werden nur bei UI-Parameteränderungen aktualisiert
                # SPL-Plots sollen nur ihre eigenen Aufgaben ausführen
                # draw_spl_plotter.update_overlays(self.settings, self.container)
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
            # #region agent log
            try:
                import json
                import time as time_module
                phase_diff_before = calc_spl.get('sound_field_phase_diff')
                phase_shape_before = None
                phase_valid_before = None
                if phase_diff_before is not None:
                    try:
                        if isinstance(phase_diff_before, list):
                            phase_array = np.array(phase_diff_before)
                        else:
                            phase_array = phase_diff_before
                        phase_shape_before = list(phase_array.shape) if hasattr(phase_array, 'shape') else None
                        phase_valid_before = int(np.sum(~np.isnan(phase_array))) if hasattr(phase_array, 'shape') else None
                    except Exception:
                        pass
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "J3",
                        "location": "WindowPlotsMainwindow.py:plot_spl:phase_data_check",
                        "message": "Phase-Daten-Prüfung in plot_spl",
                        "data": {
                            "has_phase_diff": "sound_field_phase_diff" in calc_spl,
                            "phase_shape": phase_shape_before,
                            "phase_valid_count": phase_valid_before,
                            "is_empty": self._is_empty_data(calc_spl.get('sound_field_phase_diff'), allow_all_zero=True) if "sound_field_phase_diff" in calc_spl else True
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            if self._is_empty_data(calc_spl.get('sound_field_phase_diff'), allow_all_zero=True):
                if not self._calculate_phase_alignment_field():
                    calc_spl['sound_field_phase_diff'] = []
            # #region agent log
            try:
                import json
                import time as time_module
                phase_diff_after = calc_spl.get('sound_field_phase_diff')
                phase_shape_after = None
                phase_valid_after = None
                if phase_diff_after is not None:
                    try:
                        if isinstance(phase_diff_after, list):
                            phase_array = np.array(phase_diff_after)
                        else:
                            phase_array = phase_diff_after
                        phase_shape_after = list(phase_array.shape) if hasattr(phase_array, 'shape') else None
                        phase_valid_after = int(np.sum(~np.isnan(phase_array))) if hasattr(phase_array, 'shape') else None
                    except Exception:
                        pass
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "J4",
                        "location": "WindowPlotsMainwindow.py:plot_spl:phase_data_after_check",
                        "message": "Phase-Daten nach Prüfung in plot_spl",
                        "data": {
                            "has_phase_diff": "sound_field_phase_diff" in calc_spl,
                            "phase_shape": phase_shape_after,
                            "phase_valid_count": phase_valid_after
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
        # Prüfe ob gültige Daten vorhanden sind (nicht nur NaN)
        time_snapshot = None
        if time_mode_enabled:
            time_snapshot = self._compute_time_mode_field()
            has_data = time_snapshot is not None
        else:
            # 🎯 FIX: Für Phase-Modus prüfe nur ob Daten vorhanden sind, nicht ob sie gültig sind
            # Phase-Daten können alle NaN sein (z.B. wenn keine Arrays vorhanden sind), sollten aber trotzdem verwendet werden
            if plot_mode == "Phase alignment":
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
                    len(calc_spl.get(field_key, [])) > 0
                    # Für Phase-Modus prüfen wir NICHT ob Daten leer sind (können alle NaN sein)
                )
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
        
        # 🎯 WICHTIG: Prüfe auch ob surface_grids_data vorhanden ist (für vertikale Flächen)
        # Auch wenn globale Daten leer sind, sollten wir plotten, wenn einzelne Surfaces Daten haben
        has_surface_data = False
        if isinstance(calc_spl, dict):
            surface_grids_data = calc_spl.get('surface_grids', {})
            surface_results_data = calc_spl.get('surface_results', {})
            has_surface_data = (
                isinstance(surface_grids_data, dict) and len(surface_grids_data) > 0
            ) or (
                isinstance(surface_results_data, dict) and len(surface_results_data) > 0
            )
        
        if not has_data and isinstance(calc_spl, dict):
            if field_key in calc_spl:
                field_data = calc_spl[field_key]
                is_empty = self._is_empty_data(field_data, allow_all_zero=allow_zero_phase)
            # Prüfe ob surface_grids_data vorhanden ist (für vertikale Flächen)
            surface_grids_data = calc_spl.get('surface_grids', {})
            surface_results_data = calc_spl.get('surface_results', {})
        
        # #region agent log
        import json
        import time as time_module
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "WindowPlotsMainwindow.py:plot_spl:data_check",
                    "message": "Data validation check before plot",
                    "data": {
                        "has_data": has_data,
                        "has_surface_data": has_surface_data,
                        "time_mode_enabled": time_mode_enabled,
                        "field_key": field_key,
                        "calc_spl_keys": list(calc_spl.keys()) if isinstance(calc_spl, dict) else None,
                        "sound_field_x_len": len(calc_spl.get('sound_field_x', [])) if isinstance(calc_spl, dict) else 0,
                        "sound_field_y_len": len(calc_spl.get('sound_field_y', [])) if isinstance(calc_spl, dict) else 0,
                        "field_data_len": len(calc_spl.get(field_key, [])) if isinstance(calc_spl, dict) else 0
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 🎯 WICHTIG: Wenn keine globalen Daten vorhanden sind, aber Surface-Daten existieren,
        # sollten wir trotzdem plotten (z.B. für vertikale Flächen)
        if not has_data and not has_surface_data:
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "WindowPlotsMainwindow.py:plot_spl:early_return",
                        "message": "EARLY RETURN - no data and no surface data",
                        "data": {
                            "has_data": has_data,
                            "has_surface_data": has_surface_data
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            # Keine Daten - zeige leere Szene
            if draw_spl_plotter is not None:
                draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
                # 🎯 FIX: update_overlays() entfernt - Overlays (Lautsprecher) werden nur bei UI-Parameteränderungen aktualisiert
                # SPL-Plots sollen nur ihre eigenen Aufgaben ausführen
                # draw_spl_plotter.update_overlays(self.settings, self.container)
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
            # 🎯 WICHTIG: Wenn keine globalen Daten vorhanden sind, aber Surface-Daten existieren,
            # verwende Dummy-Daten für die globalen Koordinaten (update_spl_plot wird diese ignorieren)
            if has_data:
                sound_field_x = self.container.calculation_spl['sound_field_x']
                sound_field_y = self.container.calculation_spl['sound_field_y']
                sound_field_values = self.container.calculation_spl[field_key]
            elif has_surface_data:
                # Dummy-Daten für globale Koordinaten (werden in update_spl_plot ignoriert, wenn surface_overrides vorhanden sind)
                sound_field_x = np.array([-1.0, 1.0])
                sound_field_y = np.array([-1.0, 1.0])
                sound_field_values = np.array([[0.0, 0.0], [0.0, 0.0]])
            else:
                # Sollte nicht erreicht werden, da wir bereits oben return haben
                return

        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                sound_field_values_arr = np.asarray(sound_field_values) if hasattr(sound_field_values, '__len__') else None
                valid_values = sound_field_values_arr[np.isfinite(sound_field_values_arr)] if sound_field_values_arr is not None and sound_field_values_arr.size > 0 else np.array([])
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "WindowPlotsMainwindow.py:plot_spl:calling_update_spl_plot",
                    "message": "Calling update_spl_plot",
                    "data": {
                        "plot_mode": plot_mode,
                        "field_key": field_key,
                        "sound_field_x_shape": np.array(sound_field_x).shape if hasattr(sound_field_x, '__len__') else None,
                        "sound_field_y_shape": np.array(sound_field_y).shape if hasattr(sound_field_y, '__len__') else None,
                        "sound_field_values_shape": sound_field_values_arr.shape if sound_field_values_arr is not None else None,
                        "sound_field_values_type": type(sound_field_values).__name__,
                        "valid_count": len(valid_values),
                        "min_value": float(np.nanmin(valid_values)) if len(valid_values) > 0 else None,
                        "max_value": float(np.nanmax(valid_values)) if len(valid_values) > 0 else None,
                        "colorization_mode": self.settings.colorization_mode
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 3D-Plot aktualisieren
        draw_spl_plotter.update_spl_plot(
            sound_field_x,
            sound_field_y,
            sound_field_values,
            self.settings.colorization_mode,
        )
        # 🎯 FIX: update_overlays() aufrufen, damit Surface-Overlays (Linien) gezeichnet werden
        # update_spl_plot() zeichnet die SPL-Flächen, aber update_overlays() zeichnet die Surface-Linien
        if hasattr(draw_spl_plotter, 'update_overlays'):
            draw_spl_plotter.update_overlays(self.settings, self.container)
            # Render explizit aufrufen, damit Änderungen sofort sichtbar sind
            if hasattr(draw_spl_plotter, 'render'):
                draw_spl_plotter.render()
        
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
        
        # 🎯 FIX: Prüfe ob sound_field_p vorhanden ist (auch wenn leer)
        # Wenn nicht vorhanden, können wir keine Phase-Daten berechnen
        if calc_spl.get('sound_field_p') is None:
            return False

        active_arrays = [
            arr for arr in self.settings.speaker_arrays.values()
            if not getattr(arr, 'mute', False) and not getattr(arr, 'hide', False)
        ]
        # 🎯 FIX: Berechne Phase-Daten auch wenn keine aktiven Arrays vorhanden sind
        # (dann wird ein leeres Array mit richtiger Form erstellt)
        
        try:
            calculator = SoundFieldCalculatorPhaseDiff(self.settings, self.container.data, calc_spl)
            calculator.set_data_container(self.container)
            calculator.calculate_phase_alignment()
            return True
        except Exception as e:
            print(f"Fehler bei Phase-Alignment-Berechnung: {e}")
            import traceback
            print(traceback.format_exc())
            return False

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
        # 🎯 FIX: Prüfe ob Snapshots vorhanden sind, bevor geprüft wird ob alle Sources muted sind
        # Snapshots sollen auch angezeigt werden, wenn alle Lautsprecher muted sind
        calculation_axes = self.container.calculation_axes
        has_snapshots = False
        if isinstance(calculation_axes, dict):
            # Prüfe ob Snapshots vorhanden sind (alle Keys außer "aktuelle_simulation")
            snapshot_keys = [k for k in calculation_axes.keys() if k != "aktuelle_simulation"]
            for key in snapshot_keys:
                snapshot_data = calculation_axes.get(key, {})
                if isinstance(snapshot_data, dict) and snapshot_data.get("show_in_plot", False):
                    # Prüfe ob tatsächlich Daten vorhanden sind
                    if ('x_data_xaxis' in snapshot_data and 'y_data_xaxis' in snapshot_data and
                        len(snapshot_data.get('x_data_xaxis', [])) > 0 and
                        len(snapshot_data.get('y_data_xaxis', [])) > 0):
                        has_snapshots = True
                        break
        
        # Wenn keine Snapshots vorhanden sind, prüfe ob alle Sources muted sind
        if not has_snapshots:
            has_active_sources = any(
                not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values()
            ) if hasattr(self.settings, 'speaker_arrays') and self.settings.speaker_arrays else False
            
            if not has_active_sources:
                # Alle Sources sind hidden/muted und keine Snapshots → zeige Empty Plot
                self.draw_spl_plot_xaxis.initialize_empty_plots()
                self.matplotlib_canvas_xaxis.draw()
                return
        
        # Aufrufen der plot_xaxis Methode von DrawSPLPlot_Xaxis
        # (plot_xaxis() prüft selbst, ob Daten vorhanden sind)
        self.draw_spl_plot_xaxis.plot_xaxis(calculation_axes)

        # Aktualisieren des Plots
        self.matplotlib_canvas_xaxis.draw()

    def plot_yaxis(self):
        # 🎯 FIX: Prüfe ob Snapshots vorhanden sind, bevor geprüft wird ob alle Sources muted sind
        # Snapshots sollen auch angezeigt werden, wenn alle Lautsprecher muted sind
        calculation_axes = self.container.calculation_axes
        has_snapshots = False
        if isinstance(calculation_axes, dict):
            # Prüfe ob Snapshots vorhanden sind (alle Keys außer "aktuelle_simulation")
            snapshot_keys = [k for k in calculation_axes.keys() if k != "aktuelle_simulation"]
            for key in snapshot_keys:
                snapshot_data = calculation_axes.get(key, {})
                if isinstance(snapshot_data, dict) and snapshot_data.get("show_in_plot", False):
                    # Prüfe ob tatsächlich Daten vorhanden sind
                    if ('x_data_yaxis' in snapshot_data and 'y_data_yaxis' in snapshot_data and
                        len(snapshot_data.get('x_data_yaxis', [])) > 0 and
                        len(snapshot_data.get('y_data_yaxis', [])) > 0):
                        has_snapshots = True
                        break
        
        # Wenn keine Snapshots vorhanden sind, prüfe ob alle Sources muted sind
        if not has_snapshots:
            has_active_sources = any(
                not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values()
            ) if hasattr(self.settings, 'speaker_arrays') and self.settings.speaker_arrays else False
            
            if not has_active_sources:
                # Alle Sources sind hidden/muted und keine Snapshots → zeige Empty Plot
                self.draw_spl_plot_yaxis.initialize_empty_plots()
                self.matplotlib_canvas_yaxis.draw()
                return
        
        # Aufrufen der plot_yaxis Methode von DrawSPLPlot_Yaxis
        # (plot_yaxis() prüft selbst, ob Daten vorhanden sind)
        self.draw_spl_plot_yaxis.plot_yaxis(calculation_axes)

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

            # Prüfe ob es X- oder Y-Achsen-Plot ist
            is_xaxis = ax == self.matplotlib_canvas_xaxis.ax
            is_yaxis = ax == self.matplotlib_canvas_yaxis.ax

            # Setze neue Grenzen
            ax.set_xlim(x_new_min, x_new_max)  # Horizontale Grenzen
            ax.set_ylim(y_new_min, y_new_max)  # Vertikale Grenzen

            # Manuell Callbacks aufrufen, da Matplotlib sie manchmal nicht automatisch auslöst
            if is_xaxis and hasattr(self.draw_spl_plot_xaxis, '_on_xlim_changed'):
                self.draw_spl_plot_xaxis._on_xlim_changed(ax)
            if is_xaxis and hasattr(self.draw_spl_plot_xaxis, '_on_ylim_changed'):
                self.draw_spl_plot_xaxis._on_ylim_changed(ax)
            
            if is_yaxis and hasattr(self.draw_spl_plot_yaxis, '_on_xlim_changed'):
                self.draw_spl_plot_yaxis._on_xlim_changed(ax)
            if is_yaxis and hasattr(self.draw_spl_plot_yaxis, '_on_ylim_changed'):
                self.draw_spl_plot_yaxis._on_ylim_changed(ax)

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
        
        # 🎯 FIX: Für Phase-Modus berechne Phase-Daten nur wenn noch keine vorhanden sind
        # (z.B. wenn sie aus einem Snapshot geladen wurden, sollten diese verwendet werden)
        if selection == "Phase alignment" and isinstance(calc_spl, dict):
            # Prüfe ob bereits Phase-Daten vorhanden sind
            has_existing_phase_data = False
            phase_diff = calc_spl.get("sound_field_phase_diff")
            if phase_diff is not None:
                try:
                    import numpy as np
                    if isinstance(phase_diff, list):
                        phase_array = np.array(phase_diff)
                    else:
                        phase_array = phase_diff
                    # Prüfe ob Array gültige Daten enthält (nicht nur NaN oder leer)
                    if hasattr(phase_array, 'size') and phase_array.size > 0:
                        valid_count = np.sum(np.isfinite(phase_array))
                        if valid_count > 0:
                            has_existing_phase_data = True
                except Exception:
                    pass
            
            # Nur berechnen wenn keine Phase-Daten vorhanden sind
            if not has_existing_phase_data:
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "J1",
                            "location": "WindowPlotsMainwindow.py:on_plot_mode_changed:phase_mode_selected",
                            "message": "Phase-Modus ausgewählt - VOR Berechnung",
                            "data": {
                                "selection": selection,
                                "calc_spl_is_dict": isinstance(calc_spl, dict),
                                "has_phase_diff_before": "sound_field_phase_diff" in calc_spl if isinstance(calc_spl, dict) else False,
                                "phase_diff_type": type(calc_spl.get("sound_field_phase_diff")).__name__ if isinstance(calc_spl, dict) and "sound_field_phase_diff" in calc_spl else None
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                # Versuche Phase-Daten zu berechnen, auch wenn es fehlschlägt
                calc_result = self._calculate_phase_alignment_field()
            else:
                # Phase-Daten bereits vorhanden (z.B. aus Snapshot) - keine Neuberechnung nötig
                calc_result = True
            # Aktualisiere calc_spl nach Berechnung (kann sich geändert haben)
            calc_spl = getattr(self.container, 'calculation_spl', None)
            # #region agent log
            try:
                import json
                import time as time_module
                import numpy as np
                phase_diff = calc_spl.get("sound_field_phase_diff") if isinstance(calc_spl, dict) else None
                phase_shape = None
                phase_valid_count = None
                if phase_diff is not None:
                    try:
                        if isinstance(phase_diff, list):
                            phase_array = np.array(phase_diff)
                        else:
                            phase_array = phase_diff
                        phase_shape = list(phase_array.shape) if hasattr(phase_array, 'shape') else None
                        phase_valid_count = int(np.sum(~np.isnan(phase_array))) if hasattr(phase_array, 'shape') else None
                    except Exception:
                        pass
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "J2",
                        "location": "WindowPlotsMainwindow.py:on_plot_mode_changed:phase_calc_result",
                        "message": "Phase-Berechnung abgeschlossen - NACH Berechnung",
                        "data": {
                            "calc_result": calc_result,
                            "has_phase_diff_after": "sound_field_phase_diff" in calc_spl if isinstance(calc_spl, dict) else False,
                            "phase_diff_type": type(phase_diff).__name__ if phase_diff is not None else None,
                            "phase_shape": phase_shape,
                            "phase_valid_count": phase_valid_count
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion

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
        if not self._try_fast_time_frame_update():
            self.plot_spl(self.settings, None, update_axes=False)

    def _try_fast_time_frame_update(self) -> bool:
        """Schneller Pfad für den Zeit-Slider, bei dem nur die Skalare aktualisiert werden."""
        plot_mode = getattr(self.settings, 'spl_plot_mode', self.PLOT_MODE_OPTIONS[0])
        if plot_mode != "SPL over time":
            return False

        draw_spl_plotter = self._get_current_spl_plotter()
        if draw_spl_plotter is None or not hasattr(draw_spl_plotter, 'update_time_frame_values'):
            return False

        snapshot = self._compute_time_mode_field()
        if snapshot is None:
            return False

        _, snapshot_total_frames, sound_field_x, sound_field_y, snapshot_values = snapshot
        if not draw_spl_plotter.update_time_frame_values(sound_field_x, sound_field_y, snapshot_values):
            return False

        self._time_frames_per_period = max(1, snapshot_total_frames - 1)
        self.settings.fem_time_frames_per_period = self._time_frames_per_period
        draw_spl_plotter.update_time_control(
            True,
            snapshot_total_frames,
            self._time_frame_index % snapshot_total_frames,
            0.2,
        )
        self.colorbar_canvas.draw()
        return True

    def on_pan(self, event):
        """Handler für Mausbewegung während des Pannens"""
        if event.inaxes and hasattr(event.inaxes, '_pan_start'):
            ax = event.inaxes
            dx = event.xdata - ax._pan_start[0]
            dy = event.ydata - ax._pan_start[1]
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_new_min = x_min - dx
            x_new_max = x_max - dx
            y_new_min = y_min - dy
            y_new_max = y_max - dy
            
            ax.set_xlim(x_new_min, x_new_max)
            ax.set_ylim(y_new_min, y_new_max)
            
            # Prüfe ob es X- oder Y-Achsen-Plot ist
            is_xaxis = ax == self.matplotlib_canvas_xaxis.ax
            is_yaxis = ax == self.matplotlib_canvas_yaxis.ax
            
            # Manuell Callbacks aufrufen
            if is_xaxis and hasattr(self.draw_spl_plot_xaxis, '_on_xlim_changed'):
                self.draw_spl_plot_xaxis._on_xlim_changed(ax)
            if is_xaxis and hasattr(self.draw_spl_plot_xaxis, '_on_ylim_changed'):
                self.draw_spl_plot_xaxis._on_ylim_changed(ax)
            
            if is_yaxis and hasattr(self.draw_spl_plot_yaxis, '_on_xlim_changed'):
                self.draw_spl_plot_yaxis._on_xlim_changed(ax)
            if is_yaxis and hasattr(self.draw_spl_plot_yaxis, '_on_ylim_changed'):
                self.draw_spl_plot_yaxis._on_ylim_changed(ax)
            
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
        self._camera_debug("reset_zoom -> plotter.reset_camera()")
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
            self._camera_debug("set_view_isometric() invoked")
            plotter.set_view_isometric()

    def set_view_top(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            self._camera_debug("set_view_top() invoked")
            plotter.set_view_top()

    def set_view_side_y(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_side_y()

    def set_view_side_x(self):
        plotter = self._get_current_spl_plotter()
        if plotter is not None:
            plotter.set_view_side_x()

    def _camera_debug(self, message: str) -> None:
        if self._camera_debug_enabled:
            pass

    def _resolve_camera_debug_flag(self) -> bool:
        env_value = os.environ.get("LFO_DEBUG_CAMERA")
        if env_value is None:
            return False
        env_value = env_value.strip().lower()
        if env_value in {"0", "false", "off"}:
            return False
        if env_value in {"1", "true", "on"}:
            return True
        try:
            return bool(int(env_value))
        except ValueError:
            return True


    # ========== DEBUG-METHODEN FÜR SPLITTER ==========

    def _debug_splitter_top(self, pos, index):
        """Debug-Ausgabe für horizontal_splitter_top (SPLPlot und YAxis)"""
        pass
    
    def _debug_splitter_vertical(self, pos, index):
        """Debug-Ausgabe für vertical_splitter (Oberer und unterer Bereich)"""
        pass
    
    def _debug_splitter_bottom(self, pos, index):
        """Debug-Ausgabe für horizontal_splitter_bottom (XAxis und PolarPattern)"""
        pass

    def _setup_mouse_position_tracking(self):
        """Richtet Mouse-Move-Events für alle Plots ein, um Mausposition anzuzeigen"""
        # X-Achsen-Plot
        self.matplotlib_canvas_xaxis.mpl_connect('motion_notify_event', self._on_mouse_move_xaxis)
        self.matplotlib_canvas_xaxis.mpl_connect('axes_leave_event', self._on_axes_leave)
        
        # Y-Achsen-Plot
        self.matplotlib_canvas_yaxis.mpl_connect('motion_notify_event', self._on_mouse_move_yaxis)
        self.matplotlib_canvas_yaxis.mpl_connect('axes_leave_event', self._on_axes_leave)
        
        # Polar-Plot
        self.matplotlib_canvas_polar_pattern.mpl_connect('motion_notify_event', self._on_mouse_move_polar)
        self.matplotlib_canvas_polar_pattern.mpl_connect('axes_leave_event', self._on_axes_leave)
        
        # 3D-Plot (bereits vorhanden, erweitern)
        if self.draw_spl_plotter and hasattr(self.draw_spl_plotter, 'widget'):
            # Mouse-Tracking ist bereits aktiviert, wir müssen nur den Event-Filter erweitern
            pass

    def _on_axes_leave(self, event):
        """Wird aufgerufen, wenn die Maus den Plot verlässt"""
        self._clear_mouse_position()

    def _on_mouse_move_xaxis(self, event):
        """Handler für Mouse-Move im X-Achsen-Plot"""
        if not event.inaxes or event.inaxes != self.matplotlib_canvas_xaxis.ax:
            self._clear_mouse_position()
            return
        
        try:
            # X-Achse = Position (Meter), Y-Achse = SPL (dB)
            x_pos = event.xdata  # Position in Metern
            y_spl = event.ydata  # SPL in dB
            
            if x_pos is None or y_spl is None:
                self._clear_mouse_position()
                return
            
            # Hole genauen SPL-Wert aus den Daten durch Interpolation
            spl_value = self._get_spl_from_xaxis_plot(x_pos)
            
            if spl_value is not None:
                text = f"X-Axis:\nPos: {x_pos:.2f} m\nSPL: {spl_value:.1f} dB"
            else:
                text = f"X-Axis:\nPos: {x_pos:.2f} m\nSPL: {y_spl:.1f} dB"
            
            self._update_mouse_position(text)
        except Exception as e:
            pass

    def _on_mouse_move_yaxis(self, event):
        """Handler für Mouse-Move im Y-Achsen-Plot"""
        if not event.inaxes or event.inaxes != self.matplotlib_canvas_yaxis.ax:
            self._clear_mouse_position()
            return
        
        try:
            # Y-Achse = Position (Meter), X-Achse = SPL (dB) - invertiert
            y_pos = event.ydata  # Position in Metern
            x_spl = event.xdata  # SPL in dB (invertiert)
            
            if y_pos is None or x_spl is None:
                self._clear_mouse_position()
                return
            
            # Hole genauen SPL-Wert aus den Daten durch Interpolation
            spl_value = self._get_spl_from_yaxis_plot(y_pos)
            
            if spl_value is not None:
                text = f"Y-Axis:\nPos: {y_pos:.2f} m\nSPL: {spl_value:.1f} dB"
            else:
                text = f"Y-Axis:\nPos: {y_pos:.2f} m\nSPL: {x_spl:.1f} dB"
            
            self._update_mouse_position(text)
        except Exception as e:
            pass

    def _on_mouse_move_polar(self, event):
        """Handler für Mouse-Move im Polar-Plot"""
        if not event.inaxes or event.inaxes != self.matplotlib_canvas_polar_pattern.ax:
            self._clear_mouse_position()
            return
        
        try:
            # Polar-Koordinaten: theta (Winkel), r (Radius = SPL)
            theta = event.xdata  # Winkel in Radiant (im Plot-Koordinatensystem)
            r = event.ydata      # Radius = SPL in dB
            
            if theta is None or r is None:
                self._clear_mouse_position()
                return
            
            # Konvertiere Plot-Winkel zu angezeigtem Winkel
            # Plot hat: 0° oben, 90° links, 180° unten, 270° rechts
            # theta=0 (oben) → 0°, theta=π/2 (links) → 90°, theta=π (unten) → 180°, theta=3π/2 (rechts) → 270°
            # Da set_theta_zero_location('N') und set_theta_direction(1) verwendet wird:
            # theta im Plot entspricht direkt dem angezeigten Winkel
            angle_deg = np.rad2deg(theta) % 360
            
            # Im Polar-Plot ist der Radius r direkt der SPL-Wert in dB
            # Der Plot zeigt: r = 0 dB am äußeren Rand, r = -24 dB in der Mitte
            # Der Event-Radius r ist bereits der korrekte SPL-Wert, der im Plot angezeigt wird
            spl_value = float(r) if r is not None else None
            
            # Baue Text zusammen
            if spl_value is not None:
                text = f"Polar:\nAngle: {angle_deg:.1f}°\nSPL: {spl_value:.1f} dB"
            else:
                text = f"Polar:\nAngle: {angle_deg:.1f}°\nSPL: -- dB"
            
            self._update_mouse_position(text)
        except Exception:
            pass

    def _get_spl_from_xaxis_plot(self, x_pos):
        """Holt SPL-Wert aus X-Achsen-Plot durch Interpolation"""
        try:
            calculation_axes = getattr(self.container, 'calculation_axes', {})
            if not calculation_axes:
                return None
            
            # Suche nach aktiven Daten
            for key, value in calculation_axes.items():
                if value.get('show_in_plot', False):
                    if 'x_data_xaxis' in value and 'y_data_xaxis' in value:
                        x_data = np.array(value['x_data_xaxis'])  # SPL-Werte
                        y_data = np.array(value['y_data_xaxis'])  # Positionen
                        
                        # Entferne NaN-Werte
                        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                        if not np.any(valid_mask):
                            continue
                        
                        x_data_clean = x_data[valid_mask]
                        y_data_clean = y_data[valid_mask]
                        
                        # Interpoliere SPL-Wert für gegebene Position
                        if len(y_data_clean) > 1:
                            # Sortiere nach Position
                            sort_idx = np.argsort(y_data_clean)
                            y_sorted = y_data_clean[sort_idx]
                            x_sorted = x_data_clean[sort_idx]
                            
                            # Interpoliere
                            if y_sorted[0] <= x_pos <= y_sorted[-1]:
                                spl_value = np.interp(x_pos, y_sorted, x_sorted)
                                return float(spl_value)
            
            return None
        except Exception as e:
            return None

    def _get_spl_from_yaxis_plot(self, y_pos):
        """Holt SPL-Wert aus Y-Achsen-Plot durch Interpolation"""
        try:
            calculation_axes = getattr(self.container, 'calculation_axes', {})
            if not calculation_axes:
                return None
            
            # Suche nach aktiven Daten
            for key, value in calculation_axes.items():
                if value.get('show_in_plot', False):
                    if 'x_data_yaxis' in value and 'y_data_yaxis' in value:
                        x_data = np.array(value['x_data_yaxis'])  # SPL-Werte
                        y_data = np.array(value['y_data_yaxis'])  # Positionen
                        
                        # Entferne NaN-Werte
                        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                        if not np.any(valid_mask):
                            continue
                        
                        x_data_clean = x_data[valid_mask]
                        y_data_clean = y_data[valid_mask]
                        
                        # Interpoliere SPL-Wert für gegebene Position
                        if len(y_data_clean) > 1:
                            # Sortiere nach Position
                            sort_idx = np.argsort(y_data_clean)
                            y_sorted = y_data_clean[sort_idx]
                            x_sorted = x_data_clean[sort_idx]
                            
                            # Interpoliere
                            if y_sorted[0] <= y_pos <= y_sorted[-1]:
                                spl_value = np.interp(y_pos, y_sorted, x_sorted)
                                return float(spl_value)
            
            return None
        except Exception as e:
            return None

    def _get_spl_from_polar_plot(self, angle_deg):
        """Holt SPL-Wert aus Polar-Plot durch Interpolation"""
        try:
            if not hasattr(self.container, 'calculation_polar'):
                return None
            
            polar_data = self.container.calculation_polar
            if not polar_data:
                return None
            
            # Polar-Daten haben die Struktur: {'angles': array, 'sound_field_p': {freq: array}}
            if 'angles' not in polar_data or 'sound_field_p' not in polar_data:
                return None
            
            angles = np.array(polar_data['angles'])
            sound_field_p = polar_data['sound_field_p']
            
            if len(angles) == 0 or not sound_field_p:
                return None
            
            # Verwende die erste verfügbare Frequenz (oder die aktivste)
            # In der Praxis könnte man die aktuell angezeigte Frequenz verwenden
            if not sound_field_p:
                return None
            
            # Nimm die erste Frequenz (oder die aktivste)
            first_freq = next(iter(sound_field_p.keys()))
            values = np.array(sound_field_p[first_freq])
            
            # Die Werte in calculation_polar sind bereits normalisiert (0 dB Maximum, -24 bis 0 dB)
            # und bereits in dB, daher keine weitere Konvertierung nötig
            values_db = values
            
            # Entferne NaN-Werte
            valid_mask = ~(np.isnan(angles) | np.isnan(values_db))
            if not np.any(valid_mask):
                return None
            
            angles_clean = angles[valid_mask]
            values_clean = values_db[valid_mask]
            
            # Interpoliere SPL-Wert für gegebenen Winkel
            if len(angles_clean) > 1:
                # Normalisiere Winkel auf 0-360
                angle_norm = angle_deg % 360
                
                # Normalisiere auch die Datenwinkel auf 0-360
                angles_normalized = angles_clean % 360
                
                # Sortiere nach Winkel
                sort_idx = np.argsort(angles_normalized)
                angles_sorted = angles_normalized[sort_idx]
                values_sorted = values_clean[sort_idx]
                
                # Interpoliere
                if angles_sorted[0] <= angle_norm <= angles_sorted[-1]:
                    # Normaler Fall: Winkel liegt im Bereich
                    spl_value = np.interp(angle_norm, angles_sorted, values_sorted)
                    return float(spl_value)
                else:
                    # Wrap-around: Winkel liegt außerhalb des Bereichs
                    # Für zirkuläre Interpolation: Füge erste Werte am Ende hinzu und letzte am Anfang
                    # Erweitere Arrays für Wrap-around
                    angles_extended = np.concatenate([
                        angles_sorted - 360,  # Werte von -360 bis 0
                        angles_sorted,        # Werte von 0 bis 360
                        angles_sorted + 360   # Werte von 360 bis 720
                    ])
                    values_extended = np.concatenate([
                        values_sorted,
                        values_sorted,
                        values_sorted
                    ])
                    
                    # Sortiere erweiterte Arrays
                    sort_idx_ext = np.argsort(angles_extended)
                    angles_ext_sorted = angles_extended[sort_idx_ext]
                    values_ext_sorted = values_extended[sort_idx_ext]
                    
                    # Interpoliere im erweiterten Bereich
                    if len(angles_ext_sorted) > 0:
                        spl_value = np.interp(angle_norm, angles_ext_sorted, values_ext_sorted)
                        return float(spl_value)
            
            return None
        except Exception:
            return None

    def _get_spl_from_3d_plot(self, x_pos, y_pos):
        """Holt SPL-Wert aus 3D-Plot durch Interpolation"""
        try:
            calculation_spl = getattr(self.container, 'calculation_spl', {})
            if not calculation_spl:
                return None
            
            sound_field_x = calculation_spl.get('sound_field_x')
            sound_field_y = calculation_spl.get('sound_field_y')
            sound_field_p = calculation_spl.get('sound_field_p')
            
            if not all([sound_field_x, sound_field_y, sound_field_p]):
                return None
            
            x_array = np.array(sound_field_x)
            y_array = np.array(sound_field_y)
            p_array = np.array(sound_field_p)
            
            # Prüfe ob Position im gültigen Bereich liegt
            if (x_array[0] <= x_pos <= x_array[-1] and 
                y_array[0] <= y_pos <= y_array[-1]):
                
                # Interpoliere 2D
                from scipy.interpolate import griddata
                
                # Erstelle Grid
                X, Y = np.meshgrid(x_array, y_array)
                points = np.column_stack([X.ravel(), Y.ravel()])
                values = p_array.ravel()
                
                # Entferne NaN-Werte
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    return None
                
                points_clean = points[valid_mask]
                values_clean = values[valid_mask]
                
                # Interpoliere
                spl_value = griddata(points_clean, values_clean, (x_pos, y_pos), method='linear')
                
                if not np.isnan(spl_value):
                    # Konvertiere zu dB
                    spl_db = self.functions.mag2db(np.abs(spl_value))
                    return float(spl_db)
            
            return None
        except Exception:
            return None

    def _update_mouse_position(self, text):
        """Aktualisiert die Mauspositions-Anzeige"""
        if hasattr(self.main_window.ui, 'mouse_position_label'):
            self.main_window.ui.mouse_position_label.setText(text)

    def _clear_mouse_position(self):
        """Löscht die Mauspositions-Anzeige"""
        if hasattr(self.main_window.ui, 'mouse_position_label'):
            self.main_window.ui.mouse_position_label.setText("")

