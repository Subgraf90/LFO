from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import logging
import os
from time import perf_counter
from typing import Optional

from Module_LFO.Modules_Plot.PlotMPLCanvas import MplCanvas
from Module_LFO.Modules_Init.ModuleBase import ModuleBase

try:
    from Module_LFO.Modules_Plot.PlotSPL3D import DrawSPLPlot3D
except ImportError:  # PyVista optional
    DrawSPLPlot3D = None
from Module_LFO.Modules_Plot.PlotSPLXaxis import DrawSPLPlot_Xaxis
from Module_LFO.Modules_Plot.PlotSPLYaxis import DrawSPLPlot_Yaxis
from Module_LFO.Modules_Plot.PlotPolarPattern import DrawPolarPattern


class _PerfScope:
    def __init__(self, owner, label: str):
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


class DrawPlotsMainwindow(ModuleBase):
    def __init__(self, main_window, settings, container):
        super().__init__(settings)  
        self.main_window = main_window
        self.settings = settings
        self.container = container
        
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
        self._perf_enabled = bool(int(os.environ.get("LFO_DEBUG_PERF", "1")))
        self._perf_logger = logging.getLogger("LFO.Performance")

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
        self._initialize_spl_plotter()
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

        self.container.calculation_spl['sound_field_x'] = sound_field_x
        self.container.calculation_spl['sound_field_y'] = sound_field_y
        self.container.calculation_spl['sound_field_p'] = sound_field_p.tolist()

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
        
        # Prüfe ob gültige Daten vorhanden sind (nicht nur NaN)
        has_data = (
            isinstance(self.container.calculation_spl, dict) and
            len(self.container.calculation_spl) > 0 and
            'sound_field_x' in self.container.calculation_spl and
            'sound_field_y' in self.container.calculation_spl and
            'sound_field_p' in self.container.calculation_spl and
            self.container.calculation_spl.get('sound_field_x') is not None and
            self.container.calculation_spl.get('sound_field_y') is not None and
            self.container.calculation_spl.get('sound_field_p') is not None and
            len(self.container.calculation_spl.get('sound_field_x', [])) > 0 and
            len(self.container.calculation_spl.get('sound_field_y', [])) > 0 and
            len(self.container.calculation_spl.get('sound_field_p', [])) > 0 and
            not self._is_empty_data(self.container.calculation_spl['sound_field_p'])
        )
        
        draw_spl_plotter = self._get_current_spl_plotter()

        if not has_data:
            # Keine Daten - zeige leere Szene
            draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
            draw_spl_plotter.update_overlays(self.settings, self.container)

            self.colorbar_canvas.draw()

            if update_axes:
                self.plot_xaxis()
                self.plot_yaxis()
                self.plot_polar_pattern()
            return
        
        # Daten aus dem Container abrufen
        sound_field_x = self.container.calculation_spl['sound_field_x']
        sound_field_y = self.container.calculation_spl['sound_field_y']
        sound_field_pressure = self.container.calculation_spl['sound_field_p']

        # 3D-Plot aktualisieren
        with self._perf_scope("Plot3D update_spl_plot"):
            draw_spl_plotter.update_spl_plot(
                sound_field_x,
                sound_field_y,
                sound_field_pressure,
                self.settings.colorization_mode,
            )
        with self._perf_scope("Plot3D overlays"):
            draw_spl_plotter.update_overlays(self.settings, self.container)
        with self._perf_scope("Plot3D colorbar draw"):
            self.colorbar_canvas.draw()
        if reset_camera:
            self.reset_zoom()

        if update_axes:
            with self._perf_scope("Plot axes update"):
                self.plot_xaxis()
                self.plot_yaxis()
                self.plot_polar_pattern()
        self._log_async("Plot SPL")
   
    def _is_empty_data(self, sound_field_p):
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
            
            # Prüfe ob alle Werte NaN oder 0 sind
            return np.all(np.isnan(arr)) or np.all(arr == 0)
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
        with self._perf_scope("Plot x-axis data"):
            self.draw_spl_plot_xaxis.plot_xaxis(self.container.calculation_axes)

        # Aktualisieren des Plots
        with self._perf_scope("Plot x-axis draw"):
            self.matplotlib_canvas_xaxis.draw()

    def plot_yaxis(self):
        # Aufrufen der plot_yaxis Methode von DrawSPLPlot_Yaxis
        with self._perf_scope("Plot y-axis data"):
            self.draw_spl_plot_yaxis.plot_yaxis(self.container.calculation_axes)

        # Aktualisieren des Plots
        with self._perf_scope("Plot y-axis draw"):
            self.matplotlib_canvas_yaxis.draw()

    def plot_polar_pattern(self):
        polar_data = self.container.get_polar_data()

        # Update Polar-Plot (zeigt Empty Plot wenn keine Daten vorhanden)
        with self._perf_scope("Plot polar data"):
            self.draw_polar_pattern.update_polar_pattern(polar_data)

        # Aktualisieren des Plots
        with self._perf_scope("Plot polar draw"):
            self.matplotlib_canvas_polar_pattern.draw()
        self._log_async("Plot polar")

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
            with self._perf_scope("Plot3D overlays only"):
                plotter.update_overlays(self.settings, self.container)

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

    def set_splitter_positions(self):
        # Hole die Referenzen zu den Splittern
        self.horizontal_splitter_top = self.main_window.ui.horizontal_splitter_top
        self.vertical_splitter = self.main_window.ui.vertical_splitter
        self.horizontal_splitter_bottom = self.main_window.ui.horizontal_splitter_bottom
        
        # Verbinde Debug-Handler für manuelle Splitter-Änderungen
        # self.horizontal_splitter_top.splitterMoved.connect(self._debug_splitter_top)
        # self.vertical_splitter.splitterMoved.connect(self._debug_splitter_vertical)
        # self.horizontal_splitter_bottom.splitterMoved.connect(self._debug_splitter_bottom)

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

