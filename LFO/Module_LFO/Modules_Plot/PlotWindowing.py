import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi
from Module_LFO.Modules_Plot.PlotStacks2Windowing import StackDraw_Windowing


class WindowingPlot(QWidget):
    def __init__(self, parent=None, settings=None, container=None, width=6, height=4, dpi=100):
        super().__init__(parent)
        self.settings = settings
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.container = container
        self.figure.tight_layout()
        
        # Zoom-Callback-Verwaltung
        self._zoom_callback_connected = False
        self._updating_ticks = False
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Keine Margins für maximale Ausnutzung
        layout.setSpacing(0)  # Kein Spacing
        layout.addWidget(self.canvas, 1)  # Stretch-Faktor 1 für Expansion
        self.setLayout(layout)

        # Canvas soll auch expandieren
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()
        
        # Verbinde Mouse-Zoom und Pan Events
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        
        # Verbinde Resize-Event für automatische Anpassung
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)


    def windowing_plot(self, windowing, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw()
            return
        
        self.plot_Stacks2Windowing(speaker_array_id)

        if not isinstance(windowing, dict):
            windowing = {}

        self.plot_windowing(windowing, speaker_array)

        window_distance = windowing.get('window_distance')
        if window_distance is None:
            window_distance = np.array([])

        window_restriction_db = windowing.get('window_restriction_db')
        if window_restriction_db is None:
            window_restriction_db = np.array([])

        x_values = []
        x_values.extend(np.atleast_1d(window_distance).tolist())
        if hasattr(speaker_array, 'source_position_x'):
            x_values.extend(np.atleast_1d(speaker_array.source_position_x).tolist())
        if hasattr(speaker_array, 'source_position_y'):
            x_values.extend(np.atleast_1d(speaker_array.source_position_y).tolist())

        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            x_range = max(x_max - x_min, 0)
            x_step = self._select_metric_tick_step(x_range)
            # Mehr Padding links und rechts für mehr Luft
            if x_range == 0:
                x_padding = max(0.5, x_step * 0.5)
                x_min -= x_padding
                x_max += x_padding
            else:
                # Mehr Padding (5% der Range) für mehr Luft links und rechts
                x_padding = x_range * 0.05
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.xaxis.set_major_locator(MultipleLocator(x_step))

        y_values = []
        y_values.extend(np.atleast_1d(window_restriction_db).tolist())
        if hasattr(speaker_array, 'source_lp_zero'):
            y_values.extend(np.atleast_1d(speaker_array.source_lp_zero).tolist())
        window_restriction = getattr(speaker_array, 'window_restriction', None)
        if window_restriction is not None:
            y_values.extend(np.atleast_1d(window_restriction).tolist())

        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            
            # Berücksichtige Lautsprecher-Stacks: Finde minimale Y-Position aller Patches
            min_patch_y = float('inf')
            for patch in self.ax.patches:
                if hasattr(patch, 'get_y'):
                    patch_y = patch.get_y()
                    if hasattr(patch, 'get_height'):
                        patch_height = patch.get_height()
                        # Rechtecke gehen von patch_y nach unten (patch_y - height)
                        min_patch_y = min(min_patch_y, patch_y - patch_height)
                    else:
                        min_patch_y = min(min_patch_y, patch_y)
            
            # Verwende das Minimum aus Daten und Patches
            if min_patch_y != float('inf'):
                y_min = min(y_min, min_patch_y)
            
            y_range = max(y_max - y_min, 0)
            y_step = self._select_db_tick_step(y_range)
            # Padding für Y-Achse: Mehr Platz oben für letzte Daten (saubere Darstellung)
            if y_range == 0:
                y_padding_bottom = max(0.5, y_step * 0.1)
                y_padding_top = max(0.5, y_step * 0.1)
                y_min -= y_padding_bottom
                y_max += y_padding_top
            else:
                # Unten: Minimales Padding (2% der Range)
                y_padding_bottom = max(y_range * 0.02, 0.1)  # Mindestens 0.1 für kleine Ranges
                # Oben: Mehr Padding (8% der Range) für saubere Darstellung der letzten Daten
                y_padding_top = max(y_range * 0.08, 0.2)  # Mindestens 0.2 für kleine Ranges
                y_min -= y_padding_bottom
                y_max += y_padding_top
            self.ax.set_ylim(y_min, y_max)
            self.ax.yaxis.set_major_locator(MultipleLocator(y_step))

        # Setze aspect='equal' für 1:1 Skalierung (1 Meter X = 1 Meter Y in Pixeln)
        # Plot-Dimensionen bleiben starr, nur Limits ändern sich beim Zoom
        self.ax.set_aspect('equal', adjustable='box')
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.3)
        self.ax.set_xlabel("[m]", fontsize=6)
        self.ax.set_ylabel("[dB]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)
        
        # Zeichne gestrichelte horizontale Linie bei window_restriction
        window_restriction = getattr(speaker_array, 'window_restriction', None)
        if window_restriction is not None:
            x_min, x_max = self.ax.get_xlim()
            self.ax.axhline(y=window_restriction, color='gray', linestyle='--', linewidth=0.8)
        
        # Layout-Anpassungen für konsistente Größe und um Achsenbeschriftungen nicht abzuschneiden
        self._apply_layout()
        
        # Aktualisieren Sie die Darstellung
        self.canvas.draw_idle()
        

    def plot_windowing(self, windowing, speaker_array):
        window_distance = windowing.get('window_distance', [0])
        window_restriction_db = windowing.get('window_restriction_db', [-9.8])

        self.ax.plot(window_distance, window_restriction_db)

        # "Setze Gain-Position der Stack Grafiken in der Fensteransicht"
        for iy, il in zip(speaker_array.source_position_x, speaker_array.source_lp_zero):
            self.ax.add_patch(plt.Circle((iy, il), 0.2, color='r'))

        # draw lines into window-function at eff. source positions
        for i in range(len(speaker_array.source_position_x)):
            y_data = speaker_array.source_position_x[i]
            x_data = speaker_array.window_restriction
            gain = speaker_array.source_lp_zero[i]
            self.ax.plot([y_data, y_data], [x_data, gain], linestyle='dotted', color='red')
            
        self.canvas.draw_idle()


    def plot_Stacks2Windowing(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data['cabinet_data']
            speaker_names = self.container.data['speaker_names']
            
            # Löschen Sie alle vorhandenen Patches (Stacks) aus dem Axes-Objekt
            for patch in self.ax.patches:
                patch.remove()
            
            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            for pattern_name in speaker_array.source_polar_pattern:
                try:
                    speaker_index = speaker_names.index(pattern_name)
                    cabinet = cabinet_data[speaker_index]
                    
                    # Akzeptiere np.ndarray, list und dict
                    if isinstance(cabinet, (np.ndarray, list, dict)):
                        # Konvertiere zu np.ndarray für einheitliche Verarbeitung
                        if isinstance(cabinet, dict):
                            cabinet = np.array([cabinet])
                        elif isinstance(cabinet, list):
                            cabinet = np.array(cabinet)
                        array_cabinets.append(cabinet)
                except (ValueError, IndexError) as e:
                    print(f"Pattern {pattern_name} übersprungen: {str(e)}")
                    continue
            
            if array_cabinets:
                stack_drawer = StackDraw_Windowing(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    speaker_array.window_restriction,
                    self.ax,
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_drawer.draw_stack(isrc)
            
            # Aktualisieren Sie die Darstellung
            self.canvas.draw()
            
        except Exception as e:
            print(f"Fehler in plot_Stacks2Windowing: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _select_metric_tick_step(self, value_range):
        # Dynamische Schrittweite für metrische Achsen (1, 2, 5, 10, 20 m)
        thresholds = [
            (80, 20),
            (40, 10),
            (20, 5),
            (10, 2),
        ]
        for threshold, step in thresholds:
            if value_range > threshold:
                return step
        return 1

    def _select_db_tick_step(self, value_range):
        # Dynamische Schrittweite für dB-Achsen (3, 6, 10 dB)
        if value_range > 35:
            return 10
        if value_range > 20:
            return 6
        return 3
    
    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an - passt sich der Fenstergröße an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Optimierte Ränder für saubere Darstellung: Mehr Platz oben und unten
                # left: Platz für Y-Achsenbeschriftung
                # bottom: Platz für X-Achsenbeschriftung
                # top und bottom: Mehr Platz für saubere Darstellung
                self.ax.figure.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.18)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()
    
    def _connect_zoom_callbacks(self):
        """Verbindet Callbacks für Zoom-Events, um Achsenbeschriftung dynamisch anzupassen"""
        if self._zoom_callback_connected:
            return
        
        try:
            # Callback für X-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            # Callback für Y-Achsen-Zoom (dB)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            self._zoom_callback_connected = True
        except Exception as e:
            print(f"[WindowingPlot] Fehler beim Verbinden von Zoom-Callbacks: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_xlim_changed(self, ax):
        """Wird aufgerufen, wenn sich die X-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        # Verhindere Endlosschleife
        if self._updating_ticks:
            return
        
        try:
            self._updating_ticks = True
            x_min, x_max = ax.get_xlim()
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            # Die Schrittweite wird automatisch feiner bei kleinerem Bereich (Zoom-in)
            # und gröber bei größerem Bereich (Zoom-out)
            x_range = x_max - x_min
            x_step = self._select_metric_tick_step(x_range)
            ax.xaxis.set_major_locator(MultipleLocator(x_step))
            
            # Keine Anpassung der Y-Limits - beide Achsen bleiben unabhängig
            # Plot-Dimensionen bleiben starr, nur Limits ändern sich
            
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
        finally:
            self._updating_ticks = False
    
    def _on_ylim_changed(self, ax):
        """Wird aufgerufen, wenn sich die Y-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        # Verhindere Endlosschleife
        if self._updating_ticks:
            return
        
        try:
            self._updating_ticks = True
            y_min, y_max = ax.get_ylim()
            
            # Berechne neue SPL-Ticks basierend auf sichtbarem Bereich
            y_range = y_max - y_min
            y_step = self._select_db_tick_step(y_range)
            ax.yaxis.set_major_locator(MultipleLocator(y_step))
            
            # Keine Anpassung der X-Limits - beide Achsen bleiben unabhängig
            # Plot-Dimensionen bleiben starr, nur Limits ändern sich
            
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
        finally:
            self._updating_ticks = False
    
    def resizeEvent(self, event):
        """Wird aufgerufen, wenn sich die Widget-Größe ändert"""
        super().resizeEvent(event)
        try:
            # Passe Figure-Größe an Widget-Größe an
            width_inch = event.size().width() / self.figure.dpi
            height_inch = event.size().height() / self.figure.dpi
            self.figure.set_size_inches(width_inch, height_inch)
            # Passe Layout an neue Größe an
            self._apply_layout()
            # Stelle sicher, dass aspect='equal' beibehalten wird
            self.ax.set_aspect('equal', adjustable='box')
            self.canvas.draw_idle()
        except Exception:
            pass
    
    def _on_canvas_resize(self, event):
        """Wird aufgerufen, wenn sich die Canvas-Größe ändert"""
        try:
            # Passe Layout an neue Größe an
            self._apply_layout()
            # Stelle sicher, dass aspect='equal' beibehalten wird
            self.ax.set_aspect('equal', adjustable='box')
        except Exception:
            pass
    
    def _on_scroll(self, event):
        """Handler für Mausrad-Zoom Events"""
        if event.inaxes != self.ax:
            return
        
        scale_factor = 1.1 if event.button == 'up' else 1/1.1
        
        # Hole aktuelle Grenzen
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # Mausposition als Zentrum des Zooms
        x_center = event.xdata
        y_center = event.ydata
        
        if x_center is None or y_center is None:
            return
        
        # Berechne neue Grenzen
        x_new_min = x_center - (x_center - x_min) * scale_factor
        x_new_max = x_center + (x_max - x_center) * scale_factor
        y_new_min = y_center - (y_center - y_min) * scale_factor
        y_new_max = y_center + (y_max - y_center) * scale_factor
        
        # Setze neue Grenzen
        self.ax.set_xlim(x_new_min, x_new_max)
        self.ax.set_ylim(y_new_min, y_new_max)
        
        # Manuell Callbacks aufrufen für adaptive Ticks (werden automatisch durch set_xlim/set_ylim getriggert, 
        # aber sicherstellen dass sie aufgerufen werden)
        if not self._updating_ticks:
            self._on_xlim_changed(self.ax)
            self._on_ylim_changed(self.ax)
        
        self.canvas.draw_idle()
    
    def _on_button_press(self, event):
        """Handler für Maus-Pan Events (Linke Maustaste)"""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        self.ax._pan_start = (event.xdata, event.ydata)
        self._pan_cid = self.canvas.mpl_connect('motion_notify_event', self._on_pan_move)
        self._pan_release_cid = self.canvas.mpl_connect('button_release_event', self._on_pan_release)
    
    def _on_pan_move(self, event):
        """Handler für Mausbewegung während des Pannens"""
        if event.inaxes != self.ax or not hasattr(self.ax, '_pan_start'):
            return
        
        if self.ax._pan_start[0] is None or self.ax._pan_start[1] is None:
            return
        
        dx = event.xdata - self.ax._pan_start[0]
        dy = event.ydata - self.ax._pan_start[1]
        
        if event.xdata is None or event.ydata is None:
            return
        
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        x_new_min = x_min - dx
        x_new_max = x_max - dx
        y_new_min = y_min - dy
        y_new_max = y_max - dy
        
        self.ax.set_xlim(x_new_min, x_new_max)
        self.ax.set_ylim(y_new_min, y_new_max)
        
        # Aktualisiere Start-Position für nächste Bewegung
        self.ax._pan_start = (event.xdata, event.ydata)
        
        # Manuell Callbacks aufrufen für adaptive Ticks (werden automatisch durch set_xlim/set_ylim getriggert,
        # aber sicherstellen dass sie aufgerufen werden)
        if not self._updating_ticks:
            self._on_xlim_changed(self.ax)
            self._on_ylim_changed(self.ax)
        
        self.canvas.draw_idle()
    
    def _on_pan_release(self, event):
        """Handler für Loslassen der Maustaste nach dem Pannens"""
        if hasattr(self, '_pan_cid'):
            self.canvas.mpl_disconnect(self._pan_cid)
        if hasattr(self, '_pan_release_cid'):
            self.canvas.mpl_disconnect(self._pan_release_cid)
        if hasattr(self.ax, '_pan_start'):
            del self.ax._pan_start