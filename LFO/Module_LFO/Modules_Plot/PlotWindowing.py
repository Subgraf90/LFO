import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
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
        self._y_locator_step = None
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()


    def windowing_plot(self, windowing, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw()
            return

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
            # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
            current_xlim = self.ax.get_xlim()
            if abs(current_xlim[0] - x_min) > 0.01 or abs(current_xlim[1] - x_max) > 0.01:
                # Bereits gezoomt - verwende sichtbare Grenzen
                x_min_visible, x_max_visible = current_xlim
                x_ticks, step = self._calculate_metric_ticks(x_min_visible, x_max_visible)
            else:
                # Nicht gezoomt - verwende Daten-Grenzen (exakt, ohne Padding)
                x_ticks, step = self._calculate_metric_ticks(x_min, x_max)
                self.ax.set_xlim(x_min, x_max)
            self.ax.set_xticks(x_ticks)

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
            # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
            current_ylim = self.ax.get_ylim()
            if abs(current_ylim[0] - y_min) > 0.01 or abs(current_ylim[1] - y_max) > 0.01:
                # Bereits gezoomt - verwende sichtbare Grenzen
                y_min_visible, y_max_visible = current_ylim
                self._update_spl_ticks(y_min_visible, y_max_visible)
            else:
                # Nicht gezoomt - verwende Daten-Grenzen (exakt, ohne Padding)
                self.ax.set_ylim(y_min, y_max)
                self._update_spl_ticks(y_min, y_max)

        # KEIN axis('equal') - Achsen können unterschiedlich skaliert sein
        # Lautsprecher werden durch Transform in StackDraw_Windowing unverzerrt dargestellt
        # Plot hat feste Breite und Höhe (800x220 Pixel) durch setFixedWidth/setFixedHeight in UiSourceManagement.py
        
        # Zeichne einmal, um sicherzustellen, dass get_window_extent() korrekte Werte liefert
        self.figure.canvas.draw()
        
        # Zeichne Stacks NACH dem Setzen der Achsengrenzen (für korrekten Transform)
        self.plot_Stacks2Windowing(speaker_array_id)
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.set_ylabel("Windowfunction [dB]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)

        # Layout-Anpassungen für konsistente Größe
        self._apply_layout()
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

    def _calculate_metric_ticks(self, min_val, max_val):
        """
        Berechnet intelligente Ticks für metrische Achsen basierend auf dem Bereich.
        Ziel: 4-8 Ticks für optimale Lesbarkeit mit passender Schrittweite.
        
        Args:
            min_val: Minimum-Wert der Achse
            max_val: Maximum-Wert der Achse
            
        Returns:
            tuple: (ticks_list, step) - Liste der Tick-Positionen und verwendete Schrittweite
        """
        range_val = max_val - min_val
        
        if range_val <= 0:
            # Fallback für ungültige Bereiche
            return [min_val, max_val], 1.0
        
        # Ziel: 4-8 Ticks für optimale Lesbarkeit
        # Berechne optimale Schrittweite basierend auf dem Bereich
        ideal_num_ticks = 6  # Ziel: ~6 Ticks
        rough_step = range_val / ideal_num_ticks
        
        # Wähle passende Schrittweite aus einer Liste von "schönen" Werten
        # Sortiert von klein nach groß
        nice_steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        
        # Finde die passende Schrittweite (nächstgrößere oder gleichgroße)
        step = nice_steps[0]
        for nice_step in nice_steps:
            if nice_step >= rough_step:
                step = nice_step
                break
        else:
            # Falls alle zu klein sind, verwende die größte
            step = nice_steps[-1]
        
        # Für sehr kleine Bereiche: verwende noch feinere Auflösung
        if range_val < 1.0:
            fine_steps = [0.05, 0.1, 0.2, 0.5]
            for fine_step in fine_steps:
                if fine_step >= rough_step:
                    step = fine_step
                    break
        
        # Generiere Ticks
        # Runde min_val nach unten auf das nächste Vielfache von step
        import math
        start_tick = math.floor(min_val / step) * step
        # Stelle sicher, dass start_tick <= min_val
        if start_tick > min_val:
            start_tick -= step
        
        # Runde max_val nach oben auf das nächste Vielfache von step
        end_tick = math.ceil(max_val / step) * step
        # Stelle sicher, dass end_tick >= max_val
        if end_tick < max_val:
            end_tick += step
        
        # Erstelle Tick-Liste
        ticks = []
        current = start_tick
        tolerance = step * 0.001  # Kleine Toleranz für Rundungsfehler
        while current <= end_tick + tolerance:
            if min_val - tolerance <= current <= max_val + tolerance:
                ticks.append(current)
            current += step
        
        # Stelle sicher, dass min und max enthalten sind (falls sie nicht schon als Ticks vorhanden sind)
        if len(ticks) == 0:
            ticks = [min_val, max_val]
        else:
            if abs(ticks[0] - min_val) > tolerance:
                ticks.insert(0, min_val)
            if abs(ticks[-1] - max_val) > tolerance:
                ticks.append(max_val)
        
        # Entferne Duplikate und sortiere
        ticks = sorted(list(set(ticks)))
        
        return ticks, step
    
    def _set_y_locator(self, step):
        """Setzt den Y-Achsen-Locator für dB-Ticks"""
        if step != self._y_locator_step:
            self.ax.yaxis.set_major_locator(MultipleLocator(step))
            self._y_locator_step = step
    
    def _update_spl_ticks(self, y_min, y_max):
        """Aktualisiert SPL-Ticks basierend auf sichtbarem Bereich"""
        y_range = y_max - y_min
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if y_range > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif y_range > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self._set_y_locator(step)
    
    def _connect_zoom_callbacks(self):
        """Verbindet Callbacks für Zoom-Events, um Achsenbeschriftung dynamisch anzupassen"""
        if self._zoom_callback_connected:
            return
        
        try:
            # Callback für X-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            # Callback für Y-Achsen-Zoom (SPL dB)
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
            x_ticks, step = self._calculate_metric_ticks(x_min, x_max)
            
            ax.set_xticks(x_ticks)
            
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
            self._update_spl_ticks(y_min, y_max)
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
        finally:
            self._updating_ticks = False
    
    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Passe die Plot-Ränder an Canvas-Größe an (Windowing-Plot)
                self.ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()