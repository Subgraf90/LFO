import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize
from Module_LFO.Modules_Plot.PlotStacks2Beamsteering import StackDraw_Beamsteering


class BeamsteeringPlot(QWidget):
    def __init__(self, parent=None, settings=None, container=None, width=6, height=4, dpi=100):
        super().__init__(parent)
        self.settings = settings
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Sekundäre Y-Achse für Lautsprecher mit gleicher Skalierung wie X-Achse
        self.ax_speakers = None  # Wird beim Zeichnen erstellt
        self.container = container
        self.figure.tight_layout()
        
        # Zoom-Callback-Verwaltung
        self._zoom_callback_connected = False
        self._updating_ticks = False
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()


    def beamsteering_plot(self, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw_idle()
            return
        
        # Extrahieren der Lautsprecherpositionen
        source_position_y = self._to_float_array(speaker_array.source_position_y)
        source_position_x = self._to_float_array(speaker_array.source_position_x)
        virtual_source_position_x = self._to_float_array(speaker_array.virtual_source_position_x)
        virtual_source_position_y = self._to_float_array(speaker_array.virtual_source_position_y)

        # Plot der virtuellen Lautsprecherpositionen
        self.ax.plot(virtual_source_position_x, virtual_source_position_y, 'o-', label='Virtual Source Positions')

        # Zeichnen gestrichelter Linien von realen zu virtuellen Lautsprecherpositionen
        for sx, sy, vx, vy in zip(source_position_x, source_position_y, virtual_source_position_x, virtual_source_position_y):
            self.ax.plot([sx, vx], [sy, vy], linestyle='--', color='grey')

        x_values = self._concatenate_arrays(source_position_x, virtual_source_position_x)
        y_values = self._concatenate_arrays(source_position_y, virtual_source_position_y)

        if x_values.size:
            x_min = float(np.min(x_values))
            x_max = float(np.max(x_values))
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

        if y_values.size:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
            current_ylim = self.ax.get_ylim()
            if abs(current_ylim[0] - y_min) > 0.01 or abs(current_ylim[1] - y_max) > 0.01:
                # Bereits gezoomt - verwende sichtbare Grenzen
                y_min_visible, y_max_visible = current_ylim
                y_ticks, step = self._calculate_metric_ticks(y_min_visible, y_max_visible)
            else:
                # Nicht gezoomt - verwende Daten-Grenzen (exakt, ohne Padding)
                y_ticks, step = self._calculate_metric_ticks(y_min, y_max)
                self.ax.set_ylim(y_min, y_max)
            self.ax.set_yticks(y_ticks)
            # Formatierung auf eine Nachkommastelle
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

        # KEIN set_aspect('equal') - würde die Plot-Höhe stark reduzieren
        # Die Lautsprecher werden durch Skalierung in StackDraw_Beamsteering unverzerrt dargestellt

        # Finale Plot-Einstellungen
        # KEIN axis('equal') - Plot hat feste Breite, beide Achsen sind in Metern
        # Lautsprecher werden durch Transform in StackDraw_Beamsteering unverzerrt dargestellt
        # Plot hat feste Breite und Höhe (800x220 Pixel) durch setFixedWidth/setFixedHeight in UiSourceManagement.py
        
        # WICHTIG: Layout-Anpassungen VOR dem Zeichnen der Stacks, damit Pixel-Größe korrekt ist
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.set_ylabel("Virtual position [m]", fontsize=6)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)
        
        # Layout-Anpassungen für konsistente Größe (muss VOR Stack-Zeichnung sein!)
        self._apply_layout()
        
        # Zeichne einmal, um sicherzustellen, dass get_window_extent() korrekte Werte liefert
        # (nach _apply_layout(), damit subplots_adjust() bereits angewendet wurde)
        self.figure.canvas.draw()
        
        # #region agent log
        import json
        bbox = self.ax.get_window_extent()
        axes_width_pixels = bbox.width
        axes_height_pixels = bbox.height
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'PlotBeamsteering.py:99',
                'message': 'Beamsteering: Achsengrenzen und Pixel-Größe nach Layout-Anpassung, vor Stack-Zeichnung',
                'data': {
                    'xlim': list(self.ax.get_xlim()),
                    'ylim': list(self.ax.get_ylim()),
                    'axes_width_pixels': float(axes_width_pixels),
                    'axes_height_pixels': float(axes_height_pixels)
                },
                'timestamp': int(__import__('time').time() * 1000)
            }) + '\n')
        # #endregion
        
        # Erstelle sekundäre Y-Achse für Lautsprecher mit gleicher Skalierung wie X-Achse
        # Diese Achse wird nur für die Lautsprecher verwendet und hat die gleiche Meter-pro-Pixel-Skalierung wie X
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        ylim = self.ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        
        # Berechne die Skalierung der X-Achse (Meter pro Pixel)
        axes_width_pixels = bbox.width
        axes_height_pixels = bbox.height
        x_units_per_pixel = x_range / axes_width_pixels if axes_width_pixels > 0 and x_range > 0 else 1.0
        y_units_per_pixel = y_range / axes_height_pixels if axes_height_pixels > 0 and y_range > 0 else 1.0
        
        # Erstelle sekundäre Y-Achse mit gleicher Skalierung wie X
        # Die sekundäre Achse hat die gleiche Meter-pro-Pixel-Skalierung wie X
        self.ax_speakers = self.ax.twinx()
        
        # Setze die Y-Limits der sekundären Achse so, dass sie die gleiche Skalierung wie X hat
        # Wenn X: 18.7 m in 613 px = 0.0305 m/px
        # Dann soll Y auch 0.0305 m/px haben
        # Wenn Y-Range z.B. 8.2 m ist, dann brauchen wir: 8.2 / 0.0305 = 269 px
        # Aber die Achse hat axes_height_pixels = 139 px
        # Also müssen wir die Y-Range anpassen: y_range_speakers = axes_height_pixels * x_units_per_pixel
        y_range_speakers = axes_height_pixels * x_units_per_pixel if x_units_per_pixel > 0 else y_range
        
        # Setze Y-Limits der sekundären Achse so, dass sie die gleiche Skalierung wie X hat
        # Wir verwenden die gleiche Y-Position wie die Hauptachse, aber mit angepasster Range
        y_center = (ylim[0] + ylim[1]) / 2.0
        y_min_speakers = y_center - y_range_speakers / 2.0
        y_max_speakers = y_center + y_range_speakers / 2.0
        self.ax_speakers.set_ylim(y_min_speakers, y_max_speakers)
        
        # Mache die sekundäre Achse unsichtbar (keine Ticks/Labels), da sie nur für Lautsprecher-Darstellung verwendet wird
        self.ax_speakers.set_yticks([])
        self.ax_speakers.spines['right'].set_visible(False)
        
        # Zeichne Stacks NACH Layout-Anpassung (für korrekte Pixel-Größe)
        # Übergebe die sekundäre Achse für die Lautsprecher
        self.plot_Stacks2Beamsteering(speaker_array_id)
        
        # Passe Y-Achsen-Grenzen an, damit Lautsprecher nicht abgeschnitten werden
        self._adjust_ylim_for_speakers()
        
        self.canvas.draw_idle()


    def plot_Stacks2Beamsteering(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data.get('cabinet_data', [])
            speaker_names = self.container.data.get('speaker_names', [])

            if speaker_array is None:
                print("Warnung: speaker_array ist nicht initialisiert")
                return

            # Vorhandene Patches entfernen
            if self.ax.patches:
                self.ax.patches.clear()

            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            name_to_index = {name: idx for idx, name in enumerate(speaker_names)}
            for pattern_name in speaker_array.source_polar_pattern:
                if not pattern_name:
                    continue
                try:
                    speaker_index = name_to_index.get(pattern_name)
                    if speaker_index is None or speaker_index >= len(cabinet_data):
                        raise ValueError(f"{pattern_name} nicht gefunden")
                    cabinet = cabinet_data[speaker_index]
                    normalized_cabinets = self._normalize_cabinet_entries(cabinet)
                    if normalized_cabinets:
                        array_cabinets.append(normalized_cabinets)
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Verarbeiten von Pattern {pattern_name}: {str(e)}")
                    continue

            if len(array_cabinets) > 0:
                # Verwende sekundäre Achse für Lautsprecher, falls vorhanden, sonst normale Achse
                ax_for_speakers = self.ax_speakers if self.ax_speakers is not None else self.ax
                
                stack_drawer = StackDraw_Beamsteering(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    ax_for_speakers,  # Verwende sekundäre Achse für Lautsprecher
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_drawer.draw_stack(isrc)
            else:
                print("Keine gültigen Cabinet-Daten gefunden")

            # Kein draw_idle() hier - wird am Ende von beamsteering_plot() aufgerufen

        except Exception as e:
            print(f"Fehler in plot_Stacks2Beamsteering: {str(e)}")
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
    
    def _connect_zoom_callbacks(self):
        """Verbindet Callbacks für Zoom-Events, um Achsenbeschriftung dynamisch anzupassen"""
        if self._zoom_callback_connected:
            return
        
        try:
            # Callback für X-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            # Callback für Y-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            self._zoom_callback_connected = True
        except Exception as e:
            print(f"[BeamsteeringPlot] Fehler beim Verbinden von Zoom-Callbacks: {e}")
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
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            y_ticks, step = self._calculate_metric_ticks(y_min, y_max)
            
            ax.set_yticks(y_ticks)
            # Formatierung auf eine Nachkommastelle
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
            
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
                # Passe die Plot-Ränder an Canvas-Größe an (Beamsteering-Plot)
                # Mehr Platz unten/links, damit Achsenbeschriftungen vollständig sichtbar sind
                self.ax.figure.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.22)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()

    @staticmethod
    def _to_float_array(values):
        if values is None:
            return np.array([], dtype=float)
        array = np.atleast_1d(values)
        result = []
        for item in array:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return np.array(result, dtype=float) if result else np.array([], dtype=float)

    @staticmethod
    def _concatenate_arrays(*arrays):
        valid = [arr for arr in arrays if arr is not None and arr.size]
        if not valid:
            return np.array([], dtype=float)
        return np.concatenate(valid)

    def _adjust_ylim_for_speakers(self):
        """Passt Y-Achsen-Grenzen an, damit Lautsprecher nicht abgeschnitten werden"""
        try:
            import matplotlib.patches as patches
            current_ylim = self.ax.get_ylim()
            
            # Finde maximale Y-Position + Höhe aller gezeichneten Rechtecke (Lautsprecher)
            max_y = current_ylim[1]
            for patch in self.ax.patches:
                if isinstance(patch, patches.Rectangle):
                    patch_y = patch.get_y()
                    patch_height = patch.get_height()
                    max_y = max(max_y, patch_y + patch_height)
            
            # Füge 0.1m Padding oberhalb der Lautsprecher hinzu
            padding = 0.1
            max_y_with_padding = max_y + padding
            
            # Erweitere Y-Achsen-Grenze nur nach oben, wenn nötig
            if max_y_with_padding > current_ylim[1]:
                # Berechne neue Ticks für erweiterte Grenze
                y_min = current_ylim[0]
                y_ticks, step = self._calculate_metric_ticks(y_min, max_y_with_padding)
                self.ax.set_ylim(y_min, max_y_with_padding)
                self.ax.set_yticks(y_ticks)
                # Formatierung auf eine Nachkommastelle
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
        except Exception:
            # Bei Fehler nichts ändern
            pass

    @classmethod
    def _normalize_cabinet_entries(cls, cabinet):
        if cabinet is None:
            return []
        if isinstance(cabinet, dict):
            return [cabinet]
        if isinstance(cabinet, np.ndarray):
            cabinet = cabinet.tolist()
        if isinstance(cabinet, (list, tuple)):
            normalized = []
            for item in cabinet:
                normalized.extend(cls._normalize_cabinet_entries(item))
            return normalized
        return []