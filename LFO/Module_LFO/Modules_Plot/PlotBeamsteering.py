import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize, Qt
from Module_LFO.Modules_Plot.PlotStacks2Beamsteering import StackDraw_Beamsteering


class BeamsteeringPlot(QWidget):
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
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()
        
        # Verbinde Resize-Event für automatische Anpassung
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)


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
                # Nicht gezoomt - verwende Daten-Grenzen mit Padding
                y_range_data = y_max - y_min
                padding = max(y_range_data * 0.1, 0.1)  # 10% Padding oder mindestens 0.1m
                y_min_with_padding = y_min - padding
                y_max_with_padding = y_max + padding
                y_ticks, step = self._calculate_metric_ticks(y_min_with_padding, y_max_with_padding)
                self.ax.set_ylim(y_min_with_padding, y_max_with_padding)
            self.ax.set_yticks(y_ticks)
            # Formatierung auf eine Nachkommastelle
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

        # Setze aspect='equal' für gleichmäßige Skalierung (1:1 Verhältnis)
        self.ax.set_aspect('equal', adjustable='box')

        # Finale Plot-Einstellungen
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.set_ylabel("Virtual position [m]", fontsize=6)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)
        
        # Layout-Anpassungen für konsistente Größe (muss VOR Stack-Zeichnung sein!)
        self._apply_layout()
        
        # Zeichne einmal, um sicherzustellen, dass get_window_extent() korrekte Werte liefert
        # (nach _apply_layout(), damit subplots_adjust() bereits angewendet wurde)
        self.figure.canvas.draw()
        
        # Zeichne Stacks NACH Layout-Anpassung (für korrekte Pixel-Größe)
        self.plot_Stacks2Beamsteering(speaker_array_id)
        
        # Passe Y-Achsen-Grenzen an, damit Lautsprecher nicht abgeschnitten werden
        self._adjust_ylim_for_speakers()
        
        # Stelle sicher, dass aspect='equal' nach _adjust_ylim_for_speakers() wieder gesetzt wird
        self.ax.set_aspect('equal', adjustable='box')
        
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
            # WICHTIG: Entferne Patches VOR dem Zeichnen, um doppelte Darstellung zu vermeiden
            if self.ax.patches:
                for patch in list(self.ax.patches):  # Liste kopieren, da wir während Iteration entfernen
                    try:
                        patch.remove()
                    except:
                        pass
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
                # Verwende Hauptachse für Lautsprecher-Darstellung
                stack_drawer = StackDraw_Beamsteering(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    self.ax,  # Verwende Hauptachse für Lautsprecher
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
            
            # Für aspect='equal': Passe Y-Limits an X-Limits an
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Verwende die größere Range für beide Achsen
            max_range = max(x_range, y_range)
            
            # Zentriere beide Achsen um ihre Mitte
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Setze Limits für equal aspect
            x_min_equal = x_center - max_range / 2
            x_max_equal = x_center + max_range / 2
            y_min_equal = y_center - max_range / 2
            y_max_equal = y_center + max_range / 2
            
            ax.set_xlim(x_min_equal, x_max_equal)
            ax.set_ylim(y_min_equal, y_max_equal)
            ax.set_aspect('equal', adjustable='box')
            
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
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Für aspect='equal': Passe X-Limits an Y-Limits an
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Verwende die größere Range für beide Achsen
            max_range = max(x_range, y_range)
            
            # Zentriere beide Achsen um ihre Mitte
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Setze Limits für equal aspect
            x_min_equal = x_center - max_range / 2
            x_max_equal = x_center + max_range / 2
            y_min_equal = y_center - max_range / 2
            y_max_equal = y_center + max_range / 2
            
            ax.set_xlim(x_min_equal, x_max_equal)
            ax.set_ylim(y_min_equal, y_max_equal)
            ax.set_aspect('equal', adjustable='box')
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            y_ticks, step = self._calculate_metric_ticks(y_min_equal, y_max_equal)
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
    
    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an - passt sich der Fenstergröße an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Passe die Plot-Ränder an Canvas-Größe an (wie bei X/Y-Axis Plots)
                # Mehr Platz für Achsenbeschriftungen: left größer, bottom größer
                self.ax.figure.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
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
            
            # Finde minimale und maximale Y-Positionen aller gezeichneten Rechtecke (Lautsprecher)
            max_y_position = -float('inf')
            min_y_position = float('inf')
            
            for patch in self.ax.patches:
                if isinstance(patch, patches.Rectangle):
                    patch_height = patch.get_height()
                    patch_y = patch.get_y()
                    # Rechtecke gehen von patch_y nach oben (patch_y + height)
                    max_y_position = max(max_y_position, patch_y + patch_height)
                    min_y_position = min(min_y_position, patch_y)
            
            if max_y_position != -float('inf') and min_y_position != float('inf'):
                # Füge minimales Padding hinzu
                padding = 0.1
                
                # Erweitere Limits falls nötig
                current_ylim = self.ax.get_ylim()
                y_min = current_ylim[0]
                y_max = current_ylim[1]
                
                if max_y_position + padding > y_max:
                    y_max = max_y_position + padding
                if min_y_position - padding < y_min:
                    y_min = min_y_position - padding
                
                # Für aspect='equal': Passe beide Achsen an
                current_xlim = self.ax.get_xlim()
                x_range = current_xlim[1] - current_xlim[0]
                y_range = y_max - y_min
                
                # Verwende die größere Range für beide Achsen
                max_range = max(x_range, y_range)
                
                # Zentriere beide Achsen um ihre Mitte
                x_center = (current_xlim[0] + current_xlim[1]) / 2
                y_center = (y_min + y_max) / 2
                
                # Setze Limits für equal aspect
                x_min_equal = x_center - max_range / 2
                x_max_equal = x_center + max_range / 2
                y_min_equal = y_center - max_range / 2
                y_max_equal = y_center + max_range / 2
                
                # Stelle sicher, dass alle Patches enthalten sind
                if max_y_position + padding > y_max_equal:
                    y_max_equal = max_y_position + padding
                    # Passe X-Limits entsprechend an
                    y_range_new = y_max_equal - y_min_equal
                    x_center = (current_xlim[0] + current_xlim[1]) / 2
                    x_min_equal = x_center - y_range_new / 2
                    x_max_equal = x_center + y_range_new / 2
                
                if min_y_position - padding < y_min_equal:
                    y_min_equal = min_y_position - padding
                    # Passe X-Limits entsprechend an
                    y_range_new = y_max_equal - y_min_equal
                    x_center = (current_xlim[0] + current_xlim[1]) / 2
                    x_min_equal = x_center - y_range_new / 2
                    x_max_equal = x_center + y_range_new / 2
                
                self.ax.set_xlim(x_min_equal, x_max_equal)
                self.ax.set_ylim(y_min_equal, y_max_equal)
                
                # Aktualisiere Ticks
                y_ticks, step = self._calculate_metric_ticks(y_min_equal, y_max_equal)
                self.ax.set_yticks(y_ticks)
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