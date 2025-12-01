# "Grafische Ausgabe fuer SPL Plot X-Axis"

from matplotlib.ticker import MultipleLocator
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi

"x-axis SPL Plot ist der untere Plot. Er zeigt die SPL-Werte in Abhängigkeit der Breite des Schallfeldes"

class DrawSPLPlot_Xaxis(ModuleBase):
    def __init__(self, matplotlib_canvas_xaxis, settings, main_window=None):
        super().__init__(settings)  # Korrekte Initialisierung der Basisklasse "Modulebase"
        self.ax = matplotlib_canvas_xaxis
        self.im = None
        self.settings = settings
        self.main_window = main_window  # Referenz zum Main Window
        self._y_locator_step = None
        self._zoom_callback_connected = False
        self._updating_ticks = False  # Flagge um Endlosschleifen zu vermeiden
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()
    
    def initialize_empty_plots(self):
        """Initialisiert den Plot mit sinnvoller Leer-Darstellung"""
        self.ax.clear()
        self.ax.set_ylabel("SPL [dB]", fontsize=6)  # Konsistent mit Polar
        
        # Verwende Settings für realistische Grenzen
        # X-Achse = SPL-Werte (y_data im normalen Plot)
        # Y-Achse = Position entlang Länge (x_data_xaxis im normalen Plot = length)
        y_min = -24  # SPL-Minimum
        y_max = 0    # SPL-Maximum
        x_min = -(self.settings.width / 2)
        x_max = (self.settings.width / 2)
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Intelligente X-Ticks (Position) - adaptive Schrittweite
        x_ticks, _ = self._calculate_metric_ticks(x_min, x_max)
        self.ax.set_xticks(x_ticks)
        
        # Adaptive Y-Ticks (SPL dB) - basierend auf Bereich
        self._update_spl_ticks(y_min, y_max)
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.tick_params(axis='both', labelsize=6)  # Konsistent mit Polar
        # self.ax.text(0.5, 0.5, 'No active speakerz', 
        #             transform=self.ax.transAxes,
        #             ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Layout-Anpassungen für konsistente Größe (identisch mit plot_xaxis)
        self._apply_layout()


    def plot_xaxis(self, calculation_Xaxis):
        import sys
        # Stelle sicher, dass Callbacks verbunden sind
        if not self._zoom_callback_connected:
            self._connect_zoom_callbacks()
        
        # Prüfe ob Daten zum Plotten vorhanden sind (inkl. Snapshots)
        has_data_to_plot = False
        for key, value in calculation_Xaxis.items():
            if 'show_in_plot' in value and value['show_in_plot']:
                # Prüfe ob tatsächlich Daten vorhanden sind (nicht nur leeres Dict)
                if 'x_data_xaxis' in value and 'y_data_xaxis' in value:
                    has_data_to_plot = True
                    break
        
        # Wenn keine Daten vorhanden sind, zeige leere Darstellung
        if not has_data_to_plot:
            self.initialize_empty_plots()
            return
        
        self.ax.clear()
        
        # Sammle alle geplotteten Daten für Achsen-Setup
        all_y_data = []
        all_x_data = []
        
        for key, value in calculation_Xaxis.items():
            # boleanabfrage. wenn show_in_plot true, wird der plot ausgeführt
            if 'show_in_plot' in value and value['show_in_plot']: 
                if 'x_data_xaxis' not in value or 'y_data_xaxis' not in value:
                    continue
                x_data = value['x_data_xaxis']
                y_data = value['y_data_xaxis']
                
                # Prüfe ob Arrays nicht leer sind
                if len(x_data) == 0 or len(y_data) == 0:
                    continue

                # Abrufen der Farbe aus dem Wert des Schlüssels "color"
                color = value.get('color', 'b')  # 'b' ist ein Standardwert, falls keine Farbe gefunden wird
                
                # Liniendicke: 1.5 für ausgewähltes Item, 0.5 für andere
                linewidth = 0.5  # Standard für nicht-ausgewählte Items
                # Hole snapshot_engine dynamisch vom main_window
                if self.main_window and hasattr(self.main_window, 'snapshot_engine'):
                    snapshot_engine = self.main_window.snapshot_engine
                    if hasattr(snapshot_engine, 'selected_snapshot_key'):
                        if key == snapshot_engine.selected_snapshot_key:
                            linewidth = 1.5  # Ausgewähltes Item
                
                self.ax.plot(y_data, x_data, label=key, color=color, linewidth=linewidth)
                
                # Sammle Daten für Achsen-Setup
                all_y_data.extend(y_data)
                all_x_data.extend(x_data)

        # Prüfe ob tatsächlich Daten geplottet wurden
        if len(all_y_data) == 0 or len(all_x_data) == 0:
            self.initialize_empty_plots()
            return

        # Sammle Segment-Grenzen für "aktuelle_simulation" (werden nach Achsen-Setup gezeichnet)
        segment_boundaries_to_draw = []
        if "aktuelle_simulation" in calculation_Xaxis:
            sim_value = calculation_Xaxis["aktuelle_simulation"]
            if 'segment_boundaries_xaxis' in sim_value:
                segment_boundaries_to_draw = sim_value.get('segment_boundaries_xaxis', [])

        # Intelligente X-Achse Ticks (Position) - adaptive Schrittweite
        x_min_data = min(all_y_data)
        x_max_data = max(all_y_data)
        
        # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
        current_xlim = self.ax.get_xlim()
        
        if abs(current_xlim[0] - x_min_data) > 0.01 or abs(current_xlim[1] - x_max_data) > 0.01:
            # Bereits gezoomt - verwende sichtbare Grenzen
            x_min_visible, x_max_visible = current_xlim
            x_ticks, step = self._calculate_metric_ticks(x_min_visible, x_max_visible)
        else:
            # Nicht gezoomt - verwende Daten-Grenzen
            x_ticks, step = self._calculate_metric_ticks(x_min_data, x_max_data)
            self.ax.set_xlim(x_min_data, x_max_data)  # Exakt gem. width, ohne Zusatzwert
        
        self.ax.tick_params(axis='x', labelsize=6)  # Konsistent mit Polar
        self.ax.set_xticks(x_ticks)
        # self.ax.set_xlabel("Distance to array [m]")    
        
        # Adaptive Y-Achse (SPL dB) - basierend auf sichtbarem Bereich
        current_ylim = self.ax.get_ylim()
        if current_ylim[0] != min(all_x_data) or current_ylim[1] != max(all_x_data):
            # Bereits gezoomt - verwende sichtbare Grenzen
            y_min_visible, y_max_visible = current_ylim
            self._update_spl_ticks(y_min_visible, y_max_visible)
        else:
            # Nicht gezoomt - verwende Daten-Grenzen
            y_min_spl = min(all_x_data)
            y_max_spl = max(all_x_data)
            self._update_spl_ticks(y_min_spl, y_max_spl)
            
        self.ax.tick_params(axis='y', labelsize=6)  # Konsistent mit Polar
        self.ax.set_ylabel("SPL [dB]", fontsize=6)  # Konsistent mit Polar
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        
        # Zeichne vertikale gestrichelte Linien an Segment-Grenzen (NACH Achsen-Setup)
        if segment_boundaries_to_draw:
            y_lim = self.ax.get_ylim()
            for x_pos in segment_boundaries_to_draw:
                # Vertikale gestrichelte Linie ohne Label (wird nicht in Snapshots gespeichert)
                # Verwende zorder=1 damit Linien über dem Grid, aber unter den Daten sind
                self.ax.axvline(x=x_pos, color='darkgray', linestyle='--', linewidth=0.8, alpha=0.8, zorder=1)
        
        # Layout-Anpassungen für konsistente Größe (identisch mit initialize_empty_plots)
        self._apply_layout()


    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Passe die Plot-Ränder an Canvas-Größe an (X-Achsen-Plot)
                # Weniger hoch: top kleiner, bottom größer
                self.ax.figure.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.30)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()

    def _set_y_locator(self, step):
        if step != self._y_locator_step:
            self.ax.yaxis.set_major_locator(MultipleLocator(step))
            self._y_locator_step = step

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
            old_step = step
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
            # Callback für Y-Achsen-Zoom (SPL dB)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            self._zoom_callback_connected = True
            print(f"[PlotSPLXaxis] Zoom-Callbacks erfolgreich verbunden")
        except Exception as e:
            print(f"[PlotSPLXaxis] Fehler beim Verbinden von Zoom-Callbacks: {e}")
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