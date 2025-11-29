# "Grafische Ausgabe fuer SPL Plot Y-Axis"

from matplotlib.ticker import MultipleLocator
from Module_LFO.Modules_Init.ModuleBase import ModuleBase

"y-axis SPL Plot ist der rechte Plot. Er zeigt die SPL-Werte in Abhängigkeit der Länge des Schallfeldes"

class DrawSPLPlot_Yaxis(ModuleBase):
    def __init__(self, matplotlib_canvas_yaxis, settings, main_window=None):
        super().__init__(settings)
        self.ax = matplotlib_canvas_yaxis
        self.im = None
        self.settings = settings
        self.main_window = main_window  # Referenz zum Main Window
        self._x_locator_step = None
        self._zoom_callback_connected = False
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()
    
    def initialize_empty_plots(self):
        """Initialisiert den Plot mit sinnvoller Leer-Darstellung"""
        self.ax.clear()
        self.ax.set_xlabel("SPL [dB]", fontsize=6)  # Konsistent mit Polar
        
        # Verwende Settings für realistische Grenzen
        # X-Achse = SPL-Werte (x_data im normalen Plot)
        # Y-Achse = Position entlang Breite = width (y_data_yaxis im normalen Plot)
        x_min = -24  # SPL-Minimum
        x_max = 0    # SPL-Maximum
        y_min = -(self.settings.length / 2)
        y_max = (self.settings.length / 2)
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Intelligente Y-Ticks (Position) - adaptive Schrittweite
        y_ticks, _ = self._calculate_metric_ticks(y_min, y_max)
        self.ax.set_yticks(y_ticks)
        
        # Adaptive X-Ticks (SPL dB) - basierend auf Bereich
        self._update_spl_ticks(x_min, x_max)
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.tick_params(axis='both', labelsize=6)  # Konsistent mit Polar
        self.ax.yaxis.tick_right()
        self.ax.invert_xaxis()
        
        # Layout-Anpassungen für konsistente Größe (identisch mit plot_yaxis)
        self._apply_layout()


    def plot_yaxis(self, calculation_Yaxis):
        # Stelle sicher, dass Callbacks verbunden sind
        if not self._zoom_callback_connected:
            self._connect_zoom_callbacks()
        
        # Prüfe ob Daten zum Plotten vorhanden sind (inkl. Snapshots)
        has_data_to_plot = False
        for key, value in calculation_Yaxis.items():
            if 'show_in_plot' in value and value['show_in_plot']:
                # Prüfe ob tatsächlich Daten vorhanden sind (nicht nur leeres Dict)
                if 'x_data_yaxis' in value and 'y_data_yaxis' in value:
                    has_data_to_plot = True
                    break
        
        # Wenn keine Daten vorhanden sind, zeige leere Darstellung
        if not has_data_to_plot:
            self.initialize_empty_plots()
            return
        self.ax.clear()        

        for key, value in calculation_Yaxis.items():
            # boleanabfrage. wenn show_in_plot true, wird der plot ausgeführt
            if 'show_in_plot' in value and value['show_in_plot']: 
                if 'x_data_yaxis' not in value or 'y_data_yaxis' not in value:
                    continue
                x_data = value['x_data_yaxis']
                y_data = value['y_data_yaxis']

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
                
                self.ax.plot(x_data, y_data, label=key, color=color, linewidth=linewidth)

        # Invertiere die x-Achse
        self.ax.invert_xaxis()

        # Intelligente Y-Achse Ticks (Position) - adaptive Schrittweite
        y_min_data = min(y_data)
        y_max_data = max(y_data)
        print(f"[DEBUG PlotSPLYaxis] plot_yaxis: Daten-Bereich Y: {y_min_data:.3f} bis {y_max_data:.3f}")
        
        # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
        current_ylim = self.ax.get_ylim()
        print(f"[DEBUG PlotSPLYaxis] plot_yaxis: Aktuelle Y-Limits: {current_ylim[0]:.3f} bis {current_ylim[1]:.3f}")
        
        if abs(current_ylim[0] - y_min_data) > 0.01 or abs(current_ylim[1] - y_max_data) > 0.01:
            # Bereits gezoomt - verwende sichtbare Grenzen
            y_min_visible, y_max_visible = current_ylim
            print(f"[DEBUG PlotSPLYaxis] Bereits gezoomt! Verwende sichtbare Grenzen: {y_min_visible:.3f} bis {y_max_visible:.3f}")
            y_ticks, step = self._calculate_metric_ticks(y_min_visible, y_max_visible)
        else:
            # Nicht gezoomt - verwende Daten-Grenzen
            print(f"[DEBUG PlotSPLYaxis] Nicht gezoomt, verwende Daten-Grenzen")
            y_ticks, step = self._calculate_metric_ticks(y_min_data, y_max_data)
            self.ax.set_ylim(y_min_data, y_max_data)  # Exakt gem. length, ohne Zusatzwert
        
        self.ax.tick_params(axis='y', labelsize=6)  # Konsistent mit Polar
        self.ax.set_yticks(y_ticks)
        
        # Adaptive X-Achse (SPL dB) - basierend auf sichtbarem Bereich
        current_xlim = self.ax.get_xlim()
        if current_xlim[0] != min(x_data) or current_xlim[1] != max(x_data):
            # Bereits gezoomt - verwende sichtbare Grenzen
            x_min_visible, x_max_visible = current_xlim
            self._update_spl_ticks(x_min_visible, x_max_visible)
        else:
            # Nicht gezoomt - verwende Daten-Grenzen
            x_min_spl = min(x_data)
            x_max_spl = max(x_data)
            self._update_spl_ticks(x_min_spl, x_max_spl)
        
        self.ax.yaxis.tick_right()
        self.ax.tick_params(axis='x', labelsize=6)  # Konsistent mit Polar
        self.ax.set_xlabel("SPL [dB]", fontsize=6)  # Konsistent mit Polar
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        
        # Layout-Anpassungen für konsistente Größe (identisch mit initialize_empty_plots)
        self._apply_layout()



    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Passe die Plot-Ränder an Canvas-Größe an (Y-Achsen-Plot)
                # Weniger breit: left größer, right kleiner
                # Weniger hoch: top kleiner, bottom größer
                self.ax.figure.subplots_adjust(left=0.25, right=0.80, top=0.90, bottom=0.15)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()

    def _set_x_locator(self, step):
        if step != self._x_locator_step:
            self.ax.xaxis.set_major_locator(MultipleLocator(step))
            self._x_locator_step = step

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
        print(f"[DEBUG PlotSPLYaxis] _calculate_metric_ticks: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
        
        if range_val <= 0:
            # Fallback für ungültige Bereiche
            print(f"[DEBUG PlotSPLYaxis] Ungültiger Bereich, verwende Fallback")
            return [min_val, max_val], 1.0
        
        # Ziel: 4-8 Ticks für optimale Lesbarkeit
        # Berechne optimale Schrittweite basierend auf dem Bereich
        ideal_num_ticks = 6  # Ziel: ~6 Ticks
        rough_step = range_val / ideal_num_ticks
        print(f"[DEBUG PlotSPLYaxis] Rough step berechnet: {rough_step:.3f}")
        
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
            if step != old_step:
                print(f"[DEBUG PlotSPLYaxis] Feine Schrittweite gewählt: {step:.3f} (vorher: {old_step:.3f})")
        
        print(f"[DEBUG PlotSPLYaxis] Gewählte Schrittweite: {step:.3f} m")
        
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
        
        print(f"[DEBUG PlotSPLYaxis] Generierte {len(ticks)} Ticks: {ticks[:5]}..." if len(ticks) > 5 else f"[DEBUG PlotSPLYaxis] Generierte {len(ticks)} Ticks: {ticks}")
        
        return ticks, step

    def _connect_zoom_callbacks(self):
        """Verbindet Callbacks für Zoom-Events, um Achsenbeschriftung dynamisch anzupassen"""
        if self._zoom_callback_connected:
            return
        
        try:
            # Callback für Y-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            # Callback für X-Achsen-Zoom (SPL dB)
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            self._zoom_callback_connected = True
            print(f"[PlotSPLYaxis] Zoom-Callbacks erfolgreich verbunden")
        except Exception as e:
            print(f"[PlotSPLYaxis] Fehler beim Verbinden von Zoom-Callbacks: {e}")
            import traceback
            traceback.print_exc()

    def _on_ylim_changed(self, ax):
        """Wird aufgerufen, wenn sich die Y-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        try:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            print(f"[DEBUG PlotSPLYaxis] ===== CALLBACK _on_ylim_changed aufgerufen =====")
            print(f"[DEBUG PlotSPLYaxis] Y-Limits geändert: {y_min:.3f} bis {y_max:.3f} (Range: {y_range:.3f} m)")
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            # Die Schrittweite wird automatisch feiner bei kleinerem Bereich (Zoom-in)
            # und gröber bei größerem Bereich (Zoom-out)
            y_ticks, step = self._calculate_metric_ticks(y_min, y_max)
            print(f"[DEBUG PlotSPLYaxis] Neue Y-Ticks berechnet: {len(y_ticks)} Ticks, Schrittweite: {step:.3f} m")
            
            # Temporär Callback deaktivieren, um Endlosschleife zu vermeiden
            ax.callbacks.block('ylim_changed')
            ax.set_yticks(y_ticks)
            print(f"[DEBUG PlotSPLYaxis] Y-Ticks gesetzt: {len(ax.get_yticks())} Ticks")
            ax.callbacks.unblock('ylim_changed')
            
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
                print(f"[DEBUG PlotSPLYaxis] Canvas Redraw getriggert")
        except Exception as e:
            print(f"[DEBUG PlotSPLYaxis] FEHLER in _on_ylim_changed: {e}")
            import traceback
            traceback.print_exc()

    def _on_xlim_changed(self, ax):
        """Wird aufgerufen, wenn sich die X-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        try:
            x_min, x_max = ax.get_xlim()
            # Berechne neue SPL-Ticks basierend auf sichtbarem Bereich
            # Temporär Callback deaktivieren, um Endlosschleife zu vermeiden
            ax.callbacks.block('xlim_changed')
            self._update_spl_ticks(x_min, x_max)
            ax.callbacks.unblock('xlim_changed')
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            print(f"[PlotSPLYaxis] Fehler in _on_xlim_changed: {e}")

    def _update_spl_ticks(self, x_min, x_max):
        """Aktualisiert SPL-Ticks basierend auf sichtbarem Bereich"""
        x_range = x_max - x_min
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if x_range > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif x_range > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self._set_x_locator(step)