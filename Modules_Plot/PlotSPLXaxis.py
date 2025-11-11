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
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
    
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
        
        # Adaptive X-Ticks (Position) - nur ganze Zahlen
        x_range = x_max - x_min
        # Optimierte Schrittweite für bessere Lesbarkeit
        if x_range > 80:
            step = 10  # Große Bereiche
        elif x_range > 40:
            step = 10  # Mittlere Bereiche
        elif x_range > 20:
            step = 5   # Kleinere Bereiche
        else:
            step = 5   # Sehr kleine Bereiche
        x_ticks = list(range(int(x_min), int(x_max) + step, step))
        self.ax.set_xticks(x_ticks)
        
        # Adaptive Y-Ticks (SPL dB) - 3, 6 oder 10 dB Schritte
        y_range = y_max - y_min
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if y_range > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif y_range > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self._set_y_locator(step)
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.tick_params(axis='both', labelsize=6)  # Konsistent mit Polar
        # self.ax.text(0.5, 0.5, 'No active speakerz', 
        #             transform=self.ax.transAxes,
        #             ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Layout-Anpassungen für konsistente Größe (identisch mit plot_xaxis)
        self._apply_layout()


    def plot_xaxis(self, calculation_Xaxis):
        import sys
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
        
        for key, value in calculation_Xaxis.items():
            # boleanabfrage. wenn show_in_plot true, wird der plot ausgeführt
            if 'show_in_plot' in value and value['show_in_plot']: 
                x_data = value['x_data_xaxis']
                y_data = value['y_data_xaxis']

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

                # self.ax.plot.tight_layout(y_data, x_data, label=key, color=color)

        # Adaptive X-Achse Ticks (Position) - nur ganze Zahlen
        x_min_data = int(min(y_data))
        x_max_data = int(max(y_data))
        x_range = x_max_data - x_min_data
        
        # Optimierte Schrittweite für bessere Lesbarkeit
        if x_range > 80:
            step = 10  # Große Bereiche
        elif x_range > 40:
            step = 10  # Mittlere Bereiche
        elif x_range > 20:
            step = 5   # Kleinere Bereiche
        else:
            step = 5   # Sehr kleine Bereiche
            
        x_ticks = list(range(x_min_data, x_max_data + step, step))
        
        self.ax.tick_params(axis='x', labelsize=6)  # Konsistent mit Polar
        self.ax.set_xticks(x_ticks)
        self.ax.set_xlim(x_min_data, x_max_data)  # Exakt gem. width, ohne Zusatzwert
        # self.ax.set_xlabel("Distance to array [m]")    
        
        # Adaptive Y-Achse (SPL dB) - 3, 6 oder 10 dB Schritte
        y_min_spl = min(x_data)
        y_max_spl = max(x_data)
        y_range_spl = y_max_spl - y_min_spl
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if y_range_spl > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif y_range_spl > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self._set_y_locator(step)
            
        self.ax.tick_params(axis='y', labelsize=6)  # Konsistent mit Polar
        self.ax.set_ylabel("SPL [dB]", fontsize=6)  # Konsistent mit Polar
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        
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