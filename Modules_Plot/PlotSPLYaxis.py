# "Grafische Ausgabe fuer SPL Plot Y-Axis"

import numpy as np
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
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
    
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
        
        # Adaptive Y-Ticks (Position) - nur ganze Zahlen
        y_range = y_max - y_min
        # Optimierte Schrittweite für bessere Lesbarkeit
        if y_range > 80:
            step = 10  # Große Bereiche
        elif y_range > 40:
            step = 10  # Mittlere Bereiche
        elif y_range > 20:
            step = 5   # Kleinere Bereiche
        else:
            step = 5   # Sehr kleine Bereiche
        y_ticks = np.arange(int(y_min), int(y_max) + step, step)
        self.ax.set_yticks(y_ticks)
        
        # Adaptive X-Ticks (SPL dB) - 3, 6 oder 10 dB Schritte
        x_range = x_max - x_min
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if x_range > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif x_range > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self.ax.xaxis.set_major_locator(MultipleLocator(step))
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.tick_params(axis='both', labelsize=6)  # Konsistent mit Polar
        self.ax.yaxis.tick_right()
        self.ax.invert_xaxis()
        
        # Layout-Anpassungen für konsistente Größe (identisch mit plot_yaxis)
        self._apply_layout()


    def plot_yaxis(self, calculation_Yaxis):
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

        # Adaptive Y-Achse Ticks (Position) - nur ganze Zahlen
        y_min_data = int(min(y_data))
        y_max_data = int(max(y_data))
        y_range = y_max_data - y_min_data
        
        # Optimierte Schrittweite für bessere Lesbarkeit
        if y_range > 80:
            step = 10  # Große Bereiche (z.B. -50 bis +50 = 100)
        elif y_range > 40:
            step = 10  # Mittlere Bereiche
        elif y_range > 20:
            step = 5   # Kleinere Bereiche
        else:
            step = 5   # Sehr kleine Bereiche
            
        y_ticks = np.arange(y_min_data, y_max_data + step, step)
        self.ax.tick_params(axis='y', labelsize=6)  # Konsistent mit Polar
        self.ax.set_yticks(y_ticks)
        self.ax.set_ylim(y_min_data, y_max_data)  # Exakt gem. length, ohne Zusatzwert
        
        # Adaptive X-Achse (SPL dB) - 3, 6 oder 10 dB Schritte
        x_min_spl = min(x_data)
        x_max_spl = max(x_data)
        x_range_spl = x_max_spl - x_min_spl
        # Ziel: 4-7 Ticks für optimale Lesbarkeit
        if x_range_spl > 35:
            step = 10  # z.B. -60 bis 0 (60 dB) → 7 Ticks oder -40 bis 0 (40 dB) → 5 Ticks
        elif x_range_spl > 20:
            step = 6   # z.B. -24 bis 0 (24 dB) → 5 Ticks
        else:
            step = 3   # z.B. -12 bis 0 (12 dB) → 5 Ticks
        self.ax.xaxis.set_major_locator(MultipleLocator(step))
        
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
                self.ax.figure.canvas.draw()