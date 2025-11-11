import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DrawPolarPattern(ModuleBase):
    def __init__(self, matplotlib_canvas_polar_pattern, settings):
        super().__init__(settings)
        self.ax = matplotlib_canvas_polar_pattern
        self.settings = settings
        
        # Konfiguriere die Plot-Eigenschaften
        self.ax.set_theta_zero_location('N')  # 0 Grad ist oben
        self.ax.set_theta_direction(1)       # gegen den Uhrzeigersinn
        self.ax.grid(True)
        
        # Nutze automatisches Layout statt fester Position
        # (set_position wird von constrained_layout überschrieben)
        
        # Dictionary für Farben
        self.colors = {
            'red': 'red',
            'yellow': 'yellow',
            'green': 'green',
            'cyan': 'cyan'
        }
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
    
    def initialize_empty_plots(self):
        """Initialisiert den Plot mit sinnvoller Leer-Darstellung"""
        self.ax.clear()
        
        # Setze Plot-Eigenschaften
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(1)
        self.ax.grid(True)
        
        # Setze Achsengrenzen
        self.ax.set_rlabel_position(0)  # Position: oben (vertikal)
        self.ax.set_rticks([-24, -18, -12, -6, 0])
        self.ax.set_rlim([-24, 0])
        # Leerzeichen vor den Labels für Verschiebung nach rechts
        self.ax.set_yticklabels(['  -24 dB', '  ', '  -12 dB', ' ', '  0 dB'], 
                               fontsize=6, rotation=0)
        self.ax.tick_params(axis='x', labelsize=6, pad=-2)  # Winkel-Labels etwas näher ran
        self.ax.tick_params(axis='y', labelsize=6)
        
        # Erstelle Legende mit Farbbeschriftung (identisch zum Datenplot)
        # Hole die Farb-Frequenz-Zuordnung aus den Settings
        freq_colors = self.settings.polar_frequencies
        
        # Erstelle unsichtbare Linien für die Legende (mit default linewidth wie bei Daten)
        for color, freq in freq_colors.items():
            self.ax.plot([], [], color=self.colors[color], label=f'{freq} Hz')
        
        # Legende mit Quadraten - unten rechts (identisch zum Datenplot)
        legend = self.ax.legend(bbox_to_anchor=(1.20, 0.0),
                               loc='lower left',
                               fontsize=6,
                               ncol=1,
                               frameon=False,
                               handlelength=0.8,
                               handleheight=0.8)
        
        # Ändere die Legende-Handles zu Quadraten (identisch zum Datenplot)
        legend_handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
        for handle in legend_handles:
            handle.set_marker('s')  # Quadrat
            handle.set_markersize(6)  # Größe des Quadrats
            handle.set_linestyle('')  # Keine Linie
        
        # Layout-Anpassungen für konsistente Größe (identisch mit update_polar_pattern)
        self._apply_layout()
    
    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Verwende feste subplots_adjust Werte statt tight_layout
                # für konsistente Größe (Empty-Plot und Datenplot)
                # Polar-Plot noch kleiner: mehr Rand links/rechts/oben/unten
                self.ax.figure.subplots_adjust(left=0.18, right=0.67, top=0.82, bottom=0.18)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw()

    def update_polar_pattern(self, polar_data):
        """
        Aktualisiert den Polar Plot
        """
        try:
            # Prüfe ob gültige Daten vorhanden sind
            if not polar_data or 'angles' not in polar_data or 'sound_field_p' not in polar_data:
                self.initialize_empty_plots()
                return
            
            # Prüfe ob sound_field_p tatsächlich Daten enthält
            if not polar_data['sound_field_p'] or len(polar_data['sound_field_p']) == 0:
                self.initialize_empty_plots()
                return
            
            self.ax.clear()
            
            # Plot-Eigenschaften
            self.ax.set_theta_zero_location('N')  # 0 Grad ist oben
            self.ax.set_theta_direction(1)       # gegen den Uhrzeigersinn
            self.ax.grid(True)
                
            # Konvertiere Winkel zu Radiant und verschiebe um -90°
            theta = np.deg2rad(polar_data['angles'] - 90)
            
            # Hole die Farb-Frequenz-Zuordnung aus den Settings
            freq_colors = self.settings.polar_frequencies
            
            # Prüfe ob überhaupt Daten zu plotten sind
            has_data = False
            
            # Plotte für jede Frequenz
            for freq, values in polar_data['sound_field_p'].items():
                # Finde die zugehörige Farbe
                color = next((color for color, f in freq_colors.items() 
                             if abs(float(f) - float(freq)) < 0.1), 'gray')
                
                self.ax.plot(theta, values, 
                            color=self.colors[color],
                            label=f'{freq} Hz')
                has_data = True
            
            # Wenn keine Daten geplottet wurden, zeige Empty Plot
            if not has_data:
                self.initialize_empty_plots()
                return
            
            # Achsenbeschriftungen vertikal
            self.ax.set_rlabel_position(0)  # Position: oben (vertikal)
            self.ax.set_rticks([-24, -18, -12, -6, 0])  # Ticks müssen mit Labels übereinstimmen
            self.ax.set_rlim([-24, 0])
            
            # Labels vertikal mit Schriftgröße - gleiche Anzahl wie Ticks
            # Leerzeichen vor den Labels für Verschiebung nach rechts
            self.ax.set_yticklabels(['  -24 dB', '  ', '  -12 dB', ' ', '  0 dB'], 
                                   fontsize=6,
                                   rotation=0)  # Vertikale Ausrichtung

            # Winkel-Ticks - Mitte zwischen vorher (0) und -5
            self.ax.tick_params(axis='x', labelsize=6, pad=-2)  # Milder: -2 statt -5
            self.ax.tick_params(axis='y', labelsize=6)
            
            # Legende mit Quadraten - unten rechts
            legend = self.ax.legend(bbox_to_anchor=(1.20, 0.0),  # Weiter nach rechts (vorher 1.12)
                                   loc='lower left',  # Von unten links ausgerichtet
                                   fontsize=6,
                                   ncol=1,
                                   frameon=False,  # Keine Umrandung
                                   handlelength=0.8,
                                   handleheight=0.8)
            
            # Ändere die Legende-Handles zu Quadraten
            legend_handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
            for handle in legend_handles:
                handle.set_marker('s')  # Quadrat
                handle.set_markersize(6)  # Größe des Quadrats
                handle.set_linestyle('')  # Keine Linie
            
            # Layout-Anpassungen für konsistente Größe (identisch mit initialize_empty_plots)
            self._apply_layout()
            
        except Exception as e:
            print(f"Fehler beim Plotten: {e}")
            self.ax.figure.canvas.draw()

    # def normalize_values(self, values):
        """
        Normalisiert die Werte für die Polar-Darstellung auf 0 dB Maximum
        """
        try:
            # Konvertiere zu float64 Array
            values = np.array(values, dtype=np.float64)
            
            # Ersetze ungültige Werte
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(values) == 0 or np.max(values) <= 0:
                return np.zeros_like(values)
                
            # Finde Maximum für diese Frequenz
            max_value = np.max(values)
            
            # Berechne dB-Werte relativ zum Maximum (max wird 0 dB)
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = 20 * np.log10(values / max_value)
            
            # Ersetze -inf durch -40 und begrenze auf -40 bis 0 dB
            normalized = np.nan_to_num(normalized, nan=-40.0, neginf=-40.0)
            normalized = np.clip(normalized, -40, 0)
            
            return normalized
            
        except Exception as e:
            return np.zeros(360)