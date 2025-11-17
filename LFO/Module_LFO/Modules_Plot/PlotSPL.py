# "Grafische Ausgabe für SPL Plot - Universell für beide Colorization-Modi"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib import cm
from Module_LFO.Modules_Init.ModuleBase import ModuleBase


class DrawSPLPlot(ModuleBase):
    """Universeller SPL-Plotter für beide Colorization-Modi"""
    num = 0
    
    def __init__(self, ax, settings, colorbar_ax):
        super().__init__(settings)  # Initialisierung der Basisklasse
        
        self.ax = ax  # Matplotlib-Achsenobjekt
        self.colorbar_ax = colorbar_ax  # Colorbar-Achsenobjekt
        self.im = None  # Für das Image-Objekt
        self.cbar = None  # Für die Farbskala
        self.settings = settings  # Einstellungen für den Plot
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()
    
    def initialize_empty_plots(self):
        """Initialisiert den Plot mit sinnvoller Leer-Darstellung"""
        self.ax.clear()
        
        # Verwende Settings für realistische Grenzen
        x_min = -(self.settings.width / 2)
        x_max = (self.settings.width / 2)
        y_min = -(self.settings.length / 2)
        y_max = (self.settings.length / 2)
        
        # Setze grundlegende Achseneinstellungen
        self.ax.set_aspect('equal')
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Setze Ticks basierend auf Schallfeld-Größe
        x_spacing = 5
        y_spacing = 5
        x_ticks = np.arange(x_min, x_max + x_spacing, x_spacing)
        y_ticks = np.arange(y_min, y_max + y_spacing, y_spacing)
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        
        self.ax.grid(color='k', linestyle='-', linewidth=0.4, alpha=0.3)
        self.ax.tick_params(axis='both', labelsize=7)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.text(0.5, 0.5, 'No active speaker', 
                    transform=self.ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Initialisiere leere Colorbar
        self._initialize_empty_colorbar()
        
        # Layout-Anpassungen für konsistente Größe
        if hasattr(self.ax, 'figure') and hasattr(self.ax.figure, 'canvas'):
            try:
                # Verwende constrained_layout für SPL-Plot
                self.ax.figure.set_constrained_layout(True)
            except Exception:
                pass

    def _initialize_empty_colorbar(self):
        """Erstellt eine leere Colorbar beim Start gemäß colorization_mode aus Settings"""
        try:
            # Colorbar-Einstellungen aus settings verwenden
            cbar_min = self.settings.colorbar_range['min']
            cbar_max = self.settings.colorbar_range['max']
            cbar_step = self.settings.colorbar_range['step']
            cbar_tick_step = self.settings.colorbar_range['tick_step']
            
            # Clear colorbar axis
            if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
                self.colorbar_ax.cla()
            
            # Colorbar-Ticks
            cbar_ticks = np.arange(cbar_min, cbar_max + cbar_tick_step, cbar_tick_step)
            
            # Wähle Normalisierung basierend auf colorization_mode aus Settings
            if self.settings.colorization_mode == 'Color step':
                # 3dB Steps
                levels = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
                norm = BoundaryNorm(levels, ncolors=256, clip=True)
            else:
                # Gradient (linear)
                norm = Normalize(vmin=cbar_min, vmax=cbar_max)
            
            # Erstelle eine ScalarMappable für die Colorbar
            sm = cm.ScalarMappable(cmap='jet', norm=norm)
            sm.set_array([])
            
            # Erstelle die Colorbar
            self.cbar = plt.colorbar(sm, cax=self.colorbar_ax, ticks=cbar_ticks)
            self.colorbar_ax.tick_params(labelsize=7)
            self.colorbar_ax.set_position([0.1, 0.05, 0.1, 0.9])
            
        except Exception as e:
            print(f"Warning: Leere Colorbar konnte nicht erstellt werden: {e}")


    def update_spl_plot(self, sound_field_x, sound_field_y, sound_field_pressure, colorization_mode='Gradient'):
        """
        Aktualisiert den SPL-Plot mit den gegebenen Daten
        
        Args:
            sound_field_x: X-Koordinaten
            sound_field_y: Y-Koordinaten
            sound_field_pressure: Druckwerte
            colorization_mode: 'Color step' oder 'Gradient'
        """
        # Prüfe ob gültige Daten vorhanden sind
        if (sound_field_x is None or sound_field_y is None or sound_field_pressure is None or
            len(sound_field_x) == 0 or len(sound_field_y) == 0 or len(sound_field_pressure) == 0):
            self.initialize_empty_plots()
            return
            
        self.ax.clear()  # Löschen des vorherigen Plots
        
        # Umformen der Daten für den Plot
        sound_field_pressure_2d = np.reshape(sound_field_pressure, (len(sound_field_y), len(sound_field_x)))

        # Colorbar-Einstellungen aus settings verwenden
        cbar_min = self.settings.colorbar_range['min']
        cbar_max = self.settings.colorbar_range['max']
        cbar_step = self.settings.colorbar_range['step']
        cbar_tick_step = self.settings.colorbar_range['tick_step']
        
        # Wähle die passende Normalisierung basierend auf colorization_mode
        if colorization_mode == 'Color step':
            # 3dB Steps
            levels = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
            norm = BoundaryNorm(levels, ncolors=256, clip=True)
        else:
            # Gradient (linear)
            norm = Normalize(vmin=cbar_min, vmax=cbar_max)
        
        # Erstellen des Plots mit imshow
        im = self.ax.imshow(self.functions.mag2db(sound_field_pressure_2d),
                    extent=[min(sound_field_x), max(sound_field_x),
                            min(sound_field_y), max(sound_field_y)],
                    cmap='jet',
                    origin='lower',
                    norm=norm)  # ← Verwendet die gewählte Normalisierung

        # Achseneinstellungen
        self.ax.set_aspect('equal')
        self.ax.tick_params(axis='both', which='both', labelsize=9)

        # Setzen der Grid-Linien-Abstände
        x_min, x_max = min(sound_field_x), max(sound_field_x)
        y_min, y_max = min(sound_field_y), max(sound_field_y)
        
        x_spacing = 5
        y_spacing = 5
        
        x_ticks = np.arange(x_min, x_max + x_spacing, x_spacing)
        y_ticks = np.arange(y_min, y_max + y_spacing, y_spacing)
        
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)

        # Anpassen des Grids
        self.ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.5)

        # Entfernen der Achsenbeschriftungen
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Colorbar-Handling: Update statt Neuerstellung wenn möglich
        cbar_ticks = np.arange(cbar_min, cbar_max + cbar_tick_step, cbar_tick_step)
        
        if hasattr(self, 'cbar') and self.cbar is not None:
            # Versuche Colorbar zu updaten statt neu zu erstellen
            try:
                self.cbar.update_normal(im)
                self.cbar.set_ticks(cbar_ticks)
                self.colorbar_ax.tick_params(labelsize=7)
            except Exception:
                # Falls Update fehlschlägt, neu erstellen
                try:
                    self.cbar.remove()
                except Exception:
                    pass
                self.cbar = None
        
        # Erstelle Colorbar nur wenn noch keine existiert
        if self.cbar is None:
            # Clear colorbar axis nur wenn sicher
            if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
                try:
                    self.colorbar_ax.cla()
                except Exception:
                    pass
            
            # Erstellen der neuen Colorbar
            try:
                self.cbar = self.ax.figure.colorbar(im, cax=self.colorbar_ax, ticks=cbar_ticks)
                self.colorbar_ax.tick_params(labelsize=7)
                
                # Anpassen der Colorbar-Größe
                self.colorbar_ax.set_position([0.1, 0.05, 0.1, 0.9])
            except AttributeError as e:
                print(f"Warning: Colorbar AttributeError: {e}")
            except Exception as e:
                print(f"Warning: Colorbar konnte nicht erstellt werden: {e}")
        
        # Layout-Anpassungen für konsistente Größe
        try:
            self.ax.figure.set_constrained_layout(True)
        except Exception:
            pass
        
        # Aktualisieren der Figur (mit Error Handling)
        try:
            self.ax.figure.canvas.draw()
        except Exception:
            pass
        
        # Colorbar canvas draw nur wenn Figure vorhanden
        if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'figure') and self.colorbar_ax.figure is not None:
            try:
                self.colorbar_ax.figure.canvas.draw()
            except Exception:
                pass

