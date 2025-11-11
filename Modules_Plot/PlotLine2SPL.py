import matplotlib.pyplot as plt
import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi

class Lineplot2SPL(ModuleBase):
    def __init__(self, ax, settings, speaker_array):
        super().__init__(settings)  # Korrekte Initialisierung der Basisklasse "ModuleBase"
        self.ax = ax
        self.im = None
        self.settings = settings
        self.speaker_array = speaker_array  # Hinzugefügt: Initialisierung von speaker_array
            
    def draw_axis_lines(self):
        self.ax.vlines(self.settings.position_x_axis, -self.settings.length/2, self.settings.length/2, color='black', linestyle='-.', linewidth=1)
        self.ax.hlines(self.settings.position_y_axis, -self.settings.width/2, self.settings.width/2, color='black', linestyle='-.', linewidth=1)

    def draw_wall_lines(self, wall_index):
        return
                        
    def draw_impulse_points(self):
        triangle_size = self.settings.measurement_size
        
        for point in self.settings.impulse_points:
            key = point['key']
            center = point['data']
            # Erzeuge ein gleichseitiges Dreieck um die (x, y)-Position
            triangle = self.create_triangle(center, triangle_size)
            # Zeichne das Dreieck
            triangle_patch = plt.Polygon(triangle, closed=True, edgecolor='r')
            self.ax.add_patch(triangle_patch)

    def create_triangle(self, center, size):
        x, y = center
        h = size * (3 ** 0.5) / 2  # Höhe eines gleichseitigen Dreiecks
        points = np.array([
            [x, y + 2 * h / 3],        # Spitze oben
            [x - size / 2, y - h / 3], # Ecke unten links
            [x + size / 2, y - h / 3]  # Ecke unten rechts
        ])
        return points

