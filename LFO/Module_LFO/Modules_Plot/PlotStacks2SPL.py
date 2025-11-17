# "Grafische Ausgabe fuer Stacks im SPL Plot"

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class StackDraw_SPLPlot:
    def __init__(self, source_polar_pattern, source_position_x, source_position_y, source_azimuth, width, length, ax, cabinet_data=None):
        self.source_polar_pattern = source_polar_pattern
        self.source_position_x = source_position_x
        self.source_position_y = source_position_y
        self.source_azimuth_deg = source_azimuth
        self.source_azimuth = np.deg2rad(source_azimuth)
        self.width = width
        self.length = length
        self.ax = ax
        self.left_positions = []
        self.cabinet_data = cabinet_data

    def draw_stack(self, isrc):
        """Zeichnet einen Stack von Lautsprechern"""
        try:
            if 0 <= isrc < len(self.source_polar_pattern):

                if isinstance(self.cabinet_data, list) and isrc < len(self.cabinet_data):
                    cabinet = self.cabinet_data[isrc]
                    
                    # Verarbeite nur NPZ-Format (Arrays von Lautsprechern)
                    if isinstance(cabinet, np.ndarray):
                        # Zeichne zuerst alle Lautsprecher
                        for i, single_cabinet in enumerate(cabinet):
                            self.draw_cabinet_based_stack(isrc, single_cabinet)
                        
                        # Dann zeichne eine einzelne Ausrichtungslinie in der Mitte
                        base_x = float(self.source_position_x[isrc])
                        base_y = float(self.source_position_y[isrc])
                        
                        # Berechne Endpunkt für die zentrale Linie
                        line_point_x, line_point_y = self.calculate_azi_line_point(isrc)
                        
                        # Zeichne die zentrale Ausrichtungslinie
                        self.ax.plot([base_x, line_point_x], 
                                   [base_y, line_point_y], 
                                   color='black', linestyle='-', linewidth=0.3)
                            
        except Exception as e:
            print(f"ERROR beim Zeichnen des Stacks {isrc}: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_cabinet_based_stack(self, isrc, cabinet):
        """Zeichnet einen einzelnen Lautsprecher aus dem Stack"""
        try:
            width = float(cabinet.get('width', 1.35))
            depth = float(cabinet.get('depth', 0.72))
            is_cardio = bool(cabinet.get('cardio', False))
            x_offset = float(cabinet.get('x_offset', 0.0))

            # Basis-Position ist die Mitte des Clusters
            base_x = float(self.source_position_x[isrc])
            base_y = float(self.source_position_y[isrc])
            
            # Addiere den Offset zur X-Position
            x = base_x + x_offset
            y = base_y

            # Zeichne nur das Rechteck, keine Linie
            rect = patches.Rectangle(
                (x, y - depth),
                width, 
                depth, 
                ec='black',
                fc='#909090' if is_cardio else '#C5C5C5'
            )
            self.ax.add_patch(rect)
            self.left_positions.append(x - width/2)
            
            self.ax.set_xlim([-self.width / 2, self.width / 2])
            self.ax.set_ylim([-self.length / 2, self.length / 2])
            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")

    def calculate_azi_line_point(self, isrc):
        """Berechnet den Endpunkt der Ausrichtungslinie"""
        angle_rad = np.radians(self.source_azimuth_deg[isrc])

        dy = self.width / 2 - abs(self.source_position_y[isrc])
        dx = self.length / 2 - abs(self.source_position_x[isrc])

        if abs(np.sin(angle_rad)) > 1e-6:
            t1 = dx / abs(np.sin(angle_rad))
        else:
            t1 = np.inf
        if abs(np.cos(angle_rad)) > 1e-6:
            t2 = dy / abs(np.cos(angle_rad))
        else:
            t2 = np.inf
        
        t = min(t1, t2)

        line_point_x = self.source_position_x[isrc] + t * np.sin(angle_rad)
        line_point_y = self.source_position_y[isrc] + t * np.cos(angle_rad)

        return line_point_x, line_point_y


