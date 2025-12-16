# "Grafische Ausgabe fuer Stacks im Windowing Plot"

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from Module_LFO.Modules_Calculate.Functions import FunctionToolbox


class StackDraw_Beamsteering:
    def __init__(self, source_polar_pattern, source_position_x, source_position_y, source_azimuth, width, length, ax, cabinet_data=None):
        self.source_polar_pattern = source_polar_pattern
        self.source_position_x = source_position_x
        self.source_position_y = source_position_y
        self.source_azimuth_deg = source_azimuth
        self.source_azimuth = FunctionToolbox.deg2rad(self, source_azimuth)
        self.width = width
        self.length = length
        self.ax = ax
        self.cabinet_data = cabinet_data

    def draw_stack(self, isrc):
        """Zeichnet einen Stack von Lautsprechern"""
        try:
            if 0 <= isrc < len(self.source_polar_pattern):
                if isinstance(self.cabinet_data, list) and isrc < len(self.cabinet_data):
                    cabinet = self.cabinet_data[isrc]
                    
                    # Verarbeite verschiedene Datentypen
                    if isinstance(cabinet, (np.ndarray, list)):
                        # Zeichne alle Lautsprecher (Array oder Liste)
                        for single_cabinet in cabinet:
                            self.draw_cabinet_based_stack(isrc, single_cabinet)
                    elif isinstance(cabinet, dict):
                        # Behandle dict als einzelnen Lautsprecher
                        self.draw_cabinet_based_stack(isrc, cabinet)
                            
        except Exception as e:
            print(f"Fehler beim Zeichnen des Stacks {isrc}: {str(e)}")

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

            # Zeichne das Rechteck
            rect = patches.Rectangle(
                (x, y - depth),
                width, 
                depth, 
                ec='black',
                fc='#909090' if is_cardio else '#C5C5C5'
            )
            self.ax.add_patch(rect)
            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
    
