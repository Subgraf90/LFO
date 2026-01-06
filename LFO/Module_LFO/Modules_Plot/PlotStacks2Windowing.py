# "Grafische Ausgabe fuer Stacks im Windowing Plot"

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class StackDraw_Windowing:
    def __init__(self, source_polar_pattern, source_position_x, source_position_y, source_azimuth, width, length, window_restriction, ax, cabinet_data=None):
        self.source_polar_pattern = source_polar_pattern
        self.source_position_x = source_position_x
        self.source_position_y = source_position_y
        self.source_azimuth_deg = source_azimuth
        self.source_azimuth = np.deg2rad(source_azimuth)
        self.width = width
        self.length = length
        self.window_restriction = window_restriction
        self.ax = ax
        self.left_positions = []
        self.cabinet_data = cabinet_data

    def draw_stack(self, isrc):
        """Zeichnet einen Stack von Lautsprechern"""
        try:
            
            if 0 <= isrc < len(self.source_polar_pattern):
                
                if isinstance(self.cabinet_data, list):
                    
                    if isrc < len(self.cabinet_data):
                        cabinet = self.cabinet_data[isrc]
                        
                        # Verarbeite verschiedene Datentypen
                        if isinstance(cabinet, (np.ndarray, list)):
                            # Zeichne alle Lautsprecher (Array oder Liste)
                            for i, single_cabinet in enumerate(cabinet):
                                self.draw_cabinet_based_stack(isrc, single_cabinet)
                        elif isinstance(cabinet, dict):
                            # Behandle dict als einzelnen Lautsprecher
                            self.draw_cabinet_based_stack(isrc, cabinet)
                        else:
                            print(f"  cabinet ist weder np.ndarray, list noch dict, sondern: {type(cabinet)}")
                    else:
                        print(f"  isrc={isrc} ist größer/gleich cabinet_data Länge={len(self.cabinet_data)}")
                else:
                    print(f"  cabinet_data ist KEINE Liste, sondern: {type(self.cabinet_data)}")
                            
        except Exception as e:
            print(f"Fehler beim Zeichnen des Stacks {isrc}: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_cabinet_based_stack(self, isrc, cabinet):
        """Zeichnet einen einzelnen Lautsprecher aus dem Stack"""
        try:
            
            width = float(cabinet.get('width', 0))
            if width <= 0:
                return
            
            frontheight = float(cabinet.get('front_height', 0))
            if frontheight <= 0:
                return
            
            is_cardio = bool(cabinet.get('cardio', False))
            x_offset = float(cabinet.get('x_offset', 0.0))
            
            # Basis-Position ist die Mitte des Clusters
            base_x = float(self.source_position_x[isrc])
            base_y = float(self.window_restriction)  # Y-Position ist window_restriction
                        
            # Addiere den Offset zur X-Position
            x = base_x + x_offset
            y = base_y
            
            # Zeichne nur das Rechteck
            # Normal: dunkle Fläche ist vorne (sichtbar) → dunkel
            # Cardio: helle Fläche/Body ist vorne (sichtbar) → hell
            rect = patches.Rectangle(
                (x, y - frontheight),  # Position ist (x, window_restriction - frontheight)
                width, 
                frontheight, 
                ec='black',
                fc='#C5C5C5' if is_cardio else '#909090'
            )
            self.ax.add_patch(rect)
            self.left_positions.append(x - width/2)            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
            import traceback
            traceback.print_exc()
    
