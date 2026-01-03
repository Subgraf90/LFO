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
        """Zeichnet einen einzelnen Lautsprecher aus dem Stack (Frontansicht: width x height)"""
        try:
            if not isinstance(cabinet, dict):
                return
            
            # Werte direkt aus Metadaten holen (wie im 3D-Plot)
            width = float(cabinet.get('width', 0))
            if width <= 0:
                return
            
            # Frontansicht: verwende front_height statt depth
            # Fallback auf height (wie im 3D-Plot bei back_height)
            front_height = float(cabinet.get('front_height', cabinet.get('height', 0)))
            if front_height <= 0:
                return
            
            is_cardio = bool(cabinet.get('cardio', False))
            x_offset = float(cabinet.get('x_offset', 0.0))
            
            # Basis-Position ist die Mitte des Clusters
            base_x = float(self.source_position_x[isrc])
            base_y = float(self.window_restriction)  # Y-Position ist window_restriction
                        
            # x_offset ist bereits zentriert (Mittelpunkt des Stacks = 0)
            # Positioniere den Lautsprecher so, dass der Stack-Mittelpunkt auf base_x liegt
            # x_offset gibt die Position der linken Kante des Lautsprechers relativ zum Stack-Mittelpunkt an
            x = base_x + x_offset
            y = base_y
            
            # Zeichne nur das Rechteck (Frontansicht: width x height)
            # Farbe basierend auf cardio: cardio hat Exit-Face hinten (helleres Grau), normale vorne (mittelgrau)
            body_color_cardio = '#d0d0d0'  # Helleres Grau für Cardio-Body (heller als 3D-Plot für bessere Sichtbarkeit)
            exit_color_front = '#808080'  # Mittleres Grau für Front-Exit-Face (heller als 3D-Plot für bessere Sichtbarkeit)
            # Bei cardio: exit_at_front = False → Frontansicht zeigt Body
            # Bei normal: exit_at_front = True → Frontansicht zeigt Exit-Face
            color = body_color_cardio if is_cardio else exit_color_front
            rect = patches.Rectangle(
                (x, y),
                width, 
                front_height, 
                ec='black',
                fc=color
            )
            self.ax.add_patch(rect)
            self.left_positions.append(x)            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
            import traceback
            traceback.print_exc()
    
