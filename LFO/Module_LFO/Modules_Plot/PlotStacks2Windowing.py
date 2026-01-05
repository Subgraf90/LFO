# "Grafische Ausgabe fuer Stacks im Windowing Plot"

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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
            
            # Berechne Transform für unverzerrte Lautsprecher-Darstellung
            # Die Lautsprecher haben Dimensionen in Metern (width x front_height)
            # X-Achse ist in Metern, Y-Achse ist in dB
            # Um die Lautsprecher unverzerrt zu halten, müssen wir die Höhe basierend auf dem Aspect-Ratio skalieren
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # Verwende get_window_extent() um die tatsächliche Pixel-Größe der Axes zu erhalten
            # (wird nach canvas.draw() aufgerufen, daher sollte es korrekte Werte liefern)
            try:
                bbox = self.ax.get_window_extent()
                axes_width_pixels = bbox.width
                axes_height_pixels = bbox.height
            except:
                # Fallback: Verwende Figure-Größe in Pixeln
                fig = self.ax.figure
                dpi = fig.get_dpi()
                axes_width_pixels = fig.get_figwidth() * dpi * 0.8  # Geschätzt
                axes_height_pixels = fig.get_figheight() * dpi * 0.8  # Geschätzt
            
            if axes_width_pixels > 0 and axes_height_pixels > 0 and x_range > 0 and y_range > 0:
                # Berechne Daten-Einheiten pro Pixel
                x_units_per_pixel = x_range / axes_width_pixels  # Meter pro Pixel
                y_units_per_pixel = y_range / axes_height_pixels  # dB pro Pixel
                
                # Für unverzerrte Darstellung: 1 Meter in X = 1 Meter visuell in Y
                # front_height ist in Metern, muss aber in dB-Einheiten umgerechnet werden
                # Um die gleiche visuelle Größe zu erreichen (gleiche Anzahl Pixel):
                # front_height (Meter) / x_units_per_pixel = scaled_height (dB) / y_units_per_pixel
                # scaled_height_basis = front_height * (y_units_per_pixel / x_units_per_pixel)
                scale_factor = y_units_per_pixel / x_units_per_pixel if x_units_per_pixel > 0 else 1.0
                scaled_front_height = front_height * scale_factor

                # Variante A: zusätzliche visuelle Skalierung der Lautsprecherhöhe
                visual_scale_factor = 1.6
                scaled_front_height *= visual_scale_factor
            else:
                # Fallback wenn Größe nicht verfügbar
                scaled_front_height = front_height
            
            # Berechne Pixel-Dimensionen für Verzerrungsprüfung
            width_pixels = (width / x_range) * axes_width_pixels if x_range > 0 else 0
            height_pixels = (scaled_front_height / y_range) * axes_height_pixels if y_range > 0 else 0
            aspect_ratio_pixels = height_pixels / width_pixels if width_pixels > 0 else 0
            aspect_ratio_real = front_height / width if width > 0 else 0
            
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
                scaled_front_height, 
                ec='black',
                fc=color
            )
            self.ax.add_patch(rect)
            self.left_positions.append(x)
            
            # #region agent log
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'E',
                    'location': 'PlotStacks2Windowing.py:135',
                    'message': 'Windowing: Rechteck erstellt - Verzerrungsprüfung',
                    'data': {
                        'x': float(x),
                        'y': float(y),
                        'width_m': float(width),
                        'scaled_front_height_dB': float(scaled_front_height),
                        'front_height_original_m': float(front_height),
                        'width_pixels': float(width_pixels),
                        'height_pixels': float(height_pixels),
                        'aspect_ratio_pixels': float(aspect_ratio_pixels),
                        'aspect_ratio_real': float(aspect_ratio_real),
                        'x_range': float(x_range),
                        'y_range': float(y_range),
                        'axes_width_pixels': float(axes_width_pixels),
                        'axes_height_pixels': float(axes_height_pixels)
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
            import traceback
            traceback.print_exc()
    
