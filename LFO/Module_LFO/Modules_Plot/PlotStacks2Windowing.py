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
            
            # Werte direkt aus Metadaten holen
            width = float(cabinet.get('width', 0))
            if width <= 0:
                return
            
            # Verwende front_height aus Metadaten
            front_height = float(cabinet.get('front_height', cabinet.get('height', 0)))
            if front_height <= 0:
                return
            
            # #region agent log - Ursprungsabmessungen Windowing
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Windowing.py:70',
                    'message': 'Windowing: Ursprungsabmessungen aus Cabinet-Daten',
                    'data': {
                        'isrc': int(isrc),
                        'width_original_m': float(width),
                        'front_height_original_m': float(front_height),
                        'cabinet_width': cabinet.get('width'),
                        'cabinet_front_height': cabinet.get('front_height'),
                        'cabinet_height': cabinet.get('height'),
                        'aspect_ratio_original': float(front_height / width) if width > 0 else 0,
                        'vergleich': {
                            'breite_m': float(width),
                            'hoehe_m': float(front_height),
                            'hoehe_div_breite': float(front_height / width) if width > 0 else 0
                        }
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion
            
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
            
            # Im Windowing-Plot ist die Y-Achse in dB (nicht Meter), daher keine Umrechnung für Seitenverhältnis nötig
            # Die Lautsprecher haben Dimensionen in Metern (width x front_height)
            # X-Achse ist in Metern, Y-Achse ist in dB
            # Verwende die originale Höhe direkt (in dB-Einheiten)
            scaled_front_height = front_height
            
            # #region agent log - Abmessungsberechnung Windowing
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Windowing.py:122',
                    'message': 'Windowing: Abmessungsberechnung - keine Skalierung (Y-Achse in dB)',
                    'data': {
                        'isrc': int(isrc),
                        'breite': {
                            'original_m': float(width),
                            'verwendet_m': float(width),
                            'skaliert': False,
                            'skalierungsfaktor': 1.0
                        },
                        'hoehe': {
                            'original_m': float(front_height),
                            'verwendet_dB': float(scaled_front_height),
                            'skaliert': False,
                            'skalierungsfaktor': 1.0
                        },
                        'keine_skalierung': True,
                        'grund': 'Y-Achse in dB, keine Seitenverhältnis-Korrektur nötig',
                        'vergleich': {
                            'breite_original_m': float(width),
                            'hoehe_original_m': float(front_height),
                            'breite_verwendet_m': float(width),
                            'hoehe_verwendet_dB': float(scaled_front_height),
                            'aspect_ratio_original': float(front_height / width) if width > 0 else 0
                        }
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion
            
            # Hole Achsengrenzen und Pixel-Größe für Verzerrungsprüfung
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
            
            # Berechne Pixel-Dimensionen für Verzerrungsprüfung
            width_pixels = (width / x_range) * axes_width_pixels if x_range > 0 else 0
            height_pixels = (scaled_front_height / y_range) * axes_height_pixels if y_range > 0 else 0
            aspect_ratio_pixels = height_pixels / width_pixels if width_pixels > 0 else 0
            aspect_ratio_real = front_height / width if width > 0 else 0
            
            # Zeichne nur das Rechteck (Frontansicht: width x front_height)
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
            
            # #region agent log - Finale Abmessungen beim Zeichnen Windowing
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Windowing.py:192',
                    'message': 'Windowing: Rechteck gezeichnet - verwendete Abmessungen',
                    'data': {
                        'isrc': int(isrc),
                        'position': {
                            'x_m': float(x),
                            'y_dB': float(y),
                            'oberkante_y_dB': float(y + scaled_front_height)
                        },
                        'abmessungen_original': {
                            'breite_m': float(width),
                            'hoehe_m': float(front_height),
                            'aspect_ratio': float(aspect_ratio_real)
                        },
                        'abmessungen_verwendet': {
                            'breite_m': float(width),
                            'hoehe_dB': float(scaled_front_height),
                            'aspect_ratio_pixel': float(aspect_ratio_pixels)
                        },
                        'pixel_dimensionen': {
                            'breite_px': float(width_pixels),
                            'hoehe_px': float(height_pixels),
                            'aspect_ratio_px': float(aspect_ratio_pixels)
                        },
                        'skalierung': {
                            'breite_faktor': 1.0,
                            'hoehe_faktor': 1.0,
                            'hoehe_original_m': float(front_height),
                            'hoehe_verwendet_dB': float(scaled_front_height),
                            'keine_skalierung': True
                        },
                        'achsen': {
                            'x_range_m': float(x_range),
                            'y_range_dB': float(y_range),
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels)
                        },
                        'vergleich': {
                            'breite_original_vs_verwendet': float(width) == float(width),
                            'hoehe_original_vs_verwendet': float(front_height) == float(scaled_front_height),
                            'aspect_ratio_original': float(aspect_ratio_real),
                            'aspect_ratio_pixel': float(aspect_ratio_pixels),
                            'hinweis': 'Y-Achse in dB, daher keine direkte Vergleichbarkeit der Höhe'
                        }
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
            import traceback
            traceback.print_exc()
    
