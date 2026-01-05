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
            base_y = float(self.source_position_y[isrc])
            
            # x_offset ist bereits zentriert (Mittelpunkt des Stacks = 0)
            # Positioniere den Lautsprecher so, dass der Stack-Mittelpunkt auf base_x liegt
            # x_offset gibt die Position der linken Kante des Lautsprechers relativ zum Stack-Mittelpunkt an
            x = base_x + x_offset
            y = base_y

            # Berechne Transform für unverzerrte Lautsprecher-Darstellung
            # Die Lautsprecher haben Dimensionen in Metern (width x front_height)
            # Beide Achsen sind in Metern
            # Um die Lautsprecher unverzerrt zu halten, müssen wir die Höhe basierend auf dem Aspect-Ratio skalieren
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # #region agent log
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'PlotStacks2Beamsteering.py:74',
                    'message': 'Beamsteering: Achsengrenzen vor Pixel-Berechnung',
                    'data': {
                        'xlim': list(xlim),
                        'ylim': list(ylim),
                        'x_range': float(x_range),
                        'y_range': float(y_range),
                        'front_height_m': float(front_height),
                        'width_m': float(width),
                        'isrc': int(isrc)
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion
            
            # Verwende get_window_extent() um die tatsächliche Pixel-Größe der Axes zu erhalten
            # (wird nach canvas.draw() aufgerufen, daher sollte es korrekte Werte liefern)
            # WICHTIG: get_window_extent() liefert die tatsächliche Pixel-Größe nach allen Layout-Änderungen,
            # während get_position() nur relative Koordinaten (0-1) liefert, die nach subplots_adjust() nicht mehr korrekt sind
            try:
                bbox = self.ax.get_window_extent()
                axes_width_pixels = bbox.width
                axes_height_pixels = bbox.height
                
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'PlotStacks2Beamsteering.py:96',
                        'message': 'Beamsteering: Pixel-Größen berechnet',
                        'data': {
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels),
                            'method': 'get_window_extent'
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            except:
                # Fallback: Verwende Figure-Größe in Pixeln
                fig = self.ax.figure
                dpi = fig.get_dpi()
                axes_width_pixels = fig.get_figwidth() * dpi * 0.8  # Geschätzt
                axes_height_pixels = fig.get_figheight() * dpi * 0.8  # Geschätzt
                
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'PlotStacks2Beamsteering.py:103',
                        'message': 'Beamsteering: Fallback Pixel-Größen',
                        'data': {
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels)
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            
            if axes_width_pixels > 0 and axes_height_pixels > 0 and x_range > 0 and y_range > 0:
                # Berechne Daten-Einheiten pro Pixel
                x_units_per_pixel = x_range / axes_width_pixels  # Meter pro Pixel (X)
                y_units_per_pixel = y_range / axes_height_pixels  # Meter pro Pixel (Y)
                
                # Für unverzerrte Darstellung: Skaliere Höhe basierend auf Aspect-Ratio
                # Gleiche Logik wie in Windowing-Plot
                scale_factor = y_units_per_pixel / x_units_per_pixel if x_units_per_pixel > 0 else 1.0
                scaled_front_height = front_height * scale_factor
                
                # Variante A: zusätzliche visuelle Skalierung der Lautsprecherhöhe (wie in Windowing)
                visual_scale_factor = 1.6
                scaled_front_height *= visual_scale_factor
                
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'C',
                        'location': 'PlotStacks2Beamsteering.py:107',
                        'message': 'Beamsteering: Skalierung berechnet für unverzerrte Darstellung',
                        'data': {
                            'x_units_per_pixel': float(x_units_per_pixel),
                            'y_units_per_pixel': float(y_units_per_pixel),
                            'scale_factor': float(scale_factor),
                            'visual_scale_factor': float(visual_scale_factor),
                            'front_height_m': float(front_height),
                            'scaled_front_height_m': float(scaled_front_height)
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            else:
                # Fallback wenn Größe nicht verfügbar: zeichne ohne Korrektur (kann verzerrt sein)
                scaled_front_height = front_height
                
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'B',
                        'location': 'PlotStacks2Beamsteering.py:111',
                        'message': 'Beamsteering: Fallback verwendet, Höhe 1:1 in m',
                        'data': {
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels),
                            'x_range': float(x_range),
                            'y_range': float(y_range),
                            'scaled_front_height': float(scaled_front_height)
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion

            # Zeichne das Rechteck (Frontansicht: width x height)
            # Farbe basierend auf cardio: cardio hat Exit-Face hinten (helleres Grau), normale vorne (mittelgrau)
            body_color_cardio = '#d0d0d0'  # Helleres Grau für Cardio-Body (heller als 3D-Plot für bessere Sichtbarkeit)
            exit_color_front = '#808080'  # Mittleres Grau für Front-Exit-Face (heller als 3D-Plot für bessere Sichtbarkeit)
            # Bei cardio: exit_at_front = False → Frontansicht zeigt Body
            # Bei normal: exit_at_front = True → Frontansicht zeigt Exit-Face
            color = body_color_cardio if is_cardio else exit_color_front
            # Berechne Pixel-Dimensionen für Verzerrungsprüfung
            width_pixels = (width / x_range) * axes_width_pixels if x_range > 0 else 0
            height_pixels = (scaled_front_height / y_range) * axes_height_pixels if y_range > 0 else 0
            aspect_ratio_pixels = height_pixels / width_pixels if width_pixels > 0 else 0
            aspect_ratio_real = front_height / width if width > 0 else 0
            
            rect = patches.Rectangle(
                (x, y),
                width, 
                scaled_front_height, 
                ec='black',
                fc=color
            )
            self.ax.add_patch(rect)
            
            # #region agent log
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'E',
                    'location': 'PlotStacks2Beamsteering.py:230',
                    'message': 'Beamsteering: Rechteck erstellt - Verzerrungsprüfung',
                    'data': {
                        'x': float(x),
                        'y': float(y),
                        'width_m': float(width),
                        'scaled_front_height_m': float(scaled_front_height),
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
    
