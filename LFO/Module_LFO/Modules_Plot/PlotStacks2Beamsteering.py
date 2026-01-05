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
            
            # Verwende get_position() und Figure-Größe um die Pixel-Größe der Axes zu berechnen
            try:
                fig = self.ax.figure
                dpi = fig.get_dpi()
                fig_width_pixels = fig.get_figwidth() * dpi
                fig_height_pixels = fig.get_figheight() * dpi
                
                # Hole die Position der Axes in Figure-Koordinaten (0-1)
                pos = self.ax.get_position()
                axes_width_pixels = pos.width * fig_width_pixels
                axes_height_pixels = pos.height * fig_height_pixels
                
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'PlotStacks2Beamsteering.py:88',
                        'message': 'Beamsteering: Pixel-Größen berechnet',
                        'data': {
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels),
                            'fig_width_pixels': float(fig_width_pixels),
                            'fig_height_pixels': float(fig_height_pixels),
                            'pos_width': float(pos.width),
                            'pos_height': float(pos.height),
                            'dpi': float(dpi),
                            'method': 'get_position'
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            except Exception as e:
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
                        'location': 'PlotStacks2Beamsteering.py:95',
                        'message': 'Beamsteering: Fallback Pixel-Größen',
                        'data': {
                            'axes_width_pixels': float(axes_width_pixels),
                            'axes_height_pixels': float(axes_height_pixels),
                            'error': str(e)
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            
            if axes_width_pixels > 0 and axes_height_pixels > 0 and x_range > 0 and y_range > 0:
                # Berechne Daten-Einheiten pro Pixel
                x_units_per_pixel = x_range / axes_width_pixels  # Meter pro Pixel
                y_units_per_pixel = y_range / axes_height_pixels  # Meter pro Pixel
                
                # Ziel: Lautsprecher sollen im Plot nicht verzerrt werden.
                # D.h. das Seitenverhältnis (width : front_height) soll in Pixeln erhalten bleiben,
                # auch wenn die Achsen unterschiedlich skaliert sind.
                #
                # width_pixels  = width / x_units_per_pixel
                # height_pixels = scaled_front_height / y_units_per_pixel
                # Für unverzerrte Darstellung gilt:
                #   width_pixels / height_pixels = width / front_height
                # => scaled_front_height = front_height * (y_units_per_pixel / x_units_per_pixel)
                scale_factor = y_units_per_pixel / x_units_per_pixel if x_units_per_pixel > 0 else 1.0
                scaled_front_height = front_height * scale_factor
                
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
                    'location': 'PlotStacks2Beamsteering.py:127',
                    'message': 'Beamsteering: Rechteck erstellt',
                    'data': {
                        'x': float(x),
                        'y': float(y),
                        'width': float(width),
                        'scaled_front_height': float(scaled_front_height),
                        'front_height_original': float(front_height)
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion
            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
    
