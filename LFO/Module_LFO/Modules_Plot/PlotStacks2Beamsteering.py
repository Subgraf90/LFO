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
            
            # Werte direkt aus Metadaten holen
            width = float(cabinet.get('width', 0))
            if width <= 0:
                return
            
            # Verwende front_height aus Metadaten
            front_height = float(cabinet.get('front_height', cabinet.get('height', 0)))
            if front_height <= 0:
                return
            
            # #region agent log - Ursprungsabmessungen
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Beamsteering.py:53',
                    'message': 'Beamsteering: Ursprungsabmessungen aus Cabinet-Daten',
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
                }, default=str) + '\n')
            # #endregion
            
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
            
            # #region agent log - Position vor Berechnung
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'PlotStacks2Beamsteering.py:94',
                    'message': 'Beamsteering: Position vor Berechnung',
                    'data': {
                        'isrc': int(isrc),
                        'base_x': float(base_x),
                        'base_y': float(base_y),
                        'x_offset': float(x_offset),
                        'x': float(x),
                        'y': float(y),
                        'ax_ylim': list(self.ax.get_ylim()) if hasattr(self.ax, 'get_ylim') else None
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
            # #endregion

            # Berechne Transform für unverzerrte Lautsprecher-Darstellung
            # Die Lautsprecher haben Dimensionen in Metern (width x height)
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
                }, default=str) + '\n')
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
            
            # KEINE SKALIERUNG MEHR: Die sekundäre Y-Achse (ax_speakers) hat die gleiche Skalierung wie die X-Achse
            # Daher verwenden wir direkt front_height ohne Skalierung
            scaled_front_height = front_height
            
            # #region agent log - Abmessungsberechnung ohne Skalierung (sekundäre Achse verwendet)
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Beamsteering.py:197',
                    'message': 'Beamsteering: Abmessungsberechnung - keine Skalierung (sekundäre Achse mit gleicher Skalierung wie X)',
                    'data': {
                        'isrc': int(isrc),
                        'breite': {
                            'original_m': float(width),
                            'verwendet_m': float(width),
                            'skaliert': False
                        },
                        'hoehe': {
                            'original_m': float(front_height),
                            'verwendet_m': float(scaled_front_height),
                            'skaliert': False,
                            'hinweis': 'Sekundäre Y-Achse hat gleiche Skalierung wie X-Achse, daher keine Skalierung nötig'
                        }
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
            # #endregion
            
            # #region agent log - Nach Skalierung, vor Rectangle
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'PlotStacks2Beamsteering.py:300',
                    'message': 'Beamsteering: Nach Skalierung, vor Rectangle-Zeichnung',
                    'data': {
                        'isrc': int(isrc),
                        'scaled_front_height': float(scaled_front_height),
                        'width': float(width),
                        'x': float(x),
                        'y': float(y)
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
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
            
            # #region agent log - Vor dem Zeichnen
            import json
            try:
                ax_ylim = list(self.ax.get_ylim()) if hasattr(self.ax, 'get_ylim') else None
                ax_xlim = list(self.ax.get_xlim()) if hasattr(self.ax, 'get_xlim') else None
                y_in_range = None
                if ax_ylim is not None:
                    y_in_range = bool(float(y) >= ax_ylim[0] and float(y + scaled_front_height) <= ax_ylim[1])
            except Exception as e:
                ax_ylim = None
                ax_xlim = None
                y_in_range = None
            
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'PlotStacks2Beamsteering.py:261',
                    'message': 'Beamsteering: Vor Rectangle-Zeichnung',
                    'data': {
                        'isrc': int(isrc),
                        'position': {'x': float(x), 'y': float(y)},
                        'dimensions': {'width': float(width), 'height': float(scaled_front_height)},
                        'ax_ylim': ax_ylim,
                        'ax_xlim': ax_xlim,
                        'y_in_range': bool(y_in_range) if y_in_range is not None else None
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
            # #endregion
            
            rect = patches.Rectangle(
                (x, y),
                width, 
                scaled_front_height, 
                ec='black',
                fc=color
            )
            self.ax.add_patch(rect)
            
            # #region agent log - Nach dem Zeichnen
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'PlotStacks2Beamsteering.py:342',
                    'message': 'Beamsteering: Nach Rectangle-Zeichnung',
                    'data': {
                        'isrc': int(isrc),
                        'patches_count': len(self.ax.patches),
                        'rect_added': True
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
            # #endregion
            
            # #region agent log - Finale Abmessungen beim Zeichnen
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'DIMENSIONS',
                    'location': 'PlotStacks2Beamsteering.py:232',
                    'message': 'Beamsteering: Rechteck gezeichnet - verwendete Abmessungen',
                    'data': {
                        'isrc': int(isrc),
                        'position': {
                            'x_m': float(x),
                            'y_m': float(y),
                            'oberkante_y_m': float(y + scaled_front_height)
                        },
                        'abmessungen_original': {
                            'breite_m': float(width),
                            'hoehe_m': float(front_height),
                            'aspect_ratio': float(aspect_ratio_real)
                        },
                        'abmessungen_verwendet': {
                            'breite_m': float(width),
                            'hoehe_m': float(scaled_front_height),
                            'aspect_ratio': float(aspect_ratio_pixels)
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
                            'hoehe_verwendet_m': float(scaled_front_height),
                            'keine_skalierung': bool(True),
                            'hinweis': 'Höhe wird nicht skaliert, bleibt in Metern'
                        },
                        'vergleich': {
                            'breite_original_vs_verwendet': bool(float(width) == float(width)),
                            'hoehe_original_vs_verwendet': bool(float(front_height) == float(scaled_front_height)),
                            'aspect_ratio_original': float(aspect_ratio_real),
                            'aspect_ratio_pixel': float(aspect_ratio_pixels),
                            'hinweis': 'Höhe wird nicht skaliert, daher kann Aspect-Ratio im Pixel-Bereich abweichen'
                        }
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }, default=str) + '\n')
            # #endregion
            
        except Exception as e:
            print(f"Fehler beim Zeichnen eines Lautsprechers: {str(e)}")
    
