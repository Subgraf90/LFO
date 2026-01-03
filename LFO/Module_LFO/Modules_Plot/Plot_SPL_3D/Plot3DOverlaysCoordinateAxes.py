"""Koordinatenachsen Overlay-Rendering für den 3D-SPL-Plot.

Zeichnet die x-, y-, z-Achsen vom Zeichnungsnullpunkt (Ursprung) aus.
- x-Achse: rot
- y-Achse: grün
- z-Achse: blau
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysBase import SPL3DOverlayBase


class SPL3DOverlayCoordinateAxes(SPL3DOverlayBase):
    """
    Mixin-Klasse für Koordinatenachsen Overlay-Rendering.
    
    Zeichnet die x-, y-, z-Achsen vom Ursprung (0, 0, 0) aus.
    """
    
    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert das Coordinate Axes Overlay."""
        super().__init__(plotter, pv_module)
        self._overlay_prefix = "coord_axes_"
        self._last_axes_state: Optional[tuple] = None
    
    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um State zurückzusetzen."""
        super().clear_category(category)
        if category == 'coordinate_axes':
            self._last_axes_state = None
    
    def draw_coordinate_axes(self, settings, enabled: bool = True) -> None:
        """Zeichnet die x-, y-, z-Achsen vom Ursprung aus.
        
        Args:
            settings: Settings-Objekt
            enabled: Wenn False, werden die Achsen nicht gezeichnet
        """
        if not enabled:
            self.clear_category('coordinate_axes')
            self._last_axes_state = None
            return
        
        # Verwende eine sehr große Achsenlänge, damit die Achsen praktisch "unendlich" sind
        # und immer sichtbar bleiben, unabhängig vom Zoom-Level
        # 10000 Meter sollte für praktisch alle Anwendungsfälle ausreichen
        axis_length = 10000.0
        
        # Erstelle Signatur für State-Vergleich
        state = (
            float(axis_length),
            bool(enabled),
        )
        
        # Prüfe ob sich der State geändert hat
        existing_names = self._category_actors.get('coordinate_axes', [])
        if self._last_axes_state is not None and self._last_axes_state == state and existing_names:
            # Zusätzliche Prüfung: Wenn Actors fehlen, trotzdem neu zeichnen
            if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer'):
                renderer = self.plotter.renderer
                if renderer and existing_names:
                    # Prüfe ob alle Actors noch im Renderer existieren
                    actors_exist = all(name in renderer.actors for name in existing_names)
                    if actors_exist:
                        return
        
        # Wenn wir hier ankommen, müssen wir neu zeichnen
        self.clear_category('coordinate_axes')
        
        axis_length_from_state, _ = state
        
        # Zeichne die drei Achsen vom Ursprung (0, 0, 0) aus in beide Richtungen
        origin = np.array([0.0, 0.0, 0.0])
        
        # X-Achse: rot, in beide Richtungen (negative Richtung dünner)
        x_axis_positive = np.array([axis_length_from_state, 0.0, 0.0])
        x_axis_negative = np.array([-axis_length_from_state, 0.0, 0.0])
        self._draw_axis_line(origin, x_axis_positive, color='red', axis_name='x', line_width=2.0)
        self._draw_axis_line(origin, x_axis_negative, color='red', axis_name='x', line_width=1.0)
        
        # Y-Achse: grün, in beide Richtungen (negative Richtung dünner)
        y_axis_positive = np.array([0.0, axis_length_from_state, 0.0])
        y_axis_negative = np.array([0.0, -axis_length_from_state, 0.0])
        self._draw_axis_line(origin, y_axis_positive, color='green', axis_name='y', line_width=2.0)
        self._draw_axis_line(origin, y_axis_negative, color='green', axis_name='y', line_width=1.0)
        
        # Z-Achse: blau, in beide Richtungen (negative Richtung dünner)
        z_axis_positive = np.array([0.0, 0.0, axis_length_from_state])
        z_axis_negative = np.array([0.0, 0.0, -axis_length_from_state])
        self._draw_axis_line(origin, z_axis_positive, color='blue', axis_name='z', line_width=2.0)
        self._draw_axis_line(origin, z_axis_negative, color='blue', axis_name='z', line_width=1.0)
        
        self._last_axes_state = state
    
    def _draw_axis_line(self, start_point: np.ndarray, end_point: np.ndarray, color: str, axis_name: str, line_width: float = 2.0) -> None:
        """Zeichnet eine einzelne Achsenlinie.
        
        Args:
            start_point: Startpunkt der Linie (3D-Koordinaten)
            end_point: Endpunkt der Linie (3D-Koordinaten)
            color: Farbe der Linie ('red', 'green', 'blue')
            axis_name: Name der Achse ('x', 'y', 'z')
            line_width: Basis-Linienbreite (Standard: 2.5)
        """
        try:
            # Erstelle Punkte für die Linie mit Z-Offset
            # Füge kleine Z-Offset hinzu, damit die Achsen über dem SPL-Plot liegen
            # Verwende den gleichen Offset wie bei anderen Achsenlinien
            start_with_offset = start_point.copy()
            end_with_offset = end_point.copy()
            start_with_offset[2] += self._axis_z_offset
            end_with_offset[2] += self._axis_z_offset
            
            points = np.array([start_with_offset, end_with_offset])
            
            # Erstelle Polyline
            n_pts = len(points)
            lines = np.array([n_pts] + list(range(n_pts)), dtype=np.int64)
            polyline = self.pv.PolyData(points)
            polyline.lines = lines
            
            # 🎯 WICHTIG: Entferne explizit alle Vertices, damit keine Punkte an Segmentenden gerendert werden
            try:
                polyline.verts = np.empty(0, dtype=np.int64)
            except Exception:
                try:
                    polyline.verts = []
                except Exception:
                    pass
            
            # Nur DPI-Skalierung, kein Zoom-Faktor (wie bei anderen Achsenlinien)
            scaled_line_width = self._get_scaled_line_width(line_width, apply_zoom=False)
            
            # Füge die Linie als Overlay hinzu
            self._add_overlay_mesh(
                polyline,
                color=color,
                line_width=scaled_line_width,
                category='coordinate_axes',
                render_lines_as_tubes=False,
            )
        except Exception:  # noqa: BLE001
            # Bei Fehler einfach überspringen
            pass


__all__ = ['SPL3DOverlayCoordinateAxes']

