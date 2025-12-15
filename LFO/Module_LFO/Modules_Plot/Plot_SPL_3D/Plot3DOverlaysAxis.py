"""Axis Lines Overlay-Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysBase import SPL3DOverlayBase

DEBUG_OVERLAY_PERF = bool(int(os.environ.get("LFO_DEBUG_OVERLAY_PERF", "1")))


class SPL3DOverlayAxis(SPL3DOverlayBase):
    """
    Mixin-Klasse für Axis Lines Overlay-Rendering.
    
    Zeichnet X- und Y-Achsenlinien als Strich-Punkt-Linien auf aktiven Surfaces
    sowie halbtransparente 3D-Achsenflächen.
    """
    
    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert das Axis Overlay."""
        super().__init__(plotter, pv_module)
        self._overlay_prefix = "axis_"
        self._last_axis_state: Optional[tuple] = None
    
    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um State zurückzusetzen."""
        super().clear_category(category)
        if category == 'axis':
            self._last_axis_state = None
    
    def draw_axis_lines(self, settings, selected_axis: Optional[str] = None) -> None:
        """Zeichnet X- und Y-Achsenlinien als Strich-Punkt-Linien auf aktiven Surfaces.
        
        Args:
            settings: Settings-Objekt
            selected_axis: 'x' oder 'y' für ausgewählte Achse (wird rot gezeichnet), None wenn keine ausgewählt
        """
        t_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Berechne maximale Surface-Dimension für Achsenflächen-Größe
        max_surface_dim = self._get_max_surface_dimension(settings)
        
        # Erstelle Signatur der aktiven Surface-Punkte, damit Änderungen erkannt werden
        active_surfaces = self._get_active_xy_surfaces(settings)
        surface_points_signature = []
        for surface_id, surface in active_surfaces:
            if isinstance(surface, SurfaceDefinition):
                points = surface.points
            else:
                points = surface.get('points', [])
            
            # Erstelle Signatur aus Surface-Punkten
            points_tuple = []
            for point in points:
                try:
                    x = float(point.get('x', 0.0))
                    y = float(point.get('y', 0.0))
                    z = float(point.get('z', 0.0)) if point.get('z') is not None else 0.0
                    points_tuple.append((round(x, 6), round(y, 6), round(z, 6)))
                except (ValueError, TypeError, AttributeError):
                    continue
            
            surface_points_signature.append((str(surface_id), tuple(points_tuple)))
        
        # Sortiere nach surface_id für konsistente Reihenfolge
        surface_points_signature.sort(key=lambda x: x[0])
        
        state = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
            float(getattr(settings, 'axis_3d_transparency', 10.0)),
            float(max_surface_dim),
            selected_axis,
            tuple(surface_points_signature),
        )

        # 🐞 DEBUG: Protokolliere, wann und mit welchen Parametern die Achsen gezeichnet werden
        try:
            print(
                "[SPL3DOverlayAxis] draw_axis_lines call – "
                f"pos_x={state[0]}, pos_y={state[1]}, "
                f"len={state[2]}, width={state[3]}, "
                f"axis_3d_transparency={state[4]}, "
                f"max_surface_dim={state[5]}, "
                f"selected_axis={selected_axis}, "
                f"num_active_surfaces={len(active_surfaces)}"
            )
        except Exception:
            pass
        existing_names = self._category_actors.get('axis', [])
        # 🎯 WICHTIG: Beim Laden (_last_axis_state ist None) immer neu zeichnen
        # Prüfe auch, ob Actors fehlen (z. B. nach File-Reload)
        if self._last_axis_state is not None and self._last_axis_state == state and existing_names:
            # Zusätzliche Prüfung: Wenn Actors fehlen, trotzdem neu zeichnen
            if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer'):
                renderer = self.plotter.renderer
                if renderer and existing_names:
                    # Prüfe ob alle Actors noch im Renderer existieren
                    actors_exist = all(name in renderer.actors for name in existing_names)
                    if actors_exist:
                        return
        # Wenn wir hier ankommen, müssen wir neu zeichnen (State geändert, _last_axis_state ist None, oder Actors fehlen)

        t_clear_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        # Linien und evtl. Achsenflächen leeren
        self.clear_category('axis')
        self.clear_category('axis_plane')
        t_clear_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        x_axis, y_axis, length, width, axis_3d_transparency_from_state, max_surface_dim_from_state, selected_axis_from_state, _ = state

        # Strich-/Punkt-Pattern für OpenGL-Stipple (line_pattern, wird über die GPU wiederholt)
        # 0xF0F0 = längere Striche und Lücken, besser sichtbar als 0xF4F4
        dash_dot_pattern = 0xF0F0

        # ---- X-Achsenlinie (y = y_axis, konstant) pro Surface als gestrichelte Linie ----
        t_x_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_xz(y_axis, surface, settings)
            if intersection_points is None:
                continue
            x_coords, z_coords = intersection_points
            if len(x_coords) < 2:
                continue
            pts = np.column_stack([x_coords, np.full_like(x_coords, y_axis), z_coords])
            # Sortiere nach X-Koordinaten für konsistente Reihenfolge
            sort_idx = np.argsort(pts[:, 0])
            pts = pts[sort_idx]

            # Kleiner Z-Offset, damit die Linie beim Picking bevorzugt wird
            pts[:, 2] += self._axis_z_offset

            # Sanftes Ausdünnen pro Surface
            simplified: List[np.ndarray] = []
            eps = 1e-4
            last_pt = None
            for pt in pts:
                if last_pt is None or np.linalg.norm(pt - last_pt) > eps:
                    simplified.append(pt)
                    last_pt = pt
            if len(simplified) >= 2:
                simplified_points = np.vstack(simplified)
                # Eine Polyline pro Surface, Dash-Muster über line_pattern (wie Surface-Kanten)
                n_pts = len(simplified_points)
                lines = np.array([n_pts] + list(range(n_pts)), dtype=np.int64)
                polyline = self.pv.PolyData(simplified_points)
                polyline.lines = lines

                line_color = 'red' if selected_axis == 'x' else 'black'
                base_line_width = 2.0 if selected_axis == 'x' else 1.5
                # Wie bei Surface-Kanten: nur DPI-Skalierung, kein Zoom-Faktor
                line_width = self._get_scaled_line_width(base_line_width, apply_zoom=False)
                self._add_overlay_mesh(
                    polyline,
                    color=line_color,
                    line_width=line_width,
                    line_pattern=dash_dot_pattern,
                    line_repeat=1,
                    category='axis',
                    render_lines_as_tubes=False,
                )
        t_x_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None

        # ---- Y-Achsenlinie (x = x_axis, konstant) pro Surface als gestrichelte Linie ----
        t_y_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_yz(x_axis, surface, settings)
            if intersection_points is None:
                continue
            y_coords, z_coords = intersection_points
            if len(y_coords) < 2:
                continue
            pts = np.column_stack([np.full_like(y_coords, x_axis), y_coords, z_coords])

            sort_idx = np.argsort(pts[:, 1])
            pts = pts[sort_idx]

            pts[:, 2] += self._axis_z_offset

            simplified: List[np.ndarray] = []
            eps = 1e-4
            last_pt = None
            for pt in pts:
                if last_pt is None or np.linalg.norm(pt - last_pt) > eps:
                    simplified.append(pt)
                    last_pt = pt
            if len(simplified) >= 2:
                simplified_points = np.vstack(simplified)
                n_pts = len(simplified_points)
                lines = np.array([n_pts] + list(range(n_pts)), dtype=np.int64)
                polyline = self.pv.PolyData(simplified_points)
                polyline.lines = lines

                line_color = 'red' if selected_axis == 'y' else 'black'
                base_line_width = 2.0 if selected_axis == 'y' else 1.5
                line_width = self._get_scaled_line_width(base_line_width, apply_zoom=False)
                self._add_overlay_mesh(
                    polyline,
                    color=line_color,
                    line_width=line_width,
                    line_pattern=dash_dot_pattern,
                    line_repeat=1,
                    category='axis',
                    render_lines_as_tubes=False,
                )
        t_y_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        t_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # 3D-Achsenfläche immer zeichnen
        try:
            self._draw_axis_planes(x_axis, y_axis, length, width, settings)
        except Exception:  # noqa: BLE001
            pass

        self._last_axis_state = state

    def _draw_axis_planes(self, x_axis: float, y_axis: float, length: float, width: float, settings) -> None:
        """Zeichnet halbtransparente, quadratische Flächen durch die X- und Y-Achse."""
        # Berechne Größe basierend auf allen nicht-versteckten Surfaces
        L_base = self._get_max_surface_dimension(settings)
        if L_base <= 0.0:
            return
        
        # Faktor 3 der max Dimension: L_base ist bereits max_dim * 1.5, also L = L_base * 2 = max_dim * 3
        L = L_base * 2.0
        # Vertikale Höhe soll kürzer sein: Faktor 0.75 der Länge
        H = L * 0.75

        # Transparenz: Wert in Prozent (0–100), 10 % Standard
        transparency_pct = float(getattr(settings, "axis_3d_transparency", 10.0))
        transparency_pct = max(0.0, min(100.0, transparency_pct))
        opacity = max(0.0, min(1.0, transparency_pct / 100.0))

        # Diskrete Auflösung der Fläche
        n = 2

        # X-Achsen-Ebene: X-Z-Fläche bei y = y_axis
        try:
            x_vals = np.linspace(-L / 2.0, L / 2.0, n)
            z_vals = np.linspace(-H / 2.0, H / 2.0, n)
            X, Z = np.meshgrid(x_vals, z_vals)
            Y = np.full_like(X, y_axis)
            grid_x = self.pv.StructuredGrid(X, Y, Z)
            points = grid_x.points
            points[:, 2] += self._planar_z_offset
            grid_x.points = points
            self._add_overlay_mesh(
                grid_x,
                color="gray",
                opacity=opacity,
                category="axis_plane",
            )
        except Exception:  # noqa: BLE001
            pass

        # Y-Achsen-Ebene: Y-Z-Fläche bei x = x_axis
        try:
            y_vals = np.linspace(-L / 2.0, L / 2.0, n)
            z_vals = np.linspace(-H / 2.0, H / 2.0, n)
            Y2, Z2 = np.meshgrid(y_vals, z_vals)
            X2 = np.full_like(Y2, x_axis)
            grid_y = self.pv.StructuredGrid(X2, Y2, Z2)
            points2 = grid_y.points
            points2[:, 2] += self._planar_z_offset
            grid_y.points = points2
            self._add_overlay_mesh(
                grid_y,
                color="gray",
                opacity=opacity,
                category="axis_plane",
            )
        except Exception:  # noqa: BLE001
            pass

    def _get_surface_intersection_points_xz(self, y_const: float, surface: Any, settings) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Berechnet Schnittpunkte der Linie y=y_const mit dem Surface-Polygon (Projektion auf XZ-Ebene)."""
        if isinstance(surface, SurfaceDefinition):
            points = surface.points
        else:
            points = surface.get('points', [])
        
        if len(points) < 3:
            return None
        
        # Extrahiere alle Koordinaten
        points_3d = []
        for p in points:
            x = float(p.get('x', 0.0))
            y = float(p.get('y', 0.0))
            z = float(p.get('z', 0.0)) if p.get('z') is not None else 0.0
            points_3d.append((x, y, z))
        
        # Prüfe jede Kante des Polygons auf Schnitt mit Linie y=y_const
        intersection_points = []
        n = len(points_3d)
        for i in range(n):
            p1 = points_3d[i]
            p2 = points_3d[(i + 1) % n]
            
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            
            if (y1 <= y_const <= y2) or (y2 <= y_const <= y1):
                if abs(y2 - y1) > 1e-10:
                    t = (y_const - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    z_intersect = z1 + t * (z2 - z1)
                    intersection_points.append((x_intersect, z_intersect))
        
        if len(intersection_points) < 2:
            return None
        
        # Entferne Duplikate
        unique_points = []
        eps = 1e-6
        for x, z in intersection_points:
            is_duplicate = False
            for ux, uz in unique_points:
                if abs(x - ux) < eps and abs(z - uz) < eps:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append((x, z))
        
        if len(unique_points) < 2:
            return None
        
        # Extrahiere X- und Z-Koordinaten
        x_coords = np.array([p[0] for p in unique_points])
        z_coords = np.array([p[1] for p in unique_points])
        
        # Sortiere nach X-Koordinaten
        sort_indices = np.argsort(x_coords)
        x_coords = x_coords[sort_indices]
        z_coords = z_coords[sort_indices]
        
        return (x_coords, z_coords)

    def _get_surface_intersection_points_yz(self, x_const: float, surface: Any, settings) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Berechnet Schnittpunkte der Linie x=x_const mit dem Surface-Polygon (Projektion auf YZ-Ebene)."""
        if isinstance(surface, SurfaceDefinition):
            points = surface.points
        else:
            points = surface.get('points', [])
        
        if len(points) < 3:
            return None
        
        # Extrahiere alle Koordinaten
        points_3d = []
        for p in points:
            x = float(p.get('x', 0.0))
            y = float(p.get('y', 0.0))
            z = float(p.get('z', 0.0)) if p.get('z') is not None else 0.0
            points_3d.append((x, y, z))
        
        # Prüfe jede Kante des Polygons auf Schnitt mit Linie x=x_const
        intersection_points = []
        n = len(points_3d)
        for i in range(n):
            p1 = points_3d[i]
            p2 = points_3d[(i + 1) % n]
            
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            
            if (x1 <= x_const <= x2) or (x2 <= x_const <= x1):
                if abs(x2 - x1) > 1e-10:
                    t = (x_const - x1) / (x2 - x1)
                    y_intersect = y1 + t * (y2 - y1)
                    z_intersect = z1 + t * (z2 - z1)
                    intersection_points.append((y_intersect, z_intersect))
        
        if len(intersection_points) < 2:
            return None
        
        # Entferne Duplikate
        unique_points = []
        eps = 1e-6
        for y, z in intersection_points:
            is_duplicate = False
            for uy, uz in unique_points:
                if abs(y - uy) < eps and abs(z - uz) < eps:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append((y, z))
        
        if len(unique_points) < 2:
            return None
        
        # Extrahiere Y- und Z-Koordinaten
        y_coords = np.array([p[0] for p in unique_points])
        z_coords = np.array([p[1] for p in unique_points])
        
        # Sortiere nach Y-Koordinaten
        sort_indices = np.argsort(y_coords)
        y_coords = y_coords[sort_indices]
        z_coords = z_coords[sort_indices]
        
        return (y_coords, z_coords)

    def _create_dash_dot_line_segments_optimized(self, points_3d: np.ndarray, dash_length: float = 1.0, dot_length: float = 0.2, gap_length: float = 0.3) -> List[np.ndarray]:
        """Optimierte Version: Teilt eine 3D-Linie in Strich-Punkt-Segmente auf."""
        if len(points_3d) < 2:
            return []
        
        if len(points_3d) == 2:
            total_length = np.linalg.norm(points_3d[1] - points_3d[0])
            if total_length <= dash_length:
                return [points_3d]
        
        # Berechne kumulative Distanzen entlang der Linie
        diffs = np.diff(points_3d, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_distances[-1]
        
        if total_length <= 0:
            return []
        
        # Pattern: Strich - Lücke - Punkt - Lücke - ...
        segments = []
        current_pos = 0.0
        
        while current_pos < total_length:
            # Strich-Segment
            dash_start = current_pos
            dash_end = min(current_pos + dash_length, total_length)
            if dash_end > dash_start:
                dash_points = self._interpolate_line_segment_simple(
                    points_3d,
                    cumulative_distances,
                    dash_start,
                    dash_end,
                )
                if len(dash_points) >= 2:
                    segments.append(dash_points)
            current_pos = dash_end + gap_length
            
            if current_pos >= total_length:
                break
            
            # Punkt-Segment (kurzer Strich)
            dot_start = current_pos
            dot_end = min(current_pos + dot_length, total_length)
            if dot_end > dot_start:
                dot_points = self._interpolate_line_segment_simple(
                    points_3d,
                    cumulative_distances,
                    dot_start,
                    dot_end,
                )
                if len(dot_points) >= 2:
                    segments.append(dot_points)
            current_pos = dot_end + gap_length
        
        return segments

    def _combine_line_segments(self, segments: List[np.ndarray]) -> Optional[Any]:
        """Kombiniert mehrere Liniensegmente in ein einziges PolyData-Mesh."""
        if not segments:
            return None
        
        try:
            all_points = []
            all_lines = []
            point_offset = 0
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                
                all_points.append(segment)
                
                n_pts = len(segment)
                line_array = [n_pts] + [point_offset + i for i in range(n_pts)]
                all_lines.extend(line_array)
                
                point_offset += n_pts
            
            if not all_points:
                return None
            
            combined_points = np.vstack(all_points)
            polyline = self.pv.PolyData(combined_points)
            polyline.lines = np.array(all_lines, dtype=np.int64)
            
            return polyline
        except Exception:  # noqa: BLE001
            return None

    def _interpolate_line_segment_simple(self, points_3d: np.ndarray, cumulative_distances: np.ndarray, start_dist: float, end_dist: float) -> np.ndarray:
        """Vereinfachte Interpolation: Verwendet nur Start- und Endpunkt für kurze Segmente."""
        if len(points_3d) < 2:
            return points_3d
        
        start_idx = np.searchsorted(cumulative_distances, start_dist, side='right') - 1
        end_idx = np.searchsorted(cumulative_distances, end_dist, side='right')
        
        start_idx = max(0, min(start_idx, len(points_3d) - 1))
        end_idx = max(0, min(end_idx, len(points_3d) - 1))
        
        if end_idx - start_idx <= 1:
            start_point = points_3d[start_idx]
            end_point = points_3d[min(end_idx, len(points_3d) - 1)]
            
            if start_dist > cumulative_distances[start_idx] and start_idx < len(points_3d) - 1:
                t = (start_dist - cumulative_distances[start_idx]) / (cumulative_distances[start_idx + 1] - cumulative_distances[start_idx])
                start_point = points_3d[start_idx] + t * (points_3d[start_idx + 1] - points_3d[start_idx])
            
            if end_dist < cumulative_distances[end_idx] and end_idx > 0:
                t = (end_dist - cumulative_distances[end_idx - 1]) / (cumulative_distances[end_idx] - cumulative_distances[end_idx - 1])
                end_point = points_3d[end_idx - 1] + t * (points_3d[end_idx] - points_3d[end_idx - 1])
            
            return np.array([start_point, end_point])
        
        segment_points = [points_3d[start_idx]]
        for i in range(start_idx + 1, end_idx):
            segment_points.append(points_3d[i])
        segment_points.append(points_3d[end_idx])
        
        return np.array(segment_points)


__all__ = ['SPL3DOverlayAxis']

