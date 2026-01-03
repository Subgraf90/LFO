"""Axis Lines Overlay-Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

import json
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
        # 🎯 NEU: Verwende _get_active_xy_surfaces_for_axis_lines() statt _get_active_xy_surfaces()
        # Damit werden Axis-Linien auch auf disabled Surfaces gezeichnet, wenn xy_enabled=True
        active_surfaces = self._get_active_xy_surfaces_for_axis_lines(settings)
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
            pass  # Debug-Ausgabe entfernt
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
        print(f"[draw_axis_lines] BEFORE clear_category: axis actors={self._category_actors.get('axis', [])}, axis_plane actors={self._category_actors.get('axis_plane', [])}")
        # Prüfe ob Actors noch im Plotter existieren, bevor wir sie entfernen
        if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
            axis_actors_in_plotter = [name for name in self._category_actors.get('axis', []) if name in self.plotter.renderer.actors]
            if axis_actors_in_plotter:
                print(f"[draw_axis_lines] WARNING: Found {len(axis_actors_in_plotter)} axis actors still in plotter before clear: {axis_actors_in_plotter}")
            # Zähle ALLE Actors im Renderer, die mit 'axis' beginnen (auch solche, die nicht in _category_actors sind)
            all_axis_like_actors = [name for name in self.plotter.renderer.actors.keys() if isinstance(name, str) and name.startswith('axis_')]
            if all_axis_like_actors:
                print(f"[draw_axis_lines] Found {len(all_axis_like_actors)} actors with 'axis_' prefix in plotter: {all_axis_like_actors}")
        self.clear_category('axis')
        self.clear_category('axis_plane')
        print(f"[draw_axis_lines] AFTER clear_category: axis actors={self._category_actors.get('axis', [])}, axis_plane actors={self._category_actors.get('axis_plane', [])}")
        # Prüfe ob Actors wirklich entfernt wurden
        if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
            remaining_axis_actors = [name for name in ['axis_x_batch', 'axis_y_batch'] if name in self.plotter.renderer.actors]
            if remaining_axis_actors:
                print(f"[draw_axis_lines] ERROR: Found {len(remaining_axis_actors)} axis actors still in plotter after clear: {remaining_axis_actors}")
            # Zähle ALLE verbleibenden Actors mit 'axis_' Präfix
            remaining_axis_like_actors = [name for name in self.plotter.renderer.actors.keys() if isinstance(name, str) and name.startswith('axis_')]
            if remaining_axis_like_actors:
                print(f"[draw_axis_lines] WARNING: Found {len(remaining_axis_like_actors)} actors with 'axis_' prefix still in plotter after clear: {remaining_axis_like_actors}")
        t_clear_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        x_axis, y_axis, length, width, axis_3d_transparency_from_state, max_surface_dim_from_state, selected_axis_from_state, _ = state

        # Strich-/Punkt-Pattern für OpenGL-Stipple (line_pattern, wird über die GPU wiederholt)
        # 0xF0F0 = längere Striche und Lücken, besser sichtbar als 0xF4F4
        dash_dot_pattern = 0xF0F0

        # 🎯 OPTIMIERUNG: Batch-Rendering - sammle alle Segmente und rendere in einem Mesh
        # ---- X-Achsenlinie (y = y_axis, konstant) pro Surface als gestrichelte Linie ----
        t_x_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        x_segments_all = []
        x_line_color = 'red' if selected_axis == 'x' else 'black'
        x_base_line_width = 2.0 if selected_axis == 'x' else 1.5
        x_line_width = self._get_scaled_line_width(x_base_line_width, apply_zoom=False)
        print(f"[Axis Lines] X-Achse: base_width={x_base_line_width:.2f}, scaled_width={x_line_width:.2f}, color={x_line_color}, selected={selected_axis}")
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "line-width-check",
                    "hypothesisId": "LINE_WIDTH_X",
                    "location": "Plot3DOverlaysAxis.draw_axis_lines",
                    "message": "X-Achse line_width berechnet",
                    "data": {
                        "selected_axis": selected_axis,
                        "x_base_line_width": float(x_base_line_width),
                        "x_line_width": float(x_line_width),
                        "x_line_color": x_line_color
                    },
                    "timestamp": int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Sammle alle X- und Z-Koordinaten der Schnittpunkte für Flächenausdehnung
        all_x_coords = []
        all_z_coords_from_x = []
        
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_xz(y_axis, surface, settings)
            if intersection_points is not None:
                x_coords, z_coords = intersection_points
                all_x_coords.extend(x_coords)
                all_z_coords_from_x.extend(z_coords)
            segments = self._prepare_axis_line_segments(
                intersection_points,
                y_axis,
                'x',
                sort_dim=0,  # Sortiere nach X
                const_dim=1,  # Y ist konstant
            )
            if segments:
                x_segments_all.extend(segments)
        
        # Rendere alle X-Achsen-Segmente in einem Batch
        if x_segments_all:
            self._render_batched_segments(x_segments_all, x_line_color, x_line_width, 'axis_x_batch')
        t_x_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None

        # ---- Y-Achsenlinie (x = x_axis, konstant) pro Surface als gestrichelte Linie ----
        t_y_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        y_segments_all = []
        y_line_color = 'red' if selected_axis == 'y' else 'black'
        y_base_line_width = 2.0 if selected_axis == 'y' else 1.5
        y_line_width = self._get_scaled_line_width(y_base_line_width, apply_zoom=False)
        print(f"[Axis Lines] Y-Achse: base_width={y_base_line_width:.2f}, scaled_width={y_line_width:.2f}, color={y_line_color}, selected={selected_axis}")
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "line-width-check",
                    "hypothesisId": "LINE_WIDTH_Y",
                    "location": "Plot3DOverlaysAxis.draw_axis_lines",
                    "message": "Y-Achse line_width berechnet",
                    "data": {
                        "selected_axis": selected_axis,
                        "y_base_line_width": float(y_base_line_width),
                        "y_line_width": float(y_line_width),
                        "y_line_color": y_line_color
                    },
                    "timestamp": int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Sammle alle Y- und Z-Koordinaten der Schnittpunkte für Flächenausdehnung
        all_y_coords = []
        all_z_coords_from_y = []
        
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_yz(x_axis, surface, settings)
            if intersection_points is not None:
                y_coords, z_coords = intersection_points
                all_y_coords.extend(y_coords)
                all_z_coords_from_y.extend(z_coords)
            segments = self._prepare_axis_line_segments(
                intersection_points,
                x_axis,
                'y',
                sort_dim=1,  # Sortiere nach Y
                const_dim=0,  # X ist konstant
            )
            if segments:
                y_segments_all.extend(segments)
        
        # Rendere alle Y-Achsen-Segmente in einem Batch
        if y_segments_all:
            self._render_batched_segments(y_segments_all, y_line_color, y_line_width, 'axis_y_batch')
        t_y_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        t_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Berechne Flächenausdehnung basierend auf äußersten Schnittpunkten
        # Kombiniere alle X-, Y- und Z-Koordinaten
        all_z_coords = all_z_coords_from_x + all_z_coords_from_y
        
        # Berechne äußerste Punkte (Min/Max-Werte) für X-Achsen-Ebene
        if all_x_coords:
            min_x = float(np.min(all_x_coords))
            max_x = float(np.max(all_x_coords))
            # 20% Puffer hinzufügen
            x_range = max_x - min_x
            min_x_with_buffer = min_x - x_range * 0.2
            max_x_with_buffer = max_x + x_range * 0.2
        else:
            min_x_with_buffer = None
            max_x_with_buffer = None
        
        # Berechne äußerste Punkte (Min/Max-Werte) für Y-Achsen-Ebene
        if all_y_coords:
            min_y = float(np.min(all_y_coords))
            max_y = float(np.max(all_y_coords))
            # 20% Puffer hinzufügen
            y_range = max_y - min_y
            min_y_with_buffer = min_y - y_range * 0.2
            max_y_with_buffer = max_y + y_range * 0.2
        else:
            min_y_with_buffer = None
            max_y_with_buffer = None
        
        # Berechne Z-Ausdehnung: min_z oder 0 als unterer Punkt, max_z als oberer Punkt
        if all_z_coords:
            min_z = float(np.min(all_z_coords))
            max_z = float(np.max(all_z_coords))
            # Unterer Punkt: min_z oder 0, dann 20% Puffer
            z_bottom = min(min_z, 0.0)
            z_range = max_z - z_bottom
            min_z_with_buffer = z_bottom - z_range * 0.2
            max_z_with_buffer = max_z + z_range * 0.2
        else:
            min_z_with_buffer = None
            max_z_with_buffer = None
        
        # 3D-Achsenfläche immer zeichnen
        try:
            self._draw_axis_planes(
                x_axis, y_axis, 
                min_x_with_buffer, max_x_with_buffer,  # X-Bereich für X-Achsen-Ebene
                min_y_with_buffer, max_y_with_buffer,  # Y-Bereich für Y-Achsen-Ebene
                min_z_with_buffer, max_z_with_buffer,  # Z-Bereich für beide Ebenen
                settings
            )
        except Exception:  # noqa: BLE001
            pass

        self._last_axis_state = state

    def _draw_axis_planes(
        self, 
        x_axis: float, 
        y_axis: float, 
        min_x: Optional[float], 
        max_x: Optional[float],
        min_y: Optional[float], 
        max_y: Optional[float],
        min_z: Optional[float], 
        max_z: Optional[float],
        settings
    ) -> None:
        """Zeichnet halbtransparente Flächen durch die X- und Y-Achse.
        
        Args:
            x_axis: X-Position der Y-Achsen-Ebene
            y_axis: Y-Position der X-Achsen-Ebene
            min_x, max_x: X-Bereich für X-Achsen-Ebene (kann None sein)
            min_y, max_y: Y-Bereich für Y-Achsen-Ebene (kann None sein)
            min_z, max_z: Z-Bereich für beide Ebenen (kann None sein)
            settings: Settings-Objekt
        """
        # Transparenz: Wert in Prozent (0–100), 10 % Standard
        transparency_pct = float(getattr(settings, "axis_3d_transparency", 10.0))
        transparency_pct = max(0.0, min(100.0, transparency_pct))
        opacity = max(0.0, min(1.0, transparency_pct / 100.0))

        # Diskrete Auflösung der Fläche
        n = 2

        # X-Achsen-Ebene: X-Z-Fläche bei y = y_axis
        # Geht von min_x bis max_x (äußerste Schnittpunkte, keine Symmetrie)
        if min_x is not None and max_x is not None and min_z is not None and max_z is not None:
            try:
                x_vals = np.linspace(min_x, max_x, n)
                z_vals = np.linspace(min_z, max_z, n)
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
        # Geht von min_y bis max_y (äußerste Schnittpunkte, keine Symmetrie)
        if min_y is not None and max_y is not None and min_z is not None and max_z is not None:
            try:
                y_vals = np.linspace(min_y, max_y, n)
                z_vals = np.linspace(min_z, max_z, n)
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

    def _create_dash_dot_segments(self, points: np.ndarray, dash_length: float = 0.3, dot_length: float = 0.1, gap_length: float = 0.15) -> List[np.ndarray]:
        """Teilt eine Linie in Dash-Dot-Segmente auf (Strich-Punkt-Muster).
        
        Optimiert für Performance: Berechnet kumulative Distanzen nur einmal.
        
        Args:
            points: Array von 3D-Punkten der Linie
            dash_length: Länge eines Strich-Segments
            dot_length: Länge eines Punkt-Segments (kurzer Strich)
            gap_length: Länge der Lücke zwischen Segmenten
            
        Returns:
            Liste von Punkt-Arrays für jedes Segment
        """
        if len(points) < 2:
            return []
        
        # Berechne kumulative Distanzen nur einmal
        if len(points) == 2:
            total_length = np.linalg.norm(points[1] - points[0])
            if total_length <= dash_length:
                return [points]
            # Für 2 Punkte: Erstelle cumulative_distances direkt
            cumulative_distances = np.array([0.0, total_length])
            total_length = cumulative_distances[-1]
        else:
            diffs = np.diff(points, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
            total_length = cumulative_distances[-1]
        
        if total_length <= 0:
            return []
        
        # 🎯 OPTIMIERUNG: Berechne Anzahl der Segmente vorab, um Liste zu pre-allokieren
        # Pattern: Strich - Lücke - Punkt - Lücke
        pattern_length = dash_length + gap_length + dot_length + gap_length
        estimated_segments = int(total_length / pattern_length) * 2 + 2  # +2 für Sicherheit
        segments = []
        segments.reserve(estimated_segments) if hasattr(segments, 'reserve') else None  # Python-Listen haben kein reserve
        
        current_pos = 0.0
        
        # Pattern: Strich - Lücke - Punkt - Lücke - Strich - ...
        while current_pos < total_length:
            # Strich-Segment
            dash_end = min(current_pos + dash_length, total_length)
            if dash_end > current_pos:
                dash_seg = self._get_line_segment_at_distance(points, cumulative_distances, current_pos, dash_end)
                if dash_seg is not None and len(dash_seg) >= 2:
                    segments.append(dash_seg)
            current_pos = dash_end + gap_length
            
            if current_pos >= total_length:
                break
            
            # Punkt-Segment (kurzer Strich)
            dot_end = min(current_pos + dot_length, total_length)
            if dot_end > current_pos:
                dot_seg = self._get_line_segment_at_distance(points, cumulative_distances, current_pos, dot_end)
                if dot_seg is not None and len(dot_seg) >= 2:
                    segments.append(dot_seg)
            current_pos = dot_end + gap_length
        
        return segments

    def _get_line_segment_at_distance(self, points: np.ndarray, cumulative_distances: Optional[np.ndarray], start_dist: float, end_dist: float) -> Optional[np.ndarray]:
        """Gibt die Punkte der Linie zwischen start_dist und end_dist zurück.
        
        Optimiert für Performance: Minimiert Array-Erstellungen und verwendet vectorisierte Operationen.
        """
        if len(points) < 2:
            return None
        
        if len(points) == 2 or cumulative_distances is None:
            # Einfacher Fall: nur 2 Punkte - sehr schnell
            total_length = np.linalg.norm(points[1] - points[0])
            if total_length <= 0:
                return None
            t_start = np.clip(start_dist / total_length, 0.0, 1.0)
            t_end = np.clip(end_dist / total_length, 0.0, 1.0)
            # Vectorisierte Interpolation
            start_pt = points[0] + t_start * (points[1] - points[0])
            end_pt = points[0] + t_end * (points[1] - points[0])
            return np.array([start_pt, end_pt])
        
        # Komplexerer Fall: mehrere Punkte
        start_idx = np.searchsorted(cumulative_distances, start_dist, side='right') - 1
        end_idx = np.searchsorted(cumulative_distances, end_dist, side='right')
        
        start_idx = max(0, min(start_idx, len(points) - 1))
        end_idx = max(0, min(end_idx, len(points) - 1))
        
        if end_idx <= start_idx:
            # Interpoliere zwischen zwei Punkten - optimiert
            if start_idx < len(cumulative_distances) - 1:
                t = (start_dist - cumulative_distances[start_idx]) / (cumulative_distances[start_idx + 1] - cumulative_distances[start_idx])
                start_pt = points[start_idx] + t * (points[start_idx + 1] - points[start_idx])
            else:
                start_pt = points[start_idx]
            
            if end_idx < len(cumulative_distances) - 1 and end_idx > 0:
                t = (end_dist - cumulative_distances[end_idx - 1]) / (cumulative_distances[end_idx] - cumulative_distances[end_idx - 1])
                end_pt = points[end_idx - 1] + t * (points[end_idx] - points[end_idx - 1])
            else:
                end_pt = points[min(end_idx, len(points) - 1)]
            
            return np.array([start_pt, end_pt])
        
        # Mehrere Punkte im Segment - optimiert mit Pre-Allokation
        n_mid_points = end_idx - start_idx - 1
        segment_points = np.empty((2 + n_mid_points, 3), dtype=points.dtype)
        seg_idx = 0
        
        # Startpunkt interpolieren
        if start_dist > cumulative_distances[start_idx] and start_idx < len(points) - 1:
            t = (start_dist - cumulative_distances[start_idx]) / (cumulative_distances[start_idx + 1] - cumulative_distances[start_idx])
            segment_points[seg_idx] = points[start_idx] + t * (points[start_idx + 1] - points[start_idx])
        else:
            segment_points[seg_idx] = points[start_idx]
        seg_idx += 1
        
        # Mittlere Punkte hinzufügen (vectorisiert)
        if n_mid_points > 0:
            segment_points[seg_idx:seg_idx + n_mid_points] = points[start_idx + 1:end_idx]
            seg_idx += n_mid_points
        
        # Endpunkt interpolieren
        if end_dist < cumulative_distances[end_idx] and end_idx > 0:
            t = (end_dist - cumulative_distances[end_idx - 1]) / (cumulative_distances[end_idx] - cumulative_distances[end_idx - 1])
            segment_points[seg_idx] = points[end_idx - 1] + t * (points[end_idx] - points[end_idx - 1])
        else:
            segment_points[seg_idx] = points[min(end_idx, len(points) - 1)]
        
        return segment_points

    def _prepare_axis_line_segments(
        self,
        intersection_points: Optional[Tuple[np.ndarray, np.ndarray]],
        const_coord: float,
        axis_type: str,
        sort_dim: int,
        const_dim: int,
    ) -> List[np.ndarray]:
        """Bereitet Dash-Dot-Segmente für eine Achsenlinie vor (ohne Rendering).
        
        Returns:
            Liste von Segment-Arrays (jedes Segment ist ein np.ndarray von Punkten)
        """
        if intersection_points is None:
            return []
        
        coords_1, z_coords = intersection_points
        if len(coords_1) < 2:
            return []
        
        # Erstelle 3D-Punkte: X-Achse: [x, y_axis, z], Y-Achse: [x_axis, y, z]
        if axis_type == 'x':
            pts = np.column_stack([coords_1, np.full_like(coords_1, const_coord), z_coords])
        else:  # axis_type == 'y'
            pts = np.column_stack([np.full_like(coords_1, const_coord), coords_1, z_coords])
        
        # Sortiere nach entsprechender Dimension
        sort_idx = np.argsort(pts[:, sort_dim])
        pts = pts[sort_idx]
        
        # Z-Offset hinzufügen, damit die Linie beim Picking bevorzugt wird
        pts[:, 2] += self._axis_z_offset
        
        # 🎯 OPTIMIERUNG: Vectorisierte Point-Simplification
        if len(pts) > 2:
            # Berechne Distanzen zwischen aufeinanderfolgenden Punkten
            diffs = np.diff(pts, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            # Verwende cumsum mit Mask für bessere Performance
            eps = 1e-4
            keep_mask = np.concatenate([[True], distances > eps])
            simplified_points = pts[keep_mask]
        else:
            simplified_points = pts
        
        if len(simplified_points) < 2:
            return []
        
        # Teile die Linie in Dash-Dot-Segmente auf
        segments = self._create_dash_dot_segments(simplified_points, dash_length=0.3, dot_length=0.1, gap_length=0.15)
        return segments
    
    def _render_batched_segments(self, segments: List[np.ndarray], color: str, line_width: float, actor_name: str) -> None:
        """Rendert alle Segmente in einem einzigen Mesh (Batch-Rendering für bessere Performance)."""
        if not segments:
            return
        
        # Pre-allocate Arrays
        total_points = sum(len(seg) for seg in segments if len(seg) >= 2)
        if total_points == 0:
            return
        
        # Pre-allocate combined_points Array
        combined_points = np.empty((total_points, 3), dtype=segments[0].dtype)
        all_lines = []
        point_offset = 0
        
        for segment in segments:
            if len(segment) < 2:
                continue
            n_pts = len(segment)
            # Vectorisierte Kopie
            combined_points[point_offset:point_offset + n_pts] = segment
            # Erstelle line_array
            all_lines.append(n_pts)
            all_lines.extend(range(point_offset, point_offset + n_pts))
            point_offset += n_pts
        
        if point_offset > 0:
            polyline = self.pv.PolyData(combined_points[:point_offset])
            polyline.lines = np.array(all_lines, dtype=np.int64)
            
            # 🎯 WICHTIG: Entferne explizit alle Vertices, damit keine Punkte an Segmentenden gerendert werden
            # Dies verhindert, dass Punkte beim zweiten Durchlauf sichtbar werden
            try:
                polyline.verts = np.empty(0, dtype=np.int64)
            except Exception:
                try:
                    polyline.verts = []
                except Exception:
                    pass
            
            # Rendere alle Segmente in einem einzigen Actor
            # 🎯 Verwende die spezielle Methode für Axis-Linien, die vollständige Kontrolle über das Rendering bietet
            print(f"[Axis Lines] Rendering: actor={actor_name}, line_width={line_width:.2f}, color={color}, segments={len(segments)}")
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "line-width-check",
                        "hypothesisId": "LINE_WIDTH_RENDER",
                        "location": "Plot3DOverlaysAxis._render_batched_segments",
                        "message": "_add_axis_line_mesh aufgerufen mit line_width",
                        "data": {
                            "actor_name": actor_name,
                            "line_width": float(line_width),
                            "color": color,
                            "n_segments": len(segments),
                            "total_points": point_offset
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + '\n')
            except Exception:
                pass
            # #endregion
            self._add_axis_line_mesh(
                polyline,
                color=color,
                line_width=line_width,
                name=actor_name,
            )

    def _draw_axis_line_for_surface(
        self,
        intersection_points: Optional[Tuple[np.ndarray, np.ndarray]],
        const_coord: float,
        axis_type: str,
        sort_dim: int,
        const_dim: int,
        selected_axis: Optional[str],
        dash_dot_pattern: int,
    ) -> None:
        """Zeichnet eine Achsenlinie für ein Surface.
        
        Args:
            intersection_points: Tuple von (coords_1, z_coords) - entweder (x_coords, z_coords) für X-Achse
                                oder (y_coords, z_coords) für Y-Achse
            const_coord: Konstante Koordinate (y_axis für X-Achse, x_axis für Y-Achse)
            axis_type: 'x' oder 'y'
            sort_dim: Dimension zum Sortieren (0 für X-Achse, 1 für Y-Achse)
            const_dim: Dimension mit konstanter Koordinate (1 für X-Achse, 0 für Y-Achse)
            selected_axis: 'x' oder 'y' für ausgewählte Achse (wird rot gezeichnet)
            dash_dot_pattern: Hex-Wert für das Line-Pattern (wird nicht direkt verwendet, da VTK es nicht unterstützt)
        """
        if intersection_points is None:
            return
        
        coords_1, z_coords = intersection_points
        if len(coords_1) < 2:
            return
        
        # Erstelle 3D-Punkte: X-Achse: [x, y_axis, z], Y-Achse: [x_axis, y, z]
        if axis_type == 'x':
            pts = np.column_stack([coords_1, np.full_like(coords_1, const_coord), z_coords])
        else:  # axis_type == 'y'
            pts = np.column_stack([np.full_like(coords_1, const_coord), coords_1, z_coords])
        
        # Sortiere nach entsprechender Dimension
        sort_idx = np.argsort(pts[:, sort_dim])
        pts = pts[sort_idx]
        
        # Z-Offset hinzufügen, damit die Linie beim Picking bevorzugt wird
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
            
            # 🎯 FIX: Teile die Linie in Dash-Dot-Segmente auf, da VTK SetLineStipplePattern nicht unterstützt
            segments = self._create_dash_dot_segments(simplified_points, dash_length=0.3, dot_length=0.1, gap_length=0.15)
            
            if not segments:
                return
            
            line_color = 'red' if selected_axis == axis_type else 'black'
            base_line_width = 2.0 if selected_axis == axis_type else 1.5
            line_width = self._get_scaled_line_width(base_line_width, apply_zoom=False)
            
            # 🎯 OPTIMIERUNG: Erstelle Mesh mit allen Segmenten effizienter
            # Pre-allocate Arrays für bessere Performance
            total_points = sum(len(seg) for seg in segments if len(seg) >= 2)
            if total_points == 0:
                return
            
            # Pre-allocate combined_points Array
            combined_points = np.empty((total_points, 3), dtype=simplified_points.dtype)
            all_lines = []
            point_offset = 0
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                n_pts = len(segment)
                # Vectorisierte Kopie statt append
                combined_points[point_offset:point_offset + n_pts] = segment
                # Erstelle line_array effizienter
                all_lines.append(n_pts)
                all_lines.extend(range(point_offset, point_offset + n_pts))
                point_offset += n_pts
            
            if point_offset > 0:
                polyline = self.pv.PolyData(combined_points[:point_offset])
                polyline.lines = np.array(all_lines, dtype=np.int64)
                
                # 🎯 WICHTIG: Entferne explizit alle Vertices, damit keine Punkte an Segmentenden gerendert werden
                try:
                    polyline.verts = np.empty(0, dtype=np.int64)
                except Exception:
                    try:
                        polyline.verts = []
                    except Exception:
                        pass
                
                # Kein line_pattern mehr - verwende durchgezogene Linien für die Segmente
                self._add_overlay_mesh(
                    polyline,
                    color=line_color,
                    line_width=line_width,
                    line_pattern=None,  # Kein Pattern mehr, da wir Segmente verwenden
                    category='axis',
                    render_lines_as_tubes=False,
                )


__all__ = ['SPL3DOverlayAxis']

