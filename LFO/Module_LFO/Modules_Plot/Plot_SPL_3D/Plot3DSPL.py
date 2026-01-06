from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Iterable, Optional

import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from PyQt5 import QtWidgets, QtCore

try:
    import pyvista as pv
except Exception:
    pv = None  # type: ignore

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    build_full_floor_mesh,
    build_surface_mesh,
    build_vertical_surface_mesh,
    derive_surface_plane,
    prepare_plot_geometry,
    prepare_vertical_plot_geometry,
    VerticalPlotGeometry,
    _evaluate_plane_on_grid,
)
from Module_LFO.Modules_Init.Logging import measure_time
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DHelpers import (
    has_valid_data,
    compute_surface_signature,
    quantize_to_steps,
)
from Module_LFO.Modules_Data.SurfaceValidator import triangulate_points
from Module_LFO.Modules_Plot.Plot_SPL_3D.ColorbarManager import PHASE_CMAP

DEBUG_PLOT3D_TIMING = bool(int(os.environ.get("LFO_DEBUG_PLOT3D_TIMING", "1")))

# 🎯 Plot-Subdivision-Level: Anzahl Subdivision-Schritte für schärfere Plots
# 0 = keine Subdivision (Original-Polygone)
# 1 = 4x mehr Faces (jedes Dreieck → 4)
# 2 = 16x mehr Faces (jedes Dreieck → 4 → 16)
# 3 = 64x mehr Faces (jedes Dreieck → 4 → 16 → 64)
PLOT_SUBDIVISION_LEVEL = 1


class SPL3DPlotRenderer:
    """
    Mixin-Klasse für SPL-Plot-Rendering-Funktionalität.
    
    Diese Klasse kapselt alle Methoden, die direkt mit dem Rendering
    des SPL-Plots (Flächen, Texturen, Interpolation) zu tun haben.
    
    Sie erwartet, dass die aufnehmende Klasse folgende Attribute bereitstellt:
    - `plotter` (PyVista Plotter)
    - `settings` (Settings-Objekt)
    - `colorbar_manager` (ColorbarManager)
    - `functions` (FunctionToolbox)
    - `surface_mesh` (PyVista Mesh)
    - `_surface_actors` (dict)
    - `_surface_texture_actors` (dict)
    - `_surface_texture_cache` (dict)
    - `_surface_signature_cache` (dict)
    - `_phase_mode_active` (bool)
    - `_time_mode_active` (bool)
    - `_has_plotted_data` (bool)
    - `_camera_state` (dict)
    - `SURFACE_NAME` (str)
    - `FLOOR_NAME` (str)
    """
    
    # 🗑️ ENTFERNT: _combine_group_meshes - nicht mehr benötigt, da alle Surfaces einzeln geplottet werden
    
    # Statische Hilfsmethoden sind jetzt in Plot3DHelpers.py
    # Verwende Module-Level-Funktionen statt statischer Methoden
    _has_valid_data = staticmethod(has_valid_data)
    _compute_surface_signature = staticmethod(compute_surface_signature)
    _quantize_to_steps = staticmethod(quantize_to_steps)

    @staticmethod
    def _fallback_cmap_to_lut(_self, cmap, n_colors: int | None = None, flip: bool = False):
        if pv is None:
            return None
        if isinstance(cmap, pv.LookupTable):
            lut = cmap
        else:
            lut = pv.LookupTable()
            lut.apply_cmap(cmap, n_values=n_colors or 256, flip=flip)
        if flip and isinstance(cmap, pv.LookupTable):
            lut.values[:] = lut.values[::-1]
        return lut

    @staticmethod
    def _bilinear_interpolate_grid(
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
        ) -> np.ndarray:
        """
        Bilineare Interpolation auf einem regulären Grid.
        Interpoliert zwischen den 4 umgebenden Grid-Punkten für glatte Farbübergänge.
        Ideal für Gradient-Modus.
        """
        source_x = np.asarray(source_x, dtype=float).reshape(-1)
        source_y = np.asarray(source_y, dtype=float).reshape(-1)
        vals = np.asarray(values, dtype=float)
        xq = np.asarray(xq, dtype=float).reshape(-1)
        yq = np.asarray(yq, dtype=float).reshape(-1)

        ny, nx = vals.shape
        if ny != len(source_y) or nx != len(source_x):
            raise ValueError(
                f"_bilinear_interpolate_grid: Shape mismatch values={vals.shape}, "
                f"expected=({len(source_y)}, {len(source_x)})"
            )

        # Randbehandlung: Punkte außerhalb des Grids werden auf den Rand geclippt
        x_min, x_max = source_x[0], source_x[-1]
        y_min, y_max = source_y[0], source_y[-1]
        xq_clip = np.clip(xq, x_min, x_max)
        yq_clip = np.clip(yq, y_min, y_max)

        # Finde Indizes für bilineare Interpolation
        idx_x = np.searchsorted(source_x, xq_clip, side="right") - 1
        idx_y = np.searchsorted(source_y, yq_clip, side="right") - 1
        
        # Clamp auf gültigen Bereich
        idx_x = np.clip(idx_x, 0, nx - 2)  # -2 weil wir idx_x+1 brauchen
        idx_y = np.clip(idx_y, 0, ny - 2)  # -2 weil wir idx_y+1 brauchen
        
        # Hole die 4 umgebenden Punkte
        x0 = source_x[idx_x]
        x1 = source_x[idx_x + 1]
        y0 = source_y[idx_y]
        y1 = source_y[idx_y + 1]
        
        # Werte an den 4 Ecken
        f00 = vals[idx_y, idx_x]
        f10 = vals[idx_y, idx_x + 1]
        f01 = vals[idx_y + 1, idx_x]
        f11 = vals[idx_y + 1, idx_x + 1]
        
        # Berechne Interpolationsgewichte
        dx = x1 - x0
        dy = y1 - y0
        # Vermeide Division durch Null
        dx = np.where(dx > 1e-10, dx, 1.0)
        dy = np.where(dy > 1e-10, dy, 1.0)
        
        wx = (xq_clip - x0) / dx
        wy = (yq_clip - y0) / dy
        
        # Bilineare Interpolation
        result = (
            (1 - wx) * (1 - wy) * f00 +
            wx * (1 - wy) * f10 +
            (1 - wx) * wy * f01 +
            wx * wy * f11
        )
        
        return result

    @staticmethod
    def _nearest_interpolate_grid(
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
        ) -> np.ndarray:
        """
        Nearest-Neighbor Interpolation auf einem regulären Grid.
        Jeder Abfragepunkt erhält exakt den Wert des nächstgelegenen Grid-Punkts.
        Ideal für Color step-Modus (harte Stufen).
        """
        source_x = np.asarray(source_x, dtype=float).reshape(-1)
        source_y = np.asarray(source_y, dtype=float).reshape(-1)
        vals = np.asarray(values, dtype=float)
        xq = np.asarray(xq, dtype=float).reshape(-1)
        yq = np.asarray(yq, dtype=float).reshape(-1)

        ny, nx = vals.shape
        if ny != len(source_y) or nx != len(source_x):
            raise ValueError(
                f"_nearest_interpolate_grid: Shape mismatch values={vals.shape}, "
                f"expected=({len(source_y)}, {len(source_x)})"
            )

        # Randbehandlung: Punkte außerhalb des Grids werden auf den Rand geclippt
        x_min, x_max = source_x[0], source_x[-1]
        y_min, y_max = source_y[0], source_y[-1]
        xq_clip = np.clip(xq, x_min, x_max)
        yq_clip = np.clip(yq, y_min, y_max)

        # Finde nächstgelegenen Index für X
        idx_x = np.searchsorted(source_x, xq_clip, side="left")
        idx_x = np.clip(idx_x, 0, nx - 1)
        # Korrektur: wirklich nächsten Nachbarn wählen
        left_x = idx_x - 1
        right_x = idx_x
        left_x = np.clip(left_x, 0, nx - 1)
        dist_left_x = np.abs(xq_clip - source_x[left_x])
        dist_right_x = np.abs(xq_clip - source_x[right_x])
        use_left_x = dist_left_x < dist_right_x
        idx_x[use_left_x] = left_x[use_left_x]

        # Finde nächstgelegenen Index für Y
        idx_y = np.searchsorted(source_y, yq_clip, side="left")
        idx_y = np.clip(idx_y, 0, ny - 1)
        # Korrektur: wirklich nächsten Nachbarn wählen
        left_y = idx_y - 1
        right_y = idx_y
        left_y = np.clip(left_y, 0, ny - 1)
        dist_left_y = np.abs(yq_clip - source_y[left_y])
        dist_right_y = np.abs(yq_clip - source_y[right_y])
        use_left_y = dist_left_y < dist_right_y
        idx_y[use_left_y] = left_y[use_left_y]

        # Weise jedem Abfragepunkt den Wert des nächstgelegenen Grid-Punkts zu
        result = vals[idx_y, idx_x]
        return result

    @staticmethod
    def _subdivide_triangles(
        vertices: np.ndarray,
        faces: np.ndarray,
        scalars: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unterteilt jedes Dreieck in 4 kleinere Dreiecke für schärfere Plots.
        
        Jedes Dreieck (A, B, C) wird zu 4 Dreiecken:
        1. (A, M_AB, M_AC) - Mitte von AB, Mitte von AC
        2. (M_AB, B, M_BC) - Mitte von AB, B, Mitte von BC
        3. (M_AC, M_BC, C) - Mitte von AC, Mitte von BC, C
        4. (M_AB, M_BC, M_AC) - Zentrum-Dreieck
        
        Args:
            vertices: Array der Vertex-Koordinaten (Shape: N, 3)
            faces: Array der Face-Indizes im PyVista-Format [n, i1, i2, i3, n, i4, i5, i6, ...]
            scalars: Array der Skalar-Werte pro Vertex (Shape: N,)
            
        Returns:
            Tuple von (neue_vertices, neue_faces, neue_scalars)
        """
        # Parse faces im PyVista-Format [n, v1, v2, v3, n, v4, v5, v6, ...]
        n_faces = len(faces) // 4  # 4 Elemente pro Face
        original_faces_list = []
        for i in range(n_faces):
            idx = i * 4
            if faces[idx] == 3:  # Dreieck
                v1, v2, v3 = faces[idx + 1], faces[idx + 2], faces[idx + 3]
                original_faces_list.append((v1, v2, v3))
        
        if len(original_faces_list) == 0:
            return vertices, faces, scalars
        
        # Verwende Dictionary für Kanten-Mittelpunkte (Cache für gemeinsame Kanten)
        edge_midpoints = {}  # (min_idx, max_idx) -> midpoint_vertex_idx
        
        def get_edge_midpoint(v1_idx: int, v2_idx: int, vertex_counter: int) -> tuple[int, int]:
            """Gibt Index des Kanten-Mittelpunkts zurück, erstellt ihn falls nötig."""
            edge_key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            if edge_key not in edge_midpoints:
                # Berechne Mittelpunkt
                mid_coord = (vertices[v1_idx] + vertices[v2_idx]) / 2.0
                mid_scalar = (scalars[v1_idx] + scalars[v2_idx]) / 2.0
                new_vertices.append(mid_coord)
                new_scalars.append(mid_scalar)
                edge_midpoints[edge_key] = vertex_counter
                vertex_counter += 1
            return edge_midpoints[edge_key], vertex_counter
        
        # Neue Vertices und Scalars (starte mit Original-Vertices)
        new_vertices = [v.copy() for v in vertices]
        new_scalars = [s for s in scalars]
        vertex_counter = len(vertices)
        
        # Neue Faces
        new_faces_list = []
        
        # Verarbeite jedes Original-Dreieck
        for v1_idx, v2_idx, v3_idx in original_faces_list:
            # Berechne Mittelpunkte der 3 Kanten
            m12_idx, vertex_counter = get_edge_midpoint(v1_idx, v2_idx, vertex_counter)
            m23_idx, vertex_counter = get_edge_midpoint(v2_idx, v3_idx, vertex_counter)
            m13_idx, vertex_counter = get_edge_midpoint(v1_idx, v3_idx, vertex_counter)
            
            # Erstelle 4 neue Dreiecke
            # 1. (v1, m12, m13)
            new_faces_list.extend([3, v1_idx, m12_idx, m13_idx])
            # 2. (m12, v2, m23)
            new_faces_list.extend([3, m12_idx, v2_idx, m23_idx])
            # 3. (m13, m23, v3)
            new_faces_list.extend([3, m13_idx, m23_idx, v3_idx])
            # 4. (m12, m23, m13) - Zentrum-Dreieck
            new_faces_list.extend([3, m12_idx, m23_idx, m13_idx])
        
        # Konvertiere zu NumPy-Arrays
        new_vertices_array = np.array(new_vertices, dtype=float)
        new_scalars_array = np.array(new_scalars, dtype=float)
        new_faces_array = np.array(new_faces_list, dtype=np.int64)
        
        return new_vertices_array, new_faces_array, new_scalars_array
    
    @staticmethod
    def _project_point_to_polyline(x: float, y: float, poly_x: np.ndarray, poly_y: np.ndarray) -> tuple[float, float]:
        """
        Projiziert einen Punkt (x, y) auf die nächste Kante eines Polygons.
        Gibt die projizierten Koordinaten zurück.
        """
        if poly_x.size < 2 or poly_y.size < 2:
            return (x, y)
        
        if poly_x.size != poly_y.size:
            return (x, y)
        
        min_dist_sq = float('inf')
        best_x, best_y = x, y
        
        # Iteriere über alle Kanten
        for i in range(poly_x.size):
            x0 = float(poly_x[i])
            y0 = float(poly_y[i])
            x1 = float(poly_x[(i + 1) % poly_x.size])
            y1 = float(poly_y[(i + 1) % poly_y.size])
            
            # Vektor entlang der Kante
            dx = x1 - x0
            dy = y1 - y0
            edge_len_sq = dx * dx + dy * dy
            
            if edge_len_sq < 1e-12:
                # Degenerierte Kante - verwende Endpunkt
                dist_sq = (x - x0) * (x - x0) + (y - y0) * (y - y0)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_x, best_y = x0, y0
            else:
                # Projiziere Punkt auf Kante
                t = ((x - x0) * dx + (y - y0) * dy) / edge_len_sq
                t = max(0.0, min(1.0, t))  # Clamp auf [0, 1]
                
                proj_x = x0 + t * dx
                proj_y = y0 + t * dy
                
                dist_sq = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_x, best_y = proj_x, proj_y
        
        return (best_x, best_y)

    @measure_time("PlotSPL3D._render_surfaces")
    def _render_surfaces(
        self,
        geometry,
        original_plot_values: np.ndarray,
        cbar_min: float,
        cbar_max: float,
        cmap_object: str | Any,
        colorization_mode: str,
        cbar_step: float,
        surface_overrides: Optional[dict[str, dict[str, np.ndarray]]] = None,
        phase_mode: bool = False,
        time_mode: bool = False,
        ) -> None:
        """
        Renderpfad für Surfaces mit Triangulation.

        - Verwendet triangulierte Meshes aus surface_grids_data
        - Interpoliert SPL-Werte auf Vertex-Koordinaten
        - Unterstützt Gruppen-Meshes und einzelne Surfaces
        - Verwendet Subdivision für höhere Auflösung
        """
        if not hasattr(self, "plotter") or self.plotter is None:
            return

        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return

        # Optionales Feintiming für diesen Pfad
        t_render_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0

        if not surface_overrides:
            return

        # Bestimme ob Step-Modus aktiv ist (vor der Verwendung definieren)
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0

        # Colormap vorbereiten
        if isinstance(cmap_object, str):
            base_cmap = cm.get_cmap(cmap_object)
        else:
            base_cmap = cmap_object
        
        # 🎯 Für Color Step: Erstelle diskrete Colormap mit exakten Levels (wie ColorbarManager)
        if is_step_mode:
            # Verwende die gleiche Logik wie ColorbarManager
            num_segments = 10  # Immer 10 Segmente/Farben (wie in ColorbarManager.NUM_COLORS_STEP_MODE - 1)

            # Resample Colormap
            if isinstance(base_cmap, str):
                base_cmap_obj = cm.get_cmap(base_cmap)
            else:
                base_cmap_obj = base_cmap
            
            if hasattr(base_cmap_obj, "resampled"):
                sampled_cmap = base_cmap_obj.resampled(num_segments)
            else:
                sampled_cmap = cm.get_cmap(base_cmap, num_segments)
            
            # Erstelle Farb-Liste (wie in ColorbarManager)
            sample_points = (np.arange(num_segments, dtype=float) + 0.5) / max(num_segments, 1)
            color_list = sampled_cmap(sample_points)
            cmap_object = ListedColormap(color_list)
        else:
            # Gradient-Modus: Verwende die kontinuierliche Colormap
            cmap_object = base_cmap
        
        norm = Normalize(vmin=cbar_min, vmax=cbar_max)

        # Aktive Surfaces ermitteln
        surface_definitions = getattr(self.settings, "surface_definitions", {}) or {}
        
        # 🎯 NEU: Prüfe welche Surfaces als "empty" markiert sind (inkompatibel bei Snapshot)
        calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
        surface_grids_data = calc_spl.get("surface_grids", {}) or {} if isinstance(calc_spl, dict) else {}
        empty_surface_ids = set()
        if isinstance(surface_grids_data, dict):
            for sid, grid_data in surface_grids_data.items():
                if isinstance(grid_data, dict) and grid_data.get('is_empty', False):
                    empty_surface_ids.add(sid)
        
        # 🎯 NEU: Erstelle Mapping von group_id zu Gruppen-Status für schnellen Zugriff
        surface_groups = getattr(self.settings, 'surface_groups', {})
        group_status: dict[str, dict[str, bool]] = {}  # group_id -> {'enabled': bool, 'hidden': bool}
        if isinstance(surface_groups, dict):
            for group_id, group_data in surface_groups.items():
                if hasattr(group_data, 'enabled'):
                    group_status[group_id] = {
                        'enabled': bool(group_data.enabled),
                        'hidden': bool(getattr(group_data, 'hidden', False))
                    }
                elif isinstance(group_data, dict):
                    group_status[group_id] = {
                        'enabled': bool(group_data.get('enabled', True)),
                        'hidden': bool(group_data.get('hidden', False))
                    }
        
        enabled_surfaces: list[tuple[str, list[dict[str, float]], Any]] = []
        if isinstance(surface_definitions, dict):
            for surface_id, surface_def in surface_definitions.items():
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, "enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False))
                    points = getattr(surface_def, "points", []) or []
                    surface_obj = surface_def
                    group_id = getattr(surface_def, 'group_id', None)
                else:
                    enabled = bool(surface_def.get("enabled", False))
                    hidden = bool(surface_def.get("hidden", False))
                    points = surface_def.get("points", []) or []
                    surface_obj = surface_def
                    group_id = surface_def.get('group_id') or surface_def.get('group_name')
                
                # 🎯 NEU: Berücksichtige Gruppen-Status (wie in draw_surfaces)
                if group_id and group_id in group_status:
                    group_enabled = group_status[group_id]['enabled']
                    group_hidden = group_status[group_id]['hidden']
                    
                    # Wenn Gruppe hidden ist → Surface komplett überspringen
                    if group_hidden:
                        hidden = True
                    # Wenn Gruppe disabled ist → Surface als disabled behandeln
                    elif not group_enabled:
                        enabled = False
                
                # 🎯 NEU: Überspringe Surfaces, die als "empty" markiert sind (inkompatibel)
                if str(surface_id) in empty_surface_ids:
                    continue  # Surface ist inkompatibel → nicht zu enabled_surfaces hinzufügen
                
                if enabled and not hidden and len(points) >= 3:
                    enabled_surfaces.append((str(surface_id), points, surface_obj))
                    
                    
        

        # Nicht mehr benötigte Actors entfernen
        # 🎯 NEU: Entferne auch Surfaces mit is_empty=True aus active_ids, damit ihre Actors entfernt werden
        active_ids = {sid for sid, _, _ in enabled_surfaces}
        active_ids = active_ids - empty_surface_ids  # Entferne empty Surfaces aus active_ids
        
        # Entferne alte Texture-Actors (falls noch vorhanden)
        if hasattr(self, "_surface_texture_actors") and isinstance(self._surface_texture_actors, dict):
            for sid, texture_data in list(self._surface_texture_actors.items()):
                if sid not in active_ids:
                    try:
                        actor = texture_data.get('actor') if isinstance(texture_data, dict) else texture_data
                        if actor is not None:
                            self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                    self._surface_texture_actors.pop(sid, None)

        # Gecachte triangulierte Surface-Meshes entfernen, wenn die zugehörige Surface
        # deaktiviert oder versteckt wurde. So werden keine alten Meshes für versteckte
        # Flächen weiterverwendet.
        if hasattr(self, "_surface_actors") and isinstance(self._surface_actors, dict):
            for sid, entry in list(self._surface_actors.items()):
                if sid not in active_ids:
                    try:
                        actor = entry.get("actor") if isinstance(entry, dict) else entry
                        if actor is not None:
                            self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                    self._surface_actors.pop(sid, None)
        
        # 🎯 NEU: Entferne Gruppen-Actors für inaktive, disabled oder hidden Gruppen
        if hasattr(self, "_group_actors") and isinstance(self._group_actors, dict):
            # Prüfe welche Gruppen noch aktiv sind (basierend auf surface_results_data)
            calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
            surface_results_data = calc_spl.get("surface_results", {}) or {} if isinstance(calc_spl, dict) else {}
            
            active_group_ids = set()
            if isinstance(surface_results_data, dict):
                for result_data in surface_results_data.values():
                    if isinstance(result_data, dict) and result_data.get('is_group_sum', False):
                        group_id = result_data.get('group_id')
                        if group_id:
                            # Prüfe zusätzlich, ob die Gruppe enabled und nicht hidden ist
                            group_enabled = True
                            group_hidden = False
                            surface_groups = getattr(self.settings, 'surface_groups', {}) if hasattr(self, 'settings') else {}
                            if isinstance(surface_groups, dict) and group_id in surface_groups:
                                group_data = surface_groups[group_id]
                                if hasattr(group_data, 'enabled'):
                                    group_enabled = bool(group_data.enabled)
                                    group_hidden = bool(getattr(group_data, 'hidden', False))
                                elif isinstance(group_data, dict):
                                    group_enabled = bool(group_data.get('enabled', True))
                                    group_hidden = bool(group_data.get('hidden', False))
                            
                            # Nur als aktiv markieren, wenn enabled und nicht hidden
                            if group_enabled and not group_hidden:
                                active_group_ids.add(group_id)
            
            for group_id, actor in list(self._group_actors.items()):
                # Entferne Actor wenn Gruppe nicht mehr aktiv ist oder disabled/hidden
                should_remove = group_id not in active_group_ids
                
                # Zusätzliche Prüfung: Entferne auch wenn Gruppe disabled oder hidden ist
                if not should_remove:
                    group_enabled = True
                    group_hidden = False
                    surface_groups = getattr(self.settings, 'surface_groups', {}) if hasattr(self, 'settings') else {}
                    if isinstance(surface_groups, dict) and group_id in surface_groups:
                        group_data = surface_groups[group_id]
                        if hasattr(group_data, 'enabled'):
                            group_enabled = bool(group_data.enabled)
                            group_hidden = bool(getattr(group_data, 'hidden', False))
                        elif isinstance(group_data, dict):
                            group_enabled = bool(group_data.get('enabled', True))
                            group_hidden = bool(group_data.get('hidden', False))
                    should_remove = not group_enabled or group_hidden
                
                if should_remove:
                    
                    
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                    self._group_actors.pop(group_id, None)

        # 🎯 WICHTIG: Wenn keine enabled_surfaces vorhanden sind, aber surface_overrides existieren,
        # sollten wir trotzdem fortfahren, um vertikale Flächen zu plotten
        if not enabled_surfaces and not surface_overrides:
            return

        # Verarbeite Surfaces sequenziell
        surfaces_to_process = []
        
        # SCHRITT 1: Sammle Surfaces für Verarbeitung
        
        for surface_id, points, surface_obj in enabled_surfaces:
            # Berechne Signatur für Cache-Prüfung (schnell, kann im Hauptthread bleiben)
            poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
            poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
            if poly_x.size == 0 or poly_y.size == 0:
                continue
            
            # Plane-Model wird nicht mehr für Textur-Cache benötigt, aber für spätere Verwendung berechnen
            dict_points = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                for p in points
            ]
            plane_model, _ = derive_surface_plane(dict_points)
            
            # Per-Surface Overrides aus neuem Grid nutzen (nur Positionen; keine Fallbacks)
            override = surface_overrides.get(surface_id) if surface_overrides else None
            if not override:
                continue
            sx = override.get("source_x", np.array([]))
            sy = override.get("source_y", np.array([]))
            vals = override.get("values", np.array([]))
            override_used = True
            
            # Muss verarbeitet werden (Triangulation-Pfad)
            surfaces_to_process.append((surface_id, points, surface_obj))
        
        # SCHRITT 2: Sequenzielle Verarbeitung der Surfaces
        surfaces_processed = 0
        
        # 🎯 PRIORITÄT 1: Versuche Triangulation (wenn verfügbar)
        # Hole triangulierte Daten aus surface_grids_data
        calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
        surface_grids_data = calc_spl.get("surface_grids", {}) or {} if isinstance(calc_spl, dict) else {}
        surface_results_data = calc_spl.get("surface_results", {}) or {} if isinstance(calc_spl, dict) else {}
        
        # 🎯 ERWEITERE surfaces_to_process: Füge Surfaces hinzu, die Overrides haben,
        # aber nicht in enabled_surfaces sind – z.B. vertikale Flächen aus calculation_spl.
        # WICHTIG: Surfaces mit hidden=True oder enabled=False werden hier NICHT geplottet.
        override_surface_ids = set(surface_overrides.keys()) if surface_overrides else set()
        enabled_surface_ids = {sid for sid, _, _ in enabled_surfaces} if enabled_surfaces else set()
        surface_definitions = getattr(self.settings, "surface_definitions", {}) or {}

        additional_surface_ids: set[str] = set()
        for sid in (override_surface_ids - enabled_surface_ids):
            surface_def = surface_definitions.get(sid)
            if surface_def is None:
                continue
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, "enabled", False))
                hidden = bool(getattr(surface_def, "hidden", False))
                points = getattr(surface_def, "points", []) or []
            else:
                enabled = bool(surface_def.get("enabled", False))
                hidden = bool(surface_def.get("hidden", False))
                points = surface_def.get("points", []) or []
            if not enabled or hidden or len(points) < 3:
                continue
            additional_surface_ids.add(sid)
        
        if additional_surface_ids and isinstance(surface_grids_data, dict):
            for sid in additional_surface_ids:
                if sid in surface_grids_data:
                    surface_def = surface_definitions.get(sid)
                    if surface_def:
                        if isinstance(surface_def, SurfaceDefinition):
                            points = getattr(surface_def, "points", []) or []
                        else:
                            points = surface_def.get("points", []) or []
                        if len(points) >= 3:
                            surfaces_to_process.append((sid, points, surface_def))
        
        # Sequenzielle Verarbeitung aller Surfaces (alle werden identisch behandelt)
        
        for surface_id, points, surface_obj in surfaces_to_process:
            
            try:
                # Verarbeite Surface – nur mit Override (keine globalen Fallbacks)
                override = surface_overrides.get(surface_id)
                if not override:
                    
                    continue
                sx = override.get("source_x", np.array([]))
                sy = override.get("source_y", np.array([]))
                vals = override.get("values", np.array([]))
                override_used_here = True
                
                # #region agent log
                try:
                    import json
                    import time as time_module
                    vals_valid = vals[np.isfinite(vals)] if vals.size > 0 else np.array([])
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "K3",
                            "location": "Plot3DSPL.py:_render_surfaces:surface_override_used",
                            "message": "Surface Override verwendet",
                            "data": {
                                "surface_id": surface_id,
                                "phase_mode": phase_mode,
                                "vals_shape": list(vals.shape) if vals.size > 0 else None,
                                "vals_valid_count": len(vals_valid),
                                "vals_min": float(np.nanmin(vals_valid)) if len(vals_valid) > 0 else None,
                                "vals_max": float(np.nanmax(vals_valid)) if len(vals_valid) > 0 else None,
                                "vals_mean": float(np.nanmean(vals_valid)) if len(vals_valid) > 0 else None
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion

                # 🎯 PRIORITÄT 1: Prüfe ob triangulierte Daten verfügbar sind
                use_triangulation = False
                triangulated_vertices = None
                triangulated_faces = None
                grid_data = None
                
                if isinstance(surface_grids_data, dict) and surface_id in surface_grids_data:
                    grid_data = surface_grids_data[surface_id]
                    
                    # 🎯 NEU: Prüfe ob Surface als "empty" markiert ist (inkompatibel bei Snapshot)
                    is_empty = grid_data.get('is_empty', False) if isinstance(grid_data, dict) else False
                    if is_empty:
                        # Surface ist inkompatibel → überspringe Rendering (wird als graue Fläche in draw_surfaces gezeichnet)
                        continue
                    
                    # 🎯 IDENTISCHE BEHANDLUNG: Planare und vertikale Flächen verwenden beide Triangulation
                    orientation = grid_data.get("orientation", "").lower() if isinstance(grid_data, dict) else ""
                    
                    # 🎯 DEBUG: Ermittle Orientation aus Geometry-Objekt
                    surface_orientation = orientation
                    geometry_obj = grid_data.get('geometry')
                    if geometry_obj and hasattr(geometry_obj, 'orientation'):
                        surface_orientation = geometry_obj.orientation
                    
                    triangulated_success = grid_data.get('triangulated_success', False)
                    
                    # 🎯 VERTIKALE FLÄCHEN: Werden jetzt identisch wie planare Flächen behandelt
                    # Keine separate Behandlung mehr nötig - beide verwenden Triangulation
                    
                    # Prüfe Triangulation-Status (mit Orientation)
                    
                    if triangulated_success:
                        triangulated_vertices_list = grid_data.get('triangulated_vertices')
                        triangulated_faces_list = grid_data.get('triangulated_faces')
                        
                        if triangulated_vertices_list and triangulated_faces_list:
                            try:
                                triangulated_vertices = np.array(triangulated_vertices_list, dtype=float)
                                triangulated_faces = np.array(triangulated_faces_list, dtype=np.int64)
                                
                                # 🎯 VERGLEICH: Surface-Punkte vs. Grid-Koordinaten
                                surface_points = None
                                if geometry_obj and hasattr(geometry_obj, 'points') and geometry_obj.points:
                                    surface_points = geometry_obj.points
                                elif hasattr(self.settings, 'surface_definitions'):
                                    surface_def = self.settings.surface_definitions.get(surface_id)
                                    if surface_def and hasattr(surface_def, 'points'):
                                        surface_points = surface_def.points
                                
                                
                                # 🎯 VERGLEICH: Grid-Koordinaten vs. Vertices
                                # Lade Grid-Daten für Vergleich
                                Xg_check = np.asarray(grid_data.get("X_grid", []))
                                Yg_check = np.asarray(grid_data.get("Y_grid", []))
                                if Xg_check.size > 0 and Yg_check.size > 0:
                                    Zg_check = np.asarray(grid_data.get("Z_grid", np.zeros_like(Xg_check)))
                                    if Zg_check.shape != Xg_check.shape:
                                        Zg_check = np.zeros_like(Xg_check)
                                    grid_vertices = np.column_stack([Xg_check.ravel(), Yg_check.ravel(), Zg_check.ravel()])
                                
                                use_triangulation = True
                            except Exception as e:
                                use_triangulation = False
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                
                # 🎯 TRIANGULATION: Wenn verfügbar, verwende trianguliertes Mesh (PRIORITÄT 1)
                if use_triangulation and triangulated_vertices is not None and triangulated_faces is not None:
                    try:
                        from scipy.interpolate import griddata
                        
                        # Lade SPL-Werte aus surface_results_data
                        result_data = surface_results_data.get(surface_id) if isinstance(surface_results_data, dict) else None
                        if result_data is None:
                            use_triangulation = False
                        else:
                            sound_field_p_complex = np.array(result_data.get('sound_field_p', []), dtype=complex)
                            
                            # Lade Grid-Daten (benötigt für beide Pfade)
                            Xg = np.asarray(grid_data.get("X_grid", []))
                            Yg = np.asarray(grid_data.get("Y_grid", []))
                            
                            
                            # 🎯 Stelle sicher, dass surface_orientation verfügbar ist
                            if 'surface_orientation' not in locals():
                                geometry_obj = grid_data.get('geometry')
                                if geometry_obj and hasattr(geometry_obj, 'orientation'):
                                    surface_orientation = geometry_obj.orientation
                                else:
                                    surface_orientation = grid_data.get("orientation", "").lower() if isinstance(grid_data, dict) else None
                            
                            # 🎯 FIX: Für Phase-Modus verwende Phase-Daten aus Surface-Overrides, nicht aus sound_field_p_complex
                            # Die Surface-Overrides enthalten bereits die korrekt interpolierten Phase-Daten
                            if phase_mode and override_used_here and vals.size > 0:
                                # Verwende Phase-Daten aus Surface-Overrides
                                # Stelle sicher, dass vals die richtige Shape hat
                                if vals.shape == Xg.shape:
                                    spl_values_2d = vals.copy()
                                elif vals.size == Xg.size:
                                    spl_values_2d = vals.reshape(Xg.shape)
                                else:
                                    # Fallback: Verwende np.angle wenn Shape nicht passt
                                    spl_values_2d = np.angle(sound_field_p_complex)
                                    spl_values_2d = np.where(np.isinf(spl_values_2d), 0.0, spl_values_2d)
                                    # Behalte NaN-Werte für transparente Darstellung
                            elif time_mode:
                                spl_values_2d = np.real(sound_field_p_complex)
                                spl_values_2d = np.nan_to_num(spl_values_2d, nan=0.0, posinf=0.0, neginf=0.0)
                            elif phase_mode:
                                # Fallback: Wenn keine Overrides vorhanden sind, verwende np.angle
                                spl_values_2d = np.angle(sound_field_p_complex)
                                spl_values_2d = np.where(np.isinf(spl_values_2d), 0.0, spl_values_2d)
                                # Behalte NaN-Werte für transparente Darstellung
                            else:
                                pressure_magnitude = np.abs(sound_field_p_complex)
                                pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                                spl_values_2d = self.functions.mag2db(pressure_magnitude)
                                spl_values_2d = np.nan_to_num(spl_values_2d, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # ⚠️ StructuredGrid-Pfad vorübergehend deaktiviert wegen Orientierungsproblemen
                            # Verwende stattdessen den Triangulation-Pfad, der konsistente Orientierung garantiert
                            use_structured_grid = False  # Deaktiviert bis Orientierung geklärt
                            
                            if use_structured_grid:
                                try:
                                    if Xg.ndim == 2 and Yg.ndim == 2 and spl_values_2d.ndim == 2 and Xg.shape == Yg.shape == spl_values_2d.shape:
                                        Zg = np.asarray(grid_data.get("Z_grid", np.zeros_like(Xg)))
                                        if Zg.shape != Xg.shape:
                                            Zg = np.zeros_like(Xg)
                                        scalars_grid = np.array(spl_values_2d, copy=True)
                                        surface_mask_arr = np.asarray(grid_data.get("surface_mask", []), dtype=bool)
                                        if surface_mask_arr.shape == Xg.shape:
                                            scalars_grid = scalars_grid.copy()
                                            scalars_grid[~surface_mask_arr] = np.nan
                                        
                                        grid_mesh = pv.StructuredGrid(Xg, Yg, Zg)
                                        grid_mesh["plot_scalars"] = scalars_grid.ravel(order="F")
                                        finite_mask = np.isfinite(grid_mesh["plot_scalars"])
                                        if not np.any(finite_mask):
                                            raise ValueError("Keine gültigen Skalarwerte im Grid")
                                        clim_min = float(np.nanmin(grid_mesh["plot_scalars"]))
                                        clim_max = float(np.nanmax(grid_mesh["plot_scalars"]))
                                        if np.isclose(clim_min, clim_max):
                                            clim_min -= 1.0
                                            clim_max += 1.0
                                        actor_name = f"{self.SURFACE_NAME}_gridtri_{surface_id}"
                                        
                                        old_texture_data = self._surface_texture_actors.get(surface_id)
                                        if old_texture_data is not None:
                                            old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                                            if old_actor is not None:
                                                try:
                                                    self.plotter.remove_actor(old_actor)
                                                except Exception:
                                                    pass
                                        
                                        surf = grid_mesh.extract_surface().triangulate()
                                        surf["plot_scalars"] = grid_mesh["plot_scalars"]
                                        actor = self.plotter.add_mesh(
                                            surf,
                                            name=actor_name,
                                            scalars="plot_scalars",
                                            cmap=cmap_object,
                                            clim=(clim_min, clim_max),
                                            smooth_shading=False,
                                            show_scalar_bar=False,
                                            reset_camera=False,
                                            interpolate_before_map=False,
                                        )
                                        if hasattr(actor, 'SetPickable'):
                                            actor.SetPickable(False)
                                        if not hasattr(self, '_surface_actors'):
                                            self._surface_actors = {}
                                        self._surface_actors[surface_id] = actor
                                        self._surface_texture_actors.pop(surface_id, None)
                                        surfaces_processed += 1
                                        continue
                                except Exception as e:
                                    print(f"  └─ StructuredGrid-Plot fehlgeschlagen, fahre mit Triangulation fort: {e}")
                            
                            # 🚀 OPTIMIERUNG: Vertices ↔ Grid-Punkte
                            # - Color step: immer schnellster Nearest-Neighbour-Pfad (stufiges Bild gewollt)
                            # - Gradient: bilineare Interpolation auf die Vertex-Koordinaten
                            surface_mask = np.asarray(grid_data.get("surface_mask", []))
                            n_vertices = len(triangulated_vertices)
                            n_grid_points = Xg.size
                            
                            # 🎯 OPTIMIERUNG: Lade vertex_source_indices und additional_vertices_spl
                            vertex_source_indices_list = grid_data.get('vertex_source_indices')
                            vertex_source_indices = None
                            if vertex_source_indices_list is not None:
                                try:
                                    vertex_source_indices = np.array(vertex_source_indices_list, dtype=np.int64)
                                except Exception:
                                    vertex_source_indices = None
                            
                            additional_vertices_spl_list = result_data.get('additional_vertices_spl') if result_data else None
                            additional_vertices_spl = None
                            if additional_vertices_spl_list is not None:
                                try:
                                    additional_vertices_spl = np.array(additional_vertices_spl_list, dtype=complex)
                                    # 🎯 FIX: Für Phase-Modus verwende Phase-Daten aus Surface-Overrides
                                    # Für additional_vertices müssen wir interpolieren, da sie nicht in Overrides sind
                                    if phase_mode and override_used_here and vals.size > 0:
                                        # Interpoliere Phase-Daten von Grid auf additional_vertices
                                        from scipy.interpolate import griddata
                                        if Xg.size > 0 and Yg.size > 0 and vals.size > 0:
                                            # Erstelle Quell-Punkte (Grid-Koordinaten)
                                            Xg_flat = Xg.ravel()
                                            Yg_flat = Yg.ravel()
                                            vals_flat = vals.ravel()
                                            points_source = np.column_stack([Xg_flat, Yg_flat])
                                            values_source = vals_flat
                                            
                                            # Ziel-Punkte (additional_vertices X/Y-Koordinaten)
                                            additional_vertices_coords_temp = grid_data.get('additional_vertices')
                                            if additional_vertices_coords_temp is not None:
                                                try:
                                                    additional_vertices_coords_temp = np.array(additional_vertices_coords_temp, dtype=float)
                                                    if additional_vertices_coords_temp.ndim == 2 and additional_vertices_coords_temp.shape[1] >= 2:
                                                        points_target = additional_vertices_coords_temp[:, :2]  # Nur X, Y
                                                        # Interpoliere Phase-Daten
                                                        additional_vertices_spl_db = griddata(
                                                            points_source, values_source, points_target,
                                                            method='linear', fill_value=np.nan
                                                        )
                                                        # Setze nur inf auf 0, behalte NaN
                                                        additional_vertices_spl_db = np.where(np.isinf(additional_vertices_spl_db), 0.0, additional_vertices_spl_db)
                                                    else:
                                                        additional_vertices_spl_db = None
                                                except Exception:
                                                    additional_vertices_spl_db = None
                                            else:
                                                additional_vertices_spl_db = None
                                        else:
                                            additional_vertices_spl_db = None
                                    elif time_mode:
                                        additional_vertices_spl_db = np.real(additional_vertices_spl)
                                        additional_vertices_spl_db = np.nan_to_num(additional_vertices_spl_db, nan=0.0, posinf=0.0, neginf=0.0)
                                    elif phase_mode:
                                        # Fallback: Wenn keine Overrides vorhanden sind, verwende np.angle
                                        additional_vertices_spl_db = np.angle(additional_vertices_spl)
                                        additional_vertices_spl_db = np.where(np.isinf(additional_vertices_spl_db), 0.0, additional_vertices_spl_db)
                                        # Behalte NaN-Werte für transparente Darstellung
                                    else:
                                        pressure_magnitude_add = np.abs(additional_vertices_spl)
                                        pressure_magnitude_add = np.clip(pressure_magnitude_add, 1e-12, None)
                                        additional_vertices_spl_db = self.functions.mag2db(pressure_magnitude_add)
                                        additional_vertices_spl_db = np.nan_to_num(additional_vertices_spl_db, nan=0.0, posinf=0.0, neginf=0.0)
                                except Exception:
                                    additional_vertices_spl_db = None
                            else:
                                additional_vertices_spl_db = None
                            
                            # 🎯 Lade additional_vertices-Koordinaten für Gradient-Modus
                            additional_vertices_coords = None
                            if additional_vertices_spl_db is not None:
                                additional_vertices_list = grid_data.get('additional_vertices')
                                if additional_vertices_list is not None:
                                    try:
                                        additional_vertices_coords = np.array(additional_vertices_list, dtype=float)
                                        if additional_vertices_coords.ndim != 2 or additional_vertices_coords.shape[1] != 3:
                                            additional_vertices_coords = None
                                        elif len(additional_vertices_coords) != len(additional_vertices_spl_db):
                                            # Anzahl stimmt nicht überein - ignoriere
                                            additional_vertices_coords = None
                                    except Exception:
                                        additional_vertices_coords = None
                            
                            if is_step_mode:
                                # ===========================
                                # COLOR-STEP → NEAREST NEIGHBOUR
                                # ===========================
                                # 🎯 OPTIMIERUNG: Verwende vertex_source_indices wenn verfügbar
                                if vertex_source_indices is not None and len(vertex_source_indices) == n_vertices:
                                    # Verwende vertex_source_indices für direkte Zuordnung
                                    spl_at_verts = np.full(n_vertices, np.nan, dtype=float)
                                    for v_idx in range(n_vertices):
                                        source_idx = vertex_source_indices[v_idx]
                                        if source_idx < n_grid_points:
                                            # Grid-Vertex: Verwende Grid-SPL-Werte
                                            if source_idx < spl_values_2d.size:
                                                spl_at_verts[v_idx] = spl_values_2d.ravel()[source_idx]
                                        else:
                                            # Additional-Vertex: Verwende additional_vertices_spl
                                            if additional_vertices_spl_db is not None:
                                                additional_idx = source_idx - n_grid_points
                                                if 0 <= additional_idx < len(additional_vertices_spl_db):
                                                    spl_at_verts[v_idx] = additional_vertices_spl_db[additional_idx]
                                    
                                    valid_mask = np.isfinite(spl_at_verts)
                                    if not np.any(valid_mask):
                                        # Fallback: Verwende alte Logik
                                        if n_vertices == n_grid_points:
                                            spl_at_verts = spl_values_2d.ravel().copy()
                                        else:
                                            spl_at_verts = None
                                elif n_vertices == n_grid_points:
                                    # 🎯 Direkte Zuordnung: Vertices = Grid-Punkte in derselben Reihenfolge
                                    spl_at_verts = spl_values_2d.ravel().copy()
                                    if surface_mask.size == n_grid_points and surface_mask.shape == Xg.shape:
                                        mask_flat = surface_mask.ravel().astype(bool)
                                        spl_at_verts[~mask_flat] = np.nan
                                else:
                                    # 🎯 Zusätzliche Vertices: Nearest-Neighbour auf Gridpunkte (diskrete Stufen)
                                    try:
                                        from scipy.spatial import cKDTree
                                        # Erstelle Grid-Punkte für Nearest-Map (2D reicht hier)
                                        grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
                                        grid_vals = spl_values_2d.ravel()
                                        
                                        if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                            mask_flat = surface_mask.ravel().astype(bool)
                                            grid_pts = grid_pts[mask_flat]
                                            grid_vals = grid_vals[mask_flat]
                                        
                                        tree = cKDTree(grid_pts)
                                        _, nn_idx = tree.query(triangulated_vertices[:, :2], k=1)
                                        spl_at_verts = grid_vals[nn_idx]
                                        
                                        valid_mask = np.isfinite(spl_at_verts)
                                        if not np.any(valid_mask):
                                            raise ValueError(f"Surface '{surface_id}': Nearest-Map liefert keine gültigen Werte")
                                    except Exception:
                                        from scipy.interpolate import griddata
                                        points_new = triangulated_vertices[:, :2]
                                        points_orig = np.column_stack([Xg.ravel(), Yg.ravel()])
                                        values_orig = spl_values_2d.ravel()
                                        if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                            mask_flat = surface_mask.ravel().astype(bool)
                                            if np.any(mask_flat):
                                                points_orig = points_orig[mask_flat]
                                                values_orig = values_orig[mask_flat]
                                        spl_at_verts = griddata(
                                            points_orig,
                                            values_orig,
                                            points_new,
                                            method='nearest',
                                            fill_value=np.nan,
                                        )
                                        valid_mask = np.isfinite(spl_at_verts)
                                        if not np.any(valid_mask):
                                            raise ValueError(f"Surface '{surface_id}': Griddata(nearest) liefert keine gültigen Werte")
                            else:
                                # ===========================
                                # GRADIENT → BILINEARE INTERPOLATION ODER LINEAR MIT RANDPUNKTEN
                                # ===========================
                                # 🎯 FIX: Wenn additional_vertices vorhanden sind, verwende griddata mit linear
                                # statt bilinearer Interpolation, um Randpunkte einzubeziehen
                                if additional_vertices_coords is not None and additional_vertices_spl_db is not None:
                                    # Verwende scipy.interpolate.griddata mit linear für unregelmäßige Datenpunkte
                                    try:
                                        from scipy.interpolate import griddata
                                        
                                        # Kombiniere Grid-Punkte und Randpunkte
                                        points_grid = np.column_stack([Xg.ravel(), Yg.ravel()])
                                        values_grid = spl_values_2d.ravel()
                                        
                                        # Filtere Grid-Punkte nach surface_mask
                                        if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                            mask_flat = surface_mask.ravel().astype(bool)
                                            if np.any(mask_flat):
                                                points_grid = points_grid[mask_flat]
                                                values_grid = values_grid[mask_flat]
                                        
                                        # Kombiniere mit Randpunkten
                                        points_combined = np.vstack([
                                            points_grid,
                                            additional_vertices_coords[:, :2]  # Nur X, Y für 2D-Interpolation
                                        ])
                                        values_combined = np.concatenate([
                                            values_grid,
                                            additional_vertices_spl_db
                                        ])
                                        
                                        # Interpoliere auf Vertex-Koordinaten
                                        points_new = triangulated_vertices[:, :2]
                                        spl_at_verts = griddata(
                                            points_combined,
                                            values_combined,
                                            points_new,
                                            method='linear',
                                            fill_value=np.nan,
                                        )
                                        
                                        valid_mask = np.isfinite(spl_at_verts)
                                        
                                        # #region agent log
                                        try:
                                            import json
                                            import time as time_module
                                            nan_count = np.sum(~valid_mask)
                                            nan_indices = np.where(~valid_mask)[0]
                                            nan_vertices = triangulated_vertices[nan_indices] if len(nan_indices) > 0 else None
                                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                                f.write(json.dumps({
                                                    "sessionId": "debug-session",
                                                    "runId": "run1",
                                                    "hypothesisId": "L1",
                                                    "location": "Plot3DSPL.py:_render_surfaces:gradient_interpolation_result",
                                                    "message": "Gradient-Interpolation Ergebnis",
                                                    "data": {
                                                        "surface_id": surface_id,
                                                        "phase_mode": phase_mode,
                                                        "n_vertices": n_vertices,
                                                        "valid_count": np.sum(valid_mask),
                                                        "nan_count": nan_count,
                                                        "has_additional_vertices": additional_vertices_coords is not None,
                                                        "additional_vertices_count": len(additional_vertices_coords) if additional_vertices_coords is not None else 0,
                                                        "grid_points_count": len(points_grid),
                                                        "combined_points_count": len(points_combined),
                                                        "nan_vertices_sample": nan_vertices[:5].tolist() if nan_vertices is not None and len(nan_vertices) > 0 else None,
                                                        "grid_x_range": [float(np.min(points_grid[:, 0])), float(np.max(points_grid[:, 0]))] if len(points_grid) > 0 else None,
                                                        "grid_y_range": [float(np.min(points_grid[:, 1])), float(np.max(points_grid[:, 1]))] if len(points_grid) > 0 else None
                                                    },
                                                    "timestamp": int(time_module.time() * 1000)
                                                }) + "\n")
                                        except Exception:
                                            pass
                                        # #endregion
                                        
                                        if not np.any(valid_mask):
                                            raise ValueError(
                                                f"Surface '{surface_id}': Linear-Interpolation mit Randpunkten liefert keine gültigen Werte"
                                            )
                                    except Exception as e:
                                        # Fallback auf bilineare Interpolation ohne Randpunkte
                                        try:
                                            # Stelle sicher, dass Xg/Yg als reguläres Grid interpretiert werden können
                                            Xg_arr = np.asarray(Xg)
                                            Yg_arr = np.asarray(Yg)
                                            if Xg_arr.ndim != 2 or Yg_arr.ndim != 2:
                                                raise ValueError("X_grid/Y_grid sind nicht 2D - bilineare Interpolation nicht möglich")

                                            # Extrahiere eindeutige sortierte Achsen
                                            x_axis = np.unique(Xg_arr[0, :])
                                            y_axis = np.unique(Yg_arr[:, 0])
                                            if x_axis.size < 2 or y_axis.size < 2:
                                                raise ValueError("Zu wenige Grid-Punkte für bilineare Interpolation")

                                            # Gitterabstände (angenommen äquidistant)
                                            dx = np.diff(x_axis).mean()
                                            dy = np.diff(y_axis).mean()
                                            if dx == 0.0 or dy == 0.0:
                                                raise ValueError("dx/dy = 0 in Grid - bilineare Interpolation nicht möglich")

                                            # Vertex-Koordinaten (x_v, y_v)
                                            vx = triangulated_vertices[:, 0]
                                            vy = triangulated_vertices[:, 1]

                                            # Indizes im Gridraum (Gleitkomma)
                                            ix_float = (vx - x_axis[0]) / dx
                                            iy_float = (vy - y_axis[0]) / dy

                                            # Unteres Gitterfeld (i0,j0) und Abstand (tx, ty) im Feld
                                            nx = x_axis.size
                                            ny = y_axis.size

                                            i0 = np.floor(ix_float).astype(int)
                                            j0 = np.floor(iy_float).astype(int)

                                            # Clipping auf gültigen Bereich [0, nx-2] / [0, ny-2]
                                            i0 = np.clip(i0, 0, nx - 2)
                                            j0 = np.clip(j0, 0, ny - 2)

                                            tx = ix_float - i0
                                            ty = iy_float - j0
                                            tx = np.clip(tx, 0.0, 1.0)
                                            ty = np.clip(ty, 0.0, 1.0)

                                            # Vier Nachbarn pro Vertex
                                            i1 = i0 + 1
                                            j1 = j0 + 1

                                            # Indizes in 2D-Arrays
                                            v00 = spl_values_2d[j0, i0]
                                            v10 = spl_values_2d[j0, i1]
                                            v01 = spl_values_2d[j1, i0]
                                            v11 = spl_values_2d[j1, i1]

                                            # Bilineare Interpolation
                                            # 🎯 FIX: Für Phase-Modus behandle NaN-Werte in Grid-Werten korrekt
                                            # Wenn einer der vier Nachbarn NaN ist, sollte das Ergebnis auch NaN sein (nicht 0)
                                            if phase_mode:
                                                # Prüfe welche Nachbarn gültig sind
                                                valid_00 = np.isfinite(v00)
                                                valid_10 = np.isfinite(v10)
                                                valid_01 = np.isfinite(v01)
                                                valid_11 = np.isfinite(v11)
                                                
                                                # Berechne Interpolation nur für gültige Nachbarn
                                                spl_at_verts = np.full(n_vertices, np.nan, dtype=float)
                                                for v_idx in range(n_vertices):
                                                    if valid_00[v_idx] and valid_10[v_idx] and valid_01[v_idx] and valid_11[v_idx]:
                                                        # Alle Nachbarn gültig → bilineare Interpolation
                                                        spl_at_verts[v_idx] = (
                                                            (1 - tx[v_idx]) * (1 - ty[v_idx]) * v00[v_idx]
                                                            + tx[v_idx] * (1 - ty[v_idx]) * v10[v_idx]
                                                            + (1 - tx[v_idx]) * ty[v_idx] * v01[v_idx]
                                                            + tx[v_idx] * ty[v_idx] * v11[v_idx]
                                                        )
                                                    elif valid_00[v_idx] or valid_10[v_idx] or valid_01[v_idx] or valid_11[v_idx]:
                                                        # Mindestens ein Nachbar gültig → verwende nearest neighbour
                                                        valid_vals = []
                                                        valid_weights = []
                                                        if valid_00[v_idx]:
                                                            valid_vals.append(v00[v_idx])
                                                            valid_weights.append((1 - tx[v_idx]) * (1 - ty[v_idx]))
                                                        if valid_10[v_idx]:
                                                            valid_vals.append(v10[v_idx])
                                                            valid_weights.append(tx[v_idx] * (1 - ty[v_idx]))
                                                        if valid_01[v_idx]:
                                                            valid_vals.append(v01[v_idx])
                                                            valid_weights.append((1 - tx[v_idx]) * ty[v_idx])
                                                        if valid_11[v_idx]:
                                                            valid_vals.append(v11[v_idx])
                                                            valid_weights.append(tx[v_idx] * ty[v_idx])
                                                        if valid_vals:
                                                            # Gewichtete Mittelung der gültigen Nachbarn
                                                            weights_sum = sum(valid_weights)
                                                            if weights_sum > 0:
                                                                spl_at_verts[v_idx] = sum(v * w for v, w in zip(valid_vals, valid_weights)) / weights_sum
                                                    # Sonst bleibt NaN
                                            else:
                                                # Standard bilineare Interpolation (für SPL-Modus)
                                                spl_at_verts = (
                                                    (1 - tx) * (1 - ty) * v00
                                                    + tx * (1 - ty) * v10
                                                    + (1 - tx) * ty * v01
                                                    + tx * ty * v11
                                                )

                                            # Maske anwenden (Punkte außerhalb der Surface -> NaN)
                                            if surface_mask.size == Xg_arr.size and surface_mask.shape == Xg_arr.shape:
                                                mask_flat = surface_mask.ravel().astype(bool)
                                                # Erzeuge Zellmaske im gleichen Verfahren wie oben: nur Zellen mit allen 4 Punkten
                                                cell_mask = (
                                                    mask_flat.reshape(Yg_arr.shape)[j0, i0]
                                                    & mask_flat.reshape(Yg_arr.shape)[j0, i1]
                                                    & mask_flat.reshape(Yg_arr.shape)[j1, i0]
                                                    & mask_flat.reshape(Yg_arr.shape)[j1, i1]
                                                )
                                                spl_at_verts = np.where(cell_mask, spl_at_verts, np.nan)

                                            valid_mask = np.isfinite(spl_at_verts)
                                            
                                            # #region agent log
                                            try:
                                                import json
                                                import time as time_module
                                                nan_count = np.sum(~valid_mask)
                                                nan_indices = np.where(~valid_mask)[0]
                                                nan_vertices = triangulated_vertices[nan_indices] if len(nan_indices) > 0 else None
                                                valid_vertices = triangulated_vertices[valid_mask] if np.any(valid_mask) else None
                                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                                    f.write(json.dumps({
                                                        "sessionId": "debug-session",
                                                        "runId": "run1",
                                                        "hypothesisId": "L2",
                                                        "location": "Plot3DSPL.py:_render_surfaces:bilinear_interpolation_result",
                                                        "message": "Bilineare Interpolation Ergebnis",
                                                        "data": {
                                                            "surface_id": surface_id,
                                                            "phase_mode": phase_mode,
                                                            "n_vertices": n_vertices,
                                                            "valid_count": np.sum(valid_mask),
                                                            "nan_count": nan_count,
                                                            "x_axis_range": [float(x_axis[0]), float(x_axis[-1])] if len(x_axis) > 0 else None,
                                                            "y_axis_range": [float(y_axis[0]), float(y_axis[-1])] if len(y_axis) > 0 else None,
                                                            "nan_vertices_sample": nan_vertices[:5].tolist() if nan_vertices is not None and len(nan_vertices) > 0 else None,
                                                            "valid_vertices_sample": valid_vertices[:5].tolist() if valid_vertices is not None and len(valid_vertices) > 0 else None,
                                                            "spl_values_2d_nan_count": np.sum(~np.isfinite(spl_values_2d)) if spl_values_2d.size > 0 else 0
                                                        },
                                                        "timestamp": int(time_module.time() * 1000)
                                                    }) + "\n")
                                            except Exception:
                                                pass
                                            # #endregion
                                            
                                            if not np.any(valid_mask):
                                                raise ValueError(
                                                    f"Surface '{surface_id}': Bilineare Interpolation liefert keine gültigen Werte"
                                                )
                                        except Exception:
                                            # Letzter Fallback: Nearest-Neighbour
                                            from scipy.interpolate import griddata
                                            points_new = triangulated_vertices[:, :2]
                                            points_orig = np.column_stack([Xg.ravel(), Yg.ravel()])
                                            values_orig = spl_values_2d.ravel()
                                            if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                                mask_flat = surface_mask.ravel().astype(bool)
                                                if np.any(mask_flat):
                                                    points_orig = points_orig[mask_flat]
                                                    values_orig = values_orig[mask_flat]
                                            spl_at_verts = griddata(
                                                points_orig,
                                                values_orig,
                                                points_new,
                                                method='nearest',
                                                fill_value=np.nan,
                                            )
                                            valid_mask = np.isfinite(spl_at_verts)
                                            if not np.any(valid_mask):
                                                raise ValueError(
                                                    f"Surface '{surface_id}': Fallback-Nearest liefert keine gültigen Werte"
                                                )
                                else:
                                    # Standard bilineare Interpolation ohne Randpunkte
                                    try:
                                        # Stelle sicher, dass Xg/Yg als reguläres Grid interpretiert werden können
                                        Xg_arr = np.asarray(Xg)
                                        Yg_arr = np.asarray(Yg)
                                        if Xg_arr.ndim != 2 or Yg_arr.ndim != 2:
                                            raise ValueError("X_grid/Y_grid sind nicht 2D - bilineare Interpolation nicht möglich")

                                        # Extrahiere eindeutige sortierte Achsen
                                        x_axis = np.unique(Xg_arr[0, :])
                                        y_axis = np.unique(Yg_arr[:, 0])
                                        if x_axis.size < 2 or y_axis.size < 2:
                                            raise ValueError("Zu wenige Grid-Punkte für bilineare Interpolation")

                                        # Gitterabstände (angenommen äquidistant)
                                        dx = np.diff(x_axis).mean()
                                        dy = np.diff(y_axis).mean()
                                        if dx == 0.0 or dy == 0.0:
                                            raise ValueError("dx/dy = 0 in Grid - bilineare Interpolation nicht möglich")

                                        # Vertex-Koordinaten (x_v, y_v)
                                        vx = triangulated_vertices[:, 0]
                                        vy = triangulated_vertices[:, 1]

                                        # Indizes im Gridraum (Gleitkomma)
                                        ix_float = (vx - x_axis[0]) / dx
                                        iy_float = (vy - y_axis[0]) / dy

                                        # Unteres Gitterfeld (i0,j0) und Abstand (tx, ty) im Feld
                                        nx = x_axis.size
                                        ny = y_axis.size

                                        i0 = np.floor(ix_float).astype(int)
                                        j0 = np.floor(iy_float).astype(int)

                                        # Clipping auf gültigen Bereich [0, nx-2] / [0, ny-2]
                                        i0 = np.clip(i0, 0, nx - 2)
                                        j0 = np.clip(j0, 0, ny - 2)

                                        tx = ix_float - i0
                                        ty = iy_float - j0
                                        tx = np.clip(tx, 0.0, 1.0)
                                        ty = np.clip(ty, 0.0, 1.0)

                                        # Vier Nachbarn pro Vertex
                                        i1 = i0 + 1
                                        j1 = j0 + 1

                                        # Indizes in 2D-Arrays
                                        v00 = spl_values_2d[j0, i0]
                                        v10 = spl_values_2d[j0, i1]
                                        v01 = spl_values_2d[j1, i0]
                                        v11 = spl_values_2d[j1, i1]

                                        # Bilineare Interpolation
                                        spl_at_verts = (
                                            (1 - tx) * (1 - ty) * v00
                                            + tx * (1 - ty) * v10
                                            + (1 - tx) * ty * v01
                                            + tx * ty * v11
                                        )

                                        # Maske anwenden (Punkte außerhalb der Surface -> NaN)
                                        if surface_mask.size == Xg_arr.size and surface_mask.shape == Xg_arr.shape:
                                            mask_flat = surface_mask.ravel().astype(bool)
                                            # Erzeuge Zellmaske im gleichen Verfahren wie oben: nur Zellen mit allen 4 Punkten
                                            cell_mask = (
                                                mask_flat.reshape(Yg_arr.shape)[j0, i0]
                                                & mask_flat.reshape(Yg_arr.shape)[j0, i1]
                                                & mask_flat.reshape(Yg_arr.shape)[j1, i0]
                                                & mask_flat.reshape(Yg_arr.shape)[j1, i1]
                                            )
                                            spl_at_verts = np.where(cell_mask, spl_at_verts, np.nan)

                                        valid_mask = np.isfinite(spl_at_verts)
                                        if not np.any(valid_mask):
                                            raise ValueError(
                                                f"Surface '{surface_id}': Bilineare Interpolation liefert keine gültigen Werte"
                                            )
                                    except Exception:
                                        # Fallback: Wenn bilinear scheitert, nutze Nearest-Neighbour (wie im Step-Modus)
                                        from scipy.interpolate import griddata
                                        points_new = triangulated_vertices[:, :2]
                                        points_orig = np.column_stack([Xg.ravel(), Yg.ravel()])
                                        values_orig = spl_values_2d.ravel()
                                        if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                            mask_flat = surface_mask.ravel().astype(bool)
                                            if np.any(mask_flat):
                                                points_orig = points_orig[mask_flat]
                                                values_orig = values_orig[mask_flat]
                                        spl_at_verts = griddata(
                                            points_orig,
                                            values_orig,
                                            points_new,
                                            method='nearest',
                                            fill_value=np.nan,
                                        )
                                        valid_mask = np.isfinite(spl_at_verts)
                                        if not np.any(valid_mask):
                                            raise ValueError(
                                                f"Surface '{surface_id}': Fallback-Nearest liefert keine gültigen Werte"
                                            )
                            
                            # 🎯 DEBUG: Prüfe, wie viele Vertices gültige Daten haben
                            
                            # Bestimme Colorbar-Bereich für Visualisierung
                            cbar_min_local = cbar_min
                            cbar_max_local = cbar_max
                            try:
                                debug_min = float(np.nanmin(spl_at_verts))
                                debug_max = float(np.nanmax(spl_at_verts))
                                # Für exakte Werte: verwende den tatsächlichen Bereich als Colorbar
                                cbar_min_local = debug_min
                                cbar_max_local = debug_max
                                if np.isclose(cbar_min_local, cbar_max_local):
                                    cbar_min_local -= 1.0
                                    cbar_max_local += 1.0
                            except Exception:
                                pass
                            
                            # Clipping nur für Visualisierung
                            spl_at_verts_before_clip = spl_at_verts.copy()
                            spl_at_verts = np.clip(spl_at_verts, cbar_min_local, cbar_max_local)
                            
                            # 🎯 SUBDIVISION + MESH-CACHING:
                            # Wir wollen pro Surface genau EIN Mesh erzeugen und anschließend nur noch die
                            # plot_scalars aktualisieren. Das reduziert die Kosten von add_mesh/remove_actor.
                            if not hasattr(self, "_surface_actors") or not isinstance(self._surface_actors, dict):
                                self._surface_actors = {}
                            
                            cached_entry = self._surface_actors.get(surface_id)
                            cached_mesh = None
                            cached_actor = None
                            cached_signature = None
                            if isinstance(cached_entry, dict):
                                cached_mesh = cached_entry.get("mesh")
                                cached_actor = cached_entry.get("actor")
                                cached_signature = cached_entry.get("signature")
                            elif cached_entry is not None:
                                # Rückwärtskompatibilität: alter Code speicherte direkt den Actor
                                cached_actor = cached_entry

                            mesh = None
                            actor = None

                            # Signatur für aktuelles Geometrie-/Subdivision-Setup berechnen
                            current_signature = None
                            try:
                                if triangulated_vertices is not None and triangulated_faces is not None:
                                    v = np.asarray(triangulated_vertices)
                                    current_signature = (
                                        int(PLOT_SUBDIVISION_LEVEL),
                                        int(v.shape[0]),
                                        int(len(triangulated_faces)),
                                        float(np.nanmin(v[:, 0])),
                                        float(np.nanmax(v[:, 0])),
                                        float(np.nanmin(v[:, 1])),
                                        float(np.nanmax(v[:, 1])),
                                        float(np.nanmin(v[:, 2])),
                                        float(np.nanmax(v[:, 2])),
                                    )
                            except Exception:
                                current_signature = None

                            # Versuche, ein vorhandenes Mesh wiederzuverwenden
                            if (
                                cached_mesh is not None
                                and cached_signature is not None
                                and current_signature is not None
                                and cached_signature == current_signature
                                and cached_mesh.n_points == spl_at_verts.size
                            ):
                                try:
                                    mesh = cached_mesh
                                    mesh["plot_scalars"] = spl_at_verts
                                    actor = cached_actor
                                except Exception:
                                    mesh = None
                                    actor = None

                            # Falls kein geeignetes Mesh vorhanden ist, neu erzeugen (inkl. optionaler Subdivision)
                            if mesh is None:
                                # 🎯 SUBDIVISION: Unterteile jedes Dreieck in 4 kleinere für schärfere Plots
                                # subdivision_level: 0=keine, 1=4x Faces, 2=16x Faces, 3=64x Faces, etc.
                                subdivision_level = max(0, min(PLOT_SUBDIVISION_LEVEL, 3))  # Limit auf 0-3 für Performance
                                
                                if subdivision_level > 0:
                                    try:
                                        current_vertices = triangulated_vertices
                                        current_faces = triangulated_faces
                                        current_scalars = spl_at_verts
                                        
                                        # Iterative Subdivision (mehrfache Anwendung für höhere Level)
                                        for level in range(subdivision_level):
                                            current_vertices, current_faces, current_scalars = self._subdivide_triangles(
                                                current_vertices, current_faces, current_scalars
                                            )
                                        
                                        
                                        # Erstelle PyVista PolyData Mesh mit subdividierten Daten
                                        mesh = pv.PolyData(current_vertices, current_faces)
                                        mesh["plot_scalars"] = current_scalars
                                    except Exception as e_subdiv:
                                        raise RuntimeError(f"Surface '{surface_id}': Subdivision fehlgeschlagen: {e_subdiv}")
                                else:
                                    # Keine Subdivision
                                    mesh = pv.PolyData(triangulated_vertices, triangulated_faces)
                                    mesh["plot_scalars"] = spl_at_verts

                            # Versuche zunächst, vorhandene NaN-Ergebnisse aus der Berechnung zu verwenden,
                            # bevor neue Refinement-Requests erzeugt werden.
                            try:
                                calc_spl = getattr(self.container, "calculation_spl", None)
                                extra_nan_data = None
                                if isinstance(calc_spl, dict):
                                    surface_results = calc_spl.get("surface_results", {})
                                    res_entry = surface_results.get(surface_id, {})
                                    extra_nan_data = res_entry.get("nan_vertices")
                                if extra_nan_data is not None:
                                    coords_extra = np.asarray(extra_nan_data.get("coords", []), dtype=float)
                                    p_extra = np.asarray(extra_nan_data.get("sound_field_p", []), dtype=complex)
                                    if coords_extra.size and p_extra.size and coords_extra.shape[0] == p_extra.shape[0]:
                                        from scipy.spatial import cKDTree
                                        tree = cKDTree(coords_extra)
                                        vertex_coords = np.asarray(mesh.points, dtype=float)
                                        nan_local = ~np.isfinite(mesh["plot_scalars"])
                                        if np.any(nan_local):
                                            nan_indices = np.where(nan_local)[0]
                                            _, idx = tree.query(vertex_coords[nan_indices], k=1)
                                            # Konvertiere komplexen Druck zu SPL in dB
                                            spl_extra_db = self.functions.mag2db(np.abs(p_extra[idx]))
                                            mesh["plot_scalars"][nan_indices] = spl_extra_db
                            except Exception:
                                # Fallback: Wenn etwas schiefgeht, arbeiten wir mit den
                                # ursprünglichen Werten weiter und erzeugen unten Requests.
                                pass

                            # Debug-Ausgabe: Prüfe auf NaN/Inf in den Vertex-Skalaren (Debug-Logging entfernt)
                            
                            # Falls wir keinen Actor wiederverwenden konnten, jetzt neu hinzufügen
                            if actor is None:
                                # Entferne alten Texture-Actor (falls noch vorhanden)
                                old_texture_data = self._surface_texture_actors.get(surface_id)
                                if old_texture_data is not None:
                                    old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                                    if old_actor is not None:
                                        try:
                                            self.plotter.remove_actor(old_actor)
                                        except Exception:
                                            pass
                                
                                actor_name = f"{self.SURFACE_NAME}_tri_{surface_id}"
                                actor = self.plotter.add_mesh(
                                    mesh,
                                    name=actor_name,
                                    scalars="plot_scalars",
                                    cmap=cmap_object,
                                    clim=(cbar_min, cbar_max),
                                    smooth_shading=False,  # Kein Smooth Shading für schärfere Plots nach Subdivision
                                    show_scalar_bar=False,
                                    reset_camera=False,
                                    interpolate_before_map=False,  # Keine Interpolation - verwende exakte Vertex-Werte
                                )
                            
                            # 🎯 Für schärfere Plots: Deaktiviere Interpolation (auch außerhalb step_mode)
                            # Nach Subdivision haben wir bereits mehr Polygone, daher keine zusätzliche Interpolation nötig
                            if hasattr(actor, "prop") and actor.prop is not None:
                                try:
                                    actor.prop.interpolation = "flat"  # Flat shading für schärfere Kanten
                                except Exception:
                                    pass
                            
                            if hasattr(actor, 'SetPickable'):
                                actor.SetPickable(False)
                            
                            # Speichere Actor UND Mesh in _surface_actors (nicht _surface_texture_actors)
                            if not hasattr(self, '_surface_actors') or not isinstance(self._surface_actors, dict):
                                self._surface_actors = {}
                            self._surface_actors[surface_id] = {"actor": actor, "mesh": mesh}
                            
                            # Entferne aus texture_actors falls vorhanden
                            self._surface_texture_actors.pop(surface_id, None)
                            
                            surfaces_processed += 1
                            continue  # Überspringe Texture-Pfad
                            
                    except Exception as e:
                        use_triangulation = False
                
                # Triangulation muss erfolgreich sein
                if not use_triangulation:
                    raise ValueError(f"Surface '{surface_id}': Triangulation fehlgeschlagen – kein Plot möglich")
            except Exception as e:
                pass

    @measure_time("PlotSPL3D.update_spl_plot")
    def update_spl_plot(
        self,
        sound_field_x: Iterable[float],
        sound_field_y: Iterable[float],
        sound_field_pressure: Iterable[float],
        colorization_mode: str = "Gradient",
        ):
        """Aktualisiert die SPL-Fläche."""
        t_start_total = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0

        # 🎯 GEOMETRIE-VERSION: Wenn sich die Geometrie geändert hat, alten Mesh-Cache verwerfen
        try:
            current_geom_version = int(getattr(self.settings, "geometry_version", 0))
        except Exception:
            current_geom_version = 0
        last_geom_version = getattr(self, "_last_geometry_version", None)

        
        if last_geom_version is None or last_geom_version != current_geom_version:
            # Surface-Mesh-Cache leeren, damit bei neuer Geometrie alle Meshes neu aufgebaut werden

            if hasattr(self, "_surface_actors") and isinstance(self._surface_actors, dict):
                try:
                    for sid, entry in list(self._surface_actors.items()):
                        actor = entry.get("actor") if isinstance(entry, dict) else entry
                        if actor is not None and hasattr(self, "plotter") and self.plotter is not None:
                            try:
                                self.plotter.remove_actor(actor)
                            except Exception:
                                pass
                    self._surface_actors.clear()
                except Exception:
                    self._surface_actors = {}
            self._last_geometry_version = current_geom_version

        camera_state = self._camera_state or self._capture_camera()

        # Prüfe ob Phase-Modus aktiv ist
        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'

        surface_overrides: dict[str, dict[str, np.ndarray]] = {}
        calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
        surface_grids_data = {}
        surface_results_data = {}
        if isinstance(calc_spl, dict):
            surface_grids_data = calc_spl.get("surface_grids", {}) or {}
            surface_results_data = calc_spl.get("surface_results", {}) or {}

        # 🎯 FIX: Hole globale Phase-Daten für Phase-Modus
        global_phase_data = None
        global_phase_x = None
        global_phase_y = None
        if phase_mode and isinstance(calc_spl, dict):
            phase_diff = calc_spl.get('sound_field_phase_diff')
            phase_x = calc_spl.get('sound_field_x')
            phase_y = calc_spl.get('sound_field_y')
            if phase_diff is not None and phase_x is not None and phase_y is not None:
                try:
                    global_phase_data = np.asarray(phase_diff, dtype=float)
                    global_phase_x = np.asarray(phase_x, dtype=float)
                    global_phase_y = np.asarray(phase_y, dtype=float)
                    # Stelle sicher, dass global_phase_data 2D ist
                    if global_phase_data.ndim == 1 and len(global_phase_x) > 0 and len(global_phase_y) > 0:
                        try:
                            global_phase_data = global_phase_data.reshape(len(global_phase_y), len(global_phase_x))
                        except Exception:
                            global_phase_data = None
                except Exception:
                    global_phase_data = None

        if isinstance(surface_grids_data, dict) and surface_grids_data and isinstance(surface_results_data, dict) and surface_results_data:
            orientations_found = {}
            for sid, grid_data in surface_grids_data.items():
                orientation = grid_data.get("orientation", "").lower() if isinstance(grid_data, dict) else ""
                if orientation not in orientations_found:
                    orientations_found[orientation] = []
                orientations_found[orientation].append(sid)
                
                if sid not in surface_results_data:
                    continue
                try:
                    Xg = np.asarray(grid_data.get("X_grid", []))
                    Yg = np.asarray(grid_data.get("Y_grid", []))
                    
                    # 🎯 FIX: Prüfe auch auf additional_vertices_spl für Surfaces mit leeren Grids
                    result_data_check = surface_results_data.get(sid, {})
                    has_additional_vertices_spl_check = result_data_check.get('additional_vertices_spl') is not None
                    
                    if (Xg.size == 0 or Yg.size == 0) and not has_additional_vertices_spl_check:
                        continue
                    
                    result_data = surface_results_data[sid]
                    
                    # 🎯 FIX: Für Surfaces mit leeren Grids aber mit additional_vertices_spl
                    # erstelle surface_overrides mit Dummy-Daten direkt
                    if Xg.size == 0 or Yg.size == 0:
                        # Surface hat leere Grids, aber additional_vertices_spl ist vorhanden
                        # Erstelle Dummy-Daten als Marker (die eigentliche Triangulation kommt aus surface_grids_data)
                        surface_overrides[sid] = {
                            "source_x": np.array([0.0], dtype=float),
                            "source_y": np.array([0.0], dtype=float),
                            "values": np.array([[0.0]], dtype=float),
                        }
                        continue  # Überspringe weitere Verarbeitung für diesen Fall
                    
                    if Xg.ndim == 2 and Yg.ndim == 2:
                        # Achsen aus dem strukturierten Grid ableiten
                        gx = Xg[0, :] if Xg.shape[1] > 0 else Xg.ravel()
                        gy = Yg[:, 0] if Yg.shape[0] > 0 else Yg.ravel()
                    else:
                        
                        continue

                    sound_field_p_complex = np.array(result_data.get('sound_field_p', []), dtype=complex)
                    
                    # 🎯 FIX: Prüfe auch auf additional_vertices_spl für Surfaces mit 0 Grid-Punkten
                    has_additional_vertices_spl = result_data.get('additional_vertices_spl') is not None
                    
                    # 🎯 FIX: Prüfe, ob sound_field_p_complex nur Nullen/NaNs enthält (keine aktiven Grid-Punkte)
                    # Dies kann passieren, wenn Xg.size != 0, aber alle Grid-Punkte außerhalb des Polygons sind
                    has_valid_grid_points = False
                    if sound_field_p_complex.size > 0:
                        # Prüfe, ob es gültige (nicht-Null, nicht-NaN) Werte gibt
                        valid_mask = np.isfinite(sound_field_p_complex) & (np.abs(sound_field_p_complex) > 1e-12)
                        has_valid_grid_points = np.any(valid_mask)
                    
                    # Wenn keine gültigen Grid-Punkte vorhanden sind, aber additional_vertices_spl vorhanden ist,
                    # verwende den Dummy-Pfad (wie bei leeren Grids)
                    if not has_valid_grid_points and has_additional_vertices_spl:
                        # Surface hat keine gültigen Grid-Punkte, aber additional_vertices_spl ist vorhanden
                        # Erstelle Dummy-Daten als Marker (die eigentliche Triangulation kommt aus surface_grids_data)
                        surface_overrides[sid] = {
                            "source_x": np.array([0.0], dtype=float),
                            "source_y": np.array([0.0], dtype=float),
                            "values": np.array([[0.0]], dtype=float),
                        }
                        continue  # Überspringe weitere Verarbeitung für diesen Fall
                    
                    if sound_field_p_complex.size == 0 and not has_additional_vertices_spl:
                        continue

                    # 🎯 FIX: Im Phase-Modus verwende Phase-Daten statt Druck-Daten
                    surface_values = None
                    use_phase_data = False
                    if phase_mode and global_phase_data is not None:
                        # Interpoliere Phase-Daten von globalem Grid auf Surface-Grid
                        from scipy.interpolate import griddata
                        # Erstelle Quell-Grid (globales Grid)
                        X_global, Y_global = np.meshgrid(global_phase_x, global_phase_y)
                        points_source = np.column_stack([X_global.ravel(), Y_global.ravel()])
                        values_source = global_phase_data.ravel()
                        
                        # Ziel-Punkte auf Surface-Grid
                        points_target = np.column_stack([Xg.ravel(), Yg.ravel()])
                        
                        # Interpoliere Phase-Daten
                        try:
                            phase_values_interp = griddata(
                                points_source, values_source, points_target,
                                method='linear', fill_value=np.nan
                            )
                            phase_values_2d = phase_values_interp.reshape(Xg.shape)
                            # 🎯 FIX: Für Phase-Modus behalte NaN-Werte (nicht auf 0.0 setzen)
                            # NaN-Werte werden transparent dargestellt, 0.0 würde als 0° angezeigt werden
                            # Setze nur inf/neginf auf 0, aber behalte NaN
                            phase_values_2d = np.where(np.isinf(phase_values_2d), 0.0, phase_values_2d)
                            surface_values = phase_values_2d  # Behalte NaN-Werte für transparente Darstellung
                            use_phase_data = True
                        except Exception:
                            # Falls Interpolation fehlschlägt, verwende Standard-Pfad
                            pass

                    # 🎯 FIX: Im Phase-Modus überspringe Shape-Prüfung für sound_field_p_complex
                    if not use_phase_data and sound_field_p_complex.shape != Xg.shape:
                        # Prüfe ob es Gruppen-Grid-Daten gibt (für Gruppen-Surfaces)
                        group_X_grid = result_data.get('group_X_grid')
                        group_Y_grid = result_data.get('group_Y_grid')
                        group_mask = result_data.get('group_mask')
                        
                        if group_X_grid is not None and group_Y_grid is not None:
                            # Verwende Gruppen-Grid-Daten
                            group_X = np.asarray(group_X_grid, dtype=float)
                            group_Y = np.asarray(group_Y_grid, dtype=float)
                            if group_X.shape == sound_field_p_complex.shape and group_Y.shape == sound_field_p_complex.shape:
                                # Interpoliere von Gruppen-Grid auf Surface-Grid
                                from scipy.interpolate import griddata
                                group_mask_array = np.asarray(group_mask, dtype=bool) if group_mask is not None else np.ones_like(sound_field_p_complex, dtype=bool)
                                
                                # Erstelle Interpolations-Punkte (nur gültige Punkte)
                                valid_mask = group_mask_array & np.isfinite(sound_field_p_complex)
                                if np.any(valid_mask):
                                    points_source = np.column_stack([group_X[valid_mask].ravel(), group_Y[valid_mask].ravel()])
                                    values_source = sound_field_p_complex[valid_mask].ravel()
                                    
                                    # Ziel-Punkte auf Surface-Grid
                                    points_target = np.column_stack([Xg.ravel(), Yg.ravel()])
                                    
                                    # Interpoliere
                                    sound_field_p_interp = griddata(
                                        points_source, values_source, points_target,
                                        method='linear', fill_value=0.0
                                    )
                                    sound_field_p_complex = sound_field_p_interp.reshape(Xg.shape)
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Für einzelne Surfaces: Versuche direkte Reshape oder Interpolation
                            # Wenn die Gesamtgröße übereinstimmt, versuche Reshape
                            if sound_field_p_complex.size == Xg.size:
                                try:
                                    sound_field_p_complex = sound_field_p_complex.reshape(Xg.shape)
                                except Exception:
                                    # Reshape fehlgeschlagen, versuche Interpolation
                                    from scipy.interpolate import griddata
                                    # Erstelle Quell-Grid (muss aus surface_grids_data kommen)
                                    # Für jetzt: Skip, da wir keine Quell-Koordinaten haben
                                    continue
                            else:
                                # Größe stimmt nicht überein, kann nicht interpoliert werden ohne Quell-Koordinaten
                                continue
                    
                    # 🎯 FIX: Für Surfaces mit 0 Grid-Punkten aber mit additional_vertices_spl
                    # erstelle surface_overrides mit Dummy-Daten (als Marker, da Triangulation aus surface_grids_data kommt)
                    if use_phase_data and surface_values is not None:
                        # Verwende interpolierte Phase-Daten (Phase-Modus)
                        # #region agent log
                        try:
                            import json
                            import time as time_module
                            surface_values_valid = surface_values[np.isfinite(surface_values)] if surface_values.size > 0 else np.array([])
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "K2",
                                    "location": "Plot3DSPL.py:update_spl_plot:surface_phase_override",
                                    "message": "Surface Phase-Override erstellt",
                                    "data": {
                                        "surface_id": sid,
                                        "surface_values_shape": list(surface_values.shape),
                                        "surface_values_valid_count": len(surface_values_valid),
                                        "surface_values_min": float(np.nanmin(surface_values_valid)) if len(surface_values_valid) > 0 else None,
                                        "surface_values_max": float(np.nanmax(surface_values_valid)) if len(surface_values_valid) > 0 else None,
                                        "surface_values_mean": float(np.nanmean(surface_values_valid)) if len(surface_values_valid) > 0 else None
                                    },
                                    "timestamp": int(time_module.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        surface_overrides[sid] = {
                            "source_x": np.asarray(gx, dtype=float),
                            "source_y": np.asarray(gy, dtype=float),
                            "values": surface_values,  # Phase-Daten in Grad
                        }
                    elif sound_field_p_complex.size == 0:
                        # Surface hat keine Grid-Punkte, aber additional_vertices_spl ist vorhanden
                        # Erstelle Dummy-Daten als Marker (die eigentliche Triangulation kommt aus surface_grids_data)
                        surface_overrides[sid] = {
                            "source_x": np.array([0.0], dtype=float),
                            "source_y": np.array([0.0], dtype=float),
                            "values": np.array([[0.0]], dtype=float),
                        }
                    else:
                        # Konvertiere komplexe Werte zu dB (wie im neuen Modul)
                        pressure_magnitude = np.abs(sound_field_p_complex)
                        pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                        spl_values_db = self.functions.mag2db(pressure_magnitude)
                        spl_values_db = np.nan_to_num(spl_values_db, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        surface_overrides[sid] = {
                            "source_x": np.asarray(gx, dtype=float),
                            "source_y": np.asarray(gy, dtype=float),
                            "values": spl_values_db,  # Direkt die berechneten Werte verwenden
                        }
                except Exception as e:
                    # Wenn etwas schiefgeht, einfach ohne Override weiterarbeiten
                    continue
        
        # Wenn keine gültigen SPL-Daten vorliegen, belassen wir die bestehende Szene
        # (inkl. Lautsprechern) unverändert und brechen nur das SPL-Update ab.
        # ABER: Wenn surface_overrides vorhanden sind, müssen wir trotzdem plotten!
        has_surface_overrides = bool(surface_overrides)
        is_valid_global_data = self._has_valid_data(sound_field_x, sound_field_y, sound_field_pressure)

        if not is_valid_global_data:
            if not has_surface_overrides:
                
                return
            # Wenn nur surface_overrides vorhanden sind, verwende Dummy-Daten für globale Plot-Geometrie
            # Erstelle minimale Dummy-Daten für die globale Plot-Geometrie
            sound_field_x = np.array([-1.0, 1.0])
            sound_field_y = np.array([-1.0, 1.0])
            sound_field_pressure = np.array([[0.0, 0.0], [0.0, 0.0]])

        x = np.asarray(sound_field_x, dtype=float)
        y = np.asarray(sound_field_y, dtype=float)

        try:
            pressure = np.asarray(sound_field_pressure, dtype=float)
            
            # #region agent log
            try:
                import json
                import time as time_module
                plot_mode_check = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
                phase_mode_check = plot_mode_check == 'Phase alignment'
                pressure_valid = pressure[np.isfinite(pressure)] if pressure.size > 0 else np.array([])
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "K1",
                        "location": "Plot3DSPL.py:update_spl_plot:pressure_loaded",
                        "message": "Pressure-Daten geladen (können Phase-Daten sein)",
                        "data": {
                            "phase_mode": phase_mode_check,
                            "pressure_shape": list(pressure.shape) if pressure.size > 0 else None,
                            "pressure_size": pressure.size,
                            "pressure_valid_count": len(pressure_valid),
                            "pressure_min": float(np.nanmin(pressure_valid)) if len(pressure_valid) > 0 else None,
                            "pressure_max": float(np.nanmax(pressure_valid)) if len(pressure_valid) > 0 else None,
                            "pressure_mean": float(np.nanmean(pressure_valid)) if len(pressure_valid) > 0 else None,
                            "x_len": len(x),
                            "y_len": len(y)
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
        except Exception as e:  # noqa: BLE001
            
            # Ungültige SPL-Daten → Szene nicht leeren, nur SPL-Update abbrechen
            return

        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'
        time_mode = plot_mode == 'SPL over time'
        self._phase_mode_active = phase_mode
        self._time_mode_active = time_mode

        if pressure.ndim != 2:
            if pressure.size == (len(y) * len(x)):
                try:
                    pressure = pressure.reshape(len(y), len(x))
                except Exception:  # noqa: BLE001
                    if not has_surface_overrides:
                        return
            else:
                if not has_surface_overrides:
                    return

        if pressure.shape != (len(y), len(x)):
            
            if not has_surface_overrides:
                
                return

        # Aktualisiere ColorbarManager Modes
        self.colorbar_manager.update_modes(phase_mode_active=phase_mode, time_mode_active=time_mode)
        self.colorbar_manager.set_override(None)

        if time_mode:
            pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                if not has_surface_overrides:
                    return
            # Verwende feste Colorbar-Parameter (keine dynamische Skalierung)
            plot_values = pressure
        elif phase_mode:
            # #region agent log
            try:
                import json
                import time as time_module
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    pressure_arr = np.asarray(pressure)
                    valid_before = pressure_arr[np.isfinite(pressure_arr)]
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "Plot3DSPL.py:update_spl_plot:phase_mode_before",
                        "message": "Phase-Modus: Vor Verarbeitung",
                        "data": {
                            "pressure_shape": list(pressure_arr.shape),
                            "pressure_size": pressure_arr.size,
                            "valid_count_before": len(valid_before),
                            "pressure_min": float(np.nanmin(valid_before)) if len(valid_before) > 0 else None,
                            "pressure_max": float(np.nanmax(valid_before)) if len(valid_before) > 0 else None
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 FIX: Für Phase-Modus behalte NaN-Werte (nicht auf 0 setzen)
            # NaN bedeutet "keine gültigen Daten" und sollte transparent dargestellt werden
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                if not has_surface_overrides:
                    return
            # Behalte NaN-Werte für Phase-Daten (werden später beim Rendering transparent dargestellt)
            phase_values = np.where(np.isfinite(pressure), pressure, np.nan)
            # Setze nur inf/neginf auf 0, aber behalte NaN
            phase_values = np.where(np.isinf(phase_values), 0.0, phase_values)
            plot_values = phase_values
            
            # #region agent log
            try:
                valid_phase = phase_values[np.isfinite(phase_values)]
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "Plot3DSPL.py:update_spl_plot:phase_mode_after",
                        "message": "Phase-Modus: Nach Verarbeitung",
                        "data": {
                            "phase_values_shape": list(phase_values.shape),
                            "valid_count_after": len(valid_phase),
                            "phase_min": float(np.nanmin(valid_phase)) if len(valid_phase) > 0 else None,
                            "phase_max": float(np.nanmax(valid_phase)) if len(valid_phase) > 0 else None,
                            "phase_mean": float(np.nanmean(valid_phase)) if len(valid_phase) > 0 else None
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 DEBUG: Ausgabe der Phase-Daten-Statistik
            if getattr(self.settings, "fem_debug_logging", True):
                try:
                    if len(valid_phase) > 0:
                        phase_min = float(np.nanmin(valid_phase))
                        phase_max = float(np.nanmax(valid_phase))
                        phase_mean = float(np.nanmean(valid_phase))
                        print(f"[Phase Plot] Phase-Daten: min={phase_min:.2f}°, max={phase_max:.2f}°, mean={phase_mean:.2f}°, valid={len(valid_phase)}/{phase_values.size}")
                    else:
                        print(f"[Phase Plot] Keine gültigen Phase-Daten (alle NaN)")
                except Exception as e:
                    print(f"[Phase Plot] Fehler bei Debug-Ausgabe: {e}")
        else:
            pressure_2d = np.nan_to_num(np.abs(pressure), nan=0.0, posinf=0.0, neginf=0.0)
            pressure_2d = np.clip(pressure_2d, 1e-12, None)
            spl_db = self.functions.mag2db(pressure_2d)
            if getattr(self.settings, "fem_debug_logging", True):
                try:
                    p_min = float(np.nanmin(pressure_2d))
                    p_max = float(np.nanmax(pressure_2d))
                    p_mean = float(np.nanmean(pressure_2d))
                    spl_min = float(np.nanmin(spl_db))
                    spl_max = float(np.nanmax(spl_db))
                    spl_mean = float(np.nanmean(spl_db))
                except Exception:
                    pass
                try:
                    x_idx = int(np.argmin(np.abs(x - 0.0)))
                    target_distances = (1.0, 2.0, 10.0, 20.0)
                    for distance in target_distances:
                        y_idx = int(np.argmin(np.abs(y - distance)))
                        value = float(spl_db[y_idx, x_idx])
                        actual_y = float(y[y_idx])
                except Exception:
                    pass
            
            finite_mask = np.isfinite(spl_db)
            if not np.any(finite_mask):
                if not has_surface_overrides:
                    return
            plot_values = spl_db
        
        if DEBUG_PLOT3D_TIMING:
            t_after_spl = time.perf_counter()

        colorbar_params = self.colorbar_manager.get_colorbar_params(phase_mode)
        cbar_min = colorbar_params['min']
        cbar_max = colorbar_params['max']
        cbar_step = colorbar_params['step']
        tick_step = colorbar_params['tick_step']
        self.colorbar_manager.set_override(colorbar_params)
        
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                valid_plot = plot_values[np.isfinite(plot_values)] if hasattr(plot_values, '__len__') else np.array([])
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "Plot3DSPL.py:update_spl_plot:colorbar_params",
                    "message": "Colorbar-Parameter gesetzt",
                    "data": {
                        "phase_mode": phase_mode,
                        "cbar_min": cbar_min,
                        "cbar_max": cbar_max,
                        "cbar_step": cbar_step,
                        "tick_step": tick_step,
                        "plot_values_shape": list(plot_values.shape) if hasattr(plot_values, 'shape') else None,
                        "plot_values_valid_count": len(valid_plot),
                        "plot_values_min": float(np.nanmin(valid_plot)) if len(valid_plot) > 0 else None,
                        "plot_values_max": float(np.nanmax(valid_plot)) if len(valid_plot) > 0 else None
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion

        # 🎯 WICHTIG: Speichere ursprüngliche Berechnungs-Werte für PyVista sample-Modus VOR Clipping!
        # Clipping ändert die Werte und sollte nur für Visualisierung erfolgen, nicht für Sampling
        original_plot_values = plot_values.copy() if hasattr(plot_values, 'copy') else plot_values
        
        # 🎯 Cache original_plot_values für Debug-Vergleiche
        self._cached_original_plot_values = original_plot_values
        
        # 🎯 Validierung: Stelle sicher, dass original_plot_values die korrekte Shape hat
        if original_plot_values.shape != (len(y), len(x)):
            # Wenn surface_overrides vorhanden sind, erstelle eine Dummy-Geometrie
            if has_surface_overrides:
                # Erstelle minimale Dummy-Geometrie für die Validierung
                original_plot_values = np.zeros((len(y), len(x)), dtype=float)
                plot_values = np.zeros((len(y), len(x)), dtype=float)
            else:
                raise ValueError(
                    f"original_plot_values Shape stimmt nicht überein: "
                    f"erhalten {original_plot_values.shape}, erwartet ({len(y)}, {len(x)})"
                )

        # Clipping nur für Visualisierung (nicht für Sampling)
        if time_mode:
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
        elif phase_mode:
            # #region agent log
            try:
                import json
                import time as time_module
                valid_before_clip = plot_values[np.isfinite(plot_values)]
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "Plot3DSPL.py:update_spl_plot:clip_before",
                        "message": "Vor Clipping",
                        "data": {
                            "cbar_min": cbar_min,
                            "cbar_max": cbar_max,
                            "plot_values_min": float(np.nanmin(valid_before_clip)) if len(valid_before_clip) > 0 else None,
                            "plot_values_max": float(np.nanmax(valid_before_clip)) if len(valid_before_clip) > 0 else None
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 FIX: Für Phase-Modus behalte NaN-Werte beim Clipping
            # Clip nur gültige Werte, behalte NaN
            valid_mask = np.isfinite(plot_values)
            if np.any(valid_mask):
                plot_values_clipped = np.clip(plot_values[valid_mask], cbar_min, cbar_max)
                plot_values = plot_values.copy()
                plot_values[valid_mask] = plot_values_clipped
            # NaN-Werte bleiben NaN (werden transparent dargestellt)
            
            # #region agent log
            try:
                valid_after_clip = plot_values[np.isfinite(plot_values)]
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "Plot3DSPL.py:update_spl_plot:clip_after",
                        "message": "Nach Clipping",
                        "data": {
                            "plot_values_min": float(np.nanmin(valid_after_clip)) if len(valid_after_clip) > 0 else None,
                            "plot_values_max": float(np.nanmax(valid_after_clip)) if len(valid_after_clip) > 0 else None,
                            "clipped_count": len(valid_before_clip) - len(valid_after_clip) if len(valid_before_clip) > 0 else 0
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
        else:
            plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)
        
        # 🎯 Gradient-Modus: Normales Upscaling (Performance-Optimierung)
        # Bilineare Interpolation sorgt bereits für glatte Übergänge, daher kein zusätzliches Upscaling nötig
        colorization_mode_used = colorization_mode
        is_step_mode = colorization_mode_used == 'Color step' and cbar_step > 0
        # Verwende normales Upscaling für beide Modi (Performance)
        upscale_factor = self.UPSCALE_FACTOR
        
        # 🎯 PRÜFE: Gibt es horizontale Surfaces, die eine Maske benötigen?
        # Vertikale Flächen haben ihre eigenen Masken in surface_grids_data und benötigen keine globale Geometrie
        has_horizontal_surfaces = False
        if isinstance(surface_grids_data, dict):
            for sid, grid_data in surface_grids_data.items():
                orientation = grid_data.get("orientation", "").lower() if isinstance(grid_data, dict) else ""
                if orientation in ("planar", "sloped"):
                    has_horizontal_surfaces = True
                    break
        
        # Wenn nur vertikale Flächen vorhanden sind, erstelle direkt Dummy-Geometrie
        if has_surface_overrides and not has_horizontal_surfaces:
            try:
                from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import PlotSurfaceGeometry
                dummy_x = np.array([-1.0, 1.0])
                dummy_y = np.array([-1.0, 1.0])
                dummy_values = np.array([[0.0, 0.0], [0.0, 0.0]])
                dummy_mask = np.ones_like(dummy_values, dtype=bool)
                dummy_z = np.zeros_like(dummy_values)
                
                geometry = PlotSurfaceGeometry(
                    plot_x=dummy_x,
                    plot_y=dummy_y,
                    plot_values=dummy_values,
                    z_coords=dummy_z,
                    surface_mask=dummy_mask,
                    source_x=dummy_x,
                    source_y=dummy_y,
                    requires_resample=False,
                    was_upscaled=False,
                )
            except Exception as e_dummy:
                import traceback
                traceback.print_exc()
                return
        else:
            # Normale Geometrie-Erzeugung für horizontale Surfaces
            try:
                geometry = prepare_plot_geometry(
                    x,
                    y,
                    plot_values,
                    settings=self.settings,
                    container=self.container,
                    default_upscale=upscale_factor,
                )
            except RuntimeError as exc:
                # Fehler bei der Geometrie-Erzeugung → keine Szene leeren, nur SPL-Update abbrechen
                # Vertikale SPL-Flächen werden ebenfalls nicht angepasst.
                if has_surface_overrides:
                    # Erstelle minimale Dummy-Geometrie für surface_overrides
                    # WICHTIG: prepare_plot_geometry benötigt eine surface_mask, die bei Dummy-Daten nicht existiert
                    # Daher erstellen wir die Geometrie direkt ohne prepare_plot_geometry
                    try:
                        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import PlotSurfaceGeometry
                        # Erstelle minimale gültige Dummy-Daten mit surface_mask
                        dummy_x = np.array([-1.0, 1.0])
                        dummy_y = np.array([-1.0, 1.0])
                        dummy_values = np.array([[0.0, 0.0], [0.0, 0.0]])
                        dummy_mask = np.ones_like(dummy_values, dtype=bool)  # Alle Punkte aktiv
                        dummy_z = np.zeros_like(dummy_values)
                        
                        geometry = PlotSurfaceGeometry(
                            plot_x=dummy_x,
                            plot_y=dummy_y,
                            plot_values=dummy_values,
                            z_coords=dummy_z,
                            surface_mask=dummy_mask,
                            source_x=dummy_x,
                            source_y=dummy_y,
                            requires_resample=False,
                            was_upscaled=False,
                        )
                    except Exception as e_dummy:
                        import traceback
                        traceback.print_exc()
                        return
                else:
                    return

        if DEBUG_PLOT3D_TIMING:
            t_after_geom = time.perf_counter()

        plot_x = geometry.plot_x
        plot_y = geometry.plot_y
        plot_values = geometry.plot_values
        z_coords = geometry.z_coords
        
        # 🎯 Cache Plot-Geometrie für Click-Handling
        self._plot_geometry_cache = {
            'plot_x': plot_x.copy() if hasattr(plot_x, 'copy') else plot_x,
            'plot_y': plot_y.copy() if hasattr(plot_y, 'copy') else plot_y,
            'source_x': geometry.source_x.copy() if hasattr(geometry.source_x, 'copy') else geometry.source_x,
            'source_y': geometry.source_y.copy() if hasattr(geometry.source_y, 'copy') else geometry.source_y,
        }
        
        # 🎯 is_step_mode wurde bereits oben definiert, verwende es hier
        if is_step_mode:
            scalars = self._quantize_to_steps(plot_values, cbar_step)
        else:
            scalars = plot_values
        
        mesh = build_surface_mesh(
            plot_x,
            plot_y,
            scalars,
            z_coords=z_coords,
            surface_mask=geometry.surface_mask,
            pv_module=pv,
            settings=self.settings,
            container=self.container,
            source_x=geometry.source_x,
            source_y=geometry.source_y,
            source_scalars=None,
        )
        
        if time_mode:
            cmap_object = 'RdBu_r'
        elif phase_mode:
            cmap_object = self.PHASE_CMAP
        else:
            cmap_object = 'jet'

        # ------------------------------------------------------------------
        # Horizontale SPL-Surfaces - nur Texture-Modus
        # ------------------------------------------------------------------
        # Kombiniertes Mesh für Click-Handling behalten
        if mesh is not None and mesh.n_points > 0:
            self.surface_mesh = mesh.copy(deep=True)

        # Entferne alten Basis-Actor für den kombinierten SPL-Teppich (falls vorhanden)
        base_actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
        if base_actor is not None:
            try:
                self.plotter.remove_actor(base_actor)
            except Exception:
                pass
        
        # 🎯 Entferne graue Fläche für enabled Surfaces aus dem leeren Plot (falls vorhanden)
        # Dies muss VOR dem Zeichnen der neuen SPL-Daten passieren
        if hasattr(self, 'overlay_surfaces'):
            try:
                empty_plot_actor = self.plotter.renderer.actors.get('surface_enabled_empty_plot_batch')
                if empty_plot_actor is not None:
                    self.plotter.remove_actor('surface_enabled_empty_plot_batch')
                    if hasattr(self.overlay_surfaces, 'overlay_actor_names'):
                        if 'surface_enabled_empty_plot_batch' in self.overlay_surfaces.overlay_actor_names:
                            self.overlay_surfaces.overlay_actor_names.remove('surface_enabled_empty_plot_batch')
                    if hasattr(self.overlay_surfaces, '_category_actors'):
                        if 'surfaces' in self.overlay_surfaces._category_actors:
                            if 'surface_enabled_empty_plot_batch' in self.overlay_surfaces._category_actors['surfaces']:
                                self.overlay_surfaces._category_actors['surfaces'].remove('surface_enabled_empty_plot_batch')
            except Exception:  # noqa: BLE001
                pass

        # Zeichne alle aktiven Surfaces mit Triangulation
        self._render_surfaces(
            geometry,
            original_plot_values,
            cbar_min,
            cbar_max,
            cmap_object,
            colorization_mode_used,
            cbar_step,
            surface_overrides=surface_overrides,
            phase_mode=phase_mode,
            time_mode=time_mode,
        )
        
        if DEBUG_PLOT3D_TIMING:
            t_after_textures = time.perf_counter()
        
        # 🎯 WICHTIG: Setze Signatur zurück, damit draw_surfaces nach update_overlays aufgerufen wird
        # (um die graue Fläche zu entfernen, wenn SPL-Daten vorhanden sind)
        if hasattr(self, '_last_overlay_signatures') and 'surfaces' in self._last_overlay_signatures:
            # Setze auf None, damit die Signatur als geändert erkannt wird
            self._last_overlay_signatures['surfaces'] = None

        # ------------------------------------------------------------
        # 🎯 SPL-Teppich (Floor) nur anzeigen, wenn KEINE Surfaces aktiv sind
        # ODER wenn Surfaces enabled sind, aber keine Surface-Actors erstellt wurden
        # (z.B. weil keine surface_overrides vorhanden sind)
        # ------------------------------------------------------------
        surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
        has_enabled_surfaces = False
        if isinstance(surface_definitions, dict) and len(surface_definitions) > 0:
            for surface_id, surface_def in surface_definitions.items():
                try:
                    # Kompatibel zu SurfaceDefinition-Objekten und Dicts
                    enabled = bool(getattr(surface_def, "enabled", False)) if hasattr(
                        surface_def, "enabled"
                    ) else bool(surface_def.get("enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False)) if hasattr(
                        surface_def, "hidden"
                    ) else bool(surface_def.get("hidden", False))
                    points = (
                        getattr(surface_def, "points", []) if hasattr(surface_def, "points") else surface_def.get("points", [])
                    ) or []
                    if enabled and not hidden and len(points) >= 3:
                        has_enabled_surfaces = True
                        break
                except Exception:  # noqa: BLE001
                    # Bei inkonsistenten Oberflächen-Konfigurationen Floor lieber anzeigen
                    continue

        # 🎯 WICHTIG: Prüfe ob tatsächlich Surface-Actors erstellt wurden
        # Wenn keine surface_overrides vorhanden sind, wurden keine Surface-Actors erstellt,
        # auch wenn Surfaces enabled sind. In diesem Fall sollte der Floor erstellt werden.
        has_surface_actors = False
        if hasattr(self, '_surface_actors') and isinstance(self._surface_actors, dict) and len(self._surface_actors) > 0:
            has_surface_actors = True
        if hasattr(self, '_group_actors') and isinstance(self._group_actors, dict) and len(self._group_actors) > 0:
            has_surface_actors = True
        if hasattr(self, '_surface_texture_actors') and isinstance(self._surface_texture_actors, dict) and len(self._surface_texture_actors) > 0:
            has_surface_actors = True
        
        # Prüfe auch direkt im Plotter nach Surface-Actors
        if not has_surface_actors and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
            actor_names = list(self.plotter.renderer.actors.keys())
            for name in actor_names:
                if name.startswith('spl_surface_') or name.startswith('spl_surface_group_'):
                    has_surface_actors = True
                    break
        
        # Floor nur erstellen, wenn keine Surface-Actors vorhanden sind
        # (auch wenn Surfaces enabled sind, aber keine surface_overrides vorhanden sind)
        should_create_floor = not has_surface_actors

        if should_create_floor:
            # Erstelle vollständigen Floor-Mesh (ohne Surface-Maskierung oder Clipping)
            floor_mesh = build_full_floor_mesh(
                plot_x,
                plot_y,
                scalars,
                z_coords=z_coords,
                pv_module=pv,
            )
            
            # Plotte (oder update) den Floor nur, wenn keine Surfaces aktiv sind
            floor_actor = self.plotter.renderer.actors.get(self.FLOOR_NAME)
            if floor_actor is None:
                
                self.plotter.add_mesh(
                    floor_mesh,
                    name=self.FLOOR_NAME,
                    scalars="plot_scalars",
                    cmap=cmap_object,
                    clim=(cbar_min, cbar_max),
                    # 🎯 Gradient: smooth rendering, Color step: harte Stufen
                    smooth_shading=not is_step_mode,
                    show_scalar_bar=False,
                    reset_camera=False,
                    interpolate_before_map=not is_step_mode,
                )
                
            else:
                if not hasattr(self, "floor_mesh"):
                    self.floor_mesh = floor_mesh.copy(deep=True)
                else:
                    self.floor_mesh.deep_copy(floor_mesh)
                mapper = floor_actor.mapper
                if mapper is not None:
                    mapper.array_name = "plot_scalars"
                    mapper.scalar_range = (cbar_min, cbar_max)
                    mapper.lookup_table = self.plotter._cmap_to_lut(cmap_object)
                    # 🎯 Gradient: smooth rendering, Color step: harte Stufen
                    mapper.interpolate_before_map = not is_step_mode
        else:
            # Wenn Surface-Actors vorhanden sind, Floor ausblenden (falls vorher gezeichnet)
            floor_actor = self.plotter.renderer.actors.get(self.FLOOR_NAME)
            if floor_actor is not None:
                try:
                    self.plotter.remove_actor(floor_actor)
                except Exception:  # noqa: BLE001
                    pass

        # Aktualisiere Colorbar über ColorbarManager
        self.colorbar_manager.render_colorbar(
            colorization_mode_used,
            force=False,
            tick_step=tick_step,
            phase_mode_active=phase_mode,
            time_mode_active=time_mode,
        )
        self.cbar = self.colorbar_manager.cbar  # Synchronisiere cbar-Referenz

        # ------------------------------------------------------------
        # Vertikale Flächen: werden jetzt identisch wie planare Flächen behandelt
        # (über _render_surfaces mit Triangulation)
        # ------------------------------------------------------------
        # Merke den aktuell verwendeten Colorization-Mode
        self._last_colorization_mode = colorization_mode_used

        # 🎯 VERTIKALE FLÄCHEN: Werden jetzt in _render_surfaces behandelt
        # Entferne alte separate Vertical-Actors (falls vorhanden)
        self._clear_vertical_spl_surfaces()
        
        # ------------------------------------------------------------
        # 🎯 VALIDIERUNG: Prüfe ob alle Surfaces korrekt geplottet wurden
        # ------------------------------------------------------------
        if DEBUG_PLOT3D_TIMING:
            self._validate_surface_plotting()

        if DEBUG_PLOT3D_TIMING:
            t_after_vertical = time.perf_counter()

        self.has_data = True
        if time_mode and self.surface_mesh is not None:
            # Prüfe ob Upscaling aktiv ist - wenn ja, deaktiviere schnellen Update-Pfad
            # da Resampling zu Verzerrungen führen kann
            source_shape = tuple(pressure.shape)
            cache_entry = {
                'source_shape': source_shape,
                'source_x': geometry.source_x.copy(),
                'source_y': geometry.source_y.copy(),
                'target_x': geometry.plot_x.copy(),
                'target_y': geometry.plot_y.copy(),
                'needs_resample': geometry.requires_resample,
                'has_upscaling': geometry.was_upscaled,
                'colorbar_range': (cbar_min, cbar_max),
                'colorization_mode': colorization_mode_used,
                'color_step': cbar_step,
                'grid_shape': geometry.grid_shape,
                'expected_points': self.surface_mesh.n_points,
            }
            self._time_mode_surface_cache = cache_entry
        else:
            self._time_mode_surface_cache = None

        if camera_state is not None:
            self._restore_camera(camera_state)
        # Nur beim ersten Plot mit Daten den Zoom maximieren
        # Bei späteren Updates bleibt der Zoom erhalten (wird durch preserve_camera=True sichergestellt)
        if not self._has_plotted_data:
            self._maximize_camera_view(add_padding=True)
            self._has_plotted_data = True
        
        self.render()
        
        self._save_camera_state()
        self.colorbar_manager.set_override(None)

        if DEBUG_PLOT3D_TIMING:
            t_end = time.perf_counter()
            print(
                "[PlotSPL3D] update_spl_plot timings:\n"
                f"  SPL transform   : {(t_after_spl - t_start_total) * 1000.0:7.2f} ms\n"
                f"  geometry+mask   : {(t_after_geom - t_after_spl) * 1000.0:7.2f} ms\n"
                f"  textures/floor  : {(t_after_textures - t_after_geom) * 1000.0:7.2f} ms\n"
                f"  vertical surfaces: {(t_after_vertical - t_after_textures) * 1000.0:7.2f} ms\n"
                f"  camera/render   : {(t_end - t_after_vertical) * 1000.0:7.2f} ms\n"
                f"  TOTAL           : {(t_end - t_start_total) * 1000.0:7.2f} ms"
            )

    def _get_vertical_color_limits(self) -> tuple[float, float]:
        """
        Liefert die Farbskalen-Grenzen (min, max) für vertikale SPL-Flächen.
        Bevorzugt aktuelle Override-Range aus der Colorbar, sonst Settings-Range.
        """
        params = self.colorbar_manager.get_colorbar_params(self._phase_mode_active)
        try:
            cmin = float(params.get("min", 0.0))
            cmax = float(params.get("max", 0.0))
            return cmin, cmax
        except Exception:
            pass
        try:
            rng = self.settings.colorbar_range
            return float(rng["min"]), float(rng["max"])
        except Exception:
            return 0.0, 120.0

    def _clear_vertical_spl_surfaces(self) -> None:
        """Entfernt alle zusätzlichen SPL-Meshes für senkrechte Flächen."""
        if not hasattr(self, "plotter") or self.plotter is None:
            if hasattr(self, "_vertical_surface_meshes"):
                self._vertical_surface_meshes.clear()
            return
        if not hasattr(self, "_vertical_surface_meshes"):
            return
        for name, actor in list(self._vertical_surface_meshes.items()):
            try:
                # Actor kann entweder direkt der Actor oder nur der Name sein
                if isinstance(name, str):
                    self.plotter.remove_actor(name)
                elif actor is not None:
                    # Falls der Key kein String ist, versuche den Actor direkt zu entfernen
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
            except Exception:
                pass
        self._vertical_surface_meshes.clear()

    def _validate_surface_plotting(self) -> None:
        """
        Validiert, ob alle erkannten Surfaces korrekt geplottet wurden.
        Prüft für jede Surface:
        - Ob sie in surface_grids_data vorhanden ist
        - Ob sie geplottet wurde (Actor vorhanden)
        - Ob die Plot-Positionen korrekt sind
        - Ob die SPL-Werte korrekt sind
        """
        # Debug-Ausgaben entfernt
        return
        
        try:
            calc_spl = getattr(self.container, "calculation_spl", None)
            if not isinstance(calc_spl, dict):
                return
            
            surface_grids_data = calc_spl.get("surface_grids", {})
            surface_results_data = calc_spl.get("surface_results", {})
            surface_definitions = getattr(self.settings, "surface_definitions", {})
            
            if not isinstance(surface_grids_data, dict) or not isinstance(surface_definitions, dict):
                return
            
            # Sammle alle erkannten Surfaces
            all_surface_ids = set(surface_grids_data.keys())
            enabled_surface_ids = set()
            
            for surface_id, surface_def in surface_definitions.items():
                try:
                    if hasattr(surface_def, "to_dict"):
                        surf_data = surface_def.to_dict()
                    elif isinstance(surface_def, dict):
                        surf_data = surface_def
                    else:
                        surf_data = {
                            "enabled": getattr(surface_def, "enabled", False),
                            "hidden": getattr(surface_def, "hidden", False),
                        }
                    
                    enabled = surf_data.get("enabled", False)
                    hidden = surf_data.get("hidden", False)
                    
                    if enabled and not hidden:
                        enabled_surface_ids.add(surface_id)
                except Exception:
                    continue
            
            # Prüfe jede Surface
            plotted_count = 0
            skipped_count = 0
            missing_count = 0
            error_count = 0
            
            for surface_id in sorted(all_surface_ids):
                try:
                    grid_data = surface_grids_data.get(surface_id)
                    if grid_data is None:
                        missing_count += 1
                        continue
                    
                    orientation = grid_data.get('orientation', 'unknown')
                    dominant_axis = grid_data.get('dominant_axis', None)
                    X_grid = np.array(grid_data.get('X_grid', []), dtype=float)
                    Y_grid = np.array(grid_data.get('Y_grid', []), dtype=float)
                    Z_grid = np.array(grid_data.get('Z_grid', []), dtype=float)
                    surface_mask = np.array(grid_data.get('surface_mask', []), dtype=bool)
                    
                    # Prüfe ob Surface enabled ist
                    is_enabled = surface_id in enabled_surface_ids
                    
                    # Prüfe ob Actor vorhanden ist
                    # 🎯 IDENTISCHE BEHANDLUNG: Alle Flächen (planar, sloped, vertical) verwenden jetzt Triangulation
                    # Actor-Namen: "spl_surface_tri_{surface_id}" (Triangulation) oder "spl_surface_tex_{surface_id}" (Texture)
                    # Legacy: "vertical_spl_{surface_id}" (veraltet, wird nicht mehr verwendet)
                    has_actor = False
                    actor_name = None  # 🎯 Initialisiere actor_name, damit es immer definiert ist
                    if hasattr(self, 'plotter') and self.plotter is not None:
                        try:
                            surface_name = getattr(self, 'SURFACE_NAME', 'spl_surface')
                            actor_names = []
                            # 🎯 IDENTISCHE BEHANDLUNG: Alle Orientierungen verwenden die gleichen Actor-Namen
                            actor_names.append(f"{surface_name}_tri_{surface_id}")  # Triangulations-Pfad (Hauptpfad)
                            actor_names.append(f"{surface_name}_tex_{surface_id}")  # Texture-Pfad (Fallback)
                            actor_names.append(f"{surface_name}_gridtri_{surface_id}")  # StructuredGrid-Pfad (Fallback)
                            # Legacy: Vertikale Flächen könnten noch den alten Namen haben (für Rückwärtskompatibilität)
                            if orientation == 'vertical':
                                actor_names.append(f"vertical_spl_{surface_id}")  # Legacy-Name
                            for aname in actor_names:
                                actor = self.plotter.renderer.actors.get(aname)
                                if actor is not None:
                                    has_actor = True
                                    actor_name = aname
                                    break
                        except Exception:
                            pass
                    
                    # Prüfe ob in surface_results_data vorhanden
                    has_results = surface_id in surface_results_data if isinstance(surface_results_data, dict) else False
                    
                    # Prüfe Grid-Daten
                    grid_valid = (
                        X_grid.ndim == 2 and Y_grid.ndim == 2 and Z_grid.ndim == 2 and
                        X_grid.shape == Y_grid.shape == Z_grid.shape and
                        surface_mask.shape == X_grid.shape
                    )
                    
                    active_points = int(np.sum(surface_mask)) if grid_valid else 0
                    
                    # Status bestimmen
                    status = "✅"
                    status_text = "OK"
                    issues = []
                    
                    if not is_enabled:
                        status = "⏭️"
                        status_text = "ÜBERSPRUNGEN (nicht enabled)"
                        skipped_count += 1
                    elif not grid_valid:
                        status = "❌"
                        status_text = "FEHLER (ungültige Grid-Daten)"
                        issues.append(f"Grid-Shapes: X={X_grid.shape}, Y={Y_grid.shape}, Z={Z_grid.shape}")
                        error_count += 1
                    # Vertikale Flächen werden jetzt über Triangulation geplottet → prüfe normal
                    # (keine spezielle Behandlung mehr nötig)
                    elif active_points == 0:
                        status = "⚠️"
                        status_text = "WARNUNG (keine aktiven Punkte)"
                        issues.append(f"Active points: {active_points}")
                        error_count += 1
                    elif not has_results:
                        status = "⚠️"
                        status_text = "WARNUNG (keine Results-Daten)"
                        issues.append("Nicht in surface_results_data")
                        error_count += 1
                    elif not has_actor:
                        status = "⚠️"
                        status_text = "WARNUNG (kein Actor)"
                        if actor_name:
                            issues.append(f"Actor '{actor_name}' nicht gefunden")
                        else:
                            issues.append("Kein Actor gefunden (keine Actor-Namen übereinstimmten)")
                        error_count += 1
                    else:
                        plotted_count += 1
                    
                    # Debug-Ausgabe entfernt
                    
                    if grid_valid:
                        print(f"     └─ X-Range: [{X_grid.min():.2f}, {X_grid.max():.2f}], Span: {X_grid.max()-X_grid.min():.2f} m")
                        print(f"     └─ Y-Range: [{Y_grid.min():.2f}, {Y_grid.max():.2f}], Span: {Y_grid.max()-Y_grid.min():.2f} m")
                        print(f"     └─ Z-Range: [{Z_grid.min():.2f}, {Z_grid.max():.2f}], Span: {Z_grid.max()-Z_grid.min():.2f} m")
                    
                    if has_results:
                        try:
                            result_data = surface_results_data[surface_id]
                            sound_field_p = np.array(result_data.get('sound_field_p', []), dtype=complex)
                            if sound_field_p.size > 0:
                                # Verwende mag2db wie in Plot3DSPL_new.py
                                pressure_magnitude = np.abs(sound_field_p)
                                pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                                if not (hasattr(self, 'functions') and hasattr(self.functions, 'mag2db')):
                                    raise ValueError("functions.mag2db nicht verfügbar")
                                spl_values = self.functions.mag2db(pressure_magnitude)
                                spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                                valid_spl = spl_values[np.isfinite(spl_values) & (spl_values > 0)]
                                if valid_spl.size > 0:
                                    print(f"     └─ SPL-Range: [{np.min(valid_spl):.1f}, {np.max(valid_spl):.1f}] dB, Mean: {np.mean(valid_spl):.1f} dB")
                                else:
                                    issues.append("Keine gültigen SPL-Werte")
                        except Exception as e:
                            issues.append(f"Fehler beim Laden SPL-Werten: {e}")
                    
                    if issues:
                        for issue in issues:
                            print(f"     └─ ⚠️ {issue}")
                    
                except Exception as e:
                    print(f"  ❌ {surface_id}: FEHLER bei Validierung: {e}")
                    error_count += 1
            
            # Zusammenfassung
            print("\n" + "-"*80)
            print(f"[VALIDIERUNG] Zusammenfassung:")
            print(f"  ✅ Korrekt geplottet: {plotted_count}")
            print(f"  ⏭️ Übersprungen (nicht enabled): {skipped_count}")
            print(f"  ⚠️ Warnings/Fehler: {error_count}")
            print(f"  ❌ Fehlend: {missing_count}")
            print(f"  📊 Gesamt erkannt: {len(all_surface_ids)}")
            print("="*80 + "\n")
            
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[VALIDIERUNG] Fehler bei Validierung: {e}")


class SPLTimeControlBar(QtWidgets.QFrame):
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("spl_time_control")
        self.setStyleSheet(
            "QFrame#spl_time_control {"
            "  background-color: rgba(240, 240, 240, 200);"
            "  border: 1px solid rgba(0, 0, 0, 150);"
            "  border-radius: 4px;"
            "}"
        )
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self._frames = 1
        self._simulation_time = 0.2  # 200ms (Standard)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)

        # Container für Slider und schwarze Linie
        slider_container = QtWidgets.QWidget(self)
        slider_layout = QtWidgets.QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(0)

        # Schwarze Linie links vom Slider
        self.track_line = QtWidgets.QFrame(slider_container)
        self.track_line.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
        self.track_line.setFixedWidth(2)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical, slider_container)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)
        self.slider.setInvertedAppearance(True)  # min (t=0) unten
        self.slider.setStyleSheet(
            "QSlider::groove:vertical {"
            "  border: none;"
            "  background: rgba(200, 200, 200, 150);"
            "  width: 4px;"
            "  border-radius: 2px;"
            "}"
            "QSlider::handle:vertical {"
            "  background: rgba(50, 50, 50, 220);"
            "  border: 1px solid rgba(0, 0, 0, 200);"
            "  width: 12px;"
            "  height: 12px;"
            "  border-radius: 6px;"
            "  margin: 0px -4px;"
            "}"
            "QSlider::handle:vertical:hover {"
            "  background: rgba(30, 30, 30, 240);"
            "}"
        )

        slider_layout.addWidget(self.track_line)
        slider_layout.addWidget(self.slider, 1)

        layout.addWidget(slider_container, 1)

        # Zeit-Label als separates Widget außerhalb des Fader-Containers (wird in _reposition platziert)
        self.label = QtWidgets.QLabel("t=0", parent)
        self.label.setAlignment(QtCore.Qt.AlignHCenter)
        self.label.setStyleSheet("font-size: 9px; font-weight: bold; color: #333; background-color: rgba(240, 240, 240, 200); padding: 2px; border-radius: 2px;")
        self.label.hide()

        self.slider.valueChanged.connect(self._on_value_changed)
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event):  # noqa: D401
        if obj is self.parent() and event.type() == QtCore.QEvent.Resize:
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self):
        parent = self.parent()
        if parent is None:
            return
        margin = 12
        width = 32  # Schmaler: 32px statt 60px
        height = max(150, parent.height() - 2 * margin - 20)  # Platz für Label unterhalb
        x = parent.width() - width - margin
        y = margin
        self.setGeometry(x, y, width, height)

        # Positioniere Label unterhalb des Fader-Objekts (breiter als Fader für vollständigen Text)
        label_height = 16
        label_width = 60  # Breiter als Fader, damit Text nicht abgeschnitten wird
        label_x = x - (label_width - width) // 2  # Zentriert unter dem Fader
        label_y = y + height + 2
        self.label.setGeometry(label_x, label_y, label_width, label_height)

    def configure(self, frames: int, value: int, simulation_time: float = 0.2):
        frames = max(1, int(frames))
        self._frames = frames
        self._simulation_time = float(simulation_time)
        self.slider.setMaximum(frames - 1)
        clamped = max(0, min(value, frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(clamped)
        self.slider.blockSignals(False)
        self._update_label(clamped)
        self._reposition()
        # Zeige Label an, wenn Fader sichtbar ist
        if self.isVisible() and self.label:
            self.label.show()

    def _on_value_changed(self, value: int):
        self._update_label(value)
        self.valueChanged.emit(int(value))

    def _update_label(self, value: int):
        # Berechne Zeit in ms: Frames werden mit np.linspace(0.0, simulation_time, frames_per_period + 1) erstellt
        # Frame 0 = t=0, Frame 1..N = gleichmäßig über simulation_time verteilt
        if value == 0:
            time_ms = 0.0
        else:
            # self._frames = frames_per_period + 1 (inkl. Frame 0)
            # Frames 1..N sind gleichmäßig über simulation_time verteilt
            frames_per_period = self._frames - 1  # Ohne Frame 0
            if frames_per_period > 0:
                # Zeit für Frame value: value * (simulation_time / frames_per_period)
                # Das entspricht np.linspace(0.0, simulation_time, frames_per_period + 1)[value]
                time_ms = (value * self._simulation_time / frames_per_period) * 1000.0
            else:
                time_ms = 0.0

        # Zeige Zeit in ms (mit 1 Dezimalstelle)
        self.label.setText(f"t={time_ms:.1f}ms")

    def show(self):
        """Zeige Fader und Label."""
        super().show()
        if self.label:
            self.label.show()

    def hide(self):
        """Verstecke Fader und Label."""
        super().hide()
        if self.label:
            self.label.hide()









__all__ = ['SPL3DPlotRenderer', 'SPLTimeControlBar']