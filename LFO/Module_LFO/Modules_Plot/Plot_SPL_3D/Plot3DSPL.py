from __future__ import annotations

import hashlib
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
PLOT_SUBDIVISION_LEVEL = 2


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
    @staticmethod
    def _subdivide_triangles(
        vertices: np.ndarray,
        faces: np.ndarray,
        scalars: np.ndarray
    ) -> tuple:
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
    def _interpolate_grid_3d(
        source_x: np.ndarray,
        source_y: np.ndarray,
        source_z: np.ndarray,
        values: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
        zq: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """
        3D-Interpolation auf einem regulären Grid unter Verwendung von scipy.interpolate.griddata.
        
        Args:
            source_x: 1D-Array der X-Koordinaten des Source-Grids
            source_y: 1D-Array der Y-Koordinaten des Source-Grids
            source_z: 2D-Array der Z-Koordinaten des Source-Grids (Shape: len(source_y) x len(source_x))
            values: 2D-Array der SPL-Werte (Shape: len(source_y) x len(source_x))
            xq: 1D-Array der X-Abfragepunkte
            yq: 1D-Array der Y-Abfragepunkte
            zq: 1D-Array der Z-Abfragepunkte
            method: Interpolationsmethode ('linear' oder 'nearest')
            
        Returns:
            1D-Array der interpolierten Werte
        """
        try:
            from scipy.interpolate import griddata
        except ImportError:
            # Fallback: 2D-Interpolation wenn scipy nicht verfügbar
            if method == "nearest":
                return SPL3DPlotRenderer._nearest_interpolate_grid(source_x, source_y, values, xq, yq)
            else:
                return SPL3DPlotRenderer._bilinear_interpolate_grid(source_x, source_y, values, xq, yq)
        
        source_x = np.asarray(source_x, dtype=float).reshape(-1)
        source_y = np.asarray(source_y, dtype=float).reshape(-1)
        source_z = np.asarray(source_z, dtype=float)
        vals = np.asarray(values, dtype=float)
        xq = np.asarray(xq, dtype=float).reshape(-1)
        yq = np.asarray(yq, dtype=float).reshape(-1)
        zq = np.asarray(zq, dtype=float).reshape(-1)
        
        ny, nx = vals.shape
        if ny != len(source_y) or nx != len(source_x):
            raise ValueError(
                f"_interpolate_grid_3d: Shape mismatch values={vals.shape}, "
                f"expected=({len(source_y)}, {len(source_x)})"
            )
        if source_z.shape != (ny, nx):
            raise ValueError(
                f"_interpolate_grid_3d: Shape mismatch source_z={source_z.shape}, "
                f"expected=({ny}, {nx})"
            )
        
        # Erstelle 3D-Punkte für Source-Grid
        X_source, Y_source = np.meshgrid(source_x, source_y, indexing="xy")
        points_3d = np.column_stack((
            X_source.ravel(),
            Y_source.ravel(),
            source_z.ravel()
        ))
        values_flat = vals.ravel()
        
        # Entferne NaN-Werte
        valid_mask = np.isfinite(values_flat) & np.isfinite(points_3d).all(axis=1)
        if not np.any(valid_mask):
            raise ValueError("Keine gültigen Werte für 3D-Interpolation verfügbar")
        
        points_3d_clean = points_3d[valid_mask]
        values_clean = values_flat[valid_mask]
        
        # Erstelle Abfragepunkte
        query_points = np.column_stack((xq, yq, zq))
        
        # 3D-Interpolation
        result = griddata(
            points_3d_clean,
            values_clean,
            query_points,
            method=method,
            fill_value=np.nan
        )
        
        # Prüfe auf NaN-Werte
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            raise ValueError(f"3D-Interpolation liefert {np.sum(nan_mask)} NaN-Werte")
        
        return result

    def _calculate_texture_signature(
        self,
        surface_id: str,
        points: list[dict[str, float]],
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        cbar_min: float,
        cbar_max: float,
        cmap_object: str | Any,
        colorization_mode: str,
        cbar_step: float,
        tex_res_surface: float,
        plane_model: dict | None,
    ) -> str:
        """
        Berechnet eine Hash-Signatur für eine Textur, die sich ändert, wenn sich
        Surface-Geometrie, SPL-Werte oder Colormap-Parameter ändern.
        
        Returns:
            str: Hex-String der Hash-Signatur
        """
        # Erstelle Hash-Objekt
        h = hashlib.sha256()
        
        # 1. Surface-Geometrie (Punkte)
        points_tuple = tuple(
            (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
            for p in points
        )
        h.update(str(points_tuple).encode('utf-8'))
        
        # 2. Source-Grid (für Interpolation)
        h.update(source_x.tobytes())
        h.update(source_y.tobytes())
        
        # 3. SPL-Werte (Hash des Arrays - nur wenn sich Werte ändern)
        # Verwende einen repräsentativen Hash: Min, Max, Mean, Std
        if values.size > 0:
            val_min = float(np.nanmin(values))
            val_max = float(np.nanmax(values))
            val_mean = float(np.nanmean(values))
            val_std = float(np.nanstd(values))
            # Zusätzlich: Hash eines Sample-Grids (jeder 10. Wert) für bessere Erkennung
            sample_indices = np.unravel_index(
                np.arange(0, values.size, max(1, values.size // 100)),
                values.shape
            )
            sample_values = values[sample_indices]
            h.update(f"{val_min:.6f}_{val_max:.6f}_{val_mean:.6f}_{val_std:.6f}".encode('utf-8'))
            h.update(sample_values.tobytes())
        
        # 4. Colormap-Parameter
        cmap_name = cmap_object if isinstance(cmap_object, str) else str(type(cmap_object))
        h.update(f"{cmap_name}_{cbar_min:.6f}_{cbar_max:.6f}_{colorization_mode}_{cbar_step:.6f}".encode('utf-8'))
        
        # 5. Textur-Auflösung
        h.update(f"{tex_res_surface:.8f}".encode('utf-8'))
        
        # 6. Planmodell (für geneigte Flächen)
        if plane_model:
            plane_str = str(sorted(plane_model.items()))
            h.update(plane_str.encode('utf-8'))
        
        return h.hexdigest()

    def _update_surface_scalars(self, flat_scalars: np.ndarray) -> bool:
        """Aktualisiert die Skalare des Surface-Meshes."""
        if self.surface_mesh is None:
            return False

        if flat_scalars.size == self.surface_mesh.n_points:
            self.surface_mesh.point_data['plot_scalars'] = flat_scalars
            if hasattr(self.surface_mesh, "modified"):
                self.surface_mesh.modified()
            return True

        if flat_scalars.size == self.surface_mesh.n_cells:
            self.surface_mesh.cell_data['plot_scalars'] = flat_scalars
            if hasattr(self.surface_mesh, "modified"):
                self.surface_mesh.modified()
            return True

        return False

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

    def get_texture_metadata(self, surface_id: str) -> Optional[dict[str, Any]]:
        """
        Gibt die Metadaten einer Texture-Surface zurück.
        
        Args:
            surface_id: ID der Surface
            
        Returns:
            Dict mit Metadaten oder None, falls Surface nicht gefunden wurde.
            Metadaten enthalten:
            - 'actor': PyVista Actor
            - 'grid': StructuredGrid mit Weltkoordinaten
            - 'texture': PyVista Texture-Objekt
            - 'grid_bounds': (xmin, xmax, ymin, ymax, zmin, zmax)
            - 'world_coords_x': 1D-Array der X-Koordinaten in Metern
            - 'world_coords_y': 1D-Array der Y-Koordinaten in Metern
            - 'world_coords_grid_x': 2D-Meshgrid der X-Koordinaten
            - 'world_coords_grid_y': 2D-Meshgrid der Y-Koordinaten
            - 'texture_resolution': Auflösung in Metern
            - 'texture_size': (H, W) in Pixeln
            - 'image_shape': (H, W, 4) Shape des Bildes
            - 'polygon_bounds': Dict mit xmin, xmax, ymin, ymax
            - 'polygon_points': Original Polygon-Punkte
            - 't_coords': Textur-Koordinaten (n_points, 2)
            - 'surface_id': Surface-ID
        """
        texture_data = self._surface_texture_actors.get(surface_id)
        if texture_data is None:
            return None
        if isinstance(texture_data, dict) and 'metadata' in texture_data:
            return texture_data['metadata']
        if isinstance(texture_data, dict) and 'actor' in texture_data:
            # Neue Struktur: Dict mit Metadaten
            return texture_data
        # Alte Struktur: Nur Actor (für Rückwärtskompatibilität)
        return None

    def get_texture_world_coords(self, surface_id: str, texture_u: float, texture_v: float) -> Optional[tuple[float, float, float]]:
        """
        Konvertiert Textur-Koordinaten (u, v) zu Weltkoordinaten (x, y, z).
        
        Args:
            surface_id: ID der Surface
            texture_u: U-Koordinate [0, 1] (horizontal)
            texture_v: V-Koordinate [0, 1] (vertikal)
            
        Returns:
            (x, y, z) Weltkoordinaten in Metern oder None bei Fehler
        """
        metadata = self.get_texture_metadata(surface_id)
        if metadata is None:
            return None
        
        try:
            bounds = metadata.get('grid_bounds')
            if bounds is None or len(bounds) < 6:
                return None
            
            xmin, xmax = bounds[0], bounds[1]
            ymin, ymax = bounds[2], bounds[3]
            zmin, zmax = bounds[4], bounds[5]
            
            # Konvertiere normalisierte Textur-Koordinaten zu Weltkoordinaten
            x = xmin + texture_u * (xmax - xmin)
            y = ymin + texture_v * (ymax - ymin)
            z = zmin  # Für horizontale Surfaces ist Z konstant
            
            return (x, y, z)
        except Exception:
            return None

    def get_world_coords_to_texture_coords(self, surface_id: str, world_x: float, world_y: float) -> Optional[tuple[float, float]]:
        """
        Konvertiert Weltkoordinaten (x, y) zu Textur-Koordinaten (u, v).
        
        Args:
            surface_id: ID der Surface
            world_x: X-Koordinate in Metern
            world_y: Y-Koordinate in Metern
            
        Returns:
            (u, v) Textur-Koordinaten [0, 1] oder None bei Fehler
        """
        metadata = self.get_texture_metadata(surface_id)
        if metadata is None:
            return None
        
        try:
            bounds = metadata.get('grid_bounds')
            if bounds is None or len(bounds) < 6:
                return None
            
            xmin, xmax = bounds[0], bounds[1]
            ymin, ymax = bounds[2], bounds[3]
            
            # Konvertiere Weltkoordinaten zu normalisierten Textur-Koordinaten
            if xmax > xmin:
                u = (world_x - xmin) / (xmax - xmin)
            else:
                u = 0.0
            
            if ymax > ymin:
                v = (world_y - ymin) / (ymax - ymin)
            else:
                v = 0.0
            
            # Clippe auf [0, 1]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            
            return (u, v)
        except Exception:
            return None

    def _render_surfaces_textured(
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
        Renderpfad für horizontale Surfaces als 2D-Texturen.

        - Pro Surface wird ein lokales 2D-Bild in Welt-X/Y berechnet.
        - SPL-Werte stammen aus original_plot_values (coarse Grid) via bilinearer Interpolation.
        - Die Textur wird auf ein flaches StructuredGrid in derselben XY-Position gelegt.
        """
        if not hasattr(self, "plotter") or self.plotter is None:
            return

        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return

        # Optionales Feintiming für diesen Pfad
        t_textures_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0

        # 🎯 Nur mit per-Surface-Overrides arbeiten (keine globalen Positionen mehr)
        if not surface_overrides:
            return

        # Bestimme ob Step-Modus aktiv ist (vor der Verwendung definieren)
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0

        # Auflösung der Textur in Metern (XY)
        # Standard-Basiswert aus den Settings (z.B. 0.03m)
        # 🎯 Gradient-Modus: etwas feinere Auflösung als im Color-Step-Modus
        base_tex_res = float(
            getattr(self.settings, "spl_surface_texture_resolution", 0.03) or 0.03
        )
        if is_step_mode:
            # Color step: Normale Auflösung (harte Stufen benötigen keine hohe Auflösung)
            tex_res_global = base_tex_res
        else:
            # Gradient: 2x feinere Auflösung (Performance-Kompromiss)
            # Bilineare Interpolation sorgt bereits für glatte Übergänge
            # (0.01m = 1cm pro Pixel statt 20mm = 2x mehr Pixel)
            tex_res_global = base_tex_res * 0.5

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

        # --------------------------------------------------------------
        # Texturen global deaktivieren: nur Triangulation rendern
        # --------------------------------------------------------------
        disable_textures = True
        if disable_textures:
            texture_actors = getattr(self, "_surface_texture_actors", {}) or {}
            for sid, tex_data in list(texture_actors.items()):
                actor = tex_data.get("actor") if isinstance(tex_data, dict) else tex_data
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
            self._surface_texture_actors = {}
            self._surface_texture_cache = {}

        # Aktive Surfaces ermitteln
        surface_definitions = getattr(self.settings, "surface_definitions", {}) or {}
        enabled_surfaces: list[tuple[str, list[dict[str, float]], Any]] = []
        if isinstance(surface_definitions, dict):
            for surface_id, surface_def in surface_definitions.items():
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, "enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False))
                    points = getattr(surface_def, "points", []) or []
                    surface_obj = surface_def
                else:
                    enabled = bool(surface_def.get("enabled", False))
                    hidden = bool(surface_def.get("hidden", False))
                    points = surface_def.get("points", []) or []
                    surface_obj = surface_def
                if enabled and not hidden and len(points) >= 3:
                    enabled_surfaces.append((str(surface_id), points, surface_obj))
        

        # Nicht mehr benötigte Textur-Actors entfernen
        active_ids = {sid for sid, _, _ in enabled_surfaces}
        for sid, texture_data in list(self._surface_texture_actors.items()):
            if sid not in active_ids:
                try:
                    actor = texture_data.get('actor') if isinstance(texture_data, dict) else texture_data
                    if actor is not None:
                        self.plotter.remove_actor(actor)
                except Exception:
                    pass
                self._surface_texture_actors.pop(sid, None)
                # 🚀 TEXTUR-CACHE: Auch Cache-Eintrag entfernen
                self._surface_texture_cache.pop(sid, None)

        # 🎯 WICHTIG: Wenn keine enabled_surfaces vorhanden sind, aber surface_overrides existieren,
        # sollten wir trotzdem fortfahren, um vertikale Flächen zu plotten
        if not enabled_surfaces and not surface_overrides:
            return

        # Effektiver Upscaling-Faktor (Standardwert)
        effective_upscale_factor: int = 4


        # 🚀 PARALLELISIERUNG: Verarbeite Surfaces parallel
        # Cache-Prüfung erfolgt im Hauptthread, Berechnungen parallel
        surfaces_to_process = []
        cached_surfaces = []
        
        # SCHRITT 1: Cache-Prüfung im Hauptthread (thread-safe)
        for surface_id, points, surface_obj in enabled_surfaces:
            # Berechne Signatur für Cache-Prüfung (schnell, kann im Hauptthread bleiben)
            poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
            poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
            if poly_x.size == 0 or poly_y.size == 0:
                continue
            
            dict_points = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                for p in points
            ]
            plane_model, _ = derive_surface_plane(dict_points)
            
            # 🎯 WICHTIG: Berechne tex_res_surface mit der GLEICHEN Logik wie in _process_planar_surface_texture
            # Dies ist kritisch für korrekte Cache-Prüfung, da die Signatur sonst nicht übereinstimmt
            tex_res_surface = tex_res_global
            
            # Heuristik: Erkenne axis-aligned Rechtecke (gleiche Logik wie in _process_planar_surface_texture)
            is_axis_aligned_rectangle = False
            try:
                if poly_x.size >= 4 and poly_y.size >= 4:
                    px = poly_x
                    py = poly_y
                    if (
                        poly_x.size >= 2
                        and abs(poly_x[0] - poly_x[-1]) < 1e-6
                        and abs(poly_y[0] - poly_y[-1]) < 1e-6
                    ):
                        px = poly_x[:-1]
                        py = poly_y[:-1]
                    if px.size >= 4:
                        xmin_rect = float(px.min())
                        xmax_rect = float(px.max())
                        ymin_rect = float(py.min())
                        ymax_rect = float(py.max())
                        span_x = xmax_rect - xmin_rect
                        span_y = ymax_rect - ymin_rect
                        tol = 1e-3
                        
                        if span_x > tol and span_y > tol:
                            on_left = np.isclose(px, xmin_rect, atol=tol)
                            on_right = np.isclose(px, xmax_rect, atol=tol)
                            on_bottom = np.isclose(py, ymin_rect, atol=tol)
                            on_top = np.isclose(py, ymax_rect, atol=tol)
                            on_edge = on_left | on_right | on_bottom | on_top
                            if np.all(on_edge):
                                has_left = bool(np.any(on_left))
                                has_right = bool(np.any(on_right))
                                has_bottom = bool(np.any(on_bottom))
                                has_top = bool(np.any(on_top))
                                has_tl = bool(np.any(on_left & on_top))
                                has_tr = bool(np.any(on_right & on_top))
                                has_bl = bool(np.any(on_left & on_bottom))
                                has_br = bool(np.any(on_right & on_bottom))
                                
                                if has_left and has_right and has_bottom and has_top and has_tl and has_tr and has_bl and has_br:
                                    is_axis_aligned_rectangle = True
            except Exception:
                is_axis_aligned_rectangle = False
            
            # Wenn axis-aligned rectangle und upscale_factor > 1, verwende größere tex_res_surface
            if is_axis_aligned_rectangle and effective_upscale_factor > 1:
                tex_res_surface = tex_res_global * float(effective_upscale_factor)
            
            # Per-Surface Overrides aus neuem Grid nutzen (nur Positionen; keine Fallbacks)
            override = surface_overrides.get(surface_id)
            if not override:
                continue
            sx = override.get("source_x", np.array([]))
            sy = override.get("source_y", np.array([]))
            vals = override.get("values", np.array([]))
            override_used = True

            # Berechne Signatur mit korrekter tex_res_surface
            texture_signature = self._calculate_texture_signature(
                surface_id=surface_id,
                points=points,
                source_x=sx,
                source_y=sy,
                values=vals,
                cbar_min=cbar_min,
                cbar_max=cbar_max,
                cmap_object=cmap_object,
                colorization_mode=colorization_mode,
                cbar_step=cbar_step,
                tex_res_surface=tex_res_surface,
                plane_model=plane_model,
            )
            
            # Prüfe Cache
            cached_texture_data = self._surface_texture_actors.get(surface_id)
            cached_signature = self._surface_texture_cache.get(surface_id)

            if disable_textures:
                cached_texture_data = None
                cached_signature = None
            
            if cached_texture_data is not None and cached_signature == texture_signature:
                cached_actor = cached_texture_data.get('actor') if isinstance(cached_texture_data, dict) else None
                if cached_actor is not None:
                    # Cache-HIT: Wiederverwenden
                    cached_surfaces.append((surface_id, cached_texture_data))
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[PlotSPL3D] surface {surface_id}: CACHE-HIT - Textur wiederverwendet")
                    continue
            
            # Cache-MISS: Muss verarbeitet werden
            surfaces_to_process.append((surface_id, points, surface_obj))
        
        # SCHRITT 2: Sequenzielle Verarbeitung der Surfaces ohne Cache (Parallelisierung entfernt)
        axis_aligned_count = 0
        surfaces_processed = 0
        
        # 🎯 PRIORITÄT 1: Versuche Triangulation (wenn verfügbar)
        # Hole triangulierte Daten aus surface_grids_data
        calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
        surface_grids_data = calc_spl.get("surface_grids", {}) or {} if isinstance(calc_spl, dict) else {}
        surface_results_data = calc_spl.get("surface_results", {}) or {} if isinstance(calc_spl, dict) else {}
        
        # 🎯 ERWEITERE surfaces_to_process: Füge Surfaces hinzu, die Overrides haben, aber nicht in enabled_surfaces sind
        # Dies ermöglicht das Plotten von vertikalen Flächen, die zwar Grids/Results haben, aber nicht aktiviert sind
        # WICHTIG: Dies funktioniert auch, wenn enabled_surfaces leer ist (keine Surfaces aktiviert)
        override_surface_ids = set(surface_overrides.keys()) if surface_overrides else set()
        enabled_surface_ids = {sid for sid, _, _ in enabled_surfaces} if enabled_surfaces else set()
        additional_surface_ids = override_surface_ids - enabled_surface_ids
        
        if additional_surface_ids and isinstance(surface_grids_data, dict):
            surface_definitions = getattr(self.settings, "surface_definitions", {}) or {}
            for sid in additional_surface_ids:
                if sid in surface_grids_data:
                    # Hole Surface-Definition für zusätzliche Surfaces
                    surface_def = surface_definitions.get(sid)
                    if surface_def:
                        if isinstance(surface_def, SurfaceDefinition):
                            points = getattr(surface_def, "points", []) or []
                        else:
                            points = surface_def.get("points", []) or []
                        if len(points) >= 3:
                            surfaces_to_process.append((sid, points, surface_def))
        
        # Sequenzielle Verarbeitung
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

                # 🎯 PRIORITÄT 1: Prüfe ob triangulierte Daten verfügbar sind
                use_triangulation = False
                triangulated_vertices = None
                triangulated_faces = None
                grid_data = None
                
                if isinstance(surface_grids_data, dict) and surface_id in surface_grids_data:
                    grid_data = surface_grids_data[surface_id]
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
                            
                            # Konvertiere zu SPL in dB
                            if time_mode:
                                spl_values_2d = np.real(sound_field_p_complex)
                                spl_values_2d = np.nan_to_num(spl_values_2d, nan=0.0, posinf=0.0, neginf=0.0)
                            elif phase_mode:
                                spl_values_2d = np.angle(sound_field_p_complex)
                                spl_values_2d = np.nan_to_num(spl_values_2d, nan=0.0, posinf=0.0, neginf=0.0)
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
                                        print(f"  └─ ✅ STRUCTUREDGRID-PFAD: Direkter Plot aus Grid ({Xg.size} Punkte), Range=[{clim_min:.1f}, {clim_max:.1f}] dB")
                                        
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
                            
                            # 🚀 OPTIMIERUNG: Vertices entsprechen jetzt Grid-Punkten direkt
                            # Keine Interpolation nötig - direkte Zuordnung möglich!
                            # (Xg, Yg bereits oben geladen)
                            surface_mask = np.asarray(grid_data.get("surface_mask", []))
                            
                            # Prüfe, ob Vertices direkt Grid-Punkten entsprechen (neue Triangulation)
                            # Vertices sollten die gleiche Anzahl und Reihenfolge wie X_grid.ravel() haben
                            n_vertices = len(triangulated_vertices)
                            n_grid_points = Xg.size
                            
                            if n_vertices == n_grid_points:
                                # 🎯 Direkte Zuordnung: Vertices = Grid-Punkte in derselben Reihenfolge
                                # Direkte Zuordnung: spl_at_verts = spl_values_2d.ravel()
                                spl_at_verts = spl_values_2d.ravel().copy()
                                
                                # Setze NaN für inaktive Grid-Punkte
                                if surface_mask.size == n_grid_points and surface_mask.shape == Xg.shape:
                                    mask_flat = surface_mask.ravel().astype(bool)
                                    spl_at_verts[~mask_flat] = np.nan
                            else:
                                # Vertices ≠ Grid-Punkte - verwende Nearest-Map für zusätzliche Vertices
                                try:
                                    from scipy.spatial import cKDTree
                                    # Erstelle Grid-Punkte für Nearest-Map
                                    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg_check.ravel() if 'Zg_check' in locals() else np.zeros(Xg.size)])
                                    grid_vals = spl_values_2d.ravel()
                                    
                                    # Filtere nach surface_mask, falls vorhanden
                                    if surface_mask.size == Xg.size and surface_mask.shape == Xg.shape:
                                        mask_flat = surface_mask.ravel().astype(bool)
                                        grid_pts = grid_pts[mask_flat]
                                        grid_vals = grid_vals[mask_flat]
                                    
                                    # Erstelle KDTree für Nearest-Neighbor-Suche
                                    tree = cKDTree(grid_pts)
                                    dists, nn_idx = tree.query(triangulated_vertices, k=1)
                                    spl_at_verts = grid_vals[nn_idx]
                                    
                                    valid_mask = np.isfinite(spl_at_verts)
                                    n_valid = int(np.sum(valid_mask))
                                    if n_valid == 0:
                                        raise ValueError(f"Surface '{surface_id}': Nearest-Map liefert keine gültigen Werte")
                                except Exception as e_nn:
                                    from scipy.interpolate import griddata
                                    points_new = triangulated_vertices
                                    points_orig = np.column_stack([Xg.ravel(), Yg.ravel(), Zg_check.ravel() if 'Zg_check' in locals() else np.zeros(Xg.size)])
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
                                        fill_value=np.nan
                                    )
                                    valid_mask = np.isfinite(spl_at_verts)
                                    n_valid = np.sum(valid_mask)
                                    if n_valid == 0:
                                        raise ValueError(f"Surface '{surface_id}': Griddata(nearest) liefert keine gültigen Werte")
                            
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
                            
                            # 🎯 SUBDIVISION: Unterteile jedes Dreieck in 4 kleinere für schärfere Plots
                            # Dies erhöht die visuelle Auflösung ohne die Berechnungszeit zu erhöhen
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
                            
                            # Entferne alten Actor
                            old_texture_data = self._surface_texture_actors.get(surface_id)
                            if old_texture_data is not None:
                                old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                                if old_actor is not None:
                                    try:
                                        self.plotter.remove_actor(old_actor)
                                    except Exception:
                                        pass
                            
                            # Füge trianguliertes Mesh hinzu
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
                            
                            # Speichere in _surface_actors (nicht _surface_texture_actors)
                            if not hasattr(self, '_surface_actors'):
                                self._surface_actors = {}
                            self._surface_actors[surface_id] = actor
                            
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

        camera_state = self._camera_state or self._capture_camera()

        # 🎯 WICHTIG: Erstelle surface_overrides VOR den Validierungsprüfungen,
        # damit vertikale Flächen auch ohne gültige globale SPL-Daten geplottet werden können
        surface_overrides: dict[str, dict[str, np.ndarray]] = {}
        calc_spl = getattr(self.container, "calculation_spl", {}) if hasattr(self, "container") else {}
        surface_grids_data = {}
        surface_results_data = {}
        if isinstance(calc_spl, dict):
            surface_grids_data = calc_spl.get("surface_grids", {}) or {}
            surface_results_data = calc_spl.get("surface_results", {}) or {}
        
        if isinstance(surface_grids_data, dict) and surface_grids_data and isinstance(surface_results_data, dict) and surface_results_data:
            for sid, grid_data in surface_grids_data.items():
                orientation = grid_data.get("orientation", "").lower() if isinstance(grid_data, dict) else ""
                if sid not in surface_results_data:
                    continue
                try:
                    Xg = np.asarray(grid_data.get("X_grid", []))
                    Yg = np.asarray(grid_data.get("Y_grid", []))
                    if Xg.size == 0 or Yg.size == 0:
                        continue
                    if Xg.ndim == 2 and Yg.ndim == 2:
                        # Achsen aus dem strukturierten Grid ableiten
                        gx = Xg[0, :] if Xg.shape[1] > 0 else Xg.ravel()
                        gy = Yg[:, 0] if Yg.shape[0] > 0 else Yg.ravel()
                    else:
                        continue
                    
                    # 🎯 NEU: Verwende direkt die berechneten SPL-Werte aus surface_results
                    #         Keine Interpolation mehr!
                    result_data = surface_results_data[sid]
                    sound_field_p_complex = np.array(result_data.get('sound_field_p', []), dtype=complex)
                    
                    if sound_field_p_complex.size == 0 or sound_field_p_complex.shape != Xg.shape:
                        continue
                    
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
        except Exception:  # noqa: BLE001
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
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                if not has_surface_overrides:
                    return
            phase_values = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            plot_values = phase_values
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
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
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
        
        # 🎯 surface_overrides wurden bereits am Anfang der Funktion erstellt
        # (vor den Validierungsprüfungen, damit vertikale Flächen auch ohne gültige globale Daten geplottet werden können)
        
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

        # Entferne existierende Mesh-Actors für horizontale Surfaces
        base_actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
        if base_actor is not None:
            try:
                self.plotter.remove_actor(base_actor)
            except Exception:
                pass
        for sid, actor in list(self._surface_actors.items()):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
            self._surface_actors.pop(sid, None)
        
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

        # Zeichne alle aktiven Surfaces als Texturflächen
        self._render_surfaces_textured(
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

        if not has_enabled_surfaces:
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
            # Wenn Surfaces aktiv sind, Floor ausblenden (falls vorher gezeichnet)
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
        # (über _render_surfaces_textured mit Triangulation)
        # ------------------------------------------------------------
        # Merke den aktuell verwendeten Colorization-Mode
        self._last_colorization_mode = colorization_mode_used

        # 🎯 VERTIKALE FLÄCHEN: Werden jetzt in _render_surfaces_textured behandelt
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

    def _update_vertical_spl_surfaces(self) -> None:
        """
        🎯 DEPRECATED: Vertikale Surfaces werden jetzt identisch wie planare Surfaces behandelt.
        
        Diese Funktion ist nicht mehr nötig, da vertikale Flächen jetzt in _render_surfaces_textured()
        mit der gleichen Triangulation-Logik wie planare Flächen geplottet werden.
        """
        """
        Zeichnet / aktualisiert SPL-Flächen für senkrechte Surfaces auf Basis von
        calculation_spl['surface_samples'] und calculation_spl['surface_fields'].
        
        Hinweis:
        - Für horizontale Flächen verwenden wir ausschließlich den Texture-Pfad.
        - Für senkrechte / stark geneigte Flächen rendern wir hier explizite Meshes
          (vertical_spl_<surface_id>), damit sie im 3D-Plot separat anwählbar sind.
        """
        container = getattr(self, "container", None)
        if container is None or not hasattr(container, "calculation_spl"):
            self._clear_vertical_spl_surfaces()
            return

        calc_spl = getattr(container, "calculation_spl", {}) or {}
        sample_payloads = calc_spl.get("surface_samples")
        if not isinstance(sample_payloads, list):
            self._clear_vertical_spl_surfaces()
            return

        # Aktueller Surface-Status (enabled/hidden) aus den Settings
        surface_definitions = getattr(self.settings, "surface_definitions", {})
        if not isinstance(surface_definitions, dict):
            surface_definitions = {}

        # Vertikale Surfaces analog zu prepare_plot_geometry behandeln:
        # lokales (u,v)-Raster + strukturiertes Mesh über build_vertical_surface_mesh.
        new_vertical_meshes: dict[str, Any] = {}

        # Aktuellen Colorization-Mode verwenden (wie für die Hauptfläche).
        colorization_mode = getattr(self, "_last_colorization_mode", None)
        if colorization_mode not in {"Color step", "Gradient"}:
            colorization_mode = getattr(self.settings, "colorization_mode", "Gradient")
        try:
            cbar_range = getattr(self.settings, "colorbar_range", {})
            cbar_step = float(cbar_range.get("step", 0.0))
        except Exception:
            cbar_step = 0.0
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0
        # 🐛 DEBUG: Prüfe alle Payloads
        if DEBUG_PLOT3D_TIMING:
            print(f"[DEBUG Vertical] Found {len(sample_payloads)} surface sample payloads")
            for i, payload in enumerate(sample_payloads):
                kind = payload.get("kind", "planar")
                surface_id = payload.get("surface_id", "unknown")
                print(f"  Payload {i}: surface_id={surface_id}, kind={kind}")
        
        for payload in sample_payloads:
            # Nur Payloads verarbeiten, die explizit als "vertical" markiert sind.
            kind = payload.get("kind", "planar")
            if kind != "vertical":
                if DEBUG_PLOT3D_TIMING:
                    surface_id_debug = payload.get("surface_id", "unknown")
                    print(f"[DEBUG Vertical] Skipping {surface_id_debug}: kind={kind} (not 'vertical')")
                continue

            surface_id = payload.get("surface_id")
            if surface_id is None:
                continue
            
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical] Processing vertical surface: {surface_id}")

            # Nur Surfaces zeichnen, die aktuell enabled und nicht hidden sind
            surf_def = surface_definitions.get(surface_id)
            if surf_def is None:
                continue
            if hasattr(surf_def, "to_dict"):
                surf_data = surf_def.to_dict()
            elif isinstance(surf_def, dict):
                surf_data = surf_def
            else:
                surf_data = {
                    "enabled": getattr(surf_def, "enabled", False),
                    "hidden": getattr(surf_def, "hidden", False),
                    "points": getattr(surf_def, "points", []),
                }
            if not surf_data.get("enabled", False) or surf_data.get("hidden", False):
                continue

            # Lokale vertikale Plot-Geometrie aufbauen (u,v-Grid, SPL-Werte, Maske)
            try:
                geom: VerticalPlotGeometry | None = prepare_vertical_plot_geometry(
                    surface_id,
                    self.settings,
                    container,
                    default_upscale=getattr(self, "UPSCALE_FACTOR", 1),
                )
            except Exception:
                geom = None

            if geom is None:
                continue

            # Strukturiertes Mesh in Weltkoordinaten erstellen
            try:
                grid = build_vertical_surface_mesh(geom, pv_module=pv)
            except Exception:
                continue

            # Color step: Werte in diskrete Stufen quantisieren, analog zur Hauptfläche.
            if is_step_mode and "plot_scalars" in grid.array_names:
                try:
                    vals = np.asarray(grid["plot_scalars"], dtype=float)
                    grid["plot_scalars"] = self._quantize_to_steps(vals, cbar_step)
                except Exception:
                    pass

            actor_name = f"vertical_spl_{surface_id}"
            # Entferne ggf. alten Actor
            try:
                if actor_name in self.plotter.renderer.actors:
                    self.plotter.remove_actor(actor_name)
            except Exception:
                pass

            # Farbschema und CLim an das Haupt-SPL anlehnen
            cbar_min, cbar_max = self._get_vertical_color_limits()
            try:
                actor = self.plotter.add_mesh(
                    grid,
                    name=actor_name,
                    scalars="plot_scalars",
                    cmap="jet",
                    clim=(cbar_min, cbar_max),
                    # Gradient: weiche Darstellung, Color step: harte Stufen.
                    smooth_shading=not is_step_mode,
                    show_scalar_bar=False,
                    reset_camera=False,
                    interpolate_before_map=not is_step_mode,
                )
                # Stelle sicher, dass senkrechte Flächen pickable sind
                if actor and hasattr(actor, 'SetPickable'):
                    actor.SetPickable(True)
                # Im Color-Step-Modus explizit flache Interpolation erzwingen,
                # damit die Stufen wie bei der horizontalen Fläche erscheinen.
                if is_step_mode and hasattr(actor, "prop") and actor.prop is not None:
                    try:
                        actor.prop.interpolation = "flat"
                    except Exception:  # noqa: BLE001
                        pass
                new_vertical_meshes[actor_name] = actor
            except Exception:
                continue

        # Alte Actors entfernen, die nicht mehr gebraucht werden
        if hasattr(self, "_vertical_surface_meshes"):
            for old_name in list(self._vertical_surface_meshes.keys()):
                if old_name not in new_vertical_meshes:
                    try:
                        if old_name in self.plotter.renderer.actors:
                            self.plotter.remove_actor(old_name)
                    except Exception:
                        pass

        self._vertical_surface_meshes = new_vertical_meshes

    def _update_vertical_spl_surfaces_from_grids(self) -> None:
        """
        🎯 DEPRECATED: Vertikale Surfaces werden jetzt identisch wie planare Surfaces behandelt.
        
        Diese Funktion ist nicht mehr nötig, da vertikale Flächen jetzt in _render_surfaces_textured()
        mit der gleichen Triangulation-Logik wie planare Flächen geplottet werden.
        
        Die Funktion bleibt für Rückwärtskompatibilität, macht aber nichts mehr.
        """
        # 🎯 VERTIKALE FLÄCHEN: Werden jetzt in _render_surfaces_textured behandelt
        # Diese separate Funktion ist nicht mehr nötig
        if DEBUG_PLOT3D_TIMING:
            print("[DEBUG Vertical Grids] Vertikale Plots werden jetzt in _render_surfaces_textured behandelt – skip separate Funktion")
        self._clear_vertical_spl_surfaces()
        return
        if DEBUG_PLOT3D_TIMING:
            print(f"[DEBUG Vertical Grids] Starte _update_vertical_spl_surfaces_from_grids()")
        
        container = getattr(self, "container", None)
        if container is None or not hasattr(container, "calculation_spl"):
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical Grids] Kein Container oder calculation_spl")
            self._clear_vertical_spl_surfaces()
            return

        calc_spl = getattr(container, "calculation_spl", {}) or {}
        surface_grids_data = calc_spl.get("surface_grids", {})
        surface_results_data = calc_spl.get("surface_results", {})
        
        if not surface_grids_data or not surface_results_data:
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical Grids] Keine surface_grids_data oder surface_results_data")
            self._clear_vertical_spl_surfaces()
            return
        
        if DEBUG_PLOT3D_TIMING:
            print(f"[DEBUG Vertical Grids] Gefunden: {len(surface_grids_data)} Surfaces in surface_grids_data")

        # Aktueller Surface-Status (enabled/hidden) aus den Settings
        surface_definitions = getattr(self.settings, "surface_definitions", {})
        if not isinstance(surface_definitions, dict):
            surface_definitions = {}

        # Aktuellen Colorization-Mode verwenden
        colorization_mode = getattr(self, "_last_colorization_mode", None)
        if colorization_mode not in {"Color step", "Gradient"}:
            colorization_mode = getattr(self.settings, "colorization_mode", "Gradient")
        try:
            cbar_range = getattr(self.settings, "colorbar_range", {})
            cbar_step = float(cbar_range.get("step", 0.0))
        except Exception:
            cbar_step = 0.0
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0
        
        # Plot-Mode bestimmen
        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'
        time_mode = plot_mode == 'SPL over time'
        
        # Colorbar-Parameter
        cbar_min, cbar_max = self._get_vertical_color_limits()
        
        # Colormap
        if time_mode:
            base_cmap = 'RdBu_r'
        elif phase_mode:
            base_cmap = PHASE_CMAP
        else:
            base_cmap = 'jet'
        
        # 🎯 Für Color Step: Erstelle diskrete Colormap mit exakten Levels (wie Colorbar)
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
            cmap_object = base_cmap
        
        new_vertical_meshes: dict[str, Any] = {}
        
        # Finde alle vertikalen Surfaces
        if DEBUG_PLOT3D_TIMING:
            print(f"[DEBUG Vertical Grids] Prüfe {len(surface_grids_data)} Surfaces auf vertikale Orientierung")
        
        for surface_id in surface_grids_data.keys():
            if surface_id not in surface_results_data:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Vertical Grids] Surface '{surface_id}': Nicht in surface_results_data")
                continue
            
            grid_data = surface_grids_data[surface_id]
            orientation = grid_data.get('orientation', 'unknown')
            
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical Grids] Surface '{surface_id}': orientation={orientation}")
            
            # Prüfe, ob Surface vertikal ist (entweder "vertical" oder "sloped" mit vertikaler Ausrichtung)
            is_vertical = False
            if orientation == 'vertical':
                is_vertical = True
            elif orientation == 'sloped':
                # Prüfe, ob es eine schräge vertikale Fläche ist (z_span > max(x_span, y_span) * 0.5)
                # Hole Surface-Definition für Koordinatenprüfung
                surf_def = surface_definitions.get(surface_id)
                if surf_def is not None:
                    if hasattr(surf_def, "to_dict"):
                        surf_data = surf_def.to_dict()
                    elif isinstance(surf_def, dict):
                        surf_data = surf_def
                    else:
                        surf_data = {}
                    points = surf_data.get('points', [])
                    if len(points) >= 3:
                        xs = np.array([p.get('x', 0.0) if isinstance(p, dict) else getattr(p, 'x', 0.0) for p in points], dtype=float)
                        ys = np.array([p.get('y', 0.0) if isinstance(p, dict) else getattr(p, 'y', 0.0) for p in points], dtype=float)
                        zs = np.array([p.get('z', 0.0) if isinstance(p, dict) else getattr(p, 'z', 0.0) for p in points], dtype=float)
                        x_span = float(np.ptp(xs))
                        y_span = float(np.ptp(ys))
                        z_span = float(np.ptp(zs))
                        # Schräge vertikale Fläche: z_span ist signifikant größer als x_span und y_span
                        if z_span > max(x_span, y_span) * 0.5 and z_span > 1e-3:
                            is_vertical = True
                            if DEBUG_PLOT3D_TIMING:
                                print(f"[DEBUG Vertical Grids] Surface '{surface_id}': Schräge vertikale Fläche erkannt (z_span={z_span:.3f} > max(x_span={x_span:.3f}, y_span={y_span:.3f}) * 0.5)")
            
            if not is_vertical:
                continue
            
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical Grids] ✅ Surface '{surface_id}' ist vertikal, verarbeite...")
            
            # Prüfe ob Surface enabled ist
            surf_def = surface_definitions.get(surface_id)
            if surf_def is None:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Vertical Grids] Surface '{surface_id}': Nicht in surface_definitions")
                continue
            if hasattr(surf_def, "to_dict"):
                surf_data = surf_def.to_dict()
            elif isinstance(surf_def, dict):
                surf_data = surf_def
            else:
                surf_data = {
                    "enabled": getattr(surf_def, "enabled", False),
                    "hidden": getattr(surf_def, "hidden", False),
                    "points": getattr(surf_def, "points", []),
                }
            enabled = surf_data.get("enabled", False)
            hidden = surf_data.get("hidden", False)
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Vertical Grids] Surface '{surface_id}': enabled={enabled}, hidden={hidden}")
            if not enabled or hidden:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Vertical Grids] Surface '{surface_id}': Überspringe (nicht enabled oder hidden)")
                continue
            
            try:
                # 🎯 IDENTISCH ZU HORIZONTALEN FLÄCHEN: Verwende Grid-Daten direkt
                # Lade Grid-Daten (bereits in 3D-Koordinaten)
                X_grid = np.array(grid_data['X_grid'], dtype=float)
                Y_grid = np.array(grid_data['Y_grid'], dtype=float)
                Z_grid = np.array(grid_data['Z_grid'], dtype=float)
                sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)
                sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)
                surface_mask = np.array(grid_data['surface_mask'], dtype=bool)
                
                if DEBUG_PLOT3D_TIMING:
                    active_points = int(np.sum(surface_mask))
                    print(f"[DEBUG Vertical Grids] Surface '{surface_id}': Grid-Shape={X_grid.shape}, Active={active_points}/{X_grid.size}")
                    print(f"  └─ [DEBUG] X_grid nach Laden: min={X_grid.min():.2f}, max={X_grid.max():.2f}, shape={X_grid.shape}")
                    print(f"  └─ [DEBUG] Y_grid nach Laden: min={Y_grid.min():.2f}, max={Y_grid.max():.2f}, shape={Y_grid.shape}")
                    print(f"  └─ [DEBUG] Z_grid nach Laden: min={Z_grid.min():.2f}, max={Z_grid.max():.2f}, shape={Z_grid.shape}")
                    print(f"  └─ [DEBUG] sound_field_x: min={sound_field_x.min():.2f}, max={sound_field_x.max():.2f}, len={len(sound_field_x)}")
                    print(f"  └─ [DEBUG] sound_field_y: min={sound_field_y.min():.2f}, max={sound_field_y.max():.2f}, len={len(sound_field_y)}")
                    
                    if active_points == 0:
                        print(f"[DEBUG Vertical Grids] ⚠️ Surface '{surface_id}': KEINE aktiven Punkte in surface_mask!")
                        continue
                
                # Lade SPL-Werte
                result_data = surface_results_data[surface_id]
                sound_field_p_complex = np.array(result_data['sound_field_p'], dtype=complex)
                
                # Konvertiere zu SPL in dB (identisch zu horizontalen Flächen)
                if time_mode:
                    spl_values = np.real(sound_field_p_complex)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                    spl_values = np.clip(spl_values, cbar_min, cbar_max)
                elif phase_mode:
                    spl_values = np.angle(sound_field_p_complex)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                    spl_values = np.clip(spl_values, cbar_min, cbar_max)
                else:
                    pressure_magnitude = np.abs(sound_field_p_complex)
                    pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                    spl_values = self.functions.mag2db(pressure_magnitude)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                ny, nx = X_grid.shape
                
                # 🎯 BESTIMME ORIENTIERUNG: X-Z oder Y-Z für Koordinatentransformation
                points = surf_data.get('points', [])
                vertical_orientation = None
                wall_value = None
                if len(points) >= 3:
                    xs = np.array([p.get('x', 0.0) for p in points], dtype=float)
                    ys = np.array([p.get('y', 0.0) for p in points], dtype=float)
                    zs = np.array([p.get('z', 0.0) for p in points], dtype=float)
                    x_span = float(np.ptp(xs))
                    y_span = float(np.ptp(ys))
                    z_span = float(np.ptp(zs))
                    
                    eps_line = 1e-6
                    if y_span < eps_line and x_span >= eps_line and z_span >= eps_line:
                        # X-Z-Wand: y ≈ const
                        vertical_orientation = "xz"
                        wall_value = float(np.mean(ys))  # Konstanter Y-Wert
                    elif x_span < eps_line and y_span >= eps_line and z_span >= eps_line:
                        # Y-Z-Wand: x ≈ const
                        vertical_orientation = "yz"
                        wall_value = float(np.mean(xs))  # Konstanter X-Wert
                    elif z_span > max(x_span, y_span) * 0.5 and x_span >= eps_line and y_span >= eps_line:
                        # Schräge vertikale Surface: z_span ist signifikant UND beide x_span und y_span variieren
                        # Diese Fläche liegt schräg im Raum und muss in ihrer lokalen (u,v)-Ebene behandelt werden
                        # u = Projektion auf XY-Ebene (entlang der längsten Ausdehnung), v = Z
                        # Bestimme dominante Richtung in XY
                        if x_span >= y_span:
                            # Dominante Richtung ist X → u = X, v = Z, Y variiert
                            vertical_orientation = "xz_slanted"
                            # Y variiert entlang der Fläche, wir verwenden den Mittelwert als Referenz
                            wall_value = float(np.mean(ys))
                        else:
                            # Dominante Richtung ist Y → u = Y, v = Z, X variiert
                            vertical_orientation = "yz_slanted"
                            # X variiert entlang der Fläche, wir verwenden den Mittelwert als Referenz
                            wall_value = float(np.mean(xs))
                    elif x_span >= eps_line and y_span >= eps_line and z_span >= eps_line:
                        # Schräge vertikale Surface: Variiert in X, Y und Z
                        # Bestimme dominante Richtung in XY basierend auf dominant_axis
                        if hasattr(surface_def, 'dominant_axis') and surface_def.dominant_axis:
                            if surface_def.dominant_axis == "xz":
                                vertical_orientation = "xz_slanted"
                                wall_value = float(np.mean(ys))
                            elif surface_def.dominant_axis == "yz":
                                vertical_orientation = "yz_slanted"
                                wall_value = float(np.mean(xs))
                            else:
                                raise ValueError(f"Surface '{surface_id}': Unbekannter dominant_axis '{surface_def.dominant_axis}' für schräge vertikale Surface")
                        else:
                            raise ValueError(f"Surface '{surface_id}': dominant_axis nicht verfügbar für schräge vertikale Surface")
                
                # 🎯 DEBUG: Zeige Koordinaten-Informationen
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Vertical Plot] Surface '{surface_id}':")
                    print(f"  └─ Orientierung: {vertical_orientation}")
                    print(f"  └─ Wall-Value: {wall_value}")
                    print(f"  └─ Grid-Shape: {X_grid.shape}")
                    print(f"  └─ X_grid: min={X_grid.min():.2f}, max={X_grid.max():.2f}, span={X_grid.max()-X_grid.min():.2f}")
                    print(f"  └─ Y_grid: min={Y_grid.min():.2f}, max={Y_grid.max():.2f}, span={Y_grid.max()-Y_grid.min():.2f}")
                    print(f"  └─ Z_grid: min={Z_grid.min():.2f}, max={Z_grid.max():.2f}, span={Z_grid.max()-Z_grid.min():.2f}")
                    print(f"  └─ sound_field_x: min={sound_field_x.min():.2f}, max={sound_field_x.max():.2f}, len={len(sound_field_x)}")
                    print(f"  └─ sound_field_y: min={sound_field_y.min():.2f}, max={sound_field_y.max():.2f}, len={len(sound_field_y)}")
                    if len(points) >= 3:
                        print(f"  └─ Surface-Punkte: x_span={x_span:.6f}, y_span={y_span:.6f}, z_span={z_span:.6f}")
                
                # 🎯 KOORDINATENTRANSFORMATION: Für vertikale Flächen müssen wir die Koordinaten transformieren
                # WICHTIG: sound_field_x und sound_field_y sind IMMER in XY-Ebene erstellt!
                # Für vertikale Flächen müssen wir die tatsächlichen Koordinaten aus X_grid, Y_grid, Z_grid extrahieren
                # build_surface_mesh erwartet (x, y) als 2D-Grid und z_coords als Höhe
                # Für X-Z-Wand: (X, Z) → (x, y) für build_surface_mesh, Y konstant → z_coords
                # Für Y-Z-Wand: (Y, Z) → (x, y) für build_surface_mesh, X konstant → z_coords
                if vertical_orientation == "xz":
                    # X-Z-Wand: y ≈ const
                    # 🎯 Verwende sound_field_x und sound_field_y direkt (bereits korrekt aus FlexibleGridGenerator)
                    # sound_field_x enthält X-Koordinaten, sound_field_y enthält Z-Koordinaten
                    u_axis = sound_field_x  # X-Koordinaten (bereits sortiert)
                    v_axis = sound_field_y  # Z-Koordinaten (bereits sortiert)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ X-Z-Wand: u_axis (X) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "yz":
                    # Y-Z-Wand: x ≈ const
                    # 🎯 Verwende sound_field_x und sound_field_y direkt (bereits korrekt aus FlexibleGridGenerator)
                    # sound_field_x enthält Y-Koordinaten, sound_field_y enthält Z-Koordinaten
                    u_axis = sound_field_x  # Y-Koordinaten (bereits sortiert)
                    v_axis = sound_field_y  # Z-Koordinaten (bereits sortiert)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Y-Z-Wand: u_axis (Y) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "xz_slanted":
                    # Schräge X-Z-Wand: X und Z variieren, Y variiert entlang der Fläche
                    # u = X (dominante Richtung), v = Z
                    # Y wird aus der Flächengeometrie interpoliert
                    # 🎯 Verwende sound_field_x und sound_field_y direkt (bereits korrekt aus FlexibleGridGenerator)
                    # sound_field_x enthält X-Koordinaten, sound_field_y enthält Z-Koordinaten
                    u_axis = sound_field_x  # X-Koordinaten (bereits sortiert)
                    v_axis = sound_field_y  # Z-Koordinaten (bereits sortiert)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Schräge X-Z-Wand: u_axis (X) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "yz_slanted":
                    # Schräge Y-Z-Wand: Y und Z variieren, X variiert entlang der Fläche
                    # u = Y (dominante Richtung), v = Z
                    # X wird aus der Flächengeometrie interpoliert
                    # 🎯 Verwende sound_field_x und sound_field_y direkt (bereits korrekt aus FlexibleGridGenerator)
                    # sound_field_x enthält Y-Koordinaten, sound_field_y enthält Z-Koordinaten
                    u_axis = sound_field_x  # Y-Koordinaten (bereits sortiert)
                    v_axis = sound_field_y  # Z-Koordinaten (bereits sortiert)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Schräge Y-Z-Wand: u_axis (Y) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                else:
                    # Keine Orientierung erkannt - Fehler
                    raise ValueError(f"Surface '{surface_id}': Keine Orientierung erkannt für vertikale Surface")
                
                # 🎯 UPSCALING: Erhöhe Grid-Auflösung (in UV-Koordinaten)
                upscale_factor = getattr(self, 'UPSCALE_FACTOR', 1)
                if upscale_factor > 1:
                    # Erstelle feineres Grid basierend auf den originalen UV-Koordinaten
                    u_fine = np.linspace(u_axis.min(), u_axis.max(), len(u_axis) * upscale_factor)
                    v_fine = np.linspace(v_axis.min(), v_axis.max(), len(v_axis) * upscale_factor)
                    U_fine, V_fine = np.meshgrid(u_fine, v_fine, indexing='xy')
                    
                    # 🎯 INTERPOLATION: Erstelle Meshgrid aus originalen Koordinaten für Interpolation
                    if vertical_orientation == "xz":
                        # X-Z-Wand: Original-Grid ist (X, Y, Z), wir brauchen (X, Z) mit Y=wall_value
                        # Interpoliere SPL-Werte von originalen 3D-Koordinaten auf neue 3D-Koordinaten
                        from scipy.interpolate import griddata
                        # Original-Koordinaten: (X, Y, Z) - die tatsächlichen 3D-Grid-Koordinaten
                        points_orig = np.column_stack([
                            X_grid.flatten(),
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        # Neue 3D-Koordinaten: (X, Y=wall_value, Z) = (U_fine, wall_value, V_fine)
                        Y_new = np.full_like(U_fine, wall_value, dtype=float)
                        points_new = np.column_stack([
                            U_fine.ravel(),
                            Y_new.ravel(),
                            V_fine.ravel()
                        ])
                        # Interpoliere SPL-Werte von (X, Y, Z) auf (X, Y=wall_value, Z)
                        if is_step_mode:
                            spl_fine = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_fine = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='linear', fill_value=0.0
                            )
                        spl_fine = spl_fine.reshape(U_fine.shape)
                        # Interpoliere Maske von (X, Y, Z) auf (X, Y=wall_value, Z)
                        mask_fine = griddata(
                            points_orig, surface_mask.flatten().astype(float), points_new,
                            method='nearest', fill_value=0.0
                        )
                        mask_fine = mask_fine.reshape(U_fine.shape).astype(bool)
                        # Z_fine = Y konstant
                        Z_fine = np.full_like(U_fine, wall_value, dtype=float)
                    elif vertical_orientation == "yz":
                        # Y-Z-Wand: Original-Grid ist (X, Y, Z), wir brauchen (Y, Z) mit X=wall_value
                        # Interpoliere SPL-Werte von originalen 3D-Koordinaten auf neue 3D-Koordinaten
                        from scipy.interpolate import griddata
                        # Original-Koordinaten: (X, Y, Z) - die tatsächlichen 3D-Grid-Koordinaten
                        points_orig = np.column_stack([
                            X_grid.flatten(),
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        # Neue 3D-Koordinaten: (X=wall_value, Y, Z) = (wall_value, U_fine, V_fine)
                        X_new = np.full_like(U_fine, wall_value, dtype=float)
                        points_new = np.column_stack([
                            X_new.ravel(),
                            U_fine.ravel(),
                            V_fine.ravel()
                        ])
                        # Interpoliere SPL-Werte von (X, Y, Z) auf (X=wall_value, Y, Z)
                        if is_step_mode:
                            spl_fine = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_fine = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='linear', fill_value=0.0
                            )
                        spl_fine = spl_fine.reshape(U_fine.shape)
                        # Interpoliere Maske von (X, Y, Z) auf (X=wall_value, Y, Z)
                        mask_fine = griddata(
                            points_orig, surface_mask.flatten().astype(float), points_new,
                            method='nearest', fill_value=0.0
                        )
                        mask_fine = mask_fine.reshape(U_fine.shape).astype(bool)
                        # Z_fine = X konstant
                        Z_fine = np.full_like(U_fine, wall_value, dtype=float)
                    elif vertical_orientation == "xz_slanted":
                        # Schräge X-Z-Wand: X und Z variieren, Y variiert entlang der Fläche
                        # Y_grid wurde bereits in FlexibleGridGenerator interpoliert
                        # Interpoliere nur in 2D (X, Z) auf (U_fine, V_fine), nicht in 3D
                        from scipy.interpolate import griddata
                        # Original-Koordinaten in 2D (u,v) = (X, Z)
                        points_orig_2d = np.column_stack([
                            X_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        # Neue 2D-Koordinaten: (U_fine, V_fine) = (X, Z)
                        points_new_2d = np.column_stack([
                            U_fine.ravel(),
                            V_fine.ravel()
                        ])
                        # Interpoliere Y-Koordinaten in 2D (X, Z) → (U_fine, V_fine)
                        # Y_grid wurde bereits interpoliert, wir interpolieren nur auf feineres Grid
                        Y_interp = griddata(
                            points_orig_2d, Y_grid.flatten(), points_new_2d,
                            method='linear', fill_value=wall_value
                        )
                        Y_interp = Y_interp.reshape(U_fine.shape)
                        # Interpoliere SPL-Werte in 2D (X, Z) → (U_fine, V_fine)
                        if is_step_mode:
                            spl_fine = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_fine = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='linear', fill_value=0.0
                            )
                        spl_fine = spl_fine.reshape(U_fine.shape)
                        # Interpoliere Maske in 2D (X, Z) → (U_fine, V_fine)
                        mask_fine = griddata(
                            points_orig_2d, surface_mask.flatten().astype(float), points_new_2d,
                            method='nearest', fill_value=0.0
                        )
                        mask_fine = mask_fine.reshape(U_fine.shape).astype(bool)
                        # Z_fine = interpoliertes Y
                        Z_fine = Y_interp
                    elif vertical_orientation == "yz_slanted":
                        # Schräge Y-Z-Wand: Y und Z variieren, X variiert entlang der Fläche
                        # X_grid wurde bereits in FlexibleGridGenerator interpoliert
                        # Interpoliere nur in 2D (Y, Z) auf (U_fine, V_fine), nicht in 3D
                        from scipy.interpolate import griddata
                        # Original-Koordinaten in 2D (u,v) = (Y, Z)
                        points_orig_2d = np.column_stack([
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        # Neue 2D-Koordinaten: (U_fine, V_fine) = (Y, Z)
                        points_new_2d = np.column_stack([
                            U_fine.ravel(),
                            V_fine.ravel()
                        ])
                        # Interpoliere X-Koordinaten in 2D (Y, Z) → (U_fine, V_fine)
                        # X_grid wurde bereits interpoliert, wir interpolieren nur auf feineres Grid
                        X_interp = griddata(
                            points_orig_2d, X_grid.flatten(), points_new_2d,
                            method='linear', fill_value=wall_value
                        )
                        X_interp = X_interp.reshape(U_fine.shape)
                        # Interpoliere SPL-Werte in 2D (Y, Z) → (U_fine, V_fine)
                        if is_step_mode:
                            spl_fine = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_fine = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='linear', fill_value=0.0
                            )
                        spl_fine = spl_fine.reshape(U_fine.shape)
                        # Interpoliere Maske in 2D (Y, Z) → (U_fine, V_fine)
                        mask_fine = griddata(
                            points_orig_2d, surface_mask.flatten().astype(float), points_new_2d,
                            method='nearest', fill_value=0.0
                        )
                        mask_fine = mask_fine.reshape(U_fine.shape).astype(bool)
                        # Z_fine = interpoliertes X
                        Z_fine = X_interp
                    else:
                        # Unbekannte vertikale Orientierung - Fehler
                        raise ValueError(f"Surface '{surface_id}': Unbekannte vertikale Orientierung '{vertical_orientation}'")
                    
                    # Transformiere für build_surface_mesh
                    if vertical_orientation == "xz":
                        # (U, V) = (X, Z) → (x, y) für build_surface_mesh, Y konstant → z_coords
                        x_plot = u_fine  # X-Koordinaten
                        y_plot = v_fine  # Z-Koordinaten
                        z_coords_plot = np.full_like(U_fine, wall_value, dtype=float)  # Y konstant
                        # 3D-Koordinaten für finales Mesh
                        X_plot = U_fine  # X-Koordinaten
                        Y_plot = np.full_like(U_fine, wall_value, dtype=float)  # Y konstant
                        Z_plot = V_fine  # Z-Koordinaten
                    elif vertical_orientation == "yz":
                        # (U, V) = (Y, Z) → (x, y) für build_surface_mesh, X konstant → z_coords
                        x_plot = u_fine  # Y-Koordinaten
                        y_plot = v_fine  # Z-Koordinaten
                        z_coords_plot = np.full_like(U_fine, wall_value, dtype=float)  # X konstant
                        # 3D-Koordinaten für finales Mesh
                        X_plot = np.full_like(U_fine, wall_value, dtype=float)  # X konstant
                        Y_plot = U_fine  # Y-Koordinaten
                        Z_plot = V_fine  # Z-Koordinaten
                    elif vertical_orientation == "xz_slanted":
                        # Schräge X-Z-Wand: (U, V) = (X, Z) → (x, y) für build_surface_mesh, Y interpoliert → z_coords
                        x_plot = u_fine  # X-Koordinaten
                        y_plot = v_fine  # Z-Koordinaten
                        z_coords_plot = Z_fine  # Y interpoliert (aus Z_fine, das Y_interp enthält)
                        # 3D-Koordinaten für finales Mesh
                        X_plot = U_fine  # X-Koordinaten
                        Y_plot = Z_fine  # Y interpoliert
                        Z_plot = V_fine  # Z-Koordinaten
                    elif vertical_orientation == "yz_slanted":
                        # Schräge Y-Z-Wand: (U, V) = (Y, Z) → (x, y) für build_surface_mesh, X interpoliert → z_coords
                        x_plot = u_fine  # Y-Koordinaten
                        y_plot = v_fine  # Z-Koordinaten
                        z_coords_plot = Z_fine  # X interpoliert (aus Z_fine, das X_interp enthält)
                        # 3D-Koordinaten für finales Mesh
                        X_plot = Z_fine  # X interpoliert
                        Y_plot = U_fine  # Y-Koordinaten
                        Z_plot = V_fine  # Z-Koordinaten
                    else:
                        x_plot = u_fine
                        y_plot = v_fine
                        z_coords_plot = Z_fine
                        X_plot = U_fine
                        Y_plot = V_fine
                        Z_plot = Z_fine
                    
                    if DEBUG_PLOT3D_TIMING:
                        active_coarse = int(np.sum(surface_mask))
                        active_fine = int(np.sum(mask_fine))
                        print(
                            f"[DEBUG Plot] Upscale x{upscale_factor} (vertical {vertical_orientation}): "
                            f"coarse {surface_mask.shape}->{active_coarse} aktiv, "
                            f"fine {mask_fine.shape}->{active_fine} aktiv"
                        )
                    
                    spl_plot = spl_fine
                    mask_plot = mask_fine
                else:
                    # Kein Upscaling
                    # Erstelle Meshgrid aus u_axis und v_axis
                    U_plot, V_plot = np.meshgrid(u_axis, v_axis, indexing='xy')
                    
                    # Transformiere für build_surface_mesh
                    if vertical_orientation == "xz":
                        # (U, V) = (X, Z) → (x, y) für build_surface_mesh, Y konstant → z_coords
                        x_plot = u_axis  # X-Koordinaten
                        y_plot = v_axis  # Z-Koordinaten
                        z_coords_plot = np.full_like(U_plot, wall_value, dtype=float)  # Y konstant
                        # 3D-Koordinaten für finales Mesh
                        X_plot = U_plot  # X-Koordinaten
                        Y_plot = np.full_like(U_plot, wall_value, dtype=float)  # Y konstant
                        Z_plot = V_plot  # Z-Koordinaten
                        
                        # Interpoliere SPL-Werte von (X, Y, Z) auf (X, Y=wall_value, Z)
                        from scipy.interpolate import griddata
                        points_orig = np.column_stack([
                            X_grid.flatten(),
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        Y_new = np.full_like(U_plot, wall_value, dtype=float)
                        points_new = np.column_stack([
                            U_plot.ravel(),
                            Y_new.ravel(),
                            V_plot.ravel()
                        ])
                        if is_step_mode:
                            spl_plot = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_plot = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='linear', fill_value=0.0
                            )
                        spl_plot = spl_plot.reshape(U_plot.shape)
                        # Interpoliere Maske
                        mask_plot = griddata(
                            points_orig, surface_mask.flatten().astype(float), points_new,
                            method='nearest', fill_value=0.0
                        )
                        mask_plot = mask_plot.reshape(U_plot.shape).astype(bool)
                    elif vertical_orientation == "yz":
                        # (U, V) = (Y, Z) → (x, y) für build_surface_mesh, X konstant → z_coords
                        x_plot = u_axis  # Y-Koordinaten
                        y_plot = v_axis  # Z-Koordinaten
                        z_coords_plot = np.full_like(U_plot, wall_value, dtype=float)  # X konstant
                        # 3D-Koordinaten für finales Mesh
                        X_plot = np.full_like(U_plot, wall_value, dtype=float)  # X konstant
                        Y_plot = U_plot  # Y-Koordinaten
                        Z_plot = V_plot  # Z-Koordinaten
                        
                        # Interpoliere SPL-Werte von (X, Y, Z) auf (X=wall_value, Y, Z)
                        from scipy.interpolate import griddata
                        points_orig = np.column_stack([
                            X_grid.flatten(),
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        X_new = np.full_like(U_plot, wall_value, dtype=float)
                        points_new = np.column_stack([
                            X_new.ravel(),
                            U_plot.ravel(),
                            V_plot.ravel()
                        ])
                        if is_step_mode:
                            spl_plot = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_plot = griddata(
                                points_orig, spl_values.flatten(), points_new,
                                method='linear', fill_value=0.0
                            )
                        spl_plot = spl_plot.reshape(U_plot.shape)
                        # Interpoliere Maske
                        mask_plot = griddata(
                            points_orig, surface_mask.flatten().astype(float), points_new,
                            method='nearest', fill_value=0.0
                        )
                        mask_plot = mask_plot.reshape(U_plot.shape).astype(bool)
                    elif vertical_orientation == "xz_slanted":
                        # Schräge X-Z-Wand: (U, V) = (X, Z) → (x, y) für build_surface_mesh, Y interpoliert → z_coords
                        # Y_grid wurde bereits in FlexibleGridGenerator interpoliert
                        x_plot = u_axis  # X-Koordinaten
                        y_plot = v_axis  # Z-Koordinaten
                        # Y_grid ist bereits interpoliert, verwende direkt
                        z_coords_plot = Y_grid  # Y interpoliert (bereits vorhanden)
                        # 3D-Koordinaten für finales Mesh
                        X_plot = U_plot  # X-Koordinaten
                        Y_plot = Y_grid  # Y interpoliert (bereits vorhanden)
                        Z_plot = V_plot  # Z-Koordinaten
                        # Interpoliere SPL-Werte in 2D (X, Z) → (U_plot, V_plot)
                        from scipy.interpolate import griddata
                        points_orig_2d = np.column_stack([
                            X_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        points_new_2d = np.column_stack([
                            U_plot.ravel(),
                            V_plot.ravel()
                        ])
                        if is_step_mode:
                            spl_plot = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_plot = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='linear', fill_value=0.0
                            )
                        spl_plot = spl_plot.reshape(U_plot.shape)
                        # Interpoliere Maske in 2D (X, Z) → (U_plot, V_plot)
                        mask_plot = griddata(
                            points_orig_2d, surface_mask.flatten().astype(float), points_new_2d,
                            method='nearest', fill_value=0.0
                        )
                        mask_plot = mask_plot.reshape(U_plot.shape).astype(bool)
                    elif vertical_orientation == "yz_slanted":
                        # Schräge Y-Z-Wand: (U, V) = (Y, Z) → (x, y) für build_surface_mesh, X interpoliert → z_coords
                        # X_grid wurde bereits in FlexibleGridGenerator interpoliert
                        x_plot = u_axis  # Y-Koordinaten
                        y_plot = v_axis  # Z-Koordinaten
                        # X_grid ist bereits interpoliert, verwende direkt
                        z_coords_plot = X_grid  # X interpoliert (bereits vorhanden)
                        # 3D-Koordinaten für finales Mesh
                        X_plot = X_grid  # X interpoliert (bereits vorhanden)
                        Y_plot = U_plot  # Y-Koordinaten
                        Z_plot = V_plot  # Z-Koordinaten
                        # Interpoliere SPL-Werte in 2D (Y, Z) → (U_plot, V_plot)
                        from scipy.interpolate import griddata
                        points_orig_2d = np.column_stack([
                            Y_grid.flatten(),
                            Z_grid.flatten()
                        ])
                        points_new_2d = np.column_stack([
                            U_plot.ravel(),
                            V_plot.ravel()
                        ])
                        if is_step_mode:
                            spl_plot = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='nearest', fill_value=0.0
                            )
                        else:
                            spl_plot = griddata(
                                points_orig_2d, spl_values.flatten(), points_new_2d,
                                method='linear', fill_value=0.0
                            )
                        spl_plot = spl_plot.reshape(U_plot.shape)
                        # Interpoliere Maske in 2D (Y, Z) → (U_plot, V_plot)
                        mask_plot = griddata(
                            points_orig_2d, surface_mask.flatten().astype(float), points_new_2d,
                            method='nearest', fill_value=0.0
                        )
                        mask_plot = mask_plot.reshape(U_plot.shape).astype(bool)
                    else:
                        x_plot = sound_field_x
                        y_plot = sound_field_y
                        z_coords_plot = Z_grid
                        X_plot = X_grid
                        Y_plot = Y_grid
                        Z_plot = Z_grid
                        spl_plot = spl_values
                        mask_plot = surface_mask
                
                # 🎯 DEBUG: Zeige transformierte Koordinaten (nur für schräge/überhängende Flächen)
                is_slanted = vertical_orientation in ("xz_slanted", "yz_slanted")
                if DEBUG_PLOT3D_TIMING and is_slanted:
                    print(f"  └─ [{vertical_orientation}] Nach Transformation:")
                    print(f"     └─ X_plot: min={X_plot.min():.2f}, max={X_plot.max():.2f}, span={X_plot.max()-X_plot.min():.2f}")
                    print(f"     └─ Y_plot: min={Y_plot.min():.2f}, max={Y_plot.max():.2f}, span={Y_plot.max()-Y_plot.min():.2f} {'(interpoliert)' if is_slanted else ''}")
                    print(f"     └─ Z_plot: min={Z_plot.min():.2f}, max={Z_plot.max():.2f}, span={Z_plot.max()-Z_plot.min():.2f}")
                    print(f"     └─ mask_plot: {np.sum(mask_plot)}/{mask_plot.size} Punkte aktiv")
                
                # Color step: Clipping + Quantisierung
                if is_step_mode:
                    # 🎯 DEBUG: Zeige Werte vor Quantisierung (vertikale Surfaces)
                    if DEBUG_PLOT3D_TIMING:
                        valid_mask_plot = mask_plot & np.isfinite(spl_plot)
                        if np.any(valid_mask_plot):
                            spl_valid_plot = spl_plot[valid_mask_plot]
                            print(f"[DEBUG Color Step] Vertikale Surface '{surface_id}': Vor Quantisierung:")
                            print(f"  └─ SPL min: {np.min(spl_valid_plot):.2f} dB")
                            print(f"  └─ SPL max: {np.max(spl_valid_plot):.2f} dB")
                            print(f"  └─ SPL mean: {np.mean(spl_valid_plot):.2f} dB")
                    
                    # Clipping vor der Quantisierung, um Ausreißer außerhalb der Colorbar zu verhindern
                    spl_plot = np.clip(spl_plot, cbar_min, cbar_max)
                    scalars = self._quantize_to_steps(spl_plot, cbar_step)
                    
                    # 🎯 DEBUG: Zeige quantisierte Werte (vertikale Surfaces)
                    if DEBUG_PLOT3D_TIMING:
                        valid_mask_scalars = mask_plot & np.isfinite(scalars)
                        if np.any(valid_mask_scalars):
                            scalars_valid = scalars[valid_mask_scalars]
                            unique_values = np.unique(scalars_valid)
                            print(f"[DEBUG Color Step] Vertikale Surface '{surface_id}': Nach Quantisierung:")
                            print(f"  └─ Quantisierte SPL min: {np.min(scalars_valid):.2f} dB")
                            print(f"  └─ Quantisierte SPL max: {np.max(scalars_valid):.2f} dB")
                            print(f"  └─ Anzahl eindeutiger Stufen: {len(unique_values)}")
                            print(f"  └─ Eindeutige Stufen: {unique_values[:10] if len(unique_values) <= 10 else np.concatenate([unique_values[:5], unique_values[-5:]])}")
                else:
                    scalars = spl_plot
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Color Step] Vertikale Surface '{surface_id}': Gradient-Modus - KEINE Quantisierung")
                
                # 🎯 ERSTELLE MESH: Transformierte Koordinaten für vertikale Flächen
                scalars_for_mesh = np.clip(scalars, cbar_min, cbar_max)
                mesh = build_surface_mesh(
                    x_plot,  # Für X-Z-Wand: X-Koordinaten, für Y-Z-Wand: Y-Koordinaten
                    y_plot,  # Für X-Z-Wand: Z-Koordinaten, für Y-Z-Wand: Z-Koordinaten
                    scalars_for_mesh,
                    z_coords=z_coords_plot,  # Für X-Z-Wand: Y konstant, für Y-Z-Wand: X konstant
                    surface_mask=mask_plot,
                    pv_module=pv,
                    settings=self.settings,
                    container=container,
                )
                
                if mesh is None or mesh.n_points == 0:
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Vertical] Surface '{surface_id}': Mesh ist None oder leer")
                    continue
                
                # 🎯 KORRIGIERE KOORDINATEN: build_surface_mesh erstellt (x, y, z_coords)
                # Für vertikale Flächen müssen wir die Koordinaten richtig transformieren
                if vertical_orientation == "xz":
                    # X-Z-Wand: build_surface_mesh erstellt (X, Z, Y)
                    # Wir brauchen: (X, Y, Z) = (X, wall_value, Z)
                    # Aktuell: mesh.points = (x_plot, y_plot, z_coords) = (X, Z, Y)
                    # Korrigiere zu: (X, Y, Z) = (X, Y=wall_value, Z)
                    points_corrected = np.column_stack([
                        mesh.points[:, 0],  # X (aus x_plot)
                        mesh.points[:, 2],  # Y (aus z_coords, war konstant)
                        mesh.points[:, 1],  # Z (aus y_plot)
                    ])
                    mesh.points = points_corrected
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Vertical] XZ-Wand: Koordinaten korrigiert")
                        print(f"  └─ X: min={points_corrected[:, 0].min():.2f}, max={points_corrected[:, 0].max():.2f}")
                        print(f"  └─ Y: min={points_corrected[:, 1].min():.2f}, max={points_corrected[:, 1].max():.2f} (sollte konstant: {wall_value:.2f})")
                        print(f"  └─ Z: min={points_corrected[:, 2].min():.2f}, max={points_corrected[:, 2].max():.2f}")
                elif vertical_orientation == "yz":
                    # Y-Z-Wand: build_surface_mesh erstellt (Y, Z, X)
                    # Wir brauchen: (X, Y, Z) = (wall_value, Y, Z)
                    # Aktuell: mesh.points = (x_plot, y_plot, z_coords) = (Y, Z, X)
                    # Korrigiere zu: (X, Y, Z) = (X=wall_value, Y, Z)
                    points_corrected = np.column_stack([
                        mesh.points[:, 2],  # X (aus z_coords, war konstant)
                        mesh.points[:, 0],  # Y (aus x_plot)
                        mesh.points[:, 1],  # Z (aus y_plot)
                    ])
                    mesh.points = points_corrected
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Vertical] YZ-Wand: Koordinaten korrigiert")
                        print(f"  └─ X: min={points_corrected[:, 0].min():.2f}, max={points_corrected[:, 0].max():.2f} (sollte konstant: {wall_value:.2f})")
                        print(f"  └─ Y: min={points_corrected[:, 1].min():.2f}, max={points_corrected[:, 1].max():.2f}")
                        print(f"  └─ Z: min={points_corrected[:, 2].min():.2f}, max={points_corrected[:, 2].max():.2f}")
                elif vertical_orientation == "xz_slanted":
                    # Schräge X-Z-Wand: build_surface_mesh erstellt (X, Z, Y_interp)
                    # Wir brauchen: (X, Y, Z) = (X, Y_interp, Z)
                    # Aktuell: mesh.points = (x_plot, y_plot, z_coords) = (X, Z, Y_interp)
                    # Korrigiere zu: (X, Y, Z) = (X, Y_interp, Z)
                    points_corrected = np.column_stack([
                        mesh.points[:, 0],  # X (aus x_plot)
                        mesh.points[:, 2],  # Y (aus z_coords, war interpoliert)
                        mesh.points[:, 1],  # Z (aus y_plot)
                    ])
                    mesh.points = points_corrected
                    if DEBUG_PLOT3D_TIMING:
                        bounds_before = (mesh.points[:, 0].min(), mesh.points[:, 0].max(),
                                       mesh.points[:, 1].min(), mesh.points[:, 1].max(),
                                       mesh.points[:, 2].min(), mesh.points[:, 2].max())
                        print(f"[DEBUG Vertical] Schräge XZ-Wand: Koordinaten korrigiert")
                        print(f"  └─ X: [{points_corrected[:, 0].min():.2f}, {points_corrected[:, 0].max():.2f}] span={points_corrected[:, 0].max()-points_corrected[:, 0].min():.2f}")
                        print(f"  └─ Y: [{points_corrected[:, 1].min():.2f}, {points_corrected[:, 1].max():.2f}] span={points_corrected[:, 1].max()-points_corrected[:, 1].min():.2f} (interpoliert)")
                        print(f"  └─ Z: [{points_corrected[:, 2].min():.2f}, {points_corrected[:, 2].max():.2f}] span={points_corrected[:, 2].max()-points_corrected[:, 2].min():.2f}")
                        # Prüfe Konsistenz mit Z_plot
                        z_expected_min, z_expected_max = Z_plot.min(), Z_plot.max()
                        z_actual_min, z_actual_max = points_corrected[:, 2].min(), points_corrected[:, 2].max()
                        if abs(z_actual_min - z_expected_min) > 0.1 or abs(z_actual_max - z_expected_max) > 0.1:
                            print(f"  ⚠️  Z-Koordinaten-Abweichung: erwartet [{z_expected_min:.2f}, {z_expected_max:.2f}], tatsächlich [{z_actual_min:.2f}, {z_actual_max:.2f}]")
                elif vertical_orientation == "yz_slanted":
                    # Schräge Y-Z-Wand: build_surface_mesh erstellt (Y, Z, X_interp)
                    # Wir brauchen: (X, Y, Z) = (X_interp, Y, Z)
                    # Aktuell: mesh.points = (x_plot, y_plot, z_coords) = (Y, Z, X_interp)
                    # Korrigiere zu: (X, Y, Z) = (X_interp, Y, Z)
                    points_corrected = np.column_stack([
                        mesh.points[:, 2],  # X (aus z_coords, war interpoliert)
                        mesh.points[:, 0],  # Y (aus x_plot)
                        mesh.points[:, 1],  # Z (aus y_plot)
                    ])
                    mesh.points = points_corrected
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Vertical] Schräge YZ-Wand: Koordinaten korrigiert")
                        print(f"  └─ X: [{points_corrected[:, 0].min():.2f}, {points_corrected[:, 0].max():.2f}] span={points_corrected[:, 0].max()-points_corrected[:, 0].min():.2f} (interpoliert)")
                        print(f"  └─ Y: [{points_corrected[:, 1].min():.2f}, {points_corrected[:, 1].max():.2f}] span={points_corrected[:, 1].max()-points_corrected[:, 1].min():.2f}")
                        print(f"  └─ Z: [{points_corrected[:, 2].min():.2f}, {points_corrected[:, 2].max():.2f}] span={points_corrected[:, 2].max()-points_corrected[:, 2].min():.2f}")
                        # Prüfe Konsistenz mit Z_plot
                        z_expected_min, z_expected_max = Z_plot.min(), Z_plot.max()
                        z_actual_min, z_actual_max = points_corrected[:, 2].min(), points_corrected[:, 2].max()
                        if abs(z_actual_min - z_expected_min) > 0.1 or abs(z_actual_max - z_expected_max) > 0.1:
                            print(f"  ⚠️  Z-Koordinaten-Abweichung: erwartet [{z_expected_min:.2f}, {z_expected_max:.2f}], tatsächlich [{z_actual_min:.2f}, {z_actual_max:.2f}]")
                
                if mesh is None or mesh.n_points == 0:
                    continue
                
                actor_name = f"vertical_spl_{surface_id}"
                # Entferne ggf. alten Actor
                try:
                    if actor_name in self.plotter.renderer.actors:
                        self.plotter.remove_actor(actor_name)
                except Exception:
                    pass
                
                # Füge Mesh zum Plotter hinzu (identisch zu horizontalen Flächen)
                actor = self.plotter.add_mesh(
                    mesh,
                    name=actor_name,
                    scalars="plot_scalars",
                    cmap=cmap_object,
                    clim=(cbar_min, cbar_max),
                    smooth_shading=not is_step_mode,
                    show_scalar_bar=False,
                    reset_camera=False,
                    interpolate_before_map=not is_step_mode,
                )
                
                if actor and hasattr(actor, 'SetPickable'):
                    actor.SetPickable(True)
                
                new_vertical_meshes[actor_name] = actor
                
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Vertical] Surface '{surface_id}': Vertikales Mesh erstellt ({mesh.n_points} Punkte)")
                    print(f"  └─ Actor '{actor_name}' zum Plotter hinzugefügt")
                    
                    # 🎯 DEBUG: Zeige finale Bounds der geplotteten Fläche
                    if mesh.n_points > 0:
                        final_points = mesh.points
                        final_bounds = {
                            'X': (final_points[:, 0].min(), final_points[:, 0].max()),
                            'Y': (final_points[:, 1].min(), final_points[:, 1].max()),
                            'Z': (final_points[:, 2].min(), final_points[:, 2].max()),
                        }
                        print(f"  └─ Finale Bounds der geplotteten Fläche:")
                        print(f"     └─ X: [{final_bounds['X'][0]:.2f}, {final_bounds['X'][1]:.2f}] span={final_bounds['X'][1]-final_bounds['X'][0]:.2f}")
                        print(f"     └─ Y: [{final_bounds['Y'][0]:.2f}, {final_bounds['Y'][1]:.2f}] span={final_bounds['Y'][1]-final_bounds['Y'][0]:.2f}")
                        print(f"     └─ Z: [{final_bounds['Z'][0]:.2f}, {final_bounds['Z'][1]:.2f}] span={final_bounds['Z'][1]-final_bounds['Z'][0]:.2f}")
                        
                        # Prüfe gegen Polygon-Bounds
                        if surface_id and hasattr(self, 'settings'):
                            surface_definitions = getattr(self.settings, 'surface_definitions', {})
                            if surface_id in surface_definitions:
                                surface_def = surface_definitions[surface_id]
                                if isinstance(surface_def, SurfaceDefinition):
                                    polygon_points = getattr(surface_def, 'points', []) or []
                                else:
                                    polygon_points = surface_def.get('points', [])
                                
                                if len(polygon_points) > 0:
                                    poly_x = np.array([p.get('x', 0.0) for p in polygon_points], dtype=float)
                                    poly_y = np.array([p.get('y', 0.0) for p in polygon_points], dtype=float)
                                    poly_z = np.array([p.get('z', 0.0) for p in polygon_points], dtype=float)
                                    
                                    poly_bounds = {
                                        'X': (poly_x.min(), poly_x.max()),
                                        'Y': (poly_y.min(), poly_y.max()),
                                        'Z': (poly_z.min(), poly_z.max()),
                                    }
                                    
                                    # Prüfe ob geplottete Fläche außerhalb Polygon liegt
                                    # 🎯 Toleranz für Grid-Erweiterung: Kleine Abweichungen sind durch Grid-Erweiterung erwartet
                                    tolerance = 0.5  # 0.5 m Toleranz für Grid-Erweiterung
                                    outside_x = (final_bounds['X'][0] < poly_bounds['X'][0] - tolerance) or (final_bounds['X'][1] > poly_bounds['X'][1] + tolerance)
                                    outside_y = (final_bounds['Y'][0] < poly_bounds['Y'][0] - tolerance) or (final_bounds['Y'][1] > poly_bounds['Y'][1] + tolerance)
                                    outside_z = (final_bounds['Z'][0] < poly_bounds['Z'][0] - tolerance) or (final_bounds['Z'][1] > poly_bounds['Z'][1] + tolerance)
                                    
                                    # Berechne wie viele Punkte außerhalb liegen
                                    points_outside_x = np.sum((final_points[:, 0] < poly_bounds['X'][0]) | (final_points[:, 0] > poly_bounds['X'][1]))
                                    points_outside_y = np.sum((final_points[:, 1] < poly_bounds['Y'][0]) | (final_points[:, 1] > poly_bounds['Y'][1]))
                                    points_outside_z = np.sum((final_points[:, 2] < poly_bounds['Z'][0]) | (final_points[:, 2] > poly_bounds['Z'][1]))
                                    total_outside = points_outside_x + points_outside_y + points_outside_z
                                    outside_percentage = 100.0 * total_outside / mesh.n_points if mesh.n_points > 0 else 0.0
                                    
                                    # Warnung nur wenn signifikant außerhalb (> 5% der Punkte oder > Toleranz)
                                    if (outside_x or outside_y or outside_z) and (outside_percentage > 5.0):
                                        print(f"  ⚠️  GEPLOTTETE FLÄCHE LIEGT AUSSERHALB POLYGON-BOUNDS!")
                                        print(f"     └─ Polygon X: [{poly_bounds['X'][0]:.2f}, {poly_bounds['X'][1]:.2f}], Geplottet X: [{final_bounds['X'][0]:.2f}, {final_bounds['X'][1]:.2f}]")
                                        print(f"     └─ Polygon Y: [{poly_bounds['Y'][0]:.2f}, {poly_bounds['Y'][1]:.2f}], Geplottet Y: [{final_bounds['Y'][0]:.2f}, {final_bounds['Y'][1]:.2f}]")
                                        print(f"     └─ Polygon Z: [{poly_bounds['Z'][0]:.2f}, {poly_bounds['Z'][1]:.2f}], Geplottet Z: [{final_bounds['Z'][0]:.2f}, {final_bounds['Z'][1]:.2f}]")
                                        print(f"     └─ Punkte außerhalb X-Bounds: {points_outside_x}/{mesh.n_points} ({100*points_outside_x/mesh.n_points:.1f}%)")
                                        print(f"     └─ Punkte außerhalb Y-Bounds: {points_outside_y}/{mesh.n_points} ({100*points_outside_y/mesh.n_points:.1f}%)")
                                        print(f"     └─ Punkte außerhalb Z-Bounds: {points_outside_z}/{mesh.n_points} ({100*points_outside_z/mesh.n_points:.1f}%)")
                                    elif DEBUG_PLOT3D_TIMING and total_outside > 0:
                                        # Nur Debug-Info für kleine Abweichungen (< 5%)
                                        print(f"  [DEBUG] Kleine Grid-Erweiterung erkannt: {total_outside}/{mesh.n_points} Punkte außerhalb Polygon ({outside_percentage:.1f}%)")
                    
            except Exception as e:
                import traceback
                if DEBUG_PLOT3D_TIMING:
                    print(f"[PlotSPL3D] Error processing vertical surface {surface_id}: {e}")
                    print(f"[PlotSPL3D] Traceback:")
                    traceback.print_exc()
                continue
        
        # 🎯 DEBUG: Zeige alle hinzugefügten vertikalen Meshes
        if DEBUG_PLOT3D_TIMING:
            print(f"[DEBUG Vertical Grids] ✅ Vertikale Surfaces verarbeitet: {len(new_vertical_meshes)} Meshes hinzugefügt")
            for actor_name in new_vertical_meshes.keys():
                print(f"  └─ {actor_name}")
        
        # Alte Actors entfernen
        if hasattr(self, "_vertical_surface_meshes"):
            for old_name in list(self._vertical_surface_meshes.keys()):
                if old_name not in new_vertical_meshes:
                    try:
                        if old_name in self.plotter.renderer.actors:
                            self.plotter.remove_actor(old_name)
                    except Exception:
                        pass
        
        self._vertical_surface_meshes = new_vertical_meshes

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
                        print(f"  ❌ {surface_id}: Nicht in surface_grids_data")
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
                    
                    # Ausgabe
                    print(f"\n  {status} {surface_id}: {status_text}")
                    print(f"     └─ Orientation: {orientation}, Dominant Axis: {dominant_axis}")
                    print(f"     └─ Grid Shape: {X_grid.shape if grid_valid else 'INVALID'}")
                    print(f"     └─ Active Points: {active_points}/{X_grid.size if grid_valid else 0}")
                    print(f"     └─ Enabled: {is_enabled}, Has Actor: {has_actor}, Has Results: {has_results}")
                    
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