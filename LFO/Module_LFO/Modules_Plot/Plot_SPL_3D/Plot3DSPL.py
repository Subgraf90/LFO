from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Iterable, Optional

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
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

DEBUG_PLOT3D_TIMING = bool(int(os.environ.get("LFO_DEBUG_PLOT3D_TIMING", "1")))


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
            # Fallback: 2D-Interpolation
            if method == "nearest":
                return SPL3DPlotRenderer._nearest_interpolate_grid(source_x, source_y, values, xq, yq)
            else:
                return SPL3DPlotRenderer._bilinear_interpolate_grid(source_x, source_y, values, xq, yq)
        
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
        
        # Fallback für NaN-Werte: 2D-Interpolation
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            if method == "nearest":
                fallback = SPL3DPlotRenderer._nearest_interpolate_grid(
                    source_x, source_y, values, xq[nan_mask], yq[nan_mask]
                )
            else:
                fallback = SPL3DPlotRenderer._bilinear_interpolate_grid(
                    source_x, source_y, values, xq[nan_mask], yq[nan_mask]
                )
            result[nan_mask] = fallback
        
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

        # Quelle: Berechnungsraster
        source_x = np.asarray(getattr(geometry, "source_x", []), dtype=float)
        source_y = np.asarray(getattr(geometry, "source_y", []), dtype=float)
        values = np.asarray(original_plot_values, dtype=float)
        if values.shape != (len(source_y), len(source_x)):
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
        norm = Normalize(vmin=cbar_min, vmax=cbar_max)

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

        if not enabled_surfaces:
            return

        # Versuche, aus der Plot-Geometrie einen effektiven Upscaling-Faktor zu rekonstruieren.
        # Hintergrund: Für achsparallele Rechtecke können wir die Textur grober machen,
        #              wenn das Plot-Raster bereits hochskaliert ist.
        effective_upscale_factor: int = 4
        try:
            plot_x = np.asarray(getattr(geometry, "plot_x", []), dtype=float)
            plot_y = np.asarray(getattr(geometry, "plot_y", []), dtype=float)
            if plot_x.size > 1 and plot_y.size > 1 and source_x.size > 1 and source_y.size > 1:
                # Nutze das Verhältnis der Stützstellen als Approximation
                ratio_x = (float(plot_x.size) - 1.0) / (float(source_x.size) - 1.0)
                ratio_y = (float(plot_y.size) - 1.0) / (float(source_y.size) - 1.0)
                approx = max(ratio_x, ratio_y)
                if approx > 1.2:
                    # Runde auf ganzzahligen Faktor und begrenze auf sinnvollen Bereich
                    effective_upscale_factor = int(round(approx))
                    if effective_upscale_factor < 1:
                        effective_upscale_factor = 1
                    if effective_upscale_factor > 8:
                        effective_upscale_factor = 8
        except Exception:
            effective_upscale_factor = 1

        if DEBUG_PLOT3D_TIMING:
            print(
                "[PlotSPL3D] _render_surfaces_textured config: "
                f"base_tex_res={base_tex_res:.4f} m, "
                f"tex_res_global={tex_res_global:.4f} m, "
                f"effective_upscale_factor={effective_upscale_factor}"
            )

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
            
            # Berechne Signatur mit korrekter tex_res_surface
            texture_signature = self._calculate_texture_signature(
                surface_id=surface_id,
                points=points,
                source_x=source_x,
                source_y=source_y,
                values=values,
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
        
        # Sequenzielle Verarbeitung
        for surface_id, points, surface_obj in surfaces_to_process:
            try:
                # Verarbeite Surface
                result = self._process_single_surface_texture(
                    surface_id=surface_id,
                    points=points,
                    surface_obj=surface_obj,
                    source_x=source_x,
                    source_y=source_y,
                    values=values,
                    cbar_min=cbar_min,
                    cbar_max=cbar_max,
                    base_cmap=base_cmap,
                    norm=norm,
                    is_step_mode=is_step_mode,
                    colorization_mode=colorization_mode,
                    cbar_step=cbar_step,
                    tex_res_global=tex_res_global,
                    effective_upscale_factor=effective_upscale_factor,
                )
                
                if result is None:
                    continue
                
                grid = result['grid']
                tex = result['texture']
                metadata = result['metadata']
                texture_signature = result['texture_signature']
                
                # Alten Actor entfernen
                old_texture_data = self._surface_texture_actors.get(surface_id)
                if old_texture_data is not None:
                    old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                    if old_actor is not None:
                        try:
                            self.plotter.remove_actor(old_actor)
                        except Exception:
                            pass
                
                # Neuen Actor hinzufügen
                actor_name = f"{self.SURFACE_NAME}_tex_{surface_id}"
                actor = self.plotter.add_mesh(
                    grid,
                    name=actor_name,
                    texture=tex,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(False)
                
                # Metadaten aktualisieren
                metadata['actor'] = actor
                self._surface_texture_actors[surface_id] = metadata
                self._surface_texture_cache[surface_id] = texture_signature
                
                # Setze auch auf self.plotter
                if not hasattr(self.plotter, '_surface_texture_actors'):
                    self.plotter._surface_texture_actors = {}
                self.plotter._surface_texture_actors[surface_id] = {
                    'actor': actor,
                    'surface_id': surface_id,
                }
                
                surfaces_processed += 1
            except Exception as e:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[PlotSPL3D] Error processing surface {surface_id}: {e}")
        
        # Timing-Ausgabe
        if DEBUG_PLOT3D_TIMING:
            t_textures_end = time.perf_counter()
            print(
                f"[PlotSPL3D] _render_surfaces_textured TOTAL: "
                f"{(t_textures_end - t_textures_start) * 1000.0:7.2f} ms, "
                f"surfaces_processed={surfaces_processed}, "
                f"surfaces_cached={len(cached_surfaces)}"
            )

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

        # Wenn keine gültigen SPL-Daten vorliegen, belassen wir die bestehende Szene
        # (inkl. Lautsprechern) unverändert und brechen nur das SPL-Update ab.
        if not self._has_valid_data(sound_field_x, sound_field_y, sound_field_pressure):
            return

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
                    return
            else:
                return

        if pressure.shape != (len(y), len(x)):
            return

        # Aktualisiere ColorbarManager Modes
        self.colorbar_manager.update_modes(phase_mode_active=phase_mode, time_mode_active=time_mode)
        self.colorbar_manager.set_override(None)

        if time_mode:
            pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                return
            # Verwende feste Colorbar-Parameter (keine dynamische Skalierung)
            plot_values = pressure
        elif phase_mode:
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
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
        # Vertikale Flächen: separate SPL-Flächen rendern
        # ------------------------------------------------------------
        # Merke den aktuell verwendeten Colorization-Mode, damit
        # _update_vertical_spl_surfaces identisch reagieren kann.
        self._last_colorization_mode = colorization_mode_used
        self._update_vertical_spl_surfaces()

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