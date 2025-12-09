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
from Module_LFO.Modules_Init.Logging import PERF_ENABLED, measure_time, perf_section
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DHelpers import (
    has_valid_data,
    compute_surface_signature,
    quantize_to_steps,
)
from Module_LFO.Modules_Plot.Plot_SPL_3D.ColorbarManager import PHASE_CMAP

DEBUG_PLOT3D_TIMING = bool(int(os.environ.get("LFO_DEBUG_PLOT3D_TIMING", "1")))

# 🎯 Upscaling-Faktor für Grid-Auflösung (nur fürs Plot-Overlay, nicht für Calc)
# 1 = kein Upscaling. 2 ist ein guter Kompromiss: glatte Kanten bei moderater Last.
PLOT_UPSCALE_FACTOR = int(os.environ.get("LFO_PLOT_UPSCALE_FACTOR", "2"))
if PLOT_UPSCALE_FACTOR < 1:
    PLOT_UPSCALE_FACTOR = 1


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

    # ⚠️ AUSKOMMENTIERT: Nicht mehr benötigt - Alte Textur-Rendering-Methode
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
        ⚠️ AUSKOMMENTIERT: Nicht mehr benötigt - Wird durch direkten Grid-Plot ersetzt
        """
        return

    def _integrate_edge_points_into_mesh(
        self,
        main_mesh: Any,
        edge_x: np.ndarray,
        edge_y: np.ndarray,
        edge_z: np.ndarray,
        edge_spl: np.ndarray,
        time_mode: bool,
        phase_mode: bool,
        cbar_min: float,
        cbar_max: float,
        pv_module: Any,
        surface_id: Optional[str] = None,  # 🎯 NEU: Für Polygon-Prüfung
        settings: Optional[Any] = None,  # 🎯 NEU: Für Surface-Definition
        vertical_orientation: Optional[str] = None,  # 🎯 NEU: "xz" oder "yz" für vertikale Flächen
        wall_value: Optional[float] = None,  # 🎯 NEU: Konstanter Wert für konstante Achse
    ) -> Optional[Any]:
        """
        Integriert Randpunkte direkt in das Grid-Mesh.
        
        Die Randpunkte sind erweiterte Datenpunkte innerhalb der Surface (nicht auf der Linie),
        die mit dem Haupt-Grid kombiniert werden, um ein einheitliches Mesh zu erstellen.
        
        Args:
            main_mesh: Haupt-Grid-Mesh (PyVista PolyData)
            edge_x, edge_y, edge_z: Koordinaten der Randpunkte
            edge_spl: SPL-Werte der Randpunkte (komplex)
            time_mode: Zeit-Modus aktiv
            phase_mode: Phase-Modus aktiv
            cbar_min, cbar_max: Colorbar-Grenzen
            pv_module: PyVista-Modul
        
        Returns:
            Kombiniertes Mesh (Grid + Randpunkte) oder None
        """
        try:
            if edge_x is None or len(edge_x) == 0:
                return main_mesh
            
            # Konvertiere Randpunkte zu numpy Arrays
            edge_x = np.array(edge_x, dtype=float)
            edge_y = np.array(edge_y, dtype=float)
            edge_z = np.array(edge_z, dtype=float)
            edge_spl_complex = np.array(edge_spl, dtype=complex)
            
            # Konvertiere SPL-Werte zu dB (wie im Haupt-Grid)
            if time_mode:
                edge_spl_values = np.real(edge_spl_complex)
                edge_spl_values = np.nan_to_num(edge_spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                edge_spl_values = np.clip(edge_spl_values, cbar_min, cbar_max)
            elif phase_mode:
                edge_spl_values = np.angle(edge_spl_complex)
                edge_spl_values = np.nan_to_num(edge_spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                edge_spl_values = np.clip(edge_spl_values, cbar_min, cbar_max)
            else:
                # SPL-Modus: Betrag zu dB
                pressure_magnitude = np.abs(edge_spl_complex)
                pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                edge_spl_values = self.functions.mag2db(pressure_magnitude)
                edge_spl_values = np.nan_to_num(edge_spl_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 🎯 WICHTIG: Randpunkte werden bereits entlang der 3D-Kanten generiert und liegen daher
            # bereits exakt auf der Flächenebene. KEINE Projektion nötig!
            # Die Projektion würde Randpunkte außerhalb des Grid-Bereichs nach innen verschieben,
            # was zu unsauberen Rändern führt.
            edge_points_3d = np.column_stack([edge_x, edge_y, edge_z])
            
            is_slanted = vertical_orientation in ("xz_slanted", "yz_slanted")
            if DEBUG_PLOT3D_TIMING and is_slanted:
                print(f"[DEBUG Plot Mesh] [{vertical_orientation}] Randpunkte direkt verwendet (bereits auf 3D-Kanten)")
            
            # Hole Punkte und Scalar-Werte aus Haupt-Mesh
            main_points = main_mesh.points
            main_scalars = main_mesh["plot_scalars"]
            
            # Kombiniere alle Punkte (Grid + Randpunkte)
            combined_points = np.vstack([main_points, edge_points_3d])
            combined_scalars = np.concatenate([main_scalars, edge_spl_values])
            
            is_slanted = vertical_orientation in ("xz_slanted", "yz_slanted")
            if DEBUG_PLOT3D_TIMING and is_slanted:
                print(f"[DEBUG Plot Mesh] [{vertical_orientation}] Kombiniere: {len(main_points)} Grid-Punkte + {len(edge_points_3d)} Randpunkte = {len(combined_points)} total")
            
            # Erstelle neues Mesh mit allen kombinierten Punkten
            # 🚫 Re-Triangulation deaktiviert: vorhandene Dreiecke/Flächen nutzen
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Plot Mesh] [{vertical_orientation}] Re-Triangulation übersprungen – verwende bestehendes Mesh (Validator/Import)")
            return main_mesh
            
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Fehler beim Integrieren von Randpunkten: {e}")
            return main_mesh
    
    def _filter_triangles_outside_surface(
        self,
        mesh: Any,
        surface_id: str,
        settings: Any,
        pv_module: Any,
        vertical_orientation: Optional[str] = None,  # 🎯 NEU: "xz" oder "yz" für vertikale Flächen
    ) -> Any:
        """
        Filtert Dreiecke außerhalb der Surface nach Delaunay-Triangulation.
        
        Prüft für jedes Dreieck, ob sein Schwerpunkt innerhalb des Surface-Polygons liegt.
        Nur Dreiecke innerhalb der Surface werden behalten.
        
        Args:
            mesh: PyVista PolyData-Mesh nach Delaunay-Triangulation
            surface_id: ID der Surface
            settings: Settings-Objekt mit Surface-Definitionen
            pv_module: PyVista-Modul
        
        Returns:
            Gefiltertes Mesh (nur Dreiecke innerhalb der Surface)
        """
        try:
            # Hole Surface-Definition
            surface_definitions = getattr(settings, 'surface_definitions', {})
            if surface_id not in surface_definitions:
                return mesh
            
            surface_def = surface_definitions[surface_id]
            if isinstance(surface_def, SurfaceDefinition):
                polygon_points = getattr(surface_def, 'points', []) or []
            else:
                polygon_points = surface_def.get('points', [])
            
            if len(polygon_points) < 3:
                return mesh
            
            # 🎯 STRENGE FILTERUNG: Prüfe ALLE Eckpunkte jeder Zelle, nicht nur den Schwerpunkt
            # Eine Zelle wird nur behalten, wenn ALLE ihre Eckpunkte innerhalb des Polygons liegen
            # Das verhindert Artefakte an den Rändern, wo Dreiecke teilweise außerhalb liegen
            n_cells = mesh.n_cells
            if n_cells == 0:
                return mesh
            
            # 🎯 PROJEKTION: Wähle richtige Projektionsebene für Polygon-Prüfung
            from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import _points_in_polygon_batch_plot
            
            # Bereite Polygon vor
            if vertical_orientation in ("xz", "xz_slanted"):
                # X-Z-Wand (senkrecht oder schräg): Prüfe in XZ-Ebene
                px = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
                pz = np.array([float(p.get("z", 0.0)) for p in polygon_points], dtype=float)
                poly_points_dict = [{"x": x, "y": z} for x, z in zip(px, pz)]  # x=u, y=v
            elif vertical_orientation in ("yz", "yz_slanted"):
                # Y-Z-Wand (senkrecht oder schräg): Prüfe in YZ-Ebene
                py = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)
                pz = np.array([float(p.get("z", 0.0)) for p in polygon_points], dtype=float)
                poly_points_dict = [{"x": y, "y": z} for y, z in zip(py, pz)]  # x=u, y=v
            else:
                # Horizontale Fläche: Prüfe in XY-Ebene (Standard)
                px = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
                py = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)
                poly_points_dict = [{"x": x, "y": y} for x, y in zip(px, py)]
            
            # Hole alle Punkte des Meshes
            points_3d = mesh.points  # Shape: (n_points, 3)
            
            # Projiziere Punkte in die richtige Ebene
            if vertical_orientation in ("xz", "xz_slanted"):
                points_u = points_3d[:, 0]  # X
                points_v = points_3d[:, 2]  # Z
            elif vertical_orientation in ("yz", "yz_slanted"):
                points_u = points_3d[:, 1]  # Y
                points_v = points_3d[:, 2]  # Z
            else:
                points_u = points_3d[:, 0]  # X
                points_v = points_3d[:, 1]  # Y
            
            # Prüfe alle Punkte gleichzeitig
            points_inside = _points_in_polygon_batch_plot(
                points_u.reshape(-1, 1),  # U-Koordinaten als 2D-Array
                points_v.reshape(-1, 1),  # V-Koordinaten als 2D-Array
                poly_points_dict
            )
            
            if points_inside is None:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Filter] ⚠️ points_inside ist None für '{surface_id}'")
                return mesh
            
            points_inside_1d = points_inside.flatten()
            
            # 🎯 STRENGE FILTERUNG: Entferne Zellen, wenn:
            # 1. NICHT ALLE Eckpunkte innerhalb sind ODER
            # 2. Der Schwerpunkt außerhalb ist (für zusätzliche Sicherheit)
            cells_to_keep = []
            
            # Berechne Schwerpunkte für alle Zellen
            try:
                cell_centers = mesh.cell_centers().points
                centroids_3d = cell_centers
                if vertical_orientation in ("xz", "xz_slanted"):
                    centroids_u = centroids_3d[:, 0]
                    centroids_v = centroids_3d[:, 2]
                elif vertical_orientation in ("yz", "yz_slanted"):
                    centroids_u = centroids_3d[:, 1]
                    centroids_v = centroids_3d[:, 2]
                else:
                    centroids_u = centroids_3d[:, 0]
                    centroids_v = centroids_3d[:, 1]
                
                # Prüfe Schwerpunkte
                centroids_inside = _points_in_polygon_batch_plot(
                    centroids_u.reshape(-1, 1),
                    centroids_v.reshape(-1, 1),
                    poly_points_dict
                )
                centroids_inside_1d = centroids_inside.flatten() if centroids_inside is not None else np.ones(n_cells, dtype=bool)
            except Exception:
                centroids_inside_1d = np.ones(n_cells, dtype=bool)
                centroids_u = np.array([])
                centroids_v = np.array([])
            
            # 🎯 STRENGE FILTERUNG: Für schräge/überhängende Flächen zusätzliche Prüfung
            is_slanted_strict = vertical_orientation in ("xz_slanted", "yz_slanted")
            
            for i in range(n_cells):
                try:
                    cell = mesh.get_cell(i)
                    cell_point_ids = cell.point_ids
                    
                    # Prüfe ob ALLE Punkte der Zelle innerhalb sind
                    all_points_inside = np.all(points_inside_1d[cell_point_ids])
                    any_point_inside = np.any(points_inside_1d[cell_point_ids])
                    
                    # Prüfe ob Schwerpunkt innerhalb ist
                    centroid_inside = centroids_inside_1d[i] if i < len(centroids_inside_1d) else True
                    
                    # 🎯 VEREINFACHTE FILTERUNG: Da wir bereits vor der Triangulation filtern,
                    # sollten alle Punkte innerhalb des Polygons sein. Delaunay kann jedoch
                    # neue Punkte auf der konvexen Hülle erzeugen, die außerhalb des Polygons liegen.
                    # Daher prüfen wir: Behalte Zelle nur wenn ALLE Eckpunkte innerhalb sind.
                    # Dies sorgt für saubere Ränder ohne Zacken.
                    if all_points_inside:
                        cells_to_keep.append(i)
                    elif DEBUG_PLOT3D_TIMING and is_slanted_strict:
                        # Debug: Zeige warum Zelle entfernt wurde
                        n_outside = np.sum(~points_inside_1d[cell_point_ids])
                        print(f"[DEBUG Filter] Zelle {i} entfernt: {n_outside}/{len(cell_point_ids)} Punkte außerhalb Polygon")
                except Exception:
                    # Bei Fehler: Zelle überspringen
                    continue
            
            # Für Debug: Berechne auch Schwerpunkte (nur für Ausgabe)
            try:
                cell_centers = mesh.cell_centers().points
                centroids_3d = cell_centers
                if vertical_orientation in ("xz", "xz_slanted"):
                    centroids_u = centroids_3d[:, 0]
                    centroids_v = centroids_3d[:, 2]
                elif vertical_orientation in ("yz", "yz_slanted"):
                    centroids_u = centroids_3d[:, 1]
                    centroids_v = centroids_3d[:, 2]
                else:
                    centroids_u = centroids_3d[:, 0]
                    centroids_v = centroids_3d[:, 1]
            except Exception:
                centroids_u = np.array([])
                centroids_v = np.array([])
            
            inside_mask_1d = np.array([i in cells_to_keep for i in range(n_cells)], dtype=bool)
            
            # 🎯 DEBUG: Zeige Details zur Filterung (nur für schräge/überhängende Flächen)
            is_slanted = vertical_orientation in ("xz_slanted", "yz_slanted")
            if DEBUG_PLOT3D_TIMING and is_slanted:
                n_inside = int(np.sum(inside_mask_1d))
                n_valid = n_cells  # Alle Zellen sind gültig (Fehlerbehandlung oben)
                n_invalid = 0
                print(f"[DEBUG Filter] Surface '{surface_id}' ({vertical_orientation}):")
                print(f"  └─ Total Zellen: {n_cells}")
                print(f"  └─ Gültige Zellen (kein NaN): {n_valid}/{n_cells} (ungültig: {n_invalid})")
                print(f"  └─ Zellen innerhalb Polygon: {n_inside}/{n_cells} ({100*n_inside/n_cells:.1f}%)")
                if len(poly_points_dict) > 0:
                    poly_u = np.array([p.get('x', 0.0) for p in poly_points_dict], dtype=float)
                    poly_v = np.array([p.get('y', 0.0) for p in poly_points_dict], dtype=float)
                    print(f"  └─ Polygon: {len(poly_points_dict)} Punkte")
                    print(f"  └─ Polygon U-Range: [{poly_u.min():.3f}, {poly_u.max():.3f}]")
                    print(f"  └─ Polygon V-Range: [{poly_v.min():.3f}, {poly_v.max():.3f}]")
                    if len(centroids_u) > 0:
                        print(f"  └─ Centroids U-Range: [{centroids_u.min():.3f}, {centroids_u.max():.3f}]")
                        print(f"  └─ Centroids V-Range: [{centroids_v.min():.3f}, {centroids_v.max():.3f}]")
                    
                    # 🎯 DEBUG: Analysiere entfernte und behaltene Zellen
                    removed_cells = n_cells - n_inside
                    if removed_cells > 0:
                        # Finde Zellen, die entfernt wurden
                        removed_mask = ~inside_mask_1d
                        kept_mask = inside_mask_1d
                        
                        removed_centroids_u = centroids_u[removed_mask] if len(centroids_u) > 0 else np.array([])
                        removed_centroids_v = centroids_v[removed_mask] if len(centroids_v) > 0 else np.array([])
                        kept_centroids_u = centroids_u[kept_mask] if len(centroids_u) > 0 else np.array([])
                        kept_centroids_v = centroids_v[kept_mask] if len(centroids_v) > 0 else np.array([])
                        
                        if len(removed_centroids_u) > 0:
                            print(f"  └─ Entfernte Zellen: {removed_cells}")
                            print(f"     └─ Entfernte Centroids U-Range: [{removed_centroids_u.min():.3f}, {removed_centroids_u.max():.3f}]")
                            print(f"     └─ Entfernte Centroids V-Range: [{removed_centroids_v.min():.3f}, {removed_centroids_v.max():.3f}]")
                            
                    if len(kept_centroids_u) > 0:
                        print(f"  └─ Behaltene Zellen: {n_inside}")
                        print(f"     └─ Behaltene Centroids U-Range: [{kept_centroids_u.min():.3f}, {kept_centroids_u.max():.3f}]")
                        print(f"     └─ Behaltene Centroids V-Range: [{kept_centroids_v.min():.3f}, {kept_centroids_v.max():.3f}]")
                        
                        # 🎯 PRÜFE AUF PLANE FLÄCHEN: Analysiere Y-Variation (für xz_slanted) oder X-Variation (für yz_slanted)
                        if vertical_orientation == "xz_slanted":
                            # Für xz_slanted: Y sollte variieren (schräge Fläche)
                            # Hole Y-Koordinaten der behaltenen Zellen
                            kept_cell_indices = np.where(kept_mask)[0]
                            kept_cells_y = []
                            kept_cells_y_span = []
                            for idx in kept_cell_indices[:min(100, len(kept_cell_indices))]:  # Prüfe max 100 Zellen
                                try:
                                    cell = mesh.get_cell(idx)
                                    cell_points_3d = points_3d[cell.point_ids]
                                    y_coords = cell_points_3d[:, 1]  # Y-Koordinaten
                                    y_span = y_coords.max() - y_coords.min()
                                    kept_cells_y.append(y_coords.mean())
                                    kept_cells_y_span.append(y_span)
                                except:
                                    continue
                            
                            if len(kept_cells_y) > 0:
                                kept_cells_y = np.array(kept_cells_y)
                                kept_cells_y_span = np.array(kept_cells_y_span)
                                print(f"     └─ Y-Koordinaten der behaltenen Zellen: min={kept_cells_y.min():.3f}, max={kept_cells_y.max():.3f}, span={kept_cells_y.max()-kept_cells_y.min():.3f}")
                                print(f"     └─ Y-Span pro Zelle: min={kept_cells_y_span.min():.6f}, max={kept_cells_y_span.max():.6f}, mean={kept_cells_y_span.mean():.6f}")
                                
                                # Prüfe ob es Zellen mit sehr kleiner Y-Span gibt (plan verlaufend)
                                flat_threshold = 0.01  # 1cm Toleranz
                                flat_cells = np.sum(kept_cells_y_span < flat_threshold)
                                if flat_cells > 0:
                                    print(f"     ⚠️  PLANE FLÄCHEN ERKANNT: {flat_cells}/{len(kept_cells_y_span)} Zellen haben Y-Span < {flat_threshold}m (plan verlaufend)")
                                    flat_indices = np.where(kept_cells_y_span < flat_threshold)[0]
                                    flat_y_values = kept_cells_y[flat_indices]
                                    print(f"        └─ Plane Zellen Y-Bereich: [{flat_y_values.min():.3f}, {flat_y_values.max():.3f}]")
                                    
                                    # Prüfe ob diese Zellen innerhalb des Polygons liegen
                                    # (könnten durch Delaunay Convex Hull entstanden sein)
                                    if len(flat_y_values) > 0:
                                        # Prüfe ob die Y-Werte der planen Zellen außerhalb des erwarteten Bereichs liegen
                                        # Für xz_slanted sollte Y zwischen den Polygon-Punkten interpoliert werden
                                        # Hole ursprüngliche Y-Koordinaten der Polygon-Punkte
                                        poly_y_orig = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)
                                        if len(poly_y_orig) > 0:
                                            poly_y_min, poly_y_max = poly_y_orig.min(), poly_y_orig.max()
                                            flat_outside = np.sum((flat_y_values < poly_y_min - 0.1) | (flat_y_values > poly_y_max + 0.1))
                                            if flat_outside > 0:
                                                print(f"        ⚠️  {flat_outside} plane Zellen haben Y außerhalb Polygon Y-Bereich [{poly_y_min:.3f}, {poly_y_max:.3f}]")
                        
                        elif vertical_orientation == "yz_slanted":
                            # Für yz_slanted: X sollte variieren (schräge Fläche)
                            kept_cell_indices = np.where(kept_mask)[0]
                            kept_cells_x = []
                            kept_cells_x_span = []
                            for idx in kept_cell_indices[:min(100, len(kept_cell_indices))]:
                                try:
                                    cell = mesh.get_cell(idx)
                                    cell_points_3d = points_3d[cell.point_ids]
                                    x_coords = cell_points_3d[:, 0]  # X-Koordinaten
                                    x_span = x_coords.max() - x_coords.min()
                                    kept_cells_x.append(x_coords.mean())
                                    kept_cells_x_span.append(x_span)
                                except:
                                    continue
                            
                            if len(kept_cells_x) > 0:
                                kept_cells_x = np.array(kept_cells_x)
                                kept_cells_x_span = np.array(kept_cells_x_span)
                                print(f"     └─ X-Koordinaten der behaltenen Zellen: min={kept_cells_x.min():.3f}, max={kept_cells_x.max():.3f}, span={kept_cells_x.max()-kept_cells_x.min():.3f}")
                                print(f"     └─ X-Span pro Zelle: min={kept_cells_x_span.min():.6f}, max={kept_cells_x_span.max():.6f}, mean={kept_cells_x_span.mean():.6f}")
                                
                                flat_threshold = 0.01
                                flat_cells = np.sum(kept_cells_x_span < flat_threshold)
                                if flat_cells > 0:
                                    print(f"     ⚠️  PLANE FLÄCHEN ERKANNT: {flat_cells}/{len(kept_cells_x_span)} Zellen haben X-Span < {flat_threshold}m (plan verlaufend)")
                                    flat_indices = np.where(kept_cells_x_span < flat_threshold)[0]
                                    flat_x_values = kept_cells_x[flat_indices]
                                    print(f"        └─ Plane Zellen X-Bereich: [{flat_x_values.min():.3f}, {flat_x_values.max():.3f}]")
                                    
                                    poly_x_orig = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
                                    if len(poly_x_orig) > 0:
                                        poly_x_min, poly_x_max = poly_x_orig.min(), poly_x_orig.max()
                                        flat_outside = np.sum((flat_x_values < poly_x_min - 0.1) | (flat_x_values > poly_x_max + 0.1))
                                        if flat_outside > 0:
                                            print(f"        ⚠️  {flat_outside} plane Zellen haben X außerhalb Polygon X-Bereich [{poly_x_min:.3f}, {poly_x_max:.3f}]")
                            
                            # Prüfe ob entfernte Zellen außerhalb Polygon-Bounds liegen
                            u_outside = (removed_centroids_u.min() < poly_u.min()) or (removed_centroids_u.max() > poly_u.max())
                            v_outside = (removed_centroids_v.min() < poly_v.min()) or (removed_centroids_v.max() > poly_v.max())
                            
                            # Prüfe ob behaltene Zellen außerhalb Polygon-Bounds liegen (FEHLER!)
                            if len(kept_centroids_u) > 0:
                                kept_u_outside = (kept_centroids_u.min() < poly_u.min()) or (kept_centroids_u.max() > poly_u.max())
                                kept_v_outside = (kept_centroids_v.min() < poly_v.min()) or (kept_centroids_v.max() > poly_v.max())
                                
                                if kept_u_outside or kept_v_outside:
                                    print(f"     ⚠️  BEHALTENE ZELLEN LIEGEN AUSSERHALB POLYGON-BOUNDS! (FEHLER!)")
                                    if kept_u_outside:
                                        print(f"        └─ Behaltene U: [{kept_centroids_u.min():.3f}, {kept_centroids_u.max():.3f}], Polygon U: [{poly_u.min():.3f}, {poly_u.max():.3f}]")
                                    if kept_v_outside:
                                        print(f"        └─ Behaltene V: [{kept_centroids_v.min():.3f}, {kept_centroids_v.max():.3f}], Polygon V: [{poly_v.min():.3f}, {poly_v.max():.3f}]")
                                    
                                    # Zähle wie viele behaltene Centroids außerhalb liegen
                                    kept_outside_u = np.sum((kept_centroids_u < poly_u.min()) | (kept_centroids_u > poly_u.max()))
                                    kept_outside_v = np.sum((kept_centroids_v < poly_v.min()) | (kept_centroids_v > poly_v.max()))
                                    print(f"        └─ Behaltene Centroids außerhalb U-Bounds: {kept_outside_u}/{len(kept_centroids_u)}")
                                    print(f"        └─ Behaltene Centroids außerhalb V-Bounds: {kept_outside_v}/{len(kept_centroids_v)}")
                            
                            if u_outside or v_outside:
                                print(f"     ✅ Entfernte Zellen liegen korrekt außerhalb Polygon-Bounds")
                            else:
                                print(f"     ⚠️  Entfernte Zellen liegen innerhalb Polygon-Bounds! (Möglicher Filterfehler)")
            
            cells_to_keep = np.where(inside_mask_1d)[0].tolist()
            
            # Filtere Mesh: Nur gültige Zellen behalten
            if len(cells_to_keep) < n_cells:
                filtered_mesh = mesh.extract_cells(cells_to_keep)
                if DEBUG_PLOT3D_TIMING and is_slanted:
                    print(f"[DEBUG Plot Mesh] [{vertical_orientation}] Filterung: {n_cells} → {len(cells_to_keep)} Zellen (entfernt: {n_cells - len(cells_to_keep)})")
                    print(f"  └─ Mesh-Punkte: {mesh.n_points} → {filtered_mesh.n_points} (+{filtered_mesh.n_points - mesh.n_points})")
                    
                    # 🎯 DEBUG: Prüfe Bounds der gefilterten Fläche vs. Polygon
                    if len(filtered_mesh.points) > 0:
                        filtered_points_3d = filtered_mesh.points
                        if vertical_orientation in ("xz", "xz_slanted"):
                            filtered_u = filtered_points_3d[:, 0]  # X
                            filtered_v = filtered_points_3d[:, 2]  # Z
                        elif vertical_orientation in ("yz", "yz_slanted"):
                            filtered_u = filtered_points_3d[:, 1]  # Y
                            filtered_v = filtered_points_3d[:, 2]  # Z
                        else:
                            filtered_u = filtered_points_3d[:, 0]  # X
                            filtered_v = filtered_points_3d[:, 1]  # Y
                        
                        poly_u = np.array([p.get('x', 0.0) for p in poly_points_dict], dtype=float)
                        poly_v = np.array([p.get('y', 0.0) for p in poly_points_dict], dtype=float)
                        
                        # Prüfe ob gefilterte Fläche außerhalb Polygon-Bounds liegt
                        u_outside = (filtered_u.min() < poly_u.min()) or (filtered_u.max() > poly_u.max())
                        v_outside = (filtered_v.min() < poly_v.min()) or (filtered_v.max() > poly_v.max())
                        
                        if u_outside or v_outside:
                            print(f"  ⚠️  GEFILTERTE FLÄCHE LIEGT AUSSERHALB POLYGON-BOUNDS!")
                            print(f"     └─ Polygon U: [{poly_u.min():.3f}, {poly_u.max():.3f}], Gefiltert U: [{filtered_u.min():.3f}, {filtered_u.max():.3f}]")
                            print(f"     └─ Polygon V: [{poly_v.min():.3f}, {poly_v.max():.3f}], Gefiltert V: [{filtered_v.min():.3f}, {filtered_v.max():.3f}]")
                            
                            # Prüfe wie viele Punkte außerhalb liegen
                            points_outside_u = np.sum((filtered_u < poly_u.min()) | (filtered_u > poly_u.max()))
                            points_outside_v = np.sum((filtered_v < poly_v.min()) | (filtered_v > poly_v.max()))
                            print(f"     └─ Punkte außerhalb U-Bounds: {points_outside_u}/{len(filtered_u)}")
                            print(f"     └─ Punkte außerhalb V-Bounds: {points_outside_v}/{len(filtered_v)}")
                return filtered_mesh
            
            # 🎯 DEBUG: Auch wenn keine Filterung stattfand, prüfe Bounds
            if DEBUG_PLOT3D_TIMING and is_slanted:
                if len(mesh.points) > 0:
                    mesh_points_3d = mesh.points
                    if vertical_orientation in ("xz", "xz_slanted"):
                        mesh_u = mesh_points_3d[:, 0]  # X
                        mesh_v = mesh_points_3d[:, 2]  # Z
                    elif vertical_orientation in ("yz", "yz_slanted"):
                        mesh_u = mesh_points_3d[:, 1]  # Y
                        mesh_v = mesh_points_3d[:, 2]  # Z
                    else:
                        mesh_u = mesh_points_3d[:, 0]  # X
                        mesh_v = mesh_points_3d[:, 1]  # Y
                    
                    poly_u = np.array([p.get('x', 0.0) for p in poly_points_dict], dtype=float)
                    poly_v = np.array([p.get('y', 0.0) for p in poly_points_dict], dtype=float)
                    
                    u_outside = (mesh_u.min() < poly_u.min()) or (mesh_u.max() > poly_u.max())
                    v_outside = (mesh_v.min() < poly_v.min()) or (mesh_v.max() > poly_v.max())
                    
                    if u_outside or v_outside:
                        print(f"  ⚠️  UNGEFILTERTE FLÄCHE LIEGT AUSSERHALB POLYGON-BOUNDS!")
                        print(f"     └─ Polygon U: [{poly_u.min():.3f}, {poly_u.max():.3f}], Mesh U: [{mesh_u.min():.3f}, {mesh_u.max():.3f}]")
                        print(f"     └─ Polygon V: [{poly_v.min():.3f}, {poly_v.max():.3f}], Mesh V: [{mesh_v.min():.3f}, {mesh_v.max():.3f}]")
            
            return mesh
            
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Fehler beim Filtern von Dreiecken: {e}")
            return mesh

    @measure_time("PlotSPL3D.update_spl_plot")
    def update_spl_plot(
        self,
        sound_field_x: Iterable[float],
        sound_field_y: Iterable[float],
        sound_field_pressure: Iterable[float],
        colorization_mode: str = "Gradient",
    ):
        """
        🎯 NEUER PLOT: Plottet direkt die berechneten Grid-Punkte pro Surface.
        
        - Verwendet Daten aus calculation_spl['surface_grids'] und calculation_spl['surface_results']
        - Surface-Maske: Nur Punkte innerhalb der Surface werden geplottet
        - Upscaling für bessere Darstellung
        - Unterstützt Color step und Gradient
        """
        if not hasattr(self, "plotter") or self.plotter is None:
            return

        if pv is None:
            return

        t_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
        
        # 🎯 ENTFERNE EMPTY PLOT FLÄCHEN: Entferne graue Flächen aus Empty Plot, wenn SPL Plot erstellt wird
        # Die Rahmen (Wireframes) bleiben bestehen, nur die Flächen werden entfernt
        try:
            if hasattr(self, 'overlay_surfaces') and hasattr(self.plotter, 'renderer'):
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
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Fehler beim Entfernen von Empty Plot Flächen: {e}")
        
        # Hole Container mit Berechnungsergebnissen
        container = getattr(self, "container", None)
        if container is None or not hasattr(container, "calculation_spl"):
            return

        calc_spl = getattr(container, "calculation_spl", {}) or {}
        surface_grids_data = calc_spl.get("surface_grids", {})
        surface_results_data = calc_spl.get("surface_results", {})
        
        if not surface_grids_data or not surface_results_data:
            return

        # Plot-Mode bestimmen
        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'
        time_mode = plot_mode == 'SPL over time'
        self._phase_mode_active = phase_mode
        self._time_mode_active = time_mode

        # Colorbar-Parameter
        self.colorbar_manager.update_modes(phase_mode_active=phase_mode, time_mode_active=time_mode)
        self.colorbar_manager.set_override(None)
        colorbar_params = self.colorbar_manager.get_colorbar_params(phase_mode)
        cbar_min = colorbar_params['min']
        cbar_max = colorbar_params['max']
        cbar_step = colorbar_params['step']
        tick_step = colorbar_params['tick_step']
        
        # Colorization-Mode
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0
        
        # Colormap
        if time_mode:
            cmap_object = 'RdBu_r'
        elif phase_mode:
            cmap_object = PHASE_CMAP
        else:
            cmap_object = 'jet'
        
        # Entferne alte Surface-Actors
        if hasattr(self, '_surface_actors'):
            for surface_id, actor in list(self._surface_actors.items()):
                try:
                        self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self._surface_actors.clear()
        else:
            self._surface_actors = {}
        
        # Hole Surface-Definitionen für enabled/hidden-Prüfung
        surface_definitions = getattr(self.settings, 'surface_definitions', {})
        if not isinstance(surface_definitions, dict):
            surface_definitions = {}
        
        # Verarbeite jede Surface
        surfaces_processed = 0
        for surface_id in surface_grids_data.keys():
            if surface_id not in surface_results_data:
                continue
            
            # 🎯 PRÜFE ENABLED/HIDDEN: Überspringe disabled oder hidden Surfaces
            # Nur prüfen, wenn Surface in Definitionen vorhanden ist
            # Wenn nicht vorhanden, als enabled behandeln (Rückwärtskompatibilität)
            surf_def = surface_definitions.get(surface_id)
            if surf_def is not None:
                # Direkter Zugriff auf Attribute (schneller als to_dict())
                try:
                    if isinstance(surf_def, dict):
                        enabled = bool(surf_def.get('enabled', True))  # Default: enabled
                        hidden = bool(surf_def.get('hidden', False))
                    else:
                        # SurfaceDefinition-Objekt
                        enabled = bool(getattr(surf_def, 'enabled', True))  # Default: enabled
                        hidden = bool(getattr(surf_def, 'hidden', False))
                    
                    if not enabled or hidden:
                        if DEBUG_PLOT3D_TIMING:
                            print(f"[DEBUG Plot] Surface '{surface_id}': Überspringe (enabled={enabled}, hidden={hidden})")
                        continue
                except Exception:
                    # Bei Fehler: Weiter mit Plotting (sicherer Fallback)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Plot] Surface '{surface_id}': Fehler bei enabled/hidden-Prüfung, überspringe Prüfung")
                    pass
            
            try:
                # Lade Grid-Daten
                grid_data = surface_grids_data[surface_id]
                result_data = surface_results_data[surface_id]
                
                # 🎯 PRÜFE ORIENTIERUNG: Überspringe vertikale Surfaces (werden separat behandelt)
                orientation = grid_data.get('orientation', 'unknown')
                dominant_axis = grid_data.get('dominant_axis', None)  # 🎯 NEU: Lade dominant_axis
                
                if orientation == 'vertical':
                    if DEBUG_PLOT3D_TIMING:
                        print(f"[DEBUG Plot] Surface '{surface_id}': Überspringe vertikale Surface (wird separat behandelt)")
                    continue
                
                # 🎯 KONVERTIERE ORIENTIERUNG: Für Filterung brauchen wir "xz", "yz" oder None
                # Für planare/schräge Flächen: None (Standard XY-Ebene)
                # Für vertikale Flächen: dominant_axis ("xz" oder "yz")
                vertical_orientation_for_filter = None
                if orientation == 'vertical' and dominant_axis:
                    vertical_orientation_for_filter = dominant_axis
                # Für planare/schräge Flächen bleibt None (wird in XY-Ebene gefiltert)
                
                if DEBUG_PLOT3D_TIMING:
                    print(f"[DEBUG Plot] Surface '{surface_id}': orientation={orientation}, dominant_axis={dominant_axis}, vertical_orientation_for_filter={vertical_orientation_for_filter}")
                
                # Konvertiere zu numpy Arrays
                X_grid = np.array(grid_data['X_grid'], dtype=float)
                Y_grid = np.array(grid_data['Y_grid'], dtype=float)
                Z_grid = np.array(grid_data['Z_grid'], dtype=float)
                sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)
                sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)
                surface_mask = np.array(grid_data['surface_mask'], dtype=bool)
                
                # Lade SPL-Werte (komplex)
                sound_field_p_complex = np.array(result_data['sound_field_p'], dtype=complex)
                
                # Konvertiere zu SPL in dB
                if time_mode:
                    # Für Zeit-Modus: Direkt verwenden
                    spl_values = np.real(sound_field_p_complex)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                    spl_values = np.clip(spl_values, cbar_min, cbar_max)
                elif phase_mode:
                    # Für Phase-Modus: Phase extrahieren
                    spl_values = np.angle(sound_field_p_complex)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                    spl_values = np.clip(spl_values, cbar_min, cbar_max)
                else:
                    # Für SPL-Modus: Betrag zu dB
                    pressure_magnitude = np.abs(sound_field_p_complex)
                    pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
                    spl_values = self.functions.mag2db(pressure_magnitude)
                    spl_values = np.nan_to_num(spl_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                ny, nx = X_grid.shape
                
                # 🎯 UPSCALING: Erhöhe Grid-Auflösung (nur Plot, nicht Calc)
                if PLOT_UPSCALE_FACTOR > 1:
                    # Erstelle feineres Grid basierend auf den originalen Koordinaten
                    x_fine = np.linspace(sound_field_x.min(), sound_field_x.max(), nx * PLOT_UPSCALE_FACTOR)
                    y_fine = np.linspace(sound_field_y.min(), sound_field_y.max(), ny * PLOT_UPSCALE_FACTOR)
                    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='xy')
                    
                    # 🚀 OPTIMIERUNG: Verwende optimierte bilineare Interpolation für reguläre Grids
                    # statt scipy.interpolate.griddata (schneller für reguläre Grids)
                    # Interpoliere Z-Grid
                    Z_fine = self._bilinear_interpolate_grid(
                        sound_field_x, sound_field_y, Z_grid,
                        X_fine.ravel(), Y_fine.ravel()
                    )
                    Z_fine = Z_fine.reshape(X_fine.shape)
                    
                    # 🎯 INTERPOLATION: Wähle Methode basierend auf Color-Modus
                    # Color Step: Nearest Neighbor für harte Stufen
                    # Gradient: Bilineare Interpolation für glatte Übergänge
                    if is_step_mode:
                        # Nearest Neighbor: Jeder Punkt erhält exakt den Wert des nächstgelegenen Grid-Punkts
                        spl_fine = self._nearest_interpolate_grid(
                            sound_field_x, sound_field_y, spl_values,
                            X_fine.ravel(), Y_fine.ravel()
                        )
                    else:
                        # Bilineare Interpolation: Glatte Übergänge zwischen Grid-Punkten
                        spl_fine = self._bilinear_interpolate_grid(
                            sound_field_x, sound_field_y, spl_values,
                            X_fine.ravel(), Y_fine.ravel()
                        )
                    spl_fine = spl_fine.reshape(X_fine.shape)
                    
                    # Interpoliere Surface-Maske (nearest neighbor für bool-Werte)
                    mask_float = surface_mask.astype(float)
                    mask_fine = self._nearest_interpolate_grid(
                        sound_field_x, sound_field_y, mask_float,
                        X_fine.ravel(), Y_fine.ravel()
                    )
                    mask_fine = mask_fine.reshape(X_fine.shape).astype(bool)
                    
                    # Debug: Formen und Aktivierungsgrad
                    if DEBUG_PLOT3D_TIMING:
                        active_coarse = int(np.sum(surface_mask))
                        active_fine = int(np.sum(mask_fine))
                        print(
                            f"[DEBUG Plot] Upscale x{PLOT_UPSCALE_FACTOR}: "
                            f"coarse {surface_mask.shape}->{active_coarse} aktiv, "
                            f"fine {mask_fine.shape}->{active_fine} aktiv"
                        )

                    # Verwende upgescalte Daten
                    X_plot = X_fine
                    Y_plot = Y_fine
                    Z_plot = Z_fine
                    spl_plot = spl_fine
                    mask_plot = mask_fine
                    x_plot = x_fine
                    y_plot = y_fine
                else:
                    # Kein Upscaling
                    X_plot = X_grid
                    Y_plot = Y_grid
                    Z_plot = Z_grid
                    spl_plot = spl_values
                    mask_plot = surface_mask
                    x_plot = sound_field_x
                    y_plot = sound_field_y
                
                # Color step: Quantisiere Werte
                if is_step_mode:
                    scalars = self._quantize_to_steps(spl_plot, cbar_step)
                else:
                    scalars = spl_plot
                
                # Erstelle Mesh mit build_surface_mesh (mit Surface-Maske)
                mesh = build_surface_mesh(
                    x_plot,
                    y_plot,
                    scalars,
                    z_coords=Z_plot,
                    surface_mask=mask_plot,  # 🎯 MASKE: Nur Punkte innerhalb der Surface plotten
                    pv_module=pv,
                    settings=self.settings,
                    container=container,
                )
                
                if mesh is None or mesh.n_points == 0:
                    continue
            
                # 🎯 RANDPUNKTE: Integriere Randpunkte direkt in das Grid-Mesh
                # Legacy-Debug: deaktiviert, um unnötige Log-Ausgaben zu vermeiden
                # available_keys = list(result_data.keys())
                # has_edge_points = 'edge_points_x' in result_data
                # print(f"[DEBUG Plot] Surface '{surface_id}': Verfügbare Keys: {available_keys}")
                # print(f"[DEBUG Plot] Surface '{surface_id}': edge_points_x vorhanden: {has_edge_points}")
                
                if 'edge_points_x' in result_data:
                    edge_x = result_data.get('edge_points_x')
                    edge_y = result_data.get('edge_points_y')
                    edge_z = result_data.get('edge_points_z')
                    edge_spl = result_data.get('edge_points_spl')
                    
                    # Legacy-Debug: deaktiviert, um Log-Spam zu vermeiden
                    # print(f"[DEBUG Plot] Surface '{surface_id}': Randpunkte-Daten geladen - edge_x type: {type(edge_x)}, len: {len(edge_x) if edge_x is not None else 0}")
                
                    # Konvertiere zu NumPy-Arrays
                    if edge_x is not None:
                        edge_x = np.array(edge_x)
                    if edge_y is not None:
                        edge_y = np.array(edge_y)
                    if edge_z is not None:
                        edge_z = np.array(edge_z)
                    
                    edge_count = len(edge_x) if edge_x is not None else 0
                    if edge_count > 0:
                        # print(f"[DEBUG Plot] Surface '{surface_id}': Integriere {edge_count} Randpunkte")
                        
                        # Prüfe Randpunkte-Daten
                        if edge_spl is not None:
                            # Konvertiere von [real, imag] Paaren zu komplexen Zahlen
                            if isinstance(edge_spl[0], (list, tuple)) and len(edge_spl[0]) == 2:
                                # Liste von [real, imag] Paaren
                                edge_spl_arr = np.array([complex(r, i) for r, i in edge_spl], dtype=complex)
                            else:
                                # Bereits komplexe Zahlen
                                edge_spl_arr = np.array(edge_spl, dtype=complex)
                            edge_spl_mag = np.abs(edge_spl_arr)
                            edge_spl_db = 20 * np.log10(np.maximum(edge_spl_mag, 1e-12))
                            valid_spl = np.sum(np.isfinite(edge_spl_db))
                            print(f"  └─ Randpunkte SPL: {valid_spl}/{edge_count} gültig, Bereich: {np.nanmin(edge_spl_db):.1f} bis {np.nanmax(edge_spl_db):.1f} dB")
                        
                        # Übergebe bereits konvertierte Arrays
                        if edge_spl_arr is not None:
                            mesh_before = mesh.n_points if mesh is not None else 0
                            mesh = self._integrate_edge_points_into_mesh(
                                main_mesh=mesh,
                                edge_x=edge_x,
                                edge_y=edge_y,
                                edge_z=edge_z,
                                edge_spl=edge_spl_arr,  # Verwende bereits konvertiertes Array
                                time_mode=time_mode,
                                phase_mode=phase_mode,
                                cbar_min=cbar_min,
                                cbar_max=cbar_max,
                                pv_module=pv,
                                surface_id=surface_id,  # 🎯 NEU: Für Polygon-Prüfung
                                settings=self.settings,  # 🎯 NEU: Für Surface-Definition
                                vertical_orientation=vertical_orientation_for_filter,  # 🎯 NEU: Übergabe der Orientierung
                            )
                        else:
                            print(f"[DEBUG Plot] Surface '{surface_id}': Keine gültigen Randpunkte-SPL-Daten, überspringe Integration")
                        mesh_after = mesh.n_points if mesh is not None else 0
                        print(f"  └─ Mesh-Punkte: {mesh_before} → {mesh_after} (+{mesh_after - mesh_before})")
                else:
                    print(f"[DEBUG Plot] Surface '{surface_id}': KEINE Randpunkte gefunden!")
                
                if mesh is None or mesh.n_points == 0:
                    continue
                
                # Füge kombiniertes Mesh (Grid + Randpunkte) zum Plotter hinzu
                actor_name = f"{self.SURFACE_NAME}_grid_{surface_id}"
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
                
                self._surface_actors[surface_id] = actor
                
                surfaces_processed += 1
                
            except Exception as e:
                if DEBUG_PLOT3D_TIMING:
                    print(f"[PlotSPL3D] Error processing surface {surface_id}: {e}")
                continue
        
        # 🎯 VERTIKALE SURFACES: Behandle vertikale Surfaces separat
        self._update_vertical_spl_surfaces_from_grids()
        
        # Aktualisiere Colorbar
        self.colorbar_manager.render_colorbar(
            colorization_mode,
            force=False,
            tick_step=tick_step,
            phase_mode_active=phase_mode,
            time_mode_active=time_mode,
        )
        self.cbar = self.colorbar_manager.cbar
        
        # Camera
        if not self._has_plotted_data:
            self._maximize_camera_view(add_padding=True)
            self._has_plotted_data = True
        
        self.render()
        
        if DEBUG_PLOT3D_TIMING:
            t_end = time.perf_counter()
            print(
                f"[PlotSPL3D] update_spl_plot (NEU): "
                f"{(t_end - t_start) * 1000.0:.2f} ms, "
                f"surfaces={surfaces_processed}"
            )
        
            return

        # ============================================================
        # ⚠️ AUSKOMMENTIERTER ALTER CODE (für Referenz sichtbar)
        # Der folgende Code nach dem return ist unreachable und wurde
        # durch die neue Implementierung ersetzt (direkter Grid-Plot pro Surface).
        # Der alte Code rief _render_surfaces_textured() und
        # _update_vertical_spl_surfaces() auf.
        # 
        # WICHTIG: Dieser Code wird nicht ausgeführt, da er nach dem return steht!
        # ============================================================
        """
        # ALTER CODE BEGINNT HIER (Zeile 813-1160):
        # Der gesamte alte SPL-Plot-Code wurde hier auskommentiert,
        # da er durch die neue Implementierung ersetzt wurde.
        # 
        # Dieser Code ist unreachable (nach return) und wird nur
        # für Referenz-Zwecke sichtbar gehalten.
        
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
            cmap_object = PHASE_CMAP
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
        # Dies muss VOR dem Zeichnen der neuen SPL-Daten passieren.
        # Die Rahmen bleiben bestehen, da sie separat als Wireframes gezeichnet werden.
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

        # Zeichne alle aktiven Surfaces als Texturflächen (planar/sloped)
        with perf_section(
            "Plot3D.render_surfaces_textured",
            mode=colorization_mode_used,
            phase_mode=phase_mode,
            time_mode=time_mode,
        ):
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
        with perf_section(
            "Plot3D.update_vertical_spl_surfaces",
            mode=colorization_mode_used,
            phase_mode=phase_mode,
            time_mode=time_mode,
        ):
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
        # ALTER CODE ENDET HIER
        """

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

    def _update_vertical_spl_surfaces_from_grids(self) -> None:
        """
        🎯 NEU: Zeichnet vertikale Surfaces aus surface_grids und surface_results.
        
        Diese Funktion behandelt vertikale Surfaces, die in update_spl_plot() übersprungen wurden.
        Sie konvertiert die Grid-Daten in das (u,v)-Koordinatensystem für vertikale Surfaces.
        """
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
            cmap_object = 'RdBu_r'
        elif phase_mode:
            cmap_object = PHASE_CMAP
        else:
            cmap_object = 'jet'
        
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
                        # Schräge vertikale Surface: Variiert in X, Y und Z (Fallback)
                        # Bestimme dominante Richtung in XY
                        if x_span >= y_span:
                            vertical_orientation = "xz_slanted"
                            wall_value = float(np.mean(ys))
                        else:
                            vertical_orientation = "yz_slanted"
                            wall_value = float(np.mean(xs))
                
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
                    # Extrahiere X-Koordinaten aus X_grid (einzigartige Werte)
                    u_axis = np.unique(X_grid)  # X-Koordinaten
                    # Extrahiere Z-Koordinaten aus Z_grid (einzigartige Werte)
                    v_axis = np.unique(Z_grid)  # Z-Koordinaten
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ X-Z-Wand: u_axis (X) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "yz":
                    # Y-Z-Wand: x ≈ const
                    # Extrahiere Y-Koordinaten aus Y_grid (einzigartige Werte)
                    u_axis = np.unique(Y_grid)  # Y-Koordinaten
                    # Extrahiere Z-Koordinaten aus Z_grid (einzigartige Werte)
                    v_axis = np.unique(Z_grid)  # Z-Koordinaten
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Y-Z-Wand: u_axis (Y) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "xz_slanted":
                    # Schräge X-Z-Wand: X und Z variieren, Y variiert entlang der Fläche
                    # u = X (dominante Richtung), v = Z
                    # Y wird aus der Flächengeometrie interpoliert
                    u_axis = np.unique(X_grid)  # X-Koordinaten
                    v_axis = np.unique(Z_grid)  # Z-Koordinaten
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Schräge X-Z-Wand: u_axis (X) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                elif vertical_orientation == "yz_slanted":
                    # Schräge Y-Z-Wand: Y und Z variieren, X variiert entlang der Fläche
                    # u = Y (dominante Richtung), v = Z
                    # X wird aus der Flächengeometrie interpoliert
                    u_axis = np.unique(Y_grid)  # Y-Koordinaten
                    v_axis = np.unique(Z_grid)  # Z-Koordinaten
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ Schräge Y-Z-Wand: u_axis (Y) len={len(u_axis)}, v_axis (Z) len={len(v_axis)}")
                else:
                    # Fallback: Standard (sollte nicht vorkommen für vertical)
                    u_axis = sound_field_x
                    v_axis = sound_field_y
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  └─ ⚠️ Keine Orientierung erkannt, verwende sound_field_x/y")
                
                # 🎯 UPSCALING: Erhöhe Grid-Auflösung (in UV-Koordinaten)
                if PLOT_UPSCALE_FACTOR > 1:
                    # Erstelle feineres Grid basierend auf den originalen UV-Koordinaten
                    u_fine = np.linspace(u_axis.min(), u_axis.max(), len(u_axis) * PLOT_UPSCALE_FACTOR)
                    v_fine = np.linspace(v_axis.min(), v_axis.max(), len(v_axis) * PLOT_UPSCALE_FACTOR)
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
                        # Fallback: Normale XY-Interpolation
                        spl_fine = self._bilinear_interpolate_grid(
                            sound_field_x, sound_field_y, spl_values,
                            U_fine.ravel(), V_fine.ravel()
                        )
                        spl_fine = spl_fine.reshape(U_fine.shape)
                        mask_float = surface_mask.astype(float)
                        mask_fine = self._nearest_interpolate_grid(
                            sound_field_x, sound_field_y, mask_float,
                            U_fine.ravel(), V_fine.ravel()
                        )
                        mask_fine = mask_fine.reshape(U_fine.shape).astype(bool)
                        Z_fine = self._bilinear_interpolate_grid(
                            sound_field_x, sound_field_y, Z_grid,
                            U_fine.ravel(), V_fine.ravel()
                        )
                        Z_fine = Z_fine.reshape(U_fine.shape)
                    
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
                            f"[DEBUG Plot] Upscale x{PLOT_UPSCALE_FACTOR} (vertical {vertical_orientation}): "
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
                
                # Color step: Quantisiere Werte
                if is_step_mode:
                    scalars = self._quantize_to_steps(spl_plot, cbar_step)
                else:
                    scalars = spl_plot
                
                # 🎯 ERSTELLE MESH: Transformierte Koordinaten für vertikale Flächen
                mesh = build_surface_mesh(
                    x_plot,  # Für X-Z-Wand: X-Koordinaten, für Y-Z-Wand: Y-Koordinaten
                    y_plot,  # Für X-Z-Wand: Z-Koordinaten, für Y-Z-Wand: Z-Koordinaten
                    scalars,
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
                
                # 🎯 RANDPUNKTE: Integriere Randpunkte direkt in das Mesh (identisch zu horizontalen Flächen)
                if 'edge_points_x' in result_data:
                    edge_x = result_data.get('edge_points_x')
                    edge_y = result_data.get('edge_points_y')
                    edge_z = result_data.get('edge_points_z')
                    edge_spl = result_data.get('edge_points_spl')
                    
                    if edge_x is not None and len(edge_x) > 0:
                        # Konvertiere von [real, imag] Paaren zu komplexen Zahlen (wie bei horizontalen Flächen)
                        if isinstance(edge_spl[0], (list, tuple)) and len(edge_spl[0]) == 2:
                            edge_spl_arr = np.array([complex(r, i) for r, i in edge_spl], dtype=complex)
                        else:
                            edge_spl_arr = np.array(edge_spl, dtype=complex)
                        
                        if DEBUG_PLOT3D_TIMING:
                            print(f"[DEBUG Vertical] Surface '{surface_id}': Integriere {len(edge_x)} Randpunkte (Orientierung: {vertical_orientation})")
                        
                        mesh_before = mesh.n_points
                        mesh_bounds_before = None
                        if DEBUG_PLOT3D_TIMING and vertical_orientation in ("xz_slanted", "yz_slanted"):
                            mesh_bounds_before = (
                                mesh.points[:, 0].min(), mesh.points[:, 0].max(),
                                mesh.points[:, 1].min(), mesh.points[:, 1].max(),
                                mesh.points[:, 2].min(), mesh.points[:, 2].max()
                            )
                        mesh = self._integrate_edge_points_into_mesh(
                            main_mesh=mesh,
                            edge_x=np.array(edge_x, dtype=float),
                            edge_y=np.array(edge_y, dtype=float),
                            edge_z=np.array(edge_z, dtype=float),
                            edge_spl=edge_spl_arr,
                            time_mode=time_mode,
                            phase_mode=phase_mode,
                            cbar_min=cbar_min,
                            cbar_max=cbar_max,
                            pv_module=pv,
                            surface_id=surface_id,
                            settings=self.settings,
                            vertical_orientation=vertical_orientation,  # 🎯 NEU: Übergabe der Orientierung
                            wall_value=wall_value,  # 🎯 NEU: Konstanter Wert für konstante Achse
                        )
                        mesh_after = mesh.n_points if mesh is not None else 0
                        if DEBUG_PLOT3D_TIMING:
                            if vertical_orientation in ("xz_slanted", "yz_slanted") and mesh is not None:
                                mesh_bounds_after = (
                                    mesh.points[:, 0].min(), mesh.points[:, 0].max(),
                                    mesh.points[:, 1].min(), mesh.points[:, 1].max(),
                                    mesh.points[:, 2].min(), mesh.points[:, 2].max()
                                )
                                print(f"[DEBUG Vertical] Surface '{surface_id}' ({vertical_orientation}): Mesh-Punkte: {mesh_before} → {mesh_after} (+{mesh_after - mesh_before})")
                                if mesh_bounds_before:
                                    print(f"  └─ Bounds VOR Randpunkten: X[{mesh_bounds_before[0]:.2f}, {mesh_bounds_before[1]:.2f}] Y[{mesh_bounds_before[2]:.2f}, {mesh_bounds_before[3]:.2f}] Z[{mesh_bounds_before[4]:.2f}, {mesh_bounds_before[5]:.2f}]")
                                print(f"  └─ Bounds NACH Randpunkten: X[{mesh_bounds_after[0]:.2f}, {mesh_bounds_after[1]:.2f}] Y[{mesh_bounds_after[2]:.2f}, {mesh_bounds_after[3]:.2f}] Z[{mesh_bounds_after[4]:.2f}, {mesh_bounds_after[5]:.2f}]")
                                # Prüfe auf große Änderungen
                                if mesh_bounds_before:
                                    z_diff = abs(mesh_bounds_after[4] - mesh_bounds_before[4]) + abs(mesh_bounds_after[5] - mesh_bounds_before[5])
                                    if z_diff > 0.5:
                                        print(f"  ⚠️  Große Z-Koordinaten-Änderung nach Randpunkten: {z_diff:.2f}")
                            else:
                                    print(f"[DEBUG Vertical] Surface '{surface_id}': Mesh-Punkte: {mesh_before} → {mesh_after} (+{mesh_after - mesh_before})")
                
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
                                    outside_x = (final_bounds['X'][0] < poly_bounds['X'][0]) or (final_bounds['X'][1] > poly_bounds['X'][1])
                                    outside_y = (final_bounds['Y'][0] < poly_bounds['Y'][0]) or (final_bounds['Y'][1] > poly_bounds['Y'][1])
                                    outside_z = (final_bounds['Z'][0] < poly_bounds['Z'][0]) or (final_bounds['Z'][1] > poly_bounds['Z'][1])
                                    
                                    if outside_x or outside_y or outside_z:
                                        print(f"  ⚠️  GEPLOTTETE FLÄCHE LIEGT AUSSERHALB POLYGON-BOUNDS!")
                                        print(f"     └─ Polygon X: [{poly_bounds['X'][0]:.2f}, {poly_bounds['X'][1]:.2f}], Geplottet X: [{final_bounds['X'][0]:.2f}, {final_bounds['X'][1]:.2f}]")
                                        print(f"     └─ Polygon Y: [{poly_bounds['Y'][0]:.2f}, {poly_bounds['Y'][1]:.2f}], Geplottet Y: [{final_bounds['Y'][0]:.2f}, {final_bounds['Y'][1]:.2f}]")
                                        print(f"     └─ Polygon Z: [{poly_bounds['Z'][0]:.2f}, {poly_bounds['Z'][1]:.2f}], Geplottet Z: [{final_bounds['Z'][0]:.2f}, {final_bounds['Z'][1]:.2f}]")
                                        
                                        # Berechne wie viele Punkte außerhalb liegen
                                        points_outside_x = np.sum((final_points[:, 0] < poly_bounds['X'][0]) | (final_points[:, 0] > poly_bounds['X'][1]))
                                        points_outside_y = np.sum((final_points[:, 1] < poly_bounds['Y'][0]) | (final_points[:, 1] > poly_bounds['Y'][1]))
                                        points_outside_z = np.sum((final_points[:, 2] < poly_bounds['Z'][0]) | (final_points[:, 2] > poly_bounds['Z'][1]))
                                        print(f"     └─ Punkte außerhalb X-Bounds: {points_outside_x}/{mesh.n_points} ({100*points_outside_x/mesh.n_points:.1f}%)")
                                        print(f"     └─ Punkte außerhalb Y-Bounds: {points_outside_y}/{mesh.n_points} ({100*points_outside_y/mesh.n_points:.1f}%)")
                                        print(f"     └─ Punkte außerhalb Z-Bounds: {points_outside_z}/{mesh.n_points} ({100*points_outside_z/mesh.n_points:.1f}%)")
                    
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