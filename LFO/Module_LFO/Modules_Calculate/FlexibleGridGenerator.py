"""
FlexibleGridGenerator: Hybrid-Ansatz für Grid-Generierung

Diese Implementierung basiert auf dem Hybrid-Ansatz (Vorschlag 5):
- Surface Analyzer: Analysiert Polygone (flach/schräg/vertikal)
- Grid Builder: Erstellt Basis-Grid (3D-Punkte + Topologie)
- Grid Transformers: Wandelt Basis-Grid in berechnungsspezifisches Format
  - CartesianTransformer: Für Superposition (SPL-Berechnung)
  - FEMTransformer: Für FEM (später)
  - FDTDTransformer: Für FDTD (später)

Input: Polygone (3+ Punkte) in 3D-Koordinaten
Output: Grids für verschiedene Berechnungen (Superposition, FEM, FDTD)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import math
import os

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Init.Logging import measure_time, perf_section
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
    SurfaceDefinition,
    _evaluate_plane_on_grid,
)

DEBUG_FLEXIBLE_GRID = bool(int(os.environ.get("LFO_DEBUG_FLEXIBLE_GRID", "0")))


@dataclass
class SurfaceGeometry:
    """Geometrie-Informationen für ein Surface"""
    surface_id: str
    name: str
    points: List[Dict[str, float]]  # Original-Punkte
    plane_model: Optional[Dict[str, float]] = None
    orientation: str = "unknown"  # "planar", "sloped", "vertical"
    normal: Optional[np.ndarray] = None  # Normale (falls planare Fläche)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (min_x, max_x, min_y, max_y)


@dataclass
class BaseGrid:
    """Basis-Grid mit 3D-Punkten und Topologie-Informationen"""
    points_3d: np.ndarray  # Shape: (N, 3) - alle Punkte in 3D
    surface_masks: Dict[str, np.ndarray]  # Maske pro Surface (Shape: (N,))
    topology: Optional[Dict[str, Any]] = None  # Zusätzliche Topologie-Informationen
    resolution: float = 1.0  # Grid-Resolution
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding Box (min_x, max_x, min_y, max_y)


@dataclass
class SurfaceGrid:
    """Grid für eine einzelne Surface"""
    surface_id: str
    sound_field_x: np.ndarray  # 1D-Array X-Koordinaten
    sound_field_y: np.ndarray  # 1D-Array Y-Koordinaten
    X_grid: np.ndarray  # 2D-Meshgrid X (Shape: [ny, nx])
    Y_grid: np.ndarray  # 2D-Meshgrid Y (Shape: [ny, nx])
    Z_grid: np.ndarray  # 2D-Array Z-Koordinaten (Shape: [ny, nx])
    surface_mask: np.ndarray  # 2D-Boolean-Array (Shape: [ny, nx])
    resolution: float  # Tatsächliche Resolution (kann adaptiv sein)
    geometry: 'SurfaceGeometry'  # Surface-Geometrie


@dataclass
class CartesianGrid:
    """Cartesian Grid für Superposition-Berechnung (kompatibel mit SoundfieldCalculator)"""
    sound_field_x: np.ndarray  # 1D-Array X-Koordinaten
    sound_field_y: np.ndarray  # 1D-Array Y-Koordinaten
    X_grid: np.ndarray  # 2D-Meshgrid X (Shape: [ny, nx])
    Y_grid: np.ndarray  # 2D-Meshgrid Y (Shape: [ny, nx])
    Z_grid: np.ndarray  # 2D-Array Z-Koordinaten (Shape: [ny, nx])
    surface_mask: np.ndarray  # 2D-Boolean-Array (Shape: [ny, nx])
    surface_mask_strict: Optional[np.ndarray] = None  # Strikte Maske ohne Erweiterung


class SurfaceAnalyzer(ModuleBase):
    """Analysiert Polygone und bestimmt deren Geometrie-Charakteristika"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
    
    def analyze_surfaces(
        self,
        enabled_surfaces: List[Tuple[str, Dict]]
    ) -> List[SurfaceGeometry]:
        """
        Analysiert alle enabled Surfaces und bestimmt deren Geometrie.
        
        Args:
            enabled_surfaces: Liste von (surface_id, surface_definition) Tupeln
            
        Returns:
            Liste von SurfaceGeometry Objekten
        """
        geometries = []
        
        for surface_id, surface_def in enabled_surfaces:
            # Konvertiere zu SurfaceDefinition falls nötig
            if not isinstance(surface_def, SurfaceDefinition):
                surface_def_dict = surface_def
            else:
                surface_def_dict = surface_def.to_dict()
            
            points = surface_def_dict.get('points', [])
            if len(points) < 3:
                continue
            
            name = surface_def_dict.get('name', surface_id)
            
            # Berechne Bounding Box
            xs = [p.get('x', 0.0) for p in points]
            ys = [p.get('y', 0.0) for p in points]
            zs = [p.get('z', 0.0) for p in points]
            
            if not xs or not ys:
                continue
            
            bbox = (min(xs), max(xs), min(ys), max(ys))
            
            # Versuche Plane-Model zu bestimmen
            plane_model, error = derive_surface_plane(points)
            
            # Bestimme Orientierung
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            z_span = max(zs) - min(zs) if zs else 0.0
            
            orientation = "unknown"
            normal = None
            
            if plane_model is not None:
                mode = plane_model.get('mode', 'constant')
                if mode == 'constant':
                    orientation = "planar"
                elif mode in ['x', 'y', 'xy']:
                    # Prüfe, ob es eine schräge vertikale Fläche ist (z_span > max(x_span, y_span) * 0.5)
                    if z_span > max(x_span, y_span) * 0.5 and z_span > 1e-3:
                        # Schräge vertikale Fläche: hat Plane-Model, aber z_span ist signifikant größer
                        orientation = "vertical"
                    else:
                        orientation = "sloped"
            else:
                # Kein Plane-Model → könnte vertikal sein
                if (x_span < 1e-6 or y_span < 1e-6) and z_span > 1e-3:
                    orientation = "vertical"
                elif z_span > max(x_span, y_span) * 0.5 and z_span > 1e-3:
                    # Schräge vertikale Fläche ohne Plane-Model
                    orientation = "vertical"
            
            geometry = SurfaceGeometry(
                surface_id=surface_id,
                name=name,
                points=points,
                plane_model=plane_model,
                orientation=orientation,
                normal=normal,
                bbox=bbox
            )
            
            geometries.append(geometry)
        
        return geometries


class GridBuilder(ModuleBase):
    """Erstellt Basis-Grid aus Surface-Geometrie"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
    
    @measure_time("GridBuilder.build_base_grid")
    def build_base_grid(
        self,
        geometries: List[SurfaceGeometry],
        resolution: Optional[float] = None,
        padding_factor: float = 0.5
    ) -> BaseGrid:
        """
        Erstellt Basis-Grid aus Surface-Geometrien.
        
        Args:
            geometries: Liste von SurfaceGeometry Objekten
            resolution: Grid-Auflösung in Metern
            padding_factor: Padding-Faktor für Bounding Box
            
        Returns:
            BaseGrid Objekt
        """
        if resolution is None:
            resolution = self.settings.resolution
        
        # Berechne Bounding Box aller Surfaces
        if geometries:
            all_x = []
            all_y = []
            for geom in geometries:
                if geom.bbox:
                    min_x, max_x, min_y, max_y = geom.bbox
                    all_x.extend([min_x, max_x])
                    all_y.extend([min_y, max_y])
            
            if all_x and all_y:
                padding = resolution * padding_factor
                min_x = min(all_x) - padding
                max_x = max(all_x) + padding
                min_y = min(all_y) - padding
                max_y = max(all_y) + padding
            else:
                # Fallback: Verwende Settings-Dimensionen
                width = getattr(self.settings, 'width', 150.0)
                length = getattr(self.settings, 'length', 100.0)
                min_x = -width / 2
                max_x = width / 2
                min_y = -length / 2
                max_y = length / 2
        else:
            # Keine Surfaces → Fallback
            width = getattr(self.settings, 'width', 150.0)
            length = getattr(self.settings, 'length', 100.0)
            min_x = -width / 2
            max_x = width / 2
            min_y = -length / 2
            max_y = length / 2
        
        bbox = (min_x, max_x, min_y, max_y)
        
        # Erstelle 1D-Koordinaten-Arrays
        sound_field_x = np.arange(min_x, max_x + resolution, resolution)
        sound_field_y = np.arange(min_y, max_y + resolution, resolution)
        
        # Erstelle 2D-Meshgrid
        X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
        
        # Erstelle 3D-Punkte-Array (flach)
        ny, nx = X_grid.shape
        points_3d = np.stack([
            X_grid.flatten(),
            Y_grid.flatten(),
            np.zeros(ny * nx)  # Z wird später gesetzt
        ], axis=1)  # Shape: (N, 3)
        
        # Erstelle Surface-Masken und prüfe auf kleine Flächen
        surface_masks = {}
        small_surfaces = []  # Liste von Surfaces mit < 3 Punkten
        
        for geom in geometries:
            mask = self._create_surface_mask(X_grid, Y_grid, geom)
            points_in_surface = int(np.sum(mask))
            
            # Prüfe auf kleine Flächen
            if points_in_surface > 0 and points_in_surface < 3:
                small_surfaces.append((geom, points_in_surface, mask))
                print(f"[DEBUG Kleine Flächen] ⚠️  Surface '{geom.surface_id}' ({geom.name}): Nur {points_in_surface} Grid-Punkt(e) bei Resolution {resolution:.3f} m")
            
            surface_masks[geom.surface_id] = mask.flatten()  # Flach für BaseGrid
        
        # 🎯 Adaptive Resolution für kleine Flächen
        if small_surfaces:
            print(f"[DEBUG Kleine Flächen] Gefunden: {len(small_surfaces)} kleine Flächen")
            # Für jetzt: Warnung ausgeben
            # Später: Adaptive Resolution implementieren (siehe TODO)
            for geom, n_points, mask in small_surfaces:
                # Berechne empfohlene Resolution
                if geom.bbox:
                    min_x, max_x, min_y, max_y = geom.bbox
                    diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                    recommended_resolution = diagonal / 3.0  # Mindestens 3 Punkte
                    recommended_resolution = min(recommended_resolution, resolution * 0.5)  # Nicht kleiner als halbe Basis-Resolution
                    print(f"  └─ Empfohlene Resolution: {recommended_resolution:.3f} m (aktuell: {resolution:.3f} m, Diagonal: {diagonal:.3f} m)")
        
        return BaseGrid(
            points_3d=points_3d,
            surface_masks=surface_masks,
            topology={
                'X_grid': X_grid,
                'Y_grid': Y_grid,
                'sound_field_x': sound_field_x,
                'sound_field_y': sound_field_y,
                'resolution': resolution,
                'shape': (ny, nx)
            },
            resolution=resolution,
            bbox=bbox
        )
    
    def _create_surface_mask(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        geometry: SurfaceGeometry
    ) -> np.ndarray:
        """Erstellt Maske für ein Surface (Punkt-im-Polygon-Test)"""
        points = geometry.points
        if len(points) < 3:
            return np.zeros_like(X_grid, dtype=bool)
        
        # 🎯 VERTIKALE SURFACES: Verwende (u,v)-Koordinaten statt (x,y)
        if geometry.orientation == "vertical":
            # Bestimme Orientierung (X-Z oder Y-Z)
            xs = np.array([p.get('x', 0.0) for p in points], dtype=float)
            ys = np.array([p.get('y', 0.0) for p in points], dtype=float)
            zs = np.array([p.get('z', 0.0) for p in points], dtype=float)
            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            
            eps_line = 1e-6
            if y_span < eps_line and x_span >= eps_line:
                # X-Z-Wand: y ≈ const, verwende (x,z) = (u,v)
                U_grid = X_grid
                V_grid = np.zeros_like(X_grid)  # Wird später mit Z_grid gefüllt
                # Polygon in (u,v) = (x,z)
                polygon_uv = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
            elif x_span < eps_line and y_span >= eps_line:
                # Y-Z-Wand: x ≈ const, verwende (y,z) = (u,v)
                U_grid = Y_grid
                V_grid = np.zeros_like(Y_grid)  # Wird später mit Z_grid gefüllt
                # Polygon in (u,v) = (y,z)
                polygon_uv = [
                    {"x": float(p.get("y", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
            else:
                # Fallback: Verwende normale (x,y)-Prüfung
                return self._points_in_polygon_batch(X_grid, Y_grid, points)
            
            # Für vertikale Surfaces müssen wir Z_grid haben, um V_grid zu füllen
            # Da wir hier nur X_grid und Y_grid haben, müssen wir die Maske später anpassen
            # Oder: Wir verwenden eine vereinfachte Prüfung basierend auf den Polygon-Grenzen
            # Für jetzt: Verwende normale Prüfung als Fallback
            # TODO: Z_grid sollte hier verfügbar sein
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG Vertical Mask] Surface '{geometry.surface_id}': Verwende vereinfachte Maske für vertikale Surface")
            # Vereinfachte Maske: Alle Punkte im Grid sind aktiv (wird später in build_single_surface_grid angepasst)
            return np.ones_like(X_grid, dtype=bool)
        
        # Normale (x,y)-Prüfung für planare/schräge Surfaces
        return self._points_in_polygon_batch(X_grid, Y_grid, points)
    
    def _points_in_polygon_batch(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polygon_points: List[Dict[str, float]]
    ) -> np.ndarray:
        """Vektorisierte Punkt-im-Polygon-Prüfung"""
        if len(polygon_points) < 3:
            return np.zeros_like(x_coords, dtype=bool)
        
        px = np.array([p.get('x', 0.0) for p in polygon_points])
        py = np.array([p.get('y', 0.0) for p in polygon_points])
        
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        n_points = len(x_flat)
        
        inside = np.zeros(n_points, dtype=bool)
        boundary_eps = 1e-6
        n = len(px)
        
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            
            y_above_edge = (yi > y_flat) != (yj > y_flat)
            intersection_x = (xj - xi) * (y_flat - yi) / (yj - yi + 1e-10) + xi
            intersects = y_above_edge & (x_flat <= intersection_x + boundary_eps)
            
            inside = inside ^ intersects
            
            j = i
        
        return inside.reshape(x_coords.shape)
    
    def _points_in_polygon_batch_uv(
        self,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        polygon_points: List[Dict[str, float]]
    ) -> np.ndarray:
        """Vektorisierte Punkt-im-Polygon-Prüfung in (u,v)-Koordinaten"""
        if len(polygon_points) < 3:
            return np.zeros_like(u_coords, dtype=bool)
        
        pu = np.array([p.get('x', 0.0) for p in polygon_points])  # u-Koordinate
        pv = np.array([p.get('y', 0.0) for p in polygon_points])  # v-Koordinate
        
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        n_points = len(u_flat)
        
        inside = np.zeros(n_points, dtype=bool)
        boundary_eps = 1e-6
        n = len(pu)
        
        j = n - 1
        for i in range(n):
            ui, vi = pu[i], pv[i]
            uj, vj = pu[j], pv[j]
            
            v_above_edge = (vi > v_flat) != (vj > v_flat)
            intersection_u = (uj - ui) * (v_flat - vi) / (vj - vi + 1e-10) + ui
            intersects = v_above_edge & (u_flat <= intersection_u + boundary_eps)
            
            inside = inside ^ intersects
            
            j = i
        
        return inside.reshape(u_coords.shape)
    
    def _dilate_mask_minimal(self, mask: np.ndarray) -> np.ndarray:
        """
        Einfache 3x3-Dilatation für boolesche Masken.
        Wird genutzt, um die Sample-Maske von Surfaces um 1 Zelle zu erweitern,
        analog zur Behandlung der Plot-Maske bei horizontalen Flächen.
        (Übernommen aus SurfaceGridCalculator)
        """
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        )
        ny, nx = mask.shape
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
        dilated = np.zeros_like(mask, dtype=bool)
        for i in range(ny):
            for j in range(nx):
                region = padded[i : i + 3, j : j + 3]
                dilated[i, j] = np.any(region & kernel)
        return dilated
    
    @measure_time("GridBuilder.build_single_surface_grid")
    def build_single_surface_grid(
        self,
        geometry: SurfaceGeometry,
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6,
        padding_factor: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Erstellt Grid für eine einzelne Surface mit Mindestanzahl von Punkten.
        
        Args:
            geometry: SurfaceGeometry Objekt
            resolution: Basis-Resolution in Metern (wenn None: settings.resolution)
            min_points_per_dimension: Mindestanzahl Punkte pro Dimension (Standard: 3)
            padding_factor: Padding-Faktor für Bounding Box
        
        Returns:
            Tuple von (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask)
        """
        if resolution is None:
            resolution = self.settings.resolution
        
        # 🎯 VERTIKALE SURFACES: Erstelle Grid direkt in (u,v)-Ebene der Fläche
        # Folgt der bewährten Logik aus SurfaceGridCalculator._build_vertical_surface_samples
        if geometry.orientation == "vertical":
            # Bestimme Orientierung und erstelle Grid in (u,v)-Ebene
            points = geometry.points
            xs = np.array([p.get('x', 0.0) for p in points], dtype=float)
            ys = np.array([p.get('y', 0.0) for p in points], dtype=float)
            zs = np.array([p.get('z', 0.0) for p in points], dtype=float)
            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            z_span = float(np.ptp(zs))
            
            eps_line = 1e-6
            
            # 🎯 ALTE LOGIK: Verfeinere Resolution für vertikale Flächen (wie in _build_vertical_surface_samples)
            base_resolution = float(resolution or 1.0)
            refine_factor = 2.0
            step = base_resolution / refine_factor
            if step <= 0.0:
                step = base_resolution or 1.0
            
            if y_span < eps_line and x_span >= eps_line:
                # X-Z-Wand: y ≈ const, Grid in (x,z)-Ebene
                # u = x, v = z
                y0 = float(np.mean(ys))
                u_min, u_max = float(xs.min()), float(xs.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                
                # Prüfe auf degenerierte Flächen
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    # Fallback: Verwende normale X-Y-Ebene
                    print(f"[DEBUG Vertical Grid] ⚠️ Degenerierte X-Z-Wand, verwende Fallback")
                    if not geometry.bbox:
                        width = getattr(self.settings, 'width', 150.0)
                        length = getattr(self.settings, 'length', 100.0)
                        min_x, max_x = -width / 2, width / 2
                        min_y, max_y = -length / 2, length / 2
                    else:
                        min_x, max_x, min_y, max_y = geometry.bbox
                    width = max_x - min_x
                    height = max_y - min_y
                    nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                    ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                    total_points_base = nx_base * ny_base
                    min_total_points = min_points_per_dimension ** 2
                    if total_points_base < min_total_points:
                        diagonal = np.sqrt(width**2 + height**2)
                        if diagonal > 0:
                            adaptive_resolution = diagonal / min_points_per_dimension
                            adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                            resolution = adaptive_resolution
                    min_x -= resolution
                    max_x += resolution
                    min_y -= resolution
                    max_y += resolution
                    sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                    sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                    if len(sound_field_x) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_x)
                        step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                        if step_x <= 0:
                            step_x = resolution
                        additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                        sound_field_x = np.concatenate([sound_field_x, additional_x])
                    if len(sound_field_y) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_y)
                        step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                        if step_y <= 0:
                            step_y = resolution
                        additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                        sound_field_y = np.concatenate([sound_field_y, additional_y])
                    X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                    Z_grid = np.zeros_like(X_grid, dtype=float)
                else:
                    # 🎯 ALTE LOGIK: Keine Grid-Erweiterung für vertikale Flächen
                    # Erstelle Grid direkt in (u,v)-Ebene OHNE Padding (wie in _build_vertical_surface_samples)
                    u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                    v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                    
                    if u_axis.size < 2 or v_axis.size < 2:
                        # Fallback: Verwende normale X-Y-Ebene
                        print(f"[DEBUG Vertical Grid] ⚠️ Zu wenige Punkte in (u,v)-Ebene, verwende Fallback")
                        if not geometry.bbox:
                            width = getattr(self.settings, 'width', 150.0)
                            length = getattr(self.settings, 'length', 100.0)
                            min_x, max_x = -width / 2, width / 2
                            min_y, max_y = -length / 2, length / 2
                        else:
                            min_x, max_x, min_y, max_y = geometry.bbox
                        width = max_x - min_x
                        height = max_y - min_y
                        nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                        ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                        total_points_base = nx_base * ny_base
                        min_total_points = min_points_per_dimension ** 2
                        if total_points_base < min_total_points:
                            diagonal = np.sqrt(width**2 + height**2)
                            if diagonal > 0:
                                adaptive_resolution = diagonal / min_points_per_dimension
                                adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                                resolution = adaptive_resolution
                        min_x -= resolution
                        max_x += resolution
                        min_y -= resolution
                        max_y += resolution
                        sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                        sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                        if len(sound_field_x) < min_points_per_dimension:
                            n_points_needed = min_points_per_dimension - len(sound_field_x)
                            step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                            if step_x <= 0:
                                step_x = resolution
                            additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                            sound_field_x = np.concatenate([sound_field_x, additional_x])
                        if len(sound_field_y) < min_points_per_dimension:
                            n_points_needed = min_points_per_dimension - len(sound_field_y)
                            step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                            if step_y <= 0:
                                step_y = resolution
                            additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                            sound_field_y = np.concatenate([sound_field_y, additional_y])
                        X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                        Z_grid = np.zeros_like(X_grid, dtype=float)
                    else:
                        # Erstelle 2D-Meshgrid in (u,v)-Ebene
                        U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                        
                        # Transformiere zu (X, Y, Z) Koordinaten
                        X_grid = U_grid  # u = x
                        Y_grid = np.full_like(U_grid, y0, dtype=float)  # y = konstant
                        Z_grid = V_grid  # v = z
                        
                        # sound_field_x und sound_field_y für Rückgabe (werden für Plot verwendet)
                        sound_field_x = u_axis  # x-Koordinaten
                        sound_field_y = v_axis  # z-Koordinaten (als y für sound_field_y)
                        
                        print(f"[DEBUG Vertical Grid] X-Z-Wand: Grid in (x,z)-Ebene erstellt")
                        print(f"  └─ u_axis (x): {len(u_axis)} Punkte, min={u_min:.3f}, max={u_max:.3f}")
                        print(f"  └─ v_axis (z): {len(v_axis)} Punkte, min={v_min:.3f}, max={v_max:.3f}")
                        print(f"  └─ wall_y (konstant): {y0:.3f}")
                        print(f"  └─ step (refined): {step:.3f} m (base: {base_resolution:.3f} m)")
                
            elif x_span < eps_line and y_span >= eps_line:
                # Y-Z-Wand: x ≈ const, Grid in (y,z)-Ebene
                # u = y, v = z
                x0 = float(np.mean(xs))
                u_min, u_max = float(ys.min()), float(ys.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                
                # Prüfe auf degenerierte Flächen
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    # Fallback: Verwende normale X-Y-Ebene
                    print(f"[DEBUG Vertical Grid] ⚠️ Degenerierte Y-Z-Wand, verwende Fallback")
                    if not geometry.bbox:
                        width = getattr(self.settings, 'width', 150.0)
                        length = getattr(self.settings, 'length', 100.0)
                        min_x, max_x = -width / 2, width / 2
                        min_y, max_y = -length / 2, length / 2
                    else:
                        min_x, max_x, min_y, max_y = geometry.bbox
                    width = max_x - min_x
                    height = max_y - min_y
                    nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                    ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                    total_points_base = nx_base * ny_base
                    min_total_points = min_points_per_dimension ** 2
                    if total_points_base < min_total_points:
                        diagonal = np.sqrt(width**2 + height**2)
                        if diagonal > 0:
                            adaptive_resolution = diagonal / min_points_per_dimension
                            adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                            resolution = adaptive_resolution
                    min_x -= resolution
                    max_x += resolution
                    min_y -= resolution
                    max_y += resolution
                    sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                    sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                    if len(sound_field_x) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_x)
                        step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                        if step_x <= 0:
                            step_x = resolution
                        additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                        sound_field_x = np.concatenate([sound_field_x, additional_x])
                    if len(sound_field_y) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_y)
                        step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                        if step_y <= 0:
                            step_y = resolution
                        additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                        sound_field_y = np.concatenate([sound_field_y, additional_y])
                    X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                    Z_grid = np.zeros_like(X_grid, dtype=float)
                else:
                    # 🎯 ALTE LOGIK: Keine Grid-Erweiterung für vertikale Flächen
                    # Erstelle Grid direkt in (u,v)-Ebene OHNE Padding (wie in _build_vertical_surface_samples)
                    u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                    v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                    
                    if u_axis.size < 2 or v_axis.size < 2:
                        # Fallback: Verwende normale X-Y-Ebene
                        print(f"[DEBUG Vertical Grid] ⚠️ Zu wenige Punkte in (u,v)-Ebene, verwende Fallback")
                        if not geometry.bbox:
                            width = getattr(self.settings, 'width', 150.0)
                            length = getattr(self.settings, 'length', 100.0)
                            min_x, max_x = -width / 2, width / 2
                            min_y, max_y = -length / 2, length / 2
                        else:
                            min_x, max_x, min_y, max_y = geometry.bbox
                        width = max_x - min_x
                        height = max_y - min_y
                        nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                        ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                        total_points_base = nx_base * ny_base
                        min_total_points = min_points_per_dimension ** 2
                        if total_points_base < min_total_points:
                            diagonal = np.sqrt(width**2 + height**2)
                            if diagonal > 0:
                                adaptive_resolution = diagonal / min_points_per_dimension
                                adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                                resolution = adaptive_resolution
                        min_x -= resolution
                        max_x += resolution
                        min_y -= resolution
                        max_y += resolution
                        sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                        sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                        if len(sound_field_x) < min_points_per_dimension:
                            n_points_needed = min_points_per_dimension - len(sound_field_x)
                            step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                            if step_x <= 0:
                                step_x = resolution
                            additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                            sound_field_x = np.concatenate([sound_field_x, additional_x])
                        if len(sound_field_y) < min_points_per_dimension:
                            n_points_needed = min_points_per_dimension - len(sound_field_y)
                            step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                            if step_y <= 0:
                                step_y = resolution
                            additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                            sound_field_y = np.concatenate([sound_field_y, additional_y])
                        X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                        Z_grid = np.zeros_like(X_grid, dtype=float)
                    else:
                        # Erstelle 2D-Meshgrid in (u,v)-Ebene
                        U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                        
                        # Transformiere zu (X, Y, Z) Koordinaten
                        X_grid = np.full_like(U_grid, x0, dtype=float)  # x = konstant
                        Y_grid = U_grid  # u = y
                        Z_grid = V_grid  # v = z
                        
                        # sound_field_x und sound_field_y für Rückgabe (werden für Plot verwendet)
                        sound_field_x = u_axis  # y-Koordinaten (als x für sound_field_x)
                        sound_field_y = v_axis  # z-Koordinaten (als y für sound_field_y)
                        
                        print(f"[DEBUG Vertical Grid] Y-Z-Wand: Grid in (y,z)-Ebene erstellt")
                        print(f"  └─ u_axis (y): {len(u_axis)} Punkte, min={u_min:.3f}, max={u_max:.3f}")
                        print(f"  └─ v_axis (z): {len(v_axis)} Punkte, min={v_min:.3f}, max={v_max:.3f}")
                        print(f"  └─ wall_x (konstant): {x0:.3f}")
                        print(f"  └─ step (refined): {step:.3f} m (base: {base_resolution:.3f} m)")
                
            else:
                # Prüfe, ob es eine schräge vertikale Fläche ist (z_span > max(x_span, y_span) * 0.5)
                if z_span > max(x_span, y_span) * 0.5 and z_span > 1e-3:
                    # Schräge vertikale Fläche: Bestimme dominante Orientierung
                    if x_span < y_span:
                        # Y-Z-Wand schräg: Y und Z variieren, X variiert entlang der Fläche
                        # u = y, v = z
                        u_min, u_max = float(ys.min()), float(ys.max())
                        v_min, v_max = float(zs.min()), float(zs.max())
                        # X wird später interpoliert
                        from scipy.interpolate import griddata
                        
                        # Erstelle Grid in (u,v)-Ebene = (y,z)-Ebene
                        u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                        v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                        
                        if u_axis.size < 2 or v_axis.size < 2:
                            # Fallback
                            print(f"[DEBUG Vertical Grid] ⚠️ Zu wenige Punkte in (y,z)-Ebene, verwende Fallback")
                            if not geometry.bbox:
                                width = getattr(self.settings, 'width', 150.0)
                                length = getattr(self.settings, 'length', 100.0)
                                min_x, max_x = -width / 2, width / 2
                                min_y, max_y = -length / 2, length / 2
                            else:
                                min_x, max_x, min_y, max_y = geometry.bbox
                            width = max_x - min_x
                            height = max_y - min_y
                            nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                            ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                            total_points_base = nx_base * ny_base
                            min_total_points = min_points_per_dimension ** 2
                            if total_points_base < min_total_points:
                                diagonal = np.sqrt(width**2 + height**2)
                                if diagonal > 0:
                                    adaptive_resolution = diagonal / min_points_per_dimension
                                    adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                                    resolution = adaptive_resolution
                            min_x -= resolution
                            max_x += resolution
                            min_y -= resolution
                            max_y += resolution
                            sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                            sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                            if len(sound_field_x) < min_points_per_dimension:
                                n_points_needed = min_points_per_dimension - len(sound_field_x)
                                step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                                if step_x <= 0:
                                    step_x = resolution
                                additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                                sound_field_x = np.concatenate([sound_field_x, additional_x])
                            if len(sound_field_y) < min_points_per_dimension:
                                n_points_needed = min_points_per_dimension - len(sound_field_y)
                                step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                                if step_y <= 0:
                                    step_y = resolution
                                additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                                sound_field_y = np.concatenate([sound_field_y, additional_y])
                            X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                            Z_grid = np.zeros_like(X_grid, dtype=float)
                        else:
                            # Erstelle 2D-Meshgrid in (u,v)-Ebene = (y,z)-Ebene
                            U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                            
                            # Interpoliere X-Koordinaten von Surface-Punkten auf (y,z)-Grid
                            points_surface = np.column_stack([ys, zs])
                            points_grid = np.column_stack([U_grid.ravel(), V_grid.ravel()])
                            X_interp = griddata(
                                points_surface, xs,
                                points_grid,
                                method='linear', fill_value=float(np.mean(xs))
                            )
                            X_interp = X_interp.reshape(U_grid.shape)
                            
                            # Transformiere zu (X, Y, Z) Koordinaten
                            X_grid = X_interp  # X interpoliert
                            Y_grid = U_grid  # u = y
                            Z_grid = V_grid  # v = z
                            
                            # sound_field_x und sound_field_y für Rückgabe
                            sound_field_x = u_axis  # y-Koordinaten
                            sound_field_y = v_axis  # z-Koordinaten
                            
                            print(f"[DEBUG Vertical Grid] Y-Z-Wand schräg: Grid in (y,z)-Ebene erstellt, X interpoliert")
                            print(f"  └─ u_axis (y): {len(u_axis)} Punkte, min={u_min:.3f}, max={u_max:.3f}")
                            print(f"  └─ v_axis (z): {len(v_axis)} Punkte, min={v_min:.3f}, max={v_max:.3f}")
                            print(f"  └─ X interpoliert: min={X_interp.min():.3f}, max={X_interp.max():.3f}")
                    else:
                        # X-Z-Wand schräg: X und Z variieren, Y variiert entlang der Fläche
                        # u = x, v = z
                        u_min, u_max = float(xs.min()), float(xs.max())
                        v_min, v_max = float(zs.min()), float(zs.max())
                        # Y wird später interpoliert
                        from scipy.interpolate import griddata
                        
                        # Erstelle Grid in (u,v)-Ebene = (x,z)-Ebene
                        u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                        v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                        
                        if u_axis.size < 2 or v_axis.size < 2:
                            # Fallback
                            print(f"[DEBUG Vertical Grid] ⚠️ Zu wenige Punkte in (x,z)-Ebene, verwende Fallback")
                            if not geometry.bbox:
                                width = getattr(self.settings, 'width', 150.0)
                                length = getattr(self.settings, 'length', 100.0)
                                min_x, max_x = -width / 2, width / 2
                                min_y, max_y = -length / 2, length / 2
                            else:
                                min_x, max_x, min_y, max_y = geometry.bbox
                            width = max_x - min_x
                            height = max_y - min_y
                            nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                            ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                            total_points_base = nx_base * ny_base
                            min_total_points = min_points_per_dimension ** 2
                            if total_points_base < min_total_points:
                                diagonal = np.sqrt(width**2 + height**2)
                                if diagonal > 0:
                                    adaptive_resolution = diagonal / min_points_per_dimension
                                    adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                                    resolution = adaptive_resolution
                            min_x -= resolution
                            max_x += resolution
                            min_y -= resolution
                            max_y += resolution
                            sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                            sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                            if len(sound_field_x) < min_points_per_dimension:
                                n_points_needed = min_points_per_dimension - len(sound_field_x)
                                step_x = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                                if step_x <= 0:
                                    step_x = resolution
                                additional_x = np.arange(max_x + step_x, max_x + step_x * n_points_needed, step_x)
                                sound_field_x = np.concatenate([sound_field_x, additional_x])
                            if len(sound_field_y) < min_points_per_dimension:
                                n_points_needed = min_points_per_dimension - len(sound_field_y)
                                step_y = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                                if step_y <= 0:
                                    step_y = resolution
                                additional_y = np.arange(max_y + step_y, max_y + step_y * n_points_needed, step_y)
                                sound_field_y = np.concatenate([sound_field_y, additional_y])
                            X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                            Z_grid = np.zeros_like(X_grid, dtype=float)
                        else:
                            # Erstelle 2D-Meshgrid in (u,v)-Ebene = (x,z)-Ebene
                            U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                            
                            # Interpoliere Y-Koordinaten von Surface-Punkten auf (x,z)-Grid
                            points_surface = np.column_stack([xs, zs])
                            points_grid = np.column_stack([U_grid.ravel(), V_grid.ravel()])
                            Y_interp = griddata(
                                points_surface, ys,
                                points_grid,
                                method='linear', fill_value=float(np.mean(ys))
                            )
                            Y_interp = Y_interp.reshape(U_grid.shape)
                            
                            # Transformiere zu (X, Y, Z) Koordinaten
                            X_grid = U_grid  # u = x
                            Y_grid = Y_interp  # Y interpoliert
                            Z_grid = V_grid  # v = z
                            
                            # sound_field_x und sound_field_y für Rückgabe
                            sound_field_x = u_axis  # x-Koordinaten
                            sound_field_y = v_axis  # z-Koordinaten
                            
                            print(f"[DEBUG Vertical Grid] X-Z-Wand schräg: Grid in (x,z)-Ebene erstellt, Y interpoliert")
                            print(f"  └─ u_axis (x): {len(u_axis)} Punkte, min={u_min:.3f}, max={u_max:.3f}")
                            print(f"  └─ v_axis (z): {len(v_axis)} Punkte, min={v_min:.3f}, max={v_max:.3f}")
                            print(f"  └─ Y interpoliert: min={Y_interp.min():.3f}, max={Y_interp.max():.3f}")
                else:
                    # Fallback: Verwende normale X-Y-Ebene
                    print(f"[DEBUG Vertical Grid] ⚠️ Keine klare Orientierung, verwende X-Y-Ebene (Fallback)")
                    if not geometry.bbox:
                        width = getattr(self.settings, 'width', 150.0)
                        length = getattr(self.settings, 'length', 100.0)
                        min_x, max_x = -width / 2, width / 2
                        min_y, max_y = -length / 2, length / 2
                    else:
                        min_x, max_x, min_y, max_y = geometry.bbox
                    
                    width = max_x - min_x
                    height = max_y - min_y
                    nx_base = max(1, int(np.ceil(width / resolution)) + 1)
                    ny_base = max(1, int(np.ceil(height / resolution)) + 1)
                    total_points_base = nx_base * ny_base
                    min_total_points = min_points_per_dimension ** 2
                    
                    if total_points_base < min_total_points:
                        diagonal = np.sqrt(width**2 + height**2)
                        if diagonal > 0:
                            adaptive_resolution = diagonal / min_points_per_dimension
                            adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                            resolution = adaptive_resolution
                    
                    min_x -= resolution
                    max_x += resolution
                    min_y -= resolution
                    max_y += resolution
                    
                    sound_field_x = np.arange(min_x, max_x + resolution, resolution)
                    sound_field_y = np.arange(min_y, max_y + resolution, resolution)
                    
                    if len(sound_field_x) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_x)
                        step = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                        if step <= 0:
                            step = resolution
                        additional_x = np.arange(max_x + step, max_x + step * n_points_needed, step)
                        sound_field_x = np.concatenate([sound_field_x, additional_x])
                    
                    if len(sound_field_y) < min_points_per_dimension:
                        n_points_needed = min_points_per_dimension - len(sound_field_y)
                        step = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                        if step <= 0:
                            step = resolution
                        additional_y = np.arange(max_y + step, max_y + step * n_points_needed, step)
                        sound_field_y = np.concatenate([sound_field_y, additional_y])
                    
                    X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                    Z_grid = np.zeros_like(X_grid, dtype=float)
        else:
            # PLANARE/SCHRÄGE SURFACES: Grid in X-Y-Ebene (wie bisher)
            if not geometry.bbox:
                # Fallback: Verwende Settings-Dimensionen
                width = getattr(self.settings, 'width', 150.0)
                length = getattr(self.settings, 'length', 100.0)
                min_x, max_x = -width / 2, width / 2
                min_y, max_y = -length / 2, length / 2
            else:
                min_x, max_x, min_y, max_y = geometry.bbox
            
            # Berechne Surface-Dimensionen
            width = max_x - min_x
            height = max_y - min_y
            
            # Berechne Anzahl Punkte bei Basis-Resolution
            nx_base = max(1, int(np.ceil(width / resolution)) + 1)
            ny_base = max(1, int(np.ceil(height / resolution)) + 1)
            total_points_base = nx_base * ny_base
            
            # 🎯 Adaptive Resolution: Prüfe Mindestanzahl von Punkten
            min_total_points = min_points_per_dimension ** 2  # 3×3 = 9 Punkte
            
            if total_points_base < min_total_points:
                # Zu wenige Punkte → Passe Resolution an
                # Berechne benötigte Resolution für mindestens min_points_per_dimension Punkte pro Dimension
                diagonal = np.sqrt(width**2 + height**2)
                if diagonal > 0:
                    # Resolution so anpassen, dass mindestens min_points_per_dimension Punkte vorhanden sind
                    # Beispiel: diagonal = 10m, min_points = 3 → resolution = 10/3 ≈ 3.33m
                    adaptive_resolution = diagonal / min_points_per_dimension
                    # Begrenze: Nicht kleiner als halbe Basis-Resolution
                    adaptive_resolution = min(adaptive_resolution, resolution * 0.5)
                    
                    print(f"[DEBUG Grid pro Surface] Surface '{geometry.surface_id}': "
                          f"Zu wenige Punkte ({total_points_base} < {min_total_points})")
                    print(f"  └─ Adaptive Resolution: {adaptive_resolution:.3f} m (Basis: {resolution:.3f} m)")
                    
                    resolution = adaptive_resolution
            
            # 🎯 ERWEITERE GRID: Um genau einen Punkt (resolution) über den Rand hinaus
            min_x -= resolution
            max_x += resolution
            min_y -= resolution
            max_y += resolution
            
            # Erstelle 1D-Koordinaten-Arrays
            sound_field_x = np.arange(min_x, max_x + resolution, resolution)
            sound_field_y = np.arange(min_y, max_y + resolution, resolution)
            
            # Sicherstellen: Mindestens min_points_per_dimension Punkte
            if len(sound_field_x) < min_points_per_dimension:
                # Zu wenig Punkte → Erweitere Array
                n_points_needed = min_points_per_dimension - len(sound_field_x)
                step = resolution if len(sound_field_x) > 1 else (max_x - min_x) / (min_points_per_dimension - 1)
                if step <= 0:
                    step = resolution
                additional_x = np.arange(max_x + step, max_x + step * n_points_needed, step)
                sound_field_x = np.concatenate([sound_field_x, additional_x])
            
            if len(sound_field_y) < min_points_per_dimension:
                n_points_needed = min_points_per_dimension - len(sound_field_y)
                step = resolution if len(sound_field_y) > 1 else (max_y - min_y) / (min_points_per_dimension - 1)
                if step <= 0:
                    step = resolution
                additional_y = np.arange(max_y + step, max_y + step * n_points_needed, step)
                sound_field_y = np.concatenate([sound_field_y, additional_y])
            
            # Erstelle 2D-Meshgrid
            X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
            Z_grid = np.zeros_like(X_grid, dtype=float)
        
        ny, nx = X_grid.shape
        
        # 🎯 VERTIKALE SURFACES: Erstelle Maske direkt in (u,v)-Koordinaten
        if geometry.orientation == "vertical":
            # Bestimme Orientierung und erstelle Maske in (u,v)-Koordinaten
            points = geometry.points
            xs = np.array([p.get('x', 0.0) for p in points], dtype=float)
            ys = np.array([p.get('y', 0.0) for p in points], dtype=float)
            zs = np.array([p.get('z', 0.0) for p in points], dtype=float)
            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            
            eps_line = 1e-6
            
            if y_span < eps_line and x_span >= eps_line:
                # X-Z-Wand: Maske in (x,z)-Ebene
                # 🎯 ALTE LOGIK: Verwende _points_in_polygon_batch (funktioniert auch für (u,v))
                U_grid = X_grid  # u = x
                V_grid = Z_grid  # v = z
                polygon_uv = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
                surface_mask = self._points_in_polygon_batch(U_grid, V_grid, polygon_uv)
                # 🎯 ALTE LOGIK: Leichte Dilatation der Maske (wie in _build_vertical_surface_samples)
                surface_mask = self._dilate_mask_minimal(surface_mask)
                print(f"[DEBUG Vertical Mask] X-Z-Wand: Maske in (x,z)-Ebene erstellt")
            elif x_span < eps_line and y_span >= eps_line:
                # Y-Z-Wand: Maske in (y,z)-Ebene
                # 🎯 ALTE LOGIK: Verwende _points_in_polygon_batch (funktioniert auch für (u,v))
                U_grid = Y_grid  # u = y
                V_grid = Z_grid  # v = z
                polygon_uv = [
                    {"x": float(p.get("y", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
                surface_mask = self._points_in_polygon_batch(U_grid, V_grid, polygon_uv)
                # 🎯 ALTE LOGIK: Leichte Dilatation der Maske (wie in _build_vertical_surface_samples)
                surface_mask = self._dilate_mask_minimal(surface_mask)
                print(f"[DEBUG Vertical Mask] Y-Z-Wand: Maske in (y,z)-Ebene erstellt")
            else:
                # Prüfe, ob es eine schräge vertikale Fläche ist
                z_span = float(np.ptp(zs))
                if z_span > max(x_span, y_span) * 0.5 and z_span > 1e-3:
                    # Schräge vertikale Fläche: Bestimme dominante Orientierung
                    if x_span < y_span:
                        # Y-Z-Wand schräg: Maske in (y,z)-Ebene
                        U_grid = Y_grid  # u = y
                        V_grid = Z_grid  # v = z
                        polygon_uv = [
                            {"x": float(p.get("y", 0.0)), "y": float(p.get("z", 0.0))}
                            for p in points
                        ]
                        surface_mask = self._points_in_polygon_batch(U_grid, V_grid, polygon_uv)
                        surface_mask = self._dilate_mask_minimal(surface_mask)
                        print(f"[DEBUG Vertical Mask] Y-Z-Wand schräg: Maske in (y,z)-Ebene erstellt")
                    else:
                        # X-Z-Wand schräg: Maske in (x,z)-Ebene
                        U_grid = X_grid  # u = x
                        V_grid = Z_grid  # v = z
                        polygon_uv = [
                            {"x": float(p.get("x", 0.0)), "y": float(p.get("z", 0.0))}
                            for p in points
                        ]
                        surface_mask = self._points_in_polygon_batch(U_grid, V_grid, polygon_uv)
                        surface_mask = self._dilate_mask_minimal(surface_mask)
                        print(f"[DEBUG Vertical Mask] X-Z-Wand schräg: Maske in (x,z)-Ebene erstellt")
                else:
                    # Fallback: Verwende normale Maske
                    surface_mask = self._create_surface_mask(X_grid, Y_grid, geometry)
                    print(f"[DEBUG Vertical Mask] ⚠️ Fallback: Maske in X-Y-Ebene erstellt")
            
            # Z_grid ist bereits korrekt gesetzt (für X-Z-Wand: Z_grid = V_grid, für Y-Z-Wand: Z_grid = V_grid)
            # Keine Z-Interpolation nötig!
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
            print(f"[DEBUG Grid-Erweiterung] Surface '{geometry.surface_id}' (VERTIKAL):")
            print(f"  └─ Total Grid-Punkte: {total_grid_points}")
            print(f"  └─ Punkte IN Surface: {points_in_surface}")
            print(f"  └─ Punkte AUSSERHALB Surface (erweitert): {points_outside_surface}")
            print(f"  └─ Z_grid bereits korrekt gesetzt (keine Interpolation nötig)")
        else:
            # PLANARE/SCHRÄGE SURFACES: Normale Maske und Z-Interpolation
            surface_mask = self._create_surface_mask(X_grid, Y_grid, geometry)
            
            # 🎯 DEBUG: Grid-Erweiterung (vor Z-Interpolation, damit total_grid_points verfügbar ist)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
            print(f"[DEBUG Grid-Erweiterung] Surface '{geometry.surface_id}':")
            print(f"  └─ Total Grid-Punkte: {total_grid_points}")
            print(f"  └─ Punkte IN Surface: {points_in_surface}")
            print(f"  └─ Punkte AUSSERHALB Surface (erweitert): {points_outside_surface}")
            
            # 🎯 Z-INTERPOLATION: Für alle Punkte im Grid (auch außerhalb Surface)
            # Z-Werte linear interpolieren gemäß Plane-Model für erweiterte Randpunkte
            if Z_grid is None or np.all(Z_grid == 0):
                Z_grid = np.zeros_like(X_grid, dtype=float)
            
            if geometry.plane_model:
                # Berechne Z-Werte für ALLE Punkte im Grid (linear interpoliert gemäß Plane-Model)
                # Dies ermöglicht erweiterte Randpunkte außerhalb der Surface-Grenze
                Z_values_all = _evaluate_plane_on_grid(geometry.plane_model, X_grid, Y_grid)
                Z_grid = Z_values_all  # Setze für alle Punkte, nicht nur innerhalb Surface
                print(f"  └─ Z-Werte berechnet für ALLE {total_grid_points} Punkte (auch außerhalb Surface)")
            else:
                # 🎯 FALLBACK: Lineare Interpolation basierend auf Surface Z-Werten
                # Wenn kein Plane-Model vorhanden ist, interpoliere Z-Werte von Surface-Punkten
                if points_in_surface > 0:
                    # Extrahiere Z-Werte der Surface-Punkte (falls vorhanden)
                    surface_points = geometry.points
                    if surface_points and len(surface_points) >= 3:
                        # Extrahiere Z-Koordinaten aus Surface-Punkten
                        surface_z = np.array([p.get('z', 0.0) for p in surface_points], dtype=float)
                        surface_x = np.array([p.get('x', 0.0) for p in surface_points], dtype=float)
                        surface_y = np.array([p.get('y', 0.0) for p in surface_points], dtype=float)
                    
                        # Prüfe ob Z-Werte variieren
                        z_variation = np.any(np.abs(surface_z - surface_z[0]) > 1e-6)
                        # 🎯 DEBUG: Immer für vertikale Surfaces
                        if geometry.orientation == "vertical":
                            print(f"[DEBUG Z-Interpolation] Surface '{geometry.surface_id}' (VERTIKAL):")
                            print(f"  └─ Surface-Punkte: {len(surface_points)}")
                            print(f"  └─ Z-Werte aus Punkten: min={surface_z.min():.3f}, max={surface_z.max():.3f}, span={surface_z.max()-surface_z.min():.3f}")
                            print(f"  └─ Z-Variation: {z_variation}")
                            print(f"  └─ X-Werte: min={surface_x.min():.3f}, max={surface_x.max():.3f}, span={surface_x.max()-surface_x.min():.3f}")
                            print(f"  └─ Y-Werte: min={surface_y.min():.3f}, max={surface_y.max():.3f}, span={surface_y.max()-surface_y.min():.3f}")
                            if len(surface_points) <= 10:
                                print(f"  └─ Alle Punkte: {[(p.get('x', 0), p.get('y', 0), p.get('z', 0)) for p in surface_points[:10]]}")
                        elif DEBUG_FLEXIBLE_GRID:
                            print(f"[DEBUG Z-Interpolation] Surface '{geometry.surface_id}':")
                            print(f"  └─ Surface-Punkte: {len(surface_points)}")
                            print(f"  └─ Z-Werte: min={surface_z.min():.3f}, max={surface_z.max():.3f}, span={surface_z.max()-surface_z.min():.3f}")
                            print(f"  └─ Z-Variation: {z_variation}")
                            print(f"  └─ X-Werte: min={surface_x.min():.3f}, max={surface_x.max():.3f}, span={surface_x.max()-surface_x.min():.3f}")
                            print(f"  └─ Y-Werte: min={surface_y.min():.3f}, max={surface_y.max():.3f}, span={surface_y.max()-surface_y.min():.3f}")
                        
                        if z_variation:
                            # 🎯 VERTIKALE SURFACES: Z_grid ist bereits korrekt gesetzt, überspringe Interpolation
                            if geometry.orientation == "vertical":
                                # Z_grid wurde bereits beim Grid-Erstellen korrekt gesetzt (Z_grid = V_grid)
                                # Keine Interpolation nötig!
                                print(f"  └─ Z_grid bereits korrekt gesetzt für vertikale Surface (keine Interpolation nötig)")
                            elif geometry.orientation == "planar" or geometry.orientation == "sloped":
                                # Für planare/schräge Surfaces: Z-Werte hängen von beiden Koordinaten ab (u, v)
                                xs = surface_x
                                ys = surface_y
                                x_span = float(np.ptp(xs))
                                y_span = float(np.ptp(ys))
                                
                                eps_line = 1e-6
                                from scipy.interpolate import griddata
                                
                                if y_span < eps_line and x_span >= eps_line:
                                    # X-Z-Wand: y ≈ const, Z hängt von x ab
                                    # Verwende 1D-Interpolation basierend auf x
                                    try:
                                        from scipy.interpolate import interp1d
                                        
                                        # Sammle alle eindeutigen x-Werte und deren zugehörige Z-Werte
                                        x_z_dict = {}
                                        for x_val, z_val in zip(surface_x, surface_z):
                                            x_key = float(x_val)
                                            if x_key not in x_z_dict:
                                                x_z_dict[x_key] = []
                                            x_z_dict[x_key].append(float(z_val))
                                        
                                        # Für jeden x-Wert: Verwende den Mittelwert der Z-Werte
                                        x_unique = np.array(sorted(x_z_dict.keys()))
                                        z_unique = np.array([np.mean(x_z_dict[x]) for x in x_unique])
                                        
                                        if len(x_unique) < 2:
                                            # Nicht genug eindeutige Punkte
                                            z_mean = float(np.mean(surface_z))
                                            Z_grid.fill(z_mean)
                                            print(f"  └─ ⚠️ Zu wenige eindeutige x-Werte ({len(x_unique)}), verwende konstanten Z-Wert {z_mean:.3f}")
                                        else:
                                            # 1D-Interpolation: Z(x)
                                            interp_func = interp1d(
                                                x_unique,
                                                z_unique,
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value=(z_unique[0], z_unique[-1])
                                            )
                                            
                                            # Interpoliere für alle Grid-Punkte
                                            x_grid_flat = X_grid.ravel()
                                            z_interp_flat = interp_func(x_grid_flat)
                                            Z_grid = z_interp_flat.reshape(X_grid.shape)
                                            
                                            print(f"  └─ Z-Werte (1D linear, gemittelt) interpoliert für ALLE {total_grid_points} Punkte (X-Z-Wand)")
                                            print(f"  └─ Eindeutige x-Werte: {len(x_unique)} (von {len(surface_x)} Original-Punkten)")
                                            print(f"  └─ Ergebnis: Z_grid min={Z_grid.min():.3f}, max={Z_grid.max():.3f}, span={Z_grid.max()-Z_grid.min():.3f}")
                                    except Exception as e:
                                        z_mean = float(np.mean(surface_z))
                                        Z_grid.fill(z_mean)
                                        print(f"  └─ ⚠️ Z-Interpolation fehlgeschlagen, verwende konstanten Z-Wert {z_mean:.3f}: {e}")
                                    
                                elif x_span < eps_line and y_span >= eps_line:
                                    # Y-Z-Wand: x ≈ const, Z hängt von y ab
                                    # Problem: Es gibt mehrere Z-Werte für denselben y-Wert
                                    # Lösung: Verwende 1D-Interpolation basierend auf y, aber berücksichtige alle Z-Werte
                                    # Für identische y-Werte: Verwende den Mittelwert oder den Bereich
                                    try:
                                        from scipy.interpolate import interp1d
                                        
                                        # Sammle alle eindeutigen y-Werte und deren zugehörige Z-Werte
                                        y_z_dict = {}
                                        for y_val, z_val in zip(surface_y, surface_z):
                                            y_key = float(y_val)
                                            if y_key not in y_z_dict:
                                                y_z_dict[y_key] = []
                                            y_z_dict[y_key].append(float(z_val))
                                        
                                        # Für jeden y-Wert: Verwende den Mittelwert der Z-Werte
                                        y_unique = np.array(sorted(y_z_dict.keys()))
                                        z_unique = np.array([np.mean(y_z_dict[y]) for y in y_unique])
                                        
                                        if len(y_unique) < 2:
                                            # Nicht genug eindeutige Punkte
                                            z_mean = float(np.mean(surface_z))
                                            Z_grid.fill(z_mean)
                                            print(f"  └─ ⚠️ Zu wenige eindeutige y-Werte ({len(y_unique)}), verwende konstanten Z-Wert {z_mean:.3f}")
                                        else:
                                            # 1D-Interpolation: Z(y)
                                            interp_func = interp1d(
                                                y_unique,
                                                z_unique,
                                                kind='linear',
                                                bounds_error=False,
                                                fill_value=(z_unique[0], z_unique[-1])
                                            )
                                            
                                            # Interpoliere für alle Grid-Punkte
                                            y_grid_flat = Y_grid.ravel()
                                            z_interp_flat = interp_func(y_grid_flat)
                                            Z_grid = z_interp_flat.reshape(X_grid.shape)
                                            
                                            print(f"  └─ Z-Werte (1D linear, gemittelt) interpoliert für ALLE {total_grid_points} Punkte (Y-Z-Wand)")
                                            print(f"  └─ Eindeutige y-Werte: {len(y_unique)} (von {len(surface_y)} Original-Punkten)")
                                            print(f"  └─ Ergebnis: Z_grid min={Z_grid.min():.3f}, max={Z_grid.max():.3f}, span={Z_grid.max()-Z_grid.min():.3f}")
                                    except Exception as e:
                                        z_mean = float(np.mean(surface_z))
                                        Z_grid.fill(z_mean)
                                        print(f"  └─ ⚠️ Z-Interpolation fehlgeschlagen, verwende konstanten Z-Wert {z_mean:.3f}: {e}")
                                else:
                                    # Fallback: Verwende konstanten Z-Wert
                                    z_mean = float(np.mean(surface_z)) if len(surface_z) > 0 else 0.0
                                    Z_grid.fill(z_mean)
                                    print(f"  └─ ⚠️ Keine klare Orientierung: x_span={x_span:.6f}, y_span={y_span:.6f}")
                                    print(f"  └─ Z-Werte auf konstanten Wert {z_mean:.3f} gesetzt (planare/schräge Surface, Fallback)")
                            else:
                                # Normale Surfaces: Lineare Interpolation in (x,y)
                                from scipy.interpolate import griddata
                                points_surface = np.column_stack([surface_x, surface_y])
                                points_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                                
                                try:
                                    Z_interp = griddata(
                                        points_surface,
                                        surface_z,
                                        points_grid,
                                        method='linear',  # Lineare Interpolation
                                        fill_value=0.0   # Fallback für Punkte außerhalb
                                    )
                                    Z_grid = Z_interp.reshape(X_grid.shape)
                                    print(f"  └─ Z-Werte linear interpoliert für ALLE {total_grid_points} Punkte (basierend auf Surface-Punkten)")
                                except Exception as e:
                                    # Fallback: Nearest Neighbor wenn linear fehlschlägt
                                    try:
                                        Z_interp = griddata(
                                            points_surface,
                                            surface_z,
                                            points_grid,
                                            method='nearest',
                                            fill_value=0.0
                                        )
                                        Z_grid = Z_interp.reshape(X_grid.shape)
                                        print(f"  └─ Z-Werte (nearest) interpoliert für ALLE {total_grid_points} Punkte (Fallback nach linear-Fehler)")
                                    except Exception as e2:
                                        # Letzter Fallback: Konstanter Z-Wert
                                        z_mean = float(np.mean(surface_z))
                                        Z_grid.fill(z_mean)
                                        print(f"  └─ ⚠️ Z-Interpolation fehlgeschlagen, verwende konstanten Z-Wert {z_mean:.3f}: {e2}")
                        else:
                            # Alle Surface-Punkte haben gleichen Z-Wert
                            Z_grid.fill(surface_z[0])
                            print(f"  └─ Z-Werte auf konstanten Wert {surface_z[0]:.3f} gesetzt für ALLE {total_grid_points} Punkte")
                    else:
                        print(f"  └─ ⚠️ Kein Plane-Model und keine Surface-Punkte vorhanden, Z-Werte bleiben 0")
                else:
                    print(f"  └─ ⚠️ Kein Plane-Model vorhanden, Z-Werte bleiben 0")
        
        return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask)


class GridTransformer(ABC):
    """Basis-Klasse für Grid-Transformationen"""
    
    @abstractmethod
    def transform(self, base_grid: BaseGrid, geometries: List[SurfaceGeometry]) -> Any:
        """Transformiert Basis-Grid in berechnungsspezifisches Format"""
        pass


class CartesianTransformer(GridTransformer):
    """Transformiert Basis-Grid zu Cartesian Grid für Superposition"""
    
    def __init__(self, settings):
        self.settings = settings
    
    @measure_time("CartesianTransformer.transform")
    def transform(
        self,
        base_grid: BaseGrid,
        geometries: List[SurfaceGeometry]
    ) -> CartesianGrid:
        """
        Transformiert Basis-Grid zu Cartesian Grid.
        
        Returns:
            CartesianGrid kompatibel mit SoundfieldCalculator
        """
        topology = base_grid.topology
        if topology is None:
            raise ValueError("BaseGrid muss Topologie-Informationen enthalten")
        
        X_grid = topology['X_grid']
        Y_grid = topology['Y_grid']
        sound_field_x = topology['sound_field_x']
        sound_field_y = topology['sound_field_y']
        ny, nx = topology['shape']
        
        # Erstelle kombinierte Surface-Maske (Union aller Surfaces)
        surface_mask = np.zeros_like(X_grid, dtype=bool)
        for surface_id, mask_flat in base_grid.surface_masks.items():
            mask_2d = mask_flat.reshape(ny, nx)
            surface_mask = surface_mask | mask_2d
        
        # 🎯 NUR STRICTE MASKE: Keine erweiterte Maske mehr
        # Erweiterte Rand-Punkte (außerhalb Polygon) werden nicht berechnet
        # Später werden zusätzliche Punkte in hoher Auflösung innerhalb der 
        # Surface-Grenze erstellt, die den Surface-Abschluss bilden
        
        # Interpoliere Z-Koordinaten NUR für Punkte innerhalb der Polygon-Outline
        Z_grid = self._interpolate_z_coordinates(
            X_grid, Y_grid, geometries, surface_mask  # Nur strikte Maske
        )
        
        return CartesianGrid(
            sound_field_x=sound_field_x,
            sound_field_y=sound_field_y,
            X_grid=X_grid,
            Y_grid=Y_grid,
            Z_grid=Z_grid,
            surface_mask=surface_mask,  # Nur strikte Maske (keine Erweiterung)
            surface_mask_strict=surface_mask.copy()  # Identisch mit surface_mask
        )
    
    # ⚠️ AUSKOMMENTIERT: Dilatations-Funktionen nicht mehr benötigt
    # Erweiterte Maske wurde entfernt - nur strikte Maske wird verwendet
    # def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
    #     """Erweitert Maske um 1 Pixel (morphologische Dilatation)"""
    #     try:
    #         from scipy import ndimage
    #         structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    #         return ndimage.binary_dilation(mask, structure=structure)
    #     except ImportError:
    #         # Fallback: Manuelle Erweiterung
    #         return self._dilate_mask_manual(mask)
    #
    # def _dilate_mask_manual(self, mask: np.ndarray) -> np.ndarray:
    #     """Manuelle morphologische Dilatation (Fallback)"""
    #     dilated = mask.copy()
    #     ny, nx = mask.shape
    #
    #     for di in [-1, 0, 1]:
    #         for dj in [-1, 0, 1]:
    #             if di == 0 and dj == 0:
    #                 continue
    #             shifted = np.zeros_like(mask)
    #             i_start = max(0, -di)
    #             i_end = min(nx, nx - di)
    #             j_start = max(0, -dj)
    #             j_end = min(ny, ny - dj)
    #
    #             if i_start < i_end and j_start < j_end:
    #                 shifted[j_start:j_end, i_start:i_end] = mask[
    #                     j_start + dj:j_end + dj,
    #                     i_start + di:i_end + di
    #                 ]
    #             dilated = dilated | shifted
    #
    #     return dilated
    
    def _interpolate_z_coordinates(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        geometries: List[SurfaceGeometry],
        surface_mask: np.ndarray
    ) -> np.ndarray:
        """
        Interpoliert Z-Koordinaten NUR für Punkte innerhalb der Polygon-Outline.
        
        Wichtig: 
        - Z-Werte werden nur für Punkte INNERHALB der Surfaces gesetzt
        - Erweiterte Rand-Punkte (außerhalb Polygon, aber in erweiterten Maske) 
          bekommen KEINE Z-Werte (bleiben Z=0)
        - Später werden zusätzliche Punkte in hoher Auflösung innerhalb der 
          Surface-Grenze erstellt, die den Surface-Abschluss bilden
        """
        Z_grid = np.zeros_like(X_grid, dtype=float)
        
        if not geometries:
            return Z_grid
        
        print(f"[DEBUG Z-Interpolation] Starte Z-Interpolation für {len(geometries)} Geometrien")
        
        # Sammle Z-Beiträge von allen planaren/schrägen Surfaces
        z_contributions = {}
        z_counts = np.zeros_like(X_grid, dtype=float)
        
        processed_surfaces = 0
        skipped_vertical = 0
        skipped_no_model = 0
        skipped_no_points = 0
        
        for geom in geometries:
            # Überspringe vertikale Surfaces (werden separat behandelt)
            if geom.orientation == "vertical":
                skipped_vertical += 1
                continue
            
            if geom.plane_model is None:
                skipped_no_model += 1
                continue
            
            # Erstelle Polygon-Maske (nur Punkte INNERHALB des Polygons)
            mask = self._points_in_polygon_batch(X_grid, Y_grid, geom.points)
            
            if not np.any(mask):
                skipped_no_points += 1
                continue
            
            # Berechne Z-Werte für ALLE Punkte im Grid (wird dann nur für Polygon-Punkte verwendet)
            # Verwende vektorisierte Funktion aus SurfaceGeometryCalculator
            Z_surface = _evaluate_plane_on_grid(geom.plane_model, X_grid, Y_grid)
            
            # Debug: Zeige Plane-Model-Parameter
            mode = geom.plane_model.get('mode', 'unknown')
            points_in_polygon = int(np.sum(mask))
            z_values_in_polygon = Z_surface[mask]
            z_min = float(z_values_in_polygon.min()) if len(z_values_in_polygon) > 0 else 0.0
            z_max = float(z_values_in_polygon.max()) if len(z_values_in_polygon) > 0 else 0.0
            z_mean = float(z_values_in_polygon.mean()) if len(z_values_in_polygon) > 0 else 0.0
            z_non_zero = int(np.sum(np.abs(z_values_in_polygon) > 1e-6))
            
            print(f"  [Z-Interp] Surface {geom.surface_id} ({geom.name}): mode={mode}, points_in_polygon={points_in_polygon}")
            print(f"    Z-Werte: min={z_min:.3f}, max={z_max:.3f}, mean={z_mean:.3f}, non-zero={z_non_zero}/{points_in_polygon}")
            if mode == 'y':
                slope = geom.plane_model.get('slope', 0.0)
                intercept = geom.plane_model.get('intercept', 0.0)
                print(f"    Plane-Model: Z = {slope:.6f} * Y + {intercept:.6f}")
                # Test: Berechne Z für einen Beispielpunkt
                if points_in_polygon > 0:
                    test_idx = np.where(mask)[0][0]
                    test_y = Y_grid.flatten()[test_idx]
                    test_z = Z_surface.flatten()[test_idx]
                    expected_z = slope * test_y + intercept
                    print(f"    Test-Punkt: Y={test_y:.2f}, Z={test_z:.3f}, expected={expected_z:.3f}, diff={abs(test_z-expected_z):.3f}")
            
            # Speichere Beitrag (nur für Punkte innerhalb des Polygons)
            z_contributions[geom.surface_id] = (mask, Z_surface)
            z_counts[mask] += 1.0
            processed_surfaces += 1
        
        print(f"[DEBUG Z-Interpolation] Verarbeitet: {processed_surfaces} Surfaces")
        print(f"  └─ Übersprungen: {skipped_vertical} vertikal, {skipped_no_model} ohne Model, {skipped_no_points} ohne Punkte")
        
        # Kombiniere Z-Werte - KEINE Mittelwert-Bildung mehr!
        # 🎯 LOGIK: Jede Surface setzt ihre Z-Werte direkt, ohne Mittelwert
        # Bei Überlappung: Letzte Surface überschreibt (oder: erste Surface hat Priorität)
        # Surfaces mit Z=0 werden ignoriert, da sie keine Z-Information enthalten
        
        Z_grid = np.zeros_like(X_grid, dtype=float)
        
        # Iteriere über alle Surfaces und setze Z-Werte
        # Reihenfolge: Erste Surface hat Priorität, wird aber von späteren überschrieben
        for surface_id, (mask, Z_surface) in z_contributions.items():
            # Setze Z-Werte nur für Punkte innerhalb des Polygons
            # Ignoriere Surfaces mit Z=0 (keine Z-Information)
            z_nonzero_mask = mask & (np.abs(Z_surface) > 1e-6)
            Z_grid[z_nonzero_mask] = Z_surface[z_nonzero_mask]
        
        # Alternative: Wenn Überlappung erkannt werden soll:
        # - Erste Surface mit Z≠0 hat Priorität (aktuell: letzte überschreibt)
        # - Oder: Surface mit höherem Z-Wert hat Priorität
        
        # Debug: Zeige wie viele Punkte überschrieben wurden
        points_overwritten = 0
        if len(z_contributions) > 1:
            # Prüfe, ob Punkte von mehreren Surfaces gesetzt wurden
            for surface_id, (mask, Z_surface) in z_contributions.items():
                z_nonzero_mask = mask & (np.abs(Z_surface) > 1e-6)
                # Prüfe, ob diese Punkte bereits Z-Werte hatten
                already_set = z_nonzero_mask & (np.abs(Z_grid) > 1e-6)
                if np.any(already_set):
                    points_overwritten += int(np.sum(already_set))
        
        if points_overwritten > 0:
            print(f"[DEBUG Z-Interpolation] ⚠️  {points_overwritten} Punkte wurden überschrieben (Überlappung)")
        
        # 🔍 DEBUG: Detaillierte Analyse für halbierte Z-Werte
        points_with_overlap = int(np.sum(z_counts > 1.0))
        if points_with_overlap > 0:
            print(f"[DEBUG Z-Interpolation] ⚠️  Überlappungen gefunden:")
            print(f"  └─ Punkte mit Überlappung: {points_with_overlap} (durchschnittlich {np.mean(z_counts[z_counts > 1.0]):.2f} Surfaces)")
            
            # Analysiere Beispiel-Punkte mit Überlappung
            overlap_mask = z_counts > 1.0
            if np.any(overlap_mask):
                # Finde einige Beispiel-Punkte mit Überlappung
                overlap_indices = np.where(overlap_mask)
                sample_count = min(3, len(overlap_indices[0]))
                sample_indices = np.random.choice(len(overlap_indices[0]), sample_count, replace=False)
                
                for idx in sample_indices:
                    jj, ii = overlap_indices[0][idx], overlap_indices[1][idx]
                    x_val = X_grid[jj, ii]
                    y_val = Y_grid[jj, ii]
                    z_final = Z_grid[jj, ii]
                    n_surfaces = int(z_counts[jj, ii])
                    
                    print(f"  └─ Beispiel [jj={jj},ii={ii}]: X={x_val:.2f}, Y={y_val:.2f}, Z_final={z_final:.3f}, N_Surfaces={n_surfaces}")
                    
                    # Zeige Beiträge von jeder Surface
                    for surface_id, (mask, Z_surface) in z_contributions.items():
                        if mask[jj, ii]:
                            z_contrib = Z_surface[jj, ii]
                            print(f"      └─ {surface_id}: Z_Beitrag={z_contrib:.3f}")
        
        # Debug: Prüfe, ob Z-Werte halbiert werden könnten
        for surface_id, (mask, Z_surface) in z_contributions.items():
            # Prüfe Punkte dieser Surface, die auch in anderen Surfaces liegen
            overlap_for_surface = (mask) & (z_counts > 1.0)
            if np.any(overlap_for_surface):
                sample_points = np.where(overlap_for_surface)
                if len(sample_points[0]) > 0:
                    sample_idx = 0
                    jj, ii = sample_points[0][sample_idx], sample_points[1][sample_idx]
                    x_val = X_grid[jj, ii]
                    y_val = Y_grid[jj, ii]
                    z_from_surface = Z_surface[jj, ii]
                    z_final = Z_grid[jj, ii]
                    n_surfaces = int(z_counts[jj, ii])
                    
                    if abs(z_from_surface) > 1e-6 and n_surfaces > 1:
                        expected_ratio = z_from_surface / n_surfaces
                        if abs(z_final - expected_ratio) < abs(z_final - z_from_surface) * 0.1:
                            print(f"[DEBUG Z-Interpolation] ⚠️  HALBIERUNG VERDACHT:")
                            print(f"  └─ Surface {surface_id}: Z_original={z_from_surface:.3f}, Z_final={z_final:.3f}, N_Surfaces={n_surfaces}")
                            print(f"     └─ Verhältnis: {z_final/z_from_surface:.3f} (erwartet: {1.0/n_surfaces:.3f} bei Mittelwert)")
                        break
        
        # Debug: Statistiken
        valid_mask = np.abs(Z_grid) > 1e-6  # Punkte mit Z≠0
        points_with_z = int(np.sum(valid_mask))
        z_non_zero = points_with_z
        z_range = (float(Z_grid[valid_mask].min()), float(Z_grid[valid_mask].max())) if np.any(valid_mask) else (0.0, 0.0)
        
        print(f"[DEBUG Z-Interpolation] Ergebnis:")
        print(f"  └─ Punkte mit Z-Wert (Z≠0): {points_with_z} ({points_with_z/X_grid.size*100:.1f}%)")
        print(f"  └─ Z-Range: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
        
        return Z_grid
    
    def _points_in_polygon_batch(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polygon_points: List[Dict[str, float]]
    ) -> np.ndarray:
        """Vektorisierte Punkt-im-Polygon-Prüfung"""
        if len(polygon_points) < 3:
            return np.zeros_like(x_coords, dtype=bool)
        
        px = np.array([p.get('x', 0.0) for p in polygon_points])
        py = np.array([p.get('y', 0.0) for p in polygon_points])
        
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        n_points = len(x_flat)
        
        inside = np.zeros(n_points, dtype=bool)
        boundary_eps = 1e-6
        n = len(px)
        
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            
            y_above_edge = (yi > y_flat) != (yj > y_flat)
            intersection_x = (xj - xi) * (y_flat - yi) / (yj - yi + 1e-10) + xi
            intersects = y_above_edge & (x_flat <= intersection_x + boundary_eps)
            
            inside = inside ^ intersects
            
            j = i
        
        return inside.reshape(x_coords.shape)
    
    def _evaluate_plane_grid(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        plane_model: Dict[str, float]
    ) -> np.ndarray:
        """Bewertet planares Modell auf einem 2D-Grid"""
        mode = plane_model.get('mode', 'constant')
        
        if mode == 'x':
            slope = float(plane_model.get('slope', 0.0))
            intercept = float(plane_model.get('intercept', 0.0))
            return slope * x_coords + intercept
        elif mode == 'y':
            slope = float(plane_model.get('slope', 0.0))
            intercept = float(plane_model.get('intercept', 0.0))
            return slope * y_coords + intercept
        elif mode == 'xy':
            slope_x = float(plane_model.get('slope_x', 0.0))
            slope_y = float(plane_model.get('slope_y', 0.0))
            intercept = float(plane_model.get('intercept', 0.0))
            return slope_x * x_coords + slope_y * y_coords + intercept
        else:
            # constant
            base = float(plane_model.get('base', 0.0))
            return np.full_like(x_coords, base, dtype=float)


class FlexibleGridGenerator(ModuleBase):
    """
    Haupt-API für flexible Grid-Generierung (Hybrid-Ansatz)
    
    Unterstützt verschiedene Berechnungsmethoden:
    - 'cartesian': Für Superposition (SPL-Berechnung)
    - 'fem': Für FEM (später)
    - 'fdtd': Für FDTD (später)
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        self.analyzer = SurfaceAnalyzer(settings)
        self.builder = GridBuilder(settings)
        self.transformers = {
            'cartesian': CartesianTransformer(settings),
        }
        # Cache für Basis-Grid
        self._cached_base_grid: Optional[BaseGrid] = None
        self._cached_geometries: Optional[List[SurfaceGeometry]] = None
        self._cache_hash: Optional[str] = None
    
    @measure_time("FlexibleGridGenerator.generate")
    def generate(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        method: str = 'cartesian',
        resolution: Optional[float] = None,
        use_cache: bool = True
    ) -> Any:
        """
        Generiert Grid für spezifische Berechnungsmethode.
        
        Args:
            enabled_surfaces: Liste von (surface_id, surface_definition) Tupeln
            method: Berechnungsmethode ('cartesian', 'fem', 'fdtd')
            resolution: Grid-Auflösung in Metern (optional)
            use_cache: Wenn True, wird gecachtes Basis-Grid verwendet (falls verfügbar)
            
        Returns:
            Abhängig von method:
            - 'cartesian': CartesianGrid
            - 'fem': FEMMesh (später)
            - 'fdtd': FDTDGrid (später)
        """
        if method not in self.transformers:
            raise ValueError(f"Unbekannte Methode: {method}. Verfügbar: {list(self.transformers.keys())}")
        
        # Erstelle Cache-Hash
        cache_hash = self._create_cache_hash(enabled_surfaces, resolution)
        
        # Prüfe Cache
        if use_cache and self._cached_base_grid and self._cache_hash == cache_hash:
            geometries = self._cached_geometries
            base_grid = self._cached_base_grid
        else:
            # Analysiere Surfaces
            geometries = self.analyzer.analyze_surfaces(enabled_surfaces)
            
            # Erstelle Basis-Grid
            base_grid = self.builder.build_base_grid(geometries, resolution)
            
            # Cache speichern
            self._cached_base_grid = base_grid
            self._cached_geometries = geometries
            self._cache_hash = cache_hash
        
        # Transformiere zu spezifischem Format
        transformer = self.transformers[method]
        return transformer.transform(base_grid, geometries)
    
    @measure_time("FlexibleGridGenerator.generate_per_surface")
    def generate_per_surface(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6
    ) -> Dict[str, SurfaceGrid]:
        """
        Erstellt für jede enabled Surface ein eigenes Grid mit Mindestanzahl von Punkten.
        
        Args:
            enabled_surfaces: Liste von (surface_id, surface_definition) Tupeln
            resolution: Basis-Resolution in Metern (wenn None: settings.resolution)
            min_points_per_dimension: Mindestanzahl Punkte pro Dimension (Standard: 3)
        
        Returns:
            Dict von {surface_id: SurfaceGrid}
        """
        if resolution is None:
            resolution = self.settings.resolution
        
        # Analysiere Surfaces
        geometries = self.analyzer.analyze_surfaces(enabled_surfaces)
        
        # Erstelle Grid für jede Surface
        surface_grids = {}
        for geom in geometries:
            (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = \
                self.builder.build_single_surface_grid(
                    geometry=geom,
                    resolution=resolution,
                    min_points_per_dimension=min_points_per_dimension
                )
            
            # Berechne tatsächliche Resolution (kann adaptiv sein)
            ny, nx = X_grid.shape
            if len(sound_field_x) > 1 and len(sound_field_y) > 1:
                actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
                actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
                actual_resolution = (actual_resolution_x + actual_resolution_y) / 2.0
            else:
                actual_resolution = resolution
            
            surface_grids[geom.surface_id] = SurfaceGrid(
                surface_id=geom.surface_id,
                sound_field_x=sound_field_x,
                sound_field_y=sound_field_y,
                X_grid=X_grid,
                Y_grid=Y_grid,
                Z_grid=Z_grid,
                surface_mask=surface_mask,
                resolution=actual_resolution,
                geometry=geom
            )
            
            points_in_surface = int(np.sum(surface_mask))
            total_points = int(X_grid.size)
            
            # 🎯 DEBUG: Zusätzliche Info für vertikale Flächen
            is_vertical = geom.orientation == "vertical"
            if is_vertical:
                xs = X_grid.flatten()
                ys = Y_grid.flatten()
                zs = Z_grid.flatten()
                x_span = float(np.ptp(xs))
                y_span = float(np.ptp(ys))
                z_span = float(np.ptp(zs))
                print(f"[DEBUG Grid pro Surface] '{geom.surface_id}' (VERTIKAL):")
                print(f"  └─ Grid-Shape: {ny}×{nx} = {total_points} Punkte (gesamt)")
                print(f"  └─ Punkte in Surface: {points_in_surface}/{total_points}")
                print(f"  └─ Resolution: {actual_resolution:.3f} m")
                print(f"  └─ Koordinaten-Spannen: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
                print(f"  └─ X-Range: [{xs.min():.3f}, {xs.max():.3f}]")
                print(f"  └─ Y-Range: [{ys.min():.3f}, {ys.max():.3f}]")
                print(f"  └─ Z-Range: [{zs.min():.3f}, {zs.max():.3f}]")
            else:
                print(f"[DEBUG Grid pro Surface] '{geom.surface_id}': "
                      f"{points_in_surface}/{total_points} Punkte, Resolution: {actual_resolution:.3f} m")
        
        return surface_grids
    
    def _create_cache_hash(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float]
    ) -> str:
        """Erstellt Hash für Cache-Vergleich"""
        # Einfacher Hash: Anzahl Surfaces + Resolution
        n_surfaces = len(enabled_surfaces)
        res_str = str(resolution) if resolution else "default"
        return f"{n_surfaces}_{res_str}"

