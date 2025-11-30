"""
ShadowCalculator: Berechnet Schatten für Schallausbreitung in 3D-Umgebungen.

VERBESSERTE VERSION: Verwendet einfache Ray-Plane-Intersection statt PyVista.
Viel schneller und zuverlässiger!
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
)

# Importiere die gleiche Polygon-Prüfung wie SurfaceGridCalculator
try:
    from Module_LFO.Modules_Calculate.SurfaceGridCalculator import _point_in_polygon_core
except ImportError:
    # Fallback: Verwende eigene Implementierung
    _point_in_polygon_core = None

DEBUG_SHADOW = bool(int(os.environ.get("LFO_DEBUG_SHADOW", "1")))


class ShadowCalculator(ModuleBase):
    """
    Berechnet Schatten für Schallausbreitung basierend auf einfacher Ray-Plane-Intersection.
    
    Verfahren:
    1. Für jedes Hindernis (Surface mit plane_model):
       - Berechne Schnittpunkt Strahl-Ebene
       - Prüfe ob Schnittpunkt zwischen Quelle und Ziel liegt
       - Prüfe ob Schnittpunkt innerhalb des Polygons liegt (2D-Projektion)
    2. Wenn Schnittpunkt gefunden → Punkt ist im Schatten
    
    WICHTIG: 
    - Hindernis selbst wird NICHT als Schatten markiert (Punkt auf Hindernis = sichtbar)
    - Nur Punkte HINTER dem Hindernis werden als Schatten markiert
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
    
    def _point_in_polygon_2d(self, point: np.ndarray, polygon_points: np.ndarray) -> bool:
        """
        Prüft ob ein Punkt (x, y) innerhalb eines Polygons liegt (2D, XY-Ebene).
        Verwendet die gleiche Funktion wie SurfaceGridCalculator.
        
        Args:
            point: (x, y) Koordinaten
            polygon_points: Array (N, 2) mit Polygon-Punkten
            
        Returns:
            True wenn Punkt im Polygon liegt
        """
        if len(polygon_points) < 3:
            return False
        
        # Konvertiere zu Dict-Format für _point_in_polygon_core
        if _point_in_polygon_core is not None:
            polygon_dict = [
                {"x": float(p[0]), "y": float(p[1])} for p in polygon_points
            ]
            return _point_in_polygon_core(float(point[0]), float(point[1]), polygon_dict)
        
        # Fallback: Eigene Implementierung
        x, y = float(point[0]), float(point[1])
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = float(polygon_points[0, 0]), float(polygon_points[0, 1])
        for i in range(1, n + 1):
            p2x, p2y = float(polygon_points[i % n, 0]), float(polygon_points[i % n, 1])
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _ray_plane_intersection(
        self,
        source: np.ndarray,
        target: np.ndarray,
        plane_model: Dict[str, float],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Berechnet Schnittpunkt zwischen Strahl (source → target) und Ebene.
        
        Args:
            source: (x, y, z) Startpunkt
            target: (x, y, z) Zielpunkt
            plane_model: Dict mit "mode", "slope", "intercept", etc.
            
        Returns:
            (intersection_point, t) oder None
            t ist Parameter: intersection = source + t * (target - source)
            t=0 → source, t=1 → target, t zwischen 0 und 1 → zwischen source und target
        """
        mode = plane_model.get("mode")
        
        # Strahl-Richtung
        ray_dir = target - source
        ray_length = np.linalg.norm(ray_dir)
        if ray_length < 1e-6:
            return None
        ray_dir_normalized = ray_dir / ray_length
        
        # Ebenen-Gleichung basierend auf mode
        if mode == "constant":
            # Z = base (horizontale Ebene)
            z_plane = float(plane_model.get("base", 0.0))
            
            # Strahl: source + t * ray_dir
            # Z-Komponente: source[2] + t * ray_dir[2] = z_plane
            if abs(ray_dir[2]) < 1e-6:
                # Strahl parallel zur Ebene
                return None
            
            t = (z_plane - source[2]) / ray_dir[2]
            intersection = source + t * ray_dir
            
        elif mode == "x":
            # Z = slope * x + intercept
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            
            # Strahl: source + t * ray_dir
            # Z = slope * X + intercept
            # source[2] + t * ray_dir[2] = slope * (source[0] + t * ray_dir[0]) + intercept
            # source[2] + t * ray_dir[2] = slope * source[0] + slope * t * ray_dir[0] + intercept
            # t * (ray_dir[2] - slope * ray_dir[0]) = slope * source[0] + intercept - source[2]
            
            denominator = ray_dir[2] - slope * ray_dir[0]
            if abs(denominator) < 1e-6:
                return None
            
            t = (slope * source[0] + intercept - source[2]) / denominator
            intersection = source + t * ray_dir
            
        elif mode == "y":
            # Z = slope * y + intercept
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            
            denominator = ray_dir[2] - slope * ray_dir[1]
            if abs(denominator) < 1e-6:
                return None
            
            t = (slope * source[1] + intercept - source[2]) / denominator
            intersection = source + t * ray_dir
            
        else:
            # Allgemeine Ebene: Z = slope_x * x + slope_y * y + intercept
            slope_x = float(plane_model.get("slope_x", 0.0))
            slope_y = float(plane_model.get("slope_y", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            
            # Z = slope_x * X + slope_y * Y + intercept
            # source[2] + t * ray_dir[2] = slope_x * (source[0] + t * ray_dir[0]) + slope_y * (source[1] + t * ray_dir[1]) + intercept
            # source[2] + t * ray_dir[2] = slope_x * source[0] + slope_x * t * ray_dir[0] + slope_y * source[1] + slope_y * t * ray_dir[1] + intercept
            # t * (ray_dir[2] - slope_x * ray_dir[0] - slope_y * ray_dir[1]) = slope_x * source[0] + slope_y * source[1] + intercept - source[2]
            
            denominator = ray_dir[2] - slope_x * ray_dir[0] - slope_y * ray_dir[1]
            if abs(denominator) < 1e-6:
                return None
            
            t = (slope_x * source[0] + slope_y * source[1] + intercept - source[2]) / denominator
            intersection = source + t * ray_dir
        
        return intersection, t
    
    def compute_shadow_mask_for_calculation(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        X_grid: Optional[np.ndarray] = None,
        Y_grid: Optional[np.ndarray] = None,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Berechnet Schatten-Maske für Berechnung (einfach und schnell!).
        
        Args:
            grid_points: (N, 3) Array von 3D-Punkten
            source_positions: Liste von (x, y, z) Tupeln (eine pro Quelle)
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            X_grid, Y_grid: Optional, für Kompatibilität
            tolerance: Toleranz für numerische Vergleiche (m)
            
        Returns:
            Boolean-Array (N,): True = Punkt im Schatten, False = sichtbar
        """
        if len(grid_points) == 0:
            return np.zeros(0, dtype=bool)
        
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError("grid_points muss (N, 3) Array sein")
        
        # Initialisiere Schatten-Maske (False = sichtbar)
        shadow_mask = np.zeros(len(grid_points), dtype=bool)
        
        # Sammle Hindernisse (Surfaces mit plane_model)
        obstacles = []
        for surface_id, surface_dict in enabled_surfaces:
            points = surface_dict.get("points", [])
            if len(points) < 3:
                continue
            
            # Prüfe ob Surface ein Hindernis ist (vertikal oder signifikante Z-Ausdehnung)
            zs = [p.get("z", 0.0) for p in points]
            z_span = max(zs) - min(zs) if zs else 0.0
            
            # Nur Surfaces mit signifikanter Z-Ausdehnung sind Hindernisse
            if z_span < 0.5:  # Weniger als 50cm Z-Ausdehnung → kein Hindernis
                continue
            
            # Berechne plane_model
            plane_model, _ = derive_surface_plane(points)
            if plane_model is None:
                continue
            
            # Polygon-Punkte (2D, XY)
            polygon_xy = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in points])
            
            obstacles.append({
                "surface_id": surface_id,
                "plane_model": plane_model,
                "polygon_xy": polygon_xy,
                "points": points,
            })
        
        if len(obstacles) == 0:
            return shadow_mask
        
        if DEBUG_SHADOW:
            print(f"[ShadowCalculator] {len(obstacles)} Hindernisse gefunden")
        
        # Für jede Quelle
        for source_idx, source_pos in enumerate(source_positions):
            source_array = np.array(source_pos, dtype=float)
            
            if DEBUG_SHADOW and source_idx == 0:
                print(f"[ShadowCalculator] Prüfe Quelle {source_idx+1}/{len(source_positions)} bei {source_array}")
            
            # Für jeden Grid-Punkt
            for i, target in enumerate(grid_points):
                # 🎯 WICHTIG: Prüfe zuerst, ob der Zielpunkt selbst auf einem Hindernis liegt
                # Wenn ja, dann liegt der Punkt definitiv auf dem Hindernis → KEIN Schatten
                point_on_obstacle = False
                for obstacle in obstacles:
                    # Schnelle Bounding-Box-Prüfung (vor Polygon-Prüfung)
                    points_3d = obstacle["points"]
                    xs = [p.get("x", 0.0) for p in points_3d]
                    ys = [p.get("y", 0.0) for p in points_3d]
                    zs = [p.get("z", 0.0) for p in points_3d]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    z_min, z_max = min(zs), max(zs)
                    
                    # Prüfe ob Zielpunkt innerhalb der Bounding Box liegt (mit Toleranz)
                    bb_tolerance = max(tolerance, 0.2)  # Mindestens 20cm Toleranz für Bounding Box
                    in_bbox = (
                        (x_min - bb_tolerance <= target[0] <= x_max + bb_tolerance) and
                        (y_min - bb_tolerance <= target[1] <= y_max + bb_tolerance) and
                        (z_min - bb_tolerance <= target[2] <= z_max + bb_tolerance)
                    )
                    
                    if DEBUG_SHADOW and i < 20 and source_idx == 0 and obstacle["surface_id"] == "test_2":
                        print(
                            f"[Shadow] Punkt {i} ({target}): BBox-Prüfung für '{obstacle['surface_id']}': "
                            f"in_bbox={in_bbox}, "
                            f"X: {x_min:.2f}-{x_max:.2f} (Punkt: {target[0]:.2f}), "
                            f"Y: {y_min:.2f}-{y_max:.2f} (Punkt: {target[1]:.2f}), "
                            f"Z: {z_min:.2f}-{z_max:.2f} (Punkt: {target[2]:.2f})"
                        )
                    
                    if not in_bbox:
                        continue
                    
                    # Prüfe ob Zielpunkt innerhalb des Hindernis-Polygons liegt (2D, XY)
                    target_xy = target[:2]
                    in_polygon = self._point_in_polygon_2d(target_xy, obstacle["polygon_xy"])
                    
                    if DEBUG_SHADOW and i < 20 and source_idx == 0 and obstacle["surface_id"] == "test_2":
                        print(
                            f"[Shadow] Punkt {i}: Polygon-Prüfung für '{obstacle['surface_id']}': "
                            f"in_polygon={in_polygon}, target_xy=({target_xy[0]:.2f}, {target_xy[1]:.2f})"
                        )
                    
                    if not in_polygon:
                        continue
                    
                    # Prüfe ob Zielpunkt auf der Hindernis-Ebene liegt (Z-Diff < Toleranz)
                    plane_model = obstacle["plane_model"]
                    target_z_on_plane = evaluate_surface_plane(plane_model, target[0], target[1])
                    target_z_diff = abs(target[2] - target_z_on_plane)
                    z_tolerance = max(tolerance, 0.2)  # Mindestens 20cm Toleranz für Z
                    
                    if DEBUG_SHADOW and i < 20 and source_idx == 0 and obstacle["surface_id"] == "test_2":
                        print(
                            f"[Shadow] Punkt {i}: Z-Prüfung für '{obstacle['surface_id']}': "
                            f"target_z={target[2]:.4f}, plane_z={target_z_on_plane:.4f}, "
                            f"diff={target_z_diff:.4f}, tolerance={z_tolerance:.4f}"
                        )
                    
                    if target_z_diff < z_tolerance:
                        # Zielpunkt liegt auf Hindernis → KEIN Schatten
                        point_on_obstacle = True
                        if DEBUG_SHADOW and i < 20 and source_idx == 0:
                            print(
                                f"[Shadow] ✅ Punkt {i}: Auf Hindernis '{obstacle['surface_id']}' erkannt "
                                f"(Z-Diff={target_z_diff:.4f}m < {z_tolerance:.4f}m, in_polygon=True, in_bbox=True)"
                            )
                        break  # Ein Hindernis reicht
                
                if point_on_obstacle:
                    # Punkt liegt auf Hindernis → KEIN Schatten
                    continue
                
                # Prüfe jedes Hindernis für Schatten
                for obstacle in obstacles:
                    # Berechne Schnittpunkt Strahl-Ebene
                    result = self._ray_plane_intersection(
                        source_array,
                        target,
                        obstacle["plane_model"]
                    )
                    
                    if result is None:
                        continue
                    
                    intersection, t = result
                    
                    # 🎯 WICHTIG: Prüfe ob Schnittpunkt zwischen Quelle und Ziel liegt
                    # t sollte zwischen 0 und 1 liegen (mit Toleranz)
                    # t ≈ 1 bedeutet: Schnittpunkt genau am Ziel → Punkt liegt auf Hindernis → KEIN Schatten
                    # t < 1 - tolerance bedeutet: Schnittpunkt VOR dem Ziel → Schatten
                    
                    if t < 0.0 or t > 1.0:
                        # Schnittpunkt außerhalb des Strahls
                        continue
                    
                    # 🎯 WICHTIG: Prüfe ob Schnittpunkt genau am Ziel liegt
                    # Wenn t sehr nah bei 1.0 ist, liegt der Punkt auf dem Hindernis
                    # → KEIN Schatten (Hindernis selbst soll nicht deaktiviert werden)
                    ray_length = np.linalg.norm(target - source_array)
                    if ray_length < 1e-6:
                        continue
                    
                    # Prüfe direkt t: Wenn t sehr nah bei 1.0 ist → Punkt liegt auf Hindernis
                    # Verwende relative Toleranz für t (z.B. t > 0.99 bedeutet "sehr nah am Ziel")
                    # UND absolute Toleranz für Distanz (z.B. innerhalb 5cm vom Ziel)
                    t_threshold = 1.0 - max(tolerance / ray_length, 0.01)  # Mindestens 1% Toleranz für t
                    distance_to_target = abs(t - 1.0) * ray_length
                    distance_threshold = max(tolerance, 0.05)  # Mindestens 5cm Toleranz für Distanz
                    
                    # 🐛 DEBUG: Zeige Details für Punkte auf Hindernis
                    if DEBUG_SHADOW and i < 20 and source_idx == 0:
                        print(
                            f"[Shadow] Punkt {i}: t={t:.6f}, t_threshold={t_threshold:.6f}, "
                            f"dist_to_target={distance_to_target:.4f}m, dist_threshold={distance_threshold:.4f}m, "
                            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
                        )
                    
                    # Wenn t sehr nah bei 1.0 ist ODER Distanz sehr klein → Punkt liegt auf Hindernis
                    if t > t_threshold or distance_to_target < distance_threshold:
                        # Schnittpunkt sehr nah am Ziel → Punkt liegt auf Hindernis → KEIN Schatten
                        if DEBUG_SHADOW and i < 20 and source_idx == 0:
                            print(
                                f"[Shadow] Punkt {i}: Auf Hindernis erkannt "
                                f"(t={t:.6f} > {t_threshold:.6f} ODER dist={distance_to_target:.4f}m < {distance_threshold:.4f}m)"
                            )
                        continue
                    
                    # Prüfe ob Schnittpunkt innerhalb des Polygons liegt (2D, XY)
                    intersection_xy = intersection[:2]
                    in_polygon = self._point_in_polygon_2d(intersection_xy, obstacle["polygon_xy"])
                    
                    if DEBUG_SHADOW and i < 20 and source_idx == 0:
                        print(
                            f"[Shadow] Punkt {i}: t={t:.6f}, dist_to_target={distance_to_target:.4f}m, "
                            f"in_polygon={in_polygon}, intersection_xy=({intersection_xy[0]:.2f}, {intersection_xy[1]:.2f})"
                        )
                    
                    if not in_polygon:
                        continue
                    
                    # ✅ Schnittpunkt gefunden: zwischen Quelle und Ziel, innerhalb Polygon
                    # → Punkt ist im Schatten
                    if DEBUG_SHADOW and i < 20 and source_idx == 0:
                        print(f"[Shadow] Punkt {i}: IM SCHATTEN (t={t:.6f}, dist={distance_to_target:.4f}m)")
                    shadow_mask[i] = True
                    break  # Ein Hindernis reicht
        
        if DEBUG_SHADOW:
            num_shadow = np.count_nonzero(shadow_mask)
            print(f"[ShadowCalculator] Ergebnis: {num_shadow}/{len(grid_points)} Punkte im Schatten ({100.0*num_shadow/len(grid_points):.1f}%)")
        
        return shadow_mask
    
    def compute_high_res_shadow_mask_for_render(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        X_grid: Optional[np.ndarray] = None,
        Y_grid: Optional[np.ndarray] = None,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """Für Visualisierung - aktuell identisch mit compute_shadow_mask_for_calculation."""
        return self.compute_shadow_mask_for_calculation(
            grid_points, source_positions, enabled_surfaces, X_grid, Y_grid, tolerance
        )
    
    def compute_shadow_mask_for_visualization(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        X_grid: Optional[np.ndarray] = None,
        Y_grid: Optional[np.ndarray] = None,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """Für FEM/FDTD Visualisierung - aktuell identisch."""
        return self.compute_shadow_mask_for_calculation(
            grid_points, source_positions, enabled_surfaces, X_grid, Y_grid, tolerance
        )
