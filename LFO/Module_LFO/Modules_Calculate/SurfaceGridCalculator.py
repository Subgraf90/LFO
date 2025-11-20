"""
SurfaceGridCalculator: Erstellt Berechnungs-Grids basierend auf enabled Surfaces.

Dieses Modul kapselt die gesamte Logik zur Grid-Erstellung für Soundfield-Berechnungen:
- Bounding Box Berechnung
- Grid-Koordinaten-Generierung
- Surface-Masken-Erstellung
- Z-Koordinaten-Interpolation
- Extraktion surface-spezifischer Sampling-Grids
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

import numpy as np

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
    SurfaceDefinition,
)


@dataclass
class SurfaceGridMesh:
    surface_id: str
    name: str
    x_axis: np.ndarray
    y_axis: np.ndarray
    mask: np.ndarray
    z_grid: np.ndarray
    plane_model: Dict[str, float] | None

    def to_payload(self) -> Dict[str, List]:
        return {
            "surface_id": self.surface_id,
            "name": self.name,
            "x": self.x_axis.tolist(),
            "y": self.y_axis.tolist(),
            "mask": self.mask.tolist(),
            "z": self.z_grid.tolist(),
        }


@dataclass
class SurfaceSamplingPoints:
    surface_id: str
    name: str
    coordinates: np.ndarray  # Shape (N, 3)
    indices: np.ndarray      # Shape (N, 2) - (row, col) im lokalen Grid
    grid_shape: Tuple[int, int]

    def to_payload(self) -> Dict[str, List]:
        return {
            "surface_id": self.surface_id,
            "name": self.name,
            "coordinates": self.coordinates.tolist(),
            "indices": self.indices.tolist(),
            "grid_shape": list(self.grid_shape),
        }


class SurfaceGridCalculator(ModuleBase):
    """
    Berechnet und verwaltet Grids für Soundfield-Berechnungen basierend auf Surfaces.
    
    Features:
    - Automatische Bounding Box Berechnung aus enabled Surfaces
    - Vektorisierte Surface-Masken (Polygon-Tests)
    - Z-Koordinaten-Interpolation für nicht-ebene Surfaces
    - Resolution-Adequacy-Checks
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        self._last_surface_meshes: List[SurfaceGridMesh] = []
        self._last_surface_samples: List[SurfaceSamplingPoints] = []
    
    def create_calculation_grid(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float] = None,
        padding_factor: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        🎯 HAUPTMETHODE: Erstellt ein vollständiges Berechnungs-Grid basierend auf enabled Surfaces.
        
        Args:
            enabled_surfaces: Liste von (surface_id, surface_definition) Tupeln
            resolution: Grid-Auflösung in Metern. Wenn None, wird settings.resolution verwendet.
            padding_factor: Faktor für Padding um Bounding Box (Standard: 0.5 × resolution)
            
        Returns:
            Tuple von 6 Arrays:
            - sound_field_x: 1D-Array der X-Koordinaten
            - sound_field_y: 1D-Array der Y-Koordinaten
            - X_grid: 2D-Meshgrid der X-Koordinaten (Shape: [ny, nx])
            - Y_grid: 2D-Meshgrid der Y-Koordinaten (Shape: [ny, nx])
            - Z_grid: 2D-Array der Z-Koordinaten (Shape: [ny, nx])
            - surface_mask: 2D-Boolean-Array - True wenn Punkt in mindestens einem Surface (Shape: [ny, nx])
        """
        if resolution is None:
            resolution = self.settings.resolution
        
        # ============================================================
        # SCHRITT 1: Bounding Box berechnen
        # ============================================================
        if enabled_surfaces:
            min_x, max_x, min_y, max_y = self._calculate_bounding_box(enabled_surfaces)
            # Füge Padding hinzu, damit Randpunkte nicht abgeschnitten werden
            padding = resolution * padding_factor
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding
        else:
            # Fallback: Verwende Settings-Dimensionen (symmetrisch um 0)
            grid_width = self.settings.width
            grid_length = self.settings.length
            min_x = -grid_width / 2
            max_x = grid_width / 2
            min_y = -grid_length / 2
            max_y = grid_length / 2
        
        # ============================================================
        # SCHRITT 2: 1D-Koordinaten-Arrays erstellen
        # ============================================================
        sound_field_x = np.arange(min_x, max_x + resolution, resolution)
        sound_field_y = np.arange(min_y, max_y + resolution, resolution)
        
        # ============================================================
        # SCHRITT 3: 2D-Meshgrid erstellen
        # ============================================================
        X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
        
        # ============================================================
        # SCHRITT 4: Resolution-Check (optional, gibt Warnung aus)
        # ============================================================
        if enabled_surfaces:
            is_adequate, warning = self._check_resolution_adequacy(enabled_surfaces, resolution)
            if warning:
                print(f"[SurfaceGridCalculator] {warning}")
        
        # ============================================================
        # SCHRITT 5: Surface-Maske erstellen
        # ============================================================
        if enabled_surfaces:
            surface_mask = self._create_surface_mask(X_grid, Y_grid, enabled_surfaces)
        else:
            # Keine Surfaces → alle Punkte sind gültig
            surface_mask = np.ones_like(X_grid, dtype=bool)
        
        # ============================================================
        # SCHRITT 6: Z-Koordinaten interpolieren
        # ============================================================
        Z_grid = np.zeros_like(X_grid, dtype=float)  # Standard: Z=0
        if enabled_surfaces:
            Z_grid = self._interpolate_z_coordinates(X_grid, Y_grid, enabled_surfaces, surface_mask)
        
        self._last_surface_meshes = self._build_surface_meshes(enabled_surfaces, resolution)
        self._last_surface_samples = self._build_surface_samples(self._last_surface_meshes)
        return sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask

    def get_surface_meshes(self) -> List[SurfaceGridMesh]:
        return self._last_surface_meshes

    def get_surface_sampling_points(self) -> List[SurfaceSamplingPoints]:
        return self._last_surface_samples
    
    def _calculate_bounding_box(
        self,
        surfaces: List[Tuple[str, Dict]],
    ) -> Tuple[float, float, float, float]:
        """
        Berechnet die Bounding Box aller gegebenen Surfaces.
        
        Args:
            surfaces: Liste von (surface_id, surface_definition) Tupeln
            
        Returns:
            Tuple (min_x, max_x, min_y, max_y) - Grenzen der Bounding Box
        """
        if not surfaces:
            # Fallback: Verwende Settings-Dimensionen (symmetrisch um 0)
            width = getattr(self.settings, 'width', 150.0)
            length = getattr(self.settings, 'length', 100.0)
            return -width/2, width/2, -length/2, length/2
        
        # Sammle alle X- und Y-Koordinaten aller Surfaces
        all_x = []
        all_y = []
        
        for surface_id, surface_def in surfaces:
            points = surface_def.get('points', [])
            for point in points:
                all_x.append(point.get('x', 0.0))
                all_y.append(point.get('y', 0.0))
        
        if not all_x or not all_y:
            # Fallback: Verwende Settings-Dimensionen
            width = getattr(self.settings, 'width', 150.0)
            length = getattr(self.settings, 'length', 100.0)
            return -width/2, width/2, -length/2, length/2
        
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        
        return min_x, max_x, min_y, max_y
    
    def _check_resolution_adequacy(
        self,
        surfaces: List[Tuple[str, Dict]],
        resolution: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Prüft, ob die aktuelle Resolution für die gegebenen Surfaces ausreichend ist.
        
        Warnung: Bei zu grober Resolution können kleine Surfaces oder schmale Bereiche
        übersehen werden, da die Surface-Maske nur auf den Grid-Punkten berechnet wird.
        
        Args:
            surfaces: Liste von (surface_id, surface_definition) Tupeln
            resolution: Aktuelle Grid-Resolution in Metern
            
        Returns:
            Tuple (is_adequate, warning_message)
            - is_adequate: True wenn Resolution ausreichend ist
            - warning_message: Optional Warnung bei zu grober Resolution
        """
        if not surfaces:
            return True, None
        
        # Berechne minimale Feature-Größe aller Surfaces
        min_feature_size = float('inf')
        smallest_surface = None
        
        for surface_id, surface_def in surfaces:
            points = surface_def.get('points', [])
            if len(points) < 3:
                continue
            
            # Berechne minimale Kantenlänge des Polygons
            px = np.array([p.get('x', 0.0) for p in points])
            py = np.array([p.get('y', 0.0) for p in points])
            
            # Berechne alle Kantenlängen
            for i in range(len(px)):
                j = (i + 1) % len(px)
                edge_length = np.sqrt((px[j] - px[i])**2 + (py[j] - py[i])**2)
                if edge_length > 0 and edge_length < min_feature_size:
                    min_feature_size = edge_length
                    smallest_surface = surface_def.get('name', surface_id)
        
        # Regel: Resolution sollte mindestens 1/3 der kleinsten Feature-Größe sein
        # (d.h. mindestens 3 Punkte pro kleinster Kante)
        recommended_resolution = min_feature_size / 3.0 if min_feature_size < float('inf') else resolution
        
        if resolution > recommended_resolution * 1.5:  # 50% Toleranz
            warning = (
                f"⚠️ Resolution ({resolution:.2f}m) könnte zu grob sein für Surface '{smallest_surface}'. "
                f"Empfohlen: ≤{recommended_resolution:.2f}m "
                f"(kleinste Feature-Größe: {min_feature_size:.2f}m). "
                f"Kleine Bereiche könnten übersehen werden."
            )
            return False, warning
        
        return True, None
    
    def _create_surface_mask(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        surfaces: List[Tuple[str, Dict]],
    ) -> np.ndarray:
        """
        🚀 Erstellt eine kombinierte Maske für alle gegebenen Surfaces.
        Ein Punkt ist True, wenn er in MINDESTENS EINEM Surface liegt.
        
        Args:
            x_coords: 2D-Array von X-Koordinaten (Shape: [ny, nx])
            y_coords: 2D-Array von Y-Koordinaten (Shape: [ny, nx])
            surfaces: Liste von (surface_id, surface_definition) Tupeln
            
        Returns:
            2D-Boolean-Array (Shape: [ny, nx]) - True wenn Punkt in mindestens einem Surface liegt
        """
        if not surfaces:
            # Keine Surfaces → alle Punkte sind gültig
            return np.ones_like(x_coords, dtype=bool)
        
        # Initialisiere Maske als False
        combined_mask = np.zeros_like(x_coords, dtype=bool)
        
        # Kombiniere alle Surface-Masken (Union)
        for surface_id, surface_def in surfaces:
            points = surface_def.get('points', [])
            if len(points) >= 3:
                surface_mask = self._points_in_polygon_batch(x_coords, y_coords, points)
                combined_mask = combined_mask | surface_mask
        
        return combined_mask

    def _build_surface_meshes(
        self,
        surfaces: List[Tuple[str, Dict]],
        resolution: float,
    ) -> List[SurfaceGridMesh]:
        meshes: List[SurfaceGridMesh] = []
        if not surfaces:
            return meshes

        step = float(resolution or 1.0)
        for surface_id, definition in surfaces:
            surface = definition
            if not isinstance(surface, SurfaceDefinition):
                surface = SurfaceDefinition.from_dict(surface_id, definition)
            points = surface.points
            if len(points) < 3:
                continue

            xs = [float(pt.get("x", 0.0)) for pt in points]
            ys = [float(pt.get("y", 0.0)) for pt in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
                continue

            x_axis = np.arange(min_x, max_x + step, step)
            y_axis = np.arange(min_y, max_y + step, step)
            if x_axis.size < 2 or y_axis.size < 2:
                continue

            X_local, Y_local = np.meshgrid(x_axis, y_axis, indexing="xy")
            mask = self._points_in_polygon_batch(X_local, Y_local, points)
            if not np.any(mask):
                continue

            plane_model = surface.plane_model
            if plane_model is None:
                plane_model, _ = derive_surface_plane(points)

            z_grid = self._evaluate_plane_grid(X_local, Y_local, plane_model)

            meshes.append(
                SurfaceGridMesh(
                    surface_id=surface_id,
                    name=surface.name,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    mask=mask,
                    z_grid=z_grid,
                    plane_model=plane_model,
                )
            )
        return meshes

    def _build_surface_samples(
        self,
        meshes: List[SurfaceGridMesh],
    ) -> List[SurfaceSamplingPoints]:
        samples: List[SurfaceSamplingPoints] = []
        for mesh in meshes:
            rows, cols = np.where(mesh.mask)
            if rows.size == 0:
                continue
            xs = mesh.x_axis[cols]
            ys = mesh.y_axis[rows]
            zs = mesh.z_grid[rows, cols]
            coords = np.column_stack((xs, ys, zs))
            indices = np.column_stack((rows, cols))
            samples.append(
                SurfaceSamplingPoints(
                    surface_id=mesh.surface_id,
                    name=mesh.name,
                    coordinates=coords,
                    indices=indices,
                    grid_shape=mesh.mask.shape,
                )
            )
        return samples
    
    def _interpolate_z_coordinates(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        surfaces: List[Tuple[str, Dict]],
        surface_mask: np.ndarray,
    ) -> np.ndarray:
        """
        🎯 Interpoliert Z-Koordinaten basierend auf planaren Surface-Modellen.
        
        🎯 FESTE GEOMETRIE: X/Y-Positionen werden immer fest eingehalten, auch wenn Z variiert.
        Wenn eine Surface Z-Abweichungen hat, werden diese normalisiert (einzelne Abweichung → 4.0).
        
        Args:
            x_coords: 2D-Array von X-Koordinaten (Shape: [ny, nx])
            y_coords: 2D-Array von Y-Koordinaten (Shape: [ny, nx])
            surfaces: Liste von (surface_id, surface_definition) Tupeln
            surface_mask: 2D-Boolean-Array - True wenn Punkt in mindestens einem Surface
            
        Returns:
            2D-Array der Z-Koordinaten (Shape: [ny, nx])
        """
        Z_grid = np.zeros_like(x_coords, dtype=float)
        
        if not surfaces:
            return Z_grid
        
        # 🎯 Normalisiere Z-Koordinaten für alle Surfaces
        def _safe_coord(point_obj, key: str, default: float = 0.0) -> float:
            if isinstance(point_obj, dict):
                value = point_obj.get(key, default)
            else:
                value = getattr(point_obj, key, default)
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        normalized_surfaces = []
        for surface_id, surface_def in surfaces:
            raw_points = surface_def.get("points", [])
            if not isinstance(raw_points, list):
                continue
            points = []
            for raw_point in raw_points:
                points.append(
                    {
                        "x": _safe_coord(raw_point, "x", 0.0),
                        "y": _safe_coord(raw_point, "y", 0.0),
                        "z": _safe_coord(raw_point, "z", 0.0),
                    }
                )
            if len(points) < 3:
                continue
            
            # Extrahiere Z-Werte
            z_values = [p["z"] for p in points]
            
            # 🎯 Normalisierung: Wenn genau eine Abweichung, setze auf 4.0
            unique_z = sorted(set(z_values))
            if len(unique_z) == 2:  # Genau zwei unterschiedliche Z-Werte
                z_common = unique_z[0]
                z_deviation = unique_z[1]
                count_common = z_values.count(z_common)
                count_deviation = z_values.count(z_deviation)
                
                # Wenn nur ein Punkt abweicht, normalisiere Abweichung auf 4.0
                if (count_common == 1 and count_deviation == len(z_values) - 1) or \
                   (count_deviation == 1 and count_common == len(z_values) - 1):
                    # Normalisiere: X/Y bleiben fest, Z-Abweichung wird auf 4.0 gesetzt
                    normalized_z = z_common + (4.0 if z_deviation > z_common else -4.0)
                    
                    # Erstelle normalisierte Punkte (X/Y bleiben unverändert!)
                    normalized_points = []
                    for i, point in enumerate(points):
                        normalized_point = point.copy()
                        if abs(z_values[i] - z_deviation) < 0.001:  # Kleine Toleranz
                            normalized_point['z'] = normalized_z
                        else:
                            normalized_point['z'] = z_common
                        normalized_points.append(normalized_point)
                    points = normalized_points
            
            # Erstelle Surface-Modell mit normalisierten Punkten
            model, error = derive_surface_plane(points)
            if model is None:
                name = surface_def.get("name", surface_id)
                print(
                    f"[SurfaceGridCalculator] Surface '{name}' wird ignoriert: {error}"
                )
                continue
            normalized_surfaces.append((points, model, surface_def.get("name", surface_id)))
        
        if not normalized_surfaces:
            return Z_grid
        
        # Interpoliere Z-Koordinaten für alle gültigen Punkte
        # 🎯 X/Y-Positionen bleiben fest, nur Z wird interpoliert
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        mask_flat = surface_mask.flatten()
        
        indices = np.where(mask_flat)[0]
        for idx in indices:
            x_point = x_flat[idx]  # X fest
            y_point = y_flat[idx]  # Y fest
            
            z_values = []
            for points, model, _ in normalized_surfaces:
                if self._point_in_polygon(x_point, y_point, points):
                    z_val = evaluate_surface_plane(model, x_point, y_point)
                    z_values.append(z_val)
            
            if z_values:
                iy, ix = np.unravel_index(idx, x_coords.shape)
                Z_grid[iy, ix] = float(np.mean(z_values))  # Z wird interpoliert
        
        return Z_grid
    
    def _point_in_polygon(
        self,
        x: float,
        y: float,
        polygon_points: List[Dict[str, float]],
    ) -> bool:
        """
        Prüft, ob ein Punkt (x, y) innerhalb eines Polygons liegt.
        Verwendet den Ray Casting Algorithmus (Winding Number).
        
        Args:
            x: X-Koordinate des Punktes
            y: Y-Koordinate des Punktes
            polygon_points: Liste von Punkten mit 'x' und 'y' Schlüsseln
            
        Returns:
            True wenn Punkt innerhalb des Polygons liegt, sonst False
        """
        if len(polygon_points) < 3:
            return False
        
        # Extrahiere X- und Y-Koordinaten
        px = np.array([p.get('x', 0.0) for p in polygon_points])
        py = np.array([p.get('y', 0.0) for p in polygon_points])
        
        # Ray Casting Algorithmus
        n = len(px)
        inside = False
        
        j = n - 1
        boundary_eps = 1e-6
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            
            # Prüfe ob Strahl von (x,y) nach rechts die Kante schneidet
            if ((yi > y) != (yj > y)) and (x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi + boundary_eps):
                inside = not inside
            
            # Prüfe ob Punkt direkt auf der Kante liegt
            dx = xj - xi
            dy = yj - yi
            segment_len = math.hypot(dx, dy)
            if segment_len > 0:
                dist = abs(dy * (x - xi) - dx * (y - yi)) / segment_len
                if dist <= boundary_eps:
                    proj = ((x - xi) * dx + (y - yi) * dy) / (segment_len * segment_len)
                    if -boundary_eps <= proj <= 1 + boundary_eps:
                        return True
            j = i
        
        return inside
    
    def _points_in_polygon_batch(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polygon_points: List[Dict[str, float]],
    ) -> np.ndarray:
        """
        🚀 VEKTORISIERT: Prüft für viele Punkte gleichzeitig, ob sie innerhalb eines Polygons liegen.
        
        Args:
            x_coords: 2D-Array von X-Koordinaten (Shape: [ny, nx])
            y_coords: 2D-Array von Y-Koordinaten (Shape: [ny, nx])
            polygon_points: Liste von Punkten mit 'x' und 'y' Schlüsseln
            
        Returns:
            2D-Boolean-Array (Shape: [ny, nx]) - True wenn Punkt innerhalb liegt
        """
        if len(polygon_points) < 3:
            return np.zeros_like(x_coords, dtype=bool)
        
        # Extrahiere Polygon-Koordinaten
        px = np.array([p.get('x', 0.0) for p in polygon_points])
        py = np.array([p.get('y', 0.0) for p in polygon_points])
        
        # Flatten für einfachere Verarbeitung
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        n_points = len(x_flat)
        
        # Ray Casting für alle Punkte gleichzeitig
        inside = np.zeros(n_points, dtype=bool)
        on_edge = np.zeros(n_points, dtype=bool)
        boundary_eps = 1e-6
        n = len(px)
        
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            
            # Vektorisierte Prüfung: Welche Punkte schneiden diese Kante?
            # Bedingung: ((yi > y) != (yj > y)) AND (x < Schnittpunkt-X)
            y_above_edge = (yi > y_flat) != (yj > y_flat)
            intersection_x = (xj - xi) * (y_flat - yi) / (yj - yi + 1e-10) + xi
            intersects = y_above_edge & (x_flat <= intersection_x + boundary_eps)
            
            # XOR-Operation für Winding Number
            inside = inside ^ intersects
            
            # Prüfe Punkte, die direkt auf der Kante liegen
            dx = xj - xi
            dy = yj - yi
            segment_len = math.hypot(dx, dy)
            if segment_len > 0:
                numerator = np.abs(dy * (x_flat - xi) - dx * (y_flat - yi))
                dist = numerator / (segment_len + 1e-12)
                proj = ((x_flat - xi) * dx + (y_flat - yi) * dy) / ((segment_len ** 2) + 1e-12)
                on_edge_segment = (dist <= boundary_eps) & (proj >= -boundary_eps) & (proj <= 1 + boundary_eps)
                on_edge = on_edge | on_edge_segment
            
            j = i
        
        # Reshape zurück zu Original-Form
        return (inside | on_edge).reshape(x_coords.shape)

    @staticmethod
    def _evaluate_plane_grid(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        plane_model: Optional[Dict[str, float]],
    ) -> np.ndarray:
        if plane_model is None:
            return np.zeros_like(x_coords, dtype=float)
        mode = plane_model.get("mode")
        if mode == "x":
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            return slope * x_coords + intercept
        if mode == "y":
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            return slope * y_coords + intercept
        base = float(plane_model.get("base", 0.0))
        return np.full_like(x_coords, base, dtype=float)

