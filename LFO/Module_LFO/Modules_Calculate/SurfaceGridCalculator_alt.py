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
import os
import time

import numpy as np

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Init.Logging import measure_time, perf_section
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
    SurfaceDefinition,
)

DEBUG_SURFACE_GRID_TIMING = bool(int(os.environ.get("LFO_DEBUG_TIMING", "0")))

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
    is_vertical: bool = False

    def to_payload(self) -> Dict[str, List]:
        return {
            "surface_id": self.surface_id,
            "name": self.name,
            "coordinates": self.coordinates.tolist(),
            "indices": self.indices.tolist(),
            "grid_shape": list(self.grid_shape),
            "kind": "vertical" if self.is_vertical else "planar",
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
    
    @measure_time("SurfaceGridCalculator.create_calculation_grid")
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
        
        t_start_total = time.perf_counter() if DEBUG_SURFACE_GRID_TIMING else 0.0
        
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
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_bbox = time.perf_counter()
        
        # ============================================================
        # SCHRITT 2: 1D-Koordinaten-Arrays erstellen
        # ============================================================
        sound_field_x = np.arange(min_x, max_x + resolution, resolution)
        sound_field_y = np.arange(min_y, max_y + resolution, resolution)
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_axes = time.perf_counter()
        
        # ============================================================
        # SCHRITT 3: 2D-Meshgrid erstellen
        # ============================================================
        X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_mesh = time.perf_counter()
        
        # ============================================================
        # SCHRITT 4: Resolution-Check (optional, gibt Warnung aus)
        # ============================================================
        if enabled_surfaces:
            is_adequate, warning = self._check_resolution_adequacy(enabled_surfaces, resolution)
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_rescheck = time.perf_counter()
        
        # ============================================================
        # SCHRITT 5: Surface-Maske erstellen (ERWEITERT für Berechnung)
        # ============================================================
        if enabled_surfaces:
            # 🎯 ERWEITERTE MASKE: Erfasst auch Randpunkte für vollständige Berechnung
            surface_mask = self._create_surface_mask(X_grid, Y_grid, enabled_surfaces, include_edges=True)
        else:
            # Keine Surfaces → alle Punkte sind gültig
            surface_mask = np.ones_like(X_grid, dtype=bool)
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_mask = time.perf_counter()
        
        # ============================================================
        # SCHRITT 6: Z-Koordinaten interpolieren
        # ============================================================
        Z_grid = np.zeros_like(X_grid, dtype=float)  # Standard: Z=0
        if enabled_surfaces:
            Z_grid = self._interpolate_z_coordinates(X_grid, Y_grid, enabled_surfaces, surface_mask)
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_z = time.perf_counter()
        
        self._last_surface_meshes = self._build_surface_meshes(enabled_surfaces, resolution)
        # Horizontale/geneigte Flächen über reguläre Meshes
        self._last_surface_samples = self._build_surface_samples(self._last_surface_meshes)
        # Zusätzliche Sampling-Punkte für senkrechte Flächen (XY-Projektion ~ Linie)
        vertical_samples = self._build_vertical_surface_samples(enabled_surfaces, resolution)
        if vertical_samples:
            self._last_surface_samples.extend(vertical_samples)
        
        if DEBUG_SURFACE_GRID_TIMING:
            t_samples = time.perf_counter()
            total_ms = (t_samples - t_start_total) * 1000.0
            print(
                "[SurfaceGridCalculator] create_calculation_grid timings:\n"
                f"  bounding_box     : {(t_bbox - t_start_total) * 1000.0:7.2f} ms\n"
                f"  axes             : {(t_axes - t_bbox) * 1000.0:7.2f} ms\n"
                f"  meshgrid         : {(t_mesh - t_axes) * 1000.0:7.2f} ms\n"
                f"  resolution_check : {(t_rescheck - t_mesh) * 1000.0:7.2f} ms\n"
                f"  surface_mask     : {(t_mask - t_rescheck) * 1000.0:7.2f} ms\n"
                f"  z_interpolation  : {(t_z - t_mask) * 1000.0:7.2f} ms\n"
                f"  meshes+samples   : {(t_samples - t_z) * 1000.0:7.2f} ms\n"
                f"  TOTAL            : {total_ms:7.2f} ms"
            )
        return sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask

    def create_calculation_grid_per_group(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float] = None,
        padding_factor: float = 0.5,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Erstellt pro Surface-Gruppe (group_id / group_name) ein eigenes Grid.

        Args:
            enabled_surfaces: Liste von (surface_id, surface_definition)
            resolution: Grid-Auflösung (None -> settings.resolution)
            padding_factor: Padding-Faktor um die Bounding-Box

        Returns:
            Dict[group_id, (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask)]
        """
        if resolution is None:
            resolution = self.settings.resolution

        if not enabled_surfaces:
            return {}

        # Gruppiere Surfaces nach group_id / group_name (Fallback "__ungrouped__")
        grouped: Dict[str, List[Tuple[str, Dict]]] = {}
        for sid, sdef in enabled_surfaces:
            gid = sdef.get("group_id") or sdef.get("group_name") or "__ungrouped__"
            grouped.setdefault(gid, []).append((sid, sdef))

        result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for gid, surfaces in grouped.items():
            try:
                grid_tuple = self.create_calculation_grid(
                    surfaces,
                    resolution=resolution,
                    padding_factor=padding_factor,
                )
                result[str(gid)] = grid_tuple
            except Exception as exc:
                if DEBUG_SURFACE_GRID_TIMING:
                    print(f"[SurfaceGridCalculator] create_calculation_grid_per_group failed for group '{gid}': {exc}")
                # Fehlerhafte Gruppe überspringen
                continue

        return result

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

    def _interpolate_z_coordinates(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        surfaces: List[Tuple[str, Dict]],
        surface_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Thin-Wrapper um die eigentliche Implementierung, damit
        `SurfaceGridCalculator` das Attribut `_interpolate_z_coordinates`
        besitzt und bestehende Aufrufer weiter funktionieren.
        """
        return _interpolate_z_coordinates_impl(
            self,
            x_coords,
            y_coords,
            surfaces,
            surface_mask,
        )

    def _evaluate_plane_grid(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        plane_model: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        Thin-Wrapper um die planare Z-Auswertung, damit `_evaluate_plane_grid`
        als Methoden-Attribut der Klasse existiert.
        """
        return _evaluate_plane_grid_core(x_coords, y_coords, plane_model)

    def _point_in_polygon(
        self,
        x: float,
        y: float,
        polygon_points: List[Dict[str, float]],
    ) -> bool:
        """
        Thin-Wrapper um die Punkt-im-Polygon-Prüfung, damit `SurfaceGridCalculator`
        diese Methode immer als Attribut besitzt.
        """
        return _point_in_polygon_core(x, y, polygon_points)
    
    def _create_surface_mask(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        surfaces: List[Tuple[str, Dict]],
        include_edges: bool = True,
    ) -> np.ndarray:
        """
        🚀 Erstellt eine kombinierte Maske für alle gegebenen Surfaces.
        Ein Punkt ist True, wenn er in MINDESTENS EINEM Surface liegt.
        
        Args:
            x_coords: 2D-Array von X-Koordinaten (Shape: [ny, nx])
            y_coords: 2D-Array von Y-Koordinaten (Shape: [ny, nx])
            surfaces: Liste von (surface_id, surface_definition) Tupeln
            include_edges: Wenn True, werden auch Punkte nahe den Rändern erfasst (für Berechnung)
            
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
                
                # 🎯 DEBUG: Referenzpunkt pro Surface beim Grid erstellen
                # Wähle den Mittelpunkt der Bounding Box als Referenzpunkt
                surface_xs = [p.get("x", 0.0) for p in points]
                surface_ys = [p.get("y", 0.0) for p in points]
                if surface_xs and surface_ys:
                    ref_x = (min(surface_xs) + max(surface_xs)) / 2.0
                    ref_y = (min(surface_ys) + max(surface_ys)) / 2.0
                    # Finde nächsten Grid-Punkt der wirklich im Surface liegt
                    ny, nx = x_coords.shape
                    # Suche alle Punkte im Surface und wähle den nähesten zum BBox-Mitte
                    surface_mask_flat = surface_mask.flatten()
                    if np.any(surface_mask_flat):
                        # Finde Indizes der Punkte im Surface
                        surface_indices = np.where(surface_mask_flat)[0]
                        # Berechne Distanzen zum Referenzpunkt
                        surface_points = np.column_stack((
                            x_coords.flatten()[surface_indices],
                            y_coords.flatten()[surface_indices]
                        ))
                        ref_point_2d = np.array([ref_x, ref_y])
                        distances = np.linalg.norm(surface_points - ref_point_2d, axis=1)
                        nearest_surface_idx = surface_indices[np.argmin(distances)]
                        y_idx, x_idx = np.unravel_index(nearest_surface_idx, x_coords.shape)
                    else:
                        # Fallback: Suche einfach nächstgelegenen Punkt
                        x_idx = int(np.argmin(np.abs(x_coords[ny//2, :] - ref_x))) if nx > 0 else 0
                        y_idx = int(np.argmin(np.abs(y_coords[:, nx//2] - ref_y))) if ny > 0 else 0
                    
                    if 0 <= y_idx < ny and 0 <= x_idx < nx:
                        grid_x = float(x_coords[y_idx, x_idx])
                        grid_y = float(y_coords[y_idx, x_idx])
                        grid_z = 0.0  # Wird später in Z_grid gesetzt
                        
                        # Speichere Referenzpunkt für späteren Vergleich (in Settings oder als Attribut)
                        if not hasattr(self, '_surface_ref_points'):
                            self._surface_ref_points = {}
                        self._surface_ref_points[surface_id] = {
                            'grid_x': grid_x,
                            'grid_y': grid_y,
                            'grid_z': grid_z,
                            'grid_idx': (y_idx, x_idx),
                            'bbox_center': (ref_x, ref_y)
                        }
        
        # 🎯 ERWEITERE MASKE FÜR BERECHNUNG: Erfasse auch Randpunkte
        if include_edges:
            # Morphologische Dilatation: Erweitere die Maske um 1 Pixel in alle Richtungen
            # Dies erfasst auch Punkte, die direkt auf den Rändern liegen
            try:
                from scipy import ndimage
                # Strukturelement: 3x3-Kreis (erfasst alle 8 Nachbarn)
                structure = np.array([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]], dtype=bool)
                combined_mask = ndimage.binary_dilation(combined_mask, structure=structure)
            except ImportError:
                # Fallback: Manuelle Erweiterung wenn scipy nicht verfügbar
                combined_mask = self._dilate_mask_manual(combined_mask)
        
        return combined_mask
    
    def _dilate_mask_manual(self, mask: np.ndarray) -> np.ndarray:
        """
        Manuelle morphologische Dilatation (Fallback wenn scipy nicht verfügbar).
        Erweitert die Maske um 1 Pixel in alle Richtungen.
        """
        dilated = mask.copy()
        ny, nx = mask.shape
        
        # Erweitere in alle 8 Richtungen
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                # Verschiebe Maske und kombiniere
                shifted = np.zeros_like(mask)
                i_start = max(0, -di)
                i_end = min(nx, nx - di)
                j_start = max(0, -dj)
                j_end = min(ny, ny - dj)
                
                if i_start < i_end and j_start < j_end:
                    shifted[j_start:j_end, i_start:i_end] = mask[
                        j_start + dj:j_end + dj,
                        i_start + di:i_end + di
                    ]
                dilated = dilated | shifted
        
        return dilated

    def _points_in_polygon_batch(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polygon_points: List[Dict[str, float]],
    ) -> np.ndarray:
        """
        Wrapper um die modulare, vektorisierte Polygon-Test-Funktion.
        Stellt sicher, dass `SurfaceGridCalculator` immer ein Attribut
        `_points_in_polygon_batch` besitzt (wichtig für ältere Importe).
        """
        return _points_in_polygon_batch_core(x_coords, y_coords, polygon_points)

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
                        is_vertical=False,
                )
            )
        return samples

    def _build_vertical_surface_samples(
        self,
        surfaces: List[Tuple[str, Dict]],
        resolution: float,
    ) -> List[SurfaceSamplingPoints]:
        """
        Erstellt Sampling-Punkte für senkrechte Flächen (XY-Projektion ~ Linie).
        Diese Punkte liegen direkt auf der vertikalen Fläche und werden als
        3D-Koordinaten für die Schallfeld-Berechnung verwendet (surface_point_buffers).
        """
        vertical_samples: List[SurfaceSamplingPoints] = []
        if not surfaces:
            return vertical_samples

        # Für vertikale Flächen soll die Berechnungsauflösung NICHT vom
        # Plot-Upscale-Faktor abhängen, sondern ausschließlich von der
        # globalen Grid-Resolution. Wir verfeinern die Auflösung entlang
        # der Wand jedoch leicht (Faktor 2), damit die sichtbaren Stufen
        # an schrägen Kanten ähnlich fein wirken wie bei den XY-Plots,
        # ohne zusätzliche Interpolation der SPL-Werte.
        base_resolution = float(resolution or 1.0)
        refine_factor = 2.0
        step = base_resolution / refine_factor
        if step <= 0.0:
            step = base_resolution or 1.0

        for surface_id, definition in surfaces:
            surface = definition
            if not isinstance(surface, SurfaceDefinition):
                surface = SurfaceDefinition.from_dict(surface_id, definition)
            points = surface.points
            if len(points) < 3:
                continue

            xs = np.array([float(pt.get("x", 0.0)) for pt in points], dtype=float)
            ys = np.array([float(pt.get("y", 0.0)) for pt in points], dtype=float)
            zs = np.array([float(pt.get("z", 0.0)) for pt in points], dtype=float)

            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            z_span = float(np.ptp(zs))

            # Versuche zunächst, die Fläche als planare Z=Z(x,y)-Fläche zu modellieren.
            # Wenn das gelingt, ist sie NICHT senkrecht (Steigung < 180°) und soll
            # über die normale horizontale Berechnung laufen.
            dict_points = [
                {"x": float(pt.get("x", 0.0)), "y": float(pt.get("y", 0.0)), "z": float(pt.get("z", 0.0))}
                for pt in points
            ]
            model, _ = derive_surface_plane(dict_points)
            if model is not None:
                continue

            # Vollständig degeneriert (Punkt oder Linie ohne Ausdehnung) oder
            # praktisch keine Höhe → ignorieren.
            if (x_span < 1e-6 and y_span < 1e-6) or z_span <= 1e-3:
                continue

            # Ab hier behandeln wir nur noch echte senkrechte Flächen:
            # XY-Projektion ist (fast) eine Linie, aber Δz ist signifikant.

            # Vollständig degeneriert (Punkt oder Linie ohne Ausdehnung)
            if x_span < 1e-6 and y_span < 1e-6:
                continue

            # Fall 1: Fläche verläuft im X-Z-Raum bei (fast) konstantem Y
            if y_span < 1e-6:
                y0 = float(np.mean(ys))
                u_min, u_max = float(xs.min()), float(xs.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    continue

                u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                if u_axis.size < 2 or v_axis.size < 2:
                    continue

                U, V = np.meshgrid(u_axis, v_axis, indexing="xy")
                # 2D-Polygon im (x,z)-Raum aufbauen
                polygon_uv = [
                    {"x": float(pt.get("x", 0.0)), "y": float(pt.get("z", 0.0))}
                    for pt in points
                ]
                # Wie bei den nicht senkrechten Flächen: Maske leicht erweitern,
                # damit der Plot bis an den Rand der Fläche sauber gefüllt wird.
                mask = self._points_in_polygon_batch(U, V, polygon_uv)
                mask = _dilate_mask_minimal(mask)
                if not np.any(mask):
                    continue

                rows, cols = np.where(mask)
                coords_x = u_axis[cols]
                coords_y = np.full_like(coords_x, y0, dtype=float)
                coords_z = v_axis[rows]
                coords = np.column_stack((coords_x, coords_y, coords_z))

                vertical_samples.append(
                    SurfaceSamplingPoints(
                        surface_id=surface_id,
                        name=surface.name,
                        coordinates=coords,
                        # Speichere die lokalen Gitter-Indizes, damit wir in der
                        # Plot-Geometrie wieder ein 2D-Gitter rekonstruieren können.
                        indices=np.column_stack((rows, cols)),
                        grid_shape=mask.shape,
                        is_vertical=True,
                    )
                )

            # Fall 2: Fläche verläuft im Y-Z-Raum bei (fast) konstantem X
            elif x_span < 1e-6:
                x0 = float(np.mean(xs))
                u_min, u_max = float(ys.min()), float(ys.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    continue

                u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                if u_axis.size < 2 or v_axis.size < 2:
                    continue

                U, V = np.meshgrid(u_axis, v_axis, indexing="xy")
                # 2D-Polygon im (y,z)-Raum aufbauen
                polygon_uv = [
                    {"x": float(pt.get("y", 0.0)), "y": float(pt.get("z", 0.0))}
                    for pt in points
                ]
                # Leicht erweiterte Maske analog zu den waagrechten Flächen
                mask = self._points_in_polygon_batch(U, V, polygon_uv)
                mask = _dilate_mask_minimal(mask)
                if not np.any(mask):
                    continue

                rows, cols = np.where(mask)
                coords_x = np.full_like(u_axis[cols], x0, dtype=float)
                coords_y = u_axis[cols]
                coords_z = v_axis[rows]
                coords = np.column_stack((coords_x, coords_y, coords_z))

                vertical_samples.append(
                    SurfaceSamplingPoints(
                        surface_id=surface_id,
                        name=surface.name,
                        coordinates=coords,
                        # Lokale Indizes im (u,v)-Raster speichern
                        indices=np.column_stack((rows, cols)),
                        grid_shape=mask.shape,
                        is_vertical=True,
                    )
                )

        return vertical_samples


def _dilate_mask_minimal(mask: np.ndarray) -> np.ndarray:
    """
    Einfache 3x3-Dilatation für boolesche Masken.
    Wird genutzt, um die Sample-Maske von Surfaces um 1 Zelle zu erweitern,
    analog zur Behandlung der Plot-Maske bei horizontalen Flächen.
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
    
def _interpolate_z_coordinates_impl(
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
    """
    # 🐛 DEBUG: Funktion wurde aufgerufen (immer aktiv für Z-Grid-Debugging)
    print(f"[DEBUG Z-Interpolation] Funktion aufgerufen:")
    print(f"  Grid shape: {x_coords.shape}")
    print(f"  Anzahl Surfaces: {len(surfaces)}")
    print(f"  Surface-Maske coverage: {np.count_nonzero(surface_mask) / surface_mask.size * 100:.1f}%")
    
    Z_grid = np.zeros_like(x_coords, dtype=float)
    
    if not surfaces:
        print(f"[DEBUG Z-Interpolation] ⚠️  Keine Surfaces übergeben - Z-Grid bleibt 0")
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

        # XY-/Z-Ausdehnung bestimmen
        xs = np.array([p["x"] for p in points], dtype=float)
        ys = np.array([p["y"] for p in points], dtype=float)
        zs = np.array([p["z"] for p in points], dtype=float)
        x_span = float(np.ptp(xs))
        y_span = float(np.ptp(ys))
        z_span = float(np.ptp(zs))

        name = surface_def.get("name", surface_id)

        # Versuche zuerst immer ein normales planeres Modell zu finden
        # (konstant / X-Steigung / Y-Steigung / allgemeine Ebene).
        model, error = derive_surface_plane(points)
        
        # 🐛 DEBUG: Prüfe Plane-Model-Erstellung (immer aktiv für Z-Grid-Debugging)
        print(f"[DEBUG Z-Interpolation] Surface {name}:")
        print(f"  Points: {len(points)}")
        print(f"  X span: {x_span:.3f}, Y span: {y_span:.3f}, Z span: {z_span:.3f}")
        if model is not None:
            mode = model.get("mode", "unknown")
            print(f"  ✅ Plane model found: mode={mode}, model={model}")
            normalized_surfaces.append(
                (points, model, surface_def.get("name", surface_id))
            )
        else:
            print(f"  ❌ No plane model found: error={error}")
            print(f"  Points Z-values: {[p['z'] for p in points]}")

        # Wenn ein Plan-Modell existiert, ist die Fläche NICHT senkrecht
        # (Steigung < 180°) → normale Berechnung.
        if model is not None:
            continue

        # Kein gültiges Planmodell → kann z.B. eine echte senkrechte Wand sein.
        # Diese erkennen wir daran, dass die XY-Projektion (fast) eine Linie ist,
        # die Fläche aber eine nennenswerte Z-Höhe hat.
        is_xy_line = (x_span < 1e-6 and y_span >= 1e-6) or (
            y_span < 1e-6 and x_span >= 1e-6
        )
        has_height = z_span > 1e-3

        if is_xy_line and has_height:
            # → Echte senkrechte Fläche: hier NICHT in das horizontale Z-Grid
            # einmischen. Sie wird ausschließlich über die vertikalen
            # Surface-Samples behandelt.
            continue

        # Weder planar noch sinnvolle senkrechte Fläche → komplett ignorieren.
        continue
    
    if not normalized_surfaces:
        print(f"[DEBUG Z-Interpolation] ⚠️  Keine normalisierten Surfaces gefunden")
        return Z_grid
    
    # 🐛 DEBUG: Prüfe normalisierte Surfaces (immer aktiv für Z-Grid-Debugging)
    print(f"[DEBUG Z-Interpolation] Gefundene normalisierte Surfaces: {len(normalized_surfaces)}")
    for points, model, surface_name in normalized_surfaces:
        mode = model.get("mode", "unknown")
        print(f"  - {surface_name}: mode={mode}, model={model}")
    
    # 🚀 OPTIMIERT: Vektorisierte Z-Interpolation für alle Punkte gleichzeitig
    # Statt einzelner Punkt-Prüfungen verwenden wir Batch-Operationen
    
    # Importiere vektorisierte Funktionen
    from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
        _evaluate_plane_on_grid,
        evaluate_surface_plane,
    )
    
    # Erstelle Maske für alle Punkte im Grid
    mask_flat = surface_mask.flatten()
    masked_indices = np.where(mask_flat)[0]
    
    # 🐛 DEBUG: Surface-Maske (immer aktiv für Z-Grid-Debugging)
    print(f"[DEBUG Z-Interpolation] Surface-Maske: {len(masked_indices)} / {len(mask_flat)} Punkte maskiert")
    print(f"  Grid shape: {x_coords.shape}, total points: {x_coords.size}")
    print(f"  Surface-Maske coverage: {len(masked_indices) / len(mask_flat) * 100:.1f}%")
    
    if len(masked_indices) == 0:
        print(f"[DEBUG Z-Interpolation] ⚠️  Keine maskierten Punkte gefunden - Z-Grid bleibt 0")
        return Z_grid
    
    # Extrahiere X/Y-Koordinaten für alle maskierten Punkte
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # 🚀 SCHRITT 1: Berechne für jede Surface die Polygon-Maske vektorisiert
    # Speichere Z-Werte pro Surface (für Überlappungen später mitteln)
    z_contributions = []  # Liste von (mask, z_values) Tupeln
    z_counts = np.zeros(len(masked_indices), dtype=float)
    
    for points, model, surface_name in normalized_surfaces:
        # Vektorisierte Polygon-Prüfung für ALLE Punkte gleichzeitig
        # Erstelle temporäres 2D-Grid nur für maskierte Punkte
        X_temp = x_coords.copy()
        Y_temp = y_coords.copy()
        
        # Berechne Polygon-Maske für das gesamte Grid
        polygon_mask = self._points_in_polygon_batch(X_temp, Y_temp, points)
        
        # Extrahiere nur die maskierten Punkte
        polygon_mask_flat = polygon_mask.flatten()
        polygon_mask_masked = polygon_mask_flat[masked_indices]
        
        if not np.any(polygon_mask_masked):
            continue
        
        # 🚀 Vektorisierte Z-Berechnung für alle Punkte innerhalb des Polygons
        # Verwende _evaluate_plane_on_grid für das gesamte Grid
        Z_surface = _evaluate_plane_on_grid(model, X_temp, Y_temp)
        Z_surface_flat = Z_surface.flatten()
        Z_surface_masked = Z_surface_flat[masked_indices]
        
        # 🐛 DEBUG: Prüfe Z-Berechnung (immer aktiv für Z-Grid-Debugging)
        points_in_polygon = np.sum(polygon_mask_masked)
        if points_in_polygon > 0:
            z_min = np.min(Z_surface_masked[polygon_mask_masked])
            z_max = np.max(Z_surface_masked[polygon_mask_masked])
            z_mean = np.mean(Z_surface_masked[polygon_mask_masked])
            # Prüfe auch einige Beispielpunkte
            sample_indices = np.where(polygon_mask_masked)[0][:3] if np.any(polygon_mask_masked) else []
            print(f"[DEBUG Z-Interpolation] Surface {surface_name}: {points_in_polygon} Punkte im Polygon, Z range: [{z_min:.2f}, {z_max:.2f}], mean: {z_mean:.2f}")
            if len(sample_indices) > 0:
                for sample_idx in sample_indices:
                    orig_idx = masked_indices[sample_idx]
                    iy, ix = np.unravel_index(orig_idx, x_coords.shape)
                    x_val = x_coords[iy, ix]
                    y_val = y_coords[iy, ix]
                    z_val = Z_surface_masked[sample_idx]
                    # Berechne erwarteten Z-Wert
                    from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import evaluate_surface_plane
                    z_expected = evaluate_surface_plane(model, x_val, y_val)
                    print(f"  Sample [jj={iy},ii={ix}]: X={x_val:.2f}, Y={y_val:.2f}, Z={z_val:.2f}, Z_expected={z_expected:.2f}, diff={abs(z_val-z_expected):.3f}")

        # Speichere Beitrag dieser Surface
        z_contributions.append((polygon_mask_masked, Z_surface_masked))
    
    # 🚀 SCHRITT 2: Kombiniere Z-Werte von allen Surfaces (Mittelwert bei Überlappungen)
    Z_sum = np.zeros(len(masked_indices), dtype=float)
    
    for polygon_mask_masked, Z_surface_masked in z_contributions:
        # Addiere Z-Werte nur für Punkte innerhalb dieses Polygons
        Z_sum[polygon_mask_masked] += Z_surface_masked[polygon_mask_masked]
        z_counts[polygon_mask_masked] += 1.0
    
    # Berechne Mittelwerte (verhindere Division durch Null)
    valid_mask = z_counts > 0.0
    Z_final = np.zeros(len(masked_indices), dtype=float)
    Z_final[valid_mask] = Z_sum[valid_mask] / z_counts[valid_mask]
    
    # 🚀 SCHRITT 3: Fülle Z-Werte für Punkte in der ERWEITERTEN Maske (außerhalb der Polygone)
    # Diese Punkte bekommen Z-Werte aus planaren Fortsetzungen der Surface-Ebenen
    edge_mask = ~valid_mask  # Punkte ohne Z-Wert aus Schritt 1
    
    if np.any(edge_mask):
        edge_indices = masked_indices[edge_mask]
        x_edge = x_flat[edge_indices]
        y_edge = y_flat[edge_indices]
        
        # 🚀 Vektorisierte Edge-Behandlung: Berechne Z für alle Edge-Punkte gleichzeitig
        # Erstelle temporäres Grid nur für Edge-Punkte
        X_edge = x_edge.reshape(-1, 1)
        Y_edge = y_edge.reshape(-1, 1)
        
        # Für jede Surface: Prüfe Bounding-Box und berechne Z-Werte vektorisiert
        edge_z_contributions = []
        edge_z_counts = np.zeros(len(edge_indices), dtype=float)
        
        for points, model, surface_name in normalized_surfaces:
            # Bounding-Box-Check vektorisiert
            xs = np.array([p["x"] for p in points], dtype=float)
            ys = np.array([p["y"] for p in points], dtype=float)
            x_min, x_max = xs.min() - 1e-6, xs.max() + 1e-6
            y_min, y_max = ys.min() - 1e-6, ys.max() + 1e-6
            
            # Vektorisierte Bounding-Box-Prüfung
            in_bbox = (
                (x_edge >= x_min) & (x_edge <= x_max) &
                (y_edge >= y_min) & (y_edge <= y_max)
            )
            
            if not np.any(in_bbox):
                continue
            
            # 🚀 Vektorisierte Z-Berechnung: Verwende evaluate_surface_plane direkt
            # Da wir nur einzelne Punkte haben, können wir die Funktion vektorisiert aufrufen
            # durch Broadcasting der Model-Parameter
            mode = model.get("mode")
            if mode == "constant":
                base = float(model.get("base", 0.0))
                Z_edge_surface = np.full(len(edge_indices), base, dtype=float)
            elif mode == "x":
                slope = float(model.get("slope", 0.0))
                intercept = float(model.get("intercept", 0.0))
                Z_edge_surface = slope * x_edge + intercept
            elif mode == "y":
                slope = float(model.get("slope", 0.0))
                intercept = float(model.get("intercept", 0.0))
                Z_edge_surface = slope * y_edge + intercept
            elif mode == "xy":
                slope_x = float(model.get("slope_x", model.get("slope", 0.0)))
                slope_y = float(model.get("slope_y", 0.0))
                intercept = float(model.get("intercept", 0.0))
                Z_edge_surface = slope_x * x_edge + slope_y * y_edge + intercept
            else:
                base = float(model.get("base", 0.0))
                Z_edge_surface = np.full(len(edge_indices), base, dtype=float)
            
            # Speichere nur für Punkte innerhalb der Bounding-Box
            edge_z_contributions.append((in_bbox, Z_edge_surface))
        
        # Kombiniere Z-Werte für Edge-Punkte
        edge_Z_sum = np.zeros(len(edge_indices), dtype=float)
        for in_bbox, Z_edge_surface_flat in edge_z_contributions:
            edge_Z_sum[in_bbox] += Z_edge_surface_flat[in_bbox]
            edge_z_counts[in_bbox] += 1.0
        
        # Berechne Mittelwerte
        edge_valid = edge_z_counts > 0.0
        if np.any(edge_valid):
            edge_Z_final = np.zeros(len(edge_indices), dtype=float)
            edge_Z_final[edge_valid] = edge_Z_sum[edge_valid] / edge_z_counts[edge_valid]
            
            # Schreibe Edge-Z-Werte zurück
            for i, idx in enumerate(edge_indices):
                if edge_z_counts[i] > 0.0:
                    iy, ix = np.unravel_index(idx, x_coords.shape)
                    Z_grid[iy, ix] = edge_Z_final[i]
                    # Aktualisiere auch Z_final für konsistente Indizierung
                    orig_idx = np.where(masked_indices == idx)[0][0]
                    Z_final[orig_idx] = edge_Z_final[i]
    
    # 🚀 SCHRITT 4: Schreibe Z-Werte zurück ins Grid
    points_with_z = 0
    for i, idx in enumerate(masked_indices):
        if z_counts[i] > 0.0:  # Nur wenn Z-Wert berechnet wurde
            iy, ix = np.unravel_index(idx, x_coords.shape)
            Z_grid[iy, ix] = Z_final[i]
            points_with_z += 1
    
    # 🐛 DEBUG: Prüfe Ergebnis (immer aktiv für Z-Grid-Debugging)
    print(f"[DEBUG Z-Interpolation] Ergebnis:")
    print(f"  Punkte mit Z-Wert: {points_with_z} / {len(masked_indices)}")
    print(f"  Z-Grid range: [{Z_grid.min():.2f}, {Z_grid.max():.2f}]")
    print(f"  Z-Grid non-zero: {np.count_nonzero(Z_grid)} / {Z_grid.size}")
    # Prüfe Ecken des Grids
    ny, nx = Z_grid.shape
    corners = [
        (0, 0, "oben-links"),
        (0, nx-1, "oben-rechts"),
        (ny-1, 0, "unten-links"),
        (ny-1, nx-1, "unten-rechts"),
    ]
    for jj, ii, name in corners:
        x_val = x_coords[jj, ii]
        y_val = y_coords[jj, ii]
        z_val = Z_grid[jj, ii]
        in_mask = surface_mask[jj, ii]
        print(f"    Corner {name} [jj={jj},ii={ii}]: X={x_val:.2f}, Y={y_val:.2f}, Z={z_val:.2f}, in_mask={in_mask}")
    
    return Z_grid

def _point_in_polygon_core(
    x: float,
    y: float,
    polygon_points: List[Dict[str, float]],
) -> bool:
    """
    Prüft, ob ein Punkt (x, y) innerhalb eines Polygons liegt.
    Verwendet den Ray-Casting-Algorithmus (Winding Number).
    """
    if len(polygon_points) < 3:
        return False

    # Extrahiere X- und Y-Koordinaten
    px = np.array([p.get("x", 0.0) for p in polygon_points])
    py = np.array([p.get("y", 0.0) for p in polygon_points])

    # Ray-Casting-Algorithmus
    n = len(px)
    inside = False

    j = n - 1
    boundary_eps = 1e-6
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]

        # Prüfe ob Strahl von (x,y) nach rechts die Kante schneidet
        if ((yi > y) != (yj > y)) and (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi + boundary_eps
        ):
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

def _points_in_polygon_batch_core(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polygon_points: List[Dict[str, float]],
) -> np.ndarray:
    """
    🚀 VEKTORISIERT: Prüft für viele Punkte gleichzeitig, ob sie innerhalb eines Polygons liegen.
    Wird von `SurfaceGridCalculator._points_in_polygon_batch` als Kernfunktion verwendet.
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


def _evaluate_plane_grid_core(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    plane_model: Optional[Dict[str, float]],
) -> np.ndarray:
    """
    Kern-Implementierung für die Auswertung eines planaren Modells Z=Z(x,y)
    auf einem 2D-Grid. Wird über `SurfaceGridCalculator._evaluate_plane_grid`
    aufgerufen.
    """
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

