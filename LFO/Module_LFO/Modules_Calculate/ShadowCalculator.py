"""
ShadowCalculator: Berechnet Schatten für Schallausbreitung in 3D-Umgebungen.

Verwendet Ray-Tracing, um abgeschattete Punkte zu identifizieren, wenn Hindernisse
zwischen Schallquellen und Empfangspunkten liegen.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os

try:
    import pyvista as pv
except ImportError:
    pv = None

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    build_surface_clipping_mesh,
    build_vertical_surface_mesh,
)

DEBUG_SHADOW = bool(int(os.environ.get("LFO_DEBUG_SHADOW", "0")))


class ShadowCalculator(ModuleBase):
    
    def _log_debug(self, message: str):
        """Hilfsfunktion für konsistente Debug-Ausgaben."""
        if DEBUG_SHADOW:
            print(message)
    """
    Berechnet Schatten für Schallausbreitung basierend auf Ray-Tracing.
    
    Verfahren:
    1. Ray-Triangle Intersection: Für jeden Grid-Punkt wird ein Strahl von der
       Quelle zum Punkt geworfen und auf Schnitte mit Surface-Meshes geprüft.
    2. PyVista-basiert: Nutzt PyVista's Ray-Tracing-Funktionen für effiziente
       Schnitttests.
    
    WICHTIG: Unterschiedliche Anwendung je nach Berechnungsmethode:
    
    - Superposition (SoundfieldCalculator):
      → Schatten-Maske ANWENDEN: Punkte im Schatten werden NICHT berechnet
      → Physikalisch korrekt: Schall breitet sich geradlinig aus, keine Beugung
    
    - FEM/FDTD (SoundFieldCalculatorFEM/FDTD):
      → Schatten-Maske NICHT anwenden: Alle Punkte werden berechnet
      → Beugung wird durch die numerische Methode automatisch berücksichtigt
      → Schatten-Maske kann für Visualisierung/Optimierung genutzt werden
    
    Verwendung:
        shadow_calc = ShadowCalculator(settings)
        shadow_mask = shadow_calc.compute_shadow_mask(
            grid_points,  # (N, 3) Array von 3D-Punkten
            source_positions,  # Liste von (x, y, z) Tupeln
            enabled_surfaces  # Liste von SurfaceDefinition
        )
        
        # Bei Superposition: Maske anwenden
        if use_superposition:
            sound_field_p[shadow_mask_2d] = 0.0  # oder np.nan
        
        # Bei FEM/FDTD: Maske nur für Visualisierung
        if use_fem:
            # Berechnung läuft normal, Maske nur für Plot
            pass
    """
    
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        self._surface_meshes_cache: Dict[str, pv.PolyData] = {}
        self._combined_obstacle_mesh: Optional[pv.PolyData] = None
        
    def build_obstacle_mesh(
        self,
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        include_hidden: bool = False,
    ) -> Optional[pv.PolyData]:
        """
        Erstellt ein kombiniertes Mesh aller Hindernisse (Surfaces) für Ray-Tracing.
        
        Args:
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            include_hidden: Wenn True, werden auch hidden Surfaces einbezogen
            
        Returns:
            PyVista PolyData Mesh mit allen Hindernissen oder None
        """
        if pv is None:
            if DEBUG_SHADOW:
                self._log_debug("[ShadowCalculator] PyVista nicht verfügbar")
            return None
            
        if not enabled_surfaces:
            return None
            
        meshes = []
        
        for surface_id, surface_dict in enabled_surfaces:
            # Überspringe hidden Surfaces, wenn nicht gewünscht
            if not include_hidden and surface_dict.get("hidden", False):
                continue
                
            # Überspringe nicht-enabled Surfaces
            if not surface_dict.get("enabled", False):
                continue
                
            try:
                # Erstelle SurfaceDefinition
                surface_def = SurfaceDefinition.from_dict(surface_id, surface_dict)
                
                # Erstelle Mesh für diese Surface
                # Verwende build_surface_clipping_mesh für Hindernisse
                points = surface_dict.get("points", [])
                plane_model = surface_dict.get("plane_model")
                
                if len(points) < 3:
                    if DEBUG_SHADOW:
                        self._log_debug(
                            f"[ShadowCalculator] Surface '{surface_id}': "
                            f"Zu wenige Punkte ({len(points)} < 3)"
                        )
                    continue
                
                mesh = build_surface_clipping_mesh(
                    surface_id,
                    points,
                    plane_model,
                    resolution=self.settings.resolution,
                    pv_module=pv,
                )
                
                # WICHTIG: Für Ray-Tracing müssen wir sicherstellen, dass das Mesh
                # beide Seiten der Fläche hat oder als dünnes Volumen modelliert ist
                # PyVista's ray_trace funktioniert besser mit geschlossenen Meshes
                
                if mesh is not None and hasattr(mesh, 'n_cells') and mesh.n_cells > 0:
                    # WICHTIG: Für Ray-Tracing müssen wir sicherstellen, dass das Mesh
                    # korrekt trianguliert ist und beide Seiten der Fläche erfasst
                    # PyVista's ray_trace funktioniert besser mit geschlossenen Meshes
                    # oder zumindest mit expliziten Normalen
                    
                    # Prüfe ob Mesh gültig ist
                    if mesh.n_points >= 3:
                        meshes.append(mesh)
                        if DEBUG_SHADOW:
                            # Zeige auch Bounds des Meshes
                            if hasattr(mesh, 'bounds'):
                                bounds = mesh.bounds
                                self._log_debug(
                                    f"[ShadowCalculator] Surface '{surface_id}': "
                                    f"{mesh.n_cells} Zellen, {mesh.n_points} Punkte hinzugefügt, "
                                    f"Bounds: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
                                    f"Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], "
                                    f"Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]"
                                )
                            else:
                                self._log_debug(
                                    f"[ShadowCalculator] Surface '{surface_id}': "
                                    f"{mesh.n_cells} Zellen, {mesh.n_points} Punkte hinzugefügt"
                                )
                    elif DEBUG_SHADOW:
                        self._log_debug(
                            f"[ShadowCalculator] Surface '{surface_id}': "
                            f"Mesh hat zu wenige Punkte ({mesh.n_points} < 3)"
                        )
                elif DEBUG_SHADOW:
                    self._log_debug(
                        f"[ShadowCalculator] Surface '{surface_id}': "
                        f"Mesh ist None oder leer (n_cells={getattr(mesh, 'n_cells', 0) if mesh is not None else 0})"
                    )
                        
            except Exception as exc:
                if DEBUG_SHADOW:
                    self._log_debug(
                        f"[ShadowCalculator] Fehler bei Surface '{surface_id}': {exc}"
                    )
                continue
                
        if not meshes:
            return None
            
        # Kombiniere alle Meshes zu einem einzigen Mesh
        try:
            combined = meshes[0]
            for mesh in meshes[1:]:
                combined = combined + mesh
                
            self._combined_obstacle_mesh = combined
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Kombiniertes Hindernis-Mesh: "
                    f"{combined.n_cells} Zellen, {combined.n_points} Punkte"
                )
            return combined
            
        except Exception as exc:
            if DEBUG_SHADOW:
                self._log_debug(f"[ShadowCalculator] Fehler beim Kombinieren: {exc}")
            return None
    
    def compute_shadow_mask(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        tolerance: float = 0.01,  # Erhöht auf 1cm für bessere Erkennung dünner Flächen
    ) -> np.ndarray:
        """
        Berechnet eine Schatten-Maske für alle Grid-Punkte.
        
        Ein Punkt ist im Schatten, wenn mindestens eine Surface den Strahl
        von der Quelle zum Punkt blockiert.
        
        Args:
            grid_points: (N, 3) Array von 3D-Punkten [x, y, z]
            source_positions: Liste von (x, y, z) Tupeln für Schallquellen
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            tolerance: Numerische Toleranz für Ray-Tests (in Metern)
            
        Returns:
            (N,) Boolean-Array: True wenn Punkt im Schatten, False wenn sichtbar
        """
        if pv is None:
            if DEBUG_SHADOW:
                self._log_debug("[ShadowCalculator] PyVista nicht verfügbar - keine Schatten")
            return np.zeros(len(grid_points), dtype=bool)
            
        if not source_positions or not enabled_surfaces:
            return np.zeros(len(grid_points), dtype=bool)
            
        # Erstelle Hindernis-Mesh
        obstacle_mesh = self.build_obstacle_mesh(enabled_surfaces)
        if obstacle_mesh is None or obstacle_mesh.n_cells == 0:
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Keine Hindernisse - keine Schatten "
                    f"(enabled_surfaces={len(enabled_surfaces)})"
                )
            return np.zeros(len(grid_points), dtype=bool)
            
        # Konvertiere grid_points zu NumPy-Array falls nötig
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError(
                f"grid_points muss Shape (N, 3) haben, bekam {grid_points.shape}"
            )
            
        N = len(grid_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        # Zeitmessung für Performance-Überwachung
        import time
        start_time = time.time()
        
        if DEBUG_SHADOW:
            self._log_debug(
                f"[ShadowCalculator] Berechne Schatten für {N} Punkte, "
                f"{len(source_positions)} Quellen, {obstacle_mesh.n_cells} Hindernis-Zellen, "
                f"{obstacle_mesh.n_points} Hindernis-Punkte"
            )
            # Prüfe ob Mesh gültig ist
            if hasattr(obstacle_mesh, 'bounds'):
                bounds = obstacle_mesh.bounds
                self._log_debug(
                    f"[ShadowCalculator] Hindernis-Mesh Bounds: "
                    f"X=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
                    f"Y=[{bounds[2]:.2f}, {bounds[3]:.2f}], "
                    f"Z=[{bounds[4]:.2f}, {bounds[5]:.2f}]"
                )
            # Zeige Grid-Punkte-Bounds zum Vergleich
            if len(grid_points) > 0:
                grid_bounds = (
                    np.min(grid_points[:, 0]), np.max(grid_points[:, 0]),
                    np.min(grid_points[:, 1]), np.max(grid_points[:, 1]),
                    np.min(grid_points[:, 2]), np.max(grid_points[:, 2]),
                )
                self._log_debug(
                    f"[ShadowCalculator] Grid-Punkte Bounds: "
                    f"X=[{grid_bounds[0]:.2f}, {grid_bounds[1]:.2f}], "
                    f"Y=[{grid_bounds[2]:.2f}, {grid_bounds[3]:.2f}], "
                    f"Z=[{grid_bounds[4]:.2f}, {grid_bounds[5]:.2f}]"
                )
            
        # Für jede Quelle prüfen wir, welche Punkte im Schatten sind
        for source_idx, source_pos in enumerate(source_positions):
            source_pos = np.asarray(source_pos, dtype=float)
            
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Prüfe Quelle {source_idx+1}/{len(source_positions)} "
                    f"bei ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})"
                )
                # Teste einen Beispiel-Strahl
                if source_idx == 0 and len(grid_points) > 0:
                    test_target = grid_points[0]
                    test_dist = np.linalg.norm(test_target - source_pos)
                    self._log_debug(
                        f"[ShadowCalculator] Test-Strahl: Quelle → Punkt 0 "
                        f"({test_target[0]:.2f}, {test_target[1]:.2f}, {test_target[2]:.2f}), "
                        f"Distanz: {test_dist:.2f}m"
                    )
                
            # Berechne Richtungsvektoren von Quelle zu allen Punkten
            directions = grid_points - source_pos[None, :]  # (N, 3)
            distances = np.linalg.norm(directions, axis=1)  # (N,)
            
            # Überspringe Punkte, die zu nah an der Quelle sind (numerische Probleme)
            valid_mask = distances > tolerance
            if not np.any(valid_mask):
                continue
                
            # Normalisiere Richtungen
            directions[valid_mask] = directions[valid_mask] / distances[valid_mask, None]
            
            # Batch-Ray-Tracing: Prüfe alle Strahlen gleichzeitig
            # PyVista's ray_trace kann mehrere Strahlen gleichzeitig verarbeiten
            source_shadow = self._ray_trace_batch(
                source_pos,
                grid_points[valid_mask],
                directions[valid_mask],
                obstacle_mesh,
                tolerance,
            )
            
            # Setze Schatten-Maske für diese Quelle
            shadow_mask[valid_mask] = shadow_mask[valid_mask] | source_shadow
            
        elapsed_time = time.time() - start_time
        num_shadow = np.count_nonzero(shadow_mask)
        
        if DEBUG_SHADOW:
            self._log_debug(
                f"[ShadowCalculator] {num_shadow}/{N} Punkte im Schatten "
                f"({100.0*num_shadow/N:.1f}%) - Berechnungszeit: {elapsed_time:.2f}s"
            )
        elif elapsed_time > 1.0:  # Warnung bei langen Berechnungen (>1s)
            self._log_debug(
                f"[ShadowCalculator] Ray-Tracing dauerte {elapsed_time:.2f}s "
                f"({num_shadow}/{N} Punkte im Schatten)"
            )
            # Zeige Details für erste Schatten-Punkte
            if num_shadow > 0:
                shadow_indices = np.where(shadow_mask)[0][:5]  # Erste 5
                self._log_debug(
                    f"[ShadowCalculator] Erste Schatten-Punkte (Indizes): {shadow_indices.tolist()}"
                )
            else:
                self._log_debug(
                    f"[ShadowCalculator] WARNUNG: Keine Schatten-Punkte gefunden! "
                    f"Prüfe ob Hindernisse zwischen Quellen und Punkten liegen."
                )
            
        return shadow_mask
    
    def _ray_trace_batch(
        self,
        source_pos: np.ndarray,
        target_points: np.ndarray,
        directions: np.ndarray,
        obstacle_mesh: pv.PolyData,
        tolerance: float,
    ) -> np.ndarray:
        """
        Führt Ray-Tracing für einen Batch von Strahlen durch.
        
        OPTIMIERT: Verwendet frühe Abbruchbedingungen und Bounding-Box-Prüfungen
        für bessere Performance.
        
        Args:
            source_pos: (3,) Startposition der Strahlen
            target_points: (N, 3) Zielpunkte
            directions: (N, 3) Normalisierte Richtungsvektoren
            obstacle_mesh: PyVista Mesh der Hindernisse
            tolerance: Numerische Toleranz
            
        Returns:
            (N,) Boolean-Array: True wenn Strahl blockiert wird
        """
        N = len(target_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        # 🚀 OPTIMIERUNG 1: Hole Mesh-Bounds einmalig
        if not hasattr(obstacle_mesh, 'bounds') or obstacle_mesh.n_cells == 0:
            return shadow_mask
        
        mesh_bounds = obstacle_mesh.bounds
        mesh_min = np.array([mesh_bounds[0], mesh_bounds[2], mesh_bounds[4]])
        mesh_max = np.array([mesh_bounds[1], mesh_bounds[3], mesh_bounds[5]])
        
        # 🚀 OPTIMIERUNG 2: Precompute alle Zell-Daten einmalig
        cell_data = []
        for cell_id in range(obstacle_mesh.n_cells):
            cell = obstacle_mesh.get_cell(cell_id)
            if cell.n_points < 3:
                continue
            
            p0 = np.asarray(obstacle_mesh.points[cell.point_ids[0]])
            p1 = np.asarray(obstacle_mesh.points[cell.point_ids[1]])
            p2 = np.asarray(obstacle_mesh.points[cell.point_ids[2]])
            
            # Berechne Zell-Bounding-Box
            cell_min = np.array([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1]), min(p0[2], p1[2], p2[2])])
            cell_max = np.array([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1]), max(p0[2], p1[2], p2[2])])
            
            cell_data.append({
                'id': cell_id,
                'p0': p0,
                'p1': p1,
                'p2': p2,
                'min': cell_min,
                'max': cell_max,
            })
        
        if not cell_data:
            return shadow_mask
        
        # 🚀 OPTIMIERUNG 3: Prüfe zuerst, ob Strahl überhaupt durch Mesh-Bounds geht
        source_pos_array = np.asarray(source_pos)
        
        # Für jeden Strahl prüfen wir, ob er ein Hindernis schneidet
        for i in range(N):
            target = target_points[i]
            target_array = np.asarray(target)
            distance = np.linalg.norm(target_array - source_pos_array)
            
            # 🚀 OPTIMIERUNG 4: Frühe Abbruchbedingung: Prüfe ob Strahl durch Mesh-Bounds geht
            ray_min = np.minimum(source_pos_array, target_array)
            ray_max = np.maximum(source_pos_array, target_array)
            
            # Prüfe ob Strahl-Bounding-Box Mesh-Bounds schneidet
            if not (
                ray_min[0] <= mesh_max[0] and ray_max[0] >= mesh_min[0] and
                ray_min[1] <= mesh_max[1] and ray_max[1] >= mesh_min[1] and
                ray_min[2] <= mesh_max[2] and ray_max[2] >= mesh_min[2]
            ):
                # Strahl geht nicht durch Mesh-Bounds → kein Schatten
                continue
            
            # Werfe Strahl von Quelle zum Zielpunkt
            points = []
            try:
                # 🚀 OPTIMIERUNG 5: Iteriere nur über Zellen, die Strahl-Bounding-Box schneiden
                for cell_info in cell_data:
                    # Schnelle Bounding-Box-Prüfung: Überspringe Zellen außerhalb des Strahls
                    if not (
                        cell_info['min'][0] <= ray_max[0] and cell_info['max'][0] >= ray_min[0] and
                        cell_info['min'][1] <= ray_max[1] and cell_info['max'][1] >= ray_min[1] and
                        cell_info['min'][2] <= ray_max[2] and cell_info['max'][2] >= ray_min[2]
                    ):
                        continue
                    
                    # Ray-Triangle-Intersection (Möller-Trumbore Algorithmus)
                    intersection = self._ray_triangle_intersection(
                        source_pos_array, target_array,
                        cell_info['p0'], cell_info['p1'], cell_info['p2'],
                        tolerance
                    )
                    
                    if intersection is not None:
                        points.append(intersection)
                        # Stoppe bei erstem Schnittpunkt (für Schatten reicht das)
                        break
                
                # Falls keine Schnittpunkte gefunden, versuche PyVista's ray_trace als Fallback
                if len(points) == 0:
                    try:
                        pv_points, cell_ids = obstacle_mesh.ray_trace(
                            source_pos_array,
                            target_array,
                            first_point=True,
                        )
                        if len(pv_points) > 0:
                            points = list(pv_points)
                    except Exception:
                        pass
                
                # Wenn es Schnittpunkte gibt, prüfe ob mindestens einer zwischen Quelle und Ziel liegt
                if len(points) > 0:
                    # Konvertiere zu NumPy-Array
                    points_array = np.asarray(points)
                    
                    # Berechne Distanzen von Quelle zu allen Schnittpunkten
                    dists_to_intersections = np.linalg.norm(
                        points_array - source_pos_array.reshape(1, 3), axis=1
                    )
                    
                    # Prüfe ob mindestens ein Schnittpunkt zwischen Quelle und Ziel liegt
                    # Verwende größere Toleranz für dünne Flächen (1cm)
                    valid_intersections = dists_to_intersections < (distance - tolerance)
                    
                    if np.any(valid_intersections):
                        shadow_mask[i] = True
                        if DEBUG_SHADOW and i < 10:  # Erste 10 für Debug
                            min_dist = np.min(dists_to_intersections[valid_intersections])
                            self._log_debug(
                                f"[RayTrace] Punkt {i}: {len(points)} Schnittpunkte gefunden, "
                                f"min bei {min_dist:.3f}m (Distanz: {distance:.3f}m) → Schatten"
                            )
                    elif DEBUG_SHADOW and i < 10:
                        # Schnittpunkte gefunden, aber außerhalb des Strahls
                        self._log_debug(
                            f"[RayTrace] Punkt {i}: {len(points)} Schnittpunkte, "
                            f"aber alle außerhalb (min_dist={np.min(dists_to_intersections):.3f}m, "
                            f"target_dist={distance:.3f}m)"
                        )
                elif DEBUG_SHADOW and i < 10:
                    # Keine Schnittpunkte gefunden
                    self._log_debug(
                        f"[RayTrace] Punkt {i}: Keine Schnittpunkte (Distanz: {distance:.3f}m)"
                    )
                        
            except Exception as exc:
                # Bei Fehler loggen (für Debug)
                if DEBUG_SHADOW and i < 10:
                    self._log_debug(f"[RayTrace] Fehler bei Punkt {i}: {exc}")
                pass
                
        return shadow_mask
    
    def _ray_triangle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_end: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        tolerance: float,
    ) -> Optional[np.ndarray]:
        """
        Möller-Trumbore Ray-Triangle Intersection Algorithmus.
        
        Args:
            ray_origin: Startpunkt des Strahls (3,)
            ray_end: Endpunkt des Strahls (3,)
            v0, v1, v2: Die 3 Eckpunkte des Dreiecks (je 3,)
            tolerance: Numerische Toleranz
            
        Returns:
            Schnittpunkt (3,) oder None wenn kein Schnitt
        """
        # Ray-Richtung
        ray_dir = ray_end - ray_origin
        ray_length = np.linalg.norm(ray_dir)
        if ray_length < tolerance:
            return None
        ray_dir = ray_dir / ray_length
        
        # Dreiecks-Kanten
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Determinante für Backface-Culling
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)
        
        # Strahl ist parallel zum Dreieck
        if abs(a) < tolerance:
            return None
        
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return None
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)
        
        if v < 0.0 or u + v > 1.0:
            return None
        
        # Berechne t (Distanz entlang des Strahls)
        t = f * np.dot(edge2, q)
        
        # Schnittpunkt muss zwischen ray_origin und ray_end liegen
        if t < tolerance or t > ray_length - tolerance:
            return None
        
        # Berechne Schnittpunkt
        intersection = ray_origin + t * ray_dir
        return intersection
    
    def compute_shadow_mask_optimized(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        tolerance: float = 1e-6,
        batch_size: int = 1000,
    ) -> np.ndarray:
        """
        Optimierte Version mit Batch-Verarbeitung für große Punktmengen.
        
        Verarbeitet Punkte in Batches, um Speicher zu sparen und Performance
        zu verbessern.
        
        Args:
            grid_points: (N, 3) Array von 3D-Punkten
            source_positions: Liste von (x, y, z) Tupeln
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            tolerance: Numerische Toleranz
            batch_size: Anzahl Punkte pro Batch
            
        Returns:
            (N,) Boolean-Array: True wenn Punkt im Schatten
        """
        if pv is None:
            return np.zeros(len(grid_points), dtype=bool)
            
        obstacle_mesh = self.build_obstacle_mesh(enabled_surfaces)
        if obstacle_mesh is None:
            return np.zeros(len(grid_points), dtype=bool)
            
        grid_points = np.asarray(grid_points, dtype=float)
        N = len(grid_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        # Verarbeite in Batches
        num_batches = (N + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, N)
            batch_points = grid_points[start_idx:end_idx]
            
            batch_shadow = self.compute_shadow_mask(
                batch_points,
                source_positions,
                enabled_surfaces,
                tolerance,
            )
            
            shadow_mask[start_idx:end_idx] = batch_shadow
            
        return shadow_mask

