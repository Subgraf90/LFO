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
    build_surface_mesh,
    build_vertical_surface_mesh,
)

DEBUG_SHADOW = bool(int(os.environ.get("LFO_DEBUG_SHADOW", "0")))


class ShadowCalculator(ModuleBase):
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
                # Verwende build_surface_mesh für planare Flächen
                mesh = build_surface_mesh(
                    surface_def,
                    resolution=self.settings.resolution,
                    pv_module=pv,
                )
                
                if mesh is not None and hasattr(mesh, 'n_cells') and mesh.n_cells > 0:
                    meshes.append(mesh)
                    if DEBUG_SHADOW:
                        self._log_debug(
                            f"[ShadowCalculator] Surface '{surface_id}': "
                            f"{mesh.n_cells} Zellen hinzugefügt"
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
        tolerance: float = 1e-6,
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
                self._log_debug("[ShadowCalculator] Keine Hindernisse - keine Schatten")
            return np.zeros(len(grid_points), dtype=bool)
            
        # Konvertiere grid_points zu NumPy-Array falls nötig
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError(
                f"grid_points muss Shape (N, 3) haben, bekam {grid_points.shape}"
            )
            
        N = len(grid_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        if DEBUG_SHADOW:
            self._log_debug(
                f"[ShadowCalculator] Berechne Schatten für {N} Punkte, "
                f"{len(source_positions)} Quellen, {obstacle_mesh.n_cells} Hindernis-Zellen"
            )
            
        # Für jede Quelle prüfen wir, welche Punkte im Schatten sind
        for source_idx, source_pos in enumerate(source_positions):
            source_pos = np.asarray(source_pos, dtype=float)
            
            if DEBUG_SHADOW and source_idx == 0:
                self._log_debug(
                    f"[ShadowCalculator] Prüfe Quelle {source_idx+1}/{len(source_positions)} "
                    f"bei ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f})"
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
            
        if DEBUG_SHADOW:
            num_shadow = np.count_nonzero(shadow_mask)
            self._log_debug(
                f"[ShadowCalculator] {num_shadow}/{N} Punkte im Schatten "
                f"({100.0*num_shadow/N:.1f}%)"
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
        
        # Für jeden Strahl prüfen wir, ob er ein Hindernis schneidet
        # PyVista's ray_trace gibt die Schnittpunkte zurück
        for i in range(N):
            target = target_points[i]
            direction = directions[i]
            distance = np.linalg.norm(target - source_pos)
            
            # Werfe Strahl von Quelle zum Zielpunkt
            # PyVista's ray_trace gibt (points, cell_ids) zurück
            try:
                points, cell_ids = obstacle_mesh.ray_trace(
                    source_pos,
                    target,
                    first_point=True,  # Nur ersten Schnittpunkt
                )
                
                # Wenn es einen Schnittpunkt gibt, prüfe ob er zwischen Quelle und Ziel liegt
                if len(points) > 0:
                    intersection = points[0]
                    dist_to_intersection = np.linalg.norm(intersection - source_pos)
                    
                    # Schnittpunkt muss zwischen Quelle und Ziel liegen
                    # (mit Toleranz für numerische Fehler)
                    if dist_to_intersection < distance - tolerance:
                        shadow_mask[i] = True
                        
            except Exception:
                # Bei Fehler nehmen wir an, dass kein Schatten vorliegt
                # (konservativer Ansatz)
                pass
                
        return shadow_mask
    
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

