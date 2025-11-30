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
        
        import time
        mesh_start_time = time.time()
        if DEBUG_SHADOW:
            self._log_debug(f"[ShadowCalculator] Starte Mesh-Erstellung für {len(enabled_surfaces)} Surfaces")
            
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
                
                # 🎯 PRÜFE OB SURFACE VERTIKAL IST (von Z- nach Z+)
                # Extrahiere Koordinaten
                xs = np.array([float(p.get("x", 0.0)) for p in points], dtype=float)
                ys = np.array([float(p.get("y", 0.0)) for p in points], dtype=float)
                zs = np.array([float(p.get("z", 0.0)) for p in points], dtype=float)
                
                x_span = float(np.ptp(xs))
                y_span = float(np.ptp(ys))
                z_span = float(np.ptp(zs))
                
                # Debug: Zeige immer Surface-Informationen
                print(
                    f"[ShadowCalculator] Surface '{surface_id}': "
                    f"{len(points)} Punkte, "
                    f"Spannen: X={x_span:.3f}m, Y={y_span:.3f}m, Z={z_span:.3f}m"
                )
                
                # Prüfe ob Surface vertikal ist oder signifikante Z-Ausdehnung hat:
                # - Z-Spanne ist signifikant größer als XY-Spanne
                # - ODER XY-Spanne ist sehr klein, aber Z-Spanne ist groß
                # - ODER Z-Spanne ist signifikant (> 0.5m) - auch wenn nicht vertikal, sollte als Hindernis behandelt werden
                is_vertical = False
                vertical_reason = ""
                if z_span > 0.01:  # Mindestens 1cm Z-Ausdehnung
                    xy_span = max(x_span, y_span)
                    if xy_span < 1e-6:  # XY ist praktisch eine Linie
                        is_vertical = True
                        vertical_reason = "XY praktisch Linie"
                    elif z_span > 0.5:  # Z-Spanne ist signifikant (> 0.5m) - sollte als Hindernis behandelt werden
                        # Prüfe ob Surface nicht planar ist (kann nicht mit derive_surface_plane modelliert werden)
                        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import derive_surface_plane
                        model, _ = derive_surface_plane(points)
                        if model is None:
                            is_vertical = True
                            vertical_reason = "Nicht planar (derive_surface_plane=None)"
                        elif z_span > xy_span * 0.3:  # Z ist mindestens 30% von XY - auch als Hindernis behandeln
                            is_vertical = True
                            vertical_reason = f"Signifikante Z-Ausdehnung (Z={z_span:.3f}m > 0.5m oder > 30% von XY={xy_span:.3f}m)"
                        else:
                            vertical_reason = f"Planar mit signifikanter Z-Ausdehnung (Z={z_span:.3f}m, XY={xy_span:.3f}m)"
                    elif z_span > xy_span * 0.5:  # Z-Spanne ist mindestens halb so groß wie XY
                        # Prüfe ob Surface nicht planar ist (kann nicht mit derive_surface_plane modelliert werden)
                        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import derive_surface_plane
                        model, _ = derive_surface_plane(points)
                        if model is None:
                            is_vertical = True
                            vertical_reason = "Nicht planar (derive_surface_plane=None)"
                        else:
                            vertical_reason = f"Planar (mode={model.get('mode', 'unknown')})"
                    else:
                        vertical_reason = f"Z-Spanne zu klein relativ zu XY (Z={z_span:.3f}m, XY={xy_span:.3f}m)"
                else:
                    vertical_reason = f"Z-Spanne zu klein ({z_span:.3f}m <= 0.01m)"
                
                # 🎯 OPTIMIERUNG: Verwende REDUZIERTE Auflösung für Hindernis-Mesh
                # für bessere Performance beim Ray Tracing
                # ⚠️ PERFORMANCE: Mindestens 20cm Auflösung für Hindernisse (sonst zu viele Zellen)
                base_resolution = self.settings.resolution
                # Verwende mindestens 20cm für Hindernisse (Performance-Optimierung)
                # Bei sehr vielen Surfaces kann das Mesh sonst > 100k Zellen haben
                obstacle_resolution = max(base_resolution, 0.20)  # Mindestens 20cm
                
                print(
                    f"[ShadowCalculator] Surface '{surface_id}': "
                    f"Vertikal: {is_vertical} ({vertical_reason}), "
                    f"Auflösung: {obstacle_resolution:.3f}m (Basis: {base_resolution:.3f}m)"
                )
                
                if DEBUG_SHADOW:
                    surface_mesh_start = time.time()
                    self._log_debug(
                        f"[ShadowCalculator] Erstelle Mesh für Surface '{surface_id}': "
                        f"{len(points)} Punkte, Resolution={obstacle_resolution:.3f}m, "
                        f"Vertikal: {is_vertical}"
                    )
                
                # Für vertikale Surfaces: Erstelle 3D-Mesh direkt aus Punkten
                if is_vertical:
                    # Erstelle 3D-Mesh aus Polygon-Punkten
                    points_3d = np.array([
                        [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
                        for p in points
                    ], dtype=float)
                    
                    # Erstelle Mesh aus Polygon (Fan-Triangulation)
                    if len(points_3d) >= 3:
                        faces = []
                        # Fan-Triangulation: Verbinde ersten Punkt mit allen anderen
                        for i in range(1, len(points_3d) - 1):
                            faces.extend([3, 0, i, i + 1])
                        
                        if len(faces) > 0:
                            mesh = pv.PolyData(points_3d, faces)
                            
                            # 🎯 WICHTIG: Für Ray-Tracing müssen wir sicherstellen, dass das Mesh
                            # beide Seiten der Fläche hat. Erweitere das Mesh um eine kleine Dicke
                            # in Normalenrichtung, damit PyVista's ray_trace es zuverlässig erkennt.
                            # Alternativ: Verwende explizite Normalen
                            
                            # Berechne Normalenvektor der Fläche
                            if len(points_3d) >= 3:
                                # Verwende erste 3 Punkte für Normalenberechnung
                                v1 = points_3d[1] - points_3d[0]
                                v2 = points_3d[2] - points_3d[0]
                                normal = np.cross(v1, v2)
                                normal_length = np.linalg.norm(normal)
                                if normal_length > 1e-6:
                                    normal = normal / normal_length
                                else:
                                    normal = np.array([0, 0, 1])  # Fallback
                                
                                # Erweitere Mesh um kleine Dicke (1mm) in beide Richtungen
                                # für bessere Ray-Tracing-Erkennung
                                thickness = 0.001  # 1mm
                                
                                # Dupliziere Punkte und verschiebe sie in Normalenrichtung
                                points_front = points_3d + normal * (thickness / 2)
                                points_back = points_3d - normal * (thickness / 2)
                                
                                # Kombiniere Punkte
                                points_thick = np.vstack([points_front, points_back])
                                
                                # Erstelle neue Faces für beide Seiten
                                n_orig = len(points_3d)
                                faces_thick = []
                                
                                # Front-Seite (gleiche Faces wie original)
                                for i in range(1, n_orig - 1):
                                    faces_thick.extend([3, 0, i, i + 1])
                                
                                # Back-Seite (umgekehrte Reihenfolge für korrekte Normalen)
                                for i in range(1, n_orig - 1):
                                    faces_thick.extend([3, n_orig, n_orig + i + 1, n_orig + i])
                                
                                # Seiten-Faces (verbinde Front und Back)
                                for i in range(n_orig):
                                    next_i = (i + 1) % n_orig
                                    faces_thick.extend([4, i, next_i, n_orig + next_i, n_orig + i])
                                
                                try:
                                    mesh = pv.PolyData(points_thick, faces_thick)
                                    if DEBUG_SHADOW:
                                        self._log_debug(
                                            f"[ShadowCalculator] Vertikales Surface '{surface_id}': "
                                            f"Dickes Mesh erstellt: {mesh.n_cells} Zellen, {mesh.n_points} Punkte"
                                        )
                                except Exception as exc:
                                    if DEBUG_SHADOW:
                                        self._log_debug(
                                            f"[ShadowCalculator] Fehler beim Erstellen dicken Meshes: {exc}, "
                                            f"verwende dünnes Mesh"
                                        )
                                    # Fallback: Verwende originales dünnes Mesh
                                    mesh = pv.PolyData(points_3d, faces)
                        else:
                            mesh = None
                    else:
                        mesh = None
                    
                    if DEBUG_SHADOW and mesh is not None:
                        self._log_debug(
                            f"[ShadowCalculator] Vertikales Surface '{surface_id}': "
                            f"3D-Mesh erstellt: {mesh.n_cells} Zellen, {mesh.n_points} Punkte"
                        )
                else:
                    # Normale horizontale/geneigte Fläche: Verwende build_surface_clipping_mesh
                    mesh = build_surface_clipping_mesh(
                        surface_id,
                        points,
                        plane_model,
                        resolution=obstacle_resolution,
                        pv_module=pv,
                    )
                
                if DEBUG_SHADOW:
                    surface_mesh_time = time.time() - surface_mesh_start
                    if mesh is not None:
                        self._log_debug(
                            f"[ShadowCalculator] Surface '{surface_id}' Mesh erstellt: "
                            f"{mesh.n_cells} Zellen, {mesh.n_points} Punkte "
                            f"({surface_mesh_time:.3f}s)"
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
            if DEBUG_SHADOW:
                mesh_time = time.time() - mesh_start_time
                self._log_debug(f"[ShadowCalculator] Keine Meshes erstellt ({mesh_time:.3f}s)")
            return None
            
        # Kombiniere alle Meshes zu einem einzigen Mesh
        try:
            combine_start = time.time()
            combined = meshes[0]
            for mesh in meshes[1:]:
                combined = combined + mesh
                
            combine_time = time.time() - combine_start
            mesh_time = time.time() - mesh_start_time
            
            self._combined_obstacle_mesh = combined
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Kombiniertes Hindernis-Mesh: "
                    f"{combined.n_cells} Zellen, {combined.n_points} Punkte "
                    f"(Kombinieren: {combine_time:.3f}s, Gesamt: {mesh_time:.3f}s)"
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
        resolution_tolerance: Optional[float] = None,  # Resolution-basierte Toleranz für sichtbare Punkte
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
            resolution_tolerance: Wenn gesetzt, werden Schatten-Punkte innerhalb dieser
                Distanz zu sichtbaren Punkten trotzdem berechnet (für bessere Darstellung)
            
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
        import time
        mesh_build_start = time.time()
        print(f"[ShadowCalculator] Starte Hindernis-Mesh-Erstellung für {len(enabled_surfaces)} Surfaces...")
        
        obstacle_mesh = self.build_obstacle_mesh(enabled_surfaces)
        
        mesh_build_time = time.time() - mesh_build_start
        print(f"[ShadowCalculator] Hindernis-Mesh-Erstellung abgeschlossen ({mesh_build_time:.3f}s)")
        
        if obstacle_mesh is None or obstacle_mesh.n_cells == 0:
            print(
                f"[ShadowCalculator] Keine Hindernisse - keine Schatten "
                f"(enabled_surfaces={len(enabled_surfaces)})"
            )
            return np.zeros(len(grid_points), dtype=bool)
        
        print(
            f"[ShadowCalculator] Hindernis-Mesh: {obstacle_mesh.n_cells} Zellen, "
            f"{obstacle_mesh.n_points} Punkte"
        )
            
        # Konvertiere grid_points zu NumPy-Array falls nötig
        grid_points = np.asarray(grid_points, dtype=float)
        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError(
                f"grid_points muss Shape (N, 3) haben, bekam {grid_points.shape}"
            )
            
        N = len(grid_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        # Zeitmessung für Performance-Überwachung
        start_time = time.time()
        
        print(
            f"[ShadowCalculator] ===== SCHATTEN-BERECHNUNG GESTARTET ====="
        )
        print(
            f"[ShadowCalculator] Parameter: tolerance={tolerance:.4f}m, "
            f"resolution_tolerance={'None' if resolution_tolerance is None else f'{resolution_tolerance:.4f}m'}"
        )
        print(
            f"[ShadowCalculator] Berechne Schatten für {N} Punkte, "
            f"{len(source_positions)} Quellen, {obstacle_mesh.n_cells} Hindernis-Zellen, "
            f"{obstacle_mesh.n_points} Hindernis-Punkte"
        )
        # Prüfe ob Mesh gültig ist
        if hasattr(obstacle_mesh, 'bounds'):
            bounds = obstacle_mesh.bounds
            print(
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
            print(
                f"[ShadowCalculator] Grid-Punkte Bounds: "
                f"X=[{grid_bounds[0]:.2f}, {grid_bounds[1]:.2f}], "
                f"Y=[{grid_bounds[2]:.2f}, {grid_bounds[3]:.2f}], "
                f"Z=[{grid_bounds[4]:.2f}, {grid_bounds[5]:.2f}]"
            )
            
        # Für jede Quelle prüfen wir, welche Punkte im Schatten sind
        for source_idx, source_pos in enumerate(source_positions):
            source_pos = np.asarray(source_pos, dtype=float)
            source_start_time = time.time()
            
            print(
                f"[ShadowCalculator] --- Prüfe Quelle {source_idx+1}/{len(source_positions)} "
                f"bei ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}) ---"
            )
            # Teste einen Beispiel-Strahl
            if source_idx == 0 and len(grid_points) > 0:
                test_target = grid_points[0]
                test_dist = np.linalg.norm(test_target - source_pos)
                print(
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
            num_valid = np.count_nonzero(valid_mask)
            print(
                f"[ShadowCalculator] Ray-Trace für {num_valid}/{N} gültige Punkte "
                f"(Distanz > {tolerance:.4f}m)"
            )
            
            ray_trace_start = time.time()
            print(f"[ShadowCalculator] Starte Ray-Trace-Batch...")
            source_shadow = self._ray_trace_batch(
                source_pos,
                grid_points[valid_mask],
                directions[valid_mask],
                obstacle_mesh,
                tolerance,
            )
            ray_trace_time = time.time() - ray_trace_start
            
            num_shadow_found = np.count_nonzero(source_shadow)
            print(
                f"[ShadowCalculator] Ray-Trace abgeschlossen: {num_shadow_found}/{num_valid} Punkte im Schatten "
                f"({ray_trace_time:.3f}s, {num_valid/ray_trace_time:.0f} Punkte/s)"
            )
            
            # Setze Schatten-Maske für diese Quelle
            shadow_mask[valid_mask] = shadow_mask[valid_mask] | source_shadow
            
            source_time = time.time() - source_start_time
            print(
                f"[ShadowCalculator] Quelle {source_idx+1} abgeschlossen ({source_time:.3f}s)"
            )
        
        # 🎯 OPTIMIERUNG: Resolution-basierte Toleranz
        # Wenn ein Punkt im Schatten ist, aber innerhalb der Resolution zu einem sichtbaren Punkt liegt,
        # soll er trotzdem berechnet werden (für bessere Darstellung)
        # ⚠️ PERFORMANCE: Nur bei kleineren Grids aktivieren (< 10000 Punkte)
        # ⚠️ TEMPORÄR DEAKTIVIERT: Kann bei vielen Punkten sehr langsam sein
        resolution_tol_start = time.time()
        resolution_tol_time = 0.0  # Initialisiere für Debug-Ausgabe
        
        # TEMPORÄR DEAKTIVIERT: Resolution-Toleranz-Optimierung
        use_resolution_tolerance = False  # Temporär deaktiviert für Performance
        
        if use_resolution_tolerance and resolution_tolerance is not None and resolution_tolerance > 0 and N < 10000:
            print(f"[ShadowCalculator] ⚠️ Resolution-Toleranz-Optimierung wird übersprungen (temporär deaktiviert)")
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Prüfe Resolution-Toleranz-Optimierung "
                    f"(N={N} < 10000, tolerance={resolution_tolerance:.4f}m)"
                )
            
            visible_points = grid_points[~shadow_mask]
            num_visible = len(visible_points)
            
            if DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Sichtbare Punkte: {num_visible}/{N} "
                    f"({100.0*num_visible/N:.1f}%)"
                )
            
            if num_visible > 0 and num_visible < 5000:  # Nur wenn nicht zu viele sichtbare Punkte
                try:
                    from scipy.spatial import cKDTree
                    
                    if DEBUG_SHADOW:
                        kdtree_start = time.time()
                        self._log_debug(f"[ShadowCalculator] Erstelle KD-Tree für {num_visible} sichtbare Punkte...")
                    
                    # Erstelle KD-Tree für schnelle Nachbarschaftssuche
                    tree = cKDTree(visible_points)
                    
                    kdtree_time = time.time() - kdtree_start
                    if DEBUG_SHADOW:
                        self._log_debug(f"[ShadowCalculator] KD-Tree erstellt ({kdtree_time:.3f}s)")
                    
                    # Finde alle Schatten-Punkte
                    shadow_indices = np.where(shadow_mask)[0]
                    num_shadow = len(shadow_indices)
                    
                    if DEBUG_SHADOW:
                        self._log_debug(f"[ShadowCalculator] Schatten-Punkte: {num_shadow}/{N} ({100.0*num_shadow/N:.1f}%)")
                    
                    if num_shadow > 0 and num_shadow < 5000:  # Nur wenn nicht zu viele Schatten-Punkte
                        shadow_points = grid_points[shadow_indices]
                        
                        if DEBUG_SHADOW:
                            query_start = time.time()
                            self._log_debug(
                                f"[ShadowCalculator] Suche {num_shadow} Schatten-Punkte in KD-Tree "
                                f"(max. Distanz: {resolution_tolerance:.4f}m)..."
                            )
                        
                        # Finde Schatten-Punkte, die innerhalb der Resolution zu sichtbaren Punkten liegen
                        distances, _ = tree.query(shadow_points, k=1, distance_upper_bound=resolution_tolerance)
                        within_resolution = distances <= resolution_tolerance
                        
                        query_time = time.time() - query_start
                        num_kept = np.count_nonzero(within_resolution)
                        
                        if DEBUG_SHADOW:
                            self._log_debug(
                                f"[ShadowCalculator] KD-Tree Query abgeschlossen ({query_time:.3f}s): "
                                f"{num_kept} Punkte innerhalb Toleranz"
                            )
                        
                        # Entferne diese Punkte aus der Schatten-Maske
                        shadow_mask[shadow_indices[within_resolution]] = False
                        
                        if DEBUG_SHADOW:
                            self._log_debug(
                                f"[ShadowCalculator] Resolution-Toleranz: {num_kept} Schatten-Punkte "
                                f"innerhalb {resolution_tolerance:.3f}m zu sichtbaren Punkten → trotzdem berechnet"
                            )
                    elif DEBUG_SHADOW:
                        self._log_debug(
                            f"[ShadowCalculator] Resolution-Toleranz übersprungen: "
                            f"Zu viele Schatten-Punkte ({num_shadow} >= 5000)"
                        )
                except Exception as exc:
                    # Bei Fehler (z.B. scipy nicht verfügbar) einfach überspringen
                    if DEBUG_SHADOW:
                        self._log_debug(f"[ShadowCalculator] Resolution-Toleranz übersprungen: {exc}")
                        import traceback
                        traceback.print_exc()
                    pass
            elif DEBUG_SHADOW:
                self._log_debug(
                    f"[ShadowCalculator] Resolution-Toleranz übersprungen: "
                    f"Zu viele sichtbare Punkte ({num_visible} >= 5000) oder keine sichtbaren Punkte"
                )
        else:
            if resolution_tolerance is None or resolution_tolerance <= 0:
                print(f"[ShadowCalculator] Resolution-Toleranz deaktiviert (tolerance=None oder <=0)")
            elif N >= 10000:
                print(f"[ShadowCalculator] Resolution-Toleranz übersprungen: Grid zu groß (N={N} >= 10000)")
            elif not use_resolution_tolerance:
                print(f"[ShadowCalculator] Resolution-Toleranz übersprungen: Temporär deaktiviert")
        
        resolution_tol_time = time.time() - resolution_tol_start
        print(f"[ShadowCalculator] Resolution-Toleranz-Optimierung: {resolution_tol_time:.3f}s")
            
        elapsed_time = time.time() - start_time
        num_shadow = np.count_nonzero(shadow_mask)
        
        print(
            f"[ShadowCalculator] ===== SCHATTEN-BERECHNUNG ABGESCHLOSSEN ====="
        )
        print(
            f"[ShadowCalculator] Ergebnis: {num_shadow}/{N} Punkte im Schatten "
            f"({100.0*num_shadow/N:.1f}%)"
        )
        print(
            f"[ShadowCalculator] Zeitaufwand: Gesamt={elapsed_time:.3f}s, "
            f"Mesh-Erstellung={mesh_build_time:.3f}s, "
            f"Ray-Tracing={elapsed_time-mesh_build_time-resolution_tol_time:.3f}s, "
            f"Resolution-Toleranz={resolution_tol_time:.3f}s"
        )
        print(
            f"[ShadowCalculator] Performance: {N/elapsed_time:.0f} Punkte/s"
        )
        
        # Warnung bei langen Berechnungen (>1s)
        if elapsed_time > 1.0:
            print(
                f"[ShadowCalculator] ⚠️ Ray-Tracing dauerte {elapsed_time:.2f}s "
                f"({num_shadow}/{N} Punkte im Schatten)"
            )
            # Zeige Details für erste Schatten-Punkte
            if num_shadow > 0:
                shadow_indices = np.where(shadow_mask)[0][:5]  # Erste 5
                print(
                    f"[ShadowCalculator] Erste Schatten-Punkte (Indizes): {shadow_indices.tolist()}"
                )
            else:
                print(
                    f"[ShadowCalculator] ⚠️ WARNUNG: Keine Schatten-Punkte gefunden! "
                    f"Prüfe ob Hindernisse zwischen Quellen und Punkten liegen."
                )
            
        return shadow_mask
    
    def compute_shadow_mask_for_calculation(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Berechnet eine Schatten-Maske für die Berechnung.
        
        WICHTIG: Nur Punkte TIEF IM SCHATTEN werden deaktiviert.
        Randpunkte (die benachbarte sichtbare Punkte haben) werden TROTZDEM berechnet.
        
        Args:
            grid_points: (N, 3) Array von 3D-Punkten [x, y, z]
            source_positions: Liste von (x, y, z) Tupeln für Schallquellen
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            X_grid: (ny, nx) Array mit X-Koordinaten (für Nachbarschaftsprüfung)
            Y_grid: (ny, nx) Array mit Y-Koordinaten (für Nachbarschaftsprüfung)
            tolerance: Numerische Toleranz für Ray-Tests (in Metern)
            
        Returns:
            (N,) Boolean-Array: True wenn Punkt im Schatten UND tief im Schatten (keine Randpunkte)
        """
        # Berechne vollständige Schatten-Maske
        full_shadow_mask = self.compute_shadow_mask(
            grid_points,
            source_positions,
            enabled_surfaces,
            tolerance=tolerance,
            resolution_tolerance=None,  # Keine Resolution-Toleranz für Berechnung
        )
        
        # 🎯 ERODIERE MASKE: Entferne Randpunkte (die benachbarte sichtbare Punkte haben)
        # Nur Punkte tief im Schatten werden deaktiviert
        # TEMPORÄR: Erosion deaktiviert, da zu aggressiv
        # TODO: Weniger aggressive Erosion implementieren
        num_full = np.count_nonzero(full_shadow_mask)
        
        # TEMPORÄR DEAKTIVIERT: Erosion entfernt zu viele Punkte
        use_erosion = False  # Temporär deaktiviert
        
        if use_erosion and num_full > 10:
            shadow_mask_eroded = self._erode_shadow_mask(
                full_shadow_mask,
                X_grid,
                Y_grid,
            )
            num_eroded = np.count_nonzero(shadow_mask_eroded)
            num_edge = num_full - num_eroded
            
            print(
                f"[ShadowCalculator] Berechnungs-Maske: {num_eroded}/{len(grid_points)} Punkte tief im Schatten "
                f"({100.0*num_eroded/len(grid_points):.1f}%), "
                f"{num_edge} Randpunkte werden trotzdem berechnet"
            )
        else:
            # Keine Erosion: Verwende vollständige Schatten-Maske
            shadow_mask_eroded = full_shadow_mask
            if use_erosion:
                print(
                    f"[ShadowCalculator] Berechnungs-Maske: {num_full}/{len(grid_points)} Punkte im Schatten "
                    f"({100.0*num_full/len(grid_points):.1f}%) - Erosion übersprungen (zu wenige Punkte)"
                )
            else:
                print(
                    f"[ShadowCalculator] Berechnungs-Maske: {num_full}/{len(grid_points)} Punkte im Schatten "
                    f"({100.0*num_full/len(grid_points):.1f}%) - Erosion deaktiviert"
                )
        
        return shadow_mask_eroded
    
    def _erode_shadow_mask(
        self,
        shadow_mask: np.ndarray,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Erodiert die Schatten-Maske, um Randpunkte zu entfernen.
        
        Ein Punkt wird als Randpunkt betrachtet, wenn er im Schatten ist,
        aber mindestens ein benachbarter Punkt (oben, unten, links, rechts) sichtbar ist.
        
        Args:
            shadow_mask: (N,) Boolean-Array, True = im Schatten
            X_grid: (ny, nx) Array mit X-Koordinaten
            Y_grid: (ny, nx) Array mit Y-Koordinaten
            
        Returns:
            (N,) Boolean-Array: True nur für Punkte tief im Schatten (keine Randpunkte)
        """
        # Reshape Maske zu Grid-Form
        ny, nx = X_grid.shape
        shadow_mask_2d = shadow_mask.reshape(ny, nx)
        
        # Erstelle erodierte Maske (Kopie)
        shadow_mask_eroded = shadow_mask_2d.copy()
        
        # Iteriere über alle Punkte im Schatten
        for j in range(ny):
            for i in range(nx):
                if not shadow_mask_2d[j, i]:
                    continue  # Punkt ist nicht im Schatten
                
                # Prüfe Nachbarn (oben, unten, links, rechts)
                # Ein Punkt ist ein Randpunkt, wenn MINDESTENS EIN Nachbar sichtbar ist
                # ABER: Nur entfernen, wenn nicht ALLE Nachbarn im Schatten sind
                # (sonst wäre es zu aggressiv)
                visible_neighbors = 0
                total_neighbors = 0
                
                # Oben
                if j > 0:
                    total_neighbors += 1
                    if not shadow_mask_2d[j-1, i]:
                        visible_neighbors += 1
                # Unten
                if j < ny-1:
                    total_neighbors += 1
                    if not shadow_mask_2d[j+1, i]:
                        visible_neighbors += 1
                # Links
                if i > 0:
                    total_neighbors += 1
                    if not shadow_mask_2d[j, i-1]:
                        visible_neighbors += 1
                # Rechts
                if i < nx-1:
                    total_neighbors += 1
                    if not shadow_mask_2d[j, i+1]:
                        visible_neighbors += 1
                
                # Nur entfernen, wenn mindestens ein Nachbar sichtbar ist
                # UND nicht alle Nachbarn im Schatten sind (sonst wäre es zu aggressiv)
                if visible_neighbors > 0 and visible_neighbors < total_neighbors:
                    shadow_mask_eroded[j, i] = False
        
        # Reshape zurück zu 1D
        return shadow_mask_eroded.reshape(-1)
    
    def compute_high_res_shadow_mask_for_render(
        self,
        render_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Berechnet eine hochauflösende Schatten-Maske für das Rendering.
        
        Diese Methode wird für Pixel-Level-Schatten-Berechnung verwendet,
        um zu bestimmen, welche Pixel effektiv im Schatten sind.
        
        Args:
            render_points: (N, 3) Array von 3D-Punkten für Rendering (hochauflösend)
            source_positions: Liste von (x, y, z) Tupeln für Schallquellen
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            tolerance: Numerische Toleranz für Ray-Tests (in Metern)
            
        Returns:
            (N,) Boolean-Array: True wenn Pixel im Schatten
        """
        # Verwende die normale compute_shadow_mask, aber mit höherer Auflösung
        # (keine Erosion, alle Punkte im Schatten werden markiert)
        shadow_mask = self.compute_shadow_mask(
            render_points,
            source_positions,
            enabled_surfaces,
            tolerance=tolerance,
            resolution_tolerance=None,  # Keine Resolution-Toleranz für Rendering
        )
        
        num_shadow = np.count_nonzero(shadow_mask)
        print(
            f"[ShadowCalculator] Render-Schatten: {num_shadow}/{len(render_points)} Pixel im Schatten "
            f"({100.0*num_shadow/len(render_points):.1f}%)"
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
        
        import time
        batch_start_time = time.time()
        
        # Initialisiere Debug-Zähler
        self._debug_intersection_count = 0
        self._debug_invalid_count = 0
        
        print(
            f"[RayTrace] Starte Batch-Ray-Tracing: {N} Strahlen, "
            f"{obstacle_mesh.n_cells} Hindernis-Zellen"
        )
        
        # 🚀 OPTIMIERUNG 1: Hole Mesh-Bounds einmalig
        if not hasattr(obstacle_mesh, 'bounds') or obstacle_mesh.n_cells == 0:
            if DEBUG_SHADOW:
                self._log_debug("[RayTrace] Keine Hindernis-Zellen → keine Schatten")
            return shadow_mask
        
        mesh_bounds = obstacle_mesh.bounds
        mesh_min = np.array([mesh_bounds[0], mesh_bounds[2], mesh_bounds[4]])
        mesh_max = np.array([mesh_bounds[1], mesh_bounds[3], mesh_bounds[5]])
        
        if DEBUG_SHADOW:
            self._log_debug(
                f"[RayTrace] Mesh-Bounds: "
                f"X=[{mesh_min[0]:.2f}, {mesh_max[0]:.2f}], "
                f"Y=[{mesh_min[1]:.2f}, {mesh_max[1]:.2f}], "
                f"Z=[{mesh_min[2]:.2f}, {mesh_max[2]:.2f}]"
            )
        
        # 🚀 OPTIMIERUNG: Verwende PyVista's optimiertes ray_trace
        # → Kein Precomputing nötig, PyVista macht das intern optimiert
        # Precompute nur als Fallback für manuelles Ray-Triangle-Intersection
        cell_data = []  # Wird nur bei Fehler von PyVista's ray_trace verwendet
        use_pyvista_raytrace = True  # Primäre Methode
        
        if not use_pyvista_raytrace:
            # Fallback: Precompute nur wenn PyVista nicht verwendet wird
            precompute_start = time.time()
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
            
            precompute_time = time.time() - precompute_start
            print(
                f"[RayTrace] Precompute abgeschlossen: {len(cell_data)} Zellen "
                f"({precompute_time:.3f}s)"
            )
        else:
            print(
                f"[RayTrace] Verwende PyVista's optimiertes ray_trace "
                f"(kein Precomputing nötig)"
            )
        
        # 🚀 OPTIMIERUNG 3: Prüfe zuerst, ob Strahl überhaupt durch Mesh-Bounds geht
        source_pos_array = np.asarray(source_pos)
        
        rays_through_bounds = 0
        rays_checked = 0
        intersections_found = 0
        
        # 🚀 PERFORMANCE: Batch-Verarbeitung für bessere Performance
        # Verarbeite Strahlen in Batches, um Fortschritt zu zeigen
        batch_size = max(100, N // 10)  # 10 Batches oder min. 100 Strahlen pro Batch
        
        print(f"[RayTrace] Verarbeite {N} Strahlen in Batches von {batch_size}...")
        
        # Für jeden Strahl prüfen wir, ob er ein Hindernis schneidet
        for i in range(N):
            # Zeige Fortschritt bei jedem Batch
            if i > 0 and i % batch_size == 0:
                progress = 100.0 * i / N
                elapsed = time.time() - batch_start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (N - i) / rate if rate > 0 else 0
                print(
                    f"[RayTrace] Fortschritt: {i}/{N} ({progress:.1f}%), "
                    f"{rate:.0f} Strahlen/s, ~{remaining:.1f}s verbleibend"
                )
            
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
            
            rays_through_bounds += 1
            
            # Werfe Strahl von Quelle zum Zielpunkt
            points = []
            try:
                    # 🚀 PERFORMANCE: Verwende PyVista's optimiertes ray_trace als primäre Methode
                # (ist viel schneller als manuelles Iterieren über alle Zellen)
                try:
                    # 🎯 WICHTIG: PyVista's ray_trace gibt Schnittpunkte entlang des Strahls zurück
                    # first_point=False gibt ALLE Schnittpunkte zurück (nicht nur den ersten)
                    # Das ist wichtig, da wir den Schnittpunkt VOR dem Ziel finden müssen
                    pv_points, cell_ids = obstacle_mesh.ray_trace(
                        source_pos_array,
                        target_array,
                        first_point=False,  # Hole ALLE Schnittpunkte, nicht nur den ersten
                    )
                    
                    # Debug: Zeige alle Schnittpunkte für erste Strahlen
                    if len(pv_points) > 0 and self._debug_intersection_count < 5:
                        print(
                            f"[RayTrace] Strahl {i}: {len(pv_points)} Schnittpunkte von PyVista gefunden"
                        )
                        for j, pv_pt in enumerate(pv_points):
                            pt_dist = np.linalg.norm(pv_pt - source_pos_array)
                            print(
                                f"  PyVista Schnittpunkt {j}: Pos=({pv_pt[0]:.3f}, {pv_pt[1]:.3f}, {pv_pt[2]:.3f}), "
                                f"Dist={pt_dist:.3f}m, Ziel-Dist={distance:.3f}m, "
                                f"Diff={abs(pt_dist - distance):.4f}m"
                            )
                    
                    if len(pv_points) > 0:
                        points = list(pv_points)
                        intersections_found += len(points)
                    
                    # 🎯 ZUSÄTZLICH: Prüfe IMMER manuell, ob Strahl die Z=0-Ebene schneidet
                    # (PyVista gibt möglicherweise nicht alle Schnittpunkte zurück, besonders für 2D-Flächen)
                    # Wenn Ziel über der Hindernis-Fläche liegt (Z>0), sollte Strahl die Fläche bei Z=0 schneiden
                    ray_dir = target_array - source_pos_array
                    z0_tolerance = max(tolerance, 0.02)  # Toleranz für Z=0-Schnittpunkt
                    
                    # Prüfe IMMER, ob Strahl die Z=0-Ebene schneidet (auch wenn PyVista nichts findet)
                    # WICHTIG: Nur wenn Quelle über Z=0 liegt UND Ziel über oder auf Z=0 liegt
                    source_z = source_pos_array[2]
                    target_z = target_array[2]
                    
                    if abs(ray_dir[2]) > 1e-6:  # Strahl ist nicht parallel zur Z-Ebene
                        # Berechne Schnittpunkt mit Z=0-Ebene
                        # Parametrische Form: point = source + t * ray_dir
                        # Z = 0: source[2] + t * ray_dir[2] = 0
                        t = -source_z / ray_dir[2]
                        
                        # Prüfe ob Schnittpunkt zwischen Quelle und Ziel liegt
                        # WICHTIG: t muss zwischen 0 und 1 liegen (zwischen Quelle und Ziel)
                        if tolerance < t < (1 - tolerance):  # Zwischen Quelle und Ziel (mit Toleranz)
                            intersection_z0 = source_pos_array + t * ray_dir
                            intersection_dist = np.linalg.norm(intersection_z0 - source_pos_array)
                            
                            # Prüfe ob Schnittpunkt innerhalb der Hindernis-Mesh-Bounds liegt
                            if (
                                mesh_min[0] <= intersection_z0[0] <= mesh_max[0] and
                                mesh_min[1] <= intersection_z0[1] <= mesh_max[1] and
                                abs(intersection_z0[2]) < 0.01  # Z≈0 (1cm Toleranz)
                            ):
                                # Prüfe ob Schnittpunkt deutlich vor dem Ziel liegt
                                if intersection_dist < (distance - z0_tolerance):
                                    # Prüfe ob dieser Schnittpunkt noch nicht in der Liste ist (von PyVista)
                                    is_duplicate = False
                                    for existing_pt in points:
                                        if np.linalg.norm(existing_pt - intersection_z0) < z0_tolerance:
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        points.append(intersection_z0)
                                        intersections_found += 1
                                        
                                        if self._debug_intersection_count < 5:
                                            print(
                                                f"  ✅ Manueller Z=0 Schnittpunkt: Pos=({intersection_z0[0]:.3f}, "
                                                f"{intersection_z0[1]:.3f}, {intersection_z0[2]:.3f}), "
                                                f"Dist={intersection_dist:.3f}m, Ziel-Dist={distance:.3f}m, "
                                                f"Ziel-Z={target_z:.3f}m, Quelle-Z={source_z:.3f}m, t={t:.3f}"
                                            )
                                elif self._debug_intersection_count < 5 and target_z > 0.01:
                                    # Nur Debug für Punkte über Z=0
                                    print(
                                        f"  ⚠️ Z=0 Schnittpunkt zu nah am Ziel: Dist={intersection_dist:.3f}m, "
                                        f"Ziel-Dist={distance:.3f}m, Diff={abs(intersection_dist - distance):.4f}m, "
                                        f"Ziel-Z={target_z:.3f}m"
                                    )
                        elif self._debug_intersection_count < 5 and target_z > 0.01:
                            # Debug: Warum wird Z=0-Schnittpunkt nicht verwendet?
                            print(
                                f"  ⚠️ Z=0 Schnittpunkt außerhalb Strahl: t={t:.3f} "
                                f"(muss zwischen {tolerance:.3f} und {1-tolerance:.3f} liegen), "
                                f"Quelle-Z={source_z:.3f}m, Ziel-Z={target_z:.3f}m"
                            )
                        
                        # Debug: Zeige Details für ALLE Strahlen mit Schnittpunkten (begrenzt auf erste 10)
                        if len(pv_points) > 0:
                            # Zähle wie viele Strahlen mit Schnittpunkten wir bereits gesehen haben
                            if not hasattr(self, '_debug_intersection_count'):
                                self._debug_intersection_count = 0
                            
                            if self._debug_intersection_count < 10:  # Zeige Details für erste 10 Strahlen mit Schnittpunkten
                                print(
                                    f"[RayTrace] Strahl {i}: {len(pv_points)} Schnittpunkte gefunden, "
                                    f"Ziel=({target_array[0]:.3f}, {target_array[1]:.3f}, {target_array[2]:.3f}), "
                                    f"Ziel-Distanz={distance:.3f}m"
                                )
                                for j, pv_pt in enumerate(pv_points[:3]):  # Erste 3 Schnittpunkte
                                    pt_dist = np.linalg.norm(pv_pt - source_pos_array)
                                    # Berechne ob Schnittpunkt zwischen Quelle und Ziel liegt
                                    vec_to_pt = pv_pt - source_pos_array
                                    vec_to_target = target_array - source_pos_array
                                    # Prüfe ob Schnittpunkt in Richtung des Ziels liegt
                                    dot_product = np.dot(vec_to_pt, vec_to_target)
                                    is_between = (pt_dist > tolerance) and (pt_dist < (distance - 0.02))
                                    
                                    print(
                                        f"[RayTrace]   Schnittpunkt {j}: "
                                        f"Pos=({pv_pt[0]:.3f}, {pv_pt[1]:.3f}, {pv_pt[2]:.3f}), "
                                        f"Dist={pt_dist:.3f}m, Dot={dot_product:.3f}, "
                                        f"Zwischen={is_between}"
                                    )
                                self._debug_intersection_count += 1
                except Exception:
                    # Fallback: Manuelles Ray-Triangle-Intersection (nur bei Fehler)
                    # 🚀 OPTIMIERUNG: Iteriere nur über Zellen, die Strahl-Bounding-Box schneiden
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
                            intersections_found += 1
                            # Stoppe bei erstem Schnittpunkt (für Schatten reicht das)
                            break
                
                # Wenn es Schnittpunkte gibt, prüfe ob mindestens einer zwischen Quelle und Ziel liegt
                if len(points) > 0:
                    # Konvertiere zu NumPy-Array
                    points_array = np.asarray(points)
                    
                    # Berechne Distanzen von Quelle zu allen Schnittpunkten
                    dists_to_intersections = np.linalg.norm(
                        points_array - source_pos_array.reshape(1, 3), axis=1
                    )
                    
                    # Prüfe ob mindestens ein Schnittpunkt zwischen Quelle und Ziel liegt
                    # 🎯 WICHTIG: Ein Punkt ist im Schatten, wenn ein Hindernis ZWISCHEN Quelle und Ziel liegt
                    # 
                    # Logik:
                    # - Wenn Schnittpunkt VOR dem Ziel liegt → Schatten (Hindernis blockiert)
                    # - Wenn Schnittpunkt GENAU am Ziel liegt → KEIN Schatten (Ziel liegt auf Hindernis-Fläche, ist sichtbar)
                    # - Wenn Schnittpunkt HINTER dem Ziel liegt → KEIN Schatten (Hindernis ist hinter dem Ziel)
                    #
                    # PyVista's ray_trace gibt Schnittpunkte zurück, die:
                    # - Auf dem Strahl liegen (von Quelle zu Ziel)
                    # - Mit dem Mesh schneiden
                    # - Können vor, am oder hinter dem Ziel liegen
                    #
                    intersection_tolerance = max(tolerance, 0.02)  # Mindestens 2cm Toleranz für numerische Ungenauigkeiten
                    min_distance_from_source = tolerance  # Mindestens tolerance von der Quelle entfernt
                    
                    # Berechne Strahl-Richtung für manuelle Z=0-Prüfung (wird oben verwendet)
                    ray_dir = target_array - source_pos_array
                    
                    # Bedingung: Schnittpunkt liegt ZWISCHEN Quelle und Ziel (nicht am oder hinter dem Ziel)
                    # 
                    # WICHTIG: Wenn das Ziel auf der Hindernis-Fläche liegt (Z=0), gibt PyVista den Schnittpunkt
                    # genau am Ziel zurück. Das ist korrekt - der Punkt liegt auf der Fläche, nicht im Schatten.
                    #
                    # ABER: Wenn das Ziel ÜBER der Hindernis-Fläche liegt (z.B. Z>0), sollte der Strahl
                    # die Fläche bei Z=0 schneiden, VOR dem Ziel. Das sollte als Schatten erkannt werden.
                    #
                    # Daher: Prüfe ob Schnittpunkt DEUTLICH vor dem Ziel liegt (mit Toleranz für numerische Ungenauigkeiten)
                    # ODER ob Schnittpunkt sehr nah am Ziel liegt UND das Ziel auf der Hindernis-Fläche liegt
                    
                    # 🎯 WICHTIG: Prüfe ob Ziel auf Hindernis-Fläche liegt
                    # Wenn das Ziel auf dem Hindernis liegt, soll es NICHT als Schatten markiert werden
                    # (Das Hindernis selbst soll nicht deaktiviert werden)
                    
                    # Prüfe ob Ziel sehr nah an einem Schnittpunkt liegt (innerhalb Toleranz)
                    # Das bedeutet, dass das Ziel auf der Hindernis-Fläche liegt
                    target_on_obstacle = False
                    if len(dists_to_intersections) > 0:
                        # Prüfe ob mindestens ein Schnittpunkt sehr nah am Ziel liegt
                        dist_to_target = np.abs(dists_to_intersections - distance)
                        target_on_obstacle = np.any(dist_to_target < intersection_tolerance)
                    
                    # Bedingung für gültige Schnittpunkte:
                    # Ein Schnittpunkt ist gültig, wenn er:
                    # 1. Nicht direkt an der Quelle liegt
                    # 2. DEUTLICH vor dem Ziel liegt (mit Toleranz)
                    # 3. Das Ziel NICHT auf dem Hindernis liegt (sonst wäre es kein Schatten, sondern das Hindernis selbst)
                    if target_on_obstacle:
                        # Ziel liegt auf Hindernis → kein Schatten
                        valid_intersections = np.zeros(len(dists_to_intersections), dtype=bool)
                    else:
                        # Ziel liegt nicht auf Hindernis → prüfe normale Schatten-Bedingungen
                        valid_intersections = (
                            (dists_to_intersections > min_distance_from_source) &  # Nicht direkt an der Quelle
                            (dists_to_intersections < (distance - intersection_tolerance))  # DEUTLICH vor dem Ziel (mit Toleranz)
                        )
                    
                    if np.any(valid_intersections):
                        shadow_mask[i] = True
                        if i < 10:  # Erste 10 für Debug
                            min_dist = np.min(dists_to_intersections[valid_intersections])
                            print(
                                f"[RayTrace] Punkt {i}: {len(points)} Schnittpunkte gefunden, "
                                f"min bei {min_dist:.3f}m (Distanz: {distance:.3f}m) → Schatten"
                            )
                    else:
                        # Schnittpunkte gefunden, aber außerhalb des Strahls
                        # Zeige Details nur für erste 10 ungültige Schnittpunkte
                        if not hasattr(self, '_debug_invalid_count'):
                            self._debug_invalid_count = 0
                        
                        if self._debug_invalid_count < 10:
                            min_dist = np.min(dists_to_intersections) if len(dists_to_intersections) > 0 else 0
                            max_dist = np.max(dists_to_intersections) if len(dists_to_intersections) > 0 else 0
                            
                            # Zeige Details für erste Schnittpunkte
                            print(
                                f"[RayTrace] Punkt {i}: {len(points)} Schnittpunkte, aber alle ungültig:\n"
                                f"  Ziel=({target_array[0]:.3f}, {target_array[1]:.3f}, {target_array[2]:.3f}), "
                                f"Ziel-Distanz={distance:.3f}m\n"
                                f"  Schnittpunkt-Distanzen: min={min_dist:.3f}m, max={max_dist:.3f}m\n"
                                f"  Toleranz: {intersection_tolerance:.3f}m, Min-Dist von Quelle: {min_distance_from_source:.3f}m"
                            )
                            # Zeige Details für erste 3 Schnittpunkte
                            for idx in range(min(3, len(points_array))):
                                pt = points_array[idx]
                                dist = dists_to_intersections[idx]
                                print(
                                    f"  Schnittpunkt {idx}: Pos=({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}), "
                                    f"Dist={dist:.3f}m, "
                                    f"Vor Ziel? {dist < (distance - intersection_tolerance)}, "
                                    f"Nach Quelle? {dist > min_distance_from_source}"
                                )
                            self._debug_invalid_count += 1
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
            
            rays_checked += 1
        
        batch_time = time.time() - batch_start_time
        num_shadow = np.count_nonzero(shadow_mask)
        
        print(
            f"[RayTrace] Batch-Ray-Tracing abgeschlossen: "
            f"{num_shadow}/{N} Strahlen blockiert ({100.0*num_shadow/N:.1f}%)"
        )
        print(
            f"[RayTrace] Statistiken: {rays_through_bounds}/{N} Strahlen durch Bounds, "
            f"{rays_checked} Zellen geprüft, {intersections_found} Schnittpunkte gefunden"
        )
        print(
            f"[RayTrace] Performance: {N/batch_time:.0f} Strahlen/s "
            f"({batch_time:.3f}s gesamt)"
        )
                
        return shadow_mask
    
    def compute_shadow_mask_high_resolution(
        self,
        grid_points: np.ndarray,
        source_positions: List[Tuple[float, float, float]],
        enabled_surfaces: List[Tuple[str, Dict[str, Any]]],
        render_resolution: float = 0.01,  # Hochauflösung für Render (1cm)
    ) -> np.ndarray:
        """
        Berechnet eine hochauflösende Schatten-Maske für den Render.
        
        Diese Methode verwendet eine höhere Auflösung für das Hindernis-Mesh
        und eine feinere Toleranz für präzisere Schattenberechnung beim Rendering.
        
        Args:
            grid_points: (N, 3) Array von 3D-Punkten [x, y, z]
            source_positions: Liste von (x, y, z) Tupeln für Schallquellen
            enabled_surfaces: Liste von (surface_id, surface_dict) Tupeln
            render_resolution: Auflösung für Render-Mesh (Standard: 1cm)
            
        Returns:
            (N,) Boolean-Array: True wenn Punkt im Schatten, False wenn sichtbar
        """
        if pv is None:
            return np.zeros(len(grid_points), dtype=bool)
            
        if not source_positions or not enabled_surfaces:
            return np.zeros(len(grid_points), dtype=bool)
        
        # Erstelle hochauflösendes Hindernis-Mesh
        meshes = []
        for surface_id, surface_dict in enabled_surfaces:
            if not surface_dict.get("enabled", False) or surface_dict.get("hidden", False):
                continue
                
            points = surface_dict.get("points", [])
            plane_model = surface_dict.get("plane_model")
            
            if len(points) < 3:
                continue
                
            try:
                # Verwende höhere Auflösung für Render
                mesh = build_surface_clipping_mesh(
                    surface_id,
                    points,
                    plane_model,
                    resolution=render_resolution,
                    pv_module=pv,
                )
                
                if mesh is not None and hasattr(mesh, 'n_cells') and mesh.n_cells > 0:
                    if mesh.n_points >= 3:
                        meshes.append(mesh)
            except Exception:
                continue
        
        if not meshes:
            return np.zeros(len(grid_points), dtype=bool)
        
        # Kombiniere Meshes
        try:
            obstacle_mesh = meshes[0]
            for mesh in meshes[1:]:
                obstacle_mesh = obstacle_mesh + mesh
        except Exception:
            return np.zeros(len(grid_points), dtype=bool)
        
        # Berechne Schatten mit feinerer Toleranz
        grid_points = np.asarray(grid_points, dtype=float)
        N = len(grid_points)
        shadow_mask = np.zeros(N, dtype=bool)
        
        # Feinere Toleranz für Render
        render_tolerance = render_resolution * 0.1  # 10% der Render-Resolution
        
        for source_pos in source_positions:
            source_pos = np.asarray(source_pos, dtype=float)
            directions = grid_points - source_pos[None, :]
            distances = np.linalg.norm(directions, axis=1)
            
            valid_mask = distances > render_tolerance
            if not np.any(valid_mask):
                continue
                
            directions[valid_mask] = directions[valid_mask] / distances[valid_mask, None]
            
            source_shadow = self._ray_trace_batch(
                source_pos,
                grid_points[valid_mask],
                directions[valid_mask],
                obstacle_mesh,
                render_tolerance,
            )
            
            shadow_mask[valid_mask] = shadow_mask[valid_mask] | source_shadow
        
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
        # 🎯 OPTIMIERUNG: Erhöhte Toleranz für schräge Flächen
        parallel_tolerance = max(tolerance, 1e-6)  # Mindestens 1µm Toleranz
        if abs(a) < parallel_tolerance:
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

