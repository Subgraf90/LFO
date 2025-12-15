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
import time
import numpy as np
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Init.Logging import PERF_ENABLED, measure_time, perf_section
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
    SurfaceDefinition,
    _evaluate_plane_on_grid,
)
from Module_LFO.Modules_Data.SurfaceValidator import triangulate_points

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
    dominant_axis: Optional[str] = None  # "xz", "yz" oder None (für vertikale Flächen)


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
    # 🎯 NEU: Triangulierte Daten für Plotting
    triangulated_vertices: Optional[np.ndarray] = None  # Shape: (N, 3) - Vertex-Koordinaten
    triangulated_faces: Optional[np.ndarray] = None  # Shape: (M, 3) - Face-Indices
    triangulated_success: bool = False  # Ob Triangulation erfolgreich war
    # 🎯 NEU: Mapping von Vertex → Quell-Grid-Index (für Color-Step ohne Interpolation)
    # Länge == Anzahl Vertices (oder None, wenn nicht verfügbar)
    vertex_source_indices: Optional[np.ndarray] = None


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
    
    def _compute_robust_plane_normal_svd(
        self,
        points: List[Dict[str, float]]
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[Dict[str, float]]]:
        """
        🎯 ROBUSTESTE METHODE: SVD-basiertes Least-Squares Plane Fitting.
        
        Funktioniert für:
        - 3 bis beliebig viele Punkte
        - Beliebige Orientierung im Raum
        - Ausreißer-resistent
        - Numerisch stabil
        
        Returns:
            (normal, plane_fit_error, plane_params)
            - normal: Normale der Ebene (normalisiert)
            - plane_fit_error: RMS-Fehler der Ebenen-Anpassung
            - plane_params: Dict mit {a, b, c, d} für ax + by + cz + d = 0
        """
        if len(points) < 3:
            return None, None, None
        
        coords = np.array([
            [float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0))]
            for p in points
        ], dtype=float)
        
        n_points = coords.shape[0]
        
        # Für genau 3 Punkte: Direkte Normale über Kreuzprodukt
        if n_points == 3:
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                return None, None, None
            normal = normal / norm
            
            # Berechne d für ax + by + cz + d = 0
            # Verwende ersten Punkt
            d = -np.dot(normal, coords[0])
            
            # RMS-Fehler (sollte 0 sein für 3 Punkte)
            errors = np.abs(np.dot(coords, normal) + d)
            rms_error = float(np.sqrt(np.mean(errors**2)))
            
            return normal, rms_error, {
                'a': float(normal[0]),
                'b': float(normal[1]),
                'c': float(normal[2]),
                'd': float(d)
            }
        
        # Für 4+ Punkte: SVD-basiertes Least-Squares Plane Fitting
        # Methode: ax + by + cz + d = 0
        # Zentriere Punkte für bessere numerische Stabilität
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        # Prüfe auf degenerierte Fälle
        if np.allclose(coords_centered, 0, atol=1e-9):
            return None, None, None
        
        # SVD: coords_centered = U * S * V^T
        # Die Normale ist der rechte Singulärvektor zum kleinsten Singulärwert
        try:
            U, s, Vt = np.linalg.svd(coords_centered, full_matrices=False)
            
            # Kleinster Singulärwert → Normale der Ebene
            # Die Normale ist die letzte Zeile von Vt (entspricht kleinstem Singulärwert)
            normal = Vt[-1, :]
            norm = np.linalg.norm(normal)
            
            if norm < 1e-9:
                return None, None, None
            
            normal = normal / norm
            
            # Stelle sicher, dass Normale konsistent orientiert ist
            # (zeige in Richtung mit positiver Z-Komponente, wenn möglich)
            if normal[2] < 0:
                normal = -normal
            
            # Berechne d für ax + by + cz + d = 0
            # Verwende Zentroid
            d = -np.dot(normal, center)
            
            # Berechne RMS-Fehler der Ebenen-Anpassung
            distances = np.abs(np.dot(coords, normal) + d)
            rms_error = float(np.sqrt(np.mean(distances**2)))
            
            # Plane-Parameter
            plane_params = {
                'a': float(normal[0]),
                'b': float(normal[1]),
                'c': float(normal[2]),
                'd': float(d)
            }
            
            return normal, rms_error, plane_params
            
        except Exception as e:
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG SVD] Fehler bei SVD: {e}")
            return None, None, None
    
    def _compute_surface_normal(self, points: List[Dict[str, float]]) -> Optional[np.ndarray]:
        """
        Berechnet die Normale einer Fläche aus den Polygon-Punkten.
        Robuste Methode für kleine und große Polygone, auch bei Kugel-Geometrien.
        """
        if len(points) < 3:
            return None
        
        # Konvertiere zu numpy-Array
        coords = np.array([
            [float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0))]
            for p in points
        ], dtype=float)
        
        if coords.shape[0] < 3:
            return None
        
        # Für sehr kleine Polygone (z.B. Kugel-Faces): Verwende alle Punkte
        # Für größere Polygone: Verwende repräsentative Stichprobe
        n_points = len(coords)
        if n_points <= 4:
            # Alle Punkte verwenden
            indices = list(range(n_points))
        else:
            # Repräsentative Stichprobe (erste, mittlere, letzte Punkte)
            step = max(1, n_points // 4)
            indices = list(range(0, n_points, step))
            if indices[-1] != n_points - 1:
                indices.append(n_points - 1)
        
        # Berechne Normale über mehrere Dreiecke (robuster)
        normals = []
        center = np.mean(coords, axis=0)
        
        # Verwende verschiedene Dreiecks-Kombinationen
        used_indices = set()
        for i in range(len(indices)):
            idx0 = indices[i]
            idx1 = indices[(i + 1) % len(indices)]
            idx2 = indices[(i + 2) % len(indices)]
            
            # Vermeide doppelte Berechnungen
            triangle_key = tuple(sorted([idx0, idx1, idx2]))
            if triangle_key in used_indices:
                continue
            used_indices.add(triangle_key)
            
            p0 = coords[idx0]
            p1 = coords[idx1]
            p2 = coords[idx2]
            
            v1 = p1 - p0
            v2 = p2 - p0
            
            # Prüfe auf kollineare Punkte
            if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                continue
            
            # Kreuzprodukt für Normale
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            
            if norm > 1e-9:
                n = n / norm
                # WICHTIG: Normale NICHT umdrehen - behalte Original-Richtung
                # (kann nach oben oder unten zeigen, je nach Flächen-Orientierung)
                normals.append(n)
        
        if not normals:
            return None
        
        # Durchschnittliche Normale (gewichtet nach Flächeninhalt der Dreiecke)
        # Für einfachere Berechnung: Einfacher Durchschnitt
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-9:
            return avg_normal / norm
        
        return None
    
    def _compute_pca_orientation(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Berechnet die Hauptkomponenten-Analyse (PCA) der Punkte.
        Gibt Informationen über die Ausrichtung der Fläche zurück.
        Robust auch für kleine Polygone (z.B. Kugel-Faces).
        """
        if len(points) < 3:
            return {"vertical_score": 0.0, "dominant_axis": None}
        
        coords = np.array([
            [float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0))]
            for p in points
        ], dtype=float)
        
        # Prüfe auf degenerierte Fälle (alle Punkte identisch oder kollinear)
        if coords.shape[0] < 3:
            return {"vertical_score": 0.0, "dominant_axis": None}
        
        # Zentriere Punkte
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        # Prüfe ob alle Punkte identisch sind
        if np.allclose(coords_centered, 0, atol=1e-9):
            return {"vertical_score": 0.0, "dominant_axis": None}
        
        # Für sehr kleine Polygone (z.B. 3 Punkte): Verwende direkte Normale
        if len(coords) == 3:
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
                z_axis = np.array([0, 0, 1])
                cos_angle = np.clip(np.abs(np.dot(normal, z_axis)), 0, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                vertical_score = angle_deg / 90.0
                
                # Bestimme dominante Achse
                x_span = float(np.ptp(coords[:, 0]))
                y_span = float(np.ptp(coords[:, 1]))
                z_span = float(np.ptp(coords[:, 2]))
                
                # 🎯 VERBESSERTE ERKENNUNG: Prüfe Varianz statt nur Spanne
                x_var = float(np.var(coords[:, 0]))
                y_var = float(np.var(coords[:, 1]))
                z_var = float(np.var(coords[:, 2]))
                max_var = max(x_var, y_var, z_var)
                if max_var > 1e-12:
                    x_var_rel = x_var / max_var
                    y_var_rel = y_var / max_var
                else:
                    x_var_rel = 0.0
                    y_var_rel = 0.0
                
                dominant_axis = None
                if z_span > max(x_span, y_span) * 0.7:
                    # 🎯 VERBESSERTE BEDINGUNG: Prüfe ob X oder Y wirklich konstant ist
                    eps_constant = 1e-6
                    x_is_constant = (x_span < eps_constant) or (x_var_rel < 0.01)
                    y_is_constant = (y_span < eps_constant) or (y_var_rel < 0.01)
                    
                    if x_is_constant and not y_is_constant:
                        dominant_axis = "yz"  # Y-Z-Wand (X ist konstant)
                    elif y_is_constant and not x_is_constant:
                        dominant_axis = "xz"  # X-Z-Wand (Y ist konstant)
                    elif x_span < y_span * 0.2 and x_var_rel < y_var_rel * 0.1:
                        dominant_axis = "yz"
                    elif y_span < x_span * 0.2 and y_var_rel < x_var_rel * 0.1:
                        dominant_axis = "xz"
                
                return {
                    "vertical_score": float(vertical_score),
                    "angle_with_z": float(angle_deg),
                    "min_eigenval": 0.0,
                    "dominant_axis": dominant_axis,
                    "eigenvals": [0.0, 0.0, 0.0]
                }
        
        # Für größere Polygone: Verwende PCA
        try:
            # Kovarianz-Matrix (robust auch bei wenigen Punkten)
            if len(coords_centered) == 1:
                cov = np.zeros((3, 3))
            else:
                cov = np.cov(coords_centered.T)
                # Fallback für 1D-Arrays
                if cov.ndim == 0:
                    cov = np.zeros((3, 3))
                elif cov.ndim == 1:
                    # Diagonal-Matrix
                    cov = np.diag(cov)
                    if cov.shape[0] != 3:
                        cov = np.zeros((3, 3))
            
            # Eigenwerte und Eigenvektoren
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # Sortiere nach Eigenwerten (größte zuerst)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Normiere Eigenwerte
            total_var = np.sum(eigenvals)
            if total_var < 1e-12:
                return {"vertical_score": 0.0, "dominant_axis": None}
            
            eigenvals_norm = eigenvals / total_var
            
            # WICHTIG: Nach Sortierung ist eigenvals[0] der GRÖSSTE Eigenwert
            # Der kleinste Eigenwert (senkrecht zur Fläche) ist eigenvals[2]
            # Der zugehörige Eigenvektor gibt die Normale der Fläche an
            min_eigenval = eigenvals_norm[2]  # Kleinster Eigenwert (senkrecht zur Fläche)
            min_eigenvec = eigenvecs[:, 2]  # Normale der Fläche
            
            # Winkel zwischen Normale und Z-Achse
            z_axis = np.array([0, 0, 1])
            # Verwende Absolutwert, da Normale in beide Richtungen zeigen kann
            cos_angle = np.clip(np.abs(np.dot(min_eigenvec, z_axis)), 0, 1)
            angle_with_z = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_with_z)
            
            # 🎯 EMPFINDLICHERE VERTIKAL-SCORE für geringe Steigungen:
            # Linear interpolieren statt einfache Division
            # 0° = 0.0, 5° = 0.1, 75° = 0.9, 90° = 1.0
            if angle_deg < 5:
                vertical_score = angle_deg / 5.0 * 0.1  # 0.0 bis 0.1
            elif angle_deg < 75:
                vertical_score = 0.1 + (angle_deg - 5.0) / 70.0 * 0.8  # 0.1 bis 0.9
            else:
                vertical_score = 0.9 + (angle_deg - 75.0) / 15.0 * 0.1  # 0.9 bis 1.0
            
            vertical_score = np.clip(vertical_score, 0.0, 1.0)
            
            # Bestimme dominante Achse (X oder Y)
            # Prüfe, welche Koordinate in der Fläche am meisten variiert
            x_span = float(np.ptp(coords[:, 0]))
            y_span = float(np.ptp(coords[:, 1]))
            z_span = float(np.ptp(coords[:, 2]))
            
            # 🎯 VERBESSERTE ERKENNUNG: Prüfe Varianz statt nur Spanne
            # Für eine Y-Z-Wand sollte X konstant sein (geringe Varianz)
            # Für eine X-Z-Wand sollte Y konstant sein (geringe Varianz)
            x_var = float(np.var(coords[:, 0]))
            y_var = float(np.var(coords[:, 1]))
            z_var = float(np.var(coords[:, 2]))
            
            # Relative Varianz (normalisiert zur größten Varianz)
            max_var = max(x_var, y_var, z_var)
            if max_var > 1e-12:
                x_var_rel = x_var / max_var
                y_var_rel = y_var / max_var
            else:
                x_var_rel = 0.0
                y_var_rel = 0.0
            
            dominant_axis = None
            if z_span > max(x_span, y_span) * 0.7:
                # Z variiert am meisten → vertikal
                # 🎯 VERBESSERTE BEDINGUNG: Prüfe ob X oder Y wirklich konstant ist
                # Verwende sowohl Spanne als auch Varianz für robustere Erkennung
                eps_constant = 1e-6
                x_is_constant = (x_span < eps_constant) or (x_var_rel < 0.01)
                y_is_constant = (y_span < eps_constant) or (y_var_rel < 0.01)
                
                if x_is_constant and not y_is_constant:
                    dominant_axis = "yz"  # Y-Z-Wand (X ist konstant)
                elif y_is_constant and not x_is_constant:
                    dominant_axis = "xz"  # X-Z-Wand (Y ist konstant)
                elif x_span < y_span * 0.2 and x_var_rel < y_var_rel * 0.1:
                    # X variiert deutlich weniger als Y → Y-Z-Wand
                    dominant_axis = "yz"
                elif y_span < x_span * 0.2 and y_var_rel < x_var_rel * 0.1:
                    # Y variiert deutlich weniger als X → X-Z-Wand
                    dominant_axis = "xz"
            
            return {
                "vertical_score": float(vertical_score),
                "angle_with_z": float(angle_deg),
                "min_eigenval": float(min_eigenval),
                "dominant_axis": dominant_axis,
                "eigenvals": eigenvals_norm.tolist()
            }
        except Exception as e:
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG PCA] Fehler bei PCA-Analyse: {e}")
            return {"vertical_score": 0.0, "dominant_axis": None}
    
    def _determine_orientation_ultra_robust(
        self,
        points: List[Dict[str, float]],
        plane_model: Optional[Dict[str, float]],
        normal: Optional[np.ndarray]
    ) -> Tuple[str, Optional[str]]:
        """
        🎯 ULTRA-ROBUSTE Orientierungserkennung ohne Einschränkungen.
        
        Kombiniert mehrere Methoden:
        1. SVD-basiertes Plane Fitting (robusteste Methode)
        2. Normale-Vektor-Analyse
        3. PCA-Analyse
        4. Plane-Model-Analyse
        5. Spannen-Analyse (Fallback)
        
        Funktioniert für:
        - 3 bis beliebig viele Punkte
        - Beliebige Orientierung (auch negativ geneigt)
        - Kugel-Geometrien
        - Ausreißer-resistent
        """
        # Konvertiere zu numpy-Array
        xs = np.array([float(p.get('x', 0.0)) for p in points], dtype=float)
        ys = np.array([float(p.get('y', 0.0)) for p in points], dtype=float)
        zs = np.array([float(p.get('z', 0.0)) for p in points], dtype=float)
        
        x_span = float(np.ptp(xs))
        y_span = float(np.ptp(ys))
        z_span = float(np.ptp(zs))
        
        # 🎯 METHODE 1: SVD-basiertes Plane Fitting (robusteste Methode)
        svd_normal, svd_error, svd_params = self._compute_robust_plane_normal_svd(points)
        svd_score = 0.0
        
        if svd_normal is not None:
            z_axis = np.array([0, 0, 1])
            cos_angle = np.clip(np.abs(np.dot(svd_normal, z_axis)), 0, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            # Qualitäts-Bonus: Geringer Fit-Fehler erhöht Vertrauen
            quality_bonus = 1.0
            if svd_error is not None:
                # Wenn RMS-Fehler sehr klein, ist die Fläche sehr plan
                if svd_error < 1e-6:
                    quality_bonus = 1.1  # Leichter Bonus
                elif svd_error > 0.1:
                    quality_bonus = 0.9  # Leichter Penalty
            
            if angle_deg < 5:
                svd_score = 0.0 * quality_bonus
            elif angle_deg < 75:
                svd_score = ((angle_deg - 5.0) / 70.0) * quality_bonus
            else:
                svd_score = 1.0 * quality_bonus
            
            svd_score = np.clip(svd_score, 0.0, 1.0)
        
        # Methode 2: Normale-Vektor-Analyse (wenn verfügbar)
        normal_score = 0.0
        if normal is not None:
            z_axis = np.array([0, 0, 1])
            cos_angle = np.clip(np.abs(np.dot(normal, z_axis)), 0, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            
            if angle_deg < 5:
                normal_score = 0.0
            elif angle_deg < 75:
                normal_score = (angle_deg - 5.0) / 70.0
                normal_score = np.clip(normal_score, 0.0, 1.0)
            else:
                normal_score = 1.0
        
        # Methode 3: PCA-Analyse
        pca_result = self._compute_pca_orientation(points)
        pca_vertical_score = pca_result.get("vertical_score", 0.0)
        pca_dominant_axis = pca_result.get("dominant_axis", None)
        
        # Methode 4: Plane-Model-Analyse
        plane_score = 0.0
        if plane_model is not None:
            mode = plane_model.get('mode', 'constant')
            if mode == 'constant':
                plane_score = 0.0
            elif mode in ['x', 'y']:
                slope = abs(float(plane_model.get('slope', 0.0)))
                if slope < 0.01:
                    plane_score = 0.0
                elif slope < 0.5:
                    plane_score = (slope - 0.01) / 0.49 * 0.5
                elif slope < 2.0:
                    plane_score = 0.5 + (slope - 0.5) / 1.5 * 0.5
                else:
                    plane_score = 1.0
            elif mode == 'xy':
                slope_x = abs(float(plane_model.get('slope_x', 0.0)))
                slope_y = abs(float(plane_model.get('slope_y', 0.0)))
                combined_slope = np.sqrt(slope_x**2 + slope_y**2)
                if combined_slope < 0.01:
                    plane_score = 0.0
                elif combined_slope < 0.5:
                    plane_score = (combined_slope - 0.01) / 0.49 * 0.5
                elif combined_slope < 2.0:
                    plane_score = 0.5 + (combined_slope - 0.5) / 1.5 * 0.5
                else:
                    plane_score = 1.0
        
        # Methode 5: Spannen-Analyse (Fallback)
        span_score = 0.0
        eps_line = 1e-6
        max_horizontal_span = max(x_span, y_span)
        if max_horizontal_span < 1e-9:
            max_horizontal_span = 1e-9
        z_ratio = z_span / max_horizontal_span if max_horizontal_span > 0 else 0.0
        
        if (x_span < eps_line or y_span < eps_line) and z_span > 1e-3:
            span_score = 1.0
        elif z_span > 1e-3:
            if z_ratio < 0.01:
                span_score = 0.0
            elif z_ratio < 0.1:
                span_score = (z_ratio - 0.01) / 0.09 * 0.3
            elif z_ratio < 0.7:
                span_score = 0.3 + (z_ratio - 0.1) / 0.6 * 0.7
            else:
                span_score = 1.0
        
        # 🎯 GEWICHTETE KOMBINATION (SVD hat höchste Priorität)
        weights = {
            'svd': 0.5 if svd_normal is not None else 0.0,  # Höchste Priorität
            'normal': 0.25 if normal is not None else 0.0,
            'pca': 0.15,
            'plane': 0.08 if plane_model is not None else 0.0,
            'span': 0.02
        }
        
        # Normalisiere Gewichte
        total_weight = sum(weights.values())
        if total_weight < 1e-6:
            raise ValueError("Orientierung nicht bestimmbar: keine gültigen Gewichtungen aus Analyse.")
        else:
            for key in weights:
                weights[key] /= total_weight
            
            combined_score = (
                weights['svd'] * svd_score +
                weights['normal'] * normal_score +
                weights['pca'] * pca_vertical_score +
                weights['plane'] * plane_score +
                weights['span'] * span_score
            )
        
        # Bestimme Orientierung
        if combined_score < 0.15:
            orientation = "planar"
        elif combined_score < 0.6:
            orientation = "sloped"
        else:
            orientation = "vertical"
        
        # Bestimme dominante Achse für vertikale Flächen
        dominant_axis = None
        if orientation == "vertical":
            if pca_dominant_axis:
                dominant_axis = pca_dominant_axis
            else:
                # Fallback: Bestimme dominante Achse basierend auf verbesserter Analyse
                # 🎯 VERBESSERTE ERKENNUNG: Prüfe Varianz statt nur Spanne
                x_var = float(np.var(xs))
                y_var = float(np.var(ys))
                z_var = float(np.var(zs))
                max_var = max(x_var, y_var, z_var)
                if max_var > 1e-12:
                    x_var_rel = x_var / max_var
                    y_var_rel = y_var / max_var
                else:
                    x_var_rel = 0.0
                    y_var_rel = 0.0
                
                eps_line = 1e-6
                eps_constant = 1e-6
                
                # Prüfe ob X oder Y wirklich konstant ist
                x_is_constant = (x_span < eps_constant) or (x_var_rel < 0.01)
                y_is_constant = (y_span < eps_constant) or (y_var_rel < 0.01)
                
                if y_is_constant and not x_is_constant:
                    dominant_axis = "xz"  # X-Z-Wand (Y ist konstant)
                elif x_is_constant and not y_is_constant:
                    dominant_axis = "yz"  # Y-Z-Wand (X ist konstant)
                elif z_span > max(x_span, y_span) * 0.7:
                    # Bestimme welche Achse (X oder Y) deutlich weniger variiert
                    # Verwende sowohl Spanne als auch Varianz
                    if x_span < y_span * 0.2 and x_var_rel < y_var_rel * 0.1:
                        dominant_axis = "yz"
                    elif y_span < x_span * 0.2 and y_var_rel < x_var_rel * 0.1:
                        dominant_axis = "xz"
                
                if dominant_axis is None:
                    # Letzter Fallback: Verwende die Achse mit der kleinsten Spanne UND Varianz
                    # 🎯 VERSCHÄRFTE BEDINGUNG: Nur wenn die Differenz signifikant ist (>30%)
                    span_ratio = min(x_span, y_span) / max(x_span, y_span) if max(x_span, y_span) > 1e-6 else 1.0
                    var_ratio = min(x_var_rel, y_var_rel) / max(x_var_rel, y_var_rel) if max(x_var_rel, y_var_rel) > 1e-12 else 1.0
                    
                    # Nur wenn eine Achse deutlich kleiner ist (weniger als 70% der anderen)
                    if span_ratio < 0.7 and var_ratio < 0.7:
                        if x_span < y_span and x_var_rel < y_var_rel:
                            dominant_axis = "yz"
                        elif y_span < x_span and y_var_rel < x_var_rel:
                            dominant_axis = "xz"
                    # Wenn keine Achse deutlich kleiner ist, prüfe Kollinearität
                    else:
                        # Prüfe ob Punkte in (x,z)-Ebene kollinear sind
                        coords_xz = np.column_stack([xs, zs])
                        if len(coords_xz) >= 3:
                            # Berechne Varianz in (x,z)-Ebene (Determinante der Kovarianzmatrix)
                            cov_xz = np.cov(coords_xz.T)
                            det_cov = np.linalg.det(cov_xz)
                            # Wenn Determinante sehr klein, sind Punkte fast kollinear
                            if det_cov < 1e-10:
                                # Punkte sind kollinear in (x,z) → verwende Y-Z-Wand (X interpoliert aus (y,z))
                                dominant_axis = "yz"
                            else:
                                # Prüfe ob Punkte in (y,z)-Ebene kollinear sind
                                coords_yz = np.column_stack([ys, zs])
                                cov_yz = np.cov(coords_yz.T)
                                det_cov_yz = np.linalg.det(cov_yz)
                                if det_cov_yz < 1e-10:
                                    # Punkte sind kollinear in (y,z) → verwende X-Z-Wand (Y interpoliert aus (x,z))
                                    dominant_axis = "xz"
                                else:
                                    # Standard-Fallback: kleinere Spanne
                                    if x_span < y_span and x_var_rel < y_var_rel:
                                        dominant_axis = "yz"
                                    else:
                                        dominant_axis = "xz"
                        else:
                            # Standard-Fallback: kleinere Spanne
                            if x_span < y_span and x_var_rel < y_var_rel:
                                dominant_axis = "yz"
                            else:
                                dominant_axis = "xz"
                
                if DEBUG_FLEXIBLE_GRID:
                    print(f"[DEBUG Ultra-Robust] ⚠️ PCA lieferte keine dominante Achse, verwende verbesserte Analyse: {dominant_axis}")
                    print(f"  └─ x_span={x_span:.3f}, y_span={y_span:.3f}, x_var_rel={x_var_rel:.6f}, y_var_rel={y_var_rel:.6f}")
        
        if DEBUG_FLEXIBLE_GRID:
            print(f"[DEBUG Ultra-Robust] Score-Analyse:")
            print(f"  └─ SVD: {svd_score:.3f} (weight: {weights['svd']:.3f}, error: {svd_error:.6f if svd_error else 'N/A'})")
            print(f"  └─ Normal: {normal_score:.3f} (weight: {weights['normal']:.3f})")
            print(f"  └─ PCA: {pca_vertical_score:.3f} (weight: {weights['pca']:.3f})")
            print(f"  └─ Plane: {plane_score:.3f} (weight: {weights['plane']:.3f})")
            print(f"  └─ Span: {span_score:.3f} (weight: {weights['span']:.3f})")
            print(f"  └─ Combined: {combined_score:.3f} → {orientation}")
            if dominant_axis:
                print(f"  └─ Dominant axis: {dominant_axis}")
        
        return orientation, dominant_axis
    
    def _determine_orientation_improved(
        self,
        points: List[Dict[str, float]],
        plane_model: Optional[Dict[str, float]],
        normal: Optional[np.ndarray]
    ) -> Tuple[str, Optional[str]]:
        """
        Bestimmt die Orientierung einer Fläche mit mehreren Methoden.
        
        Returns:
            (orientation, dominant_axis)
            - orientation: "planar", "sloped", "vertical"
            - dominant_axis: "xz", "yz", oder None
        """
        # Konvertiere zu numpy-Array
        xs = np.array([float(p.get('x', 0.0)) for p in points], dtype=float)
        ys = np.array([float(p.get('y', 0.0)) for p in points], dtype=float)
        zs = np.array([float(p.get('z', 0.0)) for p in points], dtype=float)
        
        x_span = float(np.ptp(xs))
        y_span = float(np.ptp(ys))
        z_span = float(np.ptp(zs))
        
        # Methode 1: Normale-Vektor-Analyse (wenn verfügbar)
        normal_score = 0.0
        if normal is not None:
            z_axis = np.array([0, 0, 1])
            # Winkel zwischen Normale und Z-Achse
            # Normale parallel zu Z (0°) = horizontal/planar
            # Normale senkrecht zu Z (90°) = vertikal
            # WICHTIG: Verwende Absolutwert, da Normale nach oben oder unten zeigen kann
            cos_angle = np.clip(np.abs(np.dot(normal, z_axis)), 0, 1)
            angle_with_z = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_with_z)
            
            # 🎯 EMPFINDLICHERE SCHWELLENWERTE für geringe Steigungen:
            # 0-5° = planar (praktisch horizontal)
            # 5-75° = sloped (auch sehr geringe Steigungen werden erkannt)
            # 75-90° = vertical
            if angle_deg < 5:
                normal_score = 0.0  # planar (praktisch horizontal)
            elif angle_deg < 75:
                # Sloped: Linear interpolieren zwischen 0.0 und 1.0
                # 5° → 0.0, 75° → 1.0
                normal_score = (angle_deg - 5.0) / 70.0  # 0.0 bis 1.0
                normal_score = np.clip(normal_score, 0.0, 1.0)
            else:
                normal_score = 1.0  # vertical
        
        # Methode 2: PCA-Analyse
        pca_result = self._compute_pca_orientation(points)
        pca_vertical_score = pca_result.get("vertical_score", 0.0)
        pca_dominant_axis = pca_result.get("dominant_axis", None)
        
        # Methode 3: Plane-Model-Analyse
        plane_score = 0.0
        if plane_model is not None:
            mode = plane_model.get('mode', 'constant')
            if mode == 'constant':
                plane_score = 0.0  # planar
            elif mode in ['x', 'y']:
                # Prüfe Steigung (absolut, funktioniert auch für negative Steigungen)
                slope = abs(float(plane_model.get('slope', 0.0)))
                # 🎯 EMPFINDLICHERE SCHWELLENWERTE:
                # < 0.01 = praktisch planar
                # 0.01-0.5 = leicht geneigt (sloped)
                # 0.5-2.0 = mäßig geneigt (sloped)
                # > 2.0 = steil/vertikal
                if slope < 0.01:
                    plane_score = 0.0  # praktisch planar
                elif slope < 0.5:
                    # Leicht geneigt: Linear interpolieren
                    plane_score = (slope - 0.01) / 0.49 * 0.5  # 0.0 bis 0.5
                elif slope < 2.0:
                    # Mäßig geneigt: Linear interpolieren
                    plane_score = 0.5 + (slope - 0.5) / 1.5 * 0.5  # 0.5 bis 1.0
                else:
                    plane_score = 1.0  # steil/vertikal
            elif mode == 'xy':
                slope_x = abs(float(plane_model.get('slope_x', 0.0)))
                slope_y = abs(float(plane_model.get('slope_y', 0.0)))
                # Kombinierte Steigung (Euklidische Norm)
                combined_slope = np.sqrt(slope_x**2 + slope_y**2)
                if combined_slope < 0.01:
                    plane_score = 0.0
                elif combined_slope < 0.5:
                    plane_score = (combined_slope - 0.01) / 0.49 * 0.5
                elif combined_slope < 2.0:
                    plane_score = 0.5 + (combined_slope - 0.5) / 1.5 * 0.5
                else:
                    plane_score = 1.0
        
        # Methode 4: Spannen-Analyse (Fallback, besonders wichtig für kleine Polygone)
        span_score = 0.0
        eps_line = 1e-6
        
        # Berechne relative Spannen
        max_horizontal_span = max(x_span, y_span)
        if max_horizontal_span < 1e-9:
            max_horizontal_span = 1e-9  # Vermeide Division durch Null
        
        # Verhältnis von Z-Spanne zu horizontaler Spanne
        z_ratio = z_span / max_horizontal_span if max_horizontal_span > 0 else 0.0
        
        if (x_span < eps_line or y_span < eps_line) and z_span > 1e-3:
            span_score = 1.0  # vertikal (eine Dimension degeneriert)
        elif z_span > 1e-3:
            # 🎯 EMPFINDLICHERE ERKENNUNG auch für geringe Steigungen:
            # z_ratio < 0.01 = praktisch planar
            # 0.01 <= z_ratio < 0.1 = leicht geneigt
            # 0.1 <= z_ratio < 0.7 = mäßig geneigt
            # z_ratio >= 0.7 = vertikal
            if z_ratio < 0.01:
                span_score = 0.0  # praktisch planar
            elif z_ratio < 0.1:
                # Leicht geneigt: Linear interpolieren
                span_score = (z_ratio - 0.01) / 0.09 * 0.3  # 0.0 bis 0.3
            elif z_ratio < 0.7:
                # Mäßig geneigt: Linear interpolieren
                span_score = 0.3 + (z_ratio - 0.1) / 0.6 * 0.7  # 0.3 bis 1.0
            else:
                span_score = 1.0  # vertikal
        
        # Kombiniere Scores (gewichteter Durchschnitt)
        # Normale und PCA sind am zuverlässigsten
        weights = {
            'normal': 0.4 if normal is not None else 0.0,
            'pca': 0.4,
            'plane': 0.15 if plane_model is not None else 0.0,
            'span': 0.05
        }
        
        # Normalisiere Gewichte
        total_weight = sum(weights.values())
        if total_weight < 1e-6:
            raise ValueError("Orientierung nicht bestimmbar (improved): keine Gewichtungen verfügbar.")
        else:
            for key in weights:
                weights[key] /= total_weight
            
            combined_score = (
                weights['normal'] * normal_score +
                weights['pca'] * pca_vertical_score +
                weights['plane'] * plane_score +
                weights['span'] * span_score
            )
        
        # Bestimme Orientierung basierend auf Score
        # 🎯 ANGEPASSTE SCHWELLENWERTE für bessere Erkennung geringer Steigungen:
        # < 0.15 = planar (sehr empfindlich, erkennt auch fast-horizontale Flächen)
        # 0.15-0.6 = sloped (breiter Bereich für alle geneigten Flächen)
        # >= 0.6 = vertical
        if combined_score < 0.15:
            orientation = "planar"
        elif combined_score < 0.6:
            orientation = "sloped"
        else:
            orientation = "vertical"
        
        # Bestimme dominante Achse für vertikale Flächen
        dominant_axis = None
        if orientation == "vertical":
            # Verwende PCA-Ergebnis oder Spannen-Analyse
            if pca_dominant_axis:
                dominant_axis = pca_dominant_axis
            else:
                # Fallback: Bestimme dominante Achse basierend auf Spannen-Analyse
                # (kein stiller Fallback - wird verwendet wenn PCA keine Achse liefert)
                eps_line = 1e-6
                if y_span < eps_line and x_span >= eps_line:
                    dominant_axis = "xz"  # X-Z-Wand
                elif x_span < eps_line and y_span >= eps_line:
                    dominant_axis = "yz"  # Y-Z-Wand
                elif z_span > max(x_span, y_span) * 0.7:
                    # Bestimme welche Achse (X oder Y) weniger variiert
                    if x_span < y_span * 0.5:
                        dominant_axis = "yz"
                    elif y_span < x_span * 0.5:
                        dominant_axis = "xz"
                
                if dominant_axis is None:
                    # Letzter Fallback: Verwende die Achse mit der kleinsten Spanne
                    if x_span < y_span:
                        dominant_axis = "yz"
                    else:
                        dominant_axis = "xz"
                
                if DEBUG_FLEXIBLE_GRID:
                    print(f"[DEBUG Orientation] ⚠️ PCA lieferte keine dominante Achse, verwende Spannen-Analyse: {dominant_axis}")
        
        if DEBUG_FLEXIBLE_GRID:
            print(f"[DEBUG Orientation] Score-Analyse:")
            print(f"  └─ Normal: {normal_score:.3f} (weight: {weights['normal']:.3f})")
            print(f"  └─ PCA: {pca_vertical_score:.3f} (weight: {weights['pca']:.3f})")
            print(f"  └─ Plane: {plane_score:.3f} (weight: {weights['plane']:.3f})")
            print(f"  └─ Span: {span_score:.3f} (weight: {weights['span']:.3f})")
            print(f"  └─ Combined: {combined_score:.3f} → {orientation}")
            if dominant_axis:
                print(f"  └─ Dominant axis: {dominant_axis}")
        
        return orientation, dominant_axis
    
    def analyze_surfaces(
        self,
        enabled_surfaces: List[Tuple[str, Dict]]
    ) -> List[SurfaceGeometry]:
        """
        Analysiert alle enabled Surfaces und bestimmt deren Geometrie.
        Verwendet verbesserte Methoden: Normale-Vektor, PCA, Plane-Model.
        
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
            
            # Berechne Normale der Fläche (Fallback-Methode)
            normal = self._compute_surface_normal(points)
            
            # 🎯 ULTRA-ROBUSTE Orientierungserkennung (ohne Einschränkungen)
            orientation, dominant_axis = self._determine_orientation_ultra_robust(
                points, plane_model, normal
            )
            
            # Speichere zusätzliche Informationen in geometry
            geometry = SurfaceGeometry(
                surface_id=surface_id,
                name=name,
                points=points,
                plane_model=plane_model,
                orientation=orientation,
                normal=normal,
                bbox=bbox,
                dominant_axis=dominant_axis  # 🎯 NEU: Speichere dominant_axis für Plot
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
                # Kein stiller Fallback mehr: ohne Bounding Box sollen Fehler sichtbar sein
                raise ValueError("Keine gültige Bounding Box aus Surfaces ableitbar – bitte Surfaces prüfen.")
        else:
            # Kein Fallback bei fehlenden Surfaces – explizit fehlschlagen
            raise ValueError("Keine Surfaces übergeben – Grid-Erstellung abgebrochen.")
        
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
                raise ValueError(f"Vertikale Surface '{geometry.surface_id}': keine klare (u,v)-Zuordnung für Maske.")
            
            # Für vertikale Surfaces muss Z_grid vorhanden sein; ohne eindeutige Zuordnung brechen wir ab.
            raise ValueError(f"Vertikale Surface '{geometry.surface_id}': Z_grid/Maske nicht eindeutig ableitbar.")
        
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

    def _ensure_vertex_coverage(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        geometry: SurfaceGeometry,
        surface_mask_strict: np.ndarray,
    ) -> np.ndarray:
        """
        Erweiterung der strikten Maske:
        Stellt sicher, dass für jede Polygon-Ecke mindestens der nächste Grid-Punkt
        als "inside" markiert wird (falls sinnvoll nahe).
        
        Motivation:
        - Sehr schmale/spitze Polygonbereiche können bei grober Auflösung so liegen,
          dass kein Grid-Zentrum innerhalb der Spitze liegt.
        - Dann ist die Maske dort komplett False und es entstehen keine Dreiecke.
        
        Heuristik:
        - Für jede Ecke (x_v, y_v):
          - Finde den nächstgelegenen Grid-Punkt (X_ij, Y_ij).
          - Falls dieser Punkt weiter als ≈ 2 * Grid-Resolution entfernt ist,
            wird er ignoriert (damit wir keine sehr entfernten Punkte aktivieren).
          - Andernfalls wird surface_mask_strict[ij] auf True gesetzt.
        """
        try:
            pts = getattr(geometry, "points", None) or []
            if not pts or X_grid.size == 0 or Y_grid.size == 0:
                return surface_mask_strict
            
            # Nur für planare/schräge Flächen im XY-System – vertikale Flächen
            # arbeiten in (u,v) und haben eigene Logik.
            if geometry.orientation not in ("planar", "sloped"):
                return surface_mask_strict
            
            x_vertices = np.array([float(p.get("x", 0.0)) for p in pts], dtype=float)
            y_vertices = np.array([float(p.get("y", 0.0)) for p in pts], dtype=float)
            if x_vertices.size == 0 or y_vertices.size == 0:
                return surface_mask_strict
            
            x_flat = X_grid.ravel()
            y_flat = Y_grid.ravel()
            mask_flat = surface_mask_strict.ravel()
            
            # Schätze typische Grid-Resolution in X/Y aus den Koordinaten
            # (Median der Abstände entlang der Achsen der ersten Zeile/Spalte).
            def _estimate_resolution(axis_vals: np.ndarray) -> float:
                if axis_vals.size < 2:
                    return 0.0
                diffs = np.diff(axis_vals)
                diffs = diffs[np.abs(diffs) > 1e-9]
                if diffs.size == 0:
                    return 0.0
                return float(np.median(np.abs(diffs)))
            
            try:
                # Verwende erste Zeile für X, erste Spalte für Y
                res_x = _estimate_resolution(X_grid[0, :])
                res_y = _estimate_resolution(Y_grid[:, 0])
            except Exception:
                res_x = 0.0
                res_y = 0.0
            base_res = max(res_x, res_y, 0.0)
            if base_res <= 0.0:
                # Fallback: keine sinnvolle Resolution ableitbar → ohne Distanzschranke
                max_sq_dist = None
            else:
                # Erlaube bis zu ca. 2 * Resolution Abstand
                max_sq_dist = float((2.0 * base_res) ** 2)
            
            for xv, yv in zip(x_vertices, y_vertices):
                # Distanz zu allen Grid-Zentren
                dx = x_flat - xv
                dy = y_flat - yv
                d2 = dx * dx + dy * dy
                # Index des nächsten Grid-Punkts
                nearest_idx = int(np.argmin(d2))
                if max_sq_dist is not None and d2[nearest_idx] > max_sq_dist:
                    # Ecke liegt zu weit vom Grid entfernt → nicht erzwingen
                    continue
                # Markiere diesen Grid-Punkt als "inside", falls noch nicht gesetzt
                if not mask_flat[nearest_idx]:
                    mask_flat[nearest_idx] = True
            
            return mask_flat.reshape(surface_mask_strict.shape)
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict
    
    # @measure_time("GridBuilder.build_single_surface_grid")
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
        t_orientation_start = time.perf_counter() if PERF_ENABLED else None
        
        # 🎯 VERTIKALE SURFACES: Erstelle Grid direkt in (u,v)-Ebene der Fläche
        # Gleiche Funktionalität wie planare Flächen, nur andere Flächenausrichtung
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
            
            # 🎯 IDENTISCHE RESOLUTION: Keine Verdoppelung mehr, verwende Basis-Resolution
            step = float(resolution or 1.0)
            
            # 🎯 PRÜFE ZUERST OB SCHRÄGE VERTIKALE FLÄCHE (Y oder X variiert stark)
            # Wenn dominant_axis vorhanden ist und Y/X variiert, behandle als schräg
            is_slanted_vertical = False
            if hasattr(geometry, 'dominant_axis') and geometry.dominant_axis:
                # Wenn dominant_axis vorhanden ist, prüfe ob Y oder X variiert
                if geometry.dominant_axis == "xz":
                    # X-Z-Wand: Y sollte variieren können (schräg)
                    # Wenn Y variiert (y_span > eps_line), ist die Wand schräg
                    # Zusätzlich: Z-Spanne sollte signifikant sein (z_span > 1e-3)
                    is_slanted_vertical = (y_span > eps_line and z_span > 1e-3)
                elif geometry.dominant_axis == "yz":
                    # Y-Z-Wand: X sollte variieren können (schräg)
                    # Wenn X variiert (x_span > eps_line), ist die Wand schräg
                    # Zusätzlich: Z-Spanne sollte signifikant sein (z_span > 1e-3)
                    is_slanted_vertical = (x_span > eps_line and z_span > 1e-3)
            
            
            
            # 🎯 BEHANDLE SCHRÄGE FLÄCHEN ZUERST (vor den Bedingungen für konstante X/Y)
            # Wenn schräge Fläche erkannt, überspringe die Bedingungen für konstante X/Y
            if not is_slanted_vertical and y_span < eps_line and x_span >= eps_line:
                # X-Z-Wand: y ≈ const, Grid in (x,z)-Ebene
                # u = x, v = z
                y0 = float(np.mean(ys))
                u_min, u_max = float(xs.min()), float(xs.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                
                # Prüfe auf degenerierte Flächen
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    raise ValueError(f"Vertikale Surface '{geometry.surface_id}' degeneriert (X-Z-Wand) – Grid-Erstellung abgebrochen.")
                
                # 🎯 ADAPTIVE RESOLUTION: Wie bei planaren Flächen
                width = u_max - u_min
                height = v_max - v_min
                nu_base = max(1, int(np.ceil(width / step)) + 1)
                nv_base = max(1, int(np.ceil(height / step)) + 1)
                total_points_base = nu_base * nv_base
                min_total_points = min_points_per_dimension ** 2
                
                if total_points_base < min_total_points:
                    diagonal = np.sqrt(width**2 + height**2)
                    if diagonal > 0:
                        adaptive_resolution = diagonal / min_points_per_dimension
                        adaptive_resolution = min(adaptive_resolution, step * 0.5)
                        step = adaptive_resolution
                
                # 🎯 IDENTISCHES PADDING: Wie bei planaren Flächen
                u_min -= step
                u_max += step
                v_min -= step
                v_max += step
                
                u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                    
                if u_axis.size < 2 or v_axis.size < 2:
                    raise ValueError(f"Vertikale Surface '{geometry.surface_id}' (X-Z-Wand) liefert zu wenige Punkte in (u,v)-Ebene.")
                
                # Erstelle 2D-Meshgrid in (u,v)-Ebene
                U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                
                # Transformiere zu (X, Y, Z) Koordinaten
                X_grid = U_grid  # u = x
                Y_grid = np.full_like(U_grid, y0, dtype=float)  # y = konstant
                Z_grid = V_grid  # v = z
                
                # sound_field_x und sound_field_y für Rückgabe (werden für Plot verwendet)
                sound_field_x = u_axis  # x-Koordinaten
                sound_field_y = v_axis  # z-Koordinaten (als y für sound_field_y)
                
                
            elif not is_slanted_vertical and x_span < eps_line and y_span >= eps_line:
                # Y-Z-Wand: x ≈ const, Grid in (y,z)-Ebene
                # u = y, v = z
                x0 = float(np.mean(xs))
                u_min, u_max = float(ys.min()), float(ys.max())
                v_min, v_max = float(zs.min()), float(zs.max())
                
                # Prüfe auf degenerierte Flächen
                if math.isclose(u_min, u_max) or math.isclose(v_min, v_max):
                    raise ValueError(f"Vertikale Surface '{geometry.surface_id}' degeneriert (Y-Z-Wand) – Grid-Erstellung abgebrochen.")
                
                # 🎯 ADAPTIVE RESOLUTION: Wie bei planaren Flächen
                width = u_max - u_min
                height = v_max - v_min
                nu_base = max(1, int(np.ceil(width / step)) + 1)
                nv_base = max(1, int(np.ceil(height / step)) + 1)
                total_points_base = nu_base * nv_base
                min_total_points = min_points_per_dimension ** 2
                
                if total_points_base < min_total_points:
                    diagonal = np.sqrt(width**2 + height**2)
                    if diagonal > 0:
                        adaptive_resolution = diagonal / min_points_per_dimension
                        adaptive_resolution = min(adaptive_resolution, step * 0.5)
                        step = adaptive_resolution
                
                # 🎯 IDENTISCHES PADDING: Wie bei planaren Flächen
                u_min -= step
                u_max += step
                v_min -= step
                v_max += step
                
                u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                    
                if u_axis.size < 2 or v_axis.size < 2:
                    raise ValueError(f"Vertikale Surface '{geometry.surface_id}' (Y-Z-Wand) liefert zu wenige Punkte in (u,v)-Ebene.")
                
                # Erstelle 2D-Meshgrid in (u,v)-Ebene
                U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                
                # Transformiere zu (X, Y, Z) Koordinaten
                X_grid = np.full_like(U_grid, x0, dtype=float)  # x = konstant
                Y_grid = U_grid  # u = y
                Z_grid = V_grid  # v = z
                
                # sound_field_x und sound_field_y für Rückgabe (werden für Plot verwendet)
                sound_field_x = u_axis  # y-Koordinaten (als x für sound_field_x)
                sound_field_y = v_axis  # z-Koordinaten (als y für sound_field_y)
                
            else:
                if is_slanted_vertical:
                    # Schräge vertikale Fläche: Bestimme dominante Orientierung
                    # 🎯 VERWENDE dominant_axis (muss vorhanden sein)
                    if not hasattr(geometry, 'dominant_axis') or not geometry.dominant_axis:
                        raise ValueError(f"Surface '{geometry.surface_id}': dominant_axis nicht verfügbar für schräge vertikale Fläche")
                    
                    use_yz_wall = False
                    if geometry.dominant_axis == "yz":
                        use_yz_wall = True
                    elif geometry.dominant_axis == "xz":
                        use_yz_wall = False
                    else:
                        raise ValueError(f"Surface '{geometry.surface_id}': Unbekannter dominant_axis '{geometry.dominant_axis}'")
                    
                    if use_yz_wall:
                        # Y-Z-Wand schräg: Y und Z variieren, X variiert entlang der Fläche
                        # u = y, v = z
                        u_min, u_max = float(ys.min()), float(ys.max())
                        v_min, v_max = float(zs.min()), float(zs.max())
                        
                        # 🎯 ADAPTIVE RESOLUTION: Wie bei planaren Flächen
                        width = u_max - u_min
                        height = v_max - v_min
                        nu_base = max(1, int(np.ceil(width / step)) + 1)
                        nv_base = max(1, int(np.ceil(height / step)) + 1)
                        total_points_base = nu_base * nv_base
                        min_total_points = min_points_per_dimension ** 2
                        
                        if total_points_base < min_total_points:
                            diagonal = np.sqrt(width**2 + height**2)
                            if diagonal > 0:
                                adaptive_resolution = diagonal / min_points_per_dimension
                                adaptive_resolution = min(adaptive_resolution, step * 0.5)
                                step = adaptive_resolution
                        
                        # 🎯 IDENTISCHES PADDING: Wie bei planaren Flächen
                        u_min -= step
                        u_max += step
                        v_min -= step
                        v_max += step
                        
                        # X wird später interpoliert
                        from scipy.interpolate import griddata
                        
                        # Erstelle Grid in (u,v)-Ebene = (y,z)-Ebene
                        u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                        v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                        
                        if u_axis.size < 2 or v_axis.size < 2:
                            raise ValueError(f"Vertikale Surface '{geometry.surface_id}' (schräg, y-z) liefert zu wenige Punkte in (u,v)-Ebene.")
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
                            
                    else:
                        # X-Z-Wand schräg: X und Z variieren, Y variiert entlang der Fläche
                        # u = x, v = z
                        # 🎯 VERWENDE dominant_axis WENN VERFÜGBAR (wurde durch PCA/robuste Analyse bestimmt)
                        
                        # 🎯 PRÜFE KOLLINEARITÄT VOR Grid-Erstellung: Wenn Punkte in (x,z)-Ebene fast kollinear sind,
                        # dann wechsle automatisch zu Y-Z-Wand (X wird aus (y,z) interpoliert)
                        points_surface_xz = np.column_stack([xs, zs])
                        should_switch_to_yz = False
                        
                        if len(points_surface_xz) >= 3:
                            cov_xz = np.cov(points_surface_xz.T)
                            det_cov_xz = np.linalg.det(cov_xz)
                            
                            if det_cov_xz < 1e-10:
                                # Punkte sind fast kollinear in (x,z) → prüfe ob (y,z) besser ist
                                
                                points_surface_yz = np.column_stack([ys, zs])
                                cov_yz = np.cov(points_surface_yz.T)
                                det_cov_yz = np.linalg.det(cov_yz)
                                
                                if det_cov_yz > 1e-10:
                                    # Punkte sind besser verteilt in (y,z) → wechsle zu Y-Z-Wand
                                    should_switch_to_yz = True
                                    print(f"  └─ ✅ Punkte besser verteilt in (y,z)-Ebene (det={det_cov_yz:.2e}) → wechsle zu Y-Z-Wand")
                        
                        if should_switch_to_yz:
                            # Wechsle zu Y-Z-Wand: X wird aus (y,z) interpoliert
                            u_min, u_max = float(ys.min()), float(ys.max())
                            v_min, v_max = float(zs.min()), float(zs.max())
                            
                            # Adaptive Resolution
                            width = u_max - u_min
                            height = v_max - v_min
                            nu_base = max(1, int(np.ceil(width / step)) + 1)
                            nv_base = max(1, int(np.ceil(height / step)) + 1)
                            total_points_base = nu_base * nv_base
                            min_total_points = min_points_per_dimension ** 2
                            
                            if total_points_base < min_total_points:
                                diagonal = np.sqrt(width**2 + height**2)
                                if diagonal > 0:
                                    adaptive_resolution = diagonal / min_points_per_dimension
                                    adaptive_resolution = min(adaptive_resolution, step * 0.5)
                                    step = adaptive_resolution
                            
                            u_min -= step
                            u_max += step
                            v_min -= step
                            v_max += step
                            
                            u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                            v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                            
                            if u_axis.size < 2 or v_axis.size < 2:
                                raise ValueError(f"Vertikale Surface '{geometry.surface_id}' (schräg, y-z) liefert zu wenige Punkte in (u,v)-Ebene.")
                            
                            U_grid, V_grid = np.meshgrid(u_axis, v_axis, indexing='xy')
                            
                            # Interpoliere X-Koordinaten von Surface-Punkten auf (y,z)-Grid
                            points_surface = np.column_stack([ys, zs])
                            points_grid = np.column_stack([U_grid.ravel(), V_grid.ravel()])
                            
                            
                            from scipy.interpolate import griddata
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
                            
                        else:
                            # Normale X-Z-Wand: Y wird aus (x,z) interpoliert
                            u_min, u_max = float(xs.min()), float(xs.max())
                            v_min, v_max = float(zs.min()), float(zs.max())
                            
                            # 🎯 ADAPTIVE RESOLUTION: Wie bei planaren Flächen
                            width = u_max - u_min
                            height = v_max - v_min
                            nu_base = max(1, int(np.ceil(width / step)) + 1)
                            nv_base = max(1, int(np.ceil(height / step)) + 1)
                            total_points_base = nu_base * nv_base
                            min_total_points = min_points_per_dimension ** 2
                            
                            if total_points_base < min_total_points:
                                diagonal = np.sqrt(width**2 + height**2)
                                if diagonal > 0:
                                    adaptive_resolution = diagonal / min_points_per_dimension
                                    adaptive_resolution = min(adaptive_resolution, step * 0.5)
                                    step = adaptive_resolution
                            
                            # 🎯 IDENTISCHES PADDING: Wie bei planaren Flächen
                            u_min -= step
                            u_max += step
                            v_min -= step
                            v_max += step
                            
                            # Y wird später interpoliert
                            from scipy.interpolate import griddata
                            
                            # Erstelle Grid in (u,v)-Ebene = (x,z)-Ebene
                            u_axis = np.arange(u_min, u_max + step, step, dtype=float)
                            v_axis = np.arange(v_min, v_max + step, step, dtype=float)
                            
                            if u_axis.size < 2 or v_axis.size < 2:
                                raise ValueError(f"Vertikale Surface '{geometry.surface_id}' (schräg, x-z) liefert zu wenige Punkte in (u,v)-Ebene.")
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
                                
                else:
                    # Surface wurde als "vertical" klassifiziert, aber Z-Spanne ist nicht groß genug
                    raise ValueError(f"Surface '{geometry.surface_id}': Als 'vertical' klassifiziert, aber Z-Spanne ({z_span:.3f}) nicht groß genug. x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}")
        else:
            # PLANARE/SCHRÄGE SURFACES: Grid in X-Y-Ebene (wie bisher)
            if not geometry.bbox:
                raise ValueError(f"Surface '{geometry.surface_id}': keine Bounding Box vorhanden.")
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
                    
                    if DEBUG_FLEXIBLE_GRID:
                        print(f"[DEBUG Grid] Surface '{geometry.surface_id}': Adaptive Resolution {adaptive_resolution:.3f} m (Basis: {resolution:.3f} m)")
                    
                    resolution = adaptive_resolution
            
            # 🎯 ERWEITERE GRID: Um genau einen Punkt (resolution) über den Rand hinaus
            min_x -= resolution
            max_x += resolution
            min_y -= resolution
            max_y += resolution
            
            # 🎯 ADAPTIVE RESOLUTION: Höhere Dichte am Rand für schärfere Plots
            # Erstelle feineres Grid am Rand (z.B. halbe Resolution) für bessere Detailwiedergabe
            edge_resolution_factor = 0.5  # Halbe Resolution am Rand = doppelte Dichte
            edge_resolution = resolution * edge_resolution_factor
            edge_width = resolution * 2.0  # Randbereich: 2x Resolution (ca. 2m bei 1m Resolution)
            
            # Erstelle Basis-Grid mit normaler Resolution
            sound_field_x_coarse = np.arange(min_x, max_x + resolution, resolution)
            sound_field_y_coarse = np.arange(min_y, max_y + resolution, resolution)
            
            # Erstelle feineres Grid am Rand
            # Links/Rechts: Von min_x bis min_x+edge_width und von max_x-edge_width bis max_x
            x_fine_left = np.arange(min_x, min_x + edge_width + edge_resolution, edge_resolution)
            x_fine_right = np.arange(max_x - edge_width, max_x + edge_resolution, edge_resolution)
            x_middle = np.arange(min_x + edge_width, max_x - edge_width + resolution, resolution)
            
            # Oben/Unten: Von min_y bis min_y+edge_width und von max_y-edge_width bis max_y
            y_fine_bottom = np.arange(min_y, min_y + edge_width + edge_resolution, edge_resolution)
            y_fine_top = np.arange(max_y - edge_width, max_y + edge_resolution, edge_resolution)
            y_middle = np.arange(min_y + edge_width, max_y - edge_width + resolution, resolution)
            
            # Kombiniere: Entferne Duplikate (wenn edge_width < resolution könnte es Überschneidungen geben)
            sound_field_x = np.unique(np.concatenate([x_fine_left, x_middle, x_fine_right]))
            sound_field_y = np.unique(np.concatenate([y_fine_bottom, y_middle, y_fine_top]))
            
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
            z_span = float(np.ptp(zs))
            
            eps_line = 1e-6
            
            # 🎯 VERWENDE dominant_axis (muss vorhanden sein)
            if not hasattr(geometry, 'dominant_axis') or not geometry.dominant_axis:
                raise ValueError(f"Surface '{geometry.surface_id}': dominant_axis nicht verfügbar für Maske-Erstellung")
            
            use_yz_wall = False
            if geometry.dominant_axis == "yz":
                use_yz_wall = True
            elif geometry.dominant_axis == "xz":
                use_yz_wall = False
            else:
                raise ValueError(f"Surface '{geometry.surface_id}': Unbekannter dominant_axis '{geometry.dominant_axis}'")
            
            if use_yz_wall:
                # Y-Z-Wand: Maske in (y,z)-Ebene
                U_grid = Y_grid  # u = y
                V_grid = Z_grid  # v = z
                polygon_uv = [
                    {"x": float(p.get("y", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
                # 🎯 KORREKTUR: Verwende _points_in_polygon_batch_uv für (u,v)-Koordinaten
                surface_mask_strict = self._points_in_polygon_batch_uv(U_grid, V_grid, polygon_uv)
                # 🎯 IDENTISCHE DILATATION: Wie bei planaren Flächen
                surface_mask = self._dilate_mask_minimal(surface_mask_strict)
            else:
                # X-Z-Wand: Maske in (x,z)-Ebene
                U_grid = X_grid  # u = x
                V_grid = Z_grid  # v = z
                polygon_uv = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
                # 🎯 KORREKTUR: Verwende _points_in_polygon_batch_uv für (u,v)-Koordinaten
                surface_mask_strict = self._points_in_polygon_batch_uv(U_grid, V_grid, polygon_uv)
                # 🎯 IDENTISCHE DILATATION: Wie bei planaren Flächen
                surface_mask = self._dilate_mask_minimal(surface_mask_strict)
            
            # Z_grid ist bereits korrekt gesetzt (für X-Z-Wand: Z_grid = V_grid, für Y-Z-Wand: Z_grid = V_grid)
            # Keine Z-Interpolation nötig!
            # surface_mask_strict wurde bereits oben erstellt (vor Dilatation)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG Grid-Erweiterung] Surface '{geometry.surface_id}' (VERTIKAL): {points_in_surface}/{total_grid_points} Punkte in Surface")
        else:
            # PLANARE/SCHRÄGE SURFACES: Normale Maske und Z-Interpolation
            surface_mask_strict = self._create_surface_mask(X_grid, Y_grid, geometry)  # Ursprüngliche Maske
            # 🎯 SICHERSTELLEN, DASS POLYGON-ECKEN ERFASST WERDEN:
            # Ergänze die strikte Maske so, dass jede Polygon-Ecke mindestens
            # einem nächstgelegenen Grid-Punkt zugeordnet ist (falls nicht zu weit entfernt).
            surface_mask_strict = self._ensure_vertex_coverage(
                X_grid, Y_grid, geometry, surface_mask_strict
            )
            # 🎯 MASKE ERWEITERN: Damit Rand-Punkte auch aktiv sind und SPL-Werte erhalten
            # Dies ermöglicht Triangulation bis zum Rand der Surface
            surface_mask = self._dilate_mask_minimal(surface_mask_strict)  # Erweiterte Maske für SPL-Werte
            
            # 🎯 DEBUG: Grid-Erweiterung (vor Z-Interpolation, damit total_grid_points verfügbar ist)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG Grid-Erweiterung] Surface '{geometry.surface_id}': {points_in_surface}/{total_grid_points} Punkte in Surface")
            
            # 🎯 Z-INTERPOLATION: Für alle Punkte im Grid (auch außerhalb Surface)
            # Z-Werte linear interpolieren gemäß Plane-Model für erweiterte Punkte
            if Z_grid is None or np.all(Z_grid == 0):
                Z_grid = np.zeros_like(X_grid, dtype=float)
            
            if geometry.plane_model:
                # Berechne Z-Werte für ALLE Punkte im Grid (linear interpoliert gemäß Plane-Model)
                # Dies ermöglicht erweiterte Punkte außerhalb der Surface-Grenze
                Z_values_all = _evaluate_plane_on_grid(geometry.plane_model, X_grid, Y_grid)
                Z_grid = Z_values_all  # Setze für alle Punkte, nicht nur innerhalb Surface
                z_span_grid = float(np.max(Z_grid)) - float(np.min(Z_grid))
                # Nur Warnung ausgeben, wenn Problem erkannt wird
                if geometry.orientation == "sloped" and z_span_grid < 0.01:
                    plane_mode = geometry.plane_model.get('mode', 'unknown')
            else:
                # Fallback: Kein Plane-Model – nutze vorhandene Surface-Punkte
                surface_points = geometry.points or []
                if not surface_points:
                    raise ValueError(f"Surface '{geometry.surface_id}': kein Plane-Model und keine Surface-Punkte vorhanden.")
                
                surface_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                surface_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                surface_z = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                z_variation = bool(np.ptp(surface_z) > 1e-9)
                
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
                                    raise ValueError(f"Surface '{geometry.surface_id}': zu wenige eindeutige x-Werte für Z-Interpolation.")
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
                                raise RuntimeError(f"Surface '{geometry.surface_id}': Z-Interpolation (X-Z-Wand) fehlgeschlagen: {e}")
                            
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
                                    raise ValueError(f"Surface '{geometry.surface_id}': zu wenige eindeutige y-Werte für Z-Interpolation.")
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
                                raise RuntimeError(f"Surface '{geometry.surface_id}': Z-Interpolation (Y-Z-Wand) fehlgeschlagen: {e}")
                        else:
                            # Fallback: Verwende konstanten Z-Wert
                            raise ValueError(f"Surface '{geometry.surface_id}': unklare Orientierung (x_span={x_span:.6f}, y_span={y_span:.6f}) – keine Z-Interpolation.")
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
                                method='linear',
                                fill_value=np.nan  # keine stillen Fallback-Werte
                            )
                            if np.isnan(Z_interp).any():
                                raise ValueError("Z-Interpolation liefert NaN außerhalb des gültigen Polygons.")
                            Z_grid = Z_interp.reshape(X_grid.shape)
                            print(f"  └─ Z-Werte linear interpoliert für ALLE {total_grid_points} Punkte (basierend auf Surface-Punkten)")
                        except Exception as e:
                            raise RuntimeError(f"Z-Interpolation (linear) fehlgeschlagen: {e}")
                else:
                    # Alle Surface-Punkte haben gleichen Z-Wert
                    Z_grid.fill(surface_z[0])
                    # print(f"  └─ Z-Werte auf konstanten Wert {surface_z[0]:.3f} gesetzt für ALLE {total_grid_points} Punkte")
        
        # if PERF_ENABLED and t_orientation_start is not None:
        #     duration_ms = (time.perf_counter() - t_orientation_start) * 1000.0
        #     print(
        #         f"[PERF] GridBuilder.build_single_surface_grid.{geometry.orientation}: "
        #         f"{duration_ms:.2f} ms (surface={geometry.surface_id})"
        #     )
        
        # Für planare/schräge Surfaces: Gebe auch die strikte Maske zurück (ohne Erweiterung)
        # Für vertikale Surfaces: surface_mask_strict wurde bereits vorher erstellt (vor Dilatation)
        if geometry.orientation in ("planar", "sloped"):
            # surface_mask_strict wurde bereits vorher erstellt
            return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict)
        else:
            # 🎯 KORREKTUR: Vertikale Surfaces haben auch eine strikte Maske (vor Dilatation erstellt)
            # surface_mask_strict wurde bereits oben erstellt (vor Dilatation in Zeile 1598/1610)
            return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict)


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
        # Cache für per-Surface-Grids (inkl. Triangulation), um bei wiederholten
        # Berechnungen mit unveränderter Geometrie Zeit zu sparen.
        # Key: (surface_id, orientation, resolution, min_points, points_signature)
        self._surface_grid_cache: Dict[tuple, SurfaceGrid] = {}
        self._cache_lock = Lock()  # Lock für thread-sicheren Cache-Zugriff
    
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
        
        # 🚀 OPTIMIERUNG: Prüfe Cache für alle Surfaces vor der Parallelisierung
        surface_grids: Dict[str, SurfaceGrid] = {}
        geometries_to_process = []
        
        for geom in geometries:
            cache_key = self._make_surface_cache_key(
                geom=geom,
                resolution=float(resolution),
                min_points=min_points_per_dimension,
            )
            with self._cache_lock:
                cached_grid = self._surface_grid_cache.get(cache_key)
            if cached_grid is not None:
                surface_grids[geom.surface_id] = cached_grid
            else:
                geometries_to_process.append((geom, cache_key))
        
        # 🚀 PARALLELISIERUNG: Verarbeite nicht-gecachte Surfaces parallel
        if geometries_to_process:
            # Bestimme Anzahl Worker basierend auf Settings oder verwende Default
            max_workers = int(getattr(self.settings, "spl_parallel_surfaces", 0) or 0)
            if max_workers <= 0:
                max_workers = None  # ThreadPoolExecutor wählt automatisch
            
            with perf_section("FlexibleGridGenerator.generate_per_surface.parallel", 
                            task_count=len(geometries_to_process)):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for geom, cache_key in geometries_to_process:
                        future = executor.submit(
                            self._process_single_geometry,
                            geom=geom,
                            cache_key=cache_key,
                            resolution=resolution,
                            min_points_per_dimension=min_points_per_dimension
                        )
                        futures[future] = (geom.surface_id, cache_key)
                    
                    # Sammle Ergebnisse
                    for future in as_completed(futures):
                        surface_id, cache_key = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                with self._cache_lock:
                                    self._surface_grid_cache[cache_key] = result
                                surface_grids[surface_id] = result
                        except Exception as e:
                            error_type = type(e).__name__
                            print(f"⚠️  [FlexibleGridGenerator] Surface '{surface_id}' übersprungen ({error_type}): {e}")
        
        return surface_grids
    
    def _process_single_geometry(
        self,
        geom: SurfaceGeometry,
        cache_key: tuple,
        resolution: float,
        min_points_per_dimension: int
    ) -> Optional[SurfaceGrid]:
        """
        Verarbeitet eine einzelne Surface-Geometrie und erstellt das SurfaceGrid.
        Diese Methode kann parallel aufgerufen werden.
        """
        # 🛠️ VORBEREITUNG: Für "sloped" Flächen plane_model aus Punkten ableiten, BEVOR Grid erstellt wird
        # Dies stellt sicher, dass build_single_surface_grid das korrekte plane_model verwendet
        if geom.orientation == "sloped":
            if geom.points:
                try:
                    # Extrahiere Z-Werte aus Punkten zur Validierung
                    points_z = np.array([p.get('z', 0.0) for p in geom.points], dtype=float)
                    z_span_points = float(np.ptp(points_z)) if len(points_z) > 0 else 0.0
                    
                    plane_model_new, error_msg = derive_surface_plane(geom.points)
                    if plane_model_new is not None:
                        geom.plane_model = plane_model_new
                        mode = plane_model_new.get('mode', 'unknown')
                        # Nur Warnungen ausgeben, wenn Probleme erkannt werden
                        if mode == "constant":
                            pass
                        elif mode in ("x", "y") and abs(plane_model_new.get('slope', 0.0)) < 1e-6:
                            pass
                    else:
                        pass
                except Exception as e:
                    pass
            else:
                pass
        
        try:
            result = self.builder.build_single_surface_grid(
                    geometry=geom,
                    resolution=resolution,
                    min_points_per_dimension=min_points_per_dimension
                )
            if len(result) == 7:
                # Neue Version: Mit strikter Maske
                (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict) = result
            else:
                # Alte Version: Ohne strikte Maske (Fallback für vertikale)
                (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = result
                surface_mask_strict = surface_mask  # Für vertikale identisch
        except (ValueError, Exception) as e:
            # Fange alle Fehler ab (ValueError, QhullError, etc.) und überspringe die Surface
            # Kein Fallback - nur die funktionierenden Surfaces werden verwendet
            error_type = type(e).__name__
            # Immer Warnung ausgeben, wenn Surface übersprungen wird (auch ohne DEBUG_FLEXIBLE_GRID)
            print(f"⚠️  [FlexibleGridGenerator] Surface '{geom.surface_id}' übersprungen ({error_type}): {e}")
            return None
        
        # 🛠️ KORREKTUR: Z-Koordinaten bei planaren/schrägen Flächen aus der Ebene berechnen
        # Falls plane_model fehlt oder nahezu flache Z-Spanne ergibt, versuche Neu-Bestimmung aus den Punkten.
        if geom.orientation in ("planar", "sloped"):
            plane_model_local = geom.plane_model
            # Für "sloped" Flächen: IMMER plane_model aus Punkten ableiten, wenn nicht vorhanden ODER wenn vorhanden aber falsch
            # Für "planar" Flächen: Nur ableiten, wenn nicht vorhanden
            if geom.orientation == "sloped":
                # Für schräge Flächen: Immer aus Punkten ableiten, um sicherzustellen, dass es korrekt ist
                if geom.points:
                    plane_model_local, _ = derive_surface_plane(geom.points)
                    if plane_model_local is None:
                        raise ValueError(f"Surface '{geom.surface_id}' (sloped): plane_model konnte nicht abgeleitet werden")
                    geom.plane_model = plane_model_local
                else:
                    pass
            else:
                # Für planare Flächen: Nur ableiten, wenn nicht vorhanden
                if plane_model_local is None and geom.points:
                    plane_model_local, _ = derive_surface_plane(geom.points)
                    if plane_model_local is None:
                        raise ValueError(f"Surface '{geom.surface_id}' (planar): plane_model konnte nicht abgeleitet werden")
                    geom.plane_model = plane_model_local
            
            if plane_model_local:
                try:
                    z_old_span = float(Z_grid.max()) - float(Z_grid.min())
                    Z_eval = _evaluate_plane_on_grid(plane_model_local, X_grid, Y_grid)
                    if Z_eval is not None and Z_eval.shape == X_grid.shape:
                        Z_grid = Z_eval
                        zmin, zmax = float(Z_grid.min()), float(Z_grid.max())
                        z_span = zmax - zmin
                        # Nur Warnung ausgeben, wenn Problem erkannt wird
                        if geom.orientation == "sloped" and z_span < 0.01:
                            pass
                except Exception as e:
                    pass
            else:
                if geom.orientation == "sloped":
                    pass
        
        # Berechne tatsächliche Resolution (kann adaptiv sein)
        ny, nx = X_grid.shape
        if len(sound_field_x) > 1 and len(sound_field_y) > 1:
            actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
            actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
            actual_resolution = (actual_resolution_x + actual_resolution_y) / 2.0
        else:
            actual_resolution = resolution
        
        # 🎯 TRIANGULATION: Erstelle triangulierte Vertices aus Grid-Punkten
        # Pro Grid-Punkt werden Vertices verwendet, pro Grid-Quadrat 2 Dreiecke erstellt
        triangulated_vertices = None
        triangulated_faces = None
        triangulated_success = False
        vertex_source_indices: Optional[np.ndarray] = None
        
        try:
            # Verwende Grid-Punkte innerhalb der Surface-Maske als Vertices
            ny, nx = X_grid.shape
            mask_flat = surface_mask.ravel()  # Erweiterte Maske (für SPL-Werte)
            mask_strict_flat = surface_mask_strict.ravel()  # Strikte Maske (für Face-Filterung)
            
            if np.any(mask_flat):
                # Erstelle Vertex-Koordinaten aus allen Grid-Punkten (auch inaktive für konsistente Indizes)
                # Wir brauchen alle Punkte für die strukturierte Triangulation
                all_vertices = np.column_stack([
                    X_grid.ravel(),  # X-Koordinaten
                    Y_grid.ravel(),  # Y-Koordinaten
                    Z_grid.ravel()   # Z-Koordinaten
                ])  # Shape: (ny * nx, 3)
                
                # 🎯 ZUSÄTZLICHE VERTICES AN POLYGON-ECKEN HINZUFÜGEN (für höhere Auflösung)
                # Dies erhöht die Polygon-Dichte an den Ecken, um Lücken zu vermeiden
                # 🎯 IDENTISCHE BEHANDLUNG: Für alle Orientierungen (planar, sloped, vertical)
                additional_vertices = []
                surface_points = geom.points or []
                if len(surface_points) >= 3:
                    # Bestimme Koordinatensystem basierend auf Orientierung
                    if geom.orientation == "vertical":
                        # 🎯 VERTIKALE FLÄCHEN: Verwende (u,v)-Koordinaten basierend auf dominant_axis
                        xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                        ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                        zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                        x_span = float(np.ptp(xs))
                        y_span = float(np.ptp(ys))
                        eps_line = 1e-6
                        
                        # 🎯 ZUERST AUSRICHTUNG PRÜFEN: Bestimme (u,v)-Koordinaten basierend auf dominant_axis oder Spannen-Analyse
                        # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende dominant_axis wenn verfügbar
                        # Berechne Konstanten-Werte im Voraus
                        x_mean = float(np.mean(xs))
                        y_mean = float(np.mean(ys))
                        z_span = float(np.ptp(zs))
                        
                        # 🎯 PRÜFE OB SCHRÄGE WAND: Y variiert bei X-Z-Wänden, X variiert bei Y-Z-Wänden
                        is_slanted_wall = False
                        
                        if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                            # Verwende dominant_axis als primäre Quelle (konsistent mit Grid-Erstellung)
                            if geom.dominant_axis == "yz":
                                # Y-Z-Wand: u = y, v = z
                                polygon_u = ys
                                polygon_v = zs
                                is_xz_wall = False
                                # Prüfe ob schräg: X variiert (gleiche Logik wie Grid-Erstellung)
                                is_slanted_wall = (x_span > eps_line and z_span > 1e-3)
                                if DEBUG_FLEXIBLE_GRID:
                                    print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': Y-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                            elif geom.dominant_axis == "xz":
                                # X-Z-Wand: u = x, v = z
                                polygon_u = xs
                                polygon_v = zs
                                is_xz_wall = True
                                # Prüfe ob schräg: Y variiert (gleiche Logik wie Grid-Erstellung)
                                is_slanted_wall = (y_span > eps_line and z_span > 1e-3)
                                if DEBUG_FLEXIBLE_GRID:
                                    print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': X-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                            else:
                                raise ValueError(f"Surface '{geom.surface_id}': Unbekannter dominant_axis '{geom.dominant_axis}'")
                            
                            # 🎯 FÜR SCHRÄGE WÄNDE: Interpoliere Y (X-Z-Wand) bzw. X (Y-Z-Wand) aus Surface-Punkten
                            if is_slanted_wall:
                                from scipy.interpolate import griddata
                                if is_xz_wall:
                                    # X-Z-Wand schräg: Interpoliere Y aus (x,z)-Koordinaten
                                    points_surface = np.column_stack([xs, zs])
                                else:
                                    # Y-Z-Wand schräg: Interpoliere X aus (y,z)-Koordinaten
                                    points_surface = np.column_stack([ys, zs])
                            
                            # Prüfe für jede Polygon-Ecke in (u,v)-Koordinaten
                            existing_vertex_tolerance = resolution * 0.1  # Toleranz für "bereits vorhanden"
                            
                            for corner_u, corner_v in zip(polygon_u, polygon_v):
                                # Prüfe ob bereits ein Vertex sehr nahe an dieser Ecke existiert
                                # Für vertikale Flächen: Vergleiche in (u,v)-Koordinaten
                                if is_xz_wall:
                                    # X-Z-Wand: u = x, v = z
                                    distances = np.sqrt((all_vertices[:, 0] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                                else:
                                    # Y-Z-Wand: u = y, v = z
                                    distances = np.sqrt((all_vertices[:, 1] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                                
                                min_distance = np.min(distances)
                                
                                if min_distance > existing_vertex_tolerance:
                                    # Kein Vertex nahe genug → füge neuen Vertex hinzu
                                    # Transformiere (u,v) zurück zu (x,y,z)
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        corner_x = corner_u
                                        corner_z = corner_v
                                        if is_slanted_wall:
                                            # Schräge Wand: Y interpoliert aus (x,z)
                                            from scipy.interpolate import griddata
                                            corner_y = griddata(
                                                points_surface, ys,
                                                np.array([[corner_u, corner_v]]),
                                                method='linear', fill_value=y_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(additional_vertices) == 0:
                                                print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': Y interpoliert für Ecke (u={corner_u:.3f}, v={corner_v:.3f}) → Y={corner_y:.3f} (linear)")
                                        else:
                                            # Konstante Wand: Y = konstant
                                            corner_y = y_mean
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        corner_y = corner_u
                                        corner_z = corner_v
                                        if is_slanted_wall:
                                            # Schräge Wand: X interpoliert aus (y,z)
                                            from scipy.interpolate import griddata
                                            corner_x = griddata(
                                                points_surface, xs,
                                                np.array([[corner_u, corner_v]]),
                                                method='linear', fill_value=x_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(additional_vertices) == 0:
                                                print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': X interpoliert für Ecke (u={corner_u:.3f}, v={corner_v:.3f}) → X={corner_x:.3f} (linear)")
                                        else:
                                            # Konstante Wand: X = konstant
                                            corner_x = x_mean
                                    
                                    additional_vertices.append([corner_x, corner_y, corner_z])
                    else:
                        # PLANARE/SCHRÄGE FLÄCHEN: Verwende (x,y)-Koordinaten
                        polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                        polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                        
                        # Prüfe für jede Polygon-Ecke, ob nahe Grid-Punkte existieren
                        existing_vertex_tolerance = resolution * 0.1  # Toleranz für "bereits vorhanden"
                        
                        for corner_x, corner_y in zip(polygon_x, polygon_y):
                            # Prüfe ob bereits ein Vertex sehr nahe an dieser Ecke existiert
                            distances = np.sqrt((all_vertices[:, 0] - corner_x)**2 + (all_vertices[:, 1] - corner_y)**2)
                            min_distance = np.min(distances)
                            
                            if min_distance > existing_vertex_tolerance:
                                # Kein Vertex nahe genug → füge neuen Vertex hinzu
                                # Berechne Z-Koordinate für Ecke
                                corner_z = 0.0
                                if geom.plane_model:
                                    try:
                                        z_new = _evaluate_plane_on_grid(
                                            geom.plane_model,
                                            np.array([[corner_x]]),
                                            np.array([[corner_y]])
                                        )
                                        if z_new is not None and z_new.size > 0:
                                            corner_z = float(z_new.flat[0])
                                    except Exception:
                                        pass
                                
                                additional_vertices.append([corner_x, corner_y, corner_z])
                    
                    if len(additional_vertices) > 0:
                        additional_vertices_array = np.array(additional_vertices, dtype=float)
                        all_vertices = np.vstack([all_vertices, additional_vertices_array])
                        # print(f"  └─ ✅ {len(additional_vertices)} zusätzliche Vertices an Polygon-Ecken hinzugefügt ({geom.orientation})")
                
                # Speichere Offset für zusätzliche Vertices (alle Grid-Punkte kommen zuerst)
                base_vertex_count = X_grid.size
                additional_vertex_start_idx = base_vertex_count
                additional_vertices_array = np.array(additional_vertices, dtype=float) if len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
                
                # 🎯 RAND-VERTICES AUF SURFACE-GRENZE PROJIZIEREN
                # Für Rand-Vertices (in erweiterter Maske, aber nicht in strikter Maske):
                # Verschiebe sie auf die Surface-Grenze, damit Polygone exakt am Rand verlaufen
                # 🎯 IDENTISCHE BEHANDLUNG: Für alle Orientierungen (planar, sloped, vertical)
                # Berechne Rand-Vertices: in erweitert, aber nicht in strikt
                is_on_boundary = mask_flat & (~mask_strict_flat)
                boundary_indices = np.where(is_on_boundary)[0]
                
                if len(boundary_indices) > 0:
                    # Lade Surface-Polygon für Projektion
                    surface_points = geom.points or []
                    if len(surface_points) >= 3:
                        if geom.orientation == "vertical":
                            # 🎯 VERTIKALE FLÄCHEN: Projektion in (u,v)-Koordinaten
                            xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                            ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                            zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                            x_span = float(np.ptp(xs))
                            y_span = float(np.ptp(ys))
                            eps_line = 1e-6
                            
                            # 🎯 ZUERST AUSRICHTUNG PRÜFEN: Bestimme (u,v)-Koordinaten basierend auf dominant_axis oder Spannen-Analyse
                            # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende dominant_axis wenn verfügbar
                            z_span = float(np.ptp(zs))
                            x_mean = float(np.mean(xs))
                            y_mean = float(np.mean(ys))
                            
                            # 🎯 PRÜFE OB SCHRÄGE WAND: Y variiert bei X-Z-Wänden, X variiert bei Y-Z-Wänden
                            # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende gleiche Logik wie is_slanted_vertical
                            is_slanted_wall = False
                            
                            if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                # Verwende dominant_axis als primäre Quelle (konsistent mit Grid-Erstellung)
                                if geom.dominant_axis == "yz":
                                    # Y-Z-Wand: u = y, v = z
                                    polygon_u = ys
                                    polygon_v = zs
                                    is_xz_wall = False
                                    # Prüfe ob schräg: X variiert (gleiche Logik wie Grid-Erstellung)
                                    is_slanted_wall = (x_span > eps_line and z_span > 1e-3)
                                    if DEBUG_FLEXIBLE_GRID:
                                        print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': Y-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                elif geom.dominant_axis == "xz":
                                    # X-Z-Wand: u = x, v = z
                                    polygon_u = xs
                                    polygon_v = zs
                                    is_xz_wall = True
                                    # Prüfe ob schräg: Y variiert (gleiche Logik wie Grid-Erstellung)
                                    is_slanted_wall = (y_span > eps_line and z_span > 1e-3)
                                    if DEBUG_FLEXIBLE_GRID:
                                        print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': X-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                else:
                                    raise ValueError(f"Surface '{geom.surface_id}': Unbekannter dominant_axis '{geom.dominant_axis}'")
                                
                                # 🎯 FÜR SCHRÄGE WÄNDE: Interpoliere Y (X-Z-Wand) bzw. X (Y-Z-Wand) aus Surface-Punkten
                                if is_slanted_wall:
                                    from scipy.interpolate import griddata
                                    if is_xz_wall:
                                        # X-Z-Wand schräg: Interpoliere Y aus (x,z)-Koordinaten
                                        points_surface = np.column_stack([xs, zs])
                                    else:
                                        # Y-Z-Wand schräg: Interpoliere X aus (y,z)-Koordinaten
                                        points_surface = np.column_stack([ys, zs])
                                
                                # Projiziere jeden Rand-Vertex auf die nächstliegende Polygon-Kante oder Ecke in (u,v)
                                for idx in boundary_indices:
                                    v = all_vertices[idx]
                                    
                                    # Extrahiere (u,v)-Koordinaten aus Vertex
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        vu, vv = v[0], v[2]
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        vu, vv = v[1], v[2]
                                    
                                    # Finde nächstliegenden Punkt auf Polygon-Rand (Kante oder Ecke) in (u,v)
                                    min_dist_sq = np.inf
                                    closest_u, closest_v = vu, vv
                                    
                                    n_poly = len(polygon_u)
                                    
                                    # 🎯 ZUERST: Prüfe Polygon-Ecken (für bessere Abdeckung an scharfen Ecken)
                                    corner_threshold = 1e-6  # Sehr kleine Schwellenwert für "nahe an Ecke"
                                    for i in range(n_poly):
                                        corner_u, corner_v = polygon_u[i], polygon_v[i]
                                        dist_sq_to_corner = (vu - corner_u)**2 + (vv - corner_v)**2
                                        
                                        # Wenn Vertex sehr nahe an einer Polygon-Ecke ist, projiziere direkt auf Ecke
                                        if dist_sq_to_corner < corner_threshold:
                                            closest_u, closest_v = corner_u, corner_v
                                            min_dist_sq = dist_sq_to_corner
                                            break  # Direkte Ecken-Projektion hat Priorität
                                        
                                        # Wenn noch kein guter Kandidat gefunden, prüfe ob Ecke näher ist als bisher
                                        if dist_sq_to_corner < min_dist_sq:
                                            min_dist_sq = dist_sq_to_corner
                                            closest_u, closest_v = corner_u, corner_v
                                    
                                    # Dann: Prüfe alle Polygon-Kanten (kann bessere Projektion als Ecke liefern)
                                    for i in range(n_poly):
                                        p1 = np.array([polygon_u[i], polygon_v[i]])
                                        p2 = np.array([polygon_u[(i + 1) % n_poly], polygon_v[(i + 1) % n_poly]])
                                        
                                        # Berechne Projektion von v auf Kante (p1, p2) in (u,v)
                                        edge = p2 - p1
                                        edge_len_sq = np.dot(edge, edge)
                                        if edge_len_sq < 1e-12:
                                            continue  # Degenerierte Kante
                                        
                                        t = np.dot([vu - p1[0], vv - p1[1]], edge) / edge_len_sq
                                        t = np.clip(t, 0.0, 1.0)  # Clamp auf Kante
                                        proj = p1 + t * edge
                                        
                                        # Berechne Abstand
                                        dist_sq = (vu - proj[0])**2 + (vv - proj[1])**2
                                        
                                        # Verwende Kanten-Projektion nur wenn sie besser ist als Ecken-Projektion
                                        # (außer wir haben bereits eine sehr nahe Ecken-Projektion gefunden)
                                        if dist_sq < min_dist_sq or (min_dist_sq >= corner_threshold and dist_sq < min_dist_sq * 1.1):
                                            min_dist_sq = dist_sq
                                            closest_u, closest_v = proj[0], proj[1]
                                    
                                    # Transformiere (u,v) zurück zu (x,y,z) und verschiebe Vertex
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        all_vertices[idx, 0] = closest_u
                                        all_vertices[idx, 2] = closest_v
                                        if is_slanted_wall:
                                            # Schräge Wand: Y interpoliert aus (x,z)
                                            old_y = all_vertices[idx, 1]
                                            all_vertices[idx, 1] = griddata(
                                                points_surface, ys,
                                                np.array([[closest_u, closest_v]]),
                                                method='linear', fill_value=y_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(boundary_indices) > 0 and idx == boundary_indices[0]:
                                                print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': Y interpoliert für Rand-Vertex (u={closest_u:.3f}, v={closest_v:.3f}) → Y={all_vertices[idx, 1]:.3f} (linear, vorher: {old_y:.3f})")
                                        else:
                                            # Konstante Wand: Y = konstant
                                            all_vertices[idx, 1] = y_mean
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        all_vertices[idx, 1] = closest_u
                                        all_vertices[idx, 2] = closest_v
                                        if is_slanted_wall:
                                            # Schräge Wand: X interpoliert aus (y,z)
                                            old_x = all_vertices[idx, 0]
                                            all_vertices[idx, 0] = griddata(
                                                points_surface, xs,
                                                np.array([[closest_u, closest_v]]),
                                                method='linear', fill_value=x_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(boundary_indices) > 0 and idx == boundary_indices[0]:
                                                print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': X interpoliert für Rand-Vertex (u={closest_u:.3f}, v={closest_v:.3f}) → X={all_vertices[idx, 0]:.3f} (linear, vorher: {old_x:.3f})")
                                        else:
                                            # Konstante Wand: X = konstant
                                            all_vertices[idx, 0] = x_mean
                        else:
                            # PLANARE/SCHRÄGE FLÄCHEN: Projektion in (x,y)-Koordinaten
                            # Extrahiere Polygon-Koordinaten (für planare/schräge: x,y)
                            polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                            polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                            
                            # Projiziere jeden Rand-Vertex auf die nächstliegende Polygon-Kante oder Ecke
                            for idx in boundary_indices:
                                v = all_vertices[idx]
                                vx, vy = v[0], v[1]
                                
                                # Finde nächstliegenden Punkt auf Polygon-Rand (Kante oder Ecke)
                                min_dist_sq = np.inf
                                closest_x, closest_y = vx, vy
                                
                                n_poly = len(polygon_x)
                                
                                # 🎯 ZUERST: Prüfe Polygon-Ecken (für bessere Abdeckung an scharfen Ecken)
                                corner_threshold = 1e-6  # Sehr kleine Schwellenwert für "nahe an Ecke"
                                for i in range(n_poly):
                                    corner_x, corner_y = polygon_x[i], polygon_y[i]
                                    dist_sq_to_corner = (vx - corner_x)**2 + (vy - corner_y)**2
                                    
                                    # Wenn Vertex sehr nahe an einer Polygon-Ecke ist, projiziere direkt auf Ecke
                                    if dist_sq_to_corner < corner_threshold:
                                        closest_x, closest_y = corner_x, corner_y
                                        min_dist_sq = dist_sq_to_corner
                                        break  # Direkte Ecken-Projektion hat Priorität
                                    
                                    # Wenn noch kein guter Kandidat gefunden, prüfe ob Ecke näher ist als bisher
                                    if dist_sq_to_corner < min_dist_sq:
                                        min_dist_sq = dist_sq_to_corner
                                        closest_x, closest_y = corner_x, corner_y
                                
                                # Dann: Prüfe alle Polygon-Kanten (kann bessere Projektion als Ecke liefern)
                                for i in range(n_poly):
                                    p1 = np.array([polygon_x[i], polygon_y[i]])
                                    p2 = np.array([polygon_x[(i + 1) % n_poly], polygon_y[(i + 1) % n_poly]])
                                    
                                    # Berechne Projektion von v auf Kante (p1, p2)
                                    edge = p2 - p1
                                    edge_len_sq = np.dot(edge, edge)
                                    if edge_len_sq < 1e-12:
                                        continue  # Degenerierte Kante
                                    
                                    t = np.dot([vx - p1[0], vy - p1[1]], edge) / edge_len_sq
                                    t = np.clip(t, 0.0, 1.0)  # Clamp auf Kante
                                    proj = p1 + t * edge
                                    
                                    # Berechne Abstand
                                    dist_sq = (vx - proj[0])**2 + (vy - proj[1])**2
                                    
                                    # Verwende Kanten-Projektion nur wenn sie besser ist als Ecken-Projektion
                                    # (außer wir haben bereits eine sehr nahe Ecken-Projektion gefunden)
                                    if dist_sq < min_dist_sq or (min_dist_sq >= corner_threshold and dist_sq < min_dist_sq * 1.1):
                                        min_dist_sq = dist_sq
                                        closest_x, closest_y = proj[0], proj[1]
                                
                                # Verschiebe Vertex auf nächstliegenden Punkt auf Surface-Grenze
                                all_vertices[idx, 0] = closest_x
                                all_vertices[idx, 1] = closest_y
                                # Berechne Z-Koordinate neu basierend auf Plane-Model
                                if geom.plane_model:
                                    try:
                                        # Verwende _evaluate_plane_on_grid für konsistente Z-Berechnung
                                        z_new = _evaluate_plane_on_grid(
                                            geom.plane_model,
                                            np.array([[closest_x]]),
                                            np.array([[closest_y]])
                                        )
                                        if z_new is not None and z_new.size > 0:
                                            all_vertices[idx, 2] = float(z_new.flat[0])
                                    except Exception:
                                        pass  # Falls Berechnung fehlschlägt, behalte alte Z-Koordinate
                
                # Erstelle Index-Mapping: (i, j) → linearer Index
                # Für ein strukturiertes Grid: index = i * nx + j
                
                # Erstelle Faces: Pro Grid-Quadrat bis zu 2 Dreiecke
                # Ein Quadrat hat Ecken: (i,j), (i,j+1), (i+1,j), (i+1,j+1)
                # Dreieck 1: (i,j) → (i,j+1) → (i+1,j)
                # Dreieck 2: (i,j+1) → (i+1,j+1) → (i+1,j)
                faces_list = []
                active_quads = 0
                partial_quads = 0  # Quadrate mit 3 aktiven Ecken
                filtered_out = 0  # Dreiecke außerhalb der strikten Maske
                
                for i in range(ny - 1):
                    for j in range(nx - 1):
                        idx_tl = i * nx + j           # Top-Left
                        idx_tr = i * nx + (j + 1)     # Top-Right
                        idx_bl = (i + 1) * nx + j     # Bottom-Left
                        idx_br = (i + 1) * nx + (j + 1)  # Bottom-Right
                        
                        # Prüfe mit erweiterter Maske (für SPL-Werte)
                        tl_active = mask_flat[idx_tl]
                        tr_active = mask_flat[idx_tr]
                        bl_active = mask_flat[idx_bl]
                        br_active = mask_flat[idx_br]
                        
                        # Prüfe mit strikter Maske (für Face-Filterung)
                        tl_in_strict = mask_strict_flat[idx_tl]
                        tr_in_strict = mask_strict_flat[idx_tr]
                        bl_in_strict = mask_strict_flat[idx_bl]
                        br_in_strict = mask_strict_flat[idx_br]
                        
                        active_count = sum([tl_active, tr_active, bl_active, br_active])
                        strict_count = sum([tl_in_strict, tr_in_strict, bl_in_strict, br_in_strict])
                        
                        # Nur Quadrate verarbeiten, wenn mindestens eine Ecke aktiv ist (erweiterte Maske)
                        # Nach Projektion der Rand-Vertices können wir alle aktiven Dreiecke erstellen
                        if active_count == 0:
                            # Alle Ecken inaktiv → überspringe komplett
                            continue
                        
                        # Hilfsfunktion: Prüft ob ein Dreieck erstellt werden soll
                        # Nach Projektion der Rand-Vertices auf die Surface-Grenze können wir
                        # alle Dreiecke erstellen, die mindestens einen aktiven Vertex haben
                        # (entweder in strikter Maske oder projizierter Rand-Vertex)
                        def should_create_triangle(v1_in_strict, v2_in_strict, v3_in_strict, v1_active, v2_active, v3_active):
                            strict_vertices = sum([v1_in_strict, v2_in_strict, v3_in_strict])
                            active_vertices = sum([v1_active, v2_active, v3_active])
                            # Erstelle Dreieck wenn mindestens 1 aktiver Vertex vorhanden ist
                            # (nach Projektion liegen Rand-Vertices auf der Surface-Grenze)
                            return active_vertices >= 1
                        
                        # Fall 1: Alle 4 Ecken aktiv → 2 Dreiecke
                        if active_count == 4:
                            # Dreieck 1: Top-Left → Top-Right → Bottom-Left
                            if should_create_triangle(tl_in_strict, tr_in_strict, bl_in_strict, tl_active, tr_active, bl_active):
                                faces_list.extend([3, idx_tl, idx_tr, idx_bl])
                            else:
                                filtered_out += 1
                            # Dreieck 2: Top-Right → Bottom-Right → Bottom-Left
                            if should_create_triangle(tr_in_strict, br_in_strict, bl_in_strict, tr_active, br_active, bl_active):
                                faces_list.extend([3, idx_tr, idx_br, idx_bl])
                            else:
                                filtered_out += 1
                            active_quads += 1
                        
                        # Fall 2: Genau 3 Ecken aktiv → 1 Dreieck (bis zum Rand)
                        elif active_count == 3:
                            # Bestimme welche Ecke fehlt und erstelle 1 Dreieck mit den 3 aktiven Ecken
                            # Diese Dreiecke füllen die Fläche bis zum Rand
                            if not tl_active:
                                # Fehlende Ecke: Top-Left → Dreieck mit tr, bl, br
                                if should_create_triangle(tr_in_strict, bl_in_strict, br_in_strict, tr_active, bl_active, br_active):
                                    faces_list.extend([3, idx_tr, idx_br, idx_bl])
                                else:
                                    filtered_out += 1
                            elif not tr_active:
                                # Fehlende Ecke: Top-Right → Dreieck mit tl, bl, br
                                if should_create_triangle(tl_in_strict, bl_in_strict, br_in_strict, tl_active, bl_active, br_active):
                                    faces_list.extend([3, idx_tl, idx_bl, idx_br])
                                else:
                                    filtered_out += 1
                            elif not bl_active:
                                # Fehlende Ecke: Bottom-Left → Dreieck mit tl, tr, br
                                if should_create_triangle(tl_in_strict, tr_in_strict, br_in_strict, tl_active, tr_active, br_active):
                                    faces_list.extend([3, idx_tl, idx_tr, idx_br])
                                else:
                                    filtered_out += 1
                            else:  # not br_active
                                # Fehlende Ecke: Bottom-Right → Dreieck mit tl, tr, bl
                                if should_create_triangle(tl_in_strict, tr_in_strict, bl_in_strict, tl_active, tr_active, bl_active):
                                    faces_list.extend([3, idx_tl, idx_tr, idx_bl])
                                else:
                                    filtered_out += 1
                            partial_quads += 1
                        
                        # 🎯 ZUSÄTZLICHE DREIECKE MIT POLYGON-ECKEN-VERTICES
                        # Erstelle Dreiecke, die zusätzliche Ecken-Vertices verwenden (für höhere Auflösung)
                        if len(additional_vertices) > 0 and active_count >= 1:
                            # Prüfe für jeden zusätzlichen Ecken-Vertex, ob er nahe genug an diesem Quadrat ist
                            for corner_idx, corner_vertex in enumerate(additional_vertices_array):
                                corner_vertex_idx = additional_vertex_start_idx + corner_idx
                                corner_x, corner_y = corner_vertex[0], corner_vertex[1]
                                
                                # Prüfe ob Ecken-Vertex innerhalb oder nahe diesem Quadrat ist
                                x_min, x_max = X_grid[0, j], X_grid[0, j+1] if j+1 < nx else X_grid[0, j]
                                y_min, y_max = Y_grid[i, 0], Y_grid[i+1, 0] if i+1 < ny else Y_grid[i, 0]
                                
                                # Erweitere Bereich um eine Resolution (für nahe Vertices)
                                tolerance = resolution * 1.5
                                if (x_min - tolerance <= corner_x <= x_max + tolerance and
                                    y_min - tolerance <= corner_y <= y_max + tolerance):
                                    
                                    # Erstelle Dreiecke mit Ecken-Vertex + 2 aktive Grid-Ecken
                                    active_corners = []
                                    if tl_active:
                                        active_corners.append((idx_tl, tl_in_strict))
                                    if tr_active:
                                        active_corners.append((idx_tr, tr_in_strict))
                                    if bl_active:
                                        active_corners.append((idx_bl, bl_in_strict))
                                    if br_active:
                                        active_corners.append((idx_br, br_in_strict))
                                    
                                    # Erstelle Dreiecke: Ecken-Vertex + 2 benachbarte aktive Ecken
                                    if len(active_corners) >= 2:
                                        for k in range(len(active_corners) - 1):
                                            idx1, strict1 = active_corners[k]
                                            idx2, strict2 = active_corners[k + 1]
                                            # Ecken-Vertex ist immer aktiv (wird als aktiv betrachtet)
                                            if should_create_triangle(True, strict1, strict2, True, True, True):
                                                faces_list.extend([3, corner_vertex_idx, idx1, idx2])
                        
                        # Fall 3: 2 Ecken aktiv → versuche Dreieck zu erstellen, wenn Ecken benachbart sind
                        # Dies hilft, Ecken-Lücken zu füllen
                        elif active_count == 2:
                            # Finde die 2 aktiven Ecken
                            active_corners = []
                            if tl_active:
                                active_corners.append(('tl', idx_tl, tl_in_strict, tl_active))
                            if tr_active:
                                active_corners.append(('tr', idx_tr, tr_in_strict, tr_active))
                            if bl_active:
                                active_corners.append(('bl', idx_bl, bl_in_strict, bl_active))
                            if br_active:
                                active_corners.append(('br', idx_br, br_in_strict, br_active))
                            
                            if len(active_corners) == 2:
                                corner1_name, corner1_idx, corner1_strict, corner1_active = active_corners[0]
                                corner2_name, corner2_idx, corner2_strict, corner2_active = active_corners[1]
                                
                                # Prüfe ob Ecken benachbart sind (können ein Dreieck mit Rand bilden)
                                # Benachbart: (tl,tr), (tl,bl), (tr,br), (bl,br)
                                adjacent_pairs = {
                                    ('tl', 'tr'): [('bl', idx_bl), ('br', idx_br)],
                                    ('tl', 'bl'): [('tr', idx_tr), ('br', idx_br)],
                                    ('tr', 'br'): [('tl', idx_tl), ('bl', idx_bl)],
                                    ('bl', 'br'): [('tl', idx_tl), ('tr', idx_tr)]
                                }
                                
                                # Normalisiere Reihenfolge
                                pair_key = tuple(sorted([corner1_name, corner2_name]))
                                
                                # Prüfe alle Kombinationen
                                if pair_key in adjacent_pairs:
                                    # Versuche beide möglichen dritten Ecken
                                    for diag_name, diag_idx in adjacent_pairs[pair_key]:
                                        diag_active = mask_flat[diag_idx]
                                        diag_strict = mask_strict_flat[diag_idx]
                                        
                                        # Erstelle Dreieck wenn dritte Ecke auch aktiv ist (projiziert)
                                        if diag_active:
                                            if should_create_triangle(corner1_strict, corner2_strict, diag_strict, corner1_active, corner2_active, diag_active):
                                                faces_list.extend([3, corner1_idx, corner2_idx, diag_idx])
                                                partial_quads += 1
                                                break  # Ein Dreieck reicht für benachbarte Ecken
        
                if len(faces_list) > 0:
                    triangulated_vertices = all_vertices  # Alle Grid-Punkte als Vertices
                    triangulated_faces = np.array(faces_list, dtype=np.int64)
                    triangulated_success = True
                    
                    n_faces = len(faces_list) // 4  # 4 Elemente pro Face: [n, v1, v2, v3]
                    n_vertices_used = len(np.unique(triangulated_faces[1::4]))  # Eindeutige Vertex-Indizes
                    expected_faces = active_quads * 2 + partial_quads  # Volle Quadrate: 2 Dreiecke, Rand-Quadrate: 1 Dreieck
                    
                else:
                    pass
            else:
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            triangulated_success = False
        
        surface_grid = SurfaceGrid(
            surface_id=geom.surface_id,
            sound_field_x=sound_field_x,
            sound_field_y=sound_field_y,
            X_grid=X_grid,
            Y_grid=Y_grid,
            Z_grid=Z_grid,
            surface_mask=surface_mask,
            resolution=actual_resolution,
            geometry=geom,
            triangulated_vertices=triangulated_vertices,
            triangulated_faces=triangulated_faces,
            triangulated_success=triangulated_success,
        )
        
        # 🎯 DEBUG: Prüfen, ob mehrere Vertices sich in 3D überschneiden
        # (d.h. Vertices liegen sehr nahe beieinander in allen drei Dimensionen X, Y, Z).
        # Dies erkennt auch Überlagerungen bei senkrechten Flächen.
        try:
            if triangulated_success and triangulated_vertices is not None:
                verts = np.asarray(triangulated_vertices, dtype=float)
                if verts.shape[0] > 0:
                    # Toleranz: etwa 1% der typischen Grid-Resolution für alle drei Dimensionen
                    # oder ein fester kleiner Wert als Fallback.
                    if actual_resolution and actual_resolution > 0:
                        tol = float(actual_resolution) * 0.01
                    else:
                        tol = 1e-3
                    
                    # Quantisiere X, Y, Z Koordinaten (reduziert numerisches Rauschen)
                    xyz_quant = np.round(verts / tol) * tol
                    
                    # Baue Mapping: quantisierte (X,Y,Z) -> Liste von Vertex-Indizes
                    from collections import defaultdict
                    groups: defaultdict[tuple, list] = defaultdict(list)
                    for idx_v, quant_xyz in enumerate(xyz_quant):
                        groups[(float(quant_xyz[0]), float(quant_xyz[1]), float(quant_xyz[2]))].append(idx_v)
                    
                    # Prüfe, ob mehrere Vertices an derselben quantisierten 3D-Position liegen
                    overlap_count = 0
                    max_distance = 0.0
                    for quant_xyz, idx_list in groups.items():
                        if len(idx_list) <= 1:
                            continue
                        # Berechne tatsächliche Distanzen zwischen den Vertices dieser Gruppe
                        group_verts = verts[idx_list]
                        # Berechne maximale Distanz innerhalb der Gruppe (paarweise)
                        if len(group_verts) > 1:
                            max_dist_in_group = 0.0
                            for i in range(len(group_verts)):
                                for j in range(i + 1, len(group_verts)):
                                    dist = float(np.linalg.norm(group_verts[i] - group_verts[j]))
                                    if dist > max_dist_in_group:
                                        max_dist_in_group = dist
                            if max_dist_in_group > max_distance:
                                max_distance = max_dist_in_group
                            overlap_count += 1
                    
                    if overlap_count > 0:
                        print(
                            "[DEBUG VertexOverlap] Surface "
                            f"'{geom.surface_id}': {overlap_count} 3D-Positionen mit "
                            f"überlappenden Vertices (max Distanz innerhalb Gruppe={max_distance:.3f} m)."
                        )
        except Exception:
            # Reiner Debug-Helper – darf nie die Hauptlogik stören
            pass
        
        return surface_grid
            # 🛠️ VORBEREITUNG: Für "sloped" Flächen plane_model aus Punkten ableiten, BEVOR Grid erstellt wird
            # Dies stellt sicher, dass build_single_surface_grid das korrekte plane_model verwendet
            if geom.orientation == "sloped":
                if geom.points:
                    try:
                        # Extrahiere Z-Werte aus Punkten zur Validierung
                        points_z = np.array([p.get('z', 0.0) for p in geom.points], dtype=float)
                        z_span_points = float(np.ptp(points_z)) if len(points_z) > 0 else 0.0
                        
                        plane_model_new, error_msg = derive_surface_plane(geom.points)
                        if plane_model_new is not None:
                            geom.plane_model = plane_model_new
                            mode = plane_model_new.get('mode', 'unknown')
                            # Nur Warnungen ausgeben, wenn Probleme erkannt werden
                            if mode == "constant":
                                pass
                            elif mode in ("x", "y") and abs(plane_model_new.get('slope', 0.0)) < 1e-6:
                                pass
                        else:
                            pass
                    except Exception as e:
                        pass
                else:
                    pass
            
            try:
                result = self.builder.build_single_surface_grid(
                        geometry=geom,
                        resolution=resolution,
                        min_points_per_dimension=min_points_per_dimension
                    )
                if len(result) == 7:
                    # Neue Version: Mit strikter Maske
                    (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict) = result
                else:
                    # Alte Version: Ohne strikte Maske (Fallback für vertikale)
                    (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = result
                    surface_mask_strict = surface_mask  # Für vertikale identisch
            except (ValueError, Exception) as e:
                # Fange alle Fehler ab (ValueError, QhullError, etc.) und überspringe die Surface
                # Kein Fallback - nur die funktionierenden Surfaces werden verwendet
                error_type = type(e).__name__
                # Immer Warnung ausgeben, wenn Surface übersprungen wird (auch ohne DEBUG_FLEXIBLE_GRID)
                print(f"⚠️  [FlexibleGridGenerator] Surface '{geom.surface_id}' übersprungen ({error_type}): {e}")
                continue
            
            # 🛠️ KORREKTUR: Z-Koordinaten bei planaren/schrägen Flächen aus der Ebene berechnen
            # Falls plane_model fehlt oder nahezu flache Z-Spanne ergibt, versuche Neu-Bestimmung aus den Punkten.
            if geom.orientation in ("planar", "sloped"):
                plane_model_local = geom.plane_model
                # Für "sloped" Flächen: IMMER plane_model aus Punkten ableiten, wenn nicht vorhanden ODER wenn vorhanden aber falsch
                # Für "planar" Flächen: Nur ableiten, wenn nicht vorhanden
                if geom.orientation == "sloped":
                    # Für schräge Flächen: Immer aus Punkten ableiten, um sicherzustellen, dass es korrekt ist
                    if geom.points:
                        plane_model_local, _ = derive_surface_plane(geom.points)
                        if plane_model_local is None:
                            raise ValueError(f"Surface '{geom.surface_id}' (sloped): plane_model konnte nicht abgeleitet werden")
                        geom.plane_model = plane_model_local
                    else:
                        pass
                else:
                    # Für planare Flächen: Nur ableiten, wenn nicht vorhanden
                    if plane_model_local is None and geom.points:
                        plane_model_local, _ = derive_surface_plane(geom.points)
                        if plane_model_local is None:
                            raise ValueError(f"Surface '{geom.surface_id}' (planar): plane_model konnte nicht abgeleitet werden")
                        geom.plane_model = plane_model_local
                
                if plane_model_local:
                    try:
                        z_old_span = float(Z_grid.max()) - float(Z_grid.min())
                        Z_eval = _evaluate_plane_on_grid(plane_model_local, X_grid, Y_grid)
                        if Z_eval is not None and Z_eval.shape == X_grid.shape:
                            Z_grid = Z_eval
                            zmin, zmax = float(Z_grid.min()), float(Z_grid.max())
                            z_span = zmax - zmin
                            # Nur Warnung ausgeben, wenn Problem erkannt wird
                            if geom.orientation == "sloped" and z_span < 0.01:
                                pass
                    except Exception as e:
                        pass
                else:
                    if geom.orientation == "sloped":
                        pass
            
            # Berechne tatsächliche Resolution (kann adaptiv sein)
            ny, nx = X_grid.shape
            if len(sound_field_x) > 1 and len(sound_field_y) > 1:
                actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
                actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
                actual_resolution = (actual_resolution_x + actual_resolution_y) / 2.0
            else:
                actual_resolution = resolution
            
            # 🎯 TRIANGULATION: Erstelle triangulierte Vertices aus Grid-Punkten
            # Pro Grid-Punkt werden Vertices verwendet, pro Grid-Quadrat 2 Dreiecke erstellt
            triangulated_vertices = None
            triangulated_faces = None
            triangulated_success = False
            vertex_source_indices: Optional[np.ndarray] = None
            
            try:
                # Verwende Grid-Punkte innerhalb der Surface-Maske als Vertices
                ny, nx = X_grid.shape
                mask_flat = surface_mask.ravel()  # Erweiterte Maske (für SPL-Werte)
                mask_strict_flat = surface_mask_strict.ravel()  # Strikte Maske (für Face-Filterung)
                
                if np.any(mask_flat):
                    # Erstelle Vertex-Koordinaten aus allen Grid-Punkten (auch inaktive für konsistente Indizes)
                    # Wir brauchen alle Punkte für die strukturierte Triangulation
                    all_vertices = np.column_stack([
                        X_grid.ravel(),  # X-Koordinaten
                        Y_grid.ravel(),  # Y-Koordinaten
                        Z_grid.ravel()   # Z-Koordinaten
                    ])  # Shape: (ny * nx, 3)
                    
                    # 🎯 ZUSÄTZLICHE VERTICES AN POLYGON-ECKEN HINZUFÜGEN (für höhere Auflösung)
                    # Dies erhöht die Polygon-Dichte an den Ecken, um Lücken zu vermeiden
                    # 🎯 IDENTISCHE BEHANDLUNG: Für alle Orientierungen (planar, sloped, vertical)
                    additional_vertices = []
                    surface_points = geom.points or []
                    if len(surface_points) >= 3:
                        # Bestimme Koordinatensystem basierend auf Orientierung
                        if geom.orientation == "vertical":
                            # 🎯 VERTIKALE FLÄCHEN: Verwende (u,v)-Koordinaten basierend auf dominant_axis
                            xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                            ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                            zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                            x_span = float(np.ptp(xs))
                            y_span = float(np.ptp(ys))
                            eps_line = 1e-6
                            
                            # 🎯 ZUERST AUSRICHTUNG PRÜFEN: Bestimme (u,v)-Koordinaten basierend auf dominant_axis oder Spannen-Analyse
                            # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende dominant_axis wenn verfügbar
                            # Berechne Konstanten-Werte im Voraus
                            x_mean = float(np.mean(xs))
                            y_mean = float(np.mean(ys))
                            z_span = float(np.ptp(zs))
                            
                            # 🎯 PRÜFE OB SCHRÄGE WAND: Y variiert bei X-Z-Wänden, X variiert bei Y-Z-Wänden
                            is_slanted_wall = False
                            
                            if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                # Verwende dominant_axis als primäre Quelle (konsistent mit Grid-Erstellung)
                                if geom.dominant_axis == "yz":
                                    # Y-Z-Wand: u = y, v = z
                                    polygon_u = ys
                                    polygon_v = zs
                                    is_xz_wall = False
                                    # Prüfe ob schräg: X variiert (gleiche Logik wie Grid-Erstellung)
                                    is_slanted_wall = (x_span > eps_line and z_span > 1e-3)
                                    if DEBUG_FLEXIBLE_GRID:
                                        print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': Y-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                elif geom.dominant_axis == "xz":
                                    # X-Z-Wand: u = x, v = z
                                    polygon_u = xs
                                    polygon_v = zs
                                    is_xz_wall = True
                                    # Prüfe ob schräg: Y variiert (gleiche Logik wie Grid-Erstellung)
                                    is_slanted_wall = (y_span > eps_line and z_span > 1e-3)
                                    if DEBUG_FLEXIBLE_GRID:
                                        print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': X-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                else:
                                    raise ValueError(f"Surface '{geom.surface_id}': Unbekannter dominant_axis '{geom.dominant_axis}'")
                            
                            # 🎯 FÜR SCHRÄGE WÄNDE: Interpoliere Y (X-Z-Wand) bzw. X (Y-Z-Wand) aus Surface-Punkten
                            if is_slanted_wall:
                                from scipy.interpolate import griddata
                                if is_xz_wall:
                                    # X-Z-Wand schräg: Interpoliere Y aus (x,z)-Koordinaten
                                    points_surface = np.column_stack([xs, zs])
                                else:
                                    # Y-Z-Wand schräg: Interpoliere X aus (y,z)-Koordinaten
                                    points_surface = np.column_stack([ys, zs])
                            
                            # Prüfe für jede Polygon-Ecke in (u,v)-Koordinaten
                            existing_vertex_tolerance = resolution * 0.1  # Toleranz für "bereits vorhanden"
                            
                            for corner_u, corner_v in zip(polygon_u, polygon_v):
                                # Prüfe ob bereits ein Vertex sehr nahe an dieser Ecke existiert
                                # Für vertikale Flächen: Vergleiche in (u,v)-Koordinaten
                                if is_xz_wall:
                                    # X-Z-Wand: u = x, v = z
                                    distances = np.sqrt((all_vertices[:, 0] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                                else:
                                    # Y-Z-Wand: u = y, v = z
                                    distances = np.sqrt((all_vertices[:, 1] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                                
                                min_distance = np.min(distances)
                                
                                if min_distance > existing_vertex_tolerance:
                                    # Kein Vertex nahe genug → füge neuen Vertex hinzu
                                    # Transformiere (u,v) zurück zu (x,y,z)
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        corner_x = corner_u
                                        corner_z = corner_v
                                        if is_slanted_wall:
                                            # Schräge Wand: Y interpoliert aus (x,z)
                                            from scipy.interpolate import griddata
                                            corner_y = griddata(
                                                points_surface, ys,
                                                np.array([[corner_u, corner_v]]),
                                                method='linear', fill_value=y_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(additional_vertices) == 0:
                                                print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': Y interpoliert für Ecke (u={corner_u:.3f}, v={corner_v:.3f}) → Y={corner_y:.3f} (linear)")
                                        else:
                                            # Konstante Wand: Y = konstant
                                            corner_y = y_mean
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        corner_y = corner_u
                                        corner_z = corner_v
                                        if is_slanted_wall:
                                            # Schräge Wand: X interpoliert aus (y,z)
                                            from scipy.interpolate import griddata
                                            corner_x = griddata(
                                                points_surface, xs,
                                                np.array([[corner_u, corner_v]]),
                                                method='linear', fill_value=x_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(additional_vertices) == 0:
                                                print(f"[DEBUG Vertical Triangulation] Surface '{geom.surface_id}': X interpoliert für Ecke (u={corner_u:.3f}, v={corner_v:.3f}) → X={corner_x:.3f} (linear)")
                                        else:
                                            # Konstante Wand: X = konstant
                                            corner_x = x_mean
                                    
                                    additional_vertices.append([corner_x, corner_y, corner_z])
                        else:
                            # PLANARE/SCHRÄGE FLÄCHEN: Verwende (x,y)-Koordinaten
                            polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                            polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                            
                            # Prüfe für jede Polygon-Ecke, ob nahe Grid-Punkte existieren
                            existing_vertex_tolerance = resolution * 0.1  # Toleranz für "bereits vorhanden"
                            
                            for corner_x, corner_y in zip(polygon_x, polygon_y):
                                # Prüfe ob bereits ein Vertex sehr nahe an dieser Ecke existiert
                                distances = np.sqrt((all_vertices[:, 0] - corner_x)**2 + (all_vertices[:, 1] - corner_y)**2)
                                min_distance = np.min(distances)
                                
                                if min_distance > existing_vertex_tolerance:
                                    # Kein Vertex nahe genug → füge neuen Vertex hinzu
                                    # Berechne Z-Koordinate für Ecke
                                    corner_z = 0.0
                                    if geom.plane_model:
                                        try:
                                            z_new = _evaluate_plane_on_grid(
                                                geom.plane_model,
                                                np.array([[corner_x]]),
                                                np.array([[corner_y]])
                                            )
                                            if z_new is not None and z_new.size > 0:
                                                corner_z = float(z_new.flat[0])
                                        except Exception:
                                            pass
                                    
                                    additional_vertices.append([corner_x, corner_y, corner_z])
                        
                        if len(additional_vertices) > 0:
                            additional_vertices_array = np.array(additional_vertices, dtype=float)
                            all_vertices = np.vstack([all_vertices, additional_vertices_array])
                            # print(f"  └─ ✅ {len(additional_vertices)} zusätzliche Vertices an Polygon-Ecken hinzugefügt ({geom.orientation})")
                    
                    # Speichere Offset für zusätzliche Vertices (alle Grid-Punkte kommen zuerst)
                    base_vertex_count = X_grid.size
                    additional_vertex_start_idx = base_vertex_count
                    additional_vertices_array = np.array(additional_vertices, dtype=float) if len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
                    
                    # 🎯 RAND-VERTICES AUF SURFACE-GRENZE PROJIZIEREN
                    # Für Rand-Vertices (in erweiterter Maske, aber nicht in strikter Maske):
                    # Verschiebe sie auf die Surface-Grenze, damit Polygone exakt am Rand verlaufen
                    # 🎯 IDENTISCHE BEHANDLUNG: Für alle Orientierungen (planar, sloped, vertical)
                    # Berechne Rand-Vertices: in erweitert, aber nicht in strikt
                    is_on_boundary = mask_flat & (~mask_strict_flat)
                    boundary_indices = np.where(is_on_boundary)[0]
                    
                    if len(boundary_indices) > 0:
                        # Lade Surface-Polygon für Projektion
                        surface_points = geom.points or []
                        if len(surface_points) >= 3:
                            if geom.orientation == "vertical":
                                # 🎯 VERTIKALE FLÄCHEN: Projektion in (u,v)-Koordinaten
                                xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                                ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                                zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                                x_span = float(np.ptp(xs))
                                y_span = float(np.ptp(ys))
                                eps_line = 1e-6
                                
                                # 🎯 ZUERST AUSRICHTUNG PRÜFEN: Bestimme (u,v)-Koordinaten basierend auf dominant_axis oder Spannen-Analyse
                                # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende dominant_axis wenn verfügbar
                                z_span = float(np.ptp(zs))
                                x_mean = float(np.mean(xs))
                                y_mean = float(np.mean(ys))
                                
                                # 🎯 PRÜFE OB SCHRÄGE WAND: Y variiert bei X-Z-Wänden, X variiert bei Y-Z-Wänden
                                # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende gleiche Logik wie is_slanted_vertical
                                is_slanted_wall = False
                                
                                if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                    # Verwende dominant_axis als primäre Quelle (konsistent mit Grid-Erstellung)
                                    if geom.dominant_axis == "yz":
                                        # Y-Z-Wand: u = y, v = z
                                        polygon_u = ys
                                        polygon_v = zs
                                        is_xz_wall = False
                                        # Prüfe ob schräg: X variiert (gleiche Logik wie Grid-Erstellung)
                                        is_slanted_wall = (x_span > eps_line and z_span > 1e-3)
                                        if DEBUG_FLEXIBLE_GRID:
                                            print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': Y-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                    elif geom.dominant_axis == "xz":
                                        # X-Z-Wand: u = x, v = z
                                        polygon_u = xs
                                        polygon_v = zs
                                        is_xz_wall = True
                                        # Prüfe ob schräg: Y variiert (gleiche Logik wie Grid-Erstellung)
                                        is_slanted_wall = (y_span > eps_line and z_span > 1e-3)
                                        if DEBUG_FLEXIBLE_GRID:
                                            print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': X-Z-Wand, x_span={x_span:.3f}, y_span={y_span:.3f}, z_span={z_span:.3f}, is_slanted_wall={is_slanted_wall}")
                                    else:
                                        raise ValueError(f"Surface '{geom.surface_id}': Unbekannter dominant_axis '{geom.dominant_axis}'")
                                
                                # 🎯 FÜR SCHRÄGE WÄNDE: Interpoliere Y (X-Z-Wand) bzw. X (Y-Z-Wand) aus Surface-Punkten
                                if is_slanted_wall:
                                    from scipy.interpolate import griddata
                                    if is_xz_wall:
                                        # X-Z-Wand schräg: Interpoliere Y aus (x,z)-Koordinaten
                                        points_surface = np.column_stack([xs, zs])
                                    else:
                                        # Y-Z-Wand schräg: Interpoliere X aus (y,z)-Koordinaten
                                        points_surface = np.column_stack([ys, zs])
                                
                                # Projiziere jeden Rand-Vertex auf die nächstliegende Polygon-Kante oder Ecke in (u,v)
                                for idx in boundary_indices:
                                    v = all_vertices[idx]
                                    
                                    # Extrahiere (u,v)-Koordinaten aus Vertex
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        vu, vv = v[0], v[2]
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        vu, vv = v[1], v[2]
                                    
                                    # Finde nächstliegenden Punkt auf Polygon-Rand (Kante oder Ecke) in (u,v)
                                    min_dist_sq = np.inf
                                    closest_u, closest_v = vu, vv
                                    
                                    n_poly = len(polygon_u)
                                    
                                    # 🎯 ZUERST: Prüfe Polygon-Ecken (für bessere Abdeckung an scharfen Ecken)
                                    corner_threshold = 1e-6  # Sehr kleine Schwellenwert für "nahe an Ecke"
                                    for i in range(n_poly):
                                        corner_u, corner_v = polygon_u[i], polygon_v[i]
                                        dist_sq_to_corner = (vu - corner_u)**2 + (vv - corner_v)**2
                                        
                                        # Wenn Vertex sehr nahe an einer Polygon-Ecke ist, projiziere direkt auf Ecke
                                        if dist_sq_to_corner < corner_threshold:
                                            closest_u, closest_v = corner_u, corner_v
                                            min_dist_sq = dist_sq_to_corner
                                            break  # Direkte Ecken-Projektion hat Priorität
                                        
                                        # Wenn noch kein guter Kandidat gefunden, prüfe ob Ecke näher ist als bisher
                                        if dist_sq_to_corner < min_dist_sq:
                                            min_dist_sq = dist_sq_to_corner
                                            closest_u, closest_v = corner_u, corner_v
                                    
                                    # Dann: Prüfe alle Polygon-Kanten (kann bessere Projektion als Ecke liefern)
                                    for i in range(n_poly):
                                        p1 = np.array([polygon_u[i], polygon_v[i]])
                                        p2 = np.array([polygon_u[(i + 1) % n_poly], polygon_v[(i + 1) % n_poly]])
                                        
                                        # Berechne Projektion von v auf Kante (p1, p2) in (u,v)
                                        edge = p2 - p1
                                        edge_len_sq = np.dot(edge, edge)
                                        if edge_len_sq < 1e-12:
                                            continue  # Degenerierte Kante
                                        
                                        t = np.dot([vu - p1[0], vv - p1[1]], edge) / edge_len_sq
                                        t = np.clip(t, 0.0, 1.0)  # Clamp auf Kante
                                        proj = p1 + t * edge
                                        
                                        # Berechne Abstand
                                        dist_sq = (vu - proj[0])**2 + (vv - proj[1])**2
                                        
                                        # Verwende Kanten-Projektion nur wenn sie besser ist als Ecken-Projektion
                                        # (außer wir haben bereits eine sehr nahe Ecken-Projektion gefunden)
                                        if dist_sq < min_dist_sq or (min_dist_sq >= corner_threshold and dist_sq < min_dist_sq * 1.1):
                                            min_dist_sq = dist_sq
                                            closest_u, closest_v = proj[0], proj[1]
                                    
                                    # Transformiere (u,v) zurück zu (x,y,z) und verschiebe Vertex
                                    if is_xz_wall:
                                        # X-Z-Wand: u = x, v = z
                                        all_vertices[idx, 0] = closest_u
                                        all_vertices[idx, 2] = closest_v
                                        if is_slanted_wall:
                                            # Schräge Wand: Y interpoliert aus (x,z)
                                            old_y = all_vertices[idx, 1]
                                            all_vertices[idx, 1] = griddata(
                                                points_surface, ys,
                                                np.array([[closest_u, closest_v]]),
                                                method='linear', fill_value=y_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(boundary_indices) > 0 and idx == boundary_indices[0]:
                                                print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': Y interpoliert für Rand-Vertex (u={closest_u:.3f}, v={closest_v:.3f}) → Y={all_vertices[idx, 1]:.3f} (linear, vorher: {old_y:.3f})")
                                        else:
                                            # Konstante Wand: Y = konstant
                                            all_vertices[idx, 1] = y_mean
                                    else:
                                        # Y-Z-Wand: u = y, v = z
                                        all_vertices[idx, 1] = closest_u
                                        all_vertices[idx, 2] = closest_v
                                        if is_slanted_wall:
                                            # Schräge Wand: X interpoliert aus (y,z)
                                            old_x = all_vertices[idx, 0]
                                            all_vertices[idx, 0] = griddata(
                                                points_surface, xs,
                                                np.array([[closest_u, closest_v]]),
                                                method='linear', fill_value=x_mean
                                            )[0]
                                            if DEBUG_FLEXIBLE_GRID and len(boundary_indices) > 0 and idx == boundary_indices[0]:
                                                print(f"[DEBUG Vertical Boundary] Surface '{geom.surface_id}': X interpoliert für Rand-Vertex (u={closest_u:.3f}, v={closest_v:.3f}) → X={all_vertices[idx, 0]:.3f} (linear, vorher: {old_x:.3f})")
                                        else:
                                            # Konstante Wand: X = konstant
                                            all_vertices[idx, 0] = x_mean
                            else:
                                # PLANARE/SCHRÄGE FLÄCHEN: Projektion in (x,y)-Koordinaten
                                # Extrahiere Polygon-Koordinaten (für planare/schräge: x,y)
                                polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                                polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                                
                                # Projiziere jeden Rand-Vertex auf die nächstliegende Polygon-Kante oder Ecke
                                for idx in boundary_indices:
                                    v = all_vertices[idx]
                                    vx, vy = v[0], v[1]
                                    
                                    # Finde nächstliegenden Punkt auf Polygon-Rand (Kante oder Ecke)
                                    min_dist_sq = np.inf
                                    closest_x, closest_y = vx, vy
                                    
                                    n_poly = len(polygon_x)
                                    
                                    # 🎯 ZUERST: Prüfe Polygon-Ecken (für bessere Abdeckung an scharfen Ecken)
                                    corner_threshold = 1e-6  # Sehr kleine Schwellenwert für "nahe an Ecke"
                                    for i in range(n_poly):
                                        corner_x, corner_y = polygon_x[i], polygon_y[i]
                                        dist_sq_to_corner = (vx - corner_x)**2 + (vy - corner_y)**2
                                        
                                        # Wenn Vertex sehr nahe an einer Polygon-Ecke ist, projiziere direkt auf Ecke
                                        if dist_sq_to_corner < corner_threshold:
                                            closest_x, closest_y = corner_x, corner_y
                                            min_dist_sq = dist_sq_to_corner
                                            break  # Direkte Ecken-Projektion hat Priorität
                                        
                                        # Wenn noch kein guter Kandidat gefunden, prüfe ob Ecke näher ist als bisher
                                        if dist_sq_to_corner < min_dist_sq:
                                            min_dist_sq = dist_sq_to_corner
                                            closest_x, closest_y = corner_x, corner_y
                                    
                                    # Dann: Prüfe alle Polygon-Kanten (kann bessere Projektion als Ecke liefern)
                                    for i in range(n_poly):
                                        p1 = np.array([polygon_x[i], polygon_y[i]])
                                        p2 = np.array([polygon_x[(i + 1) % n_poly], polygon_y[(i + 1) % n_poly]])
                                        
                                        # Berechne Projektion von v auf Kante (p1, p2)
                                        edge = p2 - p1
                                        edge_len_sq = np.dot(edge, edge)
                                        if edge_len_sq < 1e-12:
                                            continue  # Degenerierte Kante
                                        
                                        t = np.dot([vx - p1[0], vy - p1[1]], edge) / edge_len_sq
                                        t = np.clip(t, 0.0, 1.0)  # Clamp auf Kante
                                        proj = p1 + t * edge
                                        
                                        # Berechne Abstand
                                        dist_sq = (vx - proj[0])**2 + (vy - proj[1])**2
                                        
                                        # Verwende Kanten-Projektion nur wenn sie besser ist als Ecken-Projektion
                                        # (außer wir haben bereits eine sehr nahe Ecken-Projektion gefunden)
                                        if dist_sq < min_dist_sq or (min_dist_sq >= corner_threshold and dist_sq < min_dist_sq * 1.1):
                                            min_dist_sq = dist_sq
                                            closest_x, closest_y = proj[0], proj[1]
                                    
                                    # Verschiebe Vertex auf nächstliegenden Punkt auf Surface-Grenze
                                    all_vertices[idx, 0] = closest_x
                                    all_vertices[idx, 1] = closest_y
                                    # Berechne Z-Koordinate neu basierend auf Plane-Model
                                    if geom.plane_model:
                                        try:
                                            # Verwende _evaluate_plane_on_grid für konsistente Z-Berechnung
                                            z_new = _evaluate_plane_on_grid(
                                                geom.plane_model,
                                                np.array([[closest_x]]),
                                                np.array([[closest_y]])
                                            )
                                            if z_new is not None and z_new.size > 0:
                                                all_vertices[idx, 2] = float(z_new.flat[0])
                                        except Exception:
                                            pass  # Falls Berechnung fehlschlägt, behalte alte Z-Koordinate
                            
                            # if len(boundary_indices) > 0:
                            #     print(f"  └─ ✅ {len(boundary_indices)} Rand-Vertices auf Surface-Grenze projiziert ({geom.orientation})")
                    
                    # Erstelle Index-Mapping: (i, j) → linearer Index
                    # Für ein strukturiertes Grid: index = i * nx + j
                    
                    # Erstelle Faces: Pro Grid-Quadrat bis zu 2 Dreiecke
                    # Ein Quadrat hat Ecken: (i,j), (i,j+1), (i+1,j), (i+1,j+1)
                    # Dreieck 1: (i,j) → (i,j+1) → (i+1,j)
                    # Dreieck 2: (i,j+1) → (i+1,j+1) → (i+1,j)
                    faces_list = []
                    active_quads = 0
                    partial_quads = 0  # Quadrate mit 3 aktiven Ecken
                    filtered_out = 0  # Dreiecke außerhalb der strikten Maske
                    
                    for i in range(ny - 1):
                        for j in range(nx - 1):
                            idx_tl = i * nx + j           # Top-Left
                            idx_tr = i * nx + (j + 1)     # Top-Right
                            idx_bl = (i + 1) * nx + j     # Bottom-Left
                            idx_br = (i + 1) * nx + (j + 1)  # Bottom-Right
                            
                            # Prüfe mit erweiterter Maske (für SPL-Werte)
                            tl_active = mask_flat[idx_tl]
                            tr_active = mask_flat[idx_tr]
                            bl_active = mask_flat[idx_bl]
                            br_active = mask_flat[idx_br]
                            
                            # Prüfe mit strikter Maske (für Face-Filterung)
                            tl_in_strict = mask_strict_flat[idx_tl]
                            tr_in_strict = mask_strict_flat[idx_tr]
                            bl_in_strict = mask_strict_flat[idx_bl]
                            br_in_strict = mask_strict_flat[idx_br]
                            
                            active_count = sum([tl_active, tr_active, bl_active, br_active])
                            strict_count = sum([tl_in_strict, tr_in_strict, bl_in_strict, br_in_strict])
                            
                            # Nur Quadrate verarbeiten, wenn mindestens eine Ecke aktiv ist (erweiterte Maske)
                            # Nach Projektion der Rand-Vertices können wir alle aktiven Dreiecke erstellen
                            if active_count == 0:
                                # Alle Ecken inaktiv → überspringe komplett
                                continue
                            
                            # Hilfsfunktion: Prüft ob ein Dreieck erstellt werden soll
                            # Nach Projektion der Rand-Vertices auf die Surface-Grenze können wir
                            # alle Dreiecke erstellen, die mindestens einen aktiven Vertex haben
                            # (entweder in strikter Maske oder projizierter Rand-Vertex)
                            def should_create_triangle(v1_in_strict, v2_in_strict, v3_in_strict, v1_active, v2_active, v3_active):
                                strict_vertices = sum([v1_in_strict, v2_in_strict, v3_in_strict])
                                active_vertices = sum([v1_active, v2_active, v3_active])
                                # Erstelle Dreieck wenn mindestens 1 aktiver Vertex vorhanden ist
                                # (nach Projektion liegen Rand-Vertices auf der Surface-Grenze)
                                return active_vertices >= 1
                            
                            # Fall 1: Alle 4 Ecken aktiv → 2 Dreiecke
                            if active_count == 4:
                                # Dreieck 1: Top-Left → Top-Right → Bottom-Left
                                if should_create_triangle(tl_in_strict, tr_in_strict, bl_in_strict, tl_active, tr_active, bl_active):
                                    faces_list.extend([3, idx_tl, idx_tr, idx_bl])
                                else:
                                    filtered_out += 1
                                # Dreieck 2: Top-Right → Bottom-Right → Bottom-Left
                                if should_create_triangle(tr_in_strict, br_in_strict, bl_in_strict, tr_active, br_active, bl_active):
                                    faces_list.extend([3, idx_tr, idx_br, idx_bl])
                                else:
                                    filtered_out += 1
                                active_quads += 1
                            
                            # Fall 2: Genau 3 Ecken aktiv → 1 Dreieck (bis zum Rand)
                            elif active_count == 3:
                                # Bestimme welche Ecke fehlt und erstelle 1 Dreieck mit den 3 aktiven Ecken
                                # Diese Dreiecke füllen die Fläche bis zum Rand
                                if not tl_active:
                                    # Fehlende Ecke: Top-Left → Dreieck mit tr, bl, br
                                    if should_create_triangle(tr_in_strict, bl_in_strict, br_in_strict, tr_active, bl_active, br_active):
                                        faces_list.extend([3, idx_tr, idx_br, idx_bl])
                                    else:
                                        filtered_out += 1
                                elif not tr_active:
                                    # Fehlende Ecke: Top-Right → Dreieck mit tl, bl, br
                                    if should_create_triangle(tl_in_strict, bl_in_strict, br_in_strict, tl_active, bl_active, br_active):
                                        faces_list.extend([3, idx_tl, idx_bl, idx_br])
                                    else:
                                        filtered_out += 1
                                elif not bl_active:
                                    # Fehlende Ecke: Bottom-Left → Dreieck mit tl, tr, br
                                    if should_create_triangle(tl_in_strict, tr_in_strict, br_in_strict, tl_active, tr_active, br_active):
                                        faces_list.extend([3, idx_tl, idx_tr, idx_br])
                                    else:
                                        filtered_out += 1
                                else:  # not br_active
                                    # Fehlende Ecke: Bottom-Right → Dreieck mit tl, tr, bl
                                    if should_create_triangle(tl_in_strict, tr_in_strict, bl_in_strict, tl_active, tr_active, bl_active):
                                        faces_list.extend([3, idx_tl, idx_tr, idx_bl])
                                    else:
                                        filtered_out += 1
                                partial_quads += 1
                            
                            # 🎯 ZUSÄTZLICHE DREIECKE MIT POLYGON-ECKEN-VERTICES
                            # Erstelle Dreiecke, die zusätzliche Ecken-Vertices verwenden (für höhere Auflösung)
                            if len(additional_vertices) > 0 and active_count >= 1:
                                # Prüfe für jeden zusätzlichen Ecken-Vertex, ob er nahe genug an diesem Quadrat ist
                                for corner_idx, corner_vertex in enumerate(additional_vertices_array):
                                    corner_vertex_idx = additional_vertex_start_idx + corner_idx
                                    corner_x, corner_y = corner_vertex[0], corner_vertex[1]
                                    
                                    # Prüfe ob Ecken-Vertex innerhalb oder nahe diesem Quadrat ist
                                    x_min, x_max = X_grid[0, j], X_grid[0, j+1] if j+1 < nx else X_grid[0, j]
                                    y_min, y_max = Y_grid[i, 0], Y_grid[i+1, 0] if i+1 < ny else Y_grid[i, 0]
                                    
                                    # Erweitere Bereich um eine Resolution (für nahe Vertices)
                                    tolerance = resolution * 1.5
                                    if (x_min - tolerance <= corner_x <= x_max + tolerance and
                                        y_min - tolerance <= corner_y <= y_max + tolerance):
                                        
                                        # Erstelle Dreiecke mit Ecken-Vertex + 2 aktive Grid-Ecken
                                        active_corners = []
                                        if tl_active:
                                            active_corners.append((idx_tl, tl_in_strict))
                                        if tr_active:
                                            active_corners.append((idx_tr, tr_in_strict))
                                        if bl_active:
                                            active_corners.append((idx_bl, bl_in_strict))
                                        if br_active:
                                            active_corners.append((idx_br, br_in_strict))
                                        
                                        # Erstelle Dreiecke: Ecken-Vertex + 2 benachbarte aktive Ecken
                                        if len(active_corners) >= 2:
                                            for k in range(len(active_corners) - 1):
                                                idx1, strict1 = active_corners[k]
                                                idx2, strict2 = active_corners[k + 1]
                                                # Ecken-Vertex ist immer aktiv (wird als aktiv betrachtet)
                                                if should_create_triangle(True, strict1, strict2, True, True, True):
                                                    faces_list.extend([3, corner_vertex_idx, idx1, idx2])
                            
                            # Fall 3: 2 Ecken aktiv → versuche Dreieck zu erstellen, wenn Ecken benachbart sind
                            # Dies hilft, Ecken-Lücken zu füllen
                            elif active_count == 2:
                                # Finde die 2 aktiven Ecken
                                active_corners = []
                                if tl_active:
                                    active_corners.append(('tl', idx_tl, tl_in_strict, tl_active))
                                if tr_active:
                                    active_corners.append(('tr', idx_tr, tr_in_strict, tr_active))
                                if bl_active:
                                    active_corners.append(('bl', idx_bl, bl_in_strict, bl_active))
                                if br_active:
                                    active_corners.append(('br', idx_br, br_in_strict, br_active))
                                
                                if len(active_corners) == 2:
                                    corner1_name, corner1_idx, corner1_strict, corner1_active = active_corners[0]
                                    corner2_name, corner2_idx, corner2_strict, corner2_active = active_corners[1]
                                    
                                    # Prüfe ob Ecken benachbart sind (können ein Dreieck mit Rand bilden)
                                    # Benachbart: (tl,tr), (tl,bl), (tr,br), (bl,br)
                                    adjacent_pairs = {
                                        ('tl', 'tr'): [('bl', idx_bl), ('br', idx_br)],
                                        ('tl', 'bl'): [('tr', idx_tr), ('br', idx_br)],
                                        ('tr', 'br'): [('tl', idx_tl), ('bl', idx_bl)],
                                        ('bl', 'br'): [('tl', idx_tl), ('tr', idx_tr)]
                                    }
                                    
                                    # Normalisiere Reihenfolge
                                    pair_key = tuple(sorted([corner1_name, corner2_name]))
                                    
                                    # Prüfe alle Kombinationen
                                    if pair_key in adjacent_pairs:
                                        # Versuche beide möglichen dritten Ecken
                                        for diag_name, diag_idx in adjacent_pairs[pair_key]:
                                            diag_active = mask_flat[diag_idx]
                                            diag_strict = mask_strict_flat[diag_idx]
                                            
                                            # Erstelle Dreieck wenn dritte Ecke auch aktiv ist (projiziert)
                                            if diag_active:
                                                if should_create_triangle(corner1_strict, corner2_strict, diag_strict, corner1_active, corner2_active, diag_active):
                                                    faces_list.extend([3, corner1_idx, corner2_idx, diag_idx])
                                                    partial_quads += 1
                                                    break  # Ein Dreieck reicht für benachbarte Ecken
                    
                    if len(faces_list) > 0:
                        triangulated_vertices = all_vertices  # Alle Grid-Punkte als Vertices
                        triangulated_faces = np.array(faces_list, dtype=np.int64)
                        triangulated_success = True
                        
                        n_faces = len(faces_list) // 4  # 4 Elemente pro Face: [n, v1, v2, v3]
                        n_vertices_used = len(np.unique(triangulated_faces[1::4]))  # Eindeutige Vertex-Indizes
                        expected_faces = active_quads * 2 + partial_quads  # Volle Quadrate: 2 Dreiecke, Rand-Quadrate: 1 Dreieck
                        
                    else:
                        pass
                else:
                    pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                triangulated_success = False
            
            surface_grid = SurfaceGrid(
                surface_id=geom.surface_id,
                sound_field_x=sound_field_x,
                sound_field_y=sound_field_y,
                X_grid=X_grid,
                Y_grid=Y_grid,
                Z_grid=Z_grid,
                surface_mask=surface_mask,
                resolution=actual_resolution,
                geometry=geom,
                triangulated_vertices=triangulated_vertices,
                triangulated_faces=triangulated_faces,
                triangulated_success=triangulated_success,
            )
            
            # 🎯 DEBUG: Prüfen, ob mehrere Vertices sich in 3D überschneiden
            # (d.h. Vertices liegen sehr nahe beieinander in allen drei Dimensionen X, Y, Z).
            # Dies erkennt auch Überlagerungen bei senkrechten Flächen.
            try:
                if triangulated_success and triangulated_vertices is not None:
                    verts = np.asarray(triangulated_vertices, dtype=float)
                    if verts.shape[0] > 0:
                        # Toleranz: etwa 1% der typischen Grid-Resolution für alle drei Dimensionen
                        # oder ein fester kleiner Wert als Fallback.
                        if actual_resolution and actual_resolution > 0:
                            tol = float(actual_resolution) * 0.01
                        else:
                            tol = 1e-3
                        
                        # Quantisiere X, Y, Z Koordinaten (reduziert numerisches Rauschen)
                        xyz_quant = np.round(verts / tol) * tol
                        
                        # Baue Mapping: quantisierte (X,Y,Z) -> Liste von Vertex-Indizes
                        from collections import defaultdict
                        groups: defaultdict[tuple, list] = defaultdict(list)
                        for idx_v, quant_xyz in enumerate(xyz_quant):
                            groups[(float(quant_xyz[0]), float(quant_xyz[1]), float(quant_xyz[2]))].append(idx_v)
                        
                        # Prüfe, ob mehrere Vertices an derselben quantisierten 3D-Position liegen
                        overlap_count = 0
                        max_distance = 0.0
                        for quant_xyz, idx_list in groups.items():
                            if len(idx_list) <= 1:
                                continue
                            # Berechne tatsächliche Distanzen zwischen den Vertices dieser Gruppe
                            group_verts = verts[idx_list]
                            # Berechne maximale Distanz innerhalb der Gruppe (paarweise)
                            if len(group_verts) > 1:
                                max_dist_in_group = 0.0
                                for i in range(len(group_verts)):
                                    for j in range(i + 1, len(group_verts)):
                                        dist = float(np.linalg.norm(group_verts[i] - group_verts[j]))
                                        if dist > max_dist_in_group:
                                            max_dist_in_group = dist
                                if max_dist_in_group > max_distance:
                                    max_distance = max_dist_in_group
                                overlap_count += 1
                        
                        if overlap_count > 0:
                            print(
                                "[DEBUG VertexOverlap] Surface "
                                f"'{geom.surface_id}': {overlap_count} 3D-Positionen mit "
                                f"überlappenden Vertices (max Distanz innerhalb Gruppe={max_distance:.3f} m)."
                            )
            except Exception:
                # Reiner Debug-Helper – darf nie die Hauptlogik stören
                pass
            
            # Im Cache speichern
            self._surface_grid_cache[cache_key] = surface_grid
            surface_grids[geom.surface_id] = surface_grid
            
            points_in_surface = int(np.sum(surface_mask))
            total_points = int(X_grid.size)
            
            # 🎯 QUALITÄTS-CHECK: Erkennen, ob das Polygon nur teilweise vom Grid erfasst wird
            # Idee:
            # - Fläche der Bounding-Box ~ erwartete Anzahl Grid-Punkte
            # - Vergleich mit tatsächlich aktiven Punkten in der Maske
            # - Wenn Verhältnis sehr klein ist, könnte ein Teil des Polygons (z.B. Spitzen) fehlen
            try:
                coverage_info = {}
                if geom.bbox is not None and actual_resolution is not None and actual_resolution > 0:
                    min_x, max_x, min_y, max_y = geom.bbox
                    width = float(max_x - min_x)
                    height = float(max_y - min_y)
                    bbox_area = max(width * height, 0.0)
                    
                    if bbox_area > 0.0:
                        # Erwartete Punktzahl im Sinne einer groben Obergrenze
                        approx_expected_points = bbox_area / float(actual_resolution ** 2)
                        approx_expected_points = max(approx_expected_points, 1.0)
                        coverage_ratio = points_in_surface / approx_expected_points
                        coverage_info = {
                            "bbox_area": bbox_area,
                            "approx_expected_points": approx_expected_points,
                            "coverage_ratio": coverage_ratio,
                        }
                        
                        # Heuristische Schwelle:
                        # - Mindestens 1 aktiver Punkt vorhanden (also nicht komplett leer)
                        # - Aber deutlich weniger als ~50 % der "erwarteten" Belegung
                        if points_in_surface > 0 and coverage_ratio < 0.5:
                            print(
                                "⚠️  [FlexibleGridGenerator] Surface "
                                f"'{geom.surface_id}' ({geom.orientation}): "
                                f"nur {points_in_surface} aktive Grid-Punkte, "
                                f"erwartet wären grob ≈ {approx_expected_points:.0f} "
                                f"({coverage_ratio*100:.1f} % Abdeckung der Bounding-Box). "
                                "Teile des Polygons könnten im Plot fehlen "
                                "(z.B. sehr schmale Spitzen oder Teilflächen)."
                            )
            except Exception:
                # Qualitäts-Check ist rein diagnostisch – Fehler hier dürfen niemals die Berechnung stoppen
                pass
            
            # 🎯 DEBUG: Zusätzliche Info für vertikale Flächen
            is_vertical = geom.orientation == "vertical"
            if is_vertical:
                xs = X_grid.flatten()
                ys = Y_grid.flatten()
                zs = Z_grid.flatten()
                x_span = float(np.ptp(xs))
                y_span = float(np.ptp(ys))
                z_span = float(np.ptp(zs))
                if DEBUG_FLEXIBLE_GRID:
                    print(f"[DEBUG Grid pro Surface] '{geom.surface_id}' (VERTIKAL): {points_in_surface}/{total_points} Punkte, Spannen: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
            else:
                if DEBUG_FLEXIBLE_GRID:
                    print(f"[DEBUG Grid pro Surface] '{geom.surface_id}': {points_in_surface}/{total_points} Punkte, Resolution: {actual_resolution:.3f} m")
        
        return surface_grids
    
    def generate_single_surface_grid(
        self,
        surface_id: str,
        surface_def: Dict,
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6,
    ) -> Optional[SurfaceGrid]:
        """
        Erstellt ein SurfaceGrid für genau eine Surface.
        """
        if resolution is None:
            resolution = self.settings.resolution

        geometries = self.analyzer.analyze_surfaces([(surface_id, surface_def)])
        if not geometries:
            return None

        geom = geometries[0]
        (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = \
            self.builder.build_single_surface_grid(
                geometry=geom,
                resolution=resolution,
                min_points_per_dimension=min_points_per_dimension
            )

        # 🛠️ KORREKTUR: Z-Koordinaten bei planaren/schrägen Flächen aus der Ebene berechnen
        if geom.orientation in ("planar", "sloped") and geom.plane_model:
            try:
                Z_eval = _evaluate_plane_on_grid(geom.plane_model, X_grid, Y_grid)
                if Z_eval is not None and Z_eval.shape == X_grid.shape:
                    Z_grid = Z_eval
            except Exception as e:
                pass

        ny, nx = X_grid.shape
        if len(sound_field_x) > 1 and len(sound_field_y) > 1:
            actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
            actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
            actual_resolution = (actual_resolution_x + actual_resolution_y) / 2.0
        else:
            actual_resolution = resolution

        # 🎯 TRIANGULATION: Erstelle triangulierte Vertices aus Grid-Punkten
        # Pro Grid-Punkt werden Vertices verwendet, pro Grid-Quadrat 2 Dreiecke erstellt
        triangulated_vertices = None
        triangulated_faces = None
        triangulated_success = False
        
        try:
            mask_flat = surface_mask.ravel()
            
            if np.any(mask_flat):
                # Erstelle Vertex-Koordinaten aus allen Grid-Punkten
                all_vertices = np.column_stack([
                    X_grid.ravel(),
                    Y_grid.ravel(),
                    Z_grid.ravel()
                ])
                
                # Erstelle Faces: Pro aktives Grid-Quadrat 2 Dreiecke
                faces_list = []
                active_quads = 0
                
                for i in range(ny - 1):
                    for j in range(nx - 1):
                        idx_tl = i * nx + j
                        idx_tr = i * nx + (j + 1)
                        idx_bl = (i + 1) * nx + j
                        idx_br = (i + 1) * nx + (j + 1)
                        
                        if (mask_flat[idx_tl] and mask_flat[idx_tr] and 
                            mask_flat[idx_bl] and mask_flat[idx_br]):
                            
                            faces_list.extend([3, idx_tl, idx_tr, idx_bl])
                            faces_list.extend([3, idx_tr, idx_br, idx_bl])
                            active_quads += 1
                
                if len(faces_list) > 0:
                    triangulated_vertices = all_vertices
                    triangulated_faces = np.array(faces_list, dtype=np.int64)
                    triangulated_success = True
                    
                    if DEBUG_FLEXIBLE_GRID:
                        n_faces = len(faces_list) // 4
                        print(f"[DEBUG Triangulation] Surface '{geom.surface_id}': {n_faces} Dreiecke, {len(all_vertices)} Vertices, {active_quads} aktive Quadrate")
        except Exception as e:
            if DEBUG_FLEXIBLE_GRID:
                print(f"[DEBUG Triangulation] Surface '{geom.surface_id}': Fehler bei Triangulation: {e}")
            triangulated_success = False

        return SurfaceGrid(
            surface_id=geom.surface_id,
            sound_field_x=sound_field_x,
            sound_field_y=sound_field_y,
            X_grid=X_grid,
            Y_grid=Y_grid,
            Z_grid=Z_grid,
            surface_mask=surface_mask,
            resolution=actual_resolution,
            geometry=geom,
            triangulated_vertices=triangulated_vertices,
            triangulated_faces=triangulated_faces,
            triangulated_success=triangulated_success,
        )

    def generate_per_group(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6,
    ) -> Dict[str, SurfaceGrid]:
        """
        Erzeugt EIN Grid pro aktiver Gruppe (oder ungruppierter Surface).
        Alle Surfaces der Gruppe werden in einem gemeinsamen Grid zusammengeführt.
        """
        if resolution is None:
            resolution = self.settings.resolution

        if not enabled_surfaces:
            return {}

        # Gruppiere Surfaces nach group_id (fallback "__ungrouped__")
        grouped: Dict[str, List[Tuple[str, Dict]]] = {}
        for sid, sdef in enabled_surfaces:
            gid = sdef.get("group_id") or sdef.get("group_name") or "__ungrouped__"
            grouped.setdefault(gid, []).append((sid, sdef))

        result: Dict[str, SurfaceGrid] = {}

        for gid, surfaces in grouped.items():
            # Analysiere Surfaces der Gruppe
            geometries = self.analyzer.analyze_surfaces(surfaces)
            if not geometries:
                continue

            # Bounding-Box über alle Surfaces
            all_x = []
            all_y = []
            for geom in geometries:
                pts = geom.points
                all_x.extend([p.get("x", 0.0) for p in pts])
                all_y.extend([p.get("y", 0.0) for p in pts])
            if not all_x or not all_y:
                continue

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Stelle sicher, dass wir mindestens min_points_per_dimension abdecken
            if resolution <= 0:
                resolution = self.settings.resolution or 1.0
            nx_target = max(min_points_per_dimension, int(np.ceil((max_x - min_x) / resolution)) + 1)
            ny_target = max(min_points_per_dimension, int(np.ceil((max_y - min_y) / resolution)) + 1)
            sound_field_x = np.linspace(min_x, max_x, nx_target)
            sound_field_y = np.linspace(min_y, max_y, ny_target)
            X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing="xy")

            # Union-Maske und Z-Akkumulation
            mask_union = np.zeros_like(X_grid, dtype=bool)
            Z_accum = np.zeros_like(X_grid, dtype=float)
            Z_count = np.zeros_like(X_grid, dtype=float)

            for geom in geometries:
                pts = geom.points
                if len(pts) < 3:
                    continue
                poly_x = np.array([p.get("x", 0.0) for p in pts], dtype=float)
                poly_y = np.array([p.get("y", 0.0) for p in pts], dtype=float)
                try:
                    from matplotlib.path import Path
                    path = Path(np.column_stack((poly_x, poly_y)))
                    inside = path.contains_points(np.column_stack((X_grid.ravel(), Y_grid.ravel())))
                    mask = inside.reshape(X_grid.shape)
                except Exception:
                    mask = np.zeros_like(X_grid, dtype=bool)
                if not np.any(mask):
                    continue

                # Z-Werte auf Ebene projizieren
                if geom.plane_model:
                    Z_surface = _evaluate_plane_on_grid(geom.plane_model, X_grid, Y_grid)
                    Z_accum[mask] += Z_surface[mask]
                    Z_count[mask] += 1.0
                else:
                    # keine Ebene: Z=0
                    Z_count[mask] += 1.0

                mask_union |= mask

            if not np.any(mask_union):
                continue

            Z_grid = np.zeros_like(X_grid, dtype=float)
            valid = Z_count > 0
            Z_grid[valid] = Z_accum[valid] / Z_count[valid]

            # SurfaceGeometry Platzhalter für Gruppe
            geom_group = SurfaceGeometry(
                surface_id=str(gid),
                name=str(gid),
                points=[],  # nicht benötigt im Plot
                plane_model=None,
                orientation="planar",
                bbox=(min_x, max_x, min_y, max_y),
            )

            # Effektive Auflösung
            if len(sound_field_x) > 1 and len(sound_field_y) > 1:
                actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
                actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
                actual_resolution = (actual_resolution_x + actual_resolution_y) / 2.0
            else:
                actual_resolution = resolution

            # 🎯 TRIANGULATION: Für Gruppen kombinieren wir alle Surfaces
            triangulated_vertices = None
            triangulated_faces = None
            triangulated_success = False
            
            try:
                # Kombiniere alle Surface-Punkte für Triangulation
                all_points = []
                for geom in geometries:
                    all_points.extend(geom.points)
                
                # 🎯 DEBUG: Zeige Eingabe-Punkte für Gruppe
                print(f"[DEBUG Triangulation] Gruppe '{gid}': Starte Triangulation")
                print(f"  └─ Anzahl Surfaces in Gruppe: {len(geometries)}")
                print(f"  └─ Kombinierte Punkte: {len(all_points)}")
                for i, geom in enumerate(geometries):
                    print(f"      └─ Surface {i+1} '{geom.surface_id}': {len(geom.points)} Punkte")
                
                if len(all_points) >= 3:
                    pts_array = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in all_points])
                    print(f"  └─ Kombinierte X-Range: [{pts_array[:, 0].min():.3f}, {pts_array[:, 0].max():.3f}]")
                    print(f"  └─ Kombinierte Y-Range: [{pts_array[:, 1].min():.3f}, {pts_array[:, 1].max():.3f}]")
                    print(f"  └─ Kombinierte Z-Range: [{pts_array[:, 2].min():.3f}, {pts_array[:, 2].max():.3f}]")
                    
                    tris = triangulate_points(all_points)
                    if tris and len(tris) > 0:
                        # Konvertiere Triangulation zu Vertices und Faces
                        verts_list = []
                        for tri in tris:
                            for p in tri:
                                verts_list.append([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])
                        
                        verts = np.array(verts_list, dtype=float)
                        faces_list = []
                        vertex_offset = 0
                        for _ in tris:
                            faces_list.extend([3, vertex_offset, vertex_offset + 1, vertex_offset + 2])
                            vertex_offset += 3
                        
                        triangulated_vertices = verts
                        triangulated_faces = np.array(faces_list, dtype=np.int64)
                        triangulated_success = True
                        
                        # 🎯 DEBUG: Detaillierte Ausgabe der Triangulation für Gruppe
                        print(f"[DEBUG Triangulation] ✅ Gruppe '{gid}': Triangulation erfolgreich")
                        print(f"  └─ Anzahl Dreiecke: {len(tris)}")
                        print(f"  └─ Anzahl Vertices: {len(verts)} (erwartet: {len(tris) * 3})")
                        print(f"  └─ Anzahl Faces: {len(faces_list)//4} (Format: [n, v1, v2, v3, ...])")
                        print(f"  └─ Vertices Shape: {verts.shape}")
                        print(f"  └─ Faces Shape: {triangulated_faces.shape}")
                        print(f"  └─ Vertices X-Range: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
                        print(f"  └─ Vertices Y-Range: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
                        print(f"  └─ Vertices Z-Range: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
                        
                        # Prüfe auf NaN oder Inf
                        if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                            print(f"  ⚠️  WARNUNG: NaN oder Inf in Vertices gefunden!")
                        else:
                            print(f"  └─ ✅ Alle Vertices sind gültig (keine NaN/Inf)")
                    else:
                        print(f"[DEBUG Triangulation] ❌ Gruppe '{gid}': Keine Dreiecke erstellt (tris={tris})")
                else:
                    print(f"[DEBUG Triangulation] ❌ Gruppe '{gid}': Zu wenige Punkte ({len(all_points)} < 3)")
            except Exception as e:
                print(f"[DEBUG Triangulation] ❌ Gruppe '{gid}': Fehler bei Triangulation: {e}")
                import traceback
                print(f"  └─ Traceback: {traceback.format_exc()}")
                triangulated_success = False

            result[gid] = SurfaceGrid(
                surface_id=str(gid),
                sound_field_x=sound_field_x,
                sound_field_y=sound_field_y,
                X_grid=X_grid,
                Y_grid=Y_grid,
                Z_grid=Z_grid,
                surface_mask=mask_union,
                resolution=actual_resolution,
                geometry=geom_group,
                triangulated_vertices=triangulated_vertices,
                triangulated_faces=triangulated_faces,
                triangulated_success=triangulated_success,
            )

        return result
    
    def _make_surface_cache_key(
        self,
        geom: SurfaceGeometry,
        resolution: float,
        min_points: int,
    ) -> tuple:
        """
        Erzeugt einen stabilen Cache-Key für eine Surface-Geometrie.
        Berücksichtigt:
        - surface_id
        - Orientierung
        - verwendete Resolution
        - min_points_per_dimension
        - diskretisierte Punktkoordinaten
        - globale Geometrie-Version (Settings.geometry_version)
        """
        points = geom.points or []
        # Runde Koordinaten leicht, um numerisches Rauschen zu unterdrücken
        pts_key = tuple(
            (
                round(float(p.get("x", 0.0)), 4),
                round(float(p.get("y", 0.0)), 4),
                round(float(p.get("z", 0.0)), 4),
            )
            for p in points
        )
        geometry_version = int(getattr(self.settings, "geometry_version", 0))
        return (
            str(geom.surface_id),
            str(getattr(geom, "orientation", "")),
            round(float(resolution), 4),
            int(min_points),
            pts_key,
            geometry_version,
        )
    
    def _create_cache_hash(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float]
    ) -> str:
        """Erstellt Hash für Cache-Vergleich"""
        # Hash: Anzahl Surfaces + Resolution + globale Geometrie-Version
        n_surfaces = len(enabled_surfaces)
        res_str = str(resolution) if resolution else "default"
        geometry_version = int(getattr(self.settings, "geometry_version", 0))
        return f"{n_surfaces}_{res_str}_v{geometry_version}"

