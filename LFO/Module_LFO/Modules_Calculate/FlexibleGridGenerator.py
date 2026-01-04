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
from Module_LFO.Modules_Init.Logging import PERF_ENABLED, measure_time
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    evaluate_surface_plane,
    SurfaceDefinition,
    _evaluate_plane_on_grid,
)
from Module_LFO.Modules_Data.SurfaceValidator import triangulate_points

# Prüfe ob scipy verfügbar ist
try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Debug-Modus für FlexibleGrid:
# Standard jetzt: AKTIV (\"1\"), kann über Umgebungsvariable überschrieben werden.
# Beispiel zum Deaktivieren:
#   export LFO_DEBUG_FLEXIBLE_GRID=0
DEBUG_FLEXIBLE_GRID = bool(int(os.environ.get("LFO_DEBUG_FLEXIBLE_GRID", "1")))


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
    # 🎯 NEU: Rand- und Eckpunkte (nicht Teil des Grids)
    additional_vertices: Optional[np.ndarray] = None  # Shape: (N, 3) - Rand- und Eckpunkte-Koordinaten


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
        normal: Optional[np.ndarray]  # Wird nicht mehr verwendet (SVD-Normale wird intern berechnet)
        ) -> Tuple[str, Optional[str]]:
        """
        🎯 ULTRA-ROBUSTE Orientierungserkennung ohne Einschränkungen.
        
        Kombiniert mehrere Methoden:
        1. SVD-basiertes Plane Fitting (robusteste Methode, 50% Gewicht)
        2. Normale-Vektor-Analyse (verwendet SVD-Normale, 25% Gewicht)
        3. PCA-Analyse (15% Gewicht)
        4. Plane-Model-Analyse (8% Gewicht)
        5. Spannen-Analyse (Fallback, 2% Gewicht)
        
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
        
        # Methode 2: Normale-Vektor-Analyse (verwendet SVD-Normale wenn verfügbar)
        normal_score = 0.0
        # Verwende SVD-Normale statt separater normal-Parameter (redundant)
        if svd_normal is not None:
            z_axis = np.array([0, 0, 1])
            cos_angle = np.clip(np.abs(np.dot(svd_normal, z_axis)), 0, 1)
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
        # Normale-Score verwendet jetzt SVD-Normale (redundant zu SVD-Score, aber für Robustheit beibehalten)
        weights = {
            'svd': 0.5 if svd_normal is not None else 0.0,  # Höchste Priorität
            'normal': 0.25 if svd_normal is not None else 0.0,  # Verwendet SVD-Normale
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
            
            # 🎯 ULTRA-ROBUSTE Orientierungserkennung (ohne Einschränkungen)
            # Normale wird intern aus SVD berechnet (robuster als Dreiecks-Methode)
            orientation, dominant_axis = self._determine_orientation_ultra_robust(
                points, plane_model, None  # Normale wird intern berechnet
            )
            
            # Berechne Normale aus SVD für SurfaceGeometry (wird intern bereits berechnet)
            svd_normal, _, _ = self._compute_robust_plane_normal_svd(points)
            
            # #region agent log - Orientierungserkennung
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1",
                        "location": "SurfaceAnalyzer.analyze_surfaces:orientation_detection",
                        "message": "Orientierungserkennung für Surface",
                        "data": {
                            "surface_id": str(surface_id),
                            "name": str(name),
                            "orientation": str(orientation),
                            "dominant_axis": str(dominant_axis) if dominant_axis else None,
                            "num_points": int(len(points)),
                            "x_span": float(np.ptp(xs)) if len(xs) > 0 else 0.0,
                            "y_span": float(np.ptp(ys)) if len(ys) > 0 else 0.0,
                            "z_span": float(np.ptp(zs)) if len(zs) > 0 else 0.0,
                            "x_min": float(min(xs)) if len(xs) > 0 else 0.0,
                            "x_max": float(max(xs)) if len(xs) > 0 else 0.0,
                            "y_min": float(min(ys)) if len(ys) > 0 else 0.0,
                            "y_max": float(max(ys)) if len(ys) > 0 else 0.0,
                            "z_min": float(min(zs)) if len(zs) > 0 else 0.0,
                            "z_max": float(max(zs)) if len(zs) > 0 else 0.0,
                            "has_plane_model": plane_model is not None,
                            "has_normal": svd_normal is not None
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # Speichere zusätzliche Informationen in geometry
            geometry = SurfaceGeometry(
                surface_id=surface_id,
                name=name,
                points=points,
                plane_model=plane_model,
                orientation=orientation,
                normal=svd_normal,  # Verwende SVD-Normale (robuster)
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
    
    def _deduplicate_vertices_and_faces(
        self,
        vertices: np.ndarray,
        faces_flat: np.ndarray,
        resolution: float | None,
        surface_id: str,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entfernt doppelte Vertices (gleiche 3D-Position) aus einem Mesh und
        passt die Faces entsprechend an.

        - Vertices werden mit einer kleinen Toleranz quantisiert.
        - Alle Vertices, die auf dieselbe quantisierte Position fallen,
          werden zu EINEM Vertex zusammengefasst.
        - Faces werden auf die neuen Indizes gemappt.
        - Degenerierte Dreiecke (mit weniger als 3 unterschiedlichen Vertices)
          werden entfernt.
        """
        try:
            verts = np.asarray(vertices, dtype=float)
            faces = np.asarray(faces_flat, dtype=np.int64)

            if verts.size == 0 or faces.size == 0:
                return verts, faces

            # Toleranz: ca. 5 % der effektiven Grid-Resolution, sonst fixer kleiner Wert
            # Erhöht, um auch Vertices zusammenzuführen, die durch _ensure_vertex_coverage
            # und additional_vertices sehr nahe beieinander entstanden sind
            if resolution is not None and resolution > 0:
                tol = float(resolution) * 0.05  # Erhöht von 0.01 auf 0.05 (5%)
            else:
                tol = 1e-3

            # Quantisierte 3D-Koordinaten (reduziert numerisches Rauschen)
            quant = np.round(verts / tol) * tol

            # Mapping von quantisierter Position → neuer Vertex-Index
            from collections import OrderedDict

            pos_to_new_index: "OrderedDict[tuple[float, float, float], int]" = OrderedDict()
            new_vertices_list: list[np.ndarray] = []
            old_to_new_index: list[int] = [0] * len(verts)

            for old_idx, (qx, qy, qz) in enumerate(quant):
                key = (float(qx), float(qy), float(qz))
                if key in pos_to_new_index:
                    new_idx = pos_to_new_index[key]
                else:
                    new_idx = len(new_vertices_list)
                    pos_to_new_index[key] = new_idx
                    new_vertices_list.append(verts[old_idx])
                old_to_new_index[old_idx] = new_idx

            new_vertices = np.array(new_vertices_list, dtype=float)

            # Faces im PyVista-Format [3, v1, v2, v3, 3, v4, v5, v6, ...] neu aufbauen
            if faces.size % 4 != 0:
                # Unerwartetes Format – gib Originaldaten zurück
                return new_vertices, faces

            n_faces = faces.size // 4
            new_faces_list: list[int] = []
            removed_degenerate = 0

            for i in range(n_faces):
                base = i * 4
                nverts = int(faces[base])
                if nverts != 3:
                    # Nur Dreiecke werden erwartet – andere primitives überspringen
                    continue
                v1_old = int(faces[base + 1])
                v2_old = int(faces[base + 2])
                v3_old = int(faces[base + 3])

                v1 = old_to_new_index[v1_old]
                v2 = old_to_new_index[v2_old]
                v3 = old_to_new_index[v3_old]

                # Degenerierte Dreiecke entfernen (weniger als 3 unterschiedliche Vertices)
                if v1 == v2 or v2 == v3 or v1 == v3:
                    removed_degenerate += 1
                    continue

                new_faces_list.extend([3, v1, v2, v3])

            new_faces = np.array(new_faces_list, dtype=np.int64)

            return new_vertices, new_faces

        except Exception as e:
            # Sicherheit: bei Fehlern niemals das Mesh zerstören
            return vertices, faces_flat
    
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
        
        # 🎯 VERTIKALE SURFACES: Maske wird erst in build_single_surface_grid erstellt (benötigt Z_grid)
        # Hier in build_base_grid geben wir eine leere Maske zurück, da Z_grid noch nicht vorhanden ist
        if geometry.orientation == "vertical":
            # Vertikale Surfaces werden in build_single_surface_grid behandelt
            # Hier geben wir eine leere Maske zurück, da die Maske erst später mit Z_grid erstellt wird
            # #region agent log - _create_surface_mask für vertikale Surfaces
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1",
                        "location": "GridBuilder._create_surface_mask:vertical",
                        "message": "_create_surface_mask für vertikale Surfaces - leere Maske zurückgegeben",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "orientation": str(geometry.orientation),
                            "mask_shape": list(X_grid.shape),
                            "mask_size": int(X_grid.size),
                            "empty_mask_returned": True
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            return np.zeros_like(X_grid, dtype=bool)
        
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
                # 🎯 ALLE ECKPUNKTE AKTIVIEREN: Erhöhe max_distance deutlich, damit alle Eckpunkte aktiviert werden
                # Vorher: 0.3x Resolution (zu restriktiv)
                # Jetzt: 2.0x Resolution (aktiviert alle Eckpunkte innerhalb des Polygons)
                max_sq_dist = float((2.0 * base_res) ** 2)
            
            # #region agent log - Vertex Coverage für Ecken prüfen
            activated_vertices = []
            outside_vertices = []
            try:
                from matplotlib.path import Path
                polygon_path = Path(np.column_stack([x_vertices, y_vertices]))
            except Exception:
                polygon_path = None
            # #endregion
            
            for xv, yv in zip(x_vertices, y_vertices):
                # Distanz zu allen Grid-Zentren
                dx = x_flat - xv
                dy = y_flat - yv
                d2 = dx * dx + dy * dy
                # Index des nächsten Grid-Punkts
                nearest_idx = int(np.argmin(d2))
                nearest_x = float(x_flat[nearest_idx])
                nearest_y = float(y_flat[nearest_idx])
                nearest_dist = float(np.sqrt(d2[nearest_idx]))
                
                if max_sq_dist is not None and d2[nearest_idx] > max_sq_dist:
                    # Ecke liegt zu weit vom Grid entfernt → nicht erzwingen
                    continue
                
                # Prüfe ob Grid-Punkt innerhalb des Polygons liegt
                is_inside = True
                if polygon_path is not None:
                    is_inside = polygon_path.contains_point((nearest_x, nearest_y))
                    if not is_inside:
                        outside_vertices.append({
                            "corner": (float(xv), float(yv)),
                            "grid_point": (nearest_x, nearest_y),
                            "distance": nearest_dist
                        })
                        # 🎯 FIX: Aktiviere Grid-Punkte außerhalb des Polygons NICHT
                        continue
                
                # Markiere diesen Grid-Punkt als "inside", falls noch nicht gesetzt
                # UND nur wenn er innerhalb des Polygons liegt
                if not mask_flat[nearest_idx] and is_inside:
                    mask_flat[nearest_idx] = True
                    activated_vertices.append({
                        "corner": (float(xv), float(yv)),
                        "grid_point": (nearest_x, nearest_y),
                        "distance": nearest_dist,
                        "is_inside_polygon": is_inside
                    })
            
            # #region agent log - Vertex Coverage Ergebnisse
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "VERTEX_COVERAGE_ANALYSIS",
                        "location": "FlexibleGridGenerator._ensure_vertex_coverage:vertex_activation",
                        "message": "Vertex Coverage Aktivierung prüfen",
                        "data": {
                            "surface_id": str(getattr(geometry, "surface_id", "unknown")),
                            "activated_vertices_count": len(activated_vertices),
                            "outside_vertices_count": len(outside_vertices),
                            "activated_vertices": activated_vertices,
                            "outside_vertices": outside_vertices,
                            "base_resolution": float(base_res) if 'base_res' in locals() else None,
                            "max_sq_dist": float(max_sq_dist) if max_sq_dist is not None else None
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            return mask_flat.reshape(surface_mask_strict.shape)
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict
    
    def _ensure_vertex_coverage_uv(
        self,
        U_grid: np.ndarray,
        V_grid: np.ndarray,
        polygon_uv: List[Dict[str, float]],
        surface_mask_strict: np.ndarray,
        ) -> np.ndarray:
        """
        Erweiterung der strikten Maske für vertikale Surfaces (in u,v-Koordinaten):
        Stellt sicher, dass für jede Polygon-Ecke mindestens der nächste Grid-Punkt
        als "inside" markiert wird (falls sinnvoll nahe).
        
        Analog zu _ensure_vertex_coverage, aber für (u,v)-Koordinaten.
        """
        try:
            if not polygon_uv or len(polygon_uv) < 3 or U_grid.size == 0 or V_grid.size == 0:
                return surface_mask_strict
            
            u_vertices = np.array([float(p.get("x", 0.0)) for p in polygon_uv], dtype=float)
            v_vertices = np.array([float(p.get("y", 0.0)) for p in polygon_uv], dtype=float)
            if u_vertices.size == 0 or v_vertices.size == 0:
                return surface_mask_strict
            
            u_flat = U_grid.ravel()
            v_flat = V_grid.ravel()
            mask_flat = surface_mask_strict.ravel()
            
            # Schätze typische Grid-Resolution in U/V aus den Koordinaten
            def _estimate_resolution(axis_vals: np.ndarray) -> float:
                if axis_vals.size < 2:
                    return 0.0
                diffs = np.diff(axis_vals)
                diffs = diffs[np.abs(diffs) > 1e-9]
                if diffs.size == 0:
                    return 0.0
                return float(np.median(np.abs(diffs)))
            
            try:
                res_u = _estimate_resolution(U_grid[0, :])
                res_v = _estimate_resolution(V_grid[:, 0])
            except Exception:
                res_u = 0.0
                res_v = 0.0
            base_res = max(res_u, res_v, 0.0)
            if base_res <= 0.0:
                max_sq_dist = None
            else:
                # 🎯 ALLE ECKPUNKTE AKTIVIEREN (UV): Erhöhe max_distance deutlich, damit alle Eckpunkte aktiviert werden
                # Vorher: 0.3x Resolution (zu restriktiv)
                # Jetzt: 2.0x Resolution (aktiviert alle Eckpunkte innerhalb des Polygons)
                max_sq_dist = float((2.0 * base_res) ** 2)
            
            # Prüfe welche Grid-Punkte innerhalb oder auf dem Polygon-Rand liegen
            try:
                from matplotlib.path import Path
                polygon_2d = np.column_stack([u_vertices, v_vertices])
                polygon_path = Path(polygon_2d)
            except Exception:
                polygon_path = None
            
            for uv, vv in zip(u_vertices, v_vertices):
                # Distanz zu allen Grid-Zentren
                du = u_flat - uv
                dv = v_flat - vv
                d2 = du * du + dv * dv
                # Index des nächsten Grid-Punkts
                nearest_idx = int(np.argmin(d2))
                nearest_u = float(u_flat[nearest_idx])
                nearest_v = float(v_flat[nearest_idx])
                
                if max_sq_dist is not None and d2[nearest_idx] > max_sq_dist:
                    # Ecke liegt zu weit vom Grid entfernt → nicht erzwingen
                    continue
                
                # Prüfe ob Grid-Punkt innerhalb des Polygons liegt
                is_inside = True
                if polygon_path is not None:
                    is_inside = polygon_path.contains_point((nearest_u, nearest_v))
                    if not is_inside:
                        # Grid-Punkt außerhalb des Polygons → nicht aktivieren
                        continue
                
                # Markiere diesen Grid-Punkt als "inside", falls noch nicht gesetzt
                # UND nur wenn er innerhalb des Polygons liegt
                if not mask_flat[nearest_idx] and is_inside:
                    mask_flat[nearest_idx] = True
            
            return mask_flat.reshape(surface_mask_strict.shape)
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict
    
    def _add_edge_points_on_boundary(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        geometry: SurfaceGeometry,
        surface_mask_strict: np.ndarray,
        resolution: float,
        ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Generiert Randpunkte direkt auf dem Polygon-Rand für Triangulation.
        
        Strategie:
        - Generiert gleichmäßig verteilte Punkte entlang jeder Polygon-Kante
        - Punktabstand: 0.5x Resolution
        - Randpunkte werden direkt als additional_vertices für Triangulation verwendet
        - KEINE Mapping auf Grid-Punkte mehr
        
        Args:
            X_grid: X-Koordinaten des Grids
            Y_grid: Y-Koordinaten des Grids
            geometry: SurfaceGeometry mit Polygon-Definition
            surface_mask_strict: Strikte Maske (vor Dilatation)
            resolution: Grid-Resolution
            
        Returns:
            Tuple von (surface_mask_strict, edge_points_list)
            - surface_mask_strict: Unveränderte Maske (keine Mapping mehr)
            - edge_points_list: Liste von (x, y) Tupeln der Randpunkte
        """
        try:
            pts = getattr(geometry, "points", None) or []
            if not pts or len(pts) < 3 or X_grid.size == 0 or Y_grid.size == 0:
                return surface_mask_strict
            
            # Nur für planare/schräge Flächen im XY-System – vertikale Flächen
            # arbeiten in (u,v) und haben eigene Logik.
            if geometry.orientation not in ("planar", "sloped"):
                return surface_mask_strict
            
            # 🎯 STRATEGIE: Gleichmäßige Punkteverteilung entlang jeder Polygon-Kante
            n_vertices = len(pts)
            px = np.array([float(p.get("x", 0.0)) for p in pts], dtype=float)
            py = np.array([float(p.get("y", 0.0)) for p in pts], dtype=float)
            

            point_spacing = resolution * 0.5
            # Sammle alle Boundary-Punkte entlang der Kanten
            edge_points = []
            
            # #region agent log - Edge point generation analysis
            edge_generation_stats = []
            # #endregion
            
            for i in range(n_vertices):
                x1, y1 = px[i], py[i]
                x2, y2 = px[(i + 1) % n_vertices], py[(i + 1) % n_vertices]
                
                # Kante: Von (x1, y1) zu (x2, y2)
                dx = x2 - x1
                dy = y2 - y1
                segment_len = math.hypot(dx, dy)
                
                # #region agent log - Edge analysis H1, H5
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1_H5",
                            "location": "FlexibleGridGenerator._add_edge_points_on_boundary:edge_start",
                            "message": "Kanten-Analyse Start",
                            "data": {
                                "edge_index": i,
                                "edge_start": (float(x1), float(y1)),
                                "edge_end": (float(x2), float(y2)),
                                "segment_len": float(segment_len),
                                "point_spacing": float(point_spacing),
                                "expected_points": int(np.ceil(segment_len / point_spacing)) + 1 if segment_len >= 1e-9 else 0
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                if segment_len < 1e-9:
                    # Degenerierte Kante überspringen
                    continue
                
                # 🎯 FIX: Generiere Punkte mit exaktem Abstand point_spacing entlang der Kante
                # Starte bei t=0 und füge Punkte hinzu, bis das Ende erreicht ist
                # #region agent log - Per-edge point calculation (H1, H2, H3)
                edge_points_before = len(edge_points)
                point_positions = []
                # #endregion
                
                # Generiere Punkte mit exaktem Abstand point_spacing
                current_t = 0.0
                while current_t <= 1.0:
                    x_edge = x1 + current_t * dx
                    y_edge = y1 + current_t * dy
                    edge_points.append((x_edge, y_edge))
                    
                    # #region agent log - Point position tracking H2, H5
                    point_positions.append({"t": float(current_t), "x": float(x_edge), "y": float(y_edge)})
                    # #endregion
                    
                    # Berechne nächsten t-Wert basierend auf point_spacing
                    if segment_len > 1e-9:
                        dt = point_spacing / segment_len
                        current_t += dt
                        # Stelle sicher, dass wir nicht über 1.0 hinausgehen
                        if current_t > 1.0:
                            # Füge Endpunkt hinzu, wenn noch nicht erreicht
                            if abs(current_t - 1.0) > 1e-9:
                                x_edge_end = x1 + 1.0 * dx
                                y_edge_end = y1 + 1.0 * dy
                                # Prüfe ob Endpunkt bereits hinzugefügt wurde (Toleranz)
                                if len(edge_points) == 0 or math.hypot(edge_points[-1][0] - x_edge_end, edge_points[-1][1] - y_edge_end) > resolution * 0.01:
                                    edge_points.append((x_edge_end, y_edge_end))
                                    point_positions.append({"t": 1.0, "x": float(x_edge_end), "y": float(y_edge_end)})
                            break
                    else:
                        break
                    
                    # #region agent log - Point position tracking H2, H5
                    point_positions.append({"t": float(current_t), "x": float(x_edge), "y": float(y_edge)})
                    # #endregion
                
                # #region agent log - Edge statistics (H1, H2, H3)
                edge_points_after = len(edge_points)
                points_generated = edge_points_after - edge_points_before
                # Berechne tatsächliche Abstände zwischen Punkten
                actual_distances = []
                if len(point_positions) > 1:
                    for k in range(len(point_positions) - 1):
                        p1 = point_positions[k]
                        p2 = point_positions[k + 1]
                        dist = math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
                        actual_distances.append(dist)
                elif len(point_positions) == 1:
                    actual_distances = [segment_len]  # Nur ein Punkt auf der Kante
                
                avg_spacing = float(np.mean(actual_distances)) if len(actual_distances) > 0 else segment_len
                max_spacing = float(np.max(actual_distances)) if len(actual_distances) > 0 else segment_len
                min_spacing = float(np.min(actual_distances)) if len(actual_distances) > 0 else segment_len
                
                edge_generation_stats.append({
                    "edge_index": i,
                    "segment_len": float(segment_len),
                    "point_spacing_target": float(point_spacing),
                    "n_points_generated": points_generated,
                    "avg_actual_spacing": avg_spacing,
                    "max_actual_spacing": max_spacing,
                    "min_actual_spacing": min_spacing,
                    "spacing_ratio_avg": float(avg_spacing / point_spacing) if point_spacing > 0 else 0.0,
                    "spacing_ratio_max": float(max_spacing / point_spacing) if point_spacing > 0 else 0.0,
                    "actual_distances": actual_distances,
                    "point_positions": point_positions
                })
                # #endregion
            
            # Entferne doppelte Punkte (z.B. Eckpunkte die mehrfach auftreten)
            tolerance = resolution * 0.05
            unique_edge_points = []
            duplicates_removed = 0
            
            # #region agent log - Before deduplication H3
            points_before_dedup = len(edge_points)
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H3",
                        "location": "FlexibleGridGenerator._add_edge_points_on_boundary:before_dedup",
                        "message": "Vor Deduplizierung",
                        "data": {
                            "points_before_dedup": points_before_dedup,
                            "tolerance": float(tolerance)
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            for x, y in edge_points:
                is_duplicate = False
                for ex, ey in unique_edge_points:
                    if abs(x - ex) < tolerance and abs(y - ey) < tolerance:
                        is_duplicate = True
                        # #region agent log - Duplicate found H3
                        try:
                            import json, time as _t
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H3",
                                    "location": "FlexibleGridGenerator._add_edge_points_on_boundary:duplicate_found",
                                    "message": "Doppelter Punkt gefunden",
                                    "data": {
                                        "point": (float(x), float(y)),
                                        "existing": (float(ex), float(ey)),
                                        "distance": float(math.hypot(x - ex, y - ey))
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        break
                if not is_duplicate:
                    unique_edge_points.append((x, y))
                else:
                    duplicates_removed += 1
            
            edge_points = unique_edge_points
            
            # #region agent log - After deduplication and final stats (H4)
            points_after_dedup = len(edge_points)
            
            # Berechne Abstände zwischen finalen Punkten (in Reihenfolge entlang Polygon)
            final_distances = []
            if len(edge_points) > 1:
                # Sortiere Punkte nach Polygon-Reihenfolge: für jede Kante, prüfe welche Punkte darauf liegen
                sorted_edge_points = []
                remaining_points = list(edge_points)
                
                # Für jede Kante: finde Punkte, die darauf liegen
                for i in range(n_vertices):
                    x1, y1 = px[i], py[i]
                    x2, y2 = px[(i + 1) % n_vertices], py[(i + 1) % n_vertices]
                    dx_edge = x2 - x1
                    dy_edge = y2 - y1
                    edge_len = math.hypot(dx_edge, dy_edge)
                    
                    if edge_len > 1e-9:
                        # Sammle Punkte auf dieser Kante
                        points_on_this_edge = []
                        for x, y in remaining_points:
                            # Prüfe ob Punkt auf dieser Kante liegt
                            to_point = np.array([x - x1, y - y1])
                            edge_vec = np.array([dx_edge, dy_edge])
                            t_proj = np.dot(to_point, edge_vec) / (edge_len * edge_len)
                            proj_point = np.array([x1, y1]) + t_proj * edge_vec
                            dist_to_edge = math.hypot(x - proj_point[0], y - proj_point[1])
                            
                            # Punkt liegt auf Kante wenn Abstand < tolerance und t zwischen 0 und 1
                            if dist_to_edge < tolerance and 0.0 <= t_proj <= 1.0:
                                points_on_this_edge.append((x, y, t_proj))
                        
                        # Sortiere Punkte auf dieser Kante nach t
                        points_on_this_edge.sort(key=lambda p: p[2])
                        sorted_edge_points.extend([(p[0], p[1]) for p in points_on_this_edge])
                        # Entferne bereits verwendete Punkte
                        for px_val, py_val, _ in points_on_this_edge:
                            remaining_points = [(x, y) for x, y in remaining_points if not (abs(x - px_val) < tolerance and abs(y - py_val) < tolerance)]
                
                # Berechne Abstände zwischen aufeinanderfolgenden Punkten
                if len(sorted_edge_points) > 1:
                    for i in range(len(sorted_edge_points) - 1):
                        x1, y1 = sorted_edge_points[i]
                        x2, y2 = sorted_edge_points[i + 1]
                        dist = math.hypot(x2 - x1, y2 - y1)
                        final_distances.append(dist)
            
            # Logge Zusammenfassung der Edge-Generierung
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "EDGE_POINT_DISTRIBUTION_ANALYSIS",
                        "location": "FlexibleGridGenerator._add_edge_points_on_boundary:edge_generation",
                        "message": "Randpunkt-Verteilung Analyse",
                        "data": {
                            "surface_id": str(getattr(geometry, "surface_id", "unknown")),
                            "orientation": str(getattr(geometry, "orientation", "unknown")),
                            "resolution": float(resolution),
                            "point_spacing_target": float(point_spacing),
                            "n_edges": len(edge_generation_stats),
                            "points_before_dedup": points_before_dedup,
                            "points_after_dedup": points_after_dedup,
                            "duplicates_removed": duplicates_removed,
                            "final_distances_after_dedup": final_distances if final_distances else [],
                            "final_distances_stats": {
                                "count": len(final_distances),
                                "min": float(min(final_distances)) if final_distances else None,
                                "max": float(max(final_distances)) if final_distances else None,
                                "avg": float(np.mean(final_distances)) if final_distances else None,
                                "std": float(np.std(final_distances)) if final_distances else None
                            } if final_distances else {},
                            "edge_statistics": edge_generation_stats
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if not edge_points:
                return surface_mask_strict, []
            
            # 🎯 NEUE LOGIK: Randpunkte werden direkt als additional_vertices verwendet
            # KEINE Mapping auf Grid-Punkte mehr
            # #region agent log - Randpunkte generiert (ohne Grid-Mapping)
            # (Logging entfernt - Randpunkt-Logik nicht mehr aktiv)
            # #endregion
            
            # Rückgabe: Unveränderte Maske + Liste der Randpunkte
            return surface_mask_strict, edge_points
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict, []
    
    def _add_edge_points_on_boundary_uv(
        self,
        U_grid: np.ndarray,
        V_grid: np.ndarray,
        polygon_uv: List[Dict[str, float]],
        surface_mask_strict: np.ndarray,
        resolution: float,
        ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Generiert Randpunkte direkt auf dem Polygon-Rand für vertikale Surfaces (in u,v-Koordinaten).
        
        Strategie:
        - Generiert gleichmäßig verteilte Punkte entlang jeder Polygon-Kante
        - Punktabstand: 0.5x Resolution
        - Randpunkte werden direkt als additional_vertices für Triangulation verwendet
        - KEINE Mapping auf Grid-Punkte mehr
        
        Args:
            U_grid: U-Koordinaten des Grids (u=x für X-Z-Wand, u=y für Y-Z-Wand)
            V_grid: V-Koordinaten des Grids (v=z für beide Wandtypen)
            polygon_uv: Polygon-Definition in (u,v)-Koordinaten
            surface_mask_strict: Strikte Maske (vor Randpunkten)
            resolution: Grid-Resolution
            
        Returns:
            Tuple von (surface_mask_strict, edge_points_list)
            - surface_mask_strict: Unveränderte Maske (keine Mapping mehr)
            - edge_points_list: Liste von (u, v) Tupeln der Randpunkte
        """
        try:
            if not polygon_uv or len(polygon_uv) < 3 or U_grid.size == 0 or V_grid.size == 0:
                return surface_mask_strict
            
            # 🎯 STRATEGIE: Gleichmäßige Punkteverteilung entlang jeder Polygon-Kante
            n_vertices = len(polygon_uv)
            pu = np.array([float(p.get("x", 0.0)) for p in polygon_uv], dtype=float)
            pv = np.array([float(p.get("y", 0.0)) for p in polygon_uv], dtype=float)
            
            # Punktabstand entlang der Kante: 0.5x Resolution (gleich wie für planare Flächen)
            point_spacing = resolution * 0.5
            
            # Sammle alle Boundary-Punkte entlang der Kanten
            edge_points = []
            
            # #region agent log - Edge point generation analysis (UV)
            edge_generation_stats_uv = []
            # #endregion
            
            for i in range(n_vertices):
                u1, v1 = pu[i], pv[i]
                u2, v2 = pu[(i + 1) % n_vertices], pv[(i + 1) % n_vertices]
                
                # Kante: Von (u1, v1) zu (u2, v2)
                du = u2 - u1
                dv = v2 - v1
                segment_len = math.hypot(du, dv)
                
                if segment_len < 1e-9:
                    # Degenerierte Kante überspringen
                    continue
                
                # 🎯 FIX: Generiere Punkte mit exaktem Abstand point_spacing entlang der Kante
                # #region agent log - Per-edge point calculation UV (H1, H2, H3)
                edge_points_before = len(edge_points)
                point_positions = []
                # #endregion
                
                # Generiere Punkte mit exaktem Abstand point_spacing
                current_t = 0.0
                while current_t <= 1.0:
                    u_edge = u1 + current_t * du
                    v_edge = v1 + current_t * dv
                    edge_points.append((u_edge, v_edge))
                    
                    # #region agent log - Point position tracking UV (H2)
                    point_positions.append({"t": float(current_t), "u": float(u_edge), "v": float(v_edge)})
                    # #endregion
                    
                    # Berechne nächsten t-Wert basierend auf point_spacing
                    if segment_len > 1e-9:
                        dt = point_spacing / segment_len
                        current_t += dt
                        # Stelle sicher, dass wir nicht über 1.0 hinausgehen
                        if current_t > 1.0:
                            # Füge Endpunkt hinzu, wenn noch nicht erreicht
                            if abs(current_t - 1.0) > 1e-9:
                                u_edge_end = u1 + 1.0 * du
                                v_edge_end = v1 + 1.0 * dv
                                # Prüfe ob Endpunkt bereits hinzugefügt wurde (Toleranz)
                                if len(edge_points) == 0 or math.hypot(edge_points[-1][0] - u_edge_end, edge_points[-1][1] - v_edge_end) > resolution * 0.01:
                                    edge_points.append((u_edge_end, v_edge_end))
                                    point_positions.append({"t": 1.0, "u": float(u_edge_end), "v": float(v_edge_end)})
                            break
                    else:
                        break
                
                # #region agent log - Edge statistics UV (H1, H2, H3)
                edge_points_after = len(edge_points)
                points_generated = edge_points_after - edge_points_before
                # Berechne tatsächliche Abstände zwischen Punkten
                actual_distances = []
                if len(point_positions) > 1:
                    for k in range(len(point_positions) - 1):
                        p1 = point_positions[k]
                        p2 = point_positions[k + 1]
                        dist = math.hypot(p2["u"] - p1["u"], p2["v"] - p1["v"])
                        actual_distances.append(dist)
                elif len(point_positions) == 1:
                    actual_distances = [segment_len]
                
                avg_spacing = float(np.mean(actual_distances)) if len(actual_distances) > 0 else segment_len
                max_spacing = float(np.max(actual_distances)) if len(actual_distances) > 0 else segment_len
                min_spacing = float(np.min(actual_distances)) if len(actual_distances) > 0 else segment_len
                
                edge_generation_stats_uv.append({
                    "edge_index": i,
                    "segment_len": float(segment_len),
                    "point_spacing_target": float(point_spacing),
                    "n_points_generated": points_generated,
                    "avg_actual_spacing": avg_spacing,
                    "max_actual_spacing": max_spacing,
                    "min_actual_spacing": min_spacing,
                    "spacing_ratio_avg": float(avg_spacing / point_spacing) if point_spacing > 0 else 0.0,
                    "spacing_ratio_max": float(max_spacing / point_spacing) if point_spacing > 0 else 0.0,
                    "actual_distances": actual_distances,
                    "point_positions": point_positions
                })
                # #endregion
            
            # Entferne doppelte Punkte (z.B. Eckpunkte die mehrfach auftreten)
            tolerance = resolution * 0.05
            unique_edge_points = []
            duplicates_removed = 0
            
            # #region agent log - Before deduplication UV (H4)
            points_before_dedup = len(edge_points)
            # #endregion
            
            for u, v in edge_points:
                is_duplicate = False
                for eu, ev in unique_edge_points:
                    if abs(u - eu) < tolerance and abs(v - ev) < tolerance:
                        is_duplicate = True
                        duplicates_removed += 1
                        break
                if not is_duplicate:
                    unique_edge_points.append((u, v))
            
            edge_points = unique_edge_points
            
            # #region agent log - After deduplication UV and final stats (H4)
            points_after_dedup = len(edge_points)
            
            # Logge Zusammenfassung der Edge-Generierung (UV)
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "EDGE_POINT_DISTRIBUTION_ANALYSIS_UV",
                        "location": "FlexibleGridGenerator._add_edge_points_on_boundary_uv:edge_generation",
                        "message": "Randpunkt-Verteilung Analyse (UV)",
                        "data": {
                            "resolution": float(resolution),
                            "point_spacing_target": float(point_spacing),
                            "n_edges": len(edge_generation_stats_uv),
                            "points_before_dedup": points_before_dedup,
                            "points_after_dedup": points_after_dedup,
                            "duplicates_removed": duplicates_removed,
                            "edge_statistics": edge_generation_stats_uv
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if not edge_points:
                return surface_mask_strict, []
            
            # 🎯 NEUE LOGIK: Randpunkte werden direkt als additional_vertices verwendet
            # KEINE Mapping auf Grid-Punkte mehr
            # #region agent log - Randpunkte generiert (UV, ohne Grid-Mapping)
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "EDGE_POINTS_AS_VERTICES_UV",
                        "location": "FlexibleGridGenerator._add_edge_points_on_boundary_uv:edge_points_generated",
                        "message": "Randpunkte generiert (UV, direkt als Vertices, kein Grid-Mapping)",
                        "data": {
                            "n_edge_points_generated": len(edge_points),
                            "points_before_dedup": points_before_dedup,
                            "points_after_dedup": len(edge_points),
                            "duplicates_removed": duplicates_removed,
                            "resolution": float(resolution),
                            "point_spacing": float(point_spacing)
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # Rückgabe: Unveränderte Maske + Liste der Randpunkte
            return surface_mask_strict, edge_points
            
            # #region agent log - Boundary-Punkte generiert und aktiviert (UV)
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json, time as _t
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "BOUNDARY_POINTS_CREATION",
                        "location": "FlexibleGridGenerator._add_edge_points_on_boundary_uv",
                        "message": "Randpunkte generiert und aktiviert (UV, lineare Verteilung)",
                        "data": {
                            "n_edge_points_generated": len(edge_points),
                            "edge_points_activated": edge_points_activated,
                            "points_before": points_activated_before,
                            "points_after": points_activated_after,
                            "resolution": float(resolution),
                            "point_spacing": float(point_spacing),
                            "max_distance_threshold": float(max_distance),
                            "total_grid_points": int(U_grid.size)
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            return mask_flat.reshape(surface_mask_strict.shape)
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict
    
    def _calculate_additional_vertices(
        self,
        geometry: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        resolution: Optional[float],
        ) -> np.ndarray:
        """
        Berechnet zusätzliche Vertices (Ecken + Randpunkte) für eine Surface.
        
        Args:
            geometry: SurfaceGeometry Objekt
            X_grid: X-Koordinaten des Grids
            Y_grid: Y-Koordinaten des Grids
            Z_grid: Z-Koordinaten des Grids
            surface_mask: Maske für aktive Grid-Punkte
            resolution: Grid-Resolution
            
        Returns:
            np.ndarray: Array von additional_vertices (Shape: (N, 3))
        """
        # #region agent log - Eintritt _calculate_additional_vertices
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H2",
                    "location": "GridBuilder._calculate_additional_vertices:entry",
                    "message": "Eintritt _calculate_additional_vertices",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "orientation": str(geometry.orientation),
                        "resolution": float(resolution) if resolution is not None else None,
                        "surface_points_count": len(geometry.points) if geometry.points else 0
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        additional_vertices = []
        surface_points = geometry.points or []
        
        if len(surface_points) < 3:
            # #region agent log - Zu wenige Surface-Punkte
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H3",
                        "location": "GridBuilder._calculate_additional_vertices:too_few_points",
                        "message": "Zu wenige Surface-Punkte",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "surface_points_count": len(surface_points)
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            return np.array([], dtype=float).reshape(0, 3)
        
        # Sammle alle Grid-Punkte für Duplikat-Prüfung
        grid_points_flat = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        mask_flat = surface_mask.reshape(-1)
        all_vertices = grid_points_flat[mask_flat] if np.any(mask_flat) else np.array([], dtype=float).reshape(0, 3)
        
        # 1. Ecken-Punkte hinzufügen
        if geometry.orientation == "vertical":
            # VERTIKALE FLÄCHEN: Verwende (u,v)-Koordinaten
            xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
            ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
            zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            z_span = float(np.ptp(zs))
            
            x_mean = float(np.mean(xs))
            y_mean = float(np.mean(ys))
            
            is_slanted_wall = False
            is_xz_wall = True
            
            if hasattr(geometry, 'dominant_axis') and geometry.dominant_axis:
                if geometry.dominant_axis == "yz":
                    polygon_u = ys
                    polygon_v = zs
                    is_xz_wall = False
                    max_main_span = max(y_span, z_span) if max(y_span, z_span) > 1e-6 else 1.0
                    x_variation_ratio = x_span / max_main_span if max_main_span > 1e-6 else 0.0
                    is_slanted_wall = (x_variation_ratio > 0.1 and z_span > 1e-3)
                elif geometry.dominant_axis == "xz":
                    polygon_u = xs
                    polygon_v = zs
                    is_xz_wall = True
                    max_main_span = max(x_span, z_span) if max(x_span, z_span) > 1e-6 else 1.0
                    y_variation_ratio = y_span / max_main_span if max_main_span > 1e-6 else 0.0
                    is_slanted_wall = (y_variation_ratio > 0.1 and z_span > 1e-3)
            
            if is_slanted_wall:
                from scipy.interpolate import griddata
                if is_xz_wall:
                    points_surface = np.column_stack([xs, zs])
                else:
                    points_surface = np.column_stack([ys, zs])
            
            exact_match_tolerance = 1e-9
            try:
                from matplotlib.path import Path
                polygon_2d_uv = np.column_stack([polygon_u, polygon_v])
                polygon_path_uv = Path(polygon_2d_uv)
            except Exception:
                polygon_path_uv = None
            
            for corner_u, corner_v in zip(polygon_u, polygon_v):
                if is_xz_wall:
                    distances = np.sqrt((all_vertices[:, 0] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                    nearest_idx = int(np.argmin(distances))
                    nearest_u = float(all_vertices[nearest_idx, 0])
                    nearest_v = float(all_vertices[nearest_idx, 2])
                else:
                    distances = np.sqrt((all_vertices[:, 1] - corner_u)**2 + (all_vertices[:, 2] - corner_v)**2)
                    nearest_idx = int(np.argmin(distances))
                    nearest_u = float(all_vertices[nearest_idx, 1])
                    nearest_v = float(all_vertices[nearest_idx, 2])
                
                min_distance = np.min(distances)
                nearest_inside_polygon = True
                if polygon_path_uv is not None:
                    nearest_inside_polygon = polygon_path_uv.contains_point((nearest_u, nearest_v))
                
                if min_distance > exact_match_tolerance or not nearest_inside_polygon:
                    if is_xz_wall:
                        corner_x = corner_u
                        corner_z = corner_v
                        if is_slanted_wall:
                            from scipy.interpolate import griddata
                            corner_y = griddata(
                                points_surface, ys,
                                np.array([[corner_u, corner_v]]),
                                method='linear', fill_value=y_mean
                            )[0]
                        else:
                            corner_y = y_mean
                    else:
                        corner_y = corner_u
                        corner_z = corner_v
                        if is_slanted_wall:
                            from scipy.interpolate import griddata
                            corner_x = griddata(
                                points_surface, xs,
                                np.array([[corner_u, corner_v]]),
                                method='linear', fill_value=x_mean
                            )[0]
                        else:
                            corner_x = x_mean
                    
                    additional_vertices.append([corner_x, corner_y, corner_z])
        else:
            # PLANARE/SCHRÄGE FLÄCHEN: Verwende (x,y)-Koordinaten
            polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
            polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
            
            exact_match_tolerance = 1e-9
            try:
                from matplotlib.path import Path
                polygon_path = Path(np.column_stack([polygon_x, polygon_y]))
            except Exception:
                polygon_path = None
            
            for corner_x, corner_y in zip(polygon_x, polygon_y):
                distances = np.sqrt((all_vertices[:, 0] - corner_x)**2 + (all_vertices[:, 1] - corner_y)**2)
                min_distance = np.min(distances)
                nearest_idx = int(np.argmin(distances))
                nearest_vertex = (float(all_vertices[nearest_idx, 0]), float(all_vertices[nearest_idx, 1]))
                
                nearest_inside_polygon = True
                if polygon_path is not None:
                    nearest_inside_polygon = polygon_path.contains_point(nearest_vertex)
                
                if min_distance > exact_match_tolerance or not nearest_inside_polygon:
                    corner_z = 0.0
                    if geometry.plane_model:
                        try:
                            z_new = _evaluate_plane_on_grid(
                                geometry.plane_model,
                                np.array([[corner_x]]),
                                np.array([[corner_y]])
                            )
                            if z_new is not None and z_new.size > 0:
                                corner_z = float(z_new.flat[0])
                        except Exception:
                            pass
                    
                    additional_vertices.append([corner_x, corner_y, corner_z])
        
        # 2. Randpunkte hinzufügen
        boundary_edge_points = []
        if resolution is None or resolution <= 0:
            # Fallback: Verwende Grid-Auflösung aus X/Y-Grid
            if X_grid.size > 1 and Y_grid.size > 1:
                x_diffs = np.diff(np.unique(X_grid.ravel()))
                y_diffs = np.diff(np.unique(Y_grid.ravel()))
                if x_diffs.size > 0 and y_diffs.size > 0:
                    point_spacing = min(float(np.mean(x_diffs)), float(np.mean(y_diffs)))
                else:
                    point_spacing = 0.5  # Fallback
            else:
                point_spacing = 0.5  # Fallback
        else:
            point_spacing = resolution
        
        # #region agent log - Point spacing bestimmt
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H2",
                    "location": "GridBuilder._calculate_additional_vertices:point_spacing",
                    "message": "Point spacing bestimmt",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "point_spacing": float(point_spacing),
                        "resolution_was_none": resolution is None,
                        "resolution_value": float(resolution) if resolution is not None else None
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        if geometry.orientation == "vertical":
            xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
            ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
            zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            z_span = float(np.ptp(zs))
            
            is_slanted_wall = False
            is_xz_wall = True
            
            if hasattr(geometry, 'dominant_axis') and geometry.dominant_axis:
                if geometry.dominant_axis == "yz":
                    polygon_u = ys
                    polygon_v = zs
                    is_xz_wall = False
                    max_main_span = max(y_span, z_span) if max(y_span, z_span) > 1e-6 else 1.0
                    x_variation_ratio = x_span / max_main_span if max_main_span > 1e-6 else 0.0
                    is_slanted_wall = (x_variation_ratio > 0.1 and z_span > 1e-3)
                elif geometry.dominant_axis == "xz":
                    polygon_u = xs
                    polygon_v = zs
                    is_xz_wall = True
                    max_main_span = max(x_span, z_span) if max(x_span, z_span) > 1e-6 else 1.0
                    y_variation_ratio = y_span / max_main_span if max_main_span > 1e-6 else 0.0
                    is_slanted_wall = (y_variation_ratio > 0.1 and z_span > 1e-3)
                else:
                    polygon_u = xs
                    polygon_v = zs
                    is_xz_wall = True
            else:
                polygon_u = xs
                polygon_v = zs
                is_xz_wall = True
            
            if is_slanted_wall:
                from scipy.interpolate import griddata
                if is_xz_wall:
                    points_surface = np.column_stack([xs, zs])
                else:
                    points_surface = np.column_stack([ys, zs])
            
            n_vertices = len(polygon_u)
            for i in range(n_vertices):
                u1, v1 = polygon_u[i], polygon_v[i]
                u2, v2 = polygon_u[(i + 1) % n_vertices], polygon_v[(i + 1) % n_vertices]
                
                edge_len = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
                if edge_len < 1e-9:
                    continue
                
                n_points_on_edge = max(1, int(np.ceil(edge_len / point_spacing)))
                for j in range(1, n_points_on_edge):
                    t = j / n_points_on_edge
                    u = u1 + t * (u2 - u1)
                    v = v1 + t * (v2 - v1)
                    
                    if is_xz_wall:
                        x, z = u, v
                        if is_slanted_wall:
                            from scipy.interpolate import griddata
                            y = griddata(
                                points_surface, ys,
                                np.array([[x, z]]),
                                method='linear', fill_value=float(np.mean(ys))
                            )[0]
                        else:
                            y = float(np.mean(ys))
                    else:
                        y, z = u, v
                        if is_slanted_wall:
                            from scipy.interpolate import griddata
                            x = griddata(
                                points_surface, xs,
                                np.array([[y, z]]),
                                method='linear', fill_value=float(np.mean(xs))
                            )[0]
                        else:
                            x = float(np.mean(xs))
                    
                    boundary_edge_points.append([float(x), float(y), float(z)])
        else:
            # PLANARE/SCHRÄGE FLÄCHEN
            polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
            polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
            
            n_vertices = len(polygon_x)
            for i in range(n_vertices):
                x1, y1 = polygon_x[i], polygon_y[i]
                x2, y2 = polygon_x[(i + 1) % n_vertices], polygon_y[(i + 1) % n_vertices]
                
                edge_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if edge_len < 1e-9:
                    continue
                
                n_points_on_edge = max(1, int(np.ceil(edge_len / point_spacing)))
                for j in range(1, n_points_on_edge):
                    t = j / n_points_on_edge
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    
                    z = 0.0
                    if geometry.plane_model:
                        try:
                            z_new = _evaluate_plane_on_grid(
                                geometry.plane_model,
                                np.array([[x]]),
                                np.array([[y]])
                            )
                            if z_new is not None and z_new.size > 0:
                                z = float(z_new.flat[0])
                        except Exception:
                            pass
                    
                    boundary_edge_points.append([float(x), float(y), float(z)])
        
        # #region agent log - Boundary edge points vor Duplikat-Prüfung
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H3,H4",
                    "location": "GridBuilder._calculate_additional_vertices:before_dedup",
                    "message": "Boundary edge points vor Duplikat-Prüfung",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "boundary_edge_points_count": len(boundary_edge_points),
                        "additional_vertices_corners_count": len(additional_vertices)
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 3. Duplikat-Prüfung und Zusammenführung
        dedup_tolerance = point_spacing * 0.05
        for edge_point in boundary_edge_points:
            is_duplicate = False
            
            # Prüfe gegen bestehende additional_vertices
            for existing_vertex in additional_vertices:
                dist = np.sqrt(
                    (edge_point[0] - existing_vertex[0])**2 +
                    (edge_point[1] - existing_vertex[1])**2 +
                    (edge_point[2] - existing_vertex[2])**2
                )
                if dist < dedup_tolerance:
                    is_duplicate = True
                    break
            
            # Prüfe gegen Grid-Punkte
            if not is_duplicate:
                for grid_point in all_vertices:
                    dist = np.sqrt(
                        (edge_point[0] - grid_point[0])**2 +
                        (edge_point[1] - grid_point[1])**2 +
                        (edge_point[2] - grid_point[2])**2
                    )
                    if dist < dedup_tolerance:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                additional_vertices.append(edge_point)
        
        # #region agent log - Ergebnis vor Rückgabe
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H3,H4",
                    "location": "GridBuilder._calculate_additional_vertices:before_return",
                    "message": "Ergebnis vor Rückgabe",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "additional_vertices_count": len(additional_vertices),
                        "boundary_edge_points_count": len(boundary_edge_points),
                        "dedup_tolerance": float(dedup_tolerance)
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        return np.array(additional_vertices, dtype=float) if len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
    
    @measure_time("GridBuilder.build_single_surface_grid")
    def build_single_surface_grid(
        self,
        geometry: SurfaceGeometry,
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6,
        padding_factor: float = 0.5,
        disable_edge_refinement: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # Stelle sicher, dass resolution gesetzt ist
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
            # 🎯 FIX: Bei X-Z-Wänden sollte Y immer konstant sein, auch wenn Original-Punkte leicht variieren
            # Schrägheitsprüfung nur für wirklich schräge Wände (Variation > 10% der Hauptspannen)
            is_slanted_vertical = False
            if hasattr(geometry, 'dominant_axis') and geometry.dominant_axis:
                # Wenn dominant_axis vorhanden ist, prüfe ob Y oder X signifikant variiert
                if geometry.dominant_axis == "xz":
                    # X-Z-Wand: Y sollte normalerweise konstant sein
                    # Nur als schräg behandeln, wenn Y-Variation signifikant ist (>10% von max(x_span, z_span))
                    max_main_span = max(x_span, z_span) if max(x_span, z_span) > 1e-6 else 1.0
                    y_variation_ratio = y_span / max_main_span if max_main_span > 1e-6 else 0.0
                    # Nur als schräg behandeln, wenn Y-Variation > 10% der Hauptspannen UND Z-Spanne signifikant
                    is_slanted_vertical = (y_variation_ratio > 0.1 and z_span > 1e-3)
                elif geometry.dominant_axis == "yz":
                    # Y-Z-Wand: X sollte normalerweise konstant sein
                    # Nur als schräg behandeln, wenn X-Variation signifikant ist (>10% von max(y_span, z_span))
                    max_main_span = max(y_span, z_span) if max(y_span, z_span) > 1e-6 else 1.0
                    x_variation_ratio = x_span / max_main_span if max_main_span > 1e-6 else 0.0
                    # Nur als schräg behandeln, wenn X-Variation > 10% der Hauptspannen UND Z-Spanne signifikant
                    is_slanted_vertical = (x_variation_ratio > 0.1 and z_span > 1e-3)
            
            
            
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
                
                # #region agent log - Grid-Größe für vertikale Surfaces (X-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "GridBuilder.build_single_surface_grid:vertical_xz_grid_created",
                            "message": "Grid-Größe für vertikale Surface (X-Z-Wand) erstellt",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "orientation": "vertical",
                                "wall_type": "xz",
                                "u_axis_size": int(u_axis.size),
                                "v_axis_size": int(v_axis.size),
                                "U_grid_shape": list(U_grid.shape),
                                "V_grid_shape": list(V_grid.shape),
                                "u_min": float(u_min),
                                "u_max": float(u_max),
                                "v_min": float(v_min),
                                "v_max": float(v_max),
                                "y0": float(y0),
                                "resolution": float(step)
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Transformiere zu (X, Y, Z) Koordinaten
                X_grid = U_grid  # u = x
                Y_grid = np.full_like(U_grid, y0, dtype=float)  # y = konstant
                Z_grid = V_grid  # v = z
                
                # #region agent log - Koordinatentransformation Konsistenz-Check (X-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        # Prüfe Konsistenz: Original-Punkte vs. Grid-Koordinaten
                        points_x = np.array([p.get('x', 0.0) for p in geometry.points], dtype=float)
                        points_y = np.array([p.get('y', 0.0) for p in geometry.points], dtype=float)
                        points_z = np.array([p.get('z', 0.0) for p in geometry.points], dtype=float)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "GridBuilder.build_single_surface_grid:vertical_xz_transform_consistency",
                            "message": "Koordinatentransformation Konsistenz-Check (X-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "transformation_type": "xz_wall_constant",
                                "u_axis_min": float(u_min),
                                "u_axis_max": float(u_max),
                                "v_axis_min": float(v_min),
                                "v_axis_max": float(v_max),
                                "y0_constant": float(y0),
                                "X_grid_min": float(X_grid.min()),
                                "X_grid_max": float(X_grid.max()),
                                "Y_grid_min": float(Y_grid.min()),
                                "Y_grid_max": float(Y_grid.max()),
                                "Z_grid_min": float(Z_grid.min()),
                                "Z_grid_max": float(Z_grid.max()),
                                "points_x_min": float(points_x.min()),
                                "points_x_max": float(points_x.max()),
                                "points_y_min": float(points_y.min()),
                                "points_y_max": float(points_y.max()),
                                "points_z_min": float(points_z.min()),
                                "points_z_max": float(points_z.max()),
                                "y0_matches_points": bool(np.allclose([y0], [points_y.mean()], atol=1e-6))
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # #region agent log - Koordinatentransformation (X-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "GridBuilder.build_single_surface_grid:vertical_xz_transform",
                            "message": "Koordinatentransformation für vertikale Surface (X-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "X_grid_shape": list(X_grid.shape),
                                "Y_grid_shape": list(Y_grid.shape),
                                "Z_grid_shape": list(Z_grid.shape),
                                "X_min": float(X_grid.min()),
                                "X_max": float(X_grid.max()),
                                "Y_min": float(Y_grid.min()),
                                "Y_max": float(Y_grid.max()),
                                "Z_min": float(Z_grid.min()),
                                "Z_max": float(Z_grid.max()),
                                "Y_constant": float(y0)
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
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
                
                # #region agent log - Grid-Größe für vertikale Surfaces (Y-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "GridBuilder.build_single_surface_grid:vertical_yz_grid_created",
                            "message": "Grid-Größe für vertikale Surface (Y-Z-Wand) erstellt",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "orientation": "vertical",
                                "wall_type": "yz",
                                "u_axis_size": int(u_axis.size),
                                "v_axis_size": int(v_axis.size),
                                "U_grid_shape": list(U_grid.shape),
                                "V_grid_shape": list(V_grid.shape),
                                "u_min": float(u_min),
                                "u_max": float(u_max),
                                "v_min": float(v_min),
                                "v_max": float(v_max),
                                "x0": float(x0),
                                "resolution": float(step)
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Transformiere zu (X, Y, Z) Koordinaten
                X_grid = np.full_like(U_grid, x0, dtype=float)  # x = konstant
                Y_grid = U_grid  # u = y
                Z_grid = V_grid  # v = z
                
                # #region agent log - Koordinatentransformation Konsistenz-Check (Y-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        # Prüfe Konsistenz: Original-Punkte vs. Grid-Koordinaten
                        points_x = np.array([p.get('x', 0.0) for p in geometry.points], dtype=float)
                        points_y = np.array([p.get('y', 0.0) for p in geometry.points], dtype=float)
                        points_z = np.array([p.get('z', 0.0) for p in geometry.points], dtype=float)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "GridBuilder.build_single_surface_grid:vertical_yz_transform_consistency",
                            "message": "Koordinatentransformation Konsistenz-Check (Y-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "transformation_type": "yz_wall_constant",
                                "u_axis_min": float(u_min),
                                "u_axis_max": float(u_max),
                                "v_axis_min": float(v_min),
                                "v_axis_max": float(v_max),
                                "x0_constant": float(x0),
                                "X_grid_min": float(X_grid.min()),
                                "X_grid_max": float(X_grid.max()),
                                "Y_grid_min": float(Y_grid.min()),
                                "Y_grid_max": float(Y_grid.max()),
                                "Z_grid_min": float(Z_grid.min()),
                                "Z_grid_max": float(Z_grid.max()),
                                "points_x_min": float(points_x.min()),
                                "points_x_max": float(points_x.max()),
                                "points_y_min": float(points_y.min()),
                                "points_y_max": float(points_y.max()),
                                "points_z_min": float(points_z.min()),
                                "points_z_max": float(points_z.max()),
                                "x0_matches_points": bool(np.allclose([x0], [points_x.mean()], atol=1e-6))
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # #region agent log - Koordinatentransformation (Y-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "GridBuilder.build_single_surface_grid:vertical_yz_transform",
                            "message": "Koordinatentransformation für vertikale Surface (Y-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "X_grid_shape": list(X_grid.shape),
                                "Y_grid_shape": list(Y_grid.shape),
                                "Z_grid_shape": list(Z_grid.shape),
                                "X_min": float(X_grid.min()),
                                "X_max": float(X_grid.max()),
                                "Y_min": float(Y_grid.min()),
                                "Y_max": float(Y_grid.max()),
                                "Z_min": float(Z_grid.min()),
                                "Z_max": float(Z_grid.max()),
                                "X_constant": float(x0)
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
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
                            
                            # 🎯 IDENTISCHES PADDING: Wie bei planaren Flächen
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
            
            # 🎯 ERWEITERE GRID: Einheitliches Padding für alle Surfaces
            # Padding: 1x Resolution in alle Richtungen
            padding_x = resolution * 1.0
            padding_y = resolution * 1.0
            min_x -= padding_x
            max_x += padding_x
            min_y -= padding_y
            max_y += padding_y
            
            # 🎯 EINFACHES, GLEICHMÄSSIGES GRID: Nur gemäß Resolution
            sound_field_x = np.arange(min_x, max_x + resolution, resolution)
            sound_field_y = np.arange(min_y, max_y + resolution, resolution)
            
            # Erstelle 2D-Meshgrid
            X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
            Z_grid = np.zeros_like(X_grid, dtype=float)
            
            # #region agent log - Grid-Regularität prüfen
            try:
                import json, time as _t
                # Prüfe Abstände in X- und Y-Richtung
                x_diffs = np.diff(sound_field_x) if len(sound_field_x) > 1 else np.array([])
                y_diffs = np.diff(sound_field_y) if len(sound_field_y) > 1 else np.array([])
                
                x_is_regular = True
                y_is_regular = True
                x_min_diff = float(np.min(x_diffs)) if x_diffs.size > 0 else 0.0
                x_max_diff = float(np.max(x_diffs)) if x_diffs.size > 0 else 0.0
                y_min_diff = float(np.min(y_diffs)) if y_diffs.size > 0 else 0.0
                y_max_diff = float(np.max(y_diffs)) if y_diffs.size > 0 else 0.0
                
                # Prüfe ob alle Abstände gleich sind (mit Toleranz)
                tolerance = resolution * 0.01  # 1% Toleranz
                if x_diffs.size > 0:
                    x_is_regular = np.allclose(x_diffs, x_diffs[0], atol=tolerance)
                if y_diffs.size > 0:
                    y_is_regular = np.allclose(y_diffs, y_diffs[0], atol=tolerance)
                
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "GRID_REGULARITY",
                        "location": "FlexibleGridGenerator.build_single_surface_grid:grid_regularity",
                        "message": "Grid-Regularität prüfen",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "orientation": str(geometry.orientation),
                            "x_axis_count": int(len(sound_field_x)),
                            "y_axis_count": int(len(sound_field_y)),
                            "x_is_regular": bool(x_is_regular),
                            "y_is_regular": bool(y_is_regular),
                            "x_min_diff": x_min_diff,
                            "x_max_diff": x_max_diff,
                            "y_min_diff": y_min_diff,
                            "y_max_diff": y_max_diff,
                            "x_mean_diff": float(np.mean(x_diffs)) if x_diffs.size > 0 else 0.0,
                            "y_mean_diff": float(np.mean(y_diffs)) if y_diffs.size > 0 else 0.0,
                            "x_std_diff": float(np.std(x_diffs)) if x_diffs.size > 0 else 0.0,
                            "y_std_diff": float(np.std(y_diffs)) if y_diffs.size > 0 else 0.0,
                            "base_resolution": float(resolution),
                            "grid_is_uniform": bool(x_is_regular and y_is_regular),
                            "grid_created_with_np_arange": True
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
        
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
            
            # #region agent log - Maske-Erstellung für vertikale Surface
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H3",
                        "location": "GridBuilder.build_single_surface_grid:vertical_mask_creation",
                        "message": "Maske-Erstellung für vertikale Surface",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "dominant_axis": str(geometry.dominant_axis),
                            "use_yz_wall": bool(use_yz_wall),
                            "X_grid_shape": list(X_grid.shape),
                            "Y_grid_shape": list(Y_grid.shape),
                            "Z_grid_shape": list(Z_grid.shape),
                            "X_grid_min": float(X_grid.min()),
                            "X_grid_max": float(X_grid.max()),
                            "Y_grid_min": float(Y_grid.min()),
                            "Y_grid_max": float(Y_grid.max()),
                            "Z_grid_min": float(Z_grid.min()),
                            "Z_grid_max": float(Z_grid.max()),
                            "x_span": float(x_span),
                            "y_span": float(y_span),
                            "z_span": float(z_span),
                            "xs_min": float(xs.min()),
                            "xs_max": float(xs.max()),
                            "ys_min": float(ys.min()),
                            "ys_max": float(ys.max()),
                            "zs_min": float(zs.min()),
                            "zs_max": float(zs.max())
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
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
                # 🎯 SICHERSTELLEN, DASS POLYGON-ECKEN ERFASST WERDEN (in u,v-Koordinaten)
                surface_mask_strict = self._ensure_vertex_coverage_uv(
                    U_grid, V_grid, polygon_uv, surface_mask_strict
                )
                # 🎯 KEINE DILATATION MEHR: Verwende nur strikte Maske für Berechnung
                # Randpunkte werden jetzt direkt auf Surface-Linie berechnet und als additional_vertices hinzugefügt
                # surface_mask_strict wird für Plot verwendet (nur Punkte innerhalb)
                # surface_mask wird für Berechnung verwendet (nur Punkte innerhalb, keine Dilatation mehr)
                surface_mask = surface_mask_strict.copy()  # Keine Dilatation mehr - Randpunkte werden separat hinzugefügt
                
                # #region agent log - Maske-Ergebnis (Y-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        # Prüfe ob Maske-Punkte innerhalb der Surface-Bounding Box liegen
                        mask_indices = np.where(surface_mask_strict)
                        if len(mask_indices[0]) > 0:
                            mask_y = Y_grid[mask_indices]
                            mask_z = Z_grid[mask_indices]
                            polygon_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                            polygon_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "GridBuilder.build_single_surface_grid:vertical_mask_yz_result",
                                "message": "Maske-Ergebnis für Y-Z-Wand",
                                "data": {
                                    "surface_id": str(geometry.surface_id),
                                    "mask_points_count": int(np.count_nonzero(surface_mask_strict)),
                                    "mask_y_min": float(mask_y.min()) if len(mask_y) > 0 else None,
                                    "mask_y_max": float(mask_y.max()) if len(mask_y) > 0 else None,
                                    "mask_z_min": float(mask_z.min()) if len(mask_z) > 0 else None,
                                    "mask_z_max": float(mask_z.max()) if len(mask_z) > 0 else None,
                                    "polygon_y_min": float(polygon_y.min()),
                                    "polygon_y_max": float(polygon_y.max()),
                                    "polygon_z_min": float(polygon_z.min()),
                                    "polygon_z_max": float(polygon_z.max()),
                                    "U_grid_min": float(U_grid.min()),
                                    "U_grid_max": float(U_grid.max()),
                                    "V_grid_min": float(V_grid.min()),
                                    "V_grid_max": float(V_grid.max())
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # #region agent log - Randpunkte bei Einzelsurface (Y-Z-Wand)
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json, time as _t
                        points_in_strict = int(np.count_nonzero(surface_mask_strict))
                        points_in_dilated = int(np.count_nonzero(surface_mask))
                        points_added = points_in_dilated - points_in_strict
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "single-surface-edge-points",
                            "hypothesisId": "SINGLE_SURFACE_EDGE",
                            "location": "FlexibleGridGenerator.build_single_surface_grid:vertical_yz",
                            "message": "Randpunkte bei Einzelsurface (Y-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "points_in_strict_mask": points_in_strict,
                                "points_in_dilated_mask": points_in_dilated,
                                "edge_points_added": points_added,
                                "has_edge_points": points_added > 0
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
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
                # 🎯 SICHERSTELLEN, DASS POLYGON-ECKEN ERFASST WERDEN (in u,v-Koordinaten)
                surface_mask_strict = self._ensure_vertex_coverage_uv(
                    U_grid, V_grid, polygon_uv, surface_mask_strict
                )
                # 🎯 KEINE DILATATION MEHR: Verwende nur strikte Maske für Berechnung
                # Randpunkte werden jetzt direkt auf Surface-Linie berechnet und als additional_vertices hinzugefügt
                # surface_mask_strict wird für Plot verwendet (nur Punkte innerhalb)
                # surface_mask wird für Berechnung verwendet (nur Punkte innerhalb, keine Dilatation mehr)
                surface_mask = surface_mask_strict.copy()  # Keine Dilatation mehr - Randpunkte werden separat hinzugefügt
                
                # #region agent log - Maske-Ergebnis (X-Z-Wand)
                try:
                    import json, time as _t
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        # Prüfe ob Maske-Punkte innerhalb der Surface-Bounding Box liegen
                        mask_indices = np.where(surface_mask_strict)
                        if len(mask_indices[0]) > 0:
                            mask_x = X_grid[mask_indices]
                            mask_z = Z_grid[mask_indices]
                            polygon_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                            polygon_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "GridBuilder.build_single_surface_grid:vertical_mask_xz_result",
                                "message": "Maske-Ergebnis für X-Z-Wand",
                                "data": {
                                    "surface_id": str(geometry.surface_id),
                                    "mask_points_count": int(np.count_nonzero(surface_mask_strict)),
                                    "mask_x_min": float(mask_x.min()) if len(mask_x) > 0 else None,
                                    "mask_x_max": float(mask_x.max()) if len(mask_x) > 0 else None,
                                    "mask_z_min": float(mask_z.min()) if len(mask_z) > 0 else None,
                                    "mask_z_max": float(mask_z.max()) if len(mask_z) > 0 else None,
                                    "polygon_x_min": float(polygon_x.min()),
                                    "polygon_x_max": float(polygon_x.max()),
                                    "polygon_z_min": float(polygon_z.min()),
                                    "polygon_z_max": float(polygon_z.max()),
                                    "U_grid_min": float(U_grid.min()),
                                    "U_grid_max": float(U_grid.max()),
                                    "V_grid_min": float(V_grid.min()),
                                    "V_grid_max": float(V_grid.max())
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # #region agent log - Randpunkte bei Einzelsurface (X-Z-Wand)
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json, time as _t
                        points_in_strict = int(np.count_nonzero(surface_mask_strict))
                        points_in_dilated = int(np.count_nonzero(surface_mask))
                        points_added = points_in_dilated - points_in_strict
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "single-surface-edge-points",
                            "hypothesisId": "SINGLE_SURFACE_EDGE",
                            "location": "FlexibleGridGenerator.build_single_surface_grid:vertical_xz",
                            "message": "Randpunkte bei Einzelsurface (X-Z-Wand)",
                            "data": {
                                "surface_id": str(geometry.surface_id),
                                "points_in_strict_mask": points_in_strict,
                                "points_in_dilated_mask": points_in_dilated,
                                "edge_points_added": points_added,
                                "has_edge_points": points_added > 0
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
            
            # Z_grid ist bereits korrekt gesetzt (für X-Z-Wand: Z_grid = V_grid, für Y-Z-Wand: Z_grid = V_grid)
            # Keine Z-Interpolation nötig!
            # surface_mask_strict wurde bereits oben erstellt (vor Dilatation)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
        else:
            # PLANARE/SCHRÄGE SURFACES: Normale Maske und Z-Interpolation
            surface_mask_strict = self._create_surface_mask(X_grid, Y_grid, geometry)  # Ursprüngliche Maske
            # 🎯 SICHERSTELLEN, DASS POLYGON-ECKEN ERFASST WERDEN:
            # Ergänze die strikte Maske so, dass jede Polygon-Ecke mindestens
            # einem nächstgelegenen Grid-Punkt zugeordnet ist (falls nicht zu weit entfernt).
            surface_mask_strict = self._ensure_vertex_coverage(
                X_grid, Y_grid, geometry, surface_mask_strict
            )
            # 🎯 KEINE DILATATION MEHR: Verwende nur strikte Maske für Berechnung
            # Randpunkte werden jetzt direkt auf Surface-Linie berechnet und als additional_vertices hinzugefügt
            # surface_mask_strict wird für Plot verwendet (nur Punkte innerhalb)
            # surface_mask wird für Berechnung verwendet (nur Punkte innerhalb, keine Dilatation mehr)
            surface_mask = surface_mask_strict.copy()  # Keine Dilatation mehr - Randpunkte werden separat hinzugefügt
            
            # #region agent log - Randpunkte bei Einzelsurface (planar/sloped)
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json, time as _t
                    points_in_strict = int(np.count_nonzero(surface_mask_strict))
                    points_in_dilated = int(np.count_nonzero(surface_mask))
                    points_added = points_in_dilated - points_in_strict
                    # Prüfe ob Randpunkte auf Surface-Begrenzungslinie oder darüber hinaus liegen
                    edge_points_mask = surface_mask & ~surface_mask_strict
                    edge_x = X_grid[edge_points_mask]
                    edge_y = Y_grid[edge_points_mask]
                    # Prüfe ob Edge-Punkte innerhalb oder außerhalb des Polygons liegen
                    from matplotlib.path import Path
                    poly_path = Path(np.column_stack((
                        [float(p.get("x", 0.0)) for p in geometry.points],
                        [float(p.get("y", 0.0)) for p in geometry.points]
                    )))
                    if edge_x.size > 0:
                        edge_points_2d = np.column_stack((edge_x, edge_y))
                        edge_inside = poly_path.contains_points(edge_points_2d)
                        edge_outside_count = int(np.sum(~edge_inside))
                        edge_on_boundary_count = int(np.sum(edge_inside))
                    else:
                        edge_outside_count = 0
                        edge_on_boundary_count = 0
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "edge-points-analysis",
                        "hypothesisId": "EDGE_POINTS_POSITION",
                        "location": "FlexibleGridGenerator.build_single_surface_grid:planar",
                        "message": "Randpunkte-Analyse Einzelsurface (planar/sloped)",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "points_in_strict_mask": points_in_strict,
                            "points_in_dilated_mask": points_in_dilated,
                            "edge_points_added": points_added,
                            "has_edge_points": points_added > 0,
                            "vertex_coverage_applied": True,
                            "edge_points_outside_polygon": edge_outside_count,
                            "edge_points_inside_polygon": edge_on_boundary_count,
                            "edge_points_total": int(edge_x.size) if edge_x.size > 0 else 0,
                            "edge_method": "dilate_mask_minimal_3x3"
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 DEBUG: Grid-Erweiterung (vor Z-Interpolation, damit total_grid_points verfügbar ist)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json, time as _t
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "grid-analysis",
                        "hypothesisId": "H1",
                        "location": "FlexibleGridGenerator.build_single_surface_grid",
                        "message": "Grid erstellt für Surface",
                        "data": {
                            "surface_id": str(geometry.surface_id),
                            "orientation": str(geometry.orientation),
                            "total_grid_points": int(total_grid_points),
                            "points_in_surface": int(points_in_surface),
                            "points_outside_surface": int(points_outside_surface),
                            "grid_shape": list(X_grid.shape) if hasattr(X_grid, 'shape') else None,
                            "resolution": float(resolution) if resolution else None
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
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
        
        # 🎯 NEU: Berechne additional_vertices (Ecken + Randpunkte) gleichzeitig mit Grid
        # Stelle sicher, dass resolution gesetzt ist
        if resolution is None:
            resolution = self.settings.resolution
        
        # #region agent log - Aufruf _calculate_additional_vertices
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H1",
                    "location": "GridBuilder.build_single_surface_grid:before_calculate_additional_vertices",
                    "message": "Aufruf _calculate_additional_vertices",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "orientation": str(geometry.orientation),
                        "resolution": float(resolution) if resolution is not None else None,
                        "X_grid_shape": list(X_grid.shape),
                        "Y_grid_shape": list(Y_grid.shape),
                        "Z_grid_shape": list(Z_grid.shape),
                        "surface_mask_active": int(np.count_nonzero(surface_mask)) if surface_mask is not None else 0
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        additional_vertices = self._calculate_additional_vertices(
            geometry, X_grid, Y_grid, Z_grid, surface_mask, resolution
        )
        
        # #region agent log - Ergebnis _calculate_additional_vertices
        try:
            import json, time as _t
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H1,H2,H3,H4",
                    "location": "GridBuilder.build_single_surface_grid:after_calculate_additional_vertices",
                    "message": "Ergebnis _calculate_additional_vertices",
                    "data": {
                        "surface_id": str(geometry.surface_id),
                        "additional_vertices_count": int(len(additional_vertices)) if additional_vertices is not None and additional_vertices.size > 0 else 0,
                        "additional_vertices_shape": list(additional_vertices.shape) if additional_vertices is not None and additional_vertices.size > 0 else None,
                        "additional_vertices_empty": bool(additional_vertices is None or additional_vertices.size == 0)
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Für planare/schräge Surfaces: Gebe auch die strikte Maske zurück
        # Für vertikale Surfaces: surface_mask_strict wurde bereits vorher erstellt (vor Dilatation)
        if geometry.orientation in ("planar", "sloped"):
            # surface_mask_strict wurde bereits vorher erstellt
            return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)
        else:
            # 🎯 KORREKTUR: Vertikale Surfaces haben auch eine strikte Maske (vor Dilatation erstellt)
            # surface_mask_strict wurde bereits oben erstellt (vor Dilatation)
            return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)


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
        self._surface_cache_lock = Lock()
    
    def _deduplicate_vertices_and_faces(
        self,
        vertices: np.ndarray,
        faces_flat: np.ndarray,
        resolution: float | None,
        surface_id: str,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entfernt doppelte Vertices (gleiche 3D-Position) aus einem Mesh und
        passt die Faces entsprechend an.
        
        Delegiert an GridBuilder._deduplicate_vertices_and_faces.
        Wird verwendet in generate_per_surface() für Triangulation-Deduplikation.
        """
        return self.builder._deduplicate_vertices_and_faces(
            vertices, faces_flat, resolution, surface_id
        )
    
    @measure_time("FlexibleGridGenerator.generate_per_surface")
    def generate_per_surface(
        self,
        enabled_surfaces: List[Tuple[str, Dict]],
        resolution: Optional[float] = None,
        min_points_per_dimension: int = 6,
        disable_edge_refinement: bool = False
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
        surface_grids: Dict[str, SurfaceGrid] = {}

        # 🚀 Schritt 1: Cache-Prüfung & Liste der zu berechnenden Geometrien aufbauen
        geometries_to_process: List[Tuple[SurfaceGeometry, tuple]] = []
        for geom in geometries:
            cache_key = self._make_surface_cache_key(
                geom=geom,
                resolution=float(resolution),
                min_points=min_points_per_dimension,
            )
            with self._surface_cache_lock:
                cached_grid = self._surface_grid_cache.get(cache_key)
            if cached_grid is not None:
                surface_grids[geom.surface_id] = cached_grid
            else:
                geometries_to_process.append((geom, cache_key))

        # 🚀 Schritt 2: build_single_surface_grid parallel für nicht-gecachte Surfaces ausführen
        grid_results: Dict[str, Tuple[tuple, Any]] = {}
        if geometries_to_process:
            t_parallel_start = time.perf_counter() if PERF_ENABLED else None
            max_workers = int(getattr(self.settings, "spl_parallel_surfaces", 0) or 0)
            if max_workers <= 0:
                max_workers = None  # Default: ThreadPoolExecutor wählt sinnvoll

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sid: Dict[Any, Tuple[str, tuple]] = {}
                for geom, cache_key in geometries_to_process:
                    fut = executor.submit(
                        self.builder.build_single_surface_grid,
                        geometry=geom,
                        resolution=resolution,
                        min_points_per_dimension=min_points_per_dimension,
                        disable_edge_refinement=disable_edge_refinement,
                    )
                    future_to_sid[fut] = (geom.surface_id, cache_key)

                for fut in as_completed(future_to_sid):
                    surface_id, cache_key = future_to_sid[fut]
                    try:
                        result = fut.result()
                        grid_results[surface_id] = (cache_key, result)
                    except Exception as e:
                        error_type = type(e).__name__
                        print(f"⚠️  [FlexibleGridGenerator] Surface '{surface_id}' übersprungen (parallel, {error_type}): {e}")

            if PERF_ENABLED and t_parallel_start is not None:
                duration_ms = (time.perf_counter() - t_parallel_start) * 1000.0
                print(
                    f"[PERF] FlexibleGridGenerator.generate_per_surface.parallel_build: "
                    f"{duration_ms:.2f} ms [n_surfaces={len(geometries_to_process)}]"
                )
        
        # 🚀 Schritt 3: Ergebnisse weiterverarbeiten (Plane-Korrektur, Triangulation, Cache füllen)
        for geom in geometries:
            # Wenn bereits aus dem Cache kam, ist alles erledigt
            if geom.surface_id in surface_grids:
                continue

            entry = grid_results.get(geom.surface_id)
            if not entry:
                # Wurde übersprungen oder ist fehlgeschlagen
                continue
            cache_key, result = entry
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
                        min_points_per_dimension=min_points_per_dimension,
                        disable_edge_refinement=disable_edge_refinement
                    )
                if len(result) == 8:
                    # Version: Mit strikter Maske + additional_vertices
                    (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices) = result
                    
                    # #region agent log - additional_vertices aus result
                    try:
                        import json, time as _t
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H5",
                                "location": "FlexibleGridGenerator.generate_per_surface:after_unpack_result",
                                "message": "additional_vertices aus result",
                                "data": {
                                    "surface_id": str(geom.surface_id),
                                    "additional_vertices_count": int(len(additional_vertices)) if additional_vertices is not None and additional_vertices.size > 0 else 0,
                                    "additional_vertices_shape": list(additional_vertices.shape) if additional_vertices is not None and additional_vertices.size > 0 else None,
                                    "additional_vertices_is_array": isinstance(additional_vertices, np.ndarray)
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                elif len(result) == 7:
                    # Version: Mit strikter Maske (ohne additional_vertices - Fallback)
                    (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict) = result
                    additional_vertices = np.array([], dtype=float).reshape(0, 3)  # Leeres Array als Fallback
                else:
                    # Alte Version: Ohne strikte Maske (Fallback)
                    (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = result
                    surface_mask_strict = surface_mask  # Fallback: identisch
                    additional_vertices = np.array([], dtype=float).reshape(0, 3)  # Leeres Array als Fallback
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
            # 🎯 NEU: Verwende Delaunay-Triangulation für konsistente, überschneidungsfreie Triangulation
            triangulated_vertices = None
            triangulated_faces = None
            triangulated_success = False
            vertex_source_indices: Optional[np.ndarray] = None
            # 🎯 FIX: additional_vertices wurde bereits in build_single_surface_grid berechnet
            # Nichts überschreiben - verwende die bereits berechneten additional_vertices
            
            # 🎯 FIX: Definiere additional_vertices_array außerhalb des try-Blocks, damit es immer verfügbar ist
            # Konvertiere zu numpy array falls noch nicht geschehen
            if not isinstance(additional_vertices, np.ndarray):
                additional_vertices_array = np.array(additional_vertices, dtype=float) if additional_vertices is not None and hasattr(additional_vertices, '__len__') and len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
            else:
                additional_vertices_array = additional_vertices
            
            try:
                # Verwende Grid-Punkte innerhalb der Surface-Maske als Vertices
                ny, nx = X_grid.shape
                # 🎯 KEINE ERWEITERTE MASKE MEHR: Verwende nur strikte Maske
                # Randpunkte werden als additional_vertices hinzugefügt, nicht über Dilatation
                mask_flat = surface_mask_strict.ravel()  # Nur strikte Maske (keine Dilatation mehr)
                mask_strict_flat = surface_mask_strict.ravel()  # Strikte Maske (identisch mit mask_flat)
                
                if np.any(mask_flat):
                    # Erstelle Vertex-Koordinaten aus allen Grid-Punkten (auch inaktive für konsistente Indizes)
                    # Wir brauchen alle Punkte für die strukturierte Triangulation
                    all_vertices = np.column_stack([
                        X_grid.ravel(),  # X-Koordinaten
                        Y_grid.ravel(),  # Y-Koordinaten
                        Z_grid.ravel()   # Z-Koordinaten
                    ])  # Shape: (ny * nx, 3)
                    
                    # #region agent log - Triangulation Input (vor additional_vertices)
                    try:
                        import json, time as _t
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H5",
                                "location": "FlexibleGridGenerator.generate_per_surface:triangulation_input",
                                "message": "Triangulation Input-Koordinaten",
                                "data": {
                                    "surface_id": str(geom.surface_id),
                                    "orientation": str(geom.orientation),
                                    "dominant_axis": str(geom.dominant_axis) if hasattr(geom, 'dominant_axis') and geom.dominant_axis else None,
                                    "all_vertices_shape": list(all_vertices.shape),
                                    "all_vertices_x_min": float(all_vertices[:, 0].min()),
                                    "all_vertices_x_max": float(all_vertices[:, 0].max()),
                                    "all_vertices_y_min": float(all_vertices[:, 1].min()),
                                    "all_vertices_y_max": float(all_vertices[:, 1].max()),
                                    "all_vertices_z_min": float(all_vertices[:, 2].min()),
                                    "all_vertices_z_max": float(all_vertices[:, 2].max()),
                                    "X_grid_min": float(X_grid.min()),
                                    "X_grid_max": float(X_grid.max()),
                                    "Y_grid_min": float(Y_grid.min()),
                                    "Y_grid_max": float(Y_grid.max()),
                                    "Z_grid_min": float(Z_grid.min()),
                                    "Z_grid_max": float(Z_grid.max()),
                                    "mask_points_count": int(np.count_nonzero(mask_flat))
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # 🎯 HINWEIS: additional_vertices werden jetzt bereits in build_single_surface_grid berechnet
                    # Die folgende Logik wurde nach _calculate_additional_vertices verschoben
                    # und wird hier nicht mehr benötigt, da additional_vertices bereits verfügbar sind
                    
                    # 🎯 VERWENDE BEREITS BEREICHNETE additional_vertices AUS build_single_surface_grid
                    # additional_vertices wurde bereits oben aus result extrahiert
                    
                    # 🎯 ALTE LOGIK ENTFERNT: Die Berechnung von additional_vertices erfolgt jetzt in build_single_surface_grid
                    # Dies vermeidet Duplikation und stellt sicher, dass alle Punkte gleichzeitig mit dem Grid berechnet werden
                    
                    # 🎯 FIX: additional_vertices wurde bereits in build_single_surface_grid berechnet
                    # Der folgende Code-Block wurde entfernt, da er versucht, additional_vertices.append() aufzurufen,
                    # aber additional_vertices ist jetzt ein numpy.ndarray, nicht eine Liste.
                    # Die gesamte Logik für Ecken- und Randpunkte-Berechnung ist jetzt in _calculate_additional_vertices.
                    
                    # 🎯 FIX: additional_vertices wurden bereits in build_single_surface_grid berechnet
                    # Verwende diese direkt - KEINE Neuberechnung mehr!
                    # Der gesamte Block zur Neuberechnung wurde entfernt, da er versuchte, .append() auf einem numpy.ndarray aufzurufen.
                    
                    # 🎯 FIX: additional_vertices_array wurde bereits außerhalb des try-Blocks definiert
                    # Keine redundante Definition hier nötig - verwende das bereits definierte additional_vertices_array
                    
                    # boundary_points_added wird nicht mehr verwendet - nur für Kompatibilität
                    boundary_points_added = 0
                    
                    # Aktualisiere all_vertices mit allen additional_vertices (Ecken + Randpunkte)
                    if additional_vertices_array.size > 0:
                        all_vertices = np.vstack([all_vertices, additional_vertices_array])
                        
                        # #region agent log - Additional Vertices hinzugefügt (inkl. Randpunkte)
                        try:
                            import json, time as _t
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "ADDITIONAL_VERTICES_ADDED",
                                    "location": "FlexibleGridGenerator.generate_per_surface:additional_vertices_added",
                                    "message": "Additional Vertices hinzugefügt (inkl. Randpunkte)",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "additional_vertices_count": int(additional_vertices_array.size // 3) if additional_vertices_array.size > 0 else 0,
                                        "additional_vertices_shape": list(additional_vertices_array.shape) if additional_vertices_array.size > 0 else None
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                    
                    # Speichere Offset für zusätzliche Vertices (alle Grid-Punkte kommen zuerst)
                    base_vertex_count = X_grid.size
                    additional_vertex_start_idx = base_vertex_count
                    
                    # 🎯 NEU: DELAUNAY-TRIANGULATION für alle gültigen Punkte
                    # 🎯 DELAUNAY-TRIANGULATION: Verwende nur noch Delaunay für konsistente, überschneidungsfreie Triangulation
                    faces_list = []  # Initialisiere faces_list
                    use_delaunay_triangulation = True  # Immer Delaunay verwenden
                    
                    if not HAS_SCIPY:
                        raise RuntimeError(f"scipy ist erforderlich für Delaunay-Triangulation. Surface: {geom.surface_id}")
                    
                    # Additional_vertices (Ecken + Randpunkte) werden später hinzugefügt (nach Randpunkten)
                    
                    # 🎯 NEUE RANDPUNKT-LOGIK: Direkte Berechnung von Punkten auf Surface-Linie
                    # Erstelle gleichmäßig verteilte Punkte direkt auf Polygon-Kanten
                    # Diese Punkte werden als additional_vertices hinzugefügt und für Triangulation verwendet
                    boundary_edge_points = []  # Liste von (x, y, z) Tupeln für Randpunkte
                    
                    # Berechne Randpunkte direkt auf Surface-Linie
                    surface_points_for_boundary = geom.points or []
                    if len(surface_points_for_boundary) >= 3:
                        point_spacing = actual_resolution  # Abstand zwischen Randpunkten (Größenordnung: resolution)
                        
                        if geom.orientation == "vertical":
                            # VERTIKALE FLÄCHEN: Berechne in (u,v)-Koordinaten
                            xs = np.array([p.get("x", 0.0) for p in surface_points_for_boundary], dtype=float)
                            ys = np.array([p.get("y", 0.0) for p in surface_points_for_boundary], dtype=float)
                            zs = np.array([p.get("z", 0.0) for p in surface_points_for_boundary], dtype=float)
                            
                            # Bestimme (u,v)-Koordinaten basierend auf dominant_axis
                            # 🎯 KONSISTENT MIT GRID-ERSTELLUNG: Verwende gleiche Logik für Schrägheitsprüfung
                            x_span = float(np.ptp(xs))
                            y_span = float(np.ptp(ys))
                            z_span = float(np.ptp(zs))
                            eps_line = 1e-6
                            
                            is_slanted_wall = False
                            if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                if geom.dominant_axis == "yz":
                                    polygon_u = ys
                                    polygon_v = zs
                                    is_xz_wall = False
                                    # Prüfe ob schräg: X variiert signifikant (>10% von max(y_span, z_span))
                                    max_main_span = max(y_span, z_span) if max(y_span, z_span) > 1e-6 else 1.0
                                    x_variation_ratio = x_span / max_main_span if max_main_span > 1e-6 else 0.0
                                    is_slanted_wall = (x_variation_ratio > 0.1 and z_span > 1e-3)
                                elif geom.dominant_axis == "xz":
                                    polygon_u = xs
                                    polygon_v = zs
                                    is_xz_wall = True
                                    # Prüfe ob schräg: Y variiert signifikant (>10% von max(x_span, z_span))
                                    max_main_span = max(x_span, z_span) if max(x_span, z_span) > 1e-6 else 1.0
                                    y_variation_ratio = y_span / max_main_span if max_main_span > 1e-6 else 0.0
                                    is_slanted_wall = (y_variation_ratio > 0.1 and z_span > 1e-3)
                                else:
                                    polygon_u = xs
                                    polygon_v = zs
                                    is_xz_wall = True
                            else:
                                # Fallback: X-Z-Wand
                                polygon_u = xs
                                polygon_v = zs
                                is_xz_wall = True
                            
                            # 🎯 FÜR SCHRÄGE WÄNDE: Bereite Interpolation vor
                            if is_slanted_wall:
                                from scipy.interpolate import griddata
                                if is_xz_wall:
                                    # X-Z-Wand schräg: Interpoliere Y aus (x,z)-Koordinaten
                                    points_surface = np.column_stack([xs, zs])
                                else:
                                    # Y-Z-Wand schräg: Interpoliere X aus (y,z)-Koordinaten
                                    points_surface = np.column_stack([ys, zs])
                            
                            # Generiere Punkte entlang jeder Kante
                            n_vertices = len(polygon_u)
                            for i in range(n_vertices):
                                u1, v1 = polygon_u[i], polygon_v[i]
                                u2, v2 = polygon_u[(i + 1) % n_vertices], polygon_v[(i + 1) % n_vertices]
                                
                                edge_len = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
                                if edge_len < 1e-9:
                                        continue
                                    
                                # Generiere Punkte mit Abstand point_spacing
                                n_points_on_edge = max(1, int(np.ceil(edge_len / point_spacing)))
                                for j in range(1, n_points_on_edge):  # Start bei 1, Ende bei n_points_on_edge-1 (Ecken werden separat behandelt)
                                    t = j / n_points_on_edge
                                    u = u1 + t * (u2 - u1)
                                    v = v1 + t * (v2 - v1)
                                    
                                    # 🎯 FIX: Transformiere zurück zu (x,y,z) - EXAKT auf Surfacefläche
                                    if is_xz_wall:
                                        x, z = u, v
                                        if is_slanted_wall:
                                            # Schräge Wand: Y interpoliert aus (x,z)
                                            y = griddata(
                                                points_surface, ys,
                                                np.array([[x, z]]),
                                                method='linear', fill_value=float(np.mean(ys))
                                            )[0]
                                        else:
                                            # Konstante Wand: Y = Mittelwert
                                            y = float(np.mean(ys))
                                    else:
                                        y, z = u, v
                                        if is_slanted_wall:
                                            # Schräge Wand: X interpoliert aus (y,z)
                                            x = griddata(
                                                points_surface, xs,
                                                np.array([[y, z]]),
                                                method='linear', fill_value=float(np.mean(xs))
                                            )[0]
                                        else:
                                            # Konstante Wand: X = Mittelwert
                                            x = float(np.mean(xs))
                                    
                                    boundary_edge_points.append([float(x), float(y), float(z)])
                            
                            # #region agent log - Vertikale Flächen: Randpunkte generiert mit Koordinatenprüfung
                            try:
                                import json, time as _t
                                # Prüfe ob Randpunkte auf Surfacefläche liegen
                                if len(boundary_edge_points) > 0:
                                    boundary_array = np.array(boundary_edge_points)
                                    sample_point = boundary_array[0] if len(boundary_array) > 0 else None
                                    if sample_point is not None:
                                        # Prüfe ob Punkt auf Surfacefläche liegt (für X-Z-Wand: Y sollte mit Original-Punkten übereinstimmen)
                                        if is_xz_wall:
                                            # Für X-Z-Wand: Prüfe ob Y-Wert korrekt ist
                                            # Wenn schräg: Y sollte interpoliert sein
                                            # Wenn konstant: Y sollte Mittelwert sein
                                            expected_y = float(np.mean(ys)) if not is_slanted_wall else None
                                            actual_y = float(sample_point[1])
                                        else:
                                            # Für Y-Z-Wand: Prüfe ob X-Wert korrekt ist
                                            expected_x = float(np.mean(xs)) if not is_slanted_wall else None
                                            actual_x = float(sample_point[0])
                                
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "BOUNDARY_POINTS_COORDS",
                                        "location": "FlexibleGridGenerator.generate_per_surface:vertical_boundary_points_coords",
                                        "message": "Randpunkte-Koordinaten für vertikale Flächen",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "boundary_points_count": len(boundary_edge_points),
                                            "is_xz_wall": bool(is_xz_wall),
                                            "is_slanted_wall": bool(is_slanted_wall),
                                            "dominant_axis": str(geom.dominant_axis) if hasattr(geom, 'dominant_axis') and geom.dominant_axis else None,
                                            "x_span": float(x_span),
                                            "y_span": float(y_span),
                                            "z_span": float(z_span),
                                            "xs_mean": float(np.mean(xs)),
                                            "ys_mean": float(np.mean(ys)),
                                            "sample_boundary_point": boundary_edge_points[0] if len(boundary_edge_points) > 0 else None,
                                            "original_points_x_range": [float(np.min(xs)), float(np.max(xs))],
                                            "original_points_y_range": [float(np.min(ys)), float(np.max(ys))],
                                            "original_points_z_range": [float(np.min(zs)), float(np.max(zs))]
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            
                            # #region agent log - Vertikale Flächen: Randpunkte generiert
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "VERTICAL_BOUNDARY_POINTS",
                                        "location": "FlexibleGridGenerator.generate_per_surface:vertical_boundary_points",
                                        "message": "Vertikale Flächen: Randpunkte generiert",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "boundary_points_generated": len(boundary_edge_points),
                                            "point_spacing": float(point_spacing),
                                            "resolution": float(actual_resolution),
                                            "dominant_axis": str(geom.dominant_axis) if hasattr(geom, 'dominant_axis') else None,
                                            "is_xz_wall": is_xz_wall,
                                            "n_vertices": n_vertices,
                                            "polygon_u_range": [float(np.min(polygon_u)), float(np.max(polygon_u))] if len(polygon_u) > 0 else None,
                                            "polygon_v_range": [float(np.min(polygon_v)), float(np.max(polygon_v))] if len(polygon_v) > 0 else None
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                        else:
                            # PLANARE/SCHRÄGE FLÄCHEN: Berechne in (x,y)-Koordinaten
                            polygon_x = np.array([p.get("x", 0.0) for p in surface_points_for_boundary], dtype=float)
                            polygon_y = np.array([p.get("y", 0.0) for p in surface_points_for_boundary], dtype=float)
                            
                            # Generiere Punkte entlang jeder Kante
                            n_vertices = len(polygon_x)
                            for i in range(n_vertices):
                                x1, y1 = polygon_x[i], polygon_y[i]
                                x2, y2 = polygon_x[(i + 1) % n_vertices], polygon_y[(i + 1) % n_vertices]
                                
                                edge_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                if edge_len < 1e-9:
                                    continue
                                
                                # Generiere Punkte mit Abstand point_spacing
                                n_points_on_edge = max(1, int(np.ceil(edge_len / point_spacing)))
                                for j in range(1, n_points_on_edge):  # Start bei 1, Ende bei n_points_on_edge-1 (Ecken werden separat behandelt)
                                    t = j / n_points_on_edge
                                    x = x1 + t * (x2 - x1)
                                    y = y1 + t * (y2 - y1)
                                    
                                    # Berechne Z-Koordinate
                                    z = 0.0
                                    if geom.plane_model:
                                        try:
                                            z_new = _evaluate_plane_on_grid(
                                                geom.plane_model,
                                                np.array([[x]]),
                                                np.array([[y]])
                                            )
                                            if z_new is not None and z_new.size > 0:
                                                z = float(z_new.flat[0])
                                        except Exception:
                                            pass
                                    
                                    boundary_edge_points.append([float(x), float(y), float(z)])
                    
                    # 🎯 HINWEIS: additional_vertices werden jetzt bereits in build_single_surface_grid berechnet
                    # Die Duplikat-Prüfung und Zusammenführung erfolgt jetzt in _calculate_additional_vertices
                    # boundary_points_added wird nicht mehr hier berechnet
                    boundary_points_added = 0  # Wird nicht mehr verwendet - nur für Kompatibilität
                    
                    # #region agent log - additional_vertices vor zweiter Zuweisung
                    try:
                        import json, time as _t
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "OVERWRITE_TRACK",
                                "location": "FlexibleGridGenerator.generate_per_surface:before_second_array_assignment",
                                "message": "additional_vertices vor zweiter Zuweisung",
                                "data": {
                                    "surface_id": str(geom.surface_id),
                                    "additional_vertices_is_array": isinstance(additional_vertices, np.ndarray),
                                    "additional_vertices_count": int(len(additional_vertices)) if hasattr(additional_vertices, '__len__') else 0,
                                    "additional_vertices_size": int(additional_vertices.size) if isinstance(additional_vertices, np.ndarray) else None,
                                    "additional_vertices_shape": list(additional_vertices.shape) if isinstance(additional_vertices, np.ndarray) and additional_vertices.size > 0 else None
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # 🎯 FIX: Entferne duplizierte Zuweisung - additional_vertices_array wurde bereits oben zugewiesen!
                    # Diese Zuweisung ist redundant und wird entfernt
                    
                    # Aktualisiere all_vertices mit allen additional_vertices (Ecken + Randpunkte)
                    if len(additional_vertices_array) > 0:
                        all_vertices = np.vstack([all_vertices, additional_vertices_array])
                        
                        # #region agent log - Additional Vertices hinzugefügt (inkl. Randpunkte)
                        try:
                            import json, time as _t
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "ADDITIONAL_VERTICES_ADDED",
                                    "location": "FlexibleGridGenerator.generate_per_surface:additional_vertices_added",
                                    "message": "Additional Vertices hinzugefügt (inkl. Randpunkte)",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "orientation": str(geom.orientation),
                                        "additional_vertices_count": len(additional_vertices),
                                        "boundary_points_added": boundary_points_added,
                                        "total_vertices_before": int(X_grid.size),
                                        "total_vertices_after": int(len(all_vertices))
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                    
                    # #region agent log - Neue Randpunktlogik: Prüfungen
                    try:
                        import json, time as _t
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "NEW_BOUNDARY_POINTS",
                                "location": "FlexibleGridGenerator.generate_per_surface:new_boundary_points",
                                "message": "Neue Randpunktlogik: Punkte auf Surface-Linie",
                                "data": {
                                    "surface_id": str(geom.surface_id),
                                    "boundary_edge_points_generated": len(boundary_edge_points),
                                    "boundary_points_added": boundary_points_added,
                                    "point_spacing": float(point_spacing) if len(surface_points_for_boundary) >= 3 else None,
                                    "resolution": float(actual_resolution),
                                    "orientation": str(geom.orientation) if hasattr(geom, 'orientation') else None
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # Speichere Offset für zusätzliche Vertices (alle Grid-Punkte kommen zuerst)
                    base_vertex_count = X_grid.size
                    additional_vertex_start_idx = base_vertex_count
                    
                    # #region agent log - additional_vertices vor dritter Zuweisung
                    try:
                        import json, time as _t
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "OVERWRITE_TRACK",
                                "location": "FlexibleGridGenerator.generate_per_surface:before_third_array_assignment",
                                "message": "additional_vertices vor dritter Zuweisung",
                                "data": {
                                    "surface_id": str(geom.surface_id),
                                    "additional_vertices_is_array": isinstance(additional_vertices, np.ndarray),
                                    "additional_vertices_count": int(len(additional_vertices)) if hasattr(additional_vertices, '__len__') else 0,
                                    "additional_vertices_size": int(additional_vertices.size) if isinstance(additional_vertices, np.ndarray) else None,
                                    "additional_vertices_shape": list(additional_vertices.shape) if isinstance(additional_vertices, np.ndarray) and additional_vertices.size > 0 else None,
                                    "additional_vertices_array_exists": 'additional_vertices_array' in locals(),
                                    "additional_vertices_array_count": int(len(additional_vertices_array)) if 'additional_vertices_array' in locals() and hasattr(additional_vertices_array, '__len__') else None
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # 🎯 FIX: Entferne duplizierte Zuweisung - additional_vertices_array wurde bereits oben zugewiesen!
                    # Diese Zuweisung ist redundant und wird entfernt
                    
                    # 🎯 NEU: DELAUNAY-TRIANGULATION für alle gültigen Punkte
                    # 🎯 DELAUNAY-TRIANGULATION: Verwende nur noch Delaunay für konsistente, überschneidungsfreie Triangulation
                    faces_list = []  # Initialisiere faces_list
                    use_delaunay_triangulation = True  # Immer Delaunay verwenden
                    
                    if not HAS_SCIPY:
                        raise RuntimeError(f"scipy ist erforderlich für Delaunay-Triangulation. Surface: {geom.surface_id}")
                    
                    # Sammle alle gültigen Punkte (Grid-Punkte in Maske + additional_vertices)
                    if use_delaunay_triangulation:  # Immer Delaunay verwenden
                        try:
                            # Sammle alle gültigen Punkte
                            valid_vertices_list = []
                            valid_vertex_indices = []  # Original-Indizes in all_vertices
                            
                            # Füge aktive Grid-Punkte hinzu
                            active_mask_indices = np.where(mask_flat)[0]
                            active_strict_mask_indices = np.where(mask_strict_flat)[0]
                            # #region agent log - Mask analysis H1, H2, H3
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H1_H2_H3",
                                        "location": "FlexibleGridGenerator.generate_per_surface:mask_analysis",
                                        "message": "Mask-Analyse für Triangulation",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "mask_flat_count": int(len(active_mask_indices)),
                                            "mask_strict_flat_count": int(len(active_strict_mask_indices)),
                                            "extended_points_count": int(len(active_mask_indices) - len(active_strict_mask_indices)),
                                            "total_grid_points": int(X_grid.size)
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            grid_vertices_count = 0
                            for idx in active_mask_indices:
                                if idx < len(all_vertices):
                                    valid_vertices_list.append(all_vertices[idx])
                                    valid_vertex_indices.append(idx)
                                    grid_vertices_count += 1
                            
                            # Füge additional_vertices hinzu (bereits in all_vertices)
                            boundary_vertices_count = 0
                            if len(additional_vertices_array) > 0:
                                for i in range(len(additional_vertices_array)):
                                    idx = additional_vertex_start_idx + i
                                    if idx < len(all_vertices):
                                        valid_vertices_list.append(all_vertices[idx])
                                        valid_vertex_indices.append(idx)
                                        boundary_vertices_count += 1
                            
                            # #region agent log - Triangulation input points
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "TRIANGULATION_INPUT",
                                        "location": "FlexibleGridGenerator.generate_per_surface:triangulation_input",
                                        "message": "Triangulation Input: Grid + Boundary points",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "grid_vertices_count": grid_vertices_count,
                                            "boundary_vertices_count": boundary_vertices_count,
                                            "total_valid_vertices": len(valid_vertices_list),
                                            "total_all_vertices": len(all_vertices),
                                            "additional_vertices_count": len(additional_vertices_array)
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            
                            if len(valid_vertices_list) >= 3:
                                valid_vertices_array = np.array(valid_vertices_list, dtype=float)
                                
                                # Bestimme Projektionsebene basierend auf Orientierung
                                if geom.orientation == "vertical":
                                    # Vertikale Flächen: Projiziere auf (u,v)-Ebene
                                    if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                        if geom.dominant_axis == "xz":
                                            # X-Z-Wand: u = x, v = z
                                            proj_2d = valid_vertices_array[:, [0, 2]]  # x, z
                                        elif geom.dominant_axis == "yz":
                                            # Y-Z-Wand: u = y, v = z
                                            proj_2d = valid_vertices_array[:, [1, 2]]  # y, z
                                        else:
                                            # Fallback: verwende x, y
                                            proj_2d = valid_vertices_array[:, [0, 1]]
                                    else:
                                        # Fallback: verwende x, y
                                        proj_2d = valid_vertices_array[:, [0, 1]]
                                else:
                                    # Planare/schräge Flächen: Projiziere auf (x,y)-Ebene
                                    proj_2d = valid_vertices_array[:, [0, 1]]  # x, y
                                
                                # Delaunay-Triangulation in 2D
                                tri = Delaunay(proj_2d)
                                triangles_delaunay = tri.simplices  # Nx3 Array von Indizes
                                
                                # Filtere Dreiecke, die außerhalb der Surface-Maske liegen
                                # Prüfe ob Schwerpunkt jedes Dreiecks innerhalb des Polygons liegt
                                valid_triangles = []
                                surface_points = geom.points or []
                                
                                if len(surface_points) >= 3:
                                    try:
                                        from matplotlib.path import Path
                                        
                                        # Erstelle Polygon-Pfad für Punkt-im-Polygon-Prüfung
                                        if geom.orientation == "vertical":
                                            # Vertikale Flächen: Polygon in (u,v)
                                            if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                                if geom.dominant_axis == "xz":
                                                    polygon_2d = np.array([[p.get("x", 0.0), p.get("z", 0.0)] for p in surface_points], dtype=float)
                                                elif geom.dominant_axis == "yz":
                                                    polygon_2d = np.array([[p.get("y", 0.0), p.get("z", 0.0)] for p in surface_points], dtype=float)
                                                else:
                                                    polygon_2d = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in surface_points], dtype=float)
                                            else:
                                                polygon_2d = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in surface_points], dtype=float)
                                        else:
                                            # Planare/schräge Flächen: Polygon in (x,y)
                                            polygon_2d = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in surface_points], dtype=float)
                                        
                                        polygon_path = Path(polygon_2d)
                                        
                                        # Prüfe jeden Dreieck-Schwerpunkt
                                        triangles_filtered_out = 0
                                        triangles_with_extended_vertices = 0
                                        triangles_with_strict_vertices = 0
                                        for tri_idx in triangles_delaunay:
                                                # Konvertiere Indizes: Delaunay-Indizes → valid_vertex_indices → all_vertices-Indizes
                                                v1_old = valid_vertex_indices[tri_idx[0]]
                                                v2_old = valid_vertex_indices[tri_idx[1]]
                                                v3_old = valid_vertex_indices[tri_idx[2]]
                                                
                                                # Prüfe ob Vertices in strikter oder erweiterter Maske sind
                                                v1_in_strict = (v1_old < len(mask_strict_flat) and mask_strict_flat[v1_old]) if v1_old < len(mask_strict_flat) else False
                                                v2_in_strict = (v2_old < len(mask_strict_flat) and mask_strict_flat[v2_old]) if v2_old < len(mask_strict_flat) else False
                                                v3_in_strict = (v3_old < len(mask_strict_flat) and mask_strict_flat[v3_old]) if v3_old < len(mask_strict_flat) else False
                                                has_strict_vertex = v1_in_strict or v2_in_strict or v3_in_strict
                                                
                                                v1_in_extended = v1_old < len(mask_flat) and mask_flat[v1_old]
                                                v2_in_extended = v2_old < len(mask_flat) and mask_flat[v2_old]
                                                v3_in_extended = v3_old < len(mask_flat) and mask_flat[v3_old]
                                                has_extended_vertex = v1_in_extended or v2_in_extended or v3_in_extended
                                                
                                                # Schwerpunkt des Dreiecks in 2D
                                                centroid_2d = proj_2d[tri_idx].mean(axis=0)
                                                
                                                # Prüfe ob Schwerpunkt im Polygon liegt
                                                centroid_in_polygon = polygon_path.contains_point(centroid_2d)
                                                
                                                # #region agent log - Triangle filtering H1, H2, H3
                                                if not centroid_in_polygon:
                                                    triangles_filtered_out += 1
                                                    if has_extended_vertex and not has_strict_vertex:
                                                        triangles_with_extended_vertices += 1
                                                    elif has_strict_vertex:
                                                        triangles_with_strict_vertices += 1
                                                # #endregion
                                                
                                                # 🎯 FIX: Akzeptiere Dreiecke wenn:
                                                # 1. Schwerpunkt im Polygon liegt ODER
                                                # 2. Mindestens ein Vertex in strikter Maske liegt (für Rand-Dreiecke) ODER
                                                # 3. Alle drei Vertices in erweiterter Maske liegen (für Rand-Dreiecke mit erweiterten Punkten)
                                                all_vertices_in_extended = v1_in_extended and v2_in_extended and v3_in_extended
                                                if centroid_in_polygon or has_strict_vertex or all_vertices_in_extended:
                                                    valid_triangles.append((v1_old, v2_old, v3_old))
                                        
                                        # #region agent log - Triangle filtering summary H1, H2, H3
                                        try:
                                            import json, time as _t
                                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                                f.write(json.dumps({
                                                    "sessionId": "debug-session",
                                                    "runId": "run1",
                                                    "hypothesisId": "H1_H2_H3",
                                                    "location": "FlexibleGridGenerator.generate_per_surface:triangle_filtering",
                                                    "message": "Dreieck-Filterung Zusammenfassung",
                                                    "data": {
                                                        "surface_id": str(geom.surface_id),
                                                        "total_triangles": int(len(triangles_delaunay)),
                                                        "valid_triangles": int(len(valid_triangles)),
                                                        "triangles_filtered_out": triangles_filtered_out,
                                                        "triangles_with_extended_vertices": triangles_with_extended_vertices,
                                                        "triangles_with_strict_vertices": triangles_with_strict_vertices
                                                    },
                                                    "timestamp": int(_t.time() * 1000)
                                                }) + "\n")
                                        except Exception:
                                            pass
                                        # #endregion
                                    except Exception:
                                        # Fallback: Verwende alle Delaunay-Dreiecke
                                        for tri_idx in triangles_delaunay:
                                            v1_old = valid_vertex_indices[tri_idx[0]]
                                            v2_old = valid_vertex_indices[tri_idx[1]]
                                            v3_old = valid_vertex_indices[tri_idx[2]]
                                            valid_triangles.append((v1_old, v2_old, v3_old))
                                else:
                                    # Kein Polygon verfügbar: Verwende alle Delaunay-Dreiecke
                                    for tri_idx in triangles_delaunay:
                                        v1_old = valid_vertex_indices[tri_idx[0]]
                                        v2_old = valid_vertex_indices[tri_idx[1]]
                                        v3_old = valid_vertex_indices[tri_idx[2]]
                                        valid_triangles.append((v1_old, v2_old, v3_old))
                                
                                # Konvertiere zu PyVista-Format: [3, v1, v2, v3, 3, v4, v5, v6, ...]
                                if len(valid_triangles) > 0:
                                    for v1, v2, v3 in valid_triangles:
                                        faces_list.extend([3, int(v1), int(v2), int(v3)])
                                    
                                    # #region agent log - Delaunay-Triangulation erfolgreich
                                    try:
                                        import json, time as _t
                                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "DELAUNAY_TRIANGULATION_SUCCESS",
                                                "location": "FlexibleGridGenerator.generate_per_surface:delaunay_triangulation",
                                                "message": "Delaunay-Triangulation erfolgreich",
                                                "data": {
                                                    "surface_id": str(geom.surface_id),
                                                    "valid_vertices_count": len(valid_vertices_list),
                                                    "triangles_count": len(valid_triangles),
                                                    "orientation": str(geom.orientation)
                                                },
                                                "timestamp": int(_t.time() * 1000)
                                            }) + "\n")
                                    except Exception:
                                        pass
                                    # #endregion
                        except Exception as e:
                            # Bei Fehler: Logge und werfe Exception
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "DELAUNAY_TRIANGULATION_FAILED",
                                        "location": "FlexibleGridGenerator.generate_per_surface:delaunay_triangulation",
                                        "message": f"Delaunay-Triangulation fehlgeschlagen: {str(e)}",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "error": str(e)
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                    
                    # 🎯 ALTE PROJEKTIONSLOGIK ENTFERNT - Ersetzt durch direkte Randpunkt-Berechnung oben (vor Triangulation)
                    # Die alte Projektionslogik (Dilatation + Projektion von boundary_indices) wurde entfernt
                    # und durch direkte Berechnung von Punkten auf Surface-Linie ersetzt (vor Triangulation)
                    # 🎯 Prüfe ob Triangulation erfolgreich war
                    if len(faces_list) == 0:
                        raise RuntimeError(f"Keine Faces für Surface {geom.surface_id} generiert. Delaunay-Triangulation hat keine Dreiecke erstellt.")
                    
                    # 🎯 Initialisiere vertex_indices_in_faces_set aus faces_list (für spätere Verwendung)
                    # Sammle alle eindeutigen Vertex-Indizes, die in Faces verwendet werden
                    vertex_indices_in_faces_temp = []
                    for i in range(0, len(faces_list), 4):
                        if i + 3 < len(faces_list):
                            n_verts = faces_list[i]
                            if n_verts == 3:
                                vertex_indices_in_faces_temp.extend(faces_list[i+1:i+4])
                    vertex_indices_in_faces_set = set(vertex_indices_in_faces_temp)
                    # 🎯 WICHTIG: Die ersten N additional_vertices sind Ecken (N = Anzahl Polygon-Ecken)
                    # Die restlichen additional_vertices sind Randpunkte
                    corner_vertices_count = min(len(surface_points), len(additional_vertices_array)) if len(surface_points) > 0 and len(additional_vertices_array) > 0 else 0
                    edge_vertices_start_idx = corner_vertices_count
                    edge_corner_connections = 0  # Initialisiere Zähler für Edge-Corner-Verbindungen
                    edge_to_edge_triangles = 0  # Initialisiere Zähler für Edge-to-Edge-Triangles
                    
                    if len(surface_points) >= 3 and len(additional_vertices) > 0:
                        # Bestimme Koordinatensystem basierend auf Orientierung
                        if geom.orientation in ("planar", "sloped"):
                            polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                            polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                            
                            # Finde Randpunkte (in mask_flat aber nicht in mask_strict_flat oder additional_vertices)
                            boundary_mask = mask_flat & (~mask_strict_flat)
                            boundary_indices_local = np.where(boundary_mask)[0]
                            
                            # 🎯 WICHTIG: corner_vertices_count wurde bereits oben definiert
                            n_poly = len(polygon_x)
                            
                            # Für jeden Eckpunkt: Finde benachbarte Randpunkte entlang der Kanten
                            for corner_idx in range(corner_vertices_count):
                                if corner_idx >= len(additional_vertices_array):
                                    continue
                                corner_vertex = additional_vertices_array[corner_idx]
                                corner_vertex_idx = additional_vertex_start_idx + corner_idx
                                corner_x, corner_y = corner_vertex[0], corner_vertex[1]
                                
                                # Finde die zwei Kanten, die an diesem Eckpunkt anliegen
                                corner_in_polygon = False
                                corner_poly_idx = -1
                                
                                for i in range(n_poly):
                                    if abs(polygon_x[i] - corner_x) < 1e-9 and abs(polygon_y[i] - corner_y) < 1e-9:
                                        corner_in_polygon = True
                                        corner_poly_idx = i
                                        break
                                
                                if corner_in_polygon:
                                    # Finde benachbarte Randpunkte entlang der zwei Kanten
                                    edge_threshold = resolution * 1.5  # Toleranz für "auf der Kante"
                                    
                                    # Kante 1: Vom Eckpunkt zum nächsten Polygon-Punkt
                                    next_poly_idx = (corner_poly_idx + 1) % n_poly
                                    edge1_start = np.array([corner_x, corner_y])
                                    edge1_end = np.array([polygon_x[next_poly_idx], polygon_y[next_poly_idx]])
                                    
                                    # Kante 2: Vom vorherigen Polygon-Punkt zum Eckpunkt
                                    prev_poly_idx = (corner_poly_idx - 1 + n_poly) % n_poly
                                    edge2_start = np.array([polygon_x[prev_poly_idx], polygon_y[prev_poly_idx]])
                                    edge2_end = np.array([corner_x, corner_y])
                                    
                                    # Finde Randpunkte auf beiden Kanten
                                    edge_points_on_edge1 = []
                                    edge_points_on_edge2 = []
                                    
                                    # 🎯 PRÜFE GRID-PUNKTE AUF DEM RAND
                                    for boundary_idx in boundary_indices_local:
                                        if boundary_idx >= len(all_vertices):
                                            continue
                                        boundary_vertex = all_vertices[boundary_idx]
                                        bx, by = boundary_vertex[0], boundary_vertex[1]
                                        b_point = np.array([bx, by])
                                        
                                        # Prüfe ob auf Kante 1
                                        edge1_vec = edge1_end - edge1_start
                                        edge1_len = np.linalg.norm(edge1_vec)
                                        if edge1_len > 1e-9:
                                            edge1_unit = edge1_vec / edge1_len
                                            to_point = b_point - edge1_start
                                            t = np.clip(np.dot(to_point, edge1_unit) / edge1_len, 0.0, 1.0)
                                            proj_point = edge1_start + t * edge1_vec
                                            dist_to_edge = np.linalg.norm(b_point - proj_point)
                                            
                                            if dist_to_edge < edge_threshold and 0.05 < t < 0.95:  # Nicht zu nah am Eckpunkt
                                                edge_points_on_edge1.append((boundary_idx, dist_to_edge, t))
                                        
                                        # Prüfe ob auf Kante 2
                                        edge2_vec = edge2_end - edge2_start
                                        edge2_len = np.linalg.norm(edge2_vec)
                                        if edge2_len > 1e-9:
                                            edge2_unit = edge2_vec / edge2_len
                                            to_point = b_point - edge2_start
                                            t = np.clip(np.dot(to_point, edge2_unit) / edge2_len, 0.0, 1.0)
                                            proj_point = edge2_start + t * edge2_vec
                                            dist_to_edge = np.linalg.norm(b_point - proj_point)
                                            
                                            if dist_to_edge < edge_threshold and 0.05 < t < 0.95:  # Nicht zu nah am Eckpunkt
                                                edge_points_on_edge2.append((boundary_idx, dist_to_edge, t))
                                    
                                    # 🎯 PRÜFE RANDPUNKTE ALS ADDITIONAL_VERTICES
                                    for edge_vertex_idx in range(edge_vertices_start_idx, len(additional_vertices_array)):
                                        edge_vertex = additional_vertices_array[edge_vertex_idx]
                                        edge_vertex_global_idx = additional_vertex_start_idx + edge_vertex_idx
                                        ex, ey = edge_vertex[0], edge_vertex[1]
                                        e_point = np.array([ex, ey])
                                        
                                        # Prüfe ob auf Kante 1
                                        edge1_vec = edge1_end - edge1_start
                                        edge1_len = np.linalg.norm(edge1_vec)
                                        if edge1_len > 1e-9:
                                            edge1_unit = edge1_vec / edge1_len
                                            to_point = e_point - edge1_start
                                            t = np.clip(np.dot(to_point, edge1_unit) / edge1_len, 0.0, 1.0)
                                            proj_point = edge1_start + t * edge1_vec
                                            dist_to_edge = np.linalg.norm(e_point - proj_point)
                                            
                                            if dist_to_edge < edge_threshold and 0.05 < t < 0.95:  # Nicht zu nah am Eckpunkt
                                                edge_points_on_edge1.append((edge_vertex_global_idx, dist_to_edge, t))
                                        
                                        # Prüfe ob auf Kante 2
                                        edge2_vec = edge2_end - edge2_start
                                        edge2_len = np.linalg.norm(edge2_vec)
                                        if edge2_len > 1e-9:
                                            edge2_unit = edge2_vec / edge2_len
                                            to_point = e_point - edge2_start
                                            t = np.clip(np.dot(to_point, edge2_unit) / edge2_len, 0.0, 1.0)
                                            proj_point = edge2_start + t * edge2_vec
                                            dist_to_edge = np.linalg.norm(e_point - proj_point)
                                            
                                            if dist_to_edge < edge_threshold and 0.05 < t < 0.95:  # Nicht zu nah am Eckpunkt
                                                edge_points_on_edge2.append((edge_vertex_global_idx, dist_to_edge, t))
                                    
                                    # Sortiere Randpunkte nach Abstand zum Eckpunkt entlang der Kante
                                    edge_points_on_edge1.sort(key=lambda x: x[2])  # Sortiere nach t (Position auf Kante)
                                    edge_points_on_edge2.sort(key=lambda x: x[2], reverse=True)  # Sortiere nach t (rückwärts)
                                    
                                    # Verbinde Eckpunkt mit nächsten 1-2 Randpunkten auf jeder Kante
                                    max_points_per_edge = 2  # Maximal 2 Randpunkte pro Kante
                                    connected_points = []
                                    
                                    for idx, dist, t in edge_points_on_edge1[:max_points_per_edge]:
                                        connected_points.append(idx)
                                    for idx, dist, t in edge_points_on_edge2[:max_points_per_edge]:
                                        connected_points.append(idx)
                                    
                                    # Erstelle Dreiecke: Eckpunkt + 2 benachbarte Randpunkte
                                    if len(connected_points) >= 2:
                                        # Verbinde erste beiden Punkte mit Eckpunkt
                                        faces_list.extend([3, corner_vertex_idx, connected_points[0], connected_points[1]])
                                        edge_corner_connections += 1
                                        
                                        # Verbinde weitere Punkte paarweise
                                        for i in range(1, len(connected_points) - 1):
                                            faces_list.extend([3, corner_vertex_idx, connected_points[i], connected_points[i+1]])
                                            edge_corner_connections += 1
                                    elif len(connected_points) == 1:
                                        # Nur ein Randpunkt: Finde nächsten anderen aktiven Punkt
                                        if len(vertex_indices_in_faces_set) > 0:
                                            # 🎯 SICHERHEIT: Filtere nur gültige Indizes
                                            valid_indices = [idx for idx in vertex_indices_in_faces_set if idx < len(all_vertices)]
                                            if len(valid_indices) > 0 and connected_points[0] < len(all_vertices):
                                                other_vertices = all_vertices[valid_indices]
                                                distances = np.linalg.norm(other_vertices - all_vertices[connected_points[0]], axis=1)
                                                nearest_other_idx = valid_indices[np.argmin(distances)]
                                                faces_list.extend([3, corner_vertex_idx, connected_points[0], nearest_other_idx])
                                            edge_corner_connections += 1
                        
                        # 🎯 NEU: EXPLIZITE TRIANGULATION ENTlang DER RANDLINIEN
                        # Verbinde benachbarte Randpunkte entlang jeder Polygon-Kante miteinander
                        # Dies stellt sicher, dass alle Randpunkte vollständig trianguliert werden
                        # NUR wenn manuelle Triangulation verwendet wird (nicht bei Delaunay)
                        if len(additional_vertices) > edge_vertices_start_idx:
                            # Sammle alle Randpunkte (additional_vertices ab edge_vertices_start_idx)
                            edge_vertices_list = []
                            for i in range(edge_vertices_start_idx, len(additional_vertices_array)):
                                edge_vertex = additional_vertices_array[i]
                                edge_vertex_global_idx = additional_vertex_start_idx + i
                                # 🎯 SICHERHEIT: Prüfe ob Index gültig ist
                                if edge_vertex_global_idx < len(all_vertices):
                                    edge_vertices_list.append((edge_vertex_global_idx, edge_vertex))
                            
                            if len(edge_vertices_list) > 0 and len(surface_points) >= 3:
                                # Bestimme Koordinatensystem basierend auf Orientierung
                                if geom.orientation in ("planar", "sloped"):
                                    polygon_x = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                                    polygon_y = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                                    
                                    # Für jede Polygon-Kante: Finde Randpunkte auf dieser Kante und verbinde sie
                                    n_poly = len(polygon_x)
                                    for edge_idx in range(n_poly):
                                        # Kante von Polygon-Punkt edge_idx zu (edge_idx + 1)
                                        edge_start = np.array([polygon_x[edge_idx], polygon_y[edge_idx]])
                                        edge_end = np.array([polygon_x[(edge_idx + 1) % n_poly], polygon_y[(edge_idx + 1) % n_poly]])
                                        
                                        # Finde alle Randpunkte auf dieser Kante
                                        edge_points_on_this_edge = []
                                        edge_threshold = resolution * 0.5  # Toleranz für "auf der Kante"
                                        
                                        for edge_vertex_global_idx, edge_vertex in edge_vertices_list:
                                            ex, ey = edge_vertex[0], edge_vertex[1]
                                            e_point = np.array([ex, ey])
                                            
                                            # Prüfe ob Randpunkt auf dieser Kante liegt
                                            edge_vec = edge_end - edge_start
                                            edge_len = np.linalg.norm(edge_vec)
                                            if edge_len > 1e-9:
                                                edge_unit = edge_vec / edge_len
                                                to_point = e_point - edge_start
                                                t = np.clip(np.dot(to_point, edge_unit) / edge_len, 0.0, 1.0)
                                                proj_point = edge_start + t * edge_vec
                                                dist_to_edge = np.linalg.norm(e_point - proj_point)
                                                
                                                if dist_to_edge < edge_threshold:
                                                    edge_points_on_this_edge.append((edge_vertex_global_idx, t, dist_to_edge))
                                        
                                        # Sortiere Randpunkte nach Position auf der Kante (t-Wert)
                                        edge_points_on_this_edge.sort(key=lambda x: x[1])
                                        
                                        # Verbinde benachbarte Randpunkte miteinander
                                        # Erstelle Dreiecke: Randpunkt i, Randpunkt i+1, und nächstgelegener Grid-Punkt
                                        for i in range(len(edge_points_on_this_edge) - 1):
                                            idx1, t1, dist1 = edge_points_on_this_edge[i]
                                            idx2, t2, dist2 = edge_points_on_this_edge[i + 1]
                                            
                                            # 🎯 SICHERHEIT: Prüfe ob Indizes gültig sind
                                            if idx1 >= len(all_vertices) or idx2 >= len(all_vertices):
                                                continue
                                            
                                            # Finde nächstgelegenen aktiven Grid-Punkt für das Dreieck
                                            # Verwende einen Punkt, der bereits in der Triangulation ist
                                            if len(vertex_indices_in_faces_set) > 0:
                                                # Finde den nächstgelegenen Punkt, der nicht einer der beiden Randpunkte ist
                                                other_indices = [idx for idx in vertex_indices_in_faces_set 
                                                               if idx != idx1 and idx != idx2 and idx < len(all_vertices)]
                                                if len(other_indices) > 0:
                                                    other_vertices = all_vertices[other_indices]
                                                    edge_vertex1 = all_vertices[idx1]
                                                    edge_vertex2 = all_vertices[idx2]
                                                    
                                                    # Berechne Mittelpunkt der beiden Randpunkte
                                                    midpoint = (edge_vertex1 + edge_vertex2) / 2.0
                                                    
                                                    # Finde nächstgelegenen Punkt zum Mittelpunkt
                                                    distances = np.linalg.norm(other_vertices - midpoint, axis=1)
                                                    nearest_idx = other_indices[np.argmin(distances)]
                                                    
                                                    # Erstelle Dreieck nur wenn Abstand nicht zu groß
                                                    max_edge_distance = resolution * 2.0
                                                    if np.min(distances) <= max_edge_distance:
                                                        faces_list.extend([3, idx1, idx2, nearest_idx])
                                                        edge_to_edge_triangles += 1
                                                        # Füge Indizes zu vertex_indices_in_faces_set hinzu
                                                        vertex_indices_in_faces_set.add(int(idx1))
                                                        vertex_indices_in_faces_set.add(int(idx2))
                                        
                                        # Verbinde auch den ersten und letzten Randpunkt mit den Eckpunkten
                                        if len(edge_points_on_this_edge) > 0:
                                            # Erster Randpunkt: Verbinde mit Start-Eckpunkt
                                            first_edge_idx, first_t, first_dist = edge_points_on_this_edge[0]
                                            # Finde Start-Eckpunkt in additional_vertices
                                            start_corner_idx = None
                                            for corner_idx in range(corner_vertices_count):
                                                if corner_idx < len(additional_vertices_array):
                                                    corner_vertex = additional_vertices_array[corner_idx]
                                                    corner_x, corner_y = corner_vertex[0], corner_vertex[1]
                                                    if abs(corner_x - edge_start[0]) < 1e-9 and abs(corner_y - edge_start[1]) < 1e-9:
                                                        start_corner_idx = additional_vertex_start_idx + corner_idx
                                                        break
                                            
                                            # Letzter Randpunkt: Verbinde mit End-Eckpunkt
                                            last_edge_idx, last_t, last_dist = edge_points_on_this_edge[-1]
                                            # Finde End-Eckpunkt in additional_vertices
                                            end_corner_idx = None
                                            for corner_idx in range(corner_vertices_count):
                                                if corner_idx < len(additional_vertices_array):
                                                    corner_vertex = additional_vertices_array[corner_idx]
                                                    corner_x, corner_y = corner_vertex[0], corner_vertex[1]
                                                    if abs(corner_x - edge_end[0]) < 1e-9 and abs(corner_y - edge_end[1]) < 1e-9:
                                                        end_corner_idx = additional_vertex_start_idx + corner_idx
                                                        break
                                            
                                            # Erstelle Dreiecke mit Eckpunkten und Randpunkten
                                            if start_corner_idx is not None and len(edge_points_on_this_edge) > 0:
                                                # 🎯 SICHERHEIT: Prüfe ob Indizes gültig sind
                                                if first_edge_idx >= len(all_vertices) or start_corner_idx >= len(all_vertices):
                                                    pass  # Überspringe wenn Indizes ungültig
                                                elif len(vertex_indices_in_faces_set) > 0:
                                                    other_indices = [idx for idx in vertex_indices_in_faces_set 
                                                                   if idx != first_edge_idx and idx != start_corner_idx and idx < len(all_vertices)]
                                                    if len(other_indices) > 0:
                                                        other_vertices = all_vertices[other_indices]
                                                        edge_vertex1 = all_vertices[first_edge_idx]
                                                        corner_vertex = all_vertices[start_corner_idx]
                                                        midpoint = (edge_vertex1 + corner_vertex) / 2.0
                                                        distances = np.linalg.norm(other_vertices - midpoint, axis=1)
                                                        nearest_idx = other_indices[np.argmin(distances)]
                                                        max_edge_distance = resolution * 2.0
                                                        if np.min(distances) <= max_edge_distance:
                                                            faces_list.extend([3, first_edge_idx, start_corner_idx, nearest_idx])
                                                            edge_to_edge_triangles += 1
                                                            vertex_indices_in_faces_set.add(int(first_edge_idx))
                                            
                                            if end_corner_idx is not None and len(edge_points_on_this_edge) > 0:
                                                # 🎯 SICHERHEIT: Prüfe ob Indizes gültig sind
                                                if last_edge_idx >= len(all_vertices) or end_corner_idx >= len(all_vertices):
                                                    pass  # Überspringe wenn Indizes ungültig
                                                elif len(vertex_indices_in_faces_set) > 0:
                                                    other_indices = [idx for idx in vertex_indices_in_faces_set 
                                                                   if idx != last_edge_idx and idx != end_corner_idx and idx < len(all_vertices)]
                                                    if len(other_indices) > 0:
                                                        other_vertices = all_vertices[other_indices]
                                                        edge_vertex2 = all_vertices[last_edge_idx]
                                                        corner_vertex = all_vertices[end_corner_idx]
                                                        midpoint = (edge_vertex2 + corner_vertex) / 2.0
                                                        distances = np.linalg.norm(other_vertices - midpoint, axis=1)
                                                        nearest_idx = other_indices[np.argmin(distances)]
                                                        max_edge_distance = resolution * 2.0
                                                        if np.min(distances) <= max_edge_distance:
                                                            faces_list.extend([3, last_edge_idx, end_corner_idx, nearest_idx])
                                                            edge_to_edge_triangles += 1
                                                            vertex_indices_in_faces_set.add(int(last_edge_idx))
                            
                            # #region agent log - Rand-zu-Rand Triangulation
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "EDGE_TO_EDGE_TRIANGULATION",
                                        "location": "FlexibleGridGenerator.generate_per_surface:edge_to_edge_triangulation",
                                        "message": "Rand-zu-Rand Triangulation",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "edge_vertices_count": len(edge_vertices_list),
                                            "edge_to_edge_triangles": edge_to_edge_triangles,
                                            "orientation": str(geom.orientation)
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                    
                    if len(faces_list) > 0:
                        # 🎯 OPTIMIERUNG: Nur Vertices innerhalb der Maske verwenden
                        # Sammle alle eindeutigen Vertex-Indizes, die in Faces verwendet werden
                        faces_array_temp = np.array(faces_list, dtype=np.int64)
                        # Extrahiere Vertex-Indices aus Faces (Format: [n, v1, v2, v3, n, v4, v5, v6, ...])
                        vertex_indices_in_faces = []
                        for i in range(0, len(faces_list), 4):
                            if i + 3 < len(faces_list):
                                n_verts = faces_list[i]
                                if n_verts == 3:
                                    vertex_indices_in_faces.extend(faces_list[i+1:i+4])
                        
                        # 🎯 RANDPUNKTE INKLUDIEREN: Füge alle aktiven Masken-Punkte hinzu, auch wenn sie nicht in Faces sind
                        # Dies stellt sicher, dass alle Randpunkte als Vertices erstellt werden
                        vertex_indices_in_faces_set = set(vertex_indices_in_faces)
                        
                        # Prüfe, welche aktiven Masken-Punkte nicht in Faces verwendet werden
                        # (Dies kann Randpunkte einschließen, auch wenn sie in mask_strict sind)
                        active_mask_indices = np.where(mask_flat)[0]
                        active_not_in_faces = [idx for idx in active_mask_indices if idx not in vertex_indices_in_faces_set]
                        
                        # #region agent log - Randpunkte in Triangulation prüfen
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json, time as _t
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_IN_TRIANGULATION",
                                    "location": "FlexibleGridGenerator.generate_per_surface:triangulation_vertices",
                                    "message": "Randpunkte in Triangulation prüfen",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "active_points_total": int(len(active_mask_indices)),
                                        "active_points_in_faces": int(len(active_mask_indices) - len(active_not_in_faces)),
                                        "active_points_not_in_faces": int(len(active_not_in_faces)),
                                        "vertices_in_faces_before": int(len(vertex_indices_in_faces)),
                                        "mask_flat_active": int(np.count_nonzero(mask_flat)),
                                        "mask_strict_active": int(np.count_nonzero(mask_strict_flat))
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # 🎯 OPTIMIERT: Erstelle Faces für isolierte aktive Punkte nur wenn nötig
                        # Verbinde isolierte Punkte (Randpunkte) explizit mit benachbarten Punkten
                        # und Eckpunkten entlang der Surface-Linien
                        faces_created_for_isolated = 0
                        isolated_added_to_vertices = 0
                        edge_to_corner_triangles = 0  # Dreiecke zwischen Randpunkten und Eckpunkten
                        
                        if len(active_not_in_faces) > 0:
                            # Für jeden isolierten aktiven Punkt: Finde den nächstgelegenen aktiven Punkt, der bereits in Faces ist
                            # und erstelle ein kleines Dreieck, damit der Punkt gerendert wird
                            for isolated_idx in active_not_in_faces:
                                # 🎯 SICHERHEIT: Prüfe ob Index gültig ist
                                if isolated_idx >= len(all_vertices):
                                    continue
                                
                                isolated_vertex = all_vertices[isolated_idx]
                                
                                # Finde nächstgelegenen aktiven Punkt, der bereits in Faces verwendet wird
                                if len(vertex_indices_in_faces_set) >= 2:
                                    # 🎯 SICHERHEIT: Filtere nur gültige Indizes
                                    valid_indices = [idx for idx in vertex_indices_in_faces_set if idx < len(all_vertices)]
                                    if len(valid_indices) >= 2:
                                    # Verwende Punkte, die bereits in Faces sind
                                        vertices_in_faces = all_vertices[valid_indices]
                                    
                                    # Berechne Abstände
                                    distances = np.sqrt(np.sum((vertices_in_faces - isolated_vertex)**2, axis=1))
                                    nearest_idx_in_faces = valid_indices[np.argmin(distances)]
                                    
                                    # Finde einen zweiten Punkt, um ein Dreieck zu erstellen
                                    # Verwende den nächstgelegenen Punkt, der nicht der erste ist
                                    other_indices = [idx for idx in valid_indices if idx != nearest_idx_in_faces]
                                    if len(other_indices) > 0:
                                            other_vertices = all_vertices[other_indices]
                                            distances2 = np.sqrt(np.sum((other_vertices - isolated_vertex)**2, axis=1))
                                            second_idx = other_indices[np.argmin(distances2)]
                                            
                                            # 🎯 OPTIMIERT: Erstelle Dreieck nur wenn Abstand nicht zu groß
                                            max_isolated_distance = resolution * 2.0  # Maximaler Abstand für Verbindung
                                            isolated_distance = np.linalg.norm(isolated_vertex - all_vertices[nearest_idx_in_faces])
                                            
                                            if isolated_distance <= max_isolated_distance:
                                                # Erstelle Dreieck: isolierter Punkt + 2 benachbarte Punkte
                                                faces_list.extend([3, isolated_idx, nearest_idx_in_faces, second_idx])
                                            vertex_indices_in_faces_set.add(int(isolated_idx))
                                            faces_created_for_isolated += 1
                                            isolated_added_to_vertices += 1
                                elif len(vertex_indices_in_faces_set) == 1:
                                    # Falls nur 1 Punkt in Faces ist, verwende diesen Punkt zweimal für ein degeneriertes Dreieck
                                    # (PyVista kann damit umgehen)
                                    single_idx = list(vertex_indices_in_faces_set)[0]
                                    faces_list.extend([3, isolated_idx, single_idx, single_idx])
                                    vertex_indices_in_faces_set.add(int(isolated_idx))
                                    faces_created_for_isolated += 1
                                    isolated_added_to_vertices += 1
                        
                        # #region agent log - Faces für isolierte Punkte
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json, time as _t
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_FACES_CREATED",
                                    "location": "FlexibleGridGenerator.generate_per_surface:isolated_faces",
                                    "message": "Faces für isolierte Punkte erstellt",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "isolated_points_found": int(len(active_not_in_faces)),
                                        "faces_created": int(faces_created_for_isolated),
                                        "isolated_added_to_vertices": int(isolated_added_to_vertices)
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # 🎯 WICHTIG: Füge ALLE aktiven Masken-Punkte hinzu (inkl. Randpunkte)
                        # Dies stellt sicher, dass alle Punkte als Vertices erstellt werden, auch wenn sie nicht in Faces sind
                        active_mask_indices = np.where(mask_flat)[0]
                        for idx in active_mask_indices:
                            if idx < len(all_vertices):  # Sicherstellen, dass Index gültig ist
                                vertex_indices_in_faces_set.add(int(idx))
                        
                        # Füge auch zusätzliche Vertices hinzu (Polygon-Ecken und Randpunkte)
                        # 🎯 WICHTIG: additional_vertices wurden bereits zu all_vertices hinzugefügt (Zeile 3371)
                        # Daher ist len(all_vertices) = base_vertex_count + len(additional_vertices)
                        if len(additional_vertices) > 0:
                            for i in range(len(additional_vertices)):
                                additional_idx = base_vertex_count + i
                                if additional_idx < len(all_vertices):  # all_vertices enthält bereits additional_vertices
                                    vertex_indices_in_faces_set.add(int(additional_idx))
                        
                        # Prüfe, ob alle aktiven Masken-Punkte jetzt in vertex_indices_in_faces_set sind
                        missing_indices = [idx for idx in active_mask_indices if idx not in vertex_indices_in_faces_set]
                        
                        # #region agent log - Finale Vertex-Prüfung
                        try:
                            import json, time as _t
                            # 🎯 ALTE PROJEKTIONSLOGIK ENTFERNT: boundary_indices existiert nicht mehr
                            # Randpunkte werden jetzt direkt berechnet und als additional_vertices hinzugefügt
                            boundary_in_vertices = 0  # Nicht mehr relevant, da boundary_indices entfernt wurde
                            
                            # 🎯 PRÜFUNG 1: Werden Randpunkte für Calc verwendet (Maske)?
                            # Randpunkte sind als additional_vertices hinzugefügt, nicht in mask_flat
                            # Sie werden für Berechnung verwendet, wenn sie in Triangulation sind
                            boundary_points_in_calc = boundary_points_added  # Anzahl hinzugefügter Randpunkte
                            
                            # 🎯 PRÜFUNG 2: Werden Randpunkte für Triangulation verwendet?
                            # Prüfe welche Randpunkte (als additional_vertices) in Triangulation verwendet werden
                            edge_vertices_start_idx = corner_vertices_count if 'corner_vertices_count' in locals() else len(surface_points) if len(surface_points) > 0 else 0
                            edge_vertices_count = len(additional_vertices) - edge_vertices_start_idx if len(additional_vertices) > edge_vertices_start_idx else 0
                            edge_vertices_in_faces = 0
                            boundary_points_in_triangulation = 0
                            if edge_vertices_count > 0:
                                for i in range(edge_vertices_start_idx, len(additional_vertices)):
                                    edge_vertex_idx = additional_vertex_start_idx + i
                                    if edge_vertex_idx in vertex_indices_in_faces_set:
                                        edge_vertices_in_faces += 1
                                        # Prüfe ob dieser Vertex ein Randpunkt ist (nicht Ecke)
                                        if i >= edge_vertices_start_idx:
                                            boundary_points_in_triangulation += 1
                            
                            # 🎯 PRÜFUNG 3: Plot der Randvertices
                            # Randpunkte sollten in triangulated_vertices enthalten sein für Plot
                            boundary_points_in_plot = boundary_points_in_triangulation  # Für Plot müssen sie in Triangulation sein
                            
                            # #region agent log - Neue Randpunktlogik: Prüfungen (Calc, Triangulation, Plot)
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "BOUNDARY_POINTS_CHECKS",
                                        "location": "FlexibleGridGenerator.generate_per_surface:boundary_points_checks",
                                        "message": "Prüfungen: Randpunkte für Calc, Triangulation, Plot",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "boundary_points_added": boundary_points_added,
                                            "boundary_points_in_calc": boundary_points_in_calc,
                                            "boundary_points_in_triangulation": boundary_points_in_triangulation,
                                            "boundary_points_in_plot": boundary_points_in_plot,
                                            "calc_check_passed": boundary_points_in_calc > 0,
                                            "triangulation_check_passed": boundary_points_in_triangulation > 0,
                                            "plot_check_passed": boundary_points_in_plot > 0,
                                            "all_checks_passed": (boundary_points_in_calc > 0 and boundary_points_in_triangulation > 0 and boundary_points_in_plot > 0),
                                            "edge_vertices_count": edge_vertices_count,
                                            "edge_vertices_in_faces": edge_vertices_in_faces
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_VERTICES_FINAL",
                                    "location": "FlexibleGridGenerator.generate_per_surface:final_vertices",
                                    "message": "Finale Vertex-Prüfung vor Mapping",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "active_mask_count": int(len(active_mask_indices)),
                                        "vertex_indices_count": int(len(vertex_indices_in_faces_set)),
                                        "missing_indices_count": int(len(missing_indices)),
                                        "additional_vertices_count": int(len(additional_vertices)),
                                        "edge_vertices_count": edge_vertices_count,
                                        "edge_vertices_in_faces": edge_vertices_in_faces,
                                        "all_edge_vertices_in_faces": edge_vertices_in_faces == edge_vertices_count if edge_vertices_count > 0 else True,
                                        "boundary_in_vertices_count": boundary_in_vertices
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        all_vertex_indices = np.array(sorted(vertex_indices_in_faces_set), dtype=np.int64)
                        
                        # Erstelle Mapping: alter Index → neuer Index
                        old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(all_vertex_indices)}
                        
                        # 🎯 OPTIMIERUNG: Erstelle vorläufiges vertex_source_indices-Mapping
                        # Vor der Deduplizierung: Grid-Vertices → Grid-Index, Additional-Vertices → Additional-Index
                        n_grid_vertices_selected = 0
                        n_additional_vertices_selected = 0
                        vertex_source_indices_pre_dedup = []
                        
                        # Erstelle nur die Vertices, die tatsächlich verwendet werden
                        # Für zusätzliche Vertices: Sie sind nach base_vertex_count, also müssen wir sie separat handhaben
                        if len(all_vertex_indices) > 0:
                            # Trenne normale Grid-Vertices von zusätzlichen Vertices
                            grid_vertex_indices = all_vertex_indices[all_vertex_indices < base_vertex_count]
                            additional_vertex_indices = all_vertex_indices[all_vertex_indices >= base_vertex_count]
                            
                            # Erstelle Vertices aus Grid-Punkten
                            if len(grid_vertex_indices) > 0:
                                grid_vertices = all_vertices[grid_vertex_indices]
                                n_grid_vertices_selected = len(grid_vertices)
                                # Mapping: Grid-Vertex → Grid-Index (in surface_mask_flat)
                                for old_idx in grid_vertex_indices:
                                    new_idx = old_to_new_index[old_idx]
                                    # Grid-Vertex: Index = Position in surface_mask_flat
                                    vertex_source_indices_pre_dedup.append(('grid', old_idx))
                            else:
                                grid_vertices = np.array([], dtype=float).reshape(0, 3)
                            
                            # Erstelle Vertices aus zusätzlichen Vertices (Polygon-Ecken)
                            if len(additional_vertex_indices) > 0 and len(additional_vertices_array) > 0:
                                additional_indices_local = additional_vertex_indices - base_vertex_count
                                additional_indices_local = additional_indices_local[additional_indices_local < len(additional_vertices_array)]
                                if len(additional_indices_local) > 0:
                                    additional_vertices_selected = additional_vertices_array[additional_indices_local]
                                    n_additional_vertices_selected = len(additional_vertices_selected)
                                    # Mapping: Additional-Vertex → Additional-Index (in additional_vertices_array)
                                    for old_idx in additional_vertex_indices:
                                        new_idx = old_to_new_index[old_idx]
                                        local_idx = old_idx - base_vertex_count
                                        if 0 <= local_idx < len(additional_vertices_array):
                                            vertex_source_indices_pre_dedup.append(('additional', local_idx))
                                else:
                                    additional_vertices_selected = np.array([], dtype=float).reshape(0, 3)
                            else:
                                additional_vertices_selected = np.array([], dtype=float).reshape(0, 3)
                            
                            # Kombiniere beide
                            if len(grid_vertices) > 0 and len(additional_vertices_selected) > 0:
                                triangulated_vertices = np.vstack([grid_vertices, additional_vertices_selected])
                            elif len(grid_vertices) > 0:
                                triangulated_vertices = grid_vertices
                            elif len(additional_vertices_selected) > 0:
                                triangulated_vertices = additional_vertices_selected
                            else:
                                triangulated_vertices = np.array([], dtype=float).reshape(0, 3)
                        else:
                            triangulated_vertices = np.array([], dtype=float).reshape(0, 3)
                            vertex_source_indices_pre_dedup = []
                        
                        # #region agent log - Randpunkte in finalen triangulated_vertices
                        try:
                            import json, time as _t
                            # 🎯 ALTE PROJEKTIONSLOGIK ENTFERNT: boundary_indices existiert nicht mehr
                            # Randpunkte werden jetzt direkt berechnet und als additional_vertices hinzugefügt
                            boundary_in_triangulated = 0  # Nicht mehr relevant, da boundary_indices entfernt wurde
                            
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_IN_TRIANGULATED",
                                    "location": "FlexibleGridGenerator.generate_per_surface:boundary_in_triangulated",
                                    "message": "Randpunkte in finalen triangulated_vertices",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "boundary_in_triangulated_count": boundary_in_triangulated,
                                        "triangulated_vertices_count": int(len(triangulated_vertices)) if triangulated_vertices is not None and triangulated_vertices.size > 0 else 0
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # Mappe Face-Indizes auf neue Vertex-Indizes
                        faces_list_mapped = []
                        missing_indices_count = 0
                        for i in range(0, len(faces_list), 4):
                            if i + 3 < len(faces_list):
                                n_verts = faces_list[i]
                                if n_verts == 3:
                                    v1_old = faces_list[i+1]
                                    v2_old = faces_list[i+2]
                                    v3_old = faces_list[i+3]
                                    # Prüfe ob alle Indizes in old_to_new_index vorhanden sind
                                    if v1_old in old_to_new_index and v2_old in old_to_new_index and v3_old in old_to_new_index:
                                        faces_list_mapped.extend([
                                            3,
                                            old_to_new_index[v1_old],
                                            old_to_new_index[v2_old],
                                            old_to_new_index[v3_old]
                                        ])
                                    else:
                                        missing_indices_count += 1
                        
                        if len(faces_list_mapped) > 0:
                            triangulated_faces = np.array(faces_list_mapped, dtype=np.int64)
                            triangulated_success = True
                            
                            # #region agent log - Finale Triangulation Koordinaten-Vergleich
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    # Vergleiche triangulierte Vertices mit Original-Polygon-Punkten
                                    if triangulated_vertices is not None and triangulated_vertices.size > 0:
                                        orig_points = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in (geom.points or [])], dtype=float)
                                        if len(orig_points) > 0:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "H5",
                                                "location": "FlexibleGridGenerator.generate_per_surface:triangulation_final_coords",
                                                "message": "Finale Triangulation Koordinaten-Vergleich",
                                                "data": {
                                                    "surface_id": str(geom.surface_id),
                                                    "orientation": str(geom.orientation),
                                                    "dominant_axis": str(geom.dominant_axis) if hasattr(geom, 'dominant_axis') and geom.dominant_axis else None,
                                                    "triangulated_vertices_count": int(len(triangulated_vertices)),
                                                    "triangulated_vertices_x_min": float(triangulated_vertices[:, 0].min()),
                                                    "triangulated_vertices_x_max": float(triangulated_vertices[:, 0].max()),
                                                    "triangulated_vertices_y_min": float(triangulated_vertices[:, 1].min()),
                                                    "triangulated_vertices_y_max": float(triangulated_vertices[:, 1].max()),
                                                    "triangulated_vertices_z_min": float(triangulated_vertices[:, 2].min()),
                                                    "triangulated_vertices_z_max": float(triangulated_vertices[:, 2].max()),
                                                    "original_points_count": int(len(orig_points)),
                                                    "original_points_x_min": float(orig_points[:, 0].min()),
                                                    "original_points_x_max": float(orig_points[:, 0].max()),
                                                    "original_points_y_min": float(orig_points[:, 1].min()),
                                                    "original_points_y_max": float(orig_points[:, 1].max()),
                                                    "original_points_z_min": float(orig_points[:, 2].min()),
                                                    "original_points_z_max": float(orig_points[:, 2].max()),
                                                    "triangulated_faces_count": int(len(triangulated_faces) // 4) if triangulated_faces is not None else 0
                                                },
                                                "timestamp": int(_t.time() * 1000)
                                            }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                        else:
                            # Keine Faces erstellt - Triangulation fehlgeschlagen
                            triangulated_faces = np.array([], dtype=np.int64)
                            triangulated_success = False
                            # #region agent log - Triangulation fehlgeschlagen (keine Faces)
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "TRIANGULATION_FAILED_NO_FACES",
                                        "location": "FlexibleGridGenerator.generate_per_surface:triangulation_no_faces",
                                        "message": "Triangulation fehlgeschlagen - keine Faces erstellt",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "total_grid_points": int(X_grid.size),
                                            "active_mask_points": int(np.count_nonzero(mask_flat)),
                                            "active_strict_mask_points": int(np.count_nonzero(mask_strict_flat)),
                                            "total_quads_checked": int((ny - 1) * (nx - 1)),
                                            "active_quads": active_quads,
                                            "partial_quads": partial_quads,
                                            "faces_list_length": int(len(faces_list)),
                                            "faces_list_mapped_length": int(len(faces_list_mapped)),
                                            "missing_indices_count": missing_indices_count,
                                            "old_to_new_index_size": int(len(old_to_new_index)),
                                            "additional_vertices_count": int(len(additional_vertices)) if len(additional_vertices) > 0 else 0
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                        
                        # #region agent log - Triangulation Statistiken
                        try:
                            import json, time as _t
                            total_faces = len(faces_list_mapped) // 4
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "TRIANGULATION_STATISTICS",
                                    "location": "FlexibleGridGenerator.generate_per_surface:triangulation_statistics",
                                    "message": "Triangulation Statistiken",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "total_faces": total_faces,
                                        "active_quads": active_quads,
                                        "partial_quads": partial_quads,
                                        "corner_vertex_triangles": corner_vertex_triangles,
                                        "edge_to_corner_triangles": edge_corner_connections if 'edge_corner_connections' in locals() else 0,
                                        "isolated_point_triangles": faces_created_for_isolated,
                                        "filtered_out": filtered_out,
                                        "additional_vertices_count": int(len(additional_vertices)) if len(additional_vertices) > 0 else 0
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # #region agent log - Finale Vertex-Anzahl-Prüfung
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json, time as _t
                                final_vertex_count = int(len(triangulated_vertices))
                                mask_active_count = int(np.count_nonzero(mask_flat))
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_VERTICES_COUNT",
                                    "location": "FlexibleGridGenerator.generate_per_surface:triangulation_final",
                                    "message": "Finale Vertex-Anzahl-Prüfung",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "final_vertex_count": final_vertex_count,
                                        "mask_active_count": mask_active_count,
                                        "vertex_indices_count": int(len(all_vertex_indices)),
                                        "match": final_vertex_count == mask_active_count
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # #region agent log
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json, time as _t
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "grid-analysis-v2",
                                    "hypothesisId": "H1",
                                    "location": "FlexibleGridGenerator.generate_per_surface:triangulation",
                                    "message": "Triangulation erfolgreich - Grid erstellt",
                                    "data": {
                                        "surface_id": str(geom.surface_id),
                                        "orientation": str(geom.orientation),
                                        "total_grid_points": int(X_grid.size),
                                        "grid_shape": list(X_grid.shape),
                                        "n_triangulated_vertices": int(len(triangulated_vertices)),
                                        "n_triangulated_faces": int(len(triangulated_faces) // 4) if triangulated_faces.ndim == 1 else int(len(triangulated_faces)),
                                        "points_in_mask": int(np.count_nonzero(mask_flat)),
                                        "resolution": float(actual_resolution)
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        n_faces = len(faces_list_mapped) // 4  # 4 Elemente pro Face: [n, v1, v2, v3]
                        n_vertices_used = len(all_vertex_indices)  # Anzahl der tatsächlich verwendeten Vertices
                        # Berechne erwartete Faces nur wenn manuelle Triangulation verwendet wurde
                        # (bei Delaunay ist diese Berechnung nicht relevant)
                        if not use_delaunay_triangulation:
                            expected_faces = active_quads * 2 + partial_quads  # Volle Quadrate: 2 Dreiecke, Rand-Quadrate: 1 Dreieck
                        else:
                            expected_faces = 0  # Bei Delaunay wird diese Statistik nicht verwendet
                        
                        # 🎯 DEDUPLIKATION: Entferne doppelte Vertices
                        vertices_before_dedup = int(len(triangulated_vertices)) if triangulated_vertices is not None and triangulated_vertices.size > 0 else 0
                        
                        # #region agent log - Vertex overlap before deduplication H5, H6, H7
                        try:
                            import json, time as _t
                            if vertices_before_dedup > 0:
                                verts = np.asarray(triangulated_vertices, dtype=float)
                                # Prüfe Überlappungen mit kleinerer Toleranz (1% der Resolution)
                                tol_check = float(actual_resolution * 0.01) if actual_resolution > 0 else 1e-3
                                quant_check = np.round(verts / tol_check) * tol_check
                                from collections import defaultdict
                                groups_check: dict[tuple[float, float, float], list[int]] = defaultdict(list)
                                for idx_v, (qx, qy, qz) in enumerate(quant_check):
                                    groups_check[(float(qx), float(qy), float(qz))].append(int(idx_v))
                                
                                overlap_positions_before = sum(1 for idx_list in groups_check.values() if len(idx_list) > 1)
                                max_dist_before = 0.0
                                example_overlaps = []
                                for (qx, qy, qz), idx_list in groups_check.items():
                                    if len(idx_list) > 1:
                                        local_verts = verts[idx_list]
                                        for i in range(len(local_verts)):
                                            for j in range(i + 1, len(local_verts)):
                                                d = float(np.linalg.norm(local_verts[i] - local_verts[j]))
                                                if d > max_dist_before:
                                                    max_dist_before = d
                                        if len(example_overlaps) < 3:
                                            example_overlaps.append({
                                                "position": [float(qx), float(qy), float(qz)],
                                                "vertex_count": len(idx_list),
                                                "max_distance": float(max([np.linalg.norm(verts[idx_list[i]] - verts[idx_list[j]]) for i in range(len(idx_list)) for j in range(i+1, len(idx_list))])) if len(idx_list) > 1 else 0.0
                                            })
                                
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H5_H6_H7",
                                        "location": "FlexibleGridGenerator.generate_per_surface:overlap_before_dedup",
                                        "message": "Vertex-Überlappung vor Deduplizierung",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "vertices_before_dedup": vertices_before_dedup,
                                            "overlap_positions": overlap_positions_before,
                                            "max_distance_between_overlaps": float(max_dist_before),
                                            "check_tolerance": float(tol_check),
                                            "example_overlaps": example_overlaps[:3]
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        # 🎯 OPTIMIERUNG: Speichere Vertex-Positionen vor Deduplizierung für Mapping
                        vertices_before_dedup_array = triangulated_vertices.copy() if triangulated_vertices.size > 0 else np.array([], dtype=float).reshape(0, 3)
                        
                        try:
                            # Verwende builder._deduplicate_vertices_and_faces direkt
                            triangulated_vertices, triangulated_faces = self._deduplicate_vertices_and_faces(
                                triangulated_vertices,
                                triangulated_faces,
                                actual_resolution,
                                geom.surface_id
                            )
                            vertices_after_dedup = int(len(triangulated_vertices)) if triangulated_vertices is not None and triangulated_vertices.size > 0 else 0
                            vertices_removed = vertices_before_dedup - vertices_after_dedup
                            
                            # 🎯 OPTIMIERUNG: Erstelle vertex_source_indices nach Deduplizierung
                            # Mapping: Neuer Vertex-Index → Source-Index
                            # Grid-Vertices: Source-Index = Grid-Index in surface_mask_flat (0..N-1)
                            # Additional-Vertices: Source-Index = N + Additional-Index (N..N+M-1, wobei N = base_vertex_count)
                            vertex_source_indices = None
                            if vertices_after_dedup > 0 and len(vertex_source_indices_pre_dedup) > 0 and len(vertices_before_dedup_array) > 0:
                                try:
                                    from scipy.spatial.distance import cdist
                                    # Finde für jeden deduplizierten Vertex den nächstgelegenen Vertex vor Deduplizierung
                                    distances = cdist(triangulated_vertices, vertices_before_dedup_array)
                                    nearest_pre_dedup_indices = np.argmin(distances, axis=1)
                                    
                                    # Erstelle Mapping: Neuer Index → Source-Index
                                    vertex_source_indices_mapping = []
                                    n_grid_total = base_vertex_count
                                    
                                    for new_idx in range(vertices_after_dedup):
                                        pre_dedup_idx = nearest_pre_dedup_indices[new_idx]
                                        if pre_dedup_idx < len(vertex_source_indices_pre_dedup):
                                            mapping_type, source_idx = vertex_source_indices_pre_dedup[pre_dedup_idx]
                                            if mapping_type == 'grid':
                                                # Grid-Vertex: Index = Grid-Index in surface_mask_flat (0..N-1)
                                                vertex_source_indices_mapping.append(int(source_idx))
                                            else:  # 'additional'
                                                # Additional-Vertex: Index = N + Additional-Index (N..N+M-1)
                                                vertex_source_indices_mapping.append(int(n_grid_total + source_idx))
                                        else:
                                            # Fallback: Verwende Grid-Index 0
                                            vertex_source_indices_mapping.append(0)
                                    
                                    vertex_source_indices = np.array(vertex_source_indices_mapping, dtype=np.int64)
                                except Exception as e:
                                    # Bei Fehler: Kein Mapping (wird im Plot interpoliert)
                                    vertex_source_indices = None
                            else:
                                vertex_source_indices = None
                            
                            # #region agent log - Deduplizierung Ergebnisse
                            try:
                                import json, time as _t
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "DEDUPLICATION_ANALYSIS",
                                        "location": "FlexibleGridGenerator.generate_per_surface:deduplication",
                                        "message": "Deduplizierung Ergebnisse",
                                        "data": {
                                            "surface_id": str(geom.surface_id),
                                            "vertices_before_dedup": vertices_before_dedup,
                                            "vertices_after_dedup": vertices_after_dedup,
                                            "vertices_removed": vertices_removed,
                                            "deduplication_tolerance": float(actual_resolution * 0.05) if actual_resolution > 0 else 0.0,
                                            "resolution": float(actual_resolution)
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            
                            # #region agent log - Vertex overlap after deduplication H5, H6, H7
                            try:
                                import json, time as _t
                                if vertices_after_dedup > 0:
                                    verts_after = np.asarray(triangulated_vertices, dtype=float)
                                    # Prüfe Überlappungen mit kleinerer Toleranz (1% der Resolution)
                                    tol_check = float(actual_resolution * 0.01) if actual_resolution > 0 else 1e-3
                                    quant_check = np.round(verts_after / tol_check) * tol_check
                                    from collections import defaultdict
                                    groups_check: dict[tuple[float, float, float], list[int]] = defaultdict(list)
                                    for idx_v, (qx, qy, qz) in enumerate(quant_check):
                                        groups_check[(float(qx), float(qy), float(qz))].append(int(idx_v))
                                    
                                    overlap_positions_after = sum(1 for idx_list in groups_check.values() if len(idx_list) > 1)
                                    max_dist_after = 0.0
                                    example_overlaps_after = []
                                    for (qx, qy, qz), idx_list in groups_check.items():
                                        if len(idx_list) > 1:
                                            local_verts = verts_after[idx_list]
                                            for i in range(len(local_verts)):
                                                for j in range(i + 1, len(local_verts)):
                                                    d = float(np.linalg.norm(local_verts[i] - local_verts[j]))
                                                    if d > max_dist_after:
                                                        max_dist_after = d
                                            if len(example_overlaps_after) < 3:
                                                example_overlaps_after.append({
                                                    "position": [float(qx), float(qy), float(qz)],
                                                    "vertex_count": len(idx_list),
                                                    "max_distance": float(max([np.linalg.norm(verts_after[idx_list[i]] - verts_after[idx_list[j]]) for i in range(len(idx_list)) for j in range(i+1, len(idx_list))])) if len(idx_list) > 1 else 0.0
                                                })
                                    
                                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "H5_H6_H7",
                                            "location": "FlexibleGridGenerator.generate_per_surface:overlap_after_dedup",
                                            "message": "Vertex-Überlappung nach Deduplizierung",
                                            "data": {
                                                "surface_id": str(geom.surface_id),
                                                "vertices_after_dedup": vertices_after_dedup,
                                                "overlap_positions": overlap_positions_after,
                                                "max_distance_between_overlaps": float(max_dist_after),
                                                "check_tolerance": float(tol_check),
                                                "example_overlaps": example_overlaps_after[:3]
                                        },
                                        "timestamp": int(_t.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                        except Exception as dedupe_error:
                            # Bei Fehler: Original-Vertices/Faces behalten
                            pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                triangulated_success = False
            
            # #region agent log - Triangulation Coverage prüfen
            try:
                import json, time as _t
                edge_vertices_in_triangulation = 0
                total_vertices_in_triangulation = 0
                edge_detection_success = False
                edge_detection_error = None
                min_dist_to_edge = None
                max_dist_to_edge = None
                mean_dist_to_edge = None
                if triangulated_success and triangulated_vertices is not None and triangulated_vertices.size > 0:
                    total_vertices_in_triangulation = len(triangulated_vertices)
                    # Prüfe, ob Vertices auf dem Rand in der Triangulation sind
                    surface_points = geom.points or []
                    if len(surface_points) >= 3:
                        try:
                            if geom.orientation == "vertical":
                                # Vertikale Flächen: (u,v) Koordinaten
                                xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                                ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                                zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                                if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                                    if geom.dominant_axis == "yz":
                                        polygon_2d = np.column_stack([ys, zs])
                                        vertex_coords_2d = triangulated_vertices[:, [1, 2]]
                                    else:  # xz
                                        polygon_2d = np.column_stack([xs, zs])
                                        vertex_coords_2d = triangulated_vertices[:, [0, 2]]
                                else:
                                    # Fallback: verwende xz
                                    polygon_2d = np.column_stack([xs, zs])
                                    vertex_coords_2d = triangulated_vertices[:, [0, 2]]
                            else:
                                # Planare/sloped: (x,y) Koordinaten
                                xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                                ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                                polygon_2d = np.column_stack([xs, ys])
                                vertex_coords_2d = triangulated_vertices[:, :2]
                            
                            if polygon_2d.size > 0 and len(polygon_2d) >= 3:
                                # Prüfe Vertices nahe dem Polygon-Rand
                                edge_distance_threshold = actual_resolution * 1.2  # Erhöht für bessere Erkennung
                                n_edges = len(polygon_2d)
                                edge_vertex_mask = np.zeros(len(vertex_coords_2d), dtype=bool)
                                distances_to_edge = []
                                for i in range(n_edges):
                                    p1 = polygon_2d[i]
                                    p2 = polygon_2d[(i + 1) % n_edges]
                                    edge_vec = p2 - p1
                                    edge_len = np.linalg.norm(edge_vec)
                                    if edge_len > 1e-9:
                                        edge_vec_norm = edge_vec / edge_len
                                        for j, v_coord in enumerate(vertex_coords_2d):
                                            if not edge_vertex_mask[j]:  # Noch nicht als Rand erkannt
                                                v_to_p1 = v_coord - p1
                                                t = np.clip(np.dot(v_to_p1, edge_vec_norm), 0.0, edge_len)
                                                closest_point_on_edge = p1 + t * edge_vec_norm
                                                dist_to_edge = np.linalg.norm(v_coord - closest_point_on_edge)
                                                distances_to_edge.append(dist_to_edge)
                                                if dist_to_edge <= edge_distance_threshold:
                                                    edge_vertex_mask[j] = True
                                
                                edge_vertices_in_triangulation = int(np.count_nonzero(edge_vertex_mask))
                                if len(distances_to_edge) > 0:
                                    min_dist_to_edge = float(np.min(distances_to_edge))
                                    max_dist_to_edge = float(np.max(distances_to_edge))
                                    mean_dist_to_edge = float(np.mean(distances_to_edge))
                                edge_detection_success = True
                        except Exception as e:
                            edge_detection_error = str(type(e).__name__) + ": " + str(e)
                    else:
                        edge_detection_error = f"too_few_points_{len(surface_points)}"
                
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "TRIANGULATION_COVERAGE",
                        "location": "FlexibleGridGenerator.generate_per_surface:triangulation_coverage",
                        "message": "Triangulation Coverage prüfen",
                        "data": {
                            "surface_id": str(geom.surface_id),
                            "triangulated_success": triangulated_success,
                            "total_vertices_in_triangulation": total_vertices_in_triangulation,
                            "edge_vertices_in_triangulation": edge_vertices_in_triangulation,
                            "triangulation_extends_to_edge": edge_vertices_in_triangulation > 0,
                            "mask_active_points": int(np.count_nonzero(surface_mask)),
                            "triangulation_uses_all_mask_points": total_vertices_in_triangulation >= int(np.count_nonzero(surface_mask)),
                            "edge_detection_success": edge_detection_success,
                            "edge_detection_error": edge_detection_error,
                            "surface_orientation": str(geom.orientation) if hasattr(geom, 'orientation') else None,
                            "dominant_axis": str(geom.dominant_axis) if hasattr(geom, 'dominant_axis') and geom.dominant_axis else None,
                            "min_dist_to_edge": min_dist_to_edge,
                            "max_dist_to_edge": max_dist_to_edge,
                            "mean_dist_to_edge": mean_dist_to_edge,
                            "resolution": float(actual_resolution)
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception as e:
                pass
            # #endregion
            
            # Speichere additional_vertices (wenn vorhanden)
            # 🎯 FIX: Verwende additional_vertices_array statt additional_vertices, da additional_vertices_array bereits als numpy-Array erstellt wurde
            # und additional_vertices möglicherweise überschrieben wurde
            additional_vertices_final = None
            
            # #region agent log - additional_vertices vor final-Zuweisung
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H5",
                        "location": "FlexibleGridGenerator.generate_per_surface:before_final_assignment",
                        "message": "additional_vertices vor final-Zuweisung",
                        "data": {
                            "surface_id": str(geom.surface_id),
                            "additional_vertices_count": int(len(additional_vertices)) if hasattr(additional_vertices, '__len__') else 0,
                            "additional_vertices_is_array": isinstance(additional_vertices, np.ndarray),
                            "additional_vertices_size": int(additional_vertices.size) if isinstance(additional_vertices, np.ndarray) else None,
                            "additional_vertices_shape": list(additional_vertices.shape) if isinstance(additional_vertices, np.ndarray) and additional_vertices.size > 0 else None,
                            "additional_vertices_array_exists": 'additional_vertices_array' in locals(),
                            "additional_vertices_array_count": int(len(additional_vertices_array)) if 'additional_vertices_array' in locals() and hasattr(additional_vertices_array, '__len__') else None,
                            "additional_vertices_array_is_array": isinstance(additional_vertices_array, np.ndarray) if 'additional_vertices_array' in locals() else None
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 FIX: Verwende additional_vertices_array statt additional_vertices
            # additional_vertices_array wurde bereits oben als numpy-Array erstellt und sollte korrekt sein
            if 'additional_vertices_array' in locals() and isinstance(additional_vertices_array, np.ndarray):
                if additional_vertices_array.size > 0:
                    additional_vertices_final = additional_vertices_array
                else:
                    additional_vertices_final = None
            elif isinstance(additional_vertices, np.ndarray):
                if additional_vertices.size > 0:
                    additional_vertices_final = additional_vertices
                else:
                    additional_vertices_final = None
            elif hasattr(additional_vertices, '__len__') and len(additional_vertices) > 0:
                additional_vertices_final = np.array(additional_vertices, dtype=float)
            else:
                additional_vertices_final = None
            
            # #region agent log - additional_vertices_final gesetzt
            try:
                import json, time as _t
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H5",
                        "location": "FlexibleGridGenerator.generate_per_surface:after_final_assignment",
                        "message": "additional_vertices_final gesetzt",
                        "data": {
                            "surface_id": str(geom.surface_id),
                            "additional_vertices_final_is_none": additional_vertices_final is None,
                            "additional_vertices_final_count": int(len(additional_vertices_final)) if additional_vertices_final is not None and hasattr(additional_vertices_final, '__len__') else 0,
                            "additional_vertices_final_shape": list(additional_vertices_final.shape) if additional_vertices_final is not None and isinstance(additional_vertices_final, np.ndarray) and additional_vertices_final.size > 0 else None
                        },
                        "timestamp": int(_t.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            # 🎯 OPTIMIERUNG: Hole vertex_source_indices (wurde nach Deduplizierung erstellt)
            vertex_source_indices_final = None
            if 'vertex_source_indices' in locals() and vertex_source_indices is not None:
                vertex_source_indices_final = vertex_source_indices
            else:
                vertex_source_indices_final = None
            
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
                vertex_source_indices=vertex_source_indices_final,
                additional_vertices=additional_vertices_final,
            )

            # 🎯 DEBUG: Prüfung auf überlappende Vertices in 3D
            # Idee:
            # - Vertices werden in allen drei Dimensionen (X, Y, Z) leicht quantisiert.
            # - Wenn mehrere Vertices auf derselben quantisierten 3D-Position liegen,
            #   werten wir das als „übereinander geplottet“.
            # - Zusätzlich geben wir eine grobe Spannweite der tatsächlichen Distanzen aus.
            try:
                if DEBUG_FLEXIBLE_GRID and triangulated_success and triangulated_vertices is not None:
                    verts = np.asarray(triangulated_vertices, dtype=float)
                    if verts.size > 0:
                        # Toleranz: ca. 1 % der effektiven Grid-Resolution, sonst fixer kleiner Wert
                        if actual_resolution and actual_resolution > 0:
                            tol = float(actual_resolution) * 0.01
                        else:
                            tol = 1e-3
                        # Quantisierte 3D-Koordinaten (reduziert numerisches Rauschen)
                        quant = np.round(verts / tol) * tol
                        from collections import defaultdict
                        groups: dict[tuple[float, float, float], list[int]] = defaultdict(list)
                        for idx_v, (qx, qy, qz) in enumerate(quant):
                            groups[(float(qx), float(qy), float(qz))].append(int(idx_v))

                        overlap_positions = 0
                        max_local_span = 0.0
                        # Optional: Beispielkoordinaten sammeln (max. 3)
                        example_positions: list[tuple[float, float, float]] = []

                        for (qx, qy, qz), idx_list in groups.items():
                            if len(idx_list) <= 1:
                                continue
                            overlap_positions += 1
                            # Berechne maximale euklidische Distanz innerhalb dieser Gruppe
                            local_verts = verts[idx_list]
                            local_max = 0.0
                            if len(local_verts) > 1:
                                for i in range(len(local_verts)):
                                    for j in range(i + 1, len(local_verts)):
                                        d = float(np.linalg.norm(local_verts[i] - local_verts[j]))
                                        if d > local_max:
                                            local_max = d
                            if local_max > max_local_span:
                                max_local_span = local_max
                            if len(example_positions) < 3:
                                example_positions.append((qx, qy, qz))

                        if overlap_positions > 0:
                            print(
                                "[DEBUG VertexOverlap] Surface "
                                f"'{geom.surface_id}': {overlap_positions} 3D-Positionen "
                                f"mit mehreren Vertices (max. Distanz innerhalb Gruppe="
                                f"{max_local_span:.3f} m)."
                            )
                            for pos in example_positions:
                                print(
                                    f"  └─ Beispiel-Position: X={pos[0]:.3f}, "
                                    f"Y={pos[1]:.3f}, Z={pos[2]:.3f}"
                                )
            except Exception:
                # Reiner Debug-Helper – darf die Hauptlogik nicht beeinflussen
                pass
            
            # Im Cache speichern
            self._surface_grid_cache[cache_key] = surface_grid
            surface_grids[geom.surface_id] = surface_grid
            
            # #region agent debug plot - 2D Debug Plot für Surface
            try:
                self._create_debug_plot_2d(
                    geom=geom,
                    X_grid=X_grid,
                    Y_grid=Y_grid,
                    Z_grid=Z_grid,
                    surface_mask=surface_mask,
                    triangulated_vertices=triangulated_vertices,
                    triangulated_faces=triangulated_faces,
                    resolution=actual_resolution
                )
            except Exception as e:
                # Debug-Plot-Fehler dürfen die Hauptlogik nicht stören
                pass
            # #endregion
            
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
                            pass  # Debug-Warnung entfernt
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
            else:
                pass
        
        return surface_grids
    
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
    
    def _create_debug_plot_2d(
        self,
        geom: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        triangulated_vertices: Optional[np.ndarray],
        triangulated_faces: Optional[np.ndarray],
        resolution: float
        ) -> None:
        """
        Erstellt einen 2D-Debug-Plot für ein Surface:
        - Surface-Umrandung (Polygon)
        - Grid-Punkte innerhalb und außerhalb der Maske
        - Triangulated Vertices (falls vorhanden)
        - Triangulated Faces (Dreiecke als Linien)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return  # matplotlib nicht verfügbar
        
        try:
            # Bestimme 2D-Koordinaten basierend auf Orientierung
            surface_points = geom.points or []
            if len(surface_points) < 3:
                return
            
            if geom.orientation == "vertical":
                # Vertikale Flächen: verwende (u,v) basierend auf dominant_axis
                xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                zs = np.array([p.get("z", 0.0) for p in surface_points], dtype=float)
                if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                    if geom.dominant_axis == "yz":
                        # Y-Z-Wand: u = y, v = z
                        polygon_2d = np.column_stack([ys, zs])
                        grid_x_2d = Y_grid.ravel()
                        grid_y_2d = Z_grid.ravel()
                        x_label = "Y [m]"
                        y_label = "Z [m]"
                        if triangulated_vertices is not None and triangulated_vertices.size > 0:
                            verts_2d = triangulated_vertices[:, [1, 2]]
                        else:
                            verts_2d = None
                    else:  # xz
                        # X-Z-Wand: u = x, v = z
                        polygon_2d = np.column_stack([xs, zs])
                        grid_x_2d = X_grid.ravel()
                        grid_y_2d = Z_grid.ravel()
                        x_label = "X [m]"
                        y_label = "Z [m]"
                        if triangulated_vertices is not None and triangulated_vertices.size > 0:
                            verts_2d = triangulated_vertices[:, [0, 2]]
                        else:
                            verts_2d = None
                else:
                    # Fallback: verwende xz
                    polygon_2d = np.column_stack([xs, zs])
                    grid_x_2d = X_grid.ravel()
                    grid_y_2d = Z_grid.ravel()
                    x_label = "X [m]"
                    y_label = "Z [m]"
                    if triangulated_vertices is not None and triangulated_vertices.size > 0:
                        verts_2d = triangulated_vertices[:, [0, 2]]
                    else:
                        verts_2d = None
            else:
                # Planare/sloped: verwende (x,y)
                xs = np.array([p.get("x", 0.0) for p in surface_points], dtype=float)
                ys = np.array([p.get("y", 0.0) for p in surface_points], dtype=float)
                polygon_2d = np.column_stack([xs, ys])
                grid_x_2d = X_grid.ravel()
                grid_y_2d = Y_grid.ravel()
                x_label = "X [m]"
                y_label = "Y [m]"
                if triangulated_vertices is not None and triangulated_vertices.size > 0:
                    verts_2d = triangulated_vertices[:, :2]
                else:
                    verts_2d = None
            
            # Erstelle Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Schließe Polygon für Plot
            polygon_closed = np.vstack([polygon_2d, polygon_2d[0:1]])
            
            # Zeichne Surface-Umrandung
            ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', linewidth=2, label='Surface-Umrandung', zorder=5)
            ax.plot(polygon_2d[:, 0], polygon_2d[:, 1], 'ko', markersize=6, label='Polygon-Ecken', zorder=6)
            
            # Grid-Punkte: innerhalb und außerhalb Maske
            mask_flat = surface_mask.ravel()
            grid_inside = mask_flat
            grid_outside = ~mask_flat
            
            if np.any(grid_inside):
                ax.scatter(grid_x_2d[grid_inside], grid_y_2d[grid_inside], 
                          c='green', s=15, alpha=0.6, label='Grid innerhalb Maske', zorder=2)
            
            if np.any(grid_outside):
                ax.scatter(grid_x_2d[grid_outside], grid_y_2d[grid_outside], 
                          c='red', s=15, alpha=0.3, label='Grid außerhalb Maske', zorder=1)
            
            # Triangulated Vertices (falls vorhanden)
            if verts_2d is not None and verts_2d.size > 0:
                ax.scatter(verts_2d[:, 0], verts_2d[:, 1], 
                          c='blue', s=20, alpha=0.8, marker='x', linewidths=1.5, 
                          label='Triangulated Vertices', zorder=3)
            
            # Triangulated Faces (Dreiecke als Linien zeichnen)
            if (triangulated_faces is not None and triangulated_faces.size > 0 and 
                verts_2d is not None and verts_2d.size > 0):
                try:
                    # Parse Faces im PyVista-Format [n, v1, v2, v3, n, v4, v5, v6, ...]
                    faces_flat = np.asarray(triangulated_faces, dtype=np.int64)
                    n_faces = len(faces_flat) // 4  # 4 Elemente pro Face
                    
                    # Zeichne jedes Dreieck als 3 Linien
                    for i in range(n_faces):
                        idx = i * 4
                        if idx + 3 < len(faces_flat) and faces_flat[idx] == 3:
                            v1_idx = int(faces_flat[idx + 1])
                            v2_idx = int(faces_flat[idx + 2])
                            v3_idx = int(faces_flat[idx + 3])
                            
                            # Prüfe ob Indizes gültig sind
                            if (0 <= v1_idx < len(verts_2d) and 
                                0 <= v2_idx < len(verts_2d) and 
                                0 <= v3_idx < len(verts_2d)):
                                # Zeichne 3 Kanten des Dreiecks
                                triangle_points = verts_2d[[v1_idx, v2_idx, v3_idx, v1_idx]]
                                ax.plot(triangle_points[:, 0], triangle_points[:, 1], 
                                       'b-', linewidth=0.5, alpha=0.3, zorder=4)
                except Exception:
                    # Fehler beim Zeichnen der Faces sollten den Plot nicht verhindern
                    pass
            
            # Layout
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'Debug Plot: {geom.surface_id}\nOrientierung: {geom.orientation}, Resolution: {resolution:.3f} m', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Speichere Plot
            debug_plot_dir = '/Users/MGraf/Python/LFO_Umgebung/.cursor/debug_plots'
            os.makedirs(debug_plot_dir, exist_ok=True)
            plot_filename = os.path.join(debug_plot_dir, f"surface_debug_{geom.surface_id}.png")
            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            # Fehler beim Plot-Erstellen sollten die Hauptlogik nicht beeinflussen
            pass

