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
            # Debug-Logging entfernt
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
            # Debug-Logging entfernt
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
            
            # Debug-Logging entfernt
            
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
            
            surface_masks[geom.surface_id] = mask.flatten()  # Flach für BaseGrid
        
        # 🎯 Adaptive Resolution für kleine Flächen (nur intern, ohne Logging)
        # Hinweis: small_surfaces bleibt verfügbar, falls später eine adaptive
        # Auflösung implementiert werden soll.
        
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
            
            try:
                from matplotlib.path import Path
                polygon_path = Path(np.column_stack([x_vertices, y_vertices]))
            except Exception:
                polygon_path = None
            
            # 🎯 FIX: Prüfe ob x_flat/y_flat leer sind (z.B. bei sehr schmalen Grids)
            if x_flat.size == 0 or y_flat.size == 0:
                return surface_mask_strict
            
            for xv, yv in zip(x_vertices, y_vertices):
                # Distanz zu allen Grid-Zentren
                dx = x_flat - xv
                dy = y_flat - yv
                d2 = dx * dx + dy * dy
                # 🎯 FIX: Prüfe ob d2 leer ist (sollte nicht passieren, aber sicherheitshalber)
                if d2.size == 0:
                    continue
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
                        # 🎯 FIX: Aktiviere Grid-Punkte außerhalb des Polygons NICHT
                        continue
                
                # Markiere diesen Grid-Punkt als "inside", falls noch nicht gesetzt
                # UND nur wenn er innerhalb des Polygons liegt
                if not mask_flat[nearest_idx] and is_inside:
                    mask_flat[nearest_idx] = True
            
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
            
            # 🎯 FIX: Prüfe ob u_flat/v_flat leer sind (z.B. bei sehr schmalen Grids)
            if u_flat.size == 0 or v_flat.size == 0:
                return surface_mask_strict
            
            for uv, vv in zip(u_vertices, v_vertices):
                # Distanz zu allen Grid-Zentren
                du = u_flat - uv
                dv = v_flat - vv
                d2 = du * du + dv * dv
                # 🎯 FIX: Prüfe ob d2 leer ist (sollte nicht passieren, aber sicherheitshalber)
                if d2.size == 0:
                    continue
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
            
            edge_generation_stats = []
            
            for i in range(n_vertices):
                x1, y1 = px[i], py[i]
                x2, y2 = px[(i + 1) % n_vertices], py[(i + 1) % n_vertices]
                
                # Kante: Von (x1, y1) zu (x2, y2)
                dx = x2 - x1
                dy = y2 - y1
                segment_len = math.hypot(dx, dy)
                
                
                if segment_len < 1e-9:
                    # Degenerierte Kante überspringen
                    continue
                
                # 🎯 FIX: Generiere Punkte mit exaktem Abstand point_spacing entlang der Kante
                # Starte bei t=0 und füge Punkte hinzu, bis das Ende erreicht ist
                edge_points_before = len(edge_points)
                point_positions = []
                
                # Generiere Punkte mit exaktem Abstand point_spacing
                current_t = 0.0
                while current_t <= 1.0:
                    x_edge = x1 + current_t * dx
                    y_edge = y1 + current_t * dy
                    edge_points.append((x_edge, y_edge))
                    
                    point_positions.append({"t": float(current_t), "x": float(x_edge), "y": float(y_edge)})
                    
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
                    
                    point_positions.append({"t": float(current_t), "x": float(x_edge), "y": float(y_edge)})
                
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
            
            for x, y in edge_points:
                is_duplicate = False
                for ex, ey in unique_edge_points:
                    if abs(x - ex) < tolerance and abs(y - ey) < tolerance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_edge_points.append((x, y))
                else:
                    duplicates_removed += 1
            
            edge_points = unique_edge_points
            
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
            
            
            if not edge_points:
                return surface_mask_strict, []
            
            # 🎯 NEUE LOGIK: Randpunkte werden direkt als additional_vertices verwendet
            # KEINE Mapping auf Grid-Punkte mehr
            
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
            
            points_after_dedup = len(edge_points)
            
            if not edge_points:
                return surface_mask_strict, []
            
            # 🎯 NEUE LOGIK: Randpunkte werden direkt als additional_vertices verwendet
            # KEINE Mapping auf Grid-Punkte mehr
            # Rückgabe: Unveränderte Maske + Liste der Randpunkte
            return surface_mask_strict, edge_points
        except Exception:
            # Rein heuristische Verbesserung – Fehler dürfen die Hauptlogik nicht stören
            return surface_mask_strict
    
    def _ensure_at_least_one_point(
        self,
        geometry: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        surface_mask_strict: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stellt sicher, dass mindestens ein Grid-Punkt innerhalb des Surface aktiviert ist.
        Wenn surface_mask leer ist, wird der nächstgelegene Grid-Punkt ZUM ZENTROID aktiviert,
        der sich INNERHALB des Polygons befindet.
        
        Returns:
            (surface_mask, surface_mask_strict) - aktualisierte Masken
        """
        # Prüfe ob surface_mask leer ist
        if np.any(surface_mask):
            # Maske ist nicht leer, nichts zu tun
            return surface_mask, surface_mask_strict
        
        # Maske ist leer - aktiviere nächstgelegenen Grid-Punkt INNERHALB des Polygons zum Zentroid
        points = geometry.points
        if len(points) < 3:
            # Zu wenige Punkte, kann keinen Zentroid berechnen
            return surface_mask, surface_mask_strict
        
        # Berechne Zentroid des Polygons (einfaches arithmetisches Mittel)
        centroid_x = float(np.mean([p.get('x', 0.0) for p in points]))
        centroid_y = float(np.mean([p.get('y', 0.0) for p in points]))
        
        # Für vertikale Surfaces: Berechne Zentroid in (u,v)-Koordinaten
        if geometry.orientation == "vertical":
            dominant_axis = getattr(geometry, 'dominant_axis', None)
            if dominant_axis == "xz":
                # X-Z-Wand: u = x, v = z
                centroid_u = centroid_x
                centroid_v = float(np.mean([p.get('z', 0.0) for p in points]))
                U_grid = X_grid
                V_grid = Z_grid
                polygon_uv = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
            elif dominant_axis == "yz":
                # Y-Z-Wand: u = y, v = z
                centroid_u = centroid_y
                centroid_v = float(np.mean([p.get('z', 0.0) for p in points]))
                U_grid = Y_grid
                V_grid = Z_grid
                polygon_uv = [
                    {"x": float(p.get("y", 0.0)), "y": float(p.get("z", 0.0))}
                    for p in points
                ]
            else:
                # Fallback: Verwende X-Y-Zentroid
                centroid_u = centroid_x
                centroid_v = centroid_y
                U_grid = X_grid
                V_grid = Y_grid
                polygon_uv = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))}
                    for p in points
                ]
            
            # Prüfe welche Grid-Punkte innerhalb des Polygons liegen (in u,v-Koordinaten)
            points_inside_mask = self._points_in_polygon_batch_uv(U_grid, V_grid, polygon_uv)
            
            if not np.any(points_inside_mask):
                # Keine Grid-Punkte innerhalb - verwende Fallback: nächstgelegenen Grid-Punkt zu einem Polygon-Punkt
                # Wähle den ersten Polygon-Punkt als Referenz
                ref_u = polygon_uv[0]["x"]
                ref_v = polygon_uv[0]["y"]
                u_flat = U_grid.ravel()
                v_flat = V_grid.ravel()
                distances_sq = (u_flat - ref_u) ** 2 + (v_flat - ref_v) ** 2
                nearest_idx = int(np.argmin(distances_sq))
            else:
                # Berechne Distanzen NUR für Punkte innerhalb des Polygons
                u_flat = U_grid.ravel()
                v_flat = V_grid.ravel()
                points_inside_flat = points_inside_mask.ravel()
                
                # Nur Distanzen für Punkte innerhalb berechnen
                distances_sq = np.full_like(u_flat, np.inf, dtype=float)
                distances_sq[points_inside_flat] = (
                    (u_flat[points_inside_flat] - centroid_u) ** 2 + 
                    (v_flat[points_inside_flat] - centroid_v) ** 2
                )
                nearest_idx = int(np.argmin(distances_sq))
            
            # Aktiviere nächstgelegenen Punkt
            nearest_j, nearest_i = np.unravel_index(nearest_idx, X_grid.shape)
            surface_mask[nearest_j, nearest_i] = True
            surface_mask_strict[nearest_j, nearest_i] = True
            
        else:
            # Planare/schräge Surfaces: Verwende X-Y-Koordinaten
            # Prüfe welche Grid-Punkte innerhalb des Polygons liegen
            points_inside_mask = self._points_in_polygon_batch(X_grid, Y_grid, points)
            
            if not np.any(points_inside_mask):
                # Keine Grid-Punkte innerhalb - verwende Fallback: nächstgelegenen Grid-Punkt zu einem Polygon-Punkt
                # Wähle den ersten Polygon-Punkt als Referenz
                ref_x = float(points[0].get('x', 0.0))
                ref_y = float(points[0].get('y', 0.0))
                x_flat = X_grid.ravel()
                y_flat = Y_grid.ravel()
                distances_sq = (x_flat - ref_x) ** 2 + (y_flat - ref_y) ** 2
                nearest_idx = int(np.argmin(distances_sq))
            else:
                # Berechne Distanzen NUR für Punkte innerhalb des Polygons
                x_flat = X_grid.ravel()
                y_flat = Y_grid.ravel()
                points_inside_flat = points_inside_mask.ravel()
                
                # Nur Distanzen für Punkte innerhalb berechnen
                distances_sq = np.full_like(x_flat, np.inf, dtype=float)
                distances_sq[points_inside_flat] = (
                    (x_flat[points_inside_flat] - centroid_x) ** 2 + 
                    (y_flat[points_inside_flat] - centroid_y) ** 2
                )
                nearest_idx = int(np.argmin(distances_sq))
            
            # Aktiviere nächstgelegenen Punkt
            nearest_j, nearest_i = np.unravel_index(nearest_idx, X_grid.shape)
            surface_mask[nearest_j, nearest_i] = True
            surface_mask_strict[nearest_j, nearest_i] = True
            
            # Berechne Z-Koordinate für diesen Punkt (falls plane_model verfügbar)
            if geometry.plane_model:
                from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import evaluate_surface_plane
                nearest_x = float(X_grid[nearest_j, nearest_i])
                nearest_y = float(Y_grid[nearest_j, nearest_i])
                nearest_z = evaluate_surface_plane(geometry.plane_model, nearest_x, nearest_y)
                Z_grid[nearest_j, nearest_i] = nearest_z
        
        return surface_mask, surface_mask_strict
    
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
        
        Orchestriert drei klar getrennte Schritte:
          1) _calculate_additional_vertices_collect_grid_points
          2) _calculate_additional_vertices_add_corners
          3) _calculate_additional_vertices_add_boundary_points
          4) _calculate_additional_vertices_deduplicate_and_merge
        
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
        surface_points = geometry.points or []
        if len(surface_points) < 3:
            return np.array([], dtype=float).reshape(0, 3)
        
        # 1. Sammle Grid-Punkte für Duplikat-Prüfung
        all_vertices = self._calculate_additional_vertices_collect_grid_points(
            X_grid, Y_grid, Z_grid, surface_mask
        )
        
        # 2. Ecken-Punkte hinzufügen
        additional_vertices = self._calculate_additional_vertices_add_corners(
            geometry, surface_points, all_vertices
        )
        
        # 3. Randpunkte hinzufügen
        point_spacing = self._calculate_additional_vertices_get_point_spacing(
            resolution, X_grid, Y_grid
        )
        
        boundary_edge_points = self._calculate_additional_vertices_add_boundary_points(
            geometry, surface_points, point_spacing
        )
        
        # 4. Duplikat-Prüfung und Zusammenführung
        additional_vertices = self._calculate_additional_vertices_deduplicate_and_merge(
            additional_vertices, boundary_edge_points, all_vertices, point_spacing
        )
        
        return np.array(additional_vertices, dtype=float) if len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
    
    # ------------------------------------------------------------------
    # Hilfsmethoden für _calculate_additional_vertices (klar zugeordnet über Präfix)
    # ------------------------------------------------------------------
    
    def _calculate_additional_vertices_collect_grid_points(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        ) -> np.ndarray:
        """Sammelt alle aktiven Grid-Punkte für Duplikat-Prüfung."""
        grid_points_flat = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        mask_flat = surface_mask.reshape(-1)
        return grid_points_flat[mask_flat] if np.any(mask_flat) else np.array([], dtype=float).reshape(0, 3)
    
    def _calculate_additional_vertices_add_corners(
        self,
        geometry: SurfaceGeometry,
        surface_points: List[Dict[str, float]],
        all_vertices: np.ndarray,
        ) -> List[List[float]]:
        """Fügt Ecken-Punkte hinzu (vertikal oder planar/sloped)."""
        additional_vertices = []
        
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
                # 🎯 FIX: Prüfe ob all_vertices leer ist (z.B. bei schmalen Surfaces ohne Grid-Punkte)
                if all_vertices.size == 0:
                    # Keine Grid-Punkte vorhanden - füge Eckpunkt direkt hinzu
                    if is_xz_wall:
                        corner_x = corner_u
                        corner_y = y_mean
                        corner_z = corner_v
                    else:
                        corner_x = x_mean
                        corner_y = corner_u
                        corner_z = corner_v
                    additional_vertices.append([corner_x, corner_y, corner_z])
                    continue
                
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
                # 🎯 FIX: Prüfe ob all_vertices leer ist (z.B. bei schmalen Surfaces ohne Grid-Punkte)
                if all_vertices.size == 0:
                    # Keine Grid-Punkte vorhanden - füge Eckpunkt direkt hinzu
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
                    continue
                
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
        
        return additional_vertices
    
    def _calculate_additional_vertices_get_point_spacing(
        self,
        resolution: Optional[float],
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        ) -> float:
        """Bestimmt Punktabstand für Randpunkte (mit Fallback)."""
        if resolution is None or resolution <= 0:
            # Fallback: Verwende Grid-Auflösung aus X/Y-Grid
            if X_grid.size > 1 and Y_grid.size > 1:
                x_diffs = np.diff(np.unique(X_grid.ravel()))
                y_diffs = np.diff(np.unique(Y_grid.ravel()))
                if x_diffs.size > 0 and y_diffs.size > 0:
                    return min(float(np.mean(x_diffs)), float(np.mean(y_diffs)))
            return 0.5  # Fallback
        return resolution
    
    def _calculate_additional_vertices_add_boundary_points(
        self,
        geometry: SurfaceGeometry,
        surface_points: List[Dict[str, float]],
        point_spacing: float,
        ) -> List[List[float]]:
        """Fügt Randpunkte entlang der Polygon-Kanten hinzu."""
        boundary_edge_points = []
        
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
        
        return boundary_edge_points
    
    def _calculate_additional_vertices_deduplicate_and_merge(
        self,
        additional_vertices: List[List[float]],
        boundary_edge_points: List[List[float]],
        all_vertices: np.ndarray,
        point_spacing: float,
        ) -> List[List[float]]:
        """Prüft Duplikate und fügt Randpunkte zu additional_vertices hinzu."""
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
        
        return additional_vertices
    
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
        
        Orchestriert mehrere klar getrennte Schritte:
          1) build_single_surface_grid_create_base_grid
          2) build_single_surface_grid_create_mask
          3) build_single_surface_grid_interpolate_z
          4) build_single_surface_grid_calculate_additional_vertices
        
        Args:
            geometry: SurfaceGeometry Objekt
            resolution: Basis-Resolution in Metern (wenn None: settings.resolution)
            min_points_per_dimension: Mindestanzahl Punkte pro Dimension (Standard: 3)
            padding_factor: Padding-Faktor für Bounding Box
        
        Returns:
            Tuple von (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)
        """
        # Stelle sicher, dass resolution gesetzt ist
        if resolution is None:
            resolution = self.settings.resolution
        
        t_orientation_start = time.perf_counter() if PERF_ENABLED else None
        
        # 1. Erstelle Basis-Grid (X_grid, Y_grid, Z_grid, sound_field_x, sound_field_y)
        (X_grid, Y_grid, Z_grid, sound_field_x, sound_field_y) = self.build_single_surface_grid_create_base_grid(
            geometry, resolution
        )
        
        # 2. Erstelle Maske (surface_mask, surface_mask_strict)
        (surface_mask, surface_mask_strict) = self.build_single_surface_grid_create_mask(
            geometry, X_grid, Y_grid, Z_grid
        )
        
        # 3. Interpoliere Z-Koordinaten (für planare/schräge Surfaces)
        Z_grid = self.build_single_surface_grid_interpolate_z(
            geometry, X_grid, Y_grid, Z_grid
        )
        
        # 4. Berechne additional_vertices
        additional_vertices = self.build_single_surface_grid_calculate_additional_vertices(
            geometry, X_grid, Y_grid, Z_grid, surface_mask, resolution
        )
        
        # Rückgabe
        return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)
    
    # ------------------------------------------------------------------
    # Hilfsmethoden für build_single_surface_grid (klar zugeordnet über Präfix)
    # ------------------------------------------------------------------
    
    def build_single_surface_grid_create_base_grid(
        self,
        geometry: SurfaceGeometry,
        resolution: float,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Erstellt Basis-Grid (X_grid, Y_grid, Z_grid, sound_field_x, sound_field_y)."""
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
                                    # Debug-Ausgabe entfernt
                        
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
        
        return (X_grid, Y_grid, Z_grid, sound_field_x, sound_field_y)
    
    def build_single_surface_grid_create_mask(
        self,
        geometry: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Erstellt Maske für Grid-Punkte (surface_mask, surface_mask_strict)."""
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
                # 🎯 SICHERSTELLEN, DASS POLYGON-ECKEN ERFASST WERDEN (in u,v-Koordinaten)
                surface_mask_strict = self._ensure_vertex_coverage_uv(
                    U_grid, V_grid, polygon_uv, surface_mask_strict
                )
                # 🎯 KEINE DILATATION MEHR: Verwende nur strikte Maske für Berechnung
                # Randpunkte werden jetzt direkt auf Surface-Linie berechnet und als additional_vertices hinzugefügt
                # surface_mask_strict wird für Plot verwendet (nur Punkte innerhalb)
                # surface_mask wird für Berechnung verwendet (nur Punkte innerhalb, keine Dilatation mehr)
                surface_mask = surface_mask_strict.copy()  # Keine Dilatation mehr - Randpunkte werden separat hinzugefügt
                
                # 🎯 NEU: Stelle sicher, dass mindestens ein Punkt aktiviert ist
                surface_mask, surface_mask_strict = self._ensure_at_least_one_point(
                    geometry, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict
                )
                
                
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
                
                # 🎯 NEU: Stelle sicher, dass mindestens ein Punkt aktiviert ist
                surface_mask, surface_mask_strict = self._ensure_at_least_one_point(
                    geometry, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict
                )
                
                
            
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
            
            # 🎯 NEU: Stelle sicher, dass mindestens ein Punkt aktiviert ist
            surface_mask, surface_mask_strict = self._ensure_at_least_one_point(
                geometry, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict
            )
            
            # 🎯 DEBUG: Grid-Erweiterung (vor Z-Interpolation, damit total_grid_points verfügbar ist)
            total_grid_points = X_grid.size
            points_in_surface = np.count_nonzero(surface_mask)
            points_outside_surface = total_grid_points - points_in_surface
        
        return (surface_mask, surface_mask_strict)
    
    def build_single_surface_grid_interpolate_z(
        self,
        geometry: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        ) -> np.ndarray:
        """Interpoliert Z-Koordinaten für planare/schräge Surfaces."""
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
                
                # Debug-Ausgaben entfernt
                
                if z_variation:
                    # 🎯 VERTIKALE SURFACES: Z_grid ist bereits korrekt gesetzt, überspringe Interpolation
                    if geometry.orientation == "vertical":
                        # Für vertikale Flächen wird Z_grid bereits beim Grid-Aufbau korrekt gesetzt.
                        pass

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
                                    
                                    # Debug-Ausgaben entfernt
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
                                    
                                    # Debug-Ausgaben entfernt
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
                            # Debug-Ausgabe entfernt
                        except Exception as e:
                            raise RuntimeError(f"Z-Interpolation (linear) fehlgeschlagen: {e}")
                else:
                    # Alle Surface-Punkte haben gleichen Z-Wert
                    Z_grid.fill(surface_z[0])
                    # print(f"  └─ Z-Werte auf konstanten Wert {surface_z[0]:.3f} gesetzt für ALLE {total_grid_points} Punkte")
        
        return Z_grid
    
    def build_single_surface_grid_calculate_additional_vertices(
        self,
        geometry: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        resolution: float,
        ) -> np.ndarray:
        """Berechnet additional_vertices (Ecken + Randpunkte)."""
        return self._calculate_additional_vertices(
            geometry, X_grid, Y_grid, Z_grid, surface_mask, resolution
        )


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
            
            mode = geom.plane_model.get('mode', 'unknown')
            points_in_polygon = int(np.sum(mask))
            z_values_in_polygon = Z_surface[mask]
            z_min = float(z_values_in_polygon.min()) if len(z_values_in_polygon) > 0 else 0.0
            z_max = float(z_values_in_polygon.max()) if len(z_values_in_polygon) > 0 else 0.0
            z_mean = float(z_values_in_polygon.mean()) if len(z_values_in_polygon) > 0 else 0.0
            z_non_zero = int(np.sum(np.abs(z_values_in_polygon) > 1e-6))
            
            # Speichere Beitrag (nur für Punkte innerhalb des Polygons)
            z_contributions[geom.surface_id] = (mask, Z_surface)
            z_counts[mask] += 1.0
            processed_surfaces += 1
        
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
        
        # 🔍 Interne Analyse für halbierte Z-Werte (ohne Logging)
        points_with_overlap = int(np.sum(z_counts > 1.0))
        if points_with_overlap > 0:
            pass
            
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
                    
                                # Debug-Ausgabe entfernt
                    
                    # Zeige Beiträge von jeder Surface
                    for surface_id, (mask, Z_surface) in z_contributions.items():
                        if mask[jj, ii]:
                            z_contrib = Z_surface[jj, ii]
                                # Debug-Ausgabe entfernt
        
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
                        # Debug-Ausgaben entfernt – nur interne Konsistenzprüfung
                        if abs(z_final - expected_ratio) < abs(z_final - z_from_surface) * 0.1:
                            break
        
        # Debug: Statistiken
        valid_mask = np.abs(Z_grid) > 1e-6  # Punkte mit Z≠0
        points_with_z = int(np.sum(valid_mask))
        z_non_zero = points_with_z
        z_range = (float(Z_grid[valid_mask].min()), float(Z_grid[valid_mask].max())) if np.any(valid_mask) else (0.0, 0.0)
        
        # Ergebnis wird intern verwendet, ohne Logging-Ausgabe
        
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
        
        Orchestriert mehrere klar getrennte Schritte:
          1) generate_per_surface_check_cache
          2) generate_per_surface_build_grids_parallel
          3) generate_per_surface_process_and_create_grids
        
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
        
        # Schritt 1: Cache-Prüfung
        surface_grids, geometries_to_process = self.generate_per_surface_check_cache(
            geometries, resolution, min_points_per_dimension
        )
        
        # Schritt 2: Parallele Grid-Erstellung
        grid_results = self.generate_per_surface_build_grids_parallel(
            geometries_to_process, resolution, min_points_per_dimension, disable_edge_refinement
        )
        
        # Schritt 3: Ergebnisse verarbeiten und SurfaceGrids erstellen
        self.generate_per_surface_process_and_create_grids(
            geometries, surface_grids, grid_results, resolution, min_points_per_dimension
        )
        
        return surface_grids
    
    # ------------------------------------------------------------------
    # Hilfsmethoden für generate_per_surface (klar zugeordnet über Präfix)
    # ------------------------------------------------------------------
    
    def generate_per_surface_check_cache(
        self,
        geometries: List[SurfaceGeometry],
        resolution: float,
        min_points_per_dimension: int,
        ) -> Tuple[Dict[str, SurfaceGrid], List[Tuple[SurfaceGeometry, tuple]]]:
        """Prüft Cache und gibt bereits gecachte Grids zurück."""
        surface_grids: Dict[str, SurfaceGrid] = {}
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
        
        return (surface_grids, geometries_to_process)
    
    def generate_per_surface_build_grids_parallel(
        self,
        geometries_to_process: List[Tuple[SurfaceGeometry, tuple]],
        resolution: float,
        min_points_per_dimension: int,
        disable_edge_refinement: bool,
        ) -> Dict[str, Tuple[tuple, Any]]:
        """Baut Grids parallel für nicht-gecachte Surfaces."""
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
                        error_message = str(e)
                        # Hinweis-Ausgabe entfernt

            # Performance-Messung: Parallelaufbau der Single-Surface-Grids
            if PERF_ENABLED and t_parallel_start is not None:
                duration_ms = (time.perf_counter() - t_parallel_start) * 1000.0
                print(
                    f"[PERF] FlexibleGridGenerator.generate_per_surface.parallel_build: "
                    f"{duration_ms:.2f} ms [n_surfaces={len(geometries_to_process)}]"
                )
        
        return grid_results
    
    def generate_per_surface_process_and_create_grids(
        self,
        geometries: List[SurfaceGeometry],
        surface_grids: Dict[str, SurfaceGrid],
        grid_results: Dict[str, Tuple[tuple, Any]],
        resolution: float,
        min_points_per_dimension: int,
        ) -> None:
        """
        Verarbeitet Grid-Ergebnisse und erstellt SurfaceGrids.
        
        Orchestriert mehrere klar getrennte Schritte:
          1) generate_per_surface_process_and_create_grids_extract_result
          2) generate_per_surface_process_and_create_grids_correct_z
          3) generate_per_surface_process_and_create_grids_create_triangulation
          4) generate_per_surface_process_and_create_grids_create_surface_grid
        """
        for geom in geometries:
            # Wenn bereits aus dem Cache kam, ist alles erledigt
            if geom.surface_id in surface_grids:
                continue

            entry = grid_results.get(geom.surface_id)
            if not entry:
                # Wurde übersprungen oder ist fehlgeschlagen
                continue
            cache_key, result = entry
            
            # 1. Extrahiere Grid-Ergebnis
            grid_data = self.generate_per_surface_process_and_create_grids_extract_result(result)
            if grid_data is None:
                continue
            (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices) = grid_data
            
            # 2. Korrigiere Z-Koordinaten
            Z_grid = self.generate_per_surface_process_and_create_grids_correct_z(
                geom, X_grid, Y_grid, Z_grid
            )
            
            # 3. Berechne tatsächliche Resolution
            actual_resolution = self.generate_per_surface_process_and_create_grids_calculate_resolution(
                sound_field_x, sound_field_y, resolution
            )
            
            # 4. Erstelle Triangulation
            triangulation_data = self.generate_per_surface_process_and_create_grids_create_triangulation(
                geom, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices, actual_resolution
            )
            
            # 5. Erstelle SurfaceGrid und fülle Cache
            self.generate_per_surface_process_and_create_grids_create_surface_grid(
                geom, cache_key, surface_grids,
                sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid,
                surface_mask, surface_mask_strict, actual_resolution,
                triangulation_data
            )
    
    # ------------------------------------------------------------------
    # Hilfsmethoden für generate_per_surface_process_and_create_grids
    # ------------------------------------------------------------------
    
    def generate_per_surface_process_and_create_grids_extract_result(
        self,
        result: Tuple,
        ) -> Optional[Tuple]:
        """Extrahiert Grid-Ergebnis mit Fallback-Unterstützung."""
        try:
            if len(result) == 8:
                # Version: Mit strikter Maske + additional_vertices
                return result
            elif len(result) == 7:
                # Version: Mit strikter Maske (ohne additional_vertices - Fallback)
                (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict) = result
                additional_vertices = np.array([], dtype=float).reshape(0, 3)
                return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)
            else:
                # Alte Version: Ohne strikte Maske (Fallback)
                (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask) = result
                surface_mask_strict = surface_mask  # Fallback: identisch
                additional_vertices = np.array([], dtype=float).reshape(0, 3)
                return (sound_field_x, sound_field_y, X_grid, Y_grid, Z_grid, surface_mask, surface_mask_strict, additional_vertices)
        except Exception:
            return None
    
    def generate_per_surface_process_and_create_grids_correct_z(
        self,
        geom: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        ) -> np.ndarray:
        """Korrigiert Z-Koordinaten bei planaren/schrägen Flächen aus der Ebene."""
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
        
        return Z_grid
    
    def generate_per_surface_process_and_create_grids_calculate_resolution(
        self,
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
        resolution: float,
        ) -> float:
        """Berechnet tatsächliche Resolution (kann adaptiv sein)."""
        if len(sound_field_x) > 1 and len(sound_field_y) > 1:
            actual_resolution_x = (sound_field_x.max() - sound_field_x.min()) / (len(sound_field_x) - 1)
            actual_resolution_y = (sound_field_y.max() - sound_field_y.min()) / (len(sound_field_y) - 1)
            return (actual_resolution_x + actual_resolution_y) / 2.0
        else:
            return resolution
    
    def generate_per_surface_process_and_create_grids_create_triangulation(
        self,
        geom: SurfaceGeometry,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        surface_mask_strict: np.ndarray,
        additional_vertices: np.ndarray,
        actual_resolution: float,
        ) -> Dict[str, Any]:
        """Erstellt Triangulation für SurfaceGrid."""
        # 🎯 TRIANGULATION: Erstelle triangulierte Vertices aus Grid-Punkten
        # 🎯 NEU: Verwende Delaunay-Triangulation für konsistente, überschneidungsfreie Triangulation
        triangulated_vertices = None
        triangulated_faces = None
        triangulated_success = False
        vertex_source_indices: Optional[np.ndarray] = None
        
        # Konvertiere additional_vertices zu numpy array
        if not isinstance(additional_vertices, np.ndarray):
            additional_vertices_array = np.array(additional_vertices, dtype=float) if additional_vertices is not None and hasattr(additional_vertices, '__len__') and len(additional_vertices) > 0 else np.array([], dtype=float).reshape(0, 3)
        else:
            additional_vertices_array = additional_vertices
        
        try:
            # 1. Bereite Vertices vor
            (all_vertices, base_vertex_count, additional_vertex_start_idx, mask_flat, mask_strict_flat) = \
                self.generate_per_surface_process_and_create_grids_create_triangulation_prepare_vertices(
                    X_grid, Y_grid, Z_grid, surface_mask_strict, additional_vertices_array
                )
            
            if not np.any(mask_flat):
                return {
                    'triangulated_vertices': None,
                    'triangulated_faces': None,
                    'triangulated_success': False,
                    'vertex_source_indices': None,
                    'additional_vertices_final': additional_vertices_array,
                }
            
            # 2. Erstelle Delaunay-Triangulation
            faces_list = self.generate_per_surface_process_and_create_grids_create_triangulation_delaunay(
                geom, all_vertices, mask_flat, mask_strict_flat, base_vertex_count, additional_vertex_start_idx, additional_vertices_array
            )
            
            if len(faces_list) == 0:
                raise RuntimeError(f"Keine Faces für Surface {geom.surface_id} generiert. Delaunay-Triangulation hat keine Dreiecke erstellt.")
            
            # 3. Verarbeite Vertices und Faces (Mapping, Deduplizierung)
            triangulation_result = self.generate_per_surface_process_and_create_grids_create_triangulation_process_vertices(
                geom, all_vertices, faces_list, mask_flat, base_vertex_count, additional_vertices_array, additional_vertex_start_idx, actual_resolution
            )
            
            return {
                'triangulated_vertices': triangulation_result['triangulated_vertices'],
                'triangulated_faces': triangulation_result['triangulated_faces'],
                'triangulated_success': triangulation_result['triangulated_success'],
                'vertex_source_indices': triangulation_result['vertex_source_indices'],
                'additional_vertices_final': additional_vertices_array,
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'triangulated_vertices': None,
                'triangulated_faces': None,
                'triangulated_success': False,
                'vertex_source_indices': None,
                'additional_vertices_final': additional_vertices_array,
            }
    
    def generate_per_surface_process_and_create_grids_create_triangulation_prepare_vertices(
        self,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask_strict: np.ndarray,
        additional_vertices_array: np.ndarray,
        ) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
        """Bereitet Vertices für Triangulation vor."""
        ny, nx = X_grid.shape
        mask_flat = surface_mask_strict.ravel()
        mask_strict_flat = surface_mask_strict.ravel()
        
        # Erstelle Vertex-Koordinaten aus allen Grid-Punkten
        all_vertices = np.column_stack([
            X_grid.ravel(),
            Y_grid.ravel(),
            Z_grid.ravel()
        ])
        
        # Füge additional_vertices hinzu
        if additional_vertices_array.size > 0:
            all_vertices = np.vstack([all_vertices, additional_vertices_array])
        
        base_vertex_count = X_grid.size
        additional_vertex_start_idx = base_vertex_count
        
        return (all_vertices, base_vertex_count, additional_vertex_start_idx, mask_flat, mask_strict_flat)
    
    def generate_per_surface_process_and_create_grids_create_triangulation_delaunay(
        self,
        geom: SurfaceGeometry,
        all_vertices: np.ndarray,
        mask_flat: np.ndarray,
        mask_strict_flat: np.ndarray,
        base_vertex_count: int,
        additional_vertex_start_idx: int,
        additional_vertices_array: np.ndarray,
        ) -> List[int]:
        """Erstellt Delaunay-Triangulation und gibt faces_list zurück."""
        faces_list = []
        
        if not HAS_SCIPY:
            raise RuntimeError(f"scipy ist erforderlich für Delaunay-Triangulation. Surface: {geom.surface_id}")
        
        # Sammle alle gültigen Punkte (Grid-Punkte in Maske + additional_vertices)
        valid_vertices_list = []
        valid_vertex_indices = []
        
        # Füge aktive Grid-Punkte hinzu
        active_mask_indices = np.where(mask_flat)[0]
        for idx in active_mask_indices:
            if idx < len(all_vertices):
                valid_vertices_list.append(all_vertices[idx])
                valid_vertex_indices.append(idx)
        
        # Füge additional_vertices hinzu
        if len(additional_vertices_array) > 0:
            for i in range(len(additional_vertices_array)):
                idx = additional_vertex_start_idx + i
                if idx < len(all_vertices):
                    valid_vertices_list.append(all_vertices[idx])
                    valid_vertex_indices.append(idx)
        
        if len(valid_vertices_list) < 3:
            return faces_list
        
        valid_vertices_array = np.array(valid_vertices_list, dtype=float)
        
        # Bestimme Projektionsebene basierend auf Orientierung
        if geom.orientation == "vertical":
            if hasattr(geom, 'dominant_axis') and geom.dominant_axis:
                if geom.dominant_axis == "xz":
                    proj_2d = valid_vertices_array[:, [0, 2]]  # x, z
                elif geom.dominant_axis == "yz":
                    proj_2d = valid_vertices_array[:, [1, 2]]  # y, z
                else:
                    proj_2d = valid_vertices_array[:, [0, 1]]
            else:
                proj_2d = valid_vertices_array[:, [0, 1]]
        else:
            proj_2d = valid_vertices_array[:, [0, 1]]  # x, y
        
        # Delaunay-Triangulation in 2D
        tri = Delaunay(proj_2d)
        triangles_delaunay = tri.simplices
        
        # Filtere Dreiecke, die außerhalb der Surface-Maske liegen
        valid_triangles = []
        surface_points = geom.points or []
        
        if len(surface_points) >= 3:
            try:
                from matplotlib.path import Path
                
                # Erstelle Polygon-Pfad für Punkt-im-Polygon-Prüfung
                if geom.orientation == "vertical":
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
                    polygon_2d = np.array([[p.get("x", 0.0), p.get("y", 0.0)] for p in surface_points], dtype=float)
                
                polygon_path = Path(polygon_2d)
                
                # Prüfe jeden Dreieck-Schwerpunkt
                for tri_idx in triangles_delaunay:
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
                    all_vertices_in_extended = v1_in_extended and v2_in_extended and v3_in_extended
                    
                    # Schwerpunkt des Dreiecks in 2D
                    centroid_2d = proj_2d[tri_idx].mean(axis=0)
                    centroid_in_polygon = polygon_path.contains_point(centroid_2d)
                    
                    # Akzeptiere Dreiecke wenn: Schwerpunkt im Polygon ODER mindestens ein Vertex in strikter Maske ODER alle Vertices in erweiterter Maske
                    if centroid_in_polygon or has_strict_vertex or all_vertices_in_extended:
                        valid_triangles.append((v1_old, v2_old, v3_old))
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
        for v1, v2, v3 in valid_triangles:
            faces_list.extend([3, int(v1), int(v2), int(v3)])
        
        return faces_list
    
    def generate_per_surface_process_and_create_grids_create_triangulation_process_vertices(
        self,
        geom: SurfaceGeometry,
        all_vertices: np.ndarray,
        faces_list: List[int],
        mask_flat: np.ndarray,
        base_vertex_count: int,
        additional_vertices_array: np.ndarray,
        additional_vertex_start_idx: int,
        actual_resolution: float,
        ) -> Dict[str, Any]:
        """Verarbeitet Vertices und Faces (Mapping, Deduplizierung)."""
        if len(faces_list) == 0:
            return {
                'triangulated_vertices': np.array([], dtype=float).reshape(0, 3),
                'triangulated_faces': np.array([], dtype=np.int64),
                'triangulated_success': False,
                'vertex_source_indices': None,
            }
        
        # Extrahiere Vertex-Indices aus Faces
        vertex_indices_in_faces = []
        for i in range(0, len(faces_list), 4):
            if i + 3 < len(faces_list):
                n_verts = faces_list[i]
                if n_verts == 3:
                    vertex_indices_in_faces.extend(faces_list[i+1:i+4])
        
        vertex_indices_in_faces_set = set(vertex_indices_in_faces)
        
        # Füge alle aktiven Masken-Punkte hinzu
        active_mask_indices = np.where(mask_flat)[0]
        for idx in active_mask_indices:
            if idx < len(all_vertices):
                vertex_indices_in_faces_set.add(int(idx))
        
        # Füge additional_vertices hinzu
        if len(additional_vertices_array) > 0:
            for i in range(len(additional_vertices_array)):
                additional_idx = base_vertex_count + i
                if additional_idx < len(all_vertices):
                    vertex_indices_in_faces_set.add(int(additional_idx))
        
        all_vertex_indices = np.array(sorted(vertex_indices_in_faces_set), dtype=np.int64)
        
        # Erstelle Mapping: alter Index → neuer Index
        old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(all_vertex_indices)}
        
        # Erstelle vorläufiges vertex_source_indices-Mapping
        vertex_source_indices_pre_dedup = []
        
        if len(all_vertex_indices) > 0:
            # Trenne normale Grid-Vertices von zusätzlichen Vertices
            grid_vertex_indices = all_vertex_indices[all_vertex_indices < base_vertex_count]
            additional_vertex_indices = all_vertex_indices[all_vertex_indices >= base_vertex_count]
            
            # Erstelle Vertices aus Grid-Punkten
            if len(grid_vertex_indices) > 0:
                grid_vertices = all_vertices[grid_vertex_indices]
                for old_idx in grid_vertex_indices:
                    vertex_source_indices_pre_dedup.append(('grid', old_idx))
            else:
                grid_vertices = np.array([], dtype=float).reshape(0, 3)
            
            # Erstelle Vertices aus zusätzlichen Vertices
            if len(additional_vertex_indices) > 0 and len(additional_vertices_array) > 0:
                additional_indices_local = additional_vertex_indices - base_vertex_count
                additional_indices_local = additional_indices_local[additional_indices_local < len(additional_vertices_array)]
                if len(additional_indices_local) > 0:
                    additional_vertices_selected = additional_vertices_array[additional_indices_local]
                    for old_idx in additional_vertex_indices:
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
        
        # Mappe Face-Indizes auf neue Vertex-Indizes
        faces_list_mapped = []
        for i in range(0, len(faces_list), 4):
            if i + 3 < len(faces_list):
                n_verts = faces_list[i]
                if n_verts == 3:
                    v1_old = faces_list[i+1]
                    v2_old = faces_list[i+2]
                    v3_old = faces_list[i+3]
                    if v1_old in old_to_new_index and v2_old in old_to_new_index and v3_old in old_to_new_index:
                        faces_list_mapped.extend([
                            3,
                            old_to_new_index[v1_old],
                            old_to_new_index[v2_old],
                            old_to_new_index[v3_old]
                        ])
        
        if len(faces_list_mapped) > 0:
            triangulated_faces = np.array(faces_list_mapped, dtype=np.int64)
            triangulated_success = True
        else:
            triangulated_faces = np.array([], dtype=np.int64)
            triangulated_success = False
        
        # Deduplizierung
        vertices_before_dedup_array = triangulated_vertices.copy() if triangulated_vertices.size > 0 else np.array([], dtype=float).reshape(0, 3)
        
        try:
            triangulated_vertices, triangulated_faces = self._deduplicate_vertices_and_faces(
                triangulated_vertices,
                triangulated_faces,
                actual_resolution,
                geom.surface_id
            )
            
            # Erstelle vertex_source_indices nach Deduplizierung
            vertex_source_indices = None
            if len(triangulated_vertices) > 0 and len(vertex_source_indices_pre_dedup) > 0 and len(vertices_before_dedup_array) > 0:
                try:
                    from scipy.spatial.distance import cdist
                    distances = cdist(triangulated_vertices, vertices_before_dedup_array)
                    nearest_pre_dedup_indices = np.argmin(distances, axis=1)
                    
                    vertex_source_indices_mapping = []
                    n_grid_total = base_vertex_count
                    
                    for new_idx in range(len(triangulated_vertices)):
                        pre_dedup_idx = nearest_pre_dedup_indices[new_idx]
                        if pre_dedup_idx < len(vertex_source_indices_pre_dedup):
                            mapping_type, source_idx = vertex_source_indices_pre_dedup[pre_dedup_idx]
                            if mapping_type == 'grid':
                                vertex_source_indices_mapping.append(int(source_idx))
                            else:  # 'additional'
                                vertex_source_indices_mapping.append(int(n_grid_total + source_idx))
                        else:
                            vertex_source_indices_mapping.append(0)
                    
                    vertex_source_indices = np.array(vertex_source_indices_mapping, dtype=np.int64)
                except Exception:
                    vertex_source_indices = None
        except Exception:
            pass
        
        return {
            'triangulated_vertices': triangulated_vertices,
            'triangulated_faces': triangulated_faces,
            'triangulated_success': triangulated_success,
            'vertex_source_indices': vertex_source_indices,
        }
    
    def generate_per_surface_process_and_create_grids_create_surface_grid(
        self,
        geom: SurfaceGeometry,
        cache_key: tuple,
        surface_grids: Dict[str, SurfaceGrid],
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        surface_mask_strict: np.ndarray,
        actual_resolution: float,
        triangulation_data: Dict[str, Any],
        ) -> None:
        """Erstellt SurfaceGrid und füllt Cache."""
        triangulated_vertices = triangulation_data.get('triangulated_vertices')
        triangulated_faces = triangulation_data.get('triangulated_faces')
        triangulated_success = triangulation_data.get('triangulated_success', False)
        vertex_source_indices = triangulation_data.get('vertex_source_indices')
        additional_vertices_final = triangulation_data.get('additional_vertices_final')
        
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
            vertex_source_indices=vertex_source_indices,
            additional_vertices=additional_vertices_final,
        )
        
        # Im Cache speichern
        self._surface_grid_cache[cache_key] = surface_grid
        surface_grids[geom.surface_id] = surface_grid
        
        # Debug-Plot (optional)
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
        except Exception:
            pass
    
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
        resolution: float,
        ) -> None:
        """Erstellt optionalen 2D Debug-Plot für Surface."""
        # Debug-Plot-Funktionalität (optional implementiert)
        pass
    
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

