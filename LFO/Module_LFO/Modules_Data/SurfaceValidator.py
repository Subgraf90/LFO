"""
Einfache Validierung: Prüft ob Punkte planar sind und korrigiert abweichende Punkte.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import math

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    Delaunay = None

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
)

logger = logging.getLogger(__name__)

# Toleranz für cm-genaue Rundung (0.01m = 1cm)
CM_PRECISION = 0.01

# Toleranz für redundante Punkte
REDUNDANT_POINT_TOLERANCE = 0.001  # 1mm

# Toleranz für Planaritätsprüfung (5cm - großzügig für Rundungsfehler)
PLANAR_TOLERANCE = 0.05  # 5cm


class SurfaceValidationResult:
    """Ergebnis einer Surface-Validierung"""
    
    def __init__(
        self,
        is_valid: bool,
        optimized_points: List[Dict[str, float]],
        removed_points_count: int,
        rounded_points_count: int,
        error_message: Optional[str] = None,
        rigid_axis: Optional[str] = None,
        orientation: Optional[str] = None,
        invalid_fields: Optional[List[Tuple[int, str]]] = None,
    ):
        self.is_valid = is_valid
        self.optimized_points = optimized_points
        self.removed_points_count = removed_points_count
        self.rounded_points_count = rounded_points_count
        self.error_message = error_message
        self.rigid_axis = rigid_axis  # Für Kompatibilität
        self.orientation = orientation  # Für Kompatibilität
        self.invalid_fields = invalid_fields or []


def _check_planarity_geometric(points: np.ndarray) -> Tuple[bool, float]:
    """
    Prüft geometrisch ob Punkte planar sind.
    
    Verwendet Least-Squares-Fitting für robustere Prüfung, da gerundete Punkte
    leicht von der idealen Ebene abweichen können.
    
    Returns:
        (is_planar, max_distance)
    """
    if len(points) < 3:
        return False, float('inf')
    
    # Versuche zuerst geometrische Prüfung (schneller)
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm < 1e-10:
        # Punkte sind kollinear
        return False, float('inf')
    
    normal = normal / normal_norm
    
    # Berechne Abweichungen aller Punkte von der Ebene
    max_distance_geometric = 0.0
    for point in points:
        point_vec = point - points[0]
        distance = abs(np.dot(point_vec, normal))
        max_distance_geometric = max(max_distance_geometric, distance)
    
    # Wenn geometrisch planar, fertig
    if max_distance_geometric < PLANAR_TOLERANCE:
        return True, max_distance_geometric
    
    # Wenn nicht geometrisch planar, versuche Least-Squares-Fitting
    # (robuster gegen Rundungsfehler)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    # Versuche alle drei Achsen
    best_error = float('inf')
    
    # Z = f(X, Y)
    try:
        A = np.column_stack((xs, ys, np.ones_like(xs)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, zs, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * xs + coeffs[1] * ys + coeffs[2]
            errors = np.abs(predicted - zs)
            max_error = float(np.max(errors))
            best_error = min(best_error, max_error)
    except:
        pass
    
    # Y = f(X, Z)
    try:
        A = np.column_stack((xs, zs, np.ones_like(xs)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, ys, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * xs + coeffs[1] * zs + coeffs[2]
            errors = np.abs(predicted - ys)
            max_error = float(np.max(errors))
            best_error = min(best_error, max_error)
    except:
        pass
    
    # X = f(Y, Z)
    try:
        A = np.column_stack((ys, zs, np.ones_like(ys)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, xs, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * ys + coeffs[1] * zs + coeffs[2]
            errors = np.abs(predicted - xs)
            max_error = float(np.max(errors))
            best_error = min(best_error, max_error)
    except:
        pass
    
    # Verwende den besseren Wert (geometrisch oder Least-Squares)
    max_distance = min(max_distance_geometric, best_error)
    is_planar = max_distance < PLANAR_TOLERANCE
    
    return is_planar, max_distance


def _fit_plane_least_squares(points: np.ndarray) -> Optional[Dict]:
    """
    Finde beste Ebene durch Least-Squares-Fitting.
    Versucht alle drei Achsen als abhängige Variable.
    
    Returns:
        Dict mit Modell-Informationen oder None
    """
    if len(points) < 3:
        return None
    
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    best_model = None
    best_error = float('inf')
    
    # Versuche Z = ax + by + c
    try:
        A = np.column_stack((xs, ys, np.ones_like(xs)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, zs, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * xs + coeffs[1] * ys + coeffs[2]
            errors = np.abs(predicted - zs)
            max_error = float(np.max(errors))
            if max_error < best_error:
                best_error = max_error
                best_model = {
                    "type": "z",
                    "slope_x": float(coeffs[0]),
                    "slope_y": float(coeffs[1]),
                    "intercept": float(coeffs[2]),
                    "max_error": max_error,
                }
    except:
        pass
    
    # Versuche Y = ax + bz + c
    try:
        A = np.column_stack((xs, zs, np.ones_like(xs)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, ys, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * xs + coeffs[1] * zs + coeffs[2]
            errors = np.abs(predicted - ys)
            max_error = float(np.max(errors))
            if max_error < best_error:
                best_error = max_error
                best_model = {
                    "type": "y",
                    "slope_x": float(coeffs[0]),
                    "slope_z": float(coeffs[1]),
                    "intercept": float(coeffs[2]),
                    "max_error": max_error,
                }
    except:
        pass
    
    # Versuche X = ay + bz + c
    try:
        A = np.column_stack((ys, zs, np.ones_like(ys)))
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, xs, rcond=None)
        if rank >= 3:
            predicted = coeffs[0] * ys + coeffs[1] * zs + coeffs[2]
            errors = np.abs(predicted - xs)
            max_error = float(np.max(errors))
            if max_error < best_error:
                best_error = max_error
                best_model = {
                    "type": "x",
                    "slope_y": float(coeffs[0]),
                    "slope_z": float(coeffs[1]),
                    "intercept": float(coeffs[2]),
                    "max_error": max_error,
                }
    except:
        pass
    
    return best_model


def _identify_outliers(
    points: List[Dict[str, float]],
    model: Dict,
    tolerance: float = PLANAR_TOLERANCE,
) -> List[int]:
    """
    Identifiziert Ausreißer basierend auf Abweichungen von der Ebene.
    
    Strategie:
    - Berechne Fehler für alle Punkte basierend auf dem Modell aus allen Punkten
    - Identifiziere den Punkt mit dem größten Fehler (der am meisten aus der Reihe tanzt)
    - Bei ≤4 Punkten: Nur der Punkt mit dem größten Fehler ist ein Ausreißer
    - Bei >4 Punkten: Die Punkte mit den größten Fehlern sind Ausreißer (max n-3)
    
    Returns:
        Liste von Indizes der Ausreißer
    """
    # Berechne Fehler für alle Punkte basierend auf dem Modell
    errors = []
    for i, point in enumerate(points):
        x = point.get("x", 0.0)
        y = point.get("y", 0.0)
        z = point.get("z", 0.0)
        
        if model["type"] == "z":
            predicted = model["slope_x"] * x + model["slope_y"] * y + model["intercept"]
            error = abs(predicted - z)
        elif model["type"] == "y":
            predicted = model["slope_x"] * x + model["slope_z"] * z + model["intercept"]
            error = abs(predicted - y)
        elif model["type"] == "x":
            predicted = model["slope_y"] * y + model["slope_z"] * z + model["intercept"]
            error = abs(predicted - x)
        else:
            error = 0.0
        
        errors.append((i, error))
    
    # Sortiere nach Fehler (größte zuerst)
    errors.sort(key=lambda x: x[1], reverse=True)
    
    if len(points) <= 4:
        # Bei ≤4 Punkten: Nur der Punkt mit dem größten Fehler
        if errors and errors[0][1] > tolerance:
            outlier_index = errors[0][0]
            logger.debug(
                f"Punkt {outlier_index} als Ausreißer identifiziert "
                f"(Fehler: {errors[0][1]*100:.2f}cm, größter Fehler von {len(points)} Punkten)"
            )
            return [outlier_index]
        else:
            return []
    else:
        # Bei >4 Punkten: Die Punkte mit den größten Fehlern (max n-3)
        max_outliers = len(points) - 3
        outliers = []
        for i, error in errors[:max_outliers]:
            if error > tolerance:
                outliers.append(i)
        
        logger.debug(
            f"Ausreißererkennung: {len(points)} Punkte, "
            f"max {max_outliers} Ausreißer möglich, "
            f"{len(outliers)} Ausreißer identifiziert (größter Fehler: {errors[0][1]*100:.2f}cm)"
        )
        
        return outliers


def _correct_points_to_plane(
    points: List[Dict[str, float]],
    model: Dict,
    tolerance: float = PLANAR_TOLERANCE,
) -> Tuple[List[Dict[str, float]], List[Tuple[int, str]]]:
    """
    Korrigiert abweichende Punkte auf die Ebene.
    
    Bei >4 Punkten: Nur die identifizierten Ausreißer werden korrigiert.
    Bei ≤4 Punkten: Alle Punkte mit Abweichung > Toleranz werden korrigiert.
    
    Returns:
        (corrected_points, invalid_fields)
    """
    corrected = []
    invalid_fields = []
    corrections = []
    
    # Identifiziere Ausreißer
    outlier_indices = _identify_outliers(points, model, tolerance)
    outlier_set = set(outlier_indices)
    
    if model["type"] == "z":
        # Z = ax + by + c
        for i, point in enumerate(points):
            x = point.get("x", 0.0)
            y = point.get("y", 0.0)
            z = point.get("z", 0.0)
            
            predicted_z = model["slope_x"] * x + model["slope_y"] * y + model["intercept"]
            error = abs(predicted_z - z)
            
            # Korrigiere nur wenn Ausreißer ODER (bei <4 Punkten) wenn Abweichung > Toleranz
            if i in outlier_set or (len(points) < 4 and error > tolerance):
                # Korrigiere Z-Wert
                corrected_z = round(predicted_z / CM_PRECISION) * CM_PRECISION
                corrected.append({"x": x, "y": y, "z": corrected_z})
                invalid_fields.append((i, "z"))
                corrections.append((i, error, z, corrected_z))
            else:
                corrected.append({"x": x, "y": y, "z": z})
    
    elif model["type"] == "y":
        # Y = ax + bz + c
        for i, point in enumerate(points):
            x = point.get("x", 0.0)
            y = point.get("y", 0.0)
            z = point.get("z", 0.0)
            
            predicted_y = model["slope_x"] * x + model["slope_z"] * z + model["intercept"]
            error = abs(predicted_y - y)
            
            # Korrigiere nur wenn Ausreißer ODER (bei <4 Punkten) wenn Abweichung > Toleranz
            if i in outlier_set or (len(points) < 4 and error > tolerance):
                # Korrigiere Y-Wert
                corrected_y = round(predicted_y / CM_PRECISION) * CM_PRECISION
                corrected.append({"x": x, "y": corrected_y, "z": z})
                invalid_fields.append((i, "y"))
                corrections.append((i, error, y, corrected_y))
            else:
                corrected.append({"x": x, "y": y, "z": z})
    
    elif model["type"] == "x":
        # X = ay + bz + c
        for i, point in enumerate(points):
            x = point.get("x", 0.0)
            y = point.get("y", 0.0)
            z = point.get("z", 0.0)
            
            predicted_x = model["slope_y"] * y + model["slope_z"] * z + model["intercept"]
            error = abs(predicted_x - x)
            
            # Korrigiere nur wenn Ausreißer ODER (bei <4 Punkten) wenn Abweichung > Toleranz
            if i in outlier_set or (len(points) < 4 and error > tolerance):
                # Korrigiere X-Wert
                corrected_x = round(predicted_x / CM_PRECISION) * CM_PRECISION
                corrected.append({"x": corrected_x, "y": y, "z": z})
                invalid_fields.append((i, "x"))
                corrections.append((i, error, x, corrected_x))
            else:
                corrected.append({"x": x, "y": y, "z": z})
    
    if corrections:
        if len(points) > 4:
            logger.info(
                f"Identifiziert {len(outlier_indices)} Ausreißer von {len(points)} Punkten, "
                f"korrigiert {len(corrections)} Punkt(e):"
            )
        else:
            logger.info(f"Korrigiert {len(corrections)} Punkt(e):")
        for idx, error, old_val, new_val in corrections:
            coord_name = {"x": "X", "y": "Y", "z": "Z"}[model["type"]]
            logger.info(
                f"  Punkt {idx}: {coord_name}={old_val:.4f}m → {new_val:.4f}m "
                f"(Abweichung: {error*100:.2f}cm)"
            )
    
    return corrected, invalid_fields


def _round_points_to_cm(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Rundet alle Koordinaten auf cm genau"""
    rounded = []
    for point in points:
        rounded.append({
            "x": round(point.get("x", 0.0) / CM_PRECISION) * CM_PRECISION,
            "y": round(point.get("y", 0.0) / CM_PRECISION) * CM_PRECISION,
            "z": round(point.get("z", 0.0) / CM_PRECISION) * CM_PRECISION,
        })
    return rounded


def _remove_redundant_points(
    points: List[Dict[str, float]],
) -> Tuple[List[Dict[str, float]], int]:
    """Entfernt redundante Punkte"""
    if len(points) < 2:
        return points, 0
    
    coords = np.array([
        [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
        for p in points
    ], dtype=float)
    
    # Prüfe ob Polygon geschlossen ist
    is_closed = False
    if len(points) > 0:
        first = coords[0]
        last = coords[-1]
        dist = np.sqrt(np.sum((first - last) ** 2))
        is_closed = dist <= REDUNDANT_POINT_TOLERANCE
    
    end_idx = len(coords) - 1 if is_closed else len(coords)
    unique_indices = [0]
    removed_count = 0
    
    for i in range(1, end_idx):
        prev_coords = coords[unique_indices]
        distances = np.sqrt(np.sum((prev_coords - coords[i]) ** 2, axis=1))
        min_distance = np.min(distances)
        
        if min_distance > REDUNDANT_POINT_TOLERANCE:
            unique_indices.append(i)
        else:
            removed_count += 1
    
    if is_closed:
        removed_count += 1
    
    unique_points = [points[i] for i in unique_indices]
    return unique_points, removed_count


def _project_to_plane(points: np.ndarray, model: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projiziere 3D-Punkte auf 2D-Ebene (u,v) basierend auf dem Planemodell.
    Gibt (proj_points, origin, (u,v,normal)).
    """
    if model["type"] == "z":
        normal = np.array([-model["slope_x"], -model["slope_y"], 1.0])
    elif model["type"] == "y":
        normal = np.array([-model["slope_x"], 1.0, -model["slope_z"]])
    else:  # x
        normal = np.array([1.0, -model["slope_y"], -model["slope_z"]])
    normal = normal / np.linalg.norm(normal)

    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    origin = points.mean(axis=0)
    proj = []
    for p in points:
        vec = p - origin
        proj.append([np.dot(vec, u), np.dot(vec, v)])
    return np.array(proj), origin, np.array([u, v, normal])


def _is_simple_polygon(poly: np.ndarray) -> bool:
    """
    Prüft, ob ein 2D-Polygon nicht selbstschneidend ist (simple polygon).
    poly: Nx2
    """
    n = len(poly)
    if n < 3:
        return False

    def segments_intersect(a1, a2, b1, b2) -> bool:
        def ccw(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

        return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))

    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            b1, b2 = poly[j], poly[(j + 1) % n]
            # Skip adjacent segments sharing a vertex
            if i == j or (i + 1) % n == j or i == (j + 1) % n:
                continue
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def _ear_clip_triangulate(poly: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Ear-Clipping Triangulation für einfache 2D-Polygone.
    poly: Nx2
    Returns: Liste von Triplets (Indices)
    """
    n = len(poly)
    if n < 3:
        return []
    indices = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    def area2(a, b, c):
        return (poly[b][0] - poly[a][0]) * (poly[c][1] - poly[a][1]) - (poly[b][1] - poly[a][1]) * (poly[c][0] - poly[a][0])

    def is_ear(i0, i1, i2):
        if area2(i0, i1, i2) <= 0:
            return False
        tri = np.array([poly[i0], poly[i1], poly[i2]])
        for k in indices:
            if k in (i0, i1, i2):
                continue
            # Punkt in Dreieck?
            p = poly[k]
            a, b, c = tri
            # baryzentrisch
            v0, v1, v2 = c - a, b - a, p - a
            denom = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(denom) < 1e-12:
                continue
            u = (v2[0] * v1[1] - v2[1] * v1[0]) / denom
            v = (v0[0] * v2[1] - v0[1] * v2[0]) / denom
            if u > 0 and v > 0 and (u + v) < 1:
                return False
        return True

    cnt = 0
    while len(indices) > 3 and cnt < 1000:
        ear_found = False
        m = len(indices)
        for i in range(m):
            i0, i1, i2 = indices[i - 1], indices[i], indices[(i + 1) % m]
            if is_ear(i0, i1, i2):
                tris.append((i0, i1, i2))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            break
        cnt += 1
    if len(indices) == 3:
        tris.append((indices[0], indices[1], indices[2]))
    return tris


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Point-in-Polygon-Test mit Ray-Casting-Algorithmus.
    point: (x, y)
    polygon: Nx2 Array von Polygonpunkten (geschlossen, d.h. letzter Punkt = erster)
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def triangulate_points(points: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    """
    Trianguliert ein Polygon mit scipy.spatial.Delaunay (robust, bewährt).
    Falls scipy nicht verfügbar ist oder Delaunay fehlschlägt, Fallback auf Fan-Triangulation.
    
    Rückgabe: Liste von Dreieck-Punktlisten (je 3 Punkte). Bei Fehler: leere Liste.
    """
    if len(points) < 3:
        return []
    
    pts = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points])
    model = _fit_plane_least_squares(pts)
    if model is None:
        return []

    proj, origin, basis = _project_to_plane(pts, model)
    expected_tris = max(len(points) - 2, 0)
    
    logger.debug(
        "Triangulation start: %d Punkte, model_type=%s, max_error=%.4f, expected_tris=%d",
        len(points),
        model.get("type"),
        model.get("max_error", -1.0),
        expected_tris,
    )

    # Methode 1: scipy.spatial.Delaunay (robust, bewährt)
    if HAS_SCIPY:
        try:
            # Delaunay-Triangulation auf allen Punkten
            tri = Delaunay(proj)
            triangles_delaunay = tri.simplices  # Nx3 Array von Indices
            
            # Für konkave Polygone: Filtere Dreiecke, die außerhalb des Polygons liegen
            # Dazu prüfen wir, ob der Schwerpunkt jedes Dreiecks im Polygon liegt
            polygon_closed = np.vstack([proj, proj[0:1]])  # Geschlossenes Polygon
            
            valid_triangles = []
            for tri_idx in triangles_delaunay:
                # Schwerpunkt des Dreiecks
                centroid = proj[tri_idx].mean(axis=0)
                # Prüfe ob Schwerpunkt im Polygon liegt
                if _point_in_polygon(centroid, polygon_closed):
                    valid_triangles.append(tuple(tri_idx))
            
            # Falls wir zu wenige Dreiecke haben (z.B. bei sehr konkaven Polygonen),
            # verwenden wir alle Delaunay-Dreiecke, die mindestens einen Punkt auf dem Polygonrand haben
            if len(valid_triangles) < expected_tris * 0.8:  # Weniger als 80% erwartet
                logger.debug(
                    "Triangulation: Nur %d Dreiecke innerhalb Polygon, verwende alle Delaunay-Dreiecke",
                    len(valid_triangles),
                )
                valid_triangles = [tuple(tri_idx) for tri_idx in triangles_delaunay]
            
            if valid_triangles:
                logger.debug(
                    "Triangulation: Delaunay erfolgreich, %d Dreiecke (erwartet=%d)",
                    len(valid_triangles),
                    expected_tris,
                )
                triangles: List[List[Dict[str, float]]] = []
                for a, b, c in valid_triangles:
                    triangles.append([points[a], points[b], points[c]])
                
                if len(triangles) < expected_tris:
                    logger.warning(
                        "Triangulation: Weniger Dreiecke als erwartet (got=%d, expected=%d)",
                        len(triangles),
                        expected_tris,
                    )
                else:
                    logger.debug("Triangulation: Delaunay erfolgreich, %d Dreiecke", len(triangles))
                
                return triangles
        except Exception as e:
            logger.debug("Triangulation: Delaunay fehlgeschlagen: %s", str(e))

    # Fallback: Fan-Triangulation (immer funktioniert, garantiert n-2 Dreiecke)
    logger.debug("Triangulation: Verwende Fan-Fallback")
    centroid = proj.mean(axis=0)
    rel = proj - centroid
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    
    # Entferne nahezu doppelte Punkte (gleicher Winkel & sehr nah)
    filtered_order = []
    last_angle = None
    last_pt = None
    for idx in order:
        ang = angles[idx]
        pt = proj[idx]
        if last_angle is not None:
            if abs(ang - last_angle) < 1e-6 and np.linalg.norm(pt - last_pt) < 1e-6:
                continue
        filtered_order.append(idx)
        last_angle = ang
        last_pt = pt
    
    if len(filtered_order) < 3:
        logger.debug("Triangulation failed: zu wenige Punkte nach Filterung")
        return []
    
    tris_idx = []
    anchor = filtered_order[0]
    for i in range(1, len(filtered_order) - 1):
        tris_idx.append((anchor, filtered_order[i], filtered_order[i + 1]))
    
    logger.debug(
        "Triangulation: Fan-Fallback, %d Dreiecke (n=%d, erwartet=%d)",
        len(tris_idx),
        len(filtered_order),
        expected_tris,
    )
    
    triangles: List[List[Dict[str, float]]] = []
    for a, b, c in tris_idx:
        triangles.append([points[a], points[b], points[c]])
    
    return triangles


def order_points_planar(
    points: List[Dict[str, float]],
    *,
    tolerance: float = REDUNDANT_POINT_TOLERANCE,
) -> List[Dict[str, float]]:
    """
    Sortiert Punkte konsistent auf der Ebene (Winkel um den Schwerpunkt) und
    entfernt sehr nahe Punkte im projizierten 2D-Raum.
    """
    if len(points) < 3:
        return points

    pts = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points])
    model = _fit_plane_least_squares(pts)
    if model is None:
        return points

    # Ebenennormalen bestimmen
    if model["type"] == "z":
        normal = np.array([-model["slope_x"], -model["slope_y"], 1.0])
    elif model["type"] == "y":
        normal = np.array([-model["slope_x"], 1.0, -model["slope_z"]])
    else:  # "x"
        normal = np.array([1.0, -model["slope_y"], -model["slope_z"]])
    normal = normal / np.linalg.norm(normal)

    # Lokales Koordinatensystem auf der Ebene
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    centroid = pts.mean(axis=0)

    projected: List[Tuple[float, float, int]] = []
    for idx, p in enumerate(pts):
        vec = p - centroid
        x2d = float(np.dot(vec, u))
        y2d = float(np.dot(vec, v))
        projected.append((x2d, y2d, idx))

    # Redundante Punkte im 2D-Raum entfernen
    projected.sort(key=lambda t: (t[0], t[1]))
    filtered: List[Tuple[float, float, int]] = []
    for x2d, y2d, idx in projected:
        if not filtered:
            filtered.append((x2d, y2d, idx))
            continue
        last_x, last_y, _ = filtered[-1]
        if math.hypot(x2d - last_x, y2d - last_y) > tolerance:
            filtered.append((x2d, y2d, idx))

    if len(filtered) < 3:
        return points

    # Winkel um den Ursprung (Schwerpunkt) sortieren
    filtered_sorted = sorted(filtered, key=lambda t: math.atan2(t[1], t[0]))

    ordered_points = [points[idx] for _, _, idx in filtered_sorted]
    return ordered_points


def validate_and_optimize_surface(
    surface: SurfaceDefinition,
    *,
    round_to_cm: bool = True,
    remove_redundant: bool = True,
    optimize_invalid: bool = True,
) -> SurfaceValidationResult:
    """
    Einfache Validierung: Prüft ob Punkte planar sind und korrigiert abweichende Punkte.
    
    Strategie:
    1. Entferne redundante Punkte
    2. Prüfe ob Punkte planar sind (geometrisch)
    3. Wenn nicht planar, finde beste Ebene (Least-Squares)
    4. Korrigiere abweichende Punkte
    5. Runde auf cm genau
    """
    points = list(surface.points)
    
    if len(points) < 3:
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=0,
            rounded_points_count=0,
            error_message=f"Surface hat nur {len(points)} Punkte, mindestens 3 erforderlich",
        )
    
    # Schritt 1: Entferne redundante Punkte
    removed_redundant = 0
    if remove_redundant:
        logger.debug(f"Entferne redundante Punkte von Surface '{surface.name}' ({surface.surface_id})...")
        original_count = len(points)
        points, removed_redundant = _remove_redundant_points(points)
        if removed_redundant > 0:
            logger.debug(f"  {removed_redundant} redundante Punkt(e) entfernt ({original_count} → {len(points)})")
    
    # Schritt 2: Sammle gültige Punkte
    valid_points = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        z = p.get("z")
        if x is not None and y is not None and z is not None:
            try:
                valid_points.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                })
            except (ValueError, TypeError):
                pass
    
    if len(valid_points) < 3:
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=removed_redundant,
            rounded_points_count=0,
            error_message="Zu wenige gültige Punkte (mindestens 3 erforderlich)",
        )
    
    # Schritt 3: Prüfe Planarität
    points_array = np.array([
        [p["x"], p["y"], p["z"]]
        for p in valid_points
    ])
    
    # Schritt 4: Finde beste Ebene durch Least-Squares aus allen Punkten
    model = _fit_plane_least_squares(points_array)
    
    # Bestimme rigid_axis und orientation für UI-Kompatibilität
    rigid_axis, orientation = determine_surface_orientation(valid_points)
    
    if model is None:
        # Fallback: Geometrische Prüfung
        is_planar, max_distance = _check_planarity_geometric(points_array)
        if is_planar:
            logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist planar (max Abstand: {max_distance*100:.2f}cm)")
            is_valid = True
            invalid_fields = []
            corrected_points = points
        else:
            return SurfaceValidationResult(
                is_valid=False,
                optimized_points=points,
                removed_points_count=removed_redundant,
                rounded_points_count=0,
                error_message="Konnte keine Ebene durch die Punkte legen",
                rigid_axis=rigid_axis,
                orientation=orientation,
            )
    else:
        # Aktualisiere rigid_axis basierend auf Modell
        if rigid_axis is None:
            rigid_axis = model["type"]
            orientation = "sloped"
        
        # Prüfe ob Punkte innerhalb Toleranz sind
        # Führe immer Ausreißererkennung durch, auch wenn max_error < Toleranz,
        # da Least-Squares-Fitting den Fehler minimieren kann, obwohl ein Punkt abweicht
        outliers = _identify_outliers(valid_points, model, tolerance=PLANAR_TOLERANCE)
        
        if model["max_error"] < PLANAR_TOLERANCE and not outliers:
            logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist planar (max Fehler: {model['max_error']*100:.2f}cm)")
            is_valid = True
            invalid_fields = []
            corrected_points = points
        else:
            # Schritt 5: Korrigiere abweichende Punkte iterativ
            logger.info(f"Surface '{surface.name}' ({surface.surface_id}) ist nicht planar (max Fehler: {model['max_error']*100:.2f}cm), korrigiere...")
            
            if optimize_invalid:
                # Iterative Korrektur: Korrigiere so lange, bis alle Punkte innerhalb der Toleranz sind
                corrected_points = list(points)
                max_iterations = 10  # Verhindere Endlosschleifen
                iteration = 0
                
                while iteration < max_iterations:
                    # Berechne Modell aus aktuellen Punkten
                    current_array = np.array([
                        [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
                        for p in corrected_points
                    ])
                    current_model = _fit_plane_least_squares(current_array)
                    
                    if current_model is None:
                        break
                    
                    # Prüfe ob alle Punkte innerhalb Toleranz sind
                    if current_model["max_error"] < PLANAR_TOLERANCE:
                        is_valid = True
                        logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist nach {iteration} Iteration(en) gültig (max Fehler: {current_model['max_error']*100:.2f}cm)")
                        break
                    
                    # Identifiziere Ausreißer
                    outlier_indices = _identify_outliers(corrected_points, current_model, tolerance=PLANAR_TOLERANCE)
                    
                    if not outlier_indices:
                        # Keine Ausreißer mehr gefunden, aber immer noch nicht planar
                        # Korrigiere alle Punkte mit Fehler > Toleranz
                        _, invalid_fields = _correct_points_to_plane(corrected_points, current_model, tolerance=PLANAR_TOLERANCE)
                        is_valid = len(invalid_fields) == 0
                        break
                    
                    # Korrigiere nur die identifizierten Ausreißer
                    corrected_points, invalid_fields = _correct_points_to_plane(corrected_points, current_model, tolerance=PLANAR_TOLERANCE)
                    iteration += 1
                
                if iteration >= max_iterations:
                    # Max Iterationen erreicht
                    current_array = np.array([
                        [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
                        for p in corrected_points
                    ])
                    current_model = _fit_plane_least_squares(current_array)
                    if current_model and current_model["max_error"] < PLANAR_TOLERANCE:
                        is_valid = True
                        logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist nach {iteration} Iteration(en) gültig (max Fehler: {current_model['max_error']*100:.2f}cm)")
                    else:
                        is_valid = False
                        logger.warning(
                            f"✗ Surface '{surface.name}' ({surface.surface_id}) hat nach {iteration} Iteration(en) noch "
                            f"Fehler > Toleranz (max Fehler: {current_model['max_error']*100:.2f}cm)"
                        )
            else:
                # Prüfe ohne Korrektur
                _, invalid_fields = _correct_points_to_plane(points, model, tolerance=PLANAR_TOLERANCE)
                is_valid = len(invalid_fields) == 0
                corrected_points = points
    
    # Schritt 6: Rundung auf cm genau (NACH Korrektur)
    rounded_count = 0
    if round_to_cm:
        logger.debug(f"Runde Surface '{surface.name}' ({surface.surface_id}) auf cm genau...")
        corrected_points = _round_points_to_cm(corrected_points)
        rounded_count = len(corrected_points)
    
    error_msg = None
    if not is_valid:
        error_msg = f"{len(invalid_fields)} Punkt(e) weichen von der planaren Fläche ab (Toleranz: {PLANAR_TOLERANCE*100:.1f}cm)"
    
    return SurfaceValidationResult(
        is_valid=is_valid,
        optimized_points=corrected_points,
        removed_points_count=removed_redundant,
        rounded_points_count=rounded_count,
        error_message=error_msg,
        rigid_axis=rigid_axis,
        orientation=orientation,
        invalid_fields=invalid_fields,
    )


def validate_surface_geometry(
    surface: SurfaceDefinition,
    *,
    round_to_cm: bool = False,
    remove_redundant: bool = True,
    tolerance: float = PLANAR_TOLERANCE,
) -> SurfaceValidationResult:
    """
    Führt nur die Validierung aus (keine Korrektur).
    Prüft, ob eine Ebene durch die Punkte gelegt werden kann und welche Punkte Ausreißer sind.
    """
    points = list(surface.points)

    if len(points) < 3:
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=0,
            rounded_points_count=0,
            error_message=f"Surface hat nur {len(points)} Punkte, mindestens 3 erforderlich",
        )

    removed_redundant = 0
    if remove_redundant:
        points, removed_redundant = _remove_redundant_points(points)

    valid_points: List[Dict[str, float]] = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        z = p.get("z")
        if x is not None and y is not None and z is not None:
            try:
                valid_points.append({"x": float(x), "y": float(y), "z": float(z)})
            except (ValueError, TypeError):
                pass

    if len(valid_points) < 3:
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=removed_redundant,
            rounded_points_count=0,
            error_message="Zu wenige gültige Punkte (mindestens 3 erforderlich)",
        )

    points_array = np.array([[p["x"], p["y"], p["z"]] for p in valid_points])

    model = _fit_plane_least_squares(points_array)

    if model is None:
        # Fallback: Geometrische Prüfung
        is_planar, max_distance = _check_planarity_geometric(points_array)
        return SurfaceValidationResult(
            is_valid=is_planar,
            optimized_points=points,
            removed_points_count=removed_redundant,
            rounded_points_count=0,
            error_message=None if is_planar else "Konnte keine Ebene durch die Punkte legen",
            rigid_axis=None,
            orientation=None,
            invalid_fields=[] if is_planar else [(i, "z") for i in range(len(points))],
        )

    # Bestimme Ausreißer
    outlier_indices = _identify_outliers(valid_points, model, tolerance=tolerance)
    invalid_fields: List[Tuple[int, str]] = []
    for idx in outlier_indices:
        axis = model["type"]
        invalid_fields.append((idx, axis))

    is_valid = len(outlier_indices) == 0 and model["max_error"] < tolerance

    rounded_count = 0
    if round_to_cm:
        points = _round_points_to_cm(points)
        rounded_count = len(points)

    error_msg = None
    if not is_valid:
        error_msg = f"{len(outlier_indices)} Punkt(e) weichen von der planaren Fläche ab (Toleranz: {tolerance*100:.1f}cm)"

    return SurfaceValidationResult(
        is_valid=is_valid,
        optimized_points=points,
        removed_points_count=removed_redundant,
        rounded_points_count=rounded_count,
        error_message=error_msg,
        rigid_axis=model.get("type"),
        orientation="sloped",
        invalid_fields=invalid_fields,
    )


def validate_surface_dict(
    surface_dict: Dict,
    *,
    round_to_cm: bool = True,
    remove_redundant: bool = True,
    optimize_invalid: bool = True,
) -> Dict:
    """Validiert eine Surface-Definition als Dictionary"""
    if isinstance(surface_dict, SurfaceDefinition):
        surface = surface_dict
    else:
        surface_id = surface_dict.get("surface_id", "unknown")
        surface = SurfaceDefinition.from_dict(surface_id, surface_dict)
    
    result = validate_and_optimize_surface(
        surface,
        round_to_cm=round_to_cm,
        remove_redundant=remove_redundant,
        optimize_invalid=optimize_invalid,
    )
    
    if isinstance(surface_dict, SurfaceDefinition):
        surface_dict.points = result.optimized_points
        return surface_dict
    else:
        surface_dict["points"] = result.optimized_points
        return surface_dict


def validate_surfaces_dict(
    surfaces: Dict[str, Dict],
    *,
    round_to_cm: bool = True,
    remove_redundant: bool = True,
    optimize_invalid: bool = True,
) -> Dict[str, Dict]:
    """Validiert mehrere Surfaces"""
    validated = {}
    
    for surface_id, surface_data in surfaces.items():
        try:
            validated[surface_id] = validate_surface_dict(
                surface_data,
                round_to_cm=round_to_cm,
                remove_redundant=remove_redundant,
                optimize_invalid=optimize_invalid,
            )
        except Exception as e:
            logger.warning(
                f"Fehler beim Validieren von Surface '{surface_id}': {e}",
                exc_info=True,
            )
            validated[surface_id] = surface_data
    
    return validated


# Kompatibilitätsfunktionen für UI
def determine_surface_orientation(
    points: List[Dict[str, float]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Bestimmt die Orientierung der Fläche (für UI-Kompatibilität).
    Verwendet die einfache Planaritätsprüfung.
    
    Returns:
        (rigid_axis, orientation) - rigid_axis: "x", "y" oder "z", orientation: Beschreibung
    """
    if len(points) < 3:
        return None, None
    
    # Sammle gültige Punkte
    valid_points = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        z = p.get("z")
        if x is not None and y is not None and z is not None:
            try:
                valid_points.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                })
            except (ValueError, TypeError):
                pass
    
    if len(valid_points) < 3:
        return None, None
    
    points_array = np.array([
        [p["x"], p["y"], p["z"]]
        for p in valid_points
    ])
    
    # Finde beste Ebene
    model = _fit_plane_least_squares(points_array)
    if model is None:
        return None, None
    
    # Bestimme Orientierung basierend auf Modell
    if model["type"] == "z":
        # Prüfe ob horizontal (Z konstant)
        z_vals = points_array[:, 2]
        if np.ptp(z_vals) < 0.01:  # 1cm Toleranz
            return "z", "horizontal"
        else:
            return "z", "sloped"
    elif model["type"] == "y":
        # Prüfe ob vertikal Y
        y_vals = points_array[:, 1]
        if np.ptp(y_vals) < 0.01:  # 1cm Toleranz
            return "y", "vertical_y"
        else:
            return "y", "sloped"
    elif model["type"] == "x":
        # Prüfe ob vertikal X
        x_vals = points_array[:, 0]
        if np.ptp(x_vals) < 0.01:  # 1cm Toleranz
            return "x", "vertical_x"
        else:
            return "x", "sloped"
    
    return None, None


def check_rigid_axis_validity(
    points: List[Dict[str, float]],
    rigid_axis: str,
    orientation: Optional[str] = None,
) -> Tuple[bool, List[Tuple[int, str]]]:
    """
    Prüft ob alle Punkte planar sind (für UI-Kompatibilität).
    
    Returns:
        (is_valid, invalid_fields)
    """
    if len(points) < 3:
        return False, []
    
    # Sammle gültige Punkte
    valid_points = []
    for i, p in enumerate(points):
        x = p.get("x")
        y = p.get("y")
        z = p.get("z")
        if x is not None and y is not None and z is not None:
            try:
                valid_points.append({
                    "index": i,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                })
            except (ValueError, TypeError):
                pass
    
    if len(valid_points) < 3:
        invalid_fields = [(i, rigid_axis) for i in range(len(points)) if points[i].get(rigid_axis) is None]
        return False, invalid_fields
    
    points_array = np.array([
        [p["x"], p["y"], p["z"]]
        for p in valid_points
    ])
    
    # Prüfe Planarität
    is_planar, max_distance = _check_planarity_geometric(points_array)
    
    if is_planar:
        # Prüfe auch None-Werte
        invalid_fields = []
        for i, point in enumerate(points):
            if point.get(rigid_axis) is None:
                invalid_fields.append((i, rigid_axis))
        return len(invalid_fields) == 0, invalid_fields
    else:
        # Finde beste Ebene und prüfe Abweichungen
        model = _fit_plane_least_squares(points_array)
        if model is None:
            invalid_fields = [(i, rigid_axis) for i in range(len(points))]
            return False, invalid_fields
        
        # Prüfe Abweichungen
        invalid_fields = []
        for p in valid_points:
            if model["type"] == "z":
                predicted = model["slope_x"] * p["x"] + model["slope_y"] * p["y"] + model["intercept"]
                error = abs(predicted - p["z"])
            elif model["type"] == "y":
                predicted = model["slope_x"] * p["x"] + model["slope_z"] * p["z"] + model["intercept"]
                error = abs(predicted - p["y"])
            elif model["type"] == "x":
                predicted = model["slope_y"] * p["y"] + model["slope_z"] * p["z"] + model["intercept"]
                error = abs(predicted - p["x"])
            else:
                error = PLANAR_TOLERANCE + 1.0  # Immer ungültig
            
            if error > PLANAR_TOLERANCE:
                invalid_fields.append((p["index"], rigid_axis))
        
        # Prüfe auch None-Werte
        for i, point in enumerate(points):
            if point.get(rigid_axis) is None:
                invalid_fields.append((i, rigid_axis))
        
        return len(invalid_fields) == 0, invalid_fields

