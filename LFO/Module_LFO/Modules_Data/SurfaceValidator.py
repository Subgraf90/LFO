"""
Einfache Validierung: Prüft ob Punkte planar sind und korrigiert abweichende Punkte.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

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
    - Bei ≤4 Punkten: Alle Punkte mit Abweichung > Toleranz sind Ausreißer
    - Bei >4 Punkten: Nur die Punkte mit den größten Abweichungen sind Ausreißer
      (Anzahl möglicher Ausreißer: bei 5 Punkten max 2, bei 6+ Punkten entsprechend mehr)
    
    Returns:
        Liste von Indizes der Ausreißer
    """
    if len(points) <= 4:
        # Bei ≤4 Punkten: Alle Punkte mit Abweichung > Toleranz sind Ausreißer
        outliers = []
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
            
            if error > tolerance:
                outliers.append(i)
        return outliers
    else:
        # Bei >4 Punkten: Berechne Abweichungen für alle Punkte und identifiziere die größten
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
        
        # Sortiere nach Abweichung (größte zuerst)
        errors.sort(key=lambda x: x[1], reverse=True)
        
        # Bestimme maximale Anzahl möglicher Ausreißer:
        # Bei 5 Punkten: max 2 Ausreißer
        # Bei 6 Punkten: max 2 Ausreißer
        # Bei 7+ Punkten: max 3 Ausreißer (oder mehr, je nach Gesamtzahl)
        max_outliers = min(2, len(points) - 3) if len(points) <= 6 else min(3, len(points) - 3)
        
        # Identifiziere Ausreißer: Die Punkte mit den größten Abweichungen
        # (nur wenn Abweichung > Toleranz)
        outliers = []
        for i, error in errors[:max_outliers]:
            if error > tolerance:
                outliers.append(i)
        
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
            
            # Korrigiere nur wenn Ausreißer ODER (bei ≤4 Punkten) wenn Abweichung > Toleranz
            if i in outlier_set or (len(points) <= 4 and error > tolerance):
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
            
            if error > tolerance:
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
            
            if error > tolerance:
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
    
    # Schritt 4: Finde beste Ebene durch Least-Squares (immer, für robuste Prüfung)
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
        if model["max_error"] < PLANAR_TOLERANCE:
            logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist planar (max Fehler: {model['max_error']*100:.2f}cm)")
            is_valid = True
            invalid_fields = []
            corrected_points = points
        else:
            # Schritt 5: Korrigiere abweichende Punkte
            logger.info(f"Surface '{surface.name}' ({surface.surface_id}) ist nicht planar (max Fehler: {model['max_error']*100:.2f}cm), korrigiere...")
            
            if optimize_invalid:
                corrected_points, invalid_fields = _correct_points_to_plane(points, model, tolerance=PLANAR_TOLERANCE)
                
                # Prüfe erneut nach Korrektur (mit Least-Squares, nicht geometrisch)
                corrected_array = np.array([
                    [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
                    for p in corrected_points
                ])
                corrected_model = _fit_plane_least_squares(corrected_array)
                
                if corrected_model and corrected_model["max_error"] < PLANAR_TOLERANCE:
                    is_valid = True
                    logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist nach Korrektur gültig (max Fehler: {corrected_model['max_error']*100:.2f}cm)")
                else:
                    is_valid = False
                    logger.warning(
                        f"✗ Surface '{surface.name}' ({surface.surface_id}) hat nach Korrektur noch "
                        f"{len(invalid_fields)} ungültige Punkt(e)"
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

