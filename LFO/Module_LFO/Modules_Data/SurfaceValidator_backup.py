"""
Modul zur Validierung und Optimierung von Surface-Definitionen.

Dieses Modul prüft Surfaces auf Gültigkeit basierend auf starrer Achse:
- Bestimmt Orientierung der Fläche (horizontal, vertikal, schräg)
- Definiert starre Achse (X, Y oder Z)
- Prüft ob alle Werte der starren Achse konsistent sind
- Passt abweichende Werte an, sodass alle Punkte eine planare Fläche bilden
- Entfernt redundante Datenpunkte
- Rundet Koordinaten auf cm genau
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    derive_surface_plane,
    evaluate_surface_plane,
    _fit_planar_surface,
)

logger = logging.getLogger(__name__)

# Toleranz für cm-genaue Rundung (0.01m = 1cm)
CM_PRECISION = 0.01

# Toleranz für redundante Punkte (Punkte die näher als diese Distanz sind werden als redundant betrachtet)
REDUNDANT_POINT_TOLERANCE = 0.001  # 1mm

# Toleranz für kollineare Punkte
COLLINEAR_TOLERANCE = 1e-6

# Toleranz für starre Achse (Werte müssen innerhalb dieser Toleranz sein)
RIGID_AXIS_TOLERANCE = 0.01  # 1cm

# Toleranz für planare Flächen-Validierung
# Nach cm-Rundung können Punkte bis zu 0.5mm pro Achse abweichen.
# Für 4 Punkte: max 0.5mm * 2 = 1mm pro Punkt, aber durch Rundung können sich Fehler akkumulieren.
# Daher großzügige Toleranz von 5cm, um sicherzustellen, dass alle gültigen planaren Flächen akzeptiert werden.
PLANAR_VALIDATION_TOLERANCE = 0.05  # 5cm (großzügig für Rundungsfehler und numerische Ungenauigkeiten)


class SurfaceValidationResult:
    """Ergebnis einer Surface-Validierung"""
    
    def __init__(
        self,
        is_valid: bool,
        optimized_points: List[Dict[str, float]],
        removed_points_count: int,
        rounded_points_count: int,
        error_message: Optional[str] = None,
        rigid_axis: Optional[Literal["x", "y", "z"]] = None,
        orientation: Optional[str] = None,
        invalid_fields: Optional[List[Tuple[int, str]]] = None,
    ):
        self.is_valid = is_valid
        self.optimized_points = optimized_points
        self.removed_points_count = removed_points_count
        self.rounded_points_count = rounded_points_count
        self.error_message = error_message
        self.rigid_axis = rigid_axis  # "x", "y" oder "z"
        self.orientation = orientation  # "horizontal", "vertical_x", "vertical_y", "sloped"
        self.invalid_fields = invalid_fields or []  # Liste von (point_index, coord_name) für ungültige Felder


def determine_surface_orientation(
    points: List[Dict[str, float]],
) -> Tuple[Optional[Literal["x", "y", "z"]], Optional[str]]:
    """
    Bestimmt die Orientierung der Fläche und die starre Achse.
    
    Die starre Achse muss NICHT konstant sein - sie muss nur aus den anderen beiden Achsen
    berechenbar sein (planare Fläche).
    
    Strategie:
    1. Prüfe zuerst, ob eine Achse konstant ist (horizontal/vertikal) - das ist der einfachste Fall
    2. Wenn keine Achse konstant ist, prüfe für jede Achse, ob sie aus den anderen beiden
       berechenbar ist (planare Validierung)
    3. Wähle die Achse mit dem besten Fit (kleinster Fehler)
    
    Args:
        points: Liste von Punkten
        
    Returns:
        (rigid_axis, orientation) - rigid_axis: "x", "y" oder "z", orientation: Beschreibung
    """
    if len(points) < 3:
        return None, None
    
    # Sammle gültige Punkte mit allen drei Koordinaten
    valid_points = []
    for p in points:
        x_val = p.get("x")
        y_val = p.get("y")
        z_val = p.get("z")
        if x_val is not None and y_val is not None and z_val is not None:
            try:
                valid_points.append({
                    "x": float(x_val),
                    "y": float(y_val),
                    "z": float(z_val),
                })
            except (ValueError, TypeError):
                pass
    
    if len(valid_points) < 3:
        return None, None
    
    # Sammle Werte für Spannweiten-Berechnung
    x_vals = [p["x"] for p in valid_points]
    y_vals = [p["y"] for p in valid_points]
    z_vals = [p["z"] for p in valid_points]
    
    # Berechne Spannweite pro Achse
    x_span = float(np.ptp(x_vals)) if len(x_vals) > 0 else float('inf')
    y_span = float(np.ptp(y_vals)) if len(y_vals) > 0 else float('inf')
    z_span = float(np.ptp(z_vals)) if len(z_vals) > 0 else float('inf')
    
    # Prüfe zuerst einfache Fälle: Ist eine Achse konstant?
    # Horizontale Fläche: Z-Achse konstant
    if z_span <= RIGID_AXIS_TOLERANCE:
        logger.debug(f"Horizontale Fläche erkannt: Z-Spanne = {z_span*100:.2f}cm ≤ {RIGID_AXIS_TOLERANCE*100:.1f}cm")
        return "z", "horizontal"
    
    # Vertikale Fläche X: X-Achse konstant
    if x_span <= RIGID_AXIS_TOLERANCE:
        logger.debug(f"Vertikale Fläche (X) erkannt: X-Spanne = {x_span*100:.2f}cm ≤ {RIGID_AXIS_TOLERANCE*100:.1f}cm")
        return "x", "vertical_x"
    
    # Vertikale Fläche Y: Y-Achse konstant
    if y_span <= RIGID_AXIS_TOLERANCE:
        logger.debug(f"Vertikale Fläche (Y) erkannt: Y-Spanne = {y_span*100:.2f}cm ≤ {RIGID_AXIS_TOLERANCE*100:.1f}cm")
        return "y", "vertical_y"
    
    # Keine Achse ist konstant - prüfe welche Achse am besten aus den anderen berechenbar ist
    # Teste jede Achse als starre Achse
    logger.info(f"Keine konstante Achse gefunden (X-Spanne: {x_span*100:.1f}cm, Y-Spanne: {y_span*100:.1f}cm, Z-Spanne: {z_span*100:.1f}cm)")
    logger.info(f"Prüfe welche Achse am besten aus den anderen berechenbar ist (planare Validierung)...")
    
    best_axis = None
    best_error = float('inf')
    best_orientation = "sloped"
    axis_results = {}
    
    # Teste Z-Achse als starr: Z = f(X, Y)
    xs = np.array(x_vals)
    ys = np.array(y_vals)
    zs = np.array(z_vals)
    model_z = _fit_planar_surface(xs, ys, zs)
    if model_z is not None:
        # Berechne durchschnittlichen Fehler
        errors = []
        for p in valid_points:
            predicted_z = evaluate_surface_plane(model_z, p["x"], p["y"])
            error = abs(predicted_z - p["z"])
            errors.append(error)
        avg_error_z = float(np.mean(errors))
        max_error_z = float(np.max(errors))
        axis_results["z"] = {"avg": avg_error_z, "max": max_error_z}
        logger.info(
            f"  Z-Achse als starr (Z = f(X, Y)): "
            f"durchschnittlicher Fehler: {avg_error_z*100:.2f}cm, max Fehler: {max_error_z*100:.2f}cm"
        )
        if avg_error_z < best_error:
            best_error = avg_error_z
            best_axis = "z"
            best_orientation = "sloped"
    else:
        logger.info(f"  Z-Achse als starr: Kein planares Modell erstellbar")
    
    # Teste X-Achse als starr: X = f(Y, Z)
    model_x = _fit_planar_surface(ys, zs, xs)
    if model_x is not None:
        errors = []
        # Für X = f(Y, Z): x = slope_x*y + slope_y*z + intercept
        slope_x = model_x.get("slope_x", 0.0)  # Tatsächlich Koeffizient für Y
        slope_y = model_x.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
        intercept = model_x.get("intercept", 0.0)
        for p in valid_points:
            # Berechne X direkt: x = slope_x*y + slope_y*z + intercept
            predicted_x = slope_x * p["y"] + slope_y * p["z"] + intercept
            error = abs(predicted_x - p["x"])
            errors.append(error)
        avg_error_x = float(np.mean(errors))
        max_error_x = float(np.max(errors))
        axis_results["x"] = {"avg": avg_error_x, "max": max_error_x}
        logger.info(
            f"  X-Achse als starr (X = f(Y, Z)): "
            f"durchschnittlicher Fehler: {avg_error_x*100:.2f}cm, max Fehler: {max_error_x*100:.2f}cm"
        )
        if avg_error_x < best_error:
            best_error = avg_error_x
            best_axis = "x"
            best_orientation = "sloped"
    else:
        logger.info(f"  X-Achse als starr: Kein planares Modell erstellbar")
    
    # Teste Y-Achse als starr: Y = f(X, Z)
    model_y = _fit_planar_surface(xs, zs, ys)
    if model_y is not None:
        errors = []
        # Für Y = f(X, Z): y = slope_x*x + slope_y*z + intercept
        slope_x = model_y.get("slope_x", 0.0)
        slope_y = model_y.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
        intercept = model_y.get("intercept", 0.0)
        for p in valid_points:
            # Berechne Y direkt: y = slope_x*x + slope_y*z + intercept
            predicted_y = slope_x * p["x"] + slope_y * p["z"] + intercept
            error = abs(predicted_y - p["y"])
            errors.append(error)
        avg_error_y = float(np.mean(errors))
        max_error_y = float(np.max(errors))
        axis_results["y"] = {"avg": avg_error_y, "max": max_error_y}
        logger.info(
            f"  Y-Achse als starr (Y = f(X, Z)): "
            f"durchschnittlicher Fehler: {avg_error_y*100:.2f}cm, max Fehler: {max_error_y*100:.2f}cm"
        )
        if avg_error_y < best_error:
            best_error = avg_error_y
            best_axis = "y"
            best_orientation = "sloped"
    else:
        logger.info(f"  Y-Achse als starr: Kein planares Modell erstellbar")
    
    # Wenn eine Achse gefunden wurde, verwende sie
    if best_axis is not None:
        logger.info(
            f"✓ Starre Achse bestimmt: {best_axis.upper()}-Achse (schräg), "
            f"durchschnittlicher Fehler: {best_error*100:.2f}cm"
        )
        # Zeige Vergleich mit anderen Achsen
        for axis, results in axis_results.items():
            if axis != best_axis:
                logger.debug(
                    f"  Vergleich: {axis.upper()}-Achse hätte durchschnittlichen Fehler von "
                    f"{results['avg']*100:.2f}cm gehabt"
                )
        return best_axis, best_orientation
    
    # Fallback: Wenn keine Achse funktioniert, verwende Z als Standard
    logger.warning("Keine optimale starre Achse gefunden, verwende Z als Standard")
    return "z", "sloped"


def check_rigid_axis_validity(
    points: List[Dict[str, float]],
    rigid_axis: Literal["x", "y", "z"],
    orientation: Optional[str] = None,
) -> Tuple[bool, List[Tuple[int, str]]]:
    """
    Prüft ob alle Werte der starren Achse konsistent sind.
    
    Für horizontale/vertikale Flächen: Prüft ob alle Werte gleich sind (innerhalb Toleranz).
    Für schräge Flächen: Prüft ob die Punkte einer planaren Funktion folgen.
    
    Args:
        points: Liste von Punkten
        rigid_axis: Die starre Achse ("x", "y" oder "z")
        orientation: Die Orientierung der Fläche ("horizontal", "vertical_x", "vertical_y", "sloped")
        
    Returns:
        (is_valid, invalid_fields) - invalid_fields: Liste von (point_index, coord_name) für ungültige Felder
    """
    if len(points) < 3:
        return False, []
    
    # Sammle gültige Punkte mit allen drei Koordinaten
    valid_points = []
    for i, point in enumerate(points):
        x_val = point.get("x")
        y_val = point.get("y")
        z_val = point.get("z")
        if x_val is not None and y_val is not None and z_val is not None:
            try:
                valid_points.append({
                    "index": i,
                    "x": float(x_val),
                    "y": float(y_val),
                    "z": float(z_val),
                })
            except (ValueError, TypeError):
                pass
    
    # Wenn weniger als 3 Punkte mit allen Koordinaten, ist die Fläche ungültig
    if len(valid_points) < 3:
        invalid_fields = [(i, rigid_axis) for i in range(len(points)) if points[i].get(rigid_axis) is None]
        return False, invalid_fields
    
    # Für horizontale/vertikale Flächen: Prüfe ob alle Werte der starren Achse gleich sind
    # ABER: Auch hier können die Punkte leicht abweichen durch Rundung, daher verwenden wir
    # die gleiche planare Validierung wie bei schrägen Flächen
    if orientation in ("horizontal", "vertical_x", "vertical_y"):
        # Für horizontale/vertikale Flächen können wir auch planare Validierung verwenden
        # Das ist robuster gegen Rundungsfehler
        if rigid_axis == "z":
            # Z = f(X, Y) - sollte konstant sein für horizontale Flächen
            xs = np.array([p["x"] for p in valid_points])
            ys = np.array([p["y"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            
            # Versuche planares Modell zu erstellen
            all_points_dict = [
                {"x": p["x"], "y": p["y"], "z": p["z"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            if model is None:
                model = _fit_planar_surface(xs, ys, zs)
            
            if model is not None:
                invalid_fields = []
                max_error = 0.0
                for p in valid_points:
                    predicted_z = evaluate_surface_plane(model, p["x"], p["y"])
                    actual_z = p["z"]
                    error = abs(predicted_z - actual_z)
                    max_error = max(max_error, error)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"  Punkt {p['index']} ungültig: Z={actual_z:.4f}m, "
                            f"vorhergesagt={predicted_z:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if is_valid:
                    logger.debug(
                        f"Horizontale Fläche (Z-Achse) gültig: Alle {len(valid_points)} Punkte planar "
                        f"(max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                else:
                    logger.debug(
                        f"Horizontale Fläche (Z-Achse) ungültig: {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
        
        elif rigid_axis == "x":
            # X = f(Y, Z) - sollte konstant sein für vertikale X-Flächen
            ys = np.array([p["y"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            xs = np.array([p["x"] for p in valid_points])
            
            all_points_dict = [
                {"x": p["y"], "y": p["z"], "z": p["x"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            if model is None:
                model = _fit_planar_surface(ys, zs, xs)
            
            if model is not None:
                invalid_fields = []
                max_error = 0.0
                # Für X = f(Y, Z): x = slope_x*y + slope_y*z + intercept
                slope_x = model.get("slope_x", 0.0)  # Tatsächlich Koeffizient für Y
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne X direkt: x = slope_x*y + slope_y*z + intercept
                    predicted_x = slope_x * p["y"] + slope_y * p["z"] + intercept
                    actual_x = p["x"]
                    error = abs(predicted_x - actual_x)
                    max_error = max(max_error, error)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"  Punkt {p['index']} ungültig: X={actual_x:.4f}m, "
                            f"vorhergesagt={predicted_x:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if is_valid:
                    logger.debug(
                        f"Vertikale Fläche X (X-Achse) gültig: Alle {len(valid_points)} Punkte planar "
                        f"(max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                else:
                    logger.debug(
                        f"Vertikale Fläche X (X-Achse) ungültig: {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
        
        else:  # rigid_axis == "y"
            # Y = f(X, Z) - sollte konstant sein für vertikale Y-Flächen
            xs = np.array([p["x"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            ys = np.array([p["y"] for p in valid_points])
            
            all_points_dict = [
                {"x": p["x"], "y": p["z"], "z": p["y"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            if model is None:
                model = _fit_planar_surface(xs, zs, ys)
            
            if model is not None:
                invalid_fields = []
                max_error = 0.0
                # Für Y = f(X, Z): y = slope_x*x + slope_y*z + intercept
                slope_x = model.get("slope_x", 0.0)
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne Y direkt: y = slope_x*x + slope_y*z + intercept
                    predicted_y = slope_x * p["x"] + slope_y * p["z"] + intercept
                    actual_y = p["y"]
                    error = abs(predicted_y - actual_y)
                    max_error = max(max_error, error)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"  Punkt {p['index']} ungültig: Y={actual_y:.4f}m, "
                            f"vorhergesagt={predicted_y:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if is_valid:
                    logger.debug(
                        f"Vertikale Fläche Y (Y-Achse) gültig: Alle {len(valid_points)} Punkte planar "
                        f"(max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                else:
                    logger.debug(
                        f"Vertikale Fläche Y (Y-Achse) ungültig: {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
        
        # Fallback: Wenn kein planares Modell erstellt werden konnte, verwende einfache Gleichheitsprüfung
        axis_values = []
        valid_indices = []
        for p in valid_points:
            value = p[rigid_axis]
            axis_values.append(value)
            valid_indices.append(p["index"])
        
        axis_array = np.array(axis_values)
        mean_value = float(np.mean(axis_array))
        deviations = np.abs(axis_array - mean_value)
        
        invalid_fields = []
        for i, point in enumerate(points):
            value = point.get(rigid_axis)
            if value is None:
                invalid_fields.append((i, rigid_axis))
            else:
                try:
                    float_value = float(value)
                    found = False
                    for p in valid_points:
                        if p["index"] == i:
                            idx_in_valid = valid_indices.index(i)
                            if deviations[idx_in_valid] > PLANAR_VALIDATION_TOLERANCE:
                                invalid_fields.append((i, rigid_axis))
                            found = True
                            break
                    if not found:
                        invalid_fields.append((i, rigid_axis))
                except (ValueError, TypeError):
                    invalid_fields.append((i, rigid_axis))
        
        is_valid = len(invalid_fields) == 0
        return is_valid, invalid_fields
    
    # Für schräge Flächen: Prüfe ob die Punkte einer planaren Funktion folgen
    # (Die starre Achse sollte aus den anderen beiden Achsen berechenbar sein)
    else:  # orientation == "sloped" or None
        # Versuche planares Modell zu erstellen
        if rigid_axis == "z":
            # Z = f(X, Y)
            # Für schräge Flächen: Immer versuchen, ein planares Modell zu erstellen
            xs = np.array([p["x"] for p in valid_points])
            ys = np.array([p["y"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            
            # Versuche zuerst derive_surface_plane
            all_points_dict = [
                {"x": p["x"], "y": p["y"], "z": p["z"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            # Wenn kein Modell, verwende immer Least-Squares-Fitting (funktioniert immer für 3+ Punkte)
            if model is None:
                model = _fit_planar_surface(xs, ys, zs)
            
            if model is not None:
                # Prüfe ob alle Punkte der planaren Fläche folgen
                # Verwende größere Toleranz für Validierung (berücksichtigt Rundungsfehler)
                invalid_fields = []
                max_error = 0.0
                for p in valid_points:
                    predicted_z = evaluate_surface_plane(model, p["x"], p["y"])
                    actual_z = p["z"]
                    error = abs(predicted_z - actual_z)
                    max_error = max(max_error, error)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"Punkt {p['index']} weicht ab: Z={actual_z:.4f}m, "
                            f"vorhergesagt={predicted_z:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                # Prüfe auch Punkte mit None-Werten
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if is_valid:
                    logger.debug(
                        f"Schräge Fläche (Z-Achse) gültig: Alle {len(valid_points)} Punkte planar "
                        f"(max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                else:
                    logger.debug(
                        f"Schräge Fläche (Z-Achse): {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (max Fehler: {max_error*100:.2f}cm, Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
            else:
                # Fallback: Wenn kein planares Modell erstellt werden konnte (z.B. degenerierte Punkte),
                # prüfe ob die Punkte wenigstens annähernd planar sind durch einfache Abweichungsprüfung
                logger.debug(
                    f"Kann kein planares Modell erstellen für {len(valid_points)} Punkte "
                    f"(möglicherweise degeneriert), verwende Fallback-Validierung"
                )
                # Verwende einfache Abweichungsprüfung als Fallback
                axis_values = []
                valid_indices = []
                for p in valid_points:
                    value = p[rigid_axis]
                    axis_values.append(value)
                    valid_indices.append(p["index"])
                
                if len(axis_values) >= 2:
                    axis_array = np.array(axis_values)
                    mean_value = float(np.mean(axis_array))
                    deviations = np.abs(axis_array - mean_value)
                    max_deviation = float(np.max(deviations))
                    
                    # Wenn maximale Abweichung innerhalb Toleranz, ist die Fläche gültig
                    if max_deviation <= PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields = []
                        for i, point in enumerate(points):
                            if point.get(rigid_axis) is None:
                                invalid_fields.append((i, rigid_axis))
                        return len(invalid_fields) == 0, invalid_fields
                
                # Wenn auch das fehlschlägt, alle Punkte als ungültig markieren
                invalid_fields = [(i, rigid_axis) for i in range(len(points))]
                return False, invalid_fields
        
        elif rigid_axis == "x":
            # X = f(Y, Z)
            ys = np.array([p["y"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            xs = np.array([p["x"] for p in valid_points])
            
            # Versuche zuerst derive_surface_plane
            all_points_dict = [
                {"x": p["y"], "y": p["z"], "z": p["x"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            # Wenn kein Modell, verwende immer Least-Squares-Fitting
            if model is None:
                model = _fit_planar_surface(ys, zs, xs)
            
            if model is not None:
                invalid_fields = []
                # Für X = f(Y, Z): x = slope_x*y + slope_y*z + intercept
                # (slope_x ist hier der Koeffizient für Y, slope_y für Z!)
                slope_x = model.get("slope_x", 0.0)  # Tatsächlich Koeffizient für Y
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne X direkt: x = slope_x*y + slope_y*z + intercept
                    predicted_x = slope_x * p["y"] + slope_y * p["z"] + intercept
                    actual_x = p["x"]
                    error = abs(predicted_x - actual_x)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"Punkt {p['index']} weicht ab: X={actual_x:.4f}m, "
                            f"vorhergesagt={predicted_x:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if not is_valid:
                    logger.debug(
                        f"Schräge Fläche (X-Achse): {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
            else:
                # Fallback für X-Achse
                logger.debug(
                    f"Kann kein planares Modell erstellen für {len(valid_points)} Punkte (X-Achse), "
                    f"verwende Fallback-Validierung"
                )
                axis_values = []
                valid_indices = []
                for p in valid_points:
                    value = p[rigid_axis]
                    axis_values.append(value)
                    valid_indices.append(p["index"])
                
                if len(axis_values) >= 2:
                    axis_array = np.array(axis_values)
                    mean_value = float(np.mean(axis_array))
                    deviations = np.abs(axis_array - mean_value)
                    max_deviation = float(np.max(deviations))
                    
                    if max_deviation <= PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields = []
                        for i, point in enumerate(points):
                            if point.get(rigid_axis) is None:
                                invalid_fields.append((i, rigid_axis))
                        return len(invalid_fields) == 0, invalid_fields
                
                invalid_fields = [(i, rigid_axis) for i in range(len(points))]
                return False, invalid_fields
        
        else:  # rigid_axis == "y"
            # Y = f(X, Z)
            xs = np.array([p["x"] for p in valid_points])
            zs = np.array([p["z"] for p in valid_points])
            ys = np.array([p["y"] for p in valid_points])
            
            # Versuche zuerst derive_surface_plane
            all_points_dict = [
                {"x": p["x"], "y": p["z"], "z": p["y"]}
                for p in valid_points
            ]
            model, error = derive_surface_plane(all_points_dict, tol=PLANAR_VALIDATION_TOLERANCE)
            
            # Wenn kein Modell, verwende immer Least-Squares-Fitting
            if model is None:
                model = _fit_planar_surface(xs, zs, ys)
            
            if model is not None:
                invalid_fields = []
                # Für Y = f(X, Z): y = slope_x*x + slope_y*z + intercept
                # (slope_y ist hier der Koeffizient für Z, nicht für Y!)
                slope_x = model.get("slope_x", 0.0)
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne Y direkt: y = slope_x*x + slope_y*z + intercept
                    predicted_y = slope_x * p["x"] + slope_y * p["z"] + intercept
                    actual_y = p["y"]
                    error = abs(predicted_y - actual_y)
                    
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields.append((p["index"], rigid_axis))
                        logger.debug(
                            f"Punkt {p['index']} weicht ab: Y={actual_y:.4f}m, "
                            f"vorhergesagt={predicted_y:.4f}m, Fehler={error*100:.2f}cm"
                        )
                
                for i, point in enumerate(points):
                    if point.get(rigid_axis) is None:
                        invalid_fields.append((i, rigid_axis))
                
                is_valid = len(invalid_fields) == 0
                if not is_valid:
                    logger.debug(
                        f"Schräge Fläche (Y-Achse): {len(invalid_fields)} von {len(valid_points)} Punkten "
                        f"abweichend (Toleranz: {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm)"
                    )
                return is_valid, invalid_fields
            else:
                # Fallback für X-Achse
                logger.debug(
                    f"Kann kein planares Modell erstellen für {len(valid_points)} Punkte (X-Achse), "
                    f"verwende Fallback-Validierung"
                )
                axis_values = []
                valid_indices = []
                for p in valid_points:
                    value = p[rigid_axis]
                    axis_values.append(value)
                    valid_indices.append(p["index"])
                
                if len(axis_values) >= 2:
                    axis_array = np.array(axis_values)
                    mean_value = float(np.mean(axis_array))
                    deviations = np.abs(axis_array - mean_value)
                    max_deviation = float(np.max(deviations))
                    
                    if max_deviation <= PLANAR_VALIDATION_TOLERANCE:
                        invalid_fields = []
                        for i, point in enumerate(points):
                            if point.get(rigid_axis) is None:
                                invalid_fields.append((i, rigid_axis))
                        return len(invalid_fields) == 0, invalid_fields
                
                invalid_fields = [(i, rigid_axis) for i in range(len(points))]
                return False, invalid_fields


def _fix_rigid_axis_outliers(
    points: List[Dict[str, float]],
    rigid_axis: Literal["x", "y", "z"],
) -> List[Dict[str, float]]:
    """
    Passt abweichende Werte der starren Achse an, sodass alle Punkte eine planare Fläche bilden.
    
    Strategie:
    - Versuche eine planare Fläche aus ALLEN Punkten zu erstellen
    - Prüfe welche Punkte von dieser Fläche abweichen
    - Korrigiere abweichende Punkte, sodass sie auf der planaren Fläche liegen
    
    Args:
        points: Liste von Punkten
        rigid_axis: Die starre Achse ("x", "y" oder "z")
        
    Returns:
        Liste mit angepassten Punkten
    """
    if len(points) < 3:
        return points
    
    # Erstelle Kopie der Punkte
    fixed_points = [dict(p) for p in points]
    
    if rigid_axis == "z":
        # Für Z-Achse: Versuche planare Fläche Z = f(X, Y) aus ALLEN Punkten zu erstellen
        # Sammle alle Punkte mit gültigen X, Y, Z
        valid_points = []
        for i, point in enumerate(points):
            x_val = point.get("x")
            y_val = point.get("y")
            z_val = point.get("z")
            if x_val is not None and y_val is not None and z_val is not None:
                try:
                    valid_points.append({
                        "index": i,
                        "x": float(x_val),
                        "y": float(y_val),
                        "z": float(z_val),
                    })
                except (ValueError, TypeError):
                    pass
        
        if len(valid_points) >= 3:
            # Versuche planares Modell aus ALLEN Punkten zu erstellen
            # Verwende eine weniger strikte Toleranz, damit auch leicht nicht-planare Flächen erkannt werden
            all_points_dict = [
                {"x": p["x"], "y": p["y"], "z": p["z"]}
                for p in valid_points
            ]
            # Versuche zuerst mit strikter Toleranz
            model, error = derive_surface_plane(all_points_dict, tol=1e-4)
            
            # Wenn kein Modell gefunden, versuche mit weniger strikter Toleranz
            if model is None:
                model, error = derive_surface_plane(all_points_dict, tol=RIGID_AXIS_TOLERANCE * 10)
            
            # Wenn immer noch kein Modell, erstelle ein Modell durch Least-Squares-Fitting
            if model is None:
                # Erstelle ein planares Modell durch Least-Squares-Fitting
                xs = np.array([p["x"] for p in valid_points])
                ys = np.array([p["y"] for p in valid_points])
                zs = np.array([p["z"] for p in valid_points])
                plane_model = _fit_planar_surface(xs, ys, zs)
                if plane_model is not None:
                    model = plane_model
                    logger.debug("Planares Modell durch Least-Squares-Fitting erstellt (Z-Achse)")
            
            if model is not None:
                # Prüfe welche Punkte von der planaren Fläche abweichen und korrigiere sie
                outliers = []
                adjustments = []
                for p in valid_points:
                    predicted_z = evaluate_surface_plane(model, p["x"], p["y"])
                    actual_z = p["z"]
                    error = abs(predicted_z - actual_z)
                    
                    # Verwende PLANAR_VALIDATION_TOLERANCE für die Optimierung (nicht RIGID_AXIS_TOLERANCE)
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        # Punkt weicht ab - korrigiere ihn
                        old_value = fixed_points[p["index"]][rigid_axis]
                        # Runde den korrigierten Wert auf cm genau
                        corrected_value = round(predicted_z / CM_PRECISION) * CM_PRECISION
                        fixed_points[p["index"]][rigid_axis] = corrected_value
                        outliers.append(p["index"])
                        adjustments.append({
                            "index": p["index"],
                            "old": old_value,
                            "new": corrected_value,
                            "error": error,
                            "coords": (p["x"], p["y"], p["z"])
                        })
                
                if outliers:
                    logger.info(
                        f"Planare Fläche erstellt (Z-Achse): {len(outliers)} Ausreißer korrigiert "
                        f"(von {len(valid_points)} Punkten)"
                    )
                    for adj in adjustments:
                        logger.info(
                            f"  Punkt {adj['index']}: Z={adj['old']:.4f}m → {adj['new']:.4f}m "
                            f"(Abweichung: {adj['error']*100:.2f}cm, Koords: X={adj['coords'][0]:.3f}m, Y={adj['coords'][1]:.3f}m)"
                        )
                    return fixed_points
                else:
                    logger.debug(
                        f"Planare Fläche erstellt (Z-Achse): Alle {len(valid_points)} Punkte sind bereits konsistent"
                    )
                    return fixed_points
            else:
                logger.debug(
                    f"Planare Fläche konnte nicht erstellt werden (Z-Achse): {error if error else 'Unbekannter Fehler'}"
                )
    
    elif rigid_axis == "x":
        # Für X-Achse: Versuche planare Fläche X = f(Y, Z) aus ALLEN Punkten zu erstellen
        # Sammle alle Punkte mit gültigen X, Y, Z
        valid_points = []
        for i, point in enumerate(points):
            x_val = point.get("x")
            y_val = point.get("y")
            z_val = point.get("z")
            if x_val is not None and y_val is not None and z_val is not None:
                try:
                    valid_points.append({
                        "index": i,
                        "x": float(x_val),
                        "y": float(y_val),
                        "z": float(z_val),
                    })
                except (ValueError, TypeError):
                    pass
        
        if len(valid_points) >= 3:
            # Versuche planares Modell X = f(Y, Z) zu erstellen
            # Verwende Y und Z als Eingabe, X als Ausgabe
            all_points_dict = [
                {"x": p["y"], "y": p["z"], "z": p["x"]}  # Vertausche für X = f(Y, Z)
                for p in valid_points
            ]
            # Versuche zuerst mit strikter Toleranz
            model, error = derive_surface_plane(all_points_dict, tol=1e-4)
            
            # Wenn kein Modell gefunden, versuche mit weniger strikter Toleranz
            if model is None:
                model, error = derive_surface_plane(all_points_dict, tol=RIGID_AXIS_TOLERANCE * 10)
            
            # Wenn immer noch kein Modell, erstelle ein Modell durch Least-Squares-Fitting
            if model is None:
                # Erstelle ein planares Modell durch Least-Squares-Fitting
                ys = np.array([p["y"] for p in valid_points])
                zs = np.array([p["z"] for p in valid_points])
                xs = np.array([p["x"] for p in valid_points])
                plane_model = _fit_planar_surface(ys, zs, xs)  # X = f(Y, Z)
                if plane_model is not None:
                    model = plane_model
                    logger.debug("Planares Modell durch Least-Squares-Fitting erstellt (X-Achse)")
            
            if model is not None:
                # Prüfe welche Punkte von der planaren Fläche abweichen und korrigiere sie
                outliers = []
                adjustments = []
                # Für X = f(Y, Z): x = slope_x*y + slope_y*z + intercept
                slope_x = model.get("slope_x", 0.0)  # Tatsächlich Koeffizient für Y
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne X direkt: x = slope_x*y + slope_y*z + intercept
                    predicted_x = slope_x * p["y"] + slope_y * p["z"] + intercept
                    actual_x = p["x"]
                    error = abs(predicted_x - actual_x)
                    
                    # Verwende PLANAR_VALIDATION_TOLERANCE für die Optimierung (nicht RIGID_AXIS_TOLERANCE)
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        # Punkt weicht ab - korrigiere ihn
                        old_value = fixed_points[p["index"]][rigid_axis]
                        # Runde den korrigierten Wert auf cm genau
                        corrected_value = round(predicted_x / CM_PRECISION) * CM_PRECISION
                        fixed_points[p["index"]][rigid_axis] = corrected_value
                        outliers.append(p["index"])
                        adjustments.append({
                            "index": p["index"],
                            "old": old_value,
                            "new": corrected_value,
                            "error": error,
                            "coords": (p["x"], p["y"], p["z"])
                        })
                
                if outliers:
                    logger.info(
                        f"Planare Fläche erstellt (X-Achse): {len(outliers)} Ausreißer korrigiert "
                        f"(von {len(valid_points)} Punkten)"
                    )
                    for adj in adjustments:
                        logger.info(
                            f"  Punkt {adj['index']}: X={adj['old']:.4f}m → {adj['new']:.4f}m "
                            f"(Abweichung: {adj['error']*100:.2f}cm, Koords: Y={adj['coords'][1]:.3f}m, Z={adj['coords'][2]:.3f}m)"
                        )
                    return fixed_points
                else:
                    logger.debug(
                        f"Planare Fläche erstellt (X-Achse): Alle {len(valid_points)} Punkte sind bereits konsistent"
                    )
                    return fixed_points
            else:
                logger.debug(
                    f"Planare Fläche konnte nicht erstellt werden (X-Achse): {error if error else 'Unbekannter Fehler'}"
                )
    
    elif rigid_axis == "y":
        # Für Y-Achse: Versuche planare Fläche Y = f(X, Z) aus ALLEN Punkten zu erstellen
        # Sammle alle Punkte mit gültigen X, Y, Z
        valid_points = []
        for i, point in enumerate(points):
            x_val = point.get("x")
            y_val = point.get("y")
            z_val = point.get("z")
            if x_val is not None and y_val is not None and z_val is not None:
                try:
                    valid_points.append({
                        "index": i,
                        "x": float(x_val),
                        "y": float(y_val),
                        "z": float(z_val),
                    })
                except (ValueError, TypeError):
                    pass
        
        if len(valid_points) >= 3:
            # Versuche planares Modell Y = f(X, Z) zu erstellen
            # Verwende X und Z als Eingabe, Y als Ausgabe
            all_points_dict = [
                {"x": p["x"], "y": p["z"], "z": p["y"]}  # Vertausche für Y = f(X, Z)
                for p in valid_points
            ]
            # Versuche zuerst mit strikter Toleranz
            model, error = derive_surface_plane(all_points_dict, tol=1e-4)
            
            # Wenn kein Modell gefunden, versuche mit weniger strikter Toleranz
            if model is None:
                model, error = derive_surface_plane(all_points_dict, tol=RIGID_AXIS_TOLERANCE * 10)
            
            # Wenn immer noch kein Modell, erstelle ein Modell durch Least-Squares-Fitting
            if model is None:
                # Erstelle ein planares Modell durch Least-Squares-Fitting
                xs = np.array([p["x"] for p in valid_points])
                zs = np.array([p["z"] for p in valid_points])
                ys = np.array([p["y"] for p in valid_points])
                plane_model = _fit_planar_surface(xs, zs, ys)  # Y = f(X, Z)
                if plane_model is not None:
                    model = plane_model
                    logger.debug("Planares Modell durch Least-Squares-Fitting erstellt (Y-Achse)")
            
            if model is not None:
                # Prüfe welche Punkte von der planaren Fläche abweichen und korrigiere sie
                outliers = []
                adjustments = []
                # Für Y = f(X, Z): y = slope_x*x + slope_y*z + intercept
                slope_x = model.get("slope_x", 0.0)
                slope_y = model.get("slope_y", 0.0)  # Tatsächlich Koeffizient für Z
                intercept = model.get("intercept", 0.0)
                for p in valid_points:
                    # Berechne Y direkt: y = slope_x*x + slope_y*z + intercept
                    predicted_y = slope_x * p["x"] + slope_y * p["z"] + intercept
                    actual_y = p["y"]
                    error = abs(predicted_y - actual_y)
                    
                    # Verwende PLANAR_VALIDATION_TOLERANCE für die Optimierung (nicht RIGID_AXIS_TOLERANCE)
                    if error > PLANAR_VALIDATION_TOLERANCE:
                        # Punkt weicht ab - korrigiere ihn
                        old_value = fixed_points[p["index"]][rigid_axis]
                        # Runde den korrigierten Wert auf cm genau
                        corrected_value = round(predicted_y / CM_PRECISION) * CM_PRECISION
                        fixed_points[p["index"]][rigid_axis] = corrected_value
                        outliers.append(p["index"])
                        adjustments.append({
                            "index": p["index"],
                            "old": old_value,
                            "new": corrected_value,
                            "error": error,
                            "coords": (p["x"], p["y"], p["z"])
                        })
                
                if outliers:
                    logger.info(
                        f"Planare Fläche erstellt (Y-Achse): {len(outliers)} Ausreißer korrigiert "
                        f"(von {len(valid_points)} Punkten)"
                    )
                    for adj in adjustments:
                        logger.info(
                            f"  Punkt {adj['index']}: Y={adj['old']:.4f}m → {adj['new']:.4f}m "
                            f"(Abweichung: {adj['error']*100:.2f}cm, Koords: X={adj['coords'][0]:.3f}m, Z={adj['coords'][2]:.3f}m)"
                        )
                    return fixed_points
                else:
                    logger.debug(
                        f"Planare Fläche erstellt (Y-Achse): Alle {len(valid_points)} Punkte sind bereits konsistent"
                    )
                    return fixed_points
            else:
                logger.debug(
                    f"Planare Fläche konnte nicht erstellt werden (Y-Achse): {error if error else 'Unbekannter Fehler'}"
                )
    
    # Fallback: Wenn keine planare Fläche erstellt werden konnte, verwende Durchschnitt
    # Sammle gültige Werte der starren Achse
    axis_values = []
    valid_indices = []
    for i, point in enumerate(points):
        value = point.get(rigid_axis)
        if value is not None:
            try:
                axis_values.append(float(value))
                valid_indices.append(i)
            except (ValueError, TypeError):
                pass
    
    if len(axis_values) < 2:
        logger.debug(
            f"Fallback (Durchschnitt): Zu wenige gültige Werte ({len(axis_values)}) für {rigid_axis.upper()}-Achse"
        )
        return fixed_points
    
    # Berechne Mittelwert und Abweichungen
    axis_array = np.array(axis_values)
    mean_value = float(np.mean(axis_array))
    deviations = np.abs(axis_array - mean_value)
    
    # Finde abweichende Werte (> Toleranz)
    outlier_indices = []
    for idx_in_valid, deviation in enumerate(deviations):
        if deviation > RIGID_AXIS_TOLERANCE:
            outlier_indices.append(valid_indices[idx_in_valid])
    
    if not outlier_indices:
        logger.debug(
            f"Fallback (Durchschnitt): Keine Ausreißer gefunden für {rigid_axis.upper()}-Achse "
            f"(Mittelwert: {mean_value:.4f}m, max Abweichung: {np.max(deviations)*100:.2f}cm)"
        )
        return fixed_points
    
    # Berechne Durchschnitt nur aus nicht-abweichenden Werten
    non_outlier_values = [
        axis_values[valid_indices.index(i)]
        for i in valid_indices
        if i not in outlier_indices
    ]
    
    if len(non_outlier_values) >= 2:
        target_value = float(np.mean(non_outlier_values))
    elif len(non_outlier_values) == 1:
        target_value = non_outlier_values[0]
    else:
        target_value = mean_value
    
    # Setze abweichende Werte auf Zielwert (gerundet auf cm)
    adjustments = []
    target_value_rounded = round(target_value / CM_PRECISION) * CM_PRECISION
    for outlier_idx in outlier_indices:
        old_value = fixed_points[outlier_idx].get(rigid_axis)
        fixed_points[outlier_idx][rigid_axis] = target_value_rounded
        adjustments.append({
            "index": outlier_idx,
            "old": old_value,
            "new": target_value_rounded,
        })
    
    logger.info(
        f"Fallback (Durchschnitt): {len(outlier_indices)} Ausreißer auf {target_value_rounded:.4f}m gesetzt "
        f"(von {len(axis_values)} gültigen Werten, {len(non_outlier_values)} nicht-abweichende Werte)"
    )
    for adj in adjustments:
        logger.debug(
            f"  Punkt {adj['index']}: {rigid_axis.upper()}={adj['old']:.4f}m -> {adj['new']:.4f}m"
        )
    
    return fixed_points


def validate_and_optimize_surface(
    surface: SurfaceDefinition,
    *,
    round_to_cm: bool = True,
    remove_redundant: bool = True,
    optimize_invalid: bool = True,
) -> SurfaceValidationResult:
    """
    Validiert und optimiert eine Surface-Definition basierend auf starrer Achse.
    
    Neue Validierungslogik:
    1. Bestimmt Orientierung der Fläche (horizontal, vertikal, schräg)
    2. Definiert starre Achse (X, Y oder Z)
    3. Prüft ob alle Werte der starren Achse konsistent sind
    4. Passt abweichende Werte an, sodass alle Punkte eine planare Fläche bilden
    
    Args:
        surface: Die zu validierende Surface-Definition
        round_to_cm: Ob Koordinaten auf cm genau gerundet werden sollen
        remove_redundant: Ob redundante Punkte entfernt werden sollen
        optimize_invalid: Ob ungültige Surfaces optimiert werden sollen
        
    Returns:
        SurfaceValidationResult mit validierten/optimierten Punkten und starre Achse Info
    """
    points = list(surface.points)
    original_count = len(points)
    
    if len(points) < 3:
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=0,
            rounded_points_count=0,
            error_message=f"Surface hat nur {len(points)} Punkte, mindestens 3 erforderlich",
            rigid_axis=None,
            orientation=None,
            invalid_fields=[],
        )
    
    # Schritt 1: Entferne redundante Punkte (VOR Rundung, um präzise Vergleiche zu ermöglichen)
    removed_redundant = 0
    if remove_redundant:
        logger.debug(f"Entferne redundante Punkte von Surface '{surface.name}' ({surface.surface_id})...")
        original_point_count = len(points)
        points, removed_redundant = _remove_redundant_points(points)
        if removed_redundant > 0:
            logger.debug(f"  {removed_redundant} redundante Punkt(e) entfernt ({original_point_count} → {len(points)})")
    
    # Schritt 2: Bestimme Orientierung und starre Achse (VOR Rundung, um präzise Planaritätsprüfung zu ermöglichen)
    logger.info(f"Bestimme Orientierung für Surface '{surface.name}' ({surface.surface_id}) mit {len(points)} Punkten...")
    rigid_axis, orientation = determine_surface_orientation(points)
    
    if rigid_axis is None:
        # Runde auch bei Fehler, wenn gewünscht
        if round_to_cm:
            points = _round_points_to_cm(points)
        return SurfaceValidationResult(
            is_valid=False,
            optimized_points=points,
            removed_points_count=removed_redundant,
            rounded_points_count=len(points) if round_to_cm else 0,
            error_message="Konnte Orientierung der Fläche nicht bestimmen",
            rigid_axis=None,
            orientation=None,
            invalid_fields=[],
        )
    
    # Schritt 4: Prüfe ob starre Achse konsistent ist
    logger.info(
        f"Validiere Surface '{surface.name}' ({surface.surface_id}): "
        f"Orientierung={orientation}, Starre Achse={rigid_axis.upper()}"
    )
    is_valid, invalid_fields = check_rigid_axis_validity(points, rigid_axis, orientation)
    
    if is_valid:
        logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist gültig")
    else:
        logger.info(
            f"✗ Surface '{surface.name}' ({surface.surface_id}) ist ungültig: "
            f"{len(invalid_fields)} Punkt(e) haben Abweichungen > {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm"
        )
    
    # Schritt 5: Wenn ungültig und optimize_invalid=True, passe abweichende Werte an
    if not is_valid and optimize_invalid:
        logger.info(
            f"Optimiere Surface '{surface.name}' ({surface.surface_id}): "
            f"{len(invalid_fields)} Punkt(e) müssen korrigiert werden (Starre Achse: {rigid_axis.upper()})"
        )
        points = _fix_rigid_axis_outliers(points, rigid_axis)
        # Prüfe erneut nach Anpassung (VOR Rundung)
        logger.info(f"Validiere Surface '{surface.name}' ({surface.surface_id}) nach Optimierung...")
        is_valid, invalid_fields = check_rigid_axis_validity(points, rigid_axis, orientation)
        if is_valid:
            logger.info(f"✓ Surface '{surface.name}' ({surface.surface_id}) ist nach Optimierung gültig")
        else:
            logger.warning(
                f"✗ Surface '{surface.name}' ({surface.surface_id}) ist nach Optimierung immer noch ungültig: "
                f"{len(invalid_fields)} Punkt(e) haben Abweichungen > {PLANAR_VALIDATION_TOLERANCE*100:.1f}cm"
            )
    
    # Schritt 6: Rundung auf cm genau (NACH Validierung und Optimierung)
    rounded_count = 0
    if round_to_cm:
        logger.debug(f"Runde Surface '{surface.name}' ({surface.surface_id}) auf cm genau...")
        points = _round_points_to_cm(points)
        rounded_count = len(points)
        
        # Nach dem Runden: Prüfe erneut, ob die Fläche noch gültig ist
        # Rundung kann kleine Abweichungen verursachen (max 0.5mm pro Achse), aber sollte
        # innerhalb der Toleranz bleiben. Wenn die Fläche vorher gültig war und nach
        # Rundung nur kleine Abweichungen (< 1cm) entstehen, akzeptiere sie trotzdem.
        if is_valid:
            logger.debug(f"Validiere Surface '{surface.name}' ({surface.surface_id}) nach Rundung...")
            is_valid_after_rounding, invalid_fields_after_rounding = check_rigid_axis_validity(points, rigid_axis, orientation)
            if not is_valid_after_rounding and len(invalid_fields_after_rounding) > 0:
                # Berechne maximale Abweichung nach Rundung
                max_rounding_error = 0.0
                valid_points_list = [p for p in points if p.get("x") is not None and p.get("y") is not None and p.get("z") is not None]
                
                if len(valid_points_list) >= 3:
                    xs = np.array([p["x"] for p in valid_points_list])
                    ys = np.array([p["y"] for p in valid_points_list])
                    zs = np.array([p["z"] for p in valid_points_list])
                    
                    if rigid_axis == "z":
                        model = _fit_planar_surface(xs, ys, zs)
                        if model:
                            for idx, _ in invalid_fields_after_rounding:
                                point = points[idx]
                                if point.get("x") is not None and point.get("y") is not None and point.get("z") is not None:
                                    predicted = evaluate_surface_plane(model, point["x"], point["y"])
                                    error = abs(predicted - point["z"])
                                    max_rounding_error = max(max_rounding_error, error)
                    elif rigid_axis == "y":
                        model = _fit_planar_surface(xs, zs, ys)
                        if model:
                            slope_x = model.get("slope_x", 0.0)
                            slope_y = model.get("slope_y", 0.0)
                            intercept = model.get("intercept", 0.0)
                            for idx, _ in invalid_fields_after_rounding:
                                point = points[idx]
                                if point.get("x") is not None and point.get("y") is not None and point.get("z") is not None:
                                    predicted = slope_x * point["x"] + slope_y * point["z"] + intercept
                                    error = abs(predicted - point["y"])
                                    max_rounding_error = max(max_rounding_error, error)
                    elif rigid_axis == "x":
                        model = _fit_planar_surface(ys, zs, xs)
                        if model:
                            slope_x = model.get("slope_x", 0.0)
                            slope_y = model.get("slope_y", 0.0)
                            intercept = model.get("intercept", 0.0)
                            for idx, _ in invalid_fields_after_rounding:
                                point = points[idx]
                                if point.get("x") is not None and point.get("y") is not None and point.get("z") is not None:
                                    predicted = slope_x * point["y"] + slope_y * point["z"] + intercept
                                    error = abs(predicted - point["x"])
                                    max_rounding_error = max(max_rounding_error, error)
                
                # Wenn Rundungsfehler < 1cm, akzeptiere die Fläche trotzdem
                if max_rounding_error < 0.01:
                    logger.debug(
                        f"Surface '{surface.name}' ({surface.surface_id}) nach Rundung leicht abweichend "
                        f"(max Fehler: {max_rounding_error*100:.2f}cm), aber akzeptabel - als gültig markiert"
                    )
                    is_valid = True
                    invalid_fields = []
                else:
                    logger.debug(
                        f"Surface '{surface.name}' ({surface.surface_id}) nach Rundung ungültig "
                        f"(max Fehler: {max_rounding_error*100:.2f}cm)"
                    )
    
    error_msg = None
    if not is_valid:
        axis_name = {"x": "X", "y": "Y", "z": "Z"}[rigid_axis]
        # Verwende die richtige Toleranz je nach Orientierung
        tolerance = PLANAR_VALIDATION_TOLERANCE if orientation == "sloped" or orientation is None else RIGID_AXIS_TOLERANCE
        error_msg = (
            f"Starre {axis_name}-Achse ist nicht konsistent. "
            f"{len(invalid_fields)} Punkt(e) haben Abweichungen > {tolerance*100:.1f}cm"
        )
    
    if removed_redundant > 0 or rounded_count > 0:
        logger.info(
            f"Surface '{surface.name}' ({surface.surface_id}): "
            f"{rounded_count} Punkte gerundet, {removed_redundant} Punkte entfernt, "
            f"Orientierung: {orientation}, Starre Achse: {rigid_axis.upper()}"
        )
    
    return SurfaceValidationResult(
        is_valid=is_valid,
        optimized_points=points,
        removed_points_count=removed_redundant,
        rounded_points_count=rounded_count,
        error_message=error_msg,
        rigid_axis=rigid_axis,
        orientation=orientation,
        invalid_fields=invalid_fields,
    )


def _round_points_to_cm(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Rundet alle Koordinaten auf cm genau (0.01m).
    
    Args:
        points: Liste von Punkten mit 'x', 'y', 'z'
        
    Returns:
        Liste mit gerundeten Punkten
    """
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
    """
    Entfernt redundante Punkte (duplizierte oder sehr nahe beieinander liegende).
    
    Args:
        points: Liste von Punkten
        
    Returns:
        (bereinigte Punkte, Anzahl entfernte Punkte)
    """
    if len(points) < 2:
        return points, 0
    
    # Konvertiere zu numpy Array für effiziente Berechnung
    coords = np.array([
        [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
        for p in points
    ], dtype=float)
    
    # Prüfe ob Polygon geschlossen ist (erster und letzter Punkt identisch)
    # Bei geschlossenen Polygonen entfernen wir den letzten Punkt, da er redundant ist
    is_closed = False
    if len(points) > 0:
        first_original = coords[0]
        last_original = coords[-1]
        dist_closed = np.sqrt(np.sum((first_original - last_original) ** 2))
        is_closed = dist_closed <= REDUNDANT_POINT_TOLERANCE
    
    # Finde eindeutige Punkte (basierend auf Distanz)
    # Bei geschlossenen Polygonen überspringe den letzten Punkt (wird immer entfernt)
    end_idx = len(coords) - 1 if is_closed else len(coords)
    unique_indices = [0]  # Erster Punkt ist immer eindeutig
    removed_count = 0
    
    for i in range(1, end_idx):
        # Berechne minimale Distanz zu bereits hinzugefügten Punkten
        prev_coords = coords[unique_indices]
        distances = np.sqrt(np.sum((prev_coords - coords[i]) ** 2, axis=1))
        min_distance = np.min(distances)
        
        if min_distance > REDUNDANT_POINT_TOLERANCE:
            unique_indices.append(i)
        else:
            removed_count += 1
    
    # Bei geschlossenen Polygonen: Zähle den letzten Punkt als entfernt (da redundant)
    if is_closed:
        removed_count += 1
    
    unique_points = [points[i] for i in unique_indices]
    
    return unique_points, removed_count


def validate_surface_dict(
    surface_dict: Dict,
    *,
    round_to_cm: bool = True,
    remove_redundant: bool = True,
    optimize_invalid: bool = True,
) -> Dict:
    """
    Validiert und optimiert eine Surface-Definition als Dictionary.
    
    Args:
        surface_dict: Dictionary mit Surface-Daten (kann SurfaceDefinition oder Dict sein)
        round_to_cm: Ob Koordinaten auf cm genau gerundet werden sollen
        remove_redundant: Ob redundante Punkte entfernt werden sollen
        optimize_invalid: Ob ungültige Surfaces optimiert werden sollen
        
    Returns:
        Optimiertes Dictionary mit aktualisierten Punkten
    """
    # Konvertiere zu SurfaceDefinition falls nötig
    if isinstance(surface_dict, SurfaceDefinition):
        surface = surface_dict
    else:
        surface_id = surface_dict.get("surface_id", "unknown")
        surface = SurfaceDefinition.from_dict(surface_id, surface_dict)
    
    # Validiere und optimiere
    result = validate_and_optimize_surface(
        surface,
        round_to_cm=round_to_cm,
        remove_redundant=remove_redundant,
        optimize_invalid=optimize_invalid,
    )
    
    # Aktualisiere Punkte im Dictionary
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
    """
    Validiert und optimiert mehrere Surfaces.
    
    Args:
        surfaces: Dictionary von Surface-Definitionen {surface_id: surface_dict}
        round_to_cm: Ob Koordinaten auf cm genau gerundet werden sollen
        remove_redundant: Ob redundante Punkte entfernt werden sollen
        optimize_invalid: Ob ungültige Surfaces optimiert werden sollen
        
    Returns:
        Dictionary mit optimierten Surfaces
    """
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
            # Bei Fehler: Surface unverändert übernehmen
            validated[surface_id] = surface_data
    
    return validated

