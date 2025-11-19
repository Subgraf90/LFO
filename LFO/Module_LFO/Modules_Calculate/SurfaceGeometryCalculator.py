from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def derive_surface_plane(
    points: List[Dict[str, float]],
    *,
    tol: float = 1e-4,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Analysiert die Z-Werte einer Surface-Definition und bestimmt,
    ob die Fläche plan ist. Zulässig sind:

    - Konstante Höhe (alle Z-Werte identisch)
    - Lineare Steigung entlang der X-Achse (Z = a * X + b)
    - Lineare Steigung entlang der Y-Achse (Z = a * Y + b)

    Args:
        points: Liste von Surface-Punkten mit 'x', 'y', 'z'
        tol: Toleranz für numerische Vergleiche

    Returns:
        (model, error_message)
        - model: Dict mit Planar-Informationen oder None bei Fehler
        - error_message: Beschreibung, falls Fläche nicht plan ist
    """

    if not points:
        return (
            {
                "mode": "constant",
                "base": 0.0,
                "slope": 0.0,
                "intercept": 0.0,
            },
            None,
        )

    coords = np.array(
        [
            (
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            )
            for point in points
        ],
        dtype=float,
    )
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    z_vals = coords[:, 2]

    z_span = float(np.ptp(z_vals))  # max - min
    if z_span <= tol:
        base = float(np.mean(z_vals))
        return (
            {
                "mode": "constant",
                "base": base,
                "slope": 0.0,
                "intercept": base,
            },
            None,
        )

    # Prüfe Steigung entlang X: Alle Punkte mit gleichem X benötigen identisches Z
    if _is_axis_planar(x_vals, z_vals, tol):
        slope, intercept = _fit_linear_relation(x_vals, z_vals)
        return (
            {
                "mode": "x",
                "base": intercept,
                "slope": slope,
                "intercept": intercept,
            },
            None,
        )

    # Prüfe Steigung entlang Y: Alle Punkte mit gleichem Y benötigen identisches Z
    if _is_axis_planar(y_vals, z_vals, tol):
        slope, intercept = _fit_linear_relation(y_vals, z_vals)
        return (
            {
                "mode": "y",
                "base": intercept,
                "slope": slope,
                "intercept": intercept,
            },
            None,
        )

    error = (
        "Die Z-Werte der Fläche müssen entweder konstant sein oder nur entlang "
        "einer Achse (X oder Y) linear variieren."
    )
    return None, error


def build_planar_model(
    points: List[Dict[str, float]],
    *,
    tol: float = 1e-4,
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Bestimmt das beste planare Modell (konstant, X-Steigung, Y-Steigung) und
    gibt an, ob Anpassungen der bestehenden Z-Werte nötig sind.

    Returns:
        (model, needs_adjustment)
    """
    if not points:
        return (
            {
                "mode": "constant",
                "base": 0.0,
                "slope": 0.0,
                "intercept": 0.0,
            },
            False,
        )

    coords = np.array(
        [
            (
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            )
            for point in points
        ],
        dtype=float,
    )
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    z_vals = coords[:, 2]

    candidates = []

    def add_candidate(mode: str, slope: float, intercept: float, predicted: np.ndarray) -> None:
        residuals = z_vals - predicted
        mse = float(np.mean(residuals**2))
        max_err = float(np.max(np.abs(residuals)))
        candidates.append(
            {
                "mode": mode,
                "slope": float(slope),
                "intercept": float(intercept),
                "base": float(intercept) if mode != "constant" else float(intercept),
                "mse": mse,
                "max_err": max_err,
            }
        )

    # Konstante Fläche
    base = float(np.mean(z_vals)) if len(z_vals) else 0.0
    predicted_const = np.full_like(z_vals, base)
    add_candidate("constant", 0.0, base, predicted_const)

    # Steigung entlang X (nur wenn Variation vorhanden)
    if np.ptp(x_vals) > tol:
        slope_x, intercept_x = _fit_linear_relation(x_vals, z_vals)
        predicted_x = slope_x * x_vals + intercept_x
        add_candidate("x", slope_x, intercept_x, predicted_x)

    # Steigung entlang Y
    if np.ptp(y_vals) > tol:
        slope_y, intercept_y = _fit_linear_relation(y_vals, z_vals)
        predicted_y = slope_y * y_vals + intercept_y
        add_candidate("y", slope_y, intercept_y, predicted_y)

    # Wähle Modell mit geringster MSE
    best_model = min(candidates, key=lambda c: c["mse"])
    needs_adjustment = best_model["max_err"] > tol

    model = {
        "mode": best_model["mode"],
        "slope": best_model["slope"],
        "intercept": best_model["intercept"],
        "base": best_model["base"],
    }
    return model, needs_adjustment


def evaluate_surface_plane(model: Dict[str, float], x: float, y: float) -> float:
    """
    Berechnet den Z-Wert einer Fläche anhand des Planar-Modells.
    """
    mode = model.get("mode")
    if mode == "constant":
        return float(model.get("base", 0.0))
    if mode == "x":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return float(slope * x + intercept)
    if mode == "y":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return float(slope * y + intercept)
    return float(model.get("base", 0.0))


def _is_axis_planar(axis_values: np.ndarray, z_values: np.ndarray, tol: float) -> bool:
    """
    Prüft, ob Z nur von einer Achse abhängt.
    """
    rounded_axis = np.round(axis_values, decimals=6)
    unique_vals = np.unique(rounded_axis)
    if unique_vals.size < 2:
        # Alle Punkte haben praktisch gleiche Axis-Koordinate → bereits über konstante Fläche abgedeckt
        return False

    for value in unique_vals:
        mask = rounded_axis == value
        z_span = np.ptp(z_values[mask])
        if z_span > tol:
            return False
    return True


def _fit_linear_relation(axis_values: np.ndarray, z_values: np.ndarray) -> Tuple[float, float]:
    """
    Bestimmt lineare Relation z = slope * axis + intercept (Least Squares).
    """
    if np.ptp(axis_values) <= 1e-9:
        return 0.0, float(np.mean(z_values))
    slope, intercept = np.polyfit(axis_values, z_values, 1)
    return float(slope), float(intercept)

