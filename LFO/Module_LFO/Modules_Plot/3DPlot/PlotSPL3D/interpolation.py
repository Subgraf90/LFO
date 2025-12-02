"""Interpolations-Methoden für den 3D-SPL-Plot."""

from __future__ import annotations

import numpy as np


def bilinear_interpolate_grid(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """
    Bilineare Interpolation auf einem regulären Grid.
    Interpoliert zwischen den 4 umgebenden Grid-Punkten für glatte Farbübergänge.
    Ideal für Gradient-Modus.

    Args:
        source_x: 1D-Array der X-Koordinaten (nx)
        source_y: 1D-Array der Y-Koordinaten (ny)
        values: 2D-Array (ny, nx) mit SPL-Werten
        xq, yq: 1D-Arrays mit Abfragepunkten
        
    Returns:
        Interpolierte Werte als 1D-Array
    """
    source_x = np.asarray(source_x, dtype=float).reshape(-1)
    source_y = np.asarray(source_y, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    xq = np.asarray(xq, dtype=float).reshape(-1)
    yq = np.asarray(yq, dtype=float).reshape(-1)

    ny, nx = vals.shape
    if ny != len(source_y) or nx != len(source_x):
        raise ValueError(
            f"_bilinear_interpolate_grid: Shape mismatch values={vals.shape}, "
            f"expected=({len(source_y)}, {len(source_x)})"
        )

    # Randbehandlung: Punkte außerhalb des Grids werden auf den Rand geclippt
    x_min, x_max = source_x[0], source_x[-1]
    y_min, y_max = source_y[0], source_y[-1]
    xq_clip = np.clip(xq, x_min, x_max)
    yq_clip = np.clip(yq, y_min, y_max)

    # Finde Indizes für bilineare Interpolation
    # Für jeden Abfragepunkt finden wir die 4 umgebenden Grid-Punkte
    idx_x = np.searchsorted(source_x, xq_clip, side="right") - 1
    idx_y = np.searchsorted(source_y, yq_clip, side="right") - 1
    
    # Clamp auf gültigen Bereich
    idx_x = np.clip(idx_x, 0, nx - 2)  # -2 weil wir idx_x+1 brauchen
    idx_y = np.clip(idx_y, 0, ny - 2)  # -2 weil wir idx_y+1 brauchen
    
    # Hole die 4 umgebenden Punkte
    x0 = source_x[idx_x]
    x1 = source_x[idx_x + 1]
    y0 = source_y[idx_y]
    y1 = source_y[idx_y + 1]
    
    # Werte an den 4 Ecken
    f00 = vals[idx_y, idx_x]
    f10 = vals[idx_y, idx_x + 1]
    f01 = vals[idx_y + 1, idx_x]
    f11 = vals[idx_y + 1, idx_x + 1]
    
    # Berechne Interpolationsgewichte
    dx = x1 - x0
    dy = y1 - y0
    # Vermeide Division durch Null
    dx = np.where(dx > 1e-10, dx, 1.0)
    dy = np.where(dy > 1e-10, dy, 1.0)
    
    wx = (xq_clip - x0) / dx
    wy = (yq_clip - y0) / dy
    
    # Bilineare Interpolation: f(x,y) = (1-wx)(1-wy)*f00 + wx(1-wy)*f10 + (1-wx)wy*f01 + wx*wy*f11
    result = (
        (1 - wx) * (1 - wy) * f00 +
        wx * (1 - wy) * f10 +
        (1 - wx) * wy * f01 +
        wx * wy * f11
    )
    
    return result


def nearest_interpolate_grid(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """
    Nearest-Neighbor Interpolation auf einem regulären Grid.
    Jeder Abfragepunkt erhält exakt den Wert des nächstgelegenen Grid-Punkts.
    Behält die Randbehandlung (Clipping) bei.
    Ideal für Color step-Modus (harte Stufen).

    Args:
        source_x: 1D-Array der X-Koordinaten (nx)
        source_y: 1D-Array der Y-Koordinaten (ny)
        values: 2D-Array (ny, nx) mit SPL-Werten
        xq, yq: 1D-Arrays mit Abfragepunkten
        
    Returns:
        Interpolierte Werte als 1D-Array
    """
    source_x = np.asarray(source_x, dtype=float).reshape(-1)
    source_y = np.asarray(source_y, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float)
    xq = np.asarray(xq, dtype=float).reshape(-1)
    yq = np.asarray(yq, dtype=float).reshape(-1)

    ny, nx = vals.shape
    if ny != len(source_y) or nx != len(source_x):
        raise ValueError(
            f"_nearest_interpolate_grid: Shape mismatch values={vals.shape}, "
            f"expected=({len(source_y)}, {len(source_x)})"
        )

    # Randbehandlung: Punkte außerhalb des Grids werden auf den Rand geclippt
    x_min, x_max = source_x[0], source_x[-1]
    y_min, y_max = source_y[0], source_y[-1]
    xq_clip = np.clip(xq, x_min, x_max)
    yq_clip = np.clip(yq, y_min, y_max)

    # Finde nächstgelegenen Index für X
    idx_x = np.searchsorted(source_x, xq_clip, side="left")
    idx_x = np.clip(idx_x, 0, nx - 1)
    # Korrektur: wirklich nächsten Nachbarn wählen (links/rechts vergleichen)
    left_x = idx_x - 1
    right_x = idx_x
    left_x = np.clip(left_x, 0, nx - 1)
    dist_left_x = np.abs(xq_clip - source_x[left_x])
    dist_right_x = np.abs(xq_clip - source_x[right_x])
    use_left_x = dist_left_x < dist_right_x
    idx_x[use_left_x] = left_x[use_left_x]

    # Finde nächstgelegenen Index für Y
    idx_y = np.searchsorted(source_y, yq_clip, side="left")
    idx_y = np.clip(idx_y, 0, ny - 1)
    # Korrektur: wirklich nächsten Nachbarn wählen (oben/unten vergleichen)
    left_y = idx_y - 1
    right_y = idx_y
    left_y = np.clip(left_y, 0, ny - 1)
    dist_left_y = np.abs(yq_clip - source_y[left_y])
    dist_right_y = np.abs(yq_clip - source_y[right_y])
    use_left_y = dist_left_y < dist_right_y
    idx_y[use_left_y] = left_y[use_left_y]

    # Weise jedem Abfragepunkt den Wert des nächstgelegenen Grid-Punkts zu
    result = vals[idx_y, idx_x]
    return result

