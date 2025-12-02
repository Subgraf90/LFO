"""Hilfsfunktionen für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Any

import numpy as np

try:
    import pyvista as pv
except Exception:  # noqa: BLE001
    pv = None  # type: ignore[assignment]


def has_valid_data(x, y, pressure) -> bool:
    """Prüft ob die Daten gültig sind.
    
    Args:
        x: X-Koordinaten
        y: Y-Koordinaten
        pressure: Druck-Werte
        
    Returns:
        True wenn Daten gültig sind, sonst False
    """
    if x is None or y is None or pressure is None:
        return False
    try:
        return len(x) > 0 and len(y) > 0 and len(pressure) > 0
    except TypeError:
        return False


def compute_surface_signature(x: np.ndarray, y: np.ndarray) -> tuple:
    """Berechnet eine Signatur für ein Surface basierend auf X- und Y-Koordinaten.
    
    Args:
        x: X-Koordinaten-Array
        y: Y-Koordinaten-Array
        
    Returns:
        Tuple mit Signatur-Daten (nx, x0, x1, ny, y0, y1)
    """
    if x.size == 0 or y.size == 0:
        return (0, 0, 0.0, 0.0, 0.0, 0.0)
    return (
        int(x.size),
        float(x[0]),
        float(x[-1]),
        int(y.size),
        float(y[0]),
        float(y[-1]),
    )


def quantize_to_steps(values: np.ndarray, step: float) -> np.ndarray:
    """Quantisiert Werte zu diskreten Schritten.
    
    Args:
        values: Array mit Werten
        step: Schrittweite
        
    Returns:
        Quantisierte Werte
    """
    if step <= 0:
        return values
    return np.round(values / step) * step


def to_float_array(values) -> np.ndarray:
    """Konvertiert Werte zu einem Float-Array.
    
    Args:
        values: Werte die konvertiert werden sollen
        
    Returns:
        Float-Array (1D)
    """
    if values is None:
        return np.empty(0, dtype=float)
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return np.empty(0, dtype=float)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(float)


def fallback_cmap_to_lut(_self, cmap, n_colors: int | None = None, flip: bool = False):
    """Fallback-Methode zum Konvertieren einer Colormap zu einer Lookup-Table.
    
    Args:
        _self: Self-Parameter (für MethodType)
        cmap: Colormap-Objekt
        n_colors: Anzahl der Farben
        flip: Ob die Colormap gespiegelt werden soll
        
    Returns:
        Lookup-Table
    """
    if pv is None:
        raise ImportError("PyVista ist nicht verfügbar")
    
    if isinstance(cmap, pv.LookupTable):
        lut = cmap
    else:
        lut = pv.LookupTable()
        lut.apply_cmap(cmap, n_values=n_colors or 256, flip=flip)
    if flip and isinstance(cmap, pv.LookupTable):
        lut.values[:] = lut.values[::-1]
    return lut


def compute_overlay_signatures(settings, container) -> dict[str, tuple]:
    """Erzeugt robuste Vergleichs-Signaturen für jede Overlay-Kategorie.
    
    Hintergrund:
    - Ziel ist, Änderungsdetektion ohne tiefe Objektvergleiche.
    - Wir wandeln relevante Settings in flache Tupel um, die sich leicht
      vergleichen lassen und keine PyVista-Objekte enthalten.
    
    Args:
        settings: Settings-Objekt
        container: Container-Objekt
        
    Returns:
        Dictionary mit Signaturen pro Kategorie
    """
    def _to_tuple(sequence) -> tuple:
        if sequence is None:
            return tuple()
        try:
            iterable = list(sequence)
        except TypeError:
            return (float(sequence),)
        result = []
        for item in iterable:
            try:
                value = float(item)
            except Exception:
                result.append(None)
            else:
                result.append(value)
        return tuple(result)
    
    def _to_str_tuple(sequence) -> tuple:
        if sequence is None:
            return tuple()
        try:
            iterable = list(sequence)
        except TypeError:
            return (str(sequence),)
        return tuple(str(item) for item in iterable)
    
    signatures: dict[str, tuple] = {}
    
    # Axis-Signatur
    try:
        x_axis = float(getattr(settings, 'position_x_axis', 0.0))
        y_axis = float(getattr(settings, 'position_y_axis', 0.0))
        signatures['axis'] = (x_axis, y_axis)
    except Exception:
        signatures['axis'] = (0.0, 0.0)
    
    # Impulse-Signatur
    try:
        measurement_size = float(getattr(settings, 'measurement_size', 3.0))
        impulse_points = getattr(settings, 'impulse_points', []) or []
        points_tuple = []
        for point in impulse_points:
            try:
                data = point.get('data', [])
                if len(data) >= 2:
                    points_tuple.append((round(float(data[0]), 4), round(float(data[1]), 4)))
            except Exception:
                continue
        points_tuple.sort()
        signatures['impulse'] = (round(measurement_size, 4), tuple(points_tuple))
    except Exception:
        signatures['impulse'] = (3.0, tuple())
    
    # Surfaces-Signatur
    try:
        surface_definitions = getattr(settings, 'surface_definitions', {})
        if isinstance(surface_definitions, dict):
            surfaces_list = []
            for surface_id in sorted(surface_definitions.keys()):
                surface_def = surface_definitions[surface_id]
                if isinstance(surface_def, type) and hasattr(surface_def, 'enabled'):
                    enabled = bool(getattr(surface_def, 'enabled', False))
                    hidden = bool(getattr(surface_def, 'hidden', False))
                else:
                    enabled = bool(surface_def.get('enabled', False) if isinstance(surface_def, dict) else False)
                    hidden = bool(surface_def.get('hidden', False) if isinstance(surface_def, dict) else False)
                surfaces_list.append((str(surface_id), enabled, hidden))
            signatures['surfaces'] = tuple(surfaces_list)
        else:
            signatures['surfaces'] = tuple()
    except Exception:
        signatures['surfaces'] = tuple()
    
    # Speakers-Signatur (vereinfacht)
    try:
        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        if isinstance(speaker_arrays, dict):
            array_ids = tuple(sorted(str(k) for k in speaker_arrays.keys()))
            signatures['speakers'] = array_ids
        else:
            signatures['speakers'] = tuple()
    except Exception:
        signatures['speakers'] = tuple()
    
    return signatures

