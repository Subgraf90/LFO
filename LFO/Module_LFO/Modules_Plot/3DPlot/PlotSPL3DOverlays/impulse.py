"""Impulse-Points Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import List, Optional, Tuple, Any

import numpy as np


def compute_impulse_state(settings) -> tuple:
    """Berechnet den State für Impulse-Points.
    
    Args:
        settings: Settings-Objekt
        
    Returns:
        Tuple mit (measurement_size, impulse_points)
    """
    measurement_size = float(getattr(settings, 'measurement_size', 3.0))
    impulse_points_raw = getattr(settings, 'impulse_points', []) or []
    points: List[tuple] = []

    for point in impulse_points_raw:
        try:
            data = point['data']
            x_val, y_val = data
        except Exception:
            continue
        try:
            points.append((round(float(x_val), 4), round(float(y_val), 4)))
        except Exception:
            continue

    points.sort()
    return (round(measurement_size, 4), tuple(points))


def draw_impulse_points(
    plotter: Any,
    pv_module: Any,
    settings,
    add_overlay_mesh_func,
    clear_category_func,
    category_actors: dict,
    last_impulse_state: Optional[tuple],
) -> Optional[tuple]:
    """Zeichnet Impulse-Points als Kegel im 3D-Plot.
    
    Args:
        plotter: PyVista Plotter
        pv_module: PyVista Modul
        settings: Settings-Objekt
        add_overlay_mesh_func: Funktion zum Hinzufügen von Overlay-Meshes
        clear_category_func: Funktion zum Löschen einer Kategorie
        category_actors: Dictionary mit Category-Actors
        last_impulse_state: Letzter bekannter State
        
    Returns:
        Neuer State oder None wenn unverändert
    """
    current_state = compute_impulse_state(settings)
    existing_names = category_actors.get('impulse', [])
    if last_impulse_state == current_state and existing_names:
        return last_impulse_state

    clear_category_func('impulse')
    _, impulse_points = current_state
    if not impulse_points:
        return current_state

    size = getattr(settings, 'measurement_size', 3.0)
    radius = max(size / 5.0, 0.3)
    height = radius * 2.5

    for x_val, y_val in impulse_points:
        try:
            x = float(x_val)
            y = float(y_val)
        except Exception:
            continue
        base_z = 0.0
        center_z = base_z + (height / 2.0)
        cone = pv_module.Cone(
            center=(float(x), float(y), center_z),
            direction=(0.0, 0.0, 1.0),
            height=height,
            radius=radius,
        )
        add_overlay_mesh_func(cone, color='red', opacity=0.85, category='impulse')
    
    return current_state

