"""Caching-Logik für Overlay-Rendering im 3D-SPL-Plot."""

from __future__ import annotations

from typing import Optional, Tuple, Any, List
import hashlib

import numpy as np


def get_box_template(
    width: float,
    depth: float,
    height: float,
    box_template_cache: dict,
    box_face_cache: dict,
    pv_module: Any,
    compute_box_face_indices_func,
) -> Tuple[Any, dict, dict]:
    """Holt oder erstellt ein Box-Template aus dem Cache.
    
    Args:
        width: Breite der Box
        depth: Tiefe der Box
        height: Höhe der Box
        box_template_cache: Cache für Box-Templates
        box_face_cache: Cache für Box-Face-Indices
        pv_module: PyVista Modul
        compute_box_face_indices_func: Funktion zum Berechnen der Face-Indices
        
    Returns:
        Tuple mit (template, box_template_cache, box_face_cache)
    """
    key = (float(width), float(depth), float(height))
    template = box_template_cache.get(key)
    if template is None:
        template = pv_module.Box(bounds=(0.0, width, -depth, 0.0, 0.0, height))
        box_template_cache[key] = template
        box_face_cache[key] = compute_box_face_indices_func(template)
    elif key not in box_face_cache:
        box_face_cache[key] = compute_box_face_indices_func(template)
    return template, box_template_cache, box_face_cache


def compute_box_face_indices(mesh: Any) -> Tuple[Optional[int], Optional[int]]:
    """Berechnet die Front- und Back-Face-Indices für eine Box.
    
    Args:
        mesh: PyVista Mesh (Box)
        
    Returns:
        Tuple mit (front_idx, back_idx) oder (None, None) bei Fehler
    """
    try:
        faces = mesh.faces.reshape(-1, 5)
    except Exception:  # noqa: BLE001
        return (None, None)
    try:
        points = mesh.points
    except Exception:  # noqa: BLE001
        return (None, None)
    if faces.size == 0 or points.size == 0:
        return (None, None)
    face_y_values: List[float] = []
    for face in faces:
        indices = face[1:]
        try:
            y_vals = points[indices, 1]
        except Exception:  # noqa: BLE001
            return (None, None)
        face_y_values.append(float(np.mean(y_vals)))
    if not face_y_values:
        return (None, None)
    front_idx = int(np.argmax(face_y_values))
    back_idx = int(np.argmin(face_y_values))
    return (front_idx, back_idx)


def create_geometry_cache_key(
    speaker_array,
    index: int,
    speaker_name: str,
    configuration: str,
    cabinet_raw,
    array_value_func,
) -> str:
    """Erstellt einen Cache-Key für die Geometrie basierend auf relevanten Parametern.
    
    Args:
        speaker_array: Speaker-Array-Objekt
        index: Index des Speakers
        speaker_name: Name des Speakers
        configuration: Konfiguration (flown, stack)
        cabinet_raw: Cabinet-Daten
        array_value_func: Funktion zum Extrahieren von Array-Werten
        
    Returns:
        MD5-Hash-String als Cache-Key
    """
    # Sammle alle relevanten Parameter
    params = []
    params.append(str(speaker_name))
    params.append(str(configuration))
    
    # Cabinet-Daten (nur relevante Geometrie-Parameter)
    if cabinet_raw is not None:
        if isinstance(cabinet_raw, list):
            for cab in cabinet_raw:
                if isinstance(cab, dict):
                    for key in ['width', 'depth', 'front_height', 'back_height', 'configuration', 
                               'stack_layout', 'cardio', 'angle_point', 'x_offset', 'y_offset', 'z_offset']:
                        params.append(f"{key}:{cab.get(key, '')}")
        elif isinstance(cabinet_raw, dict):
            for key in ['width', 'depth', 'front_height', 'back_height', 'configuration',
                       'stack_layout', 'cardio', 'angle_point', 'x_offset', 'y_offset', 'z_offset']:
                params.append(f"{key}:{cabinet_raw.get(key, '')}")
    
    # Relevante Array-Parameter
    azimuth = array_value_func(getattr(speaker_array, 'source_azimuth', None), index)
    params.append(f"azimuth:{azimuth:.4f}" if azimuth is not None else "azimuth:0")
    
    angle_val = array_value_func(getattr(speaker_array, 'source_angle', None), index)
    params.append(f"angle:{angle_val:.4f}" if angle_val is not None else "angle:0")
    
    site_val = array_value_func(getattr(speaker_array, 'source_site', None), index)
    params.append(f"site:{site_val:.4f}" if site_val is not None else "site:0")
    
    # Anzahl der Quellen (wichtig für flown arrays)
    params.append(f"count:{getattr(speaker_array, 'number_of_sources', 0)}")
    
    # Erstelle Hash
    param_string = '|'.join(str(p) for p in params)
    return hashlib.md5(param_string.encode('utf-8')).hexdigest()


def speaker_signature_from_mesh(mesh: Any, exit_face_index: Optional[int]) -> tuple:
    """Erstellt eine Signatur für ein Speaker-Mesh.
    
    Args:
        mesh: PyVista Mesh
        exit_face_index: Index der Exit-Face oder None
        
    Returns:
        Tuple mit Signatur-Daten
    """
    bounds = getattr(mesh, 'bounds', (0.0,) * 6)
    if bounds is None:
        bounds = (0.0,) * 6
    rounded_bounds = tuple(round(float(value), 4) for value in bounds)
    n_points = int(getattr(mesh, 'n_points', 0))
    n_cells = int(getattr(mesh, 'n_cells', 0))
    exit_idx = int(exit_face_index) if exit_face_index is not None else None
    point_sample: tuple = ()
    try:
        points = getattr(mesh, 'points', None)
        if points is not None:
            sample = points[: min(5, len(points))]
            point_sample = tuple(round(float(coord), 4) for pt in np.asarray(sample) for coord in pt)
    except Exception:
        point_sample = ()
    return (rounded_bounds, n_points, n_cells, exit_idx, point_sample)

