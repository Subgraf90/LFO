"""Picking-Logik für den 3D-SPL-Plot."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PyQt5 import QtCore


def point_in_polygon(x: float, y: float, polygon_points: List[Dict[str, float]]) -> bool:
    """Prüft, ob ein Punkt (x, y) innerhalb eines Polygons liegt (Ray-Casting-Algorithmus).
    
    Args:
        x: X-Koordinate des Punktes
        y: Y-Koordinate des Punktes
        polygon_points: Liste von Polygon-Punkten mit 'x' und 'y' Keys
        
    Returns:
        True wenn Punkt im Polygon liegt, sonst False
    """
    if len(polygon_points) < 3:
        return False
    
    # Extrahiere X- und Y-Koordinaten
    px = np.array([p.get("x", 0.0) for p in polygon_points])
    py = np.array([p.get("y", 0.0) for p in polygon_points])
    
    # Ray-Casting-Algorithmus
    n = len(px)
    inside = False
    boundary_eps = 1e-6
    
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        
        # Prüfe ob Strahl von (x,y) nach rechts die Kante schneidet
        if ((yi > y) != (yj > y)) and (
            x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi + boundary_eps
        ):
            inside = not inside
        
        # Prüfe ob Punkt direkt auf der Kante liegt
        dx = xj - xi
        dy = yj - yi
        segment_len = math.hypot(dx, dy)
        if segment_len > 0:
            dist = abs(dy * (x - xi) - dx * (y - yi)) / segment_len
            if dist <= boundary_eps:
                proj = ((x - xi) * dx + (y - yi) * dy) / (segment_len * segment_len)
                if -boundary_eps <= proj <= 1 + boundary_eps:
                    return True
        j = i
    
    return inside


def ray_triangle_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float = 1e-8,
) -> Optional[float]:
    """Berechnet die Schnittstelle eines Rays mit einem Dreieck (Möller-Trumbore).
    
    Args:
        origin: Ray-Startpunkt
        direction: Ray-Richtung (normalisiert)
        v0, v1, v2: Dreiecks-Vertices
        eps: Toleranz für numerische Fehler
        
    Returns:
        t-Wert (Distanz entlang des Rays) oder None wenn kein Schnittpunkt
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) < eps:
        return None
    inv_det = 1.0 / det
    tvec = origin - v0
    u = float(np.dot(tvec, pvec) * inv_det)
    if u < 0.0 or u > 1.0:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec) * inv_det)
    if v < 0.0 or u + v > 1.0:
        return None
    t = float(np.dot(edge2, qvec) * inv_det)
    if t <= 0.0:
        return None
    return t


def get_z_from_mesh(mesh: Any, x_pos: float, y_pos: float) -> Optional[float]:
    """Extrahiert die Z-Koordinate aus einem Mesh an einer bestimmten X/Y-Position.
    
    Args:
        mesh: PyVista Mesh
        x_pos: X-Position
        y_pos: Y-Position
        
    Returns:
        Z-Koordinate oder None wenn nicht gefunden
    """
    if mesh is None or not hasattr(mesh, 'points'):
        return None
    
    try:
        points = np.asarray(mesh.points, dtype=float)
        if points.size == 0:
            return None
        
        # Finde den nächstgelegenen Punkt
        xy_points = points[:, :2]
        query_point = np.array([x_pos, y_pos])
        distances = np.linalg.norm(xy_points - query_point, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Prüfe ob Punkt nah genug ist (Toleranz: 1m)
        if distances[nearest_idx] > 1.0:
            return None
        
        return float(points[nearest_idx, 2])
    except Exception:
        return None

