"""Linien-Segmentierung für Dash-Dot-Patterns im 3D-SPL-Plot."""

from __future__ import annotations

from typing import List, Optional, Any

import numpy as np


def create_dash_dot_line_segments_optimized(
    points_3d: np.ndarray,
    cumulative_distances: np.ndarray,
    interpolate_func,
    dash_length: float = 1.0,
    dot_length: float = 0.2,
    gap_length: float = 0.3,
) -> List[np.ndarray]:
    """
    Optimierte Version: Teilt eine 3D-Linie in Strich-Punkt-Segmente auf.
    Verwendet größere Segmente und vereinfachte Interpolation für bessere Performance.
    
    Pattern: Strich - Lücke - Punkt - Lücke - Strich - ...
    
    Args:
        points_3d: Array von 3D-Punkten (N x 3)
        cumulative_distances: Kumulative Distanzen entlang der Linie (N)
        interpolate_func: Funktion zur Interpolation (_interpolate_line_segment_simple)
        dash_length: Länge eines Strichs in Metern (größer für weniger Segmente)
        dot_length: Länge eines Punkts in Metern
        gap_length: Länge einer Lücke in Metern
        
    Returns:
        Liste von Punkt-Arrays, die die sichtbaren Segmente (Striche und Punkte) darstellen
    """
    if len(points_3d) < 2:
        return []
    
    # Vereinfachte Berechnung: Verwende nur Start- und Endpunkt für kurze Linien
    if len(points_3d) == 2:
        total_length = np.linalg.norm(points_3d[1] - points_3d[0])
        if total_length <= dash_length:
            # Zu kurz für Pattern - zeichne als durchgezogene Linie
            return [points_3d]
    
    total_length = cumulative_distances[-1]
    
    if total_length <= 0:
        return []
    
    # Pattern: Strich (dash_length) - Lücke (gap_length) - Punkt (dot_length) - Lücke (gap_length) - ...
    segments = []
    current_pos = 0.0
    
    # Vereinfachte Interpolation: Verwende direkte lineare Interpolation zwischen benachbarten Punkten
    while current_pos < total_length:
        # Strich-Segment
        dash_start = current_pos
        dash_end = min(current_pos + dash_length, total_length)
        if dash_end > dash_start:
            dash_points = interpolate_func(points_3d, cumulative_distances, dash_start, dash_end)
            if len(dash_points) >= 2:
                segments.append(dash_points)
        current_pos = dash_end + gap_length
        
        if current_pos >= total_length:
            break
        
        # Punkt-Segment
        dot_start = current_pos
        dot_end = min(current_pos + dot_length, total_length)
        if dot_end > dot_start:
            dot_points = interpolate_func(points_3d, cumulative_distances, dot_start, dot_end)
            if len(dot_points) >= 2:
                segments.append(dot_points)
        current_pos = dot_end + gap_length
    
    return segments


def create_dash_dot_line_segments(
    points_3d: np.ndarray,
    cumulative_distances: np.ndarray,
    interpolate_func,
    dash_length: float = 0.5,
    dot_length: float = 0.1,
    gap_length: float = 0.2,
) -> List[np.ndarray]:
    """
    Teilt eine 3D-Linie in Strich-Punkt-Segmente auf.
    
    Pattern: Strich - Lücke - Punkt - Lücke - Strich - ...
    
    Args:
        points_3d: Array von 3D-Punkten (N x 3)
        cumulative_distances: Kumulative Distanzen entlang der Linie (N)
        interpolate_func: Funktion zur Interpolation (_interpolate_line_segment)
        dash_length: Länge eines Strichs in Metern
        dot_length: Länge eines Punkts in Metern
        gap_length: Länge einer Lücke in Metern
        
    Returns:
        Liste von Punkt-Arrays, die die sichtbaren Segmente (Striche und Punkte) darstellen
    """
    if len(points_3d) < 2:
        return []
    
    total_length = cumulative_distances[-1]
    
    if total_length <= 0:
        return []
    
    # Pattern: Strich (dash_length) - Lücke (gap_length) - Punkt (dot_length) - Lücke (gap_length) - ...
    segments = []
    current_pos = 0.0
    
    # Interpoliere Punkte entlang der Linie für präzise Segmentierung
    while current_pos < total_length:
        # Strich-Segment
        dash_start = current_pos
        dash_end = min(current_pos + dash_length, total_length)
        if dash_end > dash_start:
            dash_points = interpolate_func(points_3d, cumulative_distances, dash_start, dash_end)
            if len(dash_points) >= 2:
                segments.append(dash_points)
        current_pos = dash_end + gap_length
        
        if current_pos >= total_length:
            break
        
        # Punkt-Segment
        dot_start = current_pos
        dot_end = min(current_pos + dot_length, total_length)
        if dot_end > dot_start:
            dot_points = interpolate_func(points_3d, cumulative_distances, dot_start, dot_end)
            if len(dot_points) >= 2:
                segments.append(dot_points)
        current_pos = dot_end + gap_length
    
    return segments


def combine_line_segments(segments: List[np.ndarray], pv_module: Any) -> Optional[Any]:
    """
    Kombiniert mehrere Liniensegmente in ein einziges PolyData-Mesh für effizientes Rendering.
    
    Args:
        segments: Liste von Punkt-Arrays (jedes Array ist N x 3)
        pv_module: PyVista Modul
        
    Returns:
        PolyData-Mesh mit allen Segmenten oder None bei Fehler
    """
    if not segments:
        return None
    
    try:
        # Sammle alle Punkte und erstelle Lines-Array
        all_points = []
        all_lines = []
        point_offset = 0
        
        for segment in segments:
            if len(segment) < 2:
                continue
            
            # Füge Punkte hinzu
            all_points.append(segment)
            
            # Erstelle Line-Array für dieses Segment: [n, 0, 1, 2, ..., n-1]
            n_pts = len(segment)
            line_array = [n_pts] + [point_offset + i for i in range(n_pts)]
            all_lines.extend(line_array)
            
            point_offset += n_pts
        
        if not all_points:
            return None
        
        # Kombiniere alle Punkte
        combined_points = np.vstack(all_points)
        
        # Erstelle PolyData
        polyline = pv_module.PolyData(combined_points)
        polyline.lines = np.array(all_lines, dtype=np.int64)
        
        return polyline
    except Exception as e:  # noqa: BLE001
        print(f"[DEBUG combine_line_segments] Fehler: {e}")
        return None


def interpolate_line_segment_simple(
    points_3d: np.ndarray,
    cumulative_distances: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> np.ndarray:
    """
    Vereinfachte Interpolation: Verwendet nur Start- und Endpunkt für kurze Segmente.
    
    Args:
        points_3d: Array von 3D-Punkten (N x 3)
        cumulative_distances: Kumulative Distanzen entlang der Linie (N)
        start_dist: Start-Distanz
        end_dist: End-Distanz
        
    Returns:
        Array von interpolierten Punkten für den Segment
    """
    if len(points_3d) < 2:
        return points_3d
    
    # Finde Start- und End-Indices
    start_idx = np.searchsorted(cumulative_distances, start_dist, side='right') - 1
    end_idx = np.searchsorted(cumulative_distances, end_dist, side='right')
    
    start_idx = max(0, min(start_idx, len(points_3d) - 1))
    end_idx = max(0, min(end_idx, len(points_3d) - 1))
    
    # Für kurze Segmente: Verwende nur Start- und Endpunkt
    if end_idx - start_idx <= 1:
        # Interpoliere Start- und Endpunkt
        start_point = points_3d[start_idx]
        end_point = points_3d[min(end_idx, len(points_3d) - 1)]
        
        if start_dist > cumulative_distances[start_idx] and start_idx < len(points_3d) - 1:
            t = (start_dist - cumulative_distances[start_idx]) / (cumulative_distances[start_idx + 1] - cumulative_distances[start_idx])
            start_point = points_3d[start_idx] + t * (points_3d[start_idx + 1] - points_3d[start_idx])
        
        if end_dist < cumulative_distances[end_idx] and end_idx > 0:
            t = (end_dist - cumulative_distances[end_idx - 1]) / (cumulative_distances[end_idx] - cumulative_distances[end_idx - 1])
            end_point = points_3d[end_idx - 1] + t * (points_3d[end_idx] - points_3d[end_idx - 1])
        
        return np.array([start_point, end_point])
    
    # Für längere Segmente: Verwende alle Punkte dazwischen
    segment_points = [points_3d[start_idx]]
    for i in range(start_idx + 1, end_idx):
        segment_points.append(points_3d[i])
    segment_points.append(points_3d[end_idx])
    
    return np.array(segment_points)


def interpolate_line_segment(
    points_3d: np.ndarray,
    cumulative_distances: np.ndarray,
    start_dist: float,
    end_dist: float,
) -> np.ndarray:
    """
    Interpoliert einen Linienabschnitt zwischen start_dist und end_dist.
    
    Args:
        points_3d: Array von 3D-Punkten (N x 3)
        cumulative_distances: Kumulative Distanzen entlang der Linie (N)
        start_dist: Start-Distanz
        end_dist: End-Distanz
        
    Returns:
        Array von interpolierten Punkten für den Segment
    """
    if len(points_3d) < 2:
        return points_3d
    
    segment_points = []
    
    # Finde Start- und End-Indices
    start_idx = np.searchsorted(cumulative_distances, start_dist, side='right') - 1
    end_idx = np.searchsorted(cumulative_distances, end_dist, side='right')
    
    start_idx = max(0, min(start_idx, len(points_3d) - 1))
    end_idx = max(0, min(end_idx, len(points_3d) - 1))
    
    # Füge Start-Punkt hinzu (interpoliert falls nötig)
    if start_dist > cumulative_distances[start_idx] and start_idx < len(points_3d) - 1:
        # Interpoliere zwischen start_idx und start_idx+1
        t = (start_dist - cumulative_distances[start_idx]) / (cumulative_distances[start_idx + 1] - cumulative_distances[start_idx])
        start_point = points_3d[start_idx] + t * (points_3d[start_idx + 1] - points_3d[start_idx])
        segment_points.append(start_point)
    else:
        segment_points.append(points_3d[start_idx])
    
    # Füge alle Punkte zwischen start_idx+1 und end_idx hinzu
    for i in range(start_idx + 1, end_idx):
        segment_points.append(points_3d[i])
    
    # Füge End-Punkt hinzu (interpoliert falls nötig)
    if end_dist < cumulative_distances[end_idx] and end_idx > 0:
        # Interpoliere zwischen end_idx-1 und end_idx
        t = (end_dist - cumulative_distances[end_idx - 1]) / (cumulative_distances[end_idx] - cumulative_distances[end_idx - 1])
        end_point = points_3d[end_idx - 1] + t * (points_3d[end_idx] - points_3d[end_idx - 1])
        segment_points.append(end_point)
    elif end_idx < len(points_3d):
        segment_points.append(points_3d[end_idx])
    
    if len(segment_points) < 2:
        # Fallback: Gib mindestens 2 Punkte zurück
        if len(segment_points) == 1:
            segment_points.append(segment_points[0])
    
    return np.array(segment_points)

