"""Axis-Lines und Axis-Planes Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, Any, Callable

import numpy as np

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
from Module_LFO.Modules_Calculate.Functions import FunctionToolbox

DEBUG_OVERLAY_PERF = bool(int(__import__('os').environ.get("LFO_DEBUG_OVERLAY_PERF", "1")))


def get_max_surface_dimension(settings) -> float:
    """Berechnet die maximale Dimension (Breite oder Länge) aller nicht-versteckten Surfaces.
    
    Args:
        settings: Settings-Objekt
        
    Returns:
        float: max(Breite, Länge) * 1.5 der größten nicht-versteckten Surface, oder 0.0 wenn keine gefunden
    """
    surface_store = getattr(settings, 'surface_definitions', {})
    if not isinstance(surface_store, dict):
        return 0.0
    
    max_dimension = 0.0
    
    for surface_id, surface in surface_store.items():
        # Prüfe ob Surface nicht versteckt ist
        if isinstance(surface, SurfaceDefinition):
            hidden = surface.hidden
            points = surface.points
        else:
            hidden = surface.get('hidden', False)
            points = surface.get('points', [])
        
        # Überspringe versteckte Surfaces
        if hidden:
            continue
        
        # Berechne Breite und Länge für dieses Surface
        if points:
            try:
                width, length = FunctionToolbox.surface_dimensions(points)
                max_dim = max(float(width), float(length))
                max_dimension = max(max_dimension, max_dim)
            except Exception:  # noqa: BLE001
                continue
    
    # Multipliziere mit 1.5 (50% größer)
    return max_dimension * 1.5


def get_active_xy_surfaces(settings) -> List[Tuple[str, Any]]:
    """Sammelt alle aktiven Surfaces für XY-Berechnung (xy_enabled=True, enabled=True, hidden=False).
    
    Args:
        settings: Settings-Objekt
        
    Returns:
        Liste von (surface_id, surface) Tuples
    """
    active_surfaces = []
    surface_store = getattr(settings, 'surface_definitions', {})
    
    if not isinstance(surface_store, dict):
        return active_surfaces
    
    for surface_id, surface in surface_store.items():
        # Prüfe ob Surface aktiv ist
        if isinstance(surface, SurfaceDefinition):
            xy_enabled = getattr(surface, 'xy_enabled', True)
            enabled = surface.enabled
            hidden = surface.hidden
        else:
            xy_enabled = surface.get('xy_enabled', True)
            enabled = surface.get('enabled', False)
            hidden = surface.get('hidden', False)
        
        if xy_enabled and enabled and not hidden:
            active_surfaces.append((str(surface_id), surface))
    
    return active_surfaces


def get_surface_intersection_points_xz(
    y_const: float,
    surface: Any,
    settings: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Berechnet Schnittpunkte der Linie y=y_const mit dem Surface-Polygon (Projektion auf XZ-Ebene).
    
    Args:
        y_const: Konstante Y-Koordinate der Linie
        surface: SurfaceDefinition oder Dict mit Surface-Daten
        settings: Settings-Objekt für Zugriff auf aktuelle Surface-Daten
        
    Returns:
        tuple: (x_coords, z_coords) als Arrays oder None wenn keine Schnittpunkte
    """
    # Hole aktuelle Surface-Punkte
    if isinstance(surface, SurfaceDefinition):
        points = surface.points
    else:
        points = surface.get('points', [])
    
    if len(points) < 3:
        return None
    
    # Extrahiere alle Koordinaten
    points_3d = []
    for p in points:
        x = float(p.get('x', 0.0))
        y = float(p.get('y', 0.0))
        z = float(p.get('z', 0.0)) if p.get('z') is not None else 0.0
        points_3d.append((x, y, z))
    
    # Prüfe jede Kante des Polygons auf Schnitt mit Linie y=y_const
    intersection_points = []
    n = len(points_3d)
    for i in range(n):
        p1 = points_3d[i]
        p2 = points_3d[(i + 1) % n]  # Nächster Punkt (geschlossenes Polygon)
        
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Prüfe ob Kante die Linie y=y_const schneidet
        if (y1 <= y_const <= y2) or (y2 <= y_const <= y1):
            if abs(y2 - y1) > 1e-10:  # Vermeide Division durch Null
                # Berechne Schnittpunkt-Parameter t
                t = (y_const - y1) / (y2 - y1)
                
                # Berechne X- und Z-Koordinaten des Schnittpunkts
                x_intersect = x1 + t * (x2 - x1)
                z_intersect = z1 + t * (z2 - z1)
                
                intersection_points.append((x_intersect, z_intersect))
    
    if len(intersection_points) < 2:
        return None
    
    # Entferne Duplikate (Punkte die sehr nah beieinander sind)
    unique_points = []
    eps = 1e-6
    for x, z in intersection_points:
        is_duplicate = False
        for ux, uz in unique_points:
            if abs(x - ux) < eps and abs(z - uz) < eps:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append((x, z))
    
    if len(unique_points) < 2:
        return None
    
    # Extrahiere X- und Z-Koordinaten
    x_coords = np.array([p[0] for p in unique_points])
    z_coords = np.array([p[1] for p in unique_points])
    
    # Sortiere nach X-Koordinaten für konsistente Reihenfolge
    sort_indices = np.argsort(x_coords)
    x_coords = x_coords[sort_indices]
    z_coords = z_coords[sort_indices]
    
    return (x_coords, z_coords)


def get_surface_intersection_points_yz(
    x_const: float,
    surface: Any,
    settings: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Berechnet Schnittpunkte der Linie x=x_const mit dem Surface-Polygon (Projektion auf YZ-Ebene).
    
    Args:
        x_const: Konstante X-Koordinate der Linie
        surface: SurfaceDefinition oder Dict mit Surface-Daten
        settings: Settings-Objekt für Zugriff auf aktuelle Surface-Daten
        
    Returns:
        tuple: (y_coords, z_coords) als Arrays oder None wenn keine Schnittpunkte
    """
    # Hole aktuelle Surface-Punkte
    if isinstance(surface, SurfaceDefinition):
        points = surface.points
    else:
        points = surface.get('points', [])
    
    if len(points) < 3:
        return None
    
    # Extrahiere alle Koordinaten
    points_3d = []
    for p in points:
        x = float(p.get('x', 0.0))
        y = float(p.get('y', 0.0))
        z = float(p.get('z', 0.0)) if p.get('z') is not None else 0.0
        points_3d.append((x, y, z))
    
    # Prüfe jede Kante des Polygons auf Schnitt mit Linie x=x_const
    intersection_points = []
    n = len(points_3d)
    for i in range(n):
        p1 = points_3d[i]
        p2 = points_3d[(i + 1) % n]  # Nächster Punkt (geschlossenes Polygon)
        
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Prüfe ob Kante die Linie x=x_const schneidet
        if (x1 <= x_const <= x2) or (x2 <= x_const <= x1):
            if abs(x2 - x1) > 1e-10:  # Vermeide Division durch Null
                # Berechne Schnittpunkt-Parameter t
                t = (x_const - x1) / (x2 - x1)
                
                # Berechne Y- und Z-Koordinaten des Schnittpunkts
                y_intersect = y1 + t * (y2 - y1)
                z_intersect = z1 + t * (z2 - z1)
                
                intersection_points.append((y_intersect, z_intersect))
    
    if len(intersection_points) < 2:
        return None
    
    # Entferne Duplikate (Punkte die sehr nah beieinander sind)
    unique_points = []
    eps = 1e-6
    for y, z in intersection_points:
        is_duplicate = False
        for uy, uz in unique_points:
            if abs(y - uy) < eps and abs(z - uz) < eps:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append((y, z))
    
    if len(unique_points) < 2:
        return None
    
    # Extrahiere Y- und Z-Koordinaten
    y_coords = np.array([p[0] for p in unique_points])
    z_coords = np.array([p[1] for p in unique_points])
    
    # Sortiere nach Y-Koordinaten für konsistente Reihenfolge
    sort_indices = np.argsort(y_coords)
    y_coords = y_coords[sort_indices]
    z_coords = z_coords[sort_indices]
    
    return (y_coords, z_coords)


def draw_axis_planes(
    x_axis: float,
    y_axis: float,
    length: float,
    width: float,
    settings: Any,
    pv_module: Any,
    add_overlay_mesh_func: Callable,
    get_max_surface_dimension_func: Callable,
    planar_z_offset: float = 0.0,
) -> None:
    """Zeichnet halbtransparente, quadratische Flächen durch die X- und Y-Achse.
    
    Args:
        x_axis: X-Achsen-Position
        y_axis: Y-Achsen-Position
        length: Länge (wird nicht verwendet, wird aus Surfaces berechnet)
        width: Breite (wird nicht verwendet, wird aus Surfaces berechnet)
        settings: Settings-Objekt
        pv_module: PyVista Modul
        add_overlay_mesh_func: Funktion zum Hinzufügen von Overlay-Meshes
        get_max_surface_dimension_func: Funktion zum Berechnen der max Surface-Dimension
        planar_z_offset: Z-Offset für planare Overlays
    """
    # Berechne Größe basierend auf allen nicht-versteckten Surfaces
    L_base = get_max_surface_dimension_func(settings)
    if L_base <= 0.0:
        return
    
    # Faktor 3 der max Dimension: L_base ist bereits max_dim * 1.5, also L = L_base * 2 = max_dim * 3
    L = L_base * 2.0
    # Vertikale Höhe soll kürzer sein: Faktor 0.75 der Länge
    H = L * 0.75

    # Transparenz: Wert in Prozent (0–100), 10 % Standard
    transparency_pct = float(getattr(settings, "axis_3d_transparency", 10.0))
    transparency_pct = max(0.0, min(100.0, transparency_pct))
    # PyVista-Opacity: 1.0 = voll sichtbar, 0.0 = komplett transparent
    opacity = max(0.0, min(1.0, transparency_pct / 100.0))

    # Diskrete Auflösung der Fläche (nur wenige Stützpunkte nötig)
    n = 2

    # X-Achsen-Ebene: X-Z-Fläche bei y = y_axis
    try:
        x_vals = np.linspace(-L / 2.0, L / 2.0, n)
        # Vertikale Höhe: zentriert mit Faktor 0.75 der Länge
        z_vals = np.linspace(-H / 2.0, H / 2.0, n)
        X, Z = np.meshgrid(x_vals, z_vals)
        Y = np.full_like(X, y_axis)
        grid_x = pv_module.StructuredGrid(X, Y, Z)
        # Kleiner Offset in Z, damit die Fläche minimal über dem Plot liegt
        points = grid_x.points
        points[:, 2] += planar_z_offset
        grid_x.points = points
        add_overlay_mesh_func(
            grid_x,
            color="gray",
            opacity=opacity,
            category="axis_plane",
        )
    except Exception:  # noqa: BLE001
        pass

    # Y-Achsen-Ebene: Y-Z-Fläche bei x = x_axis
    try:
        y_vals = np.linspace(-L / 2.0, L / 2.0, n)
        # Vertikale Höhe: zentriert mit Faktor 0.75 der Länge
        z_vals = np.linspace(-H / 2.0, H / 2.0, n)
        Y2, Z2 = np.meshgrid(y_vals, z_vals)
        X2 = np.full_like(Y2, x_axis)
        grid_y = pv_module.StructuredGrid(X2, Y2, Z2)
        points2 = grid_y.points
        points2[:, 2] += planar_z_offset
        grid_y.points = points2
        add_overlay_mesh_func(
            grid_y,
            color="gray",
            opacity=opacity,
            category="axis_plane",
        )
    except Exception:  # noqa: BLE001
        pass

