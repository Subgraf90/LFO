"""Basis-Rendering-Logik für Overlays im 3D-SPL-Plot."""

from __future__ import annotations

from typing import List, Optional, Any

import numpy as np


def add_overlay_mesh(
    plotter: Any,
    mesh: Any,
    overlay_counter: int,
    overlay_actor_names: List[str],
    category_actors: dict,
    *,
    color: Optional[str] = None,
    opacity: float = 1.0,
    line_width: float = 2.0,
    scalars: Optional[str] = None,
    cmap: Optional[List[str]] = None,
    edge_color: Optional[str] = None,
    show_edges: bool = False,
    show_vertices: bool = False,
    line_pattern: Optional[int] = None,
    line_repeat: int = 1,
    category: str = 'generic',
    render_lines_as_tubes: Optional[bool] = None,
) -> str:
    """Fügt ein Overlay-Mesh zum Plotter hinzu.
    
    Args:
        plotter: PyVista Plotter
        mesh: PyVista Mesh
        overlay_counter: Zähler für Overlay-Namen (wird erhöht)
        overlay_actor_names: Liste der Overlay-Actor-Namen (wird aktualisiert)
        category_actors: Dictionary mit Category-Actors (wird aktualisiert)
        color: Farbe des Meshes
        opacity: Opazität (0.0-1.0)
        line_width: Linienbreite
        scalars: Scalar-Array-Name
        cmap: Colormap
        edge_color: Randfarbe
        show_edges: Ob Kanten angezeigt werden sollen
        show_vertices: Ob Vertices angezeigt werden sollen
        line_pattern: Linien-Pattern (für Stippling)
        line_repeat: Wiederholungsfaktor für Pattern
        category: Kategorie des Overlays
        render_lines_as_tubes: Ob Linien als Tubes gerendert werden sollen
        
    Returns:
        Actor-Name
    """
    name = f"overlay_{overlay_counter[0]}"
    overlay_counter[0] += 1
    kwargs = {
        'name': name,
        'opacity': opacity,
        'line_width': line_width,
        'smooth_shading': False,
        'show_scalar_bar': False,
        'reset_camera': False,
    }

    if scalars is not None:
        kwargs['scalars'] = scalars
        if cmap is not None:
            kwargs['cmap'] = cmap
    elif color is not None:
        kwargs['color'] = color

    if edge_color is not None:
        kwargs['edge_color'] = edge_color
        kwargs['show_edges'] = True
    elif show_edges:
        kwargs['show_edges'] = True
    
    if render_lines_as_tubes is not None:
        kwargs['render_lines_as_tubes'] = bool(render_lines_as_tubes)
    
    # Stelle sicher, dass keine Eckpunkte angezeigt werden (nur Linien)
    if not show_vertices and hasattr(mesh, 'lines') and mesh.lines is not None:
        kwargs['render_points_as_spheres'] = False
        kwargs['point_size'] = 0

    actor = plotter.add_mesh(mesh, **kwargs)
    
    # Erhöhe Picking-Priorität für Achsenlinien
    if category == 'axis':
        try:
            if hasattr(actor, 'SetPickable'):
                actor.SetPickable(True)
            if hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if prop:
                    prop.SetOpacity(1.0)
                    prop.SetLineWidth(line_width)
            actor.Modified()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    
    # Achsenflächen sollen für Mausereignisse transparent sein
    if category == 'axis_plane':
        try:
            if hasattr(actor, 'SetPickable'):
                actor.SetPickable(False)
            if hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if prop:
                    if hasattr(prop, 'PickableOff'):
                        prop.PickableOff()
        except Exception:  # noqa: BLE001
            pass
    
    # Stelle sicher, dass Edges angezeigt werden, wenn edge_color gesetzt wurde
    if edge_color is not None and hasattr(actor, 'prop') and actor.prop is not None:
        try:
            actor.prop.SetEdgeVisibility(True)
            actor.prop.SetRepresentationToSurface()
            if edge_color == 'red':
                actor.prop.SetEdgeColor(1, 0, 0)
            elif edge_color == 'black':
                actor.prop.SetEdgeColor(0, 0, 0)
            actor.Modified()
        except Exception:  # noqa: BLE001
            pass
    
    # Prüfe ob Mesh ein Tube-Mesh ist
    is_tube_mesh = render_lines_as_tubes is None
    
    # Stelle sicher, dass Vertices nicht angezeigt werden
    if not show_vertices and hasattr(actor, 'prop') and actor.prop is not None:
        try:
            actor.prop.render_points_as_spheres = False
            actor.prop.point_size = 0
        except Exception:  # noqa: BLE001
            pass
    
    # line_pattern nur bei echten Polylines anwenden, nicht bei Tube-Meshes
    if line_pattern is not None and not is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
        try:
            actor.prop.SetRenderLinesAsTubes(False)
            actor.prop.SetLineWidth(line_width)
            actor.prop.SetLineStipplePattern(int(line_pattern))
            actor.prop.SetLineStippleRepeatFactor(max(1, int(line_repeat)))
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    elif is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
        try:
            actor.prop.SetLineStipplePattern(0xFFFF)  # Durchgezogen
            actor.prop.SetLineStippleRepeatFactor(1)
        except Exception:  # noqa: BLE001
            pass
    
    overlay_actor_names.append(name)
    category_actors.setdefault(category, []).append(name)
    return name


def clear_overlays(plotter: Any, overlay_actor_names: List[str], category_actors: dict) -> None:
    """Löscht alle Overlays.
    
    Args:
        plotter: PyVista Plotter
        overlay_actor_names: Liste der Overlay-Actor-Namen
        category_actors: Dictionary mit Category-Actors
    """
    for name in overlay_actor_names:
        try:
            plotter.remove_actor(name)
        except KeyError:
            pass
    overlay_actor_names.clear()
    category_actors.clear()


def clear_category(
    plotter: Any,
    category: str,
    overlay_actor_names: List[str],
    category_actors: dict,
    speaker_actor_cache: Optional[dict] = None,
) -> List[str]:
    """Entfernt alle Actor einer Kategorie.
    
    Args:
        plotter: PyVista Plotter
        category: Kategorie-Name
        overlay_actor_names: Liste der Overlay-Actor-Namen
        category_actors: Dictionary mit Category-Actors
        speaker_actor_cache: Optional: Speaker-Actor-Cache (für 'speakers' Kategorie)
        
    Returns:
        Liste der entfernten Actor-Namen
    """
    actor_names = category_actors.pop(category, [])
    if not isinstance(actor_names, list):
        actor_names = list(actor_names) if actor_names else []
    
    for name in actor_names:
        try:
            plotter.remove_actor(name)
        except KeyError:
            pass
        else:
            if isinstance(overlay_actor_names, list):
                if name in overlay_actor_names:
                    overlay_actor_names.remove(name)
            else:
                overlay_actor_names = list(overlay_actor_names) if overlay_actor_names else []
                if name in overlay_actor_names:
                    overlay_actor_names.remove(name)
    
    if category == 'speakers' and speaker_actor_cache is not None:
        keys_to_remove = [key for key, info in speaker_actor_cache.items() if info.get('actor') in actor_names]
        for key in keys_to_remove:
            speaker_actor_cache.pop(key, None)
    
    return actor_names

