"""Speaker-Overlay-Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import List, Optional, Tuple, Any, Callable

import numpy as np


def build_cabinet_lookup(container) -> dict:
    """Erstellt ein Lookup-Dictionary für Cabinet-Daten.
    
    Args:
        container: Container-Objekt
        
    Returns:
        Dictionary mit Speaker-Namen als Keys und Cabinet-Daten als Values
    """
    lookup: dict[str, object] = {}
    if container is None:
        return lookup

    data = getattr(container, 'data', None)
    if not isinstance(data, dict):
        return lookup

    names = data.get('speaker_names') or []
    cabinets = data.get('cabinet_data') or []

    for name, cabinet in zip(names, cabinets):
        if not isinstance(name, str):
            continue
        lookup[name] = cabinet
        lookup.setdefault(name.lower(), cabinet)

    alias_mapping = getattr(container, '_speaker_name_mapping', {})
    if isinstance(alias_mapping, dict):
        for alias, actual in alias_mapping.items():
            if actual in lookup and isinstance(alias, str):
                lookup.setdefault(alias, lookup[actual])
                lookup.setdefault(alias.lower(), lookup[actual])

    return lookup


def get_speaker_info_from_actor(
    actor: Any,
    actor_name: str,
    speaker_actor_cache: dict,
    renderer: Any,
) -> Optional[Tuple[str, int]]:
    """Extrahiert Array-ID und Speaker-Index aus einem Speaker-Actor.
    
    Args:
        actor: VTK Actor
        actor_name: Actor-Name
        speaker_actor_cache: Cache mit Speaker-Actor-Informationen
        renderer: PyVista Renderer
        
    Returns:
        Tuple[str, int] oder None: (array_id, speaker_index) wenn gefunden
    """
    try:
        if not renderer or not actor:
            return None
        
        # Durchsuche alle Cache-Einträge
        matching_entries = []
        
        for key, info in speaker_actor_cache.items():
            # Versuche zuerst actor_obj (direktes Objekt)
            cached_actor_obj = info.get('actor_obj')
            if cached_actor_obj is actor:
                array_id, speaker_idx, geom_idx = key
                matching_entries.append((array_id, speaker_idx, geom_idx, 'actor_obj'))
            
            # Fallback: Versuche über Actor-Namen
            cached_actor_name = info.get('actor')
            if cached_actor_name:
                cached_actor = renderer.actors.get(cached_actor_name)
                if cached_actor is actor:
                    array_id, speaker_idx, geom_idx = key
                    if not any(entry[0] == array_id and entry[1] == speaker_idx and entry[2] == geom_idx for entry in matching_entries):
                        matching_entries.append((array_id, speaker_idx, geom_idx, 'actor_name'))
        
        # Wenn Einträge gefunden wurden, gib den ersten zurück
        if matching_entries:
            array_id, speaker_idx, geom_idx, match_type = matching_entries[0]
            return (str(array_id), int(speaker_idx))
        
        # Fallback: Durchsuche alle Renderer-Actors
        for renderer_actor_name, renderer_actor in renderer.actors.items():
            if renderer_actor is actor:
                for cache_key, cache_info in speaker_actor_cache.items():
                    cached_actor_name = cache_info.get('actor')
                    if not cached_actor_name:
                        continue
                    cached_actor_from_renderer = renderer.actors.get(cached_actor_name)
                    if cached_actor_from_renderer is actor:
                        array_id, speaker_idx, geom_idx = cache_key
                        return (str(array_id), int(speaker_idx))
                break
        
        return None
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return None


def update_speaker_highlights(
    settings: Any,
    speaker_actor_cache: dict,
    plotter: Any,
) -> None:
    """Aktualisiert die Edge-Color für hervorgehobene Lautsprecher.
    
    Args:
        settings: Settings-Objekt
        speaker_actor_cache: Cache mit Speaker-Actor-Informationen
        plotter: PyVista Plotter
    """
    try:
        highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
        highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
        highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])
        
        if highlight_array_ids:
            highlight_array_ids_str = [str(aid) for aid in highlight_array_ids]
        elif highlight_array_id:
            highlight_array_ids_str = [str(highlight_array_id)]
        else:
            highlight_array_ids_str = []
        
        if not highlight_array_ids_str and not highlight_indices:
            # Keine Highlights - setze alle auf schwarz
            for key, info in speaker_actor_cache.items():
                actor_name = info.get('actor')
                if actor_name:
                    try:
                        actor = plotter.renderer.actors.get(actor_name)
                        if actor:
                            prop = actor.GetProperty()
                            if prop:
                                prop.SetEdgeColor(0, 0, 0)  # Schwarz
                                prop.SetEdgeVisibility(True)
                    except Exception:  # noqa: BLE001
                        pass
            return
        
        # Aktualisiere Edge-Color für hervorgehobene Lautsprecher
        for key, info in speaker_actor_cache.items():
            array_id, speaker_idx, geom_idx = key
            array_id_str = str(array_id)
            
            # Prüfe ob dieser Speaker hervorgehoben werden soll
            is_highlighted = False
            if highlight_indices:
                highlight_indices_str = [(str(aid), int(idx)) for aid, idx in highlight_indices]
                is_highlighted = (array_id_str, int(speaker_idx)) in highlight_indices_str
            elif highlight_array_ids_str:
                is_highlighted = array_id_str in highlight_array_ids_str
            
            # Versuche zuerst actor_obj (direktes Objekt), dann actor_name
            actor = info.get('actor_obj')
            actor_name = info.get('actor')
            
            if actor is None and actor_name:
                actor = plotter.renderer.actors.get(actor_name)
            
            if actor:
                try:
                    prop = actor.GetProperty()
                    if prop:
                        if is_highlighted:
                            prop.SetEdgeColor(1, 0, 0)  # Rot
                            prop.SetLineWidth(3.0)
                        else:
                            prop.SetEdgeColor(0, 0, 0)  # Schwarz
                            prop.SetLineWidth(1.5)
                        prop.SetEdgeVisibility(True)
                        prop.SetRepresentationToSurface()
                        actor.Modified()
                except Exception:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
        
        # Render-Update triggern
        try:
            plotter.render()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def draw_speakers(
    plotter: Any,
    pv_module: Any,
    settings: Any,
    container: Any,
    cabinet_lookup: dict,
    build_speaker_geometries_func: Callable,
    add_overlay_mesh_func: Callable,
    clear_category_func: Callable,
    speaker_actor_cache: dict,
    category_actors: dict,
    to_float_array_func: Callable,
    update_speaker_actor_func: Callable,
    speaker_signature_from_mesh_func: Callable,
    remove_actor_func: Callable,
) -> None:
    """Zeichnet Speaker-Overlays im 3D-Plot.
    
    Vollständige Implementierung aus PlotSPL3DOverlays.py Zeilen 998-1289.
    
    Args:
        plotter: PyVista Plotter
        pv_module: PyVista Modul
        settings: Settings-Objekt
        container: Container-Objekt
        cabinet_lookup: Cabinet-Lookup-Dictionary
        build_speaker_geometries_func: Funktion zum Erstellen von Speaker-Geometrien
        add_overlay_mesh_func: Funktion zum Hinzufügen von Overlay-Meshes
        clear_category_func: Funktion zum Löschen einer Kategorie
        speaker_actor_cache: Cache für Speaker-Actors
        category_actors: Dictionary mit Category-Actors
        to_float_array_func: Funktion zum Konvertieren zu Float-Array
        update_speaker_actor_func: Funktion zum Aktualisieren von Speaker-Actors
        speaker_signature_from_mesh_func: Funktion zum Erstellen von Speaker-Signaturen
        remove_actor_func: Funktion zum Entfernen von Actors
    """
    speaker_arrays = getattr(settings, 'speaker_arrays', {})
    if not isinstance(speaker_arrays, dict):
        clear_category_func('speakers')
        return

    body_color = '#b0b0b0'
    exit_color = '#4d4d4d'
    
    # Hole Highlight-IDs für rote Umrandung (unterstütze Liste von IDs)
    highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
    highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
    highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])
    
    # Konvertiere zu Liste von Strings für konsistenten Vergleich
    if highlight_array_ids:
        highlight_array_ids_str = [str(aid) for aid in highlight_array_ids]
    elif highlight_array_id:
        highlight_array_ids_str = [str(highlight_array_id)]
    else:
        highlight_array_ids_str = []

    old_cache = speaker_actor_cache.copy()
    new_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
    new_actor_names: List[str] = []

    for array_name, speaker_array in speaker_arrays.items():
        if getattr(speaker_array, 'hide', False):
            continue

        configuration = getattr(speaker_array, 'configuration', '')
        config_lower = configuration.lower() if isinstance(configuration, str) else ''

        # Hole Array-Positionen
        array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
        array_pos_y = getattr(speaker_array, 'array_position_y', 0.0)
        array_pos_z = getattr(speaker_array, 'array_position_z', 0.0)

        if config_lower == 'flown':
            # Für Flown: Array-Positionen sind bereits in source_position_x/y/z_flown enthalten (absolute Positionen)
            xs = to_float_array_func(getattr(speaker_array, 'source_position_x', None))
            ys = to_float_array_func(
                getattr(
                    speaker_array,
                    'source_position_calc_y',
                    getattr(speaker_array, 'source_position_y', None),
                )
            )
            zs = to_float_array_func(getattr(speaker_array, 'source_position_z_flown', None))
        else:
            # Für Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
            # source_position_calc_x/y/z ist nur für Berechnungen, nicht für den Plot
            xs_raw = to_float_array_func(getattr(speaker_array, 'source_position_x', None))
            ys_raw = to_float_array_func(getattr(speaker_array, 'source_position_y', None))
            zs_raw = to_float_array_func(getattr(speaker_array, 'source_position_z_stack', None))
            
            # Addiere Array-Positionen zu den Speaker-Positionen
            if xs_raw.size > 0:
                xs = xs_raw + array_pos_x
            else:
                xs = xs_raw
            if ys_raw.size > 0:
                ys = ys_raw + array_pos_y
            else:
                ys = ys_raw
            if zs_raw.size > 0:
                zs = zs_raw + array_pos_z
            else:
                zs = zs_raw

        min_len = min(xs.size, ys.size, zs.size)
        if min_len == 0:
            continue

        valid_mask = np.isfinite(xs[:min_len]) & np.isfinite(ys[:min_len]) & np.isfinite(zs[:min_len])
        valid_indices = np.nonzero(valid_mask)[0]
        array_identifier = str(getattr(speaker_array, 'name', array_name))
        # Speichere die Array-ID für späteren Zugriff (ID ist der Key im speaker_arrays Dict)
        # Konvertiere zu String für konsistenten Vergleich
        array_id = str(array_name)  # array_name ist bereits die ID (Key im Dict), konvertiere zu String

        for idx in valid_indices:
            x = xs[idx]
            y = ys[idx]
            z = zs[idx]
            
            # Prüfe ob dieser Speaker hervorgehoben werden soll
            is_highlighted = False
            if highlight_indices:
                highlight_indices_str = [(str(aid), int(idx_val)) for aid, idx_val in highlight_indices]
                is_highlighted = (array_id, int(idx)) in highlight_indices_str
            elif highlight_array_ids_str:
                # Prüfe ob Array-ID in der Liste ist
                is_highlighted = array_id in highlight_array_ids_str

            geometries = build_speaker_geometries_func(
                speaker_array,
                idx,
                float(x),
                float(y),
                float(z),
                cabinet_lookup,
                container,
            )
            
            # MESH MERGING: Kombiniere alle Geometrien eines Speakers zu einem Mesh
            
            if geometries and len(geometries) > 1:
                # Merge alle Body-Meshes und sammle Exit-Face-Indices
                merged_mesh = geometries[0][0].copy(deep=True)
                cell_offset = 0
                exit_face_indices = []
                
                # Sammle erste Exit-Face
                if geometries[0][1] is not None:
                    exit_face_indices.append(geometries[0][1])
                cell_offset = merged_mesh.n_cells
                
                # Merge restliche Geometrien
                for geom_idx in range(1, len(geometries)):
                    body_mesh, exit_face_idx = geometries[geom_idx]
                    
                    # Kombiniere die Meshes
                    merged_mesh = merged_mesh + body_mesh
                    
                    # Sammle Exit-Face mit korrigiertem Offset
                    if exit_face_idx is not None:
                        exit_face_indices.append(cell_offset + exit_face_idx)
                    
                    cell_offset += body_mesh.n_cells
                
                # Erstelle Scalar-Array für ALLE Exit-Faces
                if exit_face_indices:
                    # Erstelle ein Scalar-Array: 0 für Body, 1 für Exit-Faces
                    scalars = np.zeros(merged_mesh.n_cells, dtype=int)
                    for exit_idx in exit_face_indices:
                        if exit_idx < merged_mesh.n_cells:
                            scalars[exit_idx] = 1
                    merged_mesh.cell_data['speaker_face'] = scalars
                    # Signalisiere, dass wir Scalars haben (exit_face = -1 als Flag)
                    merged_exit_face = -1
                else:
                    merged_exit_face = None
                
                # Ersetze geometries mit dem gemerged mesh
                geometries = [(merged_mesh, merged_exit_face)]
            
            if not geometries:
                sphere = pv_module.Sphere(radius=0.6, center=(float(x), float(y), float(z)), theta_resolution=32, phi_resolution=32)
                key = (array_id, int(idx), 'fallback')
                existing = old_cache.get(key)
                if existing:
                    signature = existing.get('signature')
                    current_signature = speaker_signature_from_mesh_func(sphere, None)
                    if signature == current_signature:
                        # Aktualisiere actor_obj falls nötig
                        if 'actor_obj' not in existing:
                            actor_name = existing.get('actor')
                            existing['actor_obj'] = plotter.renderer.actors.get(actor_name) if actor_name else None
                        new_cache[key] = existing
                        if existing['actor'] not in new_actor_names:
                            new_actor_names.append(existing['actor'])
                        continue
                    existing_mesh = existing['mesh']
                    existing_mesh.deep_copy(sphere)
                    update_speaker_actor_func(existing['actor'], existing_mesh, None, body_color, exit_color)
                    existing['signature'] = current_signature
                    # Aktualisiere actor_obj falls nötig
                    if 'actor_obj' not in existing:
                        actor_name = existing.get('actor')
                        existing['actor_obj'] = plotter.renderer.actors.get(actor_name) if actor_name else None
                    new_cache[key] = existing
                    if existing['actor'] not in new_actor_names:
                        new_actor_names.append(existing['actor'])
                else:
                    mesh_to_add = sphere.copy(deep=True)
                    # Setze Edge-Color basierend auf Highlight-Status
                    edge_color = 'red' if is_highlighted else 'black'
                    actor_name = add_overlay_mesh_func(
                        mesh_to_add,
                        color=body_color,
                        opacity=1.0,
                        edge_color=edge_color,
                        line_width=3.0 if is_highlighted else 1.5,
                        category='speakers',
                    )
                    # Hole das Actor-Objekt aus dem Renderer
                    actor_obj = plotter.renderer.actors.get(actor_name) if actor_name else None
                    new_cache[key] = {
                        'mesh': mesh_to_add,
                        'actor': actor_name,
                        'actor_obj': actor_obj,  # Speichere Actor-Objekt für direkten Vergleich
                        'signature': speaker_signature_from_mesh_func(mesh_to_add, None),
                    }
                    if actor_name not in new_actor_names:
                        new_actor_names.append(actor_name)
            else:
                for geom_idx, (body_mesh, exit_face_index) in enumerate(geometries):
                    key = (array_id, int(idx), geom_idx)
                    existing = old_cache.get(key)
                    if existing:
                        signature = existing.get('signature')
                        current_signature = speaker_signature_from_mesh_func(body_mesh, exit_face_index)
                        if signature == current_signature:
                            # Aktualisiere actor_obj falls nötig
                            if 'actor_obj' not in existing:
                                actor_name = existing.get('actor')
                                existing['actor_obj'] = plotter.renderer.actors.get(actor_name) if actor_name else None
                            new_cache[key] = existing
                            if existing['actor'] not in new_actor_names:
                                new_actor_names.append(existing['actor'])
                            continue
                        # Signature hat sich geändert - aktualisiere existing
                        existing_mesh = existing['mesh']
                        existing_mesh.deep_copy(body_mesh)
                        update_speaker_actor_func(existing['actor'], existing_mesh, exit_face_index, body_color, exit_color)
                        existing['signature'] = current_signature
                        # Aktualisiere actor_obj falls nötig
                        if 'actor_obj' not in existing:
                            actor_name = existing.get('actor')
                            existing['actor_obj'] = plotter.renderer.actors.get(actor_name) if actor_name else None
                        new_cache[key] = existing
                        if existing['actor'] not in new_actor_names:
                            new_actor_names.append(existing['actor'])
                        continue

                    # Kein existing - erstelle neues Mesh
                    mesh_to_add = body_mesh.copy(deep=True)
                    # Prüfe ob Mesh bereits Scalars hat (merged mesh mit exit_face_index = -1)
                    has_scalars = 'speaker_face' in mesh_to_add.cell_data
                    
                    # Setze Edge-Color basierend auf Highlight-Status
                    edge_color = 'red' if is_highlighted else 'black'
                    line_width = 3.0 if is_highlighted else 1.5
                    
                    if exit_face_index == -1 or has_scalars:
                        # Mesh hat bereits Scalars (merged mesh) - nutze diese
                        actor_name = add_overlay_mesh_func(
                            mesh_to_add,
                            scalars='speaker_face',
                            cmap=[body_color, exit_color],
                            opacity=1.0,
                            edge_color=edge_color,
                            line_width=line_width,
                            category='speakers',
                        )
                    elif exit_face_index is not None and mesh_to_add.n_cells > 0:
                        # Einzelnes Mesh - erstelle Scalars für eine Exit-Face
                        scalars = np.zeros(mesh_to_add.n_cells, dtype=int)
                        scalars[int(exit_face_index)] = 1
                        mesh_to_add.cell_data['speaker_face'] = scalars
                        actor_name = add_overlay_mesh_func(
                            mesh_to_add,
                            scalars='speaker_face',
                            cmap=[body_color, exit_color],
                            opacity=1.0,
                            edge_color=edge_color,
                            line_width=line_width,
                            category='speakers',
                        )
                    else:
                        actor_name = add_overlay_mesh_func(
                            mesh_to_add,
                            color=body_color,
                            opacity=1.0,
                            edge_color=edge_color,
                            line_width=line_width,
                            category='speakers',
                        )
                    # Hole das Actor-Objekt aus dem Renderer
                    actor_obj = plotter.renderer.actors.get(actor_name) if actor_name else None
                    new_cache[key] = {'mesh': mesh_to_add, 'actor': actor_name, 'actor_obj': actor_obj}
                    if actor_name not in new_actor_names:
                        new_actor_names.append(actor_name)
                    new_cache[key]['signature'] = speaker_signature_from_mesh_func(mesh_to_add, exit_face_index)

    for key, info in old_cache.items():
        if key not in new_cache:
            remove_actor_func(info['actor'])

    speaker_actor_cache.clear()
    speaker_actor_cache.update(new_cache)
    category_actors['speakers'] = new_actor_names
    
    # Aktualisiere Edge-Color für hervorgehobene Lautsprecher
    update_speaker_highlights(settings, speaker_actor_cache, plotter)
    
    # Render-Update triggern, damit Änderungen sichtbar werden
    try:
        plotter.render()
    except Exception:  # noqa: BLE001
        pass

