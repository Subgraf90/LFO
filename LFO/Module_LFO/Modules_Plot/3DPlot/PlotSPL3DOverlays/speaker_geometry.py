"""Speaker-Geometrie-Berechnung für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import List, Optional, Tuple, Any, Callable

import numpy as np


def get_speaker_name(speaker_array, index: int):
    """Extrahiert den Speaker-Namen aus einem Speaker-Array.
    
    Args:
        speaker_array: Speaker-Array-Objekt
        index: Index des Speakers
        
    Returns:
        Speaker-Name oder None
    """
    if hasattr(speaker_array, 'source_polar_pattern'):
        pattern = speaker_array.source_polar_pattern
        try:
            if index < len(pattern):
                value = pattern[index]
                if isinstance(value, bytes):
                    return value.decode('utf-8', errors='ignore')
                return str(value)
        except Exception:  # noqa: BLE001
            pass

    if hasattr(speaker_array, 'source_type'):
        try:
            if index < len(speaker_array.source_type):
                return str(speaker_array.source_type[index])
        except Exception:  # noqa: BLE001
            pass

    return None


def safe_float(mapping, key: str, fallback: float | None = None) -> float:
    """Sichere Float-Extraktion aus einem Mapping.
    
    Args:
        mapping: Dictionary, Float oder String
        key: Key für Dictionary-Zugriff
        fallback: Fallback-Wert
        
    Returns:
        Float-Wert
    """
    if isinstance(mapping, (float, int)):
        return float(mapping)
    if isinstance(mapping, str):
        try:
            return float(mapping)
        except Exception:  # noqa: BLE001
            return float(fallback) if is_numeric(fallback) else 0.0
    if not isinstance(mapping, dict):
        return float(fallback) if is_numeric(fallback) else 0.0
    value = mapping.get(key, fallback if fallback is not None else 0.0)
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(fallback) if is_numeric(fallback) else 0.0


def is_numeric(value) -> bool:
    """Prüft ob ein Wert numerisch ist.
    
    Args:
        value: Wert zum Prüfen
        
    Returns:
        True wenn numerisch, sonst False
    """
    if isinstance(value, (float, int)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def array_value(values, index: int) -> float | None:
    """Extrahiert einen Wert aus einem Array.
    
    Args:
        values: Array oder Liste
        index: Index
        
    Returns:
        Float-Wert oder None
    """
    if values is None:
        return None
    try:
        if len(values) <= index:
            return None
        value = values[index]
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def float_sequence(values, target_length: int, default: float = 0.0) -> List[float]:
    """Erstellt eine Sequenz von Float-Werten.
    
    Args:
        values: Werte (kann einzelner Wert oder Sequenz sein)
        target_length: Ziel-Länge
        default: Standard-Wert
        
    Returns:
        Liste von Float-Werten
    """
    if target_length <= 0:
        return []
    if values is None:
        return [default] * target_length
    if isinstance(values, (float, int)):
        return [float(values)] * target_length
    result: List[float] = []
    try:
        iterable = list(values)
    except TypeError:
        iterable = [values]
    for idx in range(target_length):
        if idx < len(iterable):
            try:
                result.append(float(iterable[idx]))
            except Exception:  # noqa: BLE001
                result.append(default)
        else:
            result.append(default)
    return result


def sequence_length(values) -> int:
    """Berechnet die Länge einer Sequenz.
    
    Args:
        values: Werte
        
    Returns:
        Länge oder 0
    """
    if values is None:
        return 0
    try:
        return len(values)
    except TypeError:
        return 0


def resolve_cabinet_entries(cabinet_raw, configuration) -> List[dict]:
    """Löst Cabinet-Einträge auf.
    
    Args:
        cabinet_raw: Cabinet-Daten (dict, list, oder numpy array)
        configuration: Konfiguration ('flown', 'stack')
        
    Returns:
        Liste von Cabinet-Dictionaries
    """
    config_lower = (configuration or '').lower()
    entries: List[dict] = []

    def add_entry(entry):
        if not isinstance(entry, dict):
            return
        entry_config = str(entry.get('configuration', '')).lower() if entry.get('configuration') else ''
        if entry_config and config_lower and entry_config != config_lower:
            return
        entries.append(entry)

    if isinstance(cabinet_raw, dict):
        add_entry(cabinet_raw)
    elif isinstance(cabinet_raw, (list, tuple)):
        for item in cabinet_raw:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    add_entry(sub_item)
            else:
                add_entry(item)
    elif isinstance(cabinet_raw, np.ndarray):
        if cabinet_raw.ndim == 0:
            entries.extend(resolve_cabinet_entries(cabinet_raw.item(), configuration))
        else:
            entries.extend(resolve_cabinet_entries(cabinet_raw.tolist(), configuration))

    if not entries and isinstance(cabinet_raw, (list, tuple)):
        for item in cabinet_raw:
            if isinstance(item, dict):
                entries.append(item)

    return entries


def apply_flown_cabinet_shape(
    mesh: Any,
    *,
    front_height: float,
    back_height: float,
    top_reference: float,
    angle_point: str,
) -> Any:
    """Verformt ein würfelförmiges Gehäuse in eine Keilform anhand der Höhenmetadaten.
    
    Args:
        mesh: PyVista Mesh (Box)
        front_height: Vordere Höhe
        back_height: Hintere Höhe
        top_reference: Referenz-Z-Koordinate für die Oberseite
        angle_point: Winkelpunkt ('front', 'back', 'center', 'none')
        
    Returns:
        Verformtes Mesh
    """
    try:
        front_height = float(front_height)
        back_height = float(back_height)
        top_reference = float(top_reference)
    except Exception:  # noqa: BLE001
        return mesh

    if front_height <= 0.0 or back_height <= 0.0:
        return mesh
    if np.isclose(front_height, back_height, rtol=1e-4, atol=1e-3):
        return mesh
    if not hasattr(mesh, 'points'):
        return mesh

    angle_point_lower = (angle_point or 'front').strip().lower()
    if angle_point_lower not in {'front', 'back', 'center', 'none'}:
        angle_point_lower = 'front'

    height_diff = front_height - back_height
    delta = height_diff / 2.0

    if angle_point_lower == 'back':
        back_top = top_reference
        front_top = back_top + delta
    else:
        front_top = top_reference
        back_top = front_top - delta

    front_bottom = front_top - front_height
    back_bottom = back_top - back_height

    try:
        points = mesh.points.copy()
    except Exception:  # noqa: BLE001
        return mesh

    if points.size == 0:
        return mesh

    y_coords = points[:, 1]
    front_y = np.max(y_coords)
    back_y = np.min(y_coords)
    tol = max(abs(front_y - back_y), 1.0) * 1e-6 + 1e-9
    z_values = points[:, 2]
    z_mid = float((np.max(z_values) + np.min(z_values)) / 2.0)

    front_mask = np.isclose(y_coords, front_y, atol=tol)
    back_mask = np.isclose(y_coords, back_y, atol=tol)
    upper_mask = z_values >= z_mid
    lower_mask = ~upper_mask

    front_upper = front_mask & upper_mask
    front_lower = front_mask & lower_mask
    back_upper = back_mask & upper_mask
    back_lower = back_mask & lower_mask

    if np.any(front_upper):
        points[front_upper, 2] = front_top
    if np.any(front_lower):
        points[front_lower, 2] = front_bottom
    if np.any(back_upper):
        points[back_upper, 2] = back_top
    if np.any(back_lower):
        points[back_lower, 2] = back_bottom

    try:
        mesh.points = points
    except Exception:  # noqa: BLE001
        return mesh
    return mesh


def build_speaker_geometries(
    speaker_array: Any,
    index: int,
    x: float,
    y: float,
    z_reference: float,
    cabinet_lookup: dict,
    pv_module: Any,
    get_box_template_func: Callable,
    compute_box_face_indices_func: Callable,
    box_face_cache: dict,
    resolve_cabinet_entries_func: Callable,
    apply_flown_cabinet_shape_func: Callable,
    safe_float_func: Callable,
    array_value_func: Callable,
    float_sequence_func: Callable,
    sequence_length_func: Callable,
    get_speaker_name_func: Callable,
) -> List[Tuple[Any, Optional[int]]]:
    """Erstellt die Geometrien für einen Speaker.
    
    Vollständige Implementierung aus PlotSPL3DOverlays.py Zeilen 1577-1968.
    
    Args:
        speaker_array: Speaker-Array-Objekt
        index: Index des Speakers
        x, y, z_reference: Position
        cabinet_lookup: Cabinet-Lookup-Dictionary
        pv_module: PyVista Modul
        get_box_template_func: Funktion zum Holen von Box-Templates
        compute_box_face_indices_func: Funktion zum Berechnen von Face-Indices
        box_face_cache: Cache für Box-Face-Indices
        resolve_cabinet_entries_func: Funktion zum Auflösen von Cabinet-Einträgen
        apply_flown_cabinet_shape_func: Funktion zum Anwenden von Flown-Formen
        safe_float_func: Funktion zum sicheren Float-Extrahieren
        array_value_func: Funktion zum Extrahieren von Array-Werten
        float_sequence_func: Funktion zum Erstellen von Float-Sequenzen
        sequence_length_func: Funktion zum Berechnen der Sequenz-Länge
        get_speaker_name_func: Funktion zum Extrahieren von Speaker-Namen
        
    Returns:
        Liste von (mesh, exit_face_index) Tuples
    """
    speaker_name = get_speaker_name_func(speaker_array, index)
    configuration = getattr(speaker_array, 'configuration', '')

    cabinet_raw = cabinet_lookup.get(speaker_name)
    if cabinet_raw is None and isinstance(speaker_name, str):
        cabinet_raw = cabinet_lookup.get(speaker_name.lower())

    cabinets = resolve_cabinet_entries_func(cabinet_raw, configuration)
    if not cabinets:
        return []

    geoms: List[Tuple[Any, Optional[int]]] = []

    config_lower = (configuration or '').lower()
    if config_lower == 'flown':
        reference_array = getattr(speaker_array, 'source_position_z_flown', None)
    else:
        # Für Stack: source_position_z_stack ist die ursprüngliche Speaker Z-Position (ohne Array-Offset)
        # z_reference ist die Plot-Position (Array Z-Position + Speaker Z-Position)
        reference_array = getattr(speaker_array, 'source_position_z_stack', None)
    source_top_array = getattr(speaker_array, 'source_position_z_flown', None)
    speaker_count = 0
    for candidate in (reference_array, source_top_array):
        if candidate is not None:
            try:
                speaker_count = max(speaker_count, len(candidate))
            except Exception:  # noqa: BLE001
                pass

    if config_lower == 'flown' and cabinets:
        if len(cabinets) == 1 and speaker_count > 1:
            template = cabinets[0]
            # Erstelle eine Kopie pro Lautsprecher, damit Offsets unabhängig bleiben
            cabinets = [dict(template) for _ in range(speaker_count)]

    azimuth = array_value_func(getattr(speaker_array, 'source_azimuth', None), index)
    azimuth_rad = np.deg2rad(azimuth) if azimuth is not None else 0.0

    processed_entries: List[dict] = []
    stack_state: dict[tuple, dict[str, float]] = {}
    on_top_width: float | None = None
    flown_height_acc: float = 0.0
    flown_groups: List[List[int]] = []
    current_flown_group: List[int] = []
    flown_entry_index = 0

    for idx_cabinet, cabinet in enumerate(cabinets):
        entry_config = str(cabinet.get('configuration', configuration)).lower() if isinstance(cabinet, dict) else config_lower
        width = safe_float_func(cabinet, 'width')
        depth = safe_float_func(cabinet, 'depth')
        front_height = safe_float_func(cabinet, 'front_height')
        back_height = safe_float_func(cabinet, 'back_height', fallback=front_height)
        cardio = bool(cabinet.get('cardio', False)) if isinstance(cabinet, dict) else False
        angle_point_raw = ''
        if isinstance(cabinet, dict):
            angle_point_raw = str(cabinet.get('angle_point', '')).strip().lower()
        if angle_point_raw not in {'front', 'back', 'center', 'none'}:
            angle_point_raw = 'front'
        stack_layout = str(cabinet.get('stack_layout', 'beside')).strip().lower() if isinstance(cabinet, dict) else 'beside'

        height = max(front_height, back_height)
        if width <= 0 or depth <= 0 or height <= 0:
            continue

        x_offset = safe_float_func(cabinet, 'x_offset', fallback=(-width / 2.0))
        y_offset = safe_float_func(cabinet, 'y_offset', fallback=0.0)
        z_offset = safe_float_func(cabinet, 'z_offset', fallback=0.0)

        stack_group_key = None
        if entry_config == 'stack':
            if stack_layout == 'on top':
                if on_top_width is None:
                    on_top_width = width
                x_offset = -(on_top_width / 2.0)
                y_offset = 0.0
                stack_group_key = ('on_top', round(x_offset, 4), 0.0)
                state = stack_state.setdefault(
                    stack_group_key,
                    {'height': 0.0, 'base_x': x_offset, 'base_y': y_offset},
                )
                stack_offset = state['height']
                state['height'] += height
                x_offset = state['base_x']
                y_offset = state['base_y']
            else:
                stack_group_key = ('beside', round(x_offset, 4), round(y_offset, 4))
                stack_offset = 0.0
        else:
            if entry_config == 'flown':
                stack_offset = flown_height_acc
                flown_height_acc += height
            else:
                stack_offset = 0.0

        if entry_config != 'flown':
            if current_flown_group:
                flown_groups.append(current_flown_group)
                current_flown_group = []

        entry_idx = len(processed_entries)
        if entry_config == 'flown':
            source_idx = flown_entry_index
            flown_entry_index += 1
        else:
            source_idx = len(processed_entries)

        if angle_point_raw == 'back':
            effective_height = back_height
        elif angle_point_raw == 'center':
            effective_height = (front_height + back_height) / 2.0
        elif angle_point_raw == 'none':
            effective_height = max(front_height, back_height)
        else:
            effective_height = front_height
        if effective_height <= 0:
            effective_height = height

        processed_entries.append({
            'width': width,
            'depth': depth,
            'front_height': front_height,
            'back_height': back_height,
            'height': height,
            'cardio': cardio,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'z_offset': z_offset,
            'stack_offset': stack_offset,
            'configuration': entry_config,
            'stack_layout': stack_layout,
            'stack_group_key': stack_group_key,
            'source_index': source_idx,
            'effective_height': effective_height,
            'angle_point': angle_point_raw,
        })

        if entry_config == 'flown':
            current_flown_group.append(entry_idx)

    if current_flown_group:
        flown_groups.append(current_flown_group)

    pos_y_raw = getattr(
        speaker_array,
        'source_position_calc_y',
        getattr(speaker_array, 'source_position_y', None),
    )
    speaker_count = max(
        speaker_count,
        sequence_length_func(pos_y_raw),
    )
    site_values = np.asarray(
        float_sequence_func(getattr(speaker_array, 'source_site', None), speaker_count, 0.0),
        dtype=float,
    )
    position_y_values = np.asarray(
        float_sequence_func(
            getattr(
                speaker_array,
                'source_position_calc_y',
                getattr(speaker_array, 'source_position_y', None),
            ),
            speaker_count,
            0.0,
        ),
        dtype=float,
    )
    calc_z_values = np.asarray(
        float_sequence_func(
            getattr(speaker_array, 'source_position_calc_z', None),
            speaker_count,
            float(z_reference),
        ),
        dtype=float,
    )
    angle_values = np.asarray(
        float_sequence_func(
            getattr(speaker_array, 'source_angle', None),
            speaker_count,
            0.0,
        ),
        dtype=float,
    )
    base_site_angle = float(site_values[0]) if site_values.size > 0 else 0.0
    total_angle_values = np.full(speaker_count, base_site_angle, dtype=float)
    if speaker_count > 1:
        cumulative = np.cumsum(angle_values[1:])
        total_angle_values[1:] = base_site_angle - cumulative

    for entry_idx, entry in enumerate(processed_entries):
        width = entry['width']
        depth = entry['depth']
        front_height = entry['front_height']
        back_height = entry['back_height']
        height = entry['height']
        cardio = entry['cardio']
        x_offset = entry['x_offset']
        y_offset = entry['y_offset']
        z_offset = entry['z_offset']
        stack_offset = entry['stack_offset']
        entry_config = entry['configuration']
        stack_layout = entry['stack_layout']

        if entry_config == 'stack' and stack_layout == 'on top':
            if on_top_width is not None:
                x_offset = -(on_top_width / 2.0)
                y_offset = 0.0
                width = on_top_width
        source_idx = entry.get('source_index', entry_idx)
        box_key: Optional[tuple[float, float, float]] = None

        if entry_config == 'flown':
            effective_height = entry.get('effective_height', height)
            if effective_height <= 0:
                effective_height = height
            angle_deg = total_angle_values[source_idx] if source_idx < len(total_angle_values) else base_site_angle
            angle_point_debug = entry.get('angle_point', 'front')
            segment_info = getattr(speaker_array, '_flown_segment_geometry', [])
            segment = segment_info[source_idx] if source_idx < len(segment_info) else {}
            top_z_info = float(segment.get('top_z', calc_z_values[source_idx] + effective_height / 2.0 if source_idx < len(calc_z_values) else z_reference))
            pivot_y_info = float(segment.get('pivot_y', position_y_values[source_idx] if source_idx < len(position_y_values) else 0.0))
            pivot_z_info = float(segment.get('pivot_z', top_z_info))
            center_y_info = float(segment.get('center_y', position_y_values[source_idx] if source_idx < len(position_y_values) else 0.0))
            center_z_info = float(segment.get('center_z', calc_z_values[source_idx] if source_idx < len(calc_z_values) else z_reference))
            front_height_segment = safe_float_func(segment, 'front_height', front_height)
            back_height_segment = safe_float_func(segment, 'back_height', back_height)
            effective_height_segment = safe_float_func(segment, 'effective_height', effective_height)
            if effective_height_segment > 0:
                effective_height = effective_height_segment
            front_height_effective = front_height_segment if front_height_segment > 0 else front_height
            back_height_effective = back_height_segment if back_height_segment > 0 else back_height
            back_height_symmetrized = 0.5 * (front_height_effective + back_height_effective)
            if back_height_symmetrized <= 0:
                back_height_symmetrized = back_height_effective
            angle_point_lower = str(segment.get('angle_point', angle_point_debug)).strip().lower()
            if angle_point_lower not in {'front', 'back', 'center', 'none'}:
                angle_point_lower = str(angle_point_debug).strip().lower()
            x_min = x + x_offset
            x_max = x_min + width

            pivot_y_world = pivot_y_info + y_offset
            pivot_z_world = pivot_z_info - z_offset
            center_y_world = center_y_info + y_offset
            center_z_world = center_z_info - z_offset
            top_z_world = top_z_info - z_offset

            if angle_point_lower == 'back':
                y_min = pivot_y_world
                y_max = pivot_y_world + depth
            elif angle_point_lower == 'front':
                y_max = pivot_y_world
                y_min = pivot_y_world - depth
            else:
                y_min = pivot_y_world - depth / 2.0
                y_max = pivot_y_world + depth / 2.0

            z_max = top_z_world
            z_min = top_z_world - effective_height

            box_key = (float(width), float(depth), float(effective_height))
            template = get_box_template_func(*box_key)
            body = template.copy(deep=True)
            if angle_point_lower == 'back':
                translation_y = pivot_y_world + depth
            elif angle_point_lower == 'front':
                translation_y = pivot_y_world
            else:
                translation_y = pivot_y_world + depth / 2.0
            body.translate((x_min, translation_y, z_min), inplace=True)
            body = apply_flown_cabinet_shape_func(
                body,
                front_height=front_height_effective,
                back_height=back_height_effective,
                top_reference=top_z_world,
                angle_point=angle_point_lower,
            )
            if abs(angle_deg) > 1e-6:
                body.rotate_x(angle_deg, point=((x_min + x_max) / 2.0, pivot_y_world, pivot_z_world), inplace=True)
        else:
            x_min = x + x_offset
            x_max = x_min + width
            front_y_world = y + y_offset
            y_max = front_y_world
            y_min = front_y_world - depth
            # Für Stack: z_reference ist die Plot-Position (Array Z-Position + Speaker Z-Position)
            # reference_array (source_position_z_stack) wird nur als Fallback verwendet
            if config_lower != 'flown' and z_reference is not None:
                # Für Stack-Systeme: z_reference ist die Plot-Position (bereits Array Z-Position + Speaker Z-Position)
                reference_value = z_reference
            else:
                reference_value = array_value_func(reference_array, entry.get('source_index', index))
                if reference_value is None:
                    reference_value = z_reference
            base_world = float(reference_value) + z_offset + stack_offset
            z_min = base_world
            z_max = base_world + height
            box_key = (float(width), float(depth), float(height))
            template = get_box_template_func(*box_key)
            body = template.copy(deep=True)
            body.translate((x_min, front_y_world, z_min), inplace=True)

        exit_at_front = not cardio
        exit_face_index: Optional[int] = None
        if box_key is not None:
            face_indices = box_face_cache.get(box_key)
            if face_indices is None:
                face_indices = compute_box_face_indices_func(template)
                box_face_cache[box_key] = face_indices
            if face_indices is not None:
                front_idx, back_idx = face_indices
                if exit_at_front:
                    exit_face_index = front_idx
                else:
                    exit_face_index = back_idx

        geoms.append((body, exit_face_index))

    rotation_groups: List[List[int]] = []
    stack_indices = [idx for idx, entry in enumerate(processed_entries) if entry['configuration'] == 'stack']
    if stack_indices:
        rotation_groups.append(stack_indices)
    for indices in flown_groups:
        if indices:
            rotation_groups.append(indices)

    if rotation_groups and abs(azimuth_rad) > 1e-6:
        angle_deg = float(np.rad2deg(-azimuth_rad))
        for group_indices in rotation_groups:
            group_entries = []
            for idx in group_indices:
                if idx < 0 or idx >= len(geoms):
                    continue
                body_mesh, exit_face_index = geoms[idx]
                group_entries.append((idx, processed_entries[idx], body_mesh, exit_face_index))
            if not group_entries:
                continue

            x_mins = [mesh.bounds[0] for _, _, mesh, _ in group_entries]
            x_maxs = [mesh.bounds[1] for _, _, mesh, _ in group_entries]
            y_mins = [mesh.bounds[2] for _, _, mesh, _ in group_entries]
            y_maxs = [mesh.bounds[3] for _, _, mesh, _ in group_entries]
            center_x = 0.0
            if x_mins and x_maxs:
                center_x = 0.5 * (min(x_mins) + max(x_maxs))
            center_y = 0.0
            if y_maxs:
                center_y = max(y_maxs)

            first_config = group_entries[0][1].get('configuration', '')
            target_front_y: Optional[float] = None
            if first_config == 'flown':
                top_entry = max(group_entries, key=lambda item: item[2].bounds[5])
                top_mesh = top_entry[2]
                top_bounds = top_mesh.bounds
                top_z = top_bounds[5]
                rotation_center = (center_x, center_y, top_z)
                target_front_y = None
            else:
                z_midpoints = [(mesh.bounds[4] + mesh.bounds[5]) / 2.0 for _, _, mesh, _ in group_entries]
                z_center_group = np.mean(z_midpoints) if z_midpoints else z_reference
                rotation_center = (center_x, center_y, z_center_group)
                target_front_y = None

            for idx, entry, mesh, exit_face_index in group_entries:
                mesh.rotate_z(angle_deg, point=rotation_center, inplace=True)
                if target_front_y is not None:
                    current_front_y = mesh.bounds[3]
                    delta_y = target_front_y - current_front_y
                    if abs(delta_y) > 1e-6:
                        mesh.translate((0.0, delta_y, 0.0), inplace=True)
                geoms[idx] = (mesh, exit_face_index)

    return geoms
