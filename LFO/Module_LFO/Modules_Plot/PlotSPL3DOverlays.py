from __future__ import annotations

from typing import List, Optional, Tuple, Any

import numpy as np
from PyQt5 import QtWidgets, QtCore

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    DEBUG_SURFACE_GEOMETRY,
    SurfaceDefinition,
)


class SPL3DOverlayRenderer:
    """Verwaltet das Zeichnen und Zurücksetzen der Overlay-Elemente für den SPL-Plot."""
    """source_position_z_flown: Oberkanten der Flown-Cabinets (Referenz fürs Gehäuse-Plotting)"""
    """source_position_z_stack: Unterkanten der Stack-Cabinets (Referenz fürs Gehäuse-Plotting)"""
    """source_position_calc_z: berechnete Mittelpunkte der akustischen Zentren (für SPL-Berechnungen)"""
    """source_position_z: vom Benutzer eingegebene Z-Offsets aus der UI (Rohwerte)"""

    def __init__(self, plotter: Any, pv_module: Any):
        self.plotter = plotter
        self.pv = pv_module
        self.overlay_actor_names: List[str] = []
        self.overlay_counter = 0
        self._category_actors: dict[str, List[str]] = {}
        self._box_template_cache: dict[tuple[float, float, float], Any] = {}
        self._box_face_cache: dict[tuple[float, float, float], Tuple[Optional[int], Optional[int]]] = {}
        # Kleinster Z-Offset, um Z-Fighting mit Boden/Surface zu vermeiden
        # (insbesondere bei Draufsicht sollen Linien klar sichtbar bleiben).
        self._planar_z_offset = 0.02
        self._speaker_actor_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
        self._speaker_geometry_cache: dict[str, List[Tuple[Any, Optional[int]]]] = {}  # Cache für transformierte Geometrien
        self._geometry_cache_max_size = 100  # Maximale Anzahl gecachter Geometrien
        self._last_axis_state: Optional[tuple] = None
        self._last_impulse_state: Optional[tuple] = None
        self._last_surfaces_state: Optional[tuple] = None


    def clear(self) -> None:
        for name in self.overlay_actor_names:
            try:
                self.plotter.remove_actor(name)
            except KeyError:
                pass
        self.overlay_actor_names.clear()
        self.overlay_counter = 0
        self._category_actors.clear()
        self._speaker_actor_cache.clear()
        self._speaker_geometry_cache.clear()
        self._last_axis_state = None
        self._last_impulse_state = None

    def clear_category(self, category: str) -> None:
        """Entfernt alle Actor einer Kategorie, ohne andere Overlays anzutasten."""
        actor_names = self._category_actors.pop(category, [])
        for name in actor_names:
            try:
                self.plotter.remove_actor(name)
            except KeyError:
                pass
            else:
                if name in self.overlay_actor_names:
                    self.overlay_actor_names.remove(name)
        if category == 'speakers':
            keys_to_remove = [key for key, info in self._speaker_actor_cache.items() if info['actor'] in actor_names]
            for key in keys_to_remove:
                self._speaker_actor_cache.pop(key, None)
            # Optional: Geometrie-Cache nur teilweise löschen statt komplett
            # Für jetzt: löschen wir den gesamten Geometrie-Cache bei Speaker-Änderungen
            # self._speaker_geometry_cache.clear()
        elif category == 'axis':
            self._last_axis_state = None
        elif category == 'impulse':
            self._last_impulse_state = None
        elif category == 'surfaces':
            self._last_surfaces_state = None

    # ------------------------------------------------------------------
    # Öffentliche API zum Zeichnen der Overlays
    # ------------------------------------------------------------------
    def draw_axis_lines(self, settings) -> None:
        state = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
        )
        existing_names = self._category_actors.get('axis', [])
        if self._last_axis_state == state and existing_names:
            return

        self.clear_category('axis')
        x_axis, y_axis, length, width = state

        z_offset = self._planar_z_offset
        line_x = self.pv.Line((x_axis, -length / 2, z_offset), (x_axis, length / 2, z_offset))
        line_y = self.pv.Line((-width / 2, y_axis, z_offset), (width / 2, y_axis, z_offset))

        dashed_pattern = 0xAAAA
        self._add_overlay_mesh(line_x, color='black', line_width=1.2, line_pattern=dashed_pattern, line_repeat=2, category='axis')
        self._add_overlay_mesh(line_y, color='black', line_width=1.2, line_pattern=dashed_pattern, line_repeat=2, category='axis')
        self._last_axis_state = state

    def draw_surfaces(self, settings) -> None:
        """Zeichnet alle aktivierten, nicht versteckten Surfaces als Polygone im 3D-Plot."""
        # Erstelle Signatur für Änderungsdetektion
        surface_definitions = getattr(settings, 'surface_definitions', {})
        if not isinstance(surface_definitions, dict):
            self.clear_category('surfaces')
            self._last_surfaces_state = None
            return

        # Hole aktives Surface für Signatur und Markierung
        active_surface_id = getattr(settings, 'active_surface_id', None)

        surfaces_signature: List[tuple] = []
        for surface_id in sorted(surface_definitions.keys()):
            surface_def = surface_definitions[surface_id]
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
            else:
                enabled = bool(surface_def.get('enabled', False))
                hidden = bool(surface_def.get('hidden', False))
                points = surface_def.get('points', [])
            
            # Erstelle Signatur aus enabled, hidden, Punkten und aktiver Surface-ID
            points_tuple = []
            for point in points:
                try:
                    x = float(point.get('x', 0.0))
                    y = float(point.get('y', 0.0))
                    z = float(point.get('z', 0.0))
                    points_tuple.append((x, y, z))
                except (ValueError, TypeError, AttributeError):
                    continue
            
            surfaces_signature.append((
                str(surface_id),
                enabled,
                hidden,
                tuple(points_tuple)
            ))
        
        # Füge active_surface_id zur Signatur hinzu, damit Plot aktualisiert wird bei Auswahländerung
        signature_tuple = (tuple(surfaces_signature), active_surface_id)
        
        # Prüfe ob sich etwas geändert hat
        if self._last_surfaces_state == signature_tuple:
            return
        
        # Lösche alte Surfaces
        self.clear_category('surfaces')
        self._last_surfaces_state = signature_tuple
        
        # Zeichne alle nicht versteckten Surfaces (enabled und disabled)
        z_offset = self._planar_z_offset * 2  # Etwas höher als andere Overlays
        
        for surface_id, surface_def in surface_definitions.items():
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
                color_value = getattr(surface_def, 'color', '#888888')
            else:
                enabled = surface_def.get('enabled', False)
                hidden = surface_def.get('hidden', False)
                points = surface_def.get('points', [])
                color_value = surface_def.get('color', '#888888')
            
            # Überspringe nur versteckte Surfaces
            if hidden:
                continue
            
            if len(points) < 3:
                continue
            
            # Prüfe ob dies das aktive Surface ist
            is_active = (surface_id == active_surface_id)
            
            # Hole Farbe aus Surface-Definition oder verwende Standard
            # Aktives Surface wird rot markiert
            if is_active:
                color = '#FF0000'  # Rot für aktives Surface
            else:
                color = color_value if isinstance(color_value, str) else '#888888'
            
            # Konvertiere Punkte zu numpy-Array
            try:
                point_coords = []
                for point in points:
                    x = float(point.get('x', 0.0))
                    y = float(point.get('y', 0.0))
                    z = float(point.get('z', 0.0)) + z_offset
                    point_coords.append([x, y, z])
                
                if len(point_coords) < 3:
                    continue
                
                # Für gefüllte Polygone (z. B. disabled Surfaces) ohne doppelten Schluss-Punkt
                polygon_coords = point_coords.copy()

                # Erstelle geschlossenes Polygon (füge ersten Punkt am Ende hinzu)
                closed_coords = point_coords + [point_coords[0]]
                
                # Erstelle PolyLine aus PyVista
                polyline = self.pv.PolyData(closed_coords)
                polyline.lines = [len(closed_coords)] + list(range(len(closed_coords)))
                # Entferne Vertex-Zellen, damit keine Punkte an den Eckpunkten gerendert werden
                try:
                    polyline.verts = np.empty(0, dtype=np.int64)
                except Exception:
                    try:
                        polyline.verts = []
                    except Exception:
                        pass
                
                # 🎯 Unterscheide zwischen enabled/disabled und aktivem/inaktivem Surface
                if DEBUG_SURFACE_GEOMETRY:
                    y_vals = [p[1] for p in point_coords]
                    z_vals = [p[2] for p in point_coords]
                    print(
                        "[SurfaceOverlay]",
                        surface_id,
                        "enabled" if enabled else "disabled",
                        f"y-range=({min(y_vals):.2f}, {max(y_vals):.2f})",
                        f"z-range=({min(z_vals):.2f}, {max(z_vals):.2f})",
                    )

                if enabled:
                    if is_active:
                        # Aktives enabled Surface: Rote Umrandung, etwas hervorgehoben
                        polyline_render = polyline
                        try:
                            polyline_render = polyline.tube(radius=0.02, n_sides=24, capping=True)
                        except Exception:
                            pass
                        self._add_overlay_mesh(
                            polyline_render,
                            color='#FF0000',  # Immer Rot für aktives Surface
                            line_width=1.2,
                            opacity=0.95,
                            category='surfaces',
                            show_vertices=False,
                            render_lines_as_tubes=None,  # Bereits Tube-Mesh, daher nicht nötig
                        )
                    else:
                        # Inaktives enabled Surface: sehr schlanke Linienführung
                        polyline_render = polyline
                        try:
                            polyline_render = polyline.tube(radius=0.015, n_sides=24, capping=True)
                        except Exception:
                            pass
                        self._add_overlay_mesh(
                            polyline_render,
                            color=color,
                            line_width=1.2,
                            opacity=0.75,
                            category='surfaces',
                            show_vertices=False,
                            render_lines_as_tubes=None,  # Bereits Tube-Mesh, daher nicht nötig
                        )
                else:
                    # Disabled Surface: sichtbar mit hellgrauer Farbe
                    dashed_pattern = 0xAAAA  # Gestricheltes Muster
                    disabled_color = '#dcdcdc'

                    # Fülle disabled Surface mit hellem Grau (nur wenn genügend Punkte vorhanden)
                    try:
                        poly_points = np.asarray(polygon_coords, dtype=float)
                        if poly_points.shape[0] >= 3:
                            faces = np.concatenate(
                                ([poly_points.shape[0]], np.arange(poly_points.shape[0], dtype=np.int64))
                            )
                            polygon = self.pv.PolyData(poly_points, faces)
                            polygon = polygon.triangulate()
                            self._add_overlay_mesh(
                                polygon,
                                color=disabled_color,
                                opacity=0.35,
                                category='surfaces',
                                edge_color=disabled_color,
                                show_edges=False,
                            )
                    except Exception:
                        pass

                    if is_active:
                        # Aktives disabled Surface: Rote gestrichelte Umrandung
                        # Für gestrichelte Linien müssen wir Polylines verwenden (Tube-Meshes unterstützen kein line_pattern)
                        self._add_overlay_mesh(
                            polyline,
                            color='#FF0000',
                            line_width=1.2,
                            opacity=0.60,
                            line_pattern=dashed_pattern,
                            line_repeat=2,
                            category='surfaces',
                            show_vertices=False,
                            render_lines_as_tubes=True,  # Polyline mit Tube-Rendering für gestrichelte Linien
                        )
                    else:
                        # Inaktives disabled Surface: Standard gestrichelt, dünn
                        # Für gestrichelte Linien müssen wir Polylines verwenden (Tube-Meshes unterstützen kein line_pattern)
                        self._add_overlay_mesh(
                            polyline,
                            color=disabled_color,
                            line_width=1.2,
                            opacity=0.6,
                            line_pattern=dashed_pattern,
                            line_repeat=2,
                            category='surfaces',
                            show_vertices=False,
                            render_lines_as_tubes=True,  # Polyline mit Tube-Rendering für gestrichelte Linien
                        )
            except (ValueError, TypeError, AttributeError, Exception):
                continue

    def draw_speakers(self, settings, container, cabinet_lookup: dict) -> None:
        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        if not isinstance(speaker_arrays, dict):
            self.clear_category('speakers')
            return

        body_color = '#b0b0b0'
        exit_color = '#4d4d4d'

        old_cache = self._speaker_actor_cache
        new_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
        new_actor_names: List[str] = []

        for array_name, speaker_array in speaker_arrays.items():
            if getattr(speaker_array, 'hide', False):
                continue

            configuration = getattr(speaker_array, 'configuration', '')
            config_lower = configuration.lower() if isinstance(configuration, str) else ''

            xs = self._to_float_array(getattr(speaker_array, 'source_position_x', None))
            if config_lower == 'flown':
                ys = self._to_float_array(
                    getattr(
                        speaker_array,
                        'source_position_calc_y',
                        getattr(speaker_array, 'source_position_y', None),
                    )
                )
                zs = self._to_float_array(getattr(speaker_array, 'source_position_z_flown', None))
            else:
                ys = self._to_float_array(getattr(speaker_array, 'source_position_y', None))
                zs = self._to_float_array(getattr(speaker_array, 'source_position_z_stack', None))

            min_len = min(xs.size, ys.size, zs.size)
            if min_len == 0:
                continue

            valid_mask = np.isfinite(xs[:min_len]) & np.isfinite(ys[:min_len]) & np.isfinite(zs[:min_len])
            valid_indices = np.nonzero(valid_mask)[0]
            array_identifier = str(getattr(speaker_array, 'name', array_name))

            for idx in valid_indices:
                x = xs[idx]
                y = ys[idx]
                z = zs[idx]

                geometries = self._build_speaker_geometries(
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
                    sphere = self.pv.Sphere(radius=0.6, center=(float(x), float(y), float(z)), theta_resolution=32, phi_resolution=32)
                    key = (array_identifier, int(idx), 'fallback')
                    existing = old_cache.get(key)
                    if existing:
                        signature = existing.get('signature')
                        current_signature = self._speaker_signature_from_mesh(sphere, None)
                        if signature == current_signature:
                            new_cache[key] = existing
                            if existing['actor'] not in new_actor_names:
                                new_actor_names.append(existing['actor'])
                            continue
                        existing_mesh = existing['mesh']
                        existing_mesh.deep_copy(sphere)
                        self._update_speaker_actor(existing['actor'], existing_mesh, None, body_color, exit_color)
                        existing['signature'] = current_signature
                        new_cache[key] = existing
                        if existing['actor'] not in new_actor_names:
                            new_actor_names.append(existing['actor'])
                    else:
                        mesh_to_add = sphere.copy(deep=True)
                        actor_name = self._add_overlay_mesh(
                            mesh_to_add,
                            color=body_color,
                            opacity=1.0,
                            edge_color='black',
                            category='speakers',
                        )
                        new_cache[key] = {
                            'mesh': mesh_to_add,
                            'actor': actor_name,
                            'signature': self._speaker_signature_from_mesh(mesh_to_add, None),
                        }
                        if actor_name not in new_actor_names:
                            new_actor_names.append(actor_name)
                else:
                    for geom_idx, (body_mesh, exit_face_index) in enumerate(geometries):
                        key = (array_identifier, int(idx), geom_idx)
                        existing = old_cache.get(key)
                        if existing:
                            signature = existing.get('signature')
                            current_signature = self._speaker_signature_from_mesh(body_mesh, exit_face_index)
                            if signature == current_signature:
                                new_cache[key] = existing
                                if existing['actor'] not in new_actor_names:
                                    new_actor_names.append(existing['actor'])
                                continue
                            existing_mesh = existing['mesh']
                            existing_mesh.deep_copy(body_mesh)
                            self._update_speaker_actor(existing['actor'], existing_mesh, exit_face_index, body_color, exit_color)
                            existing['signature'] = current_signature
                            new_cache[key] = existing
                            if existing['actor'] not in new_actor_names:
                                new_actor_names.append(existing['actor'])
                            continue

                        mesh_to_add = body_mesh.copy(deep=True)
                        # Prüfe ob Mesh bereits Scalars hat (merged mesh mit exit_face_index = -1)
                        has_scalars = 'speaker_face' in mesh_to_add.cell_data
                        
                        if exit_face_index == -1 or has_scalars:
                            # Mesh hat bereits Scalars (merged mesh) - nutze diese
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                scalars='speaker_face',
                                cmap=[body_color, exit_color],
                                opacity=1.0,
                                edge_color='black',
                                line_width=1.5,
                                category='speakers',
                            )
                        elif exit_face_index is not None and mesh_to_add.n_cells > 0:
                            # Einzelnes Mesh - erstelle Scalars für eine Exit-Face
                            scalars = np.zeros(mesh_to_add.n_cells, dtype=int)
                            scalars[int(exit_face_index)] = 1
                            mesh_to_add.cell_data['speaker_face'] = scalars
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                scalars='speaker_face',
                                cmap=[body_color, exit_color],
                                opacity=1.0,
                                edge_color='black',
                                line_width=1.5,
                                category='speakers',
                            )
                        else:
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                color=body_color,
                                opacity=1.0,
                                edge_color='black',
                                line_width=1.5,
                                category='speakers',
                            )
                        new_cache[key] = {'mesh': mesh_to_add, 'actor': actor_name}
                        if actor_name not in new_actor_names:
                            new_actor_names.append(actor_name)
                        new_cache[key]['signature'] = self._speaker_signature_from_mesh(mesh_to_add, exit_face_index)

        for key, info in old_cache.items():
            if key not in new_cache:
                self._remove_actor(info['actor'])

        self._speaker_actor_cache = new_cache
        self._category_actors['speakers'] = new_actor_names

    def draw_impulse_points(self, settings) -> None:
        current_state = self._compute_impulse_state(settings)
        existing_names = self._category_actors.get('impulse', [])
        if self._last_impulse_state == current_state and existing_names:
            return

        self.clear_category('impulse')
        _, impulse_points = current_state
        if not impulse_points:
            self._last_impulse_state = current_state
            return

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
            cone = self.pv.Cone(
                center=(float(x), float(y), center_z),
                direction=(0.0, 0.0, 1.0),
                height=height,
                radius=radius,
            )
            self._add_overlay_mesh(cone, color='red', opacity=0.85, category='impulse')
        self._last_impulse_state = current_state

    def _compute_impulse_state(self, settings) -> tuple:
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

    # ------------------------------------------------------------------
    # Hilfsfunktionen für Lautsprecher
    # ------------------------------------------------------------------
    def build_cabinet_lookup(self, container) -> dict:
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


    def _create_geometry_cache_key(self, speaker_array, index: int, speaker_name: str, configuration: str, cabinet_raw) -> str:
        """Erstellt einen Cache-Key für die Geometrie basierend auf relevanten Parametern."""
        import hashlib
        
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
        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), index)
        params.append(f"azimuth:{azimuth:.4f}" if azimuth is not None else "azimuth:0")
        
        angle_val = self._array_value(getattr(speaker_array, 'source_angle', None), index)
        params.append(f"angle:{angle_val:.4f}" if angle_val is not None else "angle:0")
        
        site_val = self._array_value(getattr(speaker_array, 'source_site', None), index)
        params.append(f"site:{site_val:.4f}" if site_val is not None else "site:0")
        
        # Anzahl der Quellen (wichtig für flown arrays)
        params.append(f"count:{getattr(speaker_array, 'number_of_sources', 0)}")
        
        # Erstelle Hash
        param_string = '|'.join(str(p) for p in params)
        return hashlib.md5(param_string.encode('utf-8')).hexdigest()

    def _build_speaker_geometries(
        self,
        speaker_array,
        index: int,
        x: float,
        y: float,
        z_reference: float,
        cabinet_lookup: dict,
        container,
        ) -> List[Tuple[Any, Optional[int]]]:
        del container
        speaker_name = self._get_speaker_name(speaker_array, index)
        configuration = getattr(speaker_array, 'configuration', '')

        cabinet_raw = cabinet_lookup.get(speaker_name)
        if cabinet_raw is None and isinstance(speaker_name, str):
            cabinet_raw = cabinet_lookup.get(speaker_name.lower())

        # Cache deaktiviert - zu komplex für flown arrays mit Rotationen
        # cache_key = self._create_geometry_cache_key(speaker_array, index, speaker_name, configuration, cabinet_raw)
        
        cabinets = self._resolve_cabinet_entries(cabinet_raw, configuration)
        if not cabinets:
            return []

        geoms: List[Tuple[Any, Optional[int]]] = []

        config_lower = (configuration or '').lower()
        if config_lower == 'flown':
            reference_array = getattr(speaker_array, 'source_position_z_flown', None)
        else:
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

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), index)
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
            width = self._safe_float(cabinet, 'width')
            depth = self._safe_float(cabinet, 'depth')
            front_height = self._safe_float(cabinet, 'front_height')
            back_height = self._safe_float(cabinet, 'back_height', fallback=front_height)
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

            x_offset = self._safe_float(cabinet, 'x_offset', fallback=(-width / 2.0))
            y_offset = self._safe_float(cabinet, 'y_offset', fallback=0.0)
            z_offset = self._safe_float(cabinet, 'z_offset', fallback=0.0)

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
            self._sequence_length(pos_y_raw),
        )
        site_values = np.asarray(
            self._float_sequence(getattr(speaker_array, 'source_site', None), speaker_count, 0.0),
            dtype=float,
        )
        position_y_values = np.asarray(
            self._float_sequence(
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
            self._float_sequence(
                getattr(speaker_array, 'source_position_calc_z', None),
                speaker_count,
                float(z_reference),
            ),
            dtype=float,
        )
        angle_values = np.asarray(
            self._float_sequence(
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
                front_height_segment = self._safe_float(segment, 'front_height', front_height)
                back_height_segment = self._safe_float(segment, 'back_height', back_height)
                effective_height_segment = self._safe_float(segment, 'effective_height', effective_height)
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
                template = self._get_box_template(*box_key)
                body = template.copy(deep=True)
                if angle_point_lower == 'back':
                    translation_y = pivot_y_world + depth
                elif angle_point_lower == 'front':
                    translation_y = pivot_y_world
                else:
                    translation_y = pivot_y_world + depth / 2.0
                body.translate((x_min, translation_y, z_min), inplace=True)
                body = self._apply_flown_cabinet_shape(
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
                reference_value = self._array_value(reference_array, entry.get('source_index', index))
                if reference_value is None:
                    reference_value = z_reference
                base_world = float(reference_value) + z_offset + stack_offset
                z_min = base_world
                z_max = base_world + height
                box_key = (float(width), float(depth), float(height))
                template = self._get_box_template(*box_key)
                body = template.copy(deep=True)
                body.translate((x_min, front_y_world, z_min), inplace=True)

            exit_at_front = not cardio
            exit_face_index: Optional[int] = None
            if box_key is not None:
                face_indices = self._box_face_cache.get(box_key)
                if face_indices is None and box_key in self._box_template_cache:
                    face_indices = self._compute_box_face_indices(self._box_template_cache[box_key])
                    self._box_face_cache[box_key] = face_indices
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

        # Cache deaktiviert - verursachte doppelte/falsch positionierte Geometrien
        # cache_geoms = [(mesh.copy(deep=True), exit_idx) for mesh, exit_idx in geoms]
        # self._speaker_geometry_cache[cache_key] = cache_geoms
        
        return geoms

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def _apply_flown_cabinet_shape(
        self,
        mesh: Any,
        *,
        front_height: float,
        back_height: float,
        top_reference: float,
        angle_point: str,
    ):
        """Verformt ein würfelförmiges Gehäuse in eine Keilform anhand der Höhenmetadaten."""
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

    def _add_overlay_mesh(
        self,
        mesh: Any,
        *,
        color: Optional[str] = None,
        opacity: float = 1.0,
        line_width: float = 2.0,
        scalars: Optional[str] = None,
        cmap: Optional[List[str]] = None,
        edge_color: Optional[str] = None,
        show_edges: bool = False,
        show_vertices: bool = False,  # Standard: Keine Eckpunkte anzeigen
        line_pattern: Optional[int] = None,
        line_repeat: int = 1,
        category: str = 'generic',
        render_lines_as_tubes: Optional[bool] = None,
        ) -> None:
        name = f"overlay_{self.overlay_counter}"
        self.overlay_counter += 1
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
        
        # 🎯 Stelle sicher, dass keine Eckpunkte angezeigt werden (nur Linien)
        if not show_vertices and hasattr(mesh, 'lines') and mesh.lines is not None:
            # Für PolyLines: Keine Vertices anzeigen
            kwargs['render_points_as_spheres'] = False
            kwargs['point_size'] = 0

        actor = self.plotter.add_mesh(mesh, **kwargs)
        
        # Prüfe ob Mesh ein Tube-Mesh ist: render_lines_as_tubes=None bedeutet bereits konvertiertes Tube-Mesh
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
                actor.prop.line_stipple_pattern = line_pattern
                actor.prop.line_stipple_repeat_factor = max(1, int(line_repeat))
            except Exception:  # noqa: BLE001
                pass
        elif is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
            # Für Tube-Meshes: Explizit durchgezogene Linie setzen (kein Stipple-Pattern)
            try:
                actor.prop.line_stipple_pattern = 0xFFFF  # Durchgezogen
                actor.prop.line_stipple_repeat_factor = 1
            except Exception:  # noqa: BLE001
                pass
        self.overlay_actor_names.append(name)
        self._category_actors.setdefault(category, []).append(name)
        return name

    @staticmethod
    def _to_float_array(values) -> np.ndarray:
        if values is None:
            return np.empty(0, dtype=float)
        try:
            array = np.asarray(values, dtype=float)
        except Exception:
            return np.empty(0, dtype=float)
        if array.ndim > 1:
            array = array.reshape(-1)
        return array.astype(float)

    def _remove_actor(self, name: str) -> None:
        try:
            self.plotter.remove_actor(name)
        except KeyError:
            pass
        if name in self.overlay_actor_names:
            self.overlay_actor_names.remove(name)
        for actors in self._category_actors.values():
            if name in actors:
                actors.remove(name)

    def _update_speaker_actor(
        self,
        actor_name: str,
        mesh: Any,
        exit_face_index: Optional[int],
        body_color: str,
        exit_color: str,
    ) -> None:
        actor = self.plotter.renderer.actors.get(actor_name)
        if actor is None:
            return
        mapper = getattr(actor, 'mapper', None)
        
        # Prüfe ob Mesh bereits Scalars hat (merged mesh mit exit_face_index = -1)
        has_scalars = 'speaker_face' in mesh.cell_data
        
        if exit_face_index == -1 or has_scalars:
            # Mesh hat bereits korrekte Scalars - nur Mapper updaten
            if mapper is not None:
                mapper.array_name = 'speaker_face'
                mapper.scalar_range = (0, 1)
                mapper.lookup_table = self.plotter._cmap_to_lut([body_color, exit_color])
        elif exit_face_index is not None and mesh.n_cells > 0:
            # Einzelnes Mesh - erstelle Scalars für eine Exit-Face
            scalars = np.zeros(mesh.n_cells, dtype=int)
            clamped_index = int(np.clip(exit_face_index, 0, mesh.n_cells - 1))
            scalars[clamped_index] = 1
            mesh.cell_data['speaker_face'] = scalars
            if mapper is not None:
                mapper.array_name = 'speaker_face'
                mapper.scalar_range = (0, 1)
                mapper.lookup_table = self.plotter._cmap_to_lut([body_color, exit_color])
                mapper.scalar_visibility = True
        else:
            if 'speaker_face' in mesh.cell_data:
                del mesh.cell_data['speaker_face']
            if mapper is not None:
                mapper.scalar_visibility = False
            if hasattr(actor, 'prop') and actor.prop is not None:
                actor.prop.color = body_color

    @staticmethod
    def _speaker_signature_from_mesh(mesh: Any, exit_face_index: Optional[int]) -> tuple:
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

    def _resolve_cabinet_entries(self, cabinet_raw, configuration) -> List[dict]:
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
                entries.extend(self._resolve_cabinet_entries(cabinet_raw.item(), configuration))
            else:
                entries.extend(self._resolve_cabinet_entries(cabinet_raw.tolist(), configuration))

        if not entries and isinstance(cabinet_raw, (list, tuple)):
            for item in cabinet_raw:
                if isinstance(item, dict):
                    entries.append(item)

        return entries

    def _get_box_template(self, width: float, depth: float, height: float) -> Any:
        key = (float(width), float(depth), float(height))
        template = self._box_template_cache.get(key)
        if template is None:
            template = self.pv.Box(bounds=(0.0, width, -depth, 0.0, 0.0, height))
            self._box_template_cache[key] = template
            self._box_face_cache[key] = self._compute_box_face_indices(template)
        elif key not in self._box_face_cache:
            self._box_face_cache[key] = self._compute_box_face_indices(template)
        return template

    @staticmethod
    def _compute_box_face_indices(mesh: Any) -> Tuple[Optional[int], Optional[int]]:
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

    @staticmethod
    def _get_speaker_name(speaker_array, index: int):
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

    @classmethod
    def _safe_float(cls, mapping, key: str, fallback: float | None = None) -> float:
        if isinstance(mapping, (float, int)):
            return float(mapping)
        if isinstance(mapping, str):
            try:
                return float(mapping)
            except Exception:  # noqa: BLE001
                return float(fallback) if cls._is_numeric(fallback) else 0.0
        if not isinstance(mapping, dict):
            return float(fallback) if cls._is_numeric(fallback) else 0.0
        value = mapping.get(key, fallback if fallback is not None else 0.0)
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return float(fallback) if cls._is_numeric(fallback) else 0.0

    @staticmethod
    def _array_value(values, index: int) -> float | None:
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

    @staticmethod
    def _is_numeric(value) -> bool:
        if isinstance(value, (float, int)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except Exception:  # noqa: BLE001
                return False
        return False

    @staticmethod
    def _float_sequence(values, target_length: int, default: float = 0.0) -> List[float]:
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

    @staticmethod
    def _sequence_length(values) -> int:
        if values is None:
            return 0
        try:
            return len(values)
        except TypeError:
            return 0


class SPLTimeControlBar(QtWidgets.QFrame):
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("spl_time_control")
        self.setStyleSheet(
            "QFrame#spl_time_control {"
            "  background-color: rgba(240, 240, 240, 200);"
            "  border: 1px solid rgba(0, 0, 0, 150);"
            "  border-radius: 4px;"
            "}"
        )
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self._frames = 1
        self._simulation_time = 0.2  # 200ms (Standard)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        
        # Container für Slider und schwarze Linie
        slider_container = QtWidgets.QWidget(self)
        slider_layout = QtWidgets.QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(0)
        
        # Schwarze Linie links vom Slider
        self.track_line = QtWidgets.QFrame(slider_container)
        self.track_line.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
        self.track_line.setFixedWidth(2)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical, slider_container)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)
        self.slider.setInvertedAppearance(True)  # min (t=0) unten
        self.slider.setStyleSheet(
            "QSlider::groove:vertical {"
            "  border: none;"
            "  background: rgba(200, 200, 200, 150);"
            "  width: 4px;"
            "  border-radius: 2px;"
            "}"
            "QSlider::handle:vertical {"
            "  background: rgba(50, 50, 50, 220);"
            "  border: 1px solid rgba(0, 0, 0, 200);"
            "  width: 12px;"
            "  height: 12px;"
            "  border-radius: 6px;"
            "  margin: 0px -4px;"
            "}"
            "QSlider::handle:vertical:hover {"
            "  background: rgba(30, 30, 30, 240);"
            "}"
        )
        
        slider_layout.addWidget(self.track_line)
        slider_layout.addWidget(self.slider, 1)
        
        layout.addWidget(slider_container, 1)
        
        # Zeit-Label als separates Widget außerhalb des Fader-Containers (wird in _reposition platziert)
        self.label = QtWidgets.QLabel("t=0", parent)
        self.label.setAlignment(QtCore.Qt.AlignHCenter)
        self.label.setStyleSheet("font-size: 9px; font-weight: bold; color: #333; background-color: rgba(240, 240, 240, 200); padding: 2px; border-radius: 2px;")
        self.label.hide()

        self.slider.valueChanged.connect(self._on_value_changed)
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event):  # noqa: D401
        if obj is self.parent() and event.type() == QtCore.QEvent.Resize:
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self):
        parent = self.parent()
        if parent is None:
            return
        margin = 12
        width = 32  # Schmaler: 32px statt 60px
        height = max(150, parent.height() - 2 * margin - 20)  # Platz für Label unterhalb
        x = parent.width() - width - margin
        y = margin
        self.setGeometry(x, y, width, height)
        
        # Positioniere Label unterhalb des Fader-Objekts (breiter als Fader für vollständigen Text)
        label_height = 16
        label_width = 60  # Breiter als Fader, damit Text nicht abgeschnitten wird
        label_x = x - (label_width - width) // 2  # Zentriert unter dem Fader
        label_y = y + height + 2
        self.label.setGeometry(label_x, label_y, label_width, label_height)

    def configure(self, frames: int, value: int, simulation_time: float = 0.2):
        frames = max(1, int(frames))
        self._frames = frames
        self._simulation_time = float(simulation_time)
        self.slider.setMaximum(frames - 1)
        clamped = max(0, min(value, frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(clamped)
        self.slider.blockSignals(False)
        self._update_label(clamped)
        self._reposition()
        # Zeige Label an, wenn Fader sichtbar ist
        if self.isVisible() and self.label:
            self.label.show()

    def _on_value_changed(self, value: int):
        self._update_label(value)
        self.valueChanged.emit(int(value))

    def _update_label(self, value: int):
        # Berechne Zeit in ms: Frames werden mit np.linspace(0.0, simulation_time, frames_per_period + 1) erstellt
        # Frame 0 = t=0, Frame 1..N = gleichmäßig über simulation_time verteilt
        if value == 0:
            time_ms = 0.0
        else:
            # self._frames = frames_per_period + 1 (inkl. Frame 0)
            # Frames 1..N sind gleichmäßig über simulation_time verteilt
            frames_per_period = self._frames - 1  # Ohne Frame 0
            if frames_per_period > 0:
                # Zeit für Frame value: value * (simulation_time / frames_per_period)
                # Das entspricht np.linspace(0.0, simulation_time, frames_per_period + 1)[value]
                time_ms = (value * self._simulation_time / frames_per_period) * 1000.0
            else:
                time_ms = 0.0
        
        # Zeige Zeit in ms (mit 1 Dezimalstelle)
        self.label.setText(f"t={time_ms:.1f}ms")
    
    def show(self):
        """Zeige Fader und Label."""
        super().show()
        if self.label:
            self.label.show()
    
    def hide(self):
        """Verstecke Fader und Label."""
        super().hide()
        if self.label:
            self.label.hide()


__all__ = ['SPL3DOverlayRenderer', 'SPLTimeControlBar']

