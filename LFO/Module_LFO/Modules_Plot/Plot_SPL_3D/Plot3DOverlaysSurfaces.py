"""Surfaces Overlay-Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import numpy as np

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysBase import SPL3DOverlayBase
from Module_LFO.Modules_Data.SurfaceValidator import validate_and_optimize_surface, triangulate_points

DEBUG_OVERLAY_PERF = bool(int(os.environ.get("LFO_DEBUG_OVERLAY_PERF", "1")))


class SPL3DOverlaySurfaces(SPL3DOverlayBase):
    """
    Mixin-Klasse für Surfaces Overlay-Rendering.
    
    Zeichnet alle aktivierten, nicht versteckten Surfaces als Polygone im 3D-Plot.
    """
    
    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert das Surfaces Overlay."""
        super().__init__(plotter, pv_module)
        self._overlay_prefix = "surf_"
        self._last_surfaces_state: Optional[tuple] = None
    
    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um State zurückzusetzen."""
        super().clear_category(category)
        if category == 'surfaces':
            self._last_surfaces_state = None
    
    def draw_surfaces(self, settings, container=None, create_empty_plot_surfaces=False) -> None:
        """Zeichnet alle aktivierten, nicht versteckten Surfaces als Polygone im 3D-Plot.
        
        Args:
            settings: Settings-Objekt
            container: Container-Objekt (optional)
            create_empty_plot_surfaces: Wenn True, werden graue Flächen für enabled Surfaces erstellt (nur für leeren Plot)
        """
        print(f"[PLOT] draw_surfaces() aufgerufen (create_empty_plot_surfaces={create_empty_plot_surfaces})")
        t_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Erstelle Signatur für Änderungsdetektion
        surface_definitions = getattr(settings, 'surface_definitions', {})
        if not isinstance(surface_definitions, dict):
            self.clear_category('surfaces')
            self._last_surfaces_state = None
            return

        # Hole aktives Surface bzw. Highlight-Liste für Signatur und Markierung
        active_surface_id = getattr(settings, 'active_surface_id', None)
        highlight_ids = getattr(settings, 'active_surface_highlight_ids', None)
        if isinstance(highlight_ids, (list, tuple, set)):
            active_ids_set = {str(sid) for sid in highlight_ids}
        else:
            active_ids_set = set()
        if active_surface_id is not None:
            active_ids_set.add(str(active_surface_id))

        t_sig_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        # 🎯 NEU: Hole Gruppen-Status für Signatur (damit Signatur bei Gruppen-Statusänderung invalidiert wird)
        surface_groups = getattr(settings, 'surface_groups', {})
        group_status_for_signature: dict[str, tuple[bool, bool]] = {}  # group_id -> (enabled, hidden)
        if isinstance(surface_groups, dict):
            for group_id, group_data in surface_groups.items():
                if hasattr(group_data, 'enabled'):
                    group_status_for_signature[group_id] = (
                        bool(group_data.enabled),
                        bool(getattr(group_data, 'hidden', False))
                    )
                elif isinstance(group_data, dict):
                    group_status_for_signature[group_id] = (
                        bool(group_data.get('enabled', True)),
                        bool(group_data.get('hidden', False))
                    )
        
        surfaces_signature: List[tuple] = []
        for surface_id in sorted(surface_definitions.keys()):
            surface_def = surface_definitions[surface_id]
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
                group_id = getattr(surface_def, 'group_id', None)
            else:
                enabled = bool(surface_def.get('enabled', False))
                hidden = bool(surface_def.get('hidden', False))
                points = surface_def.get('points', [])
                group_id = surface_def.get('group_id') or surface_def.get('group_name')
            
            # 🎯 NEU: Berücksichtige Gruppen-Status in Signatur
            effective_enabled = enabled
            effective_hidden = hidden
            if group_id and group_id in group_status_for_signature:
                group_enabled, group_hidden = group_status_for_signature[group_id]
                if group_hidden:
                    effective_hidden = True
                elif not group_enabled:
                    effective_enabled = False
            
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
                effective_enabled,
                effective_hidden,
                tuple(points_tuple)
            ))
        t_sig_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        has_speaker_arrays_for_signature = False
        if isinstance(speaker_arrays, dict):
            for array in speaker_arrays.values():
                if not getattr(array, 'hide', False):
                    has_speaker_arrays_for_signature = True
                    break
        
        # 🎯 WICHTIG: Signatur-Struktur muss mit _compute_overlay_signatures übereinstimmen!
        # Struktur: (surfaces_signature_tuple, active_surface_id, highlight_ids_tuple, has_spl_data_for_signature)
        active_surface_id = getattr(settings, 'active_surface_id', None)
        highlight_ids_tuple = tuple(sorted(active_ids_set))
        
        # Prüfe ob SPL-Daten vorhanden sind (für Signatur-Konsistenz)
        has_spl_data_for_signature = False
        try:
            if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                texture_actor_names = [name for name in self.plotter.renderer.actors.keys() if name.startswith('spl_surface_tex_')]
                has_texture_actors = len(texture_actor_names) > 0
                if spl_surface_actor is not None or spl_floor_actor is not None or has_texture_actors:
                    has_spl_data_for_signature = True
        except Exception:
            pass
        
        signature_tuple = (tuple(surfaces_signature), active_surface_id, highlight_ids_tuple, has_spl_data_for_signature)
        
        signature_changed = True
        if self._last_surfaces_state is not None and not create_empty_plot_surfaces:
            # Prüfe ob Signatur-Struktur übereinstimmt (4-Tupel wie in _compute_overlay_signatures)
            if len(self._last_surfaces_state) == 4:
                last_surfaces_tuple, last_active_id, last_highlight_ids, last_has_spl = self._last_surfaces_state
                last_ids_set = set(str(sid) for sid in last_highlight_ids) if isinstance(last_highlight_ids, (list, tuple, set)) else set()
                # Prüfe ob sich Surface-Definitionen ODER Highlight-IDs geändert haben
                surfaces_changed = (last_surfaces_tuple != tuple(surfaces_signature))
                highlights_changed = (last_ids_set != active_ids_set)
                active_id_changed = (last_active_id != active_surface_id)
                spl_data_changed = (last_has_spl != has_spl_data_for_signature)
                # DEBUG: Ausgabe wenn Highlights sich geändert haben
                if highlights_changed:
                    print(f"[DEBUG SURFACE] Highlights changed: last={last_ids_set}, current={active_ids_set}")
                if not surfaces_changed and not highlights_changed and not active_id_changed and not spl_data_changed:
                    signature_changed = False
            elif len(self._last_surfaces_state) == 2:
                # Alte Signatur-Struktur (2-Tupel) - behandle als geändert
                pass
            elif self._last_surfaces_state == signature_tuple:
                signature_changed = False
        
        # 🎯 WICHTIG: Auch wenn sich nur die Highlight-IDs geändert haben, müssen wir neu zeichnen
        # (damit disabled Surfaces rot umrandet werden, wenn sie angeklickt werden)
        if not signature_changed and not create_empty_plot_surfaces:
            return
        
        t_clear_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        self.clear_category('surfaces')
        try:
            empty_plot_actor = self.plotter.renderer.actors.get('surface_enabled_empty_plot_batch')
            if empty_plot_actor is not None:
                self.plotter.remove_actor('surface_enabled_empty_plot_batch')
                if 'surface_enabled_empty_plot_batch' in self.overlay_actor_names:
                    self.overlay_actor_names.remove('surface_enabled_empty_plot_batch')
                if 'surfaces' in self._category_actors and 'surface_enabled_empty_plot_batch' in self._category_actors['surfaces']:
                    self._category_actors['surfaces'].remove('surface_enabled_empty_plot_batch')
        except Exception:
            pass
        t_clear_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        self._last_surfaces_state = signature_tuple
        
        has_speaker_arrays = has_speaker_arrays_for_signature
        
        z_offset = self._planar_z_offset
        surfaces_drawn = 0
        t_draw_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        active_enabled_points_list = []
        active_enabled_lines_list = []
        active_disabled_points_list = []
        active_disabled_lines_list = []
        inactive_points_list = []
        inactive_lines_list = []
        inactive_surface_ids = []
        inactive_surface_enabled = []
        disabled_surface_ids = []
        disabled_surface_points = []
        # 🎯 NEU: Listen für ungültige enabled Surfaces (grau hinterlegen)
        invalid_enabled_points_list = []
        invalid_enabled_faces_list = []
        invalid_enabled_lines_list = []
        
        tolerance = 1e-6
        
        enabled_count = 0
        disabled_count = 0
        
        # 🎯 NEU: Erstelle Mapping von group_id zu Gruppen-Status für schnellen Zugriff
        surface_groups = getattr(settings, 'surface_groups', {})
        group_status: dict[str, dict[str, bool]] = {}  # group_id -> {'enabled': bool, 'hidden': bool}
        if isinstance(surface_groups, dict):
            for group_id, group_data in surface_groups.items():
                # #region agent log
                try:
                    import json
                    group_data_type = type(group_data).__name__
                    group_data_enabled_raw = getattr(group_data, 'enabled', None) if hasattr(group_data, 'enabled') else (group_data.get('enabled') if isinstance(group_data, dict) else None)
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H",
                            "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:group_status_extraction",
                            "message": "Extracting group status",
                            "data": {
                                "group_id": str(group_id),
                                "group_data_type": group_data_type,
                                "hasattr_enabled": hasattr(group_data, 'enabled'),
                                "isinstance_dict": isinstance(group_data, dict),
                                "group_data_enabled_raw": bool(group_data_enabled_raw) if group_data_enabled_raw is not None else None
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                if hasattr(group_data, 'enabled'):
                    group_status[group_id] = {
                        'enabled': bool(group_data.enabled),
                        'hidden': bool(getattr(group_data, 'hidden', False))
                    }
                elif isinstance(group_data, dict):
                    group_status[group_id] = {
                        'enabled': bool(group_data.get('enabled', True)),
                        'hidden': bool(group_data.get('hidden', False))
                    }
        # #region agent log
        try:
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                group_status_summary = {gid: {'enabled': status['enabled'], 'hidden': status['hidden']} for gid, status in group_status.items()}
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:group_status_loaded",
                    "message": "Group status loaded in draw_surfaces",
                    "data": {
                        "group_status": group_status_summary,
                        "surface_groups_type": type(surface_groups).__name__,
                        "surface_groups_keys": list(surface_groups.keys()) if isinstance(surface_groups, dict) else None
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        for surface_id, surface_def in surface_definitions.items():
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
                surface_obj = surface_def
                group_id = getattr(surface_def, 'group_id', None)
            else:
                enabled = surface_def.get('enabled', False)
                hidden = surface_def.get('hidden', False)
                points = surface_def.get('points', [])
                group_id = surface_def.get('group_id') or surface_def.get('group_name')
                # Erstelle SurfaceDefinition-Objekt für Validierung
                try:
                    surface_obj = SurfaceDefinition.from_dict(str(surface_id), surface_def)
                except Exception:
                    surface_obj = None
            
            # 🎯 NEU: Berücksichtige Gruppen-Status
            # Wenn Surface zu einer Gruppe gehört, verwende Gruppen-Status
            # #region agent log
            try:
                import json
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "F",
                        "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:group_check",
                        "message": "Checking group status for surface",
                        "data": {
                            "surface_id": str(surface_id),
                            "group_id": str(group_id) if group_id else None,
                            "group_id_in_group_status": group_id in group_status if group_id else False,
                            "group_status_keys": list(group_status.keys()),
                            "enabled_before_override": enabled,
                            "hidden_before_override": hidden
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            if group_id and group_id in group_status:
                group_enabled = group_status[group_id]['enabled']
                group_hidden = group_status[group_id]['hidden']
                # #region agent log
                try:
                    import json
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H",
                            "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:group_override_applied",
                            "message": "Group override applied",
                            "data": {
                                "surface_id": str(surface_id),
                                "group_id": str(group_id),
                                "group_enabled": bool(group_enabled),
                                "group_hidden": bool(group_hidden),
                                "enabled_before": enabled,
                                "enabled_after": enabled if group_hidden else (False if not group_enabled else enabled)
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # Wenn Gruppe hidden ist → Surface komplett überspringen (nichts zeichnen)
                if group_hidden:
                    continue
                
                # Wenn Gruppe disabled ist → Surface als disabled behandeln (gestrichelt zeichnen)
                if not group_enabled:
                    enabled = False
                    # #region agent log
                    try:
                        import json
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H",
                                "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:enabled_set_to_false",
                                "message": "enabled set to False due to disabled group",
                                "data": {
                                    "surface_id": str(surface_id),
                                    "group_id": str(group_id),
                                    "enabled": False
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    # hidden bleibt wie vom Surface selbst gesetzt (für zusätzliche Filterung)
            
            # Surface ist selbst hidden → überspringen
            if hidden:
                continue
            
            if len(points) < 3:
                continue
            
            # 🎯 VALIDIERUNG: Prüfe ob enabled Surface für SPL-Berechnung verwendet werden kann
            is_valid_for_spl = True
            if enabled and surface_obj is not None:
                try:
                    validation_result = validate_and_optimize_surface(
                        surface_obj,
                        round_to_cm=False,
                        remove_redundant=False,
                    )
                    is_valid_for_spl = validation_result.is_valid
                    
                    # 🎯 ZUSÄTZLICHE PRÜFUNG: Für Surfaces mit 4 oder mehr Punkten prüfe ob Triangulation möglich ist
                    # (auch 4 Punkte müssen trianguliert werden können, wenn sie nicht planar sind)
                    if is_valid_for_spl and len(points) >= 4:
                        try:
                            # Versuche Triangulation
                            triangles = triangulate_points(points)
                            if not triangles or len(triangles) == 0:
                                # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                                is_valid_for_spl = False
                        except Exception as e:
                            # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                            is_valid_for_spl = False
                except Exception:
                    # Bei Fehler in Validierung: als ungültig behandeln
                    is_valid_for_spl = False
            
            if enabled:
                enabled_count += 1
            else:
                disabled_count += 1
            
            is_active = (str(surface_id) in active_ids_set)
            
            try:
                point_coords = []
                for point in points:
                    x = float(point.get('x', 0.0))
                    y = float(point.get('y', 0.0))
                    z_original = float(point.get('z', 0.0))
                    z = z_original + z_offset
                    point_coords.append([x, y, z])
                    
                    # #region agent log
                    try:
                        import json
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "G",
                                "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:point_z_offset_applied",
                                "message": "Z-offset applied to point",
                                "data": {
                                    "surface_id": surface_id,
                                    "x": x,
                                    "y": y,
                                    "z_original": z_original,
                                    "z_offset": z_offset,
                                    "z_final": z,
                                    "point_index": len(point_coords) - 1
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                
                if len(point_coords) < 3:
                    continue
                
                first_point = point_coords[0]
                last_point = point_coords[-1]
                is_already_closed = (
                    abs(first_point[0] - last_point[0]) < tolerance
                    and abs(first_point[1] - last_point[1]) < tolerance
                    and abs(first_point[2] - last_point[2]) < tolerance
                )

                if is_already_closed:
                    closed_coords = point_coords
                else:
                    closed_coords = point_coords + [point_coords[0]]
                
                n_points = len(closed_coords)
                closed_coords_array = np.array(closed_coords, dtype=float)
                
                if not enabled:
                    # #region agent log
                    try:
                        import json
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "F",
                                "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:adding_disabled_surface",
                                "message": "Adding surface to disabled_surface_ids",
                                "data": {
                                    "surface_id": str(surface_id),
                                    "group_id": str(group_id) if group_id else None,
                                    "enabled": enabled,
                                    "hidden": hidden,
                                    "disabled_surface_ids_count_before": len(disabled_surface_ids)
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    disabled_surface_ids.append(str(surface_id))
                    # 🎯 WICHTIG: Verwende einen deutlich höheren Z-Offset für disabled Polygone,
                    # damit sie über den SPL-Plots liegen und sichtbar sind
                    # closed_coords_array hat bereits z_offset (0.005m) angewendet, füge zusätzlich 0.01m hinzu
                    # Gesamt: 0.015m über SPL-Plots (die bei Z=0 liegen)
                    disabled_coords_with_offset = closed_coords_array.copy()
                    if len(disabled_coords_with_offset.shape) == 2 and disabled_coords_with_offset.shape[1] >= 3:
                        disabled_coords_with_offset[:, 2] += 0.01  # Erhöhe Z um 1cm zusätzlich (statt 1mm)
                    disabled_surface_points.append(disabled_coords_with_offset)
                    
                    if is_active:
                        active_disabled_points_list.append(closed_coords_array)
                        active_disabled_lines_list.append(n_points)
                    else:
                        inactive_points_list.append(closed_coords_array)
                        inactive_lines_list.append(n_points)
                        inactive_surface_ids.append(str(surface_id))
                        inactive_surface_enabled.append(False)
                else:
                    # 🎯 Enabled Surface: Prüfe ob gültig für SPL
                    # #region agent log
                    try:
                        import json
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "G",
                                "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:enabled_surface_branch",
                                "message": "Processing enabled surface branch",
                                "data": {
                                    "surface_id": str(surface_id),
                                    "group_id": str(group_id) if group_id else None,
                                    "enabled": enabled,
                                    "hidden": hidden,
                                    "is_valid_for_spl": is_valid_for_spl,
                                    "will_use_invalid_enabled_list": not is_valid_for_spl
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    if not is_valid_for_spl:
                        # Ungültige enabled Surface: grau hinterlegen, aber Rahmen trotzdem zeichnen
                        invalid_enabled_points_list.append(closed_coords_array)
                        invalid_enabled_faces_list.append(n_points)
                        invalid_enabled_lines_list.append(n_points)
                        # Rahmen wird separat gezeichnet (wie bei gültigen Surfaces)
                        if is_active:
                            active_enabled_points_list.append(closed_coords_array)
                            active_enabled_lines_list.append(n_points)
                        else:
                            inactive_points_list.append(closed_coords_array)
                            inactive_lines_list.append(n_points)
                            inactive_surface_ids.append(str(surface_id))
                            inactive_surface_enabled.append(True)
                    else:
                        # Gültige enabled Surface: normal behandeln
                        if is_active:
                            active_enabled_points_list.append(closed_coords_array)
                            active_enabled_lines_list.append(n_points)
                        else:
                            inactive_points_list.append(closed_coords_array)
                            inactive_lines_list.append(n_points)
                            inactive_surface_ids.append(str(surface_id))
                            inactive_surface_enabled.append(True)
                
                surfaces_drawn += 1
            except (ValueError, TypeError, AttributeError, Exception):
                continue
        
        has_spl_data = False
        try:
            if hasattr(self.plotter, '_surface_texture_actors'):
                direct_texture_count = len(getattr(self.plotter, '_surface_texture_actors', {}))
                if direct_texture_count > 0:
                    has_spl_data = True
            
            if not has_spl_data and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                texture_actor_names = [name for name in self.plotter.renderer.actors.keys() if name.startswith('spl_surface_tex_')]
                has_texture_actors = len(texture_actor_names) > 0
                if spl_surface_actor is not None or spl_floor_actor is not None or has_texture_actors:
                    has_spl_data = True
            
            if container is not None and hasattr(container, 'calculation_spl'):
                calc_spl = container.calculation_spl
                if isinstance(calc_spl, dict) and calc_spl.get('sound_field_p') is not None:
                    if not has_spl_data:
                        has_spl_data = True
        except Exception:
            pass
        
        # Zeichne aktive ENABLED Surfaces als Batch
        if active_enabled_points_list:
            try:
                all_active_enabled_points = np.vstack(active_enabled_points_list)
                active_enabled_lines_array = []
                point_offset = 0
                for n_pts in active_enabled_lines_list:
                    active_enabled_lines_array.append(n_pts)
                    active_enabled_lines_array.extend(range(point_offset, point_offset + n_pts))
                    point_offset += n_pts
                
                active_enabled_polyline = self.pv.PolyData(all_active_enabled_points)
                active_enabled_polyline.lines = active_enabled_lines_array
                try:
                    active_enabled_polyline.verts = np.empty(0, dtype=np.int64)
                except Exception:
                    try:
                        active_enabled_polyline.verts = []
                    except Exception:
                        pass
                
                self._add_overlay_mesh(
                    active_enabled_polyline,
                    color='#FF0000',
                    # Rote Surface-Umrandungen mit Zoom-Skalierung (dicker für bessere Sichtbarkeit)
                    line_width=self._get_scaled_line_width(2.0, apply_zoom=True),
                    opacity=1.0,
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=True,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # Zeichne aktive DISABLED Surfaces als Batch
        if active_disabled_points_list:
            try:
                all_active_disabled_points = np.vstack(active_disabled_points_list)
                active_disabled_lines_array = []
                point_offset = 0
                for n_pts in active_disabled_lines_list:
                    active_disabled_lines_array.append(n_pts)
                    active_disabled_lines_array.extend(range(point_offset, point_offset + n_pts))
                    point_offset += n_pts
                
                active_disabled_polyline = self.pv.PolyData(all_active_disabled_points)
                active_disabled_polyline.lines = active_disabled_lines_array
                try:
                    active_disabled_polyline.verts = np.empty(0, dtype=np.int64)
                except Exception:
                    try:
                        active_disabled_polyline.verts = []
                    except Exception:
                        pass
                
                self._add_overlay_mesh(
                    active_disabled_polyline,
                    color='#FF0000',
                    # Rote Surface-Umrandungen mit Zoom-Skalierung (dicker für bessere Sichtbarkeit)
                    line_width=self._get_scaled_line_width(2.0, apply_zoom=True),
                    opacity=1.0,
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=True,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # Zeichne Fläche für enabled UND disabled Surfaces wenn create_empty_plot_surfaces=True
        if create_empty_plot_surfaces:
            enabled_points_for_empty_plot = []
            enabled_faces_for_empty_plot = []
            point_offset = 0
            
            # 🎯 FIX: Zeichne enabled Surfaces als graue Flächen
            if active_enabled_points_list:
                for idx, points in enumerate(active_enabled_points_list):
                    n_pts = len(points)
                    enabled_points_for_empty_plot.append(points)
                    face = [n_pts] + [point_offset + i for i in range(n_pts)]
                    enabled_faces_for_empty_plot.extend(face)
                    point_offset += n_pts
            
            inactive_enabled_count = 0
            if inactive_points_list:
                for idx, points in enumerate(inactive_points_list):
                    if idx < len(inactive_surface_ids) and idx < len(inactive_surface_enabled):
                        is_enabled = inactive_surface_enabled[idx]
                        if is_enabled:
                            inactive_enabled_count += 1
                            n_pts = len(points)
                            enabled_points_for_empty_plot.append(points)
                            face = [n_pts] + [point_offset + i for i in range(n_pts)]
                            enabled_faces_for_empty_plot.extend(face)
                            point_offset += n_pts
            
            # 🎯 FIX: Zeichne auch disabled Surfaces als graue Flächen (wenn create_empty_plot_surfaces=True)
            # Disabled Surfaces werden bereits als gestrichelte Linien gezeichnet, aber hier als graue Flächen
            if disabled_surface_points:
                for idx, points in enumerate(disabled_surface_points):
                    # Verwende die Punkte OHNE zusätzlichen Z-Offset (für Flächen)
                    # (disabled_surface_points hat bereits +0.01m Offset, entferne es für Flächen)
                    points_for_face = points.copy()
                    if len(points_for_face.shape) == 2 and points_for_face.shape[1] >= 3:
                        points_for_face[:, 2] -= 0.01  # Entferne zusätzlichen Z-Offset für Flächen
                    n_pts = len(points_for_face)
                    enabled_points_for_empty_plot.append(points_for_face)
                    face = [n_pts] + [point_offset + i for i in range(n_pts)]
                    enabled_faces_for_empty_plot.extend(face)
                    point_offset += n_pts
            
            if enabled_points_for_empty_plot:
                try:
                    all_enabled_points = np.vstack(enabled_points_for_empty_plot)
                    enabled_polygon_mesh = self.pv.PolyData(all_enabled_points)
                    enabled_polygon_mesh.faces = enabled_faces_for_empty_plot
                    actor_name = "surface_enabled_empty_plot_batch"
                    actor = self.plotter.add_mesh(
                        enabled_polygon_mesh,
                        name=actor_name,
                        color='#D3D3D3',
                        opacity=0.8,
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        show_edges=False,
                    )
                    try:
                        if actor is not None and hasattr(actor, "SetPickable"):
                            actor.SetPickable(False)
                    except Exception:
                        pass
                    
                    if actor_name not in self.overlay_actor_names:
                        self.overlay_actor_names.append(actor_name)
                    self._category_actors.setdefault('surfaces', []).append(actor_name)
                except Exception:
                    import traceback
                    traceback.print_exc()
        else:
            # 🎯 Entferne graue Fläche für enabled Surfaces, wenn SPL-Daten geplottet werden
            # Die Rahmen (Wireframes) bleiben bestehen, da sie separat gezeichnet werden
            try:
                empty_plot_actor = self.plotter.renderer.actors.get('surface_enabled_empty_plot_batch')
                if empty_plot_actor is not None:
                    self.plotter.remove_actor('surface_enabled_empty_plot_batch')
                    if 'surface_enabled_empty_plot_batch' in self.overlay_actor_names:
                        self.overlay_actor_names.remove('surface_enabled_empty_plot_batch')
                    if 'surfaces' in self._category_actors and 'surface_enabled_empty_plot_batch' in self._category_actors['surfaces']:
                        self._category_actors['surfaces'].remove('surface_enabled_empty_plot_batch')
            except Exception:
                pass
        
        # 🎯 NEU: Zeichne ungültige enabled Surfaces grau hinterlegt (immer, auch wenn SPL-Daten vorhanden)
        # #region agent log
        try:
            import json
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "G",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:before_invalid_enabled_rendering",
                    "message": "Before rendering invalid enabled surfaces",
                    "data": {
                        "invalid_enabled_points_list_count": len(invalid_enabled_points_list),
                        "disabled_surface_ids_count": len(disabled_surface_ids),
                        "has_spl_data": has_spl_data
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        if invalid_enabled_points_list:
            try:
                all_invalid_points = []
                all_invalid_faces = []
                point_offset = 0
                
                for points_array in invalid_enabled_points_list:
                    # Sammle alle Punkte zuerst
                    n_pts = len(points_array)
                    all_invalid_points.append(points_array)
                    
                    # 🎯 Trianguliere für nicht-planare Flächen
                    # Konvertiere numpy array zurück zu Dict-Format für triangulate_points
                    points_dict = []
                    for pt in points_array:
                        # Entferne z_offset für Validierung
                        z_original = pt[2] - z_offset
                        points_dict.append({
                            'x': float(pt[0]),
                            'y': float(pt[1]),
                            'z': float(z_original)
                        })
                    
                    # Versuche Triangulation
                    triangles = triangulate_points(points_dict)
                    
                    # 🎯 DEBUG: Berechne erwartete Polygon-Fläche
                    from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import _calculate_polygon_area
                    expected_polygon_area = _calculate_polygon_area(points_dict)
                    
                    # Versuche Triangulation
                    triangles = triangulate_points(points_dict)
                    
                    triangulation_success = False
                    if triangles and len(triangles) > 0:
                        # Triangulation erfolgreich: Verwende Dreiecke
                        for tri in triangles:
                            # Finde Indizes der Punkte im points_array
                            tri_indices = []
                            for tri_pt in tri:
                                # Suche passenden Punkt im points_array
                                found = False
                                for idx, pt in enumerate(points_array):
                                    # 🎯 FIX: Prüfe ob Punkt übereinstimmt (mit Z-Offset)
                                    pt_z_with_offset = pt[2]  # pt[2] sollte bereits Z-Offset haben
                                    tri_pt_z_with_offset = tri_pt['z'] + z_offset  # tri_pt['z'] ist ohne Offset
                                    if (abs(pt[0] - tri_pt['x']) < tolerance and
                                        abs(pt[1] - tri_pt['y']) < tolerance and
                                        abs(pt_z_with_offset - tri_pt_z_with_offset) < tolerance):
                                        tri_indices.append(point_offset + idx)
                                        found = True
                                        break
                                    
                                    # #region agent log
                                    try:
                                        import json
                                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "G",
                                                "location": "Plot3DOverlaysSurfaces.py:triangulation_point_match",
                                                "message": "Triangulation point matching",
                                                "data": {
                                                    "surface_id": surface_id,
                                                    "pt_x": float(pt[0]),
                                                    "pt_y": float(pt[1]),
                                                    "pt_z": float(pt[2]),
                                                    "tri_pt_x": float(tri_pt['x']),
                                                    "tri_pt_y": float(tri_pt['y']),
                                                    "tri_pt_z": float(tri_pt['z']),
                                                    "z_offset": z_offset,
                                                    "pt_z_with_offset": float(pt_z_with_offset),
                                                    "tri_pt_z_with_offset": float(tri_pt_z_with_offset),
                                                    "x_diff": abs(pt[0] - tri_pt['x']),
                                                    "y_diff": abs(pt[1] - tri_pt['y']),
                                                    "z_diff": abs(pt_z_with_offset - tri_pt_z_with_offset),
                                                    "found": found
                                                },
                                                "timestamp": int(__import__('time').time() * 1000)
                                            }) + "\n")
                                    except Exception:
                                        pass
                                    # #endregion
                                
                                if not found:
                                    # Punkt nicht gefunden - verwende Polygon statt Triangulation
                                    tri_indices = None
                                    break
                            
                            if tri_indices and len(tri_indices) == 3:
                                # Füge Dreieck hinzu (Format: [3, idx1, idx2, idx3])
                                all_invalid_faces.extend([3] + tri_indices)
                                triangulation_success = True
                            else:
                                # Triangulation fehlgeschlagen: Verwende Polygon
                                triangulation_success = False
                                break
                    
                    if not triangulation_success:
                        # Keine Triangulation möglich oder fehlgeschlagen: Verwende Polygon
                        face = [n_pts] + [point_offset + i for i in range(n_pts)]
                        all_invalid_faces.extend(face)
                    
                    point_offset += n_pts
                
                if all_invalid_points and all_invalid_faces:
                    all_points = np.vstack(all_invalid_points)
                    invalid_polygon_mesh = self.pv.PolyData(all_points)
                    invalid_polygon_mesh.faces = all_invalid_faces
                    
                    actor_name = "surface_invalid_enabled_batch"
                    actor = self.plotter.add_mesh(
                        invalid_polygon_mesh,
                        name=actor_name,
                        color='#808080',  # Grau
                        opacity=0.6,
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        show_edges=False,
                    )
                    try:
                        if actor is not None and hasattr(actor, "SetPickable"):
                            actor.SetPickable(False)
                    except Exception:
                        pass
                    
                    if actor_name not in self.overlay_actor_names:
                        self.overlay_actor_names.append(actor_name)
                    self._category_actors.setdefault('surfaces', []).append(actor_name)
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # #region agent log
        import json
        import time as time_module
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:before_clear_old_actors",
                    "message": "Before clearing old disabled actors",
                    "data": {
                        "overlay_actor_names": self.overlay_actor_names[:10] if isinstance(self.overlay_actor_names, list) else [],
                        "category_actors_surfaces": list(self._category_actors.get('surfaces', []))[:10] if isinstance(self._category_actors, dict) else []
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        old_inactive_actors = [
            name for name in self.overlay_actor_names
            if name in ("surface_disabled_polygons_batch", "surface_disabled_edges_batch")
        ]
        
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:old_inactive_actors",
                    "message": "Old inactive actors to remove",
                    "data": {
                        "old_inactive_actors": old_inactive_actors
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        for name in old_inactive_actors:
            try:
                self.plotter.remove_actor(name)
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:removed_actor",
                            "message": "Removed old disabled actor",
                            "data": {"actor_name": name},
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
            except Exception:
                pass
            if name in self.overlay_actor_names:
                self.overlay_actor_names.remove(name)
            if 'surfaces' in self._category_actors and name in self._category_actors['surfaces']:
                self._category_actors['surfaces'].remove(name)
        
        # 🎯 WICHTIG: Die graue Fläche für disabled Surfaces wird immer erstellt (auch wenn SPL-Daten vorhanden sind)
        # Die alte Fläche wurde bereits oben entfernt (Zeile 703-715), daher ist hier keine weitere Aktion nötig
        
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:disabled_surfaces_collected",
                    "message": "Disabled surfaces collected",
                    "data": {
                        "disabled_surface_count": len(disabled_surface_ids),
                        "disabled_surface_ids": disabled_surface_ids[:10],  # First 10 for brevity
                        "has_spl_data": has_spl_data
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        print(f"[DEBUG disabled_polygons] disabled_surface_ids: {len(disabled_surface_ids)} Surfaces")
        print(f"[DEBUG disabled_polygons] disabled_surface_points: {len(disabled_surface_points)} Punkt-Listen")
        
        valid_disabled_polygons = []
        
        for idx, surface_id in enumerate(disabled_surface_ids):
            if idx < len(disabled_surface_points):
                try:
                    points = disabled_surface_points[idx]
                    n_pts = len(points)
                    
                    if n_pts < 3:
                        continue
                    
                    points_array = np.array(points)
                    if len(points_array) < 3:
                        continue
                    
                    vectors = points_array[1:] - points_array[:-1]
                    if len(vectors) >= 2:
                        v1 = vectors[0]
                        v1_norm = np.linalg.norm(v1)
                        if v1_norm < 1e-6:
                            continue
                        v1_normalized = v1 / v1_norm
                        
                        all_parallel = True
                        for v in vectors[1:]:
                            v_norm = np.linalg.norm(v)
                            if v_norm < 1e-6:
                                all_parallel = False
                                break
                            v_normalized = v / v_norm
                            cross = np.cross(v1_normalized, v_normalized)
                            if np.linalg.norm(cross) > 1e-3:
                                all_parallel = False
                                break
                        
                        if all_parallel:
                            continue
                    
                    valid_disabled_polygons.append((points, surface_id))
                except Exception as e:
                    continue
        
        if valid_disabled_polygons:
            print(f"[PLOT] draw_surfaces: {len(valid_disabled_polygons)} disabled Surfaces validiert")
        
        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:before_create_disabled_polygons",
                    "message": "Before creating disabled polygons",
                    "data": {
                        "valid_disabled_polygons_count": len(valid_disabled_polygons),
                        "has_spl_data": has_spl_data,
                        "create_empty_plot_surfaces": create_empty_plot_surfaces
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        if valid_disabled_polygons:
            try:
                print(f"[DEBUG disabled_polygons] Erstelle graue Fläche für {len(valid_disabled_polygons)} disabled Surfaces")
                all_polygon_points = []
                all_polygon_faces = []
                point_offset = 0
                
                for points, surface_id in valid_disabled_polygons:
                    n_pts = len(points)
                    all_polygon_points.append(points)
                    face = [n_pts] + [point_offset + i for i in range(n_pts)]
                    all_polygon_faces.extend(face)
                    point_offset += n_pts
                
                if all_polygon_points:
                    all_points = np.vstack(all_polygon_points)
                    # #region agent log
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "F",
                                "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:disabled_polygons_z_offset",
                                "message": "Disabled polygons Z-offset check",
                                "data": {
                                    "all_points_shape": all_points.shape,
                                    "z_min": float(np.min(all_points[:, 2])) if all_points.shape[1] >= 3 else None,
                                    "z_max": float(np.max(all_points[:, 2])) if all_points.shape[1] >= 3 else None,
                                    "z_mean": float(np.mean(all_points[:, 2])) if all_points.shape[1] >= 3 else None,
                                    "expected_z_offset": 0.015  # 0.005 (base) + 0.01 (additional)
                                },
                                "timestamp": int(time_module.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    polygon_mesh = self.pv.PolyData(all_points)
                    polygon_mesh.faces = all_polygon_faces
                    
                    actor_name = "surface_disabled_polygons_batch"
                    # 🎯 Erhöhe Opacity auf 1.0 für bessere Sichtbarkeit
                    # Zusätzlich zum erhöhten Z-Offset (0.015m) für maximale Sichtbarkeit über SPL-Plots
                    actor = self.plotter.add_mesh(
                        polygon_mesh,
                        name=actor_name,
                        color='#D3D3D3',
                        opacity=1.0,  # Vollständig opak für maximale Sichtbarkeit
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        show_edges=False,
                    )
                    try:
                        if actor is not None and hasattr(actor, "SetPickable"):
                            actor.SetPickable(False)
                        # 🎯 DEBUG: Prüfe ob Actor wirklich im Plotter vorhanden ist
                        actor_in_renderer = None
                        try:
                            if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                                actor_in_renderer = self.plotter.renderer.actors.get(actor_name)
                        except Exception:
                            pass
                        print(f"[PLOT] draw_surfaces: Actor '{actor_name}' erstellt ({len(all_polygon_points)} Polygone, {len(all_points)} Punkte)")
                    except Exception as e:
                        pass
                    
                    if actor_name not in self.overlay_actor_names:
                        self.overlay_actor_names.append(actor_name)
                    self._category_actors.setdefault('surfaces', []).append(actor_name)
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        valid_inactive_lines = []
        
        for idx, surface_id in enumerate(inactive_surface_ids):
            if idx < len(inactive_points_list):
                try:
                    points = inactive_points_list[idx]
                    is_enabled = inactive_surface_enabled[idx] if idx < len(inactive_surface_enabled) else False
                    n_pts = len(points)
                    
                    if n_pts < 3:
                        continue
                    
                    valid_inactive_lines.append((points, surface_id))
                except Exception:
                    continue
        
        if valid_inactive_lines:
            try:
                all_line_points = []
                all_line_arrays = []
                point_offset = 0
                
                for points, surface_id in valid_inactive_lines:
                    n_pts = len(points)
                    all_line_points.append(points)
                    line_array = [n_pts] + [point_offset + i for i in range(n_pts)]
                    all_line_arrays.extend(line_array)
                    point_offset += n_pts
                
                if all_line_points:
                    all_points = np.vstack(all_line_points)
                    polyline_mesh = self.pv.PolyData(all_points)
                    polyline_mesh.lines = all_line_arrays
                    try:
                        polyline_mesh.verts = np.empty(0, dtype=np.int64)
                    except Exception:
                        try:
                            polyline_mesh.verts = []
                        except Exception:
                            pass
                    
                    edge_actor_name = "surface_disabled_edges_batch"
                    edge_actor = self.plotter.add_mesh(
                        polyline_mesh,
                        name=edge_actor_name,
                        color='#000000',
                        # Schwarze Surface-Umrandungen (disabled) mit Zoom-Skalierung
                        line_width=self._get_scaled_line_width(0.7, apply_zoom=True),
                        opacity=1.0,
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        render_lines_as_tubes=False,
                    )
                    try:
                        if edge_actor is not None and hasattr(edge_actor, "SetPickable"):
                            edge_actor.SetPickable(False)
                    except Exception:
                        pass
                    
                    if edge_actor_name not in self.overlay_actor_names:
                        self.overlay_actor_names.append(edge_actor_name)
                    self._category_actors.setdefault('surfaces', []).append(edge_actor_name)
            except Exception:
                pass
        
        # #region agent log
        try:
            actor_final_check = None
            try:
                if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                    actor_final_check = self.plotter.renderer.actors.get('surface_disabled_polygons_batch')
            except Exception:
                pass
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E",
                    "location": "Plot3DOverlaysSurfaces.py:draw_surfaces:end",
                    "message": "draw_surfaces completed",
                    "data": {
                        "surface_disabled_polygons_batch_in_renderer": actor_final_check is not None,
                        "overlay_actor_names_count": len(self.overlay_actor_names) if isinstance(self.overlay_actor_names, list) else 0,
                        "category_actors_surfaces_count": len(self._category_actors.get('surfaces', [])) if isinstance(self._category_actors, dict) else 0
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        if DEBUG_OVERLAY_PERF and t_start is not None:
            t_total = (t_draw_end - t_start) * 1000 if t_draw_end else 0
            t_sig = (t_sig_end - t_sig_start) * 1000 if t_sig_end and t_sig_start else 0
            t_clear = (t_clear_end - t_clear_start) * 1000 if t_clear_end and t_clear_start else 0
            t_draw = (t_draw_end - t_draw_start) * 1000 if t_draw_end and t_draw_start else 0


__all__ = ['SPL3DOverlaySurfaces']

