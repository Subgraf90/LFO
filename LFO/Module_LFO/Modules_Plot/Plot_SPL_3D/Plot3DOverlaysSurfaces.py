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
        
        print(f"[DEBUG draw_surfaces] active_ids_set = {sorted(active_ids_set)}")
        print(f"[DEBUG draw_surfaces] highlight_ids_tuple = {highlight_ids_tuple}")
        
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
                print(f"[DEBUG draw_surfaces] Signature comparison:")
                print(f"  last_ids_set: {sorted(last_ids_set)}")
                print(f"  active_ids_set: {sorted(active_ids_set)}")
                print(f"  surfaces_changed: {surfaces_changed}")
                print(f"  highlights_changed: {highlights_changed}")
                print(f"  active_id_changed: {active_id_changed}")
                print(f"  spl_data_changed: {spl_data_changed}")
                if not surfaces_changed and not highlights_changed and not active_id_changed and not spl_data_changed:
                    signature_changed = False
                    print(f"[DEBUG draw_surfaces] SIGNATURE UNCHANGED - SKIPPING REDRAW")
            elif len(self._last_surfaces_state) == 2:
                # Alte Signatur-Struktur (2-Tupel) - behandle als geändert
                print(f"[DEBUG draw_surfaces] OLD SIGNATURE FORMAT - FORCING REDRAW")
            elif self._last_surfaces_state == signature_tuple:
                signature_changed = False
                print(f"[DEBUG draw_surfaces] SIGNATURE UNCHANGED (exact match) - SKIPPING REDRAW")
        
        # 🎯 WICHTIG: Auch wenn sich nur die Highlight-IDs geändert haben, müssen wir neu zeichnen
        # (damit disabled Surfaces rot umrandet werden, wenn sie angeklickt werden)
        if not signature_changed and not create_empty_plot_surfaces:
            print(f"[DEBUG draw_surfaces] RETURNING EARLY - no redraw needed")
            return
        
        print(f"[DEBUG draw_surfaces] PROCEEDING WITH REDRAW - signature_changed={signature_changed}")
        
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
        
        for surface_id, surface_def in surface_definitions.items():
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
                surface_obj = surface_def
            else:
                enabled = surface_def.get('enabled', False)
                hidden = surface_def.get('hidden', False)
                points = surface_def.get('points', [])
                # Erstelle SurfaceDefinition-Objekt für Validierung
                try:
                    surface_obj = SurfaceDefinition.from_dict(str(surface_id), surface_def)
                except Exception:
                    surface_obj = None
            
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
                                print(f"[DEBUG draw_surfaces] Surface '{surface_id}' mit {len(points)} Punkten: Triangulation fehlgeschlagen (keine Dreiecke)")
                        except Exception as e:
                            # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                            is_valid_for_spl = False
                            print(f"[DEBUG draw_surfaces] Surface '{surface_id}' mit {len(points)} Punkten: Triangulation fehlgeschlagen ({e})")
                except Exception:
                    # Bei Fehler in Validierung: als ungültig behandeln
                    is_valid_for_spl = False
            
            if enabled:
                enabled_count += 1
            else:
                disabled_count += 1
            
            is_active = (str(surface_id) in active_ids_set)
            if is_active:
                print(f"[DEBUG draw_surfaces] Surface {surface_id} is ACTIVE (enabled={enabled}, hidden={hidden}, valid_for_spl={is_valid_for_spl})")
            
            try:
                point_coords = []
                for point in points:
                    x = float(point.get('x', 0.0))
                    y = float(point.get('y', 0.0))
                    z = float(point.get('z', 0.0)) + z_offset
                    point_coords.append([x, y, z])
                
                if len(point_coords) < 3:
                    continue
                
                # 🎯 DEBUG: Zeige Z-Koordinaten für ungültige Surfaces
                if enabled and not is_valid_for_spl:
                    z_values = [p[2] - z_offset for p in point_coords]
                    print(f"[DEBUG draw_surfaces] Ungültige Surface '{surface_id}': Z-Werte = {z_values}")
                
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
                
                # 🎯 DEBUG: Zeige Koordinaten-Array für ungültige Surfaces
                if enabled and not is_valid_for_spl:
                    print(f"[DEBUG draw_surfaces] Ungültige Surface '{surface_id}': closed_coords_array shape = {closed_coords_array.shape}")
                    print(f"[DEBUG draw_surfaces] Ungültige Surface '{surface_id}': Z-Min = {closed_coords_array[:, 2].min():.2f}, Z-Max = {closed_coords_array[:, 2].max():.2f}")
                
                if not enabled:
                    disabled_surface_ids.append(str(surface_id))
                    disabled_surface_points.append(closed_coords_array)
                    
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
            print(f"[DEBUG draw_surfaces] Drawing {len(active_enabled_points_list)} active ENABLED surfaces with RED border")
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
                    line_width=3.5,
                    opacity=1.0,
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=True,
                )
                print(f"[DEBUG draw_surfaces] Successfully added RED border for active ENABLED surfaces")
            except Exception as e:
                print(f"[DEBUG draw_surfaces] ERROR drawing active ENABLED surfaces: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG draw_surfaces] No active ENABLED surfaces to draw (active_enabled_points_list is empty)")
        
        # Zeichne aktive DISABLED Surfaces als Batch
        if active_disabled_points_list:
            print(f"[DEBUG draw_surfaces] Drawing {len(active_disabled_points_list)} active DISABLED surfaces with RED border")
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
                    line_width=3.5,
                    opacity=1.0,
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=True,
                )
                print(f"[DEBUG draw_surfaces] Successfully added RED border for active DISABLED surfaces")
            except Exception as e:
                print(f"[DEBUG draw_surfaces] ERROR drawing active DISABLED surfaces: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG draw_surfaces] No active DISABLED surfaces to draw (active_disabled_points_list is empty)")
        
        # Zeichne Fläche für enabled Surfaces NUR wenn create_empty_plot_surfaces=True
        if create_empty_plot_surfaces:
            enabled_points_for_empty_plot = []
            enabled_faces_for_empty_plot = []
            point_offset = 0
            
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
                    
                    triangulation_success = False
                    if triangles and len(triangles) > 0:
                        # Triangulation erfolgreich: Verwende Dreiecke
                        print(f"[DEBUG draw_surfaces] Trianguliere ungültige Surface: {len(triangles)} Dreiecke")
                        for tri in triangles:
                            # Finde Indizes der Punkte im points_array
                            tri_indices = []
                            for tri_pt in tri:
                                # Suche passenden Punkt im points_array
                                found = False
                                for idx, pt in enumerate(points_array):
                                    if (abs(pt[0] - tri_pt['x']) < tolerance and
                                        abs(pt[1] - tri_pt['y']) < tolerance and
                                        abs(pt[2] - (tri_pt['z'] + z_offset)) < tolerance):
                                        tri_indices.append(point_offset + idx)
                                        found = True
                                        break
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
                    print(f"[DEBUG draw_surfaces] Ungültige enabled Surfaces gezeichnet: {len(invalid_enabled_points_list)} Flächen, {len(all_invalid_faces) // 4 if all_invalid_faces else 0} Faces")
            except Exception as e:
                print(f"[DEBUG draw_surfaces] ERROR drawing invalid enabled surfaces: {e}")
                import traceback
                traceback.print_exc()
        
        old_inactive_actors = [
            name for name in self.overlay_actor_names
            if name in ("surface_disabled_polygons_batch", "surface_disabled_edges_batch")
        ]
        for name in old_inactive_actors:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
            if name in self.overlay_actor_names:
                self.overlay_actor_names.remove(name)
            if 'surfaces' in self._category_actors and name in self._category_actors['surfaces']:
                self._category_actors['surfaces'].remove(name)
        
        if has_spl_data:
            try:
                disabled_polygons_actor = self.plotter.renderer.actors.get('surface_disabled_polygons_batch')
                if disabled_polygons_actor is not None:
                    pass
            except Exception:
                pass
        
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
                except Exception:
                    continue
        
        if valid_disabled_polygons:
            try:
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
                    polygon_mesh = self.pv.PolyData(all_points)
                    polygon_mesh.faces = all_polygon_faces
                    
                    actor_name = "surface_disabled_polygons_batch"
                    actor = self.plotter.add_mesh(
                        polygon_mesh,
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
                pass
        
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
                        line_width=1.5,
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
        
        t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        if DEBUG_OVERLAY_PERF and t_start is not None:
            t_total = (t_draw_end - t_start) * 1000 if t_draw_end else 0
            t_sig = (t_sig_end - t_sig_start) * 1000 if t_sig_end and t_sig_start else 0
            t_clear = (t_clear_end - t_clear_start) * 1000 if t_clear_end and t_clear_start else 0
            t_draw = (t_draw_end - t_draw_start) * 1000 if t_draw_end and t_draw_start else 0


__all__ = ['SPL3DOverlaySurfaces']

