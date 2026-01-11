"""Basisklasse für Overlay-Rendering mit gemeinsamer Infrastruktur."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
from Module_LFO.Modules_Calculate.Functions import FunctionToolbox


class SPL3DOverlayBase:
    """
    Basisklasse für Overlay-Rendering.
    
    Stellt gemeinsame Infrastruktur bereit:
    - Actor-Verwaltung (clear, clear_category, _add_overlay_mesh, _remove_actor)
    - Cache-Verwaltung
    - Gemeinsame Hilfsfunktionen
    """
    
    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert die Basisklasse mit Plotter und PyVista-Modul."""
        self.plotter = plotter
        self.pv = pv_module
        # Präfix pro Overlay-Modul, um Actor-Namens-Kollisionen zu vermeiden
        # (z. B. 'axis_', 'surf_', 'imp_', 'spk_')
        self._overlay_prefix: str = ""
        self.overlay_actor_names: List[str] = []
        self.overlay_counter = 0
        self._category_actors: dict[str, List[str]] = {}
        # Kleiner Z-Offset für alle planaren Overlays (Surfaces, Axis-Linien),
        # damit Umrandungen leicht ÜBER dem SPL-Plot liegen und nicht verdeckt werden.
        # Etwas größer gewählt, damit Umrandungen klar über Texturen liegen.
        self._planar_z_offset = 0.005
        # Z-Offset speziell für Achsenlinien (höher als Surfaces, damit sie beim Picking bevorzugt werden)
        self._axis_z_offset = 0.01  # 1cm über Surface
        # Cache für DPI-Skalierungsfaktor (wird bei Bedarf berechnet)
        self._dpi_scale_factor: Optional[float] = None
        # Referenz-Zoom-Level wird im Plotter gespeichert, damit alle Overlays die gleiche Referenz verwenden
        
    def clear(self) -> None:
        """Löscht alle Overlay-Actors."""
        for name in self.overlay_actor_names:
            try:
                self.plotter.remove_actor(name)
            except KeyError:
                pass
        self.overlay_actor_names.clear()
        self.overlay_counter = 0
        self._category_actors.clear()
        # Zurücksetzen der Zoom-Referenz beim vollständigen Löschen (z.B. beim Laden einer neuen Datei)
        # Die Referenz wird im Plotter gespeichert, damit alle Overlays die gleiche Referenz verwenden
        if hasattr(self.plotter, '_overlay_reference_zoom_level'):
            self.plotter._overlay_reference_zoom_level = None
    
    def clear_category(self, category: str) -> None:
        """Entfernt alle Actor einer Kategorie, ohne andere Overlays anzutasten."""
        actor_names = self._category_actors.pop(category, [])
        if not isinstance(actor_names, list):
            actor_names = list(actor_names) if actor_names else []
        
        if actor_names:
            print(f"[clear_category] Removing {len(actor_names)} actors from category '{category}': {actor_names}")
        
        removed_count = 0
        for name in actor_names:
            try:
                # Prüfe ob Actor existiert, bevor wir versuchen ihn zu entfernen
                actor_exists = name in self.plotter.renderer.actors if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors') else False
                if actor_exists:
                    self.plotter.remove_actor(name)
                    removed_count += 1
                    print(f"[clear_category] Removed actor '{name}' from plotter")
                else:
                    print(f"[clear_category] Actor '{name}' not found in plotter.renderer.actors (already removed?)")
            except KeyError:
                print(f"[clear_category] KeyError when removing actor '{name}' (not in plotter)")
            except Exception as e:
                print(f"[clear_category] Exception when removing actor '{name}': {e}")
            else:
                if isinstance(self.overlay_actor_names, list):
                    if name in self.overlay_actor_names:
                        self.overlay_actor_names.remove(name)
                else:
                    self.overlay_actor_names = list(self.overlay_actor_names) if self.overlay_actor_names else []
                    if name in self.overlay_actor_names:
                        self.overlay_actor_names.remove(name)
        
        if actor_names:
            print(f"[clear_category] Removed {removed_count}/{len(actor_names)} actors successfully")
    
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
        show_vertices: bool = False,
        line_pattern: Optional[int] = None,
        line_repeat: int = 1,
        category: str = 'generic',
        render_lines_as_tubes: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> str:
        """Fügt ein Mesh als Overlay hinzu und gibt den Actor-Namen zurück.
        
        Args:
            name: Optionaler Actor-Name. Wenn nicht angegeben, wird ein automatischer Name generiert.
        """
        if name is None:
            prefix = f"{self._overlay_prefix}" if self._overlay_prefix else ""
            name = f"{prefix}overlay_{self.overlay_counter}"
            self.overlay_counter += 1
        else:
            # 🎯 FIX: Wenn ein expliziter Name angegeben wird, entferne den Actor, falls er bereits existiert
            # Dies stellt sicher, dass Actors nicht überschrieben werden, wenn Arrays unhide gesetzt werden
            if hasattr(self, 'plotter') and self.plotter is not None:
                try:
                    existing_actor = self.plotter.renderer.actors.get(name)
                    if existing_actor is not None:
                        print(f"[_add_overlay_mesh] Found existing actor '{name}' in plotter, removing before adding new one")
                        # Prüfe line_width des existierenden Actors vor dem Entfernen (für Debug)
                        if category == 'axis' and hasattr(existing_actor, 'GetProperty'):
                            try:
                                old_prop = existing_actor.GetProperty()
                                if old_prop and hasattr(old_prop, 'GetLineWidth'):
                                    old_width = old_prop.GetLineWidth()
                                    print(f"[_add_overlay_mesh] Existing actor '{name}' had line_width={old_width:.2f} before removal")
                            except Exception:
                                pass
                        # Entferne den existierenden Actor, damit er nicht überschrieben wird
                        self.plotter.remove_actor(name)
                        # Prüfe ob Actor wirklich entfernt wurde
                        if name in self.plotter.renderer.actors:
                            print(f"[_add_overlay_mesh] WARNING: Actor '{name}' still exists in plotter after remove_actor()!")
                        else:
                            print(f"[_add_overlay_mesh] Actor '{name}' successfully removed from plotter")
                        # Entferne auch aus der Liste der Actor-Namen
                        if hasattr(self, 'overlay_actor_names') and isinstance(self.overlay_actor_names, list):
                            if name in self.overlay_actor_names:
                                self.overlay_actor_names.remove(name)
                        if hasattr(self, '_category_actors') and isinstance(self._category_actors, dict):
                            category_actors = self._category_actors.get(category, [])
                            if isinstance(category_actors, list) and name in category_actors:
                                category_actors.remove(name)
                except Exception as e:
                    # Ignoriere Fehler beim Entfernen (Actor existiert möglicherweise nicht)
                    print(f"[_add_overlay_mesh] Exception while removing existing actor '{name}': {e}")
                    pass
        
        # 🎯 FIX: Für axis category entfernen wir line_width aus kwargs, damit wir volle Kontrolle über SetLineWidth() haben
        # Dies verhindert, dass PyVista die line_width bereits beim add_mesh setzt und dann möglicherweise akkumuliert wird
        kwargs = {
            'name': name,
            'opacity': opacity,
            'smooth_shading': False,
            'show_scalar_bar': False,
            'reset_camera': False,
        }
        # Nur line_width in kwargs setzen, wenn es NICHT die axis category ist
        # Für axis category wird line_width explizit via SetLineWidth() gesetzt
        if category != 'axis':
            kwargs['line_width'] = line_width

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

        actor = self.plotter.add_mesh(mesh, **kwargs)
        print(f"[_add_overlay_mesh] Added mesh actor '{name}', category='{category}', line_width={line_width:.2f}")
        
        # Erhöhe Picking-Priorität für Achsenlinien
        if category == 'axis':
            print(f"[_add_overlay_mesh] Entering axis category block for '{name}'")
            try:
                print(f"[_add_overlay_mesh] Actor has SetPickable: {hasattr(actor, 'SetPickable')}")
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(True)
                print(f"[_add_overlay_mesh] Actor has GetProperty: {hasattr(actor, 'GetProperty')}")
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    print(f"[_add_overlay_mesh] GetProperty() returned: {prop} (type: {type(prop)})")
                    if prop:
                        prop.SetOpacity(1.0)
                        # 🎯 WICHTIG: Setze render_lines_as_tubes EXPLIZIT auf False, bevor SetLineWidth() aufgerufen wird
                        # Dies stellt sicher, dass die line_width korrekt interpretiert wird
                        if render_lines_as_tubes is False:
                            if hasattr(prop, 'SetRenderLinesAsTubes'):
                                prop.SetRenderLinesAsTubes(False)
                        actual_width_before_print = None
                        if hasattr(prop, 'GetLineWidth'):
                            try:
                                actual_width_before_print = prop.GetLineWidth()
                            except Exception:
                                pass
                        print(f"[_add_overlay_mesh] Setting line_width for '{name}': requested={line_width:.2f}, before={actual_width_before_print:.2f if actual_width_before_print is not None else 'N/A'}, render_lines_as_tubes={render_lines_as_tubes}")
                        prop.SetLineWidth(line_width)
                        actual_width_after_print = None
                        if hasattr(prop, 'GetLineWidth'):
                            try:
                                actual_width_after_print = prop.GetLineWidth()
                            except Exception:
                                pass
                        print(f"[_add_overlay_mesh] LineWidth after SetLineWidth for '{name}': {actual_width_after_print:.2f if actual_width_after_print is not None else 'N/A'} (requested: {line_width:.2f})")
                        # 🎯 FIX: Setze auch das Line-Pattern hier, falls vorhanden
                        # Das verhindert, dass das Standard-Pattern (0xFFFF) verwendet wird
                        if line_pattern is not None:
                            prop.SetRenderLinesAsTubes(False)
                            prop.SetLineStipplePattern(int(line_pattern))
                            prop.SetLineStippleRepeatFactor(max(1, int(line_repeat)))
                        
                        # 🎯 WICHTIG: Deaktiviere Punkte explizit für axis category
                        # Dies verhindert, dass Punkte an Segmentenden beim zweiten Durchlauf sichtbar werden
                        if hasattr(prop, 'render_points_as_spheres'):
                            prop.render_points_as_spheres = False
                        if hasattr(prop, 'point_size'):
                            prop.point_size = 0
                        # Alternative Methoden, falls die oben genannten nicht verfügbar sind
                        try:
                            if hasattr(prop, 'SetPointSize'):
                                prop.SetPointSize(0)
                            if hasattr(prop, 'SetRenderPointsAsSpheres'):
                                prop.SetRenderPointsAsSpheres(False)
                        except Exception:
                            pass
                        print(f"[_add_overlay_mesh] Deactivated points for axis category '{name}' (point_size=0)")
                        
                actor.Modified()
            except Exception:  # noqa: BLE001
                pass
        
        # Achsenflächen sollen für Mausereignisse transparent sein
        if category == 'axis_plane':
            try:
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(False)
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if prop and hasattr(prop, 'PickableOff'):
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
        
        # Stelle sicher, dass Vertices nicht angezeigt werden
        if not show_vertices and hasattr(actor, 'prop') and actor.prop is not None:
            try:
                actor.prop.render_points_as_spheres = False
                actor.prop.point_size = 0
            except Exception:  # noqa: BLE001
                pass
        
        # Line-Pattern nur bei echten Polylines anwenden
        is_tube_mesh = render_lines_as_tubes is None
        
        if line_pattern is not None and not is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
            try:
                actor.prop.SetRenderLinesAsTubes(False)
                actor.prop.SetLineWidth(line_width)
                pattern_set = int(line_pattern)
                actor.prop.SetLineStipplePattern(pattern_set)
                actor.prop.SetLineStippleRepeatFactor(max(1, int(line_repeat)))
                # 🎯 WICHTIG: Actor.Modified() aufrufen, damit das Pattern gerendert wird
                actor.Modified()
            except Exception:  # noqa: BLE001
                pass
        elif is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
            try:
                actor.prop.SetLineStipplePattern(0xFFFF)  # Durchgezogen
                actor.prop.SetLineStippleRepeatFactor(1)
                # Bei Tubes wird line_width als Tube-Radius interpretiert - stelle sicher, dass es gesetzt ist
                if category == 'axis':
                    # Für Achsenlinien wird line_width bereits oben gesetzt, aber stellen wir sicher
                    pass  # Wird bereits in Zeile 137 gesetzt
            except Exception:  # noqa: BLE001
                pass
        
        self.overlay_actor_names.append(name)
        self._category_actors.setdefault(category, []).append(name)
        
        return name

    def _add_axis_line_mesh(
        self,
        mesh: Any,
        *,
        color: str,
        line_width: float,
        name: str,
    ) -> str:
        """Spezielle Methode zum Hinzufügen von Axis-Linien (XY-Achsen).
        
        Diese Methode bietet vollständige Kontrolle über das Rendering von Axis-Linien
        und stellt sicher, dass die line_width korrekt gesetzt wird, ohne Akkumulation.
        
        Args:
            mesh: PyVista PolyData Mesh mit Linien
            color: Farbe der Linien (z.B. 'red', 'black')
            line_width: Liniendicke (wird direkt via SetLineWidth() gesetzt)
            name: Actor-Name (muss eindeutig sein)
        
        Returns:
            Actor-Name
        """
        # Entferne existierenden Actor, falls vorhanden
        if hasattr(self, 'plotter') and self.plotter is not None:
            try:
                existing_actor = self.plotter.renderer.actors.get(name)
                if existing_actor is not None:
                    print(f"[_add_axis_line_mesh] Found existing actor '{name}', removing before adding new one")
                    # Prüfe line_width des existierenden Actors vor dem Entfernen (für Debug)
                    if hasattr(existing_actor, 'GetProperty'):
                        try:
                            old_prop = existing_actor.GetProperty()
                            if old_prop and hasattr(old_prop, 'GetLineWidth'):
                                old_width = old_prop.GetLineWidth()
                                print(f"[_add_axis_line_mesh] Existing actor '{name}' had line_width={old_width:.2f} before removal")
                        except Exception:
                            pass
                    # Entferne den existierenden Actor
                    self.plotter.remove_actor(name)
                    # Prüfe ob Actor wirklich entfernt wurde
                    if name in self.plotter.renderer.actors:
                        print(f"[_add_axis_line_mesh] WARNING: Actor '{name}' still exists in plotter after remove_actor()!")
                    else:
                        print(f"[_add_axis_line_mesh] Actor '{name}' successfully removed from plotter")
                    # Entferne auch aus der Verwaltung
                    if hasattr(self, 'overlay_actor_names') and isinstance(self.overlay_actor_names, list):
                        if name in self.overlay_actor_names:
                            self.overlay_actor_names.remove(name)
                    if hasattr(self, '_category_actors') and isinstance(self._category_actors, dict):
                        category_actors = self._category_actors.get('axis', [])
                        if isinstance(category_actors, list) and name in category_actors:
                            category_actors.remove(name)
            except Exception as e:
                print(f"[_add_axis_line_mesh] Exception while removing existing actor '{name}': {e}")
        
        # Erstelle kwargs OHNE line_width - wir setzen es explizit später
        kwargs = {
            'name': name,
            'color': color,
            'opacity': 1.0,
            'smooth_shading': False,
            'show_scalar_bar': False,
            'reset_camera': False,
            'render_lines_as_tubes': False,  # WICHTIG: Explizit False für Linien
            'render_points_as_spheres': False,
            'point_size': 0,
        }
        
        # Füge Mesh zum Plotter hinzu
        actor = self.plotter.add_mesh(mesh, **kwargs)
        print(f"[_add_axis_line_mesh] Added mesh actor '{name}', line_width will be set to {line_width:.2f}")
        
        # Setze alle Eigenschaften für Axis-Linien explizit
        try:
            # Setze Picking-Priorität
            if hasattr(actor, 'SetPickable'):
                actor.SetPickable(True)
            
            # Hole Property-Objekt
            if hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if prop:
                    # Setze Opacity
                    prop.SetOpacity(1.0)
                    
                    # 🎯 WICHTIG: Setze render_lines_as_tubes EXPLIZIT auf False, bevor SetLineWidth()
                    if hasattr(prop, 'SetRenderLinesAsTubes'):
                        prop.SetRenderLinesAsTubes(False)
                        print(f"[_add_axis_line_mesh] Set render_lines_as_tubes=False for '{name}'")
                    
                    # Prüfe line_width vor dem Setzen (für Debug)
                    actual_width_before = None
                    if hasattr(prop, 'GetLineWidth'):
                        try:
                            actual_width_before = prop.GetLineWidth()
                        except Exception:
                            pass
                    
                    before_str = f"{actual_width_before:.2f}" if actual_width_before is not None else 'N/A'
                    print(f"[_add_axis_line_mesh] Setting line_width for '{name}': requested={line_width:.2f}, before={before_str}")
                    
                    # Setze line_width explizit
                    if hasattr(prop, 'SetLineWidth'):
                        prop.SetLineWidth(line_width)
                    
                    # Prüfe line_width nach dem Setzen (für Debug)
                    actual_width_after = None
                    if hasattr(prop, 'GetLineWidth'):
                        try:
                            actual_width_after = prop.GetLineWidth()
                        except Exception:
                            pass
                    
                    after_str = f"{actual_width_after:.2f}" if actual_width_after is not None else 'N/A'
                    print(f"[_add_axis_line_mesh] LineWidth after SetLineWidth for '{name}': {after_str} (requested: {line_width:.2f})")
                    
                    # 🎯 WICHTIG: Deaktiviere Punkte explizit am Property-Objekt
                    # Dies verhindert, dass Punkte an Segmentenden beim zweiten Durchlauf sichtbar werden
                    if hasattr(prop, 'render_points_as_spheres'):
                        prop.render_points_as_spheres = False
                    if hasattr(prop, 'point_size'):
                        prop.point_size = 0
                    # Alternative Methoden, falls die oben genannten nicht verfügbar sind
                    try:
                        if hasattr(prop, 'SetPointSize'):
                            prop.SetPointSize(0)
                        if hasattr(prop, 'SetRenderPointsAsSpheres'):
                            prop.SetRenderPointsAsSpheres(False)
                    except Exception:
                        pass
                    print(f"[_add_axis_line_mesh] Deactivated points for '{name}' (point_size=0, render_points_as_spheres=False)")
                    
                    
                    # Markiere Actor als geändert
                    actor.Modified()
            
        except Exception as e:  # noqa: BLE001
            print(f"[_add_axis_line_mesh] Exception while setting properties for '{name}': {e}")
        
        # Füge zur Verwaltung hinzu
        if not hasattr(self, 'overlay_actor_names') or not isinstance(self.overlay_actor_names, list):
            self.overlay_actor_names = []
        self.overlay_actor_names.append(name)
        self._category_actors.setdefault('axis', []).append(name)
        
        return name

    def _remove_actor(self, name: str) -> None:
        """Entfernt einen Actor aus dem Plotter."""
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
        """Aktualisiert einen existierenden Speaker-Actor mit neuem Mesh und Farben."""
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
        
        # Aktualisiere das Mesh des Actors
        if hasattr(actor, 'mapper') and actor.mapper is not None:
            actor.mapper.dataset = mesh
            actor.mapper.update()
        actor.Modified()
    
    @staticmethod
    def _to_float_array(values) -> np.ndarray:
        """Konvertiert Werte zu einem 1D-Float-Array."""
        if values is None:
            return np.empty(0, dtype=float)
        try:
            array = np.asarray(values, dtype=float)
        except Exception:
            return np.empty(0, dtype=float)
        if array.ndim > 1:
            array = array.reshape(-1)
        return array.astype(float)
    
    def _get_max_surface_dimension(self, settings) -> float:
        """Berechnet die maximale Dimension (Breite oder Länge) aller nicht-versteckten Surfaces."""
        surface_store = getattr(settings, 'surface_definitions', {})
        if not isinstance(surface_store, dict):
            return 0.0
        
        max_dimension = 0.0
        
        for surface_id, surface in surface_store.items():
            if isinstance(surface, SurfaceDefinition):
                hidden = surface.hidden
                points = surface.points
            else:
                hidden = surface.get('hidden', False)
                points = surface.get('points', [])
            
            if hidden:
                continue
            
            if points:
                try:
                    width, length = FunctionToolbox.surface_dimensions(points)
                    max_dim = max(float(width), float(length))
                    max_dimension = max(max_dimension, max_dim)
                except Exception:  # noqa: BLE001
                    continue
        
        return max_dimension * 1.5

    def _get_active_xy_surfaces(self, settings) -> List[Tuple[str, Any]]:
        """Sammelt alle aktiven Surfaces für XY-Berechnung (xy_enabled=True, enabled=True, hidden=False)."""
        active_surfaces = []
        surface_store = getattr(settings, 'surface_definitions', {})
        
        if not isinstance(surface_store, dict):
            return active_surfaces
        
        for surface_id, surface in surface_store.items():
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
    
    def _get_active_xy_surfaces_for_axis_lines(self, settings) -> List[Tuple[str, Any]]:
        """Sammelt alle aktiven Surfaces für Axis-Linien (xy_enabled=True, hidden=False).
        
        Diese Methode filtert NUR nach xy_enabled und hidden, NICHT nach enabled.
        Das ermöglicht, dass Axis-Linien auch auf disabled Surfaces gezeichnet werden können.
        """
        active_surfaces = []
        surface_store = getattr(settings, 'surface_definitions', {})
        
        if not isinstance(surface_store, dict):
            return active_surfaces
        
        for surface_id, surface in surface_store.items():
            if isinstance(surface, SurfaceDefinition):
                xy_enabled = getattr(surface, 'xy_enabled', True)
                hidden = surface.hidden
            else:
                xy_enabled = surface.get('xy_enabled', True)
                hidden = surface.get('hidden', False)
            
            # 🎯 NEU: Nur xy_enabled und hidden prüfen, nicht enabled
            # Damit können Axis-Linien auch auf disabled Surfaces gezeichnet werden
            if xy_enabled and not hidden:
                active_surfaces.append((str(surface_id), surface))
        
        return active_surfaces
    
    def _get_dpi_scale_factor(self) -> float:
        """Berechnet den DPI-Skalierungsfaktor für Linienbreiten.
        
        VTK/PyVista interpretiert SetLineWidth() in Pixeln, daher müssen wir
        bei höherer Bildschirmauflösung (z.B. Retina-Displays) die Linienbreiten
        entsprechend skalieren, damit sie visuell gleich dick bleiben.
        
        Returns:
            Skalierungsfaktor (1.0 für Standard-DPI, < 1.0 für höhere DPI)
        """
        if self._dpi_scale_factor is not None:
            return self._dpi_scale_factor
        
        try:
            # Versuche, den devicePixelRatio vom QtInteractor-Widget zu ermitteln
            # Dies ist die zuverlässigste Methode für PyQt5/PyVista
            if hasattr(self.plotter, 'interactor'):
                widget = self.plotter.interactor
                if widget is not None:
                    if hasattr(widget, 'devicePixelRatio'):
                        try:
                            device_pixel_ratio = widget.devicePixelRatio()
                            if device_pixel_ratio > 1.0:
                                # Skaliere die Linienbreite umgekehrt zum Pixel-Ratio
                                # Bei 2x Retina: line_width wird halbiert, damit visuell gleich dick
                                self._dpi_scale_factor = 1.0 / device_pixel_ratio
                                return self._dpi_scale_factor
                        except Exception:
                            pass  # Fallback zu devicePixelRatioF oder Standard-Wert
                    # Alternative: Versuche devicePixelRatioF() für Float-Werte
                    if hasattr(widget, 'devicePixelRatioF'):
                        try:
                            device_pixel_ratio = widget.devicePixelRatioF()
                            if device_pixel_ratio > 1.0:
                                self._dpi_scale_factor = 1.0 / device_pixel_ratio
                                return self._dpi_scale_factor
                        except Exception:
                            pass  # Fallback zu Standard-Wert
        except Exception:
            pass  # Fallback zu Standard-Wert
        
        # 🎯 FALLBACK: Wenn DPI-Skalierung nicht bestimmt werden kann, verwende Standard-Wert 1.0
        # Dies verhindert Abstürze und ermöglicht die Verwendung der Anwendung auch ohne DPI-Erkennung
        self._dpi_scale_factor = 1.0
        return self._dpi_scale_factor
    
    def _reset_zoom_reference(self) -> None:
        """Setzt die Referenz-Zoom-Level zurück, damit sie beim nächsten Aufruf neu gesetzt wird."""
        self._reference_zoom_level = None
    
    def _get_zoom_scale_factor(self) -> float:
        """Berechnet den Zoom-Skalierungsfaktor basierend auf der Kamera.
        
        Die Linienstärke soll proportional zum Zoom skaliert werden:
        - Weiter herausgezoomt (größerer parallel_scale) → dünnere Linien
        - Weiter hineingezoomt (kleinerer parallel_scale) → dickere Linien
        
        Returns:
            Zoom-Skalierungsfaktor (1.0 bei Referenz-Zoom, < 1.0 bei herausgezoomt, > 1.0 bei hineingezoomt)
        """
        try:
            if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
                return 1.0
            
            cam = self.plotter.camera
            
            # Berechne aktuellen Zoom-Level
            current_zoom = None
            if hasattr(cam, 'parallel_projection') and getattr(cam, 'parallel_projection', False):
                # Parallelprojektion: parallel_scale bestimmt den Zoom
                if hasattr(cam, 'parallel_scale'):
                    try:
                        current_zoom = float(cam.parallel_scale)
                    except Exception:
                        pass
            else:
                # Perspektivprojektion: verwende view_angle als Proxy
                # Größerer Winkel = weiter herausgezoomt
                if hasattr(cam, 'view_angle'):
                    try:
                        current_zoom = float(cam.view_angle)
                    except Exception:
                        pass
            
            if current_zoom is None or current_zoom <= 0:
                return 1.0
            
            # 🎯 WICHTIG: Referenz-Zoom-Level wird im Plotter gespeichert, damit alle Overlays
            # die gleiche Referenz verwenden (verhindert unterschiedliche Referenzen zwischen Overlays)
            if not hasattr(self.plotter, '_overlay_reference_zoom_level'):
                self.plotter._overlay_reference_zoom_level = None
            
            reference_zoom = self.plotter._overlay_reference_zoom_level
            
            # Setze Referenz-Zoom beim ersten Aufruf (oder wenn noch nicht gesetzt)
            if reference_zoom is None or reference_zoom <= 0:
                self.plotter._overlay_reference_zoom_level = current_zoom
                return 1.0
            
            # 🎯 WICHTIG: Wenn sich der Zoom-Level deutlich ändert (Faktor > 2 oder < 0.5),
            # aktualisiere die Referenz. Das passiert typischerweise, wenn die Kamera
            # nach dem ersten Zeichnen initial gezoomt wurde (z.B. in _zoom_to_default_surface).
            # Dies verhindert, dass die Referenz auf einen falschen Wert gesetzt wird.
            zoom_ratio = current_zoom / reference_zoom if reference_zoom > 0 else 1.0
            if zoom_ratio > 2.0 or zoom_ratio < 0.5:
                # Zoom-Level hat sich deutlich geändert - aktualisiere Referenz
                self.plotter._overlay_reference_zoom_level = current_zoom
                return 1.0
            
            # Berechne Skalierungsfaktor: reference / current
            # Wenn current_zoom größer ist (herausgezoomt), wird der Faktor < 1.0
            # Wenn current_zoom kleiner ist (hineingezoomt), wird der Faktor > 1.0
            raw_factor = reference_zoom / current_zoom
            
            # Verwende Wurzel-Skalierung für sanftere Anpassung
            # Dies reduziert die Aggressivität der Zoom-Skalierung
            # z.B. bei 4x Zoom-Änderung: sqrt(4) = 2x statt 4x Skalierung
            zoom_factor = np.sqrt(raw_factor) if raw_factor > 0 else 1.0
            
            # Begrenze den Faktor auf einen sinnvollen Bereich (0.3 bis 3.0)
            zoom_factor = max(0.3, min(3.0, zoom_factor))
            
            # Optionale Debug-Ausgabe der Zoom-Berechnung (deaktiviert im Normalbetrieb)
            # print(f"[DEBUG ZOOM] current_zoom={current_zoom:.3f}, reference_zoom={reference_zoom:.3f}, "
            #       f"raw_factor={raw_factor:.3f}, zoom_factor={zoom_factor:.3f}")
            
            return zoom_factor
        except Exception:
            # Bei Fehler: keine Zoom-Skalierung
            return 1.0
    
    def _get_scaled_line_width(self, base_line_width: float, apply_zoom: bool = True) -> float:
        """Skaliert eine Linienbreite basierend auf der Bildschirmauflösung und optional dem Zoom-Level.
        
        Args:
            base_line_width: Basis-Linienbreite in Pixeln (für Standard-DPI und Referenz-Zoom)
            apply_zoom: Wenn True, wird auch die Zoom-Skalierung angewendet. Wenn False, nur DPI-Skalierung.
                       Standard ist True, aber für Achsenlinien sollte False verwendet werden, da diese
                       als geometrische Segmente gerendert werden.
        
        Returns:
            Skalierte Linienbreite, die bei höherer Auflösung visuell gleich dick bleibt
            und optional proportional zum Zoom-Level skaliert wird
        """
        dpi_factor = self._get_dpi_scale_factor()
        if apply_zoom:
            zoom_factor = self._get_zoom_scale_factor()
            return base_line_width * dpi_factor * zoom_factor
        else:
            return base_line_width * dpi_factor


__all__ = ['SPL3DOverlayBase']

