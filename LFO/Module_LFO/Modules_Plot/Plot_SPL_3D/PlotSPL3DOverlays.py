from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PyQt5 import QtWidgets, QtCore

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    DEBUG_SURFACE_GEOMETRY,
    SurfaceDefinition,
)
from Module_LFO.Modules_Calculate.Functions import FunctionToolbox
from Module_LFO.Modules_Init.Logging import perf_section
from Module_LFO.Modules_Plot.Plot_SPL_3D.PlotSPL3DSpeaker import SPL3DSpeakerMixin

DEBUG_OVERLAY_PERF = bool(int(os.environ.get("LFO_DEBUG_OVERLAY_PERF", "1")))


class SPL3DOverlayRenderer(SPL3DSpeakerMixin):
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
        # Kleiner Z-Offset für alle planaren Overlays (Surfaces, Axis-Linien),
        # damit Umrandungen leicht ÜBER dem SPL-Plot liegen und nicht verdeckt werden.
        # Wert bewusst klein halten, damit perspektivisch nichts "schwebt".
        self._planar_z_offset = 0.0
        # Z-Offset speziell für Achsenlinien (höher als Surfaces, damit sie beim Picking bevorzugt werden)
        self._axis_z_offset = 0.01  # Erhöht von 0.001 auf 0.01 für besseres Picking (1cm über Surface)
        self._speaker_actor_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
        self._speaker_geometry_cache: dict[str, List[Tuple[Any, Optional[int]]]] = {}  # Cache für transformierte Geometrien
        self._speaker_geometry_param_cache: dict[tuple[str, int], tuple] = {}  # Cache für Geometrie-Parameter-Signaturen
        # Persistenter Array-Cache auf Overlay-Ebene:
        # Key: (array_id, speaker_index)
        # Value: {
        #   'actor_entry': Cache-Eintrag wie in _speaker_actor_cache (mesh, actor, actor_obj, signature),
        #   'geometry_param_signature': Signatur der Geometrie-Parameter für diesen Speaker
        # }
        # Dieser Cache wird NICHT bei jedem draw_speakers()-Aufruf geleert,
        # sondern nur, wenn wirklich alle Arrays gelöscht werden (clear() / keine speaker_arrays).
        self._overlay_array_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._geometry_cache_max_size = 100  # Maximale Anzahl gecachter Geometrien
        # 🚀 OPTIMIERUNG: Array-Level Cache für Flown Arrays
        # Key: array_id, Value: dict mit 'geometries' (dict[sp_idx, geometries]), 'signature', 'array_pos'
        self._array_geometry_cache: dict[str, dict[str, Any]] = {}
        self._array_signature_cache: dict[str, tuple] = {}  # Cache für Array-Signaturen (Position, Rotation, etc.)
        # 🚀 OPTIMIERUNG: Stack-Level Cache für Stack Arrays
        # Key: (array_id, stack_group_key), Value: dict mit 'geometries', 'signature', 'stack_pos'
        self._stack_geometry_cache: dict[tuple[str, tuple], dict[str, Any]] = {}
        self._stack_signature_cache: dict[tuple[str, tuple], tuple] = {}  # Cache für Stack-Signaturen
        self._last_axis_state: Optional[tuple] = None
        self._last_impulse_state: Optional[tuple] = None
        self._last_surfaces_state: Optional[tuple] = None
        # Interner Objekt-Identifier (wird aktuell nicht geloggt, aber für spätere Verwendung behalten)
        self._debug_object_id = id(self)


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
        self._speaker_geometry_param_cache.clear()
        self._array_geometry_cache.clear()
        self._array_signature_cache.clear()
        self._stack_geometry_cache.clear()
        self._stack_signature_cache.clear()
        # ⚠️ WICHTIG:
        # Den persistenten Overlay-Array-Cache NICHT mehr in clear() leeren,
        # damit Speaker-Arrays zwischen Szenen-Updates inkrementell weiterverwendet werden können.
        # Ein kompletter Reset des Overlay-Array-Caches erfolgt nur noch in draw_speakers(),
        # wenn speaker_arrays wirklich leer ist.
        self._last_axis_state = None
        self._last_impulse_state = None

    def clear_category(self, category: str) -> None:
        """Entfernt alle Actor einer Kategorie, ohne andere Overlays anzutasten."""
        actor_names = self._category_actors.pop(category, [])
        # Stelle sicher, dass actor_names eine Liste ist (nicht ein Set)
        if not isinstance(actor_names, list):
            actor_names = list(actor_names) if actor_names else []
        
        for name in actor_names:
            try:
                self.plotter.remove_actor(name)
            except KeyError:
                pass
            else:
                # Stelle sicher, dass overlay_actor_names eine Liste ist
                if isinstance(self.overlay_actor_names, list):
                    if name in self.overlay_actor_names:
                        self.overlay_actor_names.remove(name)
                else:
                    # Falls es irgendwie zu einem Set wurde, konvertiere zurück zu Liste
                    self.overlay_actor_names = list(self.overlay_actor_names) if self.overlay_actor_names else []
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
    def draw_axis_lines(self, settings, selected_axis: Optional[str] = None) -> None:
        """Zeichnet X- und Y-Achsenlinien als Strich-Punkt-Linien auf aktiven Surfaces.
        
        Args:
            settings: Settings-Objekt
            selected_axis: 'x' oder 'y' für ausgewählte Achse (wird rot gezeichnet), None wenn keine ausgewählt
        """
        t_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Berechne maximale Surface-Dimension für Achsenflächen-Größe
        max_surface_dim = self._get_max_surface_dimension(settings)
        
        state = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
            float(getattr(settings, 'axis_3d_transparency', 10.0)),
            float(max_surface_dim),  # Maximale Surface-Dimension für Achsenflächen-Größe
            selected_axis,  # Highlight-Status in State aufnehmen
        )
        existing_names = self._category_actors.get('axis', [])
        if self._last_axis_state == state and existing_names:
            return

        t_clear_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        # Linien und evtl. Achsenflächen leeren
        self.clear_category('axis')
        self.clear_category('axis_plane')
        t_clear_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        x_axis, y_axis, length, width, axis_3d_transparency_from_state, max_surface_dim_from_state, selected_axis_from_state = state

        # Hole aktive Surfaces für XY-Berechnung (xy_enabled=True, enabled=True, hidden=False)
        t_surfaces_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        active_surfaces = self._get_active_xy_surfaces(settings)
        t_surfaces_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Strich-Punkt-Pattern (0xF4F4 = Strich-Punkt-Strich-Punkt, besser sichtbar als 0xF0F0)
        dash_dot_pattern = 0xF4F4
        
        # Zeichne X-Achsenlinie (y=y_axis, konstant) auf allen aktiven Surfaces
        t_x_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        x_lines_drawn = 0
        x_segments_total = 0
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_xz(y_axis, surface, settings)
            if intersection_points is not None:
                x_coords, z_coords = intersection_points
                if len(x_coords) >= 2:
                    # Erstelle 3D-Linie auf dem Surface
                    points_3d = np.column_stack([x_coords, np.full_like(x_coords, y_axis), z_coords])
                    # Sortiere nach X-Koordinaten für konsistente Reihenfolge
                    sort_idx = np.argsort(points_3d[:, 0])
                    points_3d = points_3d[sort_idx]
                    
                    # Füge kleinen Z-Offset hinzu, damit Achsenlinien beim Picking bevorzugt werden
                    points_3d[:, 2] += self._axis_z_offset
                    
                    # Erstelle Strich-Punkt-Linie durch Segmentierung (optimiert: größere Segmente, weniger Actors)
                    t_segment_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    dash_dot_segments = self._create_dash_dot_line_segments_optimized(points_3d, dash_length=1.0, dot_length=0.2, gap_length=0.3)
                    t_segment_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    
                    # Batch-Zeichnen: Kombiniere alle Segmente in einem Mesh
                    t_combine_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    if dash_dot_segments:
                        combined_mesh = self._combine_line_segments(dash_dot_segments)
                        t_combine_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        
                        t_draw_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        if combined_mesh is not None:
                            # X-Achse: rot wenn ausgewählt, sonst schwarz
                            line_color = 'red' if selected_axis == 'x' else 'black'
                            line_width = 6.0 if selected_axis == 'x' else 5.0
                            actor_name = self._add_overlay_mesh(
                                combined_mesh,
                                color=line_color,
                                line_width=line_width,
                                line_pattern=None,
                                line_repeat=1,
                                category='axis',
                                render_lines_as_tubes=True,  # Als Tubes rendern für besseres Picking
                            )
                            t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                            x_segments_total += len(dash_dot_segments)
                            t_seg = (t_segment_end - t_segment_start) * 1000 if t_segment_end and t_segment_start else 0
                            t_comb = (t_combine_end - t_combine_start) * 1000 if t_combine_end and t_combine_start else 0
                            t_dr = (t_draw_end - t_draw_start) * 1000 if t_draw_end and t_draw_start else 0
                        else:
                            t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    else:
                        t_combine_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    x_lines_drawn += 1
        t_x_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Zeichne Y-Achsenlinie (x=x_axis, konstant) auf allen aktiven Surfaces
        t_y_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        y_lines_drawn = 0
        y_segments_total = 0
        for surface_id, surface in active_surfaces:
            intersection_points = self._get_surface_intersection_points_yz(x_axis, surface, settings)
            if intersection_points is not None:
                y_coords, z_coords = intersection_points
                if len(y_coords) >= 2:
                    # Erstelle 3D-Linie auf dem Surface
                    points_3d = np.column_stack([np.full_like(y_coords, x_axis), y_coords, z_coords])
                    # Sortiere nach Y-Koordinaten für konsistente Reihenfolge
                    sort_idx = np.argsort(points_3d[:, 1])
                    points_3d = points_3d[sort_idx]
                    
                    # Füge kleinen Z-Offset hinzu, damit Achsenlinien beim Picking bevorzugt werden
                    points_3d[:, 2] += self._axis_z_offset
                    
                    # Erstelle Strich-Punkt-Linie durch Segmentierung (optimiert: größere Segmente, weniger Actors)
                    t_segment_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    dash_dot_segments = self._create_dash_dot_line_segments_optimized(points_3d, dash_length=1.0, dot_length=0.2, gap_length=0.3)
                    t_segment_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    
                    # Batch-Zeichnen: Kombiniere alle Segmente in einem Mesh
                    t_combine_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    if dash_dot_segments:
                        combined_mesh = self._combine_line_segments(dash_dot_segments)
                        t_combine_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        
                        t_draw_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        if combined_mesh is not None:
                            # Y-Achse: rot wenn ausgewählt, sonst schwarz
                            line_color = 'red' if selected_axis == 'y' else 'black'
                            line_width = 6.0 if selected_axis == 'y' else 5.0
                            actor_name = self._add_overlay_mesh(
                                combined_mesh,
                                color=line_color,
                                line_width=line_width,
                                line_pattern=None,
                                line_repeat=1,
                                category='axis',
                                render_lines_as_tubes=True,  # Als Tubes rendern für besseres Picking
                            )
                            t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                            y_segments_total += len(dash_dot_segments)
                            t_seg = (t_segment_end - t_segment_start) * 1000 if t_segment_end and t_segment_start else 0
                            t_comb = (t_combine_end - t_combine_start) * 1000 if t_combine_end and t_combine_start else 0
                            t_dr = (t_draw_end - t_draw_start) * 1000 if t_draw_end and t_draw_start else 0
                        else:
                            t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    else:
                        t_combine_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                        t_draw_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
                    y_lines_drawn += 1
        t_y_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        t_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        if DEBUG_OVERLAY_PERF and t_start is not None:
            t_total = (t_end - t_start) * 1000 if t_end else 0
            t_clear = (t_clear_end - t_clear_start) * 1000 if t_clear_end and t_clear_start else 0
            t_surfaces = (t_surfaces_end - t_surfaces_start) * 1000 if t_surfaces_end and t_surfaces_start else 0
            t_x = (t_x_end - t_x_start) * 1000 if t_x_end and t_x_start else 0
            t_y = (t_y_end - t_y_start) * 1000 if t_y_end and t_y_start else 0
        
        # 3D-Achsenfläche immer zeichnen
        try:
            self._draw_axis_planes(x_axis, y_axis, length, width, settings)
        except Exception:  # noqa: BLE001
            pass

        self._last_axis_state = state

    def _draw_axis_planes(self, x_axis: float, y_axis: float, length: float, width: float, settings) -> None:
        """Zeichnet halbtransparente, quadratische Flächen durch die X- und Y-Achse.
        
        - Größe: Basierend auf allen nicht-versteckten Surfaces: max(Breite, Länge) * 3.0
        - Orientierung:
          - X-Achse: Vertikale X-Z-Ebene bei y = y_axis.
          - Y-Achse: Vertikale Y-Z-Ebene bei x = x_axis.
        - Maus-Transparenz: Flächen werden als eigene Kategorie 'axis_plane' hinzugefügt
          und im Renderer explizit als nicht pickable markiert.
        """
        # Berechne Größe basierend auf allen nicht-versteckten Surfaces
        # _get_max_surface_dimension gibt bereits max_dimension * 1.5 zurück
        # Für Faktor 3 der max Dimension: L = max_dimension * 3 = (L_current / 1.5) * 3 = L_current * 2
        L_base = self._get_max_surface_dimension(settings)
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
        # Hier interpretieren wir den Wert als Opazität in Prozent
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
            grid_x = self.pv.StructuredGrid(X, Y, Z)
            # Kleiner Offset in Z, damit die Fläche minimal über dem Plot liegt
            points = grid_x.points
            points[:, 2] += self._planar_z_offset
            grid_x.points = points
            self._add_overlay_mesh(
                grid_x,
                color="gray",  # Grau statt weiß
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
            grid_y = self.pv.StructuredGrid(X2, Y2, Z2)
            points2 = grid_y.points
            points2[:, 2] += self._planar_z_offset
            grid_y.points = points2
            self._add_overlay_mesh(
                grid_y,
                color="gray",  # Grau statt weiß
                opacity=opacity,
                category="axis_plane",
            )
        except Exception:  # noqa: BLE001
            pass

    def _get_max_surface_dimension(self, settings) -> float:
        """Berechnet die maximale Dimension (Breite oder Länge) aller nicht-versteckten Surfaces.
        
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
        # Füge das "klassische" aktive Surface optional hinzu, falls nicht hidden
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
        t_sig_end = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Prüfe ob Speaker Arrays vorhanden sind (nicht versteckt) - nur für Darstellung, nicht für Signatur
        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        has_speaker_arrays_for_signature = False
        if isinstance(speaker_arrays, dict):
            for array in speaker_arrays.values():
                if not getattr(array, 'hide', False):
                    has_speaker_arrays_for_signature = True
                    break
        
        # 🎯 Signatur: Surface-Definitionen + aktive Highlight-IDs (für Auswahländerungen)
        signature_tuple = (tuple(surfaces_signature), tuple(sorted(active_ids_set)))
        
        # Prüfe ob sich etwas geändert hat (ohne has_speaker_arrays zu berücksichtigen)
        # Prüfe ob sich die Signatur geändert hat (Surface-Definitionen + Highlight-IDs)
        # WICHTIG: Wenn create_empty_plot_surfaces=True, immer ausführen (auch wenn Signatur gleich ist)
        signature_changed = True
        if self._last_surfaces_state is not None and not create_empty_plot_surfaces:
            if len(self._last_surfaces_state) == 2:
                last_signature_tuple, last_ids = self._last_surfaces_state
                last_ids_set = set(last_ids) if isinstance(last_ids, (list, tuple, set)) else set()
                if (last_signature_tuple == tuple(surfaces_signature) and 
                    last_ids_set == active_ids_set):
                    signature_changed = False
            elif self._last_surfaces_state == signature_tuple:
                signature_changed = False
        
        if not signature_changed and not create_empty_plot_surfaces:
            return
        
        # Lösche alte Surfaces
        t_clear_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        self.clear_category('surfaces')
        # Entferne auch die Fläche für enabled Surfaces im leeren Plot (falls vorhanden)
        # Dies stellt sicher, dass die Fläche entfernt wird, wenn SPL-Daten vorhanden sind
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
        
        # Verwende has_speaker_arrays aus der Signatur-Berechnung
        has_speaker_arrays = has_speaker_arrays_for_signature
        
        # Zeichne alle nicht versteckten Surfaces (enabled und disabled)
        # OPTIMIERUNG: Batch-Zeichnen - kombiniere alle Surfaces in zwei PolyData-Objekte
        # (eines für aktive, eines für inaktive) statt einzelner add_mesh Aufrufe
        z_offset = self._planar_z_offset
        surfaces_drawn = 0
        t_draw_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
        
        # Sammle alle Punkte und Linien für Batch-Zeichnen
        active_enabled_points_list = []  # Liste von Punkt-Arrays für aktive ENABLED Surfaces (roter Rahmen, keine Fläche)
        active_enabled_lines_list = []   # Liste von Linien-Arrays für aktive ENABLED Surfaces
        active_disabled_points_list = []  # Liste von Punkt-Arrays für aktive DISABLED Surfaces (roter Rahmen)
        active_disabled_lines_list = []   # Liste von Linien-Arrays für aktive DISABLED Surfaces
        inactive_points_list = []  # Liste von Punkt-Arrays für inaktive Surfaces
        inactive_lines_list = []   # Liste von Linien-Arrays für inaktive Surfaces
        inactive_surface_ids = []  # Liste von Surface-IDs für inaktive Surfaces (zur Zuordnung)
        inactive_surface_enabled = []  # Liste von enabled-Status für inaktive Surfaces (zur Unterscheidung)
        disabled_surface_ids = []  # Liste von ALLEN disabled Surface-IDs (aktiv und inaktiv) für graue Fläche
        disabled_surface_points = []  # Liste von Punkt-Arrays für ALLE disabled Surfaces (für graue Fläche)
        
        tolerance = 1e-6
        
        enabled_count = 0
        disabled_count = 0
        
        for surface_id, surface_def in surface_definitions.items():
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, 'enabled', False))
                hidden = bool(getattr(surface_def, 'hidden', False))
                points = getattr(surface_def, 'points', []) or []
            else:
                enabled = surface_def.get('enabled', False)
                hidden = surface_def.get('hidden', False)
                points = surface_def.get('points', [])
            
            # Überspringe nur versteckte Surfaces
            if hidden:
                continue
            
            if len(points) < 3:
                continue
            
            if enabled:
                enabled_count += 1
            else:
                disabled_count += 1
            
            # Prüfe ob dies ein aktives Surface ist (Einzel- oder Gruppen-Selektion)
            # Stelle sicher, dass surface_id als String verglichen wird
            is_active = (str(surface_id) in active_ids_set)
            
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
                
                # Prüfe ob Polygon bereits geschlossen ist
                first_point = point_coords[0]
                last_point = point_coords[-1]
                is_already_closed = (
                    abs(first_point[0] - last_point[0]) < tolerance
                    and abs(first_point[1] - last_point[1]) < tolerance
                    and abs(first_point[2] - last_point[2]) < tolerance
                )

                # Erzeuge IMMER ein explizit geschlossenes Polygon
                if is_already_closed:
                    closed_coords = point_coords
                else:
                    closed_coords = point_coords + [point_coords[0]]
                
                n_points = len(closed_coords)
                closed_coords_array = np.array(closed_coords, dtype=float)
                
                # 🎯 NEUE LOGIK: Disabled Surfaces (aktiv oder inaktiv) bekommen immer graue Fläche
                # Für Rahmen: aktiv = rot, inaktiv = schwarz
                if not enabled:
                    # Disabled Surface: Immer zur grauen Flächen-Liste hinzufügen
                    disabled_surface_ids.append(str(surface_id))
                    disabled_surface_points.append(closed_coords_array)
                    
                    # Für Rahmen: Unterscheide aktiv/inaktiv
                    if is_active:
                        # Aktive disabled: Roter Rahmen (zusätzlich zur grauen Fläche)
                        active_disabled_points_list.append(closed_coords_array)
                        active_disabled_lines_list.append(n_points)
                    else:
                        # Inaktive disabled: Schwarzer Rahmen (wie bisher)
                        inactive_points_list.append(closed_coords_array)
                        inactive_lines_list.append(n_points)
                        inactive_surface_ids.append(str(surface_id))
                        inactive_surface_enabled.append(False)
                else:
                    # Enabled Surface: Alte Logik (roter Rahmen wenn aktiv, sonst schwarzer Rahmen)
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
        
        # Debug-Ausgaben entfernt: Surfaces-Zusammenfassung
        
        # 🎯 Zeichne Surfaces EINZELN mit IDs für Picking (nur für disabled Surfaces)
        # Für enabled Surfaces verwenden wir die SPL-Surface und prüfen beim Klick, welche Surface den Punkt enthält
        # Für disabled Surfaces müssen wir jede einzeln zeichnen, damit wir sie beim Klick identifizieren können
        
        # Prüfe ob SPL-Daten vorhanden sind (durch Prüfung ob SPL-Actors existieren)
        has_spl_data = False
        try:
            # 🎯 WICHTIG: Prüfe zuerst _surface_texture_actors direkt (falls Actors noch nicht im Renderer registriert sind)
            # Dies ist wichtig beim Laden, wenn draw_surfaces vor update_spl_plot aufgerufen wird
            if hasattr(self.plotter, '_surface_texture_actors'):
                direct_texture_count = len(getattr(self.plotter, '_surface_texture_actors', {}))
                if direct_texture_count > 0:
                    has_spl_data = True
            
            # Prüfe auch Renderer-Actors (falls bereits registriert)
            if not has_spl_data and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                texture_actor_names = [name for name in self.plotter.renderer.actors.keys() if name.startswith('spl_surface_tex_')]
                has_texture_actors = len(texture_actor_names) > 0
                if spl_surface_actor is not None or spl_floor_actor is not None or has_texture_actors:
                    has_spl_data = True
            
            # 🎯 ZUSÄTZLICH: Prüfe ob container calculation_spl Daten hat (wichtig beim Laden)
            # Dies hilft beim Laden, wenn draw_surfaces vor update_spl_plot aufgerufen wird
            # WICHTIG: Prüfe IMMER, auch wenn has_spl_data bereits True ist, um sicherzustellen,
            # dass die Prüfung korrekt funktioniert
            if container is not None and hasattr(container, 'calculation_spl'):
                calc_spl = container.calculation_spl
                if isinstance(calc_spl, dict) and calc_spl.get('sound_field_p') is not None:
                    # Daten sind vorhanden, auch wenn Actors noch nicht erstellt wurden
                    # Zeichne keine graue Fläche, da SPL-Daten geplottet werden sollen
                    if not has_spl_data:
                        has_spl_data = True
        except Exception:
            # Fehler bei SPL-Daten-Prüfung ignorieren – Plot wird einfach ohne graue Fläche gezeichnet
            pass
        
        # Zeichne aktive ENABLED Surfaces als Batch (roter Rahmen, keine Fläche)
        if active_enabled_points_list:
            try:
                all_active_enabled_points = np.vstack(active_enabled_points_list)
                active_enabled_lines_array = []
                point_offset = 0
                for n_pts in active_enabled_lines_list:
                    active_enabled_lines_array.append(n_pts)
                    active_enabled_lines_array.extend(range(point_offset, point_offset + n_pts))
                    point_offset += n_pts
                
                # Zeichne Rahmen (immer)
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
                    opacity=1.0,  # Vollständig opak (keine Transparenz)
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=False,
                )
            except Exception:
                pass
        
        # Zeichne aktive DISABLED Surfaces als Batch (roter Rahmen, Fläche wird später gezeichnet)
        if active_disabled_points_list:
            try:
                all_active_disabled_points = np.vstack(active_disabled_points_list)
                active_disabled_lines_array = []
                point_offset = 0
                for n_pts in active_disabled_lines_list:
                    active_disabled_lines_array.append(n_pts)
                    active_disabled_lines_array.extend(range(point_offset, point_offset + n_pts))
                    point_offset += n_pts
                
                # Zeichne Rahmen (roter Rahmen für aktive disabled Surfaces)
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
                    opacity=1.0,  # Vollständig opak (keine Transparenz)
                    category='surfaces',
                    show_vertices=False,
                    render_lines_as_tubes=False,
                )
            except Exception:
                pass
        
        # 🎯 Zeichne Fläche für enabled Surfaces NUR wenn create_empty_plot_surfaces=True
        # (wird nur in show_empty_spl gesetzt)
        # BOTH active AND inactive enabled Surfaces bekommen graue Flächen im leeren Plot
        if create_empty_plot_surfaces:
            # Sammle alle enabled Surfaces (aktiv und inaktiv) für graue Fläche
            enabled_points_for_empty_plot = []
            enabled_faces_for_empty_plot = []
            point_offset = 0
            
            # Aktive enabled Surfaces
            if active_enabled_points_list:
                for idx, points in enumerate(active_enabled_points_list):
                    n_pts = len(points)
                    enabled_points_for_empty_plot.append(points)
                    # Face-Format: [n, 0, 1, 2, ..., n-1] mit offset
                    face = [n_pts] + [point_offset + i for i in range(n_pts)]
                    enabled_faces_for_empty_plot.extend(face)
                    point_offset += n_pts
            
            # Inaktive enabled Surfaces (nur die, die enabled sind)
            inactive_enabled_count = 0
            if inactive_points_list:
                for idx, points in enumerate(inactive_points_list):
                    if idx < len(inactive_surface_ids) and idx < len(inactive_surface_enabled):
                        is_enabled = inactive_surface_enabled[idx]
                        if is_enabled:  # Nur enabled Surfaces
                            inactive_enabled_count += 1
                            n_pts = len(points)
                            enabled_points_for_empty_plot.append(points)
                            # Face-Format: [n, 0, 1, 2, ..., n-1] mit offset
                            face = [n_pts] + [point_offset + i for i in range(n_pts)]
                            enabled_faces_for_empty_plot.extend(face)
                            point_offset += n_pts
                if inactive_enabled_count > 0:
                    # Anzahl inaktiver enabled Surfaces hier nicht mehr ausführlich loggen
                    pass
            
            # Zeichne alle enabled Surfaces (aktiv + inaktiv) als Batch
            if enabled_points_for_empty_plot:
                try:
                    all_enabled_points = np.vstack(enabled_points_for_empty_plot)
                    enabled_polygon_mesh = self.pv.PolyData(all_enabled_points)
                    enabled_polygon_mesh.faces = enabled_faces_for_empty_plot
                    actor_name = "surface_enabled_empty_plot_batch"
                    actor = self.plotter.add_mesh(
                        enabled_polygon_mesh,
                        name=actor_name,
                        color='#D3D3D3',  # Hellgrau (heller als vorher)
                        opacity=0.8,  # 80% Opacity (20% Transparenz)
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        show_edges=False,
                    )
                    # Nicht pickable, damit Klicks auf dahinterliegende Elemente funktionieren
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
            # Entferne die Fläche für leeren Plot (falls vorhanden), da sie nicht hier erstellt wird
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
        
        # 🎯 Zeichne inaktive Surfaces als transparente hellgraue Flächen
        # Lösche alte inaktive Surface-Actors (die nicht mehr existieren)
        # Batch-Actors haben feste Namen
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
        
        # 🎯 WICHTIG: Wenn SPL-Daten vorhanden sind, entferne auch die Fläche für inaktive enabled Surfaces
        # (die in surface_disabled_polygons_batch enthalten sein könnte)
        if has_spl_data:
            try:
                disabled_polygons_actor = self.plotter.renderer.actors.get('surface_disabled_polygons_batch')
                if disabled_polygons_actor is not None:
                    # Die Fläche wird unten neu gezeichnet, nur mit disabled Surfaces (ohne enabled Surfaces mit SPL-Daten)
                    pass
            except Exception:
                pass
        
        # 🎯 Zeichne graue Flächen für ALLE disabled Surfaces (aktiv und inaktiv)
        valid_disabled_polygons = []  # Liste von (points, surface_id) für gültige disabled Polygone
        
        for idx, surface_id in enumerate(disabled_surface_ids):
            if idx < len(disabled_surface_points):
                try:
                    points = disabled_surface_points[idx]
                    n_pts = len(points)
                    
                    # Prüfe ob Polygon gültig ist (mindestens 3 Punkte, nicht alle auf einer Linie)
                    if n_pts < 3:
                        continue
                    
                    # Prüfe ob alle Punkte unterschiedlich sind (nicht alle identisch)
                    points_array = np.array(points)
                    if len(points_array) < 3:
                        continue
                    
                    # Prüfe ob Punkte nicht alle auf einer Linie liegen
                    # Berechne Vektoren zwischen benachbarten Punkten
                    vectors = points_array[1:] - points_array[:-1]
                    # Prüfe ob alle Vektoren parallel sind (Kreuzprodukt = 0)
                    if len(vectors) >= 2:
                        # Normalisiere ersten Vektor
                        v1 = vectors[0]
                        v1_norm = np.linalg.norm(v1)
                        if v1_norm < 1e-6:  # Zu kurzer Vektor
                            continue
                        v1_normalized = v1 / v1_norm
                        
                        # Prüfe ob alle anderen Vektoren parallel zu v1 sind
                        all_parallel = True
                        for v in vectors[1:]:
                            v_norm = np.linalg.norm(v)
                            if v_norm < 1e-6:  # Zu kurzer Vektor
                                all_parallel = False
                                break
                            v_normalized = v / v_norm
                            # Prüfe ob Vektoren parallel sind (Kreuzprodukt sollte ~0 sein)
                            cross = np.cross(v1_normalized, v_normalized)
                            if np.linalg.norm(cross) > 1e-3:  # Nicht parallel
                                all_parallel = False
                                break
                        
                        if all_parallel:
                            # Alle Punkte liegen auf einer Linie - überspringe (keine Fläche)
                            continue
                    
                    # Disabled Surface: Füge zu Polygon-Liste hinzu
                    valid_disabled_polygons.append((points, surface_id))
                except Exception:
                    continue
        
        if valid_disabled_polygons:
            try:
                # Sammle alle Punkte und Faces für Batch-Zeichnen
                all_polygon_points = []
                all_polygon_faces = []
                point_offset = 0
                
                for points, surface_id in valid_disabled_polygons:
                    n_pts = len(points)
                    all_polygon_points.append(points)
                    # Face-Format: [n, 0, 1, 2, ..., n-1] mit offset
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
                        color='#D3D3D3',  # Hellgrau (heller als vorher)
                        opacity=0.8,  # 80% Opacity (20% Transparenz)
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        show_edges=False,
                    )
                    # WICHTIG: Disabled-Surfaces sollen nicht klickbar sein,
                    # damit Klicks auf senkrechte/horizontale aktive Flächen dahinter ankommen.
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
        
        # Sammle inaktive Linien für schwarze Rahmen (nur inaktive disabled und inaktive enabled Surfaces)
        # Aktive disabled Surfaces haben bereits rote Rahmen (oben gezeichnet)
        valid_inactive_lines = []  # Liste von (points, surface_id) für inaktive Linien
        
        for idx, surface_id in enumerate(inactive_surface_ids):
            if idx < len(inactive_points_list):
                try:
                    points = inactive_points_list[idx]
                    is_enabled = inactive_surface_enabled[idx] if idx < len(inactive_surface_enabled) else False
                    n_pts = len(points)
                    
                    if n_pts < 3:
                        continue
                    
                    # Alle inaktiven Surfaces (enabled und disabled) bekommen schwarze Rahmen
                    # (Aktive disabled Surfaces haben bereits rote Rahmen oben)
                    valid_inactive_lines.append((points, surface_id))
                except Exception:
                    continue
        
        # Zeichne alle gültigen Linien als Batch (opake schwarze Linien)
        if valid_inactive_lines:
            try:
                # Sammle alle Punkte und Lines für Batch-Zeichnen
                all_line_points = []
                all_line_arrays = []
                point_offset = 0
                
                for points, surface_id in valid_inactive_lines:
                    n_pts = len(points)
                    all_line_points.append(points)
                    # Line-Format: [n, 0, 1, 2, ..., n-1] mit offset
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
                        color='#000000',  # Schwarz
                        line_width=1.5,
                        opacity=1.0,  # Vollständig opak
                        smooth_shading=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        render_lines_as_tubes=False,
                    )
                    # Auch die Kanten der disabled-Surfaces nicht pickbar machen
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

    def draw_speakers(self, settings, container, cabinet_lookup: dict) -> None:
        """Delegiert an die Speaker-Mixin-Implementierung."""
        super().draw_speakers(settings, container, cabinet_lookup)

    # Delegations-Wrapper für Speaker-Hilfsmethoden (Implementierung im Mixin)
    def _get_speaker_info_from_actor(self, actor: Any, actor_name: str) -> Optional[Tuple[str, int]]:
        return super()._get_speaker_info_from_actor(actor, actor_name)

    def _update_speaker_highlights(self, settings) -> None:
        return super()._update_speaker_highlights(settings)

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

    # Lautsprecher-Hilfen kommen aus dem Mixin; hier nur Wrapper für bestehende Aufrufer
    def build_cabinet_lookup(self, container) -> dict:
        return super().build_cabinet_lookup(container)

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
        
        # Erhöhe Picking-Priorität für Achsenlinien (damit sie vor Surfaces gepickt werden)
        if category == 'axis':
            try:
                # Stelle sicher, dass der Actor pickable ist
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(True)
                
                # Stelle sicher, dass der Actor auch über GetProperty pickable ist
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if prop:
                        # Stelle sicher, dass der Actor sichtbar und pickable ist
                        prop.SetOpacity(1.0)
                        # Stelle sicher, dass die Linie sichtbar ist
                        prop.SetLineWidth(line_width)
                
                # Stelle sicher, dass der Actor in der Render-Liste ist
                actor.Modified()
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
        
        # Achsenflächen (axis_plane) sollen für Mausereignisse transparent sein
        if category == 'axis_plane':
            try:
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(False)
                # Zusätzliche Sicherheit über Property, falls verfügbar
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if prop:
                        # Nur Transparenz/Erscheinung, Picking bleibt über Actor deaktiviert
                        prop.PickableOff() if hasattr(prop, 'PickableOff') else None
            except Exception:  # noqa: BLE001
                pass
        
        # Stelle sicher, dass Edges angezeigt werden, wenn edge_color gesetzt wurde
        if edge_color is not None and hasattr(actor, 'prop') and actor.prop is not None:
            try:
                actor.prop.SetEdgeVisibility(True)
                actor.prop.SetRepresentationToSurface()
                # Stelle sicher, dass die Edge-Color gesetzt ist
                if edge_color == 'red':
                    actor.prop.SetEdgeColor(1, 0, 0)
                elif edge_color == 'black':
                    actor.prop.SetEdgeColor(0, 0, 0)
                actor.Modified()
            except Exception:  # noqa: BLE001
                pass
        
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
                # Stelle sicher, dass Line-Stippling aktiviert ist (Tubes deaktivieren)
                actor.prop.SetRenderLinesAsTubes(False)  # WICHTIG: Deaktiviere Tubes für Stippling
                # Stelle sicher, dass Line-Width gesetzt ist
                actor.prop.SetLineWidth(line_width)
                # Setze Pattern mit VTK-Methoden (nicht direkte Attribute!)
                actor.prop.SetLineStipplePattern(int(line_pattern))
                actor.prop.SetLineStippleRepeatFactor(max(1, int(line_repeat)))
                # Prüfe ob Pattern gesetzt wurde
                actual_pattern = actor.prop.GetLineStipplePattern()
                actual_repeat = actor.prop.GetLineStippleRepeatFactor()
                actual_width = actor.prop.GetLineWidth()
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
        elif is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
            # Für Tube-Meshes: Explizit durchgezogene Linie setzen (kein Stipple-Pattern)
            try:
                actor.prop.SetLineStipplePattern(0xFFFF)  # Durchgezogen
                actor.prop.SetLineStippleRepeatFactor(1)
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

    def _get_active_xy_surfaces(self, settings) -> List[Tuple[str, Any]]:
        """Sammelt alle aktiven Surfaces für XY-Berechnung (xy_enabled=True, enabled=True, hidden=False)."""
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

    def _get_surface_intersection_points_xz(self, y_const: float, surface: Any, settings) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    def _get_surface_intersection_points_yz(self, x_const: float, surface: Any, settings) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    def _create_dash_dot_line_segments_optimized(self, points_3d: np.ndarray, dash_length: float = 1.0, dot_length: float = 0.2, gap_length: float = 0.3) -> List[np.ndarray]:
        """
        Optimierte Version: Teilt eine 3D-Linie in Strich-Punkt-Segmente auf.
        Verwendet größere Segmente und vereinfachte Interpolation für bessere Performance.
        
        Pattern: Strich - Lücke - Punkt - Lücke - Strich - ...
        
        Args:
            points_3d: Array von 3D-Punkten (N x 3)
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
        
        # Berechne kumulative Distanzen entlang der Linie
        diffs = np.diff(points_3d, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
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
                dash_points = self._interpolate_line_segment_simple(points_3d, cumulative_distances, dash_start, dash_end)
                if len(dash_points) >= 2:
                    segments.append(dash_points)
            current_pos = dash_end + gap_length
            
            if current_pos >= total_length:
                break
            
            # Punkt-Segment
            dot_start = current_pos
            dot_end = min(current_pos + dot_length, total_length)
            if dot_end > dot_start:
                dot_points = self._interpolate_line_segment_simple(points_3d, cumulative_distances, dot_start, dot_end)
                if len(dot_points) >= 2:
                    segments.append(dot_points)
            current_pos = dot_end + gap_length
        
        return segments

    def _combine_line_segments(self, segments: List[np.ndarray]) -> Optional[Any]:
        """
        Kombiniert mehrere Liniensegmente in ein einziges PolyData-Mesh für effizientes Rendering.
        
        Args:
            segments: Liste von Punkt-Arrays (jedes Array ist N x 3)
            
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
            polyline = self.pv.PolyData(combined_points)
            polyline.lines = np.array(all_lines, dtype=np.int64)
            
            return polyline
        except Exception as e:  # noqa: BLE001
            print(f"[DEBUG _combine_line_segments] Fehler: {e}")
            return None

    def _interpolate_line_segment_simple(self, points_3d: np.ndarray, cumulative_distances: np.ndarray, start_dist: float, end_dist: float) -> np.ndarray:
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

    def _create_dash_dot_line_segments(self, points_3d: np.ndarray, dash_length: float = 0.5, dot_length: float = 0.1, gap_length: float = 0.2) -> List[np.ndarray]:
        """
        Teilt eine 3D-Linie in Strich-Punkt-Segmente auf.
        
        Pattern: Strich - Lücke - Punkt - Lücke - Strich - ...
        
        Args:
            points_3d: Array von 3D-Punkten (N x 3)
            dash_length: Länge eines Strichs in Metern
            dot_length: Länge eines Punkts in Metern
            gap_length: Länge einer Lücke in Metern
            
        Returns:
            Liste von Punkt-Arrays, die die sichtbaren Segmente (Striche und Punkte) darstellen
        """
        if len(points_3d) < 2:
            return []
        
        # Berechne kumulative Distanzen entlang der Linie
        diffs = np.diff(points_3d, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_distances[-1]
        
        if total_length <= 0:
            return []
        
        # Pattern: Strich (dash_length) - Lücke (gap_length) - Punkt (dot_length) - Lücke (gap_length) - ...
        pattern_length = dash_length + gap_length + dot_length + gap_length
        segments = []
        current_pos = 0.0
        
        # Interpoliere Punkte entlang der Linie für präzise Segmentierung
        while current_pos < total_length:
            # Strich-Segment
            dash_start = current_pos
            dash_end = min(current_pos + dash_length, total_length)
            if dash_end > dash_start:
                dash_points = self._interpolate_line_segment(points_3d, cumulative_distances, dash_start, dash_end)
                if len(dash_points) >= 2:
                    segments.append(dash_points)
            current_pos = dash_end + gap_length
            
            if current_pos >= total_length:
                break
            
            # Punkt-Segment
            dot_start = current_pos
            dot_end = min(current_pos + dot_length, total_length)
            if dot_end > dot_start:
                dot_points = self._interpolate_line_segment(points_3d, cumulative_distances, dot_start, dot_end)
                if len(dot_points) >= 2:
                    segments.append(dot_points)
            current_pos = dot_end + gap_length
        
        return segments

    def _interpolate_line_segment(self, points_3d: np.ndarray, cumulative_distances: np.ndarray, start_dist: float, end_dist: float) -> np.ndarray:
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


__all__ = ['SPL3DOverlayRenderer']

