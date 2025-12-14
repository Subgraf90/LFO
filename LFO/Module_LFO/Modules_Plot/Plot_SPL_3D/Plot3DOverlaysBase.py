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
        # Referenz-Zoom-Level für Zoom-Skalierung (wird beim ersten Aufruf gesetzt)
        self._reference_zoom_level: Optional[float] = None
        
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
    
    def clear_category(self, category: str) -> None:
        """Entfernt alle Actor einer Kategorie, ohne andere Overlays anzutasten."""
        actor_names = self._category_actors.pop(category, [])
        if not isinstance(actor_names, list):
            actor_names = list(actor_names) if actor_names else []
        
        for name in actor_names:
            try:
                self.plotter.remove_actor(name)
            except KeyError:
                pass
            else:
                if isinstance(self.overlay_actor_names, list):
                    if name in self.overlay_actor_names:
                        self.overlay_actor_names.remove(name)
                else:
                    self.overlay_actor_names = list(self.overlay_actor_names) if self.overlay_actor_names else []
                    if name in self.overlay_actor_names:
                        self.overlay_actor_names.remove(name)
    
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
    ) -> str:
        """Fügt ein Mesh als Overlay hinzu und gibt den Actor-Namen zurück."""
        prefix = f"{self._overlay_prefix}" if self._overlay_prefix else ""
        name = f"{prefix}overlay_{self.overlay_counter}"
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
        
        # Stelle sicher, dass keine Eckpunkte angezeigt werden (nur Linien)
        if not show_vertices and hasattr(mesh, 'lines') and mesh.lines is not None:
            kwargs['render_points_as_spheres'] = False
            kwargs['point_size'] = 0

        actor = self.plotter.add_mesh(mesh, **kwargs)
        
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
                actor.prop.SetLineStipplePattern(int(line_pattern))
                actor.prop.SetLineStippleRepeatFactor(max(1, int(line_repeat)))
            except Exception:  # noqa: BLE001
                pass
        elif is_tube_mesh and hasattr(actor, 'prop') and actor.prop is not None:
            try:
                actor.prop.SetLineStipplePattern(0xFFFF)  # Durchgezogen
                actor.prop.SetLineStippleRepeatFactor(1)
            except Exception:  # noqa: BLE001
                pass
        
        self.overlay_actor_names.append(name)
        self._category_actors.setdefault(category, []).append(name)
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
            
            # Setze Referenz-Zoom beim ersten Aufruf (oder wenn noch nicht gesetzt)
            if self._reference_zoom_level is None or self._reference_zoom_level <= 0:
                self._reference_zoom_level = current_zoom
                return 1.0
            
            # Berechne Skalierungsfaktor: reference / current
            # Wenn current_zoom größer ist (herausgezoomt), wird der Faktor < 1.0
            # Wenn current_zoom kleiner ist (hineingezoomt), wird der Faktor > 1.0
            raw_factor = self._reference_zoom_level / current_zoom
            
            # Verwende Wurzel-Skalierung für sanftere Anpassung
            # Dies reduziert die Aggressivität der Zoom-Skalierung
            # z.B. bei 4x Zoom-Änderung: sqrt(4) = 2x statt 4x Skalierung
            zoom_factor = np.sqrt(raw_factor) if raw_factor > 0 else 1.0
            
            # Begrenze den Faktor auf einen sinnvollen Bereich (0.3 bis 3.0)
            zoom_factor = max(0.3, min(3.0, zoom_factor))
            
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

