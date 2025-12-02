"""Camera-Management für den 3D-SPL-Plot."""

from __future__ import annotations

import math
from typing import Optional, Any

import numpy as np


class CameraManager:
    """Verwaltet Camera-Operationen für den 3D-Plot."""
    
    def __init__(self, plotter: Any, widget: Any, settings: Any, render_func, skip_render_restore_getter, skip_render_restore_setter):
        """Initialisiert den Camera-Manager.
        
        Args:
            plotter: PyVista Plotter
            widget: Widget für Größenberechnungen
            settings: Settings-Objekt
            render_func: Funktion zum Rendern
            skip_render_restore_getter: Funktion zum Abrufen von _skip_next_render_restore
            skip_render_restore_setter: Funktion zum Setzen von _skip_next_render_restore
        """
        self.plotter = plotter
        self.widget = widget
        self.settings = settings
        self._render_func = render_func
        self._get_skip_render_restore = skip_render_restore_getter
        self._set_skip_render_restore = skip_render_restore_setter
        self._camera_state: Optional[dict] = None
    
    def pan_camera(self, delta_x: float, delta_y: float):
        """Verschiebt die Kamera (Pan).
        
        Args:
            delta_x: Delta in X-Richtung (normalisiert)
            delta_y: Delta in Y-Richtung (normalisiert)
        """
        if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
            return
        camera = self.plotter.camera
        try:
            position = np.array(camera.position, dtype=float)
            focal = np.array(camera.focal_point, dtype=float)
            view_up = np.array(camera.up, dtype=float)
        except Exception:
            return

        view_vector = focal - position
        distance = float(np.linalg.norm(view_vector))
        if distance <= 0.0:
            return

        view_dir = view_vector / distance
        if np.linalg.norm(view_up) == 0.0:
            view_up = np.array([0.0, 0.0, 1.0])
        up_norm = view_up / np.linalg.norm(view_up)
        right = np.cross(view_dir, up_norm)
        if np.linalg.norm(right) == 0.0:
            return
        right_norm = right / np.linalg.norm(right)
        up_orth = np.cross(right_norm, view_dir)
        if np.linalg.norm(up_orth) == 0.0:
            up_orth = up_norm
        else:
            up_orth /= np.linalg.norm(up_orth)

        widget_width = max(1, self.widget.width())
        widget_height = max(1, self.widget.height())

        parallel_projection = bool(getattr(camera, 'parallel_projection', False))
        if parallel_projection:
            parallel_scale = float(getattr(camera, 'parallel_scale', distance))
            scale_y = (2.0 * parallel_scale) / widget_height
        else:
            view_angle = float(getattr(camera, 'view_angle', 30.0))
            view_angle = max(1e-3, min(179.0, view_angle))
            scale_y = (2.0 * distance * math.tan(math.radians(view_angle) / 2.0)) / widget_height
        scale_x = scale_y * (widget_width / widget_height)

        translation = (-delta_x * scale_x * right_norm) + (delta_y * scale_y * up_orth)

        new_position = position + translation
        new_focal = focal + translation

        camera.position = tuple(new_position)
        camera.focal_point = tuple(new_focal)
        camera.up = tuple(up_orth)
        self._set_skip_render_restore(True)
        self._render_func()
        self._set_skip_render_restore(False)

    def rotate_camera(self, delta_x: float, delta_y: float) -> None:
        """Rotiert die Kamera.
        
        Args:
            delta_x: Delta in X-Richtung (normalisiert)
            delta_y: Delta in Y-Richtung (normalisiert)
        """
        if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
            return
        camera = self.plotter.camera
        try:
            _ = camera.position
            _ = camera.focal_point
            _ = camera.up
        except Exception:
            return

        rotate_scale = 0.25
        angle_vert = float(delta_y) * rotate_scale
        angle_horiz = -float(delta_x) * rotate_scale

        if abs(angle_vert) > 1e-6:
            try:
                camera.Elevation(angle_vert)
                if hasattr(camera, 'OrthogonalizeViewUp'):
                    camera.OrthogonalizeViewUp()
            except Exception:
                pass

        if abs(angle_horiz) > 1e-6:
            self.rotate_camera_around_z(angle_horiz)

        if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'ResetCameraClippingRange'):
            try:
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass
        self._set_skip_render_restore(True)
        self._render_func()
        self._set_skip_render_restore(False)

    def rotate_camera_around_z(self, angle_deg: float) -> None:
        """Rotiert die Kamera um die Z-Achse.
        
        Args:
            angle_deg: Winkel in Grad
        """
        camera = self.plotter.camera
        try:
            position = np.array(camera.position, dtype=float)
            focal = np.array(camera.focal_point, dtype=float)
            view_up = np.array(camera.up, dtype=float)
        except Exception:
            return

        rotation_rad = math.radians(angle_deg)
        cos_a = math.cos(rotation_rad)
        sin_a = math.sin(rotation_rad)
        rot_matrix = np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        relative_pos = position - focal
        new_relative = rot_matrix @ relative_pos
        new_position = focal + new_relative

        new_up = rot_matrix @ view_up
        if np.linalg.norm(new_up) > 0.0:
            new_up = new_up / np.linalg.norm(new_up)
        else:
            new_up = np.array([0.0, 0.0, 1.0], dtype=float)

        try:
            camera.position = tuple(new_position)
            camera.up = tuple(new_up)
        except Exception:
            return

    def maximize_camera_view(self, add_padding: bool = False) -> None:
        """Maximiert die Camera-Ansicht.
        
        Args:
            add_padding: Ob zusätzlicher Padding hinzugefügt werden soll
        """
        if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
            return
        cam = self.plotter.camera
        try:
            if getattr(cam, 'parallel_projection', False):
                if hasattr(cam, 'parallel_scale'):
                    bounds = getattr(self.plotter, 'bounds', None)
                    if bounds is not None and len(bounds) == 6:
                        width = abs(bounds[1] - bounds[0])
                        height = abs(bounds[3] - bounds[2])
                        max_extent = max(width, height)
                        scale_factor = 0.45
                        if add_padding:
                            scale_factor = 0.52
                        cam.parallel_scale = max_extent * scale_factor
                    else:
                        factor = 0.7 if not add_padding else 0.78
                        cam.parallel_scale = float(cam.parallel_scale) * factor
            else:
                if hasattr(cam, 'view_angle'):
                    angle = float(cam.view_angle) * (0.55 if not add_padding else 0.65)
                    cam.view_angle = max(3.0, min(60.0, angle))
                if hasattr(cam, 'ResetClippingRange'):
                    try:
                        cam.ResetClippingRange()
                    except Exception:
                        pass
        except Exception:
            pass
    
    def zoom_to_default_surface(self, capture_camera_func, set_camera_state_func) -> None:
        """Stellt den Zoom auf das Default-Surface ein.
        
        Args:
            capture_camera_func: Funktion zum Erfassen des Camera-States
            set_camera_state_func: Funktion zum Setzen des Camera-States
        """
        try:
            # Hole Default-Surface aus Settings
            default_surface_id = getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default')
            surface_definitions = getattr(self.settings, 'surface_definitions', {})
            
            if default_surface_id not in surface_definitions:
                return
            
            surface_def = surface_definitions[default_surface_id]
            if isinstance(surface_def, dict):
                points = surface_def.get('points', [])
            else:
                points = getattr(surface_def, 'points', []) or []
            
            if not points or len(points) < 3:
                return
            
            # Berechne Bounds aus Surface-Punkten
            x_coords = [float(p.get('x', 0.0)) for p in points]
            y_coords = [float(p.get('y', 0.0)) for p in points]
            z_coords = [float(p.get('z', 0.0)) for p in points]
            
            if not x_coords or not y_coords:
                return
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            
            # Berechne die Ausdehnung des Surfaces
            width = abs(max_x - min_x) if max_x != min_x else 1.0
            height = abs(max_y - min_y) if max_y != min_y else 1.0
            max_extent = max(width, height)
            
            # Setze Bounds im Plotter und zoome darauf
            if hasattr(self.plotter, 'camera') and self.plotter.camera is not None:
                # Setze die Kamera-Position auf die Mitte des Surfaces
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                center_z = (min_z + max_z) / 2.0
                
                # Stelle sicher, dass die Kamera auf die Top-Ansicht eingestellt ist
                self.plotter.view_xy()
                
                # Setze die Kamera-Position
                cam = self.plotter.camera
                if hasattr(cam, 'position'):
                    # Aktiviere Parallelprojektion für die Draufsicht
                    if hasattr(cam, "parallel_projection"):
                        cam.parallel_projection = True

                    # Positioniere die Kamera über dem Surface
                    distance = max_extent * 1.0
                    min_distance = 20.0
                    distance = max(distance, min_distance)
                    cam.position = (center_x, center_y, center_z + distance)
                    cam.focal_point = (center_x, center_y, center_z)
                    cam.up = (0, 1, 0)
                
                # Setze den Zoom basierend auf den Bounds
                if getattr(cam, 'parallel_projection', False):
                    if hasattr(cam, 'parallel_scale'):
                        scale = max_extent * 0.56
                        cam.parallel_scale = scale
                else:
                    if hasattr(cam, 'view_angle'):
                        visible_extent = max_extent * 1.1
                        angle = 2.0 * np.arctan(visible_extent / (2.0 * distance)) * 180.0 / np.pi
                        angle = max(30.0, min(60.0, angle))
                        cam.view_angle = angle
                
                # Reset Clipping Range
                if hasattr(cam, 'ResetClippingRange'):
                    try:
                        cam.ResetClippingRange()
                    except Exception:
                        pass

                # Verhindere, dass render() direkt danach einen alten Kamera-State wiederherstellt
                self._set_skip_render_restore(True)
                # Render, um die Änderungen anzuzeigen
                self._render_func()
                # Nach dem Rendern den aktuellen Kamera-State als neuen Referenzzustand speichern
                camera_state = capture_camera_func()
                if camera_state is not None:
                    set_camera_state_func(camera_state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass

    def set_view_isometric(self):
        """Setzt die isometrische Ansicht."""
        try:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.maximize_camera_view()
            self._set_skip_render_restore(True)
            self._render_func()
            self._set_skip_render_restore(False)
        except Exception:
            pass

    def set_view_top(self):
        """Setzt die Top-Ansicht."""
        try:
            self.plotter.view_xy()
            self.plotter.reset_camera()
            self.maximize_camera_view()
            self._set_skip_render_restore(True)
            self._render_func()
            self._set_skip_render_restore(False)
        except Exception:
            pass

    def set_view_side_y(self):
        """Setzt die Y-Seitenansicht."""
        try:
            self.plotter.view_xz()
            self.plotter.reset_camera()
            self.maximize_camera_view()
            self._set_skip_render_restore(True)
            self._render_func()
            self._set_skip_render_restore(False)
        except Exception:
            pass

    def set_view_side_x(self):
        """Setzt die X-Seitenansicht."""
        try:
            self.plotter.view_yz()
            self.plotter.reset_camera()
            self.maximize_camera_view()
            self._set_skip_render_restore(True)
            self._render_func()
            self._set_skip_render_restore(False)
        except Exception:
            pass

    def save_camera_state(self, capture_func) -> None:
        """Speichert den aktuellen Camera-State.
        
        Args:
            capture_func: Funktion zum Erfassen des Camera-States
        """
        state = capture_func()
        if state is not None:
            self._camera_state = state

    def capture_camera(self) -> Optional[dict]:
        """Erfasst den aktuellen Camera-State.
        
        Returns:
            Dictionary mit Camera-State oder None bei Fehler
        """
        if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
            return None
        cam = self.plotter.camera
        try:
            state: dict[str, object] = {
                'position': tuple(cam.position),
                'focal_point': tuple(cam.focal_point),
                'view_up': tuple(cam.up),
                'clipping_range': tuple(cam.clipping_range),
            }
            if hasattr(cam, 'parallel_projection'):
                state['parallel_projection'] = bool(cam.parallel_projection)
            if hasattr(cam, 'parallel_scale'):
                try:
                    state['parallel_scale'] = float(cam.parallel_scale)
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(cam, 'view_angle'):
                try:
                    state['view_angle'] = float(cam.view_angle)
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(cam, 'distance'):
                try:
                    state['distance'] = float(cam.distance)
                except Exception:  # noqa: BLE001
                    pass
            return state
        except Exception:  # noqa: BLE001
            return None

    def restore_camera(self, camera_state: Optional[dict]):
        """Stellt einen gespeicherten Camera-State wieder her.
        
        Args:
            camera_state: Dictionary mit Camera-State
        """
        if camera_state is None:
            return
        if not hasattr(self.plotter, 'camera') or self.plotter.camera is None:
            return
        cam = self.plotter.camera
        try:
            cam.position = camera_state['position']
            cam.focal_point = camera_state['focal_point']
            cam.up = camera_state['view_up']
            if 'clipping_range' in camera_state:
                cam.clipping_range = camera_state['clipping_range']
            if 'parallel_projection' in camera_state and hasattr(cam, 'parallel_projection'):
                cam.parallel_projection = bool(camera_state['parallel_projection'])
            if 'parallel_scale' in camera_state and hasattr(cam, 'parallel_scale'):
                try:
                    cam.parallel_scale = float(camera_state['parallel_scale'])
                except Exception:  # noqa: BLE001
                    pass
            if 'view_angle' in camera_state and hasattr(cam, 'view_angle'):
                try:
                    cam.view_angle = float(camera_state['view_angle'])
                except Exception:  # noqa: BLE001
                    pass
            if 'distance' in camera_state and hasattr(cam, 'distance'):
                try:
                    cam.distance = float(camera_state['distance'])
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

