"""PyVista-basierter SPL-Plot (3D)"""

from __future__ import annotations

import math
import os
import types
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize
from PyQt5 import QtCore, QtGui, QtWidgets

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Plot.PlotSPL3DOverlays import SPL3DOverlayRenderer

DEBUG_SPL_DUMP = bool(int(os.environ.get("LFO_DEBUG_SPL", "0")))

try:  # pragma: no cover - optional Abhängigkeit
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:  # noqa: BLE001
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]
    _PYVISTA_IMPORT_ERROR = exc
else:  # pragma: no cover
    _PYVISTA_IMPORT_ERROR = None


class DrawSPLPlot3D(ModuleBase, QtCore.QObject):
    """Erstellt einen interaktiven 3D-SPL-Plot auf Basis von PyVista."""

    SURFACE_NAME = "spl_surface"
    FLOOR_NAME = "spl_floor"
    UPSCALE_FACTOR = 12

    def __init__(self, parent_widget, settings, colorbar_ax):
        if QtInteractor is None or pv is None:  # pragma: no cover - Laufzeitprüfung
            message = (
                "PyVista (und pyvistaqt) sind nicht installiert. "
                "Bitte installieren Sie die Pakete, um den 3D-Plot verwenden zu können."
            )
            raise ImportError(message) from _PYVISTA_IMPORT_ERROR

        ModuleBase.__init__(self, settings)
        QtCore.QObject.__init__(self)

        self.parent_widget = parent_widget
        self.colorbar_ax = colorbar_ax
        self.settings = settings

        self.plotter = QtInteractor(parent_widget)
        self.widget = self.plotter.interactor
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)
        self._pan_active = False
        self._pan_last_pos: Optional[QtCore.QPoint] = None
        self._rotate_active = False
        self._rotate_last_pos: Optional[QtCore.QPoint] = None
        self._camera_state: Optional[dict[str, object]] = None
        self._skip_next_render_restore = False
        self._camera_debug_enabled = False
        self._camera_debug_counter = 0

        if not hasattr(self.plotter, '_cmap_to_lut'):
            self.plotter._cmap_to_lut = types.MethodType(self._fallback_cmap_to_lut, self.plotter)  # type: ignore[attr-defined]

        self.overlay_helper = SPL3DOverlayRenderer(self.plotter, pv)

        self.cbar = None
        self.has_data = False
        self.surface_mesh = None  # type: pv.DataSet | None

        # Colorbar-Caching: merkt sich den ScalarMappable sowie den Modus,
        # damit wir bei Area-Updates keine neue Colorbar erzeugen müssen.
        self._colorbar_mappable = None
        self._colorbar_mode: Optional[tuple[str, tuple[float, ...] | None]] = None

        # Overlay-Differenzial: Signaturen pro Kategorie (Achsen, Wände, etc.),
        # um gezielt neu zu zeichnen und Renderzeit zu sparen.
        self._last_overlay_signatures: dict[str, tuple] = {}

        # Guard gegen reentrante Render-Aufrufe (z. B. Kamera-Callbacks).
        self._is_rendering = False

        self._configure_plotter()
        self.initialize_empty_scene(preserve_camera=False)
        self._setup_view_controls()

    def eventFilter(self, obj, event):  # noqa: PLR0911
        if obj is self.widget:
            etype = event.type()
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._rotate_active = True
                self._rotate_last_pos = QtCore.QPoint(event.pos())
                self.widget.setCursor(QtCore.Qt.OpenHandCursor)
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                self._rotate_active = False
                self._rotate_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
                self._pan_active = True
                self._pan_last_pos = QtCore.QPoint(event.pos())
                self.widget.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                self._pan_active = False
                self._pan_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseMove and self._rotate_active and self._rotate_last_pos is not None:
                current_pos = QtCore.QPoint(event.pos())
                delta = current_pos - self._rotate_last_pos
                self._rotate_last_pos = current_pos
                self._rotate_camera(delta.x(), delta.y())
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseMove and self._pan_active and self._pan_last_pos is not None:
                current_pos = QtCore.QPoint(event.pos())
                delta = current_pos - self._pan_last_pos
                self._pan_last_pos = current_pos
                self._pan_camera(delta.x(), delta.y())
                event.accept()
                return True
            if etype == QtCore.QEvent.Leave:
                if self._pan_active or self._rotate_active:
                    self._pan_active = False
                    self._rotate_active = False
                    self._pan_last_pos = None
                    self._rotate_last_pos = None
                    self.widget.unsetCursor()
                    self._save_camera_state()
                    event.accept()
                    return True
            if etype == QtCore.QEvent.Wheel:
                QtCore.QTimer.singleShot(0, self._save_camera_state)
        return QtCore.QObject.eventFilter(self, obj, event)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def initialize_empty_scene(self, preserve_camera: bool = True):
        """Zeigt eine leere Szene und bewahrt optional die Kameraposition."""
        if preserve_camera:
            camera_state = self._camera_state or self._capture_camera()
        else:
            camera_state = None

        self.plotter.clear()
        self.overlay_helper.clear()
        self.has_data = False
        self.surface_mesh = None
        self._last_overlay_signatures = {}

        self._configure_plotter()
        self._add_floor_plane()
        self._add_scene_frame()

        if camera_state is not None:
            self._restore_camera(camera_state)
        else:
            self.set_view_top()
            camera_state = self._camera_state
            if camera_state is not None:
                self._restore_camera(camera_state)

        self._initialize_empty_colorbar()
        self._skip_next_render_restore = True
        self.render()
        self._skip_next_render_restore = False
        self._save_camera_state()

    def update_spl_plot(
        self,
        sound_field_x: Iterable[float],
        sound_field_y: Iterable[float],
        sound_field_pressure: Iterable[float],
        colorization_mode: str = "Gradient",
    ):
        """Aktualisiert die SPL-Fläche."""

        camera_state = self._camera_state or self._capture_camera()

        if not self._has_valid_data(sound_field_x, sound_field_y, sound_field_pressure):
            self.initialize_empty_scene(preserve_camera=True)
            return

        x = np.asarray(sound_field_x, dtype=float)
        y = np.asarray(sound_field_y, dtype=float)

        try:
            pressure = np.asarray(sound_field_pressure, dtype=float)
        except Exception:  # noqa: BLE001
            self.initialize_empty_scene(preserve_camera=True)
            return

        self._debug_dump_soundfield(x, y, pressure)

        if pressure.ndim != 2:
            # Erwartet wird ein (len(y), len(x))-Raster. Bei flachen Arrays
            # versuchen wir genau einmal ein Reshape mit der Zielgröße; gelingt
            # das nicht, brechen wir frühzeitig ab und zeigen eine leere Szene.
            if pressure.size == (len(y) * len(x)):
                try:
                    pressure = pressure.reshape(len(y), len(x))
                except Exception:  # noqa: BLE001
                    self.initialize_empty_scene(preserve_camera=True)
                    return
            else:
                self.initialize_empty_scene(preserve_camera=True)
                return

        if pressure.shape != (len(y), len(x)):
            self.initialize_empty_scene(preserve_camera=True)
            return

        pressure_2d = np.nan_to_num(np.abs(pressure), nan=0.0, posinf=0.0, neginf=0.0)
        pressure_2d = np.clip(pressure_2d, 1e-12, None)
        spl_db = self.functions.mag2db(pressure_2d)

        finite_mask = np.isfinite(spl_db)
        if not np.any(finite_mask):
            self.initialize_empty_scene(preserve_camera=True)
            return

        cbar_min = self.settings.colorbar_range['min']
        cbar_max = self.settings.colorbar_range['max']
        cbar_step = self.settings.colorbar_range['step']

        spl_db = np.clip(spl_db, cbar_min - 20, cbar_max + 20)

        plot_x = x
        plot_y = y
        plot_values = spl_db

        requested_upscale = getattr(self.settings, 'plot_upscale_factor', None)
        if requested_upscale is None:
            requested_upscale = self.UPSCALE_FACTOR
        try:
            upscale_factor = int(requested_upscale)
        except (TypeError, ValueError):
            upscale_factor = self.UPSCALE_FACTOR
        upscale_factor = max(1, upscale_factor)

        if (
            upscale_factor > 1
            and plot_x.size > 1
            and plot_y.size > 1
        ):
            base_resolution = float(getattr(self.settings, 'resolution', 0.0) or 0.0)
            orig_plot_x = plot_x.copy()
            orig_plot_y = plot_y.copy()
            plot_x = self._expand_axis_for_plot(plot_x, base_resolution, upscale_factor)
            plot_y = self._expand_axis_for_plot(plot_y, base_resolution, upscale_factor)
            plot_values = self._resample_values_to_grid(plot_values, orig_plot_x, orig_plot_y, plot_x, plot_y)

        if colorization_mode == 'Color step':
            scalars = self._quantize_to_steps(plot_values, cbar_step)
        else:
            scalars = plot_values

        mesh = self._build_surface_mesh(plot_x, plot_y, scalars)
        actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
        if actor is None or self.surface_mesh is None:
            self.surface_mesh = mesh.copy(deep=True)
            actor = self.plotter.add_mesh(
                self.surface_mesh,
                name=self.SURFACE_NAME,
                scalars='plot_scalars',
                cmap='jet',
                clim=(cbar_min, cbar_max),
                smooth_shading=False,
                show_scalar_bar=False,
                reset_camera=False,
                interpolate_before_map=False,
            )
            if hasattr(actor, 'prop') and actor.prop is not None:
                try:
                    actor.prop.interpolation = 'flat'
                except Exception:  # noqa: BLE001
                    pass
        else:
            self.surface_mesh.deep_copy(mesh)
            mapper = actor.mapper
            if mapper is not None:
                mapper.array_name = 'plot_scalars'
                mapper.scalar_range = (cbar_min, cbar_max)
                mapper.lookup_table = self.plotter._cmap_to_lut('jet')
                mapper.interpolate_before_map = False

        self._update_colorbar(colorization_mode)
        self.has_data = True

        if camera_state is not None:
            self._restore_camera(camera_state)
        self._maximize_camera_view(add_padding=True)
        self.render()
        self._save_camera_state()

    def update_overlays(self, settings, container):
        """Aktualisiert Zusatzobjekte (Achsen, Lautsprecher, Messpunkte)."""
        # Wir vergleichen Hash-Signaturen pro Kategorie und zeichnen nur dort neu,
        # wo sich Parameter geändert haben. Das verhindert unnötiges Entfernen
        # und erneutes Hinzufügen zahlreicher PyVista-Actor.
        signatures = self._compute_overlay_signatures(settings, container)
        previous = self._last_overlay_signatures or {}
        if not previous:
            categories_to_refresh = set(signatures.keys())
        else:
            categories_to_refresh = {
                key for key, value in signatures.items() if value != previous.get(key)
            }
        if not categories_to_refresh:
            self._last_overlay_signatures = signatures
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
            return

        prev_debug_state = getattr(self.overlay_helper, 'DEBUG_ON_TOP', False)
        try:
            self.overlay_helper.DEBUG_ON_TOP = True
            if 'axis' in categories_to_refresh:
                self.overlay_helper.clear_category('axis')
                self.overlay_helper.draw_axis_lines(settings)
            if 'walls' in categories_to_refresh:
                self.overlay_helper.clear_category('walls')
                self.overlay_helper.draw_walls(settings)
            if 'speakers' in categories_to_refresh:
                self.overlay_helper.clear_category('speakers')
                cabinet_lookup = self.overlay_helper.build_cabinet_lookup(container)
                self.overlay_helper.draw_speakers(settings, container, cabinet_lookup)
            if 'impulse' in categories_to_refresh:
                self.overlay_helper.clear_category('impulse')
                self.overlay_helper.draw_impulse_points(settings)
        finally:
            self.overlay_helper.DEBUG_ON_TOP = prev_debug_state
        self._last_overlay_signatures = signatures
        self.render()
        if not self._rotate_active and not self._pan_active:
            self._save_camera_state()

    def _debug_dump_soundfield(self, x: np.ndarray, y: np.ndarray, pressure: np.ndarray) -> None:
        if not DEBUG_SPL_DUMP:
            return
        try:
            print("DEBUG SPL shapes:", x.shape, y.shape, pressure.shape)
            print(
                "DEBUG SPL finite:",
                np.isfinite(x).all(),
                np.isfinite(y).all(),
                np.isfinite(pressure).all(),
            )
            if pressure.size:
                print(
                    "DEBUG SPL pressure stats:",
                    float(np.nanmin(pressure)),
                    float(np.nanmax(pressure)),
                    int(np.isnan(pressure).sum()),
                )
        except Exception:
            print("DEBUG SPL dump failed")

    def render(self):
        """Erzwingt ein Rendering der Szene."""
        if self._is_rendering:
            return
        self._is_rendering = True
        try:
            try:
                self.plotter.render()
            except Exception:  # pragma: no cover - Rendering kann im Offscreen fehlschlagen
                pass
            if hasattr(self, 'view_control_widget'):
                self.view_control_widget.raise_()
            if not self._skip_next_render_restore and self._camera_state is not None:
                self._restore_camera(self._camera_state)
            self._skip_next_render_restore = False
            self._save_camera_state()
        finally:
            self._is_rendering = False

    # ------------------------------------------------------------------
    # Kamera-Zustand
    # ------------------------------------------------------------------
    def _save_camera_state(self) -> None:
        state = self._capture_camera()
        if state is not None:
            self._camera_state = state

    def _capture_camera(self) -> Optional[dict]:
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

    def _restore_camera(self, camera_state: Optional[dict]):
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

    # ------------------------------------------------------------------
    # Kamera-Ansichten
    # ------------------------------------------------------------------
    def set_view_isometric(self):
        try:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self._maximize_camera_view()
            self._skip_next_render_restore = True
            self.render()
            self._skip_next_render_restore = False
        except Exception:
            pass

    def set_view_top(self):
        try:
            self.plotter.view_xy()
            self.plotter.reset_camera()
            self._maximize_camera_view()
            self._skip_next_render_restore = True
            self.render()
            self._skip_next_render_restore = False
        except Exception:
            pass

    def set_view_side_y(self):
        try:
            self.plotter.view_xz()
            self.plotter.reset_camera()
            self._maximize_camera_view()
            self._skip_next_render_restore = True
            self.render()
            self._skip_next_render_restore = False
        except Exception:
            pass

    def set_view_side_x(self):
        try:
            self.plotter.view_yz()
            self.plotter.reset_camera()
            self._maximize_camera_view()
            self._skip_next_render_restore = True
            self.render()
            self._skip_next_render_restore = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Overlay-Steuerelemente
    # ------------------------------------------------------------------
    def _setup_view_controls(self):
        if not hasattr(self, 'widget') or self.widget is None:
            return

        self.view_control_widget = QtWidgets.QFrame(self.widget)
        self.view_control_widget.setObjectName('pv_view_controls')
        self.view_control_widget.setStyleSheet(
            "QFrame#pv_view_controls {"
            "background-color: rgba(255, 255, 255, 200);"
            "border-radius: 6px;"
            "border: 1px solid rgba(0, 0, 0, 100);"
            "}"
        )
        self.view_control_widget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.view_control_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        layout = QtWidgets.QHBoxLayout(self.view_control_widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.btn_view_top = self._create_view_button('top', "Ansicht von oben", self.set_view_top)
        self.btn_view_left = self._create_view_button('side_x', "Ansicht von links", self.set_view_side_x)
        self.btn_view_right = self._create_view_button('side_y', "Ansicht von rechts", self.set_view_side_y)

        for btn in (self.btn_view_top, self.btn_view_left, self.btn_view_right):
            layout.addWidget(btn)

        self.view_control_widget.adjustSize()
        self.view_control_widget.move(3, 3)
        self.view_control_widget.show()
        self.view_control_widget.raise_()

    def _pan_camera(self, delta_x: float, delta_y: float):
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
        self._skip_next_render_restore = True
        self.render()
        self._skip_next_render_restore = False

    def _rotate_camera(self, delta_x: float, delta_y: float) -> None:
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
            self._rotate_camera_around_z(angle_horiz)

        if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'ResetCameraClippingRange'):
            try:
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass
        self._skip_next_render_restore = True
        self.render()
        self._skip_next_render_restore = False

    def _rotate_camera_around_z(self, angle_deg: float) -> None:
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

    def _maximize_camera_view(self, add_padding: bool = False) -> None:
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

    def _add_scene_frame(self) -> None:
        if not hasattr(self.plotter, 'add_mesh'):
            return
        try:
            width = getattr(self.settings, 'width', 50.0)
            length = getattr(self.settings, 'length', 50.0)
            frame_height = getattr(self.settings, 'scene_frame_height', 0.1)
            frame_color = getattr(self.settings, 'scene_frame_color', '#999999')
            frame_offset = getattr(self.settings, 'scene_frame_offset', 2.5)

            x_min = -width / 2 - frame_offset
            x_max = width / 2 + frame_offset
            y_min = -length / 2 - frame_offset
            y_max = length / 2 + frame_offset

            corners = [
                (x_min, y_min, frame_height),
                (x_max, y_min, frame_height),
                (x_max, y_max, frame_height),
                (x_min, y_max, frame_height),
            ]

            lines = []
            for i in range(len(corners)):
                start = corners[i]
                end = corners[(i + 1) % len(corners)]
                line = pv.Line(start, end)
                lines.append(line)

            frame = pv.MultiBlock(lines)
            self.plotter.add_mesh(
                frame.merge(clean=True),
                name='scene_frame',
                color=frame_color,
                line_width=2.5,
                opacity=1.0,
                render_lines_as_tubes=True,
                reset_camera=False,
            )
        except Exception:
            pass

    def _create_view_button(self, orientation: str, tooltip: str, callback) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self.widget)
        button.setIcon(QtGui.QIcon(self._create_axis_pixmap(orientation)))
        button.setIconSize(QtCore.QSize(16, 16))
        button.setFixedSize(22, 22)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        if orientation in {'top', 'side_x', 'side_x_flip', 'side_y'}:
            button.setStyleSheet(
                "QToolButton {"
                "  border: 1px solid rgba(0, 0, 0, 120);"
                "  border-radius: 4px;"
                "  background-color: rgba(255, 255, 255, 210);"
                "}"
                "QToolButton:hover {"
                "  background-color: rgba(240, 240, 240, 210);"
                "}"
            )
        else:
            button.setStyleSheet("")
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _create_axis_pixmap(self, orientation: str, size: int = 28) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        center = QtCore.QPointF(size / 2, size / 2)
        radius = size * 0.35

        def draw_arrow(color: QtGui.QColor, start: QtCore.QPointF, end: QtCore.QPointF):
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(start, end)
            angle = math.atan2(end.y() - start.y(), end.x() - start.x())
            arrow_len = 5
            for delta in (math.pi / 6, -math.pi / 6):
                dx = arrow_len * math.cos(angle + delta)
                dy = arrow_len * math.sin(angle + delta)
                painter.drawLine(end, QtCore.QPointF(end.x() - dx, end.y() - dy))

        if orientation == 'iso':
            draw_arrow(QtGui.QColor('#d62728'), center, center + QtCore.QPointF(radius, 0))  # X
            draw_arrow(QtGui.QColor('#2ca02c'), center, center - QtCore.QPointF(radius * 0.5, radius * 0.9))  # Y
            draw_arrow(QtGui.QColor('#1f77b4'), center, center - QtCore.QPointF(0, radius))  # Z
        elif orientation == 'top':
            draw_arrow(QtGui.QColor('#d62728'), center, center + QtCore.QPointF(radius, 0))  # +X
            draw_arrow(QtGui.QColor('#2ca02c'), center, center - QtCore.QPointF(0, radius))  # +Y
        elif orientation == 'side_y':
            draw_arrow(QtGui.QColor('#d62728'), center, center + QtCore.QPointF(radius, 0))  # +X
            draw_arrow(QtGui.QColor('#1f77b4'), center, center - QtCore.QPointF(0, radius))  # +Z
        elif orientation == 'side_x':
            draw_arrow(QtGui.QColor('#2ca02c'), center, center + QtCore.QPointF(radius, 0))  # +Y
            draw_arrow(QtGui.QColor('#1f77b4'), center, center - QtCore.QPointF(0, radius))  # +Z
        elif orientation == 'side_x_flip':
            draw_arrow(QtGui.QColor('#2ca02c'), center, center - QtCore.QPointF(radius, 0))  # -Y
            draw_arrow(QtGui.QColor('#1f77b4'), center, center - QtCore.QPointF(0, radius))  # +Z

        painter.end()
        return pixmap

    # ------------------------------------------------------------------
    # Interne Hilfsfunktionen
    # ------------------------------------------------------------------
    def _configure_plotter(self):
        self.plotter.set_background('white')
        self.plotter.show_axes()
        try:
            self.plotter.disable_eye_dome_lighting()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.plotter.enable_anti_aliasing('msaa')
        except Exception:  # noqa: BLE001
            pass
        self.plotter.camera_position = 'iso'

    def _add_floor_plane(self):
        width = getattr(self.settings, 'width', 50)
        length = getattr(self.settings, 'length', 50)
        plane = pv.Plane(
            center=(0.0, 0.0, -0.05),
            direction=(0.0, 0.0, 1.0),
            i_size=width,
            j_size=length,
        )
        self.plotter.add_mesh(
            plane,
            name=self.FLOOR_NAME,
            color='#f5f5f5',
            opacity=0.4,
            reset_camera=False,
        )

    def _initialize_empty_colorbar(self):
        """Initialisiert eine leere Colorbar analog zur 2D-Version."""
        colorization_mode = getattr(self.settings, 'colorization_mode', 'Gradient')
        if colorization_mode not in {'Color step', 'Gradient'}:
            colorization_mode = 'Color step'
        # Beim Start greifen wir auf den aktuell konfigurierten Modus zurück, damit
        # Gradient/Step-Auswahl aus dem Settings-State direkt sichtbar ist.
        try:
            self._render_colorbar(colorization_mode, force=True)
        except Exception as exc:  # pragma: no cover - Farbskalen-Init sollte nicht abstürzen
            print(f"Warning: Leere 3D-Colorbar konnte nicht erstellt werden: {exc}")

    def _update_colorbar(self, colorization_mode: str):
        try:
            self._render_colorbar(colorization_mode)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Colorbar konnte nicht aktualisiert werden: {exc}")

    def _render_colorbar(self, colorization_mode: str, force: bool = False):
        """Aktualisiert die Colorbar.

        - Nutzt ein Caching der ScalarMappable-Instanz, um bei unverändertem Modus
          nur Norm/Cmap anzupassen.
        - `force=True` erzwingt einen kompletten Neuaufbau (z. B. beim Init).
        """
        cbar_min = float(self.settings.colorbar_range['min'])
        cbar_max = float(self.settings.colorbar_range['max'])
        cbar_step = float(self.settings.colorbar_range['step'])
        tick_step = float(self.settings.colorbar_range['tick_step'])

        if tick_step <= 0:
            tick_step = max((cbar_max - cbar_min) / 5.0, 1.0)

        cbar_ticks = np.arange(cbar_min, cbar_max + tick_step, tick_step)

        if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
            if force or self.cbar is None or self._colorbar_mappable is None:
                self.colorbar_ax.cla()

        is_step_mode = colorization_mode == 'Color step' and cbar_step > 0
        boundaries = None
        spacing = 'uniform'
        if is_step_mode:
            levels = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            if levels[-1] < cbar_max:
                levels = np.append(levels, cbar_max)
            if levels.size < 2:
                levels = np.array([cbar_min, cbar_max])

            cmap = cm.get_cmap('jet', len(levels) - 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            boundaries = levels
            spacing = 'proportional'
            mode_signature: tuple[str, tuple[float, ...] | None] = ('step', tuple(float(level) for level in levels))
        else:
            cmap = cm.get_cmap('jet')
            norm = Normalize(vmin=cbar_min, vmax=cbar_max)
            mode_signature = ('gradient', None)

        requires_new_colorbar = (
            force
            or self.cbar is None
            or self._colorbar_mappable is None
            or self._colorbar_mode is None
            or self._colorbar_mode[0] != mode_signature[0]
            or (
                mode_signature[1] is not None
                and (self._colorbar_mode[1] != mode_signature[1])
            )
        )

        if requires_new_colorbar:
            if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
                self.colorbar_ax.cla()
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            colorbar_kwargs = {
                'cax': self.colorbar_ax,
                'ticks': cbar_ticks,
            }
            if boundaries is not None:
                colorbar_kwargs.update({'boundaries': boundaries, 'spacing': spacing})

            self.cbar = plt.colorbar(sm, **colorbar_kwargs)
            self._colorbar_mappable = sm
            if boundaries is not None and hasattr(self.cbar, 'solids') and self.cbar.solids is not None:
                self.cbar.solids.set_edgecolor('face')
        else:
            assert self._colorbar_mappable is not None  # Für MyPy
            self._colorbar_mappable.set_cmap(cmap)
            self._colorbar_mappable.set_norm(norm)
            self.cbar.set_ticks(cbar_ticks)
            if boundaries is not None:
                self.cbar.boundaries = boundaries
                self.cbar.spacing = spacing
            self.cbar.update_normal(self._colorbar_mappable)

        if self.colorbar_ax is not None:
            self.colorbar_ax.tick_params(labelsize=7)
            self.colorbar_ax.set_position([0.1, 0.05, 0.1, 0.9])

        self._colorbar_mode = mode_signature

    @staticmethod
    def _has_valid_data(x, y, pressure) -> bool:
        if x is None or y is None or pressure is None:
            return False
        try:
            return len(x) > 0 and len(y) > 0 and len(pressure) > 0
        except TypeError:
            return False

    def _build_surface_mesh(self, x: np.ndarray, y: np.ndarray, scalars: np.ndarray) -> "pv.PolyData":
        xm, ym = np.meshgrid(x, y, indexing='xy')
        zeros = np.zeros_like(xm)
        grid = pv.StructuredGrid(xm, ym, zeros)

        flat_scalars = np.asarray(scalars, dtype=float).ravel(order='F')
        grid['plot_scalars'] = flat_scalars

        # Die SPL-Werte sollen die Fläche nicht mehr in Z-Richtung verformen,
        # daher verwenden wir direkt die extrahierte Oberfläche ohne Warp.
        return grid.extract_surface()

    @classmethod
    def _expand_axis_for_plot(
        cls,
        axis: np.ndarray,
        arg2: float | None = None,
        arg3: int | None = None,
    ) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        if axis.size <= 1:
            return axis

        # Neuer Modus: Punkte-pro-Meter übergeben (z. B. aus Texture-Resampling)
        if arg3 is None:
            points_per_meter = float(arg2 or 0.0)
            if points_per_meter <= 0:
                return axis

            start = float(axis[0])
            end = float(axis[-1])
            span = end - start
            if np.isclose(span, 0.0):
                return axis

            target_points = int(np.ceil(abs(span) * points_per_meter)) + 1
            if target_points <= axis.size:
                return axis

            return np.linspace(start, end, target_points, dtype=float)

        # Kompatibilitätsmodus: (axis, base_resolution, upscale_factor)
        base_resolution = float(arg2 or 0.0)
        upscale_factor = int(arg3 or 1)
        if upscale_factor <= 1:
            return axis

        expanded = [float(axis[0])]
        for idx in range(1, axis.size):
            start = float(axis[idx - 1])
            stop = float(axis[idx])
            if np.isclose(stop, start):
                segment = np.full(upscale_factor, start)
            else:
                segment = np.linspace(start, stop, upscale_factor + 1, dtype=float)[1:]
            expanded.extend(segment.tolist())

        return np.asarray(expanded, dtype=float)

    @staticmethod
    def _expand_scalars_for_plot(values: np.ndarray, factor: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if factor <= 1:
            return values
        if values.ndim != 2:
            return values

        expanded_cols = values.copy()
        if values.shape[1] > 1:
            repeated_cols = np.repeat(values[:, :-1], factor, axis=1)
            expanded_cols = np.concatenate((repeated_cols, values[:, -1:].copy()), axis=1)

        expanded_rows = expanded_cols
        if expanded_cols.shape[0] > 1:
            repeated_rows = np.repeat(expanded_cols[:-1, :], factor, axis=0)
            expanded_rows = np.concatenate((repeated_rows, expanded_cols[-1:, :].copy()), axis=0)

        return expanded_rows

    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except KeyError:
            pass


    # ------------------------------------------------------------------
    # Hilfsfunktionen für Farben/Skalierung
    # ------------------------------------------------------------------
    @staticmethod
    def _quantize_to_steps(values: np.ndarray, step: float) -> np.ndarray:
        if step <= 0:
            return values
        return np.round(values / step) * step

    @staticmethod
    def _fallback_cmap_to_lut(_self, cmap, n_colors: int | None = None, flip: bool = False):
        if isinstance(cmap, pv.LookupTable):
            lut = cmap
        else:
            lut = pv.LookupTable()
            lut.apply_cmap(cmap, n_values=n_colors or 256, flip=flip)
        if flip and isinstance(cmap, pv.LookupTable):
            lut.values[:] = lut.values[::-1]
        return lut

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

    @staticmethod
    def _resample_values_to_grid(
        values: np.ndarray,
        orig_x: np.ndarray,
        orig_y: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            return values

        orig_x = np.asarray(orig_x, dtype=float)
        orig_y = np.asarray(orig_y, dtype=float)
        target_x = np.asarray(target_x, dtype=float)
        target_y = np.asarray(target_y, dtype=float)

        if orig_x.size <= 1 or orig_y.size <= 1:
            return values

        if target_x.size == orig_x.size and target_y.size == orig_y.size:
            return values

        intermediate = np.empty((values.shape[0], target_x.size), dtype=float)
        for iy in range(values.shape[0]):
            intermediate[iy, :] = np.interp(target_x, orig_x, values[iy, :], left=values[iy, 0], right=values[iy, -1])

        resampled = np.empty((target_y.size, target_x.size), dtype=float)
        for ix in range(intermediate.shape[1]):
            resampled[:, ix] = np.interp(target_y, orig_y, intermediate[:, ix], left=intermediate[0, ix], right=intermediate[-1, ix])

        return resampled

    def _compute_overlay_signatures(self, settings, container) -> dict[str, tuple]:
        """Erzeugt robuste Vergleichs-Signaturen für jede Overlay-Kategorie.

        Hintergrund:
        - Ziel ist, Änderungsdetektion ohne tiefe Objektvergleiche.
        - Wir wandeln relevante Settings in flache Tupel um, die sich leicht
          vergleichen lassen und keine PyVista-Objekte enthalten.
        """
        def _to_tuple(sequence) -> tuple:
            if sequence is None:
                return tuple()
            try:
                iterable = list(sequence)
            except TypeError:
                return (float(sequence),)
            result = []
            for item in iterable:
                try:
                    value = float(item)
                except Exception:
                    result.append(None)
                else:
                    if np.isnan(value):
                        result.append(None)
                    else:
                        result.append(value)
            return tuple(result)

        axis_signature = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
        )

        walls_signature_tuple: tuple = ()

        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        speakers_signature: List[tuple] = []
        if isinstance(speaker_arrays, dict):
            for name in sorted(speaker_arrays.keys()):
                array = speaker_arrays[name]
                configuration = str(getattr(array, 'configuration', '') or '').lower()
                hide = bool(getattr(array, 'hide', False))
                xs = _to_tuple(getattr(array, 'source_position_x', None))
                ys = _to_tuple(
                    getattr(
                        array,
                        'source_position_calc_y',
                        getattr(array, 'source_position_y', None),
                    )
                )
                zs_stack = _to_tuple(getattr(array, 'source_position_z_stack', None))
                zs_flown = _to_tuple(getattr(array, 'source_position_z_flown', None))
                azimuth = _to_tuple(getattr(array, 'source_azimuth', None))
                angles = _to_tuple(getattr(array, 'source_angle', None))
                sources = (
                    tuple(xs),
                    tuple(ys),
                    tuple(zs_stack),
                    tuple(zs_flown),
                    tuple(azimuth),
                    tuple(angles),
                )
                speakers_signature.append(
                    (
                        str(name),
                        configuration,
                        hide,
                        sources,
                    )
                )
        speakers_signature_tuple = tuple(speakers_signature)

        impulse_points = getattr(settings, 'impulse_points', []) or []
        impulse_signature: List[tuple] = []
        for point in impulse_points:
            try:
                data = point['data']
                x_val, y_val = data
                impulse_signature.append((float(x_val), float(y_val)))
            except Exception:
                continue
        impulse_signature_tuple = tuple(sorted(impulse_signature))

        return {
            'axis': axis_signature,
            'walls': walls_signature_tuple,
            'speakers': speakers_signature_tuple,
            'impulse': impulse_signature_tuple,
        }


__all__ = ['DrawSPLPlot3D']


