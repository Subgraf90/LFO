"""Kern-Koordinations-Modul für den 3D-SPL-Plot.

Dieses Modul koordiniert alle Submodule und stellt die Haupt-API bereit.
"""

from __future__ import annotations

import types
from typing import Optional, Any

from PyQt5 import QtCore, QtWidgets

from Module_LFO.Modules_Init.ModuleBase import ModuleBase

# Import der Submodule
from . import utils
from . import interpolation
from . import camera
from . import picking
from . import colorbar
from . import ui_components
from . import rendering
from . import event_handler

# Import der Overlay-Module
from ..PlotSPL3DOverlays import renderer as overlay_renderer
from ..PlotSPL3DOverlays import axis as overlay_axis
from ..PlotSPL3DOverlays import surfaces as overlay_surfaces
from ..PlotSPL3DOverlays import speakers as overlay_speakers
from ..PlotSPL3DOverlays import speaker_geometry as overlay_speaker_geometry
from ..PlotSPL3DOverlays import impulse as overlay_impulse
from ..PlotSPL3DOverlays import line_utils as overlay_line_utils
from ..PlotSPL3DOverlays import caching as overlay_caching
from ..PlotSPL3DOverlays import ui_time_control

# Import für Plot-Geometrie
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    prepare_plot_geometry,
    build_surface_mesh,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:
    pv = None
    QtInteractor = None
    _PYVISTA_IMPORT_ERROR = exc
else:
    _PYVISTA_IMPORT_ERROR = None


class DrawSPLPlot3DCore(ModuleBase, QtCore.QObject):
    """Kern-Klasse für den 3D-SPL-Plot.
    
    Diese Klasse koordiniert alle Submodule und stellt die Haupt-API bereit.
    Sie ist eine vereinfachte Version der ursprünglichen DrawSPLPlot3D Klasse
    und nutzt die neuen Submodule.
    """
    
    SURFACE_NAME = "spl_surface"
    FLOOR_NAME = "spl_floor"
    UPSCALE_FACTOR = 24
    
    def __init__(self, parent_widget, settings, colorbar_ax):
        """Initialisiert den 3D-SPL-Plot.
        
        Args:
            parent_widget: Parent-Widget für den Plotter
            settings: Settings-Objekt
            colorbar_ax: Matplotlib Axes für die Colorbar
        """
        if QtInteractor is None or pv is None:
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
        self.container = None
        self.main_window = None
        
        # PyVista Plotter initialisieren
        self.plotter = QtInteractor(parent_widget)
        self.widget = self.plotter.interactor
        self.widget.setMouseTracking(True)
        
        # Deaktiviere VTK's Standard-Rotation
        self._disable_vtk_default_handlers()
        
        # Initialisiere Manager
        self._init_managers()
        
        # Initialisiere Event-Handler
        self._init_event_handler()
        
        # Initialisiere UI-Komponenten
        self._init_ui_components()
        
        # Initialisiere Overlay-Renderer
        self._init_overlay_renderer()
        
        # State-Variablen
        self.has_data = False
        self.surface_mesh = None
        self._surface_actors: dict[str, Any] = {}
        self._surface_texture_actors: dict[str, dict[str, Any]] = {}
        self._vertical_surface_meshes: dict[str, Any] = {}
        self._plot_geometry_cache = None
        self._time_mode_active = False
        self._time_mode_surface_cache: dict | None = None
        self._phase_mode_active = False
        self._colorbar_override: dict | None = None
        self._has_plotted_data = False
        self._did_initial_overlay_zoom = False
        self._last_overlay_signatures: dict[str, tuple] = {}
        self._is_rendering = False
        
        # Axis-Drag-Variablen
        self._axis_selected: Optional[str] = None
        self._axis_drag_active = False
        self._axis_drag_type: Optional[str] = None
        self._axis_drag_start_pos: Optional[QtCore.QPoint] = None
        self._axis_drag_start_value: Optional[float] = None
        
        # Pan/Rotate-Variablen
        self._pan_active = False
        self._pan_last_pos: Optional[QtCore.QPoint] = None
        self._rotate_active = False
        self._rotate_last_pos: Optional[QtCore.QPoint] = None
        
        # Verzögertes Rendering
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._delayed_render)
        self._pending_render = False
        
        # Initialisiere leere Szene
        self.initialize_empty_scene(preserve_camera=False)
    
    def _disable_vtk_default_handlers(self):
        """Deaktiviert VTK's Standard-Event-Handler."""
        try:
            if hasattr(self.plotter, 'iren') and self.plotter.iren is not None:
                interactor_style = self.plotter.iren.GetInteractorStyle()
                if interactor_style is not None:
                    def empty_handler(*args, **kwargs):
                        pass
                    
                    handlers_to_disable = [
                        'OnLeftButtonPress',
                        'OnLeftButtonRelease',
                        'OnLeftButtonDoubleClick',
                        'OnRightButtonDoubleClick',
                        'OnMiddleButtonDoubleClick',
                    ]
                    
                    for handler_name in handlers_to_disable:
                        if hasattr(interactor_style, handler_name):
                            try:
                                setattr(interactor_style, handler_name, 
                                       types.MethodType(lambda self, *args, **kwargs: None, interactor_style))
                            except Exception:
                                try:
                                    setattr(interactor_style, handler_name, empty_handler)
                                except Exception:
                                    pass
                    
                    if hasattr(interactor_style, 'OnMouseMove'):
                        try:
                            def disabled_mousemove(style_self):
                                pass
                            setattr(interactor_style, 'OnMouseMove', 
                                   types.MethodType(disabled_mousemove, interactor_style))
                        except Exception:
                            pass
        except Exception:
            pass
    
    def _init_managers(self):
        """Initialisiert die Manager-Klassen."""
        # Camera Manager
        def get_phase_mode():
            return self._phase_mode_active
        
        def get_time_mode():
            return self._time_mode_active
        
        self.camera_manager = camera.CameraManager(
            self.plotter,
            self.settings
        )
        
        # Colorbar Manager
        self.colorbar_manager = colorbar.ColorbarManager(
            self.colorbar_ax,
            self.settings,
            get_phase_mode,
            get_time_mode
        )
        
        # Picking Manager
        self.picking_manager = picking.PickingManager(
            self.plotter,
            self.settings,
            None,  # overlay_helper wird später gesetzt
            self.main_window
        )
    
    def _init_event_handler(self):
        """Initialisiert den Event-Handler."""
        self.event_handler = event_handler.EventHandler(
            self.widget,
            self.plotter,
            self.picking_manager._pick_speaker_at_position,
            self.picking_manager._pick_surface_at_position,
            self.picking_manager._pick_axis_line_at_position if hasattr(self.picking_manager, '_pick_axis_line_at_position') else lambda x: (None, None),
            self.picking_manager._select_speaker_in_treewidget,
            self.picking_manager._select_surface_in_treewidget,
            self._handle_axis_line_drag,
            self._handle_mouse_move_3d,
            self.camera_manager._pan_camera,
            self.camera_manager._rotate_camera,
            self.camera_manager._save_camera_state,
            self.main_window
        )
        self.widget.installEventFilter(self.event_handler)
    
    def _init_ui_components(self):
        """Initialisiert UI-Komponenten."""
        def create_view_button(orientation, tooltip, callback):
            return ui_components.create_view_button(
                self.widget,
                ui_components.create_axis_pixmap,
                orientation,
                tooltip,
                callback
            )
        
        self.view_control_widget = ui_components.setup_view_controls(
            self.widget,
            create_view_button,
            self.camera_manager.set_view_top,
            self.camera_manager.set_view_side_x,
            self.camera_manager.set_view_side_y
        )
        
        def time_slider_change_handler(value):
            if hasattr(self, '_time_slider_callback') and callable(self._time_slider_callback):
                self._time_slider_callback(int(value))
        
        self.time_control = ui_components.setup_time_control(
            self.widget,
            time_slider_change_handler
        )
        if self.time_control:
            self.time_control.hide()
    
    def _init_overlay_renderer(self):
        """Initialisiert den Overlay-Renderer."""
        # Overlay-State
        self.overlay_actor_names: list[str] = []
        self.overlay_counter = [0]  # Liste für mutable counter
        self._category_actors: dict[str, list[str]] = {}
        self._speaker_actor_cache: dict = {}
        
        # Helper-Funktionen für Overlay-Renderer
        def add_overlay_mesh(mesh, **kwargs):
            return overlay_renderer.add_overlay_mesh(
                self.plotter,
                mesh,
                self.overlay_counter,
                self.overlay_actor_names,
                self._category_actors,
                **kwargs
            )
        
        def clear_category(category):
            return overlay_renderer.clear_category(
                self.plotter,
                category,
                self.overlay_actor_names,
                self._category_actors,
                self._speaker_actor_cache
            )
        
        # Overlay-Manager (vereinfacht)
        self.overlay_helper = type('OverlayHelper', (), {
            'plotter': self.plotter,
            'pv': pv,
            'overlay_actor_names': self.overlay_actor_names,
            'overlay_counter': self.overlay_counter,
            '_category_actors': self._category_actors,
            '_speaker_actor_cache': self._speaker_actor_cache,
            '_add_overlay_mesh': add_overlay_mesh,
            'clear_category': clear_category,
            'clear': lambda: overlay_renderer.clear_overlays(
                self.plotter,
                self.overlay_actor_names,
                self._category_actors
            ),
        })()
        
        # Update Picking Manager mit Overlay-Helper
        self.picking_manager.overlay_helper = self.overlay_helper
    
    def initialize_empty_scene(self, preserve_camera: bool = True):
        """Initialisiert eine leere Szene.
        
        Args:
            preserve_camera: Ob die Kameraposition erhalten bleiben soll
        """
        if preserve_camera:
            camera_state = self.camera_manager._camera_state or self.camera_manager._capture_camera()
        else:
            camera_state = None
            self._has_plotted_data = False
        
        self.plotter.clear()
        if hasattr(self.overlay_helper, 'clear'):
            self.overlay_helper.clear()
        self.has_data = False
        self.surface_mesh = None
        self._last_overlay_signatures = {}
        
        # Konfiguriere Plotter
        self._configure_plotter(configure_camera=not preserve_camera)
        
        if camera_state is not None:
            self.camera_manager._restore_camera(camera_state)
        else:
            self.camera_manager.set_view_top()
            camera_state = self.camera_manager._camera_state
            if camera_state is not None:
                self.camera_manager._restore_camera(camera_state)
        
        self.colorbar_manager.initialize_empty_colorbar()
        self.render()
        self.camera_manager._save_camera_state()
        if self.time_control is not None:
            self.time_control.hide()
    
    def _configure_plotter(self, configure_camera: bool = True):
        """Konfiguriert den Plotter.
        
        Args:
            configure_camera: Ob die Kamera konfiguriert werden soll
        """
        self.plotter.set_background('white')
        self.plotter.show_axes()
        try:
            self.plotter.disable_eye_dome_lighting()
        except Exception:  # noqa: BLE001
            pass
        # Anti-Aliasing nach globaler Vorgabe konfigurieren
        try:
            from Module_LFO.Modules_Plot.PlotSPL3D import PYVISTA_AA_MODE
            aa_mode = PYVISTA_AA_MODE.lower() if isinstance(PYVISTA_AA_MODE, str) else ""
            if aa_mode in {"", "none", "off"}:
                # Explizit deaktivieren, falls Plotter das unterstützt
                if hasattr(self.plotter, "disable_anti_aliasing"):
                    self.plotter.disable_anti_aliasing()  # type: ignore[call-arg]
            else:
                self.plotter.enable_anti_aliasing(aa_mode)
        except Exception:  # noqa: BLE001
            pass
        # Nur Kamera-Position setzen, wenn gewünscht (nicht bei preserve_camera=True)
        if configure_camera:
            self.plotter.camera_position = 'iso'
    
    def _handle_axis_line_drag(self, current_pos: QtCore.QPoint) -> None:
        """Behandelt das Drag einer Achsenlinie und aktualisiert die Position.
        
        Args:
            current_pos: Aktuelle Mausposition
        """
        if not hasattr(self, '_axis_drag_active') or not self._axis_drag_active:
            return
        if not hasattr(self, '_axis_drag_type') or self._axis_drag_type is None:
            return
        
        try:
            renderer = self.plotter.renderer
            if renderer is None:
                return
            
            size = self.widget.size()
            if size.width() <= 0 or size.height() <= 0:
                return
            
            # Konvertiere 2D-Mausposition zu 3D-Weltkoordinaten
            display_x = float(current_pos.x())
            display_y = float(size.height() - current_pos.y())
            
            # Konvertiere Display zu World-Koordinaten
            renderer.SetDisplayPoint(display_x, display_y, 0.0)
            renderer.DisplayToWorld()
            world_point = renderer.GetWorldPoint()
            if len(world_point) >= 4 and world_point[3] != 0:
                world_x = world_point[0] / world_point[3]
                world_y = world_point[1] / world_point[3]
                
                # Für X-Achse: verwende world_y als neue Y-Position
                # Für Y-Achse: verwende world_x als neue X-Position
                if self._axis_drag_type == 'x':
                    # X-Achse wird gedraggt - ändere Y-Position
                    new_value = world_y
                    # Begrenze auf erlaubten Bereich
                    min_allowed = -self.settings.length / 2
                    max_allowed = self.settings.length / 2
                    new_value = max(min_allowed, min(max_allowed, new_value))
                    # Runde auf ganze Zahl
                    new_value = int(round(new_value))
                    
                    if self.settings.position_y_axis != new_value:
                        self.settings.position_y_axis = new_value
                        # Aktualisiere Overlays (Achsenlinien neu zeichnen)
                        if hasattr(self, 'update_overlays') and self.container:
                            self.update_overlays(self.settings, self.container)
                        # Aktualisiere Berechnungen
                        if self.main_window and hasattr(self.main_window, 'update_speaker_array_calculations'):
                            self.main_window.update_speaker_array_calculations()
                else:  # self._axis_drag_type == 'y'
                    # Y-Achse wird gedraggt - ändere X-Position
                    new_value = world_x
                    # Begrenze auf erlaubten Bereich
                    min_allowed = -self.settings.width / 2
                    max_allowed = self.settings.width / 2
                    new_value = max(min_allowed, min(max_allowed, new_value))
                    # Runde auf ganze Zahl
                    new_value = int(round(new_value))
                    
                    if self.settings.position_x_axis != new_value:
                        self.settings.position_x_axis = new_value
                        # Aktualisiere Overlays (Achsenlinien neu zeichnen)
                        if hasattr(self, 'update_overlays') and self.container:
                            self.update_overlays(self.settings, self.container)
                        # Aktualisiere Berechnungen
                        if self.main_window and hasattr(self.main_window, 'update_speaker_array_calculations'):
                            self.main_window.update_speaker_array_calculations()
        except Exception:
            pass
    
    def _handle_mouse_move_3d(self, mouse_pos: QtCore.QPoint) -> None:
        """Behandelt Mouse-Move im 3D-Plot und zeigt Mausposition an.
        
        Args:
            mouse_pos: Mausposition
        """
        try:
            renderer = self.plotter.renderer
            if renderer is None:
                return
            
            # Hole 3D-Koordinaten der Mausposition
            size = self.widget.size()
            if size.width() <= 0 or size.height() <= 0:
                return
            
            x_norm = mouse_pos.x() / size.width()
            y_norm = 1.0 - (mouse_pos.y() / size.height())  # VTK Y ist invertiert
            
            # Schnell: Berechne X/Y aus Schnittpunkt mit Z=0 Ebene
            render_size = renderer.GetSize()
            renderer.SetDisplayPoint(x_norm * render_size[0], y_norm * render_size[1], 0.0)
            renderer.DisplayToWorld()
            world_point_near = renderer.GetWorldPoint()
            
            renderer.SetDisplayPoint(x_norm * render_size[0], y_norm * render_size[1], 1.0)
            renderer.DisplayToWorld()
            world_point_far = renderer.GetWorldPoint()
            
            if world_point_near is None or world_point_far is None or len(world_point_near) < 4:
                return
            
            # Berechne Schnittpunkt mit Z=0 Ebene (schnell für X/Y)
            if abs(world_point_near[3]) > 1e-6 and abs(world_point_far[3]) > 1e-6:
                import numpy as np
                ray_start = np.array([
                    world_point_near[0] / world_point_near[3],
                    world_point_near[1] / world_point_near[3],
                    world_point_near[2] / world_point_near[3]
                ])
                ray_end = np.array([
                    world_point_far[0] / world_point_far[3],
                    world_point_far[1] / world_point_far[3],
                    world_point_far[2] / world_point_far[3]
                ])
                
                ray_dir = ray_end - ray_start
                if abs(ray_dir[2]) > 1e-6:
                    t = -ray_start[2] / ray_dir[2]
                    intersection = ray_start + t * ray_dir
                    x_pos = float(intersection[0])
                    y_pos = float(intersection[1])
                else:
                    return
            else:
                return
            
            # Baue Text zusammen
            text = f"3D-Plot:\nX: {x_pos:.2f} m\nY: {y_pos:.2f} m"
            
            # Aktualisiere Label über main_window
            if self.main_window and hasattr(self.main_window, 'ui') and hasattr(self.main_window.ui, 'mouse_position_label'):
                self.main_window.ui.mouse_position_label.setText(text)
        except Exception:
            pass
    
    def render(self):
        """Erzwingt ein Rendering der Szene."""
        if self._is_rendering:
            return
        self._is_rendering = True
        try:
            self.plotter.render()
        finally:
            self._is_rendering = False
    
    def _delayed_render(self):
        """Verzögertes Rendering für Batch-Updates."""
        if self._pending_render:
            self.render()
            self._pending_render = False
    
    def _schedule_render(self):
        """Plant ein verzögertes Rendering."""
        self._pending_render = True
        if not self._render_timer.isActive():
            self._render_timer.start(500)  # 500ms Delay
    
    def update_spl_plot(
        self,
        sound_field_x,
        sound_field_y,
        sound_field_pressure,
        colorization_mode: str = "Gradient",
    ):
        """Aktualisiert die SPL-Fläche.
        
        Args:
            sound_field_x: X-Koordinaten
            sound_field_y: Y-Koordinaten
            sound_field_pressure: Druck-Werte
            colorization_mode: Farbmodus ('Gradient' oder 'Color step')
        """
        camera_state = self.camera_manager._camera_state or self.camera_manager._capture_camera()
        
        # Validierung
        if not utils.has_valid_data(sound_field_x, sound_field_y, sound_field_pressure):
            self.initialize_empty_scene(preserve_camera=True)
            return
        
        import numpy as np
        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
            prepare_plot_geometry,
            build_surface_mesh,
        )
        
        x = np.asarray(sound_field_x, dtype=float)
        y = np.asarray(sound_field_y, dtype=float)
        
        try:
            pressure = np.asarray(sound_field_pressure, dtype=float)
        except Exception:  # noqa: BLE001
            self.initialize_empty_scene(preserve_camera=True)
            return
        
        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'
        time_mode = plot_mode == 'SPL over time'
        self._phase_mode_active = phase_mode
        self._time_mode_active = time_mode
        
        if pressure.ndim != 2:
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
        
        self._colorbar_override = None
        
        # Berechne Plot-Werte basierend auf Modus
        if time_mode:
            pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                self.initialize_empty_scene(preserve_camera=True)
                return
            plot_values = pressure
        elif phase_mode:
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                self.initialize_empty_scene(preserve_camera=True)
                return
            phase_values = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            plot_values = phase_values
        else:
            pressure_2d = np.nan_to_num(np.abs(pressure), nan=0.0, posinf=0.0, neginf=0.0)
            pressure_2d = np.clip(pressure_2d, 1e-12, None)
            # Verwende functions.mag2db wenn verfügbar
            if hasattr(self, 'functions') and hasattr(self.functions, 'mag2db'):
                spl_db = self.functions.mag2db(pressure_2d)
            else:
                # Fallback: direkte Berechnung
                spl_db = 20.0 * np.log10(pressure_2d / 2e-5)
            finite_mask = np.isfinite(spl_db)
            if not np.any(finite_mask):
                self.initialize_empty_scene(preserve_camera=True)
                return
            plot_values = spl_db
        
        # Colorbar-Parameter
        colorbar_params = self.colorbar_manager._get_colorbar_params(phase_mode)
        cbar_min = colorbar_params['min']
        cbar_max = colorbar_params['max']
        cbar_step = colorbar_params['step']
        tick_step = colorbar_params['tick_step']
        self._colorbar_override = colorbar_params
        
        # Speichere ursprüngliche Werte
        original_plot_values = plot_values.copy() if hasattr(plot_values, 'copy') else plot_values
        
        # Clipping
        if time_mode:
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
        elif phase_mode:
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
        else:
            plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)
        
        # Bereite Plot-Geometrie vor
        try:
            geometry = prepare_plot_geometry(
                x,
                y,
                plot_values,
                settings=self.settings,
                container=self.container,
                default_upscale=self.UPSCALE_FACTOR,
            )
        except Exception:
            self.initialize_empty_scene(preserve_camera=True)
            rendering.clear_vertical_spl_surfaces(self.plotter, self._vertical_surface_meshes)
            return
        
        plot_x = geometry.plot_x
        plot_y = geometry.plot_y
        plot_values = geometry.plot_values
        z_coords = geometry.z_coords
        
        # Cache Plot-Geometrie
        self._plot_geometry_cache = {
            'plot_x': plot_x.copy() if hasattr(plot_x, 'copy') else plot_x,
            'plot_y': plot_y.copy() if hasattr(plot_y, 'copy') else plot_y,
            'source_x': geometry.source_x.copy() if hasattr(geometry.source_x, 'copy') else geometry.source_x,
            'source_y': geometry.source_y.copy() if hasattr(geometry.source_y, 'copy') else geometry.source_y,
        }
        
        # Quantisierung für Color-Step-Modus
        is_step_mode = colorization_mode == 'Color step' and cbar_step > 0
        if is_step_mode:
            scalars = utils.quantize_to_steps(plot_values, cbar_step)
        else:
            scalars = plot_values
        
        # Erstelle Surface-Mesh
        mesh = build_surface_mesh(
            plot_x,
            plot_y,
            scalars,
            z_coords=z_coords,
            surface_mask=geometry.surface_mask,
            pv_module=pv,
            settings=self.settings,
            container=self.container,
            source_x=geometry.source_x,
            source_y=geometry.source_y,
            source_scalars=None,
        )
        
        # Colormap
        if time_mode:
            cmap_object = 'RdBu_r'
        elif phase_mode:
            from Module_LFO.Modules_Plot.PlotSPL3D import DrawSPLPlot3D
            cmap_object = DrawSPLPlot3D.PHASE_CMAP
        else:
            cmap_object = 'jet'
        
        # Speichere Mesh für Click-Handling
        if mesh is not None and mesh.n_points > 0:
            self.surface_mesh = mesh.copy(deep=True)
        
        # Entferne alte Mesh-Actors
        base_actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
        if base_actor is not None:
            try:
                self.plotter.remove_actor(base_actor)
            except Exception:
                pass
        
        # Render Surfaces als Texturen
        rendering.render_surfaces_textured(
            self.plotter,
            pv,
            self.settings,
            self.container,
            geometry,
            original_plot_values,
            cbar_min,
            cbar_max,
            cmap_object,
            colorization_mode,
            cbar_step,
            self._surface_texture_actors,
            self.SURFACE_NAME,
            interpolation.bilinear_interpolate_grid,
            interpolation.nearest_interpolate_grid,
            utils.quantize_to_steps,
        )
        
        # Update vertikale Surfaces
        rendering.update_vertical_spl_surfaces(
            self.plotter,
            pv,
            self.settings,
            self.container,
            self._vertical_surface_meshes,
            lambda: rendering.clear_vertical_spl_surfaces(self.plotter, self._vertical_surface_meshes),
            utils.quantize_to_steps,
            rendering.get_vertical_color_limits,
            self.UPSCALE_FACTOR,
        )
        
        # Update Colorbar
        self.colorbar_manager.update_colorbar(colorization_mode, tick_step)
        
        # Restore Camera
        if camera_state is not None:
            self.camera_manager._restore_camera(camera_state)
        
        self.has_data = True
        self._has_plotted_data = True
        self.render()
    
    def update_overlays(self, settings, container):
        """Aktualisiert Zusatzobjekte (Achsen, Lautsprecher, Messpunkte).
        
        Args:
            settings: Settings-Objekt
            container: Container-Objekt
        """
        import time
        t_start = time.perf_counter()
        
        # Speichere Container-Referenz
        self.container = container
        
        # Wenn kein gültiger calculation_spl-Container vorhanden ist, entferne vertikale SPL-Flächen
        if not hasattr(container, "calculation_spl"):
            rendering.clear_vertical_spl_surfaces(self.plotter, self._vertical_surface_meshes)
        
        # Berechne Signaturen
        signatures = utils.compute_overlay_signatures(settings, container, self.plotter)
        previous = self._last_overlay_signatures or {}
        if not previous:
            categories_to_refresh = set(signatures.keys())
        else:
            categories_to_refresh = {
                key for key, value in signatures.items() if value != previous.get(key)
            }
        
        if not categories_to_refresh:
            self._last_overlay_signatures = signatures
            if not hasattr(self, '_rotate_active') or (not self._rotate_active and not getattr(self, '_pan_active', False)):
                self.camera_manager._save_camera_state()
            return
        
        # Zeichne Overlays
        # Hinweis: draw_axis_lines ist noch nicht vollständig migriert,
        # daher verwenden wir die ursprüngliche overlay_helper Klasse
        if 'axis' in categories_to_refresh:
            # Verwende overlay_helper für axis (noch nicht vollständig migriert)
            if hasattr(self, 'overlay_helper') and hasattr(self.overlay_helper, 'draw_axis_lines'):
                self.overlay_helper.draw_axis_lines(settings, selected_axis=getattr(self, '_axis_selected', None))
            else:
                # Fallback: Verwende axis-Modul-Funktionen wenn verfügbar
                try:
                    from ..PlotSPL3DOverlays import axis as overlay_axis_module
                    # Erstelle Helper-Funktionen
                    def add_mesh_wrapper(mesh, **kwargs):
                        return overlay_renderer.add_overlay_mesh(
                            self.plotter,
                            mesh,
                            self.overlay_counter,
                            self.overlay_actor_names,
                            self._category_actors,
                            **kwargs
                        )
                    def clear_category_wrapper(category):
                        return overlay_renderer.clear_category(
                            self.plotter,
                            category,
                            self.overlay_actor_names,
                            self._category_actors,
                            self._speaker_actor_cache
                        )
                    # Zeichne Achsenflächen
                    x_axis = float(getattr(settings, 'position_x_axis', 0.0))
                    y_axis = float(getattr(settings, 'position_y_axis', 0.0))
                    length = float(getattr(settings, 'length', 0.0))
                    width = float(getattr(settings, 'width', 0.0))
                    overlay_axis_module.draw_axis_planes(
                        x_axis,
                        y_axis,
                        length,
                        width,
                        settings,
                        pv,
                        add_mesh_wrapper,
                        overlay_axis_module.get_max_surface_dimension,
                    )
                except Exception:
                    pass
        
        if 'surfaces' in categories_to_refresh:
            # Prüfe ob SPL-Daten vorhanden sind
            create_empty_plot_surfaces = False
            try:
                has_spl_data = False
                if container is not None and hasattr(container, 'calculation_spl'):
                    calc_spl = container.calculation_spl
                    if isinstance(calc_spl, dict) and calc_spl.get('sound_field_p') is not None:
                        has_spl_data = True
                if not has_spl_data:
                    if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                        spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                        spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                        if spl_surface_actor is not None or spl_floor_actor is not None:
                            has_spl_data = True
                if not has_spl_data:
                    create_empty_plot_surfaces = True
            except Exception:
                create_empty_plot_surfaces = False
            
            overlay_surfaces.draw_surfaces(
                self.plotter,
                pv,
                settings,
                container,
                lambda mesh, **kwargs: overlay_renderer.add_overlay_mesh(
                    self.plotter,
                    mesh,
                    self.overlay_counter,
                    self.overlay_actor_names,
                    self._category_actors,
                    **kwargs
                ),
                lambda category: overlay_renderer.clear_category(
                    self.plotter,
                    category,
                    self.overlay_actor_names,
                    self._category_actors,
                    self._speaker_actor_cache
                ),
                self._category_actors,
                self.overlay_actor_names,
                getattr(self.overlay_helper, '_last_surfaces_state', None) if hasattr(self, 'overlay_helper') else None,
                create_empty_plot_surfaces,
            )
        
        if 'speakers' in categories_to_refresh:
            cabinet_lookup = overlay_speakers.build_cabinet_lookup(container)
            # Erstelle Helper-Funktionen für build_speaker_geometries
            def build_speaker_geometries_wrapper(speaker_array, idx, x, y, z, cabinet_lookup, container):
                return overlay_speaker_geometry.build_speaker_geometries(
                    speaker_array,
                    idx,
                    x,
                    y,
                    z,
                    cabinet_lookup,
                    pv,
                    overlay_caching.get_box_template,
                    overlay_caching.compute_box_face_indices,
                    getattr(self.overlay_helper, '_box_face_cache', {}) if hasattr(self, 'overlay_helper') else {},
                    overlay_speaker_geometry.resolve_cabinet_entries,
                    overlay_speaker_geometry.apply_flown_cabinet_shape,
                    overlay_speaker_geometry.safe_float,
                    overlay_speaker_geometry.array_value,
                    overlay_speaker_geometry.float_sequence,
                    overlay_speaker_geometry.sequence_length,
                    overlay_speaker_geometry.get_speaker_name,
                )
            
            def update_speaker_actor_wrapper(actor_name, mesh, exit_face_index, body_color, exit_color):
                # Update Speaker Actor (vereinfacht)
                try:
                    actor = self.plotter.renderer.actors.get(actor_name)
                    if actor is None:
                        return
                    # Vereinfachte Update-Logik
                    if hasattr(mesh, 'cell_data') and 'speaker_face' in mesh.cell_data:
                        mapper = getattr(actor, 'mapper', None)
                        if mapper is not None:
                            mapper.array_name = 'speaker_face'
                            mapper.scalar_range = (0, 1)
                            if hasattr(self.plotter, '_cmap_to_lut'):
                                mapper.lookup_table = self.plotter._cmap_to_lut([body_color, exit_color])
                except Exception:
                    pass
            
            overlay_speakers.draw_speakers(
                self.plotter,
                pv,
                settings,
                container,
                cabinet_lookup,
                build_speaker_geometries_wrapper,
                lambda mesh, **kwargs: overlay_renderer.add_overlay_mesh(
                    self.plotter,
                    mesh,
                    self.overlay_counter,
                    self.overlay_actor_names,
                    self._category_actors,
                    **kwargs
                ),
                lambda category: overlay_renderer.clear_category(
                    self.plotter,
                    category,
                    self.overlay_actor_names,
                    self._category_actors,
                    self._speaker_actor_cache
                ),
                self._speaker_actor_cache,
                self._category_actors,
                utils.to_float_array,
                update_speaker_actor_wrapper,
                overlay_caching.speaker_signature_from_mesh,
                lambda name: self.plotter.remove_actor(name) if name in self.plotter.renderer.actors else None,
            )
        else:
            # Update Highlights auch wenn sich Speaker-Definitionen nicht geändert haben
            overlay_speakers.update_speaker_highlights(settings, self._speaker_actor_cache, self.plotter)
        
        if 'impulse' in categories_to_refresh:
            overlay_impulse.draw_impulse_points(
                self.plotter,
                pv,
                settings,
                lambda mesh, **kwargs: overlay_renderer.add_overlay_mesh(
                    self.plotter,
                    mesh,
                    self.overlay_counter,
                    self.overlay_actor_names,
                    self._category_actors,
                    **kwargs
                ),
                lambda category: overlay_renderer.clear_category(
                    self.plotter,
                    category,
                    self.overlay_actor_names,
                    self._category_actors,
                    self._speaker_actor_cache
                ),
                self._category_actors,
                getattr(self.overlay_helper, '_last_impulse_state', None) if hasattr(self, 'overlay_helper') else None,
            )
        
        self._last_overlay_signatures = signatures
        
        # Verzögertes Rendering
        self._schedule_render()
        if not hasattr(self, '_rotate_active') or (not self._rotate_active and not getattr(self, '_pan_active', False)):
            self.camera_manager._save_camera_state()

