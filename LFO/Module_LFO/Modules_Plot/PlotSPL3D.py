"""PyVista-basierter SPL-Plot (3D)"""

from __future__ import annotations

import math
import os
import time
import types
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Plot.PlotSPL3DOverlays import SPL3DOverlayRenderer, SPLTimeControlBar
from Module_LFO.Modules_Calculate\
    .SurfaceGeometryCalculator import (
    SurfaceDefinition,
    DEBUG_SURFACE_GEOMETRY,
    build_surface_mesh,
    build_vertical_surface_mesh,
    prepare_plot_geometry,
    prepare_vertical_plot_geometry,
    VerticalPlotGeometry,
    build_full_floor_mesh,
    clip_floor_with_surfaces,
    derive_surface_plane,
)

DEBUG_SPL_DUMP = bool(int(os.environ.get("LFO_DEBUG_SPL", "0")))
DEBUG_OVERLAY_PERF = bool(int(os.environ.get("LFO_DEBUG_OVERLAY_PERF", "1")))
DEBUG_SURFACE_REF = bool(int(os.environ.get("LFO_DEBUG_SURFACE_REF", "1")))

# Steuerung des Anti-Aliasing-Modus für PyVista:
# Mögliche Werte (abhängig von PyVista/VTK-Version):
#   - "msaa"  : Multi-Sample Anti-Aliasing (Standard, guter Kompromiss)
#   - "ssaa"  : Super-Sample Anti-Aliasing (höhere Qualität, mehr Last)
#   - "fxaa"  : Fast Approximate Anti-Aliasing (shaderbasiert)
#   - "none" / "off" / "" : Anti-Aliasing komplett deaktivieren
#
# Diese Variable kann bei Bedarf direkt angepasst werden, ohne weiteren Code zu ändern.
PYVISTA_AA_MODE = "ssaa"

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
    PHASE_CMAP = LinearSegmentedColormap.from_list(
        "phase_wheel",
        [
            (0.0, "#2ecc71"),          # 0°
            (38 / 180.0, "#8bd852"),   # 38°
            (90 / 180.0, "#f4e34d"),   # 90°
            (120 / 180.0, "#f7b731"),  # 120°
            (150 / 180.0, "#eb8334"),  # 150°
            (1.0, "#d64545"),          # 180°
        ],
    )

    SURFACE_NAME = "spl_surface"
    FLOOR_NAME = "spl_floor"
    UPSCALE_FACTOR = 24  # Erhöht die Anzahl der Grafikpunkte für schärferen Plot (mit Interpolation)

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
        self.container = None  # Wird bei update_overlays gesetzt

        self.plotter = QtInteractor(parent_widget)
        self.widget = self.plotter.interactor
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)
        self._pan_active = False
        self._pan_last_pos: Optional[QtCore.QPoint] = None
        self._rotate_active = False
        self._rotate_last_pos: Optional[QtCore.QPoint] = None
        self._click_start_pos: Optional[QtCore.QPoint] = None  # Für Click-Erkennung (vs. Drag)
        self._last_click_time: Optional[float] = None  # Zeitpunkt des letzten Klicks (für Doppelklick-Erkennung)
        self._last_click_pos: Optional[QtCore.QPoint] = None  # Position des letzten Klicks
        self._camera_state: Optional[dict[str, object]] = None
        self._skip_next_render_restore = False
        self._camera_debug_enabled = self._init_camera_debug_flag()
        self._camera_debug_counter = 0
        self._phase_mode_active = False
        self._colorbar_override: dict | None = None
        self._has_plotted_data = False  # Flag: ob bereits ein Plot mit Daten durchgeführt wurde
        # Flag: ob der initiale Zoom auf das Default-Surface bereits gesetzt wurde
        self._did_initial_overlay_zoom = False

        if not hasattr(self.plotter, '_cmap_to_lut'):
            self.plotter._cmap_to_lut = types.MethodType(self._fallback_cmap_to_lut, self.plotter)  # type: ignore[attr-defined]

        self.overlay_helper = SPL3DOverlayRenderer(self.plotter, pv)

        self.cbar = None
        self.has_data = False
        self.surface_mesh = None  # type: pv.DataSet | None
        # Einzelne Actors pro horizontaler Surface (SPL-Flächen, Mesh-Modus)
        self._surface_actors: dict[str, Any] = {}
        # Textur-Actors pro horizontaler Surface (Texture-Modus)
        # Struktur: {surface_id: {'actor': actor, 'metadata': {...}}}
        self._surface_texture_actors: dict[str, dict[str, Any]] = {}
        self._plot_geometry_cache = None  # Cache für Plot-Geometrie (plot_x, plot_y)
        # Zusätzliche SPL-Meshes für senkrechte Flächen (werden über calculation_spl gespeist)
        self._vertical_surface_meshes: dict[str, Any] = {}
        self.time_control: Optional[SPLTimeControlBar] = None
        self._time_slider_callback = None
        self._time_mode_active = False
        self._time_mode_surface_cache: dict | None = None

        # Colorbar-Caching: merkt sich den ScalarMappable sowie den Modus,
        # damit wir bei Area-Updates keine neue Colorbar erzeugen müssen.
        self._colorbar_mappable = None
        self._colorbar_mode: Optional[tuple[str, tuple[float, ...] | None]] = None

        # Overlay-Differenzial: Signaturen pro Kategorie (Achsen, Wände, etc.),
        # um gezielt neu zu zeichnen und Renderzeit zu sparen.
        self._last_overlay_signatures: dict[str, tuple] = {}

        # Guard gegen reentrante Render-Aufrufe (z. B. Kamera-Callbacks).
        self._is_rendering = False

        # Verzögertes Rendering: Timer für Batch-Updates (500ms)
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._delayed_render)
        self._pending_render = False

        # initialize_empty_scene(preserve_camera=False) setzt bereits die Top-Ansicht via set_view_top()
        # daher ist der explizite Aufruf hier nicht nötig und würde den Zoom unnötig zweimal ändern
        self.initialize_empty_scene(preserve_camera=False)
        self._setup_view_controls()
        self._setup_time_control()

    def eventFilter(self, obj, event):  # noqa: PLR0911
        if obj is self.widget:
            etype = event.type()
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._rotate_active = True
                self._rotate_last_pos = QtCore.QPoint(event.pos())
                self._click_start_pos = QtCore.QPoint(event.pos())  # Merke Start-Position für Click-Erkennung
                self.widget.setCursor(QtCore.Qt.OpenHandCursor)
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                # Prüfe ob es ein Klick war (kein Drag)
                click_pos = QtCore.QPoint(event.pos())
                is_click = (
                    self._click_start_pos is not None
                    and abs(click_pos.x() - self._click_start_pos.x()) < 3
                    and abs(click_pos.y() - self._click_start_pos.y()) < 3
                )
                
                print(f"[DEBUG Click] MouseButtonRelease: pos=({click_pos.x()}, {click_pos.y()}), is_click={is_click}, start_pos={self._click_start_pos}")
                
                self._rotate_active = False
                self._rotate_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                
                # Prüfe ob es ein Doppelklick ist
                is_double_click = False
                if is_click and self._last_click_time is not None and self._last_click_pos is not None:
                    current_time = time.time()
                    time_diff = current_time - self._last_click_time
                    pos_diff = (
                        abs(click_pos.x() - self._last_click_pos.x()) < 5
                        and abs(click_pos.y() - self._last_click_pos.y()) < 5
                    )
                    # Doppelklick wenn innerhalb von 300ms und ähnlicher Position
                    if time_diff < 0.3 and pos_diff:
                        is_double_click = True
                        print(f"[DEBUG Click] Double click detected (time_diff={time_diff*1000:.0f}ms)")
                
                # Wenn es ein Klick war (kein Drag) und kein Doppelklick, prüfe ob eine Surface geklickt wurde
                if is_click and not is_double_click:
                    print(f"[DEBUG Click] Calling _handle_surface_click")
                    self._handle_surface_click(click_pos)
                    # Speichere Zeitpunkt und Position für Doppelklick-Erkennung
                    self._last_click_time = time.time()
                    self._last_click_pos = QtCore.QPoint(click_pos)
                elif is_double_click:
                    # Bei Doppelklick: Ignoriere den zweiten Klick (verhindert doppelte Surface-Auswahl)
                    print(f"[DEBUG Click] Double click ignored")
                    self._last_click_time = None
                    self._last_click_pos = None
                
                self._click_start_pos = None
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

    def _handle_surface_click(self, click_pos: QtCore.QPoint) -> None:
        """Behandelt einen Klick auf eine Surface im 3D-Plot und wählt das entsprechende Item im TreeWidget aus."""
        try:
            print(f"[DEBUG Click] _handle_surface_click called with pos=({click_pos.x()}, {click_pos.y()})")
            # QtInteractor erbt von Plotter, aber pick() könnte nicht verfügbar sein
            # Verwende stattdessen den Renderer direkt für Picking
            renderer = self.plotter.renderer
            if renderer is None:
                print(f"[DEBUG Click] No renderer available")
                return
            
            # Verwende VTK CellPicker für präzises Picking
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker
                picker = vtkCellPicker()
                picker.SetTolerance(0.001)
                
                # Konvertiere Qt-Koordinaten zu VTK-Koordinaten (Viewport-Koordinaten)
                # VTK verwendet Viewport-Koordinaten (0-1 normalisiert)
                size = self.widget.size()
                if size.width() > 0 and size.height() > 0:
                    x_norm = click_pos.x() / size.width()
                    y_norm = 1.0 - (click_pos.y() / size.height())  # VTK Y ist invertiert
                    
                    print(f"[DEBUG Click] Normalized coords: x={x_norm:.3f}, y={y_norm:.3f}")
                    picker.Pick(x_norm, y_norm, 0.0, renderer)
                    
                    picked_actor = picker.GetActor()
                    picked_point = picker.GetPickPosition()
                    
                    print(f"[DEBUG Click] VTK picker: actor={picked_actor}, point={picked_point}")
                    
                    if picked_actor is not None:
                        # Hole Actor-Name
                        actor_name = None
                        if hasattr(picked_actor, 'GetProperty'):
                            # Versuche Name über verschiedene Wege zu bekommen
                            actor_name = getattr(picked_actor, 'name', None)
                            if actor_name is None:
                                # Prüfe ob Actor in plotter.renderer.actors ist
                                for name, actor in renderer.actors.items():
                                    if actor == picked_actor:
                                        actor_name = name
                                        break
                        
                        print(f"[DEBUG Click] actor_name={actor_name}")
                        
                        # Prüfe ob es eine disabled Surface-Batch-Fläche ist
                        # (Batch-Actors haben feste Namen, Picking erfolgt über point-in-polygon-Prüfung)
                        if actor_name and isinstance(actor_name, str) and actor_name in ('surface_disabled_polygons_batch', 'surface_disabled_edges_batch'):
                            # Batch-Actor - hole 3D-Koordinaten vom Picker und prüfe point-in-polygon
                            if picked_point and len(picked_point) >= 3:
                                x_click, y_click = picked_point[0], picked_point[1]
                                print(f"[DEBUG Click] Found disabled surface batch actor, picked point: x={x_click:.2f}, y={y_click:.2f}")
                                # Rufe _handle_spl_surface_click auf für point-in-polygon-Prüfung
                                self._handle_spl_surface_click(click_pos)
                                return
                            else:
                                # Kein Punkt vom Picker - verwende mesh-basierten Lookup
                                print(f"[DEBUG Click] Found disabled surface batch actor but no point, using mesh lookup")
                                self._handle_spl_surface_click(click_pos)
                                return
                        
                        # Prüfe ob es die SPL-Surface ist (für enabled Surfaces)
                        if actor_name == self.SURFACE_NAME:
                            print(f"[DEBUG Click] Found SPL surface, handling SPL surface click")
                            self._handle_spl_surface_click(click_pos)
                            return
                    
                    # Wenn kein Actor gepickt wurde, aber ein Punkt vorhanden ist, prüfe SPL-Surface
                    if picked_point and len(picked_point) >= 3:
                        print(f"[DEBUG Click] No actor but point found, trying SPL surface click")
                        self._handle_spl_surface_click(click_pos)
                        return
                else:
                    print(f"[DEBUG Click] Invalid widget size: {size}")
            except ImportError:
                print(f"[DEBUG Click] vtkCellPicker not available, trying alternative method")
                # Fallback: Versuche PyVista pick über renderer
                try:
                    # QtInteractor sollte Plotter-Methoden haben, aber vielleicht nicht pick()
                    # Versuche über renderer.actors zu iterieren und Distanz zu prüfen
                    pass
                except Exception as e:
                    print(f"[DEBUG Click] Fallback also failed: {e}")
            
            # Kein Actor gefunden - prüfe ob auf SPL-Surface geklickt wurde (für enabled Surfaces)
            print(f"[DEBUG Click] No actor picked, trying SPL surface click")
            self._handle_spl_surface_click(click_pos)
            
        except Exception as e:  # noqa: BLE001
            # Bei Fehler einfach ignorieren
            print(f"[DEBUG Click] Exception in _handle_surface_click: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_spl_surface_click(self, click_pos: QtCore.QPoint) -> None:
        """Behandelt einen Klick auf die SPL-Surface (für enabled Surfaces)."""
        try:
            print(f"[DEBUG Click] _handle_spl_surface_click called")
            # Hole 3D-Koordinaten des geklickten Punktes über VTK CellPicker
            renderer = self.plotter.renderer
            if renderer is None:
                print(f"[DEBUG Click] No renderer available for SPL click")
                return
            
            # Versuche direkt auf das Surface-Mesh zuzugreifen und die Zelle an der geklickten Position zu finden
            if self.surface_mesh is None:
                print(f"[DEBUG Click] Surface mesh is None, cannot determine clicked point")
                return
            
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker
                from vtkmodules.util.numpy_support import vtk_to_numpy
                
                picker = vtkCellPicker()
                picker.SetTolerance(0.01)
                
                size = self.widget.size()
                if size.width() > 0 and size.height() > 0:
                    x_norm = click_pos.x() / size.width()
                    y_norm = 1.0 - (click_pos.y() / size.height())  # VTK Y ist invertiert
                    
                    print(f"[DEBUG Click] Picking at normalized coords: x={x_norm:.3f}, y={y_norm:.3f}")
                    
                    # Picke auf dem Renderer
                    picker.Pick(x_norm, y_norm, 0.0, renderer)
                    picked_cell_id = picker.GetCellId()
                    picked_point = picker.GetPickPosition()
                    
                    print(f"[DEBUG Click] Picker result: cell_id={picked_cell_id}, point={picked_point}")
                    
                    # Hilfsfunktion für mesh-basierten Lookup
                    def use_mesh_lookup():
                        print(f"[DEBUG Click] Using mesh-based coordinate lookup")
                        
                        try:
                            # Verwende VTK Renderer mit normalisierten Display-Koordinaten
                            # VTK Renderer erwartet normalisierte Koordinaten (0-1) in SetDisplayPoint
                            # Aber SetDisplayPoint verwendet Viewport-Koordinaten (nicht Pixel)
                            # Wir müssen die normalisierten Koordinaten korrekt konvertieren
                            
                            # Hole Viewport-Größe vom Renderer
                            viewport = renderer.GetViewport()
                            render_size = renderer.GetSize()
                            
                            if render_size[0] <= 0 or render_size[1] <= 0:
                                print(f"[DEBUG Click] Invalid renderer size: {render_size}")
                                return None
                            
                            # Konvertiere normalisierte Koordinaten (0-1) zu Viewport-Koordinaten
                            viewport_width = render_size[0]
                            viewport_height = render_size[1]
                            
                            # Viewport-Koordinaten: (0,0) ist unten links, (width, height) ist oben rechts
                            # x_norm und y_norm sind bereits normalisiert (0-1), y_norm ist invertiert (oben = 0)
                            viewport_x = x_norm * viewport_width
                            viewport_y = y_norm * viewport_height
                            
                            print(f"[DEBUG Click] Viewport coords: x={viewport_x:.1f}, y={viewport_y:.1f} (size={viewport_width}x{viewport_height})")
                            
                            # Konvertiere Display-Koordinaten (Viewport) zu World-Koordinaten
                            renderer.SetDisplayPoint(viewport_x, viewport_y, 0.0)
                            renderer.DisplayToWorld()
                            world_point_near = renderer.GetWorldPoint()
                            
                            renderer.SetDisplayPoint(viewport_x, viewport_y, 1.0)
                            renderer.DisplayToWorld()
                            world_point_far = renderer.GetWorldPoint()
                            
                            if world_point_near is not None and world_point_far is not None and len(world_point_near) >= 4:
                                # VTK gibt homogene Koordinaten zurück [x, y, z, w]
                                if abs(world_point_near[3]) > 1e-6 and abs(world_point_far[3]) > 1e-6:
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
                                    
                                    print(f"[DEBUG Click] Ray start: {ray_start}, Ray end: {ray_end}")
                                    
                                    # Berechne Schnittpunkt des Rays mit der Z=0 Ebene
                                    ray_dir = ray_end - ray_start
                                    if abs(ray_dir[2]) > 1e-6:  # Ray ist nicht parallel zur Z-Ebene
                                        # Parametrische Form: point = ray_start + t * ray_dir
                                        # Z = 0: ray_start[2] + t * ray_dir[2] = 0
                                        t = -ray_start[2] / ray_dir[2]
                                        intersection = ray_start + t * ray_dir
                                        
                                        print(f"[DEBUG Click] Ray intersection with Z=0 plane: x={intersection[0]:.2f}, y={intersection[1]:.2f}, z={intersection[2]:.2f}")
                                        return (float(intersection[0]), float(intersection[1]), float(intersection[2]))
                                    else:
                                        # Ray ist parallel zur Z-Ebene - verwende Ray-Mitte
                                        ray_mid = (ray_start + ray_end) / 2.0
                                        print(f"[DEBUG Click] Ray parallel to Z plane, using ray midpoint: x={ray_mid[0]:.2f}, y={ray_mid[1]:.2f}, z={ray_mid[2]:.2f}")
                                        return (float(ray_mid[0]), float(ray_mid[1]), float(ray_mid[2]))
                                else:
                                    print(f"[DEBUG Click] Invalid homogeneous coordinates (w=0)")
                                    return None
                            else:
                                print(f"[DEBUG Click] Invalid world point conversion")
                                return None
                        except Exception as e:
                            print(f"[DEBUG Click] Exception in use_mesh_lookup: {e}")
                            import traceback
                            traceback.print_exc()
                            return None
                    
                    # Wenn Picker keine Zelle findet, verwende mesh-basierten Lookup
                    if picked_cell_id < 0 or picked_point is None:
                        result = use_mesh_lookup()
                        if result is None:
                            return
                        x_click, y_click, z_click = result
                        print(f"[DEBUG Click] Found nearest mesh point: x={x_click:.2f}, y={y_click:.2f}, z={z_click:.2f}")
                    else:
                        # Picker hat einen Punkt gefunden - prüfe ob er im gültigen Bereich liegt
                        x_click, y_click, z_click = picked_point[0], picked_point[1], picked_point[2]
                        
                        # Berechne gültigen Bereich aus allen enabled Surfaces
                        surface_definitions = getattr(self.settings, 'surface_definitions', {})
                        if isinstance(surface_definitions, dict):
                            valid_x_min, valid_x_max = None, None
                            valid_y_min, valid_y_max = None, None
                            
                            for surface_def in surface_definitions.values():
                                if isinstance(surface_def, SurfaceDefinition):
                                    enabled = bool(getattr(surface_def, 'enabled', False))
                                    points = getattr(surface_def, 'points', []) or []
                                else:
                                    enabled = surface_def.get('enabled', False)
                                    points = surface_def.get('points', [])
                                
                                if enabled and len(points) >= 3:
                                    surface_xs = [p.get('x', 0.0) if isinstance(p, dict) else getattr(p, 'x', 0.0) for p in points]
                                    surface_ys = [p.get('y', 0.0) if isinstance(p, dict) else getattr(p, 'y', 0.0) for p in points]
                                    if surface_xs and surface_ys:
                                        if valid_x_min is None:
                                            valid_x_min, valid_x_max = min(surface_xs), max(surface_xs)
                                            valid_y_min, valid_y_max = min(surface_ys), max(surface_ys)
                                        else:
                                            valid_x_min = min(valid_x_min, min(surface_xs))
                                            valid_x_max = max(valid_x_max, max(surface_xs))
                                            valid_y_min = min(valid_y_min, min(surface_ys))
                                            valid_y_max = max(valid_y_max, max(surface_ys))
                            
                            # Prüfe ob Picker-Punkt im gültigen Bereich liegt (mit Toleranz)
                            tolerance = 5.0  # 5 Meter Toleranz
                            if valid_x_min is not None:
                                if (x_click < valid_x_min - tolerance or x_click > valid_x_max + tolerance or
                                    y_click < valid_y_min - tolerance or y_click > valid_y_max + tolerance):
                                    print(f"[DEBUG Click] Picker point ({x_click:.2f}, {y_click:.2f}) outside valid range "
                                          f"({valid_x_min:.2f}..{valid_x_max:.2f}, {valid_y_min:.2f}..{valid_y_max:.2f}), using mesh lookup")
                                    result = use_mesh_lookup()
                                    if result is None:
                                        return
                                    x_click, y_click, z_click = result
                                    print(f"[DEBUG Click] Found nearest mesh point: x={x_click:.2f}, y={y_click:.2f}, z={z_click:.2f}")
                                else:
                                    print(f"[DEBUG Click] Using picker point: x={x_click:.2f}, y={y_click:.2f}, z={z_click:.2f}")
                            else:
                                print(f"[DEBUG Click] No enabled surfaces found, using picker point: x={x_click:.2f}, y={y_click:.2f}, z={z_click:.2f}")
                        else:
                            print(f"[DEBUG Click] Using picker point: x={x_click:.2f}, y={y_click:.2f}, z={z_click:.2f}")
                else:
                    print(f"[DEBUG Click] Invalid widget size for SPL click")
                    return
            except ImportError:
                print(f"[DEBUG Click] VTK modules not available for SPL click")
                return
            except Exception as e:
                print(f"[DEBUG Click] Exception in coordinate conversion: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Prüfe welche Surface (enabled oder disabled) diesen Punkt enthält
            surface_definitions = getattr(self.settings, 'surface_definitions', {})
            print(f"[DEBUG Click] surface_definitions count: {len(surface_definitions) if isinstance(surface_definitions, dict) else 0}")
            if not isinstance(surface_definitions, dict):
                print(f"[DEBUG Click] surface_definitions is not a dict")
                return
            
            # Durchsuche alle Surfaces (enabled und disabled)
            checked_count = 0
            skipped_disabled = 0
            skipped_hidden = 0
            skipped_too_few_points = 0
            
            # Zuerst prüfe enabled Surfaces (höhere Priorität)
            for surface_id, surface_def in surface_definitions.items():
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, 'enabled', False))
                    hidden = bool(getattr(surface_def, 'hidden', False))
                    points = getattr(surface_def, 'points', []) or []
                else:
                    enabled = surface_def.get('enabled', False)
                    hidden = surface_def.get('hidden', False)
                    points = surface_def.get('points', [])
                
                if hidden:
                    skipped_hidden += 1
                    continue
                if len(points) < 3:
                    skipped_too_few_points += 1
                    continue
                
                if not enabled:
                    skipped_disabled += 1
                    # Prüfe disabled Surfaces später (niedrigere Priorität)
                    continue
                
                checked_count += 1
                
                # Debug: Berechne Bounding Box der Surface für Vergleich
                surface_xs = [p.get('x', 0.0) for p in points]
                surface_ys = [p.get('y', 0.0) for p in points]
                if surface_xs and surface_ys:
                    min_x, max_x = min(surface_xs), max(surface_xs)
                    min_y, max_y = min(surface_ys), max(surface_ys)
                    
                    # Prüfe ob Punkt in diesem Polygon liegt (nur X/Y, Z wird ignoriert)
                    is_inside = self._point_in_polygon(x_click, y_click, points)
                    
                    # Debug: Zeige erste paar Punkte des Polygons für Vergleich
                    if checked_count <= 10 or is_inside:
                        first_point = points[0] if points else {}
                        print(f"[DEBUG Click] Surface {surface_id}: enabled={enabled}, hidden={hidden}, points_count={len(points)}, "
                              f"bbox=({min_x:.2f}..{max_x:.2f}, {min_y:.2f}..{max_y:.2f}), "
                              f"click=({x_click:.2f}, {y_click:.2f}), "
                              f"is_inside={is_inside}")
                    
                    if is_inside:
                        print(f"[DEBUG Click] Found matching surface: {surface_id}")
                        self._select_surface_in_treewidget(str(surface_id))
                        return
                else:
                    print(f"[DEBUG Click] Surface {surface_id}: No valid points")
            
            # Wenn keine enabled Surface gefunden wurde, prüfe disabled Surfaces
            # (immer prüfen, auch wenn enabled Surfaces gefunden wurden, falls der Klick auf disabled Surface war)
            if skipped_disabled > 0:
                # Prüfe disabled Surfaces
                for surface_id, surface_def in surface_definitions.items():
                    if isinstance(surface_def, SurfaceDefinition):
                        enabled = bool(getattr(surface_def, 'enabled', False))
                        hidden = bool(getattr(surface_def, 'hidden', False))
                        points = getattr(surface_def, 'points', []) or []
                    else:
                        enabled = surface_def.get('enabled', False)
                        hidden = surface_def.get('hidden', False)
                        points = surface_def.get('points', [])
                    
                    if enabled or hidden or len(points) < 3:
                        continue
                    
                    # Prüfe ob Punkt in diesem Polygon liegt
                    surface_xs = [p.get('x', 0.0) for p in points]
                    surface_ys = [p.get('y', 0.0) for p in points]
                    if surface_xs and surface_ys:
                        min_x, max_x = min(surface_xs), max(surface_xs)
                        min_y, max_y = min(surface_ys), max(surface_ys)
                        
                        # Schnelle Bounding-Box-Prüfung zuerst
                        if x_click < min_x or x_click > max_x or y_click < min_y or y_click > max_y:
                            continue
                        
                        is_inside = self._point_in_polygon(x_click, y_click, points)
                        if is_inside:
                            print(f"[DEBUG Click] Found matching disabled surface: {surface_id}")
                            self._select_surface_in_treewidget(str(surface_id))
                            return
            
            print(f"[DEBUG Click] Checked {checked_count} enabled surfaces, skipped: disabled={skipped_disabled}, "
                  f"hidden={skipped_hidden}, too_few_points={skipped_too_few_points}, no match found")
            
        except Exception as e:  # noqa: BLE001
            print(f"[DEBUG Click] Exception in _handle_spl_surface_click: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon_points: List[Dict[str, float]]) -> bool:
        """Prüft, ob ein Punkt (x, y) innerhalb eines Polygons liegt (Ray-Casting-Algorithmus)."""
        if len(polygon_points) < 3:
            return False
        
        # Extrahiere X- und Y-Koordinaten
        px = np.array([p.get("x", 0.0) for p in polygon_points])
        py = np.array([p.get("y", 0.0) for p in polygon_points])
        
        # Ray-Casting-Algorithmus
        n = len(px)
        inside = False
        boundary_eps = 1e-6
        
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            
            # Prüfe ob Strahl von (x,y) nach rechts die Kante schneidet
            if ((yi > y) != (yj > y)) and (
                x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi + boundary_eps
            ):
                inside = not inside
            
            # Prüfe ob Punkt direkt auf der Kante liegt
            dx = xj - xi
            dy = yj - yi
            segment_len = math.hypot(dx, dy)
            if segment_len > 0:
                dist = abs(dy * (x - xi) - dx * (y - yi)) / segment_len
                if dist <= boundary_eps:
                    proj = ((x - xi) * dx + (y - yi) * dy) / (segment_len * segment_len)
                    if -boundary_eps <= proj <= 1 + boundary_eps:
                        return True
            j = i
        
        return inside
    
    def _select_surface_in_treewidget(self, surface_id: str) -> None:
        """Wählt eine Surface im TreeWidget aus und öffnet die zugehörige Gruppe."""
        try:
            print(f"[DEBUG Click] _select_surface_in_treewidget called with surface_id={surface_id}")
            # Finde das Main-Window über parent_widget
            # parent_widget ist self.main_window.ui.SPLPlot (ein QWidget)
            # Wir müssen durch die Widget-Hierarchie gehen, um das MainWindow zu finden
            parent = self.parent_widget
            print(f"[DEBUG Click] parent_widget={parent}, type={type(parent).__name__}")
            main_window = None
            
            # Gehe durch die Widget-Hierarchie, um das Main-Window zu finden
            depth = 0
            while parent is not None and depth < 15:
                parent_type = type(parent).__name__
                has_surface_manager = hasattr(parent, 'surface_manager')
                print(f"[DEBUG Click] Checking parent at depth {depth}: {parent_type}, has surface_manager={has_surface_manager}")
                
                if has_surface_manager:
                    main_window = parent
                    print(f"[DEBUG Click] Found main_window: {main_window}")
                    break
                
                # Versuche verschiedene Wege, um zum nächsten Parent zu kommen
                next_parent = None
                if hasattr(parent, 'parent'):
                    next_parent = parent.parent()
                elif hasattr(parent, 'parentWidget'):
                    next_parent = parent.parentWidget()
                elif hasattr(parent, 'parent_widget'):
                    next_parent = parent.parent_widget
                elif hasattr(parent, 'window'):
                    # QWidget.window() gibt das Top-Level-Window zurück
                    next_parent = parent.window()
                
                if next_parent == parent:
                    # Verhindere Endlosschleife
                    break
                parent = next_parent
                depth += 1
            
            if main_window is None or not hasattr(main_window, 'surface_manager'):
                print(f"[DEBUG Click] Could not find main_window or surface_manager after {depth} levels")
                # Versuche alternativen Weg: Suche nach QMainWindow
                parent = self.parent_widget
                while parent is not None:
                    if isinstance(parent, QtWidgets.QMainWindow):
                        if hasattr(parent, 'surface_manager'):
                            main_window = parent
                            print(f"[DEBUG Click] Found QMainWindow with surface_manager: {main_window}")
                            break
                    if hasattr(parent, 'parent'):
                        parent = parent.parent()
                    elif hasattr(parent, 'parentWidget'):
                        parent = parent.parentWidget()
                    else:
                        break
                
                if main_window is None:
                    print(f"[DEBUG Click] Could not find main_window")
                    return
            
            # Hole das SurfaceDockWidget über surface_manager
            surface_manager = main_window.surface_manager
            print(f"[DEBUG Click] surface_manager={surface_manager}")
            if not hasattr(surface_manager, 'surface_tree_widget'):
                print(f"[DEBUG Click] surface_tree_widget not found")
                return
            
            # UISurfaceManager verwendet direkt ein QTreeWidget, nicht WindowSurfaceWidget
            # Verwende _select_surface_in_tree Methode
            if hasattr(surface_manager, '_select_surface_in_tree'):
                print(f"[DEBUG Click] Calling _select_surface_in_tree({surface_id})")
                # Wähle Surface aus
                surface_manager._select_surface_in_tree(surface_id)
                
                # Finde das Surface-Item und expandiere die Parent-Gruppe
                tree_widget = surface_manager.surface_tree_widget
                if tree_widget is not None:
                    # Finde das Item rekursiv
                    def find_item(parent_item=None):
                        if parent_item is None:
                            for i in range(tree_widget.topLevelItemCount()):
                                item = tree_widget.topLevelItem(i)
                                found = find_item(item)
                                if found:
                                    return found
                        else:
                            item_surface_id = parent_item.data(0, Qt.UserRole)
                            if isinstance(item_surface_id, dict):
                                item_surface_id = item_surface_id.get('id')
                            if item_surface_id == surface_id:
                                return parent_item
                            for i in range(parent_item.childCount()):
                                child = parent_item.child(i)
                                found = find_item(child)
                                if found:
                                    return found
                        return None
                    
                    item = find_item()
                    if item is not None:
                        print(f"[DEBUG Click] Found surface item: {item}")
                        # Finde Parent-Gruppe und expandiere sie
                        parent = item.parent()
                        while parent is not None:
                            item_type = parent.data(0, Qt.UserRole + 1)
                            if item_type == "group":
                                print(f"[DEBUG Click] Found parent group, expanding")
                                parent.setExpanded(True)
                                # Scrolle zur Gruppe, damit sie sichtbar ist
                                tree_widget.scrollToItem(parent, QtWidgets.QAbstractItemView.PositionAtTop)
                                break
                            parent = parent.parent()
                        
                        # Scrolle zur Surface, damit sie sichtbar ist
                        tree_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
                        print(f"[DEBUG Click] Scrolled to item")
                    else:
                        print(f"[DEBUG Click] Could not find surface item in tree")
            else:
                print(f"[DEBUG Click] surface_manager has no _select_surface_in_tree method")
                    
        except Exception as e:  # noqa: BLE001
            # Bei Fehler einfach ignorieren
            print(f"[DEBUG Click] Exception in _select_surface_in_treewidget: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_surface_dock_widget(self, widget) -> Optional[Any]:
        """Findet rekursiv das SurfaceDockWidget (WindowSurfaceWidget) in der Widget-Hierarchie."""
        if widget is None:
            return None
        
        # Prüfe ob dieses Widget das SurfaceDockWidget ist (hat _select_surface Methode)
        if hasattr(widget, '_select_surface') and hasattr(widget, 'surface_tree'):
            return widget
        
        # Prüfe Kinder
        if hasattr(widget, 'children'):
            for child in widget.children():
                result = self._find_surface_dock_widget(child)
                if result is not None:
                    return result
        
        # Prüfe Layout
        if hasattr(widget, 'layout'):
            layout = widget.layout()
            if layout is not None:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is not None:
                        child_widget = item.widget()
                        if child_widget is not None:
                            result = self._find_surface_dock_widget(child_widget)
                            if result is not None:
                                return result
        
        return None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def initialize_empty_scene(self, preserve_camera: bool = True):
        """Zeigt eine leere Szene und bewahrt optional die Kameraposition."""
        if preserve_camera:
            camera_state = self._camera_state or self._capture_camera()
        else:
            camera_state = None
            # Wenn Kamera nicht erhalten wird, Flag zurücksetzen, damit beim nächsten Plot mit Daten Zoom maximiert wird
            self._has_plotted_data = False

        self.plotter.clear()
        self.overlay_helper.clear()
        self.has_data = False
        self.surface_mesh = None
        self._last_overlay_signatures = {}
        # 🎯 Setze auch _last_surfaces_state zurück, damit Surfaces nach initialize_empty_scene() neu gezeichnet werden
        if hasattr(self.overlay_helper, '_last_surfaces_state'):
            self.overlay_helper._last_surfaces_state = None
        # Cache zurücksetzen
        if hasattr(self, '_surface_signature_cache'):
            self._surface_signature_cache.clear()
        
        # Konfiguriere Plotter nur bei Bedarf (nicht die Kamera überschreiben)
        self._configure_plotter(configure_camera=not preserve_camera)
        # scene_frame wurde entfernt - nicht mehr benötigt

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
        if self.time_control is not None:
            self.time_control.hide()

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

        plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        phase_mode = plot_mode == 'Phase alignment'
        time_mode = plot_mode == 'SPL over time'
        self._phase_mode_active = phase_mode
        self._time_mode_active = time_mode

        self._debug_dump_soundfield(x, y, pressure)

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

        if time_mode:
            pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            finite_mask = np.isfinite(pressure)
            if not np.any(finite_mask):
                self.initialize_empty_scene(preserve_camera=True)
                return
            # Verwende feste Colorbar-Parameter (keine dynamische Skalierung)
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
            spl_db = self.functions.mag2db(pressure_2d)
            if getattr(self.settings, "fem_debug_logging", True):
                try:
                    p_min = float(np.nanmin(pressure_2d))
                    p_max = float(np.nanmax(pressure_2d))
                    p_mean = float(np.nanmean(pressure_2d))
                    spl_min = float(np.nanmin(spl_db))
                    spl_max = float(np.nanmax(spl_db))
                    spl_mean = float(np.nanmean(spl_db))
                except Exception:
                    pass
                try:
                    x_idx = int(np.argmin(np.abs(x - 0.0)))
                    target_distances = (1.0, 2.0, 10.0, 20.0)
                    for distance in target_distances:
                        y_idx = int(np.argmin(np.abs(y - distance)))
                        value = float(spl_db[y_idx, x_idx])
                        actual_y = float(y[y_idx])
                except Exception:
                    pass
  
            
            finite_mask = np.isfinite(spl_db)
            if not np.any(finite_mask):
                self.initialize_empty_scene(preserve_camera=True)
                return
            plot_values = spl_db

        colorbar_params = self._get_colorbar_params(phase_mode)
        cbar_min = colorbar_params['min']
        cbar_max = colorbar_params['max']
        cbar_step = colorbar_params['step']
        tick_step = colorbar_params['tick_step']
        self._colorbar_override = colorbar_params

        # 🎯 WICHTIG: Speichere ursprüngliche Berechnungs-Werte für PyVista sample-Modus VOR Clipping!
        # Clipping ändert die Werte und sollte nur für Visualisierung erfolgen, nicht für Sampling
        original_plot_values = plot_values.copy() if hasattr(plot_values, 'copy') else plot_values
        
        # 🎯 Cache original_plot_values für Debug-Vergleiche
        self._cached_original_plot_values = original_plot_values
        
        # 🎯 Validierung: Stelle sicher, dass original_plot_values die korrekte Shape hat
        if original_plot_values.shape != (len(y), len(x)):
            raise ValueError(
                f"original_plot_values Shape stimmt nicht überein: "
                f"erhalten {original_plot_values.shape}, erwartet ({len(y)}, {len(x)})"
            )

        # Clipping nur für Visualisierung (nicht für Sampling)
        if time_mode:
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
        elif phase_mode:
            plot_values = np.clip(plot_values, cbar_min, cbar_max)
        else:
            plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)
        
        try:
            geometry = prepare_plot_geometry(
                x,
                y,
                plot_values,
                settings=self.settings,
                container=self.container,
                default_upscale=self.UPSCALE_FACTOR,
            )
        except RuntimeError as exc:
            print(f"[Plot SPL3D] Abbruch: {exc}")
            self.initialize_empty_scene(preserve_camera=True)
            # Auch vertikale SPL-Flächen entfernen, da keine gültige Geometrie vorliegt
            self._clear_vertical_spl_surfaces()
            return

        # Debug: Prüfen, ob das Plot-Grid gegenüber dem Berechnungs-Grid verfeinert wurde
        if DEBUG_SPL_DUMP:
            try:
                print(
                    "[Plot SPL3D] Grid-Info:",
                    f"source_grid=({len(y)}x{len(x)}), "
                    f"plot_grid=({geometry.plot_y.size}x{geometry.plot_x.size}), "
                    f"upscaled={geometry.was_upscaled}, "
                    f"requires_resample={geometry.requires_resample}",
                )
            except Exception:
                pass
        plot_x = geometry.plot_x
        plot_y = geometry.plot_y
        plot_values = geometry.plot_values
        z_coords = geometry.z_coords
        
        # 🎯 Cache Plot-Geometrie für Click-Handling
        self._plot_geometry_cache = {
            'plot_x': plot_x.copy() if hasattr(plot_x, 'copy') else plot_x,
            'plot_y': plot_y.copy() if hasattr(plot_y, 'copy') else plot_y,
            'source_x': geometry.source_x.copy() if hasattr(geometry.source_x, 'copy') else geometry.source_x,
            'source_y': geometry.source_y.copy() if hasattr(geometry.source_y, 'copy') else geometry.source_y,
        }
        
        colorization_mode_used = colorization_mode
        if colorization_mode_used == 'Color step':
            scalars = self._quantize_to_steps(plot_values, cbar_step)
        else:
            scalars = plot_values
        
        # Für PyVista sample-Modus: Verwende ursprüngliche Berechnungs-Werte (vor Upscaling)
        # 🎯 Validierung: Stelle sicher, dass source_x/source_y mit original_plot_values übereinstimmen
        source_scalars_for_sample = None
        if getattr(self.settings, "spl_plot_use_pyvista_sample", False):
            # Prüfe ob Shapes übereinstimmen
            if original_plot_values.shape != (len(geometry.source_y), len(geometry.source_x)):
                raise ValueError(
                    f"Shape-Mismatch zwischen original_plot_values und source_x/source_y: "
                    f"original_plot_values.shape={original_plot_values.shape}, "
                    f"source_grid=({len(geometry.source_y)}, {len(geometry.source_x)})"
                )
            source_scalars_for_sample = original_plot_values
        
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
            source_scalars=source_scalars_for_sample,
        )
        
        # 🎯 DEBUG: Referenzpunkt pro Surface beim Plotten
        if mesh is not None and hasattr(mesh, 'points') and mesh.n_points > 0:
            # Finde Referenzpunkte pro Surface (wenn PyVista sample-Modus aktiv)
            if getattr(self.settings, "spl_plot_use_pyvista_sample", False):
                # Im PyVista sample-Modus: Mesh enthält alle kombinierten Surfaces
                # Suche nach enabled Surfaces und gebe für jedes einen Referenzpunkt aus
                surface_definitions = getattr(self.settings, 'surface_definitions', {})
                if isinstance(surface_definitions, dict):
                    for surface_id, surface_def in surface_definitions.items():
                        if isinstance(surface_def, SurfaceDefinition):
                            enabled = bool(getattr(surface_def, 'enabled', False))
                            hidden = bool(getattr(surface_def, 'hidden', False))
                            points = getattr(surface_def, 'points', []) or []
                        else:
                            enabled = bool(surface_def.get('enabled', False))
                            hidden = bool(surface_def.get('hidden', False))
                            points = surface_def.get('points', [])
                        
                        if enabled and not hidden and len(points) >= 3:
                            # Berechne Mittelpunkt der Surface-Bounding-Box
                            surface_xs = [p.get("x", 0.0) for p in points]
                            surface_ys = [p.get("y", 0.0) for p in points]
                            if surface_xs and surface_ys:
                                ref_x = (min(surface_xs) + max(surface_xs)) / 2.0
                                ref_y = (min(surface_ys) + max(surface_ys)) / 2.0
                                # Finde nächsten Punkt im Mesh
                                mesh_points = mesh.points
                                if mesh_points.size > 0:
                                    mesh_points_2d = mesh_points[:, :2]  # Nur X, Y
                                    ref_point_2d = np.array([ref_x, ref_y])
                                    distances = np.linalg.norm(mesh_points_2d - ref_point_2d, axis=1)
                                    nearest_idx = int(np.argmin(distances))
                                    nearest_point = mesh_points[nearest_idx]
                                    debug_plot_x = float(nearest_point[0])
                                    debug_plot_y = float(nearest_point[1])
                                    debug_plot_z = float(nearest_point[2])
                                    
                                    # Hole SPL-Wert an dieser Stelle (wenn verfügbar)
                                    plot_spl = None
                                    if hasattr(mesh, 'point_data') and 'plot_scalars' in mesh.point_data:
                                        plot_spl = float(mesh.point_data['plot_scalars'][nearest_idx])
                                    
                                    # 🎯 DEBUG: Vergleiche mit ursprünglichem Berechnungs-Grid
                                    source_spl = None
                                    if hasattr(geometry, 'source_x') and hasattr(geometry, 'source_y'):
                                        # Finde nächstgelegenen Punkt im ursprünglichen Berechnungs-Grid
                                        source_x_arr = np.asarray(geometry.source_x, dtype=float)
                                        source_y_arr = np.asarray(geometry.source_y, dtype=float)
                                        
                                        # Finde Index im source-Grid
                                        x_idx = int(np.argmin(np.abs(source_x_arr - debug_plot_x))) if len(source_x_arr) > 0 else 0
                                        y_idx = int(np.argmin(np.abs(source_y_arr - debug_plot_y))) if len(source_y_arr) > 0 else 0
                                        
                                        if 0 <= y_idx < len(source_y_arr) and 0 <= x_idx < len(source_x_arr):
                                            # Hole SPL-Wert aus original_plot_values (vor Clipping)
                                            if hasattr(self, '_cached_original_plot_values'):
                                                orig_vals = self._cached_original_plot_values
                                                if orig_vals.shape == (len(source_y_arr), len(source_x_arr)):
                                                    source_spl = float(orig_vals[y_idx, x_idx])
                                    
                                    spl_str = f", SPL={plot_spl:.2f} dB" if plot_spl is not None else ""
                                    source_str = f", Source-Grid SPL={source_spl:.2f} dB" if source_spl is not None else ""
                                    diff_str = f" [Diff: {plot_spl - source_spl:.2f} dB]" if plot_spl is not None and source_spl is not None else ""
                                    

        if time_mode:
            cmap_object = 'RdBu_r'
        elif phase_mode:
            cmap_object = self.PHASE_CMAP
        else:
            cmap_object = 'jet'

        # ------------------------------------------------------------------
        # Horizontale SPL-Surfaces
        # ------------------------------------------------------------------
        render_mode = getattr(self.settings, "spl_surface_render_mode", "mesh") or "mesh"

        # Kombiniertes Mesh für Click-Handling behalten
        if mesh is not None and mesh.n_points > 0:
            self.surface_mesh = mesh.copy(deep=True)

        surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
        enabled_surfaces: list[tuple[str, list[dict[str, float]]]] = []
        if isinstance(surface_definitions, dict):
            for surface_id, surface_def in surface_definitions.items():
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, "enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False))
                    points = getattr(surface_def, "points", []) or []
                else:
                    enabled = bool(surface_def.get("enabled", False))
                    hidden = bool(surface_def.get("hidden", False))
                    points = surface_def.get("points", []) or []
                if enabled and not hidden and len(points) >= 3:
                    enabled_surfaces.append((str(surface_id), points))

        if render_mode.lower() == "texture":
            # Entferne existierende Mesh-Actors für horizontale Surfaces
            base_actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
            if base_actor is not None:
                try:
                    self.plotter.remove_actor(base_actor)
                except Exception:
                    pass
            for sid, actor in list(self._surface_actors.items()):
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
                self._surface_actors.pop(sid, None)

            # Zeichne alle aktiven Surfaces als Texturflächen
            self._render_surfaces_textured(
                geometry,
                original_plot_values,
                cbar_min,
                cbar_max,
                cmap_object,
                colorization_mode_used,
                cbar_step,
            )
        else:
            # Klassischer Mesh-Pfad
            if mesh is None or mesh.n_cells == 0 or not enabled_surfaces:
                # Fallback: wie bisher ein einzelner Actor für das gesamte Mesh
                actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
                if actor is None or self.surface_mesh is None:
                    self.surface_mesh = mesh.copy(deep=True) if mesh is not None else None
                    if self.surface_mesh is not None:
                        actor = self.plotter.add_mesh(
                            self.surface_mesh,
                            name=self.SURFACE_NAME,
                            scalars='plot_scalars',
                            cmap=cmap_object,
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
                        mapper.lookup_table = self.plotter._cmap_to_lut(cmap_object)
                        mapper.interpolate_before_map = False
            else:
                # Mehrere aktive Surfaces: pro Surface ein eigener Actor
                t_total_start = time.perf_counter()
                # Berechne Zellzentren einmal
                cell_centers = mesh.cell_centers().points  # (n_cells, 3)
                cell_xy = cell_centers[:, :2]

                # Entferne Actors für Surfaces, die nicht mehr aktiv sind
                active_ids = {sid for sid, _ in enabled_surfaces}
                for sid in list(self._surface_actors.keys()):
                    if sid not in active_ids:
                        actor = self._surface_actors.pop(sid, None)
                        if actor is not None:
                            try:
                                self.plotter.remove_actor(actor)
                            except Exception:  # noqa: BLE001
                                pass

                for surface_id, points in enabled_surfaces:
                    t_surf_start = time.perf_counter()

                    poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                    poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                    if poly_x.size == 0 or poly_y.size == 0:
                        continue

                    poly_path = Path(np.column_stack((poly_x, poly_y)))
                    inside = poly_path.contains_points(cell_xy)
                    cell_indices = np.nonzero(inside)[0]

                    if cell_indices.size == 0:
                        if DEBUG_SURFACE_GEOMETRY:
                            print(
                                f"[SurfaceGeometry] Plot-Surface '{surface_id}': "
                                f"keine Zellen im Polygon gefunden"
                            )
                        # Falls diese Surface zuvor einen Actor hatte, entfernen
                        old_actor = self._surface_actors.pop(surface_id, None)
                        if old_actor is not None:
                            try:
                                self.plotter.remove_actor(old_actor)
                            except Exception:  # noqa: BLE001
                                pass
                        continue

                    t_clip_ms = (time.perf_counter() - t_surf_start) * 1000.0

                    # Teilmesh extrahieren
                    submesh = mesh.extract_cells(cell_indices)
                    t_extract_ms = (time.perf_counter() - t_surf_start) * 1000.0 - t_clip_ms

                    # Actor anlegen oder aktualisieren
                    actor = self._surface_actors.get(surface_id)
                    if actor is None:
                        actor_name = f"{self.SURFACE_NAME}_{surface_id}"
                        actor = self.plotter.add_mesh(
                            submesh,
                            name=actor_name,
                            scalars="plot_scalars",
                            cmap=cmap_object,
                            clim=(cbar_min, cbar_max),
                            smooth_shading=False,
                            show_scalar_bar=False,
                            reset_camera=False,
                            interpolate_before_map=False,
                        )
                        self._surface_actors[surface_id] = actor
                        if hasattr(actor, "prop") and actor.prop is not None:
                            try:
                                actor.prop.interpolation = "flat"
                            except Exception:  # noqa: BLE001
                                pass
                    else:
                        # Update bestehenden Actor
                        actor_mesh = actor.mapper.GetInput() if hasattr(actor, "mapper") else None
                        if actor_mesh is not None:
                            try:
                                actor_mesh.ShallowCopy(submesh)
                            except Exception:
                                # Fallback: ersetze Actor
                                try:
                                    self.plotter.remove_actor(actor)
                                except Exception:  # noqa: BLE001
                                    pass
                                actor_name = f"{self.SURFACE_NAME}_{surface_id}"
                                actor = self.plotter.add_mesh(
                                    submesh,
                                    name=actor_name,
                                    scalars="plot_scalars",
                                    cmap=cmap_object,
                                    clim=(cbar_min, cbar_max),
                                    smooth_shading=False,
                                    show_scalar_bar=False,
                                    reset_camera=False,
                                    interpolate_before_map=False,
                                )
                                self._surface_actors[surface_id] = actor

                        mapper = actor.mapper
                        if mapper is not None:
                            mapper.array_name = "plot_scalars"
                            mapper.scalar_range = (cbar_min, cbar_max)
                            mapper.lookup_table = self.plotter._cmap_to_lut(cmap_object)
                            mapper.interpolate_before_map = False

                    t_actor_ms = (time.perf_counter() - t_surf_start) * 1000.0 - t_clip_ms
                    if DEBUG_SURFACE_GEOMETRY:
                        print(
                            f"[SurfaceGeometry] Plot-Surface '{surface_id}': "
                            f"cells={submesh.n_cells}, "
                            f"clip={t_clip_ms:.2f}ms, "
                            f"extract+actor={t_actor_ms:.2f}ms"
                        )

                if DEBUG_SURFACE_GEOMETRY:
                    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
                    print(
                        f"[SurfaceGeometry] Plot-Surfaces: {len(enabled_surfaces)} Actors, "
                        f"gesamt {t_total_ms:.2f}ms"
                    )

        # ------------------------------------------------------------
        # 🎯 SPL-Teppich (Floor) nur anzeigen, wenn KEINE Surfaces aktiv sind
        # ------------------------------------------------------------
        surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
        has_enabled_surfaces = False
        if isinstance(surface_definitions, dict) and len(surface_definitions) > 0:
            for surface_id, surface_def in surface_definitions.items():
                try:
                    # Kompatibel zu SurfaceDefinition-Objekten und Dicts
                    enabled = bool(getattr(surface_def, "enabled", False)) if hasattr(
                        surface_def, "enabled"
                    ) else bool(surface_def.get("enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False)) if hasattr(
                        surface_def, "hidden"
                    ) else bool(surface_def.get("hidden", False))
                    points = (
                        getattr(surface_def, "points", []) if hasattr(surface_def, "points") else surface_def.get("points", [])
                    ) or []
                    if enabled and not hidden and len(points) >= 3:
                        has_enabled_surfaces = True
                        break
                except Exception:  # noqa: BLE001
                    # Bei inkonsistenten Oberflächen-Konfigurationen Floor lieber anzeigen
                    continue

        if not has_enabled_surfaces:
            # Erstelle vollständigen Floor-Mesh (ohne Surface-Maskierung oder Clipping)
            floor_mesh = build_full_floor_mesh(
                plot_x,
                plot_y,
                scalars,
                z_coords=z_coords,
                pv_module=pv,
            )

            # Plotte (oder update) den Floor nur, wenn keine Surfaces aktiv sind
            floor_actor = self.plotter.renderer.actors.get(self.FLOOR_NAME)
            if floor_actor is None:
                self.plotter.add_mesh(
                    floor_mesh,
                    name=self.FLOOR_NAME,
                    scalars="plot_scalars",
                    cmap=cmap_object,
                    clim=(cbar_min, cbar_max),
                    smooth_shading=False,
                    show_scalar_bar=False,
                    reset_camera=False,
                    interpolate_before_map=False,
                )
            else:
                if not hasattr(self, "floor_mesh"):
                    self.floor_mesh = floor_mesh.copy(deep=True)
                else:
                    self.floor_mesh.deep_copy(floor_mesh)
                mapper = floor_actor.mapper
                if mapper is not None:
                    mapper.array_name = "plot_scalars"
                    mapper.scalar_range = (cbar_min, cbar_max)
                    mapper.lookup_table = self.plotter._cmap_to_lut(cmap_object)
                    mapper.interpolate_before_map = False
        else:
            # Wenn Surfaces aktiv sind, Floor ausblenden (falls vorher gezeichnet)
            floor_actor = self.plotter.renderer.actors.get(self.FLOOR_NAME)
            if floor_actor is not None:
                try:
                    self.plotter.remove_actor(floor_actor)
                except Exception:  # noqa: BLE001
                    pass

        self._update_colorbar(colorization_mode_used, tick_step=tick_step)

        # ------------------------------------------------------------
        # Vertikale Flächen: separate SPL-Flächen rendern
        # ------------------------------------------------------------
        # Merke den aktuell verwendeten Colorization-Mode, damit
        # _update_vertical_spl_surfaces identisch reagieren kann.
        self._last_colorization_mode = colorization_mode_used
        self._update_vertical_spl_surfaces()

        self.has_data = True
        if time_mode and self.surface_mesh is not None:
            # Prüfe ob Upscaling aktiv ist - wenn ja, deaktiviere schnellen Update-Pfad
            # da Resampling zu Verzerrungen führen kann
            source_shape = tuple(pressure.shape)
            cache_entry = {
                'source_shape': source_shape,
                'source_x': geometry.source_x.copy(),
                'source_y': geometry.source_y.copy(),
                'target_x': geometry.plot_x.copy(),
                'target_y': geometry.plot_y.copy(),
                'needs_resample': geometry.requires_resample,
                'has_upscaling': geometry.was_upscaled,
                'colorbar_range': (cbar_min, cbar_max),
                'colorization_mode': colorization_mode_used,
                'color_step': cbar_step,
                'grid_shape': geometry.grid_shape,
                'expected_points': self.surface_mesh.n_points,
            }
            self._time_mode_surface_cache = cache_entry
        else:
            self._time_mode_surface_cache = None

        if camera_state is not None:
            self._restore_camera(camera_state)
        # Nur beim ersten Plot mit Daten den Zoom maximieren
        # Bei späteren Updates bleibt der Zoom erhalten (wird durch preserve_camera=True sichergestellt)
        if not self._has_plotted_data:
            self._maximize_camera_view(add_padding=True)
            self._has_plotted_data = True
        self.render()
        self._save_camera_state()
        self._colorbar_override = None
        
        # 🎯 DEBUG: Automatischer Aufruf der Surface-Referenzierungs-Analyse
        # 🎯 DEBUG: Automatischer Aufruf der Surface-Referenzierungs-Analyse
        # (nur wenn Environment-Variable gesetzt ist)
        if DEBUG_SURFACE_REF:
            self.debug_positioning_comparison()  # Vergleich Overlays vs. SPL-Plots
            self.debug_surface_referencing()     # Detaillierte Referenzierung

    def update_time_frame_values(
        self,
        sound_field_x: Iterable[float],
        sound_field_y: Iterable[float],
        sound_field_pressure: Iterable[float],
    ) -> bool:
        """
        Aktualisiert nur die Skalare für den Zeitmodus, um schnelle Frame-Wechsel zu ermöglichen.
        
        DEAKTIVIERT: Verursacht Verzerrungen im Plot. Verwende stattdessen den normalen Update-Pfad.
        """
        # Schneller Update-Pfad deaktiviert, da er zu Verzerrungen führt
        # Der normale update_spl_plot() Pfad wird stattdessen verwendet
        return False

        x_arr = np.asarray(sound_field_x, dtype=float)
        y_arr = np.asarray(sound_field_y, dtype=float)
        if x_arr.shape != cache['source_x'].shape or y_arr.shape != cache['source_y'].shape:
            return False
        # Verwende numerische Toleranz statt exakter Gleichheit
        if not (np.allclose(x_arr, cache['source_x'], rtol=1e-10, atol=1e-12) and 
                np.allclose(y_arr, cache['source_y'], rtol=1e-10, atol=1e-12)):
            return False

        pressure = np.asarray(sound_field_pressure, dtype=float)
        if pressure.ndim != 2 or tuple(pressure.shape) != cache['source_shape']:
            return False

        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
        cbar_min, cbar_max = cache['colorbar_range']
        pressure = np.clip(pressure, cbar_min, cbar_max)

        # WICHTIG: Resampling muss exakt wie im normalen Pfad funktionieren
        if cache['needs_resample']:
            # Verwende die exakt gleichen Koordinaten wie beim ersten Plot
            pressure = self._resample_values_to_grid(
                pressure,
                cache['source_x'],
                cache['source_y'],
                cache['target_x'],
                cache['target_y'],
            )
        else:
            # Kein Resampling nötig, aber sicherstellen dass Shape stimmt
            if pressure.shape != cache['source_shape']:
                return False

        if pressure.shape != cache['grid_shape']:
            return False

        if cache['colorization_mode'] == 'Color step':
            pressure = self._quantize_to_steps(pressure, cache['color_step'])

        # WICHTIG: Verwende exakt die gleiche Ravel-Ordnung wie beim Mesh-Build
        flat_scalars = np.asarray(pressure, dtype=float).ravel(order='F')
        if flat_scalars.size != cache['expected_points']:
            return False

        if not self._update_surface_scalars(flat_scalars):
            return False

        self.render()
        return True

    def update_overlays(self, settings, container):
        """Aktualisiert Zusatzobjekte (Achsen, Lautsprecher, Messpunkte)."""
        t_start = time.perf_counter()
        
        # Speichere Container-Referenz für Z-Koordinaten-Zugriff
        self.container = container
        t_container = time.perf_counter()
        
        # Wenn kein gültiger calculation_spl-Container vorhanden ist, entferne vertikale SPL-Flächen
        if not hasattr(container, "calculation_spl"):
            self._clear_vertical_spl_surfaces()
        t_clear = time.perf_counter()
        
        # Wir vergleichen Hash-Signaturen pro Kategorie und zeichnen nur dort neu,
        # wo sich Parameter geändert haben. Das verhindert unnötiges Entfernen
        # und erneutes Hinzufügen zahlreicher PyVista-Actor.
        t_sig_start = time.perf_counter()
        signatures = self._compute_overlay_signatures(settings, container)
        t_sig_end = time.perf_counter()
        previous = self._last_overlay_signatures or {}
        if not previous:
            categories_to_refresh = set(signatures.keys())
        else:
            categories_to_refresh = {
                key for key, value in signatures.items() if value != previous.get(key)
            }
        t_compare = time.perf_counter()
        
        if not categories_to_refresh:
            self._last_overlay_signatures = signatures
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
            return
        
        prev_debug_state = getattr(self.overlay_helper, 'DEBUG_ON_TOP', False)
        t_draw_start = time.perf_counter()
        try:
            self.overlay_helper.DEBUG_ON_TOP = True
            t_axis_start = t_speakers_start = t_surfaces_start = t_impulse_start = t_draw_start
            t_axis_end = t_speakers_end = t_surfaces_end = t_impulse_end = t_draw_start
            
            if 'axis' in categories_to_refresh:
                t_axis_start = time.perf_counter()
                self.overlay_helper.draw_axis_lines(settings)
                t_axis_end = time.perf_counter()
            if 'surfaces' in categories_to_refresh:
                t_surfaces_start = time.perf_counter()
                self.overlay_helper.draw_surfaces(settings)
                t_surfaces_end = time.perf_counter()
            if 'speakers' in categories_to_refresh:
                t_speakers_start = time.perf_counter()
                cabinet_lookup = self.overlay_helper.build_cabinet_lookup(container)
                t_cabinet_lookup = time.perf_counter()
                self.overlay_helper.draw_speakers(settings, container, cabinet_lookup)
                t_speakers_end = time.perf_counter()
            if 'impulse' in categories_to_refresh:
                t_impulse_start = time.perf_counter()
                self.overlay_helper.draw_impulse_points(settings)
                t_impulse_end = time.perf_counter()
        finally:
            self.overlay_helper.DEBUG_ON_TOP = prev_debug_state
        t_draw_end = time.perf_counter()
        
        self._last_overlay_signatures = signatures
        t_signature_save = time.perf_counter()
        
        # 🎯 Beim ersten Start: Zoom auf das Default-Surface einstellen (nach dem Zeichnen aller Overlays)
        t_zoom_start = time.perf_counter()
        if (not getattr(self, "_did_initial_overlay_zoom", False)) and 'surfaces' in categories_to_refresh:
            self._zoom_to_default_surface()
        t_zoom_end = time.perf_counter()
        
        # Verzögertes Rendering: Timer starten/stoppen für Batch-Updates
        self._schedule_render()
        if not self._rotate_active and not self._pan_active:
            self._save_camera_state()
        t_end = time.perf_counter()
        
        if DEBUG_OVERLAY_PERF:
            t_total = (t_end - t_start) * 1000
            t_sig = (t_sig_end - t_sig_start) * 1000
            t_compare_time = (t_compare - t_sig_end) * 1000
            t_draw_total = (t_draw_end - t_draw_start) * 1000
            t_axis = (t_axis_end - t_axis_start) * 1000 if 'axis' in categories_to_refresh else 0
            t_surfaces = (t_surfaces_end - t_surfaces_start) * 1000 if 'surfaces' in categories_to_refresh else 0
            t_speakers = (t_speakers_end - t_speakers_start) * 1000 if 'speakers' in categories_to_refresh else 0
            t_cabinet = (t_cabinet_lookup - t_speakers_start) * 1000 if 'speakers' in categories_to_refresh else 0
            t_impulse = (t_impulse_end - t_impulse_start) * 1000 if 'impulse' in categories_to_refresh else 0
            t_zoom = (t_zoom_end - t_zoom_start) * 1000
            print(f"  ├─ Signaturen speichern: {(t_signature_save-t_draw_end)*1000:.2f}ms")
            print(f"  └─ Rest (Timer/Save): {(t_end-t_signature_save)*1000:.2f}ms")

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

    def _schedule_render(self):
        """Plant ein verzögertes Rendering (500ms), um mehrere schnelle Updates zu bündeln."""
        self._pending_render = True
        # Stoppe laufenden Timer und starte neu (debouncing)
        self._render_timer.stop()
        self._render_timer.start(500)  # 500ms Verzögerung
        if DEBUG_OVERLAY_PERF:
            if not hasattr(self, '_render_schedule_count'):
                self._render_schedule_count = 0
            self._render_schedule_count += 1

    def _delayed_render(self):
        """Führt das verzögerte Rendering aus, wenn der Timer abgelaufen ist."""
        if self._pending_render:
            t_render_start = time.perf_counter() if DEBUG_OVERLAY_PERF else None
            self._pending_render = False
            self.render()
            if DEBUG_OVERLAY_PERF and t_render_start is not None:
                self._render_schedule_count = 0

    def _get_vertical_color_limits(self) -> tuple[float, float]:
        """
        Liefert die Farbskalen-Grenzen (min, max) für vertikale SPL-Flächen.
        Bevorzugt aktuelle Override-Range aus der Colorbar, sonst Settings-Range.
        """
        if self._colorbar_override:
            try:
                cmin = float(self._colorbar_override.get("min", 0.0))
                cmax = float(self._colorbar_override.get("max", 0.0))
                return cmin, cmax
            except Exception:
                pass
        try:
            rng = self.settings.colorbar_range
            return float(rng["min"]), float(rng["max"])
        except Exception:
            return 0.0, 120.0

    # ------------------------------------------------------------------
    # Vertikale SPL-Flächen (senkrechte Surfaces)
    # ------------------------------------------------------------------
    def _clear_vertical_spl_surfaces(self) -> None:
        """Entfernt alle zusätzlichen SPL-Meshes für senkrechte Flächen."""
        if not hasattr(self, "plotter") or self.plotter is None:
            self._vertical_surface_meshes.clear()
            return
        for name, actor in list(self._vertical_surface_meshes.items()):
            try:
                # Actor kann entweder direkt der Actor oder nur der Name sein
                if isinstance(name, str):
                    self.plotter.remove_actor(name)
            except Exception:
                pass
        self._vertical_surface_meshes.clear()

    def _update_vertical_spl_surfaces(self) -> None:
        """
        Zeichnet / aktualisiert SPL-Flächen für senkrechte Surfaces auf Basis von
        calculation_spl['surface_samples'] und calculation_spl['surface_fields'].
        
        🎯 WICHTIG: Im Texture-Modus werden senkrechte Flächen bereits als Textur gerendert,
        daher werden sie hier übersprungen, um Doppel-Rendering zu vermeiden.
        """
        # Prüfe ob Texture-Modus aktiv ist
        render_mode = getattr(self.settings, "spl_surface_render_mode", "mesh") or "mesh"
        is_texture_mode = render_mode.lower() == "texture"
        
        # Im Texture-Modus: Senkrechte Flächen werden bereits als Textur gerendert
        # Entferne alte Mesh-Actors für senkrechte Flächen, falls vorhanden
        if is_texture_mode:
            self._clear_vertical_spl_surfaces()
            return
        
        container = self.container
        if container is None or not hasattr(container, "calculation_spl"):
            self._clear_vertical_spl_surfaces()
            return

        calc_spl = getattr(container, "calculation_spl", {}) or {}
        sample_payloads = calc_spl.get("surface_samples")
        if not isinstance(sample_payloads, list):
            self._clear_vertical_spl_surfaces()
            return

        # Aktueller Surface-Status (enabled/hidden) aus den Settings
        surface_definitions = getattr(self.settings, "surface_definitions", {})
        if not isinstance(surface_definitions, dict):
            surface_definitions = {}

        # Vertikale Surfaces analog zu prepare_plot_geometry behandeln:
        # lokales (u,v)-Raster + strukturiertes Mesh über build_vertical_surface_mesh.
        new_vertical_meshes: dict[str, Any] = {}

        # Aktuellen Colorization-Mode verwenden (wie für die Hauptfläche).
        colorization_mode = getattr(self, "_last_colorization_mode", None)
        if colorization_mode not in {"Color step", "Gradient"}:
            colorization_mode = getattr(self.settings, "colorization_mode", "Gradient")
        try:
            cbar_range = getattr(self.settings, "colorbar_range", {})
            cbar_step = float(cbar_range.get("step", 0.0))
        except Exception:
            cbar_step = 0.0
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0
        for payload in sample_payloads:
            # Nur Payloads verarbeiten, die explizit als "vertical" markiert sind.
            kind = payload.get("kind", "planar")
            if kind != "vertical":
                continue

            surface_id = payload.get("surface_id")
            if surface_id is None:
                continue

            # Nur Surfaces zeichnen, die aktuell enabled und nicht hidden sind
            surf_def = surface_definitions.get(surface_id)
            if surf_def is None:
                continue
            if hasattr(surf_def, "to_dict"):
                surf_data = surf_def.to_dict()
            elif isinstance(surf_def, dict):
                surf_data = surf_def
            else:
                surf_data = {
                    "enabled": getattr(surf_def, "enabled", False),
                    "hidden": getattr(surf_def, "hidden", False),
                    "points": getattr(surf_def, "points", []),
                }
            if not surf_data.get("enabled", False) or surf_data.get("hidden", False):
                continue

            # Lokale vertikale Plot-Geometrie aufbauen (u,v-Grid, SPL-Werte, Maske)
            try:
                geom: VerticalPlotGeometry | None = prepare_vertical_plot_geometry(
                    surface_id,
                    self.settings,
                    container,
                    default_upscale=self.UPSCALE_FACTOR,
                )
            except Exception:
                geom = None

            if geom is None:
                continue

            # Strukturiertes Mesh in Weltkoordinaten erstellen
            try:
                grid = build_vertical_surface_mesh(geom, pv_module=pv)
            except Exception:
                continue

            # Color step: Werte in diskrete Stufen quantisieren, analog zur Hauptfläche.
            if is_step_mode and "plot_scalars" in grid.array_names:
                try:
                    vals = np.asarray(grid["plot_scalars"], dtype=float)
                    grid["plot_scalars"] = self._quantize_to_steps(vals, cbar_step)
                except Exception:
                    pass

            actor_name = f"vertical_spl_{surface_id}"
            # Entferne ggf. alten Actor
            try:
                if actor_name in self.plotter.renderer.actors:
                    self.plotter.remove_actor(actor_name)
            except Exception:
                pass

            # Farbschema und CLim an das Haupt-SPL anlehnen
            cbar_min, cbar_max = self._get_vertical_color_limits()
            try:
                actor = self.plotter.add_mesh(
                    grid,
                    name=actor_name,
                    scalars="plot_scalars",
                    cmap="jet",
                    clim=(cbar_min, cbar_max),
                    # Gradient: weiche Darstellung, Color step: harte Stufen.
                    smooth_shading=not is_step_mode,
                    show_scalar_bar=False,
                    reset_camera=False,
                    interpolate_before_map=not is_step_mode,
                )
                # Im Color-Step-Modus explizit flache Interpolation erzwingen,
                # damit die Stufen wie bei der horizontalen Fläche erscheinen.
                if is_step_mode and hasattr(actor, "prop") and actor.prop is not None:
                    try:
                        actor.prop.interpolation = "flat"
                    except Exception:  # noqa: BLE001
                        pass
                new_vertical_meshes[actor_name] = actor
            except Exception:
                continue

        # Alte Actors entfernen, die nicht mehr gebraucht werden
        for old_name in list(self._vertical_surface_meshes.keys()):
            if old_name not in new_vertical_meshes:
                try:
                    if old_name in self.plotter.renderer.actors:
                        self.plotter.remove_actor(old_name)
                except Exception:
                    pass

        self._vertical_surface_meshes = new_vertical_meshes

    # ------------------------------------------------------------------
    # Kamera-Zustand
    # ------------------------------------------------------------------
    def _camera_debug(self, message: str) -> None:
        if self._camera_debug_enabled:
            print(f"[CAM DEBUG] {message}")

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

    def _init_camera_debug_flag(self) -> bool:
        env_value = os.environ.get("LFO_DEBUG_CAMERA")
        if env_value is None:
            # Debug standardmäßig deaktiviert
            return False
        env_value = env_value.strip().lower()
        if env_value in {"0", "false", "off"}:
            return False
        if env_value in {"1", "true", "on"}:
            return True
        try:
            return bool(int(env_value))
        except ValueError:
            return True

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

    def _setup_time_control(self):
        try:
            self.time_control = SPLTimeControlBar(self.widget)
            self.time_control.hide()
            self.time_control.valueChanged.connect(self._handle_time_slider_change)
        except Exception:
            self.time_control = None

    def set_time_slider_callback(self, callback):
        self._time_slider_callback = callback

    def update_time_control(self, active: bool, frames: int, value: int, simulation_time: float = 0.2):
        if self.time_control is None:
            return
        if not active:
            self.time_control.hide()
            return
        self.time_control.configure(frames, value, simulation_time)
        self.time_control.show()

    def _handle_time_slider_change(self, value: int):
        if callable(self._time_slider_callback):
            self._time_slider_callback(int(value))

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

    # _add_scene_frame() wurde entfernt - nicht mehr benötigt
    
    def _zoom_to_default_surface(self) -> None:
        """Stellt den Zoom auf das Default-Surface ein."""
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
                    # Aktiviere Parallelprojektion für die Draufsicht, damit der Zoom
                    # ausschließlich über parallel_scale gesteuert wird (orthografische Ansicht).
                    if hasattr(cam, "parallel_projection"):
                        cam.parallel_projection = True

                    # Positioniere die Kamera über dem Surface mit moderatem Abstand.
                    # Der Abstand ist bei Parallelprojektion weniger kritisch, sollte aber
                    # groß genug sein, um numerische Effekte zu vermeiden.
                    distance = max_extent * 1.0
                    min_distance = 20.0
                    distance = max(distance, min_distance)
                    cam.position = (center_x, center_y, center_z + distance)
                    cam.focal_point = (center_x, center_y, center_z)
                    cam.up = (0, 1, 0)
                
                # Setze den Zoom basierend auf den Bounds mit mehr Padding
                # Parallelprojektion: parallel_scale bestimmt die halbe sichtbare Höhe.
                # Wir wählen einen Faktor so, dass das Surface ca. 90–95% der Widget-Höhe füllt.
                if getattr(cam, 'parallel_projection', False):
                    if hasattr(cam, 'parallel_scale'):
                        # Für eine fast widget-füllende Darstellung:
                        # visible_height ≈ 2 * parallel_scale
                        # Wir wollen max_extent (Breite oder Höhe) ≈ 0.9 * visible_height
                        # → parallel_scale ≈ max_extent / (2 * 0.9) ≈ max_extent * 0.56
                        scale = max_extent * 0.56
                        cam.parallel_scale = scale
                else:
                    if hasattr(cam, 'view_angle'):
                        # Berechne view_angle basierend auf der Entfernung
                        # Verwende einen etwas kleineren Faktor (1.1), damit wir etwas näher dran sind
                        # Der Winkel sollte so sein, dass max_extent * 1.1 im Sichtfeld ist
                        visible_extent = max_extent * 1.1
                        angle = 2.0 * np.arctan(visible_extent / (2.0 * distance)) * 180.0 / np.pi
                        angle = max(30.0, min(60.0, angle))  # Mindestwinkel erhöht auf 30°
                        cam.view_angle = angle
                
                # Reset Clipping Range
                if hasattr(cam, 'ResetClippingRange'):
                    try:
                        cam.ResetClippingRange()
                    except Exception:
                        pass

                # Merke, dass der initiale Overlay-Zoom gesetzt wurde
                self._did_initial_overlay_zoom = True

                # Verhindere, dass render() direkt danach einen alten Kamera-State wiederherstellt
                self._skip_next_render_restore = True
                # Render, um die Änderungen anzuzeigen
                self.render()
                # Nach dem Rendern den aktuellen Kamera-State als neuen Referenzzustand speichern
                self._camera_state = self._capture_camera()
        except Exception as e:
            import traceback
            traceback.print_exc()
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
    def _configure_plotter(self, configure_camera: bool = True):
        self.plotter.set_background('white')
        self.plotter.show_axes()
        try:
            self.plotter.disable_eye_dome_lighting()
        except Exception:  # noqa: BLE001
            pass
        # Anti-Aliasing nach globaler Vorgabe konfigurieren
        try:
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

    def _get_colorbar_params(self, phase_mode: bool) -> dict[str, float]:
        if self._colorbar_override:
            return self._colorbar_override
        if phase_mode:
            return {
                'min': 0.0,
                'max': 180.0,
                'step': 10.0,
                'tick_step': 30.0,
                'label': "Phase difference (°)",
            }
        if self._time_mode_active:
            # Fester Wertebereich für Zeitmodus (in Pascal)
            # Standardwert auf 50 Pa gesetzt für höheren Kontrast
            max_pressure = float(getattr(self.settings, 'fem_time_plot_max_pressure', 50.0) or 50.0)
            return {
                'min': -max_pressure,
                'max': max_pressure,
                'step': max_pressure / 5.0,
                'tick_step': max_pressure / 5.0,
                'label': "Pressure (Pa)",
            }
        rng = self.settings.colorbar_range
        return {
            'min': float(rng['min']),
            'max': float(rng['max']),
            'step': float(rng['step']),
            'tick_step': float(rng['tick_step']),
            'label': "SPL (dB)",
        }

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

    def _update_colorbar(self, colorization_mode: str, tick_step: float | None = None):
        try:
            self._render_colorbar(colorization_mode, tick_step=tick_step)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Colorbar konnte nicht aktualisiert werden: {exc}")

    def _render_colorbar(self, colorization_mode: str, force: bool = False, tick_step: float | None = None):
        """Aktualisiert die Colorbar.

        - Nutzt ein Caching der ScalarMappable-Instanz, um bei unverändertem Modus
          nur Norm/Cmap anzupassen.
        - `force=True` erzwingt einen kompletten Neuaufbau (z. B. beim Init).
        """
        if self._colorbar_override:
            params = self._colorbar_override
            cbar_min = float(params['min'])
            cbar_max = float(params['max'])
            cbar_step = float(params['step'])
            tick_step_val = tick_step if tick_step is not None else float(params['tick_step'])
        else:
            cbar_min = float(self.settings.colorbar_range['min'])
            cbar_max = float(self.settings.colorbar_range['max'])
            cbar_step = float(self.settings.colorbar_range['step'])
            tick_step_val = float(self.settings.colorbar_range['tick_step'])

        if tick_step_val <= 0:
            tick_step_val = max((cbar_max - cbar_min) / 5.0, 1.0)

        cbar_ticks = np.arange(cbar_min, cbar_max + tick_step_val, tick_step_val)

        if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
            if force or self.cbar is None or self._colorbar_mappable is None:
                self.colorbar_ax.cla()

        boundaries = None
        spacing = 'uniform'
        base_cmap = self.PHASE_CMAP if self._phase_mode_active else cm.get_cmap('jet')
        is_step_mode = colorization_mode == 'Color step' and cbar_step > 0
        if is_step_mode:
            levels = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            if levels[-1] < cbar_max:
                levels = np.append(levels, cbar_max)
            if levels.size < 2:
                levels = np.array([cbar_min, cbar_max])

            num_segments = max(1, len(levels) - 1)
            if hasattr(base_cmap, "resampled"):
                sampled_cmap = base_cmap.resampled(num_segments)
            else:
                sampled_cmap = cm.get_cmap(base_cmap, num_segments)
            sample_points = (np.arange(num_segments, dtype=float) + 0.5) / max(num_segments, 1)
            color_list = sampled_cmap(sample_points)
            cmap = ListedColormap(color_list)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            boundaries = levels
            spacing = 'proportional'
            mode_signature: tuple[str, tuple[float, ...] | None] = (
                'phase_step' if self._phase_mode_active else 'step',
                tuple(float(level) for level in levels),
            )
        else:
            cmap = base_cmap
            norm = Normalize(vmin=cbar_min, vmax=cbar_max)
            mode_signature = ('phase_grad' if self._phase_mode_active else 'gradient', None)

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
        if self.cbar is not None:
            label = None
            if self._colorbar_override and 'label' in self._colorbar_override:
                label = str(self._colorbar_override['label'])
            elif self._time_mode_active:
                label = "Pressure (Pa)"
            elif self._phase_mode_active:
                label = "Phase difference (°)"
            else:
                label = "SPL (dB)"
            self.cbar.set_label(label, fontsize=8)
            if self._phase_mode_active:
                self.cbar.ax.set_ylim(cbar_max, cbar_min)
            else:
                self.cbar.ax.set_ylim(cbar_min, cbar_max)

        self._colorbar_mode = mode_signature

    @staticmethod
    def _has_valid_data(x, y, pressure) -> bool:
        if x is None or y is None or pressure is None:
            return False
        try:
            return len(x) > 0 and len(y) > 0 and len(pressure) > 0
        except TypeError:
            return False

    @staticmethod
    def _compute_surface_signature(x: np.ndarray, y: np.ndarray) -> tuple:
        if x.size == 0 or y.size == 0:
            return (0, 0, 0.0, 0.0, 0.0, 0.0)
        return (
            int(x.size),
            float(x[0]),
            float(x[-1]),
            int(y.size),
            float(y[0]),
            float(y[-1]),
        )

    def _update_surface_scalars(self, flat_scalars: np.ndarray) -> bool:
        if self.surface_mesh is None:
            return False

        if flat_scalars.size == self.surface_mesh.n_points:
            self.surface_mesh.point_data['plot_scalars'] = flat_scalars
            if hasattr(self.surface_mesh, "modified"):
                self.surface_mesh.modified()
            return True

        if flat_scalars.size == self.surface_mesh.n_cells:
            self.surface_mesh.cell_data['plot_scalars'] = flat_scalars
            if hasattr(self.surface_mesh, "modified"):
                self.surface_mesh.modified()
            return True

        return False

    # ------------------------------------------------------------------
    # Bilineare Interpolation und Textur-Rendering für horizontale Surfaces
    # ------------------------------------------------------------------
    @staticmethod
    def _bilinear_interpolate_grid(
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        xq: np.ndarray,
        yq: np.ndarray,
    ) -> np.ndarray:
        """
        Bilineare Interpolation auf einem regulären Grid.

        source_x: 1D-Array der X-Koordinaten (nx)
        source_y: 1D-Array der Y-Koordinaten (ny)
        values  : 2D-Array (ny, nx) mit SPL-Werten
        xq, yq  : 1D-Arrays mit Abfragepunkten
        """
        source_x = np.asarray(source_x, dtype=float).reshape(-1)
        source_y = np.asarray(source_y, dtype=float).reshape(-1)
        vals = np.asarray(values, dtype=float)
        xq = np.asarray(xq, dtype=float).reshape(-1)
        yq = np.asarray(yq, dtype=float).reshape(-1)

        ny, nx = vals.shape
        if ny != len(source_y) or nx != len(source_x):
            raise ValueError(
                f"_bilinear_interpolate_grid: Shape mismatch values={vals.shape}, "
                f"expected=({len(source_y)}, {len(source_x)})"
            )

        # Punkte außerhalb des Grids werden auf den Rand geclippt
        x_min, x_max = source_x[0], source_x[-1]
        y_min, y_max = source_y[0], source_y[-1]
        xq_clip = np.clip(xq, x_min, x_max)
        yq_clip = np.clip(yq, y_min, y_max)

        # Linken/unteren Index finden
        ix1 = np.searchsorted(source_x, xq_clip, side="right") - 1
        iy1 = np.searchsorted(source_y, yq_clip, side="right") - 1
        ix1 = np.clip(ix1, 0, nx - 2)
        iy1 = np.clip(iy1, 0, ny - 2)
        ix2 = ix1 + 1
        iy2 = iy1 + 1

        x1 = source_x[ix1]
        x2 = source_x[ix2]
        y1 = source_y[iy1]
        y2 = source_y[iy2]

        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(x2 != x1, (xq_clip - x1) / (x2 - x1), 0.0)
            ty = np.where(y2 != y1, (yq_clip - y1) / (y2 - y1), 0.0)

        v11 = vals[iy1, ix1]
        v21 = vals[iy1, ix2]
        v12 = vals[iy2, ix1]
        v22 = vals[iy2, ix2]

        interp = (
            (1 - tx) * (1 - ty) * v11
            + tx * (1 - ty) * v21
            + (1 - tx) * ty * v12
            + tx * ty * v22
        )
        return interp

    def _render_surfaces_textured(
        self,
        geometry,
        original_plot_values: np.ndarray,
        cbar_min: float,
        cbar_max: float,
        cmap_object: str | Any,
        colorization_mode: str,
        cbar_step: float,
    ) -> None:
        """
        Renderpfad für horizontale Surfaces als 2D-Texturen.

        - Pro Surface wird ein lokales 2D-Bild in Welt-X/Y berechnet.
        - SPL-Werte stammen aus original_plot_values (coarse Grid) via bilinearer Interpolation.
        - Die Textur wird auf ein flaches StructuredGrid in derselben XY-Position gelegt.
        """
        if not hasattr(self, "plotter") or self.plotter is None:
            return

        try:
            import pyvista as pv  # type: ignore
        except Exception:
            # Fallback: kein Texture-Rendering möglich
            if DEBUG_SURFACE_GEOMETRY:
                print("[SurfaceGeometry] Texture-Rendering nicht verfügbar (PyVista-Import fehlgeschlagen).")
            return

        # Quelle: Berechnungsraster
        source_x = np.asarray(getattr(geometry, "source_x", []), dtype=float)
        source_y = np.asarray(getattr(geometry, "source_y", []), dtype=float)
        values = np.asarray(original_plot_values, dtype=float)
        if values.shape != (len(source_y), len(source_x)):
            if DEBUG_SURFACE_GEOMETRY:
                print(
                    "[SurfaceGeometry] Texture-Rendering abgebrochen: "
                    f"Shape-Mismatch original_values={values.shape}, "
                    f"source=({len(source_y)}, {len(source_x)})"
                )
            return

        # Auflösung der Textur in Metern (XY)
        # Standard: 0.02m (2cm pro Pixel) für schärfere Darstellung (vorher: 0.05m)
        tex_res = float(getattr(self.settings, "spl_surface_texture_resolution", 0.02) or 0.02)

        # Colormap vorbereiten
        if isinstance(cmap_object, str):
            base_cmap = cm.get_cmap(cmap_object)
        else:
            base_cmap = cmap_object
        norm = Normalize(vmin=cbar_min, vmax=cbar_max)
        is_step_mode = colorization_mode == "Color step" and cbar_step > 0

        # Aktive Surfaces ermitteln
        surface_definitions = getattr(self.settings, "surface_definitions", {}) or {}
        enabled_surfaces: list[tuple[str, list[dict[str, float]], Any]] = []
        if isinstance(surface_definitions, dict):
            for surface_id, surface_def in surface_definitions.items():
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, "enabled", False))
                    hidden = bool(getattr(surface_def, "hidden", False))
                    points = getattr(surface_def, "points", []) or []
                    surface_obj = surface_def
                else:
                    enabled = bool(surface_def.get("enabled", False))
                    hidden = bool(surface_def.get("hidden", False))
                    points = surface_def.get("points", []) or []
                    surface_obj = surface_def
                if enabled and not hidden and len(points) >= 3:
                    enabled_surfaces.append((str(surface_id), points, surface_obj))

        # Nicht mehr benötigte Textur-Actors entfernen
        active_ids = {sid for sid, _, _ in enabled_surfaces}
        for sid, texture_data in list(self._surface_texture_actors.items()):
            if sid not in active_ids:
                try:
                    actor = texture_data.get('actor') if isinstance(texture_data, dict) else texture_data
                    if actor is not None:
                        self.plotter.remove_actor(actor)
                except Exception:
                    pass
                self._surface_texture_actors.pop(sid, None)

        if not enabled_surfaces:
            print(f"[SurfaceGeometry] Texture-Rendering: Keine enabled Surfaces gefunden")
            return

        print(f"[SurfaceGeometry] Texture-Rendering: {len(enabled_surfaces)} enabled Surface(s) gefunden")
        for surface_id, points, surface_obj in enabled_surfaces:
            try:
                t_start = time.perf_counter()
                poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                poly_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
                if poly_x.size == 0 or poly_y.size == 0:
                    continue

                xmin, xmax = float(poly_x.min()), float(poly_x.max())
                ymin, ymax = float(poly_y.min()), float(poly_y.max())
                
                # 🎯 Berechne Planmodell für geneigte Flächen
                # Konvertiere Punkte in Dict-Format für derive_surface_plane
                dict_points = [
                    {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                    for p in points
                ]
                plane_model, _ = derive_surface_plane(dict_points)
                
                # 🎯 Prüfe ob Fläche senkrecht ist (nicht planar)
                # Senkrechte Flächen benötigen spezielle Behandlung mit (u,v)-Koordinaten
                is_vertical = False
                orientation = None
                wall_axis = None
                wall_value = None
                
                if plane_model is None:
                    # Prüfe ob es eine senkrechte Fläche ist
                    x_span = float(np.ptp(poly_x)) if poly_x.size > 0 else 0.0
                    y_span = float(np.ptp(poly_y)) if poly_y.size > 0 else 0.0
                    z_span = float(np.ptp(poly_z)) if poly_z.size > 0 else 0.0
                    
                    # Senkrechte Fläche: XY-Projektion ist fast eine Linie, aber signifikante Z-Höhe
                    eps_line = 1e-6
                    has_height = z_span > 1e-3
                    
                    if y_span < eps_line and x_span >= eps_line and has_height:
                        # X-Z-Wand: y ≈ const (Fläche verläuft in X-Z-Richtung)
                        is_vertical = True
                        orientation = "xz"
                        wall_axis = "y"
                        wall_value = float(np.mean(poly_y))
                    elif x_span < eps_line and y_span >= eps_line and has_height:
                        # Y-Z-Wand: x ≈ const (Fläche verläuft in Y-Z-Richtung)
                        is_vertical = True
                        orientation = "yz"
                        wall_axis = "x"
                        wall_value = float(np.mean(poly_x))
                    
                    if not is_vertical:
                        # Fallback: Wenn kein Planmodell gefunden wurde, verwende konstante Höhe (Mittelwert)
                        plane_model = {
                            "mode": "constant",
                            "base": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                            "slope": 0.0,
                            "intercept": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                        }

                # 🎯 SPEZIELLER PFAD FÜR SENKRECHTE FLÄCHEN
                if is_vertical:
                    # Senkrechte Flächen verwenden (u,v)-Koordinaten statt (x,y)
                    # u = x oder y (je nach Orientierung), v = z
                    if orientation == "xz":
                        # X-Z-Wand: u = x, v = z, y = const
                        poly_u = poly_x
                        poly_v = poly_z
                        umin, umax = xmin, xmax
                        vmin, vmax = float(poly_z.min()), float(poly_z.max())
                    else:  # orientation == "yz"
                        # Y-Z-Wand: u = y, v = z, x = const
                        poly_u = poly_y
                        poly_v = poly_z
                        umin, umax = ymin, ymax
                        vmin, vmax = float(poly_z.min()), float(poly_z.max())
                    
                    # Erstelle (u,v)-Grid für senkrechte Fläche
                    margin = tex_res * 0.5
                    u_start = umin - margin
                    u_end = umax + margin
                    v_start = vmin - margin
                    v_end = vmax + margin
                    
                    num_u = int(np.ceil((u_end - u_start) / tex_res)) + 1
                    num_v = int(np.ceil((v_end - v_start) / tex_res)) + 1
                    
                    us = np.linspace(u_start, u_end, num_u, dtype=float)
                    vs = np.linspace(v_start, v_end, num_v, dtype=float)
                    if us.size < 2 or vs.size < 2:
                        continue
                    
                    U, V = np.meshgrid(us, vs, indexing="xy")
                    points_uv = np.column_stack((U.ravel(), V.ravel()))
                    
                    # Maske im Polygon (u,v-Ebene)
                    poly_path_uv = Path(np.column_stack((poly_u, poly_v)))
                    inside_uv = poly_path_uv.contains_points(points_uv)
                    inside_uv = inside_uv.reshape(U.shape)
                    
                    if not np.any(inside_uv):
                        if DEBUG_SURFACE_GEOMETRY:
                            print(
                                f"[SurfaceGeometry] Texture-Surface '{surface_id}': "
                                f"keine Pixel im Polygon gefunden (senkrecht)"
                            )
                        continue
                    
                    # 🎯 Hole SPL-Werte aus vertikalen Samples
                    # Verwende prepare_vertical_plot_geometry für SPL-Werte
                    try:
                        geom_vertical = prepare_vertical_plot_geometry(
                            surface_id,
                            self.settings,
                            self.container,
                            default_upscale=1,  # Kein Upscaling, da wir bereits feines Grid haben
                        )
                    except Exception:
                        geom_vertical = None
                    
                    if geom_vertical is None:
                        if DEBUG_SURFACE_GEOMETRY:
                            print(
                                f"[SurfaceGeometry] Texture-Surface '{surface_id}': "
                                f"Keine vertikalen Samples gefunden"
                            )
                        continue
                    
                    # Interpoliere SPL-Werte auf (u,v)-Grid
                    # geom_vertical enthält plot_u, plot_v, plot_values
                    plot_u_geom = np.asarray(geom_vertical.plot_u, dtype=float)
                    plot_v_geom = np.asarray(geom_vertical.plot_v, dtype=float)
                    plot_values_geom = np.asarray(geom_vertical.plot_values, dtype=float)
                    
                    # Bilineare Interpolation auf (u,v)-Grid
                    spl_flat_uv = self._bilinear_interpolate_grid(
                        plot_u_geom,
                        plot_v_geom,
                        plot_values_geom,
                        U.ravel(),
                        V.ravel(),
                    )
                    spl_img_uv = spl_flat_uv.reshape(U.shape)
                    
                    # Werte clippen
                    spl_clipped_uv = np.clip(spl_img_uv, cbar_min, cbar_max)
                    if is_step_mode:
                        spl_clipped_uv = self._quantize_to_steps(spl_clipped_uv, cbar_step)
                    
                    # In Farbe umsetzen
                    rgba_uv = base_cmap(norm(spl_clipped_uv))
                    
                    # Alpha-Maske
                    alpha_mask_uv = inside_uv & np.isfinite(spl_clipped_uv)
                    rgba_uv[..., 3] = np.where(alpha_mask_uv, 1.0, 0.0)
                    
                    # Nach uint8 wandeln
                    img_rgba_uv = (np.clip(rgba_uv, 0.0, 1.0) * 255).astype(np.uint8)
                    
                    # 🎯 Erstelle 3D-Grid für senkrechte Fläche
                    if orientation == "xz":
                        # X-Z-Wand: X = U, Y = wall_value, Z = V
                        X_3d = U
                        Y_3d = np.full_like(U, wall_value)
                        Z_3d = V
                    else:  # orientation == "yz"
                        # Y-Z-Wand: X = wall_value, Y = U, Z = V
                        X_3d = np.full_like(U, wall_value)
                        Y_3d = U
                        Z_3d = V
                    
                    grid = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
                    
                    # Textur-Koordinaten für senkrechte Fläche
                    try:
                        grid.texture_map_to_plane(inplace=True)
                        t_coords = grid.point_data.get("TCoords")
                    except Exception:
                        t_coords = None
                    
                    # Manuelle Textur-Koordinaten-Berechnung für senkrechte Fläche
                    bounds = grid.bounds
                    if bounds is not None and len(bounds) >= 6:
                        if orientation == "xz":
                            u_min, u_max = bounds[0], bounds[1]  # X-Bounds
                            v_min, v_max = bounds[4], bounds[5]  # Z-Bounds
                        else:  # yz
                            u_min, u_max = bounds[2], bounds[3]  # Y-Bounds
                            v_min, v_max = bounds[4], bounds[5]  # Z-Bounds
                        
                        u_span = u_max - u_min
                        v_span = v_max - v_min
                        
                        if u_span > 1e-10 and v_span > 1e-10:
                            if orientation == "xz":
                                u_coords = (X_3d - u_min) / u_span
                            else:  # yz
                                u_coords = (Y_3d - u_min) / u_span
                            v_coords_raw = (Z_3d - v_min) / v_span
                            
                            u_coords = np.clip(u_coords, 0.0, 1.0)
                            v_coords_raw = np.clip(v_coords_raw, 0.0, 1.0)
                            v_coords = 1.0 - v_coords_raw  # PyVista-Konvention: v=0 oben
                            
                            # Achsen-Invertierung (wie bei planaren Flächen)
                            invert_x = False
                            invert_y = True
                            swap_axes = False
                            
                            if isinstance(surface_obj, SurfaceDefinition):
                                if hasattr(surface_obj, "invert_texture_x"):
                                    invert_x = bool(getattr(surface_obj, "invert_texture_x"))
                                if hasattr(surface_obj, "invert_texture_y"):
                                    invert_y = bool(getattr(surface_obj, "invert_texture_y"))
                                if hasattr(surface_obj, "swap_texture_axes"):
                                    swap_axes = bool(getattr(surface_obj, "swap_texture_axes"))
                            elif isinstance(surface_obj, dict):
                                if "invert_texture_x" in surface_obj:
                                    invert_x = bool(surface_obj["invert_texture_x"])
                                if "invert_texture_y" in surface_obj:
                                    invert_y = bool(surface_obj["invert_texture_y"])
                                if "swap_texture_axes" in surface_obj:
                                    swap_axes = bool(surface_obj["swap_texture_axes"])
                            
                            img_rgba_final = img_rgba_uv.copy()
                            if invert_x:
                                img_rgba_final = np.fliplr(img_rgba_final)
                                u_coords = 1.0 - u_coords
                            if invert_y:
                                img_rgba_final = np.flipud(img_rgba_final)
                                v_coords = 1.0 - v_coords
                            if swap_axes:
                                img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                                u_coords, v_coords = v_coords.copy(), u_coords.copy()
                            
                            t_coords = np.column_stack((u_coords.ravel(), v_coords.ravel()))
                            grid.point_data["TCoords"] = t_coords
                            img_rgba_uv = img_rgba_final
                        else:
                            # Fallback: Erstelle einfache Textur-Koordinaten
                            if t_coords is None:
                                ny_uv, nx_uv = U.shape
                                u_coords_fallback = np.linspace(0, 1, nx_uv)
                                v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)  # PyVista: v=0 oben
                                U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                                t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                            grid.point_data["TCoords"] = t_coords
                    else:
                        # Fallback wenn bounds nicht verfügbar
                        if t_coords is None:
                            ny_uv, nx_uv = U.shape
                            u_coords_fallback = np.linspace(0, 1, nx_uv)
                            v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)
                            U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                            t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                        grid.point_data["TCoords"] = t_coords
                    
                    # Erstelle Textur
                    tex = pv.Texture(img_rgba_uv)
                    tex.interpolate = False
                    
                    actor_name = f"{self.SURFACE_NAME}_tex_{surface_id}"
                    old_texture_data = self._surface_texture_actors.get(surface_id)
                    if old_texture_data is not None:
                        old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                        if old_actor is not None:
                            try:
                                self.plotter.remove_actor(old_actor)
                            except Exception:
                                pass
                    
                    actor = self.plotter.add_mesh(
                        grid,
                        name=actor_name,
                        texture=tex,
                        show_scalar_bar=False,
                        reset_camera=False,
                    )
                    
                    # Speichere Metadaten
                    metadata = {
                        'actor': actor,
                        'grid': grid,
                        'texture': tex,
                        'grid_bounds': tuple(bounds) if bounds is not None else None,
                        'orientation': orientation,
                        'wall_axis': wall_axis,
                        'wall_value': wall_value,
                        'surface_id': surface_id,
                        'is_vertical': True,
                    }
                    self._surface_texture_actors[surface_id] = metadata
                    
                    if DEBUG_SURFACE_GEOMETRY:
                        t_ms = (time.perf_counter() - t_start) * 1000.0
                        valid_count = int(alpha_mask_uv.sum())
                        print(
                            f"[SurfaceGeometry] Texture-Surface '{surface_id}' (senkrecht, {orientation}): "
                            f"tex_res={tex_res:.3f}m, size={vs.size}x{us.size}, "
                            f"pixels_in_polygon={valid_count}, render={t_ms:.2f}ms"
                        )
                    continue  # Überspringe normalen planaren Pfad
                
                # 🎯 NORMALER PFAD FÜR PLANARE FLÄCHEN (horizontal oder geneigt)
                # 🎯 FIX: Reduzierter Margin für schärfere Kanten
                # Reduziere Margin von 2.0 auf 0.5 für weniger Punkte außerhalb des Polygons
                # Dies reduziert die "Zacken" am Rand, da weniger Zellen außerhalb gerendert werden
                margin = tex_res * 0.5  # Nur halber Pixel-Abstand als Margin (vorher: 2.0)
                
                # Erstelle Grid mit reduziertem Margin (für minimale Interpolation an Rändern)
                x_start = xmin - margin
                x_end = xmax + margin
                y_start = ymin - margin
                y_end = ymax + margin
                
                # Berechne Anzahl Punkte (inklusive Endpunkte)
                num_x = int(np.ceil((x_end - x_start) / tex_res)) + 1
                num_y = int(np.ceil((y_end - y_start) / tex_res)) + 1
                
                # Erstelle Grid mit reduziertem Margin
                xs = np.linspace(x_start, x_end, num_x, dtype=float)
                ys = np.linspace(y_start, y_end, num_y, dtype=float)
                if xs.size < 2 or ys.size < 2:
                    continue

                # Erstelle meshgrid mit indexing="xy" für korrekte Zuordnung
                # Bei indexing="xy": X[j, i] = xs[i], Y[j, i] = ys[j]
                # Shape ist (len(ys), len(xs)) = (ny, nx)
                X, Y = np.meshgrid(xs, ys, indexing="xy")
                points_2d = np.column_stack((X.ravel(), Y.ravel()))

                # Maske im Polygon
                poly_path = Path(np.column_stack((poly_x, poly_y)))
                inside = poly_path.contains_points(points_2d)
                inside = inside.reshape(X.shape)

                if not np.any(inside):
                    if DEBUG_SURFACE_GEOMETRY:
                        print(
                            f"[SurfaceGeometry] Texture-Surface '{surface_id}': "
                            f"keine Pixel im Polygon gefunden"
                        )
                    continue

                # SPL bilinear vom coarse Grid interpolieren
                spl_flat = self._bilinear_interpolate_grid(
                    source_x,
                    source_y,
                    values,
                    X.ravel(),
                    Y.ravel(),
                )
                spl_img = spl_flat.reshape(X.shape)

                # Werte clippen
                spl_clipped = np.clip(spl_img, cbar_min, cbar_max)
                # Optional in Stufen quantisieren (Color step)
                if is_step_mode:
                    spl_clipped = self._quantize_to_steps(spl_clipped, cbar_step)

                # In Farbe umsetzen
                rgba = base_cmap(norm(spl_clipped))  # float [0,1], shape (H,W,4)

                # 🎯 Verbesserte Alpha-Maske: Schärfere Kanten durch strikte Polygon-Prüfung
                # Nur Pixel, die definitiv im Polygon liegen, sind sichtbar
                alpha_mask = inside & np.isfinite(spl_clipped)
                rgba[..., 3] = np.where(alpha_mask, 1.0, 0.0)

                # Nach uint8 wandeln
                img_rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

                # 🎯 Berechne Z-Koordinaten basierend auf Planmodell für geneigte Flächen
                mode = plane_model.get("mode", "constant")
                if mode == "constant":
                    # Konstante Höhe
                    Z = np.full_like(X, float(plane_model.get("base", 0.0)))
                elif mode == "x":
                    # Lineare Steigung entlang X-Achse: Z = slope * X + intercept
                    slope = float(plane_model.get("slope", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z = slope * X + intercept
                elif mode == "y":
                    # Lineare Steigung entlang Y-Achse: Z = slope * Y + intercept
                    slope = float(plane_model.get("slope", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z = slope * Y + intercept
                else:  # mode == "xy" (allgemeine Ebene)
                    # Allgemeine Ebene: Z = slope_x * X + slope_y * Y + intercept
                    slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                    slope_y = float(plane_model.get("slope_y", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z = slope_x * X + slope_y * Y + intercept
                
                # Erstelle Grid mit korrekten Z-Koordinaten (für horizontale und geneigte Flächen)
                grid = pv.StructuredGrid(X, Y, Z)

                # Verwende PyVista's texture_map_to_plane() für automatische Textur-Koordinaten
                # Dies stellt sicher, dass PyVista die Textur-Koordinaten erkennt
                try:
                    grid.texture_map_to_plane(inplace=True)
                    # Hole die generierten Textur-Koordinaten
                    t_coords = grid.point_data.get("TCoords")
                except Exception as tex_map_exc:
                    if DEBUG_SURFACE_GEOMETRY:
                        print(f"[SurfaceGeometry] texture_map_to_plane() fehlgeschlagen für '{surface_id}': {tex_map_exc}")
                    t_coords = None

                # Texture-Koordinaten manuell anpassen (basierend auf Welt-Koordinaten für korrekte Orientierung)
                bounds = None
                try:
                    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                    xmin_grid, xmax_grid = bounds[0], bounds[1]
                    ymin_grid, ymax_grid = bounds[2], bounds[3]
                    
                    # 🎯 KORREKTE Textur-Koordinaten-Berechnung:
                    # Das Bild img_rgba hat Shape (H, W, 4) wo:
                    #   H = len(ys) = Anzahl Y-Punkte (von ymin nach ymax)
                    #   W = len(xs) = Anzahl X-Punkte (von xmin nach xmax)
                    # 
                    # Im Bild-Array: img_rgba[j, i, :] entspricht:
                    #   i = Index in xs (0 = xmin, W-1 = xmax)
                    #   j = Index in ys (0 = ymin, H-1 = ymax)
                    #
                    # Textur-Koordinaten (u, v):
                    #   u = X-Position im Bild (0 = links/xmin, 1 = rechts/xmax)
                    #   v = Y-Position im Bild (0 = unten/ymin, 1 = oben/ymax)
                    # 
                    # WICHTIG: In Bild-Arrays ist j=0 oben und j=H-1 unten (invertiert zu v)
                    # Aber bei meshgrid mit indexing="xy": Y[j, i] = ys[j], X[j, i] = xs[i]
                    #   j=0 entspricht ymin (unten), j=H-1 entspricht ymax (oben)
                    #   i=0 entspricht xmin (links), i=W-1 entspricht xmax (rechts)
                    #
                    # Für Textur-Koordinaten müssen wir:
                    #   u = (X - xmin) / (xmax - xmin)  -> 0 (xmin/links) bis 1 (xmax/rechts)
                    #   v = (Y - ymin) / (ymax - ymin)  -> 0 (ymin/unten) bis 1 (ymax/oben)
                    #
                    # Das Bild wurde erstellt mit: spl_img[j, i] = spl_flat[j*W + i]
                    #   wobei X[j, i] = xs[i] und Y[j, i] = ys[j]
                    # Also: img_rgba[j, i] entspricht Welt-Koordinate (xs[i], ys[j])
                    #
                    # Textur-Koordinate für Punkt (X[j,i], Y[j,i]) sollte sein:
                    #   u = (i) / (W-1) = (xs[i] - xmin) / (xmax - xmin)  (bei gleichmäßiger Verteilung)
                    #   v = (j) / (H-1) = (ys[j] - ymin) / (ymax - ymin)  (bei gleichmäßiger Verteilung)
                    
                    # Berechne Textur-Koordinaten direkt aus Welt-Koordinaten
                    x_span = xmax_grid - xmin_grid
                    y_span = ymax_grid - ymin_grid
                    
                    if x_span > 1e-10 and y_span > 1e-10:
                        # Überschreibe die automatisch generierten Textur-Koordinaten mit manuell berechneten
                        # für exakte Zuordnung zwischen Welt-Koordinaten und Bild-Pixeln
                        # Normalisiere Welt-Koordinaten auf [0, 1]
                        # u: 0 = xmin (links), 1 = xmax (rechts)
                        # v: 0 = ymin (unten), 1 = ymax (oben)
                        u_coords = (X - xmin_grid) / x_span
                        v_coords_raw = (Y - ymin_grid) / y_span
                        
                        # Stelle sicher, dass u und v im Bereich [0, 1] sind (aufgrund numerischer Ungenauigkeiten)
                        u_coords = np.clip(u_coords, 0.0, 1.0)
                        v_coords_raw = np.clip(v_coords_raw, 0.0, 1.0)
                        
                        # 🎯 WICHTIG: Spiegle v-Koordinate für PyVista-Konvention
                        # PyVista erwartet: v=0 oben, v=1 unten (Standard-Textur-Koordinaten)
                        # Wir haben: v_raw=0 unten (ymin), v_raw=1 oben (ymax)
                        # Also: v = 1 - v_raw
                        v_coords = 1.0 - v_coords_raw
                        
                        # 🎯 ACHSEN-INVERTIERUNG: Konfigurierbar pro Surface
                        # Prüfe, ob X- und/oder Y-Achsen invertiert werden sollen
                        # Defaults (werden verwendet, wenn Attribute nicht gesetzt sind)
                        invert_x = False
                        invert_y = True
                        swap_axes = False 
                        
                        if isinstance(surface_obj, SurfaceDefinition):
                            # Nur überschreiben, wenn Attribut existiert
                            if hasattr(surface_obj, "invert_texture_x"):
                                invert_x = bool(getattr(surface_obj, "invert_texture_x"))
                            if hasattr(surface_obj, "invert_texture_y"):
                                invert_y = bool(getattr(surface_obj, "invert_texture_y"))
                            if hasattr(surface_obj, "swap_texture_axes"):
                                swap_axes = bool(getattr(surface_obj, "swap_texture_axes"))
                        elif isinstance(surface_obj, dict):
                            # Nur überschreiben, wenn Key existiert
                            if "invert_texture_x" in surface_obj:
                                invert_x = bool(surface_obj["invert_texture_x"])
                            if "invert_texture_y" in surface_obj:
                                invert_y = bool(surface_obj["invert_texture_y"])
                            if "swap_texture_axes" in surface_obj:
                                swap_axes = bool(surface_obj["swap_texture_axes"])
                        
                        # Debug-Ausgabe (immer anzeigen, damit wir sehen, was passiert)
                        print(
                            f"[Texture-Achsen] Surface '{surface_id}': "
                            f"invert_x={invert_x}, invert_y={invert_y}, swap_axes={swap_axes}"
                        )
                        
                        # 🎯 Wende Bild-Spiegelung an (wenn nötig), BEVOR wir die Textur erstellen
                        # Wenn wir das Bild spiegeln, müssen wir die Texturkoordinaten NICHT zusätzlich invertieren
                        img_rgba_final = img_rgba.copy()
                        if invert_x:
                            # Spiegle das Bild horizontal (links <-> rechts)
                            img_rgba_final = np.fliplr(img_rgba_final)
                            # Invertiere U-Koordinaten, damit sie zum gespiegelten Bild passen
                            u_coords = 1.0 - u_coords
                        if invert_y:
                            # Spiegle das Bild vertikal (oben <-> unten)
                            img_rgba_final = np.flipud(img_rgba_final)
                            # Invertiere V-Koordinaten, damit sie zum gespiegelten Bild passen
                            v_coords = 1.0 - v_coords
                        if swap_axes:
                            # Transponiere das Bild (X <-> Y)
                            img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                            # Vertausche U und V Koordinaten
                            u_coords, v_coords = v_coords.copy(), u_coords.copy()
                        
                        # Textur-Koordinaten als (N, 2) Array: [u, v]
                        # u = X-Koordinate, v = Y-Koordinate
                        t_coords = np.column_stack((u_coords.ravel(), v_coords.ravel()))
                        grid.point_data["TCoords"] = t_coords
                        
                        # Verwende das gespiegelte Bild
                        img_rgba = img_rgba_final
                    else:
                        # Fallback für degenerierte Fälle
                        if t_coords is None:
                            ny, nx = X.shape
                            u_coords = np.linspace(0, 1, nx)
                            v_coords_raw = np.linspace(0, 1, ny)
                            # Spiegle v für PyVista-Konvention (v=0 oben, v=1 unten)
                            v_coords = 1.0 - v_coords_raw
                            U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
                            t_coords = np.column_stack((U.ravel(), V.ravel()))
                        grid.point_data["TCoords"] = t_coords
                    
                    # ANALYSE: Koordinaten-Referenzierung
                    if DEBUG_SURFACE_GEOMETRY:
                        n_points = grid.n_points
                        ny_grid, nx_grid = Y.shape[0], X.shape[1]
                        
                        # Indizes für Ecken: (0,0), (0,nx-1), (ny-1,0), (ny-1,nx-1)
                        corner_indices = [
                            (0, 0, "links-unten"),
                            (0, nx_grid - 1, "rechts-unten"),
                            (ny_grid - 1, 0, "links-oben"),
                            (ny_grid - 1, nx_grid - 1, "rechts-oben"),
                        ]
                        
                        print(f"\n[Texture-Analyse] Surface '{surface_id}':")
                        print(f"  Grid-Bounds: x=[{bounds[0]:.3f}, {bounds[1]:.3f}], y=[{bounds[2]:.3f}, {bounds[3]:.3f}]")
                        print(f"  Grid-Shape: {ny_grid}x{nx_grid} (ny x nx)")
                        print(f"  Bild-Shape: {img_rgba.shape} (H x W x 4)")
                        print(f"  Polygon-Punkte: {len(poly_x)} Punkte")
                        print(f"    X-Bereich: [{xmin:.3f}, {xmax:.3f}]")
                        print(f"    Y-Bereich: [{ymin:.3f}, {ymax:.3f}]")
                        print(f"  Grid-X-Bereich: [{xs.min():.3f}, {xs.max():.3f}] ({xs.size} Punkte)")
                        print(f"  Grid-Y-Bereich: [{ys.min():.3f}, {ys.max():.3f}] ({ys.size} Punkte)")
                        
                        if t_coords is not None and t_coords.shape[0] == n_points:
                            print(f"  Textur-Koordinaten vorhanden: {t_coords.shape}")
                            print(f"  Ecken-Referenzierung:")
                            for j, i, name in corner_indices:
                                idx_1d = j * nx_grid + i
                                if idx_1d < n_points:
                                    world_x = X[j, i]
                                    world_y = Y[j, i]
                                    tex_u = t_coords[idx_1d, 0]
                                    tex_v = t_coords[idx_1d, 1]
                                    # Bild-Pixel-Position (vor Flip)
                                    img_i = i  # X-Index im Bild
                                    img_j = j  # Y-Index im Bild
                                    # Nach Flip: img_i_flipped = (xs.size - 1) - i
                                    img_i_flipped = (xs.size - 1) - i
                                    print(
                                        f"    {name:15s}: "
                                        f"Welt=({world_x:7.3f}, {world_y:7.3f}) | "
                                        f"Tex=(u={tex_u:.3f}, v={tex_v:.3f}) | "
                                        f"Bild-Pixel=({img_i:3d}, {img_j:3d}) | "
                                        f"Nach-Flip=({img_i_flipped:3d}, {img_j:3d})"
                                    )
                        else:
                            print(f"  ⚠️ Textur-Koordinaten fehlen oder haben falsche Shape")
                            print(f"     t_coords={t_coords is not None}, shape={t_coords.shape if t_coords is not None else None}, n_points={n_points}")
                        
                        # Erste und letzte Polygon-Punkte
                        print(f"  Polygon-Punkte (erste 3):")
                        for idx in range(min(3, len(poly_x))):
                            print(f"    P{idx}: ({poly_x[idx]:7.3f}, {poly_y[idx]:7.3f})")
                        if len(poly_x) > 3:
                            print(f"    ... ({len(poly_x) - 3} weitere Punkte)")
                            print(f"    P{len(poly_x)-1}: ({poly_x[-1]:7.3f}, {poly_y[-1]:7.3f})")
                        
                except Exception as tex_exc:
                    error_msg = f"[SurfaceGeometry] Texture-Surface '{surface_id}' FEHLER bei Textur-Koordinaten-Setzung: {tex_exc}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    continue

                # Prüfe ob bounds und t_coords gesetzt wurden
                if bounds is None:
                    bounds = grid.bounds
                if t_coords is None:
                    # Fallback: Hole TCoords vom Grid falls verfügbar
                    t_coords = grid.point_data.get("TCoords")
                
                # Validiere Textur-Koordinaten vor dem Rendern
                if "TCoords" not in grid.point_data:
                    error_msg = f"[SurfaceGeometry] Texture-Surface '{surface_id}': TCoords nicht im Grid gefunden!"
                    print(error_msg)
                    continue
                
                # Prüfe ob TCoords die richtige Shape haben
                tcoords_data = grid.point_data["TCoords"]
                if tcoords_data is None or tcoords_data.shape[0] != grid.n_points:
                    error_msg = f"[SurfaceGeometry] Texture-Surface '{surface_id}': TCoords haben falsche Shape! Erwartet: ({grid.n_points}, 2), erhalten: {tcoords_data.shape if tcoords_data is not None else None}"
                    print(error_msg)
                    continue

                tex = pv.Texture(img_rgba)
                # Interpolation deaktiviert für schärfere Darstellung
                tex.interpolate = False

                actor_name = f"{self.SURFACE_NAME}_tex_{surface_id}"
                # Alten Actor ggf. entfernen
                old_actor = None
                old_texture_data = self._surface_texture_actors.get(surface_id)
                if old_texture_data is not None:
                    old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                if old_actor is not None:
                    try:
                        self.plotter.remove_actor(old_actor)
                    except Exception:
                        pass

                actor = self.plotter.add_mesh(
                    grid,
                    name=actor_name,
                    texture=tex,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
                
                # 🎯 Speichere Metadaten zusammen mit dem Actor
                # Diese können für Click-Handling, Koordinaten-Transformation, etc. verwendet werden
                metadata = {
                    'actor': actor,
                    'grid': grid,
                    'texture': tex,
                    'grid_bounds': tuple(bounds) if bounds is not None else None,  # (xmin, xmax, ymin, ymax, zmin, zmax)
                    'world_coords_x': xs.copy(),  # 1D-Array der X-Koordinaten in Metern
                    'world_coords_y': ys.copy(),  # 1D-Array der Y-Koordinaten in Metern
                    'world_coords_grid_x': X.copy(),  # 2D-Meshgrid der X-Koordinaten
                    'world_coords_grid_y': Y.copy(),  # 2D-Meshgrid der Y-Koordinaten
                    'texture_resolution': tex_res,  # Auflösung der Textur in Metern
                    'texture_size': (ys.size, xs.size),  # (H, W) in Pixeln
                    'image_shape': img_rgba.shape,  # (H, W, 4)
                    'polygon_bounds': {
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                    },
                    'polygon_points': points,  # Original Polygon-Punkte
                    't_coords': t_coords.copy() if t_coords is not None else None,  # Textur-Koordinaten
                    'surface_id': surface_id,
                }
                
                self._surface_texture_actors[surface_id] = metadata

                if DEBUG_SURFACE_GEOMETRY:
                    t_ms = (time.perf_counter() - t_start) * 1000.0
                    valid_count = int(alpha_mask.sum())
                    print(
                        f"[SurfaceGeometry] Texture-Surface '{surface_id}': "
                        f"tex_res={tex_res:.3f}m, size={ys.size}x{xs.size}, "
                        f"pixels_in_polygon={valid_count}, render={t_ms:.2f}ms"
                    )
            except Exception as exc:
                error_msg = f"[SurfaceGeometry] Texture-Surface '{surface_id}' FEHLER: {exc}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                continue

    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except KeyError:
            pass

    def get_texture_metadata(self, surface_id: str) -> Optional[dict[str, Any]]:
        """
        Gibt die Metadaten einer Texture-Surface zurück.
        
        Args:
            surface_id: ID der Surface
            
        Returns:
            Dict mit Metadaten oder None, falls Surface nicht gefunden wurde.
            Metadaten enthalten:
            - 'actor': PyVista Actor
            - 'grid': StructuredGrid mit Weltkoordinaten
            - 'texture': PyVista Texture-Objekt
            - 'grid_bounds': (xmin, xmax, ymin, ymax, zmin, zmax)
            - 'world_coords_x': 1D-Array der X-Koordinaten in Metern
            - 'world_coords_y': 1D-Array der Y-Koordinaten in Metern
            - 'world_coords_grid_x': 2D-Meshgrid der X-Koordinaten
            - 'world_coords_grid_y': 2D-Meshgrid der Y-Koordinaten
            - 'texture_resolution': Auflösung in Metern
            - 'texture_size': (H, W) in Pixeln
            - 'image_shape': (H, W, 4) Shape des Bildes
            - 'polygon_bounds': Dict mit xmin, xmax, ymin, ymax
            - 'polygon_points': Original Polygon-Punkte
            - 't_coords': Textur-Koordinaten (n_points, 2)
            - 'surface_id': Surface-ID
        """
        texture_data = self._surface_texture_actors.get(surface_id)
        if texture_data is None:
            return None
        if isinstance(texture_data, dict) and 'metadata' in texture_data:
            return texture_data['metadata']
        if isinstance(texture_data, dict) and 'actor' in texture_data:
            # Neue Struktur: Dict mit Metadaten
            return texture_data
        # Alte Struktur: Nur Actor (für Rückwärtskompatibilität)
        return None

    def get_texture_world_coords(self, surface_id: str, texture_u: float, texture_v: float) -> Optional[tuple[float, float, float]]:
        """
        Konvertiert Textur-Koordinaten (u, v) zu Weltkoordinaten (x, y, z).
        
        Args:
            surface_id: ID der Surface
            texture_u: U-Koordinate [0, 1] (horizontal)
            texture_v: V-Koordinate [0, 1] (vertikal)
            
        Returns:
            (x, y, z) Weltkoordinaten in Metern oder None bei Fehler
        """
        metadata = self.get_texture_metadata(surface_id)
        if metadata is None:
            return None
        
        try:
            bounds = metadata.get('grid_bounds')
            if bounds is None or len(bounds) < 6:
                return None
            
            xmin, xmax = bounds[0], bounds[1]
            ymin, ymax = bounds[2], bounds[3]
            zmin, zmax = bounds[4], bounds[5]
            
            # Konvertiere normalisierte Textur-Koordinaten zu Weltkoordinaten
            x = xmin + texture_u * (xmax - xmin)
            y = ymin + texture_v * (ymax - ymin)
            z = zmin  # Für horizontale Surfaces ist Z konstant
            
            return (x, y, z)
        except Exception:
            return None

    def get_world_coords_to_texture_coords(self, surface_id: str, world_x: float, world_y: float) -> Optional[tuple[float, float]]:
        """
        Konvertiert Weltkoordinaten (x, y) zu Textur-Koordinaten (u, v).
        
        Args:
            surface_id: ID der Surface
            world_x: X-Koordinate in Metern
            world_y: Y-Koordinate in Metern
            
        Returns:
            (u, v) Textur-Koordinaten [0, 1] oder None bei Fehler
        """
        metadata = self.get_texture_metadata(surface_id)
        if metadata is None:
            return None
        
        try:
            bounds = metadata.get('grid_bounds')
            if bounds is None or len(bounds) < 6:
                return None
            
            xmin, xmax = bounds[0], bounds[1]
            ymin, ymax = bounds[2], bounds[3]
            
            # Konvertiere Weltkoordinaten zu normalisierten Textur-Koordinaten
            if xmax > xmin:
                u = (world_x - xmin) / (xmax - xmin)
            else:
                u = 0.0
            
            if ymax > ymin:
                v = (world_y - ymin) / (ymax - ymin)
            else:
                v = 0.0
            
            # Clippe auf [0, 1]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            
            return (u, v)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Debug-Funktionen für Surface-Referenzierung
    # ------------------------------------------------------------------
    def debug_positioning_comparison(self, surface_id: Optional[str] = None) -> None:
        """
        Vergleicht die Positionierung von Surface-Rändern (Overlays) und SPL-Plots.
        
        Analysiert die Koordinaten-Diskrepanz zwischen:
        - Surface-Ränder (Overlays): Direkte Polygon-Punkte
        - SPL-Plots (Textures): Grid-basierte Koordinaten mit Margin
        
        Args:
            surface_id: Optional - spezifische Surface-ID zum Debuggen.
                       Wenn None, werden alle Surfaces analysiert.
        """
        if not DEBUG_SURFACE_REF:
            return
        
        print("\n" + "=" * 80)
        print("🔍 DEBUG: Positionierungs-Vergleich (Overlays vs. SPL-Plots)")
        print("=" * 80)
        
        surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
        if not isinstance(surface_definitions, dict):
            print("❌ Keine Surface-Definitionen gefunden")
            return
        
        # Filtere nach surface_id falls angegeben
        surfaces_to_analyze = {}
        if surface_id is not None:
            if surface_id in surface_definitions:
                surfaces_to_analyze[surface_id] = surface_definitions[surface_id]
            else:
                print(f"❌ Surface '{surface_id}' nicht gefunden")
                return
        else:
            surfaces_to_analyze = surface_definitions
        
        print(f"\n📋 Analysiere {len(surfaces_to_analyze)} Surface(s)")
        
        # Hole Texture-Metadaten
        render_mode = getattr(self.settings, "spl_surface_render_mode", "mesh") or "mesh"
        is_texture_mode = render_mode.lower() == "texture"
        
        for surf_id, surf_def in surfaces_to_analyze.items():
            print("\n" + "-" * 80)
            print(f"🎯 SURFACE: {surf_id}")
            print("-" * 80)
            
            # 1. Surface-Definition (Quelle)
            if isinstance(surf_def, SurfaceDefinition):
                enabled = bool(getattr(surf_def, "enabled", False))
                hidden = bool(getattr(surf_def, "hidden", False))
                points = getattr(surf_def, "points", []) or []
            else:
                enabled = bool(surf_def.get("enabled", False))
                hidden = bool(surf_def.get("hidden", False))
                points = surf_def.get("points", [])
            
            if not points:
                print("  ⚠️  Keine Polygon-Punkte gefunden")
                continue
            
            # Extrahiere Koordinaten aus Surface-Definition
            poly_x = np.array([float(p.get('x', 0.0)) for p in points])
            poly_y = np.array([float(p.get('y', 0.0)) for p in points])
            poly_z = np.array([float(p.get('z', 0.0)) for p in points])
            
            # Overlay-Koordinaten (wie sie geplottet werden)
            overlay_x_min, overlay_x_max = float(poly_x.min()), float(poly_x.max())
            overlay_y_min, overlay_y_max = float(poly_y.min()), float(poly_y.max())
            overlay_z_min, overlay_z_max = float(poly_z.min()), float(poly_z.max())
            
            print(f"\n  📐 OVERLAY-KOORDINATEN (Surface-Ränder):")
            print(f"  {'─' * 78}")
            print(f"    Polygon-Bounds:")
            print(f"      X: [{overlay_x_min:.3f}, {overlay_x_max:.3f}] m")
            print(f"      Y: [{overlay_y_min:.3f}, {overlay_y_max:.3f}] m")
            print(f"      Z: [{overlay_z_min:.3f}, {overlay_z_max:.3f}] m")
            print(f"    Polygon-Punkte: {len(points)}")
            print(f"    Erster Punkt: ({poly_x[0]:.3f}, {poly_y[0]:.3f}, {poly_z[0]:.3f})")
            if len(points) > 1:
                print(f"    Letzter Punkt: ({poly_x[-1]:.3f}, {poly_y[-1]:.3f}, {poly_z[-1]:.3f})")
            
            # 2. SPL-Plot-Koordinaten (Texture-Mode)
            if is_texture_mode and enabled and not hidden:
                texture_metadata = self.get_texture_metadata(surf_id)
                if texture_metadata:
                    print(f"\n  📊 SPL-PLOT-KOORDINATEN (Texture-Grid):")
                    print(f"  {'─' * 78}")
                    
                    bounds = texture_metadata.get('grid_bounds')
                    if bounds is not None and len(bounds) >= 6:
                        spl_x_min, spl_x_max = bounds[0], bounds[1]
                        spl_y_min, spl_y_max = bounds[2], bounds[3]
                        spl_z_min, spl_z_max = bounds[4], bounds[5]
                        
                        print(f"    Grid-Bounds:")
                        print(f"      X: [{spl_x_min:.3f}, {spl_x_max:.3f}] m")
                        print(f"      Y: [{spl_y_min:.3f}, {spl_y_max:.3f}] m")
                        print(f"      Z: [{spl_z_min:.3f}, {spl_z_max:.3f}] m")
                        
                        # Berechne Diskrepanz
                        x_offset_min = spl_x_min - overlay_x_min
                        x_offset_max = spl_x_max - overlay_x_max
                        y_offset_min = spl_y_min - overlay_y_min
                        y_offset_max = spl_y_max - overlay_y_max
                        
                        print(f"\n    ⚠️  DISKREPANZ (Grid - Polygon):")
                        print(f"      X-Min Offset: {x_offset_min:+.3f} m")
                        print(f"      X-Max Offset: {x_offset_max:+.3f} m")
                        print(f"      Y-Min Offset: {y_offset_min:+.3f} m")
                        print(f"      Y-Max Offset: {y_offset_max:+.3f} m")
                        
                        # Erwarteter Margin (aus Code: margin = tex_res * 2.0)
                        tex_res = texture_metadata.get('texture_resolution')
                        if tex_res is not None:
                            expected_margin = tex_res * 2.0
                            print(f"\n    📏 Erwarteter Margin: {expected_margin:.3f} m (= 2 * tex_res)")
                            print(f"    Tatsächlicher Margin X-Min: {abs(x_offset_min):.3f} m")
                            print(f"    Tatsächlicher Margin X-Max: {abs(x_offset_max):.3f} m")
                            print(f"    Tatsächlicher Margin Y-Min: {abs(y_offset_min):.3f} m")
                            print(f"    Tatsächlicher Margin Y-Max: {abs(y_offset_max):.3f} m")
                            
                            # Prüfe ob Margin korrekt ist
                            margin_tolerance = 0.01  # 1cm Toleranz
                            if abs(abs(x_offset_min) - expected_margin) > margin_tolerance:
                                print(f"    ❌ FEHLER: X-Min Margin stimmt nicht! Erwartet: {expected_margin:.3f}, Tatsächlich: {abs(x_offset_min):.3f}")
                            if abs(abs(x_offset_max) - expected_margin) > margin_tolerance:
                                print(f"    ❌ FEHLER: X-Max Margin stimmt nicht! Erwartet: {expected_margin:.3f}, Tatsächlich: {abs(x_offset_max):.3f}")
                            if abs(abs(y_offset_min) - expected_margin) > margin_tolerance:
                                print(f"    ❌ FEHLER: Y-Min Margin stimmt nicht! Erwartet: {expected_margin:.3f}, Tatsächlich: {abs(y_offset_min):.3f}")
                            if abs(abs(y_offset_max) - expected_margin) > margin_tolerance:
                                print(f"    ❌ FEHLER: Y-Max Margin stimmt nicht! Erwartet: {expected_margin:.3f}, Tatsächlich: {abs(y_offset_max):.3f}")
                        
                        # Polygon-Bounds aus Metadaten
                        polygon_bounds = texture_metadata.get('polygon_bounds')
                        if polygon_bounds:
                            meta_x_min = polygon_bounds.get('xmin')
                            meta_y_min = polygon_bounds.get('ymin')
                            meta_x_max = polygon_bounds.get('xmax')
                            meta_y_max = polygon_bounds.get('ymax')
                            
                            print(f"\n    🔍 VALIDIERUNG (Metadaten vs. Overlay):")
                            print(f"      Polygon X: Overlay=[{overlay_x_min:.3f}, {overlay_x_max:.3f}], Metadata=[{meta_x_min:.3f}, {meta_x_max:.3f}]")
                            print(f"      Polygon Y: Overlay=[{overlay_y_min:.3f}, {overlay_y_max:.3f}], Metadata=[{meta_y_min:.3f}, {meta_y_max:.3f}]")
                            
                            # Prüfe Konsistenz
                            if meta_x_min is not None and abs(meta_x_min - overlay_x_min) > 0.001:
                                print(f"      ❌ FEHLER: X-Min stimmt nicht überein! Differenz: {abs(meta_x_min - overlay_x_min):.3f} m")
                            if meta_x_max is not None and abs(meta_x_max - overlay_x_max) > 0.001:
                                print(f"      ❌ FEHLER: X-Max stimmt nicht überein! Differenz: {abs(meta_x_max - overlay_x_max):.3f} m")
                            if meta_y_min is not None and abs(meta_y_min - overlay_y_min) > 0.001:
                                print(f"      ❌ FEHLER: Y-Min stimmt nicht überein! Differenz: {abs(meta_y_min - overlay_y_min):.3f} m")
                            if meta_y_max is not None and abs(meta_y_max - overlay_y_max) > 0.001:
                                print(f"      ❌ FEHLER: Y-Max stimmt nicht überein! Differenz: {abs(meta_y_max - overlay_y_max):.3f} m")
                        
                        # Erste und letzte Grid-Punkte
                        world_x = texture_metadata.get('world_coords_x')
                        world_y = texture_metadata.get('world_coords_y')
                        if world_x is not None and world_y is not None and len(world_x) > 0 and len(world_y) > 0:
                            print(f"\n    🔍 GRID-KOORDINATEN-DETAILS:")
                            print(f"      Grid X: [{world_x[0]:.3f}, ..., {world_x[-1]:.3f}] ({len(world_x)} Punkte)")
                            print(f"      Grid Y: [{world_y[0]:.3f}, ..., {world_y[-1]:.3f}] ({len(world_y)} Punkte)")
                            print(f"      Grid-Punkt (0,0): ({world_x[0]:.3f}, {world_y[0]:.3f})")
                            print(f"      Polygon-Punkt (0): ({poly_x[0]:.3f}, {poly_y[0]:.3f})")
                            print(f"      Differenz: ({world_x[0] - poly_x[0]:+.3f}, {world_y[0] - poly_y[0]:+.3f})")
                else:
                    print(f"\n  ❌ Keine Texture-Metadaten gefunden für Surface '{surf_id}'")
                    print(f"     (Surface ist möglicherweise disabled oder hidden)")
            else:
                print(f"\n  ⚠️  Texture-Modus nicht aktiv oder Surface disabled/hidden")
                print(f"     Render-Modus: {render_mode}, enabled={enabled}, hidden={hidden}")
        
        print("\n" + "=" * 80)
    
    def debug_surface_referencing(self, surface_id: Optional[str] = None) -> None:
        """
        Debug-Ausgabe für die Referenzierung von Surfaces im Plot.
        
        Analysiert und zeigt:
        - Surface-Actors (Mesh- und Texture-Modus)
        - Weltkoordinaten
        - Textur-Koordinaten (TCoords)
        - Mapping zwischen Koordinaten-Systemen
        
        Args:
            surface_id: Optional - spezifische Surface-ID zum Debuggen. 
                       Wenn None, werden alle Surfaces analysiert.
        """
        # Funktion kann auch manuell aufgerufen werden (ohne Environment-Variable)
        # Für automatischen Aufruf: Setze LFO_DEBUG_SURFACE_REF=1
        
        print("\n" + "=" * 80)
        print("🔍 DEBUG: Surface-Referenzierung im 3D-Plot")
        print("=" * 80)
        
        render_mode = getattr(self.settings, "spl_surface_render_mode", "mesh") or "mesh"
        print(f"\n📊 Render-Modus: {render_mode.upper()}")
        
        surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
        if not isinstance(surface_definitions, dict):
            print("❌ Keine Surface-Definitionen gefunden")
            return
        
        # Filtere nach surface_id falls angegeben
        surfaces_to_analyze = {}
        if surface_id is not None:
            if surface_id in surface_definitions:
                surfaces_to_analyze[surface_id] = surface_definitions[surface_id]
            else:
                print(f"❌ Surface '{surface_id}' nicht gefunden")
                return
        else:
            surfaces_to_analyze = surface_definitions
        
        print(f"\n📋 Analysiere {len(surfaces_to_analyze)} Surface(s)")
        
        for surf_id, surf_def in surfaces_to_analyze.items():
            print("\n" + "-" * 80)
            print(f"🎯 SURFACE: {surf_id}")
            print("-" * 80)
            
            # Surface-Definition
            if isinstance(surf_def, SurfaceDefinition):
                enabled = bool(getattr(surf_def, "enabled", False))
                hidden = bool(getattr(surf_def, "hidden", False))
                points = getattr(surf_def, "points", []) or []
            else:
                enabled = bool(surf_def.get("enabled", False))
                hidden = bool(surf_def.get("hidden", False))
                points = surf_def.get("points", [])
            
            print(f"  Status: enabled={enabled}, hidden={hidden}")
            print(f"  Polygon-Punkte: {len(points)}")
            
            if points:
                poly_x = [float(p.get('x', 0.0)) for p in points]
                poly_y = [float(p.get('y', 0.0)) for p in points]
                poly_z = [float(p.get('z', 0.0)) for p in points]
                
                print(f"  Polygon-Bounds:")
                print(f"    X: [{min(poly_x):.3f}, {max(poly_x):.3f}] m")
                print(f"    Y: [{min(poly_y):.3f}, {max(poly_y):.3f}] m")
                print(f"    Z: [{min(poly_z):.3f}, {max(poly_z):.3f}] m")
                
                if len(points) <= 6:
                    print(f"  Polygon-Punkte (alle):")
                    for i, p in enumerate(points):
                        print(f"    P{i}: ({p.get('x', 0.0):.3f}, {p.get('y', 0.0):.3f}, {p.get('z', 0.0):.3f})")
                else:
                    print(f"  Polygon-Punkte (erste 3):")
                    for i, p in enumerate(points[:3]):
                        print(f"    P{i}: ({p.get('x', 0.0):.3f}, {p.get('y', 0.0):.3f}, {p.get('z', 0.0):.3f})")
                    print(f"    ... ({len(points) - 3} weitere Punkte)")
            
            # Analyse basierend auf Render-Modus
            if render_mode.lower() == "texture":
                self._debug_texture_surface_ref(surf_id, enabled, hidden)
            else:
                self._debug_mesh_surface_ref(surf_id, enabled, hidden)
        
        print("\n" + "=" * 80)
    
    def _debug_texture_surface_ref(self, surface_id: str, enabled: bool, hidden: bool) -> None:
        """Debug-Ausgabe für Texture-Surface."""
        if not enabled or hidden:
            print(f"  ⚠️  Surface ist disabled oder hidden - keine Actor-Daten verfügbar")
            return
        
        texture_data = self._surface_texture_actors.get(surface_id)
        if texture_data is None:
            print(f"  ❌ Keine Texture-Actor-Daten gefunden")
            return
        
        # Hole Metadaten (neue Struktur) oder Actor (alte Struktur)
        if isinstance(texture_data, dict) and 'actor' in texture_data:
            metadata = texture_data
            actor = metadata.get('actor')
        else:
            actor = texture_data
            metadata = None
        
        print(f"\n  📐 TEXTURE-MODUS")
        print(f"  {'─' * 78}")
        
        if metadata:
            print(f"\n  ✅ Metadaten verfügbar")
            
            bounds = metadata.get('grid_bounds')
            if bounds is not None and len(bounds) >= 6:
                print(f"  Grid-Bounds (Weltkoordinaten):")
                print(f"    X: [{bounds[0]:.3f}, {bounds[1]:.3f}] m")
                print(f"    Y: [{bounds[2]:.3f}, {bounds[3]:.3f}] m")
                print(f"    Z: [{bounds[4]:.3f}, {bounds[5]:.3f}] m")
                print(f"    Breite: {bounds[1] - bounds[0]:.3f} m")
                print(f"    Höhe: {bounds[3] - bounds[2]:.3f} m")
            
            world_x = metadata.get('world_coords_x')
            world_y = metadata.get('world_coords_y')
            if world_x is not None and world_y is not None:
                print(f"  Weltkoordinaten-Arrays:")
                print(f"    X: {len(world_x)} Punkte, [{world_x[0]:.3f}, ..., {world_x[-1]:.3f}] m")
                print(f"    Y: {len(world_y)} Punkte, [{world_y[0]:.3f}, ..., {world_y[-1]:.3f}] m")
            
            grid_x = metadata.get('world_coords_grid_x')
            grid_y = metadata.get('world_coords_grid_y')
            if grid_x is not None and grid_y is not None:
                print(f"  Grid-Meshgrid:")
                print(f"    Shape X: {grid_x.shape}")
                print(f"    Shape Y: {grid_y.shape}")
                print(f"    Beispiel-Punkt (0,0): x={grid_x[0,0]:.3f}m, y={grid_y[0,0]:.3f}m")
                if grid_x.shape[0] > 1 and grid_x.shape[1] > 1:
                    print(f"    Beispiel-Punkt (letztes): x={grid_x[-1,-1]:.3f}m, y={grid_y[-1,-1]:.3f}m")
            
            tex_res = metadata.get('texture_resolution')
            if tex_res is not None:
                print(f"  Texture-Auflösung: {tex_res:.4f} m")
            
            tex_size = metadata.get('texture_size')
            if tex_size is not None:
                print(f"  Texture-Größe: {tex_size[0]}x{tex_size[1]} Pixel")
            
            img_shape = metadata.get('image_shape')
            if img_shape is not None:
                print(f"  Bild-Shape: {img_shape}")
            
            t_coords = metadata.get('t_coords')
            if t_coords is not None:
                print(f"  Textur-Koordinaten (TCoords):")
                print(f"    Shape: {t_coords.shape}")
                print(f"    U-Bereich: [{np.min(t_coords[:, 0]):.3f}, {np.max(t_coords[:, 0]):.3f}]")
                print(f"    V-Bereich: [{np.min(t_coords[:, 1]):.3f}, {np.max(t_coords[:, 1]):.3f}]")
                
                # Beispiel-Mapping
                if t_coords.shape[0] > 0:
                    idx_example = 0
                    u_example = t_coords[idx_example, 0]
                    v_example = t_coords[idx_example, 1]
                    print(f"    Beispiel TCoords[0]: u={u_example:.3f}, v={v_example:.3f}")
                    
                    # Konvertiere zu Weltkoordinaten
                    world_coords = self.get_texture_world_coords(surface_id, u_example, v_example)
                    if world_coords:
                        print(f"      → Weltkoordinaten: x={world_coords[0]:.3f}m, y={world_coords[1]:.3f}m, z={world_coords[2]:.3f}m")
                    
                    # Rück-Konvertierung
                    if grid_x is not None and grid_y is not None:
                        x_actual = grid_x.ravel()[idx_example] if grid_x.size > idx_example else None
                        y_actual = grid_y.ravel()[idx_example] if grid_y.size > idx_example else None
                        if x_actual is not None and y_actual is not None:
                            tex_back = self.get_world_coords_to_texture_coords(surface_id, float(x_actual), float(y_actual))
                            if tex_back:
                                print(f"      → Rück-Konvertierung: u={tex_back[0]:.3f}, v={tex_back[1]:.3f}")
            
            polygon_bounds = metadata.get('polygon_bounds')
            if polygon_bounds:
                print(f"  Polygon-Bounds (aus Metadaten):")
                print(f"    X: [{polygon_bounds.get('xmin', 0):.3f}, {polygon_bounds.get('xmax', 0):.3f}] m")
                print(f"    Y: [{polygon_bounds.get('ymin', 0):.3f}, {polygon_bounds.get('ymax', 0):.3f}] m")
            
            # Prüfe Grid im Actor
            if actor is not None:
                try:
                    grid = metadata.get('grid')
                    if grid is not None and hasattr(grid, 'points'):
                        grid_points = grid.points
                        if grid_points is not None and len(grid_points) > 0:
                            print(f"  Grid im Actor:")
                            print(f"    Anzahl Punkte: {len(grid_points)}")
                            print(f"    Erster Punkt: ({grid_points[0,0]:.3f}, {grid_points[0,1]:.3f}, {grid_points[0,2]:.3f}) m")
                            print(f"    Letzter Punkt: ({grid_points[-1,0]:.3f}, {grid_points[-1,1]:.3f}, {grid_points[-1,2]:.3f}) m")
                            
                            # Vergleiche mit Metadaten
                            if bounds is not None:
                                grid_bounds_actual = grid.bounds
                                print(f"    Grid-Bounds (vom Grid):")
                                print(f"      X: [{grid_bounds_actual[0]:.3f}, {grid_bounds_actual[1]:.3f}] m")
                                print(f"      Y: [{grid_bounds_actual[2]:.3f}, {grid_bounds_actual[3]:.3f}] m")
                                
                                # Validierung
                                if abs(bounds[0] - grid_bounds_actual[0]) > 0.001:
                                    print(f"    ⚠️  WARNUNG: X-Min stimmt nicht überein!")
                                if abs(bounds[1] - grid_bounds_actual[1]) > 0.001:
                                    print(f"    ⚠️  WARNUNG: X-Max stimmt nicht überein!")
                                
                            # Prüfe TCoords im Grid
                            if hasattr(grid, 'point_data'):
                                grid_tcoords = grid.point_data.get('TCoords')
                                if grid_tcoords is not None:
                                    print(f"    TCoords im Grid:")
                                    print(f"      Shape: {grid_tcoords.shape}")
                                    print(f"      Erste TCoords: u={grid_tcoords[0,0]:.3f}, v={grid_tcoords[0,1]:.3f}")
                except Exception as e:
                    print(f"  ⚠️  Fehler beim Lesen des Grids: {e}")
        
        else:
            print(f"  ⚠️  Alte Struktur (nur Actor, keine Metadaten)")
            if actor is not None:
                print(f"  Actor gefunden: {type(actor).__name__}")
                try:
                    if hasattr(actor, 'mapper') and actor.mapper is not None:
                        mesh = actor.mapper.GetInput()
                        if mesh is not None and hasattr(mesh, 'points'):
                            points = mesh.points
                            if points is not None and len(points) > 0:
                                print(f"    Mesh-Punkte: {len(points)}")
                                print(f"    Erster Punkt: ({points[0,0]:.3f}, {points[0,1]:.3f}, {points[0,2]:.3f}) m")
                except Exception as e:
                    print(f"    ⚠️  Fehler beim Lesen des Actors: {e}")
    
    def _debug_mesh_surface_ref(self, surface_id: str, enabled: bool, hidden: bool) -> None:
        """Debug-Ausgabe für Mesh-Surface."""
        if not enabled or hidden:
            print(f"  ⚠️  Surface ist disabled oder hidden - keine Actor-Daten verfügbar")
            return
        
        actor = self._surface_actors.get(surface_id)
        if actor is None:
            # Prüfe ob es im kombinierten Mesh ist
            base_actor = self.plotter.renderer.actors.get(self.SURFACE_NAME)
            if base_actor is not None:
                print(f"\n  📐 MESH-MODUS (kombiniertes Mesh)")
                print(f"  {'─' * 78}")
                print(f"  ✅ Surface ist Teil des kombinierten Mesh-Actors '{self.SURFACE_NAME}'")
                try:
                    if hasattr(base_actor, 'mapper') and base_actor.mapper is not None:
                        mesh = base_actor.mapper.GetInput()
                        if mesh is not None and hasattr(mesh, 'points'):
                            points = mesh.points
                            if points is not None and len(points) > 0:
                                print(f"    Mesh-Punkte: {len(points)}")
                                print(f"    Erster Punkt: ({points[0,0]:.3f}, {points[0,1]:.3f}, {points[0,2]:.3f}) m")
                                print(f"    Letzter Punkt: ({points[-1,0]:.3f}, {points[-1,1]:.3f}, {points[-1,2]:.3f}) m")
                                
                                bounds = mesh.bounds if hasattr(mesh, 'bounds') else None
                                if bounds is not None:
                                    print(f"    Mesh-Bounds:")
                                    print(f"      X: [{bounds[0]:.3f}, {bounds[1]:.3f}] m")
                                    print(f"      Y: [{bounds[2]:.3f}, {bounds[3]:.3f}] m")
                                    print(f"      Z: [{bounds[4]:.3f}, {bounds[5]:.3f}] m")
                except Exception as e:
                    print(f"    ⚠️  Fehler beim Lesen des Mesh-Actors: {e}")
            else:
                print(f"  ❌ Keine Mesh-Actor-Daten gefunden")
            return
        
        print(f"\n  📐 MESH-MODUS (individueller Actor)")
        print(f"  {'─' * 78}")
        print(f"  ✅ Individueller Actor gefunden")
        
        try:
            if hasattr(actor, 'mapper') and actor.mapper is not None:
                mesh = actor.mapper.GetInput()
                if mesh is not None and hasattr(mesh, 'points'):
                    points = mesh.points
                    if points is not None and len(points) > 0:
                        print(f"    Mesh-Punkte: {len(points)}")
                        print(f"    Mesh-Zellen: {mesh.n_cells if hasattr(mesh, 'n_cells') else 'unbekannt'}")
                        print(f"    Erster Punkt: ({points[0,0]:.3f}, {points[0,1]:.3f}, {points[0,2]:.3f}) m")
                        if len(points) > 1:
                            print(f"    Letzter Punkt: ({points[-1,0]:.3f}, {points[-1,1]:.3f}, {points[-1,2]:.3f}) m")
                        
                        bounds = mesh.bounds if hasattr(mesh, 'bounds') else None
                        if bounds is not None:
                            print(f"    Mesh-Bounds (Weltkoordinaten):")
                            print(f"      X: [{bounds[0]:.3f}, {bounds[1]:.3f}] m")
                            print(f"      Y: [{bounds[2]:.3f}, {bounds[3]:.3f}] m")
                            print(f"      Z: [{bounds[4]:.3f}, {bounds[5]:.3f}] m")
                            print(f"      Breite: {bounds[1] - bounds[0]:.3f} m")
                            print(f"      Höhe: {bounds[3] - bounds[2]:.3f} m")
                        
                        # Prüfe Scalars
                        if hasattr(mesh, 'point_data'):
                            scalars = mesh.point_data.get('plot_scalars')
                            if scalars is not None:
                                print(f"    Scalars (SPL-Werte):")
                                print(f"      Shape: {scalars.shape}")
                                print(f"      Min: {np.nanmin(scalars):.2f}")
                                print(f"      Max: {np.nanmax(scalars):.2f}")
                                print(f"      Mean: {np.nanmean(scalars):.2f}")
        except Exception as e:
            print(f"    ⚠️  Fehler beim Lesen des Mesh-Actors: {e}")
            import traceback
            traceback.print_exc()

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

    def _compute_overlay_signatures(self, settings, container) -> dict[str, tuple]:
        """Erzeugt robuste Vergleichs-Signaturen für jede Overlay-Kategorie.

        Hintergrund:
        - Ziel ist, Änderungsdetektion ohne tiefe Objektvergleiche.
        - Wir wandeln relevante Settings in flache Tupel um, die sich leicht
          vergleichen lassen und keine PyVista-Objekte enthalten.
        """
        if DEBUG_OVERLAY_PERF:
            t_start = time.perf_counter()
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

        def _to_str_tuple(sequence) -> tuple:
            if sequence is None:
                return tuple()
            if isinstance(sequence, (str, bytes)):
                return (sequence.decode('utf-8', errors='ignore') if isinstance(sequence, bytes) else str(sequence),)
            try:
                iterable = list(sequence)
            except TypeError:
                return (str(sequence),)
            result: List[str] = []
            for item in iterable:
                if isinstance(item, bytes):
                    result.append(item.decode('utf-8', errors='ignore'))
                else:
                    result.append(str(item))
            return tuple(result)

        axis_signature = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
        )

        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        speakers_signature: List[tuple] = []
        if isinstance(speaker_arrays, dict):
            for name in sorted(speaker_arrays.keys()):
                array = speaker_arrays[name]
                configuration = str(getattr(array, 'configuration', '') or '').lower()
                hide = bool(getattr(array, 'hide', False))
                
                # Hole Array-Positionen
                array_pos_x = getattr(array, 'array_position_x', 0.0)
                array_pos_y = getattr(array, 'array_position_y', 0.0)
                array_pos_z = getattr(array, 'array_position_z', 0.0)
                
                # Für die Signatur verwenden wir die Plot-Positionen:
                # Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
                # Flown: Absolute Positionen (bereits in source_position_x/y/z_flown enthalten)
                if configuration == 'flown':
                    # Für Flown: Array-Positionen sind bereits in source_position_x/y/z_flown enthalten
                    xs = _to_tuple(getattr(array, 'source_position_x', None))
                    ys = _to_tuple(
                        getattr(
                            array,
                            'source_position_calc_y',
                            getattr(array, 'source_position_y', None),
                        )
                    )
                    zs_stack = None
                    zs_flown = _to_tuple(getattr(array, 'source_position_z_flown', None))
                else:
                    # Für Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
                    xs_raw = _to_tuple(getattr(array, 'source_position_x', None))
                    ys_raw = _to_tuple(getattr(array, 'source_position_y', None))
                    zs_raw = _to_tuple(getattr(array, 'source_position_z_stack', None))
                    
                    # Addiere Array-Positionen zu den Speaker-Positionen
                    if xs_raw and array_pos_x != 0.0:
                        xs = tuple(x + array_pos_x for x in xs_raw)
                    else:
                        xs = xs_raw
                    if ys_raw and array_pos_y != 0.0:
                        ys = tuple(y + array_pos_y for y in ys_raw)
                    else:
                        ys = ys_raw
                    if zs_raw and array_pos_z != 0.0:
                        zs_stack = tuple(z + array_pos_z for z in zs_raw)
                    else:
                        zs_stack = zs_raw
                    zs_flown = None
                
                azimuth = _to_tuple(getattr(array, 'source_azimuth', None))
                angles = _to_tuple(getattr(array, 'source_angle', None))
                polar_pattern = _to_str_tuple(getattr(array, 'source_polar_pattern', None))
                source_types = _to_str_tuple(getattr(array, 'source_type', None))
                
                sources = (
                    tuple(xs) if xs else tuple(),
                    tuple(ys) if ys else tuple(),
                    tuple(zs_stack) if zs_stack is not None else tuple(),
                    tuple(zs_flown) if zs_flown is not None else tuple(),
                    tuple(azimuth) if azimuth else tuple(),
                    tuple(angles) if angles else tuple(),
                    (round(float(array_pos_x), 4), round(float(array_pos_y), 4), round(float(array_pos_z), 4)),
                )
                speakers_signature.append(
                    (
                        str(name),
                        configuration,
                        hide,
                        sources,
                        polar_pattern,
                        source_types,
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

        # Surfaces-Signatur (optimiert: nur geänderte Surfaces neu berechnen)
        surface_definitions = getattr(settings, 'surface_definitions', {})
        surfaces_signature: List[tuple] = []
        
        # Cache für Surface-Signaturen (nur bei wiederholten Aufrufen)
        if not hasattr(self, '_surface_signature_cache'):
            self._surface_signature_cache: dict[str, tuple] = {}
        
        if isinstance(surface_definitions, dict):
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
                
                # Erstelle schnelle Hash-Signatur für Vergleich
                # (nur enabled/hidden + Anzahl Punkte + Hash der Punkte)
                points_hash = hash(tuple(
                    (float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0)))
                    for p in points[:10]  # Nur erste 10 Punkte für Hash (Performance)
                )) if points else 0
                
                # Prüfe ob Surface sich geändert hat
                cache_key = f"{surface_id}_{enabled}_{hidden}_{len(points)}_{points_hash}"
                if cache_key in self._surface_signature_cache:
                    # Verwende gecachte Signatur
                    surfaces_signature.append(self._surface_signature_cache[cache_key])
                    continue
                
                # Neu berechnen nur wenn nötig
                points_tuple = []
                for point in points:
                    try:
                        x = float(point.get('x', 0.0))
                        y = float(point.get('y', 0.0))
                        z = float(point.get('z', 0.0))
                        points_tuple.append((x, y, z))
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                sig = (
                    str(surface_id),
                    enabled,
                    hidden,
                    tuple(points_tuple)
                )
                surfaces_signature.append(sig)
                # Cache speichern
                self._surface_signature_cache[cache_key] = sig
        
        surfaces_signature_tuple = tuple(surfaces_signature)
        
        # 🎯 WICHTIG: has_speaker_arrays wird NICHT in die Signatur aufgenommen,
        # da Surfaces unabhängig von der Anwesenheit von Sources gezeichnet werden sollen.
        # has_speaker_arrays beeinflusst nur die Darstellung (gestrichelt vs. durchgezogen),
        # nicht ob Surfaces gezeichnet werden.
        # Die Signatur besteht aus den Surface-Definitionen, active_surface_id und active_surface_highlight_ids.
        active_surface_id = getattr(settings, 'active_surface_id', None)
        highlight_ids = getattr(settings, 'active_surface_highlight_ids', None)
        if isinstance(highlight_ids, (list, tuple, set)):
            highlight_ids_tuple = tuple(sorted(str(sid) for sid in highlight_ids))
        else:
            highlight_ids_tuple = tuple()
        
        surfaces_signature_with_active = (surfaces_signature_tuple, active_surface_id, highlight_ids_tuple)
        
        result = {
            'axis': axis_signature,
            'speakers': speakers_signature_tuple,
            'impulse': impulse_signature_tuple,
            'surfaces': surfaces_signature_with_active,  # Enthält active_surface_id und has_speaker_arrays
        }
        
        
        return result




__all__ = ['DrawSPLPlot3D']


