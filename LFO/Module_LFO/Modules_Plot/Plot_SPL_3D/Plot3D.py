"""PyVista-basierter SPL-Plot (3D)"""

from __future__ import annotations

import hashlib
import math
import os
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Plot.Plot_SPL_3D.PlotSPL3DOverlays import SPL3DOverlayRenderer
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DSPL import SPL3DPlotRenderer, SPLTimeControlBar
from Module_LFO.Modules_Plot.Plot_SPL_3D.ColorbarManager import ColorbarManager, PHASE_CMAP
from Module_LFO.Modules_Init.Logging import measure_time, perf_section
from Module_LFO.Modules_Calculate\
    .SurfaceGeometryCalculator import (
    SurfaceDefinition,
    build_surface_mesh,
    prepare_plot_geometry,
    prepare_vertical_plot_geometry,
    build_full_floor_mesh,
    clip_floor_with_surfaces,
    derive_surface_plane,
)


DEBUG_PLOT3D_TIMING = bool(int(os.environ.get("LFO_DEBUG_PLOT3D_TIMING", "1")))

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


class DrawSPLPlot3D(SPL3DPlotRenderer, ModuleBase, QtCore.QObject):
    """Erstellt einen interaktiven 3D-SPL-Plot auf Basis von PyVista."""
    # PHASE_CMAP wird jetzt aus ColorbarManager importiert
    PHASE_CMAP = PHASE_CMAP

    SURFACE_NAME = "spl_surface"
    FLOOR_NAME = "spl_floor"
    UPSCALE_FACTOR = 1  # Erhöht die Anzahl der Grafikpunkte für schärferen Plot (mit Interpolation)

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
        self.main_window = None  # Wird bei Bedarf gesetzt

        self.plotter = QtInteractor(parent_widget)
        self.widget = self.plotter.interactor
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)
        
        # 🎯 DEAKTIVIERE VTK's Standard-Rotation komplett (inkl. Doppelklick)
        # VTK's TrackballCamera führt bei Linksklick-Drag Rotation aus
        # Wir fangen alle Events selbst ab, daher deaktivieren wir VTK's Standard-Handler komplett
        try:
            if hasattr(self.plotter, 'iren') and self.plotter.iren is not None:
                interactor_style = self.plotter.iren.GetInteractorStyle()
                if interactor_style is not None:
                    # Deaktiviere ALLE Linksklick-Events im InteractorStyle
                    # VTK's InteractorStyleTrackballCamera hat:
                    # - OnLeftButtonPress() - startet Rotation
                    # - OnLeftButtonRelease() - beendet Rotation
                    # - OnLeftButtonDoubleClick() - Reset Camera
                    # - OnMouseMove() - führt Rotation aus
                    
                    # Leere Handler-Funktionen
                    def empty_handler(*args, **kwargs):
                        pass
                    
                    # Überschreibe alle Linksklick-Handler
                    handlers_to_disable = [
                        'OnLeftButtonPress',
                        'OnLeftButtonRelease', 
                        'OnLeftButtonDoubleClick',
                    ]
                    
                    for handler_name in handlers_to_disable:
                        if hasattr(interactor_style, handler_name):
                            try:
                                # Versuche, die Methode zu überschreiben
                                # Verwende types.MethodType für bessere Kompatibilität
                                setattr(interactor_style, handler_name, types.MethodType(lambda self, *args, **kwargs: None, interactor_style))
                            except Exception:
                                try:
                                    # Fallback: Direkte Zuweisung
                                    setattr(interactor_style, handler_name, empty_handler)
                                except Exception:
                                    pass
                    
                    # ZUSÄTZLICH: Deaktiviere auch alle anderen Doppelklick-Handler
                    all_double_click_handlers = [
                        'OnRightButtonDoubleClick',
                        'OnMiddleButtonDoubleClick',
                    ]
                    for handler_name in all_double_click_handlers:
                        if hasattr(interactor_style, handler_name):
                            try:
                                setattr(interactor_style, handler_name, types.MethodType(lambda self, *args, **kwargs: None, interactor_style))
                            except Exception:
                                try:
                                    setattr(interactor_style, handler_name, empty_handler)
                                except Exception:
                                    pass
                    
                    # ZUSÄTZLICH: Deaktiviere auch OnMouseMove für Linksklick-Rotation
                    # Wir fangen MouseMove-Events im eventFilter ab, daher können wir OnMouseMove komplett deaktivieren
                    if hasattr(interactor_style, 'OnMouseMove'):
                        try:
                            # Leere Funktion für OnMouseMove (wird im eventFilter behandelt)
                            def disabled_mousemove(style_self):
                                # Komplett deaktiviert - MouseMove wird im eventFilter behandelt
                                pass
                            setattr(interactor_style, 'OnMouseMove', types.MethodType(disabled_mousemove, interactor_style))
                        except Exception:
                            pass
        except Exception:
            pass
        self._pan_active = False
        self._pan_last_pos: Optional[QtCore.QPoint] = None
        self._rotate_active = False
        self._rotate_last_pos: Optional[QtCore.QPoint] = None
        self._click_start_pos: Optional[QtCore.QPoint] = None  # Für Click-Erkennung (vs. Drag)
        self._last_click_time: Optional[float] = None  # Zeitpunkt des letzten Releases (für einfache Klicks)
        self._last_click_pos: Optional[QtCore.QPoint] = None  # Position des letzten Releases
        self._last_press_time: Optional[float] = None  # Zeitpunkt des letzten Press (nicht mehr für Doppelklick verwendet)
        self._last_press_pos: Optional[QtCore.QPoint] = None  # Position des letzten Press (nicht mehr für Doppelklick verwendet)
        # 🎯 DOPPELKLICK DEAKTIVIERT - _double_click_handled wird nicht mehr verwendet (nur noch auf Colorbar)
        self._double_click_handled = False
        self._camera_state: Optional[dict[str, object]] = None
        self._skip_next_render_restore = False
        self._phase_mode_active = False
        
        # ColorbarManager initialisieren
        self.colorbar_manager = ColorbarManager(
            colorbar_ax=colorbar_ax,
            settings=settings,
            phase_mode_active=False,
            time_mode_active=False,
        )
        # Doppelklick-Callback für ColorbarManager
        self.colorbar_manager.on_double_click = self._handle_double_click_auto_range
        self._has_plotted_data = False  # Flag: ob bereits ein Plot mit Daten durchgeführt wurde
        # Flag: ob der initiale Zoom auf das Default-Surface bereits gesetzt wurde
        self._did_initial_overlay_zoom = False
        # Drag-Variablen für Achsenlinien
        self._axis_selected: Optional[str] = None  # 'x' oder 'y' für ausgewählte Achse, None wenn keine ausgewählt
        self._axis_drag_active = False  # True wenn eine Achsenlinie gedraggt wird
        self._axis_drag_type: Optional[str] = None  # 'x' oder 'y' für X- oder Y-Achse
        self._axis_drag_start_pos: Optional[QtCore.QPoint] = None  # Start-Position des Drags (2D)
        self._axis_drag_start_value: Optional[float] = None  # Start-Wert der Achsenposition

        # Letztes Speaker-Picking in Bildschirmkoordinaten (für Screen-Space-Heuristik)
        self._last_speaker_pick_screen_dist: Optional[float] = None
        self._last_speaker_pick_is_fallback: bool = False

        if not hasattr(self.plotter, '_cmap_to_lut'):
            self.plotter._cmap_to_lut = types.MethodType(self._fallback_cmap_to_lut, self.plotter)  # type: ignore[attr-defined]

        self.overlay_helper = SPL3DOverlayRenderer(self.plotter, pv)

        # cbar wird jetzt über colorbar_manager verwaltet
        self.cbar = None  # Wird von colorbar_manager gesetzt
        self.has_data = False
        self.surface_mesh = None  # type: pv.DataSet | None
        # Einzelne Actors pro horizontaler Surface (SPL-Flächen, Mesh-Modus)
        self._surface_actors: dict[str, Any] = {}
        # Textur-Actors pro horizontaler Surface (Texture-Modus)
        # Struktur: {surface_id: {'actor': actor, 'metadata': {...}}}
        self._surface_texture_actors: dict[str, dict[str, Any]] = {}
        # 🚀 TEXTUR-CACHE: Speichert Signaturen für Textur-Wiederverwendung
        # Struktur: {surface_id: texture_signature}
        self._surface_texture_cache: dict[str, str] = {}
        self._plot_geometry_cache = None  # Cache für Plot-Geometrie (plot_x, plot_y)
        # Zusätzliche SPL-Meshes für senkrechte Flächen (werden über calculation_spl gespeist)
        self._vertical_surface_meshes: dict[str, Any] = {}
        self.time_control: Optional[SPLTimeControlBar] = None
        self._time_slider_callback = None
        self._time_mode_active = False
        self._time_mode_surface_cache: dict | None = None

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
            # 🎯 DOPPELKLICK KOMPLETT SPERREN - Fange Doppelklick-Events ab, bevor PyVista sie verarbeitet
            if etype == QtCore.QEvent.MouseButtonDblClick:
                event.accept()
                return True  # Event abgefangen, PyVista verarbeitet es nicht
            
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                press_pos = QtCore.QPoint(event.pos())
                # 🎯 DOPPELKLICK DEAKTIVIERT - Alle Doppelklick-Funktionen sind auskommentiert
                # # 🎯 WICHTIG: Doppelklick-Erkennung BEIM PRESS, nicht beim Release!
                # # Prüfe ob es ein Doppelklick ist (basierend auf letztem Press, nicht Release)
                # is_double_click = False
                # if self._last_press_time is not None and self._last_press_pos is not None:
                #     current_time = time.time()
                #     time_diff = current_time - self._last_press_time
                #     pos_x_diff = abs(press_pos.x() - self._last_press_pos.x())
                #     pos_y_diff = abs(press_pos.y() - self._last_press_pos.y())
                #     pos_diff = pos_x_diff < 15 and pos_y_diff < 15  # Erhöht auf 15 Pixel Toleranz
                #     print(f"[DEBUG] Doppelklick-Prüfung: time_diff={time_diff:.3f}s (max 0.5s), pos_diff_x={pos_x_diff:.1f}px, pos_diff_y={pos_y_diff:.1f}px (max 15px), pos_diff={pos_diff}")
                #     # Doppelklick wenn innerhalb von 500ms und ähnlicher Position (15 Pixel Toleranz)
                #     if time_diff < 0.5 and pos_diff:
                #         is_double_click = True
                #         print(f"[DEBUG] ✅ Doppelklick BEIM PRESS erkannt! time_diff={time_diff:.3f}, pos_diff={pos_diff}")
                #         # 🎯 KRITISCH: Rotation SOFORT beenden, auch wenn sie bereits aktiv ist!
                #         # Dies verhindert, dass Rotation "fix mit Mausbewegung verbunden" bleibt
                #         if self._rotate_active:
                #             print(f"[DEBUG] ⚠️ Rotation war aktiv - SOFORT gestoppt!")
                #         self._rotate_active = False
                #         self._rotate_last_pos = None
                #         self._double_click_handled = True
                #         self.widget.unsetCursor()
                #         # Automatische Colorbar-Range-Anpassung
                #         self._handle_double_click_auto_range()
                #         # Reset für nächsten Klick
                #         self._last_press_time = None
                #         self._last_press_pos = None
                #         self._click_start_pos = None
                #         # 🎯 WICHTIG: Event akzeptieren und NICHT weitergeben, damit PyVista keine Rotation ausführt
                #         event.accept()
                #         return True  # Event abgefangen, nicht an PyVista weitergeben
                #     else:
                #         reason = []
                #         if time_diff >= 0.5:
                #             reason.append(f"Zeit zu lang ({time_diff:.3f}s >= 0.5s)")
                #         if not pos_diff:
                #             reason.append(f"Position zu weit (X: {pos_x_diff:.1f}px, Y: {pos_y_diff:.1f}px > 15px)")
                #         print(f"[DEBUG] ❌ Kein Doppelklick: {', '.join(reason) if reason else 'Unbekannter Grund'}")
                # else:
                #     print(f"[DEBUG] Doppelklick-Prüfung übersprungen: _last_press_time={self._last_press_time}, _last_press_pos={self._last_press_pos}")
                
                # 🎯 DOPPELKLICK KOMPLETT DEAKTIVIERT - Nur noch auf Colorbar aktiv
                # Keine Doppelklick-Erkennung im 3D-Plot mehr
                is_double_click = False
                self._double_click_handled = False
                
                # WICHTIG: _click_start_pos MUSS beim Press gesetzt werden, nicht beim Release
                self._click_start_pos = press_pos  # Merke Start-Position für Click-Erkennung
                # Speichere Press-Zeitpunkt und Position für Doppelklick-Erkennung
                self._last_press_time = time.time()
                self._last_press_pos = press_pos
                # 🎯 WICHTIG: Event akzeptieren, damit PyVista's Standard-Handler NICHT ausgeführt wird.
                # Dies verhindert, dass PyVista's Standard-Rotation aktiviert wird.
                # Achsen-Klick und -Drag im 3D-Plot sind deaktiviert; Achsenpositionen
                # werden nur noch über die UI gesteuert.
                event.accept()
                # 🎯 DOPPELKLICK DEAKTIVIERT - Rotation wird normal aktiviert
                # Rotation wird erst im MouseMove-Handler aktiviert, wenn sich die Maus tatsächlich bewegt.
                # Nur die Start-Position speichern, aber Rotation noch nicht aktivieren.
                self._rotate_last_pos = press_pos
                # Cursor erst setzen, wenn Rotation tatsächlich startet (im MouseMove)
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                # Prüfe ob es ein Klick war (kein Drag)
                click_pos = QtCore.QPoint(event.pos())
                move_dx = abs(click_pos.x() - self._click_start_pos.x()) if self._click_start_pos is not None else 0
                move_dy = abs(click_pos.y() - self._click_start_pos.y()) if self._click_start_pos is not None else 0
                # Etwas großzügigerer Schwellenwert, damit leichte Handbewegungen
                # (z.B. beim Klicken auf senkrechte Flächen) trotzdem als Klick zählen.
                click_threshold = 6
                is_click = (
                    self._click_start_pos is not None
                    and move_dx < click_threshold
                    and move_dy < click_threshold
                )
                
                # Rotation IMMER beenden bei ButtonRelease
                self._rotate_active = False
                self._rotate_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                # Reset Doppelklick-Flag beim Release (damit nach Doppelklick wieder normal gearbeitet werden kann)
                # 🎯 DOPPELKLICK DEAKTIVIERT - Doppelklick-Check entfernt
                # if self._double_click_handled:
                #     print(f"[DEBUG] MouseButtonRelease: Doppelklick-Flag zurückgesetzt")
                #     self._double_click_handled = False
                
                # 🎯 Doppelklick ist deaktiviert
                # Wenn es ein Klick war (kein Drag), prüfe Speaker und Surfaces gleichwertig.
                if is_click:
                    # 🎯 GLEICHWERTIGE BEHANDLUNG: Ray-Casting für beide (Speaker UND Surfaces)
                    # und wähle das Element, das näher zur Kamera ist (kleineres t)
                    
                    # 1. Prüfe Speaker
                    speaker_t, speaker_array_id, speaker_index = self._pick_speaker_at_position(click_pos)
                    speaker_screen_dist = getattr(self, "_last_speaker_pick_screen_dist", None)
                    speaker_is_fallback = getattr(self, "_last_speaker_pick_is_fallback", False)
                    
                    # 2. Prüfe Surfaces
                    surface_t, surface_id = self._pick_surface_at_position(click_pos)
                    
                    # 3. Vergleiche Distanzen und wähle näheres Element
                    if speaker_t is not None and surface_t is not None:
                        # Beide gefunden
                        if speaker_is_fallback and speaker_screen_dist is not None:
                            # 🎯 Screen-Space-Heuristik: wenn der Speaker-Fallback aktiv ist
                            # und wir nahe an der 2D-Projektion geklickt haben, bevorzuge
                            # den Speaker unabhängig vom exakten t-Vergleich.
                            self._select_speaker_in_treewidget(speaker_array_id, speaker_index)
                        else:
                            # Standard: wähle das näherliegende Objekt entlang des Rays
                            depth_eps = 0.5  # Meter Toleranz entlang des Rays
                            if speaker_t <= surface_t or abs(speaker_t - surface_t) <= depth_eps:
                                # Speaker ist näher oder praktisch gleich weit -> bevorzuge Speaker,
                                # damit Subs direkt auf der Stage gegenüber der Fläche gewinnen.
                                self._select_speaker_in_treewidget(speaker_array_id, speaker_index)
                            else:
                                self._select_surface_in_treewidget(surface_id)
                    elif speaker_t is not None:
                        # Nur Speaker gefunden
                        self._select_speaker_in_treewidget(speaker_array_id, speaker_index)
                    elif surface_t is not None:
                        # Nur Surface gefunden
                        self._select_surface_in_treewidget(surface_id)
                    else:
                        # Nichts gefunden
                        pass
                    # Speichere Zeitpunkt und Position für einfache Klicks (nicht für Doppelklick)
                    self._last_click_time = time.time()
                    self._last_click_pos = QtCore.QPoint(click_pos)
                    # _click_start_pos erst hier zurücksetzen (nach einfachem Klick)
                    self._click_start_pos = None
                else:
                    # Kein Klick (Drag) - _click_start_pos zurücksetzen
                    self._click_start_pos = None
                # Beende Achsenlinien-Drag beim Release (aber behalte Auswahl)
                if self._axis_drag_active:
                    self._axis_drag_active = False
                    self._axis_drag_start_pos = None
                    self._axis_drag_start_value = None
                    # Cursor zurücksetzen, aber Auswahl behalten
                    if self._axis_selected is not None:
                        self.widget.setCursor(QtCore.Qt.PointingHandCursor)
                    else:
                        self.widget.unsetCursor()
                
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
            if etype == QtCore.QEvent.MouseMove and event.buttons() & QtCore.Qt.LeftButton:
                # 🎯 DOPPELKLICK KOMPLETT DEAKTIVIERT - Nur noch auf Colorbar aktiv
                # Keine Doppelklick-Prüfung mehr im 3D-Plot
                
                # Wenn _rotate_last_pos gesetzt ist, aber _rotate_active noch nicht, dann Rotation jetzt starten
                if self._rotate_last_pos is not None and not self._rotate_active:
                    current_pos = QtCore.QPoint(event.pos())
                    # Prüfe ob sich die Maus tatsächlich bewegt hat (mindestens 3 Pixel)
                    move_distance = ((current_pos.x() - self._rotate_last_pos.x()) ** 2 + 
                                    (current_pos.y() - self._rotate_last_pos.y()) ** 2) ** 0.5
                    if move_distance >= 3:
                        # Maus hat sich bewegt - Rotation jetzt aktivieren
                        # NUR wenn Maus gedrückt bleibt (buttons() & QtCore.Qt.LeftButton)
                        self._rotate_active = True
                        self.widget.setCursor(QtCore.Qt.OpenHandCursor)
                
                # Wenn Rotation aktiv ist, verarbeite die Bewegung
                # DOPPELKLICK DEAKTIVIERT - Doppelklick-Check entfernt
                if self._rotate_active and self._rotate_last_pos is not None and self._axis_selected is None:
                    current_pos = QtCore.QPoint(event.pos())
                    delta = current_pos - self._rotate_last_pos
                    self._rotate_last_pos = current_pos
                    self._rotate_camera(delta.x(), delta.y())
                    event.accept()
                    return True
                # Wenn Achse ausgewählt ist, verhindere Rotation
                if self._axis_selected is not None and self._rotate_active:
                    self._rotate_active = False
                    self._rotate_last_pos = None
            if etype == QtCore.QEvent.MouseMove and self._pan_active and self._pan_last_pos is not None:
                current_pos = QtCore.QPoint(event.pos())
                delta = current_pos - self._pan_last_pos
                self._pan_last_pos = current_pos
                self._pan_camera(delta.x(), delta.y())
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseMove and self._axis_drag_active and self._axis_drag_start_pos is not None:
                # Drag einer Achsenlinie
                current_pos = QtCore.QPoint(event.pos())
                self._handle_axis_line_drag(current_pos)
                event.accept()
                return True
            if etype == QtCore.QEvent.MouseMove and not self._pan_active and not self._rotate_active and not self._axis_drag_active:
                # Mouse-Move ohne Pan/Rotate - zeige Mausposition an
                self._handle_mouse_move_3d(event.pos())
                # Nicht akzeptieren, damit andere Handler auch reagieren können
            if etype == QtCore.QEvent.Leave:
                if self._pan_active or self._rotate_active or self._axis_drag_active:
                    self._pan_active = False
                    self._rotate_active = False
                    self._axis_drag_active = False
                    self._axis_drag_type = None
                    self._axis_drag_start_pos = None
                    self._axis_drag_start_value = None
                    self._pan_last_pos = None
                    self._rotate_last_pos = None
                    self.widget.unsetCursor()
                    self._save_camera_state()
                    event.accept()
                    return True
                # Leere Mauspositions-Anzeige beim Verlassen
                if self.main_window and hasattr(self.main_window, 'ui') and hasattr(self.main_window.ui, 'mouse_position_label'):
                    self.main_window.ui.mouse_position_label.setText("")
            if etype == QtCore.QEvent.Wheel:
                QtCore.QTimer.singleShot(0, self._save_camera_state)
        return QtCore.QObject.eventFilter(self, obj, event)

    def _handle_surface_click(self, click_pos: QtCore.QPoint) -> None:
        """Behandelt einen Klick auf eine Surface im 3D-Plot und wählt das entsprechende Item im TreeWidget aus."""
        try:
            # Prüfe ZUERST ob eine Achsenlinie geklickt wurde (verhindere Surface-Aktivierung)
            axis_actor_names = self.overlay_helper._category_actors.get('axis', [])
            if axis_actor_names:
                # Verwende VTK CellPicker um zu prüfen ob Achsenlinie gepickt wurde
                try:
                    from vtkmodules.vtkRenderingCore import vtkCellPicker
                    renderer = self.plotter.renderer
                    if renderer is None:
                        return
                    
                    picker = vtkCellPicker()
                    picker.SetTolerance(0.05)  # Erhöht von 0.01 auf 0.05 für besseres Picking
                    
                    size = self.widget.size()
                    if size.width() > 0 and size.height() > 0:
                        x_norm = click_pos.x() / size.width()
                        y_norm = 1.0 - (click_pos.y() / size.height())
                        
                        picker.Pick(x_norm, y_norm, 0.0, renderer)
                        picked_actor = picker.GetActor()
                        
                        if picked_actor is not None:
                            # Prüfe ob gepickter Actor eine Achsenlinie ist
                            actor_name = None
                            if hasattr(picked_actor, 'name'):
                                actor_name = picked_actor.name
                            else:
                                for name, actor in renderer.actors.items():
                                    if actor == picked_actor:
                                        actor_name = name
                                        break
                            # Wenn Achsenlinie gepickt wurde, ignoriere Surface-Click
                            if actor_name and actor_name in axis_actor_names:
                                return
                except Exception:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
            
            # QtInteractor erbt von Plotter, aber pick() könnte nicht verfügbar sein
            # Verwende stattdessen den Renderer direkt für Picking
            renderer = self.plotter.renderer
            if renderer is None:
                return
            
            # Verwende VTK CellPicker für präzises Picking (Vertikalflächen + horizontale Fläche)
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker
                picker = vtkCellPicker()
                picker.SetTolerance(0.05)  # Erhöht für besseres Picking von senkrechten Flächen
                
                # Konvertiere Qt-Koordinaten zu VTK-Display-Koordinaten (Pixel)
                size = self.widget.size()
                if size.width() > 0 and size.height() > 0:
                    display_x = float(click_pos.x())
                    display_y = float(size.height() - click_pos.y())  # VTK Y ist invertiert
                    
                    # 1) Picke Actor an dieser Position
                    picker.Pick(display_x, display_y, 0.0, renderer)
                    picked_actor = picker.GetActor()
                    picked_point = picker.GetPickPosition()
                    
                    actor_name = None
                    if picked_actor is not None:
                        # Versuche Actor-Name herauszufinden
                        actor_name = getattr(picked_actor, "name", None)
                        if actor_name is None:
                            for name, actor in renderer.actors.items():
                                if actor == picked_actor:
                                    actor_name = name
                                    break
                    
                    # 2) Vertikalflächen: vertical_spl_<surface_id> (direkt getroffen)
                    if isinstance(actor_name, str) and actor_name.startswith("vertical_spl_"):
                        surface_id = actor_name[len("vertical_spl_") :]
                        self._select_surface_in_treewidget(surface_id)
                        return
                    
                    # 3) Prüfe zuerst senkrechte Flächen mit Ray-Casting (auch disabled)
                    # Dies muss VOR der Prüfung auf planare Flächen passieren, damit senkrechte Flächen Vorrang haben
                    # Auch wenn surface_disabled_polygons_batch getroffen wurde, prüfe ob dahinter eine senkrechte Fläche liegt
                    if (picked_actor is None or not isinstance(actor_name, str) or not actor_name.startswith("vertical_spl_")):
                        # Berechne Ray vom Klickpunkt für Fallback-Prüfung
                        render_size = renderer.GetSize()
                        if render_size[0] > 0 and render_size[1] > 0:
                            viewport_x = display_x
                            viewport_y = display_y
                            
                            renderer.SetDisplayPoint(viewport_x, viewport_y, 0.0)
                            renderer.DisplayToWorld()
                            world_point_near = renderer.GetWorldPoint()
                            
                            renderer.SetDisplayPoint(viewport_x, viewport_y, 1.0)
                            renderer.DisplayToWorld()
                            world_point_far = renderer.GetWorldPoint()
                            
                            if world_point_near and world_point_far and len(world_point_near) >= 4 and len(world_point_far) >= 4:
                                if abs(world_point_near[3]) > 1e-6 and abs(world_point_far[3]) > 1e-6:
                                    ray_start = np.array([
                                        world_point_near[0] / world_point_near[3],
                                        world_point_near[1] / world_point_near[3],
                                        world_point_near[2] / world_point_near[3],
                                    ])
                                    ray_end = np.array([
                                        world_point_far[0] / world_point_far[3],
                                        world_point_far[1] / world_point_far[3],
                                        world_point_far[2] / world_point_far[3],
                                    ])
                                    ray_dir = ray_end - ray_start
                                    ray_len = np.linalg.norm(ray_dir)
                                    if ray_len > 1e-8:
                                        ray_dir = ray_dir / ray_len
                                        
                                        # Prüfe alle senkrechten Flächen
                                        best_surface_id = None
                                        best_t = float('inf')
                                        
                                        for vertical_actor_name, vertical_actor in self._vertical_surface_meshes.items():
                                            if not vertical_actor_name.startswith("vertical_spl_"):
                                                continue
                                            
                                            surface_id_candidate = vertical_actor_name[len("vertical_spl_"):]
                                            
                                            # Versuche Ray mit Mesh zu schneiden
                                            try:
                                                # Hole das Mesh aus dem Actor
                                                if hasattr(vertical_actor, 'mapper') and hasattr(vertical_actor.mapper, 'GetInput'):
                                                    mesh = vertical_actor.mapper.GetInput()
                                                    if mesh is not None:
                                                        # Prüfe ob Ray das Mesh schneidet
                                                        bounds = mesh.GetBounds()
                                                        if bounds and len(bounds) >= 6:
                                                            # Einfache Bounding-Box-Prüfung
                                                            t_min = float('inf')
                                                            t_max = float('-inf')
                                                            
                                                            for i in range(3):
                                                                if abs(ray_dir[i]) > 1e-8:
                                                                    t1 = (bounds[i*2] - ray_start[i]) / ray_dir[i]
                                                                    t2 = (bounds[i*2+1] - ray_start[i]) / ray_dir[i]
                                                                    t_min = min(t_min, max(t1, t2))
                                                                    t_max = max(t_max, min(t1, t2))
                                                            
                                                            if t_max >= 0 and t_min < best_t:
                                                                # Verwende VTK's CellLocator für robuste Ray-Mesh-Intersection
                                                                # CellLocator ist viel robuster bei schrägen Winkeln als PointLocator
                                                                try:
                                                                    from vtkmodules.vtkCommonDataModel import vtkCellLocator
                                                                    cell_locator = vtkCellLocator()
                                                                    cell_locator.SetDataSet(mesh)
                                                                    cell_locator.BuildLocator()
                                                                    
                                                                    # Ray-Casting: Finde Schnittpunkt des Rays mit dem Mesh
                                                                    tolerance = 0.01  # Toleranz für Ray-Intersection (1 cm)
                                                                    t_hit = [0.0]  # Output-Parameter für t
                                                                    x_hit = [0.0, 0.0, 0.0]  # Output-Parameter für Schnittpunkt
                                                                    pcoords = [0.0, 0.0, 0.0]  # Output-Parameter für Zellkoordinaten
                                                                    sub_id = [0]  # Output-Parameter für Sub-Zell-ID
                                                                    cell_id = cell_locator.IntersectWithLine(
                                                                        ray_start.tolist(),
                                                                        (ray_start + ray_dir * 1000.0).tolist(),  # Ray-Endpunkt weit weg
                                                                        tolerance,
                                                                        t_hit,
                                                                        x_hit,
                                                                        pcoords,
                                                                        sub_id
                                                                    )
                                                                    
                                                                    # Wenn eine Zelle getroffen wurde (cell_id >= 0)
                                                                    if cell_id >= 0 and t_hit[0] >= 0 and t_hit[0] < best_t:
                                                                        best_t = t_hit[0]
                                                                        best_surface_id = surface_id_candidate
                                                                except Exception:
                                                                    # Fallback: Wenn CellLocator fehlschlägt, verwende Bounding-Box mit größerer Toleranz
                                                                    # Verwende t_max (nähester Punkt der Bounding-Box) mit größerer Toleranz
                                                                    if t_max >= 0 and t_max < best_t:
                                                                        # Berechne Distanz vom Ray zum Mesh-Zentrum als zusätzliche Prüfung
                                                                        mesh_center = np.array([
                                                                            (bounds[0] + bounds[1]) / 2.0,
                                                                            (bounds[2] + bounds[3]) / 2.0,
                                                                            (bounds[4] + bounds[5]) / 2.0,
                                                                        ])
                                                                        ray_to_center = mesh_center - ray_start
                                                                        proj_length = np.dot(ray_to_center, ray_dir)
                                                                        if proj_length > 0:
                                                                            proj_point = ray_start + ray_dir * proj_length
                                                                            dist_to_center = np.linalg.norm(proj_point - mesh_center)
                                                                            # Toleranz für schräge Winkel bei senkrechten Flächen: 2 Meter
                                                                            # (höher als bei planaren Flächen, da senkrechte Flächen bei schrägen Winkeln schwerer zu treffen sind)
                                                                            if dist_to_center < 2.0:
                                                                                best_t = t_max
                                                                                best_surface_id = surface_id_candidate
                                            except Exception:
                                                continue
                                        
                                        if best_surface_id is not None:
                                            self._select_surface_in_treewidget(best_surface_id)
                                            return
                    
                    # 4) Disabled-Batch-Flächen (horizontale Surfaces als Batch-Mesh)
                    # Nur prüfen, wenn keine senkrechte Fläche gefunden wurde
                    # HINWEIS: Diese sind nicht pickable, daher wird actor_name niemals auf diese zeigen
                    # Wir müssen stattdessen immer _handle_spl_surface_click aufrufen, damit Ray-Casting
                    # auch disabled Surfaces findet
                    if isinstance(actor_name, str) and actor_name in (
                        "surface_disabled_polygons_batch",
                        "surface_disabled_edges_batch",
                    ):
                        self._handle_spl_surface_click(click_pos)
                        return
                    
                    # 5) SPL-Hauptfläche oder andere pickable Surfaces
                    # Auch hier müssen wir _handle_spl_surface_click aufrufen, damit Ray-Casting
                    # die am nächsten zur Kamera liegende Surface findet (enabled oder disabled)
                    if actor_name == self.SURFACE_NAME:
                        self._handle_spl_surface_click(click_pos)
                        return
                    
                    # 6) Kein klarer Actor → versuche generisch SPL-Surface mit Ray-Casting
                    # Ray-Casting findet sowohl enabled als auch disabled Surfaces
                    if picked_point and len(picked_point) >= 3:
                        self._handle_spl_surface_click(click_pos)
                        return
                    
                    # 7) Wenn ein pickable Actor gefunden wurde, aber nicht erkannt wurde,
                    # verwende trotzdem Ray-Casting, um die am nächsten zur Kamera liegende
                    # Surface zu finden (kann enabled oder disabled sein)
                    if picked_actor is not None:
                        self._handle_spl_surface_click(click_pos)
                        return
            except ImportError:
                # Fallback: Versuche PyVista pick über renderer
                try:
                    pass
                except Exception:
                    pass
            
            # Kein Actor gefunden - prüfe ob auf SPL-Surface geklickt wurde (für enabled und disabled Surfaces)
            # Ray-Casting findet sowohl enabled als auch disabled Surfaces
            self._handle_spl_surface_click(click_pos)
        except Exception as e:  # noqa: BLE001
            # Bei Fehler einfach ignorieren
            pass

    def _handle_axis_line_click(self, click_pos: QtCore.QPoint) -> bool:
        """Behandelt einen Klick auf eine Achsenlinie im 3D-Plot.
        
        Returns:
            bool: True wenn eine Achsenlinie geklickt wurde, False sonst.
        """
        try:
            renderer = self.plotter.renderer
            if renderer is None:
                return False
            
            # Hole alle Achsenlinien-Actors aus dem Overlay-Renderer
            axis_actor_names = self.overlay_helper._category_actors.get('axis', [])
            if not axis_actor_names:
                return False
            
            # Verwende VTK CellPicker für präzises Picking
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker
                picker = vtkCellPicker()
                # Erhöhte Toleranz für besseres Picking von dünnen Linien (in Display-Koordinaten)
                # Toleranz in Viewport-Koordinaten (0-1), 0.1 = 10% der Bildschirmbreite
                picker.SetTolerance(0.1)  # Erhöht auf 0.1 für besseres Picking von dünnen Linien
                
                size = self.widget.size()
                if size.width() <= 0 or size.height() <= 0:
                    return False
                
                # VTK Display-Koordinaten (Pixel-Koordinaten)
                display_x = float(click_pos.x())
                display_y = float(size.height() - click_pos.y())  # VTK Y ist invertiert
                
                # Normale Viewport-Koordinaten für Pick
                x_norm = click_pos.x() / size.width()
                y_norm = 1.0 - (click_pos.y() / size.height())
                
                picker.Pick(x_norm, y_norm, 0.0, renderer)
                picked_actor = picker.GetActor()
                
                if picked_actor is None:
                    # Kein Actor gepickt
                    return False
                
                # Prüfe ob der gepickte Actor eine Achsenlinie ist
                actor_name = None
                if hasattr(picked_actor, 'name'):
                    actor_name = picked_actor.name
                else:
                    # Suche in renderer.actors
                    for name, actor in renderer.actors.items():
                        if actor == picked_actor:
                            actor_name = name
                            break
                
                if actor_name and actor_name in axis_actor_names:
                    # Achsenlinie wurde geklickt - bestimme ob X- oder Y-Achse
                    # Hole die gepickte 3D-Position
                    picked_point = picker.GetPickPosition()
                    if picked_point and len(picked_point) >= 3:
                        x_picked, y_picked = picked_point[0], picked_point[1]
                        # Prüfe welche Achse näher ist (X-Achse: y=y_axis, Y-Achse: x=x_axis)
                        x_axis = self.settings.position_x_axis
                        y_axis = self.settings.position_y_axis
                        
                        # Berechne Distanzen zu beiden Achsen
                        dist_to_x_axis = abs(y_picked - y_axis)
                        dist_to_y_axis = abs(x_picked - x_axis)
                        
                        # Bestimme welche Achse geklickt wurde (die nähere)
                        if dist_to_x_axis < dist_to_y_axis:
                            self._axis_drag_type = 'x'
                        else:
                            self._axis_drag_type = 'y'
                        
                        self._axis_drag_active = True
                        return True
                
                return False
            except ImportError:
                return False
            except Exception:
                return False
        except Exception:
            return False

    def _handle_axis_line_drag(self, current_pos: QtCore.QPoint) -> None:
        """Behandelt das Drag einer Achsenlinie und aktualisiert die Position."""
        if not self._axis_drag_active or self._axis_drag_type is None:
            return
        
        try:
            renderer = self.plotter.renderer
            if renderer is None:
                return
            
            size = self.widget.size()
            if size.width() <= 0 or size.height() <= 0:
                return
            
            # Konvertiere 2D-Mausposition zu 3D-Weltkoordinaten
            # Verwende DisplayToWorld um 3D-Koordinaten zu bekommen
            # Für X-Achse: wir wollen die Y-Koordinate ändern
            # Für Y-Achse: wir wollen die X-Koordinate ändern
            
            # Hole Display-Koordinaten
            display_x = float(current_pos.x())
            display_y = float(size.height() - current_pos.y())
            
            # Konvertiere Display zu World-Koordinaten
            # Verwende renderer.SetDisplayPoint und renderer.DisplayToWorld
            renderer.SetDisplayPoint(display_x, display_y, 0.0)
            renderer.DisplayToWorld()
            world_point = renderer.GetWorldPoint()
            if len(world_point) >= 4 and world_point[3] != 0:
                world_x = world_point[0] / world_point[3]
                world_y = world_point[1] / world_point[3]
                world_z = world_point[2] / world_point[3]
                
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
                        # Aktualisiere UI
                        self._update_axis_position_ui()
                        # Aktualisiere Overlays (Achsenlinien neu zeichnen)
                        if hasattr(self, 'update_overlays'):
                            self.update_overlays()
                        # Aktualisiere Highlight nach Drag
                        self._update_axis_highlight()
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
                        # Aktualisiere UI
                        self._update_axis_position_ui()
                        # Aktualisiere Overlays (Achsenlinien neu zeichnen)
                        if hasattr(self, 'update_overlays'):
                            self.update_overlays()
                        # Aktualisiere Highlight nach Drag
                        self._update_axis_highlight()
                        # Aktualisiere Berechnungen
                        if self.main_window and hasattr(self.main_window, 'update_speaker_array_calculations'):
                            self.main_window.update_speaker_array_calculations()
        except Exception:
            import traceback
            traceback.print_exc()

    def _update_axis_position_ui(self) -> None:
        """Aktualisiert die UI-Felder für die Achsenpositionen."""
        try:
            if self.main_window and hasattr(self.main_window, 'ui'):
                ui = self.main_window.ui
                # Prüfe ob es ein Settings-Widget gibt
                if hasattr(ui, 'position_plot_length'):
                    ui.position_plot_length.setText(str(self.settings.position_x_axis))
                if hasattr(ui, 'position_plot_width'):
                    ui.position_plot_width.setText(str(self.settings.position_y_axis))
        except Exception:
            pass

    def _update_axis_highlight(self) -> None:
        """Aktualisiert die Highlight-Farbe der Achsenlinien (rot wenn ausgewählt).
        Zeichnet die Achsenlinien neu mit der richtigen Farbe.
        """
        try:
            # Zeichne Achsenlinien neu mit Highlight-Status
            # Die Signatur enthält jetzt selected_axis, daher wird 'axis' automatisch in categories_to_refresh sein
            if hasattr(self, 'update_overlays') and hasattr(self, 'settings') and hasattr(self, 'container'):
                self.update_overlays(self.settings, self.container)
        except Exception:
            import traceback
            traceback.print_exc()
    
    def _pick_speaker_at_position(self, click_pos: QtCore.QPoint) -> Tuple[Optional[float], Optional[str], Optional[int]]:
        """Findet den Speaker am Klickpunkt und gibt die Distanz zur Kamera zurück.

        Args:
            click_pos: Klickposition im Widget
        
        Returns:
            Tuple[Optional[float], Optional[str], Optional[int]]:
                (distanz_zur_kamera, array_id, speaker_index) oder (None, None, None)
        """
        try:
            # Reset Screen-Space-Status für dieses Picking
            self._last_speaker_pick_screen_dist = None
            self._last_speaker_pick_is_fallback = False

            renderer = self.plotter.renderer
            if renderer is None:
                return (None, None, None)
            
            try:
                from vtkmodules.vtkRenderingCore import vtkCellPicker
                
                size = self.widget.size()
                if size.width() <= 0 or size.height() <= 0:
                    return (None, None, None)
                
                # 🎯 WICHTIG: Verwende dieselbe Normalisierung wie bei _pick_surface_at_position,
                # damit der Ray wirklich durch den sichtbaren Klickpunkt im Render-Viewport geht.
                render_size = renderer.GetSize()
                if render_size[0] <= 0 or render_size[1] <= 0:
                    return (None, None, None)
                x_norm = click_pos.x() / size.width()
                y_norm = 1.0 - (click_pos.y() / size.height())  # VTK Y ist invertiert
                display_x = float(x_norm * render_size[0])
                display_y = float(y_norm * render_size[1])

                # Berechne Ray für alle nachfolgenden Berechnungen
                renderer.SetDisplayPoint(display_x, display_y, 0.0)
                renderer.DisplayToWorld()
                world_point_near = renderer.GetWorldPoint()
                
                renderer.SetDisplayPoint(display_x, display_y, 1.0)
                renderer.DisplayToWorld()
                world_point_far = renderer.GetWorldPoint()
                
                if not world_point_near or not world_point_far or len(world_point_near) < 4 or len(world_point_far) < 4:
                    return (None, None, None)
                if abs(world_point_near[3]) <= 1e-6 or abs(world_point_far[3]) <= 1e-6:
                    return (None, None, None)

                # Berechne Ray-Start und Ray-Ende
                ray_start = np.array(
                    [
                        world_point_near[0] / world_point_near[3],
                        world_point_near[1] / world_point_near[3],
                        world_point_near[2] / world_point_near[3],
                    ]
                )
                ray_end = np.array(
                    [
                        world_point_far[0] / world_point_far[3],
                        world_point_far[1] / world_point_far[3],
                        world_point_far[2] / world_point_far[3],
                    ]
                )
                ray_dir = ray_end - ray_start
                ray_length = np.linalg.norm(ray_dir)
                if ray_length <= 0:
                    return (None, None, None)
                ray_dir = ray_dir / ray_length

                # Vereinfachtes, robustes Picking: nur Speaker-Overlay-Actors in Pick-Liste
                if not (hasattr(self, "overlay_helper") and hasattr(self.overlay_helper, "_speaker_actor_cache")):
                    return (None, None, None)

                speaker_cache = self.overlay_helper._speaker_actor_cache
                if not isinstance(speaker_cache, dict) or not speaker_cache:
                    return (None, None, None)

                picker = vtkCellPicker()
                # Erhöhte Toleranz, damit Lautsprecher-Gehäuse auch bei größerem Zoom
                # bzw. kleinen Screen‑Projektionen zuverlässiger getroffen werden.
                # Vorher: 0.02 – jetzt 0.08 für robusteres Picking.
                picker.SetTolerance(0.08)
                picker.PickFromListOn()

                actor_name_to_obj: dict[str, object] = {}
                for _key, info in speaker_cache.items():
                    cached_actor_name = info.get("actor")
                    if not cached_actor_name:
                        continue
                    actor = renderer.actors.get(cached_actor_name)
                    if actor is None:
                        continue
                    picker.AddPickList(actor)
                    actor_name_to_obj[str(cached_actor_name)] = actor

                # Pick ausführen
                picker.Pick(display_x, display_y, 0.0, renderer)
                picked_actor = picker.GetActor()
                picked_point = picker.GetPickPosition()
                
                if picked_actor is None:
                    # 🎯 Fallback: Exaktes Ray‑Mesh‑Intersection über alle Speaker-Meshes
                    # Analog zur Surface-Logik, aber mit echten 3D-Meshes (Dreiecke),
                    # damit auch geflogene/schräge Gehäuse robust getroffen werden.
                    from math import isfinite

                    def _ray_triangle_intersect(
                        origin: np.ndarray,
                        direction: np.ndarray,
                        v0: np.ndarray,
                        v1: np.ndarray,
                        v2: np.ndarray,
                        eps: float = 1e-8,
                    ) -> Optional[float]:
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        pvec = np.cross(direction, edge2)
                        det = float(np.dot(edge1, pvec))
                        if abs(det) < eps:
                            return None
                        inv_det = 1.0 / det
                        tvec = origin - v0
                        u = float(np.dot(tvec, pvec) * inv_det)
                        if u < 0.0 or u > 1.0:
                            return None
                        qvec = np.cross(tvec, edge1)
                        v = float(np.dot(direction, qvec) * inv_det)
                        if v < 0.0 or u + v > 1.0:
                            return None
                        t = float(np.dot(edge2, qvec) * inv_det)
                        if t <= 0.0:
                            return None
                        return t

                    best_t: Optional[float] = None
                    best_array_id: Optional[str] = None
                    best_speaker_idx: Optional[int] = None

                    for cache_key, cache_info in speaker_cache.items():
                        try:
                            array_id_key, speaker_idx_key, _geom_idx = cache_key
                        except Exception:
                            continue

                        mesh = cache_info.get("mesh")
                        if mesh is None or not hasattr(mesh, "points") or not hasattr(mesh, "faces"):
                            continue

                        try:
                            points = np.asarray(mesh.points, dtype=float)
                            faces = np.asarray(mesh.faces, dtype=int)
                        except Exception:
                            continue
                        if points.size == 0 or faces.size == 0:
                            continue

                        i = 0
                        while i < faces.size:
                            n = faces[i]
                            if n < 3:
                                i += 1 + n
                                continue
                            idx0 = faces[i + 1]
                            idx1 = faces[i + 2]
                            idx2 = faces[i + 3]
                            if (
                                idx0 < 0
                                or idx1 < 0
                                or idx2 < 0
                                or idx0 >= points.shape[0]
                                or idx1 >= points.shape[0]
                                or idx2 >= points.shape[0]
                            ):
                                i += 1 + n
                                continue
                            v0 = points[idx0]
                            v1 = points[idx1]
                            v2 = points[idx2]
                            t_hit = _ray_triangle_intersect(ray_start, ray_dir, v0, v1, v2)
                            if t_hit is not None and isfinite(t_hit):
                                if best_t is None or t_hit < best_t:
                                    best_t = t_hit
                                    best_array_id = str(array_id_key)
                                    best_speaker_idx = int(speaker_idx_key)

                            # Wenn es sich um ein Quad (4 Punkte) handelt, trianguliere in zwei Dreiecke
                            if n == 4:
                                idx3 = faces[i + 4]
                                if (
                                    0 <= idx3 < points.shape[0]
                                    and 0 <= idx2 < points.shape[0]
                                ):
                                    v3 = points[idx3]
                                    t_hit2 = _ray_triangle_intersect(ray_start, ray_dir, v0, v2, v3)
                                    if t_hit2 is not None and isfinite(t_hit2):
                                        if best_t is None or t_hit2 < best_t:
                                            best_t = t_hit2
                                            best_array_id = str(array_id_key)
                                            best_speaker_idx = int(speaker_idx_key)

                            i += 1 + n

                    if best_t is not None and best_array_id is not None and best_speaker_idx is not None:
                        # Prüfe auch hier die Distanz zwischen Trefferpunkt und Gehäusezentrum
                        try:
                            cache_info = speaker_cache.get((best_array_id, best_speaker_idx, 0)) or None
                            mesh_for_hit = None
                            if cache_info is not None:
                                mesh_for_hit = cache_info.get("mesh")
                            if (
                                mesh_for_hit is not None
                                and hasattr(mesh_for_hit, "points")
                                and hasattr(mesh_for_hit, "bounds")
                            ):
                                pts = np.asarray(mesh_for_hit.points, dtype=float)
                                bounds = getattr(mesh_for_hit, "bounds", None)
                                if pts.size > 0 and bounds is not None and len(bounds) >= 6:
                                    center = pts.mean(axis=0)
                                    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
                                    extents = np.array(
                                        [
                                            0.5 * (xmax - xmin),
                                            0.5 * (ymax - ymin),
                                            0.5 * (zmax - zmin),
                                        ],
                                        dtype=float,
                                    )
                                    box_radius = float(np.linalg.norm(extents))
                                    max_pick_dist = max(0.5, 0.5 * box_radius)
                                    hit_pt = ray_start + float(best_t) * ray_dir
                                    dist_center = float(np.linalg.norm(center - hit_pt))
                                    if dist_center > max_pick_dist:
                                        # Treffer außerhalb des zulässigen Abstands – keinen Speaker wählen
                                        return (None, None, None)
                        except Exception:
                            pass

                        print(
                            "[DEBUG] _pick_speaker_at_position: Ray-Mesh-Fallback-Treffer für Speaker "
                            f"array={best_array_id}, idx={best_speaker_idx}, t={best_t:.3f}"
                        )
                        return (float(best_t), best_array_id, best_speaker_idx)

                    return (None, None, None)

                # Actor-Namen ermitteln
                picked_name: Optional[str] = None
                for name, actor in actor_name_to_obj.items():
                    if actor is picked_actor:
                        picked_name = name
                        break
                if not isinstance(picked_name, str):
                    return (None, None, None)

                # Speaker-Info aus Cache ermitteln
                speaker_info = self.overlay_helper._get_speaker_info_from_actor(picked_actor, picked_name)
                if not speaker_info:
                    return (None, None, None)

                array_id, speaker_index = speaker_info

                # t-Wert entlang des Rays bestimmen (für Vergleich mit Surface-t)
                picked = np.array(picked_point[:3], dtype=float)
                vec = picked - ray_start
                t_hit = float(np.dot(vec, ray_dir))
                if t_hit < 0:
                    # Fallback: direkte Distanz nutzen, wenn Projektion „hinter“ der Kamera liegt
                    t_hit = float(np.linalg.norm(vec))

                # 🎯 Klick-Präzision verfeinern: akzeptiere nur Treffer, deren
                # Mittelpunkt in sinnvoller Distanz zum Pick-Punkt liegt.
                try:
                    mesh_for_hit = None
                    for cache_key, cache_info in speaker_cache.items():
                        try:
                            array_id_key, speaker_idx_key, _geom_idx = cache_key
                        except Exception:
                            continue
                        if str(array_id_key) == str(array_id) and int(speaker_idx_key) == int(speaker_index):
                            mesh_for_hit = cache_info.get("mesh")
                            if mesh_for_hit is not None:
                                break
                    if mesh_for_hit is not None and hasattr(mesh_for_hit, "points") and hasattr(
                        mesh_for_hit, "bounds"
                    ):
                        pts = np.asarray(mesh_for_hit.points, dtype=float)
                        bounds = getattr(mesh_for_hit, "bounds", None)
                        if pts.size > 0 and bounds is not None and len(bounds) >= 6:
                            center = pts.mean(axis=0)
                            xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
                            extents = np.array(
                                [
                                    0.5 * (xmax - xmin),
                                    0.5 * (ymax - ymin),
                                    0.5 * (zmax - zmin),
                                ],
                                dtype=float,
                            )
                            # Erlaube Klicks bis zur halben Diagonale des Gehäuses,
                            # mindestens aber 0.5 m (für sehr kleine Kisten).
                            box_radius = float(np.linalg.norm(extents))
                            max_pick_dist = max(0.5, 0.5 * box_radius)
                            dist_center = float(np.linalg.norm(center - picked))
                            if dist_center > max_pick_dist:
                                # Trefferpunkt zu weit vom Gehäusezentrum entfernt – verwerfen
                                return (None, None, None)
                except Exception:
                    # Bei Problemen mit der Distanzprüfung den Treffer nicht komplett verwerfen
                    pass

                return (t_hit, str(array_id), int(speaker_index))
            except ImportError as e:
                return (None, None, None)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                return (None, None, None)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return (None, None, None)
    
    def _handle_speaker_click(self, click_pos: QtCore.QPoint) -> bool:
        """Behandelt einen Klick auf einen Lautsprecher im 3D-Plot und wählt das entsprechende Array im TreeWidget aus.
        
        Returns:
            bool: True wenn ein Lautsprecher geklickt wurde, False sonst.
        """
        t, array_id, speaker_index = self._pick_speaker_at_position(click_pos)
        if array_id is not None and speaker_index is not None:
            self._select_speaker_in_treewidget(array_id, speaker_index)
            return True
        return False
    
    def _select_speaker_in_treewidget(self, array_id: str, speaker_index: int = None) -> None:
        """Wählt ein Speaker-Array im Sources-TreeWidget aus und wechselt zum Sources Widget."""
        try:
            # Finde das Main-Window
            parent = self.parent_widget
            main_window = None
            
            depth = 0
            while parent is not None and depth < 15:
                if hasattr(parent, 'sources_instance'):
                    main_window = parent
                    break
                
                next_parent = None
                if hasattr(parent, 'parent'):
                    next_parent = parent.parent()
                elif hasattr(parent, 'parentWidget'):
                    next_parent = parent.parentWidget()
                elif hasattr(parent, 'parent_widget'):
                    next_parent = parent.parent_widget
                elif hasattr(parent, 'window'):
                    next_parent = parent.window()
                
                if next_parent == parent:
                    break
                parent = next_parent
                depth += 1
            
            if main_window is None:
                # Versuche alternativen Weg
                parent = self.parent_widget
                while parent is not None:
                    if isinstance(parent, QtWidgets.QMainWindow):
                        if hasattr(parent, 'sources_instance'):
                            main_window = parent
                            break
                    if hasattr(parent, 'parent'):
                        parent = parent.parent()
                    elif hasattr(parent, 'parentWidget'):
                        parent = parent.parentWidget()
                    else:
                        break
                
                if main_window is None:
                    return
            
            # Wechsle zum Sources Widget (falls im Surface Widget)
            if hasattr(main_window, 'sources_instance') and hasattr(main_window, 'surface_manager'):
                sources_dock = getattr(main_window.sources_instance, 'sources_dockWidget', None)
                if sources_dock:
                    sources_dock.raise_()  # Zeige Sources Widget
            
            # Hole Sources-Instanz
            sources_instance = main_window.sources_instance
            if not sources_instance:
                return
            if not hasattr(sources_instance, 'sources_tree_widget'):
                return
            
            # Finde Array-Item im TreeWidget
            tree_widget = sources_instance.sources_tree_widget
            if tree_widget is None:
                return
            
            # Suche nach Array-Item (rekursiv durch alle Items, inkl. Gruppen)
            # Konvertiere array_id zu String für konsistenten Vergleich
            array_id_str = str(array_id)
            
            def find_array_item(parent_item=None, depth=0):
                """Rekursive Suche nach Array-Item, auch in Gruppen"""
                indent = "  " * depth
                if parent_item is None:
                    # Suche in allen Top-Level Items
                    for i in range(tree_widget.topLevelItemCount()):
                        item = tree_widget.topLevelItem(i)
                        if item:
                            found = find_array_item(item, depth + 1)
                        if found:
                            return found
                else:
                    # Prüfe ob dieses Item ein Array-Item ist (nicht eine Gruppe)
                    item_type = parent_item.data(0, Qt.UserRole + 1)
                    item_array_id = parent_item.data(0, Qt.UserRole)
                    item_text = parent_item.text(0)
                    
                    # Konvertiere item_array_id zu String für Vergleich
                    item_array_id_str = str(item_array_id) if item_array_id is not None else None
                    
                    # Prüfe ob es ein Array-Item ist (nicht eine Gruppe)
                    # Gruppen haben item_type == "group", Arrays haben item_type == None oder einen anderen Wert
                    if item_type != "group" and item_array_id_str == array_id_str:
                        return parent_item
                    
                    # Suche rekursiv in Children (auch wenn es eine Gruppe ist)
                    child_count = parent_item.childCount()
                    if child_count > 0:
                        for i in range(child_count):
                            child = parent_item.child(i)
                            if child:
                                found = find_array_item(child, depth + 1)
                                if found:
                                    return found
                
                return None
            
            item = find_array_item()
            if item is not None:
                # Blockiere Signale während der programmatischen Auswahl
                tree_widget.blockSignals(True)
                try:
                    # Expandiere Parent-Items (falls Item in einer Gruppe ist)
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                    
                    # Setze sowohl currentItem als auch Selection, damit itemSelectionChanged ausgelöst wird
                    tree_widget.clearSelection()
                    tree_widget.setCurrentItem(item)
                    item.setSelected(True)
                    tree_widget.scrollToItem(item)
                finally:
                    tree_widget.blockSignals(False)
                
                # OPTIMIERUNG: Manuell Widgets aktualisieren, da setCurrentItem nicht immer itemSelectionChanged auslöst
                # Rufe show_sources_tab() direkt auf, um sicherzustellen, dass alle Widgets aktualisiert werden
                if hasattr(sources_instance, 'show_sources_tab'):
                    sources_instance.show_sources_tab()
                
                # Setze Highlight-IDs für rote Umrandung ARRAY-BASIERT:
                # Wenn ein einzelner Lautsprecher angeklickt wird, soll das komplette
                # zugehörige Array hervorgehoben werden (Stack und Flown).
                setattr(self.settings, "active_speaker_array_highlight_id", array_id)
                setattr(self.settings, "active_speaker_array_highlight_ids", [str(array_id)])
                # Leere die per-Speaker-Highlight-Liste, damit _update_speaker_highlights
                # alle Lautsprecher des Arrays rot zeichnet.
                setattr(self.settings, "active_speaker_highlight_indices", [])
                
                # Aktualisiere Overlays für rote Umrandung
                if hasattr(main_window, "draw_plots") and hasattr(main_window.draw_plots, "draw_spl_plotter"):
                    draw_spl = main_window.draw_plots.draw_spl_plotter
                    if hasattr(draw_spl, "update_overlays"):
                        draw_spl.update_overlays(self.settings, self.container)
            else:
                # Nichts zu tun, wenn kein Item gefunden wurde
                pass
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    
    def _get_z_from_mesh(self, x_pos: float, y_pos: float) -> Optional[float]:
        """Holt Z-Koordinate direkt aus Mesh-Punkten basierend auf X/Y-Position (schnell, ohne Picking)"""
        try:
            if self.surface_mesh is None:
                return None
            
            mesh_points = self.surface_mesh.points
            if len(mesh_points) == 0:
                return None
            
            # Finde nächste Punkte im Mesh für X/Y-Position (ignoriere Z)
            xy_distances = np.linalg.norm(mesh_points[:, :2] - np.array([x_pos, y_pos]), axis=1)
            
            # Nimm die 4 nächsten Punkte für Interpolation
            closest_indices = np.argsort(xy_distances)[:4]
            closest_distances = xy_distances[closest_indices]
            
            # Wenn der nächste Punkt sehr nah ist, verwende direkt dessen Z-Wert
            if closest_distances[0] < 0.01:  # Sehr nah (< 1cm)
                return float(mesh_points[closest_indices[0], 2])
            
            # Interpoliere Z-Wert basierend auf gewichteter Distanz
            # Verwende inverse Distanz-gewichtete Interpolation
            weights = 1.0 / (closest_distances + 1e-6)  # Vermeide Division durch Null
            weights = weights / np.sum(weights)  # Normalisiere Gewichte
            
            z_values = mesh_points[closest_indices, 2]
            z_interpolated = np.sum(weights * z_values)
            
            return float(z_interpolated)
        except Exception:
            return None
    
    def _handle_mouse_move_3d(self, mouse_pos: QtCore.QPoint) -> None:
        """Behandelt Mouse-Move im 3D-Plot und zeigt Mausposition an"""
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
            renderer.SetDisplayPoint(x_norm * renderer.GetSize()[0], y_norm * renderer.GetSize()[1], 0.0)
            renderer.DisplayToWorld()
            world_point_near = renderer.GetWorldPoint()
            
            renderer.SetDisplayPoint(x_norm * renderer.GetSize()[0], y_norm * renderer.GetSize()[1], 1.0)
            renderer.DisplayToWorld()
            world_point_far = renderer.GetWorldPoint()
            
            if world_point_near is None or world_point_far is None or len(world_point_near) < 4:
                return
            
            # Berechne Schnittpunkt mit Z=0 Ebene (schnell für X/Y)
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
            
            # Hole SPL-Wert aus calculation_spl
            spl_value = None
            if self.container:
                spl_value = self._get_spl_from_3d_data(x_pos, y_pos)
            
            # Baue Text zusammen (ohne Z-Wert)
            text = f"3D-Plot:\nX: {x_pos:.2f} m\nY: {y_pos:.2f} m"
            
            # Füge SPL hinzu falls verfügbar
            if spl_value is not None:
                text += f"\nSPL: {spl_value:.1f} dB"
            
            # Aktualisiere Label über main_window
            if self.main_window and hasattr(self.main_window, 'ui') and hasattr(self.main_window.ui, 'mouse_position_label'):
                self.main_window.ui.mouse_position_label.setText(text)
        except Exception as e:
            # Fehler ignorieren (nicht kritisch)
            pass

    def _get_spl_from_3d_data(self, x_pos, y_pos):
        """Holt SPL-Wert aus 3D-Daten durch Interpolation"""
        try:
            if not self.container:
                return None
            
            calculation_spl = getattr(self.container, 'calculation_spl', {})
            if not calculation_spl:
                return None
            
            # Hole Daten aus calculation_spl
            sound_field_x = calculation_spl.get('sound_field_x')
            sound_field_y = calculation_spl.get('sound_field_y')
            sound_field_p = calculation_spl.get('sound_field_p')
            
            if not all([sound_field_x, sound_field_y, sound_field_p]):
                return None
            
            x_array = np.array(sound_field_x)
            y_array = np.array(sound_field_y)
            p_array = np.array(sound_field_p)
            
            # Prüfe ob Position im gültigen Bereich liegt
            if (x_array[0] <= x_pos <= x_array[-1] and 
                y_array[0] <= y_pos <= y_array[-1]):
                
                # Interpoliere 2D
                from scipy.interpolate import griddata
                
                # Erstelle Grid
                X, Y = np.meshgrid(x_array, y_array)
                points = np.column_stack([X.ravel(), Y.ravel()])
                values = p_array.ravel()
                
                # Entferne NaN-Werte
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    return None
                
                points_clean = points[valid_mask]
                values_clean = values[valid_mask]
                
                # Interpoliere
                spl_value = griddata(points_clean, values_clean, (x_pos, y_pos), method='linear')
                
                if not np.isnan(spl_value):
                    # Konvertiere zu dB
                    spl_db = self.functions.mag2db(np.abs(spl_value))
                    return float(spl_db)
            
            return None
        except Exception as e:
            return None

    def _get_z_and_spl_data_from_container(self, x_pos, y_pos):
        """Holt Z-Daten und SPL-Daten aus dem Container für die angegebene Position"""
        try:
            if not self.container:
                return None, None
            
            calculation_spl = getattr(self.container, 'calculation_spl', {})
            if not calculation_spl:
                return None, None
            
            # Hole Daten aus calculation_spl
            sound_field_x = calculation_spl.get('sound_field_x')
            sound_field_y = calculation_spl.get('sound_field_y')
            sound_field_z = calculation_spl.get('sound_field_z')
            sound_field_p = calculation_spl.get('sound_field_p')
            
            if not all([sound_field_x, sound_field_y, sound_field_p]):
                return None, None
            
            x_array = np.array(sound_field_x)
            y_array = np.array(sound_field_y)
            p_array = np.array(sound_field_p)
            
            # Prüfe ob Position im gültigen Bereich liegt
            if not (x_array[0] <= x_pos <= x_array[-1] and 
                    y_array[0] <= y_pos <= y_array[-1]):
                return None, None
            
            # Finde die nächsten Indizes
            x_idx = np.argmin(np.abs(x_array - x_pos))
            y_idx = np.argmin(np.abs(y_array - y_pos))
            
            # Hole Z-Daten falls vorhanden
            z_data = None
            if sound_field_z is not None:
                z_array = np.array(sound_field_z)
                if z_array.ndim == 2 and z_array.shape[0] > y_idx and z_array.shape[1] > x_idx:
                    z_data = float(z_array[y_idx, x_idx])
                elif z_array.size == p_array.size:
                    # Z-Daten als 1D-Array, reshape wenn möglich
                    try:
                        z_reshaped = z_array.reshape(p_array.shape)
                        z_data = float(z_reshaped[y_idx, x_idx] if p_array.ndim == 2 else z_reshaped[y_idx * len(x_array) + x_idx])
                    except:
                        pass
            
            # Hole SPL-Daten
            spl_data = None
            if p_array.ndim == 2 and p_array.shape[0] > y_idx and p_array.shape[1] > x_idx:
                p_value = p_array[y_idx, x_idx]
                if not np.isnan(p_value):
                    spl_data = float(self.functions.mag2db(np.abs(p_value)))
            elif p_array.ndim == 1:
                idx = y_idx * len(x_array) + x_idx
                if idx < len(p_array):
                    p_value = p_array[idx]
                    if not np.isnan(p_value):
                        spl_data = float(self.functions.mag2db(np.abs(p_value)))
            
            return z_data, spl_data
        except Exception as e:
            return None, None

    def _pick_surface_at_position(self, click_pos: QtCore.QPoint) -> Tuple[Optional[float], Optional[str]]:
        """Findet die Surface am Klickpunkt und gibt die Distanz zur Kamera zurück.
        
        Args:
            click_pos: Klickposition im Widget
            
        Returns:
            Tuple[Optional[float], Optional[str]]: (distanz_zur_kamera, surface_id) oder (None, None)
        """
        try:
            # Hole 3D-Koordinaten des geklickten Punktes über VTK CellPicker
            renderer = self.plotter.renderer
            if renderer is None:
                return

            # Ray-Casting für Surface-Erkennung (funktioniert auch ohne SPL-Daten/surface_mesh)
            # Das erlaubt echtes 3D‑Picking auch für schräge/vertikale Flächen.
            # Hinweis: Das Ray-Casting benötigt self.surface_mesh nicht - es verwendet nur surface_definitions
            try:
                # Hilfsfunktion: Ray-Schnitt mit planarem Modell der Surface.
                def _intersect_ray_with_surface_plane(
                    ray_start: np.ndarray,
                    ray_dir: np.ndarray,
                    plane_model: dict | None,
                    fallback_z: float,
                ) -> tuple[float, float, float] | None:
                    """
                    Berechnet den Schnittpunkt eines Rays mit der Fläche:
                    - nutzt das planare Modell (mode: constant, x, y, xy)
                    - fällt bei fehlendem Modell auf eine konstante Z-Ebene (fallback_z) zurück
                    Gibt (x, y) des Schnittpunkts zurück oder None bei Parallelität / Fehler.
                    """
                    # Sicherstellen, dass wir keine Division durch 0 haben
                    if plane_model is None:
                        # Konstante Ebene z = fallback_z
                        denom = ray_dir[2]
                        if abs(denom) < 1e-8:
                            return None
                        t = (fallback_z - ray_start[2]) / denom
                        if t <= 0:
                            return None
                        p = ray_start + t * ray_dir
                        return float(p[0]), float(p[1]), float(t)
                    
                    mode = plane_model.get("mode", "constant")
                    # Allgemeine Form: z = f(x, y)
                    if mode == "constant":
                        z_plane = float(plane_model.get("base", plane_model.get("intercept", fallback_z)))
                        denom = ray_dir[2]
                        if abs(denom) < 1e-8:
                            return None
                        t = (z_plane - ray_start[2]) / denom
                    elif mode == "x":
                        slope = float(plane_model.get("slope", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        # z0 + t*dz = slope*(x0 + t*dx) + intercept
                        # (dz - slope*dx)*t + (z0 - slope*x0 - intercept) = 0
                        denom = ray_dir[2] - slope * ray_dir[0]
                        num = ray_start[2] - slope * ray_start[0] - intercept
                        if abs(denom) < 1e-8:
                            return None
                        t = -num / denom
                    elif mode == "y":
                        slope = float(plane_model.get("slope", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        # z0 + t*dz = slope*(y0 + t*dy) + intercept
                        denom = ray_dir[2] - slope * ray_dir[1]
                        num = ray_start[2] - slope * ray_start[1] - intercept
                        if abs(denom) < 1e-8:
                            return None
                        t = -num / denom
                    elif mode == "xy":
                        slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                        slope_y = float(plane_model.get("slope_y", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        # z0 + t*dz = sx*(x0 + t*dx) + sy*(y0 + t*dy) + c
                        denom = ray_dir[2] - slope_x * ray_dir[0] - slope_y * ray_dir[1]
                        num = ray_start[2] - slope_x * ray_start[0] - slope_y * ray_start[1] - intercept
                        if abs(denom) < 1e-8:
                            return None
                        t = -num / denom
                    else:
                        # Unbekannter Modus – fallback auf konstante Ebene
                        denom = ray_dir[2]
                        if abs(denom) < 1e-8:
                            return None
                        t = (fallback_z - ray_start[2]) / denom
                    
                    if t <= 0:
                        return None
                    p = ray_start + t * ray_dir
                    return float(p[0]), float(p[1]), float(t)

                from vtkmodules.vtkRenderingCore import vtkCellPicker
                from vtkmodules.util.numpy_support import vtk_to_numpy
                
                picker = vtkCellPicker()
                picker.SetTolerance(0.01)
                
                size = self.widget.size()
                if size.width() > 0 and size.height() > 0:
                    x_norm = click_pos.x() / size.width()
                    y_norm = 1.0 - (click_pos.y() / size.height())  # VTK Y ist invertiert
                    
                    # Berechne den 3D-Ray durch den Klickpunkt (Near/Far auf dem Viewport)
                    render_size = renderer.GetSize()
                    if render_size[0] <= 0 or render_size[1] <= 0:
                        return
                    viewport_width = render_size[0]
                    viewport_height = render_size[1]
                    viewport_x = x_norm * viewport_width
                    viewport_y = y_norm * viewport_height

                    renderer.SetDisplayPoint(viewport_x, viewport_y, 0.0)
                    renderer.DisplayToWorld()
                    world_point_near = renderer.GetWorldPoint()

                    renderer.SetDisplayPoint(viewport_x, viewport_y, 1.0)
                    renderer.DisplayToWorld()
                    world_point_far = renderer.GetWorldPoint()

                    if world_point_near is None or world_point_far is None or len(world_point_near) < 4:
                        return
                    if abs(world_point_near[3]) <= 1e-6 or abs(world_point_far[3]) <= 1e-6:
                        return

                    ray_start = np.array([
                        world_point_near[0] / world_point_near[3],
                        world_point_near[1] / world_point_near[3],
                        world_point_near[2] / world_point_near[3],
                    ])
                    ray_end = np.array([
                        world_point_far[0] / world_point_far[3],
                        world_point_far[1] / world_point_far[3],
                        world_point_far[2] / world_point_far[3],
                    ])
                    ray_dir = ray_end - ray_start
                    ray_len = np.linalg.norm(ray_dir)
                    if ray_len <= 1e-8:
                        return
                    ray_dir = ray_dir / ray_len
                else:
                    return
            except ImportError:
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                return
            
            # Prüfe welche Surface (enabled oder disabled) den Ray schneidet
            surface_definitions = getattr(self.settings, 'surface_definitions', {})
            if not isinstance(surface_definitions, dict):
                return (None, None)
            
            # Durchsuche alle Surfaces (enabled und disabled)
            checked_count = 0
            skipped_hidden = 0
            skipped_too_few_points = 0
            
            # Prüfe alle Surfaces (enabled und disabled) gleichberechtigt, sortiert nur nach Distanz zur Kamera
            best_surface_id: str | None = None
            best_t: float = float("inf")

            # Prüfe alle Surfaces in einer Schleife (enabled und disabled gleichberechtigt)
            for surface_id, surface_def in surface_definitions.items():
                # Normalisiere Surface-Definition
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, 'enabled', False))
                    hidden = bool(getattr(surface_def, 'hidden', False))
                    points = getattr(surface_def, 'points', []) or []
                    plane_model = getattr(surface_def, 'plane_model', None)
                else:
                    enabled = surface_def.get('enabled', False)
                    hidden = surface_def.get('hidden', False)
                    points = surface_def.get('points', [])
                    plane_model = surface_def.get('plane_model')
                
                if hidden:
                    skipped_hidden += 1
                    continue
                if len(points) < 3:
                    skipped_too_few_points += 1
                    continue
                
                # Prüfe sowohl enabled als auch disabled Surfaces (keine Priorität)
                checked_count += 1
                
                # Fallback-Z für konstante Ebene aus erstem Punkt
                fallback_z = float(points[0].get("z", 0.0)) if points else 0.0
                
                # Extrahiere Koordinaten für Analyse
                surface_xs = [p.get('x', 0.0) for p in points]
                surface_ys = [p.get('y', 0.0) for p in points]
                surface_zs = [p.get('z', 0.0) for p in points]
                
                if not surface_xs or not surface_ys:
                    continue
                
                # Prüfe ob Fläche senkrecht ist (analog zu _render_surfaces_textured)
                poly_x = np.array(surface_xs, dtype=float)
                poly_y = np.array(surface_ys, dtype=float)
                poly_z = np.array(surface_zs, dtype=float)
                
                x_span = float(np.ptp(poly_x)) if poly_x.size > 0 else 0.0
                y_span = float(np.ptp(poly_y)) if poly_y.size > 0 else 0.0
                z_span = float(np.ptp(poly_z)) if poly_z.size > 0 else 0.0
                
                eps_line = 1e-6
                has_height = z_span > 1e-3
                is_vertical = False
                wall_axis = None
                wall_value = None
                
                # Prüfe ob senkrechte Fläche (X-Z-Wand oder Y-Z-Wand)
                if y_span < eps_line and x_span >= eps_line and has_height:
                    # X-Z-Wand: y ≈ const
                    is_vertical = True
                    wall_axis = "y"
                    wall_value = float(np.mean(poly_y))
                elif x_span < eps_line and y_span >= eps_line and has_height:
                    # Y-Z-Wand: x ≈ const
                    is_vertical = True
                    wall_axis = "x"
                    wall_value = float(np.mean(poly_x))
                
                # Für senkrechte Flächen: Ray-Casting mit der Fläche direkt
                if is_vertical:
                    # Berechne Schnittpunkt des Rays mit der senkrechten Fläche
                    if wall_axis == "y":
                        # X-Z-Wand: y = wall_value
                        # Ray: ray_start + t * ray_dir
                        # y = ray_start[1] + t * ray_dir[1] = wall_value
                        if abs(ray_dir[1]) < 1e-8:
                            continue  # Ray parallel zur Wand
                        t_hit = (wall_value - ray_start[1]) / ray_dir[1]
                        if t_hit <= 0:
                            continue
                        hit_point = ray_start + ray_dir * t_hit
                        x_click = hit_point[0]
                        y_click = wall_value
                        z_click = hit_point[2]
                        
                        # Prüfe ob Punkt in X-Z-Projektion der Fläche liegt
                        min_x, max_x = float(poly_x.min()), float(poly_x.max())
                        min_z, max_z = float(poly_z.min()), float(poly_z.max())
                        if x_click < min_x or x_click > max_x or z_click < min_z or z_click > max_z:
                            continue
                        
                        # Prüfe ob Punkt in Polygon liegt (X-Z-Projektion)
                        points_xz = [{"x": p.get("x", 0.0), "y": p.get("z", 0.0)} for p in points]
                        is_inside = self._point_in_polygon(x_click, z_click, points_xz)
                    else:  # wall_axis == "x"
                        # Y-Z-Wand: x = wall_value
                        if abs(ray_dir[0]) < 1e-8:
                            continue  # Ray parallel zur Wand
                        t_hit = (wall_value - ray_start[0]) / ray_dir[0]
                        if t_hit <= 0:
                            continue
                        hit_point = ray_start + ray_dir * t_hit
                        x_click = wall_value
                        y_click = hit_point[1]
                        z_click = hit_point[2]
                        
                        # Prüfe ob Punkt in Y-Z-Projektion der Fläche liegt
                        min_y, max_y = float(poly_y.min()), float(poly_y.max())
                        min_z, max_z = float(poly_z.min()), float(poly_z.max())
                        if y_click < min_y or y_click > max_y or z_click < min_z or z_click > max_z:
                            continue
                        
                        # Prüfe ob Punkt in Polygon liegt (Y-Z-Projektion)
                        points_yz = [{"x": p.get("y", 0.0), "y": p.get("z", 0.0)} for p in points]
                        is_inside = self._point_in_polygon(y_click, z_click, points_yz)
                else:
                    # Planare Fläche: verwende plane_model
                    # Berechne plane_model dynamisch, falls nicht vorhanden
                    if plane_model is None:
                        plane_model, _ = derive_surface_plane(points)
                        # Wenn plane_model immer noch None ist, verwende konstante Ebene
                        if plane_model is None:
                            plane_model = {"mode": "constant", "base": fallback_z, "intercept": fallback_z}

                    # Berechne Schnittpunkt des Rays mit der Ebene dieser Surface
                    intersect = _intersect_ray_with_surface_plane(ray_start, ray_dir, plane_model, fallback_z)
                    if intersect is None:
                        continue
                    x_click, y_click, t_hit = intersect

                    # Berechne Bounding Box der Surface für schnelle Vorprüfung
                    min_x, max_x = min(surface_xs), max(surface_xs)
                    min_y, max_y = min(surface_ys), max(surface_ys)
                    
                    # Schnelle Bounding-Box-Prüfung zuerst
                    if x_click < min_x or x_click > max_x or y_click < min_y or y_click > max_y:
                        continue
                    
                    # Prüfe ob Punkt in diesem Polygon liegt (nur X/Y, Z wird ignoriert)
                    # Für schräge Flächen wird der Ray bereits mit der Ebene geschnitten,
                    # daher ist die XY-Prüfung ausreichend
                    is_inside = self._point_in_polygon(x_click, y_click, points)
                    t_hit = t_hit  # t_hit wurde bereits von _intersect_ray_with_surface_plane berechnet
                
                # Wähle die am nächsten zur Kamera liegende Surface (unabhängig von enabled/disabled)
                if is_inside and t_hit < best_t:
                    best_t = t_hit
                    best_surface_id = str(surface_id)
            
            # Gebe die am nächsten zur Kamera liegende Surface zurück, falls vorhanden
            if best_surface_id is not None:
                return (best_t, best_surface_id)
            else:
                return (None, None)
            
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return (None, None)
    
    def _handle_spl_surface_click(self, click_pos: QtCore.QPoint) -> None:
        """Behandelt einen Klick auf die SPL-Surface (Wrapper für Kompatibilität).
        
        Diese Methode wird von älteren Code-Stellen aufgerufen und ruft intern
        _pick_surface_at_position auf.
        """
        t, surface_id = self._pick_surface_at_position(click_pos)
        if surface_id is not None:
            self._select_surface_in_treewidget(surface_id)
    
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
    
    def _handle_double_click_auto_range(self) -> None:
        """Behandelt Doppelklick auf Colorbar: Findet den höchsten SPL und passt die Colorbar-Range an."""
        try:
            # Prüfe ob Settings verfügbar sind
            if not self.settings:
                return
            
            # Prüfe ob main_window verfügbar ist (für Plot-Update und Container-Zugriff)
            if not self.main_window:
                return
            
            # Hole Container (entweder direkt oder über main_window)
            container = self.container
            if container is None and hasattr(self.main_window, 'container'):
                container = self.main_window.container
            
            if container is None:
                return
            
            # Hole die aktuellen Schallfeld-Daten
            calc_spl = getattr(container, 'calculation_spl', {})
            if not isinstance(calc_spl, dict):
                return
            
            # Prüfe Plot-Modus
            plot_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
            time_mode = plot_mode == 'SPL over time'
            phase_mode = plot_mode == 'Phase alignment'
            
            # Für Phase-Mode und Time-Mode keine automatische Anpassung
            if phase_mode or time_mode:
                return
            
            # Hole Schallfeld-Daten
            sound_field_p = calc_spl.get('sound_field_p')
            if sound_field_p is None:
                return
            
            # Konvertiere zu numpy array
            pressure_array = np.array(sound_field_p, dtype=float)
            
            # Prüfe ob Daten vorhanden sind
            if pressure_array.size == 0:
                return
            
            # Entferne NaN und Inf Werte
            finite_mask = np.isfinite(pressure_array)
            if not np.any(finite_mask):
                return
            
            # Berechne SPL in dB
            pressure_abs = np.abs(pressure_array)
            pressure_abs = np.clip(pressure_abs, 1e-12, None)
            spl_db = self.functions.mag2db(pressure_abs)
            
            # Finde minimalen und maximalen SPL (nur finite Werte)
            spl_min = float(np.nanmin(spl_db[finite_mask]))
            spl_max = float(np.nanmax(spl_db[finite_mask]))
            
            # Runde auf nächste ganze Zahl (für bessere Lesbarkeit)
            spl_max_rounded = np.ceil(spl_max)
            spl_min_rounded = np.floor(spl_min)
            
            # Hole aktuelle Colorbar-Range
            colorbar_range = getattr(self.settings, 'colorbar_range', {})
            if not isinstance(colorbar_range, dict):
                colorbar_range = {}
            
            # Hole Color-Step
            step = colorbar_range.get('step', 3.0)
            if step <= 0:
                step = 3.0
            
            # 🎯 LOGIK IDENTISCH ZUR SETTINGS UI: Verwende 11 Farben (num_colors = 11)
            # Wie in on_max_spl_changed: new_min = value - (step * (num_colors - 1))
            # Wie in on_min_spl_changed: new_max = value + (step * (num_colors - 1))
            num_colors = 11  # Identisch zur Settings UI
            
            # Berechne Max als Vielfaches von step (aufgerundet)
            new_max = np.ceil(spl_max_rounded / step) * step
            
            # Berechne Min basierend auf Max und num_colors (wie in Settings UI)
            # Min = Max - (step * (num_colors - 1))
            new_min = new_max - (step * (num_colors - 1))
            
            # Stelle sicher, dass Min nicht kleiner als der tatsächliche Min ist
            # Aber runde auf Vielfaches von step ab
            actual_min_rounded = np.floor(spl_min_rounded / step) * step
            
            # Wenn berechnetes Min kleiner als tatsächlicher Min ist, passe Max an
            if new_min < actual_min_rounded:
                # Rechne Max neu, damit Min = actual_min_rounded
                new_max = actual_min_rounded + (step * (num_colors - 1))
                new_min = actual_min_rounded
            
            # Stelle sicher, dass Min nicht größer oder gleich Max ist
            if new_min >= new_max:
                new_min = new_max - step
            
            # Stelle sicher, dass Min nicht negativ wird
            if new_min < 0:
                new_min = 0.0
                # Rechne Max neu, damit die Anzahl der Farben stimmt
                new_max = new_min + (step * (num_colors - 1))
            
            # Stelle sicher, dass Max nicht größer als 150 ist (wie in Settings UI)
            if new_max > 150:
                new_max = 150.0
                # Rechne Min neu
                new_min = new_max - (step * (num_colors - 1))
                if new_min < 0:
                    new_min = 0.0
            
            # Hole alte Werte für Vergleich
            old_min = colorbar_range.get('min', 90.0)
            old_max = colorbar_range.get('max', 120.0)
            
            # Nur aktualisieren wenn sich die Werte signifikant geändert haben (mindestens 1 dB)
            if abs(new_max - old_max) < 1.0 and abs(new_min - old_min) < 1.0:
                return
            
            # Aktualisiere Colorbar-Range
            colorbar_range['min'] = float(new_min)
            colorbar_range['max'] = float(new_max)
            
            # Stelle sicher, dass step und tick_step sinnvoll sind
            tick_step = colorbar_range.get('tick_step', 6.0)
            
            # Setze colorbar_range im Settings
            setattr(self.settings, 'colorbar_range', colorbar_range)
            
            # 🎯 AKTUALISIERE SETTINGS UI EINGABEFELDER
            if hasattr(self.main_window, 'ui_settings') and self.main_window.ui_settings is not None:
                ui_settings = self.main_window.ui_settings
                # Aktualisiere die Eingabefelder (ohne die Signal-Handler auszulösen)
                if hasattr(ui_settings, 'min_spl'):
                    ui_settings.min_spl.blockSignals(True)
                    ui_settings.min_spl.setText(str(int(new_min)))
                    ui_settings.min_spl.blockSignals(False)
                if hasattr(ui_settings, 'max_spl'):
                    ui_settings.max_spl.blockSignals(True)
                    ui_settings.max_spl.setText(str(int(new_max)))
                    ui_settings.max_spl.blockSignals(False)
                if hasattr(ui_settings, 'db_step'):
                    ui_settings.db_step.blockSignals(True)
                    ui_settings.db_step.setText(str(int(step)))
                    ui_settings.db_step.blockSignals(False)
            
            # Aktualisiere Plot über main_window
            if hasattr(self.main_window, 'plot_spl'):
                self.main_window.plot_spl(update_axes=False, reset_camera=False)
            
        except Exception as e:  # noqa: BLE001
            # Fehler ignorieren (nicht kritisch)
            import traceback
            traceback.print_exc()
    
    # Doppelklick-Handler wird jetzt vom ColorbarManager verwaltet
    
    def _select_surface_in_treewidget(self, surface_id: str) -> None:
        """Wählt eine Surface im TreeWidget aus und öffnet die zugehörige Gruppe."""
        try:
            # Finde das Main-Window über parent_widget
            # parent_widget ist self.main_window.ui.SPLPlot (ein QWidget)
            # Wir müssen durch die Widget-Hierarchie gehen, um das MainWindow zu finden
            parent = self.parent_widget
            main_window = None
            
            # Gehe durch die Widget-Hierarchie, um das Main-Window zu finden
            depth = 0
            while parent is not None and depth < 15:
                parent_type = type(parent).__name__
                has_surface_manager = hasattr(parent, 'surface_manager')
                
                if has_surface_manager:
                    main_window = parent
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
                # Versuche alternativen Weg: Suche nach QMainWindow
                parent = self.parent_widget
                while parent is not None:
                    if isinstance(parent, QtWidgets.QMainWindow):
                        if hasattr(parent, 'surface_manager'):
                            main_window = parent
                            break
                    if hasattr(parent, 'parent'):
                        parent = parent.parent()
                    elif hasattr(parent, 'parentWidget'):
                        parent = parent.parentWidget()
                    else:
                        break
                
                if main_window is None:
                    return
            
            # Wechsle zum Surface Widget (falls im Sources Widget)
            if hasattr(main_window, 'sources_instance') and hasattr(main_window, 'surface_manager'):
                surface_dock = getattr(main_window.surface_manager, 'surface_dockWidget', None)
                if surface_dock:
                    surface_dock.raise_()  # Zeige Surface Widget
            
            # Hole das SurfaceDockWidget über surface_manager
            surface_manager = main_window.surface_manager
            if not hasattr(surface_manager, 'surface_tree_widget'):
                return
            
            # UISurfaceManager verwendet direkt ein QTreeWidget, nicht WindowSurfaceWidget
            # Verwende _select_surface_in_tree Methode
            if hasattr(surface_manager, '_select_surface_in_tree'):
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
                        # Finde Parent-Gruppe und expandiere sie
                        parent = item.parent()
                        while parent is not None:
                            item_type = parent.data(0, Qt.UserRole + 1)
                            if item_type == "group":
                                parent.setExpanded(True)
                                # Scrolle zur Gruppe, damit sie sichtbar ist
                                tree_widget.scrollToItem(parent, QtWidgets.QAbstractItemView.PositionAtTop)
                                break
                            parent = parent.parent()
                        
                        # Scrolle zur Surface, damit sie sichtbar ist
                        tree_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
                    else:
                        pass
            else:
                pass
                    
        except Exception as e:  # noqa: BLE001
            # Bei Fehler einfach ignorieren
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
    def initialize_empty_spl_plot(self, preserve_camera: bool = True):
        """Initialisiert nur den SPL-Plot (Fläche/Floor), Lautsprecher-/Overlays bleiben erhalten."""
        if preserve_camera:
            camera_state = self._camera_state or self._capture_camera()
        else:
            camera_state = None
            # Wenn Kamera nicht erhalten wird, Flag zurücksetzen, damit beim nächsten Plot mit Daten Zoom maximiert wird
            self._has_plotted_data = False

        # Sprecher-Arrays-Länge zur Laufzeit ermitteln
        try:
            _sa = getattr(self.settings, 'speaker_arrays', {})
            _sa_len = len(_sa) if isinstance(_sa, dict) else 0
        except Exception:
            _sa_len = 0

        # Zwei Modi:
        # - Keine Speaker-Arrays → komplette Szene inkl. Overlays leeren (Startzustand)
        # - Speaker-Arrays vorhanden → nur SPL-bezogene Actor löschen, Lautsprecher/Overlays behalten
        if _sa_len == 0:
            # Ursprüngliches Verhalten: komplette Szene leeren
            if hasattr(self, "plotter") and self.plotter is not None:
                self.plotter.clear()
            if hasattr(self, "overlay_helper") and self.overlay_helper is not None:
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
        else:
            # Nur SPL-bezogene Actor entfernen, Lautsprecher/Overlays unangetastet lassen
            if hasattr(self, "plotter") and self.plotter is not None:
                try:
                    renderer = self.plotter.renderer
                except Exception:
                    renderer = None
                if renderer is not None:
                    # Haupt-SPL-Surface (Mesh)
                    base_actor = renderer.actors.get(self.SURFACE_NAME)
                    if base_actor is not None:
                        try:
                            self.plotter.remove_actor(base_actor)
                        except Exception:
                            pass
                    # Floor-Actor
                    floor_actor = renderer.actors.get(self.FLOOR_NAME)
                    if floor_actor is not None:
                        try:
                            self.plotter.remove_actor(floor_actor)
                        except Exception:
                            pass
                # Horizontale Surface-Mesh-Actors
                for sid, actor in list(getattr(self, "_surface_actors", {}).items()):
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
                    self._surface_actors.pop(sid, None)
                # Texture-Actors für horizontale Surfaces
                for tex_data in list(getattr(self, "_surface_texture_actors", {}).values()):
                    actor = None
                    if isinstance(tex_data, dict):
                        actor = tex_data.get("actor")
                    elif tex_data is not None:
                        actor = tex_data
                    if actor is not None:
                        try:
                            self.plotter.remove_actor(actor)
                        except Exception:
                            pass
                if hasattr(self, "_surface_texture_actors"):
                    self._surface_texture_actors.clear()
                # 🚀 TEXTUR-CACHE: Auch Cache löschen
                if hasattr(self, "_surface_texture_cache"):
                    self._surface_texture_cache.clear()

                # Vertikale SPL-Flächen entfernen
                if hasattr(self, "_clear_vertical_spl_surfaces"):
                    try:
                        self._clear_vertical_spl_surfaces()
                    except Exception:
                        pass

            # SPL-interne Zustände zurücksetzen
            self.has_data = False
            self.surface_mesh = None
            if hasattr(self, '_surface_signature_cache'):
                self._surface_signature_cache.clear()
        
        # Konfiguriere Plotter nur bei Bedarf (nicht die Kamera überschreiben)
        self._configure_plotter(configure_camera=not preserve_camera)

        if camera_state is not None:
            self._restore_camera(camera_state)
        else:
            self.set_view_top()
            camera_state = self._camera_state
            if camera_state is not None:
                self._restore_camera(camera_state)

        # Initialisiere leere Colorbar über ColorbarManager
        colorization_mode = getattr(self.settings, 'colorization_mode', 'Gradient')
        if colorization_mode not in {'Color step', 'Gradient'}:
            colorization_mode = 'Color step'
        self.colorbar_manager.render_colorbar(colorization_mode, force=True)
        self.cbar = self.colorbar_manager.cbar  # Synchronisiere cbar-Referenz
        self._skip_next_render_restore = True
        self.render()
        self._skip_next_render_restore = False
        self._save_camera_state()
        if self.time_control is not None:
            self.time_control.hide()

    def initialize_empty_scene(self, preserve_camera: bool = True):
        """Abwärtskompatibler Wrapper – bitte initialize_empty_spl_plot verwenden."""
        return self.initialize_empty_spl_plot(preserve_camera=preserve_camera)

    # update_spl_plot ist jetzt in SPL3DPlotRenderer (Mixin)
        
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

    def update_overlays(self, settings, container):
        """Aktualisiert Zusatzobjekte (Achsen, Lautsprecher, Messpunkte)."""
        t_start = time.perf_counter()
        
        # Speichere Container-Referenz für Z-Koordinaten-Zugriff
        with perf_section("PlotSPL3D.update_overlays.setup"):
            self.container = container
            
            # Wenn kein gültiger calculation_spl-Container vorhanden ist, entferne vertikale SPL-Flächen
            if not hasattr(container, "calculation_spl"):
                self._clear_vertical_spl_surfaces()
        
        # Wir vergleichen Hash-Signaturen pro Kategorie und zeichnen nur dort neu,
        # wo sich Parameter geändert haben. Das verhindert unnötiges Entfernen
        # und erneutes Hinzufügen zahlreicher PyVista-Actor.
        with perf_section("PlotSPL3D.update_overlays.compute_signatures"):
            signatures = self._compute_overlay_signatures(settings, container)
            previous = self._last_overlay_signatures or {}
            if not previous:
                categories_to_refresh = set(signatures.keys())
            else:
                categories_to_refresh = {
                    key for key, value in signatures.items() 
                    if key != 'speakers_highlights' and value != previous.get(key)
                }
        
        # 🚀 OPTIMIERUNG: Prüfe ob sich nur Highlights geändert haben
        highlight_changed = False
        if previous:
            prev_highlights = previous.get('speakers_highlights')
            curr_highlights = signatures.get('speakers_highlights')
            if prev_highlights != curr_highlights:
                highlight_changed = True
        
        # Wenn sich nur Highlights geändert haben und Speaker bereits gezeichnet sind, nur Highlights updaten
        if highlight_changed and 'speakers' not in categories_to_refresh and self.overlay_helper._speaker_actor_cache:
            with perf_section("PlotSPL3D.update_overlays.update_speaker_highlights_only"):
                if hasattr(self.overlay_helper, '_update_speaker_highlights'):
                    self.overlay_helper._update_speaker_highlights(settings)
                    self._last_overlay_signatures = signatures
                    if not self._rotate_active and not self._pan_active:
                        self._save_camera_state()
                    return
        
        if not categories_to_refresh:
            self._last_overlay_signatures = signatures
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
            return
        
        prev_debug_state = getattr(self.overlay_helper, 'DEBUG_ON_TOP', False)
        try:
            self.overlay_helper.DEBUG_ON_TOP = True
            
            if 'axis' in categories_to_refresh:
                with perf_section("PlotSPL3D.update_overlays.draw_axis"):
                    self.overlay_helper.draw_axis_lines(settings, selected_axis=self._axis_selected)
            
            if 'surfaces' in categories_to_refresh:
                with perf_section("PlotSPL3D.update_overlays.draw_surfaces"):
                    # Prüfe ob SPL-Daten vorhanden sind, um zu entscheiden ob enabled Surfaces gezeichnet werden sollen
                    create_empty_plot_surfaces = False
                    try:
                        # Prüfe ob SPL-Daten vorhanden sind (gleiche Logik wie in draw_surfaces)
                        # WICHTIG: Die entscheidende Prüfung ist sound_field_p, nicht nur ob Texture-Actors existieren
                        has_spl_data = False
                        
                        # 🎯 ZUERST: Prüfe container.calculation_spl (entscheidende Prüfung)
                        # Dies ist die zuverlässigste Methode, um zu prüfen ob echte SPL-Daten vorhanden sind
                        if container is not None and hasattr(container, 'calculation_spl'):
                            calc_spl = container.calculation_spl
                            if isinstance(calc_spl, dict) and calc_spl.get('sound_field_p') is not None:
                                has_spl_data = True
                        
                        # Nur wenn sound_field_p nicht vorhanden ist, prüfe auf Actors
                        # (Texture-Actors können auch im leeren Plot existieren als leere Texturen)
                        if not has_spl_data:
                            if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                                # Prüfe nur auf spl_surface und spl_floor, nicht auf Texture-Actors
                                # (Texture-Actors können leer sein)
                                if spl_surface_actor is not None or spl_floor_actor is not None:
                                    has_spl_data = True
                        
                        # Wenn keine SPL-Daten vorhanden sind, zeichne enabled Surfaces für leeren Plot
                        if not has_spl_data:
                            create_empty_plot_surfaces = True
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        # Bei Fehler: konservativ, zeichne keine enabled Surfaces
                        create_empty_plot_surfaces = False
                    
                    self.overlay_helper.draw_surfaces(settings, container, create_empty_plot_surfaces=create_empty_plot_surfaces)
            
            if 'speakers' in categories_to_refresh:
                with perf_section("PlotSPL3D.update_overlays.draw_speakers"):
                    with perf_section("PlotSPL3D.update_overlays.build_cabinet_lookup"):
                        cabinet_lookup = self.overlay_helper.build_cabinet_lookup(container)
                    self.overlay_helper.draw_speakers(settings, container, cabinet_lookup)
            else:
                # Auch wenn sich die Speaker-Definitionen nicht geändert haben,
                # müssen wir die Highlights aktualisieren, falls sich die Highlight-IDs geändert haben
                with perf_section("PlotSPL3D.update_overlays.update_speaker_highlights"):
                    if hasattr(self.overlay_helper, '_update_speaker_highlights'):
                        self.overlay_helper._update_speaker_highlights(settings)
                        # Render-Update triggern, damit Änderungen sichtbar werden
                        try:
                            self.plotter.render()
                        except Exception:  # noqa: BLE001
                            pass
            
            if 'impulse' in categories_to_refresh:
                with perf_section("PlotSPL3D.update_overlays.draw_impulse"):
                    self.overlay_helper.draw_impulse_points(settings)
        finally:
            self.overlay_helper.DEBUG_ON_TOP = prev_debug_state
        
        with perf_section("PlotSPL3D.update_overlays.finalize"):
            self._last_overlay_signatures = signatures
            
            # 🎯 Beim ersten Start: Zoom auf das Default-Surface einstellen (nach dem Zeichnen aller Overlays)
            if (not getattr(self, "_did_initial_overlay_zoom", False)) and 'surfaces' in categories_to_refresh:
                self._zoom_to_default_surface()
            
            # Verzögertes Rendering: Timer starten/stoppen für Batch-Updates
            self._schedule_render()
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
        t_end = time.perf_counter()
        


    def render(self):
        """Erzwingt ein Rendering der Szene."""
        if self._is_rendering:
            return
        t_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
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
            if DEBUG_PLOT3D_TIMING:
                t_end = time.perf_counter()

    def _schedule_render(self):
        """Plant ein verzögertes Rendering (500ms), um mehrere schnelle Updates zu bündeln."""
        self._pending_render = True
        # Stoppe laufenden Timer und starte neu (debouncing)
        self._render_timer.stop()
        self._render_timer.start(500)  # 500ms Verzögerung

    def _delayed_render(self):
        """Führt das verzögerte Rendering aus, wenn der Timer abgelaufen ist."""
        if self._pending_render:
            self._pending_render = False
            self.render()

    # Vertikale SPL-Flächen sind jetzt in SPL3DPlotRenderer (Plot3DSPL.py)

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
            pass

    def _update_colorbar(self, colorization_mode: str, tick_step: float | None = None):
        try:
            self._render_colorbar(colorization_mode, tick_step=tick_step)
        except Exception as exc:  # pragma: no cover
            pass

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
            # 🎯 IMMER GENAU 10 FARBEN (11 LEVELS) VERWENDEN
            # Dies stellt sicher, dass auch nach Doppelklick auf Colorbar immer 10 Farben verwendet werden
            num_colors_fixed = 11  # 11 Levels = 10 Farben
            num_segments = num_colors_fixed - 1  # Immer 10 Segmente/Farben
            
            # Berechne Levels so, dass sie genau num_colors_fixed Levels ergeben
            # Verteile die Levels gleichmäßig zwischen min und max
            levels = np.linspace(cbar_min, cbar_max, num_colors_fixed)
            
            # Stelle sicher, dass levels nicht leer ist
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            if levels.size < 2:
                levels = np.array([cbar_min, cbar_max])
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
            
            # 🎯 Doppelklick-Handler zur Colorbar hinzufügen
            if requires_new_colorbar or not hasattr(self, '_colorbar_double_click_connected'):
                # Entferne alte Handler falls vorhanden
                if hasattr(self, '_colorbar_double_click_cid'):
                    try:
                        self.colorbar_ax.figure.canvas.mpl_disconnect(self._colorbar_double_click_cid)
                    except Exception:
                        pass
                # Füge Doppelklick-Handler hinzu
                self._colorbar_double_click_cid = self.colorbar_ax.figure.canvas.mpl_connect(
                    'button_press_event',
                    self._on_colorbar_double_click
                )
                self._colorbar_double_click_connected = True

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

    # ------------------------------------------------------------------
    # Bilineare Interpolation und Textur-Rendering für horizontale Surfaces
    # ------------------------------------------------------------------
    # Hinweis: _bilinear_interpolate_grid und _nearest_interpolate_grid sind jetzt in SPL3DPlotRenderer (Mixin)
    # Die Methoden werden durch Vererbung vom Mixin bereitgestellt.

    def _process_single_surface_texture(
        self,
        surface_id: str,
        points: list[dict[str, float]],
        surface_obj: Any,
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        cbar_min: float,
        cbar_max: float,
        base_cmap: Any,
        norm: Any,
        is_step_mode: bool,
        colorization_mode: str,
        cbar_step: float,
        tex_res_global: float,
        effective_upscale_factor: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet eine einzelne Surface und erstellt Textur + Grid.
        Diese Methode kann parallel aufgerufen werden.
        
        Returns:
            Dict mit 'surface_id', 'grid', 'texture', 'metadata', 'texture_signature', etc.
            oder None bei Fehler/Skip
        """
        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return None
        
        try:
            t_surface_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
            t_geom_done = t_surface_start
            
            poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
            poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
            poly_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
            if poly_x.size == 0 or poly_y.size == 0:
                return None

            xmin, xmax = float(poly_x.min()), float(poly_x.max())
            ymin, ymax = float(poly_y.min()), float(poly_y.max())
            
            # 🎯 Berechne Planmodell für geneigte Flächen
            dict_points = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                for p in points
            ]
            plane_model, _ = derive_surface_plane(dict_points)
            
            # 🎯 Prüfe ob Fläche senkrecht ist
            is_vertical = False
            orientation = None
            wall_axis = None
            wall_value = None
            
            if plane_model is None:
                x_span = float(np.ptp(poly_x)) if poly_x.size > 0 else 0.0
                y_span = float(np.ptp(poly_y)) if poly_y.size > 0 else 0.0
                z_span = float(np.ptp(poly_z)) if poly_z.size > 0 else 0.0
                eps_line = 1e-6
                has_height = z_span > 1e-3
                
                if y_span < eps_line and x_span >= eps_line and has_height:
                    is_vertical = True
                    orientation = "xz"
                    wall_axis = "y"
                    wall_value = float(np.mean(poly_y))
                elif x_span < eps_line and y_span >= eps_line and has_height:
                    is_vertical = True
                    orientation = "yz"
                    wall_axis = "x"
                    wall_value = float(np.mean(poly_x))
                
                if not is_vertical:
                    plane_model = {
                        "mode": "constant",
                        "base": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                        "slope": 0.0,
                        "intercept": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                    }

            # 🚀 TEXTUR-CACHE: Prüfe ob Textur wiederverwendet werden kann
            # (Cache-Prüfung erfolgt im Hauptthread, hier nur Berechnung)
            
            # Für senkrechte Flächen: spezieller Pfad
            if is_vertical:
                return self._process_vertical_surface_texture(
                    surface_id=surface_id,
                    points=points,
                    surface_obj=surface_obj,
                    poly_x=poly_x,
                    poly_y=poly_y,
                    poly_z=poly_z,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    orientation=orientation,
                    wall_axis=wall_axis,
                    wall_value=wall_value,
                    tex_res_global=tex_res_global,
                    cbar_min=cbar_min,
                    cbar_max=cbar_max,
                    base_cmap=base_cmap,
                    norm=norm,
                    is_step_mode=is_step_mode,
                    cbar_step=cbar_step,
                    t_surface_start=t_surface_start,
                )
            
            # 🎯 NORMALER PFAD FÜR PLANARE FLÄCHEN
            return self._process_planar_surface_texture(
                surface_id=surface_id,
                points=points,
                surface_obj=surface_obj,
                poly_x=poly_x,
                poly_y=poly_y,
                poly_z=poly_z,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                plane_model=plane_model,
                source_x=source_x,
                source_y=source_y,
                values=values,
                tex_res_global=tex_res_global,
                effective_upscale_factor=effective_upscale_factor,
                cbar_min=cbar_min,
                cbar_max=cbar_max,
                base_cmap=base_cmap,
                norm=norm,
                is_step_mode=is_step_mode,
                cbar_step=cbar_step,
                t_surface_start=t_surface_start,
            )
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Error processing surface {surface_id}: {e}")
            return None

    def _process_vertical_surface_texture(
        self,
        surface_id: str,
        points: list[dict[str, float]],
        surface_obj: Any,
        poly_x: np.ndarray,
        poly_y: np.ndarray,
        poly_z: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        orientation: str,
        wall_axis: str,
        wall_value: float,
        tex_res_global: float,
        cbar_min: float,
        cbar_max: float,
        base_cmap: Any,
        norm: Any,
        is_step_mode: bool,
        cbar_step: float,
        t_surface_start: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet eine senkrechte Surface und erstellt Textur + Grid.
        Diese Methode kann parallel aufgerufen werden.
        """
        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return None
        
        try:
            # Senkrechte Flächen verwenden (u,v)-Koordinaten
            if orientation == "xz":
                poly_u = poly_x
                poly_v = poly_z
                umin, umax = xmin, xmax
                vmin, vmax = float(poly_z.min()), float(poly_z.max())
            else:  # orientation == "yz"
                poly_u = poly_y
                poly_v = poly_z
                umin, umax = ymin, ymax
                vmin, vmax = float(poly_z.min()), float(poly_z.max())
            
            # Erstelle (u,v)-Grid
            margin = tex_res_global * 0.5
            u_start = umin - margin
            u_end = umax + margin
            v_start = vmin - margin
            v_end = vmax + margin
            
            num_u = int(np.ceil((u_end - u_start) / tex_res_global)) + 1
            num_v = int(np.ceil((v_end - v_start) / tex_res_global)) + 1
            
            us = np.linspace(u_start, u_end, num_u, dtype=float)
            vs = np.linspace(v_start, v_end, num_v, dtype=float)
            if us.size < 2 or vs.size < 2:
                return None
            
            U, V = np.meshgrid(us, vs, indexing="xy")
            points_uv = np.column_stack((U.ravel(), V.ravel()))
            
            # Maske im Polygon
            poly_path_uv = Path(np.column_stack((poly_u, poly_v)))
            inside_uv = poly_path_uv.contains_points(points_uv)
            inside_uv = inside_uv.reshape(U.shape)
            
            if not np.any(inside_uv):
                return None
            
            t_geom_done = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
            
            # Hole SPL-Werte aus vertikalen Samples
            try:
                geom_vertical = prepare_vertical_plot_geometry(
                    surface_id,
                    self.settings,
                    self.container,
                    default_upscale=1,
                )
            except Exception:
                return None
            
            if geom_vertical is None:
                return None
            
            plot_u_geom = np.asarray(geom_vertical.plot_u, dtype=float)
            plot_v_geom = np.asarray(geom_vertical.plot_v, dtype=float)
            plot_values_geom = np.asarray(geom_vertical.plot_values, dtype=float)
            
            # Berechne Signatur
            texture_signature = self._calculate_texture_signature(
                surface_id=surface_id,
                points=points,
                source_x=plot_u_geom,
                source_y=plot_v_geom,
                values=plot_values_geom,
                cbar_min=cbar_min,
                cbar_max=cbar_max,
                cmap_object=base_cmap,
                colorization_mode="Color step" if is_step_mode else "Gradient",
                cbar_step=cbar_step,
                tex_res_surface=tex_res_global,
                plane_model=None,
            )
            
            # Interpoliere SPL-Werte
            if is_step_mode:
                spl_flat_uv = self._nearest_interpolate_grid(
                    plot_u_geom,
                    plot_v_geom,
                    plot_values_geom,
                    U.ravel(),
                    V.ravel(),
                )
            else:
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
            alpha_mask_uv = inside_uv & np.isfinite(spl_clipped_uv)
            rgba_uv[..., 3] = np.where(alpha_mask_uv, 1.0, 0.0)
            img_rgba_uv = (np.clip(rgba_uv, 0.0, 1.0) * 255).astype(np.uint8)
            
            t_color_done = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
            
            # Erstelle 3D-Grid
            if orientation == "xz":
                X_3d = U
                Y_3d = np.full_like(U, wall_value)
                Z_3d = V
            else:  # orientation == "yz"
                X_3d = np.full_like(U, wall_value)
                Y_3d = U
                Z_3d = V
            
            grid = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
            
            # Textur-Koordinaten
            try:
                grid.texture_map_to_plane(inplace=True)
                t_coords = grid.point_data.get("TCoords")
            except Exception:
                t_coords = None
            
            # Manuelle Textur-Koordinaten-Berechnung
            bounds = grid.bounds
            if bounds is not None and len(bounds) >= 6:
                if orientation == "xz":
                    u_min, u_max = bounds[0], bounds[1]
                    v_min, v_max = bounds[4], bounds[5]
                else:  # yz
                    u_min, u_max = bounds[2], bounds[3]
                    v_min, v_max = bounds[4], bounds[5]
                
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
                    v_coords = 1.0 - v_coords_raw
                    
                    # Achsen-Invertierung
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
                    if t_coords is None:
                        ny_uv, nx_uv = U.shape
                        u_coords_fallback = np.linspace(0, 1, nx_uv)
                        v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)
                        U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                        t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                    grid.point_data["TCoords"] = t_coords
            else:
                if t_coords is None:
                    ny_uv, nx_uv = U.shape
                    u_coords_fallback = np.linspace(0, 1, nx_uv)
                    v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)
                    U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                    t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                grid.point_data["TCoords"] = t_coords
            
            # Erstelle Textur
            tex = pv.Texture(img_rgba_uv)
            tex.interpolate = not is_step_mode
            
            # Metadaten
            metadata = {
                'grid': grid,
                'texture': tex,
                'grid_bounds': tuple(bounds) if bounds is not None else None,
                'orientation': orientation,
                'wall_axis': wall_axis,
                'wall_value': wall_value,
                'surface_id': surface_id,
                'is_vertical': True,
            }
            
            if DEBUG_PLOT3D_TIMING:
                t_surface_end = time.perf_counter()
                print(
                    f"[PlotSPL3D] surface {surface_id} (vertical, parallel): "
                    f"geom={(t_geom_done - t_surface_start) * 1000.0:7.2f} ms, "
                    f"color/tex={(t_color_done - t_geom_done) * 1000.0:7.2f} ms, "
                    f"total={(t_surface_end - t_surface_start) * 1000.0:7.2f} ms"
                )
            
            return {
                'surface_id': surface_id,
                'grid': grid,
                'texture': tex,
                'metadata': metadata,
                'texture_signature': texture_signature,
            }
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Error in _process_vertical_surface_texture for {surface_id}: {e}")
            return None

    def _process_planar_surface_texture(
        self,
        surface_id: str,
        points: list[dict[str, float]],
        surface_obj: Any,
        poly_x: np.ndarray,
        poly_y: np.ndarray,
        poly_z: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        plane_model: dict,
        source_x: np.ndarray,
        source_y: np.ndarray,
        values: np.ndarray,
        tex_res_global: float,
        effective_upscale_factor: int,
        cbar_min: float,
        cbar_max: float,
        base_cmap: Any,
        norm: Any,
        is_step_mode: bool,
        cbar_step: float,
        t_surface_start: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet eine planare Surface und erstellt Textur + Grid.
        Diese Methode kann parallel aufgerufen werden.
        """
        try:
            import pyvista as pv  # type: ignore
        except Exception:
            return None
        
        try:
            tex_res_surface = tex_res_global
            
            # Heuristik: Erkenne axis-aligned Rechtecke
            is_axis_aligned_rectangle = False
            try:
                if poly_x.size >= 4 and poly_y.size >= 4:
                    px = poly_x
                    py = poly_y
                    if (
                        poly_x.size >= 2
                        and abs(poly_x[0] - poly_x[-1]) < 1e-6
                        and abs(poly_y[0] - poly_y[-1]) < 1e-6
                    ):
                        px = poly_x[:-1]
                        py = poly_y[:-1]
                    if px.size >= 4:
                        xmin_rect = float(px.min())
                        xmax_rect = float(px.max())
                        ymin_rect = float(py.min())
                        ymax_rect = float(py.max())
                        span_x = xmax_rect - xmin_rect
                        span_y = ymax_rect - ymin_rect
                        tol = 1e-3
                        
                        if span_x > tol and span_y > tol:
                            on_left = np.isclose(px, xmin_rect, atol=tol)
                            on_right = np.isclose(px, xmax_rect, atol=tol)
                            on_bottom = np.isclose(py, ymin_rect, atol=tol)
                            on_top = np.isclose(py, ymax_rect, atol=tol)
                            on_edge = on_left | on_right | on_bottom | on_top
                            if np.all(on_edge):
                                has_left = bool(np.any(on_left))
                                has_right = bool(np.any(on_right))
                                has_bottom = bool(np.any(on_bottom))
                                has_top = bool(np.any(on_top))
                                has_tl = bool(np.any(on_left & on_top))
                                has_tr = bool(np.any(on_right & on_top))
                                has_bl = bool(np.any(on_left & on_bottom))
                                has_br = bool(np.any(on_right & on_bottom))
                                
                                if has_left and has_right and has_bottom and has_top and has_tl and has_tr and has_bl and has_br:
                                    is_axis_aligned_rectangle = True
            except Exception:
                is_axis_aligned_rectangle = False
            
            if is_axis_aligned_rectangle and effective_upscale_factor > 1:
                tex_res_surface = tex_res_global * float(effective_upscale_factor)
            
            # Berechne Signatur
            texture_signature = self._calculate_texture_signature(
                surface_id=surface_id,
                points=points,
                source_x=source_x,
                source_y=source_y,
                values=values,
                cbar_min=cbar_min,
                cbar_max=cbar_max,
                cmap_object=base_cmap,
                colorization_mode="Color step" if is_step_mode else "Gradient",
                cbar_step=cbar_step,
                tex_res_surface=tex_res_surface,
                plane_model=plane_model,
            )
            
            # Erstelle Grid
            margin = tex_res_surface * 0.5
            x_start = xmin - margin
            x_end = xmax + margin
            y_start = ymin - margin
            y_end = ymax + margin
            
            num_x = int(np.ceil((x_end - x_start) / tex_res_surface)) + 1
            num_y = int(np.ceil((y_end - y_start) / tex_res_surface)) + 1
            
            xs = np.linspace(x_start, x_end, num_x, dtype=float)
            ys = np.linspace(y_start, y_end, num_y, dtype=float)
            if xs.size < 2 or ys.size < 2:
                return None
            
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            points_2d = np.column_stack((X.ravel(), Y.ravel()))
            
            # Maske im Polygon
            poly_path = Path(np.column_stack((poly_x, poly_y)))
            inside = poly_path.contains_points(points_2d)
            inside = inside.reshape(X.shape)
            
            if not np.any(inside):
                return None
            
            # Randoptimierung
            ny_tex, nx_tex = inside.shape
            edge_mask = np.zeros_like(inside, dtype=bool)
            for jj in range(ny_tex):
                for ii in range(nx_tex):
                    if not inside[jj, ii]:
                        continue
                    for dj in (-1, 0, 1):
                        for di in (-1, 0, 1):
                            if dj == 0 and di == 0:
                                continue
                            nj = jj + dj
                            ni = ii + di
                            if nj < 0 or nj >= ny_tex or ni < 0 or ni >= nx_tex or not inside[nj, ni]:
                                edge_mask[jj, ii] = True
                                break
                        if edge_mask[jj, ii]:
                            break
            
            if np.any(edge_mask):
                edge_indices = np.argwhere(edge_mask)
                for (jj, ii) in edge_indices:
                    x_old = float(X[jj, ii])
                    y_old = float(Y[jj, ii])
                    x_new, y_new = self._project_point_to_polyline(x_old, y_old, poly_x, poly_y)
                    X[jj, ii] = x_new
                    Y[jj, ii] = y_new
            
            t_geom_done = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
            
            # Interpoliere SPL-Werte
            if is_step_mode:
                spl_flat = self._nearest_interpolate_grid(
                    source_x,
                    source_y,
                    values,
                    X.ravel(),
                    Y.ravel(),
                )
            else:
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
            if is_step_mode:
                spl_clipped = self._quantize_to_steps(spl_clipped, cbar_step)
            
            # In Farbe umsetzen
            rgba = base_cmap(norm(spl_clipped))
            alpha_mask = inside & np.isfinite(spl_clipped)
            rgba[..., 3] = np.where(alpha_mask, 1.0, 0.0)
            img_rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
            
            t_color_done = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
            
            # Berechne Z-Koordinaten
            mode = plane_model.get("mode", "constant")
            if mode == "constant":
                Z = np.full_like(X, float(plane_model.get("base", 0.0)))
            elif mode == "x":
                slope = float(plane_model.get("slope", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope * X + intercept
            elif mode == "y":
                slope = float(plane_model.get("slope", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope * Y + intercept
            else:  # mode == "xy"
                slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                slope_y = float(plane_model.get("slope_y", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope_x * X + slope_y * Y + intercept
            
            grid = pv.StructuredGrid(X, Y, Z)
            
            # Textur-Koordinaten
            try:
                grid.texture_map_to_plane(inplace=True)
                t_coords = grid.point_data.get("TCoords")
            except Exception:
                t_coords = None
            
            # Manuelle Textur-Koordinaten-Berechnung
            bounds = grid.bounds
            if bounds is not None:
                xmin_grid, xmax_grid = bounds[0], bounds[1]
                ymin_grid, ymax_grid = bounds[2], bounds[3]
                x_span = xmax_grid - xmin_grid
                y_span = ymax_grid - ymin_grid
                
                if x_span > 1e-10 and y_span > 1e-10:
                    u_coords = (X - xmin_grid) / x_span
                    v_coords_raw = (Y - ymin_grid) / y_span
                    u_coords = np.clip(u_coords, 0.0, 1.0)
                    v_coords_raw = np.clip(v_coords_raw, 0.0, 1.0)
                    
                    v_coords = 1.0 - v_coords_raw
                    
                    # Achsen-Invertierung
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
                    
                    img_rgba_final = img_rgba.copy()
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
                    img_rgba = img_rgba_final
                else:
                    if t_coords is None:
                        ny, nx = X.shape
                        u_coords = np.linspace(0, 1, nx)
                        v_coords_raw = np.linspace(0, 1, ny)
                        v_coords = 1.0 - v_coords_raw
                        U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
                        t_coords = np.column_stack((U.ravel(), V.ravel()))
                    grid.point_data["TCoords"] = t_coords
            else:
                if t_coords is None:
                    ny, nx = X.shape
                    u_coords = np.linspace(0, 1, nx)
                    v_coords_raw = np.linspace(0, 1, ny)
                    v_coords = 1.0 - v_coords_raw
                    U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
                    t_coords = np.column_stack((U.ravel(), V.ravel()))
                grid.point_data["TCoords"] = t_coords
            
            # Validiere Textur-Koordinaten
            if "TCoords" not in grid.point_data:
                return None
            tcoords_data = grid.point_data["TCoords"]
            if tcoords_data is None or tcoords_data.shape[0] != grid.n_points:
                return None
            
            # Erstelle Textur
            tex = pv.Texture(img_rgba)
            tex.interpolate = not is_step_mode
            
            # Metadaten
            metadata = {
                'grid': grid,
                'texture': tex,
                'grid_bounds': tuple(bounds) if bounds is not None else None,
                'world_coords_x': xs.copy(),
                'world_coords_y': ys.copy(),
                'world_coords_grid_x': X.copy(),
                'world_coords_grid_y': Y.copy(),
                'texture_resolution': tex_res_surface,
                'texture_size': (ys.size, xs.size),
                'image_shape': img_rgba.shape,
                'polygon_bounds': {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                },
                'polygon_points': points,
                't_coords': t_coords.copy() if t_coords is not None else None,
                'surface_id': surface_id,
                'is_axis_aligned_rectangle': is_axis_aligned_rectangle,
                'tex_res_surface': tex_res_surface,
                'effective_upscale_factor': effective_upscale_factor,
                'is_vertical': False,
            }
            
            if DEBUG_PLOT3D_TIMING:
                t_surface_end = time.perf_counter()
                ny_tex, nx_tex = inside.shape
                num_inside = int(np.count_nonzero(inside))
                print(
                    f"[PlotSPL3D] surface {surface_id} (planar, parallel): "
                    f"axis_aligned_rect={is_axis_aligned_rectangle}, "
                    f"tex_res_surface={tex_res_surface:.4f} m, "
                    f"grid={nx_tex}x{ny_tex}, inside={num_inside} "
                    f"geom={(t_geom_done - t_surface_start) * 1000.0:7.2f} ms, "
                    f"color/tex={(t_color_done - t_geom_done) * 1000.0:7.2f} ms, "
                    f"total={(t_surface_end - t_surface_start) * 1000.0:7.2f} ms"
                )
            
            return {
                'surface_id': surface_id,
                'grid': grid,
                'texture': tex,
                'metadata': metadata,
                'texture_signature': texture_signature,
            }
        except Exception as e:
            if DEBUG_PLOT3D_TIMING:
                print(f"[PlotSPL3D] Error in _process_planar_surface_texture for {surface_id}: {e}")
            return None

    # _render_surfaces_textured ist jetzt in SPL3DPlotRenderer (Mixin)

    def _remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except KeyError:
            pass

    # Texture-Metadaten-Methoden sind jetzt in SPL3DPlotRenderer (Mixin)
    # get_texture_metadata, get_texture_world_coords, get_world_coords_to_texture_coords

    # ------------------------------------------------------------------
    # Debug-Funktionen entfernt - nur Texture-Rendering
    # ------------------------------------------------------------------

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

        # 🎯 WICHTIG: selected_axis zur Signatur hinzufügen, damit Highlight-Änderungen erkannt werden
        # Berechne maximale Surface-Dimension für Achsenflächen-Größe
        max_surface_dim = self.overlay_helper._get_max_surface_dimension(settings)
        
        axis_signature = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
            float(getattr(settings, 'axis_3d_transparency', 10.0)),
            float(max_surface_dim),  # Maximale Surface-Dimension für Achsenflächen-Größe
            getattr(self, '_axis_selected', None),  # Highlight-Status in Signatur aufnehmen
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
        
        # 🚀 OPTIMIERUNG: Highlight-IDs NICHT zur Signatur hinzufügen
        # Wenn sich nur Highlights ändern, sollen nicht alle Speaker neu gezeichnet werden
        # Stattdessen wird nur _update_speaker_highlights() aufgerufen
        # highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
        # highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
        # highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])
        
        # if isinstance(highlight_indices, (list, tuple, set)):
        #     highlight_indices_tuple = tuple(sorted((str(aid), int(idx)) for aid, idx in highlight_indices))
        # else:
        #     highlight_indices_tuple = tuple()
        
        # 🚀 OPTIMIERUNG: Highlights nicht in Speaker-Signatur, damit sich Signatur nicht ändert bei nur Highlight-Änderungen
        # Separate Highlight-Signatur für schnelles Highlight-Update ohne Neuzeichnen
        highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
        highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
        highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])
        
        if highlight_array_ids:
            highlight_array_ids_list = [str(aid) for aid in highlight_array_ids]
        elif highlight_array_id:
            highlight_array_ids_list = [str(highlight_array_id)]
        else:
            highlight_array_ids_list = []
        
        if isinstance(highlight_indices, (list, tuple, set)):
            highlight_indices_tuple = tuple(sorted((str(aid), int(idx)) for aid, idx in highlight_indices))
        else:
            highlight_indices_tuple = tuple()
        
        highlight_array_ids_tuple = tuple(sorted(highlight_array_ids_list))
        highlight_signature = (highlight_array_ids_tuple, highlight_indices_tuple)
        
        speakers_signature_with_highlights = speakers_signature_tuple

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
        
        # 🎯 Prüfe ob SPL-Daten vorhanden sind (für Signatur, damit draw_surfaces nach update_spl_plot aufgerufen wird)
        has_spl_data_for_signature = False
        try:
            if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                texture_actor_names = [name for name in self.plotter.renderer.actors.keys() if name.startswith('spl_surface_tex_')]
                has_texture_actors = len(texture_actor_names) > 0
                # Prüfe auch _surface_texture_actors direkt (falls Actors noch nicht im Renderer registriert sind)
                has_texture_actors_direct = hasattr(self, '_surface_texture_actors') and len(self._surface_texture_actors) > 0
                if spl_surface_actor is not None or spl_floor_actor is not None or has_texture_actors or has_texture_actors_direct:
                    has_spl_data_for_signature = True
        except Exception:
            pass
        
        surfaces_signature_with_active = (surfaces_signature_tuple, active_surface_id, highlight_ids_tuple, has_spl_data_for_signature)
        
        result = {
            'axis': axis_signature,
            'speakers': speakers_signature_with_highlights,  # 🚀 OPTIMIERT: Enthält KEINE Highlight-IDs mehr
            'speakers_highlights': highlight_signature,  # 🚀 NEU: Separate Highlight-Signatur für schnelles Update
            'impulse': impulse_signature_tuple,
            'surfaces': surfaces_signature_with_active,  # Enthält active_surface_id, highlight_ids und has_spl_data
        }
        
        
        return result




__all__ = ['DrawSPLPlot3D']


