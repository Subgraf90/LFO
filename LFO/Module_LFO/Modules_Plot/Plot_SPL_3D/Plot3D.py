"""PyVista-basierter SPL-Plot (3D) - alte Variante"""

from __future__ import annotations

import hashlib
import json
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
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysAxis import SPL3DOverlayAxis
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysSurfaces import SPL3DOverlaySurfaces
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysImpulse import SPL3DOverlayImpulse
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysCoordinateAxes import SPL3DOverlayCoordinateAxes
from Module_LFO.Modules_Plot.Plot_SPL_3D.PlotSPL3DSpeaker import SPL3DSpeakerMixin
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DSPL import SPL3DPlotRenderer, SPLTimeControlBar
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DCamera import SPL3DCameraController
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DInteraction import SPL3DInteractionHandler
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DViewControls import SPL3DViewControls
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DHelpers import SPL3DHelpers
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
    evaluate_surface_plane,
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
PYVISTA_AA_MODE = "msaa"  # Multi-Sample Anti-Aliasing für schärfere Plots

try:  # pragma: no cover - optional Abhängigkeit
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA_AVAILABLE = True
    _PYVISTA_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]
    _PYVISTA_AVAILABLE = False
    _PYVISTA_IMPORT_ERROR = exc


class DrawSPLPlot3D(SPL3DPlotRenderer, SPL3DCameraController, SPL3DInteractionHandler, SPL3DViewControls, SPL3DHelpers, ModuleBase, QtCore.QObject):
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

        # Direkte Instanzen der Overlay-Module
        self.overlay_axis = SPL3DOverlayAxis(self.plotter, pv)
        self.overlay_surfaces = SPL3DOverlaySurfaces(self.plotter, pv)
        self.overlay_impulse = SPL3DOverlayImpulse(self.plotter, pv)
        self.overlay_speakers = SPL3DSpeakerMixin(self.plotter, pv)
        self.overlay_coordinate_axes = SPL3DOverlayCoordinateAxes(self.plotter, pv)

        # cbar wird jetzt über colorbar_manager verwaltet
        self.cbar = None  # Wird von colorbar_manager gesetzt
        self.has_data = False
        self.surface_mesh = None  # type: pv.DataSet | None
        # Einzelne Actors pro horizontaler Surface (SPL-Flächen, Mesh-Modus)
        self._surface_actors: dict[str, Any] = {}
        # Gruppen-Actors für kombinierte Meshes (wenn Gruppen-Summen-Grid aktiviert)
        self._group_actors: dict[str, Any] = {}
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

        # initialize_empty_scene(preserve_camera=False) setzt bereits die Top-Ansicht via set_view_top()
        # daher ist der explizite Aufruf hier nicht nötig und würde den Zoom unnötig zweimal ändern
        self.initialize_empty_scene(preserve_camera=False)
        self._setup_view_controls()
        self._setup_time_control()

    
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
            
            # Sammle alle SPL-Werte: globale Daten + aktive Surface-Flächen
            all_spl_values = []
            
            # 1. Hole globale Schallfeld-Daten
            sound_field_p = calc_spl.get('sound_field_p')
            if sound_field_p is not None:
                pressure_array = np.array(sound_field_p, dtype=complex)
                if pressure_array.size > 0:
                    # Berechne SPL in dB
                    pressure_abs = np.abs(pressure_array)
                    pressure_abs = np.clip(pressure_abs, 1e-12, None)
                    spl_db = self.functions.mag2db(pressure_abs)
                    # Entferne NaN und Inf Werte
                    finite_mask = np.isfinite(spl_db)
                    if np.any(finite_mask):
                        all_spl_values.append(spl_db[finite_mask])
            
            # 2. Hole SPL-Werte von aktiven Surface-Flächen
            surface_results_data = calc_spl.get('surface_results', {})
            surface_grids_data = calc_spl.get('surface_grids', {})
            surface_definitions = getattr(self.settings, 'surface_definitions', {}) or {}
            
            if isinstance(surface_results_data, dict) and isinstance(surface_definitions, dict):
                # Prüfe welche Surfaces als "empty" markiert sind
                empty_surface_ids = set()
                if isinstance(surface_grids_data, dict):
                    for sid, grid_data in surface_grids_data.items():
                        if isinstance(grid_data, dict) and grid_data.get('is_empty', False):
                            empty_surface_ids.add(sid)
                
                # Prüfe Gruppen-Status
                surface_groups = getattr(self.settings, 'surface_groups', {})
                group_status: dict[str, dict[str, bool]] = {}
                if isinstance(surface_groups, dict):
                    for group_id, group in surface_groups.items():
                        if isinstance(group, dict):
                            group_status[group_id] = {
                                'enabled': group.get('enabled', True),
                                'hidden': group.get('hidden', False),
                            }
                
                # Iteriere über alle Surfaces
                for surface_id, surface_def in surface_definitions.items():
                    # Prüfe ob Surface aktiv ist
                    if isinstance(surface_def, SurfaceDefinition):
                        enabled = bool(getattr(surface_def, "enabled", False))
                        hidden = bool(getattr(surface_def, "hidden", False))
                        group_id = getattr(surface_def, 'group_id', None)
                    else:
                        enabled = bool(surface_def.get("enabled", False))
                        hidden = bool(surface_def.get("hidden", False))
                        group_id = surface_def.get('group_id') or surface_def.get('group_name')
                    
                    # Berücksichtige Gruppen-Status
                    if group_id and group_id in group_status:
                        group_enabled = group_status[group_id]['enabled']
                        group_hidden = group_status[group_id]['hidden']
                        if group_hidden:
                            hidden = True
                        elif not group_enabled:
                            enabled = False
                    
                    # Überspringe inaktive oder leere Surfaces
                    if not enabled or hidden or str(surface_id) in empty_surface_ids:
                        continue
                    
                    # Hole Surface-Daten
                    if surface_id not in surface_results_data:
                        continue
                    
                    result_data = surface_results_data[surface_id]
                    if not isinstance(result_data, dict):
                        continue
                    
                    # Hole sound_field_p für dieses Surface
                    surface_sound_field_p = result_data.get('sound_field_p')
                    if surface_sound_field_p is not None:
                        try:
                            surface_pressure = np.array(surface_sound_field_p, dtype=complex)
                            if surface_pressure.size > 0:
                                # Berechne SPL in dB
                                surface_pressure_abs = np.abs(surface_pressure)
                                surface_pressure_abs = np.clip(surface_pressure_abs, 1e-12, None)
                                surface_spl_db = self.functions.mag2db(surface_pressure_abs)
                                # Entferne NaN und Inf Werte
                                surface_finite_mask = np.isfinite(surface_spl_db)
                                if np.any(surface_finite_mask):
                                    all_spl_values.append(surface_spl_db[surface_finite_mask])
                        except Exception:
                            # Fehler bei Surface-Daten ignorieren
                            pass
            
            # Prüfe ob überhaupt Daten vorhanden sind
            if not all_spl_values:
                return
            
            # Kombiniere alle SPL-Werte
            combined_spl = np.concatenate(all_spl_values)
            
            # Finde minimalen und maximalen SPL
            spl_min = float(np.nanmin(combined_spl))
            spl_max = float(np.nanmax(combined_spl))
            
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
            # Lösche alle Overlay-Module
            if hasattr(self, "overlay_axis") and self.overlay_axis is not None:
                self.overlay_axis.clear()
            if hasattr(self, "overlay_surfaces") and self.overlay_surfaces is not None:
                self.overlay_surfaces.clear()
            if hasattr(self, "overlay_impulse") and self.overlay_impulse is not None:
                self.overlay_impulse.clear()
            if hasattr(self, "overlay_speakers") and self.overlay_speakers is not None:
                self.overlay_speakers.clear()
            if hasattr(self, "overlay_coordinate_axes") and self.overlay_coordinate_axes is not None:
                self.overlay_coordinate_axes.clear()
            self.has_data = False
            self.surface_mesh = None
            self._last_overlay_signatures = {}
            # 🎯 Setze auch _last_surfaces_state zurück, damit Surfaces nach initialize_empty_scene() neu gezeichnet werden
            if hasattr(self, "overlay_surfaces") and self.overlay_surfaces is not None:
                self.overlay_surfaces._last_surfaces_state = None
            # 🎯 Setze auch _last_axis_state zurück, damit Achsen nach initialize_empty_scene() neu gezeichnet werden
            if hasattr(self, "overlay_axis") and self.overlay_axis is not None:
                self.overlay_axis._last_axis_state = None
            # 🎯 Setze auch _last_axes_state zurück, damit Koordinatenachsen nach initialize_empty_scene() neu gezeichnet werden
            if hasattr(self, "overlay_coordinate_axes") and self.overlay_coordinate_axes is not None:
                self.overlay_coordinate_axes._last_axes_state = None
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
                
                # 🎯 NEU: Gruppen-Surface-Actors entfernen (spl_surface_group_*)
                if hasattr(self, "_group_actors") and isinstance(self._group_actors, dict):
                    for group_id, group_actor in list(self._group_actors.items()):
                        try:
                            actor_name = f"{self.SURFACE_NAME}_group_{group_id}"
                            if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                                if actor_name in self.plotter.renderer.actors:
                                    self.plotter.remove_actor(actor_name)
                            elif group_actor is not None:
                                self.plotter.remove_actor(group_actor)
                        except Exception:
                            pass
                    self._group_actors.clear()
                
                # 🎯 FIX: Entferne ALLE SPL-bezogenen Actors direkt aus dem Plotter
                # (auch solche, die nicht in den internen Dictionaries gespeichert sind)
                if renderer is not None and hasattr(renderer, 'actors'):
                    actors_to_remove = []
                    for actor_name in list(renderer.actors.keys()):
                        # Entferne alle SPL-bezogenen Actors
                        # Prüfe auf alle möglichen Actor-Namen-Patterns
                        if (actor_name == self.SURFACE_NAME or 
                            actor_name == self.FLOOR_NAME or
                            actor_name.startswith('spl_surface_tex_') or
                            actor_name.startswith('spl_surface_group_') or
                            actor_name.startswith('spl_surface_tri_') or
                            actor_name.startswith('spl_surface_gridtri_') or
                            actor_name.startswith('vertical_spl_') or
                            actor_name.startswith(f'{self.SURFACE_NAME}_group_') or
                            actor_name.startswith(f'{self.SURFACE_NAME}_tri_') or
                            actor_name.startswith(f'{self.SURFACE_NAME}_tex_') or
                            actor_name.startswith(f'{self.SURFACE_NAME}_gridtri_')):
                            actors_to_remove.append(actor_name)
                    
                    for actor_name in actors_to_remove:
                        try:
                            self.plotter.remove_actor(actor_name)
                        except Exception:
                            pass
                    
                    # 🎯 FIX: Stelle sicher, dass auch alle vertical SPL-Actors entfernt werden
                    # (auch wenn _clear_vertical_spl_surfaces() bereits aufgerufen wurde)
                    if hasattr(self, '_clear_vertical_spl_surfaces'):
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

    def update_overlays(self, settings, container):
        """Aktualisiert Zusatzobjekte (Achsen, Lautsprecher, Messpunkte)."""
        t_start = time.perf_counter()
        
        # #region agent log
        import json
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                impulse_points = getattr(settings, "impulse_points", []) or []
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3D.py:596","message":"update_overlays called","data":{"impulse_points_count":len(impulse_points),"settings_id":id(settings)},"timestamp":__import__('time').time()*1000}) + '\n')
        except: pass
        # #endregion
        
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
            
            # 🎯 FIX: Prüfe IMMER, ob sich der hide-Status eines Arrays geändert hat
            # (auch wenn die Gesamtsignatur gleich ist, müssen Speakers neu gezeichnet werden)
            # Diese Prüfung muss VOR der normalen Signatur-Vergleichung stattfinden,
            # damit wir die betroffenen Array-IDs tracken können
            affected_array_ids_for_speakers = set()
            hide_status_changed = False
            if previous:
                prev_speakers_sig = previous.get('speakers')
                curr_speakers_sig = signatures.get('speakers')
                # 🎯 FIX: Wenn prev_speakers_sig None ist (z.B. nach Cache-Löschung), 
                # müssen alle Arrays neu gezeichnet werden
                if prev_speakers_sig is None and curr_speakers_sig:
                    # Wenn die vorherige Signatur None ist, müssen alle Arrays neu gezeichnet werden
                    if isinstance(curr_speakers_sig, (list, tuple)):
                        for entry in curr_speakers_sig:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                                array_name = str(entry[0])
                                affected_array_ids_for_speakers.add(array_name)
                elif prev_speakers_sig and curr_speakers_sig:
                    # 🎯 FIX: Vergleiche die gesamte Signatur für jedes Array, nicht nur hide-Status
                    # So erkennen wir auch Parameter-Änderungen (z.B. Position)
                    prev_sig_dict = {}
                    curr_sig_dict = {}
                    # Extrahiere vollständige Signatur aus vorheriger Signatur
                    if isinstance(prev_speakers_sig, (list, tuple)):
                        for entry in prev_speakers_sig:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                                array_name = str(entry[0])
                                prev_sig_dict[array_name] = entry  # Speichere gesamte Signatur
                    # Extrahiere vollständige Signatur aus aktueller Signatur
                    if isinstance(curr_speakers_sig, (list, tuple)):
                        for entry in curr_speakers_sig:
                            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                                array_name = str(entry[0])
                                curr_sig_dict[array_name] = entry  # Speichere gesamte Signatur
                    
                    # Prüfe, ob sich die Signatur für jedes Array geändert hat
                    # 🎯 FIX: Tracke welche Arrays betroffen sind (hide-Status oder Parameter-Änderung)
                    all_array_names = set(prev_sig_dict.keys()) | set(curr_sig_dict.keys())
                    for array_name in all_array_names:
                        prev_sig = prev_sig_dict.get(array_name)
                        curr_sig = curr_sig_dict.get(array_name)
                        
                        # Wenn Array in einer Signatur fehlt, wurde es hinzugefügt oder entfernt
                        if prev_sig is None or curr_sig is None:
                            affected_array_ids_for_speakers.add(array_name)
                            continue
                        
                        # Vergleiche vollständige Signatur
                        if prev_sig != curr_sig:
                            affected_array_ids_for_speakers.add(array_name)
                            # Prüfe speziell hide-Status-Änderung
                            if len(prev_sig) >= 3 and len(curr_sig) >= 3:
                                prev_hide = bool(prev_sig[2])
                                curr_hide = bool(curr_sig[2])
                                if prev_hide != curr_hide:
                                    hide_status_changed = True
                    if hide_status_changed:
                        # 🎯 FIX: Speichere betroffene Array-IDs in overlay_speakers, damit draw_speakers() darauf zugreifen kann
                        if hasattr(self, 'overlay_speakers'):
                            # Konvertiere alle Array-IDs zu Strings, um Konsistenz zu gewährleisten
                            affected_array_ids_str = {str(aid) for aid in affected_array_ids_for_speakers}
                            self.overlay_speakers._affected_array_ids_for_speakers = affected_array_ids_str
            
            if not previous:
                categories_to_refresh = set(signatures.keys())
            else:
                categories_to_refresh = {
                    key for key, value in signatures.items() 
                    if key != 'speakers_highlights' and value != previous.get(key)
                }
            
            # 🎯 FIX: Wenn sich Parameter oder hide-Status geändert haben, füge 'speakers' zu categories_to_refresh hinzu
            # UND setze die betroffenen Array-IDs, auch wenn die Signatur bereits als geändert markiert wurde
            if affected_array_ids_for_speakers:
                categories_to_refresh.add('speakers')
                # 🎯 FIX: Setze betroffene Array-IDs, damit nur diese Arrays neu gezeichnet werden
                if hasattr(self, 'overlay_speakers'):
                    # Konvertiere alle Array-IDs zu Strings, um Konsistenz zu gewährleisten
                    affected_array_ids_str = {str(aid) for aid in affected_array_ids_for_speakers}
                    self.overlay_speakers._affected_array_ids_for_speakers = affected_array_ids_str
            
        # 🚀 OPTIMIERUNG: Prüfe ob sich nur Highlights geändert haben
        highlight_changed = False
        if previous:
            prev_highlights = previous.get('speakers_highlights')
            curr_highlights = signatures.get('speakers_highlights')
            if prev_highlights != curr_highlights:
                highlight_changed = True
        
        # 🐛 Falls die Axis-Actors fehlen (z. B. nach File-Reload/Plotter-Clear),
        # erzwinge ein Redraw der Achsen, auch wenn die Signatur unverändert ist.
        try:
            renderer = self.plotter.renderer if hasattr(self.plotter, "renderer") else None
            axis_names = []
            if hasattr(self, "overlay_axis") and hasattr(self.overlay_axis, "_category_actors"):
                axis_names = self.overlay_axis._category_actors.get('axis', []) or []
            axis_missing = False
            if renderer is not None and axis_names:
                # Wenn keiner der gespeicherten Axis-Actor im Renderer liegt, neu zeichnen
                axis_missing = not any(name in renderer.actors for name in axis_names)
            elif renderer is not None and not axis_names:
                # Kein bekannter Axis-Actor -> ebenfalls neu zeichnen
                axis_missing = True
            if axis_missing:
                categories_to_refresh.add('axis')
        except Exception:
            # Fällt zurück auf normales Verhalten
            pass
        
        # Wenn sich nur Highlights geändert haben und Speaker bereits gezeichnet sind, nur Highlights updaten
        if highlight_changed and 'speakers' not in categories_to_refresh and hasattr(self.overlay_speakers, '_speaker_actor_cache') and self.overlay_speakers._speaker_actor_cache:
            with perf_section("PlotSPL3D.update_overlays.update_speaker_highlights_only"):
                if hasattr(self.overlay_speakers, '_update_speaker_highlights'):
                    self.overlay_speakers._update_speaker_highlights(settings)
                    self._last_overlay_signatures = signatures
                    if not self._rotate_active and not self._pan_active:
                        self._save_camera_state()
                    return
        
        if not categories_to_refresh:
            self._last_overlay_signatures = signatures
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
            return
        
        # Debug-Flag wird nicht mehr benötigt (war nur für Koordinator)
        if 'axis' in categories_to_refresh:
            with perf_section("PlotSPL3D.update_overlays.draw_axis"):
                self.overlay_axis.draw_axis_lines(settings, selected_axis=self._axis_selected)
        
        if 'surfaces' in categories_to_refresh:
            with perf_section("PlotSPL3D.update_overlays.draw_surfaces"):
                # Prüfe ob SPL-Daten vorhanden sind, um zu entscheiden ob enabled Surfaces gezeichnet werden sollen
                create_empty_plot_surfaces = False
                try:
                    # 🎯 FIX: Prüfe zuerst, ob alle Lautsprecher muted/hidden sind
                    # Wenn ja, sollten keine SPL-Daten angezeigt werden
                    has_active_sources = False
                    speaker_arrays = getattr(settings, 'speaker_arrays', {})
                    if isinstance(speaker_arrays, dict) and speaker_arrays:
                        has_active_sources = any(
                            not (getattr(arr, 'mute', False) or getattr(arr, 'hide', False))
                            for arr in speaker_arrays.values()
                        )
                    
                    # Prüfe ob SPL-Daten vorhanden sind (gleiche Logik wie in draw_surfaces)
                    # WICHTIG: Die entscheidende Prüfung ist sound_field_p, nicht nur ob Texture-Actors existieren
                    has_spl_data = False
                    
                    # Nur wenn aktive Sources vorhanden sind, prüfe auf SPL-Daten
                    if has_active_sources:
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
                    
                    # Wenn keine aktiven Sources ODER keine SPL-Daten vorhanden sind, zeichne enabled Surfaces für leeren Plot
                    if not has_active_sources or not has_spl_data:
                        create_empty_plot_surfaces = True
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # Bei Fehler: konservativ, zeichne keine enabled Surfaces
                    create_empty_plot_surfaces = False
                
                self.overlay_surfaces.draw_surfaces(settings, container, create_empty_plot_surfaces=create_empty_plot_surfaces)
        
        if 'speakers' in categories_to_refresh:
            with perf_section("PlotSPL3D.update_overlays.draw_speakers"):
                with perf_section("PlotSPL3D.update_overlays.build_cabinet_lookup"):
                    cabinet_lookup = self.overlay_speakers.build_cabinet_lookup(container)
                self.overlay_speakers.draw_speakers(settings, container, cabinet_lookup)
                # 🎯 FIX: Setze _affected_array_ids_for_speakers zurück, damit es nicht bei der nächsten Aktualisierung verwendet wird
                if hasattr(self.overlay_speakers, '_affected_array_ids_for_speakers'):
                    self.overlay_speakers._affected_array_ids_for_speakers = None
        else:
            # Auch wenn sich die Speaker-Definitionen nicht geändert haben,
            # müssen wir die Highlights aktualisieren, falls sich die Highlight-IDs geändert haben
            with perf_section("PlotSPL3D.update_overlays.update_speaker_highlights"):
                if hasattr(self.overlay_speakers, '_update_speaker_highlights'):
                    self.overlay_speakers._update_speaker_highlights(settings)
                    # Render-Update triggern, damit Änderungen sichtbar werden
                    try:
                        self.plotter.render()
                    except Exception:  # noqa: BLE001
                        pass
        
        if 'impulse' in categories_to_refresh:
            with perf_section("PlotSPL3D.update_overlays.draw_impulse"):
                self.overlay_impulse.draw_impulse_points(settings)
        
        # Zeichne Koordinatenachsen (immer aktiviert)
        with perf_section("PlotSPL3D.update_overlays.draw_coordinate_axes"):
            self.overlay_coordinate_axes.draw_coordinate_axes(settings, enabled=True)
        
        with perf_section("PlotSPL3D.update_overlays.finalize"):
            self._last_overlay_signatures = signatures
            
            # 🎯 Beim ersten Start: Zoom auf das Default-Surface einstellen (nach dem Zeichnen aller Overlays)
            if (not getattr(self, "_did_initial_overlay_zoom", False)) and 'surfaces' in categories_to_refresh:
                self._zoom_to_default_surface()
            
            if not self._rotate_active and not self._pan_active:
                self._save_camera_state()
        t_end = time.perf_counter()
        
    def render(self):
        """Erzwingt ein Rendering der Szene."""
        if self._is_rendering:
            return
        # 🛡️ SICHERHEIT: Prüfe ob Plotter noch gültig ist
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        t_start = time.perf_counter() if DEBUG_PLOT3D_TIMING else 0.0
        self._is_rendering = True
        try:
            try:
                self.plotter.render()
            except Exception:  # pragma: no cover - Rendering kann im Offscreen fehlschlagen
                pass
            if hasattr(self, 'view_control_widget'):
                try:
                    self.view_control_widget.raise_()
                except Exception:
                    pass
            if not self._skip_next_render_restore and self._camera_state is not None:
                try:
                    self._restore_camera(self._camera_state)
                except Exception:
                    pass
            self._skip_next_render_restore = False
            try:
                self._save_camera_state()
            except Exception:
                pass
        finally:
            self._is_rendering = False
            if DEBUG_PLOT3D_TIMING:
                t_end = time.perf_counter()

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

   