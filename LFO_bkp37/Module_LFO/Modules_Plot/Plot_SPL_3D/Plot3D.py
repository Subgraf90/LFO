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
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysAxis import SPL3DOverlayAxis
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysSurfaces import SPL3DOverlaySurfaces
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysImpulse import SPL3DOverlayImpulse
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
            self.has_data = False
            self.surface_mesh = None
            self._last_overlay_signatures = {}
            # 🎯 Setze auch _last_surfaces_state zurück, damit Surfaces nach initialize_empty_scene() neu gezeichnet werden
            if hasattr(self, "overlay_surfaces") and self.overlay_surfaces is not None:
                self.overlay_surfaces._last_surfaces_state = None
            # 🎯 Setze auch _last_axis_state zurück, damit Achsen nach initialize_empty_scene() neu gezeichnet werden
            if hasattr(self, "overlay_axis") and self.overlay_axis is not None:
                self.overlay_axis._last_axis_state = None
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
                
                self.overlay_surfaces.draw_surfaces(settings, container, create_empty_plot_surfaces=create_empty_plot_surfaces)
        
        if 'speakers' in categories_to_refresh:
            with perf_section("PlotSPL3D.update_overlays.draw_speakers"):
                with perf_section("PlotSPL3D.update_overlays.build_cabinet_lookup"):
                    cabinet_lookup = self.overlay_speakers.build_cabinet_lookup(container)
                self.overlay_speakers.draw_speakers(settings, container, cabinet_lookup)
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
    # Kamera-Zustand und -Ansichten sind jetzt in SPL3DCameraController (Plot3DCamera.py)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # View-Controls und UI-Widgets sind jetzt in SPL3DViewControls (Plot3DViewControls.py)
    # ------------------------------------------------------------------

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

    # Statische Hilfsmethoden sind jetzt in Plot3DHelpers.py
    # _has_valid_data und _compute_surface_signature werden durch Vererbung bereitgestellt

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
                    
                    # 🐛 DEBUG: Textur-Koordinaten vor Transformation (vertikal)
                    if DEBUG_PLOT3D_TIMING:
                        ny_uv, nx_uv = img_rgba_uv.shape[:2]
                        print(f"[DEBUG Texture] {surface_id} (vertical, {orientation}):")
                        print(f"  img_rgba_uv.shape = {img_rgba_uv.shape} (H={ny_uv}, W={nx_uv})")
                        print(f"  orientation={orientation}, wall_axis={wall_axis}, wall_value={wall_value:.2f}")
                        print(f"  u range: [{u_min:.2f}, {u_max:.2f}], v range: [{v_min:.2f}, {v_max:.2f}]")
                        print(f"  u_coords range: [{u_coords.min():.3f}, {u_coords.max():.3f}]")
                        print(f"  v_coords_raw range: [{v_coords_raw.min():.3f}, {v_coords_raw.max():.3f}]")
                        print(f"  v_coords range (after 1.0-v): [{v_coords.min():.3f}, {v_coords.max():.3f}]")
                        if orientation == "xz":
                            print(f"  Corner [0,0]: u={u_coords[0,0]:.3f}, v={v_coords[0,0]:.3f}, world=({X_3d[0,0]:.2f}, {Y_3d[0,0]:.2f}, {Z_3d[0,0]:.2f}), SPL={spl_img_uv[0,0]:.1f}->{spl_clipped_uv[0,0]:.1f} dB")
                            print(f"  Corner [0,nx-1]: u={u_coords[0,nx_uv-1]:.3f}, v={v_coords[0,nx_uv-1]:.3f}, world=({X_3d[0,nx_uv-1]:.2f}, {Y_3d[0,nx_uv-1]:.2f}, {Z_3d[0,nx_uv-1]:.2f}), SPL={spl_img_uv[0,nx_uv-1]:.1f}->{spl_clipped_uv[0,nx_uv-1]:.1f} dB")
                            print(f"  Corner [ny-1,0]: u={u_coords[ny_uv-1,0]:.3f}, v={v_coords[ny_uv-1,0]:.3f}, world=({X_3d[ny_uv-1,0]:.2f}, {Y_3d[ny_uv-1,0]:.2f}, {Z_3d[ny_uv-1,0]:.2f}), SPL={spl_img_uv[ny_uv-1,0]:.1f}->{spl_clipped_uv[ny_uv-1,0]:.1f} dB")
                            print(f"  Corner [ny-1,nx-1]: u={u_coords[ny_uv-1,nx_uv-1]:.3f}, v={v_coords[ny_uv-1,nx_uv-1]:.3f}, world=({X_3d[ny_uv-1,nx_uv-1]:.2f}, {Y_3d[ny_uv-1,nx_uv-1]:.2f}, {Z_3d[ny_uv-1,nx_uv-1]:.2f}), SPL={spl_img_uv[ny_uv-1,nx_uv-1]:.1f}->{spl_clipped_uv[ny_uv-1,nx_uv-1]:.1f} dB")
                        else:  # yz
                            print(f"  Corner [0,0]: u={u_coords[0,0]:.3f}, v={v_coords[0,0]:.3f}, world=({X_3d[0,0]:.2f}, {Y_3d[0,0]:.2f}, {Z_3d[0,0]:.2f}), SPL={spl_img_uv[0,0]:.1f}->{spl_clipped_uv[0,0]:.1f} dB")
                            print(f"  Corner [0,nx-1]: u={u_coords[0,nx_uv-1]:.3f}, v={v_coords[0,nx_uv-1]:.3f}, world=({X_3d[0,nx_uv-1]:.2f}, {Y_3d[0,nx_uv-1]:.2f}, {Z_3d[0,nx_uv-1]:.2f}), SPL={spl_img_uv[0,nx_uv-1]:.1f}->{spl_clipped_uv[0,nx_uv-1]:.1f} dB")
                            print(f"  Corner [ny-1,0]: u={u_coords[ny_uv-1,0]:.3f}, v={v_coords[ny_uv-1,0]:.3f}, world=({X_3d[ny_uv-1,0]:.2f}, {Y_3d[ny_uv-1,0]:.2f}, {Z_3d[ny_uv-1,0]:.2f}), SPL={spl_img_uv[ny_uv-1,0]:.1f}->{spl_clipped_uv[ny_uv-1,0]:.1f} dB")
                            print(f"  Corner [ny-1,nx-1]: u={u_coords[ny_uv-1,nx_uv-1]:.3f}, v={v_coords[ny_uv-1,nx_uv-1]:.3f}, world=({X_3d[ny_uv-1,nx_uv-1]:.2f}, {Y_3d[ny_uv-1,nx_uv-1]:.2f}, {Z_3d[ny_uv-1,nx_uv-1]:.2f}), SPL={spl_img_uv[ny_uv-1,nx_uv-1]:.1f}->{spl_clipped_uv[ny_uv-1,nx_uv-1]:.1f} dB")
                    
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
                    
                    # 🐛 DEBUG: Transformations-Einstellungen (vertikal)
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  Transform settings: invert_x={invert_x}, invert_y={invert_y}, swap_axes={swap_axes}")
                    
                    img_rgba_final = img_rgba_uv.copy()
                    if invert_x:
                        img_rgba_final = np.fliplr(img_rgba_final)
                        u_coords = 1.0 - u_coords
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: fliplr (horizontal)")
                    if invert_y:
                        img_rgba_final = np.flipud(img_rgba_final)
                        v_coords = 1.0 - v_coords
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: flipud (vertical)")
                    if swap_axes:
                        img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                        u_coords, v_coords = v_coords.copy(), u_coords.copy()
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: transpose (swap axes)")
                    
                    # 🐛 DEBUG: Textur-Koordinaten nach Transformation (vertikal)
                    if DEBUG_PLOT3D_TIMING:
                        ny_final, nx_final = img_rgba_final.shape[:2]
                        print(f"  After transform: img_rgba_final.shape = {img_rgba_final.shape}")
                        print(f"  Final u_coords range: [{u_coords.min():.3f}, {u_coords.max():.3f}]")
                        print(f"  Final v_coords range: [{v_coords.min():.3f}, {v_coords.max():.3f}]")
                        print(f"  Final Corner [0,0]: u={u_coords[0,0]:.3f}, v={v_coords[0,0]:.3f}")
                        print(f"  Final Corner [0,nx-1]: u={u_coords[0,nx_final-1]:.3f}, v={v_coords[0,nx_final-1]:.3f}")
                        print(f"  Final Corner [ny-1,0]: u={u_coords[ny_final-1,0]:.3f}, v={v_coords[ny_final-1,0]:.3f}")
                        print(f"  Final Corner [ny-1,nx-1]: u={u_coords[ny_final-1,nx_final-1]:.3f}, v={v_coords[ny_final-1,nx_final-1]:.3f}")
                    
                    # 🎯 WICHTIG: PyVista StructuredGrid verwendet column-major (Fortran-style) Indizierung!
                    # Die Arrays müssen in column-major Reihenfolge flach gemacht werden
                    # row-major: idx = jj * nx + ii
                    # col-major: idx = ii * ny + jj
                    # Verwende order='F' (Fortran-style) für column-major Reihenfolge
                    t_coords = np.column_stack((
                        u_coords.ravel(order='F'),
                        v_coords.ravel(order='F')
                    ))
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
            
            # 🐛 DEBUG: Grid-Erstellung
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Grid Creation] {surface_id}:")
                print(f"  Bounding box: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
                print(f"  Margin: {margin:.4f} m")
                print(f"  Grid range: x=[{x_start:.2f}, {x_end:.2f}], y=[{y_start:.2f}, {y_end:.2f}]")
                print(f"  Grid resolution: {tex_res_surface:.4f} m")
                print(f"  Grid size: num_x={num_x}, num_y={num_y}")
                print(f"  xs: first={xs[0]:.2f}, last={xs[-1]:.2f}, count={len(xs)}")
                print(f"  ys: first={ys[0]:.2f}, last={ys[-1]:.2f}, count={len(ys)}")
            
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            
            # 🐛 DEBUG: Meshgrid-Verifikation
            if DEBUG_PLOT3D_TIMING:
                print(f"  Meshgrid shape: X.shape={X.shape}, Y.shape={Y.shape}")
                print(f"  Meshgrid indexing verification:")
                print(f"    X[0,0]={X[0,0]:.2f} (should be xs[0]={xs[0]:.2f})")
                print(f"    X[0,-1]={X[0,-1]:.2f} (should be xs[-1]={xs[-1]:.2f})")
                print(f"    Y[0,0]={Y[0,0]:.2f} (should be ys[0]={ys[0]:.2f})")
                print(f"    Y[-1,0]={Y[-1,0]:.2f} (should be ys[-1]={ys[-1]:.2f})")
                print(f"    Corner [0,0]: X={X[0,0]:.2f}, Y={Y[0,0]:.2f}")
                print(f"    Corner [0,nx-1]: X={X[0,-1]:.2f}, Y={Y[0,-1]:.2f}")
                print(f"    Corner [ny-1,0]: X={X[-1,0]:.2f}, Y={Y[-1,0]:.2f}")
                print(f"    Corner [ny-1,nx-1]: X={X[-1,-1]:.2f}, Y={Y[-1,-1]:.2f}")
            
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
            
            # 🎯 WICHTIG: Für schräge Flächen müssen wir die Z-Koordinaten VOR der SPL-Interpolation berechnen
            # und dann die SPL-Werte an den 3D-Positionen (X, Y, Z) interpolieren, nicht nur an (X, Y)
            mode = plane_model.get("mode", "constant")
            is_slanted = mode != "constant"
            
            # Berechne Z-Koordinaten für schräge Flächen
            if is_slanted:
                if mode == "x":
                    slope = float(plane_model.get("slope", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z_surface = slope * X + intercept
                elif mode == "y":
                    slope = float(plane_model.get("slope", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z_surface = slope * Y + intercept
                else:  # mode == "xy"
                    slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                    slope_y = float(plane_model.get("slope_y", 0.0))
                    intercept = float(plane_model.get("intercept", 0.0))
                    Z_surface = slope_x * X + slope_y * Y + intercept
            else:
                Z_surface = None
            
            # 🐛 DEBUG: Plane model für schräge Flächen
            if DEBUG_PLOT3D_TIMING:
                print(f"[DEBUG Grid] {surface_id}: plane_model mode={mode}, is_slanted={is_slanted}")
                if is_slanted:
                    if mode == "x":
                        slope = float(plane_model.get("slope", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        print(f"  Z = {slope:.4f} * X + {intercept:.4f}")
                    elif mode == "y":
                        slope = float(plane_model.get("slope", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        print(f"  Z = {slope:.4f} * Y + {intercept:.4f}")
                    elif mode == "xy":
                        slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                        slope_y = float(plane_model.get("slope_y", 0.0))
                        intercept = float(plane_model.get("intercept", 0.0))
                        print(f"  Z = {slope_x:.4f} * X + {slope_y:.4f} * Y + {intercept:.4f}")
                    # Berechne Z an Ecken für Debug
                    z_00 = Z_surface[0,0]
                    z_0n = Z_surface[0,-1]
                    z_n0 = Z_surface[-1,0]
                    z_nn = Z_surface[-1,-1]
                    print(f"  Z at corners: [0,0]={z_00:.2f}, [0,nx-1]={z_0n:.2f}, [ny-1,0]={z_n0:.2f}, [ny-1,nx-1]={z_nn:.2f}")
                print(f"  Grid shape: {X.shape}, X range: [{X.min():.2f}, {X.max():.2f}], Y range: [{Y.min():.2f}, {Y.max():.2f}]")
                print(f"  Source grid: X={len(source_x)} points, Y={len(source_y)} points, values shape={values.shape}")
            
            # 🎯 FÜR SCHRÄGE FLÄCHEN: Interpoliere SPL-Werte an 3D-Positionen (X, Y, Z)
            # Versuche 3D-Interpolation, wenn Z-Koordinaten verfügbar sind
            source_z = None
            if is_slanted and hasattr(self, 'container') and self.container is not None:
                try:
                    calc_spl = getattr(self.container, "calculation_spl", None)
                    if isinstance(calc_spl, dict) and "sound_field_z" in calc_spl:
                        raw_z = calc_spl["sound_field_z"]
                        if raw_z is not None:
                            source_z = np.asarray(raw_z, dtype=float)
                            if source_z.shape != (len(source_y), len(source_x)):
                                if source_z.size == len(source_y) * len(source_x):
                                    source_z = source_z.reshape(len(source_y), len(source_x))
                                else:
                                    source_z = None
                except Exception:
                    source_z = None
            
            # Verwende 3D-Interpolation für schräge Flächen, wenn Z-Koordinaten verfügbar sind
            if is_slanted and source_z is not None and Z_surface is not None:
                if DEBUG_PLOT3D_TIMING:
                    print(f"  ✅ Using 3D interpolation for slanted surface")
                method = "nearest" if is_step_mode else "linear"
                spl_flat = self._interpolate_grid_3d(
                    source_x,
                    source_y,
                    source_z,
                    values,
                    X.ravel(),
                    Y.ravel(),
                    Z_surface.ravel(),
                    method=method,
                )
            else:
                # Fallback: 2D-Interpolation
                if DEBUG_PLOT3D_TIMING and is_slanted:
                    print(f"  ⚠️  Using 2D interpolation (Z-coordinates not available)")
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
            
            # 🐛 DEBUG: SPL-Mapping-Prüfung nach Interpolation
            if DEBUG_PLOT3D_TIMING:
                print(f"  [DEBUG SPL Mapping] {surface_id}:")
                print(f"    spl_img shape: {spl_img.shape}")
                print(f"    spl_img range: [{spl_img.min():.1f}, {spl_img.max():.1f}] dB")
                # Prüfe Ecken nach Interpolation
                corners_check = [
                    (0, 0, "oben-links"),
                    (0, spl_img.shape[1]-1, "oben-rechts"),
                    (spl_img.shape[0]-1, 0, "unten-links"),
                    (spl_img.shape[0]-1, spl_img.shape[1]-1, "unten-rechts"),
                ]
                for jj, ii, name in corners_check:
                    x_val = X[jj, ii]
                    y_val = Y[jj, ii]
                    spl_val = spl_img[jj, ii]
                    print(f"    Corner {name} [jj={jj},ii={ii}]: X={x_val:.2f}, Y={y_val:.2f}, SPL={spl_val:.1f} dB")
            
            # 🐛 DEBUG: Grid-Punkte-Prüfung für schräge Flächen
            if DEBUG_PLOT3D_TIMING and is_slanted:
                interp_method = "3D" if (source_z is not None and Z_surface is not None) else "2D"
                print(f"  SPL at corners ({interp_method} interpolation): [0,0]={spl_img[0,0]:.1f} dB, [0,nx-1]={spl_img[0,-1]:.1f} dB, [ny-1,0]={spl_img[-1,0]:.1f} dB, [ny-1,nx-1]={spl_img[-1,-1]:.1f} dB")
                if source_z is None:
                    print(f"  ⚠️  WARNING: Z-Koordinaten des Source-Grids nicht verfügbar - verwende 2D-Interpolation")
                elif Z_surface is None:
                    print(f"  ⚠️  WARNING: Z-Koordinaten der Fläche nicht berechnet - verwende 2D-Interpolation")
                else:
                    print(f"  ✅ Using 3D interpolation with source_z shape={source_z.shape}")
                    
                    # 🐛 Detaillierte Grid-Punkte-Prüfung
                    print(f"  [DEBUG Grid Points] {surface_id}:")
                    # Prüfe Ecken
                    corners = [
                        (0, 0, "oben-links"),
                        (0, X.shape[1]-1, "oben-rechts"),
                        (X.shape[0]-1, 0, "unten-links"),
                        (X.shape[0]-1, X.shape[1]-1, "unten-rechts"),
                    ]
                    for jj, ii, name in corners:
                        x_val = X[jj, ii]
                        y_val = Y[jj, ii]
                        z_surface_val = Z_surface[jj, ii]
                        
                        # Finde nächstgelegenen Source-Grid-Punkt
                        idx_x = np.searchsorted(source_x, x_val, side="left")
                        idx_x = np.clip(idx_x, 0, len(source_x) - 1)
                        idx_y = np.searchsorted(source_y, y_val, side="left")
                        idx_y = np.clip(idx_y, 0, len(source_y) - 1)
                        
                        # Korrigiere auf wirklich nächsten Nachbarn
                        if idx_x > 0:
                            dist_left = abs(x_val - source_x[idx_x - 1])
                            dist_right = abs(x_val - source_x[idx_x])
                            if dist_left < dist_right:
                                idx_x = idx_x - 1
                        if idx_y > 0:
                            dist_left = abs(y_val - source_y[idx_y - 1])
                            dist_right = abs(y_val - source_y[idx_y])
                            if dist_left < dist_right:
                                idx_y = idx_y - 1
                        
                        z_grid_val = source_z[idx_y, idx_x] if source_z is not None else 0.0
                        spl_source = values[idx_y, idx_x]
                        z_diff = z_surface_val - z_grid_val
                        
                        print(f"    Corner {name} [jj={jj},ii={ii}]:")
                        print(f"      World: X={x_val:.2f}, Y={y_val:.2f}, Z_surface={z_surface_val:.2f}")
                        print(f"      Source grid [idx_y={idx_y},idx_x={idx_x}]: X={source_x[idx_x]:.2f}, Y={source_y[idx_y]:.2f}, Z_grid={z_grid_val:.2f}")
                        print(f"      Z-Differenz: {z_diff:.2f} m (surface - grid)")
                        print(f"      SPL: interpolated={spl_img[jj,ii]:.1f} dB, source_grid={spl_source:.1f} dB")
                    
                    # Prüfe Mitte
                    mid_jj, mid_ii = X.shape[0] // 2, X.shape[1] // 2
                    x_mid = X[mid_jj, mid_ii]
                    y_mid = Y[mid_jj, mid_ii]
                    z_surface_mid = Z_surface[mid_jj, mid_ii]
                    
                    idx_x_mid = np.searchsorted(source_x, x_mid, side="left")
                    idx_x_mid = np.clip(idx_x_mid, 0, len(source_x) - 1)
                    idx_y_mid = np.searchsorted(source_y, y_mid, side="left")
                    idx_y_mid = np.clip(idx_y_mid, 0, len(source_y) - 1)
                    
                    if idx_x_mid > 0:
                        dist_left = abs(x_mid - source_x[idx_x_mid - 1])
                        dist_right = abs(x_mid - source_x[idx_x_mid])
                        if dist_left < dist_right:
                            idx_x_mid = idx_x_mid - 1
                    if idx_y_mid > 0:
                        dist_left = abs(y_mid - source_y[idx_y_mid - 1])
                        dist_right = abs(y_mid - source_y[idx_y_mid])
                        if dist_left < dist_right:
                            idx_y_mid = idx_y_mid - 1
                    
                    z_grid_mid = source_z[idx_y_mid, idx_x_mid] if source_z is not None else 0.0
                    spl_source_mid = values[idx_y_mid, idx_x_mid]
                    z_diff_mid = z_surface_mid - z_grid_mid
                    
                    print(f"    Center [jj={mid_jj},ii={mid_ii}]:")
                    print(f"      World: X={x_mid:.2f}, Y={y_mid:.2f}, Z_surface={z_surface_mid:.2f}")
                    print(f"      Source grid [idx_y={idx_y_mid},idx_x={idx_x_mid}]: X={source_x[idx_x_mid]:.2f}, Y={source_y[idx_y_mid]:.2f}, Z_grid={z_grid_mid:.2f}")
                    print(f"      Z-Differenz: {z_diff_mid:.2f} m (surface - grid)")
                    print(f"      SPL: interpolated={spl_img[mid_jj,mid_ii]:.1f} dB, source_grid={spl_source_mid:.1f} dB")
            
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
            
            # 🐛 DEBUG: Prüfe Surface-Punkte vs. gerenderte Mesh-Positionen
            if DEBUG_PLOT3D_TIMING:
                print(f"  [DEBUG Surface Points vs Mesh] {surface_id}:")
                print(f"    Original Surface Points ({len(points)} points):")
                for i, p in enumerate(points):
                    print(f"      Point {i}: X={p.get('x', 0.0):.2f}, Y={p.get('y', 0.0):.2f}, Z={p.get('z', 0.0):.2f}")
                print(f"    Rendered Mesh Bounds:")
                print(f"      X: [{X.min():.2f}, {X.max():.2f}]")
                print(f"      Y: [{Y.min():.2f}, {Y.max():.2f}]")
                print(f"      Z: [{Z.min():.2f}, {Z.max():.2f}]")
                # Prüfe ob Surface-Punkte innerhalb der Mesh-Bounds liegen
                poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                poly_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
                print(f"    Surface Polygon Bounds:")
                print(f"      X: [{poly_x.min():.2f}, {poly_x.max():.2f}]")
                print(f"      Y: [{poly_y.min():.2f}, {poly_y.max():.2f}]")
                print(f"      Z: [{poly_z.min():.2f}, {poly_z.max():.2f}]")
                # Prüfe Ecken des Meshes vs. erwartete Ecken aus Polygon
                mesh_x_min, mesh_x_max = X.min(), X.max()
                mesh_y_min, mesh_y_max = Y.min(), Y.max()
                poly_x_min, poly_x_max = poly_x.min(), poly_x.max()
                poly_y_min, poly_y_max = poly_y.min(), poly_y.max()
                x_diff_min = abs(mesh_x_min - poly_x_min)
                x_diff_max = abs(mesh_x_max - poly_x_max)
                y_diff_min = abs(mesh_y_min - poly_y_min)
                y_diff_max = abs(mesh_y_max - poly_y_max)
                print(f"    Position Differences (mesh - polygon):")
                print(f"      X-min diff: {x_diff_min:.3f} m, X-max diff: {x_diff_max:.3f} m")
                print(f"      Y-min diff: {y_diff_min:.3f} m, Y-max diff: {y_diff_max:.3f} m")
                if x_diff_min > 0.1 or x_diff_max > 0.1 or y_diff_min > 0.1 or y_diff_max > 0.1:
                    print(f"      ⚠️  WARNING: Mesh-Positionen weichen deutlich von Surface-Polygon ab!")
            
            # 🐛 DEBUG: Mesh-Geometrie-Prüfung
            if DEBUG_PLOT3D_TIMING:
                print(f"  [DEBUG Mesh Geometry] {surface_id}:")
                print(f"    Grid shape: {X.shape}, Z shape: {Z.shape}")
                print(f"    Grid bounds: {grid.bounds}")
                print(f"    Grid points: {grid.n_points}")
                # Prüfe erste und letzte Punkte des Meshes
                points = grid.points
                if points is not None and len(points) > 0:
                    print(f"    First point: ({points[0,0]:.2f}, {points[0,1]:.2f}, {points[0,2]:.2f})")
                    print(f"    Last point: ({points[-1,0]:.2f}, {points[-1,1]:.2f}, {points[-1,2]:.2f})")
                    # Prüfe Ecken des Meshes
                    ny_mesh, nx_mesh = X.shape
                    print(f"    Array dimensions: ny={ny_mesh}, nx={nx_mesh}")
                    print(f"    Array corner values:")
                    print(f"      X[0,0]={X[0,0]:.2f}, X[0,nx-1]={X[0,nx_mesh-1]:.2f}, X[ny-1,0]={X[ny_mesh-1,0]:.2f}, X[ny-1,nx-1]={X[ny_mesh-1,nx_mesh-1]:.2f}")
                    print(f"      Y[0,0]={Y[0,0]:.2f}, Y[0,nx-1]={Y[0,nx_mesh-1]:.2f}, Y[ny-1,0]={Y[ny_mesh-1,0]:.2f}, Y[ny-1,nx-1]={Y[ny_mesh-1,nx_mesh-1]:.2f}")
                    print(f"      Z[0,0]={Z[0,0]:.2f}, Z[0,nx-1]={Z[0,nx_mesh-1]:.2f}, Z[ny-1,0]={Z[ny_mesh-1,0]:.2f}, Z[ny-1,nx-1]={Z[ny_mesh-1,nx_mesh-1]:.2f}")
                    
                    # Prüfe verschiedene Indizierungs-Möglichkeiten
                    corner_indices_row_major = [
                        (0, "oben-links (row-major)"),
                        (nx_mesh - 1, "oben-rechts (row-major)"),
                        ((ny_mesh - 1) * nx_mesh, "unten-links (row-major)"),
                        ((ny_mesh - 1) * nx_mesh + nx_mesh - 1, "unten-rechts (row-major)"),
                    ]
                    corner_indices_col_major = [
                        (0, "oben-links (col-major)"),
                        ((ny_mesh - 1), "oben-rechts (col-major)"),
                        ((nx_mesh - 1) * ny_mesh, "unten-links (col-major)"),
                        ((nx_mesh - 1) * ny_mesh + ny_mesh - 1, "unten-rechts (col-major)"),
                    ]
                    
                    print(f"    Mesh corner points (row-major indexing):")
                    for idx, name in corner_indices_row_major:
                        if idx < len(points):
                            print(f"      {name} [idx={idx}]: ({points[idx,0]:.2f}, {points[idx,1]:.2f}, {points[idx,2]:.2f})")
                            jj = idx // nx_mesh
                            ii = idx % nx_mesh
                            print(f"        Expected from arrays [jj={jj},ii={ii}]: X={X[jj,ii]:.2f}, Y={Y[jj,ii]:.2f}, Z={Z[jj,ii]:.2f}")
                    
                    print(f"    Mesh corner points (col-major indexing):")
                    for idx, name in corner_indices_col_major:
                        if idx < len(points):
                            print(f"      {name} [idx={idx}]: ({points[idx,0]:.2f}, {points[idx,1]:.2f}, {points[idx,2]:.2f})")
                            ii = idx // ny_mesh
                            jj = idx % ny_mesh
                            print(f"        Expected from arrays [jj={jj},ii={ii}]: X={X[jj,ii]:.2f}, Y={Y[jj,ii]:.2f}, Z={Z[jj,ii]:.2f}")
                    
                    # Prüfe, ob PyVista die Arrays transponiert hat
                    # Suche nach dem Punkt, der X[0,0], Y[0,0], Z[0,0] entspricht
                    target_x, target_y, target_z = X[0,0], Y[0,0], Z[0,0]
                    matches = np.where(
                        (np.abs(points[:,0] - target_x) < 1e-6) &
                        (np.abs(points[:,1] - target_y) < 1e-6) &
                        (np.abs(points[:,2] - target_z) < 1e-6)
                    )[0]
                    if len(matches) > 0:
                        found_idx = matches[0]
                        print(f"    Found point matching X[0,0],Y[0,0],Z[0,0] at mesh index: {found_idx}")
                        # Versuche beide Indizierungs-Methoden
                        jj_row = found_idx // nx_mesh
                        ii_row = found_idx % nx_mesh
                        ii_col = found_idx // ny_mesh
                        jj_col = found_idx % ny_mesh
                        print(f"      Row-major interpretation: jj={jj_row}, ii={ii_row} -> X={X[jj_row,ii_row]:.2f}, Y={Y[jj_row,ii_row]:.2f}")
                        print(f"      Col-major interpretation: jj={jj_col}, ii={ii_col} -> X={X[jj_col,ii_col]:.2f}, Y={Y[jj_col,ii_col]:.2f}")
            
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
                    
                    # 🐛 DEBUG: Textur-Koordinaten vor Transformation
                    if DEBUG_PLOT3D_TIMING:
                        ny_img, nx_img = img_rgba.shape[:2]
                        print(f"[DEBUG Texture] {surface_id}:")
                        print(f"  img_rgba.shape = {img_rgba.shape} (H={ny_img}, W={nx_img})")
                        print(f"  grid bounds: x=[{xmin_grid:.2f}, {xmax_grid:.2f}], y=[{ymin_grid:.2f}, {ymax_grid:.2f}]")
                        print(f"  u_coords range: [{u_coords.min():.3f}, {u_coords.max():.3f}]")
                        print(f"  v_coords_raw range: [{v_coords_raw.min():.3f}, {v_coords_raw.max():.3f}]")
                        print(f"  v_coords range (after 1.0-v): [{v_coords.min():.3f}, {v_coords.max():.3f}]")
                        # Beispiel-Koordinaten an Ecken und Mitte
                        print(f"  Corner [0,0]: u={u_coords[0,0]:.3f}, v={v_coords[0,0]:.3f}, world=({X[0,0]:.2f}, {Y[0,0]:.2f}), SPL={spl_img[0,0]:.1f} dB")
                        print(f"  Corner [0,nx-1]: u={u_coords[0,nx_img-1]:.3f}, v={v_coords[0,nx_img-1]:.3f}, world=({X[0,nx_img-1]:.2f}, {Y[0,nx_img-1]:.2f}), SPL={spl_img[0,nx_img-1]:.1f} dB")
                        print(f"  Corner [ny-1,0]: u={u_coords[ny_img-1,0]:.3f}, v={v_coords[ny_img-1,0]:.3f}, world=({X[ny_img-1,0]:.2f}, {Y[ny_img-1,0]:.2f}), SPL={spl_img[ny_img-1,0]:.1f} dB")
                        print(f"  Corner [ny-1,nx-1]: u={u_coords[ny_img-1,nx_img-1]:.3f}, v={v_coords[ny_img-1,nx_img-1]:.3f}, world=({X[ny_img-1,nx_img-1]:.2f}, {Y[ny_img-1,nx_img-1]:.2f}), SPL={spl_img[ny_img-1,nx_img-1]:.1f} dB")
                        mid_y, mid_x = ny_img // 2, nx_img // 2
                        print(f"  Center [{mid_y},{mid_x}]: u={u_coords[mid_y,mid_x]:.3f}, v={v_coords[mid_y,mid_x]:.3f}, world=({X[mid_y,mid_x]:.2f}, {Y[mid_y,mid_x]:.2f}), SPL={spl_img[mid_y,mid_x]:.1f} dB")
                        # Zusätzliche Info: Bild-Array-Index vs. Textur-Koordinate
                        print(f"  Mapping: img_rgba[jj,ii] -> world(X[jj,ii], Y[jj,ii]) -> texture(u[jj,ii], v[jj,ii])")
                        print(f"  img_rgba[0,0] = Bild oben-links -> world({X[0,0]:.2f}, {Y[0,0]:.2f}) -> texture({u_coords[0,0]:.3f}, {v_coords[0,0]:.3f}) [SPL={spl_img[0,0]:.1f}->{spl_clipped[0,0]:.1f} dB]")
                        print(f"  img_rgba[ny-1,nx-1] = Bild unten-rechts -> world({X[ny_img-1,nx_img-1]:.2f}, {Y[ny_img-1,nx_img-1]:.2f}) -> texture({u_coords[ny_img-1,nx_img-1]:.3f}, {v_coords[ny_img-1,nx_img-1]:.3f}) [SPL={spl_img[ny_img-1,nx_img-1]:.1f}->{spl_clipped[ny_img-1,nx_img-1]:.1f} dB]")
                    
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
                    
                    # 🐛 DEBUG: Transformations-Einstellungen
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  Transform settings: invert_x={invert_x}, invert_y={invert_y}, swap_axes={swap_axes}")
                    
                    img_rgba_final = img_rgba.copy()
                    if invert_x:
                        img_rgba_final = np.fliplr(img_rgba_final)
                        u_coords = 1.0 - u_coords
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: fliplr (horizontal)")
                    if invert_y:
                        img_rgba_final = np.flipud(img_rgba_final)
                        v_coords = 1.0 - v_coords
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: flipud (vertical)")
                    if swap_axes:
                        img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                        u_coords, v_coords = v_coords.copy(), u_coords.copy()
                        if DEBUG_PLOT3D_TIMING:
                            print(f"  Applied: transpose (swap axes)")
                    
                    # 🐛 DEBUG: Textur-Koordinaten nach Transformation
                    if DEBUG_PLOT3D_TIMING:
                        ny_final, nx_final = img_rgba_final.shape[:2]
                        print(f"  After transform: img_rgba_final.shape = {img_rgba_final.shape}")
                        print(f"  Final u_coords range: [{u_coords.min():.3f}, {u_coords.max():.3f}]")
                        print(f"  Final v_coords range: [{v_coords.min():.3f}, {v_coords.max():.3f}]")
                        print(f"  Final Corner [0,0]: u={u_coords[0,0]:.3f}, v={v_coords[0,0]:.3f}")
                        print(f"  Final Corner [0,nx-1]: u={u_coords[0,nx_final-1]:.3f}, v={v_coords[0,nx_final-1]:.3f}")
                        print(f"  Final Corner [ny-1,0]: u={u_coords[ny_final-1,0]:.3f}, v={v_coords[ny_final-1,0]:.3f}")
                        print(f"  Final Corner [ny-1,nx-1]: u={u_coords[ny_final-1,nx_final-1]:.3f}, v={v_coords[ny_final-1,nx_final-1]:.3f}")
                        print(f"  Final Center [{mid_y},{mid_x}]: u={u_coords[mid_y,mid_x]:.3f}, v={v_coords[mid_y,mid_x]:.3f}")
                    
                    # 🎯 WICHTIG: PyVista StructuredGrid verwendet column-major (Fortran-style) Indizierung!
                    # Die Arrays müssen in column-major Reihenfolge flach gemacht werden
                    # row-major: idx = jj * nx + ii
                    # col-major: idx = ii * ny + jj
                    # Verwende order='F' (Fortran-style) für column-major Reihenfolge
                    t_coords = np.column_stack((
                        u_coords.ravel(order='F'),
                        v_coords.ravel(order='F')
                    ))
                    
                    # 🐛 DEBUG: Textur-Mapping-Prüfung
                    if DEBUG_PLOT3D_TIMING:
                        print(f"  [DEBUG Texture Mapping] {surface_id}:")
                        print(f"    Texture image shape: {img_rgba_final.shape}")
                        print(f"    Texture coordinates shape: {t_coords.shape}")
                        print(f"    Mesh points shape: {grid.points.shape if grid.points is not None else 'None'}")
                        # Prüfe Mapping an Ecken
                        ny_tex, nx_tex = img_rgba_final.shape[:2]
                        print(f"    Texture array dimensions: ny_tex={ny_tex}, nx_tex={nx_tex}")
                        print(f"    Mesh array dimensions: ny_mesh={ny_mesh}, nx_mesh={nx_mesh}")
                        
                        # Prüfe, ob die Dimensionen übereinstimmen
                        if ny_tex != ny_mesh or nx_tex != nx_mesh:
                            print(f"    ⚠️  WARNING: Dimension mismatch! Texture: ({ny_tex}, {nx_tex}), Mesh arrays: ({ny_mesh}, {nx_mesh})")
                        
                        corner_mappings = [
                            (0, 0, "oben-links"),
                            (0, nx_tex-1, "oben-rechts"),
                            (ny_tex-1, 0, "unten-links"),
                            (ny_tex-1, nx_tex-1, "unten-rechts"),
                        ]
                        for jj, ii, name in corner_mappings:
                            # 🎯 WICHTIG: Textur-Koordinaten sind jetzt in column-major Reihenfolge!
                            # row-major: idx = jj * nx + ii
                            # col-major: idx = ii * ny + jj
                            tex_idx_row = jj * nx_tex + ii  # Für Debug-Vergleich
                            tex_idx_col = ii * ny_tex + jj  # Tatsächlicher Index (column-major)
                            
                            if tex_idx_col < len(t_coords):
                                u_val = t_coords[tex_idx_col, 0]
                                v_val = t_coords[tex_idx_col, 1]
                                # Finde entsprechenden Mesh-Punkt (PyVista verwendet column-major)
                                mesh_idx_col = ii * ny_tex + jj  # PyVista StructuredGrid verwendet column-major
                                
                                print(f"    Corner {name} [tex_jj={jj},tex_ii={ii}]:")
                                print(f"      Texture coord index (col-major): {tex_idx_col} (row-major wäre: {tex_idx_row})")
                                print(f"      Texture coord: u={u_val:.3f}, v={v_val:.3f}")
                                print(f"      Texture pixel value: img_rgba_final[{jj},{ii}] = {img_rgba_final[jj,ii,:3]}")
                                
                                # Prüfe Mesh-Punkt (PyVista verwendet column-major)
                                if mesh_idx_col < len(grid.points):
                                    mesh_point = grid.points[mesh_idx_col]
                                    print(f"      Mesh point (col-major idx={mesh_idx_col}): ({mesh_point[0]:.2f}, {mesh_point[1]:.2f}, {mesh_point[2]:.2f})")
                                    print(f"        Expected from arrays [jj={jj},ii={ii}]: X={X[jj,ii]:.2f}, Y={Y[jj,ii]:.2f}, Z={Z[jj,ii]:.2f}")
                                    match = (abs(mesh_point[0] - X[jj,ii]) < 1e-3 and 
                                             abs(mesh_point[1] - Y[jj,ii]) < 1e-3)
                                    print(f"        Match: {match}")
                                    if not match:
                                        print(f"        ⚠️  WARNING: Texture coordinate does not match mesh point!")
                                
                                # Prüfe ob Textur-Koordinaten im gültigen Bereich sind
                                if u_val < 0 or u_val > 1 or v_val < 0 or v_val > 1:
                                    print(f"      ⚠️  WARNING: Texture coordinates out of range!")
                    
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

    # _remove_actor ist jetzt in SPL3DHelpers (Mixin)

    # Texture-Metadaten-Methoden sind jetzt in SPL3DPlotRenderer (Mixin)
    # get_texture_metadata, get_texture_world_coords, get_world_coords_to_texture_coords

    # ------------------------------------------------------------------
    # Debug-Funktionen entfernt - nur Texture-Rendering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Hilfsfunktionen für Farben/Skalierung
    # ------------------------------------------------------------------
    # Statische Hilfsmethoden sind jetzt in Plot3DHelpers.py
    # _quantize_to_steps, _fallback_cmap_to_lut, _to_float_array werden als Module-Level-Funktionen bereitgestellt
    # _compute_overlay_signatures ist jetzt in SPL3DHelpers (Mixin) - wird durch Vererbung bereitgestellt




__all__ = ['DrawSPLPlot3D']


