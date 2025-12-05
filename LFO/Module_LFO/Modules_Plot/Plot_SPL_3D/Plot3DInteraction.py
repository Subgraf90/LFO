"""Event-Handling und Picking für den 3D-SPL-Plot."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

try:
    import pyvista as pv
except Exception:
    pv = None  # type: ignore

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    derive_surface_plane,
)


class SPL3DInteractionHandler:
    """
    Mixin-Klasse für Event-Handling und Picking im 3D-SPL-Plot.

    Diese Klasse kapselt alle Methoden, die direkt mit Event-Handling,
    Picking (Speaker, Surfaces) und UI-Interaktionen zu tun haben.

    Sie erwartet, dass die aufnehmende Klasse folgende Attribute bereitstellt:
    - `plotter` (PyVista Plotter)
    - `widget` (Qt Widget)
    - `overlay_axis` (SPL3DOverlayAxis)
    - `overlay_surfaces` (SPL3DOverlaySurfaces)
    - `overlay_impulse` (SPL3DOverlayImpulse)
    - `overlay_speakers` (SPL3DSpeakerMixin)
    - `settings` (Settings-Objekt)
    - `container` (Container-Objekt)
    - `main_window` (MainWindow-Objekt)
    - `parent_widget` (Parent Widget)
    - `functions` (FunctionToolbox)
    - `surface_mesh` (PyVista Mesh, optional)
    - `_vertical_surface_meshes` (dict)
    - `SURFACE_NAME` (str)
    - Event-State-Variablen: `_pan_active`, `_rotate_active`, `_axis_drag_active`, etc.
    - Kamera-Methoden: `_rotate_camera`, `_pan_camera`, `_save_camera_state` (aus SPL3DCameraController)
    - Render-Methode: `render()`
    - Overlay-Methode: `update_overlays()`
    """

    # ------------------------------------------------------------------
    # Event-Handling
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: PLR0911
        if obj is self.widget:
            etype = event.type()
            # 🎯 DOPPELKLICK KOMPLETT SPERREN - Fange Doppelklick-Events ab, bevor PyVista sie verarbeitet
            if etype == QtCore.QEvent.MouseButtonDblClick:
                event.accept()
                return True  # Event abgefangen, PyVista verarbeitet es nicht
            
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                press_pos = QtCore.QPoint(event.pos())

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
            axis_actor_names = self.overlay_axis._category_actors.get('axis', [])
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
            axis_actor_names = self.overlay_axis._category_actors.get('axis', [])
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
                if not (hasattr(self, "overlay_speakers") and hasattr(self.overlay_speakers, "_speaker_actor_cache")):
                    return (None, None, None)

                speaker_cache = self.overlay_speakers._speaker_actor_cache
                if not isinstance(speaker_cache, dict) or not speaker_cache:
                    return (None, None, None)

                picker = vtkCellPicker()
                # Toleranz reduziert, um Flächen hinter dem Speaker weniger oft zu treffen.
                # Vorher: 0.08 – jetzt 0.03 (präziseres Picking).
                picker.SetTolerance(0.03)
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
                speaker_info = self.overlay_speakers._get_speaker_info_from_actor(picked_actor, picked_name)
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

__all__ = ['SPL3DInteractionHandler']
