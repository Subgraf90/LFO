"""Event-Handling für den 3D-SPL-Plot."""

from __future__ import annotations

import time
from typing import Optional, Callable, Any

from PyQt5 import QtCore, QtWidgets


class EventHandler:
    """Verwaltet Mouse-Events für den 3D-Plot."""
    
    def __init__(
        self,
        widget: Any,
        plotter: Any,
        pick_speaker_func: Callable,
        pick_surface_func: Callable,
        pick_axis_line_func: Callable,
        select_speaker_func: Callable,
        select_surface_func: Callable,
        handle_axis_line_drag_func: Callable,
        handle_mouse_move_func: Callable,
        pan_camera_func: Callable,
        rotate_camera_func: Callable,
        save_camera_state_func: Callable,
        main_window: Any = None,
    ):
        """Initialisiert den Event-Handler.
        
        Args:
            widget: Widget für Events
            plotter: PyVista Plotter
            pick_speaker_func: Funktion zum Picken von Speakers
            pick_surface_func: Funktion zum Picken von Surfaces
            pick_axis_line_func: Funktion zum Picken von Axis-Lines
            select_speaker_func: Funktion zum Auswählen von Speakers
            select_surface_func: Funktion zum Auswählen von Surfaces
            handle_axis_line_drag_func: Funktion zum Behandeln von Axis-Line-Drags
            handle_mouse_move_func: Funktion zum Behandeln von Mouse-Move
            pan_camera_func: Funktion zum Pan der Kamera
            rotate_camera_func: Funktion zum Rotieren der Kamera
            save_camera_state_func: Funktion zum Speichern des Camera-States
            main_window: Main-Window (optional)
        """
        self.widget = widget
        self.plotter = plotter
        self._pick_speaker = pick_speaker_func
        self._pick_surface = pick_surface_func
        self._pick_axis_line = pick_axis_line_func
        self._select_speaker = select_speaker_func
        self._select_surface = select_surface_func
        self._handle_axis_line_drag = handle_axis_line_drag_func
        self._handle_mouse_move = handle_mouse_move_func
        self._pan_camera = pan_camera_func
        self._rotate_camera = rotate_camera_func
        self._save_camera_state = save_camera_state_func
        self.main_window = main_window
        
        # Event-State-Variablen
        self._pan_active = False
        self._pan_last_pos: Optional[QtCore.QPoint] = None
        self._rotate_active = False
        self._rotate_last_pos: Optional[QtCore.QPoint] = None
        self._click_start_pos: Optional[QtCore.QPoint] = None
        self._last_click_time: Optional[float] = None
        self._last_click_pos: Optional[QtCore.QPoint] = None
        self._last_press_time: Optional[float] = None
        self._last_press_pos: Optional[QtCore.QPoint] = None
        self._axis_selected: Optional[str] = None
        self._axis_drag_active = False
        self._axis_drag_type: Optional[str] = None
        self._axis_drag_start_pos: Optional[QtCore.QPoint] = None
        self._axis_drag_start_value: Optional[float] = None
    
    def eventFilter(self, obj, event) -> bool:
        """Haupt-Event-Filter für Mouse-Events.
        
        Args:
            obj: Event-Quell-Objekt
            event: Qt Event
            
        Returns:
            True wenn Event behandelt wurde, sonst False
        """
        if obj is self.widget:
            etype = event.type()
            
            # Doppelklick komplett sperren
            if etype == QtCore.QEvent.MouseButtonDblClick:
                event.accept()
                return True
            
            # Linksklick-Press
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                press_pos = QtCore.QPoint(event.pos())
                self._click_start_pos = press_pos
                self._last_press_time = time.time()
                self._last_press_pos = press_pos
                self._rotate_last_pos = press_pos
                event.accept()
                return True
            
            # Linksklick-Release
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                click_pos = QtCore.QPoint(event.pos())
                move_dx = abs(click_pos.x() - self._click_start_pos.x()) if self._click_start_pos is not None else 0
                move_dy = abs(click_pos.y() - self._click_start_pos.y()) if self._click_start_pos is not None else 0
                click_threshold = 6
                is_click = (
                    self._click_start_pos is not None
                    and move_dx < click_threshold
                    and move_dy < click_threshold
                )
                
                # Rotation beenden
                self._rotate_active = False
                self._rotate_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                
                # Wenn es ein Klick war, prüfe Speaker und Surfaces
                if is_click:
                    speaker_t, speaker_array_id, speaker_index = self._pick_speaker(click_pos)
                    surface_t, surface_id = self._pick_surface(click_pos)
                    
                    # Vergleiche Distanzen und wähle näheres Element
                    if speaker_t is not None and surface_t is not None:
                        depth_eps = 0.5
                        if speaker_t <= surface_t or abs(speaker_t - surface_t) <= depth_eps:
                            self._select_speaker(speaker_array_id, speaker_index)
                        else:
                            self._select_surface(surface_id)
                    elif speaker_t is not None:
                        self._select_speaker(speaker_array_id, speaker_index)
                    elif surface_id is not None:
                        self._select_surface(surface_id)
                    
                    self._last_click_time = time.time()
                    self._last_click_pos = QtCore.QPoint(click_pos)
                    self._click_start_pos = None
                else:
                    self._click_start_pos = None
                
                # Beende Achsenlinien-Drag
                if self._axis_drag_active:
                    self._axis_drag_active = False
                    self._axis_drag_start_pos = None
                    self._axis_drag_start_value = None
                    if self._axis_selected is not None:
                        self.widget.setCursor(QtCore.Qt.PointingHandCursor)
                    else:
                        self.widget.unsetCursor()
                
                event.accept()
                return True
            
            # Rechtsklick-Press (Pan)
            if etype == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
                self._pan_active = True
                self._pan_last_pos = QtCore.QPoint(event.pos())
                self.widget.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
                return True
            
            # Rechtsklick-Release
            if etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                self._pan_active = False
                self._pan_last_pos = None
                self.widget.unsetCursor()
                self._save_camera_state()
                event.accept()
                return True
            
            # Mouse-Move mit Linksklick (Rotation)
            if etype == QtCore.QEvent.MouseMove and event.buttons() & QtCore.Qt.LeftButton:
                if self._rotate_last_pos is not None and not self._rotate_active:
                    current_pos = QtCore.QPoint(event.pos())
                    move_distance = ((current_pos.x() - self._rotate_last_pos.x()) ** 2 + 
                                    (current_pos.y() - self._rotate_last_pos.y()) ** 2) ** 0.5
                    if move_distance >= 3:
                        self._rotate_active = True
                        self.widget.setCursor(QtCore.Qt.OpenHandCursor)
                
                if self._rotate_active and self._rotate_last_pos is not None and self._axis_selected is None:
                    current_pos = QtCore.QPoint(event.pos())
                    delta = current_pos - self._rotate_last_pos
                    self._rotate_last_pos = current_pos
                    self._rotate_camera(delta.x(), delta.y())
                    event.accept()
                    return True
                
                if self._axis_selected is not None and self._rotate_active:
                    self._rotate_active = False
                    self._rotate_last_pos = None
            
            # Mouse-Move mit Rechtsklick (Pan)
            if etype == QtCore.QEvent.MouseMove and self._pan_active and self._pan_last_pos is not None:
                current_pos = QtCore.QPoint(event.pos())
                delta = current_pos - self._pan_last_pos
                self._pan_last_pos = current_pos
                self._pan_camera(delta.x(), delta.y())
                event.accept()
                return True
            
            # Mouse-Move für Axis-Line-Drag
            if etype == QtCore.QEvent.MouseMove and self._axis_drag_active and self._axis_drag_start_pos is not None:
                current_pos = QtCore.QPoint(event.pos())
                self._handle_axis_line_drag(current_pos)
                event.accept()
                return True
            
            # Mouse-Move ohne Pan/Rotate
            if etype == QtCore.QEvent.MouseMove and not self._pan_active and not self._rotate_active and not self._axis_drag_active:
                self._handle_mouse_move(event.pos())
            
            # Leave-Event
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
                
                # Leere Mauspositions-Anzeige
                if self.main_window and hasattr(self.main_window, 'ui') and hasattr(self.main_window.ui, 'mouse_position_label'):
                    self.main_window.ui.mouse_position_label.setText("")
            
            # Wheel-Event
            if etype == QtCore.QEvent.Wheel:
                QtCore.QTimer.singleShot(0, self._save_camera_state)
        
        return False
    
    def set_axis_selected(self, axis: Optional[str]):
        """Setzt die ausgewählte Achse.
        
        Args:
            axis: 'x', 'y' oder None
        """
        self._axis_selected = axis
    
    def set_axis_drag_active(self, active: bool, drag_type: Optional[str] = None):
        """Setzt den Axis-Drag-Status.
        
        Args:
            active: Ob Drag aktiv ist
            drag_type: 'x' oder 'y'
        """
        self._axis_drag_active = active
        self._axis_drag_type = drag_type

