"""UI-Komponenten für den 3D-SPL-Plot."""

from __future__ import annotations

import math
from typing import Any, Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from ..PlotSPL3DOverlays.ui_time_control import SPLTimeControlBar


def setup_view_controls(
    widget: Any,
    create_view_button_func: Callable,
    set_view_top_func: Callable,
    set_view_side_x_func: Callable,
    set_view_side_y_func: Callable,
) -> QtWidgets.QFrame:
    """Erstellt die View-Control-Buttons.
    
    Args:
        widget: Widget für die Buttons
        create_view_button_func: Funktion zum Erstellen von View-Buttons
        set_view_top_func: Callback für Top-Ansicht
        set_view_side_x_func: Callback für X-Seitenansicht
        set_view_side_y_func: Callback für Y-Seitenansicht
        
    Returns:
        QFrame mit View-Controls
    """
    if widget is None:
        return None

    view_control_widget = QtWidgets.QFrame(widget)
    view_control_widget.setObjectName('pv_view_controls')
    view_control_widget.setStyleSheet(
        "QFrame#pv_view_controls {"
        "background-color: rgba(255, 255, 255, 200);"
        "border-radius: 6px;"
        "border: 1px solid rgba(0, 0, 0, 100);"
        "}"
    )
    view_control_widget.setFrameShape(QtWidgets.QFrame.NoFrame)
    view_control_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

    layout = QtWidgets.QHBoxLayout(view_control_widget)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)

    btn_view_top = create_view_button_func('top', "Ansicht von oben", set_view_top_func)
    btn_view_left = create_view_button_func('side_x', "Ansicht von links", set_view_side_x_func)
    btn_view_right = create_view_button_func('side_y', "Ansicht von rechts", set_view_side_y_func)

    for btn in (btn_view_top, btn_view_left, btn_view_right):
        layout.addWidget(btn)

    view_control_widget.adjustSize()
    view_control_widget.move(3, 3)
    view_control_widget.show()
    view_control_widget.raise_()
    
    return view_control_widget


def setup_time_control(widget: Any, time_slider_change_handler: Callable) -> SPLTimeControlBar | None:
    """Erstellt das Time-Control-Widget.
    
    Args:
        widget: Parent-Widget
        time_slider_change_handler: Handler für Time-Slider-Änderungen
        
    Returns:
        SPLTimeControlBar oder None bei Fehler
    """
    try:
        time_control = SPLTimeControlBar(widget)
        time_control.hide()
        time_control.valueChanged.connect(time_slider_change_handler)
        return time_control
    except Exception:
        return None


def create_view_button(
    widget: Any,
    create_axis_pixmap_func: Callable,
    orientation: str,
    tooltip: str,
    callback: Callable,
) -> QtWidgets.QToolButton:
    """Erstellt einen View-Button.
    
    Args:
        widget: Parent-Widget
        create_axis_pixmap_func: Funktion zum Erstellen von Axis-Pixmaps
        orientation: Orientierung ('top', 'side_x', 'side_y')
        tooltip: Tooltip-Text
        callback: Callback-Funktion
        
    Returns:
        QToolButton
    """
    button = QtWidgets.QToolButton(widget)
    button.setIcon(QtGui.QIcon(create_axis_pixmap_func(orientation)))
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


def create_axis_pixmap(orientation: str, size: int = 28) -> QtGui.QPixmap:
    """Erstellt ein Pixmap für eine Axis-Orientierung.
    
    Args:
        orientation: Orientierung ('iso', 'top', 'side_x', 'side_y', 'side_x_flip')
        size: Größe des Pixmaps
        
    Returns:
        QPixmap mit Axis-Darstellung
    """
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

