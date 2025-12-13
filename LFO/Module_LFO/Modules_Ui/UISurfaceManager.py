from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QTreeWidget, QTreeWidgetItem, QCheckBox, QPushButton, 
    QTabWidget, QSizePolicy, QGridLayout, QLabel, QLineEdit, 
    QMenu, QAbstractItemView, QGroupBox, QScrollArea, QColorDialog
)
from PyQt5.QtCore import Qt, QPoint, QObject, QEvent
from PyQt5.QtGui import QDoubleValidator, QColor
from PyQt5 import QtWidgets, QtGui

import numpy as np
from typing import Dict, Optional, List
import uuid

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    SurfaceGroup,
    derive_surface_plane,
    evaluate_surface_plane,
)
from Module_LFO.Modules_Data.SurfaceValidator import validate_and_optimize_surface, triangulate_points
import logging


class UISurfaceManager(ModuleBase):
    """
    Verwaltet das SurfaceDockWidget mit TreeWidget für Surfaces/Gruppen
    und TabWidget für Surface-Parameter.
    Layout und Funktionalität basierend auf UiSourceManagement.
    """
    
    def __init__(self, main_window, settings, container):
        super().__init__(settings)
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.surface_dock_widget = None
        self._group_controller = _SurfaceGroupController(self.settings)
        self._group_controller.ensure_structure()
        
        # Speicher für ursprüngliche Positionen von Surfaces in Gruppen
        self.surface_groups = {}  # {group_id: {'name': str, 'enabled': bool, 'hidden': bool, 'child_surface_ids': [str], 'original_surface_positions': {surface_id: {'x': float, 'y': float, 'z': float}}}}
        
        # Flag für das Laden von Punkten (verhindert Signal-Loops)
        self._loading_points = False
        
    def is_autocalc_active(self):
        """Prüft, ob automatische Berechnung aktiv ist."""
        return getattr(self.settings, "update_pressure_soundfield", True)
    
    def calculate_single_surface(self, surface_id):
        """
        Berechnet nur ein einzelnes Surface neu, ohne andere Surfaces neu zu berechnen.
        Behält bestehende Daten für andere Surfaces bei.
        """
        if not self.is_autocalc_active():
            return
        
        # Prüfe ob bereits SPL-Daten vorhanden sind
        has_existing_data = (
            hasattr(self.container, 'calculation_spl') and
            isinstance(self.container.calculation_spl, dict) and
            'surface_grids' in self.container.calculation_spl and
            'surface_results' in self.container.calculation_spl
        )
        
        # Auch wenn keine Daten vorhanden sind, berechne nur dieses einzelne Surface
        # (nicht alle enabled Surfaces)
        
        # Berechne nur das neue Surface
        try:
            from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator
            from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator
            
            # Hole das Surface
            surface = self._get_surface(surface_id)
            if not surface:
                return
            
            # Prüfe ob Surface enabled und nicht hidden ist
            if isinstance(surface, SurfaceDefinition):
                is_enabled = surface.enabled
                is_hidden = surface.hidden
            else:
                is_enabled = surface.get('enabled', False)
                is_hidden = surface.get('hidden', False)
            
            if not is_enabled or is_hidden:
                return
            
            # Erstelle Calculator
            calculator = SoundFieldCalculator(
                self.settings,
                self.container.data,
                self.container.calculation_spl
            )
            calculator.set_data_container(self.container)
            
            # Erstelle Grid nur für dieses Surface
            grid_generator = FlexibleGridGenerator(self.settings)
            surface_data = surface.to_dict() if hasattr(surface, 'to_dict') else surface
            enabled_surfaces = [(surface_id, surface_data)]
            
            surface_grids = grid_generator.generate_per_surface(
                enabled_surfaces,
                resolution=self.settings.resolution,
                min_points_per_dimension=3
            )
            
            if not surface_grids or surface_id not in surface_grids:
                return
            
            # Berechne nur dieses Surface
            grid = surface_grids[surface_id]
            
            # Physikalische Konstanten
            speed_of_sound = self.settings.speed_of_sound
            wave_number = calculator.functions.wavenumber(speed_of_sound, self.settings.calculate_frequency)
            phys_constants = {
                'speed_of_sound': speed_of_sound,
                'wave_number': wave_number,
                'calculate_frequency': self.settings.calculate_frequency,
                'a_source_pa': calculator.functions.db2spl(calculator.functions.db2mag(self.settings.a_source_db)),
            }
            
            # Berechne Schallfeld für dieses Surface
            sound_field_p_surface, _ = calculator._calculate_sound_field_for_surface_grid(
                grid,
                phys_constants,
                capture_arrays=False
            )
            
            # Aktualisiere nur die Daten für dieses Surface
            if 'surface_grids' not in self.container.calculation_spl:
                self.container.calculation_spl['surface_grids'] = {}
            if 'surface_results' not in self.container.calculation_spl:
                self.container.calculation_spl['surface_results'] = {}
            
            # Speichere Grid-Daten
            self.container.calculation_spl['surface_grids'][surface_id] = {
                'sound_field_x': grid.sound_field_x.tolist(),
                'sound_field_y': grid.sound_field_y.tolist(),
                'X_grid': grid.X_grid.tolist(),
                'Y_grid': grid.Y_grid.tolist(),
                'Z_grid': grid.Z_grid.tolist(),
                'surface_mask': grid.surface_mask.astype(bool).tolist(),
                'resolution': grid.resolution,
            }
            
            result = {
                'sound_field_p': sound_field_p_surface,
                'sound_field_x': grid.sound_field_x,
                'sound_field_y': grid.sound_field_y,
                'X_grid': grid.X_grid,
                'Y_grid': grid.Y_grid,
                'Z_grid': grid.Z_grid,
                'surface_mask': grid.surface_mask,
            }
            
            # Speichere Ergebnis (konvertiere zu Listen für JSON-Kompatibilität)
            sound_field_p_complex = result['sound_field_p']
            self.container.calculation_spl['surface_results'][surface_id] = {
                'sound_field_p': np.array(sound_field_p_complex).tolist(),
                'sound_field_p_magnitude': np.abs(sound_field_p_complex).tolist(),
            }
            
            # Aktualisiere auch orientation in surface_grids
            if hasattr(grid, 'geometry') and hasattr(grid.geometry, 'orientation'):
                self.container.calculation_spl['surface_grids'][surface_id]['orientation'] = grid.geometry.orientation
            
            # Setze Signatur zurück, damit update_overlays (wird von plot_spl aufgerufen) die Änderung erkennt
            # und XY-Linien neu zeichnet (wenn xy_enabled=True)
            if (hasattr(self.main_window, 'draw_plots') and 
                hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                plotter = self.main_window.draw_plots.draw_spl_plotter
                if plotter:
                    # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                    if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                        plotter.overlay_axis._last_axis_state = None
                    # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                    if hasattr(plotter, '_last_overlay_signatures'):
                        # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                        if isinstance(plotter._last_overlay_signatures, dict):
                            plotter._last_overlay_signatures.pop('axis', None)
            
            # Aktualisiere Plot nur für dieses Surface
            # plot_spl ruft intern update_overlays auf, das jetzt die Signatur-Änderung erkennen wird
            if hasattr(self.main_window, 'plot_spl'):
                self.main_window.plot_spl(update_axes=False)
            
        except Exception as e:
            print(f"Fehler beim Berechnen eines einzelnen Surfaces: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: Berechne alle Surfaces neu
            if hasattr(self.main_window, 'update_speaker_array_calculations'):
                self.main_window.update_speaker_array_calculations()
        
    # ---- public API -------------------------------------------------
    
    def show_surface_dock_widget(self):
        """
        Erstellt (falls nötig) und zeigt das SurfaceDockWidget an.
        Layout und Größe gemäß SourceManagement.
        """
        if not hasattr(self, 'surface_tree_widget'):
            if hasattr(self, 'surface_tree_widget'):
                self.surface_tree_widget.blockSignals(True)
        
        if not hasattr(self, 'surface_dockWidget') or self.surface_dockWidget is None:
            self.surface_dockWidget = QDockWidget("Surfaces", self.main_window)
            
            # Event-Filter für resize-Events hinzufügen
            class SurfaceResizeFilter(QObject):
                def __init__(self, dock_widget):
                    super().__init__()
                    self.dock_widget = dock_widget
                
                def eventFilter(self, obj, event):
                    if obj == self.dock_widget and event.type() == QEvent.Resize:
                        # Keine spezielle Behandlung mehr, Event normal weiterreichen
                        return False
                    # Für alle anderen Events ebenfalls False zurückgeben (Standard-Handling)
                    return False
            
            self._surface_resize_filter = SurfaceResizeFilter(self.surface_dockWidget)
            self.surface_dockWidget.installEventFilter(self._surface_resize_filter)
            
            # Prüfe, ob Sources-DockWidget bereits existiert und tabbifiziere es
            sources_dock = None
            if hasattr(self.main_window, 'sources_instance') and self.main_window.sources_instance:
                sources_dock = getattr(self.main_window.sources_instance, 'sources_dockWidget', None)
            
            if sources_dock:
                # Tabbifiziere: Beide DockWidgets übereinander mit Tab-Buttons
                self.main_window.addDockWidget(Qt.BottomDockWidgetArea, self.surface_dockWidget)
                self.main_window.tabifyDockWidget(sources_dock, self.surface_dockWidget)
                # Aktiviere das Surface-DockWidget (wird als aktiver Tab angezeigt)
                self.surface_dockWidget.raise_()
            else:
                # Füge normal hinzu
                self.main_window.addDockWidget(Qt.BottomDockWidgetArea, self.surface_dockWidget)
            
            dock_content = QWidget()
            dock_layout = QVBoxLayout(dock_content)
            self.surface_dockWidget.setWidget(dock_content)
            
            splitter = QSplitter(Qt.Horizontal)
            dock_layout.addWidget(splitter)
            
            # Linke Seite mit TreeWidget und Buttons
            left_side_widget = QWidget()
            left_side_layout = QVBoxLayout(left_side_widget)
            left_side_layout.setContentsMargins(0, 0, 0, 0)
            splitter.addWidget(left_side_widget)
            
            # TreeWidget für Surfaces und Gruppen
            self.surface_tree_widget = QTreeWidget()
            self.surface_tree_widget.setHeaderLabels(["Surface Name", "E", "H", "XY"])
            
            # Drag & Drop Funktionalität (identisch zu SourceManagement)
            original_dropEvent = self.surface_tree_widget.dropEvent
            original_startDrag = self.surface_tree_widget.startDrag
            _pending_drag_items = []
            
            def custom_startDrag(supported_actions):
                """Speichert die ausgewählten Items vor dem Drag"""
                _pending_drag_items.clear()
                _pending_drag_items.extend(self.surface_tree_widget.selectedItems())
                original_startDrag(supported_actions)
            
            def custom_dropEvent(event):
                """Behandelt Drop-Events für Drag & Drop mit Gruppen und Surfaces"""
                drop_item = self.surface_tree_widget.itemAt(event.pos())
                indicator_pos = self.surface_tree_widget.dropIndicatorPosition()
                
                if not _pending_drag_items:
                    event.ignore()
                    _pending_drag_items.clear()
                    return
                
                # Trenne Surfaces und Gruppen
                dragged_surfaces = [item for item in _pending_drag_items 
                                   if item.data(0, Qt.UserRole + 1) == "surface"]
                dragged_groups = [item for item in _pending_drag_items 
                                 if item.data(0, Qt.UserRole + 1) == "group"]
                
                handled = False
                
                # Behandle Gruppen-Verschiebung
                if dragged_groups:
                    for group_item in dragged_groups:
                        group_id = group_item.data(0, Qt.UserRole)
                        
                        # Bestimme Ziel-Gruppe (None = Top-Level)
                        target_group_id = None
                        if drop_item:
                            drop_type = drop_item.data(0, Qt.UserRole + 1)
                            if drop_type == "group":
                                # Droppe auf Gruppe - verschiebe Gruppe als Child
                                target_group_id = drop_item.data(0, Qt.UserRole)
                            elif drop_type == "surface":
                                # Droppe auf Surface - verwende Parent-Gruppe
                                parent = drop_item.parent()
                                if parent:
                                    target_group_id = parent.data(0, Qt.UserRole)
                        
                        # Prüfe, ob Ziel-Gruppe nicht die Root-Gruppe ist
                        if target_group_id == self._group_controller.root_group_id:
                            target_group_id = None
                        
                        if group_id != target_group_id:
                            # Prüfe, ob Ziel-Gruppe nicht ein Child der zu verschiebenden Gruppe ist
                            target_group = self._group_controller.get_group(target_group_id)
                            if target_group:
                                # Prüfe, ob target_group ein Child von group_id ist
                                is_child = False
                                check_group = target_group
                                while check_group and check_group.parent_id:
                                    if check_group.parent_id == group_id:
                                        is_child = True
                                        break
                                    check_group = self._group_controller.get_group(check_group.parent_id)
                                
                                if not is_child:
                                    success = self._group_controller.move_surface_group(group_id, target_group_id)
                                    if success:
                                        handled = True
                
                # Behandle Surface-Verschiebung
                moved_surfaces = []  # Liste für verschobene Flächen (für visuelle Aktualisierung)
                if dragged_surfaces:
                    default_surface_id = getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default')
                    
                    # Prüfe, ob Default-Fläche verschoben werden soll (verhindern)
                    for item in dragged_surfaces:
                        surface_id = item.data(0, Qt.UserRole)
                        if isinstance(surface_id, dict):
                            surface_id = surface_id.get('id')
                        if surface_id == default_surface_id:
                            # Default-Fläche darf nicht verschoben werden
                            from PyQt5.QtWidgets import QMessageBox
                            QMessageBox.information(
                                self.surface_dockWidget,
                                "Action not possible",
                                "The default surface cannot be moved to a group."
                            )
                            event.ignore()
                            _pending_drag_items.clear()
                            return
                    
                    # Prüfe, ob auf eine Gruppe gedroppt wird
                    if drop_item and drop_item.data(0, Qt.UserRole + 1) == "group":
                        # Droppe auf Gruppe - füge Surfaces als Childs hinzu
                        target_group_id = drop_item.data(0, Qt.UserRole)
                        # Hole tatsächlichen Checkbox-Zustand der Gruppe (inkl. Tristate)
                        group_enable_checkbox = self.surface_tree_widget.itemWidget(drop_item, 1)
                        group_hide_checkbox = self.surface_tree_widget.itemWidget(drop_item, 2)
                        group_xy_checkbox = self.surface_tree_widget.itemWidget(drop_item, 3)
                        
                        # Bestimme Checkbox-Zustände (Checked, Unchecked oder PartiallyChecked)
                        group_state_enabled = group_enable_checkbox.checkState() if group_enable_checkbox else Qt.Unchecked
                        group_state_hidden = group_hide_checkbox.checkState() if group_hide_checkbox else Qt.Unchecked
                        group_state_xy = group_xy_checkbox.checkState() if group_xy_checkbox else Qt.Checked
                        
                        # Wenn PartiallyChecked, verwende Mehrheitszustand der Child-Items
                        if group_state_enabled == Qt.PartiallyChecked:
                            group_state_enabled = self._get_group_majority_state(drop_item, 1)
                        if group_state_hidden == Qt.PartiallyChecked:
                            group_state_hidden = self._get_group_majority_state(drop_item, 2)
                        if group_state_xy == Qt.PartiallyChecked:
                            group_state_xy = self._get_group_majority_state(drop_item, 3)
                        
                        # Speichere verschobene Flächen für visuelle Aktualisierung nach load_surfaces()
                        for item in dragged_surfaces:
                            surface_id = item.data(0, Qt.UserRole)
                            if isinstance(surface_id, dict):
                                surface_id = surface_id.get('id')
                            
                            # Verschiebe Surface zur Gruppe
                            self._group_controller.assign_surface_to_group(surface_id, target_group_id, create_missing=False)
                            
                            # Übernehme Checkbox-Zustand der Gruppe auf die Fläche
                            self.on_surface_enable_changed(surface_id, group_state_enabled, skip_calculations=True)
                            self.on_surface_hide_changed(surface_id, group_state_hidden, skip_calculations=True)
                            self.on_surface_xy_changed(surface_id, group_state_xy, skip_calculations=True)
                            
                            # Speichere für visuelle Aktualisierung
                            moved_surfaces.append((surface_id, group_state_enabled, group_state_hidden, group_state_xy))
                            
                            handled = True
                    elif indicator_pos == QAbstractItemView.OnItem and drop_item:
                        # Droppe auf ein Surface Item
                        parent = drop_item.parent()
                        if parent and parent.data(0, Qt.UserRole + 1) == "group":
                            # Ziel-Item ist bereits in einer Gruppe - füge zur bestehenden Gruppe hinzu
                            target_group_id = parent.data(0, Qt.UserRole)
                            # Hole tatsächlichen Checkbox-Zustand der Gruppe (inkl. Tristate)
                            group_enable_checkbox = self.surface_tree_widget.itemWidget(parent, 1)
                            group_hide_checkbox = self.surface_tree_widget.itemWidget(parent, 2)
                            group_xy_checkbox = self.surface_tree_widget.itemWidget(parent, 3)
                            
                            # Bestimme Checkbox-Zustände (Checked, Unchecked oder PartiallyChecked)
                            group_state_enabled = group_enable_checkbox.checkState() if group_enable_checkbox else Qt.Unchecked
                            group_state_hidden = group_hide_checkbox.checkState() if group_hide_checkbox else Qt.Unchecked
                            group_state_xy = group_xy_checkbox.checkState() if group_xy_checkbox else Qt.Checked
                            
                            # Wenn PartiallyChecked, verwende Mehrheitszustand der Child-Items
                            if group_state_enabled == Qt.PartiallyChecked:
                                group_state_enabled = self._get_group_majority_state(parent, 1)
                            if group_state_hidden == Qt.PartiallyChecked:
                                group_state_hidden = self._get_group_majority_state(parent, 2)
                            if group_state_xy == Qt.PartiallyChecked:
                                group_state_xy = self._get_group_majority_state(parent, 3)
                            
                            # Speichere verschobene Flächen für visuelle Aktualisierung nach load_surfaces()
                            for item in dragged_surfaces:
                                surface_id = item.data(0, Qt.UserRole)
                                if isinstance(surface_id, dict):
                                    surface_id = surface_id.get('id')
                                self._group_controller.assign_surface_to_group(surface_id, target_group_id, create_missing=False)
                                
                                # Übernehme Checkbox-Zustand der Gruppe auf die Fläche
                                self.on_surface_enable_changed(surface_id, group_state_enabled, skip_calculations=True)
                                self.on_surface_hide_changed(surface_id, group_state_hidden, skip_calculations=True)
                                self.on_surface_xy_changed(surface_id, group_state_xy, skip_calculations=True)
                                
                                # Speichere für visuelle Aktualisierung
                                moved_surfaces.append((surface_id, group_state_enabled, group_state_hidden, group_state_xy))
                                
                                handled = True
                    elif indicator_pos in (QAbstractItemView.AboveItem, QAbstractItemView.BelowItem):
                        # Droppe oberhalb/unterhalb - entferne aus Gruppe (als Top-Level-Item)
                        for item in dragged_surfaces:
                            surface_id = item.data(0, Qt.UserRole)
                            if isinstance(surface_id, dict):
                                surface_id = surface_id.get('id')
                            # Entferne aus Gruppe (setze group_id auf None)
                            self._group_controller.assign_surface_to_group(surface_id, None, create_missing=False)
                            handled = True
                
                if handled:
                    # Stelle sicher, dass Gruppen-Struktur aktuell ist
                    self._group_controller.ensure_structure()
                    # Lade TreeWidget neu
                    self.load_surfaces()
                    
                    # Führe visuelle Aktualisierung für verschobene Flächen durch
                    # (wichtig für Hide- und Enable-Zustand, die mit skip_calculations=True gesetzt wurden)
                    if moved_surfaces:
                        for surface_id, state_enabled, state_hidden, state_xy in moved_surfaces:
                            # Hide-Zustand immer visuell aktualisieren (verstecken/zeigen)
                            self.on_surface_hide_changed(surface_id, state_hidden, skip_calculations=False)
                            
                            # Enable-Zustand immer visuell aktualisieren (entfernt/zeigt SPL Plot)
                            # Wichtig: Auch wenn Hide aktiv ist, muss Enable aktualisiert werden,
                            # um den SPL Plot zu entfernen/anzeigen
                            self.on_surface_enable_changed(surface_id, state_enabled, skip_calculations=False)
                    
                    event.accept()
                    event.setDropAction(Qt.MoveAction)
                else:
                    # Standard Drag & Drop Verhalten
                    original_dropEvent(event)
                
                # Validiere alle Checkboxen nach Drag & Drop
                self.validate_all_checkboxes()
                
                # Passe Spaltenbreite an den Inhalt an
                self.adjust_column_width_to_content()
                
                _pending_drag_items.clear()
            
            # Überschreibe die Methoden
            self.surface_tree_widget.startDrag = custom_startDrag
            self.surface_tree_widget.dropEvent = custom_dropEvent
            
            # Schriftgröße im TreeWidget auf 11pt setzen
            tree_font = QtGui.QFont()
            tree_font.setPointSize(11)
            self.surface_tree_widget.setFont(tree_font)
            
            # Source Name linksbündig ausrichten
            header = self.surface_tree_widget.header()
            header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            # Spaltenbreiten konfigurieren
            self.surface_tree_widget.setColumnWidth(0, 160)  # Surface Name - 10% schlanker (von 180px auf 160px)
            self.surface_tree_widget.setColumnWidth(1, 25)   # Enable (schmaler)
            self.surface_tree_widget.setColumnWidth(2, 25)   # Hide (schmaler)
            self.surface_tree_widget.setColumnWidth(3, 25)   # XY (schmaler)
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
            
            # Scrollbar aktivieren
            self.surface_tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Indentation für Gruppen aktivieren
            self.surface_tree_widget.setIndentation(15)
            
            # Drag & Drop aktivieren
            self.surface_tree_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
            self.surface_tree_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.surface_tree_widget.setDragEnabled(True)
            self.surface_tree_widget.setAcceptDrops(True)
            self.surface_tree_widget.setDropIndicatorShown(True)
            
            self.surface_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.surface_tree_widget.customContextMenuRequested.connect(self.show_context_menu)
            
            # Setze die SizePolicy
            self.surface_tree_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.surface_tree_widget.setMinimumHeight(100)  # Reduziert für kompakte UI (140px DockWidget - 24px Buttons - ~15px Margins)
            self.surface_tree_widget.setFixedWidth(270)  # 10% schlanker (von 300px auf 270px), gleiche Breite wie Sources UI
            
            # Verbinde Signale mit Slots
            # Eigener Handler, der sowohl die UI-Tabs als auch die 3D-Overlays aktualisiert
            self.surface_tree_widget.itemSelectionChanged.connect(self._handle_surface_tree_selection_changed)
            # Zusätzlich auch auf Änderungen des aktuellen Items reagieren (falls Selektion unverändert bleibt)
            self.surface_tree_widget.currentItemChanged.connect(
                lambda _current, _previous: self._handle_surface_tree_selection_changed()
            )
            self.surface_tree_widget.itemChanged.connect(self.on_surface_item_text_changed)
            
            # Füge das TreeWidget zum Layout hinzu
            left_side_layout.addWidget(self.surface_tree_widget, 1)
            
            # Button zum Hinzufügen mit Menü
            buttons_layout = QHBoxLayout()
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(5)
            buttons_layout.setAlignment(Qt.AlignLeft)
            
            self.pushbutton_add = QPushButton("Add")
            self.pushbutton_add.setFixedWidth(120)
            btn_font = QtGui.QFont()
            btn_font.setPointSize(11)
            self.pushbutton_add.setFont(btn_font)
            self.pushbutton_add.setFixedHeight(24)
            buttons_layout.addWidget(self.pushbutton_add)
            
            buttons_layout.addStretch(1)
            
            left_side_layout.addLayout(buttons_layout)
            
            # Erstelle Menü für Add-Button
            add_menu = QMenu(self.surface_dockWidget)
            add_surface_action = add_menu.addAction("Add Surface")
            add_group_action = add_menu.addAction("Add Group")
            add_menu.addSeparator()
            import_surfaces_action = add_menu.addAction("Import Surfaces")
            
            # Verbinde Aktionen
            add_surface_action.triggered.connect(self.add_surface)
            add_group_action.triggered.connect(lambda: self.add_group())
            import_surfaces_action.triggered.connect(self.load_dxf)
            
            # Setze Menü für Button
            self.pushbutton_add.setMenu(add_menu)
            
            # Rechte Seite für TabWidget
            self.right_side_widget = QWidget()
            splitter.addWidget(self.right_side_widget)
            
            # Erstelle das TabWidget
            self.tab_widget = QTabWidget()
            self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.tab_widget.setMinimumWidth(800)
            
            # Erstelle die Tabs
            self.create_surface_tab()
            self.create_group_tab_placeholder()
            
            # Füge TabWidget zum rechten Widget hinzu
            right_layout = QVBoxLayout(self.right_side_widget)
            right_layout.addWidget(self.tab_widget)
            
            # Lade bestehende Surfaces
            self.load_surfaces()
            
            # Rufe show_surfaces_tab auf, um die Tabs zu konfigurieren
            self.show_surfaces_tab()
            
        # Entsperre Signale am Ende
        if hasattr(self, 'surface_tree_widget'):
            self.surface_tree_widget.blockSignals(False)
            
        self.surface_tree_widget.clearSelection()
        
        # Setze Mindesthöhe des DockWidgets, damit Qt's Layout-System die Größe respektiert
        target_height = 140
        self.surface_dockWidget.setMinimumHeight(target_height)
        self.surface_dockWidget.setMaximumHeight(target_height)  # Temporär fixieren, damit Qt nicht überschreibt
        
        self.surface_dockWidget.show()
        
        # Setze initiale Größe des DockWidgets NACH show() mit Timer, damit Qt's Layout fertig ist
        from PyQt5.QtCore import QTimer
        def apply_resize():
            self.surface_dockWidget.setMaximumHeight(16777215)  # Entferne Max-Höhe wieder
            self.surface_dockWidget.resize(1200, target_height)
        
        QTimer.singleShot(100, apply_resize)  # 100ms Verzögerung, damit Qt's Layout fertig ist
    
    def close_surface_dock_widget(self):
        """Schließt das SurfaceDockWidget"""
        if hasattr(self, 'surface_dockWidget') and self.surface_dockWidget:
            self.surface_dockWidget.close()
            if hasattr(self.main_window, 'removeDockWidget'):
                self.main_window.removeDockWidget(self.surface_dockWidget)
            self.surface_dockWidget.deleteLater()
            self.surface_dockWidget = None
    
    def bind_to_sources_widget(self, sources_widget):
        """
        Verbindet die Auswahländerungen aus dem Sources-TreeWidget mit dem Surface-Widget.
        (Für Kompatibilität mit alter Implementierung)
        """
        # Diese Methode wird für Kompatibilität beibehalten
        # Die neue Implementierung benötigt keine direkte Bindung
        pass
    
    # ---- TreeWidget Management --------------------------------------
    
    def _save_expand_state(self):
        """Speichert den Expand/Collapse-Zustand aller Gruppen im TreeWidget."""
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return {}
        
        expand_state = {}
        
        def save_item_state(item, path=""):
            """Rekursiv speichert den Expand-Zustand von Items."""
            if not item:
                return
            
            item_type = item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                # Speichere den Zustand für Gruppen
                group_id_data = item.data(0, Qt.UserRole)
                if isinstance(group_id_data, dict):
                    group_id = group_id_data.get("id")
                else:
                    group_id = group_id_data
                
                # Verwende group_id als eindeutigen Pfad
                item_path = f"{path}/{group_id}" if path else str(group_id)
                expand_state[item_path] = item.isExpanded()
                
                # Rekursiv für Child-Items
                for i in range(item.childCount()):
                    child = item.child(i)
                    save_item_state(child, item_path)
            elif item_type == "surface":
                # Für Surfaces: Rekursiv für Child-Items (falls sie in Gruppen sind)
                for i in range(item.childCount()):
                    child = item.child(i)
                    save_item_state(child, path)
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.surface_tree_widget.topLevelItemCount()):
            item = self.surface_tree_widget.topLevelItem(i)
            save_item_state(item)
        
        return expand_state
    
    def _restore_expand_state(self, expand_state):
        """Stellt den Expand/Collapse-Zustand aller Gruppen im TreeWidget wieder her."""
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return
        if not expand_state:
            return
        
        def restore_item_state(item, path=""):
            """Rekursiv stellt den Expand-Zustand von Items wieder her."""
            if not item:
                return
            
            item_type = item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                # Stelle den Zustand für Gruppen wieder her
                group_id_data = item.data(0, Qt.UserRole)
                if isinstance(group_id_data, dict):
                    group_id = group_id_data.get("id")
                else:
                    group_id = group_id_data
                
                # Verwende group_id als eindeutigen Pfad
                item_path = f"{path}/{group_id}" if path else str(group_id)
                
                if item_path in expand_state:
                    item.setExpanded(expand_state[item_path])
                
                # Rekursiv für Child-Items
                for i in range(item.childCount()):
                    child = item.child(i)
                    restore_item_state(child, item_path)
            elif item_type == "surface":
                # Für Surfaces: Rekursiv für Child-Items (falls sie in Gruppen sind)
                for i in range(item.childCount()):
                    child = item.child(i)
                    restore_item_state(child, path)
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.surface_tree_widget.topLevelItemCount()):
            item = self.surface_tree_widget.topLevelItem(i)
            restore_item_state(item)
    
    def load_surfaces(self):
        """Lädt alle Surfaces und Gruppen in das TreeWidget"""
        if not hasattr(self, 'surface_tree_widget'):
            return
        
        logger = logging.getLogger(__name__)
        
        # Nur Validierung (keine Korrektur) beim Laden
        surface_store = getattr(self.settings, 'surface_definitions', {})
        if surface_store:
            try:
                from Module_LFO.Modules_Data.SurfaceValidator import validate_surface_geometry

                surfaces_dict = {}
                for surface_id, surface in surface_store.items():
                    if isinstance(surface, SurfaceDefinition):
                        surfaces_dict[surface_id] = surface.to_dict()
                    else:
                        surfaces_dict[surface_id] = surface

                for surface_id, surface_data in surfaces_dict.items():
                    try:
                        surface_obj = SurfaceDefinition.from_dict(surface_id, surface_data)
                        validation_result = validate_surface_geometry(
                            surface_obj,
                            round_to_cm=False,
                            remove_redundant=True,
                        )
                        if not validation_result.is_valid:
                            logger.warning(
                                f"Surface '{surface_obj.name}' ({surface_id}) ist ungültig beim Laden: "
                                f"{validation_result.error_message}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Fehler bei der Validierung von Surface '{surface_id}': {e}",
                            exc_info=True,
                        )

                logger.debug(f"Surfaces geladen: {len(surfaces_dict)} Surfaces")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Surfaces: {e}", exc_info=True)
                # Bei Fehler: Weiter mit unveränderten Surfaces
        
        # Speichere aktuellen Expand-Zustand vor dem Neuaufbau
        expand_state = self._save_expand_state()
        
        self.surface_tree_widget.blockSignals(True)
        self.surface_tree_widget.clear()
        
        # Stelle sicher, dass die Gruppen-Struktur aktuell ist
        self._group_controller.ensure_structure()
        
        # Hole Default-Fläche ID
        default_surface_id = getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default')
        root_group_id = self._group_controller.root_group_id

        logger.debug(
            "load_surfaces: %d surfaces im Store (Default=%s)",
            len(surface_store),
            default_surface_id if default_surface_id in surface_store else "none",
        )
        for surface_id, surface in surface_store.items():
            name = surface.name if isinstance(surface, SurfaceDefinition) else surface.get('name')
            group_id = surface.group_id if isinstance(surface, SurfaceDefinition) else surface.get('group_id')
            logger.debug("  Surface %s (%s) -> group %s", surface_id, name, group_id)
        
        # Verwende Fortschrittsanzeige, falls main_window verfügbar ist und viele Surfaces vorhanden sind
        use_progress = (
            hasattr(self, 'main_window') and 
            self.main_window and 
            hasattr(self.main_window, 'run_tasks_with_progress') and
            len(surface_store) > 10  # Nur bei vielen Surfaces Fortschrittsanzeige verwenden
        )
        
        if use_progress:
            try:
                self._load_surfaces_with_progress(default_surface_id, surface_store, root_group_id)
            except Exception as exc:
                logger.error(f"Fehler beim Laden der Surfaces mit Fortschrittsanzeige: {exc}")
                # Fallback: Laden ohne Fortschrittsanzeige
                self._load_surfaces_without_progress(default_surface_id, surface_store, root_group_id)
        else:
            self._load_surfaces_without_progress(default_surface_id, surface_store, root_group_id)
        
        self.validate_all_checkboxes()
        
        # Aktualisiere alle Gruppen-Checkbox-Zustände basierend auf den Childs
        self._update_all_group_checkbox_states()
        
        # Stelle den Expand-Zustand wieder her (anstatt expandAll())
        self._restore_expand_state(expand_state)
        
        self.surface_tree_widget.blockSignals(False)
        # Nach dem Laden keine Auswahl markieren → Highlight-Liste leeren
        setattr(self.settings, "active_surface_highlight_ids", [])
    
    def _load_surfaces_without_progress(self, default_surface_id, surface_store, root_group_id):
        """Lädt Surfaces ohne Fortschrittsanzeige (Fallback)"""
        # Lade Default-Fläche als Top-Level-Item ganz oben (wenn sie existiert)
        if default_surface_id in surface_store:
            default_surface = surface_store[default_surface_id]
            # Entferne Default-Fläche aus Gruppe, falls sie in einer ist
            if isinstance(default_surface, SurfaceDefinition):
                if default_surface.group_id:
                    self._group_controller.detach_surface(default_surface_id)
                    default_surface.group_id = None
            else:
                if default_surface.get('group_id'):
                    self._group_controller.detach_surface(default_surface_id)
                    default_surface['group_id'] = None
            
            # Füge Default-Fläche als Top-Level-Item ganz oben hinzu
            default_item = self._create_surface_item(default_surface_id, default_surface)
            self.surface_tree_widget.insertTopLevelItem(0, default_item)
            self.ensure_surface_checkboxes(default_item)
        
        # Lade manuelle Gruppen (ohne Root-Gruppe)
        group_store = self._group_controller.list_groups()
        # Lade alle Gruppen außer der Root-Gruppe
        for group_id, group in group_store.items():
            if group_id != root_group_id:
                # Nur Top-Level-Gruppen (ohne Parent oder Parent ist Root)
                if not group.parent_id or group.parent_id == root_group_id:
                    self._populate_group_tree(None, group)
        
        # Lade Surfaces ohne Gruppe als Top-Level-Items
        for surface_id, surface in surface_store.items():
            # Überspringe Default-Fläche, die bereits oben hinzugefügt wurde
            if surface_id == default_surface_id:
                continue
            
            # Prüfe, ob Surface eine Gruppe hat
            if isinstance(surface, SurfaceDefinition):
                group_id = surface.group_id
            else:
                group_id = surface.get('group_id')
            
            # Wenn keine Gruppe oder Root-Gruppe, zeige als Top-Level-Item
            if not group_id or group_id == root_group_id:
                # Entferne aus Root-Gruppe, falls vorhanden
                if group_id == root_group_id:
                    self._group_controller.detach_surface(surface_id)
                    if isinstance(surface, SurfaceDefinition):
                        surface.group_id = None
                    else:
                        surface['group_id'] = None
                
                # Füge als Top-Level-Item hinzu
                surface_item = self._create_surface_item(surface_id, surface)
                # Füge nach Default-Fläche, aber vor Gruppen hinzu
                insert_index = 1 if default_surface_id in surface_store else 0
                self.surface_tree_widget.insertTopLevelItem(insert_index, surface_item)
                self.ensure_surface_checkboxes(surface_item)
    
    def _load_surfaces_with_progress(self, default_surface_id, surface_store, root_group_id):
        """Lädt Surfaces mit Fortschrittsanzeige"""
        from Module_LFO.Modules_Init.Progress import ProgressCancelled
        
        tasks = []
        
        # Task 1: Default-Fläche laden
        def load_default_task():
            if default_surface_id in surface_store:
                default_surface = surface_store[default_surface_id]
                # Entferne Default-Fläche aus Gruppe, falls sie in einer ist
                if isinstance(default_surface, SurfaceDefinition):
                    if default_surface.group_id:
                        self._group_controller.detach_surface(default_surface_id)
                        default_surface.group_id = None
                else:
                    if default_surface.get('group_id'):
                        self._group_controller.detach_surface(default_surface_id)
                        default_surface['group_id'] = None
                
                # Füge Default-Fläche als Top-Level-Item ganz oben hinzu
                default_item = self._create_surface_item(default_surface_id, default_surface)
                self.surface_tree_widget.insertTopLevelItem(0, default_item)
                self.ensure_surface_checkboxes(default_item)
        
        tasks.append(("Default-Fläche laden", load_default_task))
        
        # Task 2: Gruppen laden
        def load_groups_task():
            group_store = self._group_controller.list_groups()
            # Lade alle Gruppen außer der Root-Gruppe
            for group_id, group in group_store.items():
                if group_id != root_group_id:
                    # Nur Top-Level-Gruppen (ohne Parent oder Parent ist Root)
                    if not group.parent_id or group.parent_id == root_group_id:
                        self._populate_group_tree(None, group)
        
        tasks.append(("Gruppen laden", load_groups_task))
        
        # Task 3: Surfaces ohne Gruppe laden
        def load_surfaces_task():
            for surface_id, surface in surface_store.items():
                # Überspringe Default-Fläche, die bereits oben hinzugefügt wurde
                if surface_id == default_surface_id:
                    continue
                
                # Prüfe, ob Surface eine Gruppe hat
                if isinstance(surface, SurfaceDefinition):
                    group_id = surface.group_id
                else:
                    group_id = surface.get('group_id')
                
                # Wenn keine Gruppe oder Root-Gruppe, zeige als Top-Level-Item
                if not group_id or group_id == root_group_id:
                    # Entferne aus Root-Gruppe, falls vorhanden
                    if group_id == root_group_id:
                        self._group_controller.detach_surface(surface_id)
                        if isinstance(surface, SurfaceDefinition):
                            surface.group_id = None
                        else:
                            surface['group_id'] = None
                    
                    # Füge als Top-Level-Item hinzu
                    surface_item = self._create_surface_item(surface_id, surface)
                    # Füge nach Default-Fläche, aber vor Gruppen hinzu
                    insert_index = 1 if default_surface_id in surface_store else 0
                    self.surface_tree_widget.insertTopLevelItem(insert_index, surface_item)
                    self.ensure_surface_checkboxes(surface_item)
        
        tasks.append(("Surfaces laden", load_surfaces_task))
        
        # Führe Tasks mit Fortschrittsanzeige aus
        try:
            self.main_window.run_tasks_with_progress("TreeWidget aktualisieren", tasks)
        except ProgressCancelled:
            # Benutzer hat abgebrochen - TreeWidget bleibt im aktuellen Zustand
            pass
    
    def _populate_group_tree(self, parent_item, group):
        """Rekursiv befüllt das TreeWidget mit Gruppen und deren Surfaces"""
        # Erstelle Gruppen-Item
        group_item = self._create_group_item(group)
        
        if parent_item is None:
            # Füge oben hinzu (neue Items erscheinen oben)
            self.surface_tree_widget.insertTopLevelItem(0, group_item)
        else:
            # Füge als erstes Child hinzu (oben)
            parent_item.insertChild(0, group_item)
        
        # Füge Child-Gruppen hinzu (umgekehrte Reihenfolge, damit neue oben erscheinen)
        for child_group_id in reversed(group.child_groups):
            child_group = self._group_controller.get_group(child_group_id)
            if child_group:
                self._populate_group_tree(group_item, child_group)
        
        # Füge Surfaces der Gruppe hinzu (umgekehrte Reihenfolge, damit neue oben erscheinen)
        surface_store = getattr(self.settings, 'surface_definitions', {})
        for surface_id in reversed(group.surface_ids):
            surface = surface_store.get(surface_id)
            if surface:
                surface_item = self._create_surface_item(surface_id, surface)
                group_item.insertChild(0, surface_item)
                # ensure_surface_checkboxes prüft automatisch, ob Surface in Gruppe ist und entfernt Checkboxen
                self.ensure_surface_checkboxes(surface_item)
        
        # Expand-Zustand wird später durch _restore_expand_state() wiederhergestellt
        # group_item.setExpanded(True)  # Entfernt, da Zustand wiederhergestellt wird
    
    def _create_surface_item(self, surface_id, surface_data):
        """Erstellt ein TreeWidgetItem für ein Surface"""
        item = QTreeWidgetItem()
        
        # Name
        if isinstance(surface_data, SurfaceDefinition):
            name = surface_data.name
        else:
            name = surface_data.get('name', surface_id)
        item.setText(0, name)
        
        # UserRole: Surface-ID
        item.setData(0, Qt.UserRole, surface_id)
        item.setData(0, Qt.UserRole + 1, "surface")
        
        # Flags - Default-Fläche darf nicht per Drag & Drop verschoben werden
        default_surface_id = getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default')
        if surface_id == default_surface_id:
            # Default-Fläche: kein Drag & Drop
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable |
                Qt.ItemIsUserCheckable
            )
        else:
            # Normale Surfaces: mit Drag & Drop
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable |
                Qt.ItemIsDragEnabled | Qt.ItemIsUserCheckable
            )
        
        # Checkboxen für Enable und Hide
        self.ensure_surface_checkboxes(item)
        
        return item
    
    def _create_group_item(self, group):
        """Erstellt ein TreeWidgetItem für eine Gruppe"""
        import logging
        logger = logging.getLogger(__name__)
        item = QTreeWidgetItem()
        item.setText(0, group.name)
        item.setData(0, Qt.UserRole, group.group_id)
        item.setData(0, Qt.UserRole + 1, "group")
        
        # Erhöhe Item-Höhe für mehr Abstand zwischen Gruppen
        from PyQt5.QtCore import QSize
        item.setSizeHint(0, QSize(0, 32))  # Standard ist ~24px, erhöht auf 32px
        
        # Flags
        item.setFlags(
            Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable |
            Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled | Qt.ItemIsUserCheckable
        )
        
        # Checkboxen für Enable und Hide
        self.ensure_group_checkboxes(item)
        
        return item

    # ---- Auswahl-Handling für rote Umrandung ------------------------
    
    def _find_tree_item_by_id(self, item_id):
        """
        Findet ein TreeWidgetItem anhand seiner ID (rekursiv durchsucht den Tree).
        
        Args:
            item_id: Die ID des gesuchten Items (aus Qt.UserRole)
            
        Returns:
            QTreeWidgetItem oder None, falls nicht gefunden
        """
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return None
        
        def search_item(parent_item):
            """Rekursive Suche durch Tree-Items"""
            if parent_item is None:
                # Suche auf Top-Level
                for i in range(self.surface_tree_widget.topLevelItemCount()):
                    item = self.surface_tree_widget.topLevelItem(i)
                    try:
                        if item.data(0, Qt.UserRole) == item_id:
                            return item
                        # Suche rekursiv in Children
                        result = search_item(item)
                        if result:
                            return result
                    except RuntimeError:
                        # Item wurde gelöscht, überspringe
                        continue
            else:
                # Suche in Children des Parent-Items
                try:
                    for i in range(parent_item.childCount()):
                        child = parent_item.child(i)
                        try:
                            if child.data(0, Qt.UserRole) == item_id:
                                return child
                            # Suche rekursiv in Children
                            result = search_item(child)
                            if result:
                                return result
                        except RuntimeError:
                            # Item wurde gelöscht, überspringe
                            continue
                except RuntimeError:
                    # Parent-Item wurde gelöscht
                    return None
            return None
        
        return search_item(None)

    def _collect_all_surfaces_from_group(self, group_id: str, surface_store: Dict) -> List[str]:
        """
        Sammelt rekursiv alle Surface-IDs aus einer Gruppe und ihren Untergruppen.
        
        Args:
            group_id: ID der Gruppe
            surface_store: Dictionary mit allen Surface-Definitionen
            
        Returns:
            Liste aller Surface-IDs aus der Gruppe und ihren Untergruppen
        """
        highlight_ids = []
        group = self._group_controller.get_group(group_id)
        if not group:
            return highlight_ids
        
        # Sammle direkte Surfaces der Gruppe
        for sid in getattr(group, "surface_ids", []):
            if sid in surface_store:
                highlight_ids.append(sid)
        
        # Sammle rekursiv Surfaces aus Untergruppen
        for child_group_id in getattr(group, "child_groups", []):
            child_surfaces = self._collect_all_surfaces_from_group(child_group_id, surface_store)
            highlight_ids.extend(child_surfaces)
        
        return highlight_ids

    def _handle_surface_tree_selection_changed(self):
        """
        Reagiert auf Auswahländerungen im Surface-Tree:
        - Aktualisiert die Tabs (wie bisher über show_surfaces_tab)
        - Setzt active_surface_highlight_ids für den 3D-Plot
        - Unterstützt mehrere ausgewählte Items und Gruppen
        - Triggert update_overlays() für rote Umrandung.
        """
        if not hasattr(self, 'surface_tree_widget'):
            return

        # Hole alle ausgewählten Items (nicht nur currentItem) für Mehrfachauswahl
        selected_items = self.surface_tree_widget.selectedItems()
        
        if not selected_items:
            # Keine Auswahl → keine Highlights
            setattr(self.settings, "active_surface_highlight_ids", [])
            setattr(self.settings, "active_surface_id", None)
            return

        # UI-Tabs wie bisher aktualisieren (basierend auf currentItem)
        try:
            self.show_surfaces_tab()
        except Exception:
            pass

        highlight_ids = []
        surface_store = getattr(self.settings, "surface_definitions", {}) or {}
        active_surface_id = None

        # Verarbeite alle ausgewählten Items
        for selected_item in selected_items:
            try:
                item_type = selected_item.data(0, Qt.UserRole + 1)
            except RuntimeError:
                # Item wurde gelöscht, überspringe
                continue

            if item_type == "surface":
                surface_id = selected_item.data(0, Qt.UserRole)
                if isinstance(surface_id, dict):
                    surface_id = surface_id.get("id")
                if isinstance(surface_id, str) and surface_id in surface_store:
                    if surface_id not in highlight_ids:
                        highlight_ids.append(surface_id)
                    # Setze active_surface_id auf das erste ausgewählte Surface
                    if active_surface_id is None:
                        active_surface_id = surface_id
            elif item_type == "group":
                group_id_data = selected_item.data(0, Qt.UserRole)
                # group_id kann ein Dict sein (wie in WindowSurfaceWidget) oder direkt die ID
                if isinstance(group_id_data, dict):
                    group_id = group_id_data.get("id")
                else:
                    group_id = group_id_data
                
                # Sammle rekursiv alle Surfaces aus der Gruppe und ihren Untergruppen
                if group_id:
                    group_surfaces = self._collect_all_surfaces_from_group(group_id, surface_store)
                    for surface_id in group_surfaces:
                        if surface_id not in highlight_ids:
                            highlight_ids.append(surface_id)
                    
                    # Setze active_surface_id auf das erste Surface der ersten Gruppe (falls noch nicht gesetzt)
                    if active_surface_id is None and group_surfaces:
                        active_surface_id = group_surfaces[0]

        # Setze active_surface_id (nur wenn ein Surface ausgewählt ist, nicht nur Gruppen)
        if active_surface_id:
            try:
                setattr(self.settings, "active_surface_id", active_surface_id)
            except Exception:
                pass
        else:
            # Nur Gruppen ausgewählt → setze active_surface_id auf None
            try:
                setattr(self.settings, "active_surface_id", None)
            except Exception:
                pass

        # In Settings speichern, damit PlotSPL3DOverlays die Info nutzen kann
        setattr(self.settings, "active_surface_highlight_ids", highlight_ids)

        # Overlays im 3D-Plot aktualisieren (nur visuell, keine Neuberechnung)
        main_window = getattr(self, "main_window", None)
        if (
            main_window is not None
            and hasattr(main_window, "draw_plots")
            and hasattr(main_window.draw_plots, "draw_spl_plotter")
        ):
            draw_spl = main_window.draw_plots.draw_spl_plotter
            if hasattr(draw_spl, "update_overlays"):
                try:
                    draw_spl.update_overlays(self.settings, self.container)
                except Exception:
                    # Fehler hier sollen die restliche UI nicht blockieren
                    pass
    
    def _get_group_majority_state(self, group_item, column):
        """
        Bestimmt den Mehrheitszustand einer Gruppe basierend auf ihren Child-Items.
        Wird verwendet, wenn die Gruppe PartiallyChecked ist.
        
        Args:
            group_item: Das Gruppen-Item
            column: Die Spalte (1=Enable, 2=Hide, 3=XY)
            
        Returns:
            Qt.Checked oder Qt.Unchecked basierend auf dem Mehrheitszustand
        """
        if not group_item:
            return Qt.Unchecked
        
        checked_count = 0
        unchecked_count = 0
        
        def count_child_states(item):
            nonlocal checked_count, unchecked_count
            for i in range(item.childCount()):
                child = item.child(i)
                child_type = child.data(0, Qt.UserRole + 1)
                child_checkbox = self.surface_tree_widget.itemWidget(child, column)
                
                if child_type == "surface":
                    # Surface: Prüfe Checkbox oder Surface-Daten
                    if child_checkbox:
                        if child_checkbox.isChecked():
                            checked_count += 1
                        else:
                            unchecked_count += 1
                    else:
                        # Surface hat keine Checkbox (in Gruppe) - prüfe Surface-Daten
                        surface_id_data = child.data(0, Qt.UserRole)
                        if isinstance(surface_id_data, dict):
                            surface_id = surface_id_data.get("id")
                        else:
                            surface_id = surface_id_data
                        
                        if surface_id:
                            surface = self._get_surface(surface_id)
                            if surface:
                                is_checked = False
                                if column == 1:  # Enable
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = surface.enabled
                                    else:
                                        is_checked = surface.get('enabled', False)
                                elif column == 2:  # Hide
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = surface.hidden
                                    else:
                                        is_checked = surface.get('hidden', False)
                                elif column == 3:  # XY
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = getattr(surface, 'xy_enabled', True)
                                    else:
                                        is_checked = surface.get('xy_enabled', True)
                                
                                if is_checked:
                                    checked_count += 1
                                else:
                                    unchecked_count += 1
                elif child_type == "group":
                    # Gruppe: Prüfe Checkbox
                    if child_checkbox:
                        if child_checkbox.isChecked():
                            checked_count += 1
                        else:
                            unchecked_count += 1
                    
                    # Rekursiv für Untergruppen
                    count_child_states(child)
        
        count_child_states(group_item)
        
        # Mehrheitszustand: Wenn mehr checked als unchecked, dann Checked, sonst Unchecked
        return Qt.Checked if checked_count >= unchecked_count else Qt.Unchecked
    
    def ensure_surface_checkboxes(self, item):
        """
        Stellt sicher, dass Checkboxen für ein Surface-Item existieren.
        Wenn das Surface in einer Gruppe ist, werden die Checkboxen entfernt.
        """
        # Prüfe, ob das Surface in einer Gruppe ist
        parent = item.parent()
        is_in_group = parent is not None and parent.data(0, Qt.UserRole + 1) == "group"
        
        if is_in_group:
            # Surface ist in einer Gruppe: Entferne alle Checkboxen
            for column in [1, 2, 3]:
                widget = self.surface_tree_widget.itemWidget(item, column)
                if widget:
                    self.surface_tree_widget.removeItemWidget(item, column)
            return
        
        # Surface ist nicht in einer Gruppe: Stelle sicher, dass Checkboxen existieren
        # Enable Checkbox (Spalte 1)
        if self.surface_tree_widget.itemWidget(item, 1) is None:
            enable_checkbox = self.create_checkbox()
            surface_id = item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            
            # Lade aktuellen Enable-Status
            surface = self._get_surface(surface_id)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    enabled = surface.enabled
                else:
                    enabled = surface.get('enabled', False)
                enable_checkbox.setChecked(enabled)
            
            enable_checkbox.stateChanged.connect(
                lambda state, sid=surface_id: self.on_surface_enable_changed(sid, state)
            )
            self.surface_tree_widget.setItemWidget(item, 1, enable_checkbox)
        
        # Hide Checkbox (Spalte 2)
        if self.surface_tree_widget.itemWidget(item, 2) is None:
            hide_checkbox = self.create_checkbox()
            surface_id = item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            
            # Lade aktuellen Hide-Status
            surface = self._get_surface(surface_id)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    hidden = surface.hidden
                else:
                    hidden = surface.get('hidden', False)
                hide_checkbox.setChecked(hidden)
            
            hide_checkbox.stateChanged.connect(
                lambda state, sid=surface_id: self.on_surface_hide_changed(sid, state)
            )
            self.surface_tree_widget.setItemWidget(item, 2, hide_checkbox)
        
        # XY Checkbox (Spalte 3)
        if self.surface_tree_widget.itemWidget(item, 3) is None:
            xy_checkbox = self.create_checkbox()
            xy_checkbox.setChecked(True)  # Per Default aktiv
            surface_id = item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            
            # Lade aktuellen XY-Status (falls vorhanden)
            surface = self._get_surface(surface_id)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    xy_enabled = getattr(surface, 'xy_enabled', True)
                else:
                    xy_enabled = surface.get('xy_enabled', True)
                xy_checkbox.setChecked(xy_enabled)
            
            xy_checkbox.stateChanged.connect(
                lambda state, sid=surface_id: self.on_surface_xy_changed(sid, state)
            )
            self.surface_tree_widget.setItemWidget(item, 3, xy_checkbox)
    
    def ensure_group_checkboxes(self, item):
        """Stellt sicher, dass Checkboxen für ein Gruppen-Item existieren"""
        # Enable Checkbox (Spalte 1) - mit Tristate für teilweise aktivierte Gruppen
        enable_checkbox = self.surface_tree_widget.itemWidget(item, 1)
        if enable_checkbox is None:
            enable_checkbox = self.create_checkbox(tristate=True)
            group_id = item.data(0, Qt.UserRole)
            
            # Lade aktuellen Enable-Status
            group = self._group_controller.get_group(group_id)
            if group:
                # Verwende setCheckState für tristate-Checkboxen
                enable_checkbox.setCheckState(Qt.Checked if group.enabled else Qt.Unchecked)
            
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 1)
            
            enable_checkbox.stateChanged.connect(
                lambda state, g_item=item: self.on_group_enable_changed(g_item, state)
            )
            self.surface_tree_widget.setItemWidget(item, 1, enable_checkbox)
        else:
            # Stelle sicher, dass bestehende Checkbox tristate aktiviert hat
            if not enable_checkbox.isTristate():
                enable_checkbox.setTristate(True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 1)
        
        # Hide Checkbox (Spalte 2) - mit Tristate für teilweise aktivierte Gruppen
        hide_checkbox = self.surface_tree_widget.itemWidget(item, 2)
        if hide_checkbox is None:
            hide_checkbox = self.create_checkbox(tristate=True)
            group_id = item.data(0, Qt.UserRole)
            
            # Lade aktuellen Hide-Status
            group = self._group_controller.get_group(group_id)
            if group:
                # Verwende setCheckState für tristate-Checkboxen
                hide_checkbox.setCheckState(Qt.Checked if group.hidden else Qt.Unchecked)
            
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 2)
            
            hide_checkbox.stateChanged.connect(
                lambda state, g_item=item: self.on_group_hide_changed(g_item, state)
            )
            self.surface_tree_widget.setItemWidget(item, 2, hide_checkbox)
        else:
            # Stelle sicher, dass bestehende Checkbox tristate aktiviert hat
            if not hide_checkbox.isTristate():
                hide_checkbox.setTristate(True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 2)
        
        # XY Checkbox (Spalte 3) - mit Tristate für teilweise aktivierte Gruppen
        xy_checkbox = self.surface_tree_widget.itemWidget(item, 3)
        if xy_checkbox is None:
            xy_checkbox = self.create_checkbox(tristate=True)
            group_id = item.data(0, Qt.UserRole)
            
            # Lade aktuellen XY-Status (falls vorhanden)
            group = self._group_controller.get_group(group_id)
            if group:
                xy_enabled = getattr(group, 'xy_enabled', True)
                # Verwende setCheckState für tristate-Checkboxen
                xy_checkbox.setCheckState(Qt.Checked if xy_enabled else Qt.Unchecked)
            else:
                # Per Default aktiv
                xy_checkbox.setCheckState(Qt.Checked)
            
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 3)
            
            xy_checkbox.stateChanged.connect(
                lambda state, g_item=item: self.on_group_xy_changed(g_item, state)
            )
            self.surface_tree_widget.setItemWidget(item, 3, xy_checkbox)
        else:
            # Stelle sicher, dass bestehende Checkbox tristate aktiviert hat
            if not xy_checkbox.isTristate():
                xy_checkbox.setTristate(True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 3)
    
    def _update_group_child_checkboxes(self, group_item, column, checked, update_data=True, skip_calculations=False, skip_state_update=False):
        """
        Aktualisiert rekursiv die Checkboxen aller Child-Surfaces/-Gruppen einer Gruppe.
        
        Args:
            group_item: Das Gruppen-Item
            column: Die Spalte (1=Enable, 2=Hide, 3=XY)
            checked: Der neue Checkbox-Zustand
            update_data: Wenn True, werden auch die tatsächlichen Surface-Daten aktualisiert
            skip_calculations: Wenn True, werden keine Berechnungen ausgelöst (für Gruppen-Updates)
            skip_state_update: Wenn True, wird _update_group_checkbox_state nicht aufgerufen (für Gruppen-Updates)
        """
        if not group_item:
            return
        
        group_id_data = group_item.data(0, Qt.UserRole)
        if isinstance(group_id_data, dict):
            group_id = group_id_data.get("id")
        else:
            group_id = group_id_data
        print(f"[DEBUG _update_group_child_checkboxes] Gruppe '{group_id}': column={column}, checked={checked}, update_data={update_data}, childCount={group_item.childCount()}")
        
        for i in range(group_item.childCount()):
            child = group_item.child(i)
            child_type = child.data(0, Qt.UserRole + 1)
            checkbox = self.surface_tree_widget.itemWidget(child, column)
            
            if child_type == "surface":
                # Surface: Aktualisiere direkt die Surface-Daten (auch wenn keine Checkbox sichtbar ist)
                surface_id_data = child.data(0, Qt.UserRole)
                if isinstance(surface_id_data, dict):
                    surface_id = surface_id_data.get("id")
                else:
                    surface_id = surface_id_data
                
                print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}' (column={column}): checkbox vorhanden={checkbox is not None}, update_data={update_data}, checked={checked}")
                
                # 🎯 WICHTIG: Stelle sicher, dass Checkbox existiert, bevor wir sie setzen
                # Wenn Surface in einer Gruppe ist, hat es keine Checkbox (wird in ensure_surface_checkboxes entfernt)
                # Aber wenn Surface später aus Gruppe entfernt wird, sollte die Checkbox erstellt werden
                if surface_id:
                    # Prüfe, ob Surface in einer Gruppe ist
                    parent = child.parent()
                    is_in_group = parent is not None and parent.data(0, Qt.UserRole + 1) == "group"
                    
                    if not is_in_group and not checkbox:
                        # Surface ist nicht in einer Gruppe, aber Checkbox fehlt - erstelle sie
                        print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}': Checkbox fehlt, erstelle sie (column={column})")
                        self.ensure_surface_checkboxes(child)
                        checkbox = self.surface_tree_widget.itemWidget(child, column)
                    
                    # Wenn Checkbox vorhanden ist, aktualisiere sie
                    if checkbox:
                        checkbox.blockSignals(True)
                        # Verwende setCheckState für tristate-Checkboxen
                        checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                        checkbox.blockSignals(False)
                        print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}': Checkbox auf {checked} gesetzt")
                
                # Aktualisiere Surface-Daten durch Aufruf des entsprechenden Handlers
                # (wird auch ausgeführt, wenn keine Checkbox vorhanden ist, da Surface in Gruppe ist)
                if update_data and surface_id:
                    state = Qt.Checked if checked else Qt.Unchecked
                    print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}': Rufe Handler auf (column={column}, state={state})")
                    if column == 1:  # Enable
                        self.on_surface_enable_changed(surface_id, state, skip_calculations=skip_calculations)
                    elif column == 2:  # Hide
                        self.on_surface_hide_changed(surface_id, state, skip_calculations=skip_calculations)
                    elif column == 3:  # XY
                        self.on_surface_xy_changed(surface_id, state, skip_calculations=skip_calculations)
                    print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}': Handler aufgerufen")
                elif not update_data:
                    print(f"[DEBUG _update_group_child_checkboxes] Surface '{surface_id}': update_data=False - überspringe Datenaktualisierung")
                elif not surface_id:
                    print(f"[DEBUG _update_group_child_checkboxes] Surface: surface_id ist None - überspringe")
            elif child_type == "group":
                # Gruppe: Aktualisiere Checkbox (Gruppen behalten immer ihre Checkboxen)
                # 🎯 WICHTIG: Stelle sicher, dass Checkbox existiert
                if not checkbox:
                    print(f"[DEBUG _update_group_child_checkboxes] Verschachtelte Gruppe: Checkbox fehlt, erstelle sie (column={column})")
                    self.ensure_group_checkboxes(child)
                    checkbox = self.surface_tree_widget.itemWidget(child, column)
                
                if checkbox:
                    checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    checkbox.blockSignals(False)
                    print(f"[DEBUG _update_group_child_checkboxes] Verschachtelte Gruppe: Checkbox auf {checked} gesetzt")
                
                # Aktualisiere Gruppen-Daten durch Aufruf des entsprechenden Handlers
                # (übergeordnete Gruppe kann untergeordnete Gruppe steuern)
                if update_data:
                    state = Qt.Checked if checked else Qt.Unchecked
                    if column == 1:  # Enable
                        group_id_data = child.data(0, Qt.UserRole)
                        if isinstance(group_id_data, dict):
                            group_id = group_id_data.get("id")
                        else:
                            group_id = group_id_data
                        if group_id:
                            # Setze Gruppen-Status
                            self._group_controller.set_surface_group_enabled(group_id, checked)
                            if group_id in self.surface_groups:
                                self.surface_groups[group_id]['enabled'] = checked
                            # Aktualisiere auch alle Childs der untergeordneten Gruppe
                            # (rekursiv, damit auch Surfaces in der untergeordneten Gruppe aktualisiert werden)
                            self._update_group_child_checkboxes(child, column, checked, update_data=True, skip_calculations=skip_calculations, skip_state_update=skip_state_update)
                    elif column == 2:  # Hide
                        group_id_data = child.data(0, Qt.UserRole)
                        if isinstance(group_id_data, dict):
                            group_id = group_id_data.get("id")
                        else:
                            group_id = group_id_data
                        if group_id:
                            # Setze Gruppen-Status
                            self._group_controller.set_surface_group_hidden(group_id, checked)
                            if group_id in self.surface_groups:
                                self.surface_groups[group_id]['hidden'] = checked
                            # Aktualisiere auch alle Childs der untergeordneten Gruppe
                            self._update_group_child_checkboxes(child, column, checked, update_data=True, skip_calculations=skip_calculations, skip_state_update=skip_state_update)
                    elif column == 3:  # XY
                        group_id_data = child.data(0, Qt.UserRole)
                        if isinstance(group_id_data, dict):
                            group_id = group_id_data.get("id")
                        else:
                            group_id = group_id_data
                        if group_id:
                            # Setze Gruppen-Status
                            group = self._group_controller.get_group(group_id)
                            if group:
                                group.xy_enabled = checked
                            if group_id in self.surface_groups:
                                self.surface_groups[group_id]['xy_enabled'] = checked
                            # Aktualisiere auch alle Childs der untergeordneten Gruppe
                            self._update_group_child_checkboxes(child, column, checked, update_data=True, skip_calculations=skip_calculations, skip_state_update=skip_state_update)
        
        # Aktualisiere Gruppen-Checkbox-Zustand nach Änderung der Child-Items
        # WICHTIG: Dies wird NACH dem Update aller Childs aufgerufen, daher sollte der Zustand korrekt sein
        # Überspringe, wenn skip_state_update=True (wird bereits explizit gesetzt)
        if not skip_state_update:
            self._update_group_checkbox_state(group_item, column)
    
    def _update_group_checkbox_state(self, group_item, column):
        """
        Aktualisiert den Zustand einer Gruppen-Checkbox basierend auf den Child-Items.
        Setzt PartiallyChecked (Klammer), wenn einige Child-Items checked und andere unchecked sind.
        Berücksichtigt auch Surfaces ohne Checkboxen (die in Gruppen sind).
        
        Args:
            group_item: Das Gruppen-Item
            column: Die Spalte (1=Enable, 2=Hide, 3=XY)
        """
        if not group_item:
            return
        
        checkbox = self.surface_tree_widget.itemWidget(group_item, column)
        if not checkbox:
            return
        
        # Stelle sicher, dass die Checkbox tristate aktiviert hat
        if not checkbox.isTristate():
            checkbox.setTristate(True)
        
        # Sammle alle Child-Checkboxen und Surface-Daten (rekursiv)
        checked_count = 0
        unchecked_count = 0
        total_count = 0
        
        def count_child_checkboxes(item):
            nonlocal checked_count, unchecked_count, total_count
            for i in range(item.childCount()):
                child = item.child(i)
                child_type = child.data(0, Qt.UserRole + 1)
                child_checkbox = self.surface_tree_widget.itemWidget(child, column)
                
                if child_type == "surface":
                    # Surface: Prüfe Checkbox oder Surface-Daten
                    if child_checkbox:
                        # Surface hat Checkbox (nicht in Gruppe)
                        total_count += 1
                        if child_checkbox.isChecked():
                            checked_count += 1
                        else:
                            unchecked_count += 1
                    else:
                        # Surface hat keine Checkbox (in Gruppe) - prüfe Surface-Daten
                        surface_id_data = child.data(0, Qt.UserRole)
                        if isinstance(surface_id_data, dict):
                            surface_id = surface_id_data.get("id")
                        else:
                            surface_id = surface_id_data
                        
                        if surface_id:
                            surface = self._get_surface(surface_id)
                            if surface:
                                total_count += 1
                                is_checked = False
                                if column == 1:  # Enable
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = surface.enabled
                                    else:
                                        is_checked = surface.get('enabled', False)
                                elif column == 2:  # Hide
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = surface.hidden
                                    else:
                                        is_checked = surface.get('hidden', False)
                                elif column == 3:  # XY
                                    if isinstance(surface, SurfaceDefinition):
                                        is_checked = getattr(surface, 'xy_enabled', True)
                                    else:
                                        is_checked = surface.get('xy_enabled', True)
                                
                                if is_checked:
                                    checked_count += 1
                                else:
                                    unchecked_count += 1
                elif child_type == "group":
                    # Gruppe: Prüfe Checkbox-Zustand (inkl. PartiallyChecked)
                    if child_checkbox:
                        total_count += 1
                        child_state = child_checkbox.checkState()
                        if child_state == Qt.Checked:
                            checked_count += 1
                        elif child_state == Qt.Unchecked:
                            unchecked_count += 1
                        elif child_state == Qt.PartiallyChecked:
                            # PartiallyChecked bedeutet gemischter Zustand -> zählt als beide
                            checked_count += 1
                            unchecked_count += 1
                    
                    # Rekursiv für Untergruppen
                    count_child_checkboxes(child)
        
        count_child_checkboxes(group_item)
        
        # Setze Checkbox-Zustand basierend auf Child-Items
        checkbox.blockSignals(True)
        if total_count == 0:
            # Keine Child-Items: Verwende Gruppen-Daten als Fallback
            group_id_data = group_item.data(0, Qt.UserRole)
            if isinstance(group_id_data, dict):
                group_id = group_id_data.get("id")
            else:
                group_id = group_id_data
            
            if group_id:
                group = self._group_controller.get_group(group_id)
                if group:
                    is_checked = False
                    if column == 1:  # Enable
                        is_checked = group.enabled
                    elif column == 2:  # Hide
                        is_checked = group.hidden
                    elif column == 3:  # XY
                        is_checked = getattr(group, 'xy_enabled', True)
                    
                    checkbox.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
                # Wenn keine Gruppe gefunden, Zustand bleibt unverändert
        elif checked_count == total_count:
            # Alle Child-Items sind checked
            checkbox.setCheckState(Qt.Checked)
        elif checked_count == 0:
            # Alle Child-Items sind unchecked
            checkbox.setCheckState(Qt.Unchecked)
        else:
            # Gemischter Zustand: Einige checked, einige unchecked -> PartiallyChecked
            # Dies tritt auf, wenn checked_count > 0 und unchecked_count > 0
            checkbox.setCheckState(Qt.PartiallyChecked)
        checkbox.blockSignals(False)
    
    def create_checkbox(self, checked=False, tristate=False):
        """
        Erstellt eine Checkbox mit Standard-Style
        
        Args:
            checked: Initialer Checked-Zustand
            tristate: Wenn True, aktiviert Tristate-Modus (für teilweise aktivierte Gruppen)
        """
        checkbox = QCheckBox()
        checkbox.setFixedSize(18, 18)
        if tristate:
            checkbox.setTristate(True)
        checkbox.setChecked(checked)
        return checkbox
    
    def validate_all_checkboxes(self):
        """Validiert alle Checkboxen im TreeWidget"""
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return
        
        def _ensure_item(item):
            if not item:
                return
            item_type = item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                self.ensure_group_checkboxes(item)
            elif item_type == "surface":
                self.ensure_surface_checkboxes(item)
            
            for idx in range(item.childCount()):
                _ensure_item(item.child(idx))
        
        for i in range(self.surface_tree_widget.topLevelItemCount()):
            _ensure_item(self.surface_tree_widget.topLevelItem(i))
    
    def _update_all_group_checkbox_states(self):
        """
        Aktualisiert alle Gruppen-Checkbox-Zustände basierend auf den Child-Items.
        Wird nach dem Laden von Dateien aufgerufen, um sicherzustellen, dass die Zustände korrekt sind.
        """
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return
        
        def _update_group_states_recursive(item):
            """Rekursiv alle Gruppen-Checkbox-Zustände aktualisieren"""
            if not item:
                return
            
            item_type = item.data(0, Qt.UserRole + 1)
            
            # Wenn es eine Gruppe ist, aktualisiere ihre Checkbox-Zustände
            if item_type == "group":
                # Aktualisiere alle Spalten (1=Enable, 2=Hide, 3=XY)
                for column in [1, 2, 3]:
                    self._update_group_checkbox_state(item, column)
            
            # Rekursiv für alle Childs
            for idx in range(item.childCount()):
                _update_group_states_recursive(item.child(idx))
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.surface_tree_widget.topLevelItemCount()):
            _update_group_states_recursive(self.surface_tree_widget.topLevelItem(i))
    
    def adjust_column_width_to_content(self):
        """Setzt die Spaltenbreiten auf feste Werte (gleich wie Source UI)"""
        if not hasattr(self, 'surface_tree_widget') or not self.surface_tree_widget:
            return
        # Gleiche Spaltenbreiten wie in Source UI
        self.surface_tree_widget.setColumnWidth(0, 160)  # Surface Name - 10% schlanker (von 180px auf 160px)
        self.surface_tree_widget.setColumnWidth(1, 25)   # Enable
        self.surface_tree_widget.setColumnWidth(2, 25)   # Hide
        self.surface_tree_widget.setColumnWidth(3, 25)   # XY
    
    # ---- Surface Management -----------------------------------------
    
    def add_surface(self):
        """Fügt ein neues Surface hinzu"""
        surface_store = getattr(self.settings, 'surface_definitions', {})
        
        # Generiere neue Surface-ID - prüfe alle existierenden IDs
        index = 1
        existing_ids = set(surface_store.keys())
        while f"surface_{index}" in existing_ids:
            index += 1
        surface_id = f"surface_{index}"
        
        # Bestimme Ziel-Gruppe (ausgewähltes Item oder None für Top-Level)
        target_group_id = None
        selected_item = self.surface_tree_widget.currentItem()
        if selected_item:
            item_type = selected_item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                target_group_id = selected_item.data(0, Qt.UserRole)
                # Wenn Root-Gruppe, setze auf None
                if target_group_id == self._group_controller.root_group_id:
                    target_group_id = None
            elif item_type == "surface":
                # Surface - finde Parent-Gruppe
                parent = selected_item.parent()
                if parent:
                    target_group_id = parent.data(0, Qt.UserRole)
                    # Wenn Root-Gruppe, setze auf None
                    if target_group_id == self._group_controller.root_group_id:
                        target_group_id = None
        
        # Erstelle neues Surface mit korrekter group_id
        new_surface = SurfaceDefinition(
            surface_id=surface_id,
            name=f"Surface {index}",
            enabled=False,
            hidden=False,
            locked=False,
            points=[
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 1.0, "y": 0.0, "z": 0.0},
                {"x": 1.0, "y": 1.0, "z": 0.0},
                {"x": 0.0, "y": 1.0, "z": 0.0},
            ],
            group_id=target_group_id
        )
        # Setze XY-Status auf aktiv (per Default)
        new_surface.xy_enabled = True
        
        # Verwende Settings-Methode für konsistente Behandlung
        if hasattr(self.settings, 'add_surface_definition'):
            self.settings.add_surface_definition(surface_id, new_surface, make_active=False)
        else:
            surface_store[surface_id] = new_surface
            self.settings.surface_definitions = surface_store
        
        # Ordne Surface zur Gruppe zu
        self._group_controller.assign_surface_to_group(surface_id, target_group_id, create_missing=False)
        
        # Stelle sicher, dass Gruppen-Struktur aktuell ist
        self._group_controller.ensure_structure()
        
        # Lade TreeWidget neu
        self.load_surfaces()
        
        # Wähle das neue Surface aus (rekursiv durchsuchen)
        self._select_surface_in_tree(surface_id)
    
    def _select_surface_in_tree(self, surface_id, skip_overlays=False):
        """Wählt ein Surface im TreeWidget aus (rekursiv)
        
        Args:
            surface_id: Die ID des auszuwählenden Surfaces
            skip_overlays: Wenn True, wird update_overlays() nicht aufgerufen (Standard: False)
        """
        def find_item(parent_item=None):
            if parent_item is None:
                # Durchsuche Top-Level-Items
                for i in range(self.surface_tree_widget.topLevelItemCount()):
                    item = self.surface_tree_widget.topLevelItem(i)
                    found = find_item(item)
                    if found:
                        return found
            else:
                # Prüfe aktuelles Item
                item_surface_id = parent_item.data(0, Qt.UserRole)
                if isinstance(item_surface_id, dict):
                    item_surface_id = item_surface_id.get('id')
                if item_surface_id == surface_id:
                    return parent_item
                
                # Durchsuche Children
                for i in range(parent_item.childCount()):
                    child = parent_item.child(i)
                    found = find_item(child)
                    if found:
                        return found
            return None
        
        item = find_item()
        if item:
            # Blockiere Signale während der programmatischen Auswahl, um unnötige Plot-Updates zu vermeiden
            self.surface_tree_widget.blockSignals(True)
            try:
                self.surface_tree_widget.setCurrentItem(item)
                self.surface_tree_widget.scrollToItem(item)
                
                # Expandiere Parent-Gruppe falls vorhanden
                parent = item.parent()
                if parent:
                    parent.setExpanded(True)
            finally:
                # Signale wieder aktivieren
                self.surface_tree_widget.blockSignals(False)
            
            # Manuell die Auswahl-Handler aufrufen, aber ohne Plot-Update
            # (nur für UI-Synchronisation, z.B. Tab-Anzeige)
            try:
                # Aktualisiere nur die Tabs, nicht die Overlays
                self.show_surfaces_tab()
                
                # Setze highlight_ids direkt, ohne update_overlays aufzurufen
                # (update_overlays wird später durch den normalen Plot-Update-Mechanismus aufgerufen)
                item_type = item.data(0, Qt.UserRole + 1)
                highlight_ids = []
                surface_store = getattr(self.settings, "surface_definitions", {}) or {}
                
                if item_type == "surface":
                    surface_id_data = item.data(0, Qt.UserRole)
                    if isinstance(surface_id_data, dict):
                        surface_id_data = surface_id_data.get("id")
                    if isinstance(surface_id_data, str) and surface_id_data in surface_store:
                        highlight_ids = [surface_id_data]
                        setattr(self.settings, "active_surface_id", surface_id_data)
                        print(f"[DEBUG SURFACE CLICK] Surface item clicked: {surface_id_data}")
                elif item_type == "group":
                    group_id_data = item.data(0, Qt.UserRole)
                    if isinstance(group_id_data, dict):
                        group_id = group_id_data.get("id")
                    else:
                        group_id = group_id_data
                    if group_id:
                        highlight_ids = self._collect_all_surfaces_from_group(group_id, surface_store)
                        print(f"[DEBUG SURFACE CLICK] Group item clicked: {group_id}, surfaces: {highlight_ids}")
                    setattr(self.settings, "active_surface_id", None)
                
                setattr(self.settings, "active_surface_highlight_ids", highlight_ids)
                print(f"[DEBUG SURFACE CLICK] Set active_surface_highlight_ids = {highlight_ids}")
                
                # Overlays im 3D-Plot aktualisieren (nur visuell, keine Neuberechnung)
                # für rote Umrandung der ausgewählten Fläche (nur wenn nicht übersprungen)
                if not skip_overlays:
                    main_window = getattr(self, "main_window", None)
                    if (
                        main_window is not None
                        and hasattr(main_window, "draw_plots")
                        and hasattr(main_window.draw_plots, "draw_spl_plotter")
                    ):
                        draw_spl = main_window.draw_plots.draw_spl_plotter
                        if hasattr(draw_spl, "update_overlays"):
                            try:
                                print(f"[DEBUG SURFACE CLICK] Calling update_overlays...")
                                draw_spl.update_overlays(self.settings, self.container)
                                print(f"[DEBUG SURFACE CLICK] update_overlays returned")
                            except Exception as e:
                                # Fehler hier sollen die restliche UI nicht blockieren
                                print(f"[DEBUG SURFACE CLICK] ERROR in update_overlays: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"[DEBUG SURFACE CLICK] draw_spl has no update_overlays method")
                    else:
                        print(f"[DEBUG SURFACE CLICK] Cannot access draw_spl (main_window={main_window is not None})")
            except Exception:
                # Fehler hier sollen die Auswahl nicht blockieren
                pass
    
    def load_dxf(self):
        """Lädt DXF-Dateien über SurfaceDataImporter"""
        from Module_LFO.Modules_Data.SurfaceDataImporter import SurfaceDataImporter
        
        importer = SurfaceDataImporter(
            self.surface_dockWidget if hasattr(self, 'surface_dockWidget') else self.main_window,
            self.settings,
            self.container,
            group_manager=self._group_controller,
            main_window=self.main_window  # Übergebe main_window für Fortschrittsanzeige
        )
        
        result = importer.execute()
        if result:
            # Stelle sicher, dass Gruppen-Struktur aktuell ist
            self._group_controller.ensure_structure()
            # Lade TreeWidget neu
            self.load_surfaces()
            
            # Explizites Update des TreeWidgets
            if hasattr(self, 'surface_tree_widget'):
                self.surface_tree_widget.update()
                # expandAll() entfernt - Zustand wird bereits durch load_surfaces() wiederhergestellt
            
            # Aktualisiere Plots und Overlays (inkl. neue Surfaces im 3D-Plot)
            if hasattr(self.main_window, 'draw_plots') and hasattr(self.main_window.draw_plots, 'update_plots_for_surface_state'):
                self.main_window.draw_plots.update_plots_for_surface_state()
            elif hasattr(self.main_window, 'update_speaker_array_calculations'):
                # Fallback: Berechnungen aktualisieren
                # 🚀 OPTIMIERUNG: update_speaker_array_calculations() ruft intern plot_spl() auf,
                # was wiederum update_overlays() aufruft, daher ist der redundante Aufruf entfernt
                self.main_window.update_speaker_array_calculations()
    
    def _get_surface(self, surface_id):
        """Holt ein Surface aus den Settings"""
        surface_store = getattr(self.settings, 'surface_definitions', {})
        return surface_store.get(surface_id)
    
    # ---- Event Handlers ---------------------------------------------
    
    def on_surface_enable_changed(self, surface_id, state, skip_calculations=False):
        """Wird aufgerufen, wenn sich der Enable-Status eines Surfaces ändert. Bei Mehrfachauswahl werden alle ausgewählten Surfaces aktualisiert."""
        enable_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.surface_tree_widget.selectedItems()
        surfaces_to_update = []
        
        if len(selected_items) > 1:
            # Mehrfachauswahl: Sammle alle ausgewählten Surfaces
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type == "surface":
                        item_surface_id = item.data(0, Qt.UserRole)
                        if isinstance(item_surface_id, dict):
                            item_surface_id = item_surface_id.get('id')
                        if item_surface_id:
                            surfaces_to_update.append(item_surface_id)
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Nur das aktuelle Surface
            surfaces_to_update = [surface_id]
        
        # Wende Enable auf alle Surfaces an
        for sid in surfaces_to_update:
            surface = self._get_surface(sid)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    surface.enabled = enable_value
                else:
                    surface['enabled'] = enable_value
                
                # Aktualisiere auch die Checkbox im TreeWidget
                item = self._find_tree_item_by_id(sid)
                if item:
                    enable_checkbox = self.surface_tree_widget.itemWidget(item, 1)
                    if enable_checkbox:
                        enable_checkbox.blockSignals(True)
                        # Verwende setCheckState für tristate-Checkboxen
                        enable_checkbox.setCheckState(Qt.Checked if enable_value else Qt.Unchecked)
                        enable_checkbox.blockSignals(False)
                    
                    # Aktualisiere Gruppen-Checkbox-Zustand (falls Surface in Gruppe)
                    # Aktualisiere rekursiv alle Parent-Gruppen
                    parent = item.parent()
                    while parent:
                        self._update_group_checkbox_state(parent, 1)
                        parent = parent.parent()
        
        # Prüfe, ob Surface aktiviert oder deaktiviert wurde
        if not skip_calculations:
            if enable_value:
                # Surface aktiviert: Nur dieses Surface neu berechnen (wenn autocalc aktiv und nicht hidden)
                for sid in surfaces_to_update:
                    surface = self._get_surface(sid)
                    if surface:
                        is_hidden = False
                        if isinstance(surface, SurfaceDefinition):
                            is_hidden = bool(getattr(surface, 'hidden', False))
                        else:
                            is_hidden = bool(surface.get('hidden', False))
                        
                        # Nur berechnen wenn nicht hidden
                        if not is_hidden:
                            self.calculate_single_surface(sid)
            else:
                # Surface deaktiviert: Nur SPL Plot auf diesem Surface entfernen, andere Plots belassen
                if (hasattr(self.main_window, 'draw_plots') and 
                    hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                    plotter = self.main_window.draw_plots.draw_spl_plotter
                    if plotter:
                        # Entferne nur den Actor für dieses Surface
                        for sid in surfaces_to_update:
                            # Entferne horizontale Surface-Actors
                            if hasattr(plotter, '_surface_actors') and sid in plotter._surface_actors:
                                actor = plotter._surface_actors[sid]
                                try:
                                    plotter.plotter.remove_actor(actor)
                                except Exception:
                                    pass
                                del plotter._surface_actors[sid]
                            # Entferne auch Texture-Actor falls vorhanden
                            if hasattr(plotter, '_surface_texture_actors') and sid in plotter._surface_texture_actors:
                                tex_data = plotter._surface_texture_actors[sid]
                                actor = None
                                if isinstance(tex_data, dict):
                                    actor = tex_data.get("actor")
                                elif tex_data is not None:
                                    actor = tex_data
                                if actor is not None:
                                    try:
                                        plotter.plotter.remove_actor(actor)
                                    except Exception:
                                        pass
                                del plotter._surface_texture_actors[sid]
                            
                            # Entferne vertikale Surface-Actors (falls vorhanden)
                            if hasattr(plotter, '_vertical_surface_meshes'):
                                actor_name = f"vertical_spl_{sid}"
                                if actor_name in plotter._vertical_surface_meshes:
                                    actor = plotter._vertical_surface_meshes[actor_name]
                                    try:
                                        plotter.plotter.remove_actor(actor)
                                    except Exception:
                                        pass
                                    del plotter._vertical_surface_meshes[actor_name]
                        
                        # Aktualisiere Overlays (zeigt Empty Plot für disabled Surface)
                        if hasattr(plotter, 'update_overlays'):
                            plotter.update_overlays(self.settings, self.container)
                        
                        # Rufe _update_vertical_spl_surfaces_from_grids auf, um sicherzustellen,
                        # dass vertikale Surfaces korrekt aktualisiert werden
                        # (disabled Surfaces werden dort übersprungen und entfernt)
                        if hasattr(plotter, '_update_vertical_spl_surfaces_from_grids'):
                            plotter._update_vertical_spl_surfaces_from_grids()
    
    def on_surface_hide_changed(self, surface_id, state, skip_calculations=False):
        """Wird aufgerufen, wenn sich der Hide-Status eines Surfaces ändert. Bei Mehrfachauswahl werden alle ausgewählten Surfaces aktualisiert."""
        hide_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.surface_tree_widget.selectedItems()
        surfaces_to_update = []
        
        if len(selected_items) > 1:
            # Mehrfachauswahl: Sammle alle ausgewählten Surfaces
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type == "surface":
                        item_surface_id = item.data(0, Qt.UserRole)
                        if isinstance(item_surface_id, dict):
                            item_surface_id = item_surface_id.get('id')
                        if item_surface_id:
                            surfaces_to_update.append(item_surface_id)
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Nur das aktuelle Surface
            surfaces_to_update = [surface_id]
        
        # Wende Hide auf alle Surfaces an
        for sid in surfaces_to_update:
            surface = self._get_surface(sid)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    surface.hidden = hide_value
                else:
                    surface['hidden'] = hide_value
                
                # Speichere Hide-Status in Settings
                if hasattr(self.settings, 'set_surface_hidden'):
                    self.settings.set_surface_hidden(sid, hide_value)
                
                # Aktualisiere auch die Checkbox im TreeWidget
                item = self._find_tree_item_by_id(sid)
                if item:
                    hide_checkbox = self.surface_tree_widget.itemWidget(item, 2)
                    if hide_checkbox:
                        hide_checkbox.blockSignals(True)
                        # Verwende setCheckState für tristate-Checkboxen
                        hide_checkbox.setCheckState(Qt.Checked if hide_value else Qt.Unchecked)
                        hide_checkbox.blockSignals(False)
                    
                    # Aktualisiere Gruppen-Checkbox-Zustand (falls Surface in Gruppe)
                    # Aktualisiere rekursiv alle Parent-Gruppen
                    parent = item.parent()
                    while parent:
                        self._update_group_checkbox_state(parent, 2)
                        parent = parent.parent()
        
        # Prüfe, ob Hide aktiviert oder deaktiviert wird
        if not skip_calculations:
            if hide_value:
                # Hide aktiviert: Nur Overlays aktualisieren (entfernt Fläche, SPL Plot, Achslinien, Rahmen)
                if (hasattr(self.main_window, 'draw_plots') and 
                    hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                    plotter = self.main_window.draw_plots.draw_spl_plotter
                    if plotter:
                        # Entferne SPL Plot für versteckte Surfaces
                        for sid in surfaces_to_update:
                            surface = self._get_surface(sid)
                            if surface:
                                is_enabled = False
                                if isinstance(surface, SurfaceDefinition):
                                    is_enabled = bool(getattr(surface, 'enabled', False))
                                else:
                                    is_enabled = bool(surface.get('enabled', False))
                                
                                # Wenn enabled und SPL geplottet, entferne SPL Plot für dieses Surface
                                if is_enabled:
                                    # Entferne horizontale Surface-Actors
                                    if hasattr(plotter, '_surface_actors') and sid in plotter._surface_actors:
                                        actor = plotter._surface_actors[sid]
                                        try:
                                            plotter.plotter.remove_actor(actor)
                                        except Exception:
                                            pass
                                        del plotter._surface_actors[sid]
                                    # Entferne auch Texture-Actor falls vorhanden
                                    if hasattr(plotter, '_surface_texture_actors') and sid in plotter._surface_texture_actors:
                                        tex_data = plotter._surface_texture_actors[sid]
                                        actor = None
                                        if isinstance(tex_data, dict):
                                            actor = tex_data.get("actor")
                                        elif tex_data is not None:
                                            actor = tex_data
                                        if actor is not None:
                                            try:
                                                plotter.plotter.remove_actor(actor)
                                            except Exception:
                                                pass
                                        del plotter._surface_texture_actors[sid]
                                    
                                    # Entferne vertikale Surface-Actors (falls vorhanden)
                                    # Vertikale Surfaces werden in _vertical_surface_meshes gespeichert
                                    # mit Actor-Namen im Format "vertical_spl_<surface_id>"
                                    if hasattr(plotter, '_vertical_surface_meshes'):
                                        actor_name = f"vertical_spl_{sid}"
                                        if actor_name in plotter._vertical_surface_meshes:
                                            actor = plotter._vertical_surface_meshes[actor_name]
                                            try:
                                                plotter.plotter.remove_actor(actor)
                                            except Exception:
                                                pass
                                            del plotter._vertical_surface_meshes[actor_name]
                                    
                                    # Rufe _update_vertical_spl_surfaces_from_grids auf, um sicherzustellen,
                                    # dass vertikale Surfaces korrekt aktualisiert werden
                                    # (hidden Surfaces werden dort übersprungen und entfernt)
                                    if hasattr(plotter, '_update_vertical_spl_surfaces_from_grids'):
                                        plotter._update_vertical_spl_surfaces_from_grids()
                        
                        # Aktualisiere Overlays (entfernt Fläche, Achslinien, Rahmen)
                        # Die Achsenlinien werden automatisch entfernt, da das Surface nicht mehr in active_surfaces ist
                        # (hidden Surfaces werden von _get_active_xy_surfaces gefiltert)
                        # Wichtig: axis_signature in _compute_overlay_signatures verwendet _get_active_xy_surfaces,
                        # die nur Surfaces mit hidden=False zurückgibt. Wenn hidden=True wird,
                        # ändert sich die Signatur automatisch, und update_overlays ruft draw_axis_lines auf.
                        # Zusätzlich setzen wir _last_axis_state zurück, um sicherzustellen, dass neu gezeichnet wird.
                        if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                            # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                            plotter.overlay_axis._last_axis_state = None
                        # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                        if hasattr(plotter, '_last_overlay_signatures'):
                            # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                            if isinstance(plotter._last_overlay_signatures, dict):
                                plotter._last_overlay_signatures.pop('axis', None)
                        if hasattr(plotter, 'update_overlays'):
                            plotter.update_overlays(self.settings, self.container)
            else:
                # Hide deaktiviert: Validiere Surface und berechne/plotte wenn enabled
                for sid in surfaces_to_update:
                    surface = self._get_surface(sid)
                    if surface:
                        is_enabled = False
                        if isinstance(surface, SurfaceDefinition):
                            is_enabled = bool(getattr(surface, 'enabled', False))
                        else:
                            is_enabled = bool(surface.get('enabled', False))
                        
                        # 🎯 WICHTIG: Validiere Surface IMMER, auch wenn nicht enabled
                        # (damit UI korrekt aktualisiert wird und Validierungsfehler angezeigt werden)
                        from Module_LFO.Modules_Data.SurfaceValidator import validate_and_optimize_surface, triangulate_points
                        surface_obj = surface if isinstance(surface, SurfaceDefinition) else SurfaceDefinition.from_dict(sid, surface)
                        validation_result = validate_and_optimize_surface(
                            surface_obj,
                            round_to_cm=False,
                            remove_redundant=False,
                        )
                        
                        # Zusätzliche Prüfung: Triangulation für Surfaces mit 4+ Punkten
                        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                        is_valid_for_spl = validation_result.is_valid
                        if is_valid_for_spl and len(points) >= 4:
                            try:
                                triangles = triangulate_points(points)
                                if not triangles or len(triangles) == 0:
                                    is_valid_for_spl = False
                            except Exception:
                                is_valid_for_spl = False
                        
                        # Stelle sicher, dass Surface-Definitionen in settings aktualisiert sind
                        if hasattr(self.settings, 'surface_definitions'):
                            surface_store = self.settings.surface_definitions
                            if surface_store is None:
                                surface_store = {}
                            if isinstance(surface, SurfaceDefinition):
                                surface_store[sid] = surface
                            else:
                                surface_store[sid] = SurfaceDefinition.from_dict(sid, surface)
                            self.settings.surface_definitions = surface_store
                        
                        # Berechne nur wenn enabled und valid
                        if is_enabled:
                            if is_valid_for_spl:
                                # Surface ist enabled und valid - berechne und plotte
                                self.calculate_single_surface(sid)
                            else:
                                # Surface ist enabled aber invalid - entferne SPL-Daten
                                self._remove_spl_data_for_surface(sid)
                            
                            # Nach der Berechnung auch Overlays aktualisieren, damit XY-Linien gezeichnet werden
                            # (wenn xy_enabled=True)
                            if (hasattr(self.main_window, 'draw_plots') and 
                                hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                                plotter = self.main_window.draw_plots.draw_spl_plotter
                                if plotter:
                                    # Setze Signatur zurück, damit Achsenlinien neu gezeichnet werden
                                    if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                        plotter.overlay_axis._last_axis_state = None
                                    # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                                    if hasattr(plotter, '_last_overlay_signatures'):
                                        # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                        if isinstance(plotter._last_overlay_signatures, dict):
                                            plotter._last_overlay_signatures.pop('axis', None)
                                    if hasattr(plotter, 'update_overlays'):
                                        plotter.update_overlays(self.settings, self.container)
                        else:
                            # Surface ist nicht enabled - nur Overlays aktualisieren (zeigt Surface wieder)
                            # Aber Validierung wurde bereits ausgeführt, damit UI korrekt aktualisiert wird
                            if (hasattr(self.main_window, 'draw_plots') and 
                                hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                                plotter = self.main_window.draw_plots.draw_spl_plotter
                                if plotter:
                                    # Setze Signatur zurück, damit Achsenlinien neu gezeichnet werden
                                    if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                        plotter.overlay_axis._last_axis_state = None
                                    # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                                    if hasattr(plotter, '_last_overlay_signatures'):
                                        # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                        if isinstance(plotter._last_overlay_signatures, dict):
                                            plotter._last_overlay_signatures.pop('axis', None)
                                    if hasattr(plotter, 'update_overlays'):
                                        plotter.update_overlays(self.settings, self.container)
    
    def on_surface_xy_changed(self, surface_id, state, skip_calculations=False):
        """Wird aufgerufen, wenn sich der XY-Status eines Surfaces ändert"""
        surface = self._get_surface(surface_id)
        if surface:
            checked = (state == Qt.Checked)
            if isinstance(surface, SurfaceDefinition):
                surface.xy_enabled = checked
                is_hidden = bool(getattr(surface, 'hidden', False))
                is_enabled = bool(getattr(surface, 'enabled', False))
            else:
                surface['xy_enabled'] = checked
                is_hidden = bool(surface.get('hidden', False))
                is_enabled = bool(surface.get('enabled', False))
            
            # Aktualisiere Gruppen-Checkbox-Zustand (falls Surface in Gruppe)
            item = self._find_tree_item_by_id(surface_id)
            if item:
                xy_checkbox = self.surface_tree_widget.itemWidget(item, 3)
                if xy_checkbox:
                    xy_checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    xy_checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    xy_checkbox.blockSignals(False)
                
                # Aktualisiere rekursiv alle Parent-Gruppen
                parent = item.parent()
                while parent:
                    self._update_group_checkbox_state(parent, 3)
                    parent = parent.parent()

            # Nur neu berechnen / Overlays aktualisieren, wenn Surface nicht versteckt ist
            if not skip_calculations and not is_hidden:
                if checked:
                    # XY Checkbox aktiviert: Linie auf Surface zeichnen und XY Plot aktualisieren
                    # Wichtig: axis_signature in _compute_overlay_signatures verwendet _get_active_xy_surfaces,
                    # die nur Surfaces mit xy_enabled=True zurückgibt. Wenn xy_enabled=True wird,
                    # ändert sich die Signatur automatisch, und update_overlays ruft draw_axis_lines auf.
                    # Zusätzlich setzen wir die Signatur zurück, um sicherzustellen, dass die Änderung erkannt wird.
                    if (
                        hasattr(self.main_window, 'draw_plots')
                        and hasattr(self.main_window.draw_plots, 'draw_spl_plotter')
                    ):
                        plotter = self.main_window.draw_plots.draw_spl_plotter
                        if plotter:
                            # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                            if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                plotter.overlay_axis._last_axis_state = None
                            # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                            if hasattr(plotter, '_last_overlay_signatures'):
                                # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                if isinstance(plotter._last_overlay_signatures, dict):
                                    plotter._last_overlay_signatures.pop('axis', None)
                            # Aktualisiere Overlays (zeichnet Linie auf Surface)
                            if hasattr(plotter, 'update_overlays'):
                                plotter.update_overlays(self.settings, self.container)
                    
                    # XY Plot aktualisieren, wenn es Änderungen gegeben hat
                    if hasattr(self.main_window, 'calculate_axes') and is_enabled:
                        self.main_window.calculate_axes(update_plot=True)
                else:
                    # XY Checkbox deaktiviert: Achsenlinien auf dem Surface entfernen
                    # Wichtig: axis_signature in _compute_overlay_signatures verwendet _get_active_xy_surfaces,
                    # die nur Surfaces mit xy_enabled=True zurückgibt. Wenn xy_enabled=False wird,
                    # ändert sich die Signatur automatisch, und update_overlays ruft draw_axis_lines auf.
                    # Zusätzlich setzen wir _last_axis_state zurück, um sicherzustellen, dass neu gezeichnet wird.
                    if (
                        hasattr(self.main_window, 'draw_plots')
                        and hasattr(self.main_window.draw_plots, 'draw_spl_plotter')
                    ):
                        plotter = self.main_window.draw_plots.draw_spl_plotter
                        if plotter:
                            # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                            if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                plotter.overlay_axis._last_axis_state = None
                            # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                            if hasattr(plotter, '_last_overlay_signatures'):
                                # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                if isinstance(plotter._last_overlay_signatures, dict):
                                    plotter._last_overlay_signatures.pop('axis', None)
                            # Aktualisiere Overlays (entfernt Achsenlinien auf diesem Surface)
                            if hasattr(plotter, 'update_overlays'):
                                plotter.update_overlays(self.settings, self.container)
                    
                    # Entferne Daten aus XY Plot für dieses Surface
                if hasattr(self.main_window, 'calculate_axes'):
                        self.main_window.calculate_axes(update_plot=True)
    
    def on_group_enable_changed(self, group_item, state):
        """Wird aufgerufen, wenn sich der Enable-Status einer Gruppe ändert. Bei Mehrfachauswahl werden alle ausgewählten Gruppen aktualisiert."""
        # Wenn PartiallyChecked geklickt wird, setze auf Checked (alle aktivieren)
        if state == Qt.PartiallyChecked:
            enable_value = True
        else:
            enable_value = (state == Qt.Checked)
        group_id_data = group_item.data(0, Qt.UserRole)
        if isinstance(group_id_data, dict):
            group_id = group_id_data.get("id")
        else:
            group_id = group_id_data
        print(f"[DEBUG on_group_enable_changed] Gruppe '{group_id}': enable_value={enable_value}, childCount={group_item.childCount()}")
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.surface_tree_widget.selectedItems()
        groups_to_update = []
        
        if len(selected_items) > 1:
            # Mehrfachauswahl: Sammle alle ausgewählten Gruppen
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type == "group":
                        groups_to_update.append(item)
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Nur die aktuelle Gruppe
            groups_to_update = [group_item]
        
        # Wende Enable auf alle Gruppen an
        for group in groups_to_update:
            try:
                # Setze Gruppen-Checkbox explizit auf den neuen Zustand
                group_checkbox = self.surface_tree_widget.itemWidget(group, 1)
                if group_checkbox:
                    group_checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    group_checkbox.setCheckState(Qt.Checked if enable_value else Qt.Unchecked)
                    group_checkbox.blockSignals(False)
                
                group_id_data = group.data(0, Qt.UserRole)
                if isinstance(group_id_data, dict):
                    group_id = group_id_data.get("id")
                else:
                    group_id = group_id_data
                
                if group_id:
                    self._group_controller.set_surface_group_enabled(group_id, enable_value)
                    
                    # Speichere Status in lokalem Cache
                    if group_id in self.surface_groups:
                        self.surface_groups[group_id]['enabled'] = enable_value
                    
                    # Aktualisiere alle Child-Checkboxen (ohne Berechnungen, bis alle Zustände gespeichert sind)
                    # skip_state_update=True: Überspringe _update_group_checkbox_state, da wir die Checkbox bereits explizit gesetzt haben
                    # update_data=True: WICHTIG - aktualisiere die tatsächlichen Surface-Daten
                    self._update_group_child_checkboxes(group, 1, enable_value, update_data=True, skip_calculations=True, skip_state_update=True)
                    
                    # Aktualisiere Gruppen-Checkbox-Zustand explizit NACH allen Child-Updates
                    # (um sicherzustellen, dass der Zustand korrekt ist, auch wenn alle Childs aktualisiert wurden)
                    # WICHTIG: Dies muss NACH der Datenaktualisierung erfolgen, damit der korrekte Zustand gelesen wird
                    self._update_group_checkbox_state(group, 1)
                    
                    # Aktualisiere Parent-Gruppen rekursiv
                    parent = group.parent()
                    while parent:
                        self._update_group_checkbox_state(parent, 1)
                        parent = parent.parent()
            except RuntimeError:
                # Item wurde gelöscht, überspringe
                continue
        
        # 🎯 WICHTIG: Aktualisiere Berechnungen erst NACH allen Zustandsänderungen
        if hasattr(self.main_window, 'update_speaker_array_calculations'):
            self.main_window.update_speaker_array_calculations()
    
    def on_group_hide_changed(self, group_item, state):
        """Wird aufgerufen, wenn sich der Hide-Status einer Gruppe ändert. Bei Mehrfachauswahl werden alle ausgewählten Gruppen aktualisiert."""
        # Wenn PartiallyChecked geklickt wird, setze auf Checked (alle verstecken)
        if state == Qt.PartiallyChecked:
            hide_value = True
        else:
            hide_value = (state == Qt.Checked)
        group_id_data = group_item.data(0, Qt.UserRole)
        if isinstance(group_id_data, dict):
            group_id = group_id_data.get("id")
        else:
            group_id = group_id_data
        print(f"[DEBUG on_group_hide_changed] Gruppe '{group_id}': hide_value={hide_value}, childCount={group_item.childCount()}")
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.surface_tree_widget.selectedItems()
        groups_to_update = []
        
        if len(selected_items) > 1:
            # Mehrfachauswahl: Sammle alle ausgewählten Gruppen
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type == "group":
                        groups_to_update.append(item)
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Nur die aktuelle Gruppe
            groups_to_update = [group_item]
        
        # Wende Hide auf alle Gruppen an
        for group in groups_to_update:
            try:
                # Setze Gruppen-Checkbox explizit auf den neuen Zustand
                group_checkbox = self.surface_tree_widget.itemWidget(group, 2)
                if group_checkbox:
                    group_checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    group_checkbox.setCheckState(Qt.Checked if hide_value else Qt.Unchecked)
                    group_checkbox.blockSignals(False)
                
                group_id_data = group.data(0, Qt.UserRole)
                if isinstance(group_id_data, dict):
                    group_id = group_id_data.get("id")
                else:
                    group_id = group_id_data
                
                if group_id:
                    self._group_controller.set_surface_group_hidden(group_id, hide_value)
                    
                    # Speichere Status in lokalem Cache
                    if group_id in self.surface_groups:
                        self.surface_groups[group_id]['hidden'] = hide_value
                    
                    # Aktualisiere alle Child-Checkboxen (ohne Berechnungen, bis alle Zustände gespeichert sind)
                    # skip_state_update=True: Überspringe _update_group_checkbox_state, da wir die Checkbox bereits explizit gesetzt haben
                    # update_data=True: WICHTIG - aktualisiere die tatsächlichen Surface-Daten
                    self._update_group_child_checkboxes(group, 2, hide_value, update_data=True, skip_calculations=True, skip_state_update=True)
                    
                    # Aktualisiere Gruppen-Checkbox-Zustand explizit NACH allen Child-Updates
                    # (um sicherzustellen, dass der Zustand korrekt ist, auch wenn alle Childs aktualisiert wurden)
                    self._update_group_checkbox_state(group, 2)
                    
                    # Aktualisiere Parent-Gruppen rekursiv
                    parent = group.parent()
                    while parent:
                        self._update_group_checkbox_state(parent, 2)
                        parent = parent.parent()
            except RuntimeError:
                # Item wurde gelöscht, überspringe
                continue
        
        # 🎯 WICHTIG: Aktualisiere Plots und Overlays erst NACH allen Zustandsänderungen
        if hasattr(self.main_window, 'draw_plots') and hasattr(self.main_window.draw_plots, 'update_plots_for_surface_state'):
            self.main_window.draw_plots.update_plots_for_surface_state()
        elif hasattr(self.main_window, 'update_speaker_array_calculations'):
            # Fallback: Berechnungen aktualisieren
            # 🚀 OPTIMIERUNG: update_speaker_array_calculations() ruft intern plot_spl() auf,
            # was wiederum update_overlays() aufruft, daher ist der redundante Aufruf entfernt
            self.main_window.update_speaker_array_calculations()
    
    def on_group_xy_changed(self, group_item, state):
        """Wird aufgerufen, wenn sich der XY-Status einer Gruppe ändert"""
        group_id = group_item.data(0, Qt.UserRole)
        # Wenn PartiallyChecked geklickt wird, setze auf Checked (alle aktivieren)
        if state == Qt.PartiallyChecked:
            checked = True
        else:
            checked = (state == Qt.Checked)
        print(f"[DEBUG on_group_xy_changed] Gruppe '{group_id}': checked={checked}, childCount={group_item.childCount()}")
        
        # Setze Gruppen-Checkbox explizit auf den neuen Zustand
        group_checkbox = self.surface_tree_widget.itemWidget(group_item, 3)
        if group_checkbox:
            group_checkbox.blockSignals(True)
            # Verwende setCheckState für tristate-Checkboxen
            group_checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            group_checkbox.blockSignals(False)
        
        # Speichere XY-Status in der Gruppe
        group = self._group_controller.get_group(group_id)
        if group:
            group.xy_enabled = checked
        
        # Speichere Status in lokalem Cache
        if group_id in self.surface_groups:
            self.surface_groups[group_id]['xy_enabled'] = checked
        
        # Aktualisiere alle Child-Checkboxen (ohne Berechnungen, bis alle Zustände gespeichert sind)
        # skip_state_update=True: Überspringe _update_group_checkbox_state, da wir die Checkbox bereits explizit gesetzt haben
        # update_data=True: WICHTIG - aktualisiere die tatsächlichen Surface-Daten
        self._update_group_child_checkboxes(group_item, 3, checked, update_data=True, skip_calculations=True, skip_state_update=True)
        
        # Aktualisiere Gruppen-Checkbox-Zustand explizit NACH allen Child-Updates
        # (um sicherzustellen, dass der Zustand korrekt ist, auch wenn alle Childs aktualisiert wurden)
        self._update_group_checkbox_state(group_item, 3)
        
        # Aktualisiere Parent-Gruppen rekursiv
        parent = group_item.parent()
        while parent:
            self._update_group_checkbox_state(parent, 3)
            parent = parent.parent()

        # Nur neu berechnen / Overlays aktualisieren, wenn Gruppe nicht versteckt ist
        is_hidden = False
        is_enabled = True
        if group_id in self.surface_groups:
            is_hidden = bool(self.surface_groups[group_id].get('hidden', False))
            is_enabled = bool(self.surface_groups[group_id].get('enabled', True))
        elif group:
            # Fallback, falls Cache noch nicht befüllt ist
            is_hidden = bool(getattr(group, 'hidden', False))
            is_enabled = bool(getattr(group, 'enabled', True))

        if not is_hidden:
            # Leichte Variante: nur Overlays (XY-Achsen) neu zeichnen
            if (
                hasattr(self.main_window, 'draw_plots')
                and hasattr(self.main_window.draw_plots, 'draw_spl_plotter')
                and hasattr(self.main_window.draw_plots.draw_spl_plotter, 'update_overlays')
            ):
                self.main_window.draw_plots.draw_spl_plotter.update_overlays(self.settings, self.container)
            # Zusätzlich: Achsen-SPL neu berechnen, wenn Bedingungen erfüllt
            if hasattr(self.main_window, 'calculate_axes'):
                # Beim Aktivieren der Gruppe: nur wenn Gruppe sichtbar und enabled
                if checked and is_enabled and not is_hidden:
                    self.main_window.calculate_axes(update_plot=True)
                # Beim Deaktivieren: immer neu berechnen, um Kurven zu entfernen
                elif not checked:
                    self.main_window.calculate_axes(update_plot=True)
    
    def on_surface_item_text_changed(self, item, column):
        """Wird aufgerufen, wenn sich der Text eines Surface-Items ändert"""
        if column != 0:
            return
        
        item_type = item.data(0, Qt.UserRole + 1)
        if item_type == "group":
            # Gruppen-Name wurde geändert
            new_name = item.text(0)
            group_id = item.data(0, Qt.UserRole)
            self._group_controller.rename_surface_group(group_id, new_name)
        elif item_type == "surface":
            # Surface-Name wurde geändert
            new_name = item.text(0)
            surface_id = item.data(0, Qt.UserRole)
            surface = self._get_surface(surface_id)
            if surface:
                if isinstance(surface, SurfaceDefinition):
                    surface.name = new_name
                else:
                    surface['name'] = new_name
    
    # ---- Tab Management ---------------------------------------------
    
    def show_surfaces_tab(self):
        """Zeigt den Surface-Parameter-Tab basierend auf der Auswahl"""
        # Verwende currentItem(), falls nicht verfügbar, nimm das erste ausgewählte Item
        selected_item = self.surface_tree_widget.currentItem()
        if not selected_item:
            selected_items = self.surface_tree_widget.selectedItems()
            if selected_items:
                selected_item = selected_items[0]
                # Setze currentItem explizit
                self.surface_tree_widget.setCurrentItem(selected_item)
        
        # Wenn kein Item ausgewählt ist, entferne alle Tabs
        if not selected_item:
            if hasattr(self, 'tab_widget') and self.tab_widget is not None:
                while self.tab_widget.count() > 0:
                    self.tab_widget.removeTab(0)
            return
        
        item_type = selected_item.data(0, Qt.UserRole + 1)
        
        if item_type == "group":
            # Zeige Gruppen-Tab
            self.create_group_tab(selected_item)
        elif item_type == "surface":
            # Zeige Surface-Parameter-Tab
            surface_id = selected_item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            self.create_surface_parameter_tab(surface_id)
    
    def create_surface_tab(self):
        """Erstellt den Surface-Parameter-Tab (Placeholder)"""
        # Wird von create_surface_parameter_tab überschrieben
        pass
    
    def create_surface_parameter_tab(self, surface_id):
        """Erstellt den Tab mit Surface-Parametern - zeigt Koordinaten der Punkte"""
        # Entferne alle vorhandenen Tabs
        if hasattr(self, 'tab_widget') and self.tab_widget is not None:
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
        
        # Erstelle neuen Tab
        surface_tab = QWidget()
        self.tab_widget.addTab(surface_tab, "Surface Parameters")
        
        # Hauptlayout
        main_layout = QVBoxLayout(surface_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Schriftgröße
        font = QtGui.QFont()
        font.setPointSize(11)
        
        # Lade Surface-Daten
        surface = self._get_surface(surface_id)
        if not surface:
            return
        
        # TreeWidget für Koordinaten
        self.points_tree = QTreeWidget()
        
        # Feste Header-Labels (keine starre Achsen-Logik mehr)
        header_labels = ["Point", "X (m)", "Y (m)", "Z (m)", ""]  # Leere Spalte am Ende
        
        self.points_tree.setHeaderLabels(header_labels)
        self.points_tree.setRootIsDecorated(False)
        self.points_tree.setAlternatingRowColors(True)
        self.points_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.points_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points_tree.setUniformRowHeights(True)
        self.points_tree.setColumnWidth(0, 55)
        self.points_tree.setColumnWidth(1, 85)
        self.points_tree.setColumnWidth(2, 85)
        self.points_tree.setColumnWidth(3, 85)
        self.points_tree.setColumnWidth(4, 10)
        header_points = self.points_tree.header()
        header_points.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        header_points.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        header_points.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        header_points.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        header_points.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.points_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.points_tree.setFont(font)
        self.points_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.points_tree.customContextMenuRequested.connect(
            lambda pos: self._show_points_context_menu(pos, surface_id)
        )
        self.points_tree.setStyleSheet(
            "QTreeWidget { font-size: 11pt; }"
            "QTreeWidget::item { height: 24px; }"
        )
        
        # Lade Punkte
        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
        self._loading_points = True
        self.points_tree.blockSignals(True)
        self.points_tree.clear()
        
        # Validiere Surface um ungültige Felder zu bestimmen
        validation_result = validate_and_optimize_surface(
            surface if isinstance(surface, SurfaceDefinition) else SurfaceDefinition.from_dict(surface_id, surface),
            round_to_cm=False,
            remove_redundant=False,
        )
        invalid_fields_set = {(idx, coord) for idx, coord in (validation_result.invalid_fields or [])}
        
        for index, point in enumerate(points):
            item = QTreeWidgetItem(self.points_tree)
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            )
            item.setText(0, f"P{index + 1}")
            item.setData(0, Qt.UserRole, index)
            
            x_val = point.get('x', 0.0)
            y_val = point.get('y', 0.0)
            z_val = point.get('z', 0.0)
            
            item.setText(1, f"{float(x_val):.2f}" if x_val is not None else "")
            item.setText(2, f"{float(y_val):.2f}" if y_val is not None else "")
            item.setText(3, f"{float(z_val):.2f}" if z_val is not None else "")
            
            # Erstelle Editoren für X, Y, Z
            self._create_point_editor(item, 1, surface_id, index, 'x')
            self._create_point_editor(item, 2, surface_id, index, 'y')
            self._create_point_editor(item, 3, surface_id, index, 'z')
        
        # Nummeriere Punkte neu
        self._renumber_points()
        
        self.points_tree.blockSignals(False)
        self._loading_points = False
        
        main_layout.addWidget(self.points_tree)
        
        # Buttons zum Hinzufügen/Löschen von Punkten
        buttons_layout = QHBoxLayout()
        add_point_button = QPushButton("Add Point")
        add_point_button.setFont(font)
        add_point_button.setFixedHeight(24)
        add_point_button.clicked.connect(lambda: self._handle_add_point(surface_id))
        
        delete_point_button = QPushButton("Delete Point")
        delete_point_button.setFont(font)
        delete_point_button.setFixedHeight(24)
        delete_point_button.clicked.connect(lambda: self._handle_delete_point(surface_id))
        
        buttons_layout.addWidget(add_point_button)
        buttons_layout.addWidget(delete_point_button)
        buttons_layout.addStretch()
        
        main_layout.addLayout(buttons_layout)
    
    def create_group_tab_placeholder(self):
        """Placeholder für Gruppen-Tab"""
        pass
    
    def create_group_tab(self, group_item):
        """Erstellt die UI für Gruppen-Einstellungen mit Relativpositionsverarbeitung"""
        # Entferne alle vorhandenen Tabs
        if hasattr(self, 'tab_widget') and self.tab_widget is not None:
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
        
        # Erstelle neuen Tab
        group_tab = QWidget()
        self.tab_widget.addTab(group_tab, "Group Settings")
        
        # Hauptlayout
        main_layout = QVBoxLayout(group_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Schriftgröße
        font = QtGui.QFont()
        font.setPointSize(11)
        
        # ScrollArea für den gesamten Inhalt
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(250)
        scroll_area.setMaximumWidth(250)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(10)
        
        # Bereich 1: Change relative position
        relative_position_group = QGroupBox("Change relative position")
        relative_position_group.setFont(font)
        relative_position_layout = QGridLayout()
        relative_position_layout.setVerticalSpacing(3)
        relative_position_layout.setContentsMargins(5, 10, 5, 5)
        
        # Relative X-Position
        rel_x_label = QLabel("Relative X (m)")
        rel_x_label.setFont(font)
        self.group_rel_x_edit = QLineEdit()
        self.group_rel_x_edit.setFont(font)
        self.group_rel_x_edit.setFixedHeight(18)
        self.group_rel_x_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.group_rel_x_edit.setText("0.00")
        rel_x_label.setFixedWidth(150)
        self.group_rel_x_edit.setFixedWidth(40)
        relative_position_layout.addWidget(rel_x_label, 0, 0)
        relative_position_layout.addWidget(self.group_rel_x_edit, 0, 1)
        
        # Relative Y-Position
        rel_y_label = QLabel("Relative Y (m)")
        rel_y_label.setFont(font)
        self.group_rel_y_edit = QLineEdit()
        self.group_rel_y_edit.setFont(font)
        self.group_rel_y_edit.setFixedHeight(18)
        self.group_rel_y_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.group_rel_y_edit.setText("0.00")
        rel_y_label.setFixedWidth(150)
        self.group_rel_y_edit.setFixedWidth(40)
        relative_position_layout.addWidget(rel_y_label, 1, 0)
        relative_position_layout.addWidget(self.group_rel_y_edit, 1, 1)
        
        # Relative Z-Position
        rel_z_label = QLabel("Relative Z (m)")
        rel_z_label.setFont(font)
        self.group_rel_z_edit = QLineEdit()
        self.group_rel_z_edit.setFont(font)
        self.group_rel_z_edit.setFixedHeight(18)
        self.group_rel_z_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.group_rel_z_edit.setText("0.00")
        rel_z_label.setFixedWidth(150)
        self.group_rel_z_edit.setFixedWidth(40)
        relative_position_layout.addWidget(rel_z_label, 2, 0)
        relative_position_layout.addWidget(self.group_rel_z_edit, 2, 1)
        
        relative_position_group.setLayout(relative_position_layout)
        scroll_layout.addWidget(relative_position_group)
        
        # Spacer
        scroll_layout.addStretch()
        
        # Apply Changes Button
        # Extrahiere group_id vor dem Lambda, um zu vermeiden, dass auf ein gelöschtes Item zugegriffen wird
        group_id = group_item.data(0, Qt.UserRole) if group_item else None
        apply_button = QPushButton("Apply Changes")
        apply_button.setFont(font)
        apply_button.setFixedHeight(30)
        apply_button.clicked.connect(lambda checked, gid=group_id: self.apply_group_changes_by_id(gid))
        scroll_layout.addWidget(apply_button)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Lade aktuelle Werte aus der Gruppe
        self.load_group_values(group_item)
    
    def on_surface_name_changed(self, surface_id):
        """Wird aufgerufen, wenn sich der Surface-Name ändert"""
        if not hasattr(self, 'surface_name_edit'):
            return
        
        new_name = self.surface_name_edit.text()
        surface = self._get_surface(surface_id)
        if surface:
            if isinstance(surface, SurfaceDefinition):
                surface.name = new_name
            else:
                surface['name'] = new_name
            
            # Aktualisiere TreeWidget
            for i in range(self.surface_tree_widget.topLevelItemCount()):
                item = self.surface_tree_widget.topLevelItem(i)
                item_surface_id = item.data(0, Qt.UserRole)
                if isinstance(item_surface_id, dict):
                    item_surface_id = item_surface_id.get('id')
                if item_surface_id == surface_id:
                    item.setText(0, new_name)
                    break
    
    def _create_point_editor(self, item, column, surface_id, point_index, coord_name):
        """Erstellt einen LineEdit-Editor für einen Koordinatenwert"""
        editor = QLineEdit()
        editor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        editor.setText(item.text(column))
        editor.setFixedHeight(22)
        editor.setValidator(QDoubleValidator(-1000.0, 1000.0, 2))
        editor.editingFinished.connect(
            lambda: self._on_point_editor_finished(surface_id, point_index, coord_name, editor)
        )
        
        self.points_tree.setItemWidget(item, column, editor)
    
    def _on_point_editor_finished(self, surface_id, point_index, coord_name, editor):
        """Wird aufgerufen, wenn ein Koordinatenwert geändert wurde"""
        if self._loading_points:
            return
        
        text = editor.text().strip()
        if not text:
            text = "0"
        
        try:
            value = float(text)
        except ValueError:
            # Ungültiger Wert - setze zurück
            surface = self._get_surface(surface_id)
            if surface:
                points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                if 0 <= point_index < len(points):
                    old_value = points[point_index].get(coord_name, 0.0)
                    editor.setText(f"{float(old_value):.2f}")
            return
        
        # Aktualisiere Surface-Punkt
        surface = self._get_surface(surface_id)
        if surface:
            points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
            if 0 <= point_index < len(points):
                old_value = points[point_index].get(coord_name)
                points[point_index][coord_name] = value
                
                # Aktualisiere Anzeige
                formatted = f"{value:.2f}"
                editor.setText(formatted)
                
                # Aktualisiere UI: Markiere ungültige Felder
                self._update_surface_validation_ui(surface_id)
                
                # Prüfe ob Surface versteckt ist - wenn ja, keine Calc/Plot Aktualisierung
                if isinstance(surface, SurfaceDefinition):
                    hidden = getattr(surface, 'hidden', False)
                else:
                    hidden = surface.get('hidden', False)
                
                # Validiere Surface
                validation_result = validate_and_optimize_surface(
                    surface if isinstance(surface, SurfaceDefinition) else SurfaceDefinition.from_dict(surface_id, surface),
                    round_to_cm=False,  # Keine automatische Rundung beim manuellen Editieren
                    remove_redundant=False,  # Keine automatische Entfernung beim manuellen Editieren
                )
                
                # 🎯 ZUSÄTZLICHE PRÜFUNG: Für Surfaces mit 4 oder mehr Punkten prüfe ob Triangulation möglich ist
                # (auch 4 Punkte müssen trianguliert werden können, wenn sie nicht planar sind)
                is_valid_for_spl = validation_result.is_valid
                if is_valid_for_spl and len(points) >= 4:
                    try:
                        # Versuche Triangulation
                        triangles = triangulate_points(points)
                        if not triangles or len(triangles) == 0:
                            # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                            is_valid_for_spl = False
                            print(f"[DEBUG] Surface '{surface_id}' mit {len(points)} Punkten: Triangulation fehlgeschlagen (keine Dreiecke)")
                    except Exception as e:
                        # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                        is_valid_for_spl = False
                        print(f"[DEBUG] Surface '{surface_id}' mit {len(points)} Punkten: Triangulation fehlgeschlagen ({e})")
                
                # 🎯 Plot immer aktualisieren (auch bei ungültigen Surfaces), damit geänderte Koordinaten sichtbar werden
                if not hidden:
                    # 🚨 PRÜFUNG: Verhindere neue Berechnung während laufender Berechnung
                    calculation_running = False
                    try:
                        if hasattr(self.main_window, '_current_progress_session') and self.main_window._current_progress_session:
                            calculation_running = not self.main_window._current_progress_session.is_cancelled()
                    except Exception:
                        pass
                    
                    if calculation_running:
                        # Berechnung läuft bereits - nur Plot aktualisieren, keine neue Berechnung starten
                        print(f"[DEBUG] Berechnung läuft bereits - überspringe update_speaker_array_calculations für Surface '{surface_id}'")
                        if hasattr(self.main_window, 'plot_spl'):
                            self.main_window.plot_spl(update_axes=False)
                    else:
                        # Keine laufende Berechnung - normale Verarbeitung
                        # 🎯 WICHTIG: Stelle sicher, dass Surface-Definitionen in self.settings aktualisiert sind
                        # bevor die Achsen berechnet werden (die Achsen verwenden self.settings.surface_definitions)
                        # Da surface direkt aus _get_surface() kommt und die Punkte direkt modifiziert werden,
                        # sollte die Referenz in settings bereits aktualisiert sein. Aber wir stellen sicher,
                        # dass die Surface-Definitionen explizit in settings gespeichert werden.
                        if hasattr(self.settings, 'surface_definitions'):
                            surface_store = self.settings.surface_definitions
                            if surface_store is None:
                                surface_store = {}
                            
                            # Stelle sicher, dass die aktualisierte Surface-Definition in settings gespeichert wird
                            if isinstance(surface, SurfaceDefinition):
                                surface_store[surface_id] = surface
                            else:
                                # Surface ist ein Dict - konvertiere zu SurfaceDefinition und aktualisiere
                                surface_obj = SurfaceDefinition.from_dict(surface_id, surface)
                                surface_store[surface_id] = surface_obj
                            
                            self.settings.surface_definitions = surface_store
                        
                        # 🎯 Setze Signatur zurück, damit update_overlays (wird von plot_spl aufgerufen) die Änderung erkennt
                        # und XY-Linien im 3D-Plot neu zeichnet
                        if (hasattr(self.main_window, 'draw_plots') and 
                            hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                            plotter = self.main_window.draw_plots.draw_spl_plotter
                            if plotter:
                                print(f"[DEBUG UISurfaceManager] Setze axis Signatur zurück für 3D-Plot-Update")
                                # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                                if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                    plotter.overlay_axis._last_axis_state = None
                                # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                                if hasattr(plotter, '_last_overlay_signatures'):
                                    # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                    if isinstance(plotter._last_overlay_signatures, dict):
                                        plotter._last_overlay_signatures.pop('axis', None)
                        
                        # 🎯 Achsen ZUERST neu berechnen, da Surface-Koordinaten geändert wurden
                        # (muss vor update_speaker_array_calculations erfolgen, damit aktualisierte Koordinaten verwendet werden)
                        print(f"[DEBUG UISurfaceManager] Surface '{surface_id}' Koordinaten geändert - rufe calculate_axes() auf")
                        if hasattr(self.main_window, 'calculate_axes'):
                            print(f"[DEBUG UISurfaceManager] calculate_axes() wird aufgerufen mit update_plot=True")
                            self.main_window.calculate_axes(update_plot=True)
                        else:
                            print(f"[DEBUG UISurfaceManager] FEHLER: main_window hat kein calculate_axes()")
                        
                        if is_valid_for_spl:
                            # Surface ist gültig - aktualisiere Berechnungen/Plot
                            if hasattr(self.main_window, 'update_speaker_array_calculations'):
                                self.main_window.update_speaker_array_calculations()
                            else:
                                # Fallback: Wenn update_speaker_array_calculations nicht verfügbar ist,
                                # rufe plot_spl() auf, damit update_overlays() die Achsen neu zeichnet
                                if hasattr(self.main_window, 'plot_spl'):
                                    print(f"[DEBUG UISurfaceManager] Rufe plot_spl() auf, damit 3D-Plot-Achsen aktualisiert werden")
                                    self.main_window.plot_spl(update_axes=False)
                        else:
                            # Surface ist ungültig - entferne SPL-Daten für diese Surface und zeige nur graue Fläche
                            self._remove_spl_data_for_surface(surface_id)
                            
                            # 🎯 Setze Signatur zurück, damit update_overlays (wird von plot_spl aufgerufen) die Änderung erkennt
                            # und XY-Linien im 3D-Plot neu zeichnet
                            if (hasattr(self.main_window, 'draw_plots') and 
                                hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                                plotter = self.main_window.draw_plots.draw_spl_plotter
                                if plotter:
                                    print(f"[DEBUG UISurfaceManager] Setze axis Signatur zurück für 3D-Plot-Update (ungültige Surface)")
                                    # Setze Signatur zurück, damit draw_axis_lines definitiv neu zeichnet
                                    if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                                        plotter.overlay_axis._last_axis_state = None
                                    # Setze auch die overlay_signature zurück, damit update_overlays die Änderung erkennt
                                    if hasattr(plotter, '_last_overlay_signatures'):
                                        # Entferne 'axis' aus der Signatur, damit es neu berechnet wird
                                        if isinstance(plotter._last_overlay_signatures, dict):
                                            plotter._last_overlay_signatures.pop('axis', None)
                            
                            # Aktualisiere Plot (zeigt nur graue Fläche, keine SPL-Daten)
                            if hasattr(self.main_window, 'plot_spl'):
                                self.main_window.plot_spl(update_axes=False)
                            # 🎯 Achsen neu berechnen, da Surface-Koordinaten geändert wurden
                            # (auch bei ungültigen Surfaces müssen Achsen neu berechnet werden)
                            if hasattr(self.main_window, 'calculate_axes'):
                                self.main_window.calculate_axes(update_plot=True)
    
    def _remove_spl_data_for_surface(self, surface_id: str) -> None:
        """
        Entfernt SPL-Daten (Texturen) für eine ungültige Surface aus dem Plot.
        
        Args:
            surface_id: ID der Surface, für die SPL-Daten entfernt werden sollen
        """
        try:
            # Zugriff auf den Plotter
            if not (hasattr(self.main_window, 'draw_plots') and 
                    hasattr(self.main_window.draw_plots, 'draw_spl_plotter')):
                return
            
            plotter = self.main_window.draw_plots.draw_spl_plotter
            if not plotter:
                return
            
            # Entferne Surface-Textur-Actor
            actor_name_tex = f"spl_surface_tex_{surface_id}"
            actor_name_tri = f"spl_surface_tri_{surface_id}"
            
            try:
                if hasattr(plotter, 'renderer') and hasattr(plotter.renderer, 'actors'):
                    # Entferne Textur-Actor
                    if actor_name_tex in plotter.renderer.actors:
                        plotter.remove_actor(actor_name_tex)
                    
                    # Entferne Triangulation-Actor (falls vorhanden)
                    if actor_name_tri in plotter.renderer.actors:
                        plotter.remove_actor(actor_name_tri)
            except Exception:
                pass
            
            # Entferne aus _surface_texture_actors
            try:
                if hasattr(plotter, '_surface_texture_actors'):
                    if surface_id in plotter._surface_texture_actors:
                        del plotter._surface_texture_actors[surface_id]
            except Exception:
                pass
            
            # Entferne aus internen Caches (falls vorhanden)
            try:
                if hasattr(plotter, '_surface_texture_cache') and surface_id in plotter._surface_texture_cache:
                    del plotter._surface_texture_cache[surface_id]
            except Exception:
                pass
            
            try:
                if hasattr(plotter, '_surface_signature_cache') and surface_id in plotter._surface_signature_cache:
                    del plotter._surface_signature_cache[surface_id]
            except Exception:
                pass
            
            # Entferne auch aus calculation_spl (falls vorhanden)
            try:
                if hasattr(self.container, 'calculation_spl') and isinstance(self.container.calculation_spl, dict):
                    # Entferne Surface-Grid und Surface-Results für diese Surface
                    if 'surface_grids' in self.container.calculation_spl:
                        if surface_id in self.container.calculation_spl['surface_grids']:
                            del self.container.calculation_spl['surface_grids'][surface_id]
                    
                    if 'surface_results' in self.container.calculation_spl:
                        if surface_id in self.container.calculation_spl['surface_results']:
                            del self.container.calculation_spl['surface_results'][surface_id]
            except Exception:
                pass
            
            print(f"[DEBUG] SPL-Daten für ungültige Surface '{surface_id}' entfernt")
        except Exception as e:
            print(f"[DEBUG] Fehler beim Entfernen der SPL-Daten für Surface '{surface_id}': {e}")
            import traceback
            traceback.print_exc()
    
    def _update_surface_validation_ui(self, surface_id):
        """Aktualisiert die UI basierend auf Validierungsergebnissen"""
        surface = self._get_surface(surface_id)
        if not surface:
            return
        
        # Validiere Surface
        validation_result = validate_and_optimize_surface(
            surface if isinstance(surface, SurfaceDefinition) else SurfaceDefinition.from_dict(surface_id, surface),
            round_to_cm=False,
            remove_redundant=False,
        )
        
        # Erstelle Set von ungültigen Feldern für schnellen Lookup
        invalid_fields_set = {(idx, coord) for idx, coord in (validation_result.invalid_fields or [])}
        
        # Aktualisiere alle Editoren im Points Tree
        if hasattr(self, 'points_tree'):
            for idx in range(self.points_tree.topLevelItemCount()):
                item = self.points_tree.topLevelItem(idx)
                if item:
                    point_index = item.data(0, Qt.UserRole)
                    if isinstance(point_index, int):
                        # Aktualisiere X, Y, Z Editoren
                        for col, coord_name in [(1, 'x'), (2, 'y'), (3, 'z')]:
                            editor = self.points_tree.itemWidget(item, col)
                            if editor:
                                # Wenn Wert None, zeige leeres Feld
                                points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                                if 0 <= point_index < len(points):
                                    point_value = points[point_index].get(coord_name)
                                    if point_value is None:
                                        editor.setText("")
                                    else:
                                        editor.setText(f"{float(point_value):.2f}")
    
    def _handle_add_point(self, surface_id):
        """Fügt einen neuen Punkt zum Surface hinzu"""
        surface = self._get_surface(surface_id)
        if not surface:
            return
        
        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
        new_point = {"x": 0.0, "y": 0.0, "z": 0.0}
        points.append(new_point)
        
        # Aktualisiere Anzeige
        self.create_surface_parameter_tab(surface_id)
        
        # Prüfe ob Surface versteckt ist - wenn ja, keine Calc/Plot Aktualisierung
        if isinstance(surface, SurfaceDefinition):
            hidden = getattr(surface, 'hidden', False)
        else:
            hidden = surface.get('hidden', False)
        
        if not hidden:
            # Aktualisiere Berechnungen nur wenn Surface nicht versteckt ist
            if hasattr(self.main_window, 'update_speaker_array_calculations'):
                self.main_window.update_speaker_array_calculations()
    
    def _handle_delete_point(self, surface_id):
        """Löscht den ausgewählten Punkt vom Surface"""
        surface = self._get_surface(surface_id)
        if not surface:
            return
        
        item = self.points_tree.currentItem()
        if not item:
            return
        
        point_index = item.data(0, Qt.UserRole)
        if not isinstance(point_index, int):
            return
        
        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
        if 0 <= point_index < len(points):
            del points[point_index]
            
            # Aktualisiere Anzeige
            self.create_surface_parameter_tab(surface_id)
            
            # Prüfe ob Surface versteckt ist - wenn ja, keine Calc/Plot Aktualisierung
            if isinstance(surface, SurfaceDefinition):
                hidden = getattr(surface, 'hidden', False)
            else:
                hidden = surface.get('hidden', False)
            
            if not hidden:
                # Aktualisiere Berechnungen nur wenn Surface nicht versteckt ist
                if hasattr(self.main_window, 'update_speaker_array_calculations'):
                    self.main_window.update_speaker_array_calculations()
    
    def _renumber_points(self):
        """Nummeriert die Punkte im TreeWidget neu"""
        if not hasattr(self, 'points_tree'):
            return
        
        for idx in range(self.points_tree.topLevelItemCount()):
            item = self.points_tree.topLevelItem(idx)
            if item:
                item.setText(0, f"P{idx + 1}")
                item.setData(0, Qt.UserRole, idx)
    
    def _show_points_context_menu(self, position, surface_id):
        """Zeigt das Kontextmenü für Punkte"""
        menu = QMenu(self.points_tree)
        add_action = menu.addAction("Add Point")
        delete_action = menu.addAction("Delete Point")
        
        index = self.points_tree.indexAt(position)
        if not index.isValid():
            delete_action.setEnabled(False)
        
        action = menu.exec_(self.points_tree.viewport().mapToGlobal(position))
        
        if action == add_action:
            self._handle_add_point(surface_id)
        elif action == delete_action:
            self._handle_delete_point(surface_id)
    
    # ---- Group Management --------------------------------------------
    
    def save_original_positions_for_group(self, group_item):
        """Speichert die ursprünglichen Positionen für alle Surfaces in einer Gruppe"""
        if not group_item:
            return
        
        # Finde die Gruppen-ID
        group_id = group_item.data(0, Qt.UserRole)
        
        # Hole oder erstelle Gruppen-Daten
        if group_id not in self.surface_groups:
            group = self._group_controller.get_group(group_id)
            if group:
                self.surface_groups[group_id] = {
                    'name': group.name,
                    'enabled': group.enabled,
                    'hidden': group.hidden,
                    'child_surface_ids': [],
                    'original_surface_positions': {}
                }
        
        # Hole oder erstelle ursprüngliche Positionen
        original_positions = self.surface_groups[group_id].get('original_surface_positions', {})
        
        # Speichere ursprüngliche Positionen für alle Child-Surfaces
        child_count = group_item.childCount()
        for i in range(child_count):
            child_item = group_item.child(i)
            child_type = child_item.data(0, Qt.UserRole + 1)
            
            if child_type == "surface":
                surface_id = child_item.data(0, Qt.UserRole)
                if isinstance(surface_id, dict):
                    surface_id = surface_id.get('id')
                
                if surface_id is not None and surface_id not in original_positions:
                    surface = self._get_surface(surface_id)
                    if surface:
                        # Berechne Zentrum der Surface-Punkte als Position
                        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                        if points:
                            x_coords = [p.get('x', 0.0) for p in points]
                            y_coords = [p.get('y', 0.0) for p in points]
                            z_coords = [p.get('z', 0.0) for p in points]
                            
                            center_x = sum(x_coords) / len(x_coords) if x_coords else 0.0
                            center_y = sum(y_coords) / len(y_coords) if y_coords else 0.0
                            center_z = sum(z_coords) / len(z_coords) if z_coords else 0.0
                            
                            original_positions[surface_id] = {
                                'x': center_x,
                                'y': center_y,
                                'z': center_z
                            }
        
        # Speichere aktualisierte ursprüngliche Positionen
        self.surface_groups[group_id]['original_surface_positions'] = original_positions
    
    def load_group_values(self, group_item):
        """Lädt die aktuellen Werte der Gruppe in die Eingabefelder"""
        if not group_item:
            return
        
        group_id = group_item.data(0, Qt.UserRole)
        
        if group_id not in self.surface_groups:
            # Initialisiere mit Standardwerten
            self.group_rel_x_edit.setText("0.00")
            self.group_rel_y_edit.setText("0.00")
            self.group_rel_z_edit.setText("0.00")
            return
        
        group_data = self.surface_groups[group_id]
        
        # Lade relative Positionen
        rel_pos = group_data.get('relative_position', {'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.group_rel_x_edit.setText(f"{rel_pos.get('x', 0.0):.2f}")
        self.group_rel_y_edit.setText(f"{rel_pos.get('y', 0.0):.2f}")
        self.group_rel_z_edit.setText(f"{rel_pos.get('z', 0.0):.2f}")
    
    def apply_group_changes_by_id(self, group_id):
        """Wendet die Gruppen-Änderungen auf alle Child-Surfaces an (verwendet group_id statt group_item)"""
        if not group_id:
            return
        
        # Hole die eingegebenen Werte
        try:
            rel_x = float(self.group_rel_x_edit.text())
            rel_y = float(self.group_rel_y_edit.text())
            rel_z = float(self.group_rel_z_edit.text())
        except ValueError:
            print("Fehler: Ungültige Eingabewerte")
            return
        
        # Stelle sicher, dass Gruppen-Daten existieren
        if group_id not in self.surface_groups:
            group = self._group_controller.get_group(group_id)
            if group:
                self.surface_groups[group_id] = {
                    'name': group.name,
                    'enabled': group.enabled,
                    'hidden': group.hidden,
                    'child_surface_ids': [],
                    'original_surface_positions': {}
                }
        
        # Speichere die relativen Werte in der Gruppe
        self.surface_groups[group_id]['relative_position'] = {
            'x': rel_x,
            'y': rel_y,
            'z': rel_z
        }
        
        # Hole ursprüngliche Surface-Positionen (sollten bereits beim Hinzufügen zur Gruppe gespeichert sein)
        original_positions = self.surface_groups[group_id].get('original_surface_positions', {})
        
        # Versuche, das Tree-Item zu finden, falls es noch existiert
        group_item = None
        if hasattr(self, 'surface_tree_widget') and self.surface_tree_widget:
            group_item = self._find_tree_item_by_id(group_id)
        
        # Sammle alle Child-Surface-IDs
        child_surface_ids = []
        
        # Versuche zuerst, die IDs aus dem Tree-Item zu holen (falls es noch existiert)
        if group_item:
            try:
                child_count = group_item.childCount()
                for i in range(child_count):
                    child_item = group_item.child(i)
                    child_type = child_item.data(0, Qt.UserRole + 1)
                    
                    if child_type == "surface":
                        surface_id = child_item.data(0, Qt.UserRole)
                        if isinstance(surface_id, dict):
                            surface_id = surface_id.get('id')
                        
                        if surface_id is not None:
                            child_surface_ids.append(surface_id)
            except RuntimeError:
                # Item wurde gelöscht, verwende Fallback
                group_item = None
        
        # Fallback: Hole Child-Surface-IDs aus der Gruppe oder aus gespeicherten Daten
        if not child_surface_ids:
            # Versuche aus gespeicherten Daten
            if 'child_surface_ids' in self.surface_groups[group_id]:
                child_surface_ids = self.surface_groups[group_id]['child_surface_ids']
            
            # Falls immer noch leer, hole aus der Gruppe selbst
            if not child_surface_ids:
                group = self._group_controller.get_group(group_id)
                if group:
                    child_surface_ids = getattr(group, 'surface_ids', [])
                    # Speichere für zukünftige Verwendung
                    self.surface_groups[group_id]['child_surface_ids'] = child_surface_ids
        
        # Speichere die Child-Surface-IDs für zukünftige Verwendung
        self.surface_groups[group_id]['child_surface_ids'] = child_surface_ids
        
        # Sammle ursprüngliche Positionen für alle Surfaces
        for surface_id in child_surface_ids:
            if surface_id not in original_positions:
                surface = self._get_surface(surface_id)
                if surface:
                    points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                    if points:
                        x_coords = [p.get('x', 0.0) for p in points]
                        y_coords = [p.get('y', 0.0) for p in points]
                        z_coords = [p.get('z', 0.0) for p in points]
                        
                        center_x = sum(x_coords) / len(x_coords) if x_coords else 0.0
                        center_y = sum(y_coords) / len(y_coords) if y_coords else 0.0
                        center_z = sum(z_coords) / len(z_coords) if z_coords else 0.0
                        
                        original_positions[surface_id] = {
                            'x': center_x,
                            'y': center_y,
                            'z': center_z
                        }
        
        # Speichere aktualisierte ursprüngliche Positionen
        if original_positions:
            self.surface_groups[group_id]['original_surface_positions'] = original_positions
        
        # Wende die Änderungen auf alle Child-Surfaces an
        # WICHTIG: Die relativen Werte werden zu den AKTUELLEN Surface-Punkt-Werten hinzugefügt
        for surface_id in child_surface_ids:
            surface = self._get_surface(surface_id)
            if surface:
                points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
                
                # Addiere relative Positionen zu allen Punkten
                for point in points:
                    point['x'] = point.get('x', 0.0) + rel_x
                    point['y'] = point.get('y', 0.0) + rel_y
                    point['z'] = point.get('z', 0.0) + rel_z
                
                # Aktualisiere Surface
                if isinstance(surface, SurfaceDefinition):
                    surface.points = points
                else:
                    surface['points'] = points
        
        # Aktualisiere Berechnungen
        if hasattr(self.main_window, 'update_speaker_array_calculations'):
            self.main_window.update_speaker_array_calculations()
        
        # Aktualisiere TreeWidget
        self.load_surfaces()
    
    # ---- Context Menu -----------------------------------------------
    
    def show_context_menu(self, position):
        """Zeigt das Kontextmenü für das TreeWidget"""
        item = self.surface_tree_widget.itemAt(position)
        selected_items = self.surface_tree_widget.selectedItems()
        
        menu = QMenu(self.surface_tree_widget)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        multiple_selected = len(selected_items) > 1
        
        if item or selected_items:
            # Bestimme Item-Typen
            has_groups = False
            has_surfaces = False
            
            if selected_items:
                for sel_item in selected_items:
                    item_type = sel_item.data(0, Qt.UserRole + 1)
                    if item_type == "group":
                        has_groups = True
                    else:
                        has_surfaces = True
            elif item:
                item_type = item.data(0, Qt.UserRole + 1)
                if item_type == "group":
                    has_groups = True
                else:
                    has_surfaces = True
            
            # Add-Optionen am Anfang
            add_surface_action = menu.addAction("Add Surface")
            add_group_action = menu.addAction("Add Group")
            menu.addSeparator()
            
            # Einheitliche Reihenfolge: Duplicate, Rename, ---, Delete
            duplicate_action = menu.addAction("Duplicate")
            rename_action = menu.addAction("Rename")
            menu.addSeparator()
            delete_action = menu.addAction("Delete")
            
            # Anpasse Text basierend auf Typ
            if has_groups and not has_surfaces:
                delete_action.setText("Delete Group")
            elif has_surfaces and not has_groups:
                delete_action.setText("Delete Surface")
            
            # Rename deaktivieren, wenn mehrere Items ausgewählt
            if multiple_selected:
                rename_action.setEnabled(False)
            else:
                # Bei Einzelauswahl: Setze aktuelles Item
                if item:
                    self.surface_tree_widget.setCurrentItem(item)
        else:
            # Kein Item ausgewählt: Zeige Add-Optionen
            add_surface_action = menu.addAction("Add Surface")
            add_group_action = menu.addAction("Add Group")
            menu.addSeparator()
            duplicate_action = menu.addAction("Duplicate")
            rename_action = menu.addAction("Rename")
            menu.addSeparator()
            delete_action = menu.addAction("Delete")
            
            delete_action.setEnabled(False)
            duplicate_action.setEnabled(False)
            rename_action.setEnabled(False)
        
        action = menu.exec_(self.surface_tree_widget.viewport().mapToGlobal(position))
        
        if action:
            if action.text() == "Add Surface":
                self.add_surface()
            elif action.text() == "Add Group":
                self.add_group(item)
            elif action == duplicate_action:
                if multiple_selected:
                    self._handle_multiple_duplicate(selected_items)
                else:
                    self.duplicate_item(item)
            elif action == rename_action:
                if not multiple_selected and item:
                    self.surface_tree_widget.editItem(item)
            elif action == delete_action:
                if multiple_selected:
                    self._handle_multiple_delete(selected_items)
                else:
                    self.delete_item(item)
    
    def add_group(self, reference_item=None):
        """Fügt eine neue Gruppe hinzu"""
        # Finde die nächste verfügbare Gruppennummer
        existing_group_names = set()
        group_store = self._group_controller.list_groups()
        for group in group_store.values():
            existing_group_names.add(group.name)
        
        # Finde die nächste verfügbare Zahl
        group_number = 1
        while f"Group {group_number}" in existing_group_names:
            group_number += 1
        
        # Verwende automatisch generierten Namen
        name = f"Group {group_number}"
        
        # Bestimme Parent-Gruppe
        parent_group_id = None
        if reference_item:
            item_type = reference_item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                parent_group_id = reference_item.data(0, Qt.UserRole)
            else:
                # Surface - finde Parent-Gruppe
                parent = reference_item.parent()
                if parent:
                    parent_group_id = parent.data(0, Qt.UserRole)
        
        # Erstelle Gruppe (ohne Root-Gruppe als Parent)
        group = self._group_controller.create_surface_group(name, parent_id=parent_group_id)
        
        # Setze XY-Status auf aktiv (per Default)
        if group:
            group.xy_enabled = True
        
        # Lade TreeWidget neu
        self.load_surfaces()
        
        # Wähle die neue Gruppe aus
        for i in range(self.surface_tree_widget.topLevelItemCount()):
            item = self.surface_tree_widget.topLevelItem(i)
            if item.data(0, Qt.UserRole) == group.group_id:
                self.surface_tree_widget.setCurrentItem(item)
                break
    
    def duplicate_item(self, item, update_calculations=True):
        """Dupliziert ein Surface oder eine Gruppe
        
        Args:
            item: Das zu duplizierende Item
            update_calculations: Wenn True, werden Berechnungen am Ende ausgeführt (Standard: True)
        """
        if not item:
            return
        
        item_type = item.data(0, Qt.UserRole + 1)
        
        if item_type == "surface":
            surface_id = item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            
            # Dupliziere Surface
            surface = self._get_surface(surface_id)
            if surface:
                import copy
                surface_store = getattr(self.settings, 'surface_definitions', {})
                
                # Generiere neue ID - prüfe alle existierenden IDs
                index = 1
                existing_ids = set(surface_store.keys())
                while f"surface_{index}" in existing_ids:
                    index += 1
                new_surface_id = f"surface_{index}"
                
                # Bestimme Ziel-Gruppe (gleiche Gruppe wie Original)
                target_group_id = None
                if isinstance(surface, SurfaceDefinition):
                    target_group_id = surface.group_id
                else:
                    target_group_id = surface.get('group_id')
                
                # Wenn Root-Gruppe, setze auf None (keine Gruppe)
                if target_group_id == self._group_controller.root_group_id:
                    target_group_id = None
                
                # Hole Original-Namen
                original_name = surface.name if isinstance(surface, SurfaceDefinition) else surface.get('name', 'Surface')
                
                # Kopiere Surface
                if isinstance(surface, SurfaceDefinition):
                    new_surface = SurfaceDefinition(
                        surface_id=new_surface_id,
                        name=f"copy of {original_name}",
                        enabled=surface.enabled,
                        hidden=surface.hidden,
                        locked=False,
                        points=copy.deepcopy(surface.points),
                        plane_model=copy.deepcopy(surface.plane_model) if surface.plane_model else None,
                        color=surface.color,
                        group_id=target_group_id
                    )
                    # Kopiere XY-Status oder setze auf True (per Default)
                    new_surface.xy_enabled = getattr(surface, 'xy_enabled', True)
                else:
                    new_surface = copy.deepcopy(surface)
                    new_surface['name'] = f"copy of {original_name}"
                    new_surface['group_id'] = target_group_id
                    # Kopiere XY-Status oder setze auf True (per Default)
                    new_surface['xy_enabled'] = surface.get('xy_enabled', True)
                
                # Verwende Settings-Methode für konsistente Behandlung
                if hasattr(self.settings, 'add_surface_definition'):
                    self.settings.add_surface_definition(new_surface_id, new_surface, make_active=False)
                else:
                    surface_store[new_surface_id] = new_surface
                    self.settings.surface_definitions = surface_store
                
                # Ordne Surface zur Gruppe zu
                self._group_controller.assign_surface_to_group(new_surface_id, target_group_id, create_missing=False)
                
                # Stelle sicher, dass die neue Surface direkt nach dem Original in der Gruppe erscheint
                if target_group_id:
                    group = self._group_controller.get_group(target_group_id)
                    if group and surface_id in group.surface_ids:
                        # Finde Position des Originals
                        original_index = group.surface_ids.index(surface_id)
                        # Entferne neue Surface von ihrer aktuellen Position (wenn vorhanden)
                        if new_surface_id in group.surface_ids:
                            group.surface_ids.remove(new_surface_id)
                        # Füge neue Surface direkt nach dem Original ein
                        # Da _populate_group_tree reversed() verwendet, erscheint sie direkt nach dem Original
                        group.surface_ids.insert(original_index + 1, new_surface_id)
                
                # Stelle sicher, dass Gruppen-Struktur aktuell ist
                self._group_controller.ensure_structure()
                
                # Lade TreeWidget neu (nur wenn update_calculations=True, sonst wird es am Ende in _handle_multiple_duplicate gemacht)
                if update_calculations:
                    # Blockiere Signale während load_surfaces, um Berechnungen zu vermeiden
                    self.surface_tree_widget.blockSignals(True)
                    try:
                        self.load_surfaces()
                        
                        # Wähle das neue Surface aus (Signale sind bereits blockiert)
                        self._select_surface_in_tree(new_surface_id, skip_overlays=False)
                    finally:
                        self.surface_tree_widget.blockSignals(False)
                else:
                    # Bei Mehrfachauswahl: Nur Surface hinzufügen, TreeWidget wird später neu geladen
                    pass
        
        elif item_type == "group":
            group_id = item.data(0, Qt.UserRole)
            if isinstance(group_id, dict):
                group_id = group_id.get('id')
            
            # Prüfe, ob es die Root-Gruppe ist
            if group_id == self._group_controller.root_group_id:
                return
            
            # Dupliziere Gruppe
            original_group = self._group_controller.get_group(group_id)
            if not original_group:
                return
            
            import copy
            
            # Generiere neue Gruppen-ID
            new_group_id = self._group_controller._generate_surface_group_id()
            
            # Hole Parent-Gruppe
            parent_group_id = original_group.parent_id
            if parent_group_id == self._group_controller.root_group_id:
                parent_group_id = None
            
            # Erstelle neue Gruppe mit kopiertem Namen
            new_group = self._group_controller.create_surface_group(
                f"copy of {original_group.name}",
                parent_id=parent_group_id,
                group_id=new_group_id
            )
            
            if not new_group:
                return
            
            # Kopiere Eigenschaften
            new_group.enabled = original_group.enabled
            new_group.hidden = original_group.hidden
            new_group.locked = False
            
            # Stelle sicher, dass die neue Gruppe direkt nach dem Original in der Parent-Gruppe erscheint
            if parent_group_id:
                parent_group = self._group_controller.get_group(parent_group_id)
                if parent_group and group_id in parent_group.child_groups:
                    # Finde Position des Originals
                    original_index = parent_group.child_groups.index(group_id)
                    # Entferne neue Gruppe von ihrer aktuellen Position (wenn vorhanden)
                    if new_group_id in parent_group.child_groups:
                        parent_group.child_groups.remove(new_group_id)
                    # Füge neue Gruppe direkt nach dem Original ein
                    # Da _populate_group_tree reversed() verwendet, erscheint sie direkt nach dem Original
                    parent_group.child_groups.insert(original_index + 1, new_group_id)
            
            # Dupliziere alle Surfaces aus der Original-Gruppe
            surface_store = getattr(self.settings, 'surface_definitions', {})
            for surface_id in original_group.surface_ids:
                surface = surface_store.get(surface_id)
                if surface:
                    # Generiere neue Surface-ID
                    index = 1
                    existing_ids = set(surface_store.keys())
                    while f"surface_{index}" in existing_ids:
                        index += 1
                    new_surface_id = f"surface_{index}"
                    
                    # Kopiere Surface
                    if isinstance(surface, SurfaceDefinition):
                        new_surface = SurfaceDefinition(
                            surface_id=new_surface_id,
                            name=f"copy of {surface.name}",
                            enabled=surface.enabled,
                            hidden=surface.hidden,
                            locked=False,
                            points=copy.deepcopy(surface.points),
                            plane_model=copy.deepcopy(surface.plane_model) if surface.plane_model else None,
                            color=surface.color,
                            group_id=new_group_id
                        )
                        new_surface.xy_enabled = getattr(surface, 'xy_enabled', True)
                    else:
                        new_surface = copy.deepcopy(surface)
                        new_surface['name'] = f"copy of {surface.name}"
                        new_surface['group_id'] = new_group_id
                        new_surface['xy_enabled'] = surface.get('xy_enabled', True)
                    
                    # Füge Surface hinzu
                    if hasattr(self.settings, 'add_surface_definition'):
                        self.settings.add_surface_definition(new_surface_id, new_surface, make_active=False)
                    else:
                        surface_store[new_surface_id] = new_surface
                        self.settings.surface_definitions = surface_store
                    
                    # Ordne Surface zur neuen Gruppe zu
                    self._group_controller.assign_surface_to_group(new_surface_id, new_group_id, create_missing=False)
            
            # Dupliziere rekursiv alle Child-Gruppen
            for child_group_id in original_group.child_groups:
                child_item = None
                # Finde das Child-Gruppen-Item im TreeWidget (für rekursive Duplikation)
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.data(0, Qt.UserRole + 1) == "group":
                        child_id = child.data(0, Qt.UserRole)
                        if isinstance(child_id, dict):
                            child_id = child_id.get('id')
                        if child_id == child_group_id:
                            child_item = child
                            break
                
                if child_item:
                    # Rekursiv duplizieren (ohne Berechnungen, da wir am Ende load_surfaces aufrufen)
                    self.duplicate_item(child_item, update_calculations=False)
                    # Die duplizierte Gruppe wird automatisch zur neuen Parent-Gruppe hinzugefügt
                    # durch create_surface_group, aber wir müssen die Parent-ID setzen
                    duplicated_child_group = self._group_controller.get_group(child_group_id)
                    if duplicated_child_group:
                        # Finde die duplizierte Gruppe (hat "copy of" im Namen)
                        group_store = self._group_controller.list_groups()
                        for gid, g in group_store.items():
                            if g.name == f"copy of {duplicated_child_group.name}" and g.parent_id != new_group_id:
                                # Setze Parent auf neue Gruppe
                                self._group_controller.move_surface_group(gid, new_group_id)
                                break
            
            # Stelle sicher, dass Gruppen-Struktur aktuell ist
            self._group_controller.ensure_structure()
            
            # Lade TreeWidget neu (nur wenn update_calculations=True, sonst wird es am Ende in _handle_multiple_duplicate gemacht)
            if update_calculations:
                # Blockiere Signale während load_surfaces, um Berechnungen zu vermeiden
                self.surface_tree_widget.blockSignals(True)
                try:
                    self.load_surfaces()
                finally:
                    self.surface_tree_widget.blockSignals(False)
            else:
                # Bei Mehrfachauswahl: Nur Gruppe hinzufügen, TreeWidget wird später neu geladen
                pass
    
    def _handle_multiple_duplicate(self, selected_items):
        """Dupliziert mehrere ausgewählte Items. Berechnungen werden erst am Ende ausgeführt."""
        if not selected_items:
            return
        
        # Blockiere Signale während der gesamten Mehrfach-Duplikation
        self.surface_tree_widget.blockSignals(True)
        
        try:
            # Stelle sicher, dass Gruppen-Struktur aktuell ist
            self._group_controller.ensure_structure()
            
            # Dupliziere alle Items (ohne Berechnungen während der Duplikation)
            for item in selected_items:
                self.duplicate_item(item, update_calculations=False)
            
            # Lade TreeWidget einmal am Ende neu (nach allen Duplikationen)
            # Signale sind bereits blockiert, daher keine Berechnungen
            self.load_surfaces()
            
            # Aktualisiere Plots und Overlays erst nach allen UI-Updates
            if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()
            elif hasattr(self.main_window, 'update_speaker_array_calculations'):
                self.main_window.update_speaker_array_calculations()
        finally:
            # Signale wieder aktivieren
            self.surface_tree_widget.blockSignals(False)
    
    def _handle_multiple_delete(self, selected_items):
        """Löscht mehrere ausgewählte Items"""
        if not selected_items:
            return
        
        # Sammle zuerst alle IDs, bevor wir etwas löschen (sonst werden Items ungültig)
        surface_ids_to_delete = []
        group_ids_to_delete = []
        default_surface_id = getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default')
        
        for item in selected_items:
            if item is None:
                continue
            
            try:
                item_type = item.data(0, Qt.UserRole + 1)
            except RuntimeError:
                # Item wurde bereits gelöscht, überspringe
                continue
            
            if item_type == "surface":
                surface_id = item.data(0, Qt.UserRole)
                if isinstance(surface_id, dict):
                    surface_id = surface_id.get('id')
                
                # Prüfe, ob es die Default-Surface ist
                if surface_id != default_surface_id:
                    surface_ids_to_delete.append(surface_id)
                else:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self.surface_dockWidget,
                        "Action not possible",
                        "The default surface cannot be deleted."
                    )
            elif item_type == "group":
                group_id = item.data(0, Qt.UserRole)
                if isinstance(group_id, dict):
                    group_id = group_id.get('id')
                
                # Prüfe, ob es die Root-Gruppe ist
                if group_id != self._group_controller.root_group_id:
                    group_ids_to_delete.append(group_id)
                else:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self.surface_dockWidget,
                        "Aktion nicht möglich",
                        "Die Root-Gruppe kann nicht gelöscht werden."
                    )
        
        # Lösche alle Surfaces auf einmal
        surface_store = getattr(self.settings, 'surface_definitions', {})
        for surface_id in surface_ids_to_delete:
            if surface_id in surface_store:
                # Entferne aus Gruppe
                self._group_controller.detach_surface(surface_id)
                del surface_store[surface_id]
        if surface_ids_to_delete:
            self.settings.surface_definitions = surface_store
        
        # Lösche alle Gruppen auf einmal
        for group_id in group_ids_to_delete:
            try:
                success = self._group_controller.remove_surface_group(group_id)
                if success and hasattr(self, 'surface_groups') and group_id in self.surface_groups:
                    del self.surface_groups[group_id]
            except Exception as e:
                # Fehler beim Löschen einer Gruppe ignorieren und weitermachen
                print(f"Fehler beim Löschen der Gruppe {group_id}: {e}")
        
        # Lade TreeWidget nur einmal am Ende neu
        if surface_ids_to_delete or group_ids_to_delete:
            self.load_surfaces()
            
            # 🎯 Trigger Calc/Plot Update: Surfaces/Gruppen wurden gelöscht
            if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()
    
    def delete_item(self, item):
        """Löscht ein Surface oder eine Gruppe"""
        if not item:
            return
        
        item_type = item.data(0, Qt.UserRole + 1)
        
        if item_type == "surface":
            surface_id = item.data(0, Qt.UserRole)
            if isinstance(surface_id, dict):
                surface_id = surface_id.get('id')
            
            # Prüfe, ob es die Default-Surface ist
            if surface_id == getattr(self.settings, 'DEFAULT_SURFACE_ID', 'surface_default'):
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.surface_dockWidget,
                    "Action not possible",
                    "The default surface cannot be deleted."
                )
                return
            
            # Lösche Surface
            surface_store = getattr(self.settings, 'surface_definitions', {})
            if surface_id in surface_store:
                # Entferne aus Gruppe
                self._group_controller.detach_surface(surface_id)
                del surface_store[surface_id]
                self.settings.surface_definitions = surface_store
                
                # Lade TreeWidget neu
                self.load_surfaces()
                
                # 🎯 Trigger Calc/Plot Update: Surface wurde gelöscht
                if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        
        elif item_type == "group":
            group_id = item.data(0, Qt.UserRole)
            
            # Prüfe, ob es die Root-Gruppe ist
            if group_id == self._group_controller.root_group_id:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.surface_dockWidget,
                    "Aktion nicht möglich",
                    "Die Root-Gruppe kann nicht gelöscht werden."
                )
                return
            
            # Lösche Gruppe
            try:
                success = self._group_controller.remove_surface_group(group_id)
                if success:
                    if group_id in self.surface_groups:
                        del self.surface_groups[group_id]
                    self.load_surfaces()
                    # 🎯 Trigger Calc/Plot Update: Gruppe wurde gelöscht
                    if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                        self.main_window.draw_plots.update_plots_for_surface_state()
            except ValueError:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.surface_dockWidget,
                    "Aktion nicht möglich",
                    "Die Gruppe ist nicht leer und kann daher nicht gelöscht werden."
                )


# ---- SurfaceGroupController (aus alter Implementierung übernommen) ----

class _SurfaceGroupController:
    """Controller für Surface-Gruppen-Verwaltung"""
    
    def __init__(self, settings):
        self.settings = settings
    
    @property
    def root_group_id(self) -> str:
        return getattr(self.settings, "ROOT_SURFACE_GROUP_ID", "surface_group_root")
    
    def ensure_structure(self):
        """Stellt sicher, dass die Gruppen-Struktur korrekt ist"""
        store = self._ensure_group_store()
        
        # Entferne Root-Gruppe, falls sie existiert (wird nicht mehr benötigt)
        if self.root_group_id in store:
            root_group = store.get(self.root_group_id)
            if root_group:
                # Entferne alle Surfaces aus Root-Gruppe
                for surface_id in list(root_group.surface_ids):
                    self.detach_surface(surface_id)
                # Entferne Root-Gruppe selbst
                del store[self.root_group_id]
        
        surface_store = getattr(self.settings, "surface_definitions", {})
        valid_surface_ids = set(surface_store.keys())
        claimed_surface_ids: set[str] = set()
        
        # Bereinige nur manuelle Gruppen (ohne Root-Gruppe)
        for group_id in list(store.keys()):
            if group_id == self.root_group_id:
                continue  # Überspringe Root-Gruppe
            
            group = self._ensure_group_object(group_id)
            if group is None:
                continue
            
            # Wenn Parent nicht existiert oder Root-Gruppe ist, entferne Parent
            if group.parent_id and (group.parent_id not in store or group.parent_id == self.root_group_id):
                group.parent_id = None
            
            # Aktualisiere Parent-Child-Beziehungen
            if group.parent_id:
                parent = self._ensure_group_object(group.parent_id)
                if parent and group_id not in parent.child_groups:
                    parent.child_groups.append(group_id)
            
            group.child_groups = self._deduplicate_sequence(
                gid
                for gid in group.child_groups
                if gid in store and gid != group_id and gid != self.root_group_id
            )
            
            # Bereinige Surface-IDs in Gruppen
            cleaned_surface_ids = []
            for surface_id in group.surface_ids:
                if surface_id not in valid_surface_ids:
                    continue
                surface = self._ensure_surface_object(surface_id)
                if surface and surface.group_id and surface.group_id != group_id:
                    continue
                if surface_id in claimed_surface_ids:
                    continue
                cleaned_surface_ids.append(surface_id)
                claimed_surface_ids.add(surface_id)
            group.surface_ids = cleaned_surface_ids
        
        # Entferne Root-Gruppe-Referenzen von Surfaces
        for surface_id in surface_store.keys():
            surface = self._ensure_surface_object(surface_id)
            if surface and surface.group_id == self.root_group_id:
                surface.group_id = None

    def ensure_surface_group_structure(self):
        """Stellt sicher, dass die Gruppen-Struktur korrekt ist (Alias für ensure_structure)"""
        self.ensure_structure()

    def reset_surface_storage(self):
        """Setzt alle Surface- und Gruppen-Definitionen zurück"""
        if hasattr(self.settings, "surface_definitions"):
            initializer = getattr(self.settings, "_initialize_surface_definitions", None)
            if callable(initializer):
                self.settings.surface_definitions = initializer()
            else:
                self.settings.surface_definitions = {}
        if hasattr(self.settings, "surface_groups"):
            initializer = getattr(self.settings, "_initialize_surface_groups", None)
            if callable(initializer):
                self.settings.surface_groups = initializer()
            else:
                self.settings.surface_groups = {}
    
    @staticmethod
    def _deduplicate_sequence(sequence):
        seen = set()
        result = []
        for item in sequence:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
    
    def list_groups(self) -> Dict[str, SurfaceGroup]:
        store = self._ensure_group_store()
        return {gid: self._ensure_group_object(gid) for gid in store.keys()}
    
    def get_group(self, group_id: Optional[str]) -> Optional[SurfaceGroup]:
        if not group_id:
            return None
        return self._ensure_group_object(group_id)
    
    def assign_surface_to_group(self, surface_id: str, target_group_id: Optional[str], *, create_missing: bool = False):
        surface = self._ensure_surface_object(surface_id)
        if surface is None:
            return
        
        # Wenn keine Gruppe angegeben oder Root-Gruppe, entferne aus Gruppe
        if not target_group_id or target_group_id == self.root_group_id:
            previous_group_id = surface.group_id or self._find_surface_group_id(surface_id)
            if previous_group_id:
                self._remove_surface_from_group(surface_id, previous_group_id)
            surface.group_id = None
            return
        
        group = self._ensure_group_object(target_group_id)
        if group is None:
            # Prüfe nochmal im Store, ob die Gruppe vielleicht als Dict existiert
            store = self._ensure_group_store()
            if target_group_id in store:
                # Gruppe existiert bereits, lade sie
                group = self._ensure_group_object(target_group_id)
        
        if group is None and create_missing:
            # Gruppe existiert nicht - erstelle sie mit group_id als temporären Namen
            # (der Name sollte später aktualisiert werden, wenn die Gruppe durch _ensure_group_for_label erstellt wurde)
            # Aber eigentlich sollte die Gruppe bereits existieren, wenn sie durch _ensure_group_for_label erstellt wurde
            # Wenn group_id wie "group_1" aussieht, versuche nicht, eine Gruppe zu erstellen
            # (die Gruppe sollte bereits existieren)
            if target_group_id.startswith("group_"):
                return
            # Ansonsten erstelle eine neue Gruppe mit group_id als Name (Fallback)
            group = self.create_surface_group(target_group_id, parent_id=None, group_id=target_group_id)
        
        if group is None:
            # Gruppe existiert nicht und soll nicht erstellt werden
            return
        
        previous_group_id = surface.group_id or self._find_surface_group_id(surface_id)
        if previous_group_id and previous_group_id != target_group_id:
            self._remove_surface_from_group(surface_id, previous_group_id)
        
        if group and surface_id not in group.surface_ids:
            # Füge am Anfang hinzu (neue Surfaces erscheinen oben)
            group.surface_ids.insert(0, surface_id)
        surface.group_id = target_group_id

    def find_surface_group_by_name(self, name: str) -> Optional[SurfaceGroup]:
        target = (name or "").strip().lower()
        if not target:
            return None
        for group_id in list(self._ensure_group_store().keys()):
            group = self._ensure_group_object(group_id)
            if not group:
                continue
            if (group.name or "").strip().lower() == target:
                return group
        return None

    def ensure_group_path(self, path: str) -> Optional[str]:
        segments = [segment.strip() for segment in (path or "").split("/") if segment.strip()]
        if not segments:
            return None

        parent_id = None
        last_group_id = None
        for segment in segments:
            group = self._find_group_by_name_in_parent(segment, parent_id)
            if group is None:
                group = self.create_surface_group(segment, parent_id=parent_id)
            if group is None:
                return last_group_id
            last_group_id = group.group_id
            parent_id = group.group_id
        return last_group_id
    
    def create_surface_group(self, name: str, parent_id: Optional[str] = None, *, group_id: Optional[str] = None, locked: bool = False) -> SurfaceGroup:
        # Wenn parent_id nicht angegeben oder Root-Gruppe, setze auf None (Top-Level-Gruppe)
        if parent_id == self.root_group_id:
            parent_id = None
        
        parent = self._ensure_group_object(parent_id) if parent_id else None
        
        if group_id is None:
            group_id = self._generate_surface_group_id()
        
        store = self._ensure_group_store()
        if group_id in store:
            existing_group = self._ensure_group_object(group_id)
            # Wenn die Gruppe bereits existiert, aktualisiere den Namen nur, wenn er noch nicht gesetzt ist
            if existing_group and existing_group.name == group_id and name != group_id:
                existing_group.name = name
            return existing_group
        
        group = SurfaceGroup(
            group_id=group_id,
            name=name or group_id,
            enabled=parent.enabled if parent else False,
            hidden=parent.hidden if parent else False,
            parent_id=parent_id,
            locked=locked,
        )
        store[group_id] = group
        
        if parent and group_id not in parent.child_groups:
            # Füge am Anfang hinzu (neue Gruppen erscheinen oben)
            parent.child_groups.insert(0, group_id)
        
        return group
    
    def move_surface_group(self, group_id: str, target_parent_id: Optional[str]) -> bool:
        """Verschiebt eine Gruppe zu einer anderen Parent-Gruppe (oder None für Top-Level)"""
        group = self._ensure_group_object(group_id)
        if not group or group.locked:
            return False
        
        # Wenn target_parent_id die Root-Gruppe ist, setze auf None
        if target_parent_id == self.root_group_id:
            target_parent_id = None
        
        # Prüfe, ob Ziel-Parent existiert (wenn angegeben)
        if target_parent_id:
            target_parent = self._ensure_group_object(target_parent_id)
            if not target_parent:
                return False
        
        # Entferne Gruppe von altem Parent
        old_parent = self._ensure_group_object(group.parent_id) if group.parent_id else None
        if old_parent and group_id in old_parent.child_groups:
            old_parent.child_groups.remove(group_id)
        
        # Füge Gruppe zu neuem Parent hinzu
        group.parent_id = target_parent_id
        if target_parent_id:
            new_parent = self._ensure_group_object(target_parent_id)
            if new_parent and group_id not in new_parent.child_groups:
                new_parent.child_groups.insert(0, group_id)
        
        return True
    
    def remove_surface_group(self, group_id: str) -> bool:
        group = self._ensure_group_object(group_id)
        if not group or group.locked:
            return False
        if group.child_groups or group.surface_ids:
            raise ValueError("Group must be empty before removal.")
        
        parent = self._ensure_group_object(group.parent_id) if group.parent_id else None
        if parent and group_id in parent.child_groups:
            parent.child_groups.remove(group_id)
        
        store = self._ensure_group_store()
        if group_id in store:
            del store[group_id]
        return True
    
    def rename_surface_group(self, group_id: str, new_name: str):
        group = self._ensure_group_object(group_id)
        if not group:
            return
        name = (new_name or "").strip()
        if name:
            group.name = name
    
    def set_surface_group_enabled(self, group_id: str, enabled: bool):
        group = self._ensure_group_object(group_id)
        if not group:
            return
        group.enabled = bool(enabled)
        for child_group_id in group.child_groups:
            self.set_surface_group_enabled(child_group_id, enabled)
        for surface_id in group.surface_ids:
            self._set_surface_enabled(surface_id, enabled)
    
    def set_surface_group_hidden(self, group_id: str, hidden: bool):
        group = self._ensure_group_object(group_id)
        if not group:
            return
        group.hidden = bool(hidden)
        for child_group_id in group.child_groups:
            self.set_surface_group_hidden(child_group_id, hidden)
        for surface_id in group.surface_ids:
            surface = self._ensure_surface_object(surface_id)
            if surface:
                surface.hidden = bool(hidden)
    
    def detach_surface(self, surface_id: str):
        group_id = self._find_surface_group_id(surface_id)
        if group_id:
            self._remove_surface_from_group(surface_id, group_id)
    
    def _ensure_group_store(self) -> Dict[str, SurfaceGroup]:
        store = getattr(self.settings, "surface_groups", None)
        if not isinstance(store, dict):
            # Store ist None oder kein Dict - erstelle neuen Store
            store = self._create_default_group_store()
            self.settings.surface_groups = store
        elif not store:
            # Store ist ein leeres Dict - behalte es, aber setze es explizit
            self.settings.surface_groups = store
        else:
            # Store existiert und hat Inhalt - setze ihn explizit (falls nötig)
            self.settings.surface_groups = store
        return store
    
    def _create_default_group_store(self) -> Dict[str, SurfaceGroup]:
        # Keine Root-Gruppe mehr - nur manuelle Gruppen
        return {}
    
    def _ensure_group_object(self, group_id: Optional[str]) -> Optional[SurfaceGroup]:
        if not group_id:
            return None
        store = self._ensure_group_store()
        group = store.get(group_id)
        if group is None:
            return None
        if isinstance(group, SurfaceGroup):
            return group
        # Gruppe ist ein Dict - konvertiere zu SurfaceGroup
        obj = SurfaceGroup.from_dict(group_id, group)
        store[group_id] = obj
        return obj
    
    def _ensure_surface_object(self, surface_id: str) -> Optional[SurfaceDefinition]:
        ensure_fn = getattr(self.settings, "_ensure_surface_object", None)
        if callable(ensure_fn):
            return ensure_fn(surface_id)
        surface = getattr(self.settings, "surface_definitions", {}).get(surface_id)
        if isinstance(surface, SurfaceDefinition):
            return surface
        if isinstance(surface, dict):
            obj = SurfaceDefinition.from_dict(surface_id, surface)
            self.settings.surface_definitions[surface_id] = obj
            return obj
        return None
    
    def _remove_surface_from_group(self, surface_id: str, group_id: str):
        group = self._ensure_group_object(group_id)
        if not group:
            return
        if surface_id in group.surface_ids:
            group.surface_ids.remove(surface_id)
    
    def _find_surface_group_id(self, surface_id: str) -> Optional[str]:
        for group_id in list(self._ensure_group_store().keys()):
            group = self._ensure_group_object(group_id)
            if group and surface_id in group.surface_ids:
                return group.group_id
        return None
    
    def _generate_surface_group_id(self) -> str:
        store = self._ensure_group_store()
        index = 1
        while True:
            candidate = f"group_{index}"
            if candidate not in store:
                return candidate
            index += 1

    def _find_group_by_name_in_parent(self, name: str, parent_id: Optional[str]) -> Optional[SurfaceGroup]:
        target = (name or "").strip().lower()
        if not target:
            return None
        for group_id in list(self._ensure_group_store().keys()):
            group = self._ensure_group_object(group_id)
            if not group:
                continue
            current_parent = group.parent_id or None
            if current_parent == parent_id and (group.name or "").strip().lower() == target:
                return group
        return None
    
    def _set_surface_enabled(self, surface_id: str, enabled: bool):
        set_enabled = getattr(self.settings, "set_surface_enabled", None)
        if callable(set_enabled):
            try:
                set_enabled(surface_id, enabled)
                return
            except KeyError:
                pass
        surface = self._ensure_surface_object(surface_id)
        if surface:
            surface.enabled = bool(enabled)
