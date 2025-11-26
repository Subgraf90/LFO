from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QSizePolicy,
    QAbstractItemView,
    QSplitter,
    QToolButton,
    QLineEdit,
    QStyle,
    QInputDialog,
    QHeaderView,
)
from PyQt5.QtGui import QFont, QDoubleValidator

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    SurfaceGroup,
    build_planar_model,
    derive_surface_plane,
    evaluate_surface_plane,
)
from Module_LFO.Modules_Data.SurfaceDataImporter import SurfaceDataImporter


SurfacePoint = Dict[str, Any]
SurfaceRecord = SurfaceDefinition | Dict[str, Any]


class PointsTreeWidget(QTreeWidget):
    """
    TreeWidget, das Drag & Drop zulässt und nach einer Neuordnung einen Callback ausführt.
    """

    def __init__(self, parent=None, reorder_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.reorder_callback = reorder_callback

    def dropEvent(self, event):
        super().dropEvent(event)
        if self.reorder_callback:
            self.reorder_callback()


class SurfaceTreeWidget(QTreeWidget):
    """
    TreeWidget für Surface-/Gruppenstruktur mit erweitertem Drag & Drop.
    """

    def __init__(self, parent=None, drop_callback: Optional[Callable] = None):
        super().__init__(parent)
        self._drop_callback = drop_callback
        self._pending_drag_items: List[QTreeWidgetItem] = []
        self._selection_snapshot: List[QTreeWidgetItem] = []

    def startDrag(self, supported_actions):
        current_selection = list(self.selectedItems())
        if len(current_selection) <= 1 and self._selection_snapshot:
            self._pending_drag_items = list(self._selection_snapshot)
        else:
            self._pending_drag_items = current_selection
        super().startDrag(supported_actions)
        # Pending Items werden erst nach DropEvent zurückgesetzt

    def mousePressEvent(self, event):
        self._selection_snapshot = list(self.selectedItems())
        super().mousePressEvent(event)

    def dropEvent(self, event):
        handled = False
        if self._drop_callback:
            drag_items = self._pending_drag_items or self.selectedItems()
            target = self.itemAt(event.pos())
            debug_drag = []
            for item in drag_items:
                data = item.data(0, Qt.UserRole)
                if isinstance(data, dict):
                    debug_drag.append(f"{data.get('type')}:{data.get('id')}")
                else:
                    debug_drag.append(str(data))
            target_data = None
            if target:
                t_data = target.data(0, Qt.UserRole)
                if isinstance(t_data, dict):
                    target_data = f"{t_data.get('type')}:{t_data.get('id')}"
                else:
                    target_data = str(t_data)
            print(
                f"[SurfaceTreeWidget] dropEvent count={len(drag_items)}, items={debug_drag}, "
                f"target={target_data}, indicator={self.dropIndicatorPosition()}"
            )
            handled = self._drop_callback(
                drag_items,
                target,
                self.dropIndicatorPosition(),
            )
        if handled:
            event.accept()
            event.setDropAction(Qt.MoveAction)
        else:
            event.ignore()
        self._pending_drag_items = []
        self._selection_snapshot = []


class PointValueEdit(QLineEdit):
    """
    LineEdit für Punktwerte, setzt beim Fokuserhalt die Selektion im TreeWidget.
    """

    def __init__(self, tree_widget: QTreeWidget, item: QTreeWidgetItem, column: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tree_widget = tree_widget
        self._item = item
        self._column = column

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._tree_widget and self._item:
            self._tree_widget.setCurrentItem(self._item)
            self._tree_widget.scrollToItem(self._item, QTreeWidget.PositionAtCenter)


class SurfaceDockWidget(QDockWidget):
    """
    DockWidget zur Verwaltung von Flächen und deren Punkten.

    Darstellung:
        - Linkes TreeWidget: Liste der Flächen mit Enable-/Hide-Checkboxen
        - Rechter Bereich: Button zum Hinzufügen von Punkten und TreeWidget mit X/Y/Z-Eingaben
    """

    DEFAULT_SURFACE_ID = "surface_default"
    DEFAULT_SURFACE_NAME = "Default Surface"

    ITEM_TYPE_SURFACE = "surface"
    ITEM_TYPE_GROUP = "group"

    def __init__(self, main_window, settings, container, surface_manager):
        super().__init__("Surface", main_window)
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.surface_manager = surface_manager

        self.current_surface_id: Optional[str] = None
        self._loading_surfaces = False
        self._loading_points = False
        self._surface_items: Dict[str, QTreeWidgetItem] = {}
        self._group_items: Dict[str, QTreeWidgetItem] = {}

        self.setObjectName("SurfaceDockWidget")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )

        self._points_panel_visible = False
        self._last_splitter_sizes = [400, 180]
        self._init_ui()

    # ---- public API -------------------------------------------------

    def initialize(self) -> None:
        """
        Initialisiert die Anzeige basierend auf dem aktuell ausgewählten Array.
        """
        array_id = self.main_window.get_selected_speaker_array_id() if hasattr(self.main_window, "get_selected_speaker_array_id") else None
        if array_id is None:
            array_ids = list(self.settings.get_all_speaker_array_ids())
            if array_ids:
                array_id = array_ids[0]
        self.load_surfaces()

    def load_surfaces(self) -> None:
        """
        Lädt alle Flächen aus den Settings in das TreeWidget.
        """
        import time
        t_start = time.perf_counter()

        self._loading_surfaces = True
        self.surface_tree.blockSignals(True)
        self.surface_tree.clear()
        self._surface_items.clear()
        self._group_items.clear()

        surface_store = self._get_surface_store()
        self._ensure_default_surface(surface_store)
        if self.surface_manager:
            self.surface_manager.ensure_surface_group_structure()
        group_store = self.surface_manager.list_surface_groups() if self.surface_manager else None
        root_group_id = self.surface_manager.get_root_group_id() if self.surface_manager else None

        if group_store and root_group_id and group_store.get(root_group_id):
            root_group = self._get_surface_group(root_group_id)
            if root_group:
                self._populate_group_contents(None, root_group, surface_store)
        else:
            for surface_id, surface_data in surface_store.items():
                item = self._create_surface_item(surface_id, surface_data)
                self.surface_tree.addTopLevelItem(item)

        # Einmal alle Gruppen/Surfaces expandieren, ohne übertrieben viele Updates
        self.surface_tree.expandAll()

        self.surface_tree.blockSignals(False)
        self._loading_surfaces = False

        # Einmalige Layout-Aktualisierung reicht
        self.surface_tree.doItemsLayout()

        target_surface = self.current_surface_id or self.DEFAULT_SURFACE_ID
        self._select_surface(target_surface)

        if self.surface_tree.currentItem() is None:
            self._loading_points = True
            self.points_tree.clear()
            self._loading_points = False

        self._activate_surface(self.current_surface_id)

        # Grobes Timing für das UI-Nachladen der Surfaces
        t_end = time.perf_counter()
        print(
            f"Surface-UI: load_surfaces() für {len(surface_store)} Surfaces in {t_end - t_start:.3f}s"
        )

    def _on_surface_load_clicked(self) -> None:
        """
        Startet den Surface-Importer und aktualisiert bei Erfolg die UI.
        """
        importer = SurfaceDataImporter(
            self,
            self.settings,
            self.container,
            group_manager=self.surface_manager,
        )
        result = importer.execute()
        if result:
            self.load_surfaces()

    # ---- UI setup ---------------------------------------------------

    def _init_ui(self) -> None:
        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Toggle-Button für Punktebereich
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(4)

        self.toggle_points_button = QToolButton()
        self.toggle_points_button.setObjectName("toolButtonTogglePoints")
        self.toggle_points_button.setCheckable(True)
        self.toggle_points_button.setChecked(False)
        self.toggle_points_button.setArrowType(Qt.DownArrow)
        self.toggle_points_button.setToolTip("Punktebereich ein-/ausklappen")
        self.toggle_points_button.toggled.connect(self._on_points_toggle)
        self.toggle_points_button.setFixedSize(14, 14)
        self.toggle_points_button.setStyleSheet("QToolButton { padding: 0px; }")

        toggle_layout.addWidget(self.toggle_points_button)
        
        self.reload_surfaces_button = QToolButton()
        self.reload_surfaces_button.setObjectName("toolButtonReloadSurfaces")
        self.reload_surfaces_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.reload_surfaces_button.setToolTip("Flächen erneut laden")
        self.reload_surfaces_button.clicked.connect(self._on_surface_load_clicked)
        toggle_layout.addWidget(self.reload_surfaces_button)

        toggle_layout.addStretch(1)
        main_layout.addLayout(toggle_layout)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(6)

        # Gemeinsame Schriftgröße
        tree_font = QFont()
        tree_font.setPointSize(11)

        # TreeWidget für Flächen
        self.surface_tree = SurfaceTreeWidget(drop_callback=self._handle_surface_tree_drop)
        self.surface_tree.setHeaderLabels(["Surface", "Enable", "Hide"])
        self.surface_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.surface_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.surface_tree.setRootIsDecorated(True)
        self.surface_tree.setAlternatingRowColors(False)
        self.surface_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.surface_tree.setColumnWidth(0, 115)
        self.surface_tree.setColumnWidth(1, 60)
        self.surface_tree.setColumnWidth(2, 55)
        self.surface_tree.setFont(tree_font)
        self.surface_tree.setDragEnabled(True)
        self.surface_tree.setAcceptDrops(True)
        self.surface_tree.setDefaultDropAction(Qt.MoveAction)
        self.surface_tree.setDropIndicatorShown(True)
        self.surface_tree.setDragDropMode(QAbstractItemView.DragDrop)
        header = self.surface_tree.header()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.surface_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.surface_tree.customContextMenuRequested.connect(self._show_surface_context_menu)
        self.surface_tree.setStyleSheet("QTreeWidget { font-size: 11pt; } QTreeWidget::item { height: 22px; }")
        self.splitter.addWidget(self.surface_tree)

        # Unterer Bereich: TreeWidget für Punkte
        self.points_tree = PointsTreeWidget(reorder_callback=self._sync_points_order)
        self.points_tree.setHeaderLabels(["Point", "X (m)", "Y (m)", "Z (m)"])
        self.points_tree.setRootIsDecorated(False)
        self.points_tree.setAlternatingRowColors(True)
        self.points_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.points_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points_tree.setUniformRowHeights(True)
        self.points_tree.setColumnWidth(0, 55)
        self.points_tree.setColumnWidth(1, 65)
        self.points_tree.setColumnWidth(2, 65)
        self.points_tree.setColumnWidth(3, 65)
        self.points_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.points_tree.setFont(tree_font)
        self.points_tree.setDragEnabled(True)
        self.points_tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.points_tree.setDefaultDropAction(Qt.MoveAction)
        self.points_tree.setDropIndicatorShown(True)
        self.points_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.points_tree.customContextMenuRequested.connect(self._show_points_context_menu)
        self.points_tree.setStyleSheet(
            "QTreeWidget { font-size: 11pt; }"
            "QTreeWidget::item { height: 24px; }"
            "QTreeWidget QLineEdit { font-size: 11pt; padding: 1px; }"
        )
        self.splitter.addWidget(self.points_tree)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        main_layout.addWidget(self.splitter, 1)

        self.setWidget(content)
        self.resize(520, 600)

        # Signalverbindungen
        self.surface_tree.itemSelectionChanged.connect(self._handle_surface_selection_changed)
        self.surface_tree.itemChanged.connect(self._handle_surface_item_changed)
        self.points_tree.itemChanged.connect(self._handle_point_item_changed)

        # Panel initial einklappen
        self._set_points_panel_visible(False, force=True)

    # ---- surface handling -------------------------------------------

    def _create_surface_item(self, surface_id: str, data: SurfaceRecord) -> QTreeWidgetItem:
        # Prüfe, ob bereits ein Item für dieses Surface existiert
        if surface_id in self._surface_items:
            existing_item = self._surface_items[surface_id]
            # Wenn das Item bereits ein Parent hat, entferne es zuerst
            if existing_item.parent() is not None:
                parent = existing_item.parent()
                parent.removeChild(existing_item)
            # Aktualisiere die Daten des bestehenden Items
            name = self._surface_get(data, "name", "Surface")
            existing_item.setText(0, str(name))
            enabled = bool(self._surface_get(data, "enabled", False))
            hidden = bool(self._surface_get(data, "hidden", False))
            existing_item.setCheckState(1, Qt.Checked if enabled else Qt.Unchecked)
            existing_item.setCheckState(2, Qt.Checked if hidden else Qt.Unchecked)
            # Stelle sicher, dass das Item sichtbar ist
            existing_item.setHidden(False)
            return existing_item
        
        # Erstelle ein neues Item
        item = QTreeWidgetItem()
        name = self._surface_get(data, "name", "Surface")
        item.setText(0, str(name))
        item.setFlags(
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
        )
        enabled = bool(self._surface_get(data, "enabled", False))
        hidden = bool(self._surface_get(data, "hidden", False))
        item.setCheckState(1, Qt.Checked if enabled else Qt.Unchecked)
        item.setCheckState(2, Qt.Checked if hidden else Qt.Unchecked)
        item.setData(0, Qt.UserRole, {"type": self.ITEM_TYPE_SURFACE, "id": surface_id})
        # Stelle sicher, dass das Item sichtbar ist
        item.setHidden(False)

        if surface_id == self.DEFAULT_SURFACE_ID:
            # Default-Fläche: nicht löschbar, aber umbenennbar
            item.setFlags(item.flags() & ~Qt.ItemIsDragEnabled)

        self._surface_items[surface_id] = item
        return item

    def _create_group_item(self, group: SurfaceGroup) -> QTreeWidgetItem:
        # Prüfe, ob bereits ein Item für diese Gruppe existiert
        if group.group_id in self._group_items:
            existing_item = self._group_items[group.group_id]
            # Wenn das Item bereits ein Parent hat, entferne es zuerst
            if existing_item.parent() is not None:
                parent = existing_item.parent()
                parent.removeChild(existing_item)
            # Aktualisiere die Daten des bestehenden Items
            existing_item.setText(0, group.name)
            existing_item.setCheckState(1, Qt.Checked if group.enabled else Qt.Unchecked)
            existing_item.setCheckState(2, Qt.Checked if group.hidden else Qt.Unchecked)
            # Stelle sicher, dass das Item sichtbar ist
            existing_item.setHidden(False)
            return existing_item
        
        # Erstelle ein neues Item
        item = QTreeWidgetItem()
        item.setText(0, group.name)
        item.setFlags(
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )
        item.setCheckState(1, Qt.Checked if group.enabled else Qt.Unchecked)
        item.setCheckState(2, Qt.Checked if group.hidden else Qt.Unchecked)
        item.setData(0, Qt.UserRole, {"type": self.ITEM_TYPE_GROUP, "id": group.group_id})
        # Markiere Gruppen-Items immer als expandierbar, auch wenn sie leer sind
        item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        # Stelle sicher, dass das Item sichtbar ist
        item.setHidden(False)
        self._group_items[group.group_id] = item
        return item

    def _populate_group_contents(
        self,
        parent_item: Optional[QTreeWidgetItem],
        group: SurfaceGroup,
        surface_store: Dict[str, SurfaceRecord],
    ) -> None:
        # Rekursiver Aufbau der Baumstruktur ohne übermäßiges Debug-Logging

        for child_group_id in group.child_groups:
            child_group = self._get_surface_group(child_group_id)
            if child_group is None:
                continue
            child_item = self._create_group_item(child_group)
            if parent_item is None:
                self.surface_tree.addTopLevelItem(child_item)
            else:
                parent_item.addChild(child_item)
            self._populate_group_contents(child_item, child_group, surface_store)

        for surface_id in group.surface_ids:
            surface_data = surface_store.get(surface_id)
            if surface_data is None:
                continue
            surface_item = self._create_surface_item(surface_id, surface_data)
            if parent_item is None:
                self.surface_tree.addTopLevelItem(surface_item)
            else:
                parent_item.addChild(surface_item)

        # Gruppe standardmäßig expandieren, falls vorhanden
        if parent_item is not None:
            parent_item.setExpanded(True)

    def _ensure_all_items_visible(self):
        """
        Stellt sicher, dass alle Items im TreeWidget sichtbar sind.
        Geht rekursiv durch alle Items und erweitert Gruppen, macht Items sichtbar.
        """
        def _process_item(item: QTreeWidgetItem):
            if item is None:
                return
            # Stelle sicher, dass das Item sichtbar ist
            item.setHidden(False)
            # Wenn es eine Gruppe ist, erweitere sie
            item_type, _ = self._get_item_identity(item)
            if item_type == self.ITEM_TYPE_GROUP:
                item.setExpanded(True)
            # Verarbeite alle Kinder rekursiv
            for idx in range(item.childCount()):
                child = item.child(idx)
                if child:
                    _process_item(child)
        
        # Verarbeite alle Top-Level-Items
        for idx in range(self.surface_tree.topLevelItemCount()):
            top_item = self.surface_tree.topLevelItem(idx)
            if top_item:
                _process_item(top_item)

    def _get_surface_group(self, group_id: Optional[str]) -> Optional[SurfaceGroup]:
        if not group_id or not self.surface_manager:
            return None
        return self.surface_manager.get_surface_group(group_id)

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
        group = self._get_surface_group(group_id)
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

    def _get_item_identity(self, item: Optional[QTreeWidgetItem]) -> tuple[Optional[str], Optional[str]]:
        if item is None:
            return None, None
        data = item.data(0, Qt.UserRole)
        if isinstance(data, dict):
            return data.get("type"), data.get("id")
        if isinstance(data, str):
            return self.ITEM_TYPE_SURFACE, data
        return None, None

    def _handle_group_item_changed(self, group_id: Optional[str], item: QTreeWidgetItem, column: int) -> None:
        if not group_id:
            return
        group = self._get_surface_group(group_id)
        if group is None:
            return

        if column == 0:
            new_name = item.text(0).strip()
            if new_name:
                if self.surface_manager:
                    self.surface_manager.rename_surface_group(group_id, new_name)
            else:
                self._loading_surfaces = True
                item.setText(0, group.name)
                self._loading_surfaces = False
        elif column == 1:
            enabled = item.checkState(1) == Qt.Checked
            if self.surface_manager:
                self.surface_manager.set_surface_group_enabled(group_id, enabled)
            self._apply_group_check_state(item, column, enabled)
            
            # 🎯 Trigger Calc/Plot Update: Enable-Status der Gruppe ändert sich
            # (Enable-Status-Änderung einer Gruppe beeinflusst alle Surfaces in der Gruppe)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        elif column == 2:
            hidden = item.checkState(2) == Qt.Checked
            if self.surface_manager:
                self.surface_manager.set_surface_group_hidden(group_id, hidden)
            self._apply_group_check_state(item, column, hidden)
            
            # 🎯 Trigger Calc/Plot Update: Hide-Status der Gruppe ändert sich
            # (Hide-Status-Änderung einer Gruppe beeinflusst alle Surfaces in der Gruppe)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()

    def _apply_group_check_state(self, item: QTreeWidgetItem, column: int, state: bool) -> None:
        previous_loading = self._loading_surfaces
        self._loading_surfaces = True
        try:
            for idx in range(item.childCount()):
                child = item.child(idx)
                child.setCheckState(column, Qt.Checked if state else Qt.Unchecked)
                child_type, _ = self._get_item_identity(child)
                if child_type == self.ITEM_TYPE_GROUP:
                    self._apply_group_check_state(child, column, state)
        finally:
            self._loading_surfaces = previous_loading

    def _determine_target_group_id(self, reference_item: Optional[QTreeWidgetItem] = None) -> Optional[str]:
        if reference_item is None:
            reference_item = self.surface_tree.currentItem()
        item_type, item_id = self._get_item_identity(reference_item)
        if item_type == self.ITEM_TYPE_GROUP:
            return item_id
        if item_type == self.ITEM_TYPE_SURFACE and item_id:
            surface = self._get_surface_store().get(item_id)
            if surface:
                group_id = self._surface_get(surface, "group_id")
                if group_id:
                    return group_id
        if self.surface_manager:
            return self.surface_manager.get_root_group_id()
        return None

    def _handle_add_surface(self) -> None:
        surface_store = self._get_surface_store()
        surface_id = self._generate_surface_id(surface_store)
        surface_data: SurfaceRecord = {
            "name": self._generate_surface_name(surface_store),
            "enabled": False,
            "hidden": False,
            "points": self._create_default_points(),
        }

        self.settings.add_surface_definition(surface_id, surface_data, make_active=True)
        parent_group_id = self._determine_target_group_id(self.surface_tree.currentItem())
        if self.surface_manager:
            self.surface_manager.assign_surface_to_group(surface_id, parent_group_id, create_missing=True)
        self.load_surfaces()
        self._select_surface(surface_id)
        self._set_points_panel_visible(True)
        self._load_points_for_surface(surface_id)
        # 🎯 Trigger Calc/Plot Update: Surface wurde hinzugefügt
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_delete_surface(self, item: Optional[QTreeWidgetItem] = None) -> None:
        if item is None:
            item = self.surface_tree.currentItem()
        if item is None:
            return

        item_type, surface_id = self._get_item_identity(item)
        if item_type != self.ITEM_TYPE_SURFACE or not surface_id:
            return
        if surface_id == self.DEFAULT_SURFACE_ID:
            QtWidgets.QMessageBox.information(
                self,
                "Aktion nicht möglich",
                "Die Standardfläche kann nicht gelöscht werden.",
            )
            return

        surface_store = self._get_surface_store()
        if surface_id in surface_store:
            if self.surface_manager:
                self.surface_manager.detach_surface_from_group(surface_id)
            self.settings.remove_surface_definition(surface_id)
        if self.current_surface_id == surface_id:
            self.current_surface_id = None
        self.load_surfaces()
        # 🎯 Trigger Calc/Plot Update: Surface wurde gelöscht
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_duplicate_item(self, item: Optional[QTreeWidgetItem]) -> None:
        if item is None:
            item = self.surface_tree.currentItem()
        if item is None:
            return

        item_type, item_id = self._get_item_identity(item)
        if item_type == self.ITEM_TYPE_SURFACE and item_id:
            self._duplicate_surface_entry(item_id, item)
        elif item_type == self.ITEM_TYPE_GROUP and item_id:
            self._duplicate_group_entry(item_id)

    def _duplicate_surface_entry(self, surface_id: str, reference_item: QTreeWidgetItem) -> None:
        parent_group_id = None
        if self.surface_manager:
            parent_group_item = self._find_parent_group_item(reference_item)
            if parent_group_item:
                _, parent_group_id = self._get_item_identity(parent_group_item)
            else:
                parent_group_id = self.surface_manager.get_root_group_id()

        duplicated_surface_id = self._duplicate_surface_definition(
            surface_id,
            parent_group_id,
            make_active=True,
        )
        if not duplicated_surface_id:
            return

        self.current_surface_id = duplicated_surface_id
        self.load_surfaces()
        self._set_points_panel_visible(True)
        # 🎯 Trigger Calc/Plot Update: Surface wurde dupliziert
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _duplicate_group_entry(self, group_id: str) -> None:
        if not self.surface_manager:
            return
        source_group = self._get_surface_group(group_id)
        if not source_group:
            return

        parent_group_id = source_group.parent_id or self.surface_manager.get_root_group_id()
        new_group_name = self._generate_group_duplicate_name(source_group.name, parent_group_id)
        new_group = self.surface_manager.create_surface_group(new_group_name, parent_id=parent_group_id)
        self._duplicate_group_contents(source_group, new_group.group_id)
        self.load_surfaces()
        if new_group.group_id in self._group_items:
            self.surface_tree.setCurrentItem(self._group_items[new_group.group_id])

    def _duplicate_surface_definition(
        self,
        surface_id: str,
        target_group_id: Optional[str],
        *,
        make_active: bool,
    ) -> Optional[str]:
        surface_store = self._get_surface_store()
        original = surface_store.get(surface_id)
        if original is None:
            return None

        new_surface_id = self._generate_surface_id(surface_store)
        new_surface_name = self._generate_surface_name(surface_store)
        duplicated = self._clone_surface_data(original)
        duplicated["name"] = new_surface_name
        duplicated.pop("group_id", None)
        duplicated.pop("locked", None)
        self.settings.add_surface_definition(new_surface_id, duplicated, make_active=make_active)

        if self.surface_manager:
            destination_group_id = target_group_id or self.surface_manager.get_root_group_id()
            self.surface_manager.assign_surface_to_group(new_surface_id, destination_group_id, create_missing=True)

        return new_surface_id

    def _duplicate_group_contents(self, source_group: SurfaceGroup, target_group_id: str) -> None:
        if not self.surface_manager:
            return

        for surface_id in list(source_group.surface_ids):
            self._duplicate_surface_definition(surface_id, target_group_id, make_active=False)

        for child_group_id in list(source_group.child_groups):
            child_group = self._get_surface_group(child_group_id)
            if not child_group:
                continue
            child_name = self._generate_group_duplicate_name(child_group.name, target_group_id)
            new_child_group = self.surface_manager.create_surface_group(child_name, parent_id=target_group_id)
            self._duplicate_group_contents(child_group, new_child_group.group_id)

    def _generate_group_duplicate_name(self, base_name: str, parent_group_id: Optional[str]) -> str:
        base = (base_name or "Group").strip() or "Group"
        existing_names: set[str] = set()
        if self.surface_manager:
            parent_id = parent_group_id or self.surface_manager.get_root_group_id()
            parent_group = self._get_surface_group(parent_id)
            if parent_group:
                for child_id in parent_group.child_groups:
                    child_group = self._get_surface_group(child_id)
                    if child_group:
                        existing_names.add(child_group.name)

        candidate = f"{base} Copy"
        index = 2
        while candidate in existing_names:
            candidate = f"{base} Copy {index}"
            index += 1
        return candidate

    def _handle_add_group(self, reference_item: Optional[QTreeWidgetItem]) -> None:
        if not self.surface_manager:
            return
        parent_group_id = self._determine_target_group_id(reference_item)
        name, ok = QInputDialog.getText(self, "Add Group", "Group name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        group = self.surface_manager.create_surface_group(name, parent_id=parent_group_id)
        self.load_surfaces()
        if group and group.group_id in self._group_items:
            self.surface_tree.setCurrentItem(self._group_items[group.group_id])

    def _handle_delete_group(self, group_id: str) -> None:
        if not self.surface_manager:
            return
        group = self._get_surface_group(group_id)
        if not group:
            return
        if group.locked:
            QtWidgets.QMessageBox.information(
                self,
                "Aktion nicht möglich",
                "Diese Gruppe ist geschützt und kann nicht gelöscht werden.",
            )
            return
        try:
            success = self.surface_manager.remove_surface_group(group_id)
        except ValueError:
            QtWidgets.QMessageBox.information(
                self,
                "Aktion nicht möglich",
                "Die Gruppe ist nicht leer und kann daher nicht gelöscht werden.",
            )
            return
        if success:
            self.load_surfaces()
            # 🎯 Trigger Calc/Plot Update: Gruppe wurde gelöscht
            # Entfernte Surfaces beeinflussen Berechnung und Overlays.
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_surface_selection_changed(self) -> None:
        if self._loading_surfaces:
            return

        item = self.surface_tree.currentItem()
        item_type, item_id = self._get_item_identity(item)
        surface_store = self._get_surface_store()

        # --- Gruppen-Selektions-Handling (rote Umrandung für alle Surfaces der Gruppe) ---
        if item_type == self.ITEM_TYPE_GROUP:
            self.current_surface_id = None
            self._loading_points = True
            self.points_tree.clear()
            self._loading_points = False

            group_id = item_id
            # Sammle rekursiv alle Surfaces aus der Gruppe und ihren Untergruppen
            highlight_ids = self._collect_all_surfaces_from_group(group_id, surface_store)

            # Setze active_surface_id auf None, damit nur highlight_ids verwendet werden
            try:
                setattr(self.settings, "active_surface_id", None)
            except Exception:
                pass
            
            # Speichere Highlight-Liste auf Settings, damit Overlays sie nutzen können
            setattr(self.settings, "active_surface_highlight_ids", highlight_ids)

            # Overlays aktualisieren (nur visuell, keine Neuberechnung)
            if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "draw_spl_plotter"):
                draw_spl = self.main_window.draw_plots.draw_spl_plotter
                if hasattr(draw_spl, "update_overlays"):
                    draw_spl.update_overlays(self.settings, self.container)
            return

        # --- Einzel-Surface-Selektions-Handling ---
        if item_type != self.ITEM_TYPE_SURFACE:
            self.current_surface_id = None
            self._loading_points = True
            self.points_tree.clear()
            self._loading_points = False
            # Keine Surface-Auswahl → keine speziellen Highlight-IDs
            setattr(self.settings, "active_surface_highlight_ids", [])
            return

        self.current_surface_id = item_id
        self._load_points_for_surface(item_id)
        self._activate_surface(item_id)

    def _handle_surface_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_surfaces:
            return

        item_type, identifier = self._get_item_identity(item)
        if item_type == self.ITEM_TYPE_GROUP:
            self._handle_group_item_changed(identifier, item, column)
            return
        if item_type != self.ITEM_TYPE_SURFACE:
            return

        surface_id = identifier
        surface_store = self._get_surface_store()
        if surface_id not in surface_store:
            return

        surface_data = surface_store[surface_id]

        if column == 0:
            new_name = item.text(0).strip()
            if not new_name:
                # Leerer Name nicht erlaubt → alten Wert wiederherstellen
                self._loading_surfaces = True
                item.setText(0, str(self._surface_get(surface_data, "name", "")))
                self._loading_surfaces = False
                return
            self._surface_set(surface_data, "name", new_name)
            # 🎯 Trigger Calc/Plot Update: Name-Änderung (Name könnte in Visualisierung angezeigt werden)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        elif column == 1:
            enabled = item.checkState(1) == Qt.Checked
            self._surface_set(surface_data, "enabled", enabled)
            self.settings.set_surface_enabled(surface_id, enabled)
            
            # 🎯 Trigger Calc/Plot Update: Enable-Status ändert sich
            # (Enable-Status-Änderung kann Berechnung beeinflussen → Empty Plot, beeinflusst immer Overlay)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        elif column == 2:
            hidden = item.checkState(2) == Qt.Checked
            self._surface_set(surface_data, "hidden", hidden)
            self.settings.set_surface_hidden(surface_id, hidden)
            
            # 🎯 Trigger Calc/Plot Update: Hide-Status ändert sich
            # (Hide-Status-Änderung kann Berechnung beeinflussen wenn Surface enabled ist, beeinflusst immer Overlay)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()

    def _activate_surface(self, surface_id: Optional[str]) -> None:
        """
        Aktiviert ein Surface für die Bearbeitung.
        Aktualisiert UI-Dimensionen und Overlays (für rote Markierung des aktiven Surfaces),
        aber löst KEINE Neuberechnung aus.
        Neuberechnungen erfolgen ausschließlich über _handle_surface_item_changed
        bei Enable/Hide-Änderungen.
        """
        if surface_id is None:
            return

        surface_store = self._get_surface_store()
        if surface_id not in surface_store:
            return

        previous_active = getattr(self.settings, "active_surface_id", None)

        try:
            self.settings.set_active_surface(surface_id)
        except KeyError:
            return

        # Einzel-Surface-Auswahl: nur dieses Surface hervorheben
        setattr(self.settings, "active_surface_highlight_ids", [surface_id])
        
        # Aktualisiere Overlays, wenn sich die Auswahl ändert (für rote Markierung)
        # KEINE Neuberechnung, nur visuelle Aktualisierung
        if surface_id != previous_active:
            if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "draw_spl_plotter"):
                if hasattr(self.main_window.draw_plots.draw_spl_plotter, "update_overlays"):
                    self.main_window.draw_plots.draw_spl_plotter.update_overlays(self.settings, self.container)

    def _on_surface_geometry_changed(self, surface_id: Optional[str]) -> None:
        """
        Wird aufgerufen, wenn sich die Geometrie eines Surfaces ändert (Punkte hinzugefügt/gelöscht/geändert).
        Prüft ob Calc/Plot aktualisiert werden muss und propagiert die Änderungen.
        """
        if surface_id is None:
            return

        surface_store = self._get_surface_store()
        if surface_id not in surface_store:
            return

        # 🎯 Trigger Calc/Plot Update: Punkt-Änderungen beeinflussen Berechnung und Plot
        # (Grid-Erstellung basiert auf Surface-Koordinaten, Overlays zeigen Surface-Geometrie)
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_surface_tree_drop(
        self,
        selected_items: List[QTreeWidgetItem],
        target_item: Optional[QTreeWidgetItem],
        position: int,
    ) -> bool:
        if not selected_items or not self.surface_manager:
            return False

        drop_pos = position
        root_group_id = self.surface_manager.get_root_group_id()
        destination_group_id = root_group_id

        def _describe_items(items):
            summary = []
            for entry in items:
                item_type, item_id = self._get_item_identity(entry)
                summary.append(f"{item_type}:{item_id}")
            return summary

        print(
            "[SurfaceDockWidget] drop handler start: items="
            + ", ".join(_describe_items(selected_items))
            + f", target={self._get_item_identity(target_item)}, pos={drop_pos}"
        )

        if target_item:
            target_type, target_id = self._get_item_identity(target_item)
            if drop_pos == QAbstractItemView.OnItem:
                # Drop direkt auf Item
                if target_type == self.ITEM_TYPE_GROUP:
                    destination_group_id = target_id
                elif target_type == self.ITEM_TYPE_SURFACE:
                    # Wenn auf Surface gedroppt wird, zur Parent-Gruppe verschieben
                    parent_group_item = self._find_parent_group_item(target_item)
                    if parent_group_item:
                        destination_group_id = self._get_item_identity(parent_group_item)[1]
                    else:
                        destination_group_id = root_group_id
                else:
                    destination_group_id = root_group_id
            elif drop_pos in (QAbstractItemView.AboveItem, QAbstractItemView.BelowItem):
                # Drop oberhalb/unterhalb eines Items - zur Parent-Gruppe des Ziel-Items
                parent_group_item = self._find_parent_group_item(target_item)
                if parent_group_item:
                    destination_group_id = self._get_item_identity(parent_group_item)[1]
                else:
                    destination_group_id = root_group_id
            else:
                # Unbekannte Position - zur Parent-Gruppe des Ziel-Items
                parent_group_item = self._find_parent_group_item(target_item)
                if parent_group_item:
                    destination_group_id = self._get_item_identity(parent_group_item)[1]
                else:
                    destination_group_id = root_group_id
        elif drop_pos == QAbstractItemView.OnViewport:
            # Drop auf leeren Bereich - zur Root-Gruppe
            destination_group_id = root_group_id
        else:
            # Unbekannte Position - zur Root-Gruppe
            destination_group_id = root_group_id

        moved = False
        for item in selected_items:
            item_type, item_id = self._get_item_identity(item)
            if not item_id:
                continue
            if item_type == self.ITEM_TYPE_SURFACE:
                print(
                    f"[SurfaceDockWidget] assign surface {item_id} -> group {destination_group_id}"
                )
                self.surface_manager.assign_surface_to_group(item_id, destination_group_id)
                moved = True
            elif item_type == self.ITEM_TYPE_GROUP:
                if item_id == destination_group_id:
                    continue
                print(
                    f"[SurfaceDockWidget] move group {item_id} -> group {destination_group_id}"
                )
                success = self.surface_manager.move_surface_group(item_id, destination_group_id)
                moved = moved or success

        if moved:
            print("[SurfaceDockWidget] drop handler finished with reload")
            # Struktur sicherstellen, bevor die Oberfläche neu geladen wird
            if self.surface_manager:
                self.surface_manager.ensure_surface_group_structure()
            self.load_surfaces()
        else:
            print("[SurfaceDockWidget] drop handler finished without move")
        return moved

    def _find_parent_group_item(self, item: Optional[QTreeWidgetItem]) -> Optional[QTreeWidgetItem]:
        parent = item.parent() if item else None
        while parent is not None:
            parent_type, _ = self._get_item_identity(parent)
            if parent_type == self.ITEM_TYPE_GROUP:
                return parent
            parent = parent.parent()
        return None

    def _show_surface_context_menu(self, position: QPoint) -> None:
        menu = QtWidgets.QMenu(self.surface_tree)
        add_surface_action = menu.addAction("Add Surface")
        duplicate_action = menu.addAction("Duplicate")
        delete_action = menu.addAction("Delete Surface")
        menu.addSeparator()
        add_group_action = menu.addAction("Add Group")
        delete_group_action = menu.addAction("Delete Group")

        item = self.surface_tree.itemAt(position)
        target_item = item if item is not None else self.surface_tree.currentItem()
        if item is not None:
            self.surface_tree.setCurrentItem(item)

        item_type, item_id = self._get_item_identity(target_item)

        if item_type != self.ITEM_TYPE_SURFACE:
            delete_action.setEnabled(False)
        else:
            if item_id == self.DEFAULT_SURFACE_ID:
                delete_action.setEnabled(False)

        if item_type != self.ITEM_TYPE_GROUP or not item_id:
            delete_group_action.setEnabled(False)

        action = menu.exec_(self.surface_tree.viewport().mapToGlobal(position))

        if action == add_surface_action:
            self._handle_add_surface()
        elif action == duplicate_action:
            self._handle_duplicate_item(target_item)
        elif action == delete_action:
            self._handle_delete_surface(target_item)
        elif action == add_group_action:
            self._handle_add_group(target_item)
        elif action == delete_group_action and item_type == self.ITEM_TYPE_GROUP and item_id:
            self._handle_delete_group(item_id)

    # ---- points handling --------------------------------------------

    def _select_surface(self, surface_id: Optional[str]) -> None:
        if surface_id and surface_id in self._surface_items:
            item = self._surface_items[surface_id]
            self.surface_tree.setCurrentItem(item)
            self.current_surface_id = surface_id
            return

        fallback_id = next(iter(self._surface_items.keys()), None)
        if fallback_id:
            item = self._surface_items[fallback_id]
            self.surface_tree.setCurrentItem(item)
            self.current_surface_id = fallback_id
        else:
            self.surface_tree.setCurrentItem(None)
            self.current_surface_id = None

    def _load_points_for_surface(self, surface_id: Optional[str]) -> None:
        self._loading_points = True
        self.points_tree.blockSignals(True)
        self.points_tree.clear()

        if surface_id is None:
            self.points_tree.blockSignals(False)
            self._loading_points = False
            return

        surface_store = self._get_surface_store()
        surface = surface_store.get(surface_id)
        if surface is None:
            self.points_tree.blockSignals(False)
            self._loading_points = False
            return

        points: List[SurfacePoint] = self._surface_points(surface)
        for index, point in enumerate(points):
            self._append_point_item(index, point)

        self._renumber_points()
        if self._points_panel_visible:
            self._set_points_panel_visible(True, force=True)
        self.points_tree.blockSignals(False)
        self._loading_points = False

    def _append_point_item(self, index: int, point: SurfacePoint) -> None:
        item = QTreeWidgetItem(self.points_tree)
        item.setFlags(
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )
        item.setText(0, f"P{index + 1}")
        item.setData(0, Qt.UserRole, index)
        item.setText(1, f"{float(point.get('x', 0.0)):.2f}")
        item.setText(2, f"{float(point.get('y', 0.0)):.2f}")
        item.setText(3, self._format_z_value(point.get("z")))

        self._create_point_editor(item, 1)
        self._create_point_editor(item, 2)
        self._create_point_editor(item, 3)

    def _handle_add_point(self) -> None:
        current_surface = self._get_current_surface_definition()
        if current_surface is None:
            QtWidgets.QMessageBox.information(
                self,
                "Keine Fläche ausgewählt",
                "Bitte zunächst eine Fläche auswählen oder anlegen.",
            )
            return

        new_point: SurfacePoint = {"x": 0.0, "y": 0.0, "z": 0.0}
        points = self._surface_points(current_surface, create=True)
        points.append(new_point)

        index = len(points) - 1
        self._append_point_item(index, new_point)
        self._set_points_panel_visible(True)
        self._renumber_points()
        self._sync_points_order()
        if self._has_complete_z_definition(current_surface):
            self._on_surface_geometry_changed(self.current_surface_id)

    def _handle_delete_point(self) -> None:
        current_surface = self._get_current_surface_definition()
        if current_surface is None:
            return

        item = self.points_tree.currentItem()
        if item is None:
            return

        index = item.data(0, Qt.UserRole)
        if not isinstance(index, int):
            return

        points = self._surface_points(current_surface)
        if 0 <= index < len(points):
            del points[index]

        root = self.points_tree.invisibleRootItem()
        root.removeChild(item)
        self._renumber_points()
        self._sync_points_order()
        if self._has_complete_z_definition(current_surface):
            self._on_surface_geometry_changed(self.current_surface_id)

    def _handle_point_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_points:
            return

        if column not in (1, 2, 3):
            return

        surface = self._get_current_surface_definition()
        if surface is None:
            return

        index = item.data(0, Qt.UserRole)
        if not isinstance(index, int):
            return

        points = self._surface_points(surface, create=True)
        all_zero_before = self._are_all_z_zero(points)

        try:
            if column == 3:
                raw_value = item.text(3).strip()
                value = None if raw_value == "" else float(raw_value.replace(",", "."))
            else:
                value = float(item.text(column).replace(",", "."))
        except ValueError:
            # Invalid auf alten Wert zurücksetzen
            self._loading_points = True
            coords = points[index]
            if column == 1:
                item.setText(1, f"{coords.get('x', 0.0):.2f}")
            elif column == 2:
                item.setText(2, f"{coords.get('y', 0.0):.2f}")
            else:
                item.setText(3, self._format_z_value(coords.get("z")))
            self._loading_points = False
            return

        coords = points[index]
        previous_coords = coords.copy()

        if column == 1:
            coords["x"] = value
            item.setText(1, f"{value:.2f}")
        elif column == 2:
            coords["y"] = value
            item.setText(2, f"{value:.2f}")
        else:
            coords["z"] = value
            item.setText(3, self._format_z_value(value))
            if value is not None and all_zero_before and not math.isclose(value, 0.0, abs_tol=1e-9):
                self._clear_other_z_values(surface, index)
            self._try_auto_complete_surface_z(surface)
        if not self._enforce_surface_planarity(surface):
            if column == 3:
                # Keine planare Fläche möglich → übrige Z-Werte zurücksetzen, Eingabe beibehalten
                item.setText(3, self._format_z_value(value))
                self._clear_other_z_values(surface, index)
            else:
                # Für X/Y weiterhin vorherigen Wert wiederherstellen
                self._loading_points = True
                coords.update(previous_coords)
                if column == 1:
                    item.setText(1, f"{previous_coords.get('x', 0.0):.2f}")
                elif column == 2:
                    item.setText(2, f"{previous_coords.get('y', 0.0):.2f}")
                self._loading_points = False
            return

        if self._has_complete_z_definition(surface):
            self._on_surface_geometry_changed(self.current_surface_id)

    # ---- helpers ----------------------------------------------------

    def _get_current_surface_definition(self) -> Optional[SurfaceRecord]:
        surface_store = self._get_surface_store()
        if not surface_store:
            return None

        surface_id = self.current_surface_id
        if surface_id is None:
            item = self.surface_tree.currentItem()
            if item is not None:
                item_type, candidate_id = self._get_item_identity(item)
                if item_type == self.ITEM_TYPE_SURFACE:
                    surface_id = candidate_id
                    self.current_surface_id = surface_id

        if surface_id is None:
            return None

        return surface_store.get(surface_id)

    def _get_surface_store(self) -> Dict[str, SurfaceRecord]:
        if not hasattr(self.settings, "surface_definitions") or self.settings.surface_definitions is None:
            self.settings.surface_definitions = {}
        return self.settings.surface_definitions

    @staticmethod
    def _surface_get(surface: SurfaceRecord | None, key: str, default: Any = None) -> Any:
        if surface is None:
            return default
        if isinstance(surface, dict):
            return surface.get(key, default)
        return getattr(surface, key, default)

    @staticmethod
    def _surface_set(surface: SurfaceRecord | None, key: str, value: Any) -> None:
        if surface is None:
            return
        if isinstance(surface, dict):
            surface[key] = value
        else:
            setattr(surface, key, value)

    @staticmethod
    def _surface_points(surface: SurfaceRecord | None, *, create: bool = False) -> List[SurfacePoint]:
        points = SurfaceDockWidget._surface_get(surface, "points", None)
        if points is None and create:
            points = []
            SurfaceDockWidget._surface_set(surface, "points", points)
        return points or []

    def _validate_surface_planarity(
        self,
        surface: Optional[SurfaceRecord],
        *,
        show_warning: bool = True,
    ) -> bool:
        """
        Prüft, ob die Z-Werte einer Surface konform mit den Planar-Regeln sind.
        """
        if surface is None:
            return True

        points = self._surface_points(surface)
        numeric_points = self._prepare_numeric_points(points, require_complete=True)
        if not numeric_points:
            # Noch nicht vollständig definiert → keine Warnung anzeigen
            return True

        model, error = derive_surface_plane(numeric_points)
        if model is None:
            if show_warning:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ungültige Z-Geometrie",
                    (
                        f"Die Fläche '{self._surface_get(surface, 'name', 'Surface')}' kann nicht gezeichnet werden:\n"
                        f"{error}"
                    ),
                )
            return False
        return True

    def _enforce_surface_planarity(self, surface: SurfaceRecord) -> bool:
        """
        Erzwingt planare Z-Werte, indem die Fläche auf das beste gültige Modell projiziert wird.
        """
        if surface is None:
            return False

        points = self._surface_points(surface)
        if not points:
            return False

        valid_points = [point for point in points if self._is_valid_z_value(point.get("z"))]
        if len(valid_points) < len(points):
            # Noch unvollständig definierte Flächen – später erneut prüfen
            return True

        numeric_points = self._prepare_numeric_points(points, require_complete=True)
        if not numeric_points:
            return False

        model, _ = build_planar_model(numeric_points)
        if model is None:
            return False

        previous_loading = self._loading_points
        self._loading_points = True
        try:
            for idx, point in enumerate(points):
                numeric_x = float(self._coerce_numeric(point.get("x")) or 0.0)
                numeric_y = float(self._coerce_numeric(point.get("y")) or 0.0)
                new_z = evaluate_surface_plane(
                    model,
                    numeric_x,
                    numeric_y,
                )
                point["z"] = new_z
                item = self.points_tree.topLevelItem(idx)
                if item is not None:
                    item.setText(3, f"{new_z:.2f}")
        finally:
            self._loading_points = previous_loading

        return True

    def _ensure_default_surface(self, surface_store: Dict[str, SurfaceRecord]) -> None:
        if self.DEFAULT_SURFACE_ID not in surface_store:
            surface_store[self.DEFAULT_SURFACE_ID] = {
                "name": self.DEFAULT_SURFACE_NAME,
                "enabled": True,
                "hidden": False,
                "points": self._create_default_points(),
                "locked": True,
            }

    def _generate_surface_id(self, surface_store: Dict[str, SurfaceRecord]) -> str:
        existing = set(surface_store.keys())
        index = 1
        while True:
            candidate = f"surface_{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _generate_surface_name(self, surface_store: Dict[str, SurfaceRecord]) -> str:
        existing_names = {str(self._surface_get(data, "name", "")) for data in surface_store.values()}
        index = 1
        while True:
            name = f"Surface {index}"
            if name not in existing_names:
                return name
            index += 1

    @staticmethod
    def _create_default_points() -> List[SurfacePoint]:
        return [
            {"x": 0, "y": 0, "z": 0.0},
            {"x": 0, "y": 0, "z": 0.0},
            {"x": 0, "y": 0, "z": 0.0},
            {"x": 0, "y": 0, "z": 0.0},
        ]

    # ---- geometry helpers -------------------------------------------

    @staticmethod
    def _clone_surface_data(surface: SurfaceRecord) -> Dict[str, Any]:
        """
        Gibt eine vollständig kopierbare Dict-Repräsentation eines Surface zurück.
        Unterstützt sowohl dict-basierte als auch dataclass-basierte Objekte.
        """
        if isinstance(surface, dict):
            cloned = copy.deepcopy(surface)
        elif hasattr(surface, "to_dict"):
            cloned = copy.deepcopy(surface.to_dict())
        else:
            # Fallback – versuche generisch zu kopieren
            cloned = copy.deepcopy(dict(surface)) if isinstance(surface, dict) else surface.to_dict()

        cloned.pop("surface_id", None)
        return cloned

    @staticmethod
    def _coerce_numeric(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            text = value.strip().replace(",", ".")
            if not text:
                return None
            try:
                numeric = float(text)
            except ValueError:
                return None
        else:
            return None
        if math.isnan(numeric):
            return None
        return numeric

    @classmethod
    def _build_numeric_point(cls, point: SurfacePoint) -> Optional[Dict[str, float]]:
        numeric_z = cls._coerce_numeric(point.get("z"))
        if numeric_z is None:
            return None
        numeric_x = cls._coerce_numeric(point.get("x"))
        numeric_y = cls._coerce_numeric(point.get("y"))
        return {
            "x": float(numeric_x if numeric_x is not None else 0.0),
            "y": float(numeric_y if numeric_y is not None else 0.0),
            "z": numeric_z,
        }

    @classmethod
    def _prepare_numeric_points(
        cls,
        points: List[SurfacePoint],
        *,
        indices: Optional[List[int]] = None,
        require_complete: bool = False,
    ) -> List[Dict[str, float]]:
        prepared: List[Dict[str, float]] = []
        iterable = indices if indices is not None else list(range(len(points)))
        for idx in iterable:
            if idx < 0 or idx >= len(points):
                continue
            numeric_point = cls._build_numeric_point(points[idx])
            if numeric_point is None:
                if require_complete:
                    return []
                continue
            prepared.append(numeric_point)
        return prepared

    @classmethod
    def _is_valid_z_value(cls, value: Any) -> bool:
        return cls._coerce_numeric(value) is not None

    @classmethod
    def _format_z_value(cls, value: Any) -> str:
        numeric = cls._coerce_numeric(value)
        if numeric is None:
            return ""
        return f"{numeric:.2f}"

    def _are_all_z_zero(self, points: List[SurfacePoint], *, tol: float = 1e-3) -> bool:
        if not points:
            return False
        for point in points:
            numeric = self._coerce_numeric(point.get("z"))
            if numeric is None:
                return False
            if not math.isclose(numeric, 0.0, abs_tol=tol):
                return False
        return True

    def _clear_other_z_values(self, surface: SurfaceRecord, keep_index: int) -> None:
        points = self._surface_points(surface)
        previous_loading = self._loading_points
        self._loading_points = True
        try:
            for idx, point in enumerate(points):
                if idx == keep_index:
                    continue
                point["z"] = None
                item = self.points_tree.topLevelItem(idx)
                if item is not None:
                    item.setText(3, "")
        finally:
            self._loading_points = previous_loading

    def _try_auto_complete_surface_z(self, surface: SurfaceRecord) -> bool:
        points = self._surface_points(surface)
        if len(points) < 3:
            return False

        defined_indices = [idx for idx, point in enumerate(points) if self._is_valid_z_value(point.get("z"))]
        missing_indices = [idx for idx in range(len(points)) if idx not in defined_indices]

        if len(defined_indices) < 3 or len(missing_indices) != 1:
            return False

        defined_points = self._prepare_numeric_points(points, indices=defined_indices)
        model, _ = build_planar_model(defined_points)
        if model is None:
            return False

        missing_index = missing_indices[0]
        missing_point = points[missing_index]
        new_z = evaluate_surface_plane(
            model,
            float(self._coerce_numeric(missing_point.get("x")) or 0.0),
            float(self._coerce_numeric(missing_point.get("y")) or 0.0),
        )

        previous_loading = self._loading_points
        self._loading_points = True
        try:
            missing_point["z"] = new_z
            item = self.points_tree.topLevelItem(missing_index)
            if item is not None:
                item.setText(3, f"{new_z:.2f}")
        finally:
            self._loading_points = previous_loading
        return True

    def _has_complete_z_definition(self, surface: SurfaceRecord) -> bool:
        points = self._surface_points(surface)
        if not points:
            return False
        return all(self._is_valid_z_value(point.get("z")) for point in points)

    # ---- UI helpers -------------------------------------------------

    def _on_points_toggle(self, checked: bool) -> None:
        self._set_points_panel_visible(checked)

    def _set_points_panel_visible(self, visible: bool, force: bool = False) -> None:
        if not force and self._points_panel_visible == visible:
            return

        self._points_panel_visible = visible

        if self.toggle_points_button.isChecked() != visible:
            self.toggle_points_button.blockSignals(True)
            self.toggle_points_button.setChecked(visible)
            self.toggle_points_button.blockSignals(False)

        if visible:
            bottom_height = self._calculate_points_panel_height()
            total_height = self.splitter.size().height()
            if total_height <= 0:
                total_height = bottom_height + 280
            top_height = max(total_height - bottom_height, 240)
            self._last_splitter_sizes = [top_height, bottom_height]
            self.points_tree.setMinimumHeight(bottom_height)
            self.points_tree.show()
            self.toggle_points_button.setArrowType(Qt.UpArrow)
            self.splitter.setSizes(self._last_splitter_sizes)
        else:
            if not force:
                current_sizes = self.splitter.sizes()
                if len(current_sizes) == 2 and current_sizes[1] > 0:
                    self._last_splitter_sizes = current_sizes
            total_height = sum(self._last_splitter_sizes)
            if total_height == 0:
                total_height = 600
                self._last_splitter_sizes = [400, 200]
            self.points_tree.hide()
            self.toggle_points_button.setArrowType(Qt.DownArrow)
            self.splitter.setSizes([total_height, 0])

    def _show_points_context_menu(self, position: QPoint) -> None:
        menu = QtWidgets.QMenu(self.points_tree)
        add_action = menu.addAction("Add Point")
        delete_action = menu.addAction("Delete Point")

        index = self.points_tree.indexAt(position)
        if not index.isValid():
            delete_action.setEnabled(False)

        action = menu.exec_(self.points_tree.viewport().mapToGlobal(position))

        if action == add_action:
            self._handle_add_point()
        elif action == delete_action:
            self._handle_delete_point()

    def _renumber_points(self) -> None:
        for idx in range(self.points_tree.topLevelItemCount()):
            item = self.points_tree.topLevelItem(idx)
            item.setText(0, f"P{idx + 1}")
            item.setData(0, Qt.UserRole, idx)

    def _sync_points_order(self) -> None:
        surface = self._get_current_surface_definition()
        if surface is None:
            return

        points = self._surface_points(surface, create=True)
        new_order: List[SurfacePoint] = []
        for idx in range(self.points_tree.topLevelItemCount()):
            item = self.points_tree.topLevelItem(idx)
            index = item.data(0, Qt.UserRole)
            if isinstance(index, int) and 0 <= index < len(points):
                new_order.append(points[index])
            else:
                coords = {
                    "x": float(item.text(1)),
                    "y": float(item.text(2)),
                    "z": float(item.text(3)),
                }
                new_order.append(coords)

        self._surface_set(surface, "points", new_order)
        self._renumber_points()
        if self._points_panel_visible:
            self._set_points_panel_visible(True, force=True)

    def _calculate_points_panel_height(self) -> int:
        header = self.points_tree.header()
        header_height = 0
        if header:
            header_height = header.height() or header.sizeHint().height()
        frame = self.points_tree.frameWidth() * 2
        row_height = self.points_tree.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 24
        visible_rows = self.points_tree.topLevelItemCount()
        if visible_rows == 0:
            visible_rows = 4
        elif visible_rows > 4:
            visible_rows = 4
        return header_height + frame + row_height * visible_rows + 8

    def _create_point_editor(self, item: QTreeWidgetItem, column: int) -> None:
        editor = PointValueEdit(self.points_tree, item, column)
        editor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        editor.setText(item.text(column))
        editor.setFixedHeight(22)
        editor.setValidator(QDoubleValidator(-1000.0, 1000.0, 2))
        editor.editingFinished.connect(
            lambda it=item, col=column, ed=editor: self._on_point_editor_finished(it, col, ed)
        )
        self.points_tree.setItemWidget(item, column, editor)

    def _on_point_editor_finished(self, item: QTreeWidgetItem, column: int, editor: QLineEdit) -> None:
        text = editor.text().strip()
        if not text:
            text = "0"
        try:
            value = float(text)
        except ValueError:
            editor.setText(item.text(column))
            return

        formatted = f"{value:.2f}"
        if item.text(column) != formatted:
            item.setText(column, formatted)
        editor.setText(formatted)


