from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

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
)
from PyQt5.QtGui import QFont, QDoubleValidator

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    build_planar_model,
    derive_surface_plane,
    evaluate_surface_plane,
)


SurfacePoint = Dict[str, float]
SurfaceDefinition = Dict[str, object]


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

    def __init__(self, main_window, settings, container):
        super().__init__("Surface", main_window)
        self.main_window = main_window
        self.settings = settings
        self.container = container

        self.current_surface_id: Optional[str] = None
        self._loading_surfaces = False
        self._loading_points = False

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
        self._loading_surfaces = True
        self.surface_tree.blockSignals(True)
        self.surface_tree.clear()

        surface_store = self._get_surface_store()
        self._ensure_default_surface(surface_store)

        for surface_id, surface_data in surface_store.items():
            item = self._create_surface_item(surface_id, surface_data)
            self.surface_tree.addTopLevelItem(item)

        self.surface_tree.expandAll()
        self.surface_tree.blockSignals(False)
        self._loading_surfaces = False

        target_surface = self.current_surface_id or self.DEFAULT_SURFACE_ID
        self._select_surface(target_surface)

        if self.surface_tree.currentItem() is None:
            self._loading_points = True
            self.points_tree.clear()
            self._loading_points = False

        self._activate_surface(self.current_surface_id)

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
        toggle_layout.addStretch(1)
        main_layout.addLayout(toggle_layout)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(6)

        # Gemeinsame Schriftgröße
        tree_font = QFont()
        tree_font.setPointSize(11)

        # TreeWidget für Flächen
        self.surface_tree = QTreeWidget()
        self.surface_tree.setHeaderLabels(["Surface", "Enable", "Hide"])
        self.surface_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.surface_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.surface_tree.setRootIsDecorated(False)
        self.surface_tree.setAlternatingRowColors(True)
        self.surface_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.surface_tree.setColumnWidth(0, 95)
        self.surface_tree.setColumnWidth(1, 50)
        self.surface_tree.setColumnWidth(2, 45)
        self.surface_tree.setFont(tree_font)
        header = self.surface_tree.header()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.surface_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.surface_tree.customContextMenuRequested.connect(self._show_surface_context_menu)
        self.surface_tree.setStyleSheet(
            "QTreeWidget { font-size: 11pt; }"
            "QTreeWidget::item { height: 22px; }"
        )
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

    def _create_surface_item(self, surface_id: str, data: SurfaceDefinition) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        item.setText(0, str(data.get("name", "Surface")))
        item.setFlags(
            Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
        )
        item.setCheckState(1, Qt.Checked if data.get("enabled", False) else Qt.Unchecked)
        item.setCheckState(2, Qt.Checked if data.get("hidden", False) else Qt.Unchecked)
        item.setData(0, Qt.UserRole, surface_id)

        if surface_id == self.DEFAULT_SURFACE_ID:
            # Default-Fläche: nicht löschbar, aber umbenennbar
            item.setFlags(item.flags() & ~Qt.ItemIsDragEnabled)

        return item

    def _handle_add_surface(self) -> None:
        surface_store = self._get_surface_store()
        surface_id = self._generate_surface_id(surface_store)
        surface_data: SurfaceDefinition = {
            "name": self._generate_surface_name(surface_store),
            "enabled": False,
            "hidden": False,
            "points": self._create_default_points(),
        }

        self.settings.add_surface_definition(surface_id, surface_data, make_active=True)

        item = self._create_surface_item(surface_id, surface_data)
        self.surface_tree.addTopLevelItem(item)
        self.surface_tree.setCurrentItem(item)
        self.current_surface_id = surface_id
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

        surface_id = item.data(0, Qt.UserRole)
        if surface_id == self.DEFAULT_SURFACE_ID:
            QtWidgets.QMessageBox.information(
                self,
                "Aktion nicht möglich",
                "Die Standardfläche kann nicht gelöscht werden.",
            )
            return

        surface_store = self._get_surface_store()
        if surface_id in surface_store:
            self.settings.remove_surface_definition(surface_id)

        index = self.surface_tree.indexOfTopLevelItem(item)
        self.surface_tree.takeTopLevelItem(index)

        if self.current_surface_id == surface_id:
            self.current_surface_id = None

        if self.surface_tree.topLevelItemCount() > 0:
            next_index = min(index, self.surface_tree.topLevelItemCount() - 1)
            next_item = self.surface_tree.topLevelItem(next_index)
            self.surface_tree.setCurrentItem(next_item)
        else:
            self.surface_tree.setCurrentItem(None)
            self._load_points_for_surface(None)

        # 🎯 Trigger Calc/Plot Update: Surface wurde gelöscht
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_duplicate_surface(self, item: Optional[QTreeWidgetItem]) -> None:
        if item is None:
            item = self.surface_tree.currentItem()
        if item is None:
            return

        surface_store = self._get_surface_store()
        surface_id = item.data(0, Qt.UserRole)
        if surface_id not in surface_store:
            return

        original = surface_store[surface_id]
        new_surface_id = self._generate_surface_id(surface_store)
        new_surface_name = self._generate_surface_name(surface_store)
        duplicated = copy.deepcopy(original)
        duplicated["name"] = new_surface_name
        duplicated.pop("locked", None)
        self.settings.add_surface_definition(new_surface_id, duplicated, make_active=True)

        new_item = self._create_surface_item(new_surface_id, duplicated)
        self.surface_tree.addTopLevelItem(new_item)
        self.surface_tree.setCurrentItem(new_item)
        self.current_surface_id = new_surface_id
        self._set_points_panel_visible(True)
        self._load_points_for_surface(new_surface_id)
        # 🎯 Trigger Calc/Plot Update: Surface wurde dupliziert
        if hasattr(self.main_window, "draw_plots"):
            if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                self.main_window.draw_plots.update_plots_for_surface_state()

    def _handle_surface_selection_changed(self) -> None:
        if self._loading_surfaces:
            return

        item = self.surface_tree.currentItem()
        surface_id = item.data(0, Qt.UserRole) if item else None
        self.current_surface_id = surface_id
        self._load_points_for_surface(surface_id)
        self._activate_surface(surface_id)

    def _handle_surface_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading_surfaces:
            return

        surface_id = item.data(0, Qt.UserRole)
        surface_store = self._get_surface_store()
        if surface_id not in surface_store:
            return

        surface_data = surface_store[surface_id]

        if column == 0:
            new_name = item.text(0).strip()
            if not new_name:
                # Leerer Name nicht erlaubt → alten Wert wiederherstellen
                self._loading_surfaces = True
                item.setText(0, str(surface_data.get("name", "")))
                self._loading_surfaces = False
                return
            surface_data["name"] = new_name
            # 🎯 Trigger Calc/Plot Update: Name-Änderung (Name könnte in Visualisierung angezeigt werden)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        elif column == 1:
            enabled = item.checkState(1) == Qt.Checked
            surface_data["enabled"] = enabled
            self.settings.set_surface_enabled(surface_id, enabled)
            
            # 🎯 Trigger Calc/Plot Update: Enable-Status ändert sich
            # (Enable-Status-Änderung kann Berechnung beeinflussen → Empty Plot, beeinflusst immer Overlay)
            if hasattr(self.main_window, "draw_plots"):
                if hasattr(self.main_window.draw_plots, "update_plots_for_surface_state"):
                    self.main_window.draw_plots.update_plots_for_surface_state()
        elif column == 2:
            hidden = item.checkState(2) == Qt.Checked
            surface_data["hidden"] = hidden
            
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

    def _show_surface_context_menu(self, position: QPoint) -> None:
        menu = QtWidgets.QMenu(self.surface_tree)
        add_action = menu.addAction("Add Surface")
        duplicate_action = menu.addAction("Duplicate Surface")
        delete_action = menu.addAction("Delete Surface")

        item = self.surface_tree.itemAt(position)
        target_item = item if item is not None else self.surface_tree.currentItem()

        if item is not None:
            self.surface_tree.setCurrentItem(item)

        if target_item is None:
            delete_action.setEnabled(False)
            duplicate_action.setEnabled(False)
        else:
            surface_id = target_item.data(0, Qt.UserRole)
            if surface_id == self.DEFAULT_SURFACE_ID:
                delete_action.setEnabled(False)
            if surface_id is None:
                duplicate_action.setEnabled(False)

        action = menu.exec_(self.surface_tree.viewport().mapToGlobal(position))

        if action == add_action:
            self._handle_add_surface()
        elif action == duplicate_action:
            self._handle_duplicate_surface(target_item)
        elif action == delete_action:
            self._handle_delete_surface(target_item)

    # ---- points handling --------------------------------------------

    def _select_surface(self, surface_id: Optional[str]) -> None:
        if surface_id is not None:
            for idx in range(self.surface_tree.topLevelItemCount()):
                item = self.surface_tree.topLevelItem(idx)
                if item.data(0, Qt.UserRole) == surface_id:
                    self.surface_tree.setCurrentItem(item)
                    self.current_surface_id = surface_id
                    return

        if self.surface_tree.topLevelItemCount() > 0:
            item = self.surface_tree.topLevelItem(0)
            self.surface_tree.setCurrentItem(item)
            self.current_surface_id = item.data(0, Qt.UserRole)
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

        points: List[SurfacePoint] = surface.get("points", [])
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
        item.setText(3, f"{float(point.get('z', 0.0)):.2f}")

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
        current_surface.setdefault("points", []).append(new_point)

        index = len(current_surface["points"]) - 1
        self._append_point_item(index, new_point)
        self._set_points_panel_visible(True)
        self._renumber_points()
        self._sync_points_order()
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

        points = current_surface.get("points", [])
        if 0 <= index < len(points):
            del points[index]

        root = self.points_tree.invisibleRootItem()
        root.removeChild(item)
        self._renumber_points()
        self._sync_points_order()
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

        try:
            value = float(item.text(column))
        except ValueError:
            # Invalid auf alten Wert zurücksetzen
            self._loading_points = True
            coords = surface["points"][index]
            if column == 1:
                item.setText(1, f"{coords.get('x', 0.0):.2f}")
            elif column == 2:
                item.setText(2, f"{coords.get('y', 0.0):.2f}")
            else:
                item.setText(3, f"{coords.get('z', 0.0):.2f}")
            self._loading_points = False
            return

        coords = surface["points"][index]
        previous_coords = coords.copy()

        if column == 1:
            coords["x"] = value
            item.setText(1, f"{value:.2f}")
        elif column == 2:
            coords["y"] = value
            item.setText(2, f"{value:.2f}")
        else:
            coords["z"] = value
            item.setText(3, f"{value:.2f}")
        if not self._enforce_surface_planarity(surface):
            # Rückgängig machen, falls Modell nicht bestimmt werden konnte
            self._loading_points = True
            coords.update(previous_coords)
            if column == 1:
                item.setText(1, f"{previous_coords.get('x', 0.0):.2f}")
            elif column == 2:
                item.setText(2, f"{previous_coords.get('y', 0.0):.2f}")
            else:
                item.setText(3, f"{previous_coords.get('z', 0.0):.2f}")
            self._loading_points = False
            return

        self._on_surface_geometry_changed(self.current_surface_id)

    # ---- helpers ----------------------------------------------------

    def _get_current_surface_definition(self) -> Optional[SurfaceDefinition]:
        surface_store = self._get_surface_store()
        if not surface_store:
            return None

        surface_id = self.current_surface_id
        if surface_id is None:
            item = self.surface_tree.currentItem()
            if item is not None:
                surface_id = item.data(0, Qt.UserRole)
                self.current_surface_id = surface_id

        if surface_id is None:
            return None

        return surface_store.get(surface_id)

    def _get_surface_store(self) -> Dict[str, SurfaceDefinition]:
        if not hasattr(self.settings, "surface_definitions") or self.settings.surface_definitions is None:
            self.settings.surface_definitions = {}
        return self.settings.surface_definitions

    def _validate_surface_planarity(
        self,
        surface: Optional[SurfaceDefinition],
        *,
        show_warning: bool = True,
    ) -> bool:
        """
        Prüft, ob die Z-Werte einer Surface konform mit den Planar-Regeln sind.
        """
        if surface is None:
            return True

        points = surface.get("points", [])
        model, error = derive_surface_plane(points)
        if model is None:
            if show_warning:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ungültige Z-Geometrie",
                    (
                        f"Die Fläche '{surface.get('name', 'Surface')}' kann nicht gezeichnet werden:\n"
                        f"{error}"
                    ),
                )
            return False
        return True

    def _enforce_surface_planarity(self, surface: SurfaceDefinition) -> bool:
        """
        Erzwingt planare Z-Werte, indem die Fläche auf das beste gültige Modell projiziert wird.
        """
        if surface is None:
            return False

        points = surface.get("points", [])
        model, needs_adjustment = build_planar_model(points)
        if model is None:
            return False

        if needs_adjustment:
            previous_loading = self._loading_points
            self._loading_points = True
            try:
                for idx, point in enumerate(points):
                    new_z = evaluate_surface_plane(
                        model,
                        float(point.get("x", 0.0)),
                        float(point.get("y", 0.0)),
                    )
                    point["z"] = new_z
                    item = self.points_tree.topLevelItem(idx)
                    if item is not None:
                        item.setText(3, f"{new_z:.2f}")
            finally:
                self._loading_points = previous_loading

        return True

    def _ensure_default_surface(self, surface_store: Dict[str, SurfaceDefinition]) -> None:
        if self.DEFAULT_SURFACE_ID not in surface_store:
            surface_store[self.DEFAULT_SURFACE_ID] = {
                "name": self.DEFAULT_SURFACE_NAME,
                "enabled": True,
                "hidden": False,
                "points": self._create_default_points(),
                "locked": True,
            }

    def _generate_surface_id(self, surface_store: Dict[str, SurfaceDefinition]) -> str:
        existing = set(surface_store.keys())
        index = 1
        while True:
            candidate = f"surface_{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _generate_surface_name(self, surface_store: Dict[str, SurfaceDefinition]) -> str:
        existing_names = {data.get("name", "") for data in surface_store.values()}
        index = 1
        while True:
            name = f"Surface {index}"
            if name not in existing_names:
                return name
            index += 1

    @staticmethod
    def _create_default_points() -> List[SurfacePoint]:
        return [
            {"x": 75.0, "y": -50.0, "z": 0.0},
            {"x": 75.0, "y": 50.0, "z": 0.0},
            {"x": -75.0, "y": 50.0, "z": 0.0},
            {"x": -75.0, "y": -50.0, "z": 0.0},
        ]

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

        points = surface.setdefault("points", [])
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

        surface["points"] = new_order
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


