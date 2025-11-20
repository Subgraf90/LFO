import math
from typing import Dict, Optional, List

from PyQt5.QtCore import Qt

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    SurfaceGroup,
    build_planar_model,
    evaluate_surface_plane,
)


class UISurfaceManager:
    """
    Verwaltet das SurfaceDockWidget und synchronisiert es mit dem ausgewählten SpeakerArray.
    """

    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.surface_dock_widget = None
        self.sources_widget = None
        self.snapshot_dock = None
        self._initial_docked = False
        self._snapshot_split_done = False
        self._group_controller = _SurfaceGroupController(self.settings)
        self._group_controller.ensure_structure()

    # ---- public API -------------------------------------------------

    def show_surface_dock_widget(self):
        """
        Erstellt (falls nötig) und zeigt das SurfaceDockWidget an.
        Positioniert es rechts neben dem Hauptfenster als schwebendes Fenster.
        """
        if self.surface_dock_widget is None:
            from Module_LFO.Modules_Window.WindowSurfaceWidget import SurfaceDockWidget

            self.surface_dock_widget = SurfaceDockWidget(
                self.main_window,
                self.settings,
                self.container,
                self,
            )
            # Automatisches Docking deaktiviert - Widget wird nicht direkt an UI angehängt

        # Vor der Darstellung sicherstellen, dass alle vorhandenen Flächen plan projiziert sind
        self._ensure_planar_surfaces()
        self.surface_dock_widget.initialize()
        
        # Als schwebendes Fenster setzen und rechts neben dem Hauptfenster positionieren
        self.surface_dock_widget.setFloating(True)
        
        # Position berechnen: rechts neben dem Hauptfenster
        main_geometry = self.main_window.geometry()
        main_x = main_geometry.x()
        main_y = main_geometry.y()
        main_width = main_geometry.width()
        main_height = main_geometry.height()
        
        # DockWidget-Größe (Standard oder bereits gesetzt)
        dock_width = self.surface_dock_widget.width() if self.surface_dock_widget.width() > 0 else 520
        dock_height = self.surface_dock_widget.height() if self.surface_dock_widget.height() > 0 else 600
        
        # Position: rechts neben dem Hauptfenster, vertikal zentriert
        offset_x = 20  # Kleiner Abstand zum Hauptfenster
        new_x = main_x + main_width + offset_x
        new_y = main_y + (main_height - dock_height) // 2  # Vertikal zentriert
        
        # Stelle sicher, dass das Fenster auf dem Bildschirm bleibt
        screen = self.main_window.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            # Wenn das Fenster außerhalb des Bildschirms wäre, positioniere es innerhalb
            if new_x + dock_width > screen_geometry.right():
                new_x = screen_geometry.right() - dock_width - 20
            if new_y + dock_height > screen_geometry.bottom():
                new_y = screen_geometry.bottom() - dock_height - 20
            if new_y < screen_geometry.top():
                new_y = screen_geometry.top() + 20
        
        self.surface_dock_widget.setGeometry(new_x, new_y, dock_width, dock_height)
        self.surface_dock_widget.show()
        # Automatische Ausrichtung mit Snapshot deaktiviert

    def close_surface_dock_widget(self):
        """
        Schließt das SurfaceDockWidget und entfernt Signalverbindungen.
        """
        if self.surface_dock_widget:
            self.surface_dock_widget.close()
            self.main_window.removeDockWidget(self.surface_dock_widget)
            self.surface_dock_widget.deleteLater()
            self.surface_dock_widget = None
            self._initial_docked = False
            self._snapshot_split_done = False

    def bind_to_sources_widget(self, sources_widget):
        """
        Verbindet die Auswahländerungen aus dem Sources-TreeWidget mit dem Surface-Widget.
        """
        if sources_widget is None or not hasattr(sources_widget, "sources_tree_widget"):
            return

        if self.sources_widget is not None:
            self._disconnect_sources_signals()

        self.sources_widget = sources_widget
        self.sources_widget.sources_tree_widget.itemSelectionChanged.connect(self._on_sources_selection_changed)

        if self.surface_dock_widget is not None:
            self._on_sources_selection_changed()

    def on_snapshot_dock_available(self, snapshot_dock):
        """
        Wird aufgerufen, sobald das Snapshot-Dock erstellt wurde.
        """
        self.snapshot_dock = snapshot_dock
        self._snapshot_split_done = False
        self._align_with_snapshot_if_possible()

    # ---- surface group helpers -------------------------------------

    def ensure_surface_group_structure(self):
        self._group_controller.ensure_structure()

    def reset_surface_storage(self):
        self._group_controller.reset_storage()

    def get_surface_group(self, group_id: Optional[str]) -> Optional[SurfaceGroup]:
        return self._group_controller.get_group(group_id)

    def list_surface_groups(self) -> Dict[str, SurfaceGroup]:
        return self._group_controller.list_groups()

    def get_root_group_id(self) -> str:
        return self._group_controller.root_group_id

    def assign_surface_to_group(self, surface_id: str, group_id: Optional[str], *, create_missing: bool = False):
        self._group_controller.assign_surface_to_group(surface_id, group_id, create_missing=create_missing)

    def create_surface_group(self, name: str, parent_id: Optional[str] = None, *, group_id: Optional[str] = None, locked: bool = False) -> SurfaceGroup:
        return self._group_controller.create_surface_group(name, parent_id=parent_id, group_id=group_id, locked=locked)

    def remove_surface_group(self, group_id: str) -> bool:
        return self._group_controller.remove_surface_group(group_id)

    def move_surface_group(self, group_id: str, target_parent_id: str) -> bool:
        return self._group_controller.move_surface_group(group_id, target_parent_id)

    def rename_surface_group(self, group_id: str, new_name: str):
        self._group_controller.rename_surface_group(group_id, new_name)

    def find_surface_group_by_name(self, name: str, parent_id: Optional[str] = None) -> Optional[SurfaceGroup]:
        return self._group_controller.find_surface_group_by_name(name, parent_id)

    def set_surface_group_enabled(self, group_id: str, enabled: bool):
        self._group_controller.set_surface_group_enabled(group_id, enabled)

    def set_surface_group_hidden(self, group_id: str, hidden: bool):
        self._group_controller.set_surface_group_hidden(group_id, hidden)

    def ensure_group_path(self, label: Optional[str]) -> Optional[str]:
        return self._group_controller.ensure_group_path(label)

    def detach_surface_from_group(self, surface_id: str):
        self._group_controller.detach_surface(surface_id)

    # ---- internal callbacks ----------------------------------------

    def _on_sources_selection_changed(self):
        if self.surface_dock_widget is None:
            return

        self._ensure_planar_surfaces()
        self.surface_dock_widget.load_surfaces()

    def _disconnect_sources_signals(self):
        if self.sources_widget and hasattr(self.sources_widget, "sources_tree_widget"):
            try:
                self.sources_widget.sources_tree_widget.itemSelectionChanged.disconnect(self._on_sources_selection_changed)
            except (RuntimeError, TypeError):
                pass

    def _ensure_planar_surfaces(self) -> bool:
        """
        Prüft alle gespeicherten Flächen auf Planarität und projiziert abweichende Punkte
        automatisch auf die dazugehörige Ebene.

        Returns:
            bool: True, wenn mindestens eine Fläche angepasst wurde.
        """
        surface_store = getattr(self.settings, "surface_definitions", None)
        if not isinstance(surface_store, dict) or not surface_store:
            return False

        adjusted = False
        for surface_id, surface in surface_store.items():
            if isinstance(surface, SurfaceDefinition):
                points = surface.points
            else:
                points = surface.get("points") or []
            if len(points) < 4:  # Mindestens vier Punkte für planare Auswertung
                continue

            if not self._surface_has_numeric_z(points):
                continue

            model, _ = build_planar_model(points)
            if model is None:
                continue

            for point in points:
                point["z"] = evaluate_surface_plane(
                    model,
                    float(point.get("x", 0.0)),
                    float(point.get("y", 0.0)),
                )
            adjusted = True
            if isinstance(surface, SurfaceDefinition):
                surface.points = points
                surface.plane_model = model
            else:
                surface["points"] = points
                surface["plane_model"] = model

        return adjusted

    @staticmethod
    def _surface_has_numeric_z(points) -> bool:
        for point in points:
            z_val = point.get("z")
            numeric = UISurfaceManager._safe_float(z_val)
            if numeric is None:
                return False
        return True

    @staticmethod
    def _safe_float(value) -> Optional[float]:
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

    # ---- docking helpers -------------------------------------------

    def _dock_surface_initial(self):
        if self.surface_dock_widget is None or self._initial_docked:
            return

        sources_dock = None
        if hasattr(self.main_window, "sources_instance") and self.main_window.sources_instance:
            sources_dock = getattr(self.main_window.sources_instance, "sources_dockWidget", None)

        if sources_dock:
            snapshot_dock = (
                self.main_window.snapshot_engine.capture_dock_widget
                if hasattr(self.main_window, "snapshot_engine") and self.main_window.snapshot_engine
                else None
            )
            primary_dock = None

            if snapshot_dock:
                try:
                    if self.main_window.dockWidgetArea(snapshot_dock) == Qt.NoDockWidgetArea:
                        self.main_window.addDockWidget(Qt.RightDockWidgetArea, snapshot_dock)
                    primary_dock = snapshot_dock
                except Exception:
                    snapshot_dock = None

            try:
                self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.surface_dock_widget)
            except Exception:
                pass

            if snapshot_dock:
                try:
                    self.main_window.tabifyDockWidget(snapshot_dock, self.surface_dock_widget)
                    self.surface_dock_widget.raise_()
                    primary_dock = snapshot_dock
                except Exception:
                    primary_dock = self.surface_dock_widget
            else:
                primary_dock = self.surface_dock_widget

            if primary_dock:
                try:
                    self.main_window.splitDockWidget(sources_dock, primary_dock, Qt.Horizontal)
                    self.main_window.resizeDocks([sources_dock, primary_dock], [360, 280], Qt.Horizontal)
                except Exception:
                    pass
        else:
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.surface_dock_widget)

        self._initial_docked = True

    def _align_with_snapshot_if_possible(self):
        if (
            self.surface_dock_widget is None
            or self.snapshot_dock is None
            or self._snapshot_split_done
        ):
            return

        if not self.snapshot_dock.isVisible():
            return

        try:
            if self.main_window.dockWidgetArea(self.snapshot_dock) == Qt.NoDockWidgetArea:
                self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.snapshot_dock)

            if self.main_window.dockWidgetArea(self.surface_dock_widget) == Qt.NoDockWidgetArea:
                self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.surface_dock_widget)

            self.main_window.tabifyDockWidget(self.snapshot_dock, self.surface_dock_widget)
            self.surface_dock_widget.raise_()

            sources_dock = None
            if hasattr(self.main_window, "sources_instance") and self.main_window.sources_instance:
                sources_dock = getattr(self.main_window.sources_instance, "sources_dockWidget", None)

            if sources_dock:
                self.main_window.splitDockWidget(sources_dock, self.snapshot_dock, Qt.Horizontal)
                self.main_window.resizeDocks([sources_dock, self.snapshot_dock], [360, 280], Qt.Horizontal)

            self._snapshot_split_done = True
        except Exception:
            pass


class _SurfaceGroupController:
    def __init__(self, settings):
        self.settings = settings

    @property
    def root_group_id(self) -> str:
        return getattr(self.settings, "ROOT_SURFACE_GROUP_ID", "surface_group_root")

    def ensure_structure(self):
        store = self._ensure_group_store()
        root = self._ensure_group_object(self.root_group_id)
        if root is None:
            store[self.root_group_id] = SurfaceGroup(
                group_id=self.root_group_id,
                name="Surfaces",
                enabled=True,
                hidden=False,
                parent_id=None,
                locked=True,
            )

        surface_store = getattr(self.settings, "surface_definitions", {})
        for group_id in list(store.keys()):
            group = self._ensure_group_object(group_id)
            group.child_groups = [
                gid for gid in group.child_groups if gid in store and gid != group_id
            ]
            group.surface_ids = [
                sid for sid in group.surface_ids if sid in surface_store
            ]

        for surface_id in surface_store.keys():
            surface = self._ensure_surface_object(surface_id)
            target_group_id = surface.group_id or self.root_group_id
            self.assign_surface_to_group(surface_id, target_group_id, create_missing=True)

    def reset_storage(self):
        init_surfaces = getattr(self.settings, "_initialize_surface_definitions", None)
        if callable(init_surfaces):
            self.settings.surface_definitions = init_surfaces()
        else:
            self.settings.surface_definitions = {}
        self.settings.surface_groups = self._create_default_group_store()
        self.ensure_structure()

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

        if not target_group_id:
            target_group_id = self.root_group_id

        group = self._ensure_group_object(target_group_id)
        if group is None and create_missing:
            group = self.create_surface_group(target_group_id, parent_id=self.root_group_id, group_id=target_group_id)
        elif group is None:
            target_group_id = self.root_group_id
            group = self._ensure_group_object(target_group_id)

        previous_group_id = surface.group_id or self._find_surface_group_id(surface_id)
        if previous_group_id and previous_group_id != target_group_id:
            self._remove_surface_from_group(surface_id, previous_group_id)

        if group and surface_id not in group.surface_ids:
            group.surface_ids.append(surface_id)
        surface.group_id = target_group_id

    def create_surface_group(self, name: str, parent_id: Optional[str] = None, *, group_id: Optional[str] = None, locked: bool = False) -> SurfaceGroup:
        parent_id = parent_id or self.root_group_id
        parent = self._ensure_group_object(parent_id)
        if parent is None:
            parent_id = self.root_group_id
            parent = self._ensure_group_object(parent_id)

        if group_id is None:
            group_id = self._generate_surface_group_id()

        store = self._ensure_group_store()
        if group_id in store:
            return self._ensure_group_object(group_id)

        group = SurfaceGroup(
            group_id=group_id,
            name=name or group_id,
            enabled=parent.enabled if parent else True,
            hidden=parent.hidden if parent else False,
            parent_id=parent_id,
            locked=locked,
        )
        store[group_id] = group

        if parent and group_id not in parent.child_groups:
            parent.child_groups.append(group_id)

        return group

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

    def move_surface_group(self, group_id: str, target_parent_id: str) -> bool:
        group = self._ensure_group_object(group_id)
        if not group or group.locked:
            return False
        if group_id == target_parent_id:
            return False

        ancestor_id = target_parent_id
        while ancestor_id:
            if ancestor_id == group_id:
                return False
            ancestor = self._ensure_group_object(ancestor_id)
            if ancestor is None:
                break
            ancestor_id = ancestor.parent_id

        old_parent = self._ensure_group_object(group.parent_id) if group.parent_id else None
        if old_parent and group_id in old_parent.child_groups:
            old_parent.child_groups.remove(group_id)

        target_parent = self._ensure_group_object(target_parent_id) or self._ensure_group_object(self.root_group_id)
        if target_parent and group_id not in target_parent.child_groups:
            target_parent.child_groups.append(group_id)

        group.parent_id = target_parent.group_id if target_parent else self.root_group_id
        return True

    def rename_surface_group(self, group_id: str, new_name: str):
        group = self._ensure_group_object(group_id)
        if not group:
            return
        name = (new_name or "").strip()
        if name:
            group.name = name

    def find_surface_group_by_name(self, name: str, parent_id: Optional[str] = None) -> Optional[SurfaceGroup]:
        target_name = (name or "").strip()
        if not target_name:
            return None
        for group_id in list(self._ensure_group_store().keys()):
            group = self._ensure_group_object(group_id)
            if not group:
                continue
            if group.name == target_name and (parent_id is None or group.parent_id == parent_id):
                return group
        return None

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

    def ensure_group_path(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        key = label.strip()
        if not key:
            return None
        segments = [part.strip() for part in key.replace("\\", "/").split("/") if part.strip()]
        parent_id = self.root_group_id
        current_group_id = parent_id
        for segment in segments:
            existing = self.find_surface_group_by_name(segment, parent_id=current_group_id)
            if existing:
                current_group_id = existing.group_id
            else:
                created = self.create_surface_group(segment, parent_id=current_group_id)
                current_group_id = created.group_id
        return current_group_id

    def detach_surface(self, surface_id: str):
        group_id = self._find_surface_group_id(surface_id)
        if group_id:
            self._remove_surface_from_group(surface_id, group_id)

    def _ensure_group_store(self) -> Dict[str, SurfaceGroup]:
        store = getattr(self.settings, "surface_groups", None)
        if not isinstance(store, dict) or not store:
            store = self._create_default_group_store()
        self.settings.surface_groups = store
        return store

    def _create_default_group_store(self) -> Dict[str, SurfaceGroup]:
        return {
            self.root_group_id: SurfaceGroup(
                group_id=self.root_group_id,
                name="Surfaces",
                enabled=True,
                hidden=False,
                parent_id=None,
                locked=True,
            )
        }

    def _ensure_group_object(self, group_id: Optional[str]) -> Optional[SurfaceGroup]:
        if not group_id:
            return None
        store = self._ensure_group_store()
        group = store.get(group_id)
        if group is None:
            return None
        if isinstance(group, SurfaceGroup):
            return group
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


