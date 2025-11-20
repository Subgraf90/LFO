from PyQt5.QtCore import Qt

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
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

    # ---- public API -------------------------------------------------

    def show_surface_dock_widget(self):
        """
        Erstellt (falls nötig) und zeigt das SurfaceDockWidget an.
        Positioniert es rechts neben dem Hauptfenster als schwebendes Fenster.
        """
        if self.surface_dock_widget is None:
            from Module_LFO.Modules_Window.WindowSurfaceWidget import SurfaceDockWidget

            self.surface_dock_widget = SurfaceDockWidget(self.main_window, self.settings, self.container)
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
        for surface in surface_store.values():
            points = surface.get("points") or []
            if len(points) < 4:  # Mindestens vier Punkte für planare Auswertung
                continue

            model, needs_adjustment = build_planar_model(points)
            if model is None or not needs_adjustment:
                continue

            for point in points:
                point["z"] = evaluate_surface_plane(
                    model,
                    float(point.get("x", 0.0)),
                    float(point.get("y", 0.0)),
                )
            adjusted = True

        return adjusted

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


