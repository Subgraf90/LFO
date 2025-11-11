from PyQt5.QtCore import Qt


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
        """
        if self.surface_dock_widget is None:
            from Module_LFO.Modules_Window.WindowSurfaceWidget import SurfaceDockWidget

            self.surface_dock_widget = SurfaceDockWidget(self.main_window, self.settings, self.container)
            self._dock_surface_initial()

        self.surface_dock_widget.initialize()
        self.surface_dock_widget.show()
        self._align_with_snapshot_if_possible()

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

        self.surface_dock_widget.load_surfaces()

    def _disconnect_sources_signals(self):
        if self.sources_widget and hasattr(self.sources_widget, "sources_tree_widget"):
            try:
                self.sources_widget.sources_tree_widget.itemSelectionChanged.disconnect(self._on_sources_selection_changed)
            except (RuntimeError, TypeError):
                pass

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


