from PyQt5.QtWidgets import QComboBox, QDockWidget, QWidget, QVBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem, QCheckBox, QPushButton, QHBoxLayout, QTabWidget, QSizePolicy, QGridLayout, QLabel, QFrame, QSpacerItem, QLineEdit, QMenu, QAbstractItemView, QGroupBox
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QDragEnterEvent, QDropEvent

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QCheckBox, QFrame, QGridLayout, QTreeWidgetItem, QComboBox, QDoubleSpinBox, QTabWidget)
from PyQt5.QtCore import Qt

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase

from Module_LFO.Modules_Calculate.WindowingCalculator import WindowingCalculator
from Module_LFO.Modules_Calculate.BeamSteering import BeamSteering
from Module_LFO.Modules_Plot.PlotBeamsteering import BeamsteeringPlot    
from Module_LFO.Modules_Plot.PlotWindowing import WindowingPlot

import time
import functools

# def measure_time(func):
#     """
#     Decorator zur Messung der Ausführungszeit einer Methode.
#     Gibt den Methodennamen und die Ausführungszeit in Millisekunden aus.
#     """
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         execution_time = (end_time - start_time) * 1000  # Umrechnung in Millisekunden
#         # print(f"DEBUG: {func.__name__} ausgeführt in {execution_time:.2f} ms")
#         return result
#     return wrapper




# from Module_LFO.Modules_Ui.Speakerspecs import Speakerspecs


class Sources(ModuleBase, QObject):
    def __init__(self, main_window, settings, container):
        ModuleBase.__init__(self, settings)
        QObject.__init__(self)
        self.main_window = main_window
        self.settings = settings
        self.calculation = {}
        self.container = container    
        self.speakerspecs_instance = []
        
        self.tab_widget = QTabWidget()
        self.beamsteering_plot = None
        self.create_beamsteering_tab()
        self.windowing_plot = None
        self.create_windowing_tab()
        
        # Debug-Schalter für Signalhandler
        self.debug_signals = False
        
        # Initialisiere speaker_tab_layout als leeres Layout
        self.speaker_tab_layout = QHBoxLayout()

    # ---- Helper methods for updates ----------------------------------
    
    def is_autocalc_active(self):
        """Prüft, ob automatische Berechnung aktiv ist."""
        return getattr(self.settings, "update_pressure_soundfield", True)
    
    def update_speaker_overlays(self):
        """Aktualisiert nur die Lautsprecher-Overlays im Plot, ohne Neuberechnung."""
        if (hasattr(self.main_window, 'draw_plots') and 
            hasattr(self.main_window.draw_plots, 'draw_spl_plotter') and
            hasattr(self.main_window.draw_plots.draw_spl_plotter, 'update_overlays')):
            self.main_window.draw_plots.draw_spl_plotter.update_overlays(
                self.settings, self.container
            )

    # ---- Debug helpers ----------------------------------------------

    def _debug_layout_snapshot(self, reason: str) -> None:
        """
        Gibt eine Momentaufnahme der aktuellen Layout-Geometrien aus.
        """
        try:
            print(f"[LFO][DEBUG][UiSourceManagement] Layout snapshot: {reason}")

            mw = getattr(self, "main_window", None)
            if mw:
                geom = mw.geometry()
                print(
                    "  MainWindow:",
                    f"size={geom.width()}x{geom.height()}",
                    f"pos=({geom.x()},{geom.y()})",
                )
                if mw.centralWidget():
                    csize = mw.centralWidget().size()
                    print(
                        "  CentralWidget:",
                        f"size={csize.width()}x{csize.height()}",
                    )

            # Sources-Dock
            sources_dock = getattr(self, "sources_dockWidget", None)
            if sources_dock:
                size = sources_dock.size()
                print(
                    "  SourcesDock:",
                    f"visible={sources_dock.isVisible()}",
                    f"floating={sources_dock.isFloating()}",
                    f"size={size.width()}x{size.height()}",
                )

            # Surface-Dock
            surface_manager = getattr(self.main_window, "surface_manager", None) if mw else None
            surface_dock = getattr(surface_manager, "surface_dock_widget", None) if surface_manager else None
            if surface_dock:
                size = surface_dock.size()
                print(
                    "  SurfaceDock:",
                    f"visible={surface_dock.isVisible()}",
                    f"floating={surface_dock.isFloating()}",
                    f"size={size.width()}x{size.height()}",
                )
                try:
                    splitter_sizes = getattr(surface_dock, "splitter", None).sizes()
                    print("    SurfaceDock splitter sizes:", splitter_sizes)
                except Exception:
                    pass

            # Source-Layout-Dock
            draw_widgets = getattr(mw, "draw_widgets", None) if mw else None
            source_layout_dock = getattr(draw_widgets, "source_layout_widget", None) if draw_widgets else None
            if source_layout_dock:
                size = source_layout_dock.size()
                print(
                    "  SourceLayoutDock:",
                    f"visible={source_layout_dock.isVisible()}",
                    f"floating={source_layout_dock.isFloating()}",
                    f"size={size.width()}x{size.height()}",
                )

        except Exception as exc:
            print(f"[LFO][DEBUG][UiSourceManagement] snapshot failed: {exc}")


    def _debug_signal(self, name, extra=""):
        if not getattr(self, "debug_signals", False):
            return
        sender = self.sender()
        sender_name = ""
        if sender:
            sender_name = sender.objectName() or sender.__class__.__name__
        focus_widget = QtWidgets.QApplication.focusWidget()
        focus_name = ""
        if focus_widget:
            focus_name = focus_widget.objectName() or focus_widget.__class__.__name__
        print(f"[LFO][DEBUG][UiSourceManagement] {name} (sender={sender_name}, focus={focus_name}) {extra}")

        
    def closeEvent(self, event):
        if self.sources_dockWidget:
            self.sources_dockWidget.close()
    
        event.accept()
        super().closeEvent(event)
        
    # @measure_time
    def init_ui(self, instance, layout):
        speaker_array = self.settings.get_speaker_array(instance['id'])
        
        instance['scroll_area'] = QScrollArea(self.main_window)
        instance['scroll_area'].setWidgetResizable(True)
        instance['scroll_area'].setMaximumHeight(190)  # Maximale Höhe für kompaktes Rechteck
        instance['scroll_area'].setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Horizontale Scrollbar bei Bedarf
        instance['scroll_content'] = QWidget()
        instance['scroll_layout'] = QVBoxLayout(instance['scroll_content'])
        # Mehr Außenabstand für bessere Lesbarkeit
        instance['scroll_layout'].setContentsMargins(5, 3, 5, 3)
        instance['scroll_layout'].setSpacing(0)
        
        # Erstelle ein Container-Widget für das Grid-Layout
        grid_container = QWidget()
        instance['gridLayout_sources'] = QGridLayout(grid_container)
        # Etwas Abstand nach außen, aber kompakt zwischen Elementen
        instance['gridLayout_sources'].setContentsMargins(5, 5, 5, 5)
        
        # Wichtig: Setze die vertikale Ausrichtung auf oben
        instance['gridLayout_sources'].setAlignment(Qt.AlignTop)
        
        # Setze minimale Abstände zwischen den Elementen - sehr kompakt in Höhe, normale Breite
        instance['gridLayout_sources'].setVerticalSpacing(0)
        instance['gridLayout_sources'].setHorizontalSpacing(5)
        
        # Füge das Grid-Container-Widget zum Scroll-Layout hinzu
        instance['scroll_layout'].addWidget(grid_container)
        
        # Füge einen Stretch-Faktor hinzu, damit alles nach oben gedrückt wird
        instance['scroll_layout'].addStretch(1)
        
        instance['scroll_area'].setWidget(instance['scroll_content'])
        layout.addWidget(instance['scroll_area'])
        
        if speaker_array and hasattr(speaker_array, 'source_polar_pattern'):
            pass
        else:
            if speaker_array and len(self.container.data['speaker_names']) > 0:
                default_speaker = self.container.data['speaker_names'][0]
                speaker_array.source_polar_pattern = np.array([default_speaker] * speaker_array.number_of_sources, dtype=object)


    def get_speakerspecs_instance(self, speaker_array_id):
        for instance in self.speakerspecs_instance:
            if instance['id'] == speaker_array_id:
                return instance
        return None

        
    def add_speakerspecs_instance(self, speaker_array_id, instance):
        self.speakerspecs_instance.append(instance)


    def create_speakerspecs_instance(self, array_id):
        return {
            'id': array_id,
            'gridLayout_sources': None,
            'scroll_area': None,
            'state': None,
            'saved_state': {}
        }



    # ---- Dock Widget ---

    def show_sources_dock_widget(self):
        # Blockiere Signale am Anfang
        if hasattr(self, 'sources_tree_widget'):
            self.sources_tree_widget.blockSignals(True)
            
        if not hasattr(self, 'sources_dockWidget') or self.sources_dockWidget is None:
            self.sources_dockWidget = QDockWidget("Sources", self.main_window)
            
            # Event-Filter für resize-Events hinzufügen
            self.sources_dockWidget.installEventFilter(self)
            
            # Prüfe, ob Surface-DockWidget bereits existiert und tabbifiziere es
            surface_dock = None
            if hasattr(self.main_window, 'surface_manager') and self.main_window.surface_manager:
                surface_dock = getattr(self.main_window.surface_manager, 'surface_dockWidget', None)
            
            if surface_dock:
                # Tabbifiziere: Beide DockWidgets übereinander mit Tab-Buttons
                self.main_window.addDockWidget(Qt.BottomDockWidgetArea, self.sources_dockWidget)
                self.main_window.tabifyDockWidget(surface_dock, self.sources_dockWidget)
                # Aktiviere das Sources-DockWidget (wird als aktiver Tab angezeigt)
                self.sources_dockWidget.raise_()
            else:
                # Füge normal hinzu
                self.main_window.addDockWidget(Qt.BottomDockWidgetArea, self.sources_dockWidget)
    
            dock_content = QWidget()
            dock_layout = QVBoxLayout(dock_content)
            self.sources_dockWidget.setWidget(dock_content)
    
            splitter = QSplitter(Qt.Horizontal)
            dock_layout.addWidget(splitter)
    
            # Linke Seite mit TreeWidget und Buttons
            left_side_widget = QWidget()
            left_side_layout = QVBoxLayout(left_side_widget)
            left_side_layout.setContentsMargins(0, 0, 0, 0)  # Keine Ränder
            splitter.addWidget(left_side_widget)
    
            # TreeWidget für die Arrays
            self.sources_tree_widget = QTreeWidget()
            self.sources_tree_widget.setHeaderLabels(["Source Name", "M", "H", ""])
            
            # Überschreibe dropEvent für Drag & Drop Funktionalität
            original_dropEvent = self.sources_tree_widget.dropEvent
            original_startDrag = self.sources_tree_widget.startDrag
            _pending_drag_items = []
            
            def custom_startDrag(supported_actions):
                """Speichert die ausgewählten Items vor dem Drag"""
                _pending_drag_items.clear()
                _pending_drag_items.extend(self.sources_tree_widget.selectedItems())
                original_startDrag(supported_actions)
            
            def custom_dropEvent(event):
                """Behandelt Drop-Events für Drag & Drop mit Gruppen"""
                drop_item = self.sources_tree_widget.itemAt(event.pos())
                indicator_pos = self.sources_tree_widget.dropIndicatorPosition()
                
                # Nur Source Items können verschoben werden (keine Gruppen)
                dragged_items = [item for item in _pending_drag_items 
                                if item.data(0, Qt.UserRole + 1) != "group"]
                
                if not dragged_items:
                    event.ignore()
                    _pending_drag_items.clear()
                    return
                
                # Prüfe, ob auf eine Gruppe gedroppt wird
                if drop_item and drop_item.data(0, Qt.UserRole + 1) == "group":
                    # Droppe auf Gruppe - füge Items als Childs hinzu
                    for item in dragged_items:
                        # Entferne Item von altem Parent
                        old_parent = item.parent()
                        if old_parent:
                            old_parent.removeChild(item)
                        else:
                            index = self.sources_tree_widget.indexOfTopLevelItem(item)
                            if index != -1:
                                self.sources_tree_widget.takeTopLevelItem(index)
                        
                        # Füge Item zur Gruppe hinzu
                        drop_item.addChild(item)
                        drop_item.setExpanded(True)
                        
                        # Stelle sicher, dass Checkboxen für das Item existieren
                        self.ensure_source_checkboxes(item)
                        
                        # Speichere ursprüngliche Positionen für die Gruppe
                        self.save_original_positions_for_group(drop_item)
                    
                    # Verbinde Checkboxen der Gruppe neu
                    mute_checkbox = self.sources_tree_widget.itemWidget(drop_item, 1)
                    hide_checkbox = self.sources_tree_widget.itemWidget(drop_item, 2)
                    if mute_checkbox:
                        try:
                            mute_checkbox.stateChanged.disconnect()
                        except:
                            pass
                        mute_checkbox.stateChanged.connect(lambda state, g_item=drop_item: self.on_group_mute_changed(g_item, state))
                    if hide_checkbox:
                        try:
                            hide_checkbox.stateChanged.disconnect()
                        except:
                            pass
                        hide_checkbox.stateChanged.connect(lambda state, g_item=drop_item: self.on_group_hide_changed(g_item, state))
                    
                    event.accept()
                    event.setDropAction(Qt.MoveAction)
                elif indicator_pos == QAbstractItemView.OnItem and drop_item:
                    # Droppe auf ein Source Item
                    # Prüfe, ob das Ziel-Item in einer Gruppe ist
                    parent = drop_item.parent()
                    if parent and parent.data(0, Qt.UserRole + 1) == "group":
                        # Ziel-Item ist bereits in einer Gruppe - füge zur bestehenden Gruppe hinzu
                        for item in dragged_items:
                            old_parent = item.parent()
                            if old_parent:
                                old_parent.removeChild(item)
                            else:
                                index = self.sources_tree_widget.indexOfTopLevelItem(item)
                                if index != -1:
                                    self.sources_tree_widget.takeTopLevelItem(index)
                            parent.addChild(item)
                            # Stelle sicher, dass Checkboxen für das Item existieren
                            self.ensure_source_checkboxes(item)
                            
                            # Speichere ursprüngliche Positionen für die Gruppe
                            self.save_original_positions_for_group(parent)
                        event.accept()
                        event.setDropAction(Qt.MoveAction)
                    else:
                        # Droppe auf ein Source Item, das nicht in einer Gruppe ist
                        # Source Items dürfen nicht auf andere Source Items gezogen werden
                        event.ignore()
                else:
                    # Standard Drag & Drop Verhalten (zwischen Items)
                    original_dropEvent(event)
                
                # Validiere alle Checkboxen nach Drag & Drop
                self.validate_all_checkboxes()
                
                # Passe Spaltenbreite an den Inhalt an
                self.adjust_column_width_to_content()
                
                _pending_drag_items.clear()
            
            # Überschreibe die Methoden
            self.sources_tree_widget.startDrag = custom_startDrag
            self.sources_tree_widget.dropEvent = custom_dropEvent
            # Schriftgröße im TreeWidget auf 11pt setzen
            tree_font = QtGui.QFont()
            tree_font.setPointSize(11)
            self.sources_tree_widget.setFont(tree_font)
            # Source Name linksbündig ausrichten
            header = self.sources_tree_widget.header()
            header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            # Spaltenbreiten konfigurieren - kompakte Darstellung (angepasst für kleinere Checkboxen)
            self.sources_tree_widget.setColumnWidth(0, 160)  # Name - 10% schlanker (von 180px auf 160px)
            self.sources_tree_widget.setColumnWidth(1, 25)   # Mute - kleiner für 18x18 Checkboxen
            self.sources_tree_widget.setColumnWidth(2, 25)   # Hide - kleiner für 18x18 Checkboxen
            self.sources_tree_widget.setColumnWidth(3, 25)   # Farb-Quadrat (gleich wie Surface UI Spalte 3)
            header.setStretchLastSection(False)  # Letzte Spalte nicht strecken
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)  # Name kann angepasst werden
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)    # Mute bleibt fix
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)    # Hide bleibt fix
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)    # Color bleibt fix
            
            # Scrollbar aktivieren, wenn Inhalt breiter als Widget
            self.sources_tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Indentation für Gruppen aktivieren
            self.sources_tree_widget.setIndentation(15)
            
            # Drag & Drop aktivieren
            self.sources_tree_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
            self.sources_tree_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.sources_tree_widget.setDragEnabled(True)
            self.sources_tree_widget.setAcceptDrops(True)
            self.sources_tree_widget.setDropIndicatorShown(True)
            
            self.sources_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.sources_tree_widget.customContextMenuRequested.connect(self.show_context_menu)
            
            # Setze die SizePolicy, damit das TreeWidget vertikal expandiert
            self.sources_tree_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.sources_tree_widget.setMinimumHeight(100)  # Reduziert für kompakte UI (140px DockWidget - 24px Buttons - ~15px Margins)
            self.sources_tree_widget.setFixedWidth(270)  # 10% schlanker (von 300px auf 270px)
            
            # OPTIMIERUNG: Nur EIN Signal verbinden - zentraler Handler koordiniert alle Updates
            # Dies verhindert 9x redundante Ausführung bei jeder Auswahländerung
            self.sources_tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
            self.sources_tree_widget.itemChanged.connect(self.on_speakerspecs_item_text_changed)
            
            # Installiere Event-Filter für MousePressEvent, um Klicks auf leeres Feld zu erkennen
            self.sources_tree_widget.viewport().installEventFilter(self)
            self._sources_tree_mouse_press_pos = None
    
            # Füge das TreeWidget zum Layout hinzu mit Stretch-Faktor
            left_side_layout.addWidget(self.sources_tree_widget, 1)  # Stretch-Faktor 1
    
            # Buttons zum Hinzufügen und Löschen von Arrays
            buttons_layout = QHBoxLayout()
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.setSpacing(5)
            buttons_layout.setAlignment(Qt.AlignLeft)
            self.pushbutton_add_stack = QPushButton("Stack")
            self.pushbutton_add_stack.setFixedWidth(120)  # Gleiche Breite wie Surface UI Buttons
            self.pushbutton_add_flown = QPushButton("Flown")
            self.pushbutton_add_flown.setFixedWidth(120)  # Gleiche Breite wie Surface UI Buttons
            self.pushbutton_add_flown.setEnabled(True)  # Aktiviere den Flown-Button
            # Buttons: Schriftgröße und Höhe
            btn_font = QtGui.QFont()
            btn_font.setPointSize(11)
            self.pushbutton_add_stack.setFont(btn_font)
            self.pushbutton_add_flown.setFont(btn_font)
            self.pushbutton_add_stack.setFixedHeight(24)
            self.pushbutton_add_flown.setFixedHeight(24)
            buttons_layout.addWidget(self.pushbutton_add_stack)
            buttons_layout.addWidget(self.pushbutton_add_flown)
            buttons_layout.addStretch(1)
    
            left_side_layout.addLayout(buttons_layout)
    
            self.pushbutton_add_stack.clicked.connect(self.add_stack)
            self.pushbutton_add_flown.clicked.connect(self.add_flown)
            
            # Rechte Seite für TabWidget
            self.right_side_widget = QWidget()
            splitter.addWidget(self.right_side_widget)
            
            # Erstelle das TabWidget
            self.tab_widget = QTabWidget()
            self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.tab_widget.setMinimumWidth(800)  # Setze die Mindestbreite des Tab-Widgets

            # Erstelle die Tabs
            self.create_speaker_tab_stack()  # Wichtig: Hier wird der Speaker Tab erstellt
            self.create_beamsteering_tab()
            self.create_windowing_tab()   
            
            # Füge TabWidget zum rechten Widget hinzu
            right_layout = QVBoxLayout(self.right_side_widget)
            right_layout.addWidget(self.tab_widget)
            
            # Lade bestehende Arrays
            all_speaker_arrays = list(self.settings.get_all_speaker_array_ids())
            if all_speaker_arrays:
                self.initialize_speakerspecs_instances(all_speaker_arrays)
            
            # Rufe show_sources_tab auf, um die Tabs zu konfigurieren
            self.show_sources_tab()
            
        # Entsperre Signale am Ende
        if hasattr(self, 'sources_tree_widget'):
            self.sources_tree_widget.blockSignals(False)
            
        self.sources_tree_widget.clearSelection()
        
        # Setze Mindesthöhe des DockWidgets, damit Qt's Layout-System die Größe respektiert
        target_height = 140
        self.sources_dockWidget.setMinimumHeight(target_height)
        self.sources_dockWidget.setMaximumHeight(target_height)  # Temporär fixieren, damit Qt nicht überschreibt
        
        self.sources_dockWidget.show()
        
        # Setze initiale Größe des DockWidgets NACH show() mit Timer, damit Qt's Layout fertig ist
        from PyQt5.QtCore import QTimer
        def apply_resize():
            self.sources_dockWidget.setMaximumHeight(16777215)  # Entferne Max-Höhe wieder
            self.sources_dockWidget.resize(1200, target_height)
        
        QTimer.singleShot(100, apply_resize)  # 100ms Verzögerung, damit Qt's Layout fertig ist
        

    def refresh_active_selection(self):
        """
        Aktualisiert alle UI-Elemente für das aktuell ausgewählte Array einmalig.
        """
        if not hasattr(self, 'sources_tree_widget') or self.sources_tree_widget is None:
            return

        selected_item = self.sources_tree_widget.currentItem()
        if selected_item is None:
            return

        speaker_array_id = selected_item.data(0, Qt.UserRole)

        instance = self.get_speakerspecs_instance(speaker_array_id)
        if instance and instance.get('gridLayout_sources') is None:
            self.init_ui(instance, self.speaker_tab_layout)

        self.display_selected_speakerspecs()
        if instance:
            self.update_input_fields(instance)
        self.update_sources_input_fields()
        self.update_source_length_input_fields()
        self.update_array_position_input_fields()
        self.update_beamsteering_input_fields()
        self.update_windowing_input_fields()
        self.update_gain_delay_input_fields()
        self.update_beamsteering_windowing_plots()
    


    # ----- Tab Widget -----

    def create_speaker_tab_stack(self, is_stack=True):
        """Erstellt den Speaker Setup Tab mit allen Eingabefeldern"""
        speaker_tab = QWidget()
        self.tab_widget.addTab(speaker_tab, "Speaker Setup")
        
        self.speaker_tab_layout = QHBoxLayout(speaker_tab)
        speaker_grid_layout = QGridLayout()
        
        # Reduziere die vertikalen Abstände für kompakteres Layout
        speaker_grid_layout.setVerticalSpacing(3)
        speaker_grid_layout.setContentsMargins(5, 5, 5, 5)
        
        # Schriftgröße für alle Widgets
        font = QtGui.QFont()
        font.setPointSize(11)
        
        # Anzahl der Lautsprecher
        number_of_sources_label = QLabel("# speakers")
        number_of_sources_label.setFont(font)
        self.number_of_sources_edit = QLineEdit()
        self.number_of_sources_edit.setFont(font)
        self.number_of_sources_edit.setFixedHeight(18)
        self.number_of_sources_edit.setValidator(QIntValidator(1, 100))
        self.number_of_sources_edit.setText("1")
        self.number_of_sources_edit.editingFinished.connect(self.on_number_of_sources_changed)
        
        number_of_sources_label.setFixedWidth(150)
        self.number_of_sources_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(number_of_sources_label, 0, 0)
        speaker_grid_layout.addWidget(self.number_of_sources_edit, 0, 1)
        
        # Trennlinie
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setFixedHeight(10)
        speaker_grid_layout.addWidget(line1, 1, 0, 1, 2)
        
        # Array-Länge
        array_length_label = QLabel("Array length")
        array_length_label.setFont(font)
        self.array_length_edit = QLineEdit()
        self.array_length_edit.setFont(font)
        self.array_length_edit.setFixedHeight(18)
        self.array_length_edit.setValidator(QDoubleValidator(0, 1000, 1))
        self.array_length_edit.setText("0")
        self.array_length_edit.editingFinished.connect(self.on_SourceLength_changed)
        array_length_label.setFixedWidth(150)
        self.array_length_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_length_label, 2, 0)
        speaker_grid_layout.addWidget(self.array_length_edit, 2, 1)
        
        # Trennlinie
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        line2.setFixedHeight(10)
        speaker_grid_layout.addWidget(line2, 3, 0, 1, 2)
        
        # Array X-Position
        array_x_label = QLabel("Array X (m)")
        array_x_label.setFont(font)
        self.array_x_edit = QLineEdit()
        self.array_x_edit.setFont(font)
        self.array_x_edit.setFixedHeight(18)
        self.array_x_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_x_edit.setText("0.00")
        self.array_x_edit.editingFinished.connect(self.on_ArrayX_changed)
        array_x_label.setFixedWidth(150)
        self.array_x_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_x_label, 4, 0)
        speaker_grid_layout.addWidget(self.array_x_edit, 4, 1)
        
        # Array Y-Position
        array_y_label = QLabel("Array Y (m)")
        array_y_label.setFont(font)
        self.array_y_edit = QLineEdit()
        self.array_y_edit.setFont(font)
        self.array_y_edit.setFixedHeight(18)
        self.array_y_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_y_edit.setText("0.00")
        self.array_y_edit.editingFinished.connect(self.on_ArrayY_changed)
        array_y_label.setFixedWidth(150)
        self.array_y_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_y_label, 5, 0)
        speaker_grid_layout.addWidget(self.array_y_edit, 5, 1)
        
        # Array Z-Position
        array_z_label = QLabel("Array Z (m)")
        array_z_label.setFont(font)
        self.array_z_edit = QLineEdit()
        self.array_z_edit.setFont(font)
        self.array_z_edit.setFixedHeight(18)
        self.array_z_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_z_edit.setText("0.00")
        self.array_z_edit.editingFinished.connect(self.on_ArrayZ_changed)
        array_z_label.setFixedWidth(150)
        self.array_z_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_z_label, 6, 0)
        speaker_grid_layout.addWidget(self.array_z_edit, 6, 1)
        
        # Trennlinie
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setFrameShadow(QFrame.Sunken)
        line3.setFixedHeight(10)
        speaker_grid_layout.addWidget(line3, 7, 0, 1, 2)
        
        # Delay
        delay_label = QLabel("Delay (ms)")
        delay_label.setFont(font)
        self.delay_edit = QLineEdit()
        self.delay_edit.setFont(font)
        self.delay_edit.setFixedHeight(18)
        self.delay_edit.setValidator(QDoubleValidator(0, 1000, 2))
        self.delay_edit.setText("0")
        self.delay_edit.editingFinished.connect(self.on_Delay_changed)
        delay_label.setFixedWidth(150)
        self.delay_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(delay_label, 8, 0)
        speaker_grid_layout.addWidget(self.delay_edit, 8, 1)
        
        # Gain
        gain_label = QLabel("Gain (dB)")
        gain_label.setFont(font)
        self.gain_edit = QLineEdit()
        self.gain_edit.setFont(font)
        self.gain_edit.setFixedHeight(18)
        self.gain_edit.setValidator(QDoubleValidator(-60, 10, 1))
        self.gain_edit.setText("0")
        self.gain_edit.editingFinished.connect(self.on_Gain_changed)
        gain_label.setFixedWidth(150)
        self.gain_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(gain_label, 9, 0)
        speaker_grid_layout.addWidget(self.gain_edit, 9, 1)
        
        # Spacer zwischen Gain und Polarity für größeren Abstand
        self.spacer_gain_polarity = QLabel("")
        self.spacer_gain_polarity.setFixedHeight(5)
        speaker_grid_layout.addWidget(self.spacer_gain_polarity, 10, 0)
        
        # Polarity inverted
        polarity_label = QLabel("Polarity inverted")
        polarity_label.setFont(font)
        self.polarity_checkbox = QCheckBox()
        self.polarity_checkbox.setChecked(False)
        self.polarity_checkbox.stateChanged.connect(self.on_Polarity_changed)
        polarity_label.setFixedWidth(150)
        speaker_grid_layout.addWidget(polarity_label, 11, 0)
        speaker_grid_layout.addWidget(self.polarity_checkbox, 11, 1)
        
        # Spacer zwischen Polarity und Symmetric für größeren Abstand
        self.spacer_label = QLabel("")
        self.spacer_label.setFixedHeight(3)
        speaker_grid_layout.addWidget(self.spacer_label, 12, 0)
        
        # Symmetric Checkbox
        self.symmetric_label = QLabel("Symmetric")
        self.symmetric_label.setFont(font)
        self.symmetric_checkbox = QCheckBox()
        self.symmetric_checkbox.setChecked(False)
        self.symmetric_label.setFixedWidth(150)
        speaker_grid_layout.addWidget(self.symmetric_label, 13, 0)
        speaker_grid_layout.addWidget(self.symmetric_checkbox, 13, 1)
        
        # Spacer zwischen Symmetric und Trennlinie für größeren Abstand
        self.spacer_symmetric_line = QLabel("")
        self.spacer_symmetric_line.setFixedHeight(5)
        speaker_grid_layout.addWidget(self.spacer_symmetric_line, 14, 0)
        
        # Trennlinie
        line4 = QFrame()
        line4.setFrameShape(QFrame.HLine)
        line4.setFrameShadow(QFrame.Sunken)
        line4.setFixedHeight(10)
        speaker_grid_layout.addWidget(line4, 15, 0, 1, 2)
        
        # Autosplay Button
        self.autosplay_button = QPushButton("Autosplay")
        self.autosplay_button.setFont(font)
        self.autosplay_button.setFixedWidth(100)
        self.autosplay_button.setFixedHeight(24)
        self.autosplay_button.clicked.connect(self.on_autosplay_changed)
        speaker_grid_layout.addWidget(self.autosplay_button, 16, 0, 1, 2)
        
        # Füge einen vertikalen Spacer hinzu, damit alles nach oben gedrückt wird
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        speaker_grid_layout.addItem(verticalSpacer, 17, 0, 1, 2)
        
        # Erstelle Container-Widget für das Grid-Layout
        grid_container = QWidget()
        grid_container.setLayout(speaker_grid_layout)
        # Setze minimale Breite basierend auf Label (150) + Eingabefeld (40) + Padding
        grid_container.setMinimumWidth(250)
        grid_container.setMaximumWidth(250)
        
        # Erstelle ScrollArea für scrollbares Layout
        scroll_area = QScrollArea()
        scroll_area.setWidget(grid_container)
        scroll_area.setWidgetResizable(True)  # Automatisch anpassen, damit alle Widgets sichtbar sind
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Keine horizontale Scrollbar
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumWidth(250)
        scroll_area.setMaximumWidth(250)
        
        self.speaker_tab_layout.addWidget(scroll_area)
    


    def create_speaker_tab_flown(self):
        """Erstellt den Speaker Setup Tab mit allen Eingabefeldern für ein Flown Array"""
        speaker_tab = QWidget()
        self.tab_widget.addTab(speaker_tab, "Speaker Setup")
        
        self.speaker_tab_layout = QHBoxLayout(speaker_tab)
        speaker_grid_layout = QGridLayout()
        
        # Reduziere die vertikalen Abstände für kompakteres Layout
        speaker_grid_layout.setVerticalSpacing(3)
        speaker_grid_layout.setContentsMargins(5, 5, 5, 5)
        
        # Schriftgröße für alle Widgets
        font = QtGui.QFont()
        font.setPointSize(11)
        
        # Anzahl der Lautsprecher
        number_of_sources_label = QLabel("# speakers")
        number_of_sources_label.setFont(font)
        self.number_of_sources_edit = QLineEdit()
        self.number_of_sources_edit.setFont(font)
        self.number_of_sources_edit.setFixedHeight(18)
        self.number_of_sources_edit.setValidator(QIntValidator(1, 100))
        self.number_of_sources_edit.setText("1")
        self.number_of_sources_edit.editingFinished.connect(self.on_number_of_sources_changed)
        
        number_of_sources_label.setFixedWidth(150)
        self.number_of_sources_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(number_of_sources_label, 0, 0)
        speaker_grid_layout.addWidget(self.number_of_sources_edit, 0, 1)
        
        # Trennlinie
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setFixedHeight(10)
        speaker_grid_layout.addWidget(line1, 1, 0, 1, 2)
        
        # Array X-Position (ersetzt die alten X/Y/Z Position Felder)
        array_x_label = QLabel("Array X (m)")
        array_x_label.setFont(font)
        self.array_x_edit = QLineEdit()
        self.array_x_edit.setFont(font)
        self.array_x_edit.setFixedHeight(18)
        self.array_x_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_x_edit.setText("0.00")
        self.array_x_edit.editingFinished.connect(self.on_ArrayX_changed)
        array_x_label.setFixedWidth(150)
        self.array_x_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_x_label, 2, 0)
        speaker_grid_layout.addWidget(self.array_x_edit, 2, 1)
        
        # Array Y-Position
        array_y_label = QLabel("Array Y (m)")
        array_y_label.setFont(font)
        self.array_y_edit = QLineEdit()
        self.array_y_edit.setFont(font)
        self.array_y_edit.setFixedHeight(18)
        self.array_y_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_y_edit.setText("0.00")
        self.array_y_edit.editingFinished.connect(self.on_ArrayY_changed)
        array_y_label.setFixedWidth(150)
        self.array_y_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_y_label, 3, 0)
        speaker_grid_layout.addWidget(self.array_y_edit, 3, 1)
        
        # Array Z-Position
        array_z_label = QLabel("Array Z (m)")
        array_z_label.setFont(font)
        self.array_z_edit = QLineEdit()
        self.array_z_edit.setFont(font)
        self.array_z_edit.setFixedHeight(18)
        self.array_z_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.array_z_edit.setText("0.00")
        self.array_z_edit.editingFinished.connect(self.on_ArrayZ_changed)
        array_z_label.setFixedWidth(150)
        self.array_z_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(array_z_label, 4, 0)
        speaker_grid_layout.addWidget(self.array_z_edit, 4, 1)

        position_site_label = QLabel("Site (°)")
        position_site_label.setFont(font)
        self.position_site_edit = QLineEdit()
        self.position_site_edit.setFont(font)
        self.position_site_edit.setFixedHeight(18)
        self.position_site_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.position_site_edit.setText("0.00")
        self.position_site_edit.editingFinished.connect(self.on_flown_site_changed)

        position_site_label.setFixedWidth(150)
        self.position_site_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(position_site_label, 5, 0)
        speaker_grid_layout.addWidget(self.position_site_edit, 5, 1)
        
        position_azimuth_label = QLabel("Azimuth (°)")
        position_azimuth_label.setFont(font)
        self.position_azimuth_edit = QLineEdit()
        self.position_azimuth_edit.setFont(font)
        self.position_azimuth_edit.setFixedHeight(18)
        self.position_azimuth_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.position_azimuth_edit.setText("0.00")
        self.position_azimuth_edit.editingFinished.connect(self.on_flown_azimuth_changed)

        position_azimuth_label.setFixedWidth(150)
        self.position_azimuth_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(position_azimuth_label, 6, 0)
        speaker_grid_layout.addWidget(self.position_azimuth_edit, 6, 1)

        # Trennlinie
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        line2.setFixedHeight(10)
        speaker_grid_layout.addWidget(line2, 7, 0, 1, 2)
        
        # Delay
        delay_label = QLabel("Delay (ms)")
        delay_label.setFont(font)
        self.delay_edit = QLineEdit()
        self.delay_edit.setFont(font)
        self.delay_edit.setFixedHeight(18)
        self.delay_edit.setValidator(QDoubleValidator(0, 1000, 2))
        self.delay_edit.setText("0")
        self.delay_edit.editingFinished.connect(self.on_Delay_changed)
        delay_label.setFixedWidth(150)
        self.delay_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(delay_label, 8, 0)
        speaker_grid_layout.addWidget(self.delay_edit, 8, 1)
        
        # Gain
        gain_label = QLabel("Gain (dB)")
        gain_label.setFont(font)
        self.gain_edit = QLineEdit()
        self.gain_edit.setFont(font)
        self.gain_edit.setFixedHeight(18)
        self.gain_edit.setValidator(QDoubleValidator(-60, 10, 1))
        self.gain_edit.setText("0")
        self.gain_edit.editingFinished.connect(self.on_Gain_changed)
        gain_label.setFixedWidth(150)
        self.gain_edit.setFixedWidth(40)
        speaker_grid_layout.addWidget(gain_label, 9, 0)
        speaker_grid_layout.addWidget(self.gain_edit, 9, 1)
        
        # Spacer zwischen Gain und Polarity für größeren Abstand
        self.spacer_gain_polarity = QLabel("")
        self.spacer_gain_polarity.setFixedHeight(5)
        speaker_grid_layout.addWidget(self.spacer_gain_polarity, 10, 0)
        
        # Polarity inverted
        polarity_label = QLabel("Polarity inverted")
        polarity_label.setFont(font)
        self.polarity_checkbox = QCheckBox()
        self.polarity_checkbox.setChecked(False)
        self.polarity_checkbox.stateChanged.connect(self.on_Polarity_changed)
        polarity_label.setFixedWidth(150)
        speaker_grid_layout.addWidget(polarity_label, 11, 0)
        speaker_grid_layout.addWidget(self.polarity_checkbox, 11, 1)
        
        # Spacer zwischen Polarity und Symmetric für größeren Abstand
        self.spacer_label = QLabel("")
        self.spacer_label.setFixedHeight(3)
        self.spacer_label.setVisible(False)  # Versteckt für Flown Arrays
        speaker_grid_layout.addWidget(self.spacer_label, 12, 0)
        
        # Symmetric Checkbox (versteckt für Flown Arrays)
        self.symmetric_label = QLabel("Symmetric")
        self.symmetric_label.setFont(font)
        self.symmetric_checkbox = QCheckBox()
        self.symmetric_checkbox.setChecked(False)
        self.symmetric_checkbox.setVisible(False)  # Versteckt für Flown Arrays
        self.symmetric_label.setFixedWidth(150)
        self.symmetric_label.setVisible(False)  # Label auch verstecken
        speaker_grid_layout.addWidget(self.symmetric_label, 13, 0)
        speaker_grid_layout.addWidget(self.symmetric_checkbox, 13, 1)
        
        # Spacer zwischen Symmetric und Trennlinie für größeren Abstand
        self.spacer_symmetric_line = QLabel("")
        self.spacer_symmetric_line.setFixedHeight(8)
        self.spacer_symmetric_line.setVisible(False)  # Versteckt für Flown Arrays
        speaker_grid_layout.addWidget(self.spacer_symmetric_line, 14, 0)
        
        # Trennlinie
        line4 = QFrame()
        line4.setFrameShape(QFrame.HLine)
        line4.setFrameShadow(QFrame.Sunken)
        line4.setFixedHeight(10)
        speaker_grid_layout.addWidget(line4, 15, 0, 1, 2)
        
        # Füge einen vertikalen Spacer hinzu, damit alles nach oben gedrückt wird
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        speaker_grid_layout.addItem(verticalSpacer, 16, 0, 1, 2)
        
        # Erstelle Container-Widget für das Grid-Layout
        grid_container = QWidget()
        grid_container.setLayout(speaker_grid_layout)
        # Setze minimale Breite basierend auf Label (150) + Eingabefeld (40) + Padding
        grid_container.setMinimumWidth(250)
        grid_container.setMaximumWidth(250)
        
        # Erstelle ScrollArea für scrollbares Layout
        scroll_area = QScrollArea()
        scroll_area.setWidget(grid_container)
        scroll_area.setWidgetResizable(False)  # Nicht automatisch anpassen
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Keine horizontale Scrollbar
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumWidth(250)
        scroll_area.setMaximumWidth(250)
        
        self.speaker_tab_layout.addWidget(scroll_area)


    # ---- Windowing - Beamsteering ----

    def create_windowing_tab(self):
        windowing_tab = QWidget()
        main_layout = QHBoxLayout(windowing_tab)

        # Linke Seite für Eingabefelder
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        gridLayout_windowing = QGridLayout()
        # Kompakter: Abstände/Margins und Schriftgröße
        gridLayout_windowing.setVerticalSpacing(3)
        gridLayout_windowing.setContentsMargins(5, 5, 5, 5)
        wdw_font = QtGui.QFont()
        wdw_font.setPointSize(11)
        
        self.Label_WindowFunction = QLabel("Window Function")
        self.Label_WindowFunction.setFont(wdw_font)
        self.Label_WindowFunction.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.WindowFunction = QComboBox()
        self.WindowFunction.setFont(wdw_font)
        self.WindowFunction.setFixedWidth(100)  # Bleibt bei 100 für ComboBox
        self.WindowFunction.addItems(["tukey", "gauss", "flattop", "blackman"])
        self.WindowFunction.currentIndexChanged.connect(self.on_WindowFunction_changed)
    
        gridLayout_windowing.addWidget(self.Label_WindowFunction, 0, 0)
        gridLayout_windowing.addWidget(self.WindowFunction, 0, 1)

        line_12 = QFrame()
        line_12.setFrameShape(QFrame.HLine)
        line_12.setFrameShadow(QFrame.Sunken)
        line_12.setFixedHeight(10)
        gridLayout_windowing.addWidget(line_12, 1, 0, 1, 2)
    
        self.Label_WindowRestriction = QLabel("Window Restriction")
        self.Label_WindowRestriction.setFont(wdw_font)
        self.Label_WindowRestriction.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.WindowRestriction = QLineEdit()
        self.WindowRestriction.setFont(wdw_font)
        self.WindowRestriction.setFixedWidth(40)  # Konsistent mit Speaker Setup
        self.WindowRestriction.setValidator(QDoubleValidator(-60.0, 0.0, 1))
        self.WindowRestriction.setText("-9.8")
        self.WindowRestriction.editingFinished.connect(self.on_WindowRestriction_changed)
        gridLayout_windowing.addWidget(self.Label_WindowRestriction, 2, 0)
        gridLayout_windowing.addWidget(self.WindowRestriction, 2, 1)
    
        self.Label_Alpha = QLabel("Alpha")
        self.Label_Alpha.setFont(wdw_font)
        self.Label_Alpha.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.Alpha = QLineEdit()
        self.Alpha.setFont(wdw_font)
        self.Alpha.setValidator(QDoubleValidator(0.01, 1.0, 2))
        self.Alpha.setFixedWidth(40)  # Konsistent mit Speaker Setup
        self.Alpha.setText("0.6")
        self.Alpha.editingFinished.connect(self.on_Alpha_changed)
        gridLayout_windowing.addWidget(self.Label_Alpha, 3, 0)
        gridLayout_windowing.addWidget(self.Alpha, 3, 1)
    
        line_11 = QFrame()
        line_11.setFrameShape(QFrame.HLine)
        line_11.setFrameShadow(QFrame.Sunken)
        line_11.setFixedHeight(10)
        gridLayout_windowing.addWidget(line_11, 4, 0, 1, 2)
    
        # Buttons vertikal untereinander - linksbündig
        self.Pushbutton_SourcesSet = QPushButton("Set")
        self.Pushbutton_SourcesSet.setFont(wdw_font)
        self.Pushbutton_SourcesSet.setFixedWidth(100)
        self.Pushbutton_SourcesSet.setFixedHeight(24)
        self.Pushbutton_SourcesSet.clicked.connect(self.on_Sources_set_changed)
        gridLayout_windowing.addWidget(self.Pushbutton_SourcesSet, 5, 0)  # Linksbündig in Spalte 0
        
        self.Pushbutton_SourcesToZero = QPushButton("Reset")
        self.Pushbutton_SourcesToZero.setFont(wdw_font)
        self.Pushbutton_SourcesToZero.setFixedWidth(100)
        self.Pushbutton_SourcesToZero.setFixedHeight(24)
        self.Pushbutton_SourcesToZero.clicked.connect(self.on_Sources_to_zero_changed)
        gridLayout_windowing.addWidget(self.Pushbutton_SourcesToZero, 5, 1)  # Rechts in Spalte 1
    
        verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        gridLayout_windowing.addItem(verticalSpacer_5, 6, 0, 1, 2)
    
        left_layout.addLayout(gridLayout_windowing)
        left_widget.setLayout(left_layout)

        # Rechte Seite für den Plot
        # OPTIMIERUNG: Plot nur erstellen, wenn er noch nicht existiert
        if self.windowing_plot is None:
            self.windowing_plot = WindowingPlot(windowing_tab, settings=self.settings, container=self.container)
            # Kein setFixedHeight - lässt Plot sich anpassen
            self.windowing_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            # Plot existiert bereits - prüfe ob er noch gültig ist
            try:
                # Prüfe ob das Objekt noch gültig ist, indem wir auf ein Attribut zugreifen
                _ = self.windowing_plot.parent()
                # Objekt ist gültig - setze Parent auf neuen Tab
                self.windowing_plot.setParent(windowing_tab)
            except RuntimeError:
                # Objekt wurde gelöscht - erstelle neuen Plot
                self.windowing_plot = WindowingPlot(windowing_tab, settings=self.settings, container=self.container)
                self.windowing_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Setze feste Breite für linke Seite (konsistent mit Speaker Setup)
        left_widget.setFixedWidth(250)
        
        # Fügen Sie beide Seiten zum Hauptlayout hinzu
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.windowing_plot)

        # Kein setStretch nötig - left_widget hat feste Breite, Plot expandiert
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.tab_widget.addTab(windowing_tab, "Windowing")


    def create_beamsteering_tab(self):
        beamsteering_tab = QWidget()
        main_layout = QHBoxLayout(beamsteering_tab)

        # Linke Seite für Eingabefelder
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        gridLayout_beamsteering = QGridLayout()
        # Kompakt + Schriftgröße
        gridLayout_beamsteering.setVerticalSpacing(3)
        gridLayout_beamsteering.setContentsMargins(5, 5, 5, 5)
        bs_font = QtGui.QFont()
        bs_font.setPointSize(11)

        self.Label_ArcShape = QLabel("Arc shape")
        self.Label_ArcShape.setFont(bs_font)
        self.Label_ArcShape.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.ArcShape = QComboBox()
        self.ArcShape.setFont(bs_font)
        self.ArcShape.addItems(["Manual", "Pointed Arc", "Circular Arc", "Spiral Arc", "Eliptical Arc", "Linear"])
        self.ArcShape.setFixedWidth(100)  # Bleibt bei 100 für ComboBox
        self.ArcShape.currentIndexChanged.connect(self.on_ArcShape_changed)
        gridLayout_beamsteering.addWidget(self.Label_ArcShape, 0, 0)
        gridLayout_beamsteering.addWidget(self.ArcShape, 0, 1)

        # Fügen Sie die Linie direkt nach ArcShape ein
        line_13 = QFrame()
        line_13.setFrameShape(QFrame.HLine)
        line_13.setFrameShadow(QFrame.Sunken)
        line_13.setFixedHeight(10)
        gridLayout_beamsteering.addWidget(line_13, 1, 0, 1, 2)  # Zeile 1, über 2 Spalten

        self.Label_ArcAngle = QLabel("Opening angle")
        self.Label_ArcAngle.setFont(bs_font)
        self.Label_ArcAngle.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.ArcAngle = QLineEdit()
        self.ArcAngle.setFont(bs_font)
        self.ArcAngle.setFixedWidth(40)  # Konsistent mit Speaker Setup
        self.ArcAngle.textChanged.connect(self.on_ArcAngle_text_changed)
        self.ArcAngle.editingFinished.connect(self.on_ArcAngle_editing_finished)
        gridLayout_beamsteering.addWidget(self.Label_ArcAngle, 2, 0)  # Zeile 2
        gridLayout_beamsteering.addWidget(self.ArcAngle, 2, 1)
    
        self.Label_ArcScaleFactor = QLabel("Scale factor")
        self.Label_ArcScaleFactor.setFont(bs_font)
        self.Label_ArcScaleFactor.setFixedWidth(150)  # Konsistent mit Speaker Setup
        self.ArcScaleFactor = QLineEdit()
        self.ArcScaleFactor.setFont(bs_font)
        self.ArcScaleFactor.setFixedWidth(40)  # Konsistent mit Speaker Setup
        self.ArcScaleFactor.textChanged.connect(self.on_ArcScaleFactor_text_changed)
        self.ArcScaleFactor.editingFinished.connect(self.on_ArcScaleFactor_editing_finished)
        gridLayout_beamsteering.addWidget(self.Label_ArcScaleFactor, 3, 0)
        gridLayout_beamsteering.addWidget(self.ArcScaleFactor, 3, 1)

        # Neue horizontale Linie
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(10)
        gridLayout_beamsteering.addWidget(line, 4, 0, 1, 2)

        # Neues Info-Label mit expandierender Höhe
        self.Label_ArcInfo = QLabel("")
        self.Label_ArcInfo.setFont(bs_font)
        self.Label_ArcInfo.setWordWrap(True)  # Ermöglicht Zeilenumbrüche
        self.Label_ArcInfo.setFixedWidth(250)  # Setzt eine feste Breite

        # Setze die Size Policy auf Expanding in vertikaler Richtung
        size_policy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.Label_ArcInfo.setSizePolicy(size_policy)

        # Optional: Setze minimale Höhe
        self.Label_ArcInfo.setMinimumHeight(50)

        gridLayout_beamsteering.addWidget(self.Label_ArcInfo, 5, 0, 1, 2)

        left_layout.addLayout(gridLayout_beamsteering)
        left_layout.addStretch(1)  # Fügt Abstand am unteren Ende hinzu

        # Rechte Seite für den Plot
        # OPTIMIERUNG: Plot nur erstellen, wenn er noch nicht existiert
        if self.beamsteering_plot is None:
            self.beamsteering_plot = BeamsteeringPlot(beamsteering_tab, settings=self.settings, container=self.container)
            # Kein setFixedHeight - lässt Plot sich anpassen
            self.beamsteering_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            # Plot existiert bereits - prüfe ob er noch gültig ist
            try:
                # Prüfe ob das Objekt noch gültig ist, indem wir auf ein Attribut zugreifen
                _ = self.beamsteering_plot.parent()
                # Objekt ist gültig - setze Parent auf neuen Tab
                self.beamsteering_plot.setParent(beamsteering_tab)
            except RuntimeError:
                # Objekt wurde gelöscht - erstelle neuen Plot
                self.beamsteering_plot = BeamsteeringPlot(beamsteering_tab, settings=self.settings, container=self.container)
                self.beamsteering_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Setze feste Breite für linke Seite (konsistent mit Speaker Setup)
        left_widget.setFixedWidth(250)
        
        # Fügen Sie beide Seiten zum Hauptlayout hinzu
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.beamsteering_plot)

        # Kein setStretch nötig - left_widget hat feste Breite, Plot expandiert
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.tab_widget.addTab(beamsteering_tab, "Beamsteering")



    def checkbox_symmetric_checked(self, instance, state):
        if state == 2:
            instance['state'] = True
        else:
            instance['state'] = False

        # Speichere den Zustand auch im SpeakerArray für Persistenz
        speaker_array_id = instance.get('id')
        if speaker_array_id:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                speaker_array.symmetric = instance['state']

        # Nur neu berechnen, wenn sich tatsächlich Werte geändert haben
        if instance['state']:
            # Prüfe, ob sich Werte durch Symmetrie geändert haben
            has_changes = self.apply_symmetric_values(instance)
            if has_changes:
                # Nur wenn sich Werte geändert haben, UI aktualisieren und neu berechnen
                self.update_input_fields(instance)
                self.main_window.update_speaker_array_calculations()
            else:
                # Wenn bereits alles symmetrisch ist, nur UI aktualisieren (keine Neuberechnung)
                self.update_input_fields(instance)
        else:
            # Wenn Symmetrie deaktiviert wird, nur UI aktualisieren (keine Neuberechnung nötig)
            self.update_input_fields(instance)


    # @measure_time
    def update_input_fields(self, instance):
        """
        Aktualisiert alle Eingabefelder für eine speakerspecs_instance.
        Löscht zuerst alle bestehenden Widgets und erstellt sie neu.
        Unterscheidet zwischen Stack und Flown Arrays.
        
        Args:
            instance (dict): Dictionary mit den UI-Elementen und Daten einer Speakerspecs-Instanz
        """
        
        try:
            # Hole das Grid-Layout aus der Instance
            gridLayout_sources = instance.get('gridLayout_sources')
            if gridLayout_sources is None:
                return
                                
            # Lösche alle existierenden Widgets aus dem Layout
            while gridLayout_sources.count():
                item = gridLayout_sources.takeAt(0)  # Entferne Item aus Layout
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)  # Entferne Parent-Beziehung
                    widget.deleteLater()    # Schedule Widget zum Löschen
            
            # Hole das zugehörige Speaker Array und aktualisiere Quellen
            speaker_array_id = instance.get('id')
                
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
                        
            # Prüfe, ob es sich um ein Stack oder Flown Array handelt
            is_stack = True  # Standardmäßig Stack annehmen
            if hasattr(speaker_array, 'configuration'):
                is_stack = speaker_array.configuration.lower() == "stack"
                        
            number_of_sources = speaker_array.number_of_sources
            speaker_array.update_sources(number_of_sources)  # Aktualisiert Arrays für die Quellen

            # Schriftgröße für Labels
            font = QtGui.QFont()
            font.setPointSize(11)

            # Definiere Labels für die Spalten basierend auf dem Array-Typ
            if is_stack:
                labels = ["Source Type", "X (m)", "Y (m)", "Z (m)", "Azimuth (°)", "Delay (ms)", "Gain (dB)"]
                
                # Erstelle Labels und Spacer im Grid für Stack Arrays
                for i, label_text in enumerate(labels):
                    # Erstelle Label
                    label = QtWidgets.QLabel(label_text, self.main_window)
                    label.setFont(font)
                    gridLayout_sources.addWidget(label, i, 2)  # Füge Label in Spalte 2 ein

                    # Füge Spacer am Ende der Zeile ein
                    spacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                    gridLayout_sources.addItem(spacer, i, 16)
            else:
                # Für Flown Arrays werden die Labels horizontal angeordnet
                labels = ["#", "Source Type", "Angle (°)", "Delay (ms)", "Gain (dB)"]
                
                # Erstelle Labels horizontal für Flown Arrays
                for i, label_text in enumerate(labels):
                    # Erstelle Label
                    label = QtWidgets.QLabel(label_text, self.main_window)
                    label.setFont(font)
                    label.setAlignment(Qt.AlignCenter)  # Zentriere den Text
                    gridLayout_sources.addWidget(label, 0, i + 2)  # Füge Label in Zeile 0 ein
                
                # Füge Spacer am Ende der Zeile ein
                spacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                gridLayout_sources.addItem(spacer, 0, len(labels) + 2)

            # Wenn symmetrisch aktiviert ist, wende symmetrische Werte an
            if instance.get('state', False):
                self.apply_symmetric_values(instance)

            # Aktualisiere Arc Info Text basierend auf der Form
            if hasattr(self, 'update_arc_info_text') and hasattr(speaker_array, 'arc_shape'):
                self.update_arc_info_text(speaker_array.arc_shape)

            # Validiere Arc Angle wenn Array existiert
            if speaker_array and hasattr(self, 'validate_arc_angle'):
                self.validate_arc_angle(speaker_array)

            # Aktualisiere alle weiteren Widgets (Comboboxen, Spinboxen etc.)
            if hasattr(self, 'update_widgets'):
                self.update_widgets(instance)
                            
        except Exception as e:
            print(f"FEHLER in update_input_fields: {e}")
            import traceback
            traceback.print_exc()
            
    def update_delay_input_fields(self, speaker_array_id):
        """
        Aktualisiert die Delay-Eingabefelder mit den aktuellen Werten aus dem Speaker Array.
        Wird aufgerufen, wenn sich Positionen oder andere Parameter ändern, die die Delays beeinflussen.
        
        Args:
            speaker_array_id: Die ID des Speaker Arrays
        """
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array or not hasattr(speaker_array, 'source_time'):
                return
                
            # Nur aktualisieren, wenn Arc Shape nicht auf 'manual' gesetzt ist
            if hasattr(speaker_array, 'arc_shape') and speaker_array.arc_shape.lower() == "manual":
                return  # Im manuellen Modus werden Delays nicht automatisch aktualisiert
            
            instance = self.get_speakerspecs_instance(speaker_array_id)
            if not instance or not instance['gridLayout_sources']:
                return
            
            # Suche nach allen Delay-Eingabefeldern und aktualisiere sie
            for i in range(speaker_array.number_of_sources):
                # Suche nach dem Delay-Eingabefeld für diese Quelle
                delay_widget = None
            
                # Methode 1: Direkt über den Namen im Instance-Dictionary suchen
                if f'delay_input_{i+1}' in instance:
                    delay_widget = instance[f'delay_input_{i+1}']
            
                # Methode 2: Im Layout nach dem Widget mit dem passenden Namen suchen
                if not delay_widget:
                    for j in range(instance['gridLayout_sources'].count()):
                        item = instance['gridLayout_sources'].itemAt(j)
                        widget = item.widget() if item else None
                        if widget and isinstance(widget, QLineEdit) and widget.objectName() == f"delay_input_{i+1}":
                            delay_widget = widget
                            break
            
                # Wenn ein Widget gefunden wurde, aktualisiere seinen Wert
                if delay_widget and i < len(speaker_array.source_time):
                    old_value = delay_widget.text()
                    new_value = f"{speaker_array.source_time[i]:.2f}"
                
                    if old_value != new_value:
                        delay_widget.blockSignals(True)
                        delay_widget.setText(new_value)
                        delay_widget.blockSignals(False)
                
        except Exception as e:
            print(f"Fehler in update_delay_input_fields: {e}")
            import traceback
            traceback.print_exc()
            

    def apply_symmetric_values(self, instance):
        """
        Wendet symmetrische Werte auf alle relevanten Felder an.
        Berücksichtigt spezielle Symmetrieregeln für verschiedene Werttypen.
        
        Returns:
            bool: True, wenn sich Werte geändert haben, False wenn bereits alles symmetrisch ist
        """
        try:
            
            speaker_array_id = instance['id']
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                return False
            
            # Prüfe, ob bereits alles symmetrisch ist
            has_changes = False
            num_sources = speaker_array.number_of_sources
            half = num_sources // 2
            
            # Y-Position: Gleicher Wert
            for i in range(half):
                aktuelles_element = speaker_array.source_position_y[i]
                symmetric_index = num_sources - i - 1
                if speaker_array.source_position_y[symmetric_index] != aktuelles_element:
                    has_changes = True
                    speaker_array.source_position_y[symmetric_index] = aktuelles_element

            # X-Position: Invertierter Wert (negativ)
            for i in range(half):
                aktuelles_element = speaker_array.source_position_x[i]
                symmetric_index = num_sources - i - 1
                expected_value = -aktuelles_element
                if abs(speaker_array.source_position_x[symmetric_index] - expected_value) > 1e-6:
                    has_changes = True
                    speaker_array.source_position_x[symmetric_index] = expected_value

            # Z-Position: Gleicher Wert
            if hasattr(speaker_array, 'source_position_z_stack') and speaker_array.source_position_z_stack is not None:
                for i in range(half):
                    aktuelles_element = speaker_array.source_position_z_stack[i]
                    symmetric_index = num_sources - i - 1
                    if speaker_array.source_position_z_stack[symmetric_index] != aktuelles_element:
                        has_changes = True
                        speaker_array.source_position_z_stack[symmetric_index] = aktuelles_element

            # Azimuth: Invertierter Wert
            for i in range(half):
                aktuelles_element = speaker_array.source_azimuth[i]
                symmetric_index = num_sources - i - 1
                expected_value = -aktuelles_element
                if abs(speaker_array.source_azimuth[symmetric_index] - expected_value) > 1e-6:
                    has_changes = True
                    speaker_array.source_azimuth[symmetric_index] = expected_value

            # Delay: Gleicher Wert
            if hasattr(speaker_array, 'source_time') and speaker_array.source_time is not None:
                for i in range(half):
                    aktuelles_element = speaker_array.source_time[i]
                    symmetric_index = num_sources - i - 1
                    if speaker_array.source_time[symmetric_index] != aktuelles_element:
                        has_changes = True
                        speaker_array.source_time[symmetric_index] = aktuelles_element

            # Level: Gleicher Wert
            for i in range(half):
                aktuelles_element = speaker_array.source_level[i]
                symmetric_index = num_sources - i - 1
                if speaker_array.source_level[symmetric_index] != aktuelles_element:
                    has_changes = True
                    speaker_array.source_level[symmetric_index] = aktuelles_element
                
            # Polar Pattern: L/R-Tausch
            if hasattr(speaker_array, 'source_polar_pattern') and speaker_array.source_polar_pattern is not None:
                for i in range(half):
                    aktuelles_element = speaker_array.source_polar_pattern[i]
                    symmetric_element = aktuelles_element
                    
                    # L/R-Tausch
                    if 'L' in str(aktuelles_element):
                        symmetric_element = str(aktuelles_element).replace('L', 'R')
                    elif 'R' in str(aktuelles_element):
                        symmetric_element = str(aktuelles_element).replace('R', 'L')
                    
                    symmetric_index = num_sources - i - 1
                    if str(speaker_array.source_polar_pattern[symmetric_index]) != symmetric_element:
                        has_changes = True
                        speaker_array.source_polar_pattern[symmetric_index] = symmetric_element
            
            # Aktualisiere die Eingabefelder
            gridLayout_sources = instance['gridLayout_sources']
            if gridLayout_sources:
                
                # Aktualisiere alle Eingabefelder für die zweite Hälfte des Arrays
                half_sources = speaker_array.number_of_sources // 2                
                # Sammle alle Comboboxen im Layout
                all_comboboxes = []
                for i in range(gridLayout_sources.count()):
                    item = gridLayout_sources.itemAt(i)
                    widget = item.widget()
                    if widget and isinstance(widget, QComboBox):
                        all_comboboxes.append((widget.objectName(), widget))
                                
                for source_index in range(half_sources, speaker_array.number_of_sources):
                    
                    # X-Position
                    for i in range(gridLayout_sources.count()):
                        item = gridLayout_sources.itemAt(i)
                        widget = item.widget()
                        if (widget and isinstance(widget, QLineEdit) and 
                            widget.objectName() == f"position_x_input_{source_index + 1}"):
                            widget.blockSignals(True)
                            widget.setText(f"{speaker_array.source_position_x[source_index]:.2f}")
                            widget.blockSignals(False)
                    
                    # Y-Position
                    for i in range(gridLayout_sources.count()):
                        item = gridLayout_sources.itemAt(i)
                        widget = item.widget()
                        if (widget and isinstance(widget, QLineEdit) and 
                            widget.objectName() == f"position_y_input_{source_index + 1}"):
                            widget.blockSignals(True)
                            widget.setText(f"{speaker_array.source_position_y[source_index]:.2f}")
                            widget.blockSignals(False)
                    
                    # Z-Position
                    if hasattr(speaker_array, 'source_position_z') and speaker_array.source_position_z_stack is not None:
                        for i in range(gridLayout_sources.count()):
                            item = gridLayout_sources.itemAt(i)
                            widget = item.widget()
                            if (widget and isinstance(widget, QLineEdit) and 
                                widget.objectName() == f"position_z_input_{source_index + 1}"):
                                widget.blockSignals(True)
                                widget.setText(f"{speaker_array.source_position_z_stack[source_index]:.2f}")
                                widget.blockSignals(False)
                    
                    # Azimuth
                    for i in range(gridLayout_sources.count()):
                        item = gridLayout_sources.itemAt(i)
                        widget = item.widget()
                        if (widget and isinstance(widget, QLineEdit) and 
                            widget.objectName() == f"azimuth_input_{source_index + 1}"):
                            widget.blockSignals(True)
                            widget.setText(f"{speaker_array.source_azimuth[source_index]:.1f}")
                            widget.blockSignals(False)
                    
                    # Delay
                    for i in range(gridLayout_sources.count()):
                        item = gridLayout_sources.itemAt(i)
                        widget = item.widget()
                        if (widget and isinstance(widget, QLineEdit) and 
                            (widget.objectName() == f"time_input_{source_index + 1}" or 
                             widget.objectName() == f"delay_input_{source_index + 1}")):
                            widget.blockSignals(True)
                            widget.setText(f"{speaker_array.source_time[source_index]:.2f}")
                            widget.blockSignals(False)
                    
                    # Level
                    for i in range(gridLayout_sources.count()):
                        item = gridLayout_sources.itemAt(i)
                        widget = item.widget()
                        if (widget and isinstance(widget, QLineEdit) and 
                            widget.objectName() == f"gain_input_{source_index + 1}"):
                            widget.blockSignals(True)
                            widget.setText(f"{speaker_array.source_level[source_index]:.2f}")
                            widget.blockSignals(False)
                    
                    # Polar Pattern
                    if hasattr(speaker_array, 'source_polar_pattern') and speaker_array.source_polar_pattern is not None:
                        pattern = speaker_array.source_polar_pattern[source_index]
                        
                        # Versuche verschiedene mögliche Namen für die Combobox
                        combobox_found = False
                        possible_names = [
                            f"polar_pattern_combobox_{source_index + 1}",
                            f"source_type_combobox_{source_index + 1}",
                            f"combobox_source_type_{source_index + 1}",
                            f"speaker_type_combo_{source_index + 1}"
                        ]
                        
                        for name in possible_names:
                            for i in range(gridLayout_sources.count()):
                                item = gridLayout_sources.itemAt(i)
                                widget = item.widget()
                                if (widget and isinstance(widget, QComboBox) and 
                                    widget.objectName() == name):
                                    pattern_index = widget.findText(pattern)
                                    if pattern_index >= 0:
                                        widget.blockSignals(True)
                                        widget.setCurrentIndex(pattern_index)
                                        widget.blockSignals(False)
                                        combobox_found = True
                                        break
                                    else:
                                        print(f"    Text '{pattern}' nicht in Combobox gefunden. Verfügbare Einträge:")
                                        for j in range(widget.count()):
                                            print(f"      {j}: {widget.itemText(j)}")
                            
                            if combobox_found:
                                break
            
        except Exception as e:
            print(f"Fehler in apply_symmetric_values: {e}")
            import traceback
            traceback.print_exc()
            return False

        return has_changes  


    def update_widgets(self, instance):
        """
        Aktualisiert alle Widgets (Comboboxen und QLineEdits) für eine speakerspecs_instance.
        Wird hauptsächlich im Speaker Setup Tab verwendet.
        
        Args:
            instance (dict): Dictionary mit den UI-Elementen und Daten einer Speakerspecs-Instanz
        """
        speaker_array = self.settings.get_speaker_array(instance['id'])
        if speaker_array is None:
            return
            
        # Prüfe, ob es sich um ein Stack oder Flown Array handelt
        is_stack = True  # Standardmäßig Stack annehmen
        if hasattr(speaker_array, 'configuration'):
            is_stack = speaker_array.configuration.lower() == "stack"
        
        # Aktualisiert die Source Type Comboboxen (Lautsprechertypen)
        self.update_speaker_comboboxes(instance)
        
        # Aktualisiert die Eingabefelder für X, Y, Z, Azimuth, Delay und Gain
        self.update_speaker_qlineedit(instance)
        
        # Aktualisiere die Symmetrie-Checkbox basierend auf dem Array-Typ
        self.update_symmetry_checkbox(instance, is_stack)
        
        # Wenn es sich um ein Flown Array handelt, aktualisiere die speziellen Eingabefelder
        if not is_stack:
            self.update_flown_array_fields(speaker_array)
        
        # Aktiviert alle Widgets im Grid-Layout
        # Dies ist wichtig nach dem Symmetric-Mode, wo einige Widgets deaktiviert wurden
        for widget in instance['gridLayout_sources'].findChildren(QWidget):
            widget.setEnabled(True)

    def update_flown_array_fields(self, speaker_array):
        """
        Aktualisiert die Felder für Flown Arrays basierend auf den Daten des übergebenen speaker_array
        
        Args:
            speaker_array: Das SpeakerArray-Objekt mit den Daten
        """
        try:
            # source_position_z sollte immer verfügbar sein als numpy array
            # Stelle sicher, dass es die richtige Länge hat
            if len(speaker_array.source_position_z) != speaker_array.number_of_sources:
                # Passe die Länge an
                import numpy as np
                first_value = speaker_array.source_position_z[0] if len(speaker_array.source_position_z) > 0 else 0.0
                speaker_array.source_position_z = np.full(speaker_array.number_of_sources, first_value)
            
            # Prüfen, ob die UI-Elemente existieren, bevor wir sie aktualisieren
            if hasattr(self, 'position_x_edit') and self.position_x_edit is not None:
                self.position_x_edit.setText(f"{speaker_array.source_position_x[0]:.2f}")
            
            if hasattr(self, 'position_y_edit') and self.position_y_edit is not None:
                self.position_y_edit.setText(f"{speaker_array.source_position_y[0]:.2f}")
            
            if hasattr(self, 'position_z_edit') and self.position_z_edit is not None:
                # Verwende für Flown-Arrays die Top-Kante als Referenz
                if hasattr(speaker_array, 'source_position_z_flown') and speaker_array.source_position_z_flown is not None:
                    top_values = speaker_array.source_position_z_flown
                    if len(top_values) < speaker_array.number_of_sources:
                        import numpy as np
                        if isinstance(top_values, np.ndarray):
                            top_values = top_values.tolist()
                        last_value = top_values[-1] if len(top_values) > 0 else 0.0
                        top_values.extend([last_value] * (speaker_array.number_of_sources - len(top_values)))
                        speaker_array.source_position_z_flown = top_values
                    display_value = speaker_array.source_position_z_flown[0]
                else:
                    display_value = speaker_array.source_position_z[0] if len(speaker_array.source_position_z) > 0 else 0.0
                self.position_z_edit.setText(f"{display_value:.2f}")
            
            if hasattr(self, 'position_site_edit') and self.position_site_edit is not None:
                if not hasattr(speaker_array, 'source_site') or speaker_array.source_site is None:
                    import numpy as np
                    speaker_array.source_site = np.zeros(speaker_array.number_of_sources)
                elif len(speaker_array.source_site) < speaker_array.number_of_sources:
                    import numpy as np
                    if isinstance(speaker_array.source_site, np.ndarray):
                        base_value = speaker_array.source_site[-1] if speaker_array.source_site.size > 0 else 0.0
                        speaker_array.source_site = np.pad(
                            speaker_array.source_site,
                            (0, speaker_array.number_of_sources - speaker_array.source_site.size),
                            mode='edge'
                        )
                    else:
                        base_value = speaker_array.source_site[-1] if len(speaker_array.source_site) > 0 else 0.0
                        speaker_array.source_site = list(speaker_array.source_site) + [
                            base_value
                        ] * (speaker_array.number_of_sources - len(speaker_array.source_site))
                site_value = speaker_array.source_site[0] if len(speaker_array.source_site) > 0 else 0.0
                self.position_site_edit.setText(f"{float(site_value):.2f}")

            if hasattr(self, 'position_azimuth_edit') and self.position_azimuth_edit is not None:
                current_azimuth = speaker_array.source_azimuth[0] if len(speaker_array.source_azimuth) > 0 else 0.0
                self.position_azimuth_edit.setText(f"{current_azimuth:.2f}")
            
            # Weitere UI-Elemente nach Bedarf prüfen und aktualisieren
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Flown Array Felder: {str(e)}")
            import traceback
            traceback.print_exc()

            
    # @measure_time
    def update_speaker_comboboxes(self, instance):
        """
        Erstellt und aktualisiert die Comboboxen für die Lautsprechertypen im Speaker Setup.
        Unterscheidet zwischen Stack und Flown Arrays.
        Optimierte Version mit blockierten UI-Updates und nur einer Berechnung am Ende.
        """
        speaker_array = self.settings.get_speaker_array(instance['id'])
        if speaker_array is None:
            return
            
        # Hole die Lautsprechernamen aus dem DataContainer
        speaker_names = self.container.get_speaker_names()
        if not speaker_names:
            return
        
        # Prüfe, ob es sich um ein Stack oder Flown Array handelt
        is_stack = True  # Standardmäßig Stack annehmen
        if hasattr(speaker_array, 'configuration'):
            is_stack = speaker_array.configuration.lower() == "stack"
        
        
        # Filtere Lautsprecher basierend auf ihrer Konfiguration
        stack_speakers = []
        flown_speakers = []
        cabinet_data = self.container.data.get('cabinet_data', [])
        
        
        for i, name in enumerate(self.container.data.get('speaker_names', [])):
            
            # Standardmäßig als Stack klassifizieren, es sei denn, es gibt eine explizite Flown-Konfiguration
            is_flown_speaker = False
            
            if i < len(cabinet_data) and cabinet_data[i] is not None:
                cabinet = cabinet_data[i]
                
                # Debug-Ausgabe für Cabinet-Typ
                cabinet_type = type(cabinet).__name__
                
                # Extrahiere die Konfiguration aus dem Cabinet-Objekt
                config = None
                
                # Für Listen von Dictionaries
                if isinstance(cabinet, list) and len(cabinet) > 0:
                    if isinstance(cabinet[0], dict):
                        config = cabinet[0].get('configuration', '').lower()
                
                # Für einzelne Dictionaries
                elif isinstance(cabinet, dict):
                    config = cabinet.get('configuration', '').lower()
                
                # Für NumPy Arrays - versuche, die Konfiguration zu extrahieren
                elif hasattr(cabinet, 'dtype'):
                    
                    # Versuche, das Array zu einem Python-Objekt zu konvertieren
                    try:
                        if cabinet.size == 1 and cabinet.dtype == np.dtype('O'):
                            # Extrahiere das Objekt aus dem Array
                            extracted_obj = cabinet.item()
                            
                            # Prüfe, ob das extrahierte Objekt eine Konfiguration hat
                            if isinstance(extracted_obj, dict):
                                config = extracted_obj.get('configuration', '').lower()
                            elif isinstance(extracted_obj, list) and len(extracted_obj) > 0 and isinstance(extracted_obj[0], dict):
                                config = extracted_obj[0].get('configuration', '').lower()
                    except Exception as e:
                        print(f"  Fehler beim Extrahieren des Objekts: {e}")
                
                # Klassifiziere basierend auf der gefundenen Konfiguration
                if config == 'flown':
                    is_flown_speaker = True
            
            # Füge den Lautsprecher zur entsprechenden Liste hinzu
            if is_flown_speaker:
                flown_speakers.append(name)
            else:
                stack_speakers.append(name)        
        # Wenn keine gefilterten Lautsprecher gefunden wurden, verwende alle
        if not stack_speakers:
            stack_speakers = speaker_names
        if not flown_speakers:
            flown_speakers = speaker_names

        # Wähle die entsprechende Liste basierend auf der Array-Konfiguration
        filtered_speakers = stack_speakers if is_stack else flown_speakers
        
        # Erstelle ein Dictionary für schnellere Lookups
        speaker_name_to_index = {name: i for i, name in enumerate(filtered_speakers)}
        
        # Initialisiere source_polar_pattern einmal vor der Schleife
        if not hasattr(speaker_array, 'source_polar_pattern') or speaker_array.source_polar_pattern is None:
            if len(filtered_speakers) > 0:
                default_speaker = filtered_speakers[0]
                speaker_array.source_polar_pattern = np.array([default_speaker] * speaker_array.number_of_sources, dtype=object)
            else:
                return
        
        # Stelle sicher, dass die Länge von source_polar_pattern korrekt ist
        if len(speaker_array.source_polar_pattern) < speaker_array.number_of_sources:
            default_speaker = speaker_array.source_polar_pattern[0] if len(speaker_array.source_polar_pattern) > 0 else filtered_speakers[0]
            speaker_array.source_polar_pattern = np.append(
                speaker_array.source_polar_pattern,
                [default_speaker] * (speaker_array.number_of_sources - len(speaker_array.source_polar_pattern))
            )
        
        # Stelle sicher, dass die aktuellen Patterns zur Konfiguration passen
        for i in range(len(speaker_array.source_polar_pattern)):
            current_pattern = speaker_array.source_polar_pattern[i]
            if current_pattern not in filtered_speakers and filtered_speakers:
                # Wenn das aktuelle Pattern nicht zur Konfiguration passt, setze es auf den ersten passenden Lautsprecher
                speaker_array.source_polar_pattern[i] = filtered_speakers[0]
        
        # Blockiere UI-Updates während der Erstellung der Widgets
        self.main_window.setUpdatesEnabled(False)
        
        # Schriftgröße für dynamisch erstellte Widgets
        font = QtGui.QFont()
        font.setPointSize(11)
        
        try:
            if is_stack:
                # Stack-Array-Logik
                for source_index in range(speaker_array.number_of_sources):
                    # Erstelle und konfiguriere Combobox
                    speaker_type_combo = QtWidgets.QComboBox(self.main_window)
                    speaker_type_combo.setObjectName(f"speaker_type_combo_{source_index + 1}")  # Wichtig: Setze einen eindeutigen Namen
                    speaker_type_combo.setFont(font)
                    speaker_type_combo.blockSignals(True)  # Blockiere Signale während der Konfiguration
                    speaker_type_combo.setFixedWidth(80)  # Schmaler für kompaktere Darstellung
                    speaker_type_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                    
                    # Füge alle gefilterten Items auf einmal hinzu (effizienter)
                    speaker_type_combo.addItems(filtered_speakers)
                    
                    # Setze den aktuellen Index
                    current_pattern = speaker_array.source_polar_pattern[source_index]
                    if current_pattern in speaker_name_to_index:
                        speaker_type_combo.setCurrentIndex(speaker_name_to_index[current_pattern])
                    
                    # Füge zum Layout hinzu
                    instance['gridLayout_sources'].addWidget(speaker_type_combo, 0, source_index + 3)
                    
                    # Verbinde Signal und aktiviere Signale wieder
                    speaker_type_combo.currentIndexChanged.connect(
                        lambda index, s_index=source_index, s_id=instance['id'], combo=speaker_type_combo: 
                        self.on_speaker_type_changed(combo, s_index, s_id)
                    )
                    speaker_type_combo.blockSignals(False)
            else:
                # Flown-Array-Logik
                for source_index in range(speaker_array.number_of_sources):
                    # Erstelle und konfiguriere Combobox für Lautsprechertyp
                    speaker_type_combo = QtWidgets.QComboBox(self.main_window)
                    speaker_type_combo.setObjectName(f"speaker_type_combo_{source_index + 1}")  # Wichtig: Setze einen eindeutigen Namen
                    speaker_type_combo.setFont(font)
                    speaker_type_combo.blockSignals(True)  # Blockiere Signale während der Konfiguration
                    speaker_type_combo.setFixedWidth(80)  # Schmaler für kompaktere Darstellung
                    speaker_type_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                    
                    # Füge alle gefilterten Items auf einmal hinzu (effizienter)
                    speaker_type_combo.addItems(filtered_speakers)
                    
                    # Setze den aktuellen Index
                    current_pattern = speaker_array.source_polar_pattern[source_index]
                    if current_pattern in speaker_name_to_index:
                        speaker_type_combo.setCurrentIndex(speaker_name_to_index[current_pattern])
                    
                    # Erstelle ein Label für die Nummerierung
                    number_label = QtWidgets.QLabel(f"{source_index + 1}", self.main_window)
                    number_label.setFont(font)
                    number_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    number_label.setFixedWidth(20)
                    
                    # Erstelle und konfiguriere Combobox für Winkel
                    angle_combo = QtWidgets.QComboBox(self.main_window)
                    angle_combo.setObjectName(f"angle_combo_{source_index + 1}")  # Wichtig: Setze einen eindeutigen Namen
                    angle_combo.setFont(font)
                    angle_combo.blockSignals(True)  # Blockiere Signale während der Konfiguration
                    angle_combo.setFixedWidth(80)  # Schmaler für kompaktere Darstellung
                    angle_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
                    
                    # Hole die verfügbaren Winkel für diesen Lautsprecher
                    angles = self.get_speaker_angles(current_pattern)
                    if source_index == 0:
                        angle_combo.addItem("0")
                        angle_combo.setEnabled(False)
                        angle_combo.setToolTip("Der oberste Lautsprecher verwendet immer 0°.")
                    else:
                        angle_combo.addItems(angles)
                    
                    # Initialisiere source_angle, wenn es noch nicht existiert
                    if not hasattr(speaker_array, 'source_angle') or speaker_array.source_angle is None:
                        speaker_array.source_angle = [None] * speaker_array.number_of_sources
                    else:
                        if isinstance(speaker_array.source_angle, np.ndarray):
                            speaker_array.source_angle = speaker_array.source_angle.tolist()
                        if len(speaker_array.source_angle) < speaker_array.number_of_sources:
                            speaker_array.source_angle.extend([None] * (speaker_array.number_of_sources - len(speaker_array.source_angle)))
                    
                    # Setze den aktuellen Winkel, wenn vorhanden
                    if source_index == 0:
                        speaker_array.source_angle[source_index] = 0.0
                        angle_combo.setCurrentIndex(0)
                    elif speaker_array.source_angle[source_index] is not None:
                        angle_str = str(speaker_array.source_angle[source_index])
                        index = angle_combo.findText(angle_str)
                        if index >= 0:
                            angle_combo.setCurrentIndex(index)
                    
                    # Verbinde Signal für Winkel-Combobox
                    if source_index != 0:
                        angle_combo.currentIndexChanged.connect(
                            lambda index, s_index=source_index, s_id=instance['id']: 
                            self.on_angle_changed(index, s_index, s_id)
                        )
                    angle_combo.blockSignals(False)
                    
                    # Füge das Label, die Lautsprecher-Combobox, das Winkel-Label und die Winkel-Combobox zum Grid-Layout hinzu
                    instance['gridLayout_sources'].addWidget(number_label, source_index + 1, 2)
                    instance['gridLayout_sources'].addWidget(speaker_type_combo, source_index + 1, 3)
                    instance['gridLayout_sources'].addWidget(angle_combo, source_index + 1, 4)
                    
                    # Verbinde Signal und aktiviere Signale wieder
                    speaker_type_combo.currentIndexChanged.connect(
                        lambda index, s_index=source_index, s_id=instance['id'], combo=speaker_type_combo: 
                        self.on_speaker_type_changed(combo, s_index, s_id)
                    )
                    speaker_type_combo.blockSignals(False)
        finally:
            # Aktiviere UI-Updates wieder
            self.main_window.setUpdatesEnabled(True)
            
        # Stelle sicher, dass alle Polar Patterns korrekt initialisiert sind
        self.ensure_polar_patterns_initialized(speaker_array)

    # @measure_time
    def ensure_polar_patterns_initialized(self, speaker_array):
        """
        Stellt sicher, dass alle Polar Patterns korrekt initialisiert sind.
        Ersetzt die einzelnen create_polar_pattern_handler Aufrufe.
        """
        if not hasattr(speaker_array, 'source_polar_pattern') or speaker_array.source_polar_pattern is None:
            # Initialisiere mit Standardwerten
            if len(self.container.data['speaker_names']) > 0:
                default_speaker = self.container.data['speaker_names'][0]
                speaker_array.source_polar_pattern = np.array([default_speaker] * speaker_array.number_of_sources, dtype=object)
            else:
                return
                
        # Stelle sicher, dass die Länge von source_polar_pattern korrekt ist
        if len(speaker_array.source_polar_pattern) < speaker_array.number_of_sources:
            # Erweitere das Array mit dem ersten Eintrag
            default_speaker = speaker_array.source_polar_pattern[0] if len(speaker_array.source_polar_pattern) > 0 else self.container.data['speaker_names'][0]
            speaker_array.source_polar_pattern = np.append(
                speaker_array.source_polar_pattern,
                [default_speaker] * (speaker_array.number_of_sources - len(speaker_array.source_polar_pattern))
            )



    # @measure_time
    def update_speaker_qlineedit(self, instance):
        """
        Aktualisiert die QLineEdit-Widgets für ein Lautsprecherarray.
        """
        """
        Erstellt und aktualisiert die Eingabefelder für jede Quelle im Speaker Setup.
        Für jede Quelle werden 5 Eingabefelder erstellt: X-Position, Y-Position, Z-Position, Azimuth, Delay und Gain.
        Unterscheidet zwischen Stack und Flown Arrays.
        """
        speaker_array = self.settings.get_speaker_array(instance['id'])
        if speaker_array is None:
            return
            
        # Prüfe, ob gridLayout_sources existiert
        if instance['gridLayout_sources'] is None:
            return
        
        # Prüfe, ob es sich um ein Stack oder Flown Array handelt
        is_stack = True  # Standardmäßig Stack annehmen
        if hasattr(speaker_array, 'configuration'):
            is_stack = speaker_array.configuration.lower() == "stack"
        
        # Stelle sicher, dass alle Arrays die richtige Größe haben
        speaker_array.update_sources(speaker_array.number_of_sources)
        
        # Schriftgröße für dynamisch erstellte Eingabefelder
        font = QtGui.QFont()
        font.setPointSize(11)
        
        if is_stack:
            # Für jede Quelle im Stack Array
            for source_index in range(speaker_array.number_of_sources):
                # X-Position Eingabefeld
                position_x_input = QLineEdit(self.main_window)
                position_x_input.setObjectName(f"position_x_input_{source_index + 1}")
                position_x_input.setFont(font)
                position_x_input.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
                position_x_input.setFixedWidth(70)
                position_x_input.setMaximumWidth(70)
                position_x_input.setText(f"{speaker_array.source_position_x[source_index]:.2f}")
                # Deaktiviere im symmetrischen Modus die zweite Hälfte
                if instance['state'] and source_index >= (len(speaker_array.source_position_x) + 1) // 2:
                    position_x_input.setEnabled(False)
                instance['gridLayout_sources'].addWidget(position_x_input, 1, source_index + 3)
                position_x_input.editingFinished.connect(
                    lambda i=source_index, edit=position_x_input: self.on_x_position_changed(edit, i, instance['id'])
                )

                # Y-Position Eingabefeld
                position_y_input = QLineEdit(self.main_window)
                position_y_input.setObjectName(f"position_y_input_{source_index + 1}")
                position_y_input.setFont(font)
                position_y_input.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
                position_y_input.setFixedWidth(70)
                position_y_input.setMaximumWidth(70)
                position_y_input.setText(f"{speaker_array.source_position_y[source_index]:.2f}")
                if instance['state'] and source_index >= (len(speaker_array.source_position_y) + 1) // 2:
                    position_y_input.setEnabled(False)
                instance['gridLayout_sources'].addWidget(position_y_input, 2, source_index + 3)
                position_y_input.editingFinished.connect(
                    lambda i=source_index, edit=position_y_input: self.on_y_position_changed(edit, i, instance['id'])
                )

                # Z-Position Eingabefeld
                position_z_input = QLineEdit(self.main_window)
                position_z_input.setObjectName(f"position_z_input_{source_index + 1}")
                position_z_input.setFont(font)
                position_z_input.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
                position_z_input.setFixedWidth(70)
                position_z_input.setMaximumWidth(70)
                
                # Prüfe, ob source_position_z existiert und initialisiere es bei Bedarf
                if not hasattr(speaker_array, 'source_position_z_stack') or speaker_array.source_position_z_stack is None:
                    speaker_array.source_position_z_stack = [0.0] * speaker_array.number_of_sources
                elif len(speaker_array.source_position_z_stack) < speaker_array.number_of_sources:
                    speaker_array.source_position_z_stack.extend([0.0] * (speaker_array.number_of_sources - len(speaker_array.source_position_z_stack)))
                
                position_z_input.setText(f"{speaker_array.source_position_z_stack[source_index]:.2f}")
                
                if instance['state'] and source_index >= (len(speaker_array.source_position_z_stack) + 1) // 2:
                    position_z_input.setEnabled(False)
                instance['gridLayout_sources'].addWidget(position_z_input, 3, source_index + 3)
                position_z_input.editingFinished.connect(
                    lambda i=source_index, edit=position_z_input: self.on_z_position_changed(edit, i, instance['id'])
                )

                # Azimuth Eingabefeld
                azimuth_input = QLineEdit(self.main_window)
                azimuth_input.setObjectName(f"azimuth_input_{source_index + 1}")
                azimuth_input.setFont(font)
                azimuth_input.setValidator(QDoubleValidator(-180, 180, 1))
                azimuth_input.setFixedWidth(70)
                azimuth_input.setText(f"{speaker_array.source_azimuth[source_index]:.1f}")
                if instance['state'] and source_index >= (len(speaker_array.source_azimuth) + 1) // 2:
                    azimuth_input.setEnabled(False)
                instance['gridLayout_sources'].addWidget(azimuth_input, 4, source_index + 3)
                azimuth_input.editingFinished.connect(
                    lambda i=source_index, edit=azimuth_input: self.on_source_azimuth_changed(edit, i, instance['id'])
                )

                # Delay Eingabefeld
                delay_input = QLineEdit(self.main_window)
                delay_input.setObjectName(f"delay_input_{source_index + 1}")
                delay_input.setFont(font)
                delay_input.setValidator(QDoubleValidator(0, float('inf'), 2))
                delay_input.setFixedWidth(70)
                delay_input.setText(f"{speaker_array.source_time[source_index]:.2f}")
                instance['gridLayout_sources'].addWidget(delay_input, 5, source_index + 3)
                delay_input.editingFinished.connect(
                    lambda i=source_index, edit=delay_input: self.on_source_delay_changed(edit, i, instance['id'])
                )

                # Gain Eingabefeld
                gain_input = QLineEdit(self.main_window)
                gain_input.setObjectName(f"gain_input_{source_index + 1}")
                gain_input.setFont(font)
                gain_input.setValidator(QDoubleValidator(-60, 10, 2))
                gain_input.setFixedWidth(70)
                gain_input.setText(f"{speaker_array.source_level[source_index]:.2f}")
                instance['gridLayout_sources'].addWidget(gain_input, 6, source_index + 3)
                gain_input.editingFinished.connect(
                    lambda i=source_index, edit=gain_input: self.on_source_level_changed(edit, i, instance['id'])
                )
        else:
            # Für jede Quelle im Flown Array
            for source_index in range(speaker_array.number_of_sources):
                # Stelle sicher, dass source_angle existiert und die richtige Größe hat
                if not hasattr(speaker_array, 'source_angle') or speaker_array.source_angle is None:
                    speaker_array.source_angle = np.zeros(speaker_array.number_of_sources)
                elif len(speaker_array.source_angle) < speaker_array.number_of_sources:
                    speaker_array.source_angle = np.append(
                        speaker_array.source_angle,
                        np.zeros(speaker_array.number_of_sources - len(speaker_array.source_angle))
                    )
                
                # Erstelle ein Label für die Nummerierung
                number_label = QtWidgets.QLabel(f"{source_index + 1}", self.main_window)
                number_label.setFont(font)
                number_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                number_label.setMaximumWidth(30)
                instance['gridLayout_sources'].addWidget(number_label, source_index + 1, 2)  # Nummerierung in Spalte 2
                
                # Stelle sicher, dass source_time existiert und die richtige Größe hat
                if not hasattr(speaker_array, 'source_time') or speaker_array.source_time is None:
                    speaker_array.source_time = np.zeros(speaker_array.number_of_sources)
                elif len(speaker_array.source_time) < speaker_array.number_of_sources:
                    speaker_array.source_time = np.append(
                        speaker_array.source_time,
                        np.zeros(speaker_array.number_of_sources - len(speaker_array.source_time))
                    )
                
                # Delay Eingabefeld
                delay_input = QLineEdit(self.main_window)
                delay_input.setObjectName(f"delay_input_{source_index + 1}")
                delay_input.setFont(font)
                delay_input.setValidator(QDoubleValidator(0, float('inf'), 2))
                delay_input.setFixedWidth(70)
                delay_input.setText(f"{speaker_array.source_time[source_index]:.2f}")
                instance['gridLayout_sources'].addWidget(delay_input, source_index + 1, 5)  # Delay in Spalte 5
                delay_input.editingFinished.connect(
                    lambda i=source_index, edit=delay_input: self.on_source_delay_changed(edit, i, instance['id'])
                )
                
                # Stelle sicher, dass source_level existiert und die richtige Größe hat
                if not hasattr(speaker_array, 'source_level') or speaker_array.source_level is None:
                    speaker_array.source_level = np.zeros(speaker_array.number_of_sources)
                elif len(speaker_array.source_level) < speaker_array.number_of_sources:
                    speaker_array.source_level = np.append(
                        speaker_array.source_level,
                        np.zeros(speaker_array.number_of_sources - len(speaker_array.source_level))
                    )
                
                # Gain Eingabefeld
                gain_input = QLineEdit(self.main_window)
                gain_input.setObjectName(f"gain_input_{source_index + 1}")
                gain_input.setFont(font)
                gain_input.setValidator(QDoubleValidator(-60, 10, 2))
                gain_input.setFixedWidth(70)
                gain_input.setText(f"{speaker_array.source_level[source_index]:.2f}")
                instance['gridLayout_sources'].addWidget(gain_input, source_index + 1, 6)  # Gain in Spalte 6
                gain_input.editingFinished.connect(
                    lambda i=source_index, edit=gain_input: self.on_source_level_changed(edit, i, instance['id'])
                )
        
        # Statt der ausführlichen Implementierung zur Aktivierung/Deaktivierung der Delay-Felder
        # nur die zentrale Funktion aufrufen
        speaker_array = self.settings.get_speaker_array(instance['id'])
        if speaker_array and hasattr(speaker_array, 'arc_shape'):
            is_manual = speaker_array.arc_shape.lower() == "manual"
            self.update_delay_fields_state(instance['id'], is_manual)
    

    def update_symmetry_checkbox(self, instance, is_stack):
        """
        Aktualisiert die Symmetrie-Checkbox basierend auf dem Array-Typ.
        Zeigt die Checkbox nur für Stack-Arrays an, nicht für Flown-Arrays.
        Args:
            instance (dict): Dictionary mit den UI-Elementen und Daten einer Speakerspecs-Instanz
            is_stack (bool): True, wenn es sich um ein Stack-Array handelt, sonst False
        """
        try:
            # Verwende die zentrale Checkbox aus dem Speaker Setup Tab
            if hasattr(self, 'symmetric_checkbox'):
                # Speichere Referenz zur Checkbox in der Instance
                instance['checkbox'] = self.symmetric_checkbox
                
                # Trenne alle vorherigen Verbindungen
                try:
                    self.symmetric_checkbox.stateChanged.disconnect()
                except:
                    pass
                
                # Verbinde die Checkbox mit der Instance
                self.symmetric_checkbox.stateChanged.connect(lambda state: self.checkbox_symmetric_checked(instance, state))
            
            # Setze die Sichtbarkeit basierend auf dem Array-Typ
                self.symmetric_checkbox.setVisible(is_stack)
                if hasattr(self, 'symmetric_label'):
                    self.symmetric_label.setVisible(is_stack)
                if hasattr(self, 'spacer_label'):
                    self.spacer_label.setVisible(is_stack)
                if hasattr(self, 'spacer_symmetric_line'):
                    self.spacer_symmetric_line.setVisible(is_stack)
                if hasattr(self, 'spacer_gain_polarity'):
                    self.spacer_gain_polarity.setVisible(True)  # Immer sichtbar
                
            # Aktualisiere den Status der Checkbox
                if is_stack and 'state' in instance and instance['state'] is not None:
                    self.symmetric_checkbox.blockSignals(True)
                    self.symmetric_checkbox.setChecked(instance['state'])
                    self.symmetric_checkbox.blockSignals(False)
                
        except Exception as e:
            print(f"Fehler in update_symmetry_checkbox: {e}")
            import traceback
            traceback.print_exc()

    # @measure_time
    def display_selected_speakerspecs(self):
        """
        Zeigt die Speakerspecs für das ausgewählte Array an und versteckt alle anderen.
        """
        # Hole das ausgewählte Item
        selected_items = self.sources_tree_widget.selectedItems()
        
        if not selected_items:
            return
            
        selected_item = selected_items[0]
        speaker_array_id = selected_item.data(0, Qt.UserRole)
        
        # Verstecke alle Speakerspecs
        for instance in self.speakerspecs_instance:
            if 'id' in instance:
                self.hide_speakerspecs(instance)
            
        # Zeige die ausgewählte Speakerspecs
        selected_instance = self.get_speakerspecs_instance(speaker_array_id)
        if selected_instance:
            self.show_speakerspecs(selected_instance)

    def create_polar_pattern_handler(self, source_index, speaker_array_id):
        """
        Erstellt einen Handler für die Änderung des Polarpatterns einer Quelle.
        
        Args:
            source_index (int): Index der Quelle
            speaker_array_id (int): ID des Lautsprecherarrays
        """
        speaker_array = self.settings.get_speaker_array(speaker_array_id)
        if speaker_array is None:
            return
            
        # Stelle sicher, dass source_polar_pattern existiert
        if not hasattr(speaker_array, 'source_polar_pattern') or speaker_array.source_polar_pattern is None:
            # Initialisiere mit Standardwerten
            if len(self.container.data['speaker_names']) > 0:
                default_speaker = self.container.data['speaker_names'][0]
                speaker_array.source_polar_pattern = np.array([default_speaker] * speaker_array.number_of_sources, dtype=object)
            else:
                return
                
        # Stelle sicher, dass die Länge von source_polar_pattern korrekt ist
        if len(speaker_array.source_polar_pattern) < speaker_array.number_of_sources:
            # Erweitere das Array mit dem ersten Eintrag
            default_speaker = speaker_array.source_polar_pattern[0] if len(speaker_array.source_polar_pattern) > 0 else self.container.data['speaker_names'][0]
            speaker_array.source_polar_pattern = np.append(
                speaker_array.source_polar_pattern,
                [default_speaker] * (speaker_array.number_of_sources - len(speaker_array.source_polar_pattern))
            )
            
        # Aktualisiere die Berechnungen
        self.main_window.update_speaker_array_calculations()

    def show_speakerspecs(self, instance):
        """
        Zeigt die Speakerspecs-Instanz an.
        
        Args:
            instance (dict): Die anzuzeigende Speakerspecs-Instanz
        """
        # Stelle sicher, dass die Scroll-Area sichtbar ist
        if 'scroll_area' in instance and instance['scroll_area'] is not None:
            instance['scroll_area'].setVisible(True)
            
            # Stelle sicher, dass der Inhalt der Scroll-Area aktualisiert wird
            if 'scroll_content' in instance and instance['scroll_content'] is not None:
                instance['scroll_content'].update()
                
            # Stelle sicher, dass das Layout aktualisiert wird
            if 'gridLayout_sources' in instance and instance['gridLayout_sources'] is not None:
                # Aktualisiere das Layout
                instance['gridLayout_sources'].update()
                
            # Aktualisiere die Scroll-Area
            instance['scroll_area'].update()
            
            # Stelle sicher, dass die Scroll-Area im Tab-Widget angezeigt wird
            if hasattr(self, 'speaker_tab_layout') and self.speaker_tab_layout is not None:
                # Prüfe, ob die Scroll-Area bereits im Layout ist
                found = False
                for i in range(self.speaker_tab_layout.count()):
                    item = self.speaker_tab_layout.itemAt(i)
                    if item and item.widget() == instance['scroll_area']:
                        found = True
                        break
                        
                if not found:
                    self.speaker_tab_layout.addWidget(instance['scroll_area'])
                    
            # Stelle sicher, dass das Tab-Widget aktualisiert wird
            if hasattr(self, 'tab_widget') and self.tab_widget is not None:
                self.tab_widget.update()
                
    def hide_speakerspecs(self, instance):
        """
        Versteckt die Speakerspecs-Instanz.
        
        Args:
            instance (dict): Die zu versteckende Speakerspecs-Instanz
        """
        if 'scroll_area' in instance and instance['scroll_area'] is not None:
            instance['scroll_area'].setVisible(False)

    def initialize_speakerspecs_instances(self, all_speaker_arrays):
        """
        Initialisiert die Speakerspecs-Instanzen für die angegebenen Speaker-Array-IDs.
        Optimiert, um redundante Updates zu vermeiden.
        """
        # Blockiere Signale während der Initialisierung
        self.sources_tree_widget.blockSignals(True)
        
        # Speichere das zuletzt ausgewählte Item
        last_selected_item = None
        
        # Deaktiviere temporär die Verbindung zwischen add_speakerspecs_instance und update_input_fields
        original_add_speakerspecs = self.add_speakerspecs_instance
        
        # Temporäre Ersatzfunktion, die keine Signale auslöst
        def temp_add_speakerspecs(speaker_array_id, instance):
            instance['id'] = speaker_array_id
            self.speakerspecs_instance.append(instance)
        
        # Ersetze die Methode temporär
        self.add_speakerspecs_instance = temp_add_speakerspecs
        
        # Sammle alle Instanzen, die wir am Ende aktualisieren werden
        instances_to_update = []
        
        for array_id in all_speaker_arrays:
            speaker_array = self.settings.get_speaker_array(array_id)
            
            if speaker_array:
                # Erstelle Tree Item
                array_name = speaker_array.name
                new_array_item = QTreeWidgetItem(self.sources_tree_widget, [array_name])
                new_array_item.setFlags(new_array_item.flags() | Qt.ItemIsEditable)
                new_array_item.setData(0, Qt.UserRole, array_id)
                new_array_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
                
                # Setze Checkboxen
                mute_checkbox = self.create_checkbox(speaker_array.mute)
                hide_checkbox = self.create_checkbox(speaker_array.hide)
                self.sources_tree_widget.setItemWidget(new_array_item, 1, mute_checkbox)
                self.sources_tree_widget.setItemWidget(new_array_item, 2, hide_checkbox)
                self._update_color_indicator(new_array_item, speaker_array)
                
                # Verbinde Signale mit verzögerter Ausführung
                mute_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_mute_state(id, state))
                hide_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_hide_state(id, state))
                
                # Erstelle Speakerspecs Instance
                speakerspecs_instance = self.create_speakerspecs_instance(array_id)
                
                # Verwende die temporäre Methode, die keine Signale auslöst
                temp_add_speakerspecs(array_id, speakerspecs_instance)
                
                # Sammle die Instanz für späteres Update
                instances_to_update.append(speakerspecs_instance)
                
                # Aktualisiere UI mit den bestehenden Werten, aber nur einmal pro Array
                self.number_of_sources_edit.setText(str(speaker_array.number_of_sources))
                self.array_length_edit.setText(str(speaker_array.source_length))
                self.delay_edit.setText(str(speaker_array.delay))
                self.gain_edit.setText(str(speaker_array.gain))
                
                # Speichere das Item für spätere Auswahl
                last_selected_item = new_array_item
        
        # Stelle die ursprüngliche Methode wieder her
        self.add_speakerspecs_instance = original_add_speakerspecs
        
        # Entsperre Signale nach der Initialisierung
        self.sources_tree_widget.blockSignals(False)
        
        # Selektiere das letzte Item erst nach der Initialisierung aller Arrays
        if last_selected_item:
            self.sources_tree_widget.setCurrentItem(last_selected_item)
            # Aktualisiere den Plot nur einmal am Ende
            self.main_window.update_speaker_array_calculations()
        
        # Validiere alle Checkboxen nach Initialisierung
        self.validate_all_checkboxes()
        
        # Passe Spaltenbreite an den Inhalt an
        self.adjust_column_width_to_content()
    
    def _on_selection_changed(self):
        """
        Zentraler Handler für alle Auswahländerungen im Sources-TreeWidget.
        OPTIMIERUNG: Ersetzt 9 separate Signal-Verbindungen durch einen koordinierten Handler.
        Verhindert redundante Ausführungen und Signal-Loops.
        """
        # Blockiere Signale während Updates, um Signal-Loops zu vermeiden
        if not hasattr(self, 'sources_tree_widget') or self.sources_tree_widget is None:
            return
        
        self.sources_tree_widget.blockSignals(True)
        try:
            # 1. Zeige Tabs und aktualisiere Eingabefelder
            # show_sources_tab() ruft bereits viele update_*_input_fields() Methoden intern auf
            self.show_sources_tab()
            
            # 2. Aktualisiere Beamsteering/Windowing Plots (wird nicht in show_sources_tab() aufgerufen)
            self.update_beamsteering_windowing_plots()
            
            # 3. Aktualisiere Highlight-IDs für 3D-Plot (rote Umrandung)
            self._handle_sources_tree_selection_changed()
        finally:
            self.sources_tree_widget.blockSignals(False)
    
    def _handle_sources_tree_selection_changed(self):
        """Reagiert auf Auswahländerungen im Sources-Tree:
        - Setzt active_speaker_array_highlight_ids (Liste) für den 3D-Plot
        - Unterstützt mehrere ausgewählte Items und Gruppen
        - Triggert update_overlays() für rote Umrandung.
        """
        if not hasattr(self, 'sources_tree_widget'):
            return
        
        # Hole alle ausgewählten Items (nicht nur currentItem)
        selected_items = self.sources_tree_widget.selectedItems()
        
        if not selected_items:
            # Keine Auswahl → keine Highlights
            setattr(self.settings, "active_speaker_array_highlight_id", None)
            setattr(self.settings, "active_speaker_array_highlight_ids", [])
            setattr(self.settings, "active_speaker_highlight_indices", [])
            # Aktualisiere Overlays
            if hasattr(self, 'main_window') and self.main_window:
                if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "draw_spl_plotter"):
                    draw_spl = self.main_window.draw_plots.draw_spl_plotter
                    if hasattr(draw_spl, "update_overlays"):
                        try:
                            draw_spl.update_overlays(self.settings, self.container)
                        except Exception:  # noqa: BLE001
                            pass
            return
        
        # Sammle alle Array-IDs aus ausgewählten Items
        highlight_array_ids = []
        
        for selected_item in selected_items:
            try:
                item_type = selected_item.data(0, Qt.UserRole + 1)
            except RuntimeError:
                # Item wurde gelöscht, überspringe
                continue
            
            if item_type == "group":
                # Für Gruppen: Sammle alle Arrays der Gruppe (rekursiv alle Child-Items)
                group_name = selected_item.text(0)
                
                # Finde die Gruppe in settings.speaker_array_groups
                group_id = None
                if hasattr(self.settings, 'speaker_array_groups'):
                    for gid, gdata in self.settings.speaker_array_groups.items():
                        if gdata.get('name') == group_name:
                            group_id = gid
                            break
                
                if group_id:
                    # Hole alle Child-Array-IDs aus der Gruppe
                    group_data = self.settings.speaker_array_groups.get(group_id, {})
                    child_array_ids = group_data.get('child_array_ids', [])
                    
                    # Füge alle Child-Array-IDs hinzu
                    for array_id in child_array_ids:
                        array_id_str = str(array_id)
                        if array_id_str not in highlight_array_ids:
                            highlight_array_ids.append(array_id_str)
                
                # Sammle auch direkt aus TreeWidget-Childs (rekursiv für verschachtelte Strukturen)
                def collect_child_arrays(parent_item):
                    """Rekursiv sammelt alle Array-IDs aus Child-Items"""
                    child_ids = []
                    try:
                        for i in range(parent_item.childCount()):
                            child_item = parent_item.child(i)
                            if child_item is None:
                                continue
                            
                            try:
                                child_type = child_item.data(0, Qt.UserRole + 1)
                                if child_type == "group":
                                    # Rekursiv für verschachtelte Gruppen (falls möglich)
                                    child_ids.extend(collect_child_arrays(child_item))
                                else:
                                    # Array-Item
                                    child_array_id = child_item.data(0, Qt.UserRole)
                                    if child_array_id is not None:
                                        child_ids.append(str(child_array_id))
                            except RuntimeError:
                                # Child-Item wurde gelöscht, überspringe
                                continue
                    except RuntimeError:
                        # Parent-Item wurde gelöscht
                        pass
                    return child_ids
                
                # Sammle alle Child-Arrays rekursiv
                child_array_ids_from_tree = collect_child_arrays(selected_item)
                for array_id_str in child_array_ids_from_tree:
                    if array_id_str not in highlight_array_ids:
                        highlight_array_ids.append(array_id_str)
            else:
                # Einzelnes Array
                try:
                    speaker_array_id = selected_item.data(0, Qt.UserRole)
                    if speaker_array_id is not None:
                        array_id_str = str(speaker_array_id)
                        if array_id_str not in highlight_array_ids:
                            highlight_array_ids.append(array_id_str)
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        
        # Setze Highlight-IDs
        if highlight_array_ids:
            # Für Rückwärtskompatibilität: Setze auch active_speaker_array_highlight_id auf das erste Element
            setattr(self.settings, "active_speaker_array_highlight_id", highlight_array_ids[0])
            setattr(self.settings, "active_speaker_array_highlight_ids", highlight_array_ids)
            setattr(self.settings, "active_speaker_highlight_indices", [])
        else:
            setattr(self.settings, "active_speaker_array_highlight_id", None)
            setattr(self.settings, "active_speaker_array_highlight_ids", [])
            setattr(self.settings, "active_speaker_highlight_indices", [])
        
        # Overlays im 3D-Plot aktualisieren (nur visuell, keine Neuberechnung)
        if hasattr(self, 'main_window') and self.main_window:
            if hasattr(self.main_window, "draw_plots") and hasattr(self.main_window.draw_plots, "draw_spl_plotter"):
                draw_spl = self.main_window.draw_plots.draw_spl_plotter
                if hasattr(draw_spl, "update_overlays"):
                    try:
                        draw_spl.update_overlays(self.settings, self.container)
                    except Exception:  # noqa: BLE001
                        import traceback
                        traceback.print_exc()
    
    def eventFilter(self, obj, event):
        """Event-Filter für TreeWidget und DockWidget, um Klicks auf leeres Feld und resize-Events zu erkennen."""
        from PyQt5.QtCore import QEvent
        
        # Resize-Events für DockWidget erfassen
        if hasattr(self, 'sources_dockWidget') and self.sources_dockWidget and obj == self.sources_dockWidget:
            if event.type() == QEvent.Resize:
                # Keine spezielle Behandlung mehr, Event normal weiterreichen
                return False  # Weiterleiten an Standard-Handler
        
        # Prüfe, ob das Widget noch existiert, bevor darauf zugegriffen wird
        if not hasattr(self, 'sources_tree_widget') or self.sources_tree_widget is None:
            return super().eventFilter(obj, event)
        
        try:
            if obj == self.sources_tree_widget.viewport():
                if event.type() == QEvent.MouseButtonPress:
                    # Prüfe ob auf ein Item geklickt wurde
                    item = self.sources_tree_widget.itemAt(event.pos())
                    if item is None:
                        # Klick auf leeres Feld - entferne Auswahl
                        self.sources_tree_widget.clearSelection()
                        # Setze currentItem auf None
                        self.sources_tree_widget.setCurrentItem(None)
                        # Trigger selection changed handler
                        self._handle_sources_tree_selection_changed()
                        return True  # Event behandelt
        except RuntimeError:
            # Widget wurde gelöscht - ignoriere das Event
            return super().eventFilter(obj, event)
        
        return super().eventFilter(obj, event)

    def _tab_exists(self, tab_name):
        """
        Prüft, ob ein Tab mit dem angegebenen Namen bereits existiert.
        
        Args:
            tab_name: Der Name des Tabs (wie er in addTab verwendet wird)
            
        Returns:
            bool: True wenn Tab existiert, False sonst
        """
        if not hasattr(self, 'tab_widget') or self.tab_widget is None:
            return False
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_name:
                return True
        return False
    
    def _get_tab_widget_by_name(self, tab_name):
        """
        Gibt das Widget eines Tabs zurück, wenn es existiert.
        
        Args:
            tab_name: Der Name des Tabs
            
        Returns:
            QWidget oder None: Das Tab-Widget, falls gefunden
        """
        if not hasattr(self, 'tab_widget') or self.tab_widget is None:
            return None
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_name:
                return self.tab_widget.widget(i)
        return None
    
    def _extract_plot_from_tab(self, tab_widget, plot_class):
        """
        Extrahiert einen Plot aus einem existierenden Tab-Widget.
        
        Args:
            tab_widget: Das Tab-Widget
            plot_class: Die Klasse des Plots (BeamsteeringPlot oder WindowingPlot)
            
        Returns:
            Plot-Instanz oder None: Der Plot, falls gefunden
        """
        if tab_widget is None:
            return None
        
        # Durchsuche alle Widgets im Tab nach dem Plot
        def find_plot(widget):
            if isinstance(widget, plot_class):
                return widget
            if hasattr(widget, 'layout'):
                layout = widget.layout()
                if layout:
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item:
                            widget_item = item.widget()
                            if widget_item:
                                result = find_plot(widget_item)
                                if result:
                                    return result
            return None
        
        return find_plot(tab_widget)
    
    # @measure_time
    def show_sources_tab(self):
        """
        Zeigt die entsprechenden Tabs basierend auf dem ausgewählten Array-Typ an.
        Wird aufgerufen, wenn ein Array im TreeWidget ausgewählt wird.
        OPTIMIERUNG: Tabs werden nur erstellt, wenn sie noch nicht existieren.
        """
        if not hasattr(self, 'tab_widget') or self.tab_widget is None:
            # Erstelle das TabWidget, falls es noch nicht existiert
            self.tab_widget = QTabWidget()
            self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Füge TabWidget zum rechten Widget hinzu
            if hasattr(self, 'right_side_widget'):
                right_layout = QVBoxLayout()
                right_layout.addWidget(self.tab_widget)
                
                # Entferne alle vorhandenen Layouts
                if self.right_side_widget.layout():
                    QWidget().setLayout(self.right_side_widget.layout())
                    
                self.right_side_widget.setLayout(right_layout)
    
        # Prüfe, ob ein Array ausgewählt ist
        selected_item = self.sources_tree_widget.currentItem() if hasattr(self, 'sources_tree_widget') else None

        if not selected_item:
            return
        
        # Prüfe, ob es sich um eine Gruppe handelt
        is_group = selected_item.data(0, Qt.UserRole + 1) == "group"
        
        if is_group:
            # Zeige Gruppen-UI
            self.create_group_tab(selected_item)
            return
            
        speaker_array_id = selected_item.data(0, Qt.UserRole)
        speaker_array = self.settings.get_speaker_array(speaker_array_id)
        
        if not speaker_array:
            print(f"Kein gültiges Lautsprecherarray mit ID {speaker_array_id} gefunden")
            return
            
        # Prüfe, ob das Array eine Konfiguration hat (Stack oder Flown)
        if hasattr(speaker_array, 'configuration'):
            configuration = speaker_array.configuration.lower()
            # print(f"Lautsprecherarray-Typ: {configuration.upper()} (ID: {speaker_array_id}, Name: {speaker_array.name})")
            
            # OPTIMIERUNG: Tabs nur erstellen, wenn sie noch nicht existieren
            # Erstelle nur die relevanten Tabs basierend auf der Konfiguration
            if configuration == "stack":
                # Für Stack-Arrays alle Tabs erstellen und aktivieren
                if not self._tab_exists("Speaker Setup"):
                    self.create_speaker_tab_stack()
                if not self._tab_exists("Beamsteering"):
                    self.create_beamsteering_tab()
                else:
                    # Tab existiert bereits - extrahiere Plot falls vorhanden
                    beamsteering_tab = self._get_tab_widget_by_name("Beamsteering")
                    if beamsteering_tab and self.beamsteering_plot is None:
                        from Module_LFO.Modules_Plot.PlotBeamsteering import BeamsteeringPlot
                        self.beamsteering_plot = self._extract_plot_from_tab(beamsteering_tab, BeamsteeringPlot)
                if not self._tab_exists("Windowing"):
                    self.create_windowing_tab()
                else:
                    # Tab existiert bereits - extrahiere Plot falls vorhanden
                    windowing_tab = self._get_tab_widget_by_name("Windowing")
                    if windowing_tab and self.windowing_plot is None:
                        from Module_LFO.Modules_Plot.PlotWindowing import WindowingPlot
                        self.windowing_plot = self._extract_plot_from_tab(windowing_tab, WindowingPlot)
                
                # Alle Tabs aktivieren
                for i in range(self.tab_widget.count()):
                    self.tab_widget.setTabEnabled(i, True)
            else:
                # Für Flown-Arrays alle Tabs erstellen
                if not self._tab_exists("Speaker Setup"):
                    self.create_speaker_tab_flown()
                if not self._tab_exists("Beamsteering"):
                    self.create_beamsteering_tab()
                else:
                    # Tab existiert bereits - extrahiere Plot falls vorhanden
                    beamsteering_tab = self._get_tab_widget_by_name("Beamsteering")
                    if beamsteering_tab and self.beamsteering_plot is None:
                        from Module_LFO.Modules_Plot.PlotBeamsteering import BeamsteeringPlot
                        self.beamsteering_plot = self._extract_plot_from_tab(beamsteering_tab, BeamsteeringPlot)
                if not self._tab_exists("Windowing"):
                    self.create_windowing_tab()
                else:
                    # Tab existiert bereits - extrahiere Plot falls vorhanden
                    windowing_tab = self._get_tab_widget_by_name("Windowing")
                    if windowing_tab and self.windowing_plot is None:
                        from Module_LFO.Modules_Plot.PlotWindowing import WindowingPlot
                        self.windowing_plot = self._extract_plot_from_tab(windowing_tab, WindowingPlot)
                
                # Für Flown-Arrays nur den Speaker Setup Tab anzeigen
                if self.tab_widget.count() > 1:
                    self.tab_widget.setCurrentIndex(0)  # Wechsle zum Speaker Setup Tab
                    
                    # Deaktiviere die anderen Tabs
                    for i in range(1, self.tab_widget.count()):
                        self.tab_widget.setTabEnabled(i, False)
        else:
            print(f"Lautsprecherarray ohne Konfiguration (ID: {speaker_array_id}, Name: {speaker_array.name})")
            # Fallback für Arrays ohne Konfiguration (alte Arrays)
            # Erstelle alle Tabs
            if not self._tab_exists("Speaker Setup"):
                self.create_speaker_tab_stack()
            if not self._tab_exists("Beamsteering"):
                self.create_beamsteering_tab()
            else:
                # Tab existiert bereits - extrahiere Plot falls vorhanden
                beamsteering_tab = self._get_tab_widget_by_name("Beamsteering")
                if beamsteering_tab and self.beamsteering_plot is None:
                    from Module_LFO.Modules_Plot.PlotBeamsteering import BeamsteeringPlot
                    self.beamsteering_plot = self._extract_plot_from_tab(beamsteering_tab, BeamsteeringPlot)
            if not self._tab_exists("Windowing"):
                self.create_windowing_tab()
            else:
                # Tab existiert bereits - extrahiere Plot falls vorhanden
                windowing_tab = self._get_tab_widget_by_name("Windowing")
                if windowing_tab and self.windowing_plot is None:
                    from Module_LFO.Modules_Plot.PlotWindowing import WindowingPlot
                    self.windowing_plot = self._extract_plot_from_tab(windowing_tab, WindowingPlot)
            
            # Standardmäßig alle Tabs aktivieren
            for i in range(self.tab_widget.count()):
                self.tab_widget.setTabEnabled(i, True)
        
        # Wichtig: Aktualisiere die Anzeige der Speakerspecs
        self.display_selected_speakerspecs()
        
        # Aktualisiere die Eingabefelder für die ausgewählte Instanz
        speakerspecs_instance = self.get_speakerspecs_instance(speaker_array_id)
        if speakerspecs_instance:
            self.update_input_fields(speakerspecs_instance)
            
            # Aktualisiere auch die anderen Eingabefelder
            self.update_sources_input_fields()
            self.update_source_length_input_fields()
            self.update_array_position_input_fields()
            self.update_beamsteering_input_fields()
            self.update_windowing_input_fields()
            self.update_gain_delay_input_fields()

    def create_group_tab(self, group_item):
        """Erstellt die UI für Gruppen-Einstellungen"""
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
        
        # Bereich 2: Change source settings
        source_settings_group = QGroupBox("Change source settings")
        source_settings_group.setFont(font)
        source_settings_layout = QGridLayout()
        source_settings_layout.setVerticalSpacing(3)
        source_settings_layout.setContentsMargins(5, 10, 5, 5)
        
        # Delay
        delay_label = QLabel("Delay (ms)")
        delay_label.setFont(font)
        self.group_delay_edit = QLineEdit()
        self.group_delay_edit.setFont(font)
        self.group_delay_edit.setFixedHeight(18)
        self.group_delay_edit.setValidator(QDoubleValidator(-float("inf"), float("inf"), 2))
        self.group_delay_edit.setText("0.00")
        delay_label.setFixedWidth(150)
        self.group_delay_edit.setFixedWidth(40)
        source_settings_layout.addWidget(delay_label, 0, 0)
        source_settings_layout.addWidget(self.group_delay_edit, 0, 1)
        
        # Gain
        gain_label = QLabel("Gain (dB)")
        gain_label.setFont(font)
        self.group_gain_edit = QLineEdit()
        self.group_gain_edit.setFont(font)
        self.group_gain_edit.setFixedHeight(18)
        self.group_gain_edit.setValidator(QDoubleValidator(-60, 10, 1))
        self.group_gain_edit.setText("0.0")
        gain_label.setFixedWidth(150)
        self.group_gain_edit.setFixedWidth(40)
        source_settings_layout.addWidget(gain_label, 1, 0)
        source_settings_layout.addWidget(self.group_gain_edit, 1, 1)
        
        source_settings_group.setLayout(source_settings_layout)
        scroll_layout.addWidget(source_settings_group)
        
        # Spacer
        scroll_layout.addStretch()
        
        # Apply Changes Button
        apply_button = QPushButton("Apply Changes")
        apply_button.setFont(font)
        apply_button.setFixedHeight(30)
        apply_button.clicked.connect(lambda: self.apply_group_changes(group_item))
        scroll_layout.addWidget(apply_button)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Lade aktuelle Werte aus der Gruppe
        self.load_group_values(group_item)

    def save_original_positions_for_group(self, group_item):
        """Speichert die ursprünglichen Positionen für alle Arrays in einer Gruppe"""
        if not group_item:
            return
        
        # Finde die Gruppen-ID
        group_name = group_item.text(0)
        group_id = None
        for gid, gdata in self.settings.speaker_array_groups.items():
            if gdata.get('name') == group_name:
                group_id = gid
                break
        
        if group_id is None:
            # Erstelle neue Gruppen-ID
            import uuid
            group_id = str(uuid.uuid4())
            self.settings.speaker_array_groups[group_id] = {
                'name': group_name,
                'mute': False,
                'hide': False,
                'child_array_ids': []
            }
        
        # Hole oder erstelle ursprüngliche Positionen
        original_positions = self.settings.speaker_array_groups[group_id].get('original_array_positions', {})
        
        # Speichere ursprüngliche Positionen für alle Child-Arrays
        child_count = group_item.childCount()
        for i in range(child_count):
            child_item = group_item.child(i)
            child_array_id = child_item.data(0, Qt.UserRole)
            
            if isinstance(child_array_id, dict):
                child_array_id = child_array_id.get('id')
            
            if child_array_id is not None and child_array_id not in original_positions:
                speaker_array = self.settings.get_speaker_array(child_array_id)
                if speaker_array:
                    original_positions[child_array_id] = {
                        'x': getattr(speaker_array, 'array_position_x', 0.0),
                        'y': getattr(speaker_array, 'array_position_y', 0.0),
                        'z': getattr(speaker_array, 'array_position_z', 0.0),
                        'delay': getattr(speaker_array, 'delay', 0.0),
                        'gain': getattr(speaker_array, 'gain', 0.0)
                    }
        
        # Speichere aktualisierte ursprüngliche Positionen
        self.settings.speaker_array_groups[group_id]['original_array_positions'] = original_positions

    def load_group_values(self, group_item):
        """Lädt die aktuellen Werte der Gruppe in die Eingabefelder"""
        if not group_item:
            return
        
        # Finde die Gruppen-ID
        group_name = group_item.text(0)
        group_id = None
        for gid, gdata in self.settings.speaker_array_groups.items():
            if gdata.get('name') == group_name:
                group_id = gid
                break
        
        if group_id is None:
            # Initialisiere mit Standardwerten
            self.group_rel_x_edit.setText("0.00")
            self.group_rel_y_edit.setText("0.00")
            self.group_rel_z_edit.setText("0.00")
            self.group_delay_edit.setText("0.00")
            self.group_gain_edit.setText("0.0")
            return
        
        group_data = self.settings.speaker_array_groups[group_id]
        
        # Lade relative Positionen
        rel_pos = group_data.get('relative_position', {'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.group_rel_x_edit.setText(f"{rel_pos.get('x', 0.0):.2f}")
        self.group_rel_y_edit.setText(f"{rel_pos.get('y', 0.0):.2f}")
        self.group_rel_z_edit.setText(f"{rel_pos.get('z', 0.0):.2f}")
        
        # Lade Delay und Gain
        self.group_delay_edit.setText(f"{group_data.get('relative_delay', 0.0):.2f}")
        self.group_gain_edit.setText(f"{group_data.get('relative_gain', 0.0):.1f}")

    def apply_group_changes(self, group_item):
        """Wendet die Gruppen-Änderungen auf alle Child-Sources an"""
        if not group_item:
            return
        
        # Hole die eingegebenen Werte
        try:
            rel_x = float(self.group_rel_x_edit.text())
            rel_y = float(self.group_rel_y_edit.text())
            rel_z = float(self.group_rel_z_edit.text())
            rel_delay = float(self.group_delay_edit.text())
            rel_gain = float(self.group_gain_edit.text())
        except ValueError:
            print("Fehler: Ungültige Eingabewerte")
            return
        
        # Finde die Gruppen-ID
        group_name = group_item.text(0)
        group_id = None
        for gid, gdata in self.settings.speaker_array_groups.items():
            if gdata.get('name') == group_name:
                group_id = gid
                break
        
        if group_id is None:
            # Erstelle neue Gruppen-ID
            import uuid
            group_id = str(uuid.uuid4())
            self.settings.speaker_array_groups[group_id] = {
                'name': group_name,
                'mute': False,
                'hide': False,
                'child_array_ids': []
            }
        
        # Speichere die relativen Werte in der Gruppe
        self.settings.speaker_array_groups[group_id]['relative_position'] = {
            'x': rel_x,
            'y': rel_y,
            'z': rel_z
        }
        self.settings.speaker_array_groups[group_id]['relative_delay'] = rel_delay
        self.settings.speaker_array_groups[group_id]['relative_gain'] = rel_gain
        
        # Hole ursprüngliche Array-Positionen (sollten bereits beim Hinzufügen zur Gruppe gespeichert sein)
        original_positions = self.settings.speaker_array_groups[group_id].get('original_array_positions', {})
        
        # Sammle alle Child-Array-IDs
        child_count = group_item.childCount()
        child_array_ids = []
        for i in range(child_count):
            child_item = group_item.child(i)
            child_array_id = child_item.data(0, Qt.UserRole)
            
            if isinstance(child_array_id, dict):
                child_array_id = child_array_id.get('id')
            
            if child_array_id is not None:
                child_array_ids.append(child_array_id)
                
                # Falls ursprüngliche Positionen noch nicht gespeichert sind, speichere sie jetzt
                if child_array_id not in original_positions:
                    speaker_array = self.settings.get_speaker_array(child_array_id)
                    if speaker_array:
                        original_positions[child_array_id] = {
                            'x': getattr(speaker_array, 'array_position_x', 0.0),
                            'y': getattr(speaker_array, 'array_position_y', 0.0),
                            'z': getattr(speaker_array, 'array_position_z', 0.0),
                            'delay': getattr(speaker_array, 'delay', 0.0),
                            'gain': getattr(speaker_array, 'gain', 0.0)
                        }
        
        # Speichere aktualisierte ursprüngliche Positionen (nur für neue Arrays)
        if original_positions:
            self.settings.speaker_array_groups[group_id]['original_array_positions'] = original_positions
        
        # Wende die Änderungen auf alle Child-Sources an
        # WICHTIG: Die relativen Werte werden zu den AKTUELLEN Array-Werten hinzugefügt (relativ),
        # nicht zu den ursprünglichen Werten. Dies ermöglicht mehrfaches "Apply" mit kumulativen Effekten.
        for child_array_id in child_array_ids:
            speaker_array = self.settings.get_speaker_array(child_array_id)
            if speaker_array:
                # Prüfe, ob es sich um ein Flown-System handelt
                is_flown = hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown"
                
                # Addiere relative Positionen zu den aktuellen Array-Positionen
                speaker_array.array_position_x = getattr(speaker_array, 'array_position_x', 0.0) + rel_x
                speaker_array.array_position_y = getattr(speaker_array, 'array_position_y', 0.0) + rel_y
                speaker_array.array_position_z = getattr(speaker_array, 'array_position_z', 0.0) + rel_z
                
                # Bei Flown-Systemen: Addiere relative Positionen auch zu source_position_x/y/z_flown
                # (wie in on_ArrayX/Y/Z_changed für Flown-Systeme)
                if is_flown:
                    # Stelle sicher, dass source_position_x/y existieren und die richtige Länge haben
                    if not hasattr(speaker_array, 'source_position_x') or speaker_array.source_position_x is None:
                        speaker_array.source_position_x = np.zeros(speaker_array.number_of_sources, dtype=float)
                    elif len(speaker_array.source_position_x) != speaker_array.number_of_sources:
                        speaker_array.source_position_x = np.full(speaker_array.number_of_sources, speaker_array.array_position_x, dtype=float)
                    else:
                        speaker_array.source_position_x = speaker_array.source_position_x + rel_x
                    
                    if not hasattr(speaker_array, 'source_position_y') or speaker_array.source_position_y is None:
                        speaker_array.source_position_y = np.zeros(speaker_array.number_of_sources, dtype=float)
                    elif len(speaker_array.source_position_y) != speaker_array.number_of_sources:
                        speaker_array.source_position_y = np.full(speaker_array.number_of_sources, speaker_array.array_position_y, dtype=float)
                    else:
                        speaker_array.source_position_y = speaker_array.source_position_y + rel_y
                    
                    if not hasattr(speaker_array, 'source_position_z_flown') or speaker_array.source_position_z_flown is None:
                        speaker_array.source_position_z_flown = np.zeros(speaker_array.number_of_sources, dtype=float)
                    elif len(speaker_array.source_position_z_flown) != speaker_array.number_of_sources:
                        speaker_array.source_position_z_flown = np.full(speaker_array.number_of_sources, speaker_array.array_position_z, dtype=float)
                    else:
                        speaker_array.source_position_z_flown = speaker_array.source_position_z_flown + rel_z
                
                # Addiere relativen Delay zu dem aktuellen Array-Delay-Wert
                current_delay = getattr(speaker_array, 'delay', 0.0)
                speaker_array.delay = current_delay + rel_delay
                self.settings.update_speaker_array_delay(child_array_id, speaker_array.delay)
                
                # Addiere relativen Gain zu dem aktuellen Array-Gain-Wert
                current_gain = getattr(speaker_array, 'gain', 0.0)
                speaker_array.gain = current_gain + rel_gain
                self.settings.update_speaker_array_gain(child_array_id, speaker_array.gain)
        
        # Aktualisiere Berechnungen für alle Arrays in der Gruppe
        # WICHTIG: Leere alle Position-Hashes, damit alle Arrays neu berechnet werden
        self.main_window.calculation_handler.clear_speaker_position_hashes()
        
        # Stelle sicher, dass die Berechnungen für alle Arrays in der Gruppe ausgelöst werden
        # Die Soundfield-Berechnungen berücksichtigen automatisch alle nicht-muted/nicht-hidden Arrays
        # Aber wir müssen sicherstellen, dass die Positionen für alle Arrays in der Gruppe neu berechnet werden
        for child_array_id in child_array_ids:
            speaker_array = self.settings.get_speaker_array(child_array_id)
            if speaker_array:
                # Berechne Positionen für jedes Array in der Gruppe
                self.main_window.speaker_position_calculator(speaker_array)
        
        # Aktualisiere die Soundfield-Berechnungen (berücksichtigt alle Arrays)
        self.main_window.update_speaker_array_calculations()

    # @measure_time
    def update_beamsteering_input_fields(self):
        selected_item = self.sources_tree_widget.currentItem()
        if selected_item:
            speaker_array_id = selected_item.data(0, Qt.UserRole)
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                self.ArcAngle.blockSignals(True)
                self.ArcAngle.setText(f"{speaker_array.arc_angle:.1f}")
                self.ArcAngle.blockSignals(False)
    
                self.ArcScaleFactor.blockSignals(True)
                self.ArcScaleFactor.setText(f"{getattr(speaker_array, 'arc_scale_factor', 0.32):.2f}")
                self.ArcScaleFactor.blockSignals(False)
    
                self.ArcShape.blockSignals(True)
                index = self.ArcShape.findText(speaker_array.arc_shape)
                self.ArcShape.setCurrentIndex(index if index != -1 else 0)
                self.ArcShape.blockSignals(False)

                # Aktualisiere den Arc-Info-Text
                self.update_arc_info_text(speaker_array.arc_shape)


    # @measure_time
    def update_beamsteering_plot(self, speakerspecs_instance):
        selected_item = self.sources_tree_widget.currentItem()
        if selected_item:
            speaker_array_id = selected_item.data(0, Qt.UserRole)
            
            if isinstance(speaker_array_id, dict):
                speaker_array_id = speaker_array_id.get('id')
            elif not isinstance(speaker_array_id, (int, str)):
                return

            # Führe die Beamsteering-Berechnung durch, damit virtuelle Positionen korrekt sind (ohne Array-Offset)
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                from Module_LFO.Modules_Calculate.BeamSteering import BeamSteering
                beamsteering = BeamSteering(speaker_array, self.container.data, self.settings)
                beamsteering.calculate(speaker_array_id)

            if self.beamsteering_plot:
                self.beamsteering_plot.beamsteering_plot(speaker_array_id)

    # @measure_time
    def update_windowing_plot(self, windowing, speaker_array_id):
        selected_item = self.sources_tree_widget.currentItem()
        if selected_item:
            speaker_array_id = selected_item.data(0, Qt.UserRole)
            
            if isinstance(speaker_array_id, dict):
                speaker_array_id = speaker_array_id.get('id')
            elif not isinstance(speaker_array_id, (int, str)):
                return

            if self.windowing_plot:
                self.windowing_plot.windowing_plot(windowing, speaker_array_id)

    def update_sources_input_fields(self):
        selected_item = self.sources_tree_widget.currentItem()
        if selected_item:
            speaker_array_id = selected_item.data(0, Qt.UserRole)
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array:
                self.number_of_sources_edit.blockSignals(True)
                self.number_of_sources_edit.setText(str(speaker_array.number_of_sources))
                self.number_of_sources_edit.blockSignals(False)
                
    def update_source_length_input_fields(self):
        selected_item = self.sources_tree_widget.currentItem()
        if selected_item:
            speaker_array_id = selected_item.data(0, Qt.UserRole)
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
    
            if speaker_array:
                self.array_length_edit.blockSignals(True)
                self.array_length_edit.setText(str(speaker_array.source_length))
                self.array_length_edit.blockSignals(False)
    
    def update_array_position_input_fields(self):
        """Aktualisiert die Array-Positions-Eingabefelder"""
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
        
        if speaker_array:
            # Prüfe, ob es sich um ein Flown-System handelt
            is_flown = hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown"
            
            # Initialisiere Array-Positionen, falls nicht vorhanden
            if not hasattr(speaker_array, 'array_position_x'):
                speaker_array.array_position_x = 0.0
            if not hasattr(speaker_array, 'array_position_y'):
                speaker_array.array_position_y = 0.0
            if not hasattr(speaker_array, 'array_position_z'):
                speaker_array.array_position_z = 0.0
            
            # Bei Flown-Systemen: Lese die Werte aus den Source-Positionen (da sie dort direkt gesetzt werden)
            if is_flown:
                # X-Position: aus source_position_x[0]
                if hasattr(speaker_array, 'source_position_x') and len(speaker_array.source_position_x) > 0:
                    array_x = float(speaker_array.source_position_x[0])
                    speaker_array.array_position_x = array_x
                else:
                    array_x = speaker_array.array_position_x
                
                # Y-Position: aus source_position_y[0]
                if hasattr(speaker_array, 'source_position_y') and len(speaker_array.source_position_y) > 0:
                    array_y = float(speaker_array.source_position_y[0])
                    speaker_array.array_position_y = array_y
                else:
                    array_y = speaker_array.array_position_y
                
                # Z-Position: aus source_position_z_flown[0]
                if hasattr(speaker_array, 'source_position_z_flown') and speaker_array.source_position_z_flown is not None and len(speaker_array.source_position_z_flown) > 0:
                    array_z = float(speaker_array.source_position_z_flown[0])
                    speaker_array.array_position_z = array_z
                else:
                    array_z = speaker_array.array_position_z
            else:
                # Bei Stack-Systemen: Verwende die Array-Positionen direkt
                array_x = speaker_array.array_position_x
                array_y = speaker_array.array_position_y
                array_z = speaker_array.array_position_z
            
            self.array_x_edit.blockSignals(True)
            self.array_x_edit.setText(f"{array_x:.2f}")
            self.array_x_edit.blockSignals(False)
            
            self.array_y_edit.blockSignals(True)
            self.array_y_edit.setText(f"{array_y:.2f}")
            self.array_y_edit.blockSignals(False)
            
            self.array_z_edit.blockSignals(True)
            self.array_z_edit.setText(f"{array_z:.2f}")
            self.array_z_edit.blockSignals(False)
                        
    def update_windowing_input_fields(self):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
        
        if speaker_array:
            self.WindowFunction.blockSignals(True)
            self.WindowFunction.setCurrentText(speaker_array.window_function)
            self.WindowFunction.blockSignals(False)
            
            self.WindowRestriction.blockSignals(True)
            self.WindowRestriction.setText(str(speaker_array.window_restriction))
            self.WindowRestriction.blockSignals(False)
            
            
            self.Alpha.blockSignals(True)
            self.Alpha.setText(str(speaker_array.alpha))
            self.Alpha.blockSignals(False)
            
    def update_gain_delay_input_fields(self):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
        
        if speaker_array:
            self.delay_edit.blockSignals(True)
            self.delay_edit.setText(str(speaker_array.delay))
            self.delay_edit.blockSignals(False)
            
            self.gain_edit.blockSignals(True)
            self.gain_edit.setText(str(speaker_array.gain))
            self.gain_edit.blockSignals(False)
            
            # Polarity checkbox aktualisieren
            if hasattr(self, 'polarity_checkbox'):
                self.polarity_checkbox.blockSignals(True)
                # Prüfe ob alle Quellen invertiert sind
                if hasattr(speaker_array, 'source_polarity') and len(speaker_array.source_polarity) > 0:
                    polarity_inverted = bool(speaker_array.source_polarity[0])
                else:
                    polarity_inverted = False
                self.polarity_checkbox.setChecked(polarity_inverted)
                self.polarity_checkbox.blockSignals(False)
    
    def update_beamsteering_windowing_plots(self):
        """Aktualisiert Beamsteering- und Windowing-Plots für das ausgewählte Array (nur Plot, keine Neuberechnung)."""
        speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speakerspecs_instance = self.get_speakerspecs_instance(speaker_array_id)
        if speakerspecs_instance:
            # Nur plotten, Berechnungen passieren in Main.update_speaker_array_calculations()
            self.main_window.plot_beamsteering(speaker_array_id)
            self.main_window.plot_windowing(speaker_array_id)
    
    def add_stack(self):
        """Fügt ein neues Stack-Array hinzu"""
        # Prüfe, ob das TabWidget existiert
        if hasattr(self, 'tab_widget') and self.tab_widget is not None:
            # Entferne alle Tabs
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
        else:
            # Erstelle das TabWidget, falls es noch nicht existiert
            self.tab_widget = QTabWidget()
            self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Füge TabWidget zum rechten Widget hinzu
            if hasattr(self, 'right_side_widget'):
                right_layout = QVBoxLayout()
                right_layout.addWidget(self.tab_widget)
                
                # Entferne alle vorhandenen Layouts
                if self.right_side_widget.layout():
                    QWidget().setLayout(self.right_side_widget.layout())
                    
                self.right_side_widget.setLayout(right_layout)
        
        # Erstelle die Tabs neu
        self.create_speaker_tab_stack()
        self.create_beamsteering_tab()
        self.create_windowing_tab()

        # Erstelle neue Array-ID
        array_id = 1
        while array_id in self.settings.get_all_speaker_array_ids():
            array_id += 1
        
        # Array erstellen und initialisieren - container übergeben!
        array_name = f"Stack Array {array_id}"
        self.settings.add_speaker_array(array_id, array_name, self.container)
        new_array = self.settings.get_speaker_array(array_id)
        
        # Setze Konfiguration auf Stack
        new_array.configuration = "Stack"
        
        # TreeWidget Item erstellen
        new_array_item = QTreeWidgetItem(self.sources_tree_widget, [array_name])
        new_array_item.setFlags(new_array_item.flags() | Qt.ItemIsEditable)
        new_array_item.setData(0, Qt.UserRole, array_id)
        new_array_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
    
        # Checkboxen erstellen und verbinden
        mute_checkbox = self.create_checkbox(False)
        hide_checkbox = self.create_checkbox(False)
        self.sources_tree_widget.setItemWidget(new_array_item, 1, mute_checkbox)
        self.sources_tree_widget.setItemWidget(new_array_item, 2, hide_checkbox)
        self._update_color_indicator(new_array_item, new_array)
        
        new_array.mute = False
        new_array.hide = False
    
        mute_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_mute_state(id, state))
        hide_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_hide_state(id, state))
    
        # Speakerspecs erstellen und initialisieren
        speakerspecs_instance = self.create_speakerspecs_instance(array_id)
        self.add_speakerspecs_instance(array_id, speakerspecs_instance)
        self.init_ui(speakerspecs_instance, self.speaker_tab_layout)
    
        # Item auswählen
        self.sources_tree_widget.setCurrentItem(new_array_item)
        
        # Aktiviere alle Tabs für Stack-Arrays
        if self.tab_widget.count() > 1:
            self.tab_widget.setCurrentIndex(0)  # Wechsle zum Speaker Setup Tab
            
            # Aktiviere alle Tabs
            for i in range(self.tab_widget.count()):
                self.tab_widget.setTabEnabled(i, True)
        
        # 🎯 Plot aktualisieren nach Array-Erstellung
        if hasattr(self.main_window, "update_speaker_array_calculations"):
            self.main_window.update_speaker_array_calculations()
        
        # Validiere alle Checkboxen nach dem Hinzufügen
        self.validate_all_checkboxes()
        
        # Passe Spaltenbreite an den Inhalt an
        self.adjust_column_width_to_content()

    def add_flown(self):
        """Fügt ein neues Flown-Array hinzu"""
        # Prüfe, ob das TabWidget existiert
        if hasattr(self, 'tab_widget') and self.tab_widget is not None:
            # Entferne alle Tabs
            while self.tab_widget.count() > 0:
                self.tab_widget.removeTab(0)
        else:
            # Erstelle das TabWidget, falls es noch nicht existiert
            self.tab_widget = QTabWidget()
            self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Füge TabWidget zum rechten Widget hinzu
            if hasattr(self, 'right_side_widget'):
                right_layout = QVBoxLayout()
                right_layout.addWidget(self.tab_widget)
                
                # Entferne alle vorhandenen Layouts
                if self.right_side_widget.layout():
                    QWidget().setLayout(self.right_side_widget.layout())
                    
                self.right_side_widget.setLayout(right_layout)
        
        # Erstelle die Tabs neu
        self.create_speaker_tab_flown()
        self.create_beamsteering_tab()
        self.create_windowing_tab()

        # Erstelle neue Array-ID
        array_id = 1
        while array_id in self.settings.get_all_speaker_array_ids():
            array_id += 1
        
        # Array erstellen und initialisieren - container übergeben!
        array_name = f"Flown Array {array_id}"
        self.settings.add_speaker_array(array_id, array_name, self.container)
        new_array = self.settings.get_speaker_array(array_id)
        
        # Setze Konfiguration auf Flown
        new_array.configuration = "Flown"
        
        # TreeWidget Item erstellen
        new_array_item = QTreeWidgetItem(self.sources_tree_widget, [array_name])
        new_array_item.setFlags(new_array_item.flags() | Qt.ItemIsEditable)
        new_array_item.setData(0, Qt.UserRole, array_id)
        new_array_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
    
        # Checkboxen erstellen und verbinden
        mute_checkbox = self.create_checkbox(False)
        hide_checkbox = self.create_checkbox(False)
        self.sources_tree_widget.setItemWidget(new_array_item, 1, mute_checkbox)
        self.sources_tree_widget.setItemWidget(new_array_item, 2, hide_checkbox)
        self._update_color_indicator(new_array_item, new_array)
        
        new_array.mute = False
        new_array.hide = False
    
        mute_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_mute_state(id, state))
        hide_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_hide_state(id, state))
    
        # Speakerspecs erstellen und initialisieren
        speakerspecs_instance = self.create_speakerspecs_instance(array_id)
        self.add_speakerspecs_instance(array_id, speakerspecs_instance)
        self.init_ui(speakerspecs_instance, self.speaker_tab_layout)
    
        # Item auswählen
        self.sources_tree_widget.setCurrentItem(new_array_item)
        
        # Deaktiviere Beamsteering und Windowing Tabs für Flown-Arrays
        if self.tab_widget.count() > 1:
            self.tab_widget.setCurrentIndex(0)  # Wechsle zum Speaker Setup Tab
            
            # Aktiviere den ersten Tab (Speaker Setup)
            self.tab_widget.setTabEnabled(0, True)
            
            # Deaktiviere die anderen Tabs (Beamsteering und Windowing)
            for i in range(1, self.tab_widget.count()):
                self.tab_widget.setTabEnabled(i, False)
                
        self.main_window.update_speaker_array_calculations()
        
        # Validiere alle Checkboxen nach dem Hinzufügen
        self.validate_all_checkboxes()
        
        # Passe Spaltenbreite an den Inhalt an
        self.adjust_column_width_to_content()




# ------ Tree widget Signale ----- 


    def update_mute_state(self, array_id, state):
        """Aktualisiert den Mute-Status eines Arrays. Bei Mehrfachauswahl werden alle ausgewählten Arrays aktualisiert."""
        mute_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.sources_tree_widget.selectedItems()
        if len(selected_items) > 1:
            # Mehrfachauswahl: Wende auf alle ausgewählten Arrays an
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type != "group":  # Nur Arrays, keine Gruppen
                        item_array_id = item.data(0, Qt.UserRole)
                        if item_array_id is not None:
                            speaker_array = self.settings.get_speaker_array(item_array_id)
                            if speaker_array:
                                speaker_array.mute = mute_value
                                
                                # Aktualisiere auch die Checkbox im TreeWidget
                                mute_checkbox = self.sources_tree_widget.itemWidget(item, 1)
                                if mute_checkbox:
                                    mute_checkbox.blockSignals(True)
                                    # Verwende setCheckState für tristate-Checkboxen
                                    mute_checkbox.setCheckState(Qt.Checked if mute_value else Qt.Unchecked)
                                    mute_checkbox.blockSignals(False)
                                
                                # Aktualisiere Gruppen-Checkbox-Zustand (falls Array in Gruppe)
                                parent = item.parent()
                                while parent:
                                    self._update_group_checkbox_state(parent, 1)
                                    parent = parent.parent()
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Normales Verhalten
            speaker_array = self.settings.get_speaker_array(array_id)
            if speaker_array:
                speaker_array.mute = mute_value
            
            # Aktualisiere Gruppen-Checkbox-Zustand (falls Array in Gruppe)
            item = None
            for i in range(self.sources_tree_widget.topLevelItemCount()):
                top_item = self.sources_tree_widget.topLevelItem(i)
                found = self._find_item_by_array_id(top_item, array_id)
                if found:
                    item = found
                    break
            
            if item:
                parent = item.parent()
                while parent:
                    self._update_group_checkbox_state(parent, 1)
                    parent = parent.parent()
        
        # Berechnen des beamsteerings, windowing, und Impulse
        self.main_window.update_speaker_array_calculations()  
    
    def update_hide_state(self, array_id, state):
        """Aktualisiert den Hide-Status eines Arrays. Bei Mehrfachauswahl werden alle ausgewählten Arrays aktualisiert."""
        hide_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.sources_tree_widget.selectedItems()
        if len(selected_items) > 1:
            # Mehrfachauswahl: Wende auf alle ausgewählten Arrays an
            for item in selected_items:
                try:
                    item_type = item.data(0, Qt.UserRole + 1)
                    if item_type != "group":  # Nur Arrays, keine Gruppen
                        item_array_id = item.data(0, Qt.UserRole)
                        if item_array_id is not None:
                            speaker_array = self.settings.get_speaker_array(item_array_id)
                            if speaker_array:
                                speaker_array.hide = hide_value
                                
                                # Aktualisiere auch die Checkbox im TreeWidget
                                hide_checkbox = self.sources_tree_widget.itemWidget(item, 2)
                                if hide_checkbox:
                                    hide_checkbox.blockSignals(True)
                                    # Verwende setCheckState für tristate-Checkboxen
                                    hide_checkbox.setCheckState(Qt.Checked if hide_value else Qt.Unchecked)
                                    hide_checkbox.blockSignals(False)
                                
                                # Aktualisiere Gruppen-Checkbox-Zustand (falls Array in Gruppe)
                                parent = item.parent()
                                while parent:
                                    self._update_group_checkbox_state(parent, 2)
                                    parent = parent.parent()
                except RuntimeError:
                    # Item wurde gelöscht, überspringe
                    continue
        else:
            # Einzelauswahl: Normales Verhalten
            speaker_array = self.settings.get_speaker_array(array_id)
            if speaker_array:
                speaker_array.hide = hide_value
            
            # Aktualisiere Gruppen-Checkbox-Zustand (falls Array in Gruppe)
            item = None
            for i in range(self.sources_tree_widget.topLevelItemCount()):
                top_item = self.sources_tree_widget.topLevelItem(i)
                found = self._find_item_by_array_id(top_item, array_id)
                if found:
                    item = found
                    break
            
            if item:
                parent = item.parent()
                while parent:
                    self._update_group_checkbox_state(parent, 2)
                    parent = parent.parent()
        
        # Berechnen des beamsteerings, windowing, und Impulse
        self.main_window.update_speaker_array_calculations()  
                    
    def update_array_id(self, item, column):
        array_id = item.data(0, Qt.UserRole)
        new_name = item.text(0)

        for id, array in self.settings.get_all_speaker_arrays().items():
            if array.name == new_name and id != array_id:
                speaker_array = self.settings.get_speaker_array(array_id)
                if speaker_array:
                    old_name = speaker_array.name
                    item.setText(0, old_name)

        speaker_array = self.settings.get_speaker_array(array_id)
        if speaker_array:
            speaker_array.name = new_name
    

    def delete_array(self):
        selected_item = self.sources_tree_widget.selectedItems()
        if selected_item:
            item = selected_item[0]
            
            # Prüfe, ob es sich um eine Gruppe handelt
            if item.data(0, Qt.UserRole + 1) == "group":
                self.delete_group(item)
                return
            
            array_id = item.data(0, Qt.UserRole)
            
            # Entferne das Item aus dem TreeWidget (berücksichtigt Parent-Gruppen)
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            else:
                index = self.sources_tree_widget.indexOfTopLevelItem(item)
                if index != -1:
                    self.sources_tree_widget.takeTopLevelItem(index)
            
            # Entferne das entsprechende speakerspecs-Objekt und Daten
            found_instance = None
            for instance in self.speakerspecs_instance:
                if instance['id'] == array_id:  # Hier wurde die nderung vorgenommen
                    found_instance = instance
                    break
            
            if found_instance:
                self.speakerspecs_instance.remove(found_instance)
                found_instance['scroll_area'].setParent(None)  # Möglicherweise muss auch dies angepasst werden
            
            # Entferne das Array aus den Einstellungen
            self.settings.remove_speaker_array(array_id)
            
            # Prüfe ob noch Arrays vorhanden sind und wähle ggf. ein anderes aus
            if self.sources_tree_widget.topLevelItemCount() > 0:
                # Wähle das erste verbleibende Array aus
                first_item = self.sources_tree_widget.topLevelItem(0)
                self.sources_tree_widget.setCurrentItem(first_item)
                
                # Aktualisiere alle Berechnungen und Plots
                self.main_window.update_speaker_array_calculations()
            else:
                # Keine Arrays mehr vorhanden - leere aktuelle Berechnungen (aber behalte Snapshots!)
                self.container.calculation_impulse.clear()
                self.container.calculation_spl.clear()
                self.container.calculation_polar.clear()
                
                # Lösche nur "aktuelle_simulation" aus calculation_axes, behalte Snapshots
                if "aktuelle_simulation" in self.container.calculation_axes:
                    del self.container.calculation_axes["aktuelle_simulation"]
                
                # Aktualisiere alle Plots (werden Snapshots anzeigen, wenn vorhanden)
                if hasattr(self.main_window, 'impulse_manager'):
                    self.main_window.impulse_manager.update_plot_impulse()
                
                if hasattr(self.main_window, 'draw_plots'):
                    # Aktualisiere alle SPL-Plots (zeigen Snapshots oder Empty, je nachdem)
                    self.main_window.plot_xaxis()
                    self.main_window.plot_yaxis()
                    
                    # Haupt-SPL-Plot: Prüfe ob Snapshots vorhanden sind
                    has_snapshots = any(key != "aktuelle_simulation" and val.get("show_in_plot", False) 
                                       for key, val in self.container.calculation_axes.items())
                    
                    if has_snapshots:
                        # Snapshots vorhanden - versuche SPL zu plotten (falls Snapshot SPL-Daten hat)
                        # Für jetzt: Empty Plot, da Snapshots keine SPL-2D-Daten enthalten
                        pass
                    
                    # Zeige Empty Plot für Haupt-SPL (3D-Version)
                    draw_spl_plotter = self.main_window.draw_plots._get_current_spl_plotter()
                    if draw_spl_plotter is not None:
                        try:
                            draw_spl_plotter.initialize_empty_scene(preserve_camera=True)
                        except AttributeError:
                            draw_spl_plotter.initialize_empty_plots()
                    self.main_window.draw_plots.colorbar_canvas.draw()
                    
                    # Polar-Plot
                    self.main_window.draw_plots.draw_polar_pattern.initialize_empty_plots()


    def duplicate_array(self, item, update_calculations=True):
        """Dupliziert das ausgewählte Array
        
        Args:
            item: Das zu duplizierende Array-Item
            update_calculations: Wenn True, werden Berechnungen am Ende ausgeführt (Standard: True)
        """
        if item:
            # Blockiere Signale während der Duplikation, um Berechnungen zu vermeiden
            self.sources_tree_widget.blockSignals(True)
            
            try:
                original_array_id = item.data(0, Qt.UserRole)
                original_array = self.settings.get_speaker_array(original_array_id)
                
                if original_array:
                    # Erstelle neue Array-ID
                    new_array_id = 1
                    while new_array_id in self.settings.get_all_speaker_array_ids():
                        new_array_id += 1
                    
                    # Kopiere Array-Einstellungen
                    self.settings.duplicate_speaker_array(original_array_id, new_array_id)
                    new_array = self.settings.get_speaker_array(new_array_id)
                    new_array.name = f"copy of {original_array.name}"
                    
                    # Erstelle neues TreeWidget Item
                    new_array_item = QTreeWidgetItem([new_array.name])
                    new_array_item.setFlags(new_array_item.flags() | Qt.ItemIsEditable)
                    new_array_item.setData(0, Qt.UserRole, new_array_id)
                    new_array_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
                    
                    # Füge Item direkt unter dem Original-Item ein
                    parent = item.parent()
                    if parent:
                        # Item ist in einer Gruppe - füge direkt nach dem Original ein
                        parent_index = parent.indexOfChild(item)
                        parent.insertChild(parent_index + 1, new_array_item)
                    else:
                        # Top-Level-Item - füge direkt nach dem Original ein
                        top_level_index = self.sources_tree_widget.indexOfTopLevelItem(item)
                        self.sources_tree_widget.insertTopLevelItem(top_level_index + 1, new_array_item)
                    
                    # Erstelle Checkboxen
                    mute_checkbox = self.create_checkbox(new_array.mute)
                    hide_checkbox = self.create_checkbox(new_array.hide)
                    self.sources_tree_widget.setItemWidget(new_array_item, 1, mute_checkbox)
                    self.sources_tree_widget.setItemWidget(new_array_item, 2, hide_checkbox)
                    self._update_color_indicator(new_array_item, new_array)
                    
                    mute_checkbox.stateChanged.connect(lambda state, id=new_array_id: self.update_mute_state(id, state))
                    hide_checkbox.stateChanged.connect(lambda state, id=new_array_id: self.update_hide_state(id, state))
                    
                    # Erstelle neue Speakerspecs-Instanz
                    speakerspecs_instance = self.create_speakerspecs_instance(new_array_id)
                    self.init_ui(speakerspecs_instance, self.speaker_tab_layout)
                    self.add_speakerspecs_instance(new_array_id, speakerspecs_instance)
                    
                    # Wähle das neue Item aus (Signale sind blockiert, daher keine Berechnungen)
                    self.sources_tree_widget.setCurrentItem(new_array_item)
                    
                    # Validiere alle Checkboxen nach Duplizierung
                    self.validate_all_checkboxes()
                    
                    # Passe Spaltenbreite an den Inhalt an
                    self.adjust_column_width_to_content()
                    
                    # Aktualisiere die Anzeige erst nach allen UI-Updates (nur wenn gewünscht)
                    if update_calculations:
                        self.main_window.update_speaker_array_calculations()
            finally:
                # Signale wieder aktivieren
                self.sources_tree_widget.blockSignals(False)


            
            



    # ---- Signalhandler ----

    def on_speaker_type_changed(self, combo, source_index, speaker_array_id):
        """
        Wird aufgerufen, wenn der Benutzer einen anderen Lautsprechertyp auswählt.
        
        Args:
            combo: Die Combobox, die geändert wurde
            source_index: Index der Quelle
            speaker_array_id: ID des Speaker Arrays
        """
        speaker_array = self.settings.get_speaker_array(speaker_array_id)
        if not speaker_array:
            return
        
        # Aktualisiere das Polar Pattern
        new_pattern = combo.currentText()
        speaker_array.source_polar_pattern[source_index] = new_pattern
        
        # Aktualisiere die Winkel-Combobox, wenn es sich um ein Flown-Array handelt
        if hasattr(speaker_array, 'configuration') and speaker_array.configuration.lower() != "stack":
            self.update_angle_combobox(speaker_array_id, source_index)
        
        # Aktualisiere Lautsprecher-Overlays (immer)
        self.update_speaker_overlays()
        
        # Neu berechnen nur wenn autocalc aktiv
        if self.is_autocalc_active():
            self.main_window.update_speaker_array_calculations()

    def on_speakerspecs_item_text_changed(self, item, column):
        try:
            # Prüfe, ob es sich um eine Gruppe handelt
            is_group = item.data(0, Qt.UserRole + 1) == "group"
            
            if is_group:
                # Gruppen-Name wurde geändert
                new_name = item.text(0)
                
                # Stelle sicher, dass die Gruppen-Struktur aktuell ist
                self.save_groups_structure()
                
                # Finde die Gruppen-ID durch Vergleich der Child-Arrays
                group_id = None
                old_name = None
                
                # Sammle die IDs der Child-Arrays dieser Gruppe
                child_array_ids = []
                for i in range(item.childCount()):
                    child_item = item.child(i)
                    child_array_id = child_item.data(0, Qt.UserRole)
                    if isinstance(child_array_id, dict):
                        child_array_id = child_array_id.get('id')
                    if child_array_id is not None:
                        child_array_ids.append(child_array_id)
                
                # Finde die Gruppe, die diese Child-Arrays enthält
                for gid, gdata in self.settings.speaker_array_groups.items():
                    saved_child_ids = gdata.get('child_array_ids', [])
                    if set(saved_child_ids) == set(child_array_ids):
                        group_id = gid
                        old_name = gdata.get('name')
                        break
                
                # Falls nicht gefunden durch Child-Arrays, versuche es durch Vergleich mit allen Gruppen
                # und verwende den Namen, der nicht dem neuen entspricht
                if group_id is None or old_name is None:
                    # Durchsuche alle Top-Level-Items und finde die Position dieser Gruppe
                    group_index = -1
                    for i in range(self.sources_tree_widget.topLevelItemCount()):
                        top_item = self.sources_tree_widget.topLevelItem(i)
                        if top_item == item:
                            group_index = i
                            break
                    
                    # Wenn wir die Position gefunden haben, durchsuche alle Gruppen
                    # und finde die, die an dieser Position sein sollte
                    if group_index >= 0:
                        # Zähle Gruppen bis zu dieser Position
                        group_count = 0
                        for i in range(group_index + 1):
                            top_item = self.sources_tree_widget.topLevelItem(i)
                            if top_item and top_item.data(0, Qt.UserRole + 1) == "group":
                                if i == group_index:
                                    # Dies ist die geänderte Gruppe
                                    # Finde die entsprechende Gruppe in den gespeicherten Daten
                                    group_keys = list(self.settings.speaker_array_groups.keys())
                                    if group_count < len(group_keys):
                                        group_id = group_keys[group_count]
                                        old_name = self.settings.speaker_array_groups[group_id].get('name')
                                        break
                                group_count += 1
                
                if group_id is None or old_name is None:
                    # Falls wir die Gruppe immer noch nicht finden können, 
                    # verwende einen Fallback: Suche nach einer Gruppe, die noch nicht den neuen Namen hat
                    for gid, gdata in self.settings.speaker_array_groups.items():
                        saved_name = gdata.get('name')
                        if saved_name != new_name:
                            # Verwende diese Gruppe als Fallback
                            group_id = gid
                            old_name = saved_name
                            break
                
                if group_id is None or old_name is None:
                    # Falls wir die Gruppe immer noch nicht finden können, 
                    # erstelle eine neue Gruppe oder verwende einen Standard-Namen
                    print(f"Warnung: Gruppe konnte nicht gefunden werden. Name '{new_name}' wird nicht gespeichert.")
                    # Setze auf einen Standard-Namen zurück
                    item.blockSignals(True)
                    item.setText(0, "Group")
                    item.blockSignals(False)
                    return
                
                # Prüfe, ob der neue Name bereits von einem Array verwendet wird
                name_exists = False
                for array_id, array in self.settings.speaker_arrays.items():
                    if array.name == new_name:
                        name_exists = True
                        break
                
                # Prüfe auch in anderen Gruppen
                if not name_exists:
                    for gid, gdata in self.settings.speaker_array_groups.items():
                        if gid != group_id and gdata.get('name') == new_name:
                            name_exists = True
                            break
                
                if name_exists:
                    # Name bereits vorhanden: Setze auf vorherigen Namen zurück
                    item.blockSignals(True)
                    item.setText(0, old_name)
                    item.blockSignals(False)
                    print(f"Name '{new_name}' wird bereits verwendet. Zurückgesetzt auf '{old_name}'.")
                    return
                
                # Name ist eindeutig: Speichere den neuen Namen
                self.settings.speaker_array_groups[group_id]['name'] = new_name
                
                # Passe Spaltenbreite an den Inhalt an
                self.adjust_column_width_to_content()
                
            else:
                # Array-Name wurde geändert
                speaker_array_id = item.data(0, Qt.UserRole)
                if speaker_array_id is not None:
                    speaker_array = self.settings.get_speaker_array(speaker_array_id)
                    if speaker_array:
                        new_name = item.text(0)
                        old_name = speaker_array.name
                        
                        # Prüfe, ob der neue Name bereits von einem anderen Array verwendet wird
                        name_exists = False
                        for array_id, array in self.settings.speaker_arrays.items():
                            if array_id != speaker_array_id and array.name == new_name:
                                name_exists = True
                                break
                        
                        # Prüfe auch in Gruppen
                        if not name_exists:
                            for group_data in self.settings.speaker_array_groups.values():
                                if group_data.get('name') == new_name:
                                    name_exists = True
                                    break
                        
                        if name_exists:
                            # Name bereits vorhanden: Setze auf vorherigen Namen zurück
                            item.blockSignals(True)
                            item.setText(0, old_name)
                            item.blockSignals(False)
                            print(f"Name '{new_name}' wird bereits verwendet. Zurückgesetzt auf '{old_name}'.")
                            return
                        
                        # Name ist eindeutig: Speichere den neuen Namen
                        speaker_array.name = new_name
                        
                        # 🎯 Plot aktualisieren wenn Array-Name geändert wird
                        if hasattr(self.main_window, "update_speaker_array_calculations"):
                            self.main_window.update_speaker_array_calculations()
                        
                        # Passe Spaltenbreite an den Inhalt an
                        self.adjust_column_width_to_content()
        except Exception as e:
            print(f"Fehler beim Ändern des Array-Namens: {e}")

    def on_number_of_sources_changed(self):
        """Handler für Änderungen der Quellenanzahl"""
        try:
            new_value = int(self.number_of_sources_edit.text())
            
            # Hole aktuelles Array
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id:
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                if speaker_array:
                    # Prüfe ob sich der Wert tatsächlich geändert hat
                    if new_value == speaker_array.number_of_sources:
                        return  # Keine Änderung, nichts tun
                    
                    # Ab hier nur ausführen wenn sich der Wert wirklich geändert hat
                    speaker_array.update_sources(new_value)
                    self.settings.update_number_of_sources(selected_speaker_array_id, new_value)

                    # Für Flown-Arrays globale Positions-/Ausrichtungswerte beibehalten
                    if hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown":
                        import numpy as np

                        def _replicate_attr(attr_name):
                            values = getattr(speaker_array, attr_name, None)
                            if values is None:
                                return
                            if len(values) == 0:
                                return
                            base_value = values[0]
                            setattr(
                                speaker_array,
                                attr_name,
                                np.full(speaker_array.number_of_sources, base_value, dtype=float)
                            )

                        _replicate_attr('source_position_x')
                        _replicate_attr('source_position_y')
                        _replicate_attr('source_site')
                        _replicate_attr('source_azimuth')

                        z_top = getattr(speaker_array, 'source_position_z_flown', None)
                        if z_top is not None and len(z_top) > 0:
                            base_top = z_top[0]
                        else:
                            base_top = 0.0
                        speaker_array.source_position_z_flown = np.full(
                            speaker_array.number_of_sources, base_top, dtype=float
                        )
                    
                    self.main_window.update_speaker_array_calculations()     
                
                    speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
                    if speakerspecs_instance:
                        self.update_input_fields(speakerspecs_instance)
        except ValueError:
            pass



    def on_x_position_changed(self, edit, source_index, speaker_array_id):
        """
        Behandelt Änderungen der X-Position einer Quelle.
        Optimiert für symmetrische Arrays.
        """
        try:
            
            # Blockiere Signale während der Verarbeitung
            edit.blockSignals(True)
            
            # Runde den Wert auf 2 Dezimalstellen
            value = round(float(edit.text()) if edit.text() else 0, 2)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                edit.blockSignals(False)
                return
                
            # Prüfe, ob sich der Wert tatsächlich geändert hat
            if speaker_array.source_position_x[source_index] == value:
                # Setze den formatierten Wert zurück und beende
                edit.setText(f"{value:.2f}")
                edit.blockSignals(False)
                return
                
            # Aktualisiere den Wert im Array
            speaker_array.source_position_x[source_index] = value
            
            # Hole die Instance für dieses Array
            instance = self.get_speakerspecs_instance(speaker_array_id)
            
            # Wenn Symmetrie aktiviert ist, wende symmetrische Werte an
            if instance and instance['state']:
                self.apply_symmetric_values(instance)
            
            # Berechne die neue Länge des Arrays nur einmal
            source_length = max(speaker_array.source_position_x) - min(speaker_array.source_position_x)
            source_length = round(source_length, 2)
            speaker_array.source_length = source_length
            
            # Aktualisiere das Eingabefeld für die Array-Länge
            self.array_length_edit.blockSignals(True)
            self.array_length_edit.setText(f"{source_length:.2f}")
            self.array_length_edit.blockSignals(False)
            
            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
            self.update_delay_input_fields(speaker_array_id)
            
            # Setze den formatierten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.2f}")
            
            # Entsperre die Signale wieder
            edit.blockSignals(False)
            
        except ValueError as e:
            # Fehlerbehandlung
            print(f"Fehler in on_x_position_changed: {e}")
            if speaker_array:
                edit.setText(f"{speaker_array.source_position_x[source_index]:.2f}")
            edit.blockSignals(False)



    def on_y_position_changed(self, edit, source_index, speaker_array_id):
        try:
            # Blockiere Signale während der Verarbeitung
            edit.blockSignals(True)
            
            # Runde den Wert auf 2 Dezimalstellen
            value = round(float(edit.text()) if edit.text() else 0, 2)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                edit.blockSignals(False)
                return
                
            # Prüfe, ob sich der Wert tatsächlich geändert hat
            if speaker_array.source_position_y[source_index] == value:
                # Setze den formatierten Wert zurück und beende
                edit.setText(f"{value:.2f}")
                edit.blockSignals(False)
                return
                
            # Aktualisiere den Wert im Array
            speaker_array.source_position_y[source_index] = value
            
            # Hole die Instance für dieses Array
            instance = self.get_speakerspecs_instance(speaker_array_id)
            
            # Wenn Symmetrie aktiviert ist, wende symmetrische Werte an
            if instance and instance['state']:
                self.apply_symmetric_values(instance)

            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
            self.update_delay_input_fields(speaker_array_id)
            
            # Setze den formatierten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.2f}")
            
            # Entsperre die Signale wieder
            edit.blockSignals(False)
            
        except ValueError as e:
            # Fehlerbehandlung
            print(f"Fehler in on_y_position_changed: {e}")
            if speaker_array:
                edit.setText(f"{speaker_array.source_position_y[source_index]:.2f}")
            edit.blockSignals(False)


    def on_z_position_changed(self, edit, source_index, speaker_array_id):
        """Handler für Änderungen der Z-Position"""
        try:
            
            # Blockiere Signale während der Verarbeitung
            edit.blockSignals(True)
            
            # Runde den Wert auf 2 Dezimalstellen
            value = round(float(edit.text()) if edit.text() else 0, 2)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                edit.blockSignals(False)
                return
                
            # Initialisiere source_position_z, falls es noch nicht existiert
            if not hasattr(speaker_array, 'source_position_z_stack') or speaker_array.source_position_z_stack is None:
                speaker_array.source_position_z_stack = [0.0] * speaker_array.number_of_sources
            
            # Stelle sicher, dass source_position_z die richtige Länge hat
            if len(speaker_array.source_position_z_stack) < speaker_array.number_of_sources:
                speaker_array.source_position_z_stack.extend([0.0] * (speaker_array.number_of_sources - len(speaker_array.source_position_z_stack)))
            
            # Prüfe, ob sich der Wert tatsächlich geändert hat
            if speaker_array.source_position_z_stack[source_index] == value:
                # Setze den formatierten Wert zurück und beende
                edit.setText(f"{value:.2f}")
                edit.blockSignals(False)
                return
                
            # Aktualisiere den Wert
            speaker_array.source_position_z_stack[source_index] = value
            
            # Hole die Instance für dieses Array
            instance = self.get_speakerspecs_instance(speaker_array_id)
            
            # Wenn Symmetrie aktiviert ist, wende symmetrische Werte an
            if instance and instance['state']:
                self.apply_symmetric_values(instance)
            
            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
            self.update_delay_input_fields(speaker_array_id)

            # Setze den formatierten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.2f}")
            
            # Entsperre die Signale wieder
            edit.blockSignals(False)
            print("speaker_array.source_position_z_stack", speaker_array.source_position_z_stack)
            
        except ValueError as e:
            # Fehlerbehandlung
            if speaker_array and hasattr(speaker_array, 'source_position_z_stack'):
                edit.setText(f"{speaker_array.source_position_z_stack[source_index]:.2f}")
            edit.blockSignals(False)

    def on_source_azimuth_changed(self, edit, source_index, speaker_array_id):
        try:
            # Runde den Wert auf 1 Dezimalstelle
            value = round(float(edit.text()) if edit.text() else 0, 1)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if speaker_array and speaker_array.source_azimuth[source_index] != value:
                speaker_array.source_azimuth[source_index] = value

                self.update_input_fields(self.get_speakerspecs_instance(speaker_array_id))
                
                # Aktualisiere Lautsprecher-Overlays (immer)
                self.update_speaker_overlays()
                
                # Neu berechnen nur wenn autocalc aktiv
                if self.is_autocalc_active():
                    self.main_window.update_speaker_array_calculations()

            # Setze den gerundeten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.1f}")
        except ValueError:
            # Setze den vorherigen gültigen Wert zurück
            edit.setText(f"{speaker_array.source_azimuth[source_index]:.1f}")

    def on_source_delay_changed(self, edit, source_index, speaker_array_id):
        try:
            
            # Blockiere Signale während der Verarbeitung
            edit.blockSignals(True)
            
            # Runde den Wert auf 2 Dezimalstellen
            value = round(float(edit.text()) if edit.text() else 0, 2)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                edit.blockSignals(False)
                return
                
            # Prüfe, ob sich der Wert tatsächlich geändert hat
            if speaker_array.source_time[source_index] == value:
                # Setze den formatierten Wert zurück und beende
                edit.setText(f"{value:.2f}")
                edit.blockSignals(False)
                return
                
            # Aktualisiere den Wert im Array
            speaker_array.source_time[source_index] = value
            
            # Hole die Instance für dieses Array
            instance = self.get_speakerspecs_instance(speaker_array_id)
            
            # Wenn Symmetrie aktiviert ist, wende symmetrische Werte an
            if instance and instance['state']:
                self.apply_symmetric_values(instance)
            
            # Neu berechnen nur wenn autocalc aktiv (keine Overlay-Aktualisierung)
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
            # Setze den formatierten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.2f}")
            
            # Entsperre die Signale wieder
            edit.blockSignals(False)
            
        except ValueError as e:
            # Fehlerbehandlung
            print(f"Fehler in on_source_delay_changed: {e}")
            if speaker_array:
                edit.setText(f"{speaker_array.source_time[source_index]:.2f}")
            edit.blockSignals(False)

                
                    
    def on_source_level_changed(self, edit, source_index, speaker_array_id):
        try:
            
            # Blockiere Signale während der Verarbeitung
            edit.blockSignals(True)
            
            # Runde den Wert auf 2 Dezimalstellen
            value = round(float(edit.text()) if edit.text() else 0, 2)
            
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                edit.blockSignals(False)
                return
                
            # Prüfe, ob sich der Wert tatsächlich geändert hat
            if speaker_array.source_level[source_index] == value:
                # Setze den formatierten Wert zurück und beende
                edit.setText(f"{value:.2f}")
                edit.blockSignals(False)
                return
                
            # Aktualisiere den Wert im Array
            speaker_array.source_level[source_index] = value
            
            # Hole die Instance für dieses Array
            instance = self.get_speakerspecs_instance(speaker_array_id)
            
            # Wenn Symmetrie aktiviert ist, wende symmetrische Werte an
            if instance and instance['state']:
                self.apply_symmetric_values(instance)
            
            # Neu berechnen nur wenn autocalc aktiv (keine Overlay-Aktualisierung)
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
            # Setze den formatierten Wert zurück in das Eingabefeld
            edit.setText(f"{value:.2f}")
            
            # Entsperre die Signale wieder
            edit.blockSignals(False)
            
        except ValueError as e:
            # Fehlerbehandlung
            print(f"Fehler in on_source_level_changed: {e}")
            if speaker_array:
                edit.setText(f"{speaker_array.source_level[source_index]:.2f}")
            edit.blockSignals(False)

            
    def on_Delay_changed(self):
        try:
            
            # Blockiere Signale während der Verarbeitung
            self.delay_edit.blockSignals(True)
            
            try:
                value = float(self.delay_edit.text())
                
                selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
                if selected_speaker_array_id is None:
                    self.delay_edit.blockSignals(False)
                    return
                    
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                if not speaker_array:
                    self.delay_edit.blockSignals(False)
                    return
                    
                # Prüfe ob sich der Wert geändert hat
                if speaker_array.delay == value:
                    self.delay_edit.blockSignals(False)
                    return  # Keine Änderung, nichts tun
                
                speaker_array.delay = value
                self.settings.update_speaker_array_delay(selected_speaker_array_id, value)
                
                # Neu berechnen nur wenn autocalc aktiv (keine Overlay-Aktualisierung)
                if self.is_autocalc_active():
                    self.main_window.update_speaker_array_calculations()
                
            except ValueError:
                pass
            
            # Entsperre die Signale wieder
            self.delay_edit.blockSignals(False)
            
        except Exception as e:
            print(f"Fehler in on_Delay_changed: {e}")
            self.delay_edit.blockSignals(False)


    def on_autosplay_changed(self):
        try:
            
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if not selected_speaker_array_id:
                return
            
            speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
            if not speakerspecs_instance:
                return
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if not speaker_array:
                return
            
            # Blockiere Signale während der Verarbeitung
            if hasattr(self, 'autosplay_button'):
                self.autosplay_button.blockSignals(True)
            
            # Prüfe, ob es sich um ein Stack-System handelt
            configuration = getattr(speaker_array, 'configuration', None)
            is_stack = configuration is None or str(configuration).lower() == "stack"
            
            # Hole Array X-Position (Mittelpunkt für Autosplay)
            array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
            
            # Bei Stack-Systemen: Entferne Array X-Position von source_position_x, damit um 0 zentriert wird
            if is_stack:
                # Entferne Array X-Position von source_position_x (falls vorhanden)
                if hasattr(speaker_array, 'source_position_x') and speaker_array.source_position_x is not None:
                    speaker_array.source_position_x = np.asarray(speaker_array.source_position_x) - array_pos_x
            
            # Lineare Verteilung der X-Positionen (um 0 zentriert)
            speaker_array.source_position_x = self.functions.linear_distribution(
                speaker_array.source_position_x,
                speaker_array.source_length,
                speaker_array.source_polar_pattern,
            )
            
            # Bei Stack-Systemen: Array X-Position wird später in calculate_stack_center hinzuaddiert
            # Bei Flown-Systemen: Array X-Position wird direkt in source_position_x gesetzt
            if not is_stack and hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown":
                # Bei Flown: Setze Array X-Position direkt
                speaker_array.source_position_x = speaker_array.source_position_x + array_pos_x

            # Führe die Beamsteering-Berechnung durch
            beamsteering = BeamSteering(speaker_array, self.container.data, self.settings)
            beamsteering.calculate(selected_speaker_array_id)
            
            # Aktualisiere die Berechnungen nur einmal am Ende
            self.main_window.update_speaker_array_calculations()
            
            self.update_input_fields(speakerspecs_instance)

            # Entsperre die Signale wieder
            if hasattr(self, 'autosplay_button'):
                self.autosplay_button.blockSignals(False)
            
        except Exception as e:
            print(f"Fehler in on_autosplay_changed: {e}")
            if hasattr(self, 'autosplay_button'):
                self.autosplay_button.blockSignals(False)





    def on_WindowRestriction_changed(self):
        try:
            value = float(self.WindowRestriction.text())
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is not None:
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                if speaker_array:
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.window_restriction == value:
                        return  # Keine Änderung, nichts tun
                    
                    speaker_array.window_restriction = value
                    self.settings.update_speaker_array_window_settings(
                        selected_speaker_array_id, 
                        speaker_array.window_function, 
                        speaker_array.alpha, 
                        value
                    )
                    self.main_window.update_speaker_array_calculations()
        except ValueError:
            pass


    def on_WindowFunction_changed(self, index):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
            
            if speaker_array:
                if index == 0:
                    text = "tukey"
                    self.Alpha.setEnabled(True)
                elif index == 1:
                    text = "gauss"
                    self.Alpha.setEnabled(False)
                elif index == 2:
                    text = "flattop"
                    self.Alpha.setEnabled(False)
                elif index == 3:
                    text = "blackman"
                    self.Alpha.setEnabled(False)
                
                speaker_array.window_function = text
                self.settings.update_speaker_array_window_settings(
                    selected_speaker_array_id, 
                    text, 
                    speaker_array.alpha, 
                    speaker_array.window_restriction
                )
    
            # Berechnen des beamsteerings, windowing, und Impulse
            self.main_window.update_speaker_array_calculations()
            
            if speakerspecs_instance:
                self.update_input_fields(speakerspecs_instance)  # Aktualisierung der Eingabefelder
                self.update_widgets(speakerspecs_instance)
    
        # self.update_windowing_plot(speakerspecs_instance)


    def on_Alpha_changed(self):
        try:
            value = float(self.Alpha.text())
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is not None:
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
        
                if speaker_array:
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.alpha == value:
                        return  # Keine Änderung, nichts tun
                    
                    speaker_array.alpha = value
                    self.settings.update_speaker_array_window_settings(
                        selected_speaker_array_id, 
                        speaker_array.window_function, 
                        value, 
                        speaker_array.window_restriction
                    )
        
                # Berechnen des beamsteerings, windowing, und Impulse
                self.main_window.update_speaker_array_calculations()
                
                if speakerspecs_instance:
                    self.update_input_fields(speakerspecs_instance)  # Aktualisierung der Eingabefelder
                    self.update_widgets(speakerspecs_instance)
        except ValueError:
            # Behandeln Sie ungültige Eingaben hier
            pass

        # self.update_windowing_plot(speakerspecs_instance)


    def on_Sources_set_changed(self):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        if selected_speaker_array_id is not None:
            
            calculator_instance = WindowingCalculator(self.settings, self.container.data, self.container.calculation_windowing, selected_speaker_array_id)
            calculator_instance.use_window_on_src()
    
            # Berechnen des beamsteerings, windowing, und Impulse
            self.main_window.update_speaker_array_calculations()
    
            speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
            if speakerspecs_instance:
                self.update_input_fields(speakerspecs_instance)  # Aktualisierung der Eingabefelder
                self.update_widgets(speakerspecs_instance)


    def on_Sources_to_zero_changed(self):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        if selected_speaker_array_id is not None:
            calculator_instance = WindowingCalculator(self.settings, self.container.data, self.container.calculation_windowing, selected_speaker_array_id)
            calculator_instance.set_sources_to_zero()
    
            # Berechnen des beamsteerings, windowing, und Impulse
            self.main_window.update_speaker_array_calculations()
    
            speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
            if speakerspecs_instance:
                self.update_input_fields(speakerspecs_instance)  # Aktualisierung der Eingabefelder
                self.update_widgets(speakerspecs_instance)


    def on_SourceLength_changed(self):
        try:
            value = float(self.array_length_edit.text())
            selected_item = self.sources_tree_widget.currentItem()
            if selected_item:
                speaker_array_id = selected_item.data(0, Qt.UserRole)
                speaker_array = self.settings.get_speaker_array(speaker_array_id)
                if speaker_array:
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.source_length == value:
                        return  # Keine Änderung, nichts tun
                    
                    self.settings.update_source_length(speaker_array_id, value)
                    # Keine Neuberechnung, da Lautsprecherpositionen erst mit autoplay geändert werden
                    # self.main_window.update_speaker_array_calculations()
                        
                    self.update_delay_input_fields(speaker_array_id)

        except ValueError:
            pass
    
    def on_ArrayX_changed(self):
        """Handler für Änderungen der Array X-Position"""
        try:
            value = round(float(self.array_x_edit.text()) if self.array_x_edit.text() else 0.0, 2)
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is None:
                return
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if not speaker_array:
                return
            
            # Initialisiere, falls nicht vorhanden
            if not hasattr(speaker_array, 'array_position_x'):
                speaker_array.array_position_x = 0.0
            
            # Prüfe ob sich der Wert geändert hat
            if speaker_array.array_position_x == value:
                self.array_x_edit.setText(f"{value:.2f}")
                return
            
            speaker_array.array_position_x = value
            
            # Bei Flown-Systemen: Setze die absolute Position für alle Quellen (wie die alten Felder)
            if hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown":
                speaker_array.source_position_x = np.full(speaker_array.number_of_sources, value, dtype=float)
            
            self.array_x_edit.setText(f"{value:.2f}")
            
            # Lösche Hash-Cache für dieses Array, damit Neuberechnung ausgelöst wird
            if hasattr(self.main_window, 'calculation_handler'):
                array_id = speaker_array.id
                if array_id in self.main_window.calculation_handler._speaker_position_hashes:
                    del self.main_window.calculation_handler._speaker_position_hashes[array_id]
            
            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
        except ValueError:
            if hasattr(self, 'main_window'):
                selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
                if selected_speaker_array_id:
                    speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                    if speaker_array and hasattr(speaker_array, 'array_position_x'):
                        self.array_x_edit.setText(f"{speaker_array.array_position_x:.2f}")
    
    def on_ArrayY_changed(self):
        """Handler für Änderungen der Array Y-Position"""
        try:
            value = round(float(self.array_y_edit.text()) if self.array_y_edit.text() else 0.0, 2)
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is None:
                return
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if not speaker_array:
                return
            
            # Initialisiere, falls nicht vorhanden
            if not hasattr(speaker_array, 'array_position_y'):
                speaker_array.array_position_y = 0.0
            
            # Prüfe ob sich der Wert geändert hat
            if speaker_array.array_position_y == value:
                self.array_y_edit.setText(f"{value:.2f}")
                return
            
            speaker_array.array_position_y = value
            
            # Bei Flown-Systemen: Setze die absolute Position für alle Quellen (wie die alten Felder)
            if hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown":
                speaker_array.source_position_y = np.full(speaker_array.number_of_sources, value, dtype=float)
            
            self.array_y_edit.setText(f"{value:.2f}")
            
            # Lösche Hash-Cache für dieses Array, damit Neuberechnung ausgelöst wird
            if hasattr(self.main_window, 'calculation_handler'):
                array_id = speaker_array.id
                if array_id in self.main_window.calculation_handler._speaker_position_hashes:
                    del self.main_window.calculation_handler._speaker_position_hashes[array_id]
            
            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
        except ValueError:
            if hasattr(self, 'main_window'):
                selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
                if selected_speaker_array_id:
                    speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                    if speaker_array and hasattr(speaker_array, 'array_position_y'):
                        self.array_y_edit.setText(f"{speaker_array.array_position_y:.2f}")
    
    def on_ArrayZ_changed(self):
        """Handler für Änderungen der Array Z-Position"""
        try:
            value = round(float(self.array_z_edit.text()) if self.array_z_edit.text() else 0.0, 2)
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is None:
                return
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if not speaker_array:
                return
            
            # Initialisiere, falls nicht vorhanden
            if not hasattr(speaker_array, 'array_position_z'):
                speaker_array.array_position_z = 0.0
            
            # Prüfe ob sich der Wert geändert hat
            if speaker_array.array_position_z == value:
                self.array_z_edit.setText(f"{value:.2f}")
                return
            
            speaker_array.array_position_z = value
            
            # Bei Flown-Systemen: Setze die absolute Position für alle Quellen (wie die alten Felder)
            if hasattr(speaker_array, 'configuration') and speaker_array.configuration and speaker_array.configuration.lower() == "flown":
                speaker_array.source_position_z_flown = np.full(speaker_array.number_of_sources, value, dtype=float)
            
            self.array_z_edit.setText(f"{value:.2f}")
            
            # Lösche Hash-Cache für dieses Array, damit Neuberechnung ausgelöst wird
            if hasattr(self.main_window, 'calculation_handler'):
                array_id = speaker_array.id
                if array_id in self.main_window.calculation_handler._speaker_position_hashes:
                    del self.main_window.calculation_handler._speaker_position_hashes[array_id]
            
            # Aktualisiere Lautsprecher-Overlays (immer)
            self.update_speaker_overlays()
            
            # Neu berechnen nur wenn autocalc aktiv
            if self.is_autocalc_active():
                self.main_window.update_speaker_array_calculations()
            
        except ValueError:
            if hasattr(self, 'main_window'):
                selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
                if selected_speaker_array_id:
                    speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                    if speaker_array and hasattr(speaker_array, 'array_position_z'):
                        self.array_z_edit.setText(f"{speaker_array.array_position_z:.2f}")


    def on_Gain_changed(self):
        try:
            value = float(self.gain_edit.text())
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is not None:
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                if speaker_array:
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.gain == value:
                        return  # Keine Änderung, nichts tun
                    
                    speaker_array.gain = value
                    self.settings.update_speaker_array_gain(selected_speaker_array_id, value)
                    
                    # Neu berechnen nur wenn autocalc aktiv (keine Overlay-Aktualisierung)
                    if self.is_autocalc_active():
                        self.main_window.update_speaker_array_calculations()
                    
                    speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
                    if speakerspecs_instance:
                        self.update_input_fields(speakerspecs_instance)
                        self.update_widgets(speakerspecs_instance)
        except ValueError:
            # Behandeln Sie ungültige Eingaben hier
            pass

    def on_Polarity_changed(self, state):
        """
        Handler für Änderungen der Polaritäts-Checkbox.
        Setzt alle Werte im source_polarity Array auf True (invertiert) oder False (normal).
        
        Args:
            state: Der Zustand der Checkbox (Qt.Checked oder Qt.Unchecked)
        """
        try:
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            if selected_speaker_array_id is not None:
                speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
                if speaker_array:
                    # Setze den neuen Wert
                    polarity_inverted = (state == Qt.Checked)
                    
                    # Prüfe ob sich der Wert geändert hat (vergleiche mit dem ersten Wert im Array)
                    if hasattr(speaker_array, 'source_polarity') and len(speaker_array.source_polarity) > 0:
                        if speaker_array.source_polarity[0] == polarity_inverted:
                            return  # Keine Änderung, nichts tun
                    
                    # Setze alle source_polarity Werte entsprechend
                    if hasattr(speaker_array, 'source_polarity'):
                        speaker_array.source_polarity[:] = polarity_inverted
                    
                    # Neu berechnen nur wenn autocalc aktiv (keine Overlay-Aktualisierung)
                    if self.is_autocalc_active():
                        self.main_window.update_speaker_array_calculations()
                    
        except Exception as e:
            print(f"Fehler in on_Polarity_changed: {e}")
            import traceback
            traceback.print_exc()

    def validate_arc_angle(self, speaker_array):
        """
        Validiert und aktualisiert die Arc-Angle und Scale-Factor Eingaben basierend auf der Arc Shape.
        Aktiviert/deaktiviert auch die Delay-Eingabefelder entsprechend.
        
        Args:
            speaker_array: Das aktuelle SpeakerArray-Objekt
        """
        # Definiere die Grenzen für verschiedene Arc Shapes
        ARC_ANGLE_LIMITS = {
            "manual": (0, 0, 0),      # (min, max, dezimalstellen)
            "pointed arc": (0, 140, 1),
            "circular arc": (0, 140, 1),
            "spiral arc": (0, 200, 1),
            "eliptical arc": (0, 200, 1),
            "linear": (-60, 60, 1)
        }

        # Definiere, welche Formen bestimmte Eingaben deaktivieren
        DISABLED_ARC_SHAPES = ["manual", "linear", "pointed arc", "circular arc"]
        DISABLED_ARC_ANGLES = ["manual"]

        arc_shape_lower = speaker_array.arc_shape.lower()

        # Aktiviere/Deaktiviere Eingabefelder
        arc_scale_factor_is_disabled = arc_shape_lower in [shape.lower() for shape in DISABLED_ARC_SHAPES]
        self.ArcScaleFactor.setEnabled(not arc_scale_factor_is_disabled)

        arc_angle_is_disabled = arc_shape_lower in [shape.lower() for shape in DISABLED_ARC_ANGLES]
        self.ArcAngle.setEnabled(not arc_angle_is_disabled)

        # Setze Label-Text
        if arc_shape_lower == "linear":
            self.Label_ArcAngle.setText("Beamsteering angle")
        else:
            self.Label_ArcAngle.setText("Opening angle")
            
        # Aktiviere/deaktiviere Delay-Eingabefelder
        # Nur im "manual" Modus können Delays manuell bearbeitet werden
        is_manual = arc_shape_lower == "manual"
        self.update_delay_fields_state(speaker_array.id, is_manual)


    def on_ArcAngle_text_changed(self, text):
        """
        Validiert die Arc Angle Eingabe sofort während des Tippens.
        """
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if speaker_array and text:
                try:
                    current_value = float(text)
                    arc_shape_lower = speaker_array.arc_shape.lower()
                    
                    # Grenzen für verschiedene Arc Shapes
                    ARC_ANGLE_LIMITS = {
                        "manual": (0, 0, 0),
                        "pointed arc": (0, 140, 1),
                        "circular arc": (0, 140, 1),
                        "spiral arc": (0, 200, 1),
                        "eliptical arc": (0, 200, 1),
                        "linear": (-60, 60, 1)
                    }
                    
                    min_val, max_val, decimals = ARC_ANGLE_LIMITS.get(arc_shape_lower, (0, 360, 1))
                    
                    # Wert außerhalb der Grenzen
                    if current_value < min_val or current_value > max_val:
                        self.ArcAngle.setStyleSheet("background-color: #FFE4E1;")
                        QtCore.QTimer.singleShot(500, lambda: self.reset_arc_angle(speaker_array, decimals))
                    else:
                        self.ArcAngle.setStyleSheet("")
                        
                except ValueError:
                    self.ArcAngle.setStyleSheet("background-color: #FFE4E1;")
                    QtCore.QTimer.singleShot(500, lambda: self.reset_arc_angle(speaker_array, 1))

    def on_flown_x_position_changed(self):
        """Handler für Änderungen der X-Position im Flown Array"""
        try:
            self._debug_signal("on_flown_x_position_changed", f"text={self.position_x_edit.text()}")
            self.position_x_edit.blockSignals(True)
            # Hole das aktuell ausgewählte Array aus dem TreeWidget
            selected_items = self.sources_tree_widget.selectedItems()
            if not selected_items:
                self.position_x_edit.blockSignals(False)
                return
                
            selected_item = selected_items[0]
            selected_array_id = selected_item.data(0, Qt.UserRole)
            if not selected_array_id:
                self.position_x_edit.blockSignals(False)
                return
                
            speaker_array = self.settings.get_speaker_array(selected_array_id)
            if not speaker_array:
                self.position_x_edit.blockSignals(False)
                return
                
            # Hole den neuen X-Wert
            new_x = round(float(self.position_x_edit.text()) if self.position_x_edit.text() else 0.0, 2)

            # Prüfe, ob sich der Wert geändert hat
            if len(speaker_array.source_position_x) > 0 and np.isclose(speaker_array.source_position_x[0], new_x):
                self.position_x_edit.setText(f"{speaker_array.source_position_x[0]:.2f}")
                self.position_x_edit.blockSignals(False)
                return

            # Setze den X-Wert für alle Quellen im Array
            speaker_array.source_position_x = np.full(speaker_array.number_of_sources, new_x, dtype=float)
                
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            self.position_x_edit.setText(f"{new_x:.2f}")
            self.position_x_edit.blockSignals(False)
            
        except Exception as e:
            self.position_x_edit.blockSignals(False)
            print(f"Fehler beim Ändern der X-Position: {str(e)}")

    def on_flown_y_position_changed(self):
        """Handler für Änderungen der Y-Position im Flown Array"""
        try:
            self._debug_signal("on_flown_y_position_changed", f"text={self.position_y_edit.text()}")
            self.position_y_edit.blockSignals(True)
            # Hole das aktuell ausgewählte Array aus dem TreeWidget
            selected_items = self.sources_tree_widget.selectedItems()
            if not selected_items:
                self.position_y_edit.blockSignals(False)
                return
                
            selected_item = selected_items[0]
            selected_array_id = selected_item.data(0, Qt.UserRole)
            if not selected_array_id:
                self.position_y_edit.blockSignals(False)
                return
                
            speaker_array = self.settings.get_speaker_array(selected_array_id)
            if not speaker_array:
                self.position_y_edit.blockSignals(False)
                return
                
            # Hole den neuen Y-Wert
            new_y = round(float(self.position_y_edit.text()) if self.position_y_edit.text() else 0.0, 2)

            # Prüfe, ob sich der Wert geändert hat
            if len(speaker_array.source_position_y) > 0 and np.isclose(speaker_array.source_position_y[0], new_y):
                self.position_y_edit.setText(f"{speaker_array.source_position_y[0]:.2f}")
                self.position_y_edit.blockSignals(False)
                return
            
            # Setze den Y-Wert für alle Quellen im Array
            speaker_array.source_position_y = np.full(speaker_array.number_of_sources, new_y, dtype=float)
                
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            self.position_y_edit.setText(f"{new_y:.2f}")
            self.position_y_edit.blockSignals(False)
            
        except Exception as e:
            self.position_y_edit.blockSignals(False)
            print(f"Fehler beim Ändern der Y-Position: {str(e)}")

    def on_flown_z_position_changed(self):
        """Handler für Änderungen der Z-Position im Flown Array"""
        try:
            self._debug_signal("on_flown_z_position_changed", f"text={self.position_z_edit.text()}")
            self.position_z_edit.blockSignals(True)
            # Hole das aktuell ausgewählte Array aus dem TreeWidget
            selected_items = self.sources_tree_widget.selectedItems()
            if not selected_items:
                self.position_z_edit.blockSignals(False)
                return
                
            selected_item = selected_items[0]
            selected_array_id = selected_item.data(0, Qt.UserRole)
            if not selected_array_id:
                self.position_z_edit.blockSignals(False)
                return
                
            speaker_array = self.settings.get_speaker_array(selected_array_id)
            if not speaker_array:
                self.position_z_edit.blockSignals(False)
                return
                
            # Hole den neuen Z-Wert
            new_z = round(float(self.position_z_edit.text()) if self.position_z_edit.text() else 0.0, 2)

            if hasattr(speaker_array, 'source_position_z_flown') and speaker_array.source_position_z_flown is not None:
                current_z = speaker_array.source_position_z_flown
            else:
                current_z = speaker_array.source_position_z

            # Prüfe, ob sich der Wert geändert hat
            if current_z is not None and len(current_z) > 0 and np.isclose(current_z[0], new_z):
                self.position_z_edit.setText(f"{current_z[0]:.2f}")
                self.position_z_edit.blockSignals(False)
                return

            speaker_array.source_position_z_flown = np.full(
                speaker_array.number_of_sources, new_z, dtype=float
            )
                
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            self.position_z_edit.setText(f"{new_z:.2f}")
            self.position_z_edit.blockSignals(False)
            
        except Exception as e:
            self.position_z_edit.blockSignals(False)
            print(f"Fehler beim Ändern der Z-Position: {str(e)}")

    def on_flown_site_changed(self):
        """Handler für Änderungen der Site im Flown Array"""
        try:
            self._debug_signal("on_flown_site_changed", f"text={self.position_site_edit.text()}")
            self.position_site_edit.blockSignals(True)
            # Hole das aktuell ausgewählte Array aus dem TreeWidget
            selected_items = self.sources_tree_widget.selectedItems()
            if not selected_items:
                self.position_site_edit.blockSignals(False)
                return
                
            selected_item = selected_items[0]
            selected_array_id = selected_item.data(0, Qt.UserRole)
            if not selected_array_id:
                self.position_site_edit.blockSignals(False)
                return
                
            speaker_array = self.settings.get_speaker_array(selected_array_id)
            if not speaker_array:
                self.position_site_edit.blockSignals(False)
                return
                
            # Hole den neuen Site-Wert
            new_site = round(float(self.position_site_edit.text()) if self.position_site_edit.text() else 0.0, 2)
            
            # Prüfe, ob das Attribut source_site existiert und initialisiere es bei Bedarf
            if not hasattr(speaker_array, 'source_site') or speaker_array.source_site is None:
                speaker_array.source_site = np.zeros(speaker_array.number_of_sources, dtype=float)
            elif len(speaker_array.source_site) < speaker_array.number_of_sources:
                speaker_array.source_site = np.append(
                    speaker_array.source_site,
                    np.zeros(speaker_array.number_of_sources - len(speaker_array.source_site))
                )

            if len(speaker_array.source_site) > 0 and np.isclose(speaker_array.source_site[0], new_site):
                self.position_site_edit.setText(f"{float(speaker_array.source_site[0]):.2f}")
                self.position_site_edit.blockSignals(False)
                return
            
            # Setze den Site-Wert für alle Quellen im Array
            for i in range(speaker_array.number_of_sources):
                speaker_array.source_site[i] = new_site
                
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            self.position_site_edit.setText(f"{new_site:.2f}")
            self.position_site_edit.blockSignals(False)
            
        except Exception as e:
            self.position_site_edit.blockSignals(False)
            print(f"Fehler beim Ändern der Site: {str(e)}")

    def on_flown_azimuth_changed(self):
        """Handler für Änderungen des Azimuth im Flown Array"""
        try:
            self._debug_signal("on_flown_azimuth_changed", f"text={self.position_azimuth_edit.text()}")
            self.position_azimuth_edit.blockSignals(True)
            # Hole das aktuell ausgewählte Array aus dem TreeWidget
            selected_items = self.sources_tree_widget.selectedItems()
            if not selected_items:
                self.position_azimuth_edit.blockSignals(False)
                return
                
            selected_item = selected_items[0]
            selected_array_id = selected_item.data(0, Qt.UserRole)
            if not selected_array_id:
                self.position_azimuth_edit.blockSignals(False)
                return
                
            speaker_array = self.settings.get_speaker_array(selected_array_id)
            if not speaker_array:
                self.position_azimuth_edit.blockSignals(False)
                return
                
            # Hole den neuen Azimuth-Wert
            new_azimuth = round(float(self.position_azimuth_edit.text()) if self.position_azimuth_edit.text() else 0.0, 1)

            if len(speaker_array.source_azimuth) > 0 and np.isclose(speaker_array.source_azimuth[0], new_azimuth):
                self.position_azimuth_edit.setText(f"{speaker_array.source_azimuth[0]:.2f}")
                self.position_azimuth_edit.blockSignals(False)
                return
            
            # Setze den Azimuth-Wert für alle Quellen im Array
            speaker_array.source_azimuth = np.full(speaker_array.number_of_sources, new_azimuth, dtype=float)
                
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            self.position_azimuth_edit.setText(f"{new_azimuth:.2f}")
            self.position_azimuth_edit.blockSignals(False)
            
        except Exception as e:
            self.position_azimuth_edit.blockSignals(False)
            print(f"Fehler beim Ändern des Azimuth: {str(e)}")

    def on_source_angle_changed(self, edit, source_index, array_id):
        print("on_source_angle_changed")
        """
        Handler für Änderungen des Winkels einer Quelle.
        
        Args:
            edit (QLineEdit): Das Eingabefeld, das geändert wurde
            source_index (int): Der Index der Quelle
            array_id (str): Die ID des Arrays
        """
        try:
            # Hole das Array
            speaker_array = self.settings.get_speaker_array(array_id)
            if speaker_array is None:
                return
                
            # Hole den neuen Winkel
            new_angle = float(edit.text())
            
            # Setze den neuen Winkel
            speaker_array.source_angle[source_index] = new_angle
            
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            
        except Exception as e:
            print(f"Fehler beim Ändern des Winkels: {str(e)}")



    def reset_arc_angle(self, speaker_array, decimals):
        """
        Setzt den Arc Angle auf den letzten gültigen Wert zurück.
        """
        self.ArcAngle.setStyleSheet("")
        self.ArcAngle.setText(f"{speaker_array.arc_angle:.{decimals}f}")

    def on_ArcAngle_editing_finished(self):
        """
        Wird aufgerufen, wenn die Bearbeitung abgeschlossen ist (Enter/Focus lost).
        Aktualisiert den Wert im speaker_array.
        """
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
       
        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if speaker_array:
                try:
                    value = float(self.ArcAngle.text())
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.arc_angle == value:
                        return  # Keine Änderung, nichts tun
                    
                    speaker_array.arc_angle = value
                    self.settings.update_speaker_array_arc_settings(
                        selected_speaker_array_id, 
                        value,
                        speaker_array.arc_shape,
                        speaker_array.arc_scale_factor
                    )
                    self.main_window.update_speaker_array_calculations()
                except ValueError:
                    pass

            if speakerspecs_instance:
                self.update_input_fields(speakerspecs_instance)
                self.update_widgets(speakerspecs_instance)
                self.update_beamsteering_plot(speakerspecs_instance)


    def on_ArcShape_changed(self, index):
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)
        
        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if speaker_array:
                arc_shape = self.ArcShape.itemText(index)
                speaker_array.arc_shape = arc_shape

                self.settings.update_speaker_array_arc_settings(
                    selected_speaker_array_id, speaker_array.arc_angle, arc_shape, speaker_array.arc_scale_factor)

                # Prüfe, ob Arc Shape auf 'manual' gesetzt ist und rufe die zentrale Funktion auf
                is_manual = arc_shape.lower() == "manual"
                self.update_delay_fields_state(selected_speaker_array_id, is_manual)
                                
                self.main_window.update_speaker_array_calculations()
                
                if speakerspecs_instance:
                    self.update_input_fields(speakerspecs_instance)
                    self.update_widgets(speakerspecs_instance)
                    self.update_beamsteering_plot(speakerspecs_instance)

                # Aktualisiere den Info-Text mit der neuen Methode
                self.update_arc_info_text(arc_shape)

    def update_arc_info_text(self, arc_shape):
        """
        Aktualisiert den Info-Text basierend auf der ausgewählten Bogenform.
        
        Args:
            arc_shape (str): Die ausgewählte Bogenform
        """
        info_texts = {
            "Manual": "",
            "Pointed Arc": "",
            "Circular Arc": "",
            "Spiral Arc": "Ausgangslage 180° / 0.32. Entspricht ca. 90° Öffnungswinkel",
            "Eliptical Arc": "Ausgangslage 180° / 0.32. Entspricht ca. 90° Öffnungswinkel",
            "Linear": ""
        }
        self.Label_ArcInfo.setText(info_texts.get(arc_shape, ""))

    def on_ArcScaleFactor_text_changed(self, text):
        """
        Validiert die Scale Factor Eingabe sofort während des Tippens.
        Einheitliches Limit von 0 bis 1 für alle Arc Shapes.
        """

        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()

        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if speaker_array and text:
                try:
                    current_value = float(text)
                    
                    # Einheitliche Grenzen für alle Arc Shapes
                    min_val, max_val, decimals = 0, 1, 2
                    
                    # Wert außerhalb der Grenzen
                    if current_value < min_val or current_value > max_val:
                        self.ArcScaleFactor.setStyleSheet("background-color: #FFE4E1;")
                        QtCore.QTimer.singleShot(500, lambda: self.reset_scale_factor(speaker_array, decimals))
                    else:
                        self.ArcScaleFactor.setStyleSheet("")
                        
                except ValueError:
                    self.ArcScaleFactor.setStyleSheet("background-color: #FFE4E1;")
                    QtCore.QTimer.singleShot(500, lambda: self.reset_scale_factor(speaker_array, 2))



    def reset_scale_factor(self, speaker_array, decimals):
        """
        Setzt den Scale Factor auf den letzten gültigen Wert zurück.
        """
        self.ArcScaleFactor.setStyleSheet("")
        self.ArcScaleFactor.setText(f"{speaker_array.arc_scale_factor:.{decimals}f}")


    def on_ArcScaleFactor_editing_finished(self):
        """
        Wird aufgerufen, wenn die Scale Factor Bearbeitung abgeschlossen ist.
        """
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        speakerspecs_instance = self.get_speakerspecs_instance(selected_speaker_array_id)

        if selected_speaker_array_id is not None:
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            if speaker_array:
                try:
                    value = float(self.ArcScaleFactor.text())
                    # Prüfe ob sich der Wert geändert hat
                    if speaker_array.arc_scale_factor == value:
                        return  # Keine Änderung, nichts tun
                    
                    speaker_array.arc_scale_factor = value
                    self.settings.update_speaker_array_arc_settings(
                        selected_speaker_array_id, 
                        speaker_array.arc_angle,
                        speaker_array.arc_shape,
                        value
                    )
                    self.main_window.update_speaker_array_calculations()
                except ValueError:
                    pass

    
            if speakerspecs_instance:
                self.update_input_fields(speakerspecs_instance)
                self.update_widgets(speakerspecs_instance)
                self.update_beamsteering_plot(speakerspecs_instance)


    def close_sources_dock_widget(self):
        if hasattr(self, 'sources_dockWidget'):
            self.sources_dockWidget.close()

    def show_context_menu(self, position):
        """Zeigt das Kontextmenü an der Mausposition"""
        item = self.sources_tree_widget.itemAt(position)
        selected_items = self.sources_tree_widget.selectedItems()
        context_menu = QMenu()
        
        # Prüfe, ob mehrere Items ausgewählt sind
        multiple_selected = len(selected_items) > 1
        
        if item or selected_items:
            # Bestimme Item-Typen der ausgewählten Items
            has_groups = False
            has_arrays = False
            
            if selected_items:
                for sel_item in selected_items:
                    item_type = sel_item.data(0, Qt.UserRole + 1)
                    if item_type == "group":
                        has_groups = True
                    else:
                        has_arrays = True
            elif item:
                item_type = item.data(0, Qt.UserRole + 1)
                if item_type == "group":
                    has_groups = True
                else:
                    has_arrays = True
            
            # Einheitliche Reihenfolge: Duplicate, Rename, ---, Change Color (nur Arrays), ---, Delete
            duplicate_action = context_menu.addAction("Duplicate" if not has_groups else "Duplicate Group" if not has_arrays else "Duplicate")
            rename_action = context_menu.addAction("Rename")
            
            # Rename deaktivieren, wenn mehrere Items ausgewählt
            if multiple_selected:
                rename_action.setEnabled(False)
            
            context_menu.addSeparator()
            
            # Change Color nur für Arrays
            if has_arrays and not has_groups:
                change_color_action = context_menu.addAction("Change Color")
                context_menu.addSeparator()
            
            delete_action = context_menu.addAction("Delete" if not has_groups else "Delete Group" if not has_arrays else "Delete")
            
            # Verbinde Aktionen mit Funktionen
            if multiple_selected:
                # Mehrere Items: Führe Aktionen für alle aus
                duplicate_action.triggered.connect(lambda: self._handle_multiple_duplicate(selected_items))
                delete_action.triggered.connect(lambda: self._handle_multiple_delete(selected_items))
                if has_arrays and not has_groups:
                    change_color_action.triggered.connect(lambda: self._handle_multiple_change_color(selected_items))
            else:
                # Einzelnes Item
                if has_groups:
                    duplicate_action.triggered.connect(lambda: self.duplicate_group(item))
                    delete_action.triggered.connect(lambda: self.delete_group(item))
                else:
                    duplicate_action.triggered.connect(lambda: self.duplicate_array(item))
                    delete_action.triggered.connect(lambda: self.delete_array())
                    if has_arrays and not has_groups:
                        change_color_action.triggered.connect(lambda: self.change_array_color(item))
                
                if not multiple_selected:
                    rename_action.triggered.connect(lambda: self.sources_tree_widget.editItem(item))
        else:
            # Kontextmenü für leeren Bereich
            create_group_action = context_menu.addAction("Create Group")
            context_menu.addSeparator()
            create_stack_action = context_menu.addAction("Create Stacked Array")
            create_flown_action = context_menu.addAction("Create Flown Array")
            
            create_group_action.triggered.connect(lambda: self.create_group())
            create_stack_action.triggered.connect(lambda: self.add_stack())
            create_flown_action.triggered.connect(lambda: self.add_flown())
        
        # Zeige Menü an Mausposition
        context_menu.exec_(self.sources_tree_widget.viewport().mapToGlobal(position))



    def create_checkbox(self, checked=False, tristate=False):
        """
        Erstellt eine Checkbox mit kleinerer Größe.
        
        Args:
            checked: Initialer Checked-Zustand
            tristate: Wenn True, aktiviert Tristate-Modus (für teilweise aktivierte Gruppen)
            
        Returns:
            QCheckBox: Die erstellte Checkbox
        """
        checkbox = QCheckBox()
        if tristate:
            checkbox.setTristate(True)
        checkbox.setChecked(checked)
        # Kleinere Checkboxen: 18x18 Pixel
        checkbox.setFixedSize(18, 18)
        return checkbox
    
    def ensure_group_checkboxes(self, item):
        """
        Stellt sicher, dass Checkboxen für eine Gruppe existieren.
        Erstellt sie, falls sie noch nicht vorhanden sind.
        
        Args:
            item: Das TreeWidgetItem für die Gruppe
        """
        # Prüfe, ob es sich um eine Gruppe handelt
        if item.data(0, Qt.UserRole + 1) != "group":
            return
        
        # Prüfe, ob Checkboxen bereits existieren
        mute_checkbox = self.sources_tree_widget.itemWidget(item, 1)
        hide_checkbox = self.sources_tree_widget.itemWidget(item, 2)
        
        # Erstelle Checkboxen, falls sie nicht existieren
        if not mute_checkbox:
            mute_checkbox = self.create_checkbox(False, tristate=True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 1)
            mute_checkbox.stateChanged.connect(lambda state, g_item=item: self.on_group_mute_changed(g_item, state))
            self.sources_tree_widget.setItemWidget(item, 1, mute_checkbox)
        else:
            # Stelle sicher, dass die Checkbox die richtige Größe hat
            mute_checkbox.setFixedSize(18, 18)
            if not mute_checkbox.isTristate():
                mute_checkbox.setTristate(True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 1)
        
        if not hide_checkbox:
            hide_checkbox = self.create_checkbox(False, tristate=True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 2)
            hide_checkbox.stateChanged.connect(lambda state, g_item=item: self.on_group_hide_changed(g_item, state))
            self.sources_tree_widget.setItemWidget(item, 2, hide_checkbox)
        else:
            # Stelle sicher, dass die Checkbox die richtige Größe hat
            hide_checkbox.setFixedSize(18, 18)
            if not hide_checkbox.isTristate():
                hide_checkbox.setTristate(True)
            # Aktualisiere Checkbox-Zustand basierend auf Child-Items
            self._update_group_checkbox_state(item, 2)
    
    def ensure_source_checkboxes(self, item):
        """
        Stellt sicher, dass Checkboxen für ein Source-Item existieren.
        Erstellt sie, falls sie noch nicht vorhanden sind.
        
        Args:
            item: Das TreeWidgetItem für das Source-Array
        """
        # Prüfe, ob es sich um ein Source-Item handelt (keine Gruppe)
        if item.data(0, Qt.UserRole + 1) == "group":
            return
        
        array_id = item.data(0, Qt.UserRole)
        if array_id is None:
            return
        
        # Prüfe, ob Checkboxen bereits existieren
        mute_checkbox = self.sources_tree_widget.itemWidget(item, 1)
        hide_checkbox = self.sources_tree_widget.itemWidget(item, 2)
        
        # Erstelle Checkboxen, falls sie nicht existieren
        if not mute_checkbox:
            speaker_array = self.settings.get_speaker_array(array_id)
            checked = speaker_array.mute if speaker_array else False
            mute_checkbox = self.create_checkbox(checked)
            mute_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_mute_state(id, state))
            self.sources_tree_widget.setItemWidget(item, 1, mute_checkbox)
        else:
            # Stelle sicher, dass die Checkbox die richtige Größe hat
            mute_checkbox.setFixedSize(18, 18)
        
        if not hide_checkbox:
            speaker_array = self.settings.get_speaker_array(array_id)
            checked = speaker_array.hide if speaker_array else False
            hide_checkbox = self.create_checkbox(checked)
            hide_checkbox.stateChanged.connect(lambda state, id=array_id: self.update_hide_state(id, state))
            self.sources_tree_widget.setItemWidget(item, 2, hide_checkbox)
        else:
            # Stelle sicher, dass die Checkbox die richtige Größe hat
            hide_checkbox.setFixedSize(18, 18)
        
        speaker_array = self.settings.get_speaker_array(array_id)
        self._update_color_indicator(item, speaker_array)
    
    def _find_item_by_array_id(self, parent_item, array_id):
        """
        Findet ein TreeWidgetItem anhand seiner array_id (rekursiv durchsucht den Tree).
        
        Args:
            parent_item: Das Parent-Item, ab dem gesucht werden soll (kann Top-Level Item sein)
            array_id: Die ID des gesuchten Arrays (aus Qt.UserRole)
            
        Returns:
            QTreeWidgetItem oder None, falls nicht gefunden
        """
        if not parent_item:
            return None
        
        try:
            # Prüfe das aktuelle Item
            if parent_item.data(0, Qt.UserRole) == array_id:
                return parent_item
            
            # Suche rekursiv in Children
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child:
                    result = self._find_item_by_array_id(child, array_id)
                    if result:
                        return result
        except RuntimeError:
            # Item wurde gelöscht, überspringe
            return None
        
        return None
    
    def _update_group_checkbox_state(self, group_item, column):
        """
        Aktualisiert den Zustand einer Gruppen-Checkbox basierend auf den Child-Items.
        Setzt PartiallyChecked (Klammer), wenn einige Child-Items checked und andere unchecked sind.
        Rekursiv für Untergruppen.
        
        Args:
            group_item: Das Gruppen-Item
            column: Die Spalte (1=Mute, 2=Hide)
        """
        if not group_item:
            return
        
        checkbox = self.sources_tree_widget.itemWidget(group_item, column)
        if not checkbox:
            return
        
        # Sammle alle Child-Checkboxen (rekursiv für Untergruppen)
        checked_count = 0
        unchecked_count = 0
        total_count = 0
        
        def count_child_checkboxes(item):
            nonlocal checked_count, unchecked_count, total_count
            for i in range(item.childCount()):
                child = item.child(i)
                child_type = child.data(0, Qt.UserRole + 1)
                child_checkbox = self.sources_tree_widget.itemWidget(child, column)
                
                if child_checkbox:
                    total_count += 1
                    if child_checkbox.isChecked():
                        checked_count += 1
                    else:
                        unchecked_count += 1
                
                # Rekursiv für Untergruppen
                if child_type == "group":
                    count_child_checkboxes(child)
        
        count_child_checkboxes(group_item)
        
        # Setze Checkbox-Zustand basierend auf Child-Items
        checkbox.blockSignals(True)
        if total_count == 0:
            # Keine Child-Items: Zustand bleibt unverändert
            pass
        elif checked_count == total_count:
            # Alle Child-Items sind checked
            checkbox.setCheckState(Qt.Checked)
        elif unchecked_count == total_count:
            # Alle Child-Items sind unchecked
            checkbox.setCheckState(Qt.Unchecked)
        else:
            # Gemischter Zustand: PartiallyChecked (Klammer)
            checkbox.setCheckState(Qt.PartiallyChecked)
        checkbox.blockSignals(False)
    
    def validate_all_checkboxes(self):
        """
        Überprüft alle Items im TreeWidget und stellt sicher, dass Checkboxen vorhanden sind.
        Wird nach Änderungen am TreeWidget aufgerufen.
        """
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            item = self.sources_tree_widget.topLevelItem(i)
            if item:
                # Prüfe, ob es eine Gruppe oder ein Source-Item ist
                if item.data(0, Qt.UserRole + 1) == "group":
                    self.ensure_group_checkboxes(item)
                else:
                    self.ensure_source_checkboxes(item)
                
                # Durchlaufe alle Child-Items (in Gruppen)
                for j in range(item.childCount()):
                    child = item.child(j)
                    if child:
                        self.ensure_source_checkboxes(child)
    
    def _update_all_group_checkbox_states(self):
        """
        Aktualisiert alle Gruppen-Checkbox-Zustände basierend auf den Child-Items.
        Wird nach dem Laden von Dateien aufgerufen, um sicherzustellen, dass die Zustände korrekt sind.
        """
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return
        
        def _update_group_states_recursive(item):
            """Rekursiv alle Gruppen-Checkbox-Zustände aktualisieren"""
            if not item:
                return
            
            item_type = item.data(0, Qt.UserRole + 1)
            
            # Wenn es eine Gruppe ist, aktualisiere ihre Checkbox-Zustände
            if item_type == "group":
                # Aktualisiere alle Spalten (1=Mute, 2=Hide)
                for column in [1, 2]:
                    self._update_group_checkbox_state(item, column)
            
            # Rekursiv für alle Childs
            for idx in range(item.childCount()):
                _update_group_states_recursive(item.child(idx))
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            _update_group_states_recursive(self.sources_tree_widget.topLevelItem(i))
    
    def adjust_column_width_to_content(self):
        """Setzt die Spaltenbreiten auf feste Werte (gleich wie Surface UI)"""
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return
        # Gleiche Spaltenbreiten wie in Surface UI
        self.sources_tree_widget.setColumnWidth(0, 180)  # Name
        self.sources_tree_widget.setColumnWidth(1, 25)   # Mute
        self.sources_tree_widget.setColumnWidth(2, 25)   # Hide
        self.sources_tree_widget.setColumnWidth(3, 25)   # Color

    def _apply_group_item_style(self, item):
        """Setzt gemeinsame Formatierung für Gruppen-Items."""
        if not item:
            return
        font = item.font(0)
        if not font.bold():
            font.setBold(True)
        item.setFont(0, font)
        # Erhöhe Item-Höhe für mehr Abstand zwischen Gruppen
        from PyQt5.QtCore import QSize
        item.setSizeHint(0, QSize(0, 32))  # Standard ist ~24px, erhöht auf 32px

    def _update_color_indicator(self, item, speaker_array):
        """Erstellt oder aktualisiert das Farb-Quadrat eines Source-Items."""
        if not item or not hasattr(self, "sources_tree_widget"):
            return
        color = getattr(speaker_array, "color", None) if speaker_array else None
        if not color:
            return
        color_label = self.sources_tree_widget.itemWidget(item, 3)
        if color_label is None:
            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setToolTip("Array-Farbe")
            self.sources_tree_widget.setItemWidget(item, 3, color_label)
        color_label.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")

    def create_group(self):
        """Erstellt eine neue Gruppe im TreeWidget"""
        # Zähle bestehende Gruppen
        group_count = 0
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            item = self.sources_tree_widget.topLevelItem(i)
            if item and item.data(0, Qt.UserRole + 1) == "group":
                group_count += 1
        
        group_name = f"Group {group_count + 1}"
        
        # Erstelle neues Group Item
        group_item = QTreeWidgetItem(self.sources_tree_widget, [group_name])
        group_item.setFlags(group_item.flags() | Qt.ItemIsEditable)
        group_item.setData(0, Qt.UserRole + 1, "group")  # Marker für Gruppe
        group_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
        self._apply_group_item_style(group_item)
        
        # Erstelle Checkboxen für die Gruppe
        mute_checkbox = self.create_checkbox(False)
        hide_checkbox = self.create_checkbox(False)
        
        # Verbinde Checkboxen mit Handler
        mute_checkbox.stateChanged.connect(lambda state, g_item=group_item: self.on_group_mute_changed(g_item, state))
        hide_checkbox.stateChanged.connect(lambda state, g_item=group_item: self.on_group_hide_changed(g_item, state))
        
        self.sources_tree_widget.setItemWidget(group_item, 1, mute_checkbox)
        self.sources_tree_widget.setItemWidget(group_item, 2, hide_checkbox)
        
        # Kein Farb-Quadrat für Gruppen
        group_item.setExpanded(True)
        
        # Passe Spaltenbreite an den Inhalt an
        self.adjust_column_width_to_content()
        
        return group_item
    
    def delete_group(self, group_item):
        """Löscht eine Gruppe und alle ihre Childs"""
        if not group_item:
            return
        
        # Prüfe, ob es wirklich eine Gruppe ist
        if group_item.data(0, Qt.UserRole + 1) != "group":
            return
        
        # Sammle alle Child-Array-IDs, die gelöscht werden müssen
        child_array_ids = []
        for i in range(group_item.childCount()):
            child = group_item.child(i)
            array_id = child.data(0, Qt.UserRole)
            if array_id is not None:
                child_array_ids.append(array_id)
        
        # Entferne alle Childs aus dem TreeWidget
        while group_item.childCount() > 0:
            child = group_item.child(0)
            group_item.removeChild(child)
        
        # Entferne die Gruppe selbst
        parent = group_item.parent()
        if parent:
            parent.removeChild(group_item)
        else:
            index = self.sources_tree_widget.indexOfTopLevelItem(group_item)
            if index != -1:
                self.sources_tree_widget.takeTopLevelItem(index)
        
        # Lösche die Arrays aus den Einstellungen
        for array_id in child_array_ids:
            # Entferne das entsprechende speakerspecs-Objekt
            found_instance = None
            for instance in self.speakerspecs_instance:
                if instance['id'] == array_id:
                    found_instance = instance
                    break
            
            if found_instance:
                self.speakerspecs_instance.remove(found_instance)
                if 'scroll_area' in found_instance:
                    found_instance['scroll_area'].setParent(None)
            
            # Entferne das Array aus den Einstellungen
            self.settings.remove_speaker_array(array_id)
        
        # Aktualisiere die Berechnungen
        if child_array_ids:
            self.main_window.update_speaker_array_calculations()
        
        # Validiere alle Checkboxen nach dem Löschen
        self.validate_all_checkboxes()
        
        # Passe Spaltenbreite an
        self.adjust_column_width_to_content()
    
    def duplicate_group(self, group_item, update_calculations=True):
        """Dupliziert eine Gruppe mit allen ihren Child-Arrays
        
        Args:
            group_item: Das zu duplizierende Gruppen-Item
            update_calculations: Wenn True, werden Berechnungen am Ende ausgeführt (Standard: True)
        """
        if not group_item:
            return
        
        # Blockiere Signale während der Duplikation, um Berechnungen zu vermeiden
        self.sources_tree_widget.blockSignals(True)
        
        try:
            # Prüfe, ob es wirklich eine Gruppe ist
            if group_item.data(0, Qt.UserRole + 1) != "group":
                return
            
            # Sammle alle Child-Array-IDs
            child_array_ids = []
            for i in range(group_item.childCount()):
                child = group_item.child(i)
                array_id = child.data(0, Qt.UserRole)
                if array_id is not None:
                    child_array_ids.append(array_id)
            
            if not child_array_ids:
                return  # Keine Arrays zum Duplizieren
            
            # Hole Gruppen-Informationen
            group_name = group_item.text(0)
            mute_checkbox = self.sources_tree_widget.itemWidget(group_item, 1)
            hide_checkbox = self.sources_tree_widget.itemWidget(group_item, 2)
            group_mute = mute_checkbox.isChecked() if mute_checkbox else False
            group_hide = hide_checkbox.isChecked() if hide_checkbox else False
            
            # Dupliziere alle Arrays in der Gruppe
            duplicated_array_ids = []
            duplicated_items = []
            
            for array_id in child_array_ids:
                original_array = self.settings.get_speaker_array(array_id)
                if not original_array:
                    continue
                
                # Erstelle neue Array-ID
                new_array_id = 1
                while new_array_id in self.settings.get_all_speaker_array_ids():
                    new_array_id += 1
                
                # Dupliziere das Array
                self.settings.duplicate_speaker_array(array_id, new_array_id)
                new_array = self.settings.get_speaker_array(new_array_id)
                new_array.name = f"copy of {original_array.name}"
                
                duplicated_array_ids.append(new_array_id)
                
                # Erstelle neues TreeWidget Item (OHNE Parent, damit es später als Child hinzugefügt werden kann)
                new_array_item = QTreeWidgetItem([new_array.name])
                new_array_item.setFlags(new_array_item.flags() | Qt.ItemIsEditable)
                new_array_item.setData(0, Qt.UserRole, new_array_id)
                new_array_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
                
                # Erstelle Checkboxen
                mute_checkbox = self.create_checkbox(new_array.mute)
                hide_checkbox = self.create_checkbox(new_array.hide)
                self.sources_tree_widget.setItemWidget(new_array_item, 1, mute_checkbox)
                self.sources_tree_widget.setItemWidget(new_array_item, 2, hide_checkbox)
                self._update_color_indicator(new_array_item, new_array)
                
                mute_checkbox.stateChanged.connect(lambda state, id=new_array_id: self.update_mute_state(id, state))
                hide_checkbox.stateChanged.connect(lambda state, id=new_array_id: self.update_hide_state(id, state))
                
                # Erstelle neue Speakerspecs-Instanz
                speakerspecs_instance = self.create_speakerspecs_instance(new_array_id)
                self.init_ui(speakerspecs_instance, self.speaker_tab_layout)
                self.add_speakerspecs_instance(new_array_id, speakerspecs_instance)
                
                duplicated_items.append(new_array_item)
            
            # Erstelle neue Gruppe
            new_group_name = f"copy of {group_name}"
            
            new_group_item = QTreeWidgetItem([new_group_name])
            new_group_item.setFlags(new_group_item.flags() | Qt.ItemIsEditable)
            new_group_item.setData(0, Qt.UserRole + 1, "group")
            new_group_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
            self._apply_group_item_style(new_group_item)
            
            # Erstelle Checkboxen für neue Gruppe
            new_mute_checkbox = self.create_checkbox(group_mute)
            new_hide_checkbox = self.create_checkbox(group_hide)
            
            # Verbinde Checkboxen
            new_mute_checkbox.stateChanged.connect(lambda state, g_item=new_group_item: self.on_group_mute_changed(g_item, state))
            new_hide_checkbox.stateChanged.connect(lambda state, g_item=new_group_item: self.on_group_hide_changed(g_item, state))
            
            self.sources_tree_widget.setItemWidget(new_group_item, 1, new_mute_checkbox)
            self.sources_tree_widget.setItemWidget(new_group_item, 2, new_hide_checkbox)
            
            # Füge alle duplizierten Items zur neuen Gruppe hinzu (BEVOR die Gruppe eingefügt wird)
            for array_item in duplicated_items:
                new_group_item.addChild(array_item)
                # Stelle sicher, dass Checkboxen existieren
                self.ensure_source_checkboxes(array_item)
            
            # Füge die neue Gruppe direkt unterhalb der Original-Gruppe ein
            parent = group_item.parent()
            if parent:
                # Gruppe ist in einer anderen Gruppe - füge direkt nach dem Original ein
                parent_index = parent.indexOfChild(group_item)
                parent.insertChild(parent_index + 1, new_group_item)
            else:
                # Top-Level-Gruppe - füge direkt nach dem Original ein
                top_level_index = self.sources_tree_widget.indexOfTopLevelItem(group_item)
                self.sources_tree_widget.insertTopLevelItem(top_level_index + 1, new_group_item)
            
            new_group_item.setExpanded(True)
            
            # Validiere alle Checkboxen nach Duplizierung der Gruppe
            self.validate_all_checkboxes()
            
            # Passe Spaltenbreite an
            self.adjust_column_width_to_content()
            
            # Aktualisiere Berechnungen erst nach allen UI-Updates (nur wenn gewünscht)
            if update_calculations:
                self.main_window.update_speaker_array_calculations()
        finally:
            # Signale wieder aktivieren
            self.sources_tree_widget.blockSignals(False)
    
    def _update_group_child_checkboxes(self, group_item, column, checked, skip_calculations=False, skip_state_update=False):
        """
        Aktualisiert rekursiv die Checkboxen aller Child-Arrays/-Gruppen einer Gruppe.
        Aktualisiert auch die tatsächlichen Array-Daten.
        
        Args:
            group_item: Das Gruppen-Item
            column: Die Spalte (1=Mute, 2=Hide)
            checked: Der neue Checkbox-Zustand
            skip_calculations: Wenn True, werden keine Berechnungen ausgelöst (für Gruppen-Updates)
            skip_state_update: Wenn True, wird _update_group_checkbox_state nicht aufgerufen (für Gruppen-Updates)
        """
        if not group_item:
            return
        
        for i in range(group_item.childCount()):
            child = group_item.child(i)
            child_type = child.data(0, Qt.UserRole + 1)
            checkbox = self.sources_tree_widget.itemWidget(child, column)
            
            if child_type == "array":
                # Array: Aktualisiere Checkbox und Array-Daten
                array_id = child.data(0, Qt.UserRole)
                if array_id is not None and checkbox:
                    checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    checkbox.blockSignals(False)
                    
                    # Aktualisiere Array-Daten
                    speaker_array = self.settings.get_speaker_array(array_id)
                    if speaker_array:
                        if column == 1:  # Mute
                            speaker_array.mute = checked
                        elif column == 2:  # Hide
                            speaker_array.hide = checked
            elif child_type == "group":
                # Gruppe: Aktualisiere Checkbox und rekursiv alle Childs
                if checkbox:
                    checkbox.blockSignals(True)
                    # Verwende setCheckState für tristate-Checkboxen
                    checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    checkbox.blockSignals(False)
                
                # Rekursiv für Untergruppen
                self._update_group_child_checkboxes(child, column, checked, skip_calculations=skip_calculations, skip_state_update=skip_state_update)
        
        # Aktualisiere Gruppen-Checkbox-Zustand nach Änderung der Child-Items
        # Überspringe, wenn skip_state_update=True (wird bereits explizit gesetzt)
        if not skip_state_update:
            self._update_group_checkbox_state(group_item, column)
    
    def on_group_mute_changed(self, group_item, state):
        """Handler für Mute-Checkbox von Gruppen - setzt Mute für alle Childs. Bei Mehrfachauswahl werden alle ausgewählten Gruppen aktualisiert."""
        if not group_item:
            return
        
        mute_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.sources_tree_widget.selectedItems()
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
        
        # Wende Mute auf alle Gruppen und deren Childs an
        for group in groups_to_update:
            # Setze Gruppen-Checkbox explizit auf den neuen Zustand
            group_checkbox = self.sources_tree_widget.itemWidget(group, 1)
            if group_checkbox:
                group_checkbox.blockSignals(True)
                # Verwende setCheckState für tristate-Checkboxen
                group_checkbox.setCheckState(Qt.Checked if mute_value else Qt.Unchecked)
                group_checkbox.blockSignals(False)
            
            # Aktualisiere alle Child-Checkboxen und Array-Daten (ohne Berechnungen, bis alle Zustände gespeichert sind)
            # skip_state_update=True: Überspringe _update_group_checkbox_state, da wir die Checkbox bereits explizit gesetzt haben
            self._update_group_child_checkboxes(group, 1, mute_value, skip_calculations=True, skip_state_update=True)
            
            # Aktualisiere Gruppen-Checkbox-Zustand explizit NACH allen Child-Updates
            # (um sicherzustellen, dass der Zustand korrekt ist, auch wenn alle Childs aktualisiert wurden)
            self._update_group_checkbox_state(group, 1)
            
            # Aktualisiere Parent-Gruppen rekursiv
            parent = group.parent()
            while parent:
                self._update_group_checkbox_state(parent, 1)
                parent = parent.parent()
        
        # 🎯 WICHTIG: Aktualisiere Berechnungen erst NACH allen Zustandsänderungen
        self.main_window.update_speaker_array_calculations()
    
    def on_group_hide_changed(self, group_item, state):
        """Handler für Hide-Checkbox von Gruppen - setzt Hide für alle Childs. Bei Mehrfachauswahl werden alle ausgewählten Gruppen aktualisiert."""
        if not group_item:
            return
        
        hide_value = (state == Qt.Checked)
        
        # Prüfe, ob mehrere Items ausgewählt sind
        selected_items = self.sources_tree_widget.selectedItems()
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
        
        # Wende Hide auf alle Gruppen und deren Childs an
        for group in groups_to_update:
            # Setze Gruppen-Checkbox explizit auf den neuen Zustand
            group_checkbox = self.sources_tree_widget.itemWidget(group, 2)
            if group_checkbox:
                group_checkbox.blockSignals(True)
                # Verwende setCheckState für tristate-Checkboxen
                group_checkbox.setCheckState(Qt.Checked if hide_value else Qt.Unchecked)
                group_checkbox.blockSignals(False)
            
            # Aktualisiere alle Child-Checkboxen und Array-Daten (ohne Berechnungen, bis alle Zustände gespeichert sind)
            # skip_state_update=True: Überspringe _update_group_checkbox_state, da wir die Checkbox bereits explizit gesetzt haben
            self._update_group_child_checkboxes(group, 2, hide_value, skip_calculations=True, skip_state_update=True)
            
            # Aktualisiere Gruppen-Checkbox-Zustand explizit NACH allen Child-Updates
            # (um sicherzustellen, dass der Zustand korrekt ist, auch wenn alle Childs aktualisiert wurden)
            self._update_group_checkbox_state(group, 2)
            
            # Aktualisiere Parent-Gruppen rekursiv
            parent = group.parent()
            while parent:
                self._update_group_checkbox_state(parent, 2)
                parent = parent.parent()
        
        # 🎯 WICHTIG: Aktualisiere Berechnungen erst NACH allen Zustandsänderungen
        self.main_window.update_speaker_array_calculations()

    def change_array_color(self, item, update_calculations=True):
        """
        Ändert die Farbe des ausgewählten Arrays
        
        Args:
            item: Das TreeWidgetItem für das Array
            update_calculations: Wenn True, wird eine vollständige Neuberechnung ausgelöst.
                                Wenn False, wird nur der Impulse-Plot aktualisiert.
        """
        if item:
            array_id = item.data(0, Qt.UserRole)
            speaker_array = self.settings.get_speaker_array(array_id)
            if speaker_array:
                # Generiere neue Farbe direkt vom SpeakerArray Objekt
                new_color = speaker_array._generate_random_color()
                speaker_array.color = new_color
                
                self._update_color_indicator(item, speaker_array)
                
                # Aktualisiere die Anzeige
                if update_calculations:
                    # Vollständige Neuberechnung
                    self.main_window.update_speaker_array_calculations()
                else:
                    # Nur Impulse-Plot aktualisieren (keine Neuberechnung)
                    if hasattr(self.main_window, 'impulse_manager'):
                        self.main_window.impulse_manager.update_plot_impulse()
    
    def _handle_multiple_duplicate(self, selected_items):
        """Dupliziert mehrere ausgewählte Items. Berechnungen werden erst am Ende ausgeführt."""
        if not selected_items:
            return
        
        # Blockiere Signale während der gesamten Mehrfach-Duplikation
        self.sources_tree_widget.blockSignals(True)
        
        try:
            # Trenne Gruppen und Arrays
            groups = [item for item in selected_items if item.data(0, Qt.UserRole + 1) == "group"]
            arrays = [item for item in selected_items if item.data(0, Qt.UserRole + 1) != "group"]
            
            # Dupliziere zuerst Arrays, dann Gruppen (ohne Berechnungen während der Duplikation)
            for item in arrays:
                self.duplicate_array(item, update_calculations=False)
            
            for item in groups:
                self.duplicate_group(item, update_calculations=False)
            
            # Validiere alle Checkboxen nach allen Duplikationen
            self.validate_all_checkboxes()
            
            # Passe Spaltenbreite an
            self.adjust_column_width_to_content()
            
            # Aktualisiere Berechnungen erst nach allen UI-Updates
            self.main_window.update_speaker_array_calculations()
        finally:
            # Signale wieder aktivieren
            self.sources_tree_widget.blockSignals(False)
    
    def _handle_multiple_delete(self, selected_items):
        """Löscht mehrere ausgewählte Items"""
        if not selected_items:
            return
        
        # Sammle zuerst alle IDs, bevor wir etwas löschen (sonst werden Items ungültig)
        array_ids_to_delete = []
        group_items_to_delete = []
        speakerspecs_to_remove = []
        
        for item in selected_items:
            if item is None:
                continue
            
            try:
                item_type = item.data(0, Qt.UserRole + 1)
            except RuntimeError:
                # Item wurde bereits gelöscht, überspringe
                continue
            
            if item_type == "group":
                group_items_to_delete.append(item)
            else:
                array_id = item.data(0, Qt.UserRole)
                if array_id is not None:
                    array_ids_to_delete.append((item, array_id))
        
        # Lösche zuerst Arrays
        for item, array_id in array_ids_to_delete:
            try:
                # Entferne das Item aus dem TreeWidget
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.sources_tree_widget.indexOfTopLevelItem(item)
                    if index != -1:
                        self.sources_tree_widget.takeTopLevelItem(index)
                
                # Entferne das Array aus den Einstellungen
                self.settings.remove_speaker_array(array_id)
                
                # Sammle speakerspecs-Objekte zum Entfernen
                for instance in self.speakerspecs_instance:
                    if instance['id'] == array_id:
                        speakerspecs_to_remove.append(instance)
                        break
            except RuntimeError:
                # Item wurde bereits gelöscht, überspringe
                continue
        
        # Entferne speakerspecs-Objekte
        for instance in speakerspecs_to_remove:
            if instance in self.speakerspecs_instance:
                self.speakerspecs_instance.remove(instance)
                if 'scroll_area' in instance:
                    instance['scroll_area'].setParent(None)
        
        # Lösche Gruppen (delete_group ruft keine TreeWidget-Neuaufbau-Funktion auf)
        for item in group_items_to_delete:
            try:
                self.delete_group(item)
            except RuntimeError:
                # Item wurde bereits gelöscht, überspringe
                continue
        
        # Aktualisiere die Berechnungen
        if array_ids_to_delete or group_items_to_delete:
            self.main_window.update_speaker_array_calculations()
        
        # Validiere alle Checkboxen nach dem Löschen
        self.validate_all_checkboxes()
        
        # Passe Spaltenbreite an
        self.adjust_column_width_to_content()
    
    def _handle_multiple_change_color(self, selected_items):
        """
        Ändert die Farbe mehrerer ausgewählter Arrays.
        Löst keine Neuberechnung aus, aktualisiert nur die Impulse-Plot-Linien.
        """
        if not selected_items:
            return
        
        # Nur Arrays, keine Gruppen
        arrays = [item for item in selected_items if item.data(0, Qt.UserRole + 1) != "group"]
        
        if not arrays:
            return
        
        # Ändere Farbe für alle Arrays (ohne Neuberechnung)
        for item in arrays:
            self.change_array_color(item, update_calculations=False)
        
        # Aktualisiere Impulse-Plot einmal am Ende (für alle geänderten Arrays)
        if hasattr(self.main_window, 'impulse_manager'):
            self.main_window.impulse_manager.update_plot_impulse()

    def update_delay_fields_state(self, speaker_array_id, is_manual):
        """
        Aktiviert oder deaktiviert die Delay-Eingabefelder basierend auf dem Arc Shape Modus.
        Delay-Felder sind nur aktiv, wenn Arc Shape auf 'manual' gesetzt ist.
        
        Args:
            speaker_array_id: Die ID des Speaker Arrays
            is_manual: True, wenn Arc Shape auf 'manual' gesetzt ist, sonst False
        """
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            if not speaker_array:
                print(f"Kein Speaker Array mit ID {speaker_array_id} gefunden")
                return
                
            instance = self.get_speakerspecs_instance(speaker_array_id)
            if not instance:
                print(f"Keine Speakerspecs-Instanz für Array {speaker_array_id} gefunden")
                return
                
            if not instance['gridLayout_sources']:
                print(f"Kein gridLayout_sources in Instanz für Array {speaker_array_id}")
                return
                
            
            # Aktiviere/deaktiviere alle Delay-Eingabefelder
            gridLayout_sources = instance['gridLayout_sources']
            
            # Sammle alle QLineEdit-Widgets im Layout
            delay_fields = []
            all_widgets = []
            
            # Durchsuche alle Widgets im Layout
            for i in range(gridLayout_sources.count()):
                item = gridLayout_sources.itemAt(i)
                widget = item.widget()
                if widget:
                    all_widgets.append((widget.objectName(), type(widget).__name__))
                    
                    # Suche nach den Delay-Eingabefeldern unterhalb von Azimuth
                    if isinstance(widget, QLineEdit):
                        # Verschiedene mögliche Namenskonventionen für Delay-Felder
                        if (widget.objectName().startswith("delay_") or 
                            widget.objectName().startswith("time_") or 
                            "delay" in widget.objectName().lower() or 
                            "time" in widget.objectName().lower()):
                            delay_fields.append(widget)
            
            if len(delay_fields) == 0:
                
                # In einem typischen Layout sind die Delay-Felder in bestimmten Zeilen/Spalten
                for row in range(gridLayout_sources.rowCount()):
                    for col in range(gridLayout_sources.columnCount()):
                        item = gridLayout_sources.itemAtPosition(row, col)
                        if item and item.widget() and isinstance(item.widget(), QLineEdit):
                            widget = item.widget()
                            # Prüfe, ob es sich um ein Delay-Feld handeln könnte (basierend auf Position)
                            # Typischerweise sind Delay-Felder in der Zeile nach Azimuth
                            delay_fields.append(widget)
            
            # Aktiviere/deaktiviere alle gefundenen Delay-Felder
            for widget in delay_fields:
                widget.setEnabled(is_manual)
                widget.setReadOnly(not is_manual)  # Zusätzlich ReadOnly setzen
                # Setze einen Tooltip, der erklärt, warum das Feld deaktiviert ist
                if not is_manual:
                    widget.setToolTip("Delay kann nur im 'manual' Arc Shape Modus bearbeitet werden")

                else:
                    widget.setToolTip("Delay in Millisekunden")
                    widget.setStyleSheet("")
            
            # Erzwinge eine Aktualisierung des Layouts
            gridLayout_sources.update()
            if 'scroll_content' in instance:
                instance['scroll_content'].update()
    
        
        except Exception as e:
            print(f"Fehler in update_delay_fields_state: {e}")
            import traceback
            traceback.print_exc()

    def get_speaker_angles(self, speaker_name):
        """
        Holt die verfügbaren Winkel für einen bestimmten Lautsprecher aus den Cabinet-Daten.
        
        Args:
            speaker_name: Name des Lautsprechers
            
        Returns:
            Liste der verfügbaren Winkel oder leere Liste, wenn keine gefunden wurden
        """
        cabinet_data = self.container.data.get('cabinet_data', [])
        speaker_names = self.container.data.get('speaker_names', [])
        
        # Finde den Index des Lautsprechers
        speaker_index = -1
        for i, name in enumerate(speaker_names):
            if name == speaker_name:
                speaker_index = i
                break
        
        if speaker_index == -1 or speaker_index >= len(cabinet_data) or cabinet_data[speaker_index] is None:
            return []
        
        cabinet = cabinet_data[speaker_index]
        angles = []
        
        # Für Listen von Dictionaries
        if isinstance(cabinet, list):
            for cab in cabinet:
                if isinstance(cab, dict) and cab.get('configuration', '').lower() == 'flown':
                    cab_angles = cab.get('angles', [])
                    if cab_angles:
                        angles.extend([str(angle) for angle in cab_angles])
        
        # Für einzelne Dictionaries
        elif isinstance(cabinet, dict) and cabinet.get('configuration', '').lower() == 'flown':
            cab_angles = cabinet.get('angles', [])
            if cab_angles:
                angles = [str(angle) for angle in cab_angles]
        
        return angles

    def update_angle_combobox(self, speaker_array_id, source_index):
        """
        Aktualisiert die Winkel-Combobox basierend auf dem ausgewählten Lautsprecher.
        
        Args:
            speaker_array_id: ID des Speaker Arrays
            source_index: Index der Quelle
        """
        speaker_array = self.settings.get_speaker_array(speaker_array_id)
        if not speaker_array or not hasattr(speaker_array, 'source_polar_pattern'):
            return
        
        instance = self.get_speakerspecs_instance(speaker_array_id)
        if not instance or 'gridLayout_sources' not in instance:
            return
        
        # Hole den aktuellen Lautsprechertyp
        current_pattern = speaker_array.source_polar_pattern[source_index]
        
        # Hole die verfügbaren Winkel für diesen Lautsprecher
        angles = self.get_speaker_angles(current_pattern)
        
        # Suche nach der Winkel-Combobox für diese Quelle
        angle_combo_name = f"angle_combo_{source_index + 1}"
        angle_combo = None
        
        # Durchsuche alle Widgets im Layout
        for i in range(instance['gridLayout_sources'].count()):
            item = instance['gridLayout_sources'].itemAt(i)
            if item.widget() and isinstance(item.widget(), QtWidgets.QComboBox) and item.widget().objectName() == angle_combo_name:
                angle_combo = item.widget()
                break
        
        if angle_combo:
            if not hasattr(speaker_array, 'source_angle') or speaker_array.source_angle is None:
                speaker_array.source_angle = [None] * speaker_array.number_of_sources
            elif len(speaker_array.source_angle) < speaker_array.number_of_sources:
                speaker_array.source_angle.extend([None] * (speaker_array.number_of_sources - len(speaker_array.source_angle)))

            # Blockiere Signale während der Aktualisierung
            angle_combo.blockSignals(True)
            current_angle = angle_combo.currentText() if angle_combo.count() > 0 else ""
            
            # Leere die Combobox und setze Status je nach Position
            angle_combo.clear()
            if source_index == 0:
                angle_combo.addItem("0")
                angle_combo.setCurrentIndex(0)
                angle_combo.setEnabled(False)
                angle_combo.setToolTip("Der oberste Lautsprecher verwendet immer 0°.")
                if hasattr(speaker_array, 'source_angle'):
                    speaker_array.source_angle[source_index] = 0.0
            else:
                angle_combo.addItems(angles)
                angle_combo.setEnabled(True)
                angle_combo.setToolTip("")
                desired_angle = None
                if speaker_array.source_angle[source_index] is not None:
                    desired_angle = str(speaker_array.source_angle[source_index])
                elif current_angle:
                    desired_angle = current_angle

                if desired_angle and desired_angle in angles:
                    angle_combo.setCurrentText(desired_angle)
                elif angle_combo.count() > 0:
                    angle_combo.setCurrentIndex(0)
                    speaker_array.source_angle[source_index] = angle_combo.currentText()
            
            # Aktiviere Signale wieder
            angle_combo.blockSignals(False)

    def on_angle_changed(self, index, source_index, array_id):
        """
        Handler für Änderungen des Winkels in der Combobox.
        
        Args:
            index (int): Der Index des ausgewählten Eintrags in der Combobox
            source_index (int): Der Index der Quelle
            array_id (str): Die ID des Arrays
        """
        try:
            if source_index == 0:
                return
            
            # Hole das Array
            speaker_array = self.settings.get_speaker_array(array_id)
            if speaker_array is None:
                print("  Array nicht gefunden")
                return
                
            # Finde die Combobox
            combobox = None
            for i in range(speaker_array.number_of_sources):
                combo = self.main_window.findChild(QtWidgets.QComboBox, f"angle_combo_{i + 1}")
                if combo and i == source_index:
                    combobox = combo
                    break
            
            if combobox is None:
                print("  Combobox nicht gefunden")
                return
                
            # Hole den ausgewählten Winkel
            angle_text = combobox.currentText()
            
            # Stelle sicher, dass source_angle existiert und die richtige Größe hat
            if not hasattr(speaker_array, 'source_angle') or speaker_array.source_angle is None:
                speaker_array.source_angle = [None] * speaker_array.number_of_sources
            else:
                if isinstance(speaker_array.source_angle, np.ndarray):
                    speaker_array.source_angle = speaker_array.source_angle.tolist()
                if len(speaker_array.source_angle) < speaker_array.number_of_sources:
                    speaker_array.source_angle.extend([None] * (speaker_array.number_of_sources - len(speaker_array.source_angle)))
            
            # Setze den neuen Winkel
            speaker_array.source_angle[source_index] = angle_text
            
            # Aktualisiere die Berechnung
            self.main_window.update_speaker_array_calculations()
            
        except Exception as e:
            print(f"Fehler beim Ändern des Winkels: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_groups_structure(self):
        """
        Speichert die Gruppen-Struktur aus dem TreeWidget in settings.speaker_array_groups.
        """
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return
        
        # Lösche alte Gruppen-Struktur
        self.settings.speaker_array_groups = {}
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            item = self.sources_tree_widget.topLevelItem(i)
            if item and item.data(0, Qt.UserRole + 1) == "group":
                # Es ist eine Gruppe
                group_id = f"group_{i}"  # Eindeutige ID für die Gruppe
                group_name = item.text(0)
                
                # Hole Mute/Hide Status aus den Checkboxen
                mute_checkbox = self.sources_tree_widget.itemWidget(item, 1)
                hide_checkbox = self.sources_tree_widget.itemWidget(item, 2)
                mute = mute_checkbox.isChecked() if mute_checkbox else False
                hide = hide_checkbox.isChecked() if hide_checkbox else False
                
                # Sammle alle Child-Array-IDs
                child_array_ids = []
                for j in range(item.childCount()):
                    child = item.child(j)
                    array_id = child.data(0, Qt.UserRole)
                    if array_id is not None:
                        child_array_ids.append(array_id)
                
                # Speichere Gruppen-Information
                # Behalte vorhandene relative Positionen und Delay/Gain Werte
                existing_data = self.settings.speaker_array_groups.get(group_id, {})
                self.settings.speaker_array_groups[group_id] = {
                    'name': group_name,
                    'mute': mute,
                    'hide': hide,
                    'child_array_ids': child_array_ids,
                    'relative_position': existing_data.get('relative_position', {'x': 0.0, 'y': 0.0, 'z': 0.0}),
                    'relative_delay': existing_data.get('relative_delay', 0.0),
                    'relative_gain': existing_data.get('relative_gain', 0.0)
                }
    
    def _save_expand_state(self):
        """Speichert den Expand/Collapse-Zustand aller Gruppen im TreeWidget."""
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return {}
        
        expand_state = {}
        
        def save_item_state(item, path=""):
            """Rekursiv speichert den Expand-Zustand von Items."""
            if not item:
                return
            
            item_type = item.data(0, Qt.UserRole + 1)
            if item_type == "group":
                # Speichere den Zustand für Gruppen
                group_name = item.text(0)
                item_path = f"{path}/{group_name}" if path else group_name
                expand_state[item_path] = item.isExpanded()
                
                # Rekursiv für Child-Items
                for i in range(item.childCount()):
                    child = item.child(i)
                    save_item_state(child, item_path)
            else:
                # Für Arrays: Rekursiv für Child-Items (falls sie in Gruppen sind)
                for i in range(item.childCount()):
                    child = item.child(i)
                    save_item_state(child, path)
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            item = self.sources_tree_widget.topLevelItem(i)
            save_item_state(item)
        
        return expand_state
    
    def _restore_expand_state(self, expand_state):
        """Stellt den Expand/Collapse-Zustand aller Gruppen im TreeWidget wieder her."""
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
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
                group_name = item.text(0)
                item_path = f"{path}/{group_name}" if path else group_name
                
                if item_path in expand_state:
                    item.setExpanded(expand_state[item_path])
                
                # Rekursiv für Child-Items
                for i in range(item.childCount()):
                    child = item.child(i)
                    restore_item_state(child, item_path)
            else:
                # Für Arrays: Rekursiv für Child-Items (falls sie in Gruppen sind)
                for i in range(item.childCount()):
                    child = item.child(i)
                    restore_item_state(child, path)
        
        # Durchlaufe alle Top-Level-Items
        for i in range(self.sources_tree_widget.topLevelItemCount()):
            item = self.sources_tree_widget.topLevelItem(i)
            restore_item_state(item)
    
    def load_groups_structure(self):
        """
        Lädt die Gruppen-Struktur aus settings.speaker_array_groups und stellt sie im TreeWidget wieder her.
        """
        if not hasattr(self, 'sources_tree_widget') or not self.sources_tree_widget:
            return
        
        if not hasattr(self.settings, 'speaker_array_groups') or not self.settings.speaker_array_groups:
            return
        
        # Speichere aktuellen Expand-Zustand vor dem Neuaufbau
        expand_state = self._save_expand_state()
        
        # Blockiere Signale während des Ladens
        self.sources_tree_widget.blockSignals(True)
        
        try:
            # Erstelle Gruppen und füge Childs hinzu
            for group_id, group_data in self.settings.speaker_array_groups.items():
                group_name = group_data.get('name', f'Group')
                child_array_ids = group_data.get('child_array_ids', [])
                
                # Erstelle Gruppe
                group_item = QTreeWidgetItem(self.sources_tree_widget, [group_name])
                group_item.setFlags(group_item.flags() | Qt.ItemIsEditable)
                group_item.setData(0, Qt.UserRole + 1, "group")
                group_item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
                self._apply_group_item_style(group_item)
                
                # Erstelle Checkboxen für Gruppe
                mute_checkbox = self.create_checkbox(group_data.get('mute', False))
                hide_checkbox = self.create_checkbox(group_data.get('hide', False))
                
                # Verbinde Checkboxen
                mute_checkbox.stateChanged.connect(lambda state, g_item=group_item: self.on_group_mute_changed(g_item, state))
                hide_checkbox.stateChanged.connect(lambda state, g_item=group_item: self.on_group_hide_changed(g_item, state))
                
                self.sources_tree_widget.setItemWidget(group_item, 1, mute_checkbox)
                self.sources_tree_widget.setItemWidget(group_item, 2, hide_checkbox)
                
                # Füge Child-Arrays zur Gruppe hinzu
                for array_id in child_array_ids:
                    # Prüfe zuerst, ob das Array überhaupt noch existiert
                    if array_id not in self.settings.speaker_arrays:
                        continue  # Array wurde gelöscht, überspringe es
                    
                    # Suche nach dem Array-Item im TreeWidget (sowohl Top-Level als auch in Gruppen)
                    array_item = None
                    for i in range(self.sources_tree_widget.topLevelItemCount()):
                        item = self.sources_tree_widget.topLevelItem(i)
                        if item:
                            # Prüfe Top-Level Item
                            if item.data(0, Qt.UserRole) == array_id:
                                array_item = item
                                break
                            # Prüfe auch Child-Items (falls bereits in einer anderen Gruppe)
                            for j in range(item.childCount()):
                                child = item.child(j)
                                if child and child.data(0, Qt.UserRole) == array_id:
                                    array_item = child
                                    break
                            if array_item:
                                break
                    
                    if array_item:
                        # Entferne Item von Top-Level
                        parent = array_item.parent()
                        if parent:
                            parent.removeChild(array_item)
                        else:
                            index = self.sources_tree_widget.indexOfTopLevelItem(array_item)
                            if index != -1:
                                self.sources_tree_widget.takeTopLevelItem(index)
                        
                        # Füge zur Gruppe hinzu
                        group_item.addChild(array_item)
                        
                        # Stelle sicher, dass Checkboxen existieren
                        self.ensure_source_checkboxes(array_item)
            
            # Validiere alle Checkboxen nach dem Laden
            self.validate_all_checkboxes()
            
            # Aktualisiere alle Gruppen-Checkbox-Zustände basierend auf den Childs
            self._update_all_group_checkbox_states()
            
            # Stelle den Expand-Zustand wieder her
            self._restore_expand_state(expand_state)
            
            # Passe Spaltenbreite an
            self.adjust_column_width_to_content()
            
        finally:
            # Entsperre Signale
            self.sources_tree_widget.blockSignals(False)
