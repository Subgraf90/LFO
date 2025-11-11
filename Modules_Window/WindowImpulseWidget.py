from PyQt5.QtWidgets import (QSpacerItem, QDockWidget, QWidget, QVBoxLayout, 
                            QPushButton, QScrollArea, QLineEdit,
                            QHBoxLayout, QSizePolicy, QSplitter, QTreeWidget, 
                            QTreeWidgetItem, QMenu, QApplication, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QDoubleValidator
import numpy as np

from Module_LFO.Modules_Plot.PlotMPLCanvas import MplCanvas
from Module_LFO.Modules_Plot.PlotImpulse import DrawImpulsePlots



class ImpulseInputDockWidget(QDockWidget):
    def __init__(self, main_window, settings, container, calculation_impulse):
        super().__init__("Impulse", main_window)
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.canvases = {}
        
        # Zoom-Status für jeden Plot
        self.zoom_states = {}

        # Debounce-Einstellungen für rechenintensive Updates
        self._calc_debounce_ms = 100
        self._calc_timer = QTimer(self)
        self._calc_timer.setSingleShot(True)
        self._calc_timer.timeout.connect(self._run_debounced_calculation)
        self._pending_recalc = False
        self._pending_speaker_update = False

        # Erstelle DrawImpulsePlots-Instanz
        self.plot_widget = DrawImpulsePlots(settings, container, main_window)
        
        # Verbinde das view_changed Signal
        self.plot_widget.view_changed.connect(self.on_view_changed)

        self.init_ui()

    def schedule_calculation(self, update_speaker_arrays=False):
        """
        Plant die Impulsberechnung (und optional Lautsprecher-Updates) mit Debounce.
        """
        self._pending_recalc = self._pending_recalc or (not update_speaker_arrays)
        self._pending_speaker_update = self._pending_speaker_update or update_speaker_arrays
        self._calc_timer.start(self._calc_debounce_ms)

    def _run_debounced_calculation(self):
        """
        Führt die geplanten Berechnungen nach Ablauf der Debounce-Zeit aus.
        """
        should_recalc = self._pending_recalc
        should_update_speakers = self._pending_speaker_update

        # Flags zurücksetzen bevor Berechnungen starten
        self._pending_recalc = False
        self._pending_speaker_update = False

        if should_recalc:
            self.main_window.calculate_impulse()

        if should_update_speakers:
            self.main_window.update_speaker_array_calculations()

    def init_ui(self):
        dock_widget_content = QWidget()
        self.dock_layout = QVBoxLayout(dock_widget_content)

        # Erstellen Sie einen QSplitter
        splitter = QSplitter(Qt.Vertical)

        # Eingabe-Bereich (oberer Teil)
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setSpacing(2)  
        
        # TreeWidget für Messpunkte
        self.impulse_tree_widget = QTreeWidget()
        self.impulse_tree_widget.setHeaderLabels(["Measurement Point", "Position X (m)", "Position Y (m)", "Time offset (ms)", ""])
        self.impulse_tree_widget.setFixedHeight(120)
        
        # Breitere Spalten
        self.impulse_tree_widget.setColumnWidth(0, 160)
        self.impulse_tree_widget.setColumnWidth(1, 100)
        self.impulse_tree_widget.setColumnWidth(2, 100)
        self.impulse_tree_widget.setColumnWidth(3, 100)
        self.impulse_tree_widget.setColumnWidth(4, 50)
        
        input_layout.addWidget(self.impulse_tree_widget)
        
        # Button-Layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        add_measurement_button = QPushButton("Add Point")
        add_measurement_button.setFixedWidth(100)
        add_measurement_button.clicked.connect(self.add_measurement_point)
        button_layout.addWidget(add_measurement_button)

        delete_measurement_button = QPushButton("Delete Point")
        delete_measurement_button.setFixedWidth(100)
        delete_measurement_button.clicked.connect(self.delete_measurement_point)
        button_layout.addWidget(delete_measurement_button)
        
        button_layout.addStretch()
        input_layout.addLayout(button_layout)

        # Vertikaler Spacer
        input_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Füge das input_widget zum Splitter hinzu
        splitter.addWidget(input_widget)

        # Füge das plot_widget zum Splitter hinzu
        splitter.addWidget(self.plot_widget)

        # Setze die initiale Größenverteilung
        total_height = 700  # Erhöhte Höhe
        splitter.setSizes([int(total_height * 0.25), int(total_height * 0.8)])  # Mehr Platz für Plots

        # Fügen Sie den Splitter zum Hauptlayout hinzu
        self.dock_layout.addWidget(splitter)

        # Setze das Widget und konfiguriere die Fensterattribute
        self.setWidget(dock_widget_content)
        self.resize(680, total_height)  # Erhöhte Breite und Höhe
        self.setFloating(True)
        
        # Aktiviere Kontextmenü
        self.impulse_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.impulse_tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        # Verbinde Item-Änderungen für Umbenennung
        self.impulse_tree_widget.itemChanged.connect(self.on_impulse_item_changed)
        
        # Übergebe das TreeWidget an das Plot-Widget
        self.plot_widget.tree_widget = self.impulse_tree_widget
        
        # Verbinde TreeWidget-Auswahl mit Plot-Update
        self.impulse_tree_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Initialisiere bestehende Messpunkte und erstelle Canvas
        self.initialize_measurement_points()

    def on_selection_changed(self):
        """Wird aufgerufen, wenn sich die Auswahl im TreeWidget ändert"""
        self.plot_widget.update_plot_impulse()
        
        # Berechne und plotte die Daten für alle existierenden Messpunkte
        self.schedule_calculation()
        
        # Zeige das Widget
        self.show()

    def initialize_measurement_points(self):
        """Initialisiert die Messpunkte aus den bestehenden Einstellungen in Erstellungsreihenfolge"""
        # Entferne alle bestehenden Messpunkte aus dem TreeWidget
        self.impulse_tree_widget.clear()
        
        # Sortiere Impulse Points nach Erstellungszeit (falls vorhanden)
        sorted_points = sorted(
            self.settings.impulse_points,
            key=lambda p: p.get('created_at', 0)
        )
        
        # Füge die Messpunkte in sortierter Reihenfolge hinzu
        for point in sorted_points:
            # Füge Messpunkt hinzu mit gespeichertem Namen
            saved_name = point.get('name', point.get('text', point['key']))
            self.add_measurement_point(initial=True, point=point['data'], key=point['key'], name=saved_name)
        
        # Wähle den ersten Messpunkt aus, falls vorhanden
        if self.settings.impulse_points:
            first_item = self.impulse_tree_widget.topLevelItem(0)
            if first_item:
                self.impulse_tree_widget.setCurrentItem(first_item)

    def add_measurement_point(self, initial=True, point=None, key=None, name=None):
        if point is None:
            point = [0, 30]  # Standardwerte für neuen Messpunkt
    
        # Generiere einen eindeutigen Schlüssel (technisch, nicht sichtbar)
        if key is None:
            existing_keys = {point['key'] for point in self.settings.impulse_points}
            key = f"measurement_{len(self.settings.impulse_points)}"
            counter = 1
            while key in existing_keys:
                key = f"measurement_{len(self.settings.impulse_points) + counter}"
                counter += 1
        
        # Beim Laden (initial=True): Verwende übergebenen Namen
        # Beim Erstellen (initial=False): Generiere neuen Namen wenn nicht vorhanden
        if name is None:
            if initial:
                # Beim Laden ohne Namen: Fallback auf key
                name = key
            else:
                # Beim neuen Erstellen: Finde nächste freie Point-Nummer
                index = 1
                existing_names = {p.get('name', p.get('text', '')) for p in self.settings.impulse_points}
                while f"Point {index}" in existing_names:
                    index += 1
                name = f"Point {index}"
    
        # Erstelle neues TreeWidget Item
        item = QTreeWidgetItem(self.impulse_tree_widget)
        item.setFlags(item.flags() | Qt.ItemIsEditable)  # Macht Item editierbar
        item.setText(0, name)  # Zeige den Namen an
        item.setData(0, Qt.UserRole, key)  # Speichere den technischen Key

        # X-Koordinate LineEdit
        x_edit = QLineEdit()
        x_edit.setValidator(QDoubleValidator(-100, 100, 2))
        x_edit.setFixedWidth(90)
        x_edit.setFixedHeight(25)
        x_edit.setText(f"{point[0]:.2f}")
        x_edit.editingFinished.connect(
            lambda x_edit=x_edit, key=key: self.update_impulse_point_x(x_edit, key))
        self.impulse_tree_widget.setItemWidget(item, 1, x_edit)

        # Y-Koordinate LineEdit
        y_edit = QLineEdit()
        y_edit.setValidator(QDoubleValidator(-100, 100, 2))
        y_edit.setFixedWidth(98)
        y_edit.setFixedHeight(25)
        y_edit.setText(f"{point[1]:.2f}")
        y_edit.editingFinished.connect(
            lambda y_edit=y_edit, key=key: self.update_impulse_point_y(y_edit, key))
        self.impulse_tree_widget.setItemWidget(item, 2, y_edit)

        # Time Offset LineEdit
        time_offset_edit = QLineEdit()
        time_offset_edit.setValidator(QDoubleValidator(-1000, 1000, 2))
        time_offset_edit.setFixedWidth(98)
        time_offset_edit.setFixedHeight(25)
        
        # Beim Laden: Verwende gespeicherten time_offset
        if initial:
            time_offset_value = next(
                (p.get('time_offset', 0.0) for p in self.settings.impulse_points if p['key'] == key),
                0.0
            )
            time_offset_edit.setText(f"{time_offset_value:.2f}")
        else:
            time_offset_edit.setText("0.00")
        
        time_offset_edit.editingFinished.connect(
            lambda edit=time_offset_edit, key=key: self.update_time_offset(edit, key))
        self.impulse_tree_widget.setItemWidget(item, 3, time_offset_edit)

        # Find Button
        find_button = QPushButton("Find")
        find_button.setFixedWidth(98)
        find_button.setFixedHeight(25)
        find_button.clicked.connect(
            lambda checked, key=key: self.find_nearest_source_delay(key))
        self.impulse_tree_widget.setItemWidget(item, 4, find_button)

        if not initial:
            # Neuen Messpunkt zu den Einstellungen hinzufügen
            import time
            self.settings.impulse_points.append({
                'key': key, 
                'data': point, 
                'name': name,  # Benutzerfreundlicher Name
                'text': key,   # Behalte text für Kompatibilität
                'time_offset': 0.0,
                'created_at': time.time()  # Zeitstempel für Sortierung
            })
            self.container.set_measurement_point(key, point)
            self.schedule_calculation(update_speaker_arrays=True)
            
            # Wähle das neu erstellte Item aus
            self.impulse_tree_widget.setCurrentItem(item)

    def on_impulse_item_changed(self, item, column):
        """
        Wird aufgerufen wenn ein Impulse Point umbenannt wird.
        """
        if column != 0:  # Nur bei Spalte 0 (Name)
            return
        
        key = item.data(0, Qt.UserRole)
        new_name = item.text(0).strip()  # Entferne Leerzeichen
        
        if not new_name:
            # Leerer Name nicht erlaubt
            for p in self.settings.impulse_points:
                if p['key'] == key:
                    old_name = p.get('name', p.get('text', key))
                    item.setText(0, old_name)
                    break
            return
        
        # Finde den Impulse Point und aktualisiere den Namen
        for p in self.settings.impulse_points:
            if p['key'] == key:
                old_name = p.get('name', p.get('text', key))
                if old_name != new_name:
                    p['name'] = new_name
                    print(f"Impulse Point umbenannt: '{old_name}' → '{new_name}'")
                break

    def update_impulse_point_x(self, edit, key):
        try:
            value = round(float(edit.text()) if edit.text() else 0, 2)
            index = next((i for i, point in enumerate(self.settings.impulse_points) 
                         if point['key'] == key), None)
            if index is not None:
                self.settings.impulse_points[index]['data'][0] = value
                self.container.set_measurement_point(key, self.settings.impulse_points[index]['data'])
                self.schedule_calculation(update_speaker_arrays=True)
            edit.setText(f"{value:.2f}")
        except ValueError:
            if index is not None:
                edit.setText(f"{self.settings.impulse_points[index]['data'][0]:.2f}")

    def update_impulse_point_y(self, edit, key):
        try:
            value = round(float(edit.text()) if edit.text() else 0, 2)
            index = next((i for i, point in enumerate(self.settings.impulse_points) 
                         if point['key'] == key), None)
            if index is not None:
                self.settings.impulse_points[index]['data'][1] = value
                self.container.set_measurement_point(key, self.settings.impulse_points[index]['data'])
                self.schedule_calculation(update_speaker_arrays=True)
            edit.setText(f"{value:.2f}")
        except ValueError:
            if index is not None:
                edit.setText(f"{self.settings.impulse_points[index]['data'][1]:.2f}")

    def delete_measurement_point(self, item=None):
        """
        Löscht einen Messpunkt aus dem TreeWidget und den Einstellungen.
        
        Args:
            item: QTreeWidgetItem oder None. Wenn None, wird das aktuell ausgewählte Item gelöscht
        """
        # Wenn item None oder bool ist, nehme das aktuell ausgewählte Item
        if item is None or isinstance(item, bool):
            item = self.impulse_tree_widget.currentItem()
            if not item:
                return  # Wenn kein Item ausgewählt ist, beende die Funktion
        
        key = item.data(0, Qt.UserRole)  # Hole den gespeicherten key
        
        # Lösche aus den Einstellungen
        index = next((i for i, point in enumerate(self.settings.impulse_points) 
                      if point['key'] == key), None)
        if index is not None:
            del self.settings.impulse_points[index]
        
        # Lösche aus dem Container
        if hasattr(self.container, 'impulse_points'):
            self.container.impulse_points.pop(key, None)
        if key in getattr(self.container, 'calculation_impulse', {}):
            self.container.delete_calculation_impulse(key)
        # Canvas werden nicht gelöscht, da sie fest sind
        
        # Lösche aus dem TreeWidget
        root = self.impulse_tree_widget.invisibleRootItem()
        root.removeChild(item)
        
        # Aktualisiere die Berechnung
        self.schedule_calculation()


    def show_context_menu(self, position):
        """Zeigt das Kontextmenü an der Mausposition"""
        item = self.impulse_tree_widget.itemAt(position)
        if item:
            context_menu = QMenu()
            
            # Menüeinträge erstellen
            rename_action = context_menu.addAction("Rename")
            context_menu.addSeparator()
            delete_action = context_menu.addAction("Delete")
            
            # Verbinde Aktionen mit Funktionen
            rename_action.triggered.connect(lambda: self.impulse_tree_widget.editItem(item))
            delete_action.triggered.connect(lambda: self.delete_measurement_point(item))
            
            # Zeige Menü an Mausposition
            context_menu.exec_(self.impulse_tree_widget.viewport().mapToGlobal(position))


    def update_time_offset(self, edit, key):
        """Aktualisiert den Time Offset und löst Neuberechnung aus"""
        try:
            value = round(float(edit.text()) if edit.text() else 0, 2)
            index = next((i for i, point in enumerate(self.settings.impulse_points) 
                         if point['key'] == key), None)
            if index is not None:
                # Update settings
                self.settings.impulse_points[index]['time_offset'] = value
                # Neuberechnung der Impulse
                self.schedule_calculation()
            edit.setText(f"{value:.2f}")
        except ValueError:
            # Bei ungültiger Eingabe zurück zum alten Wert
            if index is not None:
                edit.setText(f"{self.settings.impulse_points[index]['time_offset']:.2f}")

    def find_nearest_source_delay(self, key):
        """Findet und setzt die Verzögerung zur nächsten Quelle"""
        point = next((p for p in self.settings.impulse_points if p['key'] == key), None)
        if point is None:
            return
        
        # Hole die Koordinaten des Messpunkts
        point_x = point['data'][0]
        point_y = point['data'][1]
        
        # Finde die kürzeste Distanz zu einer aktiven Quelle
        min_distance = float('inf')
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            for i in range(len(speaker_array.source_position_x)):
                x_distance = point_x - speaker_array.source_position_x[i]
                y_distance = point_y - speaker_array.source_position_y[i]
                distance = np.sqrt(x_distance**2 + y_distance**2)
                
                if distance < min_distance:
                    min_distance = distance
        
        # Umrechnung in Millisekunden
        delay = (min_distance / self.settings.speed_of_sound) * 1000
        delay = round(delay, 2)
        
        # Update UI und Daten
        index = next((i for i, p in enumerate(self.settings.impulse_points) 
                      if p['key'] == key), None)
        if index is not None:
            # Update settings
            self.settings.impulse_points[index]['time_offset'] = delay
            
            # Update UI
            root = self.impulse_tree_widget.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                if item.data(0, Qt.UserRole) == key:
                    time_offset_edit = self.impulse_tree_widget.itemWidget(item, 3)
                    if time_offset_edit:
                        time_offset_edit.setText(f"{delay:.2f}")
                    break
            
            # Neuberechnung der Impulse
            self.schedule_calculation()


    def on_view_changed(self, view_type):
        """Handler für Änderungen der Plot-Ansicht"""
        self.schedule_calculation()




