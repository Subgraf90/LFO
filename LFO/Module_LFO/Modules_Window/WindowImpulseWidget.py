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

    def _refresh_measurement_overlays(self):
        """Aktualisiert die SPL-Overlays für Messpunkte ohne komplette Neu­berechnung."""
        draw_plots = getattr(self.main_window, 'draw_plots', None)
        if draw_plots is None:
            return
        try:
            # 🚀 FIX: Aktualisiere 3D-Overlays für Impulse Points
            # Stelle sicher, dass der Plotter die Overlays aktualisiert
            if hasattr(draw_plots, 'draw_spl_plotter') and draw_plots.draw_spl_plotter is not None:
                plotter = draw_plots.draw_spl_plotter
                if hasattr(plotter, 'update_overlays'):
                    # 🚀 FIX: Setze Signatur zurück, damit 'impulse' Kategorie erkannt wird
                    # Die Signatur wird in update_overlays neu berechnet und erkennt die Änderung
                    if hasattr(plotter, '_last_overlay_signatures'):
                        # Entferne 'impulse' aus der letzten Signatur, damit Änderung erkannt wird
                        if plotter._last_overlay_signatures and 'impulse' in plotter._last_overlay_signatures:
                            # Setze auf None, damit die Signatur als geändert erkannt wird
                            plotter._last_overlay_signatures['impulse'] = None
                    # Aktualisiere Overlays - die Signatur-Erkennung wird die Änderung erkennen
                    plotter.update_overlays(self.settings, self.container)
            else:
                # Fallback: Verwende plot_impulse_points()
                draw_plots.plot_impulse_points()
        except Exception as exc:
            print(f"[ImpulseInputDockWidget] Fehler beim Aktualisieren der Messpunkt-Overlays: {exc}")
            import traceback
            traceback.print_exc()

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
            if getattr(self.settings, "update_pressure_impulse", True):
                self.main_window.calculate_impulse()
            else:
                if hasattr(self.main_window, "impulse_manager"):
                    self.main_window.impulse_manager.show_empty_plot()

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
        self.impulse_tree_widget.setHeaderLabels(["#", "Measurement Point", "Position X (m)", "Position Y (m)", "Position Z (m)", "Offset (ms)", ""])
        self.impulse_tree_widget.setFixedHeight(120)
        
        # Breitere Spalten
        self.impulse_tree_widget.setColumnWidth(0, 45)   # Nummer-Spalte
        self.impulse_tree_widget.setColumnWidth(1, 130)  # Measurement Point
        self.impulse_tree_widget.setColumnWidth(2, 85)   # Position X
        self.impulse_tree_widget.setColumnWidth(3, 85)   # Position Y
        self.impulse_tree_widget.setColumnWidth(4, 85)   # Position Z
        self.impulse_tree_widget.setColumnWidth(5, 85)   # Offset
        self.impulse_tree_widget.setColumnWidth(6, 50)   # Button
        
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
        self.resize(800, total_height)  # Erhöhte Breite und Höhe
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
        for idx, point in enumerate(sorted_points, start=1):
            # Setze oder aktualisiere die Nummer
            point['number'] = idx
            
            # Füge Messpunkt hinzu mit gespeichertem Namen
            saved_name = point.get('name', point.get('text', point['key']))
            point_data = point['data']
            # Stelle sicher, dass point_data mindestens 3 Elemente hat (für Rückwärtskompatibilität)
            if len(point_data) < 3:
                point_data = list(point_data) + [0.0] * (3 - len(point_data))
            self.add_measurement_point(initial=True, point=point_data, key=point['key'], name=saved_name, number=idx)
        
        # Wähle den ersten Messpunkt aus, falls vorhanden
        if self.settings.impulse_points:
            first_item = self.impulse_tree_widget.topLevelItem(0)
            if first_item:
                self.impulse_tree_widget.setCurrentItem(first_item)

    def add_measurement_point(self, initial=True, point=None, key=None, name=None, number=None):
        if point is None:
            point = [0, 30, 0]  # Standardwerte für neuen Messpunkt (X, Y, Z)
        # Stelle sicher, dass point mindestens 3 Elemente hat (für Rückwärtskompatibilität)
        if len(point) < 3:
            point = list(point) + [0.0] * (3 - len(point))
    
        # Generiere einen eindeutigen Schlüssel (technisch, nicht sichtbar)
        if key is None:
            existing_keys = {point['key'] for point in self.settings.impulse_points}
            key = f"measurement_{len(self.settings.impulse_points)}"
            counter = 1
            while key in existing_keys:
                key = f"measurement_{len(self.settings.impulse_points) + counter}"
                counter += 1
        
        # Bestimme die Nummer für den neuen Messpunkt
        if number is None:
            # Beim neuen Erstellen: Nächste freie Nummer
            number = len(self.settings.impulse_points) + 1
        
        # Beim Laden (initial=True): Verwende übergebenen Namen
        # Beim Erstellen (initial=False): Generiere neuen Namen wenn nicht vorhanden
        if name is None:
            if initial:
                # Beim Laden ohne Namen: Fallback auf key
                name = key
            else:
                # Beim neuen Erstellen: Verwende fortlaufende Nummer
                name = f"Point {number}"
    
        # Erstelle neues TreeWidget Item
        item = QTreeWidgetItem(self.impulse_tree_widget)
        item.setText(0, str(number))  # Zeige die Nummer an
        item.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)  # Linksbündig ausrichten
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Spalte 0 nicht editierbar
        item.setText(1, name)  # Zeige den Namen an
        item.setFlags(item.flags() | Qt.ItemIsEditable)  # Spalte 1 editierbar
        item.setData(1, Qt.UserRole, key)  # Speichere den technischen Key

        # X-Koordinate LineEdit
        x_edit = QLineEdit()
        x_edit.setValidator(QDoubleValidator(-100, 100, 2))
        x_edit.setFixedWidth(75)
        x_edit.setFixedHeight(25)
        x_edit.setText(f"{point[0]:.2f}")
        x_edit.editingFinished.connect(
            lambda x_edit=x_edit, key=key: self.update_impulse_point_x(x_edit, key))
        self.impulse_tree_widget.setItemWidget(item, 2, x_edit)

        # Y-Koordinate LineEdit
        y_edit = QLineEdit()
        y_edit.setValidator(QDoubleValidator(-100, 100, 2))
        y_edit.setFixedWidth(75)
        y_edit.setFixedHeight(25)
        y_edit.setText(f"{point[1]:.2f}")
        y_edit.editingFinished.connect(
            lambda y_edit=y_edit, key=key: self.update_impulse_point_y(y_edit, key))
        self.impulse_tree_widget.setItemWidget(item, 3, y_edit)

        # Z-Koordinate LineEdit
        z_edit = QLineEdit()
        z_edit.setValidator(QDoubleValidator(-100, 100, 2))
        z_edit.setFixedWidth(75)
        z_edit.setFixedHeight(25)
        z_edit.setText(f"{point[2]:.2f}")
        z_edit.editingFinished.connect(
            lambda z_edit=z_edit, key=key: self.update_impulse_point_z(z_edit, key))
        self.impulse_tree_widget.setItemWidget(item, 4, z_edit)

        # Time Offset LineEdit
        time_offset_edit = QLineEdit()
        time_offset_edit.setValidator(QDoubleValidator(-1000, 1000, 2))
        time_offset_edit.setFixedWidth(75)
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
        self.impulse_tree_widget.setItemWidget(item, 5, time_offset_edit)

        # Find Button
        find_button = QPushButton("Find")
        find_button.setFixedWidth(98)
        find_button.setFixedHeight(25)
        find_button.clicked.connect(
            lambda checked, key=key: self.find_nearest_source_delay(key))
        self.impulse_tree_widget.setItemWidget(item, 6, find_button)

        if not initial:
            # Neuen Messpunkt zu den Einstellungen hinzufügen
            import time
            import json
            self.settings.impulse_points.append({
                'key': key, 
                'data': point, 
                'name': name,  # Benutzerfreundlicher Name
                'text': key,   # Behalte text für Kompatibilität
                'time_offset': 0.0,
                'number': number,  # Nummer des Messpunkts
                'created_at': time.time()  # Zeitstempel für Sortierung
            })
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"WindowImpulseWidget.py:277","message":"impulse point added","data":{"key":key,"point":point,"total_count":len(self.settings.impulse_points)},"timestamp":time.time()*1000}) + '\n')
            except: pass
            # #endregion
            self.container.set_measurement_point(key, point)
            
            # 🚀 FIX: Sofortige Berechnung und Plot-Update nach Hinzufügen eines Points
            if hasattr(self.main_window, 'impulse_manager'):
                # Erzwinge sofortige Berechnung (force=True) damit der neue Point berücksichtigt wird
                self.main_window.impulse_manager.update_calculation_impulse(force=True)
            
            # Aktualisiere 3D-Overlays
            self._refresh_measurement_overlays()
            
            # Wähle das neu erstellte Item aus
            self.impulse_tree_widget.setCurrentItem(item)

    def on_impulse_item_changed(self, item, column):
        """
        Wird aufgerufen wenn ein Impulse Point umbenannt wird.
        """
        if column != 1:  # Nur bei Spalte 1 (Name)
            return
        
        key = item.data(1, Qt.UserRole)
        new_name = item.text(1).strip()  # Entferne Leerzeichen
        
        if not new_name:
            # Leerer Name nicht erlaubt
            for p in self.settings.impulse_points:
                if p['key'] == key:
                    old_name = p.get('name', p.get('text', key))
                    item.setText(1, old_name)
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
                self.schedule_calculation()
                self._refresh_measurement_overlays()
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
                # Stelle sicher, dass data mindestens 3 Elemente hat
                if len(self.settings.impulse_points[index]['data']) < 3:
                    self.settings.impulse_points[index]['data'] = list(self.settings.impulse_points[index]['data']) + [0.0] * (3 - len(self.settings.impulse_points[index]['data']))
                self.settings.impulse_points[index]['data'][1] = value
                self.container.set_measurement_point(key, self.settings.impulse_points[index]['data'])
                self.schedule_calculation()
                self._refresh_measurement_overlays()
            edit.setText(f"{value:.2f}")
        except ValueError:
            if index is not None:
                edit.setText(f"{self.settings.impulse_points[index]['data'][1]:.2f}")

    def update_impulse_point_z(self, edit, key):
        try:
            value = round(float(edit.text()) if edit.text() else 0, 2)
            index = next((i for i, point in enumerate(self.settings.impulse_points) 
                         if point['key'] == key), None)
            if index is not None:
                # Stelle sicher, dass data mindestens 3 Elemente hat
                if len(self.settings.impulse_points[index]['data']) < 3:
                    self.settings.impulse_points[index]['data'] = list(self.settings.impulse_points[index]['data']) + [0.0] * (3 - len(self.settings.impulse_points[index]['data']))
                self.settings.impulse_points[index]['data'][2] = value
                self.container.set_measurement_point(key, self.settings.impulse_points[index]['data'])
                self.schedule_calculation()
                self._refresh_measurement_overlays()
            edit.setText(f"{value:.2f}")
        except ValueError:
            if index is not None:
                # Stelle sicher, dass data mindestens 3 Elemente hat
                if len(self.settings.impulse_points[index]['data']) < 3:
                    self.settings.impulse_points[index]['data'] = list(self.settings.impulse_points[index]['data']) + [0.0] * (3 - len(self.settings.impulse_points[index]['data']))
                edit.setText(f"{self.settings.impulse_points[index]['data'][2]:.2f}")

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
        
        key = item.data(1, Qt.UserRole)  # Hole den gespeicherten key (jetzt in Spalte 1)
        
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
        
        # Aktualisiere die Nummern nach dem Löschen
        self._renumber_measurement_points()
        
        # Aktualisiere die Berechnung
        self.schedule_calculation()

        # 🚀 FIX: Aktualisiere 3D-Overlays, damit der Messpunkt sofort verschwindet
        draw_plots = getattr(self.main_window, 'draw_plots', None)
        if draw_plots is not None and hasattr(draw_plots, 'draw_spl_plotter'):
            try:
                plotter = draw_plots.draw_spl_plotter
                if plotter is not None and hasattr(plotter, 'update_overlays'):
                    # 🚀 FIX: Setze Signatur zurück, damit 'impulse' Kategorie erkannt wird
                    if hasattr(plotter, '_last_overlay_signatures'):
                        # Entferne 'impulse' aus der letzten Signatur, damit Änderung erkannt wird
                        if plotter._last_overlay_signatures and 'impulse' in plotter._last_overlay_signatures:
                            # Setze auf None, damit die Signatur als geändert erkannt wird
                            plotter._last_overlay_signatures['impulse'] = None
                    # Aktualisiere Overlays - die Signatur-Erkennung wird die Änderung erkennen
                    plotter.update_overlays(self.settings, self.container)
            except Exception as exc:
                print(f"[ImpulseInputDockWidget] Fehler beim Aktualisieren der SPL-Overlays nach Löschen eines Messpunkts: {exc}")
                import traceback
                traceback.print_exc()


    def _renumber_measurement_points(self):
        """Aktualisiert die Nummern aller Messpunkte nach dem Löschen"""
        # Aktualisiere die Nummern in den Settings
        for idx, point in enumerate(self.settings.impulse_points, start=1):
            point['number'] = idx
        
        # Aktualisiere die Nummern im TreeWidget
        root = self.impulse_tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            item.setText(0, str(i + 1))

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
            rename_action.triggered.connect(lambda: self.impulse_tree_widget.editItem(item, 1))  # Bearbeite Spalte 1 (Name)
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
        point_data = point['data']
        point_x = point_data[0]
        point_y = point_data[1]
        point_z = point_data[2] if len(point_data) > 2 else 0.0
        
        # Finde die kürzeste Distanz zu einer aktiven Quelle
        min_distance = float('inf')
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            for i in range(len(speaker_array.source_position_x)):
                x_distance = point_x - speaker_array.source_position_x[i]
                y_distance = point_y - speaker_array.source_position_y[i]
                source_position_z = getattr(speaker_array, 'source_position_calc_z', None)
                source_z = source_position_z[i] if source_position_z is not None else 0.0
                z_distance = point_z - source_z
                
                # 3D-Distanz berechnen
                horizontal_dist = np.sqrt(x_distance**2 + y_distance**2)
                distance = np.sqrt(horizontal_dist**2 + z_distance**2)
                
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
                if item.data(1, Qt.UserRole) == key:
                    time_offset_edit = self.impulse_tree_widget.itemWidget(item, 5)
                    if time_offset_edit:
                        time_offset_edit.setText(f"{delay:.2f}")
                    break
            
            # Neuberechnung der Impulse
            self.schedule_calculation()


    def on_view_changed(self, view_type):
        """Handler für Änderungen der Plot-Ansicht"""
        self.schedule_calculation()




