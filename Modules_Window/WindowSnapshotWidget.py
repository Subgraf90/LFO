from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QCheckBox, QMenu
from PyQt5.QtCore import Qt
import random
import copy

class SnapshotWidget:
    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.capture_dock_widget = None
        self.snapshot_tree_widget = None  # TreeWidget für Snapshots
        self.selected_snapshot_key = "aktuelle_simulation"  # Standardmäßig current speakers ausgewählt
        self._current_spl_backup = None
        self._current_polar_backup = None
        self._current_axes_backup = None
        self._current_impulse_backup = None

    def show_snapshot_widget(self):
        if self.capture_dock_widget:
            self.capture_dock_widget.close()
            self.capture_dock_widget.deleteLater()
        
        self.capture_dock_widget = QtWidgets.QDockWidget("Snapshot", self.main_window)
        self.main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.capture_dock_widget)
        
        dock_widget_contents = QtWidgets.QWidget()
        self.capture_dock_widget.setWidget(dock_widget_contents)
        
        dock_layout = QVBoxLayout()
        dock_widget_contents.setLayout(dock_layout)
        
        # Capture Button
        capture_button = QtWidgets.QPushButton("Capture")
        capture_button.setFixedWidth(100)
        capture_button.clicked.connect(self.on_capture_button_clicked)
        dock_layout.addWidget(capture_button, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        
        # TreeWidget für Snapshots
        self.snapshot_tree_widget = QTreeWidget()
        self.snapshot_tree_widget.setHeaderLabels(["Snapshot", "Show", ""])
        
        # Entferne Einrückung damit Text ganz links ist
        self.snapshot_tree_widget.setRootIsDecorated(False)
        self.snapshot_tree_widget.setIndentation(0)
        
        # Spaltenbreiten konfigurieren
        self.snapshot_tree_widget.setColumnWidth(0, 150)  # Name-Spalte etwas kleiner
        self.snapshot_tree_widget.setColumnWidth(1, 40)   # Show-Checkbox schmaler
        self.snapshot_tree_widget.setColumnWidth(2, 20)   # Farb-Quadrat (nur so breit wie nötig)
        
        # Erste Spalte automatisch vergrößern beim Fenster-Resize
        header = self.snapshot_tree_widget.header()
        header.setStretchLastSection(False)  # Letzte Spalte nicht strecken
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)  # Name-Spalte streckt sich
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)    # Show bleibt fix
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)    # Color bleibt fix
        
        # Kontextmenü für Rechtsklick aktivieren
        self.snapshot_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.snapshot_tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        # Item-Änderungen erkennen (für Umbenennungen)
        self.snapshot_tree_widget.itemChanged.connect(self.on_snapshot_item_changed)
        
        # Item-Klick erkennen (für Plot-Anzeige)
        self.snapshot_tree_widget.itemClicked.connect(self.on_snapshot_item_clicked)
        
        dock_layout.addWidget(self.snapshot_tree_widget)
        
        # Setze die initiale Breite des DockWidgets
        self.main_window.resizeDocks([self.capture_dock_widget], [220], QtCore.Qt.Horizontal)

        if hasattr(self.main_window, "surface_manager"):
            try:
                self.main_window.surface_manager.on_snapshot_dock_available(self.capture_dock_widget)
            except Exception:
                pass
        
        # Widgets aktualisieren
        self.update_snapshot_widgets()

    def update_snapshot_widgets(self):
        """Aktualisiert das TreeWidget mit allen Snapshots in Erstellungsreihenfolge"""
        if not self.snapshot_tree_widget:
            return
        
        # Blockiere Signale während der Aktualisierung
        self.snapshot_tree_widget.blockSignals(True)
        
        # Leere das TreeWidget
        self.snapshot_tree_widget.clear()
        
        # Füge "current speakers" Item ganz oben ein
        current_item = QTreeWidgetItem(self.snapshot_tree_widget, ["current speakers", "", ""])
        current_item.setFlags(current_item.flags() & ~Qt.ItemIsEditable)  # NICHT editierbar
        current_item.setData(0, Qt.UserRole, "aktuelle_simulation")  # Verknüpfe mit aktuelle_simulation
        
        # Checkbox für "current speakers" (per default aktiviert)
        current_checkbox = QCheckBox()
        if "aktuelle_simulation" in self.container.calculation_axes:
            current_checkbox.setChecked(bool(self.container.calculation_axes["aktuelle_simulation"].get("show_in_plot", True)))
        else:
            current_checkbox.setChecked(True)
        current_checkbox.stateChanged.connect(lambda state: self.update_plots("aktuelle_simulation", state))
        self.snapshot_tree_widget.setItemWidget(current_item, 1, current_checkbox)
        
        # Farb-Quadrat für "current speakers" (Standardfarbe blau wie in PlotSPLXaxis.py)
        current_color_label = QtWidgets.QLabel()
        current_color_label.setFixedSize(20, 20)
        current_color_label.setStyleSheet("background-color: blue; border: 1px solid gray;")
        self.snapshot_tree_widget.setItemWidget(current_item, 2, current_color_label)
        
        # Sortiere Snapshots nach Erstellungszeit (falls vorhanden)
        snapshots = []
        for key, value in self.container.calculation_axes.items():
            if key != "aktuelle_simulation":
                # Verwende Zeitstempel falls vorhanden, sonst 0
                created_at = value.get("created_at", 0)
                snapshots.append((created_at, key, value))
        
        # Sortiere nach Zeitstempel (älteste zuerst)
        snapshots.sort(key=lambda x: x[0])
        
        # Füge alle Snapshots in sortierter Reihenfolge hinzu
        for created_at, key, value in snapshots:
                # Erstelle TreeWidget Item
                item = QTreeWidgetItem(self.snapshot_tree_widget, [key, "", ""])
                item.setFlags(item.flags() | Qt.ItemIsEditable)  # Macht Item editierbar
                item.setData(0, Qt.UserRole, key)  # Speichere original key
                
                # Erstelle Checkbox für "Show in Plot"
                checkbox = QCheckBox()
                checkbox.setChecked(bool(value.get("show_in_plot", False)))
                checkbox.stateChanged.connect(lambda state, k=key: self.update_plots(k, state))
                self.snapshot_tree_widget.setItemWidget(item, 1, checkbox)
                
                # Erstelle Farb-Quadrat
                color_label = QtWidgets.QLabel()
                color_label.setFixedSize(20, 20)
                snapshot_color = value.get("color", "#CCCCCC")  # Fallback grau
                color_label.setStyleSheet(f"background-color: {snapshot_color}; border: 1px solid gray;")
                self.snapshot_tree_widget.setItemWidget(item, 2, color_label)
        
        # Signale wieder freigeben
        self.snapshot_tree_widget.blockSignals(False)
        
        # Wähle das aktuelle Item aus (standardmäßig "current speakers")
        self._select_item_by_key(self.selected_snapshot_key)

    def on_capture_button_clicked(self): 
        """Erstellt einen neuen Snapshot der aktuellen Simulation"""
        # Prüfe ob aktuelle_simulation existiert
        if "aktuelle_simulation" not in self.container.calculation_axes:
            print("Warnung: Keine aktuelle Simulation zum Erfassen vorhanden!")
            return
        
        # ========================================
        # WICHTIG: Wechsle zu "current speakers" (lädt automatisch aktuelle Daten)
        # ========================================
        
        # Finde das "current speakers" Item im TreeWidget
        root = self.snapshot_tree_widget.invisibleRootItem()
        current_speakers_item = None
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == "aktuelle_simulation":
                current_speakers_item = item
                break
        
        # Wechsle zu "current speakers" (triggert automatisch Neuberechnung)
        if current_speakers_item:
            self.snapshot_tree_widget.setCurrentItem(current_speakers_item)
            # Rufe die Click-Handler Methode auf (führt Neuberechnung durch)
            self.on_snapshot_item_clicked(current_speakers_item, 0)
        
        # Stelle sicher, dass die aktuelle Simulation sichtbar bleibt
        aktuelle_axes = self.container.calculation_axes.get("aktuelle_simulation")
        if isinstance(aktuelle_axes, dict):
            aktuelle_axes["show_in_plot"] = True
        if hasattr(self.container, "calculation_spl"):
            self.container.calculation_spl["show_in_plot"] = True
        
        # Jetzt enthalten calculation_spl und calculation_polar die aktuellen Lautsprecherdaten!
        
        # Kopiere die aktuelle Simulation
        capture_data = copy.deepcopy(self.container.calculation_axes["aktuelle_simulation"])
        
        # Finde einen eindeutigen Namen (Snapshot 1, Snapshot 2, etc.)
        index = 1
        while f"Snapshot {index}" in self.container.calculation_axes:
            index += 1
        new_key = f"Snapshot {index}"
        
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        capture_data["color"] = color
        capture_data["show_in_plot"] = True  # Standardmäßig in Plots anzeigen
        
        # Füge Zeitstempel für Sortierung hinzu
        import time
        capture_data["created_at"] = time.time()
        
        # ========================================
        # 1. Impulse-Daten speichern
        # ========================================
        calculation_impulse = self.container.calculation_impulse.get("aktuelle_simulation", {})
        
        for point_key in self.settings.impulse_points:
            key = point_key['key']
            
            if key in calculation_impulse:
                calc_data = calculation_impulse[key]
                
                # Magnitude Response
                if calc_data.get('magnitude_data'):
                    capture_data[f"magnitude_{key}"] = {
                        'frequency': calc_data['magnitude_data']['frequency'],
                        'magnitude': calc_data['magnitude_data']['magnitude']
                    }
                
                # Phase Response
                if calc_data.get('phase_response', {}).get('combined_phase'):
                    capture_data[f"phase_{key}"] = {
                        'freq': calc_data['phase_response']['combined_phase']['freq'],
                        'phase': calc_data['phase_response']['combined_phase']['phase']
                    }
                
                # Impulse Response
                if calc_data.get('impulse_response', {}).get('combined_impulse'):
                    capture_data[f"impulse_{key}"] = {
                        'time': calc_data['impulse_response']['combined_impulse']['time'],
                        'spl': calc_data['impulse_response']['combined_impulse']['spl']
                    }
                
                # Arrival Times
                if calc_data.get('arrival_times'):
                    capture_data[f"arrival_{key}"] = {
                        'time': calc_data['arrival_times']['time'],
                        'spl_max': calc_data['arrival_times']['spl_max'],
                        'spl_min': calc_data['arrival_times']['spl_min']
                    }
        
        # ========================================
        # 2. Polarplot-Daten speichern
        # ========================================
        if hasattr(self.container, 'calculation_polar') and self.container.calculation_polar:
            # Kopiere die komplette Polarplot-Struktur
            capture_data['polar_data'] = {
                'sound_field_p': self.container.calculation_polar.get('sound_field_p', {}),
                'angles': self.container.calculation_polar.get('angles', None),
                'frequencies': self.container.calculation_polar.get('frequencies', {})
            }
        
        # ========================================
        # 3. 2D-Schallfeld-Daten (SPL Plot) speichern
        # ========================================
        if hasattr(self.container, 'calculation_spl') and self.container.calculation_spl:
            # Kopiere das vollständige 2D-Schallfeld
            import numpy as np
            
            spl_data = {}
            if 'sound_field_p' in self.container.calculation_spl:
                # Konvertiere zu Liste für JSON-Kompatibilität
                field_p = self.container.calculation_spl['sound_field_p']
                if isinstance(field_p, np.ndarray):
                    spl_data['sound_field_p'] = field_p.tolist()
                else:
                    spl_data['sound_field_p'] = field_p
                    
            if 'sound_field_x' in self.container.calculation_spl:
                field_x = self.container.calculation_spl['sound_field_x']
                if isinstance(field_x, np.ndarray):
                    spl_data['sound_field_x'] = field_x.tolist()
                else:
                    spl_data['sound_field_x'] = field_x
                    
            if 'sound_field_y' in self.container.calculation_spl:
                field_y = self.container.calculation_spl['sound_field_y']
                if isinstance(field_y, np.ndarray):
                    spl_data['sound_field_y'] = field_y.tolist()
                else:
                    spl_data['sound_field_y'] = field_y
            
            if spl_data:
                capture_data['spl_field_data'] = spl_data
        
        # Ausgabe zur Kontrolle der gespeicherten Daten
        print(f"Snapshot Capture Daten ({new_key}): {list(capture_data.keys())}")

        # Speichere den Snapshot
        self.container.calculation_axes[new_key] = capture_data
        
        self.update_snapshot_widgets()

    def on_snapshot_item_changed(self, item, column):
        """
        Wird aufgerufen wenn ein Snapshot umbenannt wird.
        """
        if column != 0:  # Nur bei Spalte 0 (Name)
            return
        
        old_key = item.data(0, Qt.UserRole)
        new_key = item.text(0).strip()  # Entferne Leerzeichen
        
        # Verbiete Umbenennung von "current speakers"
        if old_key == "aktuelle_simulation":
            item.setText(0, "current speakers")
            return
        
        # Prüfe ob der Name geändert wurde
        if old_key == new_key or not new_key:
            return
        
        # Verbiete "aktuelle_simulation" als Name
        if new_key == "aktuelle_simulation":
            print("Warnung: Der Name 'aktuelle_simulation' ist reserviert!")
            item.setText(0, old_key)
            return
        
        # Prüfe ob der neue Name bereits existiert
        if new_key in self.container.calculation_axes:
            print(f"Warnung: Snapshot mit Name '{new_key}' existiert bereits!")
            # Setze den alten Namen zurück
            item.setText(0, old_key)
            return
        
        # Hole die Snapshot-Daten
        snapshot_data = self.container.calculation_axes.get(old_key)
        if snapshot_data is None:
            return
        
        # Umbenennen: Lösche alten Key und erstelle neuen
        self.container.calculation_axes[new_key] = self.container.calculation_axes.pop(old_key)
        
        # Aktualisiere UserRole mit neuem Key
        item.setData(0, Qt.UserRole, new_key)
        
        # Aktualisiere die Checkbox-Verbindung
        checkbox = self.snapshot_tree_widget.itemWidget(item, 1)
        if checkbox:
            # Trenne alte Verbindung und erstelle neue
            try:
                checkbox.stateChanged.disconnect()
            except:
                pass
            checkbox.stateChanged.connect(lambda state, k=new_key: self.update_plots(k, state))
        
        print(f"Snapshot umbenannt: '{old_key}' → '{new_key}'")

    def update_plots(self, key, state):
        if key in self.container.calculation_axes:
            self.container.calculation_axes[key]["show_in_plot"] = bool(state)
            
            has_active = self.has_any_active_snapshot()
            if not has_active:
                if hasattr(self.main_window, 'draw_plots'):
                    self.main_window.draw_plots.show_empty_axes()
                    self.main_window.draw_plots.show_empty_polar()
                    self.main_window.draw_plots.show_empty_spl()
            
            if hasattr(self.main_window, 'plot_xaxis'):
                self.main_window.plot_xaxis()
            if hasattr(self.main_window, 'plot_yaxis'):
                self.main_window.plot_yaxis()
            if hasattr(self.main_window, 'plot_spl'):
                self.main_window.plot_spl()
            if hasattr(self.main_window, 'plot_polar_pattern'):
                self.main_window.plot_polar_pattern()
            if hasattr(self.main_window, 'update_plot_impulse'):
                self.main_window.update_plot_impulse()
            elif hasattr(self.main_window, 'impulse_manager'):
                self.main_window.impulse_manager.update_plot_impulse()
    
    def _select_item_by_key(self, snapshot_key):
        """Wählt ein Item im TreeWidget basierend auf dem Key aus"""
        root = self.snapshot_tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == snapshot_key:
                self.snapshot_tree_widget.setCurrentItem(item)
                return
    
    def on_snapshot_item_clicked(self, item, column):
        """
        Wird aufgerufen wenn ein Snapshot-Item angeklickt wird.
        Zeigt die Daten dieses Snapshots in SPL- und Polarplot an (wenn Checkbox aktiviert).
        """
        # Hole den Key des geklickten Items
        snapshot_key = item.data(0, Qt.UserRole)
        if not snapshot_key:
            return
        
        previous_key = getattr(self, "selected_snapshot_key", "aktuelle_simulation")
        # Speichere das ausgewählte Item
        self.selected_snapshot_key = snapshot_key
        
        # Aktualisiere Achsen-Plots (für Liniendicke-Aktualisierung)
        if hasattr(self.main_window, 'plot_xaxis'):
            self.main_window.plot_xaxis()
        if hasattr(self.main_window, 'plot_yaxis'):
            self.main_window.plot_yaxis()
        # Aktualisiere Impuls-Plots für Snapshot-Hervorhebung
        if hasattr(self.main_window, 'update_plot_impulse'):
            self.main_window.update_plot_impulse()
        elif hasattr(self.main_window, 'impulse_manager'):
            self.main_window.impulse_manager.update_plot_impulse()
        
        # Prüfe ob Checkbox aktiviert ist
        checkbox = self.snapshot_tree_widget.itemWidget(item, 1)
        if not checkbox or not checkbox.isChecked():
            return
        
        # ========================================
        # SPECIAL CASE: "current speakers" (aktuelle_simulation)
        # ========================================
        if snapshot_key == "aktuelle_simulation":
            # Beim Zurückkehren zur aktuellen Simulation die gesicherten Daten wiederherstellen
            self._restore_current_data()
            checkbox = self.snapshot_tree_widget.itemWidget(item, 1)
            if checkbox and not checkbox.isChecked():
                blocker = QtCore.QSignalBlocker(checkbox)
                checkbox.setChecked(True)
                del blocker
            aktuelle_axes = self.container.calculation_axes.get("aktuelle_simulation")
            if isinstance(aktuelle_axes, dict):
                aktuelle_axes["show_in_plot"] = True
            if hasattr(self.container, "calculation_spl"):
                self.container.calculation_spl["show_in_plot"] = True
            if hasattr(self.container, "calculation_polar"):
                self.container.calculation_polar["show_in_plot"] = True
            if hasattr(self.main_window, "plot_spl"):
                self.main_window.plot_spl()
                if hasattr(self.main_window, "plot_polar_pattern"):
                    self.main_window.plot_polar_pattern()
            return
        
        # ========================================
        # NORMAL CASE: Lade Snapshot-Daten
        # ========================================
        # Prüfe ob das Item im Container existiert
        if snapshot_key not in self.container.calculation_axes:
            return
        
        # Wenn wir gerade von der aktuellen Simulation zu einem Snapshot wechseln, Daten sichern
        if previous_key == "aktuelle_simulation":
            self._backup_current_data()

        snapshot_data = self.container.calculation_axes[snapshot_key]
        
        # ========================================
        # 1. Aktualisiere SPL 2D-Schallfeld-Daten
        # ========================================
        if 'spl_field_data' in snapshot_data:
            import numpy as np
            spl_data = snapshot_data['spl_field_data']
            
            # Konvertiere Listen zurück zu NumPy-Arrays
            if 'sound_field_p' in spl_data:
                self.container.calculation_spl['sound_field_p'] = (
                    np.array(spl_data['sound_field_p']) if isinstance(spl_data['sound_field_p'], list)
                    else spl_data['sound_field_p']
                )
            
            if 'sound_field_x' in spl_data:
                self.container.calculation_spl['sound_field_x'] = (
                    np.array(spl_data['sound_field_x']) if isinstance(spl_data['sound_field_x'], list)
                    else spl_data['sound_field_x']
                )
            
            if 'sound_field_y' in spl_data:
                self.container.calculation_spl['sound_field_y'] = (
                    np.array(spl_data['sound_field_y']) if isinstance(spl_data['sound_field_y'], list)
                    else spl_data['sound_field_y']
                )
        
        # ========================================
        # 2. Aktualisiere Polarplot-Daten
        # ========================================
        if 'polar_data' in snapshot_data:
            polar_data = snapshot_data['polar_data']
            self.container.calculation_polar['sound_field_p'] = polar_data.get('sound_field_p', {})
            self.container.calculation_polar['angles'] = polar_data.get('angles', None)
            self.container.calculation_polar['frequencies'] = polar_data.get('frequencies', {})
        
        # ========================================
        # 3. Plots neu zeichnen
        # ========================================
        # SPL Plot neu zeichnen
        if hasattr(self.main_window, 'plot_spl'):
            try:
                self.main_window.plot_spl()
            except Exception as e:
                print(f"Fehler beim Aktualisieren des SPL Plots: {e}")
        
        # Polarplot neu zeichnen
        if hasattr(self.main_window, 'plot_polar_pattern'):
            try:
                self.main_window.plot_polar_pattern()
            except Exception as e:
                print(f"Fehler beim Aktualisieren des Polarplots: {e}")

    def show_context_menu(self, position):
        """Zeigt das Kontextmenü beim Rechtsklick auf ein Snapshot-Item"""
        item = self.snapshot_tree_widget.itemAt(position)
        if item:
            # Prüfe ob es sich um "current speakers" handelt
            snapshot_key = item.data(0, Qt.UserRole)
            if snapshot_key == "aktuelle_simulation":
                # Kein Kontextmenü für "current speakers"
                return
            
            context_menu = QMenu()
            
            # Delete Action
            delete_action = context_menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.delete_snapshot(item))
            
            # Zeige Menü an Mausposition
            context_menu.exec_(self.snapshot_tree_widget.viewport().mapToGlobal(position))
    
    def delete_snapshot(self, item):
        """Löscht einen Snapshot"""
        if item:
            snapshot_key = item.data(0, Qt.UserRole)
            if snapshot_key in self.container.calculation_axes:
                del self.container.calculation_axes[snapshot_key]
                
                # Entferne Item aus TreeWidget
                index = self.snapshot_tree_widget.indexOfTopLevelItem(item)
                if index != -1:
                    self.snapshot_tree_widget.takeTopLevelItem(index)
                
                print(f"Snapshot '{snapshot_key}' gelöscht")
                
                # Aktualisiere alle relevanten Plots
                if hasattr(self.main_window, 'plot_xaxis'):
                    self.main_window.plot_xaxis()
                if hasattr(self.main_window, 'plot_yaxis'):
                    self.main_window.plot_yaxis()
                if hasattr(self.main_window, 'update_plot_impulse'):
                    self.main_window.update_plot_impulse()
                elif hasattr(self.main_window, 'impulse_manager'):
                    self.main_window.impulse_manager.update_plot_impulse()

    def _backup_current_data(self):
        """Sichert aktuelle SPL-, Polar-, Achsen- und Impuls-Daten bevor ein Snapshot geladen wird."""
        if hasattr(self.container, "calculation_spl"):
            self._current_spl_backup = copy.deepcopy(self.container.calculation_spl)
        else:
            self._current_spl_backup = None

        if hasattr(self.container, "calculation_polar"):
            self._current_polar_backup = copy.deepcopy(self.container.calculation_polar)
        else:
            self._current_polar_backup = None

        if hasattr(self.container, "calculation_axes"):
            aktuelle_axes = self.container.calculation_axes.get("aktuelle_simulation")
            self._current_axes_backup = copy.deepcopy(aktuelle_axes) if isinstance(aktuelle_axes, dict) else None
        else:
            self._current_axes_backup = None

        if hasattr(self.container, "calculation_impulse"):
            aktuelle_impulse = self.container.calculation_impulse.get("aktuelle_simulation")
            self._current_impulse_backup = copy.deepcopy(aktuelle_impulse) if isinstance(aktuelle_impulse, dict) else None
        else:
            self._current_impulse_backup = None

    def _restore_current_data(self):
        """Stellt gesicherte SPL-, Polar-, Achsen- und Impuls-Daten der aktuellen Simulation wieder her."""
        if self._current_spl_backup is not None and hasattr(self.container, "calculation_spl"):
            self.container.calculation_spl.clear()
            self.container.calculation_spl.update(copy.deepcopy(self._current_spl_backup))
        if self._current_polar_backup is not None and hasattr(self.container, "calculation_polar"):
            self.container.calculation_polar.clear()
            self.container.calculation_polar.update(copy.deepcopy(self._current_polar_backup))
        if self._current_axes_backup is not None and hasattr(self.container, "calculation_axes"):
            self.container.calculation_axes["aktuelle_simulation"] = copy.deepcopy(self._current_axes_backup)
        if self._current_impulse_backup is not None and hasattr(self.container, "calculation_impulse"):
            self.container.calculation_impulse["aktuelle_simulation"] = copy.deepcopy(self._current_impulse_backup)

    def close_capture_dock_widget(self):
        if self.capture_dock_widget:
            self.capture_dock_widget.close()
            self.capture_dock_widget.deleteLater()
            self.capture_dock_widget = None

    def clear_snapshot_data(self):
        self.container.calculation_axes.clear()
        self.close_capture_dock_widget()

    def has_snapshot_data(self):
        """Prüft ob Snapshots vorhanden sind (außer aktuelle_simulation)"""
        return any(key != "aktuelle_simulation" for key in self.container.calculation_axes.keys())
    
    def has_any_active_snapshot(self):
        """Prüft ob irgendein Snapshot (inkl. current speakers) aktiviert ist"""
        for key, value in self.container.calculation_axes.items():
            if value.get("show_in_plot", False):
                return True
        return False
