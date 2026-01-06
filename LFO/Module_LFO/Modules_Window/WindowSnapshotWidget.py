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
        self._current_time_frame_index_backup = 0
        self._current_time_frames_per_period_backup = 16
    
    def _is_widget_valid(self):
        """Prüft ob snapshot_tree_widget existiert und noch gültig ist (nicht gelöscht)"""
        if not self.snapshot_tree_widget:
            return False
        try:
            _ = self.snapshot_tree_widget.objectName()
            return True
        except RuntimeError:
            # Widget wurde gelöscht
            self.snapshot_tree_widget = None
            return False

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
        
        # Mehrfachauswahl aktivieren
        self.snapshot_tree_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
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
        if not self._is_widget_valid():
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
        # Stelle sicher, dass die Daten der aktuellen Simulation aktiv sind
        # ========================================

        # Finde das "current speakers" Item im TreeWidget
        root = self.snapshot_tree_widget.invisibleRootItem()
        current_speakers_item = None
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == "aktuelle_simulation":
                current_speakers_item = item
                break

        if self.selected_snapshot_key != "aktuelle_simulation":
            # Stelle ursprüngliche Daten wieder her
            self._restore_current_data()
            self.selected_snapshot_key = "aktuelle_simulation"
            if current_speakers_item is not None:
                blocker = QtCore.QSignalBlocker(self.snapshot_tree_widget)
                self.snapshot_tree_widget.setCurrentItem(current_speakers_item)
                del blocker
        
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
            
            # 🎯 NEU: Speichere Surface-Geometrie (points) für alle Surfaces mit SPL-Daten
            # Dies ermöglicht später die Validierung beim Laden des Snapshots
            if 'surface_grids' in self.container.calculation_spl or 'surface_results' in self.container.calculation_spl:
                surface_geometries = {}
                surface_definitions = getattr(self.settings, 'surface_definitions', {})
                if isinstance(surface_definitions, dict):
                    # Sammle alle Surface-IDs, die SPL-Daten haben
                    surface_ids_with_spl = set()
                    if 'surface_grids' in self.container.calculation_spl:
                        surface_ids_with_spl.update(self.container.calculation_spl['surface_grids'].keys())
                    if 'surface_results' in self.container.calculation_spl:
                        surface_ids_with_spl.update(self.container.calculation_spl['surface_results'].keys())
                    
                    # Speichere Geometrie für jede Surface mit SPL-Daten
                    for surface_id in surface_ids_with_spl:
                        surface = surface_definitions.get(surface_id)
                        if surface:
                            if isinstance(surface, dict):
                                points = surface.get('points', [])
                            else:
                                points = getattr(surface, 'points', [])
                            
                            # Speichere normalisierte Punkte (für Vergleich)
                            if points:
                                # Konvertiere zu Liste von Dictionaries mit gerundeten Werten für Vergleich
                                normalized_points = []
                                for p in points:
                                    if isinstance(p, dict):
                                        normalized_points.append({
                                            'x': round(float(p.get('x', 0.0)), 6),
                                            'y': round(float(p.get('y', 0.0)), 6),
                                            'z': round(float(p.get('z', 0.0)), 6)
                                        })
                                    else:
                                        normalized_points.append({
                                            'x': round(float(getattr(p, 'x', 0.0)), 6),
                                            'y': round(float(getattr(p, 'y', 0.0)), 6),
                                            'z': round(float(getattr(p, 'z', 0.0)), 6)
                                        })
                                surface_geometries[surface_id] = normalized_points
                
                if surface_geometries:
                    capture_data['surface_geometries'] = surface_geometries
                    # Speichere auch surface_grids und surface_results für kompatible Surfaces
                    if 'surface_grids' in self.container.calculation_spl:
                        snapshot_surface_grids = {}
                        for sid in surface_ids_with_spl:
                            if sid in self.container.calculation_spl['surface_grids']:
                                # Deep copy der Grid-Daten
                                grid_data = copy.deepcopy(self.container.calculation_spl['surface_grids'][sid])
                                # Konvertiere NumPy-Arrays zu Listen
                                import numpy as np
                                for key in ['X_grid', 'Y_grid', 'Z_grid', 'sound_field_x', 'sound_field_y']:
                                    if key in grid_data and isinstance(grid_data[key], np.ndarray):
                                        grid_data[key] = grid_data[key].tolist()
                                if 'surface_mask' in grid_data and isinstance(grid_data['surface_mask'], np.ndarray):
                                    grid_data['surface_mask'] = grid_data['surface_mask'].tolist()
                                snapshot_surface_grids[sid] = grid_data
                        if snapshot_surface_grids:
                            capture_data['surface_grids'] = snapshot_surface_grids
                    
                    if 'surface_results' in self.container.calculation_spl:
                        snapshot_surface_results = {}
                        for sid in surface_ids_with_spl:
                            if sid in self.container.calculation_spl['surface_results']:
                                # Deep copy der Result-Daten
                                result_data = copy.deepcopy(self.container.calculation_spl['surface_results'][sid])
                                # Konvertiere NumPy-Arrays zu Listen
                                import numpy as np
                                if isinstance(result_data, dict):
                                    for key, value in result_data.items():
                                        if isinstance(value, np.ndarray):
                                            result_data[key] = value.tolist()
                                snapshot_surface_results[sid] = result_data
                        if snapshot_surface_results:
                            capture_data['surface_results'] = snapshot_surface_results
        
        # ========================================
        # 4. FDTD-Simulationsdaten speichern (für "SPL over time" Modus)
        # ========================================
        if hasattr(self.container, 'calculation_spl') and self.container.calculation_spl:
            # Speichere alle FDTD-Frames für Zeit-Slider-Funktionalität
            if 'fdtd_simulation' in self.container.calculation_spl:
                import numpy as np
                fdtd_sim = self.container.calculation_spl['fdtd_simulation']
                # Deep copy der FDTD-Daten (kann große Arrays enthalten)
                fdtd_data = copy.deepcopy(fdtd_sim)
                # Konvertiere NumPy-Arrays zu Listen für JSON-Kompatibilität
                if isinstance(fdtd_data, dict):
                    for freq_key, sim_data in fdtd_data.items():
                        if isinstance(sim_data, dict) and 'pressure_frames' in sim_data:
                            pressure_frames = sim_data['pressure_frames']
                            if isinstance(pressure_frames, np.ndarray):
                                sim_data['pressure_frames'] = pressure_frames.tolist()
                            elif isinstance(pressure_frames, list):
                                # Bereits Liste, aber könnte verschachtelte Arrays enthalten
                                sim_data['pressure_frames'] = [
                                    frame.tolist() if isinstance(frame, np.ndarray) else frame
                                    for frame in pressure_frames
                                ]
                        # Konvertiere auch sound_field_x und sound_field_y falls vorhanden
                        if isinstance(sim_data, dict):
                            for key in ['sound_field_x', 'sound_field_y']:
                                if key in sim_data:
                                    arr = sim_data[key]
                                    if isinstance(arr, np.ndarray):
                                        sim_data[key] = arr.tolist()
                capture_data['fdtd_simulation'] = fdtd_data
                print(f"[Snapshot] FDTD-Daten gespeichert: {len(fdtd_data)} Frequenz(en)")
            
            # Speichere auch aktuelle Zeit-Frame-Einstellungen
            if hasattr(self.main_window, 'draw_plots'):
                draw_plots = self.main_window.draw_plots
                if hasattr(draw_plots, '_time_frame_index'):
                    capture_data['fdtd_time_frame_index'] = draw_plots._time_frame_index
                if hasattr(draw_plots, '_time_frames_per_period'):
                    capture_data['fdtd_time_frames_per_period'] = draw_plots._time_frames_per_period
        
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
        """
        Wird aufgerufen wenn sich die Checkbox ändert.
        Beeinflusst nur X/Y-Achsen-Plots und Impuls-Plot.
        SPL und Polarplot werden nicht durch Checkbox beeinflusst.
        """
        if key in self.container.calculation_axes:
            show = bool(state)
            self.container.calculation_axes[key]["show_in_plot"] = show
            is_selected = (key == getattr(self, "selected_snapshot_key", "aktuelle_simulation"))

            # Checkbox beeinflusst SPL und Polarplot NICHT mehr
            # Diese werden nur durch Item-Auswahl gesteuert

            # Achsen und Impulsplots immer anhand der Checkbox aktualisieren
            if hasattr(self.main_window, 'plot_xaxis'):
                self.main_window.plot_xaxis()
            if hasattr(self.main_window, 'plot_yaxis'):
                self.main_window.plot_yaxis()
            if hasattr(self.main_window, 'update_plot_impulse'):
                self.main_window.update_plot_impulse()
            elif hasattr(self.main_window, 'impulse_manager'):
                self.main_window.impulse_manager.update_plot_impulse()

            # Prüfe ob irgendein Snapshot für Achsen/Impuls aktiviert ist
            if not self.has_any_active_snapshot():
                if hasattr(self.main_window, 'draw_plots'):
                    self.main_window.draw_plots.show_empty_axes()
                    # SPL und Polar nur leeren wenn kein Snapshot ausgewählt ist
                    if not is_selected:
                        self.main_window.draw_plots.show_empty_polar()
                        self.main_window.draw_plots.show_empty_spl()
                return

            # SPL und Polarplot werden NICHT durch Checkbox beeinflusst
            # Sie bleiben immer angezeigt wenn der Snapshot ausgewählt ist
    
    def _select_item_by_key(self, snapshot_key):
        """Wählt ein Item im TreeWidget basierend auf dem Key aus"""
        if not self._is_widget_valid():
            return
        root = self.snapshot_tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, Qt.UserRole) == snapshot_key:
                self.snapshot_tree_widget.setCurrentItem(item)
                return
    
    def on_snapshot_item_clicked(self, item, column):
        """
        Wird aufgerufen wenn ein Snapshot-Item angeklickt wird.
        Zeigt die Daten dieses Snapshots in SPL- und Polarplot an (unabhängig von Checkbox).
        Checkbox beeinflusst nur X/Y-Achsen-Plots und Impuls-Plot.
        Wenn mehrere Items ausgewählt sind, wird das oberste angezeigt.
        """
        # Wenn mehrere Items ausgewählt sind, verwende das oberste
        selected_items = self.snapshot_tree_widget.selectedItems()
        if selected_items:
            # Finde das oberste ausgewählte Item
            root = self.snapshot_tree_widget.invisibleRootItem()
            topmost_item = None
            topmost_index = float('inf')
            
            for sel_item in selected_items:
                index = self.snapshot_tree_widget.indexOfTopLevelItem(sel_item)
                if index != -1 and index < topmost_index:
                    topmost_index = index
                    topmost_item = sel_item
            
            # Verwende das oberste Item für die Anzeige
            if topmost_item:
                item = topmost_item
        
        # Hole den Key des geklickten Items
        snapshot_key = item.data(0, Qt.UserRole)
        if not snapshot_key:
            return
        
        previous_key = getattr(self, "selected_snapshot_key", "aktuelle_simulation")
        # Speichere das ausgewählte Item (oberstes wenn mehrere ausgewählt)
        self.selected_snapshot_key = snapshot_key
        
        # Hole Checkbox-Status für Achsen/Impuls-Plots
        checkbox = self.snapshot_tree_widget.itemWidget(item, 1)
        is_checked = checkbox.isChecked() if checkbox else False
        
        # Aktualisiere Achsen-Plots basierend auf Checkbox
        if hasattr(self.main_window, 'plot_xaxis'):
            self.main_window.plot_xaxis()
        if hasattr(self.main_window, 'plot_yaxis'):
            self.main_window.plot_yaxis()
        # Aktualisiere Impuls-Plots basierend auf Checkbox
        if hasattr(self.main_window, 'update_plot_impulse'):
            self.main_window.update_plot_impulse()
        elif hasattr(self.main_window, 'impulse_manager'):
            self.main_window.impulse_manager.update_plot_impulse()
        
        # ========================================
        # SPECIAL CASE: "current speakers" (aktuelle_simulation)
        # ========================================
        if snapshot_key == "aktuelle_simulation":
            # Beim Zurückkehren zur aktuellen Simulation die gesicherten Daten wiederherstellen
            self._restore_current_data()
            aktuelle_axes = self.container.calculation_axes.get("aktuelle_simulation")
            if isinstance(aktuelle_axes, dict):
                aktuelle_axes["show_in_plot"] = bool(is_checked)
            
            # SPL und Polarplot immer anzeigen (unabhängig von Checkbox)
            if hasattr(self.container, "calculation_spl"):
                self.container.calculation_spl["show_in_plot"] = True
            if hasattr(self.container, "calculation_polar"):
                self.container.calculation_polar["show_in_plot"] = True

            # SPL und Polarplot immer plotten
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
        # 1. Aktualisiere SPL 2D-Schallfeld-Daten (immer laden wenn Snapshot ausgewählt)
        # ========================================
        has_spl_data = False
        import numpy as np
        
        # 🎯 NEU: Prüfe Surface-Kompatibilität und lade nur kompatible Surface-SPL-Daten
        compatible_surface_ids = set()
        incompatible_surface_ids = []
        if 'surface_geometries' in snapshot_data:
            # Prüfe welche Surfaces noch vorhanden sind und die gleiche Geometrie haben
            snapshot_geometries = snapshot_data['surface_geometries']
            surface_definitions = getattr(self.settings, 'surface_definitions', {})
            
            if isinstance(surface_definitions, dict) and isinstance(snapshot_geometries, dict):
                for surface_id, snapshot_points in snapshot_geometries.items():
                    # Prüfe ob Surface noch vorhanden ist
                    surface = surface_definitions.get(surface_id)
                    if not surface:
                        incompatible_surface_ids.append((surface_id, "nicht mehr vorhanden"))
                        continue  # Surface nicht mehr vorhanden
                    
                    # 🎯 NEU: Prüfe ob Surface enabled und nicht hidden ist
                    is_enabled = True
                    is_hidden = False
                    if isinstance(surface, dict):
                        is_enabled = bool(surface.get('enabled', True))
                        is_hidden = bool(surface.get('hidden', False))
                    else:
                        is_enabled = bool(getattr(surface, 'enabled', True))
                        is_hidden = bool(getattr(surface, 'hidden', False))
                    
                    if not is_enabled:
                        incompatible_surface_ids.append((surface_id, "disabled"))
                        continue  # Surface ist disabled → nicht plotten
                    
                    if is_hidden:
                        incompatible_surface_ids.append((surface_id, "hidden"))
                        continue  # Surface ist hidden → nicht plotten
                    
                    # Hole aktuelle Surface-Punkte
                    if isinstance(surface, dict):
                        current_points = surface.get('points', [])
                    else:
                        current_points = getattr(surface, 'points', [])
                    
                    if not current_points:
                        incompatible_surface_ids.append((surface_id, "keine Punkte vorhanden"))
                        continue  # Keine Punkte vorhanden
                    
                    # Normalisiere aktuelle Punkte für Vergleich
                    normalized_current = []
                    for p in current_points:
                        if isinstance(p, dict):
                            normalized_current.append({
                                'x': round(float(p.get('x', 0.0)), 6),
                                'y': round(float(p.get('y', 0.0)), 6),
                                'z': round(float(p.get('z', 0.0)), 6)
                            })
                        else:
                            normalized_current.append({
                                'x': round(float(getattr(p, 'x', 0.0)), 6),
                                'y': round(float(getattr(p, 'y', 0.0)), 6),
                                'z': round(float(getattr(p, 'z', 0.0)), 6)
                            })
                    
                    # Vergleiche Geometrien
                    if len(normalized_current) != len(snapshot_points):
                        incompatible_surface_ids.append((surface_id, f"Anzahl Punkte geändert: {len(normalized_current)} != {len(snapshot_points)}"))
                        continue
                    
                    geometries_match = True
                    for cp, sp in zip(normalized_current, snapshot_points):
                        if (abs(cp['x'] - sp['x']) > 1e-5 or
                            abs(cp['y'] - sp['y']) > 1e-5 or
                            abs(cp['z'] - sp['z']) > 1e-5):
                            geometries_match = False
                            break
                    
                    if geometries_match:
                        compatible_surface_ids.add(surface_id)
                    else:
                        incompatible_surface_ids.append((surface_id, "Geometrie geändert"))
                
                if incompatible_surface_ids:
                    pass  # Debug-Ausgabe entfernt
                    if len(incompatible_surface_ids) > 5:
                        print(f"  ... und {len(incompatible_surface_ids) - 5} weitere")
        
        # Lade globale SPL-Daten (falls vorhanden)
        if 'spl_field_data' in snapshot_data and snapshot_data['spl_field_data']:
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
            has_spl_data = True
        
        # 🎯 NEU: Lade nur Surface-SPL-Daten für kompatible Surfaces
        # WICHTIG: Entferne zuerst alle vorhandenen Surface-Daten, damit keine alten Daten übrig bleiben
        if 'surface_grids' in self.container.calculation_spl:
            self.container.calculation_spl['surface_grids'].clear()
        else:
            self.container.calculation_spl['surface_grids'] = {}
        
        if 'surface_results' in self.container.calculation_spl:
            self.container.calculation_spl['surface_results'].clear()
        else:
            self.container.calculation_spl['surface_results'] = {}
        
        # 🎯 WICHTIG: Lade Surface-Daten NUR wenn surface_geometries vorhanden ist
        # (bei alten Snapshots ohne Geometrie-Validierung werden keine Surface-Daten geladen)
        if 'surface_geometries' in snapshot_data:
            # Lade nur kompatible Surface-Grids aus dem Snapshot
            if 'surface_grids' in snapshot_data:
                snapshot_surface_grids = snapshot_data['surface_grids']
                if isinstance(snapshot_surface_grids, dict):
                    for surface_id, grid_data in snapshot_surface_grids.items():
                        if surface_id in compatible_surface_ids:
                            # Konvertiere Listen zurück zu NumPy-Arrays
                            restored_grid = copy.deepcopy(grid_data)
                            for key in ['X_grid', 'Y_grid', 'Z_grid', 'sound_field_x', 'sound_field_y']:
                                if key in restored_grid and isinstance(restored_grid[key], list):
                                    restored_grid[key] = np.array(restored_grid[key], dtype=float)
                            if 'surface_mask' in restored_grid and isinstance(restored_grid['surface_mask'], list):
                                restored_grid['surface_mask'] = np.array(restored_grid['surface_mask'], dtype=bool)
                            self.container.calculation_spl['surface_grids'][surface_id] = restored_grid
                        # Surface nicht kompatibel → nicht laden (wird nicht geplottet)
            
            # Lade nur kompatible Surface-Results aus dem Snapshot
            if 'surface_results' in snapshot_data:
                snapshot_surface_results = snapshot_data['surface_results']
                if isinstance(snapshot_surface_results, dict):
                    for surface_id, result_data in snapshot_surface_results.items():
                        if surface_id in compatible_surface_ids:
                            # Konvertiere Listen zurück zu NumPy-Arrays
                            restored_result = copy.deepcopy(result_data)
                            if isinstance(restored_result, dict):
                                for key, value in restored_result.items():
                                    if isinstance(value, list):
                                        # Prüfe ob es ein NumPy-Array sein sollte (versuche Konvertierung)
                                        try:
                                            restored_result[key] = np.array(value, dtype=float)
                                        except (ValueError, TypeError):
                                            # Bleibt als Liste wenn Konvertierung fehlschlägt
                                            pass
                            self.container.calculation_spl['surface_results'][surface_id] = restored_result
                        # Surface nicht kompatibel → nicht laden (wird nicht geplottet)
            
            if compatible_surface_ids:
                has_spl_data = True  # Mindestens eine kompatible Surface vorhanden
            elif 'surface_grids' in snapshot_data or 'surface_results' in snapshot_data:
                # Snapshot hatte Surface-Daten, aber keine sind kompatibel
                print(f"[Snapshot] Keine kompatiblen Surfaces gefunden (Geometrie geändert oder entfernt)")
        else:
            # Keine surface_geometries im Snapshot → alte Snapshot-Version ohne Validierung
            # Lade KEINE Surface-Daten, um Fehl-Plots zu vermeiden
            if 'surface_grids' in snapshot_data or 'surface_results' in snapshot_data:
                print(f"[Snapshot] Warnung: Snapshot enthält Surface-Daten, aber keine Geometrie-Validierung. Surface-Daten werden nicht geladen.")
        
        if has_spl_data:
            if hasattr(self.container, "calculation_spl"):
                self.container.calculation_spl["show_in_plot"] = True
        else:
            if hasattr(self.container, "calculation_spl"):
                try:
                    self.container.calculation_spl.clear()
                except AttributeError:
                    self.container.calculation_spl = {}
                self.container.calculation_spl["show_in_plot"] = False
            if hasattr(self.main_window, 'draw_plots'):
                self.main_window.draw_plots.show_empty_spl()
        
        # ========================================
        # 1b. FDTD-Simulationsdaten wiederherstellen (für "SPL over time" Modus)
        # ========================================
        has_fdtd_data = False
        if 'fdtd_simulation' in snapshot_data and snapshot_data['fdtd_simulation']:
            import numpy as np
            fdtd_data = snapshot_data['fdtd_simulation']
            # Konvertiere Listen zurück zu NumPy-Arrays
            if isinstance(fdtd_data, dict) and bool(fdtd_data):
                restored_fdtd = {}
                for freq_key, sim_data in fdtd_data.items():
                    if isinstance(sim_data, dict):
                        restored_sim = copy.deepcopy(sim_data)
                        # Konvertiere pressure_frames zurück zu NumPy-Array
                        if 'pressure_frames' in restored_sim:
                            pressure_frames = restored_sim['pressure_frames']
                            if isinstance(pressure_frames, list):
                                # Konvertiere verschachtelte Listen zu NumPy-Array
                                restored_sim['pressure_frames'] = np.array(pressure_frames, dtype=np.float32)
                        # Konvertiere sound_field_x und sound_field_y zurück
                        for key in ['sound_field_x', 'sound_field_y']:
                            if key in restored_sim and isinstance(restored_sim[key], list):
                                restored_sim[key] = np.array(restored_sim[key], dtype=float)
                        restored_fdtd[freq_key] = restored_sim
                self.container.calculation_spl['fdtd_simulation'] = restored_fdtd
                has_fdtd_data = True
                print(f"[Snapshot] FDTD-Daten wiederhergestellt: {len(restored_fdtd)} Frequenz(en)")
            
            # Stelle Zeit-Frame-Einstellungen wieder her
            if hasattr(self.main_window, 'draw_plots'):
                draw_plots = self.main_window.draw_plots
                if 'fdtd_time_frame_index' in snapshot_data:
                    draw_plots._time_frame_index = int(snapshot_data['fdtd_time_frame_index'])
                if 'fdtd_time_frames_per_period' in snapshot_data:
                    frames_per_period = int(snapshot_data['fdtd_time_frames_per_period'])
                    draw_plots._time_frames_per_period = frames_per_period
                    # Aktualisiere auch Settings
                    if hasattr(self.settings, 'fem_time_frames_per_period'):
                        self.settings.fem_time_frames_per_period = frames_per_period
                    print(f"[Snapshot] Zeit-Frame-Einstellungen wiederhergestellt: Frame {draw_plots._time_frame_index}/{frames_per_period-1}")
        else:
            # Keine FDTD-Daten im Snapshot - entferne vorhandene FDTD-Daten
            if hasattr(self.container, "calculation_spl") and 'fdtd_simulation' in self.container.calculation_spl:
                del self.container.calculation_spl['fdtd_simulation']
        
        # ========================================
        # 1c. Prüfe aktuellen Plot-Modus (NICHT ändern, nur aktualisieren)
        # ========================================
        current_mode = getattr(self.settings, 'spl_plot_mode', 'SPL plot')
        
        # Prüfe ob wir im "SPL over time" Modus sind, aber keine FDTD-Daten vorhanden sind
        if current_mode == "SPL over time" and not has_fdtd_data:
            # Im "SPL over time" Modus aber keine FDTD-Daten -> leerer Plot
            print(f"[Snapshot] Keine FDTD-Daten vorhanden, zeige leeren Plot")
            if hasattr(self.main_window, 'draw_plots'):
                self.main_window.draw_plots.show_empty_spl()
            # Polarplot trotzdem aktualisieren falls vorhanden
            if 'polar_data' in snapshot_data and snapshot_data['polar_data']:
                polar_data = snapshot_data['polar_data']
                self.container.calculation_polar['sound_field_p'] = polar_data.get('sound_field_p', {})
                self.container.calculation_polar['angles'] = polar_data.get('angles', None)
                self.container.calculation_polar['frequencies'] = polar_data.get('frequencies', {})
                self.container.calculation_polar['show_in_plot'] = True
                if hasattr(self.main_window, 'plot_polar_pattern'):
                    try:
                        self.main_window.plot_polar_pattern()
                    except Exception as e:
                        print(f"Fehler beim Aktualisieren des Polarplots: {e}")
            return  # Früh beenden, kein SPL-Plot nötig
        
        # ========================================
        # 2. Aktualisiere Polarplot-Daten (immer laden wenn Snapshot ausgewählt)
        # ========================================
        has_polar_data = False
        if 'polar_data' in snapshot_data and snapshot_data['polar_data']:
            polar_data = snapshot_data['polar_data']
            self.container.calculation_polar['sound_field_p'] = polar_data.get('sound_field_p', {})
            self.container.calculation_polar['angles'] = polar_data.get('angles', None)
            self.container.calculation_polar['frequencies'] = polar_data.get('frequencies', {})
            self.container.calculation_polar['show_in_plot'] = True
            has_polar_data = True
        else:
            if hasattr(self.container, "calculation_polar"):
                try:
                    self.container.calculation_polar.clear()
                except AttributeError:
                    self.container.calculation_polar = {}
                self.container.calculation_polar['show_in_plot'] = False
            if hasattr(self.main_window, 'draw_plots'):
                self.main_window.draw_plots.show_empty_polar()
        
        # ========================================
        # 3. Plots neu zeichnen (aktueller Plot-Modus bleibt erhalten)
        # ========================================
        # Aktualisiere den aktuell geöffneten Plot (nicht den Modus ändern)
        # Nur plotten wenn FDTD-Daten vorhanden sind ODER normale SPL-Daten vorhanden sind
        if has_fdtd_data or has_spl_data:
            if hasattr(self.main_window, 'plot_spl'):
                try:
                    # plot_spl() verwendet den aktuellen Plot-Modus aus Settings
                    self.main_window.plot_spl()
                except Exception as e:
                    print(f"Fehler beim Aktualisieren des SPL Plots: {e}")
        
        # Polarplot neu zeichnen
        if has_polar_data and hasattr(self.main_window, 'plot_polar_pattern'):
            try:
                self.main_window.plot_polar_pattern()
            except Exception as e:
                print(f"Fehler beim Aktualisieren des Polarplots: {e}")

    def show_context_menu(self, position):
        """Zeigt das Kontextmenü beim Rechtsklick auf ein Snapshot-Item"""
        item = self.snapshot_tree_widget.itemAt(position)
        selected_items = self.snapshot_tree_widget.selectedItems()
        
        # Wenn kein Item unter der Maus, aber Items ausgewählt sind, verwende diese
        if not item and selected_items:
            item = selected_items[0]
        
        if item:
            # Prüfe ob es sich um "current speakers" handelt
            snapshot_key = item.data(0, Qt.UserRole)
            if snapshot_key == "aktuelle_simulation":
                # Kein Kontextmenü für "current speakers"
                return
            
            # Filtere "current speakers" aus den ausgewählten Items heraus
            items_to_delete = [it for it in selected_items if it.data(0, Qt.UserRole) != "aktuelle_simulation"]
            
            # Wenn kein Item zum Löschen vorhanden, beende
            if not items_to_delete:
                return
            
            context_menu = QMenu()
            
            # Delete Action - Text anpassen je nach Anzahl
            if len(items_to_delete) > 1:
                delete_action = context_menu.addAction(f"Delete ({len(items_to_delete)} Snapshots)")
            else:
                delete_action = context_menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.delete_snapshots(items_to_delete))
            
            # Zeige Menü an Mausposition
            context_menu.exec_(self.snapshot_tree_widget.viewport().mapToGlobal(position))
    
    def delete_snapshot(self, item):
        """Löscht einen Snapshot (Legacy-Methode für Rückwärtskompatibilität)"""
        self.delete_snapshots([item])
    
    def delete_snapshots(self, items):
        """Löscht mehrere Snapshots gleichzeitig"""
        if not items:
            return
        
        # Speichere das aktuell oberste ausgewählte Item (für Plot-Anzeige)
        topmost_item = None
        topmost_index = float('inf')
        
        for item in items:
            index = self.snapshot_tree_widget.indexOfTopLevelItem(item)
            if index != -1 and index < topmost_index:
                topmost_index = index
                topmost_item = item
        
        # Prüfe ob das oberste gelöschte Item das aktuell angezeigte ist
        topmost_key = topmost_item.data(0, Qt.UserRole) if topmost_item else None
        was_topmost_selected = (topmost_key == self.selected_snapshot_key)
        
        # Sortiere Items nach Index (von unten nach oben), damit Indizes beim Löschen korrekt bleiben
        items_with_indices = []
        for item in items:
            index = self.snapshot_tree_widget.indexOfTopLevelItem(item)
            if index != -1:
                items_with_indices.append((index, item))
        items_with_indices.sort(reverse=True)  # Von unten nach oben sortieren
        
        # Lösche alle Snapshots
        deleted_keys = []
        for index, item in items_with_indices:
            snapshot_key = item.data(0, Qt.UserRole)
            if snapshot_key and snapshot_key != "aktuelle_simulation":
                if snapshot_key in self.container.calculation_axes:
                    del self.container.calculation_axes[snapshot_key]
                    deleted_keys.append(snapshot_key)
                    
                    # Entferne Item aus TreeWidget (in umgekehrter Reihenfolge)
                    self.snapshot_tree_widget.takeTopLevelItem(index)
        
        if deleted_keys:
            if len(deleted_keys) > 1:
                print(f"{len(deleted_keys)} Snapshots gelöscht: {', '.join(deleted_keys)}")
            else:
                print(f"Snapshot '{deleted_keys[0]}' gelöscht")
        
        # Wenn das oberste gelöschte Item das aktuell angezeigte war, wechsle zu "current speakers"
        if was_topmost_selected:
            self.selected_snapshot_key = "aktuelle_simulation"
            # Wähle "current speakers" Item aus
            self._select_item_by_key("aktuelle_simulation")
            # Lade "current speakers" Daten
            current_item = None
            root = self.snapshot_tree_widget.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                if item.data(0, Qt.UserRole) == "aktuelle_simulation":
                    current_item = item
                    break
            if current_item:
                self.on_snapshot_item_clicked(current_item, 0)
        
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
        
        # Sichere auch Zeit-Frame-Einstellungen
        if hasattr(self.main_window, 'draw_plots'):
            draw_plots = self.main_window.draw_plots
            self._current_time_frame_index_backup = getattr(draw_plots, '_time_frame_index', 0)
            self._current_time_frames_per_period_backup = getattr(draw_plots, '_time_frames_per_period', 16)
        else:
            self._current_time_frame_index_backup = 0
            self._current_time_frames_per_period_backup = 16

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
        
        # Stelle auch Zeit-Frame-Einstellungen wieder her
        if hasattr(self.main_window, 'draw_plots'):
            draw_plots = self.main_window.draw_plots
            if hasattr(self, '_current_time_frame_index_backup'):
                draw_plots._time_frame_index = self._current_time_frame_index_backup
            if hasattr(self, '_current_time_frames_per_period_backup'):
                draw_plots._time_frames_per_period = self._current_time_frames_per_period_backup
                # Aktualisiere auch Settings
                if hasattr(self.settings, 'fem_time_frames_per_period'):
                    self.settings.fem_time_frames_per_period = self._current_time_frames_per_period_backup

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
