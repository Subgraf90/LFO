from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, 
                           QTreeWidgetItem, QPushButton, QMenu, QMessageBox, QTextEdit, QFileDialog)
import os
import shutil
from Module_LFO.Modules_Ui.UiPolarDataGenerator import TransferFunctionViewer
import numpy as np

class UiManageSpeaker(QMainWindow):
    def __init__(self, parent=None, settings=None, container=None, main_window=None):
        super().__init__(parent)
        self.setWindowTitle("Manage speaker data")
        self.settings = settings
        self.container = container
        self.main_window = main_window
        self.transfer_function_viewer = None
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Button Layout oben
        button_layout = QHBoxLayout()
        
        # Import Button (ganz links)
        self.import_button = QPushButton("Import")
        self.import_button.setFixedWidth(80)
        self.import_button.clicked.connect(self.import_npz_files)
        
        # Export Button (zwischen Import und Generate)
        self.export_button = QPushButton("Export")
        self.export_button.setFixedWidth(80)
        self.export_button.clicked.connect(self.export_selected_files)
        
        # Generate Button (links ausgerichtet)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setFixedWidth(80)
        self.generate_button.clicked.connect(self.open_transfer_function_viewer)
        
        # Refresh Button (rechts ausgerichtet)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(80)
        self.refresh_button.clicked.connect(self.refresh_polar_data)
        
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.generate_button)
        button_layout.addStretch()  # Schiebt den Refresh Button nach rechts
        button_layout.addWidget(self.refresh_button)
        
        main_layout.addLayout(button_layout)
        
        # Mittleres Layout für Tree und Info
        middle_layout = QHBoxLayout()
        
        # TreeWidget für Lautsprecher-Liste
        self.speaker_tree = QTreeWidget()
        self.speaker_tree.setHeaderLabels(["File name"])
        self.speaker_tree.setColumnWidth(0, 300)
        self.speaker_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Mehrfachauswahl
        self.speaker_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.speaker_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.speaker_tree.itemClicked.connect(self.on_tree_item_clicked)  # Neuer Connector
        middle_layout.addWidget(self.speaker_tree)
        
        # Info Panel mit dunklem Hintergrund und hellem Text
        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setFixedWidth(300)
        self.info_panel.setStyleSheet("""
            QTextEdit { 
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 12px;
                padding: 10px;
                border: none;
            }
        """)
        middle_layout.addWidget(self.info_panel)
        
        main_layout.addLayout(middle_layout)
        
        self.resize(800, 600)
        self.load_polar_data(trigger_updates=False)

    def refresh_polar_data(self):
        """Aktualisiert die Polardaten und erzwingt das Neuladen im DataContainer."""
        self.load_polar_data(force_reload=True)
        
    def load_polar_data(self, trigger_updates=True, force_reload=False):
        """Lädt alle numpy-Dateien aus dem Polardata_mag Ordner"""
        self.speaker_tree.clear()
        
        # Pfad zum Polardata Ordner
        polar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'Modules_Data', 'Polardata_mag')
        
        try:
            # Liste für sortierte Items
            items = []
            for file in os.listdir(polar_path):
                if file.endswith(('.npy', '.npz')):  # Hier wurde .npz hinzugefügt
                    item_name = os.path.splitext(file)[0]
                    item = QTreeWidgetItem([item_name])
                    # Speichere vollständigen Dateipfad als Daten
                    item.setData(0, QtCore.Qt.UserRole, os.path.join(polar_path, file))
                    items.append(item)
            
            # Sortiere Items alphabetisch
            items.sort(key=lambda x: x.text(0).lower())
            
            # Füge sortierte Items zum Tree hinzu
            self.speaker_tree.addTopLevelItems(items)
            self.speaker_tree.expandAll()
            
            # Aktualisiere die Polardaten im Container
            self.container.load_polardata(force=force_reload)

            if trigger_updates:
                selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
                speakerspecs_instance = self.main_window.sources_instance.get_speakerspecs_instance(selected_speaker_array_id)

                if speakerspecs_instance:
                    self.main_window.sources_instance.update_input_fields(speakerspecs_instance)
                    self.main_window.update_speaker_array_calculations()
                    self.main_window.sources_instance.update_widgets(speakerspecs_instance)
                
                # Trigger Frequenzbänder-Mittelung über die bestehende Funktionalität
                if hasattr(self.main_window, 'update_freq_bandwidth'):
                    self.main_window.update_freq_bandwidth()
                elif hasattr(self.main_window, 'plot_spl'):
                    self.main_window.plot_spl()
        
        except Exception as e:
            print(f"Fehler beim Laden der Polar-Daten: {str(e)}")
    
    def show_context_menu(self, position):
        """Zeigt das Kontextmenü an der Mausposition"""
        selected_items = self.speaker_tree.selectedItems()
        
        if selected_items:
            context_menu = QMenu(self)
            
            # Einheitlicher Menütext
            delete_action = context_menu.addAction("Delete speaker file")
            
            action = context_menu.exec_(self.speaker_tree.viewport().mapToGlobal(position))
            
            if action == delete_action:
                self.delete_speaker_data(selected_items)
    
    def delete_speaker_data(self, items):
        """Löscht die ausgewählten Speakerdateien"""
        # Einheitliche Bestätigungsnachricht
        message = 'Are you sure you want to delete speaker file?'
        
        reply = QMessageBox.question(self, 'Delete speaker file',
                                   message,
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            deleted_count = 0
            error_count = 0
            
            for item in items:
                file_path = item.data(0, QtCore.Qt.UserRole)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error deleting {item.text(0)}: {str(e)}")
            
            # Entferne alle Items aus dem Tree
            for item in items:
                self.speaker_tree.takeTopLevelItem(
                    self.speaker_tree.indexOfTopLevelItem(item))
            
            # Zeige einfache Ergebnis-Meldung
            if error_count > 0:
                QMessageBox.warning(self, "Delete Result", "Some files could not be deleted.")
            else:
                QMessageBox.information(self, "Success", "Files deleted successfully.")
    
            # Aktualisiere die Polardaten im Container
            self.container.load_polardata(force=True)

            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            speakerspecs_instance = self.main_window.sources_instance.get_speakerspecs_instance(selected_speaker_array_id)

            if speakerspecs_instance:
                self.main_window.sources_instance.update_input_fields(speakerspecs_instance)
                self.main_window.update_speaker_array_calculations()
                self.main_window.sources_instance.update_widgets(speakerspecs_instance)

    def on_tree_item_clicked(self, item, column):
        # Hole alle ausgewählten Items
        selected_items = self.speaker_tree.selectedItems()
        
        # Wenn nur ein Item ausgewählt ist, zeige dessen Metadaten
        if len(selected_items) == 1:
            file_path = item.data(0, QtCore.Qt.UserRole)
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if 'metadata' in data:
                        metadata = data['metadata'].item()
                        
                        # Definiere die gewünschte Reihenfolge der Metadaten
                        ordered_keys = [
                            'meridians',
                            'meridian_measurements',
                            'interpolated_meridians',
                            'interpolated_horizontal_angles',
                            'data_source',
                            'fs',
                            'freq_range_min',
                            'freq_range_max',
                            'filter_enabled',
                            'ir_window_enabled',
                            'ir_window_length',
                            'reference_distance',
                            'reference_freq',
                            'spl_normalized',
                            'spl_value',
                            'target_spl',
                            'time_offset'
                        ]
                        
                        # Einfaches Textformat für Info-Panel erstellen
                        info_text = ""
                        
                        # Füge nur die spezifizierten Metadaten in der gewünschten Reihenfolge hinzu
                        for key in ordered_keys:
                            if key in metadata:
                                value = metadata[key]
                                # Formatiere den Wert je nach Typ
                                if value is None:
                                    formatted_value = "-"
                                elif isinstance(value, float):
                                    formatted_value = f"{value:.2f}"
                                elif isinstance(value, bool):
                                    formatted_value = "✓" if value else "✗"
                                else:
                                    formatted_value = str(value)
                                
                                # Füge Schlüssel und Wert hinzu
                                info_text += f"{key}: {formatted_value}\n"
                        
                        # Füge Cabinet-Daten hinzu, falls vorhanden
                        if 'cabinet_data' in data:
                            cabinet_data = data['cabinet_data']
                            if len(cabinet_data) > 0:
                                info_text += "\nCabinet Data:\n"
                                
                                for i, cabinet in enumerate(cabinet_data):
                                    info_text += f"\nSpeaker {i+1}:\n"
                                    info_text += f"  width: {cabinet.get('width', 0):.2f} m\n"
                                    info_text += f"  depth: {cabinet.get('depth', 0):.2f} m\n"
                                    
                                    # Ersetze height durch front_height und back_height
                                    front_height = cabinet.get('front_height', cabinet.get('height', 0))
                                    back_height = cabinet.get('back_height', cabinet.get('height', 0))
                                    info_text += f"  front_height: {front_height:.2f} m\n"
                                    info_text += f"  back_height: {back_height:.2f} m\n"
                                    
                                    info_text += f"  configuration: {cabinet.get('configuration', 'stack')}\n"
                                    info_text += f"  stack_layout: {cabinet.get('stack_layout', 'beside')}\n"
                                    info_text += f"  angle_point: {cabinet.get('angle_point', 'None')}\n"
                                    info_text += f"  cardioid: {'Yes' if cabinet.get('cardio', False) else 'No'}\n"
                                    info_text += f"  x_offset: {cabinet.get('x_offset', 0):.2f} m\n"
                                    
                                    # Zeige Winkel an, falls vorhanden
                                    if 'angles' in cabinet and cabinet['angles']:
                                        info_text += "  angles: "
                                        angles_text = ", ".join([f"{angle:.1f}°" for angle in cabinet['angles']])
                                        info_text += angles_text + "\n"
                        
                        # Setze den Text im Info-Panel
                        self.info_panel.setPlainText(info_text)
            
            except Exception as e:
                error_msg = f"Error loading metadata: {str(e)}"
                self.info_panel.setPlainText(error_msg)
                print(f"\nDEBUG: {error_msg}")
                import traceback
                traceback.print_exc()  # Füge Stacktrace hinzu für bessere Fehlerdiagnose
        
        # Wenn mehrere Items ausgewählt sind, zeige nur diesen Text
        elif len(selected_items) > 1:
            self.info_panel.setPlainText("more than 1 speaker selected")
        
        # Wenn kein Item ausgewählt ist
        else:
            self.info_panel.setPlainText("No item selected")
        
     
    def open_transfer_function_viewer(self):
        if not self.transfer_function_viewer:
            self.transfer_function_viewer = TransferFunctionViewer(
                parent=None,
                manage_speaker_window=self,  # Referenz auf UiManageSpeaker
                settings=self.settings,
                container=self.container
            )
        self.transfer_function_viewer.show()


    def closeEvent(self, event):
        """Schließt auch den Transfer Function Viewer wenn vorhanden"""
        if hasattr(self, 'transfer_function_viewer') and self.transfer_function_viewer:
            self.transfer_function_viewer.close()
        event.accept()
    
    def import_npz_files(self):
        """Importiert ausgewählte NPZ-Dateien und kopiert sie in den PolarData_mag Ordner"""
        try:
            # NPZ-Dateien auswählen (mehrere Dateien möglich)
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, 
                "choose files",
                os.path.expanduser("~"),  # Startet im Home-Verzeichnis
                "NPZ files (*.npz);;All files (*)"
            )
            
            if not file_paths:  # Benutzer hat abgebrochen
                return
            
            # Ziel-Ordner (PolarData_mag in der LFO-Struktur)
            target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'Modules_Data', 'Polardata_mag')
            
            # Stelle sicher, dass der Ziel-Ordner existiert
            os.makedirs(target_path, exist_ok=True)
            
            # Zähle importierte Dateien
            imported_count = 0
            overwritten_count = 0
            
            # Verarbeite jede ausgewählte Datei
            for source_file in file_paths:
                if source_file.endswith('.npz'):
                    filename = os.path.basename(source_file)
                    target_file = os.path.join(target_path, filename)
                    
                    # Prüfe ob Datei bereits existiert
                    if os.path.exists(target_file):
                        overwritten_count += 1
                    else:
                        imported_count += 1
                    
                    # Kopiere die Datei (überschreibt falls vorhanden)
                    shutil.copy2(source_file, target_file)
            
            # Zeige Erfolgsmeldung
            message = f"Import completed!\n\n"
            if imported_count > 0:
                message += f"New files imported: {imported_count}\n"
            if overwritten_count > 0:
                message += f"Files overwritten: {overwritten_count}\n"
            
            if imported_count == 0 and overwritten_count == 0:
                message += "No NPZ files selected."
            
            QMessageBox.information(self, "Import successful", message)
            
            # Aktualisiere die Datenstruktur (Refresh)
            self.load_polar_data(force_reload=True)
            
            # Trigger Frequenzbänder-Mittelung über die bestehende Funktionalität
            if hasattr(self.main_window, 'update_freq_bandwidth'):
                self.main_window.update_freq_bandwidth()
            elif hasattr(self.main_window, 'plot_spl'):
                self.main_window.plot_spl()
            
        except Exception as e:
            error_msg = f"Error importing: {str(e)}"
            QMessageBox.critical(self, "Import error", error_msg)
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
    
    def export_selected_files(self):
        """Exportiert ausgewählte Dateien aus der LFO-Datenstruktur an einen definierten Speicherort"""
        try:
            # Hole ausgewählte Items
            selected_items = self.speaker_tree.selectedItems()
            
            if not selected_items:
                QMessageBox.warning(self, "Export", "Please select files to export.")
                return
            
            # Wähle Zielordner aus
            target_folder = QFileDialog.getExistingDirectory(
                self,
                "Select export folder",
                os.path.expanduser("~"),  # Startet im Home-Verzeichnis
                QFileDialog.ShowDirsOnly
            )
            
            if not target_folder:  # Benutzer hat abgebrochen
                return
            
            # Zähle exportierte Dateien
            exported_count = 0
            error_count = 0
            
            # Exportiere jede ausgewählte Datei
            for item in selected_items:
                source_file_path = item.data(0, QtCore.Qt.UserRole)
                filename = os.path.basename(source_file_path)
                target_file_path = os.path.join(target_folder, filename)
                
                try:
                    # Kopiere die Datei
                    shutil.copy2(source_file_path, target_file_path)
                    exported_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error exporting {filename}: {str(e)}")
            
            # Zeige Ergebnis-Meldung
            if error_count > 0:
                QMessageBox.warning(self, "Export Result", "Some files could not be exported.")
            else:
                QMessageBox.information(self, "Success", "Files exported successfully.")
                
        except Exception as e:
            error_msg = f"Error during export: {str(e)}"
            QMessageBox.critical(self, "Export error", error_msg)
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()