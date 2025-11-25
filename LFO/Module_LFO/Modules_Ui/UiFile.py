import pickle
import numpy as np
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLabel
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSignalBlocker

from Module_LFO.Modules_Data.data_module import DataContainer
from Module_LFO.Modules_Data.settings_state import SpeakerArray
from Module_LFO.Modules_Ui.UiSettings import UiSettings

class UiFile:
    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.data = container.data
        self.save_file_path = None
        # self.ui_settings = UiSettings(self, self.settings)


    def new_pa_file(self):
        """Erstellt ein neues leeres Projekt"""
        try:
            # Container und Settings zurücksetzen
            self.main_window.init_container()
            self.settings.load_speaker_defaults()
            self.settings.impulse_points.clear()
            self.container.clear_impulse_points()
            
            # Initialisiere speaker_array_groups
            if not hasattr(self.settings, 'speaker_array_groups'):
                self.settings.speaker_array_groups = {}
            else:
                self.settings.speaker_array_groups.clear()
            
            # Lösche gespeicherte Speaker-Position-Hashes
            if hasattr(self.main_window, 'calculation_handler'):
                self.main_window.calculation_handler.clear_speaker_position_hashes()
            
            # Schließe vorhandene Widgets
            if hasattr(self.main_window, 'sources_instance') and self.main_window.sources_instance:
                if hasattr(self.main_window.sources_instance, 'sources_dockWidget'):
                    self.main_window.sources_instance.sources_dockWidget.close()
                    self.main_window.sources_instance.sources_dockWidget.deleteLater()
                # Leere speakerspecs_instance Liste
                if hasattr(self.main_window.sources_instance, 'speakerspecs_instance'):
                    self.main_window.sources_instance.speakerspecs_instance.clear()
            
            # Schließe andere Widgets
            if hasattr(self.main_window, 'impulse_manager'):
                self.main_window.impulse_manager.close_impulse_input_dock_widget()
            
            if hasattr(self.main_window, 'snapshot_engine'):
                self.main_window.snapshot_engine.close_capture_dock_widget()
            
            # Aktualisiere UI mit Standardwerten
            self.main_window.ui_settings.update_ui_from_settings()
            
            # Sources UI NEU initialisieren und öffnen
            from Module_LFO.Modules_Ui.UiSourceManagement import Sources
            self.main_window.sources_instance = Sources(self.main_window, self.settings, self.container)
            self.main_window.sources_instance.show_sources_dock_widget()
            
            # Lösche Save-Pfad (neues Projekt hat keinen Speicherort)
            self.save_file_path = None
            
            # Plots neu initialisieren und auf Draufsicht setzen
            self.main_window.init_plot(preserve_camera=False, view="top")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self.main_window,
                "Fehler",
                f"Fehler beim Erstellen eines neuen Projekts: {str(e)}"
            )

    def overwrite_file(self):
        if not self.save_file_path:
            self.save_file_as()
        
        if self.save_file_path:
            self._save_data(self.save_file_path)

    def save_file_as(self):
        self.save_file_path, _ = QFileDialog.getSaveFileName(self.main_window, 'Save Pickle File', '', 'Pickle Files (*.pickle)')
        if self.save_file_path:
            self._save_data(self.save_file_path)

    def _save_data(self, file_path):
        try:
            # Speichere Gruppen-Struktur vor dem Speichern
            if hasattr(self.main_window, 'sources_instance') and self.main_window.sources_instance:
                if hasattr(self.main_window.sources_instance, 'save_groups_structure'):
                    self.main_window.sources_instance.save_groups_structure()
            
            file_data = {
                'settings': {
                    key: getattr(self.settings, key) 
                    for key in self.settings.__dict__.keys() 
                    if not callable(getattr(self.settings, key)) and not key.startswith("__")
                },
                'speaker_arrays': {
                    id: array.__dict__ if hasattr(array, '__dict__') else array 
                    for id, array in self.settings.speaker_arrays.items()
                },
                'calculated_data_axes': self.container.calculation_axes,
                'widget_state': self._get_widget_state(),
                'impulse_points': self.settings.impulse_points,
                # Speichere Snapshots (alle außer "aktuelle_simulation")
                'snapshots': {
                    key: value for key, value in self.container.calculation_axes.items()
                    if key != "aktuelle_simulation"
                },
                # Speichere speaker_array_groups
                'speaker_array_groups': getattr(self.settings, 'speaker_array_groups', {})
            }
            self._check_picklable(file_data)
            
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(file_data, pickle_file, protocol=4)
                
        except Exception as e:
            QMessageBox.critical(self.main_window, 
                               "Fehler", 
                               f"Die Datei konnte nicht gespeichert werden: {str(e)}")

    def _check_picklable(self, data):
        for key, value in data.items():
            try:
                pickle.dumps(value)
            except (pickle.PicklingError, Exception) as e:
                print(f"Error pickling {key}: {e}")

    def _get_widget_state(self):
        return {
            'speaker_arrays': {id: array.__dict__ if hasattr(array, '__dict__') else array for id, array in self.settings.speaker_arrays.items()}
        }

    def load_file(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, 'Open Pickle File', '', 'Pickle Files (*.pickle)')
        if file_path:
            self._clear_current_state()
            self._load_data(file_path)
            self._update_ui_after_load()
        

    def _clear_current_state(self):
        
        self.settings.__dict__.clear()
        self.settings.__init__()
        self.main_window.init_container()
        self.settings.impulse_points.clear()
        self.container.clear_impulse_points()
        
        # Lösche gespeicherte Speaker-Position-Hashes
        if hasattr(self.main_window, 'calculation_handler'):
            self.main_window.calculation_handler.clear_speaker_position_hashes()
        
        self._close_widgets()

    def _safe_close(self, widget):
        if widget and hasattr(widget, 'close'):
            widget.close()

    def _close_widgets(self):

        if hasattr(self.main_window, 'sources_instance'):
            self._safe_close(getattr(self.main_window.sources_instance, 'sources_dockWidget', None))
            delattr(self.main_window, 'sources_instance')  # Entferne die Instanz komplett

        if hasattr(self.main_window, 'impulse_input_dock_widget'):
            self._safe_close(getattr(self.main_window, 'impulse_input_dock_widget', None))
            delattr(self.main_window, 'impulse_input_dock_widget')

        if hasattr(self.main_window, 'capture_dock_widget'):
            self._safe_close(getattr(self.main_window, 'capture_dock_widget', None))
            delattr(self.main_window, 'capture_dock_widget')
        
        if hasattr(self.main_window, 'impulse_manager'):
            self.main_window.impulse_manager.close_impulse_input_dock_widget()

        if hasattr(self.main_window, 'snapshot_engine'):
            self.main_window.snapshot_engine.close_capture_dock_widget()

    def _load_with_module_mapping(self, file_path):
        """
        Lädt Pickle-File mit automatischem Module_Mihilab -> Module_LFO Mapping
        """
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Mapping für Module_Mihilab -> Module_LFO
                if module.startswith('Module_Mihilab'):
                    new_module = module.replace('Module_Mihilab', 'Module_LFO')
                    try:
                        return super().find_class(new_module, name)
                    except (ImportError, AttributeError) as e:
                        # Wenn Klasse nicht gefunden, versuche Standard-Verhalten
                        pass
                
                # Standard-Verhalten für andere Module
                return super().find_class(module, name)
        
        with open(file_path, 'rb') as pickle_file:
            return CustomUnpickler(pickle_file).load()
    
    def _load_data(self, file_path):
        
        try:
            loaded_data = None
            load_error = None
            
            # Versuche ZUERST natives LFO Format (Standard)
            try:
                with open(file_path, 'rb') as pickle_file:
                    loaded_data = pickle.load(pickle_file)
                
            except (ModuleNotFoundError, AttributeError) as e:
                # LFO Format nicht gefunden - versuche MihiLab Format
                load_error = str(e)
                
                try:
                    loaded_data = self._load_with_module_mapping(file_path)
                    
                except Exception as e2:
                    # Beide Formate gescheitert
                    load_error = f"LFO: {load_error}\nMihiLab: {str(e2)}"
                    raise Exception(f"Unbekanntes Pickle-Format!\n\nDas File konnte weder als LFO noch als MihiLab Format geladen werden.\n\nDetails:\n{load_error}")
            
            # Lade Messpunkte
            if 'impulse_points' in loaded_data:
                self.settings.impulse_points = loaded_data['impulse_points']
            
            # Lade Snapshots
            if 'snapshots' in loaded_data:
                for key, value in loaded_data['snapshots'].items():
                    self.container.calculation_axes[key] = value
            
            # Lade Speaker Arrays
            if 'speaker_arrays' in loaded_data and isinstance(loaded_data['speaker_arrays'], dict):
                self.settings.speaker_arrays = {}
                for array_id, array_data in loaded_data['speaker_arrays'].items():
                    speaker_array = SpeakerArray(array_id)
                    for key, value in array_data.items():
                        if key == 'source_polar_pattern':
                            pattern = []
                            for speaker in value:
                                if speaker in self.data['speaker_names']:
                                    pattern.append(speaker)
                                else:
                                    if '4_KS28_CSA_CL_mag' in self.data['speaker_names']:
                                        pattern.append('4_KS28_CSA_CL_mag')
                                    else:
                                        pattern.append(self.data['speaker_names'][0])
                            setattr(speaker_array, key, np.array(pattern))
                        elif isinstance(value, np.ndarray):
                            setattr(speaker_array, key, value.copy())
                        else:
                            setattr(speaker_array, key, value)
                    try:
                        getattr(speaker_array, 'source_azimuth', None)
                    except Exception:
                        pass
                    self.settings.speaker_arrays[array_id] = speaker_array
            
            # Lade Settings
            if 'settings' in loaded_data:
                for k, v in loaded_data['settings'].items():
                    # Frequenzen auf ganze Zahlen runden
                    if k in ['upper_calculate_frequency', 'lower_calculate_frequency', 'fem_calculate_frequency']:
                        v = int(round(float(v)))
                    # Überspringe speaker_array_groups hier - wird separat behandelt
                    if k == 'speaker_array_groups':
                        continue
                    setattr(self.settings, k, v)  # explizit ins Objekt schreiben
            
            # Lade speaker_array_groups (falls vorhanden) - prüfe sowohl in loaded_data als auch in settings
            speaker_array_groups_loaded = False
            if 'speaker_array_groups' in loaded_data:
                groups_data = loaded_data['speaker_array_groups']
                if groups_data is not None and isinstance(groups_data, dict):
                    self.settings.speaker_array_groups = groups_data
                    speaker_array_groups_loaded = True
            elif 'settings' in loaded_data and 'speaker_array_groups' in loaded_data['settings']:
                groups_data = loaded_data['settings']['speaker_array_groups']
                if groups_data is not None and isinstance(groups_data, dict):
                    self.settings.speaker_array_groups = groups_data
                    speaker_array_groups_loaded = True
            
            # Initialisiere speaker_array_groups, falls nicht vorhanden oder None
            if not speaker_array_groups_loaded or not hasattr(self.settings, 'speaker_array_groups') or self.settings.speaker_array_groups is None:
                self.settings.speaker_array_groups = {}

        except Exception as e:
            QMessageBox.critical(
                self.main_window, 
                "Fehler", 
                f"Die Datei konnte nicht geladen werden: {str(e)}"
            )
        

    def _update_ui_after_load(self):
        try:
            self.main_window.blockSignals(True)
            self.main_window.ui_settings.update_ui_from_settings()
            self.main_window.show_sources_dock_widget()

            # Stelle sicher, dass der Surface Manager existiert und sein TreeWidget aktualisiert wird
            if hasattr(self.main_window, "_ensure_surface_manager"):
                surface_manager = self.main_window._ensure_surface_manager()
                if surface_manager:
                    surface_manager.load_surfaces()
                    surface_manager.show_surfaces_tab()

            if not hasattr(self.main_window, 'sources_instance') or self.main_window.sources_instance is None:
                self.main_window.blockSignals(False)
                return None

            sources_instance = self.main_window.sources_instance
            sources_tree = sources_instance.sources_tree_widget

            if sources_tree is None:
                self.main_window.blockSignals(False)
                return None

            tree_blocker = QSignalBlocker(sources_tree)
            updates_enabled = sources_tree.updatesEnabled()
            sources_tree.setUpdatesEnabled(False)

            root = sources_tree.invisibleRootItem()

            for i in range(root.childCount()):
                item = root.child(i)
                array_id = item.data(0, Qt.UserRole)
                array_name = item.text(0)

                if array_id is not None:
                    speaker_array = self.settings.get_speaker_array(array_id)
                    speakerspecs_instance = sources_instance.get_speakerspecs_instance(array_id)
                    if speakerspecs_instance:
                        if not hasattr(speakerspecs_instance, 'scroll_layout'):
                            sources_instance.init_ui(
                                speakerspecs_instance,
                                sources_instance.speaker_tab_layout
                            )

                    if speaker_array and hasattr(speaker_array, 'color'):
                        # Farb-Quadrat in der vierten Spalte (wie beim Snapshot Widget)
                        color_label = QLabel()
                        color_label.setFixedSize(20, 20)
                        color_label.setStyleSheet(f"background-color: {speaker_array.color}; border: 1px solid gray;")
                        sources_tree.setItemWidget(item, 3, color_label)

            selected_item = None
            if sources_tree.topLevelItemCount() > 0:
                selected_item = sources_tree.topLevelItem(0)
                sources_tree.setCurrentItem(selected_item)
            else:
                print("DEBUG: Keine Elemente im TreeWidget")

            sources_tree.setUpdatesEnabled(updates_enabled)
            del tree_blocker
            
            # Lade Gruppen-Struktur nach dem Laden der Arrays
            if hasattr(sources_instance, 'load_groups_structure'):
                sources_instance.load_groups_structure()

            self.main_window.blockSignals(False)

            if selected_item:
                with QSignalBlocker(sources_tree):
                    sources_tree.setCurrentItem(selected_item)
                sources_instance.refresh_active_selection()
            elif sources_tree.topLevelItemCount() > 0:
                first_item = sources_tree.topLevelItem(0)
                with QSignalBlocker(sources_tree):
                    sources_tree.setCurrentItem(first_item)
                sources_instance.refresh_active_selection()

            # Öffne Snapshot Widget (wie beim Init)
            if hasattr(self.main_window, 'snapshot_engine'):
                self.main_window.snapshot_engine.show_snapshot_widget()
            
            # Öffne Impulse Widget wenn Impulse Points vorhanden sind
            if hasattr(self.main_window, 'impulse_manager'):
                if self.settings.impulse_points:
                    self.main_window.impulse_manager.show_impulse_input_dock_widget()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self.main_window,
                "Fehler",
                f"Fehler beim Aktualisieren der UI: {str(e)}"
            )
        finally:
            # Stelle sicher, dass Signale immer wieder aktiviert werden
            self.main_window.blockSignals(False)


    def reset_to_factory(self):
        """Setzt alle Einstellungen auf Werkseinstellungen zurück"""
        try:
            # Lade Factory Defaults
            self.settings.load_custom_defaults()
            
            # Lösche gespeicherte Speaker-Position-Hashes
            if hasattr(self.main_window, 'calculation_handler'):
                self.main_window.calculation_handler.clear_speaker_position_hashes()
            
            # Aktualisiere UI mit Factory Settings
            self.main_window.ui_settings.update_ui_from_settings()
            
            # Sources Dock Widget anzeigen (falls noch nicht offen)
            self.main_window.show_sources_dock_widget()
            
            # Wähle das erste Item aus, falls vorhanden
            if hasattr(self.main_window, 'sources_instance') and self.main_window.sources_instance:
                if hasattr(self.main_window.sources_instance, 'sources_tree_widget'):
                    if self.main_window.sources_instance.sources_tree_widget.topLevelItemCount() > 0:
                        first_item = self.main_window.sources_instance.sources_tree_widget.topLevelItem(0)
                        self.main_window.sources_instance.sources_tree_widget.setCurrentItem(first_item)
            
            # Bandwidth-Berechnung neu ausführen
            self.main_window.update_freq_bandwidth()
            
            # Alle Berechnungen aktualisieren
            self.main_window.update_speaker_array_calculations()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self.main_window,
                "Fehler",
                f"Fehler beim Zurücksetzen der Factory Settings: {str(e)}"
            )

