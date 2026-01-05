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
        
        # #region agent log
        try:
            import json
            import time
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E",
                    "location": "UiFile.py:151",
                    "message": "FILE LOADED - clearing current state",
                    "data": {
                        "geometry_version_before": getattr(self.settings, "geometry_version", 0) if hasattr(self, "settings") else 0
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        self.settings.__dict__.clear()
        self.settings.__init__()
        self.main_window.init_container()
        self.settings.impulse_points.clear()
        self.container.clear_impulse_points()
        
        # Lösche gespeicherte Speaker-Position-Hashes
        if hasattr(self.main_window, 'calculation_handler'):
            self.main_window.calculation_handler.clear_speaker_position_hashes()
        
        # 🎯 WICHTIG: Entferne alte Speaker-/Surface-Actors und lösche Caches beim Laden einer neuen Datei
        # Alte Daten entfernen, damit neue Daten sauber geladen werden können
        if hasattr(self.main_window, 'draw_plots') and hasattr(self.main_window.draw_plots, 'draw_spl_plotter'):
            plotter = self.main_window.draw_plots.draw_spl_plotter
            if plotter is not None:
                # Setze Overlay-Signaturen zurück, damit alle Overlays (inkl. Lautsprecher) neu geplottet werden
                if hasattr(plotter, '_last_overlay_signatures'):
                    plotter._last_overlay_signatures = {}
                
                # 🎯 WICHTIG: Setze auch _last_axis_state zurück, damit Achsenlinien beim Laden immer neu gezeichnet werden
                # (konsistent mit dem Verhalten beim normalen Erstellen einer Fläche und beim Initialisieren)
                if hasattr(plotter, 'overlay_axis') and hasattr(plotter.overlay_axis, '_last_axis_state'):
                    plotter.overlay_axis._last_axis_state = None
                
                # 🎯 SPL-PLOT-CACHES LÖSCHEN: Force-Neuaufbau aller Oberflächen-Meshes und Texturen
                # Dies stellt sicher, dass bei einem neuen File keine alten Meshes/Grids wiederverwendet werden.
                if hasattr(plotter, '_surface_actors'):
                    try:
                        # Entferne alle Surface-Actors aus dem Plotter
                        for sid, entry in list(plotter._surface_actors.items()):
                            actor = entry.get("actor") if isinstance(entry, dict) else entry
                            if actor is not None and hasattr(plotter, "plotter") and plotter.plotter is not None:
                                try:
                                    plotter.plotter.remove_actor(actor)
                                except Exception:
                                    pass
                        plotter._surface_actors.clear()
                    except Exception:
                        plotter._surface_actors = {}
                if hasattr(plotter, '_surface_texture_actors'):
                    try:
                        for sid, tex_entry in list(plotter._surface_texture_actors.items()):
                            actor = tex_entry.get("actor") if isinstance(tex_entry, dict) else tex_entry
                            if actor is not None and hasattr(plotter, "plotter") and plotter.plotter is not None:
                                try:
                                    plotter.plotter.remove_actor(actor)
                                except Exception:
                                    pass
                        plotter._surface_texture_actors.clear()
                    except Exception:
                        plotter._surface_texture_actors = {}
                if hasattr(plotter, '_surface_texture_cache'):
                    try:
                        plotter._surface_texture_cache.clear()
                    except Exception:
                        plotter._surface_texture_cache = {}

                # Vertikale SPL-Flächen-Caches und Actors beim Laden ebenfalls löschen,
                # damit keine alten vertical_spl_* Meshes im neuen Projekt bleiben.
                if hasattr(plotter, '_vertical_surface_meshes'):
                    try:
                        for name, actor in list(plotter._vertical_surface_meshes.items()):
                            try:
                                if isinstance(name, str):
                                    # In _clear_vertical_spl_surfaces werden die Actor-Namen verwendet
                                    plotter.plotter.remove_actor(name)
                                elif actor is not None:
                                    plotter.plotter.remove_actor(actor)
                            except Exception:
                                pass
                        plotter._vertical_surface_meshes.clear()
                    except Exception:
                        plotter._vertical_surface_meshes = {}
                
                if hasattr(plotter, 'overlay_speakers'):
                    overlay_speakers = plotter.overlay_speakers
                    if overlay_speakers is not None:
                        # Entferne nur die Speaker-Actors (nicht alle Overlays)
                        overlay_speakers.clear_category('speakers')
                        # Lösche alle Speaker-Caches, damit sie mit neuen Daten neu gefüllt werden
                        overlay_speakers._speaker_actor_cache.clear()
                        overlay_speakers._speaker_geometry_cache.clear()
                        overlay_speakers._speaker_geometry_param_cache.clear()
                        overlay_speakers._array_geometry_cache.clear()
                        overlay_speakers._array_signature_cache.clear()
                        overlay_speakers._stack_geometry_cache.clear()
                        overlay_speakers._stack_signature_cache.clear()
                        overlay_speakers._overlay_array_cache.clear()
        
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
                # 🎯 FIX: Vergebe neue eindeutige Array-IDs beim Laden, beginnend mit 1
                # Dies stellt sicher, dass Array-IDs immer eindeutig sind pro File
                new_array_id = 1
                old_to_new_id_mapping = {}  # Mapping von alter zu neuer ID für mögliche spätere Verwendung
                
                # Sortiere die Arrays nach ihrer ursprünglichen ID, um eine konsistente Reihenfolge zu gewährleisten
                sorted_arrays = sorted(loaded_data['speaker_arrays'].items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999999)
                
                for old_array_id, array_data in sorted_arrays:
                    # Stelle sicher, dass die neue ID eindeutig ist
                    # Prüfe sowohl im Dictionary als auch mit get_all_speaker_array_ids()
                    while new_array_id in self.settings.speaker_arrays or new_array_id in self.settings.get_all_speaker_array_ids():
                        new_array_id += 1
                    
                    # Speichere das Mapping von alter zu neuer ID
                    old_to_new_id_mapping[old_array_id] = new_array_id
                    
                    # #region agent log
                    try:
                        import json
                        import time as time_module
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "ARRAY_ID_REMAP_START",
                                "location": "UiFile.py:_load_data:before_remap",
                                "message": "Before remapping array ID",
                                "data": {
                                    "old_array_id": str(old_array_id),
                                    "old_array_id_type": type(old_array_id).__name__,
                                    "new_array_id": int(new_array_id),
                                    "existing_array_ids": list(self.settings.speaker_arrays.keys()),
                                    "existing_array_ids_from_method": list(self.settings.get_all_speaker_array_ids())
                                },
                                "timestamp": int(time_module.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # Erstelle das Array mit der neuen eindeutigen ID
                    speaker_array = SpeakerArray(new_array_id)
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
                    self.settings.speaker_arrays[new_array_id] = speaker_array
                    
                    # #region agent log
                    try:
                        import json
                        import time as time_module
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "ARRAY_ID_REMAP",
                                "location": "UiFile.py:_load_data:remap_array_id",
                                "message": "Remapping array ID on file load",
                                "data": {
                                    "old_array_id": str(old_array_id),
                                    "new_array_id": int(new_array_id),
                                    "array_name": str(getattr(speaker_array, 'name', 'Unknown'))
                                },
                                "timestamp": int(time_module.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    new_array_id += 1
                
                # 🎯 FIX: Speichere das Mapping für die spätere Verwendung beim Laden der speaker_array_groups
                # Das Mapping wird später verwendet, wenn speaker_array_groups geladen werden
                if not hasattr(self, '_array_id_mapping'):
                    self._array_id_mapping = {}
                self._array_id_mapping.update(old_to_new_id_mapping)
                
                # 🎯 FIX: Berechne Speaker-Positionen für alle geladenen Arrays
                # Dies ist notwendig, da speaker_position_calculator nicht mehr in update_speaker_array_calculations aufgerufen wird
                if hasattr(self.main_window, 'speaker_position_calculator'):
                    for array_id, speaker_array in self.settings.speaker_arrays.items():
                        try:
                            self.main_window.speaker_position_calculator(speaker_array)
                        except Exception as e:
                            # Fehler beim Berechnen einer Position sollte das Laden nicht verhindern
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Fehler beim Berechnen der Speaker-Positionen für Array {array_id}: {e}")
            
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
            # 🎯 FIX: Aktualisiere die Array-IDs in den Gruppen mit dem Mapping
            speaker_array_groups_loaded = False
            array_id_mapping = getattr(self, '_array_id_mapping', {})
            
            if 'speaker_array_groups' in loaded_data:
                groups_data = loaded_data['speaker_array_groups']
                if groups_data is not None and isinstance(groups_data, dict):
                    # Aktualisiere child_array_ids mit den neuen IDs
                    updated_groups_data = {}
                    for group_id, group_data in groups_data.items():
                        new_group_data = group_data.copy() if isinstance(group_data, dict) else group_data
                        if isinstance(new_group_data, dict) and 'child_array_ids' in new_group_data and isinstance(new_group_data['child_array_ids'], list):
                            new_group_data['child_array_ids'] = [
                                array_id_mapping.get(old_id, old_id)
                                for old_id in new_group_data['child_array_ids']
                                if old_id in array_id_mapping or old_id in self.settings.get_all_speaker_array_ids()
                            ]
                        updated_groups_data[group_id] = new_group_data
                    self.settings.speaker_array_groups = updated_groups_data
                    speaker_array_groups_loaded = True
            elif 'settings' in loaded_data and 'speaker_array_groups' in loaded_data['settings']:
                groups_data = loaded_data['settings']['speaker_array_groups']
                if groups_data is not None and isinstance(groups_data, dict):
                    # Aktualisiere child_array_ids mit den neuen IDs
                    updated_groups_data = {}
                    for group_id, group_data in groups_data.items():
                        new_group_data = group_data.copy() if isinstance(group_data, dict) else group_data
                        if isinstance(new_group_data, dict) and 'child_array_ids' in new_group_data and isinstance(new_group_data['child_array_ids'], list):
                            new_group_data['child_array_ids'] = [
                                array_id_mapping.get(old_id, old_id)
                                for old_id in new_group_data['child_array_ids']
                                if old_id in array_id_mapping or old_id in self.settings.get_all_speaker_array_ids()
                            ]
                        updated_groups_data[group_id] = new_group_data
                    self.settings.speaker_array_groups = updated_groups_data
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
            
            # 🎯 WICHTIG: Aktiviere das Sources-DockWidget (wechsle von Surface zu Source Ansicht)
            if hasattr(sources_instance, 'sources_dockWidget') and sources_instance.sources_dockWidget:
                sources_instance.sources_dockWidget.raise_()
            
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

            # 🎯 ENTFERNT: Vorzeitige Auswahl hier entfernt, da sie möglicherweise eine Gruppe auswählt
            # Die finale Auswahl erfolgt nach load_groups_structure() weiter unten

            sources_tree.setUpdatesEnabled(updates_enabled)
            del tree_blocker
            
            # Lade Gruppen-Struktur nach dem Laden der Arrays
            if hasattr(sources_instance, 'load_groups_structure'):
                sources_instance.load_groups_structure()

            self.main_window.blockSignals(False)

            # 🎯 WICHTIG: Aktiviere das Sources-DockWidget erneut (falls es durch load_groups_structure in den Hintergrund geraten ist)
            if hasattr(sources_instance, 'sources_dockWidget') and sources_instance.sources_dockWidget:
                sources_instance.sources_dockWidget.raise_()

            # 🎯 WICHTIG: Wähle immer das erste Array nach dem Laden aus
            # (auch nach load_groups_structure, da sich die Struktur ändern kann)
            # Suche das erste Array-Item (nicht Gruppe) im Tree, egal ob Top-Level oder in einer Gruppe
            def find_first_array_item():
                """Findet das erste Array-Item im Tree (nicht Gruppe)"""
                # #region agent log
                try:
                    import json
                    import time as time_module
                    summary = {
                        "top_level_count": int(sources_tree.topLevelItemCount()),
                        "items": [],
                    }
                    for i in range(sources_tree.topLevelItemCount()):
                        item = sources_tree.topLevelItem(i)
                        if item is None:
                            continue
                        item_type_dbg = item.data(0, Qt.UserRole + 1)
                        array_id_dbg = item.data(0, Qt.UserRole)
                        summary["items"].append({
                            "index": i,
                            "type": str(item_type_dbg),
                            "array_id": int(array_id_dbg) if isinstance(array_id_dbg, int) else str(array_id_dbg),
                            "child_count": int(item.childCount()),
                        })
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "SRC_SELECT_H1",
                            "location": "UiFile.py:_update_ui_after_load:find_first_array_item",
                            "message": "Inspecting top-level items before selecting first array",
                            "data": summary,
                            "timestamp": int(time_module.time() * 1000),
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                # Durchsuche Top-Level-Items
                for i in range(sources_tree.topLevelItemCount()):
                    item = sources_tree.topLevelItem(i)
                    item_type = item.data(0, Qt.UserRole + 1)
                    array_id = item.data(0, Qt.UserRole)
                    
                    # Wenn es ein Array ist (nicht Gruppe), gib es zurück
                    if item_type != "group" and array_id is not None:
                        return item
                    
                    # Wenn es eine Gruppe ist, durchsuche ihre Children
                    if item_type == "group":
                        for j in range(item.childCount()):
                            child = item.child(j)
                            child_array_id = child.data(0, Qt.UserRole)
                            if child_array_id is not None:
                                return child
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "SRC_SELECT_H2",
                            "location": "UiFile.py:_update_ui_after_load:find_first_array_item",
                            "message": "No array item found in sources_tree",
                            "data": {
                                "top_level_count": int(sources_tree.topLevelItemCount()),
                            },
                            "timestamp": int(time_module.time() * 1000),
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                return None
            
            first_array_item = find_first_array_item()
            if first_array_item:
                with QSignalBlocker(sources_tree):
                    # 🎯 WICHTIG: Lösche alle vorherigen Auswahlen, damit nur ein Item ausgewählt wird
                    sources_tree.clearSelection()
                    sources_tree.setCurrentItem(first_array_item)
                    # Stelle sicher, dass nur dieses Item ausgewählt ist (nicht mehrere)
                    sources_tree.setItemSelected(first_array_item, True)
                    # Stelle sicher, dass Parent-Gruppe expandiert ist
                    parent = first_array_item.parent()
                    if parent:
                        parent.setExpanded(True)
                    # 🎯 WICHTIG: Setze Fokus auf Tree, damit Selektion visuell sichtbar ist
                    sources_tree.setFocus()
                # 🎯 WICHTIG: Aktualisiere Source-UI mit Daten des ausgewählten Arrays
                # (muss nach setCurrentItem aufgerufen werden, damit alle Eingabefelder gefüllt werden)
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "SRC_REFRESH_LOAD",
                            "location": "UiFile.py:_update_ui_after_load:before_refresh_active_selection",
                            "message": "About to call refresh_active_selection after file load",
                            "data": {
                                "has_first_array_item": True,
                                "selected_array_id": int(first_array_item.data(0, Qt.UserRole)) if first_array_item else None,
                                "selected_text": str(first_array_item.text(0)) if first_array_item else None,
                            },
                            "timestamp": int(time_module.time() * 1000),
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                sources_instance.refresh_active_selection()
                # #region agent log
                try:
                    import json
                    import time as time_module
                    selected_item_after = sources_tree.currentItem()
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "SRC_REFRESH_LOAD",
                            "location": "UiFile.py:_update_ui_after_load:after_refresh_active_selection",
                            "message": "refresh_active_selection completed",
                            "data": {
                                "has_selected_item_after": selected_item_after is not None,
                                "selected_array_id_after": int(selected_item_after.data(0, Qt.UserRole)) if selected_item_after else None,
                            },
                            "timestamp": int(time_module.time() * 1000),
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                # #region agent log
                try:
                    import json
                    import time as time_module
                    selected_item = sources_tree.currentItem()
                    if selected_item is not None:
                        sel_type = selected_item.data(0, Qt.UserRole + 1)
                        sel_array_id = selected_item.data(0, Qt.UserRole)
                        sel_text = selected_item.text(0)
                    else:
                        sel_type = None
                        sel_array_id = None
                        sel_text = None
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "SRC_SELECT_H3",
                            "location": "UiFile.py:_update_ui_after_load:after_setCurrentItem",
                            "message": "State after attempting to select first array item",
                            "data": {
                                "has_first_array_item": True,
                                "selected_type": str(sel_type),
                                "selected_array_id": int(sel_array_id) if isinstance(sel_array_id, int) else str(sel_array_id),
                                "selected_text": str(sel_text) if sel_text is not None else None,
                            },
                            "timestamp": int(time_module.time() * 1000),
                        }) + "\n")
                except Exception:
                    pass
                # #endregion

            # 🎯 WICHTIG: Aktualisiere Overlays nach dem Laden, um die neuen Lautsprecher zu plotten
            # (Die Caches wurden bereits in _clear_current_state() gelöscht)
            if hasattr(self.main_window, 'draw_plots') and hasattr(self.main_window.draw_plots, 'draw_spl_plotter'):
                plotter = self.main_window.draw_plots.draw_spl_plotter
                if plotter is not None and hasattr(plotter, 'update_overlays'):
                    try:
                        plotter.update_overlays(self.settings, self.container)
                    except Exception as e:
                        print(f"[UiFile] Fehler beim Aktualisieren der Overlays nach dem Laden: {e}")

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

