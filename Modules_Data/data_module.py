# Die Klasse Datacontainer dient zum Austausch der gesamten Daten zur ausführung des Skripts
import numpy as np
import os
import sys


class DataContainer:
    def __init__(self):
        self.data = {}
        self.calculation_spl = {}
        self.calculation_axes = {}
        self.calculation_windowing = {}
        self.calculation_impulse = {}
        self.impulse_points = {} 
        self.calculation_polar = {}

        self._fixed_frequencies = None
        self._balloon_cache = {}  # Enthält FERTIG konvertierte Daten (neue Struktur + 360°)
        self._speaker_name_mapping = {}  # Mapping: UI-Name -> Dateiname
        self._structure_validated = False
        self._polardata_loaded = False


# =============================================================================
# ---- load
# =============================================================================
    def load_polardata(self, force: bool = False):
        """🚀 OPTIMIERT: Lädt nur noch NPZ-Dateien mit fester Frequenzstruktur"""

        if self._polardata_loaded and not force and self.data.get('speaker_names'):
            return True

        try:
            all_cabinet_data = []
            all_speaker_names = []
            all_metadata = []
            all_balloon_data = []
            
            # Bestimme Pfad für optimierte NPZ-Dateien
            if hasattr(sys, '_MEIPASS'):
                # Im PyInstaller-Build
                bundle_path = os.path.join(sys._MEIPASS, 'Module_LFO', 'Modules_Data')
            else:
                bundle_path = os.path.dirname(os.path.abspath(__file__))
            
            # Nur noch NPZ-Format laden
            polar_mag_dir = os.path.join(bundle_path, 'PolarData_mag')
            
            if os.path.exists(polar_mag_dir):
                # 🚀 NUR NOCH NPZ-DATEIEN!
                all_files = sorted([f for f in os.listdir(polar_mag_dir) 
                                if f.endswith('.npz')])
                
                for file in all_files:
                    base_name = os.path.splitext(file)[0]
                    file_path = os.path.join(polar_mag_dir, file)
                    
                    # 🚀 REIN OPTIMIERT: Nur noch NPZ mit fester Struktur
                    with np.load(file_path, allow_pickle=True) as data:
                        
                        # Lade Metadata
                        if 'metadata' in data:
                            metadata = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
                            if 'data_content' in metadata:
                                fft_data_key = list(metadata['data_content'].keys())[1]
                            all_metadata.append(metadata)
                        else:
                            all_metadata.append(None)
                        
                        # 🚀 OPTIMIERT: Lade und konvertiere Cabinet-Daten vollständig
                        cabinet = data.get('cabinet_data', None)
                        
                        # Konvertiere NumPy-Strukturen zu nativen Python-Dicts/Listen
                        if cabinet is not None:
                            if isinstance(cabinet, np.ndarray):
                                if cabinet.dtype == np.dtype('O'):
                                    if cabinet.size == 1:
                                        # Einzelnes Element → Extrahiere es
                                        cabinet = cabinet.item()
                                    else:
                                        # Mehrere Elemente → Konvertiere zu Liste von Dicts
                                        cabinet_list = []
                                        for item in cabinet:
                                            if isinstance(item, dict):
                                                cabinet_list.append(item)
                                            elif isinstance(item, np.ndarray):
                                                cabinet_list.append(item.item())
                                        cabinet = cabinet_list if len(cabinet_list) > 1 else (cabinet_list[0] if cabinet_list else None)
                            elif isinstance(cabinet, list):
                                # Liste ist bereits das richtige Format
                                pass
                            
                            # Validierung: Erlaube dict ODER Liste von dicts
                            if cabinet is not None:
                                if not isinstance(cabinet, (dict, list)):
                                    print(f"⚠️ Warnung: Cabinet-Daten für {base_name} haben unerwarteten Typ: {type(cabinet)}")
                                    cabinet = None
                        
                        all_cabinet_data.append(cabinet)
                        
                        # 🚀 OPTIMIERT: Lade und validiere Balloon-Daten
                        # Prüfe ZUERST auf neue flache Struktur (balloon_data_meridians, etc.)
                        if 'balloon_data_meridians' in data:
                            # ✅ NEUE FLACHE STRUKTUR: Rekonstruiere Dictionary
                            balloon_data_raw = {
                                'meridians': data['balloon_data_meridians'],
                                'horizontal_angles': data['balloon_data_horizontal_angles'],
                                'frequencies': data['balloon_data_frequencies'],
                                'magnitude': data['balloon_data_magnitude'],
                                'phase': data['balloon_data_phase']
                            }
                            
                            # 🔄 KONVERTIERE: 3D-Kugel (180° Meridiane) → 360° Kreise
                            if len(balloon_data_raw['horizontal_angles']) == 181:
                                balloon_data = self._expand_sphere_to_360_circles(balloon_data_raw)
                                if balloon_data is None:
                                    balloon_data = balloon_data_raw  # Fallback
                            else:
                                # Bereits 360° oder andere Struktur
                                balloon_data = balloon_data_raw
                                balloon_data['vertical_angles'] = balloon_data['meridians']
                                balloon_data['freqs'] = balloon_data['frequencies']
                        else:
                            # 🔄 FALLBACK: Alte verschachtelte Struktur
                            balloon_data = data.get('balloon_data', None)
                            
                            # 🚀 FIX: Handle alte Export-Struktur (Dictionary direkt oder als Array)
                            if isinstance(balloon_data, dict):
                                # Prüfe ob neue oder alte Struktur
                                if 'meridians' in balloon_data and 'horizontal_angles' in balloon_data:
                                    # NEUE STRUKTUR (verschachtelt)
                                    # 🔄 KONVERTIERE: 3D-Kugel (180° Meridiane) → 360° Kreise
                                    if len(balloon_data['horizontal_angles']) == 181:
                                        balloon_data_expanded = self._expand_sphere_to_360_circles(balloon_data)
                                        if balloon_data_expanded is not None:
                                            balloon_data = balloon_data_expanded
                                        else:
                                            balloon_data['vertical_angles'] = balloon_data['meridians']
                                            balloon_data['freqs'] = balloon_data['frequencies']
                                    else:
                                        # Bereits 360° oder andere Struktur
                                        balloon_data['vertical_angles'] = balloon_data['meridians']
                                        balloon_data['freqs'] = balloon_data['frequencies']
                                
                                # Füge Kompatibilitäts-Keys hinzu falls noch nicht vorhanden
                                if 'vertical_angles' not in balloon_data and 'meridians' in balloon_data:
                                    balloon_data['vertical_angles'] = balloon_data['meridians']
                                if 'freqs' not in balloon_data and 'frequencies' in balloon_data:
                                    balloon_data['freqs'] = balloon_data['frequencies']
                                
                            elif isinstance(balloon_data, np.ndarray) and balloon_data.size == 1:
                                # Array-verpackte Struktur (np.savez Artefakt)
                                balloon_data = balloon_data.item()
                                
                                # Prüfe welche Struktur im Array war
                                if isinstance(balloon_data, dict):
                                    if 'meridians' in balloon_data and 'horizontal_angles' in balloon_data:
                                        # 🔄 KONVERTIERE: 3D-Kugel (180° Meridiane) → 360° Kreise
                                        if len(balloon_data['horizontal_angles']) == 181:
                                            balloon_data = self._expand_sphere_to_360_circles(balloon_data)
                                            if balloon_data is None:
                                                # Fallback: Füge nur Keys hinzu
                                                balloon_data = balloon_data
                                                balloon_data['vertical_angles'] = balloon_data['meridians']
                                                balloon_data['freqs'] = balloon_data['frequencies']
                                        else:
                                            # Bereits 360° oder andere Struktur
                                            balloon_data['vertical_angles'] = balloon_data['meridians']
                                            balloon_data['freqs'] = balloon_data['frequencies']
                                    
                                    # Füge Kompatibilitäts-Keys hinzu falls noch nicht vorhanden
                                    if 'vertical_angles' not in balloon_data and 'meridians' in balloon_data:
                                        balloon_data['vertical_angles'] = balloon_data['meridians']
                                    if 'freqs' not in balloon_data and 'frequencies' in balloon_data:
                                        balloon_data['freqs'] = balloon_data['frequencies']
                            else:
                                balloon_data = None
                        
                        # 🚀 FINALE VALIDIERUNG & KONVERTIERUNG beim Laden
                        if balloon_data is not None:
                            # Prüfe ob alte Dict-Struktur → Konvertiere zu neuer NumPy-Struktur
                            if self._is_old_structure(balloon_data):
                                balloon_data = self._convert_old_to_new_structure(balloon_data)
                                
                                # Nach Konvertierung prüfen ob 181→360° nötig
                                if balloon_data and 'horizontal_angles' in balloon_data and len(balloon_data['horizontal_angles']) == 181:
                                    balloon_data = self._expand_sphere_to_360_circles(balloon_data)
                            
                            # Stelle sicher dass Kompatibilitäts-Keys vorhanden sind
                            if balloon_data and isinstance(balloon_data, dict):
                                if 'vertical_angles' not in balloon_data and 'meridians' in balloon_data:
                                    balloon_data['vertical_angles'] = balloon_data['meridians']
                                if 'freqs' not in balloon_data and 'frequencies' in balloon_data:
                                    balloon_data['freqs'] = balloon_data['frequencies']
                            
                            # Speichere FERTIG konvertierte Daten
                            self._balloon_cache[base_name] = balloon_data
                        
                        # Alle Daten (optimiert + standard) werden normal gespeichert für Fallback
                        all_balloon_data.append(balloon_data)
                    
                    all_speaker_names.append(base_name)
                            
                # Kombiniere alle Daten
                if all_speaker_names:
                    # Speichere in der Datenstruktur (ohne Terzbänder)
                    self.data['cabinet_data'] = all_cabinet_data  # Speichere Cabinet-Daten
                    self.data['speaker_names'] = all_speaker_names
                    self.data['metadata'] = all_metadata
                    
                    # 🚀 MAPPING: Erstelle Mapping für verschiedene Namenskonventionen
                    for speaker_name in all_speaker_names:
                        # Hauptmapping: speaker_name -> speaker_name (Dateiname)
                        self._speaker_name_mapping[speaker_name] = speaker_name
                        
                        # Zusätzliche Mappings für verschiedene Formate
                        # z.B. falls UI andere Namen nutzt
                        variations = [
                            speaker_name.replace('_', ' '),  # "1_KS28_H" -> "1 KS28 H"
                            speaker_name.replace('_', '-'),  # "1_KS28_H" -> "1-KS28-H"
                            speaker_name.lower(),            # "1_KS28_H" -> "1_ks28_h"
                            speaker_name.upper(),            # "1_KS28_H" -> "1_KS28_H"
                        ]
                        for variant in variations:
                            if variant not in self._speaker_name_mapping:
                                self._speaker_name_mapping[variant] = speaker_name
                    
                    self._polardata_loaded = True
                    return True
                else:
                    return False
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def reset_runtime_state(self):
        """
        Setzt berechnete Zustände zurück, behält jedoch geladene Polardaten bei.
        """
        self.calculation_spl.clear()
        self.calculation_axes.clear()
        self.calculation_windowing.clear()
        self.calculation_impulse.clear()
        self.impulse_points.clear()
        self.calculation_polar.clear()

    def _is_old_structure(self, balloon_data):
        """
        Erkennt alte Dict-Struktur vs. neue NumPy-Struktur.
        
        ALTE Struktur:
        {
            "0": {"0": {'freq': [...], 'magnitude': [...], 'phase': [...]}, ...},
            "180": {...}
        }
        
        NEUE Struktur:
        {
            'meridians': np.array([...]),
            'horizontal_angles': np.array([...]),
            'frequencies': np.array([...]),
            'magnitude': np.array([...]),
            'phase': np.array([...])
        }
        
        Returns:
            bool: True wenn alte Struktur, False wenn neue Struktur
        """
        if not isinstance(balloon_data, dict):
            return False
        
        # NEUE Struktur hat 'meridians' + 'horizontal_angles'
        if 'meridians' in balloon_data and 'horizontal_angles' in balloon_data:
            return False
        
        # NEUE Struktur kann auch 'vertical_angles' haben (für Rückwärtskompatibilität)
        if 'vertical_angles' in balloon_data and isinstance(balloon_data.get('magnitude'), np.ndarray):
            # Prüfe ob magnitude 3D-Array ist (neue Struktur)
            if balloon_data['magnitude'].ndim == 3:
                return False
        
        # ALTE Struktur hat numerische String-Keys
        if len(balloon_data) > 0:
            first_key = next(iter(balloon_data.keys()), None)
            if first_key and isinstance(first_key, str):
                # Prüfe ob Key eine Zahl ist (wie "0", "180")
                if first_key.replace('.', '').replace('-', '').isdigit():
                    return True
        
        return False
    
    def _convert_old_to_new_structure(self, old_balloon):
        """
        Konvertiert alte Dict-Struktur → neue NumPy-Struktur.
        
        ALTE Struktur:
        {
            "0": {"0": {'freq': [...], 'magnitude': [...], 'phase': [...]}, ...},
            "180": {...}
        }
        
        NEUE Struktur:
        {
            'meridians': np.array([0, 180]),
            'horizontal_angles': np.array([0, 1, ..., 180]),
            'frequencies': np.array([...]),
            'magnitude': np.array([N_mer, N_horz, N_freq]),
            'phase': np.array([N_mer, N_horz, N_freq])
        }
        
        Returns:
            dict: Neue NumPy-Struktur
        """
        try:
            # Sammle alle Meridiane und Winkel
            meridians = sorted([int(float(k)) for k in old_balloon.keys()])
            
            all_horizontal = set()
            for meridian_dict in old_balloon.values():
                for angle_str in meridian_dict.keys():
                    all_horizontal.add(int(float(angle_str)))
            horizontal_angles = sorted(all_horizontal)
            
            # Hole Frequenzen
            first_mer = str(meridians[0])
            first_hor = str(horizontal_angles[0])
            frequencies = np.array(old_balloon[first_mer][first_hor]['freq'])
            
            # Erstelle NumPy-Arrays
            N_mer = len(meridians)
            N_horz = len(horizontal_angles)
            N_freq = len(frequencies)
            
            magnitude = np.zeros((N_mer, N_horz, N_freq))
            phase = np.zeros((N_mer, N_horz, N_freq))
            
            # Fülle Arrays
            for mer_idx, mer_angle in enumerate(meridians):
                mer_str = str(mer_angle)
                for horz_idx, horz_angle in enumerate(horizontal_angles):
                    horz_str = str(horz_angle)
                    if horz_str in old_balloon[mer_str]:
                        magnitude[mer_idx, horz_idx, :] = old_balloon[mer_str][horz_str]['magnitude']
                        phase[mer_idx, horz_idx, :] = old_balloon[mer_str][horz_str]['phase']
                    else:
                        magnitude[mer_idx, horz_idx, :] = np.nan
                        phase[mer_idx, horz_idx, :] = np.nan
            
            # Erstelle neue Struktur mit Kompatibilitäts-Keys
            converted = {
                'meridians': np.array(meridians, dtype=int),
                'horizontal_angles': np.array(horizontal_angles, dtype=int),
                'frequencies': frequencies,
                'magnitude': magnitude,
                'phase': phase
            }
            
            # 🔧 KOMPATIBILITÄT: Füge alte Keys hinzu für bestehende Module
            converted['vertical_angles'] = converted['meridians']      # Alias für meridians
            converted['freqs'] = converted['frequencies']              # Alias für frequencies
            
            return converted
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def _expand_sphere_to_360_circles(self, new_balloon):
        """
        Konvertiert NEUE 3D-Kugel-Struktur → ALTE 360° Kreis-Struktur.
        
        NEUE Struktur (3D-Kugel):
        {
            'meridians': [0, 5, ..., 355],          # 72 Meridiane (Halbkreise um Y-Achse)
            'horizontal_angles': [0, 1, ..., 180],  # 181 Winkel pro Meridian (Halbkreis)
            'magnitude': [72, 181, freq]
        }
        
        ALTE Struktur (360° Kreise):
        {
            'vertical_angles': [0, 5, ..., 355],  # 72 Kreise (= meridians)
            'magnitude': [72, 360, freq],         # 360° pro Kreis (voller Kreis!)
            'freqs': [...]
        }
        
        TRANSFORMATION:
        Für jeden Meridian (wird zu vertical_angle):
        - horizontal 0-180°: Von diesem Meridian (Halbkreis vorne)
        - horizontal 181-359°: Von gegenüberliegendem Meridian +180° (Halbkreis hinten)
        
        Beispiel Meridian 0°:
        - horizontal 0-180°: Daten von Meridian 0°, horizontal 0-180°
        - horizontal 181-359°: Daten von Meridian 180°, horizontal 179-1° (rückwärts)
        
        Returns:
            dict: Alte Struktur mit 360° horizontal_angles
        """
        try:
            meridians = new_balloon['meridians']
            horizontal_angles = new_balloon['horizontal_angles']  # [0, 1, ..., 180]
            frequencies = new_balloon['frequencies']
            mag_sphere = new_balloon['magnitude']  # [N_mer, 181, N_freq]
            phase_sphere = new_balloon['phase']
            
            N_mer = len(meridians)
            N_freq = len(frequencies)
            
            # Erstelle neue Arrays mit 360° horizontal
            mag_360 = np.zeros((N_mer, 360, N_freq))
            phase_360 = np.zeros((N_mer, 360, N_freq))
            
            # Für jeden Meridian
            for mer_idx, meridian in enumerate(meridians):
                # Finde gegenüberliegenden Meridian (+180°)
                opposite_meridian = (meridian + 180) % 360
                opposite_idx = np.where(meridians == opposite_meridian)[0]
                
                if len(opposite_idx) == 0:
                    # Verwende nur diesen Meridian (gespiegelt)
                    mag_360[mer_idx, 0:181, :] = mag_sphere[mer_idx, :, :]
                    mag_360[mer_idx, 181:360, :] = mag_sphere[mer_idx, 179:0:-1, :]  # Rückwärts gespiegelt
                    phase_360[mer_idx, 0:181, :] = phase_sphere[mer_idx, :, :]
                    phase_360[mer_idx, 181:360, :] = phase_sphere[mer_idx, 179:0:-1, :]
                else:
                    opposite_idx = opposite_idx[0]
                    
                    # horizontal 0-180°: Von diesem Meridian
                    mag_360[mer_idx, 0:181, :] = mag_sphere[mer_idx, :, :]
                    phase_360[mer_idx, 0:181, :] = phase_sphere[mer_idx, :, :]
                    
                    # horizontal 181-359°: Von gegenüberliegendem Meridian (rückwärts)
                    # horizontal 181° → opposite horizontal 179°
                    # horizontal 270° → opposite horizontal 90°
                    # horizontal 359° → opposite horizontal 1°
                    for h_new in range(181, 360):
                        h_opposite = 360 - h_new  # Spiegelung
                        mag_360[mer_idx, h_new, :] = mag_sphere[opposite_idx, h_opposite, :]
                        phase_360[mer_idx, h_new, :] = phase_sphere[opposite_idx, h_opposite, :]
            
            # Erstelle alte Struktur mit 360° horizontal
            converted = {
                'vertical_angles': meridians,
                'horizontal_angles': np.arange(0, 360, dtype=int),  # Neu: Explizit 0-359°
                'frequencies': frequencies,
                'magnitude': mag_360,
                'phase': phase_360,
                # Neue Keys für Kompatibilität
                'meridians': meridians,
                'freqs': frequencies
            }
            
            return converted
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def load_calculation_axes(self, calculation_axes): #  ueberschreien des gesamten containers self.calculation_Xaxis
       self.calculation_axes = calculation_axes


# =============================================================================
# ---- set
# =============================================================================

    def set_calculation_SPL(self, calculation_spl):
        self.calculation_spl = calculation_spl

 
    def set_calculation_axes(self, calculation_axes):
        neue_daten = calculation_axes.get("aktuelle_simulation", {})
    
        # Holen des aktuellen Werts oder leeres Dictionary, falls nicht vorhanden
        aktuelle_simulation = self.calculation_axes.get("aktuelle_simulation", {})
    
        #berprfen und Aktualisieren der Strings
        if "y_data_xaxis" in neue_daten:
            aktuelle_simulation['y_data_xaxis'] = neue_daten['y_data_xaxis']
        if "x_data_xaxis" in neue_daten:
            aktuelle_simulation["x_data_xaxis"] = neue_daten["x_data_xaxis"]
        if "y_data_yaxis" in neue_daten:
            aktuelle_simulation["y_data_yaxis"] = neue_daten["y_data_yaxis"]
        if "x_data_yaxis" in neue_daten:
            aktuelle_simulation["x_data_yaxis"] = neue_daten["x_data_yaxis"]
        if "show_in_plot" in neue_daten:
            aktuelle_simulation["show_in_plot"] = neue_daten["show_in_plot"]
        if "color" in neue_daten:
            aktuelle_simulation["color"] = neue_daten["color"]
    
        # Aktualisierte Daten in self.calculation_axes speichern
        self.calculation_axes["aktuelle_simulation"] = aktuelle_simulation

       
    def set_calculation_windowing(self, calculation_windowing):
        self.calculation_windowing = calculation_windowing

    def set_calculation_impulse(self, calculation, key="aktuelle_simulation"):
        """
        Speichert die Impuls-Berechnungsergebnisse für einen bestimmten Key.
        
        Args:
            calculation (dict): Die berechneten Impulsdaten
            key (str, optional): Der Schlüssel unter dem die Daten gespeichert werden. 
                               Standardmäßig "aktuelle_simulation"
        """
        self.calculation_impulse[key] = calculation

    def delete_calculation_impulse(self, key):
        if key in self.calculation_impulse:
            del self.calculation_impulse[key]
        else:
            print(f"No calculation impulse found with key: {key}")

    def set_measurement_point(self, key, data):
        self.impulse_points[key] = data


    def set_polar_data(self, polar_data):
        """
        Setzt die Polar-Berechnungsdaten
        """
        if isinstance(polar_data, dict):
            self.calculation_polar.update(polar_data)

    def update_key(self, old_key, new_key):
        if old_key in self.calculation_impulse:
            self.calculation_impulse[new_key] = self.calculation_impulse.pop(old_key)
        if old_key in self.data:
            self.data[new_key] = self.data.pop(old_key)
        if old_key in self.impulse_points:
            self.impulse_points[new_key] = self.impulse_points.pop(old_key)
        

# =============================================================================
# ---- get
# =============================================================================

    def get_speaker_names(self):
        """
        Gibt die Liste der Lautsprechernamen zurück
        """
        return self.data.get('speaker_names', [])

    def get_data(self):
        return self.data

    def get_calculation_axes(self):
        return self.calculation_axes

    def get_calculation_windowing(self):
        return self.calculation_windowing

    def get_calculation_SPL(self):
        return self.calculation_spl

    def clear_impulse_points(self):
        self.impulse_points.clear()
        self.calculation_impulse.clear()

    def get_polar_data(self):
        """
        Gibt die kompletten Polar-Berechnungsdaten zurück
        """
        return self.calculation_polar
        
    def get_calculation_impulse(self):
        return self.calculation_impulse
   
    def get_balloon_data(self, speaker_name, use_averaged=False, force_new_structure=True):
        """
        🚀 OPTIMIERT: Simpler Cache-Zugriff (Konvertierung erfolgt bereits beim Laden).
        
        Args:
            speaker_name (str): Name des Lautsprechers
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten
            force_new_structure (bool): DEPRECATED - wird nicht mehr gebraucht (immer konvertiert)
        
        Returns:
            dict: Balloon-Daten (bereits in neuer NumPy-Struktur mit 360° horizontal)
        """
        # 🚀 MAPPING: Konvertiere UI-Name zu Dateiname
        actual_name = self._speaker_name_mapping.get(speaker_name, speaker_name)
        
        if use_averaged:
            # 🎯 BANDGEMITTELT: Verwende bandgemittelte Daten
            avg_mag_key = f'avg_mag_{actual_name}'
            avg_phase_key = f'avg_phase_{actual_name}'
            
            if avg_mag_key in self._balloon_cache and avg_phase_key in self._balloon_cache:
                # Kombiniere Magnitude und Phase zu einem Dictionary
                return {
                    'magnitude': self._balloon_cache[avg_mag_key]['magnitude'],
                    'phase': self._balloon_cache[avg_phase_key]['phase'],
                    'vertical_angles': self._balloon_cache[avg_mag_key]['vertical_angles'],
                    'horizontal_angles': self._balloon_cache[avg_mag_key]['horizontal_angles']
                }
        
        # 🚀 SIMPLER LOOKUP: Daten sind bereits beim Laden konvertiert!
        return self._balloon_cache.get(actual_name, None)
  