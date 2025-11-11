
"""
Polar Data Calculator - PFAD 2 (UI-Manipulationen & Datenverarbeitung)
========================================================================

Speicherort: LFO/Module_LFO/Modules_Calculate/PolarDataCalculator.py

Zweck:
------
Verarbeitet Polar-Daten basierend auf UI-Eingaben und ImportCalculator-Referenzdaten.
1. Delegiert Import an PolarDataImportCalculator (WAV + TXT)
2. Wendet UI-Manipulationen auf raw_measurements an (PFAD 2)
3. Erstellt calculated_data, polar_data, balloon_data
4. Interpoliert für 360° Darstellung

Verarbeitungs-Pipeline PFAD 2:
-------------------------------
raw_measurements (PFAD 1) → UI-Manipulationen → calculated_data
  → Time Offset (optional, nur WAV)
  → IR Windowing (optional, nur WAV)
  → FFT → 400 Hz LP → Unwrap
  → HP/LP Filter (optional, UI-gesteuert)
  → 1/96 Oktave
  → SPL-Normalisierung (optional)
  → Polar/Balloon Interpolation

Abhängigkeiten:
---------------
- numpy: Berechnungen
- scipy: Interpolation, Filter
- PyQt5: UI-Interaktion (main_window)
- PolarDataImportCalculator: PFAD 1 Referenz (WAV + TXT)
- self.data: Zentrale Datenstruktur
  - Liest: raw_measurements (von ImportCalc)
  - Schreibt: calculated_data, polar_data, balloon_data, interpolated_data

Phase-Handling:
---------------
- Phase wird bidirektional unwrapped (von 50 Hz aus)
- Unwrapping: 50 Hz → hohe Frequenzen UND 50 Hz → tiefe Frequenzen
- Verhindert Messfehler-Propagation bei tiefen Frequenzen
- Interpolation: Unwrap → Interpolate → WRAP-BACK auf [-180°, +180°]
- Wrap-Back nach JEDER Interpolation verhindert Offsets
- Endergebnis: Alle Phase-Werte konsistent in [-180°, +180°]

Hauptmethoden:
--------------
1. process_impulse_response(filename)
   - Delegiert WAV-Import an PolarDataImportCalculator

2. process_txt_response(filename)
   - Delegiert TXT-Import (Mag/Phase) an PolarDataImportCalculator
   
3. recalculate_from_raw()
   - PFAD 2: Wendet UI-Manipulationen auf raw_measurements an
   - Output: calculated_data
   
4. create_balloon_data()
   - Orchestriert: import_data_folders()
   - Delegiert an PolarDataInterpolateCalculator.create_interpolated_balloon_data()
     → Erstellt balloon_data (NumPy-Struktur)
     → Konvertiert Dict zu NumPy-Arrays
     → Interpoliert auf 1° Schritte (0°-180°)

Wird verwendet von:
-------------------
- UiPolarDataGenerator: UI-gesteuerte Datenverarbeitung
- PolarDataExport: Export von calculated_data/balloon_data

Aufruf-Hierarchie:
------------------
WAV-Import:
  process_impulse_response()
    └─→ PolarDataImportCalculator.process_wav_file()

TXT-Import:
  process_txt_response()
    └─→ PolarDataImportCalculator.process_txt_file()

UI-Änderungen:
  recalculate_from_raw()
    ├─→ PolarDataImportCalculator.compute_reference_from_ir() [PFAD 1, nur WAV]
    ├─→ shift_impulse_response() [optional, nur WAV]
    ├─→ apply_ir_window() [optional, nur WAV]
    ├─→ apply_frequency_filters() [optional]
    └─→ normalize_magnitude() [optional]

Balloon-Daten Erstellung:
  create_balloon_data()
    ├─→ import_data_folders()
    └─→ PolarDataInterpolateCalculator.create_interpolated_balloon_data()
          ├─→ _convert_dict_to_numpy_structure()
          └─→ _interpolate_all_meridians_horizontal_vectorized()
"""

import numpy as np
from PyQt5.QtWidgets import (QMessageBox)
import os
import re
from scipy.io import wavfile
from Module_LFO.Modules_Calculate.PolarDataImportCalculator import Polar_Data_Import_Calculator
from Module_LFO.Modules_Calculate.PolarDataInterpolateCalculator import Polar_Data_Interpolate_Calculator
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

class Polar_Data_Calculator(ModuleBase):
    """
    PFAD 2: UI-Manipulationen & Datenverarbeitung
    Nutzt PFAD 1 (ImportCalc) als Referenz, wendet UI-Manipulationen an
    """
    def __init__(self, main_window, settings, data):
        super().__init__(settings)
        self.data = data
        self.settings = settings
        self.main_window = main_window
        
        # Initialisiere Interpolate Calculator für Balloon-Daten
        self.interpolate_calculator = Polar_Data_Interpolate_Calculator(settings, data)


    # ------ Polar Data ------

    def get_vertical_folders(self):
        """
        Holt die vertikalen Messordner basierend auf der Reihenfolge im TreeWidget.
        
        Returns:
            list: Liste der Ordnernamen (vertikal)
        """
        vertical_folders = []
        root = self.main_window.tree_widget
        
        # Wenn nur ein Ordner vorhanden ist, verwende diesen für horizontale UND vertikale Daten
        if root.topLevelItemCount() == 1:
            item = root.topLevelItem(0)
            vertical_folders.append(item.text(0))
            return vertical_folders
        
        # Bei mehreren Ordnern, verwende alle in der Reihenfolge für die vertikale Interpolation
        for i in range(root.topLevelItemCount()):
            item = root.topLevelItem(i)
            vertical_folders.append(item.text(0))
        
        return vertical_folders


# ----- create Balloon Data -----

    def import_data_folders(self):
        """Importiert und analysiert die Daten aus den TreeWidget-Ordnern"""
        try:            
            # Hole das TreeWidget
            tree_widget = self.main_window.tree_widget
            
            # Prüfe, ob Ordner vorhanden sind
            if tree_widget.topLevelItemCount() == 0:
                return False
            
            # Aktualisiere nur die spezifischen Felder in den Metadaten
            if 'metadata' not in self.data:
                self.data['metadata'] = {}
            
            # Setze nur die spezifischen Felder zurück
            self.data['metadata']['folder_data'] = []
            self.data['metadata']['vertical_angles'] = []
            self.data['metadata']['vertical_count'] = 0
            self.data['metadata']['horizontal_count'] = 0
            
            # Sammle Informationen über alle Ordner
            folder_count = tree_widget.topLevelItemCount()
            
            # Bestimme Meridian-Winkel basierend auf der Anzahl der Ordner
            # Meridian = vertikaler Kreis, vollständige Rotation um X-Achse (Lautsprecherbreite), in Y-Z-Ebene
            # Jeder Ordner hat 180° Polar-Daten (horizontale Drehung von 0° bis 180°)
            # Minimum 2 Ordner: 0° und 180° Meridian ergeben zusammen ein vollständiges 360° Polardiagramm!
            
            # Berechne Meridian-Winkel von 0° bis 360° (exklusive 360°) gleichmäßig verteilt
            # Beispiel: 2 Ordner → 0°, 180° | 4 Ordner → 0°, 90°, 180°, 270°
            step = 360.0 / folder_count
            meridian_angles = [0 + i * step for i in range(folder_count)]
            
            # Speichere Meridian-Winkel in Metadaten
            self.data['metadata']['meridian_angles'] = meridian_angles
            self.data['metadata']['vertical_angles'] = meridian_angles  # Alias für Rückwärtskompatibilität
            self.data['metadata']['vertical_count'] = folder_count
            
            # Sammle Daten für jeden Ordner
            for i in range(folder_count):
                folder_item = tree_widget.topLevelItem(i)
                folder_name = folder_item.text(0)
                meridian_angle = meridian_angles[i]
                                
                # Sammle alle Messungen im Ordner
                measurements = []
                file_count = folder_item.childCount()
                
                # Bestimme horizontale Winkel (Polar) basierend auf der Anzahl der Dateien
                # NEU: Nur noch 0° bis 180° (halbe horizontale Rotation)
                if file_count > 0:
                    if file_count == 1:
                        angle_step = 0
                        polar_angles = [0]
                    else:
                        # Lineare Verteilung von 0° bis 180°
                        angle_step = 180.0 / (file_count - 1)
                        polar_angles = [0 + j * angle_step for j in range(file_count)]
                    
                    for j in range(file_count):
                        file_item = folder_item.child(j)
                        filename = file_item.text(0)
                        
                        # Berechne horizontalen Winkel (Polar) basierend auf Position
                        horizontal_angle = polar_angles[j]
                        
                        # Prüfe, ob Daten für diese Datei vorhanden sind
                        if filename in self.data['calculated_data']:
                            measurements.append({
                                'filename': filename,
                                'index': j,
                                'horizontal_angle': horizontal_angle
                            })
                            
                            # Extrahiere Sampling-Frequenz aus den Daten, falls vorhanden
                            if 'fs' in self.data['calculated_data'][filename] and 'fs' not in self.data['metadata']:
                                self.data['metadata']['fs'] = self.data['calculated_data'][filename]['fs']
                
                # Füge Ordnerinformationen zu Metadaten hinzu
                self.data['metadata']['folder_data'].append({
                    'name': folder_name,
                    'meridian_angle': meridian_angle,
                    'vertical_angle': meridian_angle,  # Alias für Rückwärtskompatibilität
                    'measurements': measurements,
                    'count': len(measurements),
                    'angle_step': angle_step if file_count > 0 else 0
                })
                
                # Aktualisiere maximale Anzahl horizontaler Messungen
                self.data['metadata']['horizontal_count'] = max(
                    self.data['metadata']['horizontal_count'],
                    len(measurements)
                )
            
            # Stelle sicher, dass fs in den Metadaten vorhanden ist
            if 'fs' not in self.data['metadata']:
                # Versuche, fs aus den calculated_data zu extrahieren
                for filename, data in self.data['calculated_data'].items():
                    if 'fs' in data:
                        self.data['metadata']['fs'] = data['fs']
                        break
                
                # Wenn immer noch keine fs gefunden wurde, verwende einen Standardwert
                if 'fs' not in self.data['metadata']:
                    self.data['metadata']['fs'] = 44100  # Standard-Sampling-Frequenz            
            
            return True
            
        except Exception as e:
            return False

    def create_balloon_data(self):
        """
        Erstellt Balloon-Daten aus den berechneten Daten (calculated_data).
        
        ORCHESTRIERT die komplette Pipeline:
        ------------------------------------
        1. Importiere Ordnerstruktur (import_data_folders)
        2. Delegiert ALLES WEITERE an PolarDataInterpolateCalculator:
           - create_interpolated_balloon_data()
             → Erstellt balloon_data aus calculated_data (NumPy-Struktur)
             → _convert_dict_to_numpy_structure() (Dict → NumPy)
             → _interpolate_all_meridians_horizontal_vectorized() (1° Schritte, 0°-180°)
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Prüfe, ob berechnete Daten vorhanden sind
            if not self.data['calculated_data']:
                return False
            
            # 1. Importiere Ordnerstruktur
            if not self.import_data_folders():
                return False
            
            # 2. DELEGATION: Erstelle Balloon-Daten + Interpolation
            #    → PolarDataInterpolateCalculator macht ALLES!
            if not self.interpolate_calculator.create_interpolated_balloon_data():
                return False
            
            return True
            
        except Exception as e:
            return False


# ------ Calculate Measurement Points ------

    def recalculate_from_raw(self):
        """
        HAUPTMETHODE: Berechnet calculated_data neu aus raw_measurements (PFAD 2).
        
        Pipeline PFAD 2:
        ----------------
        raw_measurements → UI-Manipulationen → calculated_data
        
        Für jede Messung:
        1. PFAD 1 (Referenz): Hole bereits berechnete Daten aus raw_measurements
           - original_freq, original_magnitude, original_phase
           - Diese wurden bereits von ImportCalculator beim WAV-Import berechnet
        2. PFAD 2 (Manipuliert):
           a) Time Offset anwenden (wenn aktiviert)
           b) IR Windowing anwenden (wenn aktiviert)
           c) FFT → 400 Hz LP → Unwrap
           d) Optional HP/LP Filter (UI-gesteuert)
           e) 1/96 Oktave Mittelung
        3. SPL-Normalisierung (wenn aktiviert)
        4. Speichere in calculated_data
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """        
        try:
            # Prüfe ob Metadaten vorhanden
            if 'metadata' not in self.data:
                return False
            
            # Prüfe ob raw_measurements vorhanden
            if 'raw_measurements' not in self.data:
                return False
                
            # Hole Parameter für SPL-Normalisierung
            normalize_active = self.data['metadata'].get('spl_normalized', False)
            
            # Hole Time-Offset direkt aus der flachen Struktur
            time_offset = self.data['metadata'].get('time_offset', 0.0)
    
            
            # Hole Sample-Rate und Source-Type aus ersten Raw-Daten
            first_raw_data = next(iter(self.data['raw_measurements'].values()))
            fs = first_raw_data['fs']
            source_type = first_raw_data.get('source_type', 'Impulse Response')  # Default: Impulse Response für alte Daten
            
            # Update Metadaten mit Sample-Rate und Source-Type
            self.data['metadata'].update({
                'fs': fs,  # Speichere Sample-Rate in Hauptmetadaten
                'source_type': source_type,  # WAV: 'Impulse Response', TXT: 'txt'
            })
                        
            # Reset calculated_data
            self.data['calculated_data'] = {}
            
            # Verarbeite Raw-Daten
            processed_count = 0
    
            # Import-Modul für PFAD 1 initialisieren
            import_calc = Polar_Data_Import_Calculator(self.data)
    
            for filename, raw_data in self.data['raw_measurements'].items():
                processed_count += 1
    
                # ═══════════════════════════════════════════════════════════════════
                # PFAD 1: URSPRUNG (Referenz - immer als schwarze Linie im Plot)
                # ═══════════════════════════════════════════════════════════════════
                # Bereits von ImportCalculator berechnet und in raw_measurements gespeichert!
                
                # PFAD 1: Hole bereits berechnete Daten aus raw_measurements
                orig_avg_freq = raw_data['original_freq'].copy()
                orig_avg_magnitude = raw_data['original_magnitude'].copy()
                orig_avg_phase = raw_data['original_phase'].copy()
                
                # ═══════════════════════════════════════════════════════════════════
                # PFAD 2: MANIPULIERT (IR-Manipulationen beeinflussen Freq-Daten)
                # ═══════════════════════════════════════════════════════════════════
                # IA → Time Offset → IR Windowing → FFT → 400 Hz LP → Unwrap → Optional HP/LP → 1/96 Okt
                
                # Prüfe, ob Impulsantwort vorhanden ist (WAV) oder nicht (TXT)
                has_impulse_response = (
                    'original_impulse_response' in raw_data and 
                    raw_data['original_impulse_response'] is not None and 
                    len(raw_data['original_impulse_response']) > 0
                )
                
                # Start mit clean IR (nur bei WAV)
                if has_impulse_response:
                    impulse_response = raw_data['original_impulse_response'].copy()
                    original_impulse_response = raw_data['original_impulse_response'].copy()
                else:
                    # TXT-Datei: Keine Impulsantwort vorhanden
                    impulse_response = np.array([])
                    original_impulse_response = np.array([])
                
                # Time Offset anwenden (wenn aktiviert UND Impulsantwort vorhanden)
                if has_impulse_response and time_offset != 0:
                    shifted_ir, _ = self.shift_impulse_response(impulse_response, time_offset)
                    impulse_response = shifted_ir
                
                # IR Windowing anwenden (wenn aktiviert UND Impulsantwort vorhanden) - BEEINFLUSST FFT!
                ir_window_enabled = self.data['metadata'].get('ir_window_enabled', False)
                if has_impulse_response and ir_window_enabled:
                    windowed_response, window_function = self.apply_ir_window(impulse_response)
                    shifted_response = windowed_response
                elif has_impulse_response:
                    windowed_response = impulse_response.copy()
                    window_function = np.ones(len(impulse_response))
                    shifted_response = impulse_response.copy()
                else:
                    # TXT-Datei: Keine Impulsantworten
                    windowed_response = np.array([])
                    window_function = np.array([])
                    shifted_response = np.array([])
                
                # Prüfe ob IR-Manipulationen aktiv sind (nur relevant bei WAV)
                ir_modified = has_impulse_response and ((time_offset != 0) or ir_window_enabled)
                
                # Wenn IR modifiziert wurde: Neue FFT + Verarbeitung
                if ir_modified:
                    # FFT der manipulierten IR
                    fft_result = np.fft.rfft(shifted_response)
                    freqs = np.fft.rfftfreq(len(shifted_response), d=1/fs)
                    
                    eps = np.finfo(float).eps
                    magnitude = 20 * np.log10(np.abs(fft_result) + eps)
                    phase = np.angle(fft_result)
                    
                    # 400 Hz Tiefpass
                    magnitude_400hz, phase_400hz_rad = import_calc.apply_fixed_400hz_lowpass(
                        freqs, magnitude, phase
                    )
                    
                    # Phase Unwrap
                    ref_freq = 50.0
                    ref_idx = np.argmin(np.abs(freqs - ref_freq))
                    phase_unwrapped = np.zeros_like(phase_400hz_rad, dtype=float)
                    phase_unwrapped[ref_idx] = phase_400hz_rad[ref_idx]
                    
                    if ref_idx < len(phase_400hz_rad) - 1:
                        phase_upper = phase_400hz_rad[ref_idx:]
                        phase_unwrapped[ref_idx:] = np.unwrap(phase_upper)
                    
                    if ref_idx > 0:
                        phase_lower = phase_400hz_rad[:ref_idx+1][::-1]
                        phase_unwrapped[:ref_idx+1] = np.unwrap(phase_lower)[::-1]
                    
                    phase_deg = np.degrees(phase_unwrapped)
                    
                    # Optional HP/LP Filter
                    filter_enabled = self.data['metadata'].get('filter_enabled', False)
                    if filter_enabled:
                        freqs, magnitude_400hz, phase_deg = self.apply_optional_frequency_filters(
                            freqs, magnitude_400hz, phase_deg
                        )
                    
                    # 1/96 Oktave
                    avg_freq, avg_magnitude, avg_phase = import_calc.average_frequency_bands_1_96(
                        freqs, magnitude_400hz, phase_deg
                    )
                else:
                    # Keine IR-Manipulation: Start mit PFAD 1 Ergebnis
                    avg_freq = orig_avg_freq.copy()
                    avg_magnitude = orig_avg_magnitude.copy()
                    avg_phase = orig_avg_phase.copy()
                    
                    # Nur optionale HP/LP Filter (wenn aktiviert)
                    filter_enabled = self.data['metadata'].get('filter_enabled', False)
                    if filter_enabled:
                        avg_freq, avg_magnitude, avg_phase = self.apply_optional_frequency_filters(
                            avg_freq, avg_magnitude, avg_phase
                        )
                
                # Wrapped Phase für Debug
                orig_avg_phase_wrapped = ((orig_avg_phase + 180) % 360) - 180
                
                # Berechne verarbeitete Impulsantwort aus den gefilterten Daten
                processed_impulse, processed_time = import_calc.calculate_processed_impulse_response(
                    avg_freq, avg_magnitude, avg_phase
                )
                
                # Verwende die direkt gefilterte Phase (physikalisch korrekt)
                # Die corrected_phase aus der rekonstruierten IR kann Interpolationsfehler enthalten
                final_phase = avg_phase
                # Verwende direkt gefilterte Phase (avg_phase) für physikalisch korrekte Darstellung
                
                # Speichere Ergebnis (mit expliziten Kopien für Sicherheit)
                result_data = {
                        'freq': avg_freq.copy(),
                        'magnitude': avg_magnitude.copy(),
                        'phase': final_phase.copy(),
                        'original_freq': orig_avg_freq.copy(),  # PFAD 1: Referenz (400 Hz gefiltert)
                        'original_magnitude': orig_avg_magnitude.copy(),  # PFAD 1: Referenz (400 Hz gefiltert)
                        'original_phase': orig_avg_phase.copy(),  # PFAD 1: Referenz (400 Hz gefiltert, unwrapped)
                        'original_phase_wrapped': orig_avg_phase_wrapped.copy(),  # Phase direkt nach FFT (wrapped)
                        'impulse_response': shifted_response.copy(),
                        'original_impulse_response': original_impulse_response.copy(),
                        'windowed_impulse_response': windowed_response.copy(),
                        'window_function': window_function.copy(),
                        'fs': fs
                }
                
                # Füge verarbeitete Impulsantwort hinzu falls verfügbar
                if processed_impulse is not None and processed_time is not None:
                    result_data['processed_impulse_response'] = processed_impulse
                    result_data['processed_impulse_time'] = processed_time
                
                self.data['calculated_data'][filename] = result_data
    
            
            # Normalisierung wenn aktiv
            if normalize_active:
                # Hole das aktuell ausgewählte Item
                current_item = self.main_window.tree_widget.currentItem()
                
                # Bestimme die Referenzdatei
                reference_filename = None
                
                # Wenn ein Item ausgewählt ist
                if current_item:
                    # Wenn das aktuelle Item ein File ist
                    if current_item.parent():
                        reference_filename = current_item.text(0)
                    # Wenn das aktuelle Item ein Ordner ist und Kinder hat
                    elif current_item.childCount() > 0:
                        reference_filename = current_item.child(0).text(0)
                
                # Fallback auf erstes Item, wenn kein gültiges Item ausgewählt
                if not reference_filename:
                    root = self.main_window.tree_widget.topLevelItem(0)
                    if root and root.childCount() > 0:
                        reference_filename = root.child(0).text(0)
                    elif root:
                        reference_filename = root.text(0)
                
                # Hole Referenzfrequenz aus Metadaten
                reference_freq = self.data['metadata'].get('reference_freq')
                
                # Wende Normalisierung an, wenn Referenzdatei gefunden
                if reference_filename and reference_filename in self.data['calculated_data']:
                    ref_data = self.data['calculated_data'][reference_filename]
                    ref_freq = np.array(ref_data['freq'])
                    ref_magnitude = np.array(ref_data['magnitude'])
                    
                    # Verwende Terzbandmittelung statt Einzelfrequenz
                    current_spl_at_ref = self.get_third_octave_spl(
                        ref_freq, ref_magnitude, reference_freq
                    )

                    # Wende Normalisierung auf alle Dateien an
                    for filename in self.data['calculated_data'].keys():
                        data = self.data['calculated_data'][filename]
                        normalized_magnitude = self.normalize_magnitude(
                            data['magnitude'], 
                            current_spl_at_ref, 
                            reference_filename
                        )
                        data['magnitude'] = normalized_magnitude
            
            return True
            
        except Exception as e:
            return False


# ------ Measurement calculation ------


    def apply_optional_frequency_filters(self, freq, magnitude_db, phase):
        """
        Wendet optionale Hoch-/Tiefpassfilter an (UI-gesteuert).
        
        Filter-Eigenschaften:
        ---------------------
        - Typ: FIR (Kaiser-Window, 24 dB/Oktave)
        - Nur aktiv wenn filter_enabled = True
        - Grenzfrequenzen aus UI (filter_freq_min, filter_freq_max)
        - Magnitude wird gefiltert, Phase bleibt unverändert (FIR = linear-phasig)
        
        Args:
            freq: Frequenz-Array
            magnitude_db: Magnitude in dB
            phase: Phase in Grad (unwrapped)
            
        Returns:
            tuple: (freq, magnitude_filtered, phase)
        """
        try:
            # Wende nur Frequenzfilter an (falls aktiviert)
            filtered_magnitude, filtered_phase = self.apply_frequency_filters(
                freq, magnitude_db, phase
            )
            
            # Gebe ALLE Frequenzen zurück - keine Beschneidung mehr
            return freq, filtered_magnitude, filtered_phase
            
        except Exception as e:
            return freq, magnitude_db, phase

    def normalize_magnitude(self, magnitude, reference_spl, reference_filename):
        """
        Normalisiert die Magnitude-Daten (SPL-Normalisierung).
        
        Berechnung:
        -----------
        1. Natürlicher SPL bei Zieldistanz = reference_spl - 20*log10(distance)
        2. Korrektur = spl_value (UI-Input)
        3. Normalisierte Magnitude = magnitude + Korrektur
        
        Args:
            magnitude: Original-Magnitude in dB (bei 1m gemessen)
            reference_spl: Aktueller SPL der Referenzmessung
            reference_filename: Name der Referenzmessung
            
        Returns:
            np.array: Normalisierte Magnitude in dB
        """
        
        # Prüfe ob Normalisierung aktiv ist - mit korrekten Schlüsseln
        normalize_active = self.data['metadata'].get('spl_normalized', False)
        
        # Prüfe auch den alten Schlüssel für Abwärtskompatibilität
        old_normalize = self.data['metadata'].get('normalize', False)
        
        # Wenn keine Normalisierung aktiv ist, gib die Originaldaten zurück
        if not normalize_active and not old_normalize:
            return magnitude
        
        try:
            # Hole Normalisierungsparameter mit korrekten Schlüsseln
            norm_value = self.data['metadata'].get('spl_value', 
                                                  self.data['metadata'].get('norm_value', 0.0))
            distance = self.data['metadata'].get('reference_distance', 1.0)
            target_spl = self.data['metadata'].get('target_spl', 0.0)
            
            # 1. Berechne wie laut es bei der angegebenen Distanz natürlich wäre
            natural_spl_at_distance = reference_spl - 20 * np.log10(distance)
            
            # 2. Berechne den nötigen Offset um auf den Ziel-SPL zu kommen
            total_correction = norm_value
            
            # Wende Korrektur an
            normalized = magnitude + total_correction
            
            return normalized
            
        except Exception as e:
            return magnitude

    def get_third_octave_spl(self, freq_array, magnitude_array, center_freq):
        """
        Berechnet den energetisch gemittelten SPL über ein Terzband.
        
        Ein Terzband (1/3 Oktave) hat folgende Grenzen:
        - Untere Grenze: f_center / 2^(1/6) ≈ f_center / 1.122
        - Obere Grenze: f_center * 2^(1/6) ≈ f_center * 1.122
        
        Vorteile der Terzbandmittelung:
        --------------------------------
        - Glättung von Messschwankungen bei Einzelfrequenzen
        - Reduzierung modaler Effekte (Raumresonanzen)
        - Realistischere Werte für Breitbandsignale
        - Standard in der Akustikmesstechnik (IEC 61260)
        
        Args:
            freq_array: Array mit Frequenzen
            magnitude_array: Array mit SPL-Werten in dB
            center_freq: Mittenfrequenz des Terzbands
            
        Returns:
            float: Energetisch gemittelter SPL in dB
        """
        try:
            # Berechne Terzband-Grenzen (1/3 Oktave)
            # 2^(1/6) ≈ 1.122462
            factor = 2 ** (1/6)
            lower_freq = center_freq / factor
            upper_freq = center_freq * factor
            
            # Erstelle Maske für Frequenzen im Terzband
            freq_mask = (freq_array >= lower_freq) & (freq_array <= upper_freq)
            
            # Hole alle SPL-Werte im Terzband
            magnitudes_in_band = magnitude_array[freq_mask]
            
            if len(magnitudes_in_band) == 0:
                # Fallback: Wenn keine Werte im Band, nimm nächstgelegene Frequenz
                ref_freq_idx = np.abs(freq_array - center_freq).argmin()
                return magnitude_array[ref_freq_idx]
            
            # Energetische Mittelung (RMS)
            # Konvertiere dB zu linearen Druckwerten: p = 10^(SPL/20)
            linear_pressures = 10 ** (magnitudes_in_band / 20)
            
            # RMS über alle Werte
            rms_pressure = np.sqrt(np.mean(linear_pressures ** 2))
            
            # Zurück zu dB
            spl_averaged = 20 * np.log10(rms_pressure)
            
            return spl_averaged
            
        except Exception as e:
            # Fallback auf Einzelfrequenz
            ref_freq_idx = np.abs(freq_array - center_freq).argmin()
            return magnitude_array[ref_freq_idx]

    def process_impulse_response(self, filename):
        """
        Delegiert WAV-Import an PolarDataImportCalculator.
        
        Diese Methode ist nur noch ein Wrapper um das Import-Modul.
        Die gesamte Verarbeitung findet in PolarDataImportCalculator statt.
        
        Args:
            filename: Pfad zur WAV-Datei
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Delegiere kompletten Import an Import-Modul
            import_calc = Polar_Data_Import_Calculator(self.data)
            return import_calc.process_wav_file(filename)
            
        except Exception as e:
            return False

    def process_txt_response(self, filename):
        """
        Delegiert TXT-Import (Mag/Phase) an PolarDataImportCalculator.
        
        Diese Methode ist nur noch ein Wrapper um das Import-Modul.
        Die gesamte Verarbeitung findet in PolarDataImportCalculator statt.
        
        Args:
            filename: Pfad zur TXT-Datei (Format: Freq [Hz] \t Magnitude [dB] \t Phase [deg])
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Delegiere kompletten Import an Import-Modul
            import_calc = Polar_Data_Import_Calculator(self.data)
            result = import_calc.process_txt_file(filename)
            return result
            
        except Exception as e:
            return False

    def replace_phase_data(self, filenames, phase_file):
        """
        Ersetzt Phase-Daten für ausgewählte Messungen anhand einer TXT-Datei.

        Args:
            filenames (list[str]): Ausgewählte Mess-Dateinamen (Keys in raw_measurements)
            phase_file (str): Pfad zur TXT-Datei mit Frequenz-/Phase-Daten

        Returns:
            tuple: (success: bool, message: str, updated_items: list[str])
        """
        try:
            if not filenames:
                return False, "Keine Messungen ausgewählt.", []

            if not os.path.isfile(phase_file):
                return False, "Phase-Datei nicht gefunden.", []

            source_freq, source_phase = self._parse_phase_file(phase_file)
            if source_freq is None or source_phase is None:
                return False, "Phase-Datei konnte nicht gelesen werden.", []

            if len(source_freq) < 2:
                return False, "Phase-Datei enthält zu wenige gültige Datenpunkte.", []

            # Sortieren, Duplikate entfernen und ungültige Werte filtern
            order = np.argsort(source_freq)
            source_freq = source_freq[order]
            source_phase = source_phase[order]

            unique_freq, unique_idx = np.unique(source_freq, return_index=True)
            source_freq = unique_freq
            source_phase = source_phase[unique_idx]

            valid_mask = source_freq > 0
            source_freq = source_freq[valid_mask]
            source_phase = source_phase[valid_mask]

            if len(source_freq) < 2:
                return False, "Phase-Datei enthält keine gültigen Frequenzdaten.", []

            # Phase ent-wrappen, um Sprünge (180°) zu vermeiden
            source_phase_unwrapped = np.degrees(np.unwrap(np.radians(source_phase)))

            updated_items = []
            extrapolated_fill_used = False
            raw_measurements = self.data.get('raw_measurements', {})

            for name in filenames:
                if name not in raw_measurements:
                    continue

                raw_entry = raw_measurements[name]
                target_freq = np.asarray(raw_entry.get('freq'))
                if target_freq is None or len(target_freq) == 0:
                    continue

                existing_phase = np.asarray(raw_entry.get('phase'))
                if existing_phase is None or len(existing_phase) == 0:
                    continue

                new_phase = np.interp(
                    target_freq,
                    source_freq,
                    source_phase_unwrapped,
                    left=source_phase_unwrapped[0],
                    right=source_phase_unwrapped[-1]
                )
                if np.any((target_freq < source_freq[0]) | (target_freq > source_freq[-1])):
                    extrapolated_fill_used = True
                new_phase = np.asarray(new_phase, dtype=float)

                raw_entry['phase'] = new_phase.copy()
                raw_entry['original_phase'] = new_phase.copy()
                raw_entry['original_phase_wrapped'] = ((new_phase + 180.0) % 360.0) - 180.0
                raw_entry['phase_source'] = os.path.basename(phase_file)

                updated_items.append(name)

            if not updated_items:
                return False, "Keine Phase-Daten aktualisiert.", []

            message = f"Phase-Daten für {len(updated_items)} Messung(en) aktualisiert."
            if extrapolated_fill_used:
                message += " Frequenzbereiche außerhalb der Datei wurden mit dem ersten bzw. letzten verfügbaren Phasewert aufgefüllt."

            return True, message, updated_items

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return False, f"Phase-Ersetzung fehlgeschlagen: {exc}", []

    def _parse_phase_file(self, filepath):
        """
        Liest Frequenz-/Phase-Daten aus einer TXT-Datei.

        Unterstützte Formate:
            - Mindestens zwei numerische Spalten (Frequenz, Phase)
            - Optional zusätzliche Spalten (z.B. Magnitude), die ignoriert werden
            - Dezimaltrennzeichen '.' oder ','
            - Trennzeichen Tab, Leerzeichen oder Semikolon
            - Header- oder Kommentarzeilen werden ignoriert

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: Frequenzen und Phasen oder (None, None) bei Fehler
        """
        try:
            freq_values = []
            phase_values = []

            phase_column_index = None
            delimiter_hint = None  # None = auto/regex, '\t', ';', etc., 'whitespace'

            with open(filepath, 'r', encoding='utf-8') as phase_stream:
                for line in phase_stream:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Versuche Header zu erkennen, um die richtige Phasenspalte zu finden
                    if phase_column_index is None and re.search(r'[A-Za-z]', stripped):
                        header_tokens_found = False
                        for delim_candidate in ['\t', ';', ',', ' ']:
                            if delim_candidate == ' ':
                                tokens = [tok for tok in stripped.split() if tok]
                                delim_name = 'whitespace'
                            else:
                                tokens = [tok.strip() for tok in stripped.split(delim_candidate) if tok.strip()]
                                delim_name = delim_candidate

                            for idx, token in enumerate(tokens):
                                if 'phase' in token.lower():
                                    phase_column_index = idx
                                    delimiter_hint = delim_name
                                    header_tokens_found = True
                                    break

                            if header_tokens_found:
                                break
                        # Headerzeilen nicht als Daten weiterverarbeiten
                        continue

                    normalized = stripped.replace(',', '.')

                    if delimiter_hint == 'whitespace' or delimiter_hint is None:
                        parts = re.split(r'[;\s]+', normalized)
                    else:
                        parts = [p.strip() for p in normalized.split(delimiter_hint)]

                    numeric_parts = []
                    for part in parts:
                        if part == '':
                            continue
                        try:
                            numeric_parts.append(float(part))
                        except ValueError:
                            numeric_parts = []
                            break

                    if len(numeric_parts) < 2:
                        continue

                    freq_values.append(numeric_parts[0])
                    if phase_column_index is not None and phase_column_index < len(numeric_parts):
                        phase_values.append(numeric_parts[phase_column_index])
                    else:
                        phase_values.append(numeric_parts[-1])

            if not freq_values or not phase_values:
                return None, None

            return np.asarray(freq_values, dtype=float), np.asarray(phase_values, dtype=float)

        except Exception:
            import traceback
            traceback.print_exc()
            return None, None

    def shift_impulse_response(self, impulse_response, time_ms):
        """
        Verschiebt die Impulsantwort um die angegebene Zeit (Time Offset).
        
        Berechnung:
        -----------
        - Samples = time_ms * fs / 1000
        - Impulsantwort wird um Samples verschoben (np.roll)
        - Phase-Korrektur wird berechnet (für spätere Anwendung)
        
        Args:
            impulse_response: Original-Impulsantwort
            time_ms: Zeit-Offset in Millisekunden
            
        Returns:
            tuple: (shifted_response, phase_correction)
        """

        fs = self.data['metadata']['fs']
        samples = int(time_ms * fs / 1000)  # Konvertiere ms zu Samples
        
        # Verschiebe die Impulsantwort
        shifted_response = np.roll(impulse_response, -samples)
        
        # Berechne neue Phase basierend auf der Verschiebung
        freqs = np.fft.rfftfreq(len(impulse_response), d=1/fs)
        phase_correction = np.exp(-1j * 2 * np.pi * freqs * samples / fs)
        
        return shifted_response, phase_correction

    def apply_frequency_filters(self, freq, magnitude_db, phase):
        """Wendet steile FIR-Filter mit 24 dB/Oktave an
        
        FIR-Filter mit linearer Phase und 24 dB/Oktave Flankensteilheit.
        
        Args:
            freq: Frequenz-Array
            magnitude_db: Magnitude in dB
            phase: Phase in Grad (unwrapped!)
            
        Returns:
            filtered_magnitude: Gefilterte Magnitude
            original_phase: Phase in Grad (unverändert, FIR-Filter sind linear-phasig)
        """
        try:
            from scipy.signal import firwin, freqz, kaiserord, kaiser_beta
            
            # Prüfe, ob Filter aktiviert ist
            filter_enabled = self.data['metadata'].get('filter_enabled', False)
            if not filter_enabled:
                return magnitude_db, phase
            
            # Hole Filter-Parameter
            freq_min = self.data['metadata'].get('filter_freq_min', 10.0)
            freq_max = self.data['metadata'].get('filter_freq_max', 300.0)
            fs = self.data['metadata']['fs']  # Sample-Rate
            
            # Normalisiere Grenzfrequenzen zur Nyquist-Frequenz
            # WICHTIG: Verwende die tatsächliche maximale Frequenz der MESSDATEN!
            # Das freq-Array ist synthetisch bis 22050 Hz, aber Messdaten sind viel kleiner
            
            # Finde die maximale Frequenz der originalen Messdaten
            actual_max_freq = 20000.0  # Default
            
            # Prüfe raw_measurements für echte Messdaten-Frequenzen
            if 'raw_measurements' in self.data and self.data['raw_measurements']:
                for filename, raw_data in self.data['raw_measurements'].items():
                    if 'freq' in raw_data:
                        max_freq = max(raw_data['freq'])
                        # Für Audio-Messungen: Verwende sinnvolle obere Grenze
                        # Hohe Frequenzen sind meist nur Rauschen
                        actual_max_freq = min(max_freq, 5000.0)  # Max 5kHz für bessere Normalisierung
                        break
            
            # Fallback: Prüfe calculated_data für original_freq
            elif 'calculated_data' in self.data and self.data['calculated_data']:
                for i, data_entry in enumerate(self.data['calculated_data']):
                    if 'original_freq' in data_entry:
                        max_freq = max(data_entry['original_freq'])
                        actual_max_freq = min(max_freq, 20000.0)  # Nicht höher als 20kHz
                        break
            
            # Fallback: Prüfe direkt in metadata
            elif 'original_freq' in self.data.get('metadata', {}):
                actual_max_freq = max(self.data['metadata']['original_freq'])
            
            # Normalisiere basierend auf der echten maximalen Messfrequenz
            freq_min_norm = freq_min / actual_max_freq
            freq_max_norm = freq_max / actual_max_freq

            
            # Designparameter für FIR-Filter (24 dB/Oktave)
            # HPF und LPF getrennt konfigurieren
            # HPF braucht aggressivere Parameter
            ripple_db_hpf = 80      # Aggressive Parameter für HPF
            transition_width_hpf = 0.02
            numtaps_min_hpf = 2001
            
            # LPF mit den ursprünglich guten Parametern
            ripple_db_lpf = 60      # Ursprüngliche Parameter für LPF (war gut)
            transition_width_lpf = 0.1
            numtaps_min_lpf = 501
            
            # Separate Behandlung von Hoch- und Tiefpass für bessere Kontrolle
            magnitude_filtered = magnitude_db.copy()

            
            # Hochpass-Filter (HPF)
            if freq_min_norm < 1.0 and freq_min > 0.1:
                # Berechne optimale HPF-Parameter für steile Flanke
                numtaps_hp, beta = kaiserord(ripple_db_hpf, transition_width_hpf)
                numtaps_hp = max(numtaps_hp, numtaps_min_hpf)  # Aggressive Parameter für HPF
                if numtaps_hp % 2 == 0:
                    numtaps_hp += 1  # Ungerade Anzahl für Type I Filter
                
                # Design Hochpass mit Kaiser-Fenster für steile Flanken
                hpf_coeffs = firwin(numtaps_hp, freq_min_norm, 
                                  pass_zero=False, window=('kaiser', beta))
                
                # Berechne die Frequenzantwort - verwende die tatsächliche maximale Frequenz
                w_hp, h_hp = freqz(hpf_coeffs, worN=len(freq), fs=2*actual_max_freq)
                
                # Interpoliere auf unser Frequenz-Grid
                hpf_magnitude = np.interp(freq, w_hp, np.abs(h_hp))
                
                # Wende HPF korrekt an: Multiplikation in linearer Domain
                magnitude_linear = 10**(magnitude_filtered / 20.0)
                magnitude_linear *= hpf_magnitude  # Multiplikation, nicht Addition!
                magnitude_filtered = 20 * np.log10(magnitude_linear + np.finfo(float).eps)
            
            # Tiefpass-Filter (LPF)
            if freq_max_norm < 1.0 and freq_max < 20000.0:
                # Berechne optimale LPF-Parameter (ursprünglich gute Werte)
                numtaps_lp, beta = kaiserord(ripple_db_lpf, transition_width_lpf)
                numtaps_lp = max(numtaps_lp, numtaps_min_lpf)  # Ursprüngliche Parameter für LPF
                if numtaps_lp % 2 == 0:
                    numtaps_lp += 1  # Ungerade Anzahl für Type I Filter
                
                # Design Tiefpass mit Kaiser-Fenster für steile Flanken
                lpf_coeffs = firwin(numtaps_lp, freq_max_norm, 
                                  pass_zero=True, window=('kaiser', beta))
                
                # Berechne die Frequenzantwort - verwende die tatsächliche maximale Frequenz
                w_lp, h_lp = freqz(lpf_coeffs, worN=len(freq), fs=2*actual_max_freq)
                
                # Interpoliere auf unser Frequenz-Grid
                lpf_magnitude = np.interp(freq, w_lp, np.abs(h_lp))
                
                # Wende LPF korrekt an: Multiplikation in linearer Domain
                magnitude_linear = 10**(magnitude_filtered / 20.0)
                magnitude_linear *= lpf_magnitude  # Multiplikation, nicht Addition!
                magnitude_filtered = 20 * np.log10(magnitude_linear + np.finfo(float).eps)
            
            # WICHTIG: Phase bleibt unverändert! 
            # FIR-Filter sind linear-phasig und fügen nur konstante Gruppenlaufzeit hinzu
            phase_filtered = phase.copy()  # Keine Phasenveränderung!
            
            return magnitude_filtered, phase_filtered
            
        except Exception as e:
            return magnitude_db, phase

    def apply_ir_window(self, impulse_response, window_ms=None):
        """
        Wendet eine Fensterfunktion auf die Impulsantwort an (IR Windowing).
        
        Fenster-Design:
        ---------------
        - 0-2%: Steiler Anstieg (0 → 1)
        - 2-80%: Linear bei 1.0
        - 80-100%: Flacher Abfall (Hanning-ähnlich)
        - Rest: 0.0 (Null nach window_samples)
        
        Eigenschaften:
        --------------
        - Nur aktiv wenn ir_window_enabled = True
        - Länge aus UI (ir_window in ms)
        - Fenster gleiche Länge wie ursprüngliche IR
        - Normalisiert auf 0-1
        
        Args:
            impulse_response: Die Impulsantwort als numpy array
            window_ms: Fensterlänge in ms (aus Metadaten falls None)
            
        Returns:
            tuple: (windowed_response, window_function)
                - windowed_response: Gefensterte IR (gleiche Länge)
                - window_function: Fensterfunktion 0-1 (gleiche Länge)
        """
        try:
            # Prüfe, ob Fensterung aktiviert ist
            if not self.data['metadata'].get('ir_window_enabled', False):
                # Fensterung deaktiviert - gib ursprüngliche IR zurück
                window_function = np.ones(len(impulse_response))
                return impulse_response, window_function
            
            # Hole Fensterlänge aus Metadaten falls nicht angegeben
            if window_ms is None:
                window_ms = self.data['metadata'].get('ir_window', 100.0)
            
            fs = self.data['metadata']['fs']
            window_samples = int(window_ms * fs / 1000)
            
            # Begrenze Fensterlänge auf die Länge der Impulsantwort
            if window_samples > len(impulse_response):
                window_samples = len(impulse_response)
            
            # Erstelle eine benutzerdefinierte Fensterfunktion (normalisiert 0-1)
            # WICHTIG: Fensterfunktion mit gleicher Länge wie ursprüngliche IR
            window_function = np.zeros(len(impulse_response))
            
            # Erstelle Fenster nur für die ersten window_samples
            window_short = np.zeros(window_samples)
            
            # Berechne die Übergangspunkte
            steep_start = 0
            steep_end = int(0.02 * window_samples)  # 2% für steilen Anstieg
            linear_end = int(0.8 * window_samples)  # 80% für linearen Bereich
            flat_end = window_samples  # 100% für flachen Abfall
            
            # Steiler Anstieg (0 bis 2%) - von 0 auf 1
            if steep_end > steep_start:
                steep_range = np.arange(steep_end - steep_start)
                window_short[steep_start:steep_end] = steep_range / (steep_end - steep_start)
            
            # Linearer Bereich (2% bis 80%) - bleibt bei 1.0
            window_short[steep_end:linear_end] = 1.0
            
            # Flacher Abfall (80% bis 100%) - Hanning-ähnlich
            if flat_end > linear_end:
                fall_range = np.arange(flat_end - linear_end)
                fall_length = flat_end - linear_end
                # Hanning-ähnliche Funktion für den Abfall
                window_short[linear_end:flat_end] = 0.5 * (1 + np.cos(np.pi * fall_range / fall_length))
            
            # Setze das Fenster in die volle Länge (normalisiert 0-1)
            window_function[:window_samples] = window_short[:window_samples]
            window_function[window_samples:] = 0.0  # Null nach der Fensterlänge
            
            # Wende das Fenster an (Fensterfunktion ist bereits 0-1 normalisiert)
            windowed_response = impulse_response * window_function
            
            
            return windowed_response, window_function
            
        except Exception as e:
            return impulse_response, np.ones(len(impulse_response))
