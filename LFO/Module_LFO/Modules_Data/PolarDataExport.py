"""
Polar Data Export - Datenexport für LFO-Format
===============================================

Speicherort: LFO/Module_LFO/Modules_Data/PolarDataExport.py

Zweck:
------
Exportiert verarbeitete Polar-Daten im LFO-Format (.npz).
1. Liest balloon_data aus PolarDataCalculator
2. Begrenzt auf 400 Hz (Dateigröße-Optimierung)
3. Exportiert Magnitude, Phase, Cabinet-Daten, Metadaten

Export-Pipeline:
----------------
balloon_data → 400 Hz Begrenzung → Cabinet-Daten → Metadaten → .npz

Abhängigkeiten:
---------------
- numpy: Array-Operationen, npz-Export
- os: Dateipfad-Operationen
- PyQt5: UI-Dialoge
- self.data: Zentrale Datenstruktur
  - Liest: balloon_data, metadata
- self.speakers: Cabinet-Spezifikationen (UI-Input)

Hauptmethoden:
--------------
1. export_to_lfo(filename, speakers)
   - HAUPTMETHODE: Exportiert vollständige Daten im LFO-Format
   - Output: .npz Datei mit balloon_data, cabinet_data, metadata
   
2. apply_bandwidth_limit(freq, magnitude, phase, max_freq=400.0)
   - Begrenzt Bandbreite auf max_freq
   - Sanfter Roll-off für saubere IR-Rekonstruktion

Datenstruktur Export:
---------------------
.npz enthält:
- balloon_data:
  - vertical_angles: Array der vertikalen Winkel
  - magnitude: 3D-Array [vert, horz, freq]
  - phase: 3D-Array [vert, horz, freq]
  - freqs: Frequenz-Array (bis 400 Hz)
- cabinet_data: Liste mit Cabinet-Specs (width, depth, heights, etc.)
- metadata: Alle Processing-Parameter (Filter, Windowing, Normalisierung)

Wird verwendet von:
-------------------
- UiPolarDataGenerator: Export-Button Callback

Aufruf-Hierarchie:
------------------
export_to_lfo()
  ├─→ Hole balloon_data aus self.data
  ├─→ Begrenze auf 400 Hz (cutoff_idx)
  ├─→ Sammle Cabinet-Daten aus self.speakers
  ├─→ Erstelle metadata aus self.data['metadata']
  └─→ np.savez_compressed()
"""

import numpy as np
import os
import csv
from typing import Optional
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog
from PyQt5.QtCore import Qt
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from scipy.io import wavfile

# warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)


class Polar_Data_Export(ModuleBase):
    """
    Exportiert verarbeitete Polar-Daten im LFO-Format (.npz).
    Liest balloon_data von PolarDataCalculator, begrenzt auf 400 Hz.
    """
    EXPORT_PROGRESS_STEPS = 6

    def __init__(self, settings, data, speakers=None, container=None):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.speakers = speakers or []  # Leere Liste falls keine Lautsprecher übergeben
        self.container = container
        

    def apply_bandwidth_limit(self, freq, magnitude, phase, max_freq=400.0):
        """
        Begrenzt die Bandbreite für saubere IR-Rekonstruktion.
        
        Verfahren:
        ----------
        1. Schneide alle Frequenzen > max_freq ab
        2. Sanfter Roll-off in letzten 10% (0 bis -60 dB)
        3. Verhindert Artefakte bei IR-Rekonstruktion
        
        Args:
            freq: Frequenz-Array
            magnitude: Magnitude in dB
            phase: Phase in Radiant
            max_freq: Maximale Frequenz in Hz (Standard: 400Hz)
            
        Returns:
            tuple: (freq_limited, magnitude_limited, phase_limited)
        """
        try:
            # Stelle sicher, dass wir mit NumPy-Arrays arbeiten (Listen-Konvertierung nach vorherigem Export)
            freq = np.asarray(freq, dtype=float)
            magnitude = np.asarray(magnitude, dtype=float)
            phase = np.asarray(phase, dtype=float)

            # Finde Index für Grenzfrequenz
            cutoff_idx = np.where(freq <= max_freq)[0]
            
            if len(cutoff_idx) == 0:
                # Falls alle Frequenzen über der Grenze liegen, nur DC nehmen
                cutoff_idx = np.array([0])
            
            # Schneide Arrays ab
            freq_limited = freq[cutoff_idx]
            magnitude_limited = magnitude[cutoff_idx]
            phase_limited = phase[cutoff_idx]
            
            # Für saubere IR-Rekonstruktion: Sanfter Roll-off am Ende
            if len(freq_limited) > 1:
                # Letzte 10% der Frequenzen sanft auf -60dB dämpfen
                rolloff_start = int(len(freq_limited) * 0.9)
                rolloff_length = len(freq_limited) - rolloff_start
                
                if rolloff_length > 0:
                    # Exponentieller Roll-off für glatten Übergang
                    rolloff_factor = np.linspace(0, -60, rolloff_length)  # 0 bis -60dB
                    magnitude_limited[rolloff_start:] += rolloff_factor
            
            return freq_limited, magnitude_limited, phase_limited
            
        except Exception as e:
            print(f"Fehler bei Bandbegrenzung: {e}")
            return freq, magnitude, phase


    def export_magnitude_phase_limited(self, file_path, max_freq=400.0, file_format='txt'):
        """Exportiert Magnitude und Phase Daten mit Bandbegrenzung
        
        Args:
            file_path: Pfad zur Ausgabedatei
            max_freq: Maximale Frequenz in Hz (Standard: 400Hz)
            file_format: Format ('txt', 'frd', 'csv')
            
        Returns:
            success, message
        """
        try:
            # Prüfe ob calculated_data vorhanden ist
            if 'calculated_data' not in self.data or not self.data['calculated_data']:
                return False, "Keine berechneten Daten zum Exportieren vorhanden"
            
            # Nehme erste verfügbare Messung
            first_measurement = next(iter(self.data['calculated_data'].values()))
            
            # Hole die Daten
            freq = first_measurement.get('freq', [])
            magnitude = first_measurement.get('magnitude', [])
            phase = first_measurement.get('phase', [])
            
            if len(freq) == 0 or len(magnitude) == 0 or len(phase) == 0:
                return False, "Unvollständige Daten (Frequenz, Magnitude oder Phase fehlt)"
            
            # Wende Bandbegrenzung an
            freq_limited, magnitude_limited, phase_limited = self.apply_bandwidth_limit(
                np.array(freq), np.array(magnitude), np.array(phase), max_freq
            )
            
            # Export-Funktionen wurden entfernt
            return False, "Export-Funktionen sind nicht mehr verfügbar"
                
        except Exception as e:
            return False, f"Export fehlgeschlagen: {str(e)}"


    def export_to_lfo(self, filename, speakers, progress=None):
        """
        HAUPTMETHODE: Exportiert vollständige Daten im LFO-Format (.npz).
        
        NEUE NUMPY-STRUKTUR Export-Pipeline:
        ------------------------------------
        1. Hole balloon_data aus self.data (NEUE NumPy-Struktur)
        2. Begrenze Frequenzen auf 400 Hz
        3. Exportiere direkt aus NumPy-Arrays (keine Konvertierung nötig)
        4. Sammle Cabinet-Daten aus speakers (UI)
        5. Erstelle Metadaten aus self.data['metadata']
        6. Speichere als komprimierte .npz Datei
        
        Datenstruktur (NEUE NUMPY-STRUKTUR):
        ------------------------------------
        .npz enthält:
        - balloon_data:
          - meridians: [n_meridians] (z.B. 0-360° in 5° Schritten → 72 Werte)
          - horizontal_angles: [n_horizontal] (0-180° in 1° Schritten → 181 Werte)
          - frequencies: [n_freq] (bis 400 Hz)
          - magnitude: [n_meridians, n_horizontal, n_freq]
          - phase: [n_meridians, n_horizontal, n_freq]
        - cabinet_data: Liste mit Dicts
          - width, depth, front_height, back_height
          - configuration, angle_point, angles, cardio
          - x_offset (zentriert)
        - metadata: Alle Processing-Parameter
          - Filter, Windowing, Normalisierung, etc.
        
        Args:
            filename: Zieldateiname (ohne .npz)
            speakers: Liste mit Cabinet-Spezifikationen (UI)
            
        Returns:
            tuple: (success, message)
                - success: True/False
                - message: Erfolgsmeldung oder Fehlerbeschreibung
        """
        
        try:
            def set_progress(label: str) -> None:
                if progress:
                    progress.update(label)

            def advance_progress() -> None:
                if progress:
                    progress.advance()

            set_progress("Validating balloon data")

            # Prüfe ob Balloon-Daten vorhanden sind
            if 'balloon_data' not in self.data:
                advance_progress()
                return False, "Keine Daten zum Speichern vorhanden!"
            
            balloon = self.data['balloon_data']
            
            # ================================================================
            # SCHRITT 1: PRÜFE NEUE NUMPY-STRUKTUR
            # ================================================================
            if not isinstance(balloon, dict) or 'meridians' not in balloon:
                advance_progress()
                return False, "Balloon-Daten haben nicht die erwartete NumPy-Struktur! Bitte zuerst Interpolation durchführen."

            advance_progress()

            set_progress("Preparing frequency data")
            # Hole Frequenzen aus balloon_data
            freqs_full = np.asarray(balloon['frequencies'], dtype=float)
            
            # 🔧 NUR bis 400 Hz exportieren (reduziert Dateigröße)
            cutoff_idx = np.where(freqs_full <= 400.0)[0]
            
            if len(cutoff_idx) == 0:
                advance_progress()
                return False, "Keine Frequenzen <= 400 Hz gefunden!"
            
            freqs = freqs_full[cutoff_idx]

            advance_progress()

            set_progress("Extracting balloon arrays")
            # ================================================================
            # SCHRITT 2: EXTRAHIERE DATEN BIS 400 HZ (DIREKT AUS NUMPY)
            # ================================================================
            # NEUE STRUKTUR: Direkt aus NumPy-Arrays (nur neue Keys)
            balloon_data = {
                'meridians': balloon['meridians'].copy(),
                'horizontal_angles': balloon['horizontal_angles'].copy(),
                'frequencies': freqs,
                'magnitude': balloon['magnitude'][:, :, cutoff_idx].copy(),
                'phase': balloon['phase'][:, :, cutoff_idx].copy()
            }
            
            self.speakers = speakers

            advance_progress()

            set_progress("Aggregating cabinet data")
            # Cabinet Data mit allen neuen Parametern
            cabinet_data = []
            
            if self.speakers:
                total_stack_width = 0.0
                stack_entries = []
                for speaker in self.speakers:
                    try:
                        width = float(speaker['width'].text() or 0)
                        depth = float(speaker['depth'].text() or 0)
                        front_height = float(speaker['front_height'].text() or 0)
                        back_height = float(speaker['back_height'].text() or 0)
                        configuration = speaker['configuration'].currentText().lower()
                        is_flown_config = configuration == 'flown'
                        angle_point = speaker['angle_point'].currentText()
                        is_cardio = speaker['cardio'].isChecked()
                        
                        # Sammle die Winkel aus den Winkel-Items
                        angles_list = []
                        if angle_point != "None" and 'angle_items' in speaker:
                            for angle_editor in speaker['angle_items']:
                                try:
                                    angle = float(angle_editor.text())
                                    angles_list.append(angle)
                                except ValueError:
                                    continue
                        cabinet_entry = {
                            'width': width,
                            'depth': depth,
                            'front_height': front_height,
                            'back_height': back_height, 
                            'configuration': configuration,
                            'angle_point': angle_point,
                            'angles': angles_list,
                            'cardio': is_cardio
                        }

                        if not is_flown_config:
                            stack_layout = 'beside'
                            if 'stack_layout' in speaker and speaker['stack_layout'] is not None:
                                stack_widget = speaker['stack_layout']
                                if hasattr(stack_widget, 'currentText'):
                                    stack_layout = stack_widget.currentText().strip().lower() or 'beside'
                            cabinet_entry['stack_layout'] = stack_layout
                            cabinet_entry['x_offset'] = total_stack_width
                            total_stack_width += width
                            stack_entries.append(cabinet_entry)

                        cabinet_data.append(cabinet_entry)
                    except ValueError:
                        continue
                
                # Zentriere x_offsets
                if stack_entries:
                    center_offset = total_stack_width / 2
                    for cabinet in stack_entries:
                        cabinet['x_offset'] -= center_offset

            advance_progress()

            set_progress("Preparing metadata")
            
            # ================================================================
            # SCHRITT 3: ERSTELLE METADATEN (WIE IN UiManageSpeaker.py ERWARTET)
            # ================================================================
            metadata = {}
            
            # ================================================================
            # MESS-INFORMATIONEN (aus folder_data berechnen)
            # ================================================================
            if 'folder_data' in self.data['metadata'] and self.data['metadata']['folder_data']:
                folder_data = self.data['metadata']['folder_data']
                
                # Anzahl gemessener Meridiane
                num_meridians = len(folder_data)
                
                # Anzahl Messpunkte pro Meridian (MINIMUM bei inkonsistenten Werten)
                measurements_per_meridian_list = []
                for folder in folder_data:
                    num_measurements = len(folder.get('measurements', []))
                    measurements_per_meridian_list.append(num_measurements)
                
                # Verwende MINIMUM für Metadaten
                meridian_measurements = min(measurements_per_meridian_list) if measurements_per_meridian_list else 0
                
                metadata['meridians'] = int(num_meridians)
                metadata['meridian_measurements'] = int(meridian_measurements)
            else:
                # Fallback: Wenn folder_data fehlt
                metadata['meridians'] = 0
                metadata['meridian_measurements'] = 0
            
            # Interpolierte Winkel (aus balloon_data)
            metadata['interpolated_meridians'] = int(len(balloon_data['meridians']))
            metadata['interpolated_horizontal_angles'] = int(len(balloon_data['horizontal_angles']))
            
            # Datenquelle
            metadata['data_source'] = self.data['metadata'].get('data_source', 'measurement')
            
            # Sample Rate (None bei TXT-Import)
            fs_value = self.data['metadata'].get('fs', 44100)
            metadata['fs'] = int(fs_value) if fs_value is not None else None
            
            # Frequenzbereich
            metadata['freq_range_min'] = float(self.data['metadata'].get('filter_freq_min', 10.0))
            metadata['freq_range_max'] = float(self.data['metadata'].get('filter_freq_max', 300.0))
            
            # Filter-Informationen
            metadata['filter_enabled'] = bool(self.data['metadata'].get('filter_enabled', False))
            
            # Window-Funktion Informationen
            metadata['ir_window_enabled'] = bool(self.data['metadata'].get('ir_window_enabled', False))
            metadata['ir_window_length'] = float(self.data['metadata'].get('ir_window', 100.0))
            
            # Normalisierungswerte
            metadata['spl_normalized'] = bool(self.data['metadata'].get('spl_normalized', False))
            metadata['spl_value'] = float(self.data['metadata'].get('spl_value', 0.0))
            metadata['reference_freq'] = float(self.data['metadata'].get('reference_freq', 50.0))
            metadata['reference_distance'] = float(self.data['metadata'].get('reference_distance', 1.0))
            metadata['target_spl'] = float(self.data['metadata'].get('target_spl', 85.0))
            metadata['time_offset'] = float(self.data['metadata'].get('time_offset', 0.0))

            advance_progress()

            set_progress("Writing export archive")
            # Speichern in Ordnerstruktur
            mag_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'Modules_Data', 
                'PolarData_mag',
                f"{filename}.npz"
            )
            os.makedirs(os.path.dirname(mag_file), exist_ok=True)
            
            # ================================================================
            # SCHRITT 4: SPEICHERE NPZ-DATEI
            # ================================================================
            # 🔧 FIX: Speichere balloon_data NICHT als verschachteltes Dict, 
            # sondern als flache Struktur für korrektes Laden!
            # np.savez_compressed speichert verschachtelte Dicts als object arrays,
            # die beim Laden problematisch sind.
            save_dict = {
                # Balloon Data - als flache Struktur (NICHT verschachtelt!)
                'balloon_data_meridians': balloon_data['meridians'].astype(np.int32),
                'balloon_data_horizontal_angles': balloon_data['horizontal_angles'].astype(np.int32),
                'balloon_data_frequencies': balloon_data['frequencies'].astype(np.float32),
                'balloon_data_magnitude': balloon_data['magnitude'].astype(np.float32),
                'balloon_data_phase': balloon_data['phase'].astype(np.float32),
                # Andere Daten
                'cabinet_data': cabinet_data,
                'metadata': metadata
            }
            
            np.savez_compressed(mag_file, **save_dict)
            advance_progress()
            
            # Lade neue Daten sofort in den DataContainer, falls verfügbar
            if self.container:
                try:
                    self.container.load_polardata(force=True)
                except Exception as exc:
                    print(f"Warnung: Neuladen der Polar-Daten nach Export fehlgeschlagen: {exc}")
            
            # Dateigröße anzeigen
            file_size_mb = os.path.getsize(mag_file) / (1024 * 1024)
            
            # Erfolgsmeldung mit Details
            success_message = (
                f"Balloon data saved as: {filename}.npz ({file_size_mb:.2f} MB)\n"
            )
            
            return True, success_message
            
        except Exception as e:
            return False, f"Export fehlgeschlagen: {str(e)}"