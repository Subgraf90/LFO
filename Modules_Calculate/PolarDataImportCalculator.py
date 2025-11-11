"""
Polar Data Import Calculator - WAV & TXT Import + PFAD 1 (Referenz-Verarbeitung)
=================================================================================

Speicherort: LFO/Module_LFO/Modules_Calculate/PolarDataImportCalculator.py

Zweck:
------
Komplette Verarbeitung von Messdaten:
A) WAV-Dateien:
   1. WAV-File einlesen
   2. Berechnung der Referenz-Frequenzdaten (PFAD 1)
   3. Speichern in raw_measurements

B) TXT-Dateien (Mag/Phase):
   1. TXT-File einlesen (Tab-getrennt)
   2. Phase unwrappen
   3. 1/96 Oktave Interpolation
   4. Rekonstruktion der Impulsantwort (Inverse FFT)
   5. Speichern in raw_measurements

PFAD 1 Daten dienen als unveränderliche Baseline (schwarze Kurve in Plots).

Verarbeitungs-Pipelines:
------------------------
WAV: WAV-File → Load → Clean IA → FFT → 400 Hz LP → Phase Unwrap → 1/96 Oktave
TXT: TXT-File → Parse → 400 Hz LP → Phase Unwrap → 1/96 Oktave → Inverse FFT (IR Rekonstruktion)

Abhängigkeiten:
---------------
- numpy: Alle Berechnungen
- scipy.signal: FIR-Filter Design (firwin, freqz, kaiserord)
- scipy.io.wavfile: WAV-File Lesen
- os: Dateinamen-Verarbeitung
- self.data: Zentrale Datenstruktur
  - Schreibt: self.data['raw_measurements'][filename]
  - Schreibt: self.data['metadata']['fs'], 'fft_length', 'freq_resolution'

Methoden:
---------
1. process_wav_file(filename)
   - HAUPTMETHODE: Lädt WAV und erstellt raw_measurements
   - Input: WAV-Dateipfad (String)
   - Output: True/False (Erfolg)

2. process_txt_file(filename)
   - HAUPTMETHODE: Lädt TXT (Mag/Phase) und erstellt raw_measurements
   - Input: TXT-Dateipfad (String, Format: Freq[Hz] \t Magnitude[dB] \t Phase[deg])
   - Output: True/False (Erfolg)

3. compute_fft_from_ir(impulse_response)
   - Berechnet FFT aus Impulsantwort (nur WAV)
   - Input: Impulsantwort (NumPy Array)
   - Output: (freq, magnitude_db, phase_rad)

4. process_frequency_data(freq, magnitude_db, phase_rad)
   - KERNMETHODE: Gemeinsame Verarbeitung für WAV & TXT
   - Pipeline: 400 Hz LP → Phase Unwrap → 1/96 Oktave
   - Input: Frequenzdaten (Hz, dB, Radiant)
   - Output: (freq, magnitude, phase) - Referenzdaten
   
5. compute_reference_from_ir(impulse_response)
   - Wrapper: FFT + Frequenzdaten-Verarbeitung (für Kompatibilität)
   - Input: Clean Impulsantwort (NumPy Array)
   - Output: (freq, magnitude, phase) - Referenzdaten
   
6. apply_fixed_400hz_lowpass(freq, magnitude_db, phase)
   - 400 Hz FIR-Tiefpass (Kaiser, 24 dB/Okt, 60 dB ripple)
   - Phase bleibt unverändert (linear-phasig)
   
7. average_frequency_bands_1_96(freq, magnitude, phase)
   - 1/96 Oktave Mittelung
   - 1 Hz bis 22050 Hz → ~1386 Frequenzpunkte
   - Interpoliert auf feste Frequenzen

8. calculate_processed_impulse_response(freq, magnitude, phase)
   - Rekonstruiert IR aus Frequenzdaten (für Plot)

Wird verwendet von:
-------------------
- PolarDataCalculator.py:
  - process_impulse_response() - Delegiert WAV-Import
  - process_txt_response() - Delegiert TXT-Import
  - recalculate_from_raw() - Nutzt PFAD 1 als Basis

Aufruf-Hierarchie:
------------------
process_wav_file()
  └─→ compute_reference_from_ir()
        ├─→ compute_fft_from_ir()
        └─→ process_frequency_data()
              ├─→ apply_fixed_400hz_lowpass()
              └─→ average_frequency_bands_1_96()
  └─→ calculate_processed_impulse_response()

process_txt_file()
  └─→ process_frequency_data()
        ├─→ apply_fixed_400hz_lowpass()
        └─→ average_frequency_bands_1_96()

Keine Abhängigkeiten zu:
-------------------------
- UI-Komponenten (PyQt5)
- Plotting (Matplotlib)
- Settings-Management
- main_window

→ Reines Import- & Berechnungsmodul ohne UI-Kopplung
"""

import numpy as np
import os
from scipy.io import wavfile


class Polar_Data_Import_Calculator:
    """
    WAV-Import & PFAD 1: Referenz-Verarbeitung
    WAV → Load → Clean IA → FFT → 400 Hz LP → Unwrap → 1/96 Okt
    """

    def __init__(self, data):
        """
        Initialisiert das Import-Calculator Modul.
        
        Args:
            data: Zentrale Datenstruktur (Dictionary)
                  Wird mit 'raw_measurements' und 'metadata' befüllt
        """
        self.data = data

    def process_txt_file(self, filename):
        """
        Lädt TXT-File mit Mag/Phase-Daten und erstellt raw_measurements.
        
        Format:
        -------
        Header-Zeilen (werden übersprungen)
        Freq [Hz]	Magnitude [dB]	Phase [deg]
        22.4	78.4	171.5
        23.0	80.0	166.8
        ...
        
        Pipeline:
        ---------
        1. TXT-File einlesen und parsen
        2. Metadaten setzen (fs auf Standard 48000 Hz)
        3. Phase zu Radiant konvertieren
        4. process_frequency_data() aufrufen (400 Hz LP, Unwrap, 1/96 Okt)
        5. calculate_processed_impulse_response() aufrufen (IR Rekonstruktion)
        6. Speichern in self.data['raw_measurements'][name]
        
        Args:
            filename: Pfad zur TXT-Datei (String)
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Schritt 1: TXT-File einlesen
            freq_list = []
            magnitude_list = []
            phase_list = []
            
            header_found = False
            line_count = 0
            
            with open(filename, 'r', encoding='utf-8') as f:
                # Header überspringen bis zur Datenzeile
                for line in f:
                    line = line.strip()
                    if line.startswith('Freq [Hz]'):
                        header_found = True
                        break
                    line_count += 1
                
                if not header_found:
                    print(f"ERROR process_txt_file: Header 'Freq [Hz]' nicht gefunden!")
                    return False
                
                # Daten einlesen
                data_line_count = 0
                for line in f:
                    line = line.strip()
                    if not line:  # Leere Zeile = Ende
                        break
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        try:
                            freq = float(parts[0])
                            magnitude = float(parts[1])
                            phase = float(parts[2])
                            
                            freq_list.append(freq)
                            magnitude_list.append(magnitude)
                            phase_list.append(phase)
                            data_line_count += 1
                        except ValueError as e:
                            continue
            
            if len(freq_list) == 0:
                print("ERROR process_txt_file: Keine Daten im TXT-File gefunden")
                return False
            
            # Konvertiere zu NumPy Arrays
            freq = np.array(freq_list)
            magnitude = np.array(magnitude_list)
            phase_wrapped = np.array(phase_list)
            
            # Schritt 2: Metadaten setzen
            # TXT hat keine echte Sample-Rate (nur Frequenzdaten)
            # Verwende 48000 Hz als Fallback für IR-Rekonstruktion
            fs = 48000
            
            self.data['metadata'].update({
                'data_source': 'txt',
                'fs': None,  # Keine echte Sample-Rate bei TXT (für Metadaten-Export)
                'fft_length': len(freq),
                'freq_resolution': freq[1] - freq[0] if len(freq) > 1 else 1.0
            })
            
            # Schritt 3: Konvertiere Phase zu Radiant
            phase_rad = np.radians(phase_wrapped)
            
            # Schritt 4: Nutze gemeinsame Frequenzdaten-Verarbeitung (wie bei WAV)
            # → 400 Hz LP, Phase Unwrap, 1/96 Oktave
            avg_freq, avg_magnitude, avg_phase = self.process_frequency_data(
                freq, magnitude, phase_rad
            )
            
            # Original wrapped Phase für Debug
            avg_phase_wrapped = ((avg_phase + 180) % 360) - 180
            
            # Schritt 5: Rekonstruiere Impulsantwort aus Frequenzdaten
            impulse_response, time_axis = self.calculate_processed_impulse_response(
                avg_freq, avg_magnitude, avg_phase
            )
            if impulse_response is None:
                impulse_response = np.array([])
            
            # Erstelle Window-Funktion (für TXT-Import: dummy window, da keine echte IR vorhanden)
            if len(impulse_response) > 0:
                window_function = np.ones(len(impulse_response))
            else:
                window_function = np.array([])
            
            # Schritt 6: Speichern in raw_measurements
            name = os.path.basename(filename)
            result_data = {
                'freq': avg_freq.copy(),
                'magnitude': avg_magnitude.copy(),
                'phase': avg_phase.copy(),
                'original_freq': avg_freq.copy(),
                'original_magnitude': avg_magnitude.copy(),
                'original_phase': avg_phase.copy(),
                'original_phase_wrapped': avg_phase_wrapped.copy(),
                # Rekonstruierte Impulsantwort aus TXT-Frequenzdaten
                'impulse_response': impulse_response.copy() if impulse_response is not None else np.array([]),
                'original_impulse_response': impulse_response.copy() if impulse_response is not None else np.array([]),
                'windowed_impulse_response': impulse_response.copy() if impulse_response is not None else np.array([]),
                'window_function': window_function.copy(),
                'fs': fs,
                'source_type': 'txt'  # Metadaten: TXT-Import
            }
            
            self.data['raw_measurements'][name] = result_data
            return True
            
        except Exception as e:
            print(f"ERROR process_txt_file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_wav_file(self, filename):
        """
        HAUPTMETHODE: Lädt WAV-File und erstellt raw_measurements.
        
        Pipeline:
        ---------
        1. WAV-File einlesen (scipy.io.wavfile)
        2. Metadaten aktualisieren (fs, fft_length, freq_resolution)
        3. PFAD 1: compute_reference_from_ir()
        4. PFAD 2: Beim Import = PFAD 1 (identisch)
        5. IR-Varianten für Plot berechnen
        6. Speichern in self.data['raw_measurements'][name]
        
        Args:
            filename: Pfad zur WAV-Datei (String)
            
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Schritt 1: WAV-File einlesen
            fs, data = wavfile.read(filename)
            impulse_response = data.astype(float)
            
            # Schritt 2: Metadaten aktualisieren
            self.data['metadata'].update({
                'data_source': 'Impulse response',
                'fs': fs,
                'fft_length': len(impulse_response),
                'freq_resolution': fs/len(impulse_response)
            })
            
            # Schritt 3: PFAD 1 (Referenz)
            clean_ir = impulse_response.copy()
            orig_avg_freq, orig_avg_magnitude, orig_avg_phase = self.compute_reference_from_ir(clean_ir)
            
            # Schritt 4: PFAD 2 beim Import = PFAD 1 (keine UI-Manipulationen)
            avg_freq = orig_avg_freq.copy()
            avg_magnitude = orig_avg_magnitude.copy()
            avg_phase = orig_avg_phase.copy()
            
            # Wrapped Phase für Debug
            avg_phase_wrapped = ((orig_avg_phase + 180) % 360) - 180
            
            # Schritt 5: IR-Varianten für Plot (beim Import: keine Manipulation)
            windowed_response = impulse_response.copy()
            window_function = np.ones(len(impulse_response))
            shifted_response = impulse_response.copy()
            
            # Berechne Impulsantworten für Plot
            processed_impulse, processed_time = self.calculate_processed_impulse_response(
                avg_freq, avg_magnitude, avg_phase
            )
            
            # Schritt 6: Speichern in raw_measurements (mit expliziten Kopien)
            name = os.path.basename(filename)
            result_data = {
                'freq': avg_freq.copy(),
                'magnitude': avg_magnitude.copy(),
                'phase': avg_phase.copy(),
                'original_freq': orig_avg_freq.copy(),
                'original_magnitude': orig_avg_magnitude.copy(),
                'original_phase': orig_avg_phase.copy(),
                'original_phase_wrapped': avg_phase_wrapped.copy(),
                'impulse_response': shifted_response.copy(),
                'original_impulse_response': impulse_response.copy(),
                'windowed_impulse_response': windowed_response.copy(),
                'window_function': window_function.copy(),
                'fs': fs,
                'source_type': 'Impulse Response'  # Metadaten: WAV-Import
            }
            
            if processed_impulse is not None and processed_time is not None:
                result_data['processed_impulse_response'] = processed_impulse
                result_data['processed_impulse_time'] = processed_time
            
            self.data['raw_measurements'][name] = result_data
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False



    def compute_reference_from_ir(self, impulse_response):
        """
        Erzeugt Referenzdaten (PFAD 1) aus einer Impulsantwort.
        
        Wrapper-Methode für Kompatibilität.
        Ruft compute_fft_from_ir() und process_frequency_data() auf.
        
        Args:
            impulse_response: Clean Impulsantwort (NumPy Array)
            
        Returns:
            tuple: (freq, magnitude, phase)
                - freq: Frequenzen in Hz (~1386 Punkte, 1/96 Oktave)
                - magnitude: Magnitude in dB
                - phase: Phase in Grad (unwrapped)
        """
        # FFT berechnen
        freq, magnitude_db, phase_rad = self.compute_fft_from_ir(impulse_response)
        
        # Frequenzdaten verarbeiten
        return self.process_frequency_data(freq, magnitude_db, phase_rad)



    def compute_fft_from_ir(self, impulse_response):
        """
        Berechnet FFT aus Impulsantwort (nur für WAV-Import).
        
        Pipeline:
        ---------
        1. FFT der Impulsantwort
        2. Magnitude (dB) & Phase (Radiant) berechnen
        
        Args:
            impulse_response: Impulsantwort (NumPy Array)
            
        Returns:
            tuple: (freq, magnitude_db, phase_rad)
                - freq: Frequenzen in Hz (FFT-Auflösung)
                - magnitude_db: Magnitude in dB
                - phase_rad: Phase in Radiant
        """
        fs = self.data['metadata']['fs']
        eps = np.finfo(float).eps

        # Schritt 1: FFT
        fft_result = np.fft.rfft(impulse_response)
        freq = np.fft.rfftfreq(len(impulse_response), d=1/fs)

        # Schritt 2: Magnitude & Phase
        magnitude_db = 20 * np.log10(np.abs(fft_result) + eps)
        phase_rad = np.angle(fft_result)

        return freq, magnitude_db, phase_rad


    def calculate_processed_impulse_response(self, freq, magnitude, phase):
        """
        Rekonstruiert Impulsantwort aus Frequenzdaten (für Plot).
        
        Verfahren:
        ----------
        1. Konvertiere Magnitude (dB → linear) & Phase (Grad → Radiant)
        2. Erstelle komplexe Frequenzdaten
        3. Interpoliere auf Target-FFT-Länge (262144 Samples)
        4. Inverse FFT → Impulsantwort
        5. Berechne Zeitachse
        
        Args:
            freq: Frequenzen in Hz
            magnitude: Magnitude in dB
            phase: Phase in Grad
            
        Returns:
            tuple: (impulse_response, time_axis) oder (None, None)
                - impulse_response: Rekonstruierte IR (NumPy Array)
                - time_axis: Zeitachse in ms
        """
        try:
            # Prüfe Eingabedaten
            if len(freq) == 0 or len(magnitude) == 0 or len(phase) == 0:
                return None, None
            if len(freq) != len(magnitude) or len(freq) != len(phase):
                return None, None
            
            # Konvertiere zu linearen/Radiant-Werten
            magnitude_linear = 10**(magnitude / 20.0)
            phase_rad = np.radians(phase)
            
            # Erstelle komplexe Frequenzdaten
            # Sample-Rate (Fallback bei TXT-Import)
            fs = self.data['metadata'].get('fs', 48000)
            if fs is None:
                fs = 48000  # Fallback für TXT-Import
            
            target_length = 262144  # Standard IR-Länge
            freq_target = np.fft.rfftfreq(target_length, d=1/fs)
            
            # Interpoliere auf Target-FFT-Länge
            magnitude_interp = np.interp(freq_target, freq, magnitude_linear)
            phase_interp = np.interp(freq_target, freq, phase_rad)
            complex_freq_interp = magnitude_interp * np.exp(1j * phase_interp)
            
            # Inverse FFT
            impulse_response = np.fft.irfft(complex_freq_interp, n=target_length)
            time_axis = np.arange(len(impulse_response)) / fs * 1000  # in ms
            
            return impulse_response, time_axis
            
        except Exception:
            import traceback
            traceback.print_exc()
            return None, None



    def process_frequency_data(self, freq, magnitude_db, phase_rad):
        """
        HAUPTMETHODE: Verarbeitet Frequenzdaten (gemeinsam für WAV & TXT).
        
        Pipeline:
        ---------
        1. 400 Hz Tiefpass anwenden (FIR)
        2. Phase bidirektional unwrappen (ab 50 Hz)
        3. Phase zu Grad konvertieren
        4. 1/96 Oktave Mittelung
        
        Args:
            freq: Frequenzen in Hz (NumPy Array)
            magnitude_db: Magnitude in dB (NumPy Array)
            phase_rad: Phase in Radiant (NumPy Array)
            
        Returns:
            tuple: (freq_avg, magnitude_avg, phase_avg)
                - freq_avg: Frequenzen in Hz (~1386 Punkte, 1/96 Oktave)
                - magnitude_avg: Magnitude in dB
                - phase_avg: Phase in Grad (unwrapped)
        """
        # Schritt 1: 400 Hz Lowpass
        magnitude_400hz, phase_400hz_rad = self.apply_fixed_400hz_lowpass(
            freq, magnitude_db, phase_rad
        )

        # Schritt 2 & 3: Phase Unwrap (bidirektional ab 50 Hz) & zu Grad
        ref_freq = 50.0
        ref_idx = np.argmin(np.abs(freq - ref_freq))
        phase_unwrapped = np.zeros_like(phase_400hz_rad, dtype=float)
        phase_unwrapped[ref_idx] = phase_400hz_rad[ref_idx]

        if ref_idx < len(phase_400hz_rad) - 1:
            phase_upper = phase_400hz_rad[ref_idx:]
            phase_unwrapped[ref_idx:] = np.unwrap(phase_upper)

        if ref_idx > 0:
            phase_lower = phase_400hz_rad[:ref_idx+1][::-1]
            phase_unwrapped[:ref_idx+1] = np.unwrap(phase_lower)[::-1]

        phase_deg = np.degrees(phase_unwrapped)

        # Schritt 4: 1/96 Oktave Mittelung
        avg_freq, avg_magnitude, avg_phase = self.average_frequency_bands_1_96(
            freq, magnitude_400hz, phase_deg
        )

        return avg_freq, avg_magnitude, avg_phase



    def apply_fixed_400hz_lowpass(self, freq, magnitude_db, phase):
        """
        Wendet fixen 400 Hz FIR-Tiefpass an (IMMER für PFAD 1).
        
        Filter-Eigenschaften:
        ---------------------
        - Typ: FIR (Finite Impulse Response)
        - Grenzfrequenz: 400 Hz (fest)
        - Flankensteilheit: 24 dB/Oktave
        - Ripple: 60 dB
        - Transition Width: 0.1
        - Min. Taps: 501 (ungerade für Type I Filter)
        - Fenster: Kaiser
        
        Verhalten:
        ----------
        - Magnitude: Wird gefiltert (Dämpfung oberhalb 400 Hz)
        - Phase: Bleibt UNVERÄNDERT (FIR = linear-phasig)
        
        Args:
            freq: Frequenz-Array in Hz
            magnitude_db: Magnitude in dB
            phase: Phase in Radiant
            
        Returns:
            tuple: (magnitude_filtered, phase)
                - magnitude_filtered: Gefilterte Magnitude in dB
                - phase: Unveränderte Phase in Radiant
        """
        try:
            from scipy.signal import firwin, freqz, kaiserord
            
            freq_max = 400.0  # Fixe Grenzfrequenz
            
            # Ermittle maximale Frequenz der Messdaten
            actual_max_freq = 20000.0
            if 'raw_measurements' in self.data and self.data['raw_measurements']:
                for _, raw_data in self.data['raw_measurements'].items():
                    if 'freq' in raw_data:
                        max_freq = max(raw_data['freq'])
                        actual_max_freq = min(max_freq, 5000.0)
                        break
            
            # Normalisiere Grenzfrequenz
            freq_max_norm = freq_max / actual_max_freq

            # FIR-Filter Parameter
            ripple_db_lpf = 60
            transition_width_lpf = 0.1
            numtaps_min_lpf = 501

            magnitude_filtered = magnitude_db.copy()
            
            # Design & Anwendung des Filters
            if freq_max_norm < 1.0:
                numtaps_lp, beta = kaiserord(ripple_db_lpf, transition_width_lpf)
                numtaps_lp = max(numtaps_lp, numtaps_min_lpf)
                if numtaps_lp % 2 == 0:
                    numtaps_lp += 1  # Ungerade für Type I Filter
                
                lpf_coeffs = firwin(numtaps_lp, freq_max_norm, pass_zero=True, window=('kaiser', beta))
                w_lp, h_lp = freqz(lpf_coeffs, worN=len(freq), fs=2*actual_max_freq)
                lpf_magnitude = np.interp(freq, w_lp, np.abs(h_lp))
                
                # Filter in linearer Domain anwenden (Multiplikation!)
                magnitude_linear = 10**(magnitude_filtered / 20.0)
                magnitude_linear *= lpf_magnitude
                magnitude_filtered = 20 * np.log10(magnitude_linear + np.finfo(float).eps)

            # Phase bleibt unverändert (FIR ist linear-phasig)
            phase_filtered = phase.copy()
            return magnitude_filtered, phase_filtered
            
        except Exception:
            return magnitude_db, phase





    def average_frequency_bands_1_96(self, freq, magnitude, phase):
        """
        Erstellt feste 1/96 Oktave Frequenzpunkte über gesamte Bandbreite.
        
        Auflösung:
        ----------
        - 1/96 Oktave (fraction = 96)
        - Bereich: 1 Hz bis 22050 Hz (Nyquist bei 44.1 kHz)
        - Ergebnis: ~1386 Frequenzpunkte
        
        Berechnung:
        -----------
        - Frequenzen: f[i] = f_min * 2^(i/96)
        - Magnitude & Phase: Linear interpoliert (np.interp)
        
        Args:
            freq: Ursprüngliche Frequenzen (variabel)
            magnitude: Magnitude in dB
            phase: Phase in Grad
            
        Returns:
            tuple: (fixed_freqs, interpolated_magnitude, interpolated_phase)
                - Feste 1/96 Oktave Frequenzen
                - Interpolierte Magnitude
                - Interpolierte Phase
        """
        try:
            # 1/96 Oktave Auflösung
            fraction = 96
            f_min = 1.0       # Start: 1 Hz
            f_max = 22050.0   # Ende: Nyquist (44.1 kHz)
            
            # Berechne Anzahl der Bänder
            num_bands = int(np.ceil(fraction * np.log2(f_max/f_min)))
            
            # Erstelle feste Frequenzpunkte: f = f_min * 2^(i/fraction)
            fixed_freqs = []
            for i in range(num_bands):
                fixed_freq = f_min * (2 ** (i/fraction))
                if fixed_freq <= f_max:
                    fixed_freqs.append(fixed_freq)
            fixed_freqs = np.array(fixed_freqs)

            # Initialisiere mit Platzhaltern
            interpolated_magnitude = np.full(len(fixed_freqs), 0.001)  # Sehr kleine Magnitude
            interpolated_phase = np.full(len(fixed_freqs), 0.0)        # 0° Phase
            
            # Interpoliere (np.interp ist 10-50× schneller als scipy)
            if len(freq) > 0 and len(magnitude) > 0:
                interpolated_magnitude = np.interp(fixed_freqs, freq, magnitude)
                interpolated_phase = np.interp(fixed_freqs, freq, phase)
            
            return fixed_freqs, interpolated_magnitude, interpolated_phase
            
        except Exception:
            return freq, magnitude, phase
