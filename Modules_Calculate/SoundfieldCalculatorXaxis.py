import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase

"SoundfieldCalculatorXaxis berechnet die SPL-Werte in Abhängigkeit der Breite des Schallfeldes"

class SoundFieldCalculatorXaxis(ModuleBase):
    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt


    def calculateXAxis(self):
        resolution = 0.25
        total_rows = int(self.settings.length / resolution)
        row_index = int((self.settings.position_y_axis / resolution) + (total_rows / 2))
        row_index = max(0, min(row_index, total_rows - 1))
        
        sound_field_p = self.calculate_sound_field_row(row_index)[0].flatten()
    
        sound_field_p_calc = self.functions.mag2db(sound_field_p)
        sound_field_x_xaxis_calc = np.arange((self.settings.width / 2 * -1), ((self.settings.width / 2) + resolution), resolution)

        # Prüfen, ob es aktive Quellen gibt oder ob das Ergebnis verwertbar ist
        has_active_sources = any(not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values())
        # Wenn keine aktive Quelle beteiligt war, bleibt das Feld numerisch 0 (vor dB-Umrechnung -inf)
        # Wir prüfen konservativ auf eine sehr kleine Varianz nach dB-Umrechnung
        is_meaningful_curve = np.isfinite(sound_field_p_calc).any()

        show_curve = has_active_sources and is_meaningful_curve

        # Daten aus calculation werden zurückgegeben
        self.calculation_spl["aktuelle_simulation"] = {
            "x_data_xaxis": sound_field_p_calc,
            "y_data_xaxis": sound_field_x_xaxis_calc,
            "show_in_plot": bool(show_curve),
            "color": "#6A5ACD"
        }
      
    def get_balloon_data_batch(self, speaker_name, azimuths, elevations, use_averaged=True):
        """
        🚀 BATCH-OPTIMIERT: Holt Balloon-Daten für VIELE Winkel auf einmal
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuths (np.ndarray): Array von Azimut-Winkeln in Grad
            elevations (np.ndarray): Array von Elevations-Winkeln in Grad
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten
            
        Returns:
            tuple: (magnitudes, phases) als Arrays oder (None, None) bei Fehler
        """
        if use_averaged:
            balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=True)
            if balloon_data is not None:
                magnitude = balloon_data.get('magnitude')
                phase = balloon_data.get('phase')
                vertical_angles = balloon_data.get('vertical_angles')
                
                if magnitude is not None and phase is not None and vertical_angles is not None:
                    return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)
        
        balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
        if balloon_data is None:
            return None, None
        
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            return None, None
        
        return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)

    def get_balloon_data_at_angle(self, speaker_name, azimuth, elevation=0, use_averaged=True):
        """
        Holt die Balloon-Daten (Magnitude und Phase) für einen bestimmten Winkel
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuth (float): Azimut-Winkel in Grad (0-360)
            elevation (float, optional): Elevations-Winkel in Grad. Standard ist 0 (horizontale Ebene).
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten (Standard: True)
            
        Returns:
            tuple: (magnitude, phase) oder (None, None) wenn keine Daten gefunden wurden
        """
        # 🚀 OPTIMIERT: Verwende bandgemittelte Daten falls verfügbar
        if use_averaged:
            balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=True)
            if balloon_data is not None:
                # 🎯 BANDGEMITTELT: Verwende bereits gemittelte Daten
                magnitude = balloon_data.get('magnitude')
                phase = balloon_data.get('phase')
                vertical_angles = balloon_data.get('vertical_angles')
                
                if magnitude is not None and phase is not None and vertical_angles is not None:
                    # Bandgemittelte Daten haben keine Frequenzdimension mehr
                    return self._interpolate_angle_data(magnitude, phase, vertical_angles, azimuth, elevation)
        
        # 🔄 FALLBACK: Verwende originale Daten
        balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
        if balloon_data is None:
            print(f"❌ Keine Balloon-Daten für {speaker_name}")
            return None, None
        
        # Direkte Nutzung der optimierten Balloon-Daten
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            print(f"Fehlende Daten in Balloon-Daten für {speaker_name}")
            return None, None
        
        # Verwende die gleiche Interpolationslogik für beide Datentypen
        return self._interpolate_angle_data(magnitude, phase, vertical_angles, azimuth, elevation)

    def calculate_sound_field_row(self, row_index):
        """
        🚀 VEKTORISIERT: Berechnet SPL-Werte entlang der X-Achse (Breite)
        
        Berechnet eine einzelne Zeile des Schallfelds (konstante Y-Position).
        Nutzt Batch-Interpolation für optimale Performance.
        
        Args:
            row_index: Index der Y-Position im Grid
            
        Returns:
            tuple: (sound_field_p, sound_field_x, sound_field_y)
        """
        # ============================================================
        # SCHRITT 1: Grid-Erstellung (1D - nur X-Achse)
        # ============================================================
        width = self.settings.width
        length = self.settings.length
        resolution = 0.25

        # Erstelle 1D-Arrays für X- und Y-Koordinaten
        sound_field_y = np.arange((width / 2 * -1), ((width / 2) + resolution), resolution)
        sound_field_x = np.arange((length / 2 * -1), ((length / 2) + resolution), resolution)
    
        # Erstelle Koordinatenraster nur für eine Zeile (konstante Y-Position)
        sound_field_x, sound_field_y = np.meshgrid(sound_field_y, np.array([sound_field_x[row_index]]))
    
        # Initialisiere Schallfeld als komplexes Array
        sound_field_p = np.zeros_like(sound_field_y, dtype=complex)
    
        # ============================================================
        # SCHRITT 2: Iteriere über alle Lautsprecher-Arrays
        # ============================================================
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            # Konvertiere Lautsprechernamen in Indizes
            speaker_indices = []
            for speaker_name in speaker_array.source_polar_pattern:
                try:
                    speaker_names = self._data_container.get_speaker_names()
                    index = speaker_names.index(speaker_name)
                    speaker_indices.append(index)
                except ValueError:
                    print(f"Warnung: Lautsprecher {speaker_name} nicht gefunden")
                    speaker_indices.append(0)
            
            source_indices = np.array(speaker_indices)
            source_position_x = getattr(
                speaker_array,
                'source_position_calc_x',
                speaker_array.source_position_x,
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                speaker_array.source_position_y,
            )
            
            # Prüfe, ob Z-Position vorhanden ist
            source_position_z = getattr(speaker_array, 'source_position_calc_z', None)
            
            source_azimuth_deg = speaker_array.source_azimuth
            source_azimuth = np.deg2rad(source_azimuth_deg)
            source_delay = speaker_array.delay
            
            # Stelle sicher, dass source_time ein Array/Liste ist
            if isinstance(speaker_array.source_time, (int, float)):
                # Einzelner Wert - konvertiere zu Liste
                source_time = [speaker_array.source_time + source_delay]
            else:
                source_time = [time + source_delay for time in speaker_array.source_time]
            source_time = [x / 1000 for x in source_time]
            source_gain = speaker_array.gain
            source_level = speaker_array.source_level + source_gain
            source_level = self.functions.db2mag(np.array(source_level))
            calculate_frequency = self.settings.calculate_frequency
            a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
            wave_number = self.functions.wavenumber(self.settings.speed_of_sound, self.settings.calculate_frequency)

            # ============================================================
            # SCHRITT 3: Iteriere über alle Lautsprecher im Array
            # ============================================================
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                
                # --------------------------------------------------------
                # 3.1: VEKTORISIERTE GEOMETRIE-BERECHNUNG
                # --------------------------------------------------------
                # Berechne Distanzen für ALLE Punkte der X-Achse gleichzeitig
                x_distance = sound_field_x - source_position_x[isrc]
                y_distance = sound_field_y - source_position_y[isrc]
                
                # Horizontale Distanz (Pythagoras in 2D)
                # √(x² + y²) für ALLE Punkte gleichzeitig
                horizontal_dist = np.sqrt(x_distance**2 + y_distance**2)
                
                # Z-Distanz (konstant für alle Punkte, da Grid auf Z=0)
                z_distance = -source_position_z[isrc]
                
                # 3D-Distanz (Pythagoras in 3D)
                # √(horizontal² + z²) für ALLE Punkte gleichzeitig
                source_dist = np.sqrt(horizontal_dist**2 + z_distance**2)
                
                # --------------------------------------------------------
                # 3.2: MASKEN-LOGIK (Filtert ungültige Punkte)
                # --------------------------------------------------------
                # Filtere Punkte zu nah an der Quelle (verhindert Division durch Null)
                valid_mask = source_dist >= 0.001
                
                # Initialisiere Wellen-Array mit Nullen
                wave = np.zeros_like(source_dist, dtype=complex)
                
                # --------------------------------------------------------
                # 3.3: VEKTORISIERTE WINKEL-BERECHNUNG
                # --------------------------------------------------------
                # Berechne Azimut und Elevation für ALLE Punkte gleichzeitig
                source_to_point_angles = np.arctan2(y_distance, x_distance)
                azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                azimuths = (360 - azimuths) % 360  # Invertiere (Uhrzeigersinn)
                azimuths = (azimuths + 90) % 360  # Drehe 90° (Polar-Koordinatensystem)
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dist))
                
                # --------------------------------------------------------
                # 3.4: BATCH-INTERPOLATION (Performance-Booster! 🚀)
                # --------------------------------------------------------
                # Hole Balloon-Daten für ALLE Punkte in EINEM Aufruf
                # → Alle Interpolationen gleichzeitig statt in einer Loop!
                polar_gains, polar_phases = self.get_balloon_data_batch(speaker_name, azimuths, elevations)
                
                if polar_gains is not None and polar_phases is not None:
                    # --------------------------------------------------------
                    # 3.5: VEKTORISIERTE WELLENBERECHNUNG
                    # --------------------------------------------------------
                    # Konvertiere dB zu linear (vektorisiert)
                    magnitude_linear = 10 ** (polar_gains / 20)
                    polar_phase_rad = np.radians(polar_phases)
                    
                    # Berechne komplexe Welle für ALLE gültigen Punkte gleichzeitig
                    # Wellenformel: p(r) = (A × G × A₀ / r) × exp(i × Φ)
                    wave[valid_mask] = (magnitude_linear[valid_mask] * source_level[isrc] * a_source_pa * 
                                       np.exp(1j * (wave_number * source_dist[valid_mask] +  
                                                  polar_phase_rad[valid_mask] +     
                                                  2 * np.pi * calculate_frequency * source_time[isrc])) / 
                                       source_dist[valid_mask])
                    
                    # Polaritätsinvertierung (180° Phasenverschiebung)
                    if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                        wave = -wave

                # Akkumuliere Wellen (Interferenz)
                sound_field_p += wave

        # ============================================================
        # SCHRITT 4: FINALE BERECHNUNG
        # ============================================================
        # Berechne Absolutwert (Schalldruck) aus komplexer Amplitude
        sound_field_p = abs(sound_field_p)
    
        return sound_field_p, sound_field_x, sound_field_y

    def _interpolate_angle_data(self, magnitude, phase, vertical_angles, azimuth, elevation):
        """
        Interpoliert Magnitude und Phase für einen bestimmten Winkel
        
        Args:
            magnitude: Magnitude-Daten (shape: (vertical_angles, horizontal_angles) oder (vertical_angles, horizontal_angles, frequencies))
            phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form wie magnitude)
            vertical_angles: Array der vertikalen Winkel
            azimuth: Azimut-Winkel in Grad
            elevation: Elevations-Winkel in Grad
            
        Returns:
            tuple: (magnitude_value, phase_value in GRAD) oder (None, None) bei Fehler
            
        Note:
            Phase wird als unwrapped (in Grad) erwartet und kann daher linear interpoliert werden.
            Bei Verwendung in np.exp() muss die Phase mit np.radians() konvertiert werden.
        """
        try:
            # Normalisiere Azimut auf 0-360 Grad
            azimuth = azimuth % 360
            
            # Finde nächsten horizontalen Winkel (gerundet auf ganze Zahl)
            h_idx = int(round(azimuth)) % 360
            
            # Prüfe ob Daten Frequenzdimension haben (originale Daten)
            has_frequency_dim = len(magnitude.shape) == 3
            
            # Interpolation für vertikale Winkel
            if elevation <= vertical_angles[0]:
                # Elevation ist kleiner als der kleinste verfügbare Winkel
                v_idx = 0
                if has_frequency_dim:
                    # Originale Daten: Verwende mittleren Frequenzwert
                    freq_idx = magnitude.shape[2] // 2
                    mag_value = magnitude[v_idx, h_idx, freq_idx]
                    phase_value = phase[v_idx, h_idx, freq_idx]
                else:
                    # Bandgemittelte Daten: Direkter Zugriff
                    mag_value = magnitude[v_idx, h_idx]
                    phase_value = phase[v_idx, h_idx]
            elif elevation >= vertical_angles[-1]:
                # Elevation ist größer als der größte verfügbare Winkel
                v_idx = len(vertical_angles) - 1
                if has_frequency_dim:
                    # Originale Daten: Verwende mittleren Frequenzwert
                    freq_idx = magnitude.shape[2] // 2
                    mag_value = magnitude[v_idx, h_idx, freq_idx]
                    phase_value = phase[v_idx, h_idx, freq_idx]
                else:
                    # Bandgemittelte Daten: Direkter Zugriff
                    mag_value = magnitude[v_idx, h_idx]
                    phase_value = phase[v_idx, h_idx]
            else:
                # Elevation liegt zwischen zwei verfügbaren Winkeln - interpoliere
                # Finde die Indizes der umgebenden Winkel
                v_idx_lower = np.where(vertical_angles <= elevation)[0][-1]
                v_idx_upper = np.where(vertical_angles >= elevation)[0][0]
                
                if v_idx_lower == v_idx_upper:
                    # Exakter Treffer
                    if has_frequency_dim:
                        # Originale Daten: Verwende mittleren Frequenzwert
                        freq_idx = magnitude.shape[2] // 2
                        mag_value = magnitude[v_idx_lower, h_idx, freq_idx]
                        phase_value = phase[v_idx_lower, h_idx, freq_idx]
                    else:
                        # Bandgemittelte Daten: Direkter Zugriff
                        mag_value = magnitude[v_idx_lower, h_idx]
                        phase_value = phase[v_idx_lower, h_idx]
                else:
                    # Lineare Interpolation zwischen den beiden Winkeln
                    angle_lower = vertical_angles[v_idx_lower]
                    angle_upper = vertical_angles[v_idx_upper]
                    
                    # Interpolationsfaktor (0 = unterer Winkel, 1 = oberer Winkel)
                    t = (elevation - angle_lower) / (angle_upper - angle_lower)
                    
                    if has_frequency_dim:
                        # Originale Daten: Verwende mittleren Frequenzwert
                        freq_idx = magnitude.shape[2] // 2
                        mag_lower = magnitude[v_idx_lower, h_idx, freq_idx]
                        mag_upper = magnitude[v_idx_upper, h_idx, freq_idx]
                        phase_lower = phase[v_idx_lower, h_idx, freq_idx]
                        phase_upper = phase[v_idx_upper, h_idx, freq_idx]
                    else:
                        # Bandgemittelte Daten: Direkter Zugriff
                        mag_lower = magnitude[v_idx_lower, h_idx]
                        mag_upper = magnitude[v_idx_upper, h_idx]
                        phase_lower = phase[v_idx_lower, h_idx]
                        phase_upper = phase[v_idx_upper, h_idx]
                    
                    # Interpoliere Magnitude (in dB - linear interpolierbar)
                    mag_value = mag_lower + t * (mag_upper - mag_lower)
                    
                    # Interpoliere Phase (unwrapped - direkt linear interpolieren)
                    # Da Phase bereits unwrapped ist, keine zirkuläre Korrektur nötig
                    phase_value = phase_lower + t * (phase_upper - phase_lower)
            
            return mag_value, phase_value
            
        except IndexError:
            print(f"Indexfehler beim Zugriff auf Balloon-Daten bei Azimut {azimuth}° und Elevation {elevation}°")
            print(f"Array-Form: Magnitude {magnitude.shape}, Phase {phase.shape}")
            return None, None

    def _interpolate_angle_data_batch(self, magnitude, phase, vertical_angles, azimuths, elevations):
        """
        🚀 BATCH-OPTIMIERT: Interpoliert Magnitude und Phase für VIELE Winkel gleichzeitig
        
        Kernfunktion für die Performance-Optimierung der X-Achsen-Berechnung.
        Verwendet Masken-basierte Verarbeitung und np.searchsorted für vektorisierte Interpolation.
        
        Performance: ~100× schneller als Loop-basierte Einzelinterpolation
        
        Args:
            magnitude: Magnitude-Daten [vertical, horizontal] oder [vertical, horizontal, freq]
            phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form)
            vertical_angles: Array der vertikalen Winkel
            azimuths: Array von Azimut-Winkeln in Grad
            elevations: Array von Elevations-Winkeln in Grad
            
        Returns:
            tuple: (mag_values, phase_values) als Arrays oder (None, None) bei Fehler
        """
        try:
            # Normalisiere Azimute auf 0-360 Grad (vektorisiert)
            azimuths_norm = azimuths % 360
            h_indices = np.round(azimuths_norm).astype(int) % 360
            
            has_frequency_dim = len(magnitude.shape) == 3
            freq_idx = magnitude.shape[2] // 2 if has_frequency_dim else None
            
            mag_values = np.zeros_like(azimuths, dtype=np.float64)
            phase_values = np.zeros_like(azimuths, dtype=np.float64)
            
            # Punkte unterhalb kleinster Elevation → Verwende erste Messung
            mask_below = elevations <= vertical_angles[0]
            if has_frequency_dim:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below], freq_idx]
                phase_values[mask_below] = phase[0, h_indices[mask_below], freq_idx]
            else:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below]]
                phase_values[mask_below] = phase[0, h_indices[mask_below]]
            
            # Punkte oberhalb größter Elevation → Verwende letzte Messung
            mask_above = elevations >= vertical_angles[-1]
            v_max = len(vertical_angles) - 1
            if has_frequency_dim:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above], freq_idx]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above], freq_idx]
            else:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above]]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above]]
            
            # Punkte ZWISCHEN zwei Messungen → Lineare Interpolation
            mask_interp = ~(mask_below | mask_above)
            
            if np.any(mask_interp):
                # Extrahiere zu interpolierende Werte
                elev_interp = elevations[mask_interp]
                h_idx_interp = h_indices[mask_interp]
                
                # 🚀 Finde umgebende Indizes für ALLE Punkte gleichzeitig
                v_idx_lower = np.searchsorted(vertical_angles, elev_interp, side='right') - 1
                v_idx_upper = v_idx_lower + 1
                
                v_idx_lower = np.clip(v_idx_lower, 0, len(vertical_angles) - 1)
                v_idx_upper = np.clip(v_idx_upper, 0, len(vertical_angles) - 1)
                
                angle_lower = vertical_angles[v_idx_lower]
                angle_upper = vertical_angles[v_idx_upper]
                
                # Berechne Interpolationsfaktoren (vektorisiert)
                t = (elev_interp - angle_lower) / (angle_upper - angle_lower + 1e-10)
                t = np.clip(t, 0, 1)
                
                if has_frequency_dim:
                    mag_lower = magnitude[v_idx_lower, h_idx_interp, freq_idx]
                    mag_upper = magnitude[v_idx_upper, h_idx_interp, freq_idx]
                    phase_lower = phase[v_idx_lower, h_idx_interp, freq_idx]
                    phase_upper = phase[v_idx_upper, h_idx_interp, freq_idx]
                else:
                    mag_lower = magnitude[v_idx_lower, h_idx_interp]
                    mag_upper = magnitude[v_idx_upper, h_idx_interp]
                    phase_lower = phase[v_idx_lower, h_idx_interp]
                    phase_upper = phase[v_idx_upper, h_idx_interp]
                
                # 🚀 Lineare Interpolation für ALLE Punkte gleichzeitig
                # value = lower + t × (upper - lower)
                mag_values[mask_interp] = mag_lower + t * (mag_upper - mag_lower)
                phase_values[mask_interp] = phase_lower + t * (phase_upper - phase_lower)
            
            return mag_values, phase_values
            
        except Exception as e:
            print(f"Fehler in Batch-Interpolation: {e}")
            return None, None

    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container