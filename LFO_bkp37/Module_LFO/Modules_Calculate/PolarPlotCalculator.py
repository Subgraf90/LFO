import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase


class PolarPlotCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_polar):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_polar = calculation_polar
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt
        
        # Keine Default-Initialisierung mehr - nur initialisieren wenn calculate_polar_pressure() aufgerufen wird
    
    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container

    def _interpolate_balloon_batch(self, balloon_mag, balloon_phase, balloon_freqs, vertical_angles, 
                                   azimuths, elevations, target_freq):
        """
        🚀 BATCH-OPTIMIERT: Interpoliert Balloon-Daten für VIELE Winkel gleichzeitig
        
        Verwendet Advanced NumPy Indexing um ALLE Werte in einer Operation zu holen.
        Statt 360 Einzelzugriffe → 1 vektorisierter Zugriff!
        
        Performance: ~100× schneller als Loop-basierte Einzelinterpolation
        
        Args:
            balloon_mag: 3D-Array [vertical, horizontal, freq] oder 2D [vertical, horizontal] (Terzband)
            balloon_phase: 3D-Array [vertical, horizontal, freq] oder 2D [vertical, horizontal] (Terzband)
            balloon_freqs: 1D-Array der Frequenzen
            vertical_angles: 1D-Array der vertikalen Winkel
            azimuths: 1D-Array von Azimut-Winkeln in Grad
            elevations: 1D-Array von Elevations-Winkeln in Grad
            target_freq: Zielfrequenz in Hz
            
        Returns:
            tuple: (gains, phases) als 1D-Arrays
        """
        # 🎯 TERZBAND-OPTIMIERUNG: Wenn bereits 2D (Terzband-Mittelung), keine Frequenz-Suche nötig
        if balloon_mag.ndim == 2 and balloon_phase.ndim == 2:
            # Daten sind bereits gemittelt (2D statt 3D)
            # Normalisiere Azimute und finde horizontale Indizes (vektorisiert)
            h_indices = np.round(azimuths).astype(int) % 360
            
            # 🚀 VEKTORISIERT: Finde vertikale Indizes für ALLE Elevationen gleichzeitig
            elev_diffs = np.abs(elevations[:, None] - vertical_angles[None, :])
            v_indices = np.argmin(elev_diffs, axis=1)
            
            # 🚀 ADVANCED INDEXING: Hole ALLE Werte gleichzeitig (2D-Zugriff)
            gains = balloon_mag[v_indices, h_indices]
            phases = balloon_phase[v_indices, h_indices]
            
            return gains, phases
        
        # ❌ FALLBACK: Original-Verfahren mit Nearest Neighbor (falls keine Terzband-Daten)
        # Finde Frequenz-Index (einmal für alle Winkel)
        f_idx = np.abs(balloon_freqs - target_freq).argmin()
        
        # Normalisiere Azimute und finde horizontale Indizes (vektorisiert)
        h_indices = np.round(azimuths).astype(int) % 360
        
        # 🚀 VEKTORISIERT: Finde vertikale Indizes für ALLE Elevationen gleichzeitig
        # Broadcasting: elevations[:, None] - vertical_angles[None, :] = Matrix[360, n_vertical]
        elev_diffs = np.abs(elevations[:, None] - vertical_angles[None, :])
        v_indices = np.argmin(elev_diffs, axis=1)  # Finde min pro Zeile
        
        # 🚀 ADVANCED INDEXING: Hole ALLE Werte gleichzeitig
        # Ein einziger Array-Zugriff für alle 360 Punkte!
        gains = balloon_mag[v_indices, h_indices, f_idx]
        phases = balloon_phase[v_indices, h_indices, f_idx]
        
        return gains, phases

    def calculate_polar_pressure(self):
        try:
            # Prüfe ob es aktive Quellen gibt
            has_active_sources = any(not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values())
            
            # Wenn keine aktiven Quellen, setze leere Daten
            if not has_active_sources:
                self.calculation_polar.update({
                    'sound_field_p': {},
                    'angles': None,
                    'frequencies': self.settings.polar_frequencies
                })
                return
            
            # Initialisiere die Struktur wenn noch nicht vorhanden
            if not self.calculation_polar:
                self.calculation_polar.update({
                    'sound_field_p': {},
                    'angles': None,
                    'frequencies': self.settings.polar_frequencies
                })
            
            # Hole Frequenzen aus Settings
            frequencies = self.settings.polar_frequencies
            
            # Dictionary für die Ergebnisse
            results = {}
            
            for color, freq in frequencies.items():
                sound_field_p, angles = self.calculate_polar(float(freq))
                
                # Konvertiere und normalisiere
                sound_field_p = np.asarray(sound_field_p, dtype=np.float64)
                angles = np.asarray(angles, dtype=np.float64)
                sound_field_p = self.normalize_values(sound_field_p)
                
                # Speichere Ergebnisse
                results[float(freq)] = sound_field_p  # Frequenz als float als Key
                
                # Speichere Winkel einmal
                if self.calculation_polar.get('angles') is None:
                    self.calculation_polar['angles'] = angles
            
            # Aktualisiere calculation_polar
            self.calculation_polar['sound_field_p'] = results
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()


    def calculate_dborder(self):
        try:
            max_length = 0
            for speaker_array in self.settings.speaker_arrays.values():
                if not speaker_array.mute and not speaker_array.hide:
                    max_length = max(max_length, speaker_array.source_length)
            
            if max_length == 0:
                return 50

            f = max(self.calculation_polar['frequencies'].values())
            h = max_length

            dborder = self.functions.frauenhofer_fresnel_zone(f, h)
            
            if np.isnan(dborder):
                return 50
            
            return dborder
            
        except Exception as e:
            return 50

    def find_nearest_frequency(self, target_freq):
        """Findet die nächstliegende verfügbare Frequenz in den Rohdaten"""
        if 'raw_freqs' not in self.data or not self.data['raw_freqs']:
            return target_freq
        
        raw_freqs = self.data['raw_freqs'][0]  # Nehme die Frequenzen des ersten Lautsprechers
        idx = np.abs(np.array(raw_freqs) - target_freq).argmin()
        return raw_freqs[idx]

    def calculate_polar(self, frequency):
        """
        🚀 VEKTORISIERT: Berechnet Polar-Plot für eine spezifische Frequenz
        
        Berechnet den Schalldruck auf einem Kreis (alle 360° Winkel).
        Nutzt Batch-Interpolation für optimale Performance.
        
        Args:
            frequency: Zielfrequenz in Hz
            
        Returns:
            tuple: (sound_field_p, angles) - Schalldruck und Winkel-Array
        """
        try:
            # ============================================================
            # SCHRITT 1: Initialisierung
            # ============================================================
            actual_freq = self.find_nearest_frequency(frequency)
            
            # Erstelle Winkel-Array (0° - 359°, 360 Punkte)
            angles = np.arange(0, 360, 1, dtype=np.float64)
            angles_rad = np.deg2rad(angles)
            
            # Initialisiere komplexes Schallfeld-Array
            sound_field_p = np.zeros(len(angles), dtype=np.complex128)
            
            # ============================================================
            # SCHRITT 2: Physikalische Konstanten
            # ============================================================
            # 🌡️ Temperaturabhängige Schallgeschwindigkeit (wird in UiSettings berechnet)
            speed_of_sound = self.settings.speed_of_sound
            wave_number = 2 * np.pi * actual_freq / speed_of_sound
            
            # Berechne Messradius (aus Fraunhofer-Distanz)
            dborder = self.calculate_dborder()
            radius = max(20, dborder/3 + dborder)  # Mindestens 20m
            
            # Referenz-Schalldruck
            safe_a_source_db = np.clip(self.settings.a_source_db, -200, 200)
            a_source_pa = self.functions.db2spl(self.functions.db2mag(safe_a_source_db))
            use_air_absorption = getattr(self.settings, "use_air_absorption", False)
            air_absorption_coeff = 0.0
            if use_air_absorption:
                air_absorption_coeff = self.functions.calculate_air_absorption(
                    actual_freq,
                    getattr(self.settings, "temperature", 20.0),
                    getattr(self.settings, "humidity", 50.0),
                )

            # ============================================================
            # SCHRITT 3: VEKTORISIERTE POLAR-KOORDINATEN
            # ============================================================
            # Berechne kartesische Koordinaten für ALLE 360 Winkel gleichzeitig
            x_points = radius * np.cos(angles_rad)
            y_points = radius * np.sin(angles_rad)
            
            # ============================================================
            # SCHRITT 4: Iteriere über alle Lautsprecher-Arrays
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
                
                source_azimuth = np.deg2rad(speaker_array.source_azimuth)
                source_delay = speaker_array.delay
                source_time = [t + source_delay for t in speaker_array.source_time]
                source_time = [t / 1000 for t in source_time]
                source_gain = np.clip(speaker_array.gain, -100, 100)
                source_level = np.clip(np.array(speaker_array.source_level) + source_gain, -100, 100)
                source_level = self.functions.db2mag(source_level)

                # ============================================================
                # SCHRITT 5: Iteriere über alle Lautsprecher im Array
                # ============================================================
                for isrc in range(len(source_indices)):
                    speaker_name = speaker_array.source_polar_pattern[isrc]
                    
                    # --------------------------------------------------------
                    # 5.1: VEKTORISIERTE GEOMETRIE-BERECHNUNG
                    # --------------------------------------------------------
                    # Berechne Distanzen für ALLE 360 Winkel gleichzeitig
                    x_distances = x_points - source_position_x[isrc]
                    y_distances = y_points - source_position_y[isrc]
                    
                    # Horizontale Distanzen (Pythagoras in 2D)
                    horizontal_dists = np.sqrt(x_distances**2 + y_distances**2)
                    
                    # Z-Distanz (konstant für alle Punkte)
                    z_distance = -source_position_z[isrc]
                    
                    # 3D-Distanzen (Pythagoras in 3D)
                    source_dists = np.sqrt(horizontal_dists**2 + z_distance**2)
                    
                    # --------------------------------------------------------
                    # 5.2: VEKTORISIERTE WINKEL-BERECHNUNG
                    # --------------------------------------------------------
                    # Berechne Azimut für ALLE Winkel gleichzeitig
                    source_to_point_angles = np.arctan2(y_distances, x_distances)
                    azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                    azimuths = (360 - azimuths) % 360  # Invertiere
                    azimuths = (azimuths + 90) % 360  # Drehe 90°
                    
                    # Elevationen (vektorisiert)
                    elevations = np.degrees(np.arctan2(z_distance, horizontal_dists))
                    
                    # Distanz-Maske (filtert zu nahe Punkte)
                    valid_mask = source_dists >= 0.001
                    
                    # --------------------------------------------------------
                    # 5.3: BATCH-INTERPOLATION (Performance-Booster! 🚀)
                    # --------------------------------------------------------
                    # 🎯 OPTIMIERUNG: Versuche zuerst Terzband-Daten zu laden
                    polar_data_key = f'balloon_polar_{actual_freq}Hz_{speaker_name}'
                    balloon_data = self._data_container.get_data().get(polar_data_key, None)
                    
                    # Falls keine Terzband-Daten existieren, lade die Original-Daten
                    if balloon_data is None:
                        balloon_data = self._data_container.get_balloon_data(speaker_name)
                    
                    if balloon_data is not None and isinstance(balloon_data, dict):
                        balloon_vertical_angles = balloon_data.get('vertical_angles')
                        balloon_mag = balloon_data.get('magnitude')
                        balloon_phase = balloon_data.get('phase')
                        balloon_freqs = balloon_data.get('freqs')
                        
                        if (balloon_vertical_angles is not None and 
                            balloon_mag is not None and 
                            balloon_phase is not None and
                            balloon_freqs is not None):
                            try:
                                # 🚀 BATCH: Hole Balloon-Daten für ALLE 360 Winkel auf einmal
                                balloon_freqs_array = np.array(balloon_freqs) if isinstance(balloon_freqs, list) else balloon_freqs
                                
                                polar_gains, polar_phases = self._interpolate_balloon_batch(
                                    balloon_mag, balloon_phase, balloon_freqs_array,
                                    balloon_vertical_angles, azimuths, elevations, actual_freq
                                )
                                
                                # --------------------------------------------------------
                                # 5.4: VEKTORISIERTE WELLENBERECHNUNG
                                # --------------------------------------------------------
                                # Konvertiere dB zu linear (vektorisiert)
                                magnitude_linear = 10 ** (polar_gains / 20)
                                polar_phase_rad = np.radians(polar_phases)
                                
                                # Berechne Welle für ALLE gültigen Punkte gleichzeitig
                                # Wellenformel: p(r) = (A × G × A₀ / r) × exp(i × Φ)
                                wave = np.zeros(len(angles), dtype=np.complex128)
                                attenuation = 1.0
                                if air_absorption_coeff > 0.0:
                                    attenuation = np.exp(-air_absorption_coeff * source_dists[valid_mask])
                                wave[valid_mask] = (magnitude_linear[valid_mask] * source_level[isrc] * a_source_pa *
                                                   attenuation *
                                                   np.exp(1j * (wave_number * source_dists[valid_mask] + 
                                                              polar_phase_rad[valid_mask] + 
                                                              2 * np.pi * actual_freq * source_time[isrc])) / 
                                                   source_dists[valid_mask])
                                
                                # Polaritätsinvertierung (180° Phasenverschiebung)
                                if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                                    wave = -wave
                                
                                # Akkumuliere (Interferenz)
                                sound_field_p += wave
                                    
                            except Exception as e:
                                print(f"Fehler bei Balloon-Daten: {e}")
                                continue

            # ============================================================
            # SCHRITT 6: FINALE BERECHNUNG
            # ============================================================
            # Berechne Absolutwert (Schalldruck)
            return np.abs(sound_field_p), angles
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return np.zeros(360), np.arange(0, 360, 1)

    def normalize_values(self, values):
        """
        Normalisiert die Werte auf 0 dB Maximum
        """
        try:
            
            # Ersetze ungültige Werte
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(values) == 0 or np.max(values) <= 0:
                return np.zeros_like(values)
            
            # Finde Maximum
            max_value = np.max(values)
            
            # Berechne dB-Werte relativ zum Maximum
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = 20 * np.log10(values / max_value)
            
            # Ersetze -inf durch -30 und begrenze auf -30 bis 0 dB
            normalized = np.nan_to_num(normalized, nan=-30.0, neginf=-30.0)
            normalized = np.clip(normalized, -30, 0)
            
            return normalized
            
        except Exception as e:
            print(f"Error in normalize_values: {str(e)}")
            return np.zeros_like(values)

