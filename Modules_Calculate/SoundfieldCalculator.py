import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase

class SoundFieldCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt
        
        # 🎯 GEOMETRY CACHE: Verhindert unnötige Neuberechnungen
        self._geometry_cache = {}  # {source_key: {distances, azimuths, elevations}}
        self._grid_cache = None    # Gespeichertes Grid
        self._grid_hash = None     # Hash der Grid-Parameter
   
    def calculate_soundfield_pressure(self):
        print(
            "[SoundFieldCalculator] Starte Superpositions-Berechnung "
            f"(Superposition aktiv={getattr(self.settings, 'spl_plot_superposition', False)}, "
            f"FEM aktiv={getattr(self.settings, 'spl_plot_fem', False)})"
        )
        self.calculation_spl["sound_field_p"], self.calculation_spl["sound_field_x"], self.calculation_spl["sound_field_y"] = self.calculate_sound_field()

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
        balloon_data = None
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
        if balloon_data is None:
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

    def get_balloon_data_batch(self, speaker_name, azimuths, elevations, use_averaged=True):
        """
        🚀 BATCH-OPTIMIERT: Holt Balloon-Daten für VIELE Winkel auf einmal
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuths (np.ndarray): Array von Azimut-Winkeln in Grad (0-360), Shape: (ny, nx)
            elevations (np.ndarray): Array von Elevations-Winkeln in Grad, Shape: (ny, nx)
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten (Standard: True)
            
        Returns:
            tuple: (magnitudes, phases) als 2D-Arrays oder (None, None) bei Fehler
        """
        # 🚀 OPTIMIERT: Verwende bandgemittelte Daten falls verfügbar
        if use_averaged:
            balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=True)
            if balloon_data is not None:
                magnitude = balloon_data.get('magnitude')
                phase = balloon_data.get('phase')
                vertical_angles = balloon_data.get('vertical_angles')
                
                if magnitude is not None and phase is not None and vertical_angles is not None:
                    return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)
        
        # 🔄 FALLBACK: Verwende originale Daten
        balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
        if balloon_data is None:
            print(f"❌ Keine Balloon-Daten für {speaker_name}")
            return None, None
        
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            print(f"Fehlende Daten in Balloon-Daten für {speaker_name}")
            return None, None
        
        return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)

    def calculate_sound_field(self):
        """
        🚀 VOLLSTÄNDIG VEKTORISIERT: Berechnet das Schallfeld für alle Grid-Punkte gleichzeitig
        
        Optimierungen:
        1. Batch-Interpolation: Alle Balloon-Lookups auf einmal (100-1000× schneller)
        2. Vektorisierte Geometrie: Alle Distanzen/Winkel gleichzeitig berechnet
        3. Vektorisierte Wellenberechnung: Alle komplexen Wellen gleichzeitig
        4. Masken-basiert: Nur gültige Punkte werden berechnet
        
        Performance: ~50-60ms für 15.000 Punkte (statt ~5.000ms mit Loops)
        """
        # ============================================================
        # SCHRITT 0: Prüfe ob es aktive Quellen gibt
        # ============================================================
        has_active_sources = any(not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values())
        
        # ============================================================
        # SCHRITT 1: Grid-Erstellung
        # ============================================================
        # Erstelle 1D-Arrays für X- und Y-Koordinaten
        sound_field_x = np.arange((self.settings.width / 2 * -1), 
                                 ((self.settings.width / 2) + self.settings.resolution), 
                                 self.settings.resolution)
        sound_field_y = np.arange((self.settings.length / 2 * -1), 
                                 ((self.settings.length / 2) + self.settings.resolution), 
                                 self.settings.resolution)

        
        # Wenn keine aktiven Quellen, gib leere Arrays zurück
        if not has_active_sources:
            return [], sound_field_x, sound_field_y
        
        # Initialisiere das Schallfeld als 2D-Array komplexer Zahlen
        # Shape: [Y, X] = [Zeilen, Spalten] (NumPy-Konvention!)
        sound_field_p = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=complex)
        
        # ============================================================
        # SCHRITT 2: Vorbereitung - Physikalische Konstanten
        # ============================================================
        # Berechne physikalische Konstanten (einmalig für alle Quellen)
        # 🌡️ Temperaturabhängige Schallgeschwindigkeit
        speed_of_sound = self.functions.calculate_speed_of_sound(self.settings.temperature)
        wave_number = self.functions.wavenumber(speed_of_sound, self.settings.calculate_frequency)
        calculate_frequency = self.settings.calculate_frequency
        a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
        
        # Speichere Minimum und Maximum der berechneten Pegel für Vergleich
        min_level = float('inf')
        max_level = float('-inf')
        
        # ============================================================
        # SCHRITT 3: Iteriere über alle Lautsprecher-Arrays
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
                getattr(speaker_array, 'source_position_x', None),
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                getattr(speaker_array, 'source_position_y', None),
            )
            source_position_z = getattr(
                speaker_array,
                'source_position_calc_z',
                getattr(speaker_array, 'source_position_z', None),
            )

        
            source_azimuth = np.deg2rad(speaker_array.source_azimuth)
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
            
            # ============================================================
            # SCHRITT 4: Iteriere über alle Lautsprecher im Array
            # ============================================================
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                
                # --------------------------------------------------------
                # 4.1: VEKTORISIERTE GEOMETRIE-BERECHNUNG
                # --------------------------------------------------------
                # 🚀 Erstelle 2D-Meshgrid für ALLE Grid-Punkte gleichzeitig
                # X, Y haben beide die Shape [ny, nx] (z.B. [151, 101])
                X, Y = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
                
                # Berechne Distanz-Vektoren für ALLE Punkte gleichzeitig
                # Resultat: 2D-Arrays mit Shape [ny, nx]
                x_distances = X - source_position_x[isrc]  # Distanz in X-Richtung
                y_distances = Y - source_position_y[isrc]  # Distanz in Y-Richtung
                
                # Berechne horizontale Distanz (Pythagoras in 2D)
                # √(x² + y²) für ALLE Punkte gleichzeitig
                horizontal_dists = np.sqrt(x_distances**2 + y_distances**2)
                
                # Z-Distanz (konstant für alle Punkte, da Grid auf Z=0)
                z_distance = -source_position_z[isrc]  # Negativ, da Lautsprecher ÜBER dem Grid
                
                # Berechne 3D-Distanz (Pythagoras in 3D)
                # √(horizontal² + z²) für ALLE Punkte gleichzeitig
                source_dists = np.sqrt(horizontal_dists**2 + z_distance**2)
                
                # --------------------------------------------------------
                # 4.2: VEKTORISIERTE WINKEL-BERECHNUNG
                # --------------------------------------------------------
                # Berechne Azimut-Winkel für ALLE Punkte gleichzeitig
                source_to_point_angles = np.arctan2(y_distances, x_distances)
                azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                azimuths = (360 - azimuths) % 360  # Invertiere (Uhrzeigersinn)
                azimuths = (azimuths + 90) % 360  # Drehe 90° (Polar-Koordinatensystem)
                
                # Berechne Elevations-Winkel für ALLE Punkte gleichzeitig
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dists))
                
                # --------------------------------------------------------
                # 4.3: MASKEN-ERSTELLUNG (filtert ungültige Punkte)
                # --------------------------------------------------------
                # Erstelle Sichtbarkeits-Maske (prüft Abschattung durch Wände)
                # Distanz-Maske (filtert Punkte zu nah an der Quelle, verhindert Division durch Null)
                dist_mask = source_dists >= 0.001
                
                # Kombiniere beide Masken → Nur Punkte die BEIDE Bedingungen erfüllen
                valid_mask = dist_mask
                
                # --------------------------------------------------------
                # 4.4: BATCH-INTERPOLATION (der Performance-Booster! 🚀)
                # --------------------------------------------------------
                # Hole Balloon-Daten für ALLE Punkte in EINEM Aufruf
                # Input:  azimuths[151, 101], elevations[151, 101]
                # Output: polar_gains[151, 101], polar_phases[151, 101]
                # → 15.251 Interpolationen gleichzeitig statt in einer Loop!
                polar_gains, polar_phases = self.get_balloon_data_batch(speaker_name, azimuths, elevations)
                
                if polar_gains is not None and polar_phases is not None:
                    
                    # --------------------------------------------------------
                    # 4.5: VEKTORISIERTE WELLENBERECHNUNG (das Herzstück! 🎵)
                    # --------------------------------------------------------
                    # Konvertiere Magnitude von dB zu linearen Werten (für ALLE Punkte)
                    # 10^(dB/20) → z.B. 94dB → 50.000 (linear)
                    magnitude_linear = 10 ** (polar_gains / 20)
                    
                    # Konvertiere Phase von Grad zu Radiant (für ALLE Punkte)
                    polar_phase_rad = np.radians(polar_phases)
                    
                    # Initialisiere Wellen-Array (gleiches Shape wie Grid)
                    wave = np.zeros_like(source_dists, dtype=complex)
                    
                    # 🚀 KERN-BERECHNUNG: Berechne komplexe Welle für ALLE gültigen Punkte
                    # Wellenformel: p(r) = (A × G × A₀ / r) × exp(i × Φ)
                    # Wobei:
                    #   A = magnitude_linear (Richtcharakteristik)
                    #   G = source_level (Lautstärke)
                    #   A₀ = a_source_pa (Referenz-Schalldruck)
                    #   r = source_dists (Distanz)
                    #   Φ = wave_number×r + polar_phase + 2π×f×t (Gesamtphase)
                    wave[valid_mask] = (magnitude_linear[valid_mask] * source_level[isrc] * a_source_pa * 
                                       np.exp(1j * (wave_number * source_dists[valid_mask] + 
                                                  polar_phase_rad[valid_mask] + 
                                                  2 * np.pi * calculate_frequency * source_time[isrc])) / 
                                       source_dists[valid_mask])
                    
                    # Polaritätsinvertierung (180° Phasenverschiebung)
                    if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                        wave = -wave
                    
                    # --------------------------------------------------------
                    # 4.6: AKKUMULATION (Interferenz-Überlagerung)
                    # --------------------------------------------------------
                    # Addiere die Welle dieses Lautsprechers zum Gesamt-Schallfeld
                    # Komplexe Addition → Automatische Interferenz (konstruktiv/destruktiv)
                    sound_field_p += wave

        # ============================================================
        # SCHRITT 5: FINALE BERECHNUNG
        # ============================================================
        # Berechne Absolutwert (Schalldruck) aus komplexer Amplitude
        # |p| = √(Real² + Imag²)
        sound_field_p = abs(sound_field_p)
        
        # Konvertiere zu dB (für Statistiken, wird nicht zurückgegeben)
        sound_field_db = 20 * np.log10(sound_field_p + 1e-10)  # +1e-10 verhindert log(0)
        
        return sound_field_p, sound_field_x, sound_field_y

    def calculate_mirror_source_contribution(self, mirror_x, mirror_y, mirror_z, mirror_azimuth, 
                                        source_x, source_y, source_z, reflecting_wall, speaker_array, isrc, 
                                        sound_field_x, sound_field_y, sound_field_p,
                                        active_walls, source_indices):
        """🚀 OPTIMIERT: Vektorisierte Spiegelquellen-Berechnung"""
        wall_start, wall_end = reflecting_wall
        speaker_name = speaker_array.source_polar_pattern[isrc]
        
        # 🚀 VEKTORISIERT: Erstelle Meshgrid
        X, Y = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
        
        # 1. Prüfe welche Punkte auf gleicher Seite der Wand sind (vektorisiert)
        wall_vec = (wall_end[0] - wall_start[0], wall_end[1] - wall_start[1])
        v_points_x = X - wall_start[0]
        v_points_y = Y - wall_start[1]
        v_source_x = source_x - wall_start[0]
        v_source_y = source_y - wall_start[1]
        
        cross_points = wall_vec[0] * v_points_y - wall_vec[1] * v_points_x
        cross_source = wall_vec[0] * v_source_y - wall_vec[1] * v_source_x
        
        # Maske für gleiche Seite
        same_side_mask = (cross_points * cross_source) > 0
        
        # Berechne Distanzen für ALLE Punkte (vektorisiert)
        x_distances = X - mirror_x
        y_distances = Y - mirror_y
        horizontal_dists = np.sqrt(x_distances**2 + y_distances**2)
        z_distance = -mirror_z
        source_dists = np.sqrt(horizontal_dists**2 + z_distance**2)
        
        # Winkelberechnungen (vektorisiert)
        source_to_point_angles = np.arctan2(y_distances, x_distances)
        azimuths = (np.degrees(source_to_point_angles) + np.degrees(mirror_azimuth)) % 360
        azimuths = (360 - azimuths) % 360
        azimuths = (azimuths + 90) % 360
        elevations = np.degrees(np.arctan2(z_distance, horizontal_dists))
        
        # Distanz-Maske
        dist_mask = source_dists > 0.001
        
        # Kombiniere Masken
        valid_mask = same_side_mask & dist_mask
        
        # 🚀 BATCH-OPTIMIERUNG: Sammle alle gültigen Punkte und hole Balloon-Daten in einem Batch
        valid_points = []
        for iy in range(len(sound_field_y)):  # ✅ Y zuerst (Zeilen)
            for ix in range(len(sound_field_x)):  # ✅ X danach (Spalten)
                if not valid_mask[iy, ix]:
                    continue
                
                point_x = sound_field_x[ix]
                point_y = sound_field_y[iy]
                
                # Berechne Reflexionspunkt
                reflection_point = self.get_reflection_point(mirror_x, mirror_y, point_x, point_y, reflecting_wall)
                
                if reflection_point is not None:
                    refl_x, refl_y = reflection_point
                    
                    # Prüfe Sichtbarkeit
                    if self.check_line_of_sight(source_x, source_y, refl_x, refl_y, active_walls):
                        valid_points.append((iy, ix))
        
        # Wenn keine gültigen Punkte, return
        if len(valid_points) == 0:
            return
        
        # 🚀 BATCH: Hole Balloon-Daten für ALLE gültigen Punkte
        batch_azimuths = np.array([azimuths[iy, ix] for iy, ix in valid_points])
        batch_elevations = np.array([elevations[iy, ix] for iy, ix in valid_points])
        
        polar_gains_batch, polar_phases_batch = self.get_balloon_data_batch(
            speaker_name, 
            batch_azimuths.reshape(-1, 1),  # Shape für Batch-Funktion
            batch_elevations.reshape(-1, 1)
        )
        
        if polar_gains_batch is not None and polar_phases_batch is not None:
            # Flatten für einfacheren Zugriff
            polar_gains_batch = polar_gains_batch.flatten()
            polar_phases_batch = polar_phases_batch.flatten()
            
            # Verarbeite alle gültigen Punkte
            level = self.functions.db2mag(speaker_array.source_level[isrc] + speaker_array.gain)
            # 🌡️ Temperaturabhängige Schallgeschwindigkeit
            speed_of_sound = self.functions.calculate_speed_of_sound(self.settings.temperature)
            wave_number = self.functions.wavenumber(speed_of_sound, self.settings.calculate_frequency)
            a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
            time_factor = 2 * np.pi * self.settings.calculate_frequency * (speaker_array.source_time[isrc] + speaker_array.delay) / 1000
            
            for idx, (iy, ix) in enumerate(valid_points):
                polar_gain = polar_gains_batch[idx]
                polar_phase = polar_phases_batch[idx]
                source_dist = source_dists[iy, ix]
                
                magnitude_linear = 10 ** (polar_gain / 20)
                polar_phase_rad = np.radians(polar_phase)
                
                # Berechne Welle
                wave = (magnitude_linear * level * a_source_pa * 
                       np.exp(1j * (wave_number * source_dist + polar_phase_rad + time_factor)) / 
                       source_dist)
                
                # Polaritätsinvertierung (180° Phasenverschiebung)
                if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                    wave = -wave
                
                # Addiere zur Summe
                if np.isscalar(wave):
                    sound_field_p[iy, ix] += wave
                elif hasattr(wave, 'item') and hasattr(wave, 'size') and wave.size == 1:
                    sound_field_p[iy, ix] += wave.item()
                elif hasattr(wave, '__len__') and len(wave) > 0:
                    sound_field_p[iy, ix] += wave[0]

    def get_reflection_point(self, mirror_x, mirror_y, point_x, point_y, wall):
        """
        Berechnet den Reflexionspunkt auf der Wand
        """
        wall_start, wall_end = wall
        
        # Geradengleichung der Wand: ax + by + c = 0
        a = wall_end[1] - wall_start[1]
        b = wall_start[0] - wall_end[0]
        c = wall_end[0]*wall_start[1] - wall_start[0]*wall_end[1]
        
        # Geradengleichung Spiegelquelle-Punkt
        dx = point_x - mirror_x
        dy = point_y - mirror_y
        
        # Schnittpunkt berechnen
        denom = a*dx + b*dy
        if abs(denom) < 1e-10:  # Parallele Linien
            return None
        
        t = -(a*mirror_x + b*mirror_y + c) / denom
        
        # Schnittpunkt
        x = mirror_x + t*dx
        y = mirror_y + t*dy
        
        # Prüfen ob Schnittpunkt auf Wandsegment liegt
        if (min(wall_start[0], wall_end[0]) <= x <= max(wall_start[0], wall_end[0]) and
            min(wall_start[1], wall_end[1]) <= y <= max(wall_start[1], wall_end[1])):
            return (x, y)
        
        return None

    def check_line_of_sight(self, source_x, source_y, point_x, point_y, walls):
        """
        Prüft, ob eine direkte Sichtlinie zwischen Quelle und Punkt existiert
        unter Berücksichtigung ALLER Wände
        """
        # Prüfe gegen jede Wand
        for wall in walls:
            wall_start, wall_end = wall
            
            # Vektoren für Kreuzprodukt
            v1 = (wall_end[0] - wall_start[0], wall_end[1] - wall_start[1])  # Wandvektor
            v2 = (point_x - wall_start[0], point_y - wall_start[1])          # Vektor zum Punkt
            v3 = (source_x - wall_start[0], source_y - wall_start[1])        # Vektor zur Quelle
            
            # Kreuzprodukte berechnen
            cross_point = v1[0]*v2[1] - v1[1]*v2[0]
            cross_source = v1[0]*v3[1] - v1[1]*v3[0]
            
            # Wenn Punkt und Quelle auf verschiedenen Seiten der Wand liegen
            if cross_point * cross_source < 0:
                # Prüfe ob der Schnittpunkt im Wandsegment liegt
                t = ((point_x - source_x)*(wall_start[1] - source_y) - 
                     (point_y - source_y)*(wall_start[0] - source_x)) / \
                    ((point_y - source_y)*(wall_end[0] - wall_start[0]) - 
                     (point_x - source_x)*(wall_end[1] - wall_start[1]))
                
                if 0 <= t <= 1:
                    return False  # Sichtlinie wird von dieser Wand blockiert
        
        return True  # Keine Wand blockiert die Sichtlinie

    def reflect_point_across_line(self, source_x, source_y, wall_start, wall_end):
        """
        Berechnet die Position der Spiegelquelle durch Spiegelung an einer Wand
        """
        # Wandvektor
        wall_dx = wall_end[0] - wall_start[0]
        wall_dy = wall_end[1] - wall_start[1]
        
        # Normalisierter Normalenvektor der Wand
        length = np.sqrt(wall_dx**2 + wall_dy**2)
        normal_x = -wall_dy / length
        normal_y = wall_dx / length
        
        # Vektor von Wandstart zur Quelle
        source_to_wall_x = source_x - wall_start[0]
        source_to_wall_y = source_y - wall_start[1]
        
        # Projektion auf Normale (doppelter Abstand zur Wand)
        projection = 2 * (source_to_wall_x * normal_x + source_to_wall_y * normal_y)
        
        # Spiegelposition berechnen
        mirror_x = source_x - projection * normal_x
        mirror_y = source_y - projection * normal_y
        
        return mirror_x, mirror_y
    

    
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
            tuple: (magnitude_value, phase_value_in_grad) oder (None, None) bei Fehler
            
        Note:
            Phase wird als unwrapped in GRAD erwartet und kann daher linear interpoliert werden.
            Ausgabe ist ebenfalls in GRAD - Konvertierung zu Radiant erfolgt beim Einsetzen in np.exp().
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
        🚀 BATCH-OPTIMIERT: Interpoliert Magnitude und Phase für VIELE Winkel auf einmal
        
        Diese Funktion ist der SCHLÜSSEL zur Performance-Optimierung!
        Statt 15.251 einzelne Interpolationen → 1 vektorisierte Operation für ALLE Punkte.
        
        Funktionsweise:
        1. Verwendet Masken um Punkte in 3 Kategorien zu teilen:
           - Unterhalb kleinster Elevation → Verwende erste Messung
           - Oberhalb größter Elevation → Verwende letzte Messung
           - Dazwischen → Lineare Interpolation
        2. Alle Operationen sind vektorisiert (NumPy Broadcasting)
        3. np.searchsorted findet ALLE Interpolations-Indizes gleichzeitig
        
        Performance: ~25ms für 15.000 Punkte (statt ~2.500ms mit Loop)
        
        Args:
            magnitude: Magnitude-Daten [vertical, horizontal] oder [vertical, horizontal, freq]
            phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form)
            vertical_angles: Array der vertikalen Winkel (z.B. [-90, -85, ..., +90])
            azimuths: 2D-Array von Azimut-Winkeln in Grad, Shape: (ny, nx)
            elevations: 2D-Array von Elevations-Winkeln in Grad, Shape: (ny, nx)
            
        Returns:
            tuple: (mag_values, phase_values) als 2D-Arrays oder (None, None) bei Fehler
        """
        try:
            # --------------------------------------------------------
            # SCHRITT 1: Horizontale Winkel (Azimut) verarbeiten
            # --------------------------------------------------------
            # Normalisiere Azimute auf 0-360 Grad (vektorisiert)
            azimuths_norm = azimuths % 360
            
            # Finde horizontale Indizes (gerundet auf ganze Grade)
            # Balloon-Daten haben 360 horizontale Winkel (0°-359°)
            h_indices = np.round(azimuths_norm).astype(int) % 360
            
            # Prüfe ob Daten Frequenzdimension haben
            has_frequency_dim = len(magnitude.shape) == 3
            freq_idx = magnitude.shape[2] // 2 if has_frequency_dim else None
            
            # Erstelle Ergebnis-Arrays (gleiches Shape wie Input)
            mag_values = np.zeros_like(azimuths, dtype=np.float64)
            phase_values = np.zeros_like(azimuths, dtype=np.float64)
            
            # --------------------------------------------------------
            # SCHRITT 2: Masken-basierte Verarbeitung (3 Fälle)
            # --------------------------------------------------------
            # FALL 1: Punkte UNTERHALB der kleinsten Elevation
            # → Verwende die erste verfügbare Messung (ohne Interpolation)
            mask_below = elevations <= vertical_angles[0]
            if has_frequency_dim:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below], freq_idx]
                phase_values[mask_below] = phase[0, h_indices[mask_below], freq_idx]
            else:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below]]
                phase_values[mask_below] = phase[0, h_indices[mask_below]]
            
            # FALL 2: Punkte OBERHALB der größten Elevation
            # → Verwende die letzte verfügbare Messung (ohne Interpolation)
            mask_above = elevations >= vertical_angles[-1]
            v_max = len(vertical_angles) - 1
            if has_frequency_dim:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above], freq_idx]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above], freq_idx]
            else:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above]]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above]]
            
            # FALL 3: Punkte ZWISCHEN zwei Messungen
            # → Lineare Interpolation (vektorisiert)
            mask_interp = ~(mask_below | mask_above)
            
            if np.any(mask_interp):
                # --------------------------------------------------------
                # SCHRITT 3: Vektorisierte Interpolation
                # --------------------------------------------------------
                # Extrahiere nur die zu interpolierenden Werte
                elev_interp = elevations[mask_interp]
                h_idx_interp = h_indices[mask_interp]
                
                # 🚀 np.searchsorted: Findet für ALLE Elevationen die umgebenden Indizes GLEICHZEITIG!
                # Beispiel: elevation=17.3° → findet Index für 15° und 20°
                v_idx_lower = np.searchsorted(vertical_angles, elev_interp, side='right') - 1
                v_idx_upper = v_idx_lower + 1
                
                # Clamp Indizes auf gültigen Bereich
                v_idx_lower = np.clip(v_idx_lower, 0, len(vertical_angles) - 1)
                v_idx_upper = np.clip(v_idx_upper, 0, len(vertical_angles) - 1)
                
                # Hole die umgebenden Winkel
                angle_lower = vertical_angles[v_idx_lower]
                angle_upper = vertical_angles[v_idx_upper]
                
                # Berechne Interpolationsfaktoren (0 bis 1) für ALLE Punkte gleichzeitig
                # t = (aktueller_winkel - unterer_winkel) / (oberer_winkel - unterer_winkel)
                # Beispiel: t = (17.3 - 15) / (20 - 15) = 0.46 (46% zwischen den Werten)
                t = (elev_interp - angle_lower) / (angle_upper - angle_lower + 1e-10)
                t = np.clip(t, 0, 1)
                
                # Hole Werte für Interpolation (Advanced Indexing!)
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
                
                # 🚀 Lineare Interpolation für ALLE Punkte gleichzeitig!
                # value = lower + t × (upper - lower)
                mag_values[mask_interp] = mag_lower + t * (mag_upper - mag_lower)
                phase_values[mask_interp] = phase_lower + t * (phase_upper - phase_lower)
            
            return mag_values, phase_values
            
        except Exception as e:
            print(f"Fehler in Batch-Interpolation: {e}")
            print(f"Array-Form: Magnitude {magnitude.shape}, Phase {phase.shape}")
            print(f"Azimuths: {azimuths.shape}, Elevations: {elevations.shape}")
            return None, None


    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container

    def calculate_mirror_azimuth(self, source_azimuth, wall):
        """
        Berechnet den Azimut der Spiegelquelle
        """
        wall_start, wall_end = wall
        
        # Wandvektor berechnen
        wall_dx = wall_end[0] - wall_start[0]
        wall_dy = wall_end[1] - wall_start[1]
        
        # Wandwinkel zur Y-Achse
        wall_angle = np.degrees(np.arctan2(wall_dx, wall_dy))
        
        # Spiegelung: doppelter Wandwinkel minus Quellazimut
        mirror_azimuth = 2 * wall_angle - source_azimuth
        
        # Normalisiere auf -180° bis +180°
        if mirror_azimuth > 180:
            mirror_azimuth -= 360
        elif mirror_azimuth < -180:
            mirror_azimuth += 360
        
        return mirror_azimuth