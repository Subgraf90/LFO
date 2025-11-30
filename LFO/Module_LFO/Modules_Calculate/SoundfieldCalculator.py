from __future__ import annotations

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGridCalculator import SurfaceGridCalculator
from typing import List, Dict, Tuple, Optional, Any

DEBUG_SOUNDFIELD = bool(int(__import__("os").environ.get("LFO_DEBUG_SOUNDFIELD", "1")))

# Optional: ShadowCalculator für Schattenberechnung
try:
    from Module_LFO.Modules_Calculate.ShadowCalculator import ShadowCalculator
    SHADOW_CALCULATOR_AVAILABLE = True
except ImportError:
    SHADOW_CALCULATOR_AVAILABLE = False
    ShadowCalculator = None


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
        
        # 🎯 GRID-CALCULATOR: Separate Instanz für Grid-Erstellung
        self._grid_calculator = SurfaceGridCalculator(settings)
        
        # 🎯 SHADOW-CALCULATOR: Optional für Schattenberechnung (nur bei Superposition)
        self._shadow_calculator = None
        if SHADOW_CALCULATOR_AVAILABLE:
            try:
                self._shadow_calculator = ShadowCalculator(settings)
            except Exception:
                self._shadow_calculator = None
   
    def calculate_soundfield_pressure(self):
        (
            self.calculation_spl["sound_field_p"],
            self.calculation_spl["sound_field_x"],
            self.calculation_spl["sound_field_y"],
        ) = self.calculate_sound_field()
        # Phase-Daten sind nach jeder SPL-Neuberechnung veraltet
        if isinstance(self.calculation_spl, dict):
            self.calculation_spl.pop("sound_field_phase", None)
            self.calculation_spl.pop("sound_field_phase_diff", None)

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
            return None, None
        
        # Direkte Nutzung der optimierten Balloon-Daten
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
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
            return None, None
        
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            return None, None
        
        return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)

    def _calculate_sound_field_complex(self, capture_arrays: bool = False):
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
        has_active_sources = any(
            not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values()
        )
        
        # ============================================================
        # SCHRITT 1: Grid-Erstellung (mit Surface-Unterstützung)
        # ============================================================
        # Hole aktivierte Surfaces
        enabled_surfaces = self._get_enabled_surfaces()
        
        # 🎯 VERWENDE SURFACE-GRID-CALCULATOR: Erstellt komplettes Grid basierend auf enabled Surfaces
        (
            sound_field_x,
            sound_field_y,
            X_grid,
            Y_grid,
            Z_grid,
            surface_mask,
        ) = self._grid_calculator.create_calculation_grid(enabled_surfaces)
        # 🎯 DEBUG: Immer Resolution und Datenpunkte ausgeben
        total_points = int(X_grid.size)
        active_points = int(np.count_nonzero(surface_mask))
        resolution = self.settings.resolution
        nx_points = len(sound_field_x)
        ny_points = len(sound_field_y)
        surface_meshes = self._grid_calculator.get_surface_meshes()
        surface_samples = self._grid_calculator.get_surface_sampling_points()
        grid_points = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        surface_mask_flat = surface_mask.reshape(-1)
        
        # 🎯 DEBUG: Prüfe ob Hindernisfläche in surface_mask enthalten ist
        if DEBUG_SOUNDFIELD and enabled_surfaces:
            # Finde Hindernis-Surfaces (vertikale oder mit signifikanter Z-Ausdehnung)
            obstacle_surfaces = []
            for surface_id, surface_dict in enabled_surfaces:
                points = surface_dict.get("points", [])
                if len(points) >= 3:
                    zs = [p.get("z", 0.0) for p in points]
                    z_span = max(zs) - min(zs) if zs else 0.0
                    if z_span > 0.5:  # Signifikante Z-Ausdehnung
                        obstacle_surfaces.append(surface_id)
            
            if obstacle_surfaces:
                # Prüfe ob Punkte auf Hindernis-Surfaces in surface_mask enthalten sind
                for obstacle_id in obstacle_surfaces:
                    obstacle_dict = next((s[1] for s in enabled_surfaces if s[0] == obstacle_id), None)
                    if obstacle_dict:
                        points = obstacle_dict.get("points", [])
                        if len(points) >= 3:
                            # Prüfe ob Mittelpunkt des Hindernisses in surface_mask enthalten ist
                            xs = [p.get("x", 0.0) for p in points]
                            ys = [p.get("y", 0.0) for p in points]
                            center_x = (min(xs) + max(xs)) / 2.0
                            center_y = (min(ys) + max(ys)) / 2.0
                            
                            # Finde nächsten Grid-Punkt
                            distances = np.sqrt((X_grid - center_x)**2 + (Y_grid - center_y)**2)
                            min_idx = np.unravel_index(np.argmin(distances), X_grid.shape)
                            mask_value = surface_mask[min_idx]
                            
                            print(
                                f"[SoundFieldCalculator] Hindernis '{obstacle_id}': "
                                f"Center=({center_x:.3f}, {center_y:.3f}), "
                                f"surface_mask={mask_value} (True=in Maske, False=nicht in Maske)"
                            )
        
        # ============================================================
        # SCHATTEN-BERECHNUNG (nur bei Superposition)
        # ============================================================
        # Bei Superposition verhält sich Schall wie Licht: direkter Schatten,
        # keine Beugung. Punkte im Schatten werden NICHT berechnet.
        # Bei FEM/FDTD wird Schatten NICHT angewendet, da Beugung berücksichtigt wird.
        # 
        # WICHTIG: Schatten wird PRO LAUTSPRECHER geprüft!
        # Jeder Lautsprecher prüft einzeln, ob ein Hindernis zwischen ihm und dem Punkt liegt.
        shadow_calculator_ready = (
            self._shadow_calculator is not None 
            and enabled_surfaces
            and getattr(self.settings, "enable_shadow_calculation", True)
        )
        surface_field_buffers: Dict[str, np.ndarray] = {}
        surface_point_buffers: Dict[str, np.ndarray] = {}
        if surface_samples:
            for sample in surface_samples:
                # Planare Flächen: Feldwerte direkt aus dem globalen Grid
                # über lokale Indizes auslesen (klassischer Pfad)
                if not getattr(sample, "is_vertical", False) and sample.indices.size > 0:
                    surface_field_buffers[sample.surface_id] = np.zeros(
                        sample.indices.shape[0],
                        dtype=complex,
                    )
                # Vertikale UND planare Flächen: 3D-Sample-Punkte bekommen
                # eigene Feldpuffer (für vertikale Wände ist das der Hauptpfad)
                sample_coords = np.asarray(sample.coordinates)
                if sample_coords.size > 0:
                    surface_point_buffers[sample.surface_id] = np.zeros(
                        sample_coords.reshape(-1, 3).shape[0],
                        dtype=complex,
                    )
        
        # Wenn keine aktiven Quellen, gib leere Arrays zurück
        if not has_active_sources:
            return [], sound_field_x, sound_field_y, ({} if capture_arrays else None)
        
        # Initialisiere das Schallfeld als 2D-Array komplexer Zahlen
        # Shape: [Y, X] = [Zeilen, Spalten] (NumPy-Konvention!)
        sound_field_p = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=complex)
        array_fields = {} if capture_arrays else None
        
        # ============================================================
        # SCHRITT 2: Vorbereitung - Physikalische Konstanten
        # ============================================================
        # Berechne physikalische Konstanten (einmalig für alle Quellen)
        # 🌡️ Temperaturabhängige Schallgeschwindigkeit (wird in UiSettings berechnet)
        speed_of_sound = self.settings.speed_of_sound
        wave_number = self.functions.wavenumber(speed_of_sound, self.settings.calculate_frequency)
        calculate_frequency = self.settings.calculate_frequency
        a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
        
        # Speichere Minimum und Maximum der berechneten Pegel für Vergleich
        min_level = float('inf')
        max_level = float('-inf')
        
        # ============================================================
        # SCHRITT 3: Iteriere über alle Lautsprecher-Arrays
        # ============================================================
        for array_key, speaker_array in self.settings.speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            array_wave = (
                np.zeros_like(sound_field_p, dtype=complex) if capture_arrays else None
            )
            
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
                # Berechne Distanz-Vektoren für ALLE Punkte gleichzeitig
                # Resultat: 2D-Arrays mit Shape [ny, nx]
                x_distances = X_grid - source_position_x[isrc]  # Distanz in X-Richtung
                y_distances = Y_grid - source_position_y[isrc]  # Distanz in Y-Richtung
                
                # Berechne horizontale Distanz (Pythagoras in 2D)
                # √(x² + y²) für ALLE Punkte gleichzeitig
                horizontal_dists = np.sqrt(x_distances**2 + y_distances**2)
                
                # 🎯 Z-DISTANZ: Verwende interpolierte Z-Koordinaten aus Surfaces (falls aktiviert)
                # Z_grid enthält die Z-Koordinaten jedes Grid-Punkts (aus Surface-Interpolation)
                # Wenn keine Surfaces aktiviert sind, ist Z_grid = 0 (Standard)
                z_distance = Z_grid - source_position_z[isrc]  # Z-Distanz für jeden Punkt individuell
                
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
                # Verwende die individuellen Z-Distanzen (aus Surface-Interpolation)
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dists))
                
                # --------------------------------------------------------
                # 4.3: BATCH-INTERPOLATION (der Performance-Booster! 🚀)
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
                    source_position = np.array(
                        [
                            source_position_x[isrc],
                            source_position_y[isrc],
                            source_position_z[isrc],
                        ],
                        dtype=float,
                    )
                    polarity_flag = False
                    if hasattr(speaker_array, 'source_polarity'):
                        try:
                            polarity_flag = bool(speaker_array.source_polarity[isrc])
                        except (TypeError, IndexError):
                            polarity_flag = False

                    source_props = {
                        "magnitude_linear": magnitude_linear.reshape(-1),
                        "polar_phase_rad": polar_phase_rad.reshape(-1),
                        "source_level": source_level[isrc],
                        "a_source_pa": a_source_pa,
                        "wave_number": wave_number,
                        "frequency": calculate_frequency,
                        "source_time": source_time[isrc],
                        "polarity": polarity_flag,
                        "distances": source_dists.reshape(-1),
                    }
                    # ============================================================
                    # SCHATTEN-PRÜFUNG PRO LAUTSPRECHER
                    # ============================================================
                    # Prüfe für DIESEN spezifischen Lautsprecher, ob ein Hindernis
                    # zwischen ihm und jedem Punkt liegt
                    # 🎯 WICHTIG: Initialisiere combined_mask mit surface_mask
                    # (Punkte außerhalb von Surfaces werden sowieso nicht berechnet)
                    combined_mask = surface_mask_flat.copy() if surface_mask_flat is not None else None
                    
                    if shadow_calculator_ready:
                        try:
                            import time
                            shadow_start_time = time.time()
                            
                            # Prüfe Schatten nur für DIESEN Lautsprecher
                            # WICHTIG: Verwende akustische Positionen (source_position_calc_*)
                            # aus SpeakerPositionCalculator (berechnet in calculate_stack_center)
                            source_pos_single = [
                                (
                                    float(source_position_x[isrc]),
                                    float(source_position_y[isrc]),
                                    float(source_position_z[isrc]),
                                )
                            ]
                            
                            # Debug: Zeige verwendete akustische Position
                            if DEBUG_SOUNDFIELD:
                                print(
                                    f"[SoundFieldCalculator] ===== Schatten-Berechnung für Lautsprecher {isrc} ({speaker_name}) ====="
                                )
                                print(
                                    f"[SoundFieldCalculator] Akustische Position: ({source_pos_single[0][0]:.3f}, "
                                    f"{source_pos_single[0][1]:.3f}, {source_pos_single[0][2]:.3f}) m"
                                )
                                print(
                                    f"[SoundFieldCalculator] Grid-Punkte: {len(grid_points)}, "
                                    f"Enabled Surfaces: {len(enabled_surfaces)}"
                                )
                            
                            # Berechne Schatten-Maske nur für diesen Lautsprecher
                            # 🎯 OPTIMIERUNG: Resolution-basierte Toleranz (nur bei kleineren Grids)
                            # Punkte im Schatten, die innerhalb der Resolution zu sichtbaren Punkten liegen,
                            # werden trotzdem berechnet (für bessere Darstellung)
                            # ⚠️ PERFORMANCE: Nur bei Grids < 10000 Punkten aktivieren
                            resolution = self.settings.resolution
                            num_grid_points = len(grid_points)
                            use_resolution_tolerance = num_grid_points < 10000
                            
                            if DEBUG_SOUNDFIELD:
                                print(
                                    f"[SoundFieldCalculator] Resolution-Toleranz: "
                                    f"{'AKTIV' if use_resolution_tolerance else 'DEAKTIV'} "
                                    f"(Grid-Punkte: {num_grid_points}, Threshold: 10000)"
                                )
                            
                            # 🎯 Verwende compute_shadow_mask_for_calculation: Nur Punkte tief im Schatten werden deaktiviert
                            # Randpunkte (mit benachbarten sichtbaren Punkten) werden trotzdem berechnet
                            shadow_mask_this_source = self._shadow_calculator.compute_shadow_mask_for_calculation(
                                grid_points,
                                source_pos_single,
                                enabled_surfaces,
                                X_grid,
                                Y_grid,
                                tolerance=0.01,
                            )
                            
                            shadow_time = time.time() - shadow_start_time
                            
                            # Debug: Zeige wie viele Punkte im Schatten sind
                            if DEBUG_SOUNDFIELD:
                                num_shadow = np.count_nonzero(shadow_mask_this_source)
                                num_visible = len(grid_points) - num_shadow
                                print(
                                    f"[SoundFieldCalculator] Ergebnis: {num_shadow}/{len(grid_points)} Punkte im Schatten "
                                    f"({100.0*num_shadow/len(grid_points):.1f}%), "
                                    f"{num_visible} sichtbar ({100.0*num_visible/len(grid_points):.1f}%)"
                                )
                                print(
                                    f"[SoundFieldCalculator] Berechnungszeit: {shadow_time:.3f}s "
                                    f"({len(grid_points)/shadow_time:.0f} Punkte/s)"
                                )
                                
                                # 🎯 DEBUG: Prüfe speziell für Hindernis 'test_2'
                                obstacle_id = 'test_2'
                                obstacle_dict = next((s[1] for s in enabled_surfaces if s[0] == obstacle_id), None)
                                if obstacle_dict:
                                    points = obstacle_dict.get("points", [])
                                    if len(points) >= 3:
                                        xs = [p.get("x", 0.0) for p in points]
                                        ys = [p.get("y", 0.0) for p in points]
                                        zs = [p.get("z", 0.0) for p in points]
                                        x_min, x_max = min(xs), max(xs)
                                        y_min, y_max = min(ys), max(ys)
                                        z_min, z_max = min(zs), max(zs)
                                        
                                        # Finde Punkte auf dem Hindernis
                                        tolerance = 0.1
                                        on_obstacle = (
                                            (grid_points[:, 0] >= x_min - tolerance) &
                                            (grid_points[:, 0] <= x_max + tolerance) &
                                            (grid_points[:, 1] >= y_min - tolerance) &
                                            (grid_points[:, 1] <= y_max + tolerance) &
                                            (grid_points[:, 2] >= z_min - tolerance) &
                                            (grid_points[:, 2] <= z_max + tolerance)
                                        )
                                        
                                        num_on_obstacle = np.count_nonzero(on_obstacle)
                                        num_on_obstacle_in_shadow = np.count_nonzero(on_obstacle & shadow_mask_this_source)
                                        num_on_obstacle_visible = num_on_obstacle - num_on_obstacle_in_shadow
                                        
                                        print(
                                            f"[SoundFieldCalculator] Hindernis '{obstacle_id}': "
                                            f"{num_on_obstacle} Punkte auf Hindernis, "
                                            f"{num_on_obstacle_in_shadow} im Schatten, "
                                            f"{num_on_obstacle_visible} sichtbar"
                                        )
                                        
                                        # 🎯 DEBUG: Prüfe auch Punkte HINTER dem Hindernis (im Schatten)
                                        # Finde Punkte, die im Schatten sind, aber NICHT auf dem Hindernis
                                        points_in_shadow_not_on_obstacle = shadow_mask_this_source & ~on_obstacle
                                        num_in_shadow_not_on_obstacle = np.count_nonzero(points_in_shadow_not_on_obstacle)
                                        print(
                                            f"[SoundFieldCalculator] Punkte HINTER '{obstacle_id}' (im Schatten, nicht auf Hindernis): "
                                            f"{num_in_shadow_not_on_obstacle} Punkte"
                                        )
                                        
                                        # Prüfe auch surface_mask
                                        num_on_obstacle_in_surface_mask = np.count_nonzero(on_obstacle & surface_mask_flat)
                                        print(
                                            f"[SoundFieldCalculator] Hindernis '{obstacle_id}': "
                                            f"{num_on_obstacle_in_surface_mask}/{num_on_obstacle} Punkte in surface_mask"
                                        )
                                        
                            # 🎯 WICHTIG: Entferne Punkte auf Hindernissen aus der Schatten-Maske
                            # Punkte auf Hindernissen sollen IMMER sichtbar sein (nicht im Schatten)
                            # Finde alle Hindernisse und deren Punkte
                            obstacle_points_mask = np.zeros(len(grid_points), dtype=bool)
                            for surface_id, surface_dict in enabled_surfaces:
                                points = surface_dict.get("points", [])
                                if len(points) < 3:
                                    continue
                                
                                # Prüfe ob Surface ein Hindernis ist (vertikal oder signifikante Z-Ausdehnung)
                                zs = [p.get("z", 0.0) for p in points]
                                z_span = max(zs) - min(zs) if zs else 0.0
                                
                                if z_span < 0.5:  # Weniger als 50cm Z-Ausdehnung → kein Hindernis
                                    continue
                                
                                # Finde Punkte auf diesem Hindernis
                                xs = [p.get("x", 0.0) for p in points]
                                ys = [p.get("y", 0.0) for p in points]
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                z_min, z_max = min(zs), max(zs)
                                
                                tolerance = 0.2  # 20cm Toleranz
                                on_this_obstacle = (
                                    (grid_points[:, 0] >= x_min - tolerance) &
                                    (grid_points[:, 0] <= x_max + tolerance) &
                                    (grid_points[:, 1] >= y_min - tolerance) &
                                    (grid_points[:, 1] <= y_max + tolerance) &
                                    (grid_points[:, 2] >= z_min - tolerance) &
                                    (grid_points[:, 2] <= z_max + tolerance)
                                )
                                
                                obstacle_points_mask |= on_this_obstacle
                            
                            # 🎯 DEBUG: Berechne VOR der Korrektur, wie viele Punkte entfernt werden
                            if DEBUG_SOUNDFIELD:
                                num_obstacle_points = np.count_nonzero(obstacle_points_mask)
                                num_obstacle_points_in_shadow_before = np.count_nonzero(obstacle_points_mask & shadow_mask_this_source)
                            
                            # Entferne Punkte auf Hindernissen aus der Schatten-Maske
                            shadow_mask_this_source = shadow_mask_this_source & ~obstacle_points_mask
                            
                            if DEBUG_SOUNDFIELD:
                                print(
                                    f"[SoundFieldCalculator] {num_obstacle_points} Punkte auf Hindernissen gefunden, "
                                    f"{num_obstacle_points_in_shadow_before} aus Schatten-Maske entfernt "
                                    f"(von {num_obstacle_points} waren {num_obstacle_points_in_shadow_before} im Schatten)"
                                )
                                        
                            # Invertiere: True = sichtbar von diesem Lautsprecher, False = im Schatten
                            visible_from_this_source = ~shadow_mask_this_source
                            
                            # 🎯 WICHTIG: 
                            # - Punkte auf Hindernissen, die von diesem Lautsprecher sichtbar sind → berechnen
                            # - Punkte auf Hindernissen, die im Schatten sind → nicht berechnen
                            # - Punkte hinter Hindernissen → nicht berechnen (im Schatten)
                            # Die Schatten-Maske sollte bereits korrekt sein: Punkte auf Hindernissen
                            # werden nicht als Schatten markiert (siehe ShadowCalculator Logik)
                            
                            # 🎯 DEBUG: Prüfe combined_mask VOR Kombination
                            if DEBUG_SOUNDFIELD:
                                num_in_shadow = np.count_nonzero(shadow_mask_this_source)
                                num_in_surface = np.count_nonzero(surface_mask_flat) if surface_mask_flat is not None else len(grid_points)
                                num_in_combined_before = np.count_nonzero(combined_mask) if combined_mask is not None else len(grid_points)
                                # Prüfe wie viele Punkte im Schatten sind, die NICHT in surface_mask sind
                                if surface_mask_flat is not None:
                                    num_in_shadow_not_in_surface = np.count_nonzero(shadow_mask_this_source & ~surface_mask_flat)
                                else:
                                    num_in_shadow_not_in_surface = 0
                                print(
                                    f"[SoundFieldCalculator] VOR Kombination: "
                                    f"{num_in_shadow} Punkte im Schatten "
                                    f"(davon {num_in_shadow_not_in_surface} NICHT in surface_mask), "
                                    f"{num_in_surface} in surface_mask, "
                                    f"{num_in_combined_before} in combined_mask"
                                )
                            
                            if combined_mask is not None:
                                # Kombiniere: Punkt muss in Surface UND von diesem Lautsprecher sichtbar sein
                                combined_mask = combined_mask & visible_from_this_source
                            else:
                                combined_mask = visible_from_this_source
                            
                            # 🎯 DEBUG: Prüfe combined_mask NACH Kombination
                            if DEBUG_SOUNDFIELD:
                                num_in_combined_after = np.count_nonzero(combined_mask) if combined_mask is not None else 0
                                num_removed_from_combined = num_in_combined_before - num_in_combined_after
                                print(
                                    f"[SoundFieldCalculator] NACH Kombination: "
                                    f"{num_in_combined_after} in combined_mask, "
                                    f"{num_removed_from_combined} Punkte entfernt (sollten im Schatten sein)"
                                )
                            
                            # 🎯 DEBUG: Prüfe combined_mask NACH Kombination
                            if DEBUG_SOUNDFIELD and enabled_surfaces:
                                for obstacle_id in obstacle_surfaces:
                                    obstacle_dict = next((s[1] for s in enabled_surfaces if s[0] == obstacle_id), None)
                                    if obstacle_dict:
                                        points = obstacle_dict.get("points", [])
                                        if len(points) >= 3:
                                            xs = [p.get("x", 0.0) for p in points]
                                            ys = [p.get("y", 0.0) for p in points]
                                            zs = [p.get("z", 0.0) for p in points]
                                            x_min, x_max = min(xs), max(xs)
                                            y_min, y_max = min(ys), max(ys)
                                            z_min, z_max = min(zs), max(zs)
                                            
                                            tolerance = 0.1
                                            on_obstacle = (
                                                (grid_points[:, 0] >= x_min - tolerance) &
                                                (grid_points[:, 0] <= x_max + tolerance) &
                                                (grid_points[:, 1] >= y_min - tolerance) &
                                                (grid_points[:, 1] <= y_max + tolerance) &
                                                (grid_points[:, 2] >= z_min - tolerance) &
                                                (grid_points[:, 2] <= z_max + tolerance)
                                            )
                                            
                                            num_on_obstacle = np.count_nonzero(on_obstacle)
                                            if combined_mask is not None:
                                                num_on_obstacle_in_combined = np.count_nonzero(on_obstacle & combined_mask)
                                                print(
                                                    f"[SoundFieldCalculator] Hindernis '{obstacle_id}': "
                                                    f"{num_on_obstacle_in_combined}/{num_on_obstacle} Punkte in combined_mask "
                                                    f"(NACH Kombination mit Lautsprecher {isrc})"
                                                )
                                
                        except Exception as exc:
                            if DEBUG_SOUNDFIELD:
                                print(
                                    f"[SoundFieldCalculator] Fehler bei Schattenprüfung für Lautsprecher {isrc}: {exc}"
                                )
                                import traceback
                                traceback.print_exc()
                            # Bei Fehler: verwende nur Surface-Maske (kein Schatten)
                            pass
                    
                    mask_options = {
                        "min_distance": 0.001,
                        "additional_mask": combined_mask,
                    }
                    wave_flat = self._compute_wave_for_points(
                        grid_points,
                        source_position,
                        source_props,
                        mask_options,
                    )
                    wave = wave_flat.reshape(source_dists.shape)
                    
                    # --------------------------------------------------------
                    # 4.6: AKKUMULATION (Interferenz-Überlagerung)
                    # --------------------------------------------------------
                    # Addiere die Welle dieses Lautsprechers zum Gesamt-Schallfeld
                    # Komplexe Addition → Automatische Interferenz (konstruktiv/destruktiv)
                    sound_field_p += wave
                    if surface_field_buffers:
                        for sample in surface_samples:
                            # Nur planare Surfaces nutzen die (row,col)-Indizes
                            # im globalen Grid. Vertikale Flächen haben ein
                            # eigenes lokales Raster und werden separat über
                            # surface_point_buffers versorgt.
                            if getattr(sample, "is_vertical", False):
                                continue
                            buffer = surface_field_buffers.get(sample.surface_id)
                            if buffer is None or sample.indices.size == 0:
                                continue
                            rows = sample.indices[:, 0]
                            cols = sample.indices[:, 1]
                            buffer += wave[rows, cols]
                    if surface_point_buffers:
                        for sample in surface_samples:
                            point_buffer = surface_point_buffers.get(sample.surface_id)
                            if point_buffer is None or sample.coordinates.size == 0:
                                continue

                            coords = np.asarray(sample.coordinates, dtype=float).reshape(-1, 3)
                            dx = coords[:, 0] - source_position_x[isrc]
                            dy = coords[:, 1] - source_position_y[isrc]
                            dz = coords[:, 2] - source_position_z[isrc]
                            horizontal = np.sqrt(dx**2 + dy**2)
                            sample_dists = np.sqrt(horizontal**2 + dz**2)

                            sample_azimuths = (np.degrees(np.arctan2(dy, dx)) + np.degrees(source_azimuth[isrc])) % 360
                            sample_azimuths = (360 - sample_azimuths) % 360
                            sample_azimuths = (sample_azimuths + 90) % 360
                            sample_elevations = np.degrees(np.arctan2(dz, horizontal))

                            azimuths_payload = sample_azimuths.reshape(1, -1)
                            elevations_payload = sample_elevations.reshape(1, -1)
                            sample_gains, sample_phases = self.get_balloon_data_batch(
                                speaker_name,
                                azimuths_payload,
                                elevations_payload,
                            )
                            if sample_gains is None or sample_phases is None:
                                continue

                            sample_props = {
                                "magnitude_linear": (10 ** (sample_gains.reshape(-1) / 20)),
                                "polar_phase_rad": np.radians(sample_phases.reshape(-1)),
                                "source_level": source_level[isrc],
                                "a_source_pa": a_source_pa,
                                "wave_number": wave_number,
                                "frequency": calculate_frequency,
                                "source_time": source_time[isrc],
                                "polarity": polarity_flag,
                                "distances": sample_dists,
                            }
                            sample_wave = self._compute_wave_for_points(
                                coords,
                                source_position,
                                sample_props,
                                {"min_distance": 0.001},
                            )
                            point_buffer += sample_wave
                    if capture_arrays:
                        array_wave += wave
            if capture_arrays and array_wave is not None:
                array_fields[array_key] = array_wave

        # 🎯 Speichere Z-Grid für Plot-Verwendung
        # Konvertiere Z_grid zu Liste für JSON-Serialisierung
        surface_fields = {}
        if surface_field_buffers:
            for surface_id, buffer in surface_field_buffers.items():
                surface_fields[surface_id] = buffer.copy()
        else:
            surface_fields = self._calculate_surface_mesh_fields(sound_field_p, surface_samples)

        if surface_point_buffers:
            for surface_id, buffer in surface_point_buffers.items():
                if buffer.size == 0:
                    continue
                surface_fields[surface_id] = buffer.copy()

        sample_payloads = []
        if surface_samples:
            for sample in surface_samples:
                sample_payloads.append(
                    {
                        "surface_id": sample.surface_id,
                        "name": sample.name,
                        "coordinates": sample.coordinates.tolist(),
                        "indices": sample.indices.tolist(),
                        "grid_shape": list(sample.grid_shape),
                        "kind": "vertical" if getattr(sample, "is_vertical", False) else "planar",
                    }
                )

        if isinstance(self.calculation_spl, dict):
            self.calculation_spl['sound_field_z'] = Z_grid.tolist()
            # 🎯 Speichere erweiterte Maske für Berechnung
            self.calculation_spl['surface_mask'] = surface_mask.astype(bool).tolist()
            # 🎯 Berechne und speichere strikte Maske für Plot (ohne Erweiterung)
            enabled_surfaces = self._get_enabled_surfaces()
            if enabled_surfaces:
                surface_mask_strict = self._grid_calculator._create_surface_mask(
                    X_grid, Y_grid, enabled_surfaces, include_edges=False
                )
                self.calculation_spl['surface_mask_strict'] = surface_mask_strict.astype(bool).tolist()
            else:
                self.calculation_spl['surface_mask_strict'] = surface_mask.astype(bool).tolist()
            
            # 🎯 Schatten-Maske wird nicht mehr global gespeichert,
            # da sie pro Lautsprecher berechnet wird
            # (könnte später für Visualisierung erweitert werden)
            self.calculation_spl['shadow_mask'] = None
            if surface_meshes:
                self.calculation_spl['surface_meshes'] = [
                    mesh.to_payload() for mesh in surface_meshes
                ]
            self.calculation_spl['surface_samples'] = sample_payloads
            self.calculation_spl['surface_fields'] = {
                surface_id: values.tolist()
                for surface_id, values in surface_fields.items()
            }

        return sound_field_p, sound_field_x, sound_field_y, array_fields
    
    def _compute_wave_for_points(
        self,
        points: np.ndarray,
        source_position: np.ndarray,
        source_props: Dict[str, Any],
        mask_options: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Berechnet das komplexe Wellenfeld für beliebige 3D-Punkte.
        
        Args:
            points: Array der Form (N, 3) mit XYZ-Koordinaten der Zielpunkte.
            source_position: Array-Like mit (x, y, z) der Quelle.
            source_props: Parameterpaket mit Richtcharakteristik und Quelleigenschaften.
            mask_options: Optionale Maskenparameter (z.B. Mindestdistanz, zusätzliche Maske).
        
        Returns:
            np.ndarray: Komplexe Wellenwerte (Shape: N).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if points.size == 0:
            return np.zeros(0, dtype=complex)
        
        source_position = np.asarray(source_position, dtype=float).reshape(3)
        mask_options = mask_options or {}
        
        distances = source_props.get("distances")
        if distances is not None:
            distances = np.asarray(distances, dtype=float).reshape(-1)
        else:
            deltas = points - source_position
            distances = np.linalg.norm(deltas, axis=1)
        
        min_distance = float(mask_options.get("min_distance", 0.0))
        valid_mask = distances >= min_distance
        
        additional_mask = mask_options.get("additional_mask")
        if additional_mask is not None:
            additional_mask_array = np.asarray(additional_mask, dtype=bool).reshape(-1)
            if DEBUG_SOUNDFIELD and len(additional_mask_array) > 0:
                num_additional_mask_true = np.count_nonzero(additional_mask_array)
                num_valid_before = np.count_nonzero(valid_mask)
                print(
                    f"[_compute_wave_for_points] additional_mask: "
                    f"{num_additional_mask_true}/{len(additional_mask_array)} Punkte aktiv, "
                    f"valid_mask vorher: {num_valid_before}/{len(valid_mask)}"
                )
            valid_mask &= additional_mask_array
            if DEBUG_SOUNDFIELD and len(additional_mask_array) > 0:
                num_valid_after = np.count_nonzero(valid_mask)
                num_removed_by_mask = num_valid_before - num_valid_after
                print(
                    f"[_compute_wave_for_points] valid_mask nachher: "
                    f"{num_valid_after}/{len(valid_mask)}, "
                    f"{num_removed_by_mask} Punkte durch additional_mask entfernt"
                )
        
        wave = np.zeros(points.shape[0], dtype=complex)
        if not np.any(valid_mask):
            return wave
        
        magnitude_linear = np.asarray(source_props["magnitude_linear"], dtype=float).reshape(-1)
        polar_phase_rad = np.asarray(source_props["polar_phase_rad"], dtype=float).reshape(-1)
        source_level = float(np.asarray(source_props["source_level"], dtype=float).reshape(-1)[0])
        a_source_pa = float(np.asarray(source_props["a_source_pa"], dtype=float))
        wave_number = float(np.asarray(source_props["wave_number"], dtype=float))
        frequency = float(np.asarray(source_props["frequency"], dtype=float))
        source_time = float(np.asarray(source_props["source_time"], dtype=float))
        polarity = bool(source_props.get("polarity", False))
        
        phase = (
            wave_number * distances[valid_mask]
            + polar_phase_rad[valid_mask]
            + 2 * np.pi * frequency * source_time
        )
        wave[valid_mask] = (
            magnitude_linear[valid_mask]
            * source_level
            * a_source_pa
            * np.exp(1j * phase)
            / distances[valid_mask]
        )
        
        if polarity:
            wave = -wave
        
        return wave

    def _calculate_surface_mesh_fields(
        self,
        sound_field_complex: np.ndarray,
        surface_samples: List["SurfaceSamplingPoints"],
    ) -> Dict[str, np.ndarray]:
        surface_fields: Dict[str, np.ndarray] = {}
        if sound_field_complex is None or not surface_samples:
            return surface_fields

        try:
            for sample in surface_samples:
                if sample.indices.size == 0:
                    continue
                rows = sample.indices[:, 0]
                cols = sample.indices[:, 1]
                values = sound_field_complex[rows, cols]
                surface_fields[sample.surface_id] = values.copy()
        except Exception:
            pass
        return surface_fields

    def calculate_sound_field(self):
        """
        Liefert die Schalldruckbeträge für das komplette Grid.
        """
        (
            sound_field_complex,
            sound_field_x,
            sound_field_y,
            _,
        ) = self._calculate_sound_field_complex()
        
        # Bei fehlenden Quellen direkt zurückgeben (leere Arrays)
        if isinstance(sound_field_complex, list):
            return sound_field_complex, sound_field_x, sound_field_y
        
        sound_field_magnitude = np.abs(sound_field_complex)
        
        # Konvertiere zu dB (für Statistiken, wird nicht zurückgegeben)
        _ = 20 * np.log10(sound_field_magnitude + 1e-10)  # +1e-10 verhindert log(0)
        
        return sound_field_magnitude, sound_field_x, sound_field_y
    
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
            return None, None


    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container

    # ============================================================
    # SURFACE-INTEGRATION: Helper-Methoden
    # ============================================================
    
    def _get_enabled_surfaces(self) -> List[Tuple[str, Dict]]:
        """
        Gibt alle aktivierten Surfaces zurück, die für die Berechnung verwendet werden sollen.
        
        Filtert Surfaces nach:
        - enabled=True: Surface ist aktiviert
        - hidden=False: Surface ist nicht versteckt
        
        Nur Surfaces, die beide Bedingungen erfüllen, werden in die Berechnung einbezogen.
        Die Koordination (wann Empty Plot, wann Berechnung) erfolgt über
        WindowPlotsMainwindow.update_plots_for_surface_state().
        
        Returns:
            Liste von Tupeln (surface_id, surface_definition) - nur enabled + nicht-hidden Surfaces
        """
        if not hasattr(self.settings, 'surface_definitions'):
            return []
        
        enabled = []
        for surface_id, surface_def in self.settings.surface_definitions.items():
            if hasattr(surface_def, "to_dict"):
                surface_data = surface_def.to_dict()
            elif isinstance(surface_def, dict):
                surface_data = surface_def
            else:
                # Fallback: versuche Attribute direkt zu lesen
                surface_data = {
                    "enabled": getattr(surface_def, "enabled", False),
                    "hidden": getattr(surface_def, "hidden", False),
                    "points": getattr(surface_def, "points", []),
                    "name": getattr(surface_def, "name", surface_id),
                }
            if surface_data.get('enabled', False) and not surface_data.get('hidden', False):
                enabled.append((surface_id, surface_data))
        
        return enabled
