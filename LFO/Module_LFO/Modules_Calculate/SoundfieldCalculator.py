from __future__ import annotations

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator
from Module_LFO.Modules_Init.Logging import measure_time, perf_section
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    derive_surface_plane,
    _points_in_polygon_batch_uv,
    SurfaceDefinition,
)
from Module_LFO.Modules_Data.SurfaceValidator import validate_and_optimize_surface, triangulate_points
from typing import List, Dict, Tuple, Optional, Any

DEBUG_SOUNDFIELD = bool(int(__import__("os").environ.get("LFO_DEBUG_SOUNDFIELD", "1")))


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
        
        # 🎯 GRID-GENERATOR: Flexible Grid-Erstellung
        self._grid_generator = FlexibleGridGenerator(settings)
   
    @measure_time("SoundFieldCalculator.calculate_soundfield_pressure")
    def calculate_soundfield_pressure(self):
        with perf_section("SoundFieldCalculator.calculate_sound_field"):
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
        
        # 🎯 NEU: Flexible Grid-Erstellung via FlexibleGridGenerator (pro Group/Surface)
        if not enabled_surfaces:
            return [], np.array([]), np.array([]), ({} if capture_arrays else None)
        
        # Verwende generate_per_surface() um pro Surface Grids zu bekommen (nicht pro Gruppe)
        surface_grids_grouped: Dict[str, Any] = self._grid_generator.generate_per_surface(
            enabled_surfaces,
            resolution=self.settings.resolution,
            min_points_per_dimension=3
        )
        
        # Prüfe, welche Surfaces fehlen
        enabled_ids = {sid for sid, _ in enabled_surfaces}
        generated_ids = set(surface_grids_grouped.keys())
        missing_ids = enabled_ids - generated_ids
        if missing_ids:
            print(f"⚠️  [SoundFieldCalculator] {len(missing_ids)} Surface(s) wurden beim Grid-Generieren übersprungen: {sorted(missing_ids)}")
        
        # Debug: Zeige Orientierungen der generierten Grids
        if DEBUG_SOUNDFIELD:
            print(f"[DEBUG SoundFieldCalculator] Generierte Grids: {len(surface_grids_grouped)}")
            for sid, grid in surface_grids_grouped.items():
                orientation = getattr(grid.geometry, "orientation", None) if hasattr(grid, 'geometry') else None
                print(f"  └─ {sid}: orientation={orientation}, shape={grid.X_grid.shape if hasattr(grid, 'X_grid') else 'N/A'}")
        
        if not surface_grids_grouped:
            return [], np.array([]), np.array([]), ({} if capture_arrays else None)
        
        # 🎯 ERSTELLE SURFACE_GRIDS_DATA für Plot-Modul
        surface_grids_data = {}
        surface_results_data = {}
        
        # Kombiniere alle Grids zu einem gemeinsamen Grid für Rückwärtskompatibilität
        all_x = []
        all_y = []
        for grid in surface_grids_grouped.values():
            all_x.extend(grid.sound_field_x.tolist())
            all_y.extend(grid.sound_field_y.tolist())
        
        if all_x and all_y:
            unique_x = np.unique(np.array(all_x))
            unique_y = np.unique(np.array(all_y))
            unique_x.sort()
            unique_y.sort()
            
            # Erstelle kombiniertes Grid für Rückwärtskompatibilität
            X_grid_combined, Y_grid_combined = np.meshgrid(unique_x, unique_y, indexing='xy')
            Z_grid_combined = np.zeros_like(X_grid_combined, dtype=float)
            surface_mask_combined = np.zeros_like(X_grid_combined, dtype=bool)
        else:
            return [], np.array([]), np.array([]), ({} if capture_arrays else None)
        
        # 🎯 ALTE GRID-POSITIONEN AUSKOMMENTIERT - Verwende nur neue Grids
        # sound_field_x = cartesian_grid.sound_field_x  # ALT
        # sound_field_y = cartesian_grid.sound_field_y  # ALT
        sound_field_x = unique_x  # NEU: Aus kombinierten Surface-Grids
        sound_field_y = unique_y  # NEU: Aus kombinierten Surface-Grids
        X_grid = X_grid_combined  # NEU: Kombiniertes Grid
        Y_grid = Y_grid_combined  # NEU: Kombiniertes Grid
        
        # 🐛 DEBUG: Zeige verwendete Koordinatenachsen (kompakt)
        try:
            def _axis_summary(name: str, arr: np.ndarray) -> None:
                arr = np.asarray(arr, dtype=float)
                n = arr.size
                amin = float(arr.min()) if n else 0.0
                amax = float(arr.max()) if n else 0.0
                # zeige erste/letzte 5 Werte, um Datenflut zu vermeiden
                head = ", ".join(f"{v:.2f}" for v in arr[:5])
                tail = ", ".join(f"{v:.2f}" for v in arr[-5:]) if n > 5 else ""
                mid = " ... " if n > 10 else (" | " if n > 5 else "")
                print(f"[DEBUG GRID AXIS] {name}: len={n}, min={amin:.2f}, max={amax:.2f}")
                if n > 0:
                    print(f"  {name} values: {head}{mid}{tail}")

            _axis_summary("sound_field_x", sound_field_x)
            _axis_summary("sound_field_y", sound_field_y)
        except Exception as e:
            print(f"[DEBUG GRID AXIS] Ausgabe fehlgeschlagen: {e}")
        
        # Kombiniere Z-Grids und Masken aus einzelnen Surface-Grids
        from scipy.interpolate import griddata
        for surface_id, grid in surface_grids_grouped.items():
            is_vertical = getattr(grid.geometry, "orientation", None) == "vertical"
            
            # Vertikale Flächen nicht ins kombinierte Z-Grid mischen (werden separat geplottet)
            if not is_vertical:
                # Interpoliere Z-Grid
                points_orig = np.column_stack([grid.X_grid.ravel(), grid.Y_grid.ravel()])
                z_orig = grid.Z_grid.ravel()
                mask_orig = grid.surface_mask.ravel()
                
                points_new = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                
                # Interpoliere Z-Werte
                if np.any(mask_orig):
                    z_interp = griddata(
                        points_orig[mask_orig],
                        z_orig[mask_orig],
                        points_new,
                        method='nearest',
                        fill_value=0.0
                    )
                    Z_grid_combined = np.maximum(Z_grid_combined, z_interp.reshape(X_grid.shape))
                
                # Kombiniere Masken
                mask_interp = griddata(
                    points_orig,
                    mask_orig.astype(float),
                    points_new,
                    method='nearest',
                    fill_value=0.0
                )
                surface_mask_combined = surface_mask_combined | (mask_interp.reshape(X_grid.shape) > 0.5)
            
            # 🎯 ERSTELLE SURFACE_GRIDS_DATA für Plot-Modul (für ALLE Flächen, auch vertikale)
            # 🎯 NEU: Füge triangulierte Daten hinzu
            grid_data = {
                'sound_field_x': grid.sound_field_x.tolist(),
                'sound_field_y': grid.sound_field_y.tolist(),
                'X_grid': grid.X_grid.tolist(),
                'Y_grid': grid.Y_grid.tolist(),
                'Z_grid': grid.Z_grid.tolist(),
                'surface_mask': grid.surface_mask.astype(bool).tolist(),
                'resolution': grid.resolution,
                'orientation': grid.geometry.orientation,
                'dominant_axis': getattr(grid.geometry, 'dominant_axis', None),
            }
            
            # 🎯 TRIANGULATION: Füge triangulierte Vertices und Faces hinzu
            if hasattr(grid, 'triangulated_vertices') and grid.triangulated_vertices is not None:
                grid_data['triangulated_vertices'] = grid.triangulated_vertices.tolist()
            if hasattr(grid, 'triangulated_faces') and grid.triangulated_faces is not None:
                grid_data['triangulated_faces'] = grid.triangulated_faces.tolist()
            if hasattr(grid, 'triangulated_success'):
                grid_data['triangulated_success'] = grid.triangulated_success
            
            surface_grids_data[surface_id] = grid_data
        
        Z_grid = Z_grid_combined
        surface_mask = surface_mask_combined
        surface_mask_strict = surface_mask_combined.copy()
        # 🎯 DEBUG: Immer Resolution und Datenpunkte ausgeben
        total_points = int(X_grid.size)
        active_points = int(np.count_nonzero(surface_mask))
        resolution = self.settings.resolution
        nx_points = len(sound_field_x)
        ny_points = len(sound_field_y)
        
        # 🐛 DEBUG: Prüfe Z-Koordinaten für schräge Flächen
        if DEBUG_SOUNDFIELD and enabled_surfaces:
            # Importiere Funktionen außerhalb der Schleife
            from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
                derive_surface_plane,
                evaluate_surface_plane,
            )
            from matplotlib.path import Path
            
            print(f"[SoundFieldCalculator] Grid-Erstellung:")
            print(f"  Grid shape: X_grid={X_grid.shape}, Y_grid={Y_grid.shape}, Z_grid={Z_grid.shape}")
            print(f"  X range: [{X_grid.min():.2f}, {X_grid.max():.2f}]")
            print(f"  Y range: [{Y_grid.min():.2f}, {Y_grid.max():.2f}]")
            print(f"  Z range: [{Z_grid.min():.2f}, {Z_grid.max():.2f}]")
            print(f"  Active points: {active_points} / {total_points}")
            
            # Prüfe Z-Koordinaten für schräge Flächen
            for surface_id, surface_def in enabled_surfaces:
                # Überspringe vertikale Flächen (sie sind nicht im kombinierten Grid)
                if surface_id in surface_grids_grouped:
                    grid = surface_grids_grouped[surface_id]
                    if getattr(grid.geometry, "orientation", None) == "vertical":
                        continue
                
                points = surface_def.get("points", [])
                if len(points) < 3:
                    continue
                
                # Prüfe ob Fläche schräg ist
                model, _ = derive_surface_plane(points)
                if model is None:
                    continue
                
                mode = model.get("mode", "constant")
                if mode == "constant":
                    continue
                
                # Finde Grid-Punkte auf dieser Fläche
                poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                poly_path = Path(np.column_stack((poly_x, poly_y)))
                points_2d = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
                inside = poly_path.contains_points(points_2d)
                inside = inside.reshape(X_grid.shape)
                
                # Prüfe Z-Koordinaten
                points_with_z = np.sum((inside) & (np.abs(Z_grid) > 1e-6))
                points_in_surface = np.sum(inside)
                
                print(f"  [DEBUG Z-Grid] {surface_id} (mode={mode}):")
                print(f"    Points in surface: {points_in_surface}, Points with Z≠0: {points_with_z}")
                
                if np.any(inside):
                    # Prüfe Z-Koordinaten an Ecken
                    corners = [
                        (0, 0, "oben-links"),
                        (0, X_grid.shape[1]-1, "oben-rechts"),
                        (X_grid.shape[0]-1, 0, "unten-links"),
                        (X_grid.shape[0]-1, X_grid.shape[1]-1, "unten-rechts"),
                    ]
                    for jj, ii, name in corners:
                        if inside[jj, ii]:
                            x_val = X_grid[jj, ii]
                            y_val = Y_grid[jj, ii]
                            z_val = Z_grid[jj, ii]
                            
                            # Berechne erwarteten Z-Wert aus Plane-Model
                            z_expected = evaluate_surface_plane(model, x_val, y_val)
                            
                            print(f"    Corner {name} [jj={jj},ii={ii}]: X={x_val:.2f}, Y={y_val:.2f}, Z={z_val:.2f}, Z_expected={z_expected:.2f}, diff={abs(z_val-z_expected):.3f}")
                    
                    # Prüfe auch einige Punkte innerhalb der Fläche
                    inside_indices = np.where(inside)
                    if len(inside_indices[0]) > 0:
                        # Wähle einige zufällige Punkte
                        sample_indices = np.random.choice(len(inside_indices[0]), min(5, len(inside_indices[0])), replace=False)
                        for idx in sample_indices:
                            jj = inside_indices[0][idx]
                            ii = inside_indices[1][idx]
                            x_val = X_grid[jj, ii]
                            y_val = Y_grid[jj, ii]
                            z_val = Z_grid[jj, ii]
                            z_expected = evaluate_surface_plane(model, x_val, y_val)
                            print(f"    Sample [jj={jj},ii={ii}]: X={x_val:.2f}, Y={y_val:.2f}, Z={z_val:.2f}, Z_expected={z_expected:.2f}, diff={abs(z_val-z_expected):.3f}")
                else:
                    print(f"    ⚠️  Keine Grid-Punkte innerhalb der Surface gefunden!")
        surface_meshes = []
        surface_samples = []
        # Sichtbarkeits-Occluder aus aktuellen Surface-Definitionen ableiten
        # HINWEIS: Occlusion / Schattenbildung vorübergehend deaktiviert
        occluders: List[Dict[str, Any]] = []
        grid_points = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        surface_mask_flat = surface_mask.reshape(-1)
        # Optionaler Diagnose-Check zur Symmetrie von Grid & Masken
        if DEBUG_SOUNDFIELD:
            try:
                self._debug_check_grid_and_mask_symmetry(
                    sound_field_x,
                    sound_field_y,
                    X_grid,
                    Y_grid,
                    Z_grid,
                    surface_mask,
                    enabled_surfaces,
                )
            except Exception as e:
                print(f"[SoundFieldCalculator] Symmetrie-Check konnte nicht ausgeführt werden: {e}")

        surface_field_buffers: Dict[str, np.ndarray] = {}
        surface_point_buffers: Dict[str, np.ndarray] = {}
        # 🎯 NEU: Pufferspeicher für direkt berechnete Surface-Grids (ohne Interpolation)
        surface_results_buffers: Dict[str, np.ndarray] = {}
        surface_grid_cache: Dict[str, Dict[str, Any]] = {}
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
        # Lege pro Surface einen eigenen Grid-Puffer an, um SPL direkt dort zu berechnen
        if surface_grids_grouped:
            for sid, grid in surface_grids_grouped.items():
                try:
                    Xg = np.asarray(grid.X_grid, dtype=float)
                    Yg = np.asarray(grid.Y_grid, dtype=float)
                    Zg = np.asarray(grid.Z_grid, dtype=float)
                    if Xg.size == 0 or Yg.size == 0 or Zg.size == 0:
                        continue
                    surface_results_buffers[sid] = np.zeros_like(Zg, dtype=complex)
                    surface_grid_cache[sid] = {
                        "X": Xg,
                        "Y": Yg,
                        "Z": Zg,
                        "points": np.stack((Xg, Yg, Zg), axis=-1).reshape(-1, 3),
                        "mask_flat": np.asarray(grid.surface_mask, dtype=bool).reshape(-1),
                    }
                except Exception:
                    # Wenn etwas schiefgeht, einfach keinen Buffer anlegen
                    continue
        
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

                    # 🎯 MASKEN-LOGIK: Nur Punkte auf aktiven Surfaces berechnen
                    mask_options = {
                        "min_distance": 0.001,
                    }
                    # Sichtbarkeitsmaske (Raytracing gegen vertikale Flächen)
                    # Occlusion-Logik aktuell deaktiviert: nur Surface-Maske verwenden
                    if surface_mask_flat is not None:
                        mask_options["additional_mask"] = surface_mask_flat
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
                    # 🎯 NEU: Berechne das Schallfeld direkt auf jedem Surface-Grid
                    if surface_results_buffers:
                        for sid, cache in surface_grid_cache.items():
                            try:
                                Xg = cache["X"]
                                Yg = cache["Y"]
                                Zg = cache["Z"]
                                points_surface = cache["points"]
                                mask_flat_surface = cache.get("mask_flat")

                                x_dist_s = Xg - source_position_x[isrc]
                                y_dist_s = Yg - source_position_y[isrc]
                                horizontal_s = np.sqrt(x_dist_s**2 + y_dist_s**2)
                                z_dist_s = Zg - source_position_z[isrc]
                                source_dists_s = np.sqrt(horizontal_s**2 + z_dist_s**2)

                                azimuths_s = np.arctan2(y_dist_s, x_dist_s)
                                azimuths_s = (np.degrees(azimuths_s) + np.degrees(source_azimuth[isrc])) % 360
                                azimuths_s = (360 - azimuths_s) % 360
                                azimuths_s = (azimuths_s + 90) % 360
                                elevations_s = np.degrees(np.arctan2(z_dist_s, horizontal_s))

                                gains_s, phases_s = self.get_balloon_data_batch(
                                    speaker_name,
                                    azimuths_s,
                                    elevations_s,
                                )
                                if gains_s is None or phases_s is None:
                                    continue

                                magnitude_linear_s = 10 ** (gains_s / 20)
                                polar_phase_rad_s = np.radians(phases_s)

                                source_props_surface = {
                                    "magnitude_linear": magnitude_linear_s.reshape(-1),
                                    "polar_phase_rad": polar_phase_rad_s.reshape(-1),
                                    "source_level": source_level[isrc],
                                    "a_source_pa": a_source_pa,
                                    "wave_number": wave_number,
                                    "frequency": calculate_frequency,
                                    "source_time": source_time[isrc],
                                    "polarity": polarity_flag,
                                    "distances": source_dists_s.reshape(-1),
                                }

                                mask_options_surface = {
                                    "min_distance": 0.001,
                                }
                                if mask_flat_surface is not None:
                                    mask_options_surface["additional_mask"] = mask_flat_surface

                                wave_flat_surface = self._compute_wave_for_points(
                                    points_surface,
                                    source_position,
                                    source_props_surface,
                                    mask_options_surface,
                                )
                                surface_results_buffers[sid] += wave_flat_surface.reshape(Xg.shape)
                            except Exception:
                                # Fehler in der Surface-Berechnung sollen den Hauptpfad nicht stoppen
                                continue
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

                            # Occlusion-Logik aktuell deaktiviert
                            mask_options_samples = {"min_distance": 0.001}
                            sample_wave = self._compute_wave_for_points(
                                coords,
                                source_position,
                                sample_props,
                                mask_options_samples,
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
        elif surface_samples:
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

        # 🎯 NEU: Verwende direkt berechnete Surface-Grids, fallback auf Interpolation
        if surface_results_buffers:
            if DEBUG_SOUNDFIELD:
                print(f"[DEBUG surface_results] surface_results_buffers: {list(surface_results_buffers.keys())}")
            for surface_id, buffer in surface_results_buffers.items():
                # Prüfe ob Buffer nicht leer ist
                buffer_array = np.array(buffer, dtype=complex)
                if DEBUG_SOUNDFIELD:
                    orientation = None
                    if surface_id in surface_grids_grouped:
                        orientation = getattr(surface_grids_grouped[surface_id].geometry, "orientation", None)
                    non_zero_count = np.count_nonzero(np.abs(buffer_array))
                    print(f"[DEBUG surface_results] Surface '{surface_id}' (orientation={orientation}): Buffer shape={buffer_array.shape}, non-zero values={non_zero_count}/{buffer_array.size}")
                surface_results_data[surface_id] = {
                    'sound_field_p': buffer_array.tolist(),
                }
        else:
            # Fallback: Interpolation aus dem globalen Grid
            from scipy.interpolate import griddata
            for surface_id, grid in surface_grids_grouped.items():
                if surface_id not in surface_grids_data:
                    continue
                
                # Interpoliere sound_field_p auf das Surface-Grid
                points_orig = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                values_orig = sound_field_p.ravel()
                points_new = np.column_stack([grid.X_grid.ravel(), grid.Y_grid.ravel()])
                
                # Interpoliere komplexe Werte (Real- und Imaginärteil separat)
                if np.any(values_orig != 0):
                    real_interp = griddata(
                        points_orig,
                        np.real(values_orig),
                        points_new,
                        method='linear',
                        fill_value=0.0
                    )
                    imag_interp = griddata(
                        points_orig,
                        np.imag(values_orig),
                        points_new,
                        method='linear',
                        fill_value=0.0
                    )
                    sound_field_p_surface = (real_interp + 1j * imag_interp).reshape(grid.X_grid.shape)
                else:
                    sound_field_p_surface = np.zeros_like(grid.X_grid, dtype=complex)
                
                surface_results_data[surface_id] = {
                    'sound_field_p': np.array(sound_field_p_surface).tolist(),
                }
        
        if isinstance(self.calculation_spl, dict):
            # 🎯 ALTE DATEN AUSKOMMENTIERT - Verwende nur neue Struktur
            # self.calculation_spl['sound_field_z'] = Z_grid.tolist()  # ALT
            # self.calculation_spl['surface_mask'] = surface_mask.astype(bool).tolist()  # ALT
            # self.calculation_spl['surface_mask_strict'] = surface_mask_strict.astype(bool).tolist()  # ALT
            if surface_meshes:
                self.calculation_spl['surface_meshes'] = [
                    mesh.to_payload() for mesh in surface_meshes
                ]
            self.calculation_spl['surface_samples'] = sample_payloads
            self.calculation_spl['surface_fields'] = {
                surface_id: values.tolist()
                for surface_id, values in surface_fields.items()
            }
            
            # 🎯 NEU: Speichere surface_grids_data und surface_results_data für Plot-Modul
            self.calculation_spl['surface_grids'] = surface_grids_data
            self.calculation_spl['surface_results'] = surface_results_data

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
            valid_mask &= additional_mask_array
        
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
        - valid=True: Surface ist gültig (Validierung + Triangulation für 4+ Punkte)
        
        Nur Surfaces, die alle Bedingungen erfüllen, werden in die Berechnung einbezogen.
        Die Koordination (wann Empty Plot, wann Berechnung) erfolgt über
        WindowPlotsMainwindow.update_plots_for_surface_state().
        
        Returns:
            Liste von Tupeln (surface_id, surface_definition) - nur enabled + nicht-hidden + gültige Surfaces
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
                # Fallback deaktiviert: nicht mehr still Attribute abgreifen,
                # um fehlerhafte/grenzwertige Surface-Objekte sichtbar zu machen.
                surface_data = {}
            
            # Prüfe enabled und hidden
            if not (surface_data.get('enabled', False) and not surface_data.get('hidden', False)):
                continue
            
            # 🎯 VALIDIERUNG: Prüfe ob Surface für SPL-Berechnung verwendet werden kann
            try:
                # Erstelle SurfaceDefinition-Objekt für Validierung
                if isinstance(surface_def, SurfaceDefinition):
                    surface_obj = surface_def
                else:
                    surface_obj = SurfaceDefinition.from_dict(str(surface_id), surface_data)
                
                # Validiere Surface
                validation_result = validate_and_optimize_surface(
                    surface_obj,
                    round_to_cm=False,
                    remove_redundant=False,
                )
                
                is_valid_for_spl = validation_result.is_valid
                
                # 🎯 ZUSÄTZLICHE PRÜFUNG: Prüfe ob Triangulation möglich ist
                points = surface_data.get('points', [])
                
                # Prüfe ob bereits Grids existieren und ob Grid-Triangulation erfolgreich war
                grid_triangulation_failed = False
                if hasattr(self, 'calculation_spl') and isinstance(self.calculation_spl, dict):
                    surface_grids = self.calculation_spl.get('surface_grids', {})
                    if surface_id in surface_grids:
                        grid = surface_grids[surface_id]
                        # Prüfe triangulated_success aus dem Grid
                        if hasattr(grid, 'triangulated_success'):
                            if not grid.triangulated_success:
                                grid_triangulation_failed = True
                                if DEBUG_SOUNDFIELD:
                                    print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}': Grid-Triangulation fehlgeschlagen (triangulated_success=False)")
                        # Prüfe auch, ob aktive Grid-Punkte vorhanden sind
                        if hasattr(grid, 'surface_mask'):
                            active_points = np.sum(grid.surface_mask)
                            if active_points == 0:
                                grid_triangulation_failed = True
                                if DEBUG_SOUNDFIELD:
                                    print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}': Keine aktiven Grid-Punkte (0 aktive Punkte)")
                
                # Prüfe Punkt-basierte Triangulation (für alle Surfaces mit 3+ Punkten)
                # WICHTIG: Auch wenn Grids existieren, sollte die Punkt-Triangulation funktionieren
                if is_valid_for_spl and not grid_triangulation_failed and len(points) >= 3:
                    try:
                        # Versuche Triangulation
                        triangles = triangulate_points(points)
                        if not triangles or len(triangles) == 0:
                            # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                            is_valid_for_spl = False
                            if DEBUG_SOUNDFIELD:
                                print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}' mit {len(points)} Punkten: Punkt-Triangulation fehlgeschlagen (keine Dreiecke)")
                    except Exception as e:
                        # Triangulation fehlgeschlagen - Surface ist nicht für SPL verwendbar
                        is_valid_for_spl = False
                        if DEBUG_SOUNDFIELD:
                            print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}' mit {len(points)} Punkten: Punkt-Triangulation fehlgeschlagen ({e})")
                
                # Wenn Grid-Triangulation fehlgeschlagen ist, Surface als ungültig markieren
                if grid_triangulation_failed:
                    is_valid_for_spl = False
                
                # Nur gültige Surfaces hinzufügen
                if is_valid_for_spl:
                enabled.append((surface_id, surface_data))
                else:
                    if DEBUG_SOUNDFIELD:
                        print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}' wird übersprungen (ungültig)")
            except Exception as e:
                # Bei Fehler in Validierung: Surface überspringen
                if DEBUG_SOUNDFIELD:
                    print(f"[DEBUG SoundFieldCalculator] Surface '{surface_id}' wird übersprungen (Validierungsfehler: {e})")
                continue
        
        return enabled

    # ============================================================
    # RAYTRACING / SICHTBARKEITSPRÜFUNG
    # ============================================================

    def _build_vertical_occluders(
        self, enabled_surfaces: List[Tuple[str, Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Baut eine Liste einfacher vertikaler Occluder-Flächen auf.

        Für Rechenaufwand-Schonung werden nur klar senkrechte Flächen
        betrachtet, deren XY-Projektion praktisch eine Linie ist:
        - XZ-Wände:   y ≈ const  → axis='y'
        - YZ-Wände:   x ≈ const  → axis='x'

        Die Fläche wird als Rechteck in (u,z) angenähert; das genügt für
        eine robuste Sichtbarkeitsprüfung.
        """
        occluders: List[Dict[str, Any]] = []
        if not enabled_surfaces:
            return occluders

        for surface_id, surface_def in enabled_surfaces:
            points = surface_def.get("points") or []
            if len(points) < 3:
                continue

            # Versuche, ein planare Z(x,y)-Fläche zu erkennen. Wenn das
            # gelingt, ist die Fläche nicht senkrecht und wird hier nicht
            # als Occluder verwendet (sie wird bereits im Z_grid berücksichtigt).
            try:
                model, _ = derive_surface_plane(points)
            except Exception:
                model = None
            if model is not None:
                # Planare "Boden"- oder Rampenflächen -> keine vertikale Wand
                continue

            xs = np.array([float(p.get("x", 0.0)) for p in points], dtype=float)
            ys = np.array([float(p.get("y", 0.0)) for p in points], dtype=float)
            zs = np.array([float(p.get("z", 0.0)) for p in points], dtype=float)

            x_span = float(np.ptp(xs))
            y_span = float(np.ptp(ys))
            z_span = float(np.ptp(zs))

            # Echte vertikale Wand hat signifikante Höhe
            if z_span <= 1e-3:
                continue

            # Fall 1: XZ-Wand (y ≈ const)
            if y_span < 1e-6 and x_span >= 1e-6:
                y_wall = float(np.mean(ys))
                occluders.append(
                    {
                        "surface_id": surface_id,
                        "axis": "y",
                        "value": y_wall,
                        "u_min": float(xs.min()),
                        "u_max": float(xs.max()),
                        "z_min": float(zs.min()),
                        "z_max": float(zs.max()),
                        # Polygon in (u,z) = (x,z)
                        "poly_u": xs.copy(),
                        "poly_v": zs.copy(),
                    }
                )
            # Fall 2: YZ-Wand (x ≈ const)
            elif x_span < 1e-6 and y_span >= 1e-6:
                x_wall = float(np.mean(xs))
                occluders.append(
                    {
                        "surface_id": surface_id,
                        "axis": "x",
                        "value": x_wall,
                        "u_min": float(ys.min()),
                        "u_max": float(ys.max()),
                        "z_min": float(zs.min()),
                        "z_max": float(zs.max()),
                        # Polygon in (u,z) = (y,z)
                        "poly_u": ys.copy(),
                        "poly_v": zs.copy(),
                    }
                )

        return occluders

    def _compute_visibility_mask_for_source(
        self,
        source_position: np.ndarray,
        grid_points: np.ndarray,
        grid_z: np.ndarray,
        occluders: List[Dict[str, Any]],
        exclude_surface_ids: Optional[set] = None,
    ) -> np.ndarray:
        """
        Berechnet für einen Lautsprecher eine Sichtbarkeitsmaske über alle
        Berechnungspunkte mittels sehr einfacher Raytracing-Logik.

        Für jede vertikale Fläche (Occluder) wird geprüft, ob der Strahl von
        der Quelle zum Punkt die Ebene der Wand schneidet UND der
        Schnittpunkt innerhalb des (u,z)-Rechtecks der Wand liegt.

        Rückgabe:
            visibility_mask (bool, Shape: [N]): True = sichtbar, False = verdeckt
        """
        if not occluders or grid_points.size == 0:
            return np.ones(grid_points.shape[0], dtype=bool)

        src = np.asarray(source_position, dtype=float).reshape(3)
        pts = np.asarray(grid_points, dtype=float).reshape(-1, 3)
        z_pts = np.asarray(grid_z, dtype=float).reshape(-1)

        # Standardmäßig alle Punkte sichtbar
        visible = np.ones(pts.shape[0], dtype=bool)

        xs, ys, zs = src[0], src[1], src[2]
        px = pts[:, 0]
        py = pts[:, 1]
        pz = z_pts  # bereits aus Z_grid übernommen

        for occ in occluders:
            surf_id = occ.get("surface_id")
            if exclude_surface_ids and surf_id in exclude_surface_ids:
                continue
            axis = occ.get("axis")
            wall_val = float(occ.get("value", 0.0))
            u_min = float(occ.get("u_min", 0.0))
            u_max = float(occ.get("u_max", 0.0))
            z_min = float(occ.get("z_min", 0.0))
            z_max = float(occ.get("z_max", 0.0))
            poly_u = np.asarray(occ.get("poly_u", []), dtype=float)
            poly_v = np.asarray(occ.get("poly_v", []), dtype=float)

            # Punkte, bei denen Strahl überhaupt die Wand-Ebene schneiden kann
            if axis == "y":
                # Ebene y = wall_val
                denom = py - ys
                # Quelle und Punkt auf derselben Seite -> kein Schnitt
                side_src = ys - wall_val
                side_pts = py - wall_val
            elif axis == "x":
                # Ebene x = wall_val
                denom = px - xs
                side_src = xs - wall_val
                side_pts = px - wall_val
            else:
                continue

            # Nur Punkte mit echter Richtungsänderung zur Wand
            different_side = (side_src * side_pts) < 0.0
            # Numerische Stabilität
            valid_denom = np.abs(denom) > 1e-9
            candidates = different_side & valid_denom
            if not np.any(candidates):
                continue

            # Schnittparameter t im Intervall (0,1)
            if axis == "y":
                t = (wall_val - ys) / denom
            else:
                t = (wall_val - xs) / denom
            between = (t > 0.0) & (t < 1.0)
            cand_idx = candidates & between
            if not np.any(cand_idx):
                continue

            # Schnittkoordinaten entlang des Strahls
            x_int = xs + t * (px - xs)
            y_int = ys + t * (py - ys)
            z_int = zs + t * (pz - zs)

            if axis == "y":
                u_int = x_int
            else:
                u_int = y_int

            # Grober Bounding-Box-Test im (u,z)-Raum
            inside_u_bb = (u_int >= (u_min - 1e-6)) & (u_int <= (u_max + 1e-6))
            inside_z_bb = (z_int >= (z_min - 1e-6)) & (z_int <= (z_max + 1e-6))
            cand_poly = cand_idx & inside_u_bb & inside_z_bb
            if not np.any(cand_poly):
                continue

            # Exakter Polygon-Test im (u,z)-Raum
            if poly_u.size >= 3 and poly_v.size >= 3:
                U = u_int[cand_poly].reshape(1, -1)
                V = z_int[cand_poly].reshape(1, -1)
                poly_mask = _points_in_polygon_batch_uv(U, V, poly_u, poly_v)
                if poly_mask is not None:
                    poly_inside = poly_mask.reshape(-1)
                    tmp = np.zeros_like(cand_poly, dtype=bool)
                    tmp[np.where(cand_poly)[0]] = poly_inside
                    cand_poly = cand_poly & tmp

            occluded_here = cand_poly
            if np.any(occluded_here):
                visible[occluded_here] = False

        return visible

    # ============================================================
    # DEBUG / VALIDIERUNG
    # ============================================================

    def _debug_check_grid_and_mask_symmetry(
        self,
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
        X_grid: np.ndarray,
        Y_grid: np.ndarray,
        Z_grid: np.ndarray,
        surface_mask: np.ndarray,
        enabled_surfaces: List[Tuple[str, Dict]],
    ) -> None:
        """
        Ehemaliger Diagnose-Check (Symmetrie von Grid & Surfaces).
        Wird aktuell nicht mehr für Konsolenausgaben verwendet.
        """
        return

