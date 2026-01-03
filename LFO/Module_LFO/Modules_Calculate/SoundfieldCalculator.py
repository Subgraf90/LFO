from __future__ import annotations

import numpy as np
import time
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _compute_surface_contribution_for_source(
        self,
        cache: Dict[str, Any],
        speaker_name: str,
        source_position: np.ndarray,
        source_position_x: float,
        source_position_y: float,
        source_position_z: float,
        source_azimuth: float,
        source_level: float,
        a_source_pa: float,
        wave_number: float,
        calculate_frequency: float,
        source_time: float,
        polarity_flag: bool,
    ) -> Optional[np.ndarray]:
        """
        Berechnet den Schallfeldbeitrag einer einzelnen Quelle für EIN Surface-Grid.
        Gibt ein Array in Shape von cache['X'] zurück oder None bei Fehlern.
        """
        try:
            Xg = cache["X"]
            Yg = cache["Y"]
            Zg = cache["Z"]
            points_surface = cache["points"]
            mask_flat_surface = cache.get("mask_flat")

            x_dist_s = Xg - source_position_x
            y_dist_s = Yg - source_position_y
            horizontal_s = np.sqrt(x_dist_s**2 + y_dist_s**2)
            z_dist_s = Zg - source_position_z
            source_dists_s = np.sqrt(horizontal_s**2 + z_dist_s**2)

            azimuths_s = np.arctan2(y_dist_s, x_dist_s)
            azimuths_s = (np.degrees(azimuths_s) + np.degrees(source_azimuth)) % 360
            azimuths_s = (360 - azimuths_s) % 360
            azimuths_s = (azimuths_s + 90) % 360
            elevations_s = np.degrees(np.arctan2(z_dist_s, horizontal_s))

            gains_s, phases_s = self.get_balloon_data_batch(
                speaker_name,
                azimuths_s,
                elevations_s,
            )
            if gains_s is None or phases_s is None:
                return None

            magnitude_linear_s = 10 ** (gains_s / 20)
            polar_phase_rad_s = np.radians(phases_s)

            source_props_surface = {
                "magnitude_linear": magnitude_linear_s.reshape(-1),
                "polar_phase_rad": polar_phase_rad_s.reshape(-1),
                "source_level": source_level,
                "a_source_pa": a_source_pa,
                "wave_number": wave_number,
                "frequency": calculate_frequency,
                "source_time": source_time,
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
            return wave_flat_surface.reshape(Xg.shape)
        except Exception:
            # Fehler in der Surface-Berechnung sollen den Hauptpfad nicht stoppen
            return None

    def _compute_nan_vertices_contribution_for_source(
        self,
        coords: np.ndarray,
        speaker_name: str,
        source_position: np.ndarray,
        source_position_x: float,
        source_position_y: float,
        source_position_z: float,
        source_azimuth: float,
        source_level: float,
        a_source_pa: float,
        wave_number: float,
        calculate_frequency: float,
        source_time: float,
        polarity_flag: bool,
    ) -> Optional[np.ndarray]:
        """
        Berechnet den Schallfeldbeitrag einer Quelle für zusätzliche NaN-Refinement-Punkte.
        Gibt ein 1D-Array (len(coords)) zurück oder None bei Fehlern.
        """
        try:
            if coords.size == 0:
                return None

            dx_nv = coords[:, 0] - source_position_x
            dy_nv = coords[:, 1] - source_position_y
            dz_nv = coords[:, 2] - source_position_z
            horizontal_nv = np.sqrt(dx_nv**2 + dy_nv**2)
            dists_nv = np.sqrt(horizontal_nv**2 + dz_nv**2)

            az_nv = (np.degrees(np.arctan2(dy_nv, dx_nv)) + np.degrees(source_azimuth)) % 360
            az_nv = (360 - az_nv) % 360
            az_nv = (az_nv + 90) % 360
            el_nv = np.degrees(np.arctan2(dz_nv, horizontal_nv))

            az_payload = az_nv.reshape(1, -1)
            el_payload = el_nv.reshape(1, -1)
            gains_nv, phases_nv = self.get_balloon_data_batch(
                speaker_name,
                az_payload,
                el_payload,
            )
            if gains_nv is None or phases_nv is None:
                return None

            props_nv = {
                "magnitude_linear": (10 ** (gains_nv.reshape(-1) / 20)),
                "polar_phase_rad": np.radians(phases_nv.reshape(-1)),
                "source_level": source_level,
                "a_source_pa": a_source_pa,
                "wave_number": wave_number,
                "frequency": calculate_frequency,
                "source_time": source_time,
                "polarity": polarity_flag,
                "distances": dists_nv,
            }
            mask_opts_nv = {"min_distance": 0.001}
            wave_nv = self._compute_wave_for_points(
                coords,
                source_position,
                props_nv,
                mask_opts_nv,
            )
            return wave_nv.reshape(-1)
        except Exception:
            return None
   
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

        # Keine bandgemittelten Daten verfügbar
        return None, None

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
        
        # Keine bandgemittelten Daten verfügbar
        return None, None

    @measure_time("SoundFieldCalculator._calculate_sound_field_complex")
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
        # Stelle sicher, dass np verfügbar ist (lokale Referenz auf globales Modul)
        import numpy as np_local
        np = np_local  # Lokale Referenz, damit Python weiß, dass np lokal verfügbar ist
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
        
        # Erstelle Mapping: surface_id -> surface_definition für schnellen Zugriff
        surface_def_map = {sid: sdef for sid, sdef in enabled_surfaces}
        
        # Verwende generate_per_surface() um pro Surface Grids zu bekommen
        
        with perf_section(
            "SoundFieldCalculator._calculate_sound_field_complex.generate_per_surface",
            n_surfaces=len(enabled_surfaces),
        ):
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
        
        if not surface_grids_grouped:
            return [], np.array([]), np.array([]), ({} if capture_arrays else None)
        
        # 🎯 ERSTELLE SURFACE_GRIDS_DATA für Plot-Modul
        surface_grids_data = {}
        surface_results_data = {}
        
        # Kombiniere alle Grids zu einem gemeinsamen Grid für Rückwärtskompatibilität
        with perf_section(
            "SoundFieldCalculator._calculate_sound_field_complex.combine_surface_grids",
            n_surface_grids=len(surface_grids_grouped),
        ):
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
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json, time as _t
                        # Berechne Gesamtanzahl Gridpunkte aus einzelnen Surfaces
                        total_individual_grid_points = sum(
                            int(grid.X_grid.size) if hasattr(grid, 'X_grid') else 0
                            for grid in surface_grids_grouped.values()
                        )
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "grid-analysis-v2",
                            "hypothesisId": "H1",
                            "location": "SoundfieldCalculator._calculate_sound_field_complex",
                            "message": "Kombiniertes Grid erstellt",
                            "data": {
                                "n_surfaces": len(surface_grids_grouped),
                                "unique_x_count": int(len(unique_x)),
                                "unique_y_count": int(len(unique_y)),
                                "combined_grid_points": int(X_grid_combined.size),
                                "combined_grid_shape": list(X_grid_combined.shape),
                                "total_individual_grid_points": int(total_individual_grid_points),
                                "grid_duplication_factor": float(total_individual_grid_points / X_grid_combined.size) if X_grid_combined.size > 0 else 0.0
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
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

            _axis_summary("sound_field_x", sound_field_x)
            _axis_summary("sound_field_y", sound_field_y)
        except Exception as e:
            pass
        
        # Kombiniere Z-Grids und Masken aus einzelnen Surface-Grids
        from scipy.interpolate import griddata

        # 🔧 PERFORMANCE: Arbeite im kombinierten Grid nur innerhalb der Bounding-Box jeder Surface,
        # statt für JEDE Surface ALLE Punkte des globalen Grids zu interpolieren.

        # Vorab: flache Views des kombinierten Grids, damit wir Teilbereiche effizient adressieren können
        X_flat = X_grid.ravel()
        Y_flat = Y_grid.ravel()
        Z_combined_flat = Z_grid_combined.ravel()
        mask_combined_flat = surface_mask_combined.ravel()

        with perf_section(
            "SoundFieldCalculator._calculate_sound_field_complex.interpolate_surfaces",
            n_surface_grids=len(surface_grids_grouped),
        ):
            from concurrent.futures import ThreadPoolExecutor, as_completed

            total_surfaces = len(surface_grids_grouped)
            t_start_total = time.perf_counter()

            # Max. Threads aus Settings, wie bei generate_per_surface
            max_workers = int(getattr(self.settings, "spl_parallel_surfaces", 0) or 0)
            if max_workers <= 0:
                max_workers = None

            def _interpolate_single_surface(args):
                surface_id, grid, surface_idx = args
                t_surface_start = time.perf_counter()

                orientation = getattr(grid.geometry, "orientation", None)

                idx_bbox = None
                z_interp_sub = None
                mask_interp_sub = None

                # 🎯 IDENTISCHE BEHANDLUNG: Vertikale Flächen haben separate Grids (andere Koordinatenebene)
                # Sie werden nicht ins kombinierte (x,y)-Grid gemischt, aber gleich behandelt
                if orientation != "vertical":
                    # Interpoliere nur innerhalb der Bounding-Box der Surface ins kombinierte Grid
                    t_prep = time.perf_counter()

                    Xg = np.asarray(grid.X_grid, dtype=float)
                    Yg = np.asarray(grid.Y_grid, dtype=float)
                    Zg = np.asarray(grid.Z_grid, dtype=float)
                    mask_orig = np.asarray(grid.surface_mask, dtype=bool).ravel()

                    if Xg.size == 0 or Yg.size == 0 or Zg.size == 0 or not np.any(mask_orig):
                        # Nichts zu tun für diese Surface
                        pass
                    else:
                        # Ursprungs-Punkte (lokales Surface-Grid)
                        points_orig = np.column_stack([Xg.ravel(), Yg.ravel()])
                        z_orig = Zg.ravel()

                        # Bounding-Box in X/Y bestimmen
                        min_x = float(Xg.min())
                        max_x = float(Xg.max())
                        min_y = float(Yg.min())
                        max_y = float(Yg.max())

                        # Finde Punkte im globalen Grid innerhalb dieser Bounding-Box
                        inside_bbox = (
                            (X_flat >= min_x) & (X_flat <= max_x) &
                            (Y_flat >= min_y) & (Y_flat <= max_y)
                        )

                        # Verwende explizite Indizes, damit die Haupt-Thread-Logik Ergebnisse zielgerichtet zurückschreiben kann
                        idx_bbox = np.nonzero(inside_bbox)[0]
                        n_bbox = int(idx_bbox.size)
                        t_prep_duration = (time.perf_counter() - t_prep) * 1000

                        if n_bbox == 0:
                            # Diese Surface schneidet das globale Grid nicht
                            idx_bbox = None
                        else:
                            points_new_sub = np.column_stack(
                                [X_flat[idx_bbox], Y_flat[idx_bbox]]
                            )

                            # Interpoliere Z-Werte (nur aktive Masken-Punkte der Surface)
                            t_z_start = time.perf_counter()
                            z_interp_sub = griddata(
                                points_orig[mask_orig],
                                z_orig[mask_orig],
                                points_new_sub,
                                method='nearest',
                                fill_value=0.0,
                            )
                            t_z_duration = (time.perf_counter() - t_z_start) * 1000

                            # Kombiniere Masken (auf gesamter Surface, aber ebenfalls nur in der BBox)
                            t_mask_start = time.perf_counter()
                            mask_interp_sub = griddata(
                                points_orig,
                                mask_orig.astype(float),
                                points_new_sub,
                                method='nearest',
                                fill_value=0.0,
                            )
                            t_mask_duration = (time.perf_counter() - t_mask_start) * 1000

                # 🎯 ERSTELLE SURFACE_GRIDS_DATA für Plot-Modul (für ALLE Flächen, auch vertikale)
                # 🎯 NEU: Füge triangulierte Daten hinzu
                t_dict_start = time.perf_counter()
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
                
                # #region agent log - Randpunkte in surface_grids_data
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json, time as _t
                        mask_active = int(np.count_nonzero(grid.surface_mask))
                        mask_size = int(grid.surface_mask.size)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "BOUNDARY_POINTS_GRIDS",
                            "location": "SoundfieldCalculator._calculate_sound_field_complex:surface_grids_data",
                            "message": "Randpunkte in surface_grids_data prüfen",
                            "data": {
                                "surface_id": str(surface_id),
                                "surface_mask_active_points": mask_active,
                                "surface_mask_size": mask_size,
                                "grid_shape": list(grid.X_grid.shape) if hasattr(grid, 'X_grid') else None,
                                "grid_size": int(grid.X_grid.size) if hasattr(grid, 'X_grid') else None
                            },
                            "timestamp": int(_t.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                t_dict_duration = (time.perf_counter() - t_dict_start) * 1000

                # 🎯 TRIANGULATION: Füge triangulierte Vertices und Faces hinzu
                t_tri_start = time.perf_counter()
                if hasattr(grid, 'triangulated_vertices') and grid.triangulated_vertices is not None:
                    grid_data['triangulated_vertices'] = grid.triangulated_vertices.tolist()
                if hasattr(grid, 'triangulated_faces') and grid.triangulated_faces is not None:
                    grid_data['triangulated_faces'] = grid.triangulated_faces.tolist()
                if hasattr(grid, 'triangulated_success'):
                    grid_data['triangulated_success'] = grid.triangulated_success
                t_tri_duration = (time.perf_counter() - t_tri_start) * 1000

                t_surface_duration = (time.perf_counter() - t_surface_start) * 1000

                # Rückgabe zur Weiterverarbeitung im Haupt-Thread
                return surface_id, idx_bbox, z_interp_sub, mask_interp_sub, grid_data

            # Parallel über Surfaces interpolieren und anschließend Ergebnisse konsolidieren
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for idx, (surface_id, grid) in enumerate(
                    surface_grids_grouped.items(), start=1
                ):
                    fut = executor.submit(
                        _interpolate_single_surface,
                        (surface_id, grid, idx),
                    )
                    futures[fut] = surface_id

                for fut in as_completed(futures):
                    try:
                        surface_id, idx_bbox, z_interp_sub, mask_interp_sub, grid_data = fut.result()
                    except Exception as e:
                        print(
                            f"⚠️  [SoundFieldCalculator] Surface '{futures[fut]}' "
                            f"übersprungen (Interpolation-Fehler: {e})"
                        )
                        continue

                    # Ergebnisse der Interpolation zurück ins kombinierte Grid schreiben (nur Haupt-Thread!)
                    if idx_bbox is not None and z_interp_sub is not None:
                        Z_combined_flat[idx_bbox] = np.maximum(
                            Z_combined_flat[idx_bbox],
                            z_interp_sub,
                        )
                    if idx_bbox is not None and mask_interp_sub is not None:
                        mask_combined_flat[idx_bbox] |= (mask_interp_sub > 0.5)

                    surface_grids_data[surface_id] = grid_data
        
        # Nach der Bearbeitung über flache Views wieder in die ursprüngliche Grid-Form bringen
        Z_grid = Z_combined_flat.reshape(X_grid.shape)
        surface_mask = mask_combined_flat.reshape(X_grid.shape)
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
            
            # Prüfe Z-Koordinaten für schräge Flächen
            for surface_id, surface_def in enabled_surfaces:
                # 🎯 IDENTISCHE BEHANDLUNG: Alle Flächen werden geprüft
                # (vertikale Flächen haben separate Grids, werden aber gleich behandelt)
                if surface_id in surface_grids_grouped:
                    grid = surface_grids_grouped[surface_id]
                    orientation = getattr(grid.geometry, "orientation", None)
                    # Vertikale Flächen haben separate Grids (andere Koordinatenebene), 
                    # werden aber mit der gleichen Logik behandelt
                    if orientation == "vertical":
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
        
        # 🎯 NEU: EINZELBERECHNUNG PRO SURFACE
        # Statt kombinierte Grid-Berechnung verwenden wir jetzt für jede Surface einzeln
        # _calculate_sound_field_for_surface_grid, damit Boundary-Punkte verwendet werden
        
        # Initialisiere das Schallfeld als 2D-Array komplexer Zahlen (für Rückwärtskompatibilität)
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
        
        phys_constants = {
            'speed_of_sound': speed_of_sound,
            'wave_number': wave_number,
            'calculate_frequency': calculate_frequency,
            'a_source_pa': a_source_pa,
        }
        
        # Speichere Minimum und Maximum der berechneten Pegel für Vergleich
        min_level = float('inf')
        max_level = float('-inf')
        
        # ============================================================
        # SCHRITT 3: 🎯 NEU: EINZELBERECHNUNG PRO SURFACE
        # ============================================================
        # Berechne für jede Surface einzeln mit _calculate_sound_field_for_surface_grid
        # Dies stellt sicher, dass Boundary-Punkte verwendet werden
        # #region agent log - Einzelberechnung pro Surface
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json, time as _t
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "INDIVIDUAL_SURFACE_CALC",
                    "location": "SoundfieldCalculator._calculate_sound_field_complex:individual_surfaces",
                    "message": "Starte Einzelberechnung pro Surface",
                    "data": {
                        "n_surfaces": len(surface_grids_grouped),
                        "surface_ids": list(surface_grids_grouped.keys())
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion

        # Berechne für jede Surface einzeln
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = int(getattr(self.settings, "spl_parallel_surfaces", 0) or 0)
        if max_workers <= 0:
            max_workers = None
        
        # Initialisiere surface_results_buffers falls noch nicht vorhanden
        if not surface_results_buffers and surface_grids_grouped:
            for sid, grid in surface_grids_grouped.items():
                try:
                    Xg = np.asarray(grid.X_grid, dtype=float)
                    if Xg.size == 0:
                        continue
                    surface_results_buffers[sid] = np.zeros_like(Xg, dtype=complex)
                except Exception:
                    continue
        
        # Für jede Surface einzeln berechnen
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for surface_id, grid in surface_grids_grouped.items():
                if surface_id not in surface_results_buffers:
                    continue
                fut = executor.submit(
                    self._calculate_sound_field_for_surface_grid,
                    grid,
                    phys_constants,
                    capture_arrays
                )
                futures[fut] = surface_id
            
            # Sammle Ergebnisse
            for fut in as_completed(futures):
                surface_id = futures[fut]
                try:
                    sound_field_p_surface, array_fields_surface = fut.result()
                    # Speichere Ergebnis in surface_results_buffers
                    if sound_field_p_surface is not None and surface_id in surface_results_buffers:
                        surface_results_buffers[surface_id] = sound_field_p_surface
                        # #region agent log - Einzelberechnung erfolgreich
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json, time as _t
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "INDIVIDUAL_SURFACE_CALC",
                                    "location": "SoundfieldCalculator._calculate_sound_field_complex:surface_completed",
                                    "message": "Einzelberechnung für Surface abgeschlossen",
                                    "data": {
                                        "surface_id": str(surface_id),
                                        "result_shape": list(sound_field_p_surface.shape) if sound_field_p_surface is not None else None,
                                        "result_nonzero": int(np.count_nonzero(sound_field_p_surface)) if sound_field_p_surface is not None else None
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                except Exception as e:
                    # Fehler in der Surface-Berechnung sollen den Hauptpfad nicht stoppen
                    print(f"⚠️  [SoundFieldCalculator] Fehler bei Berechnung für Surface '{surface_id}': {e}")
                    continue

        # ============================================================
        # SCHRITT 4: ALTE KOMBINIERTE GRID-BERECHNUNG ENTFERNT
        # ============================================================
        # Die Berechnung auf dem kombinierten Grid wurde komplett entfernt.
        # Stattdessen wird für jede Surface einzeln berechnet (siehe Schritt 3).
        # Dies stellt sicher, dass Boundary-Punkte in den Surface-Masken verwendet werden.
        
        # ============================================================
        # HINWEIS: ALTE KOMBINIERTE GRID-BERECHNUNG ENTFERNT
        # ============================================================
        # Die Berechnung auf dem kombinierten Grid (X_grid, Y_grid) wurde entfernt.
        # Stattdessen wird für jede Surface einzeln über _calculate_sound_field_for_surface_grid
        # berechnet. Die Ergebnisse werden in surface_results_buffers gespeichert.
        
        # sound_field_p (kombiniertes Grid) bleibt für Rückwärtskompatibilität leer.
        # Alle Berechnungen laufen jetzt pro Surface mit Boundary-Punkten.
        
        # #endregion
        
        # ============================================================
        # ALTE KOMBINIERTE GRID-BERECHNUNG ENTFERNT
        # ============================================================
        # Die gesamte Schleife über Arrays und Sources für das kombinierte Grid
        # wurde entfernt. Stattdessen wird für jede Surface einzeln
        # _calculate_sound_field_for_surface_grid aufgerufen (siehe Schritt 3 oben).
        
        # Die nachfolgenden Zeilen sind nur noch für Rückwärtskompatibilität
        # und werden nach der Einzelberechnung ausgeführt.
        
        # ============================================================
        # ALTE KOMBINIERTE GRID-BERECHNUNG KOMPLETT ENTFERNT
        # ============================================================
        # Die gesamte Schleife über Arrays/Sources für das kombinierte Grid
        # wurde entfernt. Stattdessen wird für jede Surface einzeln
        # _calculate_sound_field_for_surface_grid aufgerufen (siehe Schritt 3).
        # 
        # sound_field_p bleibt leer (für Rückwärtskompatibilität initialisiert).
        # Alle Berechnungen laufen pro Surface mit Boundary-Punkten.
        
        # ============================================================
        # SCHRITT 5: Ergebnisse speichern
        # ============================================================
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

        # 🎯 Verwende direkt berechnete Surface-Grids für alle Surfaces
        # Es kann Fälle geben, in denen keine Flächen aktiv sind (z.B. nur Arrays ohne Flächen).
        # Dann bleibt surface_results_data leer und wir brechen später im Plot einfach ruhig ab.
        if surface_results_buffers:
            for surface_id, buffer in surface_results_buffers.items():
                # Puffer immer in komplexes Array konvertieren
                buffer_array = np.asarray(buffer, dtype=complex)
                
                # #region agent log - Randpunkte in surface_results_data
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json, time as _t
                        # Prüfe, ob Randpunkte in den SPL-Werten sind
                        if surface_id in surface_grids_data:
                            grid_data_check = surface_grids_data[surface_id]
                            surface_mask_check = np.asarray(grid_data_check.get("surface_mask", []), dtype=bool)
                            if surface_mask_check.size > 0 and buffer_array.size > 0:
                                mask_active = int(np.count_nonzero(surface_mask_check))
                                buffer_nonzero = int(np.count_nonzero(buffer_array))
                                buffer_finite = int(np.sum(np.isfinite(buffer_array)))
                                # Prüfe, ob SPL-Werte für aktive Masken-Punkte vorhanden sind
                                if surface_mask_check.size == buffer_array.size:
                                    spl_at_mask = buffer_array.ravel()[surface_mask_check.ravel()]
                                    spl_at_mask_nonzero = int(np.count_nonzero(spl_at_mask)) if spl_at_mask.size > 0 else 0
                                    spl_at_mask_finite = int(np.sum(np.isfinite(spl_at_mask))) if spl_at_mask.size > 0 else 0
                                else:
                                    spl_at_mask_nonzero = None
                                    spl_at_mask_finite = None
                                
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "BOUNDARY_POINTS_RESULTS",
                                    "location": "SoundfieldCalculator._calculate_sound_field_complex:surface_results_data",
                                    "message": "Randpunkte in surface_results_data prüfen",
                                    "data": {
                                        "surface_id": str(surface_id),
                                        "surface_mask_active_points": mask_active,
                                        "surface_mask_size": int(surface_mask_check.size),
                                        "buffer_size": int(buffer_array.size),
                                        "buffer_nonzero": buffer_nonzero,
                                        "buffer_finite": buffer_finite,
                                        "sizes_match": surface_mask_check.size == buffer_array.size,
                                        "spl_at_mask_nonzero": spl_at_mask_nonzero,
                                        "spl_at_mask_finite": spl_at_mask_finite
                                    },
                                    "timestamp": int(_t.time() * 1000)
                                }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                surface_results_data[surface_id] = {
                    "sound_field_p": buffer_array.tolist(),
                    "is_group_sum": False,
                }

        # 🚫 KEINE gruppenweise Summenbildung mehr:
        # Die berechneten Gridpunkte pro Surface werden unverändert an den Plot übergeben.

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
    
    # 🗑️ ENTFERNT: _identify_group_candidates - nicht mehr benötigt, da alle Surfaces identisch behandelt werden
    
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

    @measure_time("SoundFieldCalculator._calculate_sound_field_for_surface_grid")
    def _calculate_sound_field_for_surface_grid(
        self,
        surface_grid: 'SurfaceGrid',
        phys_constants: Dict[str, Any],
        capture_arrays: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        🚀 PROZESSOPTIMIERT: Berechnet das Schallfeld für ein einzelnes Surface-Grid.
        
        Diese Methode führt die vollständig vektorisierte Berechnung für ein Surface-Grid durch.
        Alle Optimierungen (Batch-Interpolation, vektorisierte Operationen) bleiben erhalten.
        
        Args:
            surface_grid: SurfaceGrid-Objekt mit X_grid, Y_grid, Z_grid, surface_mask
            phys_constants: Dictionary mit physikalischen Konstanten (wird einmal berechnet)
            capture_arrays: Wenn True, werden Array-Felder zurückgegeben
        
        Returns:
            Tuple von (sound_field_p, array_fields_dict)
            - sound_field_p: 2D-Array komplexer Zahlen (Shape: [ny, nx])
            - array_fields_dict: Dict von {array_key: array_wave} (wenn capture_arrays=True)
        """
        from Module_LFO.Modules_Calculate.FlexibleGridGenerator import SurfaceGrid
        
        # Extrahiere Grid-Daten
        X_grid = surface_grid.X_grid
        Y_grid = surface_grid.Y_grid
        Z_grid = surface_grid.Z_grid
        surface_mask = surface_grid.surface_mask
        # #region agent log - Maske in Berechnung
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json, time as _t
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "SoundfieldCalculator._calculate_sound_field_for_surface_grid:mask_check",
                    "message": "Maske in Berechnung",
                    "data": {
                        "surface_id": str(surface_id),
                        "mask_shape": list(surface_mask.shape) if hasattr(surface_mask, 'shape') else None,
                        "mask_size": int(surface_mask.size) if hasattr(surface_mask, 'size') else None,
                        "active_points_in_mask": int(np.count_nonzero(surface_mask)) if hasattr(surface_mask, 'size') else None,
                        "total_grid_points": int(X_grid.size) if hasattr(X_grid, 'size') else None
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 🎯 IDENTISCHE BEHANDLUNG: Vertikale und planare Flächen verwenden die gleiche Berechnung
        surface_id = surface_grid.geometry.surface_id
        orientation = surface_grid.geometry.orientation
        
        # #region agent log
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                "message": "Single surface - grid parameters",
                "data": {
                    "surface_id": str(surface_id),
                    "grid_shape": {"X": list(X_grid.shape), "Y": list(Y_grid.shape), "Z": list(Z_grid.shape)},
                    "grid_size": int(X_grid.size),
                    "x_range": [float(np.min(X_grid)), float(np.max(X_grid))],
                    "y_range": [float(np.min(Y_grid)), float(np.max(Y_grid))],
                    "z_range": [float(np.min(Z_grid)), float(np.max(Z_grid))],
                    "mask_points": int(np.sum(surface_mask)),
                    "mask_shape": list(surface_mask.shape)
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
        # #endregion
        
        # Initialisiere Schallfeld
        ny, nx = X_grid.shape
        sound_field_p = np.zeros((ny, nx), dtype=complex)
        array_fields = {} if capture_arrays else None
        
        # Extrahiere physikalische Konstanten (bereits berechnet)
        speed_of_sound = phys_constants['speed_of_sound']
        wave_number = phys_constants['wave_number']
        calculate_frequency = phys_constants['calculate_frequency']
        a_source_pa = phys_constants['a_source_pa']
        
        # Bereite Grid-Punkte vor
        grid_points = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        surface_mask_flat = surface_mask.reshape(-1)
        
        # 🎯 VERWENDE NUR SURFACE_MASK: Die Maske enthält bereits alle gewünschten Punkte
        # (Surface-Fläche + Randpunkte + Eckenpunkte)
        # Keine separate erweiterte Maske mehr nötig
        # #region agent log - Surface-Maske für SPL-Berechnung
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json, time as _t
                surface_mask_active = int(np.count_nonzero(surface_mask_flat))
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H1,H2",
                    "location": "SoundfieldCalculator._calculate_sound_field_for_surface_grid:surface_mask",
                    "message": "Surface-Maske für SPL-Berechnung",
                    "data": {
                        "surface_id": str(surface_id),
                        "surface_mask_active_points": surface_mask_active,
                        "surface_mask_size": int(surface_mask_flat.size),
                        "includes_edge_and_corner_points": True
                    },
                    "timestamp": int(_t.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # Iteriere über alle Lautsprecher-Arrays
        # #region agent log
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            import json
            all_arrays = list(self.settings.speaker_arrays.items())
            active_arrays = [(k, a) for k, a in all_arrays if not (a.mute or a.hide)]
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D",
                "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                "message": "Source filtering - single surface calculation",
                "data": {
                    "surface_id": str(surface_id),
                    "total_arrays": len(all_arrays),
                    "active_arrays_count": len(active_arrays),
                    "active_array_keys": [k for k, _ in active_arrays],
                    "skipped_arrays": [k for k, a in all_arrays if (a.mute or a.hide)],
                    "skipped_mute": [k for k, a in all_arrays if a.mute],
                    "skipped_hide": [k for k, a in all_arrays if a.hide]
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
        # #endregion
        for array_key, speaker_array in self.settings.speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            # #region agent log
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                    "message": "Processing source in single surface calculation",
                    "data": {
                        "surface_id": str(surface_id),
                        "array_key": str(array_key),
                        "source_count": len(speaker_array.source_polar_pattern),
                        "mute": bool(speaker_array.mute),
                        "hide": bool(speaker_array.hide)
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion
            
            array_wave = np.zeros_like(sound_field_p, dtype=complex) if capture_arrays else None
            
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
                source_time = [speaker_array.source_time + source_delay]
            else:
                source_time = [t + source_delay for t in speaker_array.source_time]
            source_time = [x / 1000 for x in source_time]
            source_gain = speaker_array.gain
            source_level = speaker_array.source_level + source_gain
            source_level = self.functions.db2mag(np.array(source_level))
            
            # #region agent log
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                    "message": "Source parameters for single surface calculation",
                    "data": {
                        "surface_id": str(surface_id),
                        "array_key": str(array_key),
                        "source_polar_pattern": list(speaker_array.source_polar_pattern) if hasattr(speaker_array, 'source_polar_pattern') else [],
                        "source_count": len(speaker_array.source_polar_pattern) if hasattr(speaker_array, 'source_polar_pattern') else 0,
                        "source_level_raw": float(speaker_array.source_level) if isinstance(speaker_array.source_level, (int, float)) else list(speaker_array.source_level) if hasattr(speaker_array.source_level, '__iter__') else None,
                        "source_gain": float(source_gain) if isinstance(source_gain, (int, float)) else list(source_gain) if hasattr(source_gain, '__iter__') else None,
                        "source_level_after_gain": list(source_level) if hasattr(source_level, '__iter__') else float(source_level),
                        "source_delay": float(source_delay) if isinstance(source_delay, (int, float)) else list(source_delay) if hasattr(source_delay, '__iter__') else None,
                        "source_time_raw": float(speaker_array.source_time) if isinstance(speaker_array.source_time, (int, float)) else list(speaker_array.source_time) if hasattr(speaker_array.source_time, '__iter__') else None,
                        "source_time_after_delay": list(source_time),
                        "source_azimuth_deg": list(np.degrees(source_azimuth)) if hasattr(source_azimuth, '__iter__') else float(np.degrees(source_azimuth)),
                        "source_position_x": list(source_position_x) if hasattr(source_position_x, '__iter__') else float(source_position_x) if source_position_x is not None else None,
                        "source_position_y": list(source_position_y) if hasattr(source_position_y, '__iter__') else float(source_position_y) if source_position_y is not None else None,
                        "source_position_z": list(source_position_z) if hasattr(source_position_z, '__iter__') else float(source_position_z) if source_position_z is not None else None,
                        "mute": bool(speaker_array.mute),
                        "hide": bool(speaker_array.hide)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
            # #endregion
            
            # Iteriere über alle Lautsprecher im Array
            # #region agent log
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                    "message": "Single surface calculation - before source loop",
                    "data": {
                        "surface_id": str(surface_id),
                        "array_key": str(array_key),
                        "source_indices_length": int(len(source_indices)),
                        "source_polar_pattern_length": int(len(speaker_array.source_polar_pattern)),
                        "source_indices": [int(x) for x in source_indices.tolist()],
                        "source_polar_pattern": list(speaker_array.source_polar_pattern)
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                        "message": "Single surface calculation - processing source",
                        "data": {
                            "surface_id": str(surface_id),
                            "array_key": str(array_key),
                            "isrc": int(isrc),
                            "speaker_name": str(speaker_name),
                            "source_level": float(source_level[isrc]) if hasattr(source_level, '__iter__') else float(source_level),
                            "source_position_x": float(source_position_x[isrc]) if hasattr(source_position_x, '__iter__') else float(source_position_x),
                            "source_position_y": float(source_position_y[isrc]) if hasattr(source_position_y, '__iter__') else float(source_position_y),
                            "source_position_z": float(source_position_z[isrc]) if hasattr(source_position_z, '__iter__') else float(source_position_z),
                            "source_azimuth_deg": float(np.degrees(source_azimuth[isrc])) if hasattr(source_azimuth, '__iter__') else float(np.degrees(source_azimuth))
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion
                
                # VEKTORISIERTE GEOMETRIE-BERECHNUNG
                x_distances = X_grid - source_position_x[isrc]
                y_distances = Y_grid - source_position_y[isrc]
                horizontal_dists = np.sqrt(x_distances**2 + y_distances**2)
                z_distance = Z_grid - source_position_z[isrc]
                source_dists = np.sqrt(horizontal_dists**2 + z_distance**2)
                
                # VEKTORISIERTE WINKEL-BERECHNUNG
                source_to_point_angles = np.arctan2(y_distances, x_distances)
                azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                azimuths = (360 - azimuths) % 360
                azimuths = (azimuths + 90) % 360
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dists))
                
                # BATCH-INTERPOLATION
                polar_gains, polar_phases = self.get_balloon_data_batch(speaker_name, azimuths, elevations)
                
                if polar_gains is not None and polar_phases is not None:
                    # VEKTORISIERTE WELLENBERECHNUNG
                    magnitude_linear = 10 ** (polar_gains / 20)
                    polar_phase_rad = np.radians(polar_phases)
                    
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

                    # MASKEN-LOGIK: Verwende surface_mask für SPL-Berechnung
                    # Die Maske enthält bereits alle gewünschten Punkte (Surface-Fläche + Rand + Ecken)
                    mask_options = {
                        "min_distance": 0.001,
                        "additional_mask": surface_mask_flat,  # 🎯 SURFACE_MASK: Berechne nur für gewünschte Punkte
                    }
                    
                    # #region agent log - Surface-Maske für Berechnung
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            import json, time as _t
                            surface_mask_active = int(np.count_nonzero(surface_mask_flat))
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "BOUNDARY_POINTS_CALC",
                                "location": "SoundfieldCalculator._calculate_sound_field_for_surface_grid:before_wave",
                                "message": "Surface-Maske für Berechnung - enthält Randpunkte und Ecken",
                                "data": {
                                    "surface_id": str(surface_id),
                                    "surface_mask_active_points": surface_mask_active,
                                    "grid_points_total": int(grid_points.shape[0]),
                                    "includes_edge_and_corner_points": True
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                            "message": "Single surface calculation - before wave computation",
                            "data": {
                                "surface_id": str(surface_id),
                                "grid_points_count": int(grid_points.shape[0]),
                                "surface_mask_active_points": int(np.sum(surface_mask_flat)),
                                "source_level": float(source_level[isrc]),
                                "source_dists_min": float(np.min(source_dists)),
                                "source_dists_max": float(np.max(source_dists)),
                                "source_dists_mean": float(np.mean(source_dists)),
                                "magnitude_linear_min": float(np.min(magnitude_linear)),
                                "magnitude_linear_max": float(np.max(magnitude_linear)),
                                "magnitude_linear_mean": float(np.mean(magnitude_linear))
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                    # #endregion
                    
                    wave_flat = self._compute_wave_for_points(
                        grid_points,
                        source_position,
                        source_props,
                        mask_options,
                    )
                    wave = wave_flat.reshape(source_dists.shape)
                    
                    # #region agent log - Randpunkte nach Berechnung
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            import json, time as _t
                            wave_magnitude = np.abs(wave)
                            # Prüfe, ob Randpunkte Werte haben
                            # Alle Punkte in surface_mask haben Werte (inkl. Randpunkte und Ecken)
                            wave_at_mask = wave_magnitude.ravel()[surface_mask_flat]
                            mask_nonzero = int(np.count_nonzero(wave_at_mask)) if wave_at_mask.size > 0 else 0
                            mask_wave_min = float(np.min(wave_at_mask)) if wave_at_mask.size > 0 else None
                            mask_wave_max = float(np.max(wave_at_mask)) if wave_at_mask.size > 0 else None
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "BOUNDARY_POINTS_CALC",
                                "location": "SoundfieldCalculator._calculate_sound_field_for_surface_grid:after_wave",
                                "message": "Randpunkte nach Wellenberechnung",
                                "data": {
                                    "surface_id": str(surface_id),
                                    "wave_magnitude_min": float(np.min(wave_magnitude)),
                                    "wave_magnitude_max": float(np.max(wave_magnitude)),
                                    "wave_nonzero_count": int(np.count_nonzero(wave)),
                                    "mask_points_count": int(np.count_nonzero(surface_mask_flat)),
                                    "mask_points_with_values": mask_nonzero,
                                    "mask_wave_min": mask_wave_min,
                                    "mask_wave_max": mask_wave_max,
                                    "includes_edge_and_corner_points": True
                                },
                                "timestamp": int(_t.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        wave_magnitude = np.abs(wave)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                            "message": "Single surface calculation - after wave computation",
                            "data": {
                                "surface_id": str(surface_id),
                                "wave_magnitude_min": float(np.min(wave_magnitude)),
                                "wave_magnitude_max": float(np.max(wave_magnitude)),
                                "wave_magnitude_mean": float(np.mean(wave_magnitude)),
                                "wave_nonzero_count": int(np.count_nonzero(wave))
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                    # #endregion
                    
                    # #region agent log - Wave magnitude vor Akkumulation
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        wave_magnitude_before = np.abs(sound_field_p)
                        wave_magnitude_new = np.abs(wave)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "MUTE_DEBUG",
                            "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                            "message": "Wave before accumulation",
                            "data": {
                                "surface_id": str(surface_id),
                                "array_key": str(array_key),
                                "isrc": int(isrc),
                                "field_before_mean": float(np.mean(wave_magnitude_before[surface_mask])) if np.sum(surface_mask) > 0 else 0.0,
                                "wave_new_mean": float(np.mean(wave_magnitude_new[surface_mask])) if np.sum(surface_mask) > 0 else 0.0,
                                "wave_new_max": float(np.max(wave_magnitude_new[surface_mask])) if np.sum(surface_mask) > 0 else 0.0
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                    # #endregion
                    
                    # AKKUMULATION (Interferenz-Überlagerung)
                    sound_field_p += wave
                    if capture_arrays and array_wave is not None:
                        array_wave += wave
                    
                    # #region agent log - Wave magnitude nach Akkumulation
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        wave_magnitude_after = np.abs(sound_field_p)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "MUTE_DEBUG",
                            "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                            "message": "Wave after accumulation",
                            "data": {
                                "surface_id": str(surface_id),
                                "array_key": str(array_key),
                                "isrc": int(isrc),
                                "field_after_mean": float(np.mean(wave_magnitude_after[surface_mask])) if np.sum(surface_mask) > 0 else 0.0,
                                "field_after_max": float(np.max(wave_magnitude_after[surface_mask])) if np.sum(surface_mask) > 0 else 0.0
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                    # #endregion
            
            if capture_arrays and array_wave is not None:
                array_fields[array_key] = array_wave
        
        # #region agent log
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            import json
            final_magnitude = np.abs(sound_field_p)
            # Zähle aktive Arrays
            active_array_count = sum(1 for arr in self.settings.speaker_arrays.values() if not (arr.mute or arr.hide))
            # Berechne SPL in dB für Mask-Bereich
            mask_mean_magnitude = float(np.mean(final_magnitude[surface_mask])) if np.sum(surface_mask) > 0 else 0.0
            spl_db_mean = self.functions.mag2db(mask_mean_magnitude) if mask_mean_magnitude > 0 else -np.inf
            
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "MUTE_DEBUG",
                "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                "message": "Single surface - final calculation result with SPL",
                "data": {
                    "surface_id": str(surface_id),
                    "active_array_count": int(active_array_count),
                    "active_array_keys": [str(k) for k, arr in self.settings.speaker_arrays.items() if not (arr.mute or arr.hide)],
                    "muted_array_keys": [str(k) for k, arr in self.settings.speaker_arrays.items() if arr.mute],
                    "hidden_array_keys": [str(k) for k, arr in self.settings.speaker_arrays.items() if arr.hide],
                    "final_magnitude_min": float(np.min(final_magnitude)),
                    "final_magnitude_max": float(np.max(final_magnitude)),
                    "final_magnitude_mean": float(np.mean(final_magnitude)),
                    "mask_mean_magnitude": mask_mean_magnitude,
                    "spl_db_mean": float(spl_db_mean) if np.isfinite(spl_db_mean) else None,
                    "final_shape": list(sound_field_p.shape),
                    "mask_points": int(np.sum(surface_mask))
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
        # #endregion
        
        # #region agent log
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            import json
            sound_field_magnitude = np.abs(sound_field_p)
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "SoundfieldCalculator.py:_calculate_sound_field_for_surface_grid",
                "message": "Single surface calculation - final result",
                "data": {
                    "surface_id": str(surface_id),
                    "sound_field_magnitude_min": float(np.min(sound_field_magnitude)),
                    "sound_field_magnitude_max": float(np.max(sound_field_magnitude)),
                    "sound_field_magnitude_mean": float(np.mean(sound_field_magnitude)),
                    "sound_field_nonzero_count": int(np.count_nonzero(sound_field_p))
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
        # #endregion
        return sound_field_p, array_fields
    

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

