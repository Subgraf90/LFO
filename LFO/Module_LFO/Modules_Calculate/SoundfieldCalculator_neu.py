from __future__ import annotations

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
# ⚠️ NEUER GRID-GENERATOR: Verwende FlexibleGridGenerator statt SurfaceGridCalculator
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator, SurfaceGrid
# Alte Import bleibt auskommentiert für Referenz:
# from Module_LFO.Modules_Calculate.SurfaceGridCalculator import SurfaceGridCalculator
from Module_LFO.Modules_Init.Logging import measure_time, perf_section
# ⚠️ VERALTET: Diese Imports werden nur noch in auskommentierten Methoden verwendet
# from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
#     derive_surface_plane,
#     _points_in_polygon_batch_uv,
# )
from typing import List, Dict, Tuple, Optional, Any
import math

DEBUG_SOUNDFIELD = bool(int(__import__("os").environ.get("LFO_DEBUG_SOUNDFIELD", "1")))


class SoundFieldCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt
        
        # ⚠️ VERALTET: Geometry Cache (wird nicht mehr verwendet, da FlexibleGridGenerator eigenes Caching hat)
        # self._geometry_cache = {}  # {source_key: {distances, azimuths, elevations}}
        # self._grid_cache = None    # Gespeichertes Grid
        # self._grid_hash = None     # Hash der Grid-Parameter
        
        # 🎯 NEUER GRID-GENERATOR: FlexibleGridGenerator (Hybrid-Ansatz)
        self._grid_generator = FlexibleGridGenerator(settings)
        # Alter Code auskommentiert:
        # self._grid_calculator = SurfaceGridCalculator(settings)
   
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
        
        # 🎯 NEU: Grid pro Gruppe (oder ungruppierte Surface) – nutzt FlexibleGridGenerator.generate_per_group
        print(f"[DEBUG Grid-Erstellung] ════════════════════════════════════════════════════════")
        print(f"[DEBUG Grid-Erstellung] Starte Grid-Generierung PRO GRUPPE mit FlexibleGridGenerator")
        print(f"[DEBUG Grid-Erstellung] Anzahl enabled Surfaces: {len(enabled_surfaces)}")
        print(f"[DEBUG Grid-Erstellung] Resolution: {self.settings.resolution} m")
        print(f"[DEBUG Grid-Erstellung] Mindestanzahl Punkte: 3×3 = 9 Punkte pro Surface")
        
        surface_grids_grouped: Dict[str, SurfaceGrid] = self._grid_generator.generate_per_group(
            enabled_surfaces,
            resolution=self.settings.resolution,
            min_points_per_dimension=3  # Mindestens 3×3 = 9 Punkte
        )
        
        # Debug: Zeige Grid-Informationen pro Gruppe/Surface
        total_points_all = 0
        for gid, grid in surface_grids_grouped.items():
            total_points = int(grid.X_grid.size)
            active_points = int(np.count_nonzero(grid.surface_mask))
            print(f"[DEBUG Grid-Erstellung] Gruppe '{gid}': "
                  f"Grid-Shape={grid.X_grid.shape}, Active={active_points}/{total_points}, "
                  f"Resolution={grid.resolution:.3f} m")
            total_points_all += active_points
        
        print(f"[DEBUG Grid-Erstellung] ✅ Gesamt: {len(surface_grids_grouped)} Gruppen, "
              f"{total_points_all} aktive Punkte insgesamt")
        print(f"[DEBUG Grid-Erstellung] ════════════════════════════════════════════════════════")
        
        if not surface_grids_grouped:
            return [], np.array([]), np.array([]), ({} if capture_arrays else None)

        # Flache Sicht: key = group_id (ein Grid pro Gruppe)
        surface_grids: Dict[str, SurfaceGrid] = dict(surface_grids_grouped)
        
        # ============================================================
        # SCHRITT 2: Berechne Physikalische Konstanten (EINMALIG)
        # ============================================================
        # 🚀 PROZESSOPTIMIERUNG: Konstanten nur einmal berechnen für alle Surfaces
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
        
        # Wenn keine aktiven Quellen, gib leere Ergebnisse zurück
        if not has_active_sources:
            # Speichere trotzdem Grid-Daten
            if isinstance(self.calculation_spl, dict):
                surface_grids_data = {}
                for surface_id, grid in surface_grids.items():
                    surface_grids_data[surface_id] = {
                        'sound_field_x': grid.sound_field_x.tolist(),
                        'sound_field_y': grid.sound_field_y.tolist(),
                        'X_grid': grid.X_grid.tolist(),
                        'Y_grid': grid.Y_grid.tolist(),
                        'Z_grid': grid.Z_grid.tolist(),
                        'surface_mask': grid.surface_mask.astype(bool).tolist(),
                        'resolution': grid.resolution,
                    }
                self.calculation_spl['surface_grids'] = surface_grids_data
            
            # Kombiniere Koordinaten für Rückgabewerte (leer)
            all_x = []
            all_y = []
            for grid in surface_grids.values():
                all_x.extend(grid.sound_field_x.tolist())
                all_y.extend(grid.sound_field_y.tolist())
            
            if all_x and all_y:
                unique_x = np.unique(np.array(all_x))
                unique_y = np.unique(np.array(all_y))
                unique_x.sort()
                unique_y.sort()
                return [], unique_x, unique_y, ({} if capture_arrays else None)
            else:
                return [], np.array([]), np.array([]), ({} if capture_arrays else None)
        
        # ============================================================
        # SCHRITT 3: Berechne pro Surface-Grid (PROZESSOPTIMIERT)
        # ============================================================
        print(f"[DEBUG Berechnung] ════════════════════════════════════════════════════════")
        print(f"[DEBUG Berechnung] Starte Berechnung PRO SURFACE (prozessoptimiert)")
        print(f"[DEBUG Berechnung] Anzahl Surface-Grids: {len(surface_grids)}")
        
        # Speichere Ergebnisse pro Surface
        surface_results: Dict[str, Dict[str, Any]] = {}
        combined_array_fields = {} if capture_arrays else None
        
        # Berechne für jedes Surface-Grid einzeln
        for surface_id, grid in surface_grids.items():
            total_points = grid.X_grid.size
            active_points = np.count_nonzero(grid.surface_mask)
            extended_points = total_points - active_points
            
            # 🎯 DEBUG: Prüfe ob vertikale Fläche
            is_vertical = grid.geometry.orientation == "vertical"
            orientation_info = f" (VERTIKAL)" if is_vertical else ""
            
            print(f"[DEBUG Berechnung] Berechne Surface '{surface_id}'{orientation_info}...")
            print(f"  └─ Grid-Shape: {grid.X_grid.shape}")
            print(f"  └─ Total Punkte: {total_points}, Active (in Surface): {active_points}, Extended (außerhalb): {extended_points}")
            
            if is_vertical:
                # 🎯 DEBUG: Zusätzliche Info für vertikale Flächen
                xs = grid.X_grid.flatten()
                ys = grid.Y_grid.flatten()
                zs = grid.Z_grid.flatten()
                x_span = float(np.ptp(xs))
                y_span = float(np.ptp(ys))
                z_span = float(np.ptp(zs))
                print(f"  └─ [VERTIKAL] Koordinaten-Spannen: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
                print(f"  └─ [VERTIKAL] X-Range: [{xs.min():.3f}, {xs.max():.3f}]")
                print(f"  └─ [VERTIKAL] Y-Range: [{ys.min():.3f}, {ys.max():.3f}]")
                print(f"  └─ [VERTIKAL] Z-Range: [{zs.min():.3f}, {zs.max():.3f}]")
            
            # Berechne Schallfeld für dieses Surface-Grid
            try:
                sound_field_p_surface, array_fields_surface = self._calculate_sound_field_for_surface_grid(
                    grid,
                    phys_constants,
                    capture_arrays=capture_arrays
                )
                
                # 🎯 DEBUG: Prüfe ob alle Punkte berechnet wurden
                spl_non_zero = np.count_nonzero(np.abs(sound_field_p_surface))
                spl_in_mask = np.count_nonzero(np.abs(sound_field_p_surface[grid.surface_mask]))
                print(f"  └─ Berechnete SPL-Werte: {spl_non_zero}/{sound_field_p_surface.size} nicht-null (gesamt)")
                print(f"  └─ Berechnete SPL-Werte in Maske: {spl_in_mask}/{active_points} nicht-null (in Surface)")
                
                if is_vertical:
                    # 🎯 DEBUG: Zusätzliche Analyse für vertikale Flächen
                    # Konvertiere komplexen Druck (Pa) zu SPL (dB re 20µPa)
                    p_ref = 20e-6  # Referenzdruck: 20 µPa
                    pressure_magnitude = np.abs(sound_field_p_surface)
                    spl_values = 20 * np.log10(np.maximum(pressure_magnitude / p_ref, 1e-12))
                    spl_in_mask_values = spl_values[grid.surface_mask]
                    if len(spl_in_mask_values) > 0:
                        spl_min = float(spl_in_mask_values.min())
                        spl_max = float(spl_in_mask_values.max())
                        spl_mean = float(spl_in_mask_values.mean())
                        print(f"  └─ [VERTIKAL] SPL in Maske: min={spl_min:.2f} dB, max={spl_max:.2f} dB, mean={spl_mean:.2f} dB")
                        # Zusätzlich: Druckwerte in Pascal
                        pressure_in_mask = pressure_magnitude[grid.surface_mask]
                        p_min = float(pressure_in_mask.min())
                        p_max = float(pressure_in_mask.max())
                        p_mean = float(pressure_in_mask.mean())
                        print(f"  └─ [VERTIKAL] Druck in Maske: min={p_min:.6e} Pa, max={p_max:.6e} Pa, mean={p_mean:.6e} Pa")
                    else:
                        print(f"  └─ [VERTIKAL] ⚠️ Keine SPL-Werte in Maske!")
                    
                    # Prüfe auf NaN oder Inf
                    has_nan = np.any(np.isnan(sound_field_p_surface))
                    has_inf = np.any(np.isinf(sound_field_p_surface))
                    if has_nan:
                        nan_count = int(np.sum(np.isnan(sound_field_p_surface)))
                        print(f"  └─ [VERTIKAL] ⚠️ FEHLER: {nan_count} NaN-Werte gefunden!")
                    if has_inf:
                        inf_count = int(np.sum(np.isinf(sound_field_p_surface)))
                        print(f"  └─ [VERTIKAL] ⚠️ FEHLER: {inf_count} Inf-Werte gefunden!")
                    if not has_nan and not has_inf:
                        print(f"  └─ [VERTIKAL] ✅ Keine Fehler (NaN/Inf) gefunden")
                
            except Exception as e:
                import traceback
                print(f"  └─ ⚠️ FEHLER bei Berechnung von Surface '{surface_id}': {e}")
                if is_vertical:
                    print(f"  └─ [VERTIKAL] ⚠️ FEHLER bei vertikaler Fläche!")
                print(f"  └─ Traceback:")
                traceback.print_exc()
                # Erstelle leeres Ergebnis bei Fehler
                sound_field_p_surface = np.zeros((grid.X_grid.shape[0], grid.X_grid.shape[1]), dtype=complex)
                array_fields_surface = {} if capture_arrays else None
            
            # Speichere Ergebnis pro Surface
            surface_results[surface_id] = {
                'sound_field_p': sound_field_p_surface,
                'sound_field_x': grid.sound_field_x,
                'sound_field_y': grid.sound_field_y,
                'X_grid': grid.X_grid,
                'Y_grid': grid.Y_grid,
                'Z_grid': grid.Z_grid,
                'surface_mask': grid.surface_mask,
            }
            
            # 🎯 DEBUG: Zeige Grid-Daten die gespeichert werden
            if DEBUG_SOUNDFIELD:
                print(f"[DEBUG Grid Übergabe] Surface '{surface_id}': Grid-Daten gespeichert")
                print(f"  └─ X_grid Shape: {grid.X_grid.shape}")
                print(f"  └─ Y_grid Shape: {grid.Y_grid.shape}")
                print(f"  └─ Z_grid Shape: {grid.Z_grid.shape}")
                print(f"  └─ sound_field_p Shape: {sound_field_p_surface.shape}")
                print(f"  └─ surface_mask Shape: {grid.surface_mask.shape}")
                print(f"  └─ sound_field_x: {len(grid.sound_field_x)} Werte, Range: [{grid.sound_field_x.min():.2f}, {grid.sound_field_x.max():.2f}]")
                print(f"  └─ sound_field_y: {len(grid.sound_field_y)} Werte, Range: [{grid.sound_field_y.min():.2f}, {grid.sound_field_y.max():.2f}]")
                print(f"  └─ X_grid Range: [{grid.X_grid.min():.2f}, {grid.X_grid.max():.2f}]")
                print(f"  └─ Y_grid Range: [{grid.Y_grid.min():.2f}, {grid.Y_grid.max():.2f}]")
                print(f"  └─ Z_grid Range: [{grid.Z_grid.min():.2f}, {grid.Z_grid.max():.2f}]")
                print(f"  └─ surface_mask: {np.sum(grid.surface_mask)}/{grid.surface_mask.size} aktiv")
                valid_p = sound_field_p_surface[grid.surface_mask]
                if len(valid_p) > 0:
                    p_mag = np.abs(valid_p)
                    p_db = 20 * np.log10(np.maximum(p_mag, 1e-12))
                    print(f"  └─ SPL Range (dB): [{np.nanmin(p_db):.2f}, {np.nanmax(p_db):.2f}]")
            
            # Kombiniere Array-Felder (falls gewünscht)
            if capture_arrays and array_fields_surface:
                for array_key, array_wave in array_fields_surface.items():
                    if array_key not in combined_array_fields:
                        combined_array_fields[array_key] = {}
                    combined_array_fields[array_key][surface_id] = array_wave
        
        print(f"[DEBUG Berechnung] ✅ Berechnung abgeschlossen für {len(surface_results)} Surfaces")
        print(f"[DEBUG Berechnung] ════════════════════════════════════════════════════════")
        
        # ============================================================
        # SCHRITT 4: Kombiniere Ergebnisse für Rückwärtskompatibilität
        # ============================================================
        # Für Rückwärtskompatibilität: Kombiniere alle Grids zu einem gemeinsamen Grid
        # (für Plot-Verwendung, später wird das pro Surface gerendert)
        all_x = []
        all_y = []
        
        for grid in surface_grids.values():
            all_x.extend(grid.sound_field_x.tolist())
            all_y.extend(grid.sound_field_y.tolist())
        
        if all_x and all_y:
            unique_x = np.unique(np.array(all_x))
            unique_y = np.unique(np.array(all_y))
            unique_x.sort()
            unique_y.sort()
            
            # Erstelle kombiniertes Grid für Rückgabewerte
            X_grid_combined, Y_grid_combined = np.meshgrid(unique_x, unique_y, indexing='xy')
            Z_grid_combined = np.zeros_like(X_grid_combined, dtype=float)
            surface_mask_combined = np.zeros_like(X_grid_combined, dtype=bool)
            sound_field_p_combined = np.zeros_like(X_grid_combined, dtype=complex)
            
            # Interpoliere Ergebnisse auf kombiniertes Grid
            from scipy.interpolate import griddata
            
            for surface_id, result in surface_results.items():
                grid = surface_grids[surface_id]
                
                # Interpoliere Z-Grid
                points_orig = np.column_stack([grid.X_grid.ravel(), grid.Y_grid.ravel()])
                z_orig = grid.Z_grid.ravel()
                mask_orig = grid.surface_mask.ravel()
                sound_field_orig = result['sound_field_p'].ravel()
                
                points_new = np.column_stack([X_grid_combined.ravel(), Y_grid_combined.ravel()])
                
                # Interpoliere nur aktive Punkte
                if np.any(mask_orig):
                    z_interp = griddata(
                        points_orig[mask_orig],
                        z_orig[mask_orig],
                        points_new,
                        method='nearest',
                        fill_value=0.0
                    )
                    mask_interp = griddata(
                        points_orig,
                        mask_orig.astype(float),
                        points_new,
                        method='nearest',
                        fill_value=0.0
                    ) > 0.5
                    sound_field_interp = griddata(
                        points_orig[mask_orig],
                        sound_field_orig[mask_orig],
                        points_new,
                        method='nearest',
                        fill_value=0.0 + 0.0j
                    )
                    
                    # Kombiniere (letzte Surface gewinnt bei Überlappung)
                    mask_interp_2d = mask_interp.reshape(X_grid_combined.shape)
                    Z_grid_combined[mask_interp_2d] = z_interp.reshape(X_grid_combined.shape)[mask_interp_2d]
                    surface_mask_combined |= mask_interp_2d
                    sound_field_p_combined[mask_interp_2d] = sound_field_interp.reshape(X_grid_combined.shape)[mask_interp_2d]
        else:
            # Fallback: Leere Arrays
            unique_x = np.array([])
            unique_y = np.array([])
            X_grid_combined = np.array([]).reshape(0, 0)
            Y_grid_combined = np.array([]).reshape(0, 0)
            Z_grid_combined = np.array([]).reshape(0, 0)
            surface_mask_combined = np.array([]).reshape(0, 0)
            sound_field_p_combined = np.array([]).reshape(0, 0)
        
        # ============================================================
        # SCHRITT 5: Speichere Ergebnisse
        # ============================================================
        if isinstance(self.calculation_spl, dict):
            # Speichere kombiniertes Grid für Rückwärtskompatibilität
            self.calculation_spl['sound_field_z'] = Z_grid_combined.tolist()
            self.calculation_spl['surface_mask'] = surface_mask_combined.astype(bool).tolist()
            self.calculation_spl['surface_mask_strict'] = surface_mask_combined.astype(bool).tolist()
            
            # 🎯 NEU: Speichere Grid-Daten und Ergebnisse pro Surface
            surface_grids_data = {}
            surface_results_data = {}
            
            for surface_id, grid in surface_grids.items():
                result = surface_results[surface_id]
                surface_grids_data[surface_id] = {
                    'sound_field_x': grid.sound_field_x.tolist(),
                    'sound_field_y': grid.sound_field_y.tolist(),
                    'X_grid': grid.X_grid.tolist(),
                    'Y_grid': grid.Y_grid.tolist(),
                    'Z_grid': grid.Z_grid.tolist(),
                    'surface_mask': grid.surface_mask.astype(bool).tolist(),
                    'resolution': grid.resolution,
                    'orientation': grid.geometry.orientation,  # 🎯 NEU: Speichere Orientierung für Plot
                    'dominant_axis': getattr(grid.geometry, 'dominant_axis', None),  # 🎯 NEU: Speichere dominant_axis für Plot
                }
                
                # Speichere Berechnungsergebnisse pro Surface
                sound_field_p_complex = result['sound_field_p']
                surface_results_data[surface_id] = {
                    'sound_field_p': np.array(sound_field_p_complex).tolist(),
                    'sound_field_p_magnitude': np.abs(sound_field_p_complex).tolist(),
                }
                
                # 🎯 DEBUG: Zeige gespeicherte Daten für Container
                if DEBUG_SOUNDFIELD:
                    print(f"[DEBUG Container Speicherung] Surface '{surface_id}': Daten in Container gespeichert")
                    print(f"  └─ surface_grids_data: X_grid={len(surface_grids_data[surface_id]['X_grid'])} Listen")
                    print(f"  └─ surface_results_data: sound_field_p={len(surface_results_data[surface_id]['sound_field_p'])} Listen")
                    x_grid_list = surface_grids_data[surface_id]['X_grid']
                    if isinstance(x_grid_list, list) and len(x_grid_list) > 0:
                        if isinstance(x_grid_list[0], list):
                            print(f"  └─ X_grid: {len(x_grid_list)}x{len(x_grid_list[0])} (als Listen)")
                        else:
                            print(f"  └─ X_grid: {len(x_grid_list)} Werte (als Liste)")
            
            self.calculation_spl['surface_grids'] = surface_grids_data
            self.calculation_spl['surface_results'] = surface_results_data
            
            # ⚠️ VERALTET: Leere Listen für Rückwärtskompatibilität (werden nicht mehr verwendet)
            # self.calculation_spl['surface_meshes'] = []
            # self.calculation_spl['surface_samples'] = []
            # self.calculation_spl['surface_fields'] = {}
        
        # Rückgabe: Kombiniertes Grid für Rückwärtskompatibilität
        return sound_field_p_combined, unique_x, unique_y, combined_array_fields
    
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
    
    @measure_time("SoundFieldCalculator._calculate_sound_field_for_surface_grid")
    def _calculate_sound_field_for_surface_grid(
        self,
        surface_grid: SurfaceGrid,
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
        # Extrahiere Grid-Daten
        X_grid = surface_grid.X_grid
        Y_grid = surface_grid.Y_grid
        Z_grid = surface_grid.Z_grid
        surface_mask = surface_grid.surface_mask
        
        # 🎯 DEBUG: Prüfe ob vertikale Fläche
        is_vertical = surface_grid.geometry.orientation == "vertical"
        surface_id = surface_grid.geometry.surface_id
        
        # Initialisiere Schallfeld
        ny, nx = X_grid.shape
        sound_field_p = np.zeros((ny, nx), dtype=complex)
        array_fields = {} if capture_arrays else None
        
        if is_vertical:
            print(f"  └─ [VERTIKAL Berechnung] Starte Berechnung für '{surface_id}'")
            print(f"  └─ [VERTIKAL Berechnung] Grid-Punkte: {ny}×{nx} = {ny*nx} Punkte")
            print(f"  └─ [VERTIKAL Berechnung] Aktive Punkte (in Maske): {np.count_nonzero(surface_mask)}")
        
        # Extrahiere physikalische Konstanten (bereits berechnet)
        speed_of_sound = phys_constants['speed_of_sound']
        wave_number = phys_constants['wave_number']
        calculate_frequency = phys_constants['calculate_frequency']
        a_source_pa = phys_constants['a_source_pa']
        
        # Bereite Grid-Punkte vor
        grid_points = np.stack((X_grid, Y_grid, Z_grid), axis=-1).reshape(-1, 3)
        surface_mask_flat = surface_mask.reshape(-1)
        
        # 🎯 ERWEITERTE MASKE: Erstelle Maske, die auch Punkte außerhalb der Surface enthält
        # Dies ermöglicht SPL-Berechnung für erweiterte Punkte außerhalb der Surface
        extended_mask_flat = self._create_extended_mask(surface_mask)
        
        # Iteriere über alle Lautsprecher-Arrays
        for array_key, speaker_array in self.settings.speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            
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
                source_time = [time + source_delay for time in speaker_array.source_time]
            source_time = [x / 1000 for x in source_time]
            source_gain = speaker_array.gain
            source_level = speaker_array.source_level + source_gain
            source_level = self.functions.db2mag(np.array(source_level))
            
            # Iteriere über alle Lautsprecher im Array
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                
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

                    # MASKEN-LOGIK: Verwende erweiterte Maske für SPL-Berechnung
                    # Erweiterte Maske enthält auch Punkte außerhalb der Surface (für bessere Interpolation)
                    mask_options = {
                        "min_distance": 0.001,
                        "additional_mask": extended_mask_flat,  # 🎯 ERWEITERTE MASKE: Berechne auch für Punkte außerhalb
                    }
                    
                    wave_flat = self._compute_wave_for_points(
                        grid_points,
                        source_position,
                        source_props,
                        mask_options,
                    )
                    wave = wave_flat.reshape(source_dists.shape)
                    
                    # AKKUMULATION (Interferenz-Überlagerung)
                    sound_field_p += wave
                    if capture_arrays and array_wave is not None:
                        array_wave += wave
            
            if capture_arrays and array_wave is not None:
                array_fields[array_key] = array_wave
        
        # 🎯 DEBUG: Zusammenfassung für vertikale Flächen
        if is_vertical:
            calculated_points = np.count_nonzero(np.abs(sound_field_p))
            calculated_in_mask = np.count_nonzero(np.abs(sound_field_p[surface_mask]))
            print(f"  └─ [VERTIKAL Berechnung] ✅ Berechnung abgeschlossen")
            print(f"  └─ [VERTIKAL Berechnung] Berechnete Punkte: {calculated_points}/{sound_field_p.size} (gesamt)")
            print(f"  └─ [VERTIKAL Berechnung] Berechnete Punkte in Maske: {calculated_in_mask}/{np.count_nonzero(surface_mask)} (in Surface)")

        return sound_field_p, array_fields

    # ⚠️ VERALTET: Diese Methode wird nicht mehr verwendet (alte Berechnung mit SurfaceSamplingPoints)
    # def _calculate_surface_mesh_fields(
    #     self,
    #     sound_field_complex: np.ndarray,
    #     surface_samples: List["SurfaceSamplingPoints"],
    # ) -> Dict[str, np.ndarray]:
    #     surface_fields: Dict[str, np.ndarray] = {}
    #     if sound_field_complex is None or not surface_samples:
    #         return surface_fields
    #
    #     try:
    #         for sample in surface_samples:
    #             if sample.indices.size == 0:
    #                 continue
    #             rows = sample.indices[:, 0]
    #             cols = sample.indices[:, 1]
    #             values = sound_field_complex[rows, cols]
    #             surface_fields[sample.surface_id] = values.copy()
    #     except Exception:
    #         pass
    #     return surface_fields

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
    
    def _create_extended_mask(self, surface_mask: np.ndarray) -> np.ndarray:
        """
        Erstellt eine erweiterte Maske, die auch Punkte außerhalb der Surface enthält.
        
        Die erweiterte Maske enthält:
        - Alle Punkte innerhalb der Surface (originale Maske)
        - Alle Punkte im erweiterten Grid (außerhalb der Surface)
        
        Dies ermöglicht SPL-Berechnung für erweiterte Punkte außerhalb der Surface,
        die für die Triangulation und Interpolation benötigt werden.
        
        Args:
            surface_mask: 2D-Boolean-Array (Shape: [ny, nx]) - originale Surface-Maske
        
        Returns:
            1D-Boolean-Array (Shape: [ny*nx]) - erweiterte Maske (alle Punkte im Grid)
        """
        # 🎯 ERWEITERTE MASKE: Alle Punkte im Grid sind gültig für SPL-Berechnung
        # Die erweiterten Punkte außerhalb der Surface werden ebenfalls berechnet
        # (haben aber möglicherweise niedrigere SPL-Werte)
        extended_mask = np.ones_like(surface_mask, dtype=bool)
        return extended_mask.reshape(-1)
    
    def _get_enabled_surfaces(self) -> List[Tuple[str, Dict]]:
        """
        Gibt alle aktivierten Surfaces zurück, die für die Berechnung verwendet werden sollen.
        
        Filter:
        - Surface: enabled=True und hidden=False
        - Gruppe (falls vorhanden): enabled=True und hidden=False
        
        Nur Surfaces, die beide Bedingungen erfüllen, werden einbezogen.
        """
        if not hasattr(self.settings, 'surface_definitions'):
            return []
        
        # Hole Gruppenstatus (optional)
        groups = getattr(self.settings, "surface_groups", {}) or {}
        groups_dict: Dict[str, Dict[str, Any]] = {}
        if isinstance(groups, dict):
            for gid, gdef in groups.items():
                if hasattr(gdef, "to_dict"):
                    groups_dict[gid] = gdef.to_dict()
                elif isinstance(gdef, dict):
                    groups_dict[gid] = gdef
                else:
                    # generischer Fallback
                    groups_dict[gid] = {
                        "enabled": getattr(gdef, "enabled", True),
                        "hidden": getattr(gdef, "hidden", False),
                    }
        
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
            # Gruppe prüfen
            group_ok = True
            group_id = surface_data.get("group_id") or surface_data.get("group_name")
            if group_id and group_id in groups_dict:
                g = groups_dict[group_id]
                group_ok = bool(g.get("enabled", True)) and not bool(g.get("hidden", False))

            if surface_data.get('enabled', False) and not surface_data.get('hidden', False) and group_ok:
                enabled.append((surface_id, surface_data))
            else:
                if DEBUG_SOUNDFIELD:
                    reason = []
                    if not surface_data.get('enabled', False):
                        reason.append("surface disabled")
                    if surface_data.get('hidden', False):
                        reason.append("surface hidden")
                    if not group_ok:
                        reason.append(f"group '{group_id}' disabled/hidden")
                    print(f"[DEBUG Grid-Erstellung] Skip Surface '{surface_id}': {', '.join(reason) or 'unknown reason'}")
        
        return enabled

