import numpy as np
from typing import Dict, Any, Tuple, Optional

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator


class SoundFieldCalculatorPhaseDiff(SoundFieldCalculator):
    """Berechnet Phasendifferenzen zwischen allen aktiven Arrays pro Grid-Punkt."""

    _AMPLITUDE_THRESHOLD = 1e-9
    
    def _calculate_sound_field_for_surface_grid(
        self,
        surface_grid: 'SurfaceGrid',
        phys_constants: Dict[str, Any],
        capture_arrays: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Überschreibt die Basis-Methode, um zusätzlich Phase-Differenzen für additional_vertices
        direkt während der Berechnung zu berechnen (wie SPL).
        
        Dies stellt sicher, dass Phase-Berechnung und SPL-Berechnung identisch funktionieren.
        """
        # Rufe Basis-Methode auf (berechnet alles inkl. additional_vertices_array_fields)
        sound_field_p, array_fields = super()._calculate_sound_field_for_surface_grid(
            surface_grid, phys_constants, capture_arrays
        )
        
        # 🎯 NEU: Berechne Phase-Differenzen für additional_vertices direkt hier
        # (genau wie SPL direkt berechnet wird)
        if capture_arrays and hasattr(surface_grid, 'additional_vertices_array_fields') and surface_grid.additional_vertices_array_fields is not None:
            additional_vertices_array_fields = surface_grid.additional_vertices_array_fields
            
            if additional_vertices_array_fields:
                try:
                    # Konvertiere zu numpy Arrays falls nötig
                    array_fields_dict = {}
                    for array_key, array_field in additional_vertices_array_fields.items():
                        if array_field is not None:
                            array_fields_dict[array_key] = np.asarray(array_field, dtype=complex)
                    
                    if array_fields_dict:
                        # Berechne Phase-Differenzen für additional_vertices (1D-Array)
                        phase_diff = self._compute_phase_differences_1d(array_fields_dict)
                        
                        if phase_diff is not None:
                            # Speichere direkt im surface_grid (wie additional_vertices_spl)
                            surface_grid.additional_vertices_phase = phase_diff
                        else:
                            surface_grid.additional_vertices_phase = None
                    else:
                        surface_grid.additional_vertices_phase = None
                except Exception as e:
                    print(f"[Phase Calc] Fehler bei direkter Berechnung der Phase-Daten für additional_vertices: {e}")
                    surface_grid.additional_vertices_phase = None
            else:
                surface_grid.additional_vertices_phase = None
        else:
            surface_grid.additional_vertices_phase = None
        
        return sound_field_p, array_fields

    def calculate_phase_alignment(self):
        """
        Berechnet Phasendifferenzen zwischen allen aktiven Arrays pro Grid-Punkt.
        
        Verwendet die gleiche 3D-Surface-Integration wie SoundFieldCalculator:
        - Grid wird basierend auf enabled Surfaces erstellt (über _grid_calculator)
        - Z-Koordinaten werden aus Surface-Interpolation übernommen
        - Surface-Masken werden für Berechnung verwendet
        - Alle Surface-Daten (Z_grid, surface_mask, surface_meshes, etc.) werden
          automatisch in self.calculation_spl gespeichert (durch _calculate_sound_field_complex)
        - 🎯 NEU: additional_vertices werden direkt während der Berechnung behandelt (wie SPL)
        """
        sound_field_complex, sound_field_x, sound_field_y, array_fields = self._calculate_sound_field_complex(
            capture_arrays=True
        )
        
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "I",
                    "location": "SoundFieldCalculator_PhaseDiff.py:calculate_phase_alignment:after_calc_complex",
                    "message": "Nach _calculate_sound_field_complex",
                    "data": {
                        "array_fields_is_none": array_fields is None,
                        "array_fields_type": type(array_fields).__name__ if array_fields is not None else None,
                        "array_fields_len": len(array_fields) if isinstance(array_fields, dict) else (0 if array_fields is None else "not_dict"),
                        "array_fields_keys": list(array_fields.keys()) if isinstance(array_fields, dict) else None,
                        "sound_field_x_len": len(sound_field_x) if sound_field_x is not None else 0,
                        "sound_field_y_len": len(sound_field_y) if sound_field_y is not None else 0
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion

        if not array_fields:
            # 🎯 FIX: Erstelle leeres Array mit richtiger Form statt leerer Liste
            if sound_field_x is not None and sound_field_y is not None:
                try:
                    nx = len(sound_field_x) if hasattr(sound_field_x, '__len__') else 0
                    ny = len(sound_field_y) if hasattr(sound_field_y, '__len__') else 0
                    if nx > 0 and ny > 0:
                        phase_diff = np.full((ny, nx), np.nan, dtype=float)
                        print(f"[Phase Calc] Keine Arrays vorhanden, erstelle leeres Array: shape=({ny}, {nx})")
                    else:
                        phase_diff = np.array([], dtype=float)
                        print(f"[Phase Calc] Keine Arrays vorhanden, leeres Array (nx={nx}, ny={ny})")
                except Exception as e:
                    phase_diff = np.array([], dtype=float)
                    print(f"[Phase Calc] Fehler beim Erstellen des leeren Arrays: {e}")
            else:
                phase_diff = np.array([], dtype=float)
                print(f"[Phase Calc] Keine Koordinaten vorhanden")
        else:
            phase_diff = self._compute_phase_differences(array_fields)
            # 🎯 DEBUG: Ausgabe der berechneten Phase-Daten
            valid_phase = phase_diff[np.isfinite(phase_diff)]
            if len(valid_phase) > 0:
                print(f"[Phase Calc] Phase-Daten berechnet: shape={phase_diff.shape}, "
                      f"min={np.nanmin(valid_phase):.2f}°, max={np.nanmax(valid_phase):.2f}°, "
                      f"mean={np.nanmean(valid_phase):.2f}°, valid={len(valid_phase)}/{phase_diff.size}")
            else:
                print(f"[Phase Calc] Phase-Daten berechnet, aber alle NaN: shape={phase_diff.shape}")
        
        # 🎯 NEU: additional_vertices_phase wird bereits direkt während _calculate_sound_field_for_surface_grid
        # berechnet (durch Überschreibung der Methode), daher müssen wir hier nichts mehr tun.
        # Die Phase-Daten sind bereits in surface_results['additional_vertices_phase'] gespeichert.

        # 🎯 Speichere Phase-Diff-Daten
        self.calculation_spl["sound_field_phase_diff"] = (
            phase_diff.tolist() if isinstance(phase_diff, np.ndarray) else phase_diff
        )
        self.calculation_spl["sound_field_x"] = sound_field_x
        self.calculation_spl["sound_field_y"] = sound_field_y
        
        # #region agent log
        try:
            import json
            import time as time_module
            phase_diff_arr = np.asarray(phase_diff) if isinstance(phase_diff, (list, np.ndarray)) else np.array([])
            valid_phase = phase_diff_arr[np.isfinite(phase_diff_arr)] if phase_diff_arr.size > 0 else np.array([])
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "SoundFieldCalculator_PhaseDiff.py:calculate_phase_alignment:save",
                    "message": "Phase-Daten gespeichert",
                    "data": {
                        "phase_diff_type": type(phase_diff).__name__,
                        "phase_diff_shape": list(phase_diff_arr.shape) if phase_diff_arr.size > 0 else [],
                        "phase_diff_size": phase_diff_arr.size if phase_diff_arr.size > 0 else 0,
                        "valid_count": len(valid_phase),
                        "sound_field_x_len": len(sound_field_x) if sound_field_x is not None else 0,
                        "sound_field_y_len": len(sound_field_y) if sound_field_y is not None else 0,
                        "min_deg": float(np.nanmin(valid_phase)) if len(valid_phase) > 0 else None,
                        "max_deg": float(np.nanmax(valid_phase)) if len(valid_phase) > 0 else None
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        # 🎯 WICHTIG: Surface-Daten (Z_grid, surface_mask, surface_meshes, etc.)
        # werden bereits in _calculate_sound_field_complex gespeichert (Zeilen 710-736)
        # und müssen hier NICHT erneut gespeichert werden, da sie bereits vorhanden sind.
        # Die Methode _calculate_sound_field_complex speichert automatisch:
        # - sound_field_z (Z_grid)
        # - surface_mask (erweiterte Maske für Berechnung)
        # - surface_mask_strict (strikte Maske für Plot)
        # - surface_meshes (Mesh-Geometrien)
        # - surface_samples (Sampling-Punkte)
        # - surface_fields (Feldwerte pro Surface)
        
        return phase_diff, sound_field_x, sound_field_y

    def _compute_phase_differences(self, array_fields: dict) -> np.ndarray:
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences:entry",
                    "message": "Berechne Phasendifferenzen",
                    "data": {
                        "num_arrays": len(array_fields),
                        "array_keys": list(array_fields.keys())
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        arrays = [field for field in array_fields.values() if field is not None]
        if not arrays:
            return np.array([])
        if len(arrays) == 1:
            return np.zeros_like(arrays[0], dtype=float)

        stack = np.stack(arrays, axis=0)  # [n_arrays, ny, nx]
        magnitudes = np.abs(stack)
        valid_masks = magnitudes > self._AMPLITUDE_THRESHOLD

        phases_rad = np.angle(stack)  # [-pi, pi]
        weighted_diff_sum = np.zeros_like(magnitudes[0], dtype=float)
        weight_sum = np.zeros_like(magnitudes[0], dtype=float)

        num_arrays = stack.shape[0]
        for i in range(num_arrays):
            for j in range(i + 1, num_arrays):
                pair_mask = valid_masks[i] & valid_masks[j]
                if not np.any(pair_mask):
                    continue
                diff_rad = np.abs(np.angle(np.exp(1j * (phases_rad[i] - phases_rad[j]))))
                pair_weight = magnitudes[i] * magnitudes[j]
                pair_weight = np.where(pair_mask, pair_weight, 0.0)
                if not np.any(pair_weight):
                    continue
                weighted_diff_sum += diff_rad * pair_weight
                weight_sum += pair_weight

        phase_diff_rad = np.divide(
            weighted_diff_sum,
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        phase_diff_deg = np.degrees(phase_diff_rad)
        
        # #region agent log
        try:
            valid_phase = phase_diff_deg[np.isfinite(phase_diff_deg)]
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences:exit",
                    "message": "Phasendifferenzen berechnet",
                    "data": {
                        "shape": list(phase_diff_deg.shape),
                        "valid_count": len(valid_phase),
                        "total_count": phase_diff_deg.size,
                        "min_deg": float(np.nanmin(valid_phase)) if len(valid_phase) > 0 else None,
                        "max_deg": float(np.nanmax(valid_phase)) if len(valid_phase) > 0 else None,
                        "mean_deg": float(np.nanmean(valid_phase)) if len(valid_phase) > 0 else None
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        return phase_diff_deg
    
    def _compute_phase_differences_for_additional_vertices(self) -> dict:
        """
        Berechnet Phasendifferenzen für additional_vertices (Rand- und Eckpunkte).
        
        Verwendet dieselbe Logik wie _compute_phase_differences, aber für 1D-Arrays
        statt 2D-Grids.
        
        Returns:
            dict: {surface_id: phase_diff_array} oder None wenn keine Daten vorhanden
        """
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A",
                    "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences_for_additional_vertices:entry",
                    "message": "Berechne Phase-Daten für additional_vertices",
                    "data": {
                        "has_calculation_spl": isinstance(self.calculation_spl, dict),
                        "has_surface_results": isinstance(self.calculation_spl, dict) and 'surface_results' in self.calculation_spl
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        if not isinstance(self.calculation_spl, dict) or 'surface_results' not in self.calculation_spl:
            return None
        
        surface_results = self.calculation_spl['surface_results']
        additional_vertices_phase_diff = {}
        
        for surface_id, result_data in surface_results.items():
            if not isinstance(result_data, dict):
                continue
            
            additional_vertices_array_fields = result_data.get('additional_vertices_array_fields')
            
            # #region agent log
            try:
                import json
                import time as time_module
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences_for_additional_vertices:surface_check",
                        "message": "Prüfe Surface für additional_vertices",
                        "data": {
                            "surface_id": surface_id,
                            "has_additional_vertices_array_fields": additional_vertices_array_fields is not None,
                            "array_fields_type": type(additional_vertices_array_fields).__name__ if additional_vertices_array_fields is not None else None,
                            "array_fields_keys": list(additional_vertices_array_fields.keys()) if isinstance(additional_vertices_array_fields, dict) else None
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion
            
            if not additional_vertices_array_fields:
                continue
            
            # Konvertiere Listen zurück zu numpy Arrays
            try:
                array_fields_dict = {}
                for array_key, array_field_list in additional_vertices_array_fields.items():
                    if array_field_list is not None:
                        array_fields_dict[array_key] = np.asarray(array_field_list, dtype=complex)
                
                if not array_fields_dict:
                    continue
                
                # Berechne Phase-Differenzen für additional_vertices (1D-Array)
                phase_diff = self._compute_phase_differences_1d(array_fields_dict)
                
                # #region agent log
                try:
                    import json
                    import time as time_module
                    nan_count = np.sum(np.isnan(phase_diff)) if phase_diff is not None else 0
                    valid_count = np.sum(np.isfinite(phase_diff)) if phase_diff is not None else 0
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences_for_additional_vertices:phase_calc_result",
                            "message": "Phase-Berechnung Ergebnis",
                            "data": {
                                "surface_id": surface_id,
                                "phase_diff_is_none": phase_diff is None,
                                "phase_diff_len": len(phase_diff) if phase_diff is not None else 0,
                                "nan_count": int(nan_count),
                                "valid_count": int(valid_count),
                                "min_deg": float(np.nanmin(phase_diff)) if phase_diff is not None and valid_count > 0 else None,
                                "max_deg": float(np.nanmax(phase_diff)) if phase_diff is not None and valid_count > 0 else None
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                
                if phase_diff is not None:
                    additional_vertices_phase_diff[surface_id] = phase_diff.tolist()
            except Exception as e:
                print(f"[Phase Calc] Fehler bei Berechnung der Phase-Daten für additional_vertices (Surface {surface_id}): {e}")
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "SoundFieldCalculator_PhaseDiff.py:_compute_phase_differences_for_additional_vertices:error",
                            "message": "Fehler bei Phase-Berechnung",
                            "data": {
                                "surface_id": surface_id,
                                "error": str(e),
                                "error_type": type(e).__name__
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
                continue
        
        return additional_vertices_phase_diff if additional_vertices_phase_diff else None
    
    def _compute_phase_differences_1d(self, array_fields: dict) -> np.ndarray:
        """
        Berechnet Phasendifferenzen für 1D-Arrays (additional_vertices).
        
        Args:
            array_fields: Dict von {array_key: 1D-Array komplexer Werte}
        
        Returns:
            np.ndarray: 1D-Array mit Phasendifferenzen in Grad, oder None bei Fehler
        """
        arrays = [field for field in array_fields.values() if field is not None]
        if not arrays:
            return None
        if len(arrays) == 1:
            return np.zeros_like(arrays[0], dtype=float)
        
        # Stelle sicher, dass alle Arrays die gleiche Länge haben
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            # Unterschiedliche Längen - verwende minimale Länge
            min_length = min(lengths)
            arrays = [arr[:min_length] for arr in arrays]
        
        stack = np.stack(arrays, axis=0)  # [n_arrays, n_points]
        magnitudes = np.abs(stack)
        valid_masks = magnitudes > self._AMPLITUDE_THRESHOLD
        
        phases_rad = np.angle(stack)  # [-pi, pi]
        weighted_diff_sum = np.zeros_like(magnitudes[0], dtype=float)
        weight_sum = np.zeros_like(magnitudes[0], dtype=float)
        
        num_arrays = stack.shape[0]
        for i in range(num_arrays):
            for j in range(i + 1, num_arrays):
                pair_mask = valid_masks[i] & valid_masks[j]
                if not np.any(pair_mask):
                    continue
                diff_rad = np.abs(np.angle(np.exp(1j * (phases_rad[i] - phases_rad[j]))))
                pair_weight = magnitudes[i] * magnitudes[j]
                pair_weight = np.where(pair_mask, pair_weight, 0.0)
                if not np.any(pair_weight):
                    continue
                weighted_diff_sum += diff_rad * pair_weight
                weight_sum += pair_weight
        
        phase_diff_rad = np.divide(
            weighted_diff_sum,
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        phase_diff_deg = np.degrees(phase_diff_rad)
        
        return phase_diff_deg


__all__ = ["SoundFieldCalculatorPhaseDiff"]

