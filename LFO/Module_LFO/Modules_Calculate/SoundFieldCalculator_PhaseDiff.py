import numpy as np

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator


class SoundFieldCalculatorPhaseDiff(SoundFieldCalculator):
    """Berechnet Phasendifferenzen zwischen allen aktiven Arrays pro Grid-Punkt."""

    _AMPLITUDE_THRESHOLD = 1e-9

    def calculate_phase_alignment(self):
        """
        Berechnet Phasendifferenzen zwischen allen aktiven Arrays pro Grid-Punkt.
        
        Verwendet die gleiche 3D-Surface-Integration wie SoundFieldCalculator:
        - Grid wird basierend auf enabled Surfaces erstellt (über _grid_calculator)
        - Z-Koordinaten werden aus Surface-Interpolation übernommen
        - Surface-Masken werden für Berechnung verwendet
        - Alle Surface-Daten (Z_grid, surface_mask, surface_meshes, etc.) werden
          automatisch in self.calculation_spl gespeichert (durch _calculate_sound_field_complex)
        """
        sound_field_complex, sound_field_x, sound_field_y, array_fields = self._calculate_sound_field_complex(
            capture_arrays=True
        )

        if not array_fields:
            phase_diff = []
        else:
            phase_diff = self._compute_phase_differences(array_fields)

        # 🎯 Speichere Phase-Diff-Daten
        self.calculation_spl["sound_field_phase_diff"] = (
            phase_diff.tolist() if isinstance(phase_diff, np.ndarray) else phase_diff
        )
        self.calculation_spl["sound_field_x"] = sound_field_x
        self.calculation_spl["sound_field_y"] = sound_field_y
        
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
        return phase_diff_deg


__all__ = ["SoundFieldCalculatorPhaseDiff"]

