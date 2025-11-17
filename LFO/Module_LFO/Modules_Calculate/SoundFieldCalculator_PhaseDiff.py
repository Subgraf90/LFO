import numpy as np

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator


class SoundFieldCalculatorPhaseDiff(SoundFieldCalculator):
    """Berechnet Phasendifferenzen zwischen allen aktiven Arrays pro Grid-Punkt."""

    _AMPLITUDE_THRESHOLD = 1e-9

    def calculate_phase_alignment(self):
        sound_field_complex, sound_field_x, sound_field_y, array_fields = self._calculate_sound_field_complex(
            capture_arrays=True
        )

        if not array_fields:
            phase_diff = []
        else:
            phase_diff = self._compute_phase_differences(array_fields)

        self.calculation_spl["sound_field_phase_diff"] = (
            phase_diff.tolist() if isinstance(phase_diff, np.ndarray) else phase_diff
        )
        self.calculation_spl["sound_field_x"] = sound_field_x
        self.calculation_spl["sound_field_y"] = sound_field_y
        return phase_diff, sound_field_x, sound_field_y

    def _compute_phase_differences(self, array_fields: dict) -> np.ndarray:
        arrays = [field for field in array_fields.values() if field is not None]
        if len(arrays) < 2:
            return np.full_like(arrays[0], np.nan, dtype=float) if arrays else np.array([])

        stack = np.stack(arrays, axis=0)  # [n_arrays, ny, nx]
        magnitudes = np.abs(stack)
        valid_masks = magnitudes > self._AMPLITUDE_THRESHOLD

        phases_rad = np.angle(stack)  # [-pi, pi]
        max_diff_rad = np.zeros_like(magnitudes[0], dtype=float)
        has_pair = np.zeros_like(magnitudes[0], dtype=bool)

        num_arrays = stack.shape[0]
        for i in range(num_arrays):
            for j in range(i + 1, num_arrays):
                pair_mask = valid_masks[i] & valid_masks[j]
                if not np.any(pair_mask):
                    continue
                diff_rad = np.abs(np.angle(np.exp(1j * (phases_rad[i] - phases_rad[j]))))
                update_mask = pair_mask & (diff_rad > max_diff_rad)
                max_diff_rad = np.where(update_mask, diff_rad, max_diff_rad)
                has_pair = has_pair | pair_mask

        phase_diff_deg = np.degrees(max_diff_rad)
        phase_diff_deg = np.where(has_pair, phase_diff_deg, np.nan)
        return phase_diff_deg


__all__ = ["SoundFieldCalculatorPhaseDiff"]

