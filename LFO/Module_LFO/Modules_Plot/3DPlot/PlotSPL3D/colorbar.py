"""Colorbar-Management für den 3D-SPL-Plot."""

from __future__ import annotations

import time
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap, ListedColormap

# Phase-Colormap (wird auch in der Hauptklasse verwendet)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase_wheel",
    [
        (0.0, "#2ecc71"),          # 0°
        (38 / 180.0, "#8bd852"),   # 38°
        (90 / 180.0, "#f4e34d"),   # 90°
        (120 / 180.0, "#f7b731"),  # 120°
        (150 / 180.0, "#eb8334"),  # 150°
        (1.0, "#d64545"),          # 180°
    ],
)


class ColorbarManager:
    """Verwaltet die Colorbar für den 3D-SPL-Plot."""
    
    def __init__(self, colorbar_ax: Any, settings: Any, phase_mode_active_getter, time_mode_active_getter):
        """Initialisiert den Colorbar-Manager.
        
        Args:
            colorbar_ax: Matplotlib Axes für die Colorbar
            settings: Settings-Objekt
            phase_mode_active_getter: Funktion zum Abrufen von _phase_mode_active
            time_mode_active_getter: Funktion zum Abrufen von _time_mode_active
        """
        self.colorbar_ax = colorbar_ax
        self.settings = settings
        self._get_phase_mode_active = phase_mode_active_getter
        self._get_time_mode_active = time_mode_active_getter
        
        self.cbar = None
        self._colorbar_mappable = None
        self._colorbar_mode: Optional[tuple[str, tuple[float, ...] | None]] = None
        self._colorbar_override: dict | None = None
        self._last_colorbar_click_time: Optional[float] = None
        self._last_colorbar_click_pos: Optional[tuple[int, int]] = None
        self._colorbar_double_click_connected = False
        self._colorbar_double_click_cid = None
        self._double_click_handler = None
    
    def set_double_click_handler(self, handler):
        """Setzt den Handler für Doppelklick-Events.
        
        Args:
            handler: Funktion die bei Doppelklick aufgerufen wird
        """
        self._double_click_handler = handler
    
    def get_colorbar_params(self, phase_mode: bool) -> dict[str, float]:
        """Berechnet die Colorbar-Parameter.
        
        Args:
            phase_mode: Ob Phase-Modus aktiv ist
            
        Returns:
            Dictionary mit Colorbar-Parametern
        """
        if self._colorbar_override:
            return self._colorbar_override
        if phase_mode:
            return {
                'min': 0.0,
                'max': 180.0,
                'step': 10.0,
                'tick_step': 30.0,
                'label': "Phase difference (°)",
            }
        if self._get_time_mode_active():
            max_pressure = float(getattr(self.settings, 'fem_time_plot_max_pressure', 50.0) or 50.0)
            return {
                'min': -max_pressure,
                'max': max_pressure,
                'step': max_pressure / 5.0,
                'tick_step': max_pressure / 5.0,
                'label': "Pressure (Pa)",
            }
        rng = self.settings.colorbar_range
        return {
            'min': float(rng['min']),
            'max': float(rng['max']),
            'step': float(rng['step']),
            'tick_step': float(rng['tick_step']),
            'label': "SPL (dB)",
        }
    
    def initialize_empty_colorbar(self):
        """Initialisiert eine leere Colorbar analog zur 2D-Version."""
        colorization_mode = getattr(self.settings, 'colorization_mode', 'Gradient')
        if colorization_mode not in {'Color step', 'Gradient'}:
            colorization_mode = 'Color step'
        try:
            self.render_colorbar(colorization_mode, force=True)
        except Exception:  # noqa: BLE001
            pass
    
    def update_colorbar(self, colorization_mode: str, tick_step: float | None = None):
        """Aktualisiert die Colorbar.
        
        Args:
            colorization_mode: Modus ('Gradient' oder 'Color step')
            tick_step: Optionaler Tick-Schritt
        """
        try:
            self.render_colorbar(colorization_mode, tick_step=tick_step)
        except Exception:  # noqa: BLE001
            pass
    
    def render_colorbar(self, colorization_mode: str, force: bool = False, tick_step: float | None = None):
        """Rendert die Colorbar.
        
        Args:
            colorization_mode: Modus ('Gradient' oder 'Color step')
            force: Ob ein kompletter Neuaufbau erzwungen werden soll
            tick_step: Optionaler Tick-Schritt
        """
        phase_mode = self._get_phase_mode_active()
        time_mode = self._get_time_mode_active()
        
        if self._colorbar_override:
            params = self._colorbar_override
            cbar_min = float(params['min'])
            cbar_max = float(params['max'])
            cbar_step = float(params['step'])
            tick_step_val = tick_step if tick_step is not None else float(params['tick_step'])
        else:
            cbar_min = float(self.settings.colorbar_range['min'])
            cbar_max = float(self.settings.colorbar_range['max'])
            cbar_step = float(self.settings.colorbar_range['step'])
            tick_step_val = float(self.settings.colorbar_range['tick_step'])

        if tick_step_val <= 0:
            tick_step_val = max((cbar_max - cbar_min) / 5.0, 1.0)

        cbar_ticks = np.arange(cbar_min, cbar_max + tick_step_val, tick_step_val)

        if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
            if force or self.cbar is None or self._colorbar_mappable is None:
                self.colorbar_ax.cla()

        boundaries = None
        spacing = 'uniform'
        base_cmap = PHASE_CMAP if phase_mode else cm.get_cmap('jet')
        is_step_mode = colorization_mode == 'Color step' and cbar_step > 0
        if is_step_mode:
            levels = np.arange(cbar_min, cbar_max + cbar_step, cbar_step)
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            if levels[-1] < cbar_max:
                levels = np.append(levels, cbar_max)
            if levels.size < 2:
                levels = np.array([cbar_min, cbar_max])

            num_segments = max(1, len(levels) - 1)
            if hasattr(base_cmap, "resampled"):
                sampled_cmap = base_cmap.resampled(num_segments)
            else:
                sampled_cmap = cm.get_cmap(base_cmap, num_segments)
            sample_points = (np.arange(num_segments, dtype=float) + 0.5) / max(num_segments, 1)
            color_list = sampled_cmap(sample_points)
            cmap = ListedColormap(color_list)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            boundaries = levels
            spacing = 'proportional'
            mode_signature: tuple[str, tuple[float, ...] | None] = (
                'phase_step' if phase_mode else 'step',
                tuple(float(level) for level in levels),
            )
        else:
            cmap = base_cmap
            norm = Normalize(vmin=cbar_min, vmax=cbar_max)
            mode_signature = ('phase_grad' if phase_mode else 'gradient', None)

        requires_new_colorbar = (
            force
            or self.cbar is None
            or self._colorbar_mappable is None
            or self._colorbar_mode is None
            or self._colorbar_mode[0] != mode_signature[0]
            or (
                mode_signature[1] is not None
                and (self._colorbar_mode[1] != mode_signature[1])
            )
        )

        if requires_new_colorbar:
            if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
                self.colorbar_ax.cla()
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            colorbar_kwargs = {
                'cax': self.colorbar_ax,
                'ticks': cbar_ticks,
            }
            if boundaries is not None:
                colorbar_kwargs.update({'boundaries': boundaries, 'spacing': spacing})

            self.cbar = plt.colorbar(sm, **colorbar_kwargs)
            self._colorbar_mappable = sm
            if boundaries is not None and hasattr(self.cbar, 'solids') and self.cbar.solids is not None:
                self.cbar.solids.set_edgecolor('face')
        else:
            assert self._colorbar_mappable is not None
            self._colorbar_mappable.set_cmap(cmap)
            self._colorbar_mappable.set_norm(norm)
            self.cbar.set_ticks(cbar_ticks)
            if boundaries is not None:
                self.cbar.boundaries = boundaries
                self.cbar.spacing = spacing
            self.cbar.update_normal(self._colorbar_mappable)

        if self.colorbar_ax is not None:
            self.colorbar_ax.tick_params(labelsize=7)
            self.colorbar_ax.set_position([0.1, 0.05, 0.1, 0.9])
        if self.cbar is not None:
            label = None
            if self._colorbar_override and 'label' in self._colorbar_override:
                label = str(self._colorbar_override['label'])
            elif time_mode:
                label = "Pressure (Pa)"
            elif phase_mode:
                label = "Phase difference (°)"
            else:
                label = "SPL (dB)"
            self.cbar.set_label(label, fontsize=8)
            if phase_mode:
                self.cbar.ax.set_ylim(cbar_max, cbar_min)
            else:
                self.cbar.ax.set_ylim(cbar_min, cbar_max)
            
            # Doppelklick-Handler zur Colorbar hinzufügen
            if requires_new_colorbar or not self._colorbar_double_click_connected:
                # Entferne alte Handler falls vorhanden
                if self._colorbar_double_click_cid is not None:
                    try:
                        self.colorbar_ax.figure.canvas.mpl_disconnect(self._colorbar_double_click_cid)
                    except Exception:
                        pass
                # Füge Doppelklick-Handler hinzu
                self._colorbar_double_click_cid = self.colorbar_ax.figure.canvas.mpl_connect(
                    'button_press_event',
                    self._on_colorbar_double_click
                )
                self._colorbar_double_click_connected = True

        self._colorbar_mode = mode_signature
    
    def _on_colorbar_double_click(self, event):
        """Handler für Doppelklick auf Colorbar."""
        if event.button != 1 or event.inaxes != self.colorbar_ax:
            return
        
        current_time = time.time()
        last_time = self._last_colorbar_click_time
        last_pos = self._last_colorbar_click_pos

        if last_time is not None and last_pos is not None:
            time_diff = current_time - last_time
            pos_diff = (
                abs(event.x - last_pos[0]) < 10 and
                abs(event.y - last_pos[1]) < 10
            )
            
            if time_diff < 0.5 and pos_diff:
                if self._double_click_handler:
                    self._double_click_handler()
                self._last_colorbar_click_time = None
                self._last_colorbar_click_pos = None
                return
        
        self._last_colorbar_click_time = current_time
        self._last_colorbar_click_pos = (event.x, event.y)
    
    def set_override(self, override: dict | None):
        """Setzt eine Colorbar-Override-Konfiguration.
        
        Args:
            override: Dictionary mit Override-Parametern oder None
        """
        self._colorbar_override = override

