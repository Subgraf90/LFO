"""Zentrale Verwaltung der Colorbar für den 3D SPL Plot."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap, ListedColormap

# Phase-Colormap Definition
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
    """Verwaltet die Colorbar für den 3D SPL Plot mit konsistenter Beschriftung."""
    
    # Konstante: Immer genau 10 Farben im Color step Modus
    NUM_COLORS_STEP_MODE = 11  # 11 Levels = 10 Farben
    
    def __init__(
        self,
        colorbar_ax,
        settings,
        phase_mode_active: bool = False,
        time_mode_active: bool = False,
    ):
        """Initialisiert den ColorbarManager.
        
        Args:
            colorbar_ax: Matplotlib Axes für die Colorbar
            settings: Settings-Objekt mit colorbar_range
            phase_mode_active: Ob Phase-Mode aktiv ist
            time_mode_active: Ob Time-Mode aktiv ist
        """
        self.colorbar_ax = colorbar_ax
        self.settings = settings
        self.phase_mode_active = phase_mode_active
        self.time_mode_active = time_mode_active
        
        # State-Variablen
        self.cbar = None
        self._colorbar_mappable = None
        self._colorbar_mode: Optional[tuple[str, tuple[float, ...] | None]] = None
        self._colorbar_override: dict | None = None
        
        # Doppelklick-Erkennung
        self._last_colorbar_click_time: Optional[float] = None
        self._last_colorbar_click_pos: Optional[tuple[int, int]] = None
        self._colorbar_double_click_cid = None
        self._colorbar_double_click_connected = False
    
    def get_colorbar_params(self, phase_mode: bool) -> dict[str, float]:
        """Gibt die Colorbar-Parameter für den aktuellen Modus zurück.
        
        Args:
            phase_mode: Ob Phase-Mode aktiv ist
            
        Returns:
            Dict mit min, max, step, tick_step, label
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
        
        if self.time_mode_active:
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
    
    def get_label(self) -> str:
        """Gibt das konsistente Label für die Colorbar zurück.
        
        Returns:
            Label-String für die Colorbar
        """
        if self._colorbar_override and 'label' in self._colorbar_override:
            return str(self._colorbar_override['label'])
        elif self.time_mode_active:
            return "Pressure (Pa)"
        elif self.phase_mode_active:
            return "Phase difference (°)"
        else:
            return "SPL (dB)"
    
    def render_colorbar(
        self,
        colorization_mode: str,
        force: bool = False,
        tick_step: float | None = None,
        phase_mode_active: bool | None = None,
        time_mode_active: bool | None = None,
    ):
        """Rendert die Colorbar.
        
        Args:
            colorization_mode: 'Color step' oder 'Gradient'
            force: Erzwingt kompletten Neuaufbau
            tick_step: Optionaler Tick-Step (überschreibt Settings)
            phase_mode_active: Optional Phase-Mode Status (überschreibt internen Status)
            time_mode_active: Optional Time-Mode Status (überschreibt internen Status)
        """
        # Aktualisiere interne Status-Variablen falls übergeben
        if phase_mode_active is not None:
            self.phase_mode_active = phase_mode_active
        if time_mode_active is not None:
            self.time_mode_active = time_mode_active
        
        # Hole Parameter
        params = self.get_colorbar_params(self.phase_mode_active)
        cbar_min = float(params['min'])
        cbar_max = float(params['max'])
        cbar_step = float(params['step'])
        tick_step_val = tick_step if tick_step is not None else float(params['tick_step'])
        
        if tick_step_val <= 0:
            tick_step_val = max((cbar_max - cbar_min) / 5.0, 1.0)
        
        # Bereite Axes vor
        if self.colorbar_ax is not None and hasattr(self.colorbar_ax, 'cla'):
            if force or self.cbar is None or self._colorbar_mappable is None:
                self.colorbar_ax.cla()
        
        # Wähle Colormap
        base_cmap = PHASE_CMAP if self.phase_mode_active else cm.get_cmap('jet')
        
        # Erstelle Norm und Colormap basierend auf Modus
        is_step_mode = colorization_mode == 'Color step' and cbar_step > 0
        if is_step_mode:
            # 🎯 IMMER GENAU 10 FARBEN (11 LEVELS) VERWENDEN
            # 🎯 WICHTIG: Levels müssen auf Vielfachen von cbar_step ausgerichtet sein!
            num_segments = self.NUM_COLORS_STEP_MODE - 1  # Immer 10 Segmente/Farben
            
            # Stelle sicher, dass die Range konsistent ist: max = min + (step * 10)
            # Falls nicht, passe max an (bevorzuge min)
            expected_max = cbar_min + (cbar_step * (self.NUM_COLORS_STEP_MODE - 1))
            if abs(cbar_max - expected_max) > 0.01:  # Toleranz für Floating-Point
                cbar_max = expected_max
            
            # Berechne Levels: min, min+step, min+2*step, ..., max
            # Das sollte genau NUM_COLORS_STEP_MODE Levels ergeben
            levels = np.arange(cbar_min, cbar_max + cbar_step * 0.5, cbar_step)  # +0.5*step für Inklusion von max
            
            # Stelle sicher, dass max enthalten ist
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            else:
                # Stelle sicher, dass min und max exakt enthalten sind
                if abs(levels[0] - cbar_min) > 0.01:
                    levels[0] = cbar_min
                if abs(levels[-1] - cbar_max) > 0.01:
                    levels[-1] = cbar_max
                # Entferne Duplikate (falls vorhanden)
                levels = np.unique(levels)
            
            # Stelle sicher, dass wir genau NUM_COLORS_STEP_MODE Levels haben
            if levels.size != self.NUM_COLORS_STEP_MODE:
                # Fallback: berechne Levels manuell
                levels = np.array([cbar_min + i * cbar_step for i in range(self.NUM_COLORS_STEP_MODE)])
                levels[-1] = cbar_max  # Stelle sicher, dass max exakt ist
            
            # Finale Sicherheitsprüfung
            if levels.size == 0:
                levels = np.array([cbar_min, cbar_max])
            if levels.size < 2:
                levels = np.array([cbar_min, cbar_max])
            
            # Berechne Ticks: Im Color step Modus verwenden wir tick_step für bessere Lesbarkeit
            # Aber stellen sicher, dass min und max enthalten sind
            cbar_ticks = np.arange(cbar_min, cbar_max + tick_step_val * 0.5, tick_step_val)
            # Stelle sicher, dass min und max in Ticks enthalten sind
            if cbar_ticks.size == 0:
                cbar_ticks = np.array([cbar_min, cbar_max])
            else:
                # Stelle sicher, dass min und max exakt enthalten sind
                if abs(cbar_ticks[0] - cbar_min) > 0.01:
                    cbar_ticks = np.insert(cbar_ticks, 0, cbar_min)
                if abs(cbar_ticks[-1] - cbar_max) > 0.01:
                    cbar_ticks = np.append(cbar_ticks, cbar_max)
                # Entferne Duplikate
                cbar_ticks = np.unique(cbar_ticks)
            
            # Resample Colormap
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
                'phase_step' if self.phase_mode_active else 'step',
                tuple(float(level) for level in levels),
            )
        else:
            # Gradient-Modus: Normale kontinuierliche Colormap
            cmap = base_cmap
            norm = Normalize(vmin=cbar_min, vmax=cbar_max)
            boundaries = None
            spacing = 'uniform'
            mode_signature = ('phase_grad' if self.phase_mode_active else 'gradient', None)
            
            # Berechne Ticks für Gradient-Modus
            cbar_ticks = np.arange(cbar_min, cbar_max + tick_step_val, tick_step_val)
        
        # Prüfe ob neue Colorbar benötigt wird
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
        
        # Erstelle oder aktualisiere Colorbar
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
        
        # Konfiguriere Axes
        if self.colorbar_ax is not None:
            self.colorbar_ax.tick_params(labelsize=7)
            self.colorbar_ax.set_position([0.1, 0.05, 0.1, 0.9])
        
        # Setze Label (konsistent)
        if self.cbar is not None:
            label = self.get_label()
            self.cbar.set_label(label, fontsize=8)
            
            # Setze Y-Limits (Phase-Mode invertiert)
            if self.phase_mode_active:
                self.cbar.ax.set_ylim(cbar_max, cbar_min)
            else:
                self.cbar.ax.set_ylim(cbar_min, cbar_max)
            
            # Verbinde Doppelklick-Handler
            self._connect_double_click_handler(requires_new_colorbar)
        
        self._colorbar_mode = mode_signature
    
    def _connect_double_click_handler(self, force: bool = False):
        """Verbindet den Doppelklick-Handler mit der Colorbar."""
        if self.colorbar_ax is None:
            return
        
        if force or not self._colorbar_double_click_connected:
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
    
    def _on_colorbar_double_click(self, event):
        """Handler für Doppelklick auf Colorbar.
        
        Matplotlib erkennt Doppelklicks über die Zeit zwischen zwei button_press_event Events.
        """
        # Prüfe ob es ein Linksklick auf Colorbar-Axes war
        if event.button != 1 or event.inaxes != self.colorbar_ax:
            return
        
        # Prüfe ob es ein Doppelklick ist (basierend auf letztem Klick)
        current_time = time.time()
        last_time = self._last_colorbar_click_time
        last_pos = self._last_colorbar_click_pos
        
        # Nur auswerten, wenn Zeit und Position bereits gültig gesetzt wurden
        if last_time is not None and last_pos is not None:
            time_diff = current_time - last_time
            pos_diff = (
                abs(event.x - last_pos[0]) < 10 and
                abs(event.y - last_pos[1]) < 10
            )
            
            # Doppelklick wenn innerhalb von 500ms und ähnlicher Position (10 Pixel Toleranz)
            if time_diff < 0.5 and pos_diff:
                # Signal für Doppelklick (wird von außen behandelt)
                if hasattr(self, 'on_double_click'):
                    self.on_double_click()
                # Reset für nächsten Klick
                self._last_colorbar_click_time = None
                self._last_colorbar_click_pos = None
                return
        
        # Speichere Zeit und Position für nächsten Klick
        self._last_colorbar_click_time = current_time
        self._last_colorbar_click_pos = (event.x, event.y)
    
    def set_override(self, params: dict | None):
        """Setzt temporäre Override-Parameter für die Colorbar.
        
        Args:
            params: Dict mit min, max, step, tick_step, label oder None zum Zurücksetzen
        """
        self._colorbar_override = params
    
    def update_modes(self, phase_mode_active: bool, time_mode_active: bool):
        """Aktualisiert die internen Mode-Status.
        
        Args:
            phase_mode_active: Ob Phase-Mode aktiv ist
            time_mode_active: Ob Time-Mode aktiv ist
        """
        self.phase_mode_active = phase_mode_active
        self.time_mode_active = time_mode_active

