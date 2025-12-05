"""Impulse Points Overlay-Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysBase import SPL3DOverlayBase


class SPL3DOverlayImpulse(SPL3DOverlayBase):
    """
    Overlay-Modul für Impulse Points.

    Zeichnet Messpunkte als rote Kegel. Nutzt die gemeinsame Infrastruktur aus
    `SPL3DOverlayBase` für Actor- und Kategorienverwaltung.
    """

    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert das Impulse-Overlay."""
        super().__init__(plotter, pv_module)
        self._last_impulse_state: Optional[tuple] = None

    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um State zurückzusetzen."""
        super().clear_category(category)
        if category == "impulse":
            self._last_impulse_state = None

    def draw_impulse_points(self, settings) -> None:
        """Zeichnet Impulse Points als rote Kegel."""
        current_state = self._compute_impulse_state(settings)
        existing_names = self._category_actors.get("impulse", [])

        # Skip, wenn sich nichts geändert hat und bereits Actors existieren
        if self._last_impulse_state == current_state and existing_names:
            return

        # Entferne alte Impulse
        self.clear_category("impulse")

        if not current_state:
            self._last_impulse_state = current_state
            return

        try:
            pv = self.pv
        except Exception:
            pv = None

        if pv is None:
            self._last_impulse_state = current_state
            return

        for x_val, y_val in current_state:
            try:
                # Erstelle einen kleinen Kegel (zeigt nach oben)
                cone = pv.Cone(
                    center=(x_val, y_val, 0.0),
                    direction=(0.0, 0.0, 1.0),
                    height=0.3,
                    radius=0.1,
                    resolution=20,
                )
                self._add_overlay_mesh(
                    cone,
                    color="red",
                    opacity=1.0,
                    line_width=1.0,
                    category="impulse",
                    render_lines_as_tubes=True,
                )
            except Exception:
                continue

        self._last_impulse_state = current_state

    def _compute_impulse_state(self, settings) -> tuple:
        """Erzeugt eine robuste Signatur der Impulse Points."""
        impulse_points = getattr(settings, "impulse_points", []) or []
        impulse_signature: List[Tuple[float, float]] = []
        for point in impulse_points:
            try:
                data = point["data"]
                x_val, y_val = data
                impulse_signature.append((float(x_val), float(y_val)))
            except Exception:
                continue
        return tuple(sorted(impulse_signature))


__all__ = ["SPL3DOverlayImpulse"]

