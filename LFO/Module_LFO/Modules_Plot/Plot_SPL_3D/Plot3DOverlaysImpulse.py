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
        self._overlay_prefix = "imp_"
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

        for x_val, y_val, z_val in current_state:
            try:
                # 🚀 Kugel als Messpunkt-Marker (von allen Seiten gleich gut sichtbar)
                # Radius etwas größer als Kegel-Radius für bessere Sichtbarkeit
                sphere_radius = 0.15
                sphere = pv.Sphere(
                    radius=sphere_radius,
                    center=(x_val, y_val, z_val),
                    theta_resolution=16,  # Horizontale Auflösung
                    phi_resolution=16,    # Vertikale Auflösung
                )
                self._add_overlay_mesh(
                    sphere,
                    color="red",
                    opacity=1.0,
                    line_width=self._get_scaled_line_width(1.0),
                    category="impulse",
                    render_lines_as_tubes=False,  # Kugel braucht keine Tubes
                )
            except Exception:
                continue

        self._last_impulse_state = current_state

    def _compute_impulse_state(self, settings) -> tuple:
        """Erzeugt eine robuste Signatur der Impulse Points."""
        impulse_points = getattr(settings, "impulse_points", []) or []
        # #region agent log
        import json
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:81","message":"_compute_impulse_state start","data":{"impulse_points_count":len(impulse_points),"settings_id":id(settings),"has_attr":hasattr(settings,"impulse_points"),"raw_points":str(impulse_points[:2]) if impulse_points else "empty"},"timestamp":__import__('time').time()*1000}) + '\n')
        except: pass
        # #endregion
        impulse_signature: List[Tuple[float, float, float]] = []
        for point in impulse_points:
            try:
                # 🚀 FIX: Robustere Extraktion der Koordinaten
                data = point.get("data")
                if data is None:
                    continue
                # Unterstütze sowohl Liste als auch Dict-Format
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    x_val, y_val = data[0], data[1]
                    z_val = data[2] if len(data) > 2 else 0.0
                elif isinstance(data, dict):
                    x_val = data.get("x", data.get(0, 0.0))
                    y_val = data.get("y", data.get(1, 0.0))
                    z_val = data.get("z", data.get(2, 0.0))
                else:
                    continue
                impulse_signature.append((float(x_val), float(y_val), float(z_val)))
            except (ValueError, TypeError, KeyError, IndexError) as e:
                # Debug-Output für Fehlerfälle
                print(f"[SPL3DOverlayImpulse] Fehler beim Verarbeiten eines Impulse Points: {e}, point={point}")
                continue
        return tuple(sorted(impulse_signature))


__all__ = ["SPL3DOverlayImpulse"]

