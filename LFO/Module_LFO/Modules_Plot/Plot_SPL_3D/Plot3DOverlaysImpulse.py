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
        # #region agent log
        import json
        import time
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:30","message":"draw_impulse_points called","data":{"has_settings":settings is not None},"timestamp":time.time()*1000}) + '\n')
        except: pass
        # #endregion
        current_state = self._compute_impulse_state(settings)
        existing_names = self._category_actors.get("impulse", [])

        # Skip, wenn sich nichts geändert hat und bereits Actors existieren
        # #region agent log
        import json
        import time
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:33","message":"current_state computed","data":{"current_state_len":len(current_state) if current_state else 0,"current_state":str(current_state[:1]) if current_state else None,"last_state":str(self._last_impulse_state[:1]) if self._last_impulse_state else None},"timestamp":time.time()*1000}) + '\n')
        except: pass
        # #endregion
        if self._last_impulse_state == current_state and existing_names:
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:36","message":"signature comparison","data":{"states_equal":True,"has_existing":bool(existing_names),"will_skip":True},"timestamp":time.time()*1000}) + '\n')
            except: pass
            # #endregion
            return

        # Entferne alte Impulse
        self.clear_category("impulse")

        if not current_state:
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:42","message":"current_state empty, returning","data":{},"timestamp":time.time()*1000}) + '\n')
            except: pass
            # #endregion
            self._last_impulse_state = current_state
            return

        try:
            pv = self.pv
        except Exception as e:
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:48","message":"pv access exception","data":{"error":str(e)},"timestamp":time.time()*1000}) + '\n')
            except: pass
            # #endregion
            pv = None

        if pv is None:
            # #region agent log
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:51","message":"pv is None, returning","data":{},"timestamp":time.time()*1000}) + '\n')
            except: pass
            # #endregion
            self._last_impulse_state = current_state
            return

        # #region agent log
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:55","message":"starting mesh creation loop","data":{"points_count":len(current_state),"pv_available":pv is not None},"timestamp":time.time()*1000}) + '\n')
        except: pass
        # #endregion
        for x_val, y_val, z_val in current_state:
            try:
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:56","message":"creating marker for point","data":{"x":x_val,"y":y_val,"z":z_val},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                # 🚀 Kugel + vertikale Scheibe auf Y-Seite (von z=0 bis zur Kugel)
                sphere_radius = 0.15
                disk_width = 0.05  # Breite der vertikalen Scheibe
                disk_offset_y = -sphere_radius - disk_width / 2.0  # Links von der Kugel in Y-Richtung
                
                # Kugel am Messpunkt
                sphere = pv.Sphere(
                    radius=sphere_radius,
                    center=(x_val, y_val, z_val),
                    theta_resolution=16,
                    phi_resolution=16,
                )
                
                # Vertikale Scheibe auf Y-Seite (von z=0 bis zur Kugel-Höhe)
                meshes_to_combine = [sphere]
                if z_val > 0.01:  # Nur wenn deutlich über Boden
                    disk_height = z_val
                    disk_center_y = y_val + disk_offset_y
                    disk_x_min = x_val - sphere_radius
                    disk_x_max = x_val + sphere_radius
                    
                    # Erstelle sehr dünne Box als vertikale Scheibe (dünn in Y, breit in X, hoch in Z)
                    # Von z=0 bis z=disk_height, links von der Kugel in Y-Richtung
                    disk = pv.Box(
                        bounds=(
                            disk_x_min,                        # x_min (breit wie Kugel)
                            disk_x_max,                        # x_max
                            disk_center_y - disk_width / 2.0,  # y_min
                            disk_center_y + disk_width / 2.0,  # y_max
                            0.0,                                # z_min (Boden/Nullpunkt)
                            disk_height,                        # z_max (bis zur Kugel-Höhe)
                        )
                    )
                    meshes_to_combine.append(disk)
                    # #region agent log
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:73","message":"disk created","data":{"z_val":z_val,"disk_height":disk_height,"disk_center_y":disk_center_y},"timestamp":time.time()*1000}) + '\n')
                    except: pass
                    # #endregion
                
                # Kombiniere Meshes zu einem einzigen Mesh (einfacher zu handhaben)
                if len(meshes_to_combine) > 1:
                    from pyvista import MultiBlock
                    multi_block = MultiBlock(meshes_to_combine)
                    combined_mesh = multi_block.combine()
                    # #region agent log
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:97","message":"meshes combined","data":{"mesh_count":len(meshes_to_combine)},"timestamp":time.time()*1000}) + '\n')
                    except: pass
                    # #endregion
                else:
                    combined_mesh = meshes_to_combine[0]
                
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:101","message":"calling _add_overlay_mesh","data":{},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                self._add_overlay_mesh(
                    combined_mesh,
                    color="red",
                    opacity=1.0,
                    line_width=self._get_scaled_line_width(1.0),
                    category="impulse",
                    render_lines_as_tubes=False,
                )
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:109","message":"_add_overlay_mesh completed","data":{},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
            except Exception as e:
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:110","message":"exception in mesh creation","data":{"error":str(e)},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
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

