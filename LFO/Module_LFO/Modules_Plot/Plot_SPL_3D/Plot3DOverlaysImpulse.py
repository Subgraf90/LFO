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
                # 🚀 Kugel + vertikale Kreisfläche auf Y-Minusseite (selber Z-Nullpunkt)
                sphere_radius = 0.15
                disk_radius = sphere_radius  # Durchmesser Kreis = Kugel-Durchmesser
                
                # Kugel am Messpunkt - Unterseite bei z_val (nicht zentriert)
                sphere_center_z = z_val + sphere_radius  # Kugel-Mitte so, dass Unterseite bei z_val ist
                sphere = pv.Sphere(
                    radius=sphere_radius,
                    center=(x_val, y_val, sphere_center_z),
                    theta_resolution=16,
                    phi_resolution=16,
                )
                
                # Vertikale Kreisfläche auf Y-Minusseite zur Kugel
                meshes_to_combine = [sphere]
                
                # Kreisfläche positionieren (auf Y-Minusseite)
                # Y-Position: Punktposition - Radius Kugel / 2
                disk_center_y = y_val - sphere_radius / 2.0
                disk_center_x = x_val  # X-Position gleich wie Kugel
                disk_center_z = z_val  # Z-Position: z=0 ist unten wie bei Kugel
                disk_thickness = 0.01  # Dicke für bessere Sichtbarkeit
                
                # Erstelle vertikale Kreisfläche als sehr dünnen Zylinder
                # Die Kreisfläche soll in der XZ-Ebene liegen (vertikal, senkrecht zur Y-Achse)
                # Ein Zylinder mit direction=(0,1,0) hat seine Achse in Y-Richtung
                # und die Kreisfläche liegt bereits in der XZ-Ebene - keine Rotation nötig!
                disk = pv.Cylinder(
                    center=(disk_center_x, disk_center_y, disk_center_z),
                    direction=(0, 1, 0),  # Achse in Y-Richtung = Kreisfläche in XZ-Ebene
                    radius=disk_radius,
                    height=disk_thickness,  # Dicke für Sichtbarkeit
                    resolution=16,
                )
                # Keine Rotation nötig - der Zylinder ist bereits richtig orientiert
                meshes_to_combine.append(disk)
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:142","message":"disk rotated and added","data":{"disk_thickness":disk_thickness,"disk_radius":disk_radius},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:73","message":"disk created","data":{"z_val":z_val,"sphere_center_z":sphere_center_z,"disk_radius":disk_radius,"disk_center_x":disk_center_x,"disk_center_y":disk_center_y,"disk_center_z":disk_center_z},"timestamp":time.time()*1000}) + '\n')
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
                    color="black",
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

