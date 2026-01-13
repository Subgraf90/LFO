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
                # 🚀 Kugel + vertikale Kreisfläche am Scheitelpunkt (selber Z-Nullpunkt)
                # Kugelradius aus Settings lesen
                sphere_radius = getattr(settings, "measurement_size", 4.0) / 2.0  # measurement_size ist Durchmesser, Radius = Durchmesser / 2
                # Scheibe: Durchmesser und Dicke proportional zur Kugel
                disk_radius = sphere_radius  # Durchmesser Kreis = Kugel-Durchmesser (proportional)
                
                # Kugel am Messpunkt - Unterseite bei z_val (nicht zentriert)
                sphere_center_z = z_val + sphere_radius  # Kugel-Mitte so, dass Unterseite bei z_val ist
                sphere_top_z = z_val + 2 * sphere_radius  # Oberseite der Kugel
                sphere = pv.Sphere(
                    radius=sphere_radius,
                    center=(x_val, y_val, sphere_center_z),
                    theta_resolution=16,
                    phi_resolution=16,
                )
                
                # Vertikale Kreisfläche am Scheitelpunkt der Kugel (Y-Minusseite)
                meshes_to_combine = [sphere]
                
                # Kreisfläche positionieren (am Scheitelpunkt der Kugel auf Y-Minusseite)
                # Y-Position: Punktposition - Radius Kugel (tangential am Scheitelpunkt)
                disk_center_y = y_val - sphere_radius
                disk_center_x = x_val  # X-Position gleich wie Kugel
                # Dicke der Scheibe proportional zur Kugelgröße (10% des Radius = 5% des Durchmessers)
                disk_thickness = sphere_radius * 0.1  # Proportional zum Radius der Kugel
                
                # Vertikale Scheibe: von z=0 bis zur Oberseite der Kugel
                disk_height = sphere_top_z - z_val  # Höhe von z_val bis Oberseite der Kugel
                # Scheibe am Scheitelpunkt (Oberseite) der Kugel positionieren
                # Die Scheibe soll am Scheitelpunkt tangential zur Kugel sein
                # Die Scheibe reicht vertikal von z=0 bis zur Oberseite
                # Am Scheitelpunkt bedeutet: Die Scheibe soll am höchsten Punkt der Kugel tangential sein
                # Die Oberseite der Scheibe soll am Scheitelpunkt sein, also:
                # disk_center_z + disk_height/2 = sphere_top_z
                # disk_center_z = sphere_top_z - disk_height/2
                disk_center_z = sphere_top_z - disk_height / 2.0  # Mitte so, dass Oberseite am Scheitelpunkt ist
                
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"Plot3DOverlaysImpulse.py:140","message":"disk center z calculation","data":{"sphere_top_z":sphere_top_z,"disk_height":disk_height,"disk_center_z":disk_center_z,"calculation":"sphere_top_z - disk_height/2"},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"Plot3DOverlaysImpulse.py:135","message":"disk position calculation","data":{"x_val":x_val,"y_val":y_val,"z_val":z_val,"sphere_radius":sphere_radius,"sphere_center_z":sphere_center_z,"sphere_top_z":sphere_top_z,"disk_center_x":disk_center_x,"disk_center_y":disk_center_y,"disk_center_z":disk_center_z,"disk_height":disk_height,"disk_radius":disk_radius,"calculation_z":"sphere_top_z - disk_height/2","expected_scheitel_z":sphere_top_z,"expected_disk_top_z":disk_center_z + disk_height/2,"expected_disk_bottom_z":disk_center_z - disk_height/2},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Plot3DOverlaysImpulse.py:130","message":"sphere and disk params","data":{"measurement_size":getattr(settings,"measurement_size",4.0),"sphere_radius":sphere_radius,"z_val":z_val,"sphere_center_z":sphere_center_z,"sphere_top_z":sphere_top_z,"disk_height":disk_height,"disk_center_z":disk_center_z,"disk_center_y":disk_center_y},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                
                # Erstelle vertikale Kreisfläche als sehr dünnen Zylinder
                # Die Kreisfläche soll in der XZ-Ebene liegen (vertikal, senkrecht zur Y-Achse)
                # Ein Zylinder mit direction=(0,0,1) hat seine Achse in Z-Richtung (vertikal)
                # Die Kreisfläche liegt in der XY-Ebene - wir rotieren um 90° um die X-Achse
                # damit die Kreisfläche in der XZ-Ebene liegt
                disk = pv.Cylinder(
                    center=(disk_center_x, disk_center_y, disk_center_z),
                    direction=(0, 0, 1),  # Achse in Z-Richtung (vertikal)
                    radius=disk_radius,
                    height=disk_height,  # Vertikale Höhe von z_val bis Oberseite der Kugel
                    resolution=16,
                )
                # Rotiere um 90° um die X-Achse, damit die Kreisfläche in der XZ-Ebene liegt
                disk.rotate_x(90, inplace=True)
                # #region agent log
                try:
                    # Prüfe Position nach Rotation
                    bounds = disk.bounds
                    center_after_rotation = disk.center
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"Plot3DOverlaysImpulse.py:178","message":"disk after rotation","data":{"center_before_rotation":(disk_center_x,disk_center_y,disk_center_z),"center_after_rotation":center_after_rotation,"bounds":bounds},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                # Nach Rotation um X-Achse: Y und Z werden vertauscht
                # Die Scheibe sollte bei Y=disk_center_y (tangential am Scheitelpunkt) und Z=disk_center_z sein
                # Korrigiere die Position nach Rotation
                disk.translate([
                    disk_center_x - center_after_rotation[0],
                    disk_center_y - center_after_rotation[1],
                    disk_center_z - center_after_rotation[2]
                ], inplace=True)
                # #region agent log
                try:
                    center_after_translate = disk.center
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"Plot3DOverlaysImpulse.py:192","message":"disk after translate correction","data":{"center_after_translate":center_after_translate,"expected_center":(disk_center_x,disk_center_y,disk_center_z)},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                # Skaliere die Y-Dimension auf disk_thickness für die dünne Scheibe
                disk.scale([1.0, disk_thickness / disk_height, 1.0], inplace=True)
                # #region agent log
                try:
                    # Prüfe Position nach Skalierung
                    bounds_after_scale = disk.bounds
                    center_after_scale = disk.center
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"Plot3DOverlaysImpulse.py:202","message":"disk after scale","data":{"center_after_scale":center_after_scale,"bounds_after_scale":bounds_after_scale,"scale_factor_y":disk_thickness / disk_height},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                # Korrigiere Position nach Skalierung - die Skalierung verschiebt die Position
                # Die Scheibe soll am Scheitelpunkt tangential sein: Y-Position = disk_center_y
                disk.translate([
                    disk_center_x - center_after_scale[0],
                    disk_center_y - center_after_scale[1],
                    disk_center_z - center_after_scale[2]
                ], inplace=True)
                # #region agent log
                try:
                    center_after_scale_correction = disk.center
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"Plot3DOverlaysImpulse.py:211","message":"disk after scale correction","data":{"center_after_scale_correction":center_after_scale_correction,"expected_center":(disk_center_x,disk_center_y,disk_center_z),"sphere_scheitel_y":y_val},"timestamp":time.time()*1000}) + '\n')
                except: pass
                # #endregion
                meshes_to_combine.append(disk)
                # #region agent log
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Plot3DOverlaysImpulse.py:195","message":"disk created","data":{"disk_radius":disk_radius,"disk_height":disk_height,"disk_thickness":disk_thickness,"disk_center_z":disk_center_z,"disk_center_y":disk_center_y},"timestamp":time.time()*1000}) + '\n')
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

