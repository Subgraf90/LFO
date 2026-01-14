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
        self._last_measurement_size: Optional[float] = None

    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um State zurückzusetzen."""
        super().clear_category(category)
        if category == "impulse":
            self._last_impulse_state = None
            self._last_measurement_size = None

    def draw_impulse_points(self, settings) -> None:
        """Zeichnet Impulse Points als Kugel mit vertikaler Scheibe (IEC-Mikrofonsymbol)."""
        current_state = self._compute_impulse_state(settings)
        current_measurement_size = getattr(settings, "measurement_size", 4.0)
        existing_names = self._category_actors.get("impulse", [])

        # Skip, wenn sich nichts geändert hat und bereits Actors existieren
        if (self._last_impulse_state == current_state and 
            self._last_measurement_size == current_measurement_size and 
            existing_names):
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

        # Sammle Text-Label-Positionen und -Texte
        label_positions = []
        label_texts = []
        
        for x_val, y_val, z_val, number in current_state:
            try:
                # 🚀 Kugel + vertikale Kreisfläche (IEC-Mikrofonsymbol)
                # Messpunkt-Koordinaten (x_val, y_val, z_val) definieren:
                # - Y-Achse Nullpunkt: Anfang der Scheibe (bei y_val)
                # - Z-Achse Nullpunkt: Anfang der Scheibe (bei z_val)
                # - Kugel liegt tangential (anliegend) an der Scheibe
                
                # Kugelradius aus Settings lesen
                sphere_radius = getattr(settings, "measurement_size", 4.0) / 2.0  # measurement_size ist Durchmesser, Radius = Durchmesser / 2
                disk_thickness = sphere_radius * 0.1  # Proportional zum Radius der Kugel (10%)
                disk_radius = sphere_radius  # Durchmesser Kreis = Kugel-Durchmesser (proportional)
                
                # Kugel tangential zur Scheibe positioniert (anliegend)
                # Scheibe bei y=y_val, Kugel liegt tangential auf der Plus-Seite
                sphere_center_y = y_val + disk_thickness / 2.0 + sphere_radius
                sphere_center_z = z_val  # Kugelzentrum bei z_val
                sphere = pv.Sphere(
                    radius=sphere_radius,
                    center=(x_val, sphere_center_y, sphere_center_z),
                    theta_resolution=16,
                    phi_resolution=16,
                )
                
                # Sammle Label-Position und -Text (Text rechts neben der Kugel)
                label_offset = sphere_radius * 1.5  # Abstand vom Mittelpunkt der Kugel
                label_x = x_val + label_offset
                label_y = sphere_center_y
                label_z = sphere_center_z
                label_positions.append([label_x, label_y, label_z])
                label_texts.append(str(number))
                
                meshes_to_combine = [sphere]
                
                # Vertikale Scheibe: beginnt bei y=y_val (Nullpunkt), z=z_val bis zur Oberseite der Kugel
                disk_center_y = y_val + disk_thickness / 2.0
                disk_center_x = x_val  # X-Position gleich wie Kugel
                
                # Scheibe geht von z_val bis zur Oberseite der Kugel
                sphere_top_z = sphere_center_z + sphere_radius
                disk_height = sphere_top_z - z_val  # Höhe von z_val bis Oberseite der Kugel
                disk_center_z = z_val + disk_height / 2.0  # Mitte zwischen z_val und Oberseite
                
                # Erstelle vertikale Kreisscheibe als Zylinder in XZ-Ebene
                # Direkte Erstellung ohne Rotation für präzise Positionierung
                disk = pv.Cylinder(
                    center=(disk_center_x, disk_center_y, disk_center_z),
                    direction=(0, 1, 0),  # Achse in Y-Richtung (Scheibe liegt in XZ-Ebene)
                    radius=disk_radius,
                    height=disk_thickness,  # Dicke in Y-Richtung
                    resolution=16,
                )
                meshes_to_combine.append(disk)
                
                # Kombiniere Meshes zu einem einzigen Mesh (einfacher zu handhaben)
                if len(meshes_to_combine) > 1:
                    from pyvista import MultiBlock
                    multi_block = MultiBlock(meshes_to_combine)
                    combined_mesh = multi_block.combine()
                else:
                    combined_mesh = meshes_to_combine[0]
                
                self._add_overlay_mesh(
                    combined_mesh,
                    color="black",
                    opacity=0.5,
                    line_width=self._get_scaled_line_width(1.0),
                    category="impulse",
                    render_lines_as_tubes=False,
                )
            except Exception as e:
                print(f"[SPL3DOverlayImpulse] Fehler beim Erstellen des Messpunkt-Markers: {e}")
                continue
        
        # Füge Text-Labels für alle Messpunkte hinzu
        if label_positions and label_texts:
            try:
                import numpy as np
                # Schriftgröße dynamisch basierend auf measurement_size berechnen
                # measurement_size ist der Durchmesser der Kugel
                measurement_size = getattr(settings, "measurement_size", 4.0)
                # Skaliere Schriftgröße proportional: Basis 16 bei measurement_size=4.0
                # Mindestschriftgröße: 10, damit Text auch bei kleinen Messpunkten sichtbar bleibt
                font_size = max(10, int(16 * (measurement_size / 4.0)))
                
                # Erstelle eindeutigen Namen für die Labels
                label_actor_name = f"{self._overlay_prefix}labels"
                
                # Entferne alte Labels falls vorhanden
                try:
                    if hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                        if label_actor_name in self.plotter.renderer.actors:
                            self.plotter.remove_actor(label_actor_name)
                except Exception:
                    pass
                
                label_actor = self.plotter.add_point_labels(
                    np.array(label_positions),
                    label_texts,
                    name=label_actor_name,  # Gib dem Actor einen Namen
                    font_size=font_size,
                    text_color='black',
                    point_size=0,  # Keine Punkte anzeigen, nur Text
                    render_points_as_spheres=False,
                    always_visible=True,
                    shape_opacity=0.0  # Transparenter Hintergrund
                )
                # Speichere den Label-Actor in der Kategorie "impulse"
                if label_actor is not None:
                    self.overlay_actor_names.append(label_actor_name)
                    self._category_actors.setdefault("impulse", []).append(label_actor_name)
            except Exception as e:
                print(f"[SPL3DOverlayImpulse] Fehler beim Hinzufügen der Text-Labels: {e}")

        self._last_impulse_state = current_state
        self._last_measurement_size = current_measurement_size

    def _compute_impulse_state(self, settings) -> tuple:
        """Erzeugt eine robuste Signatur der Impulse Points (inkl. Nummer)."""
        impulse_points = getattr(settings, "impulse_points", []) or []
        impulse_signature: List[Tuple[float, float, float, int]] = []
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
                # Hole die Nummer des Messpunkts
                number = point.get("number", 0)
                impulse_signature.append((float(x_val), float(y_val), float(z_val), int(number)))
            except (ValueError, TypeError, KeyError, IndexError) as e:
                # Debug-Output für Fehlerfälle
                print(f"[SPL3DOverlayImpulse] Fehler beim Verarbeiten eines Impulse Points: {e}, point={point}")
                continue
        return tuple(sorted(impulse_signature))


__all__ = ["SPL3DOverlayImpulse"]
