from __future__ import annotations

import importlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition


class SurfaceDataImporter:
    """
    Importiert Surface-Definitionen aus externen Dateien und überträgt sie in die Settings.
    """

    FILE_FILTER = "DXF-Dateien (*.dxf)"

    def __init__(self, parent_widget, settings, container, group_manager=None):
        self.parent_widget = parent_widget
        self.settings = settings
        self._container = container  # Reserviert für künftige Integrationen
        self.group_manager = group_manager
        self._group_cache: Dict[str, str] = {}

    def execute(self) -> bool | None:
        """
        Startet den Importdialog und aktualisiert bei Erfolg die Surface-Definitionen.

        Returns:
            True bei erfolgreichem Import, False bei Fehler, None wenn abgebrochen.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent_widget,
            "Surface-Datei laden",
            "",
            self.FILE_FILTER,
        )

        if not file_path:
            return None

        path = Path(file_path)

        if path.suffix.lower() == ".dxf":
            try:
                surfaces = self._load_dxf_surfaces(path)
            except ImportError:
                QMessageBox.warning(
                    self.parent_widget,
                    "DXF-Import nicht möglich",
                    "Für den Import von DXF-Dateien wird das Paket 'ezdxf' benötigt.\n"
                    "Bitte installieren Sie es z. B. mit 'pip install ezdxf'.",
                )
                return False
            except Exception as exc:
                QMessageBox.warning(
                    self.parent_widget,
                    "DXF-Import fehlgeschlagen",
                    f"Die DXF-Datei konnte nicht verarbeitet werden:\n{exc}",
                )
                return False
            else:
                if surfaces and self._ask_clear_existing_surfaces():
                    self._clear_existing_surfaces()
        else:
            try:
                payload = self._load_payload(path)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(
                    self.parent_widget,
                    "Surface-Import fehlgeschlagen",
                    f"Die Datei konnte nicht geladen werden:\n{exc}",
                )
                return False

            surfaces = self._parse_surfaces(payload)
        if not surfaces:
            QMessageBox.information(
                self.parent_widget,
                "Keine gültigen Flächen",
                "In der ausgewählten Datei wurden keine gültigen Surface-Definitionen gefunden.",
            )
            return False

        imported_count = self._store_surfaces(surfaces)
        QMessageBox.information(
            self.parent_widget,
            "Surface-Import erfolgreich",
            f"{imported_count} Flächen wurden importiert.",
        )
        return True

    def _load_payload(self, file_path: Path) -> Any:
        text = file_path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ungültiges JSON-Format ({exc})") from exc

    def _parse_surfaces(self, payload: Any) -> Dict[str, SurfaceDefinition]:
        entries: Dict[str, Dict[str, Any]] = {}

        if isinstance(payload, dict):
            if "surface_definitions" in payload and isinstance(payload["surface_definitions"], dict):
                entries = payload["surface_definitions"]
            else:
                entries = payload
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                surface_id = str(item.get("surface_id") or f"surface_{index + 1}")
                entries[surface_id] = item

        surfaces: Dict[str, SurfaceDefinition] = {}
        for surface_id, data in entries.items():
            if not isinstance(data, dict):
                continue
            normalized_id = str(data.get("surface_id") or surface_id).strip()
            if not normalized_id:
                normalized_id = f"surface_{len(surfaces) + 1}"
            surface = SurfaceDefinition.from_dict(normalized_id, data)
            surfaces[normalized_id] = surface

        return surfaces

    def _store_surfaces(self, surfaces: Dict[str, SurfaceDefinition]) -> int:
        imported = 0
        for surface_id, surface in surfaces.items():
            if hasattr(self.settings, "add_surface_definition"):
                self.settings.add_surface_definition(surface_id, surface, make_active=False)
            else:
                surface_store = getattr(self.settings, "surface_definitions", None)
                if not isinstance(surface_store, dict):
                    surface_store = {}
                surface_store[surface_id] = surface
                setattr(self.settings, "surface_definitions", surface_store)
            imported += 1
        if self.group_manager:
            self.group_manager.ensure_surface_group_structure()
        return imported

    # ---- user confirmation ------------------------------------------

    def _ask_clear_existing_surfaces(self) -> bool:
        reply = QMessageBox.question(
            self.parent_widget,
            "Replace existing surfaces?",
            (
                "Do you want to remove all currently defined surfaces before importing "
                "the DXF geometry?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _clear_existing_surfaces(self) -> None:
        if self.group_manager:
            self.group_manager.reset_surface_storage()
            return

        if hasattr(self.settings, "surface_definitions"):
            initializer = getattr(self.settings, "_initialize_surface_definitions", None)
            if callable(initializer):
                self.settings.surface_definitions = initializer()
            else:
                self.settings.surface_definitions = {}

        if hasattr(self.settings, "surface_groups"):
            group_initializer = getattr(self.settings, "_initialize_surface_groups", None)
            if callable(group_initializer):
                self.settings.surface_groups = group_initializer()
            else:
                self.settings.surface_groups = {}

    # ---- DXF support -------------------------------------------------

    def _load_dxf_surfaces(self, file_path: Path) -> Dict[str, SurfaceDefinition]:
        try:
            ezdxf = importlib.import_module("ezdxf")
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Das Paket 'ezdxf' ist nicht installiert.") from exc

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()
        self._log_dxf_debug_info(file_path, doc, msp)
        group_lookup = self._build_dxf_group_lookup(doc)
        layer_colors = self._build_layer_color_lookup(doc, ezdxf)

        surfaces: Dict[str, SurfaceDefinition] = {}
        id_counters: Dict[str, int] = {}

        for entity, transform in self._iter_surface_entities(msp):
            points = self._extract_points_from_entity(entity, transform)

            if not points:
                self._log_skipped_entity(entity)
                continue

            dxftype = entity.dxftype()
            base_name = getattr(entity.dxf, "layer", None) or dxftype or "DXF_Surface"
            group_name = group_lookup.get(entity.dxf.handle)
            target_group_id = self._ensure_group_for_label(group_name)

            safe_label = base_name.replace(" ", "_")
            id_counters[safe_label] = id_counters.get(safe_label, 0) + 1
            suffix = f"_{id_counters[safe_label]}" if id_counters[safe_label] > 1 else ""
            surface_id = f"{safe_label}{suffix}"

            color = self._resolve_entity_color(entity, layer_colors, ezdxf)

            payload = {
                "name": base_name,
                "enabled": False,
                "hidden": False,
                "locked": False,
                "points": points,
            }
            if target_group_id:
                payload["group_id"] = target_group_id
            if color:
                payload["color"] = color

            surfaces[surface_id] = SurfaceDefinition.from_dict(
                surface_id,
                payload,
            )

        return surfaces

    def _log_dxf_debug_info(self, file_path: Path, doc, msp) -> None:
        logger = logging.getLogger(__name__)
        try:
            entity_types = Counter(entity.dxftype() for entity in msp)
            top_types = ", ".join(
                f"{etype}:{count}" for etype, count in entity_types.most_common(5)
            )
            layers = getattr(doc, "layers", None)
            groups = getattr(doc, "groups", None)
            layer_count = len(list(layers)) if layers is not None else 0
            group_count = len(list(groups)) if groups is not None else 0
            logger.info(
                "DXF '%s': %d Entities im Modelspace, %d Layer, %d Gruppen. Häufigste Typen: %s",
                file_path.name,
                sum(entity_types.values()),
                layer_count,
                group_count,
                top_types or "keine",
            )
            self._log_block_summary(doc, logger)
        except Exception:
            logger.exception("DXF-Debug-Auswertung von '%s' fehlgeschlagen", file_path)

    def _log_skipped_entity(self, entity) -> None:
        logger = logging.getLogger(__name__)
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            logger.debug(
                "DXF-Entity %s auf Layer '%s' wurde übersprungen (keine Punkte). Handle=%s",
                entity.dxftype(),
                getattr(entity.dxf, "layer", "unbekannt"),
                getattr(entity.dxf, "handle", "n/a"),
            )
        except Exception:
            logger.exception("Fehler beim Loggen einer übersprungenen Entity")

    def _log_block_summary(self, doc, logger: logging.Logger) -> None:
        try:
            blocks = getattr(doc, "blocks", None)
            if not blocks:
                logger.info("DXF: keine Blockdefinitionen gefunden.")
                return
            summaries = []
            for block in blocks:
                entity_types = Counter(entity.dxftype() for entity in block)
                top_types = ", ".join(
                    f"{etype}:{count}" for etype, count in entity_types.most_common(3)
                ) or "leer"
                summaries.append(f"{block.name} [{len(entity_types)} Typen] -> {top_types}")
            if summaries:
                logger.info("DXF-Blockübersicht (%d Blöcke): %s", len(summaries), " | ".join(summaries))
        except Exception:
            logger.exception("DXF-Blockübersicht konnte nicht erstellt werden")

    def _extract_lwpolyline_points(self, entity) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        elevation = float(getattr(entity.dxf, "elevation", 0.0))
        # entity.get_points("xyb") -> (x, y, bulge)
        for x, y, *_ in entity.get_points("xyb"):
            points.append(self._make_point(x, y, elevation))

        if getattr(entity, "closed", False) and points and points[0] != points[-1]:
            points.append(points[0].copy())
        return points

    @staticmethod
    def _build_dxf_group_lookup(doc) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        groups = getattr(doc, "groups", None)
        if not groups:
            return lookup

        for group in groups:
            name = getattr(group, "name", None)
            if not name:
                continue
            try:
                handles = list(group.get_entity_handles())
            except Exception:
                continue
            for handle in handles:
                lookup[handle] = name
        return lookup

    @staticmethod
    def _build_layer_color_lookup(doc, ezdxf_module) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        layers = getattr(doc, "layers", None)
        if not layers:
            return lookup

        for layer in layers:
            name = getattr(layer.dxf, "name", None)
            if not name:
                continue
            color = SurfaceDataImporter._extract_color_from_dxf(layer.dxf, ezdxf_module)
            if color:
                lookup[name] = color
        return lookup

    def _resolve_entity_color(self, entity, layer_colors: Dict[str, str], ezdxf_module) -> Optional[str]:
        # True-Color direkt am Entity
        color = self._extract_color_from_dxf(entity.dxf, ezdxf_module)
        if color:
            return color

        # Fallback: Layer-Farbe
        layer_name = getattr(entity.dxf, "layer", None)
        if layer_name:
            return layer_colors.get(layer_name)
        return None

    @staticmethod
    def _extract_color_from_dxf(dxf_attrs, ezdxf_module) -> Optional[str]:
        true_color = getattr(dxf_attrs, "true_color", 0)
        if true_color:
            return SurfaceDataImporter._int_to_hex_color(true_color)

        color_index = getattr(dxf_attrs, "color", 0)
        if color_index:
            aci2rgb = None
            try:
                colors_module = importlib.import_module("ezdxf.colors")
                aci2rgb = getattr(colors_module, "aci2rgb", None)
            except Exception:
                pass
            if aci2rgb is None and hasattr(ezdxf_module, "colors"):
                aci2rgb = getattr(ezdxf_module.colors, "aci2rgb", None)
            if aci2rgb:
                r, g, b = aci2rgb(color_index)
                return SurfaceDataImporter._rgb_to_hex(r, g, b)
        return None

    @staticmethod
    def _int_to_hex_color(value: int) -> str:
        # DXF True Color: 0x00RRGGBB
        r = (value >> 16) & 0xFF
        g = (value >> 8) & 0xFF
        b = value & 0xFF
        return SurfaceDataImporter._rgb_to_hex(r, g, b)

    @staticmethod
    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02X}{g:02X}{b:02X}"

    def _extract_polyline_points(self, entity) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for vertex in entity.vertices:
            x = float(vertex.dxf.x)
            y = float(vertex.dxf.y)
            z = float(vertex.dxf.z)
            points.append(self._make_point(x, y, z))

        if entity.is_closed and points and points[0] != points[-1]:
            points.append(points[0].copy())
        return points

    def _extract_3dface_points(self, entity) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        for x, y, z in entity.points():
            points.append(self._make_point(x, y, z))
        return points

    @staticmethod
    def _make_point(x: float, y: float, z: float) -> Dict[str, float]:
        return {"x": float(x), "y": float(y), "z": float(z)}

    def _ensure_group_for_label(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        key = label.strip()
        if not key:
            return None
        if key in self._group_cache:
            return self._group_cache[key]

        group_id = None
        if self.group_manager:
            group_id = self.group_manager.ensure_group_path(key)

        self._group_cache[key] = group_id
        return group_id

    # ---- Helpers für DXF-Entities ------------------------------------

    def _extract_points_from_entity(self, entity, transform) -> List[Dict[str, float]] | None:
        dxftype = entity.dxftype()
        points = None
        
        if dxftype == "LWPOLYLINE":
            points = self._extract_lwpolyline_points(entity)
        elif dxftype == "POLYLINE":
            points = self._extract_polyline_points(entity)
        elif dxftype == "3DFACE":
            points = self._extract_3dface_points(entity)
        elif dxftype == "LINE":
            points = self._extract_line_points(entity)
        
        if not points:
            return None
        
        # Wende Transformation an
        if transform:
            return [self._apply_transform_to_point(p, transform) for p in points]
        
        return points

    def _extract_line_points(self, entity) -> List[Dict[str, float]]:
        """Extrahiert Punkte aus einer LINE-Entity"""
        start = getattr(entity.dxf, "start", None)
        end = getattr(entity.dxf, "end", None)
        
        if start is None or end is None:
            return []
        
        points = []
        points.append(self._make_point(
            float(start.x) if hasattr(start, 'x') else 0.0,
            float(start.y) if hasattr(start, 'y') else 0.0,
            float(start.z) if hasattr(start, 'z') else 0.0
        ))
        points.append(self._make_point(
            float(end.x) if hasattr(end, 'x') else 0.0,
            float(end.y) if hasattr(end, 'y') else 0.0,
            float(end.z) if hasattr(end, 'z') else 0.0
        ))
        
        return points

    def _iter_surface_entities(self, layout):
        for entity in layout:
            yield from self._resolve_entity(entity, depth=0, transform=None)
    
    def _get_identity_transform(self):
        """Gibt eine Identitäts-Transformation zurück"""
        return {
            "translation": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
            "rotation": 0.0,
        }

    def _resolve_entity(self, entity, depth: int, transform):
        if entity.dxftype() == "INSERT":
            yield from self._expand_insert(entity, depth, transform)
        else:
            yield (entity, transform)

    def _expand_insert(self, insert_entity, depth: int, parent_transform):
        logger = logging.getLogger(__name__)
        max_depth = 10
        if depth >= max_depth:
            logger.warning(
                "INSERT-Verschachtelungstiefe %d überschreitet das Limit (%d). Handle=%s",
                depth,
                max_depth,
                getattr(insert_entity.dxf, "handle", "n/a"),
            )
            return
        
        block_name = getattr(insert_entity.dxf, "name", "unbenannt")
        
        # Extrahiere Transformationen aus der INSERT-Entity
        insert_transform = self._extract_insert_transform(insert_entity)
        
        # Kombiniere mit Parent-Transformation
        if parent_transform:
            combined_transform = self._combine_transforms(parent_transform, insert_transform)
        else:
            combined_transform = insert_transform
        
        try:
            virtual_entities = list(insert_entity.virtual_entities())
            if logger.isEnabledFor(logging.DEBUG):
                entity_count = len(virtual_entities)
                logger.debug(
                    "INSERT '%s' (Tiefe %d) aufgelöst: %d Entities extrahiert",
                    block_name,
                    depth,
                    entity_count,
                )
        except Exception as exc:
            logger.warning(
                "INSERT '%s' konnte nicht expandiert werden: %s",
                block_name,
                exc,
            )
            return

        try:
            for virtual in virtual_entities:
                yield from self._resolve_entity(virtual, depth + 1, combined_transform)
        finally:
            for virtual in virtual_entities:
                try:
                    virtual.destroy()
                except Exception:
                    pass

    def _extract_insert_transform(self, insert_entity):
        """Extrahiert Transformation (Position, Skalierung, Rotation) aus INSERT-Entity"""
        import math
        
        # Insertion Point
        insert_point = getattr(insert_entity.dxf, "insert", None)
        if insert_point is None:
            tx, ty, tz = 0.0, 0.0, 0.0
        else:
            tx = float(insert_point.x) if hasattr(insert_point, 'x') else 0.0
            ty = float(insert_point.y) if hasattr(insert_point, 'y') else 0.0
            tz = float(insert_point.z) if hasattr(insert_point, 'z') else 0.0
        
        # Skalierung
        xscale = float(getattr(insert_entity.dxf, "xscale", 1.0))
        yscale = float(getattr(insert_entity.dxf, "yscale", 1.0))
        zscale = float(getattr(insert_entity.dxf, "zscale", 1.0))
        
        # Rotation (in Grad)
        rotation = float(getattr(insert_entity.dxf, "rotation", 0.0))
        rotation_rad = math.radians(rotation)
        
        return {
            "translation": (tx, ty, tz),
            "scale": (xscale, yscale, zscale),
            "rotation": rotation_rad,
        }

    def _combine_transforms(self, parent, child):
        """Kombiniert zwei Transformationen"""
        import math
        
        # Parent-Transformation
        pt = parent["translation"]
        ps = parent["scale"]
        pr = parent["rotation"]
        
        # Child-Transformation
        ct = child["translation"]
        cs = child["scale"]
        cr = child["rotation"]
        
        # Kombinierte Skalierung
        combined_scale = (ps[0] * cs[0], ps[1] * cs[1], ps[2] * cs[2])
        
        # Kombinierte Rotation
        combined_rotation = pr + cr
        
        # Transformiere Child-Translation mit Parent-Transformation
        # Zuerst skalieren
        scaled_ct = (ct[0] * ps[0], ct[1] * ps[1], ct[2] * ps[2])
        
        # Dann rotieren (um Z-Achse)
        cos_r = math.cos(pr)
        sin_r = math.sin(pr)
        rotated_ct = (
            scaled_ct[0] * cos_r - scaled_ct[1] * sin_r,
            scaled_ct[0] * sin_r + scaled_ct[1] * cos_r,
            scaled_ct[2]
        )
        
        # Dann translatieren
        combined_translation = (
            pt[0] + rotated_ct[0],
            pt[1] + rotated_ct[1],
            pt[2] + rotated_ct[2]
        )
        
        return {
            "translation": combined_translation,
            "scale": combined_scale,
            "rotation": combined_rotation,
        }

    def _apply_transform_to_point(self, point: Dict[str, float], transform) -> Dict[str, float]:
        """Wendet Transformation auf einen Punkt an"""
        if not transform:
            return point
        
        import math
        
        x = point.get("x", 0.0)
        y = point.get("y", 0.0)
        z = point.get("z", 0.0)
        
        # Skalierung
        sx, sy, sz = transform["scale"]
        x *= sx
        y *= sy
        z *= sz
        
        # Rotation (um Z-Achse)
        rotation = transform["rotation"]
        if rotation != 0.0:
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            x_new = x * cos_r - y * sin_r
            y_new = x * sin_r + y * cos_r
            x, y = x_new, y_new
        
        # Translation
        tx, ty, tz = transform["translation"]
        x += tx
        y += ty
        z += tz
        
        return {"x": float(x), "y": float(y), "z": float(z)}

