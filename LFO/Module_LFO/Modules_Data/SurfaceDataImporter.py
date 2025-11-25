from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition


@dataclass(frozen=True)
class InsertContext:
    name: str
    handle: Optional[str]
    instance_index: int
    tag: Optional[str]
    layer: Optional[str]


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
        self._block_tag_map: Dict[str, str] = {}  # Mapping von Block-Namen zu Tags
        self._block_instance_counters: Counter[str] = Counter()

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
        
        # Analysiere Tags/Attribute in der DXF-Datei
        self._analyze_dxf_tags(doc, msp)

        surfaces: Dict[str, SurfaceDefinition] = {}
        id_counters: Dict[str, int] = {}
        
        logger = logging.getLogger(__name__)
        entity_count = 0
        processed_count = 0
        skipped_count = 0
        entity_type_stats = Counter()  # Statistik welche Entity-Typen verarbeitet werden

        for entity, transform, block_stack in self._iter_surface_entities(msp):
            entity_count += 1
            dxftype = entity.dxftype()
            layer = getattr(entity.dxf, "layer", "unbekannt")
            
            points = self._extract_points_from_entity(entity, transform)

            if not points:
                logger.debug(
                    "Entity %d: %s auf Layer '%s' übersprungen (keine Punkte extrahiert)",
                    entity_count, dxftype, layer
                )
                entity_type_stats[f"{dxftype}_keine_punkte"] += 1
                skipped_count += 1
                self._log_skipped_entity(entity)
                continue
            
            # LINE-Entities haben nur 2 Punkte - keine Fläche, überspringen
            if dxftype == "LINE" and len(points) == 2:
                start = points[0]
                end = points[1]
                logger.info(
                    "Entity %d: LINE auf Layer '%s' übersprungen (nur 2 Punkte, keine Fläche)",
                    entity_count, layer
                )
                logger.info(
                    "  Start: (%.3f, %.3f, %.3f) -> Ende: (%.3f, %.3f, %.3f)",
                    start.get("x", 0), start.get("y", 0), start.get("z", 0),
                    end.get("x", 0), end.get("y", 0), end.get("z", 0)
                )
                entity_type_stats["LINE_übersprungen"] += 1
                skipped_count += 1
                continue
            
            # Mindestens 3 Punkte für eine Fläche erforderlich
            # (Auch offene Polylines können als Flächen verwendet werden)
            if len(points) < 3:
                logger.info(
                    "Entity %d: %s auf Layer '%s' übersprungen (nur %d Punkte, mindestens 3 erforderlich)",
                    entity_count, dxftype, layer, len(points)
                )
                entity_type_stats[f"{dxftype}_zu_wenig_punkte"] += 1
                skipped_count += 1
                continue
            
            # Prüfe ob geschlossen (für Polylines)
            is_closed = self._is_entity_closed(entity, points)
            
            # Prüfe auch, ob die Fläche nahezu geschlossen ist (erster und letzter Punkt sehr nah)
            if not is_closed and len(points) >= 3:
                first = points[0]
                last = points[-1]
                tolerance = 0.01  # 1cm Toleranz für nahezu geschlossene Flächen
                dx = abs(first.get("x", 0) - last.get("x", 0))
                dy = abs(first.get("y", 0) - last.get("y", 0))
                dz = abs(first.get("z", 0) - last.get("z", 0))
                distance = (dx**2 + dy**2 + dz**2)**0.5
                if distance < tolerance:
                    is_closed = True
                    logger.info(
                        "Entity %d: %s auf Layer '%s' als nahezu geschlossen erkannt (Abstand: %.4f)",
                        entity_count, dxftype, layer, distance
                    )
            
            # Wenn geschlossen, aber erster und letzter Punkt nicht identisch, schließe die Fläche
            if is_closed and len(points) >= 3:
                first = points[0]
                last = points[-1]
                tolerance = 1e-6
                dx = abs(first.get("x", 0) - last.get("x", 0))
                dy = abs(first.get("y", 0) - last.get("y", 0))
                dz = abs(first.get("z", 0) - last.get("z", 0))
                if dx >= tolerance or dy >= tolerance or dz >= tolerance:
                    # Füge ersten Punkt am Ende hinzu, um zu schließen
                    points.append(first.copy())
            
            processed_count += 1
            entity_type_stats[f"{dxftype}_verarbeitet"] += 1
            
            # Detaillierte Koordinaten-Ausgabe
            logger.info(
                "Entity %d: %s auf Layer '%s' verarbeitet - %d Punkte, geschlossen=%s",
                entity_count, dxftype, layer, len(points), is_closed
            )
            
            # Zeige erste und letzte Koordinaten
            if points:
                first = points[0]
                last = points[-1]
                logger.info(
                    "  Erster Punkt: (%.3f, %.3f, %.3f)",
                    first.get("x", 0), first.get("y", 0), first.get("z", 0)
                )
                if len(points) > 1:
                    logger.info(
                        "  Letzter Punkt: (%.3f, %.3f, %.3f)",
                        last.get("x", 0), last.get("y", 0), last.get("z", 0)
                    )
                if len(points) <= 10:
                    # Zeige alle Punkte wenn nicht zu viele
                    logger.info("  Alle Punkte:")
                    for i, p in enumerate(points):
                        logger.info(
                            "    P%d: (%.3f, %.3f, %.3f)",
                            i+1, p.get("x", 0), p.get("y", 0), p.get("z", 0)
                        )
                else:
                    # Zeige nur erste 3 und letzte 3 Punkte
                    logger.info("  Erste 3 Punkte:")
                    for i in range(min(3, len(points))):
                        p = points[i]
                        logger.info(
                            "    P%d: (%.3f, %.3f, %.3f)",
                            i+1, p.get("x", 0), p.get("y", 0), p.get("z", 0)
                        )
                    logger.info("  ... (%d weitere Punkte) ...", len(points) - 6)
                    logger.info("  Letzte 3 Punkte:")
                    for i in range(max(0, len(points)-3), len(points)):
                        p = points[i]
                        logger.info(
                            "    P%d: (%.3f, %.3f, %.3f)",
                            i+1, p.get("x", 0), p.get("y", 0), p.get("z", 0)
                        )
            
            # Spezielle Info für MESH-Entities
            if dxftype in ("MESH", "POLYLINE") and hasattr(entity, "is_mesh"):
                logger.info(
                    "  -> MESH-Entity erkannt (is_mesh=%s)",
                    getattr(entity, "is_mesh", False)
                )

            dxftype = entity.dxftype()
            base_name = getattr(entity.dxf, "layer", None) or dxftype or "DXF_Surface"
            
            # Extrahiere Tag/Attribut aus Entity (z.B. "WOOD", "BETON")
            tag = self._extract_tag_from_entity(entity, doc)
            if not tag:
                tag = self._resolve_tag_from_context(block_stack)
            
            # Verwende Tag als Gruppierung, falls vorhanden
            if tag:
                logger.info(
                    "Entity %d: Tag '%s' gefunden für %s auf Layer '%s'",
                    entity_count, tag, dxftype, layer
                )
                # Verwende Tag als Basis-Name, falls Layer nicht aussagekräftig
                if not base_name or base_name == dxftype:
                    base_name = tag
            # Gruppierung über Block-Hierarchie
            group_label = self._build_group_label(block_stack)
            target_group_id = self._ensure_group_for_label(group_label)
            if not target_group_id:
                # Fallback auf ursprüngliche Gruppen-Logik
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

        # Zusammenfassung
        logger.info("=" * 80)
        logger.info("DXF-IMPORT-ZUSAMMENFASSUNG")
        logger.info("=" * 80)
        logger.info("Entities gesamt: %d", entity_count)
        logger.info("Entities verarbeitet: %d", processed_count)
        logger.info("Entities übersprungen: %d", skipped_count)
        logger.info("Surfaces erstellt: %d", len(surfaces))
        
        # Entity-Typ-Statistik
        if entity_type_stats:
            logger.info("\nEntity-Typ-Statistik:")
            for stat_type, count in sorted(entity_type_stats.items()):
                logger.info("  %s: %d", stat_type, count)
        
        # Zeige welche Entity-Typen geladen wurden
        if surfaces:
            logger.info("\nErstellte Surfaces:")
            for surface_id, surface in surfaces.items():
                points = getattr(surface, "points", [])
                group_id = getattr(surface, "group_id", None)
                name = getattr(surface, "name", surface_id)
                logger.info(
                    "  - %s (ID: %s): %d Punkte, Gruppe: %s",
                    name, surface_id, len(points), group_id or "keine"
                )
        
        logger.info("=" * 80)
        
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
            
            # Detaillierte Analyse aller Entity-Typen
            if logger.isEnabledFor(logging.DEBUG):
                all_types = Counter()
                for entity in msp:
                    all_types[entity.dxftype()] += 1
                logger.debug(
                    "Alle Entity-Typen im Modelspace: %s",
                    dict(all_types)
                )
            
            self._log_block_summary(doc, logger)
            
            # Analysiere geschlossene Polylines in Blöcken
            self._log_closed_polylines_in_blocks(doc, logger)
            
            # Detaillierte Struktur-Analyse
            self._analyze_dxf_structure(doc, msp, logger)
        except Exception:
            logger.exception("DXF-Debug-Auswertung von '%s' fehlgeschlagen", file_path)
    
    def _analyze_dxf_structure(self, doc, msp, logger: logging.Logger) -> None:
        """Analysiert und erklärt die DXF-Struktur detailliert"""
        logger.info("=" * 80)
        logger.info("DXF-STRUKTUR-ANALYSE")
        logger.info("=" * 80)
        
        # 1. Modelspace-Struktur
        logger.info("\n1. MODELSPACE (Hauptebene):")
        logger.info("   - Enthält direkte Entities oder INSERT-Referenzen zu Blöcken")
        insert_count = 0
        for entity in msp:
            if entity.dxftype() == "INSERT":
                insert_count += 1
                block_name = getattr(entity.dxf, "name", "unbekannt")
                layer = getattr(entity.dxf, "layer", "unbekannt")
                insert_point = getattr(entity.dxf, "insert", None)
                logger.info(f"   - INSERT: Block '{block_name}' auf Layer '{layer}'")
                if insert_point:
                    logger.info(f"     Position: ({insert_point.x:.2f}, {insert_point.y:.2f}, {insert_point.z:.2f})")
        logger.info(f"   Gesamt: {insert_count} INSERT-Entities (Block-Referenzen)")
        
        # 2. Block-Struktur
        logger.info("\n2. BLÖCKE (Block-Definitionen):")
        blocks = getattr(doc, "blocks", None)
        if blocks:
            for block in blocks:
                if block.name.startswith("*"):  # System-Blöcke überspringen
                    continue
                
                logger.info(f"\n   Block: '{block.name}'")
                
                # Analysiere Entities im Block
                entity_types_in_block = Counter()
                polyline_count = 0
                line_count = 0
                closed_polylines = 0
                
                for entity in block:
                    etype = entity.dxftype()
                    entity_types_in_block[etype] += 1
                    
                    if etype == "POLYLINE":
                        polyline_count += 1
                        if hasattr(entity, "is_closed") and entity.is_closed:
                            closed_polylines += 1
                        # Zähle Vertices
                        try:
                            vertex_count = len(list(entity.vertices))
                            logger.info(f"     - POLYLINE: {vertex_count} Vertices, geschlossen={entity.is_closed}")
                        except:
                            logger.info(f"     - POLYLINE: (Vertices nicht lesbar)")
                    
                    elif etype == "LWPOLYLINE":
                        polyline_count += 1
                        is_closed = getattr(entity, "closed", False)
                        if is_closed:
                            closed_polylines += 1
                        try:
                            point_count = len(list(entity.get_points("xy")))
                            logger.info(f"     - LWPOLYLINE: {point_count} Punkte, geschlossen={is_closed}")
                        except:
                            logger.info(f"     - LWPOLYLINE: (Punkte nicht lesbar)")
                    
                    elif etype == "LINE":
                        line_count += 1
                
                logger.info(f"     Zusammenfassung: {dict(entity_types_in_block)}")
                logger.info(f"     Polylines: {polyline_count} (davon {closed_polylines} geschlossen)")
                logger.info(f"     Linien: {line_count}")
                
                # Prüfe Layer der Entities
                layers_in_block = set()
                for entity in block:
                    layer = getattr(entity.dxf, "layer", None)
                    if layer:
                        layers_in_block.add(layer)
                if layers_in_block:
                    logger.info(f"     Layer: {', '.join(sorted(layers_in_block))}")
        
        # 3. Tag-Zuordnung
        logger.info("\n3. TAG-ZUORDNUNG (Flächen-Kategorisierung):")
        logger.info("   Tags werden aus folgenden Quellen extrahiert:")
        logger.info("   a) Layer-Namen (z.B. Layer 'BETON' oder 'WOOD')")
        logger.info("   b) Block-Namen (wenn Block-Name ein Tag ist)")
        logger.info("   c) Block-Attribute (ATTRIB-Entities in INSERT-Blöcken)")
        logger.info("   d) XDATA/AppData (erweiterte Daten)")
        
        # 4. Flächen-Definition
        logger.info("\n4. FLÄCHEN-DEFINITION:")
        logger.info("   Flächen werden durch folgende Entity-Typen definiert:")
        logger.info("   - POLYLINE: Geschlossene oder offene Polylinie mit Vertices")
        logger.info("   - LWPOLYLINE: Leichtgewichtige Polylinie (2D, kann geschlossen sein)")
        logger.info("   - 3DFACE: Dreieckige Fläche mit 3-4 Eckpunkten")
        logger.info("   - MESH: Polygon-Mesh mit Vertices und Faces")
        logger.info("   - SOLID: Gefüllte Fläche (wird als 4-Punkt-Fläche behandelt)")
        logger.info("\n   Mindestanforderungen für eine Fläche:")
        logger.info("   - Mindestens 3 Punkte erforderlich")
        logger.info("   - Geschlossene Polylines werden bevorzugt")
        logger.info("   - Offene Polylines mit >= 3 Punkten werden auch akzeptiert")
        
        # 5. Transformationen
        logger.info("\n5. TRANSFORMATIONEN (bei INSERT-Entities):")
        logger.info("   INSERT-Entities können transformiert werden:")
        logger.info("   - Insertion Point (Verschiebung)")
        logger.info("   - Skalierung (X, Y, Z)")
        logger.info("   - Rotation (um Z-Achse)")
        logger.info("   - Verschachtelte Transformationen werden kombiniert")
        
        logger.info("\n" + "=" * 80)
    
    def _log_closed_polylines_in_blocks(self, doc, logger: logging.Logger) -> None:
        """Analysiert geschlossene Polylines in Blöcken"""
        try:
            blocks = getattr(doc, "blocks", None)
            if not blocks:
                return
            
            closed_count = 0
            open_count = 0
            line_count = 0
            
            for block in blocks:
                if block.name.startswith("*"):  # System-Blöcke überspringen
                    continue
                for entity in block:
                    dxftype = entity.dxftype()
                    if dxftype == "LWPOLYLINE":
                        if getattr(entity, "closed", False):
                            closed_count += 1
                        else:
                            open_count += 1
                    elif dxftype == "POLYLINE":
                        if hasattr(entity, "is_closed") and entity.is_closed:
                            closed_count += 1
                        else:
                            open_count += 1
                    elif dxftype == "LINE":
                        line_count += 1
            
            if closed_count > 0 or open_count > 0 or line_count > 0:
                logger.info(
                    "DXF-Blöcke: %d geschlossene Polylines, %d offene Polylines, %d Linien",
                    closed_count, open_count, line_count
                )
        except Exception:
            logger.exception("Fehler beim Analysieren geschlossener Polylines")

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
        try:
            for x, y, *_ in entity.get_points("xyb"):
                points.append(self._make_point(x, y, elevation))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Fehler beim Extrahieren von LWPOLYLINE-Punkten: %s", e)
            return []

        # Prüfe auf geschlossen - verschiedene Attribute möglich
        is_closed = False
        if hasattr(entity, "closed"):
            is_closed = bool(entity.closed)
        elif hasattr(entity.dxf, "flags"):
            # Bit 1 = closed flag
            flags = getattr(entity.dxf, "flags", 0)
            is_closed = bool(flags & 1)
        
        if is_closed and points and points[0] != points[-1]:
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
    
    @staticmethod
    def _extract_float_from_vec3(value, default=0.0):
        """
        Rekursiv extrahiert einen Float-Wert aus einem Vec3-Objekt oder einer Zahl.
        Vec3-Objekte können verschachtelt sein (z.B. Vec3 enthält Vec3).
        """
        # Wenn es bereits eine Zahl ist, direkt zurückgeben
        if isinstance(value, (int, float)):
            return float(value)
        
        # Versuche als Tupel zu konvertieren
        try:
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value_tuple = tuple(value)
                if len(value_tuple) > 0:
                    # Rekursiv: nimm das erste Element
                    return SurfaceDataImporter._extract_float_from_vec3(value_tuple[0], default)
        except (TypeError, ValueError):
            pass
        
        # Versuche Index-Zugriff
        try:
            if hasattr(value, '__getitem__'):
                first_elem = value[0]
                return SurfaceDataImporter._extract_float_from_vec3(first_elem, default)
        except (IndexError, TypeError):
            pass
        
        # Versuche direkte Attribute
        if hasattr(value, 'x'):
            return SurfaceDataImporter._extract_float_from_vec3(value.x, default)
        
        # Fallback: versuche direkt zu konvertieren
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_polyline_points(self, entity) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        logger = logging.getLogger(__name__)
        try:
            # Debug: Prüfe Entity-Attribute
            logger.info("POLYLINE-Extraktion: Prüfe Entity-Attribute")
            logger.info("  hasattr(entity, 'vertices'): %s", hasattr(entity, 'vertices'))
            logger.info("  hasattr(entity, 'vertices_with_points'): %s", hasattr(entity, 'vertices_with_points'))
            logger.info("  hasattr(entity, 'flattening'): %s", hasattr(entity, 'flattening'))
            logger.info("  hasattr(entity, 'points'): %s", hasattr(entity, 'points'))
            
            if hasattr(entity, 'vertices'):
                try:
                    vertices_list = list(entity.vertices)
                    logger.info("  Anzahl Vertices: %d", len(vertices_list))
                    if len(vertices_list) > 0:
                        first_vertex = vertices_list[0]
                        logger.info("  Erster Vertex-Typ: %s", type(first_vertex).__name__)
                        logger.info("  Erster Vertex hat 'dxf': %s", hasattr(first_vertex, 'dxf'))
                        if hasattr(first_vertex, 'dxf'):
                            logger.info("  Erster Vertex.dxf hat 'location': %s", hasattr(first_vertex.dxf, 'location'))
                except Exception as e:
                    logger.info("  Fehler beim Zählen der Vertices: %s", e)
            
            # Methode 1: Verwende vertices_with_points() falls verfügbar (ezdxf 0.18+)
            if hasattr(entity, 'vertices_with_points'):
                try:
                    logger.info("  Versuche Methode 1: vertices_with_points()")
                    for vertex, point in entity.vertices_with_points():
                        x, y, z = point
                        points.append(self._make_point(float(x), float(y), float(z)))
                    logger.info("  Methode 1 erfolgreich: %d Punkte extrahiert", len(points))
                except Exception as e1:
                    logger.info("  Methode 1 fehlgeschlagen: %s", e1)
            
            # Methode 2: Direkt über vertices mit location
            if not points:
                try:
                    logger.info("  Versuche Methode 2: vertices mit location")
                    elevation = float(getattr(entity.dxf, 'elevation', 0.0))
                    logger.info("  Elevation: %f", elevation)
                    vertex_count = 0
                    for vertex in entity.vertices:
                        vertex_count += 1
                        # Versuche location-Attribut (ist ein Vec3-Objekt)
                        if hasattr(vertex.dxf, 'location'):
                            loc = vertex.dxf.location
                            # Verwende Hilfsfunktion für rekursive Vec3-Extraktion
                            try:
                                # Methode 1: Als Tuple konvertieren (sicherste Methode)
                                try:
                                    loc_tuple = tuple(loc)
                                    if len(loc_tuple) >= 2:
                                        x = self._extract_float_from_vec3(loc_tuple[0], 0.0)
                                        y = self._extract_float_from_vec3(loc_tuple[1], 0.0)
                                        z = self._extract_float_from_vec3(loc_tuple[2], elevation) if len(loc_tuple) >= 3 else elevation
                                        points.append(self._make_point(x, y, z))
                                except (TypeError, ValueError) as tuple_err:
                                    # Methode 2: Direkte Attribute (falls .x, .y, .z vorhanden)
                                    if hasattr(loc, 'x') and hasattr(loc, 'y'):
                                        x = self._extract_float_from_vec3(loc.x, 0.0)
                                        y = self._extract_float_from_vec3(loc.y, 0.0)
                                        z = self._extract_float_from_vec3(loc.z, elevation) if hasattr(loc, 'z') else elevation
                                        points.append(self._make_point(x, y, z))
                                    else:
                                        logger.info("    Vertex %d: location kann nicht verarbeitet werden (type: %s, repr: %s)", vertex_count, type(loc), repr(loc))
                            except Exception as loc_err:
                                logger.info("    Vertex %d: Fehler beim Extrahieren aus location: %s (loc type: %s, loc repr: %s)", vertex_count, loc_err, type(loc), repr(loc))
                        # Fallback: Versuche einzelne Attribute
                        elif hasattr(vertex, 'dxf'):
                            x = float(getattr(vertex.dxf, 'x', 0.0))
                            y = float(getattr(vertex.dxf, 'y', 0.0))
                            z = float(getattr(vertex.dxf, 'z', elevation))
                            points.append(self._make_point(x, y, z))
                    logger.info("  Methode 2: %d Vertices verarbeitet, %d Punkte extrahiert", vertex_count, len(points))
                except Exception as e2:
                    logger.info("  Methode 2 fehlgeschlagen: %s", e2)
            
            # Methode 3: Verwende flattening() falls verfügbar
            if not points and hasattr(entity, 'flattening'):
                try:
                    logger.info("  Versuche Methode 3: flattening()")
                    flattened = entity.flattening(0.01)  # Toleranz für Kurven
                    elevation = float(getattr(entity.dxf, 'elevation', 0.0))
                    for point in flattened:
                        try:
                            # Prüfe ob point ein Vec3-Objekt ist
                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                x = float(point.x)
                                y = float(point.y)
                                z = float(point.z) if hasattr(point, 'z') and point.z is not None else elevation
                                points.append(self._make_point(x, y, z))
                            # Oder ein Tuple/List
                            elif isinstance(point, (tuple, list)) and len(point) >= 2:
                                x = float(point[0])
                                y = float(point[1])
                                z = float(point[2]) if len(point) >= 3 else elevation
                                points.append(self._make_point(x, y, z))
                        except Exception as point_err:
                            logger.info("    Fehler beim Verarbeiten eines flattening-Punktes: %s", point_err)
                    logger.info("  Methode 3 erfolgreich: %d Punkte extrahiert", len(points))
                except Exception as e3:
                    logger.info("  Methode 3 fehlgeschlagen: %s", e3)
            
            # Methode 4: Versuche über points() Methode
            if not points and hasattr(entity, 'points'):
                try:
                    logger.info("  Versuche Methode 4: points()")
                    elevation = float(getattr(entity.dxf, 'elevation', 0.0))
                    for point in entity.points():
                        try:
                            # Verwende Hilfsfunktion für rekursive Vec3-Extraktion
                            # Methode 1: Als Tuple konvertieren (sicherste Methode)
                            try:
                                point_tuple = tuple(point)
                                if len(point_tuple) >= 2:
                                    x = self._extract_float_from_vec3(point_tuple[0], 0.0)
                                    y = self._extract_float_from_vec3(point_tuple[1], 0.0)
                                    z = self._extract_float_from_vec3(point_tuple[2], elevation) if len(point_tuple) >= 3 else elevation
                                    points.append(self._make_point(x, y, z))
                            except (TypeError, ValueError) as tuple_err:
                                # Methode 2: Direkte Attribute (falls .x, .y, .z vorhanden)
                                if hasattr(point, 'x') and hasattr(point, 'y'):
                                    x = self._extract_float_from_vec3(point.x, 0.0)
                                    y = self._extract_float_from_vec3(point.y, 0.0)
                                    z = self._extract_float_from_vec3(point.z, elevation) if hasattr(point, 'z') else elevation
                                    points.append(self._make_point(x, y, z))
                                # Methode 3: Index-Zugriff als Fallback
                                elif hasattr(point, '__getitem__'):
                                    try:
                                        x = self._extract_float_from_vec3(point[0], 0.0)
                                        y = self._extract_float_from_vec3(point[1], 0.0)
                                        z = self._extract_float_from_vec3(point[2], elevation) if len(point) >= 3 else elevation
                                        points.append(self._make_point(x, y, z))
                                    except (IndexError, TypeError):
                                        logger.info("    Punkt Index-Zugriff fehlgeschlagen: %s", type(point))
                                else:
                                    logger.info("    Unerwartetes Punkt-Format: %s", type(point))
                        except Exception as point_err:
                            logger.info("    Fehler beim Verarbeiten eines Punktes: %s (point type: %s)", point_err, type(point))
                    logger.info("  Methode 4 erfolgreich: %d Punkte extrahiert", len(points))
                except Exception as e4:
                    logger.info("  Methode 4 fehlgeschlagen: %s", e4)

            if not points:
                logger.warning(
                    "POLYLINE-Entity konnte nicht extrahiert werden - keine Punkte gefunden"
                )
            elif entity.is_closed and points and points[0] != points[-1]:
                points.append(points[0].copy())
                
        except Exception as e:
            logger.warning(
                "Fehler beim Extrahieren von POLYLINE-Punkten: %s",
                e
            )
            import traceback
            logger.info("Traceback: %s", traceback.format_exc())
        return points

    def _extract_3dface_points(self, entity) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        try:
            for x, y, z in entity.points():
                points.append(self._make_point(x, y, z))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Fehler beim Extrahieren von 3DFACE-Punkten: %s", e)
        return points

    def _extract_mesh_points(self, entity) -> List[Dict[str, float]]:
        """Extrahiert Punkte aus einem MESH oder POLYLINE-Mesh"""
        points: List[Dict[str, float]] = []
        logger = logging.getLogger(__name__)
        
        try:
            # Versuche verschiedene Methoden, um Mesh-Vertices zu extrahieren
            # Methode 1: Direkt über vertices
            if hasattr(entity, "vertices"):
                for vertex in entity.vertices:
                    try:
                        if hasattr(vertex, "dxf") and hasattr(vertex.dxf, "location"):
                            loc = vertex.dxf.location
                            x = float(loc.x) if hasattr(loc, 'x') else 0.0
                            y = float(loc.y) if hasattr(loc, 'y') else 0.0
                            z = float(loc.z) if hasattr(loc, 'z') else 0.0
                            points.append(self._make_point(x, y, z))
                        elif hasattr(vertex, "dxf"):
                            x = float(getattr(vertex.dxf, "x", 0.0))
                            y = float(getattr(vertex.dxf, "y", 0.0))
                            z = float(getattr(vertex.dxf, "z", 0.0))
                            points.append(self._make_point(x, y, z))
                    except Exception as e:
                        logger.debug("Fehler beim Extrahieren eines Mesh-Vertices: %s", e)
                        continue
            
            # Methode 2: Über get_mesh_vertex_cache falls vorhanden
            if not points and hasattr(entity, "get_mesh_vertex_cache"):
                try:
                    vertex_cache = entity.get_mesh_vertex_cache()
                    if vertex_cache:
                        for vertex in vertex_cache:
                            if hasattr(vertex, "location"):
                                loc = vertex.location
                                x = float(loc.x) if hasattr(loc, 'x') else 0.0
                                y = float(loc.y) if hasattr(loc, 'y') else 0.0
                                z = float(loc.z) if hasattr(loc, 'z') else 0.0
                                points.append(self._make_point(x, y, z))
                except Exception as e:
                    logger.debug("Fehler bei get_mesh_vertex_cache: %s", e)
            
            # Methode 3: Für POLYLINE-Mesh, versuche über faces
            if not points and hasattr(entity, "faces"):
                # Extrahiere eindeutige Vertices aus Faces
                vertex_set = set()
                try:
                    for face in entity.faces:
                        if hasattr(face, "indices"):
                            for idx in face.indices:
                                vertex_set.add(idx)
                        elif hasattr(face, "vertices"):
                            for v in face.vertices:
                                if hasattr(v, "location"):
                                    loc = v.location
                                    x = float(loc.x) if hasattr(loc, 'x') else 0.0
                                    y = float(loc.y) if hasattr(loc, 'y') else 0.0
                                    z = float(loc.z) if hasattr(loc, 'z') else 0.0
                                    points.append(self._make_point(x, y, z))
                except Exception as e:
                    logger.debug("Fehler beim Extrahieren von Mesh-Faces: %s", e)
            
            # Wenn wir Indices haben, aber noch keine Punkte, versuche über Block-Vertices
            if not points and hasattr(entity, "get_mesh_vertex_cache"):
                try:
                    # Versuche alternative Methode
                    if hasattr(entity, "mesh_vertex_cache"):
                        cache = entity.mesh_vertex_cache
                        if cache:
                            for vertex in cache:
                                if hasattr(vertex, "location"):
                                    loc = vertex.location
                                    x = float(loc.x) if hasattr(loc, 'x') else 0.0
                                    y = float(loc.y) if hasattr(loc, 'y') else 0.0
                                    z = float(loc.z) if hasattr(loc, 'z') else 0.0
                                    points.append(self._make_point(x, y, z))
                except Exception as e:
                    logger.debug("Fehler bei alternativer Mesh-Extraktion: %s", e)
            
            if not points:
                logger.warning(
                    "MESH-Entity konnte nicht extrahiert werden - keine Vertices gefunden"
                )
                
        except Exception as e:
            logger.warning("Fehler beim Extrahieren von MESH-Punkten: %s", e)
        
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
            if "/" in key:
                group_id = self.group_manager.ensure_group_path(key)
            else:
                # Suche zuerst nach existierender Gruppe mit diesem Namen
                existing_group = self.group_manager.find_surface_group_by_name(key)
                if existing_group:
                    group_id = existing_group.group_id
                else:
                    # Erstelle neue Gruppe mit Tag-Namen
                    new_group = self.group_manager.create_surface_group(key)
                    if new_group:
                        group_id = new_group.group_id

        self._group_cache[key] = group_id
        return group_id

    def _build_group_label(self, block_stack: tuple) -> Optional[str]:
        if not block_stack:
            return None
        segments: List[str] = []
        for ctx in block_stack:
            segment = self._format_block_segment(ctx)
            if segment:
                segments.append(segment)
        if not segments:
            return None
        return "/".join(segments)

    def _format_block_segment(self, ctx: InsertContext) -> str:
        base = (ctx.name or "Block").strip()
        base = base.replace("/", "_").replace("\\", "_")
        handle_part = ctx.handle or ""
        if handle_part:
            return f"{base}#{handle_part}"
        return base

    @staticmethod
    def _resolve_tag_from_context(block_stack: tuple) -> Optional[str]:
        for ctx in reversed(block_stack):
            if ctx.tag:
                return ctx.tag
        return None

    # ---- Tag/Attribut-Extraktion -------------------------------------

    def _analyze_dxf_tags(self, doc, msp) -> None:
        """Analysiert die DXF-Datei auf Tags/Attribute"""
        logger = logging.getLogger(__name__)
        tags_found = set()
        
        # Analysiere INSERT-Entities im Modelspace, um Block-Tag-Mappings zu erstellen
        for entity in msp:
            if entity.dxftype() == "INSERT":
                block_name = getattr(entity.dxf, "name", None)
                if block_name:
                    tag = self._extract_tag_from_insert(entity, doc)
                    if tag:
                        self._block_tag_map[block_name] = tag
                        tags_found.add(tag)
        
        # Analysiere alle Entities im Modelspace und in Blöcken
        for entity, _, _ in self._iter_surface_entities(msp):
            tag = self._extract_tag_from_entity(entity, doc)
            if tag:
                tags_found.add(tag)
        
        # Analysiere auch Blöcke direkt
        blocks = getattr(doc, "blocks", None)
        if blocks:
            for block in blocks:
                if block.name.startswith("*"):  # System-Blöcke überspringen
                    continue
                
                # Prüfe Block-Namen
                block_name_upper = block.name.upper()
                if block_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                    self._block_tag_map[block.name] = block_name_upper
                    tags_found.add(block_name_upper)
                
                # Prüfe Entities im Block
                for entity in block:
                    tag = self._extract_tag_from_entity(entity, doc)
                    if tag:
                        tags_found.add(tag)
        
        if tags_found:
            logger.info(
                "DXF-Tags gefunden: %s",
                ", ".join(sorted(tags_found))
            )
        if self._block_tag_map:
            logger.info(
                "DXF-Block-Tag-Mappings: %s",
                ", ".join(f"{k} -> {v}" for k, v in self._block_tag_map.items())
            )

    def _extract_tag_from_entity(self, entity, doc) -> Optional[str]:
        """Extrahiert Tag/Attribut aus einer DXF-Entity"""
        logger = logging.getLogger(__name__)
        
        # Methode 1: Prüfe Layer-Name (wenn Layer "WOOD" oder "BETON" heißt)
        layer_name = getattr(entity.dxf, "layer", None)
        if layer_name:
            layer_name_upper = layer_name.upper()
            # Prüfe ob Layer-Name einem bekannten Tag entspricht
            if layer_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                return layer_name_upper
        
        # Methode 1b: Prüfe Block-Name (wenn Entity Teil eines Blocks ist)
        # Dies wird in _expand_insert behandelt, aber wir können auch hier prüfen
        if hasattr(entity, "dxf") and hasattr(entity.dxf, "name"):
            block_name = entity.dxf.name
            if block_name:
                block_name_upper = block_name.upper()
                if block_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                    return block_name_upper
        
        # Methode 2: Prüfe Block-Attribute (für INSERT-Entities)
        if entity.dxftype() == "INSERT":
            tag = self._extract_tag_from_insert(entity, doc)
            if tag:
                return tag
        
        # Methode 3: Prüfe XDATA (Extended Data)
        tag = self._extract_tag_from_xdata(entity)
        if tag:
            return tag
        
        # Methode 4: Prüfe AppData
        tag = self._extract_tag_from_appdata(entity)
        if tag:
            return tag
        
        # Methode 5: Prüfe Objektdaten
        tag = self._extract_tag_from_object_data(entity, doc)
        if tag:
            return tag
        
        return None

    def _extract_tag_from_insert(self, insert_entity, doc) -> Optional[str]:
        """Extrahiert Tag aus INSERT-Entity über Block-Attribute"""
        try:
            block_name = getattr(insert_entity.dxf, "name", None)
            if not block_name:
                return None
            
            # Prüfe ob Block-Name selbst ein Tag ist
            block_name_upper = block_name.upper()
            if block_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                return block_name_upper
            
            # Suche Block-Definition
            blocks = getattr(doc, "blocks", None)
            if not blocks:
                return None
            
            block = blocks.get(block_name)
            if not block:
                return None
            
            # Suche nach ATTRIB-Entities im Block
            for entity in block:
                if entity.dxftype() == "ATTRIB":
                    tag = getattr(entity.dxf, "text", None)
                    if tag and tag.strip():
                        # Prüfe ob es ein bekannter Tag ist
                        tag_upper = tag.strip().upper()
                        if tag_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                            return tag_upper
                        # Oder verwende den Tag-Text direkt, wenn er aussagekräftig ist
                        if len(tag.strip()) > 0 and not tag.strip().isdigit():
                            return tag.strip()
            
            # Prüfe Layer-Namen der Entities im Block
            for entity in block:
                layer_name = getattr(entity.dxf, "layer", None)
                if layer_name:
                    layer_name_upper = layer_name.upper()
                    if layer_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                        return layer_name_upper
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug("Fehler beim Extrahieren von Tag aus INSERT: %s", e)
        
        return None

    def _extract_tag_from_xdata(self, entity) -> Optional[str]:
        """Extrahiert Tag aus XDATA (Extended Data)"""
        try:
            if not hasattr(entity, "xdata"):
                return None
            
            xdata = entity.xdata
            if not xdata:
                return None
            
            # Suche nach Tag in XDATA
            for app_name, data in xdata.items():
                if isinstance(data, (list, tuple)):
                    for item in data:
                        if isinstance(item, str):
                            item_upper = item.upper()
                            if item_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                                return item_upper
        except Exception:
            pass
        
        return None

    def _extract_tag_from_appdata(self, entity) -> Optional[str]:
        """Extrahiert Tag aus AppData"""
        try:
            if not hasattr(entity, "appdata"):
                return None
            
            appdata = entity.appdata
            if not appdata:
                return None
            
            # Suche nach Tag in AppData
            for app_name, data in appdata.items():
                if isinstance(data, (list, tuple)):
                    for item in data:
                        if isinstance(item, str):
                            item_upper = item.upper()
                            if item_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                                return item_upper
        except Exception:
            pass
        
        return None

    def _extract_tag_from_object_data(self, entity, doc) -> Optional[str]:
        """Extrahiert Tag aus Objektdaten"""
        try:
            # Prüfe ob Entity Objektdaten hat
            if hasattr(entity, "get_xdata"):
                xdata = entity.get_xdata("ACAD")
                if xdata:
                    for item in xdata:
                        if isinstance(item, str):
                            item_upper = item.upper()
                            if item_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                                return item_upper
        except Exception:
            pass
        
        return None

    # ---- Helpers für DXF-Entities ------------------------------------

    def _is_entity_closed(self, entity, points: List[Dict[str, float]]) -> bool:
        """Prüft ob eine Entity geschlossen ist"""
        dxftype = entity.dxftype()
        
        # Für LWPOLYLINE: Prüfe verschiedene Attribute
        if dxftype == "LWPOLYLINE":
            if hasattr(entity, "closed"):
                return bool(entity.closed)
            if hasattr(entity.dxf, "flags"):
                # Bit 1 = closed flag
                flags = getattr(entity.dxf, "flags", 0)
                return bool(flags & 1)
        
        # Für POLYLINE: Prüfe is_closed Attribut
        if dxftype == "POLYLINE":
            if hasattr(entity, "is_closed"):
                return bool(entity.is_closed)
            if hasattr(entity, "closed"):
                return bool(entity.closed)
        
        # Prüfe ob erster und letzter Punkt identisch sind (mit Toleranz)
        if len(points) >= 3:
            first = points[0]
            last = points[-1]
            tolerance = 1e-6
            dx = abs(first.get("x", 0) - last.get("x", 0))
            dy = abs(first.get("y", 0) - last.get("y", 0))
            dz = abs(first.get("z", 0) - last.get("z", 0))
            return dx < tolerance and dy < tolerance and dz < tolerance
        
        return False

    def _extract_points_from_entity(self, entity, transform) -> List[Dict[str, float]] | None:
        dxftype = entity.dxftype()
        points = None
        
        if dxftype == "LWPOLYLINE":
            points = self._extract_lwpolyline_points(entity)
        elif dxftype == "POLYLINE":
            # Prüfe ob es ein Mesh ist
            is_mesh = False
            if hasattr(entity, "is_mesh"):
                is_mesh = bool(entity.is_mesh)
            elif hasattr(entity.dxf, "flags"):
                # Bit 16 = mesh flag
                flags = getattr(entity.dxf, "flags", 0)
                is_mesh = bool(flags & 16)
            
            if is_mesh:
                points = self._extract_mesh_points(entity)
            else:
                points = self._extract_polyline_points(entity)
        elif dxftype == "3DFACE":
            points = self._extract_3dface_points(entity)
        elif dxftype == "LINE":
            points = self._extract_line_points(entity)
        elif dxftype == "MESH":
            points = self._extract_mesh_points(entity)
        
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

    def _iter_surface_entities(self, layout, block_stack: Optional[tuple] = None):
        current_stack = tuple() if block_stack is None else block_stack
        for entity in layout:
            yield from self._resolve_entity(
                entity,
                depth=0,
                transform=None,
                block_stack=current_stack,
            )
    
    def _get_identity_transform(self):
        """Gibt eine Identitäts-Transformation zurück"""
        return {
            "translation": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
            "rotation": 0.0,
        }

    def _resolve_entity(self, entity, depth: int, transform, block_stack: tuple):
        if entity.dxftype() == "INSERT":
            yield from self._expand_insert(entity, depth, transform, block_stack)
        else:
            yield (entity, transform, block_stack)

    def _expand_insert(self, insert_entity, depth: int, parent_transform, block_stack: tuple):
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
        doc = getattr(insert_entity, "doc", None)

        # Extrahiere Tag aus INSERT-Entity und speichere Mapping
        block_tag = self._block_tag_map.get(block_name)
        if doc:
            try:
                tag_from_insert = self._extract_tag_from_insert(insert_entity, doc)
            except Exception:  # pragma: no cover - defensive
                tag_from_insert = None
            if tag_from_insert:
                block_tag = tag_from_insert
                self._block_tag_map[block_name] = tag_from_insert
        
        # Extrahiere Transformationen aus der INSERT-Entity
        insert_transform = self._extract_insert_transform(insert_entity)
        
        # Kombiniere mit Parent-Transformation
        if parent_transform:
            combined_transform = self._combine_transforms(parent_transform, insert_transform)
        else:
            combined_transform = insert_transform
        
        try:
            virtual_entities = list(insert_entity.virtual_entities())
            entity_count = len(virtual_entities)
            entity_types = {}
            for ve in virtual_entities:
                ve_type = ve.dxftype()
                entity_types[ve_type] = entity_types.get(ve_type, 0) + 1
            
            type_summary = ", ".join(f"{k}:{v}" for k, v in entity_types.items())
            logger.info(
                "INSERT '%s' (Tiefe %d) aufgelöst: %d Entities extrahiert [%s]",
                block_name,
                depth,
                entity_count,
                type_summary
            )
            if entity_count == 0:
                logger.warning(
                    "INSERT '%s' (Tiefe %d) enthält keine Entities",
                    block_name,
                    depth,
                )
        except Exception as exc:
            logger.warning(
                "INSERT '%s' konnte nicht expandiert werden: %s",
                block_name,
                exc,
            )
            return

        # Notiere Kontext für nachgelagerte Entities
        self._block_instance_counters[block_name] += 1
        context = InsertContext(
            name=block_name,
            handle=getattr(insert_entity.dxf, "handle", None),
            instance_index=self._block_instance_counters[block_name],
            tag=block_tag,
            layer=getattr(insert_entity.dxf, "layer", None),
        )
        new_stack = block_stack + (context,)

        try:
            logger = logging.getLogger(__name__)
            for idx, virtual in enumerate(virtual_entities):
                vtype = virtual.dxftype()
                vlayer = getattr(virtual.dxf, "layer", "unbekannt")
                logger.debug(
                    "  Block '%s' Entity %d/%d: %s auf Layer '%s'",
                    block_name, idx+1, len(virtual_entities), vtype, vlayer
                )
                yield from self._resolve_entity(
                    virtual,
                    depth + 1,
                    combined_transform,
                    new_stack,
                )
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

