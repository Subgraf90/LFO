from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
from Module_LFO.Modules_Data.SurfaceValidator import (
    validate_surface_geometry,
    triangulate_points,
    _remove_redundant_points,
)


def _strip_closing_point(points: List[Dict[str, float]], tol: float = 1e-6) -> List[Dict[str, float]]:
    """Entfernt den letzten Punkt, falls er (nahezu) identisch mit dem ersten ist."""
    if len(points) < 2:
        return points
    first = points[0]
    last = points[-1]
    dx = (last.get("x", 0.0) - first.get("x", 0.0))
    dy = (last.get("y", 0.0) - first.get("y", 0.0))
    dz = (last.get("z", 0.0) - first.get("z", 0.0))
    if dx * dx + dy * dy + dz * dz <= tol * tol:
        return points[:-1]
    return points


def _fan_triangulate(points: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    """
    Einfache Fan-Triangulation (setzt voraus, dass Punkte in sinnvoller Reihenfolge vorliegen).
    Gibt Liste von Dreiecks-Punktlisten zurück.
    """
    if len(points) < 3:
        return []
    tris: List[List[Dict[str, float]]] = []
    anchor = points[0]
    for i in range(1, len(points) - 1):
        tris.append([anchor, points[i], points[i + 1]])
    return tris


def _make_unique_surface_id(existing: Dict[str, SurfaceDefinition], base_id: str) -> str:
    """Erzeuge eindeutige Surface-ID, falls bereits vorhanden."""
    if base_id not in existing:
        return base_id
    counter = 1
    while True:
        candidate = f"{base_id}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


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

    FILE_FILTER = "DXF-Dateien (*.dxf);;Surface-Text (*.txt)"

    def __init__(self, parent_widget, settings, container, group_manager=None, main_window=None):
        self.parent_widget = parent_widget
        self.settings = settings
        self._container = container  # Reserviert für künftige Integrationen
        self.group_manager = group_manager
        self.main_window = main_window  # Für Fortschrittsanzeige
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
            "Surface-Datei importieren",
            "",
            self.FILE_FILTER,
        )

        if not file_path:
            return None

        path = Path(file_path)
        suffix = path.suffix.lower()

        # Verwende Fortschrittsanzeige, falls main_window verfügbar ist
        if self.main_window and hasattr(self.main_window, 'run_tasks_with_progress'):
            try:
                return self._execute_with_progress(path, suffix)
            except Exception as exc:
                QMessageBox.warning(
                    self.parent_widget,
                    "Surface-Import fehlgeschlagen",
                    f"Fehler beim Importieren:\n{exc}",
                )
                return False
        else:
            # Fallback ohne Fortschrittsanzeige (für Kompatibilität)
            return self._execute_without_progress(path, suffix)

    def _execute_with_progress(self, path: Path, suffix: str) -> bool | None:
        """Führt den Import mit Fortschrittsanzeige durch"""
        from Module_LFO.Modules_Init.Progress import ProgressCancelled
        
        surfaces = {}
        imported_count = 0
        
        try:
            # ZUERST: Alle Dialoge abarbeiten, BEVOR die Fortschrittsanzeige startet
            should_clear = False
            
            if suffix == ".dxf":
                # Prüfe zuerst, ob ezdxf verfügbar ist (ohne Dialog)
                try:
                    import importlib
                    importlib.import_module("ezdxf")
                except ImportError:
                    QMessageBox.warning(
                        self.parent_widget,
                        "DXF-Import nicht möglich",
                        "Für den Import von DXF-Dateien wird das Paket 'ezdxf' benötigt.\n"
                        "Bitte installieren Sie es z. B. mit 'pip install ezdxf'.",
                    )
                    return False
                
                # Lade DXF-Datei VOR dem Dialog (um zu wissen, ob Surfaces vorhanden sind)
                try:
                    surfaces = self._load_dxf_surfaces(path)
                except Exception as exc:
                    QMessageBox.warning(
                        self.parent_widget,
                        "DXF-Import fehlgeschlagen",
                        f"Die DXF-Datei konnte nicht verarbeitet werden:\n{exc}",
                    )
                    return False
                
                # Jetzt Dialog anzeigen (wenn Surfaces vorhanden)
                if surfaces and self._ask_clear_existing_surfaces():
                    should_clear = True
                    
            elif suffix == ".txt":
                # Dialog VOR dem Laden anzeigen
                if self._ask_clear_existing_surfaces():
                    should_clear = True
                    
            # Lösche existierende Surfaces, falls gewünscht (VOR Fortschrittsanzeige)
            if should_clear:
                self._clear_existing_surfaces()
            
            # JETZT: Fortschrittsanzeige starten für die eigentlichen Import-Operationen
            tasks = []
            
            if suffix == ".dxf":
                # Surfaces wurden bereits geladen (ohne Fortschrittsanzeige, um Dialog zu zeigen)
                # Jetzt nur noch speichern mit Fortschrittsanzeige
                pass  # Speichern wird als Task hinzugefügt
                
            elif suffix == ".txt":
                # Task: TXT-Datei laden
                def load_txt_task():
                    nonlocal surfaces
                    try:
                        surfaces = self._load_txt_surfaces(path)
                    except ValueError as exc:
                        QMessageBox.warning(
                            self.parent_widget,
                            "Surface-Import fehlgeschlagen",
                            f"Die TXT-Datei konnte nicht verarbeitet werden:\n{exc}",
                        )
                        raise
                
                tasks.append(("TXT-Datei laden", load_txt_task))
                
            else:
                # Task: JSON/andere Datei laden
                def load_payload_task():
                    nonlocal surfaces
                    try:
                        payload = self._load_payload(path)
                    except (OSError, ValueError) as exc:
                        QMessageBox.warning(
                            self.parent_widget,
                            "Surface-Import fehlgeschlagen",
                            f"Die Datei konnte nicht geladen werden:\n{exc}",
                        )
                        raise
                    surfaces = self._parse_surfaces(payload)
                
                tasks.append(("Datei laden", load_payload_task))
            
            # Task: Surfaces speichern
            def store_task():
                nonlocal imported_count
                if not surfaces:
                    QMessageBox.information(
                        self.parent_widget,
                        "No valid surfaces",
                        "The selected file does not contain any valid surface definitions.",
                    )
                    raise ValueError("No valid surfaces found")
                # Prüfe auf Abbruch vor dem Speichern
                if hasattr(self.main_window, '_current_progress_session') and self.main_window._current_progress_session:
                    if self.main_window._current_progress_session.is_cancelled():
                        raise ProgressCancelled("Import cancelled by user")
                imported_count = self._store_surfaces(surfaces)
            
            tasks.append(("Surfaces speichern", store_task))
            
            # Führe Tasks mit Fortschrittsanzeige aus
            # (tasks enthält immer mindestens "Surfaces speichern")
            self.main_window.run_tasks_with_progress("Surface-Daten importieren", tasks)
            
            # Erfolgsmeldung
            logger = logging.getLogger(__name__)
            msg = f"Surface-Import ({suffix}): {imported_count} Surfaces erfolgreich importiert"
            print(msg)
            if logger.isEnabledFor(logging.INFO):
                logger.info(msg)
            
            QMessageBox.information(
                self.parent_widget,
                "Surface Import Successful",
                f"{imported_count} surface(s) imported successfully.",
            )
            return True
            
        except ProgressCancelled:
            # Benutzer hat abgebrochen
            return None
        except Exception as exc:
            # Fehler wurde bereits in den Tasks behandelt
            raise

    def _execute_without_progress(self, path: Path, suffix: str) -> bool | None:
        """Führt den Import ohne Fortschrittsanzeige durch (Fallback)"""
        import time
        t_start = time.perf_counter()

        if suffix == ".dxf":
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
            t_load = time.perf_counter()
        elif suffix == ".txt":
            # WICHTIG: Lösche existierende Surfaces VOR dem Laden, damit Gruppen nicht gelöscht werden
            # (Gruppen werden während des Ladens erstellt)
            if self._ask_clear_existing_surfaces():
                self._clear_existing_surfaces()
            try:
                surfaces = self._load_txt_surfaces(path)
            except ValueError as exc:
                QMessageBox.warning(
                    self.parent_widget,
                    "Surface-Import fehlgeschlagen",
                    f"Die TXT-Datei konnte nicht verarbeitet werden:\n{exc}",
                )
                return False
            else:
                t_load = time.perf_counter()
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
            t_load = time.perf_counter()
        if not surfaces:
            QMessageBox.information(
                self.parent_widget,
                "No valid surfaces",
                "The selected file does not contain any valid surface definitions.",
            )
            return False

        imported_count = self._store_surfaces(surfaces)
        t_store = time.perf_counter()

        # Grobes Performance-Logging (nur eine Zeile, kein Spam)
        logger = logging.getLogger(__name__)
        msg = (
            f"Surface-Import ({suffix}): {imported_count} Surfaces, "
            f"Ladezeit={t_load - t_start:.3f}s, "
            f"Speichern={t_store - t_load:.3f}s, "
            f"Gesamt={t_store - t_start:.3f}s"
        )
        # Immer auf stdout ausgeben, damit es im Terminal sichtbar ist
        print(msg)
        # Zusätzlich über logging, falls konfiguriert
        if logger.isEnabledFor(logging.INFO):
            logger.info(msg)
        QMessageBox.information(
            self.parent_widget,
            "Surface Import Successful",
            f"{imported_count} surface(s) imported successfully.",
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

    def _load_txt_surfaces(self, file_path: Path) -> Dict[str, SurfaceDefinition]:
        """
        Lädt Flächen aus einem SketchUp-ähnlichen TXT-Export:
        - Abschnitte starten mit einer Label-Zeile ("Label","BETON 1")
        - Danach folgen Koordinatenzeilen (x,y,z)
        - Einfache Zeile mit ';' oder '";"' beendet den Abschnitt
        - Alle Flächen mit gleichem Labelpräfix (z.B. BETON) werden zu einer Gruppe zusammengefasst
        """
        surfaces: Dict[str, SurfaceDefinition] = {}
        current_label: Optional[str] = None
        current_points: List[Dict[str, float]] = []
        id_counters: Dict[str, int] = {}

        def finalize_current_surface():
            nonlocal current_label, current_points
            if not current_label:
                current_points = []
                return
            if len(current_points) < 3:
                current_label = None
                current_points = []
                return

            # Stelle sicher, dass Polygon geschlossen ist
            first = current_points[0]
            last = current_points[-1]
            if first != last:
                current_points.append(first.copy())

            safe_label = self._make_safe_identifier(current_label) or "surface"
            id_counters[safe_label] = id_counters.get(safe_label, 0) + 1
            suffix = f"_{id_counters[safe_label]}" if id_counters[safe_label] > 1 else ""
            surface_id = f"{safe_label}{suffix}"

            group_label = self._derive_group_label(current_label)
            group_id = self._ensure_group_for_label(group_label)

            payload = {
                "name": current_label,
                "enabled": False,
                "hidden": False,
                "locked": False,
                "points": list(current_points),
            }
            if group_id:
                payload["group_id"] = group_id

            surfaces[surface_id] = SurfaceDefinition.from_dict(surface_id, payload)
            current_label = None
            current_points = []

        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                # Einmalig trimmen, um führende/nachfolgende Whitespaces zu entfernen
                line = raw_line.strip()
                if not line:
                    continue

                # Abschnittsende-Zeilen wie ';' oder '";"' erkennen
                first_char = line[0]
                if first_char == ";" or line.startswith('";'):
                    if line in ('";"', '" ;"', ";", '";', '"'):
                        finalize_current_surface()
                    continue

                # Label-Zeilen erkennen (nur einmal tolower ausführen)
                lower_line = line.lower()
                if lower_line.startswith('"label"') or lower_line.startswith("label"):
                    finalize_current_surface()
                    label = self._extract_label_value(line, line_number)
                    current_label = label
                    current_points = []
                    continue

                # Koordinatenzeilen direkt hier parsen (spart Funktions-Overhead)
                try:
                    parts = line.split(",")
                    if len(parts) < 3:
                        raise ValueError
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                except ValueError:
                    raise ValueError(f"Ungültige Koordinaten in Zeile {line_number}: {line}")

                current_points.append(self._make_point(x, y, z))

        finalize_current_surface()
        return surfaces

    def _extract_label_value(self, line: str, line_number: int) -> str:
        try:
            _, label_part = line.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Ungültige Label-Zeile (Zeile {line_number}): {line}") from exc
        label = label_part.strip().strip('"').strip()
        if not label:
            raise ValueError(f"Leeres Label in Zeile {line_number}")
        return label

    def _parse_txt_point(self, line: str, line_number: int) -> Dict[str, float]:
        """
        Beibehaltung der bisherigen API für eventuelle zukünftige Nutzung.
        Aktuell wird im TXT-Importer direkt geparst, um Funktions-Overhead zu sparen.
        """
        try:
            parts = line.split(",")
            if len(parts) < 3:
                raise ValueError
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except ValueError:
            raise ValueError(f"Ungültige Koordinaten in Zeile {line_number}: {line}")
        return self._make_point(x, y, z)

    @staticmethod
    def _make_safe_identifier(label: str) -> str:
        text = label.strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^0-9A-Za-z_]+", "", text)
        return text.lower()

    @staticmethod
    def _derive_group_label(label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        label = label.strip()
        
        # Strategie: Extrahiere den gemeinsamen Überbegriff durch Entfernen von Zahlen am Ende
        # Beispiele:
        # "Stage 156" -> "Stage"
        # "test 1" -> "test"
        # "03_Gebaeude 01" -> "03_Gebaeude"
        # "04_Stage 02" -> "04_Stage"
        # "06_GP 0001" -> "06_GP"
        # "LP 01" -> "LP"
        # "BETON 1" -> "BETON"
        
        # Finde das letzte Vorkommen von Leerzeichen gefolgt von Zahlen am Ende
        # und entferne alles ab diesem Punkt
        # Pattern: Alles bis zum letzten Leerzeichen + Zahlen am Ende
        match = re.search(r"\s+\d+.*$", label)
        if match:
            # Entferne alles ab dem Leerzeichen vor den Zahlen
            base = label[:match.start()].strip()
            if base:
                return base
        
        # Fallback: Wenn keine Zahlen am Ende gefunden, verwende Original
        return label

    def _store_surfaces(self, surfaces: Dict[str, SurfaceDefinition]) -> int:
        from Module_LFO.Modules_Init.Progress import ProgressCancelled
        
        imported = 0
        for surface_id, surface in surfaces.items():
            # Prüfe auf Abbruch während der Schleife
            if self.main_window and hasattr(self.main_window, '_current_progress_session') and self.main_window._current_progress_session:
                if self.main_window._current_progress_session.is_cancelled():
                    raise ProgressCancelled("Import cancelled by user")
            
            # Abschließenden redundanten Punkt entfernen (Polygon wird automatisch geschlossen)
            surface.points = _strip_closing_point(surface.points)
            
            # Nur validieren (keine Korrektur) beim Import
            try:
                validation_result = validate_surface_geometry(
                    surface,
                    round_to_cm=False,
                    remove_redundant=True,
                )
                logger = logging.getLogger(__name__)
                if validation_result.invalid_fields:
                    outlier_info = ", ".join(
                        f"{idx}:{axis}" for idx, axis in validation_result.invalid_fields
                    )
                else:
                    outlier_info = "none"
                logger.info(
                    f"[Import Validation] Surface '{surface.name}' ({surface_id}) "
                    f"valid={validation_result.is_valid}, outliers={outlier_info}, "
                    f"message={validation_result.error_message or 'OK'}"
                )
                if not validation_result.is_valid:
                    logger.warning(
                        f"Surface '{surface.name}' ({surface_id}) ist ungültig beim Import: "
                        f"{validation_result.error_message}"
                    )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Fehler bei der Validierung von Surface '{surface.name}' ({surface_id}): {e}",
                    exc_info=True,
                )
                # Bei Fehler: Surface unverändert übernehmen

            # Stelle sicher, dass imported Surfaces immer enabled=False haben
            surface.enabled = False
            
            # Triangulation pro Surface (nicht pro Gruppe) - wie beim TXT-Import
            # Surfaces mit >3 Punkten werden trianguliert, damit nur noch Surfaces mit 3 Punkten resultieren
            target_surfaces: List[Tuple[str, SurfaceDefinition]] = []
            
            # WICHTIG: Entferne redundante/doppelte Punkte VOR der Triangulation
            # Dies verhindert fehlerhafte Triangulation durch degenerierte Punkte
            points_for_triangulation = list(surface.points)
            points_for_triangulation, removed_count = _remove_redundant_points(points_for_triangulation)
            if removed_count > 0:
                logger.debug(
                    "[Import] Surface '%s' (%s): %d redundante Punkt(e) entfernt vor Triangulation (%d → %d)",
                    surface.name, surface_id, removed_count, len(surface.points), len(points_for_triangulation)
                )
            
            triangulate_needed = len(points_for_triangulation) > 3  # Triangulation für Surfaces mit >3 Punkten
            
            # #region agent log - TRIANGULATION: Original-Punkte vor Triangulation
            import json, time
            try:
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TRI","location":"SurfaceDataImporter.py:_store_surfaces:before_triangulation","message":"BEFORE triangulation","data":{"surface_id":surface_id,"surface_name":surface.name,"points_count":len(surface.points),"points_after_redundant_removal":len(points_for_triangulation),"removed_redundant_count":removed_count,"triangulate_needed":triangulate_needed,"original_points":surface.points,"cleaned_points":points_for_triangulation},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            
            if triangulate_needed:
                tris = triangulate_points(points_for_triangulation)
                logger = logging.getLogger(__name__)
                logger.info(
                    "[Import Triangulation] Surface '%s' (%s): Punkte=%d, needed=%s, tris=%d",
                    surface.name,
                    surface_id,
                    len(surface.points),
                    triangulate_needed,
                    len(tris),
                )
                
                # #region agent log - TRIANGULATION: Triangulierte Dreiecke
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TRI","location":"SurfaceDataImporter.py:_store_surfaces:triangulation_result","message":"Triangulation result","data":{"surface_id":surface_id,"surface_name":surface.name,"original_points_count":len(surface.points),"triangles_count":len(tris),"triangles":[[{"x":p.get("x"),"y":p.get("y"),"z":p.get("z")} for p in tri] for tri in tris]},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
                
                base_name = surface.name or surface_id
                if tris:
                    for idx, tri_pts in enumerate(tris):
                        # #region agent log - TRIANGULATION: Einzelnes Dreieck
                        try:
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TRI","location":"SurfaceDataImporter.py:_store_surfaces:triangle_created","message":"Triangle created","data":{"surface_id":surface_id,"triangle_index":idx,"triangle_points":tri_pts,"points_count":len(tri_pts)},"timestamp":int(time.time()*1000)})+"\n")
                        except Exception:
                            pass
                        # #endregion
                        tri_id = f"{surface_id}_tri{idx+1}"
                        tri_surface = SurfaceDefinition(
                            surface_id=tri_id,
                            name=f"{base_name} (Tri {idx+1})",
                            points=tri_pts,
                            color=surface.color,
                            enabled=False,
                            hidden=surface.hidden,
                            locked=surface.locked,
                            group_id=surface.group_id,  # Gruppe wird beibehalten
                        )
                        target_surfaces.append((tri_id, tri_surface))
                else:
                    # Triangulation fehlgeschlagen -> Original übernehmen
                    logger.warning(
                        "[Import Triangulation] Surface '%s' (%s): Triangulation fehlgeschlagen, Original übernommen",
                        surface.name,
                        surface_id,
                    )
                    target_surfaces.append((surface_id, surface))
            else:
                # Surface hat bereits ≤3 Punkte -> direkt übernehmen
                # Verwende bereinigte Punkte (falls redundante entfernt wurden)
                if removed_count > 0:
                    surface_cleaned = SurfaceDefinition(
                        surface_id=surface_id,
                        name=surface.name,
                        points=points_for_triangulation,
                        color=surface.color,
                        enabled=surface.enabled,
                        locked=surface.locked,
                        group_id=surface.group_id,
                    )
                    target_surfaces.append((surface_id, surface_cleaned))
                else:
                    target_surfaces.append((surface_id, surface))

            if not hasattr(self.settings, "surface_definitions") or not isinstance(
                self.settings.surface_definitions, dict
            ):
                self.settings.surface_definitions = {}

            for sid, sdef in target_surfaces:
                sid_unique = _make_unique_surface_id(self.settings.surface_definitions, sid)
                # #region agent log - HYPOTHESIS A: Surface wird gespeichert
                import json, time
                try:
                    group_id_before = getattr(sdef, "group_id", None)
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"SurfaceDataImporter.py:_store_surfaces:before_store","message":"BEFORE storing surface","data":{"surface_id":sid_unique,"original_id":sid,"group_id":str(group_id_before) if group_id_before else None,"has_group_manager":self.group_manager is not None},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
                if hasattr(self.settings, "add_surface_definition"):
                    self.settings.add_surface_definition(sid_unique, sdef, make_active=False)
                else:
                    self.settings.surface_definitions[sid_unique] = sdef
                imported += 1
                # #region agent log - HYPOTHESIS A: Surface wurde gespeichert - FINALE DATEN
                try:
                    stored_check = sid_unique in getattr(self.settings, 'surface_definitions', {})
                    stored_surface = getattr(self.settings, 'surface_definitions', {}).get(sid_unique)
                    stored_dict = None
                    stored_points = None
                    if stored_surface:
                        if hasattr(stored_surface, "to_dict"):
                            stored_dict = stored_surface.to_dict()
                        elif isinstance(stored_surface, dict):
                            stored_dict = stored_surface
                        if stored_dict:
                            stored_points = stored_dict.get("points", [])
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STORED_DATA","location":"SurfaceDataImporter.py:_store_surfaces:after_store","message":"AFTER storing surface - FINAL DATA","data":{"surface_id":sid_unique,"stored_in_settings":stored_check,"total_surfaces":len(getattr(self.settings, 'surface_definitions', {})),"stored_dict_keys":list(stored_dict.keys()) if stored_dict else None,"stored_points_count":len(stored_points) if stored_points else 0,"stored_points":stored_points,"stored_dict":stored_dict},"timestamp":int(time.time()*1000)})+"\n")
                except Exception as e:
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STORED_DATA","location":"SurfaceDataImporter.py:_store_surfaces:after_store","message":"AFTER storing surface - ERROR","data":{"surface_id":sid_unique,"error":str(e)},"timestamp":int(time.time()*1000)})+"\n")
                    except:
                        pass
                # #endregion

                # Weise Surface direkt einer Gruppe zu (inkl. Erstellung)
                if self.group_manager:
                    group_id = getattr(sdef, "group_id", None)
                    # #region agent log - HYPOTHESIS B: Gruppen-Zuordnung
                    try:
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"SurfaceDataImporter.py:_store_surfaces:before_assign","message":"BEFORE assigning to group","data":{"surface_id":sid_unique,"group_id":str(group_id) if group_id else None},"timestamp":int(time.time()*1000)})+"\n")
                    except Exception:
                        pass
                    # #endregion
                    if group_id:
                        # Stelle sicher, dass die Gruppe existiert, bevor wir das Surface zuordnen
                        # (analog zu TXT-Import, wo _ensure_group_for_label die Gruppe erstellt)
                        group = self.group_manager.get_group(group_id) if hasattr(self.group_manager, 'get_group') else None
                        if not group:
                            # Gruppe existiert nicht - erstelle sie (sollte eigentlich durch _ensure_group_for_label erstellt worden sein)
                            # Extrahiere Gruppennamen aus group_id (falls es ein Label gibt)
                            group_name = f"Group {group_id}" if group_id.startswith("group_") else group_id
                            group = self.group_manager.create_surface_group(group_name, group_id=group_id)
                        
                        if group:
                            self.group_manager.assign_surface_to_group(
                                sid_unique,
                                group_id,
                                create_missing=False,  # Gruppe sollte bereits existieren
                            )
                        # #region agent log - HYPOTHESIS B: Surface wurde Gruppe zugeordnet
                        try:
                            assigned_group = self.group_manager.get_group(group_id) if hasattr(self.group_manager, 'get_group') else None
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"SurfaceDataImporter.py:_store_surfaces:after_assign","message":"AFTER assigning to group","data":{"surface_id":sid_unique,"group_id":str(group_id),"group_exists":assigned_group is not None,"group_name":assigned_group.name if assigned_group else None,"group_surface_count":len(assigned_group.surface_ids) if assigned_group else 0},"timestamp":int(time.time()*1000)})+"\n")
                        except Exception:
                            pass
                        # #endregion
        if self.group_manager:
            # Struktur nur einmal am Ende sicherstellen
            # #region agent log - HYPOTHESIS B: Gruppen-Struktur wird sichergestellt
            import json, time
            try:
                groups_before = len(self.group_manager.list_groups()) if hasattr(self.group_manager, 'list_groups') else 0
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"SurfaceDataImporter.py:_store_surfaces:before_ensure","message":"BEFORE ensure_structure","data":{"groups_count":groups_before,"imported_count":imported},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            self.group_manager.ensure_surface_group_structure()
            # #region agent log - HYPOTHESIS B: Gruppen-Struktur wurde sichergestellt
            try:
                groups_after = len(self.group_manager.list_groups()) if hasattr(self.group_manager, 'list_groups') else 0
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"SurfaceDataImporter.py:_store_surfaces:after_ensure","message":"AFTER ensure_structure","data":{"groups_count":groups_after,"imported_count":imported},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
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
        """
        Lädt Flächen aus einer DXF-Datei.
        
        DXF-Flächen-Entitäten:
        - POLYLINE: 2D/3D Polylinien, geschlossen = Fläche
        - LWPOLYLINE: 2D Polylinien, geschlossen = Fläche  
        - 3DFACE: 3D-Fläche mit 3-4 Eckpunkten
        - MESH: Polygon-Mesh mit Vertices und Faces
        - SOLID: Gefüllte Fläche (4-Punkt-Fläche)
        
        Koordinaten:
        - Werden im Objektkoordinatensystem (OKS) gespeichert
        - Gruppencodes 10, 20, 30 für X, Y, Z
        - INSERT-Entities transformieren Koordinaten (Translation, Skalierung, Rotation)
        
        Gruppierung:
        - BLOCKS: Wiederverwendbare Gruppen von Entitäten
        - GROUPS: DXF-Gruppen (logische Gruppierung über Handles)
        - LAYERS: Ebenen-Organisation
        - INSERT: Block-Instanzen mit Transformationen (verschachtelte Hierarchie)
        """
        try:
            ezdxf = importlib.import_module("ezdxf")
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Das Paket 'ezdxf' ist nicht installiert.") from exc

        # Verbesserte Fehlerbehandlung beim Laden
        # #region agent log - HYPOTHESIS E: DXF-Datei laden
        import json, time
        try:
            log_data = {
                "sessionId": "debug-session",
                "runId": "run3",
                "hypothesisId": "E",
                "location": "SurfaceDataImporter.py:_load_dxf_surfaces:readfile",
                "message": "VERSUCHE ezdxf.readfile()",
                "data": {
                    "file_path": str(file_path),
                    "file_exists": file_path.exists(),
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "file_suffix": file_path.suffix
                },
                "timestamp": int(time.time()*1000)
            }
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception:
            pass
        # #endregion

        try:
            doc = ezdxf.readfile(str(file_path))
        except IOError as e:
            # #region agent log - HYPOTHESIS E: IOError beim Laden
            import json, time
            try:
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "run3",
                    "hypothesisId": "E",
                    "location": "SurfaceDataImporter.py:_load_dxf_surfaces:ioerror",
                    "message": "IOError beim Laden der DXF-Datei",
                    "data": {
                        "file_path": str(file_path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    "timestamp": int(time.time()*1000)
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            raise ValueError(f"DXF-Datei konnte nicht geöffnet werden: {file_path}") from e
        except ezdxf.DXFStructureError as e:
            # #region agent log - HYPOTHESIS E: DXFStructureError
            import json, time
            try:
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "run3",
                    "hypothesisId": "E",
                    "location": "SurfaceDataImporter.py:_load_dxf_surfaces:structure_error",
                    "message": "DXFStructureError - Datei beschädigt",
                    "data": {
                        "file_path": str(file_path),
                        "error": str(e)
                    },
                    "timestamp": int(time.time()*1000)
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            raise ValueError(f"Ungültige oder beschädigte DXF-Datei: {e}") from e
        except ezdxf.DXFVersionError as e:
            # #region agent log - HYPOTHESIS E: DXFVersionError
            import json, time
            try:
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "run3",
                    "hypothesisId": "E",
                    "location": "SurfaceDataImporter.py:_load_dxf_surfaces:version_error",
                    "message": "DXFVersionError - Version nicht unterstützt",
                    "data": {
                        "file_path": str(file_path),
                        "error": str(e)
                    },
                    "timestamp": int(time.time()*1000)
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            raise ValueError(f"DXF-Version wird nicht unterstützt: {e}") from e
        except Exception as e:
            # #region agent log - HYPOTHESIS E: Unerwarteter Fehler
            import json, time
            try:
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "run3",
                    "hypothesisId": "E",
                    "location": "SurfaceDataImporter.py:_load_dxf_surfaces:unexpected_error",
                    "message": "Unerwarteter Fehler beim Laden der DXF-Datei",
                    "data": {
                        "file_path": str(file_path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    "timestamp": int(time.time()*1000)
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            raise
        
        # Lese Header-Informationen
        self._read_dxf_header(doc, file_path)
        
        msp = doc.modelspace()
        self._log_dxf_debug_info(file_path, doc, msp)
        group_lookup = self._build_dxf_group_lookup(doc)
        layer_colors = self._build_layer_color_lookup(doc, ezdxf)
        
        # Analysiere Tags/Attribute in der DXF-Datei (inkl. ATTDEF)
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
            
            # #region agent log - HYPOTHESIS B: transform Parameter VOR _extract_points_from_entity
            import json, time
            try:
                log_data = {
                    "sessionId": "debug-session",
                    "runId": "run2",
                    "hypothesisId": "B",
                    "location": "SurfaceDataImporter.py:_load_dxf_surfaces:before_extract",
                    "message": "BEFORE _extract_points_from_entity",
                    "data": {
                        "entity_count": entity_count,
                        "dxftype": dxftype,
                        "layer": layer,
                        "transform_is_none": transform is None,
                        "has_block_stack": len(block_stack) > 0 if block_stack else False
                    },
                    "timestamp": int(time.time()*1000)
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            
            points = self._extract_points_from_entity(entity, transform)

            # #region agent log - HYPOTHESIS C: points NACH _extract_points_from_entity
            import json, time
            try:
                if points and len(points) >= 3:
                    x_vals = [p.get("x", 0) for p in points]
                    log_data = {
                        "sessionId": "debug-session",
                        "runId": "run2",
                        "hypothesisId": "C",
                        "location": "SurfaceDataImporter.py:_load_dxf_surfaces:after_extract",
                        "message": "AFTER _extract_points_from_entity",
                        "data": {
                            "entity_count": entity_count,
                            "dxftype": dxftype,
                            "layer": layer,
                            "points_count": len(points),
                            "first_point": points[0],
                            "x_range": [min(x_vals), max(x_vals)]
                        },
                        "timestamp": int(time.time()*1000)
                    }
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
            
            # #region agent log - HYPOTHESIS D: Punkt-Reihenfolge
            import json, time
            try:
                if points and len(points) >= 3:
                    # Berechne Flächeninhalt (für Prüfung der Orientierung)
                    area = 0.0
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        area += points[i].get("x", 0) * points[j].get("y", 0)
                        area -= points[j].get("x", 0) * points[i].get("y", 0)
                    area = abs(area) / 2.0
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"SurfaceDataImporter.py:_load_dxf_surfaces:point_order","message":"Point order check","data":{"entity_type":dxftype,"points_count":len(points),"first_point":points[0],"last_point":points[-1],"area":area},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion

            # #region agent log - DXF ENTITY: Original-Entity-Daten
            import json, time
            try:
                entity_handle = getattr(entity.dxf, "handle", "unknown")
                entity_data = {
                    "entity_count": entity_count,
                    "dxftype": dxftype,
                    "layer": layer,
                    "handle": entity_handle,
                    "points_extracted": len(points) if points else 0,
                    "has_transform": transform is not None,
                }
                if transform:
                    entity_data["transform"] = {
                        "translation": transform.get("translation", (0,0,0)),
                        "scale": transform.get("scale", (1,1,1)),
                        "rotation_deg": transform.get("rotation", 0) * 180 / 3.14159 if transform.get("rotation") else 0
                    }
                if points:
                    entity_data["points"] = [{"x": p.get("x"), "y": p.get("y"), "z": p.get("z")} for p in points[:10]]  # Erste 10 Punkte
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DXF_ENTITY","location":"SurfaceDataImporter.py:_load_dxf_surfaces:entity_extracted","message":"DXF Entity extracted","data":entity_data,"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion

            if not points:
                logger.debug(
                    "Entity %d: %s auf Layer '%s' übersprungen (keine Punkte extrahiert)",
                    entity_count, dxftype, layer
                )
                entity_type_stats[f"{dxftype}_keine_punkte"] += 1
                skipped_count += 1
                self._log_skipped_entity(entity)
                # #region agent log - DXF ENTITY: Übersprungen (keine Punkte)
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DXF_SKIP","location":"SurfaceDataImporter.py:_load_dxf_surfaces:skipped_no_points","message":"Entity skipped - no points","data":{"entity_count":entity_count,"dxftype":dxftype,"layer":layer},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
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
                # #region agent log - DXF ENTITY: Übersprungen (LINE)
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DXF_SKIP","location":"SurfaceDataImporter.py:_load_dxf_surfaces:skipped_line","message":"Entity skipped - LINE with 2 points","data":{"entity_count":entity_count,"dxftype":dxftype,"layer":layer,"start":start,"end":end},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
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
                # #region agent log - DXF ENTITY: Übersprungen (zu wenig Punkte)
                try:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"DXF_SKIP","location":"SurfaceDataImporter.py:_load_dxf_surfaces:skipped_too_few_points","message":"Entity skipped - too few points","data":{"entity_count":entity_count,"dxftype":dxftype,"layer":layer,"points_count":len(points),"points":points},"timestamp":int(time.time()*1000)})+"\n")
                except Exception:
                    pass
                # #endregion
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
            
            # Verwende Tag als Gruppierung, falls vorhanden (wie beim TXT-Import)
            if tag:
                logger.info(
                    "Entity %d: Tag '%s' gefunden für %s auf Layer '%s'",
                    entity_count, tag, dxftype, layer
                )
                # Verwende Tag als Basis-Name, falls Layer nicht aussagekräftig
                if not base_name or base_name == dxftype:
                    base_name = tag
                # Verwende Tag für Gruppierung (analog zu TXT-Import)
                group_label = self._derive_group_label(tag)
                target_group_id = self._ensure_group_for_label(group_label)
            else:
                # Fallback: Gruppierung über Block-Hierarchie (wenn kein Tag gefunden)
                group_label = self._build_group_label(block_stack)
                target_group_id = self._ensure_group_for_label(group_label)
            if not target_group_id:
                # Fallback auf ursprüngliche Gruppen-Logik
                group_name = group_lookup.get(entity.dxf.handle)
                target_group_id = self._ensure_group_for_label(group_name)

            # Nummeriere Surfaces innerhalb derselben Gruppe (analog zu TXT-Import)
            safe_label = self._make_safe_identifier(base_name) or "surface"
            id_counters[safe_label] = id_counters.get(safe_label, 0) + 1
            surface_number = id_counters[safe_label]
            
            # Surface-ID: ohne Nummer (für interne Verwendung)
            surface_id = f"{safe_label}_{surface_number}" if surface_number > 1 else safe_label
            
            # Surface-Name: mit Nummer (für Anzeige, analog zu TXT-Import)
            surface_name = f"{base_name} {surface_number}" if surface_number > 1 else base_name

            color = self._resolve_entity_color(entity, layer_colors, ezdxf)

            payload = {
                "name": surface_name,  # Name mit Nummer (z.B. "BETON 1", "BETON 2")
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

    def _read_dxf_header(self, doc, file_path: Path) -> None:
        """
        Liest wichtige Header-Variablen aus DXF-Datei.
        
        Header enthält Metadaten wie:
        - DXF-Version
        - Einheiten (Inches, Feet, Millimeters, etc.)
        - Koordinatensystem (UCS)
        - Zeichnungsgrenzen (EXTMIN, EXTMAX)
        """
        logger = logging.getLogger(__name__)
        try:
            header = doc.header
            
            # DXF-Version
            dxf_version = doc.dxfversion
            logger.info("DXF-Version: %s", dxf_version)
            if dxf_version < "AC1009":  # R12 oder älter
                logger.warning("Sehr alte DXF-Version: %s - mögliche Kompatibilitätsprobleme", dxf_version)
            
            # Einheiten (1=Inches, 2=Feet, 4=Millimeters, 5=Centimeters, etc.)
            units = header.get('$INSUNITS', 1)
            unit_names = {
                0: "Unitless",
                1: "Inches",
                2: "Feet",
                3: "Miles",
                4: "Millimeters",
                5: "Centimeters",
                6: "Meters",
                7: "Kilometers",
                8: "Microinches",
                9: "Mils",
                10: "Yards",
                11: "Angstroms",
                12: "Nanometers",
                13: "Microns",
                14: "Decimeters",
                15: "Decameters",
                16: "Hectometers",
                17: "Gigameters",
                18: "Astronomical Units",
                19: "Light Years",
                20: "Parsecs"
            }
            unit_name = unit_names.get(units, f"Unknown ({units})")
            logger.info("DXF-Einheiten: %s (%s)", unit_name, units)
            
            # Koordinatensystem (UCS)
            ucs_origin = header.get('$UCSORG', None)
            if ucs_origin:
                logger.debug("UCS-Origin: %s", ucs_origin)
            
            # Zeichnungsgrenzen
            extmin = header.get('$EXTMIN', None)
            extmax = header.get('$EXTMAX', None)
            if extmin and extmax:
                logger.debug("Zeichnungsgrenzen: MIN=%s, MAX=%s", extmin, extmax)
            
        except Exception as e:
            logger.warning("Fehler beim Lesen der DXF-Header-Informationen: %s", e)
    
    def _extract_block_attributes(self, block) -> Dict[str, str]:
        """
        Extrahiert Attribute aus Block-Definition (ATTDEF-Entities).
        
        ATTDEF (Attribute Definition) definiert die Struktur von Attributen
        in einem Block. Diese werden in INSERT-Instanzen durch ATTRIB-Entities
        mit tatsächlichen Werten gefüllt.
        
        Returns:
            Dict mit Attribut-Tag als Key und Standardwert als Value
        """
        attributes = {}
        try:
            for entity in block:
                if entity.dxftype() == "ATTDEF":
                    tag = getattr(entity.dxf, "tag", None)
                    default = getattr(entity.dxf, "text", None)
                    if tag:
                        attributes[tag] = default or ""
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug("Fehler beim Extrahieren von Block-Attributen: %s", e)
        return attributes

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
                    raw_elevation = getattr(entity.dxf, 'elevation', 0.0)
                    elevation = self._extract_float_from_vec3(raw_elevation, 0.0)
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
                    raw_elevation = getattr(entity.dxf, 'elevation', 0.0)
                    elevation = self._extract_float_from_vec3(raw_elevation, 0.0)
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
                    raw_elevation = getattr(entity.dxf, 'elevation', 0.0)
                    elevation = self._extract_float_from_vec3(raw_elevation, 0.0)
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
            # Präfixiere Gruppennamen mit "Group " für die Anzeige
            group_name = f"Group {key}" if not key.startswith("Group ") else key
            
            if "/" in key:
                # Für verschachtelte Pfade: Präfixiere jeden Segment
                segments = [segment.strip() for segment in key.split("/") if segment.strip()]
                prefixed_segments = [f"Group {seg}" if not seg.startswith("Group ") else seg for seg in segments]
                prefixed_path = "/".join(prefixed_segments)
                group_id = self.group_manager.ensure_group_path(prefixed_path)
            else:
                # Suche zuerst nach existierender Gruppe mit diesem Namen (mit oder ohne Präfix)
                existing_group = self.group_manager.find_surface_group_by_name(group_name)
                if not existing_group:
                    # Versuche auch ohne "Group " Präfix zu suchen
                    existing_group = self.group_manager.find_surface_group_by_name(key)
                if existing_group:
                    group_id = existing_group.group_id
                    # Stelle sicher, dass die Gruppe den richtigen Namen hat
                    if existing_group.name != group_name:
                        self.group_manager.rename_surface_group(group_id, group_name)
                        existing_group.name = group_name
                else:
                    # Erstelle neue Gruppe mit "Group " Präfix
                    new_group = self.group_manager.create_surface_group(group_name)
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
        """
        Analysiert die DXF-Datei auf Tags/Attribute.
        
        Sucht nach Material-Tags in:
        - Block-Namen
        - ATTDEF-Entities (Block-Definition Attribute)
        - ATTRIB-Entities (INSERT-Instanz Attribute)
        - Layer-Namen
        - XDATA/AppData
        """
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
                
                # NEU: Prüfe ATTDEF-Entities in Block-Definition
                block_attributes = self._extract_block_attributes(block)
                for attr_tag, attr_value in block_attributes.items():
                    if attr_value:
                        attr_upper = str(attr_value).strip().upper()
                        if attr_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                            tags_found.add(attr_upper)
                            # Speichere Mapping für Block
                            if block.name not in self._block_tag_map:
                                self._block_tag_map[block.name] = attr_upper
                
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
        """
        Extrahiert Tag aus INSERT-Entity über Block-Attribute.
        
        Berücksichtigt:
        - Block-Name selbst (wenn bekanntes Material)
        - ATTDEF-Entities in Block-Definition (Standardwerte)
        - ATTRIB-Entities in INSERT-Instanz (tatsächliche Werte)
        - Layer-Namen der Entities im Block
        """
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
            
            # NEU: Extrahiere ATTDEF-Entities aus Block-Definition (Standardwerte)
            block_attributes = self._extract_block_attributes(block)
            for attr_tag, attr_value in block_attributes.items():
                if attr_value:
                    attr_upper = str(attr_value).strip().upper()
                    if attr_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                        return attr_upper
                    # Verwende als Tag, wenn aussagekräftig
                    if len(str(attr_value).strip()) > 0 and not str(attr_value).strip().isdigit():
                        return str(attr_value).strip()
            
            # Suche nach ATTRIB-Entities im Block (tatsächliche Werte in INSERT-Instanz)
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
        """
        Extrahiert Tag aus XDATA (Extended Data).
        
        XDATA wird von CAD-Anwendungen verwendet, um erweiterte Informationen
        an Entities anzuhängen. Strukturierte Suche nach bekannten App-Namen.
        """
        try:
            if not hasattr(entity, "xdata"):
                return None
            
            xdata = entity.xdata
            if not xdata:
                return None
            
            # Bekannte App-Namen, die Material-Informationen enthalten könnten
            known_apps = ["ACAD", "LFO", "SOUNDPLAN", "AUTOCAD", "MATERIAL"]
            
            # Zuerst in bekannten Apps suchen
            for app_name in known_apps:
                if app_name in xdata:
                    data = xdata[app_name]
                    tag = self._parse_xdata_for_tag(data)
                    if tag:
                        return tag
            
            # Fallback: Durchsuche alle Apps
            for app_name, data in xdata.items():
                tag = self._parse_xdata_for_tag(data)
                if tag:
                    return tag
                    
        except Exception:
            pass
        
        return None
    
    def _parse_xdata_for_tag(self, data) -> Optional[str]:
        """Hilfsfunktion zum Parsen von XDATA-Daten nach Tags"""
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, str):
                    item_upper = item.upper()
                    if item_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                        return item_upper
                    # Suche nach Material-Keywords in Strings
                    if any(keyword in item_upper for keyword in ["MATERIAL", "TAG", "TYPE"]):
                        # Versuche Wert nach Keyword zu extrahieren
                        parts = item_upper.split()
                        for i, part in enumerate(parts):
                            if part in ("MATERIAL", "TAG", "TYPE") and i + 1 < len(parts):
                                candidate = parts[i + 1]
                                if candidate in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                                    return candidate
        elif isinstance(data, str):
            data_upper = data.upper()
            if data_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
                return data_upper
        
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
        
        # #region agent log - HYPOTHESIS A: Koordinaten vor Transformation
        import json, time
        try:
            if points and transform:
                first_point = points[0] if points else None
                transform_info = {
                    "translation": transform.get("translation", (0,0,0)),
                    "scale": transform.get("scale", (1,1,1)),
                    "rotation_deg": transform.get("rotation", 0) * 180 / 3.14159 if transform.get("rotation") else 0
                }
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"SurfaceDataImporter.py:_extract_points_from_entity:before_transform","message":"BEFORE transform","data":{"entity_type":dxftype,"points_count":len(points),"first_point":first_point,"transform":transform_info},"timestamp":int(time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        
        # Wende Transformation an
        if transform:
            transformed_points = [self._apply_transform_to_point(p, transform) for p in points]
            # #region agent log - HYPOTHESIS A: Koordinaten nach Transformation
            try:
                if transformed_points:
                    first_transformed = transformed_points[0]
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"SurfaceDataImporter.py:_extract_points_from_entity:after_transform","message":"AFTER transform","data":{"entity_type":dxftype,"points_count":len(transformed_points),"first_point":first_transformed,"all_points":transformed_points[:5]},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
            return transformed_points
        
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
            # WICHTIG: Verwende explode() statt virtual_entities()!
            # explode() gibt Entities in Welt-Koordinaten (WCS) zurück,
            # virtual_entities() gibt Block-lokale Koordinaten zurück.
            virtual_entities = list(insert_entity.explode())
            entity_count = len(virtual_entities)
            entity_types = {}
            for ve in virtual_entities:
                ve_type = ve.dxftype()
                entity_types[ve_type] = entity_types.get(ve_type, 0) + 1
            
            type_summary = ", ".join(f"{k}:{v}" for k, v in entity_types.items())
            logger.info(
                "INSERT '%s' (Tiefe %d) aufgelöst (explode): %d Entities extrahiert [%s]",
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
                
                # #region agent log - HYPOTHESIS A: explode() Koordinaten VOR _resolve_entity
                import json, time
                try:
                    exploded_points = []
                    if vtype in ("POLYLINE", "LWPOLYLINE"):
                        try:
                            for vertex in virtual.vertices:
                                loc = vertex.dxf.location
                                exploded_points.append({"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)})
                        except:
                            pass
                    elif vtype == "LINE":
                        try:
                            start = virtual.dxf.start
                            end = virtual.dxf.end
                            exploded_points = [
                                {"x": float(start.x), "y": float(start.y), "z": float(start.z)},
                                {"x": float(end.x), "y": float(end.y), "z": float(end.z)}
                            ]
                        except:
                            pass

                    if exploded_points:
                        x_vals = [p["x"] for p in exploded_points]
                        log_data = {
                            "sessionId": "debug-session",
                            "runId": "run2",
                            "hypothesisId": "A",
                            "location": "SurfaceDataImporter.py:_expand_insert:explode",
                            "message": "explode() Koordinaten DIREKT nach explode()",
                            "data": {
                                "block_name": block_name,
                                "entity_idx": idx,
                                "dxftype": vtype,
                                "layer": vlayer,
                                "points_count": len(exploded_points),
                                "first_point": exploded_points[0] if exploded_points else None,
                                "x_range": [min(x_vals), max(x_vals)] if x_vals else None
                            },
                            "timestamp": int(time.time()*1000)
                        }
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
                # #endregion
                
                # KORREKTUR: Verwende explode() statt virtual_entities()
                # explode() gibt bereits Welt-Koordinaten (WCS) zurück!
                # Daher KEINE Transformation mehr anwenden.
                yield from self._resolve_entity(
                    virtual,
                    depth + 1,
                    None,  # Keine Transformation - explode() gibt bereits WCS-Koordinaten
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
        
        # #region agent log - HYPOTHESIS B: INSERT-Transformation extrahiert
        import json, time
        try:
            block_name = getattr(insert_entity.dxf, "name", "unknown")
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"SurfaceDataImporter.py:_extract_insert_transform","message":"INSERT transform extracted","data":{"block_name":block_name,"translation":(tx,ty,tz),"scale":(xscale,yscale,zscale),"rotation_deg":rotation,"has_negative_scale":xscale<0 or yscale<0 or zscale<0},"timestamp":int(time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        
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
        
        # #region agent log - HYPOTHESIS C: Punkt vor Transformation
        import json, time
        try:
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"SurfaceDataImporter.py:_apply_transform_to_point:before","message":"Point BEFORE transform","data":{"x":x,"y":y,"z":z,"scale":transform.get("scale",(1,1,1)),"rotation_rad":transform.get("rotation",0)},"timestamp":int(time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        
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

