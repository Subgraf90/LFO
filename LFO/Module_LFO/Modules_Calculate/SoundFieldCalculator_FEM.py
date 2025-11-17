"""FEM-basierte Berechnung des Schallfelds mit FEniCSx.

Dieses Modul folgt der gleichen Datenschnittstelle wie `SoundFieldCalculator`
und erweitert die Berechnung um eine Finite-Elemente-Lösung der Helmholtz-
Gleichung. Die Ergebnisse werden im selben `calculation_spl`-Dict abgelegt.

Hinweise zur Verwendung
-----------------------
- Erfordert eine funktionierende FEniCSx-Installation (dolfinx, ufl, mpi4py,
  petsc4py). Auf macOS empfiehlt sich die Installation über Conda/Mambaforge
  oder ein Docker-Image, da vorgebaute Wheels nur eingeschränkt verfügbar sind.
- Die Berechnung läuft aktuell im 2D-Setting (XY-Ebene). Für 3D-Berechnungen
  muss die Mesh-Erzeugung und die Quellenprojektion erweitert werden.
- Die erzeugten Ergebnisse werden unter `calculation_spl["fem_simulation"]`
  abgelegt und enthalten neben SPL/Phase optional auch Partikelgeschwindigkeiten.
"""

from __future__ import annotations

import math
import importlib
import os
import sys
import sysconfig
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Init.Progress import ProgressCancelled

try:  # Optional Beschleunigung für Punktquellen-Projektion
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional dependency
    cKDTree = None

MPI = None
dolfinx_pkg = None
fem = None
mesh = None
ufl = None
PETSc = None
gmshio = None
dolfinx_gmsh = None
default_scalar_type = np.float64
_fenics_import_error: Optional[BaseException] = None

try:
    gmsh = importlib.import_module("gmsh")
except ImportError:  # pragma: no cover - optionale Abhängigkeit
    gmsh = None


def _try_import_fenics_modules():
    """Versucht die optionalen FEniCSx-Module nachzuladen."""
    global MPI, dolfinx_pkg, fem, mesh, ufl, PETSc, gmshio, dolfinx_gmsh, default_scalar_type, _fenics_import_error

    if all(module is not None for module in (MPI, dolfinx_pkg, fem, mesh, ufl, PETSc, gmshio)):
        _fenics_import_error = None
        return

    try:
        MPI = importlib.import_module("mpi4py.MPI")
        dolfinx_pkg = importlib.import_module("dolfinx")
        fem = importlib.import_module("dolfinx.fem")
        mesh = importlib.import_module("dolfinx.mesh")
        ufl = importlib.import_module("ufl")
        PETSc = importlib.import_module("petsc4py.PETSc")
        try:
            gmshio = importlib.import_module("dolfinx.io.gmshio")
            dolfinx_gmsh = None
        except ImportError:
            gmshio = None
            dolfinx_gmsh = importlib.import_module("dolfinx.io.gmsh")
        default_scalar_type = dolfinx_pkg.default_scalar_type
        _fenics_import_error = None
    except Exception as exc:  # pragma: no cover - nur Diagnose
        MPI = fem = mesh = ufl = PETSc = gmshio = dolfinx_pkg = dolfinx_gmsh = None
        default_scalar_type = np.float64
        _fenics_import_error = exc


@dataclass
class SpeakerPanel:
    """Repräsentiert eine Lautsprecherfläche im 2D-Mesh."""

    identifier: str
    array_key: str
    points: np.ndarray
    width: float
    height: float
    perimeter: float
    area: float
    center: np.ndarray
    azimuth_rad: float
    level_adjust_db: float
    speaker_name: Optional[str] = None
    line_height: float = 0.0
    depth: float = 0.0
    is_muted: bool = False


@dataclass
class SpeakerCabinetObstacle:
    """Repräsentiert das Gehäuse als undurchlässiges Hindernis."""

    identifier: str
    array_key: str
    points: np.ndarray
    width: float
    depth: float
    center: np.ndarray
    azimuth_rad: float
    speaker_name: Optional[str] = None
    is_muted: bool = False


class SoundFieldCalculatorFEM(ModuleBase):
    """FEM-gestützte Helmholtz-Lösung in 2D.

    Attribute
    ---------
    settings : Any
        Globale Einstellungen (identisch zum klassischen SoundFieldCalculator).
    data : Any
        Daten-Container aus dem Hauptprogramm (z.B. Balloon-/Quellendaten).
    calculation_spl : dict
        Gemeinsames Ergebnis-Dictionary. Ergebnisse werden unter
        ``calculation_spl["fem_simulation"]`` abgelegt.
    """

    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl

        self._data_container = None
        self._mesh = None
        self._function_space = None
        self._mesh_resolution = None
        self._cell_tags = None
        self._facet_tags = None
        self._outer_boundary_tag = 1
        self._last_point_source_positions = None
        self._mesh_version = 0
        self._dof_coordinates = None
        self._dof_coords_xy = None
        self._dof_tree = None
        self._dof_cache_version = None
        self._grid_cache = None
        self._balloon_data_cache: dict[str, dict] = {}
        self._python_include_path_prepared = False
        self._frequency_progress_session = None
        self._precomputed_frequencies: Optional[list[float]] = None
        self._frequency_progress_last_third_start: Optional[int] = None
        self._timings_store: defaultdict[str, list[float]] = defaultdict(list)
        self._resolved_fem_frequency: Optional[float] = None
        self._panels: list[SpeakerPanel] = []
        self._cabinet_obstacles: list[SpeakerCabinetObstacle] = []
        self._time_snapshots_cache: dict[float, dict[str, np.ndarray]] = {}

    def _log_debug(self, message: str):
        """Hilfsfunktion für konsistente Debug-Ausgaben."""
        if getattr(self.settings, "fem_debug_logging", True):
            print(f"[FEM Debug] {message}")

    # ------------------------------------------------------------------
    # Timing-Hilfsfunktionen
    # ------------------------------------------------------------------
    def _reset_timings(self):
        self._timings_store = defaultdict(list)

    def _record_timing(self, label: str, duration: float):
        if not label:
            label = "unlabeled"
        self._timings_store[label].append(float(duration))

    def _summarize_timings(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for label, durations in self._timings_store.items():
            if not durations:
                continue
            total = float(sum(durations))
            count = len(durations)
            summary[label] = {
                "count": float(count),
                "total": total,
                "avg": total / count if count else 0.0,
                "last": float(durations[-1]),
            }
        return summary

    @contextmanager
    def _time_block(self, label: str, detail: Optional[str] = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record_timing(label, elapsed)
            if detail:
                self._record_timing(f"{label}#{detail}", elapsed)

    def _log_timings(self, summary: dict[str, dict[str, float]]):
        """Gibt die gemessenen Laufzeiten auf der Konsole aus."""
        if not summary:
            print("[FEM Timing] Keine Messdaten verfügbar.")
            return

        def _fmt(val: float) -> str:
            return f"{val * 1000:.2f} ms"

        aggregate = {k: v for k, v in summary.items() if "#" not in k}
        detail = {k: v for k, v in summary.items() if "#" in k}

        print("\n[FEM Timing] Übersicht der Rechenschritte:")
        for label, stats in sorted(
            aggregate.items(), key=lambda item: item[1]["total"], reverse=True
        ):
            count = int(stats["count"])
            print(
                f" - {label:<35} total={_fmt(stats['total'])} "
                f"avg={_fmt(stats['avg'])} count={count} last={_fmt(stats['last'])}"
            )

        if detail:
            print("[FEM Timing] Detail (pro Frequenz, Top 10 nach total):")
            for label, stats in sorted(
                detail.items(), key=lambda item: item[1]["total"], reverse=True
            )[:10]:
                count = int(stats["count"])
                print(
                    f"   • {label:<33} total={_fmt(stats['total'])} "
                    f"avg={_fmt(stats['avg'])} count={count} last={_fmt(stats['last'])}"
                )
        print()

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def calculate_soundfield_pressure(self):
        """Berechnet das Schallfeld für alle Frequenzen mit FEM.

        Ablauf:
            1. Sicherstellen, dass alle FEM-Abhängigkeiten verfügbar sind.
            2. Frequenzliste anhand der Bandbreite bestimmen (Einzelwert → Liste).
            3. Geometrie-/Materialparameter in ein FEM-Mesh übersetzen.
            4. Für jede Frequenz das Helmholtz-Problem lösen und in das
               gemeinsame Ergebnis-Dictionary schreiben.

        Ergebnisse werden pro Frequenz als Druck (Magnitude), SPL und Phase
        gespeichert. Zusätzlich werden optionale FEM-Größen abgelegt
        (z.B. Partikelgeschwindigkeit), sofern verfügbar.
        """

        self._reset_timings()

        with self._time_block("ensure_fenics_available"):
            self._ensure_fenics_available()
        with self._time_block("prepare_python_include_path"):
            self._prepare_python_include_path()

        if self._precomputed_frequencies is not None:
            with self._time_block("use_precomputed_frequencies"):
                frequencies = list(self._precomputed_frequencies)
                self._precomputed_frequencies = None
        else:
            with self._time_block("determine_frequencies"):
                frequencies = self._determine_frequencies()
        if not frequencies:
            raise ValueError("Es sind keine Frequenzen zur Berechnung definiert.")

        with self._time_block("build_domain"):
            self._build_domain(frequencies)
        
        # Info über FEM-Konfiguration
        use_direct_solver = getattr(self.settings, "fem_use_direct_solver", True)
        width = float(self.settings.width)
        length = float(self.settings.length)
        grid_resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)
        fem_resolution = getattr(self, "_mesh_resolution", grid_resolution)

        fem_results = {}
        total_freqs = len(frequencies)
        compute_velocity = bool(getattr(self.settings, "fem_compute_particle_velocity", True))
        
        for idx, frequency in enumerate(frequencies, 1):
            # Progress-Callback für UI-Updates (wenn vorhanden)
            if hasattr(self, '_progress_callback') and callable(self._progress_callback):
                self._progress_callback(f"FEM: {frequency} Hz ({idx}/{total_freqs})")
            self._raise_if_frequency_cancelled()
            if (
                self._frequency_progress_session is not None
                and self._frequency_progress_last_third_start is not None
                and idx >= self._frequency_progress_last_third_start
            ):
                self._frequency_progress_session.update(
                    f"FEM Vorbereitung: {frequency} Hz ({idx}/{total_freqs})"
                )
                self._frequency_progress_session.advance()
            if self._frequency_progress_session is not None:
                self._frequency_progress_session.update(
                    f"FEM: {frequency} Hz ({idx}/{total_freqs})"
                )
            
            with self._time_block("solve_frequency", f"{frequency:.2f}Hz"):
                solution = self._solve_frequency(frequency)
            with self._time_block("extract_pressure_and_phase", f"{frequency:.2f}Hz"):
                pressure, spl, phase = self._extract_pressure_and_phase(solution)

            velocity = None
            if compute_velocity:
                with self._time_block("compute_particle_velocity", f"{frequency:.2f}Hz"):
                    velocity = self._compute_particle_velocity(solution, frequency)

            # WICHTIG: Verwende DOF-Koordinaten, nicht geometrische Mesh-Knoten!
            # Bei quadratischen Elementen gibt es mehr DOFs als geometrische Knoten
            # (zusätzliche DOFs an Kanten-Mittelpunkten)
            dof_coordinates = self._function_space.tabulate_dof_coordinates()
            
            fem_results[float(frequency)] = {
                "points": dof_coordinates,  # Koordinaten ALLER DOFs
                "pressure_complex": solution.x.array.copy(),
                "pressure": pressure,
                "spl": spl,
                "phase": phase,
                "source_positions": getattr(self, "_last_point_source_positions", None),
            }

            if compute_velocity and velocity is not None:
                fem_results[float(frequency)]["particle_velocity"] = velocity

            if self._frequency_progress_session is not None:
                self._frequency_progress_session.advance()
            self._raise_if_frequency_cancelled()

        self.calculation_spl["fem_simulation"] = fem_results
        self.calculation_spl.setdefault("fem_time_snapshots", {})
        
        with self._time_block("assign_primary_soundfield_results"):
            self._assign_primary_soundfield_results(frequencies, fem_results)

        self._frequency_progress_session = None
        self._frequency_progress_last_third_start = None
        timing_summary = self._summarize_timings()
        self.calculation_spl["fem_timings"] = timing_summary
        self._log_timings(timing_summary)

        if not fem_results:
            self._log_debug("[Ergebnisse] Keine FEM-Frequenzen vorhanden.")
        self.calculation_spl.setdefault("fem_time_snapshots", {})
        
    def set_data_container(self, data_container):
        """Setzt den gemeinsam genutzten Daten-Container."""
        self._data_container = data_container
        self._balloon_data_cache.clear()
    
    def set_progress_callback(self, callback):
        """
        Setzt Callback für Progress-Updates während FEM-Berechnung.
        Args: callback - Funktion(str) für UI-Updates
        """
        self._progress_callback = callback

    def set_frequency_progress_session(self, progress_session):
        """Registriert eine ProgressSession für Frequenz-Schritte."""
        self._frequency_progress_session = progress_session

    def set_frequency_progress_plan(self, last_third_start: Optional[int]):
        """Definiert ab welchem Index zusätzliche Fortschritts-Schritte angezeigt werden."""
        if last_third_start is None or last_third_start < 1:
            self._frequency_progress_last_third_start = None
        else:
            self._frequency_progress_last_third_start = last_third_start

    def set_precomputed_frequencies(self, frequencies: Optional[Iterable[float]]):
        """Setzt vorab berechnete Frequenzen, um doppelte Berechnungen zu vermeiden."""
        if not frequencies:
            self._precomputed_frequencies = None
            return
        self._precomputed_frequencies = [float(f) for f in frequencies]

    # ------------------------------------------------------------------
    # FEM-Hilfsfunktionen
    # ------------------------------------------------------------------
    def _raise_if_frequency_cancelled(self):
        session = getattr(self, "_frequency_progress_session", None)
        if session is not None and hasattr(session, "raise_if_cancelled"):
            session.raise_if_cancelled()

    def _ensure_fenics_available(self):
        _try_import_fenics_modules()
        if fem is None or mesh is None or MPI is None or ufl is None or PETSc is None:
            hint = ""
            if _fenics_import_error is not None:
                hint = f" (Grund: {_fenics_import_error})"
            raise ImportError(
                "FEniCSx (dolfinx, ufl, mpi4py) ist nicht installiert oder konnte nicht geladen werden."
                f"{hint}"
            )
        
        # Aktiviere verbose Logging für JIT-Kompilierung
        import logging as py_logging
        py_logging.getLogger("ffcx").setLevel(py_logging.INFO)
        py_logging.getLogger("dolfinx").setLevel(py_logging.INFO)

    def _normalize_frequencies(self, frequencies) -> list[float]:
        if frequencies is None:
            return []
        if isinstance(frequencies, (int, float, np.integer, np.floating)):
            return [float(frequencies)]
        if isinstance(frequencies, np.ndarray):
            return [float(f) for f in frequencies.flatten()]
        if isinstance(frequencies, Iterable):
            return [float(f) for f in frequencies]
        raise TypeError(
            "calculate_frequency muss Zahl oder Iterable von Zahlen sein."
        )

    def _determine_frequencies(self) -> list[float]:
        """Ermittelt exakt eine FEM-Frequenz (ggf. auf verfügbare Daten gerundet)."""

        self._resolved_fem_frequency = None

        fem_frequency = getattr(self.settings, "fem_calculate_frequency", None)
        base_frequency = fem_frequency

        base = self._normalize_frequencies(base_frequency)
        if not base:
            return []

        target_frequency = float(base[0])

        nearest_available = self._find_nearest_available_frequency(target_frequency)
        resolved_frequency = nearest_available if nearest_available is not None else target_frequency

        self._resolved_fem_frequency = resolved_frequency
        return [resolved_frequency]

    def _find_nearest_available_frequency(self, target_frequency: float) -> Optional[float]:
        """Sucht die nächstliegende verfügbare Frequenz in den Balloon-Daten."""
        container = self._get_data_container()
        if container is None or target_frequency is None:
            return None

        best_match = None
        best_diff = math.inf

        for speaker_name in self._iter_active_speaker_names():
            balloon = container.get_balloon_data(speaker_name, use_averaged=False)
            if not balloon:
                continue

            freqs = balloon.get("freqs")
            if freqs is None:
                freqs = balloon.get("frequencies")
            if freqs is None:
                continue

            freqs = np.asarray(freqs, dtype=float).flatten()
            if freqs.size == 0:
                continue

            idx = int(np.argmin(np.abs(freqs - target_frequency)))
            candidate = float(freqs[idx])
            diff = abs(candidate - target_frequency)

            if diff < best_diff:
                best_diff = diff
                best_match = candidate

            if diff == 0:
                break

        return best_match

    def _iter_active_speaker_names(self) -> Iterable[str]:
        if not hasattr(self.settings, "speaker_arrays"):
            return []

        for speaker_array in self.settings.speaker_arrays.values():
            if getattr(speaker_array, "mute", False) or getattr(speaker_array, "hide", False):
                continue
            
            # Hole source_polar_pattern (kann None, Liste oder numpy Array sein)
            source_patterns = getattr(speaker_array, "source_polar_pattern", None)
            if source_patterns is None:
                continue
            
            # Konvertiere zu iterierbarer Liste falls numpy Array
            if isinstance(source_patterns, np.ndarray):
                source_patterns = source_patterns.tolist()
            elif not isinstance(source_patterns, (list, tuple)):
                source_patterns = [source_patterns]
            
            for speaker_name in source_patterns:
                if speaker_name:
                    yield speaker_name

    def _get_data_container(self):
        """Liefert den Daten-Container (für Cabinet-/Balloon-Daten)."""
        if self._data_container is not None:
            return self._data_container
        return getattr(self, "data", None)

    def _build_cabinet_lookup(self, container) -> Dict[str, object]:
        """
        Erstellt ein Lookup von Lautsprechernamen zu Cabinet-Metadaten.
        Orientiert sich an der Logik aus PlotSPL3DOverlays.
        """
        lookup: Dict[str, object] = {}
        if container is None:
            return lookup

        data = container
        if not isinstance(data, dict):
            data = getattr(container, "data", None)
        if not isinstance(data, dict):
            data = {}

        speaker_names = data.get("speaker_names") or []
        cabinet_data = data.get("cabinet_data") or []

        def _as_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                return list(value)
            except TypeError:
                return [value]

        speaker_names = _as_list(speaker_names)
        cabinet_data = _as_list(cabinet_data)

        for name, cabinet in zip(speaker_names, cabinet_data):
            if not isinstance(name, str):
                continue
            lookup[name] = cabinet
            lookup.setdefault(name.lower(), cabinet)

        alias_mapping = getattr(container, "_speaker_name_mapping", None)
        if isinstance(alias_mapping, dict):
            for alias, actual in alias_mapping.items():
                if not isinstance(alias, str):
                    continue
                if actual in lookup:
                    lookup.setdefault(alias, lookup[actual])
                    lookup.setdefault(alias.lower(), lookup[actual])

        return lookup

    @staticmethod
    def _decode_speaker_name(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        try:
            return str(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_source_name_sequence(source_names) -> List[str]:
        if source_names is None:
            return []
        if isinstance(source_names, np.ndarray):
            source_names = source_names.tolist()
        elif not isinstance(source_names, (list, tuple)):
            source_names = [source_names]

        normalized: List[str] = []
        for item in source_names:
            if item is None:
                continue
            if isinstance(item, bytes):
                try:
                    item = item.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            try:
                normalized.append(str(item))
            except Exception:
                continue
        return normalized

    def _resolve_cabinet_entry(self, speaker_name: Optional[str], cabinet_lookup: Dict[str, object]) -> Optional[dict]:
        if not speaker_name:
            return None
        entry = cabinet_lookup.get(speaker_name)
        if entry is None:
            entry = cabinet_lookup.get(speaker_name.lower())
        if isinstance(entry, dict):
            return entry
        if isinstance(entry, list):
            for candidate in entry:
                if isinstance(candidate, dict):
                    return candidate
        return None

    def _resolve_cabinet_dimensions(
        self,
        speaker_name: Optional[str],
        cabinet_lookup: Dict[str, object],
        default_width: float,
        default_height: float,
        default_depth: float,
    ) -> Tuple[float, float, float]:
        entry = self._resolve_cabinet_entry(speaker_name, cabinet_lookup)
        if not isinstance(entry, dict):
            raise ValueError(f"Keine Abmessungen für Lautsprecher '{speaker_name}' hinterlegt.")

        def _safe_float(key: str, fallback: float) -> float:
            try:
                value = entry.get(key, fallback)
                return float(value)
            except Exception:
                return float(fallback)

        width = _safe_float("width", default_width)
        height = _safe_float("front_height", default_height)
        depth = _safe_float("depth", default_depth)

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Ungültige Membranabmessungen für '{speaker_name}': width={width}, height={height}."
            )
        if depth <= 0.0:
            depth = default_depth
        return float(width), float(height), float(depth)

    def _derive_line_height(self, physical_height: float) -> float:
        """Leitet die effektive Linienhöhe für das 2D-Mesh ab."""
        explicit = getattr(self.settings, "fem_line_panel_height", None)
        if explicit is not None:
            try:
                value = float(explicit)
            except (TypeError, ValueError):
                value = None
            else:
                if value > 0.0:
                    return max(min(value, physical_height), 1e-3)

        base_resolution = self._mesh_resolution or float(getattr(self.settings, "resolution", 0.5) or 0.5)
        line_height = max(base_resolution * 0.1, 1e-3)
        if physical_height > 0.0:
            line_height = min(line_height, physical_height)
        return line_height

    def _create_panel_points(self, center: np.ndarray, width: float, height: float, azimuth_rad: float) -> np.ndarray:
        half_w = max(width / 2.0, 0.05)
        half_h = max(height / 2.0, 1e-3)
        local_points = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=float,
        )
        c = math.cos(azimuth_rad)
        s = math.sin(azimuth_rad)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        rotated = local_points @ rotation.T
        rotated += center.reshape(1, 2)
        return rotated

    @staticmethod
    def _compute_polygon_perimeter(points: np.ndarray) -> float:
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[0] < 2:
            return 0.0
        diffs = np.diff(points, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        closing = np.linalg.norm(points[0] - points[-1])
        return float(edge_lengths.sum() + closing)

    @staticmethod
    def _compute_polygon_area(points: np.ndarray) -> float:
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[0] < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    def _collect_speaker_panels(self) -> tuple[List[SpeakerPanel], List[SpeakerCabinetObstacle]]:
        """Erzeugt Listen von Membranflächen und Gehäusehindernissen."""
        panels: List[SpeakerPanel] = []
        cabinets: List[SpeakerCabinetObstacle] = []
        speaker_arrays = getattr(self.settings, "speaker_arrays", None)
        if not isinstance(speaker_arrays, dict):
            return panels, cabinets

        self._log_debug(f"Starte Panel-Erstellung für {len(speaker_arrays)} Lautsprecher-Arrays.")
        data_container = self._get_data_container()
        cabinet_lookup = self._build_cabinet_lookup(data_container)
        default_width = float(getattr(self.settings, "fem_default_panel_width", 0.6) or 0.6)
        default_height = float(getattr(self.settings, "fem_default_panel_height", 0.5) or 0.5)
        default_depth = float(getattr(self.settings, "fem_default_cabinet_depth", default_width) or default_width)

        for array_key, speaker_array in speaker_arrays.items():
            array_hidden = getattr(speaker_array, "hide", False)
            array_muted = getattr(speaker_array, "mute", False)
            if array_hidden:
                self._log_debug(f"[Panels] Array '{array_key}' ist ausgeblendet – übersprungen.")
                continue

            raw_names_primary = getattr(speaker_array, "source_polar_pattern", None)
            names_list = self._normalize_source_name_sequence(raw_names_primary)
            if not names_list:
                raw_names_secondary = getattr(speaker_array, "source_type", None)
                names_list = self._normalize_source_name_sequence(raw_names_secondary)
            if not names_list:
                self._log_debug(f"[Panels] Array '{array_key}' hat keine gültigen source‑Namen – übersprungen.")
                continue

            self._log_debug(
                f"[Panels] Array '{array_key}': {len(names_list)} Quellen, Konfiguration={getattr(speaker_array, 'configuration', None)}"
            )

            num_sources = len(names_list)
            if num_sources == 0:
                continue
            xs = np.asarray(
                self._normalize_sequence(
                    getattr(speaker_array, "source_position_calc_x", getattr(speaker_array, "source_position_x", None)),
                    num_sources,
                ),
                dtype=float,
            )
            ys = np.asarray(
                self._normalize_sequence(
                    getattr(speaker_array, "source_position_calc_y", getattr(speaker_array, "source_position_y", None)),
                    num_sources,
                ),
                dtype=float,
            )
            azimuths = np.deg2rad(
                self._normalize_sequence(getattr(speaker_array, "source_azimuth", [0.0] * num_sources), num_sources)
            )
            gains = np.asarray(
                self._normalize_sequence(getattr(speaker_array, "gain", [0.0] * num_sources), num_sources),
                dtype=float,
            )
            source_levels = np.asarray(
                self._normalize_sequence(getattr(speaker_array, "source_level", [0.0] * num_sources), num_sources),
                dtype=float,
            )

            a_source_db = float(getattr(self.settings, "a_source_db", 0.0) or 0.0)
            level_adjust_db = source_levels + gains + a_source_db

            for idx, raw_name in enumerate(names_list):
                speaker_name = self._decode_speaker_name(raw_name)
                try:
                    width, height, depth = self._resolve_cabinet_dimensions(
                        speaker_name, cabinet_lookup, default_width, default_height, default_depth
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"FEM Panel kann nicht erstellt werden: {exc}"
                    ) from exc
                line_height = self._derive_line_height(height)
                center = np.array([xs[idx], ys[idx]], dtype=float)
                azimuth = azimuths[idx] if idx < len(azimuths) else 0.0
                points = self._create_panel_points(center, width, line_height, azimuth)
                identifier = f"{array_key}_{idx}"
                perimeter = self._compute_polygon_perimeter(points)
                area = abs(self._compute_polygon_area(points))
                panels.append(
                    SpeakerPanel(
                        identifier=identifier,
                        array_key=str(array_key),
                        points=points,
                        width=width,
                        height=height,
                        perimeter=perimeter,
                        area=area,
                        center=center,
                        azimuth_rad=azimuth,
                        level_adjust_db=float(level_adjust_db[idx]),
                        speaker_name=speaker_name,
                        line_height=line_height,
                        depth=depth,
                        is_muted=array_muted,
                    )
                )
                self._log_debug(
                    f"[Panels] Fläche {identifier}: Speaker={speaker_name}, Breite={width:.2f} m, Höhe={height:.2f} m, "
                    f"LineHeight={line_height*1000:.1f} mm, Area={area:.3f} m², Mittelpunkt=({center[0]:.2f},{center[1]:.2f}), "
                    f"Azimut={np.rad2deg(azimuth):.1f}°"
                )
                self._log_debug(
                    f"[Panels] → Panel {identifier} @ ({center[0]:.2f}, {center[1]:.2f}), "
                    f"Breite={width:.2f} m, Höhe={height:.2f} m, LineHeight={line_height*1000:.1f} mm, Perimeter={perimeter:.3f} m, "
                    f"Azimut={np.rad2deg(azimuth):.1f}°"
                )
                if depth > 0.0:
                    cabinet_points = self._create_panel_points(center, width, depth, azimuth)
                    cabinets.append(
                        SpeakerCabinetObstacle(
                            identifier=f"{identifier}_cabinet",
                            array_key=str(array_key),
                            points=cabinet_points,
                            width=width,
                            depth=depth,
                            center=center,
                            azimuth_rad=azimuth,
                            speaker_name=speaker_name,
                            is_muted=array_muted,
                        )
                    )
                    self._log_debug(
                        f"[Cabinet] Hindernis {identifier}_cabinet: Speaker={speaker_name}, Breite={width:.2f} m, Tiefe={depth:.2f} m."
                    )
        self._log_debug(f"Panel-Erstellung abgeschlossen: {len(panels)} Panels erzeugt, {len(cabinets)} Gehäuse erkannt.")
        return panels, cabinets

    def _require_gmsh(self):
        if gmsh is None:
            raise ImportError(
                "Gmsh (inkl. Python-Bindings) ist nicht installiert. "
                "Bitte `pip install gmsh` oder `conda install -c conda-forge gmsh` im FEM-Umfeld ausführen."
            )
        if gmshio is None and dolfinx_gmsh is None:
            raise ImportError(
                "dolfinx bietet keine gmsh-Schnittstelle (gmshio/gmsh) – "
                "bitte ein vollständiges fenics-dolfinx-Paket installieren."
            )

    def _gmsh_model_to_mesh(self, gdim: int):
        """Konvertiert das aktuelle gmsh-Modell in ein dolfinx-Mesh."""
        if gmshio is not None:
            return gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)
        if dolfinx_gmsh is not None:
            mesh_data = dolfinx_gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)
            return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
        raise ImportError(
            "dolfinx.io.gmshio bzw. dolfinx.io.gmsh sind nicht verfügbar – "
            "bitte fenics-dolfinx korrekt installieren."
        )

    def _gmsh_add_polygon(self, factory, points: np.ndarray, mesh_size: float) -> Tuple[int, List[int]]:
        point_tags: List[int] = []
        for px, py in points:
            point_tags.append(factory.addPoint(float(px), float(py), 0.0, mesh_size))
        curve_tags: List[int] = []
        for idx in range(len(point_tags)):
            start = point_tags[idx]
            end = point_tags[(idx + 1) % len(point_tags)]
            curve_tags.append(factory.addLine(start, end))
        loop_tag = factory.addCurveLoop(curve_tags)
        return loop_tag, curve_tags

    def _generate_gmsh_mesh(
        self,
        width: float,
        length: float,
        resolution: float,
        panels: List[SpeakerPanel],
        cabinets: List[SpeakerCabinetObstacle],
    ) -> Tuple["mesh.Mesh", Optional["mesh.MeshTags"], Optional["mesh.MeshTags"]]:
        self._require_gmsh()
        initialized_here = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        try:
            gmsh.model.add("lfo_fem_domain")
            try:
                gmsh.option.setNumber("General.Terminal", 0)
            except Exception:
                pass
            factory = gmsh.model.occ
            mesh_size = max(resolution, 0.05)
            self._log_debug(f"[Gmsh] Erzeuge Domain {width:.2f}x{length:.2f} m mit {len(panels)} Speaker-Panels, mesh_size={mesh_size:.3f}.")

            half_w = width / 2.0
            half_l = length / 2.0
            outer_points = np.array(
                [
                    [-half_w, -half_l],
                    [half_w, -half_l],
                    [half_w, half_l],
                    [-half_w, half_l],
                ],
                dtype=float,
            )
            outer_loop, outer_curves = self._gmsh_add_polygon(factory, outer_points, mesh_size)

            cabinet_surfaces: List[Tuple[int, int]] = []
            panel_surfaces: List[Tuple[int, int]] = []

            for cabinet in cabinets:
                loop_tag, _ = self._gmsh_add_polygon(factory, cabinet.points, mesh_size)
                surface = factory.addPlaneSurface([loop_tag])
                cabinet_surfaces.append((2, surface))

            for panel in panels:
                if panel.depth > 0.0:
                    continue
                loop_tag, _ = self._gmsh_add_polygon(factory, panel.points, mesh_size)
                surface = factory.addPlaneSurface([loop_tag])
                panel_surfaces.append((2, surface))

            outer_surface = factory.addPlaneSurface([outer_loop])
            cut_result = factory.cut(
                [(2, outer_surface)],
                cabinet_surfaces + panel_surfaces,
                removeObject=True,
                removeTool=True,
            )
            factory.synchronize()

            if cut_result and cut_result[0]:
                surface_entities = [tag for (_, tag) in cut_result[0]]
            else:
                surface_entities = [outer_surface]
            if not surface_entities:
                surface_entities = [outer_surface]
            domain_phys = gmsh.model.addPhysicalGroup(2, surface_entities, 1)
            gmsh.model.setPhysicalName(2, domain_phys, "domain")

            outer_phys = gmsh.model.addPhysicalGroup(1, outer_curves, self._outer_boundary_tag)
            gmsh.model.setPhysicalName(1, outer_phys, "outer")

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(2)

            domain, cell_tags, facet_tags = self._gmsh_model_to_mesh(gdim=2)
            return domain, cell_tags, facet_tags
        finally:
            try:
                gmsh.clear()
            except Exception:
                pass
            if initialized_here:
                gmsh.finalize()

    def _estimate_dofs_for_resolution(self, width: float, length: float, resolution: float, degree: int) -> int:
        """Schätzt die DOF-Anzahl für gegebenes Mesh grob ab."""
        nx = max(2, int(round(width / resolution)))
        ny = max(2, int(round(length / resolution)))
        cells = nx * ny
        dofs_per_cell = int((degree + 1) * (degree + 2) / 2)  # Dreieckselement
        return cells * dofs_per_cell

    def _calculate_fem_mesh_resolution(self, frequencies: list[float]) -> float:
        """Leitet eine FEM-Auflösung aus max. Frequenz & DOF-Limit ab."""
        base_resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)
        width = float(self.settings.width)
        length = float(self.settings.length)
        degree = int(getattr(self.settings, "fem_polynomial_degree", 2))
        base_points_per_wavelength = float(getattr(self.settings, "fem_points_per_wavelength", 10.0) or 10.0)
        base_max_dofs = int(getattr(self.settings, "fem_max_dofs", 250_000) or 250_000)
        base_min_resolution = float(getattr(self.settings, "fem_min_resolution", 0.05) or 0.05)
        max_resolution = float(getattr(self.settings, "fem_max_resolution_limit", base_resolution * 4) or (base_resolution * 4))

        (
            points_per_wavelength,
            max_dofs,
            min_resolution,
        ) = self._derive_mesh_limits(
            width=width,
            length=length,
            base_resolution=base_resolution,
            base_points_per_wavelength=base_points_per_wavelength,
            base_max_dofs=base_max_dofs,
            base_min_resolution=base_min_resolution,
            frequencies=frequencies,
        )

        fem_resolution = base_resolution
        if frequencies:
            f_max = max(frequencies)
            if f_max > 0 and points_per_wavelength > 0:
                temperature = getattr(self.settings, "temperature", 20.0)
                speed_of_sound = self.functions.calculate_speed_of_sound(temperature)
                wavelength = speed_of_sound / f_max
                ideal_resolution = wavelength / points_per_wavelength
                fem_resolution = min(base_resolution, ideal_resolution)

        fem_resolution = max(min_resolution, fem_resolution)
        fem_resolution = min(max_resolution, fem_resolution)

        estimated_dofs = self._estimate_dofs_for_resolution(width, length, fem_resolution, degree)
        if estimated_dofs > max_dofs:
            scale = math.sqrt(estimated_dofs / max_dofs)
            fem_resolution = min(max_resolution, fem_resolution * scale)

        return fem_resolution

    def _derive_mesh_limits(
        self,
        width: float,
        length: float,
        base_resolution: float,
        base_points_per_wavelength: float,
        base_max_dofs: int,
        base_min_resolution: float,
        frequencies: list[float],
        ) -> tuple[float, int, float]:
        """Leitet adaptive Mesh-Grenzwerte aus Domain-Größe und Frequenzen ab."""

        area = max(width * length, 1e-6)
        highest_frequency = float(max(frequencies)) if frequencies else None

        points_per_wavelength = float(base_points_per_wavelength)
        max_dofs = int(base_max_dofs)
        min_resolution = float(base_min_resolution)

        # Größere Flächen → gröbere Diskretisierung erzwingen
        area_reference = 100.0  # 10 m x 10 m
        area_scale = math.sqrt(area / area_reference)
        if area_scale > 1.0:
            points_per_wavelength = max(4.0, points_per_wavelength / area_scale)
            min_resolution = max(min_resolution, base_resolution / 2.0)
            max_dofs = max(60_000, int(max_dofs / area_scale))

        # Sehr hohe Frequenzen → PPW deckeln, um exponentielles Wachstum zu vermeiden
        if highest_frequency:
            if highest_frequency >= 8000.0:
                points_per_wavelength = min(points_per_wavelength, 6.0)
            elif highest_frequency >= 4000.0:
                points_per_wavelength = min(points_per_wavelength, 8.0)

        return points_per_wavelength, max_dofs, min_resolution

    def _build_domain(self, frequencies: list[float]):
        """Erzeugt das FEM-Mesh auf Basis der Settings.

        - Das Modell arbeitet in der XY-Ebene (2D) mit Dreieckselementen.
        - Die Auflösung orientiert sich an der höchsten Frequenz & DOF-Limit
        - Die Funktion speichert Mesh und Funktionsraum für Folgeaufrufe.
        """
        desired_resolution = self._calculate_fem_mesh_resolution(frequencies)
        if self._mesh is not None:
            current_resolution = getattr(self, "_mesh_resolution", None)
            if current_resolution is not None and math.isclose(current_resolution, desired_resolution, rel_tol=1e-6, abs_tol=1e-6):
                self._ensure_dof_cache()
                return
            self._mesh = None
            self._function_space = None

        width = float(self.settings.width)
        length = float(self.settings.length)
        resolution = desired_resolution

        panels, cabinets = self._collect_speaker_panels()
        self._panels = panels
        self._cabinet_obstacles = cabinets
        self._log_debug(f"[Domain] Anzahl Panels für FEM-Domain: {len(panels)}.")
        self._log_debug(f"[Domain] Anzahl Gehäuse-Hindernisse: {len(cabinets)}.")
        if not panels:
            self._log_debug("[Domain] Keine Speaker-Panels vorhanden – es werden nur äußere Ränder meshing.")

        mesh_obj = None
        cell_tags = None
        facet_tags = None
        try:
            mesh_obj, cell_tags, facet_tags = self._generate_gmsh_mesh(width, length, resolution, panels, cabinets)
        except ImportError:
            if panels:
                raise
            # Fallback: einfaches Rechteck-Mesh ohne Lautsprecheröffnungen
            nx = max(2, int(round(width / resolution)))
            ny = max(2, int(round(length / resolution)))
            p_min = np.array([-(width / 2.0), -(length / 2.0)], dtype=np.float64)
            p_max = np.array([width / 2.0, length / 2.0], dtype=np.float64)
            mesh_obj = mesh.create_rectangle(
                MPI.COMM_WORLD,
                [p_min, p_max],
                [nx, ny],
                cell_type=mesh.CellType.triangle,
            )

        self._mesh = mesh_obj
        self._cell_tags = cell_tags
        self._facet_tags = facet_tags
        self._mesh_resolution = resolution

        element_degree = int(getattr(self.settings, "fem_polynomial_degree", 2))
        
        # Prüfe, ob dolfinx mit komplexer Arithmetik kompiliert wurde
        is_complex_dolfinx = np.issubdtype(default_scalar_type, np.complexfloating)
        
        try:
            functionspace = getattr(fem, "functionspace")
        except AttributeError:
            functionspace = None

        if callable(functionspace):
            try:
                self._function_space = functionspace(
                    self._mesh,
                    ("CG", element_degree),
                    dtype=default_scalar_type,
                )
            except TypeError:
                # Fallback ohne dtype-Argument
                self._function_space = functionspace(
                    self._mesh,
                    ("CG", element_degree),
                )
        else:
            element = ufl.FiniteElement("CG", self._mesh.ufl_cell(), element_degree)
            self._function_space = fem.FunctionSpace(self._mesh, element)
        
        num_dofs = self._function_space.dofmap.index_map.size_local
        self._mesh_version += 1
        self._cache_dof_coordinates()

    def _cache_dof_coordinates(self):
        """Speichert DOF-Koordinaten und optional einen KD-Tree für Nachbarschaftssuchen."""
        if self._function_space is None:
            self._dof_coordinates = None
            self._dof_coords_xy = None
            self._dof_tree = None
            self._dof_cache_version = None
            return

        coords = self._function_space.tabulate_dof_coordinates()
        self._dof_coordinates = coords
        if coords.ndim == 2 and coords.shape[1] >= 2:
            self._dof_coords_xy = coords[:, :2].copy()
        else:
            self._dof_coords_xy = None

        if self._dof_coords_xy is not None and cKDTree is not None:
            try:
                self._dof_tree = cKDTree(self._dof_coords_xy)
            except Exception:
                self._dof_tree = None
        else:
            self._dof_tree = None

        self._dof_cache_version = self._mesh_version

        self._log_debug(f"[Mesh] DOF-Cache aktualisiert (Version {self._dof_cache_version}).")
        
    def _ensure_dof_cache(self):
        if self._function_space is None:
            return
        if self._dof_cache_version == self._mesh_version and self._dof_coordinates is not None:
            return
        self._cache_dof_coordinates()

    def _get_dof_coords_xy(self) -> Optional[np.ndarray]:
        self._ensure_dof_cache()
        return self._dof_coords_xy

    def _find_nearby_dofs(self, point_xy: np.ndarray, radius: float) -> np.ndarray:
        self._ensure_dof_cache()
        if self._dof_coords_xy is None:
            return np.array([], dtype=int)
        if self._dof_tree is not None:
            try:
                indices = self._dof_tree.query_ball_point(point_xy, r=radius)
                return np.asarray(indices, dtype=int)
            except Exception:
                pass
        distances = np.linalg.norm(self._dof_coords_xy - point_xy, axis=1)
        return np.where(distances <= radius)[0]

    def _find_nearest_dof(self, point_xy: np.ndarray) -> Optional[int]:
        self._ensure_dof_cache()
        if self._dof_coords_xy is None:
            return None
        if self._dof_tree is not None:
            try:
                distance, index = self._dof_tree.query(point_xy, k=1)
                return int(index)
            except Exception:
                pass
        distances = np.linalg.norm(self._dof_coords_xy - point_xy, axis=1)
        if distances.size == 0:
            return None
        return int(np.argmin(distances))

    # ------------------------------------------------------------------
    # Balloon-Daten & Quellenanregung
    # ------------------------------------------------------------------
    def _get_injection_radius(self) -> float:
        if getattr(self.settings, "fem_balloon_injection_radius", None):
            return float(self.settings.fem_balloon_injection_radius)
        if self._mesh_resolution:
            return float(self._mesh_resolution)
        return float(getattr(self.settings, "resolution", 0.5) or 0.5)

    def _normalize_sequence(self, values, target_length: int, default: float = 0.0) -> list[float]:
        if values is None:
            return [default] * target_length
        if isinstance(values, (float, int, np.floating, np.integer)):
            return [float(values)] * target_length
        normalized = []
        for idx in range(target_length):
            if idx < len(values):
                normalized.append(float(values[idx]))
            else:
                normalized.append(float(values[-1]) if values else default)
        return normalized

    def _get_balloon_dataset_for_frequency(self, speaker_name: str, frequency: float) -> Optional[dict]:
        if self._data_container is None or not speaker_name:
            return None

        cache_key = f"{speaker_name}:{frequency:.3f}"
        cached = self._balloon_data_cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch_balloon_data(use_averaged: bool) -> Optional[dict]:
            try:
                data = self._data_container.get_balloon_data(speaker_name, use_averaged=use_averaged)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            return data

        def _normalize_dataset(source: dict, freq: float) -> Optional[dict]:
            magnitude = source.get("magnitude")
            phase = source.get("phase")
            vertical_angles = source.get("vertical_angles")
            if magnitude is None or vertical_angles is None:
                return None

            magnitude_np = np.asarray(magnitude)
            vertical_np = np.asarray(vertical_angles, dtype=float)
            phase_np = np.asarray(phase, dtype=float) if phase is not None else None

            # Averaged (2D) Daten → direkt übernehmen
            if magnitude_np.ndim == 2:
                dataset_dict = {
                    "magnitude": magnitude_np.astype(float, copy=False),
                    "phase": phase_np.astype(float, copy=False) if phase_np is not None else None,
                    "vertical_angles": vertical_np,
                }
            # Original (3D) Daten → Frequenz-Slice wählen
            elif magnitude_np.ndim == 3:
                freqs = source.get("freqs")
                if freqs is None:
                    freqs = source.get("frequencies")
                if freqs is None:
                    return None
                freqs_arr = np.asarray(freqs, dtype=float).flatten()
                if freqs_arr.size == 0:
                    return None
                nearest_idx = int(np.argmin(np.abs(freqs_arr - float(freq))))
                nearest_idx = max(0, min(nearest_idx, magnitude_np.shape[2] - 1))
                mag_slice = magnitude_np[:, :, nearest_idx]
                phase_slice = None
                if phase_np is not None and phase_np.ndim == 3:
                    phase_slice = phase_np[:, :, nearest_idx]
                dataset_dict = {
                    "magnitude": mag_slice.astype(float, copy=False),
                    "phase": phase_slice.astype(float, copy=False) if phase_slice is not None else None,
                    "vertical_angles": vertical_np,
                }
            else:
                return None

            horizontal = source.get("horizontal_angles")
            if horizontal is not None:
                dataset_dict["horizontal_angles"] = np.asarray(horizontal, dtype=float)
            return dataset_dict

        raw_dataset = _fetch_balloon_data(False)
        if raw_dataset is None:
            self._log_debug(
                "[BalloonDataset] "
                f"Keine Balloon-Daten für '{speaker_name}' @ {frequency:.1f} Hz gefunden."
            )
            self._balloon_data_cache[cache_key] = None
            return None

        dataset_dict = _normalize_dataset(raw_dataset, frequency)

        if dataset_dict is None:
            self._log_debug(
                "[BalloonDataset] "
                f"Dataset für '{speaker_name}' @ {frequency:.1f} Hz konnte nicht normalisiert werden."
            )
            self._balloon_data_cache[cache_key] = None
            return None

        if not self._validate_balloon_dataset(speaker_name, frequency, dataset_dict):
            self._balloon_data_cache[cache_key] = None
            return None

        self._debug_balloon_dataset_overview(speaker_name, frequency, dataset_dict)
        self._debug_balloon_cardioid_sample(speaker_name, frequency, dataset_dict)
        self._balloon_data_cache[cache_key] = dataset_dict
        return dataset_dict

    def _interpolate_balloon_values(
        self,
        dataset: dict,
        azimuths_deg: np.ndarray,
        elevations_deg: np.ndarray,
        ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        magnitude = dataset.get("magnitude")
        phase = dataset.get("phase")
        vertical_angles = dataset.get("vertical_angles")
        if magnitude is None or vertical_angles is None:
            return np.zeros_like(azimuths_deg), None

        azimuths_norm = np.round(azimuths_deg).astype(int) % magnitude.shape[1]
        elevations = elevations_deg.astype(float)
        vertical_angles = np.asarray(vertical_angles, dtype=float)

        mag_values = np.zeros_like(azimuths_deg, dtype=float)
        phase_values = np.zeros_like(azimuths_deg, dtype=float) if phase is not None else None

        mask_below = elevations <= vertical_angles[0]
        mask_above = elevations >= vertical_angles[-1]
        mask_interp = ~(mask_below | mask_above)

        if np.any(mask_below):
            indices = azimuths_norm[mask_below]
            mag_values[mask_below] = magnitude[0, indices]
            if phase_values is not None:
                phase_values[mask_below] = phase[0, indices]

        if np.any(mask_above):
            indices = azimuths_norm[mask_above]
            mag_values[mask_above] = magnitude[-1, indices]
            if phase_values is not None:
                phase_values[mask_above] = phase[-1, indices]

        if np.any(mask_interp):
            elev_vals = elevations[mask_interp]
            az_vals = azimuths_norm[mask_interp]
            lower_idx = np.searchsorted(vertical_angles, elev_vals, side="right") - 1
            lower_idx = np.clip(lower_idx, 0, len(vertical_angles) - 1)
            upper_idx = np.clip(lower_idx + 1, 0, len(vertical_angles) - 1)
            denom = vertical_angles[upper_idx] - vertical_angles[lower_idx]
            denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
            t = (elev_vals - vertical_angles[lower_idx]) / denom
            mag_lower = magnitude[lower_idx, az_vals]
            mag_upper = magnitude[upper_idx, az_vals]
            mag_values[mask_interp] = mag_lower + t * (mag_upper - mag_lower)
            if phase_values is not None:
                phase_lower = phase[lower_idx, az_vals]
                phase_upper = phase[upper_idx, az_vals]
                phase_values[mask_interp] = phase_lower + t * (phase_upper - phase_lower)

        return mag_values, phase_values

    def _validate_balloon_dataset(
        self,
        speaker_name: str,
        frequency: float,
        dataset: dict,
    ) -> bool:
        """Prüft Balloon-Datensätze auf minimale Konsistenz, bevor sie genutzt werden."""
        magnitude = dataset.get("magnitude")
        vertical_angles = dataset.get("vertical_angles")
        phase = dataset.get("phase")
        issues: list[str] = []

        if not isinstance(magnitude, np.ndarray) or magnitude.size == 0:
            issues.append("keine Magnituden")
        elif magnitude.ndim < 2:
            issues.append(f"unerwartete Magnitude-Dimension ({magnitude.ndim})")
        elif not np.all(np.isfinite(magnitude)):
            issues.append("Magnitude enthält NaN/Inf")

        if phase is not None:
            if not isinstance(phase, np.ndarray):
                issues.append("Phase ist kein Array")
            elif phase.shape != magnitude.shape:
                issues.append(
                    f"Phase-Shape {phase.shape} passt nicht zu Magnitude {magnitude.shape}"
                )

        if not isinstance(vertical_angles, np.ndarray) or vertical_angles.size == 0:
            issues.append("keine vertikalen Winkel")
        elif magnitude is not None and vertical_angles.shape[0] != magnitude.shape[0]:
            issues.append(
                f"vertical_angles ({vertical_angles.shape[0]}) != magnitude-elevations ({magnitude.shape[0]})"
            )

        if issues:
            self._log_debug(
                "[BalloonCheck] "
                f"{speaker_name or 'unbekannt'} @ {frequency:.1f} Hz: "
                + "; ".join(issues)
            )
            return False
        return True

    def _debug_balloon_dataset_overview(
        self,
        speaker_name: str,
        frequency: float,
        dataset: dict,
    ) -> None:
        if not getattr(self.settings, "fem_debug_logging", True):
            return
        mag = dataset.get("magnitude")
        phase = dataset.get("phase")
        vertical = dataset.get("vertical_angles")
        if isinstance(mag, np.ndarray):
            mag_stats = (
                float(np.nanmin(mag)),
                float(np.nanmax(mag)),
                mag.shape,
            )
        else:
            mag_stats = None
        if isinstance(phase, np.ndarray):
            phase_stats = (
                float(np.nanmin(phase)),
                float(np.nanmax(phase)),
                phase.shape,
            )
        else:
            phase_stats = None
        vertical_info = (
            float(vertical[0]),
            float(vertical[-1]),
            vertical.shape[0],
        ) if isinstance(vertical, np.ndarray) and vertical.size else None
        self._log_debug(
            "[BalloonDataset] "
            f"{speaker_name or 'unbekannt'} @ {frequency:.1f} Hz | "
            f"mag_stats={mag_stats} phase_stats={phase_stats} vertical={vertical_info}"
        )

    def _debug_balloon_source_projection(
        self,
        speaker_name: str,
        frequency: float,
        dataset: Optional[dict],
        dof_indices: np.ndarray,
        distances: np.ndarray,
        azimuths: np.ndarray,
        elevations: np.ndarray,
        mag_values: np.ndarray,
        phase_values: np.ndarray,
        wave_amplitude: np.ndarray,
    ) -> None:
        """Ausführliche Debug-Ausgabe für Balloon → Punktquellen-Projektion."""
        if not getattr(self.settings, "fem_debug_logging", True):
            return
        num_points = dof_indices.size
        sample_end = min(4, num_points)
        dataset_shape = None
        vertical_range = None
        if isinstance(dataset, dict):
            magnitude = dataset.get("magnitude")
            vertical = dataset.get("vertical_angles")
            if isinstance(magnitude, np.ndarray):
                dataset_shape = magnitude.shape
            if isinstance(vertical, np.ndarray) and vertical.size:
                vertical_range = (float(vertical[0]), float(vertical[-1]))
        self._log_debug(
            "[BalloonDebug] "
            f"{speaker_name} @ {frequency:.1f} Hz | DOFs={num_points} "
            f"dataset={dataset_shape} vert_range={vertical_range}"
        )
        if sample_end == 0:
            return
        sample = slice(0, sample_end)
        self._log_debug(
            "[BalloonDebug] "
            f"indices={dof_indices[sample].tolist()} "
            f"dist={np.round(distances[sample], 3).tolist()} m "
            f"az={np.round(azimuths[sample], 1).tolist()}° "
            f"el={np.round(elevations[sample], 1).tolist()}°"
        )
        self._log_debug(
            "[BalloonDebug] "
            f"mag(dB)={np.round(mag_values[sample], 2).tolist()} "
            f"phase(deg)={np.round(phase_values[sample], 1).tolist()} "
            f"|wave|={np.round(np.abs(wave_amplitude[sample]), 4).tolist()}"
        )

    def _debug_balloon_level_chain(
        self,
        speaker_name: str,
        frequency: float,
        base_level_db: float,
        measurement_correction_db: float,
        resulting_level_db: float,
        base_pressure_pa: float,
        relative_magnitude_sample: float,
    ) -> None:
        if not getattr(self.settings, "fem_debug_logging", True):
            return
        self._log_debug(
            "[BalloonLevel] "
            f"{speaker_name or 'unbekannt'} @ {frequency:.1f} Hz | "
            f"base_level={base_level_db:.2f} dB, "
            f"meas_corr={measurement_correction_db:.2f} dB, "
            f"result={resulting_level_db:.2f} dB, "
            f"base_pressure={base_pressure_pa:.3f} Pa, "
            f"rel_mag_sample={relative_magnitude_sample:.3f}"
        )

    def _debug_balloon_angle_chain(
        self,
        speaker_name: str,
        frequency: float,
        source_orientation_deg: float,
        raw_horizontal_angles: np.ndarray,
        final_balloon_angles: np.ndarray,
        elevations: np.ndarray,
    ) -> None:
        if not getattr(self.settings, "fem_debug_logging", True):
            return
        sample = slice(0, min(4, raw_horizontal_angles.size))
        self._log_debug(
            "[BalloonAngles] "
            f"{speaker_name or 'unbekannt'} @ {frequency:.1f} Hz | "
            f"source_az={source_orientation_deg:.1f}° "
            f"raw={np.round(raw_horizontal_angles[sample],1).tolist()}° "
            f"balloon={np.round(final_balloon_angles[sample],1).tolist()}° "
            f"elev={np.round(elevations[sample],1).tolist()}°"
        )

    def _debug_balloon_cardioid_sample(
        self,
        speaker_name: str,
        frequency: float,
        dataset: dict,
    ) -> None:
        if not getattr(self.settings, "fem_debug_logging", True):
            return
        magnitude = dataset.get("magnitude")
        phase = dataset.get("phase")
        vertical = dataset.get("vertical_angles")
        if magnitude is None or vertical is None:
            return
        mag_np = np.asarray(magnitude)
        phase_np = np.asarray(phase) if phase is not None else None
        v_idx = int(np.abs(np.asarray(vertical, dtype=float)).argmin())
        def _value_at(angle_deg: int):
            h_idx = angle_deg % 360
            mag_val = float(mag_np[v_idx, h_idx])
            phase_val = float(phase_np[v_idx, h_idx]) if phase_np is not None else None
            return mag_val, phase_val
        mag_front, phase_front = _value_at(0)
        mag_back, phase_back = _value_at(180)
        self._log_debug(
            "[BalloonCardioid] "
            f"{speaker_name or 'unbekannt'} @ {frequency:.1f} Hz | "
            f"mag0°={mag_front:.2f} dB, mag180°={mag_back:.2f} dB, "
            f"phase0°={phase_front if phase_front is not None else 'n/a'}, "
            f"phase180°={phase_back if phase_back is not None else 'n/a'}"
        )

    def _sample_balloon_on_axis(self, dataset: dict) -> tuple[Optional[float], Optional[float]]:
        magnitude = dataset.get("magnitude")
        phase = dataset.get("phase")
        vertical = dataset.get("vertical_angles")
        if magnitude is None or vertical is None:
            return None, None
        mag_np = np.asarray(magnitude)
        phase_np = np.asarray(phase) if phase is not None else None
        v_idx = int(np.abs(np.asarray(vertical, dtype=float)).argmin())
        mag0 = float(mag_np[v_idx, 0]) if mag_np.ndim >= 2 else None
        phase0 = float(phase_np[v_idx, 0]) if phase_np is not None and phase_np.ndim >= 2 else None
        return mag0, phase0


    def _build_balloon_point_sources(
        self,
        V,
        frequency: float,
        allow_complex: bool,
    ) -> Optional[dict]:
        """Erzeugt verteilte Punktquellen mit Balloon-Magnitude/Phase."""
        if self._data_container is None or not hasattr(self.settings, "speaker_arrays"):
            return None

        coords_xy = self._get_dof_coords_xy()
        if coords_xy is None:
            return None

        num_dofs = coords_xy.shape[0]
        from dolfinx import default_scalar_type

        contributions = np.zeros(num_dofs, dtype=default_scalar_type)
        handled_keys: list[tuple[str, int]] = []
        source_positions: list[list[float]] = []

        injection_radius = self._get_injection_radius()
        reference_distance = float(
            getattr(self.settings, "fem_balloon_reference_distance", 1.0) or 1.0
        )
        omega = 2.0 * np.pi * frequency
        rho = getattr(self.settings, "air_density", 1.2)
        is_complex_fem = np.issubdtype(default_scalar_type, np.complexfloating)

        for array_id, speaker_array in self.settings.speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            array_identifier = str(array_id)

            names_list = self._normalize_source_name_sequence(
                getattr(speaker_array, "source_polar_pattern", None)
            )
            if not names_list:
                names_list = self._normalize_source_name_sequence(
                    getattr(speaker_array, "source_type", None)
                )
            if not names_list:
                continue

            num_sources = len(names_list)
            xs = np.asarray(
                self._normalize_sequence(
                    getattr(
                        speaker_array,
                        "source_position_calc_x",
                        getattr(speaker_array, "source_position_x", None),
                    ),
                    num_sources,
                ),
                dtype=float,
            )
            ys = np.asarray(
                self._normalize_sequence(
                    getattr(
                        speaker_array,
                        "source_position_calc_y",
                        getattr(speaker_array, "source_position_y", None),
                    ),
                    num_sources,
                ),
                dtype=float,
            )
            zs = np.asarray(
                self._normalize_sequence(
                    getattr(
                        speaker_array,
                        "source_position_calc_z",
                        getattr(speaker_array, "source_position_z", None),
                    ),
                    num_sources,
                ),
                dtype=float,
            )
            gains = np.asarray(
                self._normalize_sequence(getattr(speaker_array, "gain", [0.0] * num_sources), num_sources),
                dtype=float,
            )
            source_levels = np.asarray(
                self._normalize_sequence(getattr(speaker_array, "source_level", [0.0] * num_sources), num_sources),
                dtype=float,
            )
            delays_ms = np.asarray(
                self._normalize_sequence(getattr(speaker_array, "delay", [0.0] * num_sources), num_sources),
                dtype=float,
            )

            source_time_entries = getattr(speaker_array, "source_time", None)
            if isinstance(source_time_entries, (int, float)):
                source_time_ms = np.asarray(
                    [float(source_time_entries) + delays_ms[i] for i in range(num_sources)],
                    dtype=float,
                )
            else:
                source_time_seq = self._normalize_sequence(source_time_entries, num_sources)
                source_time_ms = np.asarray(
                    [source_time_seq[i] + delays_ms[i] for i in range(num_sources)],
                    dtype=float,
                )
            source_time_s = source_time_ms / 1000.0

            level_corrections_db = source_levels + gains
            measurement_distance = float(
                getattr(self.settings, "fem_balloon_measurement_distance", 1.0) or 1.0
            )
            reference_distance = float(
                getattr(self.settings, "fem_balloon_reference_distance", 1.0) or 1.0
            )
            measurement_scale = reference_distance / measurement_distance
            if measurement_scale <= 0.0:
                measurement_scale = 1.0
            correction_offset_db = float(
                getattr(self.settings, "fem_debug_level_offset_db", 60.0) or 60.0
            )
            measurement_correction_db = (
                20.0 * math.log10(measurement_scale) + correction_offset_db
            )
            p_ref = 20e-6
            base_pressures = p_ref * (10 ** (level_corrections_db / 20.0))
            source_azimuth = np.deg2rad(
                self._normalize_sequence(getattr(speaker_array, "source_azimuth", [0.0] * num_sources), num_sources)
            )
            source_polarity = self._normalize_sequence(
                getattr(speaker_array, "source_polarity", [0] * num_sources),
                num_sources,
            )

            for idx, speaker_name in enumerate(names_list):
                point_xy = np.array([xs[idx], ys[idx]], dtype=float)
                dataset = self._get_balloon_dataset_for_frequency(speaker_name, frequency)
                if dataset is None:
                    continue

                handled_keys.append((array_identifier, idx))
                nearby_indices = self._find_nearby_dofs(point_xy, radius=injection_radius)
                if nearby_indices.size == 0:
                    nearest_idx = self._find_nearest_dof(point_xy)
                    if nearest_idx is None:
                        continue
                    nearby_indices = np.array([nearest_idx], dtype=int)

                dof_points = coords_xy[nearby_indices]
                dx = dof_points[:, 0] - xs[idx]
                dy = dof_points[:, 1] - ys[idx]
                horizontal = np.sqrt(dx**2 + dy**2)
                max_dofs = int(getattr(self.settings, "fem_balloon_max_dofs", 12) or 12)
                if nearby_indices.size > max_dofs:
                    order = np.argsort(horizontal)
                    select = order[:max_dofs]
                    nearby_indices = nearby_indices[select]
                    dof_points = dof_points[select]
                    dx = dx[select]
                    dy = dy[select]
                    horizontal = horizontal[select]
                z_distance = -zs[idx]
                distances = np.sqrt(horizontal**2 + z_distance**2)
                distances = np.maximum(distances, 1e-3)
                source_angles = np.arctan2(dy, dx)
                raw_horizontal_deg = (np.degrees(source_angles)) % 360.0
                azimuths = (raw_horizontal_deg + np.degrees(source_azimuth[idx])) % 360.0
                azimuths = (360.0 - azimuths) % 360.0
                azimuths = (azimuths + 90.0) % 360.0
                elevations = np.degrees(np.arctan2(z_distance, horizontal + 1e-9))

                mag_values, phase_values = self._interpolate_balloon_values(dataset, azimuths, elevations)
                if mag_values is None or mag_values.size == 0:
                    continue
                phase_values = phase_values if phase_values is not None else np.zeros_like(mag_values)

                p_ref = 20e-6
                relative_magnitude = 10 ** ((mag_values + measurement_correction_db) / 20.0)
                polar_phase_rad = np.radians(phase_values)
                phase_argument = polar_phase_rad + (2.0 * np.pi * frequency * source_time_s[idx])
                distance_scale = reference_distance / distances
                wave_amplitude = (
                    relative_magnitude
                    * base_pressures[idx]
                    * distance_scale
                    * np.exp(1j * phase_argument)
                )
                if mag_values.size > 0:
                    resulting_level_db = (
                        float(level_corrections_db[idx])
                        + measurement_correction_db
                        + float(mag_values[0])
                    )
                    self._debug_balloon_level_chain(
                        speaker_name=speaker_name,
                        frequency=frequency,
                        base_level_db=float(level_corrections_db[idx]),
                        measurement_correction_db=measurement_correction_db,
                        resulting_level_db=resulting_level_db,
                        base_pressure_pa=float(base_pressures[idx]),
                        relative_magnitude_sample=float(relative_magnitude[0]),
                    )
                self._debug_balloon_angle_chain(
                    speaker_name=speaker_name,
                    frequency=frequency,
                    source_orientation_deg=float(np.degrees(source_azimuth[idx])),
                    raw_horizontal_angles=raw_horizontal_deg,
                    final_balloon_angles=azimuths,
                    elevations=elevations,
                )
                sigma = max(injection_radius / 2.0, 0.05)
                weights = np.exp(-0.5 * (horizontal / (sigma + 1e-9)) ** 2)
                weight_sum = float(np.sum(weights))
                if weight_sum > 0.0:
                    weights /= weight_sum
                else:
                    weights = np.full_like(horizontal, 1.0 / len(horizontal))
                if source_polarity[idx]:
                    wave_amplitude = -wave_amplitude

                if is_complex_fem and allow_complex:
                    values = wave_amplitude.astype(default_scalar_type)
                else:
                    values = np.real(wave_amplitude).astype(default_scalar_type)
                self._debug_balloon_source_projection(
                    speaker_name=speaker_name or f"{array_identifier}_{idx}",
                    frequency=frequency,
                    dataset=dataset,
                    dof_indices=nearby_indices,
                    distances=distances,
                    azimuths=azimuths,
                    elevations=elevations,
                    mag_values=mag_values,
                    phase_values=phase_values,
                    wave_amplitude=wave_amplitude,
                )

                contributions[nearby_indices] += values * weights
                source_positions.append([xs[idx], ys[idx]])

        non_zero = np.flatnonzero(np.abs(contributions) > 0)
        if non_zero.size == 0:
            return None

        self._last_point_source_positions = np.array(source_positions, dtype=float) if source_positions else None
        return {
            "indices": non_zero.astype(np.int32),
            "values": contributions[non_zero],
            "handled_keys": handled_keys,
        }

    def _build_panel_neumann_loads(self, frequency: float) -> list[tuple[int, complex]]:
        loads: list[tuple[int, complex]] = []
        if not self._speaker_panel_tags or not self._panels:
            return loads

        rho = getattr(self.settings, "air_density", 1.2)
        omega = 2.0 * np.pi * frequency
        for panel in self._panels:
            tag = self._speaker_panel_tags.get(panel.identifier)
            if tag is None or not panel.speaker_name:
                continue
            dataset = self._get_balloon_dataset_for_frequency(panel.speaker_name, frequency)
            if dataset is None:
                continue
            mag_db, phase_deg = self._sample_balloon_on_axis(dataset)
            if mag_db is None:
                continue
            p_amp = 20e-6 * 10 ** (mag_db / 20.0)
            phase_rad = math.radians(phase_deg) if phase_deg is not None else 0.0
            p_complex = p_amp * np.exp(1j * phase_rad)

            effective_height = panel.line_height if panel.line_height > 0.0 else panel.height
            area = panel.area if abs(panel.area) > 1e-6 else panel.width * effective_height
            if area <= 0.0:
                continue

            # Balloondruck: p_balloon @ r_balloon → v_n = p / (iωρ)
            v_normal = p_complex / (1j * omega * rho)
            neumann_value = 1j * omega * rho * v_normal  # = p_complex
            loads.append((tag, neumann_value))
        return loads

    def _build_panel_dirichlet_bcs(self, frequency: float) -> list:
        bcs = []
        if self._function_space is None or not self._panels:
            return bcs

        coords_xy = self._get_dof_coords_xy()
        if coords_xy is None:
            return bcs

        tol = max(self._mesh_resolution or 0.1, 0.05)

        for panel in self._panels:
            if not panel.speaker_name or panel.is_muted:
                continue
            dataset = self._get_balloon_dataset_for_frequency(panel.speaker_name, frequency)
            if dataset is None:
                continue
            mag_db, phase_deg = self._sample_balloon_on_axis(dataset)
            if mag_db is None:
                continue
            phase_rad = math.radians(phase_deg) if phase_deg is not None else 0.0
            temperature = getattr(self.settings, "temperature", 20.0)
            speed_of_sound = self.functions.calculate_speed_of_sound(temperature)
            k = 2.0 * np.pi * frequency / speed_of_sound
            measurement_distance = float(
                getattr(self.settings, "fem_balloon_measurement_distance", 1.0) or 1.0
            )
            offsets_db = getattr(self.settings, "fem_debug_level_offset_db", 0.0) or 0.0
            panel_ref_distance = float(
                getattr(self.settings, "fem_panel_reference_distance", 0.5) or 0.5
            )
            total_db = mag_db + panel.level_adjust_db + offsets_db
            p_balloon = 20e-6 * 10 ** (total_db / 20.0)
            scale = measurement_distance / max(panel_ref_distance, 1e-3)
            line_height = panel.line_height if panel.line_height > 0.0 else panel.height
            physical_height = max(panel.height, 1e-3)
            line_gain = math.sqrt(max(line_height, 1e-3) / physical_height)
            p_panel = p_balloon * scale * line_gain
            distance_phase = np.exp(1j * k * (panel_ref_distance - measurement_distance))
            p_complex = p_panel * distance_phase * np.exp(1j * phase_rad)

            rel = coords_xy - panel.center.reshape(1, 2)
            c = math.cos(-panel.azimuth_rad)
            s = math.sin(-panel.azimuth_rad)
            rot = np.array([[c, -s], [s, c]], dtype=float)
            local = rel @ rot.T
            half_w = panel.width / 2.0 + tol
            line_half = max(panel.line_height / 2.0, 1e-3)
            transition = min(tol * 0.2, 0.1)
            half_h = line_half + transition
            line_center = panel.depth / 2.0 if panel.depth > 0.0 else 0.0
            mask = (np.abs(local[:, 0]) <= half_w) & (np.abs(local[:, 1] - line_center) <= half_h)
            dof_indices = np.nonzero(mask)[0].astype(np.int32)
            if dof_indices.size == 0:
                self._log_debug(
                    f"[Dirichlet] Keine DOFs für Panel {panel.identifier} gefunden – Panel übersprungen."
                )
                continue

            bc_value = fem.Constant(self._mesh, default_scalar_type(p_complex))
            bc = fem.dirichletbc(bc_value, dof_indices, self._function_space)
            bcs.append(bc)
            self._log_debug(
                f"[Dirichlet] Panel {panel.identifier} (Speaker={panel.speaker_name}) "
                f"→ DOFs={dof_indices.size}, |p|={abs(p_complex):.2f} Pa, phase={phase_deg or 0:.1f}°"
            )
        return bcs

        
    def _prepare_python_include_path(self):
        if self._python_include_path_prepared:
            return

        try:
            include_dir = Path(sysconfig.get_path("include"))
        except Exception:
            include_dir = None

        if not include_dir:
            self._python_include_path_prepared = True
            return

        include_dir_str = str(include_dir)
        if " " not in include_dir_str:
            self._python_include_path_prepared = True
            return

        cache_root = Path.home() / ".cache" / "lfo_fem"
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache_root = Path.home()

        def _sanitize_path(target: Path, label: str) -> Path:
            if " " not in str(target):
                return target
            candidate = cache_root / label
            try:
                if candidate.exists() or candidate.is_symlink():
                    try:
                        same_target = candidate.is_symlink() and candidate.resolve() == target
                    except OSError:
                        same_target = False
                    if not same_target:
                        if candidate.is_symlink() or candidate.is_file():
                            candidate.unlink()
                        elif candidate.is_dir():
                            candidate = cache_root / f"{label}_{abs(hash(str(target)))}"
                if not candidate.exists():
                    candidate.symlink_to(target, target_is_directory=True)
            except OSError:
                return target
            return candidate

        sanitized_include = _sanitize_path(include_dir, "python_include")
        include_parent = include_dir.parent
        sanitized_include_parent = _sanitize_path(include_parent, "venv_include")

        sanitized_include_str = str(sanitized_include)
        sanitized_parent_str = str(sanitized_include_parent)

        cfg = sysconfig.get_config_vars()
        if cfg is not None:
            cfg["INCLUDEPY"] = sanitized_include_str
            cfg["CONFINCLUDEPY"] = sanitized_include_str
            cfg["CONFINCLUDEDIR"] = sanitized_parent_str
            cfg["INCLUDEDIR"] = sanitized_parent_str
            cfg["PLATINCLUDEDIR"] = sanitized_parent_str

        try:
            import distutils.sysconfig as du_sysconfig  # type: ignore
        except ImportError:
            du_sysconfig = None

        if du_sysconfig is not None:
            du_cfg = du_sysconfig.get_config_vars()
            if du_cfg is not None:
                du_cfg["INCLUDEPY"] = sanitized_include_str
                du_cfg["CONFINCLUDEPY"] = sanitized_include_str
                du_cfg["CONFINCLUDEDIR"] = sanitized_parent_str
                du_cfg["INCLUDEDIR"] = sanitized_parent_str
                du_cfg["PLATINCLUDEDIR"] = sanitized_parent_str

        include_dirs = [
            sanitized_parent_str,
            sanitized_include_str,
            str(sanitized_include_parent / "include"),
            str(sanitized_include_parent / "usr" / "local" / "include"),
        ]

        existing = os.environ.get("C_INCLUDE_PATH", "").split(os.pathsep) if os.environ.get("C_INCLUDE_PATH") else []
        for path in include_dirs:
            if path and path not in existing:
                existing.insert(0, path)
        os.environ["C_INCLUDE_PATH"] = os.pathsep.join([p for p in existing if p])

        sanitized_flag = f'-I"{sanitized_parent_str}"'
        extra_compile = cfg.get("CFLAGS", "") if cfg else ""
        if sanitized_flag not in extra_compile:
            cfg["CFLAGS"] = f'{extra_compile} {sanitized_flag}'.strip()

        if du_sysconfig is not None and du_cfg is not None:
            du_extra = du_cfg.get("CFLAGS", "")
            if sanitized_flag not in du_extra:
                du_cfg["CFLAGS"] = f'{du_extra} {sanitized_flag}'.strip()

        os.environ["CPPFLAGS"] = f'{sanitized_flag} {os.environ.get("CPPFLAGS", "")}'.strip()

        self._python_include_path_prepared = True

    def _solve_frequency(self, frequency: float) -> fem.Function:
        """Löst die Helmholtz-Gleichung für eine einzelne Frequenz.

        Die schwache Form umfasst:
        - Laplace-Term (∇p · ∇q) als Steifigkeitsanteil.
        - Massenanteil für k² mit negativem Vorzeichen.
        - Robin-Randbedingung zur Approximation eines absorbierenden Randes.

        Als Löser wird GMRES mit ILU-Präkonditionierer auf dem PETSc-Stack
        verwendet. Die Toleranzen sind relativ streng gewählt, damit die
        Ergebnisse mit dem klassischen Rechner vergleichbar bleiben.
        """

        V = self._function_space
        freq_label = f"{frequency:.2f}Hz"

        from dolfinx.fem.petsc import assemble_matrix, assemble_vector

        with self._time_block("solve_frequency_setup_forms", freq_label):
            pressure_trial = ufl.TrialFunction(V)
            pressure_test = ufl.TestFunction(V)
            
            # 🌡️ Temperaturabhängige Schallgeschwindigkeit (wird in UiSettings berechnet)
            speed_of_sound = self.settings.speed_of_sound
            k = 2.0 * np.pi * frequency / float(speed_of_sound)
            
            absorption = getattr(self.settings, "fem_boundary_absorption", 1.0)
            
            dx = ufl.Measure("dx", domain=self._mesh)
            if self._facet_tags is not None:
                ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._facet_tags)
            else:
                ds = ufl.Measure("ds", domain=self._mesh)
            
            is_complex = np.issubdtype(default_scalar_type, np.complexfloating)
            
            if is_complex:
                boundary_term = 1j * k * absorption * ufl.inner(pressure_trial, pressure_test)
                if self._facet_tags is not None:
                    boundary_term = boundary_term * ds(self._outer_boundary_tag)
                else:
                    boundary_term = boundary_term * ds

                a_form = fem.form(
                    ufl.inner(ufl.grad(pressure_trial), ufl.grad(pressure_test)) * dx
                    - (k ** 2) * ufl.inner(pressure_trial, pressure_test) * dx
                    + boundary_term
                )
                
                L_form, rhs_payload = self._assemble_rhs(
                    V, pressure_test, frequency, allow_complex=True
                )
            else:
                a_form = fem.form(
                    (
                        ufl.inner(ufl.grad(pressure_trial), ufl.grad(pressure_test))
                        - (k ** 2) * pressure_trial * pressure_test
                    )
                    * dx
                )
                
                L_form, rhs_payload = self._assemble_rhs(
                    V, pressure_test, frequency, allow_complex=False
                )
        self._raise_if_frequency_cancelled()

        bcs = self._build_panel_dirichlet_bcs(frequency)

        with self._time_block("solve_frequency_assemble_matrix", freq_label):
            A = assemble_matrix(a_form, bcs=bcs)
            A.assemble()
        self._raise_if_frequency_cancelled()

        with self._time_block("solve_frequency_assemble_vector", freq_label):
            b = assemble_vector(L_form)
            if bcs:
                fem.apply_lifting(b, [a_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if bcs:
                fem.set_bc(b, bcs)
            try:
                rhs_norm_initial = b.norm()
            except Exception:
                rhs_norm_initial = None
            if rhs_norm_initial is not None:
                self._log_debug(f"[Solve {freq_label}] RHS-Norm vor Quellen: {rhs_norm_initial:.3e}")
        self._raise_if_frequency_cancelled()

        with self._time_block("solve_frequency_apply_point_sources", freq_label):
            self._apply_rhs_payload_to_vector(b, rhs_payload, V)
            try:
                rhs_norm_after = b.norm()
            except Exception:
                rhs_norm_after = None
            if rhs_norm_after is not None:
                self._log_debug(f"[Solve {freq_label}] RHS-Norm nach Quellen: {rhs_norm_after:.3e}")
        self._raise_if_frequency_cancelled()

        use_direct_solver = getattr(self.settings, "fem_use_direct_solver", True)

        with self._time_block("solve_frequency_setup_solver", freq_label):
            solver = PETSc.KSP().create(self._mesh.comm)
            solver.setOperators(A)
            
            if use_direct_solver:
                solver.setType("preonly")
                solver.getPC().setType("lu")
            else:
                solver.setType("gmres")
                solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=10000)
                solver.getPC().setType("ilu")
            
            solution = fem.Function(V, name="pressure")
        self._raise_if_frequency_cancelled()

        with self._time_block("solve_frequency_solve_system", freq_label):
            solver.solve(b, solution.x.petsc_vec)
            solution.x.scatter_forward()
            sol_array = solution.x.array
            if sol_array is not None and sol_array.size:
                sol_norm = float(np.linalg.norm(sol_array))
                sol_max = float(np.max(np.abs(sol_array)))
                self._log_debug(
                    f"[Solve {freq_label}] Lösung: ||p||₂={sol_norm:.3e}, max|p|={sol_max:.3e}"
                )
            
            if not use_direct_solver:
                reason = solver.getConvergedReason()
                if reason < 0:
                    raise RuntimeError(
                        f"Iterativer FEM-Solver konvergierte nicht (Grund: {reason})"
                    )
        self._raise_if_frequency_cancelled()

        return solution

    def _assemble_rhs(self, V, test_function, frequency: float, allow_complex: bool = True):
        """Baut rechte Seite und Punktquellenliste zusammen.

        Jeder Lautsprecher wird als Punktquelle modelliert. Sollte ein Punkt
        außerhalb des Mesh liegen, wird er (stillschweigend) ignoriert. Für
        realistischere Quellen (z.B. Membranflächen) müsste hier eine Fläche
        oder Linienquelle implementiert werden.
        
        WICHTIG: Diese Methode erstellt eine leere RHS-Form und eine Liste
        von Punktquellen-Daten. Die Punktquellen werden nach dem Assembly
        direkt in den Vektor eingefügt (moderne dolfinx-Methode).
        """
        # WICHTIG: Import am Anfang, da wir default_scalar_type benötigen
        from dolfinx import default_scalar_type

        dx = ufl.Measure("dx", domain=self._mesh)
        zero_source = fem.Constant(self._mesh, default_scalar_type(0.0))
        rhs_form = ufl.inner(zero_source, test_function) * dx
        self._last_point_source_positions = None

        # Variante 2 (verteilte Punktquellen) ist nun exklusiv aktiv; Neumann-Randquellen
        # werden unabhängig von den Settings nicht mehr verwendet.
        if getattr(self.settings, "fem_use_boundary_sources", False):
            self._log_debug(
                "[RHS] Neumann-Randquellen sind deaktiviert – verwende Punktquellen."
            )

        self._last_point_source_positions = None
        return fem.form(rhs_form), {}

    def _apply_rhs_payload_to_vector(self, b_vector, rhs_payload, V):
        """Wendet verteilte Balloon-Quellen und Fallback-Punktquellen auf den RHS-Vektor an."""
        if not rhs_payload:
            return

        # Wenn boundary_sources verwendet werden, wurde die Form bereits in _assemble_rhs integriert
        if isinstance(rhs_payload, dict) and rhs_payload.get("boundary_sources"):
            # Die Neumann-BC-Terme sind bereits in der Form enthalten, nichts zu tun
            return

        distributed = rhs_payload.get("distributed") if isinstance(rhs_payload, dict) else None
        if distributed:
            indices = distributed.get("indices")
            values = distributed.get("values")
            if indices is not None and values is not None and len(indices) > 0:
                try:
                    b_vector.setValues(indices.astype(np.int32), values, addv=True)
                except Exception:
                    for idx, val in zip(indices, values):
                        b_vector.setValue(int(idx), val, addv=True)

        try:
            arr = b_vector.getArray(readonly=True)
        except Exception:
            arr = None
        if arr is not None and arr.size > 0:
            arr_real = np.real(arr)
            arr_abs = np.abs(arr)
            self._log_debug(
                f"[RHS Debug] Entries={arr.size}, min(real)={float(np.min(arr_real)):.3e}, "
                f"max(real)={float(np.max(arr_real)):.3e}, max|.|={float(np.max(arr_abs)):.3e}"
            )
            try:
                b_vector.restoreArray(arr)
            except Exception:
                pass

        b_vector.assemble()
        
    def _speaker_positions_2d(self, speaker_array) -> np.ndarray:
        # Debug: Zeige verfügbare Attribute
        has_calc_x = hasattr(speaker_array, 'source_position_calc_x')
        has_calc_y = hasattr(speaker_array, 'source_position_calc_y')
        has_orig_x = hasattr(speaker_array, 'source_position_x')
        has_orig_y = hasattr(speaker_array, 'source_position_y')
        
        # Versuche zuerst berechnete Positionen zu holen
        xs = getattr(speaker_array, 'source_position_calc_x', None)
        ys = getattr(speaker_array, 'source_position_calc_y', None)
        
        
        # Fallback auf Original-Positionen
        if xs is None or (isinstance(xs, (list, tuple)) and len(xs) == 0) or (isinstance(xs, np.ndarray) and xs.size == 0):
            xs = getattr(speaker_array, 'source_position_x', [0.0])
        
        if ys is None or (isinstance(ys, (list, tuple)) and len(ys) == 0) or (isinstance(ys, np.ndarray) and ys.size == 0):
            ys = getattr(speaker_array, 'source_position_y', [0.0])
        
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        
        
        return np.column_stack([xs, ys])

    def _phase_factor(self, speaker_array, index: int, frequency: float) -> complex:
        delay_ms = float(getattr(speaker_array, "delay", 0.0))
        time_entries = getattr(speaker_array, "source_time", 0.0)

        if isinstance(time_entries, (int, float, np.integer, np.floating)):
            source_time_ms = float(time_entries)
        else:
            try:
                source_time_ms = float(time_entries[index])
            except (IndexError, TypeError, ValueError):
                source_time_ms = float(time_entries[0]) if time_entries else 0.0

        total_time_s = (source_time_ms + delay_ms) / 1000.0
        phase_shift = 2.0 * np.pi * frequency * total_time_s

        polarity = 1.0
        if hasattr(speaker_array, "source_polarity") and speaker_array.source_polarity[index]:
            polarity = -1.0

        phase_rad = np.deg2rad(
            getattr(speaker_array, "source_azimuth", [0.0])[index]
            if isinstance(getattr(speaker_array, "source_azimuth", [0.0]), (list, tuple, np.ndarray))
            else getattr(speaker_array, "source_azimuth", 0.0)
        )

        return polarity * np.exp(1j * (phase_shift + phase_rad))

    def _extract_pressure_and_phase(
        self, solution: fem.Function
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = solution.x.array
        pressure = np.abs(values)
        p_ref = 20e-6
        spl = self.functions.mag2db((pressure / p_ref) + 1e-12)
        phase = np.angle(values, deg=True)

        if getattr(self.settings, "fem_debug_logging", True):
            try:
                front_point = self._sample_grid_pressure(solution, np.array([0.0, 20.0]))
                back_point = self._sample_grid_pressure(solution, np.array([0.0, -20.0]))
                mag_front = self.functions.mag2db((np.abs(front_point) / p_ref) + 1e-12)
                mag_back = self.functions.mag2db((np.abs(back_point) / p_ref) + 1e-12)
                self._log_debug(
                    "[FEMCardioid] "
                    f"Grid 0°={mag_front:.2f} dB, 180°={mag_back:.2f} dB"
                )
            except Exception:
                pass

        return pressure, spl, phase

    def _sample_grid_pressure(self, solution: fem.Function, point_xy: np.ndarray) -> complex:
        coords = self._get_dof_coords_xy()
        if coords is None:
            return 0.0
        idx = self._find_nearest_dof(point_xy)
        if idx is None:
            return 0.0
        return solution.x.array[idx]

    def _compute_particle_velocity(
        self, solution: fem.Function, frequency: float
        ) -> Optional[np.ndarray]:
        """Leitet Partikelgeschwindigkeit aus dem Druckgradienten ab.

        Die Projektion des Druckgradienten erfolgt über ein separates
        Variationsproblem. Das Ergebnis liefert komplexe Partikel-
        geschwindigkeiten (û) im Frequenzbereich.
        """

        if not hasattr(self.settings, "air_density"):
            return None

        rho = float(self.settings.air_density)
        kinematic_factor = 1.0 / (1j * 2.0 * np.pi * frequency * rho)

        # API-Update: VectorFunctionSpace → functionspace mit Vektor-Element
        degree = self._function_space.element.basix_element.degree
        mesh_dim = self._mesh.geometry.dim  # 2D oder 3D
        
        # Erstelle Vektor-Funktionsraum mit neuer API
        try:
            # Neue DOLFINx API (0.10.0+): Tuple-Notation für Vektor-Elemente
            V_vec = fem.functionspace(
                self._mesh,
                ("Lagrange", degree, (mesh_dim,))
            )
        except (TypeError, AttributeError):
            # Fallback für ältere Versionen (sollte nicht mehr nötig sein)
            V_vec = fem.VectorFunctionSpace(
                self._mesh,
                ("CG", degree)
            )

        grad_p = fem.Function(V_vec, name="grad_p")

        # Erstelle Projektionsformen für L2-Projektion des Gradienten
        v = ufl.TrialFunction(V_vec)
        w = ufl.TestFunction(V_vec)
        
        # WICHTIG: Im komplexen Modus verwendet UFL automatisch konjugierte Testfunktionen
        # ufl.inner() garantiert korrekte Behandlung
        a_proj = fem.form(ufl.inner(v, w) * ufl.dx)
        L_proj = fem.form(ufl.inner(ufl.grad(solution), w) * ufl.dx)

        from dolfinx.fem.petsc import assemble_matrix, assemble_vector
        A = assemble_matrix(a_proj)
        A.assemble()
        b = assemble_vector(L_proj)

        solver = PETSc.KSP().create(self._mesh.comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.setTolerances(rtol=1e-10, atol=1e-12)
        solver.getPC().setType("ilu")

        solver.solve(b, grad_p.x.petsc_vec)
        grad_p.x.scatter_forward()

        velocity = kinematic_factor * grad_p.x.array
        return velocity.copy()

    # ------------------------------------------------------------------
    # Daten-Export
    # ------------------------------------------------------------------
    def _assign_primary_soundfield_results(self, frequencies, fem_results):
        if not fem_results:
            self._log_debug("[Assign Results] Keine FEM-Ergebnisse vorhanden.")
            return

        primary_frequency = self._select_primary_frequency(frequencies, fem_results)
        if primary_frequency is None:
            self._log_debug(f"[Assign Results] Keine primäre Frequenz gefunden. Verfügbare: {list(fem_results.keys())[:5]}")
            return

        self._log_debug(f"[Assign Results] Verwende primäre Frequenz: {primary_frequency:.2f} Hz")
        
        try:
            # WICHTIG: Verwende get_soundfield_grid() (liefert Druckwerte in Pascal).
            sound_field_x, sound_field_y, sound_field_pressure = self.get_soundfield_grid(primary_frequency)
            
            # Berechne SPL für Debug-Ausgabe
            p_ref = 20e-6
            sound_field_spl = self.functions.mag2db((np.abs(sound_field_pressure) / p_ref) + 1e-12)
            
            # Debug: Größe und Werte prüfen
            x_shape = sound_field_x.shape if isinstance(sound_field_x, np.ndarray) else (len(sound_field_x),)
            y_shape = sound_field_y.shape if isinstance(sound_field_y, np.ndarray) else (len(sound_field_y),)
            p_shape = sound_field_pressure.shape if isinstance(sound_field_pressure, np.ndarray) else (len(sound_field_pressure),)
            
            p_min = float(np.min(sound_field_pressure)) if isinstance(sound_field_pressure, np.ndarray) else float("nan")
            p_max = float(np.max(sound_field_pressure)) if isinstance(sound_field_pressure, np.ndarray) else float("nan")
            spl_min = float(np.min(sound_field_spl)) if isinstance(sound_field_spl, np.ndarray) else float("nan")
            spl_max = float(np.max(sound_field_spl)) if isinstance(sound_field_spl, np.ndarray) else float("nan")
            
            self._log_debug(
                f"[Assign Results] Grid-Daten: X={x_shape}, Y={y_shape}, P={p_shape} | "
                f"SPL [min={spl_min:.2f} dB, max={spl_max:.2f} dB] | "
                f"|p| [min={p_min:.3e}, max={p_max:.3e}]"
            )
            
            # Konvertiere zu Listen für Kompatibilität mit dem Plot-System
            self.calculation_spl["sound_field_x"] = sound_field_x.tolist() if isinstance(sound_field_x, np.ndarray) else sound_field_x
            self.calculation_spl["sound_field_y"] = sound_field_y.tolist() if isinstance(sound_field_y, np.ndarray) else sound_field_y
            self.calculation_spl["sound_field_p"] = sound_field_pressure.tolist() if isinstance(sound_field_pressure, np.ndarray) else sound_field_pressure
            
            # Prüfe ob Daten tatsächlich geschrieben wurden
            has_x = "sound_field_x" in self.calculation_spl and len(self.calculation_spl["sound_field_x"]) > 0
            has_y = "sound_field_y" in self.calculation_spl and len(self.calculation_spl["sound_field_y"]) > 0
            has_p = "sound_field_p" in self.calculation_spl and len(self.calculation_spl["sound_field_p"]) > 0
            
            self._log_debug(
                f"[Assign Results] Daten geschrieben: X={has_x}, Y={has_y}, P={has_p} | "
                f"P-Länge={len(self.calculation_spl.get('sound_field_p', []))}"
            )
            
        except Exception as e:
            self._log_debug(f"[Assign Results] FEHLER beim Zuweisen der Ergebnisse: {e}")
            import traceback
            traceback.print_exc()
            return

    def _select_primary_frequency(self, frequencies, fem_results) -> Optional[float]:
        if not fem_results:
            return None

        target_frequency = getattr(self, "_resolved_fem_frequency", None)
        if target_frequency is None:
            target_frequency = getattr(self.settings, "fem_calculate_frequency", None)
        if target_frequency is not None:
            try:
                target_frequency = float(target_frequency)
            except (TypeError, ValueError):
                target_frequency = None

        if target_frequency is not None:
            if target_frequency in fem_results:
                return target_frequency
            for freq_key in fem_results.keys():
                if math.isclose(freq_key, target_frequency, rel_tol=1e-6, abs_tol=1e-3):
                    return freq_key

        try:
            return float(next(iter(fem_results.keys())))
        except StopIteration:
            return None

    def get_soundfield_grid(self, frequency: float, decimals: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Erzeugt X, Y, und Druckwerte auf dem GLEICHEN Grid wie SoundfieldCalculator.py.
        
        Die Methode verwendet exakt die gleichen Grid-Parameter (width, length, resolution)
        und ordnet FEM-Knoten den nächstgelegenen Grid-Punkten zu.
        """
        fem_results = self.calculation_spl.get("fem_simulation")
        if not fem_results:
            raise RuntimeError(
                "Es liegen keine FEM-Ergebnisse vor. Bitte zuerst die Berechnung ausführen."
            )

        freq_key = float(frequency)
        freq_data = fem_results.get(freq_key)
        if freq_data is None:
            for key in fem_results.keys():
                if math.isclose(key, freq_key, rel_tol=1e-6, abs_tol=1e-3):
                    freq_data = fem_results[key]
                    break

        if freq_data is None:
            raise KeyError(
                f"Für die Frequenz {frequency} Hz wurden keine FEM-Ergebnisse gefunden."
            )

        points = freq_data["points"]
        pressure = freq_data["pressure"]

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Punktkoordinaten besitzen nicht genügend Dimensionen.")

        # ============================================================
        # EXAKT das gleiche Grid wie SoundfieldCalculator.py erstellen
        # ============================================================
        width = float(self.settings.width)
        length = float(self.settings.length)
        resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)
        
        sound_field_x = np.arange((width / 2 * -1), 
                                 ((width / 2) + resolution), 
                                 resolution)
        sound_field_y = np.arange((length / 2 * -1), 
                                 ((length / 2) + resolution), 
                                 resolution)
        

        # Initialisiere Grids für Mittelwertbildung
        pressure_sum = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=pressure.dtype)
        count_grid = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=int)

        # Ordne jeden FEM-Knoten dem nächsten Grid-Punkt zu
        coords_xy = points[:, :2]
        x_coords = coords_xy[:, 0]
        y_coords = coords_xy[:, 1]
        
        # Debug: Zeige FEM-Koordinaten-Bereich
        
        # Berechne Grid-Indizes durch Runden auf nächsten Grid-Punkt
        # Index = round((coordinate - grid_start) / resolution)
        x_start = sound_field_x[0]
        y_start = sound_field_y[0]
        
        
        x_indices = np.round((x_coords - x_start) / resolution).astype(int)
        y_indices = np.round((y_coords - y_start) / resolution).astype(int)
        
        # Clamp auf gültigen Bereich (falls FEM-Knoten außerhalb des Grids liegen)
        x_indices = np.clip(x_indices, 0, len(sound_field_x) - 1)
        y_indices = np.clip(y_indices, 0, len(sound_field_y) - 1)
        

        # Konvertiere 2D-Indizes zu flachen Indizes für np.add.at
        # flat_index = y * width + x
        flat_indices = y_indices * len(sound_field_x) + x_indices
        
        # Debug: Überprüfe Array-Shapes vor np.add.at
        
        # Verwende np.add.at für effizienten Accumulator (mehrere Knoten → gleicher Grid-Punkt)
        np.add.at(pressure_sum.ravel(), flat_indices, pressure)
        np.add.at(count_grid.ravel(), flat_indices, 1)
        
        # Debug: Zeige Averaging-Statistik
        unique_counts = np.unique(count_grid[count_grid > 0])
        if len(unique_counts) > 1:
            max_count = np.max(count_grid)
            avg_count = np.mean(count_grid[count_grid > 0])
            num_averaged = np.sum(count_grid > 1)
        
        # Mittelwert bilden: Summe / Anzahl
        with np.errstate(invalid='ignore', divide='ignore'):
            pressure_grid = pressure_sum / count_grid
        pressure_grid[count_grid == 0] = np.nan
        
        pressure_grid = self._apply_2p5d_correction(
            pressure_grid,
            sound_field_x,
            sound_field_y,
        )
        
        if np.iscomplexobj(pressure_grid):
            pressure_magnitude = np.abs(pressure_grid)
        else:
            pressure_magnitude = pressure_grid
        
        pressure_magnitude = self._apply_air_absorption_to_grid(
            pressure_magnitude,
            sound_field_x,
            sound_field_y,
            freq_data.get("source_positions"),
            frequency,
        )
        
        return sound_field_x, sound_field_y, pressure_magnitude
    def _apply_2p5d_correction(
        self,
        pressure_grid: np.ndarray,
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
    ) -> np.ndarray:
        """Skaliert das 2D-Ergebnis auf ein 2.5D-Feld (~1/r-Abfall)."""
        if pressure_grid is None:
            return pressure_grid
        if not getattr(self.settings, "fem_enable_2p5d_correction", True):
            return pressure_grid
        if not self._panels:
            return pressure_grid

        centers = np.array([panel.center for panel in self._panels], dtype=float)
        if centers.size == 0:
            return pressure_grid

        reference_distance = float(
            getattr(self.settings, "fem_2p5d_reference_distance", 1.0) or 1.0
        )
        min_distance = float(
            getattr(self.settings, "fem_2p5d_min_distance", 0.25) or 0.25
        )

        grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
        min_dist_sq = None
        for center in centers:
            dx = grid_x - center[0]
            dy = grid_y - center[1]
            dist_sq = dx * dx + dy * dy
            if min_dist_sq is None:
                min_dist_sq = dist_sq
            else:
                min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        if min_dist_sq is None:
            return pressure_grid

        min_dist_sq = np.maximum(min_dist_sq, min_distance ** 2)
        radial_distance = np.sqrt(min_dist_sq)
        correction = np.sqrt(reference_distance / radial_distance)
        correction = np.clip(correction, 0.0, 1e6)
        pressure_grid = pressure_grid * correction
        return pressure_grid

    def _apply_air_absorption_to_grid(
        self,
        pressure_grid: np.ndarray,
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
        source_positions: Optional[np.ndarray],
        frequency: float,
        ) -> np.ndarray:
        """Skaliert das Grid nachträglich mit Luftdämpfung (ISO 9613-1)."""
        if pressure_grid is None or not getattr(self.settings, "use_air_absorption", False):
            return pressure_grid

        temperature = float(getattr(self.settings, "temperature", 20.0))
        humidity = float(getattr(self.settings, "humidity", 50.0))
        air_pressure = float(getattr(self.settings, "air_pressure", 101325.0))

        alpha = self.functions.calculate_air_absorption(
            float(frequency),
            temperature,
            humidity,
            air_pressure,
        )
        if alpha <= 0.0 or source_positions is None or len(source_positions) == 0:
            return pressure_grid

        grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        if cKDTree is not None and len(source_positions) >= 4:
            tree = cKDTree(source_positions)
            distances, _ = tree.query(grid_points, k=1)
        else:
            diff = grid_points[:, None, :] - source_positions[None, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)

        distance_grid = distances.reshape(grid_x.shape)
        attenuation = np.exp(-alpha * distance_grid)
        pressure_grid *= attenuation
        return pressure_grid

    def get_soundfield_grid_spl(self, frequency: float, decimals: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Erzeugt X, Y, und SPL-Werte auf dem Grid (in dB SPL).
        
        Diese Methode ruft get_soundfield_grid() auf und konvertiert die Druckwerte
        in SPL (Sound Pressure Level) in dB re 20 µPa.
        
        Args:
            frequency: Frequenz in Hz
            decimals: Anzahl Nachkommastellen für Koordinatenrundung
            
        Returns:
            Tuple (sound_field_x, sound_field_y, sound_field_spl)
            - sound_field_x: X-Koordinaten des Grids (numpy array)
            - sound_field_y: Y-Koordinaten des Grids (numpy array)
            - sound_field_spl: SPL-Werte in dB re 20µPa (2D numpy array)
        """
        sound_field_x, sound_field_y, pressure_grid = self.get_soundfield_grid(frequency, decimals)
        
        # Konvertiere Druck (Pa) zu SPL (dB re 20µPa)
        # SPL = 20 * log10(p / p_ref) wobei p_ref = 20e-6 Pa
        # Verwende die gleiche Funktion wie in der FEM-Berechnung
        p_ref = 20e-6
        spl_grid = self.functions.mag2db((np.abs(pressure_grid) / p_ref) + 1e-12)
        
        
        # Debug: Zeige SPL-Werte an spezifischen Punkten
        center_x_idx = np.argmin(np.abs(sound_field_x - 0.0))
        center_y_idx = np.argmin(np.abs(sound_field_y - 0.0))
        spl_at_source = spl_grid[center_y_idx, center_x_idx]
        
        x_10m_idx = np.argmin(np.abs(sound_field_x - 10.0))
        y_0m_idx = center_y_idx
        spl_at_10m = spl_grid[y_0m_idx, x_10m_idx]
        
        # Berechne erwarteten SPL-Abfall (6 dB pro Distanzverdopplung für Punktquelle)
        # Von 1m zu 10m: log2(10) = 3.32 Verdopplungen → -20 dB
        expected_drop = 20 * np.log10(10.0 / 1.0)
        
        
        return sound_field_x, sound_field_y, spl_grid

    def export_pressure_transposed(
        self, frequency: float, decimals: int = 6
        ) -> np.ndarray:
        """Erzeugt ein transponiertes Druckfeld-Array für eine Frequenz.

        Die Methode fasst die FEM-Ausgabe zu einem 2D-Gitter zusammen und
        liefert nur den Schalldruckbetrag pro Punkt zurück. Die Achsen werden
        intern sortiert und das Ergebnis transponiert, sodass die Form dem
        klassischen SoundField (`[nx, ny]`) entspricht.

        Parameters
        ----------
        frequency : float
            Frequenz, für die das Ergebnis extrahiert werden soll.
        decimals : int, optional
            Anzahl der Nachkommastellen für die Koordinatenrundung, um
            numerische Artefakte beim Gruppieren zu vermeiden (Standard: 6).

        Returns
        -------
        np.ndarray
            Transponiertes 2D-Array des Schalldruckbetrags (Pa).

        Raises
        ------
        KeyError
            Wenn keine Ergebnisse für die gewünschte Frequenz vorliegen.
        RuntimeError
            Wenn noch keine FEM-Ergebnisse berechnet wurden.
        ValueError
            Wenn die Punktdaten nicht mindestens zweidimensional sind.
        """

        _, _, pressure_grid = self.get_soundfield_grid(frequency, decimals=decimals)
        return pressure_grid.T


