"""FEM-basierte Berechnung des Schallfelds mit FEniCSx.

Dieses Modul folgt der gleichen Datenschnittstelle wie `SoundFieldCalculator`
und erweitert die Berechnung um eine Finite-Elemente-Lösung der Helmholtz-
Gleichung. Die Ergebnisse werden im selben `calculation_spl`-Dict abgelegt.

Hinweise zur Verwendung
-----------------------
- Erfordert eine funktionierende FEniCSx-Installation (dolfinx, ufl, mpi4py,
  petsc4py). Auf macOS empfiehlt sich die Installation über Conda/Mambaforge
  oder ein Docker-Image, da vorgebaute Wheels nur eingeschränkt verfügbar sind.
- Die Berechnung läuft im 3D-Setting (Volumen-Simulation) für korrekte
  sphärische Schallausbreitung (1/r²). Die Visualisierung erfolgt auf einer
  2D-Ebene (XY-Ebene bei Panel-Höhe) durch Extraktion und Interpolation.
- Lautsprecher werden als 3D-Flächen modelliert (Höhe und Breite aus Metadaten)
  mit Dirichlet-Randbedingungen auf der abstrahlenden Membranfläche.
- Randbedingungen: Alle Flächen (Boden, Decke, Wände) sind standardmäßig
  absorbierend (Absorptionskoeffizient 1.0) für reinen Direktschall.
  Konfigurierbar über `fem_boundary_absorption_{name}` Settings.
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
    """Repräsentiert eine Lautsprecherfläche (jetzt 3D-orientiert)."""

    identifier: str
    array_key: str
    points: Optional[np.ndarray]
    width: float
    height: float
    perimeter: float
    area: float
    center: np.ndarray  # [x, y, z]
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
        self._last_point_source_positions = None
        self._domain_height: Optional[float] = None
        self._mesh_version = 0
        self._dof_coordinates = None
        self._dof_coords_xy = None
        self._dof_coords_xyz = None
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
        self._boundary_tags = {
            "floor": 11,
            "ceiling": 12,
            "wall_x_min": 13,
            "wall_x_max": 14,
            "wall_y_min": 15,
            "wall_y_max": 16,
        }
        self._panel_tags: Dict[str, int] = {}  # Panel-Identifier → Facet-Tag
        self._cabinet_tags: Dict[str, int] = {}  # Cabinet-Identifier → Facet-Tag (für Dirichlet-RB)

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
        if isinstance(self.calculation_spl, dict):
            self.calculation_spl.pop("fdtd_simulation", None)
            self.calculation_spl.pop("fdtd_time_snapshots", None)

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
        panel_centers = self._get_panel_centers_array()
        
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
                "source_positions": panel_centers,
            }

            if compute_velocity and velocity is not None:
                fem_results[float(frequency)]["particle_velocity"] = velocity

            if self._frequency_progress_session is not None:
                self._frequency_progress_session.advance()
            self._raise_if_frequency_cancelled()

        self.calculation_spl["fem_simulation"] = fem_results
        
        with self._time_block("assign_primary_soundfield_results"):
            self._assign_primary_soundfield_results(frequencies, fem_results)

        self._frequency_progress_session = None
        self._frequency_progress_last_third_start = None
        timing_summary = self._summarize_timings()
        self.calculation_spl["fem_timings"] = timing_summary
        self._log_timings(timing_summary)

        if not fem_results:
            self._log_debug("[Ergebnisse] Keine FEM-Frequenzen vorhanden.")
        self._store_panel_metadata()
        
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
        """Für 3D wird die tatsächliche Membranhöhe verwendet."""
        if physical_height <= 0.0:
            return 1e-3
        return float(physical_height)

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
                # Panel-Höhe leicht über z=0 setzen, um Überlappung mit Boden-Fläche zu vermeiden
                # Verwende halbe Panel-Höhe, damit Panel zentriert ist
                panel_z = max(line_height / 2.0, 0.01)  # Mindestens 1 cm über Boden
                center = np.array([xs[idx], ys[idx], panel_z], dtype=float)
                azimuth = azimuths[idx] if idx < len(azimuths) else 0.0
                points = None
                identifier = f"{array_key}_{idx}"
                perimeter = 2.0 * (width + line_height)
                area = max(width * line_height, 1e-6)
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
                    f"LineHeight={line_height*1000:.1f} mm, Area={area:.3f} m², Mittelpunkt=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}), "
                    f"Azimut={np.rad2deg(azimuth):.1f}°"
                )
                self._log_debug(
                    f"[Panels] → Panel {identifier} @ ({center[0]:.2f}, {center[1]:.2f}), "
                    f"Breite={width:.2f} m, Höhe={height:.2f} m, LineHeight={line_height*1000:.1f} mm, Perimeter={perimeter:.3f} m, "
                    f"Azimut={np.rad2deg(azimuth):.1f}°"
                )
                if depth > 0.0:
                    cabinet_points = None
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

    def _store_panel_metadata(self):
        """Exportiert Panel-Metadaten für zeitbasierte Auswertungen."""
        if not isinstance(self.calculation_spl, dict):
            return
        panel_info = []
        for panel in getattr(self, "_panels", []) or []:
            try:
                panel_info.append(
                    {
                        "identifier": panel.identifier,
                        "center": panel.center.tolist(),
                        "width": float(panel.width),
                        "height": float(panel.height),
                        "line_height": float(panel.line_height),
                        "speaker_name": panel.speaker_name,
                        "level_adjust_db": float(panel.level_adjust_db),
                    }
                )
            except Exception:
                continue
        self.calculation_spl["fem_panel_info"] = panel_info

    def _record_panel_drive(self, frequency: float, panel: SpeakerPanel, pressure_complex: complex, phase_deg: Optional[float]):
        if not isinstance(self.calculation_spl, dict):
            return
        drive_store = self.calculation_spl.setdefault("fem_panel_drives", {})
        freq_key = float(frequency)
        freq_store = drive_store.setdefault(freq_key, {})
        freq_store[panel.identifier] = {
            "pressure_complex": complex(pressure_complex),
            "phase_deg": float(phase_deg) if phase_deg is not None else 0.0,
            "speaker_name": panel.speaker_name,
            "level_adjust_db": float(panel.level_adjust_db),
        }

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
        height: float,
        resolution: float,
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
            self._log_debug(
                f"[Gmsh] Erzeuge 3D-Domain {width:.2f}×{length:.2f}×{height:.2f} m, mesh_size={mesh_size:.3f}."
            )

            half_w = width / 2.0
            half_l = length / 2.0
            box = factory.addBox(-half_w, -half_l, 0.0, width, length, height)
            
            # Erstelle Gehäuse-Geometrien und subtrahiere sie vom Domain
            cabinet_objects = []
            if self._cabinet_obstacles:
                self._log_debug(f"[Gmsh] Erzeuge {len(self._cabinet_obstacles)} Gehäuse-Hindernisse.")
                for cabinet in self._cabinet_obstacles:
                    # Erstelle Box für Gehäuse (als undurchlässiges Hindernis)
                    # Gehäuse liegt hinter der Membran in Abstrahlrichtung
                    center = cabinet.center
                    cab_width = cabinet.width
                    cab_depth = cabinet.depth
                    cab_height = max(0.1, cab_depth)  # Mindesthöhe für Gehäuse
                    
                    # Berechne Position in lokalen Koordinaten (Membran normal zur Abstrahlrichtung)
                    c = math.cos(cabinet.azimuth_rad)
                    s = math.sin(cabinet.azimuth_rad)
                    # Gehäuse liegt in negativer y-Richtung (hinter der Membran)
                    cab_x = center[0]
                    cab_y = center[1]
                    cab_z = center[2]
                    
                    # Erstelle Box: Position ist linker unterer Eckpunkt
                    cab_box = factory.addBox(
                        cab_x - cab_width / 2.0,
                        cab_y - cab_depth / 2.0,
                        cab_z - cab_height / 2.0,
                        cab_width,
                        cab_depth,
                        cab_height
                    )
                    cabinet_objects.append((3, cab_box))
                    self._log_debug(
                        f"[Gmsh] Gehäuse {cabinet.identifier}: Box @ ({cab_x:.2f}, {cab_y:.2f}, {cab_z:.2f}), "
                        f"Größe {cab_width:.2f}×{cab_depth:.2f}×{cab_height:.2f} m"
                    )
                
                # Subtrahiere Gehäuse vom Domain
                if cabinet_objects:
                    factory.cut([(3, box)], cabinet_objects, removeObject=False, removeTool=True)
                    self._log_debug("[Gmsh] Gehäuse vom Domain subtrahiert.")
            
            factory.synchronize()

            # Erstelle Panel-Flächen als separate Geometrie
            panel_surfaces = []
            panel_tag_start = 100  # Start-Facet-Tag für Panels (außerhalb von boundary_tags)
            self._panel_tags.clear()
            
            if self._panels:
                self._log_debug(f"[Gmsh] Erzeuge {len(self._panels)} Panel-Flächen.")
                for idx, panel in enumerate(self._panels):
                    if panel.is_muted:
                        continue
                    
                    center = panel.center
                    panel_width = panel.width
                    panel_height = panel.line_height if panel.line_height > 0.0 else panel.height
                    
                    # Erstelle Rechteck als Panel-Fläche
                    # Panel ist normal zur Abstrahlrichtung (azimuth)
                    # Erstelle Rechteck in XY-Ebene (bei z=0), dann rotiere um z-Achse und verschiebe
                    # WICHTIG: Rechteck wird leicht versetzt erstellt, um Überlappung mit Randflächen zu vermeiden
                    rect = factory.addRectangle(
                        -panel_width / 2.0,  # x-Mitte
                        -panel_height / 2.0,  # y-Mitte
                        0.0,  # z=0 (wird später verschoben)
                        panel_width,
                        panel_height
                    )
                    
                    # Rotiere um z-Achse (azimuth), dann verschiebe zu center
                    # Rotation: um z-Achse um azimuth-Winkel
                    # WICHTIG: Center ist bereits korrekt gesetzt (nicht direkt auf Randflächen)
                    factory.rotate([(2, rect)], 0.0, 0.0, 0.0, 0, 0, 1, panel.azimuth_rad)
                    factory.translate([(2, rect)], center[0], center[1], center[2])
                    
                    panel_surfaces.append((2, rect))
                    panel_tag = panel_tag_start + idx
                    self._panel_tags[panel.identifier] = panel_tag
                    
                    self._log_debug(
                        f"[Gmsh] Panel {panel.identifier}: Rechteck-Fläche @ ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                        f"Größe {panel_width:.2f}×{panel_height:.2f} m, Tag={panel_tag}"
                    )
                
                # Schneide Panel-Flächen aus dem Domain (erstellt Facets)
                if panel_surfaces:
                    factory.fragment([(3, box)], panel_surfaces)
                    self._log_debug("[Gmsh] Panel-Flächen in Domain integriert.")
            
            factory.synchronize()

            # Verwende einheitliche Mesh-Größe
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(3)

            domain_phys = gmsh.model.addPhysicalGroup(3, [box], 1)
            gmsh.model.setPhysicalName(3, domain_phys, "domain")

            # Kategorisiere Randflächen anhand ihrer Bounding Box
            surfaces = gmsh.model.getBoundary([(3, box)], oriented=False)
            surface_groups: dict[str, list[int]] = {
                "floor": [],
                "ceiling": [],
                "wall_x_min": [],
                "wall_x_max": [],
                "wall_y_min": [],
                "wall_y_max": [],
            }

            def _bbox(dim: int, tag: int):
                return gmsh.model.getBoundingBox(dim, tag)

            tol = max(height, width, length) * 1e-6
            for dim, tag in surfaces:
                if dim != 2:
                    continue
                xmin, ymin, zmin, xmax, ymax, zmax = _bbox(dim, tag)
                if abs(zmin - 0.0) < tol and abs(zmax - 0.0) < tol:
                    surface_groups["floor"].append(tag)
                elif abs(zmin - height) < tol and abs(zmax - height) < tol:
                    surface_groups["ceiling"].append(tag)
                elif abs(xmin + half_w) < tol and abs(xmax + half_w) < tol:
                    surface_groups["wall_x_min"].append(tag)
                elif abs(xmin - half_w) < tol and abs(xmax - half_w) < tol:
                    surface_groups["wall_x_max"].append(tag)
                elif abs(ymin + half_l) < tol and abs(ymax + half_l) < tol:
                    surface_groups["wall_y_min"].append(tag)
                elif abs(ymin - half_l) < tol and abs(ymax - half_l) < tol:
                    surface_groups["wall_y_max"].append(tag)

            # Markiere Randflächen als Physical Groups
            for name, tags in surface_groups.items():
                if not tags:
                    continue
                phys_tag = self._boundary_tags.get(name, None)
                if phys_tag is None:
                    continue
                group_id = gmsh.model.addPhysicalGroup(2, tags, phys_tag)
                gmsh.model.setPhysicalName(2, group_id, name)

            # Markiere Panel-Flächen als Physical Groups
            for panel in self._panels:
                if panel.is_muted or panel.identifier not in self._panel_tags:
                    continue
                # Finde Facet-Tags für Panel-Fläche
                # Nach fragment() sind Panel-Flächen als separate Flächen vorhanden
                # Wir müssen sie über Geometrie-Informationen identifizieren
                panel_tag = self._panel_tags[panel.identifier]
                
                # Finde Flächen nahe Panel-Center
                center = panel.center
                panel_width = panel.width
                panel_height = panel.line_height if panel.line_height > 0.0 else panel.height
                
                # Suche Flächen in der Nähe des Panel-Centers
                nearby_surfaces = []
                for dim, tag in surfaces:
                    if dim != 2:
                        continue
                    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(dim, tag)
                    # Prüfe ob Fläche nahe Panel-Center liegt
                    center_x = (xmin + xmax) / 2.0
                    center_y = (ymin + ymax) / 2.0
                    center_z = (zmin + zmax) / 2.0
                    
                    dist = math.sqrt(
                        (center_x - center[0])**2 + 
                        (center_y - center[1])**2 + 
                        (center_z - center[2])**2
                    )
                    # Wenn Fläche nahe Panel-Center und Größe ähnlich
                    if dist < max(panel_width, panel_height) * 1.5:
                        nearby_surfaces.append(tag)
                
                if nearby_surfaces:
                    group_id = gmsh.model.addPhysicalGroup(2, nearby_surfaces, panel_tag)
                    gmsh.model.setPhysicalName(2, group_id, f"panel_{panel.identifier}")
                    self._log_debug(
                        f"[Gmsh] Panel {panel.identifier}: {len(nearby_surfaces)} Facets als Tag {panel_tag} markiert."
                    )

            # Markiere Gehäuse-Flächen als Physical Groups (für Dirichlet-RB: p = 0)
            cabinet_tag_start = 200  # Start-Facet-Tag für Cabinets
            self._cabinet_tags.clear()
            for idx, cabinet in enumerate(self._cabinet_obstacles):
                cabinet_tag = cabinet_tag_start + idx
                self._cabinet_tags[cabinet.identifier] = cabinet_tag
                # Finde Facets nahe Cabinet-Center (nach cut() sind sie Teil der Domain-Begrenzung)
                # Für jetzt: Cabinet-Flächen werden später über DOF-Koordinaten identifiziert
                self._log_debug(f"[Gmsh] Cabinet {cabinet.identifier}: Tag {cabinet_tag} zugewiesen.")

            domain, cell_tags, facet_tags = self._gmsh_model_to_mesh(gdim=3)
            return domain, cell_tags, facet_tags
        finally:
            try:
                gmsh.clear()
            except Exception:
                pass
            if initialized_here:
                gmsh.finalize()

    def _estimate_dofs_for_resolution(
        self, width: float, length: float, height: float, resolution: float, degree: int
    ) -> int:
        """Schätzt die DOF-Anzahl für ein 3D-Gitter grob ab."""
        nx = max(2, int(round(width / resolution)))
        ny = max(2, int(round(length / resolution)))
        nz = max(2, int(round(height / resolution)))
        cells = nx * ny * nz
        dofs_per_cell = int((degree + 1) * (degree + 2) * (degree + 3) / 6)  # Tetraeder
        return cells * dofs_per_cell

    def _calculate_fem_mesh_resolution(self, frequencies: list[float]) -> float:
        """Leitet eine FEM-Auflösung aus max. Frequenz & DOF-Limit ab."""
        base_resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)
        width = float(self.settings.width)
        length = float(self.settings.length)
        domain_height = float(getattr(self.settings, "fem_domain_height", 2.0) or 2.0)
        degree = int(getattr(self.settings, "fem_polynomial_degree", 2))
        # Standard points_per_wavelength (kann erhöht werden für bessere Auflösung)
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
            height=domain_height,
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

        estimated_dofs = self._estimate_dofs_for_resolution(width, length, domain_height, fem_resolution, degree)
        if estimated_dofs > max_dofs:
            scale = math.sqrt(estimated_dofs / max_dofs)
            fem_resolution = min(max_resolution, fem_resolution * scale)

        return fem_resolution

    def _determine_domain_height(self, frequencies: list[float]) -> float:
        """Bestimmt die vertikale Ausdehnung des 3D-Domains."""
        base_height = float(getattr(self.settings, "fem_domain_height", 2.0) or 2.0)
        min_height = float(getattr(self.settings, "fem_min_domain_height", 1.0) or 1.0)
        clearance_top = float(getattr(self.settings, "fem_domain_clearance_top", 0.5) or 0.5)

        if self._panels:
            highest_panel = max((panel.center[2] + (panel.height / 2.0) for panel in self._panels), default=0.0)
            base_height = max(base_height, highest_panel + clearance_top)

        if frequencies:
            f_min = min(frequencies)
            speed_of_sound = getattr(self.settings, "speed_of_sound", None)
            if speed_of_sound is None:
                temperature = getattr(self.settings, "temperature", 20.0)
                speed_of_sound = self.functions.calculate_speed_of_sound(temperature)
            if f_min > 0.0:
                wavelength = speed_of_sound / f_min
                # Für korrekte 3D-Ausbreitung: Höhe sollte mindestens λ/2 sein
                # Aber nicht zu groß, um Performance zu gewährleisten
                auto_height = max(min_height, wavelength / 2.0)
                # Maximal 5 m Höhe für Performance
                auto_height = min(auto_height, 5.0)
                base_height = max(base_height, auto_height)
                if getattr(self.settings, "fem_debug_logging", True):
                    self._log_debug(
                        f"[Domain] Frequenz={f_min:.1f} Hz, Wellenlänge={wavelength:.2f} m, "
                        f"auto_height={auto_height:.2f} m (λ/2, max 5m), final_height={base_height:.2f} m"
                    )

        return max(base_height, min_height)

    def _get_boundary_absorption(self, boundary_name: str) -> float:
        attr = f"fem_boundary_absorption_{boundary_name}"
        default_lookup = {
            "floor": 1.0,  # Absorbierend für reinen Direktschall
            "ceiling": 1.0,
            "wall_x_min": 1.0,
            "wall_x_max": 1.0,
            "wall_y_min": 1.0,
            "wall_y_max": 1.0,
            "default": 1.0,
        }
        value = getattr(self.settings, attr, None)
        if value is None:
            return float(default_lookup.get(boundary_name, 1.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default_lookup.get(boundary_name, 1.0))

    def _build_boundary_absorption_form(self, k, trial, test, ds):
        """Erzeugt die Randabsorptions-Terme abhängig von den Tags.
        
        Robin-Randbedingung für absorbierende Ränder: ∂p/∂n + ikαp = 0
        In schwacher Form: ∫ ikα p q ds
        """
        boundary_term = 0
        if self._facet_tags is None:
            coeff = self._get_boundary_absorption("default")
            if coeff > 0.0:
                boundary_term = 1j * k * coeff * ufl.inner(trial, test) * ds
            return boundary_term

        # Debug: Sammle alle verwendeten Absorptionskoeffizienten
        used_coeffs = {}
        for name, tag in self._boundary_tags.items():
            coeff = self._get_boundary_absorption(name)
            if coeff <= 0.0:
                continue
            boundary_term += 1j * k * coeff * ufl.inner(trial, test) * ds(tag)
            used_coeffs[name] = coeff
        
        # Debug: Logge verwendete Absorptionskoeffizienten
        if getattr(self.settings, "fem_debug_logging", True) and used_coeffs:
            self._log_debug(
                f"[Boundary] Robin-Randbedingungen aktiv: {len(used_coeffs)} Flächen, "
                f"Koeffizienten={used_coeffs}, k={k:.3f} m⁻¹"
            )
        
        return boundary_term

    def _get_output_plane_height(self) -> float:
        """Höhe der Auswerte-Ebene (für 2D-Visualisierung).
        
        WICHTIG: Bei grober Mesh-Auflösung gibt es möglicherweise keine DOFs nahe der Panel-Höhe.
        In diesem Fall verwenden wir z=0 (Boden), wo die DOFs tatsächlich vorhanden sind.
        Da der Boden jetzt absorbierend ist, ist das kein Problem mehr.
        """
        explicit = getattr(self.settings, "fem_output_plane_height", None)
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        
        domain_height = self._domain_height or float(getattr(self.settings, "fem_domain_height", 2.0) or 2.0)
        
        # Standard: z=0 (Boden), da dort die DOFs vorhanden sind
        # Der Boden ist jetzt absorbierend, daher kein Problem mit Baffle-Wand
        default_plane = 0.0
        
        # listener_height überschreibt Default, falls gesetzt
        listener_height = getattr(self.settings, "listener_height", None)
        if listener_height is not None:
            try:
                plane = float(listener_height)
            except (TypeError, ValueError):
                plane = default_plane
        else:
            plane = default_plane
        
        return float(np.clip(plane, 0.0, domain_height - 1e-3))

    def _derive_mesh_limits(
        self,
        width: float,
        length: float,
        height: float,
        base_resolution: float,
        base_points_per_wavelength: float,
        base_max_dofs: int,
        base_min_resolution: float,
        frequencies: list[float],
        ) -> tuple[float, int, float]:
        """Leitet adaptive Mesh-Grenzwerte aus Domain-Größe und Frequenzen ab."""

        volume = max(width * length * height, 1e-6)
        highest_frequency = float(max(frequencies)) if frequencies else None

        points_per_wavelength = float(base_points_per_wavelength)
        max_dofs = int(base_max_dofs)
        min_resolution = float(base_min_resolution)

        # Größere Volumina → gröbere Diskretisierung erzwingen
        volume_reference = 200.0  # ca. 10m x 10m x 2m
        area_scale = (volume / volume_reference) ** (1.0 / 3.0)
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
        self._store_panel_metadata()
        self._log_debug(f"[Domain] Anzahl Panels für FEM-Domain: {len(panels)}.")
        self._log_debug(f"[Domain] Anzahl Gehäuse-Hindernisse: {len(cabinets)}.")
        if not panels:
            self._log_debug("[Domain] Keine Speaker-Panels vorhanden – es werden nur äußere Ränder meshing.")

        mesh_obj = None
        cell_tags = None
        facet_tags = None
        domain_height = self._determine_domain_height(frequencies)
        self._domain_height = domain_height
        try:
            mesh_obj, cell_tags, facet_tags = self._generate_gmsh_mesh(width, length, domain_height, resolution)
        except ImportError:
            if panels:
                raise
            # Fallback: einfaches Rechteck-Mesh ohne Lautsprecheröffnungen
            nx = max(2, int(round(width / resolution)))
            ny = max(2, int(round(length / resolution)))
            nz = max(2, int(round(domain_height / resolution)))
            p_min = np.array([-(width / 2.0), -(length / 2.0), 0.0], dtype=np.float64)
            p_max = np.array([width / 2.0, length / 2.0, domain_height], dtype=np.float64)
            mesh_obj = mesh.create_box(
                MPI.COMM_WORLD,
                [p_min, p_max],
                [nx, ny, nz],
                cell_type=mesh.CellType.tetrahedron,
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
            self._dof_coords_xyz = None
            self._dof_tree = None
            self._dof_cache_version = None
            return

        coords = self._function_space.tabulate_dof_coordinates().copy()
        mesh_dim = self._mesh.geometry.dim if self._mesh is not None else 2
        coords = coords.reshape((-1, mesh_dim))

        self._dof_coordinates = coords
        if coords.ndim == 2 and coords.shape[1] >= 2:
            self._dof_coords_xy = coords[:, :2].copy()
        else:
            self._dof_coords_xy = None

        if coords.ndim == 2 and coords.shape[1] >= 3:
            self._dof_coords_xyz = coords[:, :3].copy()
        else:
            self._dof_coords_xyz = None

        kd_tree_coords = None
        if self._dof_coords_xyz is not None:
            kd_tree_coords = self._dof_coords_xyz
        elif self._dof_coords_xy is not None:
            kd_tree_coords = self._dof_coords_xy

        if kd_tree_coords is not None and cKDTree is not None:
            try:
                self._dof_tree = cKDTree(kd_tree_coords)
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

    def _get_dof_coords_xyz(self) -> Optional[np.ndarray]:
        self._ensure_dof_cache()
        return self._dof_coords_xyz

    def _find_nearby_dofs(self, point_xy: np.ndarray, radius: float) -> np.ndarray:
        self._ensure_dof_cache()
        coords = self._dof_coords_xyz
        query_point = np.asarray(point_xy, dtype=float)
        if coords is None:
            coords = self._dof_coords_xy
            query_point = query_point[:2]
        else:
            query_point = query_point[: coords.shape[1]]
        if coords is None:
            return np.array([], dtype=int)
        if self._dof_tree is not None:
            try:
                indices = self._dof_tree.query_ball_point(query_point, r=radius)
                return np.asarray(indices, dtype=int)
            except Exception:
                pass
        distances = np.linalg.norm(coords - query_point, axis=1)
        return np.where(distances <= radius)[0]

    def _find_nearest_dof(self, point_xy: np.ndarray) -> Optional[int]:
        self._ensure_dof_cache()
        coords = self._dof_coords_xyz
        query_point = np.asarray(point_xy, dtype=float)
        if coords is None:
            coords = self._dof_coords_xy
            query_point = query_point[:2]
        else:
            query_point = query_point[: coords.shape[1]]
        if coords is None:
            return None
        if self._dof_tree is not None:
            try:
                distance, index = self._dof_tree.query(query_point, k=1)
                return int(index)
            except Exception:
                pass
        distances = np.linalg.norm(coords - query_point, axis=1)
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

    def _build_panel_neumann_loads(self, frequency: float) -> list[tuple[str, complex]]:
        """Erstellt Neumann-Randbedingungen (Geschwindigkeit) für Panel-Flächen.
        
        Die Amplitude wird aus Balloon-Daten bei 1m berechnet, damit der SPL in 1m
        den Balloon-Daten entspricht.
        
        Für eine abstrahlende Fläche: ∂p/∂n = iωρ * v
        wobei v die normale Geschwindigkeit auf der Membran ist.
        
        Balloon-Daten geben SPL in 1m an: p_1m = 20e-6 * 10^(SPL_db/20)
        Für eine kompakte Quelle: p(r) ≈ (Q/(4πr)) * exp(ikr) für r >> λ
        wobei Q = v * A der Volumenfluss ist (Geschwindigkeit * Fläche)
        
        p_1m = (v*A/(4π*1m)) * exp(ik*1m)
        v = (4π * p_1m * exp(-ik*1m)) / A
        
        Neumann-RB: ∂p/∂n = iωρ * v = iωρ * (4π * p_1m * exp(-ik*1m)) / A
        
        HINWEIS: Da Panel-Flächen nicht als Facet-Tags im Mesh existieren,
        verwenden wir die Panel-Identifikatoren und identifizieren die Flächen
        über DOF-Koordinaten in der RHS-Integration.
        """
        loads: list[tuple[str, complex]] = []
        if not self._panels:
            return loads

        rho = getattr(self.settings, "air_density", 1.2)
        omega = 2.0 * np.pi * frequency
        speed_of_sound = self.settings.speed_of_sound
        k = 2.0 * np.pi * frequency / float(speed_of_sound)
        reference_distance = 1.0  # Balloon-Daten sind bei 1m gemessen
        
        for panel in self._panels:
            if panel.is_muted:
                continue
                
            if not panel.speaker_name:
                continue
                
            dataset = self._get_balloon_dataset_for_frequency(panel.speaker_name, frequency)
            if dataset is None:
                if getattr(self.settings, "fem_debug_logging", True):
                    self._log_debug(
                        f"[Neumann] Keine Balloon-Daten für Panel {panel.identifier} "
                        f"(Speaker={panel.speaker_name}) bei {frequency:.2f} Hz"
                    )
                continue
                
            mag_db, phase_deg = self._sample_balloon_on_axis(dataset)
            if mag_db is None:
                continue
                
            # Balloon-Daten geben SPL in 1m an
            p_1m_amp = 20e-6 * 10 ** (mag_db / 20.0)
            phase_rad = math.radians(phase_deg) if phase_deg is not None else 0.0
            p_1m_complex = p_1m_amp * np.exp(1j * phase_rad)

            effective_height = panel.line_height if panel.line_height > 0.0 else panel.height
            area = panel.area if abs(panel.area) > 1e-6 else panel.width * effective_height
            if area <= 0.0:
                continue

           
            
            # Für jetzt verwenden wir die Formel, aber die Integration muss korrekt sein.
            v_normal = (4.0 * np.pi * p_1m_complex * np.exp(-1j * k * reference_distance)) / area
            
            # DEBUG: Prüfe ob Geschwindigkeit realistisch ist
            v_magnitude = abs(v_normal)
            if v_magnitude > speed_of_sound:
                if getattr(self.settings, "fem_debug_logging", True):
                    self._log_debug(
                        f"[Neumann] WARNUNG: Geschwindigkeit {v_magnitude:.1f} m/s > Schallgeschwindigkeit {speed_of_sound:.1f} m/s! "
                        f"Formel gibt unrealistische Werte für Flächenquelle."
                    )
            
            # Neumann-RB: ∂p/∂n = iωρ * v
            neumann_value = 1j * omega * rho * v_normal
            
            # Verwende Panel-Identifier statt Facet-Tag
            loads.append((panel.identifier, neumann_value))
            
            if getattr(self.settings, "fem_debug_logging", True):
                self._log_debug(
                    f"[Neumann] Panel {panel.identifier} (Speaker={panel.speaker_name}): "
                    f"SPL_1m={mag_db:.1f} dB, p_1m={abs(p_1m_complex):.3e} Pa, "
                    f"area={area:.3f} m², v={abs(v_normal):.3e} m/s, "
                    f"∂p/∂n={abs(neumann_value):.3e} Pa/m"
                )
                
        return loads

    def _build_panel_dirichlet_bcs(self, frequency: float) -> list:
        """Erstellt Dirichlet-Randbedingungen für Panel-Flächen.
        
        WICHTIGE ANFORDERUNGEN für Dirichlet-Randbedingungen in der Helmholtz-Gleichung:
        1. Konsistenz: Randwerte müssen physikalisch sinnvoll sein
        2. Eindeutigkeit: k darf keine Resonanzfrequenz des Gebiets sein
        3. Kontinuität: Randbedingung sollte kontinuierlich sein (oder stückweise)
        4. Physikalische Bedeutung: 
           - Dirichlet legt DRUCK p fest (nicht Geschwindigkeit)
           - Für abstrahlende Flächen wäre normalerweise eine Neumann-RB (Geschwindigkeit) 
             oder eine gemischte RB physikalisch korrekter
           - Konstanter Druck auf gesamter Fläche entspricht nicht einer realen Membran
        
        HINWEIS: Aktuelle Implementierung setzt konstanten Druck auf Panel-Fläche.
        Dies könnte die Ursache für falsche 1/r²-Abnahme sein, da eine reale Membran
        eine Geschwindigkeitsverteilung hat, nicht einen konstanten Druck.
        """
        bcs = []
        if self._function_space is None or not self._panels:
            return bcs

        coords_xyz = self._get_dof_coords_xyz()
        if coords_xyz is None:
            return bcs

        tol_xy = max(self._mesh_resolution or 0.1, 0.05)
        tol_z = tol_xy
        base_drive_db = float(getattr(self.settings, "fem_panel_drive_db", 94.0) or 94.0)
        phase_offset_deg = float(getattr(self.settings, "fem_panel_phase_deg", 0.0) or 0.0)
        phase_offset_rad = math.radians(phase_offset_deg)

        for panel in self._panels:
            if panel.is_muted:
                continue

            total_db = base_drive_db + panel.level_adjust_db
            pressure_linear = 20e-6 * 10 ** (total_db / 20.0)
            p_complex = pressure_linear * np.exp(1j * phase_offset_rad)

            rel = coords_xyz - panel.center.reshape(1, 3)
            c = math.cos(-panel.azimuth_rad)
            s = math.sin(-panel.azimuth_rad)
            rot = np.array([[c, -s], [s, c]], dtype=float)
            local_xy = rel[:, :2] @ rot.T
            local_z = rel[:, 2]
            half_w = panel.width / 2.0 + tol_xy
            half_depth = panel.depth / 2.0 if panel.depth > 0.0 else tol_xy
            half_depth += tol_xy
            half_height = panel.height / 2.0 + tol_z

            mask = (
                (np.abs(local_xy[:, 0]) <= half_w)
                & (np.abs(local_xy[:, 1]) <= half_depth)
                & (np.abs(local_z) <= half_height)
            )
            dof_indices = np.nonzero(mask)[0].astype(np.int32)
            if dof_indices.size == 0:
                self._log_debug(
                    f"[Dirichlet] Keine DOFs für Panel {panel.identifier} gefunden – Panel übersprungen."
                )
                continue

            bc_value = fem.Constant(self._mesh, default_scalar_type(p_complex))
            bc = fem.dirichletbc(bc_value, dof_indices, self._function_space)
            bcs.append(bc)
            self._record_panel_drive(frequency, panel, p_complex, math.degrees(phase_offset_rad))
            
            # Debug: Panel-Position und DOF-Verteilung
            panel_dofs = coords_xyz[dof_indices]
            if len(panel_dofs) > 0:
                x_range = (float(np.min(panel_dofs[:, 0])), float(np.max(panel_dofs[:, 0])))
                y_range = (float(np.min(panel_dofs[:, 1])), float(np.max(panel_dofs[:, 1])))
                z_range = (float(np.min(panel_dofs[:, 2])), float(np.max(panel_dofs[:, 2])))
                self._log_debug(
                    f"[Dirichlet] Panel {panel.identifier} (Speaker={panel.speaker_name}) "
                    f"→ Center=({panel.center[0]:.3f}, {panel.center[1]:.3f}, {panel.center[2]:.3f}), "
                    f"DOFs={dof_indices.size}, |p|={abs(p_complex):.2f} Pa, phase={math.degrees(phase_offset_rad):.1f}° | "
                    f"DOF-Bereich: X[{x_range[0]:.3f}, {x_range[1]:.3f}], Y[{y_range[0]:.3f}, {y_range[1]:.3f}], Z[{z_range[0]:.3f}, {z_range[1]:.3f}]"
                )
            else:
                self._log_debug(
                    f"[Dirichlet] Panel {panel.identifier} (Speaker={panel.speaker_name}) "
                    f"→ DOFs={dof_indices.size}, |p|={abs(p_complex):.2f} Pa, phase={math.degrees(phase_offset_rad):.1f}°"
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
            
            dx = ufl.Measure("dx", domain=self._mesh)
            if self._facet_tags is not None:
                ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._facet_tags)
            else:
                ds = ufl.Measure("ds", domain=self._mesh)
            
            is_complex = np.issubdtype(default_scalar_type, np.complexfloating)
            
            if is_complex:
                boundary_term = self._build_boundary_absorption_form(
                    k, pressure_trial, pressure_test, ds
                )

                # Helmholtz-Gleichung in schwacher Form: ∫ (∇p · ∇q - k²pq) dx = 0
                # Für eine Quelle als Dirichlet-Randbedingung wird p auf der Panel-Fläche festgelegt
                a_form = fem.form(
                    ufl.inner(ufl.grad(pressure_trial), ufl.grad(pressure_test)) * dx
                    - (k ** 2) * ufl.inner(pressure_trial, pressure_test) * dx
                    + boundary_term
                )
                
                # Debug: Prüfe Helmholtz-Parameter
                if getattr(self.settings, "fem_debug_logging", True):
                    wavelength = 2.0 * np.pi / k if k > 0 else float('inf')
                    self._log_debug(
                        f"[Helmholtz] f={frequency:.2f} Hz, k={k:.3f} m⁻¹, "
                        f"λ={wavelength:.2f} m, c={speed_of_sound:.1f} m/s"
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

        # Verwende Neumann-Randbedingungen (Geschwindigkeit) auf Panel-Flächen
        # Dies ist physikalisch korrekter für abstrahlende Flächen
        # Neumann-RB wird korrekt als Flächenintegral in der schwachen Form behandelt
        
        # Erstelle Dirichlet-RB für Gehäuse (undurchlässiges Hindernis: p = 0)
        bcs = []
        if self._facet_tags is not None and self._cabinet_tags:
            coords_xyz = self._get_dof_coords_xyz()
            if coords_xyz is not None:
                for cabinet in self._cabinet_obstacles:
                    if cabinet.is_muted:
                        continue
                    cabinet_tag = self._cabinet_tags.get(cabinet.identifier)
                    if cabinet_tag is None:
                        continue
                    
                    # Finde DOFs auf Cabinet-Fläche über Facet-Tags
                    # Cabinet-Flächen sind nach cut() Teil der Domain-Begrenzung
                    # Für jetzt: Identifiziere über Koordinaten (später: über Facet-Tags)
                    tol = max(self._mesh_resolution or 0.1, 0.05)
                    rel = coords_xyz - cabinet.center.reshape(1, 3)
                    c = math.cos(-cabinet.azimuth_rad)
                    s = math.sin(-cabinet.azimuth_rad)
                    rot = np.array([[c, -s], [s, c]], dtype=float)
                    local_xy = rel[:, :2] @ rot.T
                    local_z = rel[:, 2]
                    
                    half_w = cabinet.width / 2.0 + tol
                    half_d = cabinet.depth / 2.0 + tol
                    half_h = max(0.1, cabinet.depth / 2.0) + tol
                    
                    mask = (
                        (np.abs(local_xy[:, 0]) <= half_w)
                        & (np.abs(local_xy[:, 1]) <= half_d)
                        & (np.abs(local_z) <= half_h)
                    )
                    cabinet_dof_indices = np.nonzero(mask)[0].astype(np.int32)
                    
                    if cabinet_dof_indices.size > 0:
                        # Dirichlet-RB: p = 0 auf Gehäuse-Fläche (schallhartes Hindernis)
                        bc_value = fem.Constant(self._mesh, default_scalar_type(0.0))
                        bc = fem.dirichletbc(bc_value, cabinet_dof_indices, self._function_space)
                        bcs.append(bc)
                        self._log_debug(
                            f"[Cabinet] {cabinet.identifier}: {cabinet_dof_indices.size} DOFs als Dirichlet-RB (p=0) markiert."
                        )
        
        # Erstelle Neumann-RB-Terme für Panel-Flächen in der schwachen Form
        neumann_forms = []
        if self._facet_tags is not None:
            neumann_loads = self._build_panel_neumann_loads(frequency)
            for panel_id, neumann_value in neumann_loads:
                panel_tag = self._panel_tags.get(panel_id)
                if panel_tag is None:
                    continue
                
                # Korrekte schwache Form: ∫_S (∂p/∂n) * q ds = ∫_S (iωρ*v) * q ds
                # neumann_value ist bereits iωρ*v (in Pa/m)
                neumann_constant = fem.Constant(self._mesh, default_scalar_type(neumann_value))
                neumann_form_term = ufl.inner(neumann_constant, pressure_test) * ds(panel_tag)
                neumann_forms.append(neumann_form_term)
                
                if getattr(self.settings, "fem_debug_logging", True):
                    panel = next((p for p in self._panels if p.identifier == panel_id), None)
                    panel_area = panel.area if panel else 0.0
                    self._log_debug(
                        f"[Neumann] Panel {panel_id} (Tag {panel_tag}): "
                        f"∫_S (∂p/∂n) * q ds, |∂p/∂n|={abs(neumann_value):.3e} Pa/m, "
                        f"area={panel_area:.3f} m²"
                    )
            
            # Füge Neumann-Terme zur RHS-Form hinzu
            if neumann_forms:
                L_form = fem.form(L_form + sum(neumann_forms))
                self._log_debug(f"[Neumann] {len(neumann_forms)} Panel-Neumann-Terme zur RHS hinzugefügt.")

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
                plane_height = self._get_output_plane_height()
                front_point = self._sample_grid_pressure(solution, np.array([0.0, 20.0, plane_height]))
                back_point = self._sample_grid_pressure(solution, np.array([0.0, -20.0, plane_height]))
                mag_front = self.functions.mag2db((np.abs(front_point) / p_ref) + 1e-12)
                mag_back = self.functions.mag2db((np.abs(back_point) / p_ref) + 1e-12)
                self._log_debug(
                    "[FEMCardioid] "
                    f"Grid 0°={mag_front:.2f} dB, 180°={mag_back:.2f} dB"
                )
            except Exception:
                pass

        return pressure, spl, phase

    def _sample_grid_pressure(self, solution: fem.Function, point_xyz: np.ndarray) -> complex:
        coords = self._get_dof_coords_xyz()
        query_point = point_xyz
        if coords is None:
            coords = self._get_dof_coords_xy()
            query_point = point_xyz[:2]
        if coords is None:
            return 0.0
        idx = self._find_nearest_dof(query_point)
        if idx is None:
            return 0.0
        return solution.x.array[idx]

    def _resolve_frequency_key(self, fem_results: dict, frequency: float) -> Optional[float]:
        if fem_results is None:
            return None
        freq_key = float(frequency)
        if freq_key in fem_results:
            return freq_key
        for key in fem_results.keys():
            if math.isclose(key, freq_key, rel_tol=1e-6, abs_tol=1e-3):
                return key
        return None

    def _map_dofs_to_grid(
        self,
        points: np.ndarray,
        values: np.ndarray,
        decimals: int = 6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Punktkoordinaten besitzen nicht genügend Dimensionen.")

        width = float(self.settings.width)
        length = float(self.settings.length)
        resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)
        plane_height = self._get_output_plane_height()
        plane_tol = float(getattr(self.settings, "fem_output_plane_tolerance", 0.1) or 0.1)
        
        # Toleranz basierend auf Mesh-Auflösung anpassen
        # Für 3D-Simulation: Verwende eine sehr kleine Toleranz, um nur DOFs nahe der Auswerte-Ebene zu verwenden
        # Die Toleranz sollte deutlich kleiner sein als die Mesh-Auflösung, um präzise Z-Filterung zu gewährleisten
        mesh_resolution = self._mesh_resolution or 0.5
        # Maximal 0.2 m Toleranz für präzise Z-Filterung
        # Dies stellt sicher, dass nur DOFs sehr nahe der Auswerte-Ebene verwendet werden
        max_tol = min(resolution * 0.5, 0.2)  # Maximal 0.2 m oder 0.5x Grid-Resolution
        plane_tol = min(plane_tol, max_tol)

        sound_field_x = np.arange((width / 2 * -1), ((width / 2) + resolution), resolution)
        sound_field_y = np.arange((length / 2 * -1), ((length / 2) + resolution), resolution)
        
        # Debug: Prüfe Grid-Dimensionierung
        if getattr(self.settings, "fem_debug_logging", True):
            grid_x_min = float(sound_field_x[0])
            grid_x_max = float(sound_field_x[-1])
            grid_y_min = float(sound_field_y[0])
            grid_y_max = float(sound_field_y[-1])
            grid_x_center = (grid_x_min + grid_x_max) / 2.0
            grid_y_center = (grid_y_min + grid_y_max) / 2.0
            self._log_debug(
                f"[GridMapping] Grid-Dimensionierung: "
                f"X=[{grid_x_min:.2f}, {grid_x_max:.2f}] m (center={grid_x_center:.2f}), "
                f"Y=[{grid_y_min:.2f}, {grid_y_max:.2f}] m (center={grid_y_center:.2f}), "
                f"Resolution={resolution:.2f} m, "
                f"Size=({len(sound_field_x)}, {len(sound_field_y)})"
            )
            
            # Prüfe DOF-Koordinaten-Bereich
            if points.shape[1] >= 2:
                dof_x_min = float(np.min(points[:, 0]))
                dof_x_max = float(np.max(points[:, 0]))
                dof_y_min = float(np.min(points[:, 1]))
                dof_y_max = float(np.max(points[:, 1]))
                dof_x_center = (dof_x_min + dof_x_max) / 2.0
                dof_y_center = (dof_y_min + dof_y_max) / 2.0
                self._log_debug(
                    f"[GridMapping] DOF-Koordinaten-Bereich: "
                    f"X=[{dof_x_min:.2f}, {dof_x_max:.2f}] m (center={dof_x_center:.2f}), "
                    f"Y=[{dof_y_min:.2f}, {dof_y_max:.2f}] m (center={dof_y_center:.2f})"
                )
                
                # Prüfe Offset zwischen Grid und DOFs
                x_offset = grid_x_center - dof_x_center
                y_offset = grid_y_center - dof_y_center
                if abs(x_offset) > 0.01 or abs(y_offset) > 0.01:
                    self._log_debug(
                        f"[GridMapping] WARNUNG: Offset zwischen Grid und DOFs: "
                        f"X={x_offset:.3f} m, Y={y_offset:.3f} m"
                    )

        pressure_sum = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=values.dtype)
        count_grid = np.zeros((len(sound_field_y), len(sound_field_x)), dtype=int)

        coords_xy = points[:, :2]
        x_coords = np.round(coords_xy[:, 0], decimals=decimals)
        y_coords = np.round(coords_xy[:, 1], decimals=decimals)
        
        # Filtere nach Z-Ebene für 3D-Mesh
        if points.shape[1] >= 3:
            z_coords = points[:, 2]
            z_min = float(np.min(z_coords))
            z_max = float(np.max(z_coords))
            # Debug: Analysiere Z-Verteilung der DOFs
            z_unique = np.unique(np.round(z_coords, decimals=2))
            z_unique_sorted = np.sort(z_unique)
            z_near_plane = z_unique_sorted[np.abs(z_unique_sorted - plane_height) <= plane_tol]
            self._log_debug(
                f"[GridMapping] Z-Bereich: [{z_min:.3f}, {z_max:.3f}] m, "
                f"Auswerte-Ebene: {plane_height:.3f} m, Toleranz: {plane_tol:.3f} m | "
                f"Eindeutige Z-Werte nahe Ebene: {len(z_near_plane)} ({z_near_plane[:10] if len(z_near_plane) > 0 else 'keine'})"
            )
            
            mask = np.abs(z_coords - plane_height) <= plane_tol
            num_masked = int(np.sum(mask))
            
            if num_masked == 0:
                # Fallback: Nimm die nächsten DOFs zur Auswerte-Ebene
                z_distances = np.abs(z_coords - plane_height)
                # Nimm alle DOFs innerhalb von 2*plane_tol oder mindestens die 10% nächsten
                max_dist = max(2.0 * plane_tol, np.percentile(z_distances, 10.0))
                mask = z_distances <= max_dist
                num_masked = int(np.sum(mask))
                self._log_debug(
                    f"[GridMapping] Keine DOFs in Toleranz – verwende Fallback: "
                    f"{num_masked} DOFs innerhalb {max_dist:.3f} m"
                )
            
            # ZUSÄTZLICH: Filtere DOFs, die nahe dem Panel sind (wenn Panel vorhanden)
            # Dies verhindert, dass DOFs weit vom Panel verwendet werden
            if self._panels and num_masked > 0:
                panel_center = self._panels[0].center
                # Berechne Distanzen der gefilterten DOFs zum Panel
                filtered_coords_xy = coords_xy[mask]
                if len(filtered_coords_xy) > 0:
                    dof_distances_to_panel = np.sqrt(
                        (filtered_coords_xy[:, 0] - panel_center[0])**2 + 
                        (filtered_coords_xy[:, 1] - panel_center[1])**2
                    )
                    # Verwende nur DOFs innerhalb von max_distance vom Panel
                    # Für präzise Interpolation: Maximal 50 m vom Panel
                    # Dies stellt sicher, dass nur relevante DOFs verwendet werden
                    max_distance_from_panel = min(
                        max(width, length) * 0.3,  # 30% der Domain-Größe
                        50.0  # Maximal 50 m vom Panel
                    )
                    panel_mask = dof_distances_to_panel <= max_distance_from_panel
                    # Kombiniere mit Z-Mask
                    mask_indices = np.where(mask)[0]
                    mask[mask_indices[~panel_mask]] = False
                    num_masked = int(np.sum(mask))
                    self._log_debug(
                        f"[GridMapping] Nach Panel-Distanz-Filterung: {num_masked} DOFs "
                        f"(max_distance={max_distance_from_panel:.1f} m)"
                    )
            
            if num_masked > 0:
                # Speichere Z-Koordinaten vor dem Filtern für Debug
                z_filtered = z_coords[mask] if points.shape[1] >= 3 else np.array([])
                
                coords_xy = coords_xy[mask]
                x_coords = x_coords[mask]
                y_coords = y_coords[mask]
                values = values[mask]
                
                # Debug: Prüfe Werte
                num_nan = int(np.sum(np.isnan(values)))
                num_finite = int(np.sum(np.isfinite(values)))
                if num_finite > 0:
                    val_min = float(np.nanmin(values))
                    val_max = float(np.nanmax(values))
                    val_mean = float(np.nanmean(values))
                else:
                    val_min = val_max = val_mean = float('nan')
                
                # Debug: Prüfe Z-Koordinaten der gefilterten DOFs
                z_filtered_min = float(np.min(z_filtered)) if len(z_filtered) > 0 else float('nan')
                z_filtered_max = float(np.max(z_filtered)) if len(z_filtered) > 0 else float('nan')
                z_filtered_mean = float(np.mean(z_filtered)) if len(z_filtered) > 0 else float('nan')
                
                # Debug: Prüfe X/Y-Verteilung der gefilterten DOFs (verwende bereits gefilterte Arrays)
                x_filtered_min = float(np.min(coords_xy[:, 0])) if len(coords_xy) > 0 else float('nan')
                x_filtered_max = float(np.max(coords_xy[:, 0])) if len(coords_xy) > 0 else float('nan')
                x_filtered_mean = float(np.mean(coords_xy[:, 0])) if len(coords_xy) > 0 else float('nan')
                y_filtered_min = float(np.min(coords_xy[:, 1])) if len(coords_xy) > 0 else float('nan')
                y_filtered_max = float(np.max(coords_xy[:, 1])) if len(coords_xy) > 0 else float('nan')
                y_filtered_mean = float(np.mean(coords_xy[:, 1])) if len(coords_xy) > 0 else float('nan')
                
                # Debug: Prüfe Distanz zum Panel (wenn vorhanden)
                if self._panels and len(coords_xy) > 0:
                    panel_center = self._panels[0].center
                    dof_distances = np.sqrt((coords_xy[:, 0] - panel_center[0])**2 + (coords_xy[:, 1] - panel_center[1])**2)
                    dist_min = float(np.min(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_max = float(np.max(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_mean = float(np.mean(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_median = float(np.median(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    self._log_debug(
                        f"[GridMapping] DOF-Distanzen zum Panel @ ({panel_center[0]:.3f}, {panel_center[1]:.3f}): "
                        f"min={dist_min:.2f} m, max={dist_max:.2f} m, mean={dist_mean:.2f} m, median={dist_median:.2f} m"
                    )
                
                self._log_debug(
                    f"[GridMapping] {num_masked} DOFs für Grid-Mapping verwendet "
                    f"(von {len(points)} total) | "
                    f"Werte: finite={num_finite}, NaN={num_nan}, "
                    f"min={val_min:.3e}, max={val_max:.3e}, mean={val_mean:.3e} | "
                    f"Z-Bereich gefiltert: [{z_filtered_min:.3f}, {z_filtered_max:.3f}] m, mean={z_filtered_mean:.3f} m | "
                    f"X-Bereich: [{x_filtered_min:.3f}, {x_filtered_max:.3f}] m, mean={x_filtered_mean:.3f} m | "
                    f"Y-Bereich: [{y_filtered_min:.3f}, {y_filtered_max:.3f}] m, mean={y_filtered_mean:.3f} m"
                )
            else:
                self._log_debug(
                    f"[GridMapping] WARNUNG: Keine DOFs gefunden – Grid wird leer bleiben!"
                )
                return sound_field_x, sound_field_y, np.full((len(sound_field_y), len(sound_field_x)), np.nan, dtype=values.dtype)

        x_start = sound_field_x[0]
        y_start = sound_field_y[0]

        # Inverse Distanzgewichtung (IDW): Für jeden Grid-Punkt finden wir die nächstgelegenen DOFs
        use_interpolation = getattr(self.settings, "fem_grid_interpolation", True)
        
        if use_interpolation and len(x_coords) > 0 and cKDTree is not None:
            # Erstelle Grid-Koordinaten
            grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            dof_points_xy = np.column_stack([x_coords, y_coords])
            
            # Erstelle KD-Tree für schnelle Nachbarschaftssuche
            tree = cKDTree(dof_points_xy)
            
            # Für jeden Grid-Punkt: Finde nächstgelegene DOFs (max. 8) innerhalb von 5*resolution
            # Größerer Suchradius, um sicherzustellen, dass alle Grid-Punkte gefüllt werden
            search_radius = resolution * 5.0
            max_neighbors = 8
            power = 2.0  # Exponent für inverse Distanzgewichtung
            
            # Vektorisierte Suche für alle Grid-Punkte
            filled_count = 0
            for grid_idx in range(len(grid_points)):
                grid_point = grid_points[grid_idx:grid_idx+1]  # Shape (1, 2) für query
                # Finde Nachbarn innerhalb des Suchradius
                distances, indices = tree.query(grid_point, k=min(max_neighbors, len(dof_points_xy)), distance_upper_bound=search_radius)
                
                # Flache Arrays zurückgeben
                distances = distances.flatten()
                indices = indices.flatten()
                
                # Filtere ungültige Indizes (inf distances)
                valid = np.isfinite(distances)
                if not np.any(valid):
                    continue
                
                distances = distances[valid]
                indices = indices[valid]
                
                # Vermeide Division durch Null
                distances = np.maximum(distances, 1e-6)
                
                # Inverse Distanzgewichtung: w = 1 / d^power
                weights = 1.0 / (distances ** power)
                weight_sum = np.sum(weights)
                
                if weight_sum > 0:
                    weights /= weight_sum
                    weighted_value = np.sum(values[indices] * weights)
                    pressure_sum.ravel()[grid_idx] = weighted_value
                    count_grid.ravel()[grid_idx] = 1.0
                    filled_count += 1
            
            self._log_debug(
                f"[IDW-Interpolation] {filled_count} von {len(grid_points)} Grid-Punkten gefüllt "
                f"(Suchradius={search_radius:.2f} m, max_neighbors={max_neighbors})"
            )
        elif len(x_coords) > 0:
            # Fallback: Bilineare Interpolation (DOF → 4 Grid-Punkte)
            x_cont = (x_coords - x_start) / resolution
            y_cont = (y_coords - y_start) / resolution
            
            x_floor = np.floor(x_cont).astype(int)
            y_floor = np.floor(y_cont).astype(int)
            x_ceil = x_floor + 1
            y_ceil = y_floor + 1
            
            x_frac = x_cont - x_floor
            y_frac = y_cont - y_floor
            
            w00 = (1 - x_frac) * (1 - y_frac)
            w01 = (1 - x_frac) * y_frac
            w10 = x_frac * (1 - y_frac)
            w11 = x_frac * y_frac
            
            x_floor = np.clip(x_floor, 0, len(sound_field_x) - 1)
            y_floor = np.clip(y_floor, 0, len(sound_field_y) - 1)
            x_ceil = np.clip(x_ceil, 0, len(sound_field_x) - 1)
            y_ceil = np.clip(y_ceil, 0, len(sound_field_y) - 1)
            
            idx00 = y_floor * len(sound_field_x) + x_floor
            idx01 = y_ceil * len(sound_field_x) + x_floor
            idx10 = y_floor * len(sound_field_x) + x_ceil
            idx11 = y_ceil * len(sound_field_x) + x_ceil
            
            np.add.at(pressure_sum.ravel(), idx00, values * w00)
            np.add.at(pressure_sum.ravel(), idx01, values * w01)
            np.add.at(pressure_sum.ravel(), idx10, values * w10)
            np.add.at(pressure_sum.ravel(), idx11, values * w11)
            
            np.add.at(count_grid.ravel(), idx00, w00)
            np.add.at(count_grid.ravel(), idx01, w01)
            np.add.at(count_grid.ravel(), idx10, w10)
            np.add.at(count_grid.ravel(), idx11, w11)
        else:
            # Fallback: einfaches Runden
            x_indices = np.round((x_coords - x_start) / resolution).astype(int)
            y_indices = np.round((y_coords - y_start) / resolution).astype(int)
            x_indices = np.clip(x_indices, 0, len(sound_field_x) - 1)
            y_indices = np.clip(y_indices, 0, len(sound_field_y) - 1)

            flat_indices = y_indices * len(sound_field_x) + x_indices

            np.add.at(pressure_sum.ravel(), flat_indices, values)
            np.add.at(count_grid.ravel(), flat_indices, 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            pressure_grid = pressure_sum / count_grid
        pressure_grid[count_grid == 0] = np.nan

        # Debug: Prüfe Grid-Ergebnis
        num_grid_points = int(np.sum(count_grid > 0))
        num_grid_nan = int(np.sum(np.isnan(pressure_grid)))
        num_grid_finite = int(np.sum(np.isfinite(pressure_grid)))
        if num_grid_finite > 0:
            grid_min = float(np.nanmin(pressure_grid))
            grid_max = float(np.nanmax(pressure_grid))
            grid_mean = float(np.nanmean(pressure_grid))
        else:
            grid_min = grid_max = grid_mean = float('nan')
        
        self._log_debug(
            f"[GridMapping] Grid-Ergebnis: {num_grid_points} Punkte mit Daten, "
            f"finite={num_grid_finite}, NaN={num_grid_nan} | "
            f"min={grid_min:.3e}, max={grid_max:.3e}, mean={grid_mean:.3e}"
        )
        
        # Debug: Analysiere Druckwerte über Distanz (nur wenn Panel vorhanden)
        if self._panels and num_grid_finite > 0:
            panel_center = self._panels[0].center
            grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
            distances = np.sqrt((grid_x - panel_center[0])**2 + (grid_y - panel_center[1])**2)
            
            # Debug: Prüfe Grid-Koordinaten und Panel-Position
            grid_center_x = (sound_field_x[0] + sound_field_x[-1]) / 2.0
            grid_center_y = (sound_field_y[0] + sound_field_y[-1]) / 2.0
            panel_x_offset = panel_center[0] - grid_center_x
            panel_y_offset = panel_center[1] - grid_center_y
            self._log_debug(
                f"[Distanz-Analyse] Panel @ ({panel_center[0]:.3f}, {panel_center[1]:.3f}), "
                f"Grid-Center @ ({grid_center_x:.3f}, {grid_center_y:.3f}), "
                f"Offset: ({panel_x_offset:.3f}, {panel_y_offset:.3f}) m | "
                f"Grid X-Bereich: [{sound_field_x[0]:.2f}, {sound_field_x[-1]:.2f}] m, "
                f"Grid Y-Bereich: [{sound_field_y[0]:.2f}, {sound_field_y[-1]:.2f}] m, "
                f"Grid-Auflösung: {resolution:.2f} m"
            )
            
            # Prüfe, ob Panel im Grid-Bereich liegt
            panel_in_grid_x = sound_field_x[0] <= panel_center[0] <= sound_field_x[-1]
            panel_in_grid_y = sound_field_y[0] <= panel_center[1] <= sound_field_y[-1]
            if not (panel_in_grid_x and panel_in_grid_y):
                self._log_debug(
                    f"[Distanz-Analyse] WARNUNG: Panel liegt außerhalb des Grid-Bereichs! "
                    f"Panel in Grid X: {panel_in_grid_x}, Panel in Grid Y: {panel_in_grid_y}"
                )
            
            # Analysiere Druckwerte in Distanz-Bereichen
            dist_ranges = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 75)]
            prev_spl_mean = None
            prev_dist_center = None
            reference_dist = None
            reference_spl = None
            
            for dist_min, dist_max in dist_ranges:
                mask = (distances >= dist_min) & (distances < dist_max) & np.isfinite(pressure_grid)
                if np.any(mask):
                    p_values = pressure_grid[mask]
                    p_mean = float(np.nanmean(p_values))
                    p_max = float(np.nanmax(p_values))
                    p_min = float(np.nanmin(p_values))
                    # Konvertiere zu SPL für bessere Lesbarkeit
                    p_ref = 20e-6
                    spl_mean = 20.0 * np.log10(p_mean / p_ref) if p_mean > 0 else float('-inf')
                    spl_max = 20.0 * np.log10(p_max / p_ref) if p_max > 0 else float('-inf')
                    spl_min = 20.0 * np.log10(p_min / p_ref) if p_min > 0 else float('-inf')
                    dist_center = (dist_min + dist_max) / 2.0
                    
                    # Setze Referenz für erste Distanz
                    if reference_dist is None and dist_center > 0:
                        reference_dist = dist_center
                        reference_spl = spl_mean
                    
                    # Berechne erwartete SPL für 1/r² Abnahme (6 dB pro Verdopplung)
                    if prev_spl_mean is not None and prev_dist_center is not None:
                        # Erwartete Abnahme zwischen zwei Bereichen: 20*log10(r2/r1)
                        expected_abfall_db = 20.0 * np.log10(dist_center / prev_dist_center) if dist_center > 0 and prev_dist_center > 0 else 0.0
                        actual_abfall_db = prev_spl_mean - spl_mean if prev_spl_mean != float('-inf') and spl_mean != float('-inf') else 0.0
                        
                        # Berechne auch erwartete SPL relativ zur Referenz
                        if reference_dist is not None and reference_spl is not None:
                            expected_spl_from_ref = reference_spl - 20.0 * np.log10(dist_center / reference_dist) if dist_center > 0 and reference_dist > 0 else reference_spl
                            deviation_from_expected = spl_mean - expected_spl_from_ref if spl_mean != float('-inf') and expected_spl_from_ref != float('-inf') else 0.0
                            
                            self._log_debug(
                                f"[Distanz-Analyse] {dist_min}-{dist_max} m (center={dist_center:.1f} m): "
                                f"SPL mean={spl_mean:.1f} dB, min={spl_min:.1f} dB, max={spl_max:.1f} dB | "
                                f"p mean={p_mean:.3e} Pa | "
                                f"Abnahme vs. vorher: actual={actual_abfall_db:.1f} dB, expected={expected_abfall_db:.1f} dB (1/r²) | "
                                f"vs. Referenz ({reference_dist:.1f} m): expected={expected_spl_from_ref:.1f} dB, deviation={deviation_from_expected:.1f} dB"
                            )
                        else:
                            self._log_debug(
                                f"[Distanz-Analyse] {dist_min}-{dist_max} m (center={dist_center:.1f} m): "
                                f"SPL mean={spl_mean:.1f} dB, min={spl_min:.1f} dB, max={spl_max:.1f} dB | "
                                f"p mean={p_mean:.3e} Pa | "
                                f"Abnahme: actual={actual_abfall_db:.1f} dB, expected={expected_abfall_db:.1f} dB (1/r²)"
                            )
                    else:
                        self._log_debug(
                            f"[Distanz-Analyse] {dist_min}-{dist_max} m (center={dist_center:.1f} m): "
                            f"SPL mean={spl_mean:.1f} dB, min={spl_min:.1f} dB, max={spl_max:.1f} dB | "
                            f"p mean={p_mean:.3e} Pa"
                        )
                    prev_spl_mean = spl_mean
                    prev_dist_center = dist_center

        return sound_field_x, sound_field_y, pressure_grid

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
            
            p_min = float(np.nanmin(sound_field_pressure)) if isinstance(sound_field_pressure, np.ndarray) else float("nan")
            p_max = float(np.nanmax(sound_field_pressure)) if isinstance(sound_field_pressure, np.ndarray) else float("nan")
            spl_min = float(np.nanmin(sound_field_spl)) if isinstance(sound_field_spl, np.ndarray) else float("nan")
            spl_max = float(np.nanmax(sound_field_spl)) if isinstance(sound_field_spl, np.ndarray) else float("nan")
            
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
        
        # Debug: Prüfe Eingabedaten
        if isinstance(pressure, np.ndarray):
            num_pressure_nan = int(np.sum(np.isnan(pressure)))
            num_pressure_finite = int(np.sum(np.isfinite(pressure)))
            if num_pressure_finite > 0:
                p_min = float(np.nanmin(pressure))
                p_max = float(np.nanmax(pressure))
            else:
                p_min = p_max = float('nan')
            self._log_debug(
                f"[GridMapping] Eingabe: {len(pressure)} Druckwerte, "
                f"finite={num_pressure_finite}, NaN={num_pressure_nan}, "
                f"min={p_min:.3e}, max={p_max:.3e}"
            )

        sound_field_x, sound_field_y, pressure_grid = self._map_dofs_to_grid(points, pressure, decimals=decimals)

        # Debug: Druckwerte vor Luftabsorptions-Korrektur
        if getattr(self.settings, "fem_debug_logging", True) and pressure_grid is not None:
            p_before_air = pressure_grid.copy()
            p_min_before_air = float(np.nanmin(p_before_air)) if np.any(np.isfinite(p_before_air)) else float('nan')
            p_max_before_air = float(np.nanmax(p_before_air)) if np.any(np.isfinite(p_before_air)) else float('nan')
            p_mean_before_air = float(np.nanmean(p_before_air)) if np.any(np.isfinite(p_before_air)) else float('nan')
            self._log_debug(
                f"[GridMapping] Druckwerte VOR Luftabsorptions-Korrektur: min={p_min_before_air:.3e}, max={p_max_before_air:.3e}, mean={p_mean_before_air:.3e}"
            )

        pressure_grid = self._apply_air_absorption_to_grid(
            pressure_grid,
            sound_field_x,
            sound_field_y,
            freq_data.get("source_positions"),
            frequency,
        )

        # Debug: Druckwerte nach Luftabsorptions-Korrektur
        if getattr(self.settings, "fem_debug_logging", True) and pressure_grid is not None:
            p_after_air = pressure_grid.copy()
            p_min_after_air = float(np.nanmin(p_after_air)) if np.any(np.isfinite(p_after_air)) else float('nan')
            p_max_after_air = float(np.nanmax(p_after_air)) if np.any(np.isfinite(p_after_air)) else float('nan')
            p_mean_after_air = float(np.nanmean(p_after_air)) if np.any(np.isfinite(p_after_air)) else float('nan')
            if np.any(np.isfinite(p_before_air)) and np.any(np.isfinite(p_after_air)):
                ratio_air = p_mean_after_air / p_mean_before_air if p_mean_before_air > 0 else float('nan')
                self._log_debug(
                    f"[GridMapping] Druckwerte NACH Luftabsorptions-Korrektur: min={p_min_after_air:.3e}, max={p_max_after_air:.3e}, mean={p_mean_after_air:.3e}, ratio={ratio_air:.3f}"
                )

        if np.iscomplexobj(pressure_grid):
            pressure_grid = np.abs(pressure_grid)

        return sound_field_x, sound_field_y, pressure_grid

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
        plane_height = self._get_output_plane_height()
        grid_points = np.column_stack(
            (
                grid_x.ravel(),
                grid_y.ravel(),
                np.full(grid_x.size, plane_height),
            )
        )

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

    def _get_panel_centers_array(self) -> Optional[np.ndarray]:
        if not self._panels:
            return None
        try:
            return np.array([panel.center for panel in self._panels], dtype=float)
        except Exception:
            return None

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


