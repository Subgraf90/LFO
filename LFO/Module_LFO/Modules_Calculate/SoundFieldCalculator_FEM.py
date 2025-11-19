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
from dataclasses import dataclass, field
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


DEFAULT_RESOLUTION = 0.5           # Grundlegende Rasterweite des FEM-Netzes (Meter)
DEFAULT_FEM_DOMAIN_HEIGHT = 2.0    # Standardhöhe des 3D-Rechenraums (Meter)
DEFAULT_FEM_MIN_DOMAIN_HEIGHT = 1.0  # Untere Schranke für die Domainhöhe (Meter) / minimale domainhöhe
DEFAULT_FEM_DOMAIN_CLEARANCE_TOP = 0.5  # Sicherheitsabstand nach oben zur höchsten Quelle (Meter)
DEFAULT_FEM_POINTS_PER_WAVELENGTH = 10.0  # Zielauflösung: Knoten pro Wellenlänge
DEFAULT_FEM_MAX_DOFS = 250_000      # Obergrenze für Freiheitsgrade, um Rechenzeit zu begrenzen
DEFAULT_FEM_MIN_RESOLUTION = 0.05   # Minimal erlaubte Netzauflösung (Meter)
DEFAULT_FEM_MAX_RESOLUTION_FACTOR = 4.0  # Max. Faktor, um die Basisauflösung zu vergrößern
DEFAULT_FEM_MESH_MIN_SIZE = 0.05    # Harte Untergrenze für einzelne Elementgrößen (Meter)
DEFAULT_FEM_POLYNOMIAL_DEGREE = 2   # Standard-FEM-Basisgrad (z. B. P2)
DEFAULT_POINT_SOURCE_SPL_DB = 130.0 # Bezugspegel für Punktschallquellen (dB @ 1 m, 20 µPa)
DEFAULT_TEMPERATURE_CELSIUS = 20.0  # Referenztemperatur zur Schallgeschwindigkeitsberechnung (°C)
DEFAULT_FEM_OUTPUT_PLANE_TOLERANCE = 0.1  # Z-Toleranz für _get_output_plane_height() beim Auslesen der Ebene
DEFAULT_FEM_GRID_K_NEIGHBORS = 16   # Anzahl Nachbarn für IDW-Interpolation (Grid-Abbildung)
DEFAULT_FEM_GRID_IDW_POWER = 2.0    # Potenz im Inverse-Distance-Weighting (IDW)
DEFAULT_HUMIDITY_PERCENT = 50.0     # Standard-Luftfeuchte [%] (kann in den Settings überschrieben werden)
DEFAULT_AIR_PRESSURE_PA = 101_325.0 # Standard-Luftdruck (Pa) für Akustikmodelle
FEM_SETTING_DEFAULTS: dict[str, object] = {
    "fem_use_direct_solver": True,
    "fem_compute_particle_velocity": True,
    "fem_calculate_frequency": None,
    "resolution": DEFAULT_RESOLUTION,
    "fem_domain_height": DEFAULT_FEM_DOMAIN_HEIGHT,
    "fem_min_domain_height": DEFAULT_FEM_MIN_DOMAIN_HEIGHT,
    "fem_domain_clearance_top": DEFAULT_FEM_DOMAIN_CLEARANCE_TOP,
    "fem_points_per_wavelength": DEFAULT_FEM_POINTS_PER_WAVELENGTH,
    "fem_max_dofs": DEFAULT_FEM_MAX_DOFS,
    "fem_min_resolution": DEFAULT_FEM_MIN_RESOLUTION,
    "fem_polynomial_degree": DEFAULT_FEM_POLYNOMIAL_DEGREE,
    "fem_max_resolution_limit": None,
    "fem_debug_logging": True,
    "fem_point_source_spl_db": DEFAULT_POINT_SOURCE_SPL_DB,
    "fem_output_plane_tolerance": DEFAULT_FEM_OUTPUT_PLANE_TOLERANCE,
    "fem_output_plane_height": None,
    "fem_enable_panel_neumann": True,
    "fem_use_boundary_sources": False,
    "fem_grid_interpolation": True,
    "fem_grid_k_neighbors": DEFAULT_FEM_GRID_K_NEIGHBORS,
    "fem_grid_idw_power": DEFAULT_FEM_GRID_IDW_POWER,
    "a_source_db": 94,
    "temperature": DEFAULT_TEMPERATURE_CELSIUS,
    "speed_of_sound": None,
    "listener_height": None,
    "use_air_absorption": False,
    "humidity": DEFAULT_HUMIDITY_PERCENT,
    "air_pressure": DEFAULT_AIR_PRESSURE_PA,
}


def _try_import_fenics_modules():
    """Versucht die optionalen FEniCSx-Module nachzuladen."""
    global MPI, dolfinx_pkg, fem, mesh, ufl, PETSc, gmshio, dolfinx_gmsh, default_scalar_type, _fenics_import_error

    if all(
        module is not None
        for module in (MPI, dolfinx_pkg, fem, mesh, ufl, PETSc, gmshio)
    ):
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
        import logging

        logging.getLogger(__name__).exception(
            "Fehler beim Laden der FEniCS-Module", exc_info=exc
        )


@dataclass
class PointSource:
    """Einfache Punktschallquelle für FEM-Berechnung."""

    identifier: str
    array_key: str
    position: np.ndarray  # [x, y, z]
    level_adjust_db: float
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
        self._python_include_path_prepared = False
        self._frequency_progress_session = None
        self._precomputed_frequencies: Optional[list[float]] = None
        self._frequency_progress_last_third_start: Optional[int] = None
        self._resolved_fem_frequency: Optional[float] = None
        self._point_sources: list[PointSource] = []
        self._boundary_tags = {
            "floor": 11,
            "ceiling": 12,
            "wall_x_min": 13,
            "wall_x_max": 14,
            "wall_y_min": 15,
            "wall_y_max": 16,
        }
        self._panel_tags: Dict[str, int] = {}  # Panel-Identifier → Facet-Tag (nicht mehr verwendet)

    def _log_debug(self, message: str):
        """Hilfsfunktion für konsistente Debug-Ausgaben."""
        if self._get_setting("fem_debug_logging"):
            print(f"[FEM Debug] {message}")

    def _get_setting(self, name: str):
        """Zentraler Zugriff auf Settings mit Fallback auf Modul-Defaults."""
        default = FEM_SETTING_DEFAULTS.get(name)
        value = getattr(self.settings, name, default)
        return default if value is None else value

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

        if isinstance(self.calculation_spl, dict):
            self.calculation_spl.pop("fdtd_simulation", None)
            self.calculation_spl.pop("fdtd_time_snapshots", None)

        self._ensure_fenics_available()
        self._prepare_python_include_path()

        if self._precomputed_frequencies is not None:
            frequencies = list(self._precomputed_frequencies)
            self._precomputed_frequencies = None
        else:
            frequencies = self._determine_frequencies()
        if not frequencies:
            raise ValueError("Es sind keine Frequenzen zur Berechnung definiert.")

        # ==================================================================
        # ABSCHNITT 1: GRID-ERSTELLUNG
        # ==================================================================
        self._build_domain(frequencies)
        
        # Info über FEM-Konfiguration
        use_direct_solver = bool(self._get_setting("fem_use_direct_solver"))
        width = float(self.settings.width)
        length = float(self.settings.length)
        resolution_setting = self._get_setting("resolution")
        grid_resolution = float(resolution_setting or DEFAULT_RESOLUTION)
        fem_resolution = getattr(self, "_mesh_resolution", grid_resolution)

        fem_results = {}
        total_freqs = len(frequencies)
        compute_velocity = bool(self._get_setting("fem_compute_particle_velocity"))
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
            
            # ==================================================================
            # ABSCHNITT 3: BERECHNUNG
            # ==================================================================
            solution = self._solve_frequency(frequency)
            pressure, spl, phase = self._extract_pressure_and_phase(solution)

            velocity = None
            if compute_velocity:
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
                "source_positions": (
                    self._last_point_source_positions
                    if self._last_point_source_positions is not None
                    else panel_centers
                ),
            }

            if compute_velocity and velocity is not None:
                fem_results[float(frequency)]["particle_velocity"] = velocity

            if self._frequency_progress_session is not None:
                self._frequency_progress_session.advance()
            self._raise_if_frequency_cancelled()

        # ==================================================================
        # ABSCHNITT 4: OUTPUT/SPEICHERUNG
        # ==================================================================
        self.calculation_spl["fem_simulation"] = fem_results
        self._assign_primary_soundfield_results(
            frequencies,
            fem_results,
        )

        self._frequency_progress_session = None
        self._frequency_progress_last_third_start = None

        if not fem_results:
            self._log_debug("[Ergebnisse] Keine FEM-Frequenzen vorhanden.")
        
    def set_data_container(self, data_container):
        """Setzt den gemeinsam genutzten Daten-Container."""
        self._data_container = data_container
    
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
        required_modules = {
            "fem": fem,
            "mesh": mesh,
            "MPI": MPI,
            "ufl": ufl,
            "PETSc": PETSc,
        }
        missing = [name for name, module in required_modules.items() if module is None]
        if missing:
            hint = ""
            if _fenics_import_error is not None:
                import logging
                logging.getLogger(__name__).exception(
                    "FEniCS-Import schlug fehl", exc_info=_fenics_import_error
                )
                hint = f" (Grund: {_fenics_import_error})"
            missing_str = ", ".join(missing)
            raise ImportError(
                "FEniCSx (dolfinx, ufl, mpi4py) ist nicht installiert oder konnte nicht geladen werden. "
                f"Fehlende Module: {missing_str}.{hint}"
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

        fem_frequency = self._get_setting("fem_calculate_frequency")
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
        """Liefert den Daten-Container (für Balloon-Daten)."""
        if self._data_container is not None:
            return self._data_container
        return getattr(self, "data", None)


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

    def _collect_point_sources(self) -> list[PointSource]:
        """Erzeugt einfache Punktschallquellen aus Speaker-Arrays."""
        sources: list[PointSource] = []
        speaker_arrays = getattr(self.settings, "speaker_arrays", None)
        if not isinstance(speaker_arrays, dict):
            return sources

        self._log_debug(f"Starte Punktschallquellen-Erstellung für {len(speaker_arrays)} Lautsprecher-Arrays.")

        for array_key, speaker_array in speaker_arrays.items():
            if getattr(speaker_array, "hide", False):
                self._log_debug(f"[Sources] Array '{array_key}' ist ausgeblendet – übersprungen.")
                continue

            # Extrahiere Quellennamen
            raw_names_primary = getattr(speaker_array, "source_polar_pattern", None)
            names_list = self._normalize_source_name_sequence(raw_names_primary)
            if not names_list:
                raw_names_secondary = getattr(speaker_array, "source_type", None)
                names_list = self._normalize_source_name_sequence(raw_names_secondary)
            if not names_list:
                self._log_debug(f"[Sources] Array '{array_key}' hat keine gültigen source‑Namen – übersprungen.")
                continue

            num_sources = len(names_list)
            if num_sources == 0:
                continue

            # Extrahiere Positionen und Level
            def get_attr(primary, secondary=None, default=None):
                value = getattr(speaker_array, primary, None)
                if value is None and secondary:
                    value = getattr(speaker_array, secondary, None)
                return value if value is not None else default

            xs = np.asarray(
                self._normalize_sequence(get_attr("source_position_calc_x", "source_position_x"), num_sources),
                dtype=float,
            )
            ys = np.asarray(
                self._normalize_sequence(get_attr("source_position_calc_y", "source_position_y"), num_sources),
                dtype=float,
            )
            zs = np.asarray(
                self._normalize_sequence(get_attr("source_position_calc_z", "source_position_z"), num_sources),
                dtype=float,
            )
            gains = np.asarray(
                self._normalize_sequence(get_attr("gain", default=[0.0] * num_sources), num_sources),
                dtype=float,
            )
            source_levels = np.asarray(
                self._normalize_sequence(get_attr("source_level", default=[0.0] * num_sources), num_sources),
                dtype=float,
            )

            a_source_db = float(self._get_setting("a_source_db") or 0.0)
            level_adjust_db = source_levels + gains + a_source_db
            array_muted = getattr(speaker_array, "mute", False)

            # Erstelle Punktschallquellen
            for idx, raw_name in enumerate(names_list):
                speaker_name = self._decode_speaker_name(raw_name)
                position = np.array([xs[idx], ys[idx], zs[idx]], dtype=float)
                identifier = f"{array_key}_{idx}"

                source = PointSource(
                        identifier=identifier,
                        array_key=str(array_key),
                    position=position,
                        level_adjust_db=float(level_adjust_db[idx]),
                        speaker_name=speaker_name,
                        is_muted=array_muted,
                )
                sources.append(source)

                self._log_debug(
                    f"[Sources] {identifier}: Speaker={speaker_name}, "
                    f"Position=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}), "
                    f"Level={level_adjust_db[idx]:.1f}dB"
                )

        self._log_debug(f"Punktschallquellen-Erstellung abgeschlossen: {len(sources)} Quellen erzeugt.")
        return sources

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
            mesh_size = max(resolution, DEFAULT_FEM_MESH_MIN_SIZE)
            self._log_debug(
                f"[Gmsh] Erzeuge 3D-Domain {width:.2f}×{length:.2f}×{height:.2f} m, mesh_size={mesh_size:.3f}."
            )

            half_w = width / 2.0
            half_l = length / 2.0
            box = factory.addBox(-half_w, -half_l, 0.0, width, length, height)
            
            factory.synchronize()

            # Punktschallquellen werden später über DOF-Koordinaten identifiziert
            if self._point_sources:
                self._log_debug(
                    f"[Gmsh] {len(self._point_sources)} Punktschallquellen werden später über DOF-Koordinaten identifiziert."
                )
            
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

            # Panel-Flächen werden NICHT als Physical Groups markiert,
            # da sie nicht als separate Geometrie in Gmsh erstellt werden.
            # Sie werden später über DOF-Koordinaten identifiziert.


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
        base_resolution = float(self._get_setting("resolution") or DEFAULT_RESOLUTION)
        width = float(self.settings.width)
        length = float(self.settings.length)
        domain_height = float(self._get_setting("fem_domain_height") or DEFAULT_FEM_DOMAIN_HEIGHT)
        degree = int(self._get_setting("fem_polynomial_degree") or DEFAULT_FEM_POLYNOMIAL_DEGREE)
        # Standard points_per_wavelength (kann erhöht werden für bessere Auflösung)
        base_points_per_wavelength = float(self._get_setting("fem_points_per_wavelength") or DEFAULT_FEM_POINTS_PER_WAVELENGTH)
        base_max_dofs = int(self._get_setting("fem_max_dofs") or DEFAULT_FEM_MAX_DOFS)
        base_min_resolution = float(self._get_setting("fem_min_resolution") or DEFAULT_FEM_MIN_RESOLUTION)
        max_resolution_setting = self._get_setting("fem_max_resolution_limit")
        max_resolution = float(
            max_resolution_setting or (base_resolution * DEFAULT_FEM_MAX_RESOLUTION_FACTOR)
        )

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
                temperature = self._get_setting("temperature") or 20.0
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
        base_height = float(self._get_setting("fem_domain_height") or DEFAULT_FEM_DOMAIN_HEIGHT)
        min_height = float(self._get_setting("fem_min_domain_height") or DEFAULT_FEM_MIN_DOMAIN_HEIGHT)
        clearance_top = float(
            self._get_setting("fem_domain_clearance_top") or DEFAULT_FEM_DOMAIN_CLEARANCE_TOP
        )

        if self._point_sources:
            highest_source = max((source.position[2] for source in self._point_sources), default=0.0)
            base_height = max(base_height, highest_source + clearance_top)

        if frequencies:
            f_min = min(frequencies)
            speed_of_sound = self._get_setting("speed_of_sound")
            if speed_of_sound is None:
                temperature = self._get_setting("temperature") or 20.0
                speed_of_sound = self.functions.calculate_speed_of_sound(temperature)
            if f_min > 0.0:
                wavelength = speed_of_sound / f_min
                # Für korrekte 3D-Ausbreitung: Höhe sollte mindestens λ/2 sein
                # Aber nicht zu groß, um Performance zu gewährleisten
                auto_height = max(min_height, wavelength / 2.0)
                # Maximal 5 m Höhe für Performance
                auto_height = min(auto_height, 5.0)
                base_height = max(base_height, auto_height)
                if self._get_setting("fem_debug_logging"):
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
        if self._get_setting("fem_debug_logging") and used_coeffs:
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
        explicit = self._get_setting("fem_output_plane_height")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        
        domain_height = self._domain_height or float(
            self._get_setting("fem_domain_height") or DEFAULT_FEM_DOMAIN_HEIGHT
        )
        
        # Standard: z=0 (Boden), da dort die DOFs vorhanden sind
        # Der Boden ist jetzt absorbierend, daher kein Problem mit Baffle-Wand
        default_plane = 1.0
        
        # listener_height überschreibt Default, falls gesetzt
        listener_height = self._get_setting("listener_height")
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

    # ==================================================================
    # ABSCHNITT 1: GRID-ERSTELLUNG
    # ==================================================================

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

        self._point_sources = self._collect_point_sources()
        self._log_debug(f"[Domain] Anzahl Punktschallquellen für FEM-Domain: {len(self._point_sources)}.")
        if not self._point_sources:
            self._log_debug("[Domain] Keine Punktschallquellen vorhanden – es werden nur äußere Ränder meshing.")

        mesh_obj = None
        cell_tags = None
        facet_tags = None
        domain_height = self._determine_domain_height(frequencies)
        self._domain_height = domain_height
        try:
            mesh_obj, cell_tags, facet_tags = self._generate_gmsh_mesh(width, length, domain_height, resolution)
        except ImportError:
            if self._point_sources:
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

        element_degree = int(self._get_setting("fem_polynomial_degree") or DEFAULT_FEM_POLYNOMIAL_DEGREE)
        
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

    # ==================================================================
    # ABSCHNITT 2: QUELLEN-ERSTELLUNG
    # ==================================================================

    def _build_point_source_loads(self, frequency: float) -> list[tuple[str, complex, np.ndarray]]:
        """
        Erzeugt Punktquellen-Lasten für einfache Punktschallquellen.
        
        Für eine Punktquelle: p(r) = (Q/(4πr)) * exp(ikr)
        wobei Q die Quellstärke ist.
        
        Die Quelle wird als Dirac-Delta-Funktion δ(x-x_s) modelliert,
        was in der schwachen Form zu einem Punktlast-Term führt.
        """
        loads: list[tuple[str, complex, np.ndarray]] = []
        if not self._point_sources:
            return loads

        speed_of_sound = self.settings.speed_of_sound
        k = 2.0 * np.pi * frequency / float(speed_of_sound)
        reference_distance = 1.0  # Referenzdistanz für SPL
        
        # Standard-SPL bei 1m (kann über level_adjust_db angepasst werden)
        default_spl_db = float(
            self._get_setting("fem_point_source_spl_db") or DEFAULT_POINT_SOURCE_SPL_DB
        )
        
        for source in self._point_sources:
            if source.is_muted:
                continue
                
            # Berechne SPL aus level_adjust_db
            spl_db = default_spl_db + source.level_adjust_db
            
            # Konvertiere SPL zu Schalldruck bei 1m
            p_1m_amp = 20e-6 * 10 ** (spl_db / 20.0)
            
            # Für Punktquelle: Q = 4π * p_1m * exp(-ik*1m)
            # Die Quellstärke Q wird direkt als komplexe Amplitude verwendet
            source_strength = 4.0 * np.pi * reference_distance * p_1m_amp * np.exp(-1j * k * reference_distance)
            
            loads.append((source.identifier, source_strength, source.position))

            if self._get_setting("fem_debug_logging"):
                    self._log_debug(
                    f"[PointSource] {source.identifier}: Speaker={source.speaker_name}, "
                    f"Position=({source.position[0]:.2f},{source.position[1]:.2f},{source.position[2]:.2f}), "
                    f"SPL_1m={spl_db:.1f}dB, Q={abs(source_strength):.3e}"
                )
                
        return loads
        
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

    # ==================================================================
    # ABSCHNITT 3: BERECHNUNG
    # ==================================================================

    def _solve_frequency(self, frequency: float) -> fem.Function:
        """Löst die Helmholtz-Gleichung für eine einzelne Frequenz."""

        V = self._function_space
        freq_label = f"{frequency:.2f}Hz"

        if V is None:
            raise RuntimeError("FEM-Funktionsraum wurde nicht initialisiert. Bitte _build_domain() aufrufen.")

        from dolfinx.fem.petsc import assemble_matrix, assemble_vector

        pressure_trial = ufl.TrialFunction(V)
        pressure_test = ufl.TestFunction(V)

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
            a_form = fem.form(
                ufl.inner(ufl.grad(pressure_trial), ufl.grad(pressure_test)) * dx
                - (k ** 2) * ufl.inner(pressure_trial, pressure_test) * dx
                + boundary_term
            )
            if self._get_setting("fem_debug_logging"):
                wavelength = 2.0 * np.pi / k if k > 0 else float("inf")
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

        bcs: list[fem.DirichletBC] = []
        use_panel_neumann = bool(self._get_setting("fem_enable_panel_neumann"))
        neumann_loads = self._build_point_source_loads(frequency) if use_panel_neumann else []

        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        self._raise_if_frequency_cancelled()

        b = assemble_vector(L_form)

        if neumann_loads:
            coords_xyz = self._get_dof_coords_xyz()
            if coords_xyz is not None:
                for source_id, source_strength, source_position in neumann_loads:
                    distances = np.linalg.norm(coords_xyz - source_position.reshape(1, 3), axis=1)
                    nearest_dof_idx = int(np.argmin(distances))
                    nearest_distance = distances[nearest_dof_idx]
                    if nearest_distance < (self._mesh_resolution or 0.1) * 2.0:
                        b.array[nearest_dof_idx] += source_strength
                        if self._get_setting("fem_debug_logging"):
                            self._log_debug(
                                f"[PointSource] {source_id}: DOF={nearest_dof_idx}, "
                                f"distance={nearest_distance:.3f}m, Q={abs(source_strength):.3e}"
                            )

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

        self._apply_rhs_payload_to_vector(b, rhs_payload, V)
        try:
            rhs_norm_after = b.norm()
        except Exception:
            rhs_norm_after = None
        if rhs_norm_after is not None:
            self._log_debug(f"[Solve {freq_label}] RHS-Norm nach Quellen: {rhs_norm_after:.3e}")
        self._raise_if_frequency_cancelled()

        use_direct_solver = bool(self._get_setting("fem_use_direct_solver"))
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
        if self._get_setting("fem_use_boundary_sources"):
            self._log_debug(
                "[RHS] Neumann-Randquellen sind deaktiviert – verwende Punktquellen."
            )

        # Monopole-Quellen sind standardmäßig deaktiviert (verwende Neumann-RB für Panels)
        payload: dict[str, object] = {}
        self._last_point_source_positions = None
        return fem.form(rhs_form), payload

    def _apply_rhs_payload_to_vector(self, b_vector, rhs_payload, V):
        """Wendet verteilte Quellen auf den RHS-Vektor an.
        
        Aktuell werden keine verteilten Punktquellen verwendet (Monopole deaktiviert).
        Die Methode bleibt für zukünftige Erweiterungen erhalten.
        """
        if not rhs_payload:
            return

        # Wenn boundary_sources verwendet werden, wurde die Form bereits in _assemble_rhs integriert
        if isinstance(rhs_payload, dict) and rhs_payload.get("boundary_sources"):
            # Die Neumann-BC-Terme sind bereits in der Form enthalten, nichts zu tun
            return

        # Verteilte Punktquellen (z.B. Monopole) - aktuell deaktiviert
        distributed = rhs_payload.get("distributed") if isinstance(rhs_payload, dict) else None
        if isinstance(distributed, dict):
            indices = distributed.get("indices")
            values = distributed.get("values")
            if indices is not None and values is not None and len(indices) > 0:
                try:
                    b_vector.setValues(indices.astype(np.int32), values, addv=True)
                except Exception:
                    for idx, val in zip(indices, values):
                        b_vector.setValue(int(idx), val, addv=True)
            positions = distributed.get("positions")
            if positions is not None:
                try:
                    self._last_point_source_positions = np.array(positions, dtype=float)
                except Exception:
                    pass

        # Debug-Ausgabe
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
        
    # ==================================================================
    # ABSCHNITT 4: OUTPUT/SPEICHERUNG
    # ==================================================================

    def _extract_pressure_and_phase(
        self, solution: fem.Function
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = solution.x.array
        pressure = np.abs(values)
        p_ref = 20e-6
        spl = self.functions.mag2db((pressure / p_ref) + 1e-12)
        phase = np.angle(values, deg=True)

        if self._get_setting("fem_debug_logging"):
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
        resolution = float(self._get_setting("resolution") or DEFAULT_RESOLUTION)
        plane_height = self._get_output_plane_height()
        plane_tol = float(
            self._get_setting("fem_output_plane_tolerance") or DEFAULT_FEM_OUTPUT_PLANE_TOLERANCE
        )
        
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
        if self._get_setting("fem_debug_logging"):
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
                # Fallback: verwende die DOFs, die den Listener-Plane am nächsten liegen
                z_distances = np.abs(z_coords - plane_height)
                closest_count = max(1, int(0.1 * len(z_distances)))
                closest_indices = np.argsort(z_distances)[:closest_count]
                mask = np.zeros_like(z_coords, dtype=bool)
                mask[closest_indices] = True
                num_masked = closest_count
                self._log_debug(
                    "[GridMapping] Keine DOFs innerhalb der Ebene gefunden – "
                    f"verwende {closest_count} nächstgelegene DOFs (max Δz={float(z_distances[closest_indices[-1]]):.3f} m)."
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
                if self._point_sources and len(coords_xy) > 0:
                    source_center = self._point_sources[0].position
                    dof_distances = np.sqrt((coords_xy[:, 0] - source_center[0])**2 + (coords_xy[:, 1] - source_center[1])**2)
                    dist_min = float(np.min(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_max = float(np.max(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_mean = float(np.mean(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    dist_median = float(np.median(dof_distances)) if len(dof_distances) > 0 else float('nan')
                    self._log_debug(
                        f"[GridMapping] DOF-Distanzen zur Quelle @ ({source_center[0]:.3f}, {source_center[1]:.3f}): "
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
        use_interpolation = bool(self._get_setting("fem_grid_interpolation"))
        
        if use_interpolation and len(x_coords) > 0 and cKDTree is not None:
            # Erstelle Grid-Koordinaten
            grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            dof_points_xy = np.column_stack([x_coords, y_coords])
            
            # Erstelle KD-Tree für schnelle Nachbarschaftssuche
            tree = cKDTree(dof_points_xy)
            
            max_neighbors = int(
                self._get_setting("fem_grid_k_neighbors") or DEFAULT_FEM_GRID_K_NEIGHBORS
            )
            max_neighbors = max(4, min(max_neighbors, len(dof_points_xy)))
            power = float(self._get_setting("fem_grid_idw_power") or DEFAULT_FEM_GRID_IDW_POWER)

            distances, indices = tree.query(grid_points, k=max_neighbors)

            if max_neighbors == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            # Definiere valid immer (unabhängig von max_neighbors)
            valid = np.isfinite(distances)
            weights = np.zeros_like(distances, dtype=float)
            weights[valid] = 1.0 / np.maximum(distances[valid], 1e-6) ** power

            weight_sums = np.sum(weights, axis=1)
            nonzero_mask = weight_sums > 0
            interpolated = np.full(grid_points.shape[0], np.nan, dtype=values.dtype)
            if np.any(nonzero_mask):
                numerator = np.sum(values[indices] * weights, axis=1)
                interpolated[nonzero_mask] = numerator[nonzero_mask] / weight_sums[nonzero_mask]

            pressure_sum = interpolated.reshape(grid_x.shape)
            count_grid = np.where(np.isfinite(pressure_sum), 1, 0)

            filled_count = int(np.sum(nonzero_mask))
            self._log_debug(
                f"[IDW-Interpolation] {filled_count} von {len(grid_points)} Grid-Punkten gefüllt "
                f"(k={max_neighbors}, power={power:.2f})"
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
        if self._point_sources and num_grid_finite > 0:
            source_center = self._point_sources[0].position
            grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
            distances = np.sqrt((grid_x - source_center[0])**2 + (grid_y - source_center[1])**2)
            
            # Debug: Prüfe Grid-Koordinaten und Panel-Position
            grid_center_x = (sound_field_x[0] + sound_field_x[-1]) / 2.0
            grid_center_y = (sound_field_y[0] + sound_field_y[-1]) / 2.0
            source_x_offset = source_center[0] - grid_center_x
            source_y_offset = source_center[1] - grid_center_y
            self._log_debug(
                f"[Distanz-Analyse] Quelle @ ({source_center[0]:.3f}, {source_center[1]:.3f}), "
                f"Grid-Center @ ({grid_center_x:.3f}, {grid_center_y:.3f}), "
                f"Offset: ({source_x_offset:.3f}, {source_y_offset:.3f}) m | "
                f"Grid X-Bereich: [{sound_field_x[0]:.2f}, {sound_field_x[-1]:.2f}] m, "
                f"Grid Y-Bereich: [{sound_field_y[0]:.2f}, {sound_field_y[-1]:.2f}] m, "
                f"Grid-Auflösung: {resolution:.2f} m"
            )
            
            # Prüfe, ob Quelle im Grid-Bereich liegt
            source_in_grid_x = sound_field_x[0] <= source_center[0] <= sound_field_x[-1]
            source_in_grid_y = sound_field_y[0] <= source_center[1] <= sound_field_y[-1]
            if not (source_in_grid_x and source_in_grid_y):
                self._log_debug(
                    f"[Distanz-Analyse] WARNUNG: Quelle liegt außerhalb des Grid-Bereichs! "
                    f"Quelle in Grid X: {source_in_grid_x}, Quelle in Grid Y: {source_in_grid_y}"
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
            target_frequency = self._get_setting("fem_calculate_frequency")
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
        if self._get_setting("fem_debug_logging") and pressure_grid is not None:
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
        if self._get_setting("fem_debug_logging") and pressure_grid is not None:
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
        if pressure_grid is None or not self._get_setting("use_air_absorption"):
            return pressure_grid

        temperature = float(self._get_setting("temperature") or 20.0)
        humidity = float(self._get_setting("humidity") or 50.0)
        air_pressure = float(self._get_setting("air_pressure") or 101325.0)

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
        if not self._point_sources:
            return None
        try:
            return np.array([source.position for source in self._point_sources], dtype=float)
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
        
        if self._get_setting("fem_debug_logging") and len(sound_field_x) and len(sound_field_y):
            center_x_idx = np.argmin(np.abs(sound_field_x - 0.0))
            center_y_idx = np.argmin(np.abs(sound_field_y - 0.0))
            spl_at_source = float(spl_grid[center_y_idx, center_x_idx])

            sample_distance = 10.0
            x_sample_idx = np.argmin(np.abs(sound_field_x - sample_distance))
            spl_at_sample = float(spl_grid[center_y_idx, x_sample_idx])

            expected_drop = 20 * np.log10(sample_distance / 1.0)
            measured_drop = spl_at_source - spl_at_sample
            self._log_debug(
                "[SPL Debug] SPL@0m≈{:.2f} dB, SPL@{:.1f}m≈{:.2f} dB, Δ={:.2f} dB (Ideal {:.2f} dB)".format(
                    spl_at_source, sample_distance, spl_at_sample, measured_drop, expected_drop
                )
            )
        
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


