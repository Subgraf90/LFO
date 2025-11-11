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
from typing import Iterable, Optional

import numpy as np

from Module_LFO.Modules_Init.ModuleBase import ModuleBase

try:
    MPI = importlib.import_module("mpi4py.MPI")
    dolfinx_pkg = importlib.import_module("dolfinx")
    fem = importlib.import_module("dolfinx.fem")
    mesh = importlib.import_module("dolfinx.mesh")
    ufl = importlib.import_module("ufl")
    PETSc = importlib.import_module("petsc4py.PETSc")
    default_scalar_type = dolfinx_pkg.default_scalar_type

except ImportError:  # pragma: no cover - reine Verfügbarkeitsprüfung
    MPI = None
    fem = None
    mesh = None
    ufl = None
    PETSc = None
    default_scalar_type = np.complex128


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

        self._ensure_fenics_available()

        frequencies = self._determine_frequencies()
        if not frequencies:
            raise ValueError("Es sind keine Frequenzen zur Berechnung definiert.")

        self._build_domain()

        fem_results = {}
        for frequency in frequencies:
            solution = self._solve_frequency(frequency)
            pressure, spl, phase = self._extract_pressure_and_phase(solution)
            velocity = self._compute_particle_velocity(solution, frequency)

            fem_results[float(frequency)] = {
                "points": self._mesh.geometry.x.copy(),
                "pressure_complex": solution.x.array.copy(),
                "pressure": pressure,
                "spl": spl,
                "phase": phase,
            }

            if velocity is not None:
                fem_results[float(frequency)]["particle_velocity"] = velocity

        self.calculation_spl["fem_simulation"] = fem_results

    def set_data_container(self, data_container):
        """Setzt den gemeinsam genutzten Daten-Container."""

        self._data_container = data_container

    # ------------------------------------------------------------------
    # FEM-Hilfsfunktionen
    # ------------------------------------------------------------------
    def _ensure_fenics_available(self):
        if fem is None or mesh is None or MPI is None or ufl is None or PETSc is None:
            raise ImportError(
                "FEniCSx (dolfinx, ufl, mpi4py) ist nicht installiert. "
                "Bitte installieren und erneut versuchen."
            )

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
        """Ermittelt die zu berechnenden Frequenzen aus den Settings."""

        base = self._normalize_frequencies(
            getattr(self.settings, "calculate_frequency", None)
        )

        lower = getattr(self.settings, "lower_calculate_frequency", None)
        upper = getattr(self.settings, "upper_calculate_frequency", None)

        if lower is None or upper is None:
            return base

        try:
            lower_val = float(lower)
            upper_val = float(upper)
        except (TypeError, ValueError):
            return base

        if upper_val < lower_val:
            lower_val, upper_val = upper_val, lower_val

        band_frequencies: list[float] = []

        if self._data_container is not None:
            for speaker_name in self._iter_active_speaker_names():
                balloon = self._data_container.get_balloon_data(
                    speaker_name, use_averaged=False
                )
                if not balloon or "freqs" not in balloon:
                    continue

                freqs = np.asarray(balloon["freqs"], dtype=float).flatten()
                mask = (freqs >= lower_val) & (freqs <= upper_val)
                if not np.any(mask):
                    continue

                band_frequencies = np.unique(np.round(freqs[mask], decimals=6)).tolist()
                if band_frequencies:
                    break

        if band_frequencies:
            return band_frequencies

        base_in_band = [f for f in base if lower_val <= f <= upper_val]
        if base_in_band:
            return base_in_band

        return base

    def _iter_active_speaker_names(self) -> Iterable[str]:
        if not hasattr(self.settings, "speaker_arrays"):
            return []

        for speaker_array in self.settings.speaker_arrays.values():
            if getattr(speaker_array, "mute", False) or getattr(speaker_array, "hide", False):
                continue
            for speaker_name in getattr(speaker_array, "source_polar_pattern", []) or []:
                if speaker_name:
                    yield speaker_name

    def _build_domain(self):
        """Erzeugt das FEM-Mesh auf Basis der Settings.

        - Das Modell arbeitet in der XY-Ebene (2D) mit Dreieckselementen.
        - Die Auflösung orientiert sich an `settings.resolution`, kann aber
          bei Bedarf über `fem_polynomial_degree` verfeinert werden.
        - Die Funktion speichert Mesh und Funktionsraum für Folgeaufrufe.
        """

        if self._mesh is not None:
            return

        width = float(self.settings.width)
        length = float(self.settings.length)
        resolution = float(getattr(self.settings, "resolution", 0.5) or 0.5)

        nx = max(2, int(math.ceil(width / resolution)))
        ny = max(2, int(math.ceil(length / resolution)))

        p_min = np.array([-(width / 2.0), -(length / 2.0)], dtype=np.float64)
        p_max = np.array([width / 2.0, length / 2.0], dtype=np.float64)

        self._mesh = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [p_min, p_max],
            [nx, ny],
            cell_type=mesh.CellType.triangle,
        )

        element_degree = int(getattr(self.settings, "fem_polynomial_degree", 2))
        self._function_space = fem.FunctionSpace(
            self._mesh,
            ("CG", element_degree),
            dtype=default_scalar_type,
        )

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

        pressure_trial = ufl.TrialFunction(V)
        pressure_test = ufl.TestFunction(V)

        k = 2.0 * np.pi * frequency / float(self.settings.speed_of_sound)

        absorption = getattr(self.settings, "fem_boundary_absorption", 1.0)

        a_form = fem.form(
            ufl.inner(ufl.grad(pressure_trial), ufl.grad(pressure_test)) * ufl.dx
            - (k ** 2) * pressure_trial * pressure_test * ufl.dx
            + 1j * k * absorption * pressure_trial * pressure_test * ufl.ds
        )

        L_form, point_sources = self._assemble_rhs(V, pressure_test, frequency)

        A = fem.petsc.assemble_matrix(a_form)
        A.assemble()
        b = fem.petsc.assemble_vector(L_form)

        for point_source in point_sources:
            point_source.apply(b)

        solver = PETSc.KSP().create(self._mesh.comm)
        solver.setOperators(A)
        solver.setType("gmres")
        solver.setTolerances(rtol=1e-9, atol=1e-12)
        solver.getPC().setType("ilu")

        solution = fem.Function(V, name="pressure")
        solver.solve(b, solution.x.petsc_vec)
        solution.x.scatter_forward()

        return solution

    def _assemble_rhs(self, V, test_function, frequency: float):
        """Baut rechte Seite und Punktquellenliste zusammen.

        Jeder Lautsprecher wird als Punktquelle modelliert. Sollte ein Punkt
        außerhalb des Mesh liegen, wird er (stillschweigend) ignoriert. Für
        realistischere Quellen (z.B. Membranflächen) müsste hier eine Fläche
        oder Linienquelle implementiert werden.
        """

        if not hasattr(self.settings, "speaker_arrays"):
            return fem.form(0 * test_function * ufl.dx), []

        point_sources = []
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue

            level_linear = self._speaker_level_linear(speaker_array)
            positions = self._speaker_positions_2d(speaker_array)

            for idx, pos in enumerate(positions):
                amplitude = level_linear[idx] * self._phase_factor(speaker_array, idx, frequency)
                point = np.array([pos[0], pos[1], 0.0], dtype=np.float64)

                try:
                    point_sources.append(fem.PointSource(V, point, amplitude))
                except RuntimeError:
                    continue

        return fem.form(0 * test_function * ufl.dx), point_sources

    def _speaker_level_linear(self, speaker_array) -> np.ndarray:
        gain = getattr(speaker_array, "gain", 0.0)
        source_level_db = np.array(speaker_array.source_level, dtype=float)
        level_db = source_level_db + gain
        return self.functions.db2mag(level_db)

    def _speaker_positions_2d(self, speaker_array) -> np.ndarray:
        xs = np.array(
            getattr(
                speaker_array,
                'source_position_calc_x',
                speaker_array.source_position_x,
            ),
            dtype=float,
        )
        ys = np.array(
            getattr(
                speaker_array,
                'source_position_calc_y',
                speaker_array.source_position_y,
            ),
            dtype=float,
        )
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
        spl = self.functions.mag2db(pressure + 1e-12)
        phase = np.angle(values, deg=True)
        return pressure, spl, phase

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

        degree = self._function_space.element.basix_element.degree
        V_vec = fem.VectorFunctionSpace(
            self._mesh,
            ("CG", degree),
            dtype=default_scalar_type,
        )

        grad_p = fem.Function(V_vec, name="grad_p")

        v = ufl.TrialFunction(V_vec)
        w = ufl.TestFunction(V_vec)
        a_proj = fem.form(ufl.inner(v, w) * ufl.dx)
        L_proj = fem.form(ufl.inner(ufl.grad(solution), w) * ufl.dx)

        A = fem.petsc.assemble_matrix(a_proj)
        A.assemble()
        b = fem.petsc.assemble_vector(L_proj)

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

        fem_results = self.calculation_spl.get("fem_simulation")
        if not fem_results:
            raise RuntimeError(
                "Es liegen keine FEM-Ergebnisse vor. Bitte zuerst die Berechnung ausführen."
            )

        freq_key = float(frequency)
        freq_data = fem_results.get(freq_key)
        if freq_data is None and frequency in fem_results:
            freq_data = fem_results[frequency]
        if freq_data is None:
            raise KeyError(
                f"Für die Frequenz {frequency} Hz wurden keine FEM-Ergebnisse gefunden."
            )

        points = freq_data["points"]
        pressure = freq_data["pressure"]

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("Punktkoordinaten besitzen nicht genügend Dimensionen.")

        coords_xy = points[:, :2]
        rounded_x = np.round(coords_xy[:, 0], decimals=decimals)
        rounded_y = np.round(coords_xy[:, 1], decimals=decimals)

        x_unique = np.unique(rounded_x)
        y_unique = np.unique(rounded_y)

        pressure_grid = np.full(
            (y_unique.size, x_unique.size),
            np.nan,
            dtype=pressure.dtype,
        )

        x_indices = np.searchsorted(x_unique, rounded_x)
        y_indices = np.searchsorted(y_unique, rounded_y)

        pressure_grid[y_indices, x_indices] = pressure

        return pressure_grid.T


