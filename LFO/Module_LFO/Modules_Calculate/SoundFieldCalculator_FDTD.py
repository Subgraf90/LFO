"""
Zeitbereichs-Simulation mittels vereinfachtem 2D-FDTD für SPL-over-time.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from Module_LFO.Modules_Calculate.Functions import FunctionToolbox
from Module_LFO.Modules_Init.ModuleBase import ModuleBase


class SoundFieldCalculatorFDTD(ModuleBase):
    """
    Simuliert eine Zeitentwicklung des Schalldrucks auf einem 2D-Gitter.

    Die FDTD-Simulation ist unabhängig von FEM-Ergebnissen und greift direkt auf:
        - Speaker-Arrays aus Settings
        - Balloon-Daten aus dem Data-Container
        - Panel-Metadaten (Dimensionen) aus Cabinet-Metadaten
    """

    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl if isinstance(calculation_spl, dict) else {}
        self.functions = FunctionToolbox(settings)
        self._data_container = None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def set_data_container(self, data_container):
        self._data_container = data_container

    def clear_cached_data(self):
        if not isinstance(self.calculation_spl, dict):
            return
        self.calculation_spl.pop("fdtd_simulation", None)
        self.calculation_spl.pop("fdtd_time_snapshots", None)

    def calculate_fdtd_snapshots(self, frequency: float, frames_per_period: int = 16):
        """
        Führt eine echte FDTD-Zeitsimulation durch.
        
        Diese Methode simuliert die Wellenausbreitung im Zeitbereich,
        beginnend vom Lautsprecher und zeigt die Ausbreitung über die Zeit.
        """
        print(f"[DEBUG FDTD] calculate_fdtd_snapshots() aufgerufen: frequency={frequency} Hz, frames_per_period={frames_per_period}")
        if frames_per_period <= 0:
            frames_per_period = 16
        freq_key = float(frequency)
        sim_store = self.calculation_spl.setdefault("fdtd_simulation", {})
        entry = sim_store.get(freq_key)
        if entry and entry.get("frames_per_period") == frames_per_period:
            print(f"[DEBUG FDTD] calculate_fdtd_snapshots() - Gecachte Daten gefunden, überspringe Berechnung")
            return entry
        
        # Führe echte FDTD-Simulation durch
        print(f"[DEBUG FDTD] calculate_fdtd_snapshots() - Starte neue FDTD-Simulation")
        simulation = self._run_fdtd_simulation(freq_key, frames_per_period)
        sim_store[freq_key] = simulation
        print(f"[DEBUG FDTD] calculate_fdtd_snapshots() - FDTD-Simulation abgeschlossen, Daten gespeichert")
        return simulation

    def get_time_snapshot_grid(
        self,
        frequency: float,
        frame_index: int,
        frames_per_period: int = 16,
        decimals: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print(f"[DEBUG FDTD] get_time_snapshot_grid() aufgerufen: frequency={frequency} Hz, frame_index={frame_index}, frames_per_period={frames_per_period}")
        if frames_per_period <= 0:
            raise ValueError("frames_per_period muss > 0 sein.")
        simulation = self.calculate_fdtd_snapshots(frequency, frames_per_period)
        sound_field_x = np.asarray(simulation["sound_field_x"], dtype=float)
        sound_field_y = np.asarray(simulation["sound_field_y"], dtype=float)
        pressure_frames = np.asarray(simulation["pressure_frames"], dtype=np.float32)
        # total_frames = frames_per_period + 1 (inkl. Frame 0 für t=0)
        total_frames = int(simulation.get("total_frames", pressure_frames.shape[0]))
        if total_frames <= 0:
            raise RuntimeError("Keine FDTD-Frames verfügbar.")
        # Frame 0 = t=0 (nur Quellen), Frames 1..total_frames-1 = Ausbreitung
        frame = int(frame_index) % total_frames
        pressure_grid = pressure_frames[frame]
        
        # Debug: Zeige Frame-Informationen
        p_min = float(np.min(pressure_grid))
        p_max = float(np.max(pressure_grid))
        p_mean = float(np.mean(np.abs(pressure_grid)))
        print(f"[FDTD] get_time_snapshot_grid: Frame {frame}/{total_frames-1} (index {frame_index}): "
              f"min={p_min:.3e}, max={p_max:.3e}, mean_abs={p_mean:.3e} Pa")
        
        p_ref = 20e-6
        spl_grid = self.functions.mag2db((np.abs(pressure_grid) / p_ref) + 1e-12)
        return sound_field_x, sound_field_y, pressure_grid, spl_grid

    # ------------------------------------------------------------------
    # FDTD-Simulation
    # ------------------------------------------------------------------
    def _run_fdtd_simulation(self, frequency: float, frames_per_period: int) -> dict:
        print(f"[DEBUG FDTD] _run_fdtd_simulation() START: frequency={frequency} Hz, frames_per_period={frames_per_period}")
        if frequency <= 0.0:
            raise ValueError("Frequenz für FDTD muss > 0 Hz sein.")

        width = float(self.settings.width)
        length = float(self.settings.length)
        base_resolution = float(getattr(self.settings, "fdtd_resolution", None) or getattr(self.settings, "resolution", 0.5) or 0.5)
        
        # Optimiere Auflösung basierend auf Wellenlänge für bessere Wellenausbreitung
        # Für stabile und genaue FDTD-Simulation braucht man mindestens 10-20 Punkte pro Wellenlänge
        temperature = getattr(self.settings, "temperature", 20.0)
        c = self.functions.calculate_speed_of_sound(temperature)
        wavelength = c / frequency if frequency > 0 else float('inf')
        
        # Punkte pro Wellenlänge (Standard: 15 für gute Genauigkeit)
        points_per_wavelength = float(getattr(self.settings, "fdtd_points_per_wavelength", 15.0) or 15.0)
        points_per_wavelength = max(10.0, min(30.0, points_per_wavelength))  # Zwischen 10 und 30
        
        # Ideale Auflösung basierend auf Wellenlänge
        ideal_resolution = wavelength / points_per_wavelength if wavelength < float('inf') else base_resolution
        
        # Verwende die feinere Auflösung (besser für Wellenausbreitung)
        # Aber nicht zu fein (Performance)
        min_resolution = float(getattr(self.settings, "fdtd_min_resolution", 0.02) or 0.02)
        max_resolution = float(getattr(self.settings, "fdtd_max_resolution", base_resolution * 2) or base_resolution * 2)
        resolution = max(min_resolution, min(ideal_resolution, base_resolution, max_resolution))
        
        print(f"[FDTD] Auflösungsoptimierung: Wellenlänge={wavelength:.3f}m, Ideal={ideal_resolution:.3f}m, "
              f"Verwendet={resolution:.3f}m ({wavelength/resolution:.1f} Punkte/Wellenlänge)")

        nx_inner = max(3, int(round(width / resolution)) + 1)
        ny_inner = max(3, int(round(length / resolution)) + 1)
        sound_field_x = np.linspace(-width / 2.0, width / 2.0, nx_inner, dtype=float)
        sound_field_y = np.linspace(-length / 2.0, length / 2.0, ny_inner, dtype=float)
        
        # Berechne tatsächliche Schrittweite im Innenbereich (wichtig für korrekte FDTD-Berechnung!)
        dx = width / (nx_inner - 1) if nx_inner > 1 else resolution
        dy = length / (ny_inner - 1) if ny_inner > 1 else resolution
        # Verwende die kleinere Schrittweite für CFL (konservativ)
        grid_resolution = min(dx, dy)

        # Lege äußere PML-Zone an (außerhalb des Innenbereichs)
        min_freq_for_pml = float(getattr(self.settings, "fdtd_pml_min_frequency", 30.0) or 30.0)
        min_wavelength = c / max(min_freq_for_pml, 1e-3)
        default_pml_width = 10.0  # vom Nutzer gefordert
        min_phys_width = float(getattr(self.settings, "fdtd_absorption_min_width", default_pml_width) or default_pml_width)
        min_phys_width = max(min_phys_width, 1.2 * min_wavelength, default_pml_width)
        max_phys_width = float(getattr(self.settings, "fdtd_absorption_max_width", max(width, length) * 0.45) or max(width, length) * 0.45)
        target_amp = float(getattr(self.settings, "fdtd_absorption_edge_gain", 1e-7) or 1e-7)
        profile_order = int(getattr(self.settings, "fdtd_absorption_profile_order", 7) or 7)
        profile_order = min(9, max(5, profile_order))
        
        damping_zone_width_cells = int(round(min_phys_width / grid_resolution))
        damping_zone_width_cells = max(12, damping_zone_width_cells)
        damping_zone_width_cells = min(
            damping_zone_width_cells,
            int(round(max_phys_width / grid_resolution)),
        )
        padding_cells = max(1, damping_zone_width_cells)
        pml_phys_width = padding_cells * grid_resolution
        
        nx = nx_inner + 2 * padding_cells
        ny = ny_inner + 2 * padding_cells
        x_min_ext = sound_field_x[0] - padding_cells * grid_resolution
        x_max_ext = sound_field_x[-1] + padding_cells * grid_resolution
        y_min_ext = sound_field_y[0] - padding_cells * grid_resolution
        y_max_ext = sound_field_y[-1] + padding_cells * grid_resolution
        simulation_x = np.linspace(x_min_ext, x_max_ext, nx, dtype=float)
        simulation_y = np.linspace(y_min_ext, y_max_ext, ny, dtype=float)
        inner_slice_y = slice(padding_cells, padding_cells + ny_inner)
        inner_slice_x = slice(padding_cells, padding_cells + nx_inner)

        temperature = getattr(self.settings, "temperature", 20.0)
        c = self.functions.calculate_speed_of_sound(temperature)
        # CFL-Bedingung: dt ≤ dx / (c · √2) für 2D
        cfl_limit = grid_resolution / (c * math.sqrt(2.0))
        custom_dt = float(getattr(self.settings, "fdtd_time_step", 0.0) or 0.0)
        if custom_dt > 0:
            if custom_dt > cfl_limit * 0.9:
                print(f"[FDTD] Warnung: Benutzerdefinierter Zeitschritt ({custom_dt:.6e} s) überschreitet CFL-Limit ({cfl_limit * 0.9:.6e} s). Verwende CFL-Limit.")
                dt = cfl_limit * 0.9
            else:
                dt = custom_dt
        else:
            dt = cfl_limit * 0.9

        steps_per_period = max(16, int(round((1.0 / frequency) / dt)))
        omega = 2.0 * math.pi * frequency
        period_s = 1.0 / frequency
        
        # Simulationszeit auf 200ms begrenzen (vom Benutzer gewünscht)
        simulation_time = 0.2  # 200ms
        
        # Warmup entfernt: Frames zeigen die tatsächliche Ausbreitung ab t=0
        # Die Wellenausbreitung startet direkt ab t=0, ohne Warmup-Verschiebung
        warmup_periods = 0  # Kein Warmup mehr
        warmup_time = 0.0
        total_time = simulation_time
        total_steps = int(round(total_time / dt))
        
        # Frames gleichmäßig über die Simulationszeit verteilen
        # WICHTIG: Frames zeigen die tatsächliche Ausbreitung ab t=0
        # Frame 0 = t=0 (nur Quellen, keine Ausbreitung)
        # Frames 1..frames_per_period = gleichmäßig über simulation_time verteilt (ab t=0)
        warmup_steps = int(round(warmup_time / dt))
        # Berechne Zeitpunkte für die Frames (gleichmäßig über simulation_time, beginnend bei t=0)
        # Diese Zeiten sind die tatsächliche Zeit seit Start (ohne Warmup-Verschiebung)
        frame_times = np.linspace(0.0, simulation_time, frames_per_period + 1)  # +1 für Frame 0
        # Konvertiere Zeitpunkte zu Step-Indizes
        # WICHTIG: Frames werden bei den tatsächlichen Zeitpunkten seit Start gesampelt
        # Das bedeutet: Frame 1 bei t=5.6ms sollte bei Step = 5.6ms/dt sein (nicht warmup_steps + 5.6ms/dt)
        frame_steps = (frame_times / dt).astype(int)
        frame_steps = np.clip(frame_steps, 0, total_steps - 1)
        # Frame 0 bei Step 0, Frame 1..N bei Steps entsprechend ihrer Zeit
        # Wenn mehrere Frames auf denselben Step fallen, wird der letzte verwendet
        # (das ist korrekt, da die Zeit-Zuordnung durch frame_times definiert ist)
        sample_map = {int(step): idx for idx, step in enumerate(frame_steps.tolist())}
        
        # WICHTIG: Die Wellenausbreitung startet ab t=0, nicht nach dem Warmup
        # Daher müssen wir sicherstellen, dass die Frames die korrekte Ausbreitung zeigen
        # Die Simulation läuft von Step 0 bis total_steps, und die Wellenausbreitung
        # findet während des gesamten Laufs statt, inklusive Warmup

        sources = self._collect_sources(frequency, simulation_x, simulation_y, grid_resolution)
        max_distance = math.sqrt((width / 2.0) ** 2 + (length / 2.0) ** 2)
        
        # Debug: Berechne erwartete Ausbreitungsdistanz für Frame 1
        if len(frame_times) > 1:
            frame1_time = frame_times[1]
            expected_distance = c * frame1_time
            print(f"[FDTD] Erwartete Ausbreitung Frame 1 (t={frame1_time*1000:.1f}ms): ~{expected_distance:.2f}m")
        
        domain_width_total = width + 2.0 * pml_phys_width
        domain_length_total = length + 2.0 * pml_phys_width
        print(f"[FDTD] Simulation: {total_steps} Steps, {steps_per_period} Steps/Periode, {len(sources)} Quellen")
        print(f"[FDTD] Feldfläche: {width:.1f}m x {length:.1f}m, Domäne (inkl. 10m-PML): {domain_width_total:.1f}m x {domain_length_total:.1f}m, max. Distanz: {max_distance:.1f}m, PML-Breite ~{pml_phys_width:.1f}m")
        print(f"[FDTD] Simulationszeit: {simulation_time*1000:.1f}ms (0-200ms, kein Warmup)")
        print(f"[FDTD] Frame-Zeiten: {[f'{t*1000:.1f}ms' for t in frame_times[:5]]} ... {[f'{t*1000:.1f}ms' for t in frame_times[-3:]]} (über {simulation_time*1000:.1f}ms)")
        print(f"[FDTD] Frame-Steps: {[int(s) for s in frame_steps[:3]]} ... {[int(s) for s in frame_steps[-3:]]} (von 0 bis {total_steps-1})")

        pressure_prev = np.zeros((ny, nx), dtype=np.float32)
        pressure_curr = np.zeros_like(pressure_prev)
        pressure_next = np.zeros_like(pressure_prev)
        # +1 für Frame 0 (t=0)
        snapshots_full = np.zeros((frames_per_period + 1, ny, nx), dtype=np.float32)
        
        # Frame 0: t=0 - nur Quellen, keine Ausbreitung
        time_s_0 = 0.0
        for src in sources:
            source_value = np.real(src["amplitude"] * np.exp(1j * omega * time_s_0))
            snapshots_full[0, src["iy"], src["ix"]] = source_value

        # FDTD-Koeffizient: (c·dt/dx)² - muss die tatsächliche Grid-Schrittweite verwenden!
        coeff = (c * dt / grid_resolution) ** 2
        
        i_coords = np.arange(ny, dtype=np.float32)[:, np.newaxis]
        j_coords = np.arange(nx, dtype=np.float32)[np.newaxis, :]
        dist_top = i_coords
        dist_bottom = (ny - 1) - i_coords
        dist_left = j_coords
        dist_right = (nx - 1) - j_coords
        min_dist = np.minimum(
            np.minimum(dist_top, dist_bottom),
            np.minimum(dist_left, dist_right)
        ).astype(np.float32)
        
        damping_factor = np.ones((ny, nx), dtype=np.float32)
        pml_mask = min_dist < padding_cells
        if np.any(pml_mask):
            normalized = (padding_cells - min_dist[pml_mask]) / padding_cells
            normalized = np.clip(normalized, 0.0, 1.0)
            # Polynomprofil (σ ~ normalized^profile_order)
            sigma = normalized ** profile_order
            max_sigma = -math.log(max(target_amp, 1e-8))
            sigma = sigma * max_sigma
            damping_factor[pml_mask] = np.exp(-sigma).astype(np.float32)

        debug_pml = bool(getattr(self.settings, "fdtd_debug_pml", False))
        debug_pml_steps = int(getattr(self.settings, "fdtd_debug_pml_steps", 40) or 40)
        
        print(f"[FDTD] Grid-Auflösung: dx={dx:.3f}m, dy={dy:.3f}m (gewählt: {grid_resolution:.3f}m)")
        print(f"[FDTD] CFL-Koeffizient: {coeff:.6f}, dt={dt:.6e} s, Wellenausbreitung: {c:.1f} m/s")
        print(f"[FDTD] Randbedingungen: Absorbierend mit Dämpfungszone (Breite: {padding_cells} Zellen, ~{pml_phys_width:.2f}m, Ziel {min_phys_width:.2f}m) – vollständig außerhalb des Innenbereichs")

        for step in range(total_steps):
            """
            FDTD-Zeitschritt: Lösung der 2D-Wellengleichung
            
            Physikalische Gleichung:
                ∂²p/∂t² = c² · (∂²p/∂x² + ∂²p/∂y²)
            
            Numerische Diskretisierung (Leap-Frog-Verfahren):
                p^{n+1} = 2·p^n - p^{n-1} + (c·dt/dx)² · ∇²p^n
            
            Dabei ist:
                - p^n = Druck zum Zeitpunkt t = n·dt
                - dt = Zeitschritt (CFL-bedingt: dt ≤ dx/(c·√2))
                - dx = räumliche Auflösung
                - c = Schallgeschwindigkeit
                - ∇²p = Laplace-Operator (2. Ableitung im Raum)
            """
            
            # Berechne Laplace-Operator (2. räumliche Ableitung) mit zentralen Differenzen
            # ∇²p ≈ (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - 4·p[i,j]) / dx²
            laplace = (
                pressure_curr[0:-2, 1:-1]      # p[i-1, j] (oben)
                + pressure_curr[2:, 1:-1]      # p[i+1, j] (unten)
                + pressure_curr[1:-1, 0:-2]    # p[i, j-1] (links)
                + pressure_curr[1:-1, 2:]      # p[i, j+1] (rechts)
                - 4.0 * pressure_curr[1:-1, 1:-1]  # -4·p[i, j] (Zentrum)
            )
            
            # Leap-Frog-Zeitschritt: Berechne p^{n+1} aus p^n und p^{n-1}
            # Dies simuliert die Wellenausbreitung: Druckänderung führt zu Beschleunigung,
            # Beschleunigung führt zu neuer Position (Druck)
            pressure_next[1:-1, 1:-1] = (
                2.0 * pressure_curr[1:-1, 1:-1]    # 2·p^n (aktueller Zustand)
                - pressure_prev[1:-1, 1:-1]         # -p^{n-1} (vorheriger Zustand)
                + coeff * laplace                    # + (c·dt/dx)² · ∇²p^n (räumliche Kopplung)
            )

            # Kontinuierliche Quellenanregung: Lautsprecher strahlt Sinus-Welle ab
            # p_source(t) = Re(A · e^{iωt}) = A_real · cos(ωt) - A_imag · sin(ωt)
            # Dabei ist A die komplexe Amplitude aus den FEM-Panel-Drives (mit Phase)
            time_s = step * dt
            for src in sources:
                # Realteil der komplexen Amplitude * exp(iωt)
                # Dies erzeugt eine kontinuierliche Sinus-Welle an der Quellposition
                source_value = np.real(src["amplitude"] * np.exp(1j * omega * time_s))
                pressure_next[src["iy"], src["ix"]] += source_value

            # Absorbierende Randbedingungen: Dämpfung NUR EINMAL pro Schritt anwenden
            # Die Dämpfung wird nach dem Zeitschritt angewendet, um Wellen in der Dämpfungszone zu absorbieren
            # WICHTIG: Dämpfung nur auf pressure_next anwenden (nicht auf curr/prev, um Überdämpfung zu vermeiden)
            pressure_next = pressure_next * damping_factor

            frame_idx = sample_map.get(step)

            if debug_pml and (step < debug_pml_steps or frame_idx is not None):
                inner_vals = pressure_next[inner_slice_y, inner_slice_x]
                inner_max = float(np.max(np.abs(inner_vals)))
                transition_strip = pressure_next[
                    inner_slice_y.start - 1:inner_slice_y.start + 1,
                    inner_slice_x,
                ]
                transition_max = float(np.max(np.abs(transition_strip)))
                pml_vals = pressure_next[pml_mask]
                pml_max = float(np.max(np.abs(pml_vals))) if pml_vals.size else 0.0
                ratio = transition_max / (inner_max + 1e-20)
                print(
                    f"[DEBUG PML] Step {step:5d}: inner_max={inner_max:8.3e}, "
                    f"transition_max={transition_max:8.3e} (ratio {ratio:6.3f}), "
                    f"pml_max={pml_max:8.3e}"
                )

            # Speichere Snapshots für die Frames
            if frame_idx is not None:
                snapshots_full[frame_idx] = pressure_next.copy()
                # Berechne Zeit für diesen Frame (tatsächliche Zeit seit Start)
                frame_time = frame_times[frame_idx]
                actual_time = step * dt
                # Debug: Zeige Werte für alle Frames
                p_min = float(np.min(pressure_next))
                p_max = float(np.max(pressure_next))
                p_mean = float(np.mean(np.abs(pressure_next)))
                # Prüfe ob Wellenausbreitung sichtbar ist (Druck sollte sich vom Zentrum ausbreiten)
                center_y, center_x = ny // 2, nx // 2
                center_pressure = float(pressure_next[center_y, center_x])
                edge_pressure = float(np.mean([pressure_next[0, center_x], pressure_next[-1, center_x], 
                                               pressure_next[center_y, 0], pressure_next[center_y, -1]]))
                expected_dist = c * frame_time if frame_idx > 0 else 0.0
                print(f"[FDTD] Frame {frame_idx:2d} (t={frame_time*1000:6.1f}ms, t_actual={actual_time*1000:6.1f}ms, Step {step:5d}, erwartet ~{expected_dist:.1f}m): "
                      f"min={p_min:8.3e}, max={p_max:8.3e}, mean={p_mean:8.3e}, "
                      f"center={center_pressure:8.3e}, edge={edge_pressure:8.3e}")

            # Zeit-Update: verschiebe Felder
            # pressure_next wurde bereits gedämpft, daher nur rotieren ohne weitere Dämpfung
            pressure_prev, pressure_curr, pressure_next = pressure_curr, pressure_next, pressure_prev
            pressure_next.fill(0.0)

        inner_slice_y = slice(padding_cells, padding_cells + ny_inner)
        inner_slice_x = slice(padding_cells, padding_cells + nx_inner)
        snapshots = snapshots_full[:, inner_slice_y, inner_slice_x].copy()
        
        result = {
            "frequency": frequency,
            "frames_per_period": frames_per_period,  # Anzahl Frames für eine Periode (ohne t=0)
            "total_frames": frames_per_period + 1,  # Inkl. Frame 0 (t=0)
            "sound_field_x": sound_field_x.tolist(),
            "sound_field_y": sound_field_y.tolist(),
            "pressure_frames": snapshots.tolist(),
            "time_step": dt,
            "steps_per_period": steps_per_period,
        }
        print(f"[DEBUG FDTD] _run_fdtd_simulation() ENDE: {result['total_frames']} Frames erstellt, Grid {len(sound_field_x)}x{len(sound_field_y)}")
        return result

    def _collect_sources(self, frequency: float, x_coords: np.ndarray, y_coords: np.ndarray, resolution: float):
        """
        Sammelt Quellen direkt aus Speaker-Arrays und Balloon-Daten.
        Keine Abhängigkeit von FEM-Ergebnissen mehr.
        """
        if self._data_container is None:
            raise RuntimeError("Data container not set – please call set_data_container() first.")
        
        speaker_arrays = getattr(self.settings, "speaker_arrays", None)
        if not isinstance(speaker_arrays, dict) or not speaker_arrays:
            raise RuntimeError("No speaker arrays found. Please add at least one speaker array first.")

        x_min = x_coords[0]
        y_min = y_coords[0]
        nx = len(x_coords)
        ny = len(y_coords)
        
        # Skalierungsfaktor für FDTD-Quellen (kann in Settings angepasst werden)
        scale_factor = float(getattr(self.settings, "fdtd_source_scale", 0.0001) or 0.0001)
        
        # Offsets für Pegelberechnung (wie in SoundFieldCalculator.py)
        a_source_db = float(getattr(self.settings, "a_source_db", 0.0) or 0.0)
        
        # Standard-Panel-Dimensionen (falls nicht in Metadaten vorhanden)
        default_width = float(getattr(self.settings, "fem_default_panel_width", 0.6) or 0.6)
        default_height = float(getattr(self.settings, "fem_default_panel_height", 0.5) or 0.5)

        sources = []
        source_positions = {}  # Dictionary zum Zusammenfassen von Quellen an derselben Position
        
        # Debug: Zähle Arrays
        active_arrays = [k for k, arr in speaker_arrays.items() if not (arr.mute or arr.hide)]
        print(f"[FDTD] _collect_sources: {len(active_arrays)} aktive Arrays gefunden: {active_arrays}")
        
        # Iteriere über alle Speaker-Arrays
        for array_key, speaker_array in speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            # Array-Positionen und -Einstellungen
            source_position_x = getattr(
                speaker_array,
                'source_position_calc_x',
                getattr(speaker_array, 'source_position_x', None),
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                getattr(speaker_array, 'source_position_y', None),
            )
            source_position_z = getattr(
                speaker_array,
                'source_position_calc_z',
                getattr(speaker_array, 'source_position_z', None),
            )
            
            if source_position_x is None or source_position_y is None:
                continue
            
            array_gain = float(getattr(speaker_array, 'gain', 0.0) or 0.0)
            source_level = np.array(speaker_array.source_level) if hasattr(speaker_array, 'source_level') else np.array([0.0])
            
            # Iteriere über alle Lautsprecher im Array
            for isrc, speaker_name in enumerate(speaker_array.source_polar_pattern):
                if isrc >= len(source_position_x) or isrc >= len(source_position_y):
                    continue
                
                # Position des Lautsprechers
                cx = float(source_position_x[isrc])
                cy = float(source_position_y[isrc])
                cz = float(source_position_z[isrc]) if source_position_z is not None and isrc < len(source_position_z) else 0.0
                
                # Grid-Indizes
                ix = int(round((cx - x_min) / resolution))
                iy = int(round((cy - y_min) / resolution))
                ix = int(np.clip(ix, 1, nx - 2))
                iy = int(np.clip(iy, 1, ny - 2))
                
                # Hole Balloon-Daten für 0° (on-axis)
                try:
                    balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
                    if not balloon_data or not isinstance(balloon_data, dict):
                        continue
                    
                    # Finde nächstgelegene Frequenz (explizite Prüfung für NumPy-Arrays)
                    freqs = balloon_data.get("freqs")
                    if freqs is None:
                        freqs = balloon_data.get("frequencies")
                    if freqs is None:
                        continue
                    freqs = np.asarray(freqs)
                    if freqs.size == 0:
                        continue
                    freq_idx = np.argmin(np.abs(freqs - frequency))
                    target_freq = float(freqs[freq_idx])
                    
                    # Hole Magnitude und Phase für 0° (on-axis)
                    magnitude = balloon_data.get("magnitude")
                    phase = balloon_data.get("phase")
                    vertical_angles = balloon_data.get("vertical_angles")
                    horizontal_angles = balloon_data.get("horizontal_angles")
                    
                    if magnitude is None or phase is None:
                        continue
                    
                    magnitude = np.asarray(magnitude)
                    phase = np.asarray(phase)
                    
                    # Balloon-Daten-Struktur: [vertical, horizontal, frequency] (3D) oder [vertical, horizontal] (2D nach Frequenz-Slicing)
                    if magnitude.ndim == 3 and phase.ndim == 3:
                        # 3D: [vertical, horizontal, frequency] - wähle Frequenz-Slice
                        if freq_idx >= magnitude.shape[2] or freq_idx >= phase.shape[2]:
                            continue
                        mag_2d = magnitude[:, :, freq_idx]  # [vertical, horizontal]
                        phase_2d = phase[:, :, freq_idx]
                    elif magnitude.ndim == 2 and phase.ndim == 2:
                        # 2D: bereits für eine Frequenz gesliced [vertical, horizontal]
                        mag_2d = magnitude
                        phase_2d = phase
                    else:
                        continue
                    
                    # Finde 0° (on-axis): vertical=0°, horizontal=0°
                    if vertical_angles is not None:
                        vertical_angles = np.asarray(vertical_angles, dtype=float)
                        v_idx = int(np.abs(vertical_angles).argmin())  # Nächstgelegener zu 0°
                    else:
                        v_idx = 0  # Fallback: erster Index
                    
                    if horizontal_angles is not None:
                        horizontal_angles = np.asarray(horizontal_angles, dtype=float)
                        h_idx = int(np.abs(horizontal_angles).argmin())  # Nächstgelegener zu 0°
                    else:
                        h_idx = 0  # Fallback: erster Index
                    
                    # Sicherheitsprüfung
                    if v_idx >= mag_2d.shape[0] or h_idx >= mag_2d.shape[1]:
                        continue
                    
                    mag_db = float(mag_2d[v_idx, h_idx])
                    phase_deg = float(phase_2d[v_idx, h_idx])
                    
                except Exception as e:
                    print(f"[FDTD] Warnung: Konnte Balloon-Daten für {speaker_name} nicht laden: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Berechne Gesamt-Pegel (wie in SoundFieldCalculator.py)
                speaker_gain = float(source_level[isrc]) if isrc < len(source_level) else 0.0
                total_gain_db = mag_db + array_gain + speaker_gain + a_source_db
                
                # Konvertiere zu komplexem Druck (wie in FEM)
                # p = 20µPa · 10^(dB/20) · exp(i·phase)
                p_amp = 20e-6 * 10 ** (total_gain_db / 20.0)
                phase_rad = math.radians(phase_deg)
                pressure_complex = p_amp * np.exp(1j * phase_rad)
                
                # Skaliere für FDTD
                scaled_amplitude = pressure_complex * scale_factor
                
                # Panel-Dimensionen aus Metadaten (falls verfügbar)
                try:
                    cabinet_meta = self._data_container.get_cabinet_metadata(speaker_name)
                    panel_width = float(cabinet_meta.get("front_width", default_width)) if cabinet_meta else default_width
                    panel_height = float(cabinet_meta.get("front_height", default_height)) if cabinet_meta else default_height
                except Exception:
                    panel_width = default_width
                    panel_height = default_height
                
                # Verteile Quelle über mehrere Grid-Punkte
                source_radius_pixels = max(1, int(round(max(panel_width, panel_height) / (2.0 * resolution))))
                source_radius_pixels = min(source_radius_pixels, 5)  # Max 5 Pixel Radius
                
                # Füge Hauptquelle hinzu
                sources.append({
                    "ix": ix,
                    "iy": iy,
                    "amplitude": scaled_amplitude,
                    "radius": source_radius_pixels,
                })
                
                # Füge zusätzliche Quellen in der Nähe hinzu (für bessere Verteilung)
                if source_radius_pixels > 1:
                    for dx in range(-source_radius_pixels, source_radius_pixels + 1):
                        for dy in range(-source_radius_pixels, source_radius_pixels + 1):
                            if dx == 0 and dy == 0:
                                continue
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist > source_radius_pixels:
                                continue
                            weight = 1.0 / (1.0 + dist)  # Gewichtung nach Distanz
                            new_ix = int(np.clip(ix + dx, 1, nx - 2))
                            new_iy = int(np.clip(iy + dy, 1, ny - 2))
                            sources.append({
                                "ix": new_ix,
                                "iy": new_iy,
                                "amplitude": scaled_amplitude * weight,
                                "radius": 0,
                            })

        if not sources:
            raise RuntimeError("Keine gültigen FDTD-Quellen gefunden.")
        
        # Debug: Zeige Skalierung
        if sources:
            sample_amp = abs(sources[0]["amplitude"])
            print(f"[FDTD] {len(sources)} Quellen, Beispiel-Amplitude: {sample_amp:.3e} Pa (scale={scale_factor})")
        
        return sources

    def _select_primary_frequency(self) -> Optional[float]:
        """
        Wählt die primäre Frequenz für FDTD aus Settings.
        Keine Abhängigkeit von FEM-Ergebnissen mehr.
        """
        # Prüfe FDTD-spezifische Einstellung (kann auch fem_calculate_frequency heißen für Kompatibilität)
        target_frequency = getattr(self.settings, "fdtd_calculate_frequency", None)
        if target_frequency is None:
            target_frequency = getattr(self.settings, "fem_calculate_frequency", None)  # Fallback
        if target_frequency is not None:
            try:
                return float(target_frequency)
            except (TypeError, ValueError):
                pass
        
        # Fallback: Verwende calculate_frequency
        fallback_freq = getattr(self.settings, "calculate_frequency", None)
        if fallback_freq is not None:
            try:
                return float(fallback_freq)
            except (TypeError, ValueError):
                pass
        
        return None
    
    def _get_output_plane_height(self) -> float:
        """Höhe der Auswerte-Ebene (für 2D-Visualisierung)."""
        # Prüfe explizite Einstellung (kann auch fdtd_output_plane_height heißen)
        explicit = getattr(self.settings, "fdtd_output_plane_height", None)
        if explicit is None:
            explicit = getattr(self.settings, "fem_output_plane_height", None)  # Fallback für Kompatibilität
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        
        # Berechne Panel-Z-Positionen direkt aus Speaker-Arrays (keine FEM-Abhängigkeit)
        speaker_arrays = getattr(self.settings, "speaker_arrays", None)
        if isinstance(speaker_arrays, dict) and speaker_arrays:
            panel_z_centers = []
            for array_key, speaker_array in speaker_arrays.items():
                if speaker_array.mute or speaker_array.hide:
                    continue
                source_position_z = getattr(
                    speaker_array,
                    'source_position_calc_z',
                    getattr(speaker_array, 'source_position_z', None),
                )
                if source_position_z is not None:
                    for z_pos in source_position_z:
                        if z_pos is not None:
                            try:
                                panel_z_centers.append(float(z_pos))
                            except (TypeError, ValueError):
                                pass
            if panel_z_centers:
                avg_panel_z = float(np.mean(panel_z_centers))
                default_plane = avg_panel_z + 1.2  # Ohrhöhe über Panel
                listener_height = getattr(self.settings, "listener_height", None)
                if listener_height is not None:
                    try:
                        return float(listener_height)
                    except (TypeError, ValueError):
                        pass
                return default_plane
        
        # Fallback: Standard-Höhe
        return 1.2  # Standard Ohrhöhe
    
    def _apply_2p5d_correction_complex(
        self,
        pressure_grid_complex: np.ndarray,
        sound_field_x: np.ndarray,
        sound_field_y: np.ndarray,
    ) -> np.ndarray:
        """Skaliert das 2D-Ergebnis auf ein 2.5D-Feld (~1/r-Abfall) für komplexe Werte."""
        if pressure_grid_complex is None:
            return pressure_grid_complex
        # Prüfe ob 2.5D-Korrektur aktiviert ist (kann auch fdtd_enable_2p5d_correction heißen)
        enable_correction = getattr(self.settings, "fdtd_enable_2p5d_correction", None)
        if enable_correction is None:
            enable_correction = getattr(self.settings, "fem_enable_2p5d_correction", True)  # Fallback
        if not enable_correction:
            return pressure_grid_complex
        
        # Berechne Panel-Zentren direkt aus Speaker-Arrays (keine FEM-Abhängigkeit)
        speaker_arrays = getattr(self.settings, "speaker_arrays", None)
        if not isinstance(speaker_arrays, dict) or not speaker_arrays:
            return pressure_grid_complex
        
        centers = []
        for array_key, speaker_array in speaker_arrays.items():
            if speaker_array.mute or speaker_array.hide:
                continue
            source_position_x = getattr(
                speaker_array,
                'source_position_calc_x',
                getattr(speaker_array, 'source_position_x', None),
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                getattr(speaker_array, 'source_position_y', None),
            )
            source_position_z = getattr(
                speaker_array,
                'source_position_calc_z',
                getattr(speaker_array, 'source_position_z', None),
            )
            if source_position_x is not None and source_position_y is not None:
                for i in range(min(len(source_position_x), len(source_position_y))):
                    try:
                        x = float(source_position_x[i])
                        y = float(source_position_y[i])
                        z = float(source_position_z[i]) if source_position_z is not None and i < len(source_position_z) else 0.0
                        centers.append([x, y, z])
                    except (TypeError, ValueError, IndexError):
                        continue
        
        if not centers:
            return pressure_grid_complex
        
        centers = np.array(centers, dtype=float)
        reference_distance = float(
            getattr(self.settings, "fdtd_2p5d_reference_distance", None) or
            getattr(self.settings, "fem_2p5d_reference_distance", 1.0) or 1.0
        )
        min_distance = float(
            getattr(self.settings, "fdtd_2p5d_min_distance", None) or
            getattr(self.settings, "fem_2p5d_min_distance", 0.25) or 0.25
        )
        
        grid_x, grid_y = np.meshgrid(sound_field_x, sound_field_y)
        plane_height = self._get_output_plane_height()
        min_dist_sq = None
        for center in centers:
            dx = grid_x - center[0]
            dy = grid_y - center[1]
            dz = plane_height - center[2] if len(center) >= 3 else plane_height
            dist_sq = dx * dx + dy * dy + dz * dz
            if min_dist_sq is None:
                min_dist_sq = dist_sq
            else:
                min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        
        if min_dist_sq is None:
            return pressure_grid_complex
        
        min_dist_sq = np.maximum(min_dist_sq, min_distance ** 2)
        radial_distance = np.sqrt(min_dist_sq)
        correction = np.sqrt(reference_distance / radial_distance)
        correction = np.clip(correction, 0.0, 1e6)
        pressure_grid_complex = pressure_grid_complex * correction
        return pressure_grid_complex


__all__ = ["SoundFieldCalculatorFDTD"]

