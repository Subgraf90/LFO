from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEBUG_SURFACE_GEOMETRY = bool(int(os.environ.get("LFO_DEBUG_SURFACE_GEOMETRY", "1")))


@dataclass
class PlotSurfaceGeometry:
    plot_x: np.ndarray
    plot_y: np.ndarray
    plot_values: np.ndarray
    z_coords: Optional[np.ndarray]
    surface_mask: Optional[np.ndarray]
    source_x: np.ndarray
    source_y: np.ndarray
    requires_resample: bool
    was_upscaled: bool

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.plot_y.size, self.plot_x.size)


def derive_surface_plane(
    points: List[Dict[str, float]],
    *,
    tol: float = 1e-4,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Analysiert die Z-Werte einer Surface-Definition und bestimmt,
    ob die Fläche plan ist. Zulässig sind:

    - Konstante Höhe (alle Z-Werte identisch)
    - Lineare Steigung entlang der X-Achse (Z = a * X + b)
    - Lineare Steigung entlang der Y-Achse (Z = a * Y + b)

    Args:
        points: Liste von Surface-Punkten mit 'x', 'y', 'z'
        tol: Toleranz für numerische Vergleiche

    Returns:
        (model, error_message)
        - model: Dict mit Planar-Informationen oder None bei Fehler
        - error_message: Beschreibung, falls Fläche nicht plan ist
    """

    if not points:
        return (
            {
                "mode": "constant",
                "base": 0.0,
                "slope": 0.0,
                "intercept": 0.0,
            },
            None,
        )

    coords = np.array(
        [
            (
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            )
            for point in points
        ],
        dtype=float,
    )
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    z_vals = coords[:, 2]

    z_span = float(np.ptp(z_vals))  # max - min
    if z_span <= tol:
        base = float(np.mean(z_vals))
        return (
            {
                "mode": "constant",
                "base": base,
                "slope": 0.0,
                "intercept": base,
            },
            None,
        )

    # Prüfe Steigung entlang X: Alle Punkte mit gleichem X benötigen identisches Z
    if _is_axis_planar(x_vals, z_vals, tol):
        slope, intercept = _fit_linear_relation(x_vals, z_vals)
        return (
            {
                "mode": "x",
                "base": intercept,
                "slope": slope,
                "intercept": intercept,
            },
            None,
        )

    # Prüfe Steigung entlang Y: Alle Punkte mit gleichem Y benötigen identisches Z
    if _is_axis_planar(y_vals, z_vals, tol):
        slope, intercept = _fit_linear_relation(y_vals, z_vals)
        return (
            {
                "mode": "y",
                "base": intercept,
                "slope": slope,
                "intercept": intercept,
            },
            None,
        )

    # Allgemeine Ebene Z = ax + by + c
    plane_model = _fit_planar_surface(x_vals, y_vals, z_vals)
    if plane_model is not None:
        predicted = (
            plane_model["slope_x"] * x_vals
            + plane_model["slope_y"] * y_vals
            + plane_model["intercept"]
        )
        max_err = float(np.max(np.abs(z_vals - predicted)))
        if max_err <= tol:
            return plane_model, None

    error = (
        "Die Z-Werte der Fläche müssen entweder konstant sein oder nur entlang "
        "einer Achse (X oder Y) linear variieren oder auf einer Ebene liegen."
    )
    return None, error


def build_planar_model(
    points: List[Dict[str, float]],
    *,
    tol: float = 1e-4,
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Bestimmt das beste planare Modell (konstant, X-Steigung, Y-Steigung) und
    gibt an, ob Anpassungen der bestehenden Z-Werte nötig sind.

    Returns:
        (model, needs_adjustment)
    """
    if not points:
        return (
            {
                "mode": "constant",
                "base": 0.0,
                "slope": 0.0,
                "intercept": 0.0,
            },
            False,
        )

    coords = np.array(
        [
            (
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            )
            for point in points
        ],
        dtype=float,
    )
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    z_vals = coords[:, 2]

    candidates = []

    def add_candidate(
        mode: str,
        slope: float,
        intercept: float,
        predicted: np.ndarray,
        *,
        slope_y: float = 0.0,
    ) -> None:
        residuals = z_vals - predicted
        mse = float(np.mean(residuals**2))
        max_err = float(np.max(np.abs(residuals)))
        candidates.append(
            {
                "mode": mode,
                "slope": float(slope),
                "slope_y": float(slope_y),
                "intercept": float(intercept),
                "base": float(intercept) if mode != "constant" else float(intercept),
                "mse": mse,
                "max_err": max_err,
            }
        )

    # Konstante Fläche
    base = float(np.mean(z_vals)) if len(z_vals) else 0.0
    predicted_const = np.full_like(z_vals, base)
    add_candidate("constant", 0.0, base, predicted_const)

    # Steigung entlang X (nur wenn Variation vorhanden)
    if np.ptp(x_vals) > tol:
        slope_x, intercept_x = _fit_linear_relation(x_vals, z_vals)
        predicted_x = slope_x * x_vals + intercept_x
        add_candidate("x", slope_x, intercept_x, predicted_x)

    # Steigung entlang Y
    if np.ptp(y_vals) > tol:
        slope_y, intercept_y = _fit_linear_relation(y_vals, z_vals)
        predicted_y = slope_y * y_vals + intercept_y
        add_candidate("y", slope_y, intercept_y, predicted_y)

    # Allgemeine Ebene
    plane_model = _fit_planar_surface(x_vals, y_vals, z_vals)
    if plane_model is not None:
        predicted_plane = (
            plane_model["slope_x"] * x_vals
            + plane_model["slope_y"] * y_vals
            + plane_model["intercept"]
        )
        add_candidate(
            "xy",
            plane_model["slope_x"],
            plane_model["intercept"],
            predicted_plane,
            slope_y=plane_model["slope_y"],
        )

    # Wähle Modell mit geringster MSE
    best_model = min(candidates, key=lambda c: c["mse"])
    needs_adjustment = best_model["max_err"] > tol

    if best_model["mode"] == "xy":
        model = {
            "mode": "xy",
            "slope_x": best_model["slope"],
            "slope_y": best_model.get("slope_y", 0.0),
            "intercept": best_model["intercept"],
            "base": best_model["base"],
        }
    else:
        model = {
            "mode": best_model["mode"],
            "slope": best_model["slope"],
            "intercept": best_model["intercept"],
            "base": best_model["base"],
        }
    return model, needs_adjustment


def evaluate_surface_plane(model: Dict[str, float], x: float, y: float) -> float:
    """
    Berechnet den Z-Wert einer Fläche anhand des Planar-Modells.
    """
    mode = model.get("mode")
    if mode == "constant":
        return float(model.get("base", 0.0))
    if mode == "x":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return float(slope * x + intercept)
    if mode == "y":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return float(slope * y + intercept)
    if mode == "xy":
        slope_x = float(model.get("slope_x", model.get("slope", 0.0)))
        slope_y = float(model.get("slope_y", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return float(slope_x * x + slope_y * y + intercept)
    return float(model.get("base", 0.0))


def _is_axis_planar(axis_values: np.ndarray, z_values: np.ndarray, tol: float) -> bool:
    """
    Prüft, ob Z nur von einer Achse abhängt.
    """
    rounded_axis = np.round(axis_values, decimals=6)
    unique_vals = np.unique(rounded_axis)
    if unique_vals.size < 2:
        # Alle Punkte haben praktisch gleiche Axis-Koordinate → bereits über konstante Fläche abgedeckt
        return False

    for value in unique_vals:
        mask = rounded_axis == value
        z_span = np.ptp(z_values[mask])
        if z_span > tol:
            return False
    return True


def _fit_linear_relation(axis_values: np.ndarray, z_values: np.ndarray) -> Tuple[float, float]:
    """
    Bestimmt lineare Relation z = slope * axis + intercept (Least Squares).
    """
    if np.ptp(axis_values) <= 1e-9:
        return 0.0, float(np.mean(z_values))
    slope, intercept = np.polyfit(axis_values, z_values, 1)
    return float(slope), float(intercept)


def _fit_planar_surface(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Fit einer Ebene z = ax + by + c über alle Punkte.
    """
    if len(x_vals) < 3:
        return None

    # Matrix [x y 1]
    A = np.column_stack((x_vals, y_vals, np.ones_like(x_vals)))
    try:
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, z_vals, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3 and np.ptp(z_vals) > 1e-9:
        # Punkte degeneriert (z. B. alle gleichen x+y), Ebene nicht eindeutig
        return None

    slope_x = float(coeffs[0])
    slope_y = float(coeffs[1])
    intercept = float(coeffs[2])
    return {
        "mode": "xy",
        "slope_x": slope_x,
        "slope_y": slope_y,
        "intercept": intercept,
        "base": intercept,
    }


def generate_surface_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Erzeugt die X-, Y- und Z-Gitter für eine Surface basierend auf den
    eingegebenen Koordinaten.

    Args:
        x: 1D-Array der X-Koordinaten.
        y: 1D-Array der Y-Koordinaten.
        z_coords: Optionales 2D-Array mit Z-Werten (Shape len(y) x len(x)).

    Returns:
        Tuple (xm, ym, zm) jeweils als 2D-Arrays mit Shape [len(y), len(x)].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xm, ym = np.meshgrid(x, y, indexing="xy")

    if z_coords is not None:
        try:
            z_arr = np.asarray(z_coords, dtype=float)
            if z_arr.shape != (len(y), len(x)):
                z_arr = z_arr.reshape(len(y), len(x))
        except Exception:
            z_arr = np.zeros_like(xm, dtype=float)
    else:
        z_arr = np.zeros_like(xm, dtype=float)

    return xm, ym, z_arr


def build_surface_mesh(
    x: np.ndarray,
    y: np.ndarray,
    scalars: np.ndarray,
    *,
    z_coords: Optional[np.ndarray] = None,
    surface_mask: Optional[np.ndarray] = None,
    pv_module: Any = None,
):
    """
    Baut ein PyVista-PolyData ausschließlich aus der Topfläche des Gitters (keine Seitenflächen).
    """
    if pv_module is None:
        try:
            import pyvista as pv_module  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyVista wird benötigt, um ein Surface-Mesh zu erstellen."
            ) from exc

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(scalars, dtype=float)
    if z_coords is not None:
        z = np.asarray(z_coords, dtype=float)
    else:
        z = None

    ny, nx = values.shape
    if ny != len(y) or nx != len(x):
        raise ValueError("scalars müssen Shape (len(y), len(x)) besitzen.")

    # Erzeuge Punktkoordinaten (nur Topfläche)
    xm, ym = np.meshgrid(x, y, indexing="xy")
    if z is not None and z.shape == (ny, nx):
        zm = z
    elif z is not None and z.size == ny * nx:
        zm = z.reshape(ny, nx)
    else:
        zm = np.zeros_like(xm, dtype=float)

    mask = None
    if surface_mask is not None:
        mask = np.asarray(surface_mask, dtype=bool)
        if mask.shape != (ny, nx):
            if mask.size == nx * ny:
                mask = mask.reshape(ny, nx)
            else:
                mask = None

    points = np.column_stack((xm.ravel(), ym.ravel(), zm.ravel()))

    # Definiere Quad-Zellen (nur horizontale Deckfläche)
    face_list: List[int] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if mask is not None:
                cell_mask = mask[j : j + 2, i : i + 2]
                if not np.all(cell_mask):
                    continue
            idx0 = j * nx + i
            idx1 = idx0 + 1
            idx2 = idx0 + nx + 1
            idx3 = idx0 + nx
            face_list.extend([4, idx0, idx1, idx2, idx3])

    faces = np.asarray(face_list, dtype=np.int64)

    if faces.size > 0:
        mesh = pv_module.PolyData(points, faces)
    else:
        # Zu wenig Punkte für Polygone → Einzelpunkte rendern
        mesh = pv_module.PolyData(points)

    mesh["plot_scalars"] = values.ravel()

    # Entferne vertikale Seitenflächen (Normale nahezu horizontal)
    try:
        norm_mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        normals = norm_mesh.cell_data.get("Normals")
        if normals is not None and normals.size:
            flat_mask = np.abs(normals[:, 2]) >= 0.2
            if not np.all(flat_mask):
                flat_indices = np.nonzero(flat_mask)[0]
                if flat_indices.size > 0:
                    mesh = mesh.extract_cells(flat_indices)
            if DEBUG_SURFACE_GEOMETRY:
                print(
                    "[SurfaceGeometry] normals:",
                    f"cells={mesh.n_cells}",
                    f"removed={normals.shape[0] - flat_mask.sum()}",
                )
    except Exception:
        pass

    return mesh


def prepare_plot_geometry(
    sound_field_x,
    sound_field_y,
    plot_values,
    *,
    settings,
    container=None,
    default_upscale: int = 3,
) -> PlotSurfaceGeometry:
    """
    Bereitet die Plot-Geometrie für den SPL-Renderer vor (Upscaling + Z-Interpolation).
    """
    source_x = np.asarray(sound_field_x, dtype=float)
    source_y = np.asarray(sound_field_y, dtype=float)
    values = np.asarray(plot_values, dtype=float)

    if values.shape != (len(source_y), len(source_x)):
        raise ValueError(
            "plot_values müssen die Shape (len(y), len(x)) besitzen."
        )

    requested_upscale = getattr(settings, "plot_upscale_factor", None)
    if requested_upscale is None:
        requested_upscale = default_upscale
    try:
        upscale_factor = int(requested_upscale)
    except (TypeError, ValueError):
        upscale_factor = default_upscale
    upscale_factor = max(1, upscale_factor)

    plot_x = source_x.copy()
    plot_y = source_y.copy()
    plot_vals = values.copy()

    z_coords = _extract_plot_z_coordinates(container, len(source_y), len(source_x))
    surface_mask = _extract_surface_mask(container, len(source_y), len(source_x))

    if (
        upscale_factor > 1
        and plot_x.size > 1
        and plot_y.size > 1
    ):
        orig_plot_x = plot_x.copy()
        orig_plot_y = plot_y.copy()
        expanded_x = _expand_axis_for_plot(plot_x, upscale_factor)
        expanded_y = _expand_axis_for_plot(plot_y, upscale_factor)
        plot_vals = _resample_values_to_grid(plot_vals, orig_plot_x, orig_plot_y, expanded_x, expanded_y)

        if z_coords is not None:
            z_coords = _resample_values_to_grid(z_coords, orig_plot_x, orig_plot_y, expanded_x, expanded_y)
        if surface_mask is not None:
            surface_mask = _resample_mask_to_grid(surface_mask, orig_plot_x, orig_plot_y, expanded_x, expanded_y)

        plot_x = expanded_x
        plot_y = expanded_y

    if surface_mask is None:
        print("[SurfaceGeometry] Fehler: Keine Surface-Maske vorhanden – breche Rendering ab.")
        raise RuntimeError("Surface mask missing in calculation data.")
    _debug_surface_info(settings, plot_x, plot_y, surface_mask, "calculation mask")

    requires_resample = not (
        np.array_equal(plot_x, source_x) and np.array_equal(plot_y, source_y)
    )
    was_upscaled = upscale_factor > 1 and requires_resample

    return PlotSurfaceGeometry(
        plot_x=plot_x,
        plot_y=plot_y,
        plot_values=plot_vals,
        z_coords=z_coords,
        surface_mask=surface_mask,
        source_x=source_x,
        source_y=source_y,
        requires_resample=requires_resample,
        was_upscaled=was_upscaled,
    )


def _extract_plot_z_coordinates(container, len_y: int, len_x: int) -> Optional[np.ndarray]:
    """
    Holt Z-Koordinaten aus container.calculation_spl['sound_field_z'], falls vorhanden.
    """
    if container is None or not hasattr(container, "calculation_spl"):
        return None
    calc_spl = getattr(container, "calculation_spl", None)
    if not isinstance(calc_spl, dict) or "sound_field_z" not in calc_spl:
        return None
    try:
        raw = calc_spl["sound_field_z"]
        if raw is None:
            return None
        z_coords = np.asarray(raw, dtype=float)
        if z_coords.shape == (len_y, len_x):
            return z_coords
        if z_coords.size == len_y * len_x:
            return z_coords.reshape(len_y, len_x)
    except Exception:
        return None
    return None


def _extract_surface_mask(container, len_y: int, len_x: int) -> Optional[np.ndarray]:
    if container is None or not hasattr(container, "calculation_spl"):
        return None
    calc_spl = getattr(container, "calculation_spl", None)
    if not isinstance(calc_spl, dict):
        return None
    mask = calc_spl.get("surface_mask")
    if mask is None:
        return None
    try:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == (len_y, len_x):
            return mask_arr
        if mask_arr.size == len_y * len_x:
            return mask_arr.reshape(len_y, len_x)
    except Exception:
        return None
    return None


def _expand_axis_for_plot(
    axis: np.ndarray,
    upscale_factor: int,
) -> np.ndarray:
    """
    Skaliert eine Achse durch lineare Interpolation auf eine feinere Auflösung.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.size <= 1 or upscale_factor <= 1:
        return axis

    expanded = [float(axis[0])]
    for idx in range(1, axis.size):
        start = float(axis[idx - 1])
        stop = float(axis[idx])
        if np.isclose(stop, start):
            segment = np.full(upscale_factor, start)
        else:
            segment = np.linspace(start, stop, upscale_factor + 1, dtype=float)[1:]
        expanded.extend(segment.tolist())

    return np.asarray(expanded, dtype=float)


def _resample_mask_to_grid(
    mask: np.ndarray,
    orig_x: np.ndarray,
    orig_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=float)
    resampled = _resample_values_to_grid(mask, orig_x, orig_y, target_x, target_y)
    return resampled >= 0.5


def _debug_surface_info(settings, plot_x, plot_y, mask, source: str) -> None:
    if not DEBUG_SURFACE_GEOMETRY:
        return
    active_ids = []
    surface_definitions = getattr(settings, "surface_definitions", {})
    if isinstance(surface_definitions, dict):
        for surface_id, surface_def in surface_definitions.items():
            if surface_def.get("hidden", False):
                continue
            if surface_def.get("enabled", False):
                active_ids.append(surface_id)
    mask_info = "None"
    if mask is not None:
        true_count = int(np.sum(mask))
        mask_info = f"mask true={true_count} / total={mask.size}"
    print(
        "[SurfaceGeometry] build_surface_mesh:",
        f"source={source}",
        f"active_surfaces={active_ids}",
        f"grid=({plot_x.size}x{plot_y.size})",
        mask_info,
    )


def _resample_values_to_grid(
    values: np.ndarray,
    orig_x: np.ndarray,
    orig_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """
    Resampelt ein 2D-Array per linearer Interpolation auf ein neues Grid.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return values

    orig_x = np.asarray(orig_x, dtype=float)
    orig_y = np.asarray(orig_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)

    if orig_x.size <= 1 or orig_y.size <= 1:
        return values
    if target_x.size == orig_x.size and target_y.size == orig_y.size:
        return values

    intermediate = np.empty((values.shape[0], target_x.size), dtype=float)
    for iy in range(values.shape[0]):
        intermediate[iy, :] = np.interp(
            target_x,
            orig_x,
            values[iy, :],
            left=values[iy, 0],
            right=values[iy, -1],
        )

    resampled = np.empty((target_y.size, target_x.size), dtype=float)
    for ix in range(intermediate.shape[1]):
        resampled[:, ix] = np.interp(
            target_y,
            orig_y,
            intermediate[:, ix],
            left=intermediate[0, ix],
            right=intermediate[-1, ix],
        )

    return resampled

