from __future__ import annotations

from dataclasses import dataclass, field
import math
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


@dataclass
class SurfaceDefinition:
    surface_id: str
    name: str
    enabled: bool
    hidden: bool
    locked: bool
    points: List[Dict[str, float]] = field(default_factory=list)
    plane_model: Dict[str, float] | None = None
    color: str | None = None
    group_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "name": self.name,
            "enabled": self.enabled,
            "hidden": self.hidden,
            "locked": self.locked,
            "points": self.points,
            "plane_model": self.plane_model,
            "color": self.color,
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, surface_id: str, data: Dict[str, Any]) -> "SurfaceDefinition":
        return cls(
            surface_id=surface_id,
            name=str(data.get("name", surface_id)),
            enabled=bool(data.get("enabled", False)),
            hidden=bool(data.get("hidden", False)),
            locked=bool(data.get("locked", False)),
            points=list(data.get("points", [])),
            plane_model=data.get("plane_model"),
            color=data.get("color"),
            group_id=data.get("group_id") or data.get("group_name"),
        )


@dataclass
class SurfaceGroup:
    group_id: str
    name: str
    enabled: bool = True
    hidden: bool = False
    parent_id: str | None = None
    child_groups: List[str] = field(default_factory=list)
    surface_ids: List[str] = field(default_factory=list)
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "enabled": self.enabled,
            "hidden": self.hidden,
            "parent_id": self.parent_id,
            "child_groups": list(self.child_groups),
            "surface_ids": list(self.surface_ids),
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, group_id: str, data: Dict[str, Any]) -> "SurfaceGroup":
        return cls(
            group_id=group_id,
            name=str(data.get("name", group_id)),
            enabled=bool(data.get("enabled", True)),
            hidden=bool(data.get("hidden", False)),
            parent_id=data.get("parent_id"),
            child_groups=list(data.get("child_groups", [])),
            surface_ids=list(data.get("surface_ids", [])),
            locked=bool(data.get("locked", False)),
        )

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        if hasattr(self, key):
            value = getattr(self, key)
            if value is None:
                setattr(self, key, default)
                return default
            return value
        setattr(self, key, default)
        return default


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

    points = np.column_stack((xm.ravel(), ym.ravel(), zm.ravel()))

    # Definiere Quad-Zellen (nur horizontale Deckfläche)
    # 🎯 RENDERE NUR STRIKTE ZELLEN: Eine Zelle wird nur gerendert, wenn alle 4 Eckpunkte
    # innerhalb der strikten Surface-Maske liegen. Damit wird der Plot exakt an den
    # Surface-Grenzen beschnitten.
    face_list: List[int] = []
    cell_mask = None
    point_mask = None
    if surface_mask is not None:
        mask_arr = np.asarray(surface_mask, dtype=bool)
        if mask_arr.shape == (ny - 1, nx - 1):
            cell_mask = mask_arr
        elif mask_arr.shape == (ny, nx):
            point_mask = mask_arr
        elif mask_arr.size == (ny - 1) * (nx - 1):
            cell_mask = mask_arr.reshape(ny - 1, nx - 1)
        elif mask_arr.size == ny * nx:
            point_mask = mask_arr.reshape(ny, nx)

    for j in range(ny - 1):
        for i in range(nx - 1):
            if cell_mask is not None:
                if not cell_mask[j, i]:
                    continue
            elif point_mask is not None:
                points_sub = point_mask[j : j + 2, i : i + 2]
                if not np.all(points_sub):
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

    point_scalars = values.ravel()
    mesh["plot_scalars"] = point_scalars

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

    surface_mask = None

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

    plot_mask = _build_plot_surface_mask(plot_x, plot_y, settings)
    if plot_mask is None:
        plot_mask = _convert_point_mask_to_cell_mask(surface_mask)
    if plot_mask is None:
        print("[SurfaceGeometry] Fehler: Keine gültige Surface-Maske – breche Rendering ab.")
        raise RuntimeError("Surface mask missing for plot geometry.")
    _debug_surface_info(settings, plot_x, plot_y, plot_mask, "plot mask")

    requires_resample = not (
        np.array_equal(plot_x, source_x) and np.array_equal(plot_y, source_y)
    )
    was_upscaled = upscale_factor > 1 and requires_resample

    return PlotSurfaceGeometry(
        plot_x=plot_x,
        plot_y=plot_y,
        plot_values=plot_vals,
        z_coords=z_coords,
        surface_mask=plot_mask,
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
    """
    🎯 Extrahiert die strikte Surface-Maske zum Clipping der Plot-Geometrie.
    Die Maske wird unverändert übernommen, damit der Plot exakt an den
    Surface-Grenzen endet.
    """
    if container is None or not hasattr(container, "calculation_spl"):
        return None
    calc_spl = getattr(container, "calculation_spl", None)
    if not isinstance(calc_spl, dict):
        return None
    
    mask_strict = calc_spl.get("surface_mask_strict")
    if mask_strict is None:
        mask_strict = calc_spl.get("surface_mask")
    
    if mask_strict is None:
        return None
    
    try:
        mask_arr = np.asarray(mask_strict, dtype=bool)
        if mask_arr.shape != (len_y, len_x):
            if mask_arr.size == len_y * len_x:
                mask_arr = mask_arr.reshape(len_y, len_x)
            else:
                return None
        # Fülle kleine Lücken, damit die Plot-Maske keine Löcher enthält
        try:
            from scipy import ndimage
            structure = np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=bool,
            )
            mask_arr = ndimage.binary_closing(mask_arr, structure=structure)
        except ImportError:
            mask_arr = _binary_closing_minimal(mask_arr)
        return mask_arr
    except Exception:
        return None


def _build_plot_surface_mask(
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    settings,
) -> Optional[np.ndarray]:
    """
    Baut eine Maske im Plot-Raster basierend auf den aktiven Surfaces.
    """
    if settings is None or plot_x.size == 0 or plot_y.size == 0:
        return None

    surface_definitions = getattr(settings, "surface_definitions", {})
    if not isinstance(surface_definitions, dict):
        return None

    polygons: List[List[Dict[str, float]]] = []
    for surface_def in surface_definitions.values():
        data = surface_def
        if hasattr(surface_def, "to_dict"):
            data = surface_def.to_dict()
        enabled = bool(data.get("enabled", False))
        hidden = bool(data.get("hidden", False))
        points = data.get("points", []) or []
        if not enabled or hidden or len(points) < 3:
            continue
        polygons.append(points)

    if not polygons:
        return None

    if plot_x.size < 2 or plot_y.size < 2:
        return None

    cell_x = 0.5 * (plot_x[:-1] + plot_x[1:])
    cell_y = 0.5 * (plot_y[:-1] + plot_y[1:])
    X, Y = np.meshgrid(cell_x, cell_y, indexing="xy")
    combined_mask = np.zeros_like(X, dtype=bool)
    for polygon in polygons:
        mask = _points_in_polygon_batch_plot(X, Y, polygon)
        if mask is not None:
            combined_mask |= mask

    if not np.any(combined_mask):
        return None
    return combined_mask


def _points_in_polygon_batch_plot(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polygon_points: List[Dict[str, float]],
) -> Optional[np.ndarray]:
    if len(polygon_points) < 3:
        return None

    px = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
    py = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)

    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    inside = np.zeros_like(x_flat, dtype=bool)
    on_edge = np.zeros_like(x_flat, dtype=bool)
    boundary_eps = 1e-6
    n = len(px)

    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]

        y_above_edge = (yi > y_flat) != (yj > y_flat)
        denominator = (yj - yi) + 1e-12
        intersection_x = (xj - xi) * (y_flat - yi) / denominator + xi
        intersects = y_above_edge & (x_flat <= intersection_x + boundary_eps)
        inside ^= intersects

        dx = xj - xi
        dy = yj - yi
        segment_len = math.hypot(dx, dy)
        if segment_len > 0:
            numerator = np.abs(dy * (x_flat - xi) - dx * (y_flat - yi))
            dist = numerator / (segment_len + 1e-12)
            proj = ((x_flat - xi) * dx + (y_flat - yi) * dy) / (
                (segment_len**2) + 1e-12
            )
            on_edge_segment = (
                (dist <= boundary_eps)
                & (proj >= -boundary_eps)
                & (proj <= 1 + boundary_eps)
            )
            on_edge |= on_edge_segment
        j = i

    mask = (inside | on_edge).reshape(x_coords.shape)
    return mask


def _binary_closing_minimal(mask: np.ndarray) -> np.ndarray:
    """
    Füllt kleine Lücken durch Dilatation + Erosion mit einem 3x3-Kernel.
    """
    dilated = _dilate_mask_minimal(mask)
    return _erode_mask_minimal(dilated)


def _dilate_mask_minimal(mask: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    ny, nx = mask.shape
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
    dilated = np.zeros_like(mask, dtype=bool)
    for i in range(ny):
        for j in range(nx):
            region = padded[i : i + 3, j : j + 3]
            dilated[i, j] = np.any(region & kernel)
    return dilated


def _erode_mask_minimal(mask: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    ny, nx = mask.shape
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
    eroded = np.zeros_like(mask, dtype=bool)
    for i in range(ny):
        for j in range(nx):
            region = padded[i : i + 3, j : j + 3]
            eroded[i, j] = np.all(region[kernel])
    return eroded


def _convert_point_mask_to_cell_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    cell_mask = (
        mask[:-1, :-1]
        | mask[1:, :-1]
        | mask[:-1, 1:]
        | mask[1:, 1:]
    )
    return cell_mask


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
            if isinstance(surface_def, SurfaceDefinition):
                hidden = bool(getattr(surface_def, "hidden", False))
                enabled = bool(getattr(surface_def, "enabled", False))
            else:
                hidden = bool(surface_def.get("hidden", False))
                enabled = bool(surface_def.get("enabled", False))
            if hidden:
                continue
            if enabled:
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

