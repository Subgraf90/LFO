from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Module_LFO.Modules_Init.Logging import measure_time, perf_section

DEBUG_SURFACE_GEOMETRY = bool(int(os.environ.get("LFO_DEBUG_SURFACE_GEOMETRY", "1")))
DEBUG_GEOMETRY_TIMING = bool(int(os.environ.get("LFO_DEBUG_TIMING", "0")))


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
class VerticalPlotGeometry:
    """
    Lokale Plot-Geometrie für eine senkrechte Fläche in (u, v)-Koordinaten.
    - Für X-Z-Wände gilt: u = x, v = z, wall_axis='y'
    - Für Y-Z-Wände gilt: u = y, v = z, wall_axis='x'
    """
    surface_id: str
    orientation: str          # "xz" oder "yz"
    wall_axis: str            # "x" oder "y"
    wall_value: float         # Konstante Koordinate der Wand (y≈const bzw. x≈const)
    plot_u: np.ndarray        # 1D-Achse in Wandlängsrichtung
    plot_v: np.ndarray        # 1D-Achse in Wandhöhe (z)
    plot_values: np.ndarray   # 2D-Array (len(v), len(u)) mit SPL-Werten
    mask_uv: np.ndarray       # 2D-Bool-Array gleicher Shape wie plot_values


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
        # Berechne Residuen der linearen Anpassung entlang X
        predicted_x = slope * x_vals + intercept
        residuals_x = z_vals - predicted_x
        max_err_x = float(np.max(np.abs(residuals_x)))
        
        # Wenn Slope nahe 0 trotz Z-Spanne ODER Residuen groß → versuche allgemeine Ebene
        if (abs(slope) < 1e-6 and z_span > tol) or (max_err_x > tol * 10 and z_span > tol):
            # Versuche allgemeine Ebene zu fitten
            plane_model = _fit_planar_surface(x_vals, y_vals, z_vals)
            if plane_model is not None:
                predicted_xy = (
                    plane_model["slope_x"] * x_vals
                    + plane_model["slope_y"] * y_vals
                    + plane_model["intercept"]
                )
                max_err_xy = float(np.max(np.abs(z_vals - predicted_xy)))
                # Wenn allgemeine Ebene deutlich besser passt, verwende sie
                if max_err_xy < max_err_x * 0.5 or max_err_xy <= tol:
                    return plane_model, None
            # Fallback: Verwende mode=x mit intercept
            if abs(slope) < 1e-6:
                print(f"[DEBUG derive_surface_plane] ⚠️  mode=x mit slope≈0 ({slope:.6f}) trotz Z-Spanne={z_span:.3f} m - verwende intercept")
            elif max_err_x > tol * 10:
                print(f"[DEBUG derive_surface_plane] ⚠️  mode=x mit großen Residuen (max_err={max_err_x:.6f} m) trotz Z-Spanne={z_span:.3f} m - verwende intercept")
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
        # Berechne Residuen der linearen Anpassung entlang Y
        predicted_y = slope * y_vals + intercept
        residuals_y = z_vals - predicted_y
        max_err_y = float(np.max(np.abs(residuals_y)))
        
        # Wenn Slope nahe 0 trotz Z-Spanne ODER Residuen groß → versuche allgemeine Ebene
        if (abs(slope) < 1e-6 and z_span > tol) or (max_err_y > tol * 10 and z_span > tol):
            # Versuche allgemeine Ebene zu fitten
            plane_model = _fit_planar_surface(x_vals, y_vals, z_vals)
            if plane_model is not None:
                predicted_xy = (
                    plane_model["slope_x"] * x_vals
                    + plane_model["slope_y"] * y_vals
                    + plane_model["intercept"]
                )
                max_err_xy = float(np.max(np.abs(z_vals - predicted_xy)))
                # Wenn allgemeine Ebene deutlich besser passt, verwende sie
                if max_err_xy < max_err_y * 0.5 or max_err_xy <= tol:
                    return plane_model, None
            # Fallback: Verwende mode=y mit intercept
            if abs(slope) < 1e-6:
                print(f"[DEBUG derive_surface_plane] ⚠️  mode=y mit slope≈0 ({slope:.6f}) trotz Z-Spanne={z_span:.3f} m - verwende intercept")
            elif max_err_y > tol * 10:
                print(f"[DEBUG derive_surface_plane] ⚠️  mode=y mit großen Residuen (max_err={max_err_y:.6f} m) trotz Z-Spanne={z_span:.3f} m - verwende intercept")
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


@measure_time("SurfaceGeometry.build_surface_mesh")
def build_surface_mesh(
    x: np.ndarray,
    y: np.ndarray,
    scalars: np.ndarray,
    *,
    z_coords: Optional[np.ndarray] = None,
    surface_mask: Optional[np.ndarray] = None,
    pv_module: Any = None,
    settings: Any = None,
    container: Any = None,
    source_x: Optional[np.ndarray] = None,
    source_y: Optional[np.ndarray] = None,
    source_scalars: Optional[np.ndarray] = None,
):
    """
    Baut ein PyVista-PolyData ausschließlich aus der Topfläche des Gitters (keine Seitenflächen).
    
    Args:
        x, y: Plot-Grid-Koordinaten (können hochskaliert sein)
        scalars: SPL-Werte für Plot-Grid
        z_coords: Z-Koordinaten für Plot-Grid
        surface_mask: Maske für Plot-Grid
        pv_module: PyVista-Modul
        settings: Settings-Objekt (für PyVista sample-Modus)
        container: Container-Objekt (für Surface-Definitionen)
        source_x, source_y, source_scalars: Ursprüngliches Berechnungs-Grid (für PyVista sample-Modus)
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
    # 🎯 RENDERE ALLE ZELLEN: Alle Zellen werden gerendert, aber SPL-Werte außerhalb
    # der Surface-Maske werden auf NaN gesetzt, damit sie nicht angezeigt werden.
    face_list: List[int] = []
    cell_mask = None
    point_mask = None
    if surface_mask is not None:
        mask_arr = np.asarray(surface_mask, dtype=bool)
        if mask_arr.shape == (ny - 1, nx - 1):
            cell_mask = mask_arr
        elif mask_arr.size == (ny - 1) * (nx - 1):
            cell_mask = mask_arr.reshape(ny - 1, nx - 1)
        elif mask_arr.shape == (ny, nx):
            point_mask = mask_arr
        elif mask_arr.size == ny * nx:
            point_mask = mask_arr.reshape(ny, nx)

    # Rendere nur Zellen, die vollständig innerhalb der Surface liegen
    # 🎯 STRENGE CLIPPING: Alle vier Eckpunkte müssen in der Maske sein
    # Die Maske wurde bereits erweitert
    total_cells = (ny - 1) * (nx - 1)
    rendered_cells = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            if cell_mask is not None:
                if not cell_mask[j, i]:
                    continue
            elif point_mask is not None:
                # Prüfe alle vier Eckpunkte der Zelle
                # Nur rendern, wenn alle vier Eckpunkte in der Maske sind
                if not np.all(point_mask[j:j+2, i:i+2]):
                    continue
            idx0 = j * nx + i
            idx1 = idx0 + 1
            idx2 = idx0 + nx + 1
            idx3 = idx0 + nx
            face_list.extend([4, idx0, idx1, idx2, idx3])
            rendered_cells += 1
    
    if DEBUG_SURFACE_GEOMETRY:
        pass

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
                pass
    except Exception:
        pass

    return mesh


@measure_time("SurfaceGeometry.build_full_floor_mesh")
def build_full_floor_mesh(
    x: np.ndarray,
    y: np.ndarray,
    scalars: np.ndarray,
    *,
    z_coords: Optional[np.ndarray] = None,
    pv_module: Any = None,
) -> Any:
    """
    Erstellt ein vollständiges SPL-Teppich-Mesh OHNE Surface-Maskierung.
    Dieses Mesh wird später an den Surface-Kanten geclippt.
    
    Args:
        x, y: Plot-Grid-Koordinaten
        scalars: SPL-Werte für Plot-Grid
        z_coords: Z-Koordinaten für Plot-Grid
        pv_module: PyVista-Modul
    """
    if pv_module is None:
        try:
            import pyvista as pv_module  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyVista wird benötigt, um ein Floor-Mesh zu erstellen."
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

    # Erzeuge Punktkoordinaten (alle Punkte, keine Maskierung)
    xm, ym = np.meshgrid(x, y, indexing="xy")
    if z is not None and z.shape == (ny, nx):
        zm = z
    elif z is not None and z.size == ny * nx:
        zm = z.reshape(ny, nx)
    else:
        zm = np.zeros_like(xm, dtype=float)

    points = np.column_stack((xm.ravel(), ym.ravel(), zm.ravel()))

    # Definiere ALLE Quad-Zellen (keine Filterung)
    face_list: List[int] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx0 = j * nx + i
            idx1 = idx0 + 1
            idx2 = idx0 + nx + 1
            idx3 = idx0 + nx
            face_list.extend([4, idx0, idx1, idx2, idx3])

    faces = np.asarray(face_list, dtype=np.int64)
    if faces.size > 0:
        mesh = pv_module.PolyData(points, faces)
    else:
        mesh = pv_module.PolyData(points)

    mesh["plot_scalars"] = values.ravel()
    return mesh


@measure_time("SurfaceGeometry.build_surface_clipping_mesh")
def build_surface_clipping_mesh(
    surface_id: str,
    points: List[Dict[str, float]],
    plane_model: Optional[Dict[str, float]],
    *,
    pv_module: Any = None,
    resolution: float = 0.1,
) -> Any:
    """
    Erstellt ein 3D-Mesh für eine Surface, das zum Clipping verwendet wird.
    Das Mesh repräsentiert die Surface-Oberfläche mit korrekter Z-Steigung.
    
    Args:
        surface_id: ID der Surface
        points: Polygon-Punkte der Surface (2D: x, y)
        plane_model: Planar-Modell für Z-Berechnung
        pv_module: PyVista-Modul
        resolution: Auflösung für das Clipping-Mesh (in Metern)
    
    Returns:
        PyVista PolyData-Mesh, das die Surface-Oberfläche repräsentiert
    """
    if pv_module is None:
        try:
            import pyvista as pv_module  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyVista wird benötigt, um ein Clipping-Mesh zu erstellen."
            ) from exc
    
    if len(points) < 3:
        raise ValueError(f"Surface '{surface_id}' benötigt mindestens 3 Punkte.")
    
    # Extrahiere X/Y-Koordinaten des Polygons
    poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
    poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
    
    # Berechne Bounding-Box
    x_min, x_max = float(np.min(poly_x)), float(np.max(poly_x))
    y_min, y_max = float(np.min(poly_y)), float(np.max(poly_y))
    
    # Erstelle feines Grid innerhalb der Bounding-Box
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Anzahl der Punkte basierend auf resolution
    nx = max(2, int(np.ceil(x_range / resolution)) + 1)
    ny = max(2, int(np.ceil(y_range / resolution)) + 1)
    
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    
    # Prüfe für jeden Punkt, ob er im Polygon liegt
    from matplotlib.path import Path
    poly_path = Path(np.column_stack((poly_x, poly_y)))
    points_2d = np.column_stack((X.ravel(), Y.ravel()))
    inside_mask = poly_path.contains_points(points_2d)
    
    # Berechne Z-Werte für Punkte innerhalb des Polygons
    if plane_model is None:
        # Fallback: Z = 0
        Z = np.zeros_like(X)
    else:
        # Verwende evaluate_surface_plane für jeden Punkt
        Z = np.zeros_like(X)
        for j in range(ny):
            for i in range(nx):
                if inside_mask[j * nx + i]:
                    Z[j, i] = evaluate_surface_plane(plane_model, X[j, i], Y[j, i])
    
    # Erstelle Punkte-Array (nur für Punkte innerhalb des Polygons)
    points_3d = []
    for j in range(ny):
        for i in range(nx):
            if inside_mask[j * nx + i]:
                points_3d.append([X[j, i], Y[j, i], Z[j, i]])
    
    if len(points_3d) == 0:
        # Keine Punkte im Polygon → erstelle minimales Mesh
        # Verwende die Polygon-Punkte selbst mit Z-Werten
        points_3d = []
        for p in points:
            x_val = p.get("x", 0.0)
            y_val = p.get("y", 0.0)
            if plane_model is None:
                z_val = 0.0
            else:
                z_val = evaluate_surface_plane(plane_model, x_val, y_val)
            points_3d.append([x_val, y_val, z_val])
    
    points_array = np.array(points_3d, dtype=float)
    
    # Erstelle Mesh aus Punkten (Delaunay-Triangulation)
    mesh = pv_module.PolyData(points_array)
    if len(points_array) >= 3:
        # 2D-Delaunay-Triangulation (projiziert auf XY-Ebene, behält Z)
        mesh = mesh.delaunay_2d(alpha=0.0, tol=0.0)
    
    return mesh


@measure_time("SurfaceGeometry.clip_floor_with_surfaces")
def clip_floor_with_surfaces(
    floor_mesh: Any,
    surface_definitions: Dict[str, Any],
    *,
    pv_module: Any = None,
) -> Any:
    """
    Clippt den Floor-Mesh an allen enabled Surfaces.
    Für jede Surface wird der Floor-Mesh an der Surface-Geometrie weggeschnitten.
    
    Args:
        floor_mesh: Vollständiges Floor-Mesh (PyVista PolyData)
        surface_definitions: Dict mit Surface-Definitionen
        pv_module: PyVista-Modul
    
    Returns:
        Geclipptes Floor-Mesh
    """
    if pv_module is None:
        try:
            import pyvista as pv_module  # type: ignore
        except Exception:  # noqa: BLE001
            raise ImportError("PyVista wird benötigt für Clipping.")
    
    clipped_mesh = floor_mesh.copy(deep=True)
    
    # Sammle alle enabled Surfaces
    enabled_surfaces = []
    for surface_id, surface_def in surface_definitions.items():
        if isinstance(surface_def, SurfaceDefinition):
            enabled = bool(getattr(surface_def, "enabled", False))
            hidden = bool(getattr(surface_def, "hidden", False))
            points = getattr(surface_def, "points", []) or []
            plane_model = getattr(surface_def, "plane_model", None)
        else:
            enabled = bool(surface_def.get("enabled", False))
            hidden = bool(surface_def.get("hidden", False))
            points = surface_def.get("points", [])
            plane_model = surface_def.get("plane_model")
        
        if not enabled or hidden or len(points) < 3:
            continue
        
        enabled_surfaces.append((surface_id, points, plane_model))
    
    # Wenn keine Surfaces vorhanden, gebe unverändertes Mesh zurück
    if not enabled_surfaces:
        return clipped_mesh
    
    # Versuche clip_surface pro Surface anzuwenden
    # Falls das fehlschlägt, verwende manuelles Clipping als Fallback
    use_manual_clipping = False
    
    # Iteriere über alle Surfaces und wende clip_surface an
    for surface_id, points, plane_model in enabled_surfaces:
        try:
            # Erstelle Clipping-Mesh für diese Surface
            # Höhere Auflösung für glattere Clipping-Ränder
            clipping_mesh = build_surface_clipping_mesh(
                surface_id,
                points,
                plane_model,
                pv_module=pv_module,
                resolution=0.05,  # 5cm Auflösung für präziseres Clipping
            )
            
            # Prüfe ob Clipping-Mesh gültig ist
            if clipping_mesh.n_points == 0 or clipping_mesh.n_cells == 0:
                if DEBUG_SURFACE_GEOMETRY:
                    pass

                continue
            
            # Prüfe ob Floor-Mesh noch gültig ist
            if clipped_mesh.n_cells == 0:
                if DEBUG_SURFACE_GEOMETRY:
                    pass

                break
            
            # Versuche clip_surface anzuwenden
            try:
                # Berechne durchschnittliche Normale der Surface für korrekte Orientierung
                clipping_mesh_with_normals = clipping_mesh.compute_normals(point_normals=True, cell_normals=False)
                use_invert = False
                if clipping_mesh_with_normals.n_points > 0:
                    normals = clipping_mesh_with_normals.point_data.get("Normals")
                    if normals is not None and len(normals) > 0:
                        avg_normal = np.mean(normals, axis=0)
                        # clip_surface mit invert=False entfernt Zellen auf der Seite entgegen der Normalen
                        # clip_surface mit invert=True entfernt Zellen auf der Seite in Richtung der Normalen
                        # Wir wollen Zellen oberhalb der Surface entfernen (Z > Surface-Z)
                        # Wenn Normale nach oben zeigt (Z > 0), müssen wir invert=True verwenden
                        # Wenn Normale nach unten zeigt (Z < 0), müssen wir die Surface flippen und invert=True verwenden
                        if avg_normal[2] > 0:
                            # Normale zeigt nach oben: verwende invert=True um Zellen oberhalb zu entfernen
                            use_invert = True
                        elif avg_normal[2] < 0:
                            # Normale zeigt nach unten: invertiere Surface, dann verwende invert=True
                            clipping_mesh = clipping_mesh.flip_normals()
                            use_invert = True
                        # Wenn avg_normal[2] == 0, ist die Surface horizontal - verwende Standard-Logik
                
                # Clippe: Entferne Zellen, die die Surface schneiden oder darüber liegen
                cells_before = clipped_mesh.n_cells
                clipped_mesh = clipped_mesh.clip_surface(clipping_mesh, invert=use_invert)
                cells_after = clipped_mesh.n_cells
                
                # Validierung: Prüfe ob Clipping sinnvoll war
                if cells_after >= cells_before:
                    # Keine Zellen entfernt - möglicherweise Problem mit Orientierung
                    if DEBUG_SURFACE_GEOMETRY:
                        pass
                    # Versuche mit invertiertem Parameter
                    clipped_mesh = clipped_mesh.clip_surface(clipping_mesh, invert=not use_invert)
                    cells_after = clipped_mesh.n_cells
                
                if DEBUG_SURFACE_GEOMETRY:
                    pass
                
            except Exception as clip_exc:
                # clip_surface fehlgeschlagen - markiere für manuelles Clipping
                if DEBUG_SURFACE_GEOMETRY:
                    pass

                use_manual_clipping = True
                break
        
        except Exception as exc:  # noqa: BLE001
            if DEBUG_SURFACE_GEOMETRY:
                pass

            use_manual_clipping = True
            break
    
    # Falls clip_surface fehlgeschlagen ist, verwende manuelles Clipping
    if use_manual_clipping:
        # Manuelles Clipping für alle Surfaces in einem Durchgang
        # Starte mit frischem Mesh
        clipped_mesh = floor_mesh.copy(deep=True)
        
        if DEBUG_SURFACE_GEOMETRY:
            pass
        
        from matplotlib.path import Path
        
        # Berechne Zell-Zentren einmal
        cell_centers = clipped_mesh.cell_centers().points
        if len(cell_centers) == 0:
            return clipped_mesh
        
        # Erstelle Polygon-Pfade und Z-Schwellenwerte für alle Surfaces
        surface_polygons = []
        for surface_id, points, plane_model in enabled_surfaces:
            try:
                # Erstelle Clipping-Mesh für diese Surface
                # Höhere Auflösung für glattere Clipping-Ränder
                clipping_mesh = build_surface_clipping_mesh(
                    surface_id,
                    points,
                    plane_model,
                    pv_module=pv_module,
                    resolution=0.05,  # 5cm Auflösung für präziseres Clipping
                )
                
                surface_points = clipping_mesh.points
                if len(surface_points) > 0:
                    poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
                    poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
                    z_surface_max = np.max(surface_points[:, 2])
                    poly_path = Path(np.column_stack((poly_x, poly_y)))
                    surface_polygons.append((poly_path, z_surface_max + 0.01, surface_id))
            except Exception as exc:  # noqa: BLE001
                if DEBUG_SURFACE_GEOMETRY:
                    pass

                continue
        
        # Prüfe jede Zelle gegen alle Surfaces
        cells_to_keep = []
        for i in range(clipped_mesh.n_cells):
            cell_center = cell_centers[i]
            keep_cell = True
            
            # Prüfe gegen alle Surfaces
            for poly_path, z_threshold, surface_id in surface_polygons:
                if poly_path.contains_point((cell_center[0], cell_center[1])):
                    # Zelle liegt im Polygon: Entferne wenn über Surface
                    if cell_center[2] > z_threshold:
                        keep_cell = False
                        break
            
            if keep_cell:
                cells_to_keep.append(i)
        
        if len(cells_to_keep) < clipped_mesh.n_cells:
            clipped_mesh = clipped_mesh.extract_cells(cells_to_keep)
            if DEBUG_SURFACE_GEOMETRY:
                total_cells_before = floor_mesh.n_cells
    
    return clipped_mesh


@measure_time("SurfaceGeometry.build_vertical_surface_mesh")
def build_vertical_surface_mesh(
    geom: VerticalPlotGeometry,
    *,
    pv_module: Any = None,
):
    """
    Baut ein strukturiertes PyVista-PolyData für eine senkrechte Fläche auf
    Basis der lokalen (u, v)-Plot-Geometrie.

    - Die Eingabe ist ein VerticalPlotGeometry-Objekt mit:
        plot_u, plot_v, plot_values, mask_uv, orientation, wall_axis, wall_value.
    - Es wird ein reguläres Grid erzeugt und mit einer Zellmaske (aus mask_uv)
      streng auf den Wandbereich begrenzt – analog zu build_surface_mesh.
    """
    if pv_module is None:
        try:
            import pyvista as pv_module  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyVista wird benötigt, um ein vertikales Surface-Mesh zu erstellen."
            ) from exc

    plot_u = np.asarray(geom.plot_u, dtype=float)
    plot_v = np.asarray(geom.plot_v, dtype=float)
    values = np.asarray(geom.plot_values, dtype=float)
    mask_uv = np.asarray(geom.mask_uv, dtype=bool)

    ny, nx = values.shape
    if ny != plot_v.size or nx != plot_u.size:
        raise ValueError("plot_values müssen Shape (len(v), len(u)) besitzen.")

    # Erzeuge lokales (u, v)-Grid
    U, V = np.meshgrid(plot_u, plot_v, indexing="xy")

    # Mapping in Weltkoordinaten:
    #   X-Z-Wand: u=x, v=z, y=wall_value
    #   Y-Z-Wand: u=y, v=z, x=wall_value
    if geom.orientation == "xz":
        X = U
        Y = np.full_like(U, geom.wall_value, dtype=float)
        Z = V
    else:  # "yz"
        X = np.full_like(U, geom.wall_value, dtype=float)
        Y = U
        Z = V

    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Maskenlogik wie bei build_surface_mesh:
    # - mask_uv kann entweder Punktmaske (ny, nx) oder Zellmaske (ny-1, nx-1) sein.
    face_list: List[int] = []
    cell_mask = None
    point_mask = None

    if mask_uv.shape == (ny - 1, nx - 1):
        cell_mask = mask_uv
    elif mask_uv.shape == (ny, nx):
        point_mask = mask_uv

    total_cells = (ny - 1) * (nx - 1)
    rendered_cells = 0

    for j in range(ny - 1):
        for i in range(nx - 1):
            if cell_mask is not None:
                if not cell_mask[j, i]:
                    continue
            elif point_mask is not None:
                # Nur Zellen rendern, deren vier Eckpunkte im Polygon liegen
                if not np.all(point_mask[j:j+2, i:i+2]):
                    continue

            idx0 = j * nx + i
            idx1 = idx0 + 1
            idx2 = idx0 + nx + 1
            idx3 = idx0 + nx
            face_list.extend([4, idx0, idx1, idx2, idx3])
            rendered_cells += 1

    if DEBUG_SURFACE_GEOMETRY:
        pass

    faces = np.asarray(face_list, dtype=np.int64)
    if faces.size > 0:
        mesh = pv_module.PolyData(points, faces)
    else:
        mesh = pv_module.PolyData(points)

    mesh["plot_scalars"] = values.ravel()
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
    t_start_total = time.perf_counter() if DEBUG_GEOMETRY_TIMING else 0.0

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
    # Begrenze Upscaling für Performance: zu hohe Werte erzeugen Millionen Plot-Punkte
    upscale_factor = max(1, upscale_factor)
    if upscale_factor > 6:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        upscale_factor = 6

    plot_x = source_x.copy()
    plot_y = source_y.copy()
    plot_vals = values.copy()
    if DEBUG_SURFACE_GEOMETRY:
        total_points = int(plot_x.size * plot_y.size)

    if DEBUG_GEOMETRY_TIMING:
        t_after_setup = time.perf_counter()

    z_coords = _extract_plot_z_coordinates(container, len(source_y), len(source_x))
    surface_mask = _extract_surface_mask(container, len(source_y), len(source_x))

    # Lokale Plot-Maske initialisieren; wird ggf. im Upscaling-Block gesetzt
    plot_mask = None

    if (
        upscale_factor > 1
        and plot_x.size > 1
        and plot_y.size > 1
    ):
        orig_plot_x = plot_x.copy()
        orig_plot_y = plot_y.copy()
        expanded_x = _expand_axis_for_plot(plot_x, upscale_factor)
        expanded_y = _expand_axis_for_plot(plot_y, upscale_factor)

        # 📌 Upscaling der SPL-Werte:
        # Immer Nearest-Neighbour (exakt zu Berechnungszellen, keine Interpolation)
        plot_vals = _resample_values_to_grid_nearest(
            plot_vals, orig_plot_x, orig_plot_y, expanded_x, expanded_y
        )

        # Erstelle zwei Masken:
        # 1. Erweiterte Maske (optional, aktuell nicht für Z verwendet)
        # 2. Nicht-erweiterte Maske für strenges Clipping UND Z-Berechnung,
        #    damit die Z-Werte exakt an den Surface-Rändern enden.
        temp_plot_mask_dilated = _build_plot_surface_mask(expanded_x, expanded_y, settings, dilate=True)
        temp_plot_mask_strict = _build_plot_surface_mask(expanded_x, expanded_y, settings, dilate=False)
        
        if z_coords is not None:
            # 🎯 Z-Werte ebenfalls auf das hochskalierte Raster bringen, damit
            # die Geometrie konsistent zur SPL-Darstellung bleibt.
            orig_z_coords = z_coords.copy()
            z_coords = _resample_values_to_grid_nearest(
                orig_z_coords, orig_plot_x, orig_plot_y, expanded_x, expanded_y
            )
        if surface_mask is not None:
            surface_mask = _resample_mask_to_grid(surface_mask, orig_plot_x, orig_plot_y, expanded_x, expanded_y)

        plot_x = expanded_x
        plot_y = expanded_y
        # Verwende nicht-erweiterte Maske für strenges Clipping
        if 'temp_plot_mask_strict' in locals() and temp_plot_mask_strict is not None:
            plot_mask = temp_plot_mask_strict
        else:
            plot_mask = _build_plot_surface_mask(plot_x, plot_y, settings, dilate=False)
    
    if DEBUG_GEOMETRY_TIMING:
        t_after_upscale = time.perf_counter()

    if plot_mask is None:
        plot_mask = _build_plot_surface_mask(plot_x, plot_y, settings, dilate=False)
    if plot_mask is None and surface_mask is not None:
        plot_mask = _convert_point_mask_to_cell_mask(surface_mask)
    if plot_mask is None:
        raise RuntimeError("Surface mask missing for plot geometry.")
    _debug_surface_info(settings, plot_x, plot_y, plot_mask, "plot mask")
    if DEBUG_SURFACE_GEOMETRY and z_coords is not None:
        nonzero_z = int(np.count_nonzero(z_coords))

    requires_resample = not (
        np.array_equal(plot_x, source_x) and np.array_equal(plot_y, source_y)
    )
    was_upscaled = upscale_factor > 1 and requires_resample

    if DEBUG_GEOMETRY_TIMING:
        t_end = time.perf_counter()
        print(
            "[SurfaceGeometryCalculator] prepare_plot_geometry timings:\n"
            f"  setup/validation : {(t_after_setup - t_start_total) * 1000.0:7.2f} ms\n"
            f"  upscaling/masks  : {(t_after_upscale - t_after_setup) * 1000.0:7.2f} ms\n"
            f"  final bookkeeping: {(t_end - t_after_upscale) * 1000.0:7.2f} ms\n"
            f"  TOTAL            : {(t_end - t_start_total) * 1000.0:7.2f} ms"
        )

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


@measure_time("SurfaceGeometry.prepare_vertical_plot_geometry")
def prepare_vertical_plot_geometry(
    surface_id: str,
    settings,
    container,
    *,
    default_upscale: int = 3,
) -> Optional[VerticalPlotGeometry]:
    """
    Bereitet eine lokale Plot-Geometrie für eine senkrechte Fläche vor.

    Schritte:
    - Bestimme Orientierung (X-Z- oder Y-Z-Wand) aus dem 3D-Polygon.
    - Leite lokale Koordinaten (u, v) ab:
        X-Z-Wand: u = x, v = z, wall_axis='y'
        Y-Z-Wand: u = y, v = z, wall_axis='x'
    - Erzeuge ein regelmäßiges (u, v)-Grid mit plot_upscale_factor.
    - Interpoliere SPL-Werte aus den vertikalen Samples (`surface_samples`/`surface_fields`)
      auf dieses Grid (IDW-Interpolation).
    - Baue eine Maskenmatrix mask_uv im (u, v)-Grid (Punkte im Polygon).
    """
    if settings is None or container is None:
        return None

    t_start_total = time.perf_counter() if DEBUG_GEOMETRY_TIMING else 0.0

    surface_definitions = getattr(settings, "surface_definitions", {})
    if not isinstance(surface_definitions, dict):
        return None
    surface = surface_definitions.get(surface_id)
    if surface is None:
        return None

    # Normalisiere Surface-Daten (dict)
    if hasattr(surface, "to_dict"):
        surface_data = surface.to_dict()
    else:
        surface_data = surface
    points = surface_data.get("points", []) or []
    if len(points) < 3:
        return None

    if DEBUG_GEOMETRY_TIMING:
        t_after_surface = time.perf_counter()

    # Extrahiere Polygonkoordinaten
    xs = np.array([float(p.get("x", 0.0)) for p in points], dtype=float)
    ys = np.array([float(p.get("y", 0.0)) for p in points], dtype=float)
    zs = np.array([float(p.get("z", 0.0)) for p in points], dtype=float)
    x_span = float(np.ptp(xs))
    y_span = float(np.ptp(ys))

    # Bestimme Orientierung
    eps_line = 1e-6
    if y_span < eps_line and x_span >= eps_line:
        # X-Z-Wand: y ≈ const
        orientation = "xz"
        wall_axis = "y"
        wall_value = float(np.mean(ys))
        poly_u = xs
        poly_v = zs
    elif x_span < eps_line and y_span >= eps_line:
        # Y-Z-Wand: x ≈ const
        orientation = "yz"
        wall_axis = "x"
        wall_value = float(np.mean(xs))
        poly_u = ys
        poly_v = zs
    else:
        # Keine eindeutig senkrechte Fläche
        return None

    # Vertikale Samples + Felder aus calculation_spl holen
    calc_spl = getattr(container, "calculation_spl", None)
    if not isinstance(calc_spl, dict):
        return None
    sample_payloads = calc_spl.get("surface_samples")
    surface_fields = calc_spl.get("surface_fields")
    if not isinstance(sample_payloads, list) or not isinstance(surface_fields, dict):
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None

    payload = None
    for entry in sample_payloads:
        if not isinstance(entry, dict):
            continue
        if entry.get("surface_id") != surface_id:
            continue
        if entry.get("kind", "planar") != "vertical":
            continue
        payload = entry
        break
    if payload is None:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None

    if DEBUG_GEOMETRY_TIMING:
        t_after_samples = time.perf_counter()

    coords = np.asarray(payload.get("coordinates", []), dtype=float)
    if coords.size == 0:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None
    coords = coords.reshape(-1, 3)
    field_values = surface_fields.get(surface_id)
    if field_values is None:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None
    field_arr = np.asarray(field_values, dtype=complex).reshape(-1)
    if field_arr.size != coords.shape[0]:
        if DEBUG_SURFACE_GEOMETRY:
            pass

        return None

    # Lokale Sample-Koordinaten (u_s, v_s)
    # Lokale Sample-Koordinaten (u_s, v_s)
    if orientation == "xz":
        u_samples = coords[:, 0]  # x
        v_samples = coords[:, 2]  # z
    else:  # "yz"
        u_samples = coords[:, 1]  # y
        v_samples = coords[:, 2]  # z

    if u_samples.size == 0 or v_samples.size == 0:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None

    # Rekonstruiere das lokale (u, v)-Raster aus den Sample-Punkten:
    # Die Vertikal-Samples werden in SurfaceGridCalculator auf einem regulären
    # Gitter erzeugt. Wir nehmen die eindeutigen u- und v-Werte als Achsen.
    u_axis = np.unique(u_samples)
    v_axis = np.unique(v_samples)
    if u_axis.size < 2 or v_axis.size < 2:
        if DEBUG_SURFACE_GEOMETRY:
            pass
        return None

    if DEBUG_GEOMETRY_TIMING:
        t_after_axes = time.perf_counter()

    # 2D-Gitter der komplexen Feldwerte aufbauen:
    # Wir initialisieren mit 0 und befüllen jede (v,u)-Zelle, für die ein Sample
    # existiert. Mehrfache Treffer überschreiben sich mit identischen Werten.
    values_complex = np.zeros((v_axis.size, u_axis.size), dtype=complex)
    for coord, val in zip(coords, field_arr):
        if orientation == "xz":
            u_val = coord[0]
            v_val = coord[2]
        else:
            u_val = coord[1]
            v_val = coord[2]
        iu = int(np.searchsorted(u_axis, u_val))
        iv = int(np.searchsorted(v_axis, v_val))
        iu = max(0, min(iu, u_axis.size - 1))
        iv = max(0, min(iv, v_axis.size - 1))
        values_complex[iv, iu] = val

    plot_u = u_axis
    plot_v = v_axis
    U, V = np.meshgrid(plot_u, plot_v, indexing="xy")

    # Betrags-SPL in dB
    pressure_mag = np.abs(values_complex)
    pressure_mag = np.clip(pressure_mag, 1e-12, None)
    spl_db = 20.0 * np.log10(pressure_mag)

    if DEBUG_GEOMETRY_TIMING:
        t_after_field = time.perf_counter()

    # Maskenmatrix im (u, v)-Grid analog _build_plot_surface_mask (Punkt-Maske)
    # WICHTIG: Diese Maske ist "strict" entlang der Polygonlinie; das
    # eigentliche "calculate beyond surface" passiert bereits bei der
    # Sample-Erzeugung (dilatierte Maske in SurfaceGridCalculator).
    mask_uv = _points_in_polygon_batch_uv(U, V, poly_u, poly_v)
    if mask_uv is None:
        # Fallback: alles sichtbar
        mask_uv = np.ones_like(U, dtype=bool)

    if DEBUG_GEOMETRY_TIMING:
        t_after_mask = time.perf_counter()

    # ------------------------------------------------------------
    # Optionales Upscaling NUR für die Darstellung:
    # - Achsen werden verfeinert
    # - SPL-Werte und Maske werden per Nearest-Neighbour auf das
    #   feinere Raster kopiert (Block-Upscaling, keine Interpolation).
    # ------------------------------------------------------------
    requested_upscale = getattr(settings, "plot_upscale_factor", None)
    if requested_upscale is None:
        requested_upscale = default_upscale
    try:
        upscale_factor = int(requested_upscale)
    except (TypeError, ValueError):
        upscale_factor = default_upscale
    upscale_factor = max(1, min(6, upscale_factor))

    if upscale_factor > 1 and plot_u.size > 1 and plot_v.size > 1:
        orig_u = plot_u.copy()
        orig_v = plot_v.copy()
        expanded_u = _expand_axis_for_plot(plot_u, upscale_factor)
        expanded_v = _expand_axis_for_plot(plot_v, upscale_factor)

        # SPL-Werte: Nearest-Neighbour / Block-Upscaling
        spl_db = _resample_values_to_grid_nearest(
            spl_db, orig_u, orig_v, expanded_u, expanded_v
        )

        # Maske NICHT per Nearest-Neighbour hochskalieren (das würde
        # die Polygonkante aufweichen), sondern auf dem hochskalierten
        # Raster neu aus dem Polygon berechnen – exakt wie bei den
        # XY-Flächen (_build_plot_surface_mask).
        plot_u = expanded_u
        plot_v = expanded_v
        U, V = np.meshgrid(plot_u, plot_v, indexing="xy")
        mask_uv = _points_in_polygon_batch_uv(U, V, poly_u, poly_v)
        if mask_uv is None:
            mask_uv = np.ones_like(U, dtype=bool)
    else:
        # Kein Upscaling: Achsen/Maske bleiben wie sie sind
        plot_u = u_axis
        plot_v = v_axis
    
    if DEBUG_GEOMETRY_TIMING:
        t_end = time.perf_counter()
        print(
            f"[SurfaceGeometryCalculator] prepare_vertical_plot_geometry(surface_id={surface_id}) timings:\n"
            f"  surface_lookup   : {(t_after_surface - t_start_total) * 1000.0:7.2f} ms\n"
            f"  samples/fields   : {(t_after_samples - t_after_surface) * 1000.0:7.2f} ms\n"
            f"  axes/grid        : {(t_after_axes - t_after_samples) * 1000.0:7.2f} ms\n"
            f"  field->SPL/mask  : {(t_after_mask - t_after_field) * 1000.0:7.2f} ms\n"
            f"  upscaling        : {(t_end - t_after_mask) * 1000.0:7.2f} ms\n"
            f"  TOTAL            : {(t_end - t_start_total) * 1000.0:7.2f} ms"
        )

    return VerticalPlotGeometry(
        surface_id=surface_id,
        orientation=orientation,
        wall_axis=wall_axis,
        wall_value=wall_value,
        plot_u=plot_u,
        plot_v=plot_v,
        plot_values=spl_db,
        mask_uv=mask_uv,
    )


def _recompute_z_coordinates_in_plot_grid(
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    settings,
    container,
    plot_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Berechnet Z-Koordinaten direkt im Plot-Raster basierend auf Surface-Definitionen.
    Dies ist genauer als Resampling, da Z-Werte exakt aus Planmodellen berechnet werden.
    """
    if settings is None or plot_x.size == 0 or plot_y.size == 0:
        return None
    
    surface_definitions = getattr(settings, "surface_definitions", {})
    if not isinstance(surface_definitions, dict):
        return None
    
    # Sammle enabled Surfaces mit Planmodellen
    surfaces_with_models = []
    for surface_id, surface_def in surface_definitions.items():
        if hasattr(surface_def, "to_dict"):
            data = surface_def.to_dict()
        else:
            data = surface_def
        
        enabled = bool(data.get("enabled", False))
        hidden = bool(data.get("hidden", False))
        points = data.get("points", []) or []
        
        if not enabled or hidden or len(points) < 3:
            continue
        
        # Berechne Planmodell
        model, error = derive_surface_plane(points)
        if model is None:
            continue
        
        surfaces_with_models.append((points, model))
    
    if not surfaces_with_models:
        return None

    # Schneller Pfad: Alle aktiven Surfaces sind plan und liegen bei z≈0
    # → kein Bedarf für teure Punkt-in-Polygon-Schleifen, Z-Gitter bleibt überall 0.
    try:
        all_constant = all(
            str(model.get("mode", "constant")) == "constant"
            for _, model in surfaces_with_models
        )
        if all_constant:
            bases = np.array(
                [float(model.get("base", 0.0)) for _, model in surfaces_with_models],
                dtype=float,
            )
            if bases.size and np.all(np.abs(bases) <= 1e-6):
                if DEBUG_SURFACE_GEOMETRY:
                    pass
                return np.zeros((plot_y.size, plot_x.size), dtype=float)
    except Exception:
        # Falls etwas schiefgeht, normal weiterrechnen
        pass
    
    # Erstelle Plot-Grid
    X, Y = np.meshgrid(plot_x, plot_y, indexing="xy")
    ny, nx = X.shape
    Z_grid = np.zeros((ny, nx), dtype=float)
    
    # Erstelle Maske für Z-Berechnung (Punkt-Maske)
    if plot_mask is not None:
        # Konvertiere Cell-Maske zu Punkt-Maske falls nötig
        if plot_mask.shape == (ny - 1, nx - 1):
            # Cell-Maske: erweitere auf Punkt-Maske
            point_mask = np.zeros((ny, nx), dtype=bool)
            point_mask[:-1, :-1] |= plot_mask
            point_mask[1:, :-1] |= plot_mask
            point_mask[:-1, 1:] |= plot_mask
            point_mask[1:, 1:] |= plot_mask
        else:
            point_mask = np.asarray(plot_mask, dtype=bool)
    else:
        point_mask = np.ones((ny, nx), dtype=bool)  # Alle Punkte
    
    # ------------------------------------------------------------
    # Erste Runde (vektorisiert):
    # Für jede Surface:
    #   - Maske im Plot-Grid bestimmen
    #   - Z-Ebene auf gesamtem Grid auswerten
    #   - Beiträge mitteln, falls mehrere Surfaces überlappen
    # ------------------------------------------------------------
    Z_sum = np.zeros_like(Z_grid, dtype=float)
    Z_count = np.zeros_like(Z_grid, dtype=float)

    for points, model in surfaces_with_models:
        surface_mask = _points_in_polygon_batch_plot(X, Y, points)
        if surface_mask is None:
            continue
        # Nur Punkte berücksichtigen, die auch in der globalen Plot-Maske liegen
        effective_mask = surface_mask & point_mask
        if not np.any(effective_mask):
            continue

        Z_surface = _evaluate_plane_on_grid(model, X, Y)
        Z_sum[effective_mask] += Z_surface[effective_mask]
        Z_count[effective_mask] += 1.0

    inside_mask = (Z_count > 0.0)
    Z_grid[inside_mask] = Z_sum[inside_mask] / Z_count[inside_mask]

    # ------------------------------------------------------------
    # Zweite Runde: Fülle Z-Werte iterativ
    # (Punkte in point_mask, die noch Z==0 haben, aber an befüllte
    # Nachbarn angrenzen). Das glättet Kanten ohne die Fläche zu
    # verlassen.
    # ------------------------------------------------------------
    points_with_z_before = int(np.sum(Z_grid != 0.0))

    # Kantenmaske: Punkte in point_mask, die einen Nachbarn außerhalb der Maske haben
    # → diese markieren die eigentliche Surface-Grenze und sollen NICHT durch
    #    die Füll-Interpolation verändert werden, damit die Geometrie exakt bleibt.
    padded_pm = np.pad(point_mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    edge_mask = np.zeros_like(point_mask, dtype=bool)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ys = 1 + dj
            ye = ys + ny
            xs = 1 + di
            xe = xs + nx
            neighbor_pm = padded_pm[ys:ye, xs:xe]
            edge_mask |= point_mask & (~neighbor_pm)
    max_iterations = 5
    iteration = 0
    for iteration in range(max_iterations):
        # Punkte mit gültigem Z innerhalb der Maske
        known = (Z_grid != 0.0) & point_mask
        # Unbekannte Z nur für innere Punkte (nicht am äußeren Rand)
        unknown = (Z_grid == 0.0) & point_mask & (~edge_mask)
        if not np.any(unknown):
            break

        # Nachbar-Summen und -Anzahl vektorbasiert berechnen
        padded_Z = np.pad(Z_grid, ((1, 1), (1, 1)), mode="edge")
        padded_known = np.pad(known, ((1, 1), (1, 1)), mode="constant", constant_values=False)

        sum_neighbors = np.zeros_like(Z_grid, dtype=float)
        count_neighbors = np.zeros_like(Z_grid, dtype=float)

        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ys = 1 + dj
                ye = ys + ny
                xs = 1 + di
                xe = xs + nx
                neighbor_Z = padded_Z[ys:ye, xs:xe]
                neighbor_known = padded_known[ys:ye, xs:xe]
                sum_neighbors += neighbor_Z * neighbor_known
                count_neighbors += neighbor_known.astype(float)

        fill_mask = unknown & (count_neighbors > 0)
        if not np.any(fill_mask):
            break

        Z_grid[fill_mask] = sum_neighbors[fill_mask] / count_neighbors[fill_mask]
    
    points_with_z_after = int(np.sum(Z_grid != 0.0))
    if DEBUG_SURFACE_GEOMETRY:
        mask_points = int(np.sum(point_mask))

    return Z_grid


def _point_in_polygon_simple(x: float, y: float, polygon_points: List[Dict[str, float]]) -> bool:
    """
    Punkt-in-Polygon-Prüfung mit Kanten-Toleranz.
    Behandelt Punkte auf der Kante oder sehr nahe an der Kante als "inside".
    Konsistent mit _points_in_polygon_batch_plot.
    """
    if len(polygon_points) < 3:
        return False
    
    px = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
    py = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)
    
    n = len(px)
    inside = False
    boundary_eps = 1e-6
    j = n - 1
    
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        
        # Ray-Casting: Prüfe ob Strahl von (x,y) nach rechts die Kante schneidet
        if ((yi > y) != (yj > y)) and (x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi + boundary_eps):
            inside = not inside
        
        # Prüfe ob Punkt direkt auf der Kante liegt (wie in _points_in_polygon_batch_plot)
        dx = xj - xi
        dy = yj - yi
        segment_len = math.hypot(dx, dy)
        if segment_len > 0:
            dist = abs(dy * (x - xi) - dx * (y - yi)) / segment_len
            if dist <= boundary_eps:
                proj = ((x - xi) * dx + (y - yi) * dy) / (segment_len * segment_len)
                if -boundary_eps <= proj <= 1 + boundary_eps:
                    return True  # Punkt liegt auf der Kante -> inside
        
        j = i
    
    return inside


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
    dilate: bool = True,
) -> Optional[np.ndarray]:
    """
    Baut eine Maske im Plot-Raster basierend auf den aktiven Surfaces.
    
    Args:
        dilate: Wenn True, wird die Maske durch Dilatation erweitert (für Z-Berechnung).
                Wenn False, bleibt die Maske unverändert (für strenges Clipping).
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

    # 🎯 Erzeuge Zellmaske (ny-1, nx-1) basierend auf Pixelzentren
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

    # 🎯 Wandle Zellmaske in Punktmaske (ny, nx) um
    ny, nx = combined_mask.shape
    point_mask = np.zeros((ny + 1, nx + 1), dtype=bool)
    point_mask[:-1, :-1] |= combined_mask
    point_mask[1:, :-1] |= combined_mask
    point_mask[:-1, 1:] |= combined_mask
    point_mask[1:, 1:] |= combined_mask

    # 🎯 Erweiterung um 1 Pixel in alle Richtungen für glatte Ränder (nur wenn gewünscht)
    if dilate:
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
            point_mask = ndimage.binary_dilation(point_mask, structure=structure)
        except Exception:
            point_mask = _dilate_mask_minimal(point_mask)
    return point_mask


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


def _points_in_polygon_batch_uv(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    poly_u: np.ndarray,
    poly_v: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Vektorisierter Punkt-in-Polygon-Test im (u, v)-Raum.
    Aufbau analog zu _points_in_polygon_batch_plot, aber explizit für
    senkrechte Flächen in lokalen Koordinaten.
    """
    u_coords = np.asarray(u_coords, dtype=float)
    v_coords = np.asarray(v_coords, dtype=float)
    poly_u = np.asarray(poly_u, dtype=float).reshape(-1)
    poly_v = np.asarray(poly_v, dtype=float).reshape(-1)

    if poly_u.size < 3 or poly_v.size < 3:
        return None

    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    inside = np.zeros_like(u_flat, dtype=bool)
    on_edge = np.zeros_like(u_flat, dtype=bool)
    boundary_eps = 1e-6
    n = poly_u.size

    j = n - 1
    for i in range(n):
        ui, vi = poly_u[i], poly_v[i]
        uj, vj = poly_u[j], poly_v[j]

        # Ray-Casting: Schnitt mit waagrechter Halbgeraden von (u,v) nach rechts
        v_above_edge = (vi > v_flat) != (vj > v_flat)
        denom = (vj - vi) + 1e-12
        inter_u = (uj - ui) * (v_flat - vi) / denom + ui
        intersects = v_above_edge & (u_flat <= inter_u + boundary_eps)
        inside ^= intersects

        # Punkt liegt direkt auf der Kante?
        du = uj - ui
        dv = vj - vi
        seg_len = math.hypot(du, dv)
        if seg_len > 0.0:
            num = np.abs(dv * (u_flat - ui) - du * (v_flat - vi))
            dist = num / (seg_len + 1e-12)
            proj = ((u_flat - ui) * du + (v_flat - vi) * dv) / (
                (seg_len**2) + 1e-12
            )
            on_edge_segment = (
                (dist <= boundary_eps)
                & (proj >= -boundary_eps)
                & (proj <= 1.0 + boundary_eps)
            )
            on_edge |= on_edge_segment
        j = i

    return (inside | on_edge).reshape(u_coords.shape)


def _evaluate_plane_on_grid(
    model: Dict[str, float],
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Vektorisierte Auswertung eines Planar-Modells auf einem 2D-Grid.
    """
    mode = model.get("mode")
    if mode == "constant":
        base = float(model.get("base", 0.0))
        return np.full_like(X, base, dtype=float)
    if mode == "x":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return slope * X + intercept
    if mode == "y":
        slope = float(model.get("slope", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return slope * Y + intercept
    if mode == "xy":
        slope_x = float(model.get("slope_x", model.get("slope", 0.0)))
        slope_y = float(model.get("slope_y", 0.0))
        intercept = float(model.get("intercept", 0.0))
        return slope_x * X + slope_y * Y + intercept
    base = float(model.get("base", 0.0))
    return np.full_like(X, base, dtype=float)


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


def _resample_values_to_grid_nearest(
    values: np.ndarray,
    orig_x: np.ndarray,
    orig_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """
    Resampelt ein 2D-Array auf ein neues Grid per nearest-neighbour:
    Jeder Zielpunkt erhält den Wert des nächstgelegenen Ursprungs-Punktes.
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

    # Mapping von Ziel-x auf nächstgelegenen Ursprungsindex
    idx_x = np.searchsorted(orig_x, target_x, side="left")
    idx_x = np.clip(idx_x, 0, orig_x.size - 1)
    # Korrektur: wirklich nächsten Nachbarn wählen (links/rechts vergleichen)
    left = idx_x - 1
    right = idx_x
    left = np.clip(left, 0, orig_x.size - 1)
    dist_left = np.abs(target_x - orig_x[left])
    dist_right = np.abs(target_x - orig_x[right])
    use_left = dist_left < dist_right
    idx_x[use_left] = left[use_left]

    # Mapping von Ziel-y auf nächstgelegenen Ursprungsindex
    idx_y = np.searchsorted(orig_y, target_y, side="left")
    idx_y = np.clip(idx_y, 0, orig_y.size - 1)
    left_y = idx_y - 1
    right_y = idx_y
    left_y = np.clip(left_y, 0, orig_y.size - 1)
    dist_left_y = np.abs(target_y - orig_y[left_y])
    dist_right_y = np.abs(target_y - orig_y[right_y])
    use_left_y = dist_left_y < dist_right_y
    idx_y[use_left_y] = left_y[use_left_y]

    # Werte mittels Index-Broadcasting zuordnen
    resampled = values[np.ix_(idx_y, idx_x)]
    return resampled

