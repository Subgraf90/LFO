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
    
    # 🎯 NEUER MODUS: PyVista sample() für glatte Ränder
    use_pyvista_sample = bool(getattr(settings, "spl_plot_use_pyvista_sample", False)) if settings else False
    if use_pyvista_sample and source_x is not None and source_y is not None and source_scalars is not None:
        try:
            return _build_surface_mesh_with_pyvista_sample(
                x, y, scalars, z_coords, surface_mask,
                source_x, source_y, source_scalars,
                settings, container, pv_module
            )
        except Exception as exc:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] PyVista sample-Modus fehlgeschlagen, fallback auf altes Verfahren: {exc}")
            # Fallback auf altes Verfahren
            pass

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
    # Die Maske wurde bereits erweitert, daher sollten Randpunkte enthalten sein
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
        print(
            f"[SurfaceGeometry] build_surface_mesh clipping: "
            f"total_cells={total_cells}, rendered={rendered_cells}, "
            f"filtered={total_cells - rendered_cells}"
        )

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
        print(
            f"[SurfaceGeometry] build_vertical_surface_mesh clipping: "
            f"total_cells={total_cells}, rendered={rendered_cells}, "
            f"filtered={total_cells - rendered_cells}"
        )

    faces = np.asarray(face_list, dtype=np.int64)
    if faces.size > 0:
        mesh = pv_module.PolyData(points, faces)
    else:
        mesh = pv_module.PolyData(points)

    mesh["plot_scalars"] = values.ravel()
    return mesh


def _build_surface_mesh_with_pyvista_sample(
    plot_x: np.ndarray,  # Nicht verwendet, aber für Signatur-Kompatibilität
    plot_y: np.ndarray,  # Nicht verwendet, aber für Signatur-Kompatibilität
    plot_scalars: np.ndarray,  # Nicht verwendet, aber für Signatur-Kompatibilität
    plot_z_coords: Optional[np.ndarray],  # Nicht verwendet
    plot_surface_mask: Optional[np.ndarray],  # Nicht verwendet
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_scalars: np.ndarray,
    settings: Any,
    container: Any,
    pv_module: Any,
) -> Any:
    """
    🎯 NEUER MODUS: Erstellt ein feines Surface-Mesh basierend auf der echten Surface-Geometrie
    und mappt SPL-Werte vom groben Berechnungs-Grid mit PyVista sample().
    
    Vorteile:
    - Glatte Ränder ohne Zacken (Mesh folgt exakt der Surface-Geometrie)
    - Vollständige Flächenabdeckung
    - Bessere Darstellung bei schrägen/diagonalen Surfaces
    """
    if DEBUG_SURFACE_GEOMETRY:
        print("[SurfaceGeometry] Verwende PyVista sample()-Modus für feines Surface-Mesh")
    
    # SCHRITT 1: Erstelle grobes Berechnungs-Grid als PyVista StructuredGrid
    source_x_arr = np.asarray(source_x, dtype=float)
    source_y_arr = np.asarray(source_y, dtype=float)
    source_scalars_arr = np.asarray(source_scalars, dtype=float)
    
    if source_scalars_arr.ndim != 2:
        if source_scalars_arr.size == len(source_y_arr) * len(source_x_arr):
            source_scalars_arr = source_scalars_arr.reshape(len(source_y_arr), len(source_x_arr))
        else:
            raise ValueError("source_scalars muss Shape (len(y), len(x)) besitzen.")
    
    # Hole Z-Koordinaten für grobes Grid
    source_z = None
    if container is not None and hasattr(container, "calculation_spl"):
        calc_spl = getattr(container, "calculation_spl", {}) or {}
        z_list = calc_spl.get("sound_field_z")
        if z_list is not None:
            try:
                z_arr = np.asarray(z_list, dtype=float)
                if z_arr.size == len(source_y_arr) * len(source_x_arr):
                    source_z = z_arr.reshape(len(source_y_arr), len(source_x_arr))
            except Exception:
                pass
    
    if source_z is None:
        source_z = np.zeros((len(source_y_arr), len(source_x_arr)), dtype=float)
    
    # Erstelle StructuredGrid aus grobem Berechnungs-Grid
    X_coarse, Y_coarse = np.meshgrid(source_x_arr, source_y_arr, indexing="xy")
    Z_coarse = source_z
    coarse_grid = pv_module.StructuredGrid(X_coarse, Y_coarse, Z_coarse)
    # PyVista StructuredGrid verwendet C-order (row-major)
    coarse_grid["spl_values"] = source_scalars_arr.ravel(order="C")
    
    if DEBUG_SURFACE_GEOMETRY:
        print(
            f"[SurfaceGeometry] Grobes Grid: shape=({len(source_y_arr)}x{len(source_x_arr)}), "
            f"points={coarse_grid.n_points}"
        )
    
    # SCHRITT 2: Erstelle feines Surface-Mesh aus Surface-Definitionen
    surface_definitions = getattr(settings, "surface_definitions", {}) or {}
    if not isinstance(surface_definitions, dict):
        raise RuntimeError("Keine gültigen Surface-Definitionen gefunden.")
    
    # Sammle alle enabled, nicht-hidden Surfaces
    enabled_surfaces = []
    for surface_id, surface_def in surface_definitions.items():
        if isinstance(surface_def, SurfaceDefinition):
            enabled = bool(getattr(surface_def, "enabled", False))
            hidden = bool(getattr(surface_def, "hidden", False))
            points = getattr(surface_def, "points", []) or []
        else:
            enabled = bool(surface_def.get("enabled", False))
            hidden = bool(surface_def.get("hidden", False))
            points = surface_def.get("points", [])
        
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Surface '{surface_id}': enabled={enabled}, hidden={hidden}, "
                f"points={len(points)}"
            )
        
        if enabled and not hidden and len(points) >= 3:
            enabled_surfaces.append((surface_id, points))
            if DEBUG_SURFACE_GEOMETRY:
                # Zeige Bounding-Box dieser Surface
                xs = [p.get("x", 0.0) for p in points]
                ys = [p.get("y", 0.0) for p in points]
                print(
                    f"[SurfaceGeometry] Surface '{surface_id}' hinzugefügt: "
                    f"x=[{min(xs):.2f}, {max(xs):.2f}], y=[{min(ys):.2f}, {max(ys):.2f}], "
                    f"{len(points)} Punkte"
                )
    
    if not enabled_surfaces:
        raise RuntimeError("Keine enabled Surfaces für feines Mesh gefunden.")
    
    if DEBUG_SURFACE_GEOMETRY:
        print(f"[SurfaceGeometry] Gesamt {len(enabled_surfaces)} enabled Surfaces gefunden")
    
    # 🎯 WICHTIG: Beschränke feines Grid auf den SCHNITT von:
    # 1. Grobes Berechnungs-Grid (wo SPL-Werte berechnet wurden)
    # 2. Surface-Bounding-Box (nur enabled Surfaces)
    # So werden nur Punkte erstellt, die sowohl im Grid als auch in den Surfaces liegen
    
    grid_min_x = float(np.min(source_x_arr))
    grid_max_x = float(np.max(source_x_arr))
    grid_min_y = float(np.min(source_y_arr))
    grid_max_y = float(np.max(source_y_arr))
    
    # Berechne Bounding-Box der enabled Surfaces
    surface_min_x = min(p.get("x", 0.0) for _, pts in enabled_surfaces for p in pts)
    surface_max_x = max(p.get("x", 0.0) for _, pts in enabled_surfaces for p in pts)
    surface_min_y = min(p.get("y", 0.0) for _, pts in enabled_surfaces for p in pts)
    surface_max_y = max(p.get("y", 0.0) for _, pts in enabled_surfaces for p in pts)
    
    # Schnitt: Verwende den Bereich, der sowohl im Grid als auch in den Surfaces liegt
    fine_min_x = max(grid_min_x, surface_min_x)
    fine_max_x = min(grid_max_x, surface_max_x)
    fine_min_y = max(grid_min_y, surface_min_y)
    fine_max_y = min(grid_max_y, surface_max_y)
    
    # Prüfe ob Schnitt nicht leer ist
    if fine_min_x >= fine_max_x or fine_min_y >= fine_max_y:
        raise RuntimeError(
            f"Kein Schnitt zwischen Grid-Bereich "
            f"(x=[{grid_min_x:.2f}, {grid_max_x:.2f}], y=[{grid_min_y:.2f}, {grid_max_y:.2f}]) "
            f"und Surface-Bereich "
            f"(x=[{surface_min_x:.2f}, {surface_max_x:.2f}], y=[{surface_min_y:.2f}, {surface_max_y:.2f}])"
        )
    
    if DEBUG_SURFACE_GEOMETRY:
        print(
            f"[SurfaceGeometry] Grobes Grid-Bereich: "
            f"x=[{grid_min_x:.2f}, {grid_max_x:.2f}], y=[{grid_min_y:.2f}, {grid_max_y:.2f}]"
        )
        print(
            f"[SurfaceGeometry] Surface-Bereich: "
            f"x=[{surface_min_x:.2f}, {surface_max_x:.2f}], y=[{surface_min_y:.2f}, {surface_max_y:.2f}]"
        )
        print(
            f"[SurfaceGeometry] Feines Grid-Bereich (Schnitt): "
            f"x=[{fine_min_x:.2f}, {fine_max_x:.2f}], y=[{fine_min_y:.2f}, {fine_max_y:.2f}]"
        )
    
    # Feine Auflösung: ca. 5-6x feiner als Berechnungs-Grid für bessere Randabdeckung
    # Verwende source_x/source_y für Auflösungsberechnung (nicht plot_x/plot_y)
    source_dx = np.mean(np.diff(source_x_arr)) if len(source_x_arr) > 1 else 0.1
    source_dy = np.mean(np.diff(source_y_arr)) if len(source_y_arr) > 1 else 0.1
    fine_resolution = min(source_dx / 5, source_dy / 5, 0.05)  # Maximal 5cm Schrittweite, feiner für bessere Randabdeckung
    fine_resolution = max(fine_resolution, 0.02)  # Minimal 2cm für sehr feine Ränder
    
    # Erstelle feines Grid nur im Schnitt-Bereich (Grid ∩ Surfaces)
    # Erweitere leicht über die Surface-Grenzen hinaus, um Randpunkte zu erfassen
    margin = fine_resolution * 2  # 2x Auflösung als Rand
    fine_x = np.arange(fine_min_x - margin, fine_max_x + margin + fine_resolution, fine_resolution)
    fine_y = np.arange(fine_min_y - margin, fine_max_y + margin + fine_resolution, fine_resolution)
    X_fine, Y_fine = np.meshgrid(fine_x, fine_y, indexing="xy")
    
    if DEBUG_SURFACE_GEOMETRY:
        print(
            f"[SurfaceGeometry] Feines Grid erstellt: shape=({len(fine_y)}x{len(fine_x)}), "
            f"total_points={X_fine.size}, resolution={fine_resolution:.3f}m"
        )
    
    # 🎯 WICHTIG: Prüfe ob Z-Koordinaten im groben Grid vorhanden sind (VOR der Surface-Schleife)
    # Dies wird später in der Surface-Schleife verwendet
    has_z_in_coarse = np.any(np.abs(Z_coarse) > 1e-6)
    
    if DEBUG_SURFACE_GEOMETRY:
        z_coarse_min = float(np.min(Z_coarse))
        z_coarse_max = float(np.max(Z_coarse))
        z_coarse_mean = float(np.mean(Z_coarse))
        print(
            f"[SurfaceGeometry] Grobes Grid Z-Bereich: min={z_coarse_min:.3f}, max={z_coarse_max:.3f}, "
            f"mean={z_coarse_mean:.3f}, has_z={has_z_in_coarse}"
        )
    
    # Berechne Z-Koordinaten für feines Grid (aus Surface-Ebenen)
    # UND prüfe ob Punkt innerhalb der Surfaces liegt
    Z_fine = np.zeros_like(X_fine, dtype=float)
    fine_mask = np.zeros_like(X_fine, dtype=bool)
    
    if DEBUG_SURFACE_GEOMETRY:
        total_points = X_fine.size
        print(
            f"[SurfaceGeometry] Starte Z-Koordinaten-Berechnung für {len(enabled_surfaces)} Surfaces, "
            f"{total_points} Punkte..."
        )
    
    # 🎯 WICHTIG: Jedes Surface wird EINZELN verarbeitet, damit sich der Plot pro Surface nicht ändert
    # Erstelle für jedes Surface ein separates feines Mesh und kombiniere sie dann
    surface_meshes = []
    
    for surface_idx, (surface_id, points) in enumerate(enabled_surfaces):
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry] Verarbeite Surface {surface_idx+1}/{len(enabled_surfaces)} EINZELN: {surface_id}")
        
        # Erstelle Ebenenmodell für dieses Surface
        plane_model, _ = derive_surface_plane(points)
        if plane_model is None:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Surface {surface_id}: Kein Ebenenmodell erstellt, überspringe")
            continue
        
        # 🎯 WICHTIG: Erstelle feines Grid NUR für dieses Surface (nicht für alle zusammen)
        # Bounding-Box nur für dieses Surface
        surface_xs = [p.get("x", 0.0) for p in points]
        surface_ys = [p.get("y", 0.0) for p in points]
        surface_min_x = min(surface_xs)
        surface_max_x = max(surface_xs)
        surface_min_y = min(surface_ys)
        surface_max_y = max(surface_ys)
        
        # Schnitt: Nur Bereich, der sowohl im Grid als auch in diesem Surface liegt
        fine_min_x_surface = max(grid_min_x, surface_min_x)
        fine_max_x_surface = min(grid_max_x, surface_max_x)
        fine_min_y_surface = max(grid_min_y, surface_min_y)
        fine_max_y_surface = min(grid_max_y, surface_max_y)
        
        if fine_min_x_surface >= fine_max_x_surface or fine_min_y_surface >= fine_max_y_surface:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Surface {surface_id}: Kein Schnitt mit Grid-Bereich, überspringe")
            continue
        
        # Erweitere leicht für Randpunkte
        margin = fine_resolution * 2
        fine_x_surface = np.arange(fine_min_x_surface - margin, fine_max_x_surface + margin + fine_resolution, fine_resolution)
        fine_y_surface = np.arange(fine_min_y_surface - margin, fine_max_y_surface + margin + fine_resolution, fine_resolution)
        X_fine_surface, Y_fine_surface = np.meshgrid(fine_x_surface, fine_y_surface, indexing="xy")
        
        # Vektorisierte Punkt-im-Polygon-Prüfung NUR für dieses Surface
        point_mask_surface = _points_in_polygon_batch_plot(X_fine_surface, Y_fine_surface, points)
        if point_mask_surface is None:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Surface {surface_id}: Punkt-im-Polygon-Prüfung fehlgeschlagen")
            continue
        
        points_in_surface = np.count_nonzero(point_mask_surface)
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Surface {surface_id}: {points_in_surface} Punkte im Polygon "
                f"(von {X_fine_surface.size} total)"
            )
        
        if points_in_surface == 0:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Surface {surface_id}: Keine Punkte im Polygon gefunden")
            continue
        
        # Berechne Z-Koordinaten für dieses Surface
        mode = plane_model.get("mode", "xy")
        x_points = X_fine_surface[point_mask_surface]
        y_points = Y_fine_surface[point_mask_surface]
        
        if mode == "constant":
            z_values = np.full_like(x_points, float(plane_model.get("base", 0.0)))
        elif mode == "x":
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            z_values = slope * x_points + intercept
        elif mode == "y":
            slope = float(plane_model.get("slope", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            z_values = slope * y_points + intercept
        else:  # mode == "xy" (default)
            slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
            slope_y = float(plane_model.get("slope_y", 0.0))
            intercept = float(plane_model.get("intercept", 0.0))
            z_values = slope_x * x_points + slope_y * y_points + intercept
        
        # Wenn grobes Grid keine Z-Koordinaten hat, setze Z=0
        if not has_z_in_coarse:
            z_values = np.zeros_like(z_values)
        
        # Erstelle Punkte-Array für dieses Surface
        points_inside_surface = np.column_stack((
            x_points,
            y_points,
            z_values
        ))
        
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Surface {surface_id}: {len(points_inside_surface)} Punkte erstellt, "
                f"Bereich x=[{np.min(x_points):.2f}, {np.max(x_points):.2f}], "
                f"y=[{np.min(y_points):.2f}, {np.max(y_points):.2f}]"
            )
        
        # Erstelle PolyData-Mesh für dieses Surface
        surface_mesh = pv_module.PolyData(points_inside_surface)
        
        # Delaunay-Triangulation für vollständige Flächenabdeckung
        # WICHTIG: Trianguliere jedes Surface einzeln für bessere Randabdeckung
        try:
            # Verwende 2D-Delaunay für bessere Flächenabdeckung
            # alpha=0.0 = keine Löcher, tol=0.0 = keine Toleranz für Randpunkte
            surface_mesh = surface_mesh.delaunay_2d(alpha=0.0, tol=0.0)
            
            if DEBUG_SURFACE_GEOMETRY:
                print(
                    f"[SurfaceGeometry] Surface {surface_id} trianguliert: "
                    f"points={surface_mesh.n_points}, cells={surface_mesh.n_cells}"
                )
        except Exception as exc:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Surface {surface_id}: Delaunay-Triangulation fehlgeschlagen: {exc}, verwende Punkte ohne Zellen")
            # Fallback: Wenn Triangulation fehlschlägt, verwende nur Punkte
            # Das Mesh wird dann später beim Kombinieren trianguliert
        
        # Interpoliere SPL-Werte für dieses Surface-Mesh
        # (wird später gemacht, nachdem alle Surface-Meshes erstellt sind)
        surface_meshes.append((surface_id, surface_mesh, points_inside_surface))
    
    if not surface_meshes:
        raise RuntimeError("Keine Surface-Meshes erstellt.")
    
    if DEBUG_SURFACE_GEOMETRY:
        print(f"[SurfaceGeometry] {len(surface_meshes)} Surface-Meshes erstellt, kombiniere sie jetzt...")
    
    # Kombiniere alle Surface-Meshes zu einem einzigen Mesh
    # WICHTIG: Verwende PyVista merge() um die bereits triangulierten Meshes zu kombinieren
    # Dies behält die Zellen-Struktur bei und verhindert Randprobleme
    if len(surface_meshes) == 1:
        # Nur ein Surface: Verwende direkt das triangulierte Mesh
        surface_id, fine_mesh, points_inside = surface_meshes[0]
        combined_points = fine_mesh.points
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Einzelnes Surface-Mesh verwendet: "
                f"points={fine_mesh.n_points}, cells={fine_mesh.n_cells}"
            )
    else:
        # Mehrere Surfaces: Kombiniere mit merge()
        try:
            # PyVista merge() kombiniert Meshes und behält Zellen-Struktur bei
            meshes_to_merge = [mesh for _, mesh, _ in surface_meshes]
            fine_mesh = pv_module.merge(meshes_to_merge)
            combined_points = fine_mesh.points
        except Exception as exc:
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Fehler beim Kombinieren mit merge(): {exc}, verwende manuelle Kombination")
            # Fallback: Kombiniere Punkte und Zellen manuell
            all_points = []
            all_cells = []
            point_offset = 0
            
            for surface_id, surface_mesh, points_inside in surface_meshes:
                n_points = surface_mesh.n_points
                all_points.append(surface_mesh.points)
                
                # Kombiniere Zellen mit korrigierten Indizes
                if surface_mesh.n_cells > 0:
                    cells = surface_mesh.cells
                    # PyVista Zellen-Format: [n, i1, i2, i3, ...] für jede Zelle
                    cell_array = cells.copy()
                    idx = 0
                    while idx < len(cell_array):
                        n_verts = int(cell_array[idx])
                        if n_verts > 0:
                            # Verschiebe Indizes um point_offset
                            cell_array[idx + 1:idx + 1 + n_verts] += point_offset
                            idx += n_verts + 1
                        else:
                            idx += 1
                    all_cells.append(cell_array)
                
                point_offset += n_points
            
            combined_points = np.vstack(all_points)
            
            if all_cells:
                combined_cells = np.concatenate(all_cells)
                fine_mesh = pv_module.PolyData(combined_points, combined_cells)
            else:
                fine_mesh = pv_module.PolyData(combined_points)
                # Trianguliere neu
                try:
                    fine_mesh = fine_mesh.delaunay_2d(alpha=0.0, tol=0.0)
                except Exception:
                    fine_mesh = fine_mesh.delaunay_2d()
        
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Kombiniertes Mesh aus {len(surface_meshes)} Surfaces: "
                f"points={fine_mesh.n_points}, cells={fine_mesh.n_cells} "
                f"(Zellen aus individuellen Triangulationen beibehalten)"
            )
    
    # Setze fine_mask für Debug-Zwecke (nicht mehr verwendet für Mesh-Erstellung)
    fine_mask = np.ones(len(combined_points), dtype=bool)
    
    if DEBUG_SURFACE_GEOMETRY:
        total_points_in_mask = np.count_nonzero(fine_mask)
        total_points_in_grid = X_fine.size
        coverage_percent = 100.0 * total_points_in_mask / total_points_in_grid if total_points_in_grid > 0 else 0.0
        print(
            f"[SurfaceGeometry] Z-Koordinaten-Berechnung abgeschlossen: "
            f"{total_points_in_mask} Punkte in Surfaces (von {total_points_in_grid} total, "
            f"{coverage_percent:.1f}% Abdeckung)"
        )
        
        # Prüfe ob Surface-Polygon-Punkte im feinen Grid-Bereich liegen
        for surface_id, points in enabled_surfaces:
            poly_x = [p.get("x", 0.0) for p in points]
            poly_y = [p.get("y", 0.0) for p in points]
            poly_x_min, poly_x_max = min(poly_x), max(poly_x)
            poly_y_min, poly_y_max = min(poly_y), max(poly_y)
            print(
                f"[SurfaceGeometry] Surface '{surface_id}' Polygon-Bereich: "
                f"x=[{poly_x_min:.2f}, {poly_x_max:.2f}], y=[{poly_y_min:.2f}, {poly_y_max:.2f}]"
            )
            print(
                f"[SurfaceGeometry] Feines Grid-Bereich: "
                f"x=[{fine_min_x:.2f}, {fine_max_x:.2f}], y=[{fine_min_y:.2f}, {fine_max_y:.2f}]"
            )
            # Prüfe ob Polygon vollständig im Grid-Bereich liegt
            poly_in_grid = (
                poly_x_min >= fine_min_x - 1e-6 and poly_x_max <= fine_max_x + 1e-6 and
                poly_y_min >= fine_min_y - 1e-6 and poly_y_max <= fine_max_y + 1e-6
            )
            print(
                f"[SurfaceGeometry] Polygon vollständig im Grid-Bereich: {poly_in_grid}"
            )
    
    # 🎯 WICHTIG: Die Z-Koordinaten im feinen Mesh müssen mit denen im groben Grid übereinstimmen
    # für korrekte Interpolation. Wenn das grobe Grid Z=0 hat, müssen wir die Z-Koordinaten
    # im feinen Mesh entsprechend anpassen (oder umgekehrt).
    
    # Prüfe ob Z-Koordinaten im groben Grid vorhanden sind (VOR der Surface-Schleife)
    has_z_in_coarse = np.any(np.abs(Z_coarse) > 1e-6)
    
    if DEBUG_SURFACE_GEOMETRY:
        z_coarse_min = float(np.min(Z_coarse))
        z_coarse_max = float(np.max(Z_coarse))
        z_coarse_mean = float(np.mean(Z_coarse))
        print(
            f"[SurfaceGeometry] Grobes Grid Z-Bereich: min={z_coarse_min:.3f}, max={z_coarse_max:.3f}, "
            f"mean={z_coarse_mean:.3f}, has_z={has_z_in_coarse}"
        )
    
    # fine_mesh wurde bereits oben aus kombinierten Surface-Meshes erstellt
    if DEBUG_SURFACE_GEOMETRY:
        print(
            f"[SurfaceGeometry] Kombiniertes feines Mesh: points={fine_mesh.n_points}, "
            f"cells={fine_mesh.n_cells}, resolution={fine_resolution:.3f}m"
        )
        # Prüfe Z-Koordinaten-Bereich im feinen Mesh
        if fine_mesh.n_points > 0:
            fine_z_coords = fine_mesh.points[:, 2]
            z_fine_min = float(np.min(fine_z_coords))
            z_fine_max = float(np.max(fine_z_coords))
            z_fine_mean = float(np.mean(fine_z_coords))
            print(
                f"[SurfaceGeometry] Feines Mesh Z-Bereich: min={z_fine_min:.3f}, max={z_fine_max:.3f}, mean={z_fine_mean:.3f}"
            )
    
    # SCHRITT 3: Mappe SPL-Werte vom groben Grid auf feines Mesh
    # 🎯 WICHTIG: Interpoliere für jedes Surface EINZELN, damit sich der Plot pro Surface nicht ändert
    try:
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry] Starte Interpolation für {len(surface_meshes)} Surfaces...")
        
        # Interpoliere Werte für jedes Surface einzeln
        all_interpolated_values = []
        
        for surface_idx, (surface_id, surface_mesh, points_inside) in enumerate(surface_meshes):
            if DEBUG_SURFACE_GEOMETRY:
                print(
                    f"[SurfaceGeometry] Interpoliere SPL-Werte für Surface {surface_idx+1}/{len(surface_meshes)}: "
                    f"{surface_id} ({len(points_inside)} Punkte)"
                )
            
            # Speichere ursprüngliche Punkte für dieses Surface
            original_points = points_inside.copy()
            original_n_points = len(original_points)
            
            # Berechne Radius für Interpolation basierend auf Grid-Auflösung
            source_dx = np.mean(np.diff(source_x_arr)) if len(source_x_arr) > 1 else 0.1
            source_dy = np.mean(np.diff(source_y_arr)) if len(source_y_arr) > 1 else 0.1
            # Radius sollte groß genug sein, um mehrere Nachbarn zu erfassen
            interpolation_radius = max(source_dx, source_dy) * 2.0  # 2x Grid-Schrittweite
            
            if DEBUG_SURFACE_GEOMETRY:
                print(f"[SurfaceGeometry] Interpolations-Radius: {interpolation_radius:.3f}m")
            
            # 🎯 WICHTIG: Erstelle Surface-spezifische Maske im groben Grid
            # Nur Werte innerhalb dieser Surface sollten für die Interpolation verwendet werden
            # Dies verhindert, dass sich der Plot für dieses Surface ändert, wenn andere Surfaces enabled werden
            
            # Berechne Bounding Box für dieses Surface
            surface_xs = [p.get("x", 0.0) for p in points]
            surface_ys = [p.get("y", 0.0) for p in points]
            surface_min_x = min(surface_xs)
            surface_max_x = max(surface_xs)
            surface_min_y = min(surface_ys)
            surface_max_y = max(surface_ys)
            
            # Erweitere Bounding Box leicht für Interpolation (2x Grid-Schrittweite)
            source_dx = np.mean(np.diff(source_x_arr)) if len(source_x_arr) > 1 else 0.1
            source_dy = np.mean(np.diff(source_y_arr)) if len(source_y_arr) > 1 else 0.1
            margin = max(source_dx, source_dy) * 2.0
            surface_min_x -= margin
            surface_max_x += margin
            surface_min_y -= margin
            surface_max_y += margin
            
            # Erstelle Maske für Punkte im groben Grid, die für dieses Surface relevant sind
            # Extrahiere Punkte und Werte aus grobem Grid (einmal für alle Surfaces)
            if surface_idx == 0:
                coarse_points = coarse_grid.points
                coarse_values = coarse_grid["spl_values"]
                
                if DEBUG_SURFACE_GEOMETRY:
                    # Analysiere grobes Grid (nur einmal)
                    coarse_x_min = float(np.min(coarse_points[:, 0]))
                    coarse_x_max = float(np.max(coarse_points[:, 0]))
                    coarse_y_min = float(np.min(coarse_points[:, 1]))
                    coarse_y_max = float(np.max(coarse_points[:, 1]))
                    coarse_z_min = float(np.min(coarse_points[:, 2]))
                    coarse_z_max = float(np.max(coarse_points[:, 2]))
                    coarse_val_min = float(np.nanmin(coarse_values))
                    coarse_val_max = float(np.nanmax(coarse_values))
                    coarse_val_mean = float(np.nanmean(coarse_values))
                    print(
                        f"[SurfaceGeometry] Grobes Grid Analyse: "
                        f"points={len(coarse_points)}, "
                        f"x=[{coarse_x_min:.2f}, {coarse_x_max:.2f}], "
                        f"y=[{coarse_y_min:.2f}, {coarse_y_max:.2f}], "
                        f"z=[{coarse_z_min:.3f}, {coarse_z_max:.3f}], "
                        f"SPL=[{coarse_val_min:.2f}, {coarse_val_max:.2f}], mean={coarse_val_mean:.2f} dB"
                    )
            
            # Erstelle Maske für dieses Surface im groben Grid
            # Schritt 1: Bounding-Box-Filter (schnell)
            surface_mask_coarse_bbox = (
                (coarse_points[:, 0] >= surface_min_x) & (coarse_points[:, 0] <= surface_max_x) &
                (coarse_points[:, 1] >= surface_min_y) & (coarse_points[:, 1] <= surface_max_y)
            )
            
            # Schritt 2: Präzise Punkt-in-Polygon-Prüfung
            # _points_in_polygon_batch_plot erwartet 2D-Arrays (meshgrid-Format)
            # Rekonstruiere 2D-Form aus source_x_arr und source_y_arr
            ny, nx = len(source_y_arr), len(source_x_arr)
            if len(coarse_points) == ny * nx:
                # Reshape zu 2D-Meshgrid-Format für vektorisierte Prüfung
                X_coarse_2d = coarse_points[:, 0].reshape(ny, nx)
                Y_coarse_2d = coarse_points[:, 1].reshape(ny, nx)
                surface_mask_polygon_2d = _points_in_polygon_batch_plot(X_coarse_2d, Y_coarse_2d, points)
                if surface_mask_polygon_2d is not None:
                    surface_mask_polygon = surface_mask_polygon_2d.ravel()
                else:
                    # Fallback: Nur Bounding-Box
                    surface_mask_polygon = surface_mask_coarse_bbox
            else:
                # Fallback: Punkt-für-Punkt-Prüfung (sollte nicht vorkommen)
                coarse_points_xy = coarse_points[:, :2]
                surface_mask_polygon = np.array([
                    _point_in_polygon_simple(pt[0], pt[1], points)
                    for pt in coarse_points_xy
                ], dtype=bool)
            
            if surface_mask_polygon is not None:
                # Kombiniere Bounding-Box und Polygon-Maske
                # Erweitere Polygon-Maske um Margin (für Interpolation an Rändern)
                surface_mask_coarse = surface_mask_coarse_bbox & surface_mask_polygon
                # Erweitere um 1 Pixel in alle Richtungen für bessere Randabdeckung
                try:
                    from scipy import ndimage
                    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
                    # Wandle 1D-Maske in 2D um für Dilatation
                    if surface_mask_coarse.ndim == 1:
                        # Rekonstruiere 2D-Form aus source_x_arr und source_y_arr
                        ny, nx = len(source_y_arr), len(source_x_arr)
                        if len(surface_mask_coarse) == ny * nx:
                            surface_mask_2d = surface_mask_coarse.reshape(ny, nx)
                            surface_mask_2d = ndimage.binary_dilation(surface_mask_2d, structure=structure)
                            surface_mask_coarse = surface_mask_2d.ravel()
                except (ImportError, Exception):
                    # Fallback: Verwende nur Bounding-Box + Polygon-Maske
                    pass
            else:
                # Fallback: Verwende nur Bounding-Box-Maske
                surface_mask_coarse = surface_mask_coarse_bbox
            
            # Extrahiere nur relevante Punkte und Werte für dieses Surface
            coarse_points_surface = coarse_points[surface_mask_coarse]
            coarse_values_surface = coarse_values[surface_mask_coarse]
            
            if DEBUG_SURFACE_GEOMETRY:
                print(
                    f"[SurfaceGeometry] Surface {surface_id}: "
                    f"Verwende {len(coarse_points_surface)}/{len(coarse_points)} Punkte aus grobem Grid "
                    f"(Bounding-Box: x=[{surface_min_x:.2f}, {surface_max_x:.2f}], "
                    f"y=[{surface_min_y:.2f}, {surface_max_y:.2f}])"
                )
            
            # Prüfe ob genug Punkte vorhanden sind
            if len(coarse_points_surface) < 3:
                if DEBUG_SURFACE_GEOMETRY:
                    print(
                        f"[SurfaceGeometry] WARNUNG: Surface {surface_id}: "
                        f"Zu wenige Punkte im groben Grid ({len(coarse_points_surface)}), "
                        f"verwende alle Punkte"
                    )
                coarse_points_surface = coarse_points
                coarse_values_surface = coarse_values
            
            # Manuelle Interpolation: Für jeden Punkt in diesem Surface
            interpolated_values = np.full(original_n_points, np.nan, dtype=float)
            
            # Batch-Verarbeitung: Erstelle PointSet für alle Punkte gleichzeitig
            # Aber PyVista interpolate() reduziert das Mesh, daher müssen wir es anders machen
            # Verwende scipy.spatial.cKDTree für schnelle Nearest-Neighbor-Suche
            try:
                from scipy.spatial import cKDTree
                
                # Erstelle KD-Tree aus Surface-spezifischen Punkten im groben Grid
                # WICHTIG: Verwende nur X/Y für KD-Tree, da Z-Koordinaten möglicherweise nicht übereinstimmen
                # (z.B. wenn grobes Grid Z=0 hat, aber feines Mesh Z aus Surface-Ebenen hat)
                coarse_points_xy = coarse_points_surface[:, :2]  # Nur X/Y, nur für dieses Surface
                fine_points_xy = original_points[:, :2]  # Nur X/Y
                
                tree = cKDTree(coarse_points_xy)
                
                # Finde nächstgelegene Punkte für alle Punkte im feinen Mesh
                # Verwende mehr Nachbarn für bessere Interpolation
                # WICHTIG: Verwende nur die Anzahl der Surface-spezifischen Punkte
                k_neighbors = min(8, len(coarse_points_surface))
                distances, indices = tree.query(fine_points_xy, k=k_neighbors, distance_upper_bound=interpolation_radius)
                
                # Interpoliere Werte basierend auf Entfernung (Inverse Distance Weighting)
                if distances.ndim == 1:
                    # Nur ein Nachbar
                    # Prüfe ob Punkt innerhalb des Radius liegt
                    valid = distances < np.inf
                    # WICHTIG: Verwende Surface-spezifische Werte
                    interpolated_values[valid] = coarse_values_surface[indices[valid]]
                else:
                    # Mehrere Nachbarn - verwende IDW
                    # Markiere ungültige Punkte (außerhalb des Radius)
                    valid_mask = distances < np.inf
                    
                    for i in range(original_n_points):
                        # Finde gültige Nachbarn für diesen Punkt
                        valid_neighbors = valid_mask[i]
                        if not np.any(valid_neighbors):
                            # Kein Nachbar gefunden - verwende nächstgelegenen Punkt
                            # WICHTIG: Verwende Surface-spezifische Werte
                            interpolated_values[i] = coarse_values_surface[indices[i, 0]]
                            continue
                        
                        # Verwende nur gültige Nachbarn
                        valid_dists = distances[i, valid_neighbors]
                        valid_indices = indices[i, valid_neighbors]
                        
                        # Vermeide Division durch Null
                        valid_dists = np.maximum(valid_dists, 1e-10)
                        weights = 1.0 / (valid_dists ** 2)  # Quadratische Gewichtung
                        weight_sum = np.sum(weights)
                        
                        if weight_sum > 0:
                            # WICHTIG: Verwende Surface-spezifische Werte
                            interpolated_values[i] = np.sum(coarse_values_surface[valid_indices] * weights) / weight_sum
                        else:
                            interpolated_values[i] = coarse_values_surface[valid_indices[0]]
                
                if DEBUG_SURFACE_GEOMETRY:
                    valid_count = np.count_nonzero(np.isfinite(interpolated_values))
                    if valid_count > 0:
                        interp_min = float(np.nanmin(interpolated_values))
                        interp_max = float(np.nanmax(interpolated_values))
                        interp_mean = float(np.nanmean(interpolated_values))
                        print(
                            f"[SurfaceGeometry] Surface {surface_id} IDW-Interpolation: "
                            f"points={len(interpolated_values)}, "
                            f"valid={valid_count}/{len(interpolated_values)} "
                            f"({100.0*valid_count/len(interpolated_values):.1f}%), "
                            f"k_neighbors={k_neighbors}, "
                            f"SPL=[{interp_min:.2f}, {interp_max:.2f}], mean={interp_mean:.2f} dB"
                        )
                    else:
                        print(
                            f"[SurfaceGeometry] Surface {surface_id} IDW-Interpolation: "
                            f"points={len(interpolated_values)}, "
                            f"valid={valid_count}/{len(interpolated_values)} "
                            f"({100.0*valid_count/len(interpolated_values):.1f}%), "
                            f"k_neighbors={k_neighbors} - WARNUNG: Keine gültigen Werte!"
                        )
            except ImportError:
                # Fallback: Einfache Nearest-Neighbor-Interpolation ohne scipy
                if DEBUG_SURFACE_GEOMETRY:
                    print(f"[SurfaceGeometry] Surface {surface_id}: scipy nicht verfügbar, verwende einfache Nearest-Neighbor-Interpolation")
                
                # Einfache Nearest-Neighbor-Suche
                # WICHTIG: Verwende nur Surface-spezifische Punkte
                for i, point in enumerate(original_points):
                    # Finde nächstgelegenen Punkt im Surface-spezifischen groben Grid
                    dists = np.linalg.norm(coarse_points_surface - point, axis=1)
                    nearest_idx = np.argmin(dists)
                    interpolated_values[i] = coarse_values_surface[nearest_idx]
                
                if DEBUG_SURFACE_GEOMETRY:
                    valid_count = np.count_nonzero(np.isfinite(interpolated_values))
                    print(
                        f"[SurfaceGeometry] Surface {surface_id} Nearest-Neighbor-Interpolation: "
                        f"points={len(interpolated_values)}, "
                        f"valid={valid_count}/{len(interpolated_values)} "
                        f"({100.0*valid_count/len(interpolated_values):.1f}%)"
                    )
            
            # Weise interpolierte Werte diesem Surface zu
            all_interpolated_values.append(interpolated_values)
        
        # Kombiniere alle interpolierten Werte
        combined_interpolated_values = np.concatenate(all_interpolated_values)
        
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry] Alle Surfaces interpoliert: "
                f"total_points={len(combined_interpolated_values)}, "
                f"valid={np.count_nonzero(np.isfinite(combined_interpolated_values))}"
            )
        
        # Weise kombinierte interpolierte Werte dem kombinierten feinen Mesh zu
        fine_mesh["plot_scalars"] = combined_interpolated_values
        
        if DEBUG_SURFACE_GEOMETRY:
            valid_values = np.count_nonzero(np.isfinite(fine_mesh["plot_scalars"]))
            if fine_mesh.n_points > 0:
                spl_min = float(np.nanmin(fine_mesh["plot_scalars"]))
                spl_max = float(np.nanmax(fine_mesh["plot_scalars"]))
                spl_mean = float(np.nanmean(fine_mesh["plot_scalars"]))
                print(
                    f"[SurfaceGeometry] Sample-Ergebnis: valid_values={valid_values}/{fine_mesh.n_points} "
                    f"({100.0*valid_values/fine_mesh.n_points:.1f}%), "
                    f"SPL=[{spl_min:.2f}, {spl_max:.2f}], mean={spl_mean:.2f} dB"
                )
                # Prüfe Bereich der Punkte mit gültigen Werten
                valid_mask = np.isfinite(fine_mesh["plot_scalars"])
                if np.any(valid_mask):
                    valid_points = fine_mesh.points[valid_mask]
                    print(
                        f"[SurfaceGeometry] Gültige Punkte Bereich: "
                        f"x=[{np.min(valid_points[:, 0]):.2f}, {np.max(valid_points[:, 0]):.2f}], "
                        f"y=[{np.min(valid_points[:, 1]):.2f}, {np.max(valid_points[:, 1]):.2f}], "
                        f"count={len(valid_points)}"
                    )
            else:
                print(f"[SurfaceGeometry] Sample-Ergebnis: Mesh ist leer!")
        
        # Entferne Punkte ohne gültige Werte (außerhalb des Berechnungsbereichs)
        valid_mask = np.isfinite(fine_mesh["plot_scalars"])
        if not np.all(valid_mask):
            if DEBUG_SURFACE_GEOMETRY:
                removed = np.count_nonzero(~valid_mask)
                print(f"[SurfaceGeometry] Entferne {removed} Punkte ohne gültige Werte")
            fine_mesh = fine_mesh.extract_points(valid_mask, include_cells=True)
        
        if fine_mesh.n_points == 0:
            raise RuntimeError("Nach Entfernen ungültiger Punkte ist das Mesh leer.")
        
        return fine_mesh
    except Exception as exc:
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry] PyVista sample() fehlgeschlagen: {exc}")
        raise


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
    # Begrenze Upscaling für Performance: zu hohe Werte erzeugen Millionen Plot-Punkte
    upscale_factor = max(1, upscale_factor)
    if upscale_factor > 6:
        if DEBUG_SURFACE_GEOMETRY:
            print(
                "[SurfaceGeometry] plot_upscale_factor zu hoch, clamp auf 6 "
                f"(requested={upscale_factor})"
            )
        upscale_factor = 6

    plot_x = source_x.copy()
    plot_y = source_y.copy()
    plot_vals = values.copy()
    if DEBUG_SURFACE_GEOMETRY:
        total_points = int(plot_x.size * plot_y.size)
        print(
            "[SurfaceGeometry] Ausgangs-Plot-Grid:",
            f"shape=({plot_y.size}, {plot_x.size}) (ny, nx),",
            f"total_points={total_points}",
        )

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

        # 📌 Upscaling der SPL-Werte:
        # Umschaltbar zwischen Nearest-Neighbour (blockig, exakt) und
        # linearer Interpolation (glattere Darstellung).
        use_linear = bool(getattr(settings, "spl_plot_use_linear_resample", False))
        if use_linear:
            plot_vals = _resample_values_to_grid(
                plot_vals, orig_plot_x, orig_plot_y, expanded_x, expanded_y
            )
        else:
            plot_vals = _resample_values_to_grid_nearest(
                plot_vals, orig_plot_x, orig_plot_y, expanded_x, expanded_y
            )

        if DEBUG_SURFACE_GEOMETRY:
            mode = "linear" if use_linear else "nearest"
            print(
                "[SurfaceGeometry] Upscaling aktiv:",
                f"mode={mode}, upscale_factor={upscale_factor}, "
                f"source_grid=({orig_plot_y.size}x{orig_plot_x.size}), "
                f"plot_grid=({expanded_y.size}x{expanded_x.size})",
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
            if use_linear:
                z_coords = _resample_values_to_grid(
                    orig_z_coords, orig_plot_x, orig_plot_y, expanded_x, expanded_y
                )
            else:
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
    
    if plot_mask is None:
        plot_mask = _build_plot_surface_mask(plot_x, plot_y, settings, dilate=False)
    if plot_mask is None and surface_mask is not None:
        plot_mask = _convert_point_mask_to_cell_mask(surface_mask)
    if plot_mask is None:
        print("[SurfaceGeometry] Fehler: Keine gültige Surface-Maske – breche Rendering ab.")
        raise RuntimeError("Surface mask missing for plot geometry.")
    _debug_surface_info(settings, plot_x, plot_y, plot_mask, "plot mask")
    if DEBUG_SURFACE_GEOMETRY and z_coords is not None:
        nonzero_z = int(np.count_nonzero(z_coords))
        print(
            "[SurfaceGeometry] Plot-Z-Grid:",
            f"nonzero_points={nonzero_z},",
            f"total_points={int(z_coords.size)}",
        )

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
            print("[SurfaceGeometry][Vertical] Keine gültigen surface_samples / surface_fields vorhanden.")
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
            print(f"[SurfaceGeometry][Vertical] Kein vertical-Payload für Surface '{surface_id}' gefunden.")
        return None

    coords = np.asarray(payload.get("coordinates", []), dtype=float)
    if coords.size == 0:
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry][Vertical] Payload für '{surface_id}' enthält keine Koordinaten.")
        return None
    coords = coords.reshape(-1, 3)
    field_values = surface_fields.get(surface_id)
    if field_values is None:
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry][Vertical] Keine Feldwerte für Surface '{surface_id}' gefunden.")
        return None
    field_arr = np.asarray(field_values, dtype=complex).reshape(-1)
    if field_arr.size != coords.shape[0]:
        if DEBUG_SURFACE_GEOMETRY:
            print(
                f"[SurfaceGeometry][Vertical] Feldlänge passt nicht zu Koordinaten: "
                f"{field_arr.size} vs {coords.shape[0]}"
            )
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
            print(f"[SurfaceGeometry][Vertical] Leere u/v-Samples für Surface '{surface_id}'.")
        return None

    # Rekonstruiere das lokale (u, v)-Raster aus den Sample-Punkten:
    # Die Vertikal-Samples werden in SurfaceGridCalculator auf einem regulären
    # Gitter erzeugt. Wir nehmen die eindeutigen u- und v-Werte als Achsen.
    u_axis = np.unique(u_samples)
    v_axis = np.unique(v_samples)
    if u_axis.size < 2 or v_axis.size < 2:
        if DEBUG_SURFACE_GEOMETRY:
            print(f"[SurfaceGeometry][Vertical] Zu wenig Stützstellen für '{surface_id}'.")
        return None

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

    # Maskenmatrix im (u, v)-Grid analog _build_plot_surface_mask (Punkt-Maske)
    # WICHTIG: Diese Maske ist "strict" entlang der Polygonlinie; das
    # eigentliche "calculate beyond surface" passiert bereits bei der
    # Sample-Erzeugung (dilatierte Maske in SurfaceGridCalculator).
    mask_uv = _points_in_polygon_batch_uv(U, V, poly_u, poly_v)
    if mask_uv is None:
        # Fallback: alles sichtbar
        mask_uv = np.ones_like(U, dtype=bool)

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
                    print(
                        "[SurfaceGeometry] Z-Neuberechnung übersprungen:"
                        " alle aktiven Surfaces sind z≈0 (plan, konstant)."
                    )
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
    # Zweite Runde: Fülle Z-Werte für Randpunkte iterativ
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
        print(
            f"[SurfaceGeometry] Z-Füllung: mask_points={mask_points}, "
            f"vorher={points_with_z_before}, "
            f"nachher={points_with_z_after}, "
            f"gefüllt={points_with_z_after - points_with_z_before}, "
            f"iterationen={iteration + 1}"
        )
    
    return Z_grid


def _point_in_polygon_simple(x: float, y: float, polygon_points: List[Dict[str, float]]) -> bool:
    """Einfache Punkt-in-Polygon-Prüfung für Z-Berechnung."""
    if len(polygon_points) < 3:
        return False
    
    px = np.array([float(p.get("x", 0.0)) for p in polygon_points], dtype=float)
    py = np.array([float(p.get("y", 0.0)) for p in polygon_points], dtype=float)
    
    n = len(px)
    inside = False
    j = n - 1
    
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        
        if ((yi > y) != (yj > y)) and (x <= (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
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

