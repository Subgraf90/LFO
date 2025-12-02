"""SPL-Surface Rendering für den 3D-SPL-Plot."""

from __future__ import annotations

import time
from typing import Optional, Any, Dict, Callable

import numpy as np
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib import cm
from matplotlib.path import Path

from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import (
    SurfaceDefinition,
    derive_surface_plane,
    prepare_vertical_plot_geometry,
    build_vertical_surface_mesh,
)


def update_surface_scalars(surface_mesh: Any, flat_scalars: np.ndarray) -> bool:
    """Aktualisiert die Scalars eines Surface-Meshes.
    
    Args:
        surface_mesh: PyVista Mesh
        flat_scalars: Flache Scalar-Werte (1D-Array)
        
    Returns:
        True wenn erfolgreich, sonst False
    """
    if surface_mesh is None:
        return False

    if flat_scalars.size == surface_mesh.n_points:
        surface_mesh.point_data['plot_scalars'] = flat_scalars
        if hasattr(surface_mesh, "modified"):
            surface_mesh.modified()
        return True

    if flat_scalars.size == surface_mesh.n_cells:
        surface_mesh.cell_data['plot_scalars'] = flat_scalars
        if hasattr(surface_mesh, "modified"):
            surface_mesh.modified()
        return True

    return False


def render_surfaces_textured(
    plotter: Any,
    pv_module: Any,
    settings: Any,
    container: Any,
    geometry: Any,
    original_plot_values: np.ndarray,
    cbar_min: float,
    cbar_max: float,
    cmap_object: str | Any,
    colorization_mode: str,
    cbar_step: float,
    surface_texture_actors: dict,
    surface_name: str,
    bilinear_interpolate_func: Callable,
    nearest_interpolate_func: Callable,
    quantize_to_steps_func: Callable,
) -> None:
    """
    Renderpfad für horizontale Surfaces als 2D-Texturen.
    
    Vollständige Implementierung aus PlotSPL3D.py Zeilen 3993-4712.
    
    Args:
        plotter: PyVista Plotter
        pv_module: PyVista Modul
        settings: Settings-Objekt
        container: Container-Objekt
        geometry: Plot-Geometrie-Objekt
        original_plot_values: SPL-Werte als 2D-Array
        cbar_min: Minimum für Colorbar
        cbar_max: Maximum für Colorbar
        cmap_object: Colormap-Objekt oder String
        colorization_mode: Modus ('Gradient' oder 'Color step')
        cbar_step: Schrittweite für Colorbar
        surface_texture_actors: Dictionary für Texture-Actors
        surface_name: Name-Präfix für Surface-Actors
        bilinear_interpolate_func: Funktion für bilineare Interpolation
        nearest_interpolate_func: Funktion für Nearest-Neighbor Interpolation
        quantize_to_steps_func: Funktion für Quantisierung
    """
    if plotter is None:
        return

    try:
        import pyvista as pv  # type: ignore
    except Exception:
        return

    # Quelle: Berechnungsraster
    source_x = np.asarray(getattr(geometry, "source_x", []), dtype=float)
    source_y = np.asarray(getattr(geometry, "source_y", []), dtype=float)
    values = np.asarray(original_plot_values, dtype=float)
    if values.shape != (len(source_y), len(source_x)):
        return

    # Bestimme ob Step-Modus aktiv ist (vor der Verwendung definieren)
    is_step_mode = colorization_mode == "Color step" and cbar_step > 0

    # Auflösung der Textur in Metern (XY)
    # Standard: 0.02m (2cm pro Pixel) für schärfere Darstellung (vorher: 0.05m)
    # 🎯 Gradient-Modus: Leicht erhöhte Auflösung für smooth rendering (Performance-Optimiert)
    base_tex_res = float(getattr(settings, "spl_surface_texture_resolution", 0.02) or 0.02)
    if is_step_mode:
        # Color step: Normale Auflösung (harte Stufen benötigen keine hohe Auflösung)
        tex_res = base_tex_res
    else:
        # Gradient: 2x feinere Auflösung (Performance-Kompromiss)
        # Bilineare Interpolation sorgt bereits für glatte Übergänge
        # (0.01m = 1cm pro Pixel statt 20mm = 2x mehr Pixel)
        tex_res = base_tex_res * 0.5

    # Colormap vorbereiten
    if isinstance(cmap_object, str):
        base_cmap = cm.get_cmap(cmap_object)
    else:
        base_cmap = cmap_object
    norm = Normalize(vmin=cbar_min, vmax=cbar_max)

    # Aktive Surfaces ermitteln
    surface_definitions = getattr(settings, "surface_definitions", {}) or {}
    enabled_surfaces: list[tuple[str, list[dict[str, float]], Any]] = []
    if isinstance(surface_definitions, dict):
        for surface_id, surface_def in surface_definitions.items():
            if isinstance(surface_def, SurfaceDefinition):
                enabled = bool(getattr(surface_def, "enabled", False))
                hidden = bool(getattr(surface_def, "hidden", False))
                points = getattr(surface_def, "points", []) or []
                surface_obj = surface_def
            else:
                enabled = bool(surface_def.get("enabled", False))
                hidden = bool(surface_def.get("hidden", False))
                points = surface_def.get("points", []) or []
                surface_obj = surface_def
            if enabled and not hidden and len(points) >= 3:
                enabled_surfaces.append((str(surface_id), points, surface_obj))
    

    # Nicht mehr benötigte Textur-Actors entfernen
    active_ids = {sid for sid, _, _ in enabled_surfaces}
    for sid, texture_data in list(surface_texture_actors.items()):
        if sid not in active_ids:
            try:
                actor = texture_data.get('actor') if isinstance(texture_data, dict) else texture_data
                if actor is not None:
                    plotter.remove_actor(actor)
            except Exception:
                pass
            surface_texture_actors.pop(sid, None)

    if not enabled_surfaces:
        return
    for surface_id, points, surface_obj in enabled_surfaces:
        try:
            t_start = time.perf_counter()
            poly_x = np.array([p.get("x", 0.0) for p in points], dtype=float)
            poly_y = np.array([p.get("y", 0.0) for p in points], dtype=float)
            poly_z = np.array([p.get("z", 0.0) for p in points], dtype=float)
            if poly_x.size == 0 or poly_y.size == 0:
                continue

            xmin, xmax = float(poly_x.min()), float(poly_x.max())
            ymin, ymax = float(poly_y.min()), float(poly_y.max())
            
            # 🎯 Berechne Planmodell für geneigte Flächen
            # Konvertiere Punkte in Dict-Format für derive_surface_plane
            dict_points = [
                {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                for p in points
            ]
            plane_model, _ = derive_surface_plane(dict_points)
            
            # 🎯 Prüfe ob Fläche senkrecht ist (nicht planar)
            # Senkrechte Flächen benötigen spezielle Behandlung mit (u,v)-Koordinaten
            is_vertical = False
            orientation = None
            wall_axis = None
            wall_value = None
            
            if plane_model is None:
                # Prüfe ob es eine senkrechte Fläche ist
                x_span = float(np.ptp(poly_x)) if poly_x.size > 0 else 0.0
                y_span = float(np.ptp(poly_y)) if poly_y.size > 0 else 0.0
                z_span = float(np.ptp(poly_z)) if poly_z.size > 0 else 0.0
                
                # Senkrechte Fläche: XY-Projektion ist fast eine Linie, aber signifikante Z-Höhe
                eps_line = 1e-6
                has_height = z_span > 1e-3
                
                if y_span < eps_line and x_span >= eps_line and has_height:
                    # X-Z-Wand: y ≈ const (Fläche verläuft in X-Z-Richtung)
                    is_vertical = True
                    orientation = "xz"
                    wall_axis = "y"
                    wall_value = float(np.mean(poly_y))
                elif x_span < eps_line and y_span >= eps_line and has_height:
                    # Y-Z-Wand: x ≈ const (Fläche verläuft in Y-Z-Richtung)
                    is_vertical = True
                    orientation = "yz"
                    wall_axis = "x"
                    wall_value = float(np.mean(poly_x))
                
                if not is_vertical:
                    # Fallback: Wenn kein Planmodell gefunden wurde, verwende konstante Höhe (Mittelwert)
                    plane_model = {
                        "mode": "constant",
                        "base": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                        "slope": 0.0,
                        "intercept": float(np.mean(poly_z)) if poly_z.size > 0 else 0.0,
                    }

            # 🎯 SPEZIELLER PFAD FÜR SENKRECHTE FLÄCHEN
            if is_vertical:
                # Senkrechte Flächen verwenden (u,v)-Koordinaten statt (x,y)
                # u = x oder y (je nach Orientierung), v = z
                if orientation == "xz":
                    # X-Z-Wand: u = x, v = z, y = const
                    poly_u = poly_x
                    poly_v = poly_z
                    umin, umax = xmin, xmax
                    vmin, vmax = float(poly_z.min()), float(poly_z.max())
                else:  # orientation == "yz"
                    # Y-Z-Wand: u = y, v = z, x = const
                    poly_u = poly_y
                    poly_v = poly_z
                    umin, umax = ymin, ymax
                    vmin, vmax = float(poly_z.min()), float(poly_z.max())
                
                # Erstelle (u,v)-Grid für senkrechte Fläche
                margin = tex_res * 0.5
                u_start = umin - margin
                u_end = umax + margin
                v_start = vmin - margin
                v_end = vmax + margin
                
                num_u = int(np.ceil((u_end - u_start) / tex_res)) + 1
                num_v = int(np.ceil((v_end - v_start) / tex_res)) + 1
                
                us = np.linspace(u_start, u_end, num_u, dtype=float)
                vs = np.linspace(v_start, v_end, num_v, dtype=float)
                if us.size < 2 or vs.size < 2:
                    continue
                
                U, V = np.meshgrid(us, vs, indexing="xy")
                points_uv = np.column_stack((U.ravel(), V.ravel()))
                
                # Maske im Polygon (u,v-Ebene)
                poly_path_uv = Path(np.column_stack((poly_u, poly_v)))
                inside_uv = poly_path_uv.contains_points(points_uv)
                inside_uv = inside_uv.reshape(U.shape)
                
                if not np.any(inside_uv):
                    continue
                
                # 🎯 Hole SPL-Werte aus vertikalen Samples
                # Verwende prepare_vertical_plot_geometry für SPL-Werte
                try:
                    geom_vertical = prepare_vertical_plot_geometry(
                        surface_id,
                        settings,
                        container,
                        default_upscale=1,  # Kein Upscaling, da wir bereits feines Grid haben
                    )
                except Exception:
                    geom_vertical = None
                
                if geom_vertical is None:
                    continue
                
                # Interpoliere SPL-Werte auf (u,v)-Grid
                # geom_vertical enthält plot_u, plot_v, plot_values
                plot_u_geom = np.asarray(geom_vertical.plot_u, dtype=float)
                plot_v_geom = np.asarray(geom_vertical.plot_v, dtype=float)
                plot_values_geom = np.asarray(geom_vertical.plot_values, dtype=float)
                
                # 🎯 Gradient: Bilineare Interpolation für glatte Farbübergänge
                # Color step: Nearest-Neighbor für harte Stufen
                if is_step_mode:
                    spl_flat_uv = nearest_interpolate_func(
                        plot_u_geom,
                        plot_v_geom,
                        plot_values_geom,
                        U.ravel(),
                        V.ravel(),
                    )
                else:
                    spl_flat_uv = bilinear_interpolate_func(
                        plot_u_geom,
                        plot_v_geom,
                        plot_values_geom,
                        U.ravel(),
                        V.ravel(),
                    )
                spl_img_uv = spl_flat_uv.reshape(U.shape)
                
                # Werte clippen
                spl_clipped_uv = np.clip(spl_img_uv, cbar_min, cbar_max)
                if is_step_mode:
                    spl_clipped_uv = quantize_to_steps_func(spl_clipped_uv, cbar_step)
                
                # In Farbe umsetzen
                rgba_uv = base_cmap(norm(spl_clipped_uv))
                
                # Alpha-Maske
                alpha_mask_uv = inside_uv & np.isfinite(spl_clipped_uv)
                rgba_uv[..., 3] = np.where(alpha_mask_uv, 1.0, 0.0)
                
                # Nach uint8 wandeln
                img_rgba_uv = (np.clip(rgba_uv, 0.0, 1.0) * 255).astype(np.uint8)
                
                # 🎯 Erstelle 3D-Grid für senkrechte Fläche
                if orientation == "xz":
                    # X-Z-Wand: X = U, Y = wall_value, Z = V
                    X_3d = U
                    Y_3d = np.full_like(U, wall_value)
                    Z_3d = V
                else:  # orientation == "yz"
                    # Y-Z-Wand: X = wall_value, Y = U, Z = V
                    X_3d = np.full_like(U, wall_value)
                    Y_3d = U
                    Z_3d = V
                
                grid = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
                
                # Textur-Koordinaten für senkrechte Fläche
                try:
                    grid.texture_map_to_plane(inplace=True)
                    t_coords = grid.point_data.get("TCoords")
                except Exception:
                    t_coords = None
                
                # Manuelle Textur-Koordinaten-Berechnung für senkrechte Fläche
                bounds = grid.bounds
                if bounds is not None and len(bounds) >= 6:
                    if orientation == "xz":
                        u_min, u_max = bounds[0], bounds[1]  # X-Bounds
                        v_min, v_max = bounds[4], bounds[5]  # Z-Bounds
                    else:  # yz
                        u_min, u_max = bounds[2], bounds[3]  # Y-Bounds
                        v_min, v_max = bounds[4], bounds[5]  # Z-Bounds
                    
                    u_span = u_max - u_min
                    v_span = v_max - v_min
                    
                    if u_span > 1e-10 and v_span > 1e-10:
                        if orientation == "xz":
                            u_coords = (X_3d - u_min) / u_span
                        else:  # yz
                            u_coords = (Y_3d - u_min) / u_span
                        v_coords_raw = (Z_3d - v_min) / v_span
                        
                        u_coords = np.clip(u_coords, 0.0, 1.0)
                        v_coords_raw = np.clip(v_coords_raw, 0.0, 1.0)
                        v_coords = 1.0 - v_coords_raw  # PyVista-Konvention: v=0 oben
                        
                        # Achsen-Invertierung (wie bei planaren Flächen)
                        invert_x = False
                        invert_y = True
                        swap_axes = False
                        
                        if isinstance(surface_obj, SurfaceDefinition):
                            if hasattr(surface_obj, "invert_texture_x"):
                                invert_x = bool(getattr(surface_obj, "invert_texture_x"))
                            if hasattr(surface_obj, "invert_texture_y"):
                                invert_y = bool(getattr(surface_obj, "invert_texture_y"))
                            if hasattr(surface_obj, "swap_texture_axes"):
                                swap_axes = bool(getattr(surface_obj, "swap_texture_axes"))
                        elif isinstance(surface_obj, dict):
                            if "invert_texture_x" in surface_obj:
                                invert_x = bool(surface_obj["invert_texture_x"])
                            if "invert_texture_y" in surface_obj:
                                invert_y = bool(surface_obj["invert_texture_y"])
                            if "swap_texture_axes" in surface_obj:
                                swap_axes = bool(surface_obj["swap_texture_axes"])
                        
                        img_rgba_final = img_rgba_uv.copy()
                        if invert_x:
                            img_rgba_final = np.fliplr(img_rgba_final)
                            u_coords = 1.0 - u_coords
                        if invert_y:
                            img_rgba_final = np.flipud(img_rgba_final)
                            v_coords = 1.0 - v_coords
                        if swap_axes:
                            img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                            u_coords, v_coords = v_coords.copy(), u_coords.copy()
                        
                        t_coords = np.column_stack((u_coords.ravel(), v_coords.ravel()))
                        grid.point_data["TCoords"] = t_coords
                        img_rgba_uv = img_rgba_final
                    else:
                        # Fallback: Erstelle einfache Textur-Koordinaten
                        if t_coords is None:
                            ny_uv, nx_uv = U.shape
                            u_coords_fallback = np.linspace(0, 1, nx_uv)
                            v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)  # PyVista: v=0 oben
                            U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                            t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                        grid.point_data["TCoords"] = t_coords
                else:
                    # Fallback wenn bounds nicht verfügbar
                    if t_coords is None:
                        ny_uv, nx_uv = U.shape
                        u_coords_fallback = np.linspace(0, 1, nx_uv)
                        v_coords_fallback = 1.0 - np.linspace(0, 1, ny_uv)
                        U_fallback, V_fallback = np.meshgrid(u_coords_fallback, v_coords_fallback, indexing="xy")
                        t_coords = np.column_stack((U_fallback.ravel(), V_fallback.ravel()))
                    grid.point_data["TCoords"] = t_coords
                
                # Erstelle Textur
                tex = pv.Texture(img_rgba_uv)
                # 🎯 Gradient: smooth rendering, Color step: harte Stufen
                tex.interpolate = not is_step_mode
                
                actor_name = f"{surface_name}_tex_{surface_id}"
                old_texture_data = surface_texture_actors.get(surface_id)
                if old_texture_data is not None:
                    old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
                    if old_actor is not None:
                        try:
                            plotter.remove_actor(old_actor)
                        except Exception:
                            pass
                
                actor = plotter.add_mesh(
                    grid,
                    name=actor_name,
                    texture=tex,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
                # Markiere Surface-Actor als nicht-pickable, damit Achsenlinien gepickt werden können
                if hasattr(actor, 'SetPickable'):
                    actor.SetPickable(False)
                
                # 🎯 WICHTIG: Setze _surface_texture_actors auch auf plotter,
                # damit SPL3DOverlayRenderer darauf zugreifen kann
                if not hasattr(plotter, '_surface_texture_actors'):
                    plotter._surface_texture_actors = {}
                plotter._surface_texture_actors[surface_id] = {
                    'actor': actor,
                    'surface_id': surface_id,
                }
                
                # Speichere Metadaten
                metadata = {
                    'actor': actor,
                    'grid': grid,
                    'texture': tex,
                    'grid_bounds': tuple(bounds) if bounds is not None else None,
                    'orientation': orientation,
                    'wall_axis': wall_axis,
                    'wall_value': wall_value,
                    'surface_id': surface_id,
                    'is_vertical': True,
                }
                surface_texture_actors[surface_id] = metadata
                
                # 🎯 WICHTIG: Setze _surface_texture_actors auch auf plotter,
                # damit SPL3DOverlayRenderer darauf zugreifen kann
                if not hasattr(plotter, '_surface_texture_actors'):
                    plotter._surface_texture_actors = {}
                plotter._surface_texture_actors[surface_id] = {
                    'actor': actor,
                    'surface_id': surface_id,
                }
                
                continue  # Überspringe normalen planaren Pfad
            
            # 🎯 NORMALER PFAD FÜR PLANARE FLÄCHEN (horizontal oder geneigt)
            # 🎯 FIX: Reduzierter Margin für schärfere Kanten
            # Reduziere Margin von 2.0 auf 0.5 für weniger Punkte außerhalb des Polygons
            # Dies reduziert die "Zacken" am Rand, da weniger Zellen außerhalb gerendert werden
            margin = tex_res * 0.5  # Nur halber Pixel-Abstand als Margin (vorher: 2.0)
            
            # Erstelle Grid mit reduziertem Margin (für minimale Interpolation an Rändern)
            x_start = xmin - margin
            x_end = xmax + margin
            y_start = ymin - margin
            y_end = ymax + margin
            
            # Berechne Anzahl Punkte (inklusive Endpunkte)
            num_x = int(np.ceil((x_end - x_start) / tex_res)) + 1
            num_y = int(np.ceil((y_end - y_start) / tex_res)) + 1
            
            # Erstelle Grid mit reduziertem Margin
            xs = np.linspace(x_start, x_end, num_x, dtype=float)
            ys = np.linspace(y_start, y_end, num_y, dtype=float)
            if xs.size < 2 or ys.size < 2:
                continue

            # Erstelle meshgrid mit indexing="xy" für korrekte Zuordnung
            # Bei indexing="xy": X[j, i] = xs[i], Y[j, i] = ys[j]
            # Shape ist (len(ys), len(xs)) = (ny, nx)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            points_2d = np.column_stack((X.ravel(), Y.ravel()))

            # Maske im Polygon
            poly_path = Path(np.column_stack((poly_x, poly_y)))
            inside = poly_path.contains_points(points_2d)
            inside = inside.reshape(X.shape)

            if not np.any(inside):
                continue

            # 🎯 Gradient: Bilineare Interpolation für glatte Farbübergänge
            # Color step: Nearest-Neighbor für harte Stufen
            if is_step_mode:
                spl_flat = nearest_interpolate_func(
                    source_x,
                    source_y,
                    values,
                    X.ravel(),
                    Y.ravel(),
                )
            else:
                spl_flat = bilinear_interpolate_func(
                    source_x,
                    source_y,
                    values,
                    X.ravel(),
                    Y.ravel(),
                )
            spl_img = spl_flat.reshape(X.shape)

            # Werte clippen
            spl_clipped = np.clip(spl_img, cbar_min, cbar_max)
            # Optional in Stufen quantisieren (Color step)
            if is_step_mode:
                spl_clipped = quantize_to_steps_func(spl_clipped, cbar_step)

            # In Farbe umsetzen
            rgba = base_cmap(norm(spl_clipped))  # float [0,1], shape (H,W,4)

            # 🎯 Verbesserte Alpha-Maske: Schärfere Kanten durch strikte Polygon-Prüfung
            # Nur Pixel, die definitiv im Polygon liegen, sind sichtbar
            alpha_mask = inside & np.isfinite(spl_clipped)
            rgba[..., 3] = np.where(alpha_mask, 1.0, 0.0)

            # Nach uint8 wandeln
            img_rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

            # 🎯 Berechne Z-Koordinaten basierend auf Planmodell für geneigte Flächen
            mode = plane_model.get("mode", "constant")
            if mode == "constant":
                # Konstante Höhe
                Z = np.full_like(X, float(plane_model.get("base", 0.0)))
            elif mode == "x":
                # Lineare Steigung entlang X-Achse: Z = slope * X + intercept
                slope = float(plane_model.get("slope", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope * X + intercept
            elif mode == "y":
                # Lineare Steigung entlang Y-Achse: Z = slope * Y + intercept
                slope = float(plane_model.get("slope", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope * Y + intercept
            else:  # mode == "xy" (allgemeine Ebene)
                # Allgemeine Ebene: Z = slope_x * X + slope_y * Y + intercept
                slope_x = float(plane_model.get("slope_x", plane_model.get("slope", 0.0)))
                slope_y = float(plane_model.get("slope_y", 0.0))
                intercept = float(plane_model.get("intercept", 0.0))
                Z = slope_x * X + slope_y * Y + intercept
            
            # Erstelle Grid mit korrekten Z-Koordinaten (für horizontale und geneigte Flächen)
            grid = pv.StructuredGrid(X, Y, Z)

            # Verwende PyVista's texture_map_to_plane() für automatische Textur-Koordinaten
            # Dies stellt sicher, dass PyVista die Textur-Koordinaten erkennt
            try:
                grid.texture_map_to_plane(inplace=True)
                # Hole die generierten Textur-Koordinaten
                t_coords = grid.point_data.get("TCoords")
            except Exception:
                t_coords = None

            # Texture-Koordinaten manuell anpassen (basierend auf Welt-Koordinaten für korrekte Orientierung)
            bounds = None
            try:
                bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                xmin_grid, xmax_grid = bounds[0], bounds[1]
                ymin_grid, ymax_grid = bounds[2], bounds[3]
                
                # 🎯 KORREKTE Textur-Koordinaten-Berechnung:
                # Das Bild img_rgba hat Shape (H, W, 4) wo:
                #   H = len(ys) = Anzahl Y-Punkte (von ymin nach ymax)
                #   W = len(xs) = Anzahl X-Punkte (von xmin nach xmax)
                # 
                # Im Bild-Array: img_rgba[j, i, :] entspricht:
                #   i = Index in xs (0 = xmin, W-1 = xmax)
                #   j = Index in ys (0 = ymin, H-1 = ymax)
                #
                # Textur-Koordinaten (u, v):
                #   u = X-Position im Bild (0 = links/xmin, 1 = rechts/xmax)
                #   v = Y-Position im Bild (0 = unten/ymin, 1 = oben/ymax)
                # 
                # WICHTIG: In Bild-Arrays ist j=0 oben und j=H-1 unten (invertiert zu v)
                # Aber bei meshgrid mit indexing="xy": Y[j, i] = ys[j], X[j, i] = xs[i]
                #   j=0 entspricht ymin (unten), j=H-1 entspricht ymax (oben)
                #   i=0 entspricht xmin (links), i=W-1 entspricht xmax (rechts)
                #
                # Für Textur-Koordinaten müssen wir:
                #   u = (X - xmin) / (xmax - xmin)  -> 0 (xmin/links) bis 1 (xmax/rechts)
                #   v = (Y - ymin) / (ymax - ymin)  -> 0 (ymin/unten) bis 1 (ymax/oben)
                #
                # Das Bild wurde erstellt mit: spl_img[j, i] = spl_flat[j*W + i]
                #   wobei X[j, i] = xs[i] und Y[j, i] = ys[j]
                # Also: img_rgba[j, i] entspricht Welt-Koordinate (xs[i], ys[j])
                #
                # Textur-Koordinate für Punkt (X[j,i], Y[j,i]) sollte sein:
                #   u = (i) / (W-1) = (xs[i] - xmin) / (xmax - xmin)  (bei gleichmäßiger Verteilung)
                #   v = (j) / (H-1) = (ys[j] - ymin) / (ymax - ymin)  (bei gleichmäßiger Verteilung)
                
                # Berechne Textur-Koordinaten direkt aus Welt-Koordinaten
                x_span = xmax_grid - xmin_grid
                y_span = ymax_grid - ymin_grid
                
                if x_span > 1e-10 and y_span > 1e-10:
                    # Überschreibe die automatisch generierten Textur-Koordinaten mit manuell berechneten
                    # für exakte Zuordnung zwischen Welt-Koordinaten und Bild-Pixeln
                    # Normalisiere Welt-Koordinaten auf [0, 1]
                    # u: 0 = xmin (links), 1 = xmax (rechts)
                    # v: 0 = ymin (unten), 1 = ymax (oben)
                    u_coords = (X - xmin_grid) / x_span
                    v_coords_raw = (Y - ymin_grid) / y_span
                    
                    # Stelle sicher, dass u und v im Bereich [0, 1] sind (aufgrund numerischer Ungenauigkeiten)
                    u_coords = np.clip(u_coords, 0.0, 1.0)
                    v_coords_raw = np.clip(v_coords_raw, 0.0, 1.0)
                    
                    # 🎯 WICHTIG: Spiegle v-Koordinate für PyVista-Konvention
                    # PyVista erwartet: v=0 oben, v=1 unten (Standard-Textur-Koordinaten)
                    # Wir haben: v_raw=0 unten (ymin), v_raw=1 oben (ymax)
                    # Also: v = 1 - v_raw
                    v_coords = 1.0 - v_coords_raw
                    
                    # 🎯 ACHSEN-INVERTIERUNG: Konfigurierbar pro Surface
                    # Prüfe, ob X- und/oder Y-Achsen invertiert werden sollen
                    # Defaults (werden verwendet, wenn Attribute nicht gesetzt sind)
                    invert_x = False
                    invert_y = True
                    swap_axes = False 
                    
                    if isinstance(surface_obj, SurfaceDefinition):
                        # Nur überschreiben, wenn Attribut existiert
                        if hasattr(surface_obj, "invert_texture_x"):
                            invert_x = bool(getattr(surface_obj, "invert_texture_x"))
                        if hasattr(surface_obj, "invert_texture_y"):
                            invert_y = bool(getattr(surface_obj, "invert_texture_y"))
                        if hasattr(surface_obj, "swap_texture_axes"):
                            swap_axes = bool(getattr(surface_obj, "swap_texture_axes"))
                    elif isinstance(surface_obj, dict):
                        # Nur überschreiben, wenn Key existiert
                        if "invert_texture_x" in surface_obj:
                            invert_x = bool(surface_obj["invert_texture_x"])
                        if "invert_texture_y" in surface_obj:
                            invert_y = bool(surface_obj["invert_texture_y"])
                        if "swap_texture_axes" in surface_obj:
                            swap_axes = bool(surface_obj["swap_texture_axes"])
                    
                    # 🎯 Wende Bild-Spiegelung an (wenn nötig), BEVOR wir die Textur erstellen
                    # Wenn wir das Bild spiegeln, müssen wir die Texturkoordinaten NICHT zusätzlich invertieren
                    img_rgba_final = img_rgba.copy()
                    if invert_x:
                        # Spiegle das Bild horizontal (links <-> rechts)
                        img_rgba_final = np.fliplr(img_rgba_final)
                        # Invertiere U-Koordinaten, damit sie zum gespiegelten Bild passen
                        u_coords = 1.0 - u_coords
                    if invert_y:
                        # Spiegle das Bild vertikal (oben <-> unten)
                        img_rgba_final = np.flipud(img_rgba_final)
                        # Invertiere V-Koordinaten, damit sie zum gespiegelten Bild passen
                        v_coords = 1.0 - v_coords
                    if swap_axes:
                        # Transponiere das Bild (X <-> Y)
                        img_rgba_final = np.transpose(img_rgba_final, (1, 0, 2))
                        # Vertausche U und V Koordinaten
                        u_coords, v_coords = v_coords.copy(), u_coords.copy()
                    
                    # Textur-Koordinaten als (N, 2) Array: [u, v]
                    # u = X-Koordinate, v = Y-Koordinate
                    t_coords = np.column_stack((u_coords.ravel(), v_coords.ravel()))
                    grid.point_data["TCoords"] = t_coords
                    
                    # Verwende das gespiegelte Bild
                    img_rgba = img_rgba_final
                else:
                    # Fallback für degenerierte Fälle
                    if t_coords is None:
                        ny, nx = X.shape
                        u_coords = np.linspace(0, 1, nx)
                        v_coords_raw = np.linspace(0, 1, ny)
                        # Spiegle v für PyVista-Konvention (v=0 oben, v=1 unten)
                        v_coords = 1.0 - v_coords_raw
                        U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
                        t_coords = np.column_stack((U.ravel(), V.ravel()))
                    grid.point_data["TCoords"] = t_coords
                
            except Exception:
                continue

            # Prüfe ob bounds und t_coords gesetzt wurden
            if bounds is None:
                bounds = grid.bounds
            if t_coords is None:
                # Fallback: Hole TCoords vom Grid falls verfügbar
                t_coords = grid.point_data.get("TCoords")
            
            # Validiere Textur-Koordinaten vor dem Rendern
            if "TCoords" not in grid.point_data:
                continue
            
            # Prüfe ob TCoords die richtige Shape haben
            tcoords_data = grid.point_data["TCoords"]
            if tcoords_data is None or tcoords_data.shape[0] != grid.n_points:
                continue

            tex = pv.Texture(img_rgba)
            # 🎯 Gradient: smooth rendering, Color step: harte Stufen
            tex.interpolate = not is_step_mode

            actor_name = f"{surface_name}_tex_{surface_id}"
            # Alten Actor ggf. entfernen
            old_actor = None
            old_texture_data = surface_texture_actors.get(surface_id)
            if old_texture_data is not None:
                old_actor = old_texture_data.get('actor') if isinstance(old_texture_data, dict) else old_texture_data
            if old_actor is not None:
                try:
                    plotter.remove_actor(old_actor)
                except Exception:
                    pass

            actor = plotter.add_mesh(
                grid,
                name=actor_name,
                texture=tex,
                show_scalar_bar=False,
                reset_camera=False,
            )
            # Markiere Surface-Actor als nicht-pickable, damit Achsenlinien gepickt werden können
            if hasattr(actor, 'SetPickable'):
                actor.SetPickable(False)
            
            # 🎯 Speichere Metadaten zusammen mit dem Actor
            # Diese können für Click-Handling, Koordinaten-Transformation, etc. verwendet werden
            metadata = {
                'actor': actor,
                'grid': grid,
                'texture': tex,
                'grid_bounds': tuple(bounds) if bounds is not None else None,  # (xmin, xmax, ymin, ymax, zmin, zmax)
                'world_coords_x': xs.copy(),  # 1D-Array der X-Koordinaten in Metern
                'world_coords_y': ys.copy(),  # 1D-Array der Y-Koordinaten in Metern
                'world_coords_grid_x': X.copy(),  # 2D-Meshgrid der X-Koordinaten
                'world_coords_grid_y': Y.copy(),  # 2D-Meshgrid der Y-Koordinaten
                'texture_resolution': tex_res,  # Auflösung der Textur in Metern
                'texture_size': (ys.size, xs.size),  # (H, W) in Pixeln
                'image_shape': img_rgba.shape,  # (H, W, 4)
                'polygon_bounds': {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                },
                'polygon_points': points,  # Original Polygon-Punkte
                't_coords': t_coords.copy() if t_coords is not None else None,  # Textur-Koordinaten
                'surface_id': surface_id,
            }
            
            surface_texture_actors[surface_id] = metadata
            
            # 🎯 WICHTIG: Setze _surface_texture_actors auch auf plotter,
            # damit SPL3DOverlayRenderer darauf zugreifen kann
            if not hasattr(plotter, '_surface_texture_actors'):
                plotter._surface_texture_actors = {}
            plotter._surface_texture_actors[surface_id] = {
                'actor': actor,
                'surface_id': surface_id,
            }
        except Exception:
            continue


def clear_vertical_spl_surfaces(plotter: Any, vertical_surface_meshes: dict) -> None:
    """Löscht alle vertikalen SPL-Surface-Meshes.
    
    Args:
        plotter: PyVista Plotter
        vertical_surface_meshes: Dictionary mit vertikalen Surface-Meshes
    """
    for surface_id, mesh_info in list(vertical_surface_meshes.items()):
        try:
            actor = mesh_info.get('actor')
            if actor is not None:
                plotter.remove_actor(actor)
        except Exception:
            pass
    vertical_surface_meshes.clear()


def get_vertical_color_limits(settings: Any) -> tuple[float, float]:
    """Berechnet die Color-Limits für vertikale Surfaces.
    
    Args:
        settings: Settings-Objekt
        
    Returns:
        Tuple mit (min, max)
    """
    try:
        rng = settings.colorbar_range
        return (float(rng['min']), float(rng['max']))
    except Exception:
        return (0.0, 100.0)


def update_vertical_spl_surfaces(
    plotter: Any,
    pv_module: Any,
    settings: Any,
    container: Any,
    vertical_surface_meshes: dict,
    clear_func: Callable,
    quantize_to_steps_func: Callable,
    get_vertical_color_limits_func: Callable,
    upscale_factor: int = 3,
) -> None:
    """
    Zeichnet / aktualisiert SPL-Flächen für senkrechte Surfaces auf Basis von
    calculation_spl['surface_samples'] und calculation_spl['surface_fields'].
    
    Vollständige Implementierung aus PlotSPL3D.py Zeilen 2948-3093.
    
    Hinweis:
    - Für horizontale Flächen verwenden wir ausschließlich den Texture-Pfad.
    - Für senkrechte / stark geneigte Flächen rendern wir hier explizite Meshes
      (vertical_spl_<surface_id>), damit sie im 3D-Plot separat anwählbar sind.
    
    Args:
        plotter: PyVista Plotter
        pv_module: PyVista Modul
        settings: Settings-Objekt
        container: Container-Objekt
        vertical_surface_meshes: Dictionary mit vertikalen Surface-Meshes
        clear_func: Funktion zum Löschen
        quantize_to_steps_func: Funktion zum Quantisieren in Stufen
        get_vertical_color_limits_func: Funktion zum Holen der Color-Limits
        upscale_factor: Upscale-Faktor für Plot-Geometrie
    """
    if container is None or not hasattr(container, "calculation_spl"):
        clear_func()
        return

    calc_spl = getattr(container, "calculation_spl", {}) or {}
    sample_payloads = calc_spl.get("surface_samples")
    if not isinstance(sample_payloads, list):
        clear_func()
        return

    # Aktueller Surface-Status (enabled/hidden) aus den Settings
    surface_definitions = getattr(settings, "surface_definitions", {})
    if not isinstance(surface_definitions, dict):
        surface_definitions = {}

    # Vertikale Surfaces analog zu prepare_plot_geometry behandeln:
    # lokales (u,v)-Raster + strukturiertes Mesh über build_vertical_surface_mesh.
    new_vertical_meshes: dict[str, Any] = {}

    # Aktuellen Colorization-Mode verwenden (wie für die Hauptfläche).
    colorization_mode = getattr(settings, "colorization_mode", "Gradient")
    if colorization_mode not in {"Color step", "Gradient"}:
        colorization_mode = "Gradient"
    try:
        cbar_range = getattr(settings, "colorbar_range", {})
        cbar_step = float(cbar_range.get("step", 0.0))
    except Exception:
        cbar_step = 0.0
    is_step_mode = colorization_mode == "Color step" and cbar_step > 0
    
    for payload in sample_payloads:
        # Nur Payloads verarbeiten, die explizit als "vertical" markiert sind.
        kind = payload.get("kind", "planar")
        if kind != "vertical":
            continue

        surface_id = payload.get("surface_id")
        if surface_id is None:
            continue

        # Nur Surfaces zeichnen, die aktuell enabled und nicht hidden sind
        surf_def = surface_definitions.get(surface_id)
        if surf_def is None:
            continue
        if hasattr(surf_def, "to_dict"):
            surf_data = surf_def.to_dict()
        elif isinstance(surf_def, dict):
            surf_data = surf_def
        else:
            surf_data = {
                "enabled": getattr(surf_def, "enabled", False),
                "hidden": getattr(surf_def, "hidden", False),
                "points": getattr(surf_def, "points", []),
            }
        if not surf_data.get("enabled", False) or surf_data.get("hidden", False):
            continue

        # Lokale vertikale Plot-Geometrie aufbauen (u,v-Grid, SPL-Werte, Maske)
        try:
            geom = prepare_vertical_plot_geometry(
                surface_id,
                settings,
                container,
                default_upscale=upscale_factor,
            )
        except Exception:
            geom = None

        if geom is None:
            continue

        # Strukturiertes Mesh in Weltkoordinaten erstellen
        try:
            grid = build_vertical_surface_mesh(geom, pv_module=pv_module)
        except Exception:
            continue

        # Color step: Werte in diskrete Stufen quantisieren, analog zur Hauptfläche.
        if is_step_mode and "plot_scalars" in grid.array_names:
            try:
                vals = np.asarray(grid["plot_scalars"], dtype=float)
                grid["plot_scalars"] = quantize_to_steps_func(vals, cbar_step)
            except Exception:
                pass

        actor_name = f"vertical_spl_{surface_id}"
        # Entferne ggf. alten Actor
        try:
            if actor_name in plotter.renderer.actors:
                plotter.remove_actor(actor_name)
        except Exception:
            pass

        # Farbschema und CLim an das Haupt-SPL anlehnen
        cbar_min, cbar_max = get_vertical_color_limits_func(settings)
        try:
            actor = plotter.add_mesh(
                grid,
                name=actor_name,
                scalars="plot_scalars",
                cmap="jet",
                clim=(cbar_min, cbar_max),
                # Gradient: weiche Darstellung, Color step: harte Stufen.
                smooth_shading=not is_step_mode,
                show_scalar_bar=False,
                reset_camera=False,
                interpolate_before_map=not is_step_mode,
            )
            # Stelle sicher, dass senkrechte Flächen pickable sind
            if actor and hasattr(actor, 'SetPickable'):
                actor.SetPickable(True)
            # Im Color-Step-Modus explizit flache Interpolation erzwingen,
            # damit die Stufen wie bei der horizontalen Fläche erscheinen.
            if is_step_mode and hasattr(actor, "prop") and actor.prop is not None:
                try:
                    actor.prop.interpolation = "flat"
                except Exception:  # noqa: BLE001
                    pass
            new_vertical_meshes[actor_name] = actor
        except Exception:
            continue

    # Alte Actors entfernen, die nicht mehr gebraucht werden
    for old_name in list(vertical_surface_meshes.keys()):
        if old_name not in new_vertical_meshes:
            try:
                if old_name in plotter.renderer.actors:
                    plotter.remove_actor(old_name)
            except Exception:
                pass

    vertical_surface_meshes.clear()
    vertical_surface_meshes.update(new_vertical_meshes)

