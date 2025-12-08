#!/usr/bin/env python3
"""
Debug-Skript zur Analyse der Triangulation bei schrägen Flächen.

Prüft:
1. Korrekte Triangulation bei schrägen Flächen (xz_slanted, yz_slanted)
2. Korrekten Abschluss an Surface-Kanten
3. Gleichmäßige Auflösung bei allen möglichen Flächen
"""

import numpy as np
import sys
import os

# Füge LFO-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LFO'))

try:
    import pyvista as pv
except ImportError:
    pv = None
    print("⚠️ PyVista nicht verfügbar - Visualisierung deaktiviert")

from scipy.spatial import cKDTree
from scipy.interpolate import griddata


def analyze_triangulation_quality(points_3d, faces, polygon_points, vertical_orientation):
    """
    Analysiert die Qualität der Triangulation.
    
    Args:
        points_3d: (n, 3) Array der 3D-Punkte
        faces: PyVista Faces-Array
        polygon_points: Liste der Polygon-Punkte
        vertical_orientation: "xz_slanted", "yz_slanted", "xz", "yz", oder None
    
    Returns:
        dict mit Analyse-Ergebnissen
    """
    results = {
        'n_points': len(points_3d),
        'n_cells': len(faces) // 4 if len(faces) > 0 else 0,  # PyVista: 3 + n_points pro Face
        'edge_quality': {},
        'boundary_quality': {},
        'resolution_quality': {},
    }
    
    # 1. PRÜFE KANTENQUALITÄT
    # Berechne Kantenlängen aller Dreiecke
    edge_lengths = []
    for i in range(0, len(faces), 4):
        if len(faces) < i + 4:
            break
        n_points = faces[i]
        if n_points != 3:
            continue
        idx0, idx1, idx2 = faces[i+1], faces[i+2], faces[i+3]
        p0, p1, p2 = points_3d[idx0], points_3d[idx1], points_3d[idx2]
        
        # Berechne 3D-Kantenlängen
        e1 = np.linalg.norm(p1 - p0)
        e2 = np.linalg.norm(p2 - p1)
        e3 = np.linalg.norm(p0 - p2)
        edge_lengths.extend([e1, e2, e3])
    
    if edge_lengths:
        edge_lengths = np.array(edge_lengths)
        results['edge_quality'] = {
            'min': float(np.min(edge_lengths)),
            'max': float(np.max(edge_lengths)),
            'mean': float(np.mean(edge_lengths)),
            'std': float(np.std(edge_lengths)),
            'cv': float(np.std(edge_lengths) / np.mean(edge_lengths)) if np.mean(edge_lengths) > 0 else 0.0,  # Variationskoeffizient
        }
    
    # 2. PRÜFE RANDQUALITÄT
    # Identifiziere Rand-Kanten (Kanten, die nur in einem Dreieck vorkommen)
    from collections import defaultdict
    edge_count = defaultdict(int)
    
    for i in range(0, len(faces), 4):
        if len(faces) < i + 4:
            break
        n_points = faces[i]
        if n_points != 3:
            continue
        idx0, idx1, idx2 = faces[i+1], faces[i+2], faces[i+3]
        # Sortiere Kanten für konsistente Darstellung
        edges = [
            tuple(sorted([idx0, idx1])),
            tuple(sorted([idx1, idx2])),
            tuple(sorted([idx2, idx0])),
        ]
        for edge in edges:
            edge_count[edge] += 1
    
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_lengths = []
    for edge in boundary_edges:
        p0, p1 = points_3d[edge[0]], points_3d[edge[1]]
        boundary_lengths.append(np.linalg.norm(p1 - p0))
    
    if boundary_lengths:
        boundary_lengths = np.array(boundary_lengths)
        results['boundary_quality'] = {
            'n_boundary_edges': len(boundary_edges),
            'min_length': float(np.min(boundary_lengths)),
            'max_length': float(np.max(boundary_lengths)),
            'mean_length': float(np.mean(boundary_lengths)),
            'std_length': float(np.std(boundary_lengths)),
        }
    
    # 3. PRÜFE AUFLÖSUNG
    # Berechne Punktdichte in verschiedenen Bereichen
    if vertical_orientation in ("xz_slanted", "xz"):
        # Projiziere auf XZ-Ebene
        proj_points = points_3d[:, [0, 2]]
    elif vertical_orientation in ("yz_slanted", "yz"):
        # Projiziere auf YZ-Ebene
        proj_points = points_3d[:, [1, 2]]
    else:
        # Projiziere auf XY-Ebene
        proj_points = points_3d[:, [0, 1]]
    
    # Berechne nächste Nachbarn für jeden Punkt
    if len(proj_points) > 1:
        tree = cKDTree(proj_points)
        distances, _ = tree.query(proj_points, k=2)  # k=2: Punkt selbst + nächster Nachbar
        nn_distances = distances[:, 1]  # Ignoriere Abstand zu sich selbst
        
        results['resolution_quality'] = {
            'min_nn_distance': float(np.min(nn_distances)),
            'max_nn_distance': float(np.max(nn_distances)),
            'mean_nn_distance': float(np.mean(nn_distances)),
            'std_nn_distance': float(np.std(nn_distances)),
            'cv_nn_distance': float(np.std(nn_distances) / np.mean(nn_distances)) if np.mean(nn_distances) > 0 else 0.0,
        }
    
    # 4. PRÜFE POLYGON-ABWEICHUNG
    if polygon_points and len(polygon_points) >= 3:
        # Projiziere Polygon-Punkte in die richtige Ebene
        if vertical_orientation in ("xz_slanted", "xz"):
            poly_proj = np.array([[p.get('x', 0.0), p.get('z', 0.0)] for p in polygon_points])
        elif vertical_orientation in ("yz_slanted", "yz"):
            poly_proj = np.array([[p.get('y', 0.0), p.get('z', 0.0)] for p in polygon_points])
        else:
            poly_proj = np.array([[p.get('x', 0.0), p.get('y', 0.0)] for p in polygon_points])
        
        # Finde Randpunkte, die am weitesten vom Polygon entfernt sind
        boundary_distances = []
        for edge in boundary_edges:
            p0, p1 = points_3d[edge[0]], points_3d[edge[1]]
            # Projiziere auf 2D
            if vertical_orientation in ("xz_slanted", "xz"):
                p0_proj = np.array([p0[0], p0[2]])
                p1_proj = np.array([p1[0], p1[2]])
            elif vertical_orientation in ("yz_slanted", "yz"):
                p0_proj = np.array([p0[1], p0[2]])
                p1_proj = np.array([p1[1], p1[2]])
            else:
                p0_proj = np.array([p0[0], p0[1]])
                p1_proj = np.array([p1[0], p1[1]])
            
            # Berechne minimale Distanz zur Polygon-Kante
            min_dist = float('inf')
            for j in range(len(poly_proj)):
                poly_p0 = poly_proj[j]
                poly_p1 = poly_proj[(j + 1) % len(poly_proj)]
                
                # Distanz von Punkt zu Segment
                v = poly_p1 - poly_p0
                w0 = p0_proj - poly_p0
                w1 = p1_proj - poly_p0
                
                if np.dot(v, v) > 1e-12:
                    t0 = np.clip(np.dot(w0, v) / np.dot(v, v), 0, 1)
                    t1 = np.clip(np.dot(w1, v) / np.dot(v, v), 0, 1)
                    proj0 = poly_p0 + t0 * v
                    proj1 = poly_p0 + t1 * v
                    dist0 = np.linalg.norm(p0_proj - proj0)
                    dist1 = np.linalg.norm(p1_proj - proj1)
                    min_dist = min(min_dist, dist0, dist1)
            
            if min_dist < float('inf'):
                boundary_distances.append(min_dist)
        
        if boundary_distances:
            boundary_distances = np.array(boundary_distances)
            results['boundary_quality']['max_polygon_deviation'] = float(np.max(boundary_distances))
            results['boundary_quality']['mean_polygon_deviation'] = float(np.mean(boundary_distances))
    
    return results


def test_triangulation_xz_slanted():
    """
    Testet die Triangulation für eine schräge XZ-Fläche.
    """
    print("=" * 80)
    print("TEST: Triangulation für xz_slanted Fläche")
    print("=" * 80)
    
    # Erstelle Test-Polygon: Schräge Fläche von (0,5,5) über (-10,10,15) bis (0,20,25)
    polygon_points = [
        {'x': 0.0, 'y': 5.0, 'z': 5.0},
        {'x': -10.0, 'y': 10.0, 'z': 15.0},
        {'x': 0.0, 'y': 20.0, 'z': 25.0},
        {'x': 5.0, 'y': 15.0, 'z': 20.0},
    ]
    
    # Erstelle Grid-Punkte in XZ-Ebene
    x_coords = np.linspace(-10, 5, 20)
    z_coords = np.linspace(5, 25, 20)
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords)
    
    # Berechne Y-Koordinaten über Ebenen-Fit
    poly_x = np.array([p['x'] for p in polygon_points])
    poly_z = np.array([p['z'] for p in polygon_points])
    poly_y = np.array([p['y'] for p in polygon_points])
    
    A = np.column_stack([poly_x, poly_z, np.ones(len(poly_x))])
    coeffs, _, _, _ = np.linalg.lstsq(A, poly_y, rcond=None)
    a, b, c = coeffs
    
    Y_grid = a * X_grid + b * Z_grid + c
    
    # Erstelle kombinierte Punkte (Grid + Randpunkte)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])
    
    # Erstelle Randpunkte entlang der Polygon-Kanten
    edge_points = []
    for i in range(len(polygon_points)):
        p0 = polygon_points[i]
        p1 = polygon_points[(i + 1) % len(polygon_points)]
        
        # Erzeuge Punkte entlang der 3D-Kante
        n_edge = 50
        for j in range(n_edge):
            t = j / (n_edge - 1) if n_edge > 1 else 0.0
            x = p0['x'] + t * (p1['x'] - p0['x'])
            y = p0['y'] + t * (p1['y'] - p0['y'])
            z = p0['z'] + t * (p1['z'] - p0['z'])
            edge_points.append([x, y, z])
    
    edge_points = np.array(edge_points)
    combined_points = np.vstack([grid_points, edge_points])
    
    print(f"\n📊 Test-Daten:")
    print(f"  └─ Grid-Punkte: {len(grid_points)}")
    print(f"  └─ Randpunkte: {len(edge_points)}")
    print(f"  └─ Kombiniert: {len(combined_points)}")
    print(f"  └─ Ebenen-Fit: y = {a:.4f}*x + {b:.4f}*z + {c:.4f}")
    
    # Simuliere Delaunay-Triangulation in XZ-Ebene
    if pv is None:
        print("\n⚠️ PyVista nicht verfügbar - kann Triangulation nicht testen")
        return
    
    # Projiziere auf XZ-Ebene für Triangulation
    x_coords_2d = combined_points[:, 0]
    z_coords_2d = combined_points[:, 2]
    y_coords_orig = combined_points[:, 1]
    
    points_2d_3d = np.column_stack([
        x_coords_2d,
        z_coords_2d,
        np.zeros(len(x_coords_2d)),
    ])
    
    temp_mesh = pv.PolyData(points_2d_3d)
    temp_mesh = temp_mesh.delaunay_2d(alpha=0.0, tol=0.0)
    
    print(f"\n🔺 Triangulation:")
    print(f"  └─ Delaunay-Punkte: {temp_mesh.n_points}")
    print(f"  └─ Delaunay-Zellen: {temp_mesh.n_cells}")
    
    # Stelle 3D-Koordinaten wieder her
    x_new = temp_mesh.points[:, 0]
    z_new = temp_mesh.points[:, 1]
    y_pred = a * x_new + b * z_new + c
    y_min, y_max = float(poly_y.min()), float(poly_y.max())
    y_values = np.clip(y_pred, y_min, y_max)
    
    points_3d = np.column_stack([
        x_new,
        y_values,
        z_new,
    ])
    
    # Erstelle finales Mesh
    final_mesh = pv.PolyData(points_3d, temp_mesh.faces)
    
    print(f"\n✅ Wiederhergestellte 3D-Koordinaten:")
    print(f"  └─ X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  └─ Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] (interpoliert)")
    print(f"  └─ Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    
    # Analysiere Qualität
    results = analyze_triangulation_quality(
        points_3d, final_mesh.faces, polygon_points, "xz_slanted"
    )
    
    print(f"\n📈 Qualitäts-Analyse:")
    print(f"  └─ Punkte: {results['n_points']}")
    print(f"  └─ Zellen: {results['n_cells']}")
    
    if results['edge_quality']:
        eq = results['edge_quality']
        print(f"\n  🔹 Kantenqualität:")
        print(f"     └─ Länge: min={eq['min']:.4f}m, max={eq['max']:.4f}m, mean={eq['mean']:.4f}m")
        print(f"     └─ Variationskoeffizient: {eq['cv']:.4f} ({'✅ gut' if eq['cv'] < 0.5 else '⚠️ ungleichmäßig'})")
    
    if results['boundary_quality']:
        bq = results['boundary_quality']
        print(f"\n  🔹 Randqualität:")
        print(f"     └─ Rand-Kanten: {bq.get('n_boundary_edges', 0)}")
        if 'mean_length' in bq:
            print(f"     └─ Kantenlänge: min={bq['min_length']:.4f}m, max={bq['max_length']:.4f}m, mean={bq['mean_length']:.4f}m")
        if 'max_polygon_deviation' in bq:
            print(f"     └─ Max. Polygon-Abweichung: {bq['max_polygon_deviation']:.4f}m")
            print(f"     └─ Mean Polygon-Abweichung: {bq['mean_polygon_deviation']:.4f}m")
            if bq['max_polygon_deviation'] > 0.1:
                print(f"     ⚠️ GROSSE Abweichung vom Polygon!")
    
    if results['resolution_quality']:
        rq = results['resolution_quality']
        print(f"\n  🔹 Auflösung:")
        print(f"     └─ NN-Distanz: min={rq['min_nn_distance']:.4f}m, max={rq['max_nn_distance']:.4f}m, mean={rq['mean_nn_distance']:.4f}m")
        print(f"     └─ Variationskoeffizient: {rq['cv_nn_distance']:.4f} ({'✅ gleichmäßig' if rq['cv_nn_distance'] < 0.3 else '⚠️ ungleichmäßig'})")
    
    return results


def test_filtering_logic():
    """
    Testet die Filterungslogik für schräge Flächen.
    """
    print("\n" + "=" * 80)
    print("TEST: Filterungslogik für schräge Flächen")
    print("=" * 80)
    
    # Erstelle Test-Polygon
    polygon_points = [
        {'x': -17.321, 'y': 5.0, 'z': 5.0},
        {'x': 0.437, 'y': 20.0, 'z': 25.0},
        {'x': 0.0, 'y': 15.0, 'z': 20.0},
    ]
    
    # Erstelle Test-Mesh mit Punkten innerhalb und außerhalb
    n_points = 1000
    np.random.seed(42)
    
    # Punkte innerhalb Polygon
    points_inside = []
    for _ in range(n_points // 2):
        x = np.random.uniform(-17.0, 0.4)
        z = np.random.uniform(5.0, 25.0)
        # Y aus Ebenen-Fit
        y = 0.5774 * x - 0.5126 * z + 22.5631
        y = np.clip(y, 5.0, 20.0)
        points_inside.append([x, y, z])
    
    # Punkte außerhalb Polygon (leicht außerhalb)
    points_outside = []
    for _ in range(n_points // 2):
        x = np.random.uniform(0.5, 1.0)  # Außerhalb X-Bereich
        z = np.random.uniform(5.0, 25.0)
        y = 0.5774 * x - 0.5126 * z + 22.5631
        y = np.clip(y, 5.0, 20.0)
        points_outside.append([x, y, z])
    
    all_points = np.array(points_inside + points_outside)
    
    if pv is None:
        print("⚠️ PyVista nicht verfügbar - kann Filterung nicht testen")
        return
    
    # Erstelle Mesh
    mesh = pv.PolyData(all_points)
    mesh = mesh.delaunay_2d(alpha=0.0, tol=0.0)
    
    print(f"\n📊 Test-Mesh:")
    print(f"  └─ Punkte: {mesh.n_points}")
    print(f"  └─ Zellen: {mesh.n_cells}")
    
    # Simuliere Filterung
    try:
        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import _points_in_polygon_batch_plot
    except ImportError:
        print("⚠️ Kann _points_in_polygon_batch_plot nicht importieren")
        return
    
    # Projiziere Polygon auf XZ-Ebene
    poly_points_dict = [
        {'x': p['x'], 'y': p['z']} for p in polygon_points
    ]
    
    # Projiziere Punkte auf XZ-Ebene
    points_u = all_points[:, 0]  # X
    points_v = all_points[:, 2]  # Z
    
    points_inside = _points_in_polygon_batch_plot(
        points_u.reshape(-1, 1),
        points_v.reshape(-1, 1),
        poly_points_dict
    )
    
    if points_inside is not None:
        points_inside_1d = points_inside.flatten()
        n_inside = np.sum(points_inside_1d)
        print(f"\n✅ Punkt-in-Polygon Test:")
        print(f"  └─ Punkte innerhalb: {n_inside}/{len(all_points)} ({100*n_inside/len(all_points):.1f}%)")
        print(f"  └─ Erwartet: ~{n_points//2}/{len(all_points)} (50%)")
        
        # Prüfe Zellen
        cell_centers = mesh.cell_centers().points
        centroids_u = cell_centers[:, 0]
        centroids_v = cell_centers[:, 2]
        
        centroids_inside = _points_in_polygon_batch_plot(
            centroids_u.reshape(-1, 1),
            centroids_v.reshape(-1, 1),
            poly_points_dict
        )
        
        if centroids_inside is not None:
            centroids_inside_1d = centroids_inside.flatten()
            
            cells_to_keep = []
            for i in range(mesh.n_cells):
                cell = mesh.get_cell(i)
                cell_point_ids = cell.point_ids
                all_points_inside = np.all(points_inside_1d[cell_point_ids])
                any_point_inside = np.any(points_inside_1d[cell_point_ids])
                centroid_inside = centroids_inside_1d[i] if i < len(centroids_inside_1d) else True
                
                # Aktuelle Filterlogik
                if centroid_inside or any_point_inside:
                    cells_to_keep.append(i)
            
            print(f"\n✅ Zellen-Filterung:")
            print(f"  └─ Behaltene Zellen: {len(cells_to_keep)}/{mesh.n_cells} ({100*len(cells_to_keep)/mesh.n_cells:.1f}%)")
            
            # Prüfe Rand-Zellen
            boundary_cells = []
            for i in cells_to_keep:
                cell = mesh.get_cell(i)
                cell_point_ids = cell.point_ids
                if not np.all(points_inside_1d[cell_point_ids]):  # Nicht alle Punkte innen
                    boundary_cells.append(i)
            
            print(f"  └─ Rand-Zellen (teilweise außen): {len(boundary_cells)}")


def test_resolution_consistency():
    """
    Testet die Auflösungskonsistenz bei verschiedenen Flächentypen.
    """
    print("\n" + "=" * 80)
    print("TEST: Auflösungskonsistenz")
    print("=" * 80)
    
    # Teste verschiedene Flächentypen
    test_cases = [
        {
            'name': 'Planar (XY)',
            'orientation': None,
            'points': np.array([
                [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]
            ]),
        },
        {
            'name': 'Sloped (XY mit Z-Variation)',
            'orientation': None,
            'points': np.array([
                [0, 0, 0], [10, 0, 5], [10, 10, 10], [0, 10, 5]
            ]),
        },
        {
            'name': 'Vertical XZ',
            'orientation': 'xz',
            'points': np.array([
                [0, 5, 0], [10, 5, 0], [10, 5, 10], [0, 5, 10]
            ]),
        },
        {
            'name': 'Slanted XZ',
            'orientation': 'xz_slanted',
            'points': np.array([
                [0, 5, 0], [10, 10, 0], [10, 20, 10], [0, 15, 10]
            ]),
        },
    ]
    
    for test_case in test_cases:
        print(f"\n📐 {test_case['name']}:")
        points = test_case['points']
        
        # Berechne Punktdichte
        if len(points) > 1:
            if test_case['orientation'] in ('xz', 'xz_slanted'):
                proj_points = points[:, [0, 2]]
            elif test_case['orientation'] in ('yz', 'yz_slanted'):
                proj_points = points[:, [1, 2]]
            else:
                proj_points = points[:, [0, 1]]
            
            # Berechne Fläche
            if len(proj_points) >= 3:
                # Verwende Shoelace-Formel
                area = 0.0
                for i in range(len(proj_points)):
                    j = (i + 1) % len(proj_points)
                    area += proj_points[i][0] * proj_points[j][1]
                    area -= proj_points[j][0] * proj_points[i][1]
                area = abs(area) / 2.0
                
                # Schätze Punktdichte (bei PLOT_UPSCALE_FACTOR=6)
                # Annahme: Grid mit ~10x10 Punkten, upgescaled auf 60x60
                base_resolution = 1.0  # 1m
                upscale_factor = 6
                estimated_points = int(area / (base_resolution / upscale_factor) ** 2)
                
                print(f"  └─ Fläche (projiziert): {area:.2f} m²")
                print(f"  └─ Geschätzte Punkte (bei Resolution={base_resolution}m, Upscale={upscale_factor}): ~{estimated_points}")


if __name__ == "__main__":
    print("🔍 Debug-Skript: Triangulation-Analyse")
    print("=" * 80)
    
    # Teste Triangulation
    results = test_triangulation_xz_slanted()
    
    # Teste Filterung
    test_filtering_logic()
    
    # Teste Auflösung
    test_resolution_consistency()
    
    print("\n" + "=" * 80)
    print("✅ Analyse abgeschlossen")
    print("=" * 80)

