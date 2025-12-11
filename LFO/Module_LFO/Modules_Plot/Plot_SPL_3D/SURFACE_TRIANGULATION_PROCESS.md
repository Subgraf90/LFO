# Wie werden Surfaces in der Triangulation verarbeitet?

## 📋 Übersicht

Die Triangulation von Surfaces erfolgt in **3 Stufen**:
1. **FlexibleGridGenerator.py**: Erstellt triangulierte Vertices und Faces pro Surface
2. **SoundfieldCalculator.py**: Speichert triangulierte Daten für Plot-Modul
3. **Plot3DSPL.py**: Verwendet triangulierte Daten für Rendering

---

## 🔄 Verarbeitungsprozess

### Stufe 1: FlexibleGridGenerator.py - Triangulation erstellen

#### 1.1 Einzelne Surfaces (`generate_per_surface()`)

**Code-Stelle:** Zeile 2226-2257

```python
# Für jede Surface einzeln:
for geom in geometries:
    # Triangulation der Surface-Punkte
    if len(geom.points) >= 3:
        tris = triangulate_points(geom.points)  # ← Triangulation hier!
        
        if tris and len(tris) > 0:
            # Konvertiere zu Vertices und Faces
            verts_list = []
            for tri in tris:
                for p in tri:
                    verts_list.append([p.get("x"), p.get("y"), p.get("z")])
            
            verts = np.array(verts_list, dtype=float)
            
            # Erstelle Faces-Array (PyVista-Format)
            faces_list = []
            vertex_offset = 0
            for _ in tris:
                faces_list.extend([3, vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3
            
            triangulated_vertices = verts
            triangulated_faces = np.array(faces_list, dtype=np.int64)
            triangulated_success = True
```

**Wichtig:**
- ✅ **Jede Surface wird einzeln trianguliert**
- ✅ Verwendet `triangulate_points(geom.points)` - die Surface-Punkte direkt
- ✅ Ergebnis: `triangulated_vertices` (N×3) und `triangulated_faces` (M×4)

---

#### 1.2 Einzelne Surface (`generate_single_surface_grid()`)

**Code-Stelle:** Zeile 2302-2312

**Gleiche Logik wie 1.1**, aber für eine einzelne Surface:
```python
# Identisch zu generate_per_surface()
tris = triangulate_points(geom.points)
# ... Konvertierung zu Vertices/Faces ...
```

---

#### 1.3 Gruppen von Surfaces (`generate_per_group()`)

**Code-Stelle:** Zeile 2423-2433

**Unterschied:** Hier werden **alle Surfaces einer Gruppe kombiniert**:

```python
# Kombiniere alle Surface-Punkte für Triangulation
all_points = []
for geom in geometries:
    all_points.extend(geom.points)  # ← Alle Punkte zusammen!

if len(all_points) >= 3:
    tris = triangulate_points(all_points)  # ← Triangulation auf KOMBINIERTEN Punkten
    # ... Rest identisch ...
```

**Wichtig:**
- ⚠️ **Gruppen-Surfaces werden KOMBINIERT trianguliert**
- ⚠️ Alle Punkte aller Surfaces in der Gruppe werden zusammen trianguliert
- ⚠️ Ergebnis: Ein gemeinsames Mesh für die gesamte Gruppe

---

### Stufe 2: SoundfieldCalculator.py - Daten speichern

**Code-Stelle:** Zeile 240-250

```python
# Für jede Surface (oder Gruppe):
grid_data = {
    'X_grid': grid.X_grid.tolist(),
    'Y_grid': grid.Y_grid.tolist(),
    'Z_grid': grid.Z_grid.tolist(),
    # ... andere Felder ...
}

# 🎯 TRIANGULATION: Füge triangulierte Daten hinzu
if hasattr(grid, 'triangulated_vertices') and grid.triangulated_vertices is not None:
    grid_data['triangulated_vertices'] = grid.triangulated_vertices.tolist()
if hasattr(grid, 'triangulated_faces') and grid.triangulated_faces is not None:
    grid_data['triangulated_faces'] = grid.triangulated_faces.tolist()
if hasattr(grid, 'triangulated_success'):
    grid_data['triangulated_success'] = grid.triangulated_success

surface_grids_data[surface_id] = grid_data
```

**Wichtig:**
- ✅ Triangulierte Daten werden in `calculation_spl['surface_grids']` gespeichert
- ✅ Jede Surface-ID hat ihre eigenen triangulierten Daten

---

### Stufe 3: Plot3DSPL.py - Rendering

**Code-Stelle:** Zeile 830-950

#### 3.1 Prüfung auf verfügbare Triangulation

```python
# Für jede Surface einzeln:
for surface_id, points, surface_obj in surfaces_to_process:
    # Prüfe ob triangulierte Daten verfügbar sind
    if surface_id in surface_grids_data:
        grid_data = surface_grids_data[surface_id]
        triangulated_success = grid_data.get('triangulated_success', False)
        
        if triangulated_success:
            triangulated_vertices = np.array(grid_data.get('triangulated_vertices'))
            triangulated_faces = np.array(grid_data.get('triangulated_faces'))
            use_triangulation = True
```

#### 3.2 SPL-Werte auf Vertices interpolieren

```python
if use_triangulation:
    # Lade SPL-Werte aus surface_results_data
    result_data = surface_results_data[surface_id]
    sound_field_p_complex = np.array(result_data.get('sound_field_p', []))
    
    # Konvertiere zu SPL in dB
    if time_mode:
        spl_values_2d = np.real(sound_field_p_complex)
    elif phase_mode:
        spl_values_2d = np.angle(sound_field_p_complex)
    else:
        spl_values_2d = self.functions.mag2db(np.abs(sound_field_p_complex))
    
    # Interpoliere SPL auf Vertex-Positionen
    points_new = triangulated_vertices[:, :2]  # x, y Koordinaten der Vertices
    points_orig = np.column_stack([sx.ravel(), sy.ravel()])  # Grid-Koordinaten
    
    spl_at_verts = griddata(
        points_orig,      # Original-Grid-Positionen
        spl_values_2d.ravel(),  # SPL-Werte am Grid
        points_new,       # Vertex-Positionen
        method='nearest',
        fill_value=np.nan
    )
```

#### 3.3 PyVista Mesh erstellen und rendern

```python
    # Erstelle PyVista PolyData Mesh
    mesh = pv.PolyData(triangulated_vertices, triangulated_faces)
    mesh["plot_scalars"] = spl_at_verts  # SPL-Werte an jedem Vertex
    
    # Rendere Mesh
    actor = self.plotter.add_mesh(
        mesh,
        scalars="plot_scalars",
        cmap=cmap_object,
        clim=(cbar_min, cbar_max),
        smooth_shading=not is_step_mode,
    )
```

---

## 🔍 Details zur Triangulation (`triangulate_points`)

**Quelle:** `SurfaceValidator.py`, Zeile 586-704

### Methode 1: Delaunay-Triangulation (scipy)

```python
# 1. Projiziere Punkte auf Ebene
proj, origin, basis = _project_to_plane(pts, model)

# 2. Delaunay-Triangulation
tri = Delaunay(proj)
triangles_delaunay = tri.simplices  # N×3 Array von Indices

# 3. Filtere Dreiecke außerhalb des Polygons
valid_triangles = []
for tri_idx in triangles_delaunay:
    centroid = proj[tri_idx].mean(axis=0)
    if _point_in_polygon(centroid, polygon_closed):
        valid_triangles.append(tri_idx)

# 4. Konvertiere zu Punkt-Listen
triangles = []
for a, b, c in valid_triangles:
    triangles.append([points[a], points[b], points[c]])
```

**Vorteile:**
- ✅ Robust für konvexe und konkave Polygone
- ✅ Optimale Dreiecksform (Delaunay-Kriterium)

### Methode 2: Fan-Triangulation (Fallback)

```python
# 1. Sortiere Punkte nach Winkel
centroid = proj.mean(axis=0)
rel = proj - centroid
angles = np.arctan2(rel[:, 1], rel[:, 0])
order = np.argsort(angles)

# 2. Erstelle Dreiecke vom ersten Punkt aus
anchor = order[0]
for i in range(1, len(order) - 1):
    tris_idx.append((anchor, order[i], order[i + 1]))
```

**Vorteile:**
- ✅ Funktioniert immer (garantiert n-2 Dreiecke)
- ✅ Keine externe Abhängigkeit

---

## 📊 Datenstruktur

### Input: Surface-Punkte

```python
geom.points = [
    {"x": 0.0, "y": 0.0, "z": 0.0},
    {"x": 1.0, "y": 0.0, "z": 0.0},
    {"x": 1.0, "y": 1.0, "z": 0.0},
    {"x": 0.0, "y": 1.0, "z": 0.0},
]
```

### Output: Triangulation

```python
triangulated_vertices = np.array([
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 0.0, 0.0],  # Vertex 1
    [1.0, 1.0, 0.0],  # Vertex 2
    [0.0, 1.0, 0.0],  # Vertex 3
])

triangulated_faces = np.array([
    3, 0, 1, 2,  # Dreieck 0: Vertices 0, 1, 2
    3, 0, 2, 3,  # Dreieck 1: Vertices 0, 2, 3
])
# Format: [n_vertices, v1, v2, v3, n_vertices, v1, v2, v3, ...]
```

---

## ⚠️ Wichtige Unterschiede

### Einzelne Surfaces vs. Gruppen

| Aspekt | Einzelne Surfaces | Gruppen |
|--------|-------------------|---------|
| **Triangulation** | Pro Surface einzeln | Alle Surfaces kombiniert |
| **Input** | `geom.points` (eine Surface) | `all_points` (alle Surfaces) |
| **Ergebnis** | Ein Mesh pro Surface | Ein Mesh für gesamte Gruppe |
| **Surface-ID** | `geom.surface_id` | `gid` (Gruppen-ID) |

### Beispiel: 2 Surfaces in einer Gruppe

**Einzelne Surfaces:**
```
Surface A: 4 Punkte → 2 Dreiecke → Mesh A
Surface B: 4 Punkte → 2 Dreiecke → Mesh B
```

**Gruppe:**
```
Gruppe: 8 Punkte (A+B kombiniert) → 6 Dreiecke → Ein gemeinsames Mesh
```

---

## 🎯 Priorisierung im Plotting

1. **PRIORITÄT 1:** Triangulation (wenn `triangulated_success == True`)
   - Verwendet triangulierte Vertices direkt
   - Interpoliert SPL auf Vertex-Positionen
   - Erstellt `pv.PolyData` Mesh

2. **PRIORITÄT 2:** Texture-Pfad (Fallback)
   - Wird verwendet, wenn keine Triangulation verfügbar
   - Erstellt strukturiertes Grid mit Textur

---

## ✅ Zusammenfassung

1. **Triangulation erfolgt pro Surface** (oder pro Gruppe)
2. **Jede Surface wird einzeln verarbeitet** in Plot3DSPL.py
3. **SPL-Werte werden auf Vertex-Positionen interpoliert**
4. **PyVista Mesh wird direkt gerendert** (überspringt Texture-Pfad)
5. **Fallback zu Texture-Pfad** wenn Triangulation fehlschlägt
