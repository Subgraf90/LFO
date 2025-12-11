# Wie werden berechnete SPL-Werte auf triangulierte Vertices gemappt?

## Übersicht: Von Grid zu Vertex - Der Mapping-Prozess

Der Prozess besteht aus **4 Hauptschritten**:

```
1. Berechnung (SoundfieldCalculator) 
   → 2D-Grid mit SPL-Werten (Pa → dB konvertiert)
   
2. Triangulation (FlexibleGridGenerator)
   → 3D-Vertices + Faces aus Surface-Definition
   
3. Interpolation (Plot3DSPL._render_surfaces_textured)
   → SPL-Werte von Grid → Vertices übertragen
   
4. Visualisierung (PyVista)
   → Mesh mit skalaren Werten pro Vertex
```

---

## Schritt 1: Berechnung der SPL-Werte (Grid-basiert)

### Datenquelle: `surface_results_data[surface_id]['sound_field_p']`

**Woher kommen die Daten?**
- `SoundfieldCalculator._calculate_sound_field_complex()` berechnet komplexe Druckwerte
- Werte werden auf einem **regulären 2D-Grid** berechnet: `X_grid`, `Y_grid` (Shape: `[ny, nx]`)
- Pro Grid-Punkt: Komplexer Druckwert `sound_field_p_complex` (in Pascal)

**Konvertierung:**
```python
# Zeile 967-970 in Plot3DSPL.py
pressure_magnitude = np.abs(sound_field_p_complex)  # Betrag in Pa
pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
spl_values_2d = self.functions.mag2db(pressure_magnitude)  # Konvertierung zu dB
```

**Resultat:**
- `spl_values_2d`: 2D-Array mit SPL-Werten in dB
- Shape: `[ny, nx]` (passt zu `X_grid`, `Y_grid`)
- Koordinaten: `X_grid[i,j]`, `Y_grid[i,j]` → SPL = `spl_values_2d[i,j]` dB

---

## Schritt 2: Triangulation der Surface-Geometrie

### Datenquelle: `surface_grids_data[surface_id]`

**Woher kommen die Daten?**
- `FlexibleGridGenerator.generate_per_surface()` erstellt die Triangulation
- Input: Surface-Definition (Polygon mit 3+ Punkten)
- Output: `triangulated_vertices`, `triangulated_faces`

**Triangulation-Prozess (FlexibleGridGenerator.py, Zeile 2256-2273):**
```python
# 1. Original-Polygon-Punkte
pts = [(x1, y1, z1), (x2, y2, z2), ...]  # Aus Surface-Definition

# 2. Triangulation (Delaunay oder ähnlich)
tris = triangulate_points(pts)  # Liefert Dreiecke

# 3. Konvertierung zu Vertices + Faces
triangulated_vertices = np.array([
    [x, y, z] for tri in tris for p in tri
])  # Shape: (N*3, 3) - alle Vertex-Koordinaten

triangulated_faces = np.array([
    [3, i, i+1, i+2] for i in range(0, len(vertices), 3)
])  # PyVista-Format: [n_vertices, v1, v2, v3, ...]
```

**Resultat:**
- `triangulated_vertices`: Array mit 3D-Koordinaten (X, Y, Z) pro Vertex
- `triangulated_faces`: Array mit Face-Indizes (welche Vertices bilden ein Dreieck)
- **Wichtig:** Vertices sind NICHT auf dem Grid - sie liegen genau auf der Surface-Geometrie!

---

## Schritt 3: Interpolation - SPL von Grid → Vertices

### Der kritische Schritt (Plot3DSPL.py, Zeile 977-1027)

**Problem:**
- Grid-Punkte haben SPL-Werte, aber sie liegen NICHT auf den Vertex-Positionen
- Vertices liegen auf der exakten Surface-Geometrie, haben aber NOCH KEINE SPL-Werte

**Lösung: Interpolation mit `scipy.interpolate.griddata`**

### 3.1 Vorbereitung: Grid-Positionen + Werte

```python
# Zeile 985-1002
# Grid-Positionen (2D: nur X, Y - Z wird separat behandelt)
points_orig = np.column_stack([
    X_grid.ravel(),    # X-Koordinaten aller Grid-Punkte
    Y_grid.ravel()     # Y-Koordinaten aller Grid-Punkte
])  # Shape: (N_grid, 2)

values_orig = spl_values_2d.ravel()  # SPL-Werte zu jedem Grid-Punkt
# Shape: (N_grid,) - entspricht points_orig

# 🎯 WICHTIG: Nur gültige Punkte verwenden (innerhalb der Surface)
if surface_mask.size == Xg.size:
    mask_flat = surface_mask.ravel().astype(bool)
    points_orig = points_orig[mask_flat]  # Nur Punkte innerhalb der Surface
    values_orig = values_orig[mask_flat]  # Entsprechende SPL-Werte
```

### 3.2 Ziel-Positionen: Vertex-Koordinaten

```python
# Zeile 978
points_new = triangulated_vertices[:, :2]  # Nur X, Y - Z ignorieren wir für Interpolation
# Shape: (N_vertices, 2)
```

**Warum nur X, Y?**
- Die Z-Koordinate wird bereits bei der Triangulation korrekt gesetzt
- Die SPL-Werte variieren hauptsächlich in X, Y (horizontale Ebene)
- Z ist meist konstant oder linear (abhängig von Surface-Orientierung)

### 3.3 Interpolation

```python
# Zeile 1021-1027
spl_at_verts = griddata(
    points_orig,      # Bekannte Positionen (Grid-Punkte)
    values_orig,      # Bekannte Werte (SPL in dB)
    points_new,       # Ziel-Positionen (Vertex-Koordinaten)
    method='nearest', # Nearest-Neighbor (immer, auch im Gradient-Mode)
    fill_value=np.nan # Was bei Punkten außerhalb des Grids?
)
# Resultat: spl_at_verts - SPL-Werte für JEDEN Vertex
# Shape: (N_vertices,)
```

**Warum `method='nearest'`?**
- Grid-Punkte sind diskret verteilt
- Bilineare Interpolation würde Werte zwischen Grid-Punkten schätzen
- `nearest` nimmt den nächstgelegenen Grid-Punkt → exakter Wert

---

## Schritt 4: Zuweisung zu PyVista-Mesh

### Mesh-Erstellung (Zeile 1104-1105)

```python
# Erstelle PyVista PolyData Mesh
mesh = pv.PolyData(triangulated_vertices, triangulated_faces)
# triangulated_vertices: (N, 3) - X, Y, Z pro Vertex
# triangulated_faces: (M, 4) - [3, v1, v2, v3] pro Dreieck

# Weise SPL-Werte als "scalars" zu
mesh["plot_scalars"] = spl_at_verts
# spl_at_verts: (N,) - SPL in dB pro Vertex
```

**PyVista interpoliert dann automatisch:**
- Zwischen Vertices innerhalb eines Dreiecks
- Verwendet die Colormap für Farbzuordnung
- Berücksichtigt `clim=(cbar_min, cbar_max)` für Skalierung

---

## Visualisierung: Von Vertex-Werten zu Farben

### PyVista Rendering (Zeile 1120-1129)

```python
actor = self.plotter.add_mesh(
    mesh,
    scalars="plot_scalars",  # Verwendet die Vertex-Werte
    cmap=cmap_object,        # Colormap (jet, phase, etc.)
    clim=(cbar_min, cbar_max), # Skalierung der Colormap
    smooth_shading=not is_step_mode,  # Glattung der Dreiecke
    interpolate_before_map=not is_step_mode,  # Interpolation zwischen Vertices
)
```

**Interpolation innerhalb eines Dreiecks:**
- PyVista hat 3 Vertices mit je einem SPL-Wert
- Innerhalb des Dreiecks wird linear interpoliert
- `interpolate_before_map=False` → Keine zusätzliche Glättung (für Color Step)

---

## Zusammenfassung: Der komplette Datenfluss

```
1. BEREICHNUNG (SoundfieldCalculator)
   Input:  Lautsprecher-Positionen, Balloon-Daten
   Output: sound_field_p_complex (Grid: [ny, nx], komplex)
           → Konvertierung zu spl_values_2d (Grid: [ny, nx], dB)
   
2. GRID-STRUKTUR (FlexibleGridGenerator)
   Input:  Surface-Definition (Polygon)
   Output: X_grid, Y_grid, Z_grid (Grid: [ny, nx])
           surface_mask (Grid: [ny, nx], bool)
           → Welche Grid-Punkte liegen auf der Surface?
   
3. TRIANGULATION (FlexibleGridGenerator)
   Input:  Surface-Definition (Polygon)
   Output: triangulated_vertices (N, 3) - Vertex-Koordinaten
           triangulated_faces (M, 4) - Dreieck-Definitionen
   
4. INTERPOLATION (Plot3DSPL)
   Input:  points_orig = [X_grid.ravel(), Y_grid.ravel()]  (N_grid, 2)
           values_orig = spl_values_2d.ravel()              (N_grid,)
           points_new = triangulated_vertices[:, :2]        (N_vertices, 2)
   Output: spl_at_verts (N_vertices,) - SPL pro Vertex
   
5. VISUALISIERUNG (PyVista)
   Input:  Mesh (vertices + faces) + spl_at_verts (scalars)
   Output: 3D-Plot mit farbcodierten SPL-Werten
```

---

## Wichtige Erkenntnisse

### ✅ Was funktioniert gut:

1. **Grid-basierte Berechnung** ist effizient (nur relevante Punkte)
2. **Triangulation** erzeugt exakte Surface-Geometrie
3. **Nearest-Neighbor Interpolation** gibt exakte Grid-Werte weiter

### ⚠️ Potenzielle Probleme:

1. **Diskrepanz zwischen Grid und Vertices:**
   - Grid-Punkte: Regulär verteilt, können außerhalb der Surface liegen
   - Vertices: Liegen EXAKT auf der Surface-Geometrie
   - → Interpolation kann Werte von nahen Grid-Punkten nehmen, die nicht auf der Surface sind

2. **Z-Koordinate wird ignoriert:**
   - Interpolation nutzt nur X, Y
   - Bei schrägen Flächen kann das zu Ungenauigkeiten führen
   - → Aktuell OK, da SPL-Werte hauptsächlich von X, Y abhängen

3. **Edge-Cases:**
   - Vertices außerhalb des Grids → `fill_value=np.nan`
   - Sehr wenige Grid-Punkte → Interpolation wird ungenau

### 🎯 Verbesserungsmöglichkeiten:

1. **3D-Interpolation** (X, Y, Z) für schräge Flächen
2. **Bilineare Interpolation** statt Nearest-Neighbor (wenn Grid fein genug)
3. **Surface-bewusste Interpolation** (nur Grid-Punkte auf/nah an Surface verwenden)

---

## Code-Referenzen

- **Berechnung:** `SoundfieldCalculator._calculate_sound_field_complex()` (Zeile 130-736)
- **Triangulation:** `FlexibleGridGenerator._generate_single_surface_grid()` (Zeile 2241-2312)
- **Interpolation:** `Plot3DSPL._render_surfaces_textured()` (Zeile 977-1027)
- **Visualisierung:** `Plot3DSPL._render_surfaces_textured()` (Zeile 1104-1129)