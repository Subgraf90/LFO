# Detaillierter Vergleich: Positionierung und Plot-Erstellung

## 1. Positionierungs-Strategien

### Plot3DSPL.py: Texture-basierte Positionierung

**Konzept:**
- Erstellt ein **2D-Textur-Grid** basierend auf der **Bounding Box** der Surface-Punkte
- Interpoliert SPL-Werte auf dieses Textur-Grid
- Positioniert ein **flaches StructuredGrid** an der Surface-Position
- Textur wird auf das Grid "geklebt"

**Ablauf:**

```
1. Surface-Punkte analysieren:
   - poly_x, poly_y, poly_z aus Surface-Definition
   - Bounding Box berechnen: xmin, xmax, ymin, ymax

2. Textur-Grid erstellen:
   - Bounding Box + Margin (0.5 * tex_res)
   - Grid-Auflösung: tex_res_global (z.B. 0.03m oder 0.015m)
   - Erstellt reguläres Grid: xs, ys = np.linspace(...)
   - X, Y = np.meshgrid(xs, ys)

3. Positionierung:
   - Polygon-Maske: Prüft welche Grid-Punkte innerhalb der Surface sind
   - Randoptimierung: Projiziert Rand-Punkte auf Polygon-Kanten
   - Z-Koordinaten: Berechnet aus plane_model (für schräge Flächen)
     * Konstante Fläche: Z = intercept
     * Schräge Fläche: Z = slope * X + intercept (oder xy-Modus)

4. SPL-Werte interpolieren:
   - Von source_x, source_y, values → Textur-Grid (X, Y)
   - Bilineare Interpolation (Gradient) oder Nearest Neighbor (Color Step)

5. StructuredGrid erstellen:
   - Grid mit X, Y, Z_surface Koordinaten
   - Textur mit SPL-Farben
   - Position: Exakt an der Surface-Position (Weltkoordinaten)
```

**Code-Stellen:**
- `_process_planar_surface_texture()` (Zeile 1603-2324 in Plot3D.py)
- Grid-Erstellung: Zeile 1793-1856
- Z-Koordinaten: Zeile 1859-1879
- Interpolation: Zeile 1897-1982

**Positionierungs-Details:**
```python
# Bounding Box aus Surface-Punkten
xmin, xmax = float(poly_x.min()), float(poly_x.max())
ymin, ymax = float(poly_y.min()), float(poly_y.max())

# Textur-Grid mit Margin
margin = tex_res_surface * 0.5
x_start = xmin - margin
x_end = xmax + margin
y_start = ymin - margin
y_end = ymax + margin

# Reguläres Grid erstellen
num_x = int(np.ceil((x_end - x_start) / tex_res_surface)) + 1
num_y = int(np.ceil((y_end - y_start) / tex_res_surface)) + 1
xs = np.linspace(x_start, x_end, num_x)
ys = np.linspace(y_start, y_end, num_y)
X, Y = np.meshgrid(xs, ys, indexing="xy")

# Z-Koordinaten für schräge Flächen
if is_slanted:
    if mode == "x":
        Z_surface = slope * X + intercept
    elif mode == "y":
        Z_surface = slope * Y + intercept
    else:  # mode == "xy"
        Z_surface = slope_x * X + slope_y * Y + intercept
else:
    Z_surface = None  # Wird später aus plane_model berechnet
```

**Vorteile:**
- ✅ Einheitliche Textur-Auflösung pro Surface
- ✅ Glatte Darstellung durch bilineare Interpolation
- ✅ Cache-Mechanismus für Texturen
- ✅ Randoptimierung für saubere Kanten

**Nachteile:**
- ⚠️ Zusätzliche Interpolation-Schicht (source → textur)
- ⚠️ Grid-Größe hängt von Bounding Box ab (nicht von tatsächlicher Surface-Form)
- ⚠️ Potentiell viele leere Pixel außerhalb der Surface

---

### Plot3DSPL_new.py: Direkte Mesh-Positionierung

**Konzept:**
- Verwendet **direkt die Grid-Positionen** aus `surface_grids_data`
- Keine zusätzliche Textur-Grid-Erstellung
- Erstellt **direktes 3D-Mesh** aus den berechneten Grid-Punkten
- Position: Exakt die berechneten Weltkoordinaten

**Ablauf:**

```
1. Grid-Daten laden:
   - X_grid, Y_grid, Z_grid aus surface_grids_data[surface_id]
   - surface_mask: Welche Punkte aktiv sind
   - sound_field_x, sound_field_y: Achsen des Grids

2. SPL-Werte laden:
   - sound_field_p_complex aus surface_results_data[surface_id]
   - Konvertierung zu dB: mag2db(abs(sound_field_p_complex))

3. Optional: Upscaling:
   - PLOT_UPSCALE_FACTOR (z.B. 2x)
   - Erstellt feineres Grid: x_fine, y_fine
   - Interpoliert Z_grid und spl_values auf feineres Grid
   - Bilineare Interpolation (Gradient) oder Nearest Neighbor (Color Step)

4. Mesh-Erstellung:
   - Option A: Triangulation aus Surface-Punkten
     * triangulate_points(pts) → Dreiecke
     * Interpoliert SPL auf Vertex-Positionen (griddata)
   - Option B: Strukturiertes Grid (Fallback)
     * build_surface_mesh(x_plot, y_plot, scalars, z_coords=Z_plot)

5. Positionierung:
   - Direkt die Weltkoordinaten aus X_grid, Y_grid, Z_grid
   - Keine zusätzliche Transformation
   - Mesh-Position = Berechnete Grid-Position
```

**Code-Stellen:**
- `update_spl_plot()` (Zeile 768-1534 in Plot3DSPL_new.py)
- Grid-Laden: Zeile 965-984
- Upscaling: Zeile 1040-1112
- Triangulation: Zeile 1158-1377
- Mesh-Erstellung: Zeile 1487-1497

**Positionierungs-Details:**
```python
# Direkt aus surface_grids_data
X_grid = np.array(grid_data['X_grid'], dtype=float)  # 2D-Array, Weltkoordinaten
Y_grid = np.array(grid_data['Y_grid'], dtype=float)  # 2D-Array, Weltkoordinaten
Z_grid = np.array(grid_data['Z_grid'], dtype=float)  # 2D-Array, Weltkoordinaten
surface_mask = np.array(grid_data['surface_mask'], dtype=bool)

# Achsen extrahieren
sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)  # 1D-Array
sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)  # 1D-Array

# Optional: Upscaling
if PLOT_UPSCALE_FACTOR > 1:
    x_fine = np.linspace(sound_field_x.min(), sound_field_x.max(), 
                         nx * PLOT_UPSCALE_FACTOR)
    y_fine = np.linspace(sound_field_y.min(), sound_field_y.max(), 
                         ny * PLOT_UPSCALE_FACTOR)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='xy')
    
    # Interpoliere Z und SPL auf feineres Grid
    Z_fine = _bilinear_interpolate_grid(sound_field_x, sound_field_y, Z_grid,
                                        X_fine.ravel(), Y_fine.ravel())
    spl_fine = _bilinear_interpolate_grid(sound_field_x, sound_field_y, spl_values,
                                          X_fine.ravel(), Y_fine.ravel())
    
    X_plot = X_fine
    Y_plot = Y_fine
    Z_plot = Z_fine
    spl_plot = spl_fine
else:
    # Kein Upscaling: Direkt verwenden
    X_plot = X_grid
    Y_plot = Y_grid
    Z_plot = Z_grid
    spl_plot = spl_values

# Mesh-Erstellung (Triangulation)
tris = triangulate_points(pts)  # Aus Surface-Definition
verts = np.array([[p.get("x"), p.get("y"), p.get("z")] for tri in tris for p in tri])

# Interpoliere SPL auf Vertices
points_new = verts[:, :2]  # x, y
points_orig = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
spl_at_verts = griddata(points_orig, spl_values.ravel(), points_new, method='nearest')

# Erstelle PyVista Mesh
mesh = pv.PolyData(combined_verts, combined_faces)
mesh["plot_scalars"] = combined_scalars
```

**Vorteile:**
- ✅ Keine zusätzliche Interpolation (direkt berechnete Positionen)
- ✅ Exakte Positionen aus Berechnung
- ✅ Keine leeren Pixel (nur aktive Grid-Punkte)
- ✅ Bessere Kontrolle über Mesh-Form

**Nachteile:**
- ⚠️ Kein Textur-Caching
- ⚠️ Jedes Mesh wird neu erstellt
- ⚠️ Potentiell mehr Render-Aufwand bei vielen Surfaces

---

## 2. Detaillierte Unterschiede

### 2.1 Grid-Erstellung

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| **Quelle** | Bounding Box der Surface-Punkte | Direkt aus `surface_grids_data` |
| **Methode** | Reguläres Grid (linspace) | Berechnete Grid-Positionen |
| **Auflösung** | `tex_res_global` (z.B. 0.03m) | Berechnete Grid-Auflösung |
| **Form** | Rechteckig (Bounding Box) | Exakte Surface-Form |
| **Margin** | 0.5 * tex_res | Kein Margin (direkt) |

### 2.2 Z-Koordinaten

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| **Quelle** | `plane_model` (aus Surface-Punkten) | Direkt aus `Z_grid` |
| **Berechnung** | `Z = slope * X + intercept` (für schräge Flächen) | Berechnete Z-Koordinaten |
| **Genauigkeit** | Approximation (plane_model) | Exakt (berechnete Werte) |

### 2.3 SPL-Werte Interpolation

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| **Quelle** | `surface_overrides` (source_x, source_y, values) | Direkt aus `surface_results_data` |
| **Interpolation** | source → Textur-Grid | Optional: Original → Upscaled |
| **Methode** | Bilinear (Gradient) / Nearest (Step) | Bilinear (Gradient) / Nearest (Step) |
| **Schritte** | 2: source → textur → render | 1: source → render (oder 2 mit Upscaling) |

### 2.4 Mesh-Erstellung

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| **Typ** | StructuredGrid (flach) | PolyData (trianguliert) oder StructuredGrid |
| **Position** | Textur-Grid Position | Direkt Grid-Positionen |
| **Form** | Rechteckig (Textur) | Exakte Surface-Form |
| **Vertices** | Textur-Grid-Punkte | Triangulierte Surface-Punkte oder Grid-Punkte |

---

## 3. Positionierungs-Beispiele

### Beispiel 1: Horizontale rechteckige Surface

**Plot3DSPL.py:**
```
Surface-Punkte: [(0,0,0), (5,0,0), (5,3,0), (0,3,0)]
Bounding Box: x=[0, 5], y=[0, 3]
Textur-Grid: 
  - x: 0.0 bis 5.0 (mit Margin)
  - y: 0.0 bis 3.0 (mit Margin)
  - Auflösung: 0.03m → ~167x100 Punkte
  - Z: Konstant (0.0)
Position: StructuredGrid bei Z=0.0
```

**Plot3DSPL_new.py:**
```
Grid-Daten aus surface_grids_data:
  - X_grid: Berechnete X-Positionen (z.B. 0.0, 0.1, 0.2, ..., 5.0)
  - Y_grid: Berechnete Y-Positionen (z.B. 0.0, 0.1, 0.2, ..., 3.0)
  - Z_grid: Berechnete Z-Positionen (z.B. alle 0.0)
  - surface_mask: Welche Punkte innerhalb der Surface sind
Position: Direkt die Grid-Positionen
```

### Beispiel 2: Schräge Surface

**Plot3DSPL.py:**
```
Surface-Punkte: [(0,0,0), (5,0,1), (5,3,1), (0,3,0)]
Bounding Box: x=[0, 5], y=[0, 3]
plane_model: mode="xy", slope_x=0.2, slope_y=0.0, intercept=0.0
Textur-Grid:
  - X, Y: Wie oben
  - Z = 0.2 * X + 0.0 * Y + 0.0 = 0.2 * X
Position: StructuredGrid mit schrägen Z-Koordinaten
```

**Plot3DSPL_new.py:**
```
Grid-Daten aus surface_grids_data:
  - X_grid: Berechnete X-Positionen
  - Y_grid: Berechnete Y-Positionen
  - Z_grid: Berechnete Z-Positionen (schräg, z.B. 0.0, 0.2, 0.4, ..., 1.0)
Position: Direkt die schrägen Grid-Positionen
```

### Beispiel 3: Vertikale Surface

**Plot3DSPL.py:**
```
Surface-Punkte: [(0,2,0), (0,2,2), (5,2,2), (5,2,0)]
Orientierung: "xz" (Y=konstant)
Textur-Grid:
  - u: X-Koordinaten (0 bis 5)
  - v: Z-Koordinaten (0 bis 2)
  - Position: Bei Y=2.0
Position: StructuredGrid bei Y=2.0
```

**Plot3DSPL_new.py:**
```
Grid-Daten aus surface_grids_data:
  - orientation: "vertical"
  - dominant_axis: "xz"
  - X_grid: Berechnete X-Positionen
  - Y_grid: Konstant (z.B. alle 2.0)
  - Z_grid: Berechnete Z-Positionen
Position: Direkt die vertikalen Grid-Positionen
```

---

## 4. Zusammenfassung der Positionierungs-Unterschiede

### Plot3DSPL.py (Texture-basiert)

**Positionierungs-Strategie:**
1. **Bounding Box** aus Surface-Punkten
2. **Reguläres Textur-Grid** erstellen
3. **Interpolation** von SPL-Werten auf Textur-Grid
4. **StructuredGrid** mit Textur positionieren

**Vorteile:**
- Einheitliche Auflösung
- Textur-Caching möglich
- Glatte Darstellung

**Nachteile:**
- Zusätzliche Interpolation
- Grid-Größe hängt von Bounding Box ab
- Potentiell viele leere Pixel

---

### Plot3DSPL_new.py (Direktes Mesh)

**Positionierungs-Strategie:**
1. **Direkt Grid-Daten** aus `surface_grids_data` verwenden
2. **Optional Upscaling** für bessere Darstellung
3. **Direktes Mesh** aus Grid-Positionen erstellen
4. **Keine zusätzliche Transformation**

**Vorteile:**
- Keine zusätzliche Interpolation
- Exakte Positionen
- Keine leeren Pixel
- Bessere Kontrolle

**Nachteile:**
- Kein Caching
- Jedes Mesh neu erstellt
- Potentiell mehr Render-Aufwand

---

## 5. Empfehlung

**Plot3DSPL_new.py** scheint der bessere Ansatz für Positionierung zu sein:
- ✅ Exakte Positionen aus Berechnung
- ✅ Keine unnötige Interpolation
- ✅ Direkter Datenfluss
- ✅ Bessere Genauigkeit

**Plot3DSPL.py** könnte Vorteile haben bei:
- Performance (Textur-Caching)
- Einheitliche Darstellung
- Kompatibilität mit älteren Code-Pfaden
