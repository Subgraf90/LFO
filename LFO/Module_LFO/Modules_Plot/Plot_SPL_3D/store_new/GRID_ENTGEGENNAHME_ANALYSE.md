# Analyse: Verschiedene Möglichkeiten der Grid-Entgegennahme

## Übersicht

Die Datei `Plot3DSPL.py` nimmt Grids auf **verschiedene Weise** entgegen, abhängig von:
1. **Surface-Orientierung** (horizontal vs. vertikal)
2. **Datenquelle** (`surface_grids_data` vs. `surface_results_data`)
3. **Grid-Struktur** (2D-Grids vs. 1D-Achsen)

---

## 1. Hauptmethode: Direkt aus `calculation_spl`

### Datenquelle

```python
# Zeile 817-819
calc_spl = getattr(container, "calculation_spl", {}) or {}
surface_grids_data = calc_spl.get("surface_grids", {})
surface_results_data = calc_spl.get("surface_results", {})
```

**Struktur:**
- `calculation_spl['surface_grids'][surface_id]` → Grid-Daten
- `calculation_spl['surface_results'][surface_id]` → SPL-Werte

---

## 2. Grid-Felder aus `surface_grids_data`

### Standard-Felder (für alle Surfaces)

```python
# Zeile 968-973
X_grid = np.array(grid_data['X_grid'], dtype=float)        # 2D-Array: Weltkoordinaten X
Y_grid = np.array(grid_data['Y_grid'], dtype=float)        # 2D-Array: Weltkoordinaten Y
Z_grid = np.array(grid_data['Z_grid'], dtype=float)        # 2D-Array: Weltkoordinaten Z
sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)  # 1D-Array: X-Achse
sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)  # 1D-Array: Y-Achse
surface_mask = np.array(grid_data['surface_mask'], dtype=bool)     # 2D-Array: Aktive Punkte
```

**Bedeutung:**
- `X_grid`, `Y_grid`, `Z_grid`: 2D-Grids mit Weltkoordinaten (Shape: `(ny, nx)`)
- `sound_field_x`, `sound_field_y`: 1D-Achsen für Interpolation (Länge: `nx`, `ny`)
- `surface_mask`: Bool-Array, welche Grid-Punkte innerhalb der Surface sind

### Zusätzliche Metadaten

```python
# Zeile 948-949
orientation = grid_data.get('orientation', 'unknown')      # 'planar', 'vertical', 'sloped'
dominant_axis = grid_data.get('dominant_axis', None)       # 'xz', 'yz' für vertikale Surfaces
```

---

## 3. Unterschiedliche Verarbeitung nach Orientierung

### 3.1 Horizontale Surfaces (planar/sloped)

**Pfad:** `update_spl_plot()` → Haupt-Loop (Zeile 912-1570)

**Grid-Entgegennahme:**
```python
# Zeile 968-973
X_grid = np.array(grid_data['X_grid'], dtype=float)
Y_grid = np.array(grid_data['Y_grid'], dtype=float)
Z_grid = np.array(grid_data['Z_grid'], dtype=float)
sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)
sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)
surface_mask = np.array(grid_data['surface_mask'], dtype=bool)
```

**Verwendung:**
- Direkt für Plotting verwendet
- Optional: Upscaling mit `PLOT_UPSCALE_FACTOR`
- Interpolation auf Vertex-Positionen (Triangulation)

**SPL-Werte:**
```python
# Zeile 989
sound_field_p_complex = np.array(result_data['sound_field_p'], dtype=complex)
```

---

### 3.2 Vertikale Surfaces

**Pfad:** `_update_vertical_spl_surfaces_from_grids()` (Zeile 1755-2579)

**Grid-Entgegennahme:**
```python
# Zeile 1914-1918
X_grid = np.array(grid_data['X_grid'], dtype=float)
Y_grid = np.array(grid_data['Y_grid'], dtype=float)
Z_grid = np.array(grid_data['Z_grid'], dtype=float)
sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)
sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)
```

**Besonderheiten:**
- **Orientierung wird geprüft:** `orientation == 'vertical'` (Zeile 1853)
- **Dominante Achse wird verwendet:** `dominant_axis` ('xz' oder 'yz')
- **Koordinaten-Transformation:** Für vertikale Surfaces werden Koordinaten transformiert

**Koordinaten-Transformation (Zeile 2015-2048):**

**Für XZ-Wand (dominant_axis='xz'):**
```python
# Zeile 2021-2024
u_axis = np.unique(X_grid)  # X-Koordinaten
v_axis = np.unique(Z_grid)  # Z-Koordinaten
# Y ist konstant
```

**Für YZ-Wand (dominant_axis='yz'):**
```python
# Zeile 2029-2032
u_axis = np.unique(Y_grid)  # Y-Koordinaten
v_axis = np.unique(Z_grid)  # Z-Koordinaten
# X ist konstant
```

**Fallback (keine Orientierung):**
```python
# Zeile 2053-2054
u_axis = sound_field_x  # Standard X-Achse
v_axis = sound_field_y  # Standard Y-Achse
```

---

## 4. Unterschiedliche Datenstrukturen

### 4.1 2D-Grids (X_grid, Y_grid, Z_grid)

**Format:**
- Shape: `(ny, nx)` - 2D-Array
- Typ: `numpy.ndarray` mit `dtype=float`
- Bedeutung: Weltkoordinaten für jeden Grid-Punkt

**Verwendung:**
- Direktes Plotting (Mesh-Erstellung)
- Interpolation auf Vertex-Positionen
- Upscaling (optional)

**Beispiel:**
```python
X_grid.shape = (100, 200)  # 100 Zeilen, 200 Spalten
Y_grid.shape = (100, 200)  # Entspricht X_grid
Z_grid.shape = (100, 200)  # Entspricht X_grid
```

---

### 4.2 1D-Achsen (sound_field_x, sound_field_y)

**Format:**
- Shape: `(nx,)` bzw. `(ny,)` - 1D-Array
- Typ: `numpy.ndarray` mit `dtype=float`
- Bedeutung: Achsen-Koordinaten für Interpolation

**Verwendung:**
- Interpolation (bilinear/nearest)
- Upscaling (feineres Grid erstellen)
- Grid-Erstellung

**Beispiel:**
```python
sound_field_x.shape = (200,)  # 200 X-Koordinaten
sound_field_y.shape = (100,)  # 100 Y-Koordinaten
```

**Zusammenhang mit 2D-Grids:**
```python
# X_grid[0, :] entspricht sound_field_x
# Y_grid[:, 0] entspricht sound_field_y
```

---

### 4.3 Surface-Maske (surface_mask)

**Format:**
- Shape: `(ny, nx)` - 2D-Array
- Typ: `numpy.ndarray` mit `dtype=bool`
- Bedeutung: Welche Grid-Punkte innerhalb der Surface sind

**Verwendung:**
- Filterung aktiver Punkte
- Upscaling (Maske wird mit interpoliert)
- Mesh-Erstellung (nur aktive Punkte)

**Beispiel:**
```python
surface_mask.shape = (100, 200)  # Entspricht X_grid
active_points = np.sum(surface_mask)  # Anzahl aktiver Punkte
```

---

## 5. Unterschiedliche Verarbeitungswege

### 5.1 Mit Upscaling (PLOT_UPSCALE_FACTOR > 1)

**Pfad:** Zeile 1044-1105

**Grid-Entgegennahme:**
```python
# Originale Achsen
sound_field_x = np.array(grid_data['sound_field_x'], dtype=float)
sound_field_y = np.array(grid_data['sound_field_y'], dtype=float)

# Erstelle feineres Grid
x_fine = np.linspace(sound_field_x.min(), sound_field_x.max(), nx * PLOT_UPSCALE_FACTOR)
y_fine = np.linspace(sound_field_y.min(), sound_field_y.max(), ny * PLOT_UPSCALE_FACTOR)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='xy')

# Interpoliere Z und SPL auf feineres Grid
Z_fine = self._bilinear_interpolate_grid(sound_field_x, sound_field_y, Z_grid, ...)
spl_fine = self._bilinear_interpolate_grid(sound_field_x, sound_field_y, spl_values, ...)
```

**Verwendung:**
- Bessere Darstellung (glattere Kanten)
- Mehr Render-Aufwand

---

### 5.2 Ohne Upscaling (PLOT_UPSCALE_FACTOR = 1)

**Pfad:** Zeile 1106-1114

**Grid-Entgegennahme:**
```python
# Direkt verwenden
X_plot = X_grid
Y_plot = Y_grid
Z_plot = Z_grid
spl_plot = spl_values
```

**Verwendung:**
- Schneller (keine Interpolation)
- Geringere Auflösung

---

### 5.3 Triangulation (für planare Surfaces)

**Pfad:** Zeile 1160-1377

**Grid-Entgegennahme:**
```python
# Originale Grid-Koordinaten
points_orig = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

# SPL-Werte (quantisiert)
spl_orig_quantized = self._quantize_to_steps(spl_values.ravel(), cbar_step)

# Vertex-Positionen aus Surface-Definition
points_new = verts[:, :2]  # x, y aus triangulierten Vertices

# Interpoliere SPL auf Vertices
spl_at_verts = griddata(points_orig, spl_orig_quantized, points_new, method='nearest')
```

**Verwendung:**
- Glatte Kanten
- Exakte Surface-Form
- Mehr Render-Aufwand

---

### 5.4 Fallback: build_surface_mesh (strukturiertes Grid)

**Pfad:** Zeile 1397-1407, 1424-1433

**Grid-Entgegennahme:**
```python
# Verwendet bereits verarbeitete Daten
x_plot, y_plot  # Aus Upscaling oder original
scalars_for_mesh = np.clip(scalars, cbar_min, cbar_max)
Z_plot  # Aus Upscaling oder original

mesh = build_surface_mesh(x_plot, y_plot, scalars_for_mesh, z_coords=Z_plot, ...)
```

**Verwendung:**
- Fallback wenn Triangulation fehlschlägt
- Strukturiertes Grid (regulär)

---

## 6. Zusammenfassung der verschiedenen Möglichkeiten

| Aspekt | Horizontale Surfaces | Vertikale Surfaces |
|--------|---------------------|-------------------|
| **Datenquelle** | `surface_grids_data[surface_id]` | `surface_grids_data[surface_id]` |
| **Grid-Felder** | `X_grid`, `Y_grid`, `Z_grid`, `sound_field_x`, `sound_field_y`, `surface_mask` | Gleiche Felder |
| **Orientierung** | `orientation != 'vertical'` | `orientation == 'vertical'` |
| **Koordinaten** | Direkt X, Y, Z | Transformiert (u, v) für XZ/YZ |
| **Verarbeitung** | Triangulation oder strukturiertes Grid | Strukturiertes Grid |
| **Upscaling** | Optional (PLOT_UPSCALE_FACTOR) | Optional (PLOT_UPSCALE_FACTOR) |

---

## 7. Datenfluss-Diagramm

```
calculation_spl
  ├─ surface_grids_data[surface_id]
  │   ├─ X_grid (2D, ny×nx)
  │   ├─ Y_grid (2D, ny×nx)
  │   ├─ Z_grid (2D, ny×nx)
  │   ├─ sound_field_x (1D, nx)
  │   ├─ sound_field_y (1D, ny)
  │   ├─ surface_mask (2D, ny×nx, bool)
  │   ├─ orientation ('planar', 'vertical', 'sloped')
  │   └─ dominant_axis ('xz', 'yz', None)
  │
  └─ surface_results_data[surface_id]
      └─ sound_field_p (2D, ny×nx, complex)

↓

update_spl_plot() oder _update_vertical_spl_surfaces_from_grids()

↓

[Optional: Upscaling]
  ├─ x_fine, y_fine (1D, feiner)
  ├─ X_fine, Y_fine (2D, feiner)
  ├─ Z_fine (2D, interpoliert)
  └─ spl_fine (2D, interpoliert)

↓

[Optional: Triangulation]
  ├─ Vertices aus Surface-Definition
  └─ SPL interpoliert auf Vertices

↓

Mesh-Erstellung
  ├─ PolyData (trianguliert)
  └─ StructuredGrid (strukturiert)
```

---

## 8. Wichtige Erkenntnisse

1. **Einheitliche Datenquelle:** Alle Grids kommen aus `calculation_spl['surface_grids']`
2. **Unterschiedliche Verarbeitung:** Abhängig von Orientierung und Modus
3. **Flexible Struktur:** Unterstützt 2D-Grids und 1D-Achsen
4. **Optional Upscaling:** Für bessere Darstellung
5. **Fallback-Mechanismen:** Triangulation → Strukturiertes Grid

---

## 9. Potentielle Probleme

1. **Inkonsistente Datenstrukturen:** 
   - Grids können als Listen oder Arrays kommen
   - Prüfung: `isinstance(x_grid, list)` (Zeile 834)

2. **Fehlende Felder:**
   - `orientation` kann 'unknown' sein
   - `dominant_axis` kann None sein
   - Fallback-Logik vorhanden

3. **Shape-Mismatch:**
   - `X_grid`, `Y_grid`, `Z_grid` müssen gleiche Shape haben
   - `surface_mask` muss gleiche Shape haben
   - Prüfung: `X_grid.shape == Y_grid.shape == Z_grid.shape`
