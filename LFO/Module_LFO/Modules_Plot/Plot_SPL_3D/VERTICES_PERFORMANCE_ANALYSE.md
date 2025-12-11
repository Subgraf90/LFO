# Performance-Analyse: Auswirkungen erhöhter Vertex-Anzahl

## Übersicht

Wenn wir die Anzahl der Vertices künstlich erhöhen (z.B. durch Subdivision der Dreiecke), hat das Auswirkungen auf verschiedene Bereiche des Rendering-Pipelines.

---

## Performance-Bottlenecks bei mehr Vertices

### 1. **Interpolation (griddata)** ⚠️ **KRITISCH**

**Aktueller Code (Plot3DSPL.py, Zeile 1021-1027):**
```python
spl_at_verts = griddata(
    points_orig,      # Grid-Positionen (N_grid Punkte)
    values_orig,      # Grid-SPL-Werte (N_grid Werte)
    points_new,       # Vertex-Positionen (N_vertices Punkte)
    method='nearest',
    fill_value=np.nan
)
```

**Komplexität:**
- **Grid-Punkte**: `N_grid` (z.B. 40.000)
- **Vertices**: `N_vertices` (aktuell ~6-12, bei Subdivision deutlich mehr)

**Performance:**
- `griddata` mit `method='nearest'` verwendet einen KD-Tree oder ähnliche Datenstruktur
- **Komplexität: O(N_grid × log(N_grid)) zum Aufbauen** + **O(N_vertices × log(N_grid)) zur Interpolation**
- Bei 10× mehr Vertices: **~10× langsamer** bei der Interpolation

**Beispiel:**
- Aktuell: 12 Vertices → ~1ms Interpolation
- Mit Subdivision: 1200 Vertices → ~100ms Interpolation
- Mit feinem Mesh: 12.000 Vertices → ~1s Interpolation

**⚠️ Das ist der größte Bottleneck!**

---

### 2. **Mesh-Erstellung (PyVista PolyData)** ✅ **Schnell**

**Code (Zeile 1176-1177):**
```python
mesh = pv.PolyData(triangulated_vertices, triangulated_faces)
mesh["plot_scalars"] = spl_at_verts
```

**Performance:**
- **Komplexität: O(N_vertices)**
- PyVista kopiert die Arrays (sehr schnell in NumPy/PyVista)
- Bei 10× mehr Vertices: **~10× mehr Speicher**, aber nur minimal langsamer (< 1ms)

**Vernachlässigbar** im Vergleich zur Interpolation.

---

### 3. **Rendering (PyVista/VTK)** ⚠️ **Mittel**

**Code (Zeile 1199-1209):**
```python
actor = self.plotter.add_mesh(
    mesh,
    scalars="plot_scalars",
    cmap=cmap_object,
    clim=(cbar_min, cbar_max),
    smooth_shading=not is_step_mode,
    interpolate_before_map=not is_step_mode,
)
```

**Performance-Abhängigkeiten:**

#### 3.1 Vertex-Anzahl
- **Komplexität: O(N_vertices)** für Vertex-Processing
- Bei 10× mehr Vertices: **~10× mehr Vertices** müssen verarbeitet werden
- **Impact: Mittel** - abhängig von GPU/CPU

#### 3.2 Face-Anzahl (Dreiecke)
- **Anzahl Faces**: `N_faces ≈ N_vertices / 3` (bei typischer Triangulation)
- **Rasterisierung**: Jedes Dreieck muss gerastert werden
- Bei 10× mehr Vertices: **~10× mehr Dreiecke** → **~10× mehr Render-Aufwand**

#### 3.3 Speicher für GPU
- **Vertex-Buffer**: `N_vertices × 3 (XYZ) × 4 bytes = 12 × N_vertices bytes`
- **Normal-Buffer**: `N_vertices × 3 (XYZ) × 4 bytes = 12 × N_vertices bytes`
- **Color-Buffer**: `N_vertices × 4 (RGBA) × 4 bytes = 16 × N_vertices bytes`
- **Total pro Vertex**: ~40 bytes

**Beispiel:**
- Aktuell: 12 Vertices → ~480 bytes
- Mit Subdivision: 1200 Vertices → ~48 KB
- Mit feinem Mesh: 12.000 Vertices → ~480 KB

**Vernachlässigbar** für moderne GPUs, aber bei vielen Surfaces summiert es sich.

#### 3.4 Interpolation zwischen Vertices
- **`interpolate_before_map=True`**: Zusätzliche GPU-Interpolation
- **`smooth_shading=True`**: Normal-Berechnung pro Vertex
- Bei 10× mehr Vertices: **~10× mehr Interpolations-Arbeit**

---

### 4. **Speicherverbrauch** ⚠️ **Bei vielen Surfaces relevant**

**Pro Surface:**
```
Speicher = 
  triangulated_vertices: N_vertices × 3 (XYZ) × 8 bytes (float64)
  triangulated_faces: N_faces × 4 (Format) × 8 bytes (int64)
  spl_at_verts: N_vertices × 8 bytes (float64)
  = N_vertices × (24 + 8) + N_faces × 32 bytes
  ≈ N_vertices × 40 bytes (approximativ)
```

**Beispiel mit 10 Surfaces:**
- Aktuell: 12 Vertices × 10 = 120 Vertices → ~4.8 KB
- Mit Subdivision: 1200 Vertices × 10 = 12.000 Vertices → ~480 KB
- Mit feinem Mesh: 12.000 Vertices × 10 = 120.000 Vertices → ~4.8 MB

**Bei 100 Surfaces:**
- Aktuell: ~48 KB
- Mit Subdivision: ~4.8 MB
- Mit feinem Mesh: ~48 MB

---

## Zusammenfassung: Performance-Impact

### Zeit-Komplexität (bei N× mehr Vertices):

| Phase | Aktuell (12 Vertices) | N× mehr Vertices | Impact |
|-------|----------------------|------------------|--------|
| **Interpolation (griddata)** | ~1ms | ~N× ms | ⚠️ **KRITISCH** |
| **Mesh-Erstellung** | < 1ms | < 1ms | ✅ **Vernachlässigbar** |
| **Rendering (Vertex-Processing)** | ~1ms | ~N× ms | ⚠️ **Mittel** |
| **Rendering (Face-Rasterisierung)** | ~1ms | ~N× ms | ⚠️ **Mittel** |
| **Total (10× mehr Vertices)** | ~3ms | ~30ms+ | ⚠️ **Signifikant** |

### Speicher-Komplexität:

| Anzahl Surfaces | Aktuell | 10× Vertices | 100× Vertices |
|-----------------|---------|--------------|---------------|
| 1 Surface | ~0.5 KB | ~5 KB | ~50 KB |
| 10 Surfaces | ~5 KB | ~50 KB | ~500 KB |
| 100 Surfaces | ~50 KB | ~500 KB | ~5 MB |

---

## Wann macht es Sinn, Vertices zu erhöhen?

### ✅ **Sinnvoll, wenn:**

1. **Wenige Surfaces** (< 10)
   - Performance-Impact ist überschaubar
   - Bessere Visualisierung

2. **Große, komplexe Polygone**
   - Nur 6-12 Vertices reichen nicht für glatte Kurven
   - Subdivision verbessert die Darstellung

3. **Grid-Auflösung ist niedrig**
   - Wenige Grid-Punkte → weniger Interpolations-Arbeit
   - Mehr Vertices helfen bei der Visualisierung

4. **Interpolation ist schnell genug**
   - `griddata` mit `method='nearest'` ist relativ schnell
   - Bis ~1000 Vertices ist es meist OK

### ❌ **Nicht sinnvoll, wenn:**

1. **Viele Surfaces** (> 50)
   - Performance summiert sich
   - Speicher wird knapp

2. **Grid-Auflösung ist sehr hoch**
   - Viele Grid-Punkte → Interpolation wird langsam
   - Mehr Vertices verschlimmern es nur

3. **Einfache Geometrien** (Rechtecke)
   - Nur 4 Ecken reichen für saubere Darstellung
   - Mehr Vertices bringen keinen Mehrwert

4. **Performance ist kritisch**
   - Interaktion muss flüssig sein
   - Jede Millisekunde zählt

---

## Empfehlungen

### **Aktuelle Implementierung (6-12 Vertices):**
✅ **Optimal für die meisten Fälle**
- Schnell
- Ausreichend für einfache Geometrien
- Gut skalierbar für viele Surfaces

### **Adaptive Subdivision:**
💡 **Idealer Kompromiss**
```python
# Pseudo-Code
def adaptive_subdivide(polygon_points, max_edge_length=0.1):
    """
    Subdividiert Polygon-Kanten, wenn sie zu lang sind.
    """
    if edge_length > max_edge_length:
        # Füge Mittelpunkt ein
        subdivided_points = subdivide_edge(...)
    return subdivided_points
```

**Vorteile:**
- Mehr Vertices nur wo nötig (lange Kanten)
- Weniger Vertices bei kleinen Polygonen
- Balance zwischen Performance und Qualität

### **Grid-basierte Vertices:**
💡 **Alternative: Verwendet Grid-Punkte direkt**
```python
# Statt Triangulation der Polygon-Ecken:
# Verwende Grid-Punkte innerhalb des Polygons als Vertices

mask_flat = surface_mask.ravel().astype(bool)
vertices_from_grid = np.column_stack([
    X_grid.ravel()[mask_flat],
    Y_grid.ravel()[mask_flat],
    Z_grid.ravel()[mask_flat]
])
spl_values = spl_values_2d.ravel()[mask_flat]
```

**Vorteile:**
- Keine Interpolation nötig (Grid-Punkte haben bereits SPL-Werte!)
- Sehr schnell
- Exakte Werte

**Nachteile:**
- Viele Vertices (alle Grid-Punkte)
- Große Meshes bei hoher Grid-Auflösung

---

## Fazit

**Aktueller Ansatz (6-12 Vertices) ist optimal für:**
- ✅ Viele Surfaces
- ✅ Performance-kritische Anwendungen
- ✅ Einfache Geometrien

**Mehr Vertices machen Sinn für:**
- ✅ Komplexe, gekrümmte Geometrien
- ✅ Wenige, große Surfaces
- ✅ Höhere visuelle Qualität (wenn Performance OK ist)

**Der größte Bottleneck ist die Interpolation (`griddata`), nicht das Rendering!**

Bei 10× mehr Vertices:
- Interpolation: **10× langsamer** ⚠️
- Rendering: **10× langsamer** ⚠️
- Speicher: **10× mehr** ⚠️
- **Total: ~10× langsamer** ⚠️

Daher: **Sparsam mit Vertices sein, nur wo wirklich nötig!**