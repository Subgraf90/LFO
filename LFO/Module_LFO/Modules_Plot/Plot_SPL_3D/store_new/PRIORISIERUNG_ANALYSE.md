# Priorisierung der Verarbeitungswege und Fallback-Mechanismen

## 🎯 Priorisierungs-Reihenfolge

### Hauptpfad (Priorität 1): Triangulation

**Code-Stelle:** Zeile 1167-1382

**Bedingung:**
```python
# 🎯 Dreiecks-Mesh mit glatten Kanten (immer aktiv)
mesh = None
try:
    from scipy.interpolate import griddata
    
    # Finde Surfaces innerhalb der Gruppe
    surfaces_to_triangulate = []
    surf_def = surface_definitions.get(surface_id)
    if surf_def is not None:
        surfaces_to_triangulate.append((surface_id, surf_def))
    else:
        # Suche in Gruppe...
    
    if surfaces_to_triangulate:
        # Triangulation durchführen
        tris = triangulate_points(pts)
        if tris:
            # Interpoliere SPL auf Vertices
            # Erstelle PolyData Mesh
            mesh = pv.PolyData(combined_verts, combined_faces)
```

**Priorität:** ⭐⭐⭐ **HÖCHSTE PRIORITÄT**

**Vorteile:**
- ✅ Glatte Kanten
- ✅ Exakte Surface-Form
- ✅ Beste Darstellung

**Erfolgsbedingungen:**
1. `surface_definitions.get(surface_id)` muss existieren
2. `pts = surf_data.get("points", [])` muss ≥ 3 Punkte haben
3. `triangulate_points(pts)` muss erfolgreich sein
4. `tris` muss nicht leer sein

**Wenn erfolgreich:**
- Mesh wird erstellt: `mesh = pv.PolyData(combined_verts, combined_faces)`
- Kein Fallback nötig

---

### Fallback 1 (Priorität 2): build_surface_mesh bei Exception

**Code-Stelle:** Zeile 1386-1427

**Bedingung:**
```python
except Exception as e:
    # Fallback auf Raster-Mesh nur bei Fehler
    if DEBUG_PLOT3D_TIMING:
        print(f"[DEBUG Plot Fallback] Verwende build_surface_mesh (Fallback)")
    scalars_for_mesh = np.clip(scalars, cbar_min, cbar_max)
    mesh = build_surface_mesh(
        x_plot,
        y_plot,
        scalars_for_mesh,
        z_coords=Z_plot,
        surface_mask=mask_plot,
        ...
    )
```

**Priorität:** ⭐⭐ **ZWEITE PRIORITÄT**

**Auslöser:**
- Exception während Triangulation
- `triangulate_points()` schlägt fehl
- `griddata()` schlägt fehl
- Andere Fehler im try-Block

**Vorteile:**
- ✅ Robust (funktioniert auch bei Fehlern)
- ✅ Strukturiertes Grid (regulär)
- ✅ Schneller als Triangulation

**Nachteile:**
- ⚠️ Geringere Qualität als Triangulation
- ⚠️ Reguläres Grid (nicht exakte Surface-Form)

---

### Fallback 2 (Priorität 3): build_surface_mesh wenn mesh is None

**Code-Stelle:** Zeile 1429-1445

**Bedingung:**
```python
if mesh is None:
    if DEBUG_PLOT3D_TIMING:
        print(f"[DEBUG Plot] ⚠️ Fallback auf Raster-Mesh für Surface '{surface_id}'")
    # Fallback wenn keine Triangulation möglich
    scalars_for_mesh = np.clip(scalars, cbar_min, cbar_max)
    mesh = build_surface_mesh(
        x_plot,
        y_plot,
        scalars_for_mesh,
        z_coords=Z_plot,
        surface_mask=mask_plot,
        ...
    )
```

**Priorität:** ⭐ **NIEDRIGSTE PRIORITÄT**

**Auslöser:**
- `mesh is None` nach Triangulation
- Keine Vertices gefunden: `if not all_verts:` (Zeile 1383)
- Triangulation erfolgreich, aber keine Vertices

**Vorteile:**
- ✅ Letzte Sicherheitsnetz
- ✅ Funktioniert immer (wenn Grid-Daten vorhanden)

**Nachteile:**
- ⚠️ Geringste Qualität
- ⚠️ Nur als letzter Ausweg

---

## 📊 Entscheidungsbaum

```
update_spl_plot()
  │
  ├─ [1] Lade Grid-Daten aus calculation_spl
  │   └─ surface_grids_data[surface_id]
  │       ├─ X_grid, Y_grid, Z_grid
  │       ├─ sound_field_x, sound_field_y
  │       └─ surface_mask
  │
  ├─ [2] Optional: Upscaling (PLOT_UPSCALE_FACTOR > 1)
  │   └─ Erstelle feineres Grid
  │
  ├─ [3] PRIORITÄT 1: Triangulation (immer versucht)
  │   ├─ Prüfe: surface_definitions.get(surface_id) existiert?
  │   ├─ Prüfe: pts >= 3?
  │   ├─ Versuche: triangulate_points(pts)
  │   ├─ Versuche: griddata() Interpolation
  │   └─ Erstelle: pv.PolyData(combined_verts, combined_faces)
  │   │
  │   ├─ ✅ ERFOLG → mesh erstellt
  │   │   └─ Verwende mesh
  │   │
  │   └─ ❌ FEHLER → Fallback 1
  │       │
  │       └─ [4] PRIORITÄT 2: build_surface_mesh (bei Exception)
  │           ├─ Versuche: build_surface_mesh()
  │           │
  │           ├─ ✅ ERFOLG → mesh erstellt
  │           │   └─ Verwende mesh
  │           │
  │           └─ ❌ FEHLER → mesh bleibt None
  │               │
  │               └─ [5] PRIORITÄT 3: build_surface_mesh (wenn mesh is None)
  │                   └─ Versuche: build_surface_mesh() erneut
  │
  └─ [6] Finale Prüfung
      ├─ if mesh is None or mesh.n_points == 0:
      │   └─ continue  # Überspringe Surface
      └─ else:
          └─ Rendere mesh
```

---

## 🔍 Detaillierte Priorisierung

### 1. Upscaling-Entscheidung

**Priorität:** Optional (abhängig von `PLOT_UPSCALE_FACTOR`)

**Code-Stelle:** Zeile 1044-1114

```python
if PLOT_UPSCALE_FACTOR > 1:
    # PRIORITÄT: Upscaling aktiv
    # Erstelle feineres Grid
    x_fine = np.linspace(...)
    y_fine = np.linspace(...)
    # Interpoliere auf feineres Grid
    X_plot = X_fine
    Y_plot = Y_fine
    spl_plot = spl_fine
else:
    # PRIORITÄT: Kein Upscaling
    # Verwende originale Grids
    X_plot = X_grid
    Y_plot = Y_grid
    spl_plot = spl_values
```

**Priorisierung:**
- ✅ **Wenn `PLOT_UPSCALE_FACTOR > 1`:** Upscaling wird verwendet
- ✅ **Wenn `PLOT_UPSCALE_FACTOR = 1`:** Originale Grids werden verwendet

---

### 2. Triangulation vs. Strukturiertes Grid

**Priorität:** Triangulation > Strukturiertes Grid

**Code-Stelle:** Zeile 1167-1445

**Reihenfolge:**
1. **Versuche Triangulation** (Zeile 1169-1382)
   - Prüfe: `surfaces_to_triangulate` nicht leer?
   - Prüfe: `triangulate_points(pts)` erfolgreich?
   - Prüfe: `tris` nicht leer?
   - Prüfe: `all_verts` nicht leer?
   - ✅ **Wenn erfolgreich:** `mesh = pv.PolyData(...)`

2. **Fallback bei Exception** (Zeile 1386-1427)
   - ❌ **Wenn Exception:** `build_surface_mesh()` im except-Block

3. **Fallback wenn mesh is None** (Zeile 1429-1445)
   - ❌ **Wenn mesh is None:** `build_surface_mesh()` erneut

---

### 3. Color Step vs. Gradient

**Priorität:** Abhängig von `colorization_mode`

**Code-Stelle:** Zeile 1117-1165

```python
if is_step_mode:
    # PRIORITÄT: Color Step
    scalars = self._quantize_to_steps(spl_plot, cbar_step)
else:
    # PRIORITÄT: Gradient
    scalars = spl_plot
```

**Priorisierung:**
- ✅ **Wenn `colorization_mode == "Color step"` und `cbar_step > 0`:** Quantisierung
- ✅ **Sonst:** Gradient (keine Quantisierung)

---

### 4. Interpolations-Methode

**Priorität:** Abhängig von `is_step_mode`

**Code-Stelle:** Zeile 1056-1075

```python
if is_step_mode:
    # PRIORITÄT: Nearest Neighbor (für harte Stufen)
    spl_fine = self._nearest_interpolate_grid(...)
else:
    # PRIORITÄT: Bilinear (für glatte Übergänge)
    spl_fine = self._bilinear_interpolate_grid(...)
```

**Priorisierung:**
- ✅ **Color Step:** Nearest Neighbor
- ✅ **Gradient:** Bilinear

---

### 5. Orientierung (Horizontal vs. Vertikal)

**Priorität:** Horizontal > Vertikal (separate Behandlung)

**Code-Stelle:** Zeile 947-954

```python
orientation = grid_data.get('orientation', 'unknown')

if orientation == 'vertical':
    # PRIORITÄT: Überspringe (wird separat behandelt)
    continue
else:
    # PRIORITÄT: Verarbeite horizontal
    # ... Haupt-Loop
```

**Priorisierung:**
- ✅ **Horizontale Surfaces:** Werden im Haupt-Loop verarbeitet
- ✅ **Vertikale Surfaces:** Werden in `_update_vertical_spl_surfaces_from_grids()` verarbeitet

---

## 📋 Zusammenfassung der Priorisierung

| Aspekt | Priorität 1 | Priorität 2 | Priorität 3 |
|--------|-------------|-------------|-------------|
| **Mesh-Erstellung** | Triangulation | build_surface_mesh (Exception) | build_surface_mesh (None) |
| **Upscaling** | Wenn `PLOT_UPSCALE_FACTOR > 1` | Originale Grids | - |
| **Interpolation** | Bilinear (Gradient) / Nearest (Step) | - | - |
| **Color-Modus** | Color Step (wenn aktiv) | Gradient | - |
| **Orientierung** | Horizontal (Haupt-Loop) | Vertikal (separate Funktion) | - |

---

## 🎯 Wichtigste Erkenntnisse

1. **Triangulation hat höchste Priorität**
   - Wird immer zuerst versucht
   - Beste Qualität
   - Fallback nur bei Fehlern

2. **Fallback-Mechanismen sind robust**
   - Mehrere Sicherheitsnetze
   - Funktioniert auch bei Fehlern
   - Garantiert, dass Surface geplottet wird (wenn möglich)

3. **Upscaling ist optional**
   - Abhängig von `PLOT_UPSCALE_FACTOR`
   - Standard: `PLOT_UPSCALE_FACTOR = 1` (kein Upscaling)

4. **Color-Modus beeinflusst Interpolation**
   - Color Step: Nearest Neighbor
   - Gradient: Bilinear

5. **Orientierung trennt Verarbeitung**
   - Horizontal: Haupt-Loop
   - Vertikal: Separate Funktion
