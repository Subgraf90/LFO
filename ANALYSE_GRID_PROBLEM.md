# Analyse: Grid-Erstellung, Berechnung und Plot

## Problem: Plot pro Surface ändert sich, wenn neue Surfaces enabled werden

### 1. GRID-ERSTELLUNG (`SurfaceGridCalculator.create_calculation_grid`)

**Aktuelles Verhalten:**
- Erstellt ein **EINZIGES Grid für ALLE enabled Surfaces zusammen**
- Bounding Box wird aus **ALLEN enabled Surfaces** berechnet (Zeile 118)
- Maske ist **kombiniert** (`combined_mask = combined_mask | surface_mask`) - True wenn Punkt in **MINDESTENS EINEM** Surface liegt

**Problem:**
- Wenn Surface A enabled ist → Grid für A wird erstellt
- Wenn dann Surface B enabled wird → **NEUES Grid für A+B** wird erstellt
- **Grid-Dimensionen ändern sich** → Bounding Box wird größer
- **Grid-Punkte ändern sich** → Neue Punkte werden hinzugefügt

**Code-Stellen:**
```python
# SurfaceGridCalculator.py, Zeile 117-124
if enabled_surfaces:
    min_x, max_x, min_y, max_y = self._calculate_bounding_box(enabled_surfaces)  # ALLE Surfaces
    # ...
    sound_field_x = np.arange(min_x, max_x + resolution, resolution)  # Grid ändert sich!
    sound_field_y = np.arange(min_y, max_y + resolution, resolution)  # Grid ändert sich!
```

### 2. BERECHNUNG (`SoundFieldCalculator._calculate_sound_field_complex`)

**Aktuelles Verhalten:**
- Verwendet das **kombinierte Grid** von `create_calculation_grid`
- Berechnet SPL-Werte für **alle Grid-Punkte** (nicht pro Surface isoliert)
- Wenn Grid sich ändert → **alle berechneten Werte ändern sich**

**Problem:**
- Berechnung verwendet das sich ändernde Grid
- Wenn Grid größer wird → Neue Punkte werden berechnet
- **Bereits berechnete Punkte können sich ändern**, weil Grid-Positionen sich verschieben können

**Code-Stellen:**
```python
# SoundfieldCalculator.py, Zeile 150-160
enabled_surfaces = self._get_enabled_surfaces()
(
    sound_field_x,
    sound_field_y,
    X_grid,
    Y_grid,
    Z_grid,
    surface_mask,  # Kombinierte Maske für ALLE Surfaces
) = self._grid_calculator.create_calculation_grid(enabled_surfaces)
```

### 3. PLOT (`SurfaceGeometryCalculator._build_surface_mesh_with_pyvista_sample`)

**Aktuelles Verhalten:**
- Versucht pro Surface zu plotten (Zeile 1204: "Interpoliere für jedes Surface EINZELN")
- ABER: Interpolation verwendet das **grobe Grid**, das sich ändert
- Maske im Plot ist auch **kombiniert** (Zeile 2110: `combined_mask |= mask`)

**Problem:**
- Interpolation verwendet `coarse_grid` (aus `source_x`, `source_y`, `source_scalars`)
- Wenn `source_x`/`source_y` sich ändern (weil Grid sich ändert) → **Interpolation ändert sich**
- **Werte für bereits enabled Surfaces ändern sich**, obwohl sie sich nicht ändern sollten

**Code-Stellen:**
```python
# SurfaceGeometryCalculator.py, Zeile 857-860
# Berechne Bounding-Box der enabled Surfaces
surface_min_x = min(p.get("x", 0.0) for _, pts in enabled_surfaces for p in pts)  # ALLE Surfaces
surface_max_x = max(p.get("x", 0.0) for _, pts in enabled_surfaces for p in pts)  # ALLE Surfaces
# ...

# Zeile 1234-1235: Interpolation verwendet grobes Grid
coarse_points = coarse_grid.points  # Dieses Grid ändert sich!
coarse_values = coarse_grid["spl_values"]  # Diese Werte ändern sich!
```

## ROOT CAUSE

**Das Hauptproblem ist:**
1. Grid-Erstellung ist **nicht pro Surface isoliert** → Grid ändert sich, wenn neue Surfaces enabled werden
2. Berechnung verwendet **kombiniertes Grid** → Werte ändern sich, wenn Grid sich ändert
3. Plot-Interpolation verwendet **sich änderndes Grid** → Plot ändert sich, obwohl Surface sich nicht ändert

## LÖSUNGSANSÄTZE

### Option 1: Pro-Surface Grid-Isolation (EMPFOHLEN)
- Jedes Surface bekommt sein **eigenes Grid**
- Berechnung pro Surface isoliert
- Plot pro Surface isoliert
- **Vorteil:** Jedes Surface ist komplett unabhängig
- **Nachteil:** Mehr Berechnungen, aber parallelisierbar

### Option 2: Fixe Grid-Dimensionen
- Grid-Dimensionen werden **einmalig festgelegt** (z.B. aus Settings)
- Grid ändert sich **NICHT**, wenn neue Surfaces enabled werden
- **Vorteil:** Einfach zu implementieren
- **Nachteil:** Kann zu großem Grid führen, wenn viele Surfaces vorhanden sind

### Option 3: Grid-Caching pro Surface-Kombination
- Cache Grid pro Surface-Kombination
- Wenn neue Surface enabled wird → Prüfe ob Grid bereits existiert
- **Vorteil:** Beste Performance
- **Nachteil:** Komplexe Cache-Verwaltung

## EMPFOHLENE LÖSUNG

**Option 1: Pro-Surface Grid-Isolation**

1. **Grid-Erstellung:** Jedes Surface bekommt sein eigenes Grid
2. **Berechnung:** Pro Surface isoliert
3. **Plot:** Pro Surface isoliert (bereits teilweise implementiert)

**Implementierung:**
- `create_calculation_grid` sollte pro Surface aufgerufen werden
- ODER: Neue Methode `create_surface_specific_grid(surface_id)` erstellen
- Berechnung sollte pro Surface isoliert sein
- Plot ist bereits pro Surface isoliert (nur Interpolation muss angepasst werden)

