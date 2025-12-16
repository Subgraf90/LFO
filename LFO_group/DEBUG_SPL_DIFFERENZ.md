# Debugging: Unterschiedliche Werte zwischen Single Surface und Gruppensurface

## Problemstellung
Unterschiedliche SPL-Werte im Plot bei Verwendung von Single Surface vs. Gruppensurface.

## Strategie: Lokalisierung der Differenz

### Schritt 1: Vergleich der berechneten Werte (NACH Berechnung, VOR Plotting)

**Ort:** Nach der Berechnung, direkt bevor die Werte gespeichert werden

#### Für Gruppensurfaces:
Datei: `SoundfieldCalculator.py`, Zeile ~1555

```python
# Nach Zeile 1556, direkt nach der Extraktion:
surface_results_data[surface_id] = {
    "sound_field_p": surface_field_p.tolist(),
    # ...
}

# 🎯 DEBUG: Füge hier Vergleich hinzu
import numpy as np
surface_field_magnitude = np.abs(surface_field_p)
print(f"[DEBUG GROUP] Surface '{surface_id}':")
print(f"  Min: {np.min(surface_field_magnitude[mask]):.6f} Pa")
print(f"  Max: {np.max(surface_field_magnitude[mask]):.6f} Pa")
print(f"  Mean: {np.mean(surface_field_magnitude[mask]):.6f} Pa")
print(f"  Median: {np.median(surface_field_magnitude[mask]):.6f} Pa")
print(f"  Grid shape: {surface_field_p.shape}")
print(f"  Mask points: {np.sum(mask)}")
```

#### Für einzelne Surfaces:
Datei: `UISurfaceManager.py`, Zeile ~242

```python
# Nach Zeile 242, direkt nach der Berechnung:
self.container.calculation_spl['surface_results'][surface_id] = {
    'sound_field_p': np.array(sound_field_p_complex).tolist(),
    # ...
}

# 🎯 DEBUG: Füge hier Vergleich hinzu
import numpy as np
single_field_magnitude = np.abs(sound_field_p_complex)
print(f"[DEBUG SINGLE] Surface '{surface_id}':")
print(f"  Min: {np.min(single_field_magnitude):.6f} Pa")
print(f"  Max: {np.max(single_field_magnitude):.6f} Pa")
print(f"  Mean: {np.mean(single_field_magnitude):.6f} Pa")
print(f"  Median: {np.median(single_field_magnitude):.6f} Pa")
print(f"  Grid shape: {single_field_magnitude.shape}")
```

**Vergleich:**
- Wenn die Werte hier bereits unterschiedlich sind → **Problem liegt in der Berechnung**
- Wenn die Werte hier identisch sind → **Problem liegt im Plotting**

---

### Schritt 2: Vergleich der Werte beim Laden (IM Plot-Modul, VOR Interpolation)

**Ort:** `Plot3DSPL.py`, in `_combine_group_meshes()` oder `update_spl_plot()`

#### Für Gruppensurfaces:
Datei: `Plot3DSPL.py`, Zeile ~168-170

```python
# Nach dem Laden der Daten, vor Interpolation:
result_data = surface_results_data[sid]
sound_field_p_complex = np.array(result_data.get('sound_field_p', []), dtype=complex)

# 🎯 DEBUG: Füge hier Vergleich hinzu
spl_values_2d = np.abs(sound_field_p_complex)
print(f"[DEBUG PLOT GROUP] Surface '{sid}':")
print(f"  Min: {np.nanmin(spl_values_2d):.6f} Pa")
print(f"  Max: {np.nanmax(spl_values_2d):.6f} Pa")
print(f"  Mean: {np.nanmean(spl_values_2d):.6f} Pa")
print(f"  Shape: {spl_values_2d.shape}")
print(f"  Group mask available: {bool(result_data.get('group_mask'))}")
```

#### Für einzelne Surfaces:
Datei: `Plot3DSPL.py`, Zeile ~2345

```python
# Nach dem Laden, vor Verwendung:
result_data = surface_results_data[sid]
sound_field_p_complex = np.array(result_data.get('sound_field_p', []), dtype=complex)

# 🎯 DEBUG: Füge hier Vergleich hinzu
pressure_magnitude = np.abs(sound_field_p_complex)
print(f"[DEBUG PLOT SINGLE] Surface '{sid}':")
print(f"  Min: {np.nanmin(pressure_magnitude):.6f} Pa")
print(f"  Max: {np.nanmax(pressure_magnitude):.6f} Pa")
print(f"  Mean: {np.nanmean(pressure_magnitude):.6f} Pa")
print(f"  Shape: {pressure_magnitude.shape}")
```

**Vergleich:**
- Wenn hier bereits unterschiedlich → **Problem beim Speichern/Laden**
- Wenn hier identisch → **Problem bei der Interpolation/Visualisierung**

---

### Schritt 3: Vergleich nach Interpolation (NACH Interpolation auf Vertices)

**Ort:** `Plot3DSPL.py`, in `_combine_group_meshes()`, nach Zeile ~431 oder ~563

```python
# Nach griddata-Interpolation:
spl_at_verts = griddata(...)

# 🎯 DEBUG: Füge hier Vergleich hinzu
print(f"[DEBUG INTERPOLATION] Surface '{surface_id}':")
print(f"  Interpolated values - Min: {np.nanmin(spl_at_verts):.6f} dB")
print(f"  Interpolated values - Max: {np.nanmax(spl_at_verts):.6f} dB")
print(f"  Interpolated values - Mean: {np.nanmean(spl_at_verts):.6f} dB")
print(f"  NaN count: {np.sum(np.isnan(spl_at_verts))}")
print(f"  Valid vertices: {np.sum(~np.isnan(spl_at_verts))}")
```

**Vergleich:**
- Wenn hier unterschiedlich → **Problem bei der Interpolation**
- Wenn hier identisch → **Problem bei der Visualisierung/Color-Mapping**

---

### Schritt 4: Direkter Wertevergleich (Empfohlen - Schnellste Methode)

**Beste Methode:** Füge einen direkten Vergleich direkt nach beiden Berechnungen hinzu

#### Option A: In `SoundfieldCalculator.py` nach der Extraktion

```python
# Nach Zeile 1526 (nach Extraktion für Gruppensurface)
surface_field_p = np.zeros_like(group_field_p, dtype=complex)
surface_field_p[mask] = group_field_p[mask]

# 🎯 DEBUG: Speichere Vergleichsdaten
if surface_id == "TEST_SURFACE_ID":  # Ersetze mit echter Surface-ID
    group_values_for_comparison = {
        'magnitude': np.abs(surface_field_p[mask]),
        'shape': surface_field_p.shape,
        'mask_sum': np.sum(mask),
        'group_id': group_id
    }
```

#### Option B: In `UISurfaceManager.py` nach Single-Surface-Berechnung

```python
# Nach Zeile 242 (nach Single-Surface-Berechnung)
sound_field_p_complex = result['sound_field_p']

# 🎯 DEBUG: Vergleich mit Gruppensurface-Werten
if surface_id == "TEST_SURFACE_ID":  # Gleiche Surface-ID wie oben
    single_values_for_comparison = {
        'magnitude': np.abs(sound_field_p_complex),
        'shape': sound_field_p_complex.shape,
    }
    
    # Direkter Vergleich (falls beide vorhanden)
    if 'group_values_for_comparison' in globals():
        group_mag = group_values_for_comparison['magnitude']
        single_mag = single_values_for_comparison['magnitude']
        
        print(f"\n{'='*60}")
        print(f"[VERGLEICH] Surface '{surface_id}':")
        print(f"{'='*60}")
        print(f"Gruppensurface - Min: {np.min(group_mag):.6e} Pa, Mean: {np.mean(group_mag):.6e} Pa")
        print(f"Single Surface - Min: {np.min(single_mag):.6e} Pa, Mean: {np.mean(single_mag):.6e} Pa")
        print(f"Differenz - Min: {abs(np.min(group_mag) - np.min(single_mag)):.6e} Pa")
        print(f"Differenz - Mean: {abs(np.mean(group_mag) - np.mean(single_mag)):.6e} Pa")
        print(f"Relative Diff - Mean: {abs(np.mean(group_mag) - np.mean(single_mag)) / np.mean(single_mag) * 100:.4f}%")
        print(f"{'='*60}\n")
```

---

### Schritt 5: Grid-Vergleich (Wichtig - Könnte die Ursache sein!)

Die Grids könnten unterschiedlich sein! Prüfe:

```python
# In SoundfieldCalculator.py, nach Grid-Erstellung
# Für Gruppensurfaces:
X_group = group_grid["X"]
Y_group = group_grid["Y"]
Z_group = group_grid["Z"]
mask = surface_masks[surface_id]

print(f"[DEBUG GRID GROUP] Surface '{surface_id}':")
print(f"  Grid shape: {X_group.shape}")
print(f"  X range: [{np.min(X_group):.3f}, {np.max(X_group):.3f}]")
print(f"  Y range: [{np.min(Y_group):.3f}, {np.max(Y_group):.3f}]")
print(f"  Z range: [{np.min(Z_group):.3f}, {np.max(Z_group):.3f}]")
print(f"  Mask points: {np.sum(mask)}")

# In UISurfaceManager.py, nach Grid-Erstellung für Single Surface
print(f"[DEBUG GRID SINGLE] Surface '{surface_id}':")
print(f"  Grid shape: {grid.X_grid.shape}")
print(f"  X range: [{np.min(grid.X_grid):.3f}, {np.max(grid.X_grid):.3f}]")
print(f"  Y range: [{np.min(grid.Y_grid):.3f}, {np.max(grid.Y_grid):.3f}]")
print(f"  Z range: [{np.min(grid.Z_grid):.3f}, {np.max(grid.Z_grid):.3f}]")
print(f"  Mask points: {np.sum(grid.surface_mask)}")
```

**Wenn Grids unterschiedlich sind** → Das ist wahrscheinlich die Ursache!

---

## Empfohlene Debugging-Reihenfolge

1. **SCHNELLCHECK:** Füge direkten Vergleich hinzu (Schritt 4, Option B)
   - Zeigt sofort, ob die Differenz in der Berechnung liegt

2. **GRID-VERGLEICH:** Prüfe ob Grids identisch sind (Schritt 5)
   - Häufige Ursache für Unterschiede

3. **DETAILLIERTE VERGLEICHE:** Falls nötig, füge Logging an anderen Stellen hinzu
   - Schritte 1-3 für detaillierte Analyse

---

## Typische Ursachen für Unterschiede

### In der Berechnung:
- **Unterschiedliche Grids** (verschiedene Resolution, Padding, Koordinaten)
- **Unterschiedliche Masken** (Padding, Edge-Refinement)
- **Verschiedene Wellenberechnung** (wenn Code-Pfade unterschiedlich sind)

### Im Plotting:
- **Unterschiedliche Interpolation** (method='nearest' vs. method='linear')
- **Masken-Anwendung** (wann und wie wird die Maske angewendet)
- **Koordinaten-Umwandlung** (für vertikale Surfaces)

---

## Quick-Win: Einfachste Debug-Methode

Füge diesen Code in **beide** Berechnungs-Pfade ein (nach der Berechnung, vor dem Speichern):

```python
# Nach der Berechnung, vor dem Speichern
debug_surface_id = "YOUR_TEST_SURFACE_ID"  # Ersetze mit echter ID

if surface_id == debug_surface_id:
    import numpy as np
    magnitude = np.abs(sound_field_p)  # oder surface_field_p[mask] für Groups
    print(f"\n{'='*70}")
    print(f"[{['SINGLE', 'GROUP'][is_group_sum]} CALC] Surface '{surface_id}'")
    print(f"{'='*70}")
    print(f"Magnitude - Min: {np.min(magnitude):.10e} Pa")
    print(f"Magnitude - Max: {np.max(magnitude):.10e} Pa")
    print(f"Magnitude - Mean: {np.mean(magnitude):.10e} Pa")
    print(f"Shape: {magnitude.shape}")
    if hasattr(magnitude, 'mask') and np.any(magnitude.mask):
        print(f"Masked values: {np.sum(magnitude.mask)}")
    print(f"{'='*70}\n")
```

Dann vergleiche die Ausgaben in der Konsole!

