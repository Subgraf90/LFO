# Analyse: PyVista Sample-Modus für SPL-Plot

## Zusammenfassung

Die Implementierung des `spl_plot_use_pyvista_sample`-Modus wurde analysiert. Es wurden einige kritische Probleme gefunden, die korrigiert werden müssen.

## Datenfluss-Analyse

### 1. Berechnung → Plot (SoundfieldCalculator → PlotSPL3D)

**Aktueller Datenfluss:**
1. `SoundfieldCalculator._calculate_sound_field_complex()` erstellt das Berechnungs-Grid:
   - `create_calculation_grid()` gibt `sound_field_x`, `sound_field_y` zurück
   - SPL-Werte werden in `sound_field_p` berechnet (komplex)
   - `sound_field_magnitude` wird zurückgegeben (nur Betrag)

2. `PlotSPL3D.update_spl_plot()` erhält:
   - `sound_field_x`, `sound_field_y` (Berechnungs-Grid-Koordinaten) ✅
   - `sound_field_pressure` (SPL-Werte aus Berechnung)

3. `update_spl_plot()` verarbeitet:
   - Konvertiert zu SPL/dB (Zeile 577)
   - **Clippt Werte** (Zeile 620-625) ⚠️
   - Kopiert `original_plot_values` **NACH** Clipping (Zeile 628) ⚠️
   - Ruft `prepare_plot_geometry()` auf (Zeile 631)

4. `prepare_plot_geometry()` gibt zurück:
   - `source_x`, `source_y`: Original-Berechnungs-Grid-Koordinaten ✅
   - `plot_x`, `plot_y`: Eventuell hochskalierte Plot-Koordinaten
   - `plot_values`: Eventuell resampelte Werte

5. `build_surface_mesh()` erhält:
   - `source_x`, `source_y`: ✅ Korrekt (Original-Berechnungs-Grid)
   - `source_scalars`: ⚠️ **Problem**: Geclippte Werte!

### 2. Problem 1: Clipping vor PyVista Sample

**Problem:**
- `original_plot_values` wird NACH dem Clipping kopiert (Zeile 628)
- Clipping ändert die Werte: `np.clip(plot_values, cbar_min - 20, cbar_max + 20)`
- Diese geclippten Werte werden dann für PyVista-Sampling verwendet
- **Folge**: Werte außerhalb des Colorbar-Bereichs gehen verloren, Interpolation wird verfälscht

**Lösung:**
- `original_plot_values` muss VOR dem Clipping kopiert werden
- Clipping sollte nur für die Visualisierung (Plot) erfolgen, nicht für Sampling

### 3. Problem 2: Shape-Konsistenz für PyVista Sampling

**Aktueller Code:**
```python
# SurfaceGeometryCalculator.py, Zeile 797
coarse_grid["spl_values"] = source_scalars_arr.ravel(order="C")
```

**Prüfung:**
- `source_scalars_arr` hat Shape `(len(source_y_arr), len(source_x_arr))`
- `ravel(order="C")` ist korrekt für NumPy → PyVista StructuredGrid
- **✅ Dieser Teil ist korrekt**

### 4. Problem 3: Grid-Erstellung pro Surface

**Aktueller Code:**
- Surfaces werden einzeln verarbeitet (Zeile 960 in SurfaceGeometryCalculator.py)
- Für jedes Surface wird ein feines Grid erstellt
- Surfaces werden dann kombiniert
- Interpolation erfolgt pro Surface mit Surface-spezifischer Maske

**Prüfung:**
- ✅ Surfaces werden einzeln verarbeitet
- ✅ Surface-spezifische Masken werden erstellt (Zeile 1306-1356)
- ✅ Nur Surface-relevante Punkte werden für Interpolation verwendet (Zeile 1359-1360)
- **✅ Dieser Teil ist korrekt**

### 5. Problem 4: Z-Koordinaten-Konsistenz

**Aktueller Code:**
```python
# SurfaceGeometryCalculator.py, Zeile 776-790
source_z = None
if container is not None and hasattr(container, "calculation_spl"):
    calc_spl = getattr(container, "calculation_spl", {}) or {}
    z_list = calc_spl.get("sound_field_z")
    # ...
```

**Prüfung:**
- Z-Koordinaten werden aus `calculation_spl['sound_field_z']` geholt
- Falls nicht vorhanden, wird `Z=0` verwendet
- **✅ Dieser Teil ist korrekt**

## Korrekturen

### Korrektur 1: Clipping vor Kopieren von original_plot_values

**Datei:** `LFO/Module_LFO/Modules_Plot/PlotSPL3D.py`

**Zeile 620-628 ändern:**

```python
# VORHER (falsch):
if time_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
elif phase_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
else:
    plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)

# Speichere ursprüngliche Berechnungs-Werte für PyVista sample-Modus
original_plot_values = plot_values.copy() if hasattr(plot_values, 'copy') else plot_values

# NACHHER (korrekt):
# Speichere ursprüngliche Berechnungs-Werte für PyVista sample-Modus (VOR Clipping!)
original_plot_values = plot_values.copy() if hasattr(plot_values, 'copy') else plot_values

# Clipping nur für Visualisierung (nicht für Sampling)
if time_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
elif phase_mode:
    plot_values = np.clip(plot_values, cbar_min, cbar_max)
else:
    plot_values = np.clip(plot_values, cbar_min - 20, cbar_max + 20)
```

## Zusammenfassung der Probleme

| Problem | Kritikalität | Status |
|---------|--------------|--------|
| Clipping vor Kopieren von `original_plot_values` | **HOCH** | ✅ **BEHOBEN** |
| Shape-Validierung für `original_plot_values` | Mittel | ✅ **BEHOBEN** |
| Shape-Konsistenz für PyVista Sampling | Niedrig | ✅ **BEHOBEN** (Verbesserte Validierung) |
| Grid-Erstellung pro Surface | Niedrig | ✅ Korrekt |
| Z-Koordinaten-Konsistenz | Niedrig | ✅ Korrekt |
| Surface-spezifische Interpolation | Niedrig | ✅ Korrekt |

## Durchgeführte Korrekturen

### ✅ Korrektur 1: Clipping vor Kopieren (BEHOBEN)
**Datei:** `LFO/Module_LFO/Modules_Plot/PlotSPL3D.py`  
**Zeile:** 620-630  
**Status:** ✅ Implementiert

### ✅ Korrektur 2: Shape-Validierung für original_plot_values (BEHOBEN)
**Datei:** `LFO/Module_LFO/Modules_Plot/PlotSPL3D.py`  
**Zeile:** 625-629, 695-701  
**Status:** ✅ Implementiert - Validierung hinzugefügt

### ✅ Korrektur 3: Verbesserte Fehlermeldungen in SurfaceGeometryCalculator (BEHOBEN)
**Datei:** `LFO/Module_LFO/Modules_Calculate/SurfaceGeometryCalculator.py`  
**Zeile:** 770-785  
**Status:** ✅ Implementiert - Detaillierte Fehlermeldungen

## Finale Prüfung

Alle kritischen Probleme wurden behoben:
1. ✅ `original_plot_values` wird VOR dem Clipping kopiert
2. ✅ Shape-Validierungen wurden hinzugefügt
3. ✅ Verbesserte Fehlermeldungen für besseres Debugging
4. ✅ Grid-Erstellung pro Surface funktioniert korrekt
5. ✅ Surface-spezifische Interpolation funktioniert korrekt

