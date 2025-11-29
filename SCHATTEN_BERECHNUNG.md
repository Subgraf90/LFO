# Schattenberechnung für Schallausbreitung

## Übersicht

Das Schattenberechnungssystem identifiziert Punkte, die von Hindernissen (Surfaces) abgeschattet sind, wenn Schall sich von Quellen zu Empfangspunkten ausbreitet.

## Physikalische Grundlagen

### Superposition (Ray-Tracing)
- **Verhalten**: Schall breitet sich geradlinig aus (wie Licht)
- **Schatten**: Vollständiger Schatten, keine Beugung
- **Anwendung**: Schatten-Maske wird **angewendet** - Punkte im Schatten werden **nicht berechnet**

### FEM/FDTD (Numerische Methoden)
- **Verhalten**: Beugung (Diffraktion) wird automatisch berücksichtigt
- **Schatten**: Schall kann um Hindernisse herum gebeugt werden
- **Anwendung**: Schatten-Maske wird **nicht angewendet** - alle Punkte werden berechnet
- **Nutzung**: Schatten-Maske kann für Visualisierung/Optimierung verwendet werden

## Implementierung

### ShadowCalculator

Das Modul `ShadowCalculator` berechnet Schatten-Masken basierend auf Ray-Tracing:

```python
from Module_LFO.Modules_Calculate.ShadowCalculator import ShadowCalculator

shadow_calc = ShadowCalculator(settings)
shadow_mask = shadow_calc.compute_shadow_mask(
    grid_points,        # (N, 3) Array von 3D-Punkten
    source_positions,   # Liste von (x, y, z) Tupeln
    enabled_surfaces   # Liste von SurfaceDefinition
)
```

### Integration in SoundfieldCalculator (Superposition)

Die Schattenberechnung ist bereits integriert:

1. **Automatische Berechnung**: Wird bei aktivierten Surfaces durchgeführt
2. **Anwendung**: Punkte im Schatten werden von der Berechnung ausgeschlossen
3. **Speicherung**: Schatten-Maske wird in `calculation_spl['shadow_mask']` gespeichert

**Einstellung**:
- `settings.enable_shadow_calculation = True/False` (Standard: True)

### Integration in FEM/FDTD (Zukünftig)

Für FEM/FDTD sollte die Schatten-Maske **nicht** angewendet werden, aber für Visualisierung genutzt werden:

```python
# In SoundFieldCalculatorFEM oder SoundFieldCalculatorFDTD:

# 1. Berechne Schatten-Maske (optional, für Visualisierung)
if self._shadow_calculator and enabled_surfaces:
    shadow_mask = self._shadow_calculator.compute_shadow_mask(
        grid_points, source_positions, enabled_surfaces
    )
    # Speichere für Visualisierung
    self.calculation_spl['shadow_mask'] = shadow_mask
    
# 2. Berechnung läuft NORMAL (alle Punkte werden berechnet)
# → Beugung wird durch FEM/FDTD automatisch berücksichtigt

# 3. Optional: Nutze Schatten-Maske für Optimierung
# → Niedrigere Auflösung in Schattenbereichen
# → Initialisierung mit niedrigeren Werten
```

## Performance-Optimierungen

1. **Batch-Verarbeitung**: `compute_shadow_mask_optimized()` für große Punktmengen
2. **Caching**: Hindernis-Meshes werden gecacht
3. **Early Exit**: Bei ersten Schnittpunkt abbrechen
4. **Optional**: Parallelisierung für mehrere Quellen (zukünftig)

## Debugging

Aktiviere Debug-Ausgaben:
```bash
export LFO_DEBUG_SHADOW=1
```

## Verwendung in Plots

Die Schatten-Maske kann in PyVista-Plots verwendet werden:

```python
# In PlotSPL3D.py:
shadow_mask = calculation_spl.get('shadow_mask')
if shadow_mask is not None:
    shadow_mask_2d = np.array(shadow_mask)
    # Setze Werte im Schatten auf NaN oder spezielle Farbe
    spl_values[shadow_mask_2d] = np.nan
```

## Zusammenfassung

| Methode | Schatten anwenden? | Grund |
|---------|-------------------|-------|
| **Superposition** | ✅ **Ja** | Keine Beugung, direkter Schatten |
| **FEM** | ❌ **Nein** | Beugung wird berücksichtigt |
| **FDTD** | ❌ **Nein** | Beugung wird berücksichtigt |

Die Schatten-Maske wird immer berechnet und gespeichert, aber nur bei Superposition **angewendet**.

