# Optionen für schärfere Plots ohne Performance-Verlust

## Problem
Aktuell verschwimmen die Farbübergänge im Plot. Erhöhung der Subdivision verbessert zwar die Schärfe, verschlechtert aber die Performance erheblich.

## Verfügbare Lösungen (in Reihenfolge der Empfehlung)

### 1. ✅ **Grid-Resolution erhöhen** (BESTE OPTION - Kein Performance-Verlust beim Rendering)
- **Was:** Reduziere `settings.resolution` (z.B. von 1.0 auf 0.5 oder 0.25)
- **Warum:** Mehr Grid-Punkte = mehr Vertices = schärfere Plots ohne Subdivision nötig
- **Performance:** Berechnung langsamer, aber Rendering gleich schnell (gleiche Anzahl Polygone)
- **Anpassung:** In Settings-UI oder direkt: `self.settings.resolution = 0.5`

### 2. ✅ **Render-Auflösung erhöhen** (Multi-Sample Anti-Aliasing)
- **Was:** In `Plot3D.py` Zeile 56: `PYVISTA_AA_MODE = "msaa"` oder `"fxaa"`
- **Warum:** Besseres Anti-Aliasing = schärfere Kanten
- **Performance:** Minimaler Impact
- **Optionen:**
  - `"ssaa"` = Supersampling (sehr langsam)
  - `"msaa"` = Multi-Sampling (empfohlen)
  - `"fxaa"` = Fast Approximate (schnellste)

### 3. ⚠️ **Subdivision Level 1** (AKTUELL AKTIV)
- **Was:** `PLOT_SUBDIVISION_LEVEL = 1` in `Plot3DSPL.py`
- **Warum:** 4x mehr Polygone = schärfere Übergänge
- **Performance:** Mittlerer Impact (4x mehr Faces)
- **Status:** ✅ Aktiv, guter Kompromiss

### 4. ❌ **Subdivision Level 2+** (NICHT EMPFOHLEN)
- **Problem:** 16x+ mehr Polygone = sehr langsam
- **Nur verwenden:** Bei sehr kleinen Flächen oder für Screenshots

### 5. 🔄 **Texture-basiertes Rendering** (EXPERIMENTELL)
- **Was:** Texturen statt Polygone für SPL-Werte
- **Warum:** Texturen können schärfer sein bei gleicher Polygon-Anzahl
- **Performance:** Mittel (Textur-Erstellung benötigt Zeit)
- **Status:** Verfügbar, aber aktuell deaktiviert

## Empfohlene Kombination

Für schärfere Plots ohne Performance-Verlust:

1. **Grid-Resolution reduzieren** auf 0.5-0.25 m (mehr Grid-Punkte)
2. **Subdivision Level 1** beibehalten (4x Polygone)
3. **AA-Mode auf "msaa"** setzen (besseres Anti-Aliasing)

Das gibt die beste Balance zwischen Schärfe und Performance.

## Code-Änderungen

### Grid-Resolution:
```python
# In Settings oder Code:
self.settings.resolution = 0.5  # Statt 1.0
```

### AA-Mode:
```python
# In Plot3D.py Zeile 56:
PYVISTA_AA_MODE = "msaa"  # Statt "ssaa"
```

### Subdivision:
```python
# In Plot3DSPL.py Zeile 44:
PLOT_SUBDIVISION_LEVEL = 1  # 0-3 möglich
```
