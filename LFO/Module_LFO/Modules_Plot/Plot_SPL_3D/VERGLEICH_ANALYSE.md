# Vergleichsanalyse: Plot3DSPL.py vs Plot3DSPL_new.py

## 1. Datenentgegennahme

### ✅ Identische Funktionssignatur

Beide Module haben **identische** Funktionssignaturen für `update_spl_plot`:

```python
def update_spl_plot(
    self,
    sound_field_x: Iterable[float],
    sound_field_y: Iterable[float],
    sound_field_pressure: Iterable[float],
    colorization_mode: str = "Gradient",
):
```

### ⚠️ Unterschiedliche Verwendung der Eingabeparameter

**Plot3DSPL.py:**
- Verwendet die globalen `sound_field_x`, `sound_field_y`, `sound_field_pressure` Parameter
- Verarbeitet diese zu `plot_values` (SPL in dB)
- Erstellt globale Plot-Geometrie mit `prepare_plot_geometry()`
- **Zusätzlich**: Erstellt `surface_overrides` aus `calculation_spl['surface_grids']` und `calculation_spl['surface_results']`

**Plot3DSPL_new.py:**
- **Ignoriert** die globalen `sound_field_x`, `sound_field_y`, `sound_field_pressure` Parameter komplett
- Verwendet **nur** Daten aus `calculation_spl['surface_grids']` und `calculation_spl['surface_results']`
- Keine globale Plot-Geometrie mehr
- Verarbeitet jede Surface einzeln mit ihren eigenen Grid-Daten

---

## 2. Hauptunterschiede in der Plot-Erstellung

### Plot3DSPL.py: Texture-basiertes Rendering mit Overrides

**Ablauf:**
1. Verarbeitet globale SPL-Daten (`sound_field_x/y/pressure`)
2. Erstellt globale Plot-Geometrie (`prepare_plot_geometry()`)
3. Erstellt `surface_overrides` Dictionary:
   ```python
   surface_overrides[sid] = {
       "source_x": gx,  # Aus surface_grids_data
       "source_y": gy,  # Aus surface_grids_data
       "values": spl_values_db,  # Direkt aus surface_results_data
   }
   ```
4. Ruft `_render_surfaces_textured()` auf:
   - Erstellt 2D-Texturen für jede Surface
   - Interpoliert SPL-Werte auf Textur-Positionen
   - Verwendet bilineare oder nearest-neighbor Interpolation
   - Rendert Texturen auf flachen StructuredGrids

**Vorteile:**
- Einheitliche globale Plot-Geometrie
- Textur-Rendering ist performant
- Cache-Mechanismus für Texturen vorhanden

**Nachteile:**
- Zusätzliche Interpolation-Schicht (Textur-Grid → Surface-Positionen)
- Komplexere Datenpipeline (globale Daten + Overrides)

---

### Plot3DSPL_new.py: Direktes Mesh-Rendering pro Surface

**Ablauf:**
1. **Ignoriert** globale SPL-Daten komplett
2. Iteriert direkt über `surface_grids_data.keys()`
3. Für jede Surface:
   - Lädt Grid-Daten (`X_grid`, `Y_grid`, `Z_grid`, `surface_mask`)
   - Lädt SPL-Werte direkt aus `surface_results_data[surface_id]['sound_field_p']`
   - Konvertiert zu SPL in dB
   - Optional: Upscaling mit `PLOT_UPSCALE_FACTOR`
   - Erstellt direktes Mesh mit `triangulate_points()` oder strukturiertem Grid
   - Rendert Mesh direkt im 3D-Plot

**Vorteile:**
- Keine zusätzliche Interpolation nötig
- Direkte Verwendung der berechneten Grid-Punkte
- Einfacherer Datenfluss
- Bessere Kontrolle über jede Surface einzeln

**Nachteile:**
- Keine globale Plot-Geometrie mehr
- Kein Texture-Caching (jedes Mesh wird neu erstellt)
- Potentiell mehr Render-Aufwand bei vielen Surfaces

---

## 3. Detaillierte Unterschiede

### 3.1 Datenquelle

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| Globale Parameter | ✅ Verwendet | ❌ Ignoriert |
| `surface_grids_data` | ✅ Für Overrides | ✅ Hauptdatenquelle |
| `surface_results_data` | ✅ Für Overrides | ✅ Hauptdatenquelle |
| Interpolation | ✅ Textur-Grid → Surface | ❌ Direkt (optional Upscaling) |

### 3.2 Rendering-Methode

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| Methode | Texture-Rendering | Direktes Mesh-Rendering |
| Funktion | `_render_surfaces_textured()` | Direkt in `update_spl_plot()` |
| Mesh-Typ | StructuredGrid (flach) | Trianguliertes Mesh oder StructuredGrid |
| Cache | ✅ Textur-Cache | ❌ Kein Cache |

### 3.3 Upscaling

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| Upscaling | Via `prepare_plot_geometry()` | Via `PLOT_UPSCALE_FACTOR` |
| Faktor | `UPSCALE_FACTOR` (aus Settings) | `PLOT_UPSCALE_FACTOR` (Environment-Variable) |
| Interpolation | Bilinear/Nearest (je nach Modus) | Bilinear/Nearest (je nach Modus) |

### 3.4 Color Step Modus

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| Quantisierung | ✅ `_quantize_to_steps()` | ✅ `_quantize_to_steps()` |
| Interpolation | Nearest Neighbor für Texturen | Nearest Neighbor für Upscaling |
| Colormap | Standard Colormap | Diskrete Colormap mit `ListedColormap` |

### 3.5 Vertikale Surfaces

| Aspekt | Plot3DSPL.py | Plot3DSPL_new.py |
|--------|--------------|------------------|
| Behandlung | `_update_vertical_spl_surfaces()` | `_update_vertical_spl_surfaces_from_grids()` |
| Datenquelle | `surface_samples` Payloads | Direkt aus `surface_grids_data` |

---

## 4. Code-Struktur

### Plot3DSPL.py
```
update_spl_plot()
  ├─ Verarbeitet globale Daten
  ├─ prepare_plot_geometry() → globale Geometrie
  ├─ Erstellt surface_overrides
  └─ _render_surfaces_textured()
      ├─ _process_single_surface_texture() (pro Surface)
      │   ├─ Erstellt Textur-Grid
      │   ├─ Interpoliert SPL-Werte
      │   └─ Erstellt PyVista Texture
      └─ Rendert Texturen auf StructuredGrids
```

### Plot3DSPL_new.py
```
update_spl_plot()
  ├─ Lädt surface_grids_data und surface_results_data
  └─ Loop über jede Surface:
      ├─ Lädt Grid-Daten
      ├─ Lädt SPL-Werte
      ├─ Optional: Upscaling
      ├─ Erstellt Mesh (trianguliert oder strukturiert)
      └─ Rendert Mesh direkt
```

---

## 5. Zusammenfassung

### ✅ Datenentgegennahme: IDENTISCH
- Beide Module haben identische Funktionssignaturen
- **Aber**: Plot3DSPL_new.py ignoriert die globalen Parameter komplett

### 🔴 Große Unterschiede in der Plot-Erstellung:

1. **Datenquelle:**
   - Plot3DSPL.py: Globale Daten + Overrides
   - Plot3DSPL_new.py: Nur Surface-spezifische Daten

2. **Rendering-Methode:**
   - Plot3DSPL.py: Texture-basiert (2D-Texturen auf flachen Grids)
   - Plot3DSPL_new.py: Direktes Mesh-Rendering (3D-Meshes)

3. **Interpolation:**
   - Plot3DSPL.py: Textur-Grid → Surface-Positionen
   - Plot3DSPL_new.py: Optional Upscaling, sonst direkt

4. **Komplexität:**
   - Plot3DSPL.py: Mehrschichtige Pipeline (globale Daten → Overrides → Texturen)
   - Plot3DSPL_new.py: Direkter Pfad (Grid-Daten → Mesh → Render)

---

## 6. Empfehlungen

**Plot3DSPL_new.py scheint der modernere Ansatz zu sein:**
- Direkter Zugriff auf berechnete Daten
- Keine unnötige Interpolation
- Einfacherer Datenfluss
- Bessere Kontrolle pro Surface

**Plot3DSPL.py könnte Vorteile haben bei:**
- Performance (Textur-Caching)
- Einheitlicher globaler Plot
- Kompatibilität mit älteren Code-Pfaden
