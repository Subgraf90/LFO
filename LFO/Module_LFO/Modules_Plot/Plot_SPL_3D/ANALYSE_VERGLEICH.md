# Analyse: Vergleich Alt vs. Neu Plot-Modul

## Zusammenfassung
- **ALTES MODUL** (Plot3D.py → Plot3DSPL.py): Korrekte SPL-Werte, aber alte Plot-Flächen-Erzeugung
- **NEUES MODUL** (Plot3DSPL_neu.py): Moderne Plot-Flächen-Erzeugung, aber falsche Datenquelle

---

## WESENTLICHE UNTERSCHIEDE

### 1. DATENQUELLE

#### ALTES MODUL (Plot3DSPL.py):
- `update_spl_plot()` erhält Parameter:
  - `sound_field_x: Iterable[float]`
  - `sound_field_y: Iterable[float]`
  - `sound_field_pressure: Iterable[float]` (komplexe Werte)
- Verarbeitet diese Parameter direkt

#### NEUES MODUL (Plot3DSPL_neu.py):
- `update_spl_plot()` **IGNORIERT die Parameter komplett!**
- Liest Daten direkt aus `container.calculation_spl`:
  - `surface_grids_data = calc_spl.get("surface_grids", {})`
  - `surface_results_data = calc_spl.get("surface_results", {})`
- Verarbeitet jede Surface separat mit deren eigenen Grid-Daten

---

### 2. SPL-WERT-KONVERTIERUNG

#### ALTES MODUL (Plot3DSPL.py) - **KORREKT**:
```python
# Zeile 899: Konvertiert zu float (VERLIERT komplexe Werte!)
pressure = np.asarray(sound_field_pressure, dtype=float)

# Zeile 941: Aber dann wird abs gemacht (bei float sinnlos)
pressure_2d = np.nan_to_num(np.abs(pressure), nan=0.0, posinf=0.0, neginf=0.0)
pressure_2d = np.clip(pressure_2d, 1e-12, None)
spl_db = self.functions.mag2db(pressure_2d)  # Konvertiert zu dB
```
**⚠️ PROBLEM:** Konvertierung zu float verliert Imaginärteil, aber da die Daten bereits als Betrag kommen könnten, funktioniert es trotzdem.

#### NEUES MODUL (Plot3DSPL_neu.py) - **KORREKT**:
```python
# Zeile 986: Behält komplexe Werte!
sound_field_p_complex = np.array(result_data['sound_field_p'], dtype=complex)

# Zeile 1013-1015: Konvertiert korrekt
pressure_magnitude = np.abs(sound_field_p_complex)
pressure_magnitude = np.clip(pressure_magnitude, 1e-12, None)
spl_values = self.functions.mag2db(pressure_magnitude)
```
**✅ KORREKT:** Verwendet komplexe Werte korrekt

---

### 3. PLOT-FLÄCHEN-ERZEUGUNG

#### ALTES MODUL (Plot3DSPL.py):
- Verwendet `prepare_plot_geometry()` für Gesamt-Geometrie
- Verwendet `_render_surfaces_textured()` mit Textur-Rendering
- Verarbeitet alle Surfaces zusammen über ein gemeinsames Grid
- Verwendet `build_surface_mesh()` für Floor

#### NEUES MODUL (Plot3DSPL_neu.py):
- **Pro Surface separate Verarbeitung:**
  - Liest `X_grid`, `Y_grid`, `Z_grid` aus `surface_grids`
  - Liest `sound_field_p_complex` aus `surface_results`
  - Verwendet `surface_mask` für Punkte innerhalb der Surface
- **Upscaling:**
  - Interpoliert SPL-Werte auf feineres Grid (`PLOT_UPSCALE_FACTOR`)
  - Separate Interpolation für Color Step (nearest) vs. Gradient (bilinear)
- **Triangulation:**
  - Trianguliert Surface-Polygone
  - Interpoliert SPL-Werte auf Vertices
  - Filtert Dreiecke außerhalb der Surface

---

### 4. INTERPOLATION

#### ALTES MODUL:
- Verwendet `prepare_plot_geometry()` (interne Funktion)
- Interpoliert dann in `_process_planar_surface_texture()` mit `_bilinear_interpolate_grid()`

#### NEUES MODUL:
- **Color Step Mode:** Nearest Neighbor Interpolation (`_nearest_interpolate_grid()`)
- **Gradient Mode:** Bilineare Interpolation (`_bilinear_interpolate_grid()`)
- Interpoliert **vor** Quantisierung (bei Color Step)

---

### 5. QUANTISIERUNG (COLOR STEP MODE)

#### ALTES MODUL:
- Quantisiert in `_process_planar_surface_texture()` NACH Interpolation
- Verwendet `_quantize_to_steps()`

#### NEUES MODUL:
- **WICHTIG:** Quantisiert **VOR** Vertex-Interpolation bei Color Step!
- Zeile 1234-1236: `spl_values_clipped = np.clip(spl_values.ravel(), cbar_min, cbar_max)`
- Zeile 1236: `spl_orig_quantized = self._quantize_to_steps(spl_values_clipped, cbar_step)`
- Dann interpoliert mit **quantisierten Werten** (nearest neighbor)

---

### 6. GEOMETRIE-VERARBEITUNG

#### ALTES MODUL:
- Gemeinsames Grid für alle Surfaces
- Textur-Rendering mit 2D-Bildern pro Surface

#### NEUES MODUL:
- **Separate Grids pro Surface**
- Verwendet `X_grid`, `Y_grid`, `Z_grid` direkt aus Berechnung
- Triangulation der Surface-Polygone
- Strukturierte Grids für glatte Darstellung

---

## PROBLEM-IDENTIFIKATION

### Warum zeigt das neue Modul falsche SPL-Werte?

1. **Datenquelle:** Verwendet `calculation_spl['surface_results']` statt Parameter
2. **Mögliche Dateninkonsistenz:** Die Werte in `surface_results` könnten bereits verarbeitet/transformiert sein
3. **Surface-Maske:** Verwendet `surface_mask` um nur Punkte innerhalb der Surface zu plotten - aber die SPL-Werte werden möglicherweise falsch zugeordnet

### Warum funktioniert das alte Modul korrekt?

1. **Direkte Parameter:** Verwendet die übergebenen Parameter direkt
2. **Einfache Verarbeitung:** Keine separate Surface-Verarbeitung
3. **Bewährte Konvertierung:** `mag2db()` auf `abs()` von Druck-Werten

---

## LÖSUNGSSTRATEGIE

**Ziel:** Kombiniere das Beste aus beiden Modulen:
1. ✅ Korrekte SPL-Konvertierung aus altem Modul
2. ✅ Moderne Plot-Flächen-Erzeugung aus neuem Modul
3. ✅ Verwendung der übergebenen Parameter (nicht `calculation_spl` direkt)
4. ✅ Separate Surface-Verarbeitung mit korrekten SPL-Werten

### Umsetzung:
- Übernehme die SPL-Konvertierungs-Logik aus altem Modul (Zeilen 941-943 in Plot3DSPL.py)
- Übernehme die Plot-Flächen-Erzeugung aus neuem Modul (Triangulation, separate Surfaces)
- Verwende die **Parameter** `sound_field_x`, `sound_field_y`, `sound_field_pressure` statt `calculation_spl`
- Konvertiere SPL-Werte **vor** der Surface-Verarbeitung, nicht währenddessen

