# SPL-Berechnung pro Gridpunkt - Erklärung

## Übersicht

Die SPL-Berechnung (Sound Pressure Level) erfolgt für jeden Gridpunkt auf einer Surface durch Überlagerung der komplexen Wellenfelder aller aktiven Lautsprecherquellen.

---

## 1. GRUPPENSURFACES (Gruppen-Summen-Grids)

### Ablauf der Berechnung

#### Schritt 1: Gruppen-Identifikation
- Surfaces werden in Gruppen identifiziert über `_identify_group_candidates()`
- **Kriterien für Gruppierung:**
  - `spl_group_sum_enabled = True` (Einstellung aktiviert)
  - Mindestens `spl_group_min_surfaces` (Standard: 2) Surfaces pro Gruppe
  - Surfaces müssen nahe beieinander liegen (Abstand < `spl_group_max_distance` oder automatisch 2× resolution)
  - **Wichtig:** Vertikale Surfaces werden NICHT gruppiert (werden einzeln berechnet)

#### Schritt 2: Gemeinsames Grid erstellen
- Für jede Gruppe wird ein **gemeinsames Gruppen-Grid** erstellt via `generate_group_sum_grid()`
- Das Grid umfasst alle Surfaces der Gruppe in einem einheitlichen Koordinatensystem
- Grid-Parameter:
  - **Resolution:** Aus Settings (`settings.resolution`)
  - **Koordinaten:** X, Y, Z (3D-Grid)
  - **Maske:** Pro Surface eine eigene Maske innerhalb des Gruppen-Grids

#### Schritt 3: SPL-Berechnung pro Gridpunkt

Für **jeden Gridpunkt** im Gruppen-Grid wird folgende Berechnung durchgeführt:

**Für jede aktive Lautsprecherquelle (Array):**
1. **Geometrie-Berechnung:**
   - Distanz: `d = sqrt((X - x_source)² + (Y - y_source)² + (Z - z_source)²)`
   - Azimut: Berechnung aus Quellenposition und Gridpunkt-Position
   - Elevation: Winkel zwischen horizontaler Ebene und Quellenrichtung

2. **Polar Pattern Interpolation:**
   - Batch-Interpolation aus Balloon-Daten (`get_balloon_data_batch()`)
   - **Eingabe:** Azimut und Elevation für alle Gridpunkte gleichzeitig
   - **Ausgabe:** 
     - `polar_gains` (dB)
     - `polar_phases` (Grad)

3. **Wellenberechnung (`_compute_wave_for_points()`):**
   ```python
   # Konvertierung dB → linear
   magnitude_linear = 10^(polar_gains / 20)
   
   # Phase berechnen
   phase = wave_number × distance + polar_phase_rad + 2π × frequency × source_time
   
   # Komplexe Welle
   wave = (magnitude_linear × source_level × a_source_pa × exp(i × phase)) / distance
   
   # Polaritätsinvertierung (falls aktiviert)
   if polarity:
       wave = -wave
   ```

4. **Akkumulation:**
   - Alle Wellen werden komplex addiert: `sound_field_p += wave`
   - **Interferenz** erfolgt automatisch durch komplexe Addition (konstruktiv/destruktiv)

#### Schritt 4: Extraktion pro Surface
- Aus dem Gruppen-Grid wird für jede Surface der entsprechende Bereich extrahiert
- Verwendet wird die Surface-spezifische Maske innerhalb des Gruppen-Grids
- Ergebnis wird in `surface_results_data[surface_id]` gespeichert mit Flag `is_group_sum = True`

### Abhängigkeiten für GRUPPENSURFACES

#### 1. Surface-Eigenschaften
- **Geometrie:** Polygon-Punkte (X, Y, Z)
- **Gruppenzugehörigkeit:** `group_id` aus `settings.surface_groups`
- **Orientierung:** Planar/sloped (vertikale Surfaces werden nicht gruppiert)

#### 2. Grid-Parameter
- `settings.resolution` - Grid-Auflösung
- `spl_group_min_surfaces` - Mindestanzahl Surfaces pro Gruppe (Standard: 2)
- `spl_group_max_distance` - Maximaler Abstand zwischen Surfaces (oder automatisch 2× resolution)
- `spl_group_sum_enabled` - Ob Gruppen-Summen-Grids aktiviert sind

#### 3. Lautsprecherquellen (Arrays)
- **Position:** `source_position_x/y/z` oder `source_position_calc_x/y/z`
- **Azimut:** `source_azimuth` (Ausrichtung)
- **Zeit:** `source_time` (Verzögerung)
- **Level:** `source_level` (dB)
- **Gain:** `source_gain` (dB)
- **Delay:** `source_delay` (ms)
- **Polarität:** `source_polarity` (Invertierung)
- **Mute/Hide:** Überspringt stumm/versteckte Arrays

#### 4. Polar Pattern (Richtcharakteristik)
- **Balloon-Daten:** Geladen aus `data_container` für jeden Lautsprecher
- **Interpolation:** Azimut und Elevation → Gain (dB) und Phase (Grad)
- **Batch-Interpolation:** Alle Gridpunkte gleichzeitig (vektorisiert)

#### 5. Physikalische Konstanten
- `speed_of_sound` - Schallgeschwindigkeit (temperaturabhängig)
- `wave_number = 2π × frequency / speed_of_sound`
- `calculate_frequency` - Berechnungsfrequenz (Hz)
- `a_source_pa` - Referenz-Schalldruck (aus `a_source_db`)

#### 6. Grid-Generierung
- **Generator:** `FlexibleGridGenerator.generate_group_sum_grid()`
- **Padding:** Automatisch 5× resolution um Surfaces
- **Koordinaten:** X, Y, Z für alle Gridpunkte
- **Masken:** Pro Surface eine Maske innerhalb des Gruppen-Grids

---

## 2. EINZELNE SURFACES

### Ablauf der Berechnung

#### Schritt 1: Surface-Grid erstellen
- **Pro Surface** wird ein eigenes Grid erstellt via `generate_per_surface()`
- Grid-Parameter identisch wie bei Gruppen-Grids:
  - Resolution aus Settings
  - X, Y, Z Koordinaten
  - Surface-Maske

#### Schritt 2: SPL-Berechnung pro Gridpunkt

Für **jeden Gridpunkt** wird identisch wie bei Gruppen-Surfaces berechnet:

**Für jede aktive Lautsprecherquelle:**
1. **Geometrie-Berechnung:** Identisch zu Gruppen-Surfaces
2. **Polar Pattern Interpolation:** Identisch (Batch-Interpolation)
3. **Wellenberechnung:** Identisch (`_compute_wave_for_points()`)
4. **Akkumulation:** Identisch (komplexe Addition)

#### Unterschied zu Gruppen-Surfaces:
- **Separates Grid** pro Surface (nicht gemeinsames Grid)
- **Direkte Speicherung** in `surface_results_data[surface_id]` mit `is_group_sum = False`
- **Keine Extraktion** nötig (bereits auf Surface-Grid berechnet)

### Abhängigkeiten für EINZELNE SURFACES

#### 1. Surface-Eigenschaften
- **Geometrie:** Polygon-Punkte (X, Y, Z)
- **Orientierung:** Planar/sloped/vertical (alle Orientierungen unterstützt)
- **Enabled/Hidden:** Nur enabled und nicht-hidden Surfaces werden berechnet

#### 2. Grid-Parameter
- `settings.resolution` - Grid-Auflösung
- **Edge-Refinement:** Standard deaktiviert für konsistente Grids
- **Padding:** 5× resolution (identisch zu Gruppen-Grids)

#### 3. Lautsprecherquellen (Arrays)
- **Identisch zu Gruppen-Surfaces:**
  - Position, Azimut, Zeit, Level, Gain, Delay, Polarität
  - Mute/Hide-Filterung

#### 4. Polar Pattern (Richtcharakteristik)
- **Identisch zu Gruppen-Surfaces:**
  - Balloon-Daten
  - Batch-Interpolation (Azimut/Elevation → Gain/Phase)

#### 5. Physikalische Konstanten
- **Identisch zu Gruppen-Surfaces:**
  - `speed_of_sound`, `wave_number`, `calculate_frequency`, `a_source_pa`

#### 6. Grid-Generierung
- **Generator:** `FlexibleGridGenerator.generate_per_surface()`
- **Separates Grid** pro Surface (nicht gemeinsames Gruppen-Grid)

---

## Gemeinsame Wellenformel (für beide Fälle)

Die komplexe Welle für einen Gridpunkt berechnet sich als:

```
p(r) = (magnitude_linear × source_level × a_source_pa × exp(i × phase)) / distance
```

Mit:
- `magnitude_linear = 10^(polar_gain / 20)` (aus Polar Pattern)
- `phase = wave_number × distance + polar_phase_rad + 2π × frequency × source_time`
- `distance = sqrt((X - x_source)² + (Y - y_source)² + (Z - z_source)²)`

**SPL (dB)** wird später berechnet als:
```
SPL = 20 × log10(|p| / p_ref)
```
wobei `p_ref = 20e-6 Pa` (Referenz-Schalldruck)

---

## Wichtige Unterschiede

| Aspekt | Gruppensurfaces | Einzelne Surfaces |
|--------|----------------|-------------------|
| **Grid-Typ** | Gemeinsames Gruppen-Grid | Separates Grid pro Surface |
| **Berechnung** | Einmal für gesamtes Gruppen-Grid | Pro Surface einzeln |
| **Extraktion** | Ja (via Surface-Maske) | Nein (bereits auf Surface-Grid) |
| **Vertikale Surfaces** | Werden nicht gruppiert | Werden einzeln berechnet |
| **Performance** | Effizienter bei vielen ähnlichen Surfaces | Flexibler, aber mehr Berechnungen |

---

## Speicherung der Ergebnisse

Ergebnisse werden in `calculation_spl` gespeichert:

```python
calculation_spl = {
    'surface_grids': {
        surface_id: {
            'X_grid': [...],
            'Y_grid': [...],
            'Z_grid': [...],
            'surface_mask': [...],
            'resolution': float,
            'orientation': str
        }
    },
    'surface_results': {
        surface_id: {
            'sound_field_p': [...],  # Komplexe Druckwerte
            'is_group_sum': bool,     # True für Gruppen-Surfaces
            'group_id': str,          # Nur bei is_group_sum=True
            'group_X_grid': [...],    # Nur bei is_group_sum=True
            'group_Y_grid': [...],
            'group_Z_grid': [...],
            'group_mask': [...]       # Maske innerhalb Gruppen-Grid
        }
    }
}
```

