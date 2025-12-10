# Vergleich: Datenübergabe alter vs. neuer SoundFieldCalculator

## Übersicht

Dieses Dokument vergleicht, wie der **alte** (`SoundfieldCalculator.py`) und der **neue** (`SoundfieldCalculator_neu.py`) Calculator die berechneten Werte in `calculation_spl` speichern.

---

## 🎯 Kernunterschiede

### **ALTER CALCULATOR** (`SoundfieldCalculator.py`)

#### Datenstruktur in `calculation_spl`:

```python
calculation_spl = {
    # ✅ Globale/gepoolte Werte (EIN Grid für ALLE Surfaces)
    "sound_field_p": [...],      # 2D-Array: Magnitude des kombinierten Schallfelds
    "sound_field_x": [...],      # 1D-Array: X-Koordinaten
    "sound_field_y": [...],      # 1D-Array: Y-Koordinaten
    "sound_field_z": [...],      # 2D-Array: Z-Koordinaten (kombiniert)
    
    # ✅ Surface-Masken (kombiniert)
    "surface_mask": [...],       # 2D-Boolean: Erweiterte Maske (für Berechnung)
    "surface_mask_strict": [...], # 2D-Boolean: Strikte Maske (für Plot)
    
    # ✅ Surface-Daten (pro Surface, aber in separaten Strukturen)
    "surface_meshes": [...],     # Liste von Mesh-Objekten (Geometrie)
    "surface_samples": [...],    # Liste von Sampling-Punkten (Koordinaten)
    "surface_fields": {          # Dict: {surface_id: komplexe Feldwerte}
        "surface_1": [...],
        "surface_2": [...],
    },
}
```

#### Berechnungslogik:

1. **EIN gemeinsames Grid** wird für alle enabled Surfaces erstellt
2. **Alle Surfaces werden zusammen** berechnet (gleicher Grid-Index-Raum)
3. **Separate Puffer** (`surface_field_buffers`, `surface_point_buffers`) speichern pro-Surface-Ergebnisse
4. **Kombinierte Rückgabe** als globales Schallfeld

**Code-Stelle:**
```python
# Zeile 130-599: _calculate_sound_field_complex()
# - Erstellt EIN Grid für alle Surfaces (create_calculation_grid)
# - Berechnet alle Surfaces zusammen
# - Speichert separate surface_fields pro Surface
```

---

### **NEUER CALCULATOR** (`SoundfieldCalculator_neu.py`)

#### Datenstruktur in `calculation_spl`:

```python
calculation_spl = {
    # ✅ Globale/gepoolte Werte (KOMBINIERT für Rückwärtskompatibilität)
    "sound_field_p": [...],      # 2D-Array: Interpoliert/kombiniert aus allen Surfaces
    "sound_field_x": [...],      # 1D-Array: Kombinierte X-Koordinaten
    "sound_field_y": [...],      # 1D-Array: Kombinierte Y-Koordinaten
    "sound_field_z": [...],      # 2D-Array: Z_grid_combined (interpoliert)
    
    # ✅ Surface-Masken (kombiniert)
    "surface_mask": [...],       # 2D-Boolean: Kombinierte Maske
    "surface_mask_strict": [...], # 2D-Boolean: Kombinierte strikte Maske
    
    # 🎯 NEU: Pro-Surface-Daten (getrennte Grids!)
    "surface_grids": {           # Dict: {surface_id: Grid-Daten}
        "surface_1": {
            "sound_field_x": [...],    # 1D-Array: X-Koordinaten (nur diese Surface)
            "sound_field_y": [...],    # 1D-Array: Y-Koordinaten (nur diese Surface)
            "X_grid": [...],           # 2D-Array: Meshgrid X
            "Y_grid": [...],           # 2D-Array: Meshgrid Y
            "Z_grid": [...],           # 2D-Array: Z-Koordinaten
            "surface_mask": [...],     # 2D-Boolean: Maske für diese Surface
            "resolution": 0.5,         # Float: Grid-Auflösung
            "orientation": "planar",   # String: "planar" oder "vertical" 🎯 NEU
            "dominant_axis": None,     # Optional: "x", "y", "z" 🎯 NEU
        },
        "surface_2": { ... },
    },
    
    "surface_results": {         # Dict: {surface_id: Berechnungsergebnisse}
        "surface_1": {
            "sound_field_p": [...],        # 2D-Array: Komplexe Druckwerte
            "sound_field_p_magnitude": [...], # 2D-Array: Magnitude (|p|)
        },
        "surface_2": { ... },
    },
    
    # ⚠️ VERALTET (nicht mehr verwendet, aber für Kompatibilität leer):
    # "surface_meshes": [],
    # "surface_samples": [],
    # "surface_fields": {},
}
```

#### Berechnungslogik:

1. **Separate Grids** werden pro Surface (oder Gruppe) erstellt
2. **Jede Surface wird einzeln** berechnet (eigener Grid-Index-Raum)
3. **Pro-Surface-Ergebnisse** werden in `surface_results` gespeichert
4. **Kombinierte Rückgabe** wird für Rückwärtskompatibilität interpoliert

**Code-Stelle:**
```python
# Zeile 137-513: _calculate_sound_field_complex()
# - Erstellt PRO SURFACE ein eigenes Grid (generate_per_group)
# - Berechnet jede Surface einzeln (_calculate_sound_field_for_surface_grid)
# - Speichert separate surface_grids + surface_results
# - Kombiniert für Rückwärtskompatibilität
```

---

## 📊 Detaillierter Vergleich

### 1. Grid-Erstellung

| Aspekt | ALTER Calculator | NEUER Calculator |
|--------|------------------|------------------|
| **Methode** | `SurfaceGridCalculator.create_calculation_grid()` | `FlexibleGridGenerator.generate_per_group()` |
| **Anzahl Grids** | **1 gemeinsames Grid** für alle Surfaces | **N Grids** (ein Grid pro Surface/Gruppe) |
| **Grid-Shape** | Gleiche Shape für alle Surfaces | Unterschiedliche Shapes pro Surface |
| **Koordinaten** | `sound_field_x`, `sound_field_y` (global) | `surface_grids[surface_id]['sound_field_x']` (pro Surface) |
| **Z-Koordinaten** | `Z_grid` (kombiniert) | `surface_grids[surface_id]['Z_grid']` (pro Surface) |

### 2. Berechnung

| Aspekt | ALTER Calculator | NEUER Calculator |
|--------|------------------|------------------|
| **Berechnungsart** | Alle Surfaces **zusammen** in einem Durchgang | Jede Surface **einzeln** (`_calculate_sound_field_for_surface_grid`) |
| **Schallfeld-Speicherung** | `sound_field_p` (global kombiniert) | `surface_results[surface_id]['sound_field_p']` (pro Surface) |
| **Komplexe Werte** | In `surface_fields[surface_id]` | In `surface_results[surface_id]['sound_field_p']` |
| **Magnitude** | Wird aus `sound_field_p` berechnet | Vorkalkuliert in `surface_results[surface_id]['sound_field_p_magnitude']` |

### 3. Datenzugriff für Plotting

| Aspekt | ALTER Calculator | NEUER Calculator |
|--------|------------------|------------------|
| **Hauptdatenquelle** | `calculation_spl['sound_field_p']` | `calculation_spl['surface_results'][surface_id]['sound_field_p']` |
| **Grid-Daten** | `sound_field_x`, `sound_field_y`, `sound_field_z` | `surface_grids[surface_id]['X_grid']`, etc. |
| **Surface-Maske** | `surface_mask` oder `surface_mask_strict` (kombiniert) | `surface_grids[surface_id]['surface_mask']` (pro Surface) |
| **Orientierung** | Muss aus Surface-Definition abgeleitet werden | `surface_grids[surface_id]['orientation']` (direkt verfügbar) |

### 4. Neue Features (nur NEUER Calculator)

| Feature | Beschreibung |
|---------|--------------|
| `orientation` | Direkt verfügbar: `"planar"` oder `"vertical"` |
| `dominant_axis` | Optional: `"x"`, `"y"`, oder `"z"` (für vertikale Flächen) |
| `resolution` | Pro-Surface-Auflösung (kann unterschiedlich sein) |
| Separate Grids | Jede Surface hat eigenes Grid (bessere Performance für große Surfaces) |

---

## 🔍 Code-Beispiele

### ALTER Calculator - Datenzugriff:

```python
# Globale Werte
sound_field_p = calculation_spl['sound_field_p']  # 2D-Array
sound_field_x = calculation_spl['sound_field_x']  # 1D-Array
sound_field_y = calculation_spl['sound_field_y']  # 1D-Array
Z_grid = np.array(calculation_spl['sound_field_z'])  # 2D-Array

# Surface-spezifische Werte (nur komplexe Feldwerte)
surface_fields = calculation_spl['surface_fields']
for surface_id, field_values in surface_fields.items():
    complex_field = np.array(field_values)  # Komplexe Werte für diese Surface
    magnitude = np.abs(complex_field)       # Magnitude berechnen
```

### NEUER Calculator - Datenzugriff:

```python
# Globale Werte (für Rückwärtskompatibilität)
sound_field_p_global = calculation_spl['sound_field_p']  # Interpoliert/kombiniert
sound_field_x_global = calculation_spl['sound_field_x']
sound_field_y_global = calculation_spl['sound_field_y']

# 🎯 NEU: Pro-Surface-Zugriff (Bevorzugt!)
surface_grids = calculation_spl['surface_grids']
surface_results = calculation_spl['surface_results']

for surface_id in surface_grids.keys():
    # Grid-Daten
    grid_data = surface_grids[surface_id]
    X_grid = np.array(grid_data['X_grid'])           # 2D-Array
    Y_grid = np.array(grid_data['Y_grid'])           # 2D-Array
    Z_grid = np.array(grid_data['Z_grid'])           # 2D-Array
    surface_mask = np.array(grid_data['surface_mask'])  # 2D-Boolean
    orientation = grid_data['orientation']            # "planar" oder "vertical"
    
    # Berechnungsergebnisse
    result_data = surface_results[surface_id]
    sound_field_p_complex = np.array(result_data['sound_field_p'])  # 2D komplex
    sound_field_p_magnitude = np.array(result_data['sound_field_p_magnitude'])  # 2D Magnitude
    
    # SPL berechnen (wenn nötig)
    spl_values = 20 * np.log10(sound_field_p_magnitude / 20e-6)
```

---

## ⚠️ Wichtige Hinweise

### Migration vom ALTEN zum NEUEN Calculator:

1. **Plotting-Code muss angepasst werden:**
   - Statt `calculation_spl['sound_field_p']` → `calculation_spl['surface_results'][surface_id]['sound_field_p']`
   - Statt globales Grid → pro-Surface-Grids aus `surface_grids`

2. **Orientierung ist jetzt direkt verfügbar:**
   - `surface_grids[surface_id]['orientation']` statt manuelle Ableitung

3. **Separate Grids bedeuten:**
   - Jede Surface kann unterschiedliche Auflösung haben
   - Jede Surface hat eigenes Koordinatensystem
   - Keine gemeinsamen Indizes mehr zwischen Surfaces

4. **Rückwärtskompatibilität:**
   - Globale Werte (`sound_field_p`, `sound_field_x`, etc.) werden noch gespeichert
   - Aber: Sie sind **interpoliert/kombiniert** und können ungenauer sein
   - **Bevorzugt**: Direkter Zugriff auf pro-Surface-Daten

---

## 🎯 Empfehlung für Plotting

**Verwende den NEUEN Calculator-Ansatz:**
- Direkter Zugriff auf `surface_grids` und `surface_results`
- Pro-Surface-Rendering (wie in `Plot3DSPL_neu.py`)
- Bessere Performance und Genauigkeit
- Unterstützung für unterschiedliche Orientierungen (planar/vertical)

**Beispiel aus Plot3DSPL.py (aktuelle Implementierung):**
```python
# Zeile 1110-1300: update_spl_plot()
# Verwendet bereits surface_grids und surface_results!
surface_grids = calculation_spl.get('surface_grids', {})
surface_results = calculation_spl.get('surface_results', {})
```

---

## 📝 Zusammenfassung

| Aspekt | ALTER Calculator | NEUER Calculator |
|--------|------------------|------------------|
| **Grid-Struktur** | 1 gemeinsames Grid | N separate Grids |
| **Berechnung** | Zusammen | Pro Surface |
| **Datenstruktur** | Global + `surface_fields` | Global + `surface_grids` + `surface_results` |
| **Plotting-Zugriff** | `sound_field_p` (global) | `surface_results[id]['sound_field_p']` |
| **Orientierung** | Muss abgeleitet werden | Direkt verfügbar |
| **Performance** | Gut für kleine Surfaces | Besser für große/unterschiedliche Surfaces |

