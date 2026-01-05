# Snapshot Memory-Analyse

## Übersicht

**Ja, Snapshot-Daten werden vollständig im RAM gespeichert!**

Snapshots werden in `container.calculation_axes` gespeichert und enthalten große Datenmengen, die komplett im RAM gehalten werden.

---

## 1. Wo werden Snapshots gespeichert?

### Hauptspeicherort

```python
# In DataContainer
self.calculation_axes = {}  # Dictionary mit allen Snapshots

# Struktur:
calculation_axes = {
    "aktuelle_simulation": {...},  # Aktuelle Berechnung
    "Snapshot 1": {...},           # Snapshot 1
    "Snapshot 2": {...},           # Snapshot 2
    ...
}
```

### Speicherung beim Erstellen

```python
# In WindowSnapshotWidget.on_capture_button_clicked()
capture_data = copy.deepcopy(self.container.calculation_axes["aktuelle_simulation"])
# ... füge weitere Daten hinzu ...
self.container.calculation_axes[new_key] = capture_data
```

**Wichtig:** `copy.deepcopy()` erstellt eine vollständige Kopie aller Daten im RAM!

---

## 2. Was wird in Snapshots gespeichert?

### Gespeicherte Daten

1. **Axis-Plot-Daten** (`calculation_axes["aktuelle_simulation"]`)
   - `x_data_xaxis`, `y_data_xaxis` (NumPy Arrays)
   - `x_data_yaxis`, `y_data_yaxis` (NumPy Arrays)
   - `segment_boundaries_xaxis`, `segment_boundaries_yaxis`

2. **SPL-Feld-Daten** (`calculation_spl`)
   - `sound_field_p` (große NumPy Arrays)
   - `sound_field_x`, `sound_field_y` (große NumPy Arrays)
   - `surface_grids` (für alle Surfaces)
   - `surface_results` (für alle Surfaces)

3. **FDTD-Simulationsdaten** (`fdtd_simulation`)
   - `pressure_frames` (sehr große Arrays - 16 Frames pro Periode)
   - `sound_field_x`, `sound_field_y`

4. **Polar-Plot-Daten** (`calculation_polar`)
   - `sound_field_p`
   - `angles`, `frequencies`

5. **Impulse-Daten** (`calculation_impulse`)
   - `magnitude_data`, `phase_response`, `impulse_response`
   - `arrival_times`

### Memory-Verbrauch pro Snapshot

**Geschätzte Größe:**
- **Kleine Projekte**: ~50-200 MB pro Snapshot
- **Mittlere Projekte**: ~200-500 MB pro Snapshot
- **Große Projekte**: ~500 MB - 2 GB pro Snapshot

**Faktoren:**
- Anzahl Surfaces
- Grid-Resolution
- Anzahl Frequenzen
- FDTD-Frames (16 Frames pro Periode)

---

## 3. Memory-Verbrauch bei mehreren Snapshots

### Beispiel

```python
# 5 Snapshots mit je 200 MB
calculation_axes = {
    "aktuelle_simulation": 200 MB,
    "Snapshot 1": 200 MB,
    "Snapshot 2": 200 MB,
    "Snapshot 3": 200 MB,
    "Snapshot 4": 200 MB,
}
# Total: ~1 GB RAM nur für Snapshots!
```

### Problem

- **Jeder Snapshot** ist eine vollständige Kopie aller Berechnungsdaten
- **Keine Limits** → Unbegrenztes Wachstum möglich
- **Keine Bereinigung** → Altlasten bleiben im RAM

---

## 4. Lösungsvorschläge

### Option 1: Memory-Limit für Snapshots

```python
# In DataContainer
MAX_SNAPSHOT_MEMORY_MB = 2000.0  # 2 GB Limit

def add_snapshot(self, key, data):
    # Schätze Memory-Größe
    memory_mb = estimate_memory_size(data)
    
    # Prüfe Limit
    current_memory = sum(estimate_memory_size(s) for s in self.calculation_axes.values())
    if current_memory + memory_mb > MAX_SNAPSHOT_MEMORY_MB:
        # Entferne älteste Snapshots
        # ...
    
    self.calculation_axes[key] = data
```

### Option 2: Snapshot-Count-Limit

```python
MAX_SNAPSHOTS = 10  # Max. 10 Snapshots

def add_snapshot(self, key, data):
    if len(self.calculation_axes) >= MAX_SNAPSHOTS:
        # Entferne ältesten Snapshot
        oldest_key = min(self.calculation_axes.keys(), 
                        key=lambda k: self.calculation_axes[k].get('created_at', 0))
        del self.calculation_axes[oldest_key]
    
    self.calculation_axes[key] = data
```

### Option 3: Komprimierung

```python
# Komprimiere große Arrays beim Speichern
import pickle
import gzip

def compress_snapshot(data):
    pickled = pickle.dumps(data)
    compressed = gzip.compress(pickled)
    return compressed

def decompress_snapshot(compressed):
    pickled = gzip.decompress(compressed)
    return pickle.loads(pickled)
```

### Option 4: File-basierte Snapshots

```python
# Speichere Snapshots auf Disk statt im RAM
SNAPSHOT_DIR = "snapshots/"

def save_snapshot(self, key, data):
    file_path = os.path.join(SNAPSHOT_DIR, f"{key}.pickle")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    # Speichere nur Metadaten im RAM
    self.calculation_axes[key] = {"file_path": file_path, "created_at": time.time()}
```

---

## 5. Empfehlung

### Für Grid-Cache: Memory-basiert

✅ **Cache-Größe speicherbasiert** statt count-basiert:
- Große Surfaces → weniger Einträge im Cache
- Kleine Surfaces → mehr Einträge im Cache
- Automatische Anpassung an tatsächlichen Memory-Verbrauch

### Für Snapshots: Count-Limit + Monitoring

✅ **Snapshot-Count-Limit** (z.B. max. 10 Snapshots)
✅ **Memory-Monitoring** für Snapshots
✅ **Optional: File-basierte Snapshots** für große Projekte

---

## 6. Zusammenfassung

### Snapshots im RAM

- ✅ **Ja, vollständig im RAM** gespeichert
- ✅ **Große Datenmengen** (~50-500 MB pro Snapshot)
- ✅ **Unbegrenztes Wachstum** möglich
- ⚠️ **Kann RAM sprengen** bei vielen Snapshots

### Cache-Größe

- ✅ **Memory-basiert** statt count-basiert
- ✅ **Automatische Anpassung** an tatsächlichen Verbrauch
- ✅ **Besser für große Surface-Mengen**

Die Snapshot-Speicherung sollte ebenfalls Memory-Management haben! 🚀

