# Cache Memory-Management

## Übersicht

Das Cache-System verwendet **mehrschichtige Memory-Limits**, um sicherzustellen, dass der RAM nicht überlastet wird:

1. **Count-basierte Limits** (`max_size`) - Begrenzt Anzahl Einträge
2. **Memory-basierte Limits** (`max_memory_mb`) - Begrenzt Memory-Verbrauch
3. **LRU-Eviction** - Entfernt älteste Einträge automatisch
4. **Memory-Monitoring** - Überwacht Memory-Verbrauch und warnt bei hoher Auslastung

---

## 1. Memory-basierte Limits

### Konfiguration

```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Cache mit Memory-Limit registrieren
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,           # Max. 1000 Einträge
    max_memory_mb=500.0,    # Max. 500 MB Memory
    description="Grid Cache mit Memory-Limit"
)
```

### Verhalten

- **Beide Limits werden geprüft**: Sowohl `max_size` als auch `max_memory_mb`
- **Memory-Limit hat Priorität**: Wenn Memory-Limit erreicht wird, werden älteste Einträge entfernt
- **Count-Limit als Fallback**: Wenn Memory-Limit nicht gesetzt ist, wird nur Count-Limit verwendet

### Beispiel

```python
# Cache mit 500 MB Limit
cache = cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_memory_mb=500.0
)

# Wenn ein neuer Eintrag hinzugefügt wird:
# 1. Prüfe Memory-Limit: Aktueller Verbrauch + neuer Eintrag > 500 MB?
#    → Ja: Entferne älteste Einträge bis genug Platz
# 2. Prüfe Count-Limit: Anzahl Einträge >= 1000?
#    → Ja: Entferne ältesten Eintrag
# 3. Füge neuen Eintrag hinzu
```

---

## 2. Memory-Schätzung

### Automatische Schätzung

Das System schätzt automatisch die Memory-Größe jedes Cache-Eintrags:

```python
def estimate_memory_size(obj: Any) -> float:
    """
    Schätzt Memory-Größe in MB.
    
    Unterstützt:
    - NumPy Arrays (genaue Größe via .nbytes)
    - Dataclasses (summiert alle Attribute)
    - Standard Python-Objekte (sys.getsizeof)
    """
```

### Unterstützte Objekte

- ✅ **NumPy Arrays**: Genaue Größe via `array.nbytes`
- ✅ **CachedSurfaceGrid**: Summiert alle NumPy-Arrays
- ✅ **Standard Python-Objekte**: `sys.getsizeof()`

### Beispiel

```python
import numpy as np
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import CachedSurfaceGrid

# NumPy Array
arr = np.zeros((1000, 1000), dtype=np.float64)
memory_mb = estimate_memory_size(arr)  # ~7.6 MB

# CachedSurfaceGrid
cached_grid = CachedSurfaceGrid(
    surface_id="surface_1",
    X_grid=np.zeros((100, 100)),
    Y_grid=np.zeros((100, 100)),
    Z_grid=np.zeros((100, 100)),
    # ...
)
memory_mb = estimate_memory_size(cached_grid)  # Summiert alle Arrays
```

---

## 3. Memory-Monitoring

### Statistiken

```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Cache-Statistiken abrufen
stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Memory-Verbrauch: {stats['grid']['stats']['memory_usage_mb']:.1f} MB")
print(f"Memory-Evictions: {stats['grid']['stats']['memory_evictions']}")
```

### Warnungen

Das System warnt automatisch bei hoher Memory-Auslastung:

```
⚠️  WARNUNG: Cache 'grid' nutzt 92.3% des Memory-Limits (461.5MB / 500.0MB)
```

**Threshold:** 90% des Memory-Limits

### Globale Statistiken

```python
global_stats = cache_manager.get_global_stats()
print(f"Total Memory: {sum(s['stats']['memory_usage_mb'] for s in global_stats['caches'].values()):.1f} MB")
```

---

## 4. Adaptive Cache-Größen

### Automatische Anpassung basierend auf verfügbarem RAM

```python
def get_adaptive_cache_size(cache_type: CacheType, default_mb: float = 500.0) -> float:
    """
    Berechnet adaptive Cache-Größe basierend auf verfügbarem RAM.
    
    Strategie:
    - Nutze 10% des verfügbaren RAMs für Cache
    - Minimum: 100 MB
    - Maximum: default_mb
    """
    if not PSUTIL_AVAILABLE:
        return default_mb
    
    try:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        adaptive_mb = (available_memory_gb * 1024) * 0.1  # 10% des verfügbaren RAMs
        return max(100.0, min(adaptive_mb, default_mb))
    except Exception:
        return default_mb
```

### Verwendung

```python
# In Main.py
adaptive_memory_mb = get_adaptive_cache_size(CacheType.GRID, default_mb=500.0)
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_memory_mb=adaptive_memory_mb
)
```

---

## 5. Best Practices

### Für große Surface-Mengen

**Problem:** Viele Surfaces → Hoher Memory-Verbrauch

**Lösung:**

1. **Memory-Limit setzen:**
```python
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_memory_mb=1000.0  # 1 GB Limit
)
```

2. **File-Cache aktivieren** (für persistente Speicherung):
```python
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_memory_mb=500.0,
    enable_file_cache=True
)
```

3. **Adaptive Größen verwenden:**
```python
adaptive_mb = get_adaptive_cache_size(CacheType.GRID, default_mb=500.0)
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_memory_mb=adaptive_mb
)
```

### Monitoring

```python
# Regelmäßig Memory-Verbrauch prüfen
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
monitor.print_stats()  # Zeigt Memory-Verbrauch

# Bei hoher Auslastung: Cache leeren oder Limit erhöhen
if stats['grid']['stats']['memory_usage_mb'] > 450:
    cache_manager.clear_cache(CacheType.GRID)
    # Oder Limit erhöhen
    cache_manager.configure_cache(CacheType.GRID, max_memory_mb=1000.0)
```

---

## 6. Konfiguration in Main.py

### Aktuelle Konfiguration

```python
def _initialize_caches(self):
    # Grid Cache: Großer Cache für viele Surfaces
    cache_manager.register_cache(
        CacheType.GRID,
        max_size=int(getattr(self.settings, "surface_grid_cache_size", 1000)),
        max_memory_mb=float(getattr(self.settings, "surface_grid_cache_memory_mb", 500.0)),
        description="Surface Grid Cache"
    )
```

### Settings-Integration

```python
# In settings_state.py oder UI
settings.surface_grid_cache_size = 1000
settings.surface_grid_cache_memory_mb = 500.0  # 500 MB Limit
```

---

## 7. Memory-Eviction Strategie

### LRU-Eviction

**Priorität:**
1. **Memory-Limit** (wenn gesetzt): Entfernt älteste Einträge bis Memory-Limit eingehalten wird
2. **Count-Limit**: Entfernt ältesten Eintrag wenn Count-Limit erreicht

### Beispiel

```python
# Cache mit 500 MB Limit, aktuell 480 MB belegt
cache.set("new_large_entry", large_object)  # 50 MB

# Was passiert:
# 1. Prüfe Memory: 480 + 50 = 530 MB > 500 MB Limit
# 2. Entferne älteste Einträge bis < 450 MB (Platz für 50 MB)
# 3. Füge neuen Eintrag hinzu
# 4. Aktueller Verbrauch: ~500 MB
```

---

## 8. Zusammenfassung

### Features

- ✅ **Memory-basierte Limits**: Verhindert RAM-Überlastung
- ✅ **Automatische Memory-Schätzung**: Für NumPy Arrays und Dataclasses
- ✅ **LRU-Eviction**: Entfernt älteste Einträge automatisch
- ✅ **Memory-Monitoring**: Überwacht Verbrauch und warnt bei hoher Auslastung
- ✅ **Adaptive Größen**: Passt sich an verfügbaren RAM an

### Wichtige Einstellungen

```python
# Für große Surface-Mengen:
max_memory_mb = 1000.0  # 1 GB Limit

# Für normale Nutzung:
max_memory_mb = 500.0   # 500 MB Limit

# Für kleine Projekte:
max_memory_mb = 100.0   # 100 MB Limit
```

### Monitoring

```python
# Memory-Verbrauch prüfen
stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Memory: {stats['grid']['stats']['memory_usage_mb']:.1f} MB")
print(f"Limit: {stats['grid']['config']['max_memory_mb']:.1f} MB")
```

Das Memory-Management verhindert, dass der Cache den RAM sprengt! 🚀

