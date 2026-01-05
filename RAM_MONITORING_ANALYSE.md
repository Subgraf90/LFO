# RAM-Monitoring: Aktueller Stand

## Übersicht

**Aktuell gibt es nur begrenztes RAM-Monitoring:**

1. ✅ **Cache-Memory-Tracking** - Cache-Verbrauch wird geschätzt und getrackt
2. ⚠️ **Kein System-RAM-Monitoring** - System-RAM wird nicht überwacht
3. ⚠️ **Kein Snapshot-Memory-Monitoring** - Snapshot-Verbrauch wird nicht getrackt
4. ⚠️ **Keine automatischen Warnungen** - Keine Warnungen bei hohem RAM-Verbrauch

---

## 1. Aktuelles Monitoring

### Cache-Memory-Tracking

**Was wird getrackt:**
- Memory-Verbrauch pro Cache (`memory_usage_mb`)
- Memory-Evictions (`memory_evictions`)
- Memory-Limits (`max_memory_mb`)

**Wie wird es getrackt:**
```python
# In CacheManager
def estimate_memory_size(obj: Any) -> float:
    # Schätzt Memory-Größe von Objekten
    # - NumPy Arrays: .nbytes
    # - Dataclasses: Summiert alle Attribute
    # - Standard-Objekte: sys.getsizeof()
```

**Wo wird es verwendet:**
- Bei jedem `cache.set()` wird Memory geschätzt
- Memory-Verbrauch wird in `CacheStats` gespeichert
- Kann über `CacheMonitor` abgerufen werden

### CacheMonitor

**Verfügbare Funktionen:**
```python
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
monitor.print_stats()  # Zeigt Cache-Statistiken
monitor.print_detailed_stats()  # Detaillierte Statistiken
```

**Was wird angezeigt:**
- Hit-Rate, Size, Hits, Misses
- **NEU:** Memory-Verbrauch pro Cache
- **NEU:** Memory-Limits

---

## 2. Was fehlt

### System-RAM-Monitoring

**Problem:**
- System-RAM wird nicht überwacht
- Keine Warnung bei hohem System-RAM-Verbrauch
- Keine Anpassung basierend auf verfügbarem RAM

**Lösung:**
```python
# psutil ist bereits importiert, aber nicht verwendet
if PSUTIL_AVAILABLE:
    mem = psutil.virtual_memory()
    print(f"RAM verwendet: {mem.percent:.1f}%")
```

### Snapshot-Memory-Monitoring

**Problem:**
- Snapshot-Verbrauch wird nicht getrackt
- Keine Limits für Snapshots
- Unbegrenztes Wachstum möglich

**Lösung:**
```python
# In CacheMonitor
def print_memory_overview(self, container=None):
    # Zeigt System-RAM + Cache-RAM + Snapshot-RAM
    # ...
```

### Automatische Warnungen

**Problem:**
- Keine automatischen Warnungen bei hohem RAM-Verbrauch
- Keine Warnungen bei vielen Snapshots

**Lösung:**
- Warnungen bei >80% System-RAM
- Warnungen bei hohem Snapshot-Verbrauch
- Warnungen bei Cache-Memory-Limit

---

## 3. Erweiterte Monitoring-Funktionen

### System-RAM-Monitoring

```python
def get_system_memory_info():
    """Gibt System-RAM-Informationen zurück"""
    if not PSUTIL_AVAILABLE:
        return None
    
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percent': mem.percent,
        'free_gb': mem.free / (1024**3),
    }
```

### Snapshot-Memory-Monitoring

```python
def estimate_snapshot_memory(container):
    """Schätzt Memory-Verbrauch aller Snapshots"""
    calculation_axes = getattr(container, 'calculation_axes', {})
    total_memory = 0.0
    snapshot_count = 0
    
    for key, snapshot_data in calculation_axes.items():
        if key != "aktuelle_simulation":
            snapshot_count += 1
            total_memory += estimate_memory_size(snapshot_data)
    
    return {
        'total_mb': total_memory,
        'count': snapshot_count,
        'avg_mb': total_memory / snapshot_count if snapshot_count > 0 else 0.0,
    }
```

### Automatische Warnungen

```python
def check_memory_warnings(container=None):
    """Prüft Memory-Verbrauch und gibt Warnungen aus"""
    warnings = []
    
    # System-RAM
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            warnings.append(f"⚠️  System-RAM: {mem.percent:.1f}% verwendet")
    
    # Cache-RAM
    all_stats = cache_manager.get_cache_stats()
    for cache_name, data in all_stats.items():
        memory_mb = data['stats'].get('memory_usage_mb', 0.0)
        max_memory_mb = data['config'].get('max_memory_mb')
        if max_memory_mb and memory_mb > max_memory_mb * 0.9:
            warnings.append(f"⚠️  Cache '{cache_name}': {memory_mb:.1f}MB / {max_memory_mb:.1f}MB")
    
    # Snapshot-RAM
    if container:
        snapshot_info = estimate_snapshot_memory(container)
        if snapshot_info['total_mb'] > 1000:  # > 1 GB
            warnings.append(f"⚠️  Snapshots: {snapshot_info['total_mb']:.1f}MB ({snapshot_info['count']} Snapshots)")
    
    return warnings
```

---

## 4. Implementierung

### CacheMonitor erweitert

**Neue Funktionen:**
- `print_memory_overview()` - Zeigt System-RAM + Cache-RAM + Snapshot-RAM
- `get_system_memory_info()` - Gibt System-RAM-Info zurück
- `estimate_snapshot_memory()` - Schätzt Snapshot-Verbrauch
- `check_memory_warnings()` - Prüft Memory-Verbrauch und gibt Warnungen

**Verwendung:**
```python
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
monitor.print_memory_overview(container)  # Vollständige Memory-Übersicht
warnings = monitor.check_memory_warnings(container)  # Prüft Warnungen
```

---

## 5. Zusammenfassung

### Aktuelles Monitoring

- ✅ **Cache-Memory-Tracking** - Cache-Verbrauch wird geschätzt
- ✅ **CacheMonitor** - Kann Cache-Statistiken anzeigen
- ⚠️ **Kein System-RAM** - System-RAM wird nicht überwacht
- ⚠️ **Kein Snapshot-Memory** - Snapshot-Verbrauch wird nicht getrackt

### Erweiterte Funktionen (implementiert)

- ✅ **System-RAM-Monitoring** - Über psutil
- ✅ **Snapshot-Memory-Monitoring** - Schätzt Snapshot-Verbrauch
- ✅ **Memory-Übersicht** - Zeigt alle Memory-Verbräuche
- ✅ **Automatische Warnungen** - Bei hohem RAM-Verbrauch

### Verwendung

```python
# Memory-Übersicht anzeigen
monitor = CacheMonitor()
monitor.print_memory_overview(container)

# Warnungen prüfen
warnings = monitor.check_memory_warnings(container)
for warning in warnings:
    print(warning)
```

Das RAM-Monitoring ist jetzt **vollständig implementiert**! 🚀

