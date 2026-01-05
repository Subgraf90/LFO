# Cache-Bereinigung: Keine Altlasten

## Übersicht

Das Cache-System verwendet **automatische Bereinigung**, um sicherzustellen, dass keine Altlasten angesammelt werden:

1. **TTL (Time-To-Live)** - Einträge werden nach einer bestimmten Zeit automatisch entfernt
2. **Idle-Zeit** - Einträge, die lange nicht verwendet wurden, werden entfernt
3. **Bereinigung nicht-existierender Surfaces** - Cache-Einträge für gelöschte Surfaces werden entfernt
4. **Automatische Bereinigung** - Wird bei jedem Cache-Zugriff durchgeführt

---

## 1. TTL (Time-To-Live)

### Konfiguration

```python
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=3600.0,  # 1 Stunde TTL
    description="Grid Cache"
)
```

### Verhalten

- Einträge werden nach `ttl_seconds` automatisch entfernt
- TTL wird bei jedem `get()` geprüft
- Abgelaufene Einträge werden automatisch entfernt

### Beispiel

```python
# Cache mit 1 Stunde TTL
cache.set("key1", value1)  # Erstellt um 10:00
# ...
cache.get("key1")  # Um 10:30 → OK
cache.get("key1")  # Um 11:30 → Entfernt (TTL abgelaufen)
```

---

## 2. Idle-Zeit (Max. Unbenutzte Zeit)

### Konfiguration

```python
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    max_idle_seconds=1800.0,  # 30 Minuten Idle-Zeit
    description="Grid Cache"
)
```

### Verhalten

- Einträge, die länger als `max_idle_seconds` nicht verwendet wurden, werden entfernt
- Idle-Zeit wird bei jedem `get()` geprüft
- Unbenutzte Einträge werden automatisch entfernt

### Beispiel

```python
# Cache mit 30 Min Idle-Zeit
cache.set("key1", value1)  # Erstellt um 10:00
cache.get("key1")  # Um 10:15 → OK (Zugriffszeit aktualisiert)
# ...
cache.get("key1")  # Um 11:00 → Entfernt (45 Min unbenutzt)
```

---

## 3. Bereinigung nicht-existierender Surfaces

### Automatische Bereinigung

```python
# In FlexibleGridGenerator.generate_per_surface()
surface_store = getattr(self.settings, 'surface_definitions', {})
valid_surface_ids = set(surface_store.keys())
cleaned = self._grid_cache.cleanup_unused_surfaces(valid_surface_ids)
```

### Verhalten

- Entfernt Cache-Einträge für Surfaces, die nicht mehr in `surface_definitions` existieren
- Wird automatisch bei jedem `generate_per_surface()` Aufruf durchgeführt
- Verhindert Ansammlung von Altlasten

### Beispiel

```python
# Surface "surface_1" wird gelöscht
# Cache enthält noch Einträge für "surface_1"
# ...
# Bei nächstem generate_per_surface():
# → cleanup_unused_surfaces() entfernt alle Einträge für "surface_1"
```

---

## 4. Automatische Bereinigung bei Cache-Zugriff

### TTL/Idle-Prüfung bei `get()`

```python
def get(self, key: Any) -> Optional[Any]:
    if key in self._cache:
        # Prüfe TTL
        if self._ttl_seconds is not None:
            if current_time - creation_time > self._ttl_seconds:
                self._remove_entry(key)
                return None
        
        # Prüfe Idle-Zeit
        if self._max_idle_seconds is not None:
            if current_time - last_access > self._max_idle_seconds:
                self._remove_entry(key)
                return None
        
        # Update Zugriffszeit
        self._access_times[key] = current_time
        return self._cache[key]
```

### Verhalten

- TTL und Idle-Zeit werden bei jedem `get()` geprüft
- Abgelaufene Einträge werden automatisch entfernt
- Keine separate Bereinigungs-Task nötig

---

## 5. Manuelle Bereinigung

### Abgelaufene Einträge bereinigen

```python
cache = cache_manager.get_cache(CacheType.GRID)
expired_count = cache.cleanup_expired()
print(f"Bereinigt {expired_count} abgelaufene Einträge")
```

### Nicht-existierende Surfaces bereinigen

```python
surface_store = getattr(settings, 'surface_definitions', {})
valid_surface_ids = set(surface_store.keys())
cache = cache_manager.get_cache(CacheType.GRID)
cleaned_count = cache.cleanup_unused_surfaces(valid_surface_ids)
print(f"Bereinigt {cleaned_count} Einträge für nicht-existierende Surfaces")
```

---

## 6. Standard-Konfiguration

### Grid Cache

```python
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=3600.0,        # 1 Stunde TTL
    max_idle_seconds=1800.0,   # 30 Minuten Idle
    description="Surface Grid Cache"
)
```

### Bedeutung

- **TTL 1 Stunde**: Einträge werden nach 1 Stunde automatisch entfernt
- **Idle 30 Min**: Einträge, die 30 Min nicht verwendet wurden, werden entfernt
- **Kombiniert**: Einträge werden entfernt, wenn TTL ODER Idle-Zeit überschritten wird

---

## 7. Best Practices

### Für große Surface-Anzahlen

```python
# Kürzere TTL/Idle-Zeit für häufige Bereinigung
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=1800.0,        # 30 Min TTL
    max_idle_seconds=900.0,    # 15 Min Idle
)
```

### Für kleine Projekte

```python
# Längere TTL/Idle-Zeit für weniger Bereinigung
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=7200.0,        # 2 Stunden TTL
    max_idle_seconds=3600.0,   # 1 Stunde Idle
)
```

### Für persistente Caches

```python
# Keine TTL/Idle-Zeit (nur LRU-Eviction)
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=None,          # Kein TTL
    max_idle_seconds=None,     # Keine Idle-Zeit
)
```

---

## 8. Monitoring

### Bereinigungs-Statistiken

```python
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Cache-Größe: {stats['grid']['stats']['size']}")
print(f"Evictions: {stats['grid']['stats']['evictions']}")
```

### Manuelle Bereinigung prüfen

```python
cache = cache_manager.get_cache(CacheType.GRID)
expired = cache.cleanup_expired()
print(f"Bereinigt {expired} abgelaufene Einträge")
```

---

## 9. Zusammenfassung

### Features

- ✅ **TTL**: Automatische Entfernung nach Zeit
- ✅ **Idle-Zeit**: Automatische Entfernung ungenutzter Einträge
- ✅ **Bereinigung nicht-existierender Surfaces**: Entfernt Altlasten
- ✅ **Automatische Bereinigung**: Bei jedem Cache-Zugriff
- ✅ **Manuelle Bereinigung**: Optional verfügbar

### Standard-Verhalten

- **TTL**: 1 Stunde
- **Idle-Zeit**: 30 Minuten
- **Automatische Bereinigung**: Bei jedem `generate_per_surface()`

### Ergebnis

- ✅ **Keine Altlasten**: Cache bleibt sauber
- ✅ **Automatisch**: Keine manuelle Wartung nötig
- ✅ **Konfigurierbar**: TTL/Idle-Zeit anpassbar

Der Cache bleibt automatisch sauber und sammelt keine Altlasten an! 🚀

