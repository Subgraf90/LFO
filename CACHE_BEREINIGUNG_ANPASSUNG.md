# Cache-Bereinigung: Angepasst für echte Nutzung

## Analyse: Macht TTL/Idle-Zeit Sinn?

### ❌ Problem mit TTL/Idle-Zeit bei interaktiver Nutzung

1. **Zu aggressiv**: Bei interaktiver Nutzung arbeiten Nutzer oft länger als 1 Stunde an einem Projekt
2. **Zu häufig**: `generate_per_surface()` wird bei jedem `calculate_spl()` aufgerufen → Bereinigung zu oft
3. **Performance**: Bereinigung bei jedem Zugriff kostet Performance
4. **LRU reicht aus**: Count-basierte Eviction (`max_size`) reicht aus, um Cache klein zu halten

### ✅ Bessere Strategie: Event-basierte Bereinigung

**Bereinigung nur bei relevanten Events:**
- ✅ **Beim Laden einer Datei** → Bereinige nicht-existierende Surfaces
- ✅ **Beim neuen Projekt** → Leere Cache komplett
- ✅ **LRU-Eviction** → Automatisch bei `max_size`
- ❌ **NICHT bei jedem `generate_per_surface()`** → Zu aggressiv

---

## Implementierung

### 1. TTL/Idle-Zeit deaktiviert

```python
# In Main.py
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,
    ttl_seconds=None,  # Deaktiviert
    max_idle_seconds=None,  # Deaktiviert
)
```

**Grund:**
- Bei interaktiver Nutzung arbeiten Nutzer länger als TTL
- Surfaces werden häufig wiederverwendet
- LRU-Eviction reicht aus

### 2. Bereinigung beim Laden

```python
# In UiFile._clear_current_state()
if hasattr(self.main_window, '_grid_generator'):
    grid_cache = self.main_window._grid_generator._grid_cache
    surface_store = getattr(self.settings, 'surface_definitions', {})
    valid_surface_ids = set(surface_store.keys())
    cleaned = grid_cache.cleanup_unused_surfaces(valid_surface_ids)
```

**Grund:**
- Beim Laden einer Datei können Surfaces gelöscht worden sein
- Bereinigung nur einmal beim Laden, nicht bei jedem Zugriff

### 3. Bereinigung beim neuen Projekt

```python
# In UiFile.new_pa_file()
cache_manager.clear_cache(CacheType.GRID)
```

**Grund:**
- Neues Projekt → Cache sollte leer sein
- Verhindert Altlasten von vorherigem Projekt

### 4. Keine Bereinigung bei `generate_per_surface()`

**Entfernt:**
- ❌ TTL/Idle-Prüfung bei jedem `get()`
- ❌ Bereinigung bei jedem `generate_per_surface()`

**Grund:**
- Zu aggressiv für interaktive Nutzung
- Performance-Overhead
- LRU-Eviction reicht aus

---

## Verhalten nach Anpassung

### Cache-Verwaltung

1. **LRU-Eviction**: Automatisch bei `max_size` (1000 Einträge)
2. **Event-basierte Bereinigung**: Nur beim Laden/Neues Projekt
3. **Gezielte Invalidierung**: Bei Surface-Änderungen (hide/disable/geometry)

### Performance

- ✅ **Kein Overhead** bei jedem Cache-Zugriff
- ✅ **Bereinigung nur bei Events** (selten)
- ✅ **LRU-Eviction** hält Cache automatisch klein

### Memory-Management

- ✅ **Count-Limit** (`max_size=1000`) verhindert unbegrenztes Wachstum
- ✅ **LRU-Eviction** entfernt automatisch älteste Einträge
- ✅ **Event-basierte Bereinigung** entfernt Altlasten

---

## Zusammenfassung

### Was bleibt

- ✅ **LRU-Eviction** bei `max_size` → Cache bleibt automatisch klein
- ✅ **Event-basierte Bereinigung** → Nur bei relevanten Events
- ✅ **Gezielte Invalidierung** → Bei Surface-Änderungen

### Was entfernt wurde

- ❌ **TTL/Idle-Zeit** → Zu aggressiv für interaktive Nutzung
- ❌ **Bereinigung bei jedem Zugriff** → Performance-Overhead

### Ergebnis

- ✅ **Cache bleibt klein** durch LRU-Eviction
- ✅ **Keine Altlasten** durch Event-basierte Bereinigung
- ✅ **Gute Performance** durch keine Bereinigung bei jedem Zugriff
- ✅ **Passend für interaktive Nutzung** → Keine zu kurzen Timeouts

Die Cache-Bereinigung ist jetzt **angepasst für echte Nutzung**! 🚀

