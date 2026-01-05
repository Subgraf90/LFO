# Cache-Verwaltung: Wie funktioniert es?

## Übersicht

Der Cache-Manager verwendet ein **Singleton-Pattern** mit **individueller Cache-Verwaltung**. Jeder Cache wird zentral registriert, kann aber individuell konfiguriert und verwaltet werden.

---

## 1. Architektur

```
┌─────────────────────────────────────────────────────────┐
│              CacheManager (Singleton)                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  _caches: Dict[CacheType, LRUCache]              │  │
│  │  _configs: Dict[CacheType, CacheConfig]         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │  GRID   │         │  CALC   │         │  PLOT   │
    │  Cache  │         │  Cache  │         │  Cache  │
    └─────────┘         └─────────┘         └─────────┘
```

---

## 2. Cache-Registrierung

### Schritt 1: Cache registrieren

```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Registriere einen Cache mit individueller Konfiguration
cache_manager.register_cache(
    CacheType.GRID,
    max_size=1000,              # Maximale Anzahl Einträge
    enable_lru=True,            # LRU-Eviction aktivieren
    description="Surface Grid Cache"
)
```

**Was passiert intern:**
1. Cache-Manager prüft, ob Cache bereits existiert
2. Erstellt neuen `LRUCache` mit konfigurierter Größe
3. Speichert Cache in `_caches` Dictionary
4. Speichert Konfiguration in `_configs` Dictionary

---

## 3. Cache-Zugriff

### Get-Operation (Lesen)

```python
# Hole Cache-Instanz
grid_cache = cache_manager.get_cache(CacheType.GRID)

# Oder: Hole oder erstelle Cache
grid_cache = cache_manager.get_or_create_cache(
    CacheType.GRID,
    max_size=1000
)

# Wert aus Cache lesen
value = grid_cache.get(cache_key)
```

**Was passiert intern:**
1. Thread-Safe Lock wird gesetzt
2. Statistiken werden aktualisiert (`total_accesses++`)
3. Prüfung: Existiert Key im Cache?
   - **Ja (Cache Hit):**
     - Wert wird zurückgegeben
     - Key wird ans Ende verschoben (LRU-Update)
     - `hits++`
   - **Nein (Cache Miss):**
     - `None` wird zurückgegeben
     - `misses++`
4. Lock wird freigegeben

**LRU-Verhalten:**
- Bei jedem `get()` wird der Key ans Ende verschoben
- Älteste Einträge sind am Anfang
- Bei vollem Cache werden älteste Einträge entfernt

---

### Set-Operation (Schreiben)

```python
# Wert in Cache schreiben
grid_cache.set(cache_key, cached_grid)
```

**Was passiert intern:**
1. Thread-Safe Lock wird gesetzt
2. Prüfung: Existiert Key bereits?
   - **Ja:** Key wird ans Ende verschoben (Update)
   - **Nein:** Neuer Eintrag
3. Prüfung: Ist Cache voll?
   - **Ja:** Ältester Eintrag wird entfernt (`evictions++`)
   - **Nein:** Eintrag wird hinzugefügt
4. Statistiken werden aktualisiert (`size` aktualisiert)
5. Lock wird freigegeben

**LRU-Eviction:**
```
Cache (max_size=3):
[oldest] → [middle] → [newest]

Neuer Eintrag bei vollem Cache:
1. Entferne [oldest]
2. Füge [new] am Ende hinzu
3. Resultat: [middle] → [newest] → [new]
```

---

## 4. Cache-Verwaltung

### Individuelle Cache-Verwaltung

```python
# Nur einen spezifischen Cache leeren
cache_manager.clear_cache(CacheType.GRID)

# Alle Caches leeren
cache_manager.clear_all_caches()
```

**Was passiert intern:**
- `clear_cache()`: Leert nur den angegebenen Cache
- `clear_all_caches()`: Iteriert über alle Caches und leert sie
- Statistiken bleiben erhalten (können mit `reset_stats()` zurückgesetzt werden)

---

### Gezielte Cache-Invalidierung

```python
# Entferne nur Einträge, die einem Prädikat entsprechen
def predicate(key):
    # Entferne alle Einträge für "surface_1"
    return isinstance(key, tuple) and key[0] == "surface_1"

cache_manager.invalidate_cache(
    CacheType.GRID,
    predicate=predicate
)
```

**Was passiert intern:**
1. Cache wird durchlaufen
2. Für jeden Key wird Prädikat geprüft
3. Keys, die Prädikat erfüllen, werden entfernt
4. Statistiken werden aktualisiert

**Beispiel:**
```
Cache vorher:
  ("surface_1", ...) → Grid1
  ("surface_2", ...) → Grid2
  ("surface_1", ...) → Grid3

Nach invalidate_cache(predicate=lambda k: k[0]=="surface_1"):
  ("surface_2", ...) → Grid2
```

---

### Cache-Konfiguration ändern

```python
# Ändere maximale Größe zur Laufzeit
cache_manager.configure_cache(
    CacheType.GRID,
    max_size=2000  # Erhöhe von 1000 auf 2000
)
```

**Was passiert intern:**
1. Cache-Konfiguration wird aktualisiert
2. `LRUCache.max_size` wird geändert
3. Bestehende Einträge bleiben erhalten
4. Neue Evictions erfolgen bei neuem Limit

---

## 5. Statistiken

### Cache-Statistiken abrufen

```python
# Statistiken für einen Cache
stats = cache_manager.get_cache_stats(CacheType.GRID)
# {
#   'grid': {
#     'stats': {
#       'hits': 150,
#       'misses': 50,
#       'hit_rate': 75.0,
#       'size': 100,
#       'evictions': 10,
#       ...
#     },
#     'config': {...}
#   }
# }

# Globale Statistiken
global_stats = cache_manager.get_global_stats()
# {
#   'total_caches': 6,
#   'total_hits': 500,
#   'total_misses': 200,
#   'global_hit_rate': 71.43,
#   'total_size': 300,
#   ...
# }
```

**Was wird getrackt:**
- `hits`: Anzahl Cache-Treffer
- `misses`: Anzahl Cache-Fehler
- `evictions`: Anzahl entfernte Einträge (bei vollem Cache)
- `size`: Aktuelle Anzahl Einträge
- `hit_rate`: Hit-Rate in Prozent
- `total_accesses`: Gesamtanzahl Zugriffe
- `last_access`: Zeitpunkt des letzten Zugriffs

---

## 6. Thread-Safety

### Lock-Mechanismus

```python
class LRUCache:
    def __init__(self):
        self._lock = Lock()  # Thread-Safe Lock
    
    def get(self, key):
        with self._lock:  # Lock wird automatisch freigegeben
            # Thread-safe Operationen
            ...
```

**Warum wichtig:**
- Mehrere Threads können gleichzeitig auf Cache zugreifen
- Lock verhindert Race Conditions
- `with self._lock:` stellt sicher, dass Lock freigegeben wird

**Beispiel:**
```
Thread 1: get(key1) → Lock gesetzt → Operation → Lock freigegeben
Thread 2: get(key2) → Wartet auf Lock → Lock gesetzt → Operation → Lock freigegeben
```

---

## 7. Shared Grid-Generator Pattern

### Problem ohne Shared Instance

```
calculate_spl() → SoundFieldCalculator() → FlexibleGridGenerator()
  └─> Neuer Cache (leer!)

calculate_spl() → SoundFieldCalculator() → FlexibleGridGenerator()
  └─> Neuer Cache (leer!) ← Cache-Verlust!
```

### Lösung mit Shared Instance

```
MainWindow.__init__()
  └─> FlexibleGridGenerator() [Shared Instance]
        └─> Cache bleibt erhalten!

calculate_spl()
  └─> SoundFieldCalculator(grid_generator=shared_instance)
        └─> Nutzt bestehenden Cache ← Cache bleibt erhalten!
```

**Vorteile:**
- Cache bleibt über mehrere Berechnungen erhalten
- Deutlich schneller bei wiederholten Berechnungen
- Weniger Memory-Verbrauch

---

## 8. Praktisches Beispiel

### Initialisierung in MainWindow

```python
class MainWindow:
    def __init__(self, settings, container):
        # ...
        
        # 1. Registriere alle Caches
        self._initialize_caches(settings)
        
        # 2. Erstelle Shared Grid-Generator
        self._grid_generator = FlexibleGridGenerator(
            settings,
            grid_cache=cache_manager.get_cache(CacheType.GRID)
        )
    
    def _initialize_caches(self, settings):
        # Grid Cache: Großer Cache für viele Surfaces
        cache_manager.register_cache(
            CacheType.GRID,
            max_size=getattr(settings, "surface_grid_cache_size", 1000),
            description="Surface Grid Cache"
        )
        
        # Calc Caches: Individuell konfiguriert
        cache_manager.register_cache(
            CacheType.CALC_GEOMETRY,
            max_size=getattr(settings, "geometry_cache_size", 1000),
            description="Geometry Cache"
        )
```

### Verwendung in SoundFieldCalculator

```python
class SoundFieldCalculator:
    def __init__(self, settings, data, calculation_spl, grid_generator=None):
        # ...
        
        # Nutze Cache-Manager für Geometry Cache
        self._geometry_cache = cache_manager.get_or_create_cache(
            CacheType.CALC_GEOMETRY,
            max_size=1000
        )
        
        # Nutze Shared Grid-Generator
        if grid_generator is None:
            self._grid_generator = FlexibleGridGenerator(settings)
        else:
            self._grid_generator = grid_generator  # Shared!
```

### Cache-Zugriff in FlexibleGridGenerator

```python
class FlexibleGridGenerator:
    def __init__(self, settings, grid_cache=None):
        # ...
        
        # Nutze übergebenen Cache oder erstelle neuen
        if grid_cache is None:
            self._grid_cache = cache_manager.get_or_create_cache(
                CacheType.GRID,
                max_size=1000
            )
        else:
            self._grid_cache = grid_cache  # Shared Cache!
    
    def generate_per_surface_check_cache(self, ...):
        cache_key = self._make_surface_cache_key(...)
        
        # Cache-Zugriff über Cache-Manager
        cached_grid = self._grid_cache.get(cache_key)
        if cached_grid is not None:
            # Cache Hit!
            return cached_grid.to_surface_grid(geom)
        
        # Cache Miss - berechne neu
        # ...
```

---

## 9. Cache-Lifecycle

### 1. Initialisierung
```
MainWindow.__init__()
  └─> cache_manager.register_cache(...)
      └─> LRUCache wird erstellt
          └─> Cache ist leer (size=0)
```

### 2. Erste Berechnung
```
calculate_spl()
  └─> FlexibleGridGenerator.generate_per_surface()
      └─> Cache Miss (kein Eintrag vorhanden)
          └─> Grid wird berechnet
              └─> grid_cache.set(key, value)
                  └─> Cache enthält jetzt 1 Eintrag (size=1)
```

### 3. Wiederholte Berechnung
```
calculate_spl() [gleiche Geometrie]
  └─> FlexibleGridGenerator.generate_per_surface()
      └─> Cache Hit! (Eintrag vorhanden)
          └─> Grid wird aus Cache geladen
              └─> Keine Neuberechnung nötig!
```

### 4. Cache voll
```
Cache ist voll (size=max_size)
  └─> Neuer Eintrag wird hinzugefügt
      └─> Ältester Eintrag wird entfernt (evictions++)
          └─> Cache bleibt bei max_size
```

### 5. Cache-Invalidierung
```
Surface wird geändert
  └─> cache_manager.invalidate_cache(CacheType.GRID, predicate=...)
      └─> Betroffene Einträge werden entfernt
          └─> Cache wird kleiner (size--)
```

---

## 10. Zusammenfassung

### Wie funktioniert die Cache-Verwaltung?

1. **Zentrale Registrierung**: Alle Caches werden beim Start registriert
2. **Individuelle Konfiguration**: Jeder Cache hat eigene Größe/Einstellungen
3. **LRU-Eviction**: Älteste Einträge werden bei vollem Cache entfernt
4. **Thread-Safe**: Lock-Mechanismus verhindert Race Conditions
5. **Statistiken**: Detaillierte Metriken für jeden Cache
6. **Gezielte Verwaltung**: Caches können einzeln geleert/invalidiert werden
7. **Shared Instances**: Grid-Generator wird geteilt, Cache bleibt erhalten

### Vorteile

- ✅ **Globale Verwaltung**: Zentrale Instanz für alle Caches
- ✅ **Individuelle Kontrolle**: Jeder Cache einzeln konfigurierbar
- ✅ **Performance**: Cache bleibt über mehrere Berechnungen erhalten
- ✅ **Flexibilität**: Zur Laufzeit anpassbar
- ✅ **Monitoring**: Detaillierte Statistiken verfügbar

