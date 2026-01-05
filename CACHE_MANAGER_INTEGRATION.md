# Cache Manager Integration Guide

## Übersicht

Der `CacheManager` bietet:
- ✅ **Globale Verwaltung**: Zentrale Instanz für alle Caches
- ✅ **Individuelle Kontrolle**: Jeder Cache kann einzeln konfiguriert werden
- ✅ **Thread-Safe**: Sichere Verwendung in Multi-Thread-Umgebungen
- ✅ **Statistiken**: Detaillierte Performance-Metriken
- ✅ **Flexibilität**: Caches können individuell geleert/invalidiert werden

---

## Architektur

```
CacheManager (Singleton)
├── CacheType.GRID → LRUCache (FlexibleGridGenerator)
├── CacheType.CALC_GEOMETRY → LRUCache (SoundFieldCalculator)
├── CacheType.CALC_GRID → LRUCache (SoundFieldCalculator)
├── CacheType.PLOT_SURFACE_ACTORS → LRUCache (Plot3D)
├── CacheType.PLOT_TEXTURE → LRUCache (Plot3D)
└── CacheType.PLOT_GEOMETRY → LRUCache (Plot3D)
```

---

## Integration in bestehende Komponenten

### 1. FlexibleGridGenerator

**Vorher:**
```python
class FlexibleGridGenerator(ModuleBase):
    def __init__(self, settings):
        # ...
        self._surface_grid_cache: OrderedDict[tuple, CachedSurfaceGrid] = OrderedDict()
        self._surface_cache_max_size = max_cache_size
```

**Nachher:**
```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

class FlexibleGridGenerator(ModuleBase):
    def __init__(self, settings, grid_cache=None):
        # ...
        # Verwende Cache-Manager oder übergebenen Cache
        if grid_cache is None:
            max_cache_size = int(getattr(settings, "surface_grid_cache_size", 1000))
            self._grid_cache = cache_manager.get_or_create_cache(
                CacheType.GRID,
                max_size=max_cache_size,
                description="Surface Grid Cache"
            )
        else:
            self._grid_cache = grid_cache
        
        # Wrapper für Kompatibilität
        self._surface_grid_cache = self._grid_cache._cache
        self._surface_cache_max_size = self._grid_cache.max_size
    
    def generate_per_surface_check_cache(self, ...):
        cache_key = self._make_surface_cache_key(...)
        
        # Verwende Cache-Manager
        cached_grid = self._grid_cache.get(cache_key)
        if cached_grid is not None:
            # Move to end (LRU)
            self._grid_cache._cache.move_to_end(cache_key)
            return cached_grid.to_surface_grid(geom)
        
        # Cache miss - berechne neu
        # ...
    
    def generate_per_surface_process_and_create_grids_create_surface_grid(self, ...):
        # ...
        # Speichere im Cache-Manager
        self._grid_cache.set(cache_key, cached_grid)
```

---

### 2. SoundFieldCalculator

**Vorher:**
```python
class SoundFieldCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_spl):
        # ...
        self._geometry_cache = {}
        self._grid_cache = None
        self._grid_hash = None
        self._grid_generator = FlexibleGridGenerator(settings)
```

**Nachher:**
```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

class SoundFieldCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_spl, grid_generator=None):
        # ...
        # Geometry Cache über Cache-Manager
        max_geometry_cache_size = int(getattr(settings, "geometry_cache_size", 1000))
        self._geometry_cache = cache_manager.get_or_create_cache(
            CacheType.CALC_GEOMETRY,
            max_size=max_geometry_cache_size,
            description="Geometry Cache for Sound Field Calculation"
        )
        
        # Grid Cache über Cache-Manager
        self._grid_cache = cache_manager.get_or_create_cache(
            CacheType.CALC_GRID,
            max_size=100,  # Weniger Einträge, da Grids größer sind
            description="Grid Cache for Sound Field Calculation"
        )
        
        # Grid Generator (shared oder neu)
        if grid_generator is None:
            self._grid_generator = FlexibleGridGenerator(settings)
        else:
            self._grid_generator = grid_generator
```

---

### 3. Plot3D

**Vorher:**
```python
class DrawSPLPlot3D(...):
    def __init__(self, ...):
        # ...
        self._surface_actors: dict[str, Any] = {}
        self._surface_texture_cache: dict[str, str] = {}
        self._plot_geometry_cache = None
```

**Nachher:**
```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

class DrawSPLPlot3D(...):
    def __init__(self, ...):
        # ...
        # Surface Actors Cache
        self._surface_actors_cache = cache_manager.get_or_create_cache(
            CacheType.PLOT_SURFACE_ACTORS,
            max_size=500,
            description="Surface Actor Cache for 3D Plot"
        )
        
        # Texture Cache
        self._texture_cache = cache_manager.get_or_create_cache(
            CacheType.PLOT_TEXTURE,
            max_size=500,
            description="Texture Cache for 3D Plot"
        )
        
        # Geometry Cache
        self._geometry_cache = cache_manager.get_or_create_cache(
            CacheType.PLOT_GEOMETRY,
            max_size=100,
            description="Plot Geometry Cache"
        )
        
        # Kompatibilität: Wrapper für bestehenden Code
        self._surface_actors = self._surface_actors_cache._cache
        self._surface_texture_cache = self._texture_cache._cache
```

---

### 4. MainWindow - Initialisierung

**Vorher:**
```python
class MainWindow:
    def __init__(self, settings, container, parent=None):
        # ...
        self.calculation_handler = CalculationHandler(self.settings)
```

**Nachher:**
```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

class MainWindow:
    def __init__(self, settings, container, parent=None):
        # ...
        self.calculation_handler = CalculationHandler(self.settings)
        
        # 🎯 CACHE-MANAGER: Initialisiere alle Caches mit individuellen Konfigurationen
        self._initialize_caches(settings)
    
    def _initialize_caches(self, settings):
        """Initialisiert alle Caches mit individuellen Konfigurationen"""
        # Grid Cache: Großer Cache für viele Surfaces
        cache_manager.register_cache(
            CacheType.GRID,
            max_size=int(getattr(settings, "surface_grid_cache_size", 1000)),
            description="Surface Grid Cache"
        )
        
        # Calc Geometry Cache: Mittelgroßer Cache
        cache_manager.register_cache(
            CacheType.CALC_GEOMETRY,
            max_size=int(getattr(settings, "geometry_cache_size", 1000)),
            description="Geometry Cache for Calculations"
        )
        
        # Calc Grid Cache: Kleinerer Cache (Grids sind groß)
        cache_manager.register_cache(
            CacheType.CALC_GRID,
            max_size=100,
            description="Grid Cache for Calculations"
        )
        
        # Plot Caches: Individuell konfiguriert
        cache_manager.register_cache(
            CacheType.PLOT_SURFACE_ACTORS,
            max_size=500,
            description="Surface Actor Cache"
        )
        
        cache_manager.register_cache(
            CacheType.PLOT_TEXTURE,
            max_size=500,
            description="Texture Cache"
        )
        
        cache_manager.register_cache(
            CacheType.PLOT_GEOMETRY,
            max_size=100,
            description="Plot Geometry Cache"
        )
        
        # 🎯 SHARED GRID GENERATOR: Erstelle einmalig für alle Berechnungen
        from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator
        grid_cache = cache_manager.get_cache(CacheType.GRID)
        self._grid_generator = FlexibleGridGenerator(
            settings,
            grid_cache=grid_cache
        )
    
    def calculate_spl(self, ...):
        # ...
        calculator_cls = self.calculation_handler.select_soundfield_calculator_class()
        
        # Übergebe shared Grid-Generator
        calculator_instance = calculator_cls(
            self.settings,
            self.container.data,
            self.container.calculation_spl,
            grid_generator=self._grid_generator  # Shared!
        )
```

---

## Individuelle Cache-Verwaltung

### Cache einzeln leeren

```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Nur Grid-Cache leeren
cache_manager.clear_cache(CacheType.GRID)

# Nur Plot-Texture-Cache leeren
cache_manager.clear_cache(CacheType.PLOT_TEXTURE)

# Alle Caches leeren
cache_manager.clear_all_caches()
```

### Gezielte Cache-Invalidierung

```python
# Entferne nur Einträge für bestimmte Surface-IDs
def invalidate_surface(surface_id: str):
    def predicate(key):
        # Cache-Key ist Tuple: (surface_id, ...)
        return isinstance(key, tuple) and key[0] == surface_id
    
    cache_manager.invalidate_cache(CacheType.GRID, predicate=predicate)

# Beispiel: Entferne alle Einträge für "surface_1"
invalidate_surface("surface_1")
```

### Cache-Konfiguration zur Laufzeit ändern

```python
# Ändere maximale Größe des Grid-Caches
cache_manager.configure_cache(
    CacheType.GRID,
    max_size=2000  # Erhöhe von 1000 auf 2000
)

# Deaktiviere LRU für einen Cache (nicht empfohlen)
cache_manager.configure_cache(
    CacheType.PLOT_TEXTURE,
    enable_lru=False
)
```

### Cache-Statistiken abrufen

```python
# Statistiken für einen Cache
grid_stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Grid Cache Hit-Rate: {grid_stats['grid']['stats']['hit_rate']:.2f}%")

# Alle Statistiken
all_stats = cache_manager.get_cache_stats()
for cache_type, data in all_stats.items():
    print(f"{cache_type}: {data['stats']['hit_rate']:.2f}% Hit-Rate")

# Globale Statistiken
global_stats = cache_manager.get_global_stats()
print(f"Global Hit-Rate: {global_stats['global_hit_rate']:.2f}%")
print(f"Total Caches: {global_stats['total_caches']}")
print(f"Total Size: {global_stats['total_size']} entries")
```

---

## Vorteile dieser Architektur

### ✅ Globale Verwaltung
- Zentrale Instanz für alle Caches
- Einheitliche Statistiken und Monitoring
- Einfache Debugging-Möglichkeiten

### ✅ Individuelle Kontrolle
- Jeder Cache kann einzeln konfiguriert werden
- Individuelle Größenlimits pro Cache
- Gezielte Invalidierung möglich

### ✅ Flexibilität
- Caches können zur Laufzeit konfiguriert werden
- Einfaches Hinzufügen neuer Cache-Typen
- Backward-kompatibel (Wrapper für bestehenden Code)

### ✅ Performance
- Shared Grid-Generator verhindert Cache-Verlust
- LRU-Cache verhindert unbegrenztes Wachstum
- Thread-safe für Multi-Thread-Umgebungen

---

## Migration-Strategie

### Phase 1: Cache-Manager hinzufügen (keine Breaking Changes)
1. `CacheManager` implementieren ✅
2. `MainWindow` initialisiert Caches
3. Bestehender Code funktioniert weiterhin

### Phase 2: FlexibleGridGenerator migrieren
1. `FlexibleGridGenerator` nutzt Cache-Manager
2. Wrapper für Kompatibilität
3. Shared Grid-Generator in `MainWindow`

### Phase 3: SoundFieldCalculator migrieren
1. Geometry Cache über Cache-Manager
2. Shared Grid-Generator übergeben

### Phase 4: Plot3D migrieren
1. Plot-Caches über Cache-Manager
2. Wrapper für Kompatibilität

### Phase 5: Alte Cache-Implementierungen entfernen
1. Wrapper entfernen
2. Code aufräumen

---

## Beispiel: Cache-Performance-Monitoring

```python
class CacheMonitor:
    """Einfaches Monitoring für Cache-Performance"""
    
    def __init__(self):
        self.cache_manager = cache_manager
    
    def print_stats(self):
        """Druckt Cache-Statistiken"""
        stats = self.cache_manager.get_global_stats()
        
        print("\n=== Cache Performance ===")
        print(f"Global Hit-Rate: {stats['global_hit_rate']:.2f}%")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Total Size: {stats['total_size']} entries")
        print("\nPer-Cache Stats:")
        
        for cache_type, data in stats['caches'].items():
            cache_stats = data['stats']
            print(f"  {cache_type}:")
            print(f"    Hit-Rate: {cache_stats['hit_rate']:.2f}%")
            print(f"    Size: {cache_stats['size']}/{data['config']['max_size']}")
            print(f"    Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")

# Verwendung
monitor = CacheMonitor()
monitor.print_stats()
```

---

## Konfiguration über Settings

```python
# In settings_state.py
class Settings:
    def load_custom_defaults(self):
        # ...
        # Cache-Konfiguration
        self.surface_grid_cache_size = 1000
        self.geometry_cache_size = 1000
        self.calc_grid_cache_size = 100
        self.plot_surface_actors_cache_size = 500
        self.plot_texture_cache_size = 500
        self.plot_geometry_cache_size = 100
```

---

## Zusammenfassung

Der `CacheManager` bietet:
- ✅ **Globale Verwaltung** aller Caches
- ✅ **Individuelle Kontrolle** über jeden Cache
- ✅ **Thread-Safe** Implementation
- ✅ **Detaillierte Statistiken**
- ✅ **Flexible Konfiguration**
- ✅ **Backward-Kompatibel** durch Wrapper

Die Integration kann schrittweise erfolgen, ohne bestehenden Code zu brechen.

