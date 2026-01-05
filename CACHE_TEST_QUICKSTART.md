# Cache-Test Quickstart

## Schnellstart

### 1. Automatische Tests ausführen

```bash
cd /Users/MGraf/Python/LFO_Umgebung
python test_cache_functionality.py
```

**Erwartete Ausgabe:**
```
============================================================
CACHE-FUNKTIONALITÄT TEST-SUITE
============================================================

Test: Cache-Manager Grundfunktionalität
============================================================
✅ PASSED: Cache-Manager Grundfunktionalität

...

============================================================
TEST-ZUSAMMENFASSUNG
============================================================
✅ Bestanden: 10
❌ Fehlgeschlagen: 0
📊 Gesamt: 10

✅✅✅ ALLE TESTS BESTANDEN ✅✅✅
```

---

## 2. Cache-Statistiken in der Anwendung anzeigen

### In Python-Konsole (während LFO läuft)

```python
# Einfach
from Module_LFO.Modules_Init.CacheMonitor import print_cache_stats
print_cache_stats()

# Detailliert
from Module_LFO.Modules_Init.CacheMonitor import print_detailed_cache_stats
print_detailed_cache_stats()

# Nur Grid-Cache
from Module_LFO.Modules_Init.CacheManager import CacheType
print_cache_stats(CacheType.GRID)
```

### In Debugger oder Code

```python
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
monitor.print_stats()  # Kompakt
monitor.print_detailed_stats()  # Detailliert
monitor.print_cache_contents(CacheType.GRID, max_keys=20)  # Cache-Inhalt
```

---

## 3. Praktische Test-Szenarien

### Szenario 1: Cache-Hit testen

```python
# 1. Cache-Statistiken vorher
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor
monitor = CacheMonitor()
stats_before = cache_manager.get_cache_stats(CacheType.GRID)

# 2. Berechnung ausführen (in LFO UI)
# ... SPL berechnen ...

# 3. Cache-Statistiken nachher
stats_after = cache_manager.get_cache_stats(CacheType.GRID)

# 4. Vergleich
monitor.compare_stats(stats_before, stats_after)
```

**Erwartetes Ergebnis:**
- Erste Berechnung: Misses erhöhen sich
- Zweite Berechnung: Hits erhöhen sich
- Hit-Rate sollte steigen

### Szenario 2: Surface-Cache Invalidierung testen

```python
# 1. Cache füllen
# ... Berechnung mit Surface "surface_1" ...

# 2. Cache-Statistiken
stats_before = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Cache-Größe vorher: {stats_before['grid']['stats']['size']}")

# 3. Surface-Cache invalidieren
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator
grid_generator = main_window._grid_generator
grid_generator.invalidate_surface_cache("surface_1")

# 4. Cache-Statistiken nachher
stats_after = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Cache-Größe nachher: {stats_after['grid']['stats']['size']}")

# 5. Prüfe dass surface_1 entfernt wurde
cache = cache_manager.get_cache(CacheType.GRID)
# Prüfe Cache-Inhalt
```

**Erwartetes Ergebnis:**
- Cache-Größe sollte kleiner sein
- Nur Einträge für "surface_1" sollten entfernt worden sein

### Szenario 3: Performance-Messung

```python
import time
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()

# Cache leeren
cache_manager.clear_cache(CacheType.GRID)

# Test 1: Cache Miss
start = time.perf_counter()
# ... Berechnung ausführen ...
duration_miss = time.perf_counter() - start
print(f"Cache Miss: {duration_miss:.3f}s")

# Test 2: Cache Hit
start = time.perf_counter()
# ... Gleiche Berechnung erneut ausführen ...
duration_hit = time.perf_counter() - start
print(f"Cache Hit: {duration_hit:.3f}s")

# Verbesserung
improvement = ((duration_miss - duration_hit) / duration_miss) * 100
print(f"Verbesserung: {improvement:.1f}%")

# Statistiken
monitor.print_stats(CacheType.GRID)
```

**Erwartetes Ergebnis:**
- Cache Hit sollte 50-90% schneller sein
- Hit-Rate sollte >70% sein

---

## 4. Debugging bei Problemen

### Problem: Cache wird nicht geteilt

```python
# Prüfe Grid-Generator
main_window = ...  # MainWindow-Instanz
grid_gen_1 = main_window._grid_generator

# Prüfe Calculator
calculator = ...  # SoundFieldCalculator-Instanz
grid_gen_2 = calculator._grid_generator

print(f"Grid-Generator geteilt: {grid_gen_1 is grid_gen_2}")
```

### Problem: Cache wird nicht invalidiert

```python
# Prüfe Cache-Größe vor/nach Invalidierung
cache = cache_manager.get_cache(CacheType.GRID)
stats_before = cache.get_stats()

# Invalidierung
grid_generator.invalidate_surface_cache("surface_1")

stats_after = cache.get_stats()
print(f"Größe vorher: {stats_before.size}")
print(f"Größe nachher: {stats_after.size}")
print(f"Unterschied: {stats_before.size - stats_after.size}")
```

### Problem: Cache-Inhalt anzeigen

```python
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor

monitor = CacheMonitor()
monitor.print_cache_contents(CacheType.GRID, max_keys=20)
```

---

## 5. Häufige Test-Cases

### Test-Case 1: Basis-Funktionalität

```python
# Cache registrieren, füllen, abrufen
cache = cache_manager.register_cache(CacheType.GRID, max_size=10)
cache.set("key1", "value1")
value = cache.get("key1")
assert value == "value1"
```

### Test-Case 2: LRU-Eviction

```python
# Cache mit max_size=3 füllen
cache = cache_manager.get_cache(CacheType.GRID)
for i in range(5):
    cache.set(f"key_{i}", f"value_{i}")

# Älteste sollten entfernt sein
assert cache.get("key_0") is None
assert cache.get("key_4") is not None
```

### Test-Case 3: Gezielte Invalidierung

```python
# Cache füllen
cache.set(("surface_1", ...), "grid_1")
cache.set(("surface_2", ...), "grid_2")

# Nur surface_1 invalidieren
def predicate(key):
    return isinstance(key, tuple) and key[0] == "surface_1"

cache_manager.invalidate_cache(CacheType.GRID, predicate=predicate)

# Prüfe Ergebnisse
assert cache.get(("surface_1", ...)) is None
assert cache.get(("surface_2", ...)) is not None
```

---

## 6. Best Practices

### Regelmäßige Tests

1. **Vor Code-Änderungen:** Tests ausführen
2. **Nach Code-Änderungen:** Tests erneut ausführen
3. **Bei Problemen:** Debugging mit CacheMonitor

### Monitoring

1. **Hit-Rate überwachen:** Sollte >70% sein
2. **Cache-Größe überwachen:** Sollte nicht unbegrenzt wachsen
3. **Performance messen:** Cache Hit sollte deutlich schneller sein

### Debugging

1. **Statistiken anzeigen:** `CacheMonitor.print_stats()`
2. **Cache-Inhalt anzeigen:** `CacheMonitor.print_cache_contents()`
3. **Vor/Nach Vergleich:** `CacheMonitor.compare_stats()`

---

## Zusammenfassung

### Test-Methoden

1. ✅ **Automatische Tests:** `test_cache_functionality.py`
2. ✅ **Manuelle Tests:** In der Anwendung
3. ✅ **Monitoring:** `CacheMonitor` Klasse
4. ✅ **Debugging:** Cache-Inhalt und Statistiken anzeigen

### Wichtige Befehle

```python
# Statistiken anzeigen
from Module_LFO.Modules_Init.CacheMonitor import print_cache_stats
print_cache_stats()

# Detaillierte Statistiken
from Module_LFO.Modules_Init.CacheMonitor import print_detailed_cache_stats
print_detailed_cache_stats()

# Cache-Inhalt anzeigen
from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor
monitor = CacheMonitor()
monitor.print_cache_contents(CacheType.GRID)
```

Die Tests sollten regelmäßig ausgeführt werden! 🚀

