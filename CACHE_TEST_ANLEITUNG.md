# Cache-Test Anleitung

## Übersicht

Dieses Dokument beschreibt, wie die Cache-Logik getestet werden kann.

---

## 1. Automatische Tests

### Test-Skript ausführen

```bash
cd /Users/MGraf/Python/LFO_Umgebung
python test_cache_functionality.py
```

**Oder mit aktivierter Umgebung:**
```bash
/Users/MGraf/Python/LFO_Umgebung/Venv_FEM/bin/python test_cache_functionality.py
```

### Getestete Funktionalitäten

1. ✅ **Cache-Manager Grundfunktionalität**
   - Cache registrieren
   - Cache abrufen
   - Cache-Statistiken

2. ✅ **LRU-Cache Verhalten**
   - Cache-Füllung
   - LRU-Eviction (älteste Einträge werden entfernt)
   - Cache-Zugriff

3. ✅ **Cache Hit/Miss**
   - Cache Miss bei nicht vorhandenem Key
   - Cache Hit bei vorhandenem Key
   - Hit-Rate Berechnung

4. ✅ **Gezielte Cache-Invalidierung**
   - Invalidierung mit Prädikat-Funktion
   - Nur bestimmte Einträge werden entfernt
   - Andere Einträge bleiben erhalten

5. ✅ **Shared Grid-Generator**
   - Grid-Generator mit Shared Cache
   - Cache-Persistenz über mehrere Instanzen

6. ✅ **Surface-Cache Invalidierung**
   - `invalidate_surface_cache()` Funktionalität
   - Nur betroffene Surfaces werden invalidiert

7. ✅ **Cache-Statistiken**
   - Hits, Misses, Size, Hit-Rate
   - Globale Statistiken

8. ✅ **Thread-Safety**
   - Parallele Zugriffe ohne Fehler
   - Lock-Mechanismus funktioniert

9. ✅ **Cache-Konfiguration**
   - Zur Laufzeit konfigurierbar
   - max_size, description

10. ✅ **Mehrere Cache-Typen**
    - Verschiedene Cache-Typen gleichzeitig
    - Globale Statistiken über alle Caches

---

## 2. Manuelle Tests in der Anwendung

### Test 1: Cache-Hit bei wiederholter Berechnung

**Schritte:**
1. Öffne LFO
2. Erstelle ein Surface
3. Führe SPL-Berechnung aus (erste Berechnung)
4. Führe SPL-Berechnung erneut aus (zweite Berechnung)

**Erwartetes Ergebnis:**
- Erste Berechnung: Cache Miss (Grid wird berechnet)
- Zweite Berechnung: Cache Hit (Grid aus Cache)
- Zweite Berechnung sollte **deutlich schneller** sein

**Überprüfung:**
```python
# In Python-Konsole oder Debugger
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Grid Cache Hit-Rate: {stats['grid']['stats']['hit_rate']:.2f}%")
print(f"Hits: {stats['grid']['stats']['hits']}")
print(f"Misses: {stats['grid']['stats']['misses']}")
```

### Test 2: Gezielte Surface-Cache Invalidierung

**Schritte:**
1. Erstelle zwei Surfaces: "surface_1" und "surface_2"
2. Führe SPL-Berechnung aus (beide Surfaces werden gecacht)
3. Ändere nur "surface_1" (z.B. Höhe ändern)
4. Führe SPL-Berechnung erneut aus

**Erwartetes Ergebnis:**
- "surface_1": Cache Miss (wird neu berechnet)
- "surface_2": Cache Hit (aus Cache)
- Nur "surface_1" sollte neu berechnet werden

**Überprüfung:**
```python
# Cache-Statistiken vor/nach Änderung vergleichen
stats_before = cache_manager.get_cache_stats(CacheType.GRID)
# ... Änderung ...
stats_after = cache_manager.get_cache_stats(CacheType.GRID)
# Prüfe dass nur surface_1 invalidiert wurde
```

### Test 3: Hide/Disable Cache-Invalidierung

**Schritte:**
1. Erstelle ein Surface
2. Führe SPL-Berechnung aus (Surface wird gecacht)
3. Setze Surface auf "hide"
4. Führe SPL-Berechnung erneut aus

**Erwartetes Ergebnis:**
- Cache für verstecktes Surface wird gelöscht
- Verstecktes Surface wird nicht berechnet
- Bei Unhide: Surface wird neu berechnet (Cache wurde gelöscht)

**Überprüfung:**
```python
# Prüfe Cache-Größe vor/nach hide
stats_before = cache_manager.get_cache_stats(CacheType.GRID)
# ... hide ...
stats_after = cache_manager.get_cache_stats(CacheType.GRID)
# Cache-Größe sollte kleiner sein
```

### Test 4: Shared Grid-Generator

**Schritte:**
1. Öffne LFO
2. Führe erste SPL-Berechnung aus
3. Führe zweite SPL-Berechnung aus (ohne Änderungen)

**Erwartetes Ergebnis:**
- Beide Berechnungen verwenden denselben Grid-Generator
- Cache bleibt zwischen Berechnungen erhalten
- Zweite Berechnung nutzt Cache

**Überprüfung:**
```python
# Prüfe dass Grid-Generator geteilt wird
main_window = ...  # MainWindow-Instanz
grid_gen_1 = main_window._grid_generator
# ... Berechnung ...
grid_gen_2 = main_window._grid_generator
assert grid_gen_1 is grid_gen_2, "Sollte dieselbe Instanz sein"
```

### Test 5: Gruppen-Höhenänderung ohne Rand-Surface-Änderung

**Schritte:**
1. Erstelle eine Surface-Gruppe mit einem Surface
2. Erstelle ein Rand-Surface (nicht in Gruppe, teilt Punkte)
3. Ändere Höhe der Gruppe
4. Prüfe Rand-Surface

**Erwartetes Ergebnis:**
- Gruppe wird verschoben ✅
- Rand-Surface bleibt unverändert ✅
- Geteilte Punkte werden kopiert, nicht verschoben

**Überprüfung:**
```python
# Prüfe Punkt-Koordinaten vor/nach Änderung
rand_surface_points_before = [...]
# ... Gruppen-Änderung ...
rand_surface_points_after = [...]
assert rand_surface_points_before == rand_surface_points_after, "Rand-Surface sollte unverändert sein"
```

---

## 3. Performance-Tests

### Test: Cache-Performance messen

**Skript:**
```python
import time
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Cache leeren
cache_manager.clear_cache(CacheType.GRID)
cache = cache_manager.get_cache(CacheType.GRID)

# Test 1: Cache Miss (erste Berechnung)
start = time.perf_counter()
# ... Berechnung ...
duration_miss = time.perf_counter() - start
print(f"Cache Miss: {duration_miss:.3f}s")

# Test 2: Cache Hit (zweite Berechnung)
start = time.perf_counter()
# ... Berechnung ...
duration_hit = time.perf_counter() - start
print(f"Cache Hit: {duration_hit:.3f}s")

# Verbesserung berechnen
improvement = ((duration_miss - duration_hit) / duration_miss) * 100
print(f"Verbesserung: {improvement:.1f}%")
```

**Erwartete Ergebnisse:**
- Cache Hit sollte **50-90% schneller** sein als Cache Miss
- Hit-Rate sollte nach mehreren Berechnungen **>70%** sein

---

## 4. Debugging und Monitoring

### Cache-Statistiken anzeigen

**In Python-Konsole:**
```python
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType

# Einzelner Cache
stats = cache_manager.get_cache_stats(CacheType.GRID)
print(f"Grid Cache:")
print(f"  Hits: {stats['grid']['stats']['hits']}")
print(f"  Misses: {stats['grid']['stats']['misses']}")
print(f"  Hit-Rate: {stats['grid']['stats']['hit_rate']:.2f}%")
print(f"  Size: {stats['grid']['stats']['size']}/{stats['grid']['config']['max_size']}")

# Alle Caches
global_stats = cache_manager.get_global_stats()
print(f"\nGlobal:")
print(f"  Total Caches: {global_stats['total_caches']}")
print(f"  Global Hit-Rate: {global_stats['global_hit_rate']:.2f}%")
print(f"  Total Size: {global_stats['total_size']}")
```

### Cache-Inhalt anzeigen

**Für Debugging:**
```python
cache = cache_manager.get_cache(CacheType.GRID)
with cache._lock:
    print(f"Cache-Keys: {list(cache._cache.keys())[:10]}")  # Erste 10 Keys
    print(f"Cache-Größe: {len(cache._cache)}")
```

### Cache zurücksetzen

**Für saubere Tests:**
```python
# Einzelner Cache
cache_manager.clear_cache(CacheType.GRID)

# Alle Caches
cache_manager.clear_all_caches()

# Statistiken zurücksetzen (behält Cache-Inhalt)
cache_manager.reset_stats(CacheType.GRID)
```

---

## 5. Integrationstests

### Test: Vollständiger Workflow

**Szenario:**
1. LFO starten
2. Surface erstellen
3. SPL berechnen (Cache Miss)
4. SPL erneut berechnen (Cache Hit)
5. Surface ändern (Cache Invalidierung)
6. SPL berechnen (Cache Miss für geändertes Surface)
7. Hide Surface (Cache Invalidierung)
8. Unhide Surface (Cache wird neu erstellt)

**Erwartetes Ergebnis:**
- Alle Schritte funktionieren ohne Fehler
- Cache-Verhalten ist korrekt
- Performance-Verbesserung bei Cache Hits

---

## 6. Fehlerbehandlung

### Häufige Probleme

**Problem 1: Cache wird nicht geteilt**
```python
# Prüfe ob Grid-Generator geteilt wird
assert main_window._grid_generator is not None, "Grid-Generator sollte existieren"
assert calculator._grid_generator is main_window._grid_generator, "Sollte geteilt werden"
```

**Problem 2: Cache wird nicht invalidiert**
```python
# Prüfe Cache-Größe vor/nach Invalidierung
stats_before = cache.get_stats()
grid_generator.invalidate_surface_cache("surface_1")
stats_after = cache.get_stats()
assert stats_after.size < stats_before.size, "Cache sollte kleiner sein"
```

**Problem 3: Thread-Safety Probleme**
```python
# Prüfe auf Race Conditions
# Führe parallele Zugriffe aus und prüfe auf Fehler
```

---

## 7. Best Practices

### Testen in der Entwicklung

1. **Vor jeder Änderung:** Tests ausführen
2. **Nach jeder Änderung:** Tests erneut ausführen
3. **Bei Problemen:** Debugging mit Statistiken

### Monitoring in Produktion

1. **Cache-Statistiken regelmäßig prüfen**
2. **Hit-Rate überwachen** (sollte >70% sein)
3. **Cache-Größe überwachen** (sollte nicht unbegrenzt wachsen)

### Performance-Optimierung

1. **Cache-Größe anpassen** wenn nötig
2. **File-Cache aktivieren** für große Projekte
3. **Cache-Invalidierung optimieren** wenn zu langsam

---

## 8. Zusammenfassung

### Test-Methoden

1. ✅ **Automatische Tests:** `test_cache_functionality.py`
2. ✅ **Manuelle Tests:** In der Anwendung
3. ✅ **Performance-Tests:** Messung der Verbesserung
4. ✅ **Integrationstests:** Vollständiger Workflow

### Wichtige Metriken

- **Hit-Rate:** Sollte >70% sein
- **Performance-Verbesserung:** 50-90% bei Cache Hits
- **Cache-Größe:** Sollte begrenzt bleiben (LRU-Eviction)

Die Tests sollten regelmäßig ausgeführt werden, um sicherzustellen, dass die Cache-Logik korrekt funktioniert! 🚀

