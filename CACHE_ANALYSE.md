# Cache-Analyse: Grid, Calc und 3D Plot

## Zusammenfassung

**Hauptproblem:** Die drei Komponenten (Grid, Calc, 3D Plot) verwenden **verschiedene Cache-Instanzen**, die nicht geteilt werden. Jede Berechnung erstellt neue Instanzen, wodurch der Cache bei jedem Aufruf verloren geht.

---

## 1. Aktuelle Cache-Struktur

### 1.1 Grid Cache (FlexibleGridGenerator)

**Datei:** `LFO/Module_LFO/Modules_Calculate/FlexibleGridGenerator.py`

**Cache-Typ:** LRU-Cache mit `OrderedDict`
- **Cache-Key:** `(surface_id, orientation, resolution, min_points, points_signature)`
- **Cache-Wert:** `CachedSurfaceGrid` (optimiert, ~30-40% weniger Memory als `SurfaceGrid`)
- **Größe:** Default 1000 Surfaces (konfigurierbar via `settings.surface_grid_cache_size`)
- **Thread-Safe:** Ja (mit `Lock`)
- **Optional:** File-Cache möglich (`settings.surface_grid_cache_to_file`)

**Cache-Statistiken:**
```python
self._cache_stats = {
    'hits': 0,
    'misses': 0,
    'evictions': 0,
    'size': 0,
    'file_loads': 0,
    'file_saves': 0,
}
```

**Probleme:**
- ❌ Jede `FlexibleGridGenerator`-Instanz hat ihren eigenen Cache
- ❌ Cache geht verloren, wenn neue Instanz erstellt wird
- ✅ LRU-Cache ist gut implementiert
- ✅ Thread-Safe

---

### 1.2 Calc Cache (SoundFieldCalculator)

**Datei:** `LFO/Module_LFO/Modules_Calculate/SoundfieldCalculator.py`

**Cache-Typen:**
1. **Geometry Cache:** Einfaches `dict` - `{source_key: {distances, azimuths, elevations}}`
2. **Grid Cache:** `None` oder gespeichertes Grid
3. **Grid Hash:** Hash der Grid-Parameter

**Probleme:**
- ❌ Kein LRU-Cache (könnte unbegrenzt wachsen)
- ❌ Cache wird bei jeder neuen Instanz geleert
- ❌ Erstellt eigene `FlexibleGridGenerator`-Instanz (Zeile 36), die ihren eigenen Cache hat
- ⚠️ `_grid_cache` und `_grid_hash` werden nicht optimal genutzt

**Instanziierung:**
```python
# Main.py:790 - Bei jedem calculate_spl() Aufruf:
calculator_instance = calculator_cls(self.settings, self.container.data, self.container.calculation_spl)
# SoundfieldCalculator.py:36 - Jede Instanz erstellt neuen Grid-Generator:
self._grid_generator = FlexibleGridGenerator(settings)
```

---

### 1.3 3D Plot Cache (Plot3D)

**Datei:** `LFO/Module_LFO/Modules_Plot/Plot_SPL_3D/Plot3D.py` und `Plot3DSPL.py`

**Cache-Typen:**
1. **Surface Actors:** `dict[str, Any]` - Mesh-Actors pro Surface
2. **Surface Texture Cache:** `dict[str, str]` - Textur-Signaturen
3. **Plot Geometry Cache:** Cache für Plot-Geometrie (plot_x, plot_y)
4. **Time Mode Cache:** Cache für Zeitmodus
5. **Overlay Signatures:** Signaturen für Overlays

**Probleme:**
- ⚠️ Cache ist nicht mit Grid/Calc geteilt
- ✅ Gut strukturiert für Rendering-Optimierungen
- ✅ Mesh-Caching funktioniert gut (Signatur-basiert)

---

## 2. Hauptprobleme

### Problem 1: Keine gemeinsame Cache-Instanz

**Aktuell:**
```
Main.calculate_spl()
  └─> SoundFieldCalculator() [NEUE INSTANZ]
        └─> FlexibleGridGenerator() [NEUE INSTANZ mit leerem Cache]
              └─> Cache ist leer! ❌
```

**Ergebnis:** Bei jeder Berechnung wird der Grid-Cache neu aufgebaut, obwohl die Geometrie unverändert sein könnte.

### Problem 2: Cache-Verlust bei Instanz-Erstellung

- Jede `calculate_spl()`-Aufruf erstellt neue `SoundFieldCalculator`-Instanz
- Jede `SoundFieldCalculator`-Instanz erstellt neue `FlexibleGridGenerator`-Instanz
- Cache geht verloren → Grids werden jedes Mal neu berechnet

### Problem 3: Ineffiziente Cache-Nutzung

- `SoundFieldCalculator._grid_cache` und `_grid_hash` werden nicht optimal genutzt
- `SoundFieldCalculator._geometry_cache` hat kein LRU, könnte unbegrenzt wachsen
- Keine Cache-Statistiken für Calc-Cache

---

## 3. Optimierungsvorschläge

### Optimierung 1: Shared Grid Generator (Höchste Priorität)

**Ziel:** Eine gemeinsame `FlexibleGridGenerator`-Instanz über mehrere Berechnungen hinweg verwenden.

**Implementierung:**
1. `FlexibleGridGenerator` als Singleton oder in `MainWindow` speichern
2. `SoundFieldCalculator` erhält Grid-Generator als Parameter (Dependency Injection)
3. Cache bleibt über mehrere Berechnungen erhalten

**Vorteile:**
- ✅ Grid-Cache bleibt erhalten
- ✅ Deutlich schneller bei wiederholten Berechnungen
- ✅ Weniger Memory-Verbrauch (keine doppelten Grids)

**Code-Änderungen:**
```python
# Main.py - Grid-Generator einmalig erstellen
class MainWindow:
    def __init__(self):
        # ...
        self._grid_generator = FlexibleGridGenerator(self.settings)
    
    def calculate_spl(self):
        calculator_instance = calculator_cls(
            self.settings, 
            self.container.data, 
            self.container.calculation_spl,
            grid_generator=self._grid_generator  # Shared instance
        )
```

### Optimierung 2: LRU-Cache für Geometry Cache

**Ziel:** `SoundFieldCalculator._geometry_cache` mit LRU-Cache versehen.

**Implementierung:**
```python
from collections import OrderedDict

class SoundFieldCalculator:
    def __init__(self, ...):
        # ...
        max_geometry_cache_size = 1000  # Konfigurierbar
        self._geometry_cache: OrderedDict[str, dict] = OrderedDict()
        self._geometry_cache_max_size = max_geometry_cache_size
```

### Optimierung 3: Cache-Statistiken und Monitoring

**Ziel:** Bessere Sichtbarkeit über Cache-Performance.

**Implementierung:**
- Cache-Statistiken für `SoundFieldCalculator` hinzufügen
- Logging von Cache-Hit-Rates
- Optional: UI-Anzeige der Cache-Performance

### Optimierung 4: Cache-Invalidierung bei Änderungen

**Ziel:** Gezielte Cache-Invalidierung statt komplettes Leeren.

**Implementierung:**
- Bei Surface-Änderungen: Nur betroffene Surfaces aus Cache entfernen
- Bei Settings-Änderungen: Nur relevante Cache-Teile invalidieren
- Hash-basierte Invalidation

### Optimierung 5: Plot-Cache mit Grid-Cache synchronisieren

**Ziel:** Plot verwendet bereits gecachte Grids, wenn möglich.

**Implementierung:**
- Plot erhält Referenz auf Grid-Generator
- Prüft Cache vor Neuberechnung
- Verwendet gecachte Grids für Mesh-Erstellung

---

## 4. Empfohlene Implementierungsreihenfolge

1. **Phase 1:** Shared Grid Generator (größter Impact)
   - `MainWindow` erstellt `FlexibleGridGenerator` einmalig
   - `SoundFieldCalculator` erhält Grid-Generator als Parameter
   - Cache bleibt erhalten

2. **Phase 2:** LRU-Cache für Geometry Cache
   - `OrderedDict` für `_geometry_cache`
   - Max-Size konfigurierbar

3. **Phase 3:** Cache-Statistiken
   - Statistiken für alle Caches
   - Logging und Monitoring

4. **Phase 4:** Gezielte Cache-Invalidierung
   - Hash-basierte Invalidation
   - Nur betroffene Teile löschen

---

## 5. Erwartete Verbesserungen

### Performance-Verbesserungen:
- **Grid-Cache:** 50-90% schneller bei wiederholten Berechnungen (je nach Cache-Hit-Rate)
- **Memory:** Weniger doppelte Grids im Speicher
- **Berechnungszeit:** Deutlich reduziert bei unveränderter Geometrie

### Code-Qualität:
- Bessere Trennung von Concerns
- Dependency Injection statt direkter Instanziierung
- Bessere Testbarkeit

---

## 6. Risiken und Überlegungen

### Thread-Safety:
- ✅ `FlexibleGridGenerator` ist bereits thread-safe (Lock vorhanden)
- ⚠️ Bei Shared Instance: Sicherstellen, dass alle Zugriffe thread-safe sind

### Memory-Management:
- LRU-Cache begrenzt automatisch Memory-Verbrauch
- File-Cache kann bei Bedarf aktiviert werden

### Backward Compatibility:
- Änderungen sollten rückwärtskompatibel sein
- Optional: Grid-Generator als Parameter (mit Fallback)

---

## 7. Code-Beispiele

### Beispiel 1: Shared Grid Generator

```python
# Main.py
class MainWindow:
    def __init__(self):
        # ...
        self._grid_generator = FlexibleGridGenerator(self.settings)
    
    def calculate_spl(self, ...):
        calculator_cls = self.calculation_handler.select_soundfield_calculator_class()
        calculator_instance = calculator_cls(
            self.settings, 
            self.container.data, 
            self.container.calculation_spl,
            grid_generator=self._grid_generator  # Shared!
        )
```

```python
# SoundfieldCalculator.py
class SoundFieldCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_spl, grid_generator=None):
        # ...
        if grid_generator is None:
            # Fallback für Backward Compatibility
            self._grid_generator = FlexibleGridGenerator(settings)
        else:
            self._grid_generator = grid_generator  # Shared instance
```

### Beispiel 2: LRU-Cache für Geometry Cache

```python
# SoundfieldCalculator.py
from collections import OrderedDict

class SoundFieldCalculator(ModuleBase):
    def __init__(self, ...):
        # ...
        max_size = getattr(settings, "geometry_cache_size", 1000)
        self._geometry_cache: OrderedDict[str, dict] = OrderedDict()
        self._geometry_cache_max_size = max_size
    
    def _get_geometry_cached(self, source_key: str):
        """Holt Geometry aus Cache mit LRU-Update."""
        if source_key in self._geometry_cache:
            # Move to end (most recently used)
            self._geometry_cache.move_to_end(source_key)
            return self._geometry_cache[source_key]
        return None
    
    def _set_geometry_cached(self, source_key: str, geometry_data: dict):
        """Speichert Geometry im Cache mit LRU-Eviction."""
        if source_key in self._geometry_cache:
            self._geometry_cache.move_to_end(source_key)
        else:
            # Prüfe ob Cache-Limit erreicht
            if len(self._geometry_cache) >= self._geometry_cache_max_size:
                # Entferne ältesten Eintrag
                self._geometry_cache.popitem(last=False)
        self._geometry_cache[source_key] = geometry_data
```

---

## 8. Nächste Schritte

1. ✅ Analyse abgeschlossen
2. ⏳ Implementierung Phase 1: Shared Grid Generator
3. ⏳ Implementierung Phase 2: LRU-Cache für Geometry Cache
4. ⏳ Implementierung Phase 3: Cache-Statistiken
5. ⏳ Implementierung Phase 4: Gezielte Cache-Invalidierung
6. ⏳ Testing und Performance-Messung

