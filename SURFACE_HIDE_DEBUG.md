# Surface Hide Debug: Was passiert genau?

## Ablauf beim Setzen eines Surfaces auf "hide"

### 1. `on_surface_hide_changed()` wird aufgerufen

**Datei:** `UISurfaceManager.py`, Zeile 2600

**Schritte:**
1. Hide-Status wird gesetzt (Zeile 2631-2637)
2. UI wird aktualisiert (Zeile 2639-2654)
3. **🎯 CACHE-INVALIDIERUNG** (Zeile 2656-2662) - **NEU hinzugefügt!**
4. Wenn `hide_value == True`:
   - Actors werden entfernt (Zeile 2696-2735)
   - Overlays werden aktualisiert (Zeile 2760-2761)
   - **🎯 `calculate_axes()` wird aufgerufen** (Zeile 2770) - **KÖNNTE HÄNGEN!**
   - Grid/Ergebnisdaten werden entfernt (Zeile 2774-2782)

### 2. Cache-Invalidierung

**Code:**
```python
# Zeile 2656-2662
if hasattr(self.main_window, '_grid_generator') and self.main_window._grid_generator:
    grid_generator = self.main_window._grid_generator
    for sid in surfaces_to_update:
        if hasattr(grid_generator, 'invalidate_surface_cache'):
            grid_generator.invalidate_surface_cache(sid)
```

**Was passiert:**
- `invalidate_surface_cache()` wird aufgerufen
- Cache-Manager durchsucht alle Cache-Einträge
- Einträge mit passendem `surface_id` werden entfernt
- Lock wird verwendet (thread-safe)

**Mögliches Problem:** 
- Wenn Cache sehr groß ist, könnte die Durchsuchung langsam sein
- Lock könnte blockieren, wenn gleichzeitig auf Cache zugegriffen wird

### 3. `calculate_axes()` Aufruf

**Code:**
```python
# Zeile 2768-2770
if hasattr(self.main_window, 'calculate_axes'):
    print(f"[PLOT] Surface hide → calculate_axes() (Axis Plot ohne versteckte Surfaces)")
    self.main_window.calculate_axes(update_plot=True)
```

**Was passiert:**
- `SoundFieldCalculatorXaxis` wird erstellt
- `SoundFieldCalculatorYaxis` wird erstellt
- Berechnungen werden durchgeführt
- Plots werden aktualisiert

**Mögliches Problem:**
- Berechnungen könnten hängen, wenn:
  - Keine Daten vorhanden sind
  - Berechnungen sehr lange dauern
  - Deadlock durch gleichzeitige Cache-Zugriffe

---

## Mögliche Ursachen für das Hängen

### 1. Cache-Invalidierung blockiert

**Problem:** Cache-Invalidierung könnte blockieren, wenn:
- Cache sehr groß ist (viele Einträge zu durchsuchen)
- Lock wird von anderem Thread gehalten
- Gleichzeitiger Zugriff auf Cache

**Lösung:** Cache-Invalidierung sollte schnell sein, aber könnte optimiert werden.

### 2. `calculate_axes()` hängt

**Problem:** `calculate_axes()` könnte hängen, wenn:
- Berechnungen sehr lange dauern
- Keine Daten vorhanden sind und Fehler auftreten
- Rekursive Aufrufe (Endlosschleife)

**Lösung:** `calculate_axes()` sollte asynchron aufgerufen werden oder übersprungen werden bei hide.

### 3. Rekursive Aufrufe

**Problem:** Mögliche Endlosschleife:
```
on_surface_hide_changed()
  └─> calculate_axes()
      └─> update_plots_for_surface_state()
          └─> on_surface_hide_changed() (rekursiv?)
```

**Lösung:** `skip_calculations` Flag sollte verwendet werden.

---

## Empfohlene Fixes

### Fix 1: Cache-Invalidierung optimieren

```python
# Schnellere Invalidierung durch direkten Zugriff auf Cache
def invalidate_surface_cache(self, surface_id: str) -> int:
    """Invalidiert Cache für ein spezifisches Surface - OPTIMIERT"""
    # Direkter Zugriff auf Cache ohne Prädikat-Funktion
    with self._grid_cache._lock:
        keys_to_remove = [
            key for key in self._grid_cache._cache.keys()
            if isinstance(key, tuple) and len(key) > 0 and key[0] == surface_id
        ]
        for key in keys_to_remove:
            self._grid_cache._cache.pop(key, None)
        return len(keys_to_remove)
```

### Fix 2: `calculate_axes()` bei hide überspringen oder asynchron aufrufen

```python
# Option 1: Überspringen bei hide
if hide_value:
    # ... Actors entfernen ...
    # calculate_axes() NICHT aufrufen bei hide
    # Nur Overlays aktualisieren
    if hasattr(plotter, 'update_overlays'):
        plotter.update_overlays(self.settings, self.container)

# Option 2: Asynchron aufrufen
if hasattr(self.main_window, 'calculate_axes'):
    # Asynchron aufrufen, damit UI nicht blockiert
    QtCore.QTimer.singleShot(0, lambda: self.main_window.calculate_axes(update_plot=True))
```

### Fix 3: `skip_calculations` Flag verwenden

```python
# Bei hide: Berechnungen überspringen
if hide_value:
    # ... Cache invalidieren ...
    # ... Actors entfernen ...
    # Berechnungen NICHT ausführen
    if not skip_calculations:
        # Nur Overlays aktualisieren, keine Berechnungen
        if hasattr(plotter, 'update_overlays'):
            plotter.update_overlays(self.settings, self.container)
```

---

## Debugging-Schritte

1. **Prüfe ob Cache-Invalidierung hängt:**
   - Logging hinzufügen vor/nach `invalidate_surface_cache()`
   - Prüfe Cache-Größe

2. **Prüfe ob `calculate_axes()` hängt:**
   - Logging vor/nach `calculate_axes()`
   - Prüfe ob Berechnungen durchgeführt werden

3. **Prüfe auf rekursive Aufrufe:**
   - Stack-Trace analysieren
   - Prüfe ob `skip_calculations` Flag verwendet wird

4. **Prüfe Thread-Safety:**
   - Prüfe ob Lock-Deadlocks auftreten
   - Prüfe ob gleichzeitige Cache-Zugriffe auftreten

