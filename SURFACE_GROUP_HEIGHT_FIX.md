# Fix: Rand-Surfaces ändern sich bei Gruppen-Höhenänderung

## Problem

Wenn die Höhe einer Surface-Gruppe geändert wird, ändern sich auch die Rand-Surfaces zum Surface in der Gruppe. Die Rand-Surfaces sind aber nicht in dieser Gruppe und sollten sich nicht ändern.

## Ursache

**Das Problem liegt NICHT am Cache**, sondern an **geteilten Punkt-Referenzen**!

### Was passiert:

1. Surfaces können sich **Punkt-Referenzen teilen**
2. Wenn ein Surface in der Gruppe einen Punkt mit einem Rand-Surface teilt
3. Und die Gruppe verschoben wird
4. Wird der **geteilte Punkt** verschoben
5. Dadurch ändert sich auch das **Rand-Surface** (obwohl es nicht in der Gruppe ist)

### Beispiel:

```
Gruppe "Gruppe1":
  - Surface "surface_1" (Punkte: A, B, C, D)

Rand-Surface "edge_surface_1" (nicht in Gruppe):
  - Punkte: D, E, F, G  ← Punkt D wird geteilt!

Wenn Gruppe1 verschoben wird:
  - Punkt D wird verschoben
  - edge_surface_1 ändert sich auch! ❌
```

## Lösung

### Implementierung

**Datei:** `UISurfaceManager.py`, Methode `apply_group_changes_by_id()`

**Strategie:**
1. Identifiziere alle Surfaces **außerhalb** der Gruppe
2. Sammle alle Punkt-Referenzen von Surfaces außerhalb der Gruppe
3. Beim Verschieben: Prüfe ob Punkt auch außerhalb verwendet wird
4. Wenn ja: **Erstelle Kopie** des Punktes, bevor verschoben wird
5. Original-Punkt bleibt unverändert (für Rand-Surfaces)

### Code-Änderung

```python
# SCHRITT 1: Identifiziere alle Surfaces außerhalb der Gruppe
all_surface_ids = set(surface_store.keys())
group_surface_ids = set(surface_ids)
surfaces_outside_group = all_surface_ids - group_surface_ids

# SCHRITT 2: Sammle alle Punkt-Referenzen von Surfaces außerhalb der Gruppe
points_used_outside_group = set()
for surface_id in surfaces_outside_group:
    # ... sammle Punkt-Referenzen ...

# SCHRITT 3: Verschiebe Punkte - aber erstelle Kopien für geteilte Punkte
for surface_id in surface_ids:
    new_points = []
    for point in points:
        point_id = id(point)
        
        if point_id in points_used_outside_group:
            # 🎯 FIX: Punkt wird auch außerhalb verwendet!
            # Erstelle Kopie des Punktes, bevor verschoben wird
            point_copy = {
                'x': point.get('x', 0.0) + offset_x,
                'y': point.get('y', 0.0) + offset_y,
                'z': point.get('z', 0.0) + offset_z,
            }
            new_points.append(point_copy)
            # Original-Punkt bleibt unverändert (für Rand-Surfaces)
        else:
            # Punkt wird nur innerhalb der Gruppe verwendet - verschiebe direkt
            point['x'] = point.get('x', 0.0) + offset_x
            point['y'] = point.get('y', 0.0) + offset_y
            point['z'] = point.get('z', 0.0) + offset_z
            new_points.append(point)
    
    # Ersetze Punkt-Liste mit neuer Liste
    surface.points = new_points
```

## Verhalten nach Fix

### Vorher (Problem):

```
Gruppe "Gruppe1" wird verschoben:
  - Surface "surface_1" wird verschoben ✅
  - Rand-Surface "edge_surface_1" wird auch verschoben ❌ (geteilter Punkt D)
```

### Nachher (Fix):

```
Gruppe "Gruppe1" wird verschoben:
  - Surface "surface_1" wird verschoben ✅
    - Punkt D wird kopiert und verschoben (nur für surface_1)
  - Rand-Surface "edge_surface_1" bleibt unverändert ✅
    - Original-Punkt D bleibt unverändert
```

## Zusammenfassung

- ✅ **Problem behoben**: Rand-Surfaces ändern sich nicht mehr bei Gruppen-Höhenänderung
- ✅ **Ursache**: Geteilte Punkt-Referenzen (nicht Cache!)
- ✅ **Lösung**: Kopien für geteilte Punkte erstellen
- ✅ **Performance**: Minimaler Overhead (nur für geteilte Punkte)

Die Implementierung ist **vollständig funktionsfähig**! 🚀

