# Vorschlag: Grid-Punkte auf Surface-Begrenzungslinie

## Aktuelle Situation

### Bestehende Implementierung
1. **`_ensure_vertex_coverage`**: Aktiviert nur Ecken (wenn < 2*Resolution entfernt)
2. **`_dilate_mask_minimal`**: Erweitert Maske um 1 Zelle (Randpunkte liegen **außerhalb** des Polygons)
3. **`min_points_per_dimension`**: Standard = 6 (mindestens 6×6 = 36 Punkte pro Surface)
4. **Adaptive Resolution**: Bei kleinen Surfaces wird Resolution reduziert, um Mindestanzahl zu erreichen

### Problem
- Randpunkte liegen primär **außerhalb** des Polygon-Rands (durch Dilatation)
- Keine Punkte **direkt auf** der Surface-Begrenzungslinie
- Bei kleinen Surfaces: Randpunkte reichen möglicherweise für gesamte Fläche
- Bei großen Surfaces: Randpunkte reichen nicht für flächigen Plot

---

## Vorschlag: Adaptive Grid-Punkte auf Boundary

### Konzept

**Neue Funktion: `_add_edge_points_on_boundary`**
- Erzeugt Grid-Punkte **direkt auf** dem Polygon-Rand
- Adaptive Strategie basierend auf:
  - Surface-Größe (Umfang/Fläche)
  - Resolution
  - Minimale Punktanzahl (`min_points_per_dimension`)

### Strategie

#### 1. **Kleine Surfaces** (Umfang < 4 * Resolution)
- **Problem**: Wenige interne Punkte, Randpunkte durch Dilatation reichen möglicherweise
- **Lösung**: 
  - Erzeuge Punkte entlang des gesamten Polygon-Umfangs
  - Abstand zwischen Punkten: `resolution / 2` (doppelte Dichte)
  - Ziel: Erreiche `min_points_per_dimension²` Punkte auch bei sehr kleinen Surfaces

#### 2. **Mittlere Surfaces** (4 * Resolution ≤ Umfang < 20 * Resolution)
- **Problem**: Gemischte Situation
- **Lösung**:
  - Erzeuge Punkte entlang des Polygon-Umfangs
  - Abstand: `resolution * 0.5` bis `resolution * 1.0` (abhängig von benötigter Punktanzahl)
  - Ergänze interne Punkte durch normale Grid-Generierung

#### 3. **Große Surfaces** (Umfang ≥ 20 * Resolution)
- **Problem**: Viele interne Punkte vorhanden, aber Rand sollte auch gut abgedeckt sein
- **Lösung**:
  - Erzeuge Punkte entlang des Polygon-Umfangs
  - Abstand: `resolution * 0.5` (doppelte Dichte am Rand für bessere Triangulation)
  - Normale Grid-Generierung für interne Punkte

---

## Implementierungsvorschlag

### Funktion: `_add_edge_points_on_boundary`

```python
def _add_edge_points_on_boundary(
    self,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    geometry: SurfaceGeometry,
    surface_mask_strict: np.ndarray,
    resolution: float,
    min_points_per_dimension: int = 6,
) -> np.ndarray:
    """
    Erweitert die Maske um Punkte direkt auf dem Polygon-Rand.
    
    Strategie:
    - Berechne Polygon-Umfang
    - Erzeuge Punkte entlang des Umfangs mit adaptivem Abstand
    - Finde nächstgelegenen Grid-Punkt für jeden Boundary-Punkt
    - Aktiviere diese Grid-Punkte in der Maske
    
    Args:
        X_grid: X-Koordinaten des Grids
        Y_grid: Y-Koordinaten des Grids
        geometry: SurfaceGeometry mit Polygon-Definition
        surface_mask_strict: Strikte Maske (vor Dilatation)
        resolution: Grid-Resolution
        min_points_per_dimension: Minimale Punktanzahl pro Dimension
        
    Returns:
        Erweiterte Maske mit Boundary-Punkten
    """
```

### Schritte

1. **Polygon-Umfang berechnen**:
   ```python
   perimeter = 0.0
   for i in range(len(points)):
       p1 = points[i]
       p2 = points[(i + 1) % len(points)]
       dx = p2['x'] - p1['x']
       dy = p2['y'] - p1['y']
       perimeter += math.hypot(dx, dy)
   ```

2. **Adaptive Punkt-Abstand bestimmen**:
   ```python
   if perimeter < 4 * resolution:
       # Sehr kleine Surface: Dichte Abdeckung
       edge_point_spacing = resolution / 2.0
       min_edge_points = min_points_per_dimension ** 2
   elif perimeter < 20 * resolution:
       # Mittlere Surface: Moderate Abdeckung
       edge_point_spacing = resolution * 0.7
       min_edge_points = max(8, int(perimeter / resolution))
   else:
       # Große Surface: Standard-Rand-Abdeckung
       edge_point_spacing = resolution * 0.5
       min_edge_points = max(12, int(perimeter / (resolution * 0.5)))
   ```

3. **Punkte entlang Polygon-Rand erzeugen**:
   ```python
   edge_points = []
   for i in range(len(points)):
       p1 = points[i]
       p2 = points[(i + 1) % len(points)]
       
       segment_length = math.hypot(
           p2['x'] - p1['x'],
           p2['y'] - p1['y']
       )
       n_segment_points = max(1, int(segment_length / edge_point_spacing))
       
       for j in range(n_segment_points + 1):
           t = j / n_segment_points if n_segment_points > 0 else 0
           x = p1['x'] + t * (p2['x'] - p1['x'])
           y = p1['y'] + t * (p2['y'] - p1['y'])
           edge_points.append((x, y))
   ```

4. **Nächstgelegene Grid-Punkte finden und aktivieren**:
   ```python
   x_flat = X_grid.ravel()
   y_flat = Y_grid.ravel()
   mask_flat = surface_mask_strict.ravel()
   
   max_distance = resolution * 0.7  # Nur wenn Grid-Punkt nah genug
   
   for ex, ey in edge_points:
       dx = x_flat - ex
       dy = y_flat - ey
       d2 = dx * dx + dy * dy
       nearest_idx = int(np.argmin(d2))
       
       if d2[nearest_idx] <= max_distance ** 2:
           mask_flat[nearest_idx] = True
   ```

---

## Integration in bestehenden Code

### Erweiterte Reihenfolge

**Für planare/schräge Surfaces:**
```python
# 1. Erstelle strikte Maske
surface_mask_strict = self._create_surface_mask(X_grid, Y_grid, geometry)

# 2. Sicherstelle Ecken-Abdeckung
surface_mask_strict = self._ensure_vertex_coverage(
    X_grid, Y_grid, geometry, surface_mask_strict
)

# 3. 🆕 NEU: Füge Boundary-Punkte hinzu
surface_mask_strict = self._add_edge_points_on_boundary(
    X_grid, Y_grid, geometry, surface_mask_strict,
    resolution, min_points_per_dimension
)

# 4. Erweitere Maske (Dilatation)
surface_mask = self._dilate_mask_minimal(surface_mask_strict)
```

**Für vertikale Surfaces:**
- Ähnliche Logik in (u,v)-Koordinaten
- Polygon-Umfang in (u,v)-Ebene berechnen

---

## Vorteile

1. **✅ Punkte direkt auf Boundary**: Bessere Abdeckung der Surface-Kontur
2. **✅ Adaptive Strategie**: Berücksichtigt Surface-Größe und Resolution
3. **✅ Kleine Surfaces**: Sichert Mindestanzahl von Punkten auch bei sehr kleinen Flächen
4. **✅ Große Surfaces**: Bessere Rand-Abdeckung für glatte Triangulation
5. **✅ Resolution-bewusst**: Abstände basieren auf Grid-Resolution

---

## Beispiel-Szenarien

### Beispiel 1: Sehr kleine Surface (Umfang = 1m, Resolution = 0.5m)
- **Umfang**: 1m < 4 * 0.5m = 2m → **Kleine Surface**
- **Edge-Punkt-Abstand**: 0.5m / 2 = 0.25m
- **Ergebnis**: ~4 Punkte auf Umfang, zusätzlich zu internen Punkten
- **Vorteil**: Erreicht auch bei kleinen Surfaces `min_points_per_dimension²` Punkte

### Beispiel 2: Mittlere Surface (Umfang = 5m, Resolution = 0.5m)
- **Umfang**: 5m ≥ 2m und < 10m → **Mittlere Surface**
- **Edge-Punkt-Abstand**: 0.5m * 0.7 = 0.35m
- **Ergebnis**: ~14 Punkte auf Umfang
- **Vorteil**: Gute Balance zwischen Rand- und Flächen-Abdeckung

### Beispiel 3: Große Surface (Umfang = 30m, Resolution = 0.5m)
- **Umfang**: 30m ≥ 10m → **Große Surface**
- **Edge-Punkt-Abstand**: 0.5m * 0.5 = 0.25m
- **Ergebnis**: ~120 Punkte auf Umfang
- **Vorteil**: Sehr gute Rand-Abdeckung für präzise Triangulation

---

## Offene Fragen / Entscheidungen

1. **Sollen Boundary-Punkte Ecken-Punkte ersetzen oder ergänzen?**
   - **Empfehlung**: Ergänzen (beide Funktionen behalten)

2. **Sollen Boundary-Punkte vor oder nach `_ensure_vertex_coverage` erzeugt werden?**
   - **Empfehlung**: Nach `_ensure_vertex_coverage` (Ecken haben Priorität)

3. **Wie mit vertikalen Surfaces umgehen?**
   - **Empfehlung**: Gleiche Logik in (u,v)-Koordinaten

4. **Sollen Boundary-Punkte auch bei sehr groben Resolutions verwendet werden?**
   - **Empfehlung**: Ja, aber mit größerem Maximal-Abstand (z.B. `resolution * 1.0`)

---

## Nächste Schritte

1. ✅ Implementierung von `_add_edge_points_on_boundary`
2. ✅ Integration in `build_single_surface_grid`
3. ✅ Anpassung für vertikale Surfaces
4. ✅ Tests mit verschiedenen Surface-Größen
5. ✅ Vergleich: Vorher/Nachher (Anzahl Punkte, Plot-Qualität)

