# Grid-Punkte vs. Vertices: Auflösungsverhältnis und Mapping-Strategie

## Die zentrale Frage

**Liegen Grid-Punkte innerhalb der Vertices, oder ist die Vertex-Auflösung höher als die Grid-Auflösung?**

---

## Antwort: Vertices haben VIEL NIEDRIGERE Auflösung als das Grid

### 1. Grid-Erstellung (FlexibleGridGenerator.py, Zeile 949-953)

```python
# Grid wird mit fester Resolution erstellt
sound_field_x = np.arange(min_x, max_x + resolution, resolution)
sound_field_y = np.arange(min_y, max_y + resolution, resolution)
X_grid, Y_grid = np.meshgrid(sound_field_x, sound_field_y, indexing='xy')
```

**Beispiel:**
- Surface: 10m × 10m Rechteck
- Resolution: 0.05m (5cm)
- Grid-Punkte: `200 × 200 = 40.000 Punkte`
- **Grid-Punkte sind regelmäßig verteilt innerhalb des Polygons**

### 2. Triangulation (FlexibleGridGenerator.py, Zeile 2256)

```python
# Triangulation basiert auf Polygon-Ecken
tris = triangulate_points(geom.points)  # geom.points = Polygon-Ecken!
```

**Beispiel für ein einfaches Rechteck:**
- Input: 4 Polygon-Ecken (Surface-Definition)
- Output: 2 Dreiecke → 6 Vertices (3 Vertices pro Dreieck, mit Duplikaten an Ecken)
- **Vertices = die Polygon-Ecken**, NICHT ein feines Mesh!

**Beispiel für ein komplexeres Polygon:**
- Input: 6 Polygon-Ecken
- Output: 4 Dreiecke → 12 Vertices
- **Wieder: Nur die Polygon-Ecken!**

### 3. Das Verhältnis

| Aspekt | Grid | Vertices |
|--------|------|----------|
| **Anzahl** | 40.000 Punkte (10m×10m @ 0.05m) | 6-12 Punkte (nur Ecken) |
| **Verteilung** | Regelmäßig, dicht gepackt | Nur an Polygon-Grenzen |
| **Auflösung** | Hoch (z.B. 5cm Abstand) | Sehr niedrig (nur Ecken) |
| **Position** | Innerhalb des Polygons | Exakt auf Polygon-Grenzen |

---

## Visualisierung

```
                    Grid (40.000 Punkte)
                    
    ┌─────────────────────────────────────┐
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │ ← Grid-Punkte
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │   (regelmäßig, dicht)
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │
    │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  │
    └─────────────────────────────────────┘
    ↑                                   ↑
    │                                   │
    Vertices (nur 4 Ecken)            Vertices
    (Polygon-Grenzen)
```

**Wichtig:** 
- Grid-Punkte liegen **innerhalb** der Polygon-Grenzen
- Vertices liegen **auf** den Polygon-Grenzen
- Grid-Punkte haben SPL-Werte, Vertices nicht (direkt)

---

## Warum Nearest-Neighbor Interpolation?

### Das Problem:

1. **Grid-Punkte** haben SPL-Werte (40.000 Punkte)
2. **Vertices** sind nur die Polygon-Ecken (6-12 Punkte)
3. **Vertices liegen NICHT auf Grid-Punkten** - sie sind die Grenzen
4. Wir müssen für jeden Vertex den SPL-Wert bestimmen

### Die Lösung:

```python
# Plot3DSPL.py, Zeile 1021-1027
spl_at_verts = griddata(
    points_orig,      # Grid-Positionen (40.000 Punkte)
    values_orig,      # Grid-SPL-Werte (40.000 Werte)
    points_new,       # Vertex-Positionen (6-12 Punkte)
    method='nearest', # Finde nächstgelegenen Grid-Punkt
    fill_value=np.nan
)
```

**Warum `nearest`?**
- Vertices liegen nicht auf Grid-Punkten
- Wir wollen den **exakten SPL-Wert** des nächstgelegenen Grid-Punkts
- Bilineare Interpolation würde Werte zwischen Grid-Punkten schätzen
- → **Nearest-Neighbor ist korrekt**, weil wir diskrete Grid-Werte haben

---

## Ist Nearest-Neighbor optimal?

### Aktuelle Situation:

✅ **Vorteile von Nearest-Neighbor:**
- Exakte Grid-Werte werden übernommen (keine Schätzung)
- Einfach und schnell
- Funktioniert gut, wenn Grid fein genug ist

❌ **Potenzielle Probleme:**

1. **Große Distanzen bei kleinen Polygonen:**
   - Bei sehr kleinen Surfaces könnte ein Vertex weit vom nächsten Grid-Punkt entfernt sein
   - → Ungenauer SPL-Wert am Rand

2. **Nur 2D-Interpolation (X, Y):**
   - Z-Koordinate wird ignoriert
   - Bei schrägen Flächen könnte das zu Ungenauigkeiten führen

3. **Keine Gewichtung:**
   - Alle Nachbarn werden gleich behandelt
   - Keine Berücksichtigung der Entfernung

### Alternative: Bilineare Interpolation?

**Würde das besser sein?**
- **Nein, nicht unbedingt:**
  - Grid-Werte sind bereits diskret (berechnete Werte)
  - Bilineare Interpolation würde zwischen Grid-Punkten schätzen
  - Das wäre eine zusätzliche Unsicherheit
  - Nearest-Neighbor behält die Originalwerte bei

**Aber:** Wenn das Grid sehr fein ist und Vertices zwischen Grid-Punkten liegen, könnte bilineare Interpolation glattere Übergänge geben.

---

## Verbesserungsmöglichkeiten

### 1. Surface-bewusste Interpolation

```python
# Nur Grid-Punkte verwenden, die tatsächlich auf der Surface liegen
mask_flat = surface_mask.ravel().astype(bool)
points_orig = points_orig[mask_flat]  # Nur Punkte innerhalb der Surface
values_orig = values_orig[mask_flat]
```

**Aktuell bereits implementiert!** (Zeile 988-993)

### 2. 3D-Interpolation für schräge Flächen

```python
# Statt nur X, Y auch Z berücksichtigen
points_orig_3d = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])
points_new_3d = triangulated_vertices  # X, Y, Z

spl_at_verts = griddata(
    points_orig_3d,
    values_orig,
    points_new_3d,
    method='nearest'
)
```

**Potenzielle Verbesserung für schräge Flächen**

### 3. Adaptive Resolution für Vertices

Statt nur Polygon-Ecken zu verwenden, könnte man zusätzliche Vertices entlang der Polygon-Grenzen einfügen:
- Mehr Vertices = feinere Darstellung
- Aber: Mehr Rechenaufwand bei der Interpolation
- **Aktuell: Keine Notwendigkeit**, da PyVista zwischen Vertices interpoliert

---

## Zusammenfassung

### ✅ Aktueller Ansatz ist korrekt:

1. **Grid**: Viele Punkte (hohe Auflösung) → SPL-Werte
2. **Vertices**: Wenige Punkte (niedrige Auflösung) → Nur Polygon-Ecken
3. **Interpolation**: Nearest-Neighbor von Grid → Vertices
4. **Visualisierung**: PyVista interpoliert zwischen Vertices

### ⚠️ Wichtige Erkenntnisse:

- **Vertices liegen NICHT auf Grid-Punkten** (sie sind Polygon-Grenzen)
- **Grid-Auflösung ist VIEL höher** als Vertex-Auflösung
- **Nearest-Neighbor ist sinnvoll**, um exakte Grid-Werte zu behalten
- **PyVista interpoliert dann** zwischen den wenigen Vertices für die Visualisierung

### 🎯 Fazit:

**Die Vertices-Auflösung ist DEUTLICH NIEDRIGER als die Grid-Auflösung.** Das ist auch so gewollt:
- Grid: Für die Berechnung (viele Punkte = genau)
- Vertices: Für die Visualisierung (wenige Punkte = schnell)
- Nearest-Neighbor: Verbindet beides korrekt

Die aktuell implementierte Lösung ist sinnvoll und korrekt! 🎉