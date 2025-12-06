# Randpunkte und Maske - Analyse

## Antworten auf die Fragen

### 1. Müssen wir noch außerhalb der Surface eine Maske generieren, um Plotanteile außerhalb zu cutten?

**JA, definitiv!**

**Problem:**
- Nach der Delaunay-Triangulation (Zeile 668 in `Plot3DSPL.py`) werden ALLE kombinierten Punkte trianguliert
- Das Grid wurde um einen Punkt erweitert (außerhalb der Surface)
- Die Delaunay-Triangulation kann auch Dreiecke außerhalb der Surface erzeugen
- Diese Dreiecke müssen entfernt werden

**Aktuelle Situation:**
- Das Haupt-Grid verwendet bereits eine `surface_mask` (Zeile 873), die in `build_surface_mesh` verwendet wird
- ABER: Nach der Kombination mit Randpunkten und Delaunay-Triangulation gibt es keine weitere Filterung

**Lösung:**
- Nach der Delaunay-Triangulation sollten wir prüfen, welche Dreiecke außerhalb der Surface liegen
- Dazu prüfen wir den Schwerpunkt (Centroid) jedes Dreiecks
- Nur Dreiecke, deren Schwerpunkt innerhalb des Surface-Polygons liegt, sollten behalten werden

### 2. Werden berechnete Datenpunkte und Randdatenpunkte zusammengeplottet?

**JA, sie werden kombiniert!**

**Aktuelle Implementierung:**
- Zeile 650-651 in `Plot3DSPL.py`: `combined_points = np.vstack([main_points, edge_points_3d])`
- Zeile 668: Delaunay-Triangulation auf alle kombinierten Punkte
- Randpunkte sind AUF den Polygon-Kanten (1cm Auflösung)
- Sie werden mit dem Haupt-Grid kombiniert und dann trianguliert

## Implementierungsvorschlag

### Option 1: Filter nach Delaunay-Triangulation (Empfohlen)

Nach der Delaunay-Triangulation in `_integrate_edge_points_into_mesh`:

1. **Hole Surface-Polygon-Definition** aus `settings.surface_definitions[surface_id]`
2. **Berechne Schwerpunkt** für jedes Dreieck
3. **Prüfe** mit Punkt-in-Polygon-Test, ob Schwerpunkt innerhalb der Surface liegt
4. **Entferne** Dreiecke außerhalb der Surface mit `mesh.extract_cells(cells_to_keep)`

**Vorteile:**
- Einfach zu implementieren
- Keine Änderungen an der Triangulation nötig
- Funktioniert auch mit komplexen Polygonen

### Option 2: Nur Punkte innerhalb der Surface verwenden

Bereits vor der Kombination:
- Filtere erweiterte Grid-Punkte, die außerhalb der Surface liegen
- Verwende nur Randpunkte auf der Kante

**Nachteile:**
- Verliert erweiterte Randpunkte, die außerhalb aber nahe der Kante liegen
- Komplexer, da wir wissen müssen, welche Punkte "erweitert" sind

## Empfehlung

**Option 1** ist besser, weil:
1. Sie die erweiterten Randpunkte behält
2. Sie einfach zu implementieren ist
3. Sie flexibler ist (funktioniert auch mit zukünftigen Änderungen)

## Nächste Schritte

1. Erweitere `_integrate_edge_points_into_mesh`, um `surface_id` und `settings` als Parameter zu erhalten
2. Hole Surface-Polygon-Definition aus `settings.surface_definitions[surface_id]`
3. Nach Delaunay-Triangulation: Filtere Dreiecke außerhalb der Surface
4. Verwende `mesh.extract_cells()` um nur gültige Dreiecke zu behalten



