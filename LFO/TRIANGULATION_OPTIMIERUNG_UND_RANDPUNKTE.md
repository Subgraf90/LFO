# Triangulation-Optimierung und Randpunkte-Erweiterung

## Problem-Analyse

### 1. Triangulation dauert sehr lange
- Aktuell: Filterung nach Delaunay-Triangulation verwendet Schleife über alle Zellen
- Problem: Jede Zelle wird einzeln geprüft (langsam bei vielen Zellen)

### 2. Löcher bei großer Resolution
- Aktuell: Randpunkte werden nur AUF den Polygon-Kanten generiert (1cm Auflösung)
- Problem: Bei großer Resolution (z.B. 1m) und schrägen Linien bleiben große Löcher

## Lösungen

### 1. Vektorisierte Filterung
**Ziel**: Alle Dreieck-Schwerpunkte gleichzeitig berechnen und prüfen

**Implementierung**:
- Nutze PyVista's `mesh.cell_centers().points` für alle Schwerpunkte auf einmal
- Verwende vektorisierte Punkt-in-Polygon-Prüfung (`_points_in_polygon_batch_plot`)
- Filtere mit `mesh.extract_cells()` basierend auf Boolean-Array

**Status**: ✅ Implementiert (Zeile 729-807 in Plot3DSPL.py)

### 2. Randpunkte-Streifen
**Ziel**: Statt nur auf der Kante, einen Streifen INNERHALB der Surface generieren

**Konzept**:
- Für jeden Punkt auf der Kante:
  - Berechne Normalvektor zur Kante (nach innen gerichtet)
  - Generiere mehrere Punkte entlang dieses Normalvektors
  - Anzahl: `int(strip_width_factor * resolution / edge_resolution)`
    - Beispiel: 0.25 * 1.0m / 0.01m = 25 Punkte
  - Streifen-Breite: ~25cm bei 1m Resolution

**Implementierung** (TODO):
- Normalvektor-Berechnung: Senkrecht zur Kante, nach innen
- Punkt-Generierung entlang Normalvektor (1cm Schritte)
- Punkt-in-Polygon-Prüfung für jeden generierten Punkt
- Nur Punkte innerhalb der Surface behalten

**Status**: ⏳ In Arbeit

## Nächste Schritte

1. ✅ Filterung vektorisieren (mit PyVista cell_centers)
2. ⏳ Randpunkte-Streifen implementieren
3. ⏳ Performance-Test: Vergleich vor/nach Optimierung


