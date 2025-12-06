# Lösung für Randpunkte und Löcher-Problem

## Problem-Analyse

### Aktuell:
- Randpunkte werden AUF den Polygon-Kanten generiert (1cm Auflösung)
- Bei großer Resolution (z.B. 1m) und schrägen Linien bleiben große Löcher durch fehlende Daten

### Lösung:
Statt nur Punkte AUF der Kante zu generieren, sollten wir einen **Streifen INNERHALB der Surface** generieren:

1. **Breite des Streifens**: Proportional zur Resolution (z.B. 0.25 * resolution = 25cm bei 1m Resolution)
2. **Auflösung**: 1cm entlang der Kante, aber mehrere Linien senkrecht zur Kante nach innen
3. **Anzahl der Linien**: z.B. 25 Linien bei 1m Resolution (25cm / 1cm = 25 Punkte)

## Implementierungsplan

### 1. Filterung nach Delaunay-Triangulation
- Dreiecke außerhalb der Surface entfernen
- Prüfe Schwerpunkt jedes Dreiecks mit Punkt-in-Polygon-Test

### 2. Erweiterte Randpunkte-Generierung
- Für jeden Punkt auf der Kante:
  - Berechne Normalvektor zur Kante (nach innen gerichtet)
  - Generiere mehrere Punkte entlang dieser Normalvektor
  - Anzahl: `int(0.25 * resolution / 0.01)` (25 Punkte bei 1m Resolution)
  - Abstand zwischen Punkten: 1cm

### 3. Validierung
- Nur Punkte behalten, die innerhalb der Surface liegen
- Prüfe mit Punkt-in-Polygon-Test vor der Generierung



