# Randpunkte-Erstellung für Surfaces

## Übersicht

Alle Surfaces (einzeln und in Gruppen) werden jetzt **identisch** behandelt. Die Randpunkte-Erstellung erfolgt in zwei Schritten:

## 1. `_ensure_vertex_coverage` (nur für planare/schräge Surfaces)

**Zweck:** Stellt sicher, dass Polygon-Ecken erfasst werden.

**Funktionsweise:**
- Für jede Polygon-Ecke wird der nächstgelegene Grid-Punkt gefunden
- Falls dieser Punkt näher als `2 * Resolution` ist, wird er als "inside" markiert
- Dies verhindert, dass schmale/spitze Polygonbereiche bei grober Auflösung keine Grid-Punkte haben

**Wird verwendet für:**
- ✅ Planare Surfaces (orientation="planar")
- ✅ Schräge Surfaces (orientation="sloped")
- ❌ Vertikale Surfaces (haben eigene Logik in (u,v)-Koordinaten)

**Code-Stelle:** `FlexibleGridGenerator.py:1768-1770`

## 2. `_dilate_mask_minimal` (für alle Surfaces)

**Zweck:** Erweitert die Maske um 1 Zelle (3x3-Dilatation) für bessere Triangulation bis zum Rand.

**Funktionsweise:**
- Verwendet einen 3x3-Kernel für boolesche Dilatation
- Erweitert die Maske um 1 Zelle in alle Richtungen
- Dies ermöglicht Triangulation bis zum Rand der Surface

**Wird verwendet für:**
- ✅ Alle Surfaces (planar, sloped, vertikal)
- ✅ Einzelsurfaces
- ✅ Gruppensurfaces (werden jetzt identisch behandelt)

**Code-Stellen:**
- Planare/schräge: `FlexibleGridGenerator.py:1773`
- Vertikale (Y-Z-Wand): `FlexibleGridGenerator.py:1689`
- Vertikale (X-Z-Wand): `FlexibleGridGenerator.py:1728`

## Unterschiede zwischen Einzel- und Gruppensurfaces

**VORHER (mit Gruppensurfaces-Logik):**
- Einzelsurfaces: `_ensure_vertex_coverage` + `_dilate_mask_minimal`
- Gruppensurfaces: Nur `_dilate_mask_minimal` (in `generate_group_sum_grid`)

**JETZT (alle identisch):**
- ✅ Alle Surfaces: `_ensure_vertex_coverage` (nur planar/sloped) + `_dilate_mask_minimal`
- ✅ Alle Surfaces werden über `generate_per_surface` erstellt
- ✅ Alle Surfaces werden über `surface_results_buffers` berechnet
- ✅ Alle Surfaces werden einzeln geplottet

## Randpunkte-Position

Die Randpunkte liegen **primär außerhalb** des Polygon-Rands:
- Die 3x3-Dilatation erweitert die Maske um 1 Zelle
- Dies bedeutet, dass Randpunkte außerhalb der Polygon-Begrenzungslinie liegen
- In konkaven Bereichen können einige Randpunkte auch innerhalb liegen

## Redundante Methoden (nicht mehr verwendet)

1. **`_identify_group_candidates`** (SoundfieldCalculator.py:1563)
   - Wurde verwendet um Gruppen-Kandidaten zu identifizieren
   - Wird nicht mehr aufgerufen

2. **`_combine_group_meshes`** (Plot3DSPL.py:74)
   - Wurde verwendet um Meshes mehrerer Surfaces zu kombinieren
   - Wird nicht mehr aufgerufen

3. **`generate_group_sum_grid`** (FlexibleGridGenerator.py:3340)
   - Wurde verwendet um gemeinsame Grids für Gruppen zu erstellen
   - Wird nicht mehr aufgerufen

