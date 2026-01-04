# DXF Polygon- und Gruppenerkennung

## Wie wird ein Polygon erkannt?

### 1. Entity-Typ-Prüfung

Zuerst wird geprüft, ob die Entity überhaupt Punkte enthalten kann:

```python
for entity, transform, block_stack in self._iter_surface_entities(msp):
    dxftype = entity.dxftype()
    points = self._extract_points_from_entity(entity, transform)
```

**Unterstützte Entity-Typen:**
- ✅ `POLYLINE` - 2D/3D Polylinien
- ✅ `LWPOLYLINE` - Leichtgewichtige Polylinien (2D)
- ✅ `3DFACE` - 3D-Flächen (3-4 Eckpunkte)
- ✅ `MESH` - Polygon-Mesh
- ✅ `SOLID` - Gefüllte Fläche
- ❌ `LINE` - Wird übersprungen (nur 2 Punkte, keine Fläche)

### 2. Mindestpunktzahl-Prüfung

```python
# Mindestens 3 Punkte für eine Fläche erforderlich
if len(points) < 3:
    logger.info("Entity übersprungen (nur %d Punkte, mindestens 3 erforderlich)", len(points))
    skipped_count += 1
    continue
```

**Regel:** Eine Fläche benötigt mindestens **3 Punkte**.

### 3. Geschlossenheits-Prüfung

Die Funktion `_is_entity_closed()` prüft, ob eine Entity geschlossen ist:

```python
def _is_entity_closed(self, entity, points: List[Dict[str, float]]) -> bool:
    """Prüft ob eine Entity geschlossen ist"""
    dxftype = entity.dxftype()
    
    # Für LWPOLYLINE: Prüfe verschiedene Attribute
    if dxftype == "LWPOLYLINE":
        if hasattr(entity, "closed"):
            return bool(entity.closed)
        if hasattr(entity.dxf, "flags"):
            # Bit 1 = closed flag
            flags = getattr(entity.dxf, "flags", 0)
            return bool(flags & 1)
    
    # Für POLYLINE: Prüfe is_closed Attribut
    if dxftype == "POLYLINE":
        if hasattr(entity, "is_closed"):
            return bool(entity.is_closed)
        if hasattr(entity, "closed"):
            return bool(entity.closed)
    
    # Fallback: Prüfe ob erster und letzter Punkt identisch sind (mit Toleranz)
    if len(points) >= 3:
        first = points[0]
        last = points[-1]
        tolerance = 1e-6
        dx = abs(first.get("x", 0) - last.get("x", 0))
        dy = abs(first.get("y", 0) - last.get("y", 0))
        dz = abs(first.get("z", 0) - last.get("z", 0))
        return dx < tolerance and dy < tolerance and dz < tolerance
    
    return False
```

**Erkennungsmethoden:**

#### A. LWPOLYLINE (Leichtgewichtige Polylinien)
1. **Direktes Attribut**: `entity.closed == True`
2. **Flag-Bit**: Bit 1 in `entity.dxf.flags` gesetzt

#### B. POLYLINE (2D/3D Polylinien)
1. **Direktes Attribut**: `entity.is_closed == True`
2. **Alternatives Attribut**: `entity.closed == True`

#### C. Fallback: Koordinaten-Vergleich
- Prüft ob erster und letzter Punkt identisch sind (Toleranz: 1e-6)
- Funktioniert für alle Entity-Typen

### 4. Nahezu-geschlossen-Erkennung

Zusätzlich wird geprüft, ob eine offene Polylinie nahezu geschlossen ist:

```python
# Prüfe auch, ob die Fläche nahezu geschlossen ist (erster und letzter Punkt sehr nah)
if not is_closed and len(points) >= 3:
    first = points[0]
    last = points[-1]
    tolerance = 0.01  # 1cm Toleranz für nahezu geschlossene Flächen
    dx = abs(first.get("x", 0) - last.get("x", 0))
    dy = abs(first.get("y", 0) - last.get("y", 0))
    dz = abs(first.get("z", 0) - last.get("z", 0))
    distance = (dx**2 + dy**2 + dz**2)**0.5
    if distance < tolerance:
        is_closed = True
        logger.info("Entity als nahezu geschlossen erkannt (Abstand: %.4f)", distance)
```

**Toleranz:** 1cm (0.01 Einheiten) für nahezu geschlossene Flächen

### 5. Polygon-Schließung

Wenn eine Entity als geschlossen erkannt wird, aber erster und letzter Punkt nicht identisch sind:

```python
# Wenn geschlossen, aber erster und letzter Punkt nicht identisch, schließe die Fläche
if is_closed and len(points) >= 3:
    first = points[0]
    last = points[-1]
    tolerance = 1e-6
    dx = abs(first.get("x", 0) - last.get("x", 0))
    dy = abs(first.get("y", 0) - last.get("y", 0))
    dz = abs(first.get("z", 0) - last.get("z", 0))
    if dx >= tolerance or dy >= tolerance or dz >= tolerance:
        # Füge ersten Punkt am Ende hinzu, um zu schließen
        points.append(first.copy())
```

**Regel:** Wenn geschlossen erkannt, aber nicht identisch → füge ersten Punkt am Ende hinzu

---

## Wie werden Gruppen erkannt?

Die Gruppenerkennung erfolgt über mehrere Mechanismen:

### 1. Block-Hierarchie (Hauptmethode)

**Wie funktioniert es:**

```python
# Gruppierung über Block-Hierarchie
group_label = self._build_group_label(block_stack)
target_group_id = self._ensure_group_for_label(group_label)
```

**Block-Stack-Struktur:**

Bei verschachtelten INSERT-Entities wird ein Block-Stack aufgebaut:

```
Modelspace
  └── INSERT (BlockA)
      └── INSERT (BlockB)
          └── POLYLINE (Fläche)
```

Wird zu:
```
block_stack = (
    InsertContext(name="BlockA", ...),
    InsertContext(name="BlockB", ...)
)
```

**Gruppen-Label-Erstellung:**

```python
def _build_group_label(self, block_stack: tuple) -> Optional[str]:
    if not block_stack:
        return None
    segments: List[str] = []
    for ctx in block_stack:
        segment = self._format_block_segment(ctx)
        if segment:
            segments.append(segment)
    if not segments:
        return None
    return "/".join(segments)  # "BlockA/BlockB"
```

**Gruppen-Erstellung:**

```python
def _ensure_group_for_label(self, label: Optional[str]) -> Optional[str]:
    # ...
    if "/" in key:
        # Für verschachtelte Pfade: Präfixiere jeden Segment
        segments = [segment.strip() for segment in key.split("/") if segment.strip()]
        prefixed_segments = [f"Group {seg}" if not seg.startswith("Group ") else seg 
                            for seg in segments]
        prefixed_path = "/".join(prefixed_segments)
        group_id = self.group_manager.ensure_group_path(prefixed_path)
    # ...
```

**Beispiel:**
- Block-Hierarchie: `BlockA/BlockB`
- Wird zu: `Group BlockA/Group BlockB`
- Erstellt verschachtelte Gruppen

### 2. DXF-Gruppen (GROUP-Entities)

**Wie funktioniert es:**

```python
def _build_dxf_group_lookup(doc) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    groups = getattr(doc, "groups", None)
    if not groups:
        return lookup

    for group in groups:
        name = getattr(group, "name", None)
        if not name:
            continue
        try:
            handles = list(group.get_entity_handles())
        except Exception:
            continue
        for handle in handles:
            lookup[handle] = name
    return lookup
```

**Verwendung:**

```python
# Fallback auf ursprüngliche Gruppen-Logik
group_name = group_lookup.get(entity.dxf.handle)
target_group_id = self._ensure_group_for_label(group_name)
```

**Funktionsweise:**
- DXF-Gruppen enthalten eine Liste von Entity-Handles
- Jeder Entity-Handle wird einem Gruppen-Namen zugeordnet
- Diese Zuordnung wird als Fallback verwendet, wenn Block-Hierarchie keine Gruppe liefert

**Beispiel:**
```python
# DXF-Gruppe "Wall1" enthält Entities mit Handles: [0x123, 0x456, 0x789]
# Entity mit Handle 0x123 → Gruppe "Wall1"
group_lookup = {"0x123": "Wall1", "0x456": "Wall1", "0x789": "Wall1"}
```

### 3. Tag-basierte Gruppierung

**Wie funktioniert es:**

```python
# Extrahiere Tag/Attribut aus Entity (z.B. "WOOD", "BETON")
tag = self._extract_tag_from_entity(entity, doc)
if not tag:
    tag = self._resolve_tag_from_context(block_stack)

# Verwende Tag als Gruppierung, falls vorhanden
if tag:
    logger.info("Tag '%s' gefunden", tag)
    # Tag wird als Gruppen-Name verwendet
```

**Tag-Extraktionsquellen (in Prioritäts-Reihenfolge):**

1. **Block-Name** - Wenn Block-Name bekanntes Material ist (WOOD, BETON, etc.)
2. **ATTDEF-Entities** - Attribute-Definitionen in Block-Definition
3. **ATTRIB-Entities** - Attribute-Werte in INSERT-Instanz
4. **Layer-Name** - Wenn Layer-Name bekanntes Material ist
5. **XDATA** - Extended Data (strukturierte Suche)
6. **AppData** - Application Data
7. **Objekt-Daten** - Object Data

**Tag-Vererbung:**

```python
def _resolve_tag_from_context(block_stack: tuple) -> Optional[str]:
    """Löst Tag aus Block-Stack auf (rückwärts durchgehen)"""
    for ctx in reversed(block_stack):
        if ctx.tag:
            return ctx.tag
    return None
```

**Beispiel:**
```
INSERT (BlockA, tag="WOOD")
  └── INSERT (BlockB)
      └── POLYLINE (Fläche) → erbt Tag "WOOD"
```

### 4. Layer-basierte Gruppierung

**Wie funktioniert es:**

Layer-Namen werden als Basis-Name verwendet:

```python
base_name = getattr(entity.dxf, "layer", None) or dxftype or "DXF_Surface"
```

Wenn Layer-Name bekanntes Material ist, wird er als Tag verwendet:

```python
layer_name = getattr(entity.dxf, "layer", None)
if layer_name:
    layer_name_upper = layer_name.upper()
    if layer_name_upper in ("WOOD", "BETON", "CONCRETE", "STEEL", "GLASS"):
        return layer_name_upper  # Wird als Tag/Gruppe verwendet
```

**Beispiel:**
- Entity auf Layer "WOOD" → Tag "WOOD" → Gruppe "Group WOOD"
- Entity auf Layer "BETON_1" → Tag "BETON" → Gruppe "Group BETON"

---

## Zusammenfassung: Polygon-Erkennung

**Schritte:**
1. ✅ Entity-Typ prüfen (POLYLINE, LWPOLYLINE, 3DFACE, MESH, SOLID)
2. ✅ Mindestens 3 Punkte vorhanden?
3. ✅ Entity als geschlossen markiert? (`is_closed`, `closed`, Flag-Bit)
4. ✅ Erster und letzter Punkt identisch? (Toleranz: 1e-6)
5. ✅ Nahezu geschlossen? (Toleranz: 1cm = 0.01)
6. ✅ Wenn geschlossen → füge ersten Punkt am Ende hinzu (falls nötig)

**Ergebnis:** Geschlossenes Polygon mit mindestens 3 Punkten

---

## Zusammenfassung: Gruppenerkennung

**Mechanismen (in Prioritäts-Reihenfolge):**

1. **Block-Hierarchie** (Hauptmethode)
   - Verschachtelte INSERT-Entities → Pfad: "BlockA/BlockB"
   - Wird zu: "Group BlockA/Group BlockB"
   - Erstellt verschachtelte Gruppen

2. **DXF-Gruppen** (Fallback)
   - GROUP-Entities mit Entity-Handles
   - Handle → Gruppen-Name Mapping

3. **Tag-basierte Gruppierung**
   - Material-Tags (WOOD, BETON, etc.)
   - Aus: Block-Name, ATTDEF, ATTRIB, Layer, XDATA
   - Vererbt durch Block-Hierarchie

4. **Layer-basierte Gruppierung**
   - Layer-Name als Basis-Name
   - Wenn bekanntes Material → wird als Tag/Gruppe verwendet

**Ergebnis:** Jede Fläche wird einer oder mehreren Gruppen zugeordnet

