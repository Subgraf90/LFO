# Analyse: Unbenötigte Methoden in Plot_SPL_3D

Diese Datei listet Methoden auf, die möglicherweise nicht verwendet werden und entfernt werden können.

## ⚠️ HINWEIS
Diese Analyse basiert auf statischer Code-Analyse. Bitte prüfe vor dem Löschen, ob die Methoden:
- Eventuell über Reflection/Reflexion aufgerufen werden
- Für zukünftige Features vorgesehen sind
- In Kommentaren als "für später" markiert sind

---

## 1. Plot3DSPL.py

### 1.1 `_update_surface_scalars` (Zeile 1060-1077)
**Status:** ❌ **NICHT VERWENDET**

```python
def _update_surface_scalars(self, flat_scalars: np.ndarray) -> bool:
    """Aktualisiert die Skalare des Surface-Meshes."""
```

**Gefunden:** Nur Definition, keine Aufrufe im gesamten Codebase.

**Empfehlung:** ✅ **KANN ENTFERNT WERDEN**

---

### 1.2 `get_texture_metadata` (Zeile 1127-1162)
**Status:** ⚠️ **NUR INTERN VERWENDET**

```python
def get_texture_metadata(self, surface_id: str) -> Optional[dict[str, Any]]:
    """Gibt die Metadaten einer Texture-Surface zurück."""
```

**Gefunden:** 
- Wird nur von `get_texture_world_coords` (Zeile 1176) und `get_world_coords_to_texture_coords` (Zeile 1210) aufgerufen
- Keine externen Aufrufe außerhalb der Klasse

**Empfehlung:** ⚠️ **PRÜFEN** - Falls die beiden anderen Methoden auch nicht verwendet werden, können alle drei entfernt werden.

---

### 1.3 `get_texture_world_coords` (Zeile 1164-1196)
**Status:** ⚠️ **NUR INTERN VERWENDET**

```python
def get_texture_world_coords(self, surface_id: str, texture_u: float, texture_v: float) -> Optional[tuple[float, float, float]]:
    """Konvertiert Textur-Koordinaten (u, v) zu Weltkoordinaten (x, y, z)."""
```

**Gefunden:** 
- Ruft nur `get_texture_metadata` auf
- Keine externen Aufrufe

**Empfehlung:** ⚠️ **PRÜFEN** - Falls nicht extern verwendet, kann entfernt werden.

---

### 1.4 `get_world_coords_to_texture_coords` (Zeile 1198-1239)
**Status:** ⚠️ **NUR INTERN VERWENDET**

```python
def get_world_coords_to_texture_coords(self, surface_id: str, world_x: float, world_y: float) -> Optional[tuple[float, float]]:
    """Konvertiert Weltkoordinaten (x, y) zu Textur-Koordinaten (u, v)."""
```

**Gefunden:** 
- Ruft nur `get_texture_metadata` auf
- Keine externen Aufrufe

**Empfehlung:** ⚠️ **PRÜFEN** - Falls nicht extern verwendet, kann entfernt werden.

---

## 2. Plot3DHelpers.py

### 2.1 `_remove_actor` (Zeile 96-101)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird von `PlotSPL3DSpeaker.py` verwendet (Zeile 1080, 2160)
- Wird von `Plot3DOverlaysBase.py` überschrieben (Zeile 250)

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

### 2.2 `_compute_overlay_signatures` (Zeile 103-387)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird von `Plot3D.py` verwendet (Zeile 587)

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 3. PolarPlotCalculator.py

### 3.1 `normalize_values` (Zeile 367)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird intern verwendet (Zeile 112)

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

### 3.2 `calculate_dborder` (Zeile 130-151)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird intern verwendet (Zeile 196)

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

### 3.3 `find_nearest_frequency` (Zeile 153-160)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird intern verwendet (Zeile 179)

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 4. Plot3D.py

### 4.1 `_handle_double_click_auto_range` (Zeile 258)
**Status:** ✅ **VERWENDET**

**Gefunden:** 
- Wird als Callback für Colorbar-Doppelklick gesetzt (Zeile 192): `self.colorbar_manager.on_double_click = self._handle_double_click_auto_range`

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 5. Plot3DInteraction.py

### 5.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 6. Plot3DOverlaysBase.py

### 6.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 7. Plot3DOverlaysAxis.py

### 7.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 8. Plot3DOverlaysCoordinateAxes.py

### 8.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 9. Plot3DOverlaysSurfaces.py

### 9.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 10. Plot3DOverlaysImpulse.py

### 10.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 11. PlotSPL3DSpeaker.py

### 11.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 12. Plot3DViewControls.py

### 12.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## 13. Plot3DCamera.py

### 13.1 Alle Methoden scheinen verwendet zu werden
**Status:** ✅ **VERWENDET**

**Empfehlung:** ❌ **NICHT ENTFERNEN**

---

## ZUSAMMENFASSUNG

### ✅ Sicher zu entfernen:
1. **`Plot3DSPL._update_surface_scalars`** (Zeile 1060-1077) 
   - **Status:** Keine Verwendung gefunden
   - **Datei:** `Plot3DSPL.py`
   - **Empfehlung:** ✅ **KANN ENTFERNT WERDEN**

### ⚠️ Zu prüfen (möglicherweise ungenutzt):
2. **`Plot3DSPL.get_texture_metadata`** (Zeile 1127-1162)
   - **Status:** Nur intern verwendet (von den beiden anderen Texture-Methoden)
   - **Datei:** `Plot3DSPL.py`
   - **Empfehlung:** ⚠️ **PRÜFEN** - Falls die anderen beiden auch nicht verwendet werden, kann entfernt werden

3. **`Plot3DSPL.get_texture_world_coords`** (Zeile 1164-1196)
   - **Status:** Nur intern verwendet (ruft nur `get_texture_metadata` auf)
   - **Datei:** `Plot3DSPL.py`
   - **Empfehlung:** ⚠️ **PRÜFEN** - Falls nicht extern verwendet, kann entfernt werden

4. **`Plot3DSPL.get_world_coords_to_texture_coords`** (Zeile 1198-1239)
   - **Status:** Nur intern verwendet (ruft nur `get_texture_metadata` auf)
   - **Datei:** `Plot3DSPL.py`
   - **Empfehlung:** ⚠️ **PRÜFEN** - Falls nicht extern verwendet, kann entfernt werden

**Hinweis:** Die drei Texture-Methoden (`get_texture_metadata`, `get_texture_world_coords`, `get_world_coords_to_texture_coords`) bilden eine Gruppe. Wenn keine extern verwendet wird, können alle drei entfernt werden.

---

## NÄCHSTE SCHRITTE

1. ✅ **`_update_surface_scalars`** kann sicher entfernt werden
2. ⚠️ Prüfe, ob die Texture-Methoden eventuell für zukünftige Features (z.B. Koordinaten-Konvertierung, Picking auf Surfaces) vorgesehen sind
3. ⚠️ Entscheide über die Texture-Methoden basierend auf zukünftigen Anforderungen
4. Falls die Texture-Methoden nicht benötigt werden, können alle drei zusammen entfernt werden

