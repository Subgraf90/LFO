# Doppelklick-Reaktivierung: Sicherheitshinweise

## Aktuelle Situation
- Doppelklick ist komplett deaktiviert (Qt-Ebene + VTK-Ebene)
- Rotation funktioniert ohne Konflikte

## Falls Doppelklick wieder aktiviert wird:

### ✅ SICHERE Implementierung (verhindert Rotation-Konflikte):

1. **Im MouseButtonPress (VOR Rotation-Vorbereitung):**
   ```python
   # Doppelklick-Erkennung MUSS ZUERST passieren
   is_double_click = False
   if self._last_press_time is not None and self._last_press_pos is not None:
       current_time = time.time()
       time_diff = current_time - self._last_press_time
       pos_diff = (abs(press_pos.x() - self._last_press_pos.x()) < 15 and 
                   abs(press_pos.y() - self._last_press_pos.y()) < 15)
       
       if time_diff < 0.5 and pos_diff:
           is_double_click = True
           # 🎯 KRITISCH: Rotation SOFORT verhindern
           self._rotate_active = False
           self._rotate_last_pos = None  # ← WICHTIG: Verhindert Rotation-Start
           self._double_click_handled = True
           self.widget.unsetCursor()
           # Doppelklick-Aktion ausführen
           event.accept()
           return True  # Event abgefangen
   
   # NUR wenn KEIN Doppelklick: Rotation vorbereiten
   if not is_double_click:
       self._rotate_last_pos = press_pos
   ```

2. **Im MouseMove (Rotation-Start verhindern):**
   ```python
   # 🎯 WICHTIG: Rotation NUR wenn KEIN Doppelklick erkannt wurde
   if self._double_click_handled:
       # Doppelklick wurde behandelt - Rotation komplett ignorieren
       self._rotate_active = False
       self._rotate_last_pos = None
       self.widget.unsetCursor()
       event.accept()
       return True
   
   # Normale Rotation-Logik nur wenn kein Doppelklick
   if self._rotate_last_pos is not None and not self._rotate_active:
       # Rotation starten...
   ```

3. **Im MouseButtonRelease (Flag zurücksetzen):**
   ```python
   # Rotation beenden
   self._rotate_active = False
   self._rotate_last_pos = None
   
   # Doppelklick-Flag zurücksetzen (nach kurzer Verzögerung)
   if self._double_click_handled:
       self._double_click_handled = False
   ```

### ⚠️ KRITISCHE PUNKTE:

1. **`_rotate_last_pos` MUSS bei Doppelklick auf `None` gesetzt werden**
   - Sonst startet Rotation trotzdem im MouseMove

2. **Doppelklick-Erkennung MUSS VOR `_rotate_last_pos = press_pos` passieren**
   - Reihenfolge ist kritisch!

3. **`_double_click_handled` Flag MUSS in MouseMove geprüft werden**
   - Verhindert Rotation auch nach Doppelklick-Erkennung

4. **Event MUSS akzeptiert werden (`event.accept()` + `return True`)**
   - Verhindert, dass PyVista's Standard-Handler ausgeführt werden

### 📝 Die auskommentierte Logik (Zeilen 256-295) ist bereits korrekt implementiert!

Die bereits vorhandene auskommentierte Implementierung zeigt die richtige Lösung.
Sie muss nur wieder aktiviert werden, wenn Doppelklick benötigt wird.

