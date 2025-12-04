# Performance-Analyse: Source UI

## Zusammenfassung
Untersuchung der `show_sources_tab()` Methode und aller zugehörigen Methoden zur Identifikation von Performance-Optimierungsmöglichkeiten.

---

## 1. Hauptproblem: Tabs werden jedes Mal neu erstellt

### Aktuelles Verhalten
**Zeile 2414-2416:** `show_sources_tab()` entfernt **ALLE** Tabs bei jedem Aufruf:
```python
while self.tab_widget.count() > 0:
    self.tab_widget.removeTab(0)
```

### Problem
- **Beamsteering- und Windowing-Plots werden jedes Mal neu erstellt** (Zeilen 1127, 1028)
- Plot-Erstellung ist teuer: Matplotlib Figure, Canvas, Subplot-Initialisierung
- Bei jedem Tab-Wechsel werden Plots zerstört und neu erstellt

### Optimierungspotenzial
**Erwartete Verbesserung: 50-100ms pro Aufruf**

**Lösung:**
- Tabs nur erstellen, wenn sie noch nicht existieren
- Prüfen ob Tab mit Namen bereits existiert (`tabText()`)
- Plots wiederverwenden statt neu erstellen

---

## 2. Redundante `get_speaker_array()` Aufrufe

### Aktuelles Verhalten
**100+ Aufrufe** von `get_speaker_array()` in der gesamten Datei

**In `show_sources_tab()`:**
- Zeile 2448: `speaker_array = self.settings.get_speaker_array(speaker_array_id)`
- Zeile 2503-2508: 6 weitere Methoden rufen `get_speaker_array()` erneut auf:
  - `update_sources_input_fields()` → Zeile 2928
  - `update_source_length_input_fields()` → Zeile 2938
  - `update_array_position_input_fields()` → Zeile 2948
  - `update_beamsteering_input_fields()` → Zeile 2870
  - `update_windowing_input_fields()` → Zeile 3005
  - `update_gain_delay_input_fields()` → Zeile 3023

### Problem
- Jeder Aufruf durchsucht wahrscheinlich eine Liste/Dictionary
- 7x redundante Aufrufe für dasselbe Array

### Optimierungspotenzial
**Erwartete Verbesserung: 5-15ms pro Aufruf**

**Lösung:**
- `speaker_array` einmal in `show_sources_tab()` holen
- Als optionaler Parameter an alle `update_*_input_fields()` Methoden übergeben
- Rückwärtskompatibel: Falls `None`, wird es intern geholt

---

## 3. `update_input_fields()` - Komplette Widget-Neuerstellung

### Aktuelles Verhalten
**Zeilen 1174-1180:** Alle Widgets werden gelöscht und neu erstellt:
```python
while gridLayout_sources.count():
    item = gridLayout_sources.takeAt(0)
    widget = item.widget()
    if widget is not None:
        widget.setParent(None)
        widget.deleteLater()
```

### Problem
- Bei 20 Quellen: **140+ Widgets** (7 Felder × 20 Quellen)
- Jedes Widget: Erstellung, Signal-Verbindungen, Layout-Updates
- Sehr teuer, besonders bei vielen Quellen

### Optimierungspotenzial
**Erwartete Verbesserung: 50-150ms (abhängig von Anzahl Quellen)**

**Lösung:**
- Widgets nur neu erstellen, wenn sich Struktur ändert (Anzahl Quellen, Array-Typ)
- Cache für Struktur-Informationen (`_cached_num_sources`, `_cached_is_stack`)
- Bei unveränderter Struktur: Nur Werte aktualisieren, nicht Widgets neu erstellen

**⚠️ WICHTIG:** Diese Optimierung wurde bereits versucht, hat aber die UI "zerschossen". 
Mögliche Ursachen:
- Labels wurden nicht korrekt erstellt
- Widget-Positionen stimmten nicht
- Cache-Logik hatte Fehler

**Empfehlung:** Vorsichtige, schrittweise Implementierung mit umfangreichen Tests.

---

## 4. `update_speaker_comboboxes()` - Redundante Filterung

### Aktuelles Verhalten
**Zeilen 1716-1826:** Bei jedem Aufruf:
- Durchläuft alle `speaker_names` im Container
- Filtert Stack vs. Flown Lautsprecher
- Erstellt Dictionary für Lookups
- Erstellt/aktualisiert Comboboxen

### Problem
- Filterung wird bei jedem Aufruf neu durchgeführt
- Cabinet-Daten werden jedes Mal analysiert
- Bei vielen Lautsprechern (20+) ist das teuer

### Optimierungspotenzial
**Erwartete Verbesserung: 10-30ms pro Aufruf**

**Lösung:**
- Gefilterte Listen cachen (nur neu filtern, wenn Container-Daten sich ändern)
- Dictionary-Lookups nur einmal erstellen
- Comboboxen nur aktualisieren, wenn sich Werte geändert haben

---

## 5. `display_selected_speakerspecs()` - Ineffiziente Iteration

### Aktuelles Verhalten
**Zeilen 2089-2110:** Durchläuft alle `speakerspecs_instance`:
```python
for instance in self.speakerspecs_instance:
    if 'id' in instance:
        self.hide_speakerspecs(instance)
```

### Problem
- Versteckt ALLE Instanzen, auch die, die bereits versteckt sind
- Zeigt ausgewählte Instanz, auch wenn sie bereits sichtbar ist
- Unnötige UI-Updates

### Optimierungspotenzial
**Erwartete Verbesserung: 5-10ms pro Aufruf**

**Lösung:**
- Nur sichtbare Instanzen verstecken
- Nur versteckte Instanzen anzeigen
- Prüfen mit `scroll_area.isVisible()` vor Update

---

## 6. Plot-Erstellung im `__init__`

### Aktuelles Verhalten
**Zeilen 51-55:** Plots werden bereits im `__init__` erstellt:
```python
self.tab_widget = QTabWidget()
self.beamsteering_plot = None
self.create_beamsteering_tab()
self.windowing_plot = None
self.create_windowing_tab()
```

### Problem
- Plots werden erstellt, bevor sie benötigt werden
- Werden dann in `show_sources_tab()` wieder entfernt und neu erstellt
- Doppelte Erstellung = doppelte Kosten

### Optimierungspotenzial
**Erwartete Verbesserung: 20-50ms beim Start**

**Lösung:**
- Plots erst bei Bedarf erstellen (Lazy Loading)
- Oder: Plots im `__init__` erstellen, aber in `show_sources_tab()` wiederverwenden

---

## 7. Redundante Font-Erstellung

### Aktuelles Verhalten
In mehreren Methoden wird `QFont()` erstellt:
- `create_speaker_tab_stack()`: Zeile 532
- `create_beamsteering_tab()`: Zeile 1058
- `create_windowing_tab()`: Zeile 955
- `update_input_fields()`: Zeile 1196
- `update_speaker_qlineedit()`: Zeile 1877

### Problem
- Font-Objekt wird mehrfach erstellt mit identischen Einstellungen
- Kleine, aber unnötige Overhead

### Optimierungspotenzial
**Erwartete Verbesserung: 1-3ms pro Aufruf**

**Lösung:**
- Font als Klassen-Variable cachen
- Einmalig in `__init__` erstellen
- Wiederverwenden in allen Methoden

---

## 8. `update_widgets()` - Mehrfache Array-Zugriffe

### Aktuelles Verhalten
**Zeilen 1483-1516:** Ruft mehrere Methoden auf:
- `update_speaker_comboboxes()` → holt `speaker_array` erneut
- `update_speaker_qlineedit()` → holt `speaker_array` erneut
- `update_symmetry_checkbox()` → holt `speaker_array` erneut

### Problem
- Jede Methode holt `speaker_array` selbst
- Redundante Zugriffe

### Optimierungspotenzial
**Erwartete Verbesserung: 3-8ms pro Aufruf**

**Lösung:**
- `speaker_array` als Parameter übergeben
- Konsistent mit Optimierung #2

---

## Zusammenfassung der Optimierungspotenziale

| Optimierung | Erwartete Verbesserung | Risiko | Priorität |
|------------|------------------------|--------|-----------|
| 1. Tabs wiederverwenden | 50-100ms | Niedrig | ⭐⭐⭐ Hoch |
| 2. Redundante get_speaker_array() | 5-15ms | Niedrig | ⭐⭐⭐ Hoch |
| 3. Widget-Wiederverwendung | 50-150ms | **Hoch** ⚠️ | ⭐⭐ Mittel |
| 4. Combobox-Filterung cachen | 10-30ms | Niedrig | ⭐⭐ Mittel |
| 5. display_selected_speakerspecs | 5-10ms | Niedrig | ⭐ Niedrig |
| 6. Plot-Lazy-Loading | 20-50ms | Niedrig | ⭐⭐ Mittel |
| 7. Font-Caching | 1-3ms | Niedrig | ⭐ Niedrig |
| 8. update_widgets Parameter | 3-8ms | Niedrig | ⭐⭐ Mittel |

**Gesamtpotenzial: 144-366ms Verbesserung**

---

## Empfohlene Implementierungsreihenfolge

1. **Optimierung #1 (Tabs wiederverwenden)** - Sicher, hoher Impact
2. **Optimierung #2 (Redundante get_speaker_array())** - Sicher, einfache Änderung
3. **Optimierung #4 (Combobox-Filterung)** - Sicher, mittlerer Impact
4. **Optimierung #5 (display_selected_speakerspecs)** - Sicher, einfache Änderung
5. **Optimierung #6 (Plot-Lazy-Loading)** - Sicher, mittlerer Impact
6. **Optimierung #7 (Font-Caching)** - Sicher, kleiner Impact
7. **Optimierung #8 (update_widgets Parameter)** - Sicher, kleiner Impact
8. **Optimierung #3 (Widget-Wiederverwendung)** - ⚠️ **Vorsichtig**, bereits fehlgeschlagen

---

## Wichtige Hinweise

- **Optimierung #3** wurde bereits versucht und hat die UI beschädigt
- Alle Optimierungen sollten einzeln implementiert und getestet werden
- Performance-Messungen vor/nach jeder Optimierung durchführen
- UI-Funktionalität nach jeder Änderung gründlich testen

