# Optimierungsvorschläge: Source UI

## Zusammenfassung
Erweiterte Analyse des Source UI mit zusätzlichen Optimierungsvorschlägen, die über die bereits dokumentierten hinausgehen.

---

## 🔴 KRITISCH: 9 Signal-Verbindungen auf itemSelectionChanged

### Aktuelles Problem
**Zeilen 411-420:** Bei JEDER Auswahländerung werden **9 Methoden sequenziell aufgerufen**:
```python
self.sources_tree_widget.itemSelectionChanged.connect(self.show_sources_tab)
self.sources_tree_widget.itemSelectionChanged.connect(self.display_selected_speakerspecs)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_sources_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_source_length_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_array_position_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_beamsteering_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_windowing_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_gain_delay_input_fields)
self.sources_tree_widget.itemSelectionChanged.connect(self.update_beamsteering_windowing_plots)
self.sources_tree_widget.itemSelectionChanged.connect(self._handle_sources_tree_selection_changed)
```

### Problem
- **9x redundante Ausführung** bei jeder Auswahländerung
- `show_sources_tab()` ruft bereits viele dieser Methoden intern auf (Zeilen 2657-2670)
- `display_selected_speakerspecs()` wird sowohl direkt als auch über Signal aufgerufen
- **Signal-Loops** möglich, wenn Methoden intern die Selektion ändern

### Optimierungspotenzial
**Erwartete Verbesserung: 50-200ms pro Auswahländerung**

### Lösung
**Option 1: Einzelner Handler (Empfohlen)**
```python
# Nur EIN Signal verbinden
self.sources_tree_widget.itemSelectionChanged.connect(self._on_selection_changed)

def _on_selection_changed(self):
    """Zentraler Handler für alle Auswahländerungen"""
    # Blockiere Signale während Updates
    self.sources_tree_widget.blockSignals(True)
    try:
        # Führe alle Updates in optimaler Reihenfolge aus
        self.show_sources_tab()  # Ruft bereits viele Updates intern auf
        # Nur zusätzliche Updates, die nicht in show_sources_tab() sind
        self._handle_sources_tree_selection_changed()
    finally:
        self.sources_tree_widget.blockSignals(False)
```

**Option 2: Signal-Batching mit QTimer**
- Sammle mehrere Signal-Events
- Führe Updates nur einmal nach kurzer Verzögerung aus
- Verhindert mehrfache Updates bei schnellen Selektionen

---

## 🟡 HOCH: Redundante Methodenaufrufe in show_sources_tab()

### Aktuelles Problem
**Zeilen 2657-2670:** `show_sources_tab()` ruft Methoden auf, die bereits über Signale verbunden sind:
```python
self.display_selected_speakerspecs()  # Wird auch über Signal aufgerufen!
self.update_sources_input_fields()     # Wird auch über Signal aufgerufen!
self.update_source_length_input_fields()  # Wird auch über Signal aufgerufen!
# ... etc.
```

### Problem
- **Doppelte Ausführung** derselben Methoden
- Unnötige Performance-Kosten
- Inkonsistente Ausführungsreihenfolge

### Optimierungspotenzial
**Erwartete Verbesserung: 30-80ms pro Aufruf**

### Lösung
- Entweder: Signale entfernen und nur in `show_sources_tab()` aufrufen
- Oder: Direkte Aufrufe entfernen und nur Signale verwenden
- **Empfehlung:** Signale entfernen, da `show_sources_tab()` die zentrale Methode ist

---

## 🟡 HOCH: Code-Duplikation in show_sources_tab()

### Aktuelles Problem
**Zeilen 2577-2654:** Massive Code-Duplikation für Stack/Flown/Default:
- Gleiche Logik für Tab-Erstellung (3x)
- Gleiche Plot-Extraktion (6x)
- Gleiche Tab-Aktivierung (3x)

### Problem
- **~80 Zeilen duplizierter Code**
- Schwer zu warten
- Fehleranfällig (Änderungen müssen 3x gemacht werden)

### Optimierungspotenzial
**Erwartete Verbesserung: Code-Qualität + 5-10ms (weniger Code-Pfad)**

### Lösung
```python
def show_sources_tab(self):
    # ... Vorbereitung ...
    
    # Bestimme Konfiguration
    configuration = self._get_array_configuration(speaker_array)
    
    # Einheitliche Tab-Erstellung
    required_tabs = self._get_required_tabs(configuration)
    self._ensure_tabs_exist(required_tabs)
    
    # Einheitliche Tab-Aktivierung
    self._activate_tabs_for_configuration(configuration)
```

---

## 🟡 MITTEL: Timer-basierte resize() - Race Conditions

### Aktuelles Problem
**Zeilen 496-503:** `resize()` wird mit 100ms Timer aufgerufen:
```python
QTimer.singleShot(100, apply_resize)
```

### Problem
- **Race Condition:** Wenn Benutzer schnell interagiert, kann Timer zu spät kommen
- **Magic Number:** 100ms ist willkürlich gewählt
- **Unzuverlässig:** Abhängig von System-Performance

### Optimierungspotenzial
**Erwartete Verbesserung: Zuverlässigkeit + 0-50ms (wenn früher ausgeführt)**

### Lösung
**Option 1: resizeEvent überschreiben**
```python
class SourcesDockWidget(QDockWidget):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.height() > 140:
            # Korrigiere nur wenn zu groß
            QTimer.singleShot(0, lambda: self.resize(1200, 140))
```

**Option 2: sizeHint() überschreiben**
- Qt respektiert sizeHint() beim ersten Anzeigen
- Kein Timer nötig

---

## 🟡 MITTEL: Fehlende Validierung und Error-Handling

### Aktuelles Problem
**Viele Stellen:** `hasattr()` Checks, aber keine Validierung:
```python
if hasattr(self, 'sources_tree_widget') and self.sources_tree_widget:
    # Aber: Was wenn sources_tree_widget None ist?
    # Was wenn es ein gelöschtes Widget ist?
```

### Problem
- **RuntimeError** möglich bei gelöschten Widgets
- Unklare Fehlermeldungen
- Schwer zu debuggen

### Optimierungspotenzial
**Erwartete Verbesserung: Stabilität + Debugging**

### Lösung
```python
def _is_widget_valid(self, widget):
    """Prüft ob Widget existiert und nicht gelöscht wurde"""
    if widget is None:
        return False
    try:
        # Versuche auf C++ Objekt zuzugreifen
        _ = widget.objectName()
        return True
    except RuntimeError:
        # Widget wurde gelöscht
        return False
```

---

## 🟢 NIEDRIG: Redundante Plot-Extraktion

### Aktuelles Problem
**Zeilen 2584-2596, 2607-2620, 2637-2650:** Plot-Extraktion wird 3x mit identischem Code durchgeführt

### Problem
- Code-Duplikation
- Potenzielle Fehlerquelle

### Optimierungspotenzial
**Erwartete Verbesserung: Code-Qualität**

### Lösung
```python
def _ensure_plot_extracted(self, tab_name, plot_class, plot_attr):
    """Extrahiert Plot aus Tab, falls noch nicht vorhanden"""
    if getattr(self, plot_attr) is not None:
        return  # Bereits extrahiert
    
    tab = self._get_tab_widget_by_name(tab_name)
    if tab:
        plot = self._extract_plot_from_tab(tab, plot_class)
        setattr(self, plot_attr, plot)
```

---

## 🟢 NIEDRIG: Ineffiziente Tab-Suche

### Aktuelles Problem
**Zeile 2442-2450:** `_get_tab_widget_by_name()` durchläuft alle Tabs:
```python
for i in range(self.tab_widget.count()):
    if self.tab_widget.tabText(i) == tab_name:
        return self.tab_widget.widget(i)
```

### Problem
- **O(n)** Suche bei jedem Aufruf
- Wird mehrfach pro `show_sources_tab()` aufgerufen

### Optimierungspotenzial
**Erwartete Verbesserung: 1-5ms pro Aufruf**

### Lösung
**Tab-Index-Cache:**
```python
self._tab_index_cache = {}  # {tab_name: index}

def _get_tab_widget_by_name(self, tab_name):
    if tab_name in self._tab_index_cache:
        index = self._tab_index_cache[tab_name]
        if index < self.tab_widget.count():
            return self.tab_widget.widget(index)
    # Fallback: Suche und cache
    # ...
```

---

## 🟢 NIEDRIG: Unnötige Layout-Operationen

### Aktuelles Problem
**Zeilen 2543-2547:** Layout wird jedes Mal neu erstellt:
```python
if self.right_side_widget.layout():
    QWidget().setLayout(self.right_side_widget.layout())  # Unnötig?
    self.right_side_widget.setLayout(right_layout)
```

### Problem
- Layout-Wechsel ist teuer
- Wird bei jedem `show_sources_tab()` aufgerufen (auch wenn nicht nötig)

### Optimierungspotenzial
**Erwartete Verbesserung: 2-5ms pro Aufruf**

### Lösung
- Prüfen ob Layout bereits existiert und korrekt ist
- Nur neu erstellen wenn nötig

---

## Zusammenfassung der neuen Optimierungsvorschläge

| Optimierung | Erwartete Verbesserung | Risiko | Priorität |
|------------|------------------------|--------|-----------|
| 9 Signal-Verbindungen reduzieren | 50-200ms | Niedrig | ⭐⭐⭐ **KRITISCH** |
| Redundante Methodenaufrufe | 30-80ms | Niedrig | ⭐⭐⭐ **HOCH** |
| Code-Duplikation eliminieren | 5-10ms + Qualität | Niedrig | ⭐⭐ Mittel |
| Timer-basierte resize() verbessern | Zuverlässigkeit | Niedrig | ⭐⭐ Mittel |
| Validierung hinzufügen | Stabilität | Niedrig | ⭐⭐ Mittel |
| Plot-Extraktion refactoren | Code-Qualität | Niedrig | ⭐ Niedrig |
| Tab-Suche cachen | 1-5ms | Niedrig | ⭐ Niedrig |
| Layout-Operationen optimieren | 2-5ms | Niedrig | ⭐ Niedrig |

**Gesamtpotenzial der neuen Optimierungen: 88-300ms Verbesserung**

---

## Empfohlene Implementierungsreihenfolge

1. **Signal-Verbindungen reduzieren** ⭐⭐⭐ - Größter Impact, niedriges Risiko
2. **Redundante Methodenaufrufe entfernen** ⭐⭐⭐ - Einfach, hoher Impact
3. **Code-Duplikation eliminieren** ⭐⭐ - Verbessert Wartbarkeit
4. **Validierung hinzufügen** ⭐⭐ - Verbessert Stabilität
5. **Timer-basierte resize() verbessern** ⭐⭐ - Verbessert Zuverlässigkeit
6. **Weitere kleine Optimierungen** ⭐ - Kleine Verbesserungen

---

## Wichtige Hinweise

- **Signal-Verbindungen reduzieren** sollte höchste Priorität haben
- Alle Optimierungen sollten einzeln implementiert und getestet werden
- Performance-Messungen vor/nach jeder Optimierung durchführen
- UI-Funktionalität nach jeder Änderung gründlich testen
- Code-Duplikation sollte refactored werden, auch wenn Performance-Impact gering ist

