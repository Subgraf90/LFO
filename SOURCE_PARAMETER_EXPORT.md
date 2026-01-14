# Source Parameter Widget - Dokumentation

## Übersicht

Das **Source Parameter Widget** wurde erweitert um:
- **Tabellarische Darstellung** aller Source-Parameter unterhalb des Layout-Plots
- **Automatische Aktualisierung** bei Parameteränderungen
- **PDF-Export** für Plot und Tabelle (lizenzfrei)

## Features

### 1. Parameter-Tabelle

Die Tabelle zeigt für jede Source:

| Spalte | Beschreibung | Einheit |
|--------|--------------|---------|
| Source # | Laufende Nummer (1-basiert) | - |
| Typ | Lautsprechertyp | - |
| Pos X | X-Position | m |
| Pos Y | Y-Position | m |
| Pos Z | Z-Position | m |
| Azimuth | Horizontaler Winkel | ° |
| Site | Vertikaler Winkel | ° |
| Level | Lautstärkepegel | dB |
| Delay | Zeitverzögerung | ms |
| Pol. | Polarität (+ normal, - invertiert) | - |

**Zusätzlich werden Array-Gesamt-Parameter angezeigt:**
- Array Gain (dB)
- Array Delay (ms)
- Array Position (X, Y, Z in Metern)

### 2. Visuelle Gestaltung

- **Grüner Header** für bessere Lesbarkeit
- **Abwechselnde Zeilenfarben** (grau/weiß) für Übersichtlichkeit
- **Farbcodierung der Polarität:**
  - Grün (+) = normale Polarität
  - Rot (-) = invertierte Polarität

### 3. Automatische Aktualisierung

Die Tabelle passt sich automatisch an:
- ✅ Parameteränderungen (Level, Delay, Position, etc.)
- ✅ Hinzufügen neuer Sources
- ✅ Löschen von Sources
- ✅ Änderungen des Lautsprechertyps

### 4. PDF-Export (Lizenzfrei)

**Export-Button:** Befindet sich unten links im Widget

**Verwendete Technologie:** 
- matplotlib.backends.backend_pdf (BSD-Lizenz)
- ✅ Vollständig kostenlos
- ✅ Kommerziell nutzbar ohne Lizenzgebühren
- ✅ Professionelle Qualität (300 DPI)

**PDF enthält:**
- Source Layout Plot mit Bemaßung
- Vollständige Parameter-Tabelle
- Array-Informationen
- Metadaten (Titel, Autor, Datum)

## Verwendung

### Widget öffnen
- **Menü:** Window → Source Parameter
- **Tastenkürzel:** `Ctrl+Alt+L`

### PDF exportieren
1. Klicken Sie auf **"PDF Export"** Button (unten links)
2. Wählen Sie Speicherort und Dateiname
3. PDF wird mit 300 DPI gespeichert

## Technische Details

### Dateien
- `WindowSourceParameterWidget.py` - Hauptwidget mit Plot und Tabelle
- `WindowWidgets.py` - Widget-Management und Initialisierung

### Größenanpassung
- Widget-Höhe: 650 px (vorher 500 px)
- Canvas-Höhe: 500 px (vorher 260 px)
- Plot/Tabelle-Verhältnis: 3:2

### Layout-Struktur
```
┌─────────────────────────────────────┐
│  Toolbar (oben rechts)              │
├─────────────────────────────────────┤
│                                     │
│  Source Layout Plot (60%)           │
│  - Lautsprecher-Darstellung         │
│  - Bemaßung                         │
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Parameter-Tabelle (40%)            │
│  - Pro Source: alle Parameter       │
│  - Array-Gesamt-Parameter           │
│                                     │
└─────────────────────────────────────┘
   [PDF Export] Button (unten links)
```

## Lizenz

**PDF-Export ohne Lizenzgebühren:**
- matplotlib verwendet BSD-Lizenz
- Vollständig kostenfrei auch bei kommerziellem Verkauf
- Keine Einschränkungen oder Gebühren

## Vorteile

✅ **Übersichtlich:** Alle Parameter auf einen Blick
✅ **Druckfertig:** Hochauflösende PDFs (300 DPI)
✅ **Professionell:** Sauberes Tabellen-Layout
✅ **Flexibel:** Automatische Anpassung an Änderungen
✅ **Lizenzfrei:** Keine Kosten bei kommerziellem Verkauf
