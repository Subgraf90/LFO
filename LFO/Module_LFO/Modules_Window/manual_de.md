# LFO - Low Frequency Optimizer

**Benutzerhandbuch**

## Inhaltsverzeichnis

1. [Einführung](#einführung)
2. [Grundlagen](#grundlagen)
3. [Dateien verwalten](#dateien-verwalten)
4. [Quellen verwalten](#quellen-verwalten)
5. [Berechnungen](#berechnungen)
6. [Visualisierung](#visualisierung)
7. [Tastenkombinationen](#tastenkombinationen)
8. [Tipps & Tricks](#tipps--tricks)

## Einführung

Willkommen beim **Low Frequency Optimizer (LFO)**. 
Diese Anwendung ermöglicht die Berechnung und Visualisierung von 
Schallfeldern im Niederfrequenzbereich.

> **Hinweis:** Dieses Handbuch führt Sie durch die 
> wichtigsten Funktionen der Anwendung. Für detaillierte Informationen 
> zu spezifischen Funktionen konsultieren Sie bitte die entsprechenden 
> Abschnitte.

## Grundlagen

### Programmstart

Nach dem Start des Programms sehen Sie die Hauptansicht mit mehreren 
Bereichen:

- **SPL Plot:** 3D-Darstellung des Schalldruckpegels
- **Y-Axis Plot:** Schnittansicht entlang der Y-Achse
- **X-Axis Plot:** Schnittansicht entlang der X-Achse
- **Polar Pattern:** Polardiagramm der Schallverteilung

### Menüstruktur

Die Anwendung verfügt über folgende Hauptmenüs:

- **File:** Dateioperationen (Neu, Öffnen, Speichern)
- **Setup:** Einstellungen und Konfiguration
- **Window:** Verwaltung der Fenster und Ansichten
- **Calculation:** Berechnungsmodi und -optionen

## Dateien verwalten

### Neue Datei erstellen

Erstellen Sie eine neue Projektdatei über `File → New` 
oder mit der Tastenkombination `Ctrl+Shift+N`.

### Datei öffnen

Öffnen Sie eine gespeicherte Projektdatei über `File → Open` 
oder mit `Ctrl+O`.

### Datei speichern

Speichern Sie Ihre Arbeit mit `File → Save` (`Ctrl+S`) 
oder als neue Datei mit `File → Save as` (`Ctrl+Shift+S`).

> **Tipp:** Speichern Sie regelmäßig Ihre Arbeit, 
> um Datenverlust zu vermeiden.

## Quellen verwalten

### Quellen-Fenster öffnen

Öffnen Sie das Quellen-Verwaltungsfenster über 
`Window → Sources` oder mit `Ctrl+Alt+S`.

### Lautsprecher hinzufügen

Im Quellen-Fenster können Sie:

- Neue Lautsprecher hinzufügen
- Positionen definieren
- Lautsprecherdaten importieren
- Bestehende Quellen bearbeiten

### Lautsprecherdaten verwalten

Verwenden Sie `Setup → Manage Speaker` (`Ctrl+Alt+M`), 
um Lautsprecherdaten zu importieren, zu exportieren oder zu generieren.

## Berechnungen

### Berechnung starten

Starten Sie eine Berechnung mit dem **Calculate**-Button 
oder der Taste `C`.

### Berechnungsmodi

Unter `Calculation` stehen verschiedene Modi zur Verfügung:

- **SPL Plot:** Standard-Schalldruckpegel-Darstellung
- **Phase alignment:** Phasenausrichtung
- **SPL over time:** Zeitabhängige SPL-Darstellung

### Berechnungseinstellungen

Passen Sie die Berechnungsparameter über 
`Setup → Preferences` (`Ctrl+-`) an.

## Visualisierung

### 3D-Plot Interaktion

Im 3D-SPL-Plot können Sie:

- Mit der Maus rotieren (linke Maustaste)
- Zoomen (Mausrad)
- Verschieben (rechte Maustaste oder Shift + linke Maustaste)

### Ansichten fokussieren

Verwenden Sie die folgenden Tastenkombinationen, um bestimmte Ansichten zu fokussieren:

- `Ctrl+1`: Focus SPL
- `Ctrl+2`: Focus Yaxis
- `Ctrl+3`: Focus Xaxis
- `Ctrl+4`: Focus Polar
- `Ctrl+5`: Default View

### Overlays und Einstellungen

Im 3D-Plot können verschiedene Overlays aktiviert werden:

- Achsen-Overlays
- Impuls-Overlays
- Oberflächen-Overlays

> **Hinweis:** Die Transparenz der Achsen-Overlays kann 
> in den Einstellungen angepasst werden.

## Tastenkombinationen

| Aktion | Tastenkombination |
|--------|-------------------|
| Neue Datei | `Ctrl+Shift+N` |
| Datei öffnen | `Ctrl+O` |
| Speichern | `Ctrl+S` |
| Speichern als | `Ctrl+Shift+S` |
| Berechnung starten | `C` |
| Einstellungen | `Ctrl+-` |
| Quellen-Fenster | `Ctrl+Alt+S` |
| Impuls-Fenster | `Ctrl+Alt+I` |
| Surface-Fenster | `Ctrl+Alt+P` |
| Focus SPL | `Ctrl+1` |
| Focus Yaxis | `Ctrl+2` |
| Focus Xaxis | `Ctrl+3` |
| Focus Polar | `Ctrl+4` |

## Tipps & Tricks

### Optimale Arbeitsweise

- Definieren Sie zuerst Ihre Quellen und deren Positionen
- Laden Sie die entsprechenden Lautsprecherdaten
- Konfigurieren Sie die Berechnungsparameter
- Führen Sie Testberechnungen durch, bevor Sie komplexe Szenarien berechnen

### Performance-Optimierung

Für große Berechnungen können Sie die Berechnungsauflösung in den 
Einstellungen anpassen, um die Berechnungszeit zu verkürzen.

### Visualisierung

Experimentieren Sie mit verschiedenen Overlays und Ansichten, 
um die besten Einblicke in Ihre Daten zu erhalten.

> **Wichtig:** Bei sehr großen Datenmengen kann die 
> Visualisierung Zeit in Anspruch nehmen. Haben Sie Geduld oder reduzieren 
> Sie die Auflösung.

---

*LFO - Low Frequency Optimizer*  
*Version 1.0*

