# Handbuch-Verwaltung für LFO

## Übersicht

Das Handbuch kann auf drei Arten bearbeitet werden:

1. **Word-Datei (empfohlen für Bilder)** - Direkt aus Word konvertieren, Bilder werden automatisch extrahiert
2. **Markdown** - Einfach zu schreiben, wird zu HTML konvertiert
3. **Direktes HTML** - Vollständige Kontrolle über Formatierung

## Installation

Für Word-Unterstützung benötigen Sie:

```bash
pip install python-docx pillow
```

Oder mit conda:

```bash
conda install -c conda-forge python-docx pillow
```

## Option 1: Word-Datei verwenden (EMPFOHLEN für Bilder)

### Vorteile:
- ✅ Direkt aus Word übernehmen
- ✅ Bilder werden automatisch extrahiert
- ✅ Formatierung bleibt erhalten
- ✅ Keine manuelle Konvertierung nötig

### So geht's:

1. **Schreiben Sie in Word** (`manual_de.docx`)
   - Verwenden Sie Überschriften-Stile (Überschrift 1, Überschrift 2, etc.)
   - Fügen Sie Bilder direkt ein (Einfügen → Bild)
   - Formatieren Sie Text wie gewohnt (Fettdruck, Kursiv, etc.)

2. **Konvertieren Sie zu HTML:**
   ```bash
   cd LFO/Module_LFO/Modules_Window
   python3 markdown_to_html_converter.py manual_de.docx
   ```
   
   Oder einfach (wenn die Datei `manual_de.docx` heißt):
   ```bash
   python3 markdown_to_html_converter.py
   ```

3. **Fertig!** 
   - Das HTML wird automatisch generiert
   - Bilder werden in `images/` gespeichert
   - Alle Formatierungen bleiben erhalten

### Tipps für Word:
- Verwenden Sie **Überschrift 1-3** für Überschriften (werden zu h1-h3)
- Bilder werden automatisch an der richtigen Stelle eingefügt
- Tabellen werden automatisch konvertiert
- Fettdruck, Kursiv, etc. bleiben erhalten

## Option 2: Markdown verwenden

### Vorteile:
- ✅ Sehr einfach zu schreiben
- ✅ Keine HTML-Kenntnisse nötig
- ✅ Automatische Formatierung
- ✅ Einfache Versionierung

### So geht's:

1. **Bearbeiten Sie `manual_de.md`** mit einem beliebigen Texteditor
   - Verwenden Sie Markdown-Syntax (siehe Beispiele unten)
   
2. **Konvertieren Sie zu HTML:**
   ```bash
   cd LFO/Module_LFO/Modules_Window
   python3 markdown_to_html_converter.py
   ```
   
3. **Fertig!** Das HTML wird automatisch aktualisiert.

### Markdown-Syntax-Beispiele:

```markdown
# Hauptüberschrift
## Unterüberschrift
### Unter-Unterüberschrift

**Fettdruck** und *Kursiv*

- Listenpunkt 1
- Listenpunkt 2

1. Nummerierte Liste
2. Zweiter Punkt

`Code inline`

> **Hinweis:** Ein wichtiger Hinweis

| Spalte 1 | Spalte 2 |
|----------|----------|
| Wert 1   | Wert 2   |

[Link-Text](https://example.com)
```

## Option 3: Direktes HTML-Editing

### Vorteile:
- ✅ Vollständige Kontrolle
- ✅ Komplexe Formatierungen möglich

### So geht's:

1. **Öffnen Sie `manual_de.html`** in einem Editor
2. **Bearbeiten Sie den Inhalt** zwischen `<body>` und `</body>`
3. **Speichern** - fertig!

### HTML-Struktur:

```html
<h1>Überschrift</h1>
<p>Absatz-Text</p>
<ul>
  <li>Listenpunkt</li>
</ul>
```

## Option 4: Word-HTML direkt verwenden (mit Cleaner)

### Vorteile:
- ✅ Direkt aus Word exportieren
- ✅ Automatische Bereinigung von Word-"Müll"
- ✅ Bilder werden automatisch verarbeitet

### So geht's:

1. **Schreiben Sie in Word**
2. **Speichern als HTML:**
   - Datei → Speichern unter
   - Format: "Webseite, gefiltert" (HTML)
   - Oder: "Webseite" (HTML) - wird dann bereinigt

3. **Bereinigen Sie das HTML:**
   ```bash
   cd LFO/Module_LFO/Modules_Window
   python3 word_html_cleaner.py word_export.html manual_de.html
   ```
   
   Der Cleaner:
   - Entfernt Microsoft-spezifischen Code
   - Verarbeitet Bilder (kopiert sie ins `images/` Verzeichnis)
   - Wrappet mit dem LFO-Handbuch-Styling
   - Bereinigt Formatierungen

4. **Fertig!** Die bereinigte Datei kann direkt im Help-Fenster verwendet werden

### Alternative: Direkt einfügen

Sie können Word-HTML auch direkt in den Modulordner kopieren:
- Das HelpWindow findet alle `.html` Dateien automatisch
- **ABER:** Word-HTML enthält oft viel unnötigen Code
- **EMPFOHLEN:** Verwenden Sie den Cleaner für bessere Ergebnisse

## Workflow-Empfehlung

**Für neue Inhalte mit Bildern:**
1. Schreiben Sie in Word (`manual_de.docx`)
2. Fügen Sie Bilder direkt ein
3. Führen Sie den Konverter aus: `python3 markdown_to_html_converter.py`
4. Testen Sie im Help-Fenster (F1)

**Für Text ohne Bilder:**
1. Schreiben Sie in `manual_de.md` (Markdown)
2. Führen Sie den Konverter aus
3. Testen Sie im Help-Fenster (F1)

**Für schnelle Änderungen:**
- Bearbeiten Sie direkt `manual_de.html`

## Tipps

- **Bilder in Word:** 
  - Einfach in Word einfügen (Einfügen → Bild)
  - Werden automatisch extrahiert und in `images/` gespeichert
  - Erscheinen an der richtigen Stelle im HTML

- **Bilder in Markdown:**
  ```markdown
  ![Beschreibung](images/screenshot.png)
  ```

- **Bilder in HTML:**
  ```html
  <img src="images/screenshot.png" alt="Beschreibung">
  ```

- **Links:** Verwenden Sie Anker-Links für Sprungmarken:
  ```html
  <a href="#abschnitt-id">Link-Text</a>
  ```

- **Styling:** Das CSS ist bereits in der HTML-Datei definiert. 
  Sie können es anpassen, wenn nötig.

## Hilfe

Bei Fragen oder Problemen:
- Markdown-Syntax: https://www.markdownguide.org/
- HTML-Referenz: https://developer.mozilla.org/de/docs/Web/HTML

