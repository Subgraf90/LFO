# Anleitung: HTML-Cleaner verwenden

## Übersicht

Der HTML-Cleaner bereinigt Word-exportierte HTML-Dateien und macht sie für das LFO-Handbuch verwendbar.

## Schritt-für-Schritt Anleitung

### Schritt 1: HTML aus Word exportieren

1. Öffnen Sie Ihr Word-Dokument
2. Gehen Sie zu **Datei → Speichern unter**
3. Wählen Sie als Format:
   - **"Webseite, gefiltert"** (empfohlen - weniger Müll)
   - Oder: **"Webseite"** (wird auch bereinigt)
4. Speichern Sie die Datei (z.B. als `handbuch_word.html`)

### Schritt 2: HTML-Cleaner ausführen

Öffnen Sie ein Terminal und navigieren Sie zum Modulordner:

```bash
cd LFO/Module_LFO/Modules_Window
```

Führen Sie dann den Cleaner aus:

```bash
python3 word_html_cleaner.py handbuch_word.html manual_de.html
```

**Parameter:**
- Erster Parameter: Ihre Word-HTML-Datei (die Sie exportiert haben)
- Zweiter Parameter (optional): Name der Ausgabedatei (Standard: `handbuch_word_cleaned.html`)

### Schritt 3: Ergebnis prüfen

Nach dem Ausführen:
- Die bereinigte HTML-Datei wird im `Modules_Window/` Verzeichnis gespeichert
- Bilder werden automatisch ins `images/` Verzeichnis kopiert
- Die Datei kann direkt im Help-Fenster verwendet werden (F1)

## Beispiele

### Beispiel 1: Standard-Verwendung

```bash
cd LFO/Module_LFO/Modules_Window
python3 word_html_cleaner.py word_export.html
```

Ergebnis: `word_export_cleaned.html` wird erstellt

### Beispiel 2: Mit Ziel-Dateiname

```bash
cd LFO/Module_LFO/Modules_Window
python3 word_html_cleaner.py word_export.html manual_de.html
```

Ergebnis: `manual_de.html` wird erstellt/überschrieben

### Beispiel 3: Mit Pfad

```bash
cd LFO/Module_LFO/Modules_Window
python3 word_html_cleaner.py ~/Desktop/handbuch.html manual_de.html
```

Ergebnis: Datei von Desktop wird verarbeitet

## Was macht der Cleaner?

Der HTML-Cleaner:

1. ✅ **Entfernt Microsoft-spezifischen Code:**
   - XML-Namespaces
   - Office-Tags (`<o:>`, `<m:>`, `<w:>`)
   - VML (Vector Markup Language) Tags
   - Microsoft-Styles

2. ✅ **Verarbeitet Bilder:**
   - Findet alle Bilder im HTML
   - Kopiert sie ins `images/` Verzeichnis
   - Aktualisiert Bild-Pfade im HTML

3. ✅ **Bereinigt Formatierung:**
   - Entfernt inline-Styles
   - Normalisiert HTML-Struktur
   - Konvertiert Word-Formatierungen zu Standard-HTML

4. ✅ **Wrappet mit LFO-Styling:**
   - Fügt das LFO-Handbuch-CSS hinzu
   - Erstellt vollständiges HTML-Dokument

## Fehlerbehebung

### "Datei nicht gefunden"

**Problem:** Die HTML-Datei kann nicht gefunden werden

**Lösung:**
- Prüfen Sie den Pfad zur Datei
- Verwenden Sie absolute Pfade wenn nötig
- Stellen Sie sicher, dass die Datei existiert

### "Bilder nicht gefunden"

**Problem:** Warnung beim Verarbeiten von Bildern

**Lösung:**
- Word speichert Bilder oft in einem Unterordner
- Der Cleaner sucht automatisch in verschiedenen Verzeichnissen
- Falls Bilder fehlen, kopieren Sie sie manuell ins `images/` Verzeichnis

### Python-Fehler

**Problem:** `python3` nicht gefunden

**Lösung:**
- Verwenden Sie `python` statt `python3`
- Oder: `python3.11` (je nach Installation)

## Tipps

1. **Bilder in Word:**
   - Verwenden Sie "Einfügen → Bild" in Word
   - Bilder werden automatisch verarbeitet

2. **Formatierung:**
   - Verwenden Sie Word-Überschriften-Stile (Überschrift 1, 2, 3)
   - Diese werden zu `<h1>`, `<h2>`, `<h3>` konvertiert

3. **Tabellen:**
   - Tabellen werden automatisch übernommen
   - Erhalten das LFO-Handbuch-Styling

4. **Mehrere Dateien:**
   - Sie können mehrere HTML-Dateien bereinigen
   - Jede wird separat verarbeitet

## Alternative: Direkt .docx verwenden

Wenn Sie noch nicht exportiert haben, können Sie auch direkt die Word-Datei verwenden:

```bash
python3 markdown_to_html_converter.py manual_de.docx
```

Das ist oft einfacher, da keine Bereinigung nötig ist!

## Hilfe

Bei Problemen:
- Prüfen Sie die Fehlermeldungen im Terminal
- Stellen Sie sicher, dass Python 3 installiert ist
- Die bereinigte HTML-Datei kann im Browser geöffnet werden zur Prüfung

