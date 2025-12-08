# Schnellstart: HTML-Cleaner verwenden

## 🎯 Wo muss die HTML-Datei abgelegt werden?

**Kurzantwort:** Die HTML-Datei muss im **"HTML cleaner" Ordner** liegen und **`LFO_manual.html`** heißen!

## 📋 Schritt-für-Schritt

## 📋 Schritt-für-Schritt Anleitung

### Schritt 1: HTML aus Word exportieren

1. Öffnen Sie Ihr Word-Dokument
2. Gehen Sie zu **Datei → Speichern unter**
3. Format wählen: **"Webseite, gefiltert"** (empfohlen)
4. Speichern Sie die Datei als **`LFO_manual.html`**

### Schritt 2: Datei in den Cleaner-Ordner kopieren

**Wichtig:** Die Datei muss **`LFO_manual.html`** heißen und im **"HTML cleaner" Ordner** liegen:

```
/Users/MGraf/Python/LFO_Umgebung/HTML cleaner/LFO_manual.html
```

### Schritt 3: Cleaner ausführen

1. **Öffnen Sie ein Terminal** und navigieren Sie zum Ordner:
   ```bash
   cd "/Users/MGraf/Python/LFO_Umgebung/HTML cleaner"
   ```

2. **Führen Sie den Cleaner aus:**
   ```bash
   python3 word_html_cleaner.py
   ```
   
   Das war's! Der Cleaner verwendet automatisch `LFO_manual.html` aus dem Ordner.

3. **Ergebnis:**
   - Die bereinigte Datei wird automatisch erstellt:
     `LFO/Module_LFO/Modules_Window/manual_de.html`
   - Bilder werden ins `images/` Verzeichnis kopiert

### Alternative: Anderer Dateiname

Falls Ihre Datei anders heißt, können Sie den Namen angeben:

```bash
python3 word_html_cleaner.py mein_handbuch.html
```

## 📁 Verzeichnisstruktur

```
LFO_Umgebung/
├── HTML cleaner/                    ← Hier ist der Cleaner
│   ├── word_html_cleaner.py
│   ├── handbuch_word.html          ← Ihre Word-HTML-Datei (kann hier sein)
│   └── ...
└── LFO/
    └── Module_LFO/
        └── Modules_Window/          ← Hier landet die bereinigte Datei
            ├── manual_de.html       ← ← Ausgabe des Cleaners
            ├── images/              ← Bilder werden hier gespeichert
            └── HelpWindow.py
```

## 🔄 Kompletter Workflow

### 1. HTML aus Word exportieren
- Word öffnen → **Datei → Speichern unter**
- Format: **"Webseite, gefiltert"** (empfohlen)
- **Wichtig:** Speichern als **`LFO_manual.html`**

### 2. Datei in den Cleaner-Ordner kopieren
```bash
# Datei kopieren (im Finder oder Terminal)
# Die Datei muss "LFO_manual.html" heißen!
cp ~/Desktop/LFO_manual.html "/Users/MGraf/Python/LFO_Umgebung/HTML cleaner/"
```

### 3. Cleaner ausführen
```bash
# Zum Cleaner-Verzeichnis navigieren
cd "/Users/MGraf/Python/LFO_Umgebung/HTML cleaner"

# Cleaner ausführen (verwendet automatisch LFO_manual.html)
python3 word_html_cleaner.py
```

### 4. Ergebnis prüfen
- Öffnen Sie das LFO-Programm
- Drücken Sie **F1** (Help-Fenster)
- Die bereinigte Datei sollte sichtbar sein

## ✅ Was passiert beim Cleanen?

1. ✅ HTML wird bereinigt (Microsoft-Code entfernt)
2. ✅ Bilder werden gefunden und kopiert
3. ✅ Ausgabe wird ins HelpWindow-Verzeichnis gespeichert
4. ✅ LFO-Styling wird angewendet

## 🎯 Wichtig zu wissen

- **Eingabe-Datei:** Muss **`LFO_manual.html`** heißen und im **"HTML cleaner" Ordner** liegen
- **Ausgabe-Datei:** Wird automatisch als `manual_de.html` ins `Modules_Window/` Verzeichnis gespeichert
- **Bilder:** Werden automatisch ins `images/` Unterverzeichnis kopiert

## 💡 Einfachste Verwendung

```bash
# 1. LFO_manual.html in den "HTML cleaner" Ordner kopieren
# 2. Terminal öffnen:
cd "/Users/MGraf/Python/LFO_Umgebung/HTML cleaner"

# 3. Cleaner ausführen (ohne Parameter!)
python3 word_html_cleaner.py
```

**Das war's!** Die bereinigte Datei wird automatisch erstellt.

## ❓ Hilfe

Falls Probleme auftreten:
```bash
python3 word_html_cleaner.py --help
```

