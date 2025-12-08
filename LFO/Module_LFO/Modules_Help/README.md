# Help-Verzeichnis für LFO

Dieses Verzeichnis enthält alle Help-Dateien und Bilder für das LFO-Handbuch.

## Verzeichnisstruktur

```
Modules_Help/
├── manual_de.html          # Haupt-Handbuch (HTML)
├── images/                 # Bilder für das Handbuch
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── README.md              # Diese Datei
```

## Verwendung

### HTML-Dateien hinzufügen

1. Legen Sie HTML-Dateien direkt in dieses Verzeichnis ab
2. Das Help-Fenster findet sie automatisch (F1 im Programm)

### Bilder hinzufügen

1. Kopieren Sie Bilder ins `images/` Unterverzeichnis
2. Referenzieren Sie sie im HTML mit: `<img src="images/bildname.png">`

### Cleaner verwenden

Der HTML-Cleaner speichert automatisch hier:
- HTML-Dateien → `Modules_Help/`
- Bilder → `Modules_Help/images/`

## Workflow

1. **Word-HTML exportieren** → `LFO_manual.html` im "HTML cleaner" Ordner
2. **Cleaner ausführen:**
   ```bash
   cd "HTML cleaner"
   python3 word_html_cleaner.py
   ```
3. **Ergebnis:** Dateien werden automatisch hier gespeichert
4. **Im Programm:** F1 drücken → Help-Fenster zeigt die Dateien

## Dateien verwalten

- Alle `.html` Dateien in diesem Verzeichnis werden im Help-Fenster angezeigt
- Bilder müssen im `images/` Unterverzeichnis liegen
- Relative Pfade funktionieren automatisch

