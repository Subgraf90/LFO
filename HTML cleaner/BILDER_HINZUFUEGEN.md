# Bilder zum Handbuch hinzufügen

## Problem: Bilder werden nicht angezeigt

Wenn Bilder nicht angezeigt werden, liegt das meist daran, dass Word die Bilder nicht als separate Dateien exportiert hat.

## Lösung 1: Bilder manuell hinzufügen

### Schritt 1: Bilder aus Word extrahieren

1. Öffnen Sie die Word-Datei (`LFO_manual.docx`)
2. Rechtsklick auf das Bild → **"Bild speichern unter..."**
3. Speichern Sie das Bild als `image001.png` (oder entsprechenden Namen)

### Schritt 2: Bild ins images/ Verzeichnis kopieren

```bash
# Kopieren Sie das Bild ins images/ Verzeichnis
cp ~/Desktop/image001.png "/Users/MGraf/Python/LFO_Umgebung/LFO/Module_LFO/Modules_Window/images/"
```

### Schritt 3: Cleaner erneut ausführen (optional)

Falls das HTML bereits erstellt wurde, müssen Sie es nicht neu erstellen - 
die Bilder werden automatisch gefunden, wenn sie im `images/` Verzeichnis liegen.

## Lösung 2: Word erneut exportieren (mit Bildern)

### Option A: "Webseite, gefiltert" mit Bildern

1. Word öffnen → **Datei → Speichern unter**
2. Format: **"Webseite, gefiltert"**
3. **Wichtig:** Stellen Sie sicher, dass Word die Bilder als separate Dateien speichert
4. Prüfen Sie, ob ein `.fld` Ordner erstellt wurde

### Option B: "Webseite" (nicht gefiltert)

1. Word öffnen → **Datei → Speichern unter**
2. Format: **"Webseite"** (nicht "gefiltert")
3. Word erstellt dann meistens einen `.fld` Ordner mit den Bildern
4. Kopieren Sie den `.fld` Ordner in den "HTML cleaner" Ordner

## Lösung 3: Bilder direkt aus Word-Datei extrahieren

Wenn Sie Python haben, können Sie Bilder direkt aus der .docx-Datei extrahieren:

```python
from docx import Document
from docx.parts.image import ImagePart
import os

doc = Document('LFO_manual.docx')
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)

for rel in doc.part.rels.values():
    if "image" in rel.target_ref:
        image = rel.target_part.blob
        ext = rel.target_ref.split('.')[-1]
        filename = f"image_{len(os.listdir(images_dir))}.{ext}"
        with open(os.path.join(images_dir, filename), 'wb') as f:
            f.write(image)
```

## Aktueller Status

Nach dem Ausführen des Cleaners:
- HTML-Datei: `LFO/Module_LFO/Modules_Window/manual_de.html`
- Bilder-Verzeichnis: `LFO/Module_LFO/Modules_Window/images/`

Falls Bilder fehlen:
1. Kopieren Sie die Bilder manuell ins `images/` Verzeichnis
2. Stellen Sie sicher, dass die Dateinamen übereinstimmen (z.B. `image001.png`)

## Prüfen ob Bilder gefunden werden

```bash
# Prüfen welche Bilder im HTML referenziert werden
grep -o 'src="[^"]*"' LFO/Module_LFO/Modules_Window/manual_de.html | grep image

# Prüfen welche Bilder im images/ Verzeichnis sind
ls -la LFO/Module_LFO/Modules_Window/images/
```

