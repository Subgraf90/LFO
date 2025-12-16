"""
HTML-Cleaner für Word-exportierte HTML-Dateien.

Bereinigt Word-HTML von Microsoft-spezifischem Code und passt es
an das LFO-Handbuch-Styling an.

Verwendung:
    python word_html_cleaner.py word_export.html manual_de.html
"""

import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from html import unescape


def clean_word_html(html_content: str) -> str:
    """
    Bereinigt Word-HTML von Microsoft-spezifischem Code.
    
    Args:
        html_content: Rohes Word-HTML
        
    Returns:
        Bereinigtes HTML
    """
    # Entferne XML-Namespace-Deklarationen
    html_content = re.sub(r'xmlns[^=]*="[^"]*"', '', html_content)
    
    # Entferne Microsoft Office spezifische Tags
    html_content = re.sub(r'<o:[^>]*>', '', html_content)
    html_content = re.sub(r'</o:[^>]*>', '', html_content)
    html_content = re.sub(r'<m:[^>]*>', '', html_content)
    html_content = re.sub(r'</m:[^>]*>', '', html_content)
    html_content = re.sub(r'<w:[^>]*>', '', html_content)
    html_content = re.sub(r'</w:[^>]*>', '', html_content)
    
    # Entferne VML (Vector Markup Language) Tags
    html_content = re.sub(r'<v:[^>]*>.*?</v:[^>]*>', '', html_content, flags=re.DOTALL)
    
    # Entferne Style-Tags mit Microsoft-spezifischen Styles
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Entferne Meta-Tags mit Microsoft-spezifischen Informationen
    html_content = re.sub(r'<meta[^>]*name="?Generator"?[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<meta[^>]*name="?ProgId"?[^>]*>', '', html_content, flags=re.IGNORECASE)
    
    # Entferne Kommentare
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    
    # Entferne leere Paragraphen
    html_content = re.sub(r'<p[^>]*>\s*</p>', '', html_content, flags=re.IGNORECASE)
    
    # Konvertiere Word-spezifische Formatierungen
    # Fettdruck: <b> oder <strong>
    html_content = re.sub(r'<b\b', '<strong', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</b>', '</strong>', html_content, flags=re.IGNORECASE)
    
    # Kursiv: <i> oder <em>
    html_content = re.sub(r'<i\b', '<em', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</i>', '</em>', html_content, flags=re.IGNORECASE)
    
    # Entferne inline-Styles (werden durch CSS ersetzt)
    html_content = re.sub(r'\s*style="[^"]*"', '', html_content)
    html_content = re.sub(r"\s*style='[^']*'", '', html_content)
    
    # Entferne class-Attribute mit Microsoft-spezifischen Klassen
    html_content = re.sub(r'\s*class="[^"]*Mso[^"]*"', '', html_content, flags=re.IGNORECASE)
    
    # Konvertiere Überschriften (Word verwendet oft <p> mit Styles)
    # Erkenne Überschriften anhand von Text-Größe oder Formatierung
    # (Dies ist eine einfache Heuristik - kann angepasst werden)
    
    # Normalisiere Leerzeichen
    html_content = re.sub(r'\s+', ' ', html_content)
    html_content = re.sub(r'>\s+<', '><', html_content)
    
    # Entferne leere Attribute
    html_content = re.sub(r'\s+>', '>', html_content)
    
    return html_content


def extract_body_content(html_content: str) -> str:
    """
    Extrahiert nur den Body-Inhalt aus HTML.
    
    Args:
        html_content: Vollständiges HTML
        
    Returns:
        Nur der Body-Inhalt
    """
    # Suche nach <body> Tag
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1).strip()
    
    # Falls kein <body> Tag gefunden, nehme alles
    return html_content


def process_images(html_content: str, source_dir: Path, target_dir: Path) -> str:
    """
    Verarbeitet Bilder im HTML:
    - Kopiert Bilder ins images/ Verzeichnis
    - Aktualisiert Bild-Pfade
    
    Args:
        html_content: HTML-Inhalt
        source_dir: Verzeichnis der Quell-HTML-Datei
        target_dir: Ziel-Verzeichnis (HelpWindow-Verzeichnis)
        
    Returns:
        HTML mit aktualisierten Bild-Pfaden
    """
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Finde alle Bild-Tags
    def replace_image(match):
        img_tag = match.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', img_tag, re.IGNORECASE)
        
        if not src_match:
            return img_tag
        
        src_path = src_match.group(1)
        
        # Wenn Bild bereits im images/ Verzeichnis, nichts tun
        if src_path.startswith('images/'):
            return img_tag
        
        # Versuche Bild zu finden
        # Zuerst relativ zum Quell-Verzeichnis
        source_image = source_dir / src_path
        
        # Falls nicht gefunden, versuche im gleichen Verzeichnis wie HTML
        if not source_image.exists():
            # Word speichert Bilder oft in einem Unterordner
            possible_paths = [
                source_dir / src_path,
                source_dir.parent / src_path,
                Path(src_path),  # Absoluter Pfad
            ]
            
            for path in possible_paths:
                if path.exists() and path.is_file():
                    source_image = path
                    break
            else:
                # Bild nicht gefunden
                print(f"Warnung: Bild nicht gefunden: {src_path}")
                return img_tag
        
        # Kopiere Bild ins images/ Verzeichnis
        image_name = source_image.name
        target_image = images_dir / image_name
        
        # Falls Datei bereits existiert, füge Nummer hinzu
        counter = 1
        while target_image.exists():
            stem = source_image.stem
            suffix = source_image.suffix
            target_image = images_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            import shutil
            shutil.copy2(source_image, target_image)
            print(f"✓ Bild kopiert: {image_name} → images/{target_image.name}")
        except Exception as e:
            print(f"Fehler beim Kopieren von {source_image}: {e}")
            return img_tag
        
        # Ersetze src-Attribut
        new_src = f"images/{target_image.name}"
        return re.sub(r'src=["\'][^"\']+["\']', f'src="{new_src}"', img_tag, flags=re.IGNORECASE)
    
    # Ersetze alle Bild-Tags
    html_content = re.sub(r'<img[^>]+>', replace_image, html_content, flags=re.IGNORECASE)
    
    return html_content


def wrap_with_template(body_content: str) -> str:
    """
    Wrappet den Body-Inhalt mit dem LFO-Handbuch-Template.
    
    Args:
        body_content: Body-Inhalt
        
    Returns:
        Vollständiges HTML-Dokument
    """
    template = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LFO - Handbuch</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #555;
            margin-top: 20px;
        }
        
        p {
            margin: 10px 0;
            text-align: justify;
        }
        
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 5px 0;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
        }
        
        blockquote {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 10px 15px;
            margin: 15px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
{content}
</body>
</html>"""
    
    return template.format(content=body_content)


def main():
    """Hauptfunktion: Bereinigt Word-HTML"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("=" * 60)
        print("HTML-Cleaner für Word-exportierte HTML-Dateien")
        print("=" * 60)
        print("\nVerwendung:")
        print("  python3 word_html_cleaner.py <word_export.html> [ziel.html]")
        print("\nParameter:")
        print("  word_export.html  - Word-HTML-Datei (von Word exportiert)")
        print("  ziel.html         - Optional: Name der Ausgabedatei")
        print("                      (Standard: <dateiname>_cleaned.html)")
        print("\nBeispiele:")
        print("  python3 word_html_cleaner.py word_export.html")
        print("  python3 word_html_cleaner.py word_export.html manual_de.html")
        print("  python3 word_html_cleaner.py ~/Desktop/handbuch.html")
        print("\nWas macht der Cleaner?")
        print("  ✓ Entfernt Microsoft-spezifischen Code")
        print("  ✓ Verarbeitet Bilder (kopiert ins images/ Verzeichnis)")
        print("  ✓ Bereinigt Formatierungen")
        print("  ✓ Wrappet mit LFO-Handbuch-Styling")
        print("\nTipp: Für beste Ergebnisse verwenden Sie:")
        print("  python3 markdown_to_html_converter.py manual_de.docx")
        print("  (Direkt aus Word .docx konvertieren)")
        return
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Fehler: Datei nicht gefunden: {input_file}")
        return
    
    # Ziel-Datei bestimmen
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = input_file.parent / f"{input_file.stem}_cleaned.html"
    
    # Ziel-Verzeichnis (HelpWindow-Verzeichnis)
    script_dir = Path(__file__).parent
    target_dir = script_dir
    
    print(f"Lese Word-HTML: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("Bereinige HTML...")
    cleaned_html = clean_word_html(html_content)
    
    print("Extrahiere Body-Inhalt...")
    body_content = extract_body_content(cleaned_html)
    
    print("Verarbeite Bilder...")
    body_content = process_images(body_content, input_file.parent, target_dir)
    
    print("Wrappet mit Template...")
    final_html = wrap_with_template(body_content)
    
    print(f"Speichere bereinigtes HTML: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print("✓ Bereinigung abgeschlossen!")
    print(f"\nDie bereinigte Datei wurde gespeichert als: {output_file}")
    print("Sie können sie jetzt im Help-Fenster verwenden (F1).")


if __name__ == "__main__":
    main()

