"""
HTML Cleaner for Word-exported HTML files.

Cleans Word-HTML from Microsoft-specific code and applies
LFO manual styling.

Usage:
    python word_html_cleaner.py word_export.html manual_de.html
"""

import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from html import unescape


def clean_word_html(html_content: str) -> str:
    """
    Cleans Word-HTML from Microsoft-specific code.
    
    Args:
        html_content: Raw Word-HTML
        
    Returns:
        Cleaned HTML
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
    Extracts only the body content from HTML.
    
    Args:
        html_content: Full HTML
        
    Returns:
        Only the body content
    """
    # Suche nach <body> Tag
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1).strip()
    
    # Falls kein <body> Tag gefunden, nehme alles
    return html_content


def process_images(html_content: str, source_dir: Path, target_dir: Path, clear_images: bool = True) -> str:
    """
    Processes images in HTML:
    - Clears existing images directory (optional)
    - Copies images to images/ directory
    - Updates image paths
    
    Args:
        html_content: HTML content
        source_dir: Source HTML file directory
        target_dir: Target directory (Help directory)
        clear_images: If True, clears existing images directory before copying
        
    Returns:
        HTML with updated image paths
    """
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Clear existing images if requested
    if clear_images and images_dir.exists():
        import shutil
        for image_file in images_dir.iterdir():
            if image_file.is_file():
                try:
                    image_file.unlink()
                    print(f"  Removed old image: {image_file.name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {image_file.name}: {e}")
    
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
        
        # Falls nicht gefunden, versuche verschiedene mögliche Pfade
        if not source_image.exists():
            # Word speichert Bilder oft in einem Unterordner (.fld)
            # Beispiel: LFO_manual.html → LFO_manual.fld/image001.png
            html_name = source_dir.name.replace('.html', '').replace('.htm', '')
            possible_paths = [
                source_dir / src_path,  # Relativ zum HTML-Verzeichnis
                source_dir.parent / src_path,  # Ein Verzeichnis höher
                source_dir / f"{html_name}.fld" / src_path.split('/')[-1],  # Word .fld Ordner
                source_dir / src_path.split('/')[-1],  # Nur Dateiname im HTML-Verzeichnis
                Path(src_path),  # Absoluter Pfad
            ]
            
            # Prüfe auch ob es ein .fld Verzeichnis gibt
            fld_dir = source_dir / f"{html_name}.fld"
            if fld_dir.exists() and fld_dir.is_dir():
                # Suche Bild im .fld Verzeichnis
                image_filename = src_path.split('/')[-1]
                possible_paths.insert(0, fld_dir / image_filename)
            
            for path in possible_paths:
                if path.exists() and path.is_file():
                    source_image = path
                    break
            else:
                # Image not found - but we still replace the path
                # so HTML is correct (image can be added manually later)
                print(f"Warning: Image not found: {src_path}")
                print(f"  Searched in: {source_dir}")
                print(f"  Possible paths:")
                for p in possible_paths:
                    print(f"    - {p}")
                # Ersetze trotzdem den Pfad zu images/, auch wenn Bild fehlt
                image_filename = src_path.split('/')[-1]
                new_src = f"images/{image_filename}"
                return re.sub(r'src=["\'][^"\']+["\']', f'src="{new_src}"', img_tag, flags=re.IGNORECASE)
        
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
            print(f"✓ Image copied: {image_name} → images/{target_image.name}")
        except Exception as e:
            print(f"Error copying {source_image}: {e}")
            return img_tag
        
        # Ersetze src-Attribut
        new_src = f"images/{target_image.name}"
        return re.sub(r'src=["\'][^"\']+["\']', f'src="{new_src}"', img_tag, flags=re.IGNORECASE)
    
    # Ersetze alle Bild-Tags
    html_content = re.sub(r'<img[^>]+>', replace_image, html_content, flags=re.IGNORECASE)
    
    return html_content


def wrap_with_template(body_content: str) -> str:
    """
    Wraps body content with LFO manual template.
    
    Args:
        body_content: Body content
        
    Returns:
        Complete HTML document
    """
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LFO - User Manual</title>
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
CONTENT_PLACEHOLDER
</body>
</html>"""
    
    # Verwende einfache String-Ersetzung statt format(), um geschweifte Klammern im HTML zu vermeiden
    result = template.replace('CONTENT_PLACEHOLDER', body_content)
    
    return result


def main():
    """Main function: Cleans Word-HTML"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("=" * 60)
        print("HTML Cleaner for Word-exported HTML files")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 word_html_cleaner.py [word_export.html] [output.html]")
        print("\nParameters:")
        print("  word_export.html  - Optional: Word-HTML file")
        print("                      (Default: LFO_manual.html in cleaner folder)")
        print("  output.html      - Optional: Output file name")
        print("                      (Default: manual_de.html)")
        print("\nExamples:")
        print("  python3 word_html_cleaner.py")
        print("    → Uses: HTML cleaner/LFO_manual.html")
        print("    → Creates: Modules_Help/manual_de.html")
        print("    → Images: Modules_Help/images/")
        print("")
        print("  python3 word_html_cleaner.py LFO_manual.html")
        print("    → Uses: HTML cleaner/LFO_manual.html")
        print("    → Creates: Modules_Help/manual_de.html")
        print("")
        print("  python3 word_html_cleaner.py LFO_manual.html manual.html")
        print("    → Uses: HTML cleaner/LFO_manual.html")
        print("    → Creates: Modules_Help/manual.html")
        print("\nWhat does the cleaner do?")
        print("  ✓ Removes Microsoft-specific code")
        print("  ✓ Processes images (copies to images/ directory)")
        print("  ✓ Cleans formatting")
        print("  ✓ Wraps with LFO manual styling")
        return
    
    # Ziel-Verzeichnis (HelpWindow-Verzeichnis)
    script_dir = Path(__file__).parent
    
    # Default input file: LFO_manual.html in cleaner folder
    if len(sys.argv) < 2:
        input_file = script_dir / "LFO_manual.html"
        if not input_file.exists():
            print(f"Error: Default file not found: {input_file}")
            print(f"\nPlease place 'LFO_manual.html' in the cleaner folder:")
            print(f"  {script_dir}")
            print("\nOr specify a different filename:")
            print("  python3 word_html_cleaner.py <filename.html>")
            return
    else:
        input_file = Path(sys.argv[1])
        # If not absolute path, search in cleaner folder
        if not input_file.is_absolute():
            input_file = script_dir / input_file
    
    input_file = Path(input_file)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    # Find Help directory
    # Cleaner is in: LFO_Umgebung/HTML cleaner/
    # Help directory is in: LFO_Umgebung/LFO/Module_LFO/Modules_Help/
    help_dir = script_dir.parent / "LFO" / "Module_LFO" / "Modules_Help"
    
    # Create Help directory if it doesn't exist
    if not help_dir.exists():
        help_dir.mkdir(parents=True, exist_ok=True)
        (help_dir / "images").mkdir(exist_ok=True)
        print(f"✓ Help directory created: {help_dir}")
    
    # Determine output file
    if len(sys.argv) > 2:
        # If only filename (no path), save to Help directory
        output_file = Path(sys.argv[2])
        if not output_file.is_absolute() and output_file.parent == Path('.'):
            output_file = help_dir / output_file.name
    else:
        # Default: manual_de.html in Help directory
        output_file = help_dir / "manual_de.html"
    
    target_dir = help_dir
    
    print(f"Reading Word-HTML: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("Cleaning HTML...")
    cleaned_html = clean_word_html(html_content)
    
    print("Extracting body content...")
    body_content = extract_body_content(cleaned_html)
    
    print("Processing images...")
    print("  Clearing old images from images/ directory...")
    body_content = process_images(body_content, input_file.parent, target_dir, clear_images=True)
    
    print("Wrapping with template...")
    final_html = wrap_with_template(body_content)
    
    print(f"Saving cleaned HTML: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print("✓ Cleaning completed!")
    print(f"\nCleaned file saved as: {output_file}")
    print(f"Images directory: {target_dir / 'images'}")
    print("You can now use it in the Help window (F1).")


if __name__ == "__main__":
    main()

