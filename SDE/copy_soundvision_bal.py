"""
Kopiert und dokumentiert SoundVision BAL-Dateien

Kopiert alle BAL-Dateien von SoundVision in ein lokales Verzeichnis
und erstellt eine Dokumentation.
"""

import shutil
from pathlib import Path
import json
from datetime import datetime


def copy_balloon_files(source_dir: Path, dest_dir: Path) -> dict:
    """
    Kopiert alle BAL-Dateien von SoundVision.
    
    Returns:
        Dictionary mit Informationen über kopierte Dateien
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    bal_files = list(source_dir.glob("*.bal"))
    
    copied_files = []
    total_size = 0
    
    for bal_file in bal_files:
        dest_file = dest_dir / bal_file.name
        
        # Kopiere Datei
        shutil.copy2(bal_file, dest_file)
        
        file_info = {
            'name': bal_file.name,
            'size': bal_file.stat().st_size,
            'copied': True,
            'source': str(bal_file),
            'destination': str(dest_file)
        }
        
        copied_files.append(file_info)
        total_size += file_info['size']
    
    return {
        'files': copied_files,
        'total_count': len(copied_files),
        'total_size': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'timestamp': datetime.now().isoformat()
    }


def create_documentation(balloons_info: dict, output_file: Path):
    """Erstellt Dokumentation der BAL-Dateien"""
    with open(output_file, 'w') as f:
        f.write("# SoundVision BAL-Dateien Dokumentation\n\n")
        f.write(f"**Erstellt**: {balloons_info['timestamp']}\n\n")
        f.write(f"**Anzahl Dateien**: {balloons_info['total_count']}\n")
        f.write(f"**Gesamtgröße**: {balloons_info['total_size_mb']:.2f} MB\n\n")
        
        f.write("## Dateien\n\n")
        f.write("| Dateiname | Größe (Bytes) | Größe (MB) |\n")
        f.write("|-----------|---------------|------------|\n")
        
        for file_info in sorted(balloons_info['files'], key=lambda x: x['name']):
            size_mb = file_info['size'] / (1024 * 1024)
            f.write(f"| {file_info['name']} | {file_info['size']:,} | {size_mb:.3f} |\n")
        
        f.write("\n## Hinweise\n\n")
        f.write("- Alle Dateien sind **verschlüsselt** (Entropie ~7.8)\n")
        f.write("- Format: Binärdaten, nicht ZIP oder XML\n")
        f.write("- Header: `1df93efb1a2c50731916a8e484d9d6fc` (erste 16 Bytes)\n")
        f.write("- Verschlüsselung: Wahrscheinlich AES (OpenSSL EVP)\n")
        f.write("- Entschlüsselung: Erfordert privaten RSA-Schlüssel\n\n")
        
        f.write("## Verwendung\n\n")
        f.write("Diese Dateien können verwendet werden für:\n")
        f.write("- Analyse der Verschlüsselungsmethode\n")
        f.write("- Vergleich mit ArrayCalc BAL-Dateien\n")
        f.write("- Entwicklung von Entschlüsselungs-Tools\n")
        f.write("- Import in LFO-System (nach Entschlüsselung)\n")


def main():
    """Hauptfunktion"""
    source_dir = Path("/Applications/Soundvision.app/Contents/Resources/balloons")
    dest_dir = Path("/Users/MGraf/Python/LFO_Umgebung/SDE/soundvision_balloons")
    
    if not source_dir.exists():
        print(f"FEHLER: SoundVision Balloons-Verzeichnis nicht gefunden: {source_dir}")
        return
    
    print("=== Kopiere SoundVision BAL-Dateien ===\n")
    print(f"Quelle: {source_dir}")
    print(f"Ziel: {dest_dir}\n")
    
    # Kopiere Dateien
    balloons_info = copy_balloon_files(source_dir, dest_dir)
    
    print(f"✓ {balloons_info['total_count']} Dateien kopiert")
    print(f"✓ Gesamtgröße: {balloons_info['total_size_mb']:.2f} MB\n")
    
    # Erstelle Dokumentation
    doc_file = dest_dir / "README.md"
    create_documentation(balloons_info, doc_file)
    print(f"✓ Dokumentation erstellt: {doc_file}")
    
    # Erstelle JSON-Metadaten
    json_file = dest_dir / "metadata.json"
    with open(json_file, 'w') as f:
        json.dump(balloons_info, f, indent=2)
    print(f"✓ Metadaten gespeichert: {json_file}")
    
    # Zeige erste 10 Dateien
    print("\n=== Erste 10 Dateien ===")
    for file_info in balloons_info['files'][:10]:
        size_mb = file_info['size'] / (1024 * 1024)
        print(f"  {file_info['name']}: {size_mb:.3f} MB")
    
    print(f"\n=== Abgeschlossen ===")
    print(f"Alle Dateien in: {dest_dir}")


if __name__ == "__main__":
    main()

