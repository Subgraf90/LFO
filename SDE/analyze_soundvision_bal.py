#!/usr/bin/env python3
"""
Analysiert SoundVision BAL-Dateien direkt aus dem App-Bundle.

Geht davon aus, dass beim App-Start die Daten entpackt werden.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_bal_file(bal_path: Path) -> Dict:
    """Analysiert eine BAL-Datei und gibt Metadaten zurück."""
    logger.info(f"Analysiere: {bal_path.name}")
    
    with open(bal_path, 'rb') as f:
        data = f.read()
    
    size = len(data)
    
    # Analysiere Header
    header = data[:64] if len(data) >= 64 else data
    header_hex = header.hex()
    
    # Prüfe auf bekannte Header
    is_encrypted = True
    header_magic = None
    
    # Prüfe auf mögliche Magic Bytes
    if data[:4] == b'\x1d\xf9\x3e\xfb':
        header_magic = "SoundVision encrypted?"
    elif data[:2] == b'PK':
        header_magic = "ZIP archive"
        is_encrypted = False
    elif data[:4] == b'\x89PNG':
        header_magic = "PNG image"
        is_encrypted = False
    elif data[:2] == b'\x1f\x8b':
        header_magic = "GZIP compressed"
        is_encrypted = False
    
    # Statistische Analyse
    entropy = calculate_entropy(data[:1024])  # Erste 1KB
    
    # Suche nach Mustern
    patterns = {
        'null_bytes': data.count(b'\x00'),
        'repeating_bytes': find_repeating_patterns(data[:1024]),
        'ascii_strings': find_ascii_strings(data[:1024])
    }
    
    result = {
        'path': str(bal_path),
        'name': bal_path.name,
        'size': size,
        'header_hex': header_hex[:128],  # Erste 64 Bytes als Hex
        'header_magic': header_magic,
        'is_encrypted': is_encrypted,
        'entropy': entropy,
        'patterns': patterns
    }
    
    return result


def calculate_entropy(data: bytes) -> float:
    """Berechnet die Shannon-Entropie (höher = mehr zufällig/verschlüsselt)."""
    if not data:
        return 0.0
    
    import math
    from collections import Counter
    
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def find_repeating_patterns(data: bytes, min_length: int = 4) -> List[str]:
    """Findet sich wiederholende Byte-Muster."""
    patterns = []
    seen = {}
    
    for i in range(len(data) - min_length):
        pattern = data[i:i+min_length]
        if pattern in seen:
            patterns.append(pattern.hex())
        else:
            seen[pattern] = i
    
    return list(set(patterns))[:10]  # Max 10 Muster


def find_ascii_strings(data: bytes, min_length: int = 4) -> List[str]:
    """Findet ASCII-Strings in den Daten."""
    strings = []
    current = b''
    
    for byte in data:
        if 32 <= byte <= 126:  # Druckbare ASCII-Zeichen
            current += bytes([byte])
        else:
            if len(current) >= min_length:
                try:
                    strings.append(current.decode('ascii'))
                except:
                    pass
            current = b''
    
    if len(current) >= min_length:
        try:
            strings.append(current.decode('ascii'))
        except:
            pass
    
    return strings[:10]  # Max 10 Strings


def check_temp_files(app_bundle: Path) -> List[Path]:
    """Prüft auf temporäre/entpackte Dateien beim App-Start."""
    temp_locations = [
        Path.home() / "Library" / "Caches" / "com.d&b.soundvision",
        Path.home() / "Library" / "Application Support" / "SoundVision",
        Path("/tmp"),
        app_bundle.parent / "balloons_extracted",
    ]
    
    found_files = []
    for location in temp_locations:
        if location.exists():
            for file in location.rglob("*.bal"):
                found_files.append(file)
            for file in location.rglob("*balloon*"):
                found_files.append(file)
    
    return found_files


def main():
    """Hauptfunktion."""
    # Suche nach SoundVision App-Bundle
    app_bundles = [
        Path("/Users/MGraf/Desktop/soundvision_resigned/Soundvision.app"),
        Path("/Applications/Soundvision.app"),
    ]
    
    app_bundle = None
    for bundle in app_bundles:
        if bundle.exists():
            app_bundle = bundle
            break
    
    if not app_bundle:
        logger.error("SoundVision App-Bundle nicht gefunden!")
        return
    
    logger.info(f"Gefundenes App-Bundle: {app_bundle}")
    
    # Finde BAL-Dateien
    balloons_dir = app_bundle / "Contents" / "Resources" / "balloons"
    if not balloons_dir.exists():
        logger.error(f"Balloons-Verzeichnis nicht gefunden: {balloons_dir}")
        return
    
    bal_files = list(balloons_dir.glob("*.bal"))
    logger.info(f"Gefunden: {len(bal_files)} BAL-Dateien")
    
    if not bal_files:
        logger.error("Keine BAL-Dateien gefunden!")
        return
    
    # Analysiere erste 5 BAL-Dateien
    logger.info("\n=== Analysiere BAL-Dateien ===")
    results = []
    for bal_file in bal_files[:5]:
        try:
            result = analyze_bal_file(bal_file)
            results.append(result)
            
            print(f"\n{result['name']}:")
            print(f"  Größe: {result['size']:,} Bytes")
            print(f"  Header (Hex): {result['header_hex'][:64]}...")
            print(f"  Magic: {result['header_magic'] or 'Unbekannt'}")
            print(f"  Verschlüsselt: {result['is_encrypted']}")
            print(f"  Entropie: {result['entropy']:.2f} (max 8.0)")
            if result['patterns']['ascii_strings']:
                print(f"  ASCII-Strings: {result['patterns']['ascii_strings'][:3]}")
        except Exception as e:
            logger.error(f"Fehler bei {bal_file.name}: {e}")
    
    # Prüfe auf temporäre Dateien
    logger.info("\n=== Prüfe auf temporäre/entpackte Dateien ===")
    temp_files = check_temp_files(app_bundle)
    if temp_files:
        logger.info(f"Gefunden: {len(temp_files)} temporäre Dateien")
        for f in temp_files[:10]:
            print(f"  {f}")
    else:
        logger.info("Keine temporären Dateien gefunden")
        logger.info("Hinweis: Starte SoundVision und prüfe erneut!")
    
    # Speichere Analyse-Ergebnisse
    output_file = Path(__file__).parent / "soundvision_bal_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("=== SoundVision BAL-Dateien Analyse ===\n\n")
        for result in results:
            f.write(f"{result['name']}:\n")
            f.write(f"  Größe: {result['size']:,} Bytes\n")
            f.write(f"  Header: {result['header_hex']}\n")
            f.write(f"  Magic: {result['header_magic'] or 'Unbekannt'}\n")
            f.write(f"  Entropie: {result['entropy']:.2f}\n")
            f.write("\n")
    
    logger.info(f"\nAnalyse gespeichert in: {output_file}")


if __name__ == "__main__":
    main()
