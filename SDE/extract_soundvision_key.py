#!/usr/bin/env python3
"""
Sucht nach Verschlüsselungsschlüsseln für SoundVision BAL-Dateien im App-Bundle.
"""

import os
import re
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def search_for_keys_in_file(filepath: Path, bal_header: bytes) -> List[dict]:
    """Sucht nach möglichen Schlüsseln in einer Datei."""
    results = []
    
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
    except Exception as e:
        return results
    
    # Suche nach dem BAL-Header (könnte Teil eines Schlüssels sein)
    if bal_header in data:
        offset = data.find(bal_header)
        results.append({
            'type': 'header_match',
            'offset': offset,
            'context': data[max(0, offset-32):offset+64].hex()
        })
    
    # Suche nach AES-ähnlichen Schlüsseln (16, 24, 32 Bytes)
    for key_len in [16, 24, 32]:
        # Suche nach wiederholten Mustern, die wie Schlüssel aussehen
        for i in range(len(data) - key_len):
            key_candidate = data[i:i+key_len]
            # Prüfe auf hohe Entropie (verschlüsselte Schlüssel)
            entropy = calculate_simple_entropy(key_candidate)
            if entropy > 6.0:  # Hohe Entropie
                results.append({
                    'type': f'aes_key_candidate_{key_len}',
                    'offset': i,
                    'key_hex': key_candidate.hex(),
                    'entropy': entropy
                })
    
    # Suche nach Strings, die auf Verschlüsselung hindeuten
    try:
        text = data.decode('utf-8', errors='ignore')
        if 'aes' in text.lower() or 'encrypt' in text.lower() or 'decrypt' in text.lower():
            results.append({
                'type': 'encryption_string',
                'context': text[max(0, text.lower().find('aes')-50):text.lower().find('aes')+50]
            })
    except:
        pass
    
    return results


def calculate_simple_entropy(data: bytes) -> float:
    """Einfache Entropie-Berechnung."""
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


def search_binary_for_patterns(binary_path: Path) -> List[dict]:
    """Sucht nach Mustern in einer Binärdatei."""
    results = []
    
    try:
        with open(binary_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        return results
    
    # Bekannter BAL-Header
    bal_header = bytes.fromhex('1df93efb1a2c50731916a8e484d9d6fc')
    
    # Suche nach dem Header
    if bal_header in data:
        offset = data.find(bal_header)
        # Suche nach umgebenden Bytes, die wie ein Schlüssel aussehen
        context_start = max(0, offset - 64)
        context_end = min(len(data), offset + 128)
        context = data[context_start:context_end]
        
        results.append({
            'type': 'bal_header_found',
            'offset': offset,
            'context_hex': context.hex()
        })
    
    # Suche nach AES-Key-ähnlichen Mustern
    # AES-Keys sind oft 16, 24 oder 32 Bytes lang
    for key_len in [16, 24, 32]:
        for i in range(0, len(data) - key_len, 16):  # In 16-Byte-Schritten
            candidate = data[i:i+key_len]
            entropy = calculate_simple_entropy(candidate)
            
            # Hohe Entropie + nicht alle Bytes gleich
            if entropy > 6.5 and len(set(candidate)) > key_len // 2:
                # Prüfe ob es nicht nur zufällige Daten sind
                # (z.B. nicht alle Bytes sehr ähnlich)
                byte_variance = max(candidate) - min(candidate)
                if byte_variance > 50:  # Gute Verteilung
                    results.append({
                        'type': f'key_candidate_{key_len}',
                        'offset': i,
                        'key_hex': candidate.hex(),
                        'entropy': entropy
                    })
    
    return results


def main():
    """Hauptfunktion."""
    # Finde App-Bundle
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
    
    logger.info(f"Analysiere: {app_bundle}")
    
    # Bekannter BAL-Header
    bal_header = bytes.fromhex('1df93efb1a2c50731916a8e484d9d6fc')
    
    # Durchsuche wichtige Dateien
    search_paths = [
        app_bundle / "Contents" / "MacOS" / "Soundvision",  # Haupt-Binary
        app_bundle / "Contents" / "Frameworks",  # Frameworks
        app_bundle / "Contents" / "Resources",  # Resources
    ]
    
    all_results = []
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        logger.info(f"\nDurchsuche: {search_path}")
        
        # Durchsuche Binärdateien
        for file_path in search_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Überspringe sehr große Dateien
            try:
                if file_path.stat().st_size > 50 * 1024 * 1024:  # > 50MB
                    continue
            except:
                continue
            
            # Analysiere nur Binärdateien
            if file_path.suffix in ['.dylib', '', '.so', '.bin'] or 'Soundvision' in file_path.name:
                logger.info(f"  Analysiere: {file_path.name} ({file_path.stat().st_size:,} Bytes)")
                results = search_binary_for_patterns(file_path)
                if results:
                    for result in results:
                        result['file'] = str(file_path)
                        all_results.append(result)
                        print(f"    → {result['type']} @ offset {result.get('offset', 'N/A')}")
                        if 'key_hex' in result:
                            print(f"      Key: {result['key_hex']}")
    
    # Speichere Ergebnisse
    output_file = Path(__file__).parent / "soundvision_key_search.txt"
    with open(output_file, 'w') as f:
        f.write("=== SoundVision Schlüssel-Suche ===\n\n")
        for result in all_results:
            f.write(f"Datei: {result.get('file', 'N/A')}\n")
            f.write(f"Typ: {result['type']}\n")
            if 'offset' in result:
                f.write(f"Offset: {result['offset']}\n")
            if 'key_hex' in result:
                f.write(f"Key: {result['key_hex']}\n")
            if 'entropy' in result:
                f.write(f"Entropie: {result['entropy']:.2f}\n")
            f.write("\n")
    
    logger.info(f"\nErgebnisse gespeichert in: {output_file}")
    logger.info(f"Gefunden: {len(all_results)} Kandidaten")


if __name__ == "__main__":
    main()

