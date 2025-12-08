"""
Analyse von SoundVision für BAL-Dateien und Verschlüsselung

SoundVision verwendet auch .bal-Dateien. Diese könnten:
1. Unverschlüsselt sein
2. Ein anderes Format haben
3. Den gleichen Verschlüsselungsschlüssel verwenden
"""

import os
from pathlib import Path
import struct


def analyze_bal_file(bal_path: Path):
    """Analysiert eine BAL-Datei von SoundVision"""
    if not bal_path.exists():
        return None
    
    with open(bal_path, 'rb') as f:
        data = f.read()
    
    size = len(data)
    header = data[:100]
    
    analysis = {
        'file': bal_path.name,
        'size': size,
        'header_hex': header.hex()[:80],
        'header_bytes': list(header[:20]),
        'is_text': data[:100].decode('ascii', errors='ignore').isprintable(),
        'file_type': 'unknown'
    }
    
    # Prüfe Dateiformat
    if data.startswith(b'PK'):
        analysis['file_type'] = 'zip'
    elif data.startswith(b'<?xml') or data.startswith(b'<'):
        analysis['file_type'] = 'xml'
    elif data[:4] == b'\x00\x00\x00\x00' or all(b < 128 for b in data[:20]):
        analysis['file_type'] = 'binary_data'
    elif data[:100].decode('ascii', errors='ignore').isprintable():
        analysis['file_type'] = 'text'
    
    # Prüfe auf Verschlüsselung (hohe Entropie)
    entropy = calculate_entropy(data[:min(1000, len(data))])
    analysis['entropy'] = entropy
    analysis['is_likely_encrypted'] = entropy > 7.0
    
    return analysis


def calculate_entropy(data: bytes) -> float:
    """Berechnet Shannon-Entropie"""
    import math
    if len(data) == 0:
        return 0
    
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1
    
    entropy = 0
    for count in byte_counts:
        if count > 0:
            p = count / len(data)
            entropy -= p * math.log2(p)
    
    return entropy


def find_encryption_functions(binary_path: Path):
    """Sucht nach Verschlüsselungsfunktionen im Binary"""
    if not binary_path.exists():
        return []
    
    with open(binary_path, 'rb') as f:
        data = f.read()
    
    # Suche nach Strings, die auf Verschlüsselung hinweisen
    encryption_strings = [
        b'encrypt',
        b'decrypt',
        b'cipher',
        b'AES',
        b'RSA',
        b'EVP_',
        b'Crypto',
        b'OpenSSL',
        b'balloon',
        b'.bal',
        b'loadBalloon',
        b'readBalloon',
    ]
    
    found = []
    for enc_str in encryption_strings:
        matches = list(data.find(enc_str) for _ in range(10))  # Finde erste 10
        matches = [m for m in matches if m != -1]
        if matches:
            found.append({
                'string': enc_str.decode('ascii', errors='ignore'),
                'count': len(matches),
                'positions': matches[:5]
            })
    
    return found


def compare_with_arraycalc_bal(soundvision_bal: Path, arraycalc_bal: Path):
    """Vergleicht SoundVision BAL mit ArrayCalc BAL"""
    if not soundvision_bal.exists() or not arraycalc_bal.exists():
        return None
    
    with open(soundvision_bal, 'rb') as f:
        sv_data = f.read()
    
    with open(arraycalc_bal, 'rb') as f:
        ac_data = f.read()
    
    comparison = {
        'sv_size': len(sv_data),
        'ac_size': len(ac_data),
        'same_size': len(sv_data) == len(ac_data),
        'same_header': sv_data[:20] == ac_data[:20],
        'sv_entropy': calculate_entropy(sv_data[:min(1000, len(sv_data))]),
        'ac_entropy': calculate_entropy(ac_data[:min(1000, len(ac_data))]),
    }
    
    return comparison


def main():
    """Hauptfunktion"""
    soundvision_app = Path("/Applications/Soundvision.app")
    balloons_dir = soundvision_app / "Contents/Resources/balloons"
    binary_path = soundvision_app / "Contents/MacOS/Soundvision"
    
    print("=== SoundVision Analyse ===\n")
    
    # 1. Analysiere BAL-Dateien
    print("1. Analysiere SoundVision BAL-Dateien...")
    if balloons_dir.exists():
        bal_files = list(balloons_dir.glob("*.bal"))
        print(f"   Gefunden: {len(bal_files)} BAL-Dateien\n")
        
        for bal_file in bal_files[:10]:  # Erste 10 analysieren
            analysis = analyze_bal_file(bal_file)
            if analysis:
                print(f"   {analysis['file']}:")
                print(f"     Größe: {analysis['size']} Bytes")
                print(f"     Typ: {analysis['file_type']}")
                print(f"     Entropie: {analysis['entropy']:.2f}")
                print(f"     Wahrscheinlich verschlüsselt: {analysis['is_likely_encrypted']}")
                print(f"     Header (hex): {analysis['header_hex']}...")
                print()
    else:
        print("   ✗ Balloons-Verzeichnis nicht gefunden")
    
    # 2. Suche nach Verschlüsselungsfunktionen
    print("\n2. Suche nach Verschlüsselungsfunktionen...")
    if binary_path.exists():
        functions = find_encryption_functions(binary_path)
        if functions:
            print(f"   Gefunden: {len(functions)} Verschlüsselungs-Hinweise")
            for func in functions:
                print(f"     - {func['string']}: {func['count']} mal")
        else:
            print("   ✗ Keine Verschlüsselungsfunktionen gefunden")
    else:
        print("   ✗ Binary nicht gefunden")
    
    # 3. Vergleiche mit ArrayCalc BAL
    print("\n3. Vergleiche mit ArrayCalc BAL-Dateien...")
    arraycalc_bal = Path("/Users/MGraf/Python/LFO_Umgebung/SDE/V8.bal")
    if balloons_dir.exists() and arraycalc_bal.exists():
        sv_bal = list(balloons_dir.glob("*.bal"))[0] if list(balloons_dir.glob("*.bal")) else None
        if sv_bal:
            comparison = compare_with_arraycalc_bal(sv_bal, arraycalc_bal)
            if comparison:
                print(f"   SoundVision BAL: {comparison['sv_size']} Bytes, Entropie {comparison['sv_entropy']:.2f}")
                print(f"   ArrayCalc BAL: {comparison['ac_size']} Bytes, Entropie {comparison['ac_entropy']:.2f}")
                print(f"   Gleiche Größe: {comparison['same_size']}")
                print(f"   Gleicher Header: {comparison['same_header']}")
    
    # 4. Suche nach RSA-Schlüsseln
    print("\n4. Suche nach RSA-Schlüsseln...")
    if binary_path.exists():
        with open(binary_path, 'rb') as f:
            data = f.read()
        
        # Suche nach PEM-format RSA-Schlüsseln
        pem_patterns = [
            b'-----BEGIN RSA PRIVATE KEY-----',
            b'-----BEGIN PRIVATE KEY-----',
            b'-----BEGIN RSA PUBLIC KEY-----',
            b'-----BEGIN PUBLIC KEY-----',
        ]
        
        found_keys = []
        for pattern in pem_patterns:
            if pattern in data:
                found_keys.append(pattern.decode('ascii', errors='ignore'))
        
        if found_keys:
            print(f"   ✓ Gefunden: {len(found_keys)} RSA-Schlüssel-Hinweise")
            for key in found_keys:
                print(f"     - {key}")
        else:
            print("   ✗ Keine RSA-Schlüssel im PEM-Format gefunden")
    
    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()

