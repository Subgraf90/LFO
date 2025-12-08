"""
Detaillierte Analyse der SoundVision BAL-Dateien

Analysiert das Format, die Struktur und versucht die Daten zu verstehen.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def analyze_bal_structure(bal_path: Path) -> Dict:
    """
    Analysiert die Struktur einer SoundVision BAL-Datei.
    """
    with open(bal_path, 'rb') as f:
        data = f.read()
    
    size = len(data)
    
    analysis = {
        'file': bal_path.name,
        'size': size,
        'header': data[:64].hex(),
        'structure': {},
    }
    
    # Analysiere Header
    header = data[:32]
    analysis['header_bytes'] = list(header)
    analysis['header_hex'] = header.hex()
    
    # Prüfe auf bekannte Muster
    # Möglicherweise: Größe, Anzahl von Datenpunkten, etc.
    
    # Versuche verschiedene Interpretationen
    # 1. Erste 4 Bytes als Integer (Anzahl von Datenpunkten?)
    if size >= 4:
        first_int = struct.unpack('>I', data[:4])[0]  # Big-endian
        first_int_le = struct.unpack('<I', data[:4])[0]  # Little-endian
        analysis['first_4_bytes_as_int_be'] = first_int
        analysis['first_4_bytes_as_int_le'] = first_int_le
        
        # Prüfe ob sinnvoll (nicht zu groß)
        if 0 < first_int < size:
            analysis['possible_count_be'] = first_int
        if 0 < first_int_le < size:
            analysis['possible_count_le'] = first_int_le
    
    # 2. Prüfe auf wiederkehrende Muster (könnte auf strukturierte Daten hinweisen)
    # Suche nach häufigen Byte-Sequenzen
    byte_freq = {}
    for i in range(len(data) - 3):
        seq = data[i:i+4]
        byte_freq[seq] = byte_freq.get(seq, 0) + 1
    
    # Finde häufigste Sequenzen
    sorted_freq = sorted(byte_freq.items(), key=lambda x: x[1], reverse=True)
    analysis['common_sequences'] = [
        {'bytes': seq.hex(), 'count': count}
        for seq, count in sorted_freq[:10]
    ]
    
    # 3. Prüfe auf mögliche Frequenzdaten
    # Balloon-Daten enthalten typischerweise:
    # - Frequenzen
    # - Winkel (horizontal, vertikal)
    # - Magnitude/Phase pro Frequenz/Winkel-Kombination
    
    # Versuche verschiedene Datenstrukturen zu erkennen
    # Mögliche Struktur: Header + Datenblöcke
    
    # 4. Analysiere Datenverteilung
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1
    
    analysis['byte_distribution'] = {
        'min': min(byte_counts),
        'max': max(byte_counts),
        'mean': sum(byte_counts) / len(byte_counts),
        'zeros': byte_counts[0],
    }
    
    # 5. Prüfe auf mögliche Float/Double Arrays
    # Balloon-Daten könnten als Float-Arrays gespeichert sein
    if size % 4 == 0:
        # Könnte Float-Array sein
        try:
            floats = struct.unpack(f'>{size//4}f', data)
            analysis['possible_float_count'] = len(floats)
            analysis['float_range'] = {
                'min': min(floats),
                'max': max(floats),
                'mean': sum(floats) / len(floats)
            }
        except:
            pass
    
    if size % 8 == 0:
        # Könnte Double-Array sein
        try:
            doubles = struct.unpack(f'>{size//8}d', data)
            analysis['possible_double_count'] = len(doubles)
            analysis['double_range'] = {
                'min': min(doubles),
                'max': max(doubles),
                'mean': sum(doubles) / len(doubles)
            }
        except:
            pass
    
    return analysis


def try_decrypt_bal(bal_path: Path, methods: List[str] = None) -> Dict:
    """
    Versucht, eine SoundVision BAL-Datei zu entschlüsseln.
    """
    if methods is None:
        methods = ['none', 'xor', 'aes']
    
    with open(bal_path, 'rb') as f:
        encrypted_data = f.read()
    
    results = {}
    
    for method in methods:
        if method == 'none':
            # Keine Entschlüsselung - prüfe ob bereits unverschlüsselt
            results['none'] = {
                'success': False,
                'data': encrypted_data[:100],
                'note': 'Original-Daten'
            }
        
        elif method == 'xor':
            # Versuche einfache XOR-Entschlüsselung
            # Teste verschiedene Schlüssel-Längen
            for key_len in [1, 4, 8, 16, 32]:
                # Versuche häufige Bytes als Schlüssel
                common_bytes = [0x00, 0xFF, 0xAA, 0x55]
                for key_byte in common_bytes:
                    key = bytes([key_byte] * key_len)
                    decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted_data))
                    
                    # Prüfe ob Ergebnis sinnvoll ist
                    if is_likely_decrypted(decrypted):
                        results[f'xor_{key_len}_{key_byte:02x}'] = {
                            'success': True,
                            'data': decrypted[:100],
                            'key': key.hex()
                        }
        
        elif method == 'aes':
            # Versuche AES-Entschlüsselung
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import unpad
                
                # Teste verschiedene Schlüssel/IV-Kombinationen
                # Modus 1: IV in ersten 16 Bytes
                if len(encrypted_data) >= 32:
                    iv = encrypted_data[:16]
                    ciphertext = encrypted_data[16:]
                    
                    # Teste verschiedene Schlüssellängen
                    for key_len in [16, 24, 32]:
                        # Versuche häufige Schlüssel
                        test_keys = [
                            b'\x00' * key_len,
                            b'\xFF' * key_len,
                            encrypted_data[16:16+key_len] if len(encrypted_data) >= 16+key_len else None
                        ]
                        
                        for key in test_keys:
                            if key is None:
                                continue
                            try:
                                cipher = AES.new(key, AES.MODE_CBC, iv)
                                decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
                                if is_likely_decrypted(decrypted):
                                    results[f'aes_cbc_{key_len}'] = {
                                        'success': True,
                                        'data': decrypted[:100],
                                        'key': key.hex()[:32]
                                    }
                            except:
                                pass
            except ImportError:
                results['aes'] = {'success': False, 'error': 'pycryptodome nicht installiert'}
    
    return results


def is_likely_decrypted(data: bytes) -> bool:
    """Prüft ob Daten wahrscheinlich entschlüsselt sind"""
    if len(data) == 0:
        return False
    
    # Prüfe auf bekannte Formate
    if data.startswith(b'PK'):  # ZIP
        return True
    if data.startswith(b'<?xml') or data.startswith(b'<'):
        return True
    
    # Prüfe auf zu viele Null-Bytes (deutet auf fehlgeschlagene Entschlüsselung)
    null_count = sum(1 for b in data[:100] if b == 0)
    if null_count > 50:
        return False
    
    # Prüfe Entropie
    entropy = calculate_entropy(data[:min(1000, len(data))])
    if entropy < 6.0:  # Niedrige Entropie = strukturierte Daten
        return True
    
    return False


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


def extract_balloon_data(bal_path: Path) -> Optional[Dict]:
    """
    Versucht, Balloon-Daten aus einer BAL-Datei zu extrahieren.
    """
    with open(bal_path, 'rb') as f:
        data = f.read()
    
    # Versuche verschiedene Interpretationen
    results = {}
    
    # 1. Als Float-Array interpretieren
    if len(data) % 4 == 0:
        try:
            floats = np.frombuffer(data, dtype=np.float32)
            results['as_float32'] = {
                'count': len(floats),
                'min': float(np.min(floats)),
                'max': float(np.max(floats)),
                'mean': float(np.mean(floats)),
                'std': float(np.std(floats)),
                'first_10': floats[:10].tolist()
            }
        except:
            pass
    
    # 2. Als Double-Array interpretieren
    if len(data) % 8 == 0:
        try:
            doubles = np.frombuffer(data, dtype=np.float64)
            results['as_float64'] = {
                'count': len(doubles),
                'min': float(np.min(doubles)),
                'max': float(np.max(doubles)),
                'mean': float(np.mean(doubles)),
                'std': float(np.std(doubles)),
                'first_10': doubles[:10].tolist()
            }
        except:
            pass
    
    return results if results else None


def main():
    """Hauptfunktion"""
    balloons_dir = Path("/Applications/Soundvision.app/Contents/Resources/balloons")
    
    if not balloons_dir.exists():
        print("Balloons-Verzeichnis nicht gefunden!")
        return
    
    bal_files = list(balloons_dir.glob("*.bal"))
    print(f"=== SoundVision BAL-Dateien Analyse ===\n")
    print(f"Gefundene BAL-Dateien: {len(bal_files)}\n")
    
    # Analysiere erste 5 Dateien detailliert
    for bal_file in bal_files[:5]:
        print(f"--- {bal_file.name} ---")
        
        # Struktur-Analyse
        structure = analyze_bal_structure(bal_file)
        print(f"Größe: {structure['size']} Bytes")
        print(f"Header (hex): {structure['header_hex'][:64]}...")
        
        if 'possible_count_be' in structure:
            print(f"Mögliche Datenpunkt-Anzahl (BE): {structure['possible_count_be']}")
        if 'possible_count_le' in structure:
            print(f"Mögliche Datenpunkt-Anzahl (LE): {structure['possible_count_le']}")
        
        # Versuche Daten zu extrahieren
        balloon_data = extract_balloon_data(bal_file)
        if balloon_data:
            print("\nMögliche Daten-Interpretation:")
            for data_type, info in balloon_data.items():
                print(f"  {data_type}:")
                print(f"    Anzahl: {info['count']}")
                print(f"    Min: {info['min']:.6f}, Max: {info['max']:.6f}")
                print(f"    Mittelwert: {info['mean']:.6f}, Std: {info['std']:.6f}")
                print(f"    Erste 10 Werte: {[f'{v:.3f}' for v in info['first_10']]}")
        
        print()
    
    # Versuche Entschlüsselung bei einer Datei
    print("\n=== Versuche Entschlüsselung ===")
    test_file = bal_files[0]
    print(f"Teste: {test_file.name}")
    
    decrypt_results = try_decrypt_bal(test_file)
    if decrypt_results:
        for method, result in decrypt_results.items():
            if result.get('success'):
                print(f"  ✓ {method}: Erfolgreich!")
                print(f"    Daten (hex): {result['data'].hex()[:64]}...")
            else:
                print(f"  ✗ {method}: {result.get('error', 'Fehlgeschlagen')}")
    
    print("\n=== Analyse abgeschlossen ===")


if __name__ == "__main__":
    main()

