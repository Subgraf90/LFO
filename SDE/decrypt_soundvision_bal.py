#!/usr/bin/env python3
"""
Versucht SoundVision BAL-Dateien zu entschlüsseln.

Da alle BAL-Dateien den gleichen Header haben (1df93efb1a2c50731916a8e484d9d6fc),
könnte dieser Teil des Verschlüsselungsschemas sein.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def try_decrypt_aes_cbc(data: bytes, key: bytes, iv: Optional[bytes] = None) -> Optional[bytes]:
    """Versucht AES-CBC Entschlüsselung."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        
        if iv is None:
            # Verwende ersten 16 Bytes als IV
            iv = data[:16]
            ciphertext = data[16:]
        else:
            ciphertext = data
        
        if len(key) not in [16, 24, 32]:
            # Pad key auf 32 Bytes
            key = key[:32] + b'\x00' * (32 - len(key))
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted
    except Exception as e:
        return None


def try_decrypt_aes_ecb(data: bytes, key: bytes) -> Optional[bytes]:
    """Versucht AES-ECB Entschlüsselung."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        
        if len(key) not in [16, 24, 32]:
            key = key[:32] + b'\x00' * (32 - len(key))
        
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted = unpad(cipher.decrypt(data), AES.block_size)
        return decrypted
    except Exception as e:
        return None


def try_xor_decrypt(data: bytes, key: bytes) -> bytes:
    """XOR-Entschlüsselung."""
    result = bytearray(data)
    key_len = len(key)
    if key_len == 0:
        return bytes(result)
    
    for i in range(len(result)):
        result[i] ^= key[i % key_len]
    
    return bytes(result)


def is_likely_decrypted(data: bytes) -> bool:
    """Prüft ob Daten wahrscheinlich entschlüsselt sind."""
    if not data or len(data) < 16:
        return False
    
    # Prüfe auf bekannte Header
    if data[:4] == b'PK\x03\x04':  # ZIP
        return True
    if data[:2] == b'\x1f\x8b':  # GZIP
        return True
    if data[:4] == b'\x89PNG':  # PNG
        return True
    
    # Prüfe auf viele Null-Bytes (typisch für unverschlüsselte Binärdaten)
    null_ratio = data.count(b'\x00') / len(data)
    if null_ratio > 0.1:  # > 10% Null-Bytes
        return True
    
    # Prüfe auf niedrige Entropie (verschlüsselte Daten haben hohe Entropie)
    from collections import Counter
    import math
    counts = Counter(data[:1024])  # Erste 1KB
    entropy = 0.0
    for count in counts.values():
        p = count / min(1024, len(data))
        if p > 0:
            entropy -= p * math.log2(p)
    
    if entropy < 6.0:  # Niedrige Entropie = wahrscheinlich entschlüsselt
        return True
    
    return False


def test_decryption_methods(bal_path: Path):
    """Testet verschiedene Entschlüsselungsmethoden."""
    logger.info(f"Teste Entschlüsselung für: {bal_path.name}")
    
    with open(bal_path, 'rb') as f:
        data = f.read()
    
    # Bekannter Header
    known_header = bytes.fromhex('1df93efb1a2c50731916a8e484d9d6fc')
    
    # Methode 1: Header als IV verwenden, Rest als Key-Kandidat
    logger.info("  Methode 1: Header als IV, verschiedene Key-Längen")
    for key_len in [16, 24, 32]:
        # Versuche verschiedene Teile der Datei als Key
        for offset in [0, 16, 32, 64]:
            if offset + key_len < len(data):
                key_candidate = data[offset:offset+key_len]
                iv = known_header[:16]
                ciphertext = data[16:]  # Nach Header
                
                decrypted = try_decrypt_aes_cbc(ciphertext, key_candidate, iv)
                if decrypted and is_likely_decrypted(decrypted):
                    logger.info(f"    ✓ Erfolg mit Key-Länge {key_len} @ Offset {offset}!")
                    return decrypted
    
    # Methode 2: Header als Key verwenden
    logger.info("  Methode 2: Header als Key")
    for key_len in [16, 24, 32]:
        key = known_header[:key_len]
        decrypted = try_decrypt_aes_cbc(data, key)
        if decrypted and is_likely_decrypted(decrypted):
            logger.info(f"    ✓ Erfolg mit Header als {key_len}-Byte Key!")
            return decrypted
    
    # Methode 3: XOR mit Header
    logger.info("  Methode 3: XOR mit Header")
    decrypted = try_xor_decrypt(data, known_header)
    if is_likely_decrypted(decrypted):
        logger.info("    ✓ Erfolg mit XOR!")
        return decrypted
    
    # Methode 4: XOR mit verschiedenen Key-Längen
    logger.info("  Methode 4: XOR mit verschiedenen Key-Längen")
    for key_len in [16, 24, 32]:
        key = data[:key_len]
        decrypted = try_xor_decrypt(data, key)
        if is_likely_decrypted(decrypted):
            logger.info(f"    ✓ Erfolg mit {key_len}-Byte XOR Key!")
            return decrypted
    
    # Methode 5: AES-ECB mit Header als Key
    logger.info("  Methode 5: AES-ECB mit Header")
    for key_len in [16, 24, 32]:
        key = known_header[:key_len]
        decrypted = try_decrypt_aes_ecb(data, key)
        if decrypted and is_likely_decrypted(decrypted):
            logger.info(f"    ✓ Erfolg mit AES-ECB!")
            return decrypted
    
    logger.info("  ✗ Keine Methode erfolgreich")
    return None


def main():
    """Hauptfunktion."""
    # Finde eine BAL-Datei zum Testen
    app_bundle = Path("/Users/MGraf/Desktop/soundvision_resigned/Soundvision.app")
    if not app_bundle.exists():
        app_bundle = Path("/Applications/Soundvision.app")
    
    if not app_bundle.exists():
        logger.error("SoundVision App-Bundle nicht gefunden!")
        return
    
    balloons_dir = app_bundle / "Contents" / "Resources" / "balloons"
    if not balloons_dir.exists():
        logger.error("Balloons-Verzeichnis nicht gefunden!")
        return
    
    # Teste mit einer kleinen BAL-Datei
    bal_files = list(balloons_dir.glob("*.bal"))
    if not bal_files:
        logger.error("Keine BAL-Dateien gefunden!")
        return
    
    # Teste mit der ersten Datei
    test_file = bal_files[0]
    logger.info(f"Teste mit: {test_file.name}")
    
    decrypted = test_decryption_methods(test_file)
    
    if decrypted:
        output_file = Path(__file__).parent / f"decrypted_{test_file.stem}.bin"
        with open(output_file, 'wb') as f:
            f.write(decrypted)
        logger.info(f"Entschlüsselte Daten gespeichert in: {output_file}")
        
        # Analysiere entschlüsselte Daten
        logger.info(f"\nEntschlüsselte Daten-Analyse:")
        logger.info(f"  Größe: {len(decrypted):,} Bytes")
        logger.info(f"  Header (Hex): {decrypted[:32].hex()}")
        logger.info(f"  Header (ASCII): {decrypted[:32]}")
    else:
        logger.info("Entschlüsselung nicht erfolgreich")
        logger.info("Hinweis: Der Schlüssel könnte dynamisch generiert werden")


if __name__ == "__main__":
    main()

