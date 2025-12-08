"""
Extrahiert RSA-Schlüssel und Verschlüsselungsinformationen aus SoundVision
"""

import re
from pathlib import Path
import base64


def extract_rsa_keys(binary_path: Path):
    """Extrahiert RSA-Schlüssel aus dem Binary"""
    with open(binary_path, 'rb') as f:
        data = f.read()
    
    keys = {
        'public_keys': [],
        'private_keys': [],
    }
    
    # Suche nach PEM-format RSA-Schlüsseln
    pem_patterns = [
        (b'-----BEGIN RSA PRIVATE KEY-----', b'-----END RSA PRIVATE KEY-----', 'private'),
        (b'-----BEGIN PRIVATE KEY-----', b'-----END PRIVATE KEY-----', 'private'),
        (b'-----BEGIN RSA PUBLIC KEY-----', b'-----END RSA PUBLIC KEY-----', 'public'),
        (b'-----BEGIN PUBLIC KEY-----', b'-----END PUBLIC KEY-----', 'public'),
    ]
    
    for start_pattern, end_pattern, key_type in pem_patterns:
        matches = list(re.finditer(start_pattern, data))
        for match in matches:
            start = match.start()
            end_match = re.search(end_pattern, data[start:])
            if end_match:
                key_data = data[start:start + end_match.end()]
                try:
                    key_text = key_data.decode('ascii')
                    keys[f'{key_type}_keys'].append({
                        'type': key_type,
                        'format': 'PEM',
                        'data': key_text,
                        'offset': start
                    })
                except:
                    pass
    
    return keys


def extract_ciphering_info(binary_path: Path):
    """Extrahiert Informationen über Verschlüsselungsfunktionen"""
    with open(binary_path, 'rb') as f:
        data = f.read()
    
    info = {
        'ciphering_handler': [],
        'aes_cipher': [],
        'decrypt_functions': [],
        'encrypt_functions': [],
    }
    
    # Suche nach relevanten Strings
    patterns = {
        'ciphering_handler': [
            b'CipheringHandler',
            b'ICipheringHandler',
            b'cipheringHandler',
        ],
        'aes_cipher': [
            b'AESCipher',
            b'AES',
        ],
        'decrypt_functions': [
            b'decrypt',
            b'readEncrypted',
            b'File decryption',
        ],
        'encrypt_functions': [
            b'encrypt',
            b'writeEncrypted',
        ],
    }
    
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = list(re.finditer(pattern, data))
            for match in matches[:5]:  # Erste 5
                # Extrahiere Kontext
                start = max(0, match.start() - 100)
                end = min(len(data), match.end() + 100)
                context = data[start:end]
                try:
                    context_str = context.decode('ascii', errors='ignore')
                    info[category].append({
                        'pattern': pattern.decode('ascii', errors='ignore'),
                        'offset': match.start(),
                        'context': context_str[:200]
                    })
                except:
                    pass
    
    return info


def analyze_bal_header(soundvision_bal: Path, arraycalc_bal: Path):
    """Vergleicht Header von SoundVision und ArrayCalc BAL-Dateien"""
    with open(soundvision_bal, 'rb') as f:
        sv_data = f.read()
    
    with open(arraycalc_bal, 'rb') as f:
        ac_data = f.read()
    
    # Erste 32 Bytes vergleichen
    sv_header = sv_data[:32]
    ac_header = ac_data[:32]
    
    # Prüfe ob gemeinsamer Header vorhanden
    common_prefix = 0
    for i in range(min(len(sv_header), len(ac_header))):
        if sv_header[i] == ac_header[i]:
            common_prefix += 1
        else:
            break
    
    return {
        'sv_header_hex': sv_header.hex(),
        'ac_header_hex': ac_header.hex(),
        'common_prefix': common_prefix,
        'sv_size': len(sv_data),
        'ac_size': len(ac_data),
    }


def main():
    """Hauptfunktion"""
    soundvision_app = Path("/Applications/Soundvision.app")
    binary_path = soundvision_app / "Contents/MacOS/Soundvision"
    balloons_dir = soundvision_app / "Contents/Resources/balloons"
    
    print("=== SoundVision RSA-Schlüssel Extraktion ===\n")
    
    # 1. Extrahiere RSA-Schlüssel
    print("1. Extrahiere RSA-Schlüssel...")
    if binary_path.exists():
        keys = extract_rsa_keys(binary_path)
        
        print(f"   Öffentliche Schlüssel: {len(keys['public_keys'])}")
        for i, key in enumerate(keys['public_keys'], 1):
            print(f"     Schlüssel {i}:")
            print(f"       Offset: {key['offset']}")
            print(f"       Format: {key['format']}")
            # Zeige ersten Teil
            key_lines = key['data'].split('\n')[:3]
            for line in key_lines:
                print(f"       {line}")
            print()
            
            # Speichere Schlüssel
            output_path = Path(f"/Users/MGraf/Python/LFO_Umgebung/SDE/soundvision_public_key_{i}.pem")
            with open(output_path, 'w') as f:
                f.write(key['data'])
            print(f"       Gespeichert: {output_path}\n")
        
        print(f"   Private Schlüssel: {len(keys['private_keys'])}")
        for i, key in enumerate(keys['private_keys'], 1):
            print(f"     Schlüssel {i}:")
            print(f"       Offset: {key['offset']}")
            print(f"       Format: {key['format']}")
            # Zeige ersten Teil (ohne vollständigen Schlüssel zu zeigen)
            key_lines = key['data'].split('\n')[:2]
            for line in key_lines:
                print(f"       {line}")
            print("       ...")
            print()
            
            # Speichere Schlüssel
            output_path = Path(f"/Users/MGraf/Python/LFO_Umgebung/SDE/soundvision_private_key_{i}.pem")
            with open(output_path, 'w') as f:
                f.write(key['data'])
            print(f"       Gespeichert: {output_path}\n")
    else:
        print("   ✗ Binary nicht gefunden")
    
    # 2. Extrahiere Verschlüsselungsinformationen
    print("\n2. Extrahiere Verschlüsselungsinformationen...")
    if binary_path.exists():
        cipher_info = extract_ciphering_info(binary_path)
        
        for category, items in cipher_info.items():
            if items:
                print(f"   {category}: {len(items)} Hinweise")
                for item in items[:3]:  # Erste 3
                    print(f"     - {item['pattern']} bei Offset {item['offset']}")
    
    # 3. Analysiere BAL-Header
    print("\n3. Analysiere BAL-Header...")
    if balloons_dir.exists() and Path("/Users/MGraf/Python/LFO_Umgebung/SDE/V8.bal").exists():
        sv_bal = list(balloons_dir.glob("*.bal"))[0]
        ac_bal = Path("/Users/MGraf/Python/LFO_Umgebung/SDE/V8.bal")
        
        if sv_bal and ac_bal.exists():
            header_analysis = analyze_bal_header(sv_bal, ac_bal)
            print(f"   SoundVision Header: {header_analysis['sv_header_hex'][:64]}...")
            print(f"   ArrayCalc Header: {header_analysis['ac_header_hex'][:64]}...")
            print(f"   Gemeinsamer Präfix: {header_analysis['common_prefix']} Bytes")
            
            if header_analysis['common_prefix'] > 0:
                print(f"   ⚠ Gemeinsamer Header gefunden! Möglicherweise gleiche Verschlüsselung.")
    
    print("\n=== Extraktion abgeschlossen ===")


if __name__ == "__main__":
    main()

