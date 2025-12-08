# SoundVision Analyse - Verschlüsselung und BAL-Dateien

## Zusammenfassung

SoundVision verwendet ebenfalls verschlüsselte BAL-Dateien, aber mit einem **anderen Format** als ArrayCalc.

## Erkenntnisse

### 1. BAL-Dateien

- **Anzahl**: 105 BAL-Dateien in `/Applications/Soundvision.app/Contents/Resources/balloons/`
- **Größe**: Meist 339.536 Bytes (einige größer, z.B. 5XT_Coaxial.bal: 660.704 Bytes)
- **Verschlüsselung**: Ja (Entropie ~7.8, ähnlich wie ArrayCalc)
- **Format**: Binärdaten, nicht ZIP oder XML

### 2. Header-Analyse

**SoundVision BAL-Header** (erste 16 Bytes):
```
1d f9 3e fb 1a 2c 50 73 19 16 a8 e4 84 d9 d6 fc
```

**ArrayCalc BAL-Header** (erste 16 Bytes):
```
66 ad 28 1e bc 34 e8 b0 34 e2 c2 a7 cd 75 56 43
```

**Ergebnis**: **Unterschiedliche Header** → Wahrscheinlich **verschiedene Verschlüsselungsmethoden** oder **verschiedene Schlüssel**

### 3. Verschlüsselungsfunktionen

Gefundene Komponenten:
- `CipheringHandler` - Verschlüsselungs-Handler
- `ICipheringHandler` - Interface für Verschlüsselung
- `AESCipher` - AES-Verschlüsselung
- `readEncryptedProject` - Liest verschlüsselte Projekte
- `decrypt` - Entschlüsselungsfunktion
- `encrypt` - Verschlüsselungsfunktion

**OpenSSL EVP-Funktionen**:
- `EVP_CIPHER_CTX_new` - OpenSSL Cipher Context
- Ähnlich wie ArrayCalc

### 4. RSA-Schlüssel

- **Öffentlicher Schlüssel**: Gefunden, aber **leer** (nur BEGIN/END Marker)
- **Privater Schlüssel**: **Nicht gefunden**

### 5. Vergleich mit ArrayCalc

| Eigenschaft | SoundVision | ArrayCalc |
|------------|-------------|-----------|
| BAL-Verschlüsselung | ✅ Ja | ✅ Ja |
| Entropie | ~7.8 | ~7.79 |
| Header | `1df93efb...` | `66ad281e...` |
| Verschlüsselung | AES (vermutlich) | AES (vermutlich) |
| RSA-Schlüssel | Öffentlich (leer) | Nicht gefunden |
| OpenSSL EVP | ✅ Ja | ✅ Ja |

## Schlussfolgerung

1. **Verschiedene Verschlüsselung**: SoundVision und ArrayCalc verwenden **verschiedene Verschlüsselungsmethoden** oder **verschiedene Schlüssel**
2. **Kein gemeinsamer Schlüssel**: Die unterschiedlichen Header deuten darauf hin, dass sie nicht kompatibel sind
3. **Ähnliche Technologie**: Beide verwenden OpenSSL EVP und AES-Verschlüsselung
4. **Kein privater Schlüssel**: Weder in SoundVision noch in ArrayCalc wurde ein privater RSA-Schlüssel gefunden

## Nächste Schritte

1. **Runtime-Analyse**: Verschlüsselungsfunktionen zur Laufzeit analysieren
2. **Detaillierte Binäranalyse**: CipheringHandler genauer untersuchen
3. **Format-Analyse**: BAL-Datei-Format von SoundVision dokumentieren

## Dateien

- `analyze_soundvision.py` - Analyse-Skript
- `extract_soundvision_keys.py` - RSA-Schlüssel-Extraktion
- `soundvision_public_key_1.pem` - Extrahieter öffentlicher Schlüssel (leer)

