# SoundVision BAL-Dateien - Zusammenfassung

## ✅ Erfolgreich kopiert

**105 BAL-Dateien** von SoundVision wurden kopiert nach:
```
/Users/MGraf/Python/LFO_Umgebung/SDE/soundvision_balloons/
```

**Gesamtgröße**: 36.14 MB

## 📊 Datei-Statistiken

- **Meiste Dateien**: 339.536 Bytes (332 KB) - Standard-Größe
- **Größte Datei**: 5XT_Coaxial.bal - 660.704 Bytes (645 KB)
- **Anzahl**: 105 Dateien

## 🔐 Verschlüsselungsstatus

### Verschlüsselt: ✅ JA

- **Entropie**: ~7.8 (sehr hoch = verschlüsselt)
- **Format**: Binärdaten (nicht ZIP, nicht XML)
- **Header**: `1df93efb1a2c50731916a8e484d9d6fc` (erste 16 Bytes)
- **Verschlüsselung**: Wahrscheinlich AES (OpenSSL EVP)

### Unterschied zu ArrayCalc

| Eigenschaft | SoundVision | ArrayCalc |
|------------|-------------|-----------|
| Header | `1df93efb...` | `66ad281e...` |
| Größe (typisch) | 339.536 Bytes | 1.038.160 Bytes |
| Entropie | ~7.8 | ~7.79 |
| Verschlüsselung | AES (vermutlich) | AES (vermutlich) |
| **Kompatibel** | ❌ **Nein** | - |

**Wichtig**: SoundVision und ArrayCalc verwenden **verschiedene Verschlüsselungen**!

## 📁 Datei-Organisation

Die Dateien sind nach Lautsprecher-Modellen organisiert:

- **A10-Serie**: A10FOCUS, A10WIDE, A10iFOCUS, A10iWIDE
- **A15-Serie**: A15FOCUS, A15WIDE
- **E1027/E1028**: Verschiedene Richtcharakteristiken
- **K1/K2**: K-Serie Lautsprecher
- **SB28**: Subwoofer
- **5XT**: Coaxial-Lautsprecher

## 🔍 Nächste Schritte

### 1. Entschlüsselung verstehen
- Verschlüsselungsfunktionen in SoundVision analysieren
- CipheringHandler genauer untersuchen
- Möglicherweise Runtime-Analyse mit Frida

### 2. Format-Dokumentation
- Struktur der entschlüsselten BAL-Dateien dokumentieren
- Balloon-Daten-Format verstehen
- Import in LFO-System vorbereiten

### 3. Import-Tool entwickeln
- Entschlüsselungs-Tool für SoundVision BAL-Dateien
- Konvertierung zu LFO-Format
- Integration in LFO-System

## 📝 Verfügbare Tools

1. **`analyze_soundvision_bal.py`** - Detaillierte Analyse der BAL-Dateien
2. **`copy_soundvision_bal.py`** - Kopiert alle BAL-Dateien
3. **`extract_soundvision_keys.py`** - Extrahiert RSA-Schlüssel
4. **`analyze_soundvision.py`** - Allgemeine SoundVision-Analyse

## 📂 Verzeichnisstruktur

```
SDE/soundvision_balloons/
├── README.md              # Dokumentation
├── metadata.json          # Metadaten (JSON)
├── *.bal                  # 105 BAL-Dateien
└── ...
```

## ⚠️ Wichtige Hinweise

1. **Verschlüsselt**: Alle Dateien sind verschlüsselt und können nicht direkt gelesen werden
2. **Nicht kompatibel**: SoundVision BAL ≠ ArrayCalc BAL (verschiedene Verschlüsselung)
3. **Rechtlich**: Siehe `LEGAL_NOTICE.md` für rechtliche Hinweise
4. **Entschlüsselung**: Erfordert privaten RSA-Schlüssel oder Runtime-Analyse

## 🎯 Verwendungszweck

Diese BAL-Dateien können verwendet werden für:
- ✅ Analyse der Verschlüsselungsmethode
- ✅ Vergleich mit ArrayCalc BAL-Dateien
- ✅ Entwicklung von Entschlüsselungs-Tools
- ✅ Import in LFO-System (nach Entschlüsselung)
- ✅ Dokumentation des Balloon-Daten-Formats

