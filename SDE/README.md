# SDE-Datei Loader

Dieses Modul lädt und analysiert SDE-Dateien (SoundPLAN/ArrayCalc Projektdateien).

## Was sind SDE-Dateien?

SDE-Dateien sind ZIP-Archive, die folgende Inhalte enthalten:
- **OBJ/MTL-Dateien**: 3D-Modelle der Oberflächen (Back, Floor, Left, Right, etc.)
- **XML-Dateien**: 
  - `manifest.xml`: Metadaten über die Datei und Verschlüsselung
  - `SoundSystem.xml`: Lautsprecher-Konfiguration
- **BAL-Dateien**: Verschlüsselte Balloon-Daten (Richtcharakteristik-Daten)
- **SoundPLAN.enc**: Verschlüsselungsschlüssel für die BAL-Dateien

## Verwendung

### Grundlegende Verwendung

```python
from sde_loader import SDELoader

# Lade SDE-Datei
loader = SDELoader("sde.sde")
loader.load()

# Zeige Informationen
manifest_info = loader.get_manifest_info()
print(f"Product: {manifest_info['product']}")
print(f"Version: {manifest_info['version']}")

sound_info = loader.get_sound_system_info()
print(f"Kommentare: {sound_info['comments']}")

# Analysiere BAL-Dateien
for bal_file in loader.bal_files:
    analysis = loader.analyze_bal_file(bal_file)
    print(f"{analysis['file']}: {analysis['size']} Bytes")
```

### Kommandozeile

```bash
python3 sde_loader.py sde.sde
```

## BAL-Dateien Entschlüsselung

Die BAL-Dateien sind verschlüsselt und müssen entschlüsselt werden, um die Balloon-Daten zu verwenden.

### Aktueller Status

Die Entschlüsselung ist **noch nicht vollständig implementiert**. Das Skript versucht verschiedene Methoden:

1. **XOR-Entschlüsselung**: Einfache XOR-Verschlüsselung mit zyklischem Schlüssel
2. **AES-Entschlüsselung**: AES-Verschlüsselung (benötigt `pycryptodome`)

### Installation von Abhängigkeiten

Für AES-Entschlüsselung:
```bash
pip install pycryptodome
```

### Bekannte Probleme

- Die aktuelle XOR-Entschlüsselung funktioniert nicht korrekt
- AES-Entschlüsselung muss noch getestet werden
- Die genaue Verschlüsselungsmethode von dbaudio/ArrayCalc ist nicht dokumentiert

### Nächste Schritte

1. **Verschlüsselungsanalyse**: 
   - Analysiere die Struktur der verschlüsselten BAL-Dateien
   - Vergleiche mit bekannten Formaten
   - Teste verschiedene Entschlüsselungsalgorithmen

2. **Schlüsselanalyse**:
   - Analysiere die `SoundPLAN.enc` Datei genauer
   - Verstehe wie der Schlüssel verwendet wird
   - Möglicherweise ist der Schlüssel selbst verschlüsselt

3. **Format-Dokumentation**:
   - Dokumentiere das Format der entschlüsselten BAL-Dateien
   - Erstelle Parser für die Balloon-Daten

## Dateistruktur

Nach dem Laden werden alle Dateien in den `SDE/` Ordner extrahiert:

```
SDE/
├── Back.obj
├── Back.mtl
├── Floor.obj
├── Floor.mtl
├── Left.obj
├── Left.mtl
├── Right.obj
├── Right.mtl
├── Left Corner.obj
├── Left Corner.mtl
├── Right Corner.obj
├── Right Corner.mtl
├── manifest.xml
├── SoundSystem.xml
├── SoundPLAN.enc
├── V8.bal
├── V12.bal
├── V-SUB.bal
└── Y10P_110x40.bal
```

## API-Referenz

### SDELoader

#### `__init__(sde_file_path, output_dir=None)`
Erstellt einen neuen SDELoader.

#### `load() -> bool`
Lädt die SDE-Datei und extrahiert alle Dateien.

#### `get_manifest_info() -> Dict`
Gibt Informationen aus dem Manifest zurück.

#### `get_sound_system_info() -> Dict`
Gibt Informationen aus SoundSystem.xml zurück.

#### `analyze_bal_file(bal_file_path) -> Dict`
Analysiert eine BAL-Datei und gibt Informationen zurück.

#### `decrypt_bal_file(bal_file_path, output_path=None, method="auto") -> bool`
Versucht, eine BAL-Datei zu entschlüsseln.

#### `decrypt_all_bal_files() -> Dict[str, bool]`
Versucht, alle BAL-Dateien zu entschlüsseln.

## Hinweise

- Die Entschlüsselung der BAL-Dateien erfordert möglicherweise proprietäre Informationen von dbaudio
- Ohne die korrekte Entschlüsselungsmethode können die Balloon-Daten nicht verwendet werden
- Die OBJ/MTL-Dateien können direkt verwendet werden (nicht verschlüsselt)

