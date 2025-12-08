# Anleitung: Systematisches Testen von RSA-Kandidaten

## Übersicht

Das Skript `test_rsa_candidates_systematic.py` testet systematisch alle gefundenen RSA-Schlüssel-Kandidaten, um den privaten RSA-Schlüssel zu finden, der zum Entschlüsseln von SoundPLAN.enc benötigt wird.

## Features

- ✅ **Fortschrittsspeicherung**: Kann jederzeit gestoppt und fortgesetzt werden
- ✅ **Automatisches Speichern**: Speichert Fortschritt alle 100 Tests
- ✅ **Duplikat-Erkennung**: Überspringt bereits getestete Kandidaten
- ✅ **BAL-Test**: Testet erfolgreiche Schlüssel direkt mit BAL-Dateien
- ✅ **Statistiken**: Zeigt Fortschritt und Statistiken

## Verwendung

### Basis-Verwendung (alle Kandidaten testen)

```bash
cd /Users/MGraf/Python/LFO_Umgebung
python3 SDE/test_rsa_candidates_systematic.py
```

### Begrenzte Anzahl testen

```bash
# Teste nur die ersten 1000 Kandidaten
python3 SDE/test_rsa_candidates_systematic.py 1000
```

### Fortsetzen nach Unterbrechung

Das Skript speichert automatisch den Fortschritt. Einfach erneut ausführen:

```bash
python3 SDE/test_rsa_candidates_systematic.py
```

Es wird automatisch dort fortgesetzt, wo es aufgehört hat.

## Fortschrittsdatei

Der Fortschritt wird in `SDE/rsa_test_progress.json` gespeichert:

```json
{
  "tested_hashes": ["hash1", "hash2", ...],
  "successful_keys": [...],
  "stats": {
    "total_tested": 1234,
    "total_candidates": 893452,
    "start_time": "2025-12-08T...",
    "last_update": "2025-12-08T..."
  }
}
```

## Ausgabe

### Während des Tests

```
[100/893452] (0.0%) Offset 12152512, Entropie 7.21... [Gespeichert]
[200/893452] (0.0%) Offset 12152528, Entropie 7.26... [Gespeichert]
...
```

### Bei Erfolg

```
🎉 ERFOLG! RSA-Schlüssel gefunden bei Offset 12345678!
   Entropie: 7.45
   AES-Schlüssel: 32 Bytes
   AES-Schlüssel (hex): abc123...
   ✓ BAL-Datei V8.bal erfolgreich entschlüsselt!
   Gespeichert: found_rsa_key_12345678.bin
   AES-Schlüssel gespeichert: found_aes_key_12345678.bin
```

## Geschätzte Zeit

- **Kandidaten**: ~893.452 (bei min_entropy=7.0)
- **Geschwindigkeit**: ~10-50 Tests/Sekunde (abhängig von CPU)
- **Geschätzte Zeit**: 
  - 1000 Tests: ~20-100 Sekunden
  - 10.000 Tests: ~3-17 Minuten
  - Alle Tests: ~5-12 Stunden

## Unterbrechung und Fortsetzung

### Skript stoppen
- `Ctrl+C` drücken
- Das Skript speichert automatisch vor dem Beenden

### Fortsetzen
- Einfach erneut ausführen
- Das Skript lädt automatisch den gespeicherten Fortschritt

## Erfolgreiche Schlüssel

Wenn ein Schlüssel gefunden wird, werden folgende Dateien erstellt:

1. **`found_rsa_key_<offset>.bin`**: Der gefundene RSA-Schlüssel
2. **`found_aes_key_<offset>.bin`**: Der entschlüsselte AES-Schlüssel
3. **`rsa_test_progress.json`**: Enthält alle erfolgreichen Schlüssel

## Tipps

1. **Starte mit begrenzter Anzahl**: Teste zuerst 1000-10000 Kandidaten
   ```bash
   python3 SDE/test_rsa_candidates_systematic.py 10000
   ```

2. **Höhere Entropie zuerst**: Das Skript sortiert Kandidaten nach Entropie (höchste zuerst)

3. **Im Hintergrund laufen lassen**: 
   ```bash
   nohup python3 SDE/test_rsa_candidates_systematic.py > test.log 2>&1 &
   ```

4. **Fortschritt überwachen**:
   ```bash
   tail -f test.log
   ```

5. **Fortschritt prüfen**:
   ```bash
   cat SDE/rsa_test_progress.json | python3 -m json.tool
   ```

## Fehlerbehebung

### "pycryptodome nicht verfügbar"
```bash
pip3 install pycryptodome
```

### "ArrayCalc nicht gefunden"
- Prüfe ob ArrayCalc unter `/Applications/ArrayCalc V12.app` installiert ist

### "SoundPLAN.enc nicht gefunden"
- Stelle sicher, dass die SDE-Datei bereits extrahiert wurde
- Führe zuerst `sde_loader.py` aus

## Nächste Schritte nach erfolgreichem Fund

Wenn ein RSA-Schlüssel gefunden wurde:

1. **Teste mit allen BAL-Dateien**:
   ```python
   from sde_loader import SDELoader
   loader = SDELoader("sde.sde")
   loader.load()
   # Verwende gefundenen AES-Schlüssel
   ```

2. **Entschlüssele alle BAL-Dateien**:
   - Verwende den gefundenen AES-Schlüssel
   - Entschlüssele alle .bal Dateien
   - Konvertiere zu .bin oder anderen Formaten

## Hinweise

- Das Skript ist CPU-intensiv
- Speichert regelmäßig, kann jederzeit gestoppt werden
- Erfolgreiche Schlüssel werden sofort gespeichert
- Fortschritt wird in JSON-Format gespeichert (menschlich lesbar)

