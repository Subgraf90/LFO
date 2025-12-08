# Wie sehe ich, wenn ein Schlüssel gefunden wurde?

## Automatische Benachrichtigungen

Wenn ein RSA-Schlüssel gefunden wird, erscheint sofort eine Meldung im Log:

```
🎉 ERFOLG! RSA-Schlüssel gefunden bei Offset 12345678!
   Entropie: 7.45
   AES-Schlüssel: 32 Bytes
   AES-Schlüssel (hex): abc123...
   ✓ BAL-Datei V8.bal erfolgreich entschlüsselt!
   Gespeichert: found_rsa_key_12345678.bin
   AES-Schlüssel gespeichert: found_aes_key_12345678.bin
```

## Methoden zum Überwachen

### 1. Live-Log überwachen (Empfohlen)

```bash
cd /Users/MGraf/Python/LFO_Umgebung
tail -f SDE/test_rsa.log
```

Drücke `Ctrl+C` zum Beenden. Die Meldung erscheint sofort, wenn ein Schlüssel gefunden wird.

### 2. Automatisches Prüf-Skript

```bash
cd /Users/MGraf/Python/LFO_Umgebung
./SDE/check_results.sh
```

Dieses Skript prüft:
- ✅ Log-Datei nach Erfolgsmeldungen
- ✅ Gefundene Schlüssel-Dateien
- ✅ Fortschrittsdatei nach erfolgreichen Schlüsseln

### 3. Nach Erfolgsmeldungen suchen

```bash
cd /Users/MGraf/Python/LFO_Umgebung
grep -i "🎉\|ERFOLG\|success\|gefunden" SDE/test_rsa.log
```

### 4. Prüfe auf neue Dateien

Wenn ein Schlüssel gefunden wird, werden automatisch erstellt:

```bash
cd /Users/MGraf/Python/LFO_Umgebung/SDE
ls -lah found_*.bin
```

Erwartete Dateien:
- `found_rsa_key_<offset>.bin` - Der gefundene RSA-Schlüssel
- `found_aes_key_<offset>.bin` - Der entschlüsselte AES-Schlüssel

## Fortschrittsdatei

Die Fortschrittsdatei `rsa_test_progress.json` enthält alle erfolgreichen Schlüssel:

```json
{
  "successful_keys": [
    {
      "offset": 12345678,
      "entropy": 7.45,
      "aes_key": "abc123...",
      "aes_key_length": 32,
      "bal_success": true,
      "timestamp": "2025-12-08T..."
    }
  ]
}
```

## Kontinuierliches Monitoring

### Option 1: Watch-Befehl (alle 5 Sekunden)

```bash
watch -n 5 ./SDE/check_results.sh
```

### Option 2: In separatem Terminal

Öffne ein neues Terminal und führe aus:

```bash
cd /Users/MGraf/Python/LFO_Umgebung
tail -f SDE/test_rsa.log | grep --line-buffered -i "🎉\|ERFOLG"
```

### Option 3: Automatische Benachrichtigung

```bash
# Prüfe alle 30 Sekunden und benachrichtige bei Erfolg
while true; do
    if grep -q "🎉\|ERFOLG" SDE/test_rsa.log 2>/dev/null; then
        echo "ALARM: Schlüssel gefunden!"
        say "Schlüssel gefunden"  # macOS Sprachausgabe
        break
    fi
    sleep 30
done
```

## Was passiert bei Erfolg?

1. **Sofortige Log-Meldung**: erscheint im Log mit 🎉
2. **Dateien werden erstellt**: 
   - `found_rsa_key_<offset>.bin`
   - `found_aes_key_<offset>.bin`
3. **Fortschrittsdatei wird aktualisiert**: `rsa_test_progress.json`
4. **Test läuft weiter**: testet alle verbleibenden Kandidaten

## Aktueller Status prüfen

```bash
cd /Users/MGraf/Python/LFO_Umgebung
./SDE/check_results.sh
```

Oder:

```bash
./SDE/monitor_progress.sh
```

## Wichtig

- ✅ Erfolgsmeldungen erscheinen **sofort** im Log
- ✅ Dateien werden **sofort** erstellt
- ✅ Das Skript läuft **weiter** nach einem Fund (kann mehrere geben)
- ✅ Alle gefundenen Schlüssel werden gespeichert

## Beispiel-Ausgabe bei Erfolg

```
[123456/215903] (57.2%) Offset 12345678, Entropie 7.45...

🎉 ERFOLG! RSA-Schlüssel gefunden bei Offset 12345678!
   Entropie: 7.45
   AES-Schlüssel: 32 Bytes
   AES-Schlüssel (hex): 1a2b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890
   ✓ BAL-Datei V8.bal erfolgreich entschlüsselt!
   Gespeichert: found_rsa_key_12345678.bin
   AES-Schlüssel gespeichert: found_aes_key_12345678.bin

[123457/215903] (57.2%) Offset 12345679, Entropie 7.20...
```

