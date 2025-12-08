#!/bin/bash
# Skript zum Überwachen des RSA-Test-Fortschritts

LOG_FILE="/Users/MGraf/Python/LFO_Umgebung/SDE/test_rsa.log"
PROGRESS_FILE="/Users/MGraf/Python/LFO_Umgebung/SDE/rsa_test_progress.json"

echo "=== RSA-Test Fortschritts-Monitor ==="
echo ""

# Prüfe ob Prozess läuft
if pgrep -f "test_rsa_candidates_systematic" > /dev/null; then
    echo "✓ Prozess läuft"
else
    echo "✗ Prozess läuft nicht"
fi

echo ""
echo "=== Letzte Log-Zeilen ==="
tail -10 "$LOG_FILE" 2>/dev/null || echo "Keine Log-Datei gefunden"

echo ""
echo "=== Fortschritts-Statistiken ==="
if [ -f "$PROGRESS_FILE" ]; then
    python3 << EOF
import json
import sys
from datetime import datetime

try:
    with open("$PROGRESS_FILE", 'r') as f:
        data = json.load(f)
    
    stats = data.get('stats', {})
    tested = stats.get('total_tested', 0)
    total = stats.get('total_candidates', 0)
    successful = len(data.get('successful_keys', []))
    
    print(f"Getestet: {tested:,}")
    print(f"Gesamt: {total:,}")
    if total > 0:
        progress = (tested / total) * 100
        print(f"Fortschritt: {progress:.2f}%")
    print(f"Erfolgreiche Schlüssel: {successful}")
    
    if 'start_time' in stats:
        print(f"Start: {stats['start_time']}")
    if 'last_update' in stats:
        print(f"Letzte Aktualisierung: {stats['last_update']}")
    
    if successful > 0:
        print("\n🎉 Erfolgreiche Schlüssel gefunden!")
        for key in data.get('successful_keys', []):
            print(f"  - Offset: {key.get('offset')}")
            print(f"    AES-Schlüssel: {key.get('aes_key', '')[:32]}...")
            print(f"    BAL-Erfolg: {key.get('bal_success', False)}")
except Exception as e:
    print(f"Fehler: {e}")
EOF
else
    echo "Keine Fortschrittsdatei gefunden"
fi

echo ""
echo "=== Prozess-Info ==="
ps aux | grep "test_rsa_candidates_systematic" | grep -v grep | awk '{print "PID: "$2", CPU: "$3"%, MEM: "$4"%"}'

