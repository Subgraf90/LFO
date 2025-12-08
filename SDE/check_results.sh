#!/bin/bash
# Skript zum Prüfen ob ein Schlüssel gefunden wurde

LOG_FILE="/Users/MGraf/Python/LFO_Umgebung/SDE/test_rsa.log"
SDE_DIR="/Users/MGraf/Python/LFO_Umgebung/SDE"

echo "=== Prüfe auf gefundene RSA-Schlüssel ==="
echo ""

# 1. Prüfe Log-Datei nach Erfolgsmeldungen
echo "1. Suche in Log-Datei nach Erfolgsmeldungen..."
if [ -f "$LOG_FILE" ]; then
    success_count=$(grep -i "🎉\|ERFOLG\|success\|gefunden" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$success_count" -gt 0 ]; then
        echo "   ✓ $success_count Erfolgsmeldung(en) gefunden!"
        echo ""
        echo "   Letzte Erfolgsmeldungen:"
        grep -i "🎉\|ERFOLG\|success\|gefunden" "$LOG_FILE" 2>/dev/null | tail -5
    else
        echo "   ✗ Noch keine Erfolgsmeldungen"
    fi
else
    echo "   ✗ Log-Datei nicht gefunden"
fi

echo ""

# 2. Prüfe auf gefundene Schlüssel-Dateien
echo "2. Prüfe auf gefundene Schlüssel-Dateien..."
found_files=$(ls -1 "$SDE_DIR"/found_*.bin 2>/dev/null)
if [ -n "$found_files" ]; then
    echo "   ✓ Gefundene Dateien:"
    for file in $found_files; do
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "     - $(basename $file) ($size)"
    done
else
    echo "   ✗ Noch keine gefundenen Schlüssel-Dateien"
fi

echo ""

# 3. Prüfe Fortschrittsdatei nach erfolgreichen Schlüsseln
echo "3. Prüfe Fortschrittsdatei..."
if [ -f "$SDE_DIR/rsa_test_progress.json" ]; then
    python3 << EOF
import json
import sys

try:
    with open("$SDE_DIR/rsa_test_progress.json", 'r') as f:
        # Versuche JSON zu laden, ignoriere Fehler wenn Datei beschädigt
        content = f.read()
        try:
            data = json.loads(content)
        except:
            # Versuche nur den letzten Teil zu lesen
            print("   ⚠ Fortschrittsdatei hat JSON-Fehler (wird weiter geschrieben)")
            # Suche nach "successful_keys"
            if '"successful_keys"' in content:
                print("   ✓ 'successful_keys' Feld gefunden in Datei")
            sys.exit(0)
        
        successful = data.get('successful_keys', [])
        if successful:
            print(f"   ✓ {len(successful)} erfolgreiche Schlüssel gefunden!")
            for i, key in enumerate(successful, 1):
                print(f"     Schlüssel {i}:")
                print(f"       Offset: {key.get('offset', 'N/A')}")
                print(f"       AES-Schlüssel: {key.get('aes_key', 'N/A')[:32]}...")
                print(f"       BAL-Erfolg: {key.get('bal_success', False)}")
        else:
            print("   ✗ Noch keine erfolgreichen Schlüssel")
except Exception as e:
    print(f"   ⚠ Fehler beim Lesen: {e}")
EOF
else
    echo "   ✗ Fortschrittsdatei nicht gefunden"
fi

echo ""
echo "=== Aktueller Status ==="
if pgrep -f "test_rsa_candidates_systematic" > /dev/null; then
    echo "✓ Prozess läuft noch"
    echo ""
    echo "Zum Live-Monitoring:"
    echo "  tail -f $LOG_FILE"
else
    echo "✗ Prozess läuft nicht mehr"
fi

