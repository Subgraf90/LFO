#!/bin/bash
# Startet SoundVision mit FridaGadget und verbindet Frida

APP_PATH="/Users/MGraf/Desktop/soundvision_resigned/Soundvision.app"
GADGET_PATH="$APP_PATH/Contents/Frameworks/FridaGadget.dylib"
HOOK_SCRIPT="/Users/MGraf/Python/LFO_Umgebung/SDE/hook_soundvision_bal.js"
FRIDA_BIN="/Users/MGraf/Library/Python/3.9/bin/frida"

echo "=== Starte SoundVision mit FridaGadget ==="

# Beende alte Instanzen
killall Soundvision 2>/dev/null
sleep 1

# Starte App mit FridaGadget
echo "Starte SoundVision..."
DYLD_INSERT_LIBRARIES="$GADGET_PATH" "$APP_PATH/Contents/MacOS/Soundvision" > /tmp/soundvision.log 2>&1 &

# Warte auf FridaGadget
echo "Warte auf FridaGadget..."
for i in {1..10}; do
    if lsof -i :27042 >/dev/null 2>&1; then
        echo "✓ FridaGadget läuft auf Port 27042"
        break
    fi
    sleep 1
done

# Finde PID
PID=$(ps aux | grep soundvision_resigned | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "✗ SoundVision läuft nicht!"
    exit 1
fi

echo "SoundVision PID: $PID"

# Verbinde Frida
echo "Verbinde Frida..."
sleep 2
$FRIDA_BIN -H 127.0.0.1:27042 -p $PID -l "$HOOK_SCRIPT"

