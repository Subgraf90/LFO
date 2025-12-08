# Reverse Engineering Status - RSA-Schlüssel Extraktion

## Was wir herausgefunden haben

### ✅ Erfolgreich identifiziert:

1. **Verschlüsselungsmethode**:
   - SoundPLAN.enc: RSA-2048 verschlüsselter AES-Schlüssel (256 Bytes)
   - BAL-Dateien: AES-verschlüsselt (wahrscheinlich AES-256-CBC)

2. **Code-Stellen**:
   - `SymmetricEncryption` Klasse
   - `AesConnector` Klasse  
   - `EncryptKeysRsa` Klasse
   - OpenSSL EVP-Funktionen

3. **Binäranalyse**:
   - Viele Kandidaten mit hoher Entropie gefunden
   - Keine offensichtlichen RSA-Schlüssel im DER-Format
   - Schlüssel könnte hardcodiert, abgeleitet oder dynamisch sein

## Herausforderungen

### Problem 1: Schlüssel-Format unbekannt
- Der RSA-Schlüssel ist nicht als Standard-PEM/DER gespeichert
- Könnte in proprietärem Format vorliegen
- Könnte aus mehreren Teilen zusammengesetzt sein

### Problem 2: Schlüssel könnte abgeleitet werden
- Aus Hardware-ID
- Aus Lizenz-Informationen
- Aus Benutzer-Daten
- Dynamisch generiert

### Problem 3: Statische Analyse hat Grenzen
- Binärcode ist sehr groß (60 MB)
- Viele false positives (hohe Entropie ≠ RSA-Schlüssel)
- Verschlüsselungslogik könnte komplex sein

## Nächste Schritte

### Option 1: Runtime-Analyse (Empfohlen)
**Mit Frida oder lldb**:
- Hook von OpenSSL-Funktionen zur Laufzeit
- Extraktion des RSA-Schlüssels während der Entschlüsselung
- Direkte Beobachtung der Verschlüsselungsoperationen

**Vorteile**:
- Sehr effektiv
- Sieht tatsächliche Schlüssel-Werte
- Keine Interpretation nötig

**Nachteile**:
- Erfordert laufendes ArrayCalc
- Benötigt Test-SDE-Datei

### Option 2: Detaillierte Binäranalyse
**Mit Hopper Disassembler oder IDA Pro**:
- Analyse der Verschlüsselungsfunktionen
- Identifikation der genauen Parameter
- Verfolgung des Schlüssel-Flows

**Vorteile**:
- Sehr detailliert
- Versteht die Logik

**Nachteile**:
- Zeitaufwändig
- Erfordert spezialisierte Tools
- Komplexe Reverse Engineering

### Option 3: Brute-Force der Kandidaten
**Systematisches Testen**:
- Alle gefundenen Kandidaten testen
- Verschiedene Interpretationen versuchen
- Kombinationen ausprobieren

**Vorteile**:
- Automatisierbar
- Kann funktionieren, wenn Schlüssel statisch ist

**Nachteile**:
- Sehr viele Kandidaten (893.452!)
- Sehr zeitaufwändig
- Unwahrscheinlich erfolgreich

### Option 4: Kontakt mit dbaudio
**Offizieller Weg**:
- Nach Export-Tool fragen
- Nach API für Datenzugriff fragen
- Anwendungsfall erklären (Datenmigration)

**Vorteile**:
- Rechtlich sicher
- Offizielle Unterstützung

**Nachteile**:
- Möglicherweise keine Antwort
- Möglicherweise kostenpflichtig

## Empfehlung

**Beste Option**: Runtime-Analyse mit Frida

Warum:
1. Sehr effektiv - sieht tatsächliche Schlüssel
2. Relativ einfach zu implementieren
3. Funktioniert auch bei komplexer Verschlüsselung
4. Kann automatisiert werden

**Nächster Schritt**: Frida-Skript erstellen, das:
- OpenSSL EVP_Funktionen hooked
- RSA-Entschlüsselung abfängt
- AES-Schlüssel extrahiert
- In Datei speichert

## Code-Status

### Erstellt:
- ✅ `sde_loader.py` - Lädt SDE-Dateien
- ✅ `extract_rsa_key.py` - Basis RSA-Suche
- ✅ `extract_rsa_key_advanced.py` - Erweiterte Suche
- ✅ `test_rsa_candidates.py` - Testet Kandidaten (benötigt pycryptodome)

### Benötigt:
- ⏳ Frida-Skript für Runtime-Analyse
- ⏳ Installation von pycryptodome: `pip install pycryptodome`

## Fazit

**Statische Analyse allein reicht nicht aus.**

Der RSA-Schlüssel ist wahrscheinlich:
- Nicht als Standard-Format gespeichert
- Oder dynamisch/abgeleitet
- Oder in verschlüsselter Form

**Runtime-Analyse ist der vielversprechendste Ansatz.**

