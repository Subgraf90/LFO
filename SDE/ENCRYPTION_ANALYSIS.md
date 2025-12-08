# Verschlüsselungsanalyse - ArrayCalc V12 BAL-Dateien

## Erkenntnisse aus Reverse Engineering

### Gefundene Komponenten

Durch Analyse der ArrayCalc V12.app wurden folgende Verschlüsselungskomponenten identifiziert:

1. **SymmetricEncryption** Klasse
   - Verwendet OpenSSL EVP-Funktionen
   - Datei: `SymmetricEncryption.cpp`

2. **AesConnector** Klasse
   - Verbindet AES-Verschlüsselung mit anderen Komponenten

3. **OpenSSL EVP-Funktionen** verwendet:
   - `EVP_EncryptInit_ex` / `EVP_DecryptInit_ex`
   - `EVP_EncryptUpdate` / `EVP_DecryptUpdate`
   - `EVP_EncryptFinal_ex` / `EVP_DecryptFinal_ex`
   - `EVP_CIPHER_CTX_new`
   - `EVP_CIPHER_CTX_set_padding`
   - `RAND_bytes` (für Schlüssel/IV-Generierung)

4. **RSA-Verschlüsselung** für Schlüssel:
   - `EVP_PKEY_encrypt_init` / `EVP_PKEY_decrypt_init`
   - `EVP_PKEY_encrypt` / `EVP_PKEY_decrypt`
   - `EVP_PKEY_CTX_set_rsa_padding`
   - Klasse: `EncryptKeysRsa`

### Verschlüsselungsschema

```
┌─────────────────────────────────────────────────────────┐
│  BAL-Datei Verschlüsselung                              │
└─────────────────────────────────────────────────────────┘

1. AES-Schlüssel wird generiert (wahrscheinlich AES-256)
   └─> RAND_bytes() generiert zufälligen Schlüssel

2. AES-Schlüssel wird mit RSA verschlüsselt
   └─> Gespeichert in: SoundPLAN.enc

3. BAL-Datei wird mit AES verschlüsselt
   └─> Wahrscheinlich AES-256-CBC Modus
   └─> IV wird wahrscheinlich vor den verschlüsselten Daten gespeichert

4. Entschlüsselung (in ArrayCalc):
   └─> RSA-Entschlüsselung von SoundPLAN.enc → AES-Schlüssel
   └─> AES-Entschlüsselung der BAL-Datei mit dem Schlüssel
```

### Dateistruktur

#### SoundPLAN.enc
- **Größe**: 256 Bytes (typisch für RSA-2048 verschlüsselte Daten)
- **Inhalt**: RSA-verschlüsselter AES-Schlüssel
- **Format**: Binär, OpenSSL RSA-Verschlüsselung

#### BAL-Dateien (.bal)
- **Verschlüsselung**: AES (wahrscheinlich AES-256-CBC)
- **IV**: Wahrscheinlich erste 16 Bytes der Datei
- **Ciphertext**: Rest der Datei

### Problem bei der Entschlüsselung

**Kritisch**: Um die BAL-Dateien zu entschlüsseln, benötigen wir:

1. **Privaten RSA-Schlüssel** von dbaudio
   - Dieser ist wahrscheinlich im ArrayCalc-Programm eingebettet
   - Oder wird aus einer Lizenz/Registrierung abgeleitet

2. **RSA-Entschlüsselung** von SoundPLAN.enc
   - Ergibt den AES-Schlüssel

3. **AES-Entschlüsselung** der BAL-Datei
   - Mit dem entschlüsselten AES-Schlüssel

### Nächste Schritte

1. **RSA-Schlüssel extrahieren**:
   - Suche nach eingebetteten RSA-Schlüsseln im ArrayCalc-Programm
   - Analysiere Strings nach RSA-Schlüssel-Formaten (PEM, DER)
   - Suche nach Schlüsseln in Datenbanken oder Konfigurationsdateien

2. **Verschlüsselungsparameter identifizieren**:
   - RSA-Schlüssellänge (2048 oder 4096 Bit?)
   - RSA-Padding (OAEP oder PKCS1?)
   - AES-Modus (CBC, GCM?)
   - IV-Position und -Format

3. **Test-Entschlüsselung**:
   - Mit extrahiertem Schlüssel testen
   - Validierung der entschlüsselten Daten

### Code-Referenzen

Gefundene Strings in ArrayCalc V12:
```
SymmetricEncryption
AesConnector
EVP_EncryptInit_ex
EVP_DecryptInit_ex
EVP_PKEY_encrypt
EVP_PKEY_decrypt
EncryptKeysRsa
SoundPLAN.enc
.bal
```

### Tools für weitere Analyse

1. **Hopper Disassembler** oder **IDA Pro**:
   - Detaillierte Analyse der Verschlüsselungsfunktionen
   - Identifikation der genauen Parameter

2. **lldb** (Debugger):
   - Runtime-Analyse während der Entschlüsselung
   - Breakpoints bei Verschlüsselungsfunktionen

3. **Frida** (Dynamic Instrumentation):
   - Hook von OpenSSL-Funktionen
   - Extraktion von Schlüsseln zur Laufzeit

### Rechtliche Hinweise

⚠️ **WICHTIG**: 
- Reverse Engineering von Software kann gegen Lizenzvereinbarungen verstoßen
- Prüfe die EULA von ArrayCalc/dbaudio
- Verwende diese Informationen nur für legitime Zwecke (z.B. Migration eigener Daten)

