"""
SDE-Datei Loader und BAL-Entschlüsselung

Dieses Modul lädt SDE-Dateien (SoundPLAN/ArrayCalc Projektdateien) und
versucht, die verschlüsselten BAL-Dateien zu entschlüsseln.

SDE-Dateien sind ZIP-Archive mit:
- OBJ/MTL-Dateien (3D-Modelle)
- XML-Dateien (manifest.xml, SoundSystem.xml)
- BAL-Dateien (verschlüsselte Balloon-Daten)
- SoundPLAN.enc (Verschlüsselungsschlüssel)
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SDELoader:
    """Lädt und analysiert SDE-Dateien"""
    
    def __init__(self, sde_file_path: str, output_dir: Optional[str] = None):
        """
        Args:
            sde_file_path: Pfad zur .sde-Datei
            output_dir: Verzeichnis für extrahierte Dateien (Standard: SDE/)
        """
        self.sde_file_path = Path(sde_file_path)
        if output_dir is None:
            output_dir = self.sde_file_path.parent / "SDE"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest = None
        self.sound_system = None
        self.extracted_files = {}
        self.bal_files = []
        self.encryption_key = None
    
    def load(self) -> bool:
        """
        Lädt die SDE-Datei und extrahiert alle Dateien.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if not self.sde_file_path.exists():
                logger.error(f"SDE-Datei nicht gefunden: {self.sde_file_path}")
                return False
            
            logger.info(f"Lade SDE-Datei: {self.sde_file_path}")
            
            # Öffne ZIP-Archiv
            with zipfile.ZipFile(self.sde_file_path, 'r') as zip_ref:
                # Liste alle Dateien
                file_list = zip_ref.namelist()
                logger.info(f"Gefundene Dateien: {len(file_list)}")
                
                # Extrahiere alle Dateien
                zip_ref.extractall(self.output_dir)
                logger.info(f"Dateien extrahiert nach: {self.output_dir}")
                
                # Lade Manifest
                manifest_path = self.output_dir / "manifest.xml"
                if manifest_path.exists():
                    self.manifest = self._load_xml(manifest_path)
                    logger.info("Manifest geladen")
                
                # Lade SoundSystem
                sound_system_path = self.output_dir / "SoundSystem.xml"
                if sound_system_path.exists():
                    self.sound_system = self._load_xml(sound_system_path)
                    logger.info("SoundSystem.xml geladen")
                
                # Finde BAL-Dateien
                self.bal_files = [
                    self.output_dir / f for f in file_list 
                    if f.endswith('.bal')
                ]
                logger.info(f"Gefundene BAL-Dateien: {len(self.bal_files)}")
                for bal_file in self.bal_files:
                    logger.info(f"  - {bal_file.name}")
                
                # Lade Verschlüsselungsschlüssel
                enc_key_path = self.output_dir / "SoundPLAN.enc"
                if enc_key_path.exists():
                    with open(enc_key_path, 'rb') as f:
                        self.encryption_key = f.read()
                    logger.info(f"Verschlüsselungsschlüssel geladen: {len(self.encryption_key)} Bytes")
                
                # Speichere extrahierte Dateien
                for file_name in file_list:
                    file_path = self.output_dir / file_name
                    if file_path.exists():
                        self.extracted_files[file_name] = file_path
                
                return True
                
        except zipfile.BadZipFile:
            logger.error(f"Ungültiges ZIP-Archiv: {self.sde_file_path}")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Laden der SDE-Datei: {e}", exc_info=True)
            return False
    
    def _load_xml(self, xml_path: Path) -> Optional[ET.Element]:
        """Lädt eine XML-Datei"""
        try:
            tree = ET.parse(xml_path)
            return tree.getroot()
        except Exception as e:
            logger.error(f"Fehler beim Laden von XML {xml_path}: {e}")
            return None
    
    def get_manifest_info(self) -> Dict:
        """Gibt Informationen aus dem Manifest zurück"""
        if self.manifest is None:
            return {}
        
        info = {}
        
        # User Agent Info
        user_agent = self.manifest.find('UserAgent')
        if user_agent is not None:
            info['product'] = user_agent.get('product', '')
            info['version'] = user_agent.get('version', '')
            info['system'] = user_agent.get('systemInformation', '')
        
        # Dateien-Liste
        files = []
        for file_elem in self.manifest.findall('.//File'):
            file_info = {
                'name': file_elem.find('Name').text if file_elem.find('Name') is not None else '',
                'encrypted': file_elem.find('Encrypted').text.lower() == 'true' if file_elem.find('Encrypted') is not None else False
            }
            files.append(file_info)
        info['files'] = files
        
        # Verschlüsselungsschlüssel
        encryption = self.manifest.find('.//EncryptionKeys')
        if encryption is not None:
            key_info = encryption.find('.//KeyForImportingSoftware')
            if key_info is not None:
                info['encryption'] = {
                    'software_id': key_info.find('ImportingSoftwareID').text if key_info.find('ImportingSoftwareID') is not None else '',
                    'key_file': key_info.find('EncryptedKeyFile').text if key_info.find('EncryptedKeyFile') is not None else ''
                }
        
        return info
    
    def get_sound_system_info(self) -> Dict:
        """Gibt Informationen aus SoundSystem.xml zurück"""
        if self.sound_system is None:
            return {}
        
        info = {}
        info['id'] = self.sound_system.get('id', '')
        info['version'] = self.sound_system.get('version', '')
        
        # Kommentare
        comments = self.sound_system.find('Comments')
        if comments is not None:
            info['comments'] = comments.text
        
        # Frequenzen
        freq = self.sound_system.find('.//Frequency')
        if freq is not None:
            info['frequency'] = {
                'id': freq.get('id', ''),
                'inverse_octave': freq.find('InverseOctaveBandFraction').text if freq.find('InverseOctaveBandFraction') is not None else '',
                'kmin': freq.find('kmin').text if freq.find('kmin') is not None else '',
                'kmax': freq.find('kmax').text if freq.find('kmax') is not None else ''
            }
        
        # Lautsprecher-Gruppen
        groups = []
        for group in self.sound_system.findall('.//CoherentSourceGroup'):
            group_info = {
                'id': group.get('id', ''),
                'name': group.get('name', ''),
                'cabinets': []
            }
            
            for cabinet in group.findall('.//LoudspeakerCabinet'):
                cabinet_info = {
                    'model': cabinet.find('.//Model').get('name', '') if cabinet.find('.//Model') is not None else '',
                    'balloon_file': None
                }
                
                # Position
                pos = cabinet.find('.//Position')
                if pos is not None:
                    cabinet_info['position'] = {
                        'x': pos.find('X').text if pos.find('X') is not None else '0',
                        'y': pos.find('Y').text if pos.find('Y') is not None else '0',
                        'z': pos.find('Z').text if pos.find('Z') is not None else '0'
                    }
                
                # Balloon-Datei
                balloon = cabinet.find('.//Balloon')
                if balloon is not None:
                    cabinet_info['balloon_file'] = balloon.get('filename', '')
                
                group_info['cabinets'].append(cabinet_info)
            
            groups.append(group_info)
        
        info['source_groups'] = groups
        
        return info
    
    def analyze_bal_file(self, bal_file_path: Path) -> Dict:
        """
        Analysiert eine BAL-Datei (verschlüsselt).
        
        Args:
            bal_file_path: Pfad zur BAL-Datei
            
        Returns:
            Dictionary mit Analyse-Informationen
        """
        if not bal_file_path.exists():
            return {'error': 'Datei nicht gefunden'}
        
        info = {
            'file': str(bal_file_path.name),
            'size': bal_file_path.stat().st_size,
            'is_encrypted': True,
            'file_type': 'unknown'
        }
        
        # Lese erste Bytes zur Analyse
        with open(bal_file_path, 'rb') as f:
            header = f.read(100)
            info['header_hex'] = header.hex()
            info['header_bytes'] = list(header[:20])
        
        # Prüfe ob es ein ZIP-Archiv ist (verschlüsselt)
        if header.startswith(b'PK'):
            info['file_type'] = 'encrypted_zip'
        elif header.startswith(b'f\xff'):
            info['file_type'] = 'encrypted_bal'
        else:
            info['file_type'] = 'binary_data'
        
        return info
    
    def decrypt_bal_file(self, bal_file_path: Path, output_path: Optional[Path] = None, method: str = "auto") -> bool:
        """
        Versucht, eine BAL-Datei zu entschlüsseln.
        
        Args:
            bal_file_path: Pfad zur verschlüsselten BAL-Datei
            output_path: Pfad für entschlüsselte Datei (Standard: bal_file_path + .decrypted)
            method: Entschlüsselungsmethode ("xor", "aes", "auto")
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not bal_file_path.exists():
            logger.error(f"BAL-Datei nicht gefunden: {bal_file_path}")
            return False
        
        if output_path is None:
            output_path = bal_file_path.parent / f"{bal_file_path.stem}.decrypted"
        
        try:
            # Lese verschlüsselte Datei
            with open(bal_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            logger.info(f"Versuche BAL-Datei zu entschlüsseln: {bal_file_path.name} ({len(encrypted_data)} Bytes)")
            
            if self.encryption_key is None:
                logger.warning("Kein Verschlüsselungsschlüssel gefunden")
                return False
            
            # Versuche verschiedene Entschlüsselungsmethoden
            # BASIEREND AUF REVERSE ENGINEERING:
            # ArrayCalc verwendet: RSA-verschlüsselter AES-Schlüssel in SoundPLAN.enc
            # BAL-Dateien sind mit AES verschlüsselt
            if method == "auto":
                methods = ["aes_rsa", "aes", "xor"]  # Versuche zuerst RSA+AES, dann direkt AES
            else:
                methods = [method]
            
            for method_name in methods:
                logger.info(f"Teste Entschlüsselungsmethode: {method_name}")
                decrypted = self._decrypt_with_method(encrypted_data, method_name)
                
                if decrypted is not None:
                    # Speichere entschlüsselte Datei
                    with open(output_path, 'wb') as f:
                        f.write(decrypted)
                    
                    # Prüfe ob Ergebnis sinnvoll ist
                    if self._validate_decrypted_data(decrypted):
                        logger.info(f"Entschlüsselung erfolgreich mit Methode: {method_name}")
                        return True
                    else:
                        logger.debug(f"Methode {method_name} ergab ungültige Daten")
            
            logger.warning("Keine erfolgreiche Entschlüsselungsmethode gefunden")
            return False
            
        except Exception as e:
            logger.error(f"Fehler beim Entschlüsseln: {e}", exc_info=True)
            return False
    
    def _decrypt_with_method(self, encrypted_data: bytes, method: str) -> Optional[bytes]:
        """
        Versucht Entschlüsselung mit einer bestimmten Methode.
        
        BASIEREND AUF REVERSE ENGINEERING VON ArrayCalc V12:
        - ArrayCalc verwendet OpenSSL EVP-Funktionen
        - SoundPLAN.enc enthält einen RSA-verschlüsselten AES-Schlüssel
        - BAL-Dateien sind mit AES verschlüsselt (wahrscheinlich AES-256-CBC)
        - Der AES-Schlüssel wird mit RSA verschlüsselt gespeichert
        
        Args:
            encrypted_data: Verschlüsselte Daten
            method: Methode ("xor", "aes", "aes_rsa")
            
        Returns:
            Entschlüsselte Daten oder None bei Fehler
        """
        if method == "xor":
            # XOR-Entschlüsselung mit zyklischem Schlüssel (nur zum Testen)
            decrypted = bytearray(encrypted_data)
            key_len = len(self.encryption_key)
            
            if key_len > 0:
                for i in range(len(decrypted)):
                    decrypted[i] ^= self.encryption_key[i % key_len]
                return bytes(decrypted)
        
        elif method == "aes":
            # Direkte AES-Entschlüsselung (wenn der Schlüssel bereits bekannt ist)
            try:
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import unpad
                
                # Der Schlüssel könnte direkt verwendet werden oder muss abgeleitet werden
                # Versuche verschiedene Schlüssellängen
                key = self.encryption_key[:32]  # AES-256 benötigt 32 Bytes
                if len(key) < 32:
                    key = key + b'\x00' * (32 - len(key))
                
                # Versuche verschiedene IV-Modi
                # Modus 1: Erste 16 Bytes als IV (typisch für CBC)
                if len(encrypted_data) >= 16:
                    iv = encrypted_data[:16]
                    ciphertext = encrypted_data[16:]
                    
                    try:
                        cipher = AES.new(key, AES.MODE_CBC, iv)
                        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
                        return decrypted
                    except:
                        pass
                
                # Modus 2: Null-IV
                try:
                    iv = b'\x00' * 16
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
                    return decrypted
                except:
                    pass
                
                # Modus 3: ECB (kein IV)
                try:
                    cipher = AES.new(key, AES.MODE_ECB)
                    decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
                    return decrypted
                except:
                    pass
                    
            except ImportError:
                logger.warning("pycryptodome nicht installiert. Installiere mit: pip install pycryptodome")
            except Exception as e:
                logger.debug(f"AES-Entschlüsselung fehlgeschlagen: {e}")
        
        elif method == "aes_rsa":
            # RSA-Entschlüsselung des AES-Schlüssels, dann AES-Entschlüsselung
            # SoundPLAN.enc enthält RSA-verschlüsselten AES-Schlüssel
            try:
                from Crypto.Cipher import AES, PKCS1_OAEP
                from Crypto.PublicKey import RSA
                from Crypto.Util.Padding import unpad
                
                # Versuche SoundPLAN.enc als RSA-verschlüsselten Schlüssel zu entschlüsseln
                # PROBLEM: Wir brauchen den privaten RSA-Schlüssel von dbaudio
                # Möglicherweise ist der öffentliche Schlüssel im Programm eingebettet
                
                # Versuche verschiedene RSA-Schlüssellängen
                # Typisch: 2048 oder 4096 Bit
                for key_size in [2048, 4096]:
                    try:
                        # Versuche den privaten Schlüssel zu extrahieren oder zu generieren
                        # Dies funktioniert nur, wenn wir den privaten Schlüssel haben
                        logger.debug(f"Versuche RSA-Entschlüsselung mit {key_size}-Bit Schlüssel")
                        # TODO: RSA-Schlüssel von dbaudio extrahieren oder finden
                    except:
                        pass
                
                logger.warning("RSA-Entschlüsselung erfordert privaten Schlüssel von dbaudio")
                return None
                    
            except ImportError:
                logger.warning("pycryptodome nicht installiert. Installiere mit: pip install pycryptodome")
            except Exception as e:
                logger.debug(f"RSA/AES-Entschlüsselung fehlgeschlagen: {e}")
        
        return None
    
    def _validate_decrypted_data(self, data: bytes) -> bool:
        """
        Prüft ob entschlüsselte Daten sinnvoll aussehen.
        
        Args:
            data: Entschlüsselte Daten
            
        Returns:
            True wenn Daten gültig erscheinen
        """
        if len(data) == 0:
            return False
        
        # Prüfe auf bekannte Dateiformate
        # ZIP-Archiv
        if data.startswith(b'PK'):
            return True
        
        # XML
        if data.startswith(b'<?xml') or data.startswith(b'<'):
            return True
        
        # Text-Datei (prüfe ob viele druckbare Zeichen vorhanden)
        printable_count = sum(1 for b in data[:100] if 32 <= b <= 126)
        if printable_count > 80:  # >80% druckbare Zeichen
            return True
        
        # Prüfe auf Null-Bytes (zu viele deuten auf fehlgeschlagene Entschlüsselung hin)
        null_count = sum(1 for b in data[:100] if b == 0)
        if null_count > 50:  # >50% Null-Bytes
            return False
        
        # Prüfe auf gleichmäßige Verteilung (Entropie)
        # Verschlüsselte Daten haben hohe Entropie, entschlüsselte sollten strukturiert sein
        byte_counts = [0] * 256
        for b in data[:min(1000, len(data))]:
            byte_counts[b] += 1
        
        # Berechne Entropie
        import math
        entropy = 0
        for count in byte_counts:
            if count > 0:
                p = count / min(1000, len(data))
                entropy -= p * math.log2(p)
        
        # Niedrige Entropie (< 6) deutet auf erfolgreiche Entschlüsselung hin
        if entropy < 6:
            return True
        
        return False
    
    def decrypt_all_bal_files(self) -> Dict[str, bool]:
        """
        Versucht, alle BAL-Dateien zu entschlüsseln.
        
        Returns:
            Dictionary mit Dateinamen und Erfolgsstatus
        """
        results = {}
        
        for bal_file in self.bal_files:
            success = self.decrypt_bal_file(bal_file)
            results[bal_file.name] = success
        
        return results


def main():
    """Beispiel-Verwendung"""
    import sys
    
    if len(sys.argv) < 2:
        print("Verwendung: python sde_loader.py <sde_file.sde>")
        sys.exit(1)
    
    sde_file = sys.argv[1]
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Lade SDE-Datei
    loader = SDELoader(sde_file)
    
    if not loader.load():
        print("Fehler beim Laden der SDE-Datei")
        sys.exit(1)
    
    # Zeige Informationen
    print("\n=== Manifest-Informationen ===")
    manifest_info = loader.get_manifest_info()
    print(f"Product: {manifest_info.get('product', 'N/A')}")
    print(f"Version: {manifest_info.get('version', 'N/A')}")
    print(f"System: {manifest_info.get('system', 'N/A')}")
    
    print("\n=== SoundSystem-Informationen ===")
    sound_info = loader.get_sound_system_info()
    print(f"ID: {sound_info.get('id', 'N/A')}")
    print(f"Kommentare: {sound_info.get('comments', 'N/A')}")
    
    print("\n=== BAL-Dateien-Analyse ===")
    for bal_file in loader.bal_files:
        analysis = loader.analyze_bal_file(bal_file)
        print(f"\n{analysis['file']}:")
        print(f"  Größe: {analysis['size']} Bytes")
        print(f"  Typ: {analysis['file_type']}")
        print(f"  Header (hex): {analysis['header_hex'][:40]}...")
    
    print("\n=== Versuche Entschlüsselung ===")
    results = loader.decrypt_all_bal_files()
    for filename, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {filename}")


if __name__ == "__main__":
    main()

