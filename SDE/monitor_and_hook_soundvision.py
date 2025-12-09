#!/usr/bin/env python3
"""
Kombiniertes Monitoring und Frida-Hooking für SoundVision BAL-Entschlüsselung.

- Überwacht Dateisystem auf temporäre/entpackte Dateien
- Startet Frida mit erweiterten Hooks
- Kombiniert beide Ansätze zur Schlüssel-Extraktion
"""

import os
import time
import subprocess
import threading
from pathlib import Path
from typing import List, Set, Dict
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FileMonitor:
    """Überwacht Dateisystem auf neue Dateien."""
    
    def __init__(self):
        self.known_files = set()
        self.new_files = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Startet Dateisystem-Monitoring."""
        logger.info("Starte Dateisystem-Monitoring...")
        
        # Erfasse bekannte Dateien
        monitor_dirs = [
            Path.home() / "Library" / "Caches",
            Path.home() / "Library" / "Application Support",
            Path("/tmp"),
            # Path("/var/folders"),  # Oft Permission-Fehler
        ]
        
        for directory in monitor_dirs:
            if not directory.exists():
                continue
            try:
                # Verwende os.walk statt rglob für bessere Fehlerbehandlung
                for root, dirs, files in os.walk(str(directory), onerror=lambda x: None):
                    try:
                        # Überspringe Verzeichnisse mit Permission-Fehlern
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        for file in files:
                            try:
                                file_path = Path(root) / file
                                if file_path.is_file():
                                    self.known_files.add(file_path)
                            except (PermissionError, OSError):
                                continue
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError) as e:
                logger.debug(f"Überspringe {directory}: {e}")
                continue
            except Exception as e:
                # Fange alle anderen Fehler ab
                logger.debug(f"Fehler bei {directory}: {e}")
                continue
        
        logger.info(f"Bekannte Dateien: {len(self.known_files)}")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitoring-Schleife."""
        while self.monitoring:
            time.sleep(2)  # Prüfe alle 2 Sekunden
            
            monitor_dirs = [
                Path.home() / "Library" / "Caches",
                Path.home() / "Library" / "Application Support",
                Path("/tmp"),
            ]
            
            for directory in monitor_dirs:
                if not directory.exists():
                    continue
                try:
                    # Verwende os.walk statt rglob
                    for root, dirs, files in os.walk(str(directory)):
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        for file in files:
                            try:
                                file_path = Path(root) / file
                                if file_path.is_file() and file_path not in self.known_files:
                                    self.known_files.add(file_path)
                                    self._check_file(file_path)
                            except (PermissionError, OSError):
                                continue
                except (PermissionError, OSError) as e:
                    continue  # Überspringe Verzeichnis bei Fehler
    
    def _check_file(self, file: Path):
        """Prüft ob eine Datei interessant ist."""
        name_lower = file.name.lower()
        if ('.bal' in name_lower or 
            'balloon' in name_lower or
            'decrypt' in name_lower or
            'temp' in name_lower or
            file.suffix in ['.dat', '.bin', '.tmp']):
            
            size = file.stat().st_size if file.exists() else 0
            
            # Prüfe auf BAL-ähnliche Dateien
            if 100000 < size < 1000000:  # ~100KB - 1MB
                try:
                    with open(file, 'rb') as f:
                        header = f.read(16)
                        if header[:4] == b'\x1d\xf9\x3e\xfb':
                            logger.info(f"[MONITOR] SoundVision BAL gefunden: {file}")
                            logger.info(f"  Größe: {size:,} Bytes")
                            logger.info(f"  Header: {header.hex()}")
                            self.new_files.append({
                                'path': file,
                                'size': size,
                                'header': header.hex(),
                                'type': 'bal_file'
                            })
                except Exception as e:
                    pass
    
    def stop_monitoring(self):
        """Stoppt Monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_results(self) -> List[Dict]:
        """Gibt gefundene Dateien zurück."""
        return self.new_files.copy()


def create_enhanced_frida_script() -> Path:
    """Erstellt ein erweitertes Frida-Hook-Skript."""
    script_content = '''// Erweiterte Frida-Hooks für SoundVision BAL-Entschlüsselung
function hex(buf, len) {
  if (!buf) return 'null';
  const u = new Uint8Array(buf);
  const n = len ? Math.min(len, u.length) : u.length;
  return Array.from(u.slice(0, n)).map(b => ('0' + b.toString(16)).slice(-2)).join('');
}

function dump(ptr, len) {
  try { return Memory.readByteArray(ptr, len); } catch(e) { return null; }
}

function install() {
  console.log('=== Enhanced SoundVision BAL Decryption Hook ===');
  
  const balHeader = '1df93efb1a2c50731916a8e484d9d6fc';
  let foundKeys = [];
  
  // Hook alle Memory-Operationen, die große Datenblöcke lesen
  console.log('\\nHooking memory operations...');
  
  // Hook memcpy für große Kopien
  try {
    Process.enumerateModules().forEach(m => {
      try {
        const memcpy = Module.findExportByName(m.name, 'memcpy');
        if (memcpy && ptr(memcpy).toInt32() !== 0) {
          Interceptor.attach(memcpy, {
            onEnter(args) {
              try {
                const size = args[2].toInt32();
                if (size > 100000 && size < 1000000) {  // ~100KB - 1MB
                  const src = dump(args[1], Math.min(64, size));
                  if (src) {
                    const hexStr = hex(src, 64);
                    if (hexStr.startsWith(balHeader)) {
                      console.log('[memcpy] BAL-Daten gefunden! Größe: ' + size);
                      console.log('  Source: ' + args[1]);
                      console.log('  Header: ' + hexStr.substring(0, 32));
                    }
                  }
                }
              } catch (e) {}
            }
          });
        }
      } catch (e) {}
    });
    console.log('✓ Hooked memcpy');
  } catch (e) {
    console.log('✗ memcpy hook failed: ' + e);
  }
  
  // Hook malloc für große Allokationen
  try {
    Process.enumerateModules().forEach(m => {
      try {
        const malloc = Module.findExportByName(m.name, 'malloc');
        if (malloc && ptr(malloc).toInt32() !== 0) {
          Interceptor.attach(malloc, {
            onEnter(args) {
              const size = args[0].toInt32();
              if (size > 100000 && size < 1000000) {
                console.log('[malloc] Große Allokation: ' + size + ' bytes');
              }
            },
            onLeave(retval) {
              // Speichere Pointer für später
            }
          });
        }
      } catch (e) {}
    });
    console.log('✓ Hooked malloc');
  } catch (e) {
    console.log('✗ malloc hook failed: ' + e);
  }
  
  // Hook ObjC NSData operations
  try {
    if (ObjC && ObjC.classes) {
      const NSData = ObjC.classes.NSData;
      if (NSData) {
        // Hook dataWithContentsOfFile
        const dataWithContents = NSData['+ dataWithContentsOfFile:'];
        if (dataWithContents && dataWithContents.implementation) {
          Interceptor.attach(dataWithContents.implementation, {
            onEnter(args) {
              try {
                if (args[0] && !args[0].isNull()) {
                  const pathObj = ObjC.Object(args[0]);
                  if (pathObj) {
                    const path = pathObj.toString();
                    if (path && path.includes('.bal')) {
                      console.log('[NSData] Loading BAL: ' + path);
                    }
                  }
                }
              } catch (e) {}
            },
            onLeave(retval) {
              try {
                if (retval && !retval.isNull()) {
                  const data = ObjC.Object(retval);
                  if (data && data.length && data.length() > 100000) {
                    const dataPtr = data.bytes();
                    if (dataPtr) {
                      const header = dump(dataPtr, 16);
                      const hexStr = hex(header, 16);
                      if (hexStr === balHeader) {
                        console.log('[NSData] BAL-Daten geladen!');
                        console.log('  Größe: ' + data.length());
                        console.log('  Pointer: ' + dataPtr);
                        console.log('  Header: ' + hexStr);
                      }
                    }
                  }
                }
              } catch (e) {}
            }
          });
          console.log('✓ Hooked NSData dataWithContentsOfFile');
        }
      }
    }
  } catch (e) {
    console.log('✗ NSData hook failed: ' + e);
  }
  
  // Suche nach Verschlüsselungsfunktionen im SoundVision-Modul
  console.log('\\nSearching for encryption functions in SoundVision module...');
  try {
    const svModule = Process.findModuleByName('Soundvision');
    if (svModule) {
      console.log('Found SoundVision module @ ' + svModule.base);
      
      // Enumerate alle Funktionen
      Module.enumerateExports('Soundvision').forEach(exp => {
        const name = exp.name || '';
        if (name.includes('decrypt') || name.includes('Decrypt') ||
            name.includes('cipher') || name.includes('Cipher') ||
            name.includes('aes') || name.includes('AES') ||
            name.includes('crypto') || name.includes('Crypto') ||
            name.includes('balloon') || name.includes('Balloon') ||
            name.includes('bal') || name.includes('BAL')) {
          console.log('  Found: ' + name + ' @ ' + exp.address);
          
          // Hook die Funktion
          try {
            Interceptor.attach(exp.address, {
              onEnter(args) {
                console.log('[SoundVision::' + name + '] called');
                // Logge alle Argumente
                for (let i = 0; i < 8; i++) {
                  try {
                    const arg = args[i];
                    if (arg && !arg.isNull()) {
                      const val = dump(arg, 32);
                      if (val) {
                        const hexStr = hex(val, 32);
                        console.log('  arg[' + i + ']: ' + hexStr);
                      }
                    }
                  } catch (e) {}
                }
              },
              onLeave(retval) {
                try {
                  if (retval && !retval.isNull()) {
                    const val = dump(retval, 32);
                    if (val) {
                      console.log('  return: ' + hex(val, 32));
                    }
                  }
                } catch (e) {}
              }
            });
          } catch (e) {
            console.log('    Hook failed: ' + e);
          }
        }
      });
    }
  } catch (e) {
    console.log('SoundVision module search failed: ' + e);
  }
  
  console.log('\\n=== Hooks installed ===');
  console.log('Waiting for BAL operations...');
  console.log('Please select/change a speaker in SoundVision.');
}

setImmediate(install);
'''
    
    script_path = Path(__file__).parent / "hook_soundvision_enhanced.js"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def main():
    """Hauptfunktion."""
    logger.info("=== Kombiniertes Monitoring und Frida-Hooking ===\n")
    
    # Erstelle erweitertes Frida-Skript
    logger.info("Erstelle erweitertes Frida-Hook-Skript...")
    frida_script = create_enhanced_frida_script()
    logger.info(f"Frida-Skript erstellt: {frida_script}")
    
    # Starte File-Monitoring
    monitor = FileMonitor()
    monitor.start_monitoring()
    
    # Finde App-Bundle
    app_bundles = [
        Path("/Users/MGraf/Desktop/soundvision_resigned/Soundvision.app"),
        Path("/Applications/Soundvision.app"),
    ]
    
    app_bundle = None
    for bundle in app_bundles:
        if bundle.exists():
            app_bundle = bundle
            break
    
    if not app_bundle:
        logger.error("SoundVision App-Bundle nicht gefunden!")
        return
    
    logger.info(f"\nApp-Bundle: {app_bundle}")
    logger.info("\nStarte SoundVision mit FridaGadget...")
    
    # Starte App mit FridaGadget
    app_executable = app_bundle / "Contents" / "MacOS" / "Soundvision"
    gadget_path = app_bundle / "Contents" / "Frameworks" / "FridaGadget.dylib"
    
    if not gadget_path.exists():
        logger.error(f"FridaGadget nicht gefunden: {gadget_path}")
        logger.info("Bitte FridaGadget.dylib in Frameworks kopieren!")
        return
    
    # Starte App
    env = os.environ.copy()
    env['DYLD_INSERT_LIBRARIES'] = str(gadget_path)
    
    logger.info("Starte SoundVision...")
    app_process = subprocess.Popen(
        [str(app_executable)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    logger.info(f"SoundVision gestartet (PID: {app_process.pid})")
    logger.info("Warte 5 Sekunden auf App-Initialisierung...")
    time.sleep(5)
    
    # Prüfe ob App noch läuft
    if app_process.poll() is not None:
        logger.error(f"SoundVision wurde beendet! Exit-Code: {app_process.returncode}")
        logger.info("Versuche erneut zu starten...")
        return main()  # Rekursive Wiederholung
    
    # Starte Frida
    frida_bin = Path.home() / "Library" / "Python" / "3.9" / "bin" / "frida"
    if not frida_bin.exists():
        frida_bin = Path("/usr/local/bin/frida")
    
    if not frida_bin.exists():
        logger.error("Frida nicht gefunden!")
        return
    
    logger.info("\nVerbinde Frida mit Gadget...")
    logger.info("Frida läuft jetzt. Bitte in SoundVision einen Lautsprecher auswählen.")
    logger.info("Drücke Ctrl+C zum Beenden.\n")
    
    frida_process = None
    try:
        # Starte Frida
        frida_process = subprocess.Popen(
            [
                str(frida_bin),
                "-H", "127.0.0.1:27042",
                "-p", str(app_process.pid),
                "-l", str(frida_script)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Watchdog: Überwache App und Frida
        last_check = time.time()
        while True:
            # Prüfe ob App noch läuft
            if app_process.poll() is not None:
                logger.error(f"\nSoundVision wurde beendet! Exit-Code: {app_process.returncode}")
                logger.info("Versuche erneut zu starten in 3 Sekunden...")
                time.sleep(3)
                return main()  # Rekursive Wiederholung
            
            # Prüfe ob Frida noch läuft (nur wenn es das erste Mal ist)
            if frida_process and frida_process.poll() is not None:
                # Frida beendet sich normalerweise nach dem Ausführen des Skripts
                # Die Hooks laufen aber weiter im Prozess
                if frida_process.returncode == 0:
                    # Erfolgreich beendet - Hooks sind installiert
                    logger.info("Frida-Skript erfolgreich ausgeführt. Hooks sind aktiv.")
                    frida_process = None  # Setze auf None, damit wir nicht ständig neu verbinden
                else:
                    # Fehler - versuche erneut
                    logger.warning(f"Frida wurde mit Fehler beendet! Exit-Code: {frida_process.returncode}")
                    logger.info("Versuche Frida erneut zu verbinden in 5 Sekunden...")
                    time.sleep(5)
                    try:
                        frida_process = subprocess.Popen(
                            [
                                str(frida_bin),
                                "-H", "127.0.0.1:27042",
                                "-p", str(app_process.pid),
                                "-l", str(frida_script)
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1
                        )
                        logger.info("Frida erneut verbunden")
                    except Exception as e:
                        logger.error(f"Frida-Reconnect fehlgeschlagen: {e}")
                        time.sleep(5)
                        continue
            
            # Lese Frida-Output (mit Timeout)
            if frida_process and frida_process.stdout:
                try:
                    import select
                    ready, _, _ = select.select([frida_process.stdout], [], [], 0.1)
                    if ready:
                        line = frida_process.stdout.readline()
                        if line:
                            print(line, end='')
                            
                            # Prüfe auf Schlüssel-Funde
                            if 'key' in line.lower() or 'decrypt' in line.lower() or 'bal' in line.lower():
                                logger.info(f"[FOUND] {line.strip()}")
                except (OSError, ValueError):
                    # select() kann fehlschlagen, wenn stdout geschlossen ist
                    pass
            
            time.sleep(0.1)  # Kurze Pause
    
    except KeyboardInterrupt:
        logger.info("\n\nBeende...")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
    finally:
        # Stoppe Monitoring
        monitor.stop_monitoring()
        
        # Zeige Ergebnisse
        logger.info("\n=== Ergebnisse ===")
        results = monitor.get_results()
        if results:
            logger.info(f"Gefundene Dateien: {len(results)}")
            for r in results:
                logger.info(f"  {r['path']} ({r['size']:,} Bytes)")
        else:
            logger.info("Keine neuen Dateien gefunden")
        
        # Beende Prozesse
        if app_process.poll() is None:
            app_process.terminate()
        if 'frida_process' in locals() and frida_process.poll() is None:
            frida_process.terminate()


if __name__ == "__main__":
    main()

