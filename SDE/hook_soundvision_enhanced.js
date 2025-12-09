// Erweiterte Frida-Hooks für SoundVision BAL-Entschlüsselung
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
  console.log('\nHooking memory operations...');
  
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
  console.log('\nSearching for encryption functions in SoundVision module...');
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
  
  console.log('\n=== Hooks installed ===');
  console.log('Waiting for BAL operations...');
  console.log('Please select/change a speaker in SoundVision.');
}

setImmediate(install);
