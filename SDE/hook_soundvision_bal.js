// Frida hook: log AES key/IV + buffers when SoundVision decrypts BAL
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
  console.log('=== SoundVision BAL Decryption Hook ===');
  
  // Liste alle geladenen Module
  console.log('\nLoaded modules:');
  const modules = Process.enumerateModules();
  for (const m of modules.slice(0, 20)) {
    console.log('  - ' + m.name + ' @ ' + m.base);
  }
  
  // Suche nach OpenSSL-Bibliotheken in allen Modulen
  const candidates = [
    'libcrypto.1.1.dylib', 
    'libcrypto.dylib', 
    'libssl.1.1.dylib',
    'libcrypto.3.dylib',
    'libssl.3.dylib'
  ];
  
  let mod = null;
  let dInit = null, dUpdate = null, dFinal = null;
  
  // Durchsuche alle Module
  for (const m of modules) {
    const name = m.name;
    if (candidates.includes(name) || name.includes('crypto') || name.includes('ssl')) {
      console.log('\nChecking module: ' + name);
      try {
        dInit = Module.findExportByName(name, 'EVP_DecryptInit_ex');
        dUpdate = Module.findExportByName(name, 'EVP_DecryptUpdate');
        dFinal = Module.findExportByName(name, 'EVP_DecryptFinal_ex');
        
        if (dInit && dUpdate) {
          mod = name;
          console.log('✓ Found EVP functions in ' + name);
          break;
        }
      } catch (e) {}
    }
  }
  
  // Hooke immer Datei-Operationen (auch wenn EVP gefunden wurde)
  console.log('\nHooking file operations...');
  
  // Hook fopen - suche in allen Modulen
  let fopenHooked = false;
  try {
    Process.enumerateModules().forEach(m => {
      if (fopenHooked) return;
      try {
        const fopen = Module.findExportByName(m.name, 'fopen');
        if (fopen && ptr(fopen).toInt32() !== 0) {
          Interceptor.attach(fopen, {
            onEnter(args) {
              try {
                if (args[0] && !args[0].isNull()) {
                  const path = Memory.readUtf8String(args[0]);
                  if (path && (path.includes('.bal') || path.toLowerCase().includes('balloon'))) {
                    console.log('[fopen] BAL file: ' + path);
                  }
                }
              } catch (e) {}
            }
          });
          console.log('✓ Hooked fopen in ' + m.name);
          fopenHooked = true;
        }
      } catch (e) {}
    });
    if (!fopenHooked) {
      console.log('✗ fopen not found');
    }
  } catch (e) {
    console.log('✗ fopen hook failed: ' + e);
  }
  
  // Hook open - suche in allen Modulen
  let openHooked = false;
  try {
    Process.enumerateModules().forEach(m => {
      if (openHooked) return;
      try {
        const open = Module.findExportByName(m.name, 'open');
        if (open && ptr(open).toInt32() !== 0) {
          Interceptor.attach(open, {
            onEnter(args) {
              try {
                if (args[0] && !args[0].isNull()) {
                  const path = Memory.readUtf8String(args[0]);
                  if (path && (path.includes('.bal') || path.toLowerCase().includes('balloon'))) {
                    console.log('[open] BAL file: ' + path);
                  }
                }
              } catch (e) {}
            }
          });
          console.log('✓ Hooked open in ' + m.name);
          openHooked = true;
        }
      } catch (e) {}
    });
    if (!openHooked) {
      console.log('✗ open not found');
    }
  } catch (e) {
    console.log('✗ open hook failed: ' + e);
  }
  
  // Hook NSData dataWithContentsOfFile (ObjC)
  try {
    if (ObjC && ObjC.classes) {
      const NSData = ObjC.classes.NSData;
      if (NSData) {
        const dataWithContents = NSData['+ dataWithContentsOfFile:'];
        if (dataWithContents && dataWithContents.implementation) {
          Interceptor.attach(dataWithContents.implementation, {
            onEnter(args) {
              try {
                if (args[0] && !args[0].isNull()) {
                  const pathObj = ObjC.Object(args[0]);
                  if (pathObj) {
                    const path = pathObj.toString();
                    if (path && (path.includes('.bal') || path.toLowerCase().includes('balloon'))) {
                      console.log('[NSData] Loading BAL file: ' + path);
                    }
                  }
                }
              } catch (e) {}
            },
            onLeave(retval) {
              try {
                if (retval && !retval.isNull()) {
                  const data = ObjC.Object(retval);
                  if (data && data.length && data.length() > 1000) {
                    const dataPtr = data.bytes();
                    if (dataPtr) {
                      const bytes = Memory.readByteArray(dataPtr, Math.min(64, data.length()));
                      const hexStr = hex(bytes, 64);
                      console.log('[NSData] Loaded ' + data.length() + ' bytes, first 64: ' + hexStr);
                    }
                  }
                }
              } catch (e) {}
            }
          });
          console.log('✓ Hooked NSData dataWithContentsOfFile');
        }
      }
    } else {
      console.log('✗ ObjC not available');
    }
  } catch (e) {
    console.log('✗ NSData hook failed: ' + e);
  }
  
  // Hook read operations
  try {
    let readHooked = false;
    Process.enumerateModules().forEach(m => {
      if (readHooked) return;
      try {
        const read = Module.findExportByName(m.name, 'read');
        if (read && ptr(read).toInt32() !== 0) {
          let readCount = 0;
          Interceptor.attach(read, {
            onEnter(args) {
              try {
                const count = args[2].toInt32();
                if (count > 50000) {
                  readCount++;
                  if (readCount % 10 === 0) {
                    console.log('[read] Large read #' + readCount + ': ' + count + ' bytes');
                  }
                }
              } catch (e) {}
            }
          });
          console.log('✓ Hooked read in ' + m.name);
          readHooked = true;
        }
      } catch (e) {}
    });
    if (!readHooked) {
      console.log('✗ read not found');
    }
  } catch (e) {
    console.log('✗ read hook failed: ' + e);
  }
  
  // Suche nach SoundVision-spezifischen Funktionen
  try {
    console.log('\nSearching for SoundVision decrypt/crypto functions...');
    const foundFuncs = [];
    Process.enumerateModules().forEach(m => {
      try {
        Module.enumerateExports(m.name).forEach(exp => {
          const name = exp.name || '';
          if (name.includes('decrypt') || name.includes('Decrypt') || 
              name.includes('Cipher') || name.includes('cipher') ||
              name.includes('AES') || name.includes('aes') ||
              name.includes('crypto') || name.includes('Crypto')) {
            foundFuncs.push({module: m.name, func: name, addr: exp.address});
          }
        });
      } catch (e) {}
    });
    
    if (foundFuncs.length > 0) {
      console.log('Found ' + foundFuncs.length + ' crypto functions:');
      foundFuncs.slice(0, 20).forEach(f => {
        console.log('  ' + f.module + '::' + f.func + ' @ ' + f.addr);
      });
    } else {
      console.log('No crypto functions found in exports');
    }
  } catch (e) {
    console.log('✗ Crypto function search failed: ' + e);
  }
  
  // Suche nach BAL-Dateien im App-Bundle (ObjC)
  console.log('\nSearching for BAL files in app bundle...');
  try {
    if (ObjC && ObjC.classes) {
      const NSBundle = ObjC.classes.NSBundle;
      if (NSBundle) {
        const mainBundle = NSBundle['+ mainBundle'];
        if (mainBundle) {
          const bundleObj = mainBundle();
          if (bundleObj) {
            const bundle = new ObjC.Object(bundleObj);
            const resourcePath = bundle.resourcePath();
            if (resourcePath) {
              const bundlePath = resourcePath.toString();
              console.log('Bundle path: ' + bundlePath);
              
              // Versuche BAL-Dateien zu finden
              const NSFileManager = ObjC.classes.NSFileManager;
              if (NSFileManager) {
                const defaultManager = NSFileManager['+ defaultManager'];
                if (defaultManager) {
                  const fm = new ObjC.Object(defaultManager());
                  const enumerator = fm.enumeratorAtPath_(bundlePath);
                  if (enumerator) {
                    let balCount = 0;
                    let path;
                    while ((path = enumerator.nextObject()) !== null) {
                      const pathStr = path.toString();
                      if (pathStr && pathStr.includes('.bal')) {
                        console.log('  Found BAL: ' + pathStr);
                        balCount++;
                        if (balCount >= 10) break;
                      }
                    }
                    if (balCount === 0) {
                      console.log('  No .bal files found in bundle');
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } catch (e) {
    console.log('Bundle search failed: ' + e);
  }
  
  // Hook mmap für Memory-Mapped Files
  for (const modName of ['libSystem.B.dylib', 'libc.dylib']) {
    try {
      const mmap = Module.findExportByName(modName, 'mmap');
      if (mmap) {
        Interceptor.attach(mmap, {
          onEnter(args) {
            const length = args[1].toInt32();
            if (length > 10000) {
              console.log('[mmap] Mapping ' + length + ' bytes');
            }
          }
        });
        console.log('✓ Hooked mmap');
        break;
      }
    } catch (e) {}
  }
  
  // Hook stat/lstat für Datei-Statistiken
  for (const modName of ['libSystem.B.dylib', 'libc.dylib']) {
    try {
      const stat = Module.findExportByName(modName, 'stat');
      if (stat) {
        Interceptor.attach(stat, {
          onEnter(args) {
            try {
              const path = Memory.readUtf8String(args[0]);
              if (path && (path.includes('.bal') || path.toLowerCase().includes('balloon'))) {
                console.log('[stat] Checking BAL file: ' + path);
              }
            } catch (e) {}
          }
        });
        console.log('✓ Hooked stat');
        break;
      }
    } catch (e) {}
  }
  
  // Hook alle Datei-Operationen im SoundVision-Modul
  console.log('\nHooking SoundVision module functions...');
  try {
    const soundvisionModule = Process.findModuleByName('Soundvision');
    if (soundvisionModule) {
      console.log('Found SoundVision module @ ' + soundvisionModule.base);
      // Enumerate alle Exports und hooke die, die interessant klingen
      Module.enumerateExports('Soundvision').forEach(exp => {
        const name = exp.name || '';
        if (name.includes('File') || name.includes('file') || 
            name.includes('Load') || name.includes('load') ||
            name.includes('Read') || name.includes('read') ||
            name.includes('Data') || name.includes('data')) {
          try {
            Interceptor.attach(exp.address, {
              onEnter(args) {
                console.log('[SoundVision::' + name + '] called');
              }
            });
          } catch (e) {}
        }
      });
    }
  } catch (e) {
    console.log('SoundVision module hook failed: ' + e);
  }
  
  console.log('\n=== Hook Setup Complete ===');
  if (!dInit || !dUpdate) {
    console.log('⚠ EVP functions not found - using file operation hooks only');
  } else {
    console.log('✓ EVP functions found and hooked');
  }
  console.log('Waiting for BAL file operations...');
  console.log('Please select/change a speaker in SoundVision.');
  console.log('All file operations will be logged.');
  
  if (!dInit || !dUpdate) {
    return;
  }

  let cur = {};

  Interceptor.attach(dInit, {
    onEnter(args) {
      // EVP_DecryptInit_ex(ctx, cipher, engine, key, iv)
      const keyPtr = args[3], ivPtr = args[4];
      cur = {
        key: keyPtr ? hex(dump(keyPtr, 32), 32) : null,
        iv: ivPtr ? hex(dump(ivPtr, 16), 16) : null
      };
      console.log('[DecryptInit] key=' + cur.key + ' iv=' + cur.iv);
    }
  });

  Interceptor.attach(dUpdate, {
    onEnter(args) {
      // EVP_DecryptUpdate(ctx, out, outl, in, inl)
      const inPtr = args[3];
      const inLen = args[4].toInt32();
      cur.inLen = inLen;
      cur.inHex = hex(dump(inPtr, Math.min(64, inLen)), Math.min(64, inLen));
      cur.outPtr = args[1];
    },
    onLeave(retval) {
      try {
        const outBytes = dump(cur.outPtr, cur.inLen);
        console.log('[DecryptUpdate] inLen=' + cur.inLen +
          ' inHex=' + cur.inHex +
          ' outHex=' + (outBytes ? hex(outBytes, 64) : 'n/a'));
      } catch (e) {
        console.log('read out failed', e);
      }
    }
  });

  if (dFinal) {
    Interceptor.attach(dFinal, { onEnter() { console.log('[DecryptFinal_ex]'); }});
  }

  console.log('Hooks set on ' + mod);
}

setImmediate(install);
