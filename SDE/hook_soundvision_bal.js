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
  const candidates = ['libcrypto.1.1.dylib', 'libcrypto.dylib', 'libssl.1.1.dylib'];
  let mod = null;
  for (const name of candidates) {
    try {
      if (Module.getBaseAddress(name)) { mod = name; break; }
    } catch (e) {}
  }
  if (!mod) { console.log('libcrypto not found'); return; }

  const dInit   = Module.findExportByName(mod, 'EVP_DecryptInit_ex');
  const dUpdate = Module.findExportByName(mod, 'EVP_DecryptUpdate');
  const dFinal  = Module.findExportByName(mod, 'EVP_DecryptFinal_ex');
  if (!dInit || !dUpdate) { console.log('EVP hooks not found'); return; }

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
