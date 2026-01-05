# RAM-Management-Strategien: Wie Programme mit zu viel RAM umgehen

## Übersicht

Programme verwenden verschiedene Strategien, um RAM-Überlastung zu vermeiden. Die Wahl hängt von der Anwendung ab (Desktop-App, Web-App, Server, etc.).

---

## 1. Präventive Strategien (Verhindern)

### 1.1 Memory-Limits setzen

**Ansatz:** Maximalen Memory-Verbrauch begrenzen

```python
# Beispiel: Cache mit Memory-Limit
cache = LRUCache(max_memory_mb=500.0)  # Max. 500 MB

# Bei Überschreitung: Älteste Einträge entfernen
if current_memory + new_entry_memory > max_memory_mb:
    evict_oldest_entries()
```

**Vorteile:**
- Verhindert unbegrenztes Wachstum
- Vorhersagbarer Memory-Verbrauch
- Automatische Bereinigung

**Nachteile:**
- Kann wichtige Daten entfernen
- Muss sorgfältig konfiguriert werden

**Verwendung:**
- Caches (LRU, LFU)
- Datenstrukturen mit begrenzter Größe
- Temporäre Daten

---

### 1.2 Monitoring & Warnungen

**Ansatz:** Memory-Verbrauch überwachen und warnen

```python
# Beispiel: System-RAM-Monitoring
import psutil

mem = psutil.virtual_memory()
if mem.percent > 80:
    print("⚠️  Warnung: RAM-Auslastung > 80%")
    # Reaktive Maßnahmen einleiten
```

**Vorteile:**
- Frühe Erkennung von Problemen
- Zeit für Reaktion
- Keine Datenverluste

**Nachteile:**
- Reagiert erst nach Überschreitung
- Benötigt manuelle Intervention

**Verwendung:**
- Desktop-Anwendungen
- Server-Monitoring
- Debugging-Tools

---

### 1.3 Adaptive Limits

**Ansatz:** Limits basierend auf verfügbarem RAM anpassen

```python
# Beispiel: Adaptive Cache-Größe
import psutil

def get_adaptive_cache_size():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    if available_gb > 8:
        return 1000.0  # 1 GB Cache
    elif available_gb > 4:
        return 500.0   # 500 MB Cache
    else:
        return 100.0   # 100 MB Cache
```

**Vorteile:**
- Nutzt verfügbaren RAM optimal
- Passt sich an System an
- Benutzerfreundlich

**Nachteile:**
- Komplexere Implementierung
- Kann bei RAM-Änderungen Probleme verursachen

**Verwendung:**
- Desktop-Anwendungen
- Mobile Apps
- Multi-Tenant-Systeme

---

## 2. Reaktive Strategien (Bereinigung)

### 2.1 LRU-Eviction (Least Recently Used)

**Ansatz:** Entfernt am wenigsten genutzte Einträge

```python
# Beispiel: LRU-Cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=1000):
        self._cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)  # Markiere als zuletzt verwendet
            return self._cache[key]
        return None
    
    def set(self, key, value):
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Entferne ältesten Eintrag
        self._cache[key] = value
```

**Vorteile:**
- Einfach zu implementieren
- Effektiv für häufig genutzte Daten
- Automatische Bereinigung

**Nachteile:**
- Kann wichtige, selten genutzte Daten entfernen
- Nicht optimal für alle Anwendungsfälle

**Verwendung:**
- Browser-Caches
- Datenbank-Caches
- Application-Caches

---

### 2.2 LFU-Eviction (Least Frequently Used)

**Ansatz:** Entfernt am seltensten genutzte Einträge

```python
# Beispiel: LFU-Cache
class LFUCache:
    def __init__(self, max_size=1000):
        self._cache = {}
        self._access_count = {}  # Zähle Zugriffe
        self.max_size = max_size
    
    def get(self, key):
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None
    
    def set(self, key, value):
        if len(self._cache) >= self.max_size:
            # Entferne Eintrag mit niedrigstem Zugriffszähler
            least_used = min(self._access_count.items(), key=lambda x: x[1])
            del self._cache[least_used[0]]
            del self._access_count[least_used[0]]
        self._cache[key] = value
        self._access_count[key] = 0
```

**Vorteile:**
- Behält häufig genutzte Daten
- Besser für langfristige Nutzung

**Nachteile:**
- Komplexer als LRU
- Kann "Stale Data" behalten

**Verwendung:**
- Datenbank-Caches
- CDN-Caches
- Application-Caches

---

### 2.3 TTL-basierte Bereinigung (Time-To-Live)

**Ansatz:** Entfernt Einträge nach Ablaufzeit

```python
# Beispiel: TTL-Cache
import time

class TTLCache:
    def __init__(self, ttl_seconds=3600):
        self._cache = {}
        self._creation_times = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key):
        if key in self._cache:
            if time.time() - self._creation_times[key] > self.ttl_seconds:
                # Abgelaufen - entfernen
                del self._cache[key]
                del self._creation_times[key]
                return None
            return self._cache[key]
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._creation_times[key] = time.time()
```

**Vorteile:**
- Automatische Bereinigung alter Daten
- Verhindert "Stale Data"
- Einfach zu implementieren

**Nachteile:**
- Kann wichtige Daten entfernen
- Nicht optimal für alle Anwendungsfälle

**Verwendung:**
- Session-Caches
- Temporäre Daten
- API-Response-Caches

---

### 2.4 Idle-Time-basierte Bereinigung

**Ansatz:** Entfernt ungenutzte Einträge nach Idle-Zeit

```python
# Beispiel: Idle-Time-Cache
import time

class IdleTimeCache:
    def __init__(self, max_idle_seconds=1800):
        self._cache = {}
        self._last_access = {}
        self.max_idle_seconds = max_idle_seconds
    
    def get(self, key):
        if key in self._cache:
            current_time = time.time()
            if current_time - self._last_access[key] > self.max_idle_seconds:
                # Zu lange ungenutzt - entfernen
                del self._cache[key]
                del self._last_access[key]
                return None
            self._last_access[key] = current_time
            return self._cache[key]
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._last_access[key] = time.time()
```

**Vorteile:**
- Entfernt nur ungenutzte Daten
- Behält aktive Daten
- Gut für interaktive Anwendungen

**Nachteile:**
- Kann bei sporadischer Nutzung Probleme verursachen
- Komplexer als TTL

**Verwendung:**
- Desktop-Anwendungen
- Interaktive Tools
- User-Session-Caches

---

## 3. Architektur-Strategien

### 3.1 Lazy Loading (Lazy Evaluation)

**Ansatz:** Daten erst laden, wenn sie benötigt werden

```python
# Beispiel: Lazy Loading
class LazyDataLoader:
    def __init__(self):
        self._cache = {}
        self._loaded_keys = set()
    
    def get(self, key):
        if key not in self._loaded_keys:
            # Lade Daten erst jetzt
            self._cache[key] = self._load_from_disk(key)
            self._loaded_keys.add(key)
        return self._cache[key]
    
    def _load_from_disk(self, key):
        # Lade von Disk/DB
        pass
```

**Vorteile:**
- Spart RAM für ungenutzte Daten
- Schneller Start
- Skalierbar

**Nachteile:**
- Kann Latenz bei erstem Zugriff verursachen
- Komplexere Implementierung

**Verwendung:**
- Datenbanken
- File-System-Caches
- Große Datensätze

---

### 3.2 Streaming & Chunking

**Ansatz:** Große Daten in kleinen Chunks verarbeiten

```python
# Beispiel: Streaming-Verarbeitung
def process_large_file(file_path, chunk_size=1024*1024):  # 1 MB Chunks
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            process_chunk(chunk)  # Verarbeite Chunk
            # Chunk wird automatisch freigegeben
```

**Vorteile:**
- Konstanter Memory-Verbrauch
- Kann sehr große Dateien verarbeiten
- Skalierbar

**Nachteile:**
- Nicht für alle Anwendungsfälle geeignet
- Kann langsamer sein

**Verwendung:**
- File-Processing
- Datenbank-Abfragen
- Video/Image-Processing

---

### 3.3 Komprimierung

**Ansatz:** Daten komprimieren, um RAM zu sparen

```python
# Beispiel: Komprimierung
import gzip
import pickle

def compress_data(data):
    pickled = pickle.dumps(data)
    compressed = gzip.compress(pickled)
    return compressed

def decompress_data(compressed):
    pickled = gzip.decompress(compressed)
    return pickle.loads(pickled)

# Speichere komprimiert
compressed = compress_data(large_data)
# Später: Dekomprimiere bei Bedarf
data = decompress_data(compressed)
```

**Vorteile:**
- Spart RAM (oft 50-90% Reduktion)
- Behält alle Daten
- Gut für Archive

**Nachteile:**
- CPU-Overhead beim Komprimieren/Dekomprimieren
- Nicht für alle Datentypen geeignet

**Verwendung:**
- Backup-Systeme
- Archive
- Langzeit-Speicherung

---

### 3.4 Swapping (Auslagern auf Disk)

**Ansatz:** Weniger genutzte Daten auf Disk auslagern

```python
# Beispiel: Swap-Cache
import pickle
import os

class SwapCache:
    def __init__(self, max_memory_mb=500, swap_dir="/tmp/swap"):
        self._memory_cache = {}
        self._swap_dir = swap_dir
        self.max_memory_mb = max_memory_mb
        os.makedirs(swap_dir, exist_ok=True)
    
    def get(self, key):
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Lade von Disk
        swap_file = os.path.join(self._swap_dir, f"{key}.pkl")
        if os.path.exists(swap_file):
            with open(swap_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key, value):
        memory_mb = estimate_memory_size(value)
        if memory_mb > self.max_memory_mb:
            # Zu groß für RAM - auf Disk speichern
            swap_file = os.path.join(self._swap_dir, f"{key}.pkl")
            with open(swap_file, 'wb') as f:
                pickle.dump(value, f)
        else:
            self._memory_cache[key] = value
```

**Vorteile:**
- Kann sehr große Datenmengen handhaben
- Nutzt Disk-Speicher
- Transparent für Anwendung

**Nachteile:**
- Langsam (Disk-I/O)
- Benötigt Disk-Speicher
- Komplexe Implementierung

**Verwendung:**
- Betriebssystem-Swapping
- Datenbanken
- Große Datensätze

---

## 4. Spezifische Anwendungsfälle

### 4.1 Desktop-Anwendungen

**Strategien:**
- **Memory-Limits** für Caches
- **LRU-Eviction** für temporäre Daten
- **Monitoring** mit Warnungen
- **Adaptive Limits** basierend auf verfügbarem RAM

**Beispiel:**
```python
# Kombination aus mehreren Strategien
cache = LRUCache(
    max_size=1000,
    max_memory_mb=500.0,
    ttl_seconds=3600,
    max_idle_seconds=1800
)
```

---

### 4.2 Web-Anwendungen

**Strategien:**
- **TTL-basierte Bereinigung** für Session-Daten
- **LRU-Eviction** für Response-Caches
- **Memory-Limits** pro Request
- **Connection-Pooling** mit Limits

**Beispiel:**
```python
# Redis-ähnlicher Cache
cache = TTLCache(ttl_seconds=300)  # 5 Minuten TTL
session_cache = IdleTimeCache(max_idle_seconds=1800)  # 30 Min Idle
```

---

### 4.3 Server-Anwendungen

**Strategien:**
- **Memory-Monitoring** mit Alerts
- **Automatische Skalierung** bei hohem RAM-Verbrauch
- **Connection-Limits** pro Prozess
- **Graceful Degradation** bei RAM-Mangel

**Beispiel:**
```python
# Monitoring mit Alerts
if memory_usage > 80:
    send_alert("High memory usage")
    enable_degraded_mode()
```

---

### 4.4 Mobile Apps

**Strategien:**
- **Strikte Memory-Limits**
- **Aggressive Bereinigung** bei Background-Wechsel
- **Image-Caching** mit Größenlimits
- **Memory-Warnings** vom OS beachten

**Beispiel:**
```python
# iOS/Android Memory-Warnings
def on_memory_warning():
    clear_all_caches()
    reduce_image_quality()
    free_unused_resources()
```

---

## 5. Best Practices

### 5.1 Kombination mehrerer Strategien

**Empfehlung:** Kombiniere mehrere Strategien für optimale Ergebnisse

```python
# Beispiel: Multi-Strategy Cache
class SmartCache:
    def __init__(self):
        self.max_size = 1000
        self.max_memory_mb = 500.0
        self.ttl_seconds = 3600
        self.max_idle_seconds = 1800
    
    def get(self, key):
        # Prüfe TTL
        if self._is_expired(key):
            self.remove(key)
            return None
        
        # Prüfe Idle-Time
        if self._is_idle_too_long(key):
            self.remove(key)
            return None
        
        # LRU-Update
        self._update_lru(key)
        return self._cache[key]
    
    def set(self, key, value):
        # Prüfe Memory-Limit
        if self._exceeds_memory_limit(value):
            self._evict_oldest()
        
        # Prüfe Count-Limit
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = value
```

---

### 5.2 Monitoring & Logging

**Empfehlung:** Überwache Memory-Verbrauch kontinuierlich

```python
# Beispiel: Memory-Monitoring
import logging
import psutil

def monitor_memory():
    mem = psutil.virtual_memory()
    
    if mem.percent > 90:
        logging.critical(f"Critical: RAM usage {mem.percent}%")
    elif mem.percent > 80:
        logging.warning(f"Warning: RAM usage {mem.percent}%")
    else:
        logging.info(f"Info: RAM usage {mem.percent}%")
```

---

### 5.3 Graceful Degradation

**Empfehlung:** Reduziere Funktionalität bei RAM-Mangel

```python
# Beispiel: Graceful Degradation
def handle_memory_pressure():
    if memory_usage > 90:
        # Kritisch: Reduziere Funktionalität
        disable_non_essential_features()
        clear_all_caches()
    elif memory_usage > 80:
        # Warnung: Reduziere Cache-Größen
        reduce_cache_sizes()
        enable_aggressive_cleanup()
```

---

## 6. Zusammenfassung

### Strategien nach Anwendungsfall

| Anwendung | Präferierte Strategien |
|-----------|----------------------|
| **Desktop-App** | LRU-Eviction, Memory-Limits, Monitoring |
| **Web-App** | TTL-Cache, LRU-Eviction, Connection-Limits |
| **Server** | Monitoring, Auto-Scaling, Graceful Degradation |
| **Mobile** | Strikte Limits, Aggressive Cleanup, OS-Warnings |
| **Datenbank** | LRU/LFU-Eviction, Lazy Loading, Swapping |

### Kombination für LFO-Projekt

**Empfohlene Strategien:**
1. ✅ **Memory-basierte Limits** (bereits implementiert)
2. ✅ **LRU-Eviction** (bereits implementiert)
3. ✅ **Monitoring** (bereits implementiert)
4. ⚠️ **Snapshot-Limits** (noch zu implementieren)
5. ⚠️ **Graceful Degradation** (optional)

**Nächste Schritte:**
- Snapshot-Count-Limit implementieren
- Snapshot-Memory-Monitoring aktivieren
- Warnungen bei hohem RAM-Verbrauch

---

## 7. Referenzen

- **LRU-Cache:** Wikipedia - Least Recently Used
- **Memory-Management:** Operating System Concepts
- **psutil:** Python System and Process Utilities
- **Redis:** In-Memory Data Structure Store

Die Wahl der Strategie hängt von der Anwendung ab - für LFO sind **Memory-Limits + LRU-Eviction + Monitoring** optimal! 🚀

