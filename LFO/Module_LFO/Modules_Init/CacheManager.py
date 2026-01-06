"""
Cache Manager: Zentrale Verwaltung aller Caches mit individueller Kontrolle

Dieses Modul bietet eine zentrale Cache-Verwaltung, die:
- Globale Statistiken und Monitoring bereitstellt
- Individuelle Cache-Instanzen für verschiedene Komponenten verwaltet
- Jeden Cache individuell konfigurieren und verwalten lässt
- Thread-safe ist
- Cache-Invalidierung und -Bereinigung unterstützt
"""

from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum
import time
import sys
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class CacheType(Enum):
    """Typen von Caches im System"""
    GRID = "grid"
    CALC_GEOMETRY = "calc_geometry"
    CALC_GRID = "calc_grid"
    PLOT_SURFACE_ACTORS = "plot_surface_actors"
    PLOT_TEXTURE = "plot_texture"
    PLOT_GEOMETRY = "plot_geometry"


@dataclass
class CacheConfig:
    """Konfiguration für einen Cache"""
    cache_type: CacheType
    max_size: int = 1000  # Maximale Anzahl Einträge
    max_memory_mb: Optional[float] = None  # Maximale Memory in MB (None = unbegrenzt, nur Monitoring)
    enable_lru: bool = True
    enable_file_cache: bool = False
    file_cache_dir: Optional[str] = None
    description: str = ""
    memory_eviction_enabled: bool = False  # Memory-basierte Eviction aktivieren (False = nur Monitoring)
    ttl_seconds: Optional[float] = None  # Time-To-Live in Sekunden (None = kein TTL)
    max_idle_seconds: Optional[float] = None  # Max. Idle-Zeit in Sekunden (None = kein Idle-Limit)


@dataclass
class CacheStats:
    """Statistiken für einen Cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    file_loads: int = 0
    file_saves: int = 0
    last_access: Optional[float] = None
    total_accesses: int = 0
    memory_usage_mb: float = 0.0  # Geschätzte Memory-Nutzung in MB
    memory_evictions: int = 0  # Evictions aufgrund von Memory-Limit
    
    def hit_rate(self) -> float:
        """Berechnet die Hit-Rate als Prozent"""
        if self.total_accesses == 0:
            return 0.0
        return (self.hits / self.total_accesses) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Statistiken zu Dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'file_loads': self.file_loads,
            'file_saves': self.file_saves,
            'hit_rate': self.hit_rate(),
            'total_accesses': self.total_accesses,
            'last_access': self.last_access,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_evictions': self.memory_evictions,
        }


def estimate_memory_size(obj: Any) -> float:
    """
    Schätzt die Memory-Größe eines Objekts in MB.
    
    Args:
        obj: Objekt dessen Memory-Größe geschätzt werden soll
    
    Returns:
        Geschätzte Größe in MB
    """
    try:
        # NumPy Arrays
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.nbytes / (1024 * 1024)
        
        # CachedSurfaceGrid oder ähnliche Dataclasses
        if hasattr(obj, '__dict__'):
            total_bytes = sys.getsizeof(obj)
            for attr_name, attr_value in obj.__dict__.items():
                if NUMPY_AVAILABLE and isinstance(attr_value, np.ndarray):
                    total_bytes += attr_value.nbytes
                elif isinstance(attr_value, (list, tuple)):
                    for item in attr_value:
                        if NUMPY_AVAILABLE and isinstance(item, np.ndarray):
                            total_bytes += item.nbytes
                        else:
                            total_bytes += sys.getsizeof(item)
                else:
                    total_bytes += sys.getsizeof(attr_value)
            return total_bytes / (1024 * 1024)
        
        # Standard Python-Objekte
        return sys.getsizeof(obj) / (1024 * 1024)
    except Exception:
        # Fallback: Grobe Schätzung
        return sys.getsizeof(obj) / (1024 * 1024)


class LRUCache:
    """
    Thread-safe LRU-Cache Implementation mit Memory-Management
    
    Features:
    - Count-basierte Limits (max_size)
    - Memory-basierte Limits (max_memory_mb)
    - Automatische Memory-Schätzung
    - Memory-basierte Eviction
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: Optional[float] = None, 
                 memory_eviction_enabled: bool = False, name: str = "lru_cache",
                 ttl_seconds: Optional[float] = None, max_idle_seconds: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._memory_eviction_enabled = memory_eviction_enabled  # Memory-Eviction deaktiviert per Default
        self.name = name
        self.ttl_seconds = ttl_seconds
        self.max_idle_seconds = max_idle_seconds
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._memory_cache: Dict[Any, float] = {}  # Speichert Memory-Größe pro Key
        self._creation_times: Dict[Any, float] = {}  # Speichert Erstellungszeit pro Key (für TTL)
        self._last_access_times: Dict[Any, float] = {}  # Speichert letzte Zugriffszeit pro Key (für max_idle_seconds)
        self._lock = Lock()
        self._stats = CacheStats()
        self._ttl_seconds: Optional[float] = ttl_seconds  # Time-To-Live
        self._max_idle_seconds: Optional[float] = max_idle_seconds  # Max. Idle-Zeit
    
    def get(self, key: Any) -> Optional[Any]:
        """Holt Wert aus Cache mit LRU-Update und TTL/Idle-Prüfung"""
        with self._lock:
            self._stats.total_accesses += 1
            self._stats.last_access = time.time()
            
            if key in self._cache:
                # Prüfe TTL und Idle-Zeit
                current_time = time.time()
                
                # Prüfe TTL (Time-To-Live)
                if self._ttl_seconds is not None:
                    creation_time = self._creation_times.get(key, current_time)
                    if current_time - creation_time > self._ttl_seconds:
                        # Eintrag ist abgelaufen
                        self._remove_entry(key)
                        self._stats.misses += 1
                        return None
                
                # Prüfe Idle-Zeit (letzter Zugriff)
                if self._max_idle_seconds is not None:
                    last_access = self._last_access_times.get(key, current_time)
                    if current_time - last_access > self._max_idle_seconds:
                        # Eintrag war zu lange ungenutzt
                        self._remove_entry(key)
                        self._stats.misses += 1
                        return None
                
                # Update Zugriffszeit und LRU
                self._last_access_times[key] = current_time
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            
            self._stats.misses += 1
            return None
    
    def _remove_entry(self, key: Any) -> None:
        """Entfernt einen Eintrag aus dem Cache (intern)"""
        if key in self._cache:
            memory = self._memory_cache.pop(key, 0.0)
            self._last_access_times.pop(key, None)
            self._creation_times.pop(key, None)
            del self._cache[key]
            self._stats.memory_usage_mb -= memory
            self._stats.size = len(self._cache)
    
    def set(self, key: Any, value: Any) -> None:
        """
        Speichert Wert im Cache mit LRU-Eviction und Memory-Management
        
        Args:
            key: Cache-Key
            value: Cache-Wert
        """
        with self._lock:
            self._stats.last_access = time.time()
            
            # Schätze Memory-Größe des neuen Wertes
            value_memory_mb = estimate_memory_size(value)
            
            if key in self._cache:
                # Update existing entry
                old_memory = self._memory_cache.get(key, 0.0)
                self._cache.move_to_end(key)
                self._memory_cache[key] = value_memory_mb
                # Aktualisiere Memory-Statistik
                self._stats.memory_usage_mb += (value_memory_mb - old_memory)
            else:
                # New entry: check limits
                current_memory = self._stats.memory_usage_mb
                
                # 🎯 STRATEGIE: Memory-Limit nur für Monitoring, nicht für Eviction
                # Bei großen Surface-Anzahlen sollen wichtige Surfaces nicht gelöscht werden
                # Eviction erfolgt nur basierend auf Count-Limit (LRU)
                
                # Prüfe Count-Limit (hauptsächliche Eviction-Strategie)
                if len(self._cache) >= self.max_size:
                    # Remove oldest entry (LRU-Eviction)
                    oldest_key = next(iter(self._cache))
                    oldest_memory = self._memory_cache.pop(oldest_key, 0.0)
                    self._last_access_times.pop(oldest_key, None)
                    self._creation_times.pop(oldest_key, None)
                    del self._cache[oldest_key]
                    current_memory -= oldest_memory
                    self._stats.evictions += 1
                
                # 🎯 OPTIONAL: Memory-basierte Eviction (nur wenn explizit aktiviert)
                # Standard: Deaktiviert, um wichtige Surfaces nicht zu löschen
                if (hasattr(self, '_memory_eviction_enabled') and 
                    self._memory_eviction_enabled and 
                    self.max_memory_mb is not None):
                    # Entferne älteste Einträge bis Memory-Limit eingehalten wird
                    while (current_memory + value_memory_mb > self.max_memory_mb and 
                           len(self._cache) > 0):
                        oldest_key = next(iter(self._cache))
                        oldest_memory = self._memory_cache.pop(oldest_key, 0.0)
                        self._last_access_times.pop(oldest_key, None)
                        self._creation_times.pop(oldest_key, None)
                        del self._cache[oldest_key]
                        current_memory -= oldest_memory
                        self._stats.evictions += 1
                        self._stats.memory_evictions += 1
            
            # Füge neuen Eintrag hinzu
            current_time = time.time()
            self._cache[key] = value
            self._memory_cache[key] = value_memory_mb
            self._last_access_times[key] = current_time
            self._creation_times[key] = current_time
            self._stats.memory_usage_mb = current_memory + value_memory_mb
            self._stats.size = len(self._cache)
    
    def remove(self, key: Any) -> bool:
        """Entfernt einen Eintrag aus dem Cache"""
        with self._lock:
            return self._remove_entry(key)
    
    def cleanup_expired(self) -> int:
        """
        Bereinigt abgelaufene Einträge (TTL und Idle-Zeit).
        
        Returns:
            Anzahl entfernter Einträge
        """
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key in list(self._cache.keys()):
                should_remove = False
                
                # Prüfe TTL
                if self._ttl_seconds is not None:
                    creation_time = self._creation_times.get(key, current_time)
                    if current_time - creation_time > self._ttl_seconds:
                        should_remove = True
                
                # Prüfe Idle-Zeit
                if not should_remove and self._max_idle_seconds is not None:
                    last_access = self._last_access_times.get(key, current_time)
                    if current_time - last_access > self._max_idle_seconds:
                        should_remove = True
                
                if should_remove:
                    keys_to_remove.append(key)
            
            # Entferne abgelaufene Einträge
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def cleanup_unused_surfaces(self, valid_surface_ids: set) -> int:
        """
        Bereinigt Cache-Einträge für Surfaces, die nicht mehr existieren.
        
        Args:
            valid_surface_ids: Set von gültigen Surface-IDs
        
        Returns:
            Anzahl entfernter Einträge
        """
        with self._lock:
            keys_to_remove = []
            
            for key in list(self._cache.keys()):
                # Prüfe ob Key eine Surface-ID enthält
                if isinstance(key, tuple) and len(key) > 0:
                    surface_id = str(key[0])
                    if surface_id not in valid_surface_ids:
                        keys_to_remove.append(key)
            
            # Entferne Einträge für nicht-existierende Surfaces
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Leert den gesamten Cache"""
        with self._lock:
            self._cache.clear()
            self._memory_cache.clear()
            self._last_access_times.clear()
            self._creation_times.clear()
            self._stats.size = 0
            self._stats.memory_usage_mb = 0.0
    
    def get_stats(self) -> CacheStats:
        """Gibt Cache-Statistiken zurück"""
        with self._lock:
            self._stats.size = len(self._cache)
            # Aktualisiere Memory-Statistik (kann sich ändern wenn Objekte modifiziert wurden)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                file_loads=self._stats.file_loads,
                file_saves=self._stats.file_saves,
                last_access=self._stats.last_access,
                total_accesses=self._stats.total_accesses,
                memory_usage_mb=self._stats.memory_usage_mb,
                memory_evictions=self._stats.memory_evictions,
            )
    
    def reset_stats(self) -> None:
        """Setzt Statistiken zurück (behält Cache-Inhalt)"""
        with self._lock:
            self._stats = CacheStats()
            self._stats.size = len(self._cache)
    
    def size(self) -> int:
        """Gibt aktuelle Cache-Größe zurück"""
        with self._lock:
            return len(self._cache)


class CacheManager:
    """
    Zentrale Cache-Verwaltung mit individueller Kontrolle über jeden Cache
    
    Features:
    - Globale Verwaltung aller Caches
    - Individuelle Konfiguration pro Cache
    - Globale Statistiken
    - Gezielte Cache-Invalidierung
    - Thread-safe
    """
    
    _instance: Optional['CacheManager'] = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton-Pattern: Nur eine Instanz pro Prozess"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialisiert den Cache-Manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._caches: Dict[CacheType, LRUCache] = {}
        self._configs: Dict[CacheType, CacheConfig] = {}
        self._lock = Lock()
        self._initialized = True
    
    def register_cache(
        self,
        cache_type: CacheType,
        max_size: int = 1000,
        max_memory_mb: Optional[float] = None,
        memory_eviction_enabled: bool = False,
        enable_lru: bool = True,
        enable_file_cache: bool = False,
        file_cache_dir: Optional[str] = None,
        description: str = "",
        ttl_seconds: Optional[float] = None,
        max_idle_seconds: Optional[float] = None
    ) -> LRUCache:
        """
        Registriert einen neuen Cache mit individueller Konfiguration
        
        Args:
            cache_type: Typ des Caches
            max_size: Maximale Anzahl Einträge
            max_memory_mb: Maximale Memory in MB (None = unbegrenzt, nur Monitoring)
            memory_eviction_enabled: Memory-basierte Eviction aktivieren (False = nur Monitoring)
            enable_lru: LRU-Eviction aktivieren
            enable_file_cache: File-Cache aktivieren
            file_cache_dir: Verzeichnis für File-Cache
            description: Beschreibung des Caches
            ttl_seconds: Time-To-Live in Sekunden (None = kein TTL)
            max_idle_seconds: Maximale Idle-Zeit in Sekunden (None = kein Idle-Limit)
        
        Returns:
            LRUCache-Instanz
        """
        with self._lock:
            if cache_type in self._caches:
                # Cache existiert bereits, aktualisiere Konfiguration
                existing_cache = self._caches[cache_type]
                existing_cache.max_size = max_size
                if max_memory_mb is not None:
                    existing_cache.max_memory_mb = max_memory_mb
                existing_cache._memory_eviction_enabled = memory_eviction_enabled
                if ttl_seconds is not None:
                    existing_cache._ttl_seconds = ttl_seconds
                if max_idle_seconds is not None:
                    existing_cache._max_idle_seconds = max_idle_seconds
                return existing_cache
            
            # Erstelle neuen Cache
            config = CacheConfig(
                cache_type=cache_type,
                max_size=max_size,
                max_memory_mb=max_memory_mb,
                memory_eviction_enabled=memory_eviction_enabled,
                enable_lru=enable_lru,
                enable_file_cache=enable_file_cache,
                file_cache_dir=file_cache_dir,
                description=description,
                ttl_seconds=ttl_seconds,
                max_idle_seconds=max_idle_seconds,
            )
            
            cache = LRUCache(
                max_size=max_size, 
                max_memory_mb=max_memory_mb,
                memory_eviction_enabled=memory_eviction_enabled,
                name=cache_type.value,
                ttl_seconds=ttl_seconds,
                max_idle_seconds=max_idle_seconds
            )
            
            self._caches[cache_type] = cache
            self._configs[cache_type] = config
            
            return cache
    
    def get_cache(self, cache_type: CacheType) -> Optional[LRUCache]:
        """
        Gibt einen registrierten Cache zurück
        
        Args:
            cache_type: Typ des Caches
        
        Returns:
            LRUCache-Instanz oder None wenn nicht registriert
        """
        with self._lock:
            return self._caches.get(cache_type)
    
    def get_or_create_cache(
        self,
        cache_type: CacheType,
        max_size: int = 1000,
        **kwargs
    ) -> LRUCache:
        """
        Holt einen Cache oder erstellt ihn, falls er nicht existiert
        
        Args:
            cache_type: Typ des Caches
            max_size: Maximale Anzahl Einträge (nur bei Erstellung)
            **kwargs: Weitere Konfigurationsparameter
        
        Returns:
            LRUCache-Instanz
        """
        cache = self.get_cache(cache_type)
        if cache is None:
            cache = self.register_cache(cache_type, max_size=max_size, **kwargs)
        return cache
    
    def clear_cache(self, cache_type: CacheType) -> bool:
        """
        Leert einen spezifischen Cache
        
        Args:
            cache_type: Typ des zu leerenden Caches
        
        Returns:
            True wenn Cache geleert wurde, False wenn nicht gefunden
        """
        cache = self.get_cache(cache_type)
        if cache:
            cache.clear()
            return True
        return False
    
    def clear_all_caches(self) -> None:
        """Leert alle registrierten Caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def invalidate_cache(
        self,
        cache_type: CacheType,
        predicate: Optional[Callable[[Any], bool]] = None
    ) -> int:
        """
        Invalidiert Cache-Einträge basierend auf einem Prädikat
        
        Args:
            cache_type: Typ des Caches
            predicate: Funktion, die True zurückgibt für zu löschende Keys
        
        Returns:
            Anzahl gelöschter Einträge
        """
        cache = self.get_cache(cache_type)
        if not cache:
            return 0
        
        if predicate is None:
            # Wenn kein Prädikat: Cache komplett leeren
            cache.clear()
            return cache.size()
        
        # Gezielte Invalidierung basierend auf Prädikat
        with cache._lock:
            keys_to_remove = [
                key for key in cache._cache.keys()
                if predicate(key)
            ]
            for key in keys_to_remove:
                cache.remove(key)
            return len(keys_to_remove)
    
    def get_cache_stats(self, cache_type: Optional[CacheType] = None) -> Dict[str, Any]:
        """
        Gibt Statistiken für einen Cache oder alle Caches zurück
        
        Args:
            cache_type: Typ des Caches (None für alle)
        
        Returns:
            Dictionary mit Statistiken
        """
        with self._lock:
            if cache_type:
                cache = self._caches.get(cache_type)
                if cache:
                    config = self._configs.get(cache_type)
                    return {
                        cache_type.value: {
                            'stats': cache.get_stats().to_dict(),
                            'config': {
                                'max_size': config.max_size if config else None,
                                'max_memory_mb': config.max_memory_mb if config else None,
                                'enable_lru': config.enable_lru if config else None,
                                'description': config.description if config else '',
                            }
                        }
                    }
                return {}
            
            # Alle Caches
            result = {}
            for ct, cache in self._caches.items():
                config = self._configs.get(ct)
                result[ct.value] = {
                    'stats': cache.get_stats().to_dict(),
                    'config': {
                        'max_size': config.max_size if config else None,
                        'enable_lru': config.enable_lru if config else None,
                        'description': config.description if config else '',
                    }
                }
            return result
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Gibt globale Statistiken über alle Caches zurück
        
        Returns:
            Dictionary mit aggregierten Statistiken
        """
        all_stats = self.get_cache_stats()
        
        total_hits = sum(s['stats']['hits'] for s in all_stats.values())
        total_misses = sum(s['stats']['misses'] for s in all_stats.values())
        total_accesses = sum(s['stats']['total_accesses'] for s in all_stats.values())
        total_size = sum(s['stats']['size'] for s in all_stats.values())
        total_evictions = sum(s['stats']['evictions'] for s in all_stats.values())
        
        global_hit_rate = (total_hits / total_accesses * 100.0) if total_accesses > 0 else 0.0
        
        return {
            'total_caches': len(all_stats),
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_accesses': total_accesses,
            'total_size': total_size,
            'total_evictions': total_evictions,
            'global_hit_rate': global_hit_rate,
            'caches': all_stats,
        }
    
    def reset_stats(self, cache_type: Optional[CacheType] = None) -> None:
        """
        Setzt Statistiken zurück (behält Cache-Inhalt)
        
        Args:
            cache_type: Typ des Caches (None für alle)
        """
        with self._lock:
            if cache_type:
                cache = self._caches.get(cache_type)
                if cache:
                    cache.reset_stats()
            else:
                for cache in self._caches.values():
                    cache.reset_stats()
    
    def configure_cache(
        self,
        cache_type: CacheType,
        max_size: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        memory_eviction_enabled: Optional[bool] = None,
        enable_lru: Optional[bool] = None,
        enable_file_cache: Optional[bool] = None,
        file_cache_dir: Optional[str] = None,
        description: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        max_idle_seconds: Optional[float] = None,
    ) -> bool:
        """
        Konfiguriert einen bestehenden Cache
        
        Args:
            cache_type: Typ des Caches
            max_size: Neue maximale Größe
            max_memory_mb: Maximale Memory in MB
            memory_eviction_enabled: Memory-basierte Eviction aktivieren/deaktivieren
            enable_lru: LRU aktivieren/deaktivieren
            enable_file_cache: File-Cache aktivieren/deaktivieren
            file_cache_dir: File-Cache-Verzeichnis
            description: Neue Beschreibung
            ttl_seconds: Time-To-Live in Sekunden
            max_idle_seconds: Maximale Idle-Zeit in Sekunden
        
        Returns:
            True wenn Cache konfiguriert wurde, False wenn nicht gefunden
        """
        with self._lock:
            if cache_type not in self._caches:
                return False
            
            cache = self._caches[cache_type]
            config = self._configs.get(cache_type)
            
            if config is None:
                config = CacheConfig(cache_type=cache_type)
                self._configs[cache_type] = config
            
            if max_size is not None:
                cache.max_size = max_size
                config.max_size = max_size
            
            if max_memory_mb is not None:
                cache.max_memory_mb = max_memory_mb
                config.max_memory_mb = max_memory_mb
            
            if memory_eviction_enabled is not None:
                cache._memory_eviction_enabled = memory_eviction_enabled
                config.memory_eviction_enabled = memory_eviction_enabled
            
            if ttl_seconds is not None:
                cache._ttl_seconds = ttl_seconds
                config.ttl_seconds = ttl_seconds
            
            if max_idle_seconds is not None:
                cache._max_idle_seconds = max_idle_seconds
                config.max_idle_seconds = max_idle_seconds
            
            if enable_lru is not None:
                config.enable_lru = enable_lru
            
            if enable_file_cache is not None:
                config.enable_file_cache = enable_file_cache
            
            if file_cache_dir is not None:
                config.file_cache_dir = file_cache_dir
            
            if description is not None:
                config.description = description
            
            return True


# Globale Instanz für einfachen Zugriff
cache_manager = CacheManager()

