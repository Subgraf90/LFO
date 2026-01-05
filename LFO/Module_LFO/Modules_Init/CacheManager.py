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
    max_size: int = 1000
    enable_lru: bool = True
    enable_file_cache: bool = False
    file_cache_dir: Optional[str] = None
    description: str = ""


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
        }


class LRUCache:
    """
    Thread-safe LRU-Cache Implementation
    
    Kann als Basis für spezifische Cache-Implementierungen verwendet werden.
    """
    
    def __init__(self, max_size: int = 1000, name: str = "lru_cache"):
        self.max_size = max_size
        self.name = name
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats()
    
    def get(self, key: Any) -> Optional[Any]:
        """Holt Wert aus Cache mit LRU-Update"""
        with self._lock:
            self._stats.total_accesses += 1
            self._stats.last_access = time.time()
            
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            
            self._stats.misses += 1
            return None
    
    def set(self, key: Any, value: Any) -> None:
        """Speichert Wert im Cache mit LRU-Eviction"""
        with self._lock:
            self._stats.last_access = time.time()
            
            if key in self._cache:
                # Update existing entry
                self._cache.move_to_end(key)
            else:
                # New entry: check if cache limit reached
                if len(self._cache) >= self.max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._stats.evictions += 1
            
            self._cache[key] = value
            self._stats.size = len(self._cache)
    
    def remove(self, key: Any) -> bool:
        """Entfernt einen Eintrag aus dem Cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Leert den gesamten Cache"""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Gibt Cache-Statistiken zurück"""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                file_loads=self._stats.file_loads,
                file_saves=self._stats.file_saves,
                last_access=self._stats.last_access,
                total_accesses=self._stats.total_accesses,
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
        enable_lru: bool = True,
        enable_file_cache: bool = False,
        file_cache_dir: Optional[str] = None,
        description: str = ""
    ) -> LRUCache:
        """
        Registriert einen neuen Cache mit individueller Konfiguration
        
        Args:
            cache_type: Typ des Caches
            max_size: Maximale Anzahl Einträge
            enable_lru: LRU-Eviction aktivieren
            enable_file_cache: File-Cache aktivieren
            file_cache_dir: Verzeichnis für File-Cache
            description: Beschreibung des Caches
        
        Returns:
            LRUCache-Instanz
        """
        with self._lock:
            if cache_type in self._caches:
                # Cache existiert bereits, aktualisiere Konfiguration
                existing_cache = self._caches[cache_type]
                existing_cache.max_size = max_size
                return existing_cache
            
            # Erstelle neuen Cache
            config = CacheConfig(
                cache_type=cache_type,
                max_size=max_size,
                enable_lru=enable_lru,
                enable_file_cache=enable_file_cache,
                file_cache_dir=file_cache_dir,
                description=description,
            )
            
            cache = LRUCache(max_size=max_size, name=cache_type.value)
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
        enable_lru: Optional[bool] = None,
        enable_file_cache: Optional[bool] = None,
        file_cache_dir: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Konfiguriert einen bestehenden Cache
        
        Args:
            cache_type: Typ des Caches
            max_size: Neue maximale Größe
            enable_lru: LRU aktivieren/deaktivieren
            enable_file_cache: File-Cache aktivieren/deaktivieren
            file_cache_dir: File-Cache-Verzeichnis
            description: Neue Beschreibung
        
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

