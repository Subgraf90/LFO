"""
Cache Monitor: Debugging und Monitoring Tool für Cache-Performance

Verwendung:
    from Module_LFO.Modules_Init.CacheMonitor import CacheMonitor
    
    monitor = CacheMonitor()
    monitor.print_stats()
    monitor.print_detailed_stats()
    monitor.print_memory_overview()  # System-RAM + Cache-RAM
"""

from typing import Dict, Any, Optional
from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType, estimate_memory_size
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CacheMonitor:
    """Monitor für Cache-Performance und Debugging"""
    
    def __init__(self):
        self.cache_manager = cache_manager
    
    def print_stats(self, cache_type: Optional[CacheType] = None):
        """
        Druckt Cache-Statistiken (kompakt).
        
        Args:
            cache_type: Typ des Caches (None für alle)
        """
        if cache_type:
            stats = self.cache_manager.get_cache_stats(cache_type)
            self._print_cache_stats(cache_type.value, stats[cache_type.value])
        else:
            global_stats = self.cache_manager.get_global_stats()
            print("\n" + "="*60)
            print("CACHE-STATISTIKEN")
            print("="*60)
            print(f"Global Hit-Rate: {global_stats['global_hit_rate']:.2f}%")
            print(f"Total Caches: {global_stats['total_caches']}")
            print(f"Total Size: {global_stats['total_size']} entries")
            print(f"Total Hits: {global_stats['total_hits']}")
            print(f"Total Misses: {global_stats['total_misses']}")
            print("\nPer-Cache Stats:")
            for cache_name, data in global_stats['caches'].items():
                cache_stats = data['stats']
                print(f"  {cache_name}:")
                print(f"    Hit-Rate: {cache_stats['hit_rate']:.2f}%")
                print(f"    Size: {cache_stats['size']}/{data['config']['max_size']}")
                print(f"    Memory: {cache_stats.get('memory_usage_mb', 0.0):.1f} MB")
                print(f"    Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    
    def print_detailed_stats(self, cache_type: Optional[CacheType] = None):
        """
        Druckt detaillierte Cache-Statistiken.
        
        Args:
            cache_type: Typ des Caches (None für alle)
        """
        if cache_type:
            stats = self.cache_manager.get_cache_stats(cache_type)
            self._print_detailed_cache_stats(cache_type.value, stats[cache_type.value])
        else:
            all_stats = self.cache_manager.get_cache_stats()
            print("\n" + "="*60)
            print("DETAILLIERTE CACHE-STATISTIKEN")
            print("="*60)
            for cache_name, data in all_stats.items():
                self._print_detailed_cache_stats(cache_name, data)
    
    def _print_cache_stats(self, cache_name: str, data: Dict[str, Any]):
        """Druckt Statistiken für einen Cache"""
        stats = data['stats']
        config = data['config']
        print(f"\n{cache_name}:")
        print(f"  Hit-Rate: {stats['hit_rate']:.2f}%")
        print(f"  Size: {stats['size']}/{config['max_size']}")
        print(f"  Memory: {stats.get('memory_usage_mb', 0.0):.1f} MB")
        if config.get('max_memory_mb'):
            print(f"  Memory-Limit: {config['max_memory_mb']:.1f} MB")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    
    def _print_detailed_cache_stats(self, cache_name: str, data: Dict[str, Any]):
        """Druckt detaillierte Statistiken für einen Cache"""
        stats = data['stats']
        config = data['config']
        print(f"\n{'='*60}")
        print(f"Cache: {cache_name}")
        print(f"{'='*60}")
        print(f"Beschreibung: {config.get('description', 'N/A')}")
        print(f"\nStatistiken:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Total Accesses: {stats['total_accesses']}")
        print(f"  Hit-Rate: {stats['hit_rate']:.2f}%")
        print(f"  Size: {stats['size']}/{config['max_size']}")
        print(f"  Memory: {stats.get('memory_usage_mb', 0.0):.1f} MB")
        if config.get('max_memory_mb'):
            print(f"  Memory-Limit: {config['max_memory_mb']:.1f} MB")
            memory_usage_ratio = stats.get('memory_usage_mb', 0.0) / config['max_memory_mb'] * 100
            print(f"  Memory-Auslastung: {memory_usage_ratio:.1f}%")
        print(f"  Evictions: {stats['evictions']}")
        print(f"  Memory-Evictions: {stats.get('memory_evictions', 0)}")
        print(f"  File Loads: {stats['file_loads']}")
        print(f"  File Saves: {stats['file_saves']}")
        if stats['last_access']:
            import datetime
            last_access = datetime.datetime.fromtimestamp(stats['last_access'])
            print(f"  Last Access: {last_access.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def print_cache_contents(self, cache_type: CacheType, max_keys: int = 10):
        """
        Druckt Cache-Inhalt (für Debugging).
        
        Args:
            cache_type: Typ des Caches
            max_keys: Maximale Anzahl Keys zum Anzeigen
        """
        cache = self.cache_manager.get_cache(cache_type)
        if not cache:
            print(f"Cache {cache_type.value} nicht gefunden")
            return
        
        print(f"\n{'='*60}")
        print(f"Cache-Inhalt: {cache_type.value}")
        print(f"{'='*60}")
        
        with cache._lock:
            keys = list(cache._cache.keys())
            print(f"Total Keys: {len(keys)}")
            print(f"\nErste {min(max_keys, len(keys))} Keys:")
            for i, key in enumerate(keys[:max_keys]):
                value = cache._cache[key]
                if isinstance(key, tuple):
                    key_str = str(key)[:80] + "..." if len(str(key)) > 80 else str(key)
                else:
                    key_str = str(key)[:80] + "..." if len(str(key)) > 80 else str(key)
                print(f"  {i+1}. {key_str}")
                if hasattr(value, 'surface_id'):
                    print(f"     → Surface: {value.surface_id}")
    
    def compare_stats(self, before_stats: Dict[str, Any], after_stats: Dict[str, Any]):
        """
        Vergleicht zwei Statistik-Snapshots.
        
        Args:
            before_stats: Statistiken vor Änderung
            after_stats: Statistiken nach Änderung
        """
        print("\n" + "="*60)
        print("STATISTIK-VERGLEICH")
        print("="*60)
        
        for cache_name in set(before_stats.keys()) | set(after_stats.keys()):
            before = before_stats.get(cache_name, {}).get('stats', {})
            after = after_stats.get(cache_name, {}).get('stats', {})
            
            print(f"\n{cache_name}:")
            
            # Hits
            hits_before = before.get('hits', 0)
            hits_after = after.get('hits', 0)
            hits_diff = hits_after - hits_before
            print(f"  Hits: {hits_before} → {hits_after} ({hits_diff:+d})")
            
            # Misses
            misses_before = before.get('misses', 0)
            misses_after = after.get('misses', 0)
            misses_diff = misses_after - misses_before
            print(f"  Misses: {misses_before} → {misses_after} ({misses_diff:+d})")
            
            # Size
            size_before = before.get('size', 0)
            size_after = after.get('size', 0)
            size_diff = size_after - size_before
            print(f"  Size: {size_before} → {size_after} ({size_diff:+d})")
            
            # Hit-Rate
            hit_rate_before = before.get('hit_rate', 0.0)
            hit_rate_after = after.get('hit_rate', 0.0)
            print(f"  Hit-Rate: {hit_rate_before:.2f}% → {hit_rate_after:.2f}%")
    
    def reset_all_stats(self):
        """Setzt alle Statistiken zurück (behält Cache-Inhalt)"""
        self.cache_manager.reset_stats()
        print("✅ Alle Cache-Statistiken zurückgesetzt")
    
    def clear_all_caches(self):
        """Leert alle Caches"""
        self.cache_manager.clear_all_caches()
        print("✅ Alle Caches geleert")
    
    def print_memory_overview(self, container=None):
        """
        Druckt Memory-Übersicht: System-RAM + Cache-RAM + Snapshot-RAM
        
        Args:
            container: Optional DataContainer für Snapshot-Memory
        """
        print("\n" + "="*60)
        print("MEMORY-ÜBERSICHT")
        print("="*60)
        
        # System-RAM
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                print(f"\nSystem-RAM:")
                print(f"  Total: {mem.total / (1024**3):.2f} GB")
                print(f"  Verfügbar: {mem.available / (1024**3):.2f} GB")
                print(f"  Verwendet: {mem.used / (1024**3):.2f} GB ({mem.percent:.1f}%)")
                print(f"  Frei: {mem.free / (1024**3):.2f} GB")
            except Exception as e:
                print(f"  ⚠️  Fehler beim Abrufen der System-RAM-Info: {e}")
        else:
            print("\nSystem-RAM:")
            print("  ⚠️  psutil nicht verfügbar - System-RAM kann nicht überwacht werden")
        
        # Cache-RAM
        print(f"\nCache-RAM:")
        all_stats = self.cache_manager.get_cache_stats()
        total_cache_memory = 0.0
        for cache_name, data in all_stats.items():
            cache_memory = data['stats'].get('memory_usage_mb', 0.0)
            total_cache_memory += cache_memory
            max_memory = data['config'].get('max_memory_mb')
            if max_memory:
                print(f"  {cache_name}: {cache_memory:.1f} MB / {max_memory:.1f} MB")
            else:
                print(f"  {cache_name}: {cache_memory:.1f} MB (kein Limit)")
        print(f"  Total Cache-RAM: {total_cache_memory:.1f} MB")
        
        # Snapshot-RAM
        if container:
            print(f"\nSnapshot-RAM:")
            try:
                calculation_axes = getattr(container, 'calculation_axes', {})
                snapshot_memory = 0.0
                snapshot_count = 0
                
                for key, snapshot_data in calculation_axes.items():
                    if key != "aktuelle_simulation":  # Nur Snapshots, nicht aktuelle Simulation
                        snapshot_count += 1
                        try:
                            snapshot_mb = estimate_memory_size(snapshot_data)
                            snapshot_memory += snapshot_mb
                        except Exception:
                            pass
                
                print(f"  Anzahl Snapshots: {snapshot_count}")
                print(f"  Total Snapshot-RAM: {snapshot_memory:.1f} MB")
                if snapshot_count > 0:
                    avg_memory = snapshot_memory / snapshot_count
                    print(f"  Durchschnitt pro Snapshot: {avg_memory:.1f} MB")
            except Exception as e:
                print(f"  ⚠️  Fehler beim Berechnen der Snapshot-Memory: {e}")
        
        # Gesamt-Übersicht
        print(f"\n{'='*60}")
        print("GESAMT-ÜBERSICHT")
        print(f"{'='*60}")
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / (1024**2)
                print(f"  Prozess-RAM: {process_memory_mb:.1f} MB")
                print(f"  Cache-RAM: {total_cache_memory:.1f} MB")
                if container:
                    snapshot_memory = 0.0
                    try:
                        calculation_axes = getattr(container, 'calculation_axes', {})
                        for key, snapshot_data in calculation_axes.items():
                            if key != "aktuelle_simulation":
                                snapshot_memory += estimate_memory_size(snapshot_data)
                    except Exception:
                        pass
                    print(f"  Snapshot-RAM: {snapshot_memory:.1f} MB")
                    print(f"  Andere RAM: {process_memory_mb - total_cache_memory - snapshot_memory:.1f} MB")
                print(f"  System-RAM verwendet: {mem.percent:.1f}%")
            except Exception:
                pass


# Convenience-Funktion für einfachen Zugriff
def print_cache_stats(cache_type: Optional[CacheType] = None):
    """Druckt Cache-Statistiken (Convenience-Funktion)"""
    monitor = CacheMonitor()
    monitor.print_stats(cache_type)


def print_detailed_cache_stats(cache_type: Optional[CacheType] = None):
    """Druckt detaillierte Cache-Statistiken (Convenience-Funktion)"""
    monitor = CacheMonitor()
    monitor.print_detailed_stats(cache_type)

