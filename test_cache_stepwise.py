#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Schrittweiser Test um zu finden, wo das Programm hängt
"""

import sys
import os
import time

# Füge LFO-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LFO'))

from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType, LRUCache
from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator, CachedSurfaceGrid
from Module_LFO.Modules_Data.settings_state import Settings

def test_1():
    print("\n=== Test 1: Cache-Manager Grundfunktionalität ===")
    start = time.time()
    cache = cache_manager.register_cache(CacheType.GRID, max_size=10, description="Test Grid Cache")
    assert cache is not None
    retrieved_cache = cache_manager.get_cache(CacheType.GRID)
    assert retrieved_cache is cache
    stats = cache_manager.get_cache_stats(CacheType.GRID)
    assert 'grid' in stats
    print(f"✅ Test 1 bestanden ({time.time() - start:.2f}s)")

def test_2():
    print("\n=== Test 2: LRU-Cache Verhalten ===")
    start = time.time()
    cache_manager.clear_cache(CacheType.GRID)
    cache_manager.configure_cache(CacheType.GRID, max_size=5)
    cache = cache_manager.get_cache(CacheType.GRID)
    for i in range(7):
        cache.set(f"key_{i}", f"value_{i}")
    stats = cache.get_stats()
    assert stats.size == 5, f"Cache sollte Größe 5 haben, hat aber {stats.size}"
    assert cache.get("key_0") is None
    assert cache.get("key_6") == "value_6"
    print(f"✅ Test 2 bestanden ({time.time() - start:.2f}s)")

def test_3():
    print("\n=== Test 3: Cache Hit/Miss ===")
    start = time.time()
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    cache.reset_stats()
    value = cache.get("non_existent_key")
    assert value is None
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    assert value == "test_value"
    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    print(f"✅ Test 3 bestanden ({time.time() - start:.2f}s)")

def test_4():
    print("\n=== Test 4: Gezielte Cache-Invalidierung ===")
    start = time.time()
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    cache.set(("surface_1", "horizontal", 1.0, 3, ()), "grid_1")
    cache.set(("surface_2", "horizontal", 1.0, 3, ()), "grid_2")
    cache.set(("surface_1", "vertical", 1.0, 3, ()), "grid_3")
    def predicate(key):
        return isinstance(key, tuple) and len(key) > 0 and key[0] == "surface_1"
    invalidated = cache_manager.invalidate_cache(CacheType.GRID, predicate=predicate)
    assert invalidated == 2
    assert cache.get(("surface_1", "horizontal", 1.0, 3, ())) is None
    assert cache.get(("surface_2", "horizontal", 1.0, 3, ())) == "grid_2"
    print(f"✅ Test 4 bestanden ({time.time() - start:.2f}s)")

def test_5():
    print("\n=== Test 5: Shared Grid-Generator ===")
    start = time.time()
    print("  Erstelle Settings...")
    settings = Settings()
    print("  Erstelle Grid-Generator 1...")
    grid_generator_1 = FlexibleGridGenerator(settings)
    grid_cache_1 = grid_generator_1._grid_cache
    print("  Erstelle Grid-Generator 2...")
    grid_generator_2 = FlexibleGridGenerator(settings, grid_cache=grid_cache_1)
    grid_cache_2 = grid_generator_2._grid_cache
    assert grid_cache_1 is grid_cache_2
    print(f"✅ Test 5 bestanden ({time.time() - start:.2f}s)")

def test_6():
    print("\n=== Test 6: Surface-Cache Invalidierung ===")
    start = time.time()
    settings = Settings()
    grid_generator = FlexibleGridGenerator(settings)
    for surface_id in ["surface_1", "surface_2", "surface_3"]:
        cache_key = (surface_id, "horizontal", 1.0, 3, ())
        test_value = CachedSurfaceGrid(
            surface_id=surface_id,
            orientation="horizontal",
            sound_field_x=None,
            sound_field_y=None,
            X_grid=None,
            Y_grid=None,
            Z_grid=None,
            surface_mask=None,
            resolution=1.0
        )
        grid_generator._grid_cache.set(cache_key, test_value)
    invalidated = grid_generator.invalidate_surface_cache("surface_1")
    assert invalidated == 1
    assert grid_generator._grid_cache.get(("surface_1", "horizontal", 1.0, 3, ())) is None
    assert grid_generator._grid_cache.get(("surface_2", "horizontal", 1.0, 3, ())) is not None
    print(f"✅ Test 6 bestanden ({time.time() - start:.2f}s)")

def test_7():
    print("\n=== Test 7: Cache-Statistiken ===")
    start = time.time()
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    cache.reset_stats()
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
    cache.get("key_0")
    cache.get("key_1")
    cache.get("key_10")
    stats = cache.get_stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.size == 5
    print(f"✅ Test 7 bestanden ({time.time() - start:.2f}s)")

def test_8():
    print("\n=== Test 8: Thread-Safety ===")
    start = time.time()
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    import threading
    def worker(thread_id, num_operations):
        for i in range(num_operations):
            key = f"thread_{thread_id}_key_{i}"
            cache.set(key, f"value_{i}")
            cache.get(key)
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i, 10))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    stats = cache.get_stats()
    print(f"✅ Test 8 bestanden ({time.time() - start:.2f}s) - {stats.total_accesses} Zugriffe")

def test_9():
    print("\n=== Test 9: Cache-Konfiguration ===")
    start = time.time()
    cache_manager.configure_cache(CacheType.GRID, max_size=20, description="Neue Beschreibung")
    cache = cache_manager.get_cache(CacheType.GRID)
    assert cache.max_size == 20
    print(f"✅ Test 9 bestanden ({time.time() - start:.2f}s)")

def test_10():
    print("\n=== Test 10: Mehrere Cache-Typen ===")
    start = time.time()
    cache_manager.register_cache(CacheType.CALC_GEOMETRY, max_size=100)
    cache_manager.register_cache(CacheType.PLOT_TEXTURE, max_size=50)
    grid_cache = cache_manager.get_cache(CacheType.GRID)
    calc_cache = cache_manager.get_cache(CacheType.CALC_GEOMETRY)
    plot_cache = cache_manager.get_cache(CacheType.PLOT_TEXTURE)
    assert grid_cache is not None
    assert calc_cache is not None
    assert plot_cache is not None
    global_stats = cache_manager.get_global_stats()
    assert global_stats['total_caches'] >= 3
    print(f"✅ Test 10 bestanden ({time.time() - start:.2f}s)")

if __name__ == "__main__":
    print("="*60)
    print("SCHRITTWEISER CACHE-TEST")
    print("="*60)
    
    tests = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10]
    
    for i, test_func in enumerate(tests, 1):
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Test {i} fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*60)
    print("TESTS ABGESCHLOSSEN")
    print("="*60)

