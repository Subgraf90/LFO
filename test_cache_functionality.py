#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test-Skript für Cache-Funktionalität

Testet:
- Cache-Manager Funktionalität
- LRU-Cache Verhalten
- Gezielte Cache-Invalidierung
- Shared Grid-Generator
- Surface-Cache bei hide/disable
- Thread-Safety
"""

import sys
import os
import time
from typing import Dict, Any

# Füge LFO-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LFO'))

try:
    from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType, LRUCache
    from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator, CachedSurfaceGrid
    from Module_LFO.Modules_Data.settings_state import Settings
    from Module_LFO.Modules_Data.data_module import DataContainer
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass du im richtigen Verzeichnis bist und alle Module verfügbar sind.")
    sys.exit(1)


class CacheTestSuite:
    """Test-Suite für Cache-Funktionalität"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Führt einen Test aus und protokolliert das Ergebnis"""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        print(f"⏱️  Starte Test um {time.strftime('%H:%M:%S')}...")
        
        start_time = time.time()
        try:
            result = test_func()
            elapsed = time.time() - start_time
            if result:
                print(f"✅ PASSED: {test_name} (dauerte {elapsed:.2f}s)")
                self.tests_passed += 1
                self.test_results.append((test_name, True, None))
            else:
                print(f"❌ FAILED: {test_name} (dauerte {elapsed:.2f}s)")
                self.tests_failed += 1
                self.test_results.append((test_name, False, "Test returned False"))
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ ERROR: {test_name} (dauerte {elapsed:.2f}s)")
            print(f"   Fehler: {e}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            self.test_results.append((test_name, False, str(e)))
    
    def print_summary(self):
        """Druckt Test-Zusammenfassung"""
        print(f"\n{'='*60}")
        print("TEST-ZUSAMMENFASSUNG")
        print(f"{'='*60}")
        print(f"✅ Bestanden: {self.tests_passed}")
        print(f"❌ Fehlgeschlagen: {self.tests_failed}")
        print(f"📊 Gesamt: {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed > 0:
            print(f"\n❌ Fehlgeschlagene Tests:")
            for name, passed, error in self.test_results:
                if not passed:
                    print(f"  - {name}: {error}")
        
        print(f"\n{'='*60}")
        if self.tests_failed == 0:
            print("✅✅✅ ALLE TESTS BESTANDEN ✅✅✅")
        else:
            print("❌❌❌ EINIGE TESTS FEHLGESCHLAGEN ❌❌❌")


def test_cache_manager_basic():
    """Test 1: Grundlegende Cache-Manager Funktionalität"""
    print("\n1.1 Cache registrieren...")
    cache = cache_manager.register_cache(
        CacheType.GRID,
        max_size=10,
        description="Test Grid Cache"
    )
    assert cache is not None, "Cache sollte erstellt werden"
    print("   ✅ Cache registriert")
    
    print("\n1.2 Cache abrufen...")
    retrieved_cache = cache_manager.get_cache(CacheType.GRID)
    assert retrieved_cache is cache, "Sollte denselben Cache zurückgeben"
    print("   ✅ Cache abgerufen")
    
    print("\n1.3 Cache-Statistiken...")
    stats = cache_manager.get_cache_stats(CacheType.GRID)
    assert 'grid' in stats, "Statistiken sollten verfügbar sein"
    print(f"   ✅ Statistiken: {stats['grid']['stats']}")
    
    return True


def test_lru_cache_behavior():
    """Test 2: LRU-Cache Verhalten"""
    print("\n2.1 Cache leeren und konfigurieren (max_size=5)...")
    cache_manager.clear_cache(CacheType.GRID)
    cache_manager.configure_cache(CacheType.GRID, max_size=5)
    cache = cache_manager.get_cache(CacheType.GRID)
    
    print("\n2.2 Cache füllen (max_size=5)...")
    for i in range(7):
        cache.set(f"key_{i}", f"value_{i}")
        stats = cache.get_stats()
        print(f"   Nach key_{i}: size={stats.size}, hits={stats.hits}, misses={stats.misses}")
    
    stats = cache.get_stats()
    assert stats.size == 5, f"Cache sollte max_size=5 haben, hat aber {stats.size}"
    print("   ✅ LRU-Eviction funktioniert (älteste Einträge entfernt)")
    
    print("\n2.3 Cache-Zugriff testen...")
    # Älteste sollten entfernt sein
    value_0 = cache.get("key_0")
    assert value_0 is None, "key_0 sollte entfernt worden sein (LRU)"
    print("   ✅ key_0 wurde entfernt (LRU)")
    
    # Neueste sollten vorhanden sein
    value_6 = cache.get("key_6")
    assert value_6 == "value_6", "key_6 sollte vorhanden sein"
    print("   ✅ key_6 ist vorhanden")
    
    return True


def test_cache_hit_miss():
    """Test 3: Cache Hit/Miss Verhalten"""
    print("\n3.1 Cache leeren...")
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    cache.reset_stats()
    
    print("\n3.2 Cache Miss testen...")
    value = cache.get("non_existent_key")
    assert value is None, "Sollte None zurückgeben bei Cache Miss"
    stats = cache.get_stats()
    assert stats.misses == 1, f"Sollte 1 Miss haben, hat aber {stats.misses}"
    print(f"   ✅ Cache Miss: misses={stats.misses}")
    
    print("\n3.3 Cache Hit testen...")
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    assert value == "test_value", "Sollte gespeicherten Wert zurückgeben"
    stats = cache.get_stats()
    assert stats.hits == 1, f"Sollte 1 Hit haben, hat aber {stats.hits}"
    print(f"   ✅ Cache Hit: hits={stats.hits}")
    
    print("\n3.4 Hit-Rate berechnen...")
    hit_rate = stats.hit_rate()
    print(f"   ✅ Hit-Rate: {hit_rate:.2f}%")
    
    return True


def test_targeted_invalidation():
    """Test 4: Gezielte Cache-Invalidierung"""
    print("\n4.1 Cache füllen...")
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    
    # Fülle Cache mit verschiedenen Keys
    cache.set(("surface_1", "horizontal", 1.0, 3, ()), "grid_1")
    cache.set(("surface_2", "horizontal", 1.0, 3, ()), "grid_2")
    cache.set(("surface_1", "vertical", 1.0, 3, ()), "grid_3")
    cache.set(("surface_3", "horizontal", 1.0, 3, ()), "grid_4")
    
    stats_before = cache.get_stats()
    print(f"   Cache-Größe vor Invalidierung: {stats_before.size}")
    
    print("\n4.2 Gezielte Invalidierung (nur surface_1)...")
    def predicate(key):
        return isinstance(key, tuple) and len(key) > 0 and key[0] == "surface_1"
    
    invalidated = cache_manager.invalidate_cache(CacheType.GRID, predicate=predicate)
    print(f"   ✅ {invalidated} Einträge invalidiert")
    
    stats_after = cache.get_stats()
    print(f"   Cache-Größe nach Invalidierung: {stats_after.size}")
    
    # Prüfe dass surface_1 entfernt wurde
    value_1 = cache.get(("surface_1", "horizontal", 1.0, 3, ()))
    assert value_1 is None, "surface_1 sollte entfernt worden sein"
    
    # Prüfe dass surface_2 noch vorhanden ist
    value_2 = cache.get(("surface_2", "horizontal", 1.0, 3, ()))
    assert value_2 == "grid_2", "surface_2 sollte noch vorhanden sein"
    
    print("   ✅ Gezielte Invalidierung funktioniert")
    
    return True


def test_shared_grid_generator():
    """Test 5: Shared Grid-Generator"""
    print("\n5.1 Settings erstellen...")
    settings = Settings()
    
    print("\n5.2 Grid-Generator erstellen...")
    grid_generator_1 = FlexibleGridGenerator(settings)
    grid_cache_1 = grid_generator_1._grid_cache
    
    print("\n5.3 Zweiter Grid-Generator mit Shared Cache...")
    grid_generator_2 = FlexibleGridGenerator(settings, grid_cache=grid_cache_1)
    grid_cache_2 = grid_generator_2._grid_cache
    
    assert grid_cache_1 is grid_cache_2, "Sollte denselben Cache verwenden"
    print("   ✅ Shared Cache funktioniert")
    
    print("\n5.4 Cache-Persistenz testen...")
    # Fülle Cache über Generator 1
    test_key = ("test_surface", "horizontal", 1.0, 3, ())
    test_value = CachedSurfaceGrid(
        surface_id="test_surface",
        orientation="horizontal",
        sound_field_x=None,
        sound_field_y=None,
        X_grid=None,
        Y_grid=None,
        Z_grid=None,
        surface_mask=None,
        resolution=1.0
    )
    grid_cache_1.set(test_key, test_value)
    
    # Prüfe über Generator 2
    retrieved_value = grid_cache_2.get(test_key)
    assert retrieved_value is not None, "Cache sollte über Generator 2 zugänglich sein"
    print("   ✅ Cache-Persistenz funktioniert")
    
    return True


def test_surface_cache_invalidation():
    """Test 6: Surface-Cache Invalidierung"""
    print("\n6.1 Settings und Grid-Generator erstellen...")
    settings = Settings()
    grid_generator = FlexibleGridGenerator(settings)
    
    print("\n6.2 Cache füllen...")
    # Simuliere Cache-Einträge für verschiedene Surfaces
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
    
    stats_before = grid_generator._grid_cache.get_stats()
    print(f"   Cache-Größe vor Invalidierung: {stats_before.size}")
    
    print("\n6.3 Surface-Cache invalidieren (nur surface_1)...")
    invalidated = grid_generator.invalidate_surface_cache("surface_1")
    print(f"   ✅ {invalidated} Einträge invalidiert")
    
    stats_after = grid_generator._grid_cache.get_stats()
    print(f"   Cache-Größe nach Invalidierung: {stats_after.size}")
    
    # Prüfe dass surface_1 entfernt wurde
    cache_key_1 = ("surface_1", "horizontal", 1.0, 3, ())
    value_1 = grid_generator._grid_cache.get(cache_key_1)
    assert value_1 is None, "surface_1 sollte entfernt worden sein"
    
    # Prüfe dass surface_2 noch vorhanden ist
    cache_key_2 = ("surface_2", "horizontal", 1.0, 3, ())
    value_2 = grid_generator._grid_cache.get(cache_key_2)
    assert value_2 is not None, "surface_2 sollte noch vorhanden sein"
    
    print("   ✅ Surface-Cache Invalidierung funktioniert")
    
    return True


def test_cache_statistics():
    """Test 7: Cache-Statistiken"""
    print("\n7.1 Cache leeren und Statistiken zurücksetzen...")
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    cache.reset_stats()
    
    print("\n7.2 Cache-Operationen durchführen...")
    # Fülle Cache
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Cache-Zugriffe
    cache.get("key_0")  # Hit
    cache.get("key_1")  # Hit
    cache.get("key_10")  # Miss
    
    print("\n7.3 Statistiken abrufen...")
    stats = cache.get_stats()
    print(f"   Hits: {stats.hits}")
    print(f"   Misses: {stats.misses}")
    print(f"   Size: {stats.size}")
    print(f"   Hit-Rate: {stats.hit_rate():.2f}%")
    
    assert stats.hits == 2, f"Sollte 2 Hits haben, hat aber {stats.hits}"
    assert stats.misses == 1, f"Sollte 1 Miss haben, hat aber {stats.misses}"
    assert stats.size == 5, f"Sollte Größe 5 haben, hat aber {stats.size}"
    
    print("\n7.4 Globale Statistiken...")
    global_stats = cache_manager.get_global_stats()
    print(f"   Total Caches: {global_stats['total_caches']}")
    print(f"   Global Hit-Rate: {global_stats['global_hit_rate']:.2f}%")
    
    print("   ✅ Statistiken funktionieren korrekt")
    
    return True


def test_thread_safety():
    """Test 8: Thread-Safety (Basis-Test)"""
    print("\n8.1 Thread-Safety Test (Basis)...")
    cache_manager.clear_cache(CacheType.GRID)
    cache = cache_manager.get_cache(CacheType.GRID)
    
    print("\n8.2 Parallele Zugriffe simulieren...")
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
    print(f"   ✅ Thread-Safety: {stats.total_accesses} Zugriffe ohne Fehler")
    
    return True


def test_cache_configuration():
    """Test 9: Cache-Konfiguration"""
    print("\n9.1 Cache konfigurieren...")
    cache_manager.configure_cache(
        CacheType.GRID,
        max_size=20,
        description="Neue Beschreibung"
    )
    
    cache = cache_manager.get_cache(CacheType.GRID)
    assert cache.max_size == 20, f"Sollte max_size=20 haben, hat aber {cache.max_size}"
    print("   ✅ Cache-Konfiguration funktioniert")
    
    return True


def test_multiple_cache_types():
    """Test 10: Mehrere Cache-Typen"""
    print("\n10.1 Mehrere Caches registrieren...")
    cache_manager.register_cache(CacheType.CALC_GEOMETRY, max_size=100)
    cache_manager.register_cache(CacheType.PLOT_TEXTURE, max_size=50)
    
    print("\n10.2 Alle Caches abrufen...")
    grid_cache = cache_manager.get_cache(CacheType.GRID)
    calc_cache = cache_manager.get_cache(CacheType.CALC_GEOMETRY)
    plot_cache = cache_manager.get_cache(CacheType.PLOT_TEXTURE)
    
    assert grid_cache is not None, "Grid-Cache sollte vorhanden sein"
    assert calc_cache is not None, "Calc-Cache sollte vorhanden sein"
    assert plot_cache is not None, "Plot-Cache sollte vorhanden sein"
    
    print("\n10.3 Globale Statistiken...")
    global_stats = cache_manager.get_global_stats()
    print(f"   Total Caches: {global_stats['total_caches']}")
    
    assert global_stats['total_caches'] >= 3, "Sollte mindestens 3 Caches haben"
    print("   ✅ Mehrere Cache-Typen funktionieren")
    
    return True


def main():
    """Hauptfunktion: Führt alle Tests aus"""
    print("="*60)
    print("CACHE-FUNKTIONALITÄT TEST-SUITE")
    print("="*60)
    print("\nTestet:")
    print("  - Cache-Manager Funktionalität")
    print("  - LRU-Cache Verhalten")
    print("  - Cache Hit/Miss")
    print("  - Gezielte Cache-Invalidierung")
    print("  - Shared Grid-Generator")
    print("  - Surface-Cache Invalidierung")
    print("  - Cache-Statistiken")
    print("  - Thread-Safety")
    print("  - Cache-Konfiguration")
    print("  - Mehrere Cache-Typen")
    
    suite = CacheTestSuite()
    
    # Führe alle Tests aus
    suite.run_test("Cache-Manager Grundfunktionalität", test_cache_manager_basic)
    suite.run_test("LRU-Cache Verhalten", test_lru_cache_behavior)
    suite.run_test("Cache Hit/Miss", test_cache_hit_miss)
    suite.run_test("Gezielte Cache-Invalidierung", test_targeted_invalidation)
    suite.run_test("Shared Grid-Generator", test_shared_grid_generator)
    suite.run_test("Surface-Cache Invalidierung", test_surface_cache_invalidation)
    suite.run_test("Cache-Statistiken", test_cache_statistics)
    suite.run_test("Thread-Safety", test_thread_safety)
    suite.run_test("Cache-Konfiguration", test_cache_configuration)
    suite.run_test("Mehrere Cache-Typen", test_multiple_cache_types)
    
    # Zusammenfassung
    suite.print_summary()
    
    return suite.tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

