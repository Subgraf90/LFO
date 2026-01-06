#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Einfacher Test um zu finden, wo das Programm hängt
"""

import sys
import os
import time

# Füge LFO-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LFO'))

print("1. Starte Import-Tests...")

try:
    print("2. Importiere CacheManager...")
    from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType, LRUCache
    print("   ✅ CacheManager importiert")
except Exception as e:
    print(f"   ❌ Fehler bei CacheManager: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("3. Importiere FlexibleGridGenerator...")
    from Module_LFO.Modules_Calculate.FlexibleGridGenerator import FlexibleGridGenerator, CachedSurfaceGrid
    print("   ✅ FlexibleGridGenerator importiert")
except Exception as e:
    print(f"   ❌ Fehler bei FlexibleGridGenerator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("4. Importiere Settings...")
    start_time = time.time()
    from Module_LFO.Modules_Data.settings_state import Settings
    elapsed = time.time() - start_time
    print(f"   ✅ Settings importiert (dauerte {elapsed:.2f}s)")
except Exception as e:
    print(f"   ❌ Fehler bei Settings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("5. Erstelle Settings-Instanz...")
    start_time = time.time()
    settings = Settings()
    elapsed = time.time() - start_time
    print(f"   ✅ Settings-Instanz erstellt (dauerte {elapsed:.2f}s)")
except Exception as e:
    print(f"   ❌ Fehler bei Settings-Instanz: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("6. Erstelle FlexibleGridGenerator...")
    start_time = time.time()
    grid_generator = FlexibleGridGenerator(settings)
    elapsed = time.time() - start_time
    print(f"   ✅ FlexibleGridGenerator erstellt (dauerte {elapsed:.2f}s)")
except Exception as e:
    print(f"   ❌ Fehler bei FlexibleGridGenerator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Alle Imports erfolgreich!")

