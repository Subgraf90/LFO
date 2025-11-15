#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test-Skript für LFO Installation
"""

import sys
import sysconfig

print("=" * 60)
print("LFO Installation Test")
print("=" * 60)
print()

# Test 1: Python Version
print(f"✓ Python Version: {sys.version.split()[0]}")
print()

# Test 2: Include-Pfad (wichtig für FEM)
print("Include-Pfad Check:")
include_path = sysconfig.get_path('include')
print(f"  Pfad: {include_path}")
if ' ' in include_path:
    print("  ✗ FEHLER: Pfad enthält Leerzeichen!")
    sys.exit(1)
else:
    print("  ✓ OK: Keine Leerzeichen")
print()

# Test 3: Pakete importieren
print("Package Import Tests:")
packages = [
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('matplotlib', 'Matplotlib'),
    ('pyvista', 'PyVista'),
    ('pyvistaqt', 'PyVista Qt'),
    ('PyQt5.QtCore', 'PyQt5'),
    ('PyQt5.QtWidgets', 'PyQt5 Widgets'),
    ('qtpy', 'QtPy'),
    ('dolfinx', 'DOLFINx'),
    ('mpi4py', 'MPI4Py'),
    ('petsc4py', 'PETSc4Py'),
]

all_ok = True
for pkg_name, description in packages:
    try:
        mod = __import__(pkg_name.split('.')[0])
        version = getattr(mod, '__version__', 'ok')
        print(f"  ✓ {description:20s}: {version}")
    except ImportError as e:
        print(f"  ✗ {description:20s}: FEHLT")
        all_ok = False
print()

# Test 4: Qt Plugins
print("Qt Plugin Check:")
try:
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtWidgets import QApplication
    import os
    
    # Prüfe Qt Plugin Path
    qt_plugin_path = os.path.join(
        sys.prefix,
        'lib/python3.11/site-packages/PyQt5/Qt5/plugins'
    )
    cocoa_plugin = os.path.join(qt_plugin_path, 'platforms/libqcocoa.dylib')
    
    if os.path.exists(cocoa_plugin):
        print(f"  ✓ Cocoa Plugin gefunden: {cocoa_plugin}")
    else:
        print(f"  ✗ Cocoa Plugin fehlt: {cocoa_plugin}")
        all_ok = False
except Exception as e:
    print(f"  ✗ Fehler beim Qt-Test: {e}")
    all_ok = False
print()

# Test 5: LFO Module
print("LFO Module Check:")
sys.path.insert(0, '/Users/MGraf/Python/LFO_Umgebung/LFO')
try:
    from Module_LFO.Modules_Calculate.SoundFieldCalculator_FEM import SoundFieldCalculatorFEM
    print("  ✓ SoundFieldCalculatorFEM importiert")
except ImportError as e:
    print(f"  ✗ SoundFieldCalculatorFEM Import-Fehler: {e}")
    all_ok = False
print()

# Zusammenfassung
print("=" * 60)
if all_ok:
    print("✓✓✓ ALLE TESTS BESTANDEN ✓✓✓")
    print()
    print("Die LFO-Anwendung kann gestartet werden:")
    print(f"  {sys.executable} /Users/MGraf/Python/LFO_Umgebung/LFO/Main.py")
    sys.exit(0)
else:
    print("✗✗✗ EINIGE TESTS FEHLGESCHLAGEN ✗✗✗")
    sys.exit(1)

