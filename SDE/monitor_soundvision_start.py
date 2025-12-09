#!/usr/bin/env python3
"""
Überwacht SoundVision beim Start und sucht nach entpackten/temporären Dateien.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import List, Set
import hashlib

def get_file_hash(filepath: Path) -> str:
    """Berechnet MD5-Hash einer Datei."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def find_new_files(directory: Path, known_files: Set[Path]) -> List[Path]:
    """Findet neue Dateien in einem Verzeichnis."""
    new_files = []
    if not directory.exists():
        return new_files
    
    for file in directory.rglob("*"):
        if file.is_file() and file not in known_files:
            new_files.append(file)
    
    return new_files

def monitor_app_start():
    """Überwacht SoundVision beim Start."""
    print("=== SoundVision Start-Überwachung ===\n")
    
    # Verzeichnisse zum Überwachen
    monitor_dirs = [
        Path.home() / "Library" / "Caches",
        Path.home() / "Library" / "Application Support",
        Path("/tmp"),
        Path("/var/folders"),  # macOS temp
    ]
    
    # Bekannte Dateien vor App-Start
    print("Erfasse bekannte Dateien vor App-Start...")
    known_files = set()
    for directory in monitor_dirs:
        if directory.exists():
            for file in directory.rglob("*"):
                if file.is_file():
                    known_files.add(file)
    
    print(f"Bekannte Dateien: {len(known_files)}")
    print("\nStarte SoundVision...")
    print("Warte 10 Sekunden auf App-Start...\n")
    
    # Starte App
    app_path = Path("/Users/MGraf/Desktop/soundvision_resigned/Soundvision.app")
    if not app_path.exists():
        app_path = Path("/Applications/Soundvision.app")
    
    if app_path.exists():
        subprocess.Popen([app_path / "Contents" / "MacOS" / "Soundvision"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
    else:
        print("SoundVision App nicht gefunden!")
        return
    
    # Warte und prüfe auf neue Dateien
    time.sleep(10)
    
    print("Suche nach neuen Dateien...")
    new_files = []
    for directory in monitor_dirs:
        new = find_new_files(directory, known_files)
        new_files.extend(new)
    
    # Filtere nach interessanten Dateien
    interesting = []
    for file in new_files:
        name_lower = file.name.lower()
        if ('.bal' in name_lower or 
            'balloon' in name_lower or
            'decrypt' in name_lower or
            'temp' in name_lower or
            file.suffix in ['.dat', '.bin', '.tmp']):
            interesting.append(file)
    
    print(f"\nGefunden: {len(new_files)} neue Dateien")
    print(f"Interessante: {len(interesting)}")
    
    if interesting:
        print("\n=== Interessante neue Dateien ===")
        for file in interesting[:20]:
            size = file.stat().st_size if file.exists() else 0
            print(f"  {file} ({size:,} Bytes)")
            
            # Prüfe auf BAL-ähnliche Dateien
            if size > 100000 and size < 1000000:  # ~100KB - 1MB
                print(f"    → Mögliche BAL-Datei!")
                with open(file, 'rb') as f:
                    header = f.read(16)
                    if header[:4] == b'\x1d\xf9\x3e\xfb':
                        print(f"    → SoundVision BAL-Header erkannt!")
                    elif header[:4] != b'\x1d\xf9\x3e\xfb':
                        print(f"    → Header: {header.hex()[:32]}")
    else:
        print("\nKeine interessanten neuen Dateien gefunden.")
        print("Hinweis: BAL-Dateien könnten im Speicher bleiben oder")
        print("         direkt aus dem Bundle geladen werden.")
    
    # Prüfe auch auf Prozess-Speicher-Dumps
    print("\n=== Prüfe laufende SoundVision-Prozesse ===")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Soundvision' in line and 'grep' not in line:
                print(f"  {line}")
    except:
        pass

if __name__ == "__main__":
    monitor_app_start()

