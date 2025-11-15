# LFO - Line Array Field Optimizer

Audioanwendung für die Berechnung und Visualisierung von Lautsprecherarrays mit FEM-Unterstützung.

## Schnellstart

```bash
/Users/MGraf/Python/LFO_Umgebung/Venv_FEM/bin/python /Users/MGraf/Python/LFO_Umgebung/LFO/Main.py
```

## Installation

Die virtuelle Umgebung ist bereits konfiguriert. Details siehe [INSTALLATION.md](INSTALLATION.md).

### Installation testen

```bash
/Users/MGraf/Python/LFO_Umgebung/Venv_FEM/bin/python test_installation.py
```

## Hauptfunktionen

- 🔊 Lautsprecherarray-Berechnung mit Superposition
- 🧮 FEM-Berechnung mit FEniCSx (Finite-Elemente-Methode)
- 📊 3D-Visualisierung mit PyVista
- 📈 Polar-Pattern-Analyse
- 🎚️ Beamsteering und Windowing
- 💾 Snapshot-Management

## Technologie

- **Python:** 3.11.14
- **FEM:** DOLFINx 0.10.0
- **3D:** PyVista 0.46.4
- **GUI:** PyQt5 5.15.11
- **Numerik:** NumPy 2.3.4, SciPy 1.16.3

## Entwicklung

### Virtuelle Umgebung

Die Umgebung wurde mit Conda/Mamba erstellt:

```bash
# Aktivieren
conda activate /Users/MGraf/Python/LFO_Umgebung/Venv_FEM

# Oder direkt nutzen
/Users/MGraf/Python/LFO_Umgebung/Venv_FEM/bin/python
```

### Projekt-Struktur

```
LFO/
├── Main.py                 # Haupteinstiegspunkt
├── Module_LFO/
│   ├── Modules_Calculate/  # Berechnungsmodule
│   ├── Modules_Plot/       # Plotting-Module
│   ├── Modules_Ui/         # UI-Module
│   ├── Modules_Window/     # Fenster-Module
│   ├── Modules_Data/       # Daten-Module
│   └── Modules_Init/       # Initialisierung
```

## Lizenz

Siehe Projektdokumentation.

---

*Aktualisiert: November 2025*

