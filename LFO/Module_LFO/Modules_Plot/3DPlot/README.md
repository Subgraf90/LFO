# 3DPlot Modulstruktur

Dieses Verzeichnis enthält die aufgeteilten Module für den 3D-SPL-Plot.

## Struktur

```
3DPlot/
├── PlotSPL3D/              # Hauptmodul für den 3D-SPL-Plot
│   ├── __init__.py        # Hauptklasse DrawSPLPlot3D
│   ├── core.py            # Kernfunktionalität
│   ├── event_handler.py   # Event-Handling
│   ├── picking.py         # Picking-Logik
│   ├── camera.py          # Camera-Management
│   ├── rendering.py       # SPL-Surface Rendering
│   ├── interpolation.py   # Interpolations-Methoden
│   ├── colorbar.py        # Colorbar-Management
│   ├── ui_components.py   # UI-Setup
│   └── utils.py           # Hilfsfunktionen
│
└── PlotSPL3DOverlays/      # Overlay-Rendering
    ├── __init__.py        # Hauptklasse SPL3DOverlayRenderer
    ├── renderer.py        # Basis-Rendering-Logik
    ├── surfaces.py        # Surface-Overlay-Rendering
    ├── speakers.py        # Speaker-Overlay-Rendering
    ├── speaker_geometry.py # Speaker-Geometrie-Berechnung
    ├── axis.py            # Axis-Lines und Planes
    ├── line_utils.py      # Linien-Segmentierung
    ├── impulse.py         # Impulse-Points
    ├── caching.py         # Caching-Logik
    └── ui_time_control.py # Time-Control Widget
```

## Verwendung

Die Module können wie folgt importiert werden:

```python
# Hauptklasse
from Module_LFO.Modules_Plot.3DPlot.PlotSPL3D import DrawSPLPlot3D

# Overlay-Renderer
from Module_LFO.Modules_Plot.3DPlot.PlotSPL3DOverlays import SPL3DOverlayRenderer, SPLTimeControlBar
```

## Migration

Die ursprünglichen Dateien `PlotSPL3D.py` und `PlotSPL3DOverlays.py` bleiben zunächst erhalten.
Nach erfolgreicher Migration können sie durch Importe ersetzt werden:

```python
# Alte Datei PlotSPL3D.py wird zu:
from Module_LFO.Modules_Plot.3DPlot.PlotSPL3D import DrawSPLPlot3D
__all__ = ['DrawSPLPlot3D']
```

## Dokumentation

Jedes Modul hat eine eigene README-Datei mit detaillierter Beschreibung der Verantwortlichkeiten.

