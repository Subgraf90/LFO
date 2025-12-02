# PlotSPL3D Modul

Dieses Verzeichnis enthält die aufgeteilten Module für den 3D-SPL-Plot.

## Struktur

- **__init__.py** - Hauptklasse `DrawSPLPlot3D` (koordiniert alle Untermodule)
- **core.py** - Kernfunktionalität (Initialisierung, Basis-API)
- **event_handler.py** - Event-Handling (Mouse-Events, Clicks, Drags)
- **picking.py** - Picking-Logik (Surface, Speaker, Axis)
- **camera.py** - Camera-Management (Rotation, Pan, Views)
- **rendering.py** - SPL-Surface Rendering (Texturen, Meshes)
- **interpolation.py** - Interpolations-Methoden (bilinear, nearest)
- **colorbar.py** - Colorbar-Management
- **ui_components.py** - UI-Setup (View-Controls, Time-Control)
- **utils.py** - Hilfsfunktionen (Konvertierungen, Validierung)

## Verwendung

```python
from Module_LFO.Modules_Plot.3DPlot.PlotSPL3D import DrawSPLPlot3D

plot = DrawSPLPlot3D(parent_widget, settings, colorbar_ax)
```

## Abhängigkeiten

- PyVista / pyvistaqt
- PyQt5
- NumPy
- Matplotlib

