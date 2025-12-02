# PlotSPL3DOverlays Modul

Dieses Verzeichnis enthält die aufgeteilten Module für das Overlay-Rendering im 3D-SPL-Plot.

## Struktur

- **__init__.py** - Hauptklasse `SPL3DOverlayRenderer` (koordiniert alle Untermodule)
- **renderer.py** - Basis-Rendering-Logik (clear, clear_category, _add_overlay_mesh)
- **surfaces.py** - Surface-Overlay-Rendering
- **speakers.py** - Speaker-Overlay-Rendering
- **speaker_geometry.py** - Speaker-Geometrie-Berechnung (flown, stack)
- **axis.py** - Axis-Lines und Axis-Planes Rendering
- **line_utils.py** - Linien-Segmentierung (dash-dot patterns)
- **impulse.py** - Impulse-Points Rendering
- **caching.py** - Caching-Logik (Box-Templates, Speaker-Cache)
- **ui_time_control.py** - SPLTimeControlBar Widget

## Verwendung

```python
from Module_LFO.Modules_Plot.3DPlot.PlotSPL3DOverlays import SPL3DOverlayRenderer, SPLTimeControlBar

renderer = SPL3DOverlayRenderer(plotter, pv)
renderer.draw_surfaces(settings, container)
renderer.draw_speakers(settings, container, cabinet_lookup)
renderer.draw_axis_lines(settings)
```

## Abhängigkeiten

- PyVista
- PyQt5
- NumPy

