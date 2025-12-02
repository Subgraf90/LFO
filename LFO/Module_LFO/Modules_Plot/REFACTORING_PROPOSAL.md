# Refactoring-Vorschlag: Aufteilung von PlotSPL3D und PlotSPL3DOverlays

## Übersicht

Die beiden Module `PlotSPL3D.py` (5138 Zeilen) und `PlotSPL3DOverlays.py` (3054 Zeilen) sind sehr groß und enthalten viele verschiedene Verantwortlichkeiten. Eine Aufteilung in logische Untermodule würde die Wartbarkeit, Testbarkeit und Lesbarkeit deutlich verbessern.

## Vorgeschlagene Modulstruktur

### 1. **PlotSPL3D/** (Hauptverzeichnis)
```
PlotSPL3D/
├── __init__.py                    # Hauptklasse DrawSPLPlot3D (dünne Koordinationsschicht)
├── core.py                        # Kernfunktionalität (Initialisierung, Basis-API)
├── event_handler.py               # Event-Handling (Mouse-Events, Clicks, Drags)
├── picking.py                     # Picking-Logik (Surface, Speaker, Axis)
├── camera.py                      # Camera-Management (Rotation, Pan, Views)
├── rendering.py                   # SPL-Surface Rendering (Texturen, Meshes)
├── interpolation.py               # Interpolations-Methoden (bilinear, nearest)
├── colorbar.py                   # Colorbar-Management
├── ui_components.py              # UI-Setup (View-Controls, Time-Control)
└── utils.py                      # Hilfsfunktionen (Konvertierungen, Validierung)
```

### 2. **PlotSPL3DOverlays/** (Hauptverzeichnis)
```
PlotSPL3DOverlays/
├── __init__.py                    # Hauptklasse SPL3DOverlayRenderer (dünne Koordinationsschicht)
├── renderer.py                   # Basis-Rendering-Logik (clear, clear_category, _add_overlay_mesh)
├── surfaces.py                    # Surface-Overlay-Rendering
├── speakers.py                    # Speaker-Overlay-Rendering + Geometrie
├── axis.py                        # Axis-Lines und Axis-Planes Rendering
├── impulse.py                     # Impulse-Points Rendering
├── speaker_geometry.py            # Speaker-Geometrie-Berechnung (flown, stack)
├── caching.py                     # Caching-Logik (Box-Templates, Speaker-Cache)
├── line_utils.py                  # Linien-Segmentierung (dash-dot patterns)
└── ui_time_control.py             # SPLTimeControlBar Widget
```

## Detaillierte Aufteilung

### PlotSPL3D/core.py
**Verantwortlichkeiten:**
- `__init__()` - Initialisierung (vereinfacht, delegiert an Untermodule)
- `initialize_empty_scene()` - Leere Szene initialisieren
- `update_spl_plot()` - Haupt-Update-Methode (koordiniert)
- `update_overlays()` - Overlay-Updates koordinieren
- `render()` - Rendering koordinieren
- Basis-Properties und State-Management

**Zeilen:** ~300-400

---

### PlotSPL3D/event_handler.py
**Verantwortlichkeiten:**
- `eventFilter()` - Haupt-Event-Filter
- `_handle_surface_click()` - Surface-Klicks
- `_handle_speaker_click()` - Speaker-Klicks
- `_handle_axis_line_click()` - Axis-Line-Klicks
- `_handle_axis_line_drag()` - Axis-Line-Drags
- `_handle_spl_surface_click()` - SPL-Surface-Klicks
- `_handle_mouse_move_3d()` - Mouse-Move-Handling
- `_handle_double_click_auto_range()` - Doppelklick-Handling
- Event-State-Variablen (_pan_active, _rotate_active, etc.)

**Zeilen:** ~800-1000

---

### PlotSPL3D/picking.py
**Verantwortlichkeiten:**
- `_pick_surface_at_position()` - Surface-Picking
- `_pick_speaker_at_position()` - Speaker-Picking (mit Ray-Triangle-Intersection)
- `_get_z_from_mesh()` - Z-Koordinate aus Mesh extrahieren
- `_point_in_polygon()` - Polygon-Test
- Picking-Hilfsfunktionen

**Zeilen:** ~400-500

---

### PlotSPL3D/camera.py
**Verantwortlichkeiten:**
- `_pan_camera()` - Camera-Pan
- `_rotate_camera()` - Camera-Rotation
- `_rotate_camera_around_z()` - Z-Achsen-Rotation
- `_maximize_camera_view()` - View maximieren
- `_zoom_to_default_surface()` - Zoom auf Surface
- `set_view_isometric()` - Isometrische Ansicht
- `set_view_top()` - Top-Ansicht
- `set_view_side_y()` - Y-Seitenansicht
- `set_view_side_x()` - X-Seitenansicht
- `_save_camera_state()` - Camera-State speichern
- `_capture_camera()` - Camera-State erfassen
- `_restore_camera()` - Camera-State wiederherstellen

**Zeilen:** ~400-500

---

### PlotSPL3D/rendering.py
**Verantwortlichkeiten:**
- `_render_surfaces_textured()` - Texturiertes Surface-Rendering
- `_update_surface_scalars()` - Surface-Scalars aktualisieren
- `_update_vertical_spl_surfaces()` - Vertikale Surfaces aktualisieren
- `_clear_vertical_spl_surfaces()` - Vertikale Surfaces löschen
- `_get_vertical_color_limits()` - Color-Limits für vertikale Surfaces
- `update_time_frame_values()` - Zeit-Frame-Updates
- Texture-Metadaten-Methoden (`get_texture_metadata`, etc.)
- `_remove_actor()` - Actor entfernen

**Zeilen:** ~1000-1200

---

### PlotSPL3D/interpolation.py
**Verantwortlichkeiten:**
- `_bilinear_interpolate_grid()` - Bilineare Interpolation
- `_nearest_interpolate_grid()` - Nearest-Neighbor Interpolation
- `_quantize_to_steps()` - Quantisierung zu Schritten
- Interpolations-Hilfsfunktionen

**Zeilen:** ~200-300

---

### PlotSPL3D/colorbar.py
**Verantwortlichkeiten:**
- `_initialize_empty_colorbar()` - Leere Colorbar initialisieren
- `_update_colorbar()` - Colorbar aktualisieren
- `_render_colorbar()` - Colorbar rendern
- `_get_colorbar_params()` - Colorbar-Parameter berechnen
- `_on_colorbar_double_click()` - Colorbar-Doppelklick-Handling
- Colorbar-State-Management

**Zeilen:** ~300-400

---

### PlotSPL3D/ui_components.py
**Verantwortlichkeiten:**
- `_setup_view_controls()` - View-Controls Setup
- `_setup_time_control()` - Time-Control Setup
- `_create_view_button()` - View-Button erstellen
- `_create_axis_pixmap()` - Axis-Pixmap erstellen
- `update_time_control()` - Time-Control aktualisieren
- `set_time_slider_callback()` - Time-Slider-Callback setzen
- `_handle_time_slider_change()` - Time-Slider-Change-Handling

**Zeilen:** ~200-300

---

### PlotSPL3D/utils.py
**Verantwortlichkeiten:**
- `_has_valid_data()` - Datenvalidierung
- `_compute_surface_signature()` - Surface-Signatur berechnen
- `_compute_overlay_signatures()` - Overlay-Signaturen berechnen
- `_to_float_array()` - Array-Konvertierung
- `_fallback_cmap_to_lut()` - Colormap zu LUT
- Hilfsfunktionen

**Zeilen:** ~200-300

---

### PlotSPL3DOverlays/renderer.py
**Verantwortlichkeiten:**
- `__init__()` - Initialisierung
- `clear()` - Alle Overlays löschen
- `clear_category()` - Kategorie löschen
- `_add_overlay_mesh()` - Overlay-Mesh hinzufügen
- `_remove_actor()` - Actor entfernen
- Basis-Caching (Box-Templates, etc.)

**Zeilen:** ~300-400

---

### PlotSPL3DOverlays/surfaces.py
**Verantwortlichkeiten:**
- `draw_surfaces()` - Surfaces zeichnen
- `_get_active_xy_surfaces()` - Aktive XY-Surfaces sammeln
- Surface-Batch-Rendering-Logik
- Surface-State-Management

**Zeilen:** ~600-700

---

### PlotSPL3DOverlays/speakers.py
**Verantwortlichkeiten:**
- `draw_speakers()` - Speakers zeichnen
- `_get_speaker_info_from_actor()` - Speaker-Info aus Actor extrahieren
- `_update_speaker_highlights()` - Speaker-Highlights aktualisieren
- `_update_speaker_actor()` - Speaker-Actor aktualisieren
- `build_cabinet_lookup()` - Cabinet-Lookup erstellen
- Speaker-Caching-Logik

**Zeilen:** ~400-500

---

### PlotSPL3DOverlays/speaker_geometry.py
**Verantwortlichkeiten:**
- `_build_speaker_geometries()` - Speaker-Geometrien erstellen
- `_apply_flown_cabinet_shape()` - Flown-Cabinet-Form anwenden
- `_resolve_cabinet_entries()` - Cabinet-Einträge auflösen
- `_get_box_template()` - Box-Template holen/erstellen
- `_compute_box_face_indices()` - Box-Face-Indices berechnen
- `_get_speaker_name()` - Speaker-Name extrahieren
- Geometrie-Hilfsfunktionen (_safe_float, _array_value, etc.)

**Zeilen:** ~800-1000

---

### PlotSPL3DOverlays/axis.py
**Verantwortlichkeiten:**
- `draw_axis_lines()` - Axis-Linien zeichnen
- `_draw_axis_planes()` - Axis-Planes zeichnen
- `_get_max_surface_dimension()` - Maximale Surface-Dimension berechnen
- `_get_surface_intersection_points_xz()` - XZ-Schnittpunkte berechnen
- `_get_surface_intersection_points_yz()` - YZ-Schnittpunkte berechnen
- Axis-State-Management

**Zeilen:** ~400-500

---

### PlotSPL3DOverlays/line_utils.py
**Verantwortlichkeiten:**
- `_create_dash_dot_line_segments_optimized()` - Optimierte Dash-Dot-Segmente
- `_create_dash_dot_line_segments()` - Dash-Dot-Segmente erstellen
- `_combine_line_segments()` - Liniensegmente kombinieren
- `_interpolate_line_segment_simple()` - Vereinfachte Segment-Interpolation
- `_interpolate_line_segment()` - Segment-Interpolation

**Zeilen:** ~300-400

---

### PlotSPL3DOverlays/impulse.py
**Verantwortlichkeiten:**
- `draw_impulse_points()` - Impulse-Points zeichnen
- `_compute_impulse_state()` - Impulse-State berechnen

**Zeilen:** ~50-100

---

### PlotSPL3DOverlays/caching.py
**Verantwortlichkeiten:**
- Box-Template-Caching
- Speaker-Geometry-Caching
- `_create_geometry_cache_key()` - Cache-Key erstellen
- `_speaker_signature_from_mesh()` - Speaker-Signatur aus Mesh

**Zeilen:** ~200-300

---

### PlotSPL3DOverlays/ui_time_control.py
**Verantwortlichkeiten:**
- `SPLTimeControlBar` Klasse komplett

**Zeilen:** ~150

---

## Vorteile dieser Aufteilung

1. **Klare Verantwortlichkeiten:** Jedes Modul hat eine eindeutige Aufgabe
2. **Bessere Testbarkeit:** Einzelne Komponenten können isoliert getestet werden
3. **Wiederverwendbarkeit:** Komponenten können in anderen Kontexten genutzt werden
4. **Wartbarkeit:** Änderungen sind lokalisiert und einfacher nachvollziehbar
5. **Lesbarkeit:** Kleinere Dateien sind leichter zu verstehen
6. **Parallele Entwicklung:** Mehrere Entwickler können gleichzeitig arbeiten

## Migrationsstrategie

### Phase 1: Vorbereitung
1. Tests für kritische Funktionen schreiben (falls noch nicht vorhanden)
2. Abhängigkeiten dokumentieren
3. Interfaces definieren

### Phase 2: Schrittweise Migration
1. **Start mit isolierten Modulen:**
   - `PlotSPL3D/interpolation.py` (wenig Abhängigkeiten)
   - `PlotSPL3DOverlays/impulse.py` (einfach)
   - `PlotSPL3DOverlays/ui_time_control.py` (isoliert)

2. **Dann komplexere Module:**
   - `PlotSPL3D/camera.py` (abhängig von core)
   - `PlotSPL3D/colorbar.py` (abhängig von core)
   - `PlotSPL3DOverlays/line_utils.py` (Hilfsfunktionen)

3. **Zuletzt Kern-Module:**
   - `PlotSPL3D/core.py` (koordiniert alles)
   - `PlotSPL3D/event_handler.py` (komplexe Abhängigkeiten)
   - `PlotSPL3DOverlays/renderer.py` (Basis für Overlays)

### Phase 3: Integration
1. Hauptklassen werden zu dünnen Koordinationsschichten
2. Alte Dateien werden schrittweise durch Importe ersetzt
3. Tests laufen lassen und Fehler beheben

## Beispiel: Neue Struktur für DrawSPLPlot3D

```python
# PlotSPL3D/__init__.py
from .core import DrawSPLPlot3D
__all__ = ['DrawSPLPlot3D']

# PlotSPL3D/core.py
from .event_handler import EventHandler
from .camera import CameraManager
from .rendering import SurfaceRenderer
from .colorbar import ColorbarManager
from .picking import PickingManager
from .ui_components import UIComponentManager

class DrawSPLPlot3D(ModuleBase, QtCore.QObject):
    def __init__(self, parent_widget, settings, colorbar_ax):
        # ... Initialisierung ...
        
        # Untermodule initialisieren
        self.event_handler = EventHandler(self)
        self.camera = CameraManager(self.plotter)
        self.renderer = SurfaceRenderer(self.plotter, self.settings)
        self.colorbar = ColorbarManager(self.colorbar_ax, self.settings)
        self.picking = PickingManager(self.plotter, self.settings)
        self.ui = UIComponentManager(self)
        
        # Event-Filter delegieren
        self.widget.installEventFilter(self.event_handler)
    
    def eventFilter(self, obj, event):
        return self.event_handler.eventFilter(obj, event)
    
    # ... weitere koordinierende Methoden ...
```

## Nächste Schritte

1. **Diskussion:** Ist diese Aufteilung sinnvoll? Gibt es bessere Gruppierungen?
2. **Priorisierung:** Welche Module sollten zuerst migriert werden?
3. **Tests:** Welche Tests müssen geschrieben werden?
4. **Zeitplan:** Wie viel Zeit steht für die Refaktorierung zur Verfügung?

