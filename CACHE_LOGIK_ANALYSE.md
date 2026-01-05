# Cache-Logik Analyse: Surface, Speaker und weitere Caches

## 1. Speaker-Cache Logik

### Aktuelle Implementierung

**Datei:** `LFO/Module_LFO/Modules_Plot/Plot_SPL_3D/PlotSPL3DSpeaker.py`

### Cache-Struktur

```python
# Speaker-spezifische Caches
self._speaker_actor_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
# Key: (array_id, speaker_idx, geom_idx)
# Value: {'actor': actor, 'metadata': {...}}

self._speaker_geometry_cache: dict[str, List[Tuple[Any, Optional[int]]]] = {}
# Key: Cache-Key (String)
# Value: Liste von (Geometrie, Optional[Index])

self._speaker_geometry_param_cache: dict[tuple[str, int], tuple] = {}
# Key: (array_id, speaker_idx)
# Value: Parameter-Signatur (Tuple)

self._overlay_array_cache: dict[tuple[str, int], dict[str, Any]] = {}
# Key: (array_id, speaker_idx)
# Value: Array-Cache-Daten

self._array_geometry_cache: dict[str, dict[str, Any]] = {}
# Key: array_id (String)
# Value: Array-Geometrie-Daten

self._array_signature_cache: dict[str, tuple] = {}
# Key: array_id (String)
# Value: Array-Signatur (Tuple)

self._stack_geometry_cache: dict[tuple[str, tuple], dict[str, Any]] = {}
# Key: (array_id, stack_signature)
# Value: Stack-Geometrie-Daten

self._stack_signature_cache: dict[tuple[str, tuple], tuple] = {}
# Key: (array_id, stack_signature)
# Value: Stack-Signatur (Tuple)

self._box_template_cache: dict[tuple[float, float, float], Any] = {}
# Key: (width, height, depth)
# Value: Box-Template (wird nicht gelöscht bei clear())

self._box_face_cache: dict[tuple[float, float, float], Tuple[Optional[int], Optional[int]]] = {}
# Key: (width, height, depth)
# Value: Box-Face-Indizes
```

### Cache-Logik

#### 1. Cache-Zugriff beim Zeichnen

```python
def draw_speakers(self, settings, container, cabinet_lookup):
    # 1. Prüfe Array-Signatur
    #    - Wenn Array-Signatur unverändert → lade alle Speaker aus Array-Cache
    #    - Wenn Stack-Signatur unverändert → lade Speaker aus Stack-Cache
    
    # 2. Für jeden Speaker:
    #    - Prüfe Parameter-Signatur (_speaker_geometry_param_cache)
    #    - Prüfe Geometry-Cache (_speaker_geometry_cache)
    #    - Wenn Signatur übereinstimmt UND Cache vorhanden:
    #        → Nutze Cache (Position-Transformation)
    #    - Sonst:
    #        → Neu berechnen (parallel)
```

#### 2. Cache-Invalidierung

**Bei Array-Änderungen:**
```python
def clear_array_cache(self, array_id: str):
    # Löscht alle Cache-Einträge für ein spezifisches Array:
    # - _speaker_actor_cache (alle Keys mit array_id)
    # - _speaker_geometry_cache (über _array_id_to_cache_keys Mapping)
    # - _speaker_geometry_param_cache
    # - _overlay_array_cache
    # - _array_geometry_cache
    # - _array_signature_cache
    # - _stack_geometry_cache (alle Einträge mit array_id)
    # - _stack_signature_cache
```

**Trigger für Cache-Invalidierung:**
- `on_group_hide_changed()` → `clear_array_cache()` für alle betroffenen Arrays
- `on_group_mute_changed()` → Cache bleibt erhalten (nur Visibility ändert sich)
- Position-Änderungen → `clear_array_cache()` für betroffenes Array
- Array-Parameter-Änderungen → `clear_array_cache()` für betroffenes Array

#### 3. Cache-Hierarchie

```
Array-Signatur Cache (höchste Ebene)
  └─> Wenn unverändert → lade alle Speaker aus Array-Cache
  └─> Wenn geändert → prüfe Stack-Cache

Stack-Signatur Cache (mittlere Ebene)
  └─> Wenn unverändert → lade Speaker aus Stack-Cache
  └─> Wenn geändert → prüfe einzelne Speaker

Speaker-Parameter-Signatur Cache (niedrigste Ebene)
  └─> Wenn unverändert UND Geometry-Cache vorhanden → nutze Cache
  └─> Sonst → neu berechnen
```

#### 4. Cache-Optimierungen

- **Array-Level Cache**: Wenn Array-Signatur unverändert, werden alle Speaker aus Cache geladen
- **Stack-Level Cache**: Wenn Stack-Signatur unverändert, werden Speaker dieser Stack-Gruppe aus Cache geladen
- **Speaker-Level Cache**: Einzelne Speaker werden gecacht, wenn Parameter unverändert
- **Box-Template-Cache**: Wird nicht gelöscht (Performance-Optimierung für wiederholte Box-Geometrien)

---

## 2. Surface-Cache Logik (Aktuell)

### Aktuelle Implementierung

**Datei:** `LFO/Module_LFO/Modules_Calculate/FlexibleGridGenerator.py`

### Cache-Struktur

```python
# Grid-Cache (über Cache-Manager)
self._grid_cache: LRUCache  # CacheType.GRID
# Key: (surface_id, orientation, resolution, min_points, points_signature, geometry_version)
# Value: CachedSurfaceGrid

# Cache-Key enthält:
# - surface_id: String
# - orientation: String ("horizontal", "vertical", "slanted")
# - resolution: float
# - min_points: int
# - points_signature: Hash der Surface-Punkte
# - geometry_version: int (aus Settings.geometry_version)
```

### Cache-Logik (Aktuell)

#### 1. Cache-Zugriff

```python
def generate_per_surface_check_cache(self, geometries, resolution, min_points):
    for geom in geometries:
        cache_key = self._make_surface_cache_key(geom, resolution, min_points)
        cached_grid = self._grid_cache.get(cache_key)
        
        if cached_grid is not None:
            # Cache Hit → verwende gecachtes Grid
            return cached_grid.to_surface_grid(geom)
        else:
            # Cache Miss → muss neu berechnet werden
            geometries_to_process.append((geom, cache_key))
```

#### 2. Cache-Key-Generierung

```python
def _make_surface_cache_key(self, geom, resolution, min_points):
    # Key enthält:
    # - surface_id
    # - orientation
    # - resolution
    # - min_points
    # - points_signature (Hash der Punkte)
    # - geometry_version (aus Settings.geometry_version)
    
    # geometry_version wird erhöht bei:
    # - add_surface_definition()
    # - remove_surface_definition()
    # - Änderungen an Surface-Punkten
```

#### 3. Cache-Invalidierung (Aktuell)

**Problem:** Keine gezielte Invalidierung für einzelne Surfaces!

- Bei `geometry_version` Änderung → **ALLE** Cache-Einträge werden ungültig (weil geometry_version im Key)
- Bei hide/disable Änderung → **KEINE** Cache-Invalidierung (hide/disable nicht im Key)
- Bei einzelnen Surface-Änderungen → **ALLE** Surfaces werden neu berechnet

---

## 3. Gewünschte Surface-Cache Logik

### Anforderungen

1. **Wenn Surfaces/Surface-Gruppen unverändert:**
   - ✅ Grid-Cache nutzen
   - ✅ SPL neu berechnen (Grid bleibt gleich)

2. **Wenn nur ein Surface geändert:**
   - ✅ Nur dieses Surface neu berechnen (Grid + SPL)
   - ✅ Andere Surfaces aus Cache nutzen

3. **Wenn Surfaces unverändert:**
   - ✅ Grid-Cache nutzen
   - ✅ SPL neu berechnen

4. **Bei hide/disable Änderungen:**
   - ✅ Cache für betroffene Surfaces löschen/neu erstellen

### Implementierungsvorschlag

#### 1. Cache-Key ohne geometry_version

```python
def _make_surface_cache_key(self, geom, resolution, min_points):
    # Key OHNE geometry_version (nur Surface-spezifisch)
    return (
        geom.surface_id,
        geom.orientation,
        float(resolution),
        int(min_points),
        self._calculate_points_signature(geom.points),  # Hash der Punkte
        # geometry_version NICHT im Key!
    )
```

#### 2. Gezielte Cache-Invalidierung

```python
def invalidate_surface_cache(self, surface_id: str):
    """Invalidiert Cache für ein spezifisches Surface"""
    from Module_LFO.Modules_Init.CacheManager import cache_manager, CacheType
    
    def predicate(key):
        # Key ist Tuple: (surface_id, orientation, resolution, min_points, points_signature)
        return isinstance(key, tuple) and len(key) > 0 and key[0] == surface_id
    
    cache_manager.invalidate_cache(CacheType.GRID, predicate=predicate)

def invalidate_surface_group_cache(self, group_id: str):
    """Invalidiert Cache für alle Surfaces einer Gruppe"""
    # Hole alle Surface-IDs der Gruppe
    surface_ids = self._get_surface_ids_for_group(group_id)
    for surface_id in surface_ids:
        self.invalidate_surface_cache(surface_id)
```

#### 3. Hide/Disable Handling

```python
def on_surface_hide_changed(self, surface_id: str, hidden: bool):
    """Wird aufgerufen, wenn hide-Status eines Surfaces geändert wird"""
    if hidden:
        # Surface wird versteckt → Cache löschen (wird bei Unhide neu berechnet)
        self.invalidate_surface_cache(surface_id)
    # Wenn unhidden: Cache wird automatisch neu erstellt bei nächster Berechnung

def on_surface_enable_changed(self, surface_id: str, enabled: bool):
    """Wird aufgerufen, wenn enable-Status eines Surfaces geändert wird"""
    if not enabled:
        # Surface wird deaktiviert → Cache löschen
        self.invalidate_surface_cache(surface_id)
    # Wenn enabled: Cache wird automatisch neu erstellt bei nächster Berechnung
```

#### 4. Surface-Änderungs-Tracking

```python
# In Settings oder FlexibleGridGenerator
self._surface_change_tracker: dict[str, int] = {}
# Key: surface_id
# Value: Änderungs-Version (wird bei Änderungen erhöht)

def track_surface_change(self, surface_id: str):
    """Markiert ein Surface als geändert"""
    self._surface_change_tracker[surface_id] = \
        self._surface_change_tracker.get(surface_id, 0) + 1
    
    # Invalidiere Cache für dieses Surface
    self.invalidate_surface_cache(surface_id)

def is_surface_unchanged(self, surface_id: str) -> bool:
    """Prüft ob Surface seit letztem Cache-Eintrag unverändert ist"""
    # Implementierung: Vergleich mit Cache-Eintrag-Version
    ...
```

#### 5. Berechnungs-Logik

```python
def calculate_spl(self, ...):
    # 1. Prüfe welche Surfaces geändert wurden
    changed_surfaces = self._get_changed_surfaces()
    
    # 2. Für unveränderte Surfaces:
    #    - Nutze Grid aus Cache
    #    - Berechne SPL neu (Grid bleibt gleich)
    
    # 3. Für geänderte Surfaces:
    #    - Berechne Grid neu
    #    - Berechne SPL neu
    
    # 4. Für hide/disable Surfaces:
    #    - Cache wurde bereits gelöscht
    #    - Überspringe Berechnung
```

---

## 4. Weitere Caches im System

### Plot-Caches

#### 1. Surface Actor Cache (`Plot3DSPL.py`)

```python
self._surface_actors: dict[str, Any] = {}
# Key: surface_id
# Value: {'mesh': mesh, 'actor': actor, 'signature': signature}

# Logik:
# - Mesh wird gecacht, wenn Signatur übereinstimmt
# - Signatur basiert auf: Subdivision-Level, Vertex-Count, Bounds
# - Bei Signatur-Änderung → Mesh wird neu erstellt
```

#### 2. Surface Texture Cache (`Plot3D.py`)

```python
self._surface_texture_cache: dict[str, str] = {}
# Key: surface_id
# Value: texture_signature (String)

# Logik:
# - Textur-Signatur wird gespeichert
# - Bei Signatur-Änderung → Textur wird neu erstellt
```

#### 3. Plot Geometry Cache (`Plot3D.py`)

```python
self._plot_geometry_cache = None
# Cache für Plot-Geometrie (plot_x, plot_y)

# Logik:
# - Wird verwendet für Plot-Koordinaten
# - Wird bei Geometrie-Änderungen invalidiert
```

#### 4. Time Mode Cache (`Plot3D.py`)

```python
self._time_mode_surface_cache: dict | None = None
# Cache für Time-Mode Surface-Daten

# Logik:
# - Wird verwendet für Time-Mode Visualisierung
# - Wird bei Time-Mode-Änderungen invalidiert
```

#### 5. Overlay Signatures (`Plot3D.py`)

```python
self._last_overlay_signatures: dict[str, tuple] = {}
# Key: Kategorie (z.B. "axes", "walls", "speakers")
# Value: Signatur (Tuple)

# Logik:
# - Signatur-Vergleich für Overlays
# - Bei Signatur-Änderung → Overlay wird neu gezeichnet
```

### Calc-Caches

#### 1. Geometry Cache (`SoundfieldCalculator.py`)

```python
self._geometry_cache_obj: LRUCache  # CacheType.CALC_GEOMETRY
# Key: source_key (String)
# Value: {distances, azimuths, elevations}

# Logik:
# - Wird verwendet für Geometrie-Berechnungen pro Quelle
# - LRU-Cache mit max_size=1000
```

#### 2. Grid Cache (`SoundfieldCalculator.py`)

```python
self._grid_cache_obj: LRUCache  # CacheType.CALC_GRID
# Key: grid_hash (String)
# Value: Gespeichertes Grid

# Logik:
# - Wird verwendet für Grid-Caching in Calc
# - LRU-Cache mit max_size=100
```

### Impulse-Caches (`ImpulseCalculator.py`)

```python
self._frequency_cache: dict = {}
# Cache für Frequenz-Berechnungen

self._balloon_angle_cache: dict = {}
# Cache für Balloon-Winkel-Berechnungen

self._speaker_array_cache = None
# Cache für Speaker-Array-Daten
```

---

## 5. Zusammenfassung

### Speaker-Cache
- ✅ **Gut strukturiert**: Mehrstufige Cache-Hierarchie (Array → Stack → Speaker)
- ✅ **Gezielte Invalidierung**: `clear_array_cache()` für einzelne Arrays
- ✅ **Signatur-basiert**: Parameter-Signaturen für Cache-Vergleich
- ⚠️ **Viele Cache-Typen**: 9 verschiedene Cache-Dictionaries

### Surface-Cache (Aktuell)
- ❌ **Keine gezielte Invalidierung**: geometry_version invalidiert ALLE Surfaces
- ❌ **Hide/Disable nicht berücksichtigt**: Cache wird nicht gelöscht
- ❌ **Keine einzelne Surface-Invalidierung**: Alle Surfaces werden neu berechnet
- ✅ **Gut**: Cache-Manager Integration vorhanden

### Weitere Caches
- **Plot-Caches**: 5 verschiedene Caches (Surface Actors, Texture, Geometry, Time Mode, Overlay Signatures)
- **Calc-Caches**: 2 Caches (Geometry, Grid)
- **Impulse-Caches**: 3 Caches (Frequency, Balloon Angle, Speaker Array)

---

## 6. Empfohlene Implementierung für Surface-Cache

### Schritt 1: Cache-Key ohne geometry_version

```python
# Entferne geometry_version aus Cache-Key
# Stattdessen: Track Surface-Änderungen individuell
```

### Schritt 2: Gezielte Cache-Invalidierung

```python
# Implementiere invalidate_surface_cache()
# Implementiere invalidate_surface_group_cache()
```

### Schritt 3: Hide/Disable Handling

```python
# Bei hide/disable Änderungen → Cache löschen
# Bei unhide/enable → Cache wird automatisch neu erstellt
```

### Schritt 4: Surface-Änderungs-Tracking

```python
# Track individuelle Surface-Änderungen
# Vergleiche mit Cache-Einträgen
```

### Schritt 5: Berechnungs-Logik

```python
# Unterscheide zwischen:
# - Unveränderte Surfaces → Grid aus Cache, SPL neu
# - Geänderte Surfaces → Grid + SPL neu
# - Hide/Disable Surfaces → Überspringen
```

