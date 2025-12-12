"""Hilfsfunktionen für den 3D-SPL-Plot."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition

try:
    import pyvista as pv
except Exception:
    pv = None  # type: ignore


# ------------------------------------------------------------------
# Statische Utility-Funktionen
# ------------------------------------------------------------------


def has_valid_data(x, y, pressure) -> bool:
    """Prüft, ob die Daten gültig sind (nicht None und nicht leer)."""
    if x is None or y is None or pressure is None:
        return False
    try:
        return len(x) > 0 and len(y) > 0 and len(pressure) > 0
    except TypeError:
        return False


def compute_surface_signature(x: np.ndarray, y: np.ndarray) -> tuple:
    """Berechnet eine Signatur für eine Surface basierend auf X/Y-Koordinaten."""
    if x.size == 0 or y.size == 0:
        return (0, 0, 0.0, 0.0, 0.0, 0.0)
    return (
        int(x.size),
        float(x[0]),
        float(x[-1]),
        int(y.size),
        float(y[0]),
        float(y[-1]),
    )


def quantize_to_steps(values: np.ndarray, step: float) -> np.ndarray:
    """Quantisiert Werte auf diskrete Schritte."""
    if step <= 0:
        return values
    return np.round(values / step) * step


def fallback_cmap_to_lut(cmap, n_colors: int | None = None, flip: bool = False):
    """Konvertiert eine Colormap zu einer PyVista LookupTable."""
    if pv is None:
        return None
    if isinstance(cmap, pv.LookupTable):
        lut = cmap
    else:
        lut = pv.LookupTable()
        lut.apply_cmap(cmap, n_values=n_colors or 256, flip=flip)
    if flip and isinstance(cmap, pv.LookupTable):
        lut.values[:] = lut.values[::-1]
    return lut


def to_float_array(values) -> np.ndarray:
    """Konvertiert Werte zu einem 1D-Float-Array."""
    if values is None:
        return np.empty(0, dtype=float)
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return np.empty(0, dtype=float)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(float)


# ------------------------------------------------------------------
# Mixin-Klasse für Instanz-Methoden
# ------------------------------------------------------------------


class SPL3DHelpers:
    """
    Mixin-Klasse für Hilfsfunktionen im 3D-SPL-Plot.

    Diese Klasse kapselt Instanz-Methoden, die Hilfsfunktionen bereitstellen,
    aber Zugriff auf Instanz-Attribute benötigen.

    Sie erwartet, dass die aufnehmende Klasse folgende Attribute bereitstellt:
    - `plotter` (PyVista Plotter)
    - `overlay_axis` oder `overlay_surfaces` (für _get_max_surface_dimension)
    """

    def _remove_actor(self, name: str):
        """Entfernt einen Actor aus dem Plotter."""
        try:
            self.plotter.remove_actor(name)
        except KeyError:
            pass

    def _compute_overlay_signatures(self, settings, container) -> dict[str, tuple]:
        """Erzeugt robuste Vergleichs-Signaturen für jede Overlay-Kategorie.

        Hintergrund:
        - Ziel ist, Änderungsdetektion ohne tiefe Objektvergleiche.
        - Wir wandeln relevante Settings in flache Tupel um, die sich leicht
          vergleichen lassen und keine PyVista-Objekte enthalten.
        """
        def _to_tuple(sequence) -> tuple:
            if sequence is None:
                return tuple()
            try:
                iterable = list(sequence)
            except TypeError:
                return (float(sequence),)
            result = []
            for item in iterable:
                try:
                    value = float(item)
                except Exception:
                    result.append(None)
                else:
                    if np.isnan(value):
                        result.append(None)
                    else:
                        result.append(value)
            return tuple(result)

        def _to_str_tuple(sequence) -> tuple:
            if sequence is None:
                return tuple()
            if isinstance(sequence, (str, bytes)):
                return (sequence.decode('utf-8', errors='ignore') if isinstance(sequence, bytes) else str(sequence),)
            try:
                iterable = list(sequence)
            except TypeError:
                return (str(sequence),)
            result: List[str] = []
            for item in iterable:
                if isinstance(item, bytes):
                    result.append(item.decode('utf-8', errors='ignore'))
                else:
                    result.append(str(item))
            return tuple(result)

        # 🎯 WICHTIG: selected_axis zur Signatur hinzufügen, damit Highlight-Änderungen erkannt werden
        # Berechne maximale Surface-Dimension für Achsenflächen-Größe
        # Verwende overlay_axis oder overlay_surfaces (beide haben die Methode)
        if hasattr(self, 'overlay_axis'):
            max_surface_dim = self.overlay_axis._get_max_surface_dimension(settings)
        elif hasattr(self, 'overlay_surfaces'):
            max_surface_dim = self.overlay_surfaces._get_max_surface_dimension(settings)
        else:
            max_surface_dim = 0.0
        
        # 🎯 ACHSEN-SIGNATUR: Achspositionen + aktive Surface-Geometrien,
        # damit verschobene/editiert Surfaces die Achsen neu zeichnen.
        surface_points_signature: List[tuple] = []
        try:
            overlay_for_signature = None
            if hasattr(self, 'overlay_axis'):
                overlay_for_signature = self.overlay_axis
            elif hasattr(self, 'overlay_surfaces'):
                overlay_for_signature = self.overlay_surfaces
            if overlay_for_signature and hasattr(overlay_for_signature, '_get_active_xy_surfaces'):
                active_surfaces = overlay_for_signature._get_active_xy_surfaces(settings)
                for surface_id, surface in active_surfaces:
                    if isinstance(surface, SurfaceDefinition):
                        points = surface.points
                    else:
                        points = surface.get('points', [])
                    points_tuple: List[tuple] = []
                    for point in points:
                        try:
                            x = float(point.get('x', 0.0))
                            y = float(point.get('y', 0.0))
                            z = float(point.get('z', 0.0)) if point.get('z') is not None else 0.0
                            points_tuple.append((round(x, 6), round(y, 6), round(z, 6)))
                        except Exception:
                            continue
                    surface_points_signature.append((str(surface_id), tuple(points_tuple)))
                surface_points_signature.sort(key=lambda x: x[0])
        except Exception:
            # Bei Fehler: Verwende leere Signatur (bedeutet, dass Achsen bei jedem Update neu gezeichnet werden)
            # Dies stellt sicher, dass beim Laden die Achsen immer neu gezeichnet werden, auch wenn die Signatur-Berechnung fehlschlägt
            surface_points_signature = []  # Leere Signatur = immer neu zeichnen

        axis_signature = (
            float(getattr(settings, 'position_x_axis', 0.0)),
            float(getattr(settings, 'position_y_axis', 0.0)),
            float(getattr(settings, 'length', 0.0)),
            float(getattr(settings, 'width', 0.0)),
            float(getattr(settings, 'axis_3d_transparency', 10.0)),
            float(max_surface_dim),  # Maximale Surface-Dimension für Achsenflächen-Größe
            getattr(self, '_axis_selected', None),  # Highlight-Status in Signatur aufnehmen
            tuple(surface_points_signature),  # Aktive Surface-Geometrien
        )
        print(f"[DEBUG Plot3DHelpers._compute_overlay_signatures] axis_signature erstellt: {len(surface_points_signature)} Surfaces in Signatur")

        speaker_arrays = getattr(settings, 'speaker_arrays', {})
        speakers_signature: List[tuple] = []

        if isinstance(speaker_arrays, dict):
            for name in sorted(speaker_arrays.keys()):
                array = speaker_arrays[name]
                configuration = str(getattr(array, 'configuration', '') or '').lower()
                hide = bool(getattr(array, 'hide', False))
                
                # Hole Array-Positionen
                array_pos_x = getattr(array, 'array_position_x', 0.0)
                array_pos_y = getattr(array, 'array_position_y', 0.0)
                array_pos_z = getattr(array, 'array_position_z', 0.0)
                
                # Für die Signatur verwenden wir die Plot-Positionen:
                # Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
                # Flown: Absolute Positionen (bereits in source_position_x/y/z_flown enthalten)
                if configuration == 'flown':
                    # Für Flown: Array-Positionen sind bereits in source_position_x/y/z_flown enthalten
                    xs = _to_tuple(getattr(array, 'source_position_x', None))
                    ys = _to_tuple(
                        getattr(
                            array,
                            'source_position_calc_y',
                            getattr(array, 'source_position_y', None),
                        )
                    )
                    zs_stack = None
                    zs_flown = _to_tuple(getattr(array, 'source_position_z_flown', None))
                else:
                    # Für Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
                    xs_raw = _to_tuple(getattr(array, 'source_position_x', None))
                    ys_raw = _to_tuple(getattr(array, 'source_position_y', None))
                    zs_raw = _to_tuple(getattr(array, 'source_position_z_stack', None))
                    
                    # Addiere Array-Positionen zu den Speaker-Positionen
                    if xs_raw and array_pos_x != 0.0:
                        xs = tuple(x + array_pos_x for x in xs_raw)
                    else:
                        xs = xs_raw
                    if ys_raw and array_pos_y != 0.0:
                        ys = tuple(y + array_pos_y for y in ys_raw)
                    else:
                        ys = ys_raw
                    if zs_raw and array_pos_z != 0.0:
                        zs_stack = tuple(z + array_pos_z for z in zs_raw)
                    else:
                        zs_stack = zs_raw
                    zs_flown = None
                
                azimuth = _to_tuple(getattr(array, 'source_azimuth', None))
                angles = _to_tuple(getattr(array, 'source_angle', None))
                polar_pattern = _to_str_tuple(getattr(array, 'source_polar_pattern', None))
                source_types = _to_str_tuple(getattr(array, 'source_type', None))
                
                sources = (
                    tuple(xs) if xs else tuple(),
                    tuple(ys) if ys else tuple(),
                    tuple(zs_stack) if zs_stack is not None else tuple(),
                    tuple(zs_flown) if zs_flown is not None else tuple(),
                    tuple(azimuth) if azimuth else tuple(),
                    tuple(angles) if angles else tuple(),
                    (round(float(array_pos_x), 4), round(float(array_pos_y), 4), round(float(array_pos_z), 4)),
                )
                speakers_signature.append(
                    (
                        str(name),
                        configuration,
                        hide,
                        sources,
                        polar_pattern,
                        source_types,
                    )
                )
        speakers_signature_tuple = tuple(speakers_signature)
        
        # 🚀 OPTIMIERUNG: Highlight-IDs NICHT zur Signatur hinzufügen
        # Wenn sich nur Highlights ändern, sollen nicht alle Speaker neu gezeichnet werden
        # Stattdessen wird nur _update_speaker_highlights() aufgerufen
        # Separate Highlight-Signatur für schnelles Highlight-Update ohne Neuzeichnen
        highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
        highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
        highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])
        
        if highlight_array_ids:
            highlight_array_ids_list = [str(aid) for aid in highlight_array_ids]
        elif highlight_array_id:
            highlight_array_ids_list = [str(highlight_array_id)]
        else:
            highlight_array_ids_list = []
        
        if isinstance(highlight_indices, (list, tuple, set)):
            highlight_indices_tuple = tuple(sorted((str(aid), int(idx)) for aid, idx in highlight_indices))
        else:
            highlight_indices_tuple = tuple()
        
        highlight_array_ids_tuple = tuple(sorted(highlight_array_ids_list))
        highlight_signature = (highlight_array_ids_tuple, highlight_indices_tuple)
        
        speakers_signature_with_highlights = speakers_signature_tuple

        impulse_points = getattr(settings, 'impulse_points', []) or []
        impulse_signature: List[tuple] = []
        for point in impulse_points:
            try:
                data = point['data']
                x_val, y_val = data
                impulse_signature.append((float(x_val), float(y_val)))
            except Exception:
                continue
        impulse_signature_tuple = tuple(sorted(impulse_signature))

        # Surfaces-Signatur (optimiert: nur geänderte Surfaces neu berechnen)
        surface_definitions = getattr(settings, 'surface_definitions', {})
        surfaces_signature: List[tuple] = []
        
        # Cache für Surface-Signaturen (nur bei wiederholten Aufrufen)
        if not hasattr(self, '_surface_signature_cache'):
            self._surface_signature_cache: dict[str, tuple] = {}
        
        from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition
        
        if isinstance(surface_definitions, dict):
            for surface_id in sorted(surface_definitions.keys()):
                surface_def = surface_definitions[surface_id]
                if isinstance(surface_def, SurfaceDefinition):
                    enabled = bool(getattr(surface_def, 'enabled', False))
                    hidden = bool(getattr(surface_def, 'hidden', False))
                    points = getattr(surface_def, 'points', []) or []
                else:
                    enabled = bool(surface_def.get('enabled', False))
                    hidden = bool(surface_def.get('hidden', False))
                    points = surface_def.get('points', [])
                
                # Erstelle schnelle Hash-Signatur für Vergleich
                # (nur enabled/hidden + Anzahl Punkte + Hash der Punkte)
                points_hash = hash(tuple(
                    (float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('z', 0.0)))
                    for p in points[:10]  # Nur erste 10 Punkte für Hash (Performance)
                )) if points else 0
                
                # Prüfe ob Surface sich geändert hat
                cache_key = f"{surface_id}_{enabled}_{hidden}_{len(points)}_{points_hash}"
                if cache_key in self._surface_signature_cache:
                    # Verwende gecachte Signatur
                    surfaces_signature.append(self._surface_signature_cache[cache_key])
                    continue
                
                # Neu berechnen nur wenn nötig
                points_tuple = []
                for point in points:
                    try:
                        x = float(point.get('x', 0.0))
                        y = float(point.get('y', 0.0))
                        z = float(point.get('z', 0.0))
                        points_tuple.append((x, y, z))
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                sig = (
                    str(surface_id),
                    enabled,
                    hidden,
                    tuple(points_tuple)
                )
                surfaces_signature.append(sig)
                # Cache speichern
                self._surface_signature_cache[cache_key] = sig
        
        surfaces_signature_tuple = tuple(surfaces_signature)
        
        # 🎯 WICHTIG: has_speaker_arrays wird NICHT in die Signatur aufgenommen,
        # da Surfaces unabhängig von der Anwesenheit von Sources gezeichnet werden sollen.
        # has_speaker_arrays beeinflusst nur die Darstellung (gestrichelt vs. durchgezogen),
        # nicht ob Surfaces gezeichnet werden.
        # Die Signatur besteht aus den Surface-Definitionen, active_surface_id und active_surface_highlight_ids.
        # 🎯 WICHTIG: active_ids_set muss sowohl active_surface_id als auch active_surface_highlight_ids enthalten
        # (wie in draw_surfaces), damit die Signaturen übereinstimmen!
        active_surface_id = getattr(settings, 'active_surface_id', None)
        highlight_ids = getattr(settings, 'active_surface_highlight_ids', None)
        active_ids_set = set()
        if isinstance(highlight_ids, (list, tuple, set)):
            active_ids_set = {str(sid) for sid in highlight_ids}
        if active_surface_id is not None:
            active_ids_set.add(str(active_surface_id))
        highlight_ids_tuple = tuple(sorted(active_ids_set))
        
        # 🎯 Prüfe ob SPL-Daten vorhanden sind (für Signatur, damit draw_surfaces nach update_spl_plot aufgerufen wird)
        has_spl_data_for_signature = False
        try:
            if hasattr(self, 'plotter') and hasattr(self.plotter, 'renderer') and hasattr(self.plotter.renderer, 'actors'):
                spl_surface_actor = self.plotter.renderer.actors.get('spl_surface')
                spl_floor_actor = self.plotter.renderer.actors.get('spl_floor')
                texture_actor_names = [name for name in self.plotter.renderer.actors.keys() if name.startswith('spl_surface_tex_')]
                has_texture_actors = len(texture_actor_names) > 0
                # Prüfe auch _surface_texture_actors direkt (falls Actors noch nicht im Renderer registriert sind)
                has_texture_actors_direct = hasattr(self, '_surface_texture_actors') and len(self._surface_texture_actors) > 0
                if spl_surface_actor is not None or spl_floor_actor is not None or has_texture_actors or has_texture_actors_direct:
                    has_spl_data_for_signature = True
        except Exception:
            pass
        
        surfaces_signature_with_active = (surfaces_signature_tuple, active_surface_id, highlight_ids_tuple, has_spl_data_for_signature)
        
        result = {
            'axis': axis_signature,
            'speakers': speakers_signature_with_highlights,  # 🚀 OPTIMIERT: Enthält KEINE Highlight-IDs mehr
            'speakers_highlights': highlight_signature,  # 🚀 NEU: Separate Highlight-Signatur für schnelles Update
            'impulse': impulse_signature_tuple,
            'surfaces': surfaces_signature_with_active,  # Enthält active_surface_id, highlight_ids und has_spl_data
        }
        
        return result


__all__ = [
    'SPL3DHelpers',
    'has_valid_data',
    'compute_surface_signature',
    'quantize_to_steps',
    'fallback_cmap_to_lut',
    'to_float_array',
]

