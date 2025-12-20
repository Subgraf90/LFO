from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Module_LFO.Modules_Init.Logging import perf_section
from Module_LFO.Modules_Plot.Plot_SPL_3D.Plot3DOverlaysBase import SPL3DOverlayBase


class SPL3DSpeakerMixin(SPL3DOverlayBase):
    """
    Mixin-Klasse, die alle Lautsprecher-Zeichenroutinen und zugehörige
    Geometrie-/Cache-Logik für den SPL-3D-Plot kapselt.

    Sie erwartet, dass die aufnehmende Klasse (z. B. `SPL3DOverlayRenderer`)
    folgende Attribute/Methoden bereitstellt:

    - `plotter`
    - `pv` (PyVista-Modul)
    - `_speaker_actor_cache`
    - `_speaker_geometry_cache`
    - `_speaker_geometry_param_cache`
    - `_overlay_array_cache`
    - `_array_geometry_cache`
    - `_array_signature_cache`
    - `_stack_geometry_cache`
    - `_stack_signature_cache`
    - `_geometry_cache_max_size`
    - `_add_overlay_mesh(...)`
    - `_remove_actor(name: str)`
    """
    
    def __init__(self, plotter: Any, pv_module: Any):
        """Initialisiert das Speaker Overlay."""
        super().__init__(plotter, pv_module)
        self._overlay_prefix = "spk_"
        
        # Speaker-spezifische Caches
        self._speaker_actor_cache: dict[tuple[str, int, int | str], dict[str, Any]] = {}
        self._speaker_geometry_cache: dict[str, List[Tuple[Any, Optional[int]]]] = {}
        self._speaker_geometry_param_cache: dict[tuple[str, int], tuple] = {}
        self._overlay_array_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._geometry_cache_max_size = 100
        # Mapping von array_id zu Cache-Keys für effizientes Löschen
        self._array_id_to_cache_keys: dict[str, set[str]] = {}
        self._array_geometry_cache: dict[str, dict[str, Any]] = {}
        self._array_signature_cache: dict[str, tuple] = {}
        self._stack_geometry_cache: dict[tuple[str, tuple], dict[str, Any]] = {}
        self._stack_signature_cache: dict[tuple[str, tuple], tuple] = {}
        self._box_template_cache: dict[tuple[float, float, float], Any] = {}
        self._box_face_cache: dict[tuple[float, float, float], Tuple[Optional[int], Optional[int]]] = {}
    
    def _create_speaker_actor_name(self, array_id: str, speaker_idx: int, geom_idx: int | str) -> str:
        """Erstellt einen eindeutigen Actor-Namen für einen Speaker basierend auf Array-ID, Speaker-Index und Geometrie-Index.
        
        Args:
            array_id: Die eindeutige ID des Arrays (als String)
            speaker_idx: Der Index des Speakers im Array
            geom_idx: Der Index der Geometrie (0 für Hauptkörper, oder andere Indizes für zusätzliche Geometrien)
        
        Returns:
            Eindeutiger Actor-Name im Format: {prefix}speaker_{array_id}_{speaker_idx}_{geom_idx}
        """
        prefix = f"{self._overlay_prefix}" if self._overlay_prefix else ""
        actor_name = f"{prefix}speaker_{array_id}_{speaker_idx}_{geom_idx}"
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "ACTOR_NAME",
                    "location": "PlotSPL3DSpeaker.py:_create_speaker_actor_name",
                    "message": "Creating unique actor name",
                    "data": {
                        "array_id": str(array_id),
                        "speaker_idx": int(speaker_idx),
                        "geom_idx": str(geom_idx),
                        "actor_name": actor_name
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        return actor_name
    
    def clear_category(self, category: str) -> None:
        """Überschreibt clear_category, um Speaker-spezifische Caches zu behandeln."""
        super().clear_category(category)
        if category == 'speakers':
            keys_to_remove = [key for key, info in self._speaker_actor_cache.items() if info.get('actor') in self._category_actors.get('speakers', [])]
            for key in keys_to_remove:
                self._speaker_actor_cache.pop(key, None)
    
    def clear_array_cache(self, array_id: str) -> None:
        """Löscht den Cache für ein spezifisches Array, damit Positionen neu berechnet werden."""
        array_id_str = str(array_id)
        # Entferne alle Cache-Einträge für dieses Array
        keys_to_remove = [key for key in self._speaker_actor_cache.keys() if isinstance(key, tuple) and len(key) > 0 and str(key[0]) == array_id_str]
        for key in keys_to_remove:
            actor_info = self._speaker_actor_cache.get(key)
            if actor_info and 'actor' in actor_info:
                try:
                    self._remove_actor(actor_info['actor'])
                except Exception:
                    pass
            self._speaker_actor_cache.pop(key, None)
        
        # Entferne aus persistentem Array-Cache
        keys_to_remove_array = [key for key in self._overlay_array_cache.keys() if isinstance(key, tuple) and len(key) > 0 and str(key[0]) == array_id_str]
        for key in keys_to_remove_array:
            self._overlay_array_cache.pop(key, None)
        
        # Entferne aus Geometry-Param-Cache
        keys_to_remove_geom = [key for key in self._speaker_geometry_param_cache.keys() if isinstance(key, tuple) and len(key) > 0 and str(key[0]) == array_id_str]
        for key in keys_to_remove_geom:
            self._speaker_geometry_param_cache.pop(key, None)
        
        # 🎯 WICHTIG: Entferne aus _speaker_geometry_cache
        # Verwende das Mapping, um nur die Cache-Keys für dieses Array zu löschen
        # #region agent log
        import json
        import time as time_module
        cache_keys_before = len(self._speaker_geometry_cache)
        geometry_cache_keys_for_array = len(self._array_id_to_cache_keys.get(array_id_str, set()))
        # #endregion
        if array_id_str in self._array_id_to_cache_keys:
            keys_to_remove_geometry = list(self._array_id_to_cache_keys[array_id_str])
            for key in keys_to_remove_geometry:
                self._speaker_geometry_cache.pop(key, None)
            # Lösche das Mapping für dieses Array
            del self._array_id_to_cache_keys[array_id_str]
        # #region agent log
        cache_keys_after = len(self._speaker_geometry_cache)
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1",
                "location": "PlotSPL3DSpeaker.py:clear_array_cache:85",
                "message": "Cache cleared for array",
                "data": {
                    "array_id": array_id_str,
                    "cache_keys_before": cache_keys_before,
                    "cache_keys_after": cache_keys_after,
                    "geometry_cache_keys_removed": geometry_cache_keys_for_array,
                    "cache_cleared": True
                },
                "timestamp": int(time_module.time() * 1000)
            }) + "\n")
        # #endregion
        
        # Entferne aus Array-Geometry-Cache
        if array_id_str in self._array_geometry_cache:
            del self._array_geometry_cache[array_id_str]
        if array_id_str in self._array_signature_cache:
            del self._array_signature_cache[array_id_str]
        
        # Entferne aus Stack-Cache (alle Einträge, die dieses Array enthalten)
        keys_to_remove_stack = [key for key in self._stack_geometry_cache.keys() if isinstance(key, tuple) and len(key) > 0 and str(key[0]) == array_id_str]
        for key in keys_to_remove_stack:
            self._stack_geometry_cache.pop(key, None)
        keys_to_remove_stack_sig = [key for key in self._stack_signature_cache.keys() if isinstance(key, tuple) and len(key) > 0 and str(key[0]) == array_id_str]
        for key in keys_to_remove_stack_sig:
            self._stack_signature_cache.pop(key, None)
    
    def clear(self) -> None:
        """Löscht alle Overlay-Actors und Speaker-Caches."""
        super().clear()
        # Speaker-spezifische Caches löschen
        self._speaker_actor_cache.clear()
        self._speaker_geometry_cache.clear()
        self._speaker_geometry_param_cache.clear()
        self._array_geometry_cache.clear()
        self._array_signature_cache.clear()
        self._stack_geometry_cache.clear()
        self._stack_signature_cache.clear()
        self._array_id_to_cache_keys.clear()
        # Box-Template-Cache behalten (Performance-Optimierung)
        # self._box_template_cache.clear()
        # self._box_face_cache.clear()

    # ------------------------------------------------------------------
    # Öffentliche API zum Zeichnen der Lautsprecher
    # ------------------------------------------------------------------
    def draw_speakers(self, settings, container, cabinet_lookup: Dict) -> None:  # noqa: C901
        with perf_section("PlotSPL3DOverlays.draw_speakers.setup"):
            speaker_arrays = getattr(settings, 'speaker_arrays', {})
            # #region agent log
            try:
                import json
                import time as time_module
                arrays_info = []
                if isinstance(speaker_arrays, dict):
                    for arr_name, arr in speaker_arrays.items():
                        arr_id = getattr(arr, 'id', arr_name)
                        config = getattr(arr, 'configuration', 'unknown')
                        hide = getattr(arr, 'hide', False)
                        arrays_info.append({
                            "array_id": str(arr_id),
                            "configuration": str(config),
                            "hide": bool(hide)
                        })
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H3",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:entry",
                        "message": "draw_speakers called - will process all arrays",
                        "data": {
                            "num_arrays": len(speaker_arrays) if isinstance(speaker_arrays, dict) else 0,
                            "arrays_info": arrays_info
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion

            # Wenn keine gültigen Arrays vorhanden sind, alle Speaker und den persistenten Array-Cache leeren
            if not isinstance(speaker_arrays, dict) or not speaker_arrays:
                self.clear_category('speakers')
                # Alle Arrays wurden effektiv gelöscht → persistenten Array-Cache komplett leeren
                self._overlay_array_cache.clear()
                return

            body_color = '#b0b0b0'
            exit_color = '#4d4d4d'

            # Hole Highlight-IDs für rote Umrandung (unterstütze Liste von IDs)
            highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
            highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
            highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])

            # Konvertiere zu Liste von Strings für konsistenten Vergleich
            if highlight_array_ids:
                highlight_array_ids_str = [str(aid) for aid in highlight_array_ids]
            elif highlight_array_id:
                highlight_array_ids_str = [str(highlight_array_id)]
            else:
                highlight_array_ids_str = []

            old_cache = self._speaker_actor_cache
            new_cache: Dict[Tuple[str, int, int | str], Dict[str, Any]] = {}
            new_actor_names: List[str] = []
            # #region agent log
            import json
            import time as time_module
            cache_size_before = len(self._overlay_array_cache) if hasattr(self, "_overlay_array_cache") else 0
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H4",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:110",
                    "message": "Cache status before rehydration",
                    "data": {
                        "overlay_array_cache_size": cache_size_before,
                        "old_cache_size": len(old_cache) if old_cache else 0
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
            # #endregion
            # Rehydriere Actor- und Signatur-Cache aus persistentem Array-Cache,
            # damit unveränderte Speaker ohne Neuaufbau übernommen werden können.
            # 🎯 FIX: Verwende speaker_array.id (die "alte" Erkennung) statt array_name (Dictionary-Key)
            # Erstelle Mapping von array_name zu speaker_array.id für korrekten Cache-Zugriff
            array_name_to_id_mapping: Dict[Any, str] = {}
            for arr_name, arr in speaker_arrays.items():
                arr_id = getattr(arr, 'id', None)
                if arr_id is not None:
                    array_name_to_id_mapping[arr_name] = str(arr_id)
                else:
                    array_name_to_id_mapping[arr_name] = str(arr_name)
            
            if hasattr(self, "_overlay_array_cache") and self._overlay_array_cache:
                for (arr_id_or_name, sp_idx), entry in self._overlay_array_cache.items():
                    # 🎯 FIX: Konvertiere arr_id_or_name zu speaker_array.id, falls es array_name ist
                    # Prüfe, ob arr_id_or_name ein array_name (Dictionary-Key) ist
                    array_id_for_cache = None
                    was_array_name = False
                    if arr_id_or_name in array_name_to_id_mapping:
                        # arr_id_or_name ist ein array_name (Dictionary-Key) - konvertiere zu speaker_array.id
                        array_id_for_cache = array_name_to_id_mapping[arr_id_or_name]
                        was_array_name = True
                    else:
                        # arr_id_or_name ist bereits speaker_array.id - verwende direkt
                        array_id_for_cache = str(arr_id_or_name)
                    
                    # #region agent log
                    try:
                        import json
                        import time as time_module
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "CACHE_REHYDRATE_ID",
                                "location": "PlotSPL3DSpeaker.py:draw_speakers:rehydrate_cache",
                                "message": "Rehydrating cache - ID conversion",
                                "data": {
                                    "arr_id_or_name_from_cache": str(arr_id_or_name),
                                    "arr_id_or_name_type": type(arr_id_or_name).__name__,
                                    "array_id_for_cache": array_id_for_cache,
                                    "was_array_name": bool(was_array_name),
                                    "speaker_idx": int(sp_idx)
                                },
                                "timestamp": int(time_module.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    
                    actor_entry = entry.get('actor_entry')
                    geom_sig = entry.get('geometry_param_signature')
                    if actor_entry:
                        # 🎯 FIX: Verwende array_id_for_cache (speaker_array.id) statt arr_id_or_name
                        old_cache[(array_id_for_cache, sp_idx, 0)] = actor_entry
                    if geom_sig is not None:
                        # 🎯 FIX: Verwende array_id_for_cache (speaker_array.id) statt arr_id_or_name
                        self._speaker_geometry_param_cache[(array_id_for_cache, sp_idx)] = geom_sig

        # 🚀 OPTIMIERUNG: Sammle alle Speaker für inkrementelles Update und Parallelisierung
        # Arrays und einzelne Speaker werden nur neu gezeichnet, wenn sie erstellt oder geändert wurden
        speaker_tasks: List[Dict] = []  # Liste aller Speaker, die verarbeitet werden müssen
        total_speakers = 0

        # 🎯 FIX: Prüfe, ob nur bestimmte Arrays neu gezeichnet werden sollen (z.B. bei hide-Status-Änderung)
        # Wenn _affected_array_ids_for_speakers gesetzt ist, zeichne nur diese Arrays neu
        affected_array_ids = getattr(self, '_affected_array_ids_for_speakers', None)
        # #region agent log
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "FIX-CHECK",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:affected_check",
                    "message": "Checking affected_array_ids_for_speakers",
                    "data": {
                        "affected_array_ids": list(affected_array_ids) if affected_array_ids else None,
                        "has_attr": hasattr(self, '_affected_array_ids_for_speakers')
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        for array_name, speaker_array in speaker_arrays.items():
            # #region agent log
            hide_status = getattr(speaker_array, 'hide', False)
            array_name_type = type(array_name).__name__
            array_name_str = str(array_name)
            speaker_array_id = getattr(speaker_array, 'id', None)
            speaker_array_id_type = type(speaker_array_id).__name__ if speaker_array_id is not None else None
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "ARRAY_ID_PLOT_DEBUG",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:array_loop_start",
                    "message": "Processing speaker array - array_id mapping",
                    "data": {
                        "array_name": array_name_str,
                        "array_name_type": array_name_type,
                        "speaker_array.id": str(speaker_array_id) if speaker_array_id is not None else None,
                        "speaker_array.id_type": speaker_array_id_type,
                        "hide": bool(hide_status),
                        "array_name == speaker_array.id": bool(array_name == speaker_array_id),
                        "affected_array_ids": list(affected_array_ids) if affected_array_ids else None,
                        "is_affected": str(array_name) in affected_array_ids if affected_array_ids else None
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion
            if hide_status:
                continue

            # 🎯 FIX: Wenn nur bestimmte Arrays betroffen sind, überspringe Arrays, die nicht betroffen sind
            # ABER: Nur wenn affected_array_ids gesetzt ist (bei hide-Status-Änderung)
            if affected_array_ids:
                array_id_to_check = str(speaker_array_id) if speaker_array_id is not None else str(array_name)
                if array_id_to_check not in affected_array_ids and str(array_name) not in affected_array_ids:
                    # Array ist nicht betroffen - überspringe es, aber entferne es nicht aus dem Cache
                    continue

            # 🎯 FIX: Verwende immer speaker_array.id (die "alte" Erkennung) statt array_name (Dictionary-Key)
            # Dies stellt sicher, dass die ID konsistent ist mit speakerspecs instance und TreeWidget
            if speaker_array_id is None:
                # Fallback: Wenn speaker_array.id nicht existiert, verwende array_name
                array_id = str(array_name)
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "ARRAY_ID_FALLBACK",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:array_id_fallback",
                            "message": "Using array_name as fallback (speaker_array.id is None)",
                            "data": {
                                "array_name": array_name_str,
                                "array_id": str(array_id)
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
            else:
                # Verwende speaker_array.id (die "alte" Erkennung)
                array_id = str(speaker_array_id)
            
            # array_name behalten für Zugriff auf speaker_arrays (da Keys Integer sein können)
            
            # #region agent log
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "ARRAY_ID_PLOT_DEBUG",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:array_id_converted",
                    "message": "Array ID converted for plotting - using speaker_array.id",
                    "data": {
                        "array_name": array_name_str,
                        "array_name_type": array_name_type,
                        "array_id": str(array_id),
                        "array_id_type": type(array_id).__name__,
                        "speaker_array.id": str(speaker_array_id) if speaker_array_id is not None else None,
                        "using_speaker_array_id": bool(speaker_array_id is not None),
                        "will_plot": True
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion

            configuration = getattr(speaker_array, 'configuration', '')
            config_lower = configuration.lower() if isinstance(configuration, str) else ''

            # Hole Array-Positionen
            array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
            array_pos_y = getattr(speaker_array, 'array_position_y', 0.0)
            array_pos_z = getattr(speaker_array, 'array_position_z', 0.0)
            # #region agent log
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H3",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:137",
                    "message": "Array positions retrieved",
                    "data": {
                        "array_id": str(array_id),
                        "configuration": str(configuration),
                        "array_pos_x": float(array_pos_x),
                        "array_pos_y": float(array_pos_y),
                        "array_pos_z": float(array_pos_z)
                    },
                    "timestamp": int(__import__('time').time() * 1000)
                }) + "\n")
            # #endregion

            if config_lower == 'flown':
                # Für Flown: Array-Positionen sind bereits in source_position_x/y/z_flown enthalten (absolute Positionen)
                xs = self._to_float_array(getattr(speaker_array, 'source_position_x', None))
                ys = self._to_float_array(
                    getattr(
                        speaker_array,
                        'source_position_calc_y',
                        getattr(speaker_array, 'source_position_y', None),
                    )
                )
                zs = self._to_float_array(getattr(speaker_array, 'source_position_z_flown', None))
                # #region agent log
                if zs.size > 0:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "F2",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:280",
                            "message": "Flown positions retrieved for plotting",
                            "data": {
                                "array_id": str(array_id),
                                "array_pos_x": float(array_pos_x),
                                "array_pos_y": float(array_pos_y),
                                "array_pos_z": float(array_pos_z),
                                "zs_first": float(zs[0]) if zs.size > 0 else None,
                                "zs_size": int(zs.size),
                                "source_position_z_flown_first": float(getattr(speaker_array, 'source_position_z_flown', [None])[0]) if hasattr(speaker_array, 'source_position_z_flown') and len(getattr(speaker_array, 'source_position_z_flown', [])) > 0 else None
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                # #endregion
            else:
                # Für Stack: Gehäusenullpunkt = Array-Position + Speaker-Position
                # source_position_calc_x/y/z ist nur für Berechnungen, nicht für den Plot
                xs_raw = self._to_float_array(getattr(speaker_array, 'source_position_x', None))
                ys_raw = self._to_float_array(getattr(speaker_array, 'source_position_y', None))
                zs_raw = self._to_float_array(getattr(speaker_array, 'source_position_z_stack', None))

                # Addiere Array-Positionen zu den Speaker-Positionen
                xs = xs_raw + array_pos_x if xs_raw.size > 0 else xs_raw
                ys = ys_raw + array_pos_y if ys_raw.size > 0 else ys_raw
                zs = zs_raw + array_pos_z if zs_raw.size > 0 else zs_raw
                # #region agent log
                if xs.size > 0:
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        xs_raw_preview = list(xs_raw[:5]) if xs_raw.size >= 5 else list(xs_raw) if xs_raw.size > 0 else []
                        xs_preview = list(xs[:5]) if xs.size >= 5 else list(xs) if xs.size > 0 else []
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H3-DETAIL",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:158",
                            "message": "Stack positions calculated - full preview",
                            "data": {
                                "array_id": str(array_id),
                                "xs_raw_preview": xs_raw_preview,
                                "xs_preview": xs_preview,
                                "xs_raw_length": int(xs_raw.size),
                                "xs_length": int(xs.size),
                                "array_pos_x": float(array_pos_x),
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                # #endregion

            min_len = min(xs.size, ys.size, zs.size)
            if min_len == 0:
                continue

            valid_mask = np.isfinite(xs[:min_len]) & np.isfinite(ys[:min_len]) & np.isfinite(zs[:min_len])
            valid_indices = np.nonzero(valid_mask)[0]

            # 🚀 OPTIMIERUNG: Array-Level Cache für Flown Arrays
            # 🎯 FIX: Cache nur für hide/unhide verwenden, nicht für Geometrieänderungen
            array_can_be_skipped = False
            array_cached_geometries: Dict[int, List[Tuple[Any, Optional[int]]]] = {}  # Cache für alle Speaker eines Arrays

            # 🎯 FIX: Deaktiviere Array-Cache für Flown-Arrays bei Geometrieänderungen
            # Cache wird nur noch für hide/unhide verwendet
            # Wenn sich Geometrieparameter ändern (z.B. array_position_z), soll der Cache nicht verwendet werden
            # Prüfe, ob sich nur der hide-Status geändert hat
            use_array_cache = False
            if config_lower == 'flown' and len(self._array_geometry_cache) > 0:
                # Prüfe Array-Signatur
                array_signature = self._create_array_signature(
                    speaker_array, array_id, array_pos_x, array_pos_y, array_pos_z, cabinet_lookup
                )
                cached_array_signature = self._array_signature_cache.get(array_id)

                # 🎯 FIX: Verwende Cache nur, wenn sich nur der hide-Status geändert hat
                # Wenn sich Geometrieparameter geändert haben, soll der Cache nicht verwendet werden
                # Die Signatur enthält alle Geometrieparameter, daher wird die Änderung erkannt
                # Wenn die Signaturen übereinstimmen, bedeutet das, dass sich nur der hide-Status geändert hat
                if cached_array_signature == array_signature:
                    use_array_cache = True
                else:
                    # 🎯 FIX: Wenn sich die Signatur geändert hat, bedeutet das, dass sich Geometrieparameter geändert haben
                    # In diesem Fall soll der Cache nicht verwendet werden
                    use_array_cache = False
                    # Lösche den Cache für dieses Array, damit er nicht mehr verwendet wird
                    if array_id in self._array_geometry_cache:
                        del self._array_geometry_cache[array_id]
                    if array_id in self._array_signature_cache:
                        del self._array_signature_cache[array_id]

            # 🎯 FIX: Verwende Array-Cache nur, wenn sich nur der hide-Status geändert hat
            if use_array_cache:
                    # Array-Signatur stimmt überein - lade alle Speaker aus Array-Cache
                    array_cache_data = self._array_geometry_cache.get(array_id)
                    if array_cache_data and isinstance(array_cache_data, dict):
                        cached_geometries = array_cache_data.get('geometries', {})
                        cached_array_pos = array_cache_data.get('array_pos', (array_pos_x, array_pos_y, array_pos_z))

                        # Transformiere alle Speaker-Geometrien zur neuen Position
                        old_array_x, old_array_y, old_array_z = cached_array_pos
                        # #region agent log
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H1",
                                "location": "PlotSPL3DSpeaker.py:draw_speakers:186",
                                "message": "Flown cache found - transformation parameters",
                                "data": {
                                    "array_id": str(array_id),
                                    "cached_array_pos": [float(cached_array_pos[0]), float(cached_array_pos[1]), float(cached_array_pos[2])],
                                    "current_array_pos": [float(array_pos_x), float(array_pos_y), float(array_pos_z)],
                                    "old_array_x": float(old_array_x),
                                    "old_array_y": float(old_array_y),
                                    "old_array_z": float(old_array_z)
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                        # #endregion
                        for idx in valid_indices:
                            x = xs[idx]
                            y = ys[idx]
                            z = zs[idx]

                            if idx in cached_geometries:
                                # Transformiere gecachte Geometrien
                                cached_geoms = cached_geometries[idx]
                                # #region agent log
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    import json
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H1",
                                        "location": "PlotSPL3DSpeaker.py:draw_speakers:198",
                                        "message": "Transforming flown speaker from cache",
                                        "data": {
                                            "array_id": str(array_id),
                                            "idx": int(idx),
                                            "old_x": float(old_array_x),
                                            "old_y": float(old_array_y),
                                            "old_z": float(old_array_z),
                                            "new_x": float(x),
                                            "new_y": float(y),
                                            "new_z": float(z)
                                        },
                                        "timestamp": int(__import__('time').time() * 1000)
                                    }) + "\n")
                                # #endregion
                                transformed_geoms = self._transform_cached_geometry_to_position(
                                    cached_geoms, old_array_x, old_array_y, old_array_z, x, y, z
                                )
                                array_cached_geometries[idx] = transformed_geoms

                        # Wenn alle Speaker aus Cache geladen werden konnten, überspringe Array
                        if len(array_cached_geometries) == len(valid_indices):
                            array_can_be_skipped = True
                            # #region agent log
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H1",
                                    "location": "PlotSPL3DSpeaker.py:draw_speakers:array_can_be_skipped",
                                    "message": "Array can be skipped - all speakers from cache",
                                    "data": {
                                        "array_id": str(array_id),
                                        "cached_count": int(len(array_cached_geometries)),
                                        "valid_indices_count": int(len(valid_indices)),
                                        "cached_indices": [int(i) for i in array_cached_geometries.keys()],
                                        "valid_indices": [int(i) for i in valid_indices],
                                        "old_array_pos": [float(old_array_x), float(old_array_y), float(old_array_z)],
                                        "new_array_pos": [float(array_pos_x), float(array_pos_y), float(array_pos_z)]
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }) + "\n")
                            # #endregion
                            # Füge alle Speaker zum neuen Cache hinzu (ohne Neuberechnung)
                            for idx in valid_indices:
                                x = xs[idx]
                                y = ys[idx]
                                z = zs[idx]

                                is_highlighted = False
                                if highlight_indices:
                                    highlight_indices_str = [(str(aid), int(idx_val)) for aid, idx_val in highlight_indices]
                                    is_highlighted = (array_id, int(idx)) in highlight_indices_str
                                elif highlight_array_ids_str:
                                    is_highlighted = array_id in highlight_array_ids_str

                                # Verwende transformierte Geometrien aus Cache
                                geometries = array_cached_geometries.get(idx)
                                if geometries:
                                    # #region agent log
                                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                        import json
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "H1",
                                            "location": "PlotSPL3DSpeaker.py:draw_speakers:adding_task_from_array_cache",
                                            "message": "Adding task from array cache",
                                            "data": {
                                                "array_id": str(array_id),
                                                "idx": int(idx),
                                                "x": float(x),
                                                "y": float(y),
                                                "z": float(z),
                                                "has_geometries": bool(geometries),
                                                "has_existing_actor": bool(old_cache.get((array_id, int(idx), 0)))
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                    # #endregion
                                    # Füge Task hinzu (aber mit needs_rebuild=False, da Geometrien bereits vorhanden)
                                    speaker_tasks.append({
                                        'array_id': array_id,  # 🎯 FIX: array_id ist jetzt speaker_array.id (die "alte" Erkennung)
                                        'speaker_array': speaker_array,  # speaker_array Objekt für direkten Zugriff
                                        'idx': idx,
                                        'x': float(x),
                                        'y': float(y),
                                        'z': float(z),
                                        'speaker_name': self._get_speaker_name(speaker_array, idx),
                                        'configuration': configuration,
                                        'is_highlighted': is_highlighted,
                                        'geometry_param_key': (array_id, int(idx)),
                                        'geometry_param_signature': self._create_geometry_param_signature(
                                            speaker_array, idx, float(x), float(y), float(z), cabinet_lookup
                                        ),
                                        'cached_param_signature': None,
                                        'needs_rebuild': False,  # Geometrien bereits aus Cache
                                        'existing_actor': old_cache.get((array_id, int(idx), 0)),
                                        'cached_geometries': geometries,  # Geometrien bereits vorhanden
                                    })

                            # Aktualisiere Array-Cache mit neuer Position
                            self._array_geometry_cache[array_id] = {
                                'geometries': {idx: [(mesh.copy(deep=True), exit_idx) for mesh, exit_idx in geoms]
                                               for idx, geoms in array_cached_geometries.items()},
                                'array_pos': (array_pos_x, array_pos_y, array_pos_z)
                            }

            # 🚀 OPTIMIERUNG: Stack-Level Cache für Stack Arrays
            stack_cached_geometries: Dict[tuple, Dict[int, List[Tuple[Any, Optional[int]]]]] = {}  # Cache für Stack-Gruppen: {stack_group_key: {sp_idx: geometries}}
            stack_groups_processed: set[tuple] = set()  # Welche Stack-Gruppen wurden bereits aus Cache geladen

            # FIX: Identifiziere Stack-Gruppen immer, nicht nur wenn Cache vorhanden ist
            # Dies stellt sicher, dass alle Stack-Lautsprecher verarbeitet werden
            stack_groups: Dict[tuple, List[int]] = {}
            if config_lower == 'stack':
                stack_groups = self._identify_stack_groups_from_cabinets(speaker_array, array_id, cabinet_lookup)
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:362",
                        "message": "Stack groups identified",
                        "data": {
                            "array_id": str(array_id),
                            "stack_groups_count": int(len(stack_groups)),
                            "stack_groups": {str(k): [int(i) for i in v] for k, v in stack_groups.items()},
                            "cache_size": int(len(self._stack_geometry_cache))
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion

            if config_lower == 'stack':
                # Prüfe jede Stack-Gruppe (auch wenn Cache leer ist, z.B. nach Hide/Unhide)
                for stack_group_key, speaker_indices in stack_groups.items():
                    # Prüfe Stack-Signatur
                    stack_signature = self._create_stack_signature(
                        speaker_array, array_id, stack_group_key, array_pos_x, array_pos_y, array_pos_z, cabinet_lookup, speaker_indices
                    )
                    cached_stack_signature = self._stack_signature_cache.get((array_id, stack_group_key))
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:370",
                            "message": "Stack signature check",
                            "data": {
                                "array_id": str(array_id),
                                "stack_group_key": str(stack_group_key),
                                "has_cached_signature": cached_stack_signature is not None,
                                "signatures_match": cached_stack_signature == stack_signature,
                                "current_signature": str(stack_signature)[:100] if stack_signature else None,
                                "cached_signature": str(cached_stack_signature)[:100] if cached_stack_signature else None
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                    # #endregion

                    if cached_stack_signature == stack_signature:
                        # Stack-Signatur stimmt überein - lade alle Speaker dieser Stack-Gruppe aus Cache
                        stack_cache_data = self._stack_geometry_cache.get((array_id, stack_group_key))
                    else:
                        # Stack-Signatur stimmt NICHT überein - Speaker müssen neu gezeichnet werden
                        # Markiere diese Stack-Gruppe als verarbeitet, damit sie nicht doppelt verarbeitet wird
                        # (aber die Speaker werden später in der normalen Schleife verarbeitet)
                        stack_groups_processed.add(stack_group_key)
                        stack_cache_data = None
                    
                    if cached_stack_signature == stack_signature and stack_cache_data:
                        # Stack-Signatur stimmt überein UND Cache-Daten vorhanden - lade alle Speaker dieser Stack-Gruppe aus Cache
                        # #region agent log
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H6",
                                "location": "PlotSPL3DSpeaker.py:draw_speakers:527",
                                "message": "Stack signature matches - checking cache",
                                "data": {
                                    "array_id": str(array_id),
                                    "stack_group_key": str(stack_group_key),
                                    "has_stack_cache_data": stack_cache_data is not None,
                                    "num_speaker_indices": len(speaker_indices),
                                    "num_valid_indices": len([i for i in speaker_indices if i in valid_indices])
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                        # #endregion
                        if stack_cache_data and isinstance(stack_cache_data, dict):
                            cached_geometries = stack_cache_data.get('geometries', {})
                            cached_stack_pos = stack_cache_data.get('stack_pos', (array_pos_x, array_pos_y, array_pos_z))

                            # Transformiere alle Speaker-Geometrien zur neuen Position
                            old_stack_x, old_stack_y, old_stack_z = cached_stack_pos
                            # #region agent log
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H1",
                                    "location": "PlotSPL3DSpeaker.py:draw_speakers:277",
                                    "message": "Stack cache found - transformation parameters",
                                    "data": {
                                        "array_id": str(array_id),
                                        "stack_group_key": str(stack_group_key),
                                        "cached_stack_pos": [float(cached_stack_pos[0]), float(cached_stack_pos[1]), float(cached_stack_pos[2])],
                                        "current_array_pos": [float(array_pos_x), float(array_pos_y), float(array_pos_z)],
                                        "old_stack_x": float(old_stack_x),
                                        "old_stack_y": float(old_stack_y),
                                        "old_stack_z": float(old_stack_z)
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }) + "\n")
                            # #endregion
                            stack_cached_geoms_for_group: Dict[int, List[Tuple[Any, Optional[int]]]] = {}

                            for idx in speaker_indices:
                                if idx not in valid_indices:
                                    continue

                                x = xs[idx]
                                y = ys[idx]
                                z = zs[idx]

                                if idx in cached_geometries:
                                    # Transformiere gecachte Geometrien
                                    cached_geoms = cached_geometries[idx]
                                    # Verwende ursprüngliche Speaker-Position statt Array-Position
                                    speaker_positions = stack_cache_data.get('speaker_positions', {})
                                    old_speaker_pos = speaker_positions.get(idx)
                                    if old_speaker_pos is None:
                                        # Fallback: Verwende Array-Position, wenn Speaker-Position nicht verfügbar
                                        old_speaker_pos = (old_stack_x, old_stack_y, old_stack_z)
                                    old_speaker_x, old_speaker_y, old_speaker_z = old_speaker_pos
                                    # #region agent log
                                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                        import json
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "H1",
                                            "location": "PlotSPL3DSpeaker.py:draw_speakers:294",
                                            "message": "Transforming stack speaker from cache",
                                            "data": {
                                                "array_id": str(array_id),
                                                "idx": int(idx),
                                                "old_array_x": float(old_stack_x),
                                                "old_array_y": float(old_stack_y),
                                                "old_array_z": float(old_stack_z),
                                                "old_speaker_x": float(old_speaker_x),
                                                "old_speaker_y": float(old_speaker_y),
                                                "old_speaker_z": float(old_speaker_z),
                                                "new_x": float(x),
                                                "new_y": float(y),
                                                "new_z": float(z),
                                                "has_speaker_positions": bool(speaker_positions),
                                                # Konvertiere Keys explizit zu Python-ints, damit json.dumps
                                                # keine Probleme mit z.B. numpy.int64 hat
                                                "speaker_positions_keys": [int(k) for k in speaker_positions.keys()] if speaker_positions else []
                                            },
                                            "timestamp": int(__import__('time').time() * 1000)
                                        }) + "\n")
                                    # #endregion
                                    transformed_geoms = self._transform_cached_geometry_to_position(
                                        cached_geoms, old_speaker_x, old_speaker_y, old_speaker_z, x, y, z
                                    )
                                    stack_cached_geoms_for_group[idx] = transformed_geoms

                            # Wenn alle Speaker aus Cache geladen werden konnten
                            expected_count = len([i for i in speaker_indices if i in valid_indices])
                            actual_count = len(stack_cached_geoms_for_group)
                            # #region agent log
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                import json
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H6",
                                    "location": "PlotSPL3DSpeaker.py:draw_speakers:612",
                                    "message": "Stack cache loading result",
                                    "data": {
                                        "array_id": str(array_id),
                                        "stack_group_key": str(stack_group_key),
                                        "expected_count": expected_count,
                                        "actual_count": actual_count,
                                        "all_loaded": actual_count == expected_count,
                                        "speaker_indices": [int(i) for i in speaker_indices],
                                        "cached_indices": [int(i) for i in stack_cached_geoms_for_group.keys()]
                                    },
                                    "timestamp": int(__import__('time').time() * 1000)
                                }) + "\n")
                            # #endregion
                            if actual_count == expected_count:
                                stack_cached_geometries[stack_group_key] = stack_cached_geoms_for_group
                                stack_groups_processed.add(stack_group_key)
                            else:
                                # 🎯 FIX: Nicht alle Speaker konnten aus Cache geladen werden
                                # (z.B. nach Hide/Unhide) - ignoriere Cache und zeichne alle neu
                                # #region agent log
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    import json
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H6",
                                        "location": "PlotSPL3DSpeaker.py:draw_speakers:632",
                                        "message": "Not all speakers loaded from cache - will rebuild",
                                        "data": {
                                            "array_id": str(array_id),
                                            "stack_group_key": str(stack_group_key),
                                            "expected_count": expected_count,
                                            "actual_count": actual_count
                                        },
                                        "timestamp": int(__import__('time').time() * 1000)
                                    }) + "\n")
                                # #endregion
                                # Entferne diese Stack-Gruppe aus dem Cache, damit sie neu berechnet wird
                                self._stack_geometry_cache.pop((array_id, stack_group_key), None)
                                self._stack_signature_cache.pop((array_id, stack_group_key), None)
                                # Markiere diese Stack-Gruppe als verarbeitet, damit sie nicht doppelt verarbeitet wird
                                stack_groups_processed.add(stack_group_key)

            # 🚀 OPTIMIERUNG: Prüfe ob Array komplett übersprungen werden kann
            # (nur wenn old_cache nicht leer ist, d.h. nicht beim ersten Laden)
            if not array_can_be_skipped and len(old_cache) > 0:
                # Prüfe ob alle Speaker des Arrays bereits im Cache existieren und unverändert sind
                all_speakers_unchanged = True
                array_speaker_tasks: List[Dict] = []  # Temporäre Liste für Speaker dieses Arrays

                # Debug: Zähle Speaker mit/ohne Cache
                speakers_with_cache = 0
                speakers_without_cache = 0
                speakers_missing = 0

                for idx in valid_indices:
                    # 🚀 OPTIMIERUNG: Überspringe Speaker, die bereits aus Stack-Cache geladen wurden
                    speaker_already_cached = False
                    if config_lower == 'stack':
                        for stack_group_key, cached_geoms in stack_cached_geometries.items():
                            if idx in cached_geoms:
                                speaker_already_cached = True
                                break

                    if speaker_already_cached:
                        continue

                    x = xs[idx]
                    y = ys[idx]
                    z = zs[idx]

                    # Prüfe ob dieser Speaker hervorgehoben werden soll
                    is_highlighted = False
                    if highlight_indices:
                        highlight_indices_str = [(str(aid), int(idx_val)) for aid, idx_val in highlight_indices]
                        is_highlighted = (array_id, int(idx)) in highlight_indices_str
                    elif highlight_array_ids_str:
                        # Prüfe ob Array-ID in der Liste ist
                        is_highlighted = array_id in highlight_array_ids_str

                    # Prüfe ob Speaker geändert werden muss
                    speaker_name = self._get_speaker_name(speaker_array, idx)
                    geometry_param_key = (array_id, int(idx))
                    geometry_param_signature = self._create_geometry_param_signature(
                        speaker_array, idx, float(x), float(y), float(z), cabinet_lookup
                    )
                    cached_param_signature = self._speaker_geometry_param_cache.get(geometry_param_key)

                    # Prüfe ob Actor bereits existiert und unverändert ist
                    key = (array_id, int(idx), 0)  # Prüfe ersten Geometrie-Key
                    existing_actor = old_cache.get(key)
                    needs_rebuild = True

                    # Prüfe ob Parameter-Signatur übereinstimmt
                    if cached_param_signature is not None and cached_param_signature == geometry_param_signature:
                        # Parameter-Signatur stimmt überein
                        if existing_actor:
                            # Actor existiert UND Signatur stimmt - Speaker ist komplett unverändert
                            needs_rebuild = False
                            speakers_with_cache += 1
                        else:
                            # Actor fehlt, aber Signatur stimmt - Geometrie kann aus Cache geladen werden
                            # Prüfe ob Geometrie-Cache existiert
                            cache_key = self._create_geometry_cache_key(
                                array_id, speaker_array, idx, speaker_name, configuration, cabinet_lookup.get(speaker_name)
                            )
                            cached_data = self._speaker_geometry_cache.get(cache_key)
                            if cached_data and isinstance(cached_data, dict) and cached_data.get('geoms'):
                                # Geometrie-Cache existiert - kann aus Cache geladen werden
                                # needs_rebuild bleibt True, damit Speaker aus Cache geladen wird
                                speakers_with_cache += 1
                            else:
                                # Parameter-Signatur stimmt, aber Geometrie-Cache fehlt - muss neu berechnen
                                all_speakers_unchanged = False
                                speakers_without_cache += 1
                    elif existing_actor:
                        # Speaker existiert im Cache - prüfe ob Parameter unverändert sind
                        signature = existing_actor.get('signature')
                        if cached_param_signature is not None:
                            # Signatur existiert - aber stimmt nicht überein (wurde oben bereits geprüft)
                            # Parameter-Signatur stimmt nicht überein - Speaker wurde geändert
                            all_speakers_unchanged = False
                            speakers_without_cache += 1
                        else:
                            # Signatur fehlt im Cache - prüfe ob Mesh-Signatur existiert
                            # Wenn ja, könnte der Speaker unverändert sein (Signatur wurde noch nicht gespeichert)
                            # Beim ersten Lauf nach dem Laden wird cached_param_signature noch nicht gesetzt sein,
                            # aber wenn existing_actor existiert, wurde der Speaker bereits gezeichnet
                            # Wir nehmen an, dass der Speaker unverändert ist, wenn existing_actor existiert
                            # und beim nächsten Lauf wird cached_param_signature gesetzt sein
                            if signature:
                                # Mesh-Signatur existiert - Speaker ist wahrscheinlich unverändert
                                # Speichere die aktuelle Signatur für den nächsten Lauf
                                needs_rebuild = False
                                # WICHTIG: Speichere die Signatur sofort, damit sie beim nächsten Lauf verfügbar ist
                                self._speaker_geometry_param_cache[geometry_param_key] = geometry_param_signature
                                # Aktualisiere cached_param_signature für den Task, damit der Vergleich später funktioniert
                                cached_param_signature = geometry_param_signature
                                speakers_with_cache += 1  # Signatur wird jetzt gespeichert
                            else:
                                # Weder Parameter- noch Mesh-Signatur - Speaker wurde geändert
                                all_speakers_unchanged = False
                                speakers_without_cache += 1
                    else:
                        # Speaker fehlt im Cache - muss neu erstellt werden
                        all_speakers_unchanged = False
                        speakers_missing += 1

                    # Ermittle stack_group_key für Stack-Speaker
                    stack_group_key = None
                    if config_lower == 'stack':
                        stack_group_key = self._get_stack_group_key_for_speaker(speaker_array, idx, cabinet_lookup)

                    # Sammle Task für dieses Array
                    array_speaker_tasks.append({
                        'array_id': array_id,  # 🎯 FIX: array_id ist jetzt speaker_array.id (die "alte" Erkennung)
                        'speaker_array': speaker_array,  # speaker_array Objekt für direkten Zugriff
                        'idx': idx,
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'speaker_name': speaker_name,
                        'configuration': configuration,
                        'is_highlighted': is_highlighted,
                        'geometry_param_key': geometry_param_key,
                        'geometry_param_signature': geometry_param_signature,
                        'cached_param_signature': cached_param_signature,  # Wurde oben aktualisiert wenn nötig
                        'needs_rebuild': needs_rebuild,
                        'existing_actor': existing_actor,
                        'stack_group_key': stack_group_key,  # Für Stack-Cache
                        'force_full_rebuild': False,  # kann z.B. für Flown-Arrays überschrieben werden
                    })

                # Wenn alle Speaker unverändert sind, überspringe Array komplett
                if all_speakers_unchanged:
                    # Füge existierende Speaker zum neuen Cache hinzu (alle Geometrie-Keys)
                    for task in array_speaker_tasks:
                        idx = task['idx']
                        # Kopiere alle Cache-Keys für diesen Speaker
                        for cache_key, cache_value in old_cache.items():
                            if (isinstance(cache_key, tuple) and len(cache_key) >= 2 and
                                str(cache_key[0]) == array_id and cache_key[1] == idx):
                                new_cache[cache_key] = cache_value
                                if cache_value.get('actor') and cache_value['actor'] not in new_actor_names:
                                    new_actor_names.append(cache_value['actor'])
                        # Stelle sicher, dass Parameter-Cache aktualisiert ist
                        if task['geometry_param_signature'] is not None:
                            self._speaker_geometry_param_cache[task['geometry_param_key']] = task['geometry_param_signature']
                    array_can_be_skipped = True
                else:
                    # Array hat geänderte Speaker - füge alle Tasks hinzu
                    # 📌 Spezialfall Flown-Array:
                    # Wenn sich der Zwischenwinkel eines Lautsprechers ändert,
                    # verschieben sich alle darunterliegenden Lautsprecher → gesamte Geometrie des Arrays
                    # muss neu berechnet werden. Daher: sobald EIN Speaker needs_rebuild=True ist,
                    # werden ALLE Speaker des Flown-Arrays neu aufgebaut und der Geometrie-Cache
                    # wird für dieses Array nicht verwendet.
                    if config_lower == 'flown':
                        any_changed = any(t['needs_rebuild'] for t in array_speaker_tasks)
                        if any_changed:
                            for t in array_speaker_tasks:
                                t['needs_rebuild'] = True
                                t['force_full_rebuild'] = True

                    speaker_tasks.extend(array_speaker_tasks)
                    total_speakers += len(array_speaker_tasks)
            else:
                # Beim ersten Laden (old_cache leer) - verarbeite alle Speaker
                # 🎯 FIX: Überspringe, wenn Array bereits aus Cache geladen wurde
                if array_can_be_skipped:
                    continue
                for idx in valid_indices:
                    total_speakers += 1
                    x = xs[idx]
                    y = ys[idx]
                    z = zs[idx]

                    # Prüfe ob dieser Speaker hervorgehoben werden soll
                    is_highlighted = False
                    if highlight_indices:
                        highlight_indices_str = [(str(aid), int(idx_val)) for aid, idx_val in highlight_indices]
                        is_highlighted = (array_id, int(idx)) in highlight_indices_str
                    elif highlight_array_ids_str:
                        # Prüfe ob Array-ID in der Liste ist
                        is_highlighted = array_id in highlight_array_ids_str

                    # Speaker muss neu erstellt werden (erster Lauf)
                    speaker_name = self._get_speaker_name(speaker_array, idx)
                    geometry_param_key = (array_id, int(idx))
                    geometry_param_signature = self._create_geometry_param_signature(
                        speaker_array, idx, float(x), float(y), float(z), cabinet_lookup
                    )
                    cached_param_signature = self._speaker_geometry_param_cache.get(geometry_param_key)

                    # Prüfe ob Actor bereits existiert (sollte beim ersten Laden nicht der Fall sein)
                    key = (array_id, int(idx), 0)
                    existing_actor = old_cache.get(key)
                    needs_rebuild = True  # Beim ersten Laden immer neu bauen

                    # Ermittle stack_group_key für Stack-Speaker
                    stack_group_key = None
                    if config_lower == 'stack':
                        stack_group_key = self._get_stack_group_key_for_speaker(speaker_array, idx, cabinet_lookup)

                    # Sammle Task für Parallelisierung
                    speaker_tasks.append({
                        'array_id': array_id,
                        'array_name': array_name,
                        'speaker_array': speaker_array,
                        'idx': idx,
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'speaker_name': speaker_name,
                        'configuration': configuration,
                        'is_highlighted': is_highlighted,
                        'geometry_param_key': geometry_param_key,
                        'geometry_param_signature': geometry_param_signature,
                        'cached_param_signature': cached_param_signature,
                        'needs_rebuild': needs_rebuild,
                        'existing_actor': existing_actor,
                        'stack_group_key': stack_group_key,  # Für Stack-Cache
                    })

            # Überspringe Array, wenn alle Speaker unverändert sind
            if array_can_be_skipped:
                continue

        # 🚀 OPTIMIERUNG 3: Parallelisiere build_geometries für alle Speaker, die neu berechnet werden müssen
        geometries_results: Dict[tuple, List[Tuple[Any, Optional[int]]]] = {}

        # Teile Tasks in: mit Cache, ohne Cache (parallel)
        tasks_with_cache: List[Dict] = []
        tasks_without_cache: List[Dict] = []

        skipped_count = sum(1 for t in speaker_tasks if not t['needs_rebuild'])

        for task in speaker_tasks:
            if not task['needs_rebuild']:
                # 🚀 OPTIMIERUNG 2: Speaker muss nicht neu gezeichnet werden - überspringe
                continue

            # Wenn für diesen Task explizit ein kompletter Neuaufbau erzwungen wird
            # (z.B. Flown-Array mit geänderten Zwischenwinkeln), niemals den Geometrie-Cache verwenden.
            if task.get('force_full_rebuild'):
                tasks_without_cache.append(task)
                continue

            # Prüfe ob Cache vorhanden ist (nicht nur Signatur-Vergleich, sondern auch ob Cache-Daten existieren)
            cache_key = self._create_geometry_cache_key(
                task['array_id'], task['speaker_array'], task['idx'], task['speaker_name'],
                task['configuration'], cabinet_lookup.get(task['speaker_name'])
            )
            cached_data = self._speaker_geometry_cache.get(cache_key)

            # Prüfe ob Parameter-Signatur übereinstimmt (beide müssen nicht None sein)
            signatures_match = (task['cached_param_signature'] is not None and
                                task['cached_param_signature'] == task['geometry_param_signature'])

            # #region agent log
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                array_id_val = task.get('array_id')
                idx_val = task.get('idx')
                # Konvertiere NumPy-Typen zu native Python-Typen
                if isinstance(array_id_val, (np.integer, np.int64, np.int32)):
                    array_id_val = int(array_id_val)
                elif array_id_val is not None:
                    array_id_val = str(array_id_val)
                if isinstance(idx_val, (np.integer, np.int64, np.int32)):
                    idx_val = int(idx_val)
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H1",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:861",
                    "message": "Cache decision for task",
                    "data": {
                        "array_id": array_id_val,
                        "speaker_idx": idx_val,
                        "signatures_match": bool(signatures_match),
                        "cached_data_exists": cached_data is not None,
                        "cached_geoms_exists": cached_data.get('geoms') is not None if cached_data else False,
                        "cached_position_exists": cached_data.get('position') is not None if cached_data else False,
                        "will_use_cache": bool(signatures_match and cached_data and isinstance(cached_data, dict) and cached_data.get('geoms') and cached_data.get('position'))
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
            # #endregion
            
            if (signatures_match and
                cached_data and isinstance(cached_data, dict) and
                cached_data.get('geoms') and cached_data.get('position')):
                # 🚀 OPTIMIERUNG 1: Kann Cache verwenden (Position-Transformation)
                tasks_with_cache.append(task)
            else:
                # 🚀 OPTIMIERUNG 3: Muss neu berechnen (parallel)
                tasks_without_cache.append(task)

        # 🚀 OPTIMIERUNG 3: Parallelisiere build_geometries für Tasks ohne Cache
        if tasks_without_cache:
            # WICHTIG: build_geometry_task muss außerhalb der Funktion definiert werden oder
            # wir müssen sicherstellen, dass self, cabinet_lookup und container verfügbar sind
            # Für ThreadPoolExecutor: Die Funktion muss picklable sein, daher verwenden wir
            # eine Methode statt einer verschachtelten Funktion
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with perf_section("PlotSPL3DOverlays.draw_speakers.parallel_build", task_count=len(tasks_without_cache)):
                # Verwende ThreadPoolExecutor für Parallelisierung
                max_workers = min(4, len(tasks_without_cache))  # Max 4 Threads
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Erstelle Futures für alle Tasks
                    future_to_task = {}
                    for task in tasks_without_cache:
                        # Verwende eine Lambda-Funktion, die die benötigten Parameter erfasst
                        # #region agent log
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "PlotSPL3DSpeaker.py:draw_speakers:778",
                                "message": "Building geometries - task parameters",
                                "data": {
                                    "array_id": str(task['array_id']),
                                    "idx": int(task['idx']),
                                    "x": float(task['x']),
                                    "y": float(task['y']),
                                    "z": float(task['z']),
                                    "configuration": str(task.get('configuration', ''))
                                },
                                "timestamp": int(__import__('time').time() * 1000)
                            }) + "\n")
                        # #endregion
                        future = executor.submit(
                            self._build_speaker_geometries,
                            task['speaker_array'],
                            task['idx'],
                            task['x'],
                            task['y'],
                            task['z'],
                            cabinet_lookup,
                            container,
                        )
                        future_to_task[future] = task

                    # Sammle Ergebnisse
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            geometries = future.result()
                            if geometries:
                                geometries_results[task['geometry_param_key']] = geometries
                                # 🚀 OPTIMIERUNG 1: Cache speichern
                                cache_key = self._create_geometry_cache_key(
                                    task['array_id'], task['speaker_array'], task['idx'], task['speaker_name'],
                                    task['configuration'], cabinet_lookup.get(task['speaker_name'])
                                )
                                self._speaker_geometry_cache[cache_key] = {
                                    'geoms': [(mesh.copy(deep=True), exit_idx) for mesh, exit_idx in geometries],
                                    'position': (task['x'], task['y'], task['z'])
                                }
                                # 🎯 WICHTIG: Track Cache-Keys pro Array für effizientes Löschen
                                array_id_str = str(task['array_id'])
                                if array_id_str not in self._array_id_to_cache_keys:
                                    self._array_id_to_cache_keys[array_id_str] = set()
                                self._array_id_to_cache_keys[array_id_str].add(cache_key)
                                # #region agent log
                                try:
                                    import json
                                    import time as time_module
                                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "CACHE2",
                                            "location": "PlotSPL3DSpeaker.py:draw_speakers:restore_cache_mapping",
                                            "message": "Restoring cache mapping for array",
                                            "data": {
                                                "array_id": array_id_str,
                                                "cache_key": str(cache_key)[:100],
                                                "mapping_size": len(self._array_id_to_cache_keys.get(array_id_str, set()))
                                            },
                                            "timestamp": int(time_module.time() * 1000)
                                        }) + "\n")
                                except Exception:
                                    pass
                                # #endregion
                                # Begrenze Cache-Größe
                                if len(self._speaker_geometry_cache) > self._geometry_cache_max_size:
                                    first_key = next(iter(self._speaker_geometry_cache))
                                    del self._speaker_geometry_cache[first_key]
                                    # 🎯 WICHTIG: Entferne auch aus Mapping
                                    for array_id_in_map, cache_keys in self._array_id_to_cache_keys.items():
                                        if first_key in cache_keys:
                                            cache_keys.discard(first_key)
                                            if not cache_keys:  # Wenn Set leer ist, entferne Eintrag
                                                del self._array_id_to_cache_keys[array_id_in_map]
                                            break
                                self._speaker_geometry_param_cache[task['geometry_param_key']] = task['geometry_param_signature']
                        except Exception as e:  # noqa: BLE001
                            import traceback

                            print(f"[ERROR] Parallel build_geometries failed for {task['array_id']}:{task['idx']}: {e}")
                            traceback.print_exc()

        # 🚀 OPTIMIERUNG 1: Verarbeite Tasks mit Cache (Position-Transformation)
        for task in tasks_with_cache:
            cache_key = self._create_geometry_cache_key(
                task['array_id'], task['speaker_array'], task['idx'], task['speaker_name'],
                task['configuration'], cabinet_lookup.get(task['speaker_name'])
            )
            cached_data = self._speaker_geometry_cache.get(cache_key)
            # #region agent log
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                array_id_val = task.get('array_id')
                idx_val = task.get('idx')
                # Konvertiere NumPy-Typen zu native Python-Typen
                if isinstance(array_id_val, (np.integer, np.int64, np.int32)):
                    array_id_val = int(array_id_val)
                elif array_id_val is not None:
                    array_id_val = str(array_id_val)
                if isinstance(idx_val, (np.integer, np.int64, np.int32)):
                    idx_val = int(idx_val)
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H1",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:951",
                    "message": "Checking cache for task",
                    "data": {
                        "array_id": array_id_val,
                        "speaker_idx": idx_val,
                        "cache_key_exists": cache_key in self._speaker_geometry_cache,
                        "cached_data_exists": cached_data is not None,
                        "new_position": [float(task.get('x', 0)), float(task.get('y', 0)), float(task.get('z', 0))]
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
            # #endregion
            if cached_data and isinstance(cached_data, dict):
                cached_geoms = cached_data.get('geoms')
                cached_position = cached_data.get('position')
                if cached_geoms and cached_position:
                    # Transformiere gecachte Geometrien zur neuen Position
                    old_x, old_y, old_z = cached_position
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        array_id_val = task.get('array_id')
                        idx_val = task.get('idx')
                        # Konvertiere NumPy-Typen zu native Python-Typen
                        if isinstance(array_id_val, (np.integer, np.int64, np.int32)):
                            array_id_val = int(array_id_val)
                        elif array_id_val is not None:
                            array_id_val = str(array_id_val)
                        if isinstance(idx_val, (np.integer, np.int64, np.int32)):
                            idx_val = int(idx_val)
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:958",
                            "message": "Using cached geometry with transformation",
                            "data": {
                                "array_id": array_id_val,
                                "speaker_idx": idx_val,
                                "old_position": [float(old_x), float(old_y), float(old_z)],
                                "new_position": [float(task.get('x', 0)), float(task.get('y', 0)), float(task.get('z', 0))]
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                    # #endregion
                    geometries = self._transform_cached_geometry_to_position(
                        cached_geoms, old_x, old_y, old_z, task['x'], task['y'], task['z']
                    )
                    geometries_results[task['geometry_param_key']] = geometries

        # 🚀 OPTIMIERUNG: Speichere Flown-Arrays im Array-Cache nach Berechnung
        # Sammle alle Flown-Array-Geometrien
        flown_array_geometries: Dict[str, Dict[int, List[Tuple[Any, Optional[int]]]]] = {}
        flown_array_positions: Dict[str, tuple] = {}
        # 🎯 FIX: array_id_to_name Mapping nicht mehr benötigt - verwende speaker_array direkt aus tasks

        for task in speaker_tasks:
            array_id = task['array_id']
            configuration = task.get('configuration', '').lower()

            if configuration == 'flown':
                # Hole Array-Positionen
                speaker_array = task['speaker_array']
                array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
                array_pos_y = getattr(speaker_array, 'array_position_y', 0.0)
                array_pos_z = getattr(speaker_array, 'array_position_z', 0.0)

                # Sammle Geometrien für dieses Array
                if array_id not in flown_array_geometries:
                    flown_array_geometries[array_id] = {}
                    flown_array_positions[array_id] = (array_pos_x, array_pos_y, array_pos_z)

                # Füge Geometrien hinzu (wenn bereits berechnet)
                # Prüfe sowohl geometries_results als auch cached_geometries
                geometries = geometries_results.get(task['geometry_param_key'])
                if not geometries and 'cached_geometries' in task:
                    geometries = task['cached_geometries']

                if geometries:
                    idx = task['idx']
                    flown_array_geometries[array_id][idx] = geometries

        for array_id, geometries_dict in flown_array_geometries.items():
            if geometries_dict:
                array_pos = flown_array_positions.get(array_id, (0.0, 0.0, 0.0))
                # 🎯 FIX: Speichere im Array-Cache nur, wenn sich nur der hide-Status geändert hat
                # Wenn sich Geometrieparameter geändert haben, soll der Cache nicht gespeichert werden
                # Prüfe, ob sich nur der hide-Status geändert hat (durch Signatur-Vergleichung)
                should_cache = False
                # Hole speaker_array für Signatur-Vergleichung
                speaker_array_for_sig = None
                for task in speaker_tasks:
                    if task.get('array_id') == array_id and 'speaker_array' in task:
                        speaker_array_for_sig = task['speaker_array']
                        break
                if speaker_array_for_sig is None:
                    # Fallback: Suche in speaker_arrays
                    for arr_name, arr in speaker_arrays.items():
                        if getattr(arr, 'id', None) == array_id or str(getattr(arr, 'id', None)) == str(array_id):
                            speaker_array_for_sig = arr
                            break
                
                if speaker_array_for_sig and array_id in self._array_signature_cache:
                    current_signature = self._create_array_signature(
                        speaker_array_for_sig, array_id, array_pos[0], array_pos[1], array_pos[2], cabinet_lookup
                    )
                    cached_signature = self._array_signature_cache.get(array_id)
                    # Wenn Signaturen übereinstimmen, bedeutet das, dass sich nur der hide-Status geändert hat
                    if cached_signature == current_signature:
                        should_cache = True
                
                # Speichere im Array-Cache nur, wenn sich nur der hide-Status geändert hat
                if should_cache:
                    self._array_geometry_cache[array_id] = {
                        'geometries': {idx: [(mesh.copy(deep=True), exit_idx) for mesh, exit_idx in geoms]
                                      for idx, geoms in geometries_dict.items()},
                        'array_pos': array_pos
                    }
                else:
                    # 🎯 FIX: Lösche Cache, wenn sich Geometrieparameter geändert haben
                    if array_id in self._array_geometry_cache:
                        del self._array_geometry_cache[array_id]
                    if array_id in self._array_signature_cache:
                        del self._array_signature_cache[array_id]
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:723",
                        "message": "Saving flown array to cache",
                        "data": {
                            "array_id": str(array_id),
                            "array_pos": [float(array_pos[0]), float(array_pos[1]), float(array_pos[2])],
                            "num_speakers": int(len(geometries_dict))
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion
                # Speichere Array-Signatur
                # 🎯 FIX: Verwende speaker_array.id (die "alte" Erkennung) - hole speaker_array direkt aus tasks
                # Suche speaker_array aus tasks, da array_id jetzt speaker_array.id ist
                speaker_array = None
                for task in speaker_tasks:
                    if task.get('array_id') == array_id and 'speaker_array' in task:
                        speaker_array = task['speaker_array']
                        break
                
                # Fallback: Wenn nicht in tasks gefunden, suche in speaker_arrays mit array_id
                if speaker_array is None:
                    # Versuche direkten Zugriff mit array_id (da array_id jetzt speaker_array.id ist)
                    # Aber speaker_arrays Dictionary verwendet möglicherweise noch array_name als Key
                    # Daher durchsuchen wir alle Arrays
                    for arr_name, arr in speaker_arrays.items():
                        if getattr(arr, 'id', None) == array_id or str(getattr(arr, 'id', None)) == str(array_id):
                            speaker_array = arr
                            break
                
                if speaker_array:
                    array_signature = self._create_array_signature(
                        speaker_array, array_id, array_pos[0], array_pos[1], array_pos[2], cabinet_lookup
                    )
                    # 🎯 FIX: Speichere Array-Signatur nur, wenn Cache verwendet werden soll
                    if should_cache:
                        self._array_signature_cache[array_id] = array_signature

        # 🚀 OPTIMIERUNG: Speichere Stack-Gruppen im Stack-Cache nach Berechnung
        # Sammle alle Stack-Gruppen-Geometrien
        stack_group_geometries: Dict[tuple[str, tuple], Dict[int, List[Tuple[Any, Optional[int]]]]] = {}
        stack_group_positions: Dict[tuple[str, tuple], tuple] = {}
        # Mapping von (array_id, stack_group_key, idx) zu ursprünglicher Speaker-Position
        stack_speaker_positions: Dict[tuple[str, tuple, int], tuple] = {}
        # array_id_to_name wurde bereits oben initialisiert

        for task in speaker_tasks:
            array_id = task['array_id']
            configuration = task.get('configuration', '').lower()

            if configuration == 'stack' and 'stack_group_key' in task:
                stack_group_key = task['stack_group_key']
                if stack_group_key:
                    cache_key = (array_id, stack_group_key)

                    # Hole Array-Positionen
                    speaker_array = task['speaker_array']
                    array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
                    array_pos_y = getattr(speaker_array, 'array_position_y', 0.0)
                    array_pos_z = getattr(speaker_array, 'array_position_z', 0.0)

                    # Sammle Geometrien für diese Stack-Gruppe
                    if cache_key not in stack_group_geometries:
                        stack_group_geometries[cache_key] = {}
                        stack_group_positions[cache_key] = (array_pos_x, array_pos_y, array_pos_z)

                    # Füge Geometrien hinzu (wenn bereits berechnet)
                    geometries = geometries_results.get(task['geometry_param_key'])
                    if geometries:
                        idx = task['idx']
                        stack_group_geometries[cache_key][idx] = geometries
                        # Speichere tatsächliche Mesh-Position (mit Offset) für diesen Speaker
                        # Extrahiere den Mesh-Center aus der ersten Geometrie
                        if geometries and len(geometries) > 0:
                            first_mesh, _ = geometries[0]
                            if hasattr(first_mesh, 'bounds') and first_mesh.bounds is not None:
                                bounds = first_mesh.bounds
                                mesh_center_x = (bounds[0] + bounds[1]) / 2.0
                                mesh_center_y = (bounds[2] + bounds[3]) / 2.0
                                mesh_center_z = (bounds[4] + bounds[5]) / 2.0
                                stack_speaker_positions[(array_id, stack_group_key, idx)] = (mesh_center_x, mesh_center_y, mesh_center_z)
                            else:
                                # Fallback: Verwende Speaker-Position ohne Offset
                                stack_speaker_positions[(array_id, stack_group_key, idx)] = (task['x'], task['y'], task['z'])
                        else:
                            # Fallback: Verwende Speaker-Position ohne Offset
                            stack_speaker_positions[(array_id, stack_group_key, idx)] = (task['x'], task['y'], task['z'])

        # Speichere alle Stack-Gruppen im Stack-Cache
        for cache_key, geometries_dict in stack_group_geometries.items():
            if geometries_dict:
                array_id, stack_group_key = cache_key
                stack_pos = stack_group_positions.get(cache_key, (0.0, 0.0, 0.0))

                # Speichere im Stack-Cache (tiefe Kopie für Cache)
                # Speichere auch die ursprünglichen Speaker-Positionen für jeden Speaker
                speaker_positions_dict = {}
                for idx in geometries_dict.keys():
                    speaker_pos_key = (array_id, stack_group_key, idx)
                    if speaker_pos_key in stack_speaker_positions:
                        speaker_positions_dict[idx] = stack_speaker_positions[speaker_pos_key]
                
                self._stack_geometry_cache[cache_key] = {
                    'geometries': {idx: [(mesh.copy(deep=True), exit_idx) for mesh, exit_idx in geoms]
                                  for idx, geoms in geometries_dict.items()},
                    'stack_pos': stack_pos,  # Array-Position (für Signatur-Vergleich)
                    'speaker_positions': speaker_positions_dict  # Ursprüngliche Speaker-Positionen für Transformation
                }
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:1025",
                        "message": "Saving stack group to cache with speaker positions",
                        "data": {
                            "array_id": str(array_id),
                            "stack_group_key": str(stack_group_key),
                            "stack_pos": [float(stack_pos[0]), float(stack_pos[1]), float(stack_pos[2])],
                            "num_speakers": int(len(geometries_dict)),
                            "speaker_positions": {str(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in speaker_positions_dict.items()}
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H2",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:774",
                        "message": "Saving stack group to cache",
                        "data": {
                            "array_id": str(array_id),
                            "stack_group_key": str(stack_group_key),
                            "stack_pos": [float(stack_pos[0]), float(stack_pos[1]), float(stack_pos[2])],
                            "num_speakers": int(len(geometries_dict))
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion

                # Speichere Stack-Signatur
                # Verwende array_name statt array_id, da Keys in speaker_arrays Integer sein können
                # 🎯 FIX: Verwende speaker_array.id (die "alte" Erkennung) - hole speaker_array direkt aus tasks
                # Suche speaker_array aus tasks, da array_id jetzt speaker_array.id ist
                speaker_array = None
                for task in speaker_tasks:
                    if task.get('array_id') == array_id and 'speaker_array' in task:
                        speaker_array = task['speaker_array']
                        break
                
                # Fallback: Wenn nicht in tasks gefunden, suche in speaker_arrays
                if speaker_array is None:
                    for arr_name, arr in speaker_arrays.items():
                        if getattr(arr, 'id', None) == array_id or str(getattr(arr, 'id', None)) == str(array_id):
                            speaker_array = arr
                            break
                
                if speaker_array:
                    # Hole Speaker-Indizes für diese Stack-Gruppe aus geometries_dict
                    speaker_indices_for_signature = list(geometries_dict.keys())
                    stack_signature = self._create_stack_signature(
                        speaker_array, array_id, stack_group_key, stack_pos[0], stack_pos[1], stack_pos[2], cabinet_lookup, speaker_indices_for_signature
                    )
                    self._stack_signature_cache[cache_key] = stack_signature
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:1025",
                            "message": "Saving stack signature to cache",
                            "data": {
                                "array_id": str(array_id),
                                "stack_group_key": str(stack_group_key),
                                "cache_key": str(cache_key),
                                "signature": str(stack_signature)[:100]
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                    # #endregion
                else:
                    # #region agent log
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H2",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:1020",
                            "message": "NOT saving stack signature - speaker_array not found",
                            "data": {
                                "array_id": str(array_id),
                                "stack_group_key": str(stack_group_key),
                                "cache_key": str(cache_key),
                                "speaker_arrays_keys": list(speaker_arrays.keys()) if speaker_arrays else []
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                    # #endregion

        # Verarbeite alle Speaker mit ihren Geometrien
        # #region agent log - PLOT SPEAKER DEBUG
        try:
            import json
            import time as time_module
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "PLOT_SPEAKER_DEBUG",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:before_speaker_loop",
                    "message": "About to plot speakers - summary",
                    "data": {
                        "total_speaker_tasks": int(len(speaker_tasks)),
                        "array_ids_in_tasks": list(set([str(task.get('array_id', 'unknown')) for task in speaker_tasks]))
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        
        for task in speaker_tasks:
            array_id = task['array_id']
            idx = task['idx']
            x = task['x']
            y = task['y']
            z = task['z']
            is_highlighted = task['is_highlighted']
            
            # #region agent log - PLOT SPEAKER DEBUG
            try:
                import json
                import time as time_module
                speaker_array_obj = task.get('speaker_array')
                speaker_array_id_from_obj = getattr(speaker_array_obj, 'id', None) if speaker_array_obj else None
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "PLOT_SPEAKER_DEBUG",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:plotting_speaker",
                        "message": "Plotting speaker - array_id check",
                        "data": {
                            "task_array_id": str(array_id),
                            "task_array_id_type": type(array_id).__name__,
                            "speaker_array.id": str(speaker_array_id_from_obj) if speaker_array_id_from_obj is not None else None,
                            "speaker_idx": int(idx),
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "array_id == speaker_array.id": bool(str(array_id) == str(speaker_array_id_from_obj)) if speaker_array_id_from_obj is not None else None,
                            "needs_rebuild": bool(task.get('needs_rebuild', True)),
                            "has_cached_geometries": bool('cached_geometries' in task)
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except Exception:
                pass
            # #endregion

            # Hole Geometrien (entweder aus Array-Cache, Cache oder aus parallel berechneten Ergebnissen)
            if 'cached_geometries' in task:
                # Geometrien bereits aus Array-Cache geladen
                geometries = task['cached_geometries']
                # #region agent log
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1",
                        "location": "PlotSPL3DSpeaker.py:draw_speakers:using_cached_geometries",
                        "message": "Using cached geometries from array cache",
                        "data": {
                            "array_id": str(array_id),
                            "idx": int(idx),
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "has_geometries": bool(geometries)
                        },
                        "timestamp": int(__import__('time').time() * 1000)
                    }) + "\n")
                # #endregion
            else:
                # Hole aus geometries_results (parallel berechnet oder aus Cache transformiert)
                geometries = geometries_results.get(task['geometry_param_key'])
                # #region agent log
                if idx == 0:  # Nur für idx 0 loggen
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:using_geometries_results",
                            "message": "Using geometries from results (idx 0)",
                            "data": {
                                "array_id": str(array_id),
                                "idx": int(idx),
                                "x": float(x),
                                "y": float(y),
                                "z": float(z),
                                "has_geometries": bool(geometries),
                                "has_cached_geometries": bool('cached_geometries' in task)
                            },
                            "timestamp": int(__import__('time').time() * 1000)
                        }) + "\n")
                # #endregion

            if not task['needs_rebuild'] and task['existing_actor']:
                # 🚀 OPTIMIERUNG 2: Speaker existiert bereits - nur Highlight aktualisieren
                key = (array_id, int(idx), 0)
                existing = task['existing_actor']
                new_cache[key] = existing
                if existing.get('actor') and existing['actor'] not in new_actor_names:
                    new_actor_names.append(existing['actor'])
                continue

            if geometries is None:
                # 🎯 FIX: Wenn Geometrien fehlen, versuche sie neu zu berechnen
                # Dies kann passieren, wenn der Cache geleert wurde, aber die Geometrien noch nicht neu berechnet wurden
                speaker_array = task.get('speaker_array')
                if speaker_array:
                    # Prüfe ob Array versteckt ist - in diesem Fall überspringe
                    if getattr(speaker_array, 'hide', False):
                        continue
                    # Versuche Geometrien neu zu berechnen
                    try:
                        geometries = self._build_speaker_geometries(
                            speaker_array,
                            idx,
                            x,
                            y,
                            z,
                            cabinet_lookup,
                            container,
                        )
                        if geometries:
                            # Speichere in geometries_results für zukünftige Verwendung
                            geometries_results[task['geometry_param_key']] = geometries
                        else:
                            # Geometrien konnten nicht berechnet werden - überspringe
                            continue
                    except Exception as e:  # noqa: BLE001
                        # Bei Fehler beim Berechnen - überspringe diesen Speaker
                        import traceback
                        print(f"[WARNING] Konnte Geometrien nicht berechnen für array_id='{array_id}', speaker_idx={idx}: {e}")
                        traceback.print_exc()
                        continue
                else:
                    # Speaker-Array nicht verfügbar - überspringe
                    continue

            # MESH MERGING: Kombiniere alle Geometrien eines Speakers zu einem Mesh
            # WICHTIG: Dies muss IMMER ausgeführt werden, auch wenn Geometrien aus Parallelisierung kommen!
            if geometries and len(geometries) > 1:
                with perf_section("PlotSPL3DOverlays.draw_speakers.merge_meshes", geom_count=len(geometries)):
                    # 🚀 OPTIMIERUNG: Batch-Merge statt sequenziell
                    # Sammle alle Meshes und Exit-Face-Indices
                    meshes_to_merge = []
                    exit_face_indices = []
                    cell_offsets = [0]  # Start-Offset

                    for geom_idx, (body_mesh, exit_face_idx) in enumerate(geometries):
                        meshes_to_merge.append(body_mesh)
                        if exit_face_idx is not None:
                            exit_face_indices.append((cell_offsets[-1], exit_face_idx))
                        # Berechne nächsten Offset
                        cell_offsets.append(cell_offsets[-1] + body_mesh.n_cells)

                    # 🚀 OPTIMIERUNG: Batch-Merge aller Meshes auf einmal
                    if len(meshes_to_merge) > 1:
                        # Verwende PyVista's MultiBlock für effizientes Merging
                        from pyvista import MultiBlock
                        multi_block = MultiBlock(meshes_to_merge)
                        merged_mesh = multi_block.combine()
                    else:
                        # 🚀 OPTIMIERUNG: Verwende shallow copy wenn nur ein Mesh vorhanden
                        # Da merged_mesh später modifiziert wird, brauchen wir eine Kopie
                        # Aber shallow copy ist schneller als deep copy
                        if meshes_to_merge:
                            try:
                                merged_mesh = meshes_to_merge[0].copy(deep=False)
                            except Exception:
                                # Fallback zu deep copy wenn shallow nicht funktioniert
                                merged_mesh = meshes_to_merge[0].copy(deep=True)
                        else:
                            merged_mesh = None

                    if merged_mesh is not None:
                        # Erstelle Scalar-Array für ALLE Exit-Faces mit korrigierten Offsets
                        if exit_face_indices:
                            scalars = np.zeros(merged_mesh.n_cells, dtype=int)
                            for cell_offset, exit_idx in exit_face_indices:
                                final_idx = cell_offset + exit_idx
                                if final_idx < merged_mesh.n_cells:
                                    scalars[final_idx] = 1
                            merged_mesh.cell_data['speaker_face'] = scalars
                            merged_exit_face = -1
                        else:
                            merged_exit_face = None

                        # Ersetze geometries mit dem gemerged mesh
                        geometries = [(merged_mesh, merged_exit_face)]

            with perf_section("PlotSPL3DOverlays.draw_speakers.cache_lookup", array_id=array_id, speaker_idx=idx):
                if not geometries:
                    sphere = self.pv.Sphere(radius=0.6, center=(float(x), float(y), float(z)),
                                            theta_resolution=32, phi_resolution=32)
                    key = (array_id, int(idx), 'fallback')
                    existing = old_cache.get(key)
                    if existing:
                        signature = existing.get('signature')
                        current_signature = self._speaker_signature_from_mesh(sphere, None)
                        if signature == current_signature:
                            # Aktualisiere actor_obj falls nötig
                            if 'actor_obj' not in existing:
                                actor_name = existing.get('actor')
                                existing['actor_obj'] = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                            new_cache[key] = existing
                            if existing['actor'] not in new_actor_names:
                                new_actor_names.append(existing['actor'])
                            continue
                        existing_mesh = existing['mesh']
                        existing_mesh.deep_copy(sphere)
                        self._update_speaker_actor(existing['actor'], existing_mesh, None, body_color, exit_color)
                        existing['signature'] = current_signature
                        # Aktualisiere actor_obj falls nötig
                        if 'actor_obj' not in existing:
                            actor_name = existing.get('actor')
                            existing['actor_obj'] = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                        new_cache[key] = existing
                        if existing['actor'] not in new_actor_names:
                            new_actor_names.append(existing['actor'])
                    else:
                        mesh_to_add = sphere.copy(deep=True)
                        # Setze Edge-Color basierend auf Highlight-Status
                        edge_color = 'red' if is_highlighted else 'black'
                        # Basis-Linienbreiten: rot identisch mit Surface-Umrandungen (1.0)
                        base_line_width = 1.0 if is_highlighted else 1.0
                        # 🎯 FIX: Verwende eindeutigen Actor-Namen basierend auf Array-ID, Speaker-Index und Geometrie-Index
                        unique_actor_name = self._create_speaker_actor_name(array_id, int(idx), 'fallback')
                        
                        # #region agent log
                        try:
                            import json
                            import time as time_module
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "PLOT_ARRAY_ID_DEBUG",
                                    "location": "PlotSPL3DSpeaker.py:draw_speakers:creating_fallback_actor",
                                    "message": "Creating fallback actor for speaker",
                                    "data": {
                                        "array_id": str(array_id),
                                        "array_id_type": type(array_id).__name__,
                                        "speaker_idx": int(idx),
                                        "unique_actor_name": unique_actor_name,
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z)
                                    },
                                    "timestamp": int(time_module.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        actor_name = self._add_overlay_mesh(
                            mesh_to_add,
                            color=body_color,
                            opacity=1.0,
                            edge_color=edge_color,
                            line_width=self._get_scaled_line_width(base_line_width),
                            category='speakers',
                            name=unique_actor_name,
                        )
                        # Hole das Actor-Objekt aus dem Renderer
                        actor_obj = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                        new_cache[key] = {
                            'mesh': mesh_to_add,
                            'actor': actor_name,
                            'actor_obj': actor_obj,  # Speichere Actor-Objekt für direkten Vergleich
                            'signature': self._speaker_signature_from_mesh(mesh_to_add, None),
                        }
                        if actor_name not in new_actor_names:
                            new_actor_names.append(actor_name)
                else:
                    for geom_idx, (body_mesh, exit_face_index) in enumerate(geometries):
                        key = (array_id, int(idx), geom_idx)
                        existing = old_cache.get(key)
                        if existing:
                            signature = existing.get('signature')
                            current_signature = self._speaker_signature_from_mesh(body_mesh, exit_face_index)
                            if signature == current_signature:
                                # 🎯 FIX: Stelle sicher, dass der Actor-Name konsistent ist
                                # Wenn der Actor-Name im Cache nicht dem erwarteten eindeutigen Namen entspricht,
                                # aktualisiere ihn, damit er konsistent ist
                                expected_actor_name = self._create_speaker_actor_name(array_id, int(idx), geom_idx)
                                cached_actor_name = existing.get('actor')
                                
                                # Prüfe, ob der Actor bereits im Plotter existiert
                                actor_exists_in_plotter = False
                                if cached_actor_name and hasattr(self, 'plotter') and self.plotter is not None:
                                    try:
                                        actor_exists_in_plotter = cached_actor_name in self.plotter.renderer.actors
                                    except Exception:
                                        pass
                                
                                if cached_actor_name != expected_actor_name or not actor_exists_in_plotter:
                                    # #region agent log
                                    try:
                                        import json
                                        import time as time_module
                                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({
                                                "sessionId": "debug-session",
                                                "runId": "run1",
                                                "hypothesisId": "ACTOR_NAME_FIX",
                                                "location": "PlotSPL3DSpeaker.py:draw_speakers:fix_actor_name",
                                                "message": "Fixing inconsistent actor name from cache",
                                                "data": {
                                                    "array_id": str(array_id),
                                                    "speaker_idx": int(idx),
                                                    "geom_idx": int(geom_idx),
                                                    "cached_actor_name": str(cached_actor_name),
                                                    "expected_actor_name": expected_actor_name,
                                                    "actor_exists_in_plotter": bool(actor_exists_in_plotter)
                                                },
                                                "timestamp": int(time_module.time() * 1000)
                                            }) + "\n")
                                    except Exception:
                                        pass
                                    # #endregion
                                    
                                    # Entferne den alten Actor, falls er existiert
                                    if cached_actor_name:
                                        try:
                                            self.plotter.remove_actor(cached_actor_name)
                                        except Exception:
                                            pass
                                    
                                    # Aktualisiere den Actor-Namen im Cache
                                    existing['actor'] = expected_actor_name
                                    
                                    # Erstelle den Actor neu mit dem korrekten Namen
                                    existing_mesh = existing.get('mesh')
                                    if existing_mesh is not None:
                                        # Erstelle ein neues Mesh, um den Actor hinzuzufügen
                                        mesh_to_add = existing_mesh.copy(deep=True)
                                        
                                        # Setze Edge-Color basierend auf Highlight-Status
                                        edge_color = 'red' if is_highlighted else 'black'
                                        base_line_width = 1.0
                                        line_width = self._get_scaled_line_width(base_line_width)
                                        
                                        # Füge den Actor mit dem neuen Namen hinzu
                                        if exit_face_index == -1 or 'speaker_face' in mesh_to_add.cell_data:
                                            actor_name = self._add_overlay_mesh(
                                                mesh_to_add,
                                                scalars='speaker_face',
                                                cmap=[body_color, exit_color],
                                                opacity=1.0,
                                                edge_color=edge_color,
                                                line_width=line_width,
                                                category='speakers',
                                                name=expected_actor_name,
                                            )
                                        elif exit_face_index is not None and mesh_to_add.n_cells > 0:
                                            scalars = np.zeros(mesh_to_add.n_cells, dtype=int)
                                            scalars[int(exit_face_index)] = 1
                                            mesh_to_add.cell_data['speaker_face'] = scalars
                                            actor_name = self._add_overlay_mesh(
                                                mesh_to_add,
                                                scalars='speaker_face',
                                                cmap=[body_color, exit_color],
                                                opacity=1.0,
                                                edge_color=edge_color,
                                                line_width=line_width,
                                                category='speakers',
                                                name=expected_actor_name,
                                            )
                                        else:
                                            actor_name = self._add_overlay_mesh(
                                                mesh_to_add,
                                                color=body_color,
                                                opacity=1.0,
                                                edge_color=edge_color,
                                                line_width=line_width,
                                                category='speakers',
                                                name=expected_actor_name,
                                            )
                                        existing['actor'] = actor_name
                                        existing['mesh'] = mesh_to_add
                                
                                # Aktualisiere actor_obj falls nötig
                                if 'actor_obj' not in existing:
                                    actor_name = existing.get('actor')
                                    existing['actor_obj'] = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                                new_cache[key] = existing
                                if existing['actor'] not in new_actor_names:
                                    new_actor_names.append(existing['actor'])
                                continue
                            # Signature hat sich geändert - aktualisiere existing
                            existing_mesh = existing['mesh']
                            existing_mesh.deep_copy(body_mesh)
                            self._update_speaker_actor(existing['actor'], existing_mesh, exit_face_index, body_color, exit_color)
                            existing['signature'] = current_signature
                            # Aktualisiere actor_obj falls nötig
                            if 'actor_obj' not in existing:
                                actor_name = existing.get('actor')
                                existing['actor_obj'] = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                            new_cache[key] = existing
                            if existing['actor'] not in new_actor_names:
                                new_actor_names.append(existing['actor'])
                            continue

                        # Kein existing - erstelle neues Mesh
                        mesh_to_add = body_mesh.copy(deep=True)
                        # Prüfe ob Mesh bereits Scalars hat (merged mesh mit exit_face_index = -1)
                        has_scalars = 'speaker_face' in mesh_to_add.cell_data

                        # Setze Edge-Color basierend auf Highlight-Status
                        edge_color = 'red' if is_highlighted else 'black'
                        # Basis-Linienbreiten: rot identisch mit Surface-Umrandungen (1.0)
                        base_line_width = 1.0 if is_highlighted else 1.0
                        line_width = self._get_scaled_line_width(base_line_width)

                        # 🎯 FIX: Verwende eindeutigen Actor-Namen basierend auf Array-ID, Speaker-Index und Geometrie-Index
                        # Dies stellt sicher, dass jeder Actor eine persistente ID hat, die nicht überschrieben wird
                        unique_actor_name = self._create_speaker_actor_name(array_id, int(idx), geom_idx)
                        
                        # #region agent log
                        try:
                            import json
                            import time as time_module
                            speaker_array_obj = task.get('speaker_array')
                            speaker_array_id_from_obj = getattr(speaker_array_obj, 'id', None) if speaker_array_obj else None
                            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "PLOT_ARRAY_ID_DEBUG",
                                    "location": "PlotSPL3DSpeaker.py:draw_speakers:creating_actor",
                                    "message": "Creating actor for speaker - array_id check",
                                    "data": {
                                        "task_array_id": str(array_id),
                                        "task_array_id_type": type(array_id).__name__,
                                        "speaker_array.id": str(speaker_array_id_from_obj) if speaker_array_id_from_obj is not None else None,
                                        "speaker_idx": int(idx),
                                        "geom_idx": int(geom_idx) if isinstance(geom_idx, int) else str(geom_idx),
                                        "unique_actor_name": unique_actor_name,
                                        "x": float(x),
                                        "y": float(y),
                                        "z": float(z),
                                        "array_id == speaker_array.id": bool(str(array_id) == str(speaker_array_id_from_obj)) if speaker_array_id_from_obj is not None else None
                                    },
                                    "timestamp": int(time_module.time() * 1000)
                                }) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        
                        if exit_face_index == -1 or has_scalars:
                            # Mesh hat bereits Scalars (merged mesh) - nutze diese
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                scalars='speaker_face',
                                cmap=[body_color, exit_color],
                                opacity=1.0,
                                edge_color=edge_color,
                                line_width=line_width,
                                category='speakers',
                                name=unique_actor_name,
                            )
                        elif exit_face_index is not None and mesh_to_add.n_cells > 0:
                            # Einzelnes Mesh - erstelle Scalars für eine Exit-Face
                            scalars = np.zeros(mesh_to_add.n_cells, dtype=int)
                            scalars[int(exit_face_index)] = 1
                            mesh_to_add.cell_data['speaker_face'] = scalars
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                scalars='speaker_face',
                                cmap=[body_color, exit_color],
                                opacity=1.0,
                                edge_color=edge_color,
                                line_width=line_width,
                                category='speakers',
                                name=unique_actor_name,
                            )
                        else:
                            actor_name = self._add_overlay_mesh(
                                mesh_to_add,
                                color=body_color,
                                opacity=1.0,
                                edge_color=edge_color,
                                line_width=line_width,
                                category='speakers',
                                name=unique_actor_name,
                            )
                        # Hole das Actor-Objekt aus dem Renderer
                        actor_obj = self.plotter.renderer.actors.get(actor_name) if actor_name else None
                        new_cache[key] = {'mesh': mesh_to_add, 'actor': actor_name, 'actor_obj': actor_obj}
                        if actor_name not in new_actor_names:
                            new_actor_names.append(actor_name)
                        new_cache[key]['signature'] = self._speaker_signature_from_mesh(mesh_to_add, exit_face_index)

        with perf_section("PlotSPL3DOverlays.draw_speakers.cleanup", total_speakers=total_speakers):
            # 🎯 FIX: Entferne nur Actors, die zu Arrays gehören, die versteckt sind oder nicht mehr existieren
            # Arrays, die noch geplottet werden sollen, aber nicht in diesem Durchlauf neu geplottet wurden,
            # sollten im Cache bleiben
            for key, info in old_cache.items():
                if key not in new_cache:
                    # Prüfe, ob das Array noch existiert und nicht versteckt ist
                    array_id_from_key = str(key[0]) if isinstance(key, tuple) and len(key) > 0 else None
                    if array_id_from_key:
                        array_still_visible = False
                        # Prüfe, ob das Array noch in speaker_arrays vorhanden ist und nicht versteckt ist
                        for array_name, speaker_array in speaker_arrays.items():
                            speaker_array_id = getattr(speaker_array, 'id', None)
                            if speaker_array_id is not None and str(speaker_array_id) == array_id_from_key:
                                hide_status = getattr(speaker_array, 'hide', False)
                                if not hide_status:
                                    array_still_visible = True
                                break
                            elif str(array_name) == array_id_from_key:
                                hide_status = getattr(speaker_array, 'hide', False)
                                if not hide_status:
                                    array_still_visible = True
                                break
                        
                        # Wenn das Array noch sichtbar ist, behalte es im Cache (nicht entfernen)
                        if array_still_visible:
                            # #region agent log - ACTOR_KEEP_DEBUG
                            try:
                                import json
                                import time as time_module
                                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "ACTOR_KEEP_DEBUG",
                                        "location": "PlotSPL3DSpeaker.py:draw_speakers:cleanup",
                                        "message": "Keeping actor - array still visible",
                                        "data": {
                                            "key": str(key),
                                            "array_id": array_id_from_key,
                                            "actor_name": str(info.get('actor', 'unknown'))
                                        },
                                        "timestamp": int(time_module.time() * 1000)
                                    }) + "\n")
                            except Exception:
                                pass
                            # #endregion
                            # Füge zurück zu new_cache, damit es nicht entfernt wird
                            new_cache[key] = info
                            if info.get('actor') and info['actor'] not in new_actor_names:
                                new_actor_names.append(info['actor'])
                            continue
                    
                    # Array ist versteckt oder existiert nicht mehr - entferne Actor
                    # #region agent log - ACTOR_REMOVAL_DEBUG
                    try:
                        import json
                        import time as time_module
                        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "ACTOR_REMOVAL_DEBUG",
                                "location": "PlotSPL3DSpeaker.py:draw_speakers:cleanup",
                                "message": "Removing actor - array hidden or removed",
                                "data": {
                                    "key": str(key),
                                    "array_id": array_id_from_key or "unknown",
                                    "actor_name": str(info.get('actor', 'unknown')),
                                    "old_cache_size": len(old_cache),
                                    "new_cache_size": len(new_cache)
                                },
                                "timestamp": int(time_module.time() * 1000)
                            }) + "\n")
                    except Exception:
                        pass
                    # #endregion
                    self._remove_actor(info['actor'])

            self._speaker_actor_cache = new_cache
            self._category_actors['speakers'] = new_actor_names
            # Aktualisiere persistenten Array-Cache basierend auf dem neuen Cache:
            # 🎯 FIX: Aktualisiere nur die Arrays, die neu geplottet wurden, behalte andere im Cache
            # Speichere pro (array_id, speaker_index) die Hauptgeometrie (geom_idx == 0)
            # WICHTIG: Ersetze nicht den gesamten Cache, sondern aktualisiere nur die geänderten Arrays
            arrays_updated_in_this_run: set[str] = set()
            for key, info in new_cache.items():
                if not isinstance(key, tuple) or len(key) < 3:
                    continue
                array_id, sp_idx, geom_idx = key[0], int(key[1]), key[2]
                # Nur die erste Geometrie (Hauptkörper) pro Speaker im persistenten Cache speichern
                if geom_idx != 0:
                    continue
                geom_sig = self._speaker_geometry_param_cache.get((array_id, sp_idx))
                # 🎯 FIX: Stelle sicher, dass array_id als String gespeichert wird (speaker_array.id)
                array_id_str = str(array_id)
                arrays_updated_in_this_run.add(array_id_str)
                self._overlay_array_cache[(array_id_str, sp_idx)] = {
                    'actor_entry': info,
                    'geometry_param_signature': geom_sig,
                }
                # #region agent log
                try:
                    import json
                    import time as time_module
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "CACHE_SAVE_ID",
                            "location": "PlotSPL3DSpeaker.py:draw_speakers:save_to_overlay_cache",
                            "message": "Saving to overlay_array_cache",
                            "data": {
                                "array_id": array_id_str,
                                "array_id_type": type(array_id_str).__name__,
                                "speaker_idx": int(sp_idx),
                                "has_actor_entry": bool(info.get('actor_entry')),
                                "has_geom_sig": bool(geom_sig is not None)
                            },
                            "timestamp": int(time_module.time() * 1000)
                        }) + "\n")
                except Exception:
                    pass
                # #endregion
            
            # 🎯 FIX: Entferne nur die Arrays aus dem Cache, die nicht mehr geplottet werden (hide=True)
            # Aber behalte Arrays, die noch geplottet werden, aber nicht in diesem Durchlauf aktualisiert wurden
            arrays_to_remove_from_cache: list[tuple[str, int]] = []
            for cache_key in list(self._overlay_array_cache.keys()):
                if not isinstance(cache_key, tuple) or len(cache_key) < 2:
                    continue
                cached_array_id, cached_sp_idx = str(cache_key[0]), int(cache_key[1])
                # Entferne nur, wenn das Array nicht in diesem Durchlauf aktualisiert wurde
                # UND wenn das Array nicht mehr in speaker_arrays vorhanden ist oder hide=True
                if cached_array_id not in arrays_updated_in_this_run:
                    # Prüfe, ob das Array noch existiert und nicht versteckt ist
                    array_still_exists = False
                    for array_name, speaker_array in speaker_arrays.items():
                        speaker_array_id = getattr(speaker_array, 'id', None)
                        if speaker_array_id is not None and str(speaker_array_id) == cached_array_id:
                            hide_status = getattr(speaker_array, 'hide', False)
                            if not hide_status:
                                array_still_exists = True
                            break
                        elif str(array_name) == cached_array_id:
                            hide_status = getattr(speaker_array, 'hide', False)
                            if not hide_status:
                                array_still_exists = True
                            break
                    
                    # Wenn das Array nicht mehr existiert oder versteckt ist, entferne es aus dem Cache
                    if not array_still_exists:
                        arrays_to_remove_from_cache.append(cache_key)
            
            for cache_key in arrays_to_remove_from_cache:
                self._overlay_array_cache.pop(cache_key, None)
            
            # #region agent log
            import json
            import time as time_module
            cache_size_after = len(self._overlay_array_cache)
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H4",
                    "location": "PlotSPL3DSpeaker.py:draw_speakers:1380",
                    "message": "Cache updated after draw_speakers",
                    "data": {
                        "overlay_array_cache_size_after": cache_size_after,
                        "new_cache_size": len(new_cache),
                        "arrays_updated_in_this_run": list(arrays_updated_in_this_run),
                        "arrays_removed_from_cache": len(arrays_to_remove_from_cache)
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
            # #endregion

        with perf_section("PlotSPL3DOverlays.draw_speakers.update_highlights"):
            # Aktualisiere Edge-Color für hervorgehobene Lautsprecher
            self._update_speaker_highlights(settings)

        with perf_section("PlotSPL3DOverlays.draw_speakers.render"):
            # Render-Update triggern, damit Änderungen sichtbar werden
            try:
                self.plotter.render()
            except Exception:  # noqa: BLE001
                pass

        # Am Ende von draw_speakers keine zusätzlichen Debug-Ausgaben mehr

    # ------------------------------------------------------------------
    # Hilfsfunktionen für Lautsprecher (öffentlich über Overlay erreichbar)
    # ------------------------------------------------------------------
    def build_cabinet_lookup(self, container) -> Dict:
        lookup: Dict[str, object] = {}
        if container is None:
            return lookup

        data = getattr(container, 'data', None)
        if not isinstance(data, dict):
            return lookup

        names = data.get('speaker_names') or []
        cabinets = data.get('cabinet_data') or []

        for name, cabinet in zip(names, cabinets):
            if not isinstance(name, str):
                continue
            lookup[name] = cabinet
            lookup.setdefault(name.lower(), cabinet)

        alias_mapping = getattr(container, '_speaker_name_mapping', {})
        if isinstance(alias_mapping, dict):
            for alias, actual in alias_mapping.items():
                if actual in lookup and isinstance(alias, str):
                    lookup.setdefault(alias, lookup[actual])
                    lookup.setdefault(alias.lower(), lookup[actual])

        return lookup

    # ------------------------------------------------------------------
    # Picking / Highlighting
    # ------------------------------------------------------------------
    def _get_speaker_info_from_actor(self, actor: Any, actor_name: str) -> Optional[Tuple[str, int]]:
        """Extrahiert Array-ID und Speaker-Index aus einem Speaker-Actor.

        Returns:
            Tuple[str, int] oder None: (array_id, speaker_index) wenn gefunden, sonst None
        """
        try:
            renderer = self.plotter.renderer
            if not renderer or not actor:
                return None

            # WICHTIG: PyVista generiert Actor-Namen neu, daher müssen wir direkt über das Actor-Objekt suchen
            # Strategie: Verwende actor_obj aus dem Cache für direkten Vergleich

            # Durchsuche alle Cache-Einträge
            # WICHTIG: Ein Lautsprecher kann mehrere Flächen (Geometrien) haben, die alle zum selben Speaker gehören
            # Sammle alle passenden Einträge und wähle den ersten (array_id und speaker_idx sind für alle gleich)
            matching_entries = []

            for key, info in self._speaker_actor_cache.items():
                # Versuche zuerst actor_obj (direktes Objekt)
                cached_actor_obj = info.get('actor_obj')
                if cached_actor_obj is actor:
                    array_id, speaker_idx, geom_idx = key
                    matching_entries.append((array_id, speaker_idx, geom_idx, 'actor_obj'))

                # Fallback: Versuche über Actor-Namen
                cached_actor_name = info.get('actor')
                if cached_actor_name:
                    # Versuche den Actor aus dem Renderer zu holen (über den gespeicherten Namen)
                    cached_actor = renderer.actors.get(cached_actor_name)

                    # Direkter Objekt-Vergleich
                    if cached_actor is actor:
                        array_id, speaker_idx, geom_idx = key
                        # Prüfe ob dieser Eintrag bereits in matching_entries ist (über actor_obj gefunden)
                        if not any(entry[0] == array_id and entry[1] == speaker_idx and entry[2] == geom_idx for entry in matching_entries):
                            matching_entries.append((array_id, speaker_idx, geom_idx, 'actor_name'))

            # Wenn Einträge gefunden wurden, gib den ersten zurück (array_id und speaker_idx sind für alle gleich)
            if matching_entries:
                array_id, speaker_idx, geom_idx, match_type = matching_entries[0]
                return (str(array_id), int(speaker_idx))

            # Falls nicht gefunden: Der Actor-Name im Cache könnte veraltet sein
            # Durchsuche alle Renderer-Actors und finde den, der mit dem picked_actor übereinstimmt
            for renderer_actor_name, renderer_actor in renderer.actors.items():
                if renderer_actor is actor:
                    # Jetzt müssen wir herausfinden, welcher Cache-Eintrag zu diesem Actor gehört
                    # Da sich die Namen geändert haben, müssen wir alle Renderer-Actors durchsuchen
                    # und mit den Cache-Einträgen vergleichen
                    for cache_key, cache_info in self._speaker_actor_cache.items():
                        cached_actor_name = cache_info.get('actor')
                        if not cached_actor_name:
                            continue

                        # Hole den Actor aus dem Renderer über den gespeicherten Namen
                        cached_actor_from_renderer = renderer.actors.get(cached_actor_name)

                        # Wenn der Actor aus dem Renderer mit dem picked_actor übereinstimmt
                        if cached_actor_from_renderer is actor:
                            array_id, speaker_idx, geom_idx = cache_key
                            return (str(array_id), int(speaker_idx))

                    # Wenn wir hier ankommen, haben wir den Actor im Renderer gefunden, aber nicht im Cache
                    # Versuche über den aktuellen Renderer-Namen zu suchen (falls er zufällig im Cache ist)
                    for cache_key, cache_info in self._speaker_actor_cache.items():
                        cached_actor_name = cache_info.get('actor')
                        if cached_actor_name == renderer_actor_name:
                            array_id, speaker_idx, geom_idx = cache_key
                            return (str(array_id), int(speaker_idx))
                    break

            return None
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return None

    def _update_speaker_highlights(self, settings) -> None:
        """Aktualisiert die Edge-Color für hervorgehobene Lautsprecher."""
        try:
            # Unterstütze sowohl einzelne ID (Rückwärtskompatibilität) als auch Liste von IDs
            highlight_array_id = getattr(settings, 'active_speaker_array_highlight_id', None)
            highlight_array_ids = getattr(settings, 'active_speaker_array_highlight_ids', [])
            highlight_indices = getattr(settings, 'active_speaker_highlight_indices', [])

            # Konvertiere zu Liste von Strings für konsistenten Vergleich
            if highlight_array_ids:
                highlight_array_ids_str = [str(aid) for aid in highlight_array_ids]
            elif highlight_array_id:
                highlight_array_ids_str = [str(highlight_array_id)]
            else:
                highlight_array_ids_str = []

            if not highlight_array_ids_str and not highlight_indices:
                # Keine Highlights - setze alle auf schwarz
                for key, info in self._speaker_actor_cache.items():
                    actor_name = info.get('actor')
                    if actor_name:
                        try:
                            actor = self.plotter.renderer.actors.get(actor_name)
                            if actor:
                                prop = actor.GetProperty()
                                if prop:
                                    prop.SetEdgeColor(0, 0, 0)  # Schwarz
                                    prop.SetEdgeVisibility(True)
                        except Exception:  # noqa: BLE001
                            pass
                return

            # Aktualisiere Edge-Color für hervorgehobene Lautsprecher
            for key, info in self._speaker_actor_cache.items():
                array_id, speaker_idx, geom_idx = key
                array_id_str = str(array_id)

                # Prüfe ob dieser Speaker hervorgehoben werden soll
                is_highlighted = False
                if highlight_indices:
                    # Spezifische Speaker-Indices
                    # Konvertiere beide zu String für Vergleich
                    highlight_indices_str = [(str(aid), int(idx)) for aid, idx in highlight_indices]
                    is_highlighted = (array_id_str, int(speaker_idx)) in highlight_indices_str
                elif highlight_array_ids_str:
                    # Alle Speaker der Arrays - prüfe ob Array-ID in der Liste ist
                    is_highlighted = array_id_str in highlight_array_ids_str

                # Versuche zuerst actor_obj (direktes Objekt), dann actor_name
                actor = info.get('actor_obj')
                actor_name = info.get('actor')

                if actor is None and actor_name:
                    # Fallback: Hole Actor über Namen
                    actor = self.plotter.renderer.actors.get(actor_name)

                if actor:
                    try:
                        prop = actor.GetProperty()
                        if prop:
                            if is_highlighted:
                                prop.SetEdgeColor(1, 0, 0)  # Rot
                                # Rote Umrandung identisch mit Surface-Umrandungen (1.0)
                                prop.SetLineWidth(self._get_scaled_line_width(1.0))
                            else:
                                prop.SetEdgeColor(0, 0, 0)  # Schwarz
                                # Schwarze Umrandung
                                prop.SetLineWidth(self._get_scaled_line_width(1.0))
                            # WICHTIG: Edge-Visibility muss aktiviert sein
                            prop.SetEdgeVisibility(True)
                            # Stelle sicher, dass Edges angezeigt werden
                            prop.SetRepresentationToSurface()  # Surface-Darstellung für Edges
                            # Render-Update triggern
                            actor.Modified()
                    except Exception:  # noqa: BLE001
                        import traceback

                        traceback.print_exc()

            # Render-Update triggern, damit Änderungen sichtbar werden
            try:
                self.plotter.render()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Geometrie-Hilfsmethoden (unverändert aus Overlay übernommen)
    # ------------------------------------------------------------------
    def _transform_cached_geometry_to_position(
        self, cached_geoms: List[Tuple[Any, Optional[int]]],
        old_x: float, old_y: float, old_z: float,
        new_x: float, new_y: float, new_z: float
    ) -> List[Tuple[Any, Optional[int]]]:
        """Transformiert gecachte Geometrien von alter zu neuer Position."""
        # 🚀 OPTIMIERUNG: Berechne Delta einmal vor der Schleife
        delta_x = new_x - old_x
        delta_y = new_y - old_y
        delta_z = new_z - old_z
        
        transformed = []
        for mesh, exit_idx in cached_geoms:
            # 🚀 OPTIMIERUNG: Verwende shallow copy wenn möglich, nur kopieren wenn nötig
            # Da translate() inplace=True verwendet, können wir das Original-Mesh modifizieren
            # ABER: Nur wenn es nicht bereits im Plot verwendet wird
            # Für Sicherheit: Verwende copy(), aber ohne deep=True wenn möglich
            try:
                # Versuche shallow copy zuerst (schneller)
                new_mesh = mesh.copy(deep=False)
            except Exception:
                # Fallback zu deep copy wenn shallow nicht funktioniert
                new_mesh = mesh.copy(deep=True)

            if hasattr(new_mesh, 'bounds') and new_mesh.bounds is not None:
                bounds = new_mesh.bounds
                current_center_x = (bounds[0] + bounds[1]) / 2.0
                current_center_y = (bounds[2] + bounds[3]) / 2.0
                current_center_z = (bounds[4] + bounds[5]) / 2.0

                # Berechne Delta basierend auf Mesh-Center
                mesh_delta_x = new_x - current_center_x
                mesh_delta_y = new_y - current_center_y
                mesh_delta_z = new_z - current_center_z

                new_mesh.translate((mesh_delta_x, mesh_delta_y, mesh_delta_z), inplace=True)
            else:
                # Verwende vorberechnetes Delta
                new_mesh.translate((delta_x, delta_y, delta_z), inplace=True)

            transformed.append((new_mesh, exit_idx))

        return transformed

    def _create_geometry_param_signature(
        self, speaker_array, index: int, x: float, y: float, z: float, cabinet_lookup: Dict
    ) -> tuple:
        """Erstellt eine Signatur für Geometrie-Parameter (Position, Rotation, Cabinet-Daten)."""
        speaker_name = self._get_speaker_name(speaker_array, index)
        cabinet_raw = cabinet_lookup.get(speaker_name)
        if cabinet_raw is None and isinstance(speaker_name, str):
            cabinet_raw = cabinet_lookup.get(speaker_name.lower())

        params = [
            round(float(x), 4),
            round(float(y), 4),
            round(float(z), 4),
        ]

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), index)
        params.append(round(float(azimuth), 4) if azimuth is not None else 0.0)

        angle_val = self._array_value(getattr(speaker_array, 'source_angle', None), index)
        params.append(round(float(angle_val), 4) if angle_val is not None else 0.0)

        site_val = self._array_value(getattr(speaker_array, 'source_site', None), index)
        params.append(round(float(site_val), 4) if site_val is not None else 0.0)

        if cabinet_raw is not None:
            import hashlib

            if isinstance(cabinet_raw, (list, tuple)):
                cab_str = str(sorted(str(cab) for cab in cabinet_raw))
            else:
                cab_str = str(cabinet_raw)
            cab_hash = hashlib.md5(cab_str.encode('utf-8')).hexdigest()[:8]
            params.append(cab_hash)
        else:
            params.append(None)

        return tuple(params)

    def _create_geometry_cache_key(self, array_id: str, speaker_array, index: int,
                                   speaker_name: str, configuration: str, cabinet_raw) -> str:
        """Erstellt einen Cache-Key für die Geometrie basierend auf relevanten Parametern."""
        import hashlib

        params: List[str] = []
        params.append(str(array_id))
        params.append(str(index))
        params.append(str(speaker_name))
        params.append(str(configuration))

        if cabinet_raw is not None:
            if isinstance(cabinet_raw, list):
                for cab in cabinet_raw:
                    if isinstance(cab, dict):
                        for key in ['width', 'depth', 'front_height', 'back_height', 'configuration',
                                    'stack_layout', 'cardio', 'angle_point', 'x_offset', 'y_offset', 'z_offset']:
                            params.append(f"{key}:{cab.get(key, '')}")
            elif isinstance(cabinet_raw, dict):
                for key in ['width', 'depth', 'front_height', 'back_height', 'configuration',
                            'stack_layout', 'cardio', 'angle_point', 'x_offset', 'y_offset', 'z_offset']:
                    params.append(f"{key}:{cabinet_raw.get(key, '')}")

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), index)
        params.append(f"azimuth:{azimuth:.4f}" if azimuth is not None else "azimuth:0")

        angle_val = self._array_value(getattr(speaker_array, 'source_angle', None), index)
        params.append(f"angle:{angle_val:.4f}" if angle_val is not None else "angle:0")

        site_val = self._array_value(getattr(speaker_array, 'source_site', None), index)
        params.append(f"site:{site_val:.4f}" if site_val is not None else "site:0")

        params.append(f"count:{getattr(speaker_array, 'number_of_sources', 0)}")

        param_string = '|'.join(str(p) for p in params)
        return hashlib.md5(param_string.encode('utf-8')).hexdigest()

    def _create_array_signature(self, speaker_array, array_id: str,
                                array_pos_x: float, array_pos_y: float, array_pos_z: float,
                                cabinet_lookup: Dict) -> tuple:
        """Erstellt eine Signatur für ein Flown-Array (Array-Level Cache).
        
        Diese Signatur wird verwendet, um zu prüfen, ob alle Speaker-Geometrien eines Flown-Arrays
        aus dem Cache geladen werden können (ohne Neuberechnung). Wenn sich die Signatur ändert,
        müssen die Geometrien neu berechnet werden.
        
        Die Signatur enthält alle Parameter, die die Geometrie beeinflussen:
        - Array-Positionen (array_pos_x/y/z)
        - Azimuth (source_azimuth)
        - Site (source_site) - Position entlang der Kurve
        - Angle (source_angle) - Zwischenwinkel
        - Speaker-Namen und Cabinet-Hashes
        """
        params: List[Any] = [
            round(float(array_pos_x), 4),
            round(float(array_pos_y), 4),
            round(float(array_pos_z), 4),
        ]

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), 0)
        params.append(round(float(azimuth), 4) if azimuth is not None else 0.0)
        
        # 🎯 FIX: source_site und source_angle zur Array-Signatur hinzufügen
        # Diese Parameter beeinflussen die Geometrie und müssen daher in der Signatur enthalten sein
        site = self._array_value(getattr(speaker_array, 'source_site', None), 0)
        params.append(round(float(site), 4) if site is not None else 0.0)
        
        angle = self._array_value(getattr(speaker_array, 'source_angle', None), 0)
        params.append(round(float(angle), 4) if angle is not None else 0.0)

        speaker_names: List[str] = []
        for idx in range(getattr(speaker_array, 'number_of_sources', 0)):
            speaker_name = self._get_speaker_name(speaker_array, idx)
            if speaker_name:
                speaker_names.append(speaker_name)

        import hashlib

        cabinet_hashes: List[str] = []
        for speaker_name in sorted(set(speaker_names)):
            cabinet_raw = cabinet_lookup.get(speaker_name)
            if cabinet_raw is None and isinstance(speaker_name, str):
                cabinet_raw = cabinet_lookup.get(speaker_name.lower())
            if cabinet_raw is not None:
                if isinstance(cabinet_raw, (list, tuple)):
                    cab_str = str(sorted(str(cab) for cab in cabinet_raw))
                else:
                    cab_str = str(cabinet_raw)
                cab_hash = hashlib.md5(cab_str.encode('utf-8')).hexdigest()[:8]
                cabinet_hashes.append(f"{speaker_name}:{cab_hash}")

        params.append('|'.join(sorted(cabinet_hashes)) if cabinet_hashes else '')
        return tuple(params)

    def _create_stack_signature(self, speaker_array, array_id: str, stack_group_key: tuple,
                                array_pos_x: float, array_pos_y: float, array_pos_z: float,
                                cabinet_lookup: Dict, speaker_indices: List[int] = None) -> tuple:
        """Erstellt eine Signatur für eine Stack-Gruppe (Stack-Level Cache)."""
        params: List[Any] = [
            round(float(array_pos_x), 4),
            round(float(array_pos_y), 4),
            round(float(array_pos_z), 4),
        ]

        if stack_group_key:
            params.extend(
                [round(float(v), 4) if isinstance(v, (int, float)) else str(v) for v in stack_group_key]
            )

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), 0)
        params.append(round(float(azimuth), 4) if azimuth is not None else 0.0)

        # 🎯 WICHTIG: Füge Speaker-Positionen und Polar-Patterns zur Signatur hinzu, damit Änderungen erkannt werden
        if speaker_indices and len(speaker_indices) > 0:
            try:
                xs = getattr(speaker_array, 'source_position_x', None)
                ys = getattr(speaker_array, 'source_position_y', None)
                zs_stack = getattr(speaker_array, 'source_position_z_stack', None)
                polar_patterns = getattr(speaker_array, 'source_polar_pattern', None)
                
                if xs is not None and ys is not None:
                    position_tuples = []
                    pattern_tuples = []
                    for idx in sorted(speaker_indices):
                        # Positionen
                        if isinstance(xs, (list, tuple, np.ndarray)) and isinstance(ys, (list, tuple, np.ndarray)):
                            if idx < len(xs) and idx < len(ys):
                                try:
                                    x_val = round(float(xs[idx]), 4)
                                    y_val = round(float(ys[idx]), 4)
                                    z_val = round(float(zs_stack[idx]), 4) if zs_stack is not None and isinstance(zs_stack, (list, tuple, np.ndarray)) and idx < len(zs_stack) else 0.0
                                    position_tuples.append((x_val, y_val, z_val))
                                except (ValueError, TypeError, IndexError):
                                    position_tuples.append((0.0, 0.0, 0.0))
                            else:
                                position_tuples.append((0.0, 0.0, 0.0))
                        else:
                            position_tuples.append((0.0, 0.0, 0.0))
                        
                        # Polar-Patterns (für Typ-Änderungen)
                        if polar_patterns is not None and isinstance(polar_patterns, (list, tuple, np.ndarray)) and idx < len(polar_patterns):
                            try:
                                pattern_val = str(polar_patterns[idx]) if polar_patterns[idx] is not None else ''
                                pattern_tuples.append(pattern_val)
                            except (ValueError, TypeError, IndexError):
                                pattern_tuples.append('')
                        else:
                            pattern_tuples.append('')
                    
                    params.append(tuple(position_tuples))
                    params.append(tuple(pattern_tuples))
                else:
                    params.append(tuple())
                    params.append(tuple())
            except Exception:
                # Bei Fehler: Leere Tupel hinzufügen, damit Signatur konsistent bleibt
                params.append(tuple())
                params.append(tuple())
        else:
            params.append(tuple())
            params.append(tuple())

        import hashlib

        cabinet_hashes: List[str] = []
        for idx in range(min(3, getattr(speaker_array, 'number_of_sources', 0))):
            speaker_name = self._get_speaker_name(speaker_array, idx)
            if speaker_name:
                cabinet_raw = cabinet_lookup.get(speaker_name)
                if cabinet_raw is None and isinstance(speaker_name, str):
                    cabinet_raw = cabinet_lookup.get(speaker_name.lower())
                if cabinet_raw is not None:
                    if isinstance(cabinet_raw, (list, tuple)):
                        cab_str = str(sorted(str(cab) for cab in cabinet_raw))
                    else:
                        cab_str = str(cabinet_raw)
                    cab_hash = hashlib.md5(cab_str.encode('utf-8')).hexdigest()[:8]
                    cabinet_hashes.append(f"{speaker_name}:{cab_hash}")

        params.append('|'.join(sorted(set(cabinet_hashes))) if cabinet_hashes else '')
        return tuple(params)

    def _identify_stack_groups_from_cabinets(self, speaker_array, array_id: str,
                                             cabinet_lookup: Dict) -> Dict[tuple, List[int]]:
        """Identifiziert Stack-Gruppen basierend auf Cabinet-Daten, bevor Geometrien berechnet werden."""
        stack_groups: Dict[tuple, List[int]] = {}
        configuration = getattr(speaker_array, 'configuration', '').lower()

        if configuration != 'stack':
            return stack_groups

        speaker_count = getattr(speaker_array, 'number_of_sources', 0)
        on_top_width: Optional[float] = None

        for idx in range(speaker_count):
            speaker_name = self._get_speaker_name(speaker_array, idx)
            if not speaker_name:
                continue

            cabinet_raw = cabinet_lookup.get(speaker_name)
            if cabinet_raw is None and isinstance(speaker_name, str):
                cabinet_raw = cabinet_lookup.get(speaker_name.lower())

            if not cabinet_raw:
                continue

            cabinets = self._resolve_cabinet_entries(cabinet_raw, configuration)
            if not cabinets:
                continue

            for cabinet in cabinets:
                entry_config = str(cabinet.get('configuration', configuration)).lower() if isinstance(cabinet, dict) else configuration
                if entry_config != 'stack':
                    continue

                stack_layout = str(cabinet.get('stack_layout', 'beside')).strip().lower() if isinstance(cabinet, dict) else 'beside'
                width = self._safe_float(cabinet, 'width')

                if stack_layout == 'on top':
                    if on_top_width is None:
                        on_top_width = width
                    x_offset = -(on_top_width / 2.0)
                    y_offset = 0.0
                    stack_group_key = ('on_top', round(x_offset, 4), 0.0)
                else:
                    x_offset = self._safe_float(cabinet, 'x_offset') if 'x_offset' in cabinet else (-width / 2.0)
                    y_offset = self._safe_float(cabinet, 'y_offset') if 'y_offset' in cabinet else 0.0
                    stack_group_key = ('beside', round(x_offset, 4), round(y_offset, 4))

                if stack_group_key not in stack_groups:
                    stack_groups[stack_group_key] = []
                if idx not in stack_groups[stack_group_key]:
                    stack_groups[stack_group_key].append(idx)

        return stack_groups

    def _get_stack_group_key_for_speaker(self, speaker_array, idx: int,
                                         cabinet_lookup: Dict) -> Optional[tuple]:
        """Ermittelt den stack_group_key für einen einzelnen Speaker basierend auf Cabinet-Daten."""
        configuration = getattr(speaker_array, 'configuration', '').lower()
        if configuration != 'stack':
            return None

        speaker_name = self._get_speaker_name(speaker_array, idx)
        if not speaker_name:
            return None

        cabinet_raw = cabinet_lookup.get(speaker_name)
        if cabinet_raw is None and isinstance(speaker_name, str):
            cabinet_raw = cabinet_lookup.get(speaker_name.lower())

        if not cabinet_raw:
            return None

        cabinets = self._resolve_cabinet_entries(cabinet_raw, configuration)
        if not cabinets:
            return None

        for cabinet in cabinets:
            entry_config = str(cabinet.get('configuration', configuration)).lower() if isinstance(cabinet, dict) else configuration
            if entry_config != 'stack':
                continue

            stack_layout = str(cabinet.get('stack_layout', 'beside')).strip().lower() if isinstance(cabinet, dict) else 'beside'
            width = self._safe_float(cabinet, 'width')

            if stack_layout == 'on top':
                x_offset = -(width / 2.0)
                y_offset = 0.0
                return ('on_top', round(x_offset, 4), 0.0)
            else:
                x_offset = self._safe_float(cabinet, 'x_offset') if 'x_offset' in cabinet else (-width / 2.0)
                y_offset = self._safe_float(cabinet, 'y_offset') if 'y_offset' in cabinet else 0.0
                return ('beside', round(x_offset, 4), round(y_offset, 4))

        return None

    def _build_speaker_geometries(
        self,
        speaker_array,
        index: int,
        x: float,
        y: float,
        z_reference: float,
        cabinet_lookup: Dict,
        container,
    ) -> List[Tuple[Any, Optional[int]]]:
        """Erzeugt die Gehäusegeometrien für einen einzelnen Lautsprecher."""
        del container
        speaker_name = self._get_speaker_name(speaker_array, index)
        configuration = getattr(speaker_array, 'configuration', '')

        cabinet_raw = cabinet_lookup.get(speaker_name)
        if cabinet_raw is None and isinstance(speaker_name, str):
            cabinet_raw = cabinet_lookup.get(speaker_name.lower())

        cabinets = self._resolve_cabinet_entries(cabinet_raw, configuration)
        if not cabinets:
            return []

        geoms: List[Tuple[Any, Optional[int]]] = []

        config_lower = (configuration or '').lower()
        if config_lower == 'flown':
            reference_array = getattr(speaker_array, 'source_position_z_flown', None)
        else:
            reference_array = getattr(speaker_array, 'source_position_z_stack', None)
        source_top_array = getattr(speaker_array, 'source_position_z_flown', None)
        speaker_count = 0
        for candidate in (reference_array, source_top_array):
            if candidate is not None:
                try:
                    speaker_count = max(speaker_count, len(candidate))
                except Exception:  # noqa: BLE001
                    pass

        if config_lower == 'flown' and cabinets:
            if len(cabinets) == 1 and speaker_count > 1:
                template = cabinets[0]
                cabinets = [dict(template) for _ in range(speaker_count)]

        azimuth = self._array_value(getattr(speaker_array, 'source_azimuth', None), index)
        azimuth_rad = np.deg2rad(azimuth) if azimuth is not None else 0.0

        processed_entries: List[Dict] = []
        stack_state: Dict[tuple, Dict[str, float]] = {}
        on_top_width: Optional[float] = None
        flown_height_acc: float = 0.0
        flown_groups: List[List[int]] = []
        current_flown_group: List[int] = []
        flown_entry_index = 0

        for idx_cabinet, cabinet in enumerate(cabinets):
            entry_config = str(cabinet.get('configuration', configuration)).lower() if isinstance(cabinet, dict) else config_lower
            width = self._safe_float(cabinet, 'width')
            depth = self._safe_float(cabinet, 'depth')
            front_height = self._safe_float(cabinet, 'front_height')
            back_height = self._safe_float(cabinet, 'back_height') if 'back_height' in cabinet else front_height
            cardio = bool(cabinet.get('cardio', False)) if isinstance(cabinet, dict) else False
            angle_point_raw = ''
            if isinstance(cabinet, dict):
                angle_point_raw = str(cabinet.get('angle_point', '')).strip().lower()
            if angle_point_raw not in {'front', 'back', 'center', 'none'}:
                angle_point_raw = 'front'
            stack_layout = str(cabinet.get('stack_layout', 'beside')).strip().lower() if isinstance(cabinet, dict) else 'beside'

            height = max(front_height, back_height)
            if width <= 0 or depth <= 0 or height <= 0:
                continue

            x_offset = self._safe_float(cabinet, 'x_offset') if 'x_offset' in cabinet else (-width / 2.0)
            y_offset = self._safe_float(cabinet, 'y_offset') if 'y_offset' in cabinet else 0.0
            z_offset = self._safe_float(cabinet, 'z_offset') if 'z_offset' in cabinet else 0.0

            stack_group_key = None
            if entry_config == 'stack':
                if stack_layout == 'on top':
                    if on_top_width is None:
                        on_top_width = width
                    x_offset = -(on_top_width / 2.0)
                    y_offset = 0.0
                    stack_group_key = ('on_top', round(x_offset, 4), 0.0)
                    state = stack_state.setdefault(
                        stack_group_key,
                        {'height': 0.0, 'base_x': x_offset, 'base_y': y_offset},
                    )
                    stack_offset = state['height']
                    state['height'] += height
                    x_offset = state['base_x']
                    y_offset = state['base_y']
                else:
                    stack_group_key = ('beside', round(x_offset, 4), round(y_offset, 4))
                    stack_offset = 0.0
            else:
                if entry_config == 'flown':
                    stack_offset = flown_height_acc
                    flown_height_acc += height
                else:
                    stack_offset = 0.0

            if entry_config != 'flown':
                if current_flown_group:
                    flown_groups.append(current_flown_group)
                    current_flown_group = []

            entry_idx = len(processed_entries)
            if entry_config == 'flown':
                source_idx = flown_entry_index
                flown_entry_index += 1
            else:
                source_idx = len(processed_entries)

            if angle_point_raw == 'back':
                effective_height = back_height
            elif angle_point_raw == 'center':
                effective_height = (front_height + back_height) / 2.0
            elif angle_point_raw == 'none':
                effective_height = max(front_height, back_height)
            else:
                effective_height = front_height
            if effective_height <= 0:
                effective_height = height

            processed_entries.append({
                'width': width,
                'depth': depth,
                'front_height': front_height,
                'back_height': back_height,
                'height': height,
                'cardio': cardio,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'z_offset': z_offset,
                'stack_offset': stack_offset,
                'configuration': entry_config,
                'stack_layout': stack_layout,
                'stack_group_key': stack_group_key,
                'source_index': source_idx,
                'effective_height': effective_height,
                'angle_point': angle_point_raw,
            })

            if entry_config == 'flown':
                current_flown_group.append(entry_idx)

        if current_flown_group:
            flown_groups.append(current_flown_group)

        pos_y_raw = getattr(
            speaker_array,
            'source_position_calc_y',
            getattr(speaker_array, 'source_position_y', None),
        )
        speaker_count = max(
            speaker_count,
            self._sequence_length(pos_y_raw),
        )
        site_values = np.asarray(
            self._float_sequence(getattr(speaker_array, 'source_site', None), speaker_count, 0.0),
            dtype=float,
        )
        position_y_values = np.asarray(
            self._float_sequence(
                getattr(
                    speaker_array,
                    'source_position_calc_y',
                    getattr(speaker_array, 'source_position_y', None),
                ),
                speaker_count,
                0.0,
            ),
            dtype=float,
        )
        source_position_calc_z_raw = getattr(speaker_array, 'source_position_calc_z', None)
        calc_z_values = np.asarray(
            self._float_sequence(
                source_position_calc_z_raw,
                speaker_count,
                float(z_reference),
            ),
            dtype=float,
        )
        # #region agent log
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H3",
                "location": "PlotSPL3DSpeaker.py:_build_speaker_geometries:1921",
                "message": "Building geometries - Z values",
                "data": {
                    "index": int(index),
                    "z_reference": float(z_reference),
                    "has_source_position_calc_z": source_position_calc_z_raw is not None,
                    "source_position_calc_z_first": float(source_position_calc_z_raw[0]) if source_position_calc_z_raw is not None and len(source_position_calc_z_raw) > 0 else None,
                    "calc_z_values_first": float(calc_z_values[0]) if calc_z_values.size > 0 else None,
                    "speaker_count": int(speaker_count)
                },
                "timestamp": int(__import__('time').time() * 1000)
            }) + "\n")
        # #endregion
        angle_values = np.asarray(
            self._float_sequence(
                getattr(speaker_array, 'source_angle', None),
                speaker_count,
                0.0,
            ),
            dtype=float,
        )
        base_site_angle = float(site_values[0]) if site_values.size > 0 else 0.0
        total_angle_values = np.full(speaker_count, base_site_angle, dtype=float)
        if speaker_count > 1:
            cumulative = np.cumsum(angle_values[1:])
            total_angle_values[1:] = base_site_angle - cumulative

        for entry_idx, entry in enumerate(processed_entries):
            width = entry['width']
            depth = entry['depth']
            front_height = entry['front_height']
            back_height = entry['back_height']
            height = entry['height']
            cardio = entry['cardio']
            x_offset = entry['x_offset']
            y_offset = entry['y_offset']
            z_offset = entry['z_offset']
            stack_offset = entry['stack_offset']
            entry_config = entry['configuration']
            stack_layout = entry['stack_layout']

            if entry_config == 'stack' and stack_layout == 'on top':
                if on_top_width is not None:
                    x_offset = -(on_top_width / 2.0)
                    y_offset = 0.0
                    width = on_top_width
            source_idx = entry.get('source_index', entry_idx)
            box_key: Optional[tuple[float, float, float]] = None

            if entry_config == 'flown':
                effective_height = entry.get('effective_height', height)
                if effective_height <= 0:
                    effective_height = height
                angle_deg = total_angle_values[source_idx] if source_idx < len(total_angle_values) else base_site_angle
                angle_point_debug = entry.get('angle_point', 'front')
                segment_info = getattr(speaker_array, '_flown_segment_geometry', [])
                segment = segment_info[source_idx] if source_idx < len(segment_info) else {}
                top_z_info = float(segment.get('top_z', calc_z_values[source_idx] + effective_height / 2.0 if source_idx < len(calc_z_values) else z_reference))
                pivot_y_info = float(segment.get('pivot_y', position_y_values[source_idx] if source_idx < len(position_y_values) else 0.0))
                pivot_z_info = float(segment.get('pivot_z', top_z_info))
                center_y_info = float(segment.get('center_y', position_y_values[source_idx] if source_idx < len(position_y_values) else 0.0))
                center_z_info = float(segment.get('center_z', calc_z_values[source_idx] if source_idx < len(calc_z_values) else z_reference))
                front_height_segment = self._safe_float(segment, 'front_height', front_height)
                back_height_segment = self._safe_float(segment, 'back_height', back_height)
                effective_height_segment = self._safe_float(segment, 'effective_height', effective_height)
                if effective_height_segment > 0:
                    effective_height = effective_height_segment
                front_height_effective = front_height_segment if front_height_segment > 0 else front_height
                back_height_effective = back_height_segment if back_height_segment > 0 else back_height
                back_height_symmetrized = 0.5 * (front_height_effective + back_height_effective)
                if back_height_symmetrized <= 0:
                    back_height_symmetrized = back_height_effective
                angle_point_lower = str(segment.get('angle_point', angle_point_debug)).strip().lower()
                if angle_point_lower not in {'front', 'back', 'center', 'none'}:
                    angle_point_lower = str(angle_point_debug).strip().lower()
                x_min = x + x_offset
                x_max = x_min + width

                pivot_y_world = pivot_y_info + y_offset
                pivot_z_world = pivot_z_info - z_offset
                center_y_world = center_y_info + y_offset
                center_z_world = center_z_info - z_offset
                top_z_world = top_z_info - z_offset

                if angle_point_lower == 'back':
                    y_min = pivot_y_world
                    y_max = pivot_y_world + depth
                elif angle_point_lower == 'front':
                    y_max = pivot_y_world
                    y_min = pivot_y_world - depth
                else:
                    y_min = pivot_y_world - depth / 2.0
                    y_max = pivot_y_world + depth / 2.0

                z_max = top_z_world
                z_min = top_z_world - effective_height

                box_key = (float(width), float(depth), float(effective_height))
                template = self._get_box_template(*box_key)
                body = template.copy(deep=True)
                if angle_point_lower == 'back':
                    translation_y = pivot_y_world + depth
                elif angle_point_lower == 'front':
                    translation_y = pivot_y_world
                else:
                    translation_y = pivot_y_world + depth / 2.0
                body.translate((x_min, translation_y, z_min), inplace=True)
                body = self._apply_flown_cabinet_shape(
                    body,
                    front_height=front_height_effective,
                    back_height=back_height_effective,
                    top_reference=top_z_world,
                    angle_point=angle_point_lower,
                )
                if abs(angle_deg) > 1e-6:
                    body.rotate_x(angle_deg, point=((x_min + x_max) / 2.0, pivot_y_world, pivot_z_world), inplace=True)
            else:
                x_min = x + x_offset
                x_max = x_min + width
                front_y_world = y + y_offset
                y_max = front_y_world
                y_min = front_y_world - depth
                if config_lower != 'flown' and z_reference is not None:
                    reference_value = z_reference
                else:
                    reference_value = self._array_value(reference_array, entry.get('source_index', index))
                    if reference_value is None:
                        reference_value = z_reference
                base_world = float(reference_value) + z_offset + stack_offset
                z_min = base_world
                z_max = base_world + height
                box_key = (float(width), float(depth), float(height))
                template = self._get_box_template(*box_key)
                body = template.copy(deep=True)
                body.translate((x_min, front_y_world, z_min), inplace=True)

            exit_at_front = not cardio
            exit_face_index: Optional[int] = None
            if box_key is not None:
                face_indices = self._box_face_cache.get(box_key)
                if face_indices is None and box_key in self._box_template_cache:
                    face_indices = self._compute_box_face_indices(self._box_template_cache[box_key])
                    self._box_face_cache[box_key] = face_indices
                if face_indices is not None:
                    front_idx, back_idx = face_indices
                    exit_face_index = front_idx if exit_at_front else back_idx

            geoms.append((body, exit_face_index))

        rotation_groups: List[List[int]] = []
        stack_indices = [idx for idx, entry in enumerate(processed_entries) if entry['configuration'] == 'stack']
        if stack_indices:
            rotation_groups.append(stack_indices)
        for indices in flown_groups:
            if indices:
                rotation_groups.append(indices)

        if rotation_groups and abs(azimuth_rad) > 1e-6:
            angle_deg = float(np.rad2deg(-azimuth_rad))
            for group_indices in rotation_groups:
                group_entries = []
                for idx in group_indices:
                    if idx < 0 or idx >= len(geoms):
                        continue
                    body_mesh, exit_face_index = geoms[idx]
                    group_entries.append((idx, processed_entries[idx], body_mesh, exit_face_index))
                if not group_entries:
                    continue

                x_mins = [mesh.bounds[0] for _, _, mesh, _ in group_entries]
                x_maxs = [mesh.bounds[1] for _, _, mesh, _ in group_entries]
                y_mins = [mesh.bounds[2] for _, _, mesh, _ in group_entries]
                y_maxs = [mesh.bounds[3] for _, _, mesh, _ in group_entries]
                center_x = 0.0
                if x_mins and x_maxs:
                    center_x = 0.5 * (min(x_mins) + max(x_maxs))
                center_y = 0.0
                if y_maxs:
                    center_y = max(y_maxs)

                first_config = group_entries[0][1].get('configuration', '')
                target_front_y: Optional[float] = None
                if first_config == 'flown':
                    top_entry = max(group_entries, key=lambda item: item[2].bounds[5])
                    top_mesh = top_entry[2]
                    top_bounds = top_mesh.bounds
                    top_z = top_bounds[5]
                    rotation_center = (center_x, center_y, top_z)
                    target_front_y = None
                else:
                    z_midpoints = [(mesh.bounds[4] + mesh.bounds[5]) / 2.0 for _, _, mesh, _ in group_entries]
                    z_center_group = np.mean(z_midpoints) if z_midpoints else z_reference
                    rotation_center = (center_x, center_y, z_center_group)
                    target_front_y = None

                # 🚀 OPTIMIERUNG: Batch-Transformationen für alle Meshes in der Gruppe
                # Sammle alle Delta-Y Werte vor der Transformation
                delta_ys = []
                if target_front_y is not None:
                    for idx, entry, mesh, exit_face_index in group_entries:
                        current_front_y = mesh.bounds[3]
                        delta_y = target_front_y - current_front_y
                        delta_ys.append((idx, delta_y))
                
                # Wende Rotationen und Translationen an
                for i, (idx, entry, mesh, exit_face_index) in enumerate(group_entries):
                    mesh.rotate_z(angle_deg, point=rotation_center, inplace=True)
                    if target_front_y is not None and i < len(delta_ys):
                        _, delta_y = delta_ys[i]
                        if abs(delta_y) > 1e-6:
                            mesh.translate((0.0, delta_y, 0.0), inplace=True)
                    geoms[idx] = (mesh, exit_face_index)

        return geoms

    def _apply_flown_cabinet_shape(
        self,
        mesh: Any,
        *,
        front_height: float,
        back_height: float,
        top_reference: float,
        angle_point: str,
    ):
        """Verformt ein würfelförmiges Gehäuse in eine Keilform anhand der Höhenmetadaten."""
        try:
            front_height = float(front_height)
            back_height = float(back_height)
            top_reference = float(top_reference)
        except Exception:  # noqa: BLE001
            return mesh

        if front_height <= 0.0 or back_height <= 0.0:
            return mesh
        if np.isclose(front_height, back_height, rtol=1e-4, atol=1e-3):
            return mesh
        if not hasattr(mesh, 'points'):
            return mesh

        angle_point_lower = (angle_point or 'front').strip().lower()
        if angle_point_lower not in {'front', 'back', 'center', 'none'}:
            angle_point_lower = 'front'

        height_diff = front_height - back_height
        delta = height_diff / 2.0

        if angle_point_lower == 'back':
            back_top = top_reference
            front_top = back_top + delta
        else:
            front_top = top_reference
            back_top = front_top - delta

        front_bottom = front_top - front_height
        back_bottom = back_top - back_height

        try:
            points = mesh.points.copy()
        except Exception:  # noqa: BLE001
            return mesh

        if points.size == 0:
            return mesh

        y_coords = points[:, 1]
        front_y = np.max(y_coords)
        back_y = np.min(y_coords)
        tol = max(abs(front_y - back_y), 1.0) * 1e-6 + 1e-9
        z_values = points[:, 2]
        z_mid = float((np.max(z_values) + np.min(z_values)) / 2.0)

        front_mask = np.isclose(y_coords, front_y, atol=tol)
        back_mask = np.isclose(y_coords, back_y, atol=tol)
        upper_mask = z_values >= z_mid
        lower_mask = ~upper_mask

        front_upper = front_mask & upper_mask
        front_lower = front_mask & lower_mask
        back_upper = back_mask & upper_mask
        back_lower = back_mask & lower_mask

        if np.any(front_upper):
            points[front_upper, 2] = front_top
        if np.any(front_lower):
            points[front_lower, 2] = front_bottom
        if np.any(back_upper):
            points[back_upper, 2] = back_top
        if np.any(back_lower):
            points[back_lower, 2] = back_bottom

        try:
            mesh.points = points
        except Exception:  # noqa: BLE001
            return mesh
        return mesh

    @staticmethod
    def _speaker_signature_from_mesh(mesh: Any, exit_face_index: Optional[int]) -> tuple:
        bounds = getattr(mesh, 'bounds', (0.0,) * 6)
        if bounds is None:
            bounds = (0.0,) * 6
        rounded_bounds = tuple(round(float(value), 4) for value in bounds)
        n_points = int(getattr(mesh, 'n_points', 0))
        n_cells = int(getattr(mesh, 'n_cells', 0))
        exit_idx = int(exit_face_index) if exit_face_index is not None else None
        point_sample: tuple = ()
        try:
            points = getattr(mesh, 'points', None)
            if points is not None:
                sample = points[: min(5, len(points))]
                point_sample = tuple(round(float(coord), 4) for pt in np.asarray(sample) for coord in pt)
        except Exception:
            point_sample = ()
        return (rounded_bounds, n_points, n_cells, exit_idx, point_sample)

    def _resolve_cabinet_entries(self, cabinet_raw, configuration) -> List[Dict]:
        config_lower = (configuration or '').lower()
        entries: List[Dict] = []

        def add_entry(entry):
            if not isinstance(entry, dict):
                return
            entry_config = str(entry.get('configuration', '')).lower() if entry.get('configuration') else ''
            if entry_config and config_lower and entry_config != config_lower:
                return
            entries.append(entry)

        if isinstance(cabinet_raw, dict):
            add_entry(cabinet_raw)
        elif isinstance(cabinet_raw, (list, tuple)):
            for item in cabinet_raw:
                if isinstance(item, np.ndarray):
                    item = item.tolist()
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        add_entry(sub_item)
                else:
                    add_entry(item)
        elif isinstance(cabinet_raw, np.ndarray):
            if cabinet_raw.ndim == 0:
                entries.extend(self._resolve_cabinet_entries(cabinet_raw.item(), configuration))
            else:
                entries.extend(self._resolve_cabinet_entries(cabinet_raw.tolist(), configuration))

        if not entries and isinstance(cabinet_raw, (list, tuple)):
            for item in cabinet_raw:
                if isinstance(item, dict):
                    entries.append(item)

        return entries

    def _get_box_template(self, width: float, depth: float, height: float) -> Any:
        key = (float(width), float(depth), float(height))
        template = self._box_template_cache.get(key)
        if template is None:
            template = self.pv.Box(bounds=(0.0, width, -depth, 0.0, 0.0, height))
            self._box_template_cache[key] = template
            self._box_face_cache[key] = self._compute_box_face_indices(template)
        elif key not in self._box_face_cache:
            self._box_face_cache[key] = self._compute_box_face_indices(template)
        return template

    @staticmethod
    def _compute_box_face_indices(mesh: Any) -> Tuple[Optional[int], Optional[int]]:
        try:
            faces = mesh.faces.reshape(-1, 5)
        except Exception:  # noqa: BLE001
            return (None, None)
        try:
            points = mesh.points
        except Exception:  # noqa: BLE001
            return (None, None)
        if faces.size == 0 or points.size == 0:
            return (None, None)
        face_y_values: List[float] = []
        for face in faces:
            indices = face[1:]
            try:
                y_vals = points[indices, 1]
            except Exception:  # noqa: BLE001
                return (None, None)
            face_y_values.append(float(np.mean(y_vals)))
        if not face_y_values:
            return (None, None)
        front_idx = int(np.argmax(face_y_values))
        back_idx = int(np.argmin(face_y_values))
        return (front_idx, back_idx)

    @staticmethod
    def _get_speaker_name(speaker_array, index: int):
        if hasattr(speaker_array, 'source_polar_pattern'):
            pattern = speaker_array.source_polar_pattern
            try:
                if index < len(pattern):
                    value = pattern[index]
                    if isinstance(value, bytes):
                        return value.decode('utf-8', errors='ignore')
                    return str(value)
            except Exception:  # noqa: BLE001
                pass

        if hasattr(speaker_array, 'source_type'):
            try:
                if index < len(speaker_array.source_type):
                    return str(speaker_array.source_type[index])
            except Exception:  # noqa: BLE001
                pass

        return None

    @classmethod
    def _safe_float(cls, mapping, key: str, default=None) -> float:
        if isinstance(mapping, (float, int)):
            return float(mapping)
        if isinstance(mapping, str):
            try:
                return float(mapping)
            except Exception as e:
                raise ValueError(f"_safe_float: Konvertierung von String '{mapping}' zu float fehlgeschlagen: {e}")
        if not isinstance(mapping, dict):
            raise ValueError(f"_safe_float: mapping ist kein dict, sondern {type(mapping)}")
        if key not in mapping:
            if default is not None:
                return float(default)
            raise ValueError(f"_safe_float: Key '{key}' nicht in mapping gefunden")
        value = mapping.get(key)
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"_safe_float: Konvertierung von Wert '{value}' (Key '{key}') zu float fehlgeschlagen: {e}")

    @staticmethod
    def _array_value(values, index: int) -> float | None:
        if values is None:
            return None
        try:
            if len(values) <= index:
                return None
            value = values[index]
            if isinstance(value, (list, tuple)):
                return float(value[0])
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _is_numeric(value) -> bool:
        if isinstance(value, (float, int)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except Exception:  # noqa: BLE001
                return False
        return False

    @staticmethod
    def _float_sequence(values, target_length: int, default: float = 0.0) -> List[float]:
        if target_length <= 0:
            return []
        if values is None:
            return [default] * target_length
        if isinstance(values, (float, int)):
            return [float(values)] * target_length
        result: List[float] = []
        try:
            iterable = list(values)
        except TypeError:
            iterable = [values]
        for idx in range(target_length):
            if idx < len(iterable):
                try:
                    result.append(float(iterable[idx]))
                except Exception:  # noqa: BLE001
                    result.append(default)
            else:
                result.append(default)
        return result

    @staticmethod
    def _sequence_length(values) -> int:
        if values is None:
            return 0
        try:
            return len(values)
        except TypeError:
            return 0

    @staticmethod
    def _to_float_array(values) -> np.ndarray:
        if values is None:
            return np.empty(0, dtype=float)
        try:
            array = np.asarray(values, dtype=float)
        except Exception:
            return np.empty(0, dtype=float)
        if array.ndim > 1:
            array = array.reshape(-1)
        return array.astype(float)











__all__ = ['SPL3DSpeakerMixin']