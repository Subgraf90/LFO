import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from Module_LFO.Modules_Calculate.SurfaceGeometryCalculator import SurfaceDefinition

"SoundfieldCalculatorXaxis berechnet die SPL-Werte in Abhängigkeit der Breite des Schallfeldes"

class SoundFieldCalculatorXaxis(ModuleBase):
    def __init__(self, settings, data, calculation_spl):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_spl = calculation_spl
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt


    def _get_active_xy_surfaces(self):
        """Sammelt alle aktiven Surfaces für XY-Berechnung (xy_enabled=True, enabled=True, hidden=False)"""
        active_surfaces = []
        surface_store = getattr(self.settings, 'surface_definitions', {})
        
        
        for surface_id, surface in surface_store.items():
            # Prüfe ob Surface aktiv ist
            if isinstance(surface, SurfaceDefinition):
                xy_enabled = getattr(surface, 'xy_enabled', True)
                enabled = surface.enabled
                hidden = surface.hidden
                name = surface.name
            else:
                xy_enabled = surface.get('xy_enabled', True)
                enabled = surface.get('enabled', False)
                hidden = surface.get('hidden', False)
                name = surface.get('name', surface_id)
            
            
            if xy_enabled and enabled and not hidden:
                active_surfaces.append((surface_id, surface))
            else:
                reason = []
                if not xy_enabled:
                    reason.append("xy_disabled")
                if not enabled:
                    reason.append("disabled")
                if hidden:
                    reason.append("hidden")
        
        return active_surfaces
    
    def _line_intersects_surface_xz(self, y_const, surface):
        """
        Prüft, ob eine Linie y=const die Surface schneidet (Projektion auf XZ-Ebene).
        Für X-Axis: Linie ist y=position_y_axis (konstant)
        
        Args:
            y_const: Konstante Y-Koordinate der Linie
            surface: SurfaceDefinition oder Dict mit Surface-Daten
            
        Returns:
            bool: True wenn Linie die Surface schneidet
        """
        points = surface.points if isinstance(surface, SurfaceDefinition) else surface.get('points', [])
        if len(points) < 3:
            return False
        
        surface_name = surface.name if isinstance(surface, SurfaceDefinition) else surface.get('name', 'unknown')
        
        # Extrahiere Y-Koordinaten
        y_coords = [p.get('y', 0.0) for p in points]
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Prüfe ob y_const im Y-Bereich der Surface liegt
        intersects = y_min <= y_const <= y_max
        
        return intersects
    
    def _get_surface_intersection_points_xz(self, y_const, surface):
        """
        Gibt die X- und Z-Koordinaten der Surface-Punkte zurück, wenn Linie y=const die Surface schneidet.
        Für X-Axis: Verwende x- und z-Koordinaten der Surface-Punkte.
        
        WICHTIG: Holt die aktuellen Punkte direkt aus settings.surface_definitions, um sicherzustellen,
        dass immer die neuesten Daten verwendet werden (auch nach Aktivierung/Deaktivierung).
        
        Args:
            y_const: Konstante Y-Koordinate der Linie
            surface: SurfaceDefinition oder Dict mit Surface-Daten (kann veraltet sein)
            
        Returns:
            tuple: (x_coords, z_coords) als Arrays oder (None, None) wenn keine Schnittpunkte
        """
        # Prüfung wird bereits vor dem Aufruf durchgeführt, daher hier keine erneute Prüfung
        
        # Hole aktuelle Surface-Daten direkt aus settings, um sicherzustellen, dass wir
        # immer die neuesten Punkte verwenden (auch nach Aktivierung/Deaktivierung)
        surface_id = None
        if isinstance(surface, SurfaceDefinition):
            # Versuche surface_id zu finden
            surface_store = getattr(self.settings, 'surface_definitions', {})
            for sid, sdef in surface_store.items():
                if sdef is surface or (hasattr(sdef, 'name') and hasattr(surface, 'name') and sdef.name == surface.name):
                    surface_id = sid
                    break
        elif isinstance(surface, dict):
            # Versuche surface_id aus dem Dict zu extrahieren oder zu finden
            surface_store = getattr(self.settings, 'surface_definitions', {})
            surface_name = surface.get('name', '')
            for sid, sdef in surface_store.items():
                if isinstance(sdef, SurfaceDefinition):
                    if sdef.name == surface_name:
                        surface_id = sid
                        break
                elif isinstance(sdef, dict) and sdef.get('name') == surface_name:
                    surface_id = sid
                    break
        
        # Hole aktuelle Punkte aus settings und prüfe ob Surface noch aktiv ist
        current_points = None
        if surface_id and hasattr(self.settings, 'surface_definitions'):
            current_surface = self.settings.surface_definitions.get(surface_id)
            if current_surface:
                # Prüfe ob Surface noch aktiv ist (xy_enabled, enabled, hidden)
                if isinstance(current_surface, SurfaceDefinition):
                    xy_enabled = getattr(current_surface, 'xy_enabled', True)
                    enabled = current_surface.enabled
                    hidden = current_surface.hidden
                    if xy_enabled and enabled and not hidden:
                        current_points = current_surface.points
                    else:
                        return None, None
                elif isinstance(current_surface, dict):
                    xy_enabled = current_surface.get('xy_enabled', True)
                    enabled = current_surface.get('enabled', False)
                    hidden = current_surface.get('hidden', False)
                    if xy_enabled and enabled and not hidden:
                        current_points = current_surface.get('points', [])
                    else:
                        return None, None
        
        # Fallback: Verwende übergebene Surface-Daten (nur wenn keine aktuelle Surface gefunden wurde)
        if current_points is None:
            # Prüfe auch hier nochmal, ob die übergebene Surface aktiv ist
            if isinstance(surface, SurfaceDefinition):
                xy_enabled = getattr(surface, 'xy_enabled', True)
                enabled = surface.enabled
                hidden = surface.hidden
                if not (xy_enabled and enabled and not hidden):
                    return None, None
                current_points = surface.points
            else:
                xy_enabled = surface.get('xy_enabled', True)
                enabled = surface.get('enabled', False)
                hidden = surface.get('hidden', False)
                if not (xy_enabled and enabled and not hidden):
                    return None, None
                current_points = surface.get('points', [])
        
        if len(current_points) < 3:
            return None, None
        
        # Berechne tatsächliche Schnittpunkte der Linie y=y_const mit dem Surface-Polygon
        # Projektion auf XZ-Ebene: Linie y=y_const schneidet Polygon-Kanten
        intersection_points = []
        
        # Extrahiere alle Koordinaten
        points_3d = []
        for p in current_points:
            x = p.get('x', 0.0)
            y = p.get('y', 0.0)
            z = p.get('z', 0.0) if p.get('z') is not None else 0.0
            points_3d.append((x, y, z))
        
        # Prüfe jede Kante des Polygons auf Schnitt mit Linie y=y_const
        n = len(points_3d)
        for i in range(n):
            p1 = points_3d[i]
            p2 = points_3d[(i + 1) % n]  # Nächster Punkt (geschlossenes Polygon)
            
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            
            # Prüfe ob Kante die Linie y=y_const schneidet
            # Kante verläuft von (x1, y1, z1) nach (x2, y2, z2)
            # Linie ist y=y_const (konstant)
            
            # Prüfe ob y_const zwischen y1 und y2 liegt
            if (y1 <= y_const <= y2) or (y2 <= y_const <= y1):
                if abs(y2 - y1) > 1e-10:  # Vermeide Division durch Null
                    # Berechne Schnittpunkt-Parameter t
                    t = (y_const - y1) / (y2 - y1)
                    
                    # Berechne X- und Z-Koordinaten des Schnittpunkts
                    x_intersect = x1 + t * (x2 - x1)
                    z_intersect = z1 + t * (z2 - z1)
                    
                    intersection_points.append((x_intersect, z_intersect))
        
        if len(intersection_points) < 2:
            # Nicht genug Schnittpunkte gefunden
            return None, None
        
        # Entferne Duplikate (Punkte die sehr nah beieinander sind)
        unique_points = []
        eps = 1e-6
        for x, z in intersection_points:
            is_duplicate = False
            for ux, uz in unique_points:
                if abs(x - ux) < eps and abs(z - uz) < eps:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append((x, z))
        
        if len(unique_points) < 2:
            return None, None
        
        # Extrahiere X- und Z-Koordinaten
        x_coords = np.array([p[0] for p in unique_points])
        z_coords = np.array([p[1] for p in unique_points])
        
        # Sortiere nach X-Koordinaten für konsistente Reihenfolge
        sort_indices = np.argsort(x_coords)
        x_coords = x_coords[sort_indices]
        z_coords = z_coords[sort_indices]
        
        
        return x_coords, z_coords
    
    def _interpolate_surface_points(self, coords, z_coords, target_resolution=0.1):
        """
        Interpoliert Surface-Punkte auf eine höhere Auflösung (10cm Schritte).
        WICHTIG: Start- und Endpunkt werden exakt erreicht, damit die gesamte Länge abgedeckt wird.
        
        Args:
            coords: Array von X- oder Y-Koordinaten
            z_coords: Array von Z-Koordinaten
            target_resolution: Ziel-Auflösung in Metern (Standard: 0.1 = 10cm)
            
        Returns:
            tuple: (interpolated_coords, interpolated_z_coords) als Arrays
        """
        if len(coords) < 2:
            return coords, z_coords
        
        # Bestimme exakte Start- und Endpunkte
        min_coord = coords.min()
        max_coord = coords.max()
        length = max_coord - min_coord
        
        # Berechne Anzahl der Schritte, damit Endpunkt exakt erreicht wird
        # Anzahl = Länge / Resolution, aufgerundet, +1 für Startpunkt
        num_steps = int(np.ceil(length / target_resolution))
        
        # Erstelle Koordinaten mit exaktem Start- und Endpunkt
        # Verwende linspace, um sicherzustellen, dass min_coord und max_coord exakt erreicht werden
        interpolated_coords = np.linspace(min_coord, max_coord, num_steps + 1)
        
        # Interpoliere Z-Koordinaten linear
        interpolated_z_coords = np.interp(interpolated_coords, coords, z_coords)
        
        return interpolated_coords, interpolated_z_coords
    
    def calculateXAxis(self):
        resolution = 0.1  # 10cm Auflösung
        position_y = self.settings.position_y_axis
        
        
        # Prüfe ob aktive Surfaces vorhanden sind, die die Linie y=position_y schneiden
        active_surfaces = self._get_active_xy_surfaces()
        all_x_coords = []
        all_z_coords = []
        used_surfaces = []
        
        for surface_id, surface in active_surfaces:
            # Prüfe nochmal, ob Surface noch aktiv ist (könnte sich während der Berechnung geändert haben)
            surface_store = getattr(self.settings, 'surface_definitions', {})
            current_surface = surface_store.get(surface_id)
            if current_surface:
                if isinstance(current_surface, SurfaceDefinition):
                    xy_enabled = getattr(current_surface, 'xy_enabled', True)
                    enabled = current_surface.enabled
                    hidden = current_surface.hidden
                else:
                    xy_enabled = current_surface.get('xy_enabled', True)
                    enabled = current_surface.get('enabled', False)
                    hidden = current_surface.get('hidden', False)
                
                if not (xy_enabled and enabled and not hidden):
                    surface_name = surface.name if isinstance(surface, SurfaceDefinition) else surface.get('name', surface_id)
                    continue
            
            surface_name = surface.name if isinstance(surface, SurfaceDefinition) else surface.get('name', surface_id)
            if self._line_intersects_surface_xz(position_y, surface):
                x_coords, z_coords = self._get_surface_intersection_points_xz(position_y, surface)
                if x_coords is not None and z_coords is not None:
                    # Sammle alle Punkte von allen geschnittenen Surfaces
                    all_x_coords.append(x_coords)
                    all_z_coords.append(z_coords)
                    used_surfaces.append((surface_id, surface_name, len(x_coords)))
        
        # Behalte Segmente getrennt für separate Berechnung
        # WICHTIG: Jede Surface bleibt als separates Segment, damit deaktivierte Surfaces als Lücken sichtbar sind
        if all_x_coords:
            # Sortiere Surfaces nach X-Position (min X-Koordinate)
            surface_segments = list(zip(all_x_coords, all_z_coords, used_surfaces))
            surface_segments.sort(key=lambda seg: seg[0].min())  # Sortiere nach min X-Koordinate
            
            for i, (x_coords, z_coords, (surface_id, surface_name, _)) in enumerate(surface_segments):
                pass
        else:
            surface_segments = []
        
        # Wenn Surface-Punkte gefunden wurden, verwende diese
        if surface_segments:
            # Berechne SPL pro Segment separat
            all_interpolated_x = []
            all_sound_field_p = []
            
            for i, (x_coords, z_coords, (surface_id, surface_name, _)) in enumerate(surface_segments):
                # Sortiere Punkte innerhalb des Segments nach X
                sort_indices = np.argsort(x_coords)
                x_sorted = x_coords[sort_indices]
                z_sorted = z_coords[sort_indices]
                
                # Interpoliere Segment auf höhere Auflösung (10cm)
                interpolated_x, interpolated_z = self._interpolate_surface_points(
                    x_sorted, z_sorted, target_resolution=resolution
                )
                
                # Berechne SPL für Segment
                sound_field_p_segment = self.calculate_sound_field_at_points_xz(
                    interpolated_x, position_y, interpolated_z
                )
                
                # Füge Segment hinzu
                all_interpolated_x.append(interpolated_x)
                all_sound_field_p.append(sound_field_p_segment)
                
                # Füge NaN-Trenner zwischen Segmenten hinzu (außer beim letzten)
                if i < len(surface_segments) - 1:
                    all_interpolated_x.append(np.array([np.nan]))
                    all_sound_field_p.append(np.array([np.nan]))
            
            # Kombiniere alle Segmente
            sound_field_x_xaxis_calc = np.concatenate(all_interpolated_x)
            sound_field_p = np.concatenate(all_sound_field_p)
            
        else:
            # Standard-Verhalten: Verwende feste Breite
            total_rows = int(self.settings.length / resolution)
            row_index = int((position_y / resolution) + (total_rows / 2))
            row_index = max(0, min(row_index, total_rows - 1))
            
            sound_field_p = self.calculate_sound_field_row(row_index)[0].flatten()
            sound_field_x_xaxis_calc = np.arange((self.settings.width / 2 * -1), ((self.settings.width / 2) + resolution), resolution)
    
        sound_field_p_calc = self.functions.mag2db(sound_field_p)

        # Prüfen, ob es aktive Quellen gibt oder ob das Ergebnis verwertbar ist
        has_active_sources = any(not (arr.mute or arr.hide) for arr in self.settings.speaker_arrays.values())
        # Wenn keine aktive Quelle beteiligt war, bleibt das Feld numerisch 0 (vor dB-Umrechnung -inf)
        # Wir prüfen konservativ auf eine sehr kleine Varianz nach dB-Umrechnung
        is_meaningful_curve = np.isfinite(sound_field_p_calc).any()

        show_curve = has_active_sources and is_meaningful_curve

        # Finde Segment-Grenzen (Anfang und Ende jedes Segments) für gestrichelte Linien
        segment_boundaries = []
        # Prüfe ob surface_segments existiert und nicht leer ist
        if 'surface_segments' in locals() and len(surface_segments) > 0 and len(sound_field_x_xaxis_calc) > 0:
            # Finde NaN-Positionen in den X-Koordinaten
            nan_mask = np.isnan(sound_field_x_xaxis_calc)
            nan_indices = np.where(nan_mask)[0]
            
            # Sammle alle Segment-Start- und End-Positionen
            segment_starts = []
            segment_ends = []
            
            # Erste Segment: Start-Position
            if len(sound_field_x_xaxis_calc) > 0 and not np.isnan(sound_field_x_xaxis_calc[0]):
                segment_starts.append(float(sound_field_x_xaxis_calc[0]))
            
            # Für jede NaN-Position: Ende des vorherigen Segments und Start des nächsten Segments
            for nan_idx in nan_indices:
                if nan_idx > 0:
                    # Ende des Segments vor dem NaN
                    prev_x = sound_field_x_xaxis_calc[nan_idx - 1]
                    if not np.isnan(prev_x):
                        segment_ends.append(float(prev_x))
                
                if nan_idx < len(sound_field_x_xaxis_calc) - 1:
                    # Start des Segments nach dem NaN
                    next_x = sound_field_x_xaxis_calc[nan_idx + 1]
                    if not np.isnan(next_x):
                        segment_starts.append(float(next_x))
            
            # Letztes Segment: End-Position
            if len(sound_field_x_xaxis_calc) > 0:
                last_idx = len(sound_field_x_xaxis_calc) - 1
                last_x = sound_field_x_xaxis_calc[last_idx]
                if not np.isnan(last_x):
                    segment_ends.append(float(last_x))
            
            # Kombiniere alle Grenzen (Start und Ende jedes Segments)
            segment_boundaries = sorted(set(segment_starts + segment_ends))
        
        # Konvertiere zu Python-Liste (falls numpy array)
        segment_boundaries_list = [float(x) for x in segment_boundaries]
        
        # Daten aus calculation werden zurückgegeben
        self.calculation_spl["aktuelle_simulation"] = {
            "x_data_xaxis": sound_field_p_calc,
            "y_data_xaxis": sound_field_x_xaxis_calc,
            "show_in_plot": bool(show_curve),
            "color": "#6A5ACD",
            "segment_boundaries_xaxis": segment_boundaries_list  # X-Positionen für vertikale Linien als Python-Liste
        }
      
    def get_balloon_data_batch(self, speaker_name, azimuths, elevations, use_averaged=True):
        """
        🚀 BATCH-OPTIMIERT: Holt Balloon-Daten für VIELE Winkel auf einmal
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuths (np.ndarray): Array von Azimut-Winkeln in Grad
            elevations (np.ndarray): Array von Elevations-Winkeln in Grad
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten
            
        Returns:
            tuple: (magnitudes, phases) als Arrays oder (None, None) bei Fehler
        """
        if use_averaged:
            balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=True)
            if balloon_data is not None:
                magnitude = balloon_data.get('magnitude')
                phase = balloon_data.get('phase')
                vertical_angles = balloon_data.get('vertical_angles')
                
                if magnitude is not None and phase is not None and vertical_angles is not None:
                    return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)
        
        balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
        if balloon_data is None:
            return None, None
        
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            return None, None
        
        return self._interpolate_angle_data_batch(magnitude, phase, vertical_angles, azimuths, elevations)

    def get_balloon_data_at_angle(self, speaker_name, azimuth, elevation=0, use_averaged=True):
        """
        Holt die Balloon-Daten (Magnitude und Phase) für einen bestimmten Winkel
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuth (float): Azimut-Winkel in Grad (0-360)
            elevation (float, optional): Elevations-Winkel in Grad. Standard ist 0 (horizontale Ebene).
            use_averaged (bool): Wenn True, verwende bandgemittelte Daten (Standard: True)
            
        Returns:
            tuple: (magnitude, phase) oder (None, None) wenn keine Daten gefunden wurden
        """
        # 🚀 OPTIMIERT: Verwende bandgemittelte Daten falls verfügbar
        if use_averaged:
            balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=True)
            if balloon_data is not None:
                # 🎯 BANDGEMITTELT: Verwende bereits gemittelte Daten
                magnitude = balloon_data.get('magnitude')
                phase = balloon_data.get('phase')
                vertical_angles = balloon_data.get('vertical_angles')
                
                if magnitude is not None and phase is not None and vertical_angles is not None:
                    # Bandgemittelte Daten haben keine Frequenzdimension mehr
                    return self._interpolate_angle_data(magnitude, phase, vertical_angles, azimuth, elevation)
        
        # 🔄 FALLBACK: Verwende originale Daten
        balloon_data = self._data_container.get_balloon_data(speaker_name, use_averaged=False)
        if balloon_data is None:
            print(f"❌ Keine Balloon-Daten für {speaker_name}")
            return None, None
        
        # Direkte Nutzung der optimierten Balloon-Daten
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        vertical_angles = balloon_data.get('vertical_angles')
        
        if magnitude is None or phase is None or vertical_angles is None:
            print(f"Fehlende Daten in Balloon-Daten für {speaker_name}")
            return None, None
        
        # Verwende die gleiche Interpolationslogik für beide Datentypen
        return self._interpolate_angle_data(magnitude, phase, vertical_angles, azimuth, elevation)

    def calculate_sound_field_at_points_xz(self, x_coords, y_const, z_coords):
        """
        Berechnet SPL-Werte für spezifische Punkte (X- und Z-Koordinaten bei konstanter Y-Position).
        Für X-Axis: Berechnet SPL entlang der Surface-Punkte.
        
        Args:
            x_coords: Array von X-Koordinaten
            y_const: Konstante Y-Koordinate
            z_coords: Array von Z-Koordinaten
            
        Returns:
            np.ndarray: SPL-Werte für die gegebenen Punkte
        """
        x_coords = np.asarray(x_coords, dtype=float)
        z_coords = np.asarray(z_coords, dtype=float)
        n_points = len(x_coords)
        
        # Initialisiere Schallfeld als komplexes Array
        sound_field_p = np.zeros(n_points, dtype=complex)
        calculate_frequency = self.settings.calculate_frequency
        speed_of_sound = self.settings.speed_of_sound
        wave_number = self.functions.wavenumber(speed_of_sound, calculate_frequency)
        a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
        use_air_absorption = getattr(self.settings, "use_air_absorption", False)
        air_absorption_coeff = 0.0
        if use_air_absorption:
            air_absorption_coeff = self.functions.calculate_air_absorption(
                calculate_frequency,
                getattr(self.settings, "temperature", 20.0),
                getattr(self.settings, "humidity", 50.0),
            )
        
        # Iteriere über alle Lautsprecher-Arrays
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            # Konvertiere Lautsprechernamen in Indizes
            speaker_indices = []
            for speaker_name in speaker_array.source_polar_pattern:
                try:
                    speaker_names = self._data_container.get_speaker_names()
                    index = speaker_names.index(speaker_name)
                    speaker_indices.append(index)
                except ValueError:
                    print(f"Warnung: Lautsprecher {speaker_name} nicht gefunden")
                    speaker_indices.append(0)
            
            source_indices = np.array(speaker_indices)
            source_position_x = getattr(
                speaker_array,
                'source_position_calc_x',
                speaker_array.source_position_x,
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                speaker_array.source_position_y,
            )
            
            source_position_z = getattr(
                speaker_array,
                'source_position_calc_z',
                getattr(speaker_array, 'source_position_z', None)
            )
            
            if source_position_z is None:
                source_position_z = np.zeros(len(speaker_array.source_polar_pattern), dtype=float)
            elif not isinstance(source_position_z, (np.ndarray, list)):
                source_position_z = np.full(len(speaker_array.source_polar_pattern), float(source_position_z), dtype=float)
            else:
                source_position_z = np.asarray(source_position_z, dtype=float)
                if len(source_position_z) < len(speaker_array.source_polar_pattern):
                    padding = np.zeros(len(speaker_array.source_polar_pattern) - len(source_position_z), dtype=float)
                    source_position_z = np.concatenate([source_position_z, padding])
            
            source_azimuth_deg = speaker_array.source_azimuth
            source_azimuth = np.deg2rad(source_azimuth_deg)
            source_delay = speaker_array.delay
            
            if isinstance(speaker_array.source_time, (int, float)):
                source_time = [speaker_array.source_time + source_delay]
            else:
                source_time = [time + source_delay for time in speaker_array.source_time]
            source_time = [x / 1000 for x in source_time]
            source_gain = speaker_array.gain
            source_level = speaker_array.source_level + source_gain
            source_level = self.functions.db2mag(np.array(source_level))
            
            # Iteriere über alle Lautsprecher im Array
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                
                # Berechne Distanzen für alle Punkte
                x_distance = x_coords - source_position_x[isrc]
                y_distance = y_const - source_position_y[isrc]
                z_distance = z_coords - source_position_z[isrc]
                
                # 3D-Distanz
                source_dist = np.sqrt(x_distance**2 + y_distance**2 + z_distance**2)
                
                # Filtere Punkte zu nah an der Quelle
                valid_mask = source_dist >= 0.001
                
                # Initialisiere Wellen-Array
                wave = np.zeros(n_points, dtype=complex)
                
                # Berechne Azimut und Elevation
                source_to_point_angles = np.arctan2(y_distance, x_distance)
                azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                azimuths = (360 - azimuths) % 360
                azimuths = (azimuths + 90) % 360
                horizontal_dist = np.sqrt(x_distance**2 + y_distance**2)
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dist))
                
                # Batch-Interpolation
                polar_gains, polar_phases = self.get_balloon_data_batch(speaker_name, azimuths, elevations)
                
                if polar_gains is not None and polar_phases is not None:
                    magnitude_linear = 10 ** (polar_gains / 20)
                    polar_phase_rad = np.radians(polar_phases)
                    
                    attenuation = 1.0
                    if air_absorption_coeff > 0.0:
                        attenuation = np.exp(-air_absorption_coeff * source_dist[valid_mask])
                    wave[valid_mask] = (magnitude_linear[valid_mask] * source_level[isrc] * a_source_pa *
                                       attenuation *
                                       np.exp(1j * (wave_number * source_dist[valid_mask] +  
                                                  polar_phase_rad[valid_mask] +     
                                                  2 * np.pi * calculate_frequency * source_time[isrc])) / 
                                       source_dist[valid_mask])
                    
                    if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                        wave = -wave
                
                # Akkumuliere Wellen
                sound_field_p += wave
        
        # Berechne Absolutwert
        sound_field_p = abs(sound_field_p)
        return sound_field_p
    
    def calculate_sound_field_row(self, row_index):
        """
        🚀 VEKTORISIERT: Berechnet SPL-Werte entlang der X-Achse (Breite)
        
        Berechnet eine einzelne Zeile des Schallfelds (konstante Y-Position).
        Nutzt Batch-Interpolation für optimale Performance.
        
        Args:
            row_index: Index der Y-Position im Grid
            
        Returns:
            tuple: (sound_field_p, sound_field_x, sound_field_y)
        """
        # ============================================================
        # SCHRITT 1: Grid-Erstellung (1D - nur X-Achse)
        # ============================================================
        width = self.settings.width
        length = self.settings.length
        resolution = 0.1  # 10cm Auflösung

        # Erstelle 1D-Arrays für X- und Y-Koordinaten
        sound_field_y = np.arange((width / 2 * -1), ((width / 2) + resolution), resolution)
        sound_field_x = np.arange((length / 2 * -1), ((length / 2) + resolution), resolution)
    
        # Erstelle Koordinatenraster nur für eine Zeile (konstante Y-Position)
        sound_field_x, sound_field_y = np.meshgrid(sound_field_y, np.array([sound_field_x[row_index]]))
    
        # Initialisiere Schallfeld als komplexes Array
        sound_field_p = np.zeros_like(sound_field_y, dtype=complex)
        calculate_frequency = self.settings.calculate_frequency
        speed_of_sound = self.settings.speed_of_sound
        wave_number = self.functions.wavenumber(speed_of_sound, calculate_frequency)
        a_source_pa = self.functions.db2spl(self.functions.db2mag(self.settings.a_source_db))
        use_air_absorption = getattr(self.settings, "use_air_absorption", False)
        air_absorption_coeff = 0.0
        if use_air_absorption:
            air_absorption_coeff = self.functions.calculate_air_absorption(
                calculate_frequency,
                getattr(self.settings, "temperature", 20.0),
                getattr(self.settings, "humidity", 50.0),
            )
    
        # ============================================================
        # SCHRITT 2: Iteriere über alle Lautsprecher-Arrays
        # ============================================================
        for speaker_array in self.settings.speaker_arrays.values():
            if speaker_array.mute or speaker_array.hide:
                continue
            
            # Konvertiere Lautsprechernamen in Indizes
            speaker_indices = []
            for speaker_name in speaker_array.source_polar_pattern:
                try:
                    speaker_names = self._data_container.get_speaker_names()
                    index = speaker_names.index(speaker_name)
                    speaker_indices.append(index)
                except ValueError:
                    print(f"Warnung: Lautsprecher {speaker_name} nicht gefunden")
                    speaker_indices.append(0)
            
            source_indices = np.array(speaker_indices)
            source_position_x = getattr(
                speaker_array,
                'source_position_calc_x',
                speaker_array.source_position_x,
            )
            source_position_y = getattr(
                speaker_array,
                'source_position_calc_y',
                speaker_array.source_position_y,
            )
            
            # Prüfe, ob Z-Position vorhanden ist (mit Fallback auf source_position_z)
            source_position_z = getattr(
                speaker_array,
                'source_position_calc_z',
                getattr(speaker_array, 'source_position_z', None)
            )
            
            # Wenn source_position_z None ist oder kein Array, erstelle ein Array mit Standardwerten
            if source_position_z is None:
                # Erstelle Array mit Standardwert 0.0 für alle Quellen
                source_position_z = np.zeros(len(speaker_array.source_polar_pattern), dtype=float)
            elif not isinstance(source_position_z, (np.ndarray, list)):
                # Wenn es ein Skalar ist, konvertiere zu Array
                source_position_z = np.full(len(speaker_array.source_polar_pattern), float(source_position_z), dtype=float)
            else:
                # Stelle sicher, dass es ein numpy Array ist
                source_position_z = np.asarray(source_position_z, dtype=float)
                # Stelle sicher, dass es die richtige Länge hat
                if len(source_position_z) < len(speaker_array.source_polar_pattern):
                    # Erweitere Array mit Nullen, falls zu kurz
                    padding = np.zeros(len(speaker_array.source_polar_pattern) - len(source_position_z), dtype=float)
                    source_position_z = np.concatenate([source_position_z, padding])
            
            source_azimuth_deg = speaker_array.source_azimuth
            source_azimuth = np.deg2rad(source_azimuth_deg)
            source_delay = speaker_array.delay
            
            # Stelle sicher, dass source_time ein Array/Liste ist
            if isinstance(speaker_array.source_time, (int, float)):
                # Einzelner Wert - konvertiere zu Liste
                source_time = [speaker_array.source_time + source_delay]
            else:
                source_time = [time + source_delay for time in speaker_array.source_time]
            source_time = [x / 1000 for x in source_time]
            source_gain = speaker_array.gain
            source_level = speaker_array.source_level + source_gain
            source_level = self.functions.db2mag(np.array(source_level))

            # ============================================================
            # SCHRITT 3: Iteriere über alle Lautsprecher im Array
            # ============================================================
            for isrc in range(len(source_indices)):
                speaker_name = speaker_array.source_polar_pattern[isrc]
                
                # --------------------------------------------------------
                # 3.1: VEKTORISIERTE GEOMETRIE-BERECHNUNG
                # --------------------------------------------------------
                # Berechne Distanzen für ALLE Punkte der X-Achse gleichzeitig
                x_distance = sound_field_x - source_position_x[isrc]
                y_distance = sound_field_y - source_position_y[isrc]
                
                # Horizontale Distanz (Pythagoras in 2D)
                # √(x² + y²) für ALLE Punkte gleichzeitig
                horizontal_dist = np.sqrt(x_distance**2 + y_distance**2)
                
                # Z-Distanz (konstant für alle Punkte, da Grid auf Z=0)
                z_distance = -source_position_z[isrc]
                
                # 3D-Distanz (Pythagoras in 3D)
                # √(horizontal² + z²) für ALLE Punkte gleichzeitig
                source_dist = np.sqrt(horizontal_dist**2 + z_distance**2)
                
                # --------------------------------------------------------
                # 3.2: MASKEN-LOGIK (Filtert ungültige Punkte)
                # --------------------------------------------------------
                # Filtere Punkte zu nah an der Quelle (verhindert Division durch Null)
                valid_mask = source_dist >= 0.001
                
                # Initialisiere Wellen-Array mit Nullen
                wave = np.zeros_like(source_dist, dtype=complex)
                
                # --------------------------------------------------------
                # 3.3: VEKTORISIERTE WINKEL-BERECHNUNG
                # --------------------------------------------------------
                # Berechne Azimut und Elevation für ALLE Punkte gleichzeitig
                source_to_point_angles = np.arctan2(y_distance, x_distance)
                azimuths = (np.degrees(source_to_point_angles) + np.degrees(source_azimuth[isrc])) % 360
                azimuths = (360 - azimuths) % 360  # Invertiere (Uhrzeigersinn)
                azimuths = (azimuths + 90) % 360  # Drehe 90° (Polar-Koordinatensystem)
                elevations = np.degrees(np.arctan2(z_distance, horizontal_dist))
                
                # --------------------------------------------------------
                # 3.4: BATCH-INTERPOLATION (Performance-Booster! 🚀)
                # --------------------------------------------------------
                # Hole Balloon-Daten für ALLE Punkte in EINEM Aufruf
                # → Alle Interpolationen gleichzeitig statt in einer Loop!
                polar_gains, polar_phases = self.get_balloon_data_batch(speaker_name, azimuths, elevations)
                
                if polar_gains is not None and polar_phases is not None:
                    # --------------------------------------------------------
                    # 3.5: VEKTORISIERTE WELLENBERECHNUNG
                    # --------------------------------------------------------
                    # Konvertiere dB zu linear (vektorisiert)
                    magnitude_linear = 10 ** (polar_gains / 20)
                    polar_phase_rad = np.radians(polar_phases)
                    
                    # Berechne komplexe Welle für ALLE gültigen Punkte gleichzeitig
                    # Wellenformel: p(r) = (A × G × A₀ / r) × exp(i × Φ)
                    attenuation = 1.0
                    if air_absorption_coeff > 0.0:
                        attenuation = np.exp(-air_absorption_coeff * source_dist[valid_mask])
                    wave[valid_mask] = (magnitude_linear[valid_mask] * source_level[isrc] * a_source_pa *
                                       attenuation *
                                       np.exp(1j * (wave_number * source_dist[valid_mask] +  
                                                  polar_phase_rad[valid_mask] +     
                                                  2 * np.pi * calculate_frequency * source_time[isrc])) / 
                                       source_dist[valid_mask])
                    
                    # Polaritätsinvertierung (180° Phasenverschiebung)
                    if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[isrc]:
                        wave = -wave

                # Akkumuliere Wellen (Interferenz)
                sound_field_p += wave

        # ============================================================
        # SCHRITT 4: FINALE BERECHNUNG
        # ============================================================
        # Berechne Absolutwert (Schalldruck) aus komplexer Amplitude
        sound_field_p = abs(sound_field_p)
    
        return sound_field_p, sound_field_x, sound_field_y

    def _interpolate_angle_data(self, magnitude, phase, vertical_angles, azimuth, elevation):
        """
        Interpoliert Magnitude und Phase für einen bestimmten Winkel
        
        Args:
            magnitude: Magnitude-Daten (shape: (vertical_angles, horizontal_angles) oder (vertical_angles, horizontal_angles, frequencies))
            phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form wie magnitude)
            vertical_angles: Array der vertikalen Winkel
            azimuth: Azimut-Winkel in Grad
            elevation: Elevations-Winkel in Grad
            
        Returns:
            tuple: (magnitude_value, phase_value in GRAD) oder (None, None) bei Fehler
            
        Note:
            Phase wird als unwrapped (in Grad) erwartet und kann daher linear interpoliert werden.
            Bei Verwendung in np.exp() muss die Phase mit np.radians() konvertiert werden.
        """
        try:
            # Normalisiere Azimut auf 0-360 Grad
            azimuth = azimuth % 360
            
            # Finde nächsten horizontalen Winkel (gerundet auf ganze Zahl)
            h_idx = int(round(azimuth)) % 360
            
            # Prüfe ob Daten Frequenzdimension haben (originale Daten)
            has_frequency_dim = len(magnitude.shape) == 3
            
            # Interpolation für vertikale Winkel
            if elevation <= vertical_angles[0]:
                # Elevation ist kleiner als der kleinste verfügbare Winkel
                v_idx = 0
                if has_frequency_dim:
                    # Originale Daten: Verwende mittleren Frequenzwert
                    freq_idx = magnitude.shape[2] // 2
                    mag_value = magnitude[v_idx, h_idx, freq_idx]
                    phase_value = phase[v_idx, h_idx, freq_idx]
                else:
                    # Bandgemittelte Daten: Direkter Zugriff
                    mag_value = magnitude[v_idx, h_idx]
                    phase_value = phase[v_idx, h_idx]
            elif elevation >= vertical_angles[-1]:
                # Elevation ist größer als der größte verfügbare Winkel
                v_idx = len(vertical_angles) - 1
                if has_frequency_dim:
                    # Originale Daten: Verwende mittleren Frequenzwert
                    freq_idx = magnitude.shape[2] // 2
                    mag_value = magnitude[v_idx, h_idx, freq_idx]
                    phase_value = phase[v_idx, h_idx, freq_idx]
                else:
                    # Bandgemittelte Daten: Direkter Zugriff
                    mag_value = magnitude[v_idx, h_idx]
                    phase_value = phase[v_idx, h_idx]
            else:
                # Elevation liegt zwischen zwei verfügbaren Winkeln - interpoliere
                # Finde die Indizes der umgebenden Winkel
                v_idx_lower = np.where(vertical_angles <= elevation)[0][-1]
                v_idx_upper = np.where(vertical_angles >= elevation)[0][0]
                
                if v_idx_lower == v_idx_upper:
                    # Exakter Treffer
                    if has_frequency_dim:
                        # Originale Daten: Verwende mittleren Frequenzwert
                        freq_idx = magnitude.shape[2] // 2
                        mag_value = magnitude[v_idx_lower, h_idx, freq_idx]
                        phase_value = phase[v_idx_lower, h_idx, freq_idx]
                    else:
                        # Bandgemittelte Daten: Direkter Zugriff
                        mag_value = magnitude[v_idx_lower, h_idx]
                        phase_value = phase[v_idx_lower, h_idx]
                else:
                    # Lineare Interpolation zwischen den beiden Winkeln
                    angle_lower = vertical_angles[v_idx_lower]
                    angle_upper = vertical_angles[v_idx_upper]
                    
                    # Interpolationsfaktor (0 = unterer Winkel, 1 = oberer Winkel)
                    t = (elevation - angle_lower) / (angle_upper - angle_lower)
                    
                    if has_frequency_dim:
                        # Originale Daten: Verwende mittleren Frequenzwert
                        freq_idx = magnitude.shape[2] // 2
                        mag_lower = magnitude[v_idx_lower, h_idx, freq_idx]
                        mag_upper = magnitude[v_idx_upper, h_idx, freq_idx]
                        phase_lower = phase[v_idx_lower, h_idx, freq_idx]
                        phase_upper = phase[v_idx_upper, h_idx, freq_idx]
                    else:
                        # Bandgemittelte Daten: Direkter Zugriff
                        mag_lower = magnitude[v_idx_lower, h_idx]
                        mag_upper = magnitude[v_idx_upper, h_idx]
                        phase_lower = phase[v_idx_lower, h_idx]
                        phase_upper = phase[v_idx_upper, h_idx]
                    
                    # Interpoliere Magnitude (in dB - linear interpolierbar)
                    mag_value = mag_lower + t * (mag_upper - mag_lower)
                    
                    # Interpoliere Phase (unwrapped - direkt linear interpolieren)
                    # Da Phase bereits unwrapped ist, keine zirkuläre Korrektur nötig
                    phase_value = phase_lower + t * (phase_upper - phase_lower)
            
            return mag_value, phase_value
            
        except IndexError:
            print(f"Indexfehler beim Zugriff auf Balloon-Daten bei Azimut {azimuth}° und Elevation {elevation}°")
            print(f"Array-Form: Magnitude {magnitude.shape}, Phase {phase.shape}")
            return None, None

    def _interpolate_angle_data_batch(self, magnitude, phase, vertical_angles, azimuths, elevations):
        """
        🚀 BATCH-OPTIMIERT: Interpoliert Magnitude und Phase für VIELE Winkel gleichzeitig
        
        Kernfunktion für die Performance-Optimierung der X-Achsen-Berechnung.
        Verwendet Masken-basierte Verarbeitung und np.searchsorted für vektorisierte Interpolation.
        
        Performance: ~100× schneller als Loop-basierte Einzelinterpolation
        
        Args:
            magnitude: Magnitude-Daten [vertical, horizontal] oder [vertical, horizontal, freq]
            phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form)
            vertical_angles: Array der vertikalen Winkel
            azimuths: Array von Azimut-Winkeln in Grad
            elevations: Array von Elevations-Winkeln in Grad
            
        Returns:
            tuple: (mag_values, phase_values) als Arrays oder (None, None) bei Fehler
        """
        try:
            # Normalisiere Azimute auf 0-360 Grad (vektorisiert)
            azimuths_norm = azimuths % 360
            h_indices = np.round(azimuths_norm).astype(int) % 360
            
            has_frequency_dim = len(magnitude.shape) == 3
            freq_idx = magnitude.shape[2] // 2 if has_frequency_dim else None
            
            mag_values = np.zeros_like(azimuths, dtype=np.float64)
            phase_values = np.zeros_like(azimuths, dtype=np.float64)
            
            # Punkte unterhalb kleinster Elevation → Verwende erste Messung
            mask_below = elevations <= vertical_angles[0]
            if has_frequency_dim:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below], freq_idx]
                phase_values[mask_below] = phase[0, h_indices[mask_below], freq_idx]
            else:
                mag_values[mask_below] = magnitude[0, h_indices[mask_below]]
                phase_values[mask_below] = phase[0, h_indices[mask_below]]
            
            # Punkte oberhalb größter Elevation → Verwende letzte Messung
            mask_above = elevations >= vertical_angles[-1]
            v_max = len(vertical_angles) - 1
            if has_frequency_dim:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above], freq_idx]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above], freq_idx]
            else:
                mag_values[mask_above] = magnitude[v_max, h_indices[mask_above]]
                phase_values[mask_above] = phase[v_max, h_indices[mask_above]]
            
            # Punkte ZWISCHEN zwei Messungen → Lineare Interpolation
            mask_interp = ~(mask_below | mask_above)
            
            if np.any(mask_interp):
                # Extrahiere zu interpolierende Werte
                elev_interp = elevations[mask_interp]
                h_idx_interp = h_indices[mask_interp]
                
                # 🚀 Finde umgebende Indizes für ALLE Punkte gleichzeitig
                v_idx_lower = np.searchsorted(vertical_angles, elev_interp, side='right') - 1
                v_idx_upper = v_idx_lower + 1
                
                v_idx_lower = np.clip(v_idx_lower, 0, len(vertical_angles) - 1)
                v_idx_upper = np.clip(v_idx_upper, 0, len(vertical_angles) - 1)
                
                angle_lower = vertical_angles[v_idx_lower]
                angle_upper = vertical_angles[v_idx_upper]
                
                # Berechne Interpolationsfaktoren (vektorisiert)
                t = (elev_interp - angle_lower) / (angle_upper - angle_lower + 1e-10)
                t = np.clip(t, 0, 1)
                
                if has_frequency_dim:
                    mag_lower = magnitude[v_idx_lower, h_idx_interp, freq_idx]
                    mag_upper = magnitude[v_idx_upper, h_idx_interp, freq_idx]
                    phase_lower = phase[v_idx_lower, h_idx_interp, freq_idx]
                    phase_upper = phase[v_idx_upper, h_idx_interp, freq_idx]
                else:
                    mag_lower = magnitude[v_idx_lower, h_idx_interp]
                    mag_upper = magnitude[v_idx_upper, h_idx_interp]
                    phase_lower = phase[v_idx_lower, h_idx_interp]
                    phase_upper = phase[v_idx_upper, h_idx_interp]
                
                # 🚀 Lineare Interpolation für ALLE Punkte gleichzeitig
                # value = lower + t × (upper - lower)
                mag_values[mask_interp] = mag_lower + t * (mag_upper - mag_lower)
                phase_values[mask_interp] = phase_lower + t * (phase_upper - phase_lower)
            
            return mag_values, phase_values
            
        except Exception as e:
            print(f"Fehler in Batch-Interpolation: {e}")
            return None, None

    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container