from Module_LFO.Modules_Init.Logging import measure_time, perf_section


class SpeakerPositionCalculator:
    """
    Klasse zur Berechnung von Lautsprecherpositionen und Nullpunkten.
    Unterstützt verschiedene Konfigurationen wie Stack- und Flown-Systeme.
    """
    
    def __init__(self, container):
        """
        Initialisiert den SpeakerPositionCalculator.
        
        Args:
            container: Container-Objekt mit Zugriff auf die Anwendungsdaten
        """
        self.container = container
    

    def _resolve_cabinet_metadata(self, speaker_type, position_index, array_name="?", system_type="?"):
        """
        Liefert Metadaten (front_height, back_height, angle_point) für einen Lautsprechertyp.
        Nutzt dabei bevorzugt die Container-Datenstruktur wie im Stack-Workflow.
        """
        front_height = None
        back_height = None
        angle_point = None

        container_data = getattr(self.container, 'data', None)
        speaker_names = []
        cabinet_data = []

        if isinstance(container_data, dict):
            speaker_names = container_data.get('speaker_names') or []
            cabinet_data = container_data.get('cabinet_data') or []

        if not speaker_names and hasattr(self.container, 'speaker_names'):
            speaker_names = getattr(self.container, 'speaker_names') or []
        if not speaker_names and hasattr(self.container, 'cabinet_names'):
            speaker_names = getattr(self.container, 'cabinet_names') or []
        if not cabinet_data and hasattr(self.container, 'cabinet_data'):
            cabinet_data = getattr(self.container, 'cabinet_data') or []

        def _as_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                return list(value)
            except TypeError:
                return [value]

        speaker_names = _as_list(speaker_names)
        cabinet_data = _as_list(cabinet_data)

        entry_debug = None

        for idx, name in enumerate(speaker_names):
            if name != speaker_type:
                continue
            if idx >= len(cabinet_data):
                continue
            cabinet_entry = cabinet_data[idx]
            entry = None

            if isinstance(cabinet_entry, list):
                if position_index is not None and 0 <= position_index < len(cabinet_entry):
                    candidate = cabinet_entry[position_index]
                    if isinstance(candidate, dict):
                        entry = candidate
                if entry is None:
                    entry = next((item for item in cabinet_entry if isinstance(item, dict)), None)
            elif isinstance(cabinet_entry, dict):
                entry = cabinet_entry

            entry_debug = cabinet_entry

            if isinstance(entry, dict):
                if front_height is None:
                    front_height = entry.get('front_height', front_height)
                if back_height is None:
                    back_height = entry.get('back_height', back_height)
                if angle_point is None:
                    angle_point = entry.get('angle_point', angle_point)
            break

        return front_height, back_height, angle_point


    @measure_time("SpeakerPositionCalculator.calculate_stack_center")
    def calculate_stack_center(self, speaker_array):

        """
        Berechnet den Nullpunkt in der Mitte der Lautsprecherhöhe für Lautsprechersysteme.
        Ruft die entsprechenden Methoden für Stack- und Flown-Systeme auf.
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
        """
        # Stelle sicher, dass number_of_sources korrekt ist
        if not hasattr(speaker_array, 'number_of_sources') or speaker_array.number_of_sources <= 0:
            if hasattr(speaker_array, 'source_position_z_stack') and speaker_array.source_position_z_stack is not None:
                speaker_array.number_of_sources = len(speaker_array.source_position_z_stack)
            elif hasattr(speaker_array, 'source_position_z_flown') and speaker_array.source_position_z_flown is not None:
                speaker_array.number_of_sources = len(speaker_array.source_position_z_flown)
            else:
                speaker_array.number_of_sources = 1  # Standardwert, wenn keine Informationen verfügbar sind
        
        # Stelle sicher, dass source_position_z die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_z') or speaker_array.source_position_z is None:
            speaker_array.source_position_z = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_z) != speaker_array.number_of_sources:
            # Speichere vorhandene Werte
            existing_values = list(
                speaker_array.source_position_z[:min(len(speaker_array.source_position_z), speaker_array.number_of_sources)]
            )
            # Erstelle ein neues Array mit der richtigen Länge
            speaker_array.source_position_z = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Stelle sicher, dass source_position_calc_z die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_calc_z') or speaker_array.source_position_calc_z is None:
            speaker_array.source_position_calc_z = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_z) != speaker_array.number_of_sources:
            # Speichere vorhandene Werte
            existing_values = list(
                speaker_array.source_position_calc_z[:min(len(speaker_array.source_position_calc_z), speaker_array.number_of_sources)]
            )
            # Erstelle ein neues Array mit der richtigen Länge
            speaker_array.source_position_calc_z = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))

        if not hasattr(speaker_array, 'source_position_calc_y') or speaker_array.source_position_calc_y is None:
            if hasattr(speaker_array, 'source_position_y') and speaker_array.source_position_y is not None:
                speaker_array.source_position_calc_y = list(speaker_array.source_position_y[:speaker_array.number_of_sources])
            else:
                speaker_array.source_position_calc_y = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_y) != speaker_array.number_of_sources:
            existing_values = list(
                speaker_array.source_position_calc_y[:min(len(speaker_array.source_position_calc_y), speaker_array.number_of_sources)]
            )
            speaker_array.source_position_calc_y = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))

        configuration = getattr(speaker_array, 'configuration', None)
        if configuration is None or configuration.lower() == "stack":
            if hasattr(speaker_array, 'source_position_y') and speaker_array.source_position_y is not None:
                speaker_array.source_position_calc_y = list(
                    speaker_array.source_position_y[:speaker_array.number_of_sources]
                )
            else:
                speaker_array.source_position_calc_y = [0.0] * speaker_array.number_of_sources

        if not hasattr(speaker_array, 'source_position_x') or speaker_array.source_position_x is None:
            speaker_array.source_position_x = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_x) != speaker_array.number_of_sources:
            existing_values = list(
                speaker_array.source_position_x[:min(len(speaker_array.source_position_x), speaker_array.number_of_sources)]
            )
            speaker_array.source_position_x = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))

        # Hole Array-Positionen (fixe Offset-Positionen)
        array_pos_x = getattr(speaker_array, 'array_position_x', 0.0)
        array_pos_y = getattr(speaker_array, 'array_position_y', 0.0)
        array_pos_z = getattr(speaker_array, 'array_position_z', 0.0)
        
        # Prüfe, ob es sich um ein Flown-System handelt
        configuration = getattr(speaker_array, 'configuration', None)
        is_flown = configuration is not None and str(configuration).lower() == "flown"
        
        speaker_array.source_position_calc_x = list(
            speaker_array.source_position_x[:speaker_array.number_of_sources]
        )
        
        # Stelle sicher, dass source_position_calc_y die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_calc_y') or speaker_array.source_position_calc_y is None:
            speaker_array.source_position_calc_y = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_y) != speaker_array.number_of_sources:
            existing_values = list(
                speaker_array.source_position_calc_y[:min(len(speaker_array.source_position_calc_y), speaker_array.number_of_sources)]
            )
            speaker_array.source_position_calc_y = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Stelle sicher, dass source_position_calc_z die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_calc_z') or speaker_array.source_position_calc_z is None:
            speaker_array.source_position_calc_z = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_z) != speaker_array.number_of_sources:
            existing_values = list(
                speaker_array.source_position_calc_z[:min(len(speaker_array.source_position_calc_z), speaker_array.number_of_sources)]
            )
            speaker_array.source_position_calc_z = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Rufe die entsprechenden Methoden auf
        self.calculate_stack_z_center(speaker_array)
        self.calculate_flown_z_center(speaker_array)
        
        # Addiere Array-Positionen zu den berechneten Positionen
        # Bei Flown-Systemen sind die Array-Positionen bereits in source_position_x/y/z_flown enthalten,
        # daher werden sie hier nicht nochmal hinzuaddiert
        # Bei Stack-Systemen (oder wenn configuration None ist) werden sie hinzuaddiert
        if not is_flown:
            # Stelle sicher, dass alle Arrays die richtige Länge haben
            if len(speaker_array.source_position_calc_x) == speaker_array.number_of_sources and \
               len(speaker_array.source_position_calc_y) == speaker_array.number_of_sources and \
               len(speaker_array.source_position_calc_z) == speaker_array.number_of_sources:
                for i in range(speaker_array.number_of_sources):
                    speaker_array.source_position_calc_x[i] += array_pos_x
                    speaker_array.source_position_calc_y[i] += array_pos_y
                    speaker_array.source_position_calc_z[i] += array_pos_z


    def calculate_stack_z_center(self, speaker_array):
        """
        Berechnet den Z-Nullpunkt in der Mitte der Lautsprecherhöhe für Stack-Systeme.
        Berücksichtigt die front_height aus den Cabinet-Daten.
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
        """
        # Prüfe, ob es sich um ein Stack-System handelt
        if not hasattr(speaker_array, 'configuration') or speaker_array.configuration.lower() != "stack":
            return
            
        # Prüfe, ob source_position_z_stack existiert
        if not hasattr(speaker_array, 'source_position_z_stack') or speaker_array.source_position_z_stack is None:
            # Wenn source_position_z_stack nicht existiert, verwende source_position_z
            if hasattr(speaker_array, 'source_position_z') and speaker_array.source_position_z is not None:
                # Initialisiere source_position_calc_z aus source_position_z
                speaker_array.source_position_calc_z = list(speaker_array.source_position_z[:speaker_array.number_of_sources])
            return
        
        # Zurücksetzen von source_position_calc_z
        speaker_array.source_position_calc_z = [0.0] * len(speaker_array.source_position_z_stack)
        
        # Berechne den Z-Nullpunkt für jeden Lautsprecher unter Berücksichtigung der front_height
        for i in range(len(speaker_array.source_position_z_stack)):
            # Hole den Lautsprechertyp
            if hasattr(speaker_array, 'source_polar_pattern') and i < len(speaker_array.source_polar_pattern):
                speaker_type = speaker_array.source_polar_pattern[i]
                
                front_height, back_height, _ = self._resolve_cabinet_metadata(
                    speaker_type,
                    i,
                    array_name=getattr(speaker_array, 'name', '?'),
                    system_type="Stack"
                )

                if front_height is None and back_height is not None:
                    front_height = back_height
                if front_height is None:
                    front_height = 0.0
                if back_height is None:
                    back_height = front_height
                
                # Berechne die Z-Position mit Berücksichtigung der front_height
                speaker_array.source_position_calc_z[i] = speaker_array.source_position_z_stack[i] + front_height/2
                
            else:
                # Wenn kein Lautsprechertyp definiert ist, verwende die normale Z-Position
                speaker_array.source_position_calc_z[i] = speaker_array.source_position_z_stack[i]
                        

    @measure_time("SpeakerPositionCalculator.calculate_flown_z_center")
    def calculate_flown_z_center(self, speaker_array):
        """
        Berechnet den Z-Nullpunkt in der Mitte der Lautsprecherhöhe für Flown-Systeme.
        Berücksichtigt die front_height aus den Cabinet-Daten und den Winkelpunkt.
        Bei Flown-Systemen stellt source_position_z_flown die obere Kante des obersten Lautsprechers dar.
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
        """
        
        # Prüfe, ob es sich um ein Flown-System handelt
        if not hasattr(speaker_array, 'configuration'):
            return
        if speaker_array.configuration is None:
            return
        if speaker_array.configuration.lower() != "flown":
            return
            
        # Prüfe, ob source_position_z_flown existiert
        if not hasattr(speaker_array, 'source_position_z_flown'):
            return
        if speaker_array.source_position_z_flown is None:
            return
            
        # Stelle sicher, dass number_of_sources korrekt ist
        if not hasattr(speaker_array, 'number_of_sources') or speaker_array.number_of_sources <= 0:
            speaker_array.number_of_sources = len(speaker_array.source_position_z_flown)            
        
        # Stelle sicher, dass source_position_calc_z die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_calc_z') or speaker_array.source_position_calc_z is None:
            speaker_array.source_position_calc_z = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_z) != speaker_array.number_of_sources:
            # Speichere vorhandene Werte
            existing_values = speaker_array.source_position_calc_z[:min(len(speaker_array.source_position_calc_z), speaker_array.number_of_sources)]
            # Erstelle ein neues Array mit der richtigen Länge
            speaker_array.source_position_calc_z = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Stelle sicher, dass source_position_y existiert und die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_y') or speaker_array.source_position_y is None:
            speaker_array.source_position_y = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_y) != speaker_array.number_of_sources:
            # Speichere vorhandene Werte
            existing_values = speaker_array.source_position_y[:min(len(speaker_array.source_position_y), speaker_array.number_of_sources)]
            # Erstelle ein neues Array mit der richtigen Länge
            speaker_array.source_position_y = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))

        # Stelle sicher, dass source_position_x existiert und die richtige Länge hat
        if not hasattr(speaker_array, 'source_position_x') or speaker_array.source_position_x is None:
            speaker_array.source_position_x = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_x) != speaker_array.number_of_sources:
            existing_values = speaker_array.source_position_x[:min(len(speaker_array.source_position_x), speaker_array.number_of_sources)]
            speaker_array.source_position_x = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Stelle sicher, dass source_angle die richtige Länge hat
        if hasattr(speaker_array, 'source_angle'):
            if speaker_array.source_angle is None:
                # Wenn source_angle None ist, erstelle ein neues Array
                speaker_array.source_angle = [0.0] * speaker_array.number_of_sources
            elif len(speaker_array.source_angle) != speaker_array.number_of_sources:
                # Wenn source_angle die falsche Länge hat, passe es an
                import numpy as np
                if isinstance(speaker_array.source_angle, np.ndarray):
                    # Für NumPy-Arrays
                    # Speichere vorhandene Werte
                    existing_values = speaker_array.source_angle[:min(len(speaker_array.source_angle), speaker_array.number_of_sources)]
                    # Erstelle ein neues Array mit der richtigen Länge
                    new_array = np.zeros(speaker_array.number_of_sources)
                    new_array[:len(existing_values)] = existing_values
                    speaker_array.source_angle = new_array
                else:
                    # Für Python-Listen
                    # Speichere vorhandene Werte
                    existing_values = speaker_array.source_angle[:min(len(speaker_array.source_angle), speaker_array.number_of_sources)]
                    # Erstelle ein neues Array mit der richtigen Länge
                    speaker_array.source_angle = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        # Stelle sicher, dass source_site die richtige Länge hat
        if hasattr(speaker_array, 'source_site'):
            if speaker_array.source_site is None:
                # Wenn source_site None ist, erstelle ein neues Array
                speaker_array.source_site = [0.0] * speaker_array.number_of_sources
            elif len(speaker_array.source_site) != speaker_array.number_of_sources:
                # Wenn source_site die falsche Länge hat, passe es an
                import numpy as np
                if isinstance(speaker_array.source_site, np.ndarray):
                    # Für NumPy-Arrays
                    # Speichere vorhandene Werte
                    existing_values = speaker_array.source_site[:min(len(speaker_array.source_site), speaker_array.number_of_sources)]
                    # Erstelle ein neues Array mit der richtigen Länge
                    new_array = np.zeros(speaker_array.number_of_sources)
                    new_array[:len(existing_values)] = existing_values
                    speaker_array.source_site = new_array
                else:
                    # Für Python-Listen
                    # Speichere vorhandene Werte
                    existing_values = speaker_array.source_site[:min(len(speaker_array.source_site), speaker_array.number_of_sources)]
                    # Erstelle ein neues Array mit der richtigen Länge
                    speaker_array.source_site = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))
        
        def _to_float(value, default=0.0):
            """Konvertiert verschieden formatierte Winkel-/Positionswerte robust in float."""
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned.endswith("°"):
                    cleaned = cleaned[:-1]
                cleaned = cleaned.replace(",", ".")
                if cleaned == "":
                    return default
                try:
                    return float(cleaned)
                except ValueError:
                    return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _normalize_numeric_sequence(sequence, target_length):
            values = []
            try:
                iterable = list(sequence)
            except TypeError:
                iterable = [sequence]
            for idx in range(target_length):
                if idx < len(iterable):
                    raw_value = iterable[idx]
                elif iterable:
                    raw_value = iterable[-1]
                else:
                    raw_value = 0.0
                values.append(_to_float(raw_value))
            return values

        if hasattr(speaker_array, 'source_angle'):
            if speaker_array.source_angle is None:
                speaker_array.source_angle = [0.0] * speaker_array.number_of_sources
            else:
                speaker_array.source_angle = _normalize_numeric_sequence(
                    speaker_array.source_angle,
                    speaker_array.number_of_sources
                )
        else:
            speaker_array.source_angle = [0.0] * speaker_array.number_of_sources

        if hasattr(speaker_array, 'source_site'):
            if speaker_array.source_site is None:
                speaker_array.source_site = [0.0] * speaker_array.number_of_sources
            else:
                speaker_array.source_site = _normalize_numeric_sequence(
                    speaker_array.source_site,
                    speaker_array.number_of_sources
                )
        else:
            speaker_array.source_site = [0.0] * speaker_array.number_of_sources

        if not hasattr(speaker_array, 'source_position_calc_y') or speaker_array.source_position_calc_y is None:
            if hasattr(speaker_array, 'source_position_y') and speaker_array.source_position_y is not None:
                speaker_array.source_position_calc_y = list(speaker_array.source_position_y[:speaker_array.number_of_sources])
            else:
                speaker_array.source_position_calc_y = [0.0] * speaker_array.number_of_sources
        elif len(speaker_array.source_position_calc_y) != speaker_array.number_of_sources:
            existing_values = list(
                speaker_array.source_position_calc_y[:min(len(speaker_array.source_position_calc_y), speaker_array.number_of_sources)]
            )
            speaker_array.source_position_calc_y = existing_values + [0.0] * (speaker_array.number_of_sources - len(existing_values))

        speaker_array.source_position_calc_x = list(
            speaker_array.source_position_x[:speaker_array.number_of_sources]
        )

        # Sammle die Gehäusehöhen und Winkelpunkt-Informationen für alle Lautsprecher
        front_height_values = []
        back_height_values = []
        effective_heights = []
        angle_point_types = []
        
        for i in range(speaker_array.number_of_sources):
            # Werte werden aus den Cabinet-Metadaten gelesen
            front_height = None
            back_height = None
            angle_point_type = "Front"  # Standardwert für den Drehpunkt
            speaker_type = None
            
            # Hole den Lautsprechertyp
            if hasattr(speaker_array, 'source_type') and speaker_array.source_type is not None and i < len(speaker_array.source_type):
                speaker_type = speaker_array.source_type[i]
            elif hasattr(speaker_array, 'source_polar_pattern') and speaker_array.source_polar_pattern is not None and i < len(speaker_array.source_polar_pattern):
                speaker_type = speaker_array.source_polar_pattern[i]
            # Wenn der Lautsprechertyp bekannt ist, hole die Cabinet-Daten
            if speaker_type is not None:
                front_height, back_height, angle_point_entry = self._resolve_cabinet_metadata(
                    speaker_type,
                    i,
                    array_name=getattr(speaker_array, 'name', '?'),
                    system_type="Flown"
                )
                if angle_point_entry is not None:
                    if isinstance(angle_point_entry, str):
                        angle_point_entry = angle_point_entry.strip().capitalize()
                    angle_point_type = angle_point_entry or angle_point_type
                
                # Stelle sicher, dass angle_point_type einer der erwarteten Werte ist
                if angle_point_type not in ["Front", "Back", "Center", "None"]:
                    angle_point_type = "Front"  # Standardwert für unbekannte Werte
            
            front_height = _to_float(front_height, default=0.0)
            back_height = _to_float(back_height, default=front_height if front_height is not None else 0.0)
            if angle_point_type == "Back":
                effective_height = back_height
            elif angle_point_type == "Center":
                effective_height = (front_height + back_height) / 2
            elif angle_point_type == "None":
                effective_height = max(front_height, back_height)
            else:
                effective_height = front_height
            
            effective_height = max(effective_height, 0.0)
            
            front_height_values.append(front_height)
            back_height_values.append(back_height)
            effective_heights.append(effective_height)
            angle_point_types.append(angle_point_type)
        
        # Stelle sicher, dass source_position_z_flown die richtige Länge hat
        if len(speaker_array.source_position_z_flown) != speaker_array.number_of_sources:
            # Speichere den ersten Wert, falls vorhanden
            first_value = speaker_array.source_position_z_flown[0] if len(speaker_array.source_position_z_flown) > 0 else 0.0
            # Erstelle ein neues Array mit der richtigen Länge
            speaker_array.source_position_z_flown = [first_value] * speaker_array.number_of_sources
        
        # Berechne den Gesamtwinkel für jeden Lautsprecher
        total_angles = [0.0] * speaker_array.number_of_sources
        
        # Hole die Neigung des ersten Lautsprechers (source_site)
        first_site = 0.0
        if hasattr(speaker_array, 'source_site') and speaker_array.number_of_sources > 0:
            if len(speaker_array.source_site) > 0:
                first_site = speaker_array.source_site[0]
        
        # Für den ersten Lautsprecher ist der Gesamtwinkel gleich seiner Neigung (source_site)
        total_angles[0] = first_site
        
        # Für die restlichen Lautsprecher: Gesamtwinkel = Neigung des ersten Lautsprechers - Summe der Zwischenwinkel
        # Beachte: Wir subtrahieren die Winkel, da ein positiver source_angle bedeutet, dass der Lautsprecher stärker nach unten zeigt
        for i in range(1, speaker_array.number_of_sources):
            # Starte mit der Neigung des ersten Lautsprechers
            total_angles[i] = first_site
            
            # Subtrahiere die Zwischenwinkel zu den oberen Lautsprechern
            if hasattr(speaker_array, 'source_angle'):
                for j in range(1, i+1):  # Für alle Lautsprecher von 1 bis i
                    if j < len(speaker_array.source_angle):
                        total_angles[i] -= speaker_array.source_angle[j]
        
        # Importiere math für trigonometrische Funktionen
        import math
        
        flown_segment_info = []
        prev_bottom_front_z = None
        prev_bottom_front_y = None
        prev_bottom_back_z = None
        prev_bottom_back_y = None
        prev_bottom_center_z = None
        prev_bottom_center_y = None
        prev_angle_point = None

        base_y_offset = 0.0
        if hasattr(speaker_array, 'source_position_y') and speaker_array.source_position_y is not None:
            base_values = _normalize_numeric_sequence(
                getattr(speaker_array, 'source_position_y', []),
                1,
            )
            if base_values:
                base_y_offset = base_values[0]

        for i in range(speaker_array.number_of_sources):
            angle_rad = math.radians(total_angles[i])
            angle_point_type = angle_point_types[i]
            angle_point_lower = (angle_point_type or "front").strip().lower()
            if angle_point_lower not in {"front", "back", "center", "none"}:
                angle_point_lower = "front"
            front_height = front_height_values[i]
            back_height = back_height_values[i]

            if i == 0:
                top_z = float(speaker_array.source_position_z_flown[i])
                top_y = base_y_offset
            else:
                if prev_angle_point == "back" and prev_bottom_back_z is not None and prev_bottom_back_y is not None:
                    top_z = prev_bottom_back_z
                    top_y = prev_bottom_back_y
                elif prev_angle_point == "center" and prev_bottom_center_z is not None and prev_bottom_center_y is not None:
                    top_z = prev_bottom_center_z
                    top_y = prev_bottom_center_y
                elif prev_angle_point == "none" and prev_bottom_center_z is not None and prev_bottom_center_y is not None:
                    top_z = prev_bottom_center_z
                    top_y = prev_bottom_center_y
                elif prev_bottom_front_z is not None and prev_bottom_front_y is not None:
                    top_z = prev_bottom_front_z
                    top_y = prev_bottom_front_y
                else:
                    top_z = float(speaker_array.source_position_z_flown[i])
                    top_y = 0.0

            front_delta_z = front_height * math.cos(angle_rad)
            front_delta_y = front_height * math.sin(angle_rad)
            back_delta_z = back_height * math.cos(angle_rad)
            back_delta_y = back_height * math.sin(angle_rad)

            bottom_front_z = top_z - front_delta_z
            bottom_front_y = top_y + front_delta_y
            bottom_back_z = top_z - back_delta_z
            bottom_back_y = top_y + back_delta_y

            if angle_point_lower == "back":
                reference_height = back_height
                middle_z = top_z - (back_height / 2.0) * math.cos(angle_rad)
                middle_y = top_y + (back_height / 2.0) * math.sin(angle_rad)
                pivot_z = top_z
                pivot_y = top_y
            elif angle_point_lower == "center":
                reference_height = (front_height + back_height) / 2.0
                middle_z = top_z - (reference_height / 2.0) * math.cos(angle_rad)
                middle_y = top_y + (reference_height / 2.0) * math.sin(angle_rad)
                pivot_z = (top_z + bottom_front_z + bottom_back_z) / 3.0
                pivot_y = (top_y + bottom_front_y + bottom_back_y) / 3.0
            elif angle_point_lower == "none":
                reference_height = max(front_height, back_height)
                middle_z = top_z - (reference_height / 2.0) * math.cos(angle_rad)
                middle_y = top_y + (reference_height / 2.0) * math.sin(angle_rad)
                pivot_z = top_z
                pivot_y = top_y
            else:  # Front
                reference_height = front_height
                middle_z = top_z - (front_height / 2.0) * math.cos(angle_rad)
                middle_y = top_y + (front_height / 2.0) * math.sin(angle_rad)
                pivot_z = top_z
                pivot_y = top_y

            bottom_center_z = top_z - effective_heights[i] * math.cos(angle_rad)
            bottom_center_y = top_y + effective_heights[i] * math.sin(angle_rad)

            # Speichere die berechneten Positionen
            speaker_array.source_position_calc_z[i] = middle_z
            speaker_array.source_position_calc_y[i] = middle_y
            speaker_array.source_position_z_flown[i] = top_z

            flown_segment_info.append({
                'top_y': float(top_y),
                'top_z': float(top_z),
                'center_y': float(middle_y),
                'center_z': float(middle_z),
                'bottom_front_y': float(bottom_front_y),
                'bottom_front_z': float(bottom_front_z),
                'bottom_back_y': float(bottom_back_y),
                'bottom_back_z': float(bottom_back_z),
                'bottom_center_y': float(bottom_center_y),
                'bottom_center_z': float(bottom_center_z),
                'angle_deg': float(total_angles[i] if i < len(total_angles) else first_site),
                'effective_height': float(effective_heights[i]),
                'front_height': float(front_height_values[i]),
                'back_height': float(back_height_values[i]),
                'angle_point': angle_point_types[i],
                'pivot_y': float(pivot_y),
                'pivot_z': float(pivot_z),
            })

            prev_bottom_front_z = bottom_front_z
            prev_bottom_front_y = bottom_front_y
            prev_bottom_back_z = bottom_back_z
            prev_bottom_back_y = bottom_back_y
            prev_bottom_center_z = bottom_center_z
            prev_bottom_center_y = bottom_center_y
            prev_angle_point = angle_point_lower

        speaker_array._flown_segment_geometry = flown_segment_info
