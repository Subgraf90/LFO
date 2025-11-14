import logging
import os
from time import perf_counter

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase
from scipy import interpolate


class ImpulseCalculator(ModuleBase):
    def __init__(self, settings, data):
        super().__init__(settings)

        self.settings = settings
        self.data = data
        self.calculation = {}
        
        # 🚀 PERFORMANCE: Cache für optimierten Balloon-Data Zugriff
        self._data_container = None  # Wird bei Bedarf gesetzt
        self._speaker_name_list = None
        self._speaker_name_index = None
        self._frequency_cache = {}
        self._balloon_angle_cache = {}
        self._speaker_array_cache = None
        self._perf_enabled = bool(int(os.environ.get("LFO_DEBUG_PERF", "1")))
        self._perf_logger = logging.getLogger("LFO.Performance")

    def _perf_scope(self, label: str):
        return _PerfScope(self, label)

    def _log_perf(self, label: str, duration: float) -> None:
        if self._perf_enabled:
            self._perf_logger.info("[PERF] %s: %.3f s", label, duration)


    def calculate_impulse(self):
        """Berechnet alle Daten für die verschiedenen Plots"""
        with self._perf_scope("ImpulseCalculator total"):
            # Caches pro Lauf leeren, damit aktuelle Einstellungen berücksichtigt werden
            self._frequency_cache.clear()
            self._balloon_angle_cache.clear()

            # Berechne zuerst die Impulsantworten
            self._speaker_array_cache = None
            with self._perf_scope("ImpulseCalculator impulse_responses"):
                impulse_responses = self.calculate_impulse_responses()
            if not impulse_responses:  # Frühe Prüfung auf None
                self.calculation = {}  # Leere Berechnung zurücksetzen
                return

            # 🚀 PERFORMANCE: Lade alle Balloondaten vorab
            with self._perf_scope("ImpulseCalculator preload_balloon"):
                balloon_cache = self._preload_balloon_data(impulse_responses)

            # Berechne Individual Impulse Responses
            with self._perf_scope("ImpulseCalculator individual_impulse"):
                individual_impulse_data = self.calculate_individual_impulse_responses(impulse_responses, balloon_cache)
            with self._perf_scope("ImpulseCalculator phase"):
                phase_data = self.calculate_phase_response(impulse_responses, balloon_cache)
            with self._perf_scope("ImpulseCalculator magnitude"):
                magnitude_data = self.calculate_magnitude_response(impulse_responses, balloon_cache)
            with self._perf_scope("ImpulseCalculator arrival_times"):
                arrival_times_data = self.calculate_arrival_times(impulse_responses, balloon_cache)
            with self._perf_scope("ImpulseCalculator combined_impulse"):
                combined_impulse_data = self.calculate_combined_impulse_response(impulse_responses, phase_data, balloon_cache)
            
            # Initialisiere base_data für jeden Punkt
            with self._perf_scope("ImpulseCalculator assemble_results"):
                for point_key in self.settings.impulse_points:
                    key = point_key['key']
                    
                    base_data = {
                        "impulse_response": {
                            "spl": [],         
                            "time": [],        
                            "color": [],
                            "combined_impulse": None
                        },
                        "phase_response": {
                            "freq": [],        
                            "phase": [],       
                            "color": [],
                            "combined_phase": None
                        },
                        "magnitude_data": {
                            "frequency": None,
                            "magnitude": None,
                            "individual_magnitudes": {}
                        },
                        "arrival_times": {
                            "time": [],        
                            "spl_max": [],     
                            "spl_min": [],     
                            "color": []        
                        },
                        "show_in_plot": True
                    }
                    
                    if key in impulse_responses:
                        # Setze Individual Impulse Responses
                        if individual_impulse_data and key in individual_impulse_data:
                            base_data["impulse_response"].update({
                                "spl": individual_impulse_data[key]["spl"],
                                "time": individual_impulse_data[key]["time"],
                                "color": individual_impulse_data[key]["color"]
                            })

                        # Setze Phase Response
                        if phase_data and key in phase_data:
                            base_data["phase_response"].update({
                                "freq": phase_data[key]["freq"],
                                "phase": phase_data[key]["phase"],
                                "color": phase_data[key]["color"],
                                "combined_phase": phase_data[key]["combined_phase"]
                            })

                        # Setze Magnitude Response
                        if magnitude_data and key in magnitude_data:
                            base_data["magnitude_data"].update({
                                "frequency": magnitude_data[key]["frequency"],
                                "magnitude": magnitude_data[key]["magnitude"],
                                "individual_magnitudes": magnitude_data[key]["individual_magnitudes"]
                            })
                        
                        # Setze Arrival Times
                        if arrival_times_data and key in arrival_times_data:
                            base_data["arrival_times"].update({
                                "time": arrival_times_data[key]["time"],
                                "spl_max": arrival_times_data[key]["spl_max"],
                                "spl_min": arrival_times_data[key]["spl_min"],
                                "color": arrival_times_data[key]["color"]
                            })
                        
                        # Setze Combined Impulse Response
                        if combined_impulse_data and key in combined_impulse_data:
                            base_data["impulse_response"]["combined_impulse"] = {
                                "spl": combined_impulse_data[key]["impulse"].tolist(),
                                "time": combined_impulse_data[key]["time"].tolist()
                            }

                    self.calculation[key] = base_data


    def get_balloon_data_at_angle(self, speaker_name, azimuth, elevation=0):
        """
        Holt die Balloon-Daten (Magnitude und Phase) für einen bestimmten Winkel
        mit voller Frequenzauflösung und Elevation-Interpolation
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuth (float): Azimut-Winkel in Grad (0-360)
            elevation (float, optional): Elevations-Winkel in Grad. Standard ist 0.
            
        Returns:
            dict: Dictionary mit Frequenz-, Magnitude- und Phase-Daten oder None
            
        Note:
            Phase wird als unwrapped in GRAD zurückgegeben.
            Bei Verwendung in np.exp() muss mit np.radians() konvertiert werden.
        """
        cache_key = (
            speaker_name,
            round(float(azimuth), 2),
            round(float(elevation), 2),
        )
        cached = self._balloon_angle_cache.get(cache_key)
        if cached is not None:
            return cached

        # 🚀 REIN OPTIMIERT: Nur noch optimierte Datenstruktur
        balloon_data = self._data_container.get_balloon_data(speaker_name)
        if balloon_data is None:
            print(f"❌ Keine Balloon-Daten für {speaker_name}")
            return None
        
        if not isinstance(balloon_data, dict):
            return None
            
        # Extrahiere Daten
        vertical_angles = balloon_data.get('vertical_angles')
        magnitude = balloon_data.get('magnitude')
        phase = balloon_data.get('phase')
        freqs = balloon_data.get('freqs')
        
        if (vertical_angles is None or magnitude is None or 
            phase is None or freqs is None):
            return None
        
        # ✅ ELEVATION-INTERPOLATION: Verwende neue Interpolationsmethode
        result = self._interpolate_balloon_data_with_elevation(
            magnitude, phase, freqs, vertical_angles, azimuth, elevation
        )

        if result is not None:
            self._balloon_angle_cache[cache_key] = result

        return result


    def get_frequency_data(self, speaker_idx=0, angle_idx=0, elevation=0):
        """Holt Frequenzdaten für einen spezifischen Lautsprecher und Winkel"""
        
        try:
            cache_key = (
                int(speaker_idx),
                int(angle_idx) % 360,
                round(float(elevation), 2),
            )
            cached = self._frequency_cache.get(cache_key)
            if cached is not None:
                return cached

            # Hole Balloon-Daten für diesen Lautsprecher
            speaker_names = self._get_speaker_names_list()
            if speaker_idx < len(speaker_names):
                speaker_name = speaker_names[speaker_idx]
                
                # 🚀 REIN OPTIMIERT: Nur noch optimierte Datenstruktur
                balloon_data = self._data_container.get_balloon_data(speaker_name)
                
                if balloon_data is not None:
                    # Extrahiere die Daten aus dem Dictionary
                    if isinstance(balloon_data, dict):
                        balloon_vertical_angles = balloon_data.get('vertical_angles')
                        balloon_mag = balloon_data.get('magnitude')
                        balloon_phase = balloon_data.get('phase')
                        balloon_freqs = balloon_data.get('freqs')
                        
                        if (balloon_vertical_angles is not None and 
                            balloon_mag is not None and 
                            balloon_phase is not None and
                            balloon_freqs is not None):
                            try:
                                # Verwende den übergebenen angle_idx für den horizontalen Winkel
                                # Für Impulse verwenden wir den tatsächlichen Winkel ohne Korrektur
                                h_idx = angle_idx % 360
                                
                                # Verwende den übergebenen elevation Parameter für vertikalen Winkel
                                v_idx = np.abs(balloon_vertical_angles - elevation).argmin()
                                

                                
                                # Hole Magnitude und Phase für alle Frequenzen
                                mag = balloon_mag[v_idx, h_idx, :]
                                phase = balloon_phase[v_idx, h_idx, :]
                                freq = balloon_freqs
                                
                                # Phase ist bereits unwrapped - kein weiteres Unwrapping nötig!
                                
                                # Konvertiere dB zu linearer Magnitude
                                mag = 10 ** (mag / 20)
                                
                                data = {
                                    'freq': freq,
                                    'mag': mag,
                                    'phase': phase
                                }
                                self._frequency_cache[cache_key] = data
                                return data
                            except IndexError as e:
                                print(f"Indexfehler beim Zugriff auf Balloon-Daten: {e}")
                                print(f"Balloon-Daten-Form: {balloon_mag.shape}, Indizes: v_idx={v_idx}, h_idx={h_idx}")
                                return None
            
            return None
        except Exception as e:
            print(f"Fehler beim Abrufen der Frequenzdaten: {e}")
            return None


    def calculate_impulse_responses(self):
        """Berechnet die Impulsantworten für jeden Lautsprecher pro Array"""
        with self._perf_scope("ImpulseCalculator calculate_impulse_responses"):
            if not self.settings.impulse_points:  # Prüfe ob Impulspunkte existieren
                return None

            # 🌡️ Temperaturabhängige Schallgeschwindigkeit
            speed_of_sound = self.functions.calculate_speed_of_sound(self.settings.temperature)

            impulse_responses = {}
            speaker_names = self._get_speaker_names_list()
            speaker_index_map = self._get_speaker_name_index()
            
            # Sample-Rate aus Balloon-Daten des ersten Lautsprechers
            first_speaker_name = speaker_names[0] if speaker_names else None
            if first_speaker_name:
                # 🚀 OPTIMIERT: Verwende neuen DataContainer
                balloon_data = self._data_container.get_balloon_data(first_speaker_name)
                if balloon_data is not None and 'freqs' in balloon_data:
                    freqs = balloon_data['freqs']
                    max_freq = max(freqs)
                    sample_rate = max_freq * 2  # Nyquist
                    fft_size = len(freqs) * 2 - 2
                else:
                    sample_rate = 44100
                    fft_size = 2048
            else:
                sample_rate = 44100
                fft_size = 2048
    
            for point_key in self.settings.impulse_points:
                point_x, point_y = point_key['data']
                speaker_responses = {}
                max_amplitude = 0  # Für gemeinsame Normalisierung
                
                # Erste Schleife: Berechne alle Impulsantworten und finde maximale Amplitude
                temp_responses = {}
                for speaker_array in self.settings.speaker_arrays.values():
                    if speaker_array.mute or speaker_array.hide:
                        continue
                                        
                    for i, speaker_name in enumerate(speaker_array.source_polar_pattern):
                        speaker_idx = speaker_index_map.get(speaker_name)
                        if speaker_idx is None:
                            continue
                        
                        time_offset = point_key['time_offset']

                        # Geometrie-Berechnung
                        source_position_z = getattr(speaker_array, 'source_position_calc_z', None)

                        source_position_x = getattr(
                            speaker_array,
                            'source_position_x',
                            speaker_array.source_position_x,
                        )
                        x_distance = point_x - source_position_x[i]
                        source_position_y = getattr(
                            speaker_array,
                            'source_position_calc_y',
                            speaker_array.source_position_y,
                        )
                        y_distance = point_y - source_position_y[i]
                        z_distance = source_position_z[i]
                        
                        # Horizontale und 3D-Distanz berechnen
                        horizontal_dist = np.sqrt(x_distance**2 + y_distance**2)
                        distance = np.sqrt(horizontal_dist**2 + z_distance**2)
                        # Laufzeit in ms berechnen
                        delay_ms = (distance / speed_of_sound) * 1000
                        
                        # Winkel berechnen
                        source_to_point_angle = np.arctan2(y_distance, x_distance)
                        azimuth = (np.degrees(source_to_point_angle) + speaker_array.source_azimuth[i]) % 360
                        
                        # Winkeltransformation für Balloon-Daten
                        # Invertiere die Richtung (im Uhrzeigersinn statt gegen den Uhrzeigersinn)
                        azimuth = (360 - azimuth) % 360
                        
                        # Drehe um 90° im Uhrzeigersinn
                        azimuth = (azimuth + 90) % 360
                        
                        # Elevationswinkel berechnen
                        elevation = np.degrees(np.arctan2(z_distance, horizontal_dist))
                        
                        try:
                            # Verwende get_balloon_data_at_angle für bereits unwrapped Phase
                            freq_data = self.get_balloon_data_at_angle(speaker_name, azimuth, elevation)
                            if freq_data is None:
                                print(f"❌ Keine Balloon-Daten für {speaker_name}")
                                continue
                            
                            freq = freq_data['freq']
                            mag = freq_data['mag']  # Bereits in dB
                            phase = freq_data['phase']  # Bereits unwrapped in GRAD
                            
                            # Konvertiere dB zu linearer Magnitude
                            mag_linear = 10 ** (mag / 20)
                            
                            # Interpolation zu linearem Frequenzabstand für korrekte IFFT
                            n_interp = fft_size // 2 + 1
                            target_freq = np.linspace(freq[0], freq[-1], n_interp)
                            mag_interp = interpolate.interp1d(freq, mag_linear, kind='linear')(target_freq)
                            phase_interp = interpolate.interp1d(freq, phase, kind='linear')(target_freq)
                            
                            # Konvertiere Phase von Grad zu Radiant für np.exp()
                            phase_interp_rad = np.radians(phase_interp)
                            
                            # Komplexes Spektrum und IFFT
                            H = mag_interp * np.exp(1j * phase_interp_rad)
                            impulse = np.fft.irfft(H, n=fft_size)
                            
                            # Zeitachse mit korrekter Laufzeit
                            dt = 1/sample_rate * 1000  # Zeitschritt in ms
                            time = np.arange(0, fft_size) * dt + delay_ms - time_offset
                            
                            # Delays addieren
                            if hasattr(speaker_array, 'delay'):
                                time = time + speaker_array.delay
                            if hasattr(speaker_array, 'source_time'):
                                time = time + speaker_array.source_time[i]
                            
                            # Amplitudenskalierung
                            impulse = impulse / (distance + 1e-6)  # Pegelabnahme mit Entfernung
                            array_gain_factor = 10 ** (speaker_array.gain / 20)
                            source_level_factor = 10 ** (speaker_array.source_level[i] / 20)
                            impulse = impulse * array_gain_factor * source_level_factor
                            
                            # Polaritätsinvertierung
                            if hasattr(speaker_array, 'source_polarity') and speaker_array.source_polarity[i]:
                                impulse = -impulse
                            
                            # Finde maximale Amplitude
                            current_max = np.max(np.abs(impulse))
                            max_amplitude = max(max_amplitude, current_max)
                            
                            # Speichere temporär
                            key = f"{speaker_array.name}_Speaker_{i+1}"
                            temp_responses[key] = {
                                'impulse': impulse,
                                'time': time,
                                'color': speaker_array.color[i] if isinstance(speaker_array.color, list) else speaker_array.color,
                                'speaker_idx': speaker_idx,
                                'angle_idx': int(round(azimuth)) % 360,  # Verwende azimuth als angle_idx
                                'elevation': elevation,
                                'distance': distance
                            }
                        except Exception as e:
                            print(f"Fehler bei der Berechnung der Impulsantwort für {speaker_name}: {e}")
                            continue
                
                # Zweite Schleife: Normalisiere alle Impulsantworten gemeinsam
                if temp_responses:  # Prüfe ob Responses vorhanden sind
                    for key, response in temp_responses.items():
                        if max_amplitude > 0:
                            response['impulse'] = response['impulse'] / max_amplitude
                        speaker_responses[key] = response
                    
                    impulse_responses[point_key['key']] = {
                        'speaker_responses': speaker_responses,
                        'position': (point_x, point_y)
                    }
            
            return impulse_responses if impulse_responses else None


    def calculate_individual_impulse_responses(self, impulse_responses, balloon_cache=None):
        """Berechnet die individuellen Impulsantworten für jeden Lautsprecher und Punkt"""
        if not impulse_responses:
            return None
        
        individual_impulse_data = {}
        
        # Schleife über alle Impulspunkte
        for point_key in self.settings.impulse_points:
            key = point_key['key']
            
            if key not in impulse_responses:
                continue
            
            responses = impulse_responses[key]['speaker_responses']
            if not responses:  # Prüfe ob Responses vorhanden sind
                continue
            
            impulse_data = {
                "spl": [],
                "time": [],
                "color": []
            }
            
            # Finde den maximalen SPL über alle Lautsprecher für die Normierung
            max_spl = float('-inf')
            for response in responses.values():
                if response['impulse'].size > 0:
                    max_spl = max(max_spl, np.max(np.abs(response['impulse'])))
            
            # Verarbeite jeden Lautsprecher
            for speaker_name, response in responses.items():
                if response['impulse'].size > 0:
                    # Array settings
                    array_name = speaker_name.split('_Speaker_')[0]
                    speaker_number = int(speaker_name.split('_Speaker_')[1]) - 1
                    array_id = next((id for id, array in self.settings.get_all_speaker_arrays().items() 
                                   if array.name == array_name), None)
                    if array_id is None:
                        continue
                    
                    speaker_array = self.settings.get_speaker_array(array_id)
                    
                    # Normalisierte Impulsantwort
                    normalized_impulse = response['impulse'] / max_spl
                    impulse_data["spl"].append(normalized_impulse.tolist())
                    impulse_data["time"].append(response['time'].tolist())
                    impulse_data["color"].append(speaker_array.color)
            
            individual_impulse_data[key] = impulse_data
        
        for key in individual_impulse_data:
            for i, (spl, time) in enumerate(zip(
                individual_impulse_data[key]['spl'],
                individual_impulse_data[key]['time'])):
                peak_idx = np.argmax(np.abs(spl))
        
        return individual_impulse_data if individual_impulse_data else None


    def calculate_phase_response(self, impulse_responses, balloon_cache=None):
        """Berechnet die Phasenresponse für alle Lautsprecher und Punkte"""
        if not impulse_responses:
            return None

        speaker_arrays = self._get_speaker_array_lookup()
        phase_data = {}

        for point_key in self.settings.impulse_points:
            key = point_key['key']

            if key not in impulse_responses:
                continue

            responses = impulse_responses[key]['speaker_responses']
            if not responses:
                continue

            entries = []
            for speaker_name, response in responses.items():
                if response['impulse'].size == 0:
                    continue

                array_name = speaker_name.split('_Speaker_')[0]
                speaker_array = speaker_arrays.get(array_name)
                if speaker_array is None:
                    continue

                speaker_number = int(speaker_name.split('_Speaker_')[1]) - 1

                if balloon_cache:
                    freq_data = self._get_cached_balloon_data(
                        balloon_cache,
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                else:
                    freq_data = None

                if freq_data is None:
                    freq_data = self.get_frequency_data(
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                    if balloon_cache is not None:
                        balloon_cache[(response['speaker_idx'], response['angle_idx'], response.get('elevation', 0))] = freq_data

                if not freq_data:
                    continue

                freq = np.asarray(freq_data['freq'])
                mag = np.asarray(freq_data['mag'])
                phase_deg = np.asarray(freq_data['phase'])

                if freq.size == 0:
                    continue

                array_gain = getattr(speaker_array, 'gain', 0.0)
                speaker_gain = speaker_array.source_level[speaker_number]
                total_gain = array_gain + speaker_gain

                array_delay = getattr(speaker_array, 'delay', 0.0)
                speaker_delay = speaker_array.source_time[speaker_number]
                total_delay = array_delay + speaker_delay

                polarity_list = getattr(speaker_array, 'source_polarity', None)
                polarity_value = False
                if polarity_list is not None and len(polarity_list) > speaker_number:
                    polarity_value = bool(np.asarray(polarity_list)[speaker_number])
                polarity = -1.0 if polarity_value else 1.0

                entries.append({
                    'freq': freq,
                    'mag': mag,
                    'phase_deg': phase_deg,
                    'distance': response.get('distance', 1.0),
                    'total_gain': total_gain,
                    'total_delay': total_delay,
                    'time_offset': point_key.get('time_offset', 0.0),
                    'polarity': polarity,
                    'color': speaker_array.color
                })

            if not entries:
                continue

            min_len = min(entry['freq'].shape[0] for entry in entries)
            if min_len == 0:
                continue

            common_freq = entries[0]['freq'][:min_len].astype(float, copy=True)
            mag_stack = np.stack([entry['mag'][:min_len] for entry in entries])
            phase_rad_stack = np.deg2rad(np.stack([entry['phase_deg'][:min_len] for entry in entries]))

            distance_array = np.array([entry['distance'] for entry in entries], dtype=float)[:, None]
            gain_linear = 10 ** (np.array([entry['total_gain'] for entry in entries], dtype=float)[:, None] / 20.0)
            delay_ms = np.array(
                [entry['total_delay'] - entry['time_offset'] for entry in entries],
                dtype=float
            )[:, None]
            polarity_array = np.array([entry['polarity'] for entry in entries], dtype=float)[:, None]

            distance_safe = np.maximum(distance_array, 1e-9)
            mag_with_gain = (mag_stack / distance_safe) * gain_linear

            freq_row = common_freq[None, :]
            delay_phase = 2 * np.pi * freq_row * (delay_ms / 1000.0)
            distance_phase = 2 * np.pi * freq_row * (distance_array / speed_of_sound)
            total_phase = phase_rad_stack - delay_phase - distance_phase

            complex_contribution = polarity_array * mag_with_gain * np.exp(1j * total_phase)
            complex_sum = np.sum(complex_contribution, axis=0)

            phase_response = {
                "freq": [],
                "phase": [],
                "color": [],
                "combined_phase": None,
                "complex_spectrum": complex_sum
            }

            for idx, entry in enumerate(entries):
                phase_response["freq"].append(common_freq.tolist())
                phase_response["phase"].append(np.angle(complex_contribution[idx]).tolist())
                phase_response["color"].append(entry['color'])

            combined_phase = np.angle(complex_sum)
            phase_response["combined_phase"] = {
                "freq": common_freq.tolist(),
                "phase": combined_phase.tolist()
            }

            phase_data[key] = phase_response

        return phase_data if phase_data else None


    def calculate_magnitude_response(self, impulse_responses, balloon_cache=None):
        """Berechnet Magnitude und Phase für jeden Lautsprecher und die Summe"""
        if not impulse_responses:
            return None

        speaker_arrays = self._get_speaker_array_lookup()
        magnitude_data = {}

        for point_key in self.settings.impulse_points:
            key = point_key['key']

            if key not in impulse_responses:
                continue

            responses = impulse_responses[key]['speaker_responses']
            if not responses:
                continue

            entries = []
            for speaker_name, response in responses.items():
                array_name = speaker_name.split('_Speaker_')[0]
                speaker_array = speaker_arrays.get(array_name)
                if speaker_array is None:
                    continue

                speaker_number = int(speaker_name.split('_Speaker_')[1]) - 1

                if balloon_cache:
                    freq_data = self._get_cached_balloon_data(
                        balloon_cache,
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                else:
                    freq_data = None

                if freq_data is None:
                    freq_data = self.get_frequency_data(
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                    if balloon_cache is not None:
                        balloon_cache[(response['speaker_idx'], response['angle_idx'], response.get('elevation', 0))] = freq_data

                if not freq_data:
                    continue

                freq = np.asarray(freq_data['freq'])
                mag = np.asarray(freq_data['mag'])
                phase_deg = np.asarray(freq_data['phase'])

                if freq.size == 0:
                    continue

                array_gain = getattr(speaker_array, 'gain', 0.0)
                speaker_gain = speaker_array.source_level[speaker_number]
                total_gain = array_gain + speaker_gain

                array_delay = getattr(speaker_array, 'delay', 0.0)
                speaker_delay = speaker_array.source_time[speaker_number]
                total_delay = array_delay + speaker_delay

                polarity_list = getattr(speaker_array, 'source_polarity', None)
                polarity_value = False
                if polarity_list is not None and len(polarity_list) > speaker_number:
                    polarity_value = bool(np.asarray(polarity_list)[speaker_number])
                polarity = -1.0 if polarity_value else 1.0

                entries.append({
                    'name': speaker_name,
                    'freq': freq,
                    'mag': mag,
                    'phase_deg': phase_deg,
                    'distance': response.get('distance', 1.0),
                    'total_gain': total_gain,
                    'total_delay': total_delay,
                    'time_offset': point_key.get('time_offset', 0.0),
                    'polarity': polarity,
                    'color': speaker_array.color
                })

            if not entries:
                continue

            min_len = min(entry['freq'].shape[0] for entry in entries)
            if min_len == 0:
                continue

            common_freq = entries[0]['freq'][:min_len].astype(float, copy=True)
            mag_stack = np.stack([entry['mag'][:min_len] for entry in entries])
            phase_rad_stack = np.deg2rad(np.stack([entry['phase_deg'][:min_len] for entry in entries]))

            distance_array = np.array([entry['distance'] for entry in entries], dtype=float)[:, None]
            gain_linear = 10 ** (np.array([entry['total_gain'] for entry in entries], dtype=float)[:, None] / 20.0)
            delay_ms = np.array(
                [entry['total_delay'] - entry['time_offset'] for entry in entries],
                dtype=float
            )[:, None]
            polarity_array = np.array([entry['polarity'] for entry in entries], dtype=float)[:, None]

            distance_safe = np.maximum(distance_array, 1e-9)
            mag_with_gain = (mag_stack / distance_safe) * gain_linear

            freq_row = common_freq[None, :]
            delay_phase = 2 * np.pi * freq_row * (delay_ms / 1000.0)
            distance_phase = 2 * np.pi * freq_row * (distance_array / speed_of_sound)
            total_phase = phase_rad_stack + delay_phase + distance_phase

            complex_response = polarity_array * mag_with_gain * np.exp(1j * total_phase)
            total_complex = np.sum(complex_response, axis=0)

            individual_data = {}
            for idx, entry in enumerate(entries):
                mag_db = 20 * np.log10(np.abs(complex_response[idx]) + 1e-10)
                individual_data[entry['name']] = {
                    'magnitude': mag_db.tolist(),
                    'phase': np.angle(complex_response[idx]).tolist(),
                    'color': entry['color']
                }

            total_magnitude = 20 * np.log10(np.abs(total_complex) + 1e-10)

            magnitude_data[key] = {
                'frequency': common_freq.tolist(),
                'magnitude': total_magnitude.tolist(),
                'phase': np.angle(total_complex).tolist(),
                'individual_magnitudes': individual_data
            }

        return magnitude_data if magnitude_data else None


    def calculate_arrival_times(self, impulse_responses, balloon_cache=None):
        """Berechnet die Arrival Times für alle Lautsprecher und Punkte"""
        if not impulse_responses:
            return None

        speaker_arrays = self._get_speaker_array_lookup()
        arrival_times_data = {}

        for point_key in self.settings.impulse_points:
            key = point_key['key']

            if key not in impulse_responses:
                continue

            responses = impulse_responses[key]['speaker_responses']
            if not responses:  # Prüfe ob Responses vorhanden sind
                continue

            entries = []
            point_x, point_y = point_key['data']
            time_offset = point_key.get('time_offset', 0.0)

            for speaker_name, response in responses.items():
                array_name = speaker_name.split('_Speaker_')[0]
                speaker_array = speaker_arrays.get(array_name)
                if speaker_array is None:
                    continue

                speaker_number = int(speaker_name.split('_Speaker_')[1]) - 1

                if balloon_cache:
                    freq_data = self._get_cached_balloon_data(
                        balloon_cache,
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                else:
                    freq_data = None

                if freq_data is None:
                    freq_data = self.get_frequency_data(
                        response['speaker_idx'],
                        response['angle_idx'],
                        response.get('elevation', 0)
                    )
                    if balloon_cache is not None:
                        balloon_cache[(response['speaker_idx'], response['angle_idx'], response.get('elevation', 0))] = freq_data

                source_position_x = getattr(
                    speaker_array,
                    'source_position_calc_x',
                    speaker_array.source_position_x,
                )[speaker_number]
                source_position_y = getattr(
                    speaker_array,
                    'source_position_calc_y',
                    speaker_array.source_position_y,
                )[speaker_number]

                array_gain = getattr(speaker_array, 'gain', 0.0)
                speaker_gain = speaker_array.source_level[speaker_number]
                total_gain = array_gain + speaker_gain

                array_delay = getattr(speaker_array, 'delay', 0.0)
                speaker_delay = speaker_array.source_time[speaker_number]
                total_delay = array_delay + speaker_delay

                color = speaker_array.color
                if isinstance(color, list):
                    color = color[speaker_number]

                entries.append({
                    'freq_data': freq_data,
                    'source_x': float(source_position_x),
                    'source_y': float(source_position_y),
                    'total_gain': float(total_gain),
                    'total_delay': float(total_delay),
                    'color': color
                })

            if not entries:
                continue

            source_x = np.array([entry['source_x'] for entry in entries], dtype=float)
            source_y = np.array([entry['source_y'] for entry in entries], dtype=float)
            total_gain = np.array([entry['total_gain'] for entry in entries], dtype=float)
            total_delay = np.array([entry['total_delay'] for entry in entries], dtype=float)

            x_distance = point_x - source_x
            y_distance = point_y - source_y
            distances = np.sqrt(x_distance ** 2 + y_distance ** 2)

            arrival_time = (distances / speed_of_sound) * 1000.0 - time_offset + total_delay

            spl_values = []
            for entry, distance, gain in zip(entries, distances, total_gain):
                freq_data = entry['freq_data']
                if freq_data and freq_data['mag'] is not None:
                    magnitude_linear = np.asarray(freq_data['mag'], dtype=float)
                    if magnitude_linear.size:
                        intensity = np.square(magnitude_linear)
                        total_intensity = np.sum(intensity)
                        total_spl = 10 * np.log10(total_intensity + 1e-12)
                        source_spl = total_spl + gain
                        distance_attenuation = -20 * np.log10(max(distance, 1e-6))
                        spl = source_spl + distance_attenuation
                    else:
                        spl = 0.0
                else:
                    spl = 0.0
                spl_values.append(spl)

            arrival_data = {
                "time": arrival_time.tolist(),
                "spl_max": spl_values,
                "spl_min": [spl - 40 for spl in spl_values],
                "color": [entry['color'] for entry in entries]
            }

            arrival_times_data[key] = arrival_data

        return arrival_times_data if arrival_times_data else None


    def calculate_combined_impulse_response(self, impulse_responses, phase_data, balloon_cache=None):
        """Berechnet die Summenimpulsantwort für alle Punkte"""
        if not impulse_responses or not phase_data:
            return None
        
        combined_impulse_data = {}
        
        for point in self.settings.impulse_points:
            key = point['key']
            
            if key not in impulse_responses or key not in phase_data:
                continue
            
            # Prüfe, ob 'speaker_responses' oder 'impulse_per_speaker' verwendet wird
            if 'speaker_responses' in impulse_responses[key]:
                responses = impulse_responses[key]['speaker_responses']
            elif 'impulse_per_speaker' in impulse_responses[key]:
                responses = impulse_responses[key]['impulse_per_speaker']
            else:
                continue
            
            if not responses:
                continue
            
            # Nutze die übergebenen Phasendaten
            complex_spectrum = phase_data[key]["complex_spectrum"]
            if complex_spectrum is None:
                continue
            
            # FFT-Parameter
            first_response = next(iter(responses.values()))
            speaker_idx = first_response.get('speaker_idx', 0)
            angle_idx = first_response.get('angle_idx', 0)
            
            # Verwende Cache wenn verfügbar, sonst normale Methode
            if balloon_cache:
                freq_data = self._get_cached_balloon_data(
                    balloon_cache, 
                    speaker_idx, 
                    angle_idx, 
                    first_response.get('elevation', 0)
                )
            else:
                freq_data = self.get_frequency_data(speaker_idx, angle_idx, first_response.get('elevation', 0))
            
            if not freq_data:
                continue
            
            # Stelle sicher, dass die Arrays die gleiche Länge haben
            if len(freq_data['freq']) != len(complex_spectrum):
                # Nutze die kürzere Länge
                min_length = min(len(freq_data['freq']), len(complex_spectrum))
                freq = freq_data['freq'][:min_length]
                complex_spec = complex_spectrum[:min_length]
            else:
                freq = freq_data['freq']
                complex_spec = complex_spectrum
            
            max_freq = max(freq)
            sample_rate = max_freq * 2
            fft_size = len(freq) * 2 - 2

            # Interpolation zu linearem Frequenzabstand für korrekte IFFT
            n_interp = fft_size // 2 + 1
            target_freq = np.linspace(freq[0], freq[-1], n_interp)
            
            # Interpoliere Real- und Imaginärteil separat
            real_interp = interpolate.interp1d(freq, complex_spec.real, kind='linear')(target_freq)
            imag_interp = interpolate.interp1d(freq, complex_spec.imag, kind='linear')(target_freq)
            H = real_interp + 1j * imag_interp
            
            # IFFT
            impulse = np.fft.irfft(H, n=fft_size)
            
            # Normierung
            impulse = impulse / np.max(np.abs(impulse))
            
            # Einfache Zeitachse ohne zusätzliche Delays
            dt = 1/sample_rate * 1000
            time = np.arange(0, fft_size) * dt
            
            combined_impulse_data[key] = {
                "impulse": impulse,
                "time": time
            }
        
        return combined_impulse_data if combined_impulse_data else None
    
    def set_data_container(self, data_container):
        """🚀 Setzt den optimierten DataContainer für Performance-Zugriff"""
        self._data_container = data_container
        self._speaker_name_list = None
        self._speaker_name_index = None

    def _preload_balloon_data(self, impulse_responses):
        """Lädt alle benötigten Balloondaten vorab und speichert sie zwischen"""
        # Cache für Balloondaten
        balloon_cache = {}
        
        # Cache für Speaker-Namen (Performance-Optimierung)
        speaker_names = self._get_speaker_names_list()
        
        # Sammle alle benötigten Speaker/Winkel-Kombinationen
        for point_key in self.settings.impulse_points:
            key = point_key['key']
            
            if key not in impulse_responses:
                continue
                
            responses = impulse_responses[key]['speaker_responses']
            if not responses:
                continue
            
            for response in responses.values():
                speaker_idx = response['speaker_idx']
                angle_idx = response['angle_idx']
                elevation = response.get('elevation', 0)
                
                # Erstelle Cache-Key
                cache_key = (speaker_idx, angle_idx, elevation)
                
                if cache_key not in balloon_cache:
                    # Lade Balloondaten nur einmal
                    speaker_name = speaker_names[speaker_idx]
                    
                    # Verwende get_frequency_data für konsistente Datenstruktur
                    freq_data = self.get_frequency_data(speaker_idx, angle_idx, elevation)
                    balloon_cache[cache_key] = freq_data
        
        return balloon_cache

    def _get_cached_balloon_data(self, balloon_cache, speaker_idx, angle_idx, elevation=0):
        """Holt Balloondaten aus dem Cache"""
        cache_key = (speaker_idx, angle_idx, elevation)
        return balloon_cache.get(cache_key)

    def _interpolate_balloon_data_with_elevation(self, balloon_mag, balloon_phase, balloon_freqs,
                                                  balloon_vertical_angles, azimuth, elevation):
        """
        Interpoliert Balloon-Daten mit Elevation-Interpolation
        
        Args:
            balloon_mag: Magnitude-Daten (shape: (vertical_angles, horizontal_angles, frequencies))
            balloon_phase: Phase-Daten in GRAD, UNWRAPPED (gleiche Form wie magnitude)
            balloon_freqs: Verfügbare Frequenzen
            balloon_vertical_angles: Array der vertikalen Winkel
            azimuth: Azimut-Winkel in Grad (0-360)
            elevation: Elevations-Winkel in Grad
            
        Returns:
            dict: {'freq': freqs, 'mag': mag_values in dB, 'phase': phase_values in GRAD} oder None
            
        Note:
            Phase wird als unwrapped (in Grad) erwartet und kann daher linear interpoliert werden.
            Bei Verwendung in np.exp() muss die Phase mit np.radians() konvertiert werden.
        """
        try:
            # Normalisiere Azimut auf 0-360 Grad
            azimuth = azimuth % 360
            
            # Finde nächsten horizontalen Winkel (gerundet auf ganze Zahl)
            h_idx = int(round(azimuth)) % 360
            
            # Interpolation für vertikale Winkel
            if elevation <= balloon_vertical_angles[0]:
                # Elevation ist kleiner als der kleinste verfügbare Winkel
                v_idx = 0
                mag_values = balloon_mag[v_idx, h_idx, :]
                phase_values = balloon_phase[v_idx, h_idx, :]
                
            elif elevation >= balloon_vertical_angles[-1]:
                # Elevation ist größer als der größte verfügbare Winkel
                v_idx = len(balloon_vertical_angles) - 1
                mag_values = balloon_mag[v_idx, h_idx, :]
                phase_values = balloon_phase[v_idx, h_idx, :]
                
            else:
                # Elevation liegt zwischen zwei verfügbaren Winkeln - interpoliere
                # Finde die Indizes der umgebenden Winkel
                v_idx_lower = np.where(balloon_vertical_angles <= elevation)[0][-1]
                v_idx_upper = np.where(balloon_vertical_angles >= elevation)[0][0]
                
                if v_idx_lower == v_idx_upper:
                    # Exakter Treffer
                    mag_values = balloon_mag[v_idx_lower, h_idx, :]
                    phase_values = balloon_phase[v_idx_lower, h_idx, :]
                else:
                    # Lineare Interpolation zwischen den beiden Winkeln
                    angle_lower = balloon_vertical_angles[v_idx_lower]
                    angle_upper = balloon_vertical_angles[v_idx_upper]
                    
                    # Interpolationsfaktor (0 = unterer Winkel, 1 = oberer Winkel)
                    t = (elevation - angle_lower) / (angle_upper - angle_lower)
                    
                    # Hole Werte für alle Frequenzen
                    mag_lower = balloon_mag[v_idx_lower, h_idx, :]
                    mag_upper = balloon_mag[v_idx_upper, h_idx, :]
                    phase_lower = balloon_phase[v_idx_lower, h_idx, :]
                    phase_upper = balloon_phase[v_idx_upper, h_idx, :]
                    
                    # Interpoliere Magnitude (in dB - linear interpolierbar)
                    mag_values = mag_lower + t * (mag_upper - mag_lower)
                    
                    # Interpoliere Phase (unwrapped - direkt linear interpolieren)
                    # Da Phase bereits unwrapped ist, keine zirkuläre Korrektur nötig
                    phase_values = phase_lower + t * (phase_upper - phase_lower)
            
            return {
                'freq': balloon_freqs,
                'mag': mag_values,
                'phase': phase_values
            }
            
        except (IndexError, ValueError) as e:
            print(f"❌ Fehler bei Elevation-Interpolation: {e}")
            return None

    def _get_speaker_names_list(self):
        """Liefert eine gecachte Liste aller Lautsprechernamen."""
        if self._speaker_name_list is None:
            if self._data_container:
                try:
                    self._speaker_name_list = list(self._data_container.get_speaker_names())
                except Exception:
                    self._speaker_name_list = []
            else:
                self._speaker_name_list = []
        return self._speaker_name_list

    def _get_speaker_name_index(self):
        """Liefert ein Mapping von Lautsprechernamen zu ihren Indizes."""
        if self._speaker_name_index is None:
            speaker_names = self._get_speaker_names_list()
            self._speaker_name_index = {name: idx for idx, name in enumerate(speaker_names)}
        return self._speaker_name_index

    def _get_speaker_array_lookup(self):
        """Gibt ein Mapping von Array-Namen zu den Array-Objekten zurück (gecached)."""
        if self._speaker_array_cache is None:
            arrays = self.settings.get_all_speaker_arrays()
            self._speaker_array_cache = {array.name: array for array in arrays.values()}
        return self._speaker_array_cache


class _PerfScope:
    def __init__(self, owner: "ImpulseCalculator", label: str):
        self._owner = owner
        self._label = label
        self._start = 0.0

    def __enter__(self):
        if getattr(self._owner, "_perf_enabled", False):
            self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self._owner, "_perf_enabled", False):
            duration = perf_counter() - self._start
            self._owner._log_perf(self._label, duration)
