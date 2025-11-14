# Setting State enthält default variablen. Diese werden bei unverändertem UI-Inpult verwendet
import numpy as np
import copy

from Module_LFO.Modules_Calculate.Functions import FunctionToolbox




class SpeakerArray:
    def __init__(self, id, name="", container=None):        
        self.id = id
        self.name = "Array"
        self.mute = False
        self.hide = False
        self.number_of_sources = 1
        self.gain = 0
        self.delay = 0
        
        # Verwende den ersten verfügbaren Lautsprecher aus dem Container
        if container and 'speaker_names' in container.data and container.data['speaker_names']:
            default_speaker = container.data['speaker_names'][0]
        else:
            default_speaker = ''
            
        self.source_polar_pattern = np.array([default_speaker], dtype=object)
        self.source_length = 0
        self.source_position_x = np.array([0.0])
        self.source_position_y = np.array([0.0])
        self.source_position_z = np.array([0.0])
        self.source_position_z_stack = np.array([0.0])
        self.source_position_z_flown = np.array([0.0])
        self.source_azimuth = np.array([0.0])
        self.source_site = np.array([0.0])
        self.source_angle = np.array([0.0])
        self.source_level = np.array([0.0])
        self.source_time = np.array([0.0])
        self.source_polarity = np.array([False], dtype=bool)  # False = normal, True = inverted
        self.virtual_source_position_x = np.array([0.0])
        self.virtual_source_position_y = np.array([0.0])
        self.arc_angle = 60
        self.arc_shape = "Manual"
        self.arc_scale_factor = 0.32
        self.window_function = "tukey"
        self.alpha = 0.6
        self.window_restriction = -9.8
        self.source_lp_zero = np.array([0.0])
        self.defined_source_gains = [0, -2.9, -9.8, -12.6]
        self.tree_items = []
        self.selected_item = None
        self.color = self._generate_random_color()
        self.default_surface_id = "surface_default"
        self.surfaces = {
            self.default_surface_id: {
                "name": "Default Surface",
                "enabled": False,
                "hidden": False,
                "points": [
                    {"x": 75.0, "y": -50.0, "z": 0.0},
                    {"x": 75.0, "y": 50.0, "z": 0.0},
                    {"x": -75.0, "y": 50.0, "z": 0.0},
                    {"x": -75.0, "y": -50.0, "z": 0.0},
                ],
                "locked": True,
            }
        }


    def _generate_random_color(self):
        # Generiert eine helle, pastellige Farbe
        hue = np.random.random()  # Zufälliger Farbton
        saturation = 0.3 + 0.2 * np.random.random()  # Mittlere Sättigung
        value = 0.9 + 0.1 * np.random.random()  # Hohe Helligkeit
        
        # Konvertiere HSV zu RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Konvertiere zu QColor-kompatibler Hex-Farbe
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    def update_sources(self, number_of_sources):
        self.number_of_sources = number_of_sources
        
        # Finde den ersten gültigen Lautsprechertyp (nicht leerer String)
        default_speaker = None
        for speaker in self.source_polar_pattern:
            if speaker and speaker.strip():
                default_speaker = speaker
                break
        
        # Wenn kein gültiger Lautsprecher gefunden wurde, setze einen Default-Wert
        if not default_speaker:
            default_speaker = self.source_polar_pattern[0] if len(self.source_polar_pattern) > 0 else "Default"
        
        # Verwende diesen als default_value beim Resize
        self.source_polar_pattern = self._resize_array(self.source_polar_pattern, number_of_sources, default_speaker)
        self.source_position_x = self._resize_array(self.source_position_x, number_of_sources, 0.0)
        self.source_position_y = self._resize_array(self.source_position_y, number_of_sources, 0.0)
        self.source_position_z = self._resize_array(self.source_position_z, number_of_sources, 0.0)
        # Ergänze source_position_z_stack und source_position_z_flown
        if hasattr(self, 'source_position_z_stack'):
            self.source_position_z_stack = self._resize_array(self.source_position_z_stack, number_of_sources, 0.0)
        if hasattr(self, 'source_position_z_flown'):
            self.source_position_z_flown = self._resize_array(self.source_position_z_flown, number_of_sources, 0.0)
        self.source_azimuth = self._resize_array(self.source_azimuth, number_of_sources, 0.0)
        self.source_time = self._resize_array(self.source_time, number_of_sources, 0.0)
        self.source_level = self._resize_array(self.source_level, number_of_sources, 0.0)
        
        # Initialisiere source_polarity falls nicht vorhanden (für alte Speaker Arrays)
        if not hasattr(self, 'source_polarity'):
            self.source_polarity = np.array([False] * number_of_sources, dtype=bool)
        else:
            self.source_polarity = self._resize_array(self.source_polarity, number_of_sources, False)
        
        self.source_lp_zero = self._resize_array(self.source_lp_zero, number_of_sources, 0.0)
        self.virtual_source_position_x = self._resize_array(self.virtual_source_position_x, number_of_sources, 0.0)
        self.virtual_source_position_y = self._resize_array(self.virtual_source_position_y, number_of_sources, 0.0)


    def update_source_length(self, source_length):
        self.source_length = source_length

    def _resize_array(self, arr, new_size, default_value):
        if len(arr) < new_size:
            arr = np.append(arr, np.full(new_size - len(arr), default_value))
        elif len(arr) > new_size:
            arr = arr[:new_size]
        return arr 
    
    
    def to_dict(self):
        return vars(self)

    def from_dict(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class Settings:
    DEFAULT_SURFACE_ID = "surface_default"

    def __init__(self):
        self.surface_definitions = self._initialize_surface_definitions()
        self.active_surface_id = self.DEFAULT_SURFACE_ID
        self.load_custom_defaults()
        self.speaker_arrays = {}
        self.speaker_array_names = {}

    def load_speaker_defaults(self):
        self.speaker_array = self.speaker_arrays = {}

    def load_custom_defaults(self):
        # Temperatur und Luftfeuchtigkeit zuerst definieren
        temperature = 20.0
        humidity = 50.0
        
        # 🌡️ Berechne Schallgeschwindigkeit basierend auf Temperatur
        speed_of_sound = 331.3 + 0.606 * temperature
        
        defaults = {            
            'impulse_min_spl': -20,
            'measurement_size': 4,
            'impulse_plot_height': 180,
            'resolution': 1,
            'speed_of_sound': speed_of_sound,
            'temperature': temperature,
            'humidity': humidity,
            'air_density': 1.204,  # kg/m³ bei 20°C und 50% Luftfeuchtigkeit
            'upper_calculate_frequency': 56,
            'lower_calculate_frequency': 45,
            'calculate_frequency': 50.2,    
            'colorization_mode': "Color step",
            'a_source_db': 94,
            'position_x_axis': 0,
            'position_y_axis': 30,
            'freq_index': 3,
            'window_resolution': 10000,
            'window_std': 1000,
            'colorbar_range': {
                'min': 90,
                'max': 120,
                'step': 3,
                'tick_step': 6
            },
            'impulse_points': [],
            'polar_frequencies': {
                'red': 31.5,
                'yellow': 40,
                'green': 50,
                'cyan': 63
            },
            'polar_colors': ['red', 'yellow', 'green', 'cyan'],
            'polar_min_db': -30,
            'polar_max_db': 0,
            'spl_plot_superposition': True,
            'spl_plot_fem': False,
            'xaxis_plot_superposition': True,
            'xaxis_plot_fem': False,
            'yaxis_plot_superposition': True,
            'yaxis_plot_fem': False,
            'polar_plot_superposition': True,
            'polar_plot_fem': False,
            'impulse_plot_superposition': True,
            'impulse_plot_bem': False,
            'update_pressure_soundfield': True,
            'update_pressure_axisplot': True,
            'update_pressure_polarplot': True,
            'update_pressure_impulse': True,
        }
        for key, value in defaults.items():
            setattr(self, key, value)

        self.update_surface_dimensions()




    def to_dict(self):
        return {
            'impulse_min_spl' : self.impulse_min_spl,
            'measurement_size' : self.measurement_size,
            'impulse_plot_height': self.impulse_plot_height,
            'resolution': self.resolution,
            'speed_of_sound': self.speed_of_sound,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'air_density': self.air_density,
            'calculate_frequency': self.calculate_frequency,
            'upper_calculate_frequency': self.upper_calculate_frequency,
            'lower_calculate_frequency': self.lower_calculate_frequency,
            'colorization_mode': self.colorization_mode,
            'a_source_db': self.a_source_db,
            'position_x_axis': self.position_x_axis,
            'position_y_axis': self.position_y_axis,
            'freq_index': self.freq_index,
            'window_resolution': self.window_resolution,
            'window_std': self.window_std,
            'colorbar_range': self.colorbar_range,
            'impulse_points': self.impulse_points,
            'polar_frequencies': self.polar_frequencies,
            'polar_colors': self.polar_colors,
            'polar_min_db': self.polar_min_db,
            'polar_max_db': self.polar_max_db,
            'width': getattr(self, 'width', 0.0),
            'length': getattr(self, 'length', 0.0),
            'surface_definitions': self.surface_definitions,
            'active_surface_id': self.active_surface_id,
            'spl_plot_superposition': self.spl_plot_superposition,
            'spl_plot_fem': self.spl_plot_fem,
            'xaxis_plot_superposition': self.xaxis_plot_superposition,
            'xaxis_plot_fem': self.xaxis_plot_fem,
            'yaxis_plot_superposition': self.yaxis_plot_superposition,
            'yaxis_plot_fem': self.yaxis_plot_fem,
            'polar_plot_superposition': self.polar_plot_superposition,
            'polar_plot_fem': self.polar_plot_fem,
            'impulse_plot_superposition': self.impulse_plot_superposition,
            'impulse_plot_bem': self.impulse_plot_bem,
            'update_pressure_soundfield': self.update_pressure_soundfield,
            'update_pressure_axisplot': self.update_pressure_axisplot,
            'update_pressure_polarplot': self.update_pressure_polarplot,
            'update_pressure_impulse': self.update_pressure_impulse,
        }

    def set_active_surface(self, surface_id):
        if surface_id not in self.surface_definitions:
            raise KeyError(f"Surface with ID '{surface_id}' not found.")
        self.active_surface_id = surface_id
        self.update_surface_dimensions()

    def get_active_surface(self):
        return self.surface_definitions.get(self.active_surface_id)

    def add_surface_definition(self, surface_id, surface_data, make_active=False):
        self.surface_definitions[surface_id] = surface_data
        if make_active:
            self.set_active_surface(surface_id)
        if surface_id == self.DEFAULT_SURFACE_ID:
            self.update_surface_dimensions()

    def remove_surface_definition(self, surface_id):
        if surface_id == self.DEFAULT_SURFACE_ID:
            raise ValueError("Default surface cannot be removed.")
        if surface_id in self.surface_definitions:
            was_active = surface_id == self.active_surface_id
            del self.surface_definitions[surface_id]
            if was_active:
                if self.surface_definitions:
                    fallback_surface = next(iter(self.surface_definitions))
                    self.active_surface_id = fallback_surface
                else:
                    self.active_surface_id = None
                    self.width = 0.0
                    self.length = 0.0

    def update_surface_dimensions(self):
        surface = self.surface_definitions.get(self.DEFAULT_SURFACE_ID)
        if not surface:
            self.width = 0.0
            self.length = 0.0
            return

        width, length = FunctionToolbox.surface_dimensions(surface.get("points", []))
        self.width = width
        self.length = length

    def set_surface_enabled(self, surface_id, enabled):
        surface = self.surface_definitions.get(surface_id)
        if surface is None:
            raise KeyError(f"Surface with ID '{surface_id}' not found.")
        surface["enabled"] = bool(enabled)

    def is_surface_enabled(self, surface_id):
        surface = self.surface_definitions.get(surface_id)
        return bool(surface and surface.get("enabled"))

    def is_default_surface_enabled(self):
        return self.is_surface_enabled(self.DEFAULT_SURFACE_ID)

    def add_speaker_array(self, id, name="", container=None):
        if id not in self.speaker_arrays:
            # Container weitergeben an SpeakerArray
            self.speaker_arrays[id] = SpeakerArray(id, name, container)
        else:
            print(f"SpeakerArray with id {id} already exists.")

    def get_speaker_array(self, id):
        speaker_array = self.speaker_arrays.get(id, None)
        return speaker_array

    def get_all_speaker_arrays(self):
        return self.speaker_arrays

    def remove_speaker_array(self, id):
        if id in self.speaker_arrays:
            del self.speaker_arrays[id]
        else:
            print(f"No SpeakerArray found with ID {id}.")

    def update_speaker_array_id(self, old_id, new_id):
        if old_id in self.speaker_arrays:
            self.speaker_arrays[new_id] = self.speaker_arrays.pop(old_id)
        else:
            print(f"No SpeakerArray found with ID {old_id}")

    def get_all_speaker_array_ids(self):
        return self.speaker_arrays.keys()

    def set_speaker_array_name(self, id, name):
        if name in self.speaker_array_names.values():
            raise ValueError(f"Name '{name}' is already in use.")
        self.speaker_array_names[id] = name

    def get_speaker_array_name(self, id):
        return self.speaker_array_names.get(id, None)

    def update_sources(self, id, number_of_sources):
        speaker_array = self.get_speaker_array(id)
        if speaker_array:
            speaker_array.update_sources(number_of_sources)

    def update_number_of_sources(self, speaker_array_id, number_of_sources):
        if speaker_array_id in self.speaker_arrays:
            self.speaker_arrays[speaker_array_id].number_of_sources = number_of_sources
        else:
            print(f"SpeakerArray mit ID {speaker_array_id} nicht gefunden.")

    def update_source_length(self, speaker_array_id, source_length):
        if speaker_array_id in self.speaker_arrays:
            self.speaker_arrays[speaker_array_id].update_source_length(source_length)
        else:
            print(f"SpeakerArray mit ID {speaker_array_id} nicht gefunden.")

    def update_speaker_array_arc_settings(self, speaker_array_id, arc_angle, arc_shape, spiral_shape):
        if speaker_array_id in self.speaker_arrays:
            speaker_array = self.speaker_arrays[speaker_array_id]
            speaker_array.arc_angle = arc_angle
            speaker_array.arc_shape = arc_shape
            # speaker_array.smooth_factor = smooth_factor
            speaker_array.spiral_shape = spiral_shape
            # speaker_array.j_shape = j_shape
        else:
            print(f"SpeakerArray mit ID {speaker_array_id} nicht gefunden.")
            
    def update_speaker_array_beamsteering(self, speaker_array_id, source_time, virtual_source_position_x, virtual_source_position_y):
        speaker_array = self.get_speaker_array(speaker_array_id)
        if speaker_array:
            self.speaker_arrays[speaker_array_id].source_time = source_time
            self.speaker_arrays[speaker_array_id].virtual_source_position_x = virtual_source_position_x
            self.speaker_arrays[speaker_array_id].virtual_source_position_y = virtual_source_position_y
        else:
            print(f"SpeakerArray with id {speaker_array_id} not found.")    
            
    def update_speaker_array_window_settings(self, speaker_array_id, window_function, alpha, window_restriction):
        speaker_array = self.get_speaker_array(speaker_array_id)
        if speaker_array:
            self.speaker_arrays[speaker_array_id].window_function = window_function
            self.speaker_arrays[speaker_array_id].alpha = alpha
            self.speaker_arrays[speaker_array_id].window_restriction = window_restriction
        else:
            print(f"SpeakerArray with ID {speaker_array_id} not found.")           
            
    def update_speaker_array_delay(self, speaker_array_id, delay):
        speaker_array = self.get_speaker_array(speaker_array_id)
        if speaker_array:
            self.speaker_arrays[speaker_array_id].delay = delay
        else:
            print(f"SpeakerArray with ID {speaker_array_id} not found.")

    def update_speaker_array_gain(self, speaker_array_id, gain):
        speaker_array = self.get_speaker_array(speaker_array_id)
        if speaker_array:
            self.speaker_arrays[speaker_array_id].gain = gain
        else:
            print(f"SpeakerArray with ID {speaker_array_id} not found.")            
            
    def get_mute_state(self, array_id):
        return self.speaker_arrays[array_id].mute

    def get_hide_state(self, array_id):
        return self.speaker_arrays[array_id].hide

    def get_number_of_sources(self, speaker_array_id):
        return self.speaker_arrays[speaker_array_id].number_of_sources

    def load_custom_defaults_array(self):
        # Hier wird ein Standard-SpeakerArray mit ID 1 erstellt und hinzugefügt
        default_array = SpeakerArray(id=1, name="Default Array")
        self.speaker_arrays[1] = default_array

    def update_polar_frequency(self, color, frequency):
        """Aktualisiert eine einzelne Polar-Frequenz"""
        if color in self.polar_frequencies:
            self.polar_frequencies[color] = frequency
            # Signal emittieren für Änderung
            if hasattr(self, 'main_window'):
                self.main_window.calculate_polar()

    def duplicate_speaker_array(self, original_id, new_id):
        """
        Dupliziert ein Lautsprecherarray mit allen Einstellungen.
        
        Args:
            original_id: ID des zu kopierenden Arrays
            new_id: ID für das neue Array
        """
        original_array = self.get_speaker_array(original_id)
        if original_array:
            # Erstelle eine Kopie des original Arrays
            new_array = copy.deepcopy(original_array)
            new_array.id = new_id
            new_array.name = f"copy of {original_array.name}"
            
            # Füge das neue Array zum settings dict hinzu
            self.speaker_arrays[new_id] = new_array
            
            # Kopiere alle relevanten Einstellungen
            if hasattr(self, 'speaker_array_settings') and original_id in self.speaker_array_settings:
                self.speaker_array_settings[new_id] = copy.deepcopy(self.speaker_array_settings[original_id])

    def _initialize_surface_definitions(self):
        return {
            self.DEFAULT_SURFACE_ID: {
                "name": "Default Surface",
                "enabled": True,
                "hidden": False,
                "locked": True,
                "points": [
                    {"x": 75.0, "y": -50.0, "z": 0.0},
                    {"x": 75.0, "y": 50.0, "z": 0.0},
                    {"x": -75.0, "y": 50.0, "z": 0.0},
                    {"x": -75.0, "y": -50.0, "z": 0.0},
                ],
            }
        }