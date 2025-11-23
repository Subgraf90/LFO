from Module_LFO.Modules_Init.ModuleBase import ModuleBase
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class BeamSteering(ModuleBase):
    def __init__(self, speaker_array, data_container, settings):
        super().__init__(speaker_array)
        self.speaker_array = speaker_array
        self.data_container = data_container
        self.settings = settings

    def calculate(self, speaker_array_id):
        source_time, virtual_source_position_x, virtual_source_position_y = self.arc_select(speaker_array_id)
        # Speichere die berechneten Werte im entsprechenden SpeakerArray
        self.settings.update_speaker_array_beamsteering(
            speaker_array_id, source_time, virtual_source_position_x, virtual_source_position_y)


    def arc_select(self, speaker_array_id):
        speaker_array = self.settings.get_speaker_array(speaker_array_id)
        if speaker_array is None:
            raise ValueError(f"SpeakerArray with id {speaker_array_id} not found")
        
        arc_shape = speaker_array.arc_shape
        source_polar_pattern = speaker_array.source_polar_pattern
        source_length = speaker_array.source_length
        arc_angle = speaker_array.arc_angle
        # Verwende nur Speaker-Positionen ohne Array-Offset für Beamsteering-Berechnungen
        source_position_x = getattr(speaker_array, 'source_position_x', None)
        source_position_y = getattr(speaker_array, 'source_position_y', None)
        
        # 🌡️ Temperaturabhängige Schallgeschwindigkeit (wird in UiSettings berechnet)
        speed_of_sound = self.settings.speed_of_sound
        source_time = speaker_array.source_time
        
        arc_scale_factor = speaker_array.arc_scale_factor
        virtual_source_position_x = speaker_array.virtual_source_position_x
        virtual_source_position_y = speaker_array.virtual_source_position_y

    
        if arc_shape == "Manual":
            if len(source_time) == len(source_polar_pattern):
                if any(name != '' for name in source_polar_pattern):
                    return source_time, source_position_x, source_position_y
            if len(source_time) != len(source_polar_pattern):
                source_time += [0] * (len(source_polar_pattern) - len(source_time))
            
            virtual_source_position_y = np.array(source_time) / 2.9 * -1
            return source_time, source_position_x, virtual_source_position_y
    
        elif arc_shape == "Pointed Arc":
            source_arc_time, virtual_source_position_x, virtual_source_position_y = self.calculate_pointed_circular(source_position_x, source_position_y, source_length, arc_angle, speed_of_sound)
            return source_arc_time, virtual_source_position_x, virtual_source_position_y
    
        elif arc_shape == "Circular Arc":
            source_arc_time, virtual_source_position_x, virtual_source_position_y = self.calculate_circular(source_position_x, source_position_y, source_length, arc_angle, speed_of_sound)
            return source_arc_time, virtual_source_position_x, virtual_source_position_y
    
        elif arc_shape == "Linear":
            source_arc_time, virtual_source_position_x, virtual_source_position_y = self.calculate_linear(source_position_y, source_position_x, arc_angle, speed_of_sound)
            return source_arc_time, virtual_source_position_x, virtual_source_position_y
    
        elif arc_shape == "Spiral Arc":
            source_arc_time, virtual_source_position_x, virtual_source_position_y = self.calculate_spiral(source_position_x, source_position_y, source_length, speed_of_sound, arc_angle, arc_scale_factor)
            return source_arc_time, virtual_source_position_x, virtual_source_position_y
        
        elif arc_shape == "Eliptical Arc":
            source_arc_time, virtual_source_position_x, virtual_source_position_y = self.calculate_elliptical(source_position_x, source_position_y, source_length, speed_of_sound, arc_angle, arc_scale_factor)
            return source_arc_time, virtual_source_position_x, virtual_source_position_y
        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y


    def calculate_pointed_circular(self, source_position_x, source_position_y, source_length, arc_angle, speed_of_sound):
        if len(source_position_x) >= 2:
            arc_angle_rad = np.deg2rad(arc_angle)
            radius = (source_length / 2) / np.tan(arc_angle_rad / 2)
            vp_source_y = source_position_y + radius
            vp_source_x = np.mean(source_position_x)
            vp_source_term = np.sqrt((vp_source_y - source_position_y)**2 + (vp_source_x - source_position_x)**2)
            vp_source_del = vp_source_term / speed_of_sound
            source_arc_time = (vp_source_del - (min(vp_source_del))) * 1000
            source_arc_time_round = [round(x, 2) for x in source_arc_time]
            vp_source_dist = source_length / (2 * np.sin(arc_angle_rad / 2))
            source_angles = np.arcsin(source_position_x / vp_source_dist)
            virtual_source_position_y = radius * np.cos(source_angles)
            virtual_source_position_x = radius * np.sin(source_angles)
            virtual_source_position_y = virtual_source_position_y - max(virtual_source_position_y)
            return source_arc_time_round, virtual_source_position_x, virtual_source_position_y
        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y


    def calculate_circular(self, source_position_x, source_position_y, source_length, arc_angle, speed_of_sound):
        if len(source_position_x) >= 2:
            arc_angle = np.deg2rad(arc_angle)
            vp_source_dist = source_length / (2 * np.sin(arc_angle / 2))
            source_angles = np.arcsin(source_position_x / vp_source_dist)
            v_source_dist = vp_source_dist * (1 - np.cos(source_angles))
            vp_source_del = (v_source_dist + source_position_y) / speed_of_sound
            source_arc_time = (vp_source_del - (min(vp_source_del))) * 1000
            source_arc_time_round = [round(x, 2) for x in source_arc_time]
            virtual_source_position_y = -(v_source_dist - min(v_source_dist))
            return source_arc_time_round, source_position_x, virtual_source_position_y
        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y


    def calculate_spiral(self, source_position_x, source_position_y, source_length, speed_of_sound, arc_angle, arc_scale_factor):
        if len(source_position_x) >= 2:
            num_points = 1000
            
            arc_angle = arc_angle/4
            # Konvertiere Öffnungswinkel in Radiant
            arc_angle_rad = np.deg2rad(arc_angle)
            
            # Erzeuge Theta-Werte für die linke und rechte Spirale
            theta_left = np.linspace(0, arc_angle_rad, num_points//2)
            theta_right = np.linspace(0, arc_angle_rad, num_points//2)
            
            # Spiralgleichung
            r_left = theta_left / arc_angle_rad  # Normalisiere r auf den Bereich [0, 1]
            r_right = theta_right / arc_angle_rad
            
            # Berechne x und y Koordinaten
            spiral_x_left = -r_left * np.cos(theta_left)
            spiral_y_left = -r_left * np.sin(theta_left) * arc_scale_factor
            
            spiral_x_right = r_right * np.cos(theta_right)
            spiral_y_right = -r_right * np.sin(theta_right) * arc_scale_factor
            
            # Kombiniere linke und rechte Spirale
            spiral_x = np.concatenate([spiral_x_left[::-1], spiral_x_right])
            spiral_y = np.concatenate([spiral_y_left[::-1], spiral_y_right])
            
            # Skaliere die Punkte
            scaled_x, scaled_y = self.scale_spiral_points(spiral_x, spiral_y, source_length)
            
            # Normalisiere die y-Werte, damit der höchste Wert 0 beträgt
            scaled_y -= np.max(scaled_y)
            
            # Entferne doppelte x-Werte und sortiere
            unique_indices = np.unique(scaled_x, return_index=True)[1]
            scaled_x_unique = scaled_x[unique_indices]
            scaled_y_unique = scaled_y[unique_indices]
            sort_indices = np.argsort(scaled_x_unique)
            scaled_x_sorted = scaled_x_unique[sort_indices]
            scaled_y_sorted = scaled_y_unique[sort_indices]
            
            # Interpolationsfunktion erstellen
            interp_func = interp1d(scaled_x_sorted, scaled_y_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            # Ermitteln der y-Positionen für die gegebenen x-Positionen
            virtual_source_position_y = interp_func(source_position_x)
            
            v_source_dist = np.abs(virtual_source_position_y - source_position_y)
            vp_source_del = v_source_dist / speed_of_sound
            source_arc_time = (vp_source_del - np.min(vp_source_del)) * 1000
            source_arc_time_round = [round(x, 2) for x in source_arc_time]
            
            return source_arc_time_round, source_position_x, virtual_source_position_y
        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y

    def scale_spiral_points(self, x, y, target_width):
        """
        Skaliert die x- und y-Punkte einer Spirale so, dass die x-Breite target_width beträgt.
        Die y-Achse wird linear zur x-Achse skaliert.
        """
        if len(x) == 0 or len(y) == 0:
            return x, y  # Rückgabe der ursprünglichen Arrays, wenn sie leer sind
        
        current_width = np.max(x) - np.min(x)
        if current_width == 0:
            return x, y  # Rückgabe der ursprünglichen Arrays, wenn die Breite 0 ist
        
        scale_factor = target_width / current_width
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        return x_scaled, y_scaled


    def calculate_elliptical(self, source_position_x, source_position_y, source_length, speed_of_sound, arc_angle, arc_scale_factor):
        if arc_angle < 0:
            arc_angle = abs(arc_angle)

        if len(source_position_x) >= 2:
            # Konvertierung des Winkels von Grad zu Radiant
            b = 1  # Halbe Länge der Nebenachse (jetzt in x-Richtung)
            
            # a = arc_angle * 0.005  # Halbe Länge der Hauptachse (jetzt in y-Richtung)
            a = arc_scale_factor

            elipse_size = np.deg2rad(arc_angle)

            # Berechnung der Ellipsenpunkte
            theta = np.linspace(0, 2 * np.pi, 10000)
            ellipse_x = b * np.cos(theta)
            ellipse_y = a * np.sin(theta)

            # Filterung der Punkte im rechten Teil der Ellipse innerhalb des Öffnungswinkels
            mask_upper = (ellipse_y > 0) & (np.abs(np.arctan2(ellipse_x, ellipse_y)) <= elipse_size / 2)
            filtered_x_upper = ellipse_x[mask_upper]
            filtered_y_upper = ellipse_y[mask_upper]
           
            # Skaliere die gefilterten Punkte
            filtered_x_upper_scaled, filtered_y_upper_scaled = self.scale_ellipse_points(filtered_x_upper, filtered_y_upper, source_length)

            # Normalisiere die y-Werte, damit der höchste Wert 0 beträgt
            max_y_value = np.max(filtered_y_upper_scaled)
            filtered_y_upper_scaled -= max_y_value

            # Interpolationsfunktion erstellen
            interp_func = interp1d(filtered_x_upper_scaled, filtered_y_upper_scaled, bounds_error=False, fill_value="extrapolate")

            # Ermitteln der x-Positionen für die gegebenen y-Positionen
            virtual_source_position_y = interp_func(source_position_x)
           
            v_source_dist = np.abs(virtual_source_position_y - source_position_y)
            vp_source_del = v_source_dist / speed_of_sound
            source_arc_time = (vp_source_del - np.min(vp_source_del)) * 1000
            source_arc_time_round = [round(x, 2) for x in source_arc_time]
            
            return source_arc_time_round, source_position_x, virtual_source_position_y

        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y


    def scale_ellipse_points(self, x, y, target_width):
        """
        Skaliert die x- und y-Punkte einer Ellipse so, dass die x-Breite target_width beträgt.
        Die y-Achse wird linear zur x-Achse skaliert.
        """
        current_width = np.max(x) - np.min(x)
        scale_factor = target_width / current_width
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        return x_scaled, y_scaled


    def calculate_linear(self, source_position_y, source_position_x, arc_angle, speed_of_sound):
        if len(source_position_y) >= 2:
            arc_angle = np.deg2rad(arc_angle)
            src_max = min(source_position_x) * -1
            source_position_xaxis = [y + src_max for y in source_position_x]
            source_position_xaxis = np.array(source_position_xaxis)
            delayed_distance = source_position_xaxis * np.tan(arc_angle)
            sound_speed = 344
            source_arc_time = (1 / sound_speed) * delayed_distance * 1000
            
            virtual_source_position_y = np.array(source_arc_time) / 2.9 * -1
            return source_arc_time, source_position_x, virtual_source_position_y
        else:
            return np.zeros(len(source_position_x)), source_position_x, source_position_y





















