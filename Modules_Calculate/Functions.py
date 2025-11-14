# Assi ethält alle benötigten Funktionen für die Berechnungen, sowie die Vervollständigung der Daten wenn keine Eingabe erfolgt

import numpy as np

class FunctionToolbox:
    settings = None
    data = None

    def __init__(self, settings):
        self.settings = settings

    "Die Vektor-Positionen werden durch Nullen ersetzt"
    def vector2zero(self, source, vector):
        return np.zeros(source)
        
    "Die Positionen werden linear vergeben"
    def linear_distribution(self, source_position_x, source_length, source_polar_pattern):     
        if len(source_polar_pattern) < 1:
            return 0
        else: 
            source_position_x = np.round(np.linspace(source_length/2*(-1), source_length/2, len(source_polar_pattern)),2)
            return source_position_x

    @staticmethod
    def surface_dimensions(surface_points):
        """
        Ermittelt Breite und Länge eines Surface anhand seiner Punkte.

        Args:
            surface_points (Sequence[Mapping[str, float]]): Iterable von Punktdefinitionen
                mit mindestens den Schlüsseln 'x' und 'y'.

        Returns:
            Tuple[float, float]: Breite (Ausdehnung entlang x) und Länge (Ausdehnung entlang y).
        """
        if not surface_points:
            return 0.0, 0.0

        xs = [float(point.get("x", 0.0)) for point in surface_points]
        ys = [float(point.get("y", 0.0)) for point in surface_points]

        width = max(xs) - min(xs)
        length = max(ys) - min(ys)

        return float(np.round(width, 2)), float(np.round(length, 2))
        
    
    "Die Differenz der ersten und letzten Position wird ermittelt"
    def source_length(self, source_position_y):
        if source_position_y is not None:
            return max(source_position_y) - min(source_position_y)
        else:
            return 0  

    "Umrechnung von Winkeln in Grad (°) nach rad"
    def deg2rad(self, x):
        if isinstance(x, list):  # Überprüfen, ob x eine Liste ist
            return [y * (np.pi) / 180 for y in x]  # Umrechnung für jede Zahl in der Liste
        else:
            return x * (np.pi) / 180  # Umrechnung für eine einzelne Zahl
    
    "Sucht die den höchsten wert im Vektor, setzt diesen auf 0 und gleicht alle anderen Werte an"
    def normalize2zero_lp(self, polar_data_ip, source_polar_pattern, freq_index, source_level):
        polar_data_ip = self.mag2db(polar_data_ip)
        source_lp_zero = polar_data_ip[source_polar_pattern,0,freq_index]    
        source_lp_zero_max = (max(source_lp_zero)*(-1) + source_lp_zero) + source_level # Finde den grössten Wert, setze auf 0dB und gleiche alle anderen Werte an
        return source_lp_zero_max

    "Umrechnen von Amplitude in Dezibel"
    def mag2db(self, x):
        x = np.where(x == 0, 1e-10, x)  # Ersetzen Sie 0 durch einen sehr kleinen Wert, z.B. 1e-10

        return 20*np.log10(x)
    
    # def mag2db(self, x):
    #     # Wenn x gleich 0 ist, gebe 0 zurück, sonst berechne den Dezibel-Wert
    #     return np.where(x == 0, 0, 20 * np.log10(x))
    
    "Umrechnen von Sekunden in Millisekunden"
    def s2ms(self, x):
        return x * 1000
    
    "Umrechnen von Dezibel in Amplitude"
    def db2mag(self, x):
        return 10**(x/20)
    
    "Umrechnung von db nach SPL"
    def db2spl(self, x):
        return x*0.00002
    
    "Berechnet die Schallgeschwindigkeit basierend auf Temperatur"
    def calculate_speed_of_sound(self, temperature):
        """
        Berechnet die Schallgeschwindigkeit in Abhängigkeit der Temperatur
        
        Args:
            temperature: Temperatur in °C
            
        Returns:
            Schallgeschwindigkeit in m/s
        """
        return 331.3 + 0.606 * temperature
    
    "Berechnen der Wellennummer (m)"
    def wavenumber(self, c,f):
        return 2 * np.pi / (c / f)
    
    "Setzt die Delayzeiten auf null wenn keinde Daten vorhanden sind (m)"
    def source_time(self, source_time, source_arc_time):
        if source_time == []: # Sind keine Zeit-Korrekturwerte vorhanden, werden die Arc-Berechnungen eingetragen
            return source_arc_time
        else:
            return source_time # Sind Zeit-Korrekturwerte vorhanden, werden die Arcberechnungen ueberschrieben

    "Pythagoras Hypothenuse"
    def pythagoras_hypothenuse(self, a,b):
        return np.sqrt(a**2 + b**2)

    "Umrechnung von Winkeln in rad nach Grad (°)"
    def rad2deg(self, x):
        return x*180/np.pi

    "Berechnen der Wellenlänge (m)"
    def wavelength(self, c,f):
        return c/f

    "Berechnen des Frauenhofer-Fresnelübergangs"
    def frauenhofer_fresnel_zone(self, freq, source_length):
        fresnel_freq = freq / 343  # Frequenz / Schallgeschwindigkeit
        # Term unter der Wurzel
        term = 1 - (1/(3 * fresnel_freq * source_length)**2)
        # Nur berechnen wenn Term positiv
        if term > 0:
            result = np.round(3/2 * fresnel_freq * (source_length**2) * np.sqrt(term), 1)
            return result
        else:
            print("Term unter Wurzel negativ - verwende Standardwert")
            return 50

    "Berechne die max. Distanz zwischen den Quellen --3dB addition Punkt--"
    def coherent_add_90deg(self, source_length, speed_of_sound, calculate_frequency):
        f_lamda = speed_of_sound/calculate_frequency
        return np.round(f_lamda / 2,1)

    "Berechne die max. Distanz zwischen den Quellen --0dB addition Punkt--"
    def coherent_add_120deg(self, source_length, speed_of_sound, calculate_frequency):
        f_lamda = speed_of_sound/calculate_frequency
        return np.round(f_lamda / 3 * 2,1)
    
    "Berechne die Aliasing Frequenz in Abhängigkeit vom Öffnungswinkel"
    def aliasing_frequency(self, speed_of_sound, source_length, source_polar_pattern, arc_angle):
        source_distance = source_length / (len(source_polar_pattern) -1)
        arc_angle = arc_angle * (np.pi)/180
        return np.round(speed_of_sound / (source_distance * (1 + np.cos(arc_angle))))


    "Berechne den Winkel im Schallfeld zur Quelle"
    def alpha2source(self, y_distance, x_distance, source_azimuth, isrc=None):
        """
        Berechnet den Winkel zur Quelle
        Args:
            y_distance: Y-Distanz zum Messpunkt
            x_distance: X-Distanz zum Messpunkt
            source_azimuth: Azimut der Quelle (Array oder einzelner Wert)
            isrc: Index für Array-Zugriff (optional)
        """
        alpha = np.arctan2(x_distance, y_distance)
        
        # Prüfe ob source_azimuth ein Array ist
        if isinstance(source_azimuth, (np.ndarray, list)):
            alpha = alpha - source_azimuth[isrc]
        else:
            alpha = alpha - source_azimuth
        
        return alpha
    
    
    "Passe die Daten für die Eingabe der Lautsprechersettings an"
    def adjust_source_lists(self):
        settings = self.settings
        num_sources = settings.number_of_sources


        if num_sources < len(settings.source_polar_pattern):
            while len(settings.source_polar_pattern) > num_sources:
                settings.source_polar_pattern.pop()

        if num_sources < len(settings.source_polar_pattern):
            del settings.source_polar_pattern[num_sources:]
        elif num_sources > len(settings.source_polar_pattern):
            last_index = settings.source_polar_pattern[-1] if settings.source_polar_pattern else 0
            settings.source_polar_pattern.extend([last_index] * (num_sources - len(settings.source_polar_pattern)))

        if num_sources < len(settings.source_position_x):
            settings.source_position_x = settings.source_position_x[:num_sources]
        elif num_sources > len(settings.source_position_x):
            settings.source_position_x = np.append(settings.source_position_x, np.zeros(num_sources - len(settings.source_position_x)))

        if num_sources < len(settings.source_position_y):
            settings.source_position_y = settings.source_position_y[:num_sources]
        elif num_sources > len(settings.source_position_y):
            settings.source_position_y = np.append(settings.source_position_y, np.zeros(num_sources - len(settings.source_position_y)))

        if num_sources < len(settings.source_azimuth):
            settings.source_azimuth = settings.source_azimuth[:num_sources]
        elif num_sources > len(settings.source_azimuth):
            settings.source_azimuth = np.append(settings.source_azimuth, np.zeros(num_sources - len(settings.source_azimuth)))

        if num_sources < len(settings.source_level):
            settings.source_level = settings.source_level[:num_sources]
        elif num_sources > len(settings.source_level):
            settings.source_level = np.append(settings.source_level, np.zeros(num_sources - len(settings.source_level)))

        if num_sources < len(settings.source_time):
            settings.source_time = settings.source_time[:num_sources]
        elif num_sources > len(settings.source_time):
            settings.source_time = np.append(settings.source_time, np.zeros(num_sources - len(settings.source_time)))


    def get_freq_index(self, frequency):
        freq_map = {
            25: 0,
            31.5: 1,
            40: 2,
            50: 3,
            63: 4,
            80: 5
        }
        return freq_map.get(frequency, 0)  # Default zu Index 0 wenn Frequenz nicht gefunden