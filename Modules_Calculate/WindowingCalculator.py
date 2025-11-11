import numpy as np
import scipy.signal.windows as wd
from Module_LFO.Modules_Init.ModuleBase import ModuleBase


class WindowingCalculator(ModuleBase):
    def __init__(self, settings, data, calculation_windowing, speaker_array_id):
        super().__init__(settings)
        self.settings = settings
        self.data = data
        self.calculation_windowing = calculation_windowing
        self.speaker_array_id = speaker_array_id
        self.speaker_array = self.settings.get_speaker_array(speaker_array_id)
        if self.speaker_array is None:
            raise ValueError(f"SpeakerArray with id {speaker_array_id} not found")
        self.calculation = {}

    def calculate_windowing(self):
        # Extrahiere Polar-Daten aus Balloon-Daten für jeden Lautsprecher
        polar_data = []
        for speaker_name in self.speaker_array.source_polar_pattern:
            speaker_polar_data = self.get_polar_data_from_balloon(
                speaker_name, 
                self.settings.calculate_frequency
            )
            if speaker_polar_data is not None:
                polar_data.append(speaker_polar_data)
        
        # Wenn Polar-Daten extrahiert wurden, verwende sie
        if polar_data:
            self.data_db = self.functions.mag2db(np.array(polar_data))
        # Fallback auf vorhandene Daten
        elif 'polar_mag' in self.data:
            self.data_db = self.functions.mag2db(self.data['polar_mag'])
        else:
            self.data_db = self.functions.mag2db(self.data['polar_data'])

        source_positions = getattr(
            self.speaker_array,
            'source_position_calc_x',
            self.speaker_array.source_position_x,
        )

        
        # Zuweisung der gewaehlten Fensterfunktion zur Variable
        self.calculation["window"] = self.load_window(self.speaker_array.window_function,
                                                      self.settings.window_resolution,
                                                      self.speaker_array.alpha,
                                                      self.settings.window_std)
        
        # Separieren der Werte oberhalb der Fensterbeschränkung
        self.calculation["window_restriction_db"] = self.restricted_window(self.calculation["window"], 
                                                                          self.speaker_array.window_restriction)

        # Punkte der Fensterfunktion in Bezug zur source_length         
        self.calculation["window_distance"] = self.wdw_stack_position(self.calculation["window_restriction_db"],
                                                                     self.speaker_array.source_length)

        # Ermittlung der Dämpfung einer begrenzten Fensterung auf Position der Quellen
        self.calculation["source_gain_wdw_mag"], self.calculation["source_gain_wdw_db"] =  self.source_gain_wdw(
            self.speaker_array.source_polar_pattern,
            source_positions,
            self.speaker_array.source_length,
            self.calculation["window_restriction_db"], 
            self.speaker_array.source_level
        )

    
        # self.data_db = self.functions.mag2db(self.data['polar_mag']) 
        
        # source_lp_zero wird in wdw draw verwendet
        source_lp_zero = self.normalize2zero_lp(
            self.data_db,
            self.speaker_array.source_polar_pattern
            )
        
        # Stelle sicher, dass source_lp_zero 1D ist
        source_lp_zero = np.asarray(source_lp_zero).reshape(-1)
        source_level = np.asarray(self.speaker_array.source_level).reshape(-1)
        
        # Stelle sicher, dass beide Arrays die gleiche Länge haben
        if len(source_level) != len(source_lp_zero):
            # Kürze source_lp_zero auf die Länge von source_level
            source_lp_zero = source_lp_zero[:len(source_level)]
        
        self.speaker_array.source_lp_zero = source_lp_zero + source_level

        return self.calculation
    
    

    def use_window_on_src(self):
        source_gain_wdw_db = self.calculation_windowing['source_gain_wdw_db']
        
        # Extrahiere Polar-Daten aus Balloon-Daten für jeden Lautsprecher
        polar_data = []
        for speaker_name in self.speaker_array.source_polar_pattern:
            speaker_polar_data = self.get_polar_data_from_balloon(
                speaker_name, 
                self.settings.calculate_frequency
            )
            if speaker_polar_data is not None:
                polar_data.append(speaker_polar_data)
        
        # Wenn Polar-Daten extrahiert wurden, verwende sie
        if polar_data:
            self.data_db = self.functions.mag2db(np.array(polar_data))
        # Fallback auf vorhandene Daten
        elif 'polar_mag' in self.data:
            self.data_db = self.functions.mag2db(self.data['polar_mag'])
        else:
            self.data_db = self.functions.mag2db(self.data['polar_data'])
        
        # Stelle sicher, dass die Arrays die richtige Form haben
        source_gain_wdw_db = np.asarray(source_gain_wdw_db).reshape(-1)  # Konvertiere zu 1D Array
        
        self.speaker_array.source_lp_zero = self.normalize2zero_lp(
            self.data_db,
            self.speaker_array.source_polar_pattern
        )
        
        # Stelle sicher, dass source_lp_zero ein 1D Array ist
        self.speaker_array.source_lp_zero = np.asarray(self.speaker_array.source_lp_zero).reshape(-1)
        
        # Konvertiere source_level zu einem numpy Array, falls es noch keins ist
        self.speaker_array.source_level = np.asarray(self.speaker_array.source_level).reshape(-1)
        
        # Stelle sicher, dass beide Arrays die gleiche Länge haben
        min_length = min(len(source_gain_wdw_db), len(self.speaker_array.source_lp_zero))
        source_gain_wdw_db = source_gain_wdw_db[:min_length]
        self.speaker_array.source_lp_zero = self.speaker_array.source_lp_zero[:min_length]
        self.speaker_array.source_level = self.speaker_array.source_level[:min_length]
        
        # Berechne die Differenzen elementweise
        for i in range(min_length):
            diff = float(source_gain_wdw_db[i] - self.speaker_array.source_lp_zero[i])
            self.speaker_array.source_level[i] = diff
        
        # Aktualisiere source_lp_zero
        self.speaker_array.source_lp_zero = self.speaker_array.source_lp_zero + self.speaker_array.source_level

    def set_sources_to_zero(self):
        # Setze Nullen in source_level ein. source_level ist die Levelanpassung der UI.
        for i in range(len(self.speaker_array.source_level)):
            self.speaker_array.source_level[i] = 0

        # Extrahiere Polar-Daten aus Balloon-Daten für jeden Lautsprecher
        polar_data = []
        for speaker_name in self.speaker_array.source_polar_pattern:
            speaker_polar_data = self.get_polar_data_from_balloon(
                speaker_name, 
                self.settings.calculate_frequency
            )
            if speaker_polar_data is not None:
                polar_data.append(speaker_polar_data)
        
        # Wenn Polar-Daten extrahiert wurden, verwende sie
        if polar_data:
            self.data_db = self.functions.mag2db(np.array(polar_data))
        # Fallback auf vorhandene Daten
        elif 'polar_mag' in self.data:
            self.data_db = self.functions.mag2db(self.data['polar_mag'])
        else:
            self.data_db = self.functions.mag2db(self.data['polar_data'])
        
        # source_lp_zero wird in wdw draw verwendet
        self.speaker_array.source_lp_zero = self.normalize2zero_lp(
            self.data_db,
            self.speaker_array.source_polar_pattern,
        )

    # "Laden der definierten Fensterfunktion in den Vektor"
    def load_window(self, window_function, wdw_resolution, alpha=None, wdw_std=None):
        # Laden der definierten Fensterfunktion in den Vektor
        if window_function == "tukey":
            # Verwende tukey Fensterfunktion
            window = wd.tukey(wdw_resolution, alpha)
        
        elif window_function == "gauss":         
            # Verwende gaussian Fensterfunktion
            window = wd.gaussian(wdw_resolution, wdw_std)
            
        elif window_function == "flattop":
            # Verwende flattop Fensterfunktion
            window = wd.flattop(wdw_resolution)
        
        elif window_function == "blackman":
            # Verwende blackman Fensterfunktion
            window = wd.blackman(wdw_resolution)
    
        else:
            raise ValueError("Unbekannte Fensterfunktion")
    
        # Sicherstellen, dass keine Nullwerte im Fenstervektor enthalten sind
        window = np.where(window == 0, 1e-10, window)
        return window
    
                 
    # w_hann = hann(windowLength);                      % Stopband -90.9dB
    # w_bohman = bohmanwin(windowLength);               % Stopband -46dB
    # w_blackHarris = blackmanharris(windowLength);     % Stopband -90.9dB
    # w_cheb = chebwin(windowLength);                   % Stopband -100dB

    "separieren der Werte oberhalb der Fensterbeschraenkung"
    def restricted_window(self, window, window_restriction):      
        return self.functions.mag2db(window)[self.functions.mag2db(window) > window_restriction]
   
    
    "Stackpositionen werden auf der Fensterfunktion definiert"  
    def wdw_stack_position(self, window_restriction_db, source_length):    
        window_restriction_db_len = len(window_restriction_db)                   # Anzahl der separierten Werte
        window_distance = np.linspace(-source_length/2, source_length/2, window_restriction_db_len) # Anzahl Datenpunkte über source length für Fensterfunktion berechnen
        return window_distance    # Fenster auf Arraylaenge bezogen


    "Position der Stacks auf Y-Achse mit den definierten Source Gains auf der gewählten Fensterfunktion"
    def position_source_gains(self, defined_source_gains, window_restriction_db, position_source_gains):
        for i in np.linspace(0,len(defined_source_gains)-1,len(defined_source_gains)):      # definiere Index pro Stack der Arrayhaelfte
            source_gain = np.array(defined_source_gains[i.astype(int)])                     # niedrigster Referenz Pegel der verwendeten Stacks
            window_index = np.argmin(abs(window_restriction_db[:-(len(np.linspace(1,int(np.size(window_restriction_db)/2),int(np.size(window_restriction_db)/2))))]-source_gain))
            position_source_gains[i.astype(int)] = window_index                             # Position auf der Fensterbreite der einzelnen Stacks bezogen auf deren SPL
            return position_source_gains        


    "Ermittlung der Dämpfung einer begrenzten Fensterung auf Position der Quellen"
    "Fensterung mit weniger als 3 Quellen ist zwecklos, somit wird dann die Funktion deaktiviert."
    
    def source_gain_wdw(self, source_polar_pattern, source_positions, source_length, window_restriction_db, source_level):
        """
        Ermittlung der Dämpfung einer begrenzten Fensterung auf Position der Quellen.
        Fensterung mit weniger als 3 Quellen ist zwecklos, somit wird dann die Funktion deaktiviert.
        """
        # Prüfe, ob genügend Quellen vorhanden sind und source_length > 0
        if len(source_polar_pattern) > 2 and source_length > 0:
            # Prüfe, ob window_restriction_db leer ist
            if len(window_restriction_db) == 0:
                print("WARNUNG: window_restriction_db ist leer! Berechnung wird abgebrochen.")
                # Rückgabe von Standardwerten ohne Fensterung
                source_gain_wdw_db = [0 for _ in source_polar_pattern]
                source_gain_wdw_mag = [1 for _ in source_polar_pattern]
                return source_gain_wdw_mag, source_gain_wdw_db
            
            # Berechne die Auflösung der Fensterung
            resolution = source_length / len(window_restriction_db)
            pos_y_0 = abs(source_positions[0])
            
            # Berechne die Indizes für die Fensterung
            try:
                n_window_index = np.append(0, np.array(source_positions[1:]) / resolution + pos_y_0 / resolution)
                n_window_index = np.clip(n_window_index, 1, len(window_restriction_db))  # Begrenze den Index
                window_resample = window_restriction_db[n_window_index.astype(int)-1]
            except (IndexError, ZeroDivisionError) as e:
                print(f"Fehler bei der Berechnung der Fensterung: {e}")
                print("Berechnung wird abgebrochen.")
                # Rückgabe von Standardwerten ohne Fensterung
                source_gain_wdw_db = [0 for _ in source_polar_pattern]
                source_gain_wdw_mag = [1 for _ in source_polar_pattern]
                return source_gain_wdw_mag, source_gain_wdw_db
            
            source_gain_wdw_db = window_resample
            source_gain_wdw_mag = self.functions.db2mag(source_gain_wdw_db)
            return source_gain_wdw_mag, source_gain_wdw_db
        
        else:
            # Standardwerte ohne Fensterung
            source_gain_wdw_db = [0 for _ in source_polar_pattern]
            source_gain_wdw_mag = [1 for _ in source_polar_pattern]
            return source_gain_wdw_mag, source_gain_wdw_db


     
    "Normalisieren der Stack Gains auf den höchsten Wert der Stacks"
    "Wird benötigt für Window Darstellung und Pegel der einzelnen Sources."
    def normalize2zero_lp(self, polar_data, source_polar_pattern):
        """
        Normalisiert die Stack Gains auf den höchsten Wert der Stacks.
        Wird benötigt für Window Darstellung und Pegel der einzelnen Sources.
        """
        # Prüfe, ob Balloon-Daten für die Lautsprecher vorhanden sind
        source_lp_zero = []
        
        for i, speaker_name in enumerate(source_polar_pattern):
            # Berücksichtige den Azimuth-Winkel des Lautsprechers
            # Wenn der Lautsprecher um X Grad gedreht ist, nehme den Wert bei -X Grad
            # (da wir den Wert in Richtung der Hauptachse des Arrays benötigen)
            azimuth = -self.speaker_array.source_azimuth[i]  # Negativ, da wir den gegenüberliegenden Winkel brauchen
            
            # Hole den Wert für diesen Winkel
            value = self.get_polar_data_at_angle(
                speaker_name, 
                azimuth, 
                self.settings.calculate_frequency
            )
            
            if value is not None:
                source_lp_zero.append(value)
            else:
                # Fallback: Verwende den Wert aus polar_data, wenn vorhanden
                if isinstance(polar_data, np.ndarray) and i < len(polar_data):
                    # Finde den nächsten Winkel in den Polar-Daten
                    azimuth_idx = int(round(azimuth)) % 360
                    if azimuth_idx < len(polar_data[i]):
                        source_lp_zero.append(polar_data[i][azimuth_idx])
                    else:
                        source_lp_zero.append(-np.inf)  # Fallback-Wert
                else:
                    source_lp_zero.append(-np.inf)  # Fallback-Wert
        
        source_lp_zero = np.array(source_lp_zero)
        
        if np.isneginf(source_lp_zero).all():
            return np.array([-np.inf])
        else:
            max_value = np.max(source_lp_zero)
            source_lp_zero_max = (-max_value + source_lp_zero)  # Normalisiere auf den höchsten Wert
            return source_lp_zero_max

    def get_polar_data_from_balloon(self, speaker_name, frequency):
        """
        Extrahiert Polar-Daten aus den Balloon-Daten für eine bestimmte Frequenz
        
        Args:
            speaker_name (str): Name des Lautsprechers
            frequency (float): Frequenz in Hz
            
        Returns:
            numpy.ndarray: Polar-Daten (Magnitude) oder None wenn keine Daten gefunden wurden
        """
        # Prüfe, ob gemittelte Balloon-Daten vorhanden sind
        mag_key = f'balloon_avg_mag_{speaker_name}'
        
        if mag_key not in self.data:
            return None
        
        mag_data = self.data[mag_key]
        
        if not isinstance(mag_data, dict):
            return None
        
        # Extrahiere Magnitude Array
        magnitude = mag_data.get('magnitude')
        vertical_angles = mag_data.get('vertical_angles')
        horizontal_angles = mag_data.get('horizontal_angles')
        freqs = mag_data.get('freqs')
        
        # Debug-Ausgabe        
        if magnitude is None or vertical_angles is None:
            return None
        
        # Wenn freqs fehlt, verwenden wir die erste Dimension (falls vorhanden)
        if freqs is None:
            freq_idx = 0
        else:
            # Finde nächste Frequenz
            freq_idx = np.abs(freqs - frequency).argmin()
        
        # Verwende 0° vertikalen Winkel (horizontale Ebene)
        v_idx = np.abs(vertical_angles).argmin()
        
        # Hole Magnitude für alle horizontalen Winkel
        try:            
            if len(magnitude.shape) == 3:  # [vertical, horizontal, frequency]
                polar_data = magnitude[v_idx, :, freq_idx]
            elif len(magnitude.shape) == 2:  # [vertical, horizontal]
                polar_data = magnitude[v_idx, :]
            else:
                return None
            
            return polar_data
        except IndexError as e:
            
            # Fallback: Versuche, die erste Dimension zu verwenden
            try:
                if len(magnitude.shape) == 3:
                    return magnitude[0, :, 0]  # Erste vertikale, alle horizontalen, erste Frequenz
                elif len(magnitude.shape) == 2:
                    return magnitude[0, :]  # Erste vertikale, alle horizontalen
            except:
                pass
            
            return None

    def get_polar_data_at_angle(self, speaker_name, azimuth, frequency):
        """
        Holt die Polar-Daten für einen bestimmten Winkel
        
        Args:
            speaker_name (str): Name des Lautsprechers
            azimuth (float): Azimut-Winkel in Grad (0-360)
            frequency (float): Frequenz in Hz
            
        Returns:
            float: Magnitude-Wert oder None wenn keine Daten gefunden wurden
        """
        # Prüfe, ob gemittelte Balloon-Daten vorhanden sind
        mag_key = f'balloon_avg_mag_{speaker_name}'
        
        if mag_key not in self.data:
            print(f"Keine gemittelten Balloon-Daten für {speaker_name}")
            return None
        
        mag_data = self.data[mag_key]
        
        if not isinstance(mag_data, dict):
            print(f"Ungültiges Format für Balloon-Daten von {speaker_name}")
            return None
        
        # Extrahiere Magnitude Array
        magnitude = mag_data.get('magnitude')
        vertical_angles = mag_data.get('vertical_angles')
        horizontal_angles = mag_data.get('horizontal_angles')
        freqs = mag_data.get('freqs')
        
        if magnitude is None or vertical_angles is None:
            print(f"Fehlende Grunddaten in Balloon-Daten für {speaker_name}")
            return None
        
        # Wenn freqs fehlt, verwenden wir die erste Dimension
        if freqs is None:
            freq_idx = 0
        else:
            # Finde nächste Frequenz
            freq_idx = np.abs(freqs - frequency).argmin()
        
        # Verwende 0° vertikalen Winkel (horizontale Ebene)
        v_idx = np.abs(vertical_angles).argmin()
        
        # Normalisiere Azimut auf 0-360 Grad
        azimuth = azimuth % 360
        
        # Wenn horizontal_angles vorhanden ist, finde den nächsten Winkel
        if horizontal_angles is not None:
            h_idx = np.abs(horizontal_angles - azimuth).argmin()
        else:
            # Sonst nehme an, dass die horizontalen Winkel von 0-359 gehen
            h_idx = int(round(azimuth)) % 360
        
        # Hole Magnitude für diesen Winkel
        try:
            if len(magnitude.shape) == 3:  # [vertical, horizontal, frequency]
                mag_value = magnitude[v_idx, h_idx, freq_idx]
            elif len(magnitude.shape) == 2:  # [vertical, horizontal]
                mag_value = magnitude[v_idx, h_idx]
            else:
                print(f"Unerwartete Form der Magnitude-Daten für {speaker_name}: {magnitude.shape}")
                return None
            
            return mag_value
        except IndexError as e:
            print(f"Indexfehler beim Zugriff auf Balloon-Daten für {speaker_name} bei Azimut {azimuth}°: {e}")
            return None










