"""
Polar Data Interpolate Calculator - Interpolation & Kombination
================================================================

Speicherort: LFO/Module_LFO/Modules_Calculate/PolarDataInterpolateCalculator.py

Zweck:
------
Spezialisiertes Modul für die Interpolation und Kombination von Polardaten:
1. Erstellt balloon_data aus calculated_data (basierend auf metadata['folder_data'])
2. Konvertiert Dict → NumPy-Arrays für effiziente Verarbeitung
3. Interpoliert horizontale Daten auf 1° Schritte (0°-180°)
4. (Später) Sphärische Interpolation um Y-Achse → 3D-Kugel

Funktionen:
-----------
- create_interpolated_balloon_data(): Erstellt balloon_data + führt komplette Interpolation durch
- _convert_dict_to_numpy_structure(): Konvertiert Dict-basierte Daten zu NumPy
- _interpolate_all_meridians_horizontal_vectorized(): Interpoliert alle Meridiane auf 1° Schritte
- interpolate_meridians_spherical(): Erstellt 3D-Kugel durch sphärische Rotation (später)

NEUE DATENSTRUKTUR OUTPUT (NUMPY):
----------------------------------
balloon_data = {
    'meridians': np.array([0, 180]),              # [N_meridians] int
    'horizontal_angles': np.array([0,1,...,180]), # [N_horizontal] int (0°-180°, 1° Schritte)
    'frequencies': np.array([...]),               # [N_freq] float
    'magnitude': np.array([...]),                 # [N_mer, N_horz, N_freq] float
    'phase': np.array([...])                      # [N_mer, N_horz, N_freq] float
}

Vorteile:
---------
- Vektorisiert für NumPy-Performance
- Direkter Array-Zugriff ohne String-Konvertierung
- Effiziente Memory-Nutzung
- Vollständige Vektorisierung der Interpolation

Abhängigkeiten:
---------------
- numpy: Array-Operationen und Interpolation
- Module_LFO.Modules_Init.ModuleBase: Basis-Klasse

Autor: MGraf
Datum: 2025-10-30
"""

import numpy as np
from Module_LFO.Modules_Init.ModuleBase import ModuleBase


class Polar_Data_Interpolate_Calculator(ModuleBase):
    """
    Spezialisierte Klasse für Interpolation und Kombination von Polardaten.
    
    Verantwortlichkeiten:
    - Erstellt balloon_data aus calculated_data
    - Kombiniert Meridian-Paare zu 360° Polardaten
    - Interpoliert auf 1° Schritte
    """
    
    def __init__(self, settings, data):
        """
        Initialisiert den Interpolate Calculator.
        
        Args:
            settings: Settings-Objekt mit Konfigurationen
            data: Data-Dictionary mit balloon_data
        """
        super().__init__(settings)
        self.data = data
        self.settings = settings
    


    def create_interpolated_balloon_data(self):
        """
        Haupt-Pipeline für Balloon-Daten-Erstellung mit NEUER NumPy-Struktur.
        
        NEUE DATENSTRUKTUR:
        -------------------
        balloon_data = {
            'meridians': np.array([0, 180]),              # [N_meridians] int
            'horizontal_angles': np.array([0,1,...,180]), # [N_horizontal] int  
            'frequencies': np.array([...]),               # [N_freq] float
            'magnitude': np.array([...]),                 # [N_mer, N_horz, N_freq] float
            'phase': np.array([...])                      # [N_mer, N_horz, N_freq] float
        }
        
        PIPELINE:
        ---------
        1. Sammelt Daten aus calculated_data → Dict (Rohdaten)
        2. Konvertiert Dict → NumPy Struktur
        3. Interpoliert jeden Meridian horizontal auf 1° Schritte (0°-180°)
        4. (Später) Sphärische Interpolation → 3D-Kugel
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            # Prüfe, ob Metadaten vorhanden sind
            if 'metadata' not in self.data or not self.data['metadata'] or 'folder_data' not in self.data['metadata']:
                print("⚠ Keine Metadaten vorhanden!")
                return False
            
            # Prüfe, ob Ordner vorhanden sind
            if not self.data['metadata']['folder_data']:
                print("⚠ Keine Ordner vorhanden!")
                return False
            
            # ================================================================
            # SCHRITT 1: SAMMELE ROHDATEN IN DICT (temporär)
            # ================================================================
            raw_balloon_dict = {}
            
            for folder in self.data['metadata']['folder_data']:
                folder_name = folder['name']
                meridian_angle = folder['meridian_angle']
                measurements = folder['measurements']
                            
                # Erstelle Eintrag für diesen Meridian
                meridian_str = str(int(meridian_angle))
                raw_balloon_dict[meridian_str] = {}
                
                # Füge für jede Messung im Ordner einen Eintrag hinzu
                for measurement in measurements:
                    filename = measurement['filename']
                    polar_angle = measurement['horizontal_angle']
                    polar_str = str(int(polar_angle))
                    
                    if filename in self.data['calculated_data']:
                        raw_balloon_dict[meridian_str][polar_str] = {
                            'freq': self.data['calculated_data'][filename]['freq'].copy(),
                            'magnitude': self.data['calculated_data'][filename]['magnitude'].copy(),
                            'phase': self.data['calculated_data'][filename]['phase'].copy()
                        }
            
            # ================================================================
            # SCHRITT 2: KONVERTIERE DICT → NUMPY STRUKTUR
            # ================================================================
            self.data['balloon_data'] = self._convert_dict_to_numpy_structure(raw_balloon_dict)
            
            # ================================================================
            # SCHRITT 3: INTERPOLIERE JEDEN MERIDIAN AUF 1° SCHRITTE (0°-180°)
            # ================================================================
            num_angles_before = len(self.data['balloon_data']['horizontal_angles'])
            
            # Interpoliere ALLE Meridiane auf einmal (effizienter)
            success = self._interpolate_all_meridians_horizontal_vectorized()
            if not success:
                print(f"⚠ Interpolation fehlgeschlagen!")
                return False
            
            # ================================================================
            # SCHRITT 4: SPHÄRISCHE INTERPOLATION → 3D-KUGEL
            # ================================================================
            success = self._interpolate_to_full_sphere()
            if not success:
                print("⚠ Sphärische Interpolation fehlgeschlagen!")
                return False
            
            return True
            
        except Exception as e:
            print(f"FEHLER beim Erstellen der Balloon-Daten: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def _convert_dict_to_numpy_structure(self, raw_balloon_dict):
        """
        Konvertiert die Dict-basierte balloon_data in NumPy-Arrays.
        
        Args:
            raw_balloon_dict: Dict-Struktur {meridian: {horizontal_angle: {freq, mag, phase}}}
            
        Returns:
            dict: NumPy-basierte Struktur mit 'meridians', 'horizontal_angles', 'frequencies', 'magnitude', 'phase'
        """
        # Sammle alle Meridiane und horizontalen Winkel
        meridians = sorted([int(float(k)) for k in raw_balloon_dict.keys()])
        
        # Sammle alle horizontalen Winkel (aus allen Meridianen)
        all_horizontal_angles = set()
        for meridian_dict in raw_balloon_dict.values():
            for angle_str in meridian_dict.keys():
                all_horizontal_angles.add(int(float(angle_str)))
        horizontal_angles_raw = sorted(all_horizontal_angles)
        
        # Hole Frequenzen aus erstem Eintrag
        first_meridian = str(meridians[0])
        first_horizontal = str(horizontal_angles_raw[0])
        frequencies = raw_balloon_dict[first_meridian][first_horizontal]['freq'].copy()
        
        # Erstelle NumPy-Arrays für Magnitude und Phase
        # Shape: [N_meridians, N_horizontal_angles, N_frequencies]
        N_mer = len(meridians)
        N_horz = len(horizontal_angles_raw)
        N_freq = len(frequencies)
        
        magnitude = np.zeros((N_mer, N_horz, N_freq))
        phase = np.zeros((N_mer, N_horz, N_freq))
        
        # Fülle Arrays mit Daten
        for mer_idx, mer_angle in enumerate(meridians):
            mer_str = str(mer_angle)
            
            for horz_idx, horz_angle in enumerate(horizontal_angles_raw):
                horz_str = str(horz_angle)
                
                if horz_str in raw_balloon_dict[mer_str]:
                    magnitude[mer_idx, horz_idx, :] = raw_balloon_dict[mer_str][horz_str]['magnitude']
                    phase[mer_idx, horz_idx, :] = raw_balloon_dict[mer_str][horz_str]['phase']
                else:
                    # Fehlende Daten mit NaN füllen
                    magnitude[mer_idx, horz_idx, :] = np.nan
                    phase[mer_idx, horz_idx, :] = np.nan
        
        return {
            'meridians': np.array(meridians, dtype=int),
            'horizontal_angles': np.array(horizontal_angles_raw, dtype=int),
            'frequencies': frequencies,
            'magnitude': magnitude,
            'phase': phase
        }
    
    
    def _interpolate_all_meridians_horizontal_vectorized(self):
        """
        Interpoliert ALLE Meridiane horizontal auf 1° Schritte (0°-180°) mit NumPy.
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            balloon = self.data['balloon_data']
            
            # Zielwinkel: 0° bis 180° in 1° Schritten
            target_angles = np.arange(0, 181, dtype=int)
            
            # Prüfe, ob bereits interpoliert
            if len(balloon['horizontal_angles']) >= 181:
                # Prüfe ob genau 0°-180° vorhanden
                if balloon['horizontal_angles'][0] == 0 and balloon['horizontal_angles'][-1] == 180:
                    return True
            
            # Source-Angles aus aktueller Struktur
            source_angles = balloon['horizontal_angles'].copy()
            
            # Shape-Informationen
            N_mer = len(balloon['meridians'])
            N_freq = len(balloon['frequencies'])
            
            # Neue Magnitude/Phase Arrays mit korrekten Dimensionen
            # Shape: [N_mer, N_horz_new, N_freq]
            new_magnitude = np.zeros((N_mer, len(target_angles), N_freq))
            new_phase = np.zeros((N_mer, len(target_angles), N_freq))
            
            # 🚀 VEKTORISIERT: Interpoliere alle Meridiane
            for mer_idx in range(N_mer):
                # Hole Magnitude und Phase für diesen Meridian
                mag_meridian = balloon['magnitude'][mer_idx, :, :]  # [N_horz, N_freq]
                phase_meridian = balloon['phase'][mer_idx, :, :]    # [N_horz, N_freq]
                
                # Interpoliere über ALLE Frequenzen
                for freq_idx in range(N_freq):
                    # Finde gültige (nicht-NaN) Werte
                    valid_mask = ~np.isnan(mag_meridian[:, freq_idx])
                    
                    if np.sum(valid_mask) == 0:
                        # Keine Datenpunkte vorhanden
                        new_magnitude[mer_idx, :, freq_idx] = np.nan
                        new_phase[mer_idx, :, freq_idx] = np.nan
                        continue
                    elif np.sum(valid_mask) == 1:
                        # Nur ein Datenpunkt: Verwende diesen für alle Winkel
                        new_magnitude[mer_idx, :, freq_idx] = mag_meridian[valid_mask, freq_idx][0]
                        new_phase[mer_idx, :, freq_idx] = phase_meridian[valid_mask, freq_idx][0]
                        continue
                    
                    # Nur gültige Punkte für Interpolation verwenden
                    valid_angles = source_angles[valid_mask]
                    valid_mag = mag_meridian[valid_mask, freq_idx]
                    valid_phase = phase_meridian[valid_mask, freq_idx]
                    
                    # Magnitude: Lineare Interpolation nur mit gültigen Werten
                    new_magnitude[mer_idx, :, freq_idx] = np.interp(target_angles, valid_angles, valid_mag)
                    
                    # Phase: Unwrap → Interpolation → Wrap
                    phase_rad = np.radians(valid_phase)
                    phase_unwrapped = np.unwrap(phase_rad)
                    phase_interp_rad = np.interp(target_angles, valid_angles, phase_unwrapped)
                    phase_interp_deg = np.degrees(phase_interp_rad)
                    new_phase[mer_idx, :, freq_idx] = ((phase_interp_deg + 180) % 360) - 180
            
            # Aktualisiere Struktur
            self.data['balloon_data']['horizontal_angles'] = target_angles
            self.data['balloon_data']['magnitude'] = new_magnitude
            self.data['balloon_data']['phase'] = new_phase
            
            return True
            
        except Exception as e:
            print(f"Fehler bei horizontaler Interpolation: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _interpolate_to_full_sphere(self, meridian_step=5, max_interpolation_gap=100):
        """
        Erstellt vollständige 3D-Kugel durch Rotation und Interpolation der Meridiane.
        
        GEOMETRIE:
        ----------
        - Meridian = Halbkreis (0°-180°) entlang Y-Achse
        - Rotation = 0°-360° um Y-Achse für vollständige Kugel
        
        EINHEITLICHE STRATEGIE:
        -----------------------
        - 1 Meridian: Kopiere auf alle 72 Rotationswinkel (0°-360° in 5° Schritten)
        - 2 Meridiane (0° + 180°): Zirkuläre Interpolation → 72 Meridiane
        - N Meridiane: Zirkuläre Interpolation zwischen nächsten Nachbarn
        
        MAXIMALE INTERPOLATIONS-DISTANZ:
        --------------------------------
        - Wenn der Abstand zwischen zwei Nachbarn > max_interpolation_gap:
          → KEINE Interpolation, Ziel-Meridian wird auf NaN gesetzt
        - Verhindert falsche Interpolation über große Lücken (z.B. fehlende Meridiane)
        - Standard: 100° (bei 0°, 90°, 180° → keine Interpolation über 180°-0° Lücke)
        - AUSNAHME: Bei nur 1-2 gültigen Meridianen wird IMMER interpoliert (volle 360° Kugel)
        
        Args:
            meridian_step: Schrittweite für Rotation in Grad (Standard: 5° → 72 Meridiane)
            max_interpolation_gap: Maximaler Abstand für Interpolation in Grad (Standard: 100°)
        
        Returns:
            bool: True bei Erfolg, False bei Fehler
        """
        try:
            balloon = self.data['balloon_data']
            
            # Vorhandene Meridiane (MUSS sortiert sein für zirkuläre Interpolation!)
            source_meridians = balloon['meridians'].copy()  # z.B. [0, 180]
            
            # WICHTIG: Sortiere Meridiane, falls sie nicht sortiert sind
            # Sonst funktioniert die zirkuläre Interpolation nicht korrekt!
            if not np.all(source_meridians[:-1] <= source_meridians[1:]):
                print("⚠️ WARNUNG: Source-Meridiane nicht sortiert! Sortiere jetzt...")
                sort_indices = np.argsort(source_meridians)
                source_meridians = source_meridians[sort_indices]
                # Sortiere auch die Daten-Arrays entsprechend
                balloon['magnitude'] = balloon['magnitude'][sort_indices, :, :]
                balloon['phase'] = balloon['phase'][sort_indices, :, :]
                balloon['meridians'] = source_meridians
            
            # ============================================================================
            # 🔍 KRITISCH: Filtere LEERE Meridiane aus (nur NaN-Werte)
            # ============================================================================
            valid_meridian_mask = []
            valid_meridian_indices = []
            
            for mer_idx, mer_angle in enumerate(source_meridians):
                mag_data = balloon['magnitude'][mer_idx, :, :]
                nan_count = np.sum(np.isnan(mag_data))
                total_count = mag_data.size
                nan_percentage = nan_count / total_count
                
                is_valid = nan_percentage < 0.9  # <90% NaN = gültig
                
                if is_valid:
                    valid_meridian_mask.append(True)
                    valid_meridian_indices.append(mer_idx)
                else:
                    valid_meridian_mask.append(False)
            
            valid_meridian_mask = np.array(valid_meridian_mask)
            valid_meridian_indices = np.array(valid_meridian_indices)
            valid_source_meridians = source_meridians[valid_meridian_mask]
            
            if len(valid_source_meridians) == 0:
                print("❌ FEHLER: Keine gültigen Meridiane vorhanden!")
                return False
            
            # ============================================================================
            # DYNAMISCHER max_interpolation_gap basierend auf Anzahl gültiger Meridiane
            # ============================================================================
            # Bei wenigen Meridianen (3-4) sind größere Abstände normal und OK
            # Bei vielen Meridianen (>4) sollten Lücken vermieden werden
            if len(valid_source_meridians) == 3:
                # 3 Meridiane: Max. Abstand = 360°/3 = 120° + Toleranz
                adjusted_gap = 130
                print(f"\nℹ️  Angepasster max_interpolation_gap für 3 Meridiane: {adjusted_gap}° (statt {max_interpolation_gap}°)")
            elif len(valid_source_meridians) == 4:
                # 4 Meridiane: Max. Abstand = 360°/4 = 90° + Toleranz
                adjusted_gap = max_interpolation_gap  # 100° ist OK
            else:
                # ≥5 Meridiane: Verwende ursprünglichen Wert
                adjusted_gap = max_interpolation_gap
            
            max_interpolation_gap = adjusted_gap
            
            N_source = len(source_meridians)
            
            # Ziel-Meridiane: 0° bis 360° (exklusiv) in meridian_step Schritten
            # Vollständige Rotation um Y-Achse für 3D-Kugel
            # 360° nicht eingeschlossen, da identisch mit 0°
            target_meridians = np.arange(0, 360, meridian_step, dtype=int)
            N_target = len(target_meridians)
            
            # Wenn bereits genug Meridiane vorhanden, überspringen
            if N_source >= N_target:
                print(f"✅ Bereits {N_source} Meridiane vorhanden, keine Interpolation nötig")
                return True
            
            # Shape-Informationen
            N_horz = len(balloon['horizontal_angles'])
            N_freq = len(balloon['frequencies'])
            
            # Neue Arrays für interpolierte Kugel (mit NaN initialisiert)
            new_magnitude = np.full((N_target, N_horz, N_freq), np.nan)
            new_phase = np.full((N_target, N_horz, N_freq), np.nan)
            
            # Statistik für Debug-Ausgabe
            interpolated_count = 0
            skipped_count = 0
            copied_count = 0
            
            # Debug: Speichere Interpolations-Details für ausgewählte Meridiane
            debug_targets = [5, 45, 95, 135, 185, 225, 275, 315]  # Beispiel-Meridiane zum Debuggen
            debug_info = {}
            
            # 🚀 VEKTORISIERT: Interpoliere für jeden Ziel-Meridian
            for target_idx, target_meridian in enumerate(target_meridians):
                
                # Fall 1: Ziel-Meridian ist vorhanden → direkt kopieren
                if target_meridian in source_meridians:
                    source_idx = np.where(source_meridians == target_meridian)[0][0]
                    new_magnitude[target_idx, :, :] = balloon['magnitude'][source_idx, :, :]
                    new_phase[target_idx, :, :] = balloon['phase'][source_idx, :, :]
                    copied_count += 1
                    
                    # 🐛 DEBUG: Speichere Details für ausgewählte Meridiane
                    if target_meridian in debug_targets:
                        mag_val = balloon['magnitude'][source_idx, 90, 0]
                        debug_info[target_meridian] = {
                            'before': target_meridian,
                            'after': target_meridian,
                            'dist_before': 0,
                            'dist_after': 0,
                            'weight_before': 1.0,
                            'weight_after': 0.0,
                            'mag_before': mag_val,
                            'mag_after': mag_val,
                            'mag_result': mag_val
                        }
                
                # Fall 2: Zirkuläre Interpolation zwischen GÜLTIGEN Meridianen
                else:
                    # ================================================================
                    # ⚠️ KRITISCH: Suche NUR unter GÜLTIGEN Meridianen!
                    # ================================================================
                    # Berechne Abstände VON target ZU jedem GÜLTIGEN Meridian
                    distances_cw = np.array([(s - target_meridian) % 360 for s in valid_source_meridians])
                    
                    # Konvertiere zu [-180, +180] Bereich (kürzester Weg)
                    distances_cw[distances_cw > 180] -= 360
                    
                    # Finde Nachbar VOR target (größter negativer Wert = am nächsten gegen UZS)
                    # VOR = gegen Uhrzeigersinn von target
                    distances_before = distances_cw.copy()
                    distances_before[distances_before >= 0] = -361  # Positive ausschließen
                    valid_idx_before = np.argmax(distances_before)  # Index in valid_source_meridians
                    meridian_before = valid_source_meridians[valid_idx_before]
                    dist_from_before = abs(distances_before[valid_idx_before])
                    
                    # Finde Nachbar NACH target (kleinster positiver Wert = am nächsten im UZS)
                    # NACH = im Uhrzeigersinn von target
                    distances_after = distances_cw.copy()
                    distances_after[distances_after <= 0] = 361  # Negative ausschließen
                    valid_idx_after = np.argmin(distances_after)  # Index in valid_source_meridians
                    meridian_after = valid_source_meridians[valid_idx_after]
                    dist_to_after = abs(distances_after[valid_idx_after])
                    
                    # Konvertiere zu den tatsächlichen Array-Indizes
                    idx_before = valid_meridian_indices[valid_idx_before]
                    idx_after = valid_meridian_indices[valid_idx_after]
                    
                    # Wenn nur ein gültiger Meridian vorhanden
                    if len(valid_source_meridians) == 1 or meridian_before == meridian_after:
                        # Kopiere vom einzigen/nächsten Meridian
                        new_magnitude[target_idx, :, :] = balloon['magnitude'][idx_before, :, :]
                        new_phase[target_idx, :, :] = balloon['phase'][idx_before, :, :]
                    else:
                        # Prüfe Gesamtabstand zwischen den beiden Nachbarn
                        total_dist = dist_from_before + dist_to_after
                        
                        # ⚠️ DISTANZ-CHECK: Zu große Lücke? → Keine Interpolation!
                        # ABER: Bei nur 1-2 gültigen Meridianen → IMMER interpolieren!
                        if len(valid_source_meridians) >= 3 and total_dist > max_interpolation_gap:
                            # Lücke zu groß → Setze NaN (keine Daten)
                            new_magnitude[target_idx, :, :] = np.nan
                            new_phase[target_idx, :, :] = np.nan
                            skipped_count += 1
                            
                            # Debug-Ausgabe nur für jeden 10. übersprungenen Meridian
                            if skipped_count == 1 or skipped_count % 10 == 0:
                                print(f"   ⚠️ Meridian {target_meridian}°: Lücke zu groß ({total_dist:.0f}° zwischen {meridian_before}° und {meridian_after}°) → übersprungen")
                            
                            continue  # Nächster Ziel-Meridian
                        
                        # Interpoliere zwischen den beiden Nachbarn (ZIRKULÄR)
                        # Gewicht basierend auf zirkulärem Abstand
                        weight_before = dist_to_after / total_dist
                        weight_after = dist_from_before / total_dist
                        interpolated_count += 1
                        
                        # 🐛 DEBUG: Speichere Details für ausgewählte Meridiane
                        if target_meridian in debug_targets:
                            mag_before_val = balloon['magnitude'][idx_before, 90, 0]  # horizontal_angle=90°, freq=0
                            mag_after_val = balloon['magnitude'][idx_after, 90, 0]
                            mag_result_val = weight_before * mag_before_val + weight_after * mag_after_val
                            
                            debug_info[target_meridian] = {
                                'before': meridian_before,
                                'after': meridian_after,
                                'dist_before': dist_from_before,
                                'dist_after': dist_to_after,
                                'weight_before': weight_before,
                                'weight_after': weight_after,
                                'mag_before': mag_before_val,
                                'mag_after': mag_after_val,
                                'mag_result': mag_result_val
                            }
                        
                        # Hole Daten von beiden Nachbarn
                        mag_before = balloon['magnitude'][idx_before, :, :]
                        mag_after = balloon['magnitude'][idx_after, :, :]
                        phase_before = balloon['phase'][idx_before, :, :]
                        phase_after = balloon['phase'][idx_after, :, :]
                        
                        # NaN-bewusste Interpolation: Wenn ein Wert NaN ist, nimm den anderen
                        mag_result = np.zeros_like(mag_before)
                        phase_result = np.zeros_like(phase_before)
                        
                        # Beide gültig → Zirkuläre Interpolation
                        both_valid = ~np.isnan(mag_before) & ~np.isnan(mag_after)
                        mag_result[both_valid] = (
                            weight_before * mag_before[both_valid] + 
                            weight_after * mag_after[both_valid]
                        )
                        
                        # Nur "before" gültig → nimm "before"
                        only_before_valid = ~np.isnan(mag_before) & np.isnan(mag_after)
                        mag_result[only_before_valid] = mag_before[only_before_valid]
                        
                        # Nur "after" gültig → nimm "after"
                        only_after_valid = np.isnan(mag_before) & ~np.isnan(mag_after)
                        mag_result[only_after_valid] = mag_after[only_after_valid]
                        
                        # Beide NaN → NaN
                        both_nan = np.isnan(mag_before) & np.isnan(mag_after)
                        mag_result[both_nan] = np.nan
                        
                        new_magnitude[target_idx, :, :] = mag_result
                        
                        # Phase: Zirkuläre Interpolation (berücksichtigt -180°/+180° Wrap)
                        phase_before_rad = np.radians(phase_before)
                        phase_after_rad = np.radians(phase_after)
                        
                        # Phase-Differenz (kürzester zirkulärer Weg über komplexe Zahlen)
                        phase_diff = np.angle(np.exp(1j * (phase_after_rad - phase_before_rad)))
                        
                        # Interpolierte Phase (nur wo beide gültig)
                        phase_result[both_valid] = np.degrees(
                            phase_before_rad[both_valid] + weight_after * phase_diff[both_valid]
                        )
                        phase_result[only_before_valid] = phase_before[only_before_valid]
                        phase_result[only_after_valid] = phase_after[only_after_valid]
                        phase_result[both_nan] = np.nan
                        
                        new_phase[target_idx, :, :] = phase_result
            
            # Wrap Phase zurück auf [-180°, 180°]
            new_phase = ((new_phase + 180) % 360) - 180
            
            # Aktualisiere Struktur
            self.data['balloon_data']['meridians'] = target_meridians
            self.data['balloon_data']['magnitude'] = new_magnitude
            self.data['balloon_data']['phase'] = new_phase
            
            return True
            
        except Exception as e:
            print(f"FEHLER bei sphärischer Interpolation: {e}")
            import traceback
            traceback.print_exc()
            return False
    