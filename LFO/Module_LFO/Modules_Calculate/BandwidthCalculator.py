import numpy as np

class BandwidthCalculator:
    def __init__(self, settings, data_container):
        """
        Initialisiert den BandwidthCalculator
        
        Args:
            settings: Einstellungen
            data_container: DataContainer-Objekt
        """
        self.settings = settings
        self.data = data_container  # DataContainer-Objekt
        self.data_container = data_container  # Direkter Zugriff

    def _get_balloon_data(self, speaker_name):
        """
        🚀 OPTIMIERT: Direkter Zugriff über DataContainer
        """
        return self.data_container.get_balloon_data(speaker_name)

    def _get_correct_horizontal_index(self, azimuth_deg):
        """
        🔧 KORRIGIERT: Gleiche Index-Zuordnung wie ImpulseCalculator
        Normalisiert Azimut auf 0-360 Grad und rundet auf ganze Zahl
        """
        azimuth = azimuth_deg % 360
        return int(round(azimuth)) % 360

    def calculate_magnitude_average(self, lower_freq, upper_freq):
        """
        Berechnet den Durchschnitt der Magnitude im gewählten Frequenzbereich für alle Lautsprecher
        und speichert sie unter dem Schlüssel 'balloon_avg_mag_{speaker_name}'
        
        Args:
            lower_freq (float): Untere Frequenzgrenze in Hz
            upper_freq (float): Obere Frequenzgrenze in Hz
        """
        
        # 🔧 CACHE-REINIGUNG: Leere nur gemittelte Daten beim Start
        if not hasattr(self.data_container, '_balloon_cache'):
            self.data_container._balloon_cache = {}
        
        # Lösche NUR alte gemittelte Daten für alle Speaker (NICHT die originalen Balloon-Daten!)
        speaker_names = self.data_container.get_speaker_names()
        for speaker_name in speaker_names:
            # Lösche NUR gemittelte Daten aus DataContainer
            data_dict = self.data.get_data()
            keys_to_remove = [k for k in data_dict.keys() if f'balloon_avg_mag_{speaker_name}' in k or f'balloon_avg_phase_{speaker_name}' in k]
            for key in keys_to_remove:
                del data_dict[key]
            
            # Lösche NUR gemittelte Daten aus Cache
            cache_keys_to_remove = [k for k in self.data_container._balloon_cache.keys() if f'avg_mag_{speaker_name}' in k or f'avg_phase_{speaker_name}' in k]
            for key in cache_keys_to_remove:
                del self.data_container._balloon_cache[key]
        
        
        speaker_names = self.data_container.get_speaker_names()
        
        if not speaker_names:
            pass
            return
        
        # Verarbeite jeden Lautsprecher
        for speaker_name in speaker_names:
            # 🚀 OPTIMIERT: Einheitlicher Datenzugriff
            balloon_data = self._get_balloon_data(speaker_name)
            
            if balloon_data is None:
                pass
                continue
            
            if not isinstance(balloon_data, dict):
                pass
                continue
                
            balloon_mag = balloon_data.get('magnitude')
            balloon_freqs = balloon_data.get('freqs')
            
            if balloon_mag is None or balloon_freqs is None:
                pass
                continue
                
            # Finde Indizes für den gewünschten Frequenzbereich (sichere numpy Konvertierung)
            balloon_freqs = np.array(balloon_freqs)  # Sicherstellen dass es ein numpy array ist
            
            # 🔧 SPEZIALFALL: Wenn lower_freq == upper_freq, verwende nächstliegende Frequenz ohne Mittelung
            if lower_freq == upper_freq:
                # Finde nächstliegende Frequenz
                freq_idx = np.argmin(np.abs(balloon_freqs - lower_freq))
                actual_freq = balloon_freqs[freq_idx]
                
                
                # Kopiere Magnitude und Phase direkt ohne Mittelung
                averaged_mag = balloon_mag[:, :, freq_idx].copy()
                averaged_phase = balloon_data['phase'][:, :, freq_idx].copy()
                
                # Erstelle Dictionary für die Daten
                avg_balloon_data = {
                    'magnitude': averaged_mag,
                    'phase': averaged_phase,
                    'vertical_angles': balloon_data.get('vertical_angles'),
                    'horizontal_angles': balloon_data.get('horizontal_angles')
                }
                
                # Speichere gemittelte Daten im gleichen Format wie die Balloon-Daten
                data_dict = self.data.get_data()
                data_dict[f'balloon_avg_mag_{speaker_name}'] = avg_balloon_data
                data_dict[f'balloon_avg_phase_{speaker_name}'] = avg_balloon_data
                
                # 🚀 CACHE: Speichere im Performance-Cache
                self.data_container._balloon_cache[f'avg_mag_{speaker_name}'] = avg_balloon_data
                self.data_container._balloon_cache[f'avg_phase_{speaker_name}'] = avg_balloon_data
                
                continue  # Überspringe normale Mittelung
            
            # Normale Frequenzbereichs-Mittelung
            freq_mask = (balloon_freqs >= lower_freq) & (balloon_freqs <= upper_freq)
            freq_indices = np.where(freq_mask)[0]
            

            
            if len(freq_indices) == 0:
                pass
                continue
            
            try:
                # 🚀 VEKTORISIERT: Echte SPL-gewichtete Magnitude-Mittelung
                # Extrahiere relevante Frequenzen für ALLE Winkel: [vert, horz, freq_indices]
                relevant_magnitudes = balloon_mag[:, :, freq_indices]
                
                # 🔧 KORRIGIERT: Echte energetische Mittelung (SPL-gewichtet)
                # Konvertiere dB zu linear für korrekte energetische Mittelung
                linear_magnitudes = 10 ** (relevant_magnitudes / 20)
                
                # Berechne gewichtete Mittelung basierend auf linearer Magnitude
                # Frequenzen mit höherer Magnitude haben mehr Gewicht
                linear_weights = linear_magnitudes
                weighted_mag_sum = np.sum(relevant_magnitudes * linear_weights, axis=2)
                weight_sum = np.sum(linear_weights, axis=2)
                
                # Vermeide Division durch Null
                weight_sum = np.where(weight_sum == 0, 1, weight_sum)
                averaged_mag = weighted_mag_sum / weight_sum
                        
                
                # Erstelle ein neues Dictionary für die gemittelten Daten
                # Übernehme horizontal_angles direkt aus den ursprünglichen Daten
                avg_balloon_data = {
                    'magnitude': averaged_mag,
                    'vertical_angles': balloon_data.get('vertical_angles'),
                    'horizontal_angles': balloon_data.get('horizontal_angles')
                }
                
                # Speichere gemittelte Daten im gleichen Format wie die Balloon-Daten
                data_dict = self.data.get_data()
                data_dict[f'balloon_avg_mag_{speaker_name}'] = avg_balloon_data
                
                
                # 🚀 CACHE: Speichere im Performance-Cache
                self.data_container._balloon_cache[f'avg_mag_{speaker_name}'] = avg_balloon_data
                
            except Exception as e:
                pass
                continue
        
        # Zusätzlich: Erstelle auch die alten 'polar_mag' Daten für Kompatibilität
        # self._create_polar_mag_from_balloon_avg()
        

    def calculate_phase_average(self, lower_freq, upper_freq):
        """
        Berechnet den Durchschnitt der Phase im gewählten Frequenzbereich für alle Lautsprecher
        und speichert sie unter dem Schlüssel 'balloon_avg_phase_{speaker_name}'
        
        Args:
            lower_freq (float): Untere Frequenzgrenze in Hz
            upper_freq (float): Obere Frequenzgrenze in Hz
            
        Note:
            - Phase wird SPL-GEWICHTET gemittelt (basierend auf linearer Magnitude)
            - Frequenzen mit höherem SPL haben mehr Einfluss
            - Eingangsdaten sind UNWRAPPED in GRAD
            - Ausgabe bleibt UNWRAPPED in GRAD
        """
        speaker_names = self.data_container.get_speaker_names()
        
        if not speaker_names:
            pass
            return
        
        # Verarbeite jeden Lautsprecher
        for speaker_name in speaker_names:
            # 🚀 OPTIMIERT: Einheitlicher Datenzugriff
            balloon_data = self._get_balloon_data(speaker_name)
            
            if balloon_data is None:
                pass
                continue
            
            if not isinstance(balloon_data, dict):
                pass
                continue
                
            balloon_mag = balloon_data.get('magnitude')
            balloon_freqs = balloon_data.get('freqs')
            
            if balloon_mag is None or balloon_freqs is None:
                pass
                continue
                
            # Finde Indizes für den gewünschten Frequenzbereich (sichere numpy Konvertierung)
            balloon_freqs = np.array(balloon_freqs)  # Sicherstellen dass es ein numpy array ist
            
            # 🔧 SPEZIALFALL: Wenn lower_freq == upper_freq, verwende nächstliegende Frequenz ohne Mittelung
            if lower_freq == upper_freq:
                # Finde nächstliegende Frequenz
                freq_idx = np.argmin(np.abs(balloon_freqs - lower_freq))
                actual_freq = balloon_freqs[freq_idx]
                
                # Kopiere Phase direkt ohne Mittelung
                averaged_phase = balloon_data['phase'][:, :, freq_idx].copy()
                
                # Erstelle Dictionary für die Daten
                avg_balloon_data = {
                    'phase': averaged_phase,
                    'vertical_angles': balloon_data.get('vertical_angles'),
                    'horizontal_angles': balloon_data.get('horizontal_angles')
                }
                
                # Speichere gemittelte Daten im gleichen Format wie die Balloon-Daten
                data_dict = self.data.get_data()
                data_dict[f'balloon_avg_phase_{speaker_name}'] = avg_balloon_data
                
                # 🚀 CACHE: Speichere im Performance-Cache
                self.data_container._balloon_cache[f'avg_phase_{speaker_name}'] = avg_balloon_data
                
                continue  # Überspringe normale Mittelung
            
            # Normale Frequenzbereichs-Mittelung
            freq_mask = (balloon_freqs >= lower_freq) & (balloon_freqs <= upper_freq)
            freq_indices = np.where(freq_mask)[0]
            
            if len(freq_indices) == 0:
                pass
                continue
            
            try:
                # 🚀 VEKTORISIERT: Echte SPL-gewichtete Phase-Mittelung
                # Extrahiere relevante Daten für ALLE Winkel: [vert, horz, freq_indices]
                relevant_phases = balloon_data['phase'][:, :, freq_indices]
                relevant_magnitudes = balloon_mag[:, :, freq_indices]
                
                # 🔧 DEBUG: Teste einfache Mittelung vs. gewichtete Mittelung
                # Für Test6: Zeige was passiert
                if str(speaker_name).lower() == "test6":
                    # Einfache arithmetische Mittelung (ohne Gewichtung)
                    simple_avg = np.mean(relevant_phases, axis=2)
                    
                    # SPL-gewichtete Mittelung
                    linear_weights = 10 ** (relevant_magnitudes / 20)
                    weighted_phase_sum = np.sum(relevant_phases * linear_weights, axis=2)
                    weight_sum = np.sum(linear_weights, axis=2)
                    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
                    weighted_avg = weighted_phase_sum / weight_sum
                    
                    # Debug-Ausgabe für v≈0°, h=0° - korrekte Index-Zuordnung
                    v_angles = balloon_data.get('vertical_angles')
                    v_idx = int(np.abs(np.array(v_angles) - 0.0).argmin())
                    h_idx = int(round(0)) % 360  # Korrekte Rundung wie ImpulseCalculator
                    
                    # 🔧 CSV-EXPORT: Speichere Debug-Daten in CSV
                    import csv
                    csv_filename = f"debug_phase_averaging_{speaker_name}.csv"
                    
                    # H=0° - korrekte Index-Zuordnung
                    h_idx_0 = self._get_correct_horizontal_index(0)
                    phases_h0 = relevant_phases[v_idx, h_idx_0, :]
                    mags_h0 = relevant_magnitudes[v_idx, h_idx_0, :]
                    simple_avg_h0 = np.mean(phases_h0)
                    
                    # H=-1° - korrekte Index-Zuordnung
                    h_idx_minus1 = self._get_correct_horizontal_index(-1)
                    phases_h359 = relevant_phases[v_idx, h_idx_minus1, :]
                    mags_h359 = relevant_magnitudes[v_idx, h_idx_minus1, :]
                    simple_avg_h359 = np.mean(phases_h359)
                    
                    # 🔧 CSV-EXPORT: Lade auch NPZ-Daten für Vergleich
                    npz_phases_h0 = None
                    npz_phases_h359 = None
                    try:
                        import os
                        npz_path = f"Module_LFO/Modules_Data/PolarData_mag/{speaker_name}.npz"
                        if os.path.exists(npz_path):
                            npz_data = np.load(npz_path, allow_pickle=True)
                            if 'balloon_data' in npz_data:
                                balloon_npz = npz_data['balloon_data'].item()
                                if 'phase' in balloon_npz and 'freqs' in balloon_npz:
                                    npz_phases = balloon_npz['phase']
                                    npz_freqs = balloon_npz['freqs']
                                    
                                    # Finde Frequenzindizes in NPZ
                                    freq_mask_npz = (npz_freqs >= lower_freq) & (npz_freqs <= upper_freq)
                                    npz_freq_indices = np.where(freq_mask_npz)[0]
                                    
                                    if len(npz_freq_indices) > 0:
                                        npz_phases_h0 = npz_phases[v_idx, h_idx_0, npz_freq_indices]
                                        npz_phases_h359 = npz_phases[v_idx, h_idx_minus1, npz_freq_indices]
                    except Exception as e:
                        pass
                    
                    # 🔧 CSV-EXPORT: Erstelle CSV-Datei mit allen Details
                    csv_rows = []
                    csv_rows.append(['=== PHASE AVERAGING DEBUG ==='])
                    csv_rows.append(['Speaker', speaker_name])
                    csv_rows.append(['Frequenzbereich', f'{lower_freq}-{upper_freq} Hz'])
                    csv_rows.append(['Anzahl Frequenzen', len(freq_indices)])
                    csv_rows.append(['Magnitude-Bereich', f'{np.min(relevant_magnitudes):.1f} - {np.max(relevant_magnitudes):.1f} dB'])
                    csv_rows.append([])
                    
                    # Zeige NPZ-Status
                    if npz_phases_h0 is not None:
                        csv_rows.append(['NPZ-Daten', 'VERFÜGBAR'])
                        csv_rows.append(['NPZ Anzahl Frequenzen', len(npz_phases_h0)])
                    else:
                        csv_rows.append(['NPZ-Daten', 'NICHT VERFÜGBAR'])
                    csv_rows.append([])
                    
                    # === ABSCHNITT 1: Frequenz-Details ===
                    csv_rows.append(['=== FREQUENZ-DETAILS (alle Frequenzen) ==='])
                    csv_rows.append(['Frequenz (Hz)', 'Import H=0° Phase', 'Import H=0° Mag', 'Import H=-1° Phase', 'Import H=-1° Mag', 
                                   'NPZ H=0° Phase', 'NPZ H=-1° Phase', 'Diff Import H0-H359', 'Diff NPZ H0-H359', 'NPZ vs Import H=0°', 'NPZ vs Import H=-1°'])
                    
                    # Alle Frequenzen
                    relevant_freqs = balloon_freqs[freq_indices]
                    for i in range(len(freq_indices)):
                        freq_val = relevant_freqs[i]
                        phase_h0 = phases_h0[i]
                        mag_h0 = mags_h0[i]
                        phase_h359 = phases_h359[i]
                        mag_h359 = mags_h359[i]
                        diff_import = phase_h0 - phase_h359
                        
                        if npz_phases_h0 is not None and i < len(npz_phases_h0):
                            npz_h0 = npz_phases_h0[i]
                            npz_h359 = npz_phases_h359[i]
                            diff_npz = npz_h0 - npz_h359
                            diff_npz_vs_import_h0 = npz_h0 - phase_h0
                            diff_npz_vs_import_h359 = npz_h359 - phase_h359
                            csv_rows.append([freq_val, phase_h0, mag_h0, phase_h359, mag_h359, 
                                           npz_h0, npz_h359, diff_import, diff_npz,
                                           diff_npz_vs_import_h0, diff_npz_vs_import_h359])
                        else:
                            csv_rows.append([freq_val, phase_h0, mag_h0, phase_h359, mag_h359, 
                                           '', '', diff_import, '', '', ''])
                    
                    csv_rows.append([])
                    csv_rows.append(['Winkel', 'Einfache Mittelung', 'SPL-gewichtete Mittelung', 'Differenz'])
                    
                    # Winkel von -10 bis +10
                    for h_deg in range(-10, 11):
                        h_idx = self._get_correct_horizontal_index(h_deg)
                        simple = simple_avg[v_idx, h_idx]
                        weighted = weighted_avg[v_idx, h_idx]
                        diff = weighted - simple
                        csv_rows.append([h_deg, f'{simple:.2f}', f'{weighted:.2f}', f'{diff:.2f}'])
                    
                    # Schreibe CSV
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_rows)
                    
                    # 🔍 ERWEITERT: Untersuche Frequenzbereich 53-56 Hz
                    try:
                        # Finde Indizes für 53-56 Hz in den RELEVANTEN Frequenzen
                        relevant_freqs = balloon_freqs[freq_indices]  # Nur die relevanten Frequenzen
                        freq_mask_53_56 = (relevant_freqs >= 53) & (relevant_freqs <= 56)
                        freq_indices_53_56 = np.where(freq_mask_53_56)[0]  # Indizes in relevant_phases
                        
                        if len(freq_indices_53_56) > 0:
                            # Extrahiere Phasen für diesen Bereich - korrekte Indizes
                            h_idx_0 = self._get_correct_horizontal_index(0)
                            h_idx_minus1 = self._get_correct_horizontal_index(-1)
                            phases_h0_53_56 = relevant_phases[v_idx, h_idx_0, freq_indices_53_56]
                            phases_h359_53_56 = relevant_phases[v_idx, h_idx_minus1, freq_indices_53_56]
                            
                            # Berechne Mittelung nur für 53-56 Hz
                            avg_h0_53_56 = np.mean(phases_h0_53_56)
                            avg_h359_53_56 = np.mean(phases_h359_53_56)
                            
                            # 🔧 CSV-EXPORT: Erweiterte 53-56 Hz Daten
                            csv_filename_53_56 = f"debug_phase_53_56Hz_{speaker_name}.csv"
                            csv_rows_53_56 = []
                            csv_rows_53_56.append(['Frequenzbereich 53-56 Hz', f'{len(freq_indices_53_56)} Punkte'])
                            csv_rows_53_56.append([])
                            csv_rows_53_56.append(['Frequenz (Hz)', 'H=0° Phase', 'H=-1° Phase', 'Differenz'])
                            
                            for i in range(len(freq_indices_53_56)):
                                freq_val = relevant_freqs[freq_indices_53_56[i]]
                                phase_h0 = phases_h0_53_56[i]
                                phase_h359 = phases_h359_53_56[i]
                                diff = phase_h0 - phase_h359
                                csv_rows_53_56.append([f'{freq_val:.2f}', f'{phase_h0:.2f}', f'{phase_h359:.2f}', f'{diff:.2f}'])
                            
                            csv_rows_53_56.append([])
                            csv_rows_53_56.append(['Mittelung 53-56 Hz'])
                            csv_rows_53_56.append(['H=0°', f'{avg_h0_53_56:.2f}'])
                            csv_rows_53_56.append(['H=-1°', f'{avg_h359_53_56:.2f}'])
                            csv_rows_53_56.append(['Differenz', f'{avg_h0_53_56 - avg_h359_53_56:.2f}'])
                            
                            # Schreibe CSV
                            with open(csv_filename_53_56, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerows(csv_rows_53_56)
                    except Exception as e:
                        pass
                    
                    # Verwende einfache Mittelung für Test
                    averaged_phase = simple_avg
                else:
                    # Normale SPL-gewichtete Phase-Mittelung (für unwrapped Phase)
                    linear_weights = 10 ** (relevant_magnitudes / 20)
                    weighted_phase_sum = np.sum(relevant_phases * linear_weights, axis=2)
                    weight_sum = np.sum(linear_weights, axis=2)
                    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
                    averaged_phase = weighted_phase_sum / weight_sum
                
                
                # Erstelle ein neues Dictionary für die gemittelten Daten
                # Übernehme horizontal_angles direkt aus den ursprünglichen Daten
                avg_balloon_data = {
                    'phase': averaged_phase,
                    'vertical_angles': balloon_data.get('vertical_angles'),
                    'horizontal_angles': balloon_data.get('horizontal_angles')
                }
                
                # Speichere gemittelte Daten im gleichen Format wie die Balloon-Daten
                data_dict = self.data.get_data()
                data_dict[f'balloon_avg_phase_{speaker_name}'] = avg_balloon_data
                

                
                # 🚀 CACHE: Speichere im Performance-Cache
                self.data_container._balloon_cache[f'avg_phase_{speaker_name}'] = avg_balloon_data
                
                # Debug deaktiviert
                # if str(speaker_name).lower() == "test6":
                #     self._debug_phase_comparison(...)
                
            except Exception as e:
                pass
                continue

    def _debug_phase_comparison(self, speaker_name, lower_freq, upper_freq, 
                              relevant_phases, relevant_magnitudes, 
                              balloon_freqs, freq_indices, averaged_phase):
        """
        Zeigt Phasenvergleich zwischen Dominanz-Methode und SPL-gewichteter Mittelung
        für Test6 bei ±10° horizontal
        """
        print(f"\n🔍 PHASEN-VERGLEICH für {speaker_name}")
        print(f"🎵 Frequenzbereich: {lower_freq} - {upper_freq} Hz")
        print(f"📐 Winkelbereich: ±10° horizontal, v≈0°")
        print("=" * 70)
        
        # Finde v≈0° Index
        balloon_data = self._get_balloon_data(speaker_name)
        v_angles = balloon_data.get('vertical_angles')
        v_idx = int(np.abs(np.array(v_angles) - 0.0).argmin())
        
        # METHODE 1: Dominanz-Methode (alte)
        avg_mag_per_freq = np.mean(relevant_magnitudes, axis=(0, 1))
        global_dominant_freq_idx = np.argmax(avg_mag_per_freq)
        phase_dominance = relevant_phases[:, :, global_dominant_freq_idx]
        dominant_freq = balloon_freqs[freq_indices[global_dominant_freq_idx]]
        
        print(f"🔴 DOMINANZ-METHODE:")
        print(f"   🎵 Dominante Frequenz: {dominant_freq:.1f} Hz")
        print(f"   📊 Phase wird von EINER Frequenz übernommen")
        print()
        
        print(f"🟢 KOMPLEXE SPL-GEWICHTETE MITTELUNG:")
        print(f"   📊 Phase wird aus {len(freq_indices)} Frequenzen gemittelt")
        print(f"   ⚖️  Gewichtung basierend auf linearer Magnitude")
        print(f"   🔄 Zyklische Mittelung mit komplexen Zahlen")
        print()
        
        # Vergleichstabelle
        print("📊 VERGLEICH: ±10° horizontal (v≈0°)")
        print("-" * 70)
        print(f"{'H°':>4} | {'Phase_Dom':>10} | {'Phase_Wgt':>10} | {'Diff':>8}")
        print("-" * 70)
        
        for h_deg in range(-10, 11):
            # 🔧 KORRIGIERT: Verwende gleiche Rundungslogik wie ImpulseCalculator
            h_idx = self._get_correct_horizontal_index(h_deg)
            
            # Werte extrahieren
            phs_dom = phase_dominance[v_idx, h_idx]
            phs_wgt = averaged_phase[v_idx, h_idx]
            diff = phs_wgt - phs_dom
            
            print(f"{h_deg:>4} | {phs_dom:>10.1f} | {phs_wgt:>10.1f} | {diff:>8.1f}")
        
        print("-" * 70)
        print("📝 Legende:")
        print("   Phase_Dom: Phase der dominanten Frequenz")
        print("   Phase_Wgt: Komplexe SPL-gewichtete Phasen-Mittelung")
        print("   Diff: Differenz (Gewichtet - Dominant)")
        print()

    def get_frequency_indices(self, frequencies, lower_freq, upper_freq):
        """
        Hilfsfunktion zum Finden der Frequenzindizes im gewählten Bereich
        """
        freq_mask = (frequencies >= lower_freq) & (frequencies <= upper_freq)
        return np.where(freq_mask)[0]

    # -------------------------------------------------------------------------
    # DEBUG-HILFE: Werte bei v=0° und h=±20° ausgeben
    # -------------------------------------------------------------------------
    def debug_print_zero_vertical_pm20(self, speaker_name, max_freq=400.0, max_rows=12):
        """
        Gibt die Balloon-Daten (Magnitude/Phase) für v≈0° und h=-20°/+20° aus.
        Phase in Grad (unwrapped), Magnitude in dB. Frequenzen bis max_freq.

        Args:
            speaker_name: Name des Lautsprechers (gemäß NPZ/Cache)
            max_freq: obere Frequenzgrenze für die Ausgabe
            max_rows: maximale Zeilenanzahl für die tabellarische Ausgabe
        """
        try:
            balloon = self._get_balloon_data(speaker_name)
            if balloon is None or not isinstance(balloon, dict):
                return
                return

            v_angles = balloon.get('vertical_angles')
            mag = balloon.get('magnitude')
            phase = balloon.get('phase')
            freqs = np.array(balloon.get('freqs'))

            if v_angles is None or mag is None or phase is None or freqs is None:
                return
                return

            # Indizes bestimmen
            v_idx = int(np.abs(np.array(v_angles) - 0.0).argmin())  # v≈0°
            h_idx_n20 = (-20) % 360
            h_idx_p20 = (20) % 360

            # Frequenzbegrenzung
            cutoff = np.where(freqs <= max_freq)[0]
            if cutoff.size == 0:
                return
                return
            cutoff_len = cutoff[-1] + 1

            # Daten extrahieren
            mag_n20 = mag[v_idx, h_idx_n20, :cutoff_len]
            phs_n20 = phase[v_idx, h_idx_n20, :cutoff_len]
            mag_p20 = mag[v_idx, h_idx_p20, :cutoff_len]
            phs_p20 = phase[v_idx, h_idx_p20, :cutoff_len]
            f_out = freqs[:cutoff_len]

            # Ausgabe begrenzen
            rows = min(max_rows, len(f_out))

            pass
            # Zeige für jeden horizontalen Grad von -20 bis +20 jeweils die erste 'rows'-Zeilen
            for h_deg in range(-20, 21):
                h_idx = h_deg % 360
                mag_h = mag[v_idx, h_idx, :cutoff_len]
                phs_h = phase[v_idx, h_idx, :cutoff_len]
                for i in range(rows):
                    pass

            # Kurzstatistik
            pass

        except Exception as e:
            pass

    def calculate_third_octave_band_limits(self, center_freq):
        """
        Berechnet die Grenzfrequenzen eines Terzbandes nach IEC 61260
        
        Args:
            center_freq (float): Mittenfrequenz in Hz
            
        Returns:
            tuple: (lower_freq, upper_freq) in Hz
        """
        # Terzband: Bandbreite = Mittenfrequenz / 2^(1/6) bis Mittenfrequenz * 2^(1/6)
        # Exakter Faktor für Terz: 2^(1/6) ≈ 1.1225
        factor = 2 ** (1/6)
        lower_freq = center_freq / factor
        upper_freq = center_freq * factor
        return lower_freq, upper_freq

    def calculate_polar_frequency_averages(self, polar_frequencies):
        """
        🚀 OPTIMIERT: Berechnet Terzband-Mittelungen für ALLE Polar-Frequenzen
        
        Für jede Polar-Frequenz (z.B. 50 Hz) wird ein Terzband berechnet
        (z.B. 44.7 Hz - 56.2 Hz) und die Magnitude/Phase-Daten werden
        energetisch bzw. SPL-gewichtet gemittelt.
        
        Args:
            polar_frequencies (dict): Dictionary mit {color: frequency} Paaren
                                     z.B. {'red': 31.5, 'yellow': 40, 'green': 50, 'cyan': 63}
        """
        speaker_names = self.data_container.get_speaker_names()
        
        if not speaker_names:
            return
        
        # Verarbeite jede Polar-Frequenz
        for color, center_freq in polar_frequencies.items():
            # 🔧 SPEZIALFALL: Für einzelne Frequenzen verwende keine Terzband-Mittelung
            # Verwende die Frequenz direkt ohne Mittelung
            lower_freq = center_freq
            upper_freq = center_freq
            
            # Verarbeite jeden Lautsprecher
            for speaker_name in speaker_names:
                # 🚀 OPTIMIERT: Einheitlicher Datenzugriff
                balloon_data = self._get_balloon_data(speaker_name)
                
                if balloon_data is None or not isinstance(balloon_data, dict):
                    continue
                    
                balloon_mag = balloon_data.get('magnitude')
                balloon_phase = balloon_data.get('phase')
                balloon_freqs = balloon_data.get('freqs')
                
                if balloon_mag is None or balloon_phase is None or balloon_freqs is None:
                    continue
                
                # Finde Indizes für den Terzband-Bereich
                balloon_freqs = np.array(balloon_freqs)
                freq_mask = (balloon_freqs >= lower_freq) & (balloon_freqs <= upper_freq)
                freq_indices = np.where(freq_mask)[0]
                
                if len(freq_indices) == 0:
                    continue
                
                try:
                    # 🚀 VEKTORISIERT: Magnitude-Mittelung (energetisch)
                    relevant_magnitudes = balloon_mag[:, :, freq_indices]
                    linear_magnitudes = 10 ** (relevant_magnitudes / 20)
                    averaged_linear = np.mean(linear_magnitudes, axis=2)
                    averaged_mag = 20 * np.log10(averaged_linear)
                    
                    # 🚀 VEKTORISIERT: Phase-Mittelung (SPL-gewichtet)
                    relevant_phases = balloon_data['phase'][:, :, freq_indices]
                    linear_weights = 10 ** (relevant_magnitudes / 20)
                    weighted_phase_sum = np.sum(relevant_phases * linear_weights, axis=2)
                    weight_sum = np.sum(linear_weights, axis=2)
                    averaged_phase = weighted_phase_sum / weight_sum
                    
                    # Erstelle Dictionary mit BEIDEN Magnitude und Phase
                    avg_balloon_data = {
                        'magnitude': averaged_mag,
                        'phase': averaged_phase,
                        'vertical_angles': balloon_data.get('vertical_angles'),
                        'horizontal_angles': balloon_data.get('horizontal_angles'),
                        'freqs': np.array([center_freq])  # Speichere die Mittenfrequenz
                    }

                    # -------------------------------------------
                    # DEBUG-SNAPSHOT für File "Test" bei ~50 Hz
                    # Zeige Original-Phasen (unmittelbar aus Balloon) und
                    # die gemittelten Phasen (averaged_phase) an v≈0° für h=-20..+20°
                    # -------------------------------------------
                    try:
                        if str(speaker_name).lower() == "test":
                            target_f = 50.0
                            # Originaldaten: Nimm den nächsten Frequenzindex zu 50 Hz
                            f_idx_raw = int(np.abs(balloon_freqs - target_f).argmin())
                            v_angles = balloon_data.get('vertical_angles')
                            v_idx = int(np.abs(np.array(v_angles) - 0.0).argmin())
                            pass
                            for h_deg in range(-20, 21):
                                h_idx = h_deg % 360
                                ph_orig = balloon_data['phase'][v_idx, h_idx, f_idx_raw]
                                ph_avg = averaged_phase[v_idx, h_idx]
                                pass
                    except Exception as _e:
                        pass
                    
                    # Speichere unter eindeutigem Key
                    key = f'balloon_polar_{center_freq}Hz_{speaker_name}'
                    data_dict = self.data.get_data()
                    data_dict[key] = avg_balloon_data
                    
                    # 🚀 CACHE: Speichere im Performance-Cache
                    self.data_container._balloon_cache[f'polar_{center_freq}Hz_{speaker_name}'] = avg_balloon_data
                    
                except Exception as e:
                    continue



