"""
Handler für Berechnungslogik und Entscheidungen.

Zentrale Stelle für WANN und WIE berechnet wird (nicht die Berechnung selbst).

Hauptaufgaben:
1. Hash-Check: Speaker-Positionen nur bei Änderung neu berechnen
2. Calculator-Auswahl: FEM oder Superposition basierend auf Settings
"""

import hashlib
import numpy as np

from Module_LFO.Modules_Calculate.SoundfieldCalculator import SoundFieldCalculator
from Module_LFO.Modules_Calculate.SoundFieldCalculator_FEM import SoundFieldCalculatorFEM


class CalculationHandler:
    """
    Entscheidet WANN und WIE berechnet wird.
    CalculationHandler = Brain, Calculator = Hands
    """
    
    def __init__(self, settings):
        self.settings = settings
        # Hash-Cache verhindert unnötige Positionsberechnungen
        self._speaker_position_hashes = {}
    
    def get_physical_params_hash(self, speaker_array):
        """
        Erstellt einen Hash der relevanten physischen Parameter eines speaker_array.
        
        Relevante Parameter:
        - Anzahl der Quellen
        - Lautsprecher-Typen
        - Positionen (x, y, z, z_stack, z_flown)
        - Orientierung (azimuth, site, angle)
        - Arc-Konfiguration (arc_angle, arc_shape, arc_scale_factor)
        
        NICHT enthalten:
        - Pegel (gain, source_level)
        - Delay (delay, source_time)
        - Polarität (source_polarity)
        - Windowing-Parameter
        
        Args:
            speaker_array: Das SpeakerArray-Objekt
            
        Returns:
            str: SHA256 Hash der relevanten Parameter
        """
        def _array_to_string(arr):
            """Konvertiert ein Array zu einem konsistenten String mit gerundeten Werten."""
            if isinstance(arr, np.ndarray):
                # Runde auf 6 Dezimalstellen, um Floating-Point-Ungenauigkeiten zu vermeiden
                return ','.join(f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in arr)
            else:
                return ','.join(str(x) for x in arr)
        
        # Sammle alle relevanten physischen Parameter
        params = []
        
        # Anzahl und Typen
        params.append(str(speaker_array.number_of_sources))
        params.append(_array_to_string(speaker_array.source_polar_pattern))
        
        # Positionen
        if hasattr(speaker_array, 'source_position_x'):
            params.append(_array_to_string(speaker_array.source_position_x))
        if hasattr(speaker_array, 'source_position_y'):
            params.append(_array_to_string(speaker_array.source_position_y))
        if hasattr(speaker_array, 'source_position_z'):
            params.append(_array_to_string(speaker_array.source_position_z))
        if hasattr(speaker_array, 'source_position_z_stack'):
            params.append(_array_to_string(speaker_array.source_position_z_stack))
        if hasattr(speaker_array, 'source_position_z_flown'):
            params.append(_array_to_string(speaker_array.source_position_z_flown))
        
        # Array-Positionen (fixe Offset-Positionen für Stack, absolute Positionen für Flown)
        if hasattr(speaker_array, 'array_position_x'):
            params.append(f"{float(speaker_array.array_position_x):.6f}")
        if hasattr(speaker_array, 'array_position_y'):
            params.append(f"{float(speaker_array.array_position_y):.6f}")
        if hasattr(speaker_array, 'array_position_z'):
            params.append(f"{float(speaker_array.array_position_z):.6f}")
        
        # Orientierung
        if hasattr(speaker_array, 'source_azimuth'):
            params.append(_array_to_string(speaker_array.source_azimuth))
        if hasattr(speaker_array, 'source_site'):
            params.append(_array_to_string(speaker_array.source_site))
        if hasattr(speaker_array, 'source_angle'):
            params.append(_array_to_string(speaker_array.source_angle))
        
        # Arc-Konfiguration (runde auch hier)
        if hasattr(speaker_array, 'arc_angle'):
            params.append(f"{float(speaker_array.arc_angle):.6f}")
        if hasattr(speaker_array, 'arc_shape'):
            params.append(str(speaker_array.arc_shape))
        if hasattr(speaker_array, 'arc_scale_factor'):
            params.append(f"{float(speaker_array.arc_scale_factor):.6f}")
        
        # Erstelle Hash
        param_string = '|'.join(params)
        return hashlib.sha256(param_string.encode('utf-8')).hexdigest()
    
    def should_recalculate_speaker_positions(self, speaker_array, debug=False):
        """
        Prüft via Hash-Vergleich ob Positionsberechnung nötig ist.
        
        Performance-Optimierung: Pegel/Delay-Änderungen triggern keine Neuberechnung.
        Nur physische Parameter (Position, Arc, etc.) ändern den Hash.
        """
        array_id = speaker_array.id
        current_hash = self.get_physical_params_hash(speaker_array)
        
        # Wenn kein gespeicherter Hash existiert, muss berechnet werden
        if array_id not in self._speaker_position_hashes:
            self._speaker_position_hashes[array_id] = current_hash
            if debug:
                print(f"[DEBUG] Kein Hash gespeichert für Array {array_id} - initialer Berechnungslauf")
            return True
        
        # Wenn sich der Hash geändert hat, muss neu berechnet werden
        old_hash = self._speaker_position_hashes[array_id]
        if old_hash != current_hash:
            self._speaker_position_hashes[array_id] = current_hash
            if debug:
                print(f"[DEBUG] Hash-Änderung erkannt für Array {array_id}:")
                print(f"[DEBUG]   Alt: {old_hash[:16]}...")
                print(f"[DEBUG]   Neu: {current_hash[:16]}...")
                print(f"[DEBUG] Physische Parameter: number_of_sources={speaker_array.number_of_sources}, "
                      f"arc_angle={getattr(speaker_array, 'arc_angle', 'N/A')}, "
                      f"arc_shape={getattr(speaker_array, 'arc_shape', 'N/A')}")
            return True
        
        # Keine Änderung erkannt
        return False
    
    def clear_speaker_position_hashes(self):
        """Löscht alle gespeicherten Speaker-Position-Hashes."""
        self._speaker_position_hashes.clear()
    
    def select_soundfield_calculator_class(self):
        """
        Wählt die passende SoundFieldCalculator-Klasse basierend auf den Settings.
        
        Entweder FEM oder Superposition - kein automatischer Fallback!
        Wenn FEM ausgewählt ist und fehlschlägt, wird ein Fehler geworfen.
        
        Returns:
            class: Die zu verwendende Calculator-Klasse (SoundFieldCalculatorFEM oder SoundFieldCalculator)
        """
        use_fem = getattr(self.settings, "spl_plot_fem", False)

        if use_fem:
            return SoundFieldCalculatorFEM
        else:
            return SoundFieldCalculator

