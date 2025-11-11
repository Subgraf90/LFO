import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi
from Module_LFO.Modules_Plot.PlotStacks2SourceLayout import StackDraw_SourceLayout
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt

# Widget zur Darstellung der Lautsprecheraufstellung

class Draw_Source_Layout_Widget(ModuleBase):
    def __init__(self, matplotlib_canvas_source_layout_widget, settings, container):
        super().__init__(settings)  # Korrekte Initialisierung der Basisklasse "Modulebase"
        
        self.ax = matplotlib_canvas_source_layout_widget
        self.im = None
        self.settings = settings
        self.container = container
        
        # Erstelle Toolbar und setze sie oben rechts
        if not hasattr(self, 'toolbar'):
            self.toolbar = NavigationToolbar(self.ax.figure.canvas, self.ax.figure.canvas.parent())
            self.toolbar.setFloatable(True)
            self.toolbar.setMovable(True)
            
            # Setze die Position oben rechts
            parent = self.ax.figure.canvas.parent()
            parent_width = parent.width()
            toolbar_width = self.toolbar.width()
            self.toolbar.move(parent_width - toolbar_width, 0)
            
            # Halte die Toolbar oben rechts auch bei Fenstergrößenänderung
            parent.resizeEvent = lambda event: self.toolbar.move(
                parent.width() - self.toolbar.width(), 0
            )

    def update_source_layout_widget(self, selected_speaker_array_id):
        try:
            if selected_speaker_array_id is None:
                print("Kein Lautsprecherarray ausgewählt.")
                return

            self.ax.clear()
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            cabinet_data = self.container.data['cabinet_data']
            speaker_names = self.container.data['speaker_names']
            
            if speaker_array is None:
                return

            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            speaker_indices = []  # Speichere die Indizes der gültigen Lautsprecher
            
            for i, pattern_name in enumerate(speaker_array.source_polar_pattern):
                if not pattern_name:
                    continue
                    
                try:
                    speaker_index = speaker_names.index(pattern_name)
                    cabinet = cabinet_data[speaker_index]
                    
                    # Akzeptiere np.ndarray, list und dict
                    if isinstance(cabinet, (np.ndarray, list, dict)):
                        # Konvertiere zu np.ndarray für einheitliche Verarbeitung
                        if isinstance(cabinet, dict):
                            cabinet = np.array([cabinet])
                        elif isinstance(cabinet, list):
                            cabinet = np.array(cabinet)
                        array_cabinets.append(cabinet)
                        speaker_indices.append(i)  # Speichere den Index
                    else:
                        print(f"Cabinet-Daten für '{pattern_name}' haben falsches Format: {type(cabinet)}")
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Abrufen der Cabinet-Daten für '{pattern_name}': {str(e)}")
                    continue

            if array_cabinets:
                stack_draw = StackDraw_SourceLayout(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    width=1, 
                    length=2,
                    window_restriction=0,
                    ax=self.ax,
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_draw.draw_stack(isrc)

                # Identifiziere die Stacks basierend auf den X-Positionen
                x_positions = [speaker_array.source_position_x[i] for i in speaker_indices]
                
                # Gruppiere Lautsprecher in Stacks (Lautsprecher mit gleicher X-Position)
                stacks = {}
                for i, x in enumerate(x_positions):
                    # Runde auf 2 Dezimalstellen, um kleine Unterschiede zu ignorieren
                    x_rounded = round(x, 2)
                    if x_rounded not in stacks:
                        stacks[x_rounded] = []
                    stacks[x_rounded].append(speaker_indices[i])
                                
                # Berechne die linke Position jedes Stacks
                stack_left_positions = []
                stack_left_positions_for_measurement = []
                for x_pos, indices in stacks.items():
                    # Sammle alle Breiten der Lautsprecher im Stack
                    breiten = []
                    for idx in indices:
                        speaker_name = speaker_array.source_polar_pattern[idx]
                        try:
                            speaker_idx = speaker_names.index(speaker_name)
                            cabinet = cabinet_data[speaker_idx]
                            if isinstance(cabinet, dict):
                                breiten.append(float(cabinet.get('width', 0)))
                            elif isinstance(cabinet, (np.ndarray, list)) and len(cabinet) > 0:
                                for cab in cabinet:
                                    if isinstance(cab, dict):
                                        breiten.append(float(cab.get('width', 0)))
                        except Exception as e:
                            print(f'Fehler bei Breitenberechnung: {e}')
                            continue

                    gesamtbreite = sum(breiten)
                    left_pos = x_pos - (gesamtbreite / 2)
                    stack_left_positions.append(left_pos)
                    stack_left_positions_for_measurement.append(left_pos)
                
                if stack_left_positions_for_measurement:
                    # Sortiere die Positionen
                    stack_left_positions_for_measurement.sort()
                    first_position = stack_left_positions_for_measurement[0]
                    for i, x in enumerate(stack_left_positions_for_measurement):
                        self.ax.plot([x, x], [0.1, 2], color='black', linestyle='--')
                        distance = np.abs(x - first_position)
                        self.ax.annotate(f"{distance:.2f} m", (x, 2.2), color='black', weight='bold', fontsize=10, ha='center')
                    x_middle = 0
                    self.ax.plot([x_middle, x_middle], [0.1, 3.0], color='red', linestyle='--')
                    middle_distance = np.abs(x_middle - first_position)
                    self.ax.annotate(f"{middle_distance:.2f} m", (x_middle, 3.0), color='red', weight='bold', fontsize=12, ha='center')
                else:
                    print("DEBUG: Keine linken Positionen der Stacks für Bemaßung gefunden!")
            else:
                print("DEBUG: Keine Cabinet-Daten gefunden!")

            # Achsen-Einstellungen
            self.ax.xaxis.set_major_locator(MultipleLocator(3)) 
            self.ax.yaxis.set_major_locator(MultipleLocator(3))
            self.ax.axis('equal')
            self.ax.grid(color='k', linestyle='-', linewidth=0.3)
            self.ax.set_xlabel("Array length [m]", fontsize=9)
            self.ax.tick_params(axis='y', labelsize=7)
            self.ax.tick_params(axis='x', labelsize=7)
            
            # Toolbar
            if not hasattr(self, 'toolbar'):
                self.toolbar = NavigationToolbar(self.ax.figure.canvas, self.ax.figure.canvas.parent())
            
            self.ax.figure.canvas.draw()
            
            # Aktualisiere Toolbar-Position
            parent = self.ax.figure.canvas.parent()
            self.toolbar.move(parent.width() - self.toolbar.width(), 0)
            
        except Exception as e:
            print(f"Fehler in update_source_layout_widget: {str(e)}")
            import traceback
            print(traceback.format_exc())
