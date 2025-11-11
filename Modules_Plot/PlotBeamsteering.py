import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize
from Module_LFO.Modules_Plot.PlotStacks2Beamsteering import StackDraw_Beamsteering


class BeamsteeringPlot(QWidget):
    def __init__(self, parent=None, settings=None, container=None, width=6, height=4, dpi=100):
        super().__init__(parent)
        self.settings = settings
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.container = container
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)


    def beamsteering_plot(self, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw()
            return
        
        # Plot der Stacks und Beamsteering-Elemente
        self.plot_Stacks2Beamsteering(speaker_array_id)
        
        # Extrahieren der Lautsprecherpositionen
        source_position_y = speaker_array.source_position_y
        source_position_x = speaker_array.source_position_x
        virtual_source_position_x = speaker_array.virtual_source_position_x
        virtual_source_position_y = speaker_array.virtual_source_position_y

        # Plot der virtuellen Lautsprecherpositionen
        self.ax.plot(virtual_source_position_x, virtual_source_position_y, 'o-', label='Virtual Source Positions')

        # Zeichnen gestrichelter Linien von realen zu virtuellen Lautsprecherpositionen
        for sx, sy, vx, vy in zip(source_position_x, source_position_y, virtual_source_position_x, virtual_source_position_y):
            self.ax.plot([sx, vx], [sy, vy], linestyle='--', color='grey')

        x_values = []
        x_values.extend(np.atleast_1d(source_position_x).tolist())
        x_values.extend(np.atleast_1d(virtual_source_position_x).tolist())

        y_values = []
        y_values.extend(np.atleast_1d(source_position_y).tolist())
        y_values.extend(np.atleast_1d(virtual_source_position_y).tolist())

        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            x_range = max(x_max - x_min, 0)
            x_step = self._select_metric_tick_step(x_range)
            x_padding = max(0.5, x_step * 0.5)
            if x_min == x_max:
                x_min -= x_padding
                x_max += x_padding
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.xaxis.set_major_locator(MultipleLocator(x_step))

        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = max(y_max - y_min, 0)
            y_step = self._select_metric_tick_step(y_range)
            y_padding = max(0.5, y_step * 0.5)
            if y_min == y_max:
                y_min -= y_padding
                y_max += y_padding
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.yaxis.set_major_locator(MultipleLocator(y_step))

        # Finale Plot-Einstellungen
        self.ax.axis('equal')
        self.ax.grid(color='k', linestyle='-', linewidth=0.3)
        self.ax.set_ylabel("Virtual position [m]", fontsize=6)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)

        self.figure.subplots_adjust(left=0.15)
        self.figure.tight_layout()
        self.canvas.draw()


    def plot_Stacks2Beamsteering(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data['cabinet_data']
            speaker_names = self.container.data['speaker_names']

            if speaker_array is None:
                print("Warnung: speaker_array ist nicht initialisiert")
                return
            
            # Löschen Sie alle vorhandenen Patches
            for patch in self.ax.patches:
                patch.remove()
            
            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            for pattern_name in speaker_array.source_polar_pattern:
                # Überspringe leere Pattern-Namen
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
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Verarbeiten von Pattern {pattern_name}: {str(e)}")
                    continue
            
            if len(array_cabinets) > 0:
                stack_drawer = StackDraw_Beamsteering(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    self.ax,
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_drawer.draw_stack(isrc)
            else:
                print("Keine gültigen Cabinet-Daten gefunden")
            
            # Aktualisieren Sie die Darstellung
            self.canvas.draw()
            
        except Exception as e:
            print(f"Fehler in plot_Stacks2Beamsteering: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _select_metric_tick_step(self, value_range):
        thresholds = [
            (80, 20),
            (40, 10),
            (20, 5),
            (10, 2),
        ]
        for threshold, step in thresholds:
            if value_range > threshold:
                return step
        return 1