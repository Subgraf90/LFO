import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi
from Module_LFO.Modules_Plot.PlotStacks2Windowing import StackDraw_Windowing


class WindowingPlot(QWidget):
    def __init__(self, parent=None, settings=None, container=None, width=6, height=4, dpi=100):
        super().__init__(parent)
        self.settings = settings
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.container = container
        self.figure.tight_layout()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)


    def windowing_plot(self, windowing, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw()
            return
        
        self.plot_Stacks2Windowing(speaker_array_id)

        if not isinstance(windowing, dict):
            windowing = {}

        self.plot_windowing(windowing, speaker_array)

        window_distance = windowing.get('window_distance')
        if window_distance is None:
            window_distance = np.array([])

        window_restriction_db = windowing.get('window_restriction_db')
        if window_restriction_db is None:
            window_restriction_db = np.array([])

        x_values = []
        x_values.extend(np.atleast_1d(window_distance).tolist())
        if hasattr(speaker_array, 'source_position_x'):
            x_values.extend(np.atleast_1d(speaker_array.source_position_x).tolist())
        if hasattr(speaker_array, 'source_position_y'):
            x_values.extend(np.atleast_1d(speaker_array.source_position_y).tolist())

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

        y_values = []
        y_values.extend(np.atleast_1d(window_restriction_db).tolist())
        if hasattr(speaker_array, 'source_lp_zero'):
            y_values.extend(np.atleast_1d(speaker_array.source_lp_zero).tolist())
        window_restriction = getattr(speaker_array, 'window_restriction', None)
        if window_restriction is not None:
            y_values.extend(np.atleast_1d(window_restriction).tolist())

        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = max(y_max - y_min, 0)
            y_step = self._select_db_tick_step(y_range)
            y_padding = max(1.0, y_step * 0.5)
            if y_min == y_max:
                y_min -= y_padding
                y_max += y_padding
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.ax.yaxis.set_major_locator(MultipleLocator(y_step))


        self.ax.axis('equal')
        self.ax.grid(color='k', linestyle='-', linewidth=0.3)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.set_ylabel("Windowfunction [dB]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)

        # Aktualisieren Sie die Darstellung
        self.canvas.draw_idle()
        

    def plot_windowing(self, windowing, speaker_array):
        window_distance = windowing.get('window_distance', [0])
        window_restriction_db = windowing.get('window_restriction_db', [-9.8])

        self.ax.plot(window_distance, window_restriction_db)

        # "Setze Gain-Position der Stack Grafiken in der Fensteransicht"
        for iy, il in zip(speaker_array.source_position_x, speaker_array.source_lp_zero):
            self.ax.add_patch(plt.Circle((iy, il), 0.2, color='r'))

        # draw lines into window-function at eff. source positions
        for i in range(len(speaker_array.source_position_x)):
            y_data = speaker_array.source_position_x[i]
            x_data = speaker_array.window_restriction
            gain = speaker_array.source_lp_zero[i]
            self.ax.plot([y_data, y_data], [x_data, gain], linestyle='dotted', color='red')
            
        self.canvas.draw_idle()


    def plot_Stacks2Windowing(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data['cabinet_data']
            speaker_names = self.container.data['speaker_names']
            
            # Löschen Sie alle vorhandenen Patches (Stacks) aus dem Axes-Objekt
            for patch in self.ax.patches:
                patch.remove()
            
            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            for pattern_name in speaker_array.source_polar_pattern:
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
                    print(f"Pattern {pattern_name} übersprungen: {str(e)}")
                    continue
            
            if array_cabinets:
                stack_drawer = StackDraw_Windowing(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    speaker_array.window_restriction,
                    self.ax,
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_drawer.draw_stack(isrc)
            
            # Aktualisieren Sie die Darstellung
            self.canvas.draw()
            
        except Exception as e:
            print(f"Fehler in plot_Stacks2Windowing: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _select_metric_tick_step(self, value_range):
        # Dynamische Schrittweite für metrische Achsen (1, 2, 5, 10, 20 m)
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

    def _select_db_tick_step(self, value_range):
        # Dynamische Schrittweite für dB-Achsen (3, 6, 10 dB)
        if value_range > 35:
            return 10
        if value_range > 20:
            return 6
        return 3