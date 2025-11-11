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
            self.canvas.draw_idle()
            return
        
        # Plot der Stacks und Beamsteering-Elemente
        self.plot_Stacks2Beamsteering(speaker_array_id)
        
        # Extrahieren der Lautsprecherpositionen
        source_position_y = self._to_float_array(speaker_array.source_position_y)
        source_position_x = self._to_float_array(speaker_array.source_position_x)
        virtual_source_position_x = self._to_float_array(speaker_array.virtual_source_position_x)
        virtual_source_position_y = self._to_float_array(speaker_array.virtual_source_position_y)

        # Plot der virtuellen Lautsprecherpositionen
        self.ax.plot(virtual_source_position_x, virtual_source_position_y, 'o-', label='Virtual Source Positions')

        # Zeichnen gestrichelter Linien von realen zu virtuellen Lautsprecherpositionen
        for sx, sy, vx, vy in zip(source_position_x, source_position_y, virtual_source_position_x, virtual_source_position_y):
            self.ax.plot([sx, vx], [sy, vy], linestyle='--', color='grey')

        x_values = self._concatenate_arrays(source_position_x, virtual_source_position_x)
        y_values = self._concatenate_arrays(source_position_y, virtual_source_position_y)

        if x_values.size:
            x_min = float(np.min(x_values))
            x_max = float(np.max(x_values))
            x_range = max(x_max - x_min, 0.0)
            x_step = self._select_metric_tick_step(x_range)
            x_padding = max(0.5, x_step * 0.5)
            if x_min == x_max:
                x_min -= x_padding
                x_max += x_padding
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.xaxis.set_major_locator(MultipleLocator(x_step))

        if y_values.size:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            y_range = max(y_max - y_min, 0.0)
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
        self.canvas.draw_idle()


    def plot_Stacks2Beamsteering(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data.get('cabinet_data', [])
            speaker_names = self.container.data.get('speaker_names', [])

            if speaker_array is None:
                print("Warnung: speaker_array ist nicht initialisiert")
                return

            # Vorhandene Patches entfernen
            if self.ax.patches:
                self.ax.patches.clear()

            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            name_to_index = {name: idx for idx, name in enumerate(speaker_names)}
            for pattern_name in speaker_array.source_polar_pattern:
                if not pattern_name:
                    continue
                try:
                    speaker_index = name_to_index.get(pattern_name)
                    if speaker_index is None or speaker_index >= len(cabinet_data):
                        raise ValueError(f"{pattern_name} nicht gefunden")
                    cabinet = cabinet_data[speaker_index]
                    normalized_cabinets = self._normalize_cabinet_entries(cabinet)
                    if normalized_cabinets:
                        array_cabinets.append(normalized_cabinets)
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
            self.canvas.draw_idle()

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

    @staticmethod
    def _to_float_array(values):
        if values is None:
            return np.array([], dtype=float)
        array = np.atleast_1d(values)
        result = []
        for item in array:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return np.array(result, dtype=float) if result else np.array([], dtype=float)

    @staticmethod
    def _concatenate_arrays(*arrays):
        valid = [arr for arr in arrays if arr is not None and arr.size]
        if not valid:
            return np.array([], dtype=float)
        return np.concatenate(valid)

    @classmethod
    def _normalize_cabinet_entries(cls, cabinet):
        if cabinet is None:
            return []
        if isinstance(cabinet, dict):
            return [cabinet]
        if isinstance(cabinet, np.ndarray):
            cabinet = cabinet.tolist()
        if isinstance(cabinet, (list, tuple)):
            normalized = []
            for item in cabinet:
                normalized.extend(cls._normalize_cabinet_entries(item))
            return normalized
        return []