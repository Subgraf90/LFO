import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy

class MplCanvas(FigureCanvas):
    num = 0

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Initialisierung der Matplotlib-Figur und Achsen
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

        # Entferne automatische Achsenbeschriftungen
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')

        # Setzen des übergeordneten Widgets und Initialisierung von Daten
        self.setParent(parent)
        self.data_x = None
        self.data_y = None        
        
        # Anpassen der Größenpolitik des Canvas
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)