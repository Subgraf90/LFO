from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from Module_LFO.Modules_Plot.PlotMPLCanvas import MplCanvas
from Module_LFO.Modules_Window.WindowSourceLayoutWidget import Draw_Source_Layout_Widget

class DrawWidgets:
    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.source_layout_widget = None
        self.matplotlib_canvas_source_layout_widget = None

    def update_source_layout_widget(self):
        if (self.source_layout_widget and 
            self.source_layout_widget.isVisible() and 
            self.matplotlib_canvas_source_layout_widget):
            
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            source_layout_widget = Draw_Source_Layout_Widget(
                self.matplotlib_canvas_source_layout_widget.ax, 
                self.settings,
                self.container
            )
            source_layout_widget.update_source_layout_widget(selected_speaker_array_id)

    def show_source_layout_widget(self):
        if self.source_layout_widget:
            self.source_layout_widget.close()
            self.source_layout_widget.deleteLater()
            
        self.source_layout_widget = QDockWidget("Source Layout", self.main_window)
        self.source_layout_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
                
        dock_widget_content = QWidget()
        dock_layout = QVBoxLayout()
        canvas_container = QWidget()
        canvas_container.setFixedHeight(260)
        container_layout = QVBoxLayout(canvas_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.matplotlib_canvas_source_layout_widget = MplCanvas(parent=canvas_container, width=6, height=2.3)
        self.matplotlib_canvas_source_layout_widget.setMinimumHeight(260)
        self.matplotlib_canvas_source_layout_widget.setMaximumHeight(260)
        container_layout.addWidget(self.matplotlib_canvas_source_layout_widget)

        dock_layout.addWidget(canvas_container)
        dock_widget_content.setLayout(dock_layout)
        self.source_layout_widget.setWidget(dock_widget_content)
        
        self.source_layout_widget.resize(1400, 500)
        self.source_layout_widget.setFloating(True)
        self.source_layout_widget.show()
    
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        source_layout_widget = Draw_Source_Layout_Widget(
            self.matplotlib_canvas_source_layout_widget.ax, 
            self.settings,
            self.container
        )
        source_layout_widget.update_source_layout_widget(selected_speaker_array_id)

    def close_all_widgets(self):
        self._safe_close(self.source_layout_widget)

    def _safe_close(self, widget):
        if widget:
            widget.close()
            widget.deleteLater()
