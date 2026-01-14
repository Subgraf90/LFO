from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from Module_LFO.Modules_Plot.PlotMPLCanvas import MplCanvas
from Module_LFO.Modules_Window.WindowSourceParameterWidget import Draw_Source_Layout_Widget

class DrawWidgets:
    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.source_layout_widget = None
        self.matplotlib_canvas_source_layout_widget = None
        self.source_layout_widget_instance = None  # Persistente Instanz für Plot+Tabelle

    def update_source_layout_widget(self):
        if (self.source_layout_widget and 
            self.source_layout_widget.isVisible() and 
            self.matplotlib_canvas_source_layout_widget):
            
            selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
            
            # Verwende persistente Instanz
            if self.source_layout_widget_instance is None:
                self.source_layout_widget_instance = Draw_Source_Layout_Widget(
                    self.matplotlib_canvas_source_layout_widget, 
                    self.settings,
                    self.container
                )
            
            # Aktualisiere nur die Daten
            self.source_layout_widget_instance.update_source_layout_widget(selected_speaker_array_id)

    def show_source_layout_widget(self):
        if self.source_layout_widget:
            self.source_layout_widget.close()
            self.source_layout_widget.deleteLater()
            # Setze Instanz zurück, da Widget neu erstellt wird
            self.source_layout_widget_instance = None
            
        self.source_layout_widget = QDockWidget("Source Parameters", self.main_window)
        self.source_layout_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
                
        dock_widget_content = QWidget()
        dock_layout = QVBoxLayout(dock_widget_content)
        dock_layout.setContentsMargins(5, 5, 5, 5)
        dock_layout.setSpacing(5)

        # 1. Matplotlib Canvas für Plot (fixe Höhe, Platz für Beschriftungen)
        self.matplotlib_canvas_source_layout_widget = MplCanvas(parent=dock_widget_content, width=10, height=3)
        self.matplotlib_canvas_source_layout_widget.setFixedHeight(250)  # Genug Platz für Beschriftungen oben
        dock_layout.addWidget(self.matplotlib_canvas_source_layout_widget, 0)  # Stretch-Faktor 0

        # Erstelle persistente Instanz
        self.source_layout_widget_instance = Draw_Source_Layout_Widget(
            self.matplotlib_canvas_source_layout_widget, 
            self.settings,
            self.container
        )
        
        # 2. Array-Info-Tabelle erstellen (im Layout der Widget-Instanz)
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        from PyQt5.QtGui import QColor
        
        self.array_info_table = QTableWidget()
        self.array_info_table.setRowCount(6)  # Eine Zeile mehr
        self.array_info_table.setColumnCount(2)
        self.array_info_table.setHorizontalHeaderLabels(['Parameter', 'Wert'])
        self.array_info_table.setFixedHeight(230)  # Mehr vertikaler Platz
        self.array_info_table.verticalHeader().setVisible(False)
        
        # WICHTIG: Fixe Spaltenbreiten, kein Stretch
        self.array_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.array_info_table.setColumnWidth(0, 200)
        self.array_info_table.setColumnWidth(1, 300)
        
        self.array_info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.array_info_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                font-size: 10pt;
                color: black;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 4px;
            }
        """)
        dock_layout.addWidget(self.array_info_table, 0)  # Stretch-Faktor 0 = nicht expandieren
        
        # Speichere Referenz in Widget-Instanz
        self.source_layout_widget_instance.array_info_table_widget = self.array_info_table
        
        # 3. Source-Parameter-Tabelle (mit Scrollbar)
        self.source_table = QTableWidget()
        self.source_table.setColumnCount(10)
        self.source_table.setHorizontalHeaderLabels(['#', 'Typ', 'Pos X [m]', 'Pos Y [m]', 'Pos Z [m]', 
                                                     'Azimuth [°]', 'Site [°]', 'Level [dB]', 'Delay [ms]', 'Pol.'])
        self.source_table.setMinimumHeight(200)  # Mindesthöhe, kann größer werden
        self.source_table.verticalHeader().setVisible(False)
        
        # Fixe Spaltenbreiten (kein automatisches Stretching)
        self.source_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.source_table.setColumnWidth(0, 40)   # #
        self.source_table.setColumnWidth(1, 120)  # Typ
        self.source_table.setColumnWidth(2, 80)   # Pos X
        self.source_table.setColumnWidth(3, 80)   # Pos Y
        self.source_table.setColumnWidth(4, 80)   # Pos Z
        self.source_table.setColumnWidth(5, 90)   # Azimuth
        self.source_table.setColumnWidth(6, 80)   # Site
        self.source_table.setColumnWidth(7, 85)   # Level
        self.source_table.setColumnWidth(8, 85)   # Delay
        self.source_table.setColumnWidth(9, 50)   # Pol
        
        # Fixe Zeilenhöhe
        self.source_table.verticalHeader().setDefaultSectionSize(25)
        
        self.source_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.source_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                font-size: 9pt;
                color: black;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 4px;
            }
        """)
        dock_layout.addWidget(self.source_table, 1)  # Stretch-Faktor 1
        
        # Speichere Referenz in Widget-Instanz
        self.source_layout_widget_instance.source_table_widget = self.source_table
        
        # 4. Export-Button hinzufügen
        self.source_layout_widget_instance._create_export_button(dock_layout)
        
        self.source_layout_widget.setWidget(dock_widget_content)
        
        # Größeres Widget für alle Komponenten
        self.source_layout_widget.resize(900, 750)
        self.source_layout_widget.setFloating(True)
        self.source_layout_widget.show()
    
        # Initiales Update
        selected_speaker_array_id = self.main_window.get_selected_speaker_array_id()
        self.source_layout_widget_instance.update_source_layout_widget(selected_speaker_array_id)

    def close_all_widgets(self):
        self._safe_close(self.source_layout_widget)

    def _safe_close(self, widget):
        if widget:
            widget.close()
            widget.deleteLater()
