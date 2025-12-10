import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QTabWidget, QGroupBox,
                           QMenu, QTreeWidget, QTreeWidgetItem, 
                           QMessageBox, QCheckBox, QLineEdit, QLabel, QInputDialog, QComboBox, QFrame, QDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QEvent, QPoint, QSize
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import warnings
from scipy.io import wavfile
from Module_LFO.Modules_Plot.PlotDataImporter import Draw_Plots_DataImporter
from Module_LFO.Modules_Calculate.PolarDataCalculator import Polar_Data_Calculator
from Module_LFO.Modules_Data.PolarDataExport import Polar_Data_Export
from Module_LFO.Modules_Init.Progress import ProgressManager
import os
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# WAV-File Warnungen unterdrücken
warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)

# ---------------------
# - prüfen ob export korrekt ist. (averaged data, anzahl datenpunkte)

# 2. Schritt
# - Tab widget bei Polarplot
# - Balloon Plot
# - daten zu Balloonstruktur erweitern: Ordner für jeden Winkel??? 
# - Interpolation 3D?

# ----------------- 

class TransferFunctionViewer(QMainWindow):
    def __init__(self, parent=None, manage_speaker_window=None, settings=None, container=None):
        super().__init__(parent)
        self.manage_speaker_window = manage_speaker_window
        self.settings = settings
        if container is None and manage_speaker_window and hasattr(manage_speaker_window, "container"):
            container = manage_speaker_window.container
        self.container = container
        self.setWindowTitle("Polar data generator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Zentrale Datenstruktur
        self.data = {
            'raw_measurements': {},      
            'calculated_data': {},       
            'polar_data': {},           
            'interpolated_data': {},     
            'metadata': {   
                # Grundlegende Informationen
                'sample_rate': None,                      # Sample-Rate                
                'vertical_count': 0,             # Anzahl vertikaler Messungen
                'horizontal_count': 0,           # Anzahl horizontaler Messungen
                            'freq_range_min': float(10),     # Minimale Frequenz in Hz
            'freq_range_max': float(300),    # Maximale Frequenz in Hz
            'filter_enabled': False,         # Filter aktiviert
            'filter_order': 2,               # Filter-Ordnung (12 dB/Oktave)
                'data_source': 'measurement',    # Quelle der Daten
                'angles_horizontal': 0,          # Anzahl gemessener horizontaler Winkel
                'angles_vertical': 0,            # Anzahl gemessener vertikaler Winkel
                'interpolation_horizontal': 360, # Anzahl horizontal interpolierter Winkel
                'interpolation_vertical': 0,     # Anzahl vertikal interpolierter Winkel
                'spl_normalized': False,              # SPL-Normalisierung aktiv
                'spl_value': 0.0,                # SPL-Normalisierungswert in dB
                'reference_freq': 50.0,          # Referenzfrequenz in Hz
                'reference_distance': 1.0,       # Referenzdistanz in m
                'target_spl': 85.0,              # Ziel-SPL in dB
                'time_normalized': False,        # Zeitnormalisierung aktiv7
                'time_offset': 0.0,              # Zeitversatz in ms
                'ir_window': 200.0,              # IR Window in ms
                'ir_window_enabled': False,      # IR Window deaktiviert
            }
        }


        self.polar_calculator = Polar_Data_Calculator(self, self.settings, self.data)
        self.polar_export = Polar_Data_Export(self.settings, self.data, speakers=None, container=self.container)
        self.progress_manager = ProgressManager(self)

        # Hauptwidget und Layout erstellen§
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Tab Widget erstellen
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Plot Tab erstellen
        self.plot_tab = QWidget()
        plot_tab_layout = QHBoxLayout(self.plot_tab)  # Hier erstellen wir plot_tab_layout

        # Linke Seite: Steuerungselemente
        plot_control_panel = QVBoxLayout()
        
        # 1. Import GroupBox
        import_group = QGroupBox("")  # Leerer Titel
        nav_layout = QHBoxLayout()
        
        # Import Button mit Menü
        self.import_button = QPushButton("Import")
        import_menu = QMenu(self.import_button)
        
        # Import Optionen
        import_ir_action = import_menu.addAction("Impulse Response (wav)")
        import_txt_action = import_menu.addAction("Mag, Phase (txt)")
        import_txt_action.setEnabled(True)  # Aktiviere TXT Import
        
        # Verbinde die Actions mit den korrekten Strings
        import_ir_action.triggered.connect(lambda: self.load_files("Impulse Response"))
        import_txt_action.triggered.connect(lambda: self.load_files("txt"))
        
        self.import_button.setMenu(import_menu)
        
        # Export Button mit Menü
        self.export_button = QPushButton("Export")
        export_menu = QMenu(self.export_button)
        export_to_lfo_action = export_menu.addAction("Export to speaker list")
        export_to_lfo_action.triggered.connect(self.export_to_lfo)
        self.export_button.setMenu(export_menu)
        
        # Buttons zum Layout hinzufügen
        nav_layout.addWidget(self.import_button)
        nav_layout.addWidget(self.export_button)
        
        import_group.setLayout(nav_layout)
        plot_control_panel.addWidget(import_group)
        
        # TreeWidget
        self.setup_tree_widget()
        self.tree_widget.setFixedWidth(280)
        plot_control_panel.addWidget(self.tree_widget)
        
        # Tab Widget für die vier Bereiche
        self.settings_tab_widget = QTabWidget()
        self.settings_tab_widget.setFixedWidth(280)
        self.settings_tab_widget.setMaximumHeight(220)  # Reduziere die Höhe
        
        # Tab 1: Delay Offset
        delay_tab = QWidget()
        delay_layout = QVBoxLayout(delay_tab)
        
        # Time Input Layout
        time_input_layout = QHBoxLayout()
        time_input_layout.addWidget(QLabel("Delay offset (ms)"))
        self.time_input = QLineEdit()
        self.time_input.setFixedWidth(80)
        self.time_input.setText("0.00")
        time_input_layout.addWidget(self.time_input)
        time_input_layout.setAlignment(self.time_input, Qt.AlignRight)
        delay_layout.addLayout(time_input_layout)
        delay_layout.addStretch()  # Füge Stretch hinzu
        
        # Tab 2: IR Window
        window_tab = QWidget()
        window_layout = QVBoxLayout(window_tab)
        
        # IR Window Input Layout
        ir_window_input_layout = QHBoxLayout()
        ir_window_input_layout.addWidget(QLabel("IR window function (ms)"))
        self.ir_window_input = QLineEdit()
        self.ir_window_input.setFixedWidth(80)
        # Setze Standardwert in Metadaten und UI
        if 'ir_window' not in self.data['metadata']:
            self.data['metadata']['ir_window'] = 200.0
        self.ir_window_input.setText(f"{self.data['metadata']['ir_window']:.1f}")
        ir_window_input_layout.addWidget(self.ir_window_input)
        ir_window_input_layout.setAlignment(self.ir_window_input, Qt.AlignRight)
        window_layout.addLayout(ir_window_input_layout)
        
        # IR Window Checkbox Layout
        ir_window_checkbox_layout = QHBoxLayout()
        ir_window_checkbox_layout.addWidget(QLabel("IR window enabled"))
        self.ir_window_checkbox = QCheckBox()
        # Setze Standardwert in Metadaten
        if 'ir_window_enabled' not in self.data['metadata']:
            self.data['metadata']['ir_window_enabled'] = False
        self.ir_window_checkbox.setChecked(self.data['metadata']['ir_window_enabled'])
        ir_window_checkbox_layout.addWidget(self.ir_window_checkbox)
        ir_window_checkbox_layout.setAlignment(self.ir_window_checkbox, Qt.AlignRight)
        window_layout.addLayout(ir_window_checkbox_layout)
        
        # IR Window Eingabefeld ist immer aktiviert
        self.ir_window_input.setEnabled(True)
        
        # Setze die Abstände zwischen den Layouts
        window_layout.setSpacing(5)
        window_layout.addStretch()  # Füge Stretch hinzu
        
        # Tab 3: Bandpass Filter
        filter_tab = QWidget()
        freq_layout = QVBoxLayout(filter_tab)

        # Minimum Frequency Layout
        min_freq_layout = QHBoxLayout()
        min_freq_layout.addWidget(QLabel("Minimum (Hz)"))
        self.min_freq_input = QLineEdit()
        self.min_freq_input.setFixedWidth(80)
        min_freq = self.data['metadata'].get('filter_freq_min', 10.0)
        self.min_freq_input.setText(str(min_freq))
        min_freq_layout.addWidget(self.min_freq_input)
        min_freq_layout.setAlignment(self.min_freq_input, Qt.AlignRight)
        freq_layout.addLayout(min_freq_layout)

        # Maximum Frequency Layout
        max_freq_layout = QHBoxLayout()
        max_freq_layout.addWidget(QLabel("Maximum (Hz)"))
        self.max_freq_input = QLineEdit()
        self.max_freq_input.setFixedWidth(80)
        max_freq = self.data['metadata'].get('filter_freq_max', 300.0)
        self.max_freq_input.setText(str(max_freq))
        max_freq_layout.addWidget(self.max_freq_input)
        max_freq_layout.setAlignment(self.max_freq_input, Qt.AlignRight)
        freq_layout.addLayout(max_freq_layout)
        
        # Filter Checkbox Layout
        filter_checkbox_layout = QHBoxLayout()
        filter_checkbox_layout.addWidget(QLabel("Set bandpass filter"))
        self.filter_checkbox = QCheckBox()
        self.filter_checkbox.setChecked(False)  # Standard: deaktiviert
        filter_checkbox_layout.addWidget(self.filter_checkbox)
        filter_checkbox_layout.setAlignment(self.filter_checkbox, Qt.AlignRight)
        freq_layout.addLayout(filter_checkbox_layout)
        freq_layout.addStretch()  # Füge Stretch hinzu
        
        # Tab 4: SPL Normalization
        spl_tab = QWidget()
        spl_norm_layout = QVBoxLayout(spl_tab)
        
        # Frequenz Layout
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency band (Hz)"))
        self.spl_freq_combo = QComboBox()
        self.spl_freq_combo.addItems(['25', '31.5', '40', '50', '63', '80'])
        self.spl_freq_combo.setFixedWidth(80)
        self.spl_freq_combo.setCurrentText("50")  # Standard: 50 Hz
        self.spl_freq_combo.setEnabled(True)  # Immer aktiviert
        freq_layout.addWidget(self.spl_freq_combo)
        freq_layout.setAlignment(self.spl_freq_combo, Qt.AlignRight)
        
        # Distanz Layout
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance (m)"))
        self.spl_distance_input = QLineEdit()
        self.spl_distance_input.setFixedWidth(80)
        self.spl_distance_input.setText("1.0")  # Standard: 1m Abstand
        self.spl_distance_input.setEnabled(True)  # Immer aktiviert
        distance_layout.addWidget(self.spl_distance_input)
        distance_layout.setAlignment(self.spl_distance_input, Qt.AlignRight)
        
        # SPL Layout
        spl_layout = QHBoxLayout()
        spl_layout.addWidget(QLabel("SPL (dB)"))
        self.spl_level_input = QLineEdit()
        self.spl_level_input.setFixedWidth(80)
        self.spl_level_input.setText("0")  # Standard: 0 dB SPL
        self.spl_level_input.setEnabled(True)  # Immer aktiviert
        spl_layout.addWidget(self.spl_level_input)
        spl_layout.setAlignment(self.spl_level_input, Qt.AlignRight)
        
        # Value Layout
        value_layout = QHBoxLayout()
        value_layout.addWidget(QLabel("Value (dB)"))
        self.spl_value_input = QLineEdit()
        self.spl_value_input.setFixedWidth(80)
        self.spl_value_input.setText("0")
        self.spl_value_input.setEnabled(True)  # Immer aktiviert
        value_layout.addWidget(self.spl_value_input)
        value_layout.setAlignment(self.spl_value_input, Qt.AlignRight)
        
        # Checkbox Layout (rechts-bündig) - jetzt unterhalb Value
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(QLabel("Set SPL normalization"))
        self.spl_norm_checkbox = QCheckBox()
        checkbox_layout.addWidget(self.spl_norm_checkbox)
        checkbox_layout.setAlignment(self.spl_norm_checkbox, Qt.AlignRight)
        

        # Füge alle Layouts zum SPL Normalize Layout hinzu
        spl_norm_layout.addLayout(freq_layout)
        spl_norm_layout.addLayout(distance_layout)
        spl_norm_layout.addLayout(spl_layout)
        spl_norm_layout.addLayout(value_layout)
        spl_norm_layout.addLayout(checkbox_layout)
        spl_norm_layout.addStretch()  # Füge Stretch hinzu
        
        # Füge die Tabs zum Tab Widget hinzu
        self.settings_tab_widget.addTab(delay_tab, "Delay")
        self.settings_tab_widget.addTab(window_tab, "Window")
        self.settings_tab_widget.addTab(filter_tab, "Filter")
        self.settings_tab_widget.addTab(spl_tab, "SPL")
        
        # Füge das Tab Widget zum Control Panel hinzu
        plot_control_panel.addWidget(self.settings_tab_widget)
        


        self.spl_norm_checkbox.stateChanged.connect(self.on_checkbox_normalize_changed)
        self.spl_freq_combo.currentTextChanged.connect(self.on_freq_normalize_changed)

        self.spl_distance_input.editingFinished.connect(self.on_distance_normalize_changed)
        self.spl_level_input.editingFinished.connect(self.on_spl_normalize_changed)
        self.spl_value_input.editingFinished.connect(self.on_value_normalize_changed)

        self.time_input.editingFinished.connect(self.on_validate_time_input)
        self.time_input.editingFinished.connect(self.on_time_value_changed)


        # IR Window Input Signal-Verbindungen
        self.ir_window_input.editingFinished.connect(self.on_validate_ir_window_input)
        self.ir_window_input.editingFinished.connect(self.on_ir_window_value_changed)
        
        # IR Window Checkbox Signal-Verbindung
        self.ir_window_checkbox.stateChanged.connect(self.on_ir_window_checkbox_changed)

        # Processing GroupBox
        processing_group = QGroupBox("")
        processing_layout = QVBoxLayout()
        
        # Interpolationsmethode-Auswahl entfernt - verwende feste Y-Achsen-Interpolation
        
        # Calculate Polar Button
        calculate_polar_button = QPushButton("Calculate data")
        calculate_polar_button.setFixedWidth(120)
        calculate_polar_button.clicked.connect(self.on_calculate_polar_clicked)

        processing_layout.addWidget(calculate_polar_button)
        
        processing_group.setLayout(processing_layout)
        plot_control_panel.addWidget(processing_group)
        
        # Verbinde die Validierungsfunktionen
        self.min_freq_input.editingFinished.connect(self.validate_min_freq)
        self.max_freq_input.editingFinished.connect(self.validate_max_freq)
        self.filter_checkbox.stateChanged.connect(self.on_filter_checkbox_changed)

        # Rechte Seite: Plots
        plot_right_panel = QVBoxLayout()
        
        # Matplotlib Figure und Toolbar erstellen
        self.plot_handler = Draw_Plots_DataImporter()
        self.canvas = self.plot_handler.get_canvas()
        self.toolbar = self.plot_handler.get_toolbar()
        self.toolbar.setIconSize(QSize(16, 16))


        
        plot_right_panel.addWidget(self.toolbar)
        plot_right_panel.addWidget(self.canvas)
        
        # Panels zum Plot-Tab-Layout hinzufügen
        plot_tab_layout.addLayout(plot_control_panel, stretch=1)  # Steuerung links
        plot_tab_layout.addLayout(plot_right_panel, stretch=3)    # Plots rechts
        
        # Tab für Polarplot
        self.setup_polar_tab()
        
        # Tab für Cabinet Dimension
        self.setup_cabinet_tab()
        
        # Füge alle Tabs zum TabWidget hinzu
        self.tab_widget.addTab(self.plot_tab, "Measurement data")
        self.tab_widget.addTab(self.polar_tab, "Interpolated polardata")
        self.tab_widget.addTab(self.cabinet_tab, "Cabinet data")
        
        # TabWidget zum Hauptlayout hinzufügen
        main_layout.addWidget(self.tab_widget)        
        # Feintuning der Plot-Darstellung
        # self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        

        
        # Verbinde die Signale (bereits oben verbunden, daher hier entfernt)

# ----- Tabs ------
    def eventFilter(self, obj, event):
        """Event-Filter für Input-Felder"""
        if obj in [self.spl_distance_input, self.spl_level_input]:
            if event.type() == QEvent.KeyPress:
                if event.key() in [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab]:
                    obj.clearFocus()  # Trigger editingFinished
                    return True
        return super().eventFilter(obj, event)

    def setup_cabinet_tab(self):
        """Erstellt das Cabinet Dimension Tab"""
        self.cabinet_tab = QWidget()
        cabinet_layout = QHBoxLayout(self.cabinet_tab)
        
        # Linke Seite: Steuerung und TreeWidget
        cabinet_left_panel = QVBoxLayout()
        
        # Lautsprecher-Steuerung
        speaker_control_layout = QHBoxLayout()
        
        # Button zum Hinzufügen eines Lautsprechers
        self.add_speaker_button = QPushButton("Add Speaker")
        self.add_speaker_button.clicked.connect(self.add_speaker)
        speaker_control_layout.addWidget(self.add_speaker_button)
        speaker_control_layout.addStretch()
        
        cabinet_left_panel.addLayout(speaker_control_layout)
        
        # Füge Abstand hinzu (20 Pixel)
        cabinet_left_panel.addSpacing(20)
        
        # TreeWidget für Lautsprecher
        self.speaker_tree = QTreeWidget()
        self.speaker_tree.setHeaderLabels(["Speaker", "Value"])
        self.speaker_tree.setColumnWidth(0, 150)
        self.speaker_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.speaker_tree.customContextMenuRequested.connect(self.show_speaker_context_menu)
        cabinet_left_panel.addWidget(self.speaker_tree)
        
        # Rechte Seite: Plot mit Toolbar
        cabinet_right_panel = QVBoxLayout()
        
        # Toolbar und Canvas vom plot_handler verwenden
        self.cabinet_toolbar = NavigationToolbar(self.plot_handler.cabinet_canvas, self)
        self.cabinet_toolbar.setIconSize(QSize(16, 16))
        cabinet_right_panel.addWidget(self.cabinet_toolbar)
        cabinet_right_panel.addWidget(self.plot_handler.cabinet_canvas)
        
        # Panels zum Cabinet-Layout hinzufügen
        cabinet_layout.addLayout(cabinet_left_panel, stretch=1)
        cabinet_layout.addLayout(cabinet_right_panel, stretch=2)
        
        # Tab zum TabWidget hinzufügen
        self.tab_widget.addTab(self.cabinet_tab, "Cabinet Layout")
        
        # Initialisiere die Lautsprecherliste
        self.speakers = []
        
        # Initial einen Lautsprecher erstellen
        self.add_speaker()

    def setup_tree_widget(self):
        """Konfiguriert das TreeWidget mit vordefinierten Ordnern"""
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Measurements")
        self.tree_widget.setSelectionMode(QTreeWidget.ExtendedSelection)
        
        # Aktiviere horizontale Scrollbar für lange Namen
        self.tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree_widget.setHorizontalScrollMode(QTreeWidget.ScrollPerPixel)
        
        # Aktiviere Drag & Drop mit Einschränkungen
        self.tree_widget.setDragEnabled(True)
        self.tree_widget.setAcceptDrops(True)
        self.tree_widget.setDropIndicatorShown(True)
        self.tree_widget.setDragDropMode(QTreeWidget.InternalMove)  # Nur innerhalb des TreeWidgets
        
        # Überschreibe die dropEvent-Methode, um unerwünschte Hierarchien zu verhindern
        original_dropEvent = self.tree_widget.dropEvent
        
        def custom_dropEvent(event):
            # Hole das Item, auf das gedroppt werden soll
            drop_pos = event.pos()
            drop_index = self.tree_widget.indexAt(drop_pos)
            drop_item = self.tree_widget.itemAt(drop_pos)
            
            # Hole die Items, die gedroppt werden
            dragged_items = self.tree_widget.selectedItems()
            
            # Prüfe die Drop-Position
            indicator_pos = self.tree_widget.dropIndicatorPosition()
            
            # Verhindere das Droppen AUF eine Datei (OnItem), erlaube aber das Droppen ZWISCHEN Dateien
            if indicator_pos == QTreeWidget.OnItem and drop_item and drop_item.parent() is not None:
                event.ignore()
                return
            
            # Verhindere, dass Ordner anderen Ordnern untergeordnet werden
            # Ein Ordner ist erkennbar daran, dass er kein parent() hat (Top-Level-Item)
            if indicator_pos == QTreeWidget.OnItem and drop_item:
                # Prüfe, ob das Ziel ein Ordner ist (kein parent)
                is_drop_target_folder = (drop_item.parent() is None)
                
                # Prüfe, ob mindestens ein gedragtes Item ein Ordner ist
                has_folder_in_dragged = any(item.parent() is None for item in dragged_items)
                
                # Verhindere, dass Ordner auf Ordner gedroppt werden
                if is_drop_target_folder and has_folder_in_dragged:
                    event.ignore()
                    return
            
            # Rufe die Original-Methode auf
            original_dropEvent(event)
            
            # Aktualisiere die Plots nach dem Drop
            self.on_item_moved()
        
        self.tree_widget.dropEvent = custom_dropEvent
        
        # Erstelle Standard-Ordner (2 Ordner für vollständiges 360° Polardiagramm)
        # Gleiche Logik wie in add_multiple_vertical_folders()
        num_folders = 2
        step = 360.0 / num_folders
        meridian_angles = [0 + i * step for i in range(num_folders)]
        
        for i in range(num_folders):
            angle = meridian_angles[i]
            folder_name = f"{angle:.0f}° Meridian"
            QTreeWidgetItem(self.tree_widget, [folder_name])
        
        # Speichere Referenz zum ersten Ordner (für Kompatibilität)
        self.horizontal_folder = self.tree_widget.topLevelItem(0)
        
        # Verbinde Signale
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.show_context_menu)
        
        # Verbinde Drag & Drop Signal
        self.tree_widget.model().rowsInserted.connect(self.on_item_moved)
        
        # Stelle sicher, dass der Hauptordner ausgewählt ist
        self.ensure_selection()
    
    def ensure_selection(self):
        """
        Stellt sicher, dass immer mindestens ein Item im TreeWidget ausgewählt ist.
        Wenn keine Auswahl vorhanden ist, wird das erste Item ausgewählt.
        """
        if not hasattr(self, 'tree_widget') or self.tree_widget is None:
            return
            
        # Prüfe, ob ein Item ausgewählt ist
        selected_items = self.tree_widget.selectedItems()
        
        # Wenn keine Auswahl vorhanden ist und Items existieren, wähle das erste aus
        if not selected_items and self.tree_widget.topLevelItemCount() > 0:
            first_item = self.tree_widget.topLevelItem(0)
            self.tree_widget.setCurrentItem(first_item)

    def setup_polar_tab(self):
        """Richtet den Polar Tab ein mit Tabs für 2D und 3D Plots"""
        try:
            self.polar_tab = QWidget()
            polar_layout = QVBoxLayout(self.polar_tab)
            
            # Erstelle ein Tab-Widget für die verschiedenen Polar-Darstellungen
            self.polar_tab_widget = QTabWidget()
            polar_layout.addWidget(self.polar_tab_widget)
            
            # Tab 1: 2D Polar Plot
            self.polar_2d_tab = QWidget()
            polar_2d_layout = QVBoxLayout(self.polar_2d_tab)
            
            # Steuerungselemente für 2D Polar Plot
            polar_2d_controls = QHBoxLayout()
            
            # Range (dB) Auswahl für Polar Plot
            polar_2d_controls.addWidget(QLabel("Range (dB):"))
            self.polar_range_selector = QComboBox()
            self.polar_range_selector.addItems(['-42 dB', '-36 dB', '-30 dB', '-24 dB', '-18 dB', '-12 dB', '-6 dB'])
            self.polar_range_selector.setCurrentIndex(0)  # Default: -42 dB
            self.polar_range_selector.setMinimumWidth(100)
            polar_2d_controls.addWidget(self.polar_range_selector)
            polar_2d_controls.addStretch()
            
            # Verbinde Range-Auswahl mit Update-Funktion
            self.polar_range_selector.currentIndexChanged.connect(self.update_dynamic_range)
            
            # Toolbar und Canvas für 2D Plot
            polar_2d_layout.addLayout(polar_2d_controls)
            polar_2d_layout.addWidget(self.plot_handler.get_polar_toolbar())
            polar_2d_layout.addWidget(self.plot_handler.get_polar_canvas())
            
            # Tab 2: 3D Balloon Plot
            self.balloon_tab = QWidget()
            balloon_layout = QVBoxLayout(self.balloon_tab)
            
            # Steuerungselemente für Balloon Plot
            balloon_controls = QHBoxLayout()
            
            # Frequenzauswahl für Balloon Plot
            balloon_controls.addWidget(QLabel("Frequenz:"))
            self.balloon_freq_selector = QComboBox()
            self.balloon_freq_selector.setMinimumWidth(100)
            balloon_controls.addWidget(self.balloon_freq_selector)
            
            # Range (dB) Auswahl für Balloon Plot
            balloon_controls.addWidget(QLabel("Range (dB):"))
            self.balloon_range_selector = QComboBox()
            self.balloon_range_selector.addItems(['-42 dB', '-36 dB', '-30 dB', '-24 dB', '-18 dB', '-12 dB', '-6 dB'])
            self.balloon_range_selector.setCurrentIndex(0)  # Default: -42 dB
            self.balloon_range_selector.setMinimumWidth(100)
            balloon_controls.addWidget(self.balloon_range_selector)
            balloon_controls.addStretch()
            
            # Verbinde die Frequenzauswahl mit der Maximize-Methode
            self.balloon_freq_selector.currentIndexChanged.connect(self.update_balloon_with_selected_freq)
            
            # Verbinde Range-Auswahl mit Update-Funktion
            self.balloon_range_selector.currentIndexChanged.connect(self.update_dynamic_range)
            
            # Toolbar und Canvas für Balloon Plot
            balloon_layout.addLayout(balloon_controls)
            balloon_layout.addWidget(self.plot_handler.get_balloon_toolbar())
            balloon_layout.addWidget(self.plot_handler.get_balloon_canvas())
            
            # Tabs zum Tab-Widget hinzufügen
            self.polar_tab_widget.addTab(self.polar_2d_tab, "2D Polar Plot")
            self.polar_tab_widget.addTab(self.balloon_tab, "3D Balloon Plot")
            
            # Tab zum Haupt-TabWidget hinzufügen
            self.tab_widget.addTab(self.polar_tab, "Polar Plots")
                        
        except Exception as e:
            print(f"Fehler beim Setup des Polar Tabs: {e}")
            import traceback
            traceback.print_exc()

    def update_balloon_with_selected_freq(self, index):
        """Aktualisiert den Balloon Plot mit der ausgewählten Frequenz"""
        if index >= 0 and hasattr(self, 'data') and 'balloon_data' in self.data:
            # Hole die ausgewählte Frequenz aus der ComboBox
            freq_text = self.balloon_freq_selector.currentText()
            if freq_text:
                try:
                    selected_freq = float(freq_text.split(' ')[0])  # Extrahiere den Zahlenwert
                    
                    # Aktualisiere den Plot mit der ausgewählten Frequenz
                    self.plot_handler.update_balloon_plot(self.data, selected_freq, show_freq_in_title=False)
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Parsen der Frequenz: {e}")

    def update_dynamic_range(self):
        """Aktualisiert den Dynamikbereich für beide Plots und synchronisiert die Selektoren"""
        # Bestimme welcher Selektor die Änderung ausgelöst hat
        sender = self.sender()
        
        if sender == self.polar_range_selector:
            selected_text = self.polar_range_selector.currentText()
            # Synchronisiere Balloon-Selektor
            self.balloon_range_selector.blockSignals(True)
            self.balloon_range_selector.setCurrentText(selected_text)
            self.balloon_range_selector.blockSignals(False)
        elif sender == self.balloon_range_selector:
            selected_text = self.balloon_range_selector.currentText()
            # Synchronisiere Polar-Selektor
            self.polar_range_selector.blockSignals(True)
            self.polar_range_selector.setCurrentText(selected_text)
            self.polar_range_selector.blockSignals(False)
        else:
            # Fallback: Verwende Balloon-Selektor
            selected_text = self.balloon_range_selector.currentText()
        
        if not selected_text:
            return
        
        # Extrahiere dB-Wert (z.B. "-42 dB" -> 42)
        try:
            dynamic_range_db = abs(int(selected_text.split()[0]))
            
            # Aktualisiere den plot_handler
            self.plot_handler.dynamic_range_db = dynamic_range_db
            
            print(f"Dynamikbereich geändert auf: {dynamic_range_db} dB")
            
            # Aktualisiere beide Plots wenn Daten vorhanden sind
            if hasattr(self, 'data'):
                if 'balloon_data' in self.data:
                    # Aktualisiere Balloon Plot
                    freq_text = self.balloon_freq_selector.currentText()
                    if freq_text:
                        try:
                            selected_freq = float(freq_text.split(' ')[0])
                            self.plot_handler.update_balloon_plot(self.data, selected_freq, show_freq_in_title=False)
                        except (ValueError, IndexError):
                            pass
                    
                    # Aktualisiere Polar Plot
                    self.plot_handler.update_polar_plot(self.data)
        
        except (ValueError, IndexError) as e:
            print(f"Fehler beim Parsen des Dynamikbereichs: {e}")


# ----- Folder Management -----

    def show_context_menu(self, configuration):
        """Zeigt Kontextmenü für TreeWidget Items"""
        menu = QMenu()
        selected_items = self.tree_widget.selectedItems()
    
        
        # Wenn keine Items ausgewählt sind, zeige Menü zum Hinzufügen von Ordnern
        if not selected_items:
            add_multiple_folders_action = menu.addAction("Add vertical data folder")
            action = menu.exec_(self.tree_widget.viewport().mapToGlobal(configuration))
            if action == add_multiple_folders_action:
                self.add_multiple_vertical_folders()
            return
        
        # Prüfe, ob alle ausgewählten Items vom gleichen Typ sind (Ordner oder Dateien)
        all_folders = all(item.parent() is None for item in selected_items)
        all_files = all(item.parent() is not None for item in selected_items)
        
        
        if all_folders:  # Alle ausgewählten Items sind Ordner
            delete_action = menu.addAction("Delete folder")
            duplicate_folder_action = menu.addAction("Duplicate folder")  # Neue Aktion
            menu.addSeparator()
            add_multiple_folders_action = menu.addAction("Add vertical data folder")
        elif all_files:  # Alle ausgewählten Items sind Dateien
            delete_action = menu.addAction("Delete file")
            duplicate_action = menu.addAction("Duplicate file")
            duplicate_reverse_action = menu.addAction("Duplicate reversed")  # Nur für Dateien
            menu.addSeparator()
            replace_phase_action = menu.addAction("Replace phase trace")
        else:  # Gemischte Auswahl
            delete_action = menu.addAction("Delete selected")
            menu.addSeparator()
            add_multiple_folders_action = menu.addAction("Add vertical data folder")
        
        action = menu.exec_(self.tree_widget.viewport().mapToGlobal(configuration))
        
        
        if not action:
            return
        
        if all_folders:  # Alle ausgewählten Items sind Ordner
            if action == delete_action:
                # Lösche alle ausgewählten Ordner
                for item in selected_items:
                    self.delete_folder(item)
            elif action == add_multiple_folders_action:
                self.add_multiple_vertical_folders()
            elif action == duplicate_folder_action:
                # Dupliziere jeden ausgewählten Ordner
                for folder_item in selected_items:
                    self.duplicate_folder(folder_item)
        elif all_files:  # Alle ausgewählten Items sind Dateien
            if action == delete_action:
                # Lösche alle ausgewählten Dateien
                for item in selected_items:
                    filename = item.text(0)
                    self.delete_measurement(filename, item)
                
                # Aktualisiere die Plots
                self.update_plot_tree_measurement()
            elif action == duplicate_action:
                # Dupliziere jede ausgewählte Datei
                for item in selected_items:
                    filename = item.text(0)
                    new_filename = f"copy_{filename}"
                    
                    # Kopiere die Daten
                    if filename in self.data['raw_measurements']:
                        self.data['raw_measurements'][new_filename] = self.data['raw_measurements'][filename].copy()
                    if filename in self.data['calculated_data']:
                        self.data['calculated_data'][new_filename] = self.data['calculated_data'][filename].copy()
                    
                    # Füge die Kopie zum selben Ordner hinzu
                    parent_folder = item.parent()
                    new_item = QTreeWidgetItem(parent_folder, [new_filename])
                
                # Aktualisiere die Plots
                self.update_plot_tree_measurement()
            elif action == duplicate_reverse_action:
                # Für ausgewählte Dateien: Kopiere nur die ausgewählten Dateien in umgekehrter Reihenfolge
                # Sammle alle ausgewählten Dateien
                files_to_duplicate = []
                for item in selected_items:
                    filename = item.text(0)
                    
                    # Nur Dateien berücksichtigen, die in den Daten existieren
                    if filename in self.data['raw_measurements']:
                        files_to_duplicate.append((filename, item.parent()))
                    elif filename in self.data['calculated_data']:
                        files_to_duplicate.append((filename, item.parent()))
                                
                # Füge Kopien in umgekehrter Reihenfolge hinzu
                reversed_files = list(reversed(files_to_duplicate))
                
                for i, (orig_filename, parent_folder) in enumerate(reversed_files):
                    # Ändere den Präfix für jede Datei, um sie zu unterscheiden
                    new_filename = f"rev{i+1}_{orig_filename}"                    
                    # Kopiere die Daten
                    if orig_filename in self.data['raw_measurements']:
                        self.data['raw_measurements'][new_filename] = self.data['raw_measurements'][orig_filename].copy()
                    if orig_filename in self.data['calculated_data']:
                        self.data['calculated_data'][new_filename] = self.data['calculated_data'][orig_filename].copy()
                    
                    # Füge zum selben Ordner hinzu
                    new_item = QTreeWidgetItem(parent_folder, [new_filename])
                
                # Aktualisiere die Plots

                self.update_plot_tree_measurement()

            elif action == replace_phase_action:
                self.replace_phase_for_selected_files(selected_items)
 
        else:  # Gemischte Auswahl
            if action == delete_action:
                # Lösche alle ausgewählten Items
                for item in selected_items:
                    if item.parent() is None:  # Ordner
                        self.delete_folder(item)
                    else:  # Datei
                        self.delete_measurement(item.text(0), item)
            elif action == add_multiple_folders_action:
                self.add_multiple_vertical_folders()

    def delete_folder(self, folder_item):
        """
        Löscht einen Ordner und alle enthaltenen Dateien.
        Nach dem Löschen werden ALLE verbleibenden Ordner automatisch neu benannt
        und die Meridian-Winkel linear auf 360° verteilt.
        """
        # Lösche alle Kinder des Ordners
        for i in range(folder_item.childCount() - 1, -1, -1):  # Rückwärts durchlaufen
            child_item = folder_item.child(i)
            child_name = child_item.text(0)
            
            # Entferne das Item aus dem TreeWidget
            folder_item.removeChild(child_item)
            
            # Lösche die Daten, aber nur wenn keine anderen Items mit demselben Namen existieren
            if not self.find_items_by_name(child_name, folder_item):
                if child_name in self.data['raw_measurements']:
                    del self.data['raw_measurements'][child_name]
                if child_name in self.data['calculated_data']:
                    del self.data['calculated_data'][child_name]
        
        # Entferne den Ordner selbst
        parent = folder_item.parent()
        if parent:
            index = parent.indexOfChild(folder_item)
            parent.takeChild(index)
        else:
            index = self.tree_widget.indexOfTopLevelItem(folder_item)
            self.tree_widget.takeTopLevelItem(index)
        
        # ================================================================
        # AUTOMATISCHE UMBENENNUNG ALLER VERBLEIBENDEN ORDNER
        # ================================================================
        num_folders = self.tree_widget.topLevelItemCount()
        
        # Nur umbenennen, wenn noch Ordner vorhanden sind
        if num_folders > 0:
            # Berechne neue Winkelverteilung basierend auf der verbleibenden Anzahl
            spacing_degrees = 360.0 / num_folders
            
            # Erstelle Winkel-Array: 0°, spacing°, 2*spacing°, ...
            meridian_angles = [i * spacing_degrees for i in range(num_folders)]
            
            # Benenne ALLE verbleibenden Ordner um
            for i in range(num_folders):
                folder = self.tree_widget.topLevelItem(i)
                angle = meridian_angles[i]
                
                # Formatiere den Winkel (nur Dezimalstellen wenn nötig)
                if angle % 1 == 0:
                    new_name = f"{int(angle)}° Meridian"
                else:
                    new_name = f"{angle:.1f}° Meridian"
                
                folder.setText(0, new_name)
            
            # Stelle sicher, dass ein Item ausgewählt ist
            self.ensure_selection()
        
        # Aktualisiere die Plots
        self.update_plot_tree_measurement()

    def find_items_by_name(self, name, exclude_parent=None):
        """Findet alle Items mit dem gegebenen Namen, außer in exclude_parent"""
        found_items = []
        
        def search_in_item(item):
            # Überspringe den ausgeschlossenen Elternordner
            if item == exclude_parent:
                return
            
            # Prüfe alle Kinder des Items
            for i in range(item.childCount()):
                child = item.child(i)
                if child.text(0) == name:
                    found_items.append(child)
                # Rekursiv in Unterordnern suchen
                if child.childCount() > 0:
                    search_in_item(child)
        
        # Durchsuche alle Top-Level-Items
        for i in range(self.tree_widget.topLevelItemCount()):
            top_item = self.tree_widget.topLevelItem(i)
            if top_item is not None and top_item.text(0) == name:  # type: ignore
                found_items.append(top_item)
            if top_item is not None:
                search_in_item(top_item)
        
        return found_items

    def duplicate_folder(self, folder_item):
        """
        Dupliziert einen Ordner mit allen enthaltenen Dateien.
        Nach dem Duplizieren werden ALLE Ordner automatisch neu benannt
        und die Meridian-Winkel linear auf 360° verteilt.
        """
        folder_name = folder_item.text(0)
        new_folder_name = f"copy_{folder_name}"
        
        # Erstelle einen neuen Ordner (zunächst mit temporärem Namen)
        new_folder = QTreeWidgetItem(self.tree_widget, [new_folder_name])
        
        # Kopiere alle Dateien im Ordner
        for i in range(folder_item.childCount()):
            child = folder_item.child(i)
            filename = child.text(0)
            new_filename = f"copy_{filename}"  # Füge Präfix auch für Dateien hinzu
            
            # Kopiere die Daten
            if filename in self.data['raw_measurements']:
                self.data['raw_measurements'][new_filename] = self.data['raw_measurements'][filename].copy()
            if filename in self.data['calculated_data']:
                self.data['calculated_data'][new_filename] = self.data['calculated_data'][filename].copy()
            
            # Füge zum neuen Ordner hinzu
            QTreeWidgetItem(new_folder, [new_filename])
        
        # Expandiere den neuen Ordner
        new_folder.setExpanded(True)
        
        # ================================================================
        # AUTOMATISCHE UMBENENNUNG ALLER ORDNER
        # ================================================================
        # Berechne neue Winkelverteilung basierend auf der AKTUELLEN Anzahl Ordner
        num_folders = self.tree_widget.topLevelItemCount()
        spacing_degrees = 360.0 / num_folders
        
        # Erstelle Winkel-Array: 0°, spacing°, 2*spacing°, ...
        meridian_angles = [i * spacing_degrees for i in range(num_folders)]
        
        # Benenne ALLE Ordner um (inklusive existierende!)
        for i in range(num_folders):
            folder = self.tree_widget.topLevelItem(i)
            angle = meridian_angles[i]
            
            # Formatiere den Winkel (nur Dezimalstellen wenn nötig)
            if angle % 1 == 0:
                new_name = f"{int(angle)}° Meridian"
            else:
                new_name = f"{angle:.1f}° Meridian"
            
            folder.setText(0, new_name)
        
        # Wähle den neu duplizierten Ordner aus (letzter in der Liste)
        self.tree_widget.setCurrentItem(new_folder)
        
        print(f"✅ Ordner dupliziert! {num_folders} Ordner auf 360° verteilt (Abstand: {spacing_degrees:.1f}°)")

    def add_multiple_vertical_folders(self):
        """Fügt mehrere Ordner für vertikale Daten hinzu basierend auf Benutzer-Eingabe"""
        # Erstelle einen benutzerdefinierten Dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add vertical data folder")
        dialog.setModal(True)
        dialog.setFixedSize(300, 120)
        
        # Layout für den Dialog
        layout = QVBoxLayout(dialog)
        
        # Label und SpinBox für die Anzahl der Ordner
        input_layout = QHBoxLayout()
        label = QLabel("Number of meridian folders:")
        spinbox = QSpinBox()
        spinbox.setMinimum(1)  # Mindestens 1 Ordner
        spinbox.setMaximum(72)  # Maximum 72 Ordner (= 5° Abstände)
        spinbox.setValue(2)  # Standard: 2 Ordner
        spinbox.setFixedWidth(80)
        input_layout.addWidget(label)
        input_layout.addWidget(spinbox)
        input_layout.addStretch()
        layout.addLayout(input_layout)
        
        # Spacer
        layout.addSpacing(20)
        
        # Buttons (Cancel und Save)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Cancel")
        save_button = QPushButton("Save")
        cancel_button.setFixedWidth(80)
        save_button.setFixedWidth(80)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Verbinde die Buttons
        cancel_button.clicked.connect(dialog.reject)
        save_button.clicked.connect(dialog.accept)
        
        # Zeige den Dialog und warte auf Benutzer-Eingabe
        if dialog.exec_() == QDialog.Accepted:
            # Zeige Warnung nur, wenn tatsächlich Daten vorhanden sind
            if self.tree_widget.topLevelItemCount() > 0:
                reply = QMessageBox.question(
                    self,
                    'Warning',
                    'All measurement data and folders will be deleted. Continue?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                # Wenn der Benutzer nicht bestätigt, abbrechen
                if reply != QMessageBox.Yes:
                    return
                
                # Lösche alle existierenden Ordner und Daten
                self.clear_all_data()
            
            # Hole die Anzahl der Ordner
            num_folders = spinbox.value()
            
            # Berechne den Gradabstand basierend auf der Anzahl der Ordner
            spacing_degrees = 360.0 / num_folders
            
            # Berechne die Meridian-Winkel von 0° bis <360°
            # Meridian = vertikaler Kreis, vollständige Rotation um X-Achse (Lautsprecherbreite), in Y-Z-Ebene
            # Jeder Ordner hat 180° Polar-Daten (horizontale Drehung von 0° bis 180°)
            
            # Erstelle Winkel mit dem berechneten Abstand: 0°, spacing°, 2*spacing°, ...
            meridian_angles = [i * spacing_degrees for i in range(num_folders)]
            
            # Erstelle die angegebene Anzahl von Ordnern mit korrekten Winkeln
            last_folder = None
            for i in range(num_folders):
                angle = meridian_angles[i]
                
                # Formatiere den Winkel (nur Dezimalstellen wenn nötig)
                if angle % 1 == 0:
                    folder_name = f"{int(angle)}° Meridian"
                else:
                    folder_name = f"{angle:.1f}° Meridian"
                
                # Füge den Ordner als Top-Level-Item hinzu
                new_folder = QTreeWidgetItem(self.tree_widget, [folder_name])
                last_folder = new_folder
            
            # Wähle den letzten erstellten Ordner aus
            if last_folder:
                self.tree_widget.setCurrentItem(last_folder)

    def clear_all_data(self):
        """Löscht alle Ordner und Daten aus dem TreeWidget und der Datenstruktur"""
        try:
            # Lösche alle TreeWidget Items
            self.tree_widget.clear()
            
            # Lösche alle Daten aus der Datenstruktur
            self.data['raw_measurements'].clear()
            self.data['calculated_data'].clear()
            self.data['polar_data'].clear()
            self.data['interpolated_data'].clear()
            
            # Optional: Lösche auch balloon_data falls vorhanden
            if 'balloon_data' in self.data:
                self.data['balloon_data'].clear()
            
            # Zeige leere Plot-Darstellung
            if hasattr(self.plot_handler, 'initialize_empty_plots'):
                self.plot_handler.initialize_empty_plots()
            
        except Exception as e:
            print(f"Fehler beim Löschen aller Daten: {str(e)}")
            import traceback
            traceback.print_exc()

    def delete_measurement(self, filename, item):
        """Löscht eine Messung und alle zugehörigen Daten"""
        try:
            # Prüfe, ob es sich um einen Ordner handelt
            if item and item.childCount() > 0:
                # Lösche alle Kinder des Ordners
                for i in range(item.childCount() - 1, -1, -1):  # Rückwärts durchlaufen
                    child_item = item.child(i)
                    child_name = child_item.text(0)
                    self.delete_measurement(child_name, child_item)
                
                # Entferne den Ordner selbst
                parent = item.parent()
                if parent:
                    index = parent.indexOfChild(item)
                    parent.takeChild(index)
                else:
                    index = self.tree_widget.indexOfTopLevelItem(item)
                    self.tree_widget.takeTopLevelItem(index)
                
                return True
            
            # Lösche die ausgewählte Datei aus den Daten
            if filename in self.data['raw_measurements']:
                del self.data['raw_measurements'][filename]
            
            if filename in self.data['calculated_data']:
                del self.data['calculated_data'][filename]
            
            # Entferne das Item aus dem TreeWidget
            if item:
                parent = item.parent()
                if parent:
                    index = parent.indexOfChild(item)
                    parent.takeChild(index)
                else:
                    index = self.tree_widget.indexOfTopLevelItem(item)
                    self.tree_widget.takeTopLevelItem(index)
            
            # Prüfe ob noch Daten für verbleibende TreeWidget-Items existieren
            remaining_files = []
            
            # Sammle alle verbleibenden Dateien (nicht Ordner) im TreeWidget
            def collect_files(parent_item):
                files = []
                for i in range(parent_item.childCount()):
                    child = parent_item.child(i)
                    if child.childCount() > 0:  # Ist ein Ordner
                        files.extend(collect_files(child))
                    else:  # Ist eine Datei
                        files.append(child.text(0))
                return files
            
            # Sammle Top-Level-Items
            for i in range(self.tree_widget.topLevelItemCount()):
                top_item = self.tree_widget.topLevelItem(i)
                if top_item is not None and top_item.childCount() > 0:  # type: ignore
                    remaining_files.extend(collect_files(top_item))
                elif top_item is not None:  # type: ignore
                    remaining_files.append(top_item.text(0))  # type: ignore
            
            has_remaining_data = False
            for file in remaining_files:
                if (file in self.data['raw_measurements'] or 
                    file in self.data['calculated_data']):
                    has_remaining_data = True
                    break
            
            # Nur löschen wenn keine Daten für verbleibende Items existieren
            if not has_remaining_data:
                if 'polar_data' in self.data:
                    self.data['polar_data'] = {}
                
                if 'interpolated_data' in self.data:
                    self.data['interpolated_data'] = {}
            
            # Update UI und Plots
            if remaining_files:
                # Wenn noch Dateien übrig sind, stelle sicher dass ein Item ausgewählt ist
                self.ensure_selection()
                self.on_item_moved()
            else:
                # Keine Dateien mehr vorhanden - zeige leere Plot-Darstellung
                if hasattr(self.plot_handler, 'initialize_empty_plots'):
                    self.plot_handler.initialize_empty_plots()
            
            return True
            
        except Exception as e:
            print(f"Fehler beim Löschen der Messung: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# ------ Import Files ------

    def load_files(self, data_type):
        """Lädt Dateien und fügt sie zum ausgewählten Ordner im TreeWidget hinzu"""
        if data_type == "Impulse Response":
            file_filter = "WAV Files (*.wav)"
        else:  # TXT Frequency Response
            file_filter = "TXT Files (*.txt)"
            
        filenames, selected_filter = QFileDialog.getOpenFileNames(
            self, 
            "Messdaten laden", 
            "", 
            file_filter
        )
        if not filenames:
            return
        
        # Bestimme den Zielordner
        selected_item = self.tree_widget.currentItem()
        
        # Stelle sicher, dass immer ein Item ausgewählt ist
        if not selected_item:
            self.ensure_selection()
            selected_item = self.tree_widget.currentItem()
        
        # Wenn das ausgewählte Item ein File ist, verwende dessen Elternordner
        if selected_item.parent():
            selected_item = selected_item.parent()
        
        total_steps = len(filenames) + 1
        with self.progress_manager.start("Importing measurement data", total_steps) as progress:
            # Verarbeite die ausgewählten Dateien
            for filename in filenames:
                file_extension = filename.lower().split('.')[-1]
                basename = os.path.basename(filename)
                progress.update(f"Importing {basename}")

                try:
                    success = False
                    if file_extension == 'wav':
                        success = self.polar_calculator.process_impulse_response(filename)
                    elif file_extension == 'txt':
                        success = self.polar_calculator.process_txt_response(filename)
                    else:
                        print(f"Warnung: Nicht unterstütztes Dateiformat: {file_extension}")
                        continue

                    if not success:
                        print(f"ERROR load_files: Import fehlgeschlagen für '{filename}'")
                        continue

                    # Füge Datei zum ausgewählten Ordner hinzu
                    QTreeWidgetItem(selected_item, [basename])

                except Exception as e:
                    print(f"ERROR load_files: Fehler beim Laden von {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                finally:
                    progress.advance()

            progress.update("Updating measurement overview")
            selected_item.setExpanded(True)

            if selected_item.childCount() > 0:
                first_item = selected_item.child(0)
                self.tree_widget.setCurrentItem(first_item)
                self.sync_ui_with_metadata()
                self.polar_calculator.recalculate_from_raw()
                self.on_tree_item_clicked(first_item, 0)
            progress.advance()

    def sync_ui_with_metadata(self):
        """Synchronisiert UI-Komponenten mit den Metadaten"""
        try:
            # Synchronisiere IR Window Checkbox
            if 'ir_window_enabled' in self.data['metadata']:
                self.ir_window_checkbox.blockSignals(True)
                self.ir_window_checkbox.setChecked(self.data['metadata']['ir_window_enabled'])
                self.ir_window_checkbox.blockSignals(False)
                
                # IR Window Eingabefeld bleibt immer aktiviert
                self.ir_window_input.setEnabled(True)
            
            # Synchronisiere IR Window Input
            if 'ir_window' in self.data['metadata']:
                self.ir_window_input.blockSignals(True)
                self.ir_window_input.setText(str(self.data['metadata']['ir_window']))
                self.ir_window_input.blockSignals(False)
            
            # Synchronisiere Time Offset Input
            if 'time_offset' in self.data['metadata']:
                self.time_input.blockSignals(True)
                self.time_input.setText(str(self.data['metadata']['time_offset']))
                self.time_input.blockSignals(False)
            
            # Synchronisiere Filter Checkbox
            if 'filter_enabled' in self.data['metadata']:
                self.filter_checkbox.blockSignals(True)
                self.filter_checkbox.setChecked(self.data['metadata']['filter_enabled'])
                self.filter_checkbox.blockSignals(False)
                
        except Exception as e:
            print(f"Fehler beim Synchronisieren der UI-Komponenten: {str(e)}")
            import traceback
            traceback.print_exc()


# ----- Update -----

    def update_plot_tree_measurement(self, filename=None):
        """Aktualisiert die Plots für die ausgewählte Datei"""
        # Wenn kein Dateiname übergeben wurde, versuche einen zu finden
        if filename is None:
            # Hole das aktuell ausgewählte Item
            current_item = self.tree_widget.currentItem()
            if current_item and current_item.parent():  # Wenn es ein File ist
                filename = current_item.text(0)
            elif self.tree_widget.topLevelItemCount() > 0:  # Wenn kein File ausgewählt ist, aber Items existieren
                # Suche nach dem ersten File in allen Ordnern
                for i in range(self.tree_widget.topLevelItemCount()):
                    folder = self.tree_widget.topLevelItem(i)
                    if folder is not None and folder.childCount() > 0:  # type: ignore
                        filename = folder.child(0).text(0)  # type: ignore
                        break
        
        # Wenn ein gültiger Dateiname gefunden wurde und die Daten existieren
        if filename and (filename in self.data['raw_measurements'] or filename in self.data['calculated_data']):
            data = self.data            
            # Übergebe alle Messdaten für globale Skalierung
            self.plot_handler.update_plot(
                data, 
                filename
            )
        else:
            # Keine Daten vorhanden - zeige leere Plot-Darstellung
            self.plot_handler.initialize_empty_plots()

    def update_polar_plot(self):
        """Aktualisiert nur die Darstellung des Polar Plots"""  
        try:
            self.plot_handler.update_polar_plot(self.data)
            
        except Exception as e:
            print(f"Fehler beim Update des Polar Plots: {e}")
            import traceback
            traceback.print_exc()

    def update_balloon_plot(self):
        """Aktualisiert den Balloon Plot"""
        try:
            self.plot_handler.update_balloon_plot(self.data)
        except Exception as e:
            print(f"Fehler beim Update des Balloon Plots: {e}")
            import traceback
            traceback.print_exc()

    def update_cabinet_plot(self):
        """Aktualisiert den Cabinet Plot mit den aktuellen Lautsprecherdaten"""
        try:
            # Sammle die Daten aller Lautsprecher
            cabinet_data = []
            
            for speaker in self.speakers:
                try:
                    # Robuste Float-Konvertierung (behandelt leere Strings, ".", etc.)
                    def safe_float(text_widget, default=0.0):
                        try:
                            text = text_widget.text().strip()
                            return float(text) if text and text != '.' else default
                        except (ValueError, AttributeError):
                            return default
                    
                    # Grundlegende Dimensionen
                    width = safe_float(speaker['width'])
                    depth = safe_float(speaker['depth'])
                    
                    # Verwende die separaten Höhenfelder für Front und Back
                    front_height = safe_float(speaker['front_height'])
                    back_height = safe_float(speaker['back_height'])
                    
                    # Cardio-Eigenschaft
                    is_cardio = speaker['cardio'].isChecked()
                    
                    # Position (Stack/Flown)
                    is_flown = (speaker['configuration'].currentText() == "Flown")
                    
                    # Winkel-Punkt
                    angle_point = speaker['angle_point'].currentText()
                    
                    # Winkel-Liste
                    angles = []
                    for angle_editor in speaker['angle_items']:
                        angles.append(safe_float(angle_editor))
                    
                    # Erstelle ein Dictionary mit allen Lautsprecherdaten
                    speaker_dict = {
                        'width': width,
                        'depth': depth,
                        'front_height': front_height,
                        'back_height': back_height,
                        'cardio': is_cardio,
                        'is_flown': is_flown,
                        'angle_point': angle_point,
                        'angles': angles
                    }

                    # Stack-spezifische Parameter nur für Stack-Layouts speichern
                    if not is_flown:
                        stack_layout_combo = speaker.get('stack_layout')
                        stack_layout = 'beside'
                        if stack_layout_combo and stack_layout_combo.isEnabled():
                            stack_layout = stack_layout_combo.currentText().strip().lower()
                        speaker_dict['stack_layout'] = stack_layout
                    
                    cabinet_data.append(speaker_dict)
                    
                except ValueError as e:
                    print(f"Fehler bei der Verarbeitung eines Lautsprechers: {e}")
            
            # Aktualisiere den Plot
            self.plot_handler.update_cabinet_plot(cabinet_data)
            
            # Speichere die Daten in der zentralen Datenstruktur
            self.data['metadata']['cabinet_data'] = cabinet_data
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren des Cabinet Plots: {e}")
            import traceback
            traceback.print_exc()

    def validate_min_freq(self):
        """Validiert die minimale Frequenz"""
        try:
            min_freq = float(self.min_freq_input.text())
            max_freq = float(self.max_freq_input.text())
            
            # Prüfe Grenzen
            if min_freq < 5:
                min_freq = 5
            elif min_freq > 1000:
                min_freq = 1000
                
            # Prüfe Überlappung mit Maximum
            if min_freq >= max_freq:
                min_freq = max_freq - 1
                
            # Aktualisiere Eingabefeld
            self.min_freq_input.setText(f"{min_freq:.1f}")
            
            self.data['metadata']['filter_freq_min'] = min_freq
        except ValueError:
             self.min_freq_input.setText(f"{20:.1f}")

        self.apply_frequency_bandwidth()
        self.update_plot_tree_measurement()

    def validate_max_freq(self):
        """Validiert die maximale Frequenz"""
        try:
            min_freq = float(self.min_freq_input.text())
            max_freq = float(self.max_freq_input.text())
            
            # Prüfe Grenzen - maximale Frequenz auf 300 Hz begrenzen
            if max_freq < 50:
                max_freq = 50
            elif max_freq > 300:
                max_freq = 300
                
            # Prüfe Überlappung mit Minimum
            if max_freq <= min_freq:
                max_freq = min_freq + 1
                
            # Aktualisiere Eingabefeld
            self.max_freq_input.setText(f"{max_freq:.1f}")
              
            self.data['metadata']['filter_freq_max'] = max_freq

        except ValueError:
            self.max_freq_input.setText(f"{300:.1f}")

        self.apply_frequency_bandwidth()
        self.update_plot_tree_measurement()

    def apply_frequency_bandwidth(self):
        """Wendet die Frequenzbandbreiten-Beschränkung auf alle Messungen an"""
        try:
            if not self.data['raw_measurements']:
                return False
            
            # Neuberechnung der verarbeiteten Daten
            self.polar_calculator.recalculate_from_raw()
            
            # Update UI wenn nötig
            current_item = self.tree_widget.currentItem()
            if current_item:
                self.on_tree_item_clicked(current_item, 0)
            
            return True
            
        except Exception as e:
            print(f"Fehler bei Frequenzbandbreiten-Beschränkung: {str(e)}")
            return False


# ----- Stignalhandler ----- 

    def on_item_moved(self, parent=None, start=None, end=None):
        """
        Wird aufgerufen, wenn ein Item per Drag & Drop verschoben wurde.
        Nach dem Verschieben werden ALLE Ordner automatisch neu benannt
        und die Meridian-Winkel linear auf 360° verteilt.
        """
        
        # ================================================================
        # AUTOMATISCHE UMBENENNUNG ALLER ORDNER NACH UMSORTIERUNG
        # ================================================================
        num_folders = self.tree_widget.topLevelItemCount()
        
        if num_folders > 0:
            # Berechne neue Winkelverteilung basierend auf der aktuellen Position
            spacing_degrees = 360.0 / num_folders
            
            # Erstelle Winkel-Array: 0°, spacing°, 2*spacing°, ...
            meridian_angles = [i * spacing_degrees for i in range(num_folders)]
            
            # Benenne ALLE Ordner um basierend auf ihrer NEUEN Position
            for i in range(num_folders):
                folder = self.tree_widget.topLevelItem(i)
                angle = meridian_angles[i]
                
                # Formatiere den Winkel (nur Dezimalstellen wenn nötig)
                if angle % 1 == 0:
                    new_name = f"{int(angle)}° Meridian"
                else:
                    new_name = f"{angle:.1f}° Meridian"
                
                folder.setText(0, new_name)
            
        
        # ================================================================
        # PLOT UPDATE
        # ================================================================
        # Aktualisiere die Plots
        current_item = self.tree_widget.currentItem()
        if current_item:
            filename = current_item.text(0)
            if filename in self.data['calculated_data'] or filename in self.data['raw_measurements']:
                self.update_plot_tree_measurement(filename)
            else:
                # Wenn das aktuelle Item ein Ordner ist oder keine Daten hat
                self.update_plot_tree_measurement()
        else:
            # Wenn kein Item ausgewählt ist
            self.update_plot_tree_measurement()

    def on_tree_item_clicked(self, item, column):
        """Reagiert auf Mausklicks auf Items"""
        filename = item.text(0)
        
        # Synchronisiere UI-Komponenten mit Metadaten
        self.sync_ui_with_metadata()
        
        if filename in self.data['calculated_data']:
            self.on_item_moved()
        elif filename in self.data['raw_measurements']:
            self.on_item_moved()
        else:
            print(f"WARNUNG: Keine Daten für {filename} gefunden")

    def on_checkbox_normalize_changed(self, _=None):
        """Handler für Änderungen der Checkbox"""
        # Eingabefelder bleiben immer aktiviert
        
        checkbox_state = self.spl_norm_checkbox.isChecked()
        
        if not checkbox_state:
            # Deaktiviere Normalisierung - verwende Raw-Daten
            # Setze spl_normalized auf False in der flachen Struktur
            self.data['metadata']['spl_normalized'] = False
        else:
            # Aktiviere Normalisierung NUR wenn alle Metadaten verfügbar sind
            # Hole Raw-SPL aus dem ersten File im "Horizontal data" Ordner
            reference_filename = None
            
            # Suche nach dem "Horizontal data" Ordner (oder "0° Horizontal data")
            horizontal_folder = None
            for i in range(self.tree_widget.topLevelItemCount()):
                folder_item = self.tree_widget.topLevelItem(i)
                folder_name = folder_item.text(0)
                # Prüfe auf "Horizontal data" im Namen
                if "horizontal data" in folder_name.lower():
                    horizontal_folder = folder_item
                    break
            
            # Fallback: Wenn kein "Horizontal data" Ordner gefunden, verwende ersten Ordner
            if horizontal_folder is None and self.tree_widget.topLevelItemCount() > 0:
                horizontal_folder = self.tree_widget.topLevelItem(0)
            
            # Hole das erste File aus diesem Ordner
            if horizontal_folder and horizontal_folder.childCount() > 0:
                reference_filename = horizontal_folder.child(0).text(0)
            
            if reference_filename and reference_filename in self.data['raw_measurements']:
                ref_data = self.data['raw_measurements'][reference_filename]
                reference_freq = float(self.spl_freq_combo.currentText())
                
                # Verwende Terzbandmittelung statt Einzelfrequenz
                freq_array = np.array(ref_data['freq'])
                mag_array = np.array(ref_data['magnitude'])
                raw_spl = self.polar_calculator.get_third_octave_spl(freq_array, mag_array, reference_freq)
                
                # Verwende aktuellen SPL-Wert aus Eingabefeld
                target_spl = float(self.spl_level_input.text().strip() or str(raw_spl))
                distance = float(self.spl_distance_input.text().strip() or "1.0")
                
                # Berechne value basierend auf target_spl und raw_spl
                value = target_spl + 20 * np.log10(distance) - raw_spl
                
                # Update Value
                self.spl_value_input.blockSignals(True)
                self.spl_value_input.setText(f"{value:.1f}")
                self.spl_value_input.blockSignals(False)
                
                # Update Metadaten in flacher Struktur
                self.data['metadata'].update({
                    'spl_value': value,
                    'target_spl': target_spl,
                    'reference_freq': reference_freq,
                    'reference_distance': distance
                })

                # Entferne alte verschachtelte Struktur, falls vorhanden
                if 'normalization' in self.data['metadata']:
                    if 'spl' in self.data['metadata']['normalization']:
                        del self.data['metadata']['normalization']['spl']
                
                # WICHTIG: Aktiviere Normalisierung ERST nachdem alle Metadaten gesetzt sind!
                self.data['metadata']['spl_normalized'] = True
            else:
                # Wenn keine Referenzdaten gefunden, Checkbox deaktivieren
                print("⚠ Warnung: Keine Referenzdaten für SPL-Normalisierung gefunden!")
                self.data['metadata']['spl_normalized'] = False
                self.spl_norm_checkbox.blockSignals(True)
                self.spl_norm_checkbox.setChecked(False)
                self.spl_norm_checkbox.blockSignals(False)
                return False

        # Neuberechnung und UI-Update
        self.polar_calculator.recalculate_from_raw()
        
        # Explizites Plot-Update für blauen Plot
        if current_item := self.tree_widget.currentItem():
            filename = current_item.text(0)
            if filename in self.data['calculated_data'] or filename in self.data['raw_measurements']:
                self.update_plot_tree_measurement(filename)
            else:
                self.update_plot_tree_measurement()
        else:
            self.update_plot_tree_measurement()
        
        return True

    def on_calculate_polar_clicked(self):
        """Handler für den Calculate Polar Button"""
        # Verwende feste Y-Achsen-Interpolation (Backup-Version)
        if 'metadata' not in self.data:
            self.data['metadata'] = {}
        self.data['metadata']['interpolation_method'] = "Um Y-Achse (horizontal)"
        with self.progress_manager.start("Generating balloon data", 2) as progress:
            progress.update("Calculating interpolated data")
            success = self.polar_calculator.create_balloon_data()
            progress.advance()
            progress.update("Updating plots")
            if success:
                self.update_plot_polar_balloon()
            progress.advance()

    def on_freq_normalize_changed(self):
        """Aktualisiert die Referenzfrequenz für die SPL-Normalisierung"""
        freq = float(self.spl_freq_combo.currentText())
        self.data['metadata']['reference_freq'] = freq
        
        """Handler für Frequenz-Input: Passt value an, um SPL bei neuer Frequenz beizubehalten"""
        try:
            # Validiere Eingaben
            reference_freq = float(self.spl_freq_combo.currentText())
            target_spl = float(self.spl_level_input.text().strip())  # SPL soll konstant bleiben
            distance = float(self.spl_distance_input.text().strip())
            
            # Hole Raw-SPL aus dem ersten File im "Horizontal data" Ordner
            reference_filename = None
            
            # Suche nach dem "Horizontal data" Ordner
            horizontal_folder = None
            for i in range(self.tree_widget.topLevelItemCount()):
                folder_item = self.tree_widget.topLevelItem(i)
                folder_name = folder_item.text(0)
                if "horizontal data" in folder_name.lower():
                    horizontal_folder = folder_item
                    break
            
            # Fallback auf ersten Ordner
            if horizontal_folder is None and self.tree_widget.topLevelItemCount() > 0:
                horizontal_folder = self.tree_widget.topLevelItem(0)
            
            # Hole das erste File
            if horizontal_folder and horizontal_folder.childCount() > 0:
                reference_filename = horizontal_folder.child(0).text(0)
            
            if reference_filename and reference_filename in self.data['raw_measurements']:
                ref_data = self.data['raw_measurements'][reference_filename]
                
                # Verwende Terzbandmittelung statt Einzelfrequenz
                freq_array = np.array(ref_data['freq'])
                mag_array = np.array(ref_data['magnitude'])
                raw_spl = self.polar_calculator.get_third_octave_spl(freq_array, mag_array, reference_freq)
                
                # Berechne neuen value um target_spl beizubehalten
                value = target_spl + 20 * np.log10(distance) - raw_spl
                
                # Update UI
                self.spl_value_input.blockSignals(True)
                self.spl_value_input.setText(f"{value:.1f}")
                self.spl_value_input.blockSignals(False)
                self.last_valid_value = value
                
                # Update Metadaten (OHNE spl_normalized zu ändern!)
                self.data['metadata'].update({
                    'spl_value': value,
                    'target_spl': target_spl,
                    'reference_freq': reference_freq,
                    'reference_distance': distance
                })
                
                # Neuberechnung nur wenn Checkbox aktiv
                if self.spl_norm_checkbox.isChecked():
                    self.polar_calculator.recalculate_from_raw()
                    current_item = self.tree_widget.currentItem()
                    if current_item:
                        self.on_tree_item_clicked(current_item, 0)
                
        except ValueError as e:
            print(f"Fehler bei Frequenzänderung: {str(e)}")

    def on_distance_normalize_changed(self):
        """Handler für Distanz-Input: Passt value an, um SPL bei neuer Distanz beizubehalten"""
        try:
            # Validiere Eingaben
            distance = float(self.spl_distance_input.text().strip())
            target_spl = float(self.spl_level_input.text().strip())  # SPL soll konstant bleiben
            reference_freq = float(self.spl_freq_combo.currentText())
            
            # Hole Raw-SPL aus dem ersten File im "Horizontal data" Ordner
            reference_filename = None
            
            # Suche nach dem "Horizontal data" Ordner
            horizontal_folder = None
            for i in range(self.tree_widget.topLevelItemCount()):
                folder_item = self.tree_widget.topLevelItem(i)
                folder_name = folder_item.text(0)
                if "horizontal data" in folder_name.lower():
                    horizontal_folder = folder_item
                    break
            
            # Fallback auf ersten Ordner
            if horizontal_folder is None and self.tree_widget.topLevelItemCount() > 0:
                horizontal_folder = self.tree_widget.topLevelItem(0)
            
            # Hole das erste File
            if horizontal_folder and horizontal_folder.childCount() > 0:
                reference_filename = horizontal_folder.child(0).text(0)
            
            if reference_filename and reference_filename in self.data['raw_measurements']:
                ref_data = self.data['raw_measurements'][reference_filename]
                
                # Verwende Terzbandmittelung statt Einzelfrequenz
                freq_array = np.array(ref_data['freq'])
                mag_array = np.array(ref_data['magnitude'])
                raw_spl = self.polar_calculator.get_third_octave_spl(freq_array, mag_array, reference_freq)
                
                # Berechne neuen value um target_spl bei neuer Distanz zu erreichen
                value = target_spl + 20 * np.log10(distance) - raw_spl
                
                # Update UI
                self.spl_value_input.blockSignals(True)
                self.spl_value_input.setText(f"{value:.1f}")
                self.spl_value_input.blockSignals(False)
                self.last_valid_value = value
                
                # Update Metadaten (OHNE spl_normalized zu ändern!)
                self.data['metadata'].update({
                    'spl_value': value,
                    'target_spl': target_spl,
                    'reference_freq': reference_freq,
                    'reference_distance': distance
                })
                
                # Neuberechnung nur wenn Checkbox aktiv
                if self.spl_norm_checkbox.isChecked():
                    self.polar_calculator.recalculate_from_raw()
                    current_item = self.tree_widget.currentItem()
                    if current_item:
                        self.on_tree_item_clicked(current_item, 0)
                
        except ValueError:
            self.spl_distance_input.setStyleSheet("background-color: #FFE4E1;")
            QTimer.singleShot(500, lambda: self.reset_input_style(self.spl_distance_input))
            self.spl_distance_input.setText("1.0")

    def on_spl_normalize_changed(self):
        """Handler für Änderungen an der SPL-Normalisierung"""
        try:
            # Alle Eingabefelder bleiben immer aktiviert
            self.spl_level_input.setEnabled(True)  # Immer aktiviert
            self.spl_distance_input.setEnabled(True)  # Immer aktiviert
            self.spl_freq_combo.setEnabled(True)  # Immer aktiviert
            self.spl_value_input.setEnabled(True)  # Immer aktiviert
            
            # IMMER Value berechnen und anzeigen (unabhängig von Checkbox)
            try:
                # Hole die Werte aus den Eingabefeldern
                target_spl_at_distance = float(self.spl_level_input.text())
                distance = float(self.spl_distance_input.text())
                reference_freq = float(self.spl_freq_combo.currentText())
                
                # Speichere die letzten gültigen Werte
                self.last_valid_spl = target_spl_at_distance
                self.last_valid_distance = distance
                self.last_valid_freq = reference_freq
                
                # Hole Raw-SPL aus dem ersten File im "Horizontal data" Ordner
                reference_filename = None
                
                # Suche nach dem "Horizontal data" Ordner
                horizontal_folder = None
                for i in range(self.tree_widget.topLevelItemCount()):
                    folder_item = self.tree_widget.topLevelItem(i)
                    folder_name = folder_item.text(0)
                    if "horizontal data" in folder_name.lower():
                        horizontal_folder = folder_item
                        break
                
                # Fallback auf ersten Ordner
                if horizontal_folder is None and self.tree_widget.topLevelItemCount() > 0:
                    horizontal_folder = self.tree_widget.topLevelItem(0)
                
                # Hole das erste File
                if horizontal_folder and horizontal_folder.childCount() > 0:
                    reference_filename = horizontal_folder.child(0).text(0)
                
                if reference_filename and reference_filename in self.data['raw_measurements']:
                    ref_data = self.data['raw_measurements'][reference_filename]
                    
                    # Verwende Terzbandmittelung statt Einzelfrequenz
                    freq_array = np.array(ref_data['freq'])
                    mag_array = np.array(ref_data['magnitude'])
                    raw_spl = self.polar_calculator.get_third_octave_spl(freq_array, mag_array, reference_freq)
                    
                    # Berechne value (Korrekturwert) - IMMER
                    # target_spl_at_distance = raw_spl + value - 20*log10(distance)
                    # -> value = target_spl_at_distance + 20*log10(distance) - raw_spl
                    value = target_spl_at_distance + 20 * np.log10(distance) - raw_spl
                    
                    # Update UI - IMMER
                    self.spl_value_input.blockSignals(True)
                    self.spl_value_input.setText(f"{value:.1f}")
                    self.spl_value_input.blockSignals(False)
                    self.last_valid_value = value
                    
                    # Update Metadaten - IMMER
                    self.data['metadata'].update({
                        'spl_value': value,
                        'target_spl': target_spl_at_distance,
                        'reference_freq': reference_freq,
                        'reference_distance': distance
                    })
                    
                    # Prüfe ob Checkbox aktiviert ist
                    is_normalized = self.spl_norm_checkbox.isChecked()
                    
                    if is_normalized:
                        # Anwendung auf Plots NUR wenn Checkbox aktiv
                        self.data['metadata']['spl_normalized'] = True
                        self.polar_calculator.recalculate_from_raw()
                        current_item = self.tree_widget.currentItem()
                        if current_item:
                            self.on_tree_item_clicked(current_item, 0)
                    else:
                        # Wenn nicht normalisiert, deaktiviere Anwendung
                        self.data['metadata']['spl_normalized'] = False
                        # Neuberechnung ohne Normalisierung
                        self.polar_calculator.recalculate_from_raw()
                        current_item = self.tree_widget.currentItem()
                        if current_item:
                            self.on_tree_item_clicked(current_item, 0)
        
            except ValueError:
                self.spl_level_input.setStyleSheet("background-color: #FFE4E1;")
                QTimer.singleShot(500, lambda: self.reset_input_style(self.spl_level_input))
                if hasattr(self, 'last_valid_spl'):
                    self.spl_level_input.setText(f"{self.last_valid_spl:.1f}")
                    
        except Exception as e:
            pass

    def on_value_normalize_changed(self):
        """Handler für Value-Input: Berechnet SPL bei aktueller Distanz"""
        try:
            # Validiere Eingaben
            input_text = self.spl_value_input.text().strip()
            if not input_text:
                return
            
            value = float(input_text)
            value = round(value, 1)
            reference_freq = float(self.spl_freq_combo.currentText())
            distance = float(self.spl_distance_input.text().strip())
            
            # Hole Raw-SPL aus dem ersten File im "Horizontal data" Ordner
            reference_filename = None
            
            # Suche nach dem "Horizontal data" Ordner
            horizontal_folder = None
            for i in range(self.tree_widget.topLevelItemCount()):
                folder_item = self.tree_widget.topLevelItem(i)
                folder_name = folder_item.text(0)
                if "horizontal data" in folder_name.lower():
                    horizontal_folder = folder_item
                    break
            
            # Fallback auf ersten Ordner
            if horizontal_folder is None and self.tree_widget.topLevelItemCount() > 0:
                horizontal_folder = self.tree_widget.topLevelItem(0)
            
            # Hole das erste File
            if horizontal_folder and horizontal_folder.childCount() > 0:
                reference_filename = horizontal_folder.child(0).text(0)
            
            if reference_filename and reference_filename in self.data['raw_measurements']:
                ref_data = self.data['raw_measurements'][reference_filename]
                
                # Verwende Terzbandmittelung statt Einzelfrequenz
                freq_array = np.array(ref_data['freq'])
                mag_array = np.array(ref_data['magnitude'])
                raw_spl = self.polar_calculator.get_third_octave_spl(freq_array, mag_array, reference_freq)
                
                # Berechne SPL bei aktueller Distanz
                # target_spl_at_distance = raw_spl + value - 20*log10(distance)
                target_spl_at_distance = raw_spl + value - 20 * np.log10(distance)
                
                # Update UI
                self.spl_level_input.blockSignals(True)
                self.spl_level_input.setText(f"{target_spl_at_distance:.1f}")
                self.spl_level_input.blockSignals(False)
                self.last_valid_spl = target_spl_at_distance
                
                # Update Metadaten (OHNE spl_normalized zu ändern!)
                self.data['metadata'].update({
                    'spl_value': value,
                    'target_spl': target_spl_at_distance,
                    'reference_freq': reference_freq,
                    'reference_distance': distance
                })
                
                # Neuberechnung nur wenn Checkbox aktiv
                if self.spl_norm_checkbox.isChecked():
                    self.polar_calculator.recalculate_from_raw()
                    current_item = self.tree_widget.currentItem()
                    if current_item:
                        self.on_tree_item_clicked(current_item, 0)
                
        except ValueError:
            self.spl_value_input.setStyleSheet("background-color: #FFE4E1;")
            QTimer.singleShot(500, lambda: self.reset_input_style(self.spl_value_input))
            self.spl_value_input.setText(f"{self.last_valid_value:.1f}")

    def reset_input_style(self, input_widget):
        """Setzt den Stil eines Input-Widgets zurück"""
        input_widget.setStyleSheet("")

    def update_plot_polar_balloon(self):
        """Aktualisiert die Polar- und Balloon-Plots"""
        try:
            # Aktualisiere den Polar Plot (verwendet balloon_data["0"])
            self.plot_handler.update_polar_plot(self.data)
            
            # Aktualisiere den Balloon Plot
            if 'balloon_data' in self.data and self.data['balloon_data']:
                
                # Definiere die festen Frequenzen für die Auswahlbox (Terzbänder)
                fixed_freqs = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250]
                
                # Hole die verfügbaren Frequenzen aus balloon_data
                try:
                    # Prüfe ob neue NumPy-Struktur oder alte Dict-Struktur
                    if 'meridians' in self.data['balloon_data'] and isinstance(self.data['balloon_data']['meridians'], np.ndarray):
                        # NEUE NUMPY-STRUKTUR
                        available_freqs = np.array(self.data['balloon_data']['frequencies'])
                    else:
                        # ALTE DICT-STRUKTUR
                        first_meridian = next(iter(self.data['balloon_data']))
                        first_angle = next(iter(self.data['balloon_data'][first_meridian]))
                        available_freqs = np.array(self.data['balloon_data'][first_meridian][first_angle]['freq'])
                    
                except Exception as e:
                    print(f"⚠ Fehler beim Holen der Frequenzen aus balloon_data: {e}")
                    available_freqs = np.array([])
                
                # Aktualisiere die Frequenzauswahl
                self.balloon_freq_selector.blockSignals(True)
                self.balloon_freq_selector.clear()
                
                # Füge nur die Frequenzen hinzu, die in den Daten verfügbar sind
                if len(available_freqs) > 0:
                    for freq in fixed_freqs:
                        # Finde die nächste verfügbare Frequenz
                        closest_idx = np.abs(available_freqs - freq).argmin()
                        closest_freq = available_freqs[closest_idx]
                        # Nur hinzufügen, wenn die Frequenz nahe genug ist (±10%)
                        if abs(closest_freq - freq) / freq < 0.1:
                            self.balloon_freq_selector.addItem(f"{freq} Hz")
                
                self.balloon_freq_selector.blockSignals(False)
                
                # Setze 50 Hz als Standard-Frequenz
                default_index = self.balloon_freq_selector.findText("50 Hz")
                if default_index >= 0:
                    self.balloon_freq_selector.setCurrentIndex(default_index)
                    # Aktualisiere den Plot mit 50 Hz
                    self.plot_handler.update_balloon_plot(self.data, 50.0, show_freq_in_title=False)
                else:
                    # Wenn 50 Hz nicht verfügbar, verwende die erste verfügbare Frequenz
                    self.plot_handler.update_balloon_plot(self.data, show_freq_in_title=False)
            else:
                print("⚠ Keine balloon_data für Plots vorhanden")
                
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Polar/Balloon Plots: {e}")
            import traceback
            traceback.print_exc()

    def on_validate_time_input(self):
        """Validiert die Zeiteingabe und erlaubt positive und negative Werte"""
        try:
            time_text = self.time_input.text().strip()
            if not time_text:
                self.time_input.setText("0.00")
                return 0.0

            time_value = float(time_text)

            formatted_text = f"{time_value:.2f}"
            if self.time_input.text() != formatted_text:
                self.time_input.setText(formatted_text)
            return time_value
        except ValueError:

            self.time_input.setStyleSheet("background-color: #FFE4E1;")
            QTimer.singleShot(500, lambda: self.reset_input_style(self.time_input))
            self.time_input.setText("0.00")
        return 0.0

    def on_time_value_changed(self):
        """Handler für manuelle Änderungen des Zeitwerts"""
        try:
            time_ms = self.on_validate_time_input()
             
            # Prüfe, ob Daten vorhanden sind
            if 'raw_measurements' not in self.data:
                return
                
            # Setze Time-Offset in flacher Struktur
            self.data['metadata']['time_offset'] = time_ms
            self.data['metadata']['time_normalized'] = time_ms != 0.0
                        
            # Neuberechnung und UI-Update
            success = self.polar_calculator.recalculate_from_raw()

            # Aktualisiere alle Plots, nicht nur das ausgewählte Item
            current_item = self.tree_widget.currentItem()
            if current_item:
                # Wenn ein Item ausgewählt ist, aktualisiere es
                self.on_tree_item_clicked(current_item, 0)
            else:
                # Wenn kein Item ausgewählt ist, aktualisiere mit ersten verfügbaren Daten
                self.update_plot_tree_measurement()
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    def on_validate_ir_window_input(self):
        """Validiert die IR Window Eingabe und erlaubt nur positive Werte"""
        try:
            window_text = self.ir_window_input.text().strip()
            if window_text:
                window_value = float(window_text)
                # Prüfe, ob der Wert positiv ist
                if window_value <= 0:
                    raise ValueError("IR Window muss positiv sein")
                # Prüfe, ob der Wert zu groß ist (max 1000 ms)
                if window_value > 1000:
                    window_value = 1000
                    print("WARNUNG: IR Window auf 1000 ms begrenzt")
                # Runde auf 2 Dezimalstellen
                self.ir_window_input.setText(f"{window_value:.2f}")
                
                # Update Metadaten
                self.data['metadata']['ir_window'] = window_value
                
                # Neuberechnung nur wenn Fensterung aktiviert ist
                if self.ir_window_checkbox.isChecked():
                    self.polar_calculator.recalculate_from_raw()
                    if current_item := self.tree_widget.currentItem():
                        self.on_tree_item_clicked(current_item, 0)
                
                return window_value
        except ValueError:
            self.ir_window_input.setStyleSheet("background-color: #FFE4E1;")
            QTimer.singleShot(500, lambda: self.reset_input_style(self.ir_window_input))
            self.ir_window_input.setText("100.00")
            # Update Metadaten auf Standardwert
            self.data['metadata']['ir_window'] = 100.0
        return 100.0

    def on_ir_window_value_changed(self):
        """Handler für manuelle Änderungen des IR Window Werts"""
        try:
            window_ms = float(self.ir_window_input.text())
            
            # Setze IR Window in flacher Struktur
            self.data['metadata']['ir_window'] = window_ms
            
            # Neuberechnung nur wenn Fensterung aktiviert ist
            if self.ir_window_checkbox.isChecked():
                self.polar_calculator.recalculate_from_raw()
                if current_item := self.tree_widget.currentItem():
                    self.on_tree_item_clicked(current_item, 0)
            
        except Exception as e:
            print(f"Error in IR window value change: {str(e)}")

    def on_ir_window_checkbox_changed(self):
        """Handler für Änderungen der IR Window Checkbox"""
        try:
            # Update Metadaten
            self.data['metadata']['ir_window_enabled'] = self.ir_window_checkbox.isChecked()
            
            # IR Window Eingabefeld bleibt immer aktiviert
            self.ir_window_input.setEnabled(True)
            
            # Neuberechnung und UI-Update
            success = self.polar_calculator.recalculate_from_raw()

            if current_item := self.tree_widget.currentItem():
                self.on_tree_item_clicked(current_item, 0)
                
        except Exception as e:
            print(f"Error in IR window checkbox change: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_filter_checkbox_changed(self):
        """Handler für Filter-Checkbox - aktiviert/deaktiviert Frequenzfilter"""
        try:
            # Update Metadaten
            filter_enabled = self.filter_checkbox.isChecked()
            self.data['metadata']['filter_enabled'] = filter_enabled
            
            # Neuberechnung und UI-Update
            success = self.polar_calculator.recalculate_from_raw()

            if current_item := self.tree_widget.currentItem():
                self.on_tree_item_clicked(current_item, 0)
                
        except Exception as e:
            print(f"Error in filter checkbox change: {str(e)}")
            import traceback
            traceback.print_exc()


# ------ Export -----

    def export_wav(self):
        """UI-Methode für WAV-Export mit Winkelauswahl"""
        try:
            # Dialog für Winkelauswahl
            angles = [str(angle) for angle in range(0, 360, 5)]  # Liste der verfügbaren Winkel
            angle, ok = QInputDialog.getItem(
                self,
                'Winkel auswählen',
                'Wählen Sie den Winkel für den Export:',
                angles,
                0,  # Default-Index (0°)
                False  # Nicht editierbar
            )
            
            if ok and angle:
                # Hole den aktuell ausgewählten Item aus dem TreeWidget
                current_item = self.tree_widget.currentItem()
                if not current_item:
                    QMessageBox.warning(self, "Export", "Bitte wählen Sie zuerst eine Messung aus.")
                    return
                
                # Extrahiere den Lautsprechertyp aus dem Dateinamen
                speaker_type = current_item.text(0).split('.')[0]  # Entferne die Dateiendung
                
                # Generiere den Dateinamen: LautsprecherTyp_WinkelGrad.wav
                suggested_filename = f"{speaker_type}_{angle}deg.wav"
                
                # Dialog für Dateiauswahl mit vorgeschlagenem Namen
                filename, _ = QFileDialog.getSaveFileName(
                    self, 
                    "WAV speichern",
                    suggested_filename,  # Vorgeschlagener Name
                    "WAV Files (*.wav)"
                )
                
                if filename:
                    # Exportiere WAV mit ausgewähltem Winkel
                    success, message = self.polar_export.export_wav(filename, int(angle))
                    if success:
                        QMessageBox.information(self, "Export", message)
                    else:
                        QMessageBox.warning(self, "Export", message)
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_to_lfo(self):
        """Exportiert die Daten im LFO-Format"""
        try:
            # Dialog für Dateinamen
            filename, ok = QInputDialog.getText(
                self,
                'Save file',
                'Save polar data file:'
            )
            
            if ok and filename:
                total_steps = 4 + Polar_Data_Export.EXPORT_PROGRESS_STEPS
                success = False
                message = ""
                with self.progress_manager.start("Exporting polar data", total_steps) as progress:
                    progress.update("Generating balloon data")
                    self.polar_calculator.create_balloon_data()
                    progress.advance()

                    progress.update("Refreshing plots")
                    self.update_plot_polar_balloon()
                    progress.advance()

                    progress.update("Collecting cabinet layout")
                    self.update_cabinet_plot()
                    progress.advance()

                    progress.update("Applying bandwidth limit")
                    self.apply_bandwidth_limit_to_calculated_data(400.0)
                    progress.advance()
                
                success, message = self.polar_export.export_to_lfo(
                    filename,
                    self.speakers,
                    progress=progress
                )
                
                if success:
                    QMessageBox.information(self, "Export", message)
                    if self.manage_speaker_window:
                        self.manage_speaker_window.load_polar_data()
                else:
                    QMessageBox.warning(self, "Export", message)
                    
        except Exception as e:

            import traceback
            traceback.print_exc()

    def apply_bandwidth_limit_to_calculated_data(self, max_freq):
        """Wendet Bandbegrenzung auf alle calculated_data UND interpolated_data an"""
        try:
            # Bandbegrenzung auf calculated_data
            if 'calculated_data' in self.data:
                for filename, data in self.data['calculated_data'].items():
                    if 'freq' in data and 'magnitude' in data and 'phase' in data:
                        freq = np.array(data['freq'])
                        magnitude = np.array(data['magnitude'])
                        phase = np.array(data['phase'])
                        
                        # Wende Bandbegrenzung an
                        freq_limited, mag_limited, phase_limited = self.polar_export.apply_bandwidth_limit(
                            freq, magnitude, phase, max_freq
                        )
                        
                        # Aktualisiere die Daten als NumPy-Arrays, damit der Datentyp konsistent bleibt
                        self.data['calculated_data'][filename]['freq'] = np.asarray(freq_limited, dtype=float)
                        self.data['calculated_data'][filename]['magnitude'] = np.asarray(mag_limited, dtype=float)
                        self.data['calculated_data'][filename]['phase'] = np.asarray(phase_limited, dtype=float)
            
            # WICHTIG: Bandbegrenzung auch auf interpolated_data (wird für LFO Export verwendet!)
            if 'interpolated_data' in self.data:
                for angle, data in self.data['interpolated_data'].items():
                    if 'freq' in data and 'magnitude' in data and 'phase' in data:
                        freq = np.array(data['freq'])
                        magnitude = np.array(data['magnitude'])
                        phase = np.array(data['phase'])
                        
                        # Wende Bandbegrenzung an
                        freq_limited, mag_limited, phase_limited = self.polar_export.apply_bandwidth_limit(
                            freq, magnitude, phase, max_freq
                        )
                        
                        # Aktualisiere die Daten als NumPy-Arrays, damit der Datentyp konsistent bleibt
                        self.data['interpolated_data'][angle]['freq'] = np.asarray(freq_limited, dtype=float)
                        self.data['interpolated_data'][angle]['magnitude'] = np.asarray(mag_limited, dtype=float)
                        self.data['interpolated_data'][angle]['phase'] = np.asarray(phase_limited, dtype=float)
            
        except Exception as e:
            print(f"ERROR: Failed to apply bandwidth limit: {e}")
            import traceback
            traceback.print_exc()

    def on_configuration_changed(self, index):
        """Handler für Änderungen der Position (Stack/Flown)"""
        try:
            # Hole den Sender (QComboBox)
            sender = self.sender()
            
            # Finde den entsprechenden Speaker
            for i, speaker in enumerate(self.speakers):
                if speaker['configuration'] == sender:
                    # Prüfe, ob Stack oder Flown ausgewählt wurde
                    is_flown = (sender.currentText() == "Flown")
                    
                    # Aktiviere/Deaktiviere Angle Point basierend auf Position
                    speaker['angle_point'].setEnabled(is_flown)
                    
                    # Stack-Layout Combo aktivieren/deaktivieren
                    stack_layout_combo = speaker.get('stack_layout')
                    if stack_layout_combo:
                        if is_flown:
                            stack_layout_combo.blockSignals(True)
                            stack_layout_combo.setCurrentText("Beside")
                            stack_layout_combo.blockSignals(False)
                            stack_layout_combo.setEnabled(False)
                        else:
                            stack_layout_combo.setEnabled(True)
                    
                    # Setze Angle Point auf "None" wenn Stack
                    if not is_flown:
                        speaker['angle_point'].setCurrentText("None")
                        
                        # Setze #Angles auf 1 wenn Stack
                        speaker['num_angles'].setText("1")
                        
                        # Aktualisiere die Winkel-Items
                        self.update_angle_items(speaker['item'], speaker['num_angles'])
                        
                        # Deaktiviere das Eingabefeld für #Angles
                        speaker['num_angles'].setEnabled(False)
                        
                        # Setze den Winkel auf 0 für Stack
                        if speaker['angle_items'] and len(speaker['angle_items']) > 0:
                            speaker['angle_items'][0].setText("0")
                            speaker['angle_items'][0].setEnabled(False)
                    else:
                        # Aktiviere das Eingabefeld für #Angles bei Flown
                        speaker['num_angles'].setEnabled(True)
                        
                        # Aktiviere die Winkel-Eingabefelder bei Flown
                        for angle_item in speaker['angle_items']:
                            angle_item.setEnabled(True)
                    
                    break
            
            # Aktualisiere Cabinet-Plot
            self.update_cabinet_plot()
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Position: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_angle_items(self, speaker_item, num_angles_editor):
        """Aktualisiert die Winkel-Items basierend auf der eingegebenen Anzahl"""
        try:
            # Hole die Anzahl der Winkel
            num_angles = int(num_angles_editor.text())
            if num_angles < 1:
                raise ValueError("Mindestens ein Winkel erforderlich")
            
            # Finde den zugehörigen Speaker
            for speaker_data in self.speakers:
                if speaker_data['item'] == speaker_item:
                    angles_folder = speaker_data['angles_folder']
                    
                    # Lösche alle vorhandenen Winkel-Items
                    while angles_folder.childCount() > 0:
                        angles_folder.removeChild(angles_folder.child(0))
                    
                    # Leere die Liste der Winkel-Items
                    speaker_data['angle_items'] = []
                    
                    # Erstelle neue Winkel-Items
                    for i in range(num_angles):
                        angle_item = QTreeWidgetItem(angles_folder, [f"Angle {i+1}"])
                        angle_editor = QLineEdit()
                        angle_editor.setFixedWidth(100)
                        
                        # Setze einen leeren Wert, damit der Benutzer ihn manuell eingeben kann
                        angle_editor.setText("")
                        
                        angle_editor.textChanged.connect(self.update_cabinet_plot)
                        angle_editor.returnPressed.connect(self.update_cabinet_plot)
                        self.speaker_tree.setItemWidget(angle_item, 1, angle_editor)
                        speaker_data['angle_items'].append(angle_editor)
                    
                    # Expandiere den Winkel-Ordner
                    angles_folder.setExpanded(True)
                    
                    # Aktualisiere Cabinet-Plot
                    self.update_cabinet_plot()
                    
                    break
            
        except ValueError as e:
            QMessageBox.warning(self, "Fehler", str(e))

    def on_angle_point_changed(self, index):
        """Handler für Änderungen des Angle Points"""
        try:
            # Hole den Sender (QComboBox)
            sender = self.sender()
            
            # Finde den entsprechenden Speaker
            for i, speaker in enumerate(self.speakers):
                if speaker['angle_point'] == sender:
                    # Aktualisiere Cabinet-Plot
                    self.update_cabinet_plot()
                    break
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren des Angle Points: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_speaker(self):
        """Fügt einen neuen Lautsprecher zur Liste hinzu"""
        try:
            # Bestimme die Nummer des neuen Lautsprechers
            speaker_number = len(self.speakers) + 1
            
            # Erstelle ein neues TreeWidget-Item für den Lautsprecher
            speaker_item = QTreeWidgetItem(self.speaker_tree, [f"Speaker {speaker_number}"])
            
            # Position (Stack/Flown)
            configuration_item = QTreeWidgetItem(speaker_item, ["Configuration"])
            configuration_combo = QComboBox()
            configuration_combo.addItems(["Flown", "Stack"])
            configuration_combo.setFixedWidth(100)
            configuration_combo.currentIndexChanged.connect(self.on_configuration_changed)
            self.speaker_tree.setItemWidget(configuration_item, 1, configuration_combo)
            
            # Bestimme, ob Flown oder Stack ausgewählt ist
            is_flown = (configuration_combo.currentText() == "Flown")

            # Stack-Layout Auswahl (nur für Stack aktiv)
            stack_layout_item = QTreeWidgetItem(speaker_item, ["Stack placement"])
            stack_layout_combo = QComboBox()
            stack_layout_combo.addItems(["Beside", "On top"])
            stack_layout_combo.setFixedWidth(100)
            stack_layout_combo.currentIndexChanged.connect(self.update_cabinet_plot)
            stack_layout_combo.setEnabled(not is_flown)
            self.speaker_tree.setItemWidget(stack_layout_item, 1, stack_layout_combo)
            
            # Cardio Checkbox
            cardio_item = QTreeWidgetItem(speaker_item, ["Cardio"])
            cardio_checkbox = QCheckBox()
            cardio_checkbox.setChecked(False)  # Default-Wert
            cardio_checkbox.stateChanged.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(cardio_item, 1, cardio_checkbox)
            
            # Width
            width_item = QTreeWidgetItem(speaker_item, ["Width (m)"])
            width_editor = QLineEdit()
            width_editor.setFixedWidth(100)
            width_editor.setText("1.35")  # Default-Wert
            width_editor.textChanged.connect(self.update_cabinet_plot)
            width_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(width_item, 1, width_editor)
            
            # Depth
            depth_item = QTreeWidgetItem(speaker_item, ["Depth (m)"])
            depth_editor = QLineEdit()
            depth_editor.setFixedWidth(100)
            depth_editor.setText("0.7")  # Default-Wert
            depth_editor.textChanged.connect(self.update_cabinet_plot)
            depth_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(depth_item, 1, depth_editor)
            
            # Height
            front_height_item = QTreeWidgetItem(speaker_item, ["Front Height (m)"])
            front_height_editor = QLineEdit()
            front_height_editor.setFixedWidth(100)
            front_height_editor.setText("0.4")  # Default-Wert
            front_height_editor.textChanged.connect(self.update_cabinet_plot)
            front_height_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(front_height_item, 1, front_height_editor)

            back_height_item = QTreeWidgetItem(speaker_item, ["Back Height (m)"])
            back_height_editor = QLineEdit()
            back_height_editor.setFixedWidth(100)
            back_height_editor.setText("0.4")  # Default-Wert
            back_height_editor.textChanged.connect(self.update_cabinet_plot)
            back_height_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(back_height_item, 1, back_height_editor)
            
            # Angle Point
            angle_point_item = QTreeWidgetItem(speaker_item, ["Angle Point"])
            angle_point_combo = QComboBox()
            angle_point_combo.addItems(["None", "Front", "Back", "Center"])
            angle_point_combo.setFixedWidth(100)
            angle_point_combo.setCurrentText("None")  # Default für Stack
            angle_point_combo.setEnabled(is_flown)  # Aktiviert für Flown, deaktiviert für Stack
            angle_point_combo.currentIndexChanged.connect(self.on_angle_point_changed)
            self.speaker_tree.setItemWidget(angle_point_item, 1, angle_point_combo)
            
            # # Angles (Anzahl der Winkel)
            num_angles_item = QTreeWidgetItem(speaker_item, ["# Angles"])
            num_angles_editor = QLineEdit()
            num_angles_editor.setFixedWidth(100)
            num_angles_editor.setText("5")  # Default-Wert
            num_angles_editor.setEnabled(is_flown)  # Aktiviert für Flown, deaktiviert für Stack
            num_angles_editor.returnPressed.connect(lambda: self.update_angle_items(speaker_item, num_angles_editor))
            self.speaker_tree.setItemWidget(num_angles_item, 1, num_angles_editor)
            
            # Winkel-Unterordner
            angles_folder = QTreeWidgetItem(speaker_item, ["Winkel"])
            
            # Speichere die Referenzen auf die Widgets
            speaker_data = {
                'item': speaker_item,
                'configuration': configuration_combo,
                'cardio': cardio_checkbox,
                'width': width_editor,
                'depth': depth_editor,
                'front_height': front_height_editor,
                'back_height': back_height_editor,
                'angle_point': angle_point_combo,
                'num_angles': num_angles_editor,
                'angles_folder': angles_folder,
                'angle_items': [],  # Leere Liste für Winkel-Items
                'stack_layout': stack_layout_combo
            }
            
            self.speakers.append(speaker_data)
            
            # Expandiere den Winkel-Ordner, damit der Benutzer die Winkel sehen kann
            angles_folder.setExpanded(True)
            
            # Erstelle initial die Winkel-Items basierend auf dem Default-Wert
            self.update_angle_items(speaker_item, num_angles_editor)
            
            # Wenn Flown ausgewählt ist, setze den Winkelpunkt auf "Front"
            if is_flown:
                angle_point_combo.setCurrentText("Front")
            
            # Aktualisiere den Cabinet-Plot
            self.update_cabinet_plot()
            
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Fehler beim Hinzufügen eines Lautsprechers: {str(e)}")

    def show_speaker_context_menu(self, configuration):
        """Zeigt das Kontextmenü für Lautsprecher im TreeWidget an"""
        try:
            # Hole das Item unter dem Mauszeiger
            item = self.speaker_tree.itemAt(configuration)
            if not item:
                return
            
            # Finde das übergeordnete Speaker-Item
            speaker_item = item
            while speaker_item.parent():
                speaker_item = speaker_item.parent()
            
            # Prüfe, ob es sich um einen Lautsprecher handelt
            is_speaker = False
            for speaker_data in self.speakers:
                if speaker_data['item'] == speaker_item:
                    is_speaker = True
                    break
            
            if not is_speaker:
                return
            
            # Erstelle das Kontextmenü
            context_menu = QMenu(self)
            
            # Aktionen hinzufügen
            delete_action = context_menu.addAction("Delete speaker")
            duplicate_action = context_menu.addAction("Duplicate speaker")
            
            # Zeige das Menü an
            action = context_menu.exec_(self.speaker_tree.mapToGlobal(configuration))
            
            # Verarbeite die ausgewählte Aktion
            if action == delete_action:
                self.delete_speaker(speaker_item)
            elif action == duplicate_action:
                self.duplicate_speaker(speaker_item)
            
        except Exception as e:
            print(f"Fehler beim Anzeigen des Kontextmenüs: {str(e)}")

    def delete_speaker(self, speaker_item):
        """Löscht einen Lautsprecher"""
        try:
            # Finde den Index des Lautsprechers
            speaker_index = -1
            for i, speaker_data in enumerate(self.speakers):
                if speaker_data['item'] == speaker_item:
                    speaker_index = i
                    break
            
            if speaker_index == -1:
                return
            
            # Entferne den Lautsprecher aus der Liste
            self.speakers.pop(speaker_index)
            
            # Entferne das Item aus dem TreeWidget
            self.speaker_tree.takeTopLevelItem(self.speaker_tree.indexOfTopLevelItem(speaker_item))
            
            # Aktualisiere den Cabinet-Plot
            self.update_cabinet_plot()
            
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Fehler beim Löschen eines Lautsprechers: {str(e)}")

    def duplicate_speaker(self, speaker_item):
        """Dupliziert einen Lautsprecher"""
        try:
            # Finde den zu duplizierenden Lautsprecher
            source_speaker = None
            for speaker_data in self.speakers:
                if speaker_data['item'] == speaker_item:
                    source_speaker = speaker_data
                    break
            
            if not source_speaker:
                return
            
            # Erstelle einen neuen Lautsprecher
            new_speaker_number = len(self.speakers) + 1
            new_speaker_item = QTreeWidgetItem(self.speaker_tree, [f"Speaker {new_speaker_number}"])
            
            # Position (Stack/Flown)
            configuration_item = QTreeWidgetItem(new_speaker_item, ["Configuration"])
            configuration_combo = QComboBox()
            configuration_combo.addItems(["Stack", "Flown"])
            configuration_combo.setFixedWidth(100)
            configuration_combo.setCurrentText(source_speaker['configuration'].currentText())
            configuration_combo.currentIndexChanged.connect(self.on_configuration_changed)
            self.speaker_tree.setItemWidget(configuration_item, 1, configuration_combo)
            
            # Stack-Layout
            stack_layout_item = QTreeWidgetItem(new_speaker_item, ["Stack placement"])
            stack_layout_combo = QComboBox()
            stack_layout_combo.addItems(["Beside", "On top"])
            stack_layout_combo.setFixedWidth(100)
            stack_layout_combo.setCurrentText(source_speaker['stack_layout'].currentText())
            stack_layout_combo.currentIndexChanged.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(stack_layout_item, 1, stack_layout_combo)
            stack_layout_combo.setEnabled(configuration_combo.currentText() == "Stack")
            
            # Cardio Checkbox
            cardio_item = QTreeWidgetItem(new_speaker_item, ["Cardio"])
            cardio_checkbox = QCheckBox()
            cardio_checkbox.setChecked(source_speaker['cardio'].isChecked())
            cardio_checkbox.stateChanged.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(cardio_item, 1, cardio_checkbox)
            
            # Width
            width_item = QTreeWidgetItem(new_speaker_item, ["Width (m)"])
            width_editor = QLineEdit()
            width_editor.setFixedWidth(100)
            width_editor.setText(source_speaker['width'].text())
            width_editor.textChanged.connect(self.update_cabinet_plot)
            width_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(width_item, 1, width_editor)
            
            # Depth
            depth_item = QTreeWidgetItem(new_speaker_item, ["Depth (m)"])
            depth_editor = QLineEdit()
            depth_editor.setFixedWidth(100)
            depth_editor.setText(source_speaker['depth'].text())
            depth_editor.textChanged.connect(self.update_cabinet_plot)
            depth_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(depth_item, 1, depth_editor)
            
            # front Height
            front_height_item = QTreeWidgetItem(new_speaker_item, ["Front Height (m)"])
            front_height_editor = QLineEdit()
            front_height_editor.setFixedWidth(100)
            front_height_editor.setText(source_speaker['front_height'].text())
            front_height_editor.textChanged.connect(self.update_cabinet_plot)
            front_height_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(front_height_item, 1, front_height_editor)

            # back Height
            back_height_item = QTreeWidgetItem(new_speaker_item, ["Back Height (m)"])
            back_height_editor = QLineEdit()
            back_height_editor.setFixedWidth(100)
            back_height_editor.setText(source_speaker['back_height'].text())
            back_height_editor.textChanged.connect(self.update_cabinet_plot)
            back_height_editor.returnPressed.connect(self.update_cabinet_plot)
            self.speaker_tree.setItemWidget(back_height_item, 1, back_height_editor)
            
            # Angle Point
            angle_point_item = QTreeWidgetItem(new_speaker_item, ["Angle Point"])
            angle_point_combo = QComboBox()
            angle_point_combo.addItems(["None", "Front", "Back", "Center"])
            angle_point_combo.setFixedWidth(100)
            angle_point_combo.setCurrentText(source_speaker['angle_point'].currentText())
            angle_point_combo.setEnabled(configuration_combo.currentText() == "Flown")
            angle_point_combo.currentIndexChanged.connect(self.on_angle_point_changed)
            self.speaker_tree.setItemWidget(angle_point_item, 1, angle_point_combo)
            
            # # Angles (Anzahl der Winkel)
            num_angles_item = QTreeWidgetItem(new_speaker_item, ["# Angles"])
            num_angles_editor = QLineEdit()
            num_angles_editor.setFixedWidth(100)
            num_angles_editor.setText(source_speaker['num_angles'].text())
            num_angles_editor.returnPressed.connect(lambda: self.update_angle_items(new_speaker_item, num_angles_editor))
            self.speaker_tree.setItemWidget(num_angles_item, 1, num_angles_editor)
            
            # Winkel-Unterordner
            angles_folder = QTreeWidgetItem(new_speaker_item, ["Winkel"])
            
            # Speichere die Referenzen auf die Widgets
            new_speaker_data = {
                'item': new_speaker_item,
                'configuration': configuration_combo,
                'cardio': cardio_checkbox,
                'width': width_editor,
                'depth': depth_editor,
                'front_height': front_height_editor,
                'back_height': back_height_editor,
                'angle_point': angle_point_combo,
                'num_angles': num_angles_editor,
                'angles_folder': angles_folder,
                'angle_items': [],  # Leere Liste für Winkel-Items
                'stack_layout': stack_layout_combo
            }
            
            self.speakers.append(new_speaker_data)
            
            # Expandiere den Winkel-Ordner, damit der Benutzer die Winkel sehen kann
            angles_folder.setExpanded(True)
            
            # Erstelle die Winkel-Items basierend auf dem Quell-Lautsprecher
            self.update_angle_items(new_speaker_item, num_angles_editor)
            
            # Kopiere die Winkelwerte vom Quell-Lautsprecher
            for i, source_angle_editor in enumerate(source_speaker['angle_items']):
                if i < len(new_speaker_data['angle_items']):
                    new_speaker_data['angle_items'][i].setText(source_angle_editor.text())
            
            # Aktualisiere den Cabinet-Plot
            self.update_cabinet_plot()
            
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Fehler beim Duplizieren eines Lautsprechers: {str(e)}")

    def replace_phase_for_selected_files(self, selected_items):
        """Ersetzt die Phase-Daten der ausgewählten Messungen mittels einer TXT-Datei."""
        try:
            filenames = [item.text(0) for item in selected_items if item.parent() is not None]
            if not filenames:
                QMessageBox.warning(self, "Phase ersetzen", "Bitte wählen Sie mindestens eine Messung aus.")
                return

            phase_path, _ = QFileDialog.getOpenFileName(
                self,
                "Phase-Daten auswählen",
                "",
                "TXT Files (*.txt);;All Files (*)"
            )
            if not phase_path:
                return

            success, message, updated_items = self.polar_calculator.replace_phase_data(filenames, phase_path)

            if success:
                self.polar_calculator.recalculate_from_raw()
                self.update_plot_tree_measurement()
                QMessageBox.information(self, "Phase ersetzt", message)
            else:
                QMessageBox.warning(self, "Phase ersetzen", message)

        except Exception as exc:
            QMessageBox.critical(self, "Phase ersetzen", f"Fehler beim Ersetzen der Phase: {exc}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = TransferFunctionViewer()
    viewer.show()
    sys.exit(app.exec_())



# ---------