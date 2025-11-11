from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QDoubleValidator, QIntValidator
import numpy as np

class UiSettings(QtWidgets.QWidget):
    settings_changed = QtCore.pyqtSignal(object)  # Signal für Einstellungsänderungen

    def __init__(self, main_window, settings, data):
        super().__init__()
        self.main_window = main_window
        self.settings = settings  # Verwenden Sie die übergebene settings-Instanz
        self.data = data
        self.setup_ui()
        self.setup_connections()  # Stellen Sie sicher, dass dies aufgerufen wird
        self.hide()  # Initially hide the settings window

    def setup_ui(self):
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 300, 600)

        layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget()

        # Tab 1: Settings
        self.tab_settings = QtWidgets.QWidget()
        self.setup_tab_settings()
        self.tab_widget.addTab(self.tab_settings, "Settings")

        # Tab 2: Calculation
        self.tab_calculation = QtWidgets.QWidget()
        self.setup_tab_calculation()
        self.tab_widget.addTab(self.tab_calculation, "Calculation")

        # Tab 3: Mapping
        self.tab_mapping = QtWidgets.QWidget()
        self.setup_tab_mapping()
        self.tab_widget.addTab(self.tab_mapping, "Mapping")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def setup_tab_settings(self):
        layout = QtWidgets.QVBoxLayout(self.tab_settings)
        layout.setSpacing(10)
    
        # Gemeinsame Breite für alle Eingabefelder
        INPUT_WIDTH = 100

        # Impulse Window
        impulse_window_layout = QtWidgets.QHBoxLayout()
        impulse_window_label = QtWidgets.QLabel("Impulse window")

        # Impulse plot height
        impulse_plot_height_layout = QtWidgets.QHBoxLayout()    
        impulse_plot_height_label = QtWidgets.QLabel("Impulse plot height")
        self.impulse_plot_height = QLineEdit()
        self.impulse_plot_height.setValidator(QIntValidator(50, 1000))
        self.impulse_plot_height.setFixedWidth(INPUT_WIDTH)
        impulse_plot_height_layout.addWidget(impulse_plot_height_label)
        impulse_plot_height_layout.addStretch()
        impulse_plot_height_layout.addWidget(self.impulse_plot_height)
        layout.addLayout(impulse_plot_height_layout)
        
        # Impulse SPL range
        impulse_min_spl_layout = QtWidgets.QHBoxLayout()
        impulse_min_spl_label = QtWidgets.QLabel("Impulse SPL range")
        self.impulse_min_spl = QLineEdit()
        self.impulse_min_spl.setValidator(QIntValidator(-100, -6))
        self.impulse_min_spl.setFixedWidth(INPUT_WIDTH)
        impulse_min_spl_layout.addWidget(impulse_min_spl_label)
        impulse_min_spl_layout.addStretch()
        impulse_min_spl_layout.addWidget(self.impulse_min_spl)
        layout.addLayout(impulse_min_spl_layout)
        
        # Size measurement Point
        size_meausurement_layout = QtWidgets.QHBoxLayout()
        size_meausurement_label = QtWidgets.QLabel("Size measurement point")
        self.size_of_measurement_point = QLineEdit()
        self.size_of_measurement_point.setValidator(QDoubleValidator(0.1, 20.0, 2))
        self.size_of_measurement_point.setFixedWidth(INPUT_WIDTH)
        size_meausurement_layout.addWidget(size_meausurement_label)
        size_meausurement_layout.addStretch()
        size_meausurement_layout.addWidget(self.size_of_measurement_point)
        layout.addLayout(size_meausurement_layout)

        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        layout.addStretch()


    def setup_tab_mapping(self):
        self.mapping_layout = QtWidgets.QVBoxLayout(self.tab_mapping)
        self.mapping_layout.setSpacing(10)

        # Gemeinsame Breite für alle ComboBoxen und Eingabefelder
        INPUT_WIDTH = 120

        # Max SPL Plot
        max_spl_layout = QtWidgets.QHBoxLayout()
        max_spl_label = QtWidgets.QLabel("Max. SPL Plot")
        self.max_spl = QLineEdit()
        self.max_spl.setValidator(QIntValidator(0, 150))
        self.max_spl.setFixedWidth(INPUT_WIDTH)
        max_spl_layout.addWidget(max_spl_label)
        max_spl_layout.addStretch()
        max_spl_layout.addWidget(self.max_spl)
        self.mapping_layout.addLayout(max_spl_layout)

        # Min SPL Plot
        min_spl_layout = QtWidgets.QHBoxLayout()
        min_spl_label = QtWidgets.QLabel("Min. SPL Plot")
        self.min_spl = QLineEdit()
        self.min_spl.setValidator(QIntValidator(0, 150))
        self.min_spl.setFixedWidth(INPUT_WIDTH)
        min_spl_layout.addWidget(min_spl_label)
        min_spl_layout.addStretch()
        min_spl_layout.addWidget(self.min_spl)
        self.mapping_layout.addLayout(min_spl_layout)

        # dB per color step
        db_step_layout = QtWidgets.QHBoxLayout()
        db_step_label = QtWidgets.QLabel("dB per color step")
        self.db_step = QLineEdit()
        self.db_step.setValidator(QIntValidator(1, 20))
        self.db_step.setFixedWidth(INPUT_WIDTH)
        db_step_layout.addWidget(db_step_label)
        db_step_layout.addStretch()
        db_step_layout.addWidget(self.db_step)
        self.mapping_layout.addLayout(db_step_layout)

        # Trennlinie
        self.mapping_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Colorization mode
        colorization_layout = QtWidgets.QHBoxLayout()
        colorization_label = QtWidgets.QLabel("SPL mode")
        self.colorization_mode = QtWidgets.QComboBox()
        self.colorization_mode.addItems(["Color step", "Gradient"])
        self.colorization_mode.setFixedWidth(INPUT_WIDTH)
        colorization_layout.addWidget(colorization_label)
        colorization_layout.addStretch()
        colorization_layout.addWidget(self.colorization_mode)
        self.mapping_layout.addLayout(colorization_layout)

        # Trennlinie
        self.mapping_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Position Plot Length
        position_plot_length_layout = QtWidgets.QHBoxLayout()
        position_plot_length_label = QtWidgets.QLabel("Length plot position")
        self.position_plot_length = QLineEdit()
        self.position_plot_length.setValidator(QIntValidator(-100, 100))
        self.position_plot_length.setFixedWidth(INPUT_WIDTH)
        position_plot_length_layout.addWidget(position_plot_length_label)
        position_plot_length_layout.addStretch()
        position_plot_length_layout.addWidget(self.position_plot_length)
        self.mapping_layout.addLayout(position_plot_length_layout)

        # Position Plot Width
        position_plot_width_layout = QtWidgets.QHBoxLayout()
        position_plot_width_label = QtWidgets.QLabel("Width plot position")
        self.position_plot_width = QLineEdit()
        self.position_plot_width.setValidator(QIntValidator(-100, 100))
        self.position_plot_width.setFixedWidth(INPUT_WIDTH)
        position_plot_width_layout.addWidget(position_plot_width_label)
        position_plot_width_layout.addStretch()
        position_plot_width_layout.addWidget(self.position_plot_width)
        self.mapping_layout.addLayout(position_plot_width_layout)

        self.mapping_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Titel für Polar Frequencies
        polar_title = QtWidgets.QLabel("Polar frequencies")
        self.mapping_layout.addWidget(polar_title)

        # Frequenzauswahlwidget
        self.freq_selection_widget = QtWidgets.QWidget()
        freq_selection_layout = QtWidgets.QVBoxLayout(self.freq_selection_widget)
        freq_selection_layout.setSpacing(4)
        freq_selection_layout.setContentsMargins(0, 5, 0, 0)

        # Erstelle ComboBoxen für verschiedene Frequenzen
        self.freq_boxes = []
        frequencies = ['31.5 Hz', '40 Hz', '50 Hz', '63 Hz']
        colors = ['red', 'yellow', 'green', 'cyan']

        for freq, color in zip(frequencies, colors):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            
            # Farbbox
            color_box = QtWidgets.QLabel()
            color_box.setFixedSize(15, 15)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")
            
            # ComboBox
            combo = QtWidgets.QComboBox()
            combo.addItems(['25 Hz', '31.5 Hz', '40 Hz', '50 Hz', '63 Hz', '80 Hz'])
            combo.setCurrentText(freq)
            combo.setFixedWidth(INPUT_WIDTH)
            
            # Layout von links nach rechts mit Stretch vor der ComboBox
            row_layout.addWidget(color_box)
            row_layout.addStretch()  # Stretch zwischen Farbbox und ComboBox
            row_layout.addWidget(combo)  # ComboBox am Ende
            
            freq_selection_layout.addWidget(row_widget)
            self.freq_boxes.append(combo)

        self.mapping_layout.addWidget(self.freq_selection_widget)
        
        # Zum Schluss
        self.mapping_layout.addStretch()

    def setup_tab_calculation(self):
        layout = QtWidgets.QVBoxLayout(self.tab_calculation)
        layout.setSpacing(10)

        INPUT_WIDTH = 120

        # Upper und Lower Frequency bandwidth
        freq_bandwidth_upper_layout = QtWidgets.QHBoxLayout()
        freq_bandwidth_lower_layout = QtWidgets.QHBoxLayout()

        available_frequencies = [
            400, 375, 355, 335, 315,
            300, 280, 265, 250, 236, 224, 212, 200, 190, 180, 170, 160,
            150, 140, 132, 125, 118, 112, 106, 100, 95, 90, 85, 80,
            75, 71, 67, 63, 60, 56, 53, 50, 47, 45, 42, 40,
            37, 35, 33, 31, 30, 28, 26, 25, 24, 22, 21, 20, 15
        ]
        frequency_strings = [f"{freq} Hz" for freq in available_frequencies]

        self.freq_bandwidth_lower = QtWidgets.QComboBox()
        self.freq_bandwidth_upper = QtWidgets.QComboBox()

        self.freq_bandwidth_lower.blockSignals(True)
        self.freq_bandwidth_upper.blockSignals(True)

        self.freq_bandwidth_lower.addItems(frequency_strings)
        self.freq_bandwidth_upper.addItems(frequency_strings)

        self.freq_bandwidth_lower.setCurrentText(f"{self.settings.lower_calculate_frequency} Hz")
        self.freq_bandwidth_upper.setCurrentText(f"{self.settings.upper_calculate_frequency} Hz")

        self.freq_bandwidth_lower.blockSignals(False)
        self.freq_bandwidth_upper.blockSignals(False)

        self.freq_bandwidth_upper.currentIndexChanged.connect(self.validate_frequency_range)
        self.freq_bandwidth_lower.currentIndexChanged.connect(self.validate_frequency_range)

        freq_bandwidth_upper_label = QtWidgets.QLabel("Upper frequency bandwidth")
        self.freq_bandwidth_upper.setFixedWidth(INPUT_WIDTH)
        freq_bandwidth_upper_layout.addWidget(freq_bandwidth_upper_label)
        freq_bandwidth_upper_layout.addStretch()
        freq_bandwidth_upper_layout.addWidget(self.freq_bandwidth_upper)

        freq_bandwidth_lower_label = QtWidgets.QLabel("Lower frequency bandwidth")
        self.freq_bandwidth_lower.setFixedWidth(INPUT_WIDTH)
        freq_bandwidth_lower_layout.addWidget(freq_bandwidth_lower_label)
        freq_bandwidth_lower_layout.addStretch()
        freq_bandwidth_lower_layout.addWidget(self.freq_bandwidth_lower)

        title_style = "font-size: 11px; font-weight: bold;"

        freq_range_title = QtWidgets.QLabel("Frequency range: Axis & Soundfield")
        freq_range_title.setStyleSheet(title_style)
        layout.addWidget(freq_range_title)
        freq_range_title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        layout.addLayout(freq_bandwidth_upper_layout)
        layout.addLayout(freq_bandwidth_lower_layout)

        layout.addSpacing(18)

        resolution_title = QtWidgets.QLabel("Resolution Soundfield")
        resolution_title.setStyleSheet(title_style)
        resolution_title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(resolution_title)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Resolution
        resolution_layout = QtWidgets.QHBoxLayout()
        resolution_label = QtWidgets.QLabel("Resolution")
        self.resolution = QLineEdit()
        self.resolution.setValidator(QDoubleValidator(0.1, 3.0, 2))
        self.resolution.setFixedWidth(INPUT_WIDTH)
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addStretch()
        resolution_layout.addWidget(self.resolution)
        layout.addLayout(resolution_layout)

        layout.addSpacing(18)

        update_pressure_title = QtWidgets.QLabel("Update automatically pressure")
        update_pressure_title.setStyleSheet(title_style)
        update_pressure_title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(update_pressure_title)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        update_pressure_checks_layout = QtWidgets.QVBoxLayout()
        update_pressure_checks_layout.setSpacing(4)
        self.update_pressure_soundfield = QtWidgets.QCheckBox("Soundfield")
        self.update_pressure_axisplot = QtWidgets.QCheckBox("Axis Plot")
        self.update_pressure_polarplot = QtWidgets.QCheckBox("Polar Plot")
        self.update_pressure_impulse = QtWidgets.QCheckBox("Impulse Plot")
        update_pressure_checks_layout.addWidget(self.update_pressure_soundfield)
        update_pressure_checks_layout.addWidget(self.update_pressure_axisplot)
        update_pressure_checks_layout.addWidget(self.update_pressure_polarplot)
        update_pressure_checks_layout.addWidget(self.update_pressure_impulse)
        layout.addLayout(update_pressure_checks_layout)

        layout.addSpacing(18)

        calculation_method_title = QtWidgets.QLabel("Calculation method")
        calculation_method_title.setStyleSheet(title_style)
        calculation_method_title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(calculation_method_title)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Plot Optionen Hauptbereich
        plot_option_layout = QtWidgets.QGridLayout()
        plot_option_layout.setHorizontalSpacing(20)
        plot_option_layout.setVerticalSpacing(8)

        plot_option_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        plot_option_layout.addWidget(QtWidgets.QLabel("Superposition"), 0, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(QtWidgets.QLabel("FEM Analyse"), 0, 2, alignment=QtCore.Qt.AlignCenter)

        self.spl_plot_superposition = QtWidgets.QCheckBox()
        self.spl_plot_fem = QtWidgets.QCheckBox()
        plot_option_layout.addWidget(QtWidgets.QLabel("SPL Plot"), 1, 0)
        plot_option_layout.addWidget(self.spl_plot_superposition, 1, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(self.spl_plot_fem, 1, 2, alignment=QtCore.Qt.AlignCenter)

        self.xaxis_plot_superposition = QtWidgets.QCheckBox()
        self.xaxis_plot_fem = QtWidgets.QCheckBox()
        plot_option_layout.addWidget(QtWidgets.QLabel("X-Axis Plot"), 2, 0)
        plot_option_layout.addWidget(self.xaxis_plot_superposition, 2, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(self.xaxis_plot_fem, 2, 2, alignment=QtCore.Qt.AlignCenter)

        self.yaxis_plot_superposition = QtWidgets.QCheckBox()
        self.yaxis_plot_fem = QtWidgets.QCheckBox()
        plot_option_layout.addWidget(QtWidgets.QLabel("Y-Axis Plot"), 3, 0)
        plot_option_layout.addWidget(self.yaxis_plot_superposition, 3, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(self.yaxis_plot_fem, 3, 2, alignment=QtCore.Qt.AlignCenter)

        self.polar_plot_superposition = QtWidgets.QCheckBox()
        self.polar_plot_fem = QtWidgets.QCheckBox()
        plot_option_layout.addWidget(QtWidgets.QLabel("Polar Plot"), 4, 0)
        plot_option_layout.addWidget(self.polar_plot_superposition, 4, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(self.polar_plot_fem, 4, 2, alignment=QtCore.Qt.AlignCenter)

        self._spl_calc_group = QtWidgets.QButtonGroup(self)
        self._spl_calc_group.setExclusive(True)
        self._spl_calc_group.addButton(self.spl_plot_superposition)
        self._spl_calc_group.addButton(self.spl_plot_fem)

        self._xaxis_calc_group = QtWidgets.QButtonGroup(self)
        self._xaxis_calc_group.setExclusive(True)
        self._xaxis_calc_group.addButton(self.xaxis_plot_superposition)
        self._xaxis_calc_group.addButton(self.xaxis_plot_fem)

        self._yaxis_calc_group = QtWidgets.QButtonGroup(self)
        self._yaxis_calc_group.setExclusive(True)
        self._yaxis_calc_group.addButton(self.yaxis_plot_superposition)
        self._yaxis_calc_group.addButton(self.yaxis_plot_fem)

        self._polar_calc_group = QtWidgets.QButtonGroup(self)
        self._polar_calc_group.setExclusive(True)
        self._polar_calc_group.addButton(self.polar_plot_superposition)
        self._polar_calc_group.addButton(self.polar_plot_fem)

        layout.addLayout(plot_option_layout)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Plot Optionen Impulsbereich
        impulse_option_layout = QtWidgets.QGridLayout()
        impulse_option_layout.setHorizontalSpacing(20)
        impulse_option_layout.setVerticalSpacing(8)

        impulse_option_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        impulse_option_layout.addWidget(QtWidgets.QLabel("Superposition"), 0, 1, alignment=QtCore.Qt.AlignCenter)
        impulse_option_layout.addWidget(QtWidgets.QLabel("BEM Analyse"), 0, 2, alignment=QtCore.Qt.AlignCenter)

        self.impulse_plot_superposition = QtWidgets.QCheckBox()
        self.impulse_plot_bem = QtWidgets.QCheckBox()
        impulse_option_layout.addWidget(QtWidgets.QLabel("Impulse Plot"), 1, 0)
        impulse_option_layout.addWidget(self.impulse_plot_superposition, 1, 1, alignment=QtCore.Qt.AlignCenter)
        impulse_option_layout.addWidget(self.impulse_plot_bem, 1, 2, alignment=QtCore.Qt.AlignCenter)

        self._impulse_calc_group = QtWidgets.QButtonGroup(self)
        self._impulse_calc_group.setExclusive(True)
        self._impulse_calc_group.addButton(self.impulse_plot_superposition)
        self._impulse_calc_group.addButton(self.impulse_plot_bem)

        layout.addLayout(impulse_option_layout)
        layout.addStretch()

    def setup_connections(self):
        self.impulse_plot_height.editingFinished.connect(self.on_ImpulsePlotHeight_changed)
        self.impulse_min_spl.editingFinished.connect(self.on_ImpulseMinSPL_changed)
        self.size_of_measurement_point.editingFinished.connect(self.on_MeasurementSize_changed)
        self.colorization_mode.currentIndexChanged.connect(self.on_ColorizationMode_changed)
        self.freq_bandwidth_upper.currentIndexChanged.connect(self.validate_frequency_range)
        self.freq_bandwidth_lower.currentIndexChanged.connect(self.validate_frequency_range)
        self.resolution.editingFinished.connect(self.on_Resolution_changed)
        self.position_plot_length.editingFinished.connect(self.on_PositionLength_changed)
        self.position_plot_width.editingFinished.connect(self.on_PositionWidth_changed)

        checkbox_mapping = {
            self.spl_plot_superposition: "spl_plot_superposition",
            self.spl_plot_fem: "spl_plot_fem",
            self.xaxis_plot_superposition: "xaxis_plot_superposition",
            self.xaxis_plot_fem: "xaxis_plot_fem",
            self.yaxis_plot_superposition: "yaxis_plot_superposition",
            self.yaxis_plot_fem: "yaxis_plot_fem",
            self.polar_plot_superposition: "polar_plot_superposition",
            self.polar_plot_fem: "polar_plot_fem",
            self.impulse_plot_superposition: "impulse_plot_superposition",
            self.impulse_plot_bem: "impulse_plot_bem",
            self.update_pressure_soundfield: "update_pressure_soundfield",
            self.update_pressure_axisplot: "update_pressure_axisplot",
            self.update_pressure_polarplot: "update_pressure_polarplot",
            self.update_pressure_impulse: "update_pressure_impulse",
        }

        for checkbox, attribute in checkbox_mapping.items():
            checkbox.stateChanged.connect(lambda state, attr=attribute: self.on_calculation_option_changed(attr, state))

        # Verbindungen für die Polar-Frequenzen
        for i, color in enumerate(['red', 'yellow', 'green', 'cyan']):
            self.freq_boxes[i].currentTextChanged.connect(
                lambda text, c=color: self.on_polar_frequency_changed(text, c))

        # Neue Verbindungen für die SPL Plot Einstellungen
        self.max_spl.editingFinished.connect(self.on_max_spl_changed)
        self.min_spl.editingFinished.connect(self.on_min_spl_changed)
        self.db_step.editingFinished.connect(self.on_db_step_changed)

    def update_ui_from_settings(self):
        self.impulse_plot_height.setText(str(self.settings.impulse_plot_height))
        self.impulse_min_spl.setText(str(self.settings.impulse_min_spl))
        self.size_of_measurement_point.setText(f"{self.settings.measurement_size:.2f}")
        self.colorization_mode.setCurrentText(self.settings.colorization_mode)
        self.set_combobox_to_frequency(self.freq_bandwidth_upper, self.settings.upper_calculate_frequency)
        self.set_combobox_to_frequency(self.freq_bandwidth_lower, self.settings.lower_calculate_frequency)
        self.resolution.setText(f"{self.settings.resolution:.2f}")
        self.position_plot_length.setText(str(self.settings.position_x_axis))
        self.position_plot_width.setText(str(self.settings.position_y_axis))

        self.spl_plot_superposition.setChecked(self.settings.spl_plot_superposition)
        self.spl_plot_fem.setChecked(self.settings.spl_plot_fem)
        self.xaxis_plot_superposition.setChecked(self.settings.xaxis_plot_superposition)
        self.xaxis_plot_fem.setChecked(self.settings.xaxis_plot_fem)
        self.yaxis_plot_superposition.setChecked(self.settings.yaxis_plot_superposition)
        self.yaxis_plot_fem.setChecked(self.settings.yaxis_plot_fem)
        self.polar_plot_superposition.setChecked(self.settings.polar_plot_superposition)
        self.polar_plot_fem.setChecked(self.settings.polar_plot_fem)
        self.impulse_plot_superposition.setChecked(self.settings.impulse_plot_superposition)
        self.impulse_plot_bem.setChecked(self.settings.impulse_plot_bem)
        self.update_pressure_soundfield.setChecked(self.settings.update_pressure_soundfield)
        self.update_pressure_axisplot.setChecked(self.settings.update_pressure_axisplot)
        self.update_pressure_polarplot.setChecked(self.settings.update_pressure_polarplot)
        self.update_pressure_impulse.setChecked(self.settings.update_pressure_impulse)

        # Update SPL Plot settings
        self.max_spl.setText(str(self.settings.colorbar_range['max']))
        self.min_spl.setText(str(self.settings.colorbar_range['min']))
        self.db_step.setText(str(self.settings.colorbar_range['step']))

    def set_combobox_to_frequency(self, combobox, freq_value):
        freq_str = f"{int(freq_value)} Hz"
        index = combobox.findText(freq_str)
        combobox.blockSignals(True)
        if index != -1:
            combobox.setCurrentIndex(index)
        else:
            print(f"[WARN] Wert {freq_str} nicht in ComboBox gefunden!")
        combobox.blockSignals(False)

    # Signal Handlers
    def on_ImpulsePlotHeight_changed(self):
        try:
            value = int(self.impulse_plot_height.text())
            if self.settings.impulse_plot_height != value:
                self.settings.impulse_plot_height = value
                # Aktualisiere die Plot-Höhen
                if hasattr(self.main_window, 'impulse_manager') and self.main_window.impulse_manager:
                    self.main_window.impulse_manager.update_plot_impulse()
        except ValueError:
            self.impulse_plot_height.setText(str(self.settings.impulse_plot_height))

    def on_ImpulseMinSPL_changed(self):
        try:
            value = int(self.impulse_min_spl.text())
            if self.settings.impulse_min_spl != value:
                self.settings.impulse_min_spl = value
                self.main_window.update_speaker_array_calculations()
        except ValueError:
            self.impulse_min_spl.setText(str(self.settings.impulse_min_spl))

    def on_MeasurementSize_changed(self):
        try:
            value = round(float(self.size_of_measurement_point.text()), 2)
            if self.settings.measurement_size != value:
                self.settings.measurement_size = value
                self.main_window.update_speaker_array_calculations()
            self.size_of_measurement_point.setText(f"{value:.2f}")
        except ValueError:
            self.size_of_measurement_point.setText(f"{self.settings.measurement_size:.2f}")

    def on_max_spl_changed(self):
        try:
            num_colors = 11  # Fixe Anzahl von Farben
            value = int(self.max_spl.text())
            step = self.settings.colorbar_range['step']
            
            # Berechne den neuen Minimalwert basierend auf max und step
            new_min = value - (step * (num_colors - 1))
            
            # Prüfe ob der neue Minimalwert im erlaubten Bereich liegt
            if new_min >= 0 and new_min <= 150:
                self.settings.colorbar_range['max'] = value
                self.settings.colorbar_range['min'] = new_min
                self.min_spl.setText(str(new_min))
                # self.main_window.update_speaker_array_calculations()
            else:
                # Setze auf den letzten gültigen Wert zurück
                self.max_spl.setText(str(self.settings.colorbar_range['max']))
        except ValueError:
            self.max_spl.setText(str(self.settings.colorbar_range['max']))

        self.main_window.plot_spl()


    def on_min_spl_changed(self):
        try:
            num_colors = 11  # Fixe Anzahl von Farben
            value = int(self.min_spl.text())
            step = self.settings.colorbar_range['step']
            
            # Berechne den neuen Maximalwert basierend auf min und step
            new_max = value + (step * (num_colors - 1))
            
            # Prüfe ob der neue Maximalwert im erlaubten Bereich liegt
            if new_max >= 0 and new_max <= 150:
                self.settings.colorbar_range['min'] = value
                self.settings.colorbar_range['max'] = new_max
                self.max_spl.setText(str(new_max))
                # self.main_window.update_speaker_array_calculations()
            else:
                # Setze auf den letzten gültigen Wert zurück
                self.min_spl.setText(str(self.settings.colorbar_range['min']))
        except ValueError:
            self.min_spl.setText(str(self.settings.colorbar_range['min']))
        
        self.main_window.plot_spl()

    def on_db_step_changed(self):
        try:
            num_colors = 11  # Fixe Anzahl von Farben
            step = int(self.db_step.text())
            current_min = self.settings.colorbar_range['min']
            current_max = self.settings.colorbar_range['max']
            
            if current_min == 0:
                # Wenn min auf 0 ist, passe max an
                new_max = current_min + (step * (num_colors - 1))
                if new_max <= 150:
                    self.settings.colorbar_range['step'] = step
                    self.settings.colorbar_range['max'] = new_max
                    self.max_spl.setText(str(new_max))
                else:
                    # Setze auf den letzten gültigen Wert zurück
                    self.db_step.setText(str(self.settings.colorbar_range['step']))
                    return
            else:
                # Sonst passe min an
                new_min = current_max - (step * (num_colors - 1))
                if new_min >= 0:
                    self.settings.colorbar_range['step'] = step
                    self.settings.colorbar_range['min'] = new_min
                    self.min_spl.setText(str(new_min))
                else:
                    # Setze auf den letzten gültigen Wert zurück
                    self.db_step.setText(str(self.settings.colorbar_range['step']))
                    return
                
            # self.main_window.update_speaker_array_calculations()
        except ValueError:
            self.db_step.setText(str(self.settings.colorbar_range['step']))

        self.main_window.plot_spl()

    # def on_freq_bandwidth_changed(self):
    #     self.main_window.update_freq_bandwidth()


    def on_ColorizationMode_changed(self, index):
        # Dies bleibt gleich, da wir intern weiterhin "step" vs "linear" verwenden
        self.settings.colorization_mode = "Color step" if index == 0 else "Gradient"
        # Prüfe, ob sources_instance existiert
        if hasattr(self.main_window, "sources_instance") and self.main_window.sources_instance is not None:
            self.main_window.plot_spl()
        else:
            print("[WARN] plot_spl() nicht aufgerufen, da sources_instance nicht existiert.")

    def on_calculation_option_changed(self, attribute_name, state):
        new_value = bool(state)
        if getattr(self.settings, attribute_name, False) != new_value:
            setattr(self.settings, attribute_name, new_value)
            if hasattr(self.main_window, "update_speaker_array_calculations"):
                self.main_window.update_speaker_array_calculations()

    def on_Resolution_changed(self):
        try:
            value = round(float(self.resolution.text()), 2)
            if self.settings.resolution != value:
                self.settings.resolution = value
                self.main_window.update_speaker_array_calculations()
            self.resolution.setText(f"{value:.2f}")
        except ValueError:
            self.resolution.setText(f"{self.settings.resolution:.2f}")

    def on_PositionLength_changed(self):
        try:
            value = int(self.position_plot_length.text())
            min_allowed = -self.settings.width / 2
            max_allowed = self.settings.width / 2
            
            # Prüfe ob der Wert innerhalb der erlaubten Grenzen liegt
            if value < min_allowed or value > max_allowed:
                self.position_plot_length.setStyleSheet("background-color: #FFE4E1;")
                # Setze nach 500ms auf den letzten gültigen Wert zurück
                QtCore.QTimer.singleShot(500, lambda: self.reset_position_value(
                    self.position_plot_length, 
                    self.settings.position_x_axis,
                    "position_x_axis"
                ))
            else:
                self.position_plot_length.setStyleSheet("")
                if self.settings.position_x_axis != value:
                    self.settings.position_x_axis = value
                    self.main_window.update_speaker_array_calculations()
        except ValueError:
            self.position_plot_length.setStyleSheet("background-color: #FFE4E1;")
            QtCore.QTimer.singleShot(500, lambda: self.reset_position_value(
                self.position_plot_length, 
                self.settings.position_x_axis,
                "position_x_axis"
            ))

    def on_PositionWidth_changed(self):
        try:
            value = int(self.position_plot_width.text())
            min_allowed = -self.settings.length / 2
            max_allowed = self.settings.length / 2
            
            # Prüfe ob der Wert innerhalb der erlaubten Grenzen liegt
            if value < min_allowed or value > max_allowed:
                self.position_plot_width.setStyleSheet("background-color: #FFE4E1;")
                # Setze nach 500ms auf den letzten gültigen Wert zurück
                QtCore.QTimer.singleShot(500, lambda: self.reset_position_value(
                    self.position_plot_width, 
                    self.settings.position_y_axis,
                    "position_y_axis"
                ))
            else:
                self.position_plot_width.setStyleSheet("")
                if self.settings.position_y_axis != value:
                    self.settings.position_y_axis = value
                    self.main_window.update_speaker_array_calculations()
        except ValueError:
            self.position_plot_width.setStyleSheet("background-color: #FFE4E1;")
            QtCore.QTimer.singleShot(500, lambda: self.reset_position_value(
                self.position_plot_width, 
                self.settings.position_y_axis,
                "position_y_axis"
            ))


    def on_polar_frequency_changed(self, text, color):
        try:
            frequency = float(text.split()[0])
            self.settings.update_polar_frequency(color, frequency)
            self.main_window.update_freq_bandwidth()

        except (ValueError, IndexError) as e:
            print(f"Fehler beim Setzen der Polar-Frequenz: {e}")

    

    def reset_position_value(self, widget, value, attribute_name):
        """Setzt den Wert eines Position-Widgets auf den letzten gültigen Wert zurück."""
        widget.setStyleSheet("")
        widget.setText(str(value))

    def validate_frequency_range(self):
        """Stellt sicher, dass die obere Frequenz nicht kleiner ist als die untere"""
        if not hasattr(self, 'freq_bandwidth_upper') or not hasattr(self, 'freq_bandwidth_lower'):
            return

        upper_text = self.freq_bandwidth_upper.currentText()
        lower_text = self.freq_bandwidth_lower.currentText()        
        upper_freq = float(upper_text.split()[0])
        lower_freq = float(lower_text.split()[0])

        if upper_freq < lower_freq:
            current_lower_idx = self.freq_bandwidth_lower.currentIndex()
            self.freq_bandwidth_upper.setCurrentIndex(current_lower_idx)
            upper_freq = lower_freq

        # Berechne geometrische Mitte
        center_freq = np.sqrt(lower_freq * upper_freq)
        
        # Speichere Werte
        self.settings.lower_calculate_frequency = lower_freq
        self.settings.upper_calculate_frequency = upper_freq
        self.settings.calculate_frequency = center_freq
        self.main_window.update_freq_bandwidth()

    def open_settings_window(self):
        self.update_ui_from_settings()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()


