from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QDoubleValidator, QIntValidator
import numpy as np
from Module_LFO.Modules_Calculate.Functions import FunctionToolbox

ROW_SPACING = 8
SECTION_SPACING = 14
LINE_EDIT_WIDTH = 110
COMBO_BOX_WIDTH = 120


class UiSettings(QtWidgets.QWidget):
    settings_changed = QtCore.pyqtSignal(object)  # Signal für Einstellungsänderungen

    def __init__(self, main_window, settings, data):
        super().__init__()
        self.main_window = main_window
        self.settings = settings  # Verwenden Sie die übergebene settings-Instanz
        self.data = data
        self.function_toolbox = FunctionToolbox(settings)  # Initialisiere FunctionToolbox
        self.setup_ui()
        self.setup_connections()  # Stellen Sie sicher, dass dies aufgerufen wird
        # Initiale Berechnung der Luftdichte beim Start
        self.initialize_air_density()
        self.hide()  # Initially hide the settings window

    def setup_ui(self):
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 300, 600)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)

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
        self._apply_input_font_size()

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

    def _create_collapsible_section(self, parent_layout, title, default_open=False):
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 12, 0, 0)
        header_layout.setSpacing(6)

        toggle_button = QtWidgets.QToolButton()
        toggle_button.setObjectName("SectionToggleButton")
        toggle_button.setCheckable(True)
        toggle_button.setChecked(default_open)
        toggle_button.setArrowType(QtCore.Qt.DownArrow if default_open else QtCore.Qt.RightArrow)
        toggle_button.setFixedSize(12, 12)
        toggle_button.setToolTip("Abschnitt ein-/ausklappen")
        toggle_button.setStyleSheet("QToolButton#SectionToggleButton { border: none; padding: 0px; }")
        toggle_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        title_label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        def _toggle_from_label(event, btn=toggle_button):
            btn.toggle()
            event.accept()

        title_label.mousePressEvent = _toggle_from_label

        header_layout.addWidget(toggle_button)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        parent_layout.addLayout(header_layout)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        parent_layout.addWidget(line)

        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 6, 0, 6)
        content_layout.setSpacing(ROW_SPACING)
        content_widget.setVisible(default_open)
        parent_layout.addWidget(content_widget)
        parent_layout.addSpacing(SECTION_SPACING)

        toggle_button.toggled.connect(
            lambda state, btn=toggle_button, widget=content_widget: self._toggle_section_content(btn, widget, state)
        )

        return content_layout

    def _apply_input_font_size(self):
        base_font = QtGui.QFont(self.font())
        point_size = base_font.pointSize()
        if point_size > 0:
            base_font.setPointSize(max(point_size - 1, 8))
        for widget_type in (QtWidgets.QLineEdit, QtWidgets.QComboBox):
            for widget in self.findChildren(widget_type):
                widget.setFont(base_font)

    def _create_form_row(self, label_text, widget):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(ROW_SPACING)
        label = QtWidgets.QLabel(label_text)
        row.addWidget(label)
        row.addStretch()
        row.addWidget(widget)
        return row

    def _toggle_section_content(self, button, content_widget, expanded):
        content_widget.setVisible(expanded)
        button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)



# ----- SETUP CONNECTIONS -----

    def setup_connections(self):
        self.impulse_plot_height.editingFinished.connect(self.on_ImpulsePlotHeight_changed)
        self.impulse_min_spl.editingFinished.connect(self.on_ImpulseMinSPL_changed)
        self.size_of_measurement_point.editingFinished.connect(self.on_MeasurementSize_changed)
        self.colorization_mode.currentIndexChanged.connect(self.on_ColorizationMode_changed)
        self.freq_bandwidth_upper.currentIndexChanged.connect(self.validate_frequency_range)
        self.freq_bandwidth_lower.currentIndexChanged.connect(self.validate_frequency_range)
        self.resolution.editingFinished.connect(self.on_Resolution_changed)
        self.temperature.editingFinished.connect(self.on_Temperature_changed)
        self.humidity.editingFinished.connect(self.on_Humidity_changed)
        self.use_air_absorption.stateChanged.connect(self.on_AirAbsorption_changed)
        self.position_plot_length.editingFinished.connect(self.on_PositionLength_changed)
        self.position_plot_width.editingFinished.connect(self.on_PositionWidth_changed)
        # 3D-Achsenanzeige
        if hasattr(self, "axis_3d_transparency"):
            self.axis_3d_transparency.editingFinished.connect(self.on_AxisTransparency_changed)

        checkbox_mapping = {
            self.spl_plot_superposition: "spl_plot_superposition",
            self.spl_plot_fem: "spl_plot_fem",
            self.xaxis_plot_superposition: "xaxis_plot_superposition",
            self.xaxis_plot_fem: "xaxis_plot_fem",
            self.yaxis_plot_superposition: "yaxis_plot_superposition",
            self.yaxis_plot_fem: "yaxis_plot_fem",
            self.impulse_plot_superposition: "impulse_plot_superposition",
            self.impulse_plot_bem: "impulse_plot_bem",
            self.update_pressure_soundfield: "update_pressure_soundfield",
            self.update_pressure_axisplot: "update_pressure_axisplot",
            self.update_pressure_polarplot: "update_pressure_polarplot",
            self.update_pressure_impulse: "update_pressure_impulse",
            self.fem_particle_velocity_checkbox: "fem_compute_particle_velocity",
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

    def setup_tab_mapping(self):
        layout = QtWidgets.QVBoxLayout(self.tab_mapping)
        layout.setSpacing(ROW_SPACING)

        # SPL Farbbereich
        colorbar_section = self._create_collapsible_section(
            layout,
            "SPL color range",
            default_open=True
        )

        self.max_spl = QLineEdit()
        self.max_spl.setValidator(QIntValidator(0, 150))
        self.max_spl.setFixedWidth(LINE_EDIT_WIDTH)
        colorbar_section.addLayout(
            self._create_form_row("Max. SPL Plot", self.max_spl)
        )

        self.min_spl = QLineEdit()
        self.min_spl.setValidator(QIntValidator(0, 150))
        self.min_spl.setFixedWidth(LINE_EDIT_WIDTH)
        colorbar_section.addLayout(
            self._create_form_row("Min. SPL Plot", self.min_spl)
        )

        self.db_step = QLineEdit()
        self.db_step.setValidator(QIntValidator(1, 20))
        self.db_step.setFixedWidth(LINE_EDIT_WIDTH)
        colorbar_section.addLayout(
            self._create_form_row("dB per color step", self.db_step)
        )

        self.colorization_mode = QtWidgets.QComboBox()
        self.colorization_mode.addItems(["Color step", "Gradient"])
        self.colorization_mode.setFixedWidth(COMBO_BOX_WIDTH)
        colorbar_section.addLayout(
            self._create_form_row("SPL mode", self.colorization_mode)
        )

        # Positionsplots
        position_section = self._create_collapsible_section(
            layout,
            "Axis plots",
            default_open=False
        )

        self.position_plot_length = QLineEdit()
        self.position_plot_length.setValidator(QIntValidator(-100, 100))
        self.position_plot_length.setFixedWidth(LINE_EDIT_WIDTH)
        position_section.addLayout(
            self._create_form_row("Y Axis plot position", self.position_plot_length)
        )

        self.position_plot_width = QLineEdit()
        self.position_plot_width.setValidator(QIntValidator(-100, 100))
        self.position_plot_width.setFixedWidth(LINE_EDIT_WIDTH)
        position_section.addLayout(
            self._create_form_row("X Axis plot position", self.position_plot_width)
        )

        self.axis_3d_transparency = QLineEdit()
        # Transparenz in Prozent (0–100)
        self.axis_3d_transparency.setValidator(QIntValidator(0, 100))
        self.axis_3d_transparency.setFixedWidth(LINE_EDIT_WIDTH)
        position_section.addLayout(
            self._create_form_row("Axis transparency (%)", self.axis_3d_transparency)
        )

        # Polar Frequenzen
        polar_section = self._create_collapsible_section(
            layout,
            "Polar frequencies",
            default_open=False
        )

        self.freq_selection_widget = QtWidgets.QWidget()
        freq_selection_layout = QtWidgets.QVBoxLayout(self.freq_selection_widget)
        freq_selection_layout.setSpacing(ROW_SPACING)
        freq_selection_layout.setContentsMargins(0, 0, 0, 0)

        self.freq_boxes = []
        frequencies = ['31.5 Hz', '40 Hz', '50 Hz', '63 Hz']
        colors = ['red', 'yellow', 'green', 'cyan']

        for freq, color in zip(frequencies, colors):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(ROW_SPACING)

            color_box = QtWidgets.QLabel()
            color_box.setFixedSize(15, 15)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")

            combo = QtWidgets.QComboBox()
            combo.addItems(['25 Hz', '31.5 Hz', '40 Hz', '50 Hz', '63 Hz', '80 Hz'])
            combo.setCurrentText(freq)
            combo.setFixedWidth(COMBO_BOX_WIDTH)

            row_layout.addWidget(color_box)
            row_layout.addStretch()
            row_layout.addWidget(combo)

            freq_selection_layout.addWidget(row_widget)
            self.freq_boxes.append(combo)

        polar_section.addWidget(self.freq_selection_widget)

        layout.addStretch()

    def setup_tab_calculation(self):
        layout = QtWidgets.QVBoxLayout(self.tab_calculation)
        layout.setSpacing(ROW_SPACING)

        # Upper und Lower Frequency bandwidth
        freq_bandwidth_upper_layout = QtWidgets.QHBoxLayout()
        freq_bandwidth_upper_layout.setSpacing(ROW_SPACING)
        freq_bandwidth_lower_layout = QtWidgets.QHBoxLayout()
        freq_bandwidth_lower_layout.setSpacing(ROW_SPACING)
        freq_band_points_layout = QtWidgets.QHBoxLayout()
        freq_band_points_layout.setSpacing(ROW_SPACING)

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
        self.fem_calculate_frequency = QtWidgets.QComboBox()
        self.freq_band_points = QtWidgets.QComboBox()

        self.freq_bandwidth_lower.blockSignals(True)
        self.freq_bandwidth_upper.blockSignals(True)
        self.fem_calculate_frequency.blockSignals(True)

        self.freq_bandwidth_lower.addItems(frequency_strings)
        self.freq_bandwidth_upper.addItems(frequency_strings)
        self.fem_calculate_frequency.addItems(frequency_strings)

        # Anzahl der Frequenzpunkte für Bandmittelung (1–12 als vordefinierte Auswahl)
        self.freq_band_points.addItems([str(i) for i in range(1, 13)])
        self.freq_band_points.setFixedWidth(COMBO_BOX_WIDTH)
        freq_band_points_label = QtWidgets.QLabel("Averaging points in band")
        freq_band_points_layout.addWidget(freq_band_points_label)
        freq_band_points_layout.addStretch()
        freq_band_points_layout.addWidget(self.freq_band_points)

        self.freq_bandwidth_lower.setCurrentText(f"{self.settings.lower_calculate_frequency} Hz")
        self.freq_bandwidth_upper.setCurrentText(f"{self.settings.upper_calculate_frequency} Hz")
        fem_freq = int(getattr(self.settings, "fem_calculate_frequency", 50))
        if f"{fem_freq} Hz" in frequency_strings:
            self.fem_calculate_frequency.setCurrentText(f"{fem_freq} Hz")
        # Initialwert für Anzahl der Frequenzpunkte
        points = int(getattr(self.settings, "frequency_band_points", 3))
        if 1 <= points <= 12:
            self.freq_band_points.setCurrentText(str(points))
        else:
            self.freq_band_points.setCurrentText("3")

        self.freq_bandwidth_lower.blockSignals(False)
        self.freq_bandwidth_upper.blockSignals(False)
        self.fem_calculate_frequency.blockSignals(False)

        self.freq_bandwidth_upper.currentIndexChanged.connect(self.validate_frequency_range)
        self.freq_bandwidth_lower.currentIndexChanged.connect(self.validate_frequency_range)
        self.fem_calculate_frequency.currentIndexChanged.connect(self.on_fem_frequency_changed)
        self.freq_band_points.currentIndexChanged.connect(self.on_freq_band_points_changed)

        freq_bandwidth_upper_label = QtWidgets.QLabel("Upper frequency bandwidth")
        self.freq_bandwidth_upper.setFixedWidth(COMBO_BOX_WIDTH)
        freq_bandwidth_upper_layout.addWidget(freq_bandwidth_upper_label)
        freq_bandwidth_upper_layout.addStretch()
        freq_bandwidth_upper_layout.addWidget(self.freq_bandwidth_upper)

        freq_bandwidth_lower_label = QtWidgets.QLabel("Lower frequency bandwidth")
        self.freq_bandwidth_lower.setFixedWidth(COMBO_BOX_WIDTH)
        freq_bandwidth_lower_layout.addWidget(freq_bandwidth_lower_label)
        freq_bandwidth_lower_layout.addStretch()
        freq_bandwidth_lower_layout.addWidget(self.freq_bandwidth_lower)

        fem_frequency_layout = QtWidgets.QHBoxLayout()
        fem_frequency_label = QtWidgets.QLabel("Frequency for FEM calc")
        self.fem_calculate_frequency.setFixedWidth(COMBO_BOX_WIDTH)
        fem_frequency_layout.addWidget(fem_frequency_label)
        fem_frequency_layout.addStretch()
        fem_frequency_layout.addWidget(self.fem_calculate_frequency)

        freq_section_layout = self._create_collapsible_section(
            layout,
            "Frequency range: Axis & Soundfield",
            default_open=True
        )

        # Kleine Trennlinie + Titel für Superposition-Bereich
        super_header_layout = QtWidgets.QHBoxLayout()
        super_header_layout.setContentsMargins(0, 8, 0, 8)
        super_label = QtWidgets.QLabel("Superposition calculation")
        # Nur kursiv, nicht fett
        super_label.setStyleSheet("font-size: 10px; font-style: italic;")
        super_line_left = QtWidgets.QFrame()
        super_line_left.setFrameShape(QtWidgets.QFrame.HLine)
        super_line_left.setFrameShadow(QtWidgets.QFrame.Sunken)
        super_line_right = QtWidgets.QFrame()
        super_line_right.setFrameShape(QtWidgets.QFrame.HLine)
        super_line_right.setFrameShadow(QtWidgets.QFrame.Sunken)
        super_header_layout.addWidget(super_line_left)
        super_header_layout.addWidget(super_label)
        super_header_layout.addWidget(super_line_right)

        # Kleine Trennlinie + Titel für FEM-Bereich
        fem_header_layout = QtWidgets.QHBoxLayout()
        # Etwas größerer Abstand nach oben als beim Superposition-Header
        fem_header_layout.setContentsMargins(0, 14, 0, 8)
        fem_title_label = QtWidgets.QLabel("FEM calculation")
        fem_title_label.setStyleSheet("font-size: 10px; font-style: italic;")
        fem_line_left = QtWidgets.QFrame()
        fem_line_left.setFrameShape(QtWidgets.QFrame.HLine)
        fem_line_left.setFrameShadow(QtWidgets.QFrame.Sunken)
        fem_line_right = QtWidgets.QFrame()
        fem_line_right.setFrameShape(QtWidgets.QFrame.HLine)
        fem_line_right.setFrameShadow(QtWidgets.QFrame.Sunken)
        fem_header_layout.addWidget(fem_line_left)
        fem_header_layout.addWidget(fem_title_label)
        fem_header_layout.addWidget(fem_line_right)

        # Inhalte in der gewünschten Reihenfolge in den Abschnitt einfügen
        freq_section_layout.addLayout(super_header_layout)
        freq_section_layout.addLayout(freq_bandwidth_upper_layout)
        freq_section_layout.addLayout(freq_bandwidth_lower_layout)
        freq_section_layout.addLayout(freq_band_points_layout)
        freq_section_layout.addLayout(fem_header_layout)
        freq_section_layout.addLayout(fem_frequency_layout)

        # Resolution
        resolution_layout = QtWidgets.QHBoxLayout()
        resolution_layout.setSpacing(ROW_SPACING)
        resolution_label = QtWidgets.QLabel("Resolution")
        self.resolution = QLineEdit()
        self.resolution.setValidator(QDoubleValidator(0.1, 3.0, 2))
        self.resolution.setFixedWidth(LINE_EDIT_WIDTH)
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addStretch()
        resolution_layout.addWidget(self.resolution)

        resolution_section_layout = self._create_collapsible_section(
            layout,
            "Resolution Soundfield",
            default_open=False
        )
        resolution_section_layout.addLayout(resolution_layout)

        # Temperature
        temperature_layout = QtWidgets.QHBoxLayout()
        temperature_layout.setSpacing(ROW_SPACING)
        temperature_label = QtWidgets.QLabel("Temperature (°C)")
        self.temperature = QLineEdit()
        self.temperature.setValidator(QDoubleValidator(-20.0, 50.0, 2))
        self.temperature.setFixedWidth(LINE_EDIT_WIDTH)
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addStretch()
        temperature_layout.addWidget(self.temperature)

        # Humidity
        humidity_layout = QtWidgets.QHBoxLayout()
        humidity_layout.setSpacing(ROW_SPACING)
        humidity_label = QtWidgets.QLabel("Humidity (%)")
        self.humidity = QLineEdit()
        self.humidity.setValidator(QDoubleValidator(0.0, 100.0, 2))
        self.humidity.setFixedWidth(LINE_EDIT_WIDTH)
        humidity_layout.addWidget(humidity_label)
        humidity_layout.addStretch()
        humidity_layout.addWidget(self.humidity)

        # Air Absorption Checkbox
        air_absorption_layout = QtWidgets.QHBoxLayout()
        air_absorption_layout.setSpacing(ROW_SPACING)
        self.use_air_absorption = QtWidgets.QCheckBox("Use air absorption in calculations")
        air_absorption_layout.addWidget(self.use_air_absorption)
        air_absorption_layout.addStretch()

        environmental_section_layout = self._create_collapsible_section(
            layout,
            "Environmental conditions",
            default_open=False
        )
        environmental_section_layout.addLayout(temperature_layout)
        environmental_section_layout.addLayout(humidity_layout)
        environmental_section_layout.addLayout(air_absorption_layout)

        update_pressure_checks_layout = QtWidgets.QVBoxLayout()
        update_pressure_checks_layout.setSpacing(ROW_SPACING)
        self.update_pressure_soundfield = QtWidgets.QCheckBox("Soundfield")
        self.update_pressure_axisplot = QtWidgets.QCheckBox("Axis Plot")
        self.update_pressure_polarplot = QtWidgets.QCheckBox("Polar Plot")
        self.update_pressure_impulse = QtWidgets.QCheckBox("Impulse Plot")
        update_pressure_checks_layout.addWidget(self.update_pressure_soundfield)
        update_pressure_checks_layout.addWidget(self.update_pressure_axisplot)
        update_pressure_checks_layout.addWidget(self.update_pressure_polarplot)
        update_pressure_checks_layout.addWidget(self.update_pressure_impulse)

        update_section_layout = self._create_collapsible_section(
            layout,
            "Update automatically pressure",
            default_open=False
        )
        update_section_layout.addLayout(update_pressure_checks_layout)

        # Plot Optionen Hauptbereich
        plot_option_layout = QtWidgets.QGridLayout()
        plot_option_layout.setHorizontalSpacing(20)
        plot_option_layout.setVerticalSpacing(ROW_SPACING)

        plot_option_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        plot_option_layout.addWidget(QtWidgets.QLabel("Superposition"), 0, 1, alignment=QtCore.Qt.AlignCenter)
        plot_option_layout.addWidget(QtWidgets.QLabel("FEM Analysis"), 0, 2, alignment=QtCore.Qt.AlignCenter)

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

        calculation_section_layout = self._create_collapsible_section(
            layout,
            "Calculation method",
            default_open=False
        )
        calculation_section_layout.addLayout(plot_option_layout)

        calc_separator = QtWidgets.QFrame()
        calc_separator.setFrameShape(QtWidgets.QFrame.HLine)
        calculation_section_layout.addWidget(calc_separator)

        fem_settings_layout = self._create_collapsible_section(
            layout,
            "FEM calculation settings",
            default_open=False
        )
        self.fem_particle_velocity_checkbox = QtWidgets.QCheckBox("Particle velocity")
        fem_settings_layout.addWidget(self.fem_particle_velocity_checkbox)

        # Plot Optionen Impulsbereich
        impulse_option_layout = QtWidgets.QGridLayout()
        impulse_option_layout.setHorizontalSpacing(20)
        impulse_option_layout.setVerticalSpacing(ROW_SPACING)

        impulse_option_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        impulse_option_layout.addWidget(QtWidgets.QLabel("Superposition"), 0, 1, alignment=QtCore.Qt.AlignCenter)
        impulse_option_layout.addWidget(QtWidgets.QLabel("BEM Analysis"), 0, 2, alignment=QtCore.Qt.AlignCenter)

        self.impulse_plot_superposition = QtWidgets.QCheckBox()
        self.impulse_plot_bem = QtWidgets.QCheckBox()
        impulse_option_layout.addWidget(QtWidgets.QLabel("Impulse Plot"), 1, 0)
        impulse_option_layout.addWidget(self.impulse_plot_superposition, 1, 1, alignment=QtCore.Qt.AlignCenter)
        impulse_option_layout.addWidget(self.impulse_plot_bem, 1, 2, alignment=QtCore.Qt.AlignCenter)

        self._impulse_calc_group = QtWidgets.QButtonGroup(self)
        self._impulse_calc_group.setExclusive(True)
        self._impulse_calc_group.addButton(self.impulse_plot_superposition)
        self._impulse_calc_group.addButton(self.impulse_plot_bem)

        calculation_section_layout.addLayout(impulse_option_layout)
        layout.addStretch()

    def setup_tab_settings(self):
        layout = QtWidgets.QVBoxLayout(self.tab_settings)
        layout.setSpacing(ROW_SPACING)

        # Impulse Optionen
        impulse_section = self._create_collapsible_section(
            layout,
            "Impulse settings",
            default_open=True
        )

        self.impulse_plot_height = QLineEdit()
        self.impulse_plot_height.setValidator(QIntValidator(50, 1000))
        self.impulse_plot_height.setFixedWidth(LINE_EDIT_WIDTH)
        impulse_section.addLayout(
            self._create_form_row("Impulse plot height", self.impulse_plot_height)
        )

        self.impulse_min_spl = QLineEdit()
        self.impulse_min_spl.setValidator(QIntValidator(-100, -6))
        self.impulse_min_spl.setFixedWidth(LINE_EDIT_WIDTH)
        impulse_section.addLayout(
            self._create_form_row("Impulse SPL range", self.impulse_min_spl)
        )

        # Messpunkte
        measurement_section = self._create_collapsible_section(
            layout,
            "Measurement point display",
            default_open=False
        )

        self.size_of_measurement_point = QLineEdit()
        self.size_of_measurement_point.setValidator(QDoubleValidator(0.1, 20.0, 2))
        self.size_of_measurement_point.setFixedWidth(LINE_EDIT_WIDTH)
        measurement_section.addLayout(
            self._create_form_row("Size measurement point", self.size_of_measurement_point)
        )

        layout.addStretch()


# ----- UPDATE -----

    def update_ui_from_settings(self):
        self.impulse_plot_height.setText(str(self.settings.impulse_plot_height))
        self.impulse_min_spl.setText(str(self.settings.impulse_min_spl))
        self.size_of_measurement_point.setText(f"{self.settings.measurement_size:.2f}")
        self.colorization_mode.setCurrentText(self.settings.colorization_mode)
        self.set_combobox_to_frequency(self.freq_bandwidth_upper, self.settings.upper_calculate_frequency)
        self.set_combobox_to_frequency(self.freq_bandwidth_lower, self.settings.lower_calculate_frequency)
        self.set_combobox_to_frequency(
            self.fem_calculate_frequency,
            getattr(self.settings, "fem_calculate_frequency", 50)
        )
        # Anzahl der Frequenzpunkte im Band
        points = int(getattr(self.settings, "frequency_band_points", 3))
        if 1 <= points <= 12:
            self.freq_band_points.setCurrentText(str(points))
        else:
            self.freq_band_points.setCurrentText("3")
        self.resolution.setText(f"{self.settings.resolution:.2f}")
        self.temperature.setText(f"{self.settings.temperature:.1f}")
        self.humidity.setText(f"{self.settings.humidity:.1f}")
        self.use_air_absorption.setChecked(self.settings.use_air_absorption)
        self.position_plot_length.setText(str(self.settings.position_x_axis))
        self.position_plot_width.setText(str(self.settings.position_y_axis))

        self.spl_plot_superposition.setChecked(self.settings.spl_plot_superposition)
        self.spl_plot_fem.setChecked(self.settings.spl_plot_fem)
        self.xaxis_plot_superposition.setChecked(self.settings.xaxis_plot_superposition)
        self.xaxis_plot_fem.setChecked(self.settings.xaxis_plot_fem)
        self.yaxis_plot_superposition.setChecked(self.settings.yaxis_plot_superposition)
        self.yaxis_plot_fem.setChecked(self.settings.yaxis_plot_fem)
        self.impulse_plot_superposition.setChecked(self.settings.impulse_plot_superposition)
        self.impulse_plot_bem.setChecked(self.settings.impulse_plot_bem)
        self.update_pressure_soundfield.setChecked(self.settings.update_pressure_soundfield)
        self.update_pressure_axisplot.setChecked(self.settings.update_pressure_axisplot)
        self.update_pressure_polarplot.setChecked(self.settings.update_pressure_polarplot)
        self.update_pressure_impulse.setChecked(self.settings.update_pressure_impulse)
        self.fem_particle_velocity_checkbox.setChecked(getattr(self.settings, "fem_compute_particle_velocity", True))

        # Update SPL Plot settings
        self.max_spl.setText(str(self.settings.colorbar_range['max']))
        self.min_spl.setText(str(self.settings.colorbar_range['min']))
        self.db_step.setText(str(self.settings.colorbar_range['step']))

        # 3D-Achsen-Ansicht
        if hasattr(self, "axis_3d_view"):
            self.axis_3d_view.setChecked(getattr(self.settings, "axis_3d_view", False))
        if hasattr(self, "axis_3d_transparency"):
            transparency = float(getattr(self.settings, "axis_3d_transparency", 10.0))
            self.axis_3d_transparency.setText(str(int(round(transparency))))


# ----- SIGNAL SET -----

    def set_combobox_to_frequency(self, combobox, freq_value):
        freq_str = f"{int(freq_value)} Hz"
        index = combobox.findText(freq_str)
        combobox.blockSignals(True)
        if index != -1:
            combobox.setCurrentIndex(index)
        else:
            print(f"[WARN] Wert {freq_str} nicht in ComboBox gefunden!")
        combobox.blockSignals(False)


# ----- SIGNAL HANDLERS -----

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

    def on_freq_band_points_changed(self):
        """
        Aktualisiert die Anzahl der Mittelungspunkte im Frequenzband und
        berechnet direkt die dazugehörigen Frequenzpunkte zwischen lower/upper.

        Sonderfall:
            - Wenn lower == upper → genau ein Frequenzpunkt, nämlich diese Frequenz.
        """
        try:
            text = self.freq_band_points.currentText()
            value = int(text)
            if value < 1:
                raise ValueError("frequency_band_points must be >= 1")

            # Anzahl Punkte in Settings übernehmen
            current_points = getattr(self.settings, "frequency_band_points", 3)
            if current_points != value:
                self.settings.frequency_band_points = value

            # Aktuelle Bandgrenzen aus den Settings lesen
            lower_freq = float(getattr(self.settings, "lower_calculate_frequency", 50.0))
            upper_freq = float(getattr(self.settings, "upper_calculate_frequency", 50.0))

            # Sonderfall: identische Frequenzen → genau ein Punkt
            if upper_freq == lower_freq or value == 1:
                freq_points = [float(upper_freq)]
            else:
                # Frequenzpunkte logarithmisch zwischen lower und upper verteilen
                # (akustisch sinnvoller als linear im Hz-Raster)
                freq_points = np.geomspace(lower_freq, upper_freq, num=value).tolist()

            # In Settings ablegen, damit die Berechnung später darauf zugreifen kann
            self.settings.frequency_band_frequencies = freq_points

            # Debug-Ausgabe: Zeige gewählte Frequenzen für das Band an
            try:
                freqs_str = ", ".join(f"{f:.2f} Hz" for f in freq_points)
            except Exception:
                freqs_str = str(freq_points)
            print(
                "[Settings] Frequency band selection:",
                f"lower={lower_freq} Hz, upper={upper_freq} Hz, points={value} -> [{freqs_str}]",
            )

        except ValueError:
            # Auf letzten gültigen Wert aus Settings zurücksetzen
            points = int(getattr(self.settings, "frequency_band_points", 3))
            if 1 <= points <= 12:
                self.freq_band_points.setCurrentText(str(points))
            else:
                self.freq_band_points.setCurrentText("3")

    def on_ColorizationMode_changed(self, index):
        # Dies bleibt gleich, da wir intern weiterhin "step" vs "linear" verwenden
        self.settings.colorization_mode = "Color step" if index == 0 else "Gradient"
        if hasattr(self.main_window, "plot_spl"):
            self.main_window.plot_spl()

    def on_fem_frequency_changed(self):
        text = self.fem_calculate_frequency.currentText()
        try:
            frequency = float(text.split()[0])
        except (ValueError, IndexError):
            self.set_combobox_to_frequency(
                self.fem_calculate_frequency,
                getattr(self.settings, "fem_calculate_frequency", 50)
            )
            return

        current_value = getattr(self.settings, "fem_calculate_frequency", None)
        if current_value == frequency:
            return

        self.settings.fem_calculate_frequency = frequency
        if hasattr(self.main_window, "mark_fem_frequency_dirty"):
            self.main_window.mark_fem_frequency_dirty()

        if getattr(self.settings, "spl_plot_fem", False):
            self.main_window.update_speaker_array_calculations()

    def on_calculation_option_changed(self, attribute_name, state):
        new_value = bool(state)
        if getattr(self.settings, attribute_name, False) == new_value:
            return

        setattr(self.settings, attribute_name, new_value)

        if attribute_name == "spl_plot_fem":
            draw_plots = getattr(self.main_window, "draw_plots", None)
            if draw_plots is not None and hasattr(draw_plots, "refresh_plot_mode_availability"):
                draw_plots.refresh_plot_mode_availability()

        is_auto_flag = attribute_name.startswith("update_pressure_")
        should_trigger = False

        if is_auto_flag:
            if not new_value:
                should_trigger = True
        else:
            if attribute_name in {"spl_plot_fem", "spl_plot_superposition"}:
                should_trigger = getattr(self.settings, "update_pressure_soundfield", True)
            elif attribute_name in {"xaxis_plot_fem", "xaxis_plot_superposition"}:
                should_trigger = getattr(self.settings, "update_pressure_axisplot", True)
            elif attribute_name in {"yaxis_plot_fem", "yaxis_plot_superposition"}:
                should_trigger = getattr(self.settings, "update_pressure_axisplot", True)
            elif attribute_name in {"impulse_plot_superposition", "impulse_plot_bem"}:
                should_trigger = getattr(self.settings, "update_pressure_impulse", True)
            else:
                should_trigger = True

        if should_trigger and hasattr(self.main_window, "update_speaker_array_calculations"):
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

    def initialize_air_density(self):
        """
        Initialisiert die Luftdichte beim Start basierend auf den gespeicherten Werten.
        """
        if hasattr(self.settings, 'temperature') and hasattr(self.settings, 'humidity'):
            air_density = self.function_toolbox.calculate_air_density(
                self.settings.temperature, 
                self.settings.humidity
            )
            self.settings.air_density = air_density

    def on_Temperature_changed(self):
        try:
            value = round(float(self.temperature.text()), 1)
            if self.settings.temperature != value:
                self.settings.temperature = value
                
                # 🌡️ Berechne Schallgeschwindigkeit
                speed_of_sound = self.function_toolbox.calculate_speed_of_sound(self.settings.temperature)
                self.settings.speed_of_sound = speed_of_sound
                
                # Berechne Luftdichte
                air_density = self.function_toolbox.calculate_air_density(
                    self.settings.temperature, 
                    self.settings.humidity
                )
                self.settings.air_density = air_density
                
                print(f"[INFO] Temperatur: {value}°C → c = {speed_of_sound:.2f} m/s, ρ = {air_density:.4f} kg/m³")
                
                # Trigger Neuberechnung
                self.main_window.update_speaker_array_calculations()
            self.temperature.setText(f"{value:.1f}")
        except ValueError:
            self.temperature.setText(f"{self.settings.temperature:.1f}")

    def on_Humidity_changed(self):
        try:
            value = round(float(self.humidity.text()), 1)
            if self.settings.humidity != value:
                self.settings.humidity = value
                
                # 🌡️ Berechne Schallgeschwindigkeit (Temperatur bleibt führend)
                speed_of_sound = self.function_toolbox.calculate_speed_of_sound(self.settings.temperature)
                self.settings.speed_of_sound = speed_of_sound
                
                # Berechne Luftdichte
                air_density = self.function_toolbox.calculate_air_density(
                    self.settings.temperature, 
                    self.settings.humidity
                )
                self.settings.air_density = air_density
                
                print(f"[INFO] Luftfeuchtigkeit: {value}% → c = {speed_of_sound:.2f} m/s, ρ = {air_density:.4f} kg/m³")
                
                # Trigger Neuberechnung
                self.main_window.update_speaker_array_calculations()
            self.humidity.setText(f"{value:.1f}")
        except ValueError:
            self.humidity.setText(f"{self.settings.humidity:.1f}")

    def on_AirAbsorption_changed(self, state):
        """Handler für Air Absorption Checkbox"""
        use_absorption = bool(state)
        if self.settings.use_air_absorption != use_absorption:
            self.settings.use_air_absorption = use_absorption
            status = "aktiviert" if use_absorption else "deaktiviert"
            print(f"[INFO] Luftabsorption {status}")
            # Trigger Neuberechnung
            self.main_window.update_speaker_array_calculations()

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
                    # Aktualisiere Overlays (Achsen im 3D-Plot neu zeichnen)
                    try:
                        draw_plots = getattr(self.main_window, "draw_plots", None)
                        if draw_plots and hasattr(draw_plots, "draw_spl_plotter"):
                            plotter = draw_plots.draw_spl_plotter
                            if hasattr(plotter, "update_overlays"):
                                plotter.update_overlays(self.settings, self.main_window.container)
                    except Exception as e:
                        print(f"[UiSettings] Fehler beim Aktualisieren der Achsenposition: {e}")
                    # Nur Achsenberechnung aktualisieren (X/Y-Achsenplots), ohne komplette SPL-Neuberechnung
                    try:
                        self.main_window.calculate_axes(update_plot=True)
                    except Exception as e:
                        print(f"[UiSettings] Fehler bei calculate_axes nach Positionsänderung (Länge): {e}")
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
                    # Aktualisiere Overlays (Achsen im 3D-Plot neu zeichnen)
                    try:
                        draw_plots = getattr(self.main_window, "draw_plots", None)
                        if draw_plots and hasattr(draw_plots, "draw_spl_plotter"):
                            plotter = draw_plots.draw_spl_plotter
                            if hasattr(plotter, "update_overlays"):
                                plotter.update_overlays(self.settings, self.main_window.container)
                    except Exception as e:
                        print(f"[UiSettings] Fehler beim Aktualisieren der Achsenposition: {e}")
                    # Nur Achsenberechnung aktualisieren (X/Y-Achsenplots), ohne komplette SPL-Neuberechnung
                    try:
                        self.main_window.calculate_axes(update_plot=True)
                    except Exception as e:
                        print(f"[UiSettings] Fehler bei calculate_axes nach Positionsänderung (Breite): {e}")
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

    def on_AxisTransparency_changed(self):
        """Setzt die Transparenz der Achsenfläche (0–100 %, 10 % Default) und aktualisiert Overlays."""
        try:
            value = float(self.axis_3d_transparency.text())
        except ValueError:
            value = getattr(self.settings, "axis_3d_transparency", 10.0)

        # Begrenze auf 0–100 %
        value = max(0.0, min(100.0, value))

        if getattr(self.settings, "axis_3d_transparency", 10.0) == value:
            # UI ggf. korrigieren, aber keine Neu-Zeichnung nötig
            self.axis_3d_transparency.setText(str(int(round(value))))
            return

        self.settings.axis_3d_transparency = value
        self.axis_3d_transparency.setText(str(int(round(value))))

        # Nur Overlays aktualisieren
        try:
            draw_plots = getattr(self.main_window, "draw_plots", None)
            if draw_plots and hasattr(draw_plots, "draw_spl_plotter"):
                plotter = draw_plots.draw_spl_plotter
                if hasattr(plotter, "update_overlays"):
                    plotter.update_overlays(self.settings, self.main_window.container)
        except Exception as e:
            print(f"[UiSettings] Fehler beim Aktualisieren der Achsentransparenz: {e}")



