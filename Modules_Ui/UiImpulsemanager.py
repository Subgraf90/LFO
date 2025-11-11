from PyQt5.QtCore import Qt
from Module_LFO.Modules_Window.WindowImpulseWidget import ImpulseInputDockWidget
from Module_LFO.Modules_Calculate.ImpulseCalculator import ImpulseCalculator

class ImpulseManager:
    def __init__(self, main_window, settings, container):
        self.main_window = main_window
        self.settings = settings
        self.container = container
        self.impulse_input_dock_widget = None

    def show_impulse_input_dock_widget(self):
        if not self.impulse_input_dock_widget:
            from Module_LFO.Modules_Window.WindowImpulseWidget import ImpulseInputDockWidget
            self.impulse_input_dock_widget = ImpulseInputDockWidget(
                self.main_window, self.settings, self.container, self.container.calculation_impulse
            )
            self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.impulse_input_dock_widget)
            self.impulse_input_dock_widget.init_ui()
            
            # Setze die Größe des DockWidgets
            # Option 1: Feste Größe
            # self.impulse_input_dock_widget.setFixedWidth(800)
            
            # Option 2: Mindest-/Maximalgröße
            # self.impulse_input_dock_widget.setMinimumWidth(600)
            # self.impulse_input_dock_widget.setMaximumWidth(1200)
            
            # Option 3: Initiale Größe (kann vom Benutzer geändert werden)
            self.impulse_input_dock_widget.resize(600, 800)
        
        self.impulse_input_dock_widget.initialize_measurement_points()
        self.impulse_input_dock_widget.show()
        self.update_calculation_impulse()

    def close_impulse_input_dock_widget(self):
        """Schließt und zerstört das Widget vollständig"""
        if self.impulse_input_dock_widget:
            # Schließe das Widget
            self.impulse_input_dock_widget.close()
            # Entferne es aus dem DockWidget-Bereich
            self.main_window.removeDockWidget(self.impulse_input_dock_widget)
            # Zerstöre das Widget
            self.impulse_input_dock_widget.deleteLater()
            # Setze die Referenz auf None
            self.impulse_input_dock_widget = None

    def update_calculation_impulse(self):
        """Aktualisiert die Impulsberechnungen nur wenn das Widget aktiv ist"""
        # Prüfe ob Widget existiert und aktiv ist
        if not (self.impulse_input_dock_widget and 
                self.impulse_input_dock_widget.isVisible()):
            return
            
        # Berechne die Impulse
        calculator_instance = ImpulseCalculator(self.settings, self.container.data)
        calculator_instance.set_data_container(self.container)  # 🚀 ERFORDERLICH für optimierte Balloon-Daten!
        calculation_result = calculator_instance.calculate_impulse()
        self.container.set_calculation_impulse(calculator_instance.calculation, "aktuelle_simulation")
                
        # Update Plot
        self.update_plot_impulse()

    def update_plot_impulse(self):
        """Aktualisiert die Plots nur wenn das Widget aktiv ist"""
        if (self.impulse_input_dock_widget and 
            self.impulse_input_dock_widget.isVisible()):
            self.impulse_input_dock_widget.plot_widget.update_plot_impulse()
