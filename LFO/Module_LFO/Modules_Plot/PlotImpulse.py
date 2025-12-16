from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy, QToolBar, QAction
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QIcon
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
class DrawImpulsePlots(QWidget):
    view_changed = pyqtSignal(str)  # Signal für Ansichtsänderungen

    def __init__(self, settings, container, main_window=None):
        super().__init__()
        self.settings = settings
        self.container = container
        self.main_window = main_window
        self.canvases = {}
        self.tree_widget = None  # Wird später gesetzt
        self._scroll_update_pending = False
        
        # UI Elemente
        self.scroll_area_graphics = None
        self.scroll_widget_graphics = None
        self.scroll_layout_graphics = None
        self.ir_button = None
        self.at_button = None
        self.phase_button = None
        self.toolbar = None
        
        self.init_ui()

    def _current_view_type(self):
        if self.at_button and self.at_button.isChecked():
            return 'AT'
        return 'IR'

    def init_ui(self):
        """Initialisiert die UI-Elemente für die Plots"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(2)
        
        # Toggle-Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 2)
        button_layout.setSpacing(2)
        button_layout.setAlignment(Qt.AlignRight)
        
        # Erstelle Toggle-Buttons (nur IR und AT)
        self.ir_button = QPushButton("IR")
        self.at_button = QPushButton("AT")
        
        # Konfiguriere Buttons
        for button in [self.ir_button, self.at_button]:
            button.setCheckable(True)
            button.setFixedWidth(60)
            button_layout.addWidget(button)
        
        # Verbinde Button-Klicks
        self.ir_button.clicked.connect(lambda: self.toggle_view('IR'))
        self.at_button.clicked.connect(lambda: self.toggle_view('AT'))
        
        # Setze IR-Button standardmäßig aktiv
        self.ir_button.setChecked(True)
        
        main_layout.addWidget(button_widget)
        
        # Scrollbereich für Plots
        self.scroll_area_graphics = QScrollArea()
        self.scroll_area_graphics.setWidgetResizable(True)
        self.scroll_area_graphics.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Deaktiviere Mausrad-Scrollen auf dem QScrollArea selbst
        self.scroll_area_graphics.wheelEvent = lambda event: None
        
        self.scroll_widget_graphics = QWidget()
        self.scroll_layout_graphics = QVBoxLayout(self.scroll_widget_graphics)
        self.scroll_area_graphics.setWidget(self.scroll_widget_graphics)
        
        main_layout.addWidget(self.scroll_area_graphics)
        
        # Erstelle die zwei Haupt-Canvas
        self.create_main_canvases()
        
        # Erstelle eine Toolbar für die Haupt-Canvas mit Subplots
        if 'main' in self.canvases:
            # Erstelle Layout für die Haupt-Canvas mit Toolbar
            main_plot_layout = QVBoxLayout()
            self.create_toolbar()
            main_plot_layout.addWidget(self.toolbar)
            main_plot_layout.addWidget(self.canvases['main']['canvas'])
            
            # Erstelle Widget für das Haupt-Plot-Layout
            main_plot_widget = QWidget()
            main_plot_widget.setLayout(main_plot_layout)
            
            # Füge das Haupt-Plot-Widget zum Scroll-Layout hinzu
            self.scroll_layout_graphics.addWidget(main_plot_widget)
        else:
            # Fallback: Erstelle einfache Toolbar
            self.create_toolbar()
            main_layout.addWidget(self.toolbar)
        
        # Initialisiere leere Plot-Darstellung
        self.initialize_empty_plots()

    def _get_selected_snapshot_key(self):
        if self.main_window and hasattr(self.main_window, 'snapshot_engine'):
            snapshot_engine = self.main_window.snapshot_engine
            return getattr(snapshot_engine, 'selected_snapshot_key', None)
        return None

    def _get_snapshot_linewidth(self, snapshot_key, selected_key, default_width, selected_width):
        if selected_key and snapshot_key == selected_key:
            return selected_width
        return default_width

    def _is_current_simulation_visible(self):
        if hasattr(self.container, 'calculation_impulse'):
            impulse_data = self.container.calculation_impulse.get("aktuelle_simulation")
            if isinstance(impulse_data, dict):
                if "show_in_plot" in impulse_data:
                    return bool(impulse_data.get("show_in_plot", True))
                # Wenn Impulsdaten vorhanden sind (mindestens ein Messpunkt), zeige sie standardmäßig
                if any(key != "show_in_plot" for key in impulse_data.keys()):
                    return True

        if hasattr(self.container, 'calculation_axes'):
            current_axes = self.container.calculation_axes.get("aktuelle_simulation")
            if current_axes is not None:
                return current_axes.get("show_in_plot", True)
        return True

    def create_main_canvases(self):
        """Erstellt eine Canvas mit Subplots für Impulse/Phase/AT und Magnitude"""
        # Erstelle eine Canvas mit drei Subplots (wie in PlotDataImporter.py)
        # Verwende measurement_size für die Höhe der Subplots
        subplot_height = self.settings.measurement_size
        self.figure = Figure(figsize=(10, subplot_height * 3))
        impulse_ax = self.figure.add_subplot(3, 1, 1)
        phase_ax = self.figure.add_subplot(3, 1, 2)
        magnitude_ax = self.figure.add_subplot(3, 1, 3, sharex=phase_ax)
        main_canvas = FigureCanvas(self.figure)
        main_canvas.setFixedHeight(self.settings.impulse_plot_height * 3)  # Dreifache Höhe für drei Plots
        
        # Subplot-Anpassung (wie in PlotDataImporter.py)
        self.figure.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0.5)
        
        # Verlinke Phase und Magnitude für synchronisiertes Zoomen in X-Achse
        magnitude_ax.sharex(phase_ax)
        
        # Entferne X-Tick-Labels vom mittleren Plot (Phase), da er mit Magnitude gelinkt ist
        phase_ax.tick_params(labelbottom=False)
        
        # Speichere die Haupt-Canvas mit Subplots
        self.canvases['main'] = {
            'canvas': main_canvas,
            'impulse_ax': impulse_ax,
            'phase_ax': phase_ax,
            'magnitude_ax': magnitude_ax
        }

    def create_toolbar(self):
        """Erstellt eine Toolbar für die Haupt-Canvas mit Subplots"""
        if 'main' in self.canvases:
            # Erstelle NavigationToolbar für die Haupt-Canvas
            main_canvas = self.canvases['main']['canvas']
            self.toolbar = NavigationToolbar2QT(main_canvas, self)
            self.toolbar.setIconSize(QSize(16, 16))
        else:
            # Fallback: Erstelle leere Toolbar
            self.toolbar = QToolBar()
            self.toolbar.setMovable(False)
            self.toolbar.setFloatable(False)
            self.toolbar.setIconSize(QSize(16, 16))
    
    def initialize_empty_plots(self):
        """Initialisiert alle Plots mit sinnvoller Leer-Darstellung"""
        if 'main' not in self.canvases:
            return
        
        # Hole die Achsen
        impulse_ax = self.canvases['main']['impulse_ax']
        phase_ax = self.canvases['main']['phase_ax']
        magnitude_ax = self.canvases['main']['magnitude_ax']
        
        # Plot 1: Impulsantwort / Arrival Times
        impulse_ax.clear()
        impulse_ax.set_xlabel('Time [ms]', fontsize=8)
        impulse_ax.set_ylabel('Impulse response [%]', fontsize=8)
        impulse_ax.set_ylim(-1, 1)
        impulse_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        impulse_ax.grid(True, which='both', linestyle=':', alpha=0.5)
        impulse_ax.tick_params(axis='both', labelsize=8)
        # Textposition in der Mitte des Plots
        impulse_ax.text(0.5, 0.5, 'No measurement data', 
                       transform=impulse_ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Plot 2: Phase
        phase_ax.set_xscale('linear')  # Temporär auf linear setzen
        phase_ax.clear()
        phase_ax.set_xscale('log')     # Zurück zu log
        phase_ax.set_xlabel('Frequency [Hz]', fontsize=8)
        phase_ax.set_ylabel('Phase [deg]', fontsize=8)
        phase_ax.set_xlim(15, 400)
        phase_ax.set_ylim(-180, 180)
        freq_ticks = [20, 40, 60, 80, 100, 200, 400]
        phase_ax.set_xticks(freq_ticks)
        phase_ax.set_xticklabels([str(x) for x in freq_ticks])
        phase_ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        phase_ax.grid(True, which='both', linestyle=':', alpha=0.5)
        phase_ax.tick_params(axis='both', labelsize=8, labelbottom=True)
        phase_ax.text(0.5, 0.5, 'No measurement data', 
                     transform=phase_ax.transAxes,
                     ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Plot 3: Magnitude
        magnitude_ax.set_xscale('linear')  # Temporär auf linear setzen
        magnitude_ax.clear()
        magnitude_ax.set_xscale('log')     # Zurück zu log
        magnitude_ax.set_xlabel('Frequency [Hz]', fontsize=8)
        magnitude_ax.set_ylabel('Magnitude [dB]', fontsize=8)
        magnitude_ax.set_xlim(15, 400)
        magnitude_ax.set_ylim(-36, 0)
        magnitude_ax.set_xticks(freq_ticks)
        magnitude_ax.set_xticklabels([str(x) for x in freq_ticks])
        magnitude_ax.set_yticks(np.arange(-36, 6, 6))
        magnitude_ax.grid(True, which='both', linestyle=':', alpha=0.5)
        magnitude_ax.tick_params(axis='both', labelsize=8, labelbottom=True)
        magnitude_ax.text(0.5, 0.5, 'No measurement data', 
                         transform=magnitude_ax.transAxes,
                         ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Zeichne Canvas
        self.canvases['main']['canvas'].draw()

    def add_canvas_for_key(self, key):
        """Fügt neue Canvas für einen Messpunkt hinzu - wird nicht mehr verwendet"""
        # Keine Canvas mehr erstellen, da wir nur zwei Haupt-Canvas haben
        pass

    def update_plot_impulse(self):
        """Aktualisiert die Plots basierend auf dem aktiven View-Type und dem ausgewählten Messpunkt"""
        calculation_impulse = self.container.get_calculation_impulse().get("aktuelle_simulation", {})
        selected_key = self.get_selected_measurement_point()
        has_snapshot_data = False
        if selected_key:
            for snapshot_key, snapshot_data in self.container.calculation_axes.items():
                if (
                    snapshot_key != "aktuelle_simulation"
                    and snapshot_data.get("show_in_plot", False)
                    and (
                        f"impulse_{selected_key}" in snapshot_data
                        or f"magnitude_{selected_key}" in snapshot_data
                        or f"phase_{selected_key}" in snapshot_data
                    )
                ):
                    has_snapshot_data = True
                    break

        if not selected_key:
            self.initialize_empty_plots()
            return

        if not calculation_impulse and not has_snapshot_data:
            self.initialize_empty_plots()
            return

        view_type = 'IR' if not self.at_button.isChecked() else 'AT'

        if 'main' in self.canvases:
            self.canvases['main']['canvas'].setFixedHeight(self.settings.impulse_plot_height * 3)

        data = calculation_impulse.get(selected_key, None) if calculation_impulse else None

        phase_fn = getattr(self, "plot_phase", None)
        impulse_fn = getattr(self, "plot_impulse_response", None)
        arrival_fn = getattr(self, "plot_arrival_times", None)
        magnitude_fn = getattr(self, "plot_magnitude", None)

        def _call_plot(fn, *args):
            if callable(fn):
                fn(*args)

        try:
            if data and data.get('show_in_plot', True):
                if view_type == 'IR':
                    _call_plot(impulse_fn, 'main', data, selected_key)
                else:
                    _call_plot(arrival_fn, 'main', data, selected_key)
                _call_plot(phase_fn, 'main', data, selected_key)
                _call_plot(magnitude_fn, 'main', data, selected_key)
            elif has_snapshot_data:
                if view_type == 'IR':
                    _call_plot(impulse_fn, 'main', None, selected_key)
                else:
                    _call_plot(arrival_fn, 'main', None, selected_key)
                _call_plot(phase_fn, 'main', None, selected_key)
                _call_plot(magnitude_fn, 'main', None, selected_key)
        except Exception as e:
            print(f"Fehler beim Plotten für Key {selected_key}: {str(e)}")

        if 'main' in self.canvases:
            canvas = self.canvases['main']['canvas']
            canvas.draw_idle()

        self._schedule_scroll_update()

    def get_selected_measurement_point(self):
        """Ermittelt den ausgewählten Messpunkt aus dem TreeWidget"""
        if not self.tree_widget:
            return None
        
        current_item = self.tree_widget.currentItem()
        if current_item:
            key = current_item.data(0, Qt.UserRole)
            return key
        
        return None

    def reset_zoom_all(self):
        """Setzt Zoom für alle Plots zurück"""
        if 'main' in self.canvases:
            canvas = self.canvases['main']['canvas']
            for ax_type in ['impulse_ax', 'phase_ax', 'magnitude_ax']:
                ax = self.canvases['main'][ax_type]
                ax.set_xlim(auto=True)
                ax.set_ylim(auto=True)
            canvas.draw_idle()

    def auto_scale_all(self):
        """Automatische Skalierung für alle Plots"""
        if 'main' in self.canvases:
            for ax_type in ['impulse_ax', 'phase_ax', 'magnitude_ax']:
                ax = self.canvases['main'][ax_type]
                ax.autoscale_view()
            self.canvases['main']['canvas'].draw_idle()

    def plot_impulse_response(self, key, data, selected_key=None):
        """Plottet die Impulsantworten"""
        if key not in self.canvases:
            return
        
        canvas = self.canvases[key]['canvas']
        ax = self.canvases[key]['impulse_ax']
        ax.clear()

        current_visible = self._is_current_simulation_visible()
        selected_snapshot_key = self._get_selected_snapshot_key()
        current_linewidth = self._get_snapshot_linewidth(
            "aktuelle_simulation",
            selected_snapshot_key,
            default_width=0.5,
            selected_width=1.5,
        )
        
        # Plotte aktuelle Daten nur wenn vorhanden
        if current_visible and data and data.get('show_in_plot', True) and 'impulse_response' in data:
            # Plot der Summenimpulsantwort (immer zuvorderst, Standard-Dicke und Standard-Farbe)
            if data['impulse_response']['combined_impulse'] is not None:
                combined = data['impulse_response']['combined_impulse']
                ax.plot(
                    combined['time'],
                    combined['spl'],
                    linewidth=current_linewidth,
                    zorder=10
                )
            
            # Plot der einzelnen Impulsantworten (dünner, gestrichelt, im Hintergrund)
            for spl, time, color in zip(
                data['impulse_response']['spl'],
                data['impulse_response']['time'],
                data['impulse_response']['color']):
                ax.plot(time, spl, color=color, linewidth=0.8, alpha=0.4, linestyle='--', zorder=1)
        
        # Plot der Snapshot-Daten für den aktuellen Messpunkt (nur Summen-Impulse)
        for snapshot_key, snapshot_data in self.container.calculation_axes.items():
            if (snapshot_key != "aktuelle_simulation" and 
                snapshot_data.get("show_in_plot", False)):  # Prüfe show_in_plot
                
                # Suche nach Summen-Impulse-Daten für den aktuellen Messpunkt in diesem Snapshot
                impulse_key = f"impulse_{selected_key}"
                
                if impulse_key in snapshot_data:
                    line_width = self._get_snapshot_linewidth(
                        snapshot_key,
                        selected_snapshot_key,
                        default_width=0.5,
                        selected_width=1.5,
                    )
                    ax.plot(snapshot_data[impulse_key]['time'],
                                 snapshot_data[impulse_key]['spl'],
                                 color=snapshot_data['color'],
                                 linewidth=line_width,
                                 alpha=0.7,
                                 label=f"{snapshot_key}")
        
        # Plot-Konfiguration mit angepasster Schriftgröße
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Achsenbeschriftungen nur setzen, wenn Daten vorhanden sind (wie in PlotDataImporter.py)
        if data and 'impulse_response' in data:
            ax.set_xlabel('Time [ms]', fontsize=8)
            ax.set_ylabel('Impulse response [%]', fontsize=8)

        canvas.draw_idle()

    def _schedule_scroll_update(self):
        """Aktualisiert Scroll-Layout entkoppelt vom aktuellen Event-Loop-Zugriff."""
        if self.scroll_layout_graphics is None:
            return

        if self._scroll_update_pending:
            return

        self._scroll_update_pending = True

        def _refresh():
            self._scroll_update_pending = False
            if self.scroll_layout_graphics:
                self.scroll_layout_graphics.update()

        QTimer.singleShot(0, _refresh)

    def plot_arrival_times(self, key, data, selected_key=None):
        """Plottet die Ankunftszeiten"""
        if key not in self.canvases:
            return
        
        canvas = self.canvases[key]['canvas']
        ax = self.canvases[key]['impulse_ax']  # Verwende impulse_ax für AT
        ax.clear()
        
        # Plotte aktuelle Daten nur wenn vorhanden
        current_visible = self._is_current_simulation_visible()
        selected_snapshot_key = self._get_selected_snapshot_key()
        current_linewidth = self._get_snapshot_linewidth(
            "aktuelle_simulation",
            selected_snapshot_key,
            default_width=0.5,
            selected_width=1.5,
        )
        if current_visible and data and data.get('show_in_plot', True) and 'arrival_times' in data:
            # Plot der Arrival Times
            for time, spl_max, color in zip(
                data['arrival_times']['time'],
                data['arrival_times']['spl_max'],
                data['arrival_times']['color']):
                
                normalized_spl = spl_max - max(data['arrival_times']['spl_max'])
                
                if normalized_spl > self.settings.impulse_min_spl:
                    if abs(normalized_spl) < 0.1:
                        ax.vlines(
                            x=time,
                            ymin=self.settings.impulse_min_spl,
                            ymax=0,
                            color=color,
                            alpha=0.7,
                            linewidth=current_linewidth
                        )
                    else:
                        ax.vlines(
                            x=time,
                            ymin=self.settings.impulse_min_spl,
                            ymax=normalized_spl,
                            color=color,
                            alpha=0.7,
                            linewidth=current_linewidth
                        )
        
        # Plot der Snapshot-Daten für den aktuellen Messpunkt
        for snapshot_key, snapshot_data in self.container.calculation_axes.items():
            if (snapshot_key != "aktuelle_simulation" and 
                snapshot_data.get("show_in_plot", False)):
                
                # Suche nach Arrival-Daten für den aktuellen Messpunkt in diesem Snapshot
                arrival_key = f"arrival_{selected_key}"
                
                if arrival_key in snapshot_data:
                    times = snapshot_data[arrival_key]['time']
                    spl_max = snapshot_data[arrival_key]['spl_max']
                    max_spl = max(spl_max)
                    
                    line_width = self._get_snapshot_linewidth(
                        snapshot_key,
                        selected_snapshot_key,
                        default_width=0.5,
                        selected_width=1.5,
                    )

                    for t, s in zip(times, spl_max):
                        normalized_s = s - max_spl
                        if normalized_s > self.settings.impulse_min_spl:
                            ax.vlines(x=t, ymin=self.settings.impulse_min_spl, ymax=normalized_s,
                                           color=snapshot_data['color'],
                                           alpha=0.5, linewidth=line_width)
        
        # Generiere y-Ticks in 3dB-Schritten
        y_ticks = list(range(int(self.settings.impulse_min_spl), 1, 3))
        if 0 not in y_ticks:  # Füge 0 hinzu, wenn nicht bereits vorhanden
            y_ticks.append(0)
        
        # Plot-Konfiguration mit angepasster Schriftgröße
        ax.set_ylim(self.settings.impulse_min_spl, 0)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Achsenbeschriftungen nur setzen, wenn Daten vorhanden sind (wie in PlotDataImporter.py)
        if data and 'arrival_times' in data:
            ax.set_xlabel('Time [ms]', fontsize=8)
            ax.set_ylabel('Relative SPL [dB]', fontsize=8)
        
        canvas.draw_idle()

    def plot_phase(self, key, data, selected_key=None):
        """Plottet die Phasenbeziehungen"""
        if key not in self.canvases:
            return
        
        canvas = self.canvases[key]['canvas']
        ax = self.canvases[key]['phase_ax']  # Phase im mittleren Plot
        self._clear_semilog_axes(ax)
        ax.set_xscale('log')
        
        # Bestimme die Frequenzgrenzen
        freq_min = float('inf')
        freq_max = float('-inf')

        selected_snapshot_key = self._get_selected_snapshot_key()
        current_linewidth = self._get_snapshot_linewidth(
            "aktuelle_simulation",
            selected_snapshot_key,
            default_width=0.5,
            selected_width=1.5,
        )
        
        # Plotte aktuelle Daten nur wenn vorhanden
        current_visible = self._is_current_simulation_visible()
        if current_visible and data and data.get('show_in_plot', True) and 'phase_response' in data:
            phase_data = data['phase_response']
            
            # Plot der kombinierten Phase (immer zuvorderst, Standard-Dicke und Standard-Farbe)
            if phase_data['combined_phase']:
                combined_phase_deg = np.rad2deg(phase_data['combined_phase']['phase'])
                combined_phase_deg = np.clip(((combined_phase_deg + 180) % 360) - 180, -180, 180)
                
                ax.semilogx(phase_data['combined_phase']['freq'],
                                  combined_phase_deg,
                                  linewidth=current_linewidth,
                                  zorder=10)
                # (Konsolen-Dumps entfernt)
            
            # Plot der individuellen Phasen (dünner, gestrichelt, im Hintergrund)
            for freq, phase, color in zip(
                phase_data['freq'],
                phase_data['phase'],
                phase_data['color']):
                freq_min = min(freq_min, min(freq))
                freq_max = max(freq_max, max(freq))
            
                # Verbesserte Phasenumwandlung mit exakten Grenzen
                phase_deg = self._wrap_phase(np.rad2deg(phase))
                
                ax.semilogx(freq, phase_deg, color=color, linewidth=0.8, alpha=0.4, linestyle='--', zorder=1)
        
        # Plot der Snapshot-Daten für den aktuellen Messpunkt (nur Summen-Phase)
        for snapshot_key, snapshot_data in self.container.calculation_axes.items():
            if (snapshot_key != "aktuelle_simulation" and 
                snapshot_data.get("show_in_plot", False)):
                
                # Suche nach Summen-Phase-Daten für den aktuellen Messpunkt in diesem Snapshot
                phase_key = f"phase_{selected_key}"
                
                if phase_key in snapshot_data:
                    phase_deg = self._wrap_phase(np.rad2deg(snapshot_data[phase_key]['phase']))
                    line_width = self._get_snapshot_linewidth(
                        snapshot_key,
                        selected_snapshot_key,
                        default_width=0.5,
                        selected_width=1.5,
                    )
                    ax.semilogx(snapshot_data[phase_key]['freq'],
                                     phase_deg,
                                     color=snapshot_data['color'],
                                     linewidth=line_width,
                                     alpha=0.7,
                                     label=f"{snapshot_key}")
        
        # Plot-Konfiguration mit angepasster Schriftgröße
        # Stelle sicher, dass freq_max positiv ist für log-scale
        upper_limit = freq_max if freq_max > 0 and freq_max != float('inf') else 400
        ax.set_xlim(15, upper_limit)
        
        # Frequenz-Ticks definieren
        freq_ticks = [20, 40, 60, 80, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        visible_ticks = [t for t in freq_ticks if t <= upper_limit]
        ax.set_xticks(visible_ticks, [str(x) for x in visible_ticks])
        
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        ax.tick_params(axis='both', labelsize=8, labelbottom=True)  # Zeige X-Tick-Labels für Phase
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Achsenbeschriftungen nur setzen, wenn Daten vorhanden sind (wie in PlotDataImporter.py)
        if data and 'phase_response' in data:
            ax.set_ylabel('Phase [deg]', fontsize=8)
            ax.set_xlabel('Frequency [Hz]', fontsize=8)  # X-Achsenbeschriftung für Phase

        canvas.draw_idle()

    @staticmethod
    def _clear_semilog_axes(ax):
        current_xlim = ax.get_xlim()
        if current_xlim[0] <= 0:
            current_xlim = (1.0, max(current_xlim[1], 10.0))
        ax.set_xscale('linear')
        ax.set_xlim(*current_xlim)
        ax.clear()

    @staticmethod
    def _wrap_phase(values):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=float)
        wrapped = ((values + 180.0) % 360.0) - 180.0
        return np.clip(wrapped, -180.0, 180.0)

    def plot_magnitude(self, key, data, selected_key=None):
        if key not in self.canvases:
            print(f"Canvas nicht gefunden für Key: {key}")
            return
        
        canvas = self.canvases[key]['canvas']
        ax = self.canvases[key]['magnitude_ax']
        # Clear ohne Log-Scale Warnung
        ax.set_xscale('linear')  # Temporär linear setzen
        ax.clear()
        ax.set_xscale('log')  # Zurück zu log-scale
        
        selected_snapshot_key = self._get_selected_snapshot_key()
        current_linewidth = self._get_snapshot_linewidth(
            "aktuelle_simulation",
            selected_snapshot_key,
            default_width=0.5,
            selected_width=1.5,
        )
        
        # Plotte aktuelle Daten nur wenn vorhanden
        current_visible = self._is_current_simulation_visible()
        if current_visible and data and data.get('show_in_plot', True) and 'magnitude_data' in data:
            mag_data = data['magnitude_data']
            
            # Plot der Summen-Magnitude (immer zuvorderst, Standard-Dicke und Standard-Farbe)
            if (mag_data.get('frequency') is not None and 
                mag_data.get('magnitude') is not None):
                ax.semilogx(mag_data['frequency'],
                                  mag_data['magnitude'],
                                  linewidth=current_linewidth,
                                  zorder=10)
                # (Konsolen-Dumps entfernt)
            
            # Plot der individuellen Magnituden (dünner, gestrichelt, im Hintergrund)
            for speaker_name, speaker_data in mag_data['individual_magnitudes'].items():
                ax.semilogx(mag_data['frequency'],
                                  speaker_data['magnitude'],
                                  color=speaker_data['color'],
                                  linewidth=0.8,
                                  alpha=0.4, linestyle='--', zorder=1)

        # Plot der Snapshot-Daten für den aktuellen Messpunkt
        selected_snapshot_key = self._get_selected_snapshot_key()
        for snapshot_key, snapshot_data in self.container.calculation_axes.items():
            if (snapshot_key != "aktuelle_simulation" and 
                snapshot_data.get("show_in_plot", False)):
                
                # Suche nach Magnitude-Daten für den aktuellen Messpunkt in diesem Snapshot
                # Verwende selected_key direkt für die Snapshot-Daten-Abfrage
                mag_key = f"magnitude_{selected_key}"
                
                if mag_key in snapshot_data:
                    line_width = self._get_snapshot_linewidth(
                        snapshot_key,
                        selected_snapshot_key,
                        default_width=0.5,
                        selected_width=1.5,
                    )
                    ax.semilogx(snapshot_data[mag_key]['frequency'],
                                     snapshot_data[mag_key]['magnitude'],
                                     color=snapshot_data['color'],
                                     linewidth=line_width,
                                     alpha=0.7)
        
        # Y-Achse: Maximum plus 42dB Bereich mit 6dB Raster
        all_magnitudes = []
        
        # Sammle Magnitudes aus aktuellen Daten
        if data and 'magnitude_data' in data:
            mag_data = data['magnitude_data']
            if mag_data.get('magnitude') is not None:
                all_magnitudes.extend(list(mag_data['magnitude']))
        
        # Füge Snapshot-Daten für den aktuellen Messpunkt hinzu
        mag_key = f"magnitude_{selected_key}"
        for snapshot_key, snapshot_data in self.container.calculation_axes.items():
            if (snapshot_key != "aktuelle_simulation" and 
                snapshot_data.get("show_in_plot", False) and
                mag_key in snapshot_data):
                all_magnitudes.extend(snapshot_data[mag_key]['magnitude'])
        
        # Berechne Y-Achsen-Grenzen
        if all_magnitudes:
            mag_range = (min(all_magnitudes), max(all_magnitudes))
            y_max = np.ceil(mag_range[1] / 6) * 6
            y_min = y_max - 36
        else:
            # Fallback wenn keine Daten
            y_max = 0
            y_min = -36
        
        y_ticks = np.arange(y_min, y_max + 6, 6)
        
        # Frequenz-Ticks definieren
        freq_ticks = [20, 40, 60, 80, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        if data and 'magnitude_data' in data:
            mag_data = data['magnitude_data']
            if mag_data.get('frequency') is not None and len(mag_data['frequency']) > 0:
                freq_max = max(mag_data['frequency'])
            else:
                freq_max = 400  # Fallback-Wert für Impulse
        else:
            freq_max = 400  # Fallback-Wert für Impulse
        visible_ticks = [t for t in freq_ticks if t <= freq_max]
        
        # Plot-Konfiguration (unterer Plot behält X-Achsenbeschriftungen)
        ax.set_xlim(15, freq_max)
        ax.set_xticks(visible_ticks)
        ax.set_xticklabels([str(x) for x in visible_ticks])
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=8, labelbottom=True)  # Zeige X-Tick-Labels für Magnitude
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # Achsenbeschriftungen nur setzen, wenn Daten vorhanden sind (wie in PlotDataImporter.py)
        if data and 'magnitude_data' in data:
            ax.set_xlabel('Frequency [Hz]', fontsize=8)
            ax.set_ylabel('Magnitude [dB]', fontsize=8)
        
        canvas.draw_idle()

    def toggle_view(self, view_type):
        """Wird aufgerufen, wenn ein Toggle-Button geklickt wird"""
        # Setze alle Buttons zurück
        for button in [self.ir_button, self.at_button]:
            button.setChecked(False)
        
        # Setze den geklickten Button
        if view_type == 'IR':
            self.ir_button.setChecked(True)
        elif view_type == 'AT':
            self.at_button.setChecked(True)
        
        # Aktualisiere die Plots
        self.update_plot_impulse()



