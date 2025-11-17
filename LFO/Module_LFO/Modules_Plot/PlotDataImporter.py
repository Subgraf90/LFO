import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle, Arrow
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QSize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import signal


class Draw_Plots_DataImporter():
    def __init__(self):
        super().__init__()

        # Hauptplots - Reihenfolge wie in PlotImpulse.py: IR/AT, Phase, Magnitude
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.figure.set_constrained_layout(True)
        
        # Verlinke Phase und Magnitude für synchronisiertes Zoomen in X-Achse (wie in PlotImpulse.py)
        self.ax3.sharex(self.ax2)
        
        # Entferne X-Tick-Labels vom mittleren Plot (Phase), da er mit Magnitude gelinkt ist
        self.ax2.tick_params(labelbottom=False)
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, None)
        self.toolbar.setIconSize(QSize(16, 16))
        
        # Polar Plot mit eigener Figure
        self.polar_figure = plt.figure(figsize=(8, 8))
        self.polar_figure.set_constrained_layout(True)
        self.polar_ax = self.polar_figure.add_subplot(111, projection='polar')
        self.polar_canvas = FigureCanvas(self.polar_figure)
        self.polar_toolbar = NavigationToolbar(self.polar_canvas, None)
        self.polar_toolbar.setIconSize(QSize(16, 16))
        
        # Grundeinstellungen für Polar Plot
        self.polar_ax.set_theta_zero_location("N")  # 0° ist oben
        self.polar_ax.set_theta_direction(-1)       # Im Uhrzeigersinn
        self.polar_ax.grid(True, linestyle='-', alpha=0.2)
        
        # Winkel-Ticks alle 45°
        angles_deg = np.arange(0, 360, 45)
        self.polar_ax.set_xticks(np.deg2rad(angles_deg))
        self.polar_ax.set_xticklabels([f'{int(angle)}°' for angle in angles_deg])
        
        # Initial leeren Plot anzeigen
        self.polar_ax.set_title("Polar Pattern")
        self.polar_canvas.draw()
        
        # Cabinet Plot mit eigener Figure
        self.cabinet_figure = plt.figure(figsize=(8, 8))
        self.cabinet_figure.set_constrained_layout(True)
        self.cabinet_ax = self.cabinet_figure.add_subplot(111)
        self.cabinet_canvas = FigureCanvas(self.cabinet_figure)
        self.cabinet_toolbar = NavigationToolbar(self.cabinet_canvas, None)
        self.cabinet_toolbar.setIconSize(QSize(16, 16))
        
        # Grundeinstellungen für Cabinet Plot
        self.cabinet_ax.set_aspect('equal')
        self.cabinet_ax.grid(True, alpha=0.2)
        self.cabinet_ax.set_xlabel('Width (m)')
        self.cabinet_ax.set_ylabel('Depth (m)')
        self.cabinet_ax.set_title('Speaker Cabinet Layout')
        
        # Initial leeren Plot anzeigen
        self.cabinet_ax.set_xlim(0, 1)
        self.cabinet_ax.set_ylim(0, 1)
        self.cabinet_canvas.draw()
        
        # Balloon Plot mit eigener Figure
        self.balloon_figure = plt.figure(figsize=(10, 10))
        self.balloon_figure.set_constrained_layout(False)
        self.balloon_ax = self.balloon_figure.add_subplot(111, projection='3d')
        self.balloon_canvas = FigureCanvas(self.balloon_figure)
        self.balloon_toolbar = NavigationToolbar(self.balloon_canvas, None)
        self.balloon_toolbar.setIconSize(QSize(16, 16))
        
        # Frequenzauswahl für Balloon Plot
        self.balloon_freq_selector_widget = QWidget()
        self.balloon_freq_selector_layout = QHBoxLayout(self.balloon_freq_selector_widget)
        self.balloon_freq_selector_label = QLabel("Frequenz:")
        self.balloon_freq_selector = QComboBox()
        self.balloon_freq_selector.setMinimumWidth(100)
        self.balloon_freq_selector_layout.addWidget(self.balloon_freq_selector_label)
        self.balloon_freq_selector_layout.addWidget(self.balloon_freq_selector)
        self.balloon_freq_selector_layout.addStretch()
        
        # Verbinde Frequenzauswahl mit Update-Funktion
        self.balloon_freq_selector.currentIndexChanged.connect(self.update_balloon_plot_with_selected_freq)
        
        # Dynamikbereich-Auswahl für Balloon und Polar Plot
        self.dynamic_range_label = QLabel("Range (dB):")
        self.dynamic_range_selector = QComboBox()
        self.dynamic_range_selector.addItems(['-42 dB', '-36 dB', '-30 dB', '-24 dB', '-18 dB', '-12 dB', '-6 dB'])
        self.dynamic_range_selector.setCurrentIndex(0)  # Default: -42 dB
        self.dynamic_range_selector.setMinimumWidth(100)
        self.balloon_freq_selector_layout.addWidget(self.dynamic_range_label)
        self.balloon_freq_selector_layout.addWidget(self.dynamic_range_selector)
        
        # Verbinde Dynamikbereich-Auswahl mit Update-Funktion
        self.dynamic_range_selector.currentIndexChanged.connect(self.update_dynamic_range)
        
        # Speichere die letzten Daten für Frequenzaktualisierung
        self.last_balloon_data = None
        self.last_polar_data = None
        
        # Speichere aktuellen Dynamikbereich (Standardwert: -42 dB)
        self.dynamic_range_db = 42
        
        # Feintuning der Plot-Darstellung
        self.adjust_subplots()
        
        # Initialisiere alle Plots mit sinnvoller Leer-Darstellung
        self.initialize_empty_plots()


    def adjust_subplots(self):
        """Passt die Subplot-Darstellung an - wie in PlotImpulse.py"""
        # Nutze constrained_layout für automatische Anpassung der Ränder,
        # um abgeschnittene Beschriftungen zu vermeiden.
        try:
            self.figure.set_constrained_layout(True)
        except Exception:
            # Fallback
            self.figure.tight_layout()
    
    def initialize_empty_plots(self):
        """Initialisiert alle Plots mit sinnvoller Leer-Darstellung"""
        # --- Haupt-Plots (IR/Phase/Magnitude) ---
        
        # Plot 1: Impulsantwort
        self.ax1.clear()
        self.ax1.set_xlabel('Time [ms]', fontsize=8)
        self.ax1.set_ylabel('Impulse response [%]', fontsize=8)
        self.ax1.set_xlim(0, 500)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, which='both', linestyle=':', alpha=0.5)
        self.ax1.tick_params(axis='both', labelsize=8)
        self.ax1.text(0.5, 0.5, 'No data loaded', 
                     transform=self.ax1.transAxes,
                     ha='center', va='center', fontsize=10, color='gray', alpha=0.5)
        
        # Plot 2: Phase
        try:
            self.ax2.set_xscale('linear')
        except Exception:
            pass
        self.ax2.clear()
        self.ax2.set_xlim(10, 400)
        self.ax2.set_xscale('log')
        self.ax2.set_title('Phase response', fontsize=9)
        self.ax2.set_xlabel('Frequency [Hz]', fontsize=8)
        self.ax2.set_ylabel('Phase [deg]', fontsize=8)
        self.ax2.set_ylim(-180, 180)
        freq_ticks = [20, 40, 60, 80, 100, 200, 400]
        # set_xticks würde ValueError auslösen, solange die Achse noch keine Größe hat.
        # Sobald echte Daten geladen sind, setzt update_plot() die Ticks neu.
        self.ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        self.ax2.grid(True, which='both', linestyle=':', alpha=0.5)
        self.ax2.tick_params(axis='both', labelsize=8, labelbottom=False)
        self.ax2.set_title('Phase response (no data)', fontsize=9, color='gray')
        
        # Plot 3: Magnitude
        try:
            self.ax3.set_xscale('linear')
        except Exception:
            pass
        self.ax3.clear()
        self.ax3.set_xlim(10, 400)
        self.ax3.set_xscale('log')
        self.ax3.set_title('Magnitude response', fontsize=9)
        self.ax3.set_xlabel('Frequency [Hz]', fontsize=8)
        self.ax3.set_ylabel('Magnitude [dB]', fontsize=8)
        self.ax3.set_ylim(-36, 0)
        # log-Achse erhält ihre Ticks erst nach Datenladung (siehe update_plot())
        self.ax3.set_yticks(np.arange(-36, 6, 6))
        self.ax3.grid(True, which='both', linestyle=':', alpha=0.5)
        self.ax3.tick_params(axis='both', labelsize=8)
        self.ax3.set_title('Magnitude response (no data)', fontsize=9, color='gray')
        
        # Zeichne Haupt-Canvas
        self.canvas.draw()
        
        # --- Polar Plot (bereits gut initialisiert, nur Text hinzufügen) ---
        self.polar_ax.set_title('Polar Pattern (no data)', fontsize=11, color='gray')
        self.polar_canvas.draw()
        
        # --- Balloon Plot (Text hinzufügen) ---
        self.balloon_ax.set_title('Balloon Plot (no data)', fontsize=11, color='gray')
        self.balloon_canvas.draw()
        
        # Cabinet Plot ist bereits gut initialisiert

    def update_balloon_freq_selector(self, freqs):
        """Aktualisiert die Frequenzauswahl für den Balloon Plot"""
        # Speichere die aktuelle Auswahl
        current_selection = self.balloon_freq_selector.currentText()
        
        # Blockiere Signale während der Aktualisierung
        self.balloon_freq_selector.blockSignals(True)
        
        # Leere die Auswahlbox
        self.balloon_freq_selector.clear()
        
        # Füge alle Frequenzen hinzu
        for freq in freqs:
            self.balloon_freq_selector.addItem(f"{freq:.1f} Hz")
        
        # Versuche, die vorherige Auswahl wiederherzustellen
        if current_selection:
            index = self.balloon_freq_selector.findText(current_selection)
            if index >= 0:
                self.balloon_freq_selector.setCurrentIndex(index)
        
        # Aktiviere Signale wieder
        self.balloon_freq_selector.blockSignals(False)

    def update_balloon_plot_with_selected_freq(self):
        """Aktualisiert den Balloon Plot mit der ausgewählten Frequenz"""
        if self.last_balloon_data is None:
            return
            
        selected_freq_text = self.balloon_freq_selector.currentText()
        if not selected_freq_text:
            return
            
        # Extrahiere die Frequenz aus dem Text (z.B. "50.0 Hz" -> 50.0)
        selected_freq = float(selected_freq_text.split()[0])
        
        # Aktualisiere den Plot mit der ausgewählten Frequenz
        self.update_balloon_plot(self.last_balloon_data, selected_freq)

    def update_dynamic_range(self):
        """Aktualisiert den Dynamikbereich für Balloon und Polar Plot"""
        # Extrahiere den dB-Wert aus dem Text (z.B. "-42 dB" -> 42)
        selected_text = self.dynamic_range_selector.currentText()
        if not selected_text:
            return
        
        # Extrahiere Zahl (z.B. "-42 dB" -> 42)
        self.dynamic_range_db = abs(int(selected_text.split()[0]))
        
        # Aktualisiere beide Plots mit neuem Dynamikbereich
        if self.last_balloon_data is not None:
            self.update_balloon_plot_with_selected_freq()
        
        if self.last_polar_data is not None:
            self.update_polar_plot(self.last_polar_data)

    def update_balloon_plot(self, data, selected_freq=None, show_freq_in_title=True):
        """
        Erstellt einen 3D-Plot der Balloon-Daten für die ausgewählte Frequenz
        
        Args:
            data: Dictionary mit den Balloon-Daten
            selected_freq: Optional, die ausgewählte Frequenz für den Plot
            show_freq_in_title: Optional, ob die Frequenz im Titel angezeigt werden soll
        """
        try:            
            # Speichere die Daten für spätere Aktualisierungen
            self.last_balloon_data = data
            
            # Balloon Plot zurücksetzen
            self.balloon_ax.clear()
            
            # Prüfe, ob balloon_data existiert
            if 'balloon_data' not in data:
                print("Keine balloon_data gefunden. Verfügbare Schlüssel:", data.keys())
                self.balloon_canvas.draw()
                return False
            
            # NEUE NUMPY-STRUKTUR
            balloon = data['balloon_data']
            
            # NEUE NUMPY-STRUKTUR (Standard - alte Struktur wird von data_module automatisch konvertiert)
            if 'meridians' not in balloon or not isinstance(balloon['meridians'], np.ndarray):
                print("❌ FEHLER: Balloon-Daten haben nicht die erwartete NumPy-Struktur!")
                print(f"   Verfügbare Keys: {balloon.keys() if isinstance(balloon, dict) else 'N/A'}")
                self.balloon_canvas.draw()
                return False
            
            # Extrahiere Daten aus NumPy-Struktur
            freqs = balloon['frequencies']
            meridians = balloon['meridians']
            horizontal_angles = balloon['horizontal_angles']
            
            # Stelle sicher, dass freqs ein NumPy-Array ist
            if not isinstance(freqs, np.ndarray):
                freqs = np.array(freqs)
            
            # Bestimme die zu plottende Frequenz
            if selected_freq is not None:
                freq_idx = np.abs(freqs - selected_freq).argmin()
            else:
                target_freq = 50.0
                freq_idx = np.abs(freqs - target_freq).argmin()
            
            freq = freqs[freq_idx]
            
            # Setze den Titel
            if show_freq_in_title:
                self.balloon_ax.set_title(f"Balloon Plot @ {freq:.1f} Hz")
            else:
                self.balloon_ax.set_title("Balloon Plot")
            
            # Prüfe, ob genügend Meridiane vorhanden sind
            if len(meridians) < 3:
                print(f"WARNUNG: Zu wenige Meridiane ({len(meridians)}). Mindestens 3 benötigt.")
                self.balloon_canvas.draw()
                return False
            
            # Erstelle Gitter für sphärische Koordinaten
            # Unser System: Y = Abstrahlachse (vorne/hinten), Z = oben/unten, X = links/rechts
            # theta = horizontal_angles: 0° = vorne (Y+), 180° = hinten (Y-)
            # phi = meridians: Rotation um Y-Achse
            
            # Für geschlossenen Surface-Plot: 360° = 0° hinzufügen
            meridians_closed = np.append(meridians, 360)
            
            # meshgrid: Erste Parameter-Werte variieren in Spalten (axis=1), zweite in Zeilen (axis=0)
            # Wir wollen: THETA (horizontal_angles) in Zeilen, PHI (meridians) in Spalten
            THETA, PHI = np.meshgrid(np.radians(horizontal_angles), np.radians(meridians_closed))
            
            # Hole Magnitude-Werte direkt aus NumPy-Array
            # Shape: [N_mer, N_horz, N_freq]
            R_base = balloon['magnitude'][:, :, freq_idx]  # [N_mer, N_horz]
            
            # Füge erste Zeile (Meridian 0°) am Ende hinzu (= 360°) für geschlossene Oberfläche
            R = np.vstack([R_base, R_base[0:1, :]])  # [N_mer+1, N_horz]
            
            # Ersetze NaN-Werte mit -60 dB (niedrige Hintergrundwerte)
            if np.sum(np.isnan(R)) > 0:
                R = np.nan_to_num(R, nan=-60.0)
            
            # Normalisiere auf 0 dB Maximum
            if np.max(R) != np.min(R):  # Nur normalisieren, wenn nicht alle Werte gleich sind
                R = R - np.max(R)
            
            # Begrenze den Wertebereich auf 0 bis -dynamic_range_db dB
            R = np.clip(R, -self.dynamic_range_db, 0)
            
            # Skalierungsfaktor für die Darstellung (0 dB = 1.0, -dynamic_range_db dB = 0.0)
            scale_factor = 1.0 / self.dynamic_range_db
            
            # Skaliere die dB-Werte für die Darstellung (0 bis 1)
            R_scaled = (R + self.dynamic_range_db) * scale_factor
            
            # Konvertiere zu kartesischen Koordinaten
            # Unser System: Y = Abstrahlachse (Hauptachse)
            # Sphärische → Kartesisch mit Y als Hauptachse:
            # X = R * sin(theta) * cos(phi)    (links/rechts)
            # Y = R * cos(theta)                (vorne/hinten, Abstrahlrichtung)
            # Z = R * sin(theta) * sin(phi)    (oben/unten)
            X = R_scaled * np.sin(THETA) * np.cos(PHI)
            Y = R_scaled * np.cos(THETA)
            Z = R_scaled * np.sin(THETA) * np.sin(PHI)
            
            # Verhindere Division durch Null: Füge kleine Variation hinzu wenn alles zu ähnlich ist
            eps = 1e-10
            if np.max(np.abs(X)) < eps:
                X = X + np.random.normal(0, eps, X.shape)
            if np.max(np.abs(Y)) < eps:
                Y = Y + np.random.normal(0, eps, Y.shape)
            if np.max(np.abs(Z)) < eps:
                Z = Z + np.random.normal(0, eps, Z.shape)
                        
            # Erstelle Surface Plot mit expliziter Farbgebung basierend auf dB-Werten
            norm = plt.Normalize(-self.dynamic_range_db, 0)  # Normalisiere auf den dB-Bereich
            colors = plt.cm.RdYlGn_r(norm(R))  # Erzeuge Farben direkt aus den dB-Werten
            
            # Erstelle Surface Plot mit Fehlerbehandlung
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                surf = self.balloon_ax.plot_surface(X, Y, Z,
                                          facecolors=colors,
                                          linewidth=0.1,
                                          antialiased=True,
                                          alpha=0.9)
            
            # Gleiche Achsenskalierung
            # Verwende nan-sichere Operationen, da importierte Daten Lücken enthalten können
            range_components = np.array([
                np.nanmax(X) - np.nanmin(X),
                np.nanmax(Y) - np.nanmin(Y),
                np.nanmax(Z) - np.nanmin(Z)
            ])
            max_range = np.nanmax(range_components) / 2.0
            # Verhindere Division durch Null: Setze Mindestbereich
            if not np.isfinite(max_range) or max_range < eps:
                max_range = 1.0  # Standardbereich
            mid_x = (np.nanmax(X) + np.nanmin(X)) * 0.5
            mid_y = (np.nanmax(Y) + np.nanmin(Y)) * 0.5
            mid_z = (np.nanmax(Z) + np.nanmin(Z)) * 0.5
            if not np.isfinite(mid_x):
                mid_x = 0.0
            if not np.isfinite(mid_y):
                mid_y = 0.0
            if not np.isfinite(mid_z):
                mid_z = 0.0
            self.balloon_ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.balloon_ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.balloon_ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Entferne Achsenbeschriftungen für cleaneren Look
            self.balloon_ax.set_xticklabels([])
            self.balloon_ax.set_yticklabels([])
            self.balloon_ax.set_zticklabels([])
            
            # Setze Blickwinkel
            self.balloon_ax.view_init(elev=20, azim=45)
            
            # Setze gleiche Skalierung für alle Achsen
            self.balloon_ax.set_box_aspect([1, 1, 1])
            
            # Entferne das Raster
            self.balloon_ax.grid(False)
            
            # Zeichne Achsen durch den Lautsprecher mit Strichpunkt-Linien
            # Unser Koordinatensystem: Y = Abstrahlachse, X = links/rechts, Z = oben/unten
            
            # Y-Achse: Abstrahlrichtung (vorne = 0°)
            front_line = np.array([[0, 0, 0], [0, max_range*1.5, 0]])
            self.balloon_ax.plot(front_line[:, 0], front_line[:, 1], front_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(0, max_range*1.6, 0, "0° (vorne)", fontsize=10)
            
            # Y-Achse: Rückseite (hinten = 180°)
            back_line = np.array([[0, 0, 0], [0, -max_range*1.5, 0]])
            self.balloon_ax.plot(back_line[:, 0], back_line[:, 1], back_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(0, -max_range*1.6, 0, "180° (hinten)", fontsize=10)
            
            # X-Achse: links/rechts
            right_line = np.array([[0, 0, 0], [max_range*1.5, 0, 0]])
            self.balloon_ax.plot(right_line[:, 0], right_line[:, 1], right_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(max_range*1.6, 0, 0, "rechts", fontsize=10)
            
            left_line = np.array([[0, 0, 0], [-max_range*1.5, 0, 0]])
            self.balloon_ax.plot(left_line[:, 0], left_line[:, 1], left_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(-max_range*1.6, 0, 0, "links", fontsize=10)
            
            # Z-Achse: oben/unten
            up_line = np.array([[0, 0, 0], [0, 0, max_range*1.5]])
            self.balloon_ax.plot(up_line[:, 0], up_line[:, 1], up_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(0, 0, max_range*1.6, "oben", fontsize=10)
            
            down_line = np.array([[0, 0, 0], [0, 0, -max_range*1.5]])
            self.balloon_ax.plot(down_line[:, 0], down_line[:, 1], down_line[:, 2], 'k-.', linewidth=1.0)
            self.balloon_ax.text(0, 0, -max_range*1.6, "unten", fontsize=10)
            
            # Canvas aktualisieren, aber nur wenn der Canvas bereits eine Größe hat
            canvas_width, canvas_height = self.balloon_canvas.get_width_height()
            if canvas_width > 0 and canvas_height > 0:
                self.balloon_canvas.draw()
            
            return True
            
        except Exception as e:
            print(f"FEHLER beim Erstellen des Balloon-Plots: {e}")
            import traceback
            traceback.print_exc()
            self.balloon_canvas.draw()
            return False

    def update_polar_plot(self, data):
        """Aktualisiert den Polar Plot mit der horizontalen Ebene aus balloon_data"""
        try:
            # Speichere Daten für spätere Aktualisierungen
            self.last_polar_data = data
            self.polar_ax.set_title('Polar Pattern')
            
            # Verwende die horizontale Ebene aus balloon_data (Meridian 0°)
            if 'balloon_data' not in data or not data['balloon_data']:
                print("⚠ Keine balloon_data für Polar Plot vorhanden")
                self.polar_ax.clear()
                self.polar_ax.set_title('Polar Pattern (no data)')
                self.polar_canvas.draw()
                return
            
            # NEUE NUMPY-STRUKTUR
            balloon = data['balloon_data']
            
            # Prüfe ob neue Struktur (NumPy) oder alte Struktur (Dict)
            if 'meridians' in balloon and isinstance(balloon['meridians'], np.ndarray):
                # NEUE NUMPY-STRUKTUR
                freqs = balloon['frequencies']
                meridians = balloon['meridians']
                horizontal_angles = balloon['horizontal_angles']
                
                # Für 2D-Polarplot: Verwende MERIDIAN-Schnitt (vertikaler Schnitt)
                # Meridian 0° = vorne, mit allen horizontal_angles (0°-180°)
                # Für 360° Kreis: Kombiniere Meridian 0° mit Meridian 180°
                
                # Finde Meridian 0° (vorne)
                if 0 in meridians:
                    mer_0_idx = np.where(meridians == 0)[0][0]
                    mag_meridian_0 = balloon['magnitude'][mer_0_idx, :, :]  # [N_horz, N_freq]
                else:
                    mag_meridian_0 = None
                
                # Finde Meridian 180° (hinten)
                if 180 in meridians:
                    mer_180_idx = np.where(meridians == 180)[0][0]
                    mag_meridian_180 = balloon['magnitude'][mer_180_idx, :, :]  # [N_horz, N_freq]
                else:
                    mag_meridian_180 = None
                
                # Erstelle 360° Polarplot aus Meridian 0° und 180°
                # Mapping: 
                #   0° Polarplot = horizontal_angle 90° bei Meridian 0° (vorne)
                #  90° Polarplot = horizontal_angle 0° (oben, zwischen vorne und hinten)
                # 180° Polarplot = horizontal_angle 90° bei Meridian 180° (hinten)
                # 270° Polarplot = horizontal_angle 180° (unten, zwischen hinten und vorne)
                
                if mag_meridian_0 is not None and mag_meridian_180 is not None:
                    # Erstelle 360° Kreis aus BEIDEN Meridianen
                    # 
                    # KOORDINATENSYSTEM ERKLÄRUNG:
                    # ============================
                    # Meridian = Vertikaler Halbkreis (0° oben, 90° horizontal, 180° unten)
                    # Meridian 0° = Schnitt durch vorne (Y+)
                    # Meridian 180° = Schnitt durch hinten (Y-)
                    # 
                    # FÜR POLARPLOT (Draufsicht, 360° Kreis):
                    # ========================================
                    # 0° Polar   = Meridian 0°, horizontal_angle 90° (vorne horizontal)
                    # 90° Polar  = Meridian 0°, horizontal_angle 0° (oben) = Meridian 180°, horizontal_angle 0°
                    # 180° Polar = Meridian 180°, horizontal_angle 90° (hinten horizontal)
                    # 270° Polar = Meridian 0°, horizontal_angle 180° (unten) = Meridian 180°, horizontal_angle 180°
                    
                    N_horz = len(horizontal_angles)
                    N_freq = len(freqs)
                    
                    # Erstelle Arrays für 360° Polarplot
                    # Kombiniere beide Meridiane:
                    # - Meridian 0°: horizontal_angles 90° → 0° (vorne → oben)
                    # - Meridian 180°: horizontal_angles 90° → 180° (hinten → unten)
                    
                    # KORRIGIERTE ZUORDNUNG: 0° Polarplot = vorne (Abstrahlrichtung)
                    # ====================================================================
                    # Vertikaler Kreis um die Lautsprecher (von vorne gesehen):
                    # Polarplot 0° (oben) = vorne horizontal = Meridian 0°, horizontal_angle 90°
                    # Polarplot 90° (rechts) = unten = horizontal_angle 180° (bei beiden Meridianen)
                    # Polarplot 180° (unten) = hinten horizontal = Meridian 180°, horizontal_angle 90°
                    # Polarplot 270° (links) = oben = horizontal_angle 0° (bei beiden Meridianen)
                    
                    idx_90 = 90
                    idx_0 = 0
                    idx_180 = 180
                    
                    # Teil 1: Meridian 0° von hor_angle 90° bis 180° (vorne → unten)
                    # Polarplot 0° bis 90°
                    mag_part1 = mag_meridian_0[idx_90:idx_180+1, :]  # Von 90 bis 180 = 91 Punkte
                    angles_part1 = np.arange(0, 91)  # Polar 0° bis 90° = 91 Punkte
                    
                    # Teil 2: Meridian 180° von hor_angle 180° bis 90° RÜCKWÄRTS (unten → hinten)
                    # Polarplot 90° bis 180°
                    # Überspringe ersten Punkt (180°), da er mit Teil 1 überlappt
                    # UND letzten Punkt (90°), da er als part2b separat kommt
                    mag_part2 = mag_meridian_180[idx_180-1:idx_90:-1, :]  # Von 179 runter bis 91 = 89 Punkte
                    angles_part2 = np.arange(91, 180)  # Polar 91° bis 179° = 89 Punkte
                    
                    # Füge Meridian 180° bei horizontal_angle 90° hinzu (genau hinten)
                    mag_part2b = mag_meridian_180[idx_90:idx_90+1, :]  # horizontal_angle 90° = 1 Punkt
                    angles_part2b = np.array([180])  # Polar 180° = 1 Punkt
                    
                    # Teil 3: Meridian 180° von hor_angle 89° bis 0° RÜCKWÄRTS (hinten → oben)
                    # Polarplot 181° bis 270°
                    # Überspringe ersten Punkt (90°), da bereits in part2b
                    mag_part3 = mag_meridian_180[idx_90-1::-1, :]  # Von 89 runter bis 0 = 90 Punkte
                    angles_part3 = np.arange(181, 271)  # Polar 181° bis 270° = 90 Punkte
                    
                    # Teil 4: Meridian 0° von hor_angle 1° bis 89° VORWÄRTS (oben → vorne)
                    # Polarplot 271° bis 359°
                    # Überspringe ersten Punkt (0°), da er mit Teil 3 überlappt
                    # UND letzten Punkt (90°), da er bereits in part1 ist
                    mag_part4 = mag_meridian_0[idx_0+1:idx_90, :]  # Von 1 bis 89 = 89 Punkte
                    angles_part4 = np.arange(271, 360)  # Polar 271° bis 359° = 89 Punkte
                    
                    # Schließe den Kreis: Füge Startpunkt wieder hinzu (360° = 0°)
                    mag_close = mag_part1[0:1, :]  # Erster Punkt von Teil 1 = 1 Punkt
                    angle_close = np.array([360])  # = 1 Punkt
                    
                    # Kombiniere alle Teile: 91 + 89 + 1 + 90 + 89 + 1 = 361 Punkte
                    magnitude_line = np.vstack([mag_part1, mag_part2, mag_part2b, mag_part3, mag_part4, mag_close])
                    polar_angles_deg = np.concatenate([angles_part1, angles_part2, angles_part2b, angles_part3, angles_part4, angle_close])
                    
                    # KORREKTUR: Verschiebe um +90°, damit Abstrahlrichtung bei 0° liegt (statt bei 270°)
                    polar_angles_deg = (polar_angles_deg + 90) % 360
                    
                    # ⚠️ KRITISCH: Nach Rotation müssen Winkel NEU SORTIERT werden!
                    # Sonst springt Matplotlib zwischen unsortierten Winkeln hin und her
                    sort_indices = np.argsort(polar_angles_deg)
                    polar_angles_deg = polar_angles_deg[sort_indices]
                    magnitude_line = magnitude_line[sort_indices, :]
                    
                    polar_angles_rad = np.deg2rad(polar_angles_deg)
                    
                elif mag_meridian_0 is not None:
                    print("  ⚠ Nur Meridian 0° vorhanden - verwende symmetrische Näherung")
                    # Verwende nur Meridian 0° und spiegle ihn
                    N_horz = len(horizontal_angles)
                    N_freq = len(freqs)
                    
                    # Spiegle Meridian 0° für vollständigen Kreis
                    mag_forward = mag_meridian_0  # 0° bis 180°
                    mag_backward = mag_meridian_0[::-1, :]  # 180° bis 0° (gespiegelt)
                    
                    magnitude_line = np.vstack([mag_forward, mag_backward[1:, :]])
                    polar_angles_deg = np.concatenate([
                        horizontal_angles - 90,  # -90° bis 90°
                        180 - horizontal_angles[1:]  # 180° bis 90° (rückwärts)
                    ])
                    polar_angles_rad = np.deg2rad(polar_angles_deg)
                    
                else:
                    print("  ❌ FEHLER: Meridian 0° nicht verfügbar für Polarplot!")
                    self.polar_canvas.draw()
                    return
                
            else:
                # ALTE DICT-STRUKTUR
                if "0" in data['balloon_data']:
                    plot_data_dict = data['balloon_data']["0"]
                else:
                    first_meridian = next(iter(data['balloon_data']))
                    plot_data_dict = data['balloon_data'][first_meridian]
                    print(f"⚠ Meridian 0° nicht gefunden, verwende Meridian {first_meridian}°")
                
                # Hole und prüfe Winkel
                angles = sorted([float(angle) for angle in plot_data_dict.keys()])
                if not angles:
                    print("⚠ Keine Winkel in balloon_data gefunden")
                    return
                
                first_angle = angles[0]
                freqs = plot_data_dict[str(int(first_angle))]['freq']
                
                # Konvertiere für Kompatibilität
                plot_data = {
                    'angles': angles,
                    'freq': freqs,
                    'magnitude': np.array([[plot_data_dict[str(int(angle))]['magnitude'][f] for f in range(len(freqs))] for angle in angles]),
                    'phase': np.array([[plot_data_dict[str(int(angle))]['phase'][f] for f in range(len(freqs))] for angle in angles])
                }
            
            # Verwende neue Struktur (NumPy) oder konvertierte Dict-Struktur
            # Bei NumPy-Zweig: Winkel und Daten bereits vorbereitet (polar_angles_rad, magnitude_line)
            if 'meridians' in balloon and isinstance(balloon['meridians'], np.ndarray):
                angles = polar_angles_rad
                freqs = freqs
                # magnitude_line ist bereits in Zeile 504 definiert [N_horz+1, N_freq]
            else:
                angles = plot_data['angles']
                freqs = plot_data['freq']
                magnitude_line = plot_data['magnitude']  # [N_horz, N_freq]
            
            self.polar_ax.clear()
            
            # Plot-Frequenzen definieren
            plot_freqs = [25, 31.5, 40, 50, 63, 80]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_freqs)))
            
            all_values = []
            
            # Sicherstellen, dass freqs ein NumPy-Array ist
            freqs = np.array(freqs)
            
            # Sammle erst alle Werte für Normalisierung
            temp_data = []
            for freq in plot_freqs:
                freq_idx = np.abs(freqs - freq).argmin()
                
                # magnitude_line ist für beide Zweige definiert
                values = magnitude_line[:, freq_idx]
                
                temp_data.append((freq, freq_idx, values))
                all_values.extend(values)
            
            # Normalisiere auf 0 dB Maximum
            if all_values:
                max_val = max(all_values)
                normalization_offset = -max_val  # Verschiebung um Maximum auf 0 dB zu bringen
            else:
                normalization_offset = 0
            
            # Plot für jede Frequenz mit normalisiertem Wert
            for (freq, freq_idx, values), color in zip(temp_data, colors):
                actual_freq = freqs[freq_idx]
                
                # Normalisiere Werte
                values_normalized = values + normalization_offset
                
                if 'meridians' in balloon and isinstance(balloon['meridians'], np.ndarray):
                    # NEUE NUMPY-STRUKTUR: Bereits geschlossen
                    angles_plot = angles
                    values_plot = values_normalized
                else:
                    # ALTE DICT-STRUKTUR: Schließe den Kreis
                    angles_rad = np.deg2rad(plot_data['angles'])
                    angles_plot = np.append(angles_rad, angles_rad[0])
                    values_plot = np.append(values_normalized, values_normalized[0])
                
                self.polar_ax.plot(angles_plot, values_plot, '-', 
                                 color=color, 
                                 label=f'{actual_freq:.1f} Hz', 
                                 linewidth=2)

            # Polarplot-Ausrichtung: 0° = oben = Abstrahlrichtung vorne (Y+)
            # Verwendet Meridian-Schnitt: horizontal_angle 90° (vorne) → Polarplot 0°
            self.polar_ax.set_theta_zero_location("N")  # 0° oben
            self.polar_ax.set_theta_direction(-1)       # Uhrzeigersinn
            
            # Winkel-Ticks
            angles_deg = np.arange(0, 360, 45)
            self.polar_ax.set_xticks(np.deg2rad(angles_deg))
            self.polar_ax.set_xticklabels([f'{int(angle)}°' for angle in angles_deg])

            # Dynamische Achse: -dynamic_range_db bis 0 dB mit 6 dB Schritten
            self.polar_ax.set_ylim(-self.dynamic_range_db, 0)
            
            # 6 dB Schritte von -dynamic_range_db bis 0
            rticks = np.arange(-self.dynamic_range_db, 1, 6)
            self.polar_ax.set_rticks(rticks)
            self.polar_ax.set_yticklabels([f'{int(val)}dB' for val in rticks])

            self.polar_ax.grid(True, linestyle='-', alpha=0.2)
            self.polar_ax.legend(bbox_to_anchor=(1.2, 1.0))

            self.polar_canvas.draw()

        except Exception as e:
            print(f"\nFEHLER beim Update des Polar Plots: {e}")
            import traceback
            traceback.print_exc()
        
    def update_cabinet_plot(self, speakers):
        """Aktualisiert den Cabinet Plot in 3D mit optional gestapelten Lautsprechern."""
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            self.cabinet_ax.clear()
            
            if not hasattr(self.cabinet_ax, 'get_zlim'):
                self.cabinet_figure.delaxes(self.cabinet_ax)
                self.cabinet_ax = self.cabinet_figure.add_subplot(111, projection='3d')
            
            parsed_speakers = []
            
            for speaker in speakers:
                try:
                    width = float(speaker['width'])
                    depth = float(speaker['depth'])
                    front_height = float(speaker['front_height'])
                    back_height = float(speaker['back_height'])
                    is_cardio = bool(speaker.get('cardio', False))
                    is_flown = bool(speaker.get('is_flown', False))
                    stack_layout = speaker.get('stack_layout', 'beside') or 'beside'
                except (ValueError, TypeError, KeyError):
                    continue
            
                if width <= 0:
                    width = 0.5
                if depth <= 0:
                    depth = 0.5
                if front_height <= 0:
                    front_height = max(width, depth) / 2.0
                if back_height <= 0:
                    back_height = front_height
                
                effective_height = max(front_height, back_height)

                parsed_speakers.append({
                    'width': width,
                    'depth': depth,
                    'front_height': front_height,
                    'back_height': back_height,
                    'effective_height': effective_height,
                    'is_cardio': is_cardio,
                    'is_flown': is_flown,
                    'stack_layout': stack_layout.lower()
                })

            if not parsed_speakers:
                self.cabinet_canvas.draw()
                return

            columns = []
            current_stack_column = None

            for speaker in parsed_speakers:
                is_flown = speaker['is_flown']
                stack_layout = speaker['stack_layout']

                if is_flown or stack_layout != 'on top':
                    column = {
                        'speakers': [speaker],
                        'width': speaker['width'],
                        'stacked': False
                    }
                    columns.append(column)
                    current_stack_column = None if is_flown else len(columns) - 1
                else:
                    if current_stack_column is None:
                        column = {
                            'speakers': [speaker],
                            'width': speaker['width'],
                            'stacked': True
                        }
                        columns.append(column)
                        current_stack_column = len(columns) - 1
                    else:
                        column = columns[current_stack_column]
                        column['speakers'].append(speaker)
                        column['width'] = max(column['width'], speaker['width'])
                        column['stacked'] = True

            if not columns:
                self.cabinet_canvas.draw()
                return

            total_width = sum(col['width'] for col in columns)
            max_depth = max(sp['depth'] for sp in parsed_speakers)
            max_height = 0.0

            x_cursor = -total_width / 2.0 if total_width > 0 else 0.0

            positioned = []

            for column in columns:
                column_width = column['width']
                z_cursor = 0.0

                for speaker in column['speakers']:
                    width = speaker['width']
                    depth = speaker['depth']
                    front_height = speaker['front_height']
                    back_height = speaker['back_height']
                    effective_height = speaker['effective_height']
                    is_cardio = speaker['is_cardio']

                    x_pos = x_cursor + (column_width - width) / 2.0

                    top_height = z_cursor + max(front_height, back_height)
                    max_height = max(max_height, top_height)

                    positioned.append({
                        'x_pos': x_pos,
                        'width': width,
                        'depth': depth,
                        'front_height': front_height,
                        'back_height': back_height,
                        'is_cardio': is_cardio,
                        'z_offset': z_cursor
                    })

                    if column.get('stacked'):
                        z_cursor += effective_height

                x_cursor += column_width

            if not positioned:
                self.cabinet_canvas.draw()
                return

            min_x = min(p['x_pos'] for p in positioned)
            max_x = max(p['x_pos'] + p['width'] for p in positioned)
            total_width_span = max_x - min_x if max_x > min_x else max(p['width'] for p in positioned)

            for speaker in positioned:
                width = speaker['width']
                depth = speaker['depth']
                front_height = speaker['front_height']
                back_height = speaker['back_height']
                is_cardio = speaker['is_cardio']
                z_offset = speaker['z_offset']

                base_front = z_offset
                base_back = z_offset
                top_front = z_offset + front_height
                top_back = z_offset + back_height

                vertices = np.array([
                    [speaker['x_pos'], 0, base_front],
                    [speaker['x_pos'] + width, 0, base_front],
                    [speaker['x_pos'] + width, -depth, base_back],
                    [speaker['x_pos'], -depth, base_back],
                    [speaker['x_pos'], 0, top_front],
                    [speaker['x_pos'] + width, 0, top_front],
                    [speaker['x_pos'] + width, -depth, top_back],
                    [speaker['x_pos'], -depth, top_back]
                    ])
                
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                
                standard_color = '#E0E0E0'
                dark_color = '#505050'
                face_colors = [standard_color] * 6
                face_colors[3 if is_cardio else 2] = dark_color

                for idx, face in enumerate(faces):
                    poly3d = Poly3DCollection([face], alpha=0.5, linewidths=1, edgecolor='black')
                    poly3d.set_facecolor(face_colors[idx])
                    self.cabinet_ax.add_collection3d(poly3d)
                
            if max_depth <= 0:
                max_depth = 0.5
            if max_height <= 0:
                max_height = max(sp['effective_height'] for sp in parsed_speakers)

            span_x = total_width_span if total_width_span > 0 else max(sp['width'] for sp in parsed_speakers)
            span_y = max_depth
            span_z = max_height
            base_extent = max(span_x, span_y, span_z)
            margin = base_extent * 0.2 if base_extent > 0 else 0.2
            axis_extent = base_extent + 2 * margin
            half_extent = axis_extent / 2.0

            mid_x = (min_x + max_x) / 2.0 if span_x > 0 else 0.0
            mid_y = -max_depth / 2.0
            mid_z = max_height / 2.0

            x_min = mid_x - half_extent
            x_max = mid_x + half_extent
            y_min = mid_y - half_extent
            y_max = mid_y + half_extent

            z_min = mid_z - half_extent
            if z_min < 0:
                z_shift = -z_min
                z_min = 0
                mid_z += z_shift
            z_max = z_min + axis_extent

            self.cabinet_ax.set_xlim(x_min, x_max)
            self.cabinet_ax.set_ylim(y_min, y_max)
            self.cabinet_ax.set_zlim(z_min, z_max)

            self.cabinet_ax.set_xlabel('Width (m)')
            self.cabinet_ax.set_ylabel('Depth (m)')
            self.cabinet_ax.set_zlabel('Height (m)')
            
            self.cabinet_ax.set_box_aspect([1, 1, 1])
            self.cabinet_ax.view_init(elev=30, azim=45)
            self.cabinet_ax.grid(False)
            
            if self.cabinet_ax.get_legend():
                self.cabinet_ax.get_legend().remove()
            
            self.cabinet_canvas.draw()
                
        except Exception as e:
            print(f"Fehler beim Update des Cabinet Plots: {e}")
            import traceback
            traceback.print_exc()

    def get_canvas(self):
        """Gibt das Canvas-Widget zurück"""
        return self.canvas
        
    def get_toolbar(self):
        """Gibt die Toolbar zurück"""
        return self.toolbar

    def get_polar_canvas(self):
        """Gibt das Polar-Canvas-Widget zurück"""
        return self.polar_canvas
        
    def get_polar_toolbar(self):
        """Gibt die Polar-Toolbar zurück"""
        return self.polar_toolbar
        
    def get_balloon_canvas(self):
        """Gibt das Balloon-Canvas-Widget zurück"""
        return self.balloon_canvas
        
    def get_balloon_toolbar(self):
        """Gibt die Balloon-Toolbar zurück"""
        return self.balloon_toolbar
        
    def get_balloon_freq_selector(self):
        """Gibt das Frequenzauswahl-Widget für den Balloon Plot zurück"""
        return self.balloon_freq_selector_widget

    def update_plot(self, data, filename):
        """Aktualisiert die Plots für die ausgewählte Datei"""
        self.data = data
        
        try:
            # Prüfe, ob normalisierte Daten existieren
            if filename in data['calculated_data']:
                plot_data = data['calculated_data'][filename]
            elif filename in data['raw_measurements']:
                plot_data = data['raw_measurements'][filename]
            else:
                return
            
            # Hole die Daten
            freq = np.array(plot_data['freq'])
            magnitude = np.array(plot_data['magnitude'])
            phase = np.array(plot_data['phase'])
            
            # Nach Frequenzfilterung - versuche verschiedene Quellen für freq_range
            freq_range = self.data['metadata'].get('freq_range')
            
            # Fallback: freq_range aus freq_range_min/max erstellen falls nicht vorhanden
            if not freq_range:
                freq_min = self.data['metadata'].get('freq_range_min')
                freq_max = self.data['metadata'].get('freq_range_max')
                if freq_min is not None and freq_max is not None:
                    freq_range = {'min': freq_min, 'max': freq_max}
                else:
                    # Letzter Fallback: Standard-Werte aus UI verwenden
                    freq_range = {'min': 10.0, 'max': 300.0}
            
            # WICHTIG: Frequenzen NICHT abschneiden! Filter dämpfen nur, schneiden nicht ab.
            # Die freq_range wird nur für die Plot-Achsen verwendet, nicht zum Daten-Abschneiden
            
            # Lösche alte Plots und setze Achsen mit gültigen Log-Grenzen neu
            self.ax1.clear()
            # Verhindere Log-Warnungen, indem während des Clearings linearer Modus aktiv ist
            try:
                self.ax2.set_xscale('linear')
            except Exception:
                pass
            self.ax2.clear()
            self.ax2.set_xlim(10, 400)
            self.ax2.set_xscale('log')

            try:
                self.ax3.set_xscale('linear')
            except Exception:
                pass
            self.ax3.clear()
            self.ax3.set_xlim(10, 400)
            self.ax3.set_xscale('log')
            
            # Plot 1: IR/AT (wie in PlotImpulse.py - oberer Plot)
            ir_window_enabled = self.data['metadata'].get('ir_window_enabled', False)
            filter_enabled = self.data['metadata'].get('filter_enabled', False)
            time_offset = self.data['metadata'].get('time_offset', 0.0)
            spl_normalized = self.data['metadata'].get('spl_normalized', False)
            
            # Sample-Rate (Fallback bei TXT-Import wo fs=None)
            fs = self.data['metadata'].get('fs', 48000)
            if fs is None:
                fs = 48000  # Fallback für TXT-Import
            
            # Zeige Original-Impulsantwort IMMER (schwarze Linie = Referenz)
            if 'original_impulse_response' in plot_data and plot_data['original_impulse_response'] is not None:
                original_impulse = plot_data['original_impulse_response']
                # Prüfe ob Array nicht leer ist (für TXT-Dateien)
                if len(original_impulse) > 0:
                    original_max = np.max(np.abs(original_impulse))
                    if original_max > 0:
                        original_normalized = original_impulse / original_max
                    else:
                        original_normalized = original_impulse
                    original_smoothed = self.smooth_impulse_response(original_normalized, smoothing_factor=15)
                    time = np.arange(len(original_impulse)) / fs * 1000
                    # Original IMMER in schwarz
                    self.ax1.plot(time, original_smoothed, '-', alpha=0.8, linewidth=1.0, color='black')
                    self.ax1.set_ylabel('Impulse response [%]', fontsize=8)
            
            # Zeige manipulierte IR nur wenn sie sich unterscheidet
            if ir_window_enabled or filter_enabled or time_offset != 0.0:
                if ir_window_enabled and 'windowed_impulse_response' in plot_data and plot_data['windowed_impulse_response'] is not None:
                    # Gefensterte IR
                    impulse_response = plot_data['windowed_impulse_response']
                elif filter_enabled and 'processed_impulse_response' in plot_data and plot_data['processed_impulse_response'] is not None:
                    # Gefilterte IR
                    impulse_response = plot_data['processed_impulse_response']
                else:
                    # Standard IR
                    impulse_response = plot_data.get('impulse_response', None)
                
                if impulse_response is not None:
                    # Normalisiere relativ zum Original
                    if 'original_impulse_response' in plot_data and plot_data['original_impulse_response'] is not None:
                        original_max = np.max(np.abs(plot_data['original_impulse_response']))
                        if original_max > 0:
                            impulse_normalized = impulse_response / original_max
                        else:
                            impulse_normalized = impulse_response
                    else:
                        impulse_max = np.max(np.abs(impulse_response))
                        impulse_normalized = impulse_response / impulse_max if impulse_max > 0 else impulse_response
                    
                    impulse_smoothed = self.smooth_impulse_response(impulse_normalized, smoothing_factor=15)
                    time = np.arange(len(impulse_response)) / fs * 1000
                    self.ax1.plot(time, impulse_smoothed, linewidth=1.5)
            
            # Zeige Fensterfunktion nur falls aktiviert
            if ir_window_enabled and 'window_function' in plot_data and plot_data['window_function'] is not None:
                window_function = plot_data['window_function']
                # Prüfe ob Array nicht leer ist
                if len(window_function) > 0:
                    window_max = np.max(np.abs(window_function))
                    if window_max > 0:
                        window_normalized = window_function / window_max
                    else:
                        window_normalized = window_function
                    time = np.arange(len(window_function)) / fs * 1000
                    self.ax1.plot(time, window_normalized, '--', alpha=0.6, linewidth=1.0, color='gray')
            
            if 'impulse_response' in plot_data or 'processed_impulse_response' in plot_data or 'original_impulse_response' in plot_data:
                self.ax1.grid(True, which='both', linestyle=':', alpha=0.5)
                self.ax1.set_xlabel('Time [ms]', fontsize=8)
                self.ax1.tick_params(axis='both', labelsize=8)
                # Begrenze Zeitachse auf 0-500 ms
                self.ax1.set_xlim(0, 500)

            
            # Plot 2: Phase (wie in PlotImpulse.py - mittlerer Plot)
            if len(freq) > 0 and len(phase) > 0:
                # Phase liegt unwrapped in Grad vor - wrap auf [-180°, +180°] für korrekte Darstellung
                phase_deg = ((phase + 180) % 360) - 180
                
                filter_enabled = self.data['metadata'].get('filter_enabled', False)
                time_offset = self.data['metadata'].get('time_offset', 0.0)
                ir_window_enabled = self.data['metadata'].get('ir_window_enabled', False)
                spl_normalized = self.data['metadata'].get('spl_normalized', False)
                
                # Zeige Original-Phase IMMER (schwarze Linie = Referenz mit 400 Hz Filter)
                if 'original_phase' in plot_data and plot_data['original_phase'] is not None:
                    orig_freq = np.array(plot_data['original_freq'])
                    orig_phase = np.array(plot_data['original_phase'])
                    # Original-Phase liegt unwrapped in Grad vor - wrap auf [-180°, +180°]
                    orig_phase_deg = ((orig_phase + 180) % 360) - 180
                    # Filtere nur positive Frequenzen für log-Achse
                    mask = orig_freq > 0
                    if np.any(mask):
                        self.ax2.semilogx(orig_freq[mask], orig_phase_deg[mask], '-', color='black', alpha=0.8, linewidth=1.0)
                
                # Zeige manipulierte Phase wenn IRGENDEINE Manipulation aktiv ist
                if filter_enabled or time_offset != 0.0 or ir_window_enabled or spl_normalized:
                    # Manipulierte Phase als Hauptkurve mit Default-Farbe
                    # Filtere nur positive Frequenzen für log-Achse
                    mask = freq > 0
                    if np.any(mask):
                        self.ax2.semilogx(freq[mask], phase_deg[mask], '-', linewidth=1.5)
                else:
                    # Keine Manipulation: Zeige nur Referenz
                    pass
                
                # X-Achse (für logarithmische Achse muss xmin > 0 sein!)
                freq_ticks = [20, 40, 60, 80, 100, 200, 400]
                # Begrenze auf 400 Hz (Lowpass-Grenzfrequenz)
                freq_max = 400.0
                visible_ticks = [t for t in freq_ticks if t <= freq_max]
                
                # Setze x-Limits sicher (für log-Achse)
                x_min = 10.0  # Start bei 10 Hz (log-Achse kann nicht bei 0 starten!)
                x_max = 400.0  # Ende bei 400 Hz (Lowpass-Grenzfrequenz)
                self.ax2.set_xlim(x_min, x_max)
                self.ax2.set_xticks(visible_ticks)
                self.ax2.set_xticklabels([str(x) for x in visible_ticks])
                self.ax2.set_ylim(-180, 180)
                self.ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                self.ax2.grid(True, which='both', linestyle=':', alpha=0.5)
                self.ax2.set_ylabel('Phase [deg]', fontsize=8)
                self.ax2.set_xlabel('Frequency [Hz]', fontsize=8)
                self.ax2.tick_params(axis='both', labelsize=8, labelbottom=True)

            
            # Plot 3: Magnitude (wie in PlotImpulse.py - unterer Plot)
            if len(freq) > 0 and len(magnitude) > 0:
                filter_enabled = self.data['metadata'].get('filter_enabled', False)
                time_offset = self.data['metadata'].get('time_offset', 0.0)
                ir_window_enabled = self.data['metadata'].get('ir_window_enabled', False)
                spl_normalized = self.data['metadata'].get('spl_normalized', False)
                
                # Zeige Original-Magnitude IMMER (schwarze Linie = Referenz mit 400 Hz Filter)
                if 'original_magnitude' in plot_data and plot_data['original_magnitude'] is not None:
                    orig_freq = np.array(plot_data['original_freq'])
                    orig_magnitude = np.array(plot_data['original_magnitude'])
                    # Filtere nur positive Frequenzen für log-Achse
                    mask = orig_freq > 0
                    if np.any(mask):
                        self.ax3.semilogx(orig_freq[mask], orig_magnitude[mask], '-', color='black', alpha=0.8, linewidth=1.0)
                
                # Zeige manipulierte Magnitude wenn IRGENDEINE Manipulation aktiv ist
                if filter_enabled or time_offset != 0.0 or ir_window_enabled or spl_normalized:
                    # Filtere nur positive Frequenzen für log-Achse
                    mask = freq > 0
                    if np.any(mask):
                        # Manipulierte Daten als Hauptkurve mit Stützstellen-Markern
                        line = self.ax3.semilogx(freq[mask], magnitude[mask], '-', linewidth=1.5)
                        # Stützstellen sichtbar machen (in gleicher Farbe wie Linie)
                        line_color = line[0].get_color() if isinstance(line, list) and len(line) > 0 else None
                        self.ax3.semilogx(
                            freq[mask],
                            magnitude[mask],
                            'o',
                            ms=2,
                            alpha=0.5,
                            markerfacecolor=line_color if line_color else None,
                            markeredgecolor=line_color if line_color else None,
                        )
                else:
                    # Keine Manipulation: Zeige nur Referenz
                    pass
                
                # Plot-Begrenzungen
                # Y-Achsen Grenzen
                if freq_range:
                    optimal_limits = self.calculate_optimal_plot_limits(freq, magnitude, freq_range)
                    self.ax3.set_ylim(optimal_limits['y_min'], optimal_limits['y_max'])
                else:
                    # Dynamische y-Achsen Grenzen mit etwas Padding
                    mask_freq = freq > 20  # Nur Werte über 20 Hz berücksichtigen
                    if np.any(mask_freq):
                        mag_min = np.min(magnitude[mask_freq])
                        mag_max = np.max(magnitude[mask_freq])
                        padding = (mag_max - mag_min) * 0.1  # 10% Padding
                        self.ax3.set_ylim(mag_min - padding, mag_max + padding)
                
                # X-Achse (für logarithmische Achse muss xmin > 0 sein!)
                freq_ticks = [20, 40, 60, 80, 100, 200, 400]
                # Begrenze auf 400 Hz (Lowpass-Grenzfrequenz)
                freq_max = 400.0
                visible_ticks = [t for t in freq_ticks if t <= freq_max]
                
                # Setze x-Limits sicher (für log-Achse)
                x_min = 10.0  # Start bei 10 Hz (log-Achse kann nicht bei 0 starten!)
                x_max = 400.0  # Ende bei 400 Hz (Lowpass-Grenzfrequenz)
                self.ax3.set_xlim(x_min, x_max)
                self.ax3.set_xticks(visible_ticks)
                self.ax3.set_xticklabels([str(x) for x in visible_ticks])
                
                # Y-Achse: Maximum plus 42dB Bereich mit 6dB Raster (wie in PlotImpulse.py)
                if len(magnitude) > 0:
                    mag_max = np.max(magnitude)
                    y_max = np.ceil(mag_max / 6) * 6  # Runde auf nächstes 6dB-Vielfaches
                    y_min = y_max - 36
                    y_ticks = np.arange(y_min, y_max + 6, 6)
                    self.ax3.set_ylim(y_min, y_max)
                    self.ax3.set_yticks(y_ticks)
                
                self.ax3.grid(True, which='both', linestyle=':', alpha=0.5)
                self.ax3.set_xlabel('Frequency [Hz]', fontsize=8)
                self.ax3.set_ylabel('Magnitude [dB]', fontsize=8)
                self.ax3.tick_params(axis='both', labelsize=8, labelbottom=True)

                # (Debug-Inset und Daten-Dumps entfernt)
                            
            # Canvas neu zeichnen
            self.canvas.draw()
            
        except Exception as e:
            print(f"Fehler beim Plot-Update: {str(e)}")
            import traceback
            traceback.print_exc()

    def smooth_impulse_response(self, impulse_response, smoothing_factor=5):
        """Glättet die Impulsantwort mit einem Savitzky-Golay Filter
        
        Args:
            impulse_response: Die zu glättende Impulsantwort
            smoothing_factor: Stärke der Glättung (ungerade Zahl zwischen 3-15)
            
        Returns:
            Geglättete Impulsantwort
        """
        try:
            # Stelle sicher, dass smoothing_factor ungerade ist und in einem vernünftigen Bereich liegt
            if smoothing_factor % 2 == 0:
                smoothing_factor += 1
            smoothing_factor = max(3, min(15, smoothing_factor))
            
            # Nur glätten wenn die IR lang genug ist
            if len(impulse_response) > smoothing_factor:
                # Savitzky-Golay Filter für glatte Kurven ohne zu viel Dämpfung
                smoothed = signal.savgol_filter(impulse_response, smoothing_factor, 2)
                return smoothed
            else:
                return impulse_response
                
        except Exception as e:
            print(f"Fehler beim Glätten der IR: {e}")
            return impulse_response

    def calculate_optimal_plot_limits(self, freq, magnitude, freq_range):
        """Berechnet optimale Plot-Grenzen für gefilterte Daten
        
        Args:
            freq: Frequenz-Array
            magnitude: Magnitude-Array (bereits gefiltert)
            freq_range: Filter-Grenzen {'min': ..., 'max': ...}
            
        Returns:
            Dict mit optimalen Plot-Grenzen
        """
        try:
            f_min = freq_range['min']
            f_max = freq_range['max']
            
            # Y-Achse: Basierend auf tatsächlichen Magnitude-Werten
            # Finde den Bereich wo signifikante Energie vorhanden ist (>= -60 dB vom Maximum)
            mag_max = np.max(magnitude)
            significant_threshold = mag_max - 60  # 60 dB Dynamikbereich
            
            # Nur Punkte mit signifikanter Energie betrachten
            significant_mask = magnitude >= significant_threshold
            if np.any(significant_mask):
                mag_min_significant = np.min(magnitude[significant_mask])
                mag_max_significant = np.max(magnitude[significant_mask])
            else:
                # Fallback: alle Daten verwenden
                mag_min_significant = np.min(magnitude)
                mag_max_significant = np.max(magnitude)
            
            # Y-Padding: 10% der Spanne
            y_range = mag_max_significant - mag_min_significant
            y_padding = max(y_range * 0.1, 5)  # Mindestens 5 dB Padding
            
            y_min = mag_min_significant - y_padding
            y_max = mag_max_significant + y_padding
            
            # X-Achse: Erweitert um Filter-Wirkung zu zeigen
            # Berechne wo Filter -20 dB Dämpfung erreichen (gut sichtbar)
            target_attenuation = -20  # dB
            
            # Für HPF: Frequenz wo Filter -20 dB hat
            # -20 = 20 * log10(1/sqrt(1 + (fc/f)^4))
            # Löse nach f auf: f = fc / ((10^(-20/20))^(-2) - 1)^(1/4)
            hp_visible_ratio = 1.0 / (((10**(target_attenuation/20))**(-2) - 1)**(1/4))
            x_min = f_min * hp_visible_ratio
            
            # Für LPF: Frequenz wo Filter -20 dB hat  
            # f = fc * ((10^(-20/20))^(-2) - 1)^(1/4)
            lp_visible_ratio = (((10**(target_attenuation/20))**(-2) - 1)**(1/4))
            x_max = f_max * lp_visible_ratio
            
            # Stelle sicher, dass die Grenzen sinnvoll sind
            x_min = max(x_min, freq[0])  # Nicht unter verfügbare Daten
            x_max = min(x_max, freq[-1])  # Nicht über verfügbare Daten
            
            # Mindest-Bereich: 1 Dekade
            if x_max / x_min < 10:
                center_freq = np.sqrt(f_min * f_max)
                x_min = center_freq / 5
                x_max = center_freq * 5
            
            return {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            }
            
        except Exception as e:
            print(f"Fehler bei optimaler Plot-Berechnung: {e}")
            # Fallback: Standard-Werte
            return {
                'x_min': freq_range['min'] / 4,
                'x_max': freq_range['max'] * 4,
                'y_min': np.min(magnitude) - 10,
                'y_max': np.max(magnitude) + 10
            }

