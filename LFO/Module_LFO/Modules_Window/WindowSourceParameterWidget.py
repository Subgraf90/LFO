import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from Module_LFO.Modules_Init.ModuleBase import ModuleBase  # Importieren der Modulebase mit enthaltenem MihiAssi
from Module_LFO.Modules_Plot.PlotStacks2SourceLayout import StackDraw_SourceLayout
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QWidget
from PyQt5.QtGui import QColor, QFont
import os
import copy

# Widget zur Darstellung der Lautsprecheraufstellung mit Parameter-Tabelle

class Draw_Source_Layout_Widget(ModuleBase):
    def __init__(self, matplotlib_canvas_source_layout_widget, settings, container):
        super().__init__(settings)  # Korrekte Initialisierung der Basisklasse "Modulebase"
        
        # Speichere Canvas-Referenz
        self.canvas = matplotlib_canvas_source_layout_widget
        self.figure = self.canvas.figure
        self.settings = settings
        self.container = container
        self.im = None
        
        # Nur Plot im matplotlib canvas, Tabellen werden als Qt-Widgets erstellt
        self.figure.clear()
        self.ax_plot = self.figure.add_subplot(111)
        
        # Qt-Tabellen-Widgets (werden später im parent container erstellt)
        self.array_info_table_widget = None
        self.source_table_widget = None
        
        # KEINE TOOLBAR - Plot soll nur Darstellung sein

    def _create_export_button(self, parent_layout):
        """Erstellt einen PDF-Export-Button im Layout"""
        try:
            self.export_button = QPushButton("📄 PDF Export")
            self.export_button.setFixedHeight(35)
            self.export_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 12pt;
                    font-weight: bold;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
            """)
            self.export_button.clicked.connect(self.export_to_pdf)
            parent_layout.addWidget(self.export_button)
            
        except Exception as e:
            print(f"Fehler beim Erstellen des Export-Buttons: {e}")

    def update_source_layout_widget(self, selected_speaker_array_id):
        """Aktualisiert Plot und Tabellen mit aktuellen Source-Parametern"""
        try:
            if selected_speaker_array_id is None:
                return

            # Speichere selected_speaker_array_id für Export
            self.current_speaker_array_id = selected_speaker_array_id

            self.ax_plot.clear()
            
            speaker_array = self.settings.get_speaker_array(selected_speaker_array_id)
            cabinet_data = self.container.data['cabinet_data']
            speaker_names = self.container.data['speaker_names']
            
            if speaker_array is None:
                return

            # ========== PLOT-BEREICH ==========
            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            speaker_indices = []  # Speichere die Indizes der gültigen Lautsprecher
            
            for i, pattern_name in enumerate(speaker_array.source_polar_pattern):
                if not pattern_name:
                    continue
                    
                try:
                    speaker_index = speaker_names.index(pattern_name)
                    cabinet = cabinet_data[speaker_index]
                    
                    # Akzeptiere np.ndarray, list und dict
                    if isinstance(cabinet, (np.ndarray, list, dict)):
                        # Konvertiere zu np.ndarray für einheitliche Verarbeitung
                        if isinstance(cabinet, dict):
                            cabinet = np.array([cabinet])
                        elif isinstance(cabinet, list):
                            cabinet = np.array(cabinet)
                        array_cabinets.append(cabinet)
                        speaker_indices.append(i)  # Speichere den Index
                    else:
                        print(f"Cabinet-Daten für '{pattern_name}' haben falsches Format: {type(cabinet)}")
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Abrufen der Cabinet-Daten für '{pattern_name}': {str(e)}")
                    continue

            if array_cabinets:
                # WICHTIG: Y-Achse im Plot = Z-Position (Höhe), nicht Y-Position (Tiefe)
                stack_draw = StackDraw_SourceLayout(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_z,  # Z-Werte für Y-Achse im Plot (Höhe)
                    speaker_array.source_azimuth,
                    width=1, 
                    length=2,
                    window_restriction=0,
                    ax=self.ax_plot,
                    cabinet_data=array_cabinets
                )
                
                # Zeichne jeden Stack
                for isrc in range(len(array_cabinets)):
                    stack_draw.draw_stack(isrc)

                # Identifiziere die Stacks basierend auf den X-Positionen
                x_positions = [speaker_array.source_position_x[i] for i in speaker_indices]
                z_positions = [speaker_array.source_position_z[i] for i in speaker_indices]  # Z für Höhe
                
                # Gruppiere Lautsprecher in Stacks (Lautsprecher mit gleicher X-Position)
                stacks = {}
                for i, x in enumerate(x_positions):
                    # Runde auf 2 Dezimalstellen, um kleine Unterschiede zu ignorieren
                    x_rounded = round(x, 2)
                    if x_rounded not in stacks:
                        stacks[x_rounded] = []
                    stacks[x_rounded].append(speaker_indices[i])
                                
                # Berechne die linke Position jedes Stacks und sammle Höhen
                stack_left_positions_for_measurement = []
                stack_max_heights = []  # Maximale Höhe jedes Stacks
                
                for x_pos, indices in stacks.items():
                    # Sammle alle Breiten und Höhen der Lautsprecher im Stack
                    breiten = []
                    hoehen = []
                    for idx in indices:
                        speaker_name = speaker_array.source_polar_pattern[idx]
                        try:
                            speaker_idx = speaker_names.index(speaker_name)
                            cabinet = cabinet_data[speaker_idx]
                            if isinstance(cabinet, dict):
                                breiten.append(float(cabinet.get('width', 0)))
                                hoehen.append(float(cabinet.get('front_height', cabinet.get('height', 0))))
                            elif isinstance(cabinet, (np.ndarray, list)) and len(cabinet) > 0:
                                for cab in cabinet:
                                    if isinstance(cab, dict):
                                        breiten.append(float(cab.get('width', 0)))
                                        hoehen.append(float(cab.get('front_height', cab.get('height', 0))))
                        except Exception as e:
                            print(f'Fehler bei Breitenberechnung: {e}')
                            continue

                    gesamtbreite = sum(breiten)
                    left_pos = x_pos - (gesamtbreite / 2)
                    stack_left_positions_for_measurement.append(left_pos)
                    
                    # Berechne maximale Höhe (höchste z-Position + höchstes Cabinet)
                    max_z = max([speaker_array.source_position_z[idx] for idx in indices])
                    max_height_at_z = 0
                    for idx in indices:
                        if speaker_array.source_position_z[idx] == max_z:
                            # Hole Höhe dieses Cabinets
                            speaker_name = speaker_array.source_polar_pattern[idx]
                            try:
                                speaker_idx = speaker_names.index(speaker_name)
                                cabinet = cabinet_data[speaker_idx]
                                if isinstance(cabinet, dict):
                                    h = float(cabinet.get('front_height', cabinet.get('height', 0)))
                                    max_height_at_y = max(max_height_at_y, h)
                                elif isinstance(cabinet, (np.ndarray, list)) and len(cabinet) > 0:
                                    for cab in cabinet:
                                        if isinstance(cab, dict):
                                            h = float(cab.get('front_height', cab.get('height', 0)))
                                            max_height_at_z = max(max_height_at_z, h)
                            except:
                                pass
                    stack_max_heights.append(max_z + max_height_at_z)
                
                if stack_left_positions_for_measurement:
                    # Sortiere die Positionen
                    sorted_indices = np.argsort(stack_left_positions_for_measurement)
                    sorted_positions = [stack_left_positions_for_measurement[i] for i in sorted_indices]
                    sorted_heights = [stack_max_heights[i] for i in sorted_indices]
                    
                    first_position = sorted_positions[0]
                    max_stack_height = max(sorted_heights) if sorted_heights else 1.0
                    
                    # Linien-Höhe: über der höchsten Oberkante
                    line_top = max_stack_height + 0.4
                    text_y = line_top + 0.15
                    
                    for i, (x, stack_top) in enumerate(zip(sorted_positions, sorted_heights)):
                        # Gestrichelte Linie von Oberkante Stack nach oben
                        self.ax_plot.plot([x, x], [stack_top, line_top], color='black', linestyle='--', linewidth=0.8)
                        distance = np.abs(x - first_position)
                        # Text vertikal (90° gedreht, nach oben)
                        self.ax_plot.annotate(f"{distance:.2f} m", (x, text_y), color='black', 
                                             weight='bold', fontsize=7, ha='center', va='bottom', rotation=90)
                    
                    # Mittellinie (rot gestrichelt)
                    x_middle = 0
                    self.ax_plot.plot([x_middle, x_middle], [0, line_top + 0.3], color='red', linestyle='--', linewidth=1.0)
                    middle_distance = np.abs(x_middle - first_position)
                    # Rote Beschriftung höher, vertikal
                    self.ax_plot.annotate(f"{middle_distance:.2f} m", (x_middle, line_top + 0.45), color='red', 
                                         weight='bold', fontsize=8, ha='center', va='bottom', rotation=90)
                else:
                    print("DEBUG: Keine linken Positionen der Stacks für Bemaßung gefunden!")
            else:
                print("DEBUG: Keine Cabinet-Daten gefunden!")

            # Achsen-Einstellungen für Plot - MINIMAL, nur das Nötigste
            self.ax_plot.axis('equal')
            self.ax_plot.grid(False)  # Kein Grid
            self.ax_plot.set_xlabel("")  # Keine Achsenbeschriftung
            self.ax_plot.set_ylabel("")  # Keine Achsenbeschriftung
            
            # Entferne Achsenticks
            self.ax_plot.set_xticks([])
            self.ax_plot.set_yticks([])
            
            # Nur Titel
            self.ax_plot.set_title(f"Source Layout: {speaker_array.name}", fontsize=10, fontweight='bold', pad=10)
            
            # Horizontale Referenzlinie bei y=0 (Boden)
            xlim = self.ax_plot.get_xlim()
            self.ax_plot.plot(xlim, [0, 0], color='gray', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
            
            # Berechne optimale Plot-Grenzen basierend auf Source-Dimensionen
            if array_cabinets and speaker_indices:
                # Sammle alle X- und Y-Positionen der Sources
                all_x_positions = []
                max_top = 0  # Höchster Punkt (y + height)
                
                for idx in speaker_indices:
                    x = speaker_array.source_position_x[idx]
                    y = speaker_array.source_position_y[idx]
                    all_x_positions.append(x)
                    
                    # Hole Höhe des Cabinets und berechne Oberkante
                    speaker_name = speaker_array.source_polar_pattern[idx]
                    try:
                        speaker_idx = speaker_names.index(speaker_name)
                        cabinet = cabinet_data[speaker_idx]
                        if isinstance(cabinet, dict):
                            h = float(cabinet.get('front_height', cabinet.get('height', 0)))
                            max_top = max(max_top, y + h)
                        elif isinstance(cabinet, (np.ndarray, list)) and len(cabinet) > 0:
                            for cab in cabinet:
                                if isinstance(cab, dict):
                                    h = float(cab.get('front_height', cab.get('height', 0)))
                                    max_top = max(max_top, y + h)
                    except:
                        pass
                
                # Berechne Grenzen mit Rand
                if all_x_positions and max_top > 0:
                    x_min, x_max = min(all_x_positions), max(all_x_positions)
                    
                    # X-Range: 15% Rand links/rechts
                    x_range = x_max - x_min if x_max != x_min else 2.0
                    margin_x = x_range * 0.15
                    
                    # Y-Range: Mehr Platz oben für vertikale Beschriftungen
                    margin_y_top = 1.0  # 1m für vertikale Texte
                    
                    self.ax_plot.set_xlim(x_min - margin_x, x_max + margin_x)
                    self.ax_plot.set_ylim(-0.2, max_top + margin_y_top)
            
            # ========== TABELLEN-BEREICH (Qt-Widgets) ==========
            self._update_array_info_table(speaker_array)
            self._update_source_parameter_table(speaker_array, speaker_indices)
            
            # Canvas aktualisieren
            self.figure.canvas.draw()
            
        except Exception as e:
            print(f"Fehler in update_source_layout_widget: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _update_array_info_table(self, speaker_array):
        """
        Aktualisiert die Array-Informations-Tabelle (Qt-Widget).
        Zeigt: Array Name, Gain, Delay, Position X/Y/Z
        """
        if not self.array_info_table_widget:
            return
            
        # Array-Info-Daten vorbereiten
        array_info_data = [
            ['Array Name', speaker_array.name],
            ['Array Gain', f'{speaker_array.gain:.2f} dB'],
            ['Array Delay', f'{speaker_array.delay:.2f} ms'],
            ['Array Position X', f'{speaker_array.array_position_x:.2f} m'],
            ['Array Position Y', f'{speaker_array.array_position_y:.2f} m'],
            ['Array Position Z', f'{speaker_array.array_position_z:.2f} m']
        ]
        
        # Tabelle füllen
        self.array_info_table_widget.setRowCount(len(array_info_data))
        for i, (param, value) in enumerate(array_info_data):
            # Parameter-Spalte (fett und hellblau)
            param_item = QTableWidgetItem(param)
            param_item.setBackground(QColor('#E3F2FD'))
            param_item.setForeground(QColor('black'))  # Schwarze Schrift
            font = QFont()
            font.setBold(True)
            param_item.setFont(font)
            self.array_info_table_widget.setItem(i, 0, param_item)
            
            # Wert-Spalte
            value_item = QTableWidgetItem(str(value))
            value_item.setBackground(QColor('#FFFFFF'))
            value_item.setForeground(QColor('black'))  # Schwarze Schrift
            self.array_info_table_widget.setItem(i, 1, value_item)

    def _update_source_parameter_table(self, speaker_array, speaker_indices):
        """
        Aktualisiert die Source-Parameter-Tabelle (Qt-Widget).
        Pro Source (Stack/Enclosure) eine Zeile mit allen relevanten Parametern.
        """
        if not self.source_table_widget:
            return
            
        if not speaker_indices:
            self.source_table_widget.setRowCount(0)
            return
        
        # Hilfsfunktion für sicheren Array-Zugriff
        def safe_get(arr, index, default=0.0):
            """Sicherer Array-Zugriff mit Fallback"""
            try:
                if hasattr(arr, '__len__') and index < len(arr):
                    return arr[index]
                return default
            except (IndexError, TypeError):
                return default
        
        # Tabellen-Daten sammeln
        table_data = []
        for idx in speaker_indices:
            # Überspringe ungültige Indizes
            if idx >= len(speaker_array.source_polar_pattern):
                continue
                
            # Polarität als Symbol
            polarity = '-' if safe_get(speaker_array.source_polarity, idx, False) else '+'
            
            row = [
                f"{idx + 1}",  # Source-Nummer (1-basiert)
                safe_get(speaker_array.source_polar_pattern, idx, "N/A"),
                f"{safe_get(speaker_array.source_position_x, idx):.2f}",
                f"{safe_get(speaker_array.source_position_y, idx):.2f}",
                f"{safe_get(speaker_array.source_position_z, idx):.2f}",
                f"{safe_get(speaker_array.source_azimuth, idx):.1f}",
                f"{safe_get(speaker_array.source_site, idx):.1f}",
                f"{safe_get(speaker_array.source_level, idx):.2f}",
                f"{safe_get(speaker_array.source_time, idx):.2f}",
                polarity
            ]
            table_data.append(row)
        
        if not table_data:
            self.source_table_widget.setRowCount(0)
            return
        
        # Tabelle füllen
        self.source_table_widget.setRowCount(len(table_data))
        for i, row in enumerate(table_data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setForeground(QColor('black'))  # Schwarze Schrift
                
                # Abwechselnde Zeilenfarben
                if i % 2 == 0:
                    item.setBackground(QColor('#F5F5F5'))
                else:
                    item.setBackground(QColor('#FFFFFF'))
                
                # Polaritäts-Spalte extra einfärben
                if j == 9:  # Polaritäts-Spalte
                    if value == '-':
                        item.setBackground(QColor('#FFCDD2'))  # Rot für negativ
                    else:
                        item.setBackground(QColor('#C8E6C9'))  # Grün für positiv
                
                self.source_table_widget.setItem(i, j, item)

    def export_to_pdf(self):
        """Exportiert Plot und Tabellen als mehrseitiges PDF im Hochformat"""
        try:
            # Öffne Dateidialog zum Speichern
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "PDF Export - Source Parameter",
                os.path.expanduser("~"),
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return  # Benutzer hat abgebrochen
            
            # Stelle sicher, dass die Datei .pdf Endung hat
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            
            # Hole Speaker Array für Daten
            speaker_array = self.settings.get_speaker_array(self.current_speaker_array_id)
            
            # Sammle Source-Daten
            speaker_indices = []
            for i, pattern_name in enumerate(speaker_array.source_polar_pattern):
                if pattern_name:
                    speaker_indices.append(i)
            
            # Erstelle mehrseitiges PDF im Hochformat
            with PdfPages(file_path) as pdf:
                # ========== SEITE 1: Plot + Array-Info + Teil der Source-Tabelle ==========
                self._export_first_page(pdf, speaker_array, speaker_indices)
                
                # ========== WEITERE SEITEN: Fortsetzung der Source-Tabelle falls nötig ==========
                num_sources = len(speaker_indices)
                sources_per_page = 30  # Mehr Sources pro Folgeseite
                
                if num_sources > 15:
                    # Erstelle Folgeseiten für verbleibende Sources
                    for start_idx in range(15, num_sources, sources_per_page):
                        end_idx = min(start_idx + sources_per_page, num_sources)
                        self._export_continuation_page(pdf, speaker_array, speaker_indices, start_idx, end_idx)
                
                # Füge Metadaten hinzu
                d = pdf.infodict()
                d['Title'] = 'Source Parameter Layout'
                d['Author'] = 'LFO - Low Frequency Optimizer'
                d['Subject'] = f'Array: {speaker_array.name}'
                d['Keywords'] = 'Speaker Array, Source Parameters'
                from datetime import datetime
                d['CreationDate'] = datetime.now()
            
            print(f"PDF erfolgreich exportiert: {file_path}")
            
            # Zeige Erfolgsmeldung
            from PyQt5.QtWidgets import QMessageBox
            num_pages = 1 + max(0, (num_sources - 15 + sources_per_page - 1) // sources_per_page)
            QMessageBox.information(
                None,
                "Export erfolgreich",
                f"PDF wurde gespeichert unter:\n{file_path}\n\n"
                f"Seiten: {num_pages}\n"
                f"Sources: {num_sources}"
            )
            
        except Exception as e:
            print(f"Fehler beim PDF-Export: {e}")
            import traceback
            traceback.print_exc()
            
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "Export fehlgeschlagen",
                f"Fehler beim Exportieren:\n{str(e)}"
            )
    
    def _export_first_page(self, pdf, speaker_array, speaker_indices):
        """Erstellt die erste PDF-Seite mit Plot, Array-Info und ersten Sources"""
        # Erstelle neue Figure im Hochformat (Portrait A4: 8.27 x 11.69 inches)
        fig = plt.figure(figsize=(8.27, 11.69))
        
        # GridSpec: Plot (oben, groß), Array-Info, Source-Tabelle (erste 15 Zeilen)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 0.8, 5.2], hspace=0.4)
        
        ax_plot = fig.add_subplot(gs[0])
        ax_array = fig.add_subplot(gs[1])
        ax_sources = fig.add_subplot(gs[2])
        
        ax_array.axis('off')
        ax_sources.axis('off')
        
        # Kopiere Plot-Inhalt
        self._copy_plot_to_axis(ax_plot)
        
        # Erstelle Array-Info-Tabelle
        self._create_array_info_table_for_export(ax_array, speaker_array)
        
        # Erstelle Source-Tabelle (erste 15 Einträge)
        self._create_source_table_for_export(ax_sources, speaker_array, speaker_indices, 0, 15)
        
        # Speichere Seite
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _export_continuation_page(self, pdf, speaker_array, speaker_indices, start_idx, end_idx):
        """Erstellt eine Fortsetzungs-Seite für weitere Sources"""
        # Erstelle neue Figure im Hochformat
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Titel für Fortsetzungsseite
        fig.suptitle(f'Source Parameters (continued) - Sources {start_idx + 1} to {end_idx}', 
                    fontsize=12, fontweight='bold')
        
        # Erstelle Source-Tabelle für diesen Bereich
        self._create_source_table_for_export(ax, speaker_array, speaker_indices, start_idx, end_idx)
        
        # Speichere Seite
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _copy_plot_to_axis(self, target_ax):
        """Kopiert den Plot-Inhalt in eine neue Achse"""
        # Hole Plot-Daten und kopiere sie
        for line in self.ax_plot.get_lines():
            target_ax.plot(line.get_xdata(), line.get_ydata(), 
                          color=line.get_color(), linestyle=line.get_linestyle(),
                          linewidth=line.get_linewidth())
        
        # Kopiere Annotationen - VORSICHTIG: nur kompatible Properties verwenden
        for text in self.ax_plot.texts:
            # Verwende nur grundlegende Text-Properties, nicht Annotation-spezifische
            target_ax.text(
                text.get_position()[0], 
                text.get_position()[1], 
                text.get_text(),
                color=text.get_color(),
                fontsize=text.get_fontsize(),
                fontweight=text.get_weight(),
                ha=text.get_ha(),
                va=text.get_va(),
                rotation=text.get_rotation()
            )
        
        # Kopiere Patches (Rechtecke für Cabinet-Darstellung)
        for patch in self.ax_plot.patches:
            # Kopiere Patch direkt (robuster als manuelle Rekonstruktion)
            patch_copy = copy.copy(patch)
            patch_copy.set_transform(target_ax.transData)
            target_ax.add_patch(patch_copy)
        
        # Kopiere Achsen-Einstellungen
        target_ax.set_xlim(self.ax_plot.get_xlim())
        target_ax.set_ylim(self.ax_plot.get_ylim())
        target_ax.set_xlabel(self.ax_plot.get_xlabel())
        target_ax.set_ylabel(self.ax_plot.get_ylabel())
        target_ax.set_title(self.ax_plot.get_title())
        target_ax.grid(True, color='k', linestyle='-', linewidth=0.3)
        target_ax.axis('equal')
    
    def _create_array_info_table_for_export(self, ax, speaker_array):
        """Erstellt Array-Info-Tabelle für PDF-Export"""
        info_data = [
            ['Array Name', speaker_array.name],
            ['Array Gain', f'{speaker_array.gain:.2f} dB'],
            ['Array Delay', f'{speaker_array.delay:.2f} ms'],
            ['Array Position X', f'{speaker_array.array_position_x:.2f} m'],
            ['Array Position Y', f'{speaker_array.array_position_y:.2f} m'],
            ['Array Position Z', f'{speaker_array.array_position_z:.2f} m'],
        ]
        
        table = ax.table(
            cellText=info_data,
            colLabels=['Parameter', 'Wert'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4],
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Header
        for i in range(2):
            cell = table[(0, i)]
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')
        
        # Datenzeilen
        for i in range(len(info_data)):
            for j in range(2):
                cell = table[(i + 1, j)]
                cell.set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
    
    def _create_source_table_for_export(self, ax, speaker_array, speaker_indices, start_idx, end_idx):
        """Erstellt Source-Parameter-Tabelle für PDF-Export"""
        col_labels = ['#', 'Typ', 'X[m]', 'Y[m]', 'Z[m]', 'Az[°]', 'Site[°]', 'Lvl[dB]', 'Dly[ms]', 'Pol']
        
        def safe_get(arr, index, default=0.0):
            try:
                if hasattr(arr, '__len__') and index < len(arr):
                    return arr[index]
                return default
            except:
                return default
        
        table_data = []
        for i in range(start_idx, min(end_idx, len(speaker_indices))):
            idx = speaker_indices[i]
            
            polarity = '-' if safe_get(speaker_array.source_polarity, idx, False) else '+'
            
            row = [
                f"{idx + 1}",
                str(safe_get(speaker_array.source_polar_pattern, idx, ''))[:12],  # Kürze lange Namen
                f"{safe_get(speaker_array.source_position_x, idx):.2f}",
                f"{safe_get(speaker_array.source_position_y, idx):.2f}",
                f"{safe_get(speaker_array.source_position_z, idx):.2f}",
                f"{safe_get(speaker_array.source_azimuth, idx):.1f}",
                f"{safe_get(speaker_array.source_site, idx):.1f}",
                f"{safe_get(speaker_array.source_level, idx):.2f}",
                f"{safe_get(speaker_array.source_time, idx):.2f}",
                polarity
            ]
            table_data.append(row)
        
        if not table_data:
            return
        
        col_widths = [0.05, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05]
        
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colWidths=col_widths,
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        
        # Header
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white', fontsize=7)
        
        # Datenzeilen
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                cell = table[(i + 1, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                
                # Polarität farblich kennzeichnen
                if j == len(col_labels) - 1:
                    if table_data[i][j] == '-':
                        cell.set_facecolor('#ffcccc')
                    else:
                        cell.set_facecolor('#ccffcc')
