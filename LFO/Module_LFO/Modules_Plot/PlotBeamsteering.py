import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize
from Module_LFO.Modules_Plot.PlotStacks2Beamsteering import StackDraw_Beamsteering


class BeamsteeringPlot(QWidget):
    def __init__(self, parent=None, settings=None, container=None, width=6, height=4, dpi=100):
        super().__init__(parent)
        self.settings = settings
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Zweite Y-Achse für Lautsprecher mit aspect='equal'
        self.ax_speakers = None  # Wird in beamsteering_plot() erstellt
        self.container = container
        self.figure.tight_layout()
        
        # Zoom-Callback-Verwaltung
        self._zoom_callback_connected = False
        self._updating_ticks = False
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 200)
        
        # Verbinde Zoom-Callbacks für adaptive Achsenbeschriftung
        self._connect_zoom_callbacks()


    def beamsteering_plot(self, speaker_array_id):
        self.ax.clear()  # Löschen des vorherigen Plots
        
        # Entferne alte zweite Y-Achse falls vorhanden
        if self.ax_speakers is not None:
            try:
                # Entferne alle Patches von der alten Achse
                if self.ax_speakers.patches:
                    for patch in list(self.ax_speakers.patches):
                        try:
                            patch.remove()
                        except:
                            pass
                    self.ax_speakers.patches.clear()
                # Entferne die Achse selbst
                self.ax_speakers.remove()
            except:
                pass
            self.ax_speakers = None
        
        speaker_array = self.settings.get_speaker_array(speaker_array_id)

        if speaker_array is None:
            self.ax.set_title("Kein gültiges SpeakerArray gefunden")
            self.canvas.draw_idle()
            return
        
        # Extrahieren der Lautsprecherpositionen
        source_position_y = self._to_float_array(speaker_array.source_position_y)
        source_position_x = self._to_float_array(speaker_array.source_position_x)
        virtual_source_position_x = self._to_float_array(speaker_array.virtual_source_position_x)
        virtual_source_position_y = self._to_float_array(speaker_array.virtual_source_position_y)

        # Plot der virtuellen Lautsprecherpositionen
        self.ax.plot(virtual_source_position_x, virtual_source_position_y, 'o-', label='Virtual Source Positions')

        # Zeichnen gestrichelter Linien von realen zu virtuellen Lautsprecherpositionen
        for sx, sy, vx, vy in zip(source_position_x, source_position_y, virtual_source_position_x, virtual_source_position_y):
            self.ax.plot([sx, vx], [sy, vy], linestyle='--', color='grey')

        x_values = self._concatenate_arrays(source_position_x, virtual_source_position_x)
        y_values = self._concatenate_arrays(source_position_y, virtual_source_position_y)

        if x_values.size:
            x_min = float(np.min(x_values))
            x_max = float(np.max(x_values))
            # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
            current_xlim = self.ax.get_xlim()
            if abs(current_xlim[0] - x_min) > 0.01 or abs(current_xlim[1] - x_max) > 0.01:
                # Bereits gezoomt - verwende sichtbare Grenzen
                x_min_visible, x_max_visible = current_xlim
                x_ticks, step = self._calculate_metric_ticks(x_min_visible, x_max_visible)
            else:
                # Nicht gezoomt - verwende Daten-Grenzen (exakt, ohne Padding)
                x_ticks, step = self._calculate_metric_ticks(x_min, x_max)
                self.ax.set_xlim(x_min, x_max)
            self.ax.set_xticks(x_ticks)

        if y_values.size:
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            # #region agent log - Daten-Grenzen
            import json
            import time
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'PlotBeamsteering.py:98',
                    'message': 'Y-Daten-Grenzen vor Limit-Berechnung',
                    'data': {
                        'y_min_data': y_min,
                        'y_max_data': y_max,
                        'y_range_data': y_max - y_min
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
            # #endregion
            # Prüfe ob bereits gezoomt wurde - verwende dann die aktuellen Grenzen
            current_ylim = self.ax.get_ylim()
            if abs(current_ylim[0] - y_min) > 0.01 or abs(current_ylim[1] - y_max) > 0.01:
                # Bereits gezoomt - verwende sichtbare Grenzen
                y_min_visible, y_max_visible = current_ylim
                y_ticks, step = self._calculate_metric_ticks(y_min_visible, y_max_visible)
            else:
                # Nicht gezoomt - verwende Daten-Grenzen
                # WICHTIG: Synchronisiere Y=0 zwischen beiden Achsen
                # Die Lautsprecher-Achse ist symmetrisch um Y=0 (Y=0 bei 50%)
                # Die Hauptachse sollte auch so gesetzt werden, dass Y=0 bei 50% ist
                y_range_data = y_max - y_min
                padding_bottom = max(y_range_data * 0.15, 0.2)  # Mindestens 0.2m unten
                padding_top = 0.1  # Minimal oben (wird später durch _adjust_ylim_for_speakers angepasst)
                
                # Berechne die benötigte Range, um alle Daten + Padding zu enthalten
                required_range_below_zero = abs(y_min) + padding_bottom  # Bereich unter Y=0
                required_range_above_zero = max(y_max, 0.0) + padding_top  # Bereich über Y=0
                
                # Verwende die größere Range für symmetrische Limits um Y=0
                # Dies stellt sicher, dass Y=0 bei 50% ist (wie auf der Lautsprecher-Achse)
                max_range = max(required_range_below_zero, required_range_above_zero)
                
                # Setze symmetrische Limits um Y=0, die alle Daten + Padding enthalten
                y_min_centered = -max_range
                y_max_centered = max_range
                
                # Stelle sicher, dass die ursprünglichen Daten-Grenzen enthalten sind
                y_min_centered = min(y_min_centered, y_min - padding_bottom)
                y_max_centered = max(y_max_centered, y_max + padding_top)
                
                # Stelle sicher, dass Y=0 enthalten ist (sollte immer der Fall sein)
                if y_min_centered > 0.0:
                    y_min_centered = -max_range
                if y_max_centered < 0.0:
                    y_max_centered = max_range
                # #region agent log - Vor _calculate_metric_ticks
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'PlotBeamsteering.py:125',
                        'message': 'Y-Limits vor _calculate_metric_ticks',
                        'data': {
                            'y_min_centered': y_min_centered,
                            'y_max_centered': y_max_centered,
                            'y_range_centered': y_max_centered - y_min_centered,
                            'max_abs_deviation': max_abs_deviation,
                            'max_abs_positive': max_abs_positive,
                            'max_abs_negative': max_abs_negative
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                y_ticks, step = self._calculate_metric_ticks(y_min_centered, y_max_centered)
                # #region agent log - Nach _calculate_metric_ticks
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'C',
                        'location': 'PlotBeamsteering.py:127',
                        'message': 'Y-Limits nach _calculate_metric_ticks',
                        'data': {
                            'y_ticks': [float(t) for t in y_ticks],
                            'step': float(step),
                            'y_min_centered': y_min_centered,
                            'y_max_centered': y_max_centered
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                self.ax.set_ylim(y_min_centered, y_max_centered)
                # #region agent log - Nach set_ylim
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'PlotBeamsteering.py:129',
                        'message': 'Y-Limits nach set_ylim (initial)',
                        'data': {
                            'ax_ylim_after_set': list(self.ax.get_ylim()),
                            'y_min_centered': y_min_centered,
                            'y_max_centered': y_max_centered
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
            self.ax.set_yticks(y_ticks)
            # Formatierung auf eine Nachkommastelle
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

        # KEIN set_aspect('equal') - würde die Plot-Höhe stark reduzieren
        # Die Lautsprecher werden durch Skalierung in StackDraw_Beamsteering unverzerrt dargestellt

        # Finale Plot-Einstellungen
        # KEIN axis('equal') - Plot hat feste Breite, beide Achsen sind in Metern
        # Lautsprecher werden durch Transform in StackDraw_Beamsteering unverzerrt dargestellt
        # Plot hat feste Breite und Höhe (800x220 Pixel) durch setFixedWidth/setFixedHeight in UiSourceManagement.py
        
        # WICHTIG: Layout-Anpassungen VOR dem Zeichnen der Stacks, damit Pixel-Größe korrekt ist
        self.ax.grid(color='k', linestyle='-', linewidth=0.4)
        self.ax.set_ylabel("Virtual position [m]", fontsize=6)
        self.ax.set_xlabel("Arc width [m]", fontsize=6)
        self.ax.tick_params(axis='both', which='both', labelsize=6)
        
        # Layout-Anpassungen für konsistente Größe (muss VOR Stack-Zeichnung sein!)
        self._apply_layout()
        
        # Erstelle zweite Y-Achse für Lautsprecher mit aspect='equal'
        # Diese teilt sich die X-Achse mit der Hauptachse
        self.ax_speakers = self.ax.twinx()
        # Synchronisiere X-Limits mit Hauptachse (muss VOR set_aspect gesetzt werden)
        xlim = self.ax.get_xlim()
        self.ax_speakers.set_xlim(xlim)
        
        # Berechne Y-Limits basierend auf X-Limits für aspect='equal'
        # Ziel: Y=0 soll immer in der Mitte oder am unteren Rand sein
        # Für aspect='equal' müssen Y-Limits so gesetzt werden, dass das Aspect-Ratio 1:1 ist
        x_range = xlim[1] - xlim[0]
        # Verwende die gleiche Range für Y, zentriert bei 0
        # Dies stellt sicher, dass Y=0 immer die gleiche Position hat, unabhängig von X
        y_range = x_range  # Gleiche Range für 1:1 Aspect-Ratio
        y_center = 0.0  # Zentriert bei 0
        y_min = y_center - y_range / 2
        y_max = y_center + y_range / 2
        self.ax_speakers.set_ylim(y_min, y_max)
        
        # Setze aspect='equal' für unverzerrte Lautsprecher-Darstellung
        # WICHTIG: Verwende adjustable='datalim' für twinned Axes (adjustable='box' ist nicht erlaubt)
        # Die Y-Limits werden basierend auf den Daten angepasst, um das 1:1 Aspect-Ratio zu erreichen
        # #region agent log - Y-Limits vor set_aspect
        import json
        import time
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'D',
                'location': 'PlotBeamsteering.py:152',
                'message': 'Y-Limits vor set_aspect',
                'data': {
                    'ax_speakers_ylim_before': list(self.ax_speakers.get_ylim()),
                    'ax_speakers_xlim': list(self.ax_speakers.get_xlim())
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        self.ax_speakers.set_aspect('equal', adjustable='datalim')
        # Y-Achse der zweiten Achse unsichtbar machen (wird nur für aspect='equal' verwendet)
        self.ax_speakers.set_yticks([])
        self.ax_speakers.spines['right'].set_visible(False)
        self.ax_speakers.spines['top'].set_visible(False)
        self.ax_speakers.spines['left'].set_visible(False)
        self.ax_speakers.spines['bottom'].set_visible(False)
        
        # Zeichne einmal, um sicherzustellen, dass get_window_extent() korrekte Werte liefert
        # (nach _apply_layout(), damit subplots_adjust() bereits angewendet wurde)
        self.figure.canvas.draw()
        
        # #region agent log - Y-Limits nach set_aspect und draw
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'D',
                'location': 'PlotBeamsteering.py:170',
                'message': 'Y-Limits nach set_aspect und draw',
                'data': {
                    'ax_speakers_ylim_after': list(self.ax_speakers.get_ylim()),
                    'ax_speakers_xlim': list(self.ax_speakers.get_xlim())
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
        # WICHTIG: Nach dem Zeichnen die Y-Limits nochmal setzen, damit Y=0 zentriert bleibt
        # set_aspect('equal') kann die Y-Limits ändern, daher müssen wir sie zurücksetzen
        xlim_after = self.ax_speakers.get_xlim()
        x_range_after = xlim_after[1] - xlim_after[0]
        y_range_after = x_range_after  # Gleiche Range für 1:1 Aspect-Ratio
        y_center = 0.0
        y_min_after = y_center - y_range_after / 2
        y_max_after = y_center + y_range_after / 2
        self.ax_speakers.set_ylim(y_min_after, y_max_after)
        # #region agent log - Y-Limits nach manueller Setzung
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'D',
                'location': 'PlotBeamsteering.py:180',
                'message': 'Y-Limits nach manueller Setzung',
                'data': {
                    'ax_speakers_ylim_final': list(self.ax_speakers.get_ylim()),
                    'y_min_after': float(y_min_after),
                    'y_max_after': float(y_max_after),
                    'y_center': float(y_center)
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
        # #region agent log
        import json
        bbox = self.ax.get_window_extent()
        axes_width_pixels = bbox.width
        axes_height_pixels = bbox.height
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'PlotBeamsteering.py:99',
                'message': 'Beamsteering: Achsengrenzen und Pixel-Größe nach Layout-Anpassung, vor Stack-Zeichnung',
                'data': {
                    'xlim': list(self.ax.get_xlim()),
                    'ylim': list(self.ax.get_ylim()),
                    'axes_width_pixels': float(axes_width_pixels),
                    'axes_height_pixels': float(axes_height_pixels)
                },
                'timestamp': int(__import__('time').time() * 1000)
            }) + '\n')
        # #endregion
        
        # Zeichne Stacks NACH Layout-Anpassung (für korrekte Pixel-Größe)
        # Verwende ax_speakers für Lautsprecher-Darstellung
        self.plot_Stacks2Beamsteering(speaker_array_id)
        
        # #region agent log - Vor _adjust_ylim_for_speakers
        import json
        import time
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'PlotBeamsteering.py:269',
                'message': 'Y-Limits vor _adjust_ylim_for_speakers',
                'data': {
                    'ax_ylim_before_adjust': list(self.ax.get_ylim()),
                    'ax_speakers_ylim_before_adjust': list(self.ax_speakers.get_ylim()) if self.ax_speakers is not None else None
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        # Passe Y-Achsen-Grenzen an, damit Lautsprecher nicht abgeschnitten werden
        self._adjust_ylim_for_speakers()
        # #region agent log - Nach _adjust_ylim_for_speakers
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'PlotBeamsteering.py:272',
                'message': 'Y-Limits nach _adjust_ylim_for_speakers',
                'data': {
                    'ax_ylim_after_adjust': list(self.ax.get_ylim()),
                    'ax_speakers_ylim_after_adjust': list(self.ax_speakers.get_ylim()) if self.ax_speakers is not None else None
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
        # WICHTIG: Stelle sicher, dass die Hauptachse bei Y=0 zentriert bleibt
        # _adjust_ylim_for_speakers() könnte die Y-Limits geändert haben
        # ABER: Stelle sicher, dass alle Daten noch enthalten sind
        current_ax_ylim = self.ax.get_ylim()
        ax_y_range = current_ax_ylim[1] - current_ax_ylim[0]
        ax_y_center = (current_ax_ylim[0] + current_ax_ylim[1]) / 2
        
        # Hole die ursprünglichen Daten-Grenzen
        y_values = self._concatenate_arrays(
            self._to_float_array(speaker_array.source_position_y),
            self._to_float_array(speaker_array.virtual_source_position_y)
        )
        if y_values.size:
            y_min_data = float(np.min(y_values))
            y_max_data = float(np.max(y_values))
        else:
            y_min_data = current_ax_ylim[0]
            y_max_data = current_ax_ylim[1]
        
        # OPTIMIERT: Prüfe ob Limits angepasst werden müssen, um Daten vollständig anzuzeigen
        # Statt symmetrischer Zentrierung: Stelle sicher, dass alle Daten + Padding enthalten sind
        # #region agent log - Vor finaler Anpassung
        with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'A',
                'location': 'PlotBeamsteering.py:393',
                'message': 'Vor finaler Anpassung',
                'data': {
                    'current_ax_ylim': list(current_ax_ylim),
                    'ax_y_center': float(ax_y_center),
                    'y_min_data': float(y_min_data),
                    'y_max_data': float(y_max_data)
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
        # WICHTIG: Synchronisiere Y=0 zwischen beiden Achsen
        # Die Lautsprecher-Achse ist symmetrisch um Y=0 (Y=0 bei 50%)
        # Die Hauptachse sollte auch so gesetzt werden, dass Y=0 bei 50% ist
        # Berechne optimale Limits: Y=0 als Ankerpunkt in der Mitte, Daten + Padding
        y_range_data = y_max_data - y_min_data
        padding_bottom = max(y_range_data * 0.15, 0.2)  # 15% oder mindestens 0.2m unten
        padding_top = 0.1  # Minimal oben (wird durch _adjust_ylim_for_speakers angepasst)
        
        # Berechne die benötigte Range, um alle Daten + Padding zu enthalten
        required_range_below_zero = abs(y_min_data) + padding_bottom  # Bereich unter Y=0
        required_range_above_zero = max(y_max_data, 0.0) + padding_top  # Bereich über Y=0
        
        # Verwende die größere Range für symmetrische Limits um Y=0
        # Dies stellt sicher, dass Y=0 bei 50% ist (wie auf der Lautsprecher-Achse)
        max_range = max(required_range_below_zero, required_range_above_zero)
        
        # Setze symmetrische Limits um Y=0, die alle Daten + Padding enthalten
        y_min_optimal = -max_range
        y_max_optimal = max_range
        
        # Stelle sicher, dass die ursprünglichen Daten-Grenzen enthalten sind
        y_min_optimal = min(y_min_optimal, y_min_data - padding_bottom)
        y_max_optimal = max(y_max_optimal, y_max_data + padding_top)
        
        # Stelle sicher, dass Y=0 enthalten ist (sollte immer der Fall sein)
        if y_min_optimal > 0.0:
            y_min_optimal = -max_range
        if y_max_optimal < 0.0:
            y_max_optimal = max_range
        
        # Prüfe ob Anpassung nötig ist
        if abs(y_min_optimal - current_ax_ylim[0]) > 0.01 or abs(y_max_optimal - current_ax_ylim[1]) > 0.01:
            # #region agent log - Vor _calculate_metric_ticks in finaler Anpassung
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'PlotBeamsteering.py:404',
                    'message': 'Finale Anpassung: Vor _calculate_metric_ticks',
                    'data': {
                        'y_min_optimal': float(y_min_optimal),
                        'y_max_optimal': float(y_max_optimal),
                        'y_range_optimal': float(y_max_optimal - y_min_optimal),
                        'padding_bottom': float(padding_bottom),
                        'padding_top': float(padding_top)
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
            # #endregion
            y_ticks, step = self._calculate_metric_ticks(y_min_optimal, y_max_optimal)
            self.ax.set_ylim(y_min_optimal, y_max_optimal)
            self.ax.set_yticks(y_ticks)
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
            # #region agent log - Nach finaler Anpassung
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'PlotBeamsteering.py:410',
                    'message': 'Nach finaler Anpassung',
                    'data': {
                        'ax_ylim_final': list(self.ax.get_ylim()),
                        'y_ticks': [float(t) for t in y_ticks],
                        'y_min_optimal': float(y_min_optimal),
                        'y_max_optimal': float(y_max_optimal)
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
            # #endregion
        
        # WICHTIG: Nach dem Zeichnen die Y-Limits der zweiten Achse nochmal setzen,
        # damit Y=0 zentriert bleibt (set_aspect kann die Limits ändern)
        if self.ax_speakers is not None:
            xlim_final = self.ax_speakers.get_xlim()
            x_range_final = xlim_final[1] - xlim_final[0]
            y_range_final = x_range_final  # Gleiche Range für 1:1 Aspect-Ratio
            y_center = 0.0
            y_min_final = y_center - y_range_final / 2
            y_max_final = y_center + y_range_final / 2
            self.ax_speakers.set_ylim(y_min_final, y_max_final)
            # Setze aspect='equal' erneut mit adjustable='datalim' (für twinned Axes erforderlich)
            self.ax_speakers.set_aspect('equal', adjustable='datalim')
        
        # WICHTIG: Zeichne nochmal, damit set_aspect angewendet wird, dann setze Y-Limits erneut
        self.figure.canvas.draw()
        
        # WICHTIG: Nach dem finalen draw() die Y-Limits nochmal setzen, damit Y=0 zentriert bleibt
        # set_aspect('equal') kann die Y-Limits nach draw() ändern
        if self.ax_speakers is not None:
            xlim_very_final = self.ax_speakers.get_xlim()
            x_range_very_final = xlim_very_final[1] - xlim_very_final[0]
            y_range_very_final = x_range_very_final  # Gleiche Range für 1:1 Aspect-Ratio
            y_center = 0.0
            y_min_very_final = y_center - y_range_very_final / 2
            y_max_very_final = y_center + y_range_very_final / 2
            self.ax_speakers.set_ylim(y_min_very_final, y_max_very_final)
            # #region agent log - Y-Limits nach finalem draw und Setzung + Achsen-Differenz
            import json
            import time
            import matplotlib.patches as patches
            import matplotlib.transforms as transforms
            patch_positions_final = []
            for patch in self.ax_speakers.patches:
                if isinstance(patch, patches.Rectangle):
                    patch_positions_final.append({
                        'y': float(patch.get_y()),
                        'x': float(patch.get_x()),
                        'height': float(patch.get_height()),
                        'width': float(patch.get_width())
                    })
            # Berechne Differenz zwischen den beiden Y-Achsen
            ax_ylim = self.ax.get_ylim()
            ax_speakers_ylim = self.ax_speakers.get_ylim()
            ax_y_range = ax_ylim[1] - ax_ylim[0]
            ax_speakers_y_range = ax_speakers_ylim[1] - ax_speakers_ylim[0]
            
            # Y=0 in beiden Achsen-Koordinaten
            y_zero_in_ax_speakers = 0.0
            y_zero_in_ax = 0.0  # Sollte auch 0 sein, aber prüfe die tatsächliche Position
            
            # Normalisierte Position von Y=0 in beiden Achsen (0.0 = unten, 1.0 = oben)
            y_zero_normalized_ax_speakers = (y_zero_in_ax_speakers - ax_speakers_ylim[0]) / ax_speakers_y_range if ax_speakers_y_range > 0 else 0.5
            y_zero_normalized_ax = (y_zero_in_ax - ax_ylim[0]) / ax_y_range if ax_y_range > 0 else 0.5
            
            # Differenz in normalisierten Koordinaten (sollte 0 sein, wenn beide bei Y=0 zentriert sind)
            y_zero_diff_normalized = y_zero_normalized_ax_speakers - y_zero_normalized_ax
            
            # Transform-Koordinaten: Wie wird Y=0 von ax_speakers in ax transformiert?
            # Verwende matplotlib Transform, um die visuelle Position zu berechnen
            try:
                # Transform von ax_speakers Daten-Koordinaten zu ax Daten-Koordinaten
                # Da beide die gleiche X-Achse teilen, müssen wir nur die Y-Koordinaten transformieren
                # Für twinx(): Die X-Koordinaten sind gleich, aber Y-Koordinaten sind unterschiedlich
                # Die visuelle Position hängt von den Y-Limits ab
                
                # Berechne die visuelle Position von Y=0 in beiden Achsen
                # In normalisierten Koordinaten (0-1): 0 = unten, 1 = oben
                # ax_speakers: Y=0 ist bei normalisierter Position = (0 - (-13.5135)) / (13.5135 - (-13.5135)) = 13.5135 / 27.027 = 0.5
                # ax: Y=0 ist bei normalisierter Position = (0 - ax_ylim[0]) / ax_y_range
                
                # Die visuelle Position ist gleich, wenn beide Achsen die gleiche normale Position für Y=0 haben
                visual_y_zero_ax_speakers = y_zero_normalized_ax_speakers
                visual_y_zero_ax = y_zero_normalized_ax
                visual_y_zero_diff = visual_y_zero_ax_speakers - visual_y_zero_ax
            except Exception as e:
                visual_y_zero_diff = None
            
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'G',
                    'location': 'PlotBeamsteering.py:220',
                    'message': 'Y-Achsen-Differenz-Analyse',
                    'data': {
                        'ax_ylim': list(ax_ylim),
                        'ax_speakers_ylim': list(ax_speakers_ylim),
                        'ax_y_range': float(ax_y_range),
                        'ax_speakers_y_range': float(ax_speakers_y_range),
                        'y_zero_analysis': {
                            'y_zero_in_ax': float(y_zero_in_ax),
                            'y_zero_in_ax_speakers': float(y_zero_in_ax_speakers),
                            'y_zero_normalized_ax': float(y_zero_normalized_ax),
                            'y_zero_normalized_ax_speakers': float(y_zero_normalized_ax_speakers),
                            'y_zero_diff_normalized': float(y_zero_diff_normalized),
                            'visual_y_zero_diff': float(visual_y_zero_diff) if visual_y_zero_diff is not None else None
                        },
                        'ax_ylim_center': float((ax_ylim[0] + ax_ylim[1]) / 2),
                        'ax_speakers_ylim_center': float((ax_speakers_ylim[0] + ax_speakers_ylim[1]) / 2),
                        'ax_ylim_offset_from_zero': float(0.0 - (ax_ylim[0] + ax_ylim[1]) / 2),
                        'ax_speakers_ylim_offset_from_zero': float(0.0 - (ax_speakers_ylim[0] + ax_speakers_ylim[1]) / 2),
                        'patch_positions': patch_positions_final[:3] if len(patch_positions_final) > 3 else patch_positions_final,  # Nur erste 3 für Übersicht
                        'num_patches': len(self.ax_speakers.patches)
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
            # #endregion
        
        self.canvas.draw_idle()


    def plot_Stacks2Beamsteering(self, speaker_array_id):
        try:
            speaker_array = self.settings.get_speaker_array(speaker_array_id)
            cabinet_data = self.container.data.get('cabinet_data', [])
            speaker_names = self.container.data.get('speaker_names', [])

            if speaker_array is None:
                print("Warnung: speaker_array ist nicht initialisiert")
                return

            # Vorhandene Patches entfernen (von beiden Achsen)
            # WICHTIG: Entferne Patches VOR dem Zeichnen, um doppelte Darstellung zu vermeiden
            if self.ax.patches:
                for patch in list(self.ax.patches):  # Liste kopieren, da wir während Iteration entfernen
                    try:
                        patch.remove()
                    except:
                        pass
                self.ax.patches.clear()
            if self.ax_speakers is not None:
                if self.ax_speakers.patches:
                    for patch in list(self.ax_speakers.patches):  # Liste kopieren, da wir während Iteration entfernen
                        try:
                            patch.remove()
                        except:
                            pass
                    self.ax_speakers.patches.clear()

            # Sammle Cabinet-Daten für dieses Array
            array_cabinets = []
            name_to_index = {name: idx for idx, name in enumerate(speaker_names)}
            for pattern_name in speaker_array.source_polar_pattern:
                if not pattern_name:
                    continue
                try:
                    speaker_index = name_to_index.get(pattern_name)
                    if speaker_index is None or speaker_index >= len(cabinet_data):
                        raise ValueError(f"{pattern_name} nicht gefunden")
                    cabinet = cabinet_data[speaker_index]
                    normalized_cabinets = self._normalize_cabinet_entries(cabinet)
                    if normalized_cabinets:
                        array_cabinets.append(normalized_cabinets)
                except (ValueError, IndexError) as e:
                    print(f"Fehler beim Verarbeiten von Pattern {pattern_name}: {str(e)}")
                    continue

            if len(array_cabinets) > 0:
                # Verwende ax_speakers für Lautsprecher-Darstellung (mit aspect='equal')
                # Falls ax_speakers noch nicht erstellt wurde, verwende Hauptachse als Fallback
                speaker_ax = self.ax_speakers if self.ax_speakers is not None else self.ax
                # #region agent log - Welche Achse wird verwendet
                import json
                import time
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'E',
                        'location': 'PlotBeamsteering.py:302',
                        'message': 'Welche Achse wird verwendet',
                        'data': {
                            'ax_speakers_is_none': self.ax_speakers is None,
                            'speaker_ax_id': str(id(speaker_ax)),
                            'ax_id': str(id(self.ax)),
                            'ax_speakers_id': str(id(self.ax_speakers)) if self.ax_speakers is not None else None,
                            'speaker_ax_ylim': list(speaker_ax.get_ylim()) if hasattr(speaker_ax, 'get_ylim') else None,
                            'ax_ylim': list(self.ax.get_ylim()) if hasattr(self.ax, 'get_ylim') else None
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                stack_drawer = StackDraw_Beamsteering(
                    speaker_array.source_polar_pattern,
                    speaker_array.source_position_x,
                    speaker_array.source_position_y,
                    speaker_array.source_azimuth,
                    self.settings.width,
                    self.settings.length,
                    speaker_ax,  # Verwende zweite Y-Achse für Lautsprecher
                    cabinet_data=array_cabinets
                )
                
                # WICHTIG: Vor dem Zeichnen Y-Offsets auslesen
                # Berechne für jeden Stack den Y-Offset (Differenz zwischen base_y und 0)
                y_offsets = []
                # #region agent log - Y-Offsets vor dem Zeichnen
                import json
                import time
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'B',
                        'location': 'PlotBeamsteering.py:283',
                        'message': 'Y-Offsets vor dem Zeichnen',
                        'data': {
                            'num_stacks': len(array_cabinets),
                            'source_position_y': [float(y) for y in speaker_array.source_position_y[:len(array_cabinets)]] if hasattr(speaker_array, 'source_position_y') else []
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                for isrc in range(len(array_cabinets)):
                    if isrc < len(speaker_array.source_position_y):
                        base_y = float(speaker_array.source_position_y[isrc])
                        y_offset = base_y - 0.0  # Offset von base_y zu Y=0
                        y_offsets.append(y_offset)
                    else:
                        y_offsets.append(0.0)
                
                # Zeichne jeden Stack (wird bei Y=0 gezeichnet)
                for isrc in range(len(array_cabinets)):
                    stack_drawer.draw_stack(isrc)
                
                # WICHTIG: Nach dem Zeichnen Y-Offsets auf Y-Position anwenden
                # Verschiebe alle Patches, damit sie bei Y=0 bleiben
                if self.ax_speakers is not None:
                    import matplotlib.patches as patches
                    # #region agent log - Y-Limits vor Korrektur
                    import json
                    import time
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'PlotBeamsteering.py:298',
                            'message': 'Y-Limits vor Korrektur',
                            'data': {
                                'ax_speakers_ylim': list(self.ax_speakers.get_ylim()),
                                'ax_speakers_xlim': list(self.ax_speakers.get_xlim()),
                                'num_patches': len(self.ax_speakers.patches)
                            },
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                    # #endregion
                    patch_index = 0
                    patch_positions_before = []
                    patch_positions_after = []
                    for isrc in range(len(array_cabinets)):
                        # Zähle die Patches für diesen Stack (kann mehrere sein, wenn mehrere Cabinets)
                        cabinet = array_cabinets[isrc]
                        if isinstance(cabinet, list):
                            num_cabinets = len(cabinet)
                        else:
                            num_cabinets = 1
                        
                        # Verschiebe jedes Patch dieses Stacks
                        for _ in range(num_cabinets):
                            if patch_index < len(self.ax_speakers.patches):
                                patch = self.ax_speakers.patches[patch_index]
                                if isinstance(patch, patches.Rectangle):
                                    # Stelle sicher, dass das Patch bei Y=0 ist
                                    # Der Offset wurde bereits beim Zeichnen berücksichtigt (y=0.0)
                                    # Aber falls sich die Position geändert hat, korrigieren wir sie
                                    current_y = patch.get_y()
                                    patch_positions_before.append({
                                        'isrc': isrc,
                                        'patch_index': patch_index,
                                        'y_before': float(current_y),
                                        'x': float(patch.get_x()),
                                        'width': float(patch.get_width()),
                                        'height': float(patch.get_height())
                                    })
                                    if abs(current_y - 0.0) > 0.001:  # Toleranz für Rundungsfehler
                                        patch.set_y(0.0)
                                    patch_positions_after.append({
                                        'isrc': isrc,
                                        'patch_index': patch_index,
                                        'y_after': float(patch.get_y())
                                    })
                                patch_index += 1
                    # #region agent log - Patch-Positionen nach Korrektur
                    with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'PlotBeamsteering.py:330',
                            'message': 'Patch-Positionen nach Korrektur',
                            'data': {
                                'positions_before': patch_positions_before,
                                'positions_after': patch_positions_after,
                                'ax_speakers_ylim_after': list(self.ax_speakers.get_ylim())
                            },
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                    # #endregion
            else:
                print("Keine gültigen Cabinet-Daten gefunden")

            # Kein draw_idle() hier - wird am Ende von beamsteering_plot() aufgerufen

        except Exception as e:
            print(f"Fehler in plot_Stacks2Beamsteering: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _calculate_metric_ticks(self, min_val, max_val):
        """
        Berechnet intelligente Ticks für metrische Achsen basierend auf dem Bereich.
        Ziel: 4-8 Ticks für optimale Lesbarkeit mit passender Schrittweite.
        
        Args:
            min_val: Minimum-Wert der Achse
            max_val: Maximum-Wert der Achse
            
        Returns:
            tuple: (ticks_list, step) - Liste der Tick-Positionen und verwendete Schrittweite
        """
        range_val = max_val - min_val
        
        if range_val <= 0:
            # Fallback für ungültige Bereiche
            return [min_val, max_val], 1.0
        
        # Ziel: 4-8 Ticks für optimale Lesbarkeit
        # Berechne optimale Schrittweite basierend auf dem Bereich
        ideal_num_ticks = 6  # Ziel: ~6 Ticks
        rough_step = range_val / ideal_num_ticks
        
        # Wähle passende Schrittweite aus einer Liste von "schönen" Werten
        # Sortiert von klein nach groß
        nice_steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        
        # Finde die passende Schrittweite (nächstgrößere oder gleichgroße)
        step = nice_steps[0]
        for nice_step in nice_steps:
            if nice_step >= rough_step:
                step = nice_step
                break
        else:
            # Falls alle zu klein sind, verwende die größte
            step = nice_steps[-1]
        
        # Für sehr kleine Bereiche: verwende noch feinere Auflösung
        if range_val < 1.0:
            fine_steps = [0.05, 0.1, 0.2, 0.5]
            for fine_step in fine_steps:
                if fine_step >= rough_step:
                    step = fine_step
                    break
        
        # Generiere Ticks
        # Runde min_val nach unten auf das nächste Vielfache von step
        import math
        start_tick = math.floor(min_val / step) * step
        # Stelle sicher, dass start_tick <= min_val
        if start_tick > min_val:
            start_tick -= step
        
        # Runde max_val nach oben auf das nächste Vielfache von step
        end_tick = math.ceil(max_val / step) * step
        # Stelle sicher, dass end_tick >= max_val
        if end_tick < max_val:
            end_tick += step
        
        # Erstelle Tick-Liste
        ticks = []
        current = start_tick
        tolerance = step * 0.001  # Kleine Toleranz für Rundungsfehler
        while current <= end_tick + tolerance:
            if min_val - tolerance <= current <= max_val + tolerance:
                ticks.append(current)
            current += step
        
        # Stelle sicher, dass min und max enthalten sind (falls sie nicht schon als Ticks vorhanden sind)
        if len(ticks) == 0:
            ticks = [min_val, max_val]
        else:
            if abs(ticks[0] - min_val) > tolerance:
                ticks.insert(0, min_val)
            if abs(ticks[-1] - max_val) > tolerance:
                ticks.append(max_val)
        
        # Entferne Duplikate und sortiere
        ticks = sorted(list(set(ticks)))
        
        return ticks, step
    
    def _connect_zoom_callbacks(self):
        """Verbindet Callbacks für Zoom-Events, um Achsenbeschriftung dynamisch anzupassen"""
        if self._zoom_callback_connected:
            return
        
        try:
            # Callback für X-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            # Callback für Y-Achsen-Zoom (metrische Achse)
            self.ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
            self._zoom_callback_connected = True
        except Exception as e:
            print(f"[BeamsteeringPlot] Fehler beim Verbinden von Zoom-Callbacks: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_xlim_changed(self, ax):
        """Wird aufgerufen, wenn sich die X-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        # Verhindere Endlosschleife
        if self._updating_ticks:
            return
        
        try:
            self._updating_ticks = True
            x_min, x_max = ax.get_xlim()
            
            # Synchronisiere X-Limits der zweiten Y-Achse und passe Y-Limits für aspect='equal' an
            if self.ax_speakers is not None:
                self.ax_speakers.set_xlim(x_min, x_max)
                # Berechne Y-Limits basierend auf X-Limits für aspect='equal'
                # Y=0 soll immer zentriert bleiben
                x_range = x_max - x_min
                y_range = x_range  # Gleiche Range für 1:1 Aspect-Ratio
                y_center = 0.0  # Zentriert bei 0
                y_min_speakers = y_center - y_range / 2
                y_max_speakers = y_center + y_range / 2
                self.ax_speakers.set_ylim(y_min_speakers, y_max_speakers)
                # Setze aspect='equal' erneut mit adjustable='datalim' (für twinned Axes erforderlich)
                self.ax_speakers.set_aspect('equal', adjustable='datalim')
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            # Die Schrittweite wird automatisch feiner bei kleinerem Bereich (Zoom-in)
            # und gröber bei größerem Bereich (Zoom-out)
            x_ticks, step = self._calculate_metric_ticks(x_min, x_max)
            
            ax.set_xticks(x_ticks)
            
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
        finally:
            self._updating_ticks = False
    
    def _on_ylim_changed(self, ax):
        """Wird aufgerufen, wenn sich die Y-Achsen-Grenzen ändern (Zoom/Pan)"""
        if ax != self.ax:
            return
        
        # Verhindere Endlosschleife
        if self._updating_ticks:
            return
        
        try:
            self._updating_ticks = True
            y_min, y_max = ax.get_ylim()
            
            # Berechne neue Ticks basierend auf sichtbarem Bereich
            y_ticks, step = self._calculate_metric_ticks(y_min, y_max)
            
            ax.set_yticks(y_ticks)
            # Formatierung auf eine Nachkommastelle
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
            
            # Trigger Redraw
            if hasattr(ax.figure, 'canvas'):
                ax.figure.canvas.draw_idle()
        except Exception as e:
            pass
            import traceback
            traceback.print_exc()
        finally:
            self._updating_ticks = False
    
    def _apply_layout(self):
        """Wendet Layout-Einstellungen konsistent an"""
        if hasattr(self.ax, 'figure'):
            try:
                # Deaktiviere constrained_layout vollständig
                self.ax.figure.set_constrained_layout(False)
                # Passe die Plot-Ränder an Canvas-Größe an (Beamsteering-Plot)
                # Mehr Platz unten/links, damit Achsenbeschriftungen vollständig sichtbar sind
                self.ax.figure.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.22)
            except Exception:
                pass
            # Zeichne nur wenn Canvas vorhanden
            if hasattr(self.ax.figure, 'canvas'):
                self.ax.figure.canvas.draw_idle()

    @staticmethod
    def _to_float_array(values):
        if values is None:
            return np.array([], dtype=float)
        array = np.atleast_1d(values)
        result = []
        for item in array:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return np.array(result, dtype=float) if result else np.array([], dtype=float)

    @staticmethod
    def _concatenate_arrays(*arrays):
        valid = [arr for arr in arrays if arr is not None and arr.size]
        if not valid:
            return np.array([], dtype=float)
        return np.concatenate(valid)

    def _adjust_ylim_for_speakers(self):
        """Passt Y-Achsen-Grenzen an, damit Lautsprecher nicht abgeschnitten werden"""
        try:
            import matplotlib.patches as patches
            # Prüfe Patches auf der zweiten Y-Achse (Lautsprecher)
            speaker_ax = self.ax_speakers if self.ax_speakers is not None else self.ax
            
            # Finde maximale Höhe aller gezeichneten Rechtecke (Lautsprecher)
            # Da Lautsprecher bei Y=0 positioniert sind, suchen wir die maximale Höhe
            max_height = 0.0
            for patch in speaker_ax.patches:
                if isinstance(patch, patches.Rectangle):
                    patch_height = patch.get_height()
                    max_height = max(max_height, patch_height)
            
            # Füge Padding hinzu (nur oberhalb von Y=0, da Lautsprecher bei Y=0 starten)
            padding = 0.1
            max_y_with_padding = max_height + padding  # Lautsprecher-Höhe + Padding (Lautsprecher starten bei Y=0)
            min_y_with_padding = -padding  # Minimales Padding unterhalb von Y=0 (nur für Sicherheit)
            
            # Für die zweite Y-Achse mit aspect='equal': Berechne Y-Limits basierend auf X-Limits
            if self.ax_speakers is not None:
                xlim = self.ax.get_xlim()
                x_range = xlim[1] - xlim[0]
                
                # Stelle sicher, dass die Y-Range groß genug ist für alle Lautsprecher
                required_y_range = max_height + 2 * padding
                # Verwende das Maximum aus X-Range (für 1:1 Aspect) und erforderlicher Höhe
                y_range = max(x_range, required_y_range)
                
                # Zentriere bei Y=0
                y_center = 0.0
                y_min = y_center - y_range / 2
                y_max = y_center + y_range / 2
                
                # Setze Y-Limits der zweiten Achse
                self.ax_speakers.set_ylim(y_min, y_max)
                # Synchronisiere X-Limits
                self.ax_speakers.set_xlim(xlim)
                # Setze aspect='equal' erneut mit adjustable='datalim' (für twinned Axes erforderlich)
                self.ax_speakers.set_aspect('equal', adjustable='datalim')
            
            # Hauptachse: Passe Y-Limits an, damit Lautsprecher nicht abgeschnitten werden
            # OPTIMIERT: Erweitere nur nach oben, wenn nötig; behalte untere Grenze bei
            current_ylim = self.ax.get_ylim()
            # #region agent log - Vor Anpassung Hauptachse
            import json
            import time
            with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'B',
                    'location': 'PlotBeamsteering.py:968',
                    'message': '_adjust_ylim_for_speakers: Vor Anpassung Hauptachse',
                    'data': {
                        'max_height': float(max_height),
                        'padding': float(padding),
                        'max_y_with_padding': float(max_y_with_padding),
                        'min_y_with_padding': float(min_y_with_padding),
                        'current_ylim': list(current_ylim),
                        'needs_adjustment_top': bool(max_y_with_padding > current_ylim[1]),
                        'needs_adjustment_bottom': bool(min_y_with_padding < current_ylim[0])
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
            # #endregion
            # Erweitere nur nach oben, wenn Lautsprecher abgeschnitten werden
            # Unten: Nur erweitern, wenn absolut nötig (z.B. wenn Daten negativer sind als erwartet)
            needs_adjustment = False
            y_min = current_ylim[0]  # Behalte untere Grenze bei
            y_max = current_ylim[1]   # Starte mit aktueller oberer Grenze
            
            if max_y_with_padding > current_ylim[1]:
                # Erweitere nach oben für Lautsprecher
                y_max = max_y_with_padding
                needs_adjustment = True
            
            # Unten nur erweitern, wenn absolut nötig (z.B. wenn Y=0 nicht sichtbar ist)
            if current_ylim[0] > 0.0:
                # Y=0 ist nicht sichtbar - erweitere minimal nach unten
                y_min = min(current_ylim[0], -padding)
                needs_adjustment = True
            
            if needs_adjustment:
                # Berechne neue Ticks für erweiterte Grenze
                # #region agent log - Vor _calculate_metric_ticks in _adjust
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'B',
                        'location': 'PlotBeamsteering.py:973',
                        'message': '_adjust_ylim_for_speakers: Vor _calculate_metric_ticks',
                        'data': {
                            'y_min_before_ticks': float(y_min),
                            'y_max_before_ticks': float(y_max),
                            'y_range_before_ticks': float(y_max - y_min)
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                y_ticks, step = self._calculate_metric_ticks(y_min, y_max)
                # #region agent log - Nach _calculate_metric_ticks in _adjust
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'B',
                        'location': 'PlotBeamsteering.py:976',
                        'message': '_adjust_ylim_for_speakers: Nach _calculate_metric_ticks',
                        'data': {
                            'y_ticks': [float(t) for t in y_ticks],
                            'step': float(step),
                            'y_min': float(y_min),
                            'y_max': float(y_max)
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
                self.ax.set_ylim(y_min, y_max)
                self.ax.set_yticks(y_ticks)
                # Formatierung auf eine Nachkommastelle
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
                # #region agent log - Nach set_ylim in _adjust
                with open('/Users/MGraf/Python/LFO_Umgebung/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'B',
                        'location': 'PlotBeamsteering.py:980',
                        'message': '_adjust_ylim_for_speakers: Nach set_ylim',
                        'data': {
                            'ax_ylim_after_set': list(self.ax.get_ylim()),
                            'y_min': float(y_min),
                            'y_max': float(y_max)
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
                # #endregion
        except Exception:
            # Bei Fehler nichts ändern
            pass

    @classmethod
    def _normalize_cabinet_entries(cls, cabinet):
        if cabinet is None:
            return []
        if isinstance(cabinet, dict):
            return [cabinet]
        if isinstance(cabinet, np.ndarray):
            cabinet = cabinet.tolist()
        if isinstance(cabinet, (list, tuple)):
            normalized = []
            for item in cabinet:
                normalized.extend(cls._normalize_cabinet_entries(item))
            return normalized
        return []