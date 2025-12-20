"""Hilfsobjekte zur Anzeige und Steuerung von Fortschrittsdialogen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qtpy import QtCore, QtWidgets


class ProgressCancelled(RuntimeError):
    """Signalisiert, dass der Benutzer den Fortschrittsdialog abgebrochen hat."""


@dataclass
class ProgressConfig:
    title: str
    total_steps: int
    minimum_duration_ms: int = 500


class ProgressSession:
    """Kapselt einen einzelnen Fortschrittsdialog."""

    def __init__(self, parent: QtWidgets.QWidget, config: ProgressConfig):
        self._config = config
        self._total_steps = max(1, config.total_steps)
        self._completed = 0
        self._cancelled = False

        self._dialog = QtWidgets.QProgressDialog(parent)
        self._dialog.setWindowTitle("")
        self._dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self._dialog.setRange(0, self._total_steps)
        self._dialog.setValue(0)
        self._dialog.setMinimumDuration(config.minimum_duration_ms)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._dialog.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self._dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._dialog.setLabelText("")
        self._dialog.setFixedSize(420, 150)

        cancel_button = QtWidgets.QPushButton("Cancel", self._dialog)
        cancel_button.setMinimumWidth(75)
        cancel_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._dialog.setCancelButton(cancel_button)
        self._dialog.canceled.connect(self._on_cancelled)

        # Custom progress bar with modern styling
        bar = QtWidgets.QProgressBar(self._dialog)
        bar.setObjectName("lfoProgressBar")
        bar.setTextVisible(False)
        bar.setFixedHeight(18)
        self._dialog.setBar(bar)

        # Replace default layout with card-style container
        base_layout = QtWidgets.QVBoxLayout(self._dialog)
        base_layout.setContentsMargins(10, 10, 10, 10)
        base_layout.setSpacing(0)

        card = QtWidgets.QFrame(self._dialog)
        card.setObjectName("progressCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(14)

        base_layout.addWidget(card)

        self._title_label = QtWidgets.QLabel(config.title or "Calculation in progress", card)
        self._title_label.setObjectName("progressTitle")
        title_font = self._title_label.font()
        title_font.setPointSizeF(title_font.pointSizeF() + 1.0)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setWordWrap(True)
        card_layout.addWidget(self._title_label)

        card_layout.addWidget(bar)

        self._status_label = QtWidgets.QLabel(" ", card)
        self._status_label.setObjectName("progressStatus")
        self._status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._status_label.setMinimumHeight(18)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setContentsMargins(0, 6, 0, 0)
        bottom_row.addWidget(self._status_label, stretch=1)
        bottom_row.addSpacing(12)
        bottom_row.addWidget(cancel_button, 0, QtCore.Qt.AlignRight)
        card_layout.addLayout(bottom_row)

        # Style sheet for Material-like appearance
        self._dialog.setStyleSheet(
            """
            QDialog {
                background-color: transparent;
            }
            #progressCard {
                background-color: #2F2F2F;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            QLabel#progressTitle {
                color: #F5F5F5;
                letter-spacing: 0.4px;
            }
            QLabel#progressStatus {
                color: #D9D9D9;
                font-size: 10.5pt;
            }
            #lfoProgressBar {
                background-color: #3A3A3A;
                border-radius: 10px;
                padding: 2px;
            }
            #lfoProgressBar::chunk {
                border-radius: 8px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00C6FF,
                    stop:1 #0072FF
                );
            }
            QPushButton {
                background-color: #4C6EF5;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #3F5BD5;
            }
            QPushButton:pressed {
                background-color: #364DB3;
            }
            """
        )

    def __enter__(self) -> "ProgressSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # 🛡️ SICHERHEIT: Prüfe ob Dialog noch existiert, bevor finish aufgerufen wird
        if self._dialog is not None:
            self.finish()

    def _on_cancelled(self) -> None:
        self._cancelled = True

    def _process_events(self) -> None:
        # 🛡️ SICHERHEIT: Prüfe ob Dialog noch existiert und gültig ist, bevor processEvents aufgerufen wird
        # Vermeide processEvents während der Dialog gelöscht wird oder die App beendet wird
        if self._dialog is None:
            return
        
        # 🎯 DEAKTIVIERT: processEvents() kann Segmentation Faults verursachen, wenn Qt in ungültigem Zustand ist
        # Stattdessen verwenden wir einen Timer-basierten Ansatz oder verzichten komplett auf processEvents
        # Dies verhindert Crashes, kann aber zu weniger responsivem UI führen
        # 
        # ALTERNATIVE: processEvents() nur sehr selten aufrufen oder komplett deaktivieren
        # Für jetzt: komplett deaktiviert, um Segmentation Faults zu vermeiden
        return
        
        # CODE UNTEN WIRD NICHT AUSGEFÜHRT (nur als Backup für zukünftige Implementierung):
        try:
            # Prüfe ob Dialog noch gültig ist (nicht gelöscht)
            if not hasattr(self._dialog, 'isVisible'):
                return
            
            app = QtWidgets.QApplication.instance()
            if app is None:
                return
            
            # Prüfe ob App noch läuft (nicht während Shutdown)
            if not hasattr(app, 'processEvents'):
                return
            
            # Verwende ProcessEventsFlag.AllEvents, aber mit Timeout um Deadlocks zu vermeiden
            # Verwende ExcludeUserInputEvents um zu vermeiden, dass Benutzerinteraktionen während der Berechnung verarbeitet werden
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents, 0)
        except (RuntimeError, AttributeError, TypeError):
            # Dialog wurde bereits gelöscht, App ist nicht mehr verfügbar, oder anderer Fehler - ignoriere
            pass
        except Exception:
            # Alle anderen Fehler abfangen (z.B. Segmentation Fault wird als Exception abgefangen)
            pass

    def update(self, message: Optional[str] = None) -> None:
        # 🛡️ SICHERHEIT: Prüfe ob Dialog und Label noch existieren
        if self._dialog is None:
            return
        try:
            if message and self._status_label:
                self._status_label.setText(message)
            self._process_events()
        except RuntimeError:
            # Dialog wurde bereits gelöscht - ignoriere
            pass

    def advance(self, steps: int = 1) -> None:
        if steps < 1:
            steps = 1
        self._completed = min(self._total_steps, self._completed + steps)
        # 🛡️ SICHERHEIT: Prüfe ob Dialog noch existiert und gültig ist, bevor setValue aufgerufen wird
        if self._dialog is None:
            return
        try:
            # Prüfe ob Dialog noch gültig ist
            if not hasattr(self._dialog, 'setValue'):
                self._dialog = None
                return
            
            if self._dialog.isVisible():
                self._dialog.setValue(self._completed)
                # Nur processEvents aufrufen, wenn Dialog noch sichtbar ist
                self._process_events()
        except (RuntimeError, AttributeError, TypeError):
            # Dialog wurde bereits gelöscht - ignoriere
            self._dialog = None
        except Exception:
            # Alle anderen Fehler abfangen (z.B. Segmentation Fault wird als Exception abgefangen)
            self._dialog = None

    def finish(self) -> None:
        # 🛡️ SICHERHEIT: Prüfe ob Dialog noch existiert, bevor Operationen ausgeführt werden
        if self._dialog is None:
            return
        try:
            # Prüfe ob Dialog noch gültig ist
            if not hasattr(self._dialog, 'setValue'):
                self._dialog = None
                return
            
            self._dialog.setValue(self._total_steps)
            self._dialog.close()
            self._dialog.deleteLater()
            # Setze _dialog auf None, damit weitere Aufrufe erkannt werden
            self._dialog = None
        except (RuntimeError, AttributeError, TypeError):
            # Dialog wurde bereits gelöscht - ignoriere
            self._dialog = None
        except Exception:
            # Alle anderen Fehler abfangen (z.B. Segmentation Fault wird als Exception abgefangen)
            self._dialog = None

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise ProgressCancelled("Progress dialog cancelled by user.")



class ProgressManager:
    """Erzeugt Fortschrittsdialoge für sequentielle Aufgaben."""

    def __init__(self, parent: QtWidgets.QWidget):
        self._parent = parent

    def start(self, title: str, total_steps: int, minimum_duration_ms: int = 500) -> ProgressSession:
        config = ProgressConfig(title=title, total_steps=total_steps, minimum_duration_ms=minimum_duration_ms)
        return ProgressSession(self._parent, config)


