"""Hilfsobjekte zur Anzeige von Fortschrittsdialogen im LFO-Tool."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy import QtCore, QtWidgets


@dataclass
class ProgressConfig:
    title: str
    total_steps: int
    minimum_duration_ms: int = 500


class ProgressSession:
    def __init__(self, parent: QtWidgets.QWidget, config: ProgressConfig):
        self._config = config
        self._total_steps = max(1, config.total_steps)
        self._completed = 0

        self._dialog = QtWidgets.QProgressDialog(parent)
        self._dialog.setWindowTitle("")
        self._dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self._dialog.setCancelButton(None)
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
        self._dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
        self._dialog.setLabelText("")
        self._dialog.setFixedSize(260, 40)
        bar = self._dialog.findChild(QtWidgets.QProgressBar)
        if bar:
            bar.setTextVisible(False)
            bar.setFixedHeight(14)

    def __enter__(self) -> "ProgressSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()

    def update(self, _: str) -> None:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    def advance(self) -> None:
        if self._completed < self._total_steps:
            self._completed += 1
        self._dialog.setValue(self._completed)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    def finish(self) -> None:
        self._dialog.setValue(self._total_steps)
        self._dialog.close()
        self._dialog.deleteLater()


class ProgressManager:
    """Erzeugt Fortschrittsdialoge für sequentielle Aufgaben."""

    def __init__(self, parent: QtWidgets.QWidget):
        self._parent = parent

    def start(self, title: str, total_steps: int, minimum_duration_ms: int = 500) -> ProgressSession:
        config = ProgressConfig(title=title, total_steps=total_steps, minimum_duration_ms=minimum_duration_ms)
        return ProgressSession(self._parent, config)


