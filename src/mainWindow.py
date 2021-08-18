# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication

from Ui_mainWindow import Ui_mainWindow
from public import Config
from calibrator import calibrator
from signalGenerator import signalGenerator
from timeSpectrum import timeSpectrum
from bodePlotter import bodePlotter
from scope import scope


class mainWindow(Ui_mainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.config = Config("Audiono.ini")

        self.connect()

    def connect(self):
        self.buttons[0].clicked.connect(self.open_calibrator)
        self.buttons[1].clicked.connect(self.open_scope)
        self.buttons[2].clicked.connect(self.open_sg)
        self.buttons[3].clicked.connect(self.open_ts)
        self.buttons[4].clicked.connect(self.open_bp)

    def open_calibrator(self):
        if not hasattr(self, "calibrator"):
            self.calibrator = calibrator(parent=self, config=self.config)
        self.calibrator.show()
        self.calibrator.raise_()

    def open_scope(self):
        if not hasattr(self, "scope"):
            self.scope = scope(parent=self, config=self.config)
        self.scope.show()
        self.scope.raise_()

    def open_sg(self):
        if not hasattr(self, "sg"):
            self.sg = signalGenerator(parent=self, config=self.config)
        self.sg.show()
        self.sg.raise_()

    def open_ts(self):
        if not hasattr(self, "ts"):
            self.ts = timeSpectrum(parent=self, config=self.config)
        self.ts.show()
        self.ts.raise_()

    def open_bp(self):
        if not hasattr(self, "bp"):
            self.bp = bodePlotter(parent=self, config=self.config)
        self.bp.show()
        self.bp.raise_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    sys.exit(app.exec_())
