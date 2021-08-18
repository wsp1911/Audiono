# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QWidget,
    QGroupBox,
    QGridLayout,
)
from PyQt5.QtGui import QFont

from public import BasicWindow


class Ui_mainWindow(BasicWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.Centre(300, 400)
        self.setWindowTitle("Audiono")
        font_size = 9
        self.setFont(QFont("等线", font_size))
        self.setCentralWidget(QWidget())

        self.buttons = [
            QPushButton("校准"),
            QPushButton("示波器与频谱仪"),
            QPushButton("信号发生器"),
            QPushButton("瀑布图仪"),
            QPushButton("波特图仪"),
        ]

        for btn in self.buttons:
            btn.setMinimumHeight(70)

        self.qgb = QGroupBox(self.centralWidget())
        grid = QGridLayout(self.qgb)

        grid.addWidget(self.buttons[0], 0, 0)
        grid.addWidget(self.buttons[1], 1, 0)
        grid.addWidget(self.buttons[2], 2, 0)
        grid.addWidget(self.buttons[3], 3, 0)
        grid.addWidget(self.buttons[4], 4, 0)

        self.adjust_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_size()

    def adjust_size(self):
        self.qgb.setGeometry(0, 0, self.width(), self.height())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_mainWindow()
    ui.show()
    sys.exit(app.exec_())
