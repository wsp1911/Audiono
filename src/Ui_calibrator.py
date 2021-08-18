# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QLabel,
    QWidget,
)
from PyQt5.QtGui import QFont
from pyqtgraph import PlotWidget

from public import BasicWindow, WidgetWithSplitter


class Ui_calibrator(BasicWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.Centre(1200, 600)
        self.setWindowTitle("校准")
        font_size = 9
        self.setFont(QFont("等线", font_size))
        self.setCentralWidget(QWidget())

        self.pw = PlotWidget()
        self.curve = [self.pw.plot(pen="y"), self.pw.plot(pen="g")]

        self.MainWidget = WidgetWithSplitter(
            self.centralWidget(), sizes=[self.width(), 0]
        )
        left_grid = self.MainWidget.getGrid(0)
        left_grid.addGrid(0, 0)
        left_grid.addWidget(self.pw)

        self.SaveBtn = QPushButton()
        self.SaveBtn.setText("保存")

        self.Calibrate = [QPushButton("校准1"), QPushButton("校准2"), QPushButton("校准3")]

        right_grid = self.MainWidget.getGrid(1)
        right_grid.addGrid(0, 0)
        right_grid.nextCol()
        right_grid.addWidget(self.SaveBtn, True)
        right_grid.addWidget(QLabel("1 输入偏移\nI0，I1接GND"))
        right_grid.addWidget(self.Calibrate[0], True)
        right_grid.addWidget(QLabel("2 输入增益\nI0，I1接+5V"))
        right_grid.addWidget(self.Calibrate[1], True)
        right_grid.addWidget(QLabel("3 输出偏移与增益\nI0接O0，I1接O1"))
        right_grid.addWidget(self.Calibrate[2], True)

        self.adjust_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_size()

    def adjust_size(self):
        self.MainWidget.setGeometry(0, 0, self.width(), self.height())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_calibrator()
    ui.show()
    sys.exit(app.exec_())
