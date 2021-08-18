# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QCheckBox,
    QLineEdit,
    QFrame,
    QSplitter,
)
from PyQt5.QtGui import QFont
from pyqtgraph import PlotWidget, ArrowItem

from public import (
    logSpinBox,
    doubleSlider,
    WidgetWithSplitter,
    DataGrid,
    SwitchButton,
    BasicWindow,
)


class bp_plot_widget(PlotWidget):
    def __init__(
        self,
        parent=None,
        pen="y",
        symbol="o",
        symbolBrush="r",
        symbolPen="w",
        line_pen="FFA500",
    ):
        super(bp_plot_widget, self).__init__(parent)

        self.vline = [self.plot(pen=line_pen), self.plot(pen=line_pen)]
        self.hline = [self.plot(pen=line_pen), self.plot(pen=line_pen)]
        self.scatter = self.plot(
            pen=pen, symbol=symbol, symbolBrush=symbolBrush, symbolPen=symbolPen
        )
        self.arrow = ArrowItem()

    def set_cursors(self, X=None, Y=None):
        if X:
            x_lim = self.getAxis("bottom").range
            y_lim = self.getAxis("left").range
            self.vline[0].setData([X[0], X[0]], y_lim)
            self.vline[1].setData([X[1], X[1]], y_lim)
            self.hline[0].setData(x_lim, [Y[0], Y[0]])
            self.hline[1].setData(x_lim, [Y[1], Y[1]])
        else:
            self.vline[0].setData()
            self.vline[1].setData()
            self.hline[0].setData()
            self.hline[1].setData()

    def clear_scatter(self):
        self.scatter.setData()

    def show_arrow(self, state):
        if state and self.arrow not in self.items():
            self.addItem(self.arrow)
        elif self.arrow in self.items():
            self.removeItem(self.arrow)


class Ui_bodePlotter(BasicWindow):
    def __init__(self, parent=None, rate=48000):
        super().__init__(parent, hasStatusBar=True)
        self.Centre(0.9, 0.8)
        font_size = 9
        self.setFont(QFont("等线", font_size))
        self.setWindowTitle("波特图仪")
        self.setCentralWidget(QWidget())

        self.MainWidget = WidgetWithSplitter(
            self.centralWidget(), sizes=[self.width(), 0]
        )
        self.leftGrid = self.MainWidget.getGrid(0)
        self.rightGrid = self.MainWidget.getGrid(1)

        pw_splitter = QSplitter(orientation=Qt.Vertical)
        pw_splitter.setFrameShape(QFrame.Box)
        self.pw = [
            bp_plot_widget(parent=pw_splitter),
            bp_plot_widget(parent=pw_splitter),
        ]
        for p in self.pw:
            p.showAxis("right")
            p.showGrid(x=True, y=True, alpha=1)

        self.leftGrid.addGrid(0, 0)
        self.leftGrid.addWidget(pw_splitter)

        self.CursorX = [doubleSlider(), doubleSlider()]
        self.CursorX[0].setOrientation(QtCore.Qt.Horizontal)
        self.CursorX[0].setFixedHeight(10)
        self.CursorX[1].setOrientation(QtCore.Qt.Horizontal)
        self.CursorX[1].setFixedHeight(10)
        self.CursorX[1].setValue(1)
        self.leftGrid.addGrid(1, 0)
        self.leftGrid.addWidget(self.CursorX[0])
        self.leftGrid.addWidget(self.CursorX[1], True)

        self.CursorY = [doubleSlider(), doubleSlider()]
        self.CursorY[0].setOrientation(QtCore.Qt.Vertical)
        self.CursorY[0].setFixedWidth(10)
        self.CursorY[1].setOrientation(QtCore.Qt.Vertical)
        self.CursorY[1].setFixedWidth(10)
        self.CursorY[1].setValue(1)
        self.leftGrid.addGrid(0, 1)
        self.leftGrid.addWidget(self.CursorY[1], True)
        self.leftGrid.addWidget(self.CursorY[0], True)

        self.data_labels = [
            ["f1/Hz", "f2/Hz", "A1", "A2", "df/Hz", "A2/A1"],
            ["f1/Hz", "f2/Hz", "phi1/°", "phi2/°", "df/Hz", "dphi/°"],
            ["f1/Hz", "f2/Hz", "A1/dB", "A2/dB", "df", "dA/dB"],
        ]
        self.data_digits = [3, 2, 2]
        self.DataDispGrid = [
            DataGrid(labels=self.data_labels[0], digits=self.data_digits[0]),
            DataGrid(labels=self.data_labels[1], digits=self.data_digits[1]),
        ]
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        font = QFont("Times New Roman", font_size + 1)
        self.leftGrid.addGrid(0, 2)
        self.leftGrid.setFont(font)
        self.leftGrid.setVisible(False)
        self.leftGrid.addWidget(self.DataDispGrid[0], True)
        self.leftGrid.addWidget(line, True)
        self.leftGrid.addWidget(self.DataDispGrid[1], True)

        self.ResetButton = QPushButton("Reset")
        self.StartButton = [
            SwitchButton(texts=["Start", "Stop"]),
            SwitchButton(texts=["重新测量", "Stop"]),
        ]
        self.fLeftLabel = QLabel("起始频率")
        self.fLeft = QDoubleSpinBox()
        self.fLeft.setRange(0, rate // 2)
        self.fLeft.setSingleStep(100)
        self.fLeft.setValue(1000)
        self.fRightLabel = QLabel("终止频率")
        self.fRight = QDoubleSpinBox()
        self.fRight.setRange(0, rate // 2)
        self.fRight.setSingleStep(100)
        self.fRight.setValue(2000)
        self.fNumLabel = QLabel("频点数量")
        self.fNum = QSpinBox()
        self.fNum.setRange(1, 100)
        self.fNum.setSingleStep(1)
        self.fNum.setValue(5)
        self.rightGrid.addGrid(0, 0, 1, 2)
        self.rightGrid.setTitle("Scan")
        self.rightGrid.addWidget(self.ResetButton)
        self.rightGrid.addWidget(self.StartButton[0], True)
        self.rightGrid.addWidget(self.fLeftLabel)
        self.rightGrid.addWidget(self.fLeft, True)
        self.rightGrid.addWidget(self.fRightLabel)
        self.rightGrid.addWidget(self.fRight, True)
        self.rightGrid.addWidget(self.fNumLabel)
        self.rightGrid.addWidget(self.fNum, True)

        self.ShowArrow = QCheckBox("显示标记")
        self.PointIdLabel = QLabel("id")
        self.PointId = QSpinBox()
        self.PointId.setRange(0, 0)
        self.PointId.setValue(0)
        self.PointValue = QLineEdit()
        self.PointValue.setReadOnly(True)
        self.DeleteButton = QPushButton("删除该点")
        self.rightGrid.addGrid(1, 0, 1, 2)
        self.rightGrid.setTitle("Edit")
        self.rightGrid.addWidget(self.DeleteButton, colSpan=2)
        self.rightGrid.addWidget(self.StartButton[1], True, colSpan=2)
        self.rightGrid.addWidget(self.ShowArrow, colSpan=2)
        self.rightGrid.addWidget(self.PointIdLabel)
        self.rightGrid.addWidget(self.PointId, True)
        self.rightGrid.addWidget(self.PointValue, True, colSpan=4)

        self.fLogCB = QCheckBox("f log")
        self.ALogCB = QCheckBox("A log")
        self.fZoomLabel = QLabel("zoom")
        self.fZoom = logSpinBox()
        self.fZoom.setParameters(mi=1, ma=128, val=1, step=2, decimal=0)
        self.fPos = doubleSlider()
        self.fPos.setOrientation(QtCore.Qt.Horizontal)
        self.fPos.setFixedHeight(10)
        self.rightGrid.addGrid(2, 0, 1, 2)
        self.rightGrid.setTitle("Axis")
        self.rightGrid.addWidget(self.fLogCB)
        self.rightGrid.addWidget(self.ALogCB, True)
        self.rightGrid.addWidget(self.fZoomLabel)
        self.rightGrid.addWidget(self.fZoom, True)
        self.rightGrid.addWidget(self.fPos, True, colSpan=2)

        self.DataDispCB = QCheckBox("values")
        self.CursorTargetLabel = QLabel("测量对象")
        self.CursorTarget = QComboBox()
        self.CursorTarget.addItems(("幅频", "相频"))
        self.CursorCB = [QCheckBox("光标1"), QCheckBox("光标2")]
        self.rightGrid.addGrid(3, 0)
        self.rightGrid.setTitle("Measure")
        self.rightGrid.addWidget(self.DataDispCB, True)
        self.rightGrid.addWidget(self.CursorTargetLabel)
        self.rightGrid.addWidget(self.CursorTarget, True)
        self.rightGrid.addWidget(self.CursorCB[0])
        self.rightGrid.addWidget(self.CursorCB[1], True)

        self.SaveSrc = QComboBox()
        self.SaveSrc.addItems(("幅频", "相频"))
        self.PicSave = QPushButton("保存图片")
        self.DataSave = QPushButton("保存数据")
        self.rightGrid.addGrid(3, 1)
        self.rightGrid.setTitle("Save")
        self.rightGrid.addWidget(QLabel("数据类型"))
        self.rightGrid.addWidget(self.SaveSrc, True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.DataSave, True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.PicSave, True)

        self.adjust_size()

        self.DataDispCB.stateChanged.connect(self.on_DataDispCB_stateChanged)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_size()

    def on_DataDispCB_stateChanged(self):
        self.MainWidget.getGrid(0).setVisible(self.DataDispCB.checkState(), 3)

    def adjust_size(self):
        WIDTH, HEIGHT = self.width(), self.height()
        statusH = self.statusBar().geometry().height()
        winY = 0

        if self.DataDispCB.checkState():
            self.MainWidget.setGeometry(0, winY, WIDTH, HEIGHT - statusH)
        else:
            self.MainWidget.setGeometry(0, winY, WIDTH, HEIGHT - statusH)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_bodePlotter()
    ui.show()
    sys.exit(app.exec_())
