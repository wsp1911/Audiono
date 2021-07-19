# -*- coding: utf-8 -*-
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QFrame,
)
import pyqtgraph as pg

from public import ComboGridLayout, BasicWindow, SwitchButton


class sg_plot_widget(pg.PlotWidget):
    def __init__(self, parent=None, pen="w"):
        super(sg_plot_widget, self).__init__(parent)
        self.curve = self.plot(pen=pen)
        self.setYRange(-10, 10)
        self.showAxis("right")


class Ui_signalGenerator(BasicWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent, hasStatusBar=True, hasThread=True)

        WIDTH, HEIGHT = 1000, 800

        # self.setFixedSize(WIDTH, HEIGHT)
        self.resize(WIDTH, HEIGHT)
        self.setFont(QtGui.QFont("等线", 9))
        self.setWindowTitle("信号发生器")

        self.setCentralWidget(QWidget())

        self.pw = [sg_plot_widget(pen="y"), sg_plot_widget(pen="g")]

        self.Grid = ComboGridLayout(parent=self.centralWidget())

        self.Grid.addGrid(0, 0, 1, 2)
        self.Grid.addWidget(self.pw[0])
        self.Grid.addWidget(self.pw[1])

        self.synchSelect = QComboBox()
        self.synchSelect.addItems(("异步", "同步"))
        self.synchSelect.setCurrentIndex(0)

        self.RunButton = SwitchButton()

        self.waveSelect = [QComboBox(), QComboBox()]
        self.waveSelect[0].addItems(
            ("关闭", "正弦波", "方波", "直流", "三角波", "锯齿波", "表达式", "文件")
        )
        self.waveSelect[1].addItems(
            ("关闭", "正弦波", "方波", "直流", "三角波", "锯齿波", "表达式", "文件")
        )

        self.freqInput = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.freqInput[0].setMinimum(0.01)
        self.freqInput[0].setMaximum(20000.0)
        self.freqInput[0].setValue(1000)
        self.freqInput[1].setMinimum(0.01)
        self.freqInput[1].setMaximum(20000.0)
        self.freqInput[1].setValue(1000)

        self.vppInput = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.vppInput[0].setSingleStep(0.1)
        self.vppInput[0].setRange(-20, 20)
        self.vppInput[0].setValue(5)
        self.vppInput[1].setSingleStep(0.1)
        self.vppInput[1].setValue(5)
        self.vppInput[1].setRange(-20, 20)

        self.offsetInput = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.offsetInput[0].setSingleStep(0.1)
        self.offsetInput[0].setValue(0)
        self.offsetInput[0].setRange(-10, 10)
        self.offsetInput[1].setSingleStep(0.1)
        self.offsetInput[1].setValue(0)
        self.offsetInput[1].setRange(-20, 20)

        self.dutyInput = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.dutyInput[0].setMaximum(100.0)
        self.dutyInput[0].setSingleStep(10.0)
        self.dutyInput[0].setValue(50)
        self.dutyInput[1].setMaximum(100.0)
        self.dutyInput[1].setSingleStep(10.0)
        self.dutyInput[1].setValue(50)

        self.phiInput = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.phiInput[0].setMaximum(360.0)
        self.phiInput[0].setSingleStep(10)
        self.phiInput[0].setValue(0)
        self.phiInput[1].setMaximum(360.0)
        self.phiInput[1].setSingleStep(10)
        self.phiInput[1].setValue(0)

        self.Grid.addGrid(1, 1, 2, 1)
        self.Grid.setTitle("Normal")
        self.Grid.addWidget(QLabel("输出方式"))
        self.Grid.addWidget(self.synchSelect)
        self.Grid.addWidget(self.RunButton, True)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.Grid.addWidget(line, True, colSpan=3)
        self.Grid.nextCol()
        self.Grid.addWidget(QLabel("通道1"))
        self.Grid.addWidget(QLabel("通道2"), True)
        self.Grid.addWidget(QLabel("波形"))
        self.Grid.addWidget(self.waveSelect[0])
        self.Grid.addWidget(self.waveSelect[1], True)
        self.Grid.addWidget(QLabel("频    率/Hz"))
        self.Grid.addWidget(self.freqInput[0])
        self.Grid.addWidget(self.freqInput[1], True)
        self.Grid.addWidget(QLabel("峰峰值/V"))
        self.Grid.addWidget(self.vppInput[0])
        self.Grid.addWidget(self.vppInput[1], True)
        self.Grid.addWidget(QLabel("偏    置/V"))
        self.Grid.addWidget(self.offsetInput[0])
        self.Grid.addWidget(self.offsetInput[1], True)
        self.Grid.addWidget(QLabel("占空比/%"))
        self.Grid.addWidget(self.dutyInput[0])
        self.Grid.addWidget(self.dutyInput[1], True)
        self.Grid.addWidget(QLabel("相    位/°"))
        self.Grid.addWidget(self.phiInput[0])
        self.Grid.addWidget(self.phiInput[1], True)
        self.Grid.setSpacing(20, orientation=1)  # vertical spacing
        self.Grid.setSpacing(50)  # horizontal spacing

        self.expInput = [QLineEdit(), QLineEdit()]
        self.expInput[0].setText("0,0.1;sin(2*pi*1000*t)")
        self.expInput[1].setText("0,0.1;sin(2*pi*1000*t)")
        self.playExp = [QPushButton("输出"), QPushButton("输出")]

        self.Grid.addGrid(1, 0)
        self.Grid.setTitle("Expression")
        self.Grid.addWidget(QLabel("表达式1"))
        self.Grid.addWidget(self.expInput[0])
        self.Grid.addWidget(self.playExp[0], "True")
        self.Grid.addWidget(QLabel("表达式2"))
        self.Grid.addWidget(self.expInput[1])
        self.Grid.addWidget(self.playExp[1], "True")

        self.fileInfo = [QLineEdit(), QLineEdit()]
        self.openFile = [QPushButton("载入"), QPushButton("载入")]
        self.playFile = [QPushButton("播放"), QPushButton("播放")]
        self.interCB = QCheckBox("插值")
        self.clearFile = QPushButton("清除")

        self.Grid.addGrid(2, 0)
        self.Grid.setTitle("File")
        self.Grid.nextCol(2)
        self.Grid.addWidget(self.interCB)
        self.Grid.addWidget(self.clearFile, True)
        self.Grid.addWidget(QLabel("通道1"))
        self.Grid.addWidget(self.fileInfo[0])
        self.Grid.addWidget(self.openFile[0])
        self.Grid.addWidget(self.playFile[0], True)
        self.Grid.addWidget(QLabel("通道2"))
        self.Grid.addWidget(self.fileInfo[1])
        self.Grid.addWidget(self.openFile[1])
        self.Grid.addWidget(self.playFile[1], True)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ui = Ui_signalGenerator()
    ui.show()
    sys.exit(app.exec_())
