# -*- coding: utf-8 -*-
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QSpinBox,
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QFrame,
    QSplitter,
)
from PyQt5.QtGui import QFont
from pyqtgraph import PlotWidget

from public import (
    logSpinBox,
    doubleSlider,
    DataGrid,
    SwitchButton,
    BasicWindow,
    WidgetWithSplitter,
)


class scope_plot_widget(PlotWidget):
    def __init__(
        self, parent=None, curve_num=0, pen=[], trigger=False, line_pen="FFA500"
    ):
        super(scope_plot_widget, self).__init__(parent)

        self.curve = [self.plot(pen=pen[i]) for i in range(curve_num)]
        self.vline = [self.plot(pen=line_pen), self.plot(pen=line_pen)]
        self.hline = [self.plot(pen=line_pen), self.plot(pen=line_pen)]
        if trigger:
            self.trigger = [self.plot(pen=line_pen), self.plot(pen=line_pen)]

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

    def set_trigger(self, Y=None):
        if Y:
            x_lim = self.getAxis("bottom").range
            self.trigger[0].setData(x_lim, [Y[0], Y[0]])
            self.trigger[1].setData(x_lim, [Y[1], Y[1]])
        else:
            self.trigger[0].setData()
            self.trigger[1].setData()

    def clear_curve(self):
        for i in range(len(self.curve)):
            self.curve[i].setData()
        if hasattr(self, "trigger"):
            self.trigger[0].setData()
            self.trigger[1].setData()


class Ui_scope(BasicWindow):
    def __init__(self, parent=None, rate=48000):
        super().__init__(parent, hasStatusBar=True, hasThread=True)

        self.Centre(0.9, 0.8)
        font_size = 9
        self.setFont(QFont("等线", font_size))
        self.setWindowTitle("示波器")
        self.setCentralWidget(QWidget())

        self.MainWidget = WidgetWithSplitter(
            self.centralWidget(), sizes=[self.width(), 0]
        )
        self.leftGrid = self.MainWidget.getGrid(0)
        self.rightGrid = self.MainWidget.getGrid(1)

        pw_splitter = QSplitter(orientation=Qt.Vertical)
        self.pw = [
            scope_plot_widget(
                parent=pw_splitter, curve_num=3, pen=["y", "g", "y"], trigger=True
            ),
            scope_plot_widget(parent=pw_splitter, curve_num=2, pen=["y", "g"]),
        ]
        for p in self.pw:
            p.showGrid(x=True, y=True, alpha=1)
            p.showAxis("right")

        self.leftGrid.addGrid(0, 0)
        self.leftGrid.addWidget(pw_splitter)
        # self.leftGrid.addWidget(self.pw[0], True)
        # self.leftGrid.addWidget(self.pw[1], True)

        self.CursorX = [doubleSlider(), doubleSlider()]
        self.CursorX[0].setOrientation(Qt.Horizontal)
        self.CursorX[0].setFixedHeight(10)
        self.CursorX[1].setOrientation(Qt.Horizontal)
        self.CursorX[1].setFixedHeight(10)
        self.CursorX[1].setValue(1)
        self.leftGrid.addGrid(1, 0)
        self.leftGrid.addWidget(self.CursorX[0])
        self.leftGrid.addWidget(self.CursorX[1], True)

        self.CursorY = [doubleSlider(), doubleSlider()]
        self.CursorY[0].setOrientation(Qt.Vertical)
        self.CursorY[0].setFixedWidth(10)
        self.CursorY[1].setOrientation(Qt.Vertical)
        self.CursorY[1].setFixedWidth(10)
        self.CursorY[1].setValue(1)
        self.leftGrid.addGrid(0, 1)
        self.leftGrid.addWidget(self.CursorY[1], True)
        self.leftGrid.addWidget(self.CursorY[0], True)

        self.data_labels = [
            ["t1/ms", "t2/ms", "Y1/V", "Y2/V", "dt/ms", "1/dt/Hz", "dY/V"],
            ["f1/Hz", "f2/Hz", "Y1/dB", "Y2/dB", "df/Hz", "dY/dB"],
        ]
        self.data_digits = [3, 2]
        self.DataDispGrid = [
            DataGrid(labels=self.data_labels[0], digits=self.data_digits[0]),
            DataGrid(labels=self.data_labels[1], digits=self.data_digits[1]),
        ]
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        font = QFont("Times New Roman", font_size + 1)
        # font = QFont("Fira Code", font_size + 1)
        self.leftGrid.addGrid(0, 2)
        self.leftGrid.setFont(font)
        self.leftGrid.setVisible(False)
        self.leftGrid.addWidget(self.DataDispGrid[0], True)
        self.leftGrid.addWidget(line, True)
        self.leftGrid.addWidget(self.DataDispGrid[1], True)

        self.RunButton = SwitchButton(texts=["Run", "Stop"])
        self.ChannelLabel = QLabel("通道")
        self.Channel = QComboBox()
        self.Channel.addItems(("1 & 2", "1", "2", "XY"))
        # 水平时基调整
        self.tZoomLabel = QLabel("zoom")
        self.tZoom = logSpinBox()
        self.tZoom.setParameters(mi=1, ma=128, val=16, step=2, decimal=0)
        self.rightGrid.addGrid(0, 0, 1, 2)
        self.rightGrid.setTitle("Horizontal")
        # self.rightGrid.addWidget(self.CollectNoise)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.RunButton, True)
        self.rightGrid.addWidget(self.ChannelLabel)
        self.rightGrid.addWidget(self.Channel, True)
        self.rightGrid.addWidget(self.tZoomLabel)
        self.rightGrid.addWidget(self.tZoom, True)

        # 通道控制
        self.tYZoomLabel = [QLabel("zoom"), QLabel("zoom")]
        self.tYZoom = [logSpinBox(), logSpinBox()]
        self.OffsetLabel = [QLabel("offset"), QLabel("offset")]
        self.Offset = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.tYZoom[0].setParameters(mi=0.1, ma=20, val=1, step=2, decimal=2)
        self.tYZoom[1].setParameters(mi=0.1, ma=20, val=1, step=2, decimal=2)
        self.Offset[0].setSingleStep(0.5)
        self.Offset[0].setMinimum(-10)
        self.Offset[0].setMaximum(10)
        self.Offset[1].setSingleStep(0.5)
        self.Offset[1].setMinimum(-10)
        self.Offset[1].setMaximum(10)
        # 触发设置
        self.Trigger = [doubleSlider(), doubleSlider()]
        for t in self.Trigger:
            t.setParameters(-10, 10, 2000)
            t.setValue(0)
            t.setOrientation(Qt.Vertical)
            t.setFixedWidth(10)
        self.TriggerCB = [QCheckBox("Trig"), QCheckBox("Trig")]
        self.TriggerCB[0].setCheckState(Qt.Checked)
        self.TriggerCB[1].setCheckState(Qt.Checked)
        self.TriggerTH = [QSpinBox(), QSpinBox()]
        self.TriggerTH[0].setRange(1, 100)
        self.TriggerTH[0].setValue(5)
        self.TriggerTH[1].setRange(1, 100)
        self.TriggerTH[1].setValue(5)
        self.TriggerSlopeLabel = [QLabel("边沿"), QLabel("边沿")]
        self.TriggerSlope = [QComboBox(), QComboBox()]
        self.TriggerSlope[0].addItems(("↑", "↓"))
        self.TriggerSlope[1].addItems(("↑", "↓"))
        self.rightGrid.addGrid(1, 0)
        self.rightGrid.setTitle("Ch 1")
        self.rightGrid.addWidget(self.Trigger[0], rowSpan=5)
        self.rightGrid.addWidget(self.tYZoomLabel[0])
        self.rightGrid.addWidget(self.tYZoom[0], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.OffsetLabel[0])
        self.rightGrid.addWidget(self.Offset[0], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.TriggerCB[0], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(QLabel("脉宽"))
        self.rightGrid.addWidget(self.TriggerTH[0], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.TriggerSlopeLabel[0])
        self.rightGrid.addWidget(self.TriggerSlope[0], True)
        self.rightGrid.addGrid(1, 1)
        self.rightGrid.setTitle("Ch 2")
        self.rightGrid.addWidget(self.Trigger[1], rowSpan=5)
        self.rightGrid.addWidget(self.tYZoomLabel[1])
        self.rightGrid.addWidget(self.tYZoom[1], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.OffsetLabel[1])
        self.rightGrid.addWidget(self.Offset[1], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.TriggerCB[1], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(QLabel("脉宽"))
        self.rightGrid.addWidget(self.TriggerTH[1], True)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.TriggerSlopeLabel[1])
        self.rightGrid.addWidget(self.TriggerSlope[1], True)

        self.fLogCB = QCheckBox("f log")
        self.WinTypeLabel = QLabel("窗类型")
        self.WinType = QComboBox()
        self.WinType.addItems(("Hanning", "Hamming", "Blackman", "Bartlete", "Rect"))
        self.fftNLabel = QLabel("N")
        self.fftN = logSpinBox()
        self.fftN.setParameters(mi=128, ma=32768, val=1024, step=2, decimal=0)
        self.fftN.setValue(16384)
        self.fZoomLabel = QLabel("zoom")
        self.fZoom = logSpinBox()
        self.fZoom.setParameters(mi=1, ma=128, val=1, step=2, decimal=0)
        self.fZoom.setValue(16)
        self.fYLimLabel = [QLabel("Ymin"), QLabel("Ymax")]
        self.fYLim = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.fYLim[0].setRange(-120, 20)
        self.fYLim[0].setSingleStep(5)
        self.fYLim[0].setValue(-120)
        self.fYLim[1].setRange(-120, 20)
        self.fYLim[1].setSingleStep(5)
        self.fYLim[1].setValue(20)
        self.fPos = doubleSlider()
        self.fPos.setOrientation(Qt.Horizontal)
        self.fPos.setFixedHeight(10)
        self.rightGrid.addGrid(2, 0, 1, 2)
        self.rightGrid.setTitle("FFT")
        self.rightGrid.addWidget(self.fLogCB)
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.fftNLabel)
        self.rightGrid.addWidget(self.fftN, True)
        self.rightGrid.addWidget(self.WinTypeLabel, colSpan=2)
        self.rightGrid.addWidget(self.WinType, True, colSpan=2)
        self.rightGrid.addWidget(self.fZoomLabel, colSpan=2)
        self.rightGrid.addWidget(self.fZoom, True, colSpan=2)
        self.rightGrid.addWidget(self.fYLimLabel[0])
        self.rightGrid.addWidget(self.fYLim[0])
        self.rightGrid.addWidget(self.fYLimLabel[1])
        self.rightGrid.addWidget(self.fYLim[1], True)
        self.rightGrid.addWidget(self.fPos, True, colSpan=4)

        self.DataDispCB = QCheckBox("values")
        self.CursorTarget = QComboBox()
        self.CursorTarget.addItems(("信号", "频谱"))
        self.CursorCB = [QCheckBox("光标1"), QCheckBox("光标2")]
        self.MeasChan = QComboBox()
        self.MeasChan.addItems(("1", "2"))
        self.rightGrid.addGrid(3, 0)
        self.rightGrid.setTitle("Measure")
        self.rightGrid.addWidget(self.DataDispCB, True)
        self.rightGrid.addWidget(QLabel("测量对象"))
        self.rightGrid.addWidget(self.CursorTarget, True)
        self.rightGrid.addWidget(QLabel("测量通道"))
        self.rightGrid.addWidget(self.MeasChan, True)
        self.rightGrid.addWidget(self.CursorCB[0])
        self.rightGrid.addWidget(self.CursorCB[1], True)

        self.SaveSrc = QComboBox()
        self.SaveSrc.addItems(("信号", "频谱"))
        self.SaveChannel = QComboBox()
        self.SaveChannel.addItems(("1 & 2", "1", "2"))
        self.PicSave = QPushButton("保存图片")
        self.DataSave = QPushButton("保存数据")
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.RecordStart = SwitchButton(texts=["开始", "结束"])
        self.rightGrid.addGrid(3, 1)
        self.rightGrid.setTitle("Save")
        self.rightGrid.addWidget(QLabel("数据类型"))
        self.rightGrid.addWidget(self.SaveSrc, True)
        self.rightGrid.addWidget(QLabel("通道"))
        self.rightGrid.addWidget(self.SaveChannel, True)
        self.rightGrid.addWidget(self.DataSave)
        self.rightGrid.addWidget(self.PicSave, True)
        self.rightGrid.addWidget(line, True, colSpan=2)
        self.rightGrid.addWidget(QLabel("录音"))
        self.rightGrid.addWidget(self.RecordStart, True)

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
    ui = Ui_scope()
    ui.show()
    sys.exit(app.exec_())
