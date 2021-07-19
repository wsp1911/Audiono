# -*- coding: utf-8 -*-
import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
)
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from public import (
    logSpinBox,
    WidgetWithSplitter,
    doubleSlider,
    SwitchButton,
    BasicWindow,
)


class Ui_timeSpectrum(BasicWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent, hasThread=True)

        # 获取显示器分辨率大小
        desktop = QApplication.desktop()
        screenRect = desktop.availableGeometry()
        HEIGHT = int(0.7 * screenRect.height())
        WIDTH = HEIGHT + 300

        self.setGeometry(
            (screenRect.width() - WIDTH) // 2,
            (screenRect.height() - HEIGHT) // 2,
            WIDTH,
            HEIGHT,
        )
        self.setWindowTitle("时谱图仪")
        font_size = 9
        self.setFont(QFont("等线", font_size))

        self.setCentralWidget(QWidget())

        self.MainWidget = WidgetWithSplitter(self.centralWidget(), sizes=[WIDTH, 0])
        self.leftGrid = self.MainWidget.getGrid(0)
        self.rightGrid = self.MainWidget.getGrid(1)

        self.glw = []
        self.glw.append(pg.GraphicsLayoutWidget())
        self.plt = []
        self.plt.append(self.glw[0].addPlot())
        self.glw[0].nextRow()
        self.plt.append(self.glw[0].addPlot())
        self.plt[0].showAxis("right")
        self.plt[1].showAxis("right")

        self.glw.append(pg.GraphicsLayoutWidget())
        self.heatmap_plt = self.glw[1].addPlot()
        self.heatmap = pg.ImageItem()
        self.hist = pg.HistogramLUTItem(fillHistogram=False)
        self.hist.setImageItem(self.heatmap)
        self.glw[1].addItem(self.hist)
        self.heatmap_plt.addItem(self.heatmap)
        # p1.autoRange()

        self.glw.append(gl.GLViewWidget())
        # self.glw[2].setCameraPosition(distance=50)
        self.glw[2].setCameraPosition(distance=60, elevation=20, azimuth=0)
        # self.glw[2].setCameraPosition(pos=Vector(30, -20, 20), azimuth=-30)
        self.surface = gl.GLSurfacePlotItem(
            shader="heightColor", computeNormals=False, smooth=False
        )
        surface_cmap = [0.2, 10, 0.6, 0.2, 3, 1, 0.2, 0, 2]
        self.surface.shader()["colorMap"] = np.array(surface_cmap)
        self.glw[2].addItem(self.surface)
        self.grid_size = (30, 50, 20)
        self.gx = gl.GLGridItem()
        self.gx.setSize(self.grid_size[2], self.grid_size[1])
        self.gx.rotate(90, 0, 1, 0)
        self.gx.translate(-self.grid_size[0] / 2, 0, 0)
        self.glw[2].addItem(self.gx)
        self.gy = gl.GLGridItem()
        self.gy.setSize(self.grid_size[0], self.grid_size[2])
        self.gy.rotate(90, 1, 0, 0)
        self.gy.translate(0, -self.grid_size[1] / 2, 0)
        self.glw[2].addItem(self.gy)
        self.gz = gl.GLGridItem()
        self.gz.setSize(self.grid_size[0], self.grid_size[1])
        self.gz.translate(0, 0, -self.grid_size[2] / 2)
        self.glw[2].addItem(self.gz)

        self.leftGrid.addGrid(0, 0)
        self.leftGrid.addWidget(self.glw[0])
        self.leftGrid.addWidget(self.glw[1])
        self.glw[1].setVisible(False)
        self.leftGrid.addWidget(self.glw[2])
        self.glw[2].setVisible(False)

        self.RunButton = SwitchButton(texts=["Run", "Stop"])
        self.ModeLabel = QLabel("模式")
        self.Mode = QComboBox()
        self.Mode.addItems(("line", "heatmap", "surface"))
        self.ChannelLabel = QLabel("通道")
        self.Channel = QComboBox()
        self.Channel.addItems(("1", "2"))

        self.rightGrid.addGrid(0, 0)
        self.rightGrid.setTitle("Ctrl")
        self.rightGrid.nextCol()
        self.rightGrid.addWidget(self.RunButton, True)
        self.rightGrid.addWidget(self.ModeLabel)
        self.rightGrid.addWidget(self.Mode, True)
        self.rightGrid.addWidget(self.ChannelLabel)
        self.rightGrid.addWidget(self.Channel, True)

        self.WinTypeLabel = QLabel("窗类型")
        self.WinType = QComboBox()
        self.WinType.addItems(("Hanning", "Hamming", "Blackman", "Bartlete", "Rect"))
        self.fftNLabel = QLabel("FFT点数")
        self.fftN = logSpinBox()
        self.fftN.setParameters(mi=128, ma=32768, val=1024, step=2, decimal=0)
        self.fftN.setValue(8192)
        self.FramesLabel = QLabel("帧数")
        self.Frames = QSpinBox()
        self.Frames.setRange(10, 250)
        self.Frames.setSingleStep(5)
        self.Frames.setValue(50)
        self.ThresholdLabel = [QLabel("下阈值"), QLabel("上阈值")]
        self.Threshold = [QSpinBox(), QSpinBox()]
        self.Threshold[0].setRange(-120, 20)
        self.Threshold[0].setSingleStep(5)
        self.Threshold[0].setValue(-40)
        self.Threshold[1].setRange(-120, 20)
        self.Threshold[1].setSingleStep(5)
        # self.Threshold[1].setValue(0)
        self.fZoomLabel = QLabel("zoom")
        self.fZoom = logSpinBox()
        self.fZoom.setParameters(mi=1, ma=256, val=1, step=2, decimal=0)
        self.fZoom.setValue(16)
        self.fPos = doubleSlider()
        self.fPos.setOrientation(Qt.Horizontal)
        self.fPos.setFixedHeight(10)
        self.rightGrid.addGrid(1, 0)
        self.rightGrid.setSpacing(30, orientation=1)
        self.rightGrid.setTitle("FFT")
        self.rightGrid.addWidget(self.WinTypeLabel)
        self.rightGrid.addWidget(self.WinType, True)
        self.rightGrid.addWidget(self.ThresholdLabel[1])
        self.rightGrid.addWidget(self.Threshold[1], True)
        self.rightGrid.addWidget(self.ThresholdLabel[0])
        self.rightGrid.addWidget(self.Threshold[0], True)
        self.rightGrid.addWidget(self.FramesLabel)
        self.rightGrid.addWidget(self.Frames, True)
        self.rightGrid.addWidget(self.fftNLabel)
        self.rightGrid.addWidget(self.fftN, True)
        self.rightGrid.addWidget(self.fZoomLabel)
        self.rightGrid.addWidget(self.fZoom, True)
        self.rightGrid.addWidget(self.fPos, True, colSpan=2)

        self.rightGrid.addGrid(2, 0)
        self.rightGrid.setTitle("Size && Color")
        self.GridSize = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.rightGrid.addWidget(QLabel("尺寸"))
        for i in range(3):
            self.GridSize[i].setRange(5, 200)
            self.GridSize[i].setSingleStep(5)
            self.rightGrid.addWidget(self.GridSize[i])
        self.rightGrid.nextRow()
        self.GridSize[0].setValue(self.grid_size[0])
        self.GridSize[1].setValue(self.grid_size[1])
        self.GridSize[2].setValue(self.grid_size[2])

        self.surface_cmap = [QDoubleSpinBox() for i in range(9)]
        for i in range(3):
            self.surface_cmap[i * 3].setSingleStep(0.01)
            self.surface_cmap[i * 3 + 2].setSingleStep(0.1)
            self.rightGrid.addWidget(QLabel("RGB"[i]))
            for j in range(3):
                self.surface_cmap[i * 3 + j].setValue(surface_cmap[i * 3 + j])
                self.rightGrid.addWidget(self.surface_cmap[i * 3 + j])
            self.rightGrid.nextRow()

        self.Mode.currentIndexChanged.connect(self.switch_mode)

        self.adjust_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_size()

    def switch_mode(self):
        idx = self.Mode.currentIndex()
        for i in [0, 1, 2]:
            self.glw[i].setVisible(i == idx)

    def adjust_size(self):
        WIDTH, HEIGHT = self.width(), self.height()
        self.MainWidget.setGeometry(0, 0, WIDTH, HEIGHT)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_timeSpectrum()
    ui.show()
    sys.exit(app.exec_())
