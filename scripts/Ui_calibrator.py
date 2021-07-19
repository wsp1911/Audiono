# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
import configparser
from PyQt5 import QtCore
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QGridLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt5.QtGui import QFont
from pyaudio import PyAudio, paContinue

from public import BasicWindow


class Ui_calibrator(BasicWindow):
    def __init__(self, parent=None, scope=None):
        super().__init__(parent)
        self.scope = scope

        WIDTH, HEIGHT = 400, 300

        # self.resize(WIDTH, HEIGHT)
        self.setFixedSize(400, 300)
        self.setWindowTitle("校准")
        font_size = 9
        self.setFont(QFont("等线", font_size))

        if scope is not None:
            self.CHANNELS = self.scope.CHANNELS
            self.FORMAT = self.scope.FORMAT
            self.RATE = self.scope.RATE
            self.CHUNK = self.scope.CHUNK

        self.config = configparser.ConfigParser()
        self.config.read("Audiono.ini")

        self.rec_bytes = np.zeros(2 * self.CHUNK, dtype=np.int16).tobytes()

        t = np.arange(self.CHUNK) / self.RATE
        y = np.sin(2 * np.pi * 1000 * t)
        self.test_bytes = (-0.5 * np.repeat(y, 2) * 32768).astype(np.int16).tobytes()

        self.QGB = QGroupBox(self)
        self.QGB.setGeometry(QRect(0, 0, WIDTH, HEIGHT))

        self.ManualCB = QCheckBox("手动校准")
        self.ManualCB.setCheckState(QtCore.Qt.Checked)
        self.SaveBtn = QPushButton()
        self.SaveBtn.setText("保存")
        self.CalSelect = QComboBox()
        self.CalSelect.addItems(("示波器", "信号发生器"))
        self.StartCal = QPushButton()
        self.StartCal.setText("校准")

        self.values = [QDoubleSpinBox(), QDoubleSpinBox()]
        for item in self.values:
            item.setRange(0.1, 20)
            item.setDecimals(3)

        self.labels = [QLabel("测量值"), QLabel("真实值")]

        self.Grid = QGridLayout(self.QGB)
        self.Grid.addWidget(self.ManualCB, 0, 0)
        self.Grid.addWidget(self.SaveBtn, 0, 1)
        self.Grid.addWidget(self.CalSelect, 1, 0)
        self.Grid.addWidget(self.StartCal, 1, 1)
        self.Grid.addWidget(self.labels[0], 2, 0)
        self.Grid.addWidget(self.values[0], 2, 1)
        self.Grid.addWidget(self.labels[1], 3, 0)
        self.Grid.addWidget(self.values[1], 3, 1)

        self.connect()

    def connect(self):
        self.StartCal.clicked.connect(self.on_StartCal_clicked)
        self.CalSelect.currentIndexChanged.connect(
            self.on_CalSelect_currentIndexChanged
        )
        self.SaveBtn.clicked.connect(self.on_SaveBtn_clicked)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.test_bytes
        self.rec_bytes = in_data
        return (data, paContinue)

    def on_SaveBtn_clicked(self):
        self.config["Gain"]["input_gain"] = str(self.scope.input_gain)
        self.config["Gain"]["output_gain"] = str(self.scope.output_gain)
        with open("Audiono.ini", "w") as f:
            self.config.write(f)
        self.inform("保存成功")

    def on_CalSelect_currentIndexChanged(self):
        if self.CalSelect.currentIndex() == 0:
            self.labels[0].setText("测量值")
        else:
            self.labels[0].setText("设定值")

    def update_input_gain(self, input_gain):
        self.config["Gain"]["input_gain"] = str(input_gain)
        self.scope.input_gain = input_gain
        if hasattr(self.scope, "ts"):
            self.scope.ts.input_gain = input_gain

    def update_output_gain(self, output_gain):
        self.config["Gain"]["output_gain"] = str(output_gain)
        self.scope.output_gain = output_gain
        if hasattr(self.scope, "sg"):
            self.scope.sg.output_gain = output_gain
            self.scope.sg.refresh_wave(0)
            self.scope.sg.refresh_wave(1)

    def on_StartCal_clicked(self):
        if self.ManualCB.checkState():
            if self.CalSelect.currentIndex() == 0:
                self.update_input_gain(
                    self.scope.input_gain
                    * self.values[1].value()
                    / self.values[0].value()
                )
                self.inform("输入增益校准成功")
            else:
                self.update_output_gain(
                    self.scope.output_gain
                    * self.values[0].value()
                    / self.values[1].value()
                )
                self.inform("输出增益校准成功")
        else:
            if self.scope.RUNNING:
                self.warn("先关闭示波器")
                return
            pa = PyAudio()
            stream = pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                output=True,
                stream_callback=self.callback,
                frames_per_buffer=self.CHUNK,
            )

            stream.start_stream()

            threshold = 0.1
            detected = False

            t = time.perf_counter()
            while time.perf_counter() - t < 5:
                y = np.frombuffer(self.rec_bytes, dtype=np.int16) / 32768
                left, right = y[1::2], y[::2]
                Vmax, Vmin = np.max(right), np.min(right)
                if Vmax > threshold and Vmin < -threshold:
                    detected = True
                    break
                time.sleep(self.CHUNK / self.RATE / 2)

            if detected:
                left_v, right_v = np.mean(left), Vmax - Vmin
                self.update_input_gain(5 / left_v)
                self.update_output_gain(left_v / right_v / 5)
                self.inform("校准成功")
            else:
                self.warn("校准失败")

            stream.stop_stream()
            pa.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_calibrator()
    ui.show()
    sys.exit(app.exec_())
