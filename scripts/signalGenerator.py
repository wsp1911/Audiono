# -*- coding: utf-8 -*-
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal
from pyaudio import PyAudio, paContinue, paInt16

from Ui_signalGenerator import Ui_signalGenerator
from sigGenerate import getWave, from_file, from_exp


class signalGenerator(Ui_signalGenerator):
    signal_send = pyqtSignal()

    def __init__(
        self, parent=None, scope=None, chunk=1024, rate=48000, output_gain=1 / 15
    ):
        super().__init__(parent)
        self.scope = scope

        self.REFRESH = [False, False]
        self.rule = [
            [1, 2, 4, 5],  # 受freq影响
            [1, 2, 3, 4, 5],  # 受vpp影响
            [1, 2, 3, 4, 5],  # 受offset影响
            [2, 4],  # 受duty影响
            [1, 2, 4, 5],  # 受phi影响
        ]
        self.CHUNK = chunk
        self.RATE = rate
        self.FORMAT = paInt16
        self.CHANNELS = 2
        self.RUNNING = False
        self.output_gain = output_gain
        self.idx = [0, 0]
        self.sig = [np.zeros(self.CHUNK), np.zeros(self.CHUNK)]
        self.period_N = [0, 0]  # 一个周期的点数
        self.play_y = np.zeros(self.CHUNK * 2)
        self.play_bytes = np.zeros(self.CHUNK * 2, dtype=np.int16).tobytes()
        self.file_buffer = [0, 0]
        self.file_N = [0, 0]

        self.connect()

    def closeEvent(self, a0) -> None:
        if self.RUNNING:
            self.RUNNING = False
            self.RunButton.setState(True)
            self.delay()
            self.quit_thread()
        return super().closeEvent(a0)

    def connect(self):
        self.synchSelect.currentIndexChanged.connect(
            self.on_synchSelect_currentIndexChanged
        )
        self.RunButton.clicked.connect(self.on_RunButton_clicked)

        self.waveSelect[0].currentIndexChanged.connect(lambda: self.refresh(0, -1))
        self.freqInput[0].valueChanged.connect(lambda: self.refresh(0, 0))
        self.vppInput[0].valueChanged.connect(lambda: self.refresh(0, 1))
        self.offsetInput[0].valueChanged.connect(lambda: self.refresh(0, 2))
        self.dutyInput[0].valueChanged.connect(lambda: self.refresh(0, 3))
        self.phiInput[0].valueChanged.connect(lambda: self.refresh(0, 4))
        self.waveSelect[1].currentIndexChanged.connect(lambda: self.refresh(1, -1))
        self.freqInput[1].valueChanged.connect(lambda: self.refresh(1, 0))
        self.vppInput[1].valueChanged.connect(lambda: self.refresh(1, 1))
        self.offsetInput[1].valueChanged.connect(lambda: self.refresh(1, 2))
        self.dutyInput[1].valueChanged.connect(lambda: self.refresh(1, 3))
        self.phiInput[1].valueChanged.connect(lambda: self.refresh(1, 4))

        self.playExp[0].clicked.connect(lambda: self.on_playExp_clicked(0))
        self.playExp[1].clicked.connect(lambda: self.on_playExp_clicked(1))
        self.openFile[0].clicked.connect(lambda: self.open_file(0))
        self.playFile[0].clicked.connect(lambda: self.on_playFile_cilcked(0))
        self.openFile[1].clicked.connect(lambda: self.open_file(1))
        self.playFile[1].clicked.connect(lambda: self.on_playFile_cilcked(1))
        self.clearFile.clicked.connect(self.clear_file)

        self.signal_send.connect(self.slot_send)

    def on_RunButton_clicked(self):
        self.RUNNING = not self.RUNNING
        if self.RUNNING:
            self.start_thread()
        else:
            self.quit_thread()

    def open_file(self, channel):
        filename = QFileDialog.getOpenFileName(
            self,
            caption="选择数据文件",
            filter="MATLAB (*.mat);;Python (*.npy);;TXT (*.txt);;Excel (*.xlsx);;Wave (*.wav)",
        )
        # print(filename)
        if filename[0] == "":  # 取消选择
            return
        if self.interCB.checkState():
            N, data = from_file(self.CHUNK, filename[0], fs=self.RATE)
        else:
            N, data = from_file(self.CHUNK, filename[0])
        if N == -1:
            self.critical("读取文件时出现错误：\n%s" % data)
        elif N == -2:
            self.critical("文件为空，载入数据失败")
        else:
            if filename[1] == "Wave (*.wav)":
                data *= self.scope.input_gain
            data *= self.output_gain
            self.show_message("从" + filename[0] + "载入数据成功")
            self.update_file_buffer(channel, N, data)

    def update_file_buffer(self, channel, N, data):
        """
        N: data length in 1 period
        data: N*1 or N*2 ndarray, N>CHUNK
        """

        if data.shape[1] == 1:
            self.file_buffer[channel] = data.flatten()
            self.file_N[channel] = N
            self.fileInfo[channel].setText(
                "长度：%d 时长：%.3f" % (data.shape[0], data.shape[0] / self.RATE)
            )
        elif self.file_N[1 - channel]:
            replace = QMessageBox.question(
                self,
                "选择",
                "通道%d已存在文件数据，是否覆盖？" % (1 - channel),
                QMessageBox.Yes | QMessageBox.No,
            )
            if replace == QMessageBox.Yes:
                self.file_N = [N, N]
                for i in [0, 1]:
                    self.file_buffer[i] = data[:, i]
                    self.fileInfo[i].setText(
                        "长度：%d 时长：%.3fs" % (data.shape[0], data.shape[0] / self.RATE)
                    )
            else:
                self.file_buffer[channel] = data[:, channel].flatten()
                self.file_N[channel] = N
                self.fileInfo[channel].setText(
                    "长度：%d 时长：%.3fs" % (data.shape[0], data.shape[0] / self.RATE)
                )
        else:
            self.file_N = [N, N]
            for i in [0, 1]:
                self.file_buffer[i] = data[:, i]
                self.fileInfo[i].setText(
                    "长度：%d 时长：%.3fs" % (data.shape[0], data.shape[0] / self.RATE)
                )

    def clear_file(self):
        self.file_buffer = [0, 0]
        self.file_N = [0, 0]
        self.fileInfo[0].clear()
        self.fileInfo[1].clear()

    def on_playFile_cilcked(self, channel):
        if self.file_N[channel] == 0:
            self.warn("通道%d尚未载入文件" % (channel + 1))
            return
        else:
            self.period_N[channel] = self.file_N[channel]
            self.sig[channel] = self.file_buffer[channel]
            self.idx[channel] = 0
            if self.synchSelect.currentIndex():
                self.idx[1 - channel] = 0
            self.update_plt(channel)
            self.waveSelect[channel].setCurrentIndex(7)

    def on_playExp_clicked(self, channel):
        N, data = from_exp(
            self.RATE, self.CHUNK, self.output_gain, self.expInput[channel].text()
        )
        if N == -1:
            self.critical("计算失败，请确保表达式格式符合要求")
        else:
            self.period_N[channel] = N
            self.sig[channel] = data
            self.idx[channel] = 0
            if self.synchSelect.currentIndex():
                self.idx[1 - channel] = 0
            self.update_plt(channel)
            self.waveSelect[channel].setCurrentIndex(6)

    # 0, sin, square, DC, triangle, sawtooth,
    # 0,   1,      2,  3,        4,        5,
    # var_id:
    # freq, vpp, offset, duty, phi, exp, file
    #    0,   1,      2,    3,   4,   5,    6
    def refresh(self, c_id, var_id):
        if var_id == -1:
            if self.waveSelect[c_id].currentIndex() in [6, 7]:
                return
            self.refresh_wave(c_id)
        elif self.waveSelect[c_id].currentIndex() in self.rule[var_id]:
            self.refresh_wave(c_id)

    def refresh_wave(self, c_id):
        """
        for normal mode; expression and file mode are written separately
        """
        synch = self.synchSelect.currentIndex()
        wave_type = self.waveSelect[c_id].currentIndex()
        self.period_N[c_id], self.sig[c_id] = getWave(
            self.CHUNK,
            self.output_gain,
            self.RATE,
            wave_type,
            self.freqInput[c_id].value(),
            self.vppInput[c_id].value(),
            self.dutyInput[c_id].value() / 100,
            self.offsetInput[c_id].value(),
            self.phiInput[c_id].value(),
        )
        self.idx[c_id] = 0
        if synch:
            self.idx[1 - c_id] = 0
        self.update_plt(c_id)

    def on_synchSelect_currentIndexChanged(self):
        if self.synchSelect.currentIndex() == 1:
            self.idx[0] = 0
            self.idx[1] = 0

    def update_plt(self, c_id):
        if self.period_N[c_id] == -1:
            self.pw[c_id].curve.setData()
        else:
            self.pw[c_id].curve.setData(
                np.arange(self.period_N[c_id]) / self.RATE,
                self.sig[c_id][: self.period_N[c_id]] / self.output_gain,
            )

    def extend(self, cid):
        # 需要保证sig长度不小于CHUNK
        x_len = len(self.sig[cid])
        if self.idx[cid] + self.CHUNK <= x_len:
            return self.sig[cid][self.idx[cid] : self.idx[cid] + self.CHUNK]
        else:
            return np.r_[
                self.sig[cid][self.idx[cid] :],
                self.sig[cid][: self.CHUNK + self.idx[cid] - x_len],
            ]

    def update_sig(self):
        self.play_y = np.array([self.extend(1), self.extend(0)]).T.flatten()
        self.idx[0] = (self.idx[0] + self.CHUNK) % len(self.sig[0])
        self.idx[1] = (self.idx[1] + self.CHUNK) % len(self.sig[1])
        self.play_bytes = (-self.play_y * 32768).astype(np.int16).tobytes()

    def reset(self):
        self.play_y = np.zeros(self.CHUNK * 2)
        self.play_bytes = np.zeros(self.CHUNK * 2, dtype=np.int16).tobytes()
        self.idx = [0, 0]

    def callback(self, in_data, frame_count, time_info, status):
        data = self.play_bytes
        self.signal_send.emit()
        return (data, paContinue)

    def slot_send(self):
        self.update_sig()

    def run(self):
        pa = PyAudio()
        stream = pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            stream_callback=self.callback,
            frames_per_buffer=self.CHUNK,
        )
        stream.start_stream()
        while self.RUNNING:
            self.delay()
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = signalGenerator()
    win.show()
    sys.exit(app.exec_())

    # import matplotlib.pyplot as plt

    # win.period_N[0], win.sig[0] = getWave(
    #     win.CHUNK,
    #     1,
    #     win.RATE,
    #     1,
    #     100,
    #     10,
    #     50 / 100,
    #     0,
    #     0,
    # )
    # win.idx[0] = 0
    # y = []
    # N = 5
    # for i in range(N):
    #     y.append(win.extend(0))
    #     win.idx[0] = (win.idx[0] + win.CHUNK) % len(win.sig[0])

    # y = np.array(y).flatten()

    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(win.sig[0])) / win.CHUNK, win.sig[0])
    # plt.title("(1)")
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(len(y)) / win.CHUNK, y)
    # for i in range(N - 1):
    #     plt.axvline(x=i + 1, ls="--", c="orange")
    # plt.xlabel("N/CHUNK")
    # plt.title("(2)")
    # plt.show()
