# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import QTimer, pyqtSignal
import pyqtgraph.exporters as pg_exporter
from pyaudio import PyAudio, paInt16, paContinue

from Ui_scope import Ui_scope

from public import linear_map, save_data, find_first, Config


class scope(Ui_scope):
    signal_received = pyqtSignal()

    def __init__(self, parent=None, config=None):

        if config:
            self.config = config
        else:
            self.config = Config("Audiono.ini")

        # t1 = time.perf_counter()
        super(scope, self).__init__(parent, rate=self.config.RATE)
        # print("load scope Ui time: %f" % (time.perf_counter() - t1))

        # t1 = time.perf_counter()
        # 信号参数
        self.zero_bytes = np.zeros(2 * self.config.CHUNK, dtype=np.int16).tobytes()
        self.rec_bytes = np.zeros(2 * self.config.CHUNK, dtype=np.int16).tobytes()
        self.rec_y = [np.zeros(self.config.CHUNK), np.zeros(self.config.CHUNK)]
        self.disp_start = [0, 0]
        self.disp_len = self.config.CHUNK // 16
        self.disp_sig = [self.rec_y[0][: self.disp_len], self.rec_y[1][: self.disp_len]]
        self.T = np.arange(self.config.CHUNK) / self.config.RATE
        self.t_max = self.T[self.disp_len - 1]
        self.TX = [0, 1]
        self.TY = [0, 1]
        # fft 参数
        self.nfft = self.fftN.value()
        num = self.nfft // 2 + 1
        self.f_max = self.config.RATE / 2
        self.FFT = [np.zeros(num), np.zeros(num)]
        self.disp_FFT = [-120 * np.ones(num), -120 * np.ones(num)]
        self.f = np.linspace(0, self.f_max, num)
        self.f_log = False
        self.f_lim = [0, self.f_max / self.fZoom.value()]
        self.fY_lim = [-120, 20]
        self.fft_win = np.hanning(self.config.CHUNK)
        self.FX = [0, 1]
        self.FY = [0, 1]
        # 程序控制
        self.RUNNING = False
        # 录音
        self.RECORDING = False
        self.record_data = [[], []]
        # 数据导出
        self.PICSAVE = False
        self.DATASAVE = False

        self.connect()

        self.timer_scope = QTimer()
        self.timer_scope.timeout.connect(self.update_plt)
        self.timer_scope_t = 50
        self.timer_scope.start(self.timer_scope_t)
        self.timer_trigger = QTimer()
        self.timer_trigger.timeout.connect(self.clear_trigger)
        self.t_t = 0
        self.t_f = 0
        self.plt_cnt = 0

        # print("scope init time: %f" % (time.perf_counter() - t1))

    def closeEvent(self, a0) -> None:
        """
        效果上看实现scope是各模块的parent，但避免了children窗口一直在parent之上的问题
        """
        if self.RUNNING:
            self.RUNNING = False
            self.RunButton.setState(True)
            self.delay()
            self.quit_thread()
        return super().closeEvent(a0)

    def show(self):
        # 如果show之前设置坐标轴范围，可能导致实际范围过大
        super().show()
        self.pw[0].setXRange(0, self.t_max)
        self.pw[0].setYRange(-10, 10)
        self.pw[1].setXRange(0, self.f_max / self.fZoom.value())
        self.pw[1].setYRange(-120, 20)

    def connect(self):
        self.DataDispCB.stateChanged.connect(self.update_disp_data)
        self.CursorTarget.currentIndexChanged.connect(self.on_cursor_valueChanged)
        self.CursorCB[0].stateChanged.connect(self.redraw_cursor)
        self.CursorCB[1].stateChanged.connect(self.redraw_cursor)

        self.RunButton.clicked.connect(self.on_RunButton_clicked)
        self.Channel.currentIndexChanged.connect(self.on_Channel_currentIndexChanged)

        # 多次connect就可以触发多个事件，amazing!
        # 多个事件按照连接的先后顺序执行！
        self.Trigger[0].valueChanged.connect(self.update_trig_time)
        self.Trigger[0].valueChanged.connect(self.update_disp_sig_WNR)
        self.Trigger[1].valueChanged.connect(self.update_trig_time)
        self.Trigger[1].valueChanged.connect(self.update_disp_sig_WNR)

        self.tZoom.valueChanged.connect(self.update_t_max)
        self.tZoom.valueChanged.connect(self.update_disp_sig_WNR)  # 显示长度改变后防止越界需要再触发

        self.TriggerCB[0].stateChanged.connect(self.update_disp_sig_WNR)
        self.TriggerCB[1].stateChanged.connect(self.update_disp_sig_WNR)
        self.TriggerSlope[0].currentIndexChanged.connect(self.update_disp_sig_WNR)
        self.TriggerSlope[1].currentIndexChanged.connect(self.update_disp_sig_WNR)

        self.fftN.valueChanged.connect(self.update_nfft)
        self.fftN.valueChanged.connect(self.update_disp_FFT_WNR)
        self.WinType.currentIndexChanged.connect(self.update_fft_win)
        self.WinType.currentIndexChanged.connect(self.update_disp_FFT_WNR)

        self.CursorX[0].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorX[1].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorY[0].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorY[1].valueChanged.connect(self.on_cursor_valueChanged)

        self.tYZoom[0].valueChanged.connect(self.update_disp_data)
        self.tYZoom[1].valueChanged.connect(self.update_disp_data)
        self.Offset[0].valueChanged.connect(self.update_disp_data)
        self.Offset[1].valueChanged.connect(self.update_disp_data)
        self.MeasChan.currentIndexChanged.connect(self.update_disp_data)

        self.fLogCB.stateChanged.connect(self.on_fLogCB_stateChanged)
        self.fPos.valueChanged.connect(self.update_f_lim)
        self.fZoom.valueChanged.connect(self.update_f_lim)
        self.fYLim[0].valueChanged.connect(self.update_fY_lim)
        self.fYLim[1].valueChanged.connect(self.update_fY_lim)

        self.PicSave.clicked.connect(self.on_PicSave_clicked)
        self.DataSave.clicked.connect(self.on_DataSave_clicked)
        self.RecordStart.clicked.connect(self.on_RecordStart_clicked)

        self.signal_received.connect(self.slot_received)

    def on_RunButton_clicked(self):
        self.RUNNING = not self.RUNNING
        if self.RUNNING:
            self.start_thread()
        else:
            self.quit_thread()

    def save_pic(self):
        if self.SaveSrc.currentIndex() == 0:
            ex = pg_exporter.ImageExporter(self.pw[0].scene())
        else:
            ex = pg_exporter.ImageExporter(self.pw[1].scene())
        filename = QFileDialog.getSaveFileName(
            self, caption="窗口1图片保存为", filter="PNG (*.png);;JPEG (*.jpg)"
        )
        if filename[0]:
            ex.export(filename[0])

    def on_PicSave_clicked(self):
        if self.RUNNING:
            self.PICSAVE = True
        else:
            self.save_pic()

    def on_DataSave_clicked(self):
        src = self.SaveSrc.currentIndex()
        i = self.SaveChannel.currentIndex()
        if src == 0:
            if i == 0:
                data_to_save = np.array(self.rec_y).T
            elif i == 1:
                data_to_save = self.rec_y[0]
            else:
                data_to_save = self.rec_y[1]
        else:
            if i == 0:
                data_to_save = np.array([self.f] + self.disp_FFT).T
            elif i == 1:
                data_to_save = np.array([self.f, self.disp_FFT[0]]).T
            else:
                data_to_save = np.array([self.f, self.disp_FFT[1]]).T
        filename = QFileDialog.getSaveFileName(
            self,
            caption="保存为",
            filter="MATLAB (*.mat);;Python (*.npy);;TXT (*.txt);;Excel (*.xlsx);;Wave (*.wav)",
        )
        if src == 1 and filename[1] == "Wave (*.wav)":
            self.critical("所选数据与所选格式不匹配，保存失败")
        if filename[0]:
            if filename[1] == "Wave (*.wav)":
                data_to_save = data_to_save / self.config.input_gain
            if not save_data(
                filename[0],
                data_to_save,
                channels=data_to_save.ndim,
                sampwidth=2,
                rate=self.config.RATE,
            ):
                self.critical("保存文件失败")

    def on_RecordStart_clicked(self):
        if self.RECORDING:
            self.RECORDING = False
            filename = QFileDialog.getSaveFileName(
                self,
                caption="保存为",
                filter="MATLAB (*.mat);;Python (*.npy);;TXT (*.txt);;Excel (*.xlsx);;Wave (*.wav)",
            )
            # print(filename)
            if filename[0]:
                data0 = np.array(self.record_data[0]).flatten()
                data1 = np.array(self.record_data[1]).flatten()
                data = np.array([data0, data1]).T
                if filename[1] == "Wave (*.wav)":
                    data /= self.config.input_gain
                if not save_data(filename[0], data, sampwidth=2, rate=self.config.RATE):
                    self.critical("保存文件失败")
            self.record_data = [[], []]
        else:
            if self.RUNNING:
                self.RECORDING = True
            else:
                self.RecordStart.setState(True)
                self.warn("示波器尚未启动")

    def update_t_max(self):
        length = int(self.config.CHUNK / self.tZoom.value())
        self.t_max = self.T[length - 1]
        if self.Channel.currentIndex() != 3:
            self.pw[0].setXRange(0, self.t_max)
            self.redraw_cursor()
            self.update_disp_data()

    def update_nfft(self):
        self.nfft = self.fftN.value()
        num = self.nfft // 2 + 1
        self.f = np.linspace(0, self.f_max, num)

    def update_fft_win(self):
        win_type = self.WinType.currentIndex()
        if win_type == 0:
            self.fft_win = np.hanning(self.config.CHUNK)
        elif win_type == 1:
            self.fft_win = np.hamming(self.config.CHUNK)
        elif win_type == 2:
            self.fft_win = np.blackman(self.config.CHUNK)
        elif win_type == 3:
            self.fft_win = np.bartlett(self.config.CHUNK)
        else:
            self.fft_win = np.ones(self.config.CHUNK)

    def update_f_lim(self):
        f_range = self.f_max // self.fZoom.value()
        f_start = self.fPos.value() * (self.f_max - f_range)
        self.f_lim = [f_start, f_start + f_range]
        if self.f_log:
            self.f_lim[0] = np.log10(self.f_lim[0]) if self.f_lim[0] > 0 else 0
            self.f_lim[1] = np.log10(self.f_lim[1])
        self.pw[1].setXRange(self.f_lim[0], self.f_lim[1])
        self.redraw_cursor()
        self.update_disp_data()

    def update_fY_lim(self):
        if self.fYLim[0].value() < self.fYLim[1].value():
            self.fY_lim = [self.fYLim[i].value() for i in [0, 1]]
        else:
            self.fY_lim = [
                min(np.min(self.disp_FFT[0]), np.min(self.disp_FFT[1])),
                max(np.max(self.disp_FFT[0]), np.max(self.disp_FFT[1])),
            ]
        self.pw[1].setYRange(self.fY_lim[0], self.fY_lim[1])
        self.redraw_cursor()
        self.update_disp_data()

    def on_Channel_currentIndexChanged(self, channel):
        channel = self.Channel.currentIndex()
        if channel == 0:
            self.pw[0].setAspectLocked(False)
            self.pw[0].curve[2].setData()
            self.redraw_cursor()
        elif channel == 1:
            self.pw[0].setAspectLocked(False)
            self.pw[0].curve[1].setData()
            self.pw[0].curve[2].setData()
            self.pw[1].curve[1].setData()
            self.redraw_cursor()
        elif channel == 2:
            self.pw[0].setAspectLocked(False)
            self.pw[0].curve[0].setData()
            self.pw[0].curve[2].setData()
            self.pw[1].curve[0].setData()
            self.redraw_cursor()
        else:
            self.pw[0].setAspectLocked(True)
            self.pw[0].setYRange(-10, 10)
            self.pw[0].curve[0].setData()
            self.pw[0].curve[1].setData()

    def on_cursor_valueChanged(self):
        self.update_cursor_XY()
        self.redraw_cursor()
        self.update_disp_data()

    def update_cursor_XY(self):
        """
        更新光标位置
        """
        # 光标作用于时域
        if self.CursorTarget.currentIndex() == 0:
            self.TX = [c.value() for c in self.CursorX]
            self.TY = [c.value() for c in self.CursorY]
        # 光标作用于频域
        else:
            self.FX = [c.value() for c in self.CursorX]
            self.FY = [c.value() for c in self.CursorY]

    def redraw_cursor(self):
        """
        重绘光标
        """
        if self.CursorCB[0].checkState() and self.Channel.currentIndex() != 3:
            X = linear_map(self.TX, (0, self.t_max))
            Y = linear_map(self.TY, (-10, 10))
            self.pw[0].set_cursors(X, Y)
        else:
            self.pw[0].set_cursors()
        if self.CursorCB[1].checkState():
            X = linear_map(self.FX, self.f_lim)
            Y = linear_map(self.FY, self.fY_lim)
            self.pw[1].set_cursors(X, Y)
        else:
            self.pw[1].set_cursors()

    def update_disp_data(self):
        """
        更新测量数据
        """
        if not self.DataDispCB.checkState():
            return
        cid = self.MeasChan.currentIndex()
        X = linear_map(self.TX, (0, 1000 * self.t_max))
        Y = [
            ((v * 20 - 10) - self.Offset[cid].value()) / self.tYZoom[cid].value()
            for v in self.TY
        ]
        if X[1] != X[0]:
            tmp = 1000 / abs(X[1] - X[0])
        else:
            tmp = "Inf"
        self.DataDispGrid[0].update_data(
            [X[0], X[1], Y[0], Y[1], X[1] - X[0], tmp, Y[1] - Y[0]]
        )
        X = linear_map(self.FX, self.f_lim)
        Y = linear_map(self.FY, self.fY_lim)
        if self.f_log and max(X[0], X[1]) < 10:
            X = [10 ** x for x in X]
        self.DataDispGrid[1].update_data(
            [X[0], X[1], Y[0], Y[1], X[1] - X[0], Y[1] - Y[0]]
        )

    def on_ALogCB_stateChanged(self):
        self.update_disp_FFT()
        self.update_fY_lim()

    def on_fLogCB_stateChanged(self):
        self.f_log = self.fLogCB.checkState()
        if self.f_log:
            self.pw[1].curve[0].setLogMode(True, False)
            self.pw[1].curve[1].setLogMode(True, False)
        else:
            self.pw[1].curve[0].setLogMode(False, False)
            self.pw[1].curve[1].setLogMode(False, False)
        self.update_f_lim()
        self.redraw_cursor()
        self.update_disp_data()

    # WNR: when not running
    def update_disp_sig_WNR(self):
        self.disp_len = int(self.config.CHUNK / self.tZoom.value())
        self.trigger()
        self.disp_sig[0] = self.rec_y[0][
            self.disp_start[0] : self.disp_start[0] + self.disp_len
        ]
        self.disp_sig[1] = self.rec_y[1][
            self.disp_start[1] : self.disp_start[1] + self.disp_len
        ]

    def update_disp_sig(self):
        self.disp_len = int(self.config.CHUNK / self.tZoom.value())
        self.disp_sig[0] = self.rec_y[0][
            self.disp_start[0] : self.disp_start[0] + self.disp_len
        ]
        self.disp_sig[1] = self.rec_y[1][
            self.disp_start[1] : self.disp_start[1] + self.disp_len
        ]

    def update_disp_FFT_WNR(self):
        if not self.RUNNING:
            self.update_disp_FFT()

    def update_disp_FFT(self):
        self.FFT = [
            np.fft.rfft(self.fft_win * self.rec_y[i], n=self.nfft) / self.config.CHUNK
            for i in [0, 1]
        ]
        self.disp_FFT = [
            20 * np.log10(np.abs(self.FFT[0]) + 1e-6),
            20 * np.log10(np.abs(self.FFT[1]) + 1e-6),
        ]

    def update_trig_time(self):
        self.pw[0].set_trigger([t.value() for t in self.Trigger])
        self.timer_trigger.start(3000)

    def clear_trigger(self):
        self.pw[0].set_trigger()
        self.timer_trigger.stop()

    def update_plt(self):
        self.update_plt_t()
        self.update_plt_f()
        if self.PICSAVE:
            self.save_pic()
            self.PICSAVE = False

    def update_plt_t(self):
        if self.RUNNING:
            t1 = time.perf_counter()
        else:
            t1 = 0

        channel = self.Channel.currentIndex()

        # 显示时域信号
        if channel == 0:  # 双通道
            self.pw[0].curve[0].setData(
                self.T[: self.disp_len],
                self.tYZoom[0].value() * self.disp_sig[0] + self.Offset[0].value(),
            )
            self.pw[0].curve[1].setData(
                self.T[: self.disp_len],
                self.tYZoom[1].value() * self.disp_sig[1] + self.Offset[1].value(),
            )
        elif channel == 1:  # 通道1
            self.pw[0].curve[0].setData(
                self.T[: self.disp_len],
                self.tYZoom[0].value() * self.disp_sig[0] + self.Offset[0].value(),
            )
        elif channel == 2:  # 通道2
            self.pw[0].curve[1].setData(
                self.T[: self.disp_len],
                self.tYZoom[1].value() * self.disp_sig[1] + self.Offset[1].value(),
            )
        else:  # XY
            self.pw[0].curve[2].setData(
                self.tYZoom[0].value() * self.disp_sig[0] + self.Offset[0].value(),
                self.tYZoom[1].value() * self.disp_sig[1] + self.Offset[1].value(),
            )

        if t1 != 0:
            dt = time.perf_counter() - t1
            self.t_t = (self.t_t * self.plt_cnt + dt) / (self.plt_cnt + 1)
            self.plt_cnt += 1
        # print("t1=%f" % t2)

    def update_plt_f(self):
        if self.RUNNING:
            t1 = time.perf_counter()
        else:
            t1 = 0

        channel = self.Channel.currentIndex()

        # 显示FFT
        if len(self.f) == len(self.disp_FFT[0]):  # 改变FFT点数时可能出现f更新但disp_FFT未更新的情况，故加此判断
            if channel == 1:
                self.pw[1].curve[0].setData(self.f, self.disp_FFT[0])
            elif channel == 2:
                self.pw[1].curve[1].setData(self.f, self.disp_FFT[1])
            else:  # XY模式和双通道模式
                self.pw[1].curve[0].setData(self.f, self.disp_FFT[0])
                self.pw[1].curve[1].setData(self.f, self.disp_FFT[1])

        if t1 != 0:
            dt = time.perf_counter() - t1
            self.t_f = (self.t_f * self.plt_cnt + dt) / (self.plt_cnt + 1)
            self.plt_cnt += 1

    def callback(self, in_data, frame_count, time_info, status):
        self.rec_bytes = in_data
        self.signal_received.emit()
        return (b"", paContinue)

    def trigger(self):
        tgcb = [self.TriggerCB[0].checkState(), self.TriggerCB[1].checkState()]
        self.disp_start = [0, 0]
        for i in [0, 1]:
            if tgcb[i]:
                self.disp_start[i] = find_first(
                    self.rec_y[i][: -self.disp_len] * self.tYZoom[i].value()
                    + self.Offset[i].value(),
                    self.Trigger[i].value(),
                    self.TriggerSlope[i].currentIndex(),
                    self.TriggerTH[i].value(),
                )
        if tgcb[0] and not tgcb[1]:
            self.disp_start[1] = self.disp_start[0]
        elif tgcb[1] and not tgcb[0]:
            self.disp_start[0] = self.disp_start[1]
        return self.disp_start

    def slot_received(self):
        self.rec_y[0], self.rec_y[1] = self.config.decode(self.rec_bytes)
        self.trigger()

        if self.RECORDING:
            self.record_data[0].append(self.rec_y[0])
            self.record_data[1].append(self.rec_y[1])

        self.update_disp_sig()
        self.update_disp_FFT()

    def run(self):
        pa = PyAudio()
        stream = pa.open(
            format=paInt16,
            channels=2,
            rate=self.config.RATE,
            input=True,
            stream_callback=self.callback,
            frames_per_buffer=self.config.CHUNK,
        )
        stream.start_stream()
        while self.RUNNING:
            self.delay()
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = scope()
    win.show()
    win.update_t_max()
    sys.exit(app.exec())
