# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from collections import deque
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from pyqtgraph import PlotCurveItem
from pyaudio import PyAudio, paInt16, paContinue

from Ui_timeSpectrum import Ui_timeSpectrum
from public import array_linear_map, Config


class timeSpectrum(Ui_timeSpectrum):
    signal_received_new_data = pyqtSignal()

    def __init__(self, parent=None, config=None):
        super().__init__(parent)

        if config:
            self.config = config
        else:
            self.config = Config("Audiono.ini")

        self.rec_bytes = bytes()
        self.fft_win = np.hanning(self.config.CHUNK)

        self.RUNNING = False

        self.nframes, self.fft_num = self.Frames.value(), self.fftN.value()
        self.fft_maxn = self.fft_num // 2 + 1
        self.f_max = self.config.RATE // 2
        self.fn = [0, self.fft_maxn // self.fZoom.value()]
        self.f = np.linspace(0, self.f_max, self.fft_maxn)

        data = -120 * np.ones((self.nframes, self.fft_maxn))
        self.fft_queue = [deque(data), deque(data)]
        self.img = 0
        self.heatmap.setImage(-120 * np.ones((self.fn[1] - self.fn[0], self.nframes)))
        self.hist.setHistogramRange(-120, 20)

        self.curves = [deque(), deque()]
        self.reset_curves()
        self.reset_axis_f()
        self.reset_axis_right()
        self.reset_surface()

        self.connect()

    def closeEvent(self, a0) -> None:
        if self.RUNNING:
            self.RUNNING = False
            self.RunButton.setState(True)
            self.delay()
            self.quit_thread()
        return super().closeEvent(a0)

    def connect(self):
        self.RunButton.clicked.connect(self.on_RunButton_clicked)

        for i in range(3):
            self.GridSize[i].valueChanged.connect(self.update_grid_size)
        self.GridSize[2].valueChanged.connect(self.reset_axis_right)

        self.Threshold[0].valueChanged.connect(self.reset_axis_right)
        self.Threshold[1].valueChanged.connect(self.reset_axis_right)

        for i in range(9):
            self.surface_cmap[i].valueChanged.connect(self.update_color_map)

        self.fZoom.valueChanged.connect(self.on_fZoom_valueChanged)
        self.fPos.valueChanged.connect(self.update_fn)
        self.fPos.valueChanged.connect(self.reset_axis_f)
        self.Frames.valueChanged.connect(self.update_nframes)
        self.fftN.valueChanged.connect(self.update_fft_num)
        self.WinType.currentIndexChanged.connect(self.update_win)

        self.signal_received_new_data.connect(self.slot_received_new_data)

    def update_win(self):
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

    def on_RunButton_clicked(self):
        self.RUNNING = not self.RUNNING
        if self.RUNNING:
            self.start_thread()
        else:
            self.quit_thread()

    def update_fft_num(self):
        self.fft_num = self.fftN.value()
        self.fft_maxn = self.fft_num // 2 + 1
        f = self.f.copy()
        self.f = np.linspace(0, self.f_max, self.fft_maxn)
        for i in [0, 1]:
            for j in range(len(self.fft_queue[i])):
                self.fft_queue[i][j] = np.interp(self.f, f, self.fft_queue[i][j])
        self.update_fn()
        self.reset_axis_f()

    def update_nframes(self):
        nframes = self.nframes
        self.nframes = self.Frames.value()
        if self.nframes > nframes:
            empty_line = -120 * np.ones(self.fft_maxn)
            for i in [0, 1]:
                for j in range(self.nframes - nframes):
                    self.fft_queue[i].appendleft(empty_line)
        elif self.nframes < nframes:
            for i in [0, 1]:
                for j in range(nframes - self.nframes):
                    self.fft_queue[i].popleft()
        self.reset_surface()
        self.reset_curves()

    def update_fn(self):
        n = self.fft_maxn // self.fZoom.value()
        start = int(self.fPos.value() * (self.fft_maxn - n))
        self.fn = [start, start + n]
        self.reset_surface()

    def reset_surface(self):
        LX, LY = self.grid_size[0] / 2, self.grid_size[1] / 2
        r, c = self.nframes, self.fn[1] - self.fn[0]
        cid = 1 - self.Channel.currentIndex()
        z = array_linear_map(
            np.array(self.fft_queue[cid])[:, self.fn[0] : self.fn[1]],
            [self.Threshold[0].value(), self.Threshold[1].value()],
            [-self.grid_size[2] / 2, self.grid_size[2] / 2],
        )
        self.surface.setData(
            x=np.linspace(-LX, LX, r),
            y=np.linspace(-LY, LY, c),
            z=z,
        )

    def reset_curves(self):
        n = len(self.plt[0].items)
        if n == 0:
            for i in [0, 1]:
                for j in range(self.nframes):
                    c = PlotCurveItem(pen="g", fillLevel=0, brush=(0, 0, 0, 255))
                    self.curves[i].append(c)
                    self.plt[i].addItem(c)
                    # c = self.plt[i].plot(pen="g", fillLevel=0, brush=(0, 0, 0, 255))
                    # self.curves[i].append(c)
        elif n < self.nframes:
            for i in [0, 1]:
                for j in range(self.nframes - n):
                    c = PlotCurveItem(pen="g", fillLevel=0, brush=(0, 0, 0, 255))
                    self.curves[i].appendleft(c)
                    self.plt[i].addItem(c)
                    # c = self.plt[i].plot(pen="g", fillLevel=0, brush=(0, 0, 0, 255))
                    # self.curves[i].appendleft(c)
                for c in self.curves[i]:
                    self.plt[i].removeItem(c)
                    self.plt[i].addItem(c)
        elif n > self.nframes:
            for i in [0, 1]:
                for j in range(n - self.nframes):
                    c = self.curves[i].popleft()
                    self.plt[i].removeItem(c)

        self.curve_pos_f = np.linspace(
            0, self.f_max / self.fZoom.value() / 5, self.nframes, endpoint=True
        )
        self.curve_pos_y = np.linspace(100, 0, self.nframes, endpoint=True)

        for i in range(self.nframes):
            self.curves[0][i].setPos(self.curve_pos_f[i], self.curve_pos_y[i])
            self.curves[1][i].setPos(self.curve_pos_f[i], self.curve_pos_y[i])

        self.reset_axis_t()

    def on_fZoom_valueChanged(self):
        self.update_fn()
        self.curve_pos_f = np.linspace(
            0, self.f_max / self.fZoom.value() / 5, self.nframes, endpoint=True
        )
        self.reset_axis_f()

    def reset_axis_f(self):
        f_range = self.f_max / self.fZoom.value()
        f_start = self.fPos.value() * (self.f_max - f_range)
        f_offset = f_range / 5
        x = f_offset + np.linspace(0, f_range, 10, endpoint=True)
        ticks = np.round(x + f_start - f_offset, 1)
        for i in [0, 1]:
            self.plt[i].setXRange(0, self.f_max / self.fZoom.value() * 6 / 5)
            ax = self.plt[i].getAxis("bottom")
            ax.setTicks(
                [[(0, "f/Hz")] + [(x[i], str(ticks[i])) for i in range(len(x))]]
            )

        x = np.linspace(0, self.fn[1] - self.fn[0], 10, endpoint=True)
        ticks = np.round(x / (self.fn[1] - self.fn[0]) * f_range + f_start, 1)
        ax = self.heatmap_plt.getAxis("bottom")
        ax.setTicks([[(x[i], str(ticks[i])) for i in range(len(x))]])

    def reset_axis_t(self):
        y = np.linspace(0, 100, 10, endpoint=True)
        factor = self.nframes / 100 * self.config.CHUNK / self.config.RATE
        ticks = np.round(factor * y, 2)
        for i in [0, 1]:
            self.plt[i].setYRange(0, self.grid_size[2] + 100)
            ax = self.plt[i].getAxis("left")
            ax.setTicks(
                [
                    [(y[i], str(ticks[i])) for i in range(len(y))]
                    + [(100 + self.grid_size[2], "t/s")]
                ]
            )

        y = np.linspace(0, self.nframes, 10, endpoint=True)
        factor = self.config.CHUNK / self.config.RATE
        ticks = np.round(factor * y, 2)
        ax = self.heatmap_plt.getAxis("left")
        ax.setTicks([[(y[i], str(ticks[i])) for i in range(len(y))]])

    def reset_axis_right(self):
        for i in [0, 1]:
            ax = self.plt[i].getAxis("right")
            v = [w.value() for w in self.Threshold]
            if v[0] < v[1]:
                ax.setTicks(
                    [
                        [
                            (0, str(v[0])),
                            (self.grid_size[2], str(v[1])),
                            (self.grid_size[2] + 10, "dB"),
                        ]
                    ]
                )
            else:
                ax.setTicks(
                    [
                        [
                            (0, "-120"),
                            (self.grid_size[2], "20"),
                            (self.grid_size[2] + 10, "dB"),
                        ]
                    ]
                )

    def update_color_map(self):
        color_map = [self.surface_cmap[i].value() for i in range(9)]
        self.surface.shader()["colorMap"] = np.array(color_map)

    def update_grid_size(self):
        X, Y, Z = [self.GridSize[i].value() for i in range(3)]
        self.gx.translate(self.grid_size[0] / 2, 0, 0)
        self.gy.translate(0, self.grid_size[1] / 2, 0)
        self.gz.translate(0, 0, self.grid_size[2] / 2)

        self.gx.setSize(Z, Y)
        self.gy.setSize(X, Z)
        self.gz.setSize(X, Y)

        self.gx.translate(-X / 2, 0, 0)
        self.gy.translate(0, -Y / 2, 0)
        self.gz.translate(0, 0, -Z / 2)

        self.grid_size = [X, Y, Z]

        self.surface.setData(
            x=np.linspace(-X / 2, X / 2, self.nframes),
            y=np.linspace(-Y / 2, Y / 2, self.fn[1] - self.fn[0]),
        )

    def gen_debug_data(self):
        if hasattr(self, "cnt"):
            self.cnt = (self.cnt + 1) % 100
        else:
            self.cnt = 0
        t = np.arange(self.config.CHUNK) / self.config.RATE
        amp, f = 10, 1000
        amp_n = amp / 10
        y1 = amp * np.sin(2 * np.pi * (f - self.cnt * 10) * t) + amp_n * np.random.rand(
            self.config.CHUNK
        )
        y2 = amp * np.sin(2 * np.pi * (f + self.cnt * 10) * t) + amp_n * np.random.rand(
            self.config.CHUNK
        )
        self.rec_bytes = self.config.encode(y1, y2)

    def callback(self, in_data, frame_count, time_info, status):
        data = bytes()
        self.rec_bytes = in_data
        # self.gen_debug_data()
        self.signal_received_new_data.emit()
        return data, paContinue

    def slot_received_new_data(self):
        self.update_fft_queue()
        cid = self.Channel.currentIndex()
        self.img = np.array(self.fft_queue[cid])[:, self.fn[0] : self.fn[1]]
        self.update_plts()

    def update_fft_queue(self):
        # 如果先乘后除，乘完后仍是int16，会截断
        data = self.config.decode(self.rec_bytes)
        for cid in [0, 1]:
            FFT = (
                np.fft.rfft(data[cid] * self.fft_win, n=self.fft_num)
                / self.config.CHUNK
            )
            FFT_dB = 20 * np.log10(np.abs(FFT) + 1e-6)
            self.fft_queue[cid].popleft()
            self.fft_queue[cid].append(FFT_dB)

    def update_plts(self):
        mode = self.Mode.currentIndex()
        if mode == 0:
            self.update_curves()
        elif mode == 1:
            self.update_heatmap()
        else:
            self.update_surface()

    def update_curves(self):
        for i in [0, 1]:
            c = self.curves[i].popleft()
            self.plt[i].removeItem(c)
            self.plt[i].addItem(c)
            new_line = array_linear_map(
                self.fft_queue[i][-1][self.fn[0] : self.fn[1]],
                [self.Threshold[0].value(), self.Threshold[1].value()],
                [0, self.grid_size[2]],
            )
            c.setData(x=self.f[0 : self.fn[1] - self.fn[0]], y=new_line)
            self.curves[i].append(c)

        for i in range(self.nframes):
            self.curves[0][i].setPos(self.curve_pos_f[i], self.curve_pos_y[i])
            self.curves[1][i].setPos(self.curve_pos_f[i], self.curve_pos_y[i])

    def update_heatmap(self):
        img = self.img
        if self.Threshold[0].value() < self.Threshold[1].value():
            img = np.clip(
                self.img, self.Threshold[0].value(), self.Threshold[1].value()
            )
        self.heatmap.setImage(img[::-1, :].T)

    def update_surface(self):
        img = array_linear_map(
            self.img,
            [self.Threshold[0].value(), self.Threshold[1].value()],
            [-self.grid_size[2] / 2, self.grid_size[2] / 2],
        )
        self.surface.setData(z=img)

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
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()  # 关闭
        pa.terminate()  # 终结


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = timeSpectrum()
    win.show()
    sys.exit(app.exec_())
