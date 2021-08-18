# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
from PyQt5.QtWidgets import QApplication
from pyaudio import PyAudio, paContinue, paInt16

from Ui_calibrator import Ui_calibrator
from public import Config


def get_signal_params(y, N, T0, t=2):
    start = np.argmax(y > t)
    y0 = y[start : start + N]
    T1 = (N - T0) // 2
    p0 = T1 // 10
    L0 = 8 * T1 // 10
    vmax = np.mean(y0[p0 : p0 + L0])
    p1 = N - T1 // 10
    vmin = np.mean(y0[p1 - L0 : p1])
    zero = np.argmin(np.abs(y0))
    return vmax - vmin, zero


class calibrator(Ui_calibrator):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)

        if config:
            self.config = config
        else:
            self.config = Config("Audiono.ini")

        self.rec_bytes = b""

        # t = np.arange(self.config.CHUNK)
        # y = (t < self.config.CHUNK / 2) / 2
        self.y = np.zeros(2 * self.config.CHUNK)
        T0 = 6000
        T1 = (len(self.y) - T0) // 2
        self.y[:T1] = 0.5 * 32768
        self.y[-T1:] = -0.5 * 32768
        self.y[T1 : T1 + T0] = np.arange(3000, -3000, -1)
        self.play_from_zero = True
        self.play_bytes = []
        self.play_bytes.append(
            np.repeat(-self.y[: self.config.CHUNK], 2).astype(np.int16).tobytes()
        )
        self.play_bytes.append(
            np.repeat(-self.y[self.config.CHUNK :], 2).astype(np.int16).tobytes()
        )
        self.connect()

    def connect(self):
        self.SaveBtn.clicked.connect(self.on_SaveBtn_clicked)
        self.Calibrate[0].clicked.connect(self.calibrate_input_offset)
        self.Calibrate[1].clicked.connect(self.calibrate_input_gain)
        self.Calibrate[2].clicked.connect(self.calibrate_output)

    def on_SaveBtn_clicked(self):
        self.config.save("Audiono.ini")
        self.inform("保存成功")

    def update_input_offset(self, left, right):
        self.config.input_offset[0] = left
        self.config.input_offset[1] = right

    def update_input_gain(self, left, right):
        self.config.input_gain[0] = left
        self.config.input_gain[1] = right

    def update_output_gain(self, left, right):
        self.config.output_gain[0] = left
        self.config.output_gain[1] = right

    def update_output_offset(self, left, right):
        self.config.output_offset[0] = left
        self.config.output_offset[1] = right

    def callback_i(self, in_data, frame_count, time_info, status):
        self.rec_bytes = in_data
        return (b"", paContinue)

    def calibrate_input_offset(self):
        """
        两输入接地
        """
        pa = PyAudio()
        stream = pa.open(
            format=paInt16,
            channels=2,
            rate=self.config.RATE,
            input=True,
            stream_callback=self.callback_i,
            frames_per_buffer=self.config.CHUNK,
        )

        stream.start_stream()

        rec_y = [[], []]
        cnt = 0
        while cnt < 5:
            if self.rec_bytes:
                cnt += 1
                data = np.frombuffer(self.rec_bytes, dtype=np.int16) / 32768
                rec_y[0].append(data[1::2])
                rec_y[1].append(data[::2])
                self.rec_bytes = b""
            else:
                time.sleep(self.config.CHUNK / self.config.RATE)
        stream.stop_stream()
        stream.close()
        pa.terminate()

        rec_y[0] = np.array(rec_y[0]).flatten()
        rec_y[1] = np.array(rec_y[1]).flatten()
        # np.save("test/shift/left_input_offset.npy", rec_y[0])
        # np.save("test/shift/right_input_offset.npy", rec_y[1])
        self.curve[0].setData(rec_y[0])
        self.curve[1].setData(rec_y[1])
        self.update_input_offset(np.mean(rec_y[0]), np.mean(rec_y[1]))
        self.inform("输入偏移校准成功")

    def calibrate_input_gain(self):
        """
        两输入接5V
        """
        pa = PyAudio()
        stream = pa.open(
            format=paInt16,
            channels=2,
            rate=self.config.RATE,
            input=True,
            stream_callback=self.callback_i,
            frames_per_buffer=self.config.CHUNK,
        )

        stream.start_stream()

        rec_y = [[], []]
        cnt = 0
        while cnt < 5:
            if self.rec_bytes:
                cnt += 1
                data = np.frombuffer(self.rec_bytes, dtype=np.int16) / 32768
                rec_y[0].append(data[1::2] - self.config.input_offset[0])
                rec_y[1].append(data[::2] - self.config.input_offset[1])
                self.rec_bytes = b""
            else:
                time.sleep(self.config.CHUNK / self.config.RATE)
        stream.stop_stream()
        stream.close()
        pa.terminate()

        rec_y[0] = np.array(rec_y[0]).flatten()
        rec_y[1] = np.array(rec_y[1]).flatten()

        # np.save("test/shift/left_input_gain.npy", rec_y[0])
        # np.save("test/shift/right_input_gain.npy", rec_y[1])
        idx0, idx1 = np.argmax(rec_y[0] > 0.1) + 20, np.argmax(rec_y[1] > 0.1) + 20

        self.curve[0].setData(rec_y[0])
        self.curve[1].setData(rec_y[1])
        if rec_y[0][idx0] > 0.1 and rec_y[1][idx1] > 0.1:
            self.update_input_gain(
                5 / np.mean(rec_y[0][idx0:]), 5 / np.mean(rec_y[1][idx1:])
            )
            self.inform("输入增益校准成功")
        else:
            self.critical("校准失败，请检查连线后重试")

    def callback_io(self, in_data, frame_count, time_info, status):
        data = self.play_bytes[0] if self.play_from_zero else self.play_bytes[1]
        self.play_from_zero = not self.play_from_zero
        self.rec_bytes = in_data
        return (data, paContinue)

    def calibrate_output(self):
        """
        输入1接输出1，输入2接输出2
        """
        pa = PyAudio()
        stream = pa.open(
            format=paInt16,
            channels=2,
            rate=self.config.RATE,
            input=True,
            output=True,
            stream_callback=self.callback_io,
            frames_per_buffer=self.config.CHUNK,
        )

        stream.start_stream()

        rec_y = [[], []]
        cnt = 0
        while cnt < 10:
            if self.rec_bytes:
                cnt += 1
                y0, y1 = self.config.decode(self.rec_bytes)
                rec_y[0].append(y0)
                rec_y[1].append(y1)
                self.rec_bytes = b""
            else:
                time.sleep(self.config.CHUNK / self.config.RATE)
        stream.stop_stream()
        stream.close()
        pa.terminate()

        rec_y[0] = np.array(rec_y[0]).flatten()
        rec_y[1] = np.array(rec_y[1]).flatten()

        # debug
        # np.save("test/shift/left.npy", rec_y[0])
        # np.save("test/shift/right.npy", rec_y[1])
        # end debug
        self.curve[0].setData(rec_y[0])
        self.curve[1].setData(rec_y[1])

        if np.max(rec_y[0]) > 1:

            vpp, zero = get_signal_params(rec_y[0], len(self.y), 6000)
            a0, b0 = 1 / vpp, self.y[zero] / 32768

            vpp, zero = get_signal_params(rec_y[1], len(self.y), 6000)
            a1, b1 = 1 / vpp, self.y[zero] / 32768

            self.update_output_gain(a0, a1)
            self.update_output_offset(b0, b1)
            self.inform("输出偏移与增益校准成功")
        else:
            self.critical("校准失败，请检查连线后重试")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = calibrator()
    ui.show()
    sys.exit(app.exec_())
