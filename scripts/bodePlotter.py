# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QFileDialog
import pyqtgraph.exporters as pg_exporter
from pyaudio import PyAudio, paInt16, paContinue

from Ui_bodePlotter import Ui_bodePlotter
from cal_response import get_chunk, get_response
from public import save_data, linear_map, worker


# class for the application window
class bodePlotter(Ui_bodePlotter):
    def __init__(self, parent=None, rate=48000):
        super().__init__(parent, rate)

        self.CHUNK = 1024
        self.FORMAT = paInt16
        self.CHANNELS = 2
        self.RATE = rate
        self.play_bytes = b""
        self.rec_bytes = bytes()

        self.f = np.array([])
        self.A = np.array([])
        self.phi = np.array([])

        self.INTERRUPT = False
        self.MEASURING = False
        self.UPDATE = False

        self.MEASURE_TYPE = 0  # 0表示测量新频点，1表示测量选中的已有频点

        self.f_log = False
        self.f_lim = [0, self.RATE / 2]
        self.A_lim = [-120, 0]
        self.phi_lim = [0, 360]

        self.AX = [0, 1]
        self.AY = [0, 1]
        self.PX = [0, 1]
        self.PY = [0, 1]

        self.worker = worker(win=self)
        self.process_thread = QThread()

        self.worker.moveToThread(self.process_thread)

        self.connect()

        # self.f = np.array([0, 5.1, 100, 1234.666, 5000, 10000.0001])
        # self.A = np.array([0.05, 0.1, 0.2, 0.67, 0.8888, 0.90])
        # self.phi = np.array([90, 90, 80, 77.777, 60.54321, 20])
        # self.update_result()

    def connect(self):
        self.process_thread.started.connect(self.worker.run)

        self.ResetButton.clicked.connect(self.on_ResetButton_clicked)
        self.StartButton[0].clicked.connect(lambda: self.start_measuring(0))
        self.StartButton[1].clicked.connect(lambda: self.start_measuring(1))
        self.DeleteButton.clicked.connect(self.on_DeleteButton_clicked)

        self.PointId.valueChanged.connect(self.update_current_point)
        self.ShowArrow.stateChanged.connect(self.on_ShowArrow_stateChanged)
        self.fLogCB.stateChanged.connect(self.on_fLogCB_stateChanged)
        self.fLogCB.stateChanged.connect(self.update_current_point)
        self.ALogCB.stateChanged.connect(self.on_ALogCB_stateChanged)
        self.ALogCB.stateChanged.connect(self.update_current_point)
        self.fPos.valueChanged.connect(self.update_f_lim)
        self.fZoom.valueChanged.connect(self.update_f_lim)
        self.CursorCB[0].stateChanged.connect(self.redraw_cursor)
        self.CursorCB[1].stateChanged.connect(self.redraw_cursor)
        self.CursorX[0].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorX[1].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorY[0].valueChanged.connect(self.on_cursor_valueChanged)
        self.CursorY[1].valueChanged.connect(self.on_cursor_valueChanged)

        self.DataDispCB.stateChanged.connect(self.update_disp_data)

        self.PicSave.clicked.connect(self.on_PicSave_clicked)
        self.DataSave.clicked.connect(self.on_DataSave_clicked)

    def closeEvent(self, event):
        if self.MEASURING:
            self.MEASURING = False
            self.process_thread.quit()
        super().closeEvent(event)

    def save_pic(self):
        if self.SaveSrc.currentIndex() == 0:
            ex = pg_exporter.ImageExporter(self.pwt.scene())
        else:
            ex = pg_exporter.ImageExporter(self.pwf.scene())
        filename = QFileDialog.getSaveFileName(
            self, caption="窗口1图片保存为", filter="PNG (*.png);;JPEG (*.jpg)"
        )
        if filename[0]:
            ex.export(filename[0])

    def on_PicSave_clicked(self):
        if not self.MEASURING:
            self.save_pic()
        else:
            self.critical("正在测量中，无法保存")

    def on_DataSave_clicked(self):
        src = self.SaveSrc.currentIndex()
        if src == 0:
            data_to_save = np.array([self.f, self.A]).T
        else:
            data_to_save = np.array([self.f, self.phi]).T
        filename = QFileDialog.getSaveFileName(
            self,
            caption="保存为",
            filter="MATLAB (*.mat);;Python (*.npy);;TXT (*.txt);;Excel (*.xlsx)",
        )
        if filename[0]:
            if not save_data(filename[0], data_to_save):
                self.critical("保存文件失败")

    def on_cursor_valueChanged(self):
        self.update_cursor_XY()
        self.redraw_cursor()
        self.update_disp_data()

    def update_cursor_XY(self):
        # 光标作用于幅频
        if self.CursorTarget.currentIndex() == 0:
            self.AX = [c.value() for c in self.CursorX]
            self.AY = [c.value() for c in self.CursorY]
        # 光标作用于相频
        else:
            self.PX = [c.value() for c in self.CursorX]
            self.PY = [c.value() for c in self.CursorY]

    def redraw_cursor(self):
        """
        重绘光标
        """
        if self.CursorCB[0].checkState():
            X = linear_map(self.AX, self.f_lim)
            Y = linear_map(self.AY, self.A_lim)
            self.pw[0].set_cursors(X, Y)
        else:
            self.pw[0].set_cursors()
        if self.CursorCB[1].checkState():
            X = linear_map(self.PX, self.f_lim)
            Y = linear_map(self.PY, self.phi_lim)
            self.pw[1].set_cursors(X, Y)
        else:
            self.pw[1].set_cursors()

    def update_disp_data(self):
        """
        更新测量数据
        """
        if not self.DataDispCB.checkState():
            return

        X = linear_map(self.AX, self.f_lim)
        Y = linear_map(self.AY, self.A_lim)
        if self.fLogCB.checkState() and max(X[0], X[1]) < 10:
            X = [10 ** x for x in X]
        if self.ALogCB.checkState():
            self.DataDispGrid[0].update_data(
                [X[0], X[1], Y[0], Y[1], X[1] - X[0], Y[1] - Y[0]]
            )
        else:
            if Y[0] == 0:
                self.DataDispGrid[0].update_data(
                    [X[0], X[1], Y[0], Y[1], X[1] - X[0], "Inf"]
                )
            else:
                self.DataDispGrid[0].update_data(
                    [X[0], X[1], Y[0], Y[1], X[1] - X[0], Y[1] / Y[0]]
                )
        X = linear_map(self.PX, self.f_lim)
        Y = linear_map(self.PY, self.phi_lim)
        if self.fLogCB.checkState() and max(X[0], X[1]) < 10:
            X = [10 ** x for x in X]
        self.DataDispGrid[1].update_data(
            [X[0], X[1], Y[0], Y[1], X[1] - X[0], Y[1] - Y[0]]
        )

    def start_measuring(self, type):
        if not self.MEASURING:
            self.MEASURING = True
            self.MEASURE_TYPE = type
            self.process_thread.start()
        else:
            self.INTERRUPT = True

    def on_ResetButton_clicked(self):
        if not self.MEASURING:
            self.f = np.array([])
            self.A = np.array([])
            self.phi = np.array([])
            self.PointId.setMaximum(0)
            self.PointValue.clear()
            self.pw[0].scatter.setData()
            self.pw[1].scatter.setData()
            self.update_f_lim()
            self.update_A_lim()
            self.update_phi_lim()

    def on_ShowArrow_stateChanged(self):
        if self.ShowArrow.checkState():
            self.pw[0].show_arrow(True)
            self.pw[1].show_arrow(True)
        else:
            self.pw[0].show_arrow(False)
            self.pw[1].show_arrow(False)

    def on_DeleteButton_clicked(self):
        if len(self.f) == 0:
            self.critical("已测量频点为空")
            return
        Id = self.PointId.value()
        self.f = np.delete(self.f, Id)
        self.A = np.delete(self.A, Id)
        self.phi = np.delete(self.phi, Id)
        self.update_result()
        if len(self.f) == 0:
            self.PointValue.clear()

    def update_current_point(self):
        if len(self.f):
            Id = self.PointId.value()
            self.PointValue.setText(
                "%.2fHz  %.2fdB  %.2f°"
                % (self.f[Id], 20 * np.log10(self.A[Id] + 1e-6), self.phi[Id])
            )
            f = np.log10(self.f[Id]) if self.fLogCB.checkState() else self.f[Id]
            A = (
                20 * np.log10(self.A[Id] + 1e-6)
                if self.ALogCB.checkState()
                else self.A[Id]
            )
            self.pw[0].arrow.setPos(f, A)
            self.pw[1].arrow.setPos(f, self.phi[Id])

    def update_f_lim(self):
        if len(self.f):
            if self.f[0] == 0:
                self.f[0] = 1
        if len(self.f) > 2:
            f_min, f_max = np.min(self.f), np.max(self.f)
            f_range = (f_max - f_min) / self.fZoom.value()
            self.f_lim[0] = self.fPos.value() * (f_max - f_range - f_min) + f_min
            self.f_lim[1] = self.f_lim[0] + f_range
        else:
            self.f_lim = [0, self.RATE / 2]
        if self.fLogCB.checkState():
            self.f_lim[0] = np.log10(self.f_lim[0]) if self.f_lim[0] > 0 else 0
            self.f_lim[1] = np.log10(self.f_lim[1])
        self.pw[0].setXRange(self.f_lim[0], self.f_lim[1])
        self.pw[1].setXRange(self.f_lim[0], self.f_lim[1])
        self.redraw_cursor()
        self.update_disp_data()

    def update_A_lim(self):
        if len(self.f) > 2:
            self.A_lim = [np.min(self.A), np.max(self.A)]
        else:
            self.A_lim = [0, 1]
        if self.ALogCB.checkState():
            self.A_lim = [20 * np.log10(x + 1e-6) for x in self.A_lim]
        self.pw[0].setYRange(self.A_lim[0], self.A_lim[1])
        self.redraw_cursor()
        self.update_disp_data()

    def update_phi_lim(self):
        if len(self.f) > 2:
            self.phi_lim = [np.min(self.phi), np.max(self.phi)]
        else:
            self.phi_lim = [0, 360]
        self.pw[1].setYRange(self.phi_lim[0], self.phi_lim[1])
        self.redraw_cursor()
        self.update_disp_data()

    def update_result(self):
        self.PointId.setRange(0, len(self.f) - 1)
        self.update_f_lim()
        self.update_A_lim()
        self.update_phi_lim()
        self.plot_fig()
        self.update_current_point()

    def on_ALogCB_stateChanged(self):
        if self.ALogCB.checkState():
            if len(self.f):
                self.pw[0].scatter.setData(self.f, 20 * np.log10(self.A + 1e-6))
            self.DataDispGrid[0].update_labels(self.data_labels[2])
        else:
            if len(self.f):
                self.pw[0].scatter.setData(self.f, self.A)
            self.DataDispGrid[0].update_labels(self.data_labels[0])
        self.update_A_lim()
        self.redraw_cursor()
        self.update_disp_data()

    def on_fLogCB_stateChanged(self):
        if self.fLogCB.checkState():
            self.pw[0].scatter.setLogMode(True, False)
            self.pw[1].scatter.setLogMode(True, False)
        else:
            self.pw[0].scatter.setLogMode(False, False)
            self.pw[1].scatter.setLogMode(False, False)
        self.update_f_lim()
        self.redraw_cursor()
        self.update_disp_data()

    def plot_fig(self):
        if not self.MEASURING:
            if len(self.f):
                if self.ALogCB.checkState():
                    self.pw[0].scatter.setData(self.f, 20 * np.log10(self.A + 1e-6))
                else:
                    self.pw[0].scatter.setData(self.f, self.A)
                self.pw[1].scatter.setData(self.f, self.phi)
            else:
                self.pw[0].scatter.setData()
                self.pw[1].scatter.setData()

    def callback(self, in_data, frame_count, time_info, status):
        data = self.play_bytes
        self.UPDATE = True
        self.rec_bytes += in_data
        return (data, paContinue)

    def sine_bytes(self, f, tn, cnt):
        y = 0.5 * np.sin(2 * np.pi * f * ((tn + self.CHUNK * cnt) / self.RATE))
        y = np.repeat(y, 2)
        return (-y * 32768).astype(np.int16).tobytes()

    def quit_thread(self):
        self.MEASURING = False
        self.process_thread.quit()
        self.StartButton[self.MEASURE_TYPE].setState(True)

    def run(self):
        if self.MEASURE_TYPE == 0:
            fLeft, fRight, fNum = (
                self.fLeft.value(),
                self.fRight.value(),
                int(self.fNum.value()),
            )
            if fNum == 1:
                f_arr = np.array([fLeft])
            else:
                if fLeft >= fRight:
                    self.critical("起始频率应小于终止频率，请检查输入后重试")
                    self.quit_thread()
                    return
                f_arr = np.linspace(fLeft, fRight, fNum, endpoint=True)
        else:
            fNum = 1
            if len(self.f):
                f_arr = np.array([self.f[self.PointId.value()]])
            else:
                self.critical("已测量频点为空，本功能暂无法使用")
                self.quit_thread()
                return
        if f_arr[0] < 5 and f_arr[0] != 0:
            self.critical("最小频率应为 0 或在 [5, %d] 区间内，请检查输入后重试" % (self.RATE // 2))
            self.quit_thread()
            return

        resp = dict(zip(self.f, zip(self.A, self.phi)))

        self.show_message("开始测量")
        MeasureException = False
        fail_list = []
        pa = PyAudio()
        for fid, f in enumerate(f_arr):
            self.show_message("测量中：" + str(fid + 1) + "/" + str(len(f_arr)))
            pos_cnt = 0
            self.rec_bytes = bytes()
            self.CHUNK = get_chunk(f, self.RATE)
            tn = np.arange(self.CHUNK)
            self.play_bytes = self.sine_bytes(f, tn, 0)
            cnt = 1

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

            start_t = time.perf_counter()
            L_reg = 0
            while pos_cnt < 2:
                if self.INTERRUPT:
                    break

                if self.UPDATE:
                    self.play_bytes = self.sine_bytes(f, tn, cnt)
                    cnt += 1
                    self.UPDATE = False

                rec_bytes_copy = self.rec_bytes
                rec_data = (
                    np.fromstring(rec_bytes_copy[-self.CHUNK :], dtype=np.int16) / 32768
                )
                if L_reg != len(rec_bytes_copy) and np.max(rec_data) > 0.02:
                    L_reg = len(rec_bytes_copy)
                    pos_cnt += 1
                if time.perf_counter() - start_t > 5:
                    MeasureException = True
                    break

            stream.stop_stream()
            stream.close()  # 关闭

            if self.INTERRUPT:
                undo_list = f_arr[fid:]
                break

            if MeasureException:
                self.show_message("%.2fHz超时" % f)
                time.sleep(1)
                MeasureException = False
                fail_list.append(f)
                continue

            rec_data = np.fromstring(self.rec_bytes, dtype=np.int16) / 32768
            y0 = rec_data[1::2]
            y1 = rec_data[::2]
            # np.save("bp_rec_test/y_%d.npy" % f, np.array([y0, y1]))
            A, phi = get_response(f, self.RATE, self.CHUNK, y0, y1)
            resp[f] = (A, phi)

        resp_list = []
        for f in resp.keys():
            resp_list.append((f, resp[f][0], resp[f][1]))
        if resp_list:
            resp_arr = np.array(resp_list)
            resp_arr = resp_arr[np.argsort(resp_arr[:, 0])]
            self.f, self.A, self.phi = resp_arr[:, 0], resp_arr[:, 1], resp_arr[:, 2]
            if self.f[0] == 0 and len(self.f) > 1:
                if abs(self.phi[1]) < 10:
                    self.phi[0] = 0
                elif 0 < 90 - self.phi[1] < 10:
                    self.phi[0] = 90
                else:
                    self.phi[0] = self.phi[1]

        self.MEASURING = False

        self.update_result()

        pa.terminate()  # 终结

        if self.INTERRUPT:
            self.show_message("测量中断")
            if fail_list:
                self.warn(
                    str(np.round(fail_list, 2))
                    + " Hz\n处测量失败，请检查连线是否正确\n"
                    + str(np.round(undo_list, 2))
                    + " Hz\n未测量"
                )
            else:
                self.warn(str(np.round(undo_list, 2)) + " Hz\n未测量")
            self.INTERRUPT = False
        else:
            self.show_message("测量完成")
            if fail_list:
                self.warn(str(np.round(fail_list, 2)) + " Hz\n处测量失败，请检查连线是否正确")
        self.process_thread.quit()
        self.StartButton[self.MEASURE_TYPE].setState(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = bodePlotter()
    win.show()
    sys.exit(app.exec())
