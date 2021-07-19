# -*- coding: utf-8 -*-
import numpy as np
import time
import wave
from scipy import signal
from scipy.io import savemat, loadmat
import pandas as pd
from PyQt5.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QDoubleSpinBox,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal, QThread

# from PyQt5.QtGui import QColor


def linear_map(v, lim):
    """
    v:0~1
    lim[1]>lim[0]
    """
    return [v[0] * (lim[1] - lim[0]) + lim[0], v[1] * (lim[1] - lim[0]) + lim[0]]


def array_linear_map(arr, v, lim, lim1=[-120, 20]):
    """
    将arr从 [v[0],v[1]] 映射到 [lim[0],lim[1]]
    超出 [v[0],v[1]] 的部分将截取
    v[0]>=v[1] 时，按照lim1归一化
    """
    if v[0] >= v[1]:
        v = lim1
    tmp = (arr - v[0]) / (v[1] - v[0])
    tmp = np.clip(tmp, 0, 1)
    return tmp * (lim[1] - lim[0]) + lim[0]


def find_first(arr, t, slope=0, N=5):
    if len(arr) == 0:
        return 0
    grad = (arr > t) * 1
    grad[1:] -= grad[:-1]
    grad[0] = 0
    if slope == 0:
        idx = np.where(grad == 1)
        for i in idx[0]:
            if (arr[i : i + N] > t).all():
                return i
        return 0
    else:
        idx = np.where(grad == -1)
        for i in idx[0]:
            if (arr[i : i + N] < t).all():
                return i
        return 0


class worker(QObject):
    def __init__(self, parent=None, win=None):
        super(worker, self).__init__(parent=parent)
        self.win = win

    def run(self):
        self.win.run()


class BasicWindow(QMainWindow):
    signal_messagebox_active = pyqtSignal(int, str)

    def __init__(
        self,
        parent=None,
        hasStatusBar=False,
        statusShowTime=10,
        hasThread=False,
        threadDelay=0.1,
    ):
        super().__init__(parent)
        self.signal_messagebox_active.connect(self.slot_messagebox)

        if hasStatusBar:
            self.statusbar = QStatusBar()
            self.setStatusBar(self.statusbar)
            if statusShowTime:
                self.statusShowTime = 1000 * statusShowTime
                self.timer_status = QTimer()
                self.timer_status.timeout.connect(self.clear_status)
                self.statusbar.messageChanged.connect(self.update_status_time)

        if hasThread:
            self.worker = worker(win=self)
            self.process_thread = QThread()
            self.worker.moveToThread(self.process_thread)
            self.process_thread.started.connect(self.worker.run)
            self.thread_delay = threadDelay

    def delay(self):
        time.sleep(self.thread_delay)

    def start_thread(self):
        self.process_thread.start()

    def quit_thread(self):
        self.process_thread.quit()

    def messagebox(self, type, s):
        self.signal_messagebox_active.emit(type, s)

    def warn(self, s):
        self.signal_messagebox_active.emit(2, s)

    def inform(self, s):
        self.signal_messagebox_active.emit(0, s)

    def critical(self, s):
        self.signal_messagebox_active.emit(3, s)

    def slot_messagebox(self, type, s):
        if type == 0:
            QMessageBox.information(self, "信息", s)
        elif type == 1:
            QMessageBox.question(self, "问题", s)
        elif type == 2:
            QMessageBox.warning(self, "警告", s)
        elif type == 3:
            QMessageBox.critical(self, "错误", s)
        elif type == 4:
            QMessageBox.about(self, "关于", s)

    def show_message(self, s):
        self.statusbar.showMessage(s)

    def update_status_time(self):
        self.timer_status.start(self.statusShowTime)

    def clear_status(self):
        self.statusbar.clearMessage()
        self.timer_status.stop()


class DataGrid(QWidget):
    def __init__(self, parent=None, labels=[], data=None, digits=2):
        super().__init__(parent)
        self.labels = []
        self.data = []
        self.digits = digits
        self.layout = QVBoxLayout(self)
        if data is None:
            data = [0 for i in labels]
        for i in range(len(labels)):
            self.labels.append(QLabel(labels[i]))
            self.layout.addWidget(self.labels[-1])
            self.data.append(
                QLabel(str(round(data[i], digits)), alignment=Qt.AlignRight)
            )
            self.layout.addWidget(self.data[-1])

    def set_label(self, id, text):
        self.labels[id].setText(text)

    def set_data(self, id, data):
        if isinstance(data, str):
            self.data[id].setText(data)
        else:
            self.data[id].setText(str(round(data, self.digits)))

    def update_labels(self, label_list):
        for i, label in enumerate(label_list):
            self.labels[i].setText(label)
        for j in range(i + 1, len(self.labels)):
            self.labels[j].clear()
        for item in self.data:
            item.clear()

    def update_data(self, data_list):
        for i, data in enumerate(data_list):
            if isinstance(data, str):
                self.data[i].setText(data)
            else:
                self.data[i].setText(str(round(data, self.digits)))

    def set_digits(self, digits):
        self.digits = digits


class logSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super(logSpinBox, self).__init__(parent)
        self.__vals = []

    def stepBy(self, steps):
        if self.__vals:
            self.setValue(self.value())  # 用户输入后，更新idx
            if self.value() not in self.__vals and steps < 0:
                steps += 1
            if 0 <= self.__idx + steps < len(self.__vals):
                self.__idx += steps
                super(logSpinBox, self).setValue(self.__vals[self.__idx])
        else:
            super(logSpinBox, self).stepBy(steps)

    def setSingleStep(self, step):
        if step > 1:
            super(logSpinBox, self).setSingleStep(step)
            self.refreshVals()

    def setCenterValue(self, val):
        if self.minimum() <= val <= self.maximum():
            self.__center_val = val
            self.refreshVals()

    def setParameters(self, mi, ma, val, step, decimal):
        mi = round(mi, decimal)
        ma = round(ma, decimal)
        if 0 < mi <= val <= ma:
            super(logSpinBox, self).setMinimum(mi)
            super(logSpinBox, self).setMaximum(ma)
            self.__center_val = val
            if step > 1:
                super(logSpinBox, self).setSingleStep(step)
            self.setDecimals(decimal)
            self.refreshVals()
            super(logSpinBox, self).setValue(val)

    def refreshVals(self):
        self.__vals = [self.__center_val]
        self.__idx = 0
        step = self.singleStep()
        decimal = self.decimals()
        mi, ma = self.minimum() * step, self.maximum() / step

        tmp = self.__vals[0]
        while tmp > mi:
            self.__vals.insert(0, round(tmp / step, decimal))
            self.__idx += 1
            tmp /= step
        self.__vals.insert(0, self.minimum())
        self.__idx += 1

        tmp = self.__vals[-1]
        while tmp < ma:
            self.__vals.append(round(tmp * step, decimal))
            tmp *= step
        self.__vals.append(self.maximum())

    def setMinimum(self, mi):
        if 0 < mi <= self.__center_val <= self.maximum():
            super(logSpinBox, self).setMinimum(mi)
            self.refreshVals()

    def setMaximum(self, ma):
        if ma >= self.__center_val >= self.minimum():
            super(logSpinBox, self).setMaximum(ma)
            self.refreshVals()

    def setValue(self, val):
        if self.minimum() <= val <= self.maximum():
            super(logSpinBox, self).setValue(val)
            if val == self.maximum():
                self.__idx = len(self.__vals) - 1
            else:
                for i in range(len(self.__vals) - 1):
                    if self.__vals[i] <= val < self.__vals[i + 1]:
                        self.__idx = i
                        break

    def value(self):
        val = super(logSpinBox, self).value()
        if self.decimals() == 0:
            return int(val)
        return val


class doubleSlider(QSlider):
    def __init__(self, parent=None, num=10000):
        super(doubleSlider, self).__init__(parent)
        self.setParameters(0, 1, num)

    def setParameters(self, mi, ma, num):
        self.__mi, self.__ma = mi, ma
        super(doubleSlider, self).setMaximum(num)
        self.__step = (ma - mi) / num

    def value(self):
        return self.__mi + super(doubleSlider, self).value() * self.__step

    def setValue(self, val):
        if self.__mi <= val <= self.__ma:
            super(doubleSlider, self).setValue(int((val - self.__mi) / self.__step))

    def maximum(self):
        return self.__ma

    def setMaximum(self, ma):
        if ma >= self.__mi:
            self.__ma = ma
            self.__step = (self.__ma - self.__mi) / (
                super(doubleSlider, self).maximum()
            )

    def minimum(self):
        return self.__mi

    def setMinimum(self, mi):
        if mi <= self.__ma:
            self.__mi = mi
            self.__step = (self.__ma - self.__mi) / (
                super(doubleSlider, self).maximum()
            )

    def setRange(self, mi, ma):
        if ma >= mi:
            self.__ma, self.__mi = ma, mi
            self.__step = (self.__ma - self.__mi) / (
                super(doubleSlider, self).maximum()
            )


# class Splitter(QSplitter):
#     def __init__(self, parent=None, orientation=Qt.Horizontal):
#         super().__init__(orientation, parent)
#         # palette = self.palette()
#         # role = self.backgroundRole()
#         # palette.setColor(role, QColor("#CFCFCF"))
#         # self.setPalette(palette)
#         # self.setFrameShape(QFrame.Box)
#         # self.setLineWidth(1)
#         self.setHandleWidth(1)
#         self.setColor("red")

#     def setColor(self, color):
#         palette = self.palette()
#         role = self.backgroundRole()
#         palette.setColor(role, QColor(color))
#         self.setPalette(palette)


class WidgetWithSplitter(QWidget):
    def __init__(self, parent=None, orientation=Qt.Horizontal, rect=[], sizes=[]):
        super().__init__(parent=parent)
        if rect:
            self.setGeometry(rect[0], rect[1], rect[2], rect[3])
        self.grid = QGridLayout(self)
        self.splitter = QSplitter(self)

        self.splitter.setOrientation(orientation)

        self.leftWidget = QWidget(self.splitter)
        self.rightWidget = QWidget(self.splitter)
        if sizes:
            self.splitter.setSizes(sizes)

        self.leftGrid = ComboGridLayout(self.leftWidget)
        self.rightGrid = ComboGridLayout(self.rightWidget)

        self.grid.addWidget(self.splitter)

    def getGrid(self, left=0):
        return self.rightGrid if left else self.leftGrid


class ComboGridLayout(QGridLayout):
    def __init__(self, parent=None, row=[], col=[], rowSpan=[], colSpan=[]):
        super().__init__(parent)
        # 记录每个layout中当前插入的行/列
        self.qgbs = [QGroupBox() for i in row]
        for i, qgb in enumerate(self.qgbs):
            super().addWidget(qgb, row[i], col[i], rowSpan[i], colSpan[i])

        self.grids = [QGridLayout(qgb) for qgb in self.qgbs]
        self.currentRow, self.currentCol = [0 for i in row], [0 for i in col]

        self.currentId = 0

    def setCurrentId(self, i):
        if 0 <= i < len(self.qgbs):
            self.currentId = i

    def addGrid(self, row, col, rowSpan=1, colSpan=1):
        self.qgbs.append(QGroupBox())
        super().addWidget(self.qgbs[-1], row, col, rowSpan, colSpan)
        self.grids.append(QGridLayout(self.qgbs[-1]))
        self.currentRow.append(0)
        self.currentCol.append(0)
        self.currentId = len(self.qgbs) - 1

    def addWidget(self, widget, nextRow=False, rowSpan=1, colSpan=1):
        i = self.currentId
        self.grids[i].addWidget(
            widget, self.currentRow[i], self.currentCol[i], rowSpan, colSpan
        )
        self.currentCol[i] += colSpan
        if nextRow:
            self.currentRow[i] += 1
            self.currentCol[i] = 0

    def nextRow(self, num=1):
        self.currentRow[self.currentId] += num
        self.currentCol[self.currentId] = 0

    def nextCol(self, num=1):
        self.currentCol[self.currentId] += num

    def setTitle(self, title="", i=-1):
        if i == -1:
            self.qgbs[self.currentId].setTitle(title)
        elif 0 <= i < len(self.qgbs):
            self.qgbs[i].setTitle(title)

    def setFont(self, font=None, i=-1):
        if i == -1:
            self.qgbs[self.currentId].setFont(font)
        elif 0 <= i < len(self.qgbs):
            self.qgbs[i].setFont(font)

    def setVisible(self, visible=True, i=-1):
        if i == -1:
            self.qgbs[self.currentId].setVisible(visible)
        elif 0 <= i < len(self.qgbs):
            self.qgbs[i].setVisible(visible)

    def setSpacing(self, spacing=0, orientation=0, i=-1) -> None:
        if orientation == 0:
            if i == -1:
                self.grids[self.currentId].setHorizontalSpacing(spacing)
            elif 0 <= i < len(self.grids):
                self.grids[i].setHorizontalSpacing(spacing)
        else:
            if i == -1:
                self.grids[self.currentId].setVerticalSpacing(spacing)
            elif 0 <= i < len(self.grids):
                self.grids[i].setVerticalSpacing(spacing)

    def getGroupBox(self, i=-1):
        if i == -1:
            return self.qgbs[self.currentId]
        elif 0 <= i < len(self.qgbs):
            return self.qgbs[i]

    def getGridLayout(self, i=-1):
        if i == -1:
            return self.grids[self.currentId]
        elif 0 <= i < len(self.grids):
            return self.grids[i]


class SwitchButton(QPushButton):
    def __init__(self, parent=None, texts=["Run", "Stop"], state=True):
        super().__init__(parent=parent, text=texts[0] if state else texts[1])
        self.texts = texts
        self.state = state
        self.clicked.connect(self.switch)

    def switch(self):
        self.state = not self.state
        self.setText(self.texts[0] if self.state else self.texts[1])

    def setState(self, state):
        self.state = state
        self.setText(self.texts[0] if state else self.texts[1])


def save_data(filename, data, channels=2, sampwidth=2, rate=96000):
    """
    data: np.array
    if channels==2, make sure data is N*2
    """
    try:
        if filename[-4:] == ".npy":
            np.save(filename, data)
        elif filename[-4:] == ".mat":
            savemat(filename, {"data": data})
        elif filename[-4:] == ".txt":
            np.savetxt(filename, data, "%5.5f")
        elif filename[-5:] == ".xlsx":
            data_df = pd.DataFrame(data)
            writer = pd.ExcelWriter(filename)
            data_df.to_excel(writer, "page_1", float_format="%5.5f", index=False)
            writer.save()
        elif filename[-4:] == ".wav":
            if channels == 2:
                data = data.flatten()
            wf = wave.open(filename, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(rate)
            wf.writeframes((data * 32768).astype(np.int16).tobytes())
            wf.close()
        return True
    except Exception:
        return False


def reshape_data(data):
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data


def load_data(filename: str, fs=0):
    """
    从文件中载入数据，返回N*C的array，C为通道数，最大为2
    """
    try:
        if filename[-4:] == ".wav":
            f = wave.open(filename, "rb")
            params = f.getparams()
            nchannels, svppwidth, framerate, nframes = params[:4]
            strData = f.readframes(nframes)
            waveData = np.frombuffer(strData, dtype=np.int16) / 32768
            if nchannels == 1:
                waveData = waveData.reshape(-1, 1)
            else:
                waveData = np.array([waveData[::nchannels], waveData[1::nchannels]]).T
            if fs and framerate != fs:
                waveData = signal.resample_poly(waveData, fs, framerate, axis=0)
            return waveData
        else:
            if filename[-4:] == ".mat":
                data = loadmat(filename)
                data = reshape_data(data["data"])[:, :2]
            elif filename[-4:] == ".npy":
                data = reshape_data(np.load(filename))[:, :2]
            elif filename[-4:] == ".txt":
                data = reshape_data(np.loadtxt(filename))[:, :2]
            elif filename[-5:] == ".xlsx":
                data_df = pd.read_excel(filename)
                data = data_df.values[:, :2]
            return data
    except Exception as e:
        return repr(e)


if __name__ == "__main__":
    pass
