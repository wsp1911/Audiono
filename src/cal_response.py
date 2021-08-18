# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

# 0 for standard signal; 1 for response signal


def get_chunk(f, RATE):
    if f == 0:
        return 1024
    N = int(RATE / f)
    chunk = 2 ** int(np.log2(N * 6) + 1)
    return max(chunk, 1024)


def split_signal(f, RATE, CHUNK, y0, y1, height=0.01, connect=False):

    if f == 0:
        return 1, y0[-CHUNK:], y1[-CHUNK:]

    y0 = y0[-CHUNK:]
    y1 = y1[-CHUNK:]

    N = int(RATE / f)
    pks = signal.find_peaks(-y0, height=height, distance=N * 0.9)

    return len(pks[0]) - 1, y0[pks[0][0] : pks[0][-1]], y1[pks[0][0] : pks[0][-1]]


def cal_response(f, dt, N, y0, y1):

    if f == 0:
        return np.mean(y1) / np.mean(y0), 0

    t = np.linspace(0, N / f, len(y0))
    s0 = np.sin(2 * np.pi * f * t)
    s1 = np.cos(2 * np.pi * f * t)

    Integral = [
        [np.sum(y0 * s0 * dt) * 2 * f / N, np.sum(y0 * s1 * dt) * 2 * f / N],
        [np.sum(y1 * s0 * dt) * 2 * f / N, np.sum(y1 * s1 * dt) * 2 * f / N],
    ]

    A0 = np.sqrt(Integral[0][0] ** 2 + Integral[0][1] ** 2)
    A1 = np.sqrt(Integral[1][0] ** 2 + Integral[1][1] ** 2)

    phi0 = np.arccos(Integral[0][0] / A0)
    phi1 = np.arccos(Integral[1][0] / A1)

    return A1 / A0, (phi0 - phi1) * 180 / np.pi


def get_response(f, RATE, CHUNK, y0, y1):
    N, y0, y1 = split_signal(f, RATE, CHUNK, y0, y1)
    A, phi = cal_response(f, 1 / RATE, N, y0, y1)
    return A, phi


# matplotlib版本问题会导致打包后无法运行，这段代码用于测试
# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     RATE = 48000
#     f = 550
#     CHUNK = get_chunk(f, RATE)

#     y = np.load("bp_rec_test/y_%d.npy" % f)

#     # y0, y1 = y[0][: -2 * CHUNK], y[1][: -2 * CHUNK]
#     y0, y1 = y[0], y[1]

#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(y0, linewidth=1)
#     plt.plot(y1, linewidth=1)
#     for i in range(round(len(y0) / CHUNK) - 1):
#         plt.axvline(x=CHUNK * (i + 1), linewidth=1, ls="--", c="r")
#     plt.title("(1)")
#     plt.xticks([])
#     plt.yticks([])
#     # plt.show()

#     N, yy0, yy1 = split_signal(f, RATE, CHUNK, y0, y1)

#     plt.subplot(3, 1, 2)
#     pos = len(y0) - 2 * CHUNK + 300
#     plt.plot(y0[pos:], linewidth=1)
#     plt.plot(y1[pos:], linewidth=1)
#     N1 = int(RATE / f)
#     pks = signal.find_peaks(-y0, height=0.01, distance=N1 * 0.9)
#     p = []
#     for i in pks[0]:
#         p.append((i - pos, y0[i]))
#     p = np.array(p)
#     plt.plot(p[:, 0], p[:, 1], "ro", markersize=2)
#     plt.title("(2)")
#     plt.xticks([])
#     plt.yticks([])

#     plt.subplot(3, 1, 3)
#     plt.plot(yy0, linewidth=1)
#     plt.plot(yy1, linewidth=1)
#     plt.title("(3)")
#     plt.xticks([])
#     plt.yticks([])

#     plt.show()

#     # A, phi = cal(y[1], y[0])
#     # print(A, phi)
