import numpy as np


def h():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def rx(p):
    c = np.cos(p / 2)
    s = np.sin(p / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(p):
    c = np.cos(p / 2)
    s = np.sin(p / 2)
    return np.array([[c, -s], [s, c]])


def rz(p):
    return np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]])


def s():
    return np.array([[1, 0], [0, 1j]])


def t():
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def x():
    return np.array([[0, 1], [1, 0]])


def y():
    return np.array([[0, -1j], [1j, 0]])


def z():
    return np.array([[1, 0], [0, -1]])
