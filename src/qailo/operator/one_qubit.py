import numpy as np


def h():
    return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)


def p(phi):
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * phi)]])


def rx(phi):
    c = np.cos(phi / 2)
    s = np.sin(phi / 2)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]])


def ry(phi):
    c = np.cos(phi / 2)
    s = np.sin(phi / 2)
    return np.array([[c, -s], [s, c]])


def rz(phi):
    return np.array([[np.exp(-1.0j * phi / 2), 0], [0.0, np.exp(1.0j * phi / 2)]])


def s():
    return np.array([[1.0, 0.0], [0.0, 1.0j]])


def t():
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4)]])


def x():
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def y():
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def z():
    return np.array([[1.0, 0.0], [0.0, -1.0]])
