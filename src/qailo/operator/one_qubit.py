from __future__ import annotations

import numpy as np
import numpy.typing as npt


def h() -> npt.NDArray:
    return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)


def p(phi: float) -> npt.NDArray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * phi)]])


def rx(phi: float) -> npt.NDArray:
    c = np.cos(phi / 2)
    s = np.sin(phi / 2)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]])


def ry(phi: float) -> npt.NDArray:
    c = np.cos(phi / 2)
    s = np.sin(phi / 2)
    return np.array([[c, -s], [s, c]])


def rz(phi: float) -> npt.NDArray:
    return np.array([[np.exp(-1.0j * phi / 2), 0], [0.0, np.exp(1.0j * phi / 2)]])


def s() -> npt.NDArray:
    return np.array([[1.0, 0.0], [0.0, 1.0j]])


def t() -> npt.NDArray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4)]])


def x() -> npt.NDArray:
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def y() -> npt.NDArray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def z() -> npt.NDArray:
    return np.array([[1.0, 0.0], [0.0, -1.0]])
