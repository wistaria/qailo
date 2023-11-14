import numpy as np


def fidelity(v0, v1):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    return np.abs(np.vdot(v0, v1)) ** 2
