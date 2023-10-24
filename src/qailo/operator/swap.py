import numpy as np


def swap():
    op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return op.reshape([2, 2, 2, 2])
