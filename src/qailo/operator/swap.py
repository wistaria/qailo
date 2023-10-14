import numpy as np

from .shape import shape


def swap():
    op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return op.reshape(shape(2))
