import numpy as np

from ..util.shape import shape


def swap():
    op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return op.reshape(shape(4))
