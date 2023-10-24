import numpy as np


def identity(n):
    return np.identity(2**n).reshape((2,) * (2 * n))
