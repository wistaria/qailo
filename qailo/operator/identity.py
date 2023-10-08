import numpy as np

from .shape import shape


def identity(n):
    return np.identity(2**n).reshape(shape(n))
