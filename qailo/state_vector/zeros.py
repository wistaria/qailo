import numpy as np

from ..util.shape import shape


def zeros(n):
    v = np.zeros(2**n)
    v[0] = 1
    return v.reshape(shape(n))
