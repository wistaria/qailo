import numpy as np

from .shape import shape


def state_vector(n, c=0):
    v = np.zeros(2**n)
    v[c] = 1
    return v.reshape(shape(n))
