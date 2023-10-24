import numpy as np


def state_vector(n, c=0):
    v = np.zeros(2**n)
    v[c] = 1
    return v.reshape((2,) * n + (1,))
