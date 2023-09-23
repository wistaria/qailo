import numpy as np

from ..util.shape import shape

def identity(n):
    return np.identity(2**n).reshape(shape(2*n))
