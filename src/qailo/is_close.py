import numpy as np


def is_close(p0, p1):
    return p0.shape == p1.shape and np.allclose(p0, p1)
