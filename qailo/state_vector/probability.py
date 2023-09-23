import numpy as np

from .vector import vector


def probability(v):
    v = v / np.linalg.norm(v)
    return abs(vector(v)) ** 2
