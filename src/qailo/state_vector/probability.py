import numpy as np

from .type import num_qubits
from .vector import vector


def probability(v, c=None):
    if c is None:
        v = v / np.linalg.norm(v)
        return abs(vector(v)) ** 2
    else:
        assert len(c) == num_qubits(v)
        return abs(v(c)) ** 2
