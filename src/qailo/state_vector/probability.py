import numpy as np

from ..operator.matrix import matrix
from ..operator.trace import trace
from .density_matrix import density_matrix
from .type import is_state_vector, num_qubits
from .vector import vector


def probability(v, pos=None):
    assert is_state_vector(v)
    w = v / np.linalg.norm(v)
    if pos is None:
        return abs(vector(w)) ** 2
    else:
        tpos = []
        for k in range(num_qubits(w)):
            if k not in pos:
                tpos.append(k)
        return np.diag(matrix(trace(density_matrix(w), tpos).real))
