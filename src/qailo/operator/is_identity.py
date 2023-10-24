import numpy as np

from .type import is_operator, num_qubits
from .identity import identity


def is_identity(op):
    assert is_operator(op)
    n = num_qubits(op)
    return np.allclose(op, identity(n))
