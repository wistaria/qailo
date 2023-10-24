import numpy as np

from .identity import identity
from .type import is_operator, num_qubits


def is_identity(op):
    assert is_operator(op)
    n = num_qubits(op)
    return np.allclose(op, identity(n))
