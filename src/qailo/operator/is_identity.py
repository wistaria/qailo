import numpy as np

from ..num_qubits import num_qubits
from .identity import identity


def is_identity(op):
    n = num_qubits(op)
    return np.allclose(op, identity(n))
