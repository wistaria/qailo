from ..is_equal import is_equal
from ..num_qubits import num_qubits
from .identity import identity


def is_identity(op):
    n = num_qubits(op)
    return is_equal(op, identity(n))
