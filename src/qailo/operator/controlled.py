import numpy as np

from ..is_operator import is_operator
from .matrix import matrix
from ..num_qubits import num_qubits
from .pauli import x, z
from .shape import shape


def controlled(u):
    assert is_operator(u)
    m = num_qubits(u)
    n = m + 1

    op = np.identity(2**n).reshape([2, 2**m, 2, 2**m])
    op[1, :, 1, :] = matrix(u)
    return op.reshape(shape(n))


def cx(n=2):
    op = x()
    for _ in range(n - 1):
        op = controlled(op)
    return op


def cz(n=2):
    op = z()
    for _ in range(n - 1):
        op = controlled(op)
    return op
