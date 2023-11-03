import numpy as np

from .type import is_operator, num_qubits


def hconj(op):
    assert is_operator(op)
    n = num_qubits(op)
    ss_from = list(range(2 * n))
    ss_to = list(range(n, 2 * n)) + list(range(n))
    return np.einsum(op, ss_from, ss_to).conjugate()
