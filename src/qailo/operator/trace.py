import numpy as np

from .type import is_density_matrix, is_operator, num_qubits


def trace(q, pos=None):
    assert is_density_matrix(q) or is_operator(q)
    n = num_qubits(q)
    if pos is None:
        pos = range(n)
    m = n - len(pos)
    assert m >= 0
    for i in pos:
        assert i < n

    ss = list(range(2 * n))
    for i in pos:
        ss[n + i] = ss[i]
    return np.einsum(q, ss)
