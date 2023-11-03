import numpy as np

from .type import is_operator, num_qubits


def multiply(pin, p, pos=None):
    assert is_operator(pin)
    assert is_operator(p)
    n = num_qubits(pin)
    m = num_qubits(p)
    if pos is None:
        assert m == n
        pos = range(n)
    assert len(pos) == m and m <= n
    for i in pos:
        assert i < n

    ss_pin = list(range(2 * n))
    ss_p = list(range(2 * n, 2 * n + 2 * m))
    ss_to = list(range(2 * n))
    for i in range(m):
        ss_pin[pos[i]] = ss_p[m + i]
        ss_to[pos[i]] = ss_p[i]
    return np.einsum(pin, ss_pin, p, ss_p, ss_to)
