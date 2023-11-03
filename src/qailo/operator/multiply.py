import numpy as np

from ..util.strops import letters, replace
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

    ss_opi = ss_to = letters()[: 2 * n]
    ss_op = letters()[2 * n : 2 * n + 2 * m]
    for i in range(m):
        ss_opi = replace(ss_opi, pos[i], ss_op[m + i])
        ss_to = replace(ss_to, pos[i], ss_op[i])
    return np.einsum("{},{}->{}".format(ss_opi, ss_op, ss_to), pin, p)
