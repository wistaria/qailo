import numpy as np

from ..is_operator import is_operator
from ..num_qubits import num_qubits
from ..util.letters import letters
from ..util.replace import replace


def multiply(op, opi, pos=None):
    assert is_operator(opi)
    assert is_operator(op)
    n = num_qubits(opi)
    m = num_qubits(op)
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
    print(opi.shape, op.shape)
    print("{},{}->{}".format(ss_opi, ss_op, ss_to))
    return np.einsum("{},{}->{}".format(ss_opi, ss_op, ss_to), opi, op)
