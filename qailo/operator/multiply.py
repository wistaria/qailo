import numpy as np

from ..util.letters import letters
from ..util.replace import replace


def multiply(op, opi, pos=None):
    n = len(opi.shape) // 2
    m = len(op.shape) // 2
    if pos == None:
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
    return np.einsum("{},{}->{}".format(ss_opi, ss_op, ss_to), opi, op)
