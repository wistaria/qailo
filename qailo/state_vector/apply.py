import numpy as np

from ..util.letters import letters
from ..util.replace import replace

def apply(op, v, pos = None):
    n = len(v.shape)
    m = len(op.shape) // 2
    if pos == None:
        assert m == n
        pos = range(n)
    assert len(pos) == m
    assert m <= n
    for i in pos:
        assert(i < n)

    ss_v = ss_to = letters()[:n]
    ss_op = letters()[n:n+2*m]
    for i in range(m):
        ss_v = replace(ss_v, pos[i], ss_op[m+i])
        ss_to = replace(ss_to, pos[i], ss_op[i])
    return np.einsum("{},{}->{}".format(ss_v, ss_op, ss_to), v, op)
