import string
import numpy as np

from ..util.replace import replace

def multiply(op, v, pos):
    n = len(v.shape)
    m = len(op.shape) // 2
    assert(len(pos) == m)
    assert(m <= n)
    for i in pos:
        assert(i < n)
    ss_v = ss_to = string.ascii_lowercase[:n]
    ss_op = string.ascii_uppercase[:2*m]
    for i in range(m):
        ss_v = replace(ss_v, pos[i], ss_op[m+i])
        ss_to = replace(ss_to, pos[i], ss_op[i])
    return np.einsum("{},{}->{}".format(ss_v, ss_op, ss_to), v, op)
