import string
import numpy as np

from ..util.replace import replace

def multiply(op, opi, pos):
    n = len(opi.shape) // 2
    m = len(op.shape) // 2
    assert len(pos) == m and m <= n
    for i in pos: assert(i < n)
    ss_opi = ss_to = string.ascii_lowercase[:2*n]
    ss_op = string.ascii_uppercase[:2*m]
    for i in range(m):
        ss_opi = replace(ss_opi, pos[i], ss_op[m+i])
        ss_to = replace(ss_to, pos[i], ss_op[i])
    return np.einsum("{},{}->{}".format(ss_opi, ss_op, ss_to), opi, op)