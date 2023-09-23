import numpy as np

from ..util.letters import letters

def trace(op, pos = None):
    n = len(op.shape) // 2
    if pos == None:
        pos = range(n)
    assert len(pos) <= n
    for i in pos: assert(i < n)

    ss = letters()[:n] + letters()[:n]
    return np.einsum(ss, op)
