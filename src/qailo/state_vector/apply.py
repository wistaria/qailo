import numpy as np

from ..operator import type as op
from . import type as sv


def apply(v, p, pos=None):
    assert sv.is_state_vector(v) and op.is_operator(p)
    n = sv.num_qubits(v)
    m = op.num_qubits(p)
    if pos is None:
        assert m == n
        pos = range(n)
    assert len(pos) == m

    ss_v = list(range(2 * m, 2 * m + n + 1))
    ss_op = list(range(2 * m))
    ss_to = list(range(2 * m, 2 * m + n + 1))
    for i in range(m):
        ss_v[pos[i]] = ss_op[m + i]
        ss_to[pos[i]] = ss_op[i]
    return np.einsum(v, ss_v, p, ss_op, ss_to)


def apply_seq(v, seq):
    for p, qubit in seq:
        v = apply(v, p, qubit)
    return v
