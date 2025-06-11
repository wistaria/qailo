from copy import deepcopy

import numpy as np

from ..operator import type as op
from . import type as mps


def _swap_tensors(m, s):
    """
    swap neighboring two tensors at s and s+1
    """
    assert s in range(0, mps.num_qubits(m) - 1)
    d0 = m.tensors[s].shape[1]
    d1 = m.tensors[s + 1].shape[1]
    op = np.identity(d0 * d1).reshape([d0, d1, d0, d1]).transpose(1, 0, 2, 3)
    m._apply_two(op, s)
    p0, p1 = m.t2q[s], m.t2q[s + 1]
    m.q2t[p0], m.q2t[p1] = s + 1, s
    m.t2q[s], m.t2q[s + 1] = p1, p0


def _move_qubit(m, p, s):
    if m.q2t[p] != s:
        # print(f"moving qubit {p} at {m.q2t[p]} to {s}")
        for u in range(m.q2t[p], s):
            # print(f"swap tensors {u} and {u+1}")
            _swap_tensors(m, u)
        for u in range(m.q2t[p], s, -1):
            # print(f"swap tensors {u-1} and {u}")
            _swap_tensors(m, u - 1)


def _apply(m, p, pos=None):
    assert op.is_operator(p)
    n = mps.num_qubits(m)
    if pos is None:
        assert op.num_qubits(p) == n
        pos = list(range(n))
    assert len(pos) == op.num_qubits(p)
    if op.num_qubits(p) == 1:
        m._apply_one(p, m.q2t[pos[0]])
    elif op.num_qubits(p) == 2:
        ss = [m.q2t[pos[0]], m.q2t[pos[1]]]
        assert ss[0] != ss[1]
        if ss[0] < ss[1]:
            _move_qubit(m, pos[1], ss[0] + 1)
            m._apply_two(p, ss[0])
        else:
            _move_qubit(m, pos[0], ss[1] + 1)
            m._apply_two(p, ss[1], reverse=True)
    else:
        raise ValueError
    return m


def apply(m, p, pos=None):
    return _apply(deepcopy(m), p, pos)


def _apply_seq(m, seq):
    for p, qubit in seq:
        _apply(m, p, qubit)
    return m


def apply_seq(m, seq):
    return _apply_seq(deepcopy(m), seq)
