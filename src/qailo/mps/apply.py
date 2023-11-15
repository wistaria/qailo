from copy import deepcopy

from ..operator import type as op
from ..operator.swap import swap
from . import type as mps


def _swap_tensors(m, s, maxdim=None):
    """
    swap neighboring two tensors at s and s+1
    """
    assert s in range(0, mps.num_qubits(m) - 1)
    m._apply_two(swap(), s, maxdim=maxdim)
    p0, p1 = m.t2q[s], m.t2q[s + 1]
    m.q2t[p0], m.q2t[p1] = s + 1, s
    m.t2q[s], m.t2q[s + 1] = p1, p0


def _move_qubit(m, p, s, maxdim=None):
    if m.q2t[p] != s:
        # print(f"moving qubit {p} at {m.q2t[p]} to {s}")
        for u in range(m.q2t[p], s):
            # print(f"swap tensors {u} and {u+1}")
            _swap_tensors(m, u, maxdim=maxdim)
        for u in range(m.q2t[p], s, -1):
            # print(f"swap tensors {u-1} and {u}")
            _swap_tensors(m, u - 1, maxdim=maxdim)


def _apply(m, p, pos=None, maxdim=None):
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
            m._apply_two(p, ss[0], maxdim=maxdim)
        else:
            _move_qubit(m, pos[0], ss[1] + 1)
            m._apply_two(p, ss[1], maxdim=maxdim, reverse=True)
    else:
        raise ValueError
    return m


def apply(m, p, pos=None, maxdim=None):
    return _apply(deepcopy(m), p, pos, maxdim)


def _apply_seq(m, seq, maxdim=None):
    for p, qubit in seq:
        _apply(m, p, qubit, maxdim)
    return m


def apply_seq(m, seq, maxdim=None):
    return _apply_seq(deepcopy(m), seq, maxdim)
