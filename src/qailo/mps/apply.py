from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence

import numpy.typing as npt

from ..operator import type as op
from ..operator.swap import swap
from ..util.helpertype import OPSeqElement
from . import type as mps


def _swap_tensors(m: mps.mps, s: int) -> None:
    """
    swap neighboring two tensors at s and s+1
    """
    assert s in range(0, mps.num_qubits(m) - 1)
    m._apply_two(swap(), s)
    p0, p1 = m.t2q[s], m.t2q[s + 1]
    m.q2t[p0], m.q2t[p1] = s + 1, s
    m.t2q[s], m.t2q[s + 1] = p1, p0


def _move_qubit(m: mps.mps, p: int, s: int) -> None:
    if m.q2t[p] != s:
        # print(f"moving qubit {p} at {m.q2t[p]} to {s}")
        for u in range(m.q2t[p], s):
            # print(f"swap tensors {u} and {u+1}")
            _swap_tensors(m, u)
        for u in range(m.q2t[p], s, -1):
            # print(f"swap tensors {u-1} and {u}")
            _swap_tensors(m, u - 1)


def _apply(m: mps.mps, p: npt.NDArray, pos: Sequence[int] | None = None) -> mps.mps:
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


def apply(m: mps.mps, p: npt.NDArray, pos: Sequence[int] | None = None) -> mps.mps:
    return _apply(deepcopy(m), p, pos)


def _apply_seq(m: mps.mps, seq: Iterable[OPSeqElement]) -> mps.mps:
    for p, qubit in seq:
        _apply(m, p, qubit)
    return m


def apply_seq(m: mps.mps, seq: Iterable[OPSeqElement]) -> mps.mps:
    return _apply_seq(deepcopy(m), seq)
