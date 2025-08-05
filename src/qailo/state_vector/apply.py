from __future__ import annotations

from typing import Iterable, Sequence

import numpy.typing as npt

from ..operator import type as op
from ..typeutil import eincheck as ec
from . import type as sv


def apply(
    v: npt.NDArray,
    p: npt.NDArray,
    pos: Sequence[int] | None = None,
) -> npt.NDArray:
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
    return ec.einsum_cast(v, ss_v, p, ss_op, ss_to)


def apply_seq(
    v: npt.NDArray,
    seq: Iterable[tuple[npt.NDArray, list[int]]],
) -> npt.NDArray:
    for p, qubit in seq:
        v = apply(v, p, qubit)
    return v
