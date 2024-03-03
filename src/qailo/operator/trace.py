from __future__ import annotations

from typing import Collection

import numpy.typing as npt

from ..typeutil import eincheck as ec
from .type import is_density_matrix, is_operator, num_qubits


def trace(q: npt.NDArray, pos: Collection[int] | None = None) -> npt.ArrayLike:
    assert is_density_matrix(q) or is_operator(q)
    n = num_qubits(q)
    if pos is None:
        pos = range(n)
    m = n - len(pos)
    assert m >= 0
    for i in pos:
        assert i < n

    ss = list(range(2 * n))
    for i in pos:
        ss[n + i] = ss[i]
    return ec.einsum(q.reshape((2,) * (2 * n)), ss)
