from __future__ import annotations

from typing import Sequence

import numpy.typing as npt

from ..typeutil import eincheck as ec
from .type import is_operator, num_qubits


def multiply(
    pin: npt.NDArray, p: npt.NDArray, pos: Sequence[int] | None = None
) -> npt.NDArray:
    assert is_operator(pin)
    assert is_operator(p)
    n = num_qubits(pin)
    m = num_qubits(p)
    if pos is None:
        assert m == n
        pos = range(n)
    assert len(pos) == m and m <= n
    for i in pos:
        assert i < n

    ss_pin = list(range(2 * n))
    ss_p = list(range(2 * n, 2 * n + 2 * m))
    ss_to = list(range(2 * n))
    for i in range(m):
        ss_pin[pos[i]] = ss_p[m + i]
        ss_to[pos[i]] = ss_p[i]
    return ec.einsum_cast(pin, ss_pin, p, ss_p, ss_to)
