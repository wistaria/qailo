from __future__ import annotations

import numpy.typing as npt

from ..typeutil import eincheck as ec
from .type import is_operator, num_qubits


def hconj(op: npt.NDArray) -> npt.NDArray:
    assert is_operator(op)
    n = num_qubits(op)
    ss_from = list(range(2 * n))
    ss_to = list(range(n, 2 * n)) + list(range(n))
    return ec.einsum_cast(op, ss_from, ss_to).conjugate()
