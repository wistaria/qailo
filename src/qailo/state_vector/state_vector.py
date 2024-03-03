from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..typeutil import eincheck as ec
from .type import is_state_vector, num_qubits


def product_state(states: Sequence[npt.NDArray]) -> npt.NDArray:
    m = len(states)
    assert m > 0
    v = states[0]
    assert is_state_vector(v)
    for i in range(1, m):
        n0 = num_qubits(v)
        n1 = num_qubits(states[i])
        v = ec.einsum_cast(
            v.reshape((2,) * n0),
            list(range(n0)),
            states[i],
            list(range(n0, n0 + n1 + 1)),
        )
    return v


def zero(n: int = 1) -> npt.NDArray:
    assert n > 0
    if n == 1:
        return np.array((1.0, 0.0)).reshape((2, 1))
    else:
        return product_state([zero(1)] * n)


def one(n: int = 1) -> npt.NDArray:
    assert n > 0
    if n == 1:
        return np.array((0.0, 1.0)).reshape((2, 1))
    else:
        return product_state([one(1)] * n)
