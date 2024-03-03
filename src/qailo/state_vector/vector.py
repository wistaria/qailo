from __future__ import annotations

import numpy.typing as npt

from . import type as sv


def vector(v: npt.NDArray, c: list | None = None) -> npt.NDArray:
    assert sv.is_state_vector(v)
    if c is None:
        n = sv.num_qubits(v)
        return v.reshape((2**n,))
    else:
        assert isinstance(c, list)
        w = v
        for x in c:
            w = w[x]
        return w
