from __future__ import annotations

import numpy.typing as npt

from .type import is_mps, mps


def state_vector(m: mps) -> npt.NDArray:
    assert is_mps(m)
    return m._state_vector()
