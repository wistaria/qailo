from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeGuard


def is_density_matrix(v: Any) -> TypeGuard[np.ndarray]:
    if isinstance(v, np.ndarray):
        return (
            v.shape[-1] == 1
            and v.shape[-2] == 1
            and v.shape[-3] > 1
            and v.ndim % 2 == 0
        )
    return False


def is_operator(v: Any) -> TypeGuard[np.ndarray]:
    if isinstance(v, np.ndarray):
        return v.shape[-1] > 1 and v.ndim % 2 == 0
    return False


def num_qubits(v: npt.NDArray) -> int:
    if is_density_matrix(v):
        return (v.ndim - 2) // 2
    elif is_operator(v):
        return v.ndim // 2
    raise ValueError


class OPAutomaton(NamedTuple):
    op: npt.NDArray
    pos: list[int]
