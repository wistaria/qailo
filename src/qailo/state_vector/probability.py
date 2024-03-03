from __future__ import annotations

from typing import Container

import numpy as np
import numpy.typing as npt

from ..operator.matrix import matrix
from ..operator.trace import trace
from .density_matrix import density_matrix
from .type import is_state_vector, num_qubits
from .vector import vector


def probability(v: npt.NDArray, pos: Container[int] | None = None) -> npt.NDArray:
    assert is_state_vector(v)
    w = v / float(np.linalg.norm(v))
    if pos is None:
        return abs(vector(w)) ** 2
    else:
        tpos: list[int] = []
        for k in range(num_qubits(w)):
            if k not in pos:
                tpos.append(k)
        traced = np.real(trace(density_matrix(w), tpos))
        return np.diag(matrix(traced))
