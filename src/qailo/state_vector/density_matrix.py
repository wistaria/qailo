from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..typeutil import eincheck as ec
from .type import is_state_vector, num_qubits


def density_matrix(v: npt.NDArray) -> npt.NDArray:
    assert is_state_vector(v)
    n = num_qubits(v)
    w = v.copy()
    w = (w / np.linalg.norm(w)).reshape((2,) * n)
    ss0 = list(range(n))
    ss1 = list(range(n, 2 * n))
    shape = (2,) * (2 * n) + (1, 1)
    return ec.einsum_cast(w, ss0, w.conj(), ss1).reshape(shape)
