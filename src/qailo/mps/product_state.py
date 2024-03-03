from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

from ..dispatch import num_qubits
from ..state_vector.state_vector import one as sv_one
from ..state_vector.state_vector import zero as sv_zero
from ..state_vector.vector import vector
from .mps_c import canonical_mps
from .svd import LegPartition, tensor_svd
from .type import is_mps, mps


def tensor_decomposition(
    v: mps | npt.NDArray, nkeep: int | None = None, tol: float = 1e-12
) -> list[npt.NDArray]:
    if is_mps(v):
        return [v._tensor(s) for s in range(num_qubits(v))]
    else:
        assert isinstance(v, np.ndarray)
        n = num_qubits(v)
        w = vector(v).reshape((1, 2**n))
        tensors: list[npt.NDArray] = []
        for t in range(n - 1):
            dims = w.shape
            w = w.reshape(dims[0], 2, dims[1] // 2)
            t, w = tensor_svd(w, LegPartition([0, 1], [2]), "left", nkeep, tol)
            tensors.append(t)
        tensors.append(w.reshape(w.shape + (1,)))
        return tensors


def product_state(
    states: Iterable[mps | npt.NDArray],
    nkeep: int | None = None,
    mps: type[mps] = canonical_mps,
) -> mps:
    tensors: list[npt.NDArray] = []
    for s in states:
        tensors = tensors + tensor_decomposition(s, nkeep)
    return mps(tensors, nkeep)


def zero(n: int = 1, nkeep: int | None = None, mps: type[mps] = canonical_mps) -> mps:
    return product_state([sv_zero()] * n, nkeep, mps)


def one(n: int = 1, nkeep: int | None = None, mps: type[mps] = canonical_mps) -> mps:
    return product_state([sv_one()] * n, nkeep, mps)
