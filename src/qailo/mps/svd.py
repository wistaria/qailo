from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from ..typeutil import eincheck as ec


class LegPartition(NamedTuple):
    leg0: list[int]
    leg1: list[int]


def compact_svd(
    A: npt.NDArray, nkeep: int | None = None, tol: float = 1e-12
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    assert A.ndim == 2
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    dimS = sum([1 if x > tol * S[0] else 0 for x in S])
    dimS = dimS if nkeep is None else min(dimS, nkeep)
    return S[:dimS], U[:, :dimS], V[:, :dimS]


def tensor_svd(
    T: npt.NDArray,
    partition: LegPartition,
    canonical: Literal["center", "left", "right"] = "center",
    nkeep: int | None = None,
    tol: float = 1e-12,
) -> tuple[npt.NDArray, npt.NDArray]:
    legsL = len(partition[0])
    legsR = len(partition[1])
    assert T.ndim == legsL + legsR
    assert sorted(partition[0] + partition[1]) == list(range(T.ndim))
    dimsL = [T.shape[i] for i in partition[0]]
    dimsR = [T.shape[i] for i in partition[1]]
    m = np.einsum(T, partition[0] + partition[1]).reshape(
        np.prod(dimsL), np.prod(dimsR)
    )
    S, U, V = compact_svd(m, nkeep=nkeep, tol=tol)
    L = U
    R = V.conj().T
    if canonical == "center":
        L = ec.einsum_cast("ij,j->ij", L, np.sqrt(S))
        R = ec.einsum_cast("i,ij->ij", np.sqrt(S), R)
    elif canonical == "left":
        R = ec.einsum_cast("i,ij->ij", S, R)
    elif canonical == "right":
        L = ec.einsum_cast("ij,j->ij", L, S)
    else:
        raise ValueError
    L = L.reshape(dimsL + [S.shape[0]])
    R = R.reshape([S.shape[0]] + dimsR)
    return L, R
