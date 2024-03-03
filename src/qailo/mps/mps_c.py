from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt

from ..operator import type as op
from ..typeutil import eincheck as ec
from .svd import LegPartition, tensor_svd
from .type import mps


class canonical_mps(mps):
    """
    MPS representation of quantum pure state

    shape of tensors: [du, dp, dl]
        du: dimension of left leg (1 for left-edge tensor)
        dp: dimension of physical leg (typically 2)
        dl: dimension of right leg (1 for right-edge tensor)

    canonical position: cp in range(n)
        0 <= cp(0) <= cp(1) < n
        tensors [0...cp(0)-1]: left canonical
        tensors [cp(1)+1...n-1]: right canonical
    """

    tensors: list[npt.NDArray]
    nkeep: int | None
    cp: list[int]

    def __init__(self, tensors: list[npt.NDArray], nkeep: int | None = None) -> None:
        assert isinstance(tensors, list)
        self.tensors = deepcopy(tensors)
        self.nkeep = nkeep
        n = len(self.tensors)
        self.q2t = list(range(n))
        self.t2q = list(range(n))
        self.cp = [0, n - 1]

    def _num_qubits(self) -> int:
        return len(self.tensors)

    def _norm(self) -> float:
        A = np.identity(1)
        for t in range(self._num_qubits()):
            A = ec.einsum_cast("ij,jkl->ikl", A, self._tensor(t))
            A = ec.einsum_cast("ijk,ijl->kl", A, self._tensor(t).conj())
        ret = np.sqrt(np.trace(A))
        assert isinstance(ret, float)
        return ret

    def _state_vector(self) -> npt.NDArray:
        n = self._num_qubits()
        v = self._tensor(0)
        for t in range(1, n):
            ss0 = list(range(t + 1)) + [t + 3]
            ss1 = [t + 3, t + 1, t + 2]
            v = ec.einsum_cast(v, ss0, self._tensor(t), ss1)
        v = v.reshape((2,) * n)
        return ec.einsum_cast(v, self.t2q).reshape((2,) * n + (1,))

    def _tensor(self, t: int) -> npt.NDArray:
        return self.tensors[t]

    def _canonicalize(self, p0: int, p1: int | None = None) -> None:
        p1 = p0 if p1 is None else p1
        n = len(self.tensors)
        assert 0 <= p0 and p0 <= p1 and p1 < n
        if self.cp[0] < p0:
            for t in range(self.cp[0], p0):
                L, R = tensor_svd(self.tensors[t], LegPartition([0, 1], [2]), "left")
                self.tensors[t] = L
                self.tensors[t + 1] = ec.einsum_cast(
                    R, [0, 3], self.tensors[t + 1], [3, 1, 2]
                )
        self.cp[0] = p0
        self.cp[1] = max(p0, self.cp[1])
        if self.cp[1] > p1:
            for t in range(self.cp[1], p1, -1):
                L, R = tensor_svd(self.tensors[t], LegPartition([0], [1, 2]), "right")
                self.tensors[t - 1] = ec.einsum_cast(
                    self.tensors[t - 1], [0, 1, 3], L, [3, 2]
                )
                self.tensors[t] = R
        self.cp[1] = p1

    def _is_canonical(self) -> bool:
        # tensor shape
        n = len(self.tensors)
        dims = []
        assert self.tensors[0].shape[0] == 1
        dims.append(self.tensors[0].shape[0])
        for t in range(1, n - 1):
            dims.append(self.tensors[t].shape[0])
            assert self.tensors[t].shape[0] == self.tensors[t - 1].shape[2]
            assert self.tensors[t].shape[2] == self.tensors[t + 1].shape[0]
        assert self.tensors[n - 1].shape[2] == 1
        dims.append(self.tensors[n - 1].shape[0])
        dims.append(self.tensors[n - 1].shape[2])

        # qubit <-> tensor mapping
        for q in range(n):
            assert self.t2q[self.q2t[q]] == q
        for t in range(n):
            assert self.q2t[self.t2q[t]] == t

        # canonicality
        assert self.cp[0] in range(n)
        assert self.cp[1] in range(n)
        for t in range(0, self.cp[0]):
            A = ec.einsum_cast(
                self.tensors[t], [2, 3, 1], self.tensors[t].conj(), [2, 3, 0]
            )
            assert np.allclose(A, np.identity(A.shape[0]))
        for t in range(self.cp[1] + 1, n):
            A = ec.einsum_cast(
                self.tensors[t], [1, 3, 2], self.tensors[t].conj(), [0, 3, 2]
            )
            assert np.allclose(A, np.identity(A.shape[0]))
        return True

    def _apply_one(self, p: npt.NDArray, s: int) -> None:
        """
        apply 1-qubit operator on tensor at s
        """
        assert op.num_qubits(p) == 1
        self.tensors[s] = ec.einsum_cast(self.tensors[s], [0, 3, 2], p, [1, 3])
        self.cp[0] = min(self.cp[0], s)
        self.cp[1] = max(self.cp[1], s)

    def _apply_two(self, p: npt.NDArray, s: int, reverse: bool = False) -> None:
        """
        apply 2-qubit operator on neighboring tensors at s and s+1
        """
        assert op.num_qubits(p) == 2
        self._canonicalize(s, s + 1)
        t0 = self.tensors[s]
        t1 = self.tensors[s + 1]
        t = ec.einsum_cast(t0, [0, 1, 4], t1, [4, 2, 3])
        if not reverse:
            t = ec.einsum_cast(t, [0, 4, 5, 3], p, [1, 2, 4, 5])
        else:
            t = ec.einsum_cast(t, [0, 4, 5, 3], p, [2, 1, 5, 4])
        L, R = tensor_svd(t, LegPartition([0, 1], [2, 3]), nkeep=self.nkeep)
        self.tensors[s] = L
        self.tensors[s + 1] = R
