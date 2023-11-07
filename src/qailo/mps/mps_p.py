from copy import deepcopy

import numpy as np

from ..operator import type as op
from .projector import _full_projector
from .svd import tensor_svd


class MPS_P:
    """
    MPS representation of quantum pure state

    shape of tensors: [du, dp, dl]
        du: dimension of upper leg (1 for top tensor)
        dp: dimension of physical leg (typically 2)
        dl: dimension of lower leg (1 for bottom tensor)

    canonical position: cp in range(n)
        0 <= cp(0) <= cp(1) < n
        tensors [0...cp(0)-1]: top canonical
        tensors [cp(1)+1...n-1]: bottom canonical
    """

    def __init__(self, tensors):
        assert isinstance(tensors, list)
        self.tensors = deepcopy(tensors)
        n = len(self.tensors)
        self.q2t = list(range(n))
        self.t2q = list(range(n))
        self.cp = [0, n - 1]
        # canonicalization matrices
        # put sentinels (1x1 identities) at t = 0 and t = n
        self.cmat = [np.identity(1)] + [None] * (n - 1) + [np.identity(1)]

    def _canonicalize(self, p0, p1=None):
        p1 = p0 if p1 is None else p1
        n = len(self.tensors)
        assert 0 <= p0 and p0 <= p1 and p1 < n
        if self.cp[0] < p0:
            for t in range(self.cp[0], p0):
                A = np.einsum(self.cmat[t], [0, 3], self.tensors[t], [3, 1, 2])
                _, self.cmat[t + 1] = tensor_svd(A, [[0, 1], [2]], "left")
        self.cp[0] = p0
        self.cp[1] = max(p0, self.cp[1])
        if self.cp[1] > p1:
            for t in range(self.cp[1], p1, -1):
                A = np.einsum(self.tensors[t], [0, 1, 3], self.cmat[t + 1], [3, 2])
                self.cmat[t], _ = tensor_svd(A, [[0], [1, 2]], "right")
        self.cp[1] = p1

    def _is_canonical(self):
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
        A = np.identity(1)
        for t in range(0, self.cp[0]):
            A = np.einsum(A, [0, 3], self.tensors[t], [3, 1, 2])
            A = np.einsum(A, [2, 3, 1], self.tensors[t].conj(), [2, 3, 0])
            B = np.einsum(self.cmat[t + 1], [2, 1], self.cmat[t + 1].conj(), [2, 0])
            assert A.shape == B.shape
            assert np.allclose(A, B)
        A = np.identity(1)
        for t in range(n - 1, self.cp[1], -1):
            A = np.einsum(self.tensors[t], [0, 1, 3], A, [3, 2])
            A = np.einsum(self.tensors[t].conj(), [1, 2, 3], A, [0, 2, 3])
            B = np.einsum(self.cmat[t], [0, 2], self.cmat[t].conj(), [1, 2])
            assert np.allclose(A, B)
        return True

    def _apply_one(self, p, s):
        assert op.num_qubits(p) == 1
        self.tensors[s] = np.einsum(self.tensors[s], [0, 3, 2], p, [1, 3])
        self.cp[0] = min(self.cp[0], s)
        self.cp[1] = max(self.cp[1], s)

    def _apply_two(self, p, s, maxdim=None, reverse=False):
        """
        apply 2-qubit operator on neighboring tensors, s and s+1
        """
        self._canonicalize(s, s + 1)
        p0, p1 = tensor_svd(p, [[0, 2], [1, 3]])
        t0 = self.tensors[s]
        t1 = self.tensors[s + 1]
        if not reverse:
            t0 = np.einsum(t0, [0, 4, 3], p0, [1, 4, 2])
            t1 = np.einsum(t1, [0, 4, 3], p1, [1, 2, 4])
        else:
            t0 = np.einsum(t0, [0, 4, 3], p1, [2, 1, 4])
            t1 = np.einsum(t1, [0, 4, 3], p0, [2, 4, 1])
        tt0 = np.einsum(self.cmat[s], [0, 4], t0, [4, 1, 2, 3])
        tt1 = np.einsum(t1, [0, 1, 2, 4], self.cmat[s + 2], [4, 3])
        _, d, WL, WR = _full_projector(tt0, [0, 1, 4, 5], tt1, [5, 4, 2, 3])
        if maxdim is not None:
            d = min(d, maxdim)
        self.tensors[s] = np.einsum(t0, [0, 1, 3, 4], WR[:, :, :d], [3, 4, 2])
        self.tensors[s + 1] = np.einsum(
            WL.conj()[:, :, :d], [3, 4, 0], t1, [4, 3, 1, 2]
        )
