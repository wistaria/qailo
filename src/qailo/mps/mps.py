import numpy as np

from ..operator import type as op
from .svd import tensor_svd


class MPS:
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

    def __init__(self, tensors):
        self.tensors = tensors
        n = len(self.tensors)
        self.q2t = list(range(n))
        self.t2q = [0] * n
        for p in range(n):
            self.t2q[self.q2t[p]] = p
        self.cp = [0, n - 1]

    def _canonicalize(self, p0, p1=None):
        p1 = p0 if p1 is None else p1
        n = len(self.tensors)
        assert 0 <= p0 and p0 <= p1 and p1 < n
        if self.cp[0] < p0:
            for t in range(self.cp[0], p0):
                L, R = tensor_svd(self.tensors[t], [[0, 1], [2]], "left")
                self.tensors[t] = L
                self.tensors[t + 1] = np.einsum("il,ljk->ijk", R, self.tensors[t + 1])
        self.cp[0] = p0
        self.cp[1] = max(p0, self.cp[1])
        if self.cp[1] > p1:
            for t in range(self.cp[1], p1, -1):
                L, R = tensor_svd(self.tensors[t], [[0], [1, 2]], "right")
                self.tensors[t - 1] = np.einsum("ijl,lk->ijk", self.tensors[t - 1], L)
                self.tensors[t] = R
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
        for t in range(0, self.cp[0]):
            A = np.einsum("ijk,ijl->kl", self.tensors[t], self.tensors[t].conj())
            assert np.allclose(A, np.identity(A.shape[0]))
        for t in range(self.cp[1] + 1, n):
            A = np.einsum("ijk,ljk->il", self.tensors[t], self.tensors[t].conj())
            assert np.allclose(A, np.identity(A.shape[0]))
        return True

    def _apply_one(self, p, s):
        assert op.num_qubits(p) == 1
        self.tensors[s] = np.einsum("abc,db->adc", self.tensors[s], p)
        self.cp[0] = min(self.cp[0], s)
        self.cp[1] = max(self.cp[1], s)

    def _apply_two(self, p, s, maxdim=None, reverse=False):
        """
        apply 2-qubit operator on neighboring tensors, s and s+1
        """
        assert op.num_qubits(p) == 2
        self._canonicalize(s, s + 1)
        t0 = self.tensors[s]
        t1 = self.tensors[s + 1]
        if not reverse:
            t = np.einsum("abc,cde,fgbd->afge", t0, t1, p)
        else:
            t = np.einsum("abc,cde,fgdb->agfe", t0, t1, p)
        L, R = tensor_svd(t, [[0, 1], [2, 3]], nkeep=maxdim)
        self.tensors[s] = L
        self.tensors[s + 1] = R
