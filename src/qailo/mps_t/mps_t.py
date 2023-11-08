from copy import deepcopy

import numpy as np

from ..mps.svd import tensor_svd
from ..mps.type import MPS
from ..mps_p.projector import _full_projector
from ..operator import type as op


class MPS_T(MPS):
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
        n = len(tensors)
        self.tpool = []
        self.tcurrent = []
        for t in tensors:
            self.tcurrent.append(self._register(deepcopy(t)))
        self.gpool = []
        self.q2t = list(range(n))
        self.t2q = list(range(n))
        self.cp = [0, n - 1]
        # canonicalization matrices
        # put sentinels (1x1 identities) at t = 0 and t = n
        self.env = [np.identity(1)] + [None] * (n - 1) + [np.identity(1)]

    def _num_qubits(self):
        return len(self.tcurrent)

    def _tensor(self, t):
        return self.tpool[self.tcurrent[t]][0]

    def _canonicalize(self, p0, p1=None):
        p1 = p0 if p1 is None else p1
        assert 0 <= p0 and p0 <= p1 and p1 < self._num_qubits()
        if self.cp[0] < p0:
            for t in range(self.cp[0], p0):
                A = np.einsum(self.env[t], [0, 3], self._tensor(t), [3, 1, 2])
                _, self.env[t + 1] = tensor_svd(A, [[0, 1], [2]], "left")
        self.cp[0] = p0
        self.cp[1] = max(p0, self.cp[1])
        if self.cp[1] > p1:
            for t in range(self.cp[1], p1, -1):
                A = np.einsum(self._tensor(t), [0, 1, 3], self.env[t + 1], [3, 2])
                self.env[t], _ = tensor_svd(A, [[0], [1, 2]], "right")
        self.cp[1] = p1

    def _is_canonical(self):
        # tensor shape
        n = len(self.tcurrent)
        dims = []
        assert self._tensor(0).shape[0] == 1
        dims.append(self._tensor(0).shape[0])
        for t in range(1, n - 1):
            dims.append(self._tensor(t).shape[0])
            assert self._tensor(t).shape[0] == self._tensor(t - 1).shape[2]
            assert self._tensor(t).shape[2] == self._tensor(t + 1).shape[0]
        assert self._tensor(n - 1).shape[2] == 1
        dims.append(self._tensor(n - 1).shape[0])
        dims.append(self._tensor(n - 1).shape[2])

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
            A = np.einsum(A, [0, 3], self._tensor(t), [3, 1, 2])
            A = np.einsum(A, [2, 3, 1], self._tensor(t).conj(), [2, 3, 0])
            B = np.einsum(self.env[t + 1], [2, 1], self.env[t + 1].conj(), [2, 0])
            assert A.shape == B.shape
            assert np.allclose(A, B)
        A = np.identity(1)
        for t in range(n - 1, self.cp[1], -1):
            A = np.einsum(self._tensor(t), [0, 1, 3], A, [3, 2])
            A = np.einsum(self._tensor(t).conj(), [1, 2, 3], A, [0, 2, 3])
            B = np.einsum(self.env[t], [0, 2], self.env[t].conj(), [1, 2])
            assert np.allclose(A, B)
        return True

    def _apply_one(self, p, s):
        assert op.num_qubits(p) == 1
        pid = self._register(p)
        self.tcurrent[s] = self._contract(self.tcurrent[s], [0, 3, 2], pid, [1, 3])
        self.cp[0] = min(self.cp[0], s)
        self.cp[1] = max(self.cp[1], s)

    def _apply_two(self, p, s, maxdim=None, reverse=False):
        """
        apply 2-qubit operator on neighboring tensors, s and s+1
        """
        self._canonicalize(s, s + 1)
        tid0 = self.tcurrent[s]
        tid1 = self.tcurrent[s + 1]
        p0, p1 = tensor_svd(p, [[0, 2], [1, 3]])
        pid0 = self._register(p0)
        pid1 = self._register(p1)
        if not reverse:
            tid0 = self._contract(tid0, [0, 4, 3], pid0, [1, 4, 2])
            tid1 = self._contract(tid1, [0, 4, 3], pid1, [1, 2, 4])
        else:
            tid0 = self._contract(tid0, [0, 4, 3], pid1, [2, 1, 4])
            tid1 = self._contract(tid1, [0, 4, 3], pid0, [2, 4, 1])
        tt0 = np.einsum(self.env[s], [0, 4], self.tpool[tid0][0], [4, 1, 2, 3])
        tt1 = np.einsum(self.tpool[tid1][0], [0, 1, 2, 4], self.env[s + 2], [4, 3])
        _, WLhid, WRid = self._projector(tt0, [0, 1, 4, 5], tt1, [5, 4, 2, 3], maxdim)
        self.tcurrent[s] = self._contract(tid0, [0, 1, 3, 4], WRid, [3, 4, 2])
        self.tcurrent[s + 1] = self._contract(WLhid, [3, 4, 0], tid1, [4, 3, 1, 2])

    def _register(self, tensor):
        id = len(self.tpool)
        self.tpool.append([tensor, "initial", None])
        return id

    def _contract(self, tid0, ss0, tid1, ss1, ss2=None):
        id = len(self.tpool)
        if ss2 is None:
            t = np.einsum(self.tpool[tid0][0], ss0, self.tpool[tid1][0], ss1)
        else:
            np.einsum(self.tpool[tid0][0], ss0, self.tpool[tid1][0], ss1, ss2)
        self.tpool.append([t, "product", [tid0, tid1]])
        return id

    def _projector(self, t0, ss0, t1, ss1, maxdim=None):
        S, d, WLh, WR = _full_projector(t0, ss0, t1, ss1)
        d = d if maxdim is None else min(d, maxdim)
        gid = len(self.gpool)
        self.gpool.append([S, d, WLh, WR])
        lid = len(self.tpool)
        shape = WLh.shape
        WLh = WLh.reshape((np.prod(shape[:-1]), shape[-1]))
        WLh = WLh[:, :d].reshape(shape[:-1] + (d,))
        self.tpool.append([WLh, "squeezer", [gid]])
        rid = len(self.tpool)
        shape = WR.shape
        WR = WR.reshape((np.prod(shape[:-1]), shape[-1]))
        WR = WR[:, :d].reshape(shape[:-1] + (d,))
        self.tpool.append([WR, "squeezer", [gid]])
        return gid, lid, rid

    def _dump(self, prefix):
        import json
        import pickle

        dic = {"prefix": prefix}
        tlist = []
        with open(f"{prefix}-tensor.pkl", "wb") as f:
            for id, tp in enumerate(self.tpool):
                m = {}
                m["id"] = id
                m["shape"] = tp[0].shape
                m["type"] = tp[1]
                m["from"] = tp[2]
                tlist.append(m)
                if tp[1] == "initial":
                    pickle.dump(tp[0], f)
        dic["tensor"] = tlist
        glist = []
        with open(f"{prefix}-generator.pkl", "wb") as f:
            for id, gp in enumerate(self.gpool):
                m = {}
                m["id"] = id
                m["d"] = gp[1]
                m["shape S"] = gp[0].shape
                m["shape L"] = gp[2].shape
                m["shape R"] = gp[3].shape
                glist.append(m)
                pickle.dump(gp[0], f)
                pickle.dump(gp[2], f)
                pickle.dump(gp[3], f)
        dic["generator"] = glist

        with open(prefix + "-graph.json", mode="w") as f:
            json.dump(dic, f, indent=2)
