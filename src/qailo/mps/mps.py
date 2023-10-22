import numpy as np


class mps:
    """
    mps representation of quantum pure state

    shape of tensors: [du, dp, dl]
        du: dimension of upper leg (1 for top tensor)
        dp: dimension of physical leg (typically 2)
        dl: dimension of lower leg (1 for bottom tensor)

    canonical position: cp in range(n)
        tensors [0...cp-1]: top canonical
        tensors [cp+1...n-1]: bottom canonical
    """

    def __init__(self, tensors, q2t, t2q, cp):
        self.tensors_ = tensors
        self.q2t_ = q2t
        self.t2q_ = t2q
        self.cp_ = cp
        mps.check(self)

    def num_qubits(self):
        return len(self.tensors_)

    def tensor(self, t):
        return self.tensors_[t]

    def q2t(self, q):
        return self.q2t_[q]

    def qubit2tensor(self, q):
        return self.q2t(q)

    def t2q(self, t):
        return self.t2q_[t]

    def tensor2qubit(self, t):
        return self.t2q(t)

    def cp(self):
        return self.cp_

    def canonical_position(self):
        return self.cp()

    @staticmethod
    def check(mps):
        n = mps.num_qubits()

        # tensor shape
        assert mps.tensor(0).shape[0] == 1
        for t in range(1, n - 1):
            assert mps.tensor(t).shape[0] == mps.tensor(t - 1).shape[2]
            assert mps.tensor(t).shape[2] == mps.tensor(t + 1).shape[0]
        assert mps.tensor(n - 1).shape[2] == 1

        # qubit <-> tensor mapping
        for q in range(n):
            assert mps.t2q(mps.q2t(q)) == q
            assert mps.tensor2qubit(mps.qubit2tensor(q)) == q
        for t in range(n):
            assert mps.q2t(mps.t2q(t)) == t
            assert mps.qubit2tensor(mps.tensor2qubit(t)) == t

        # canonical form
        assert mps.cp() in range(n + 1)
        assert mps.canonical_position() in range(n + 1)


def check(m):
    mps.check(m)


def product_state(n: int, c=0):
    assert n > 0
    tensors = []
    for t in range(n):
        tensor = np.zeros((1, 2, 1))
        tensor[0, (c >> (n - t - 1)) & 1, 0] = 1
        tensors.append(tensor)
    return mps(tensors, range(n), range(n), 0)


def norm(m: mps):
    A = np.identity(2)
    for t in range(m.num_qubits()):
        A = np.einsum("ij,ikl,jkm->lm", A, m.tensor(t), m.tensor(t).conj())
    return np.sqrt(np.trace(A))
