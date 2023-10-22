import numpy as np

from .mps import mps


def check_canonical(m: mps):
    for t in range(0, m.cp()):
        A = np.einsum("ijk,ijl->kl", m.tensors_[t], m.tensors_[t].conj())
        assert A.shape[0] == A.shape[1]
        if not np.allclose(A, np.identity(A.shape[0])):
            return False
    for t in range(m.cp() + 1, m.num_qubits()):
        A = np.einsum("ijk,ljk->il", m.tensors_[t], m.tensors_[t].conj())
        assert A.shape[0] == A.shape[1]
        if not np.allclose(A, np.identity(A.shape[0])):
            return False
    return True
