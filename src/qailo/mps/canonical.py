import numpy as np

from .type import num_qubits


def is_canonical(m):
    if m.cp(0) == m.cp(1):
        for t in range(0, m.cp(0)):
            A = np.einsum("ijk,ijl->kl", m._tensors[t], m._tensors[t].conj())
            if not np.allclose(A, np.identity(A.shape[0])):
                return False
        for t in range(m.cp(1) + 1, num_qubits(m)):
            A = np.einsum("ijk,ljk->il", m._tensors[t], m._tensors[t].conj())
            if not np.allclose(A, np.identity(A.shape[0])):
                return False
        return True
    else:
        return False
