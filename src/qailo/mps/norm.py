import numpy as np

from .type import num_qubits


def norm(m):
    A = np.identity(1)
    for t in range(num_qubits(m)):
        A = np.einsum("ij,jkl->ikl", A, m.tensors[t])
        A = np.einsum("ijk,ijl->kl", A, m.tensors[t].conj())
    return np.sqrt(np.trace(A))
