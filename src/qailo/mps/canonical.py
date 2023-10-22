from copy import deepcopy

import numpy as np

from .mps import mps


def canonicalize(m: mps, p: int):
    assert 0 <= p and p < m.num_qubits()
    tensors = deepcopy(m.tensors_)
    if m.cp() < p:
        for t in range(m.cp(), p):
            dims = list(tensors[t].shape)
            A = tensors[t].reshape((dims[0] * dims[1], dims[2]))
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            dims[2] = S.shape[0]
            tensors[t] = U.reshape(dims)
            tensors[t + 1] = np.einsum("i,ij,jkl->ikl", S, Vh, tensors[t + 1])
    else:
        for t in range(m.cp(), p, -1):
            dims = list(tensors[t].shape)
            A = tensors[t].reshape((dims[0], dims[1] * dims[2]))
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            dims[0] = S.shape[0]
            tensors[t] = Vh.reshape(dims)
            tensors[t - 1] = np.einsum("ijk,kl,l->ijl", tensors[t - 1], U, S)
    return mps(tensors, m.q2t_, m.t2q_, p)


def check_canonical(m):
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
