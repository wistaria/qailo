import numpy as np


def compact_svd(A, nkeep=None, tol=1e-12):
    assert len(A.shape) == 2
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    dimS = sum([1 if x > tol * S[0] else 0 for x in S])
    dimS = dimS if nkeep is None else min(dimS, nkeep)
    return S[:dimS], U[:, :dimS], V[:, :dimS]


def tensor_svd(T, partition, canonical="center", nkeep=None, tol=1e-12):
    legs0 = len(partition[0])
    legs1 = len(partition[1])
    assert len(T.shape) == legs0 + legs1
    assert sorted(partition[0] + partition[1]) == list(range(len(T.shape)))
    dims0 = [T.shape[i] for i in partition[0]]
    dims1 = [T.shape[i] for i in partition[1]]
    m = np.einsum(T, partition[0] + partition[1]).reshape(
        np.prod(dims0), np.prod(dims1)
    )
    S, U, V = compact_svd(m, nkeep, tol)
    L = U
    R = V.conj().T
    if canonical == "center":
        L = np.einsum("ij,j->ij", L, np.sqrt(S))
        R = np.einsum("i,ij->ij", np.sqrt(S), R)
    elif canonical == "left":
        R = np.einsum("i,ij->ij", S, R)
    elif canonical == "right":
        L = np.einsum("ij,j->ij", L, S)
    else:
        raise ValueError
    L = L.reshape(dims0 + [S.shape[0]])
    R = R.reshape([S.shape[0]] + dims1)
    return L, R
