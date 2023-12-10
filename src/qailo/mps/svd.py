import numpy as np


def compact_svd(A, nkeep=None, tol=1e-12):
    assert len(A.shape) == 2
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    dimS = sum([1 if x > tol * S[0] else 0 for x in S])
    dimS = dimS if nkeep is None else min(dimS, nkeep)
    return S[:dimS], U[:, :dimS], V[:, :dimS]


def tensor_svd(T, partition, canonical="center", nkeep=None, tol=1e-12):
    legsL = len(partition[0])
    legsR = len(partition[1])
    assert len(T.shape) == legsL + legsR
    assert sorted(partition[0] + partition[1]) == list(range(len(T.shape)))
    dimsL = [T.shape[i] for i in partition[0]]
    dimsR = [T.shape[i] for i in partition[1]]
    m = np.einsum(T, partition[0] + partition[1]).reshape(
        np.prod(dimsL), np.prod(dimsR)
    )
    S, U, V = compact_svd(m, nkeep=nkeep, tol=tol)
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
    L = L.reshape(dimsL + [S.shape[0]])
    R = R.reshape([S.shape[0]] + dimsR)
    return L, R
