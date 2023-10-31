import numpy as np


def svd(A, nkeep=None, canonical=None, tol=1e-12):
    assert len(A.shape) == 2
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    dimS = sum([1 if x > tol * S[0] else 0 for x in S])
    if nkeep is not None:
        dimS = min(dimS, nkeep)
    if canonical is None:
        return S[:dimS], U[:, :dimS], V[:, :dimS]
    elif canonical == "center":
        L = np.einsum("ij,j->ij", U[:, :dimS], np.sqrt(S[:dimS]))
        R = np.einsum("ij,j->ij", V[:, :dimS], np.sqrt(S[:dimS]))
        return L, R
    elif canonical == "left":
        R = np.einsum("ij,j->ij", V[:, :dimS], S[:dimS])
        return U[:, :dimS], R
    elif canonical == "right":
        L = np.einsum("ij,j->ij", U[:, :dimS], S[:dimS])
        return L, V[:, :dimS]
    else:
        raise ValueError


def svd_left(T, nkeep=None, tol=1e-12):
    assert len(T.shape) == 3
    dims = T.shape
    A = T.reshape((dims[0] * dims[1], dims[2]))
    L, R = svd(A, nkeep, "left", tol)
    assert L.shape[0] == dims[0] * dims[1]
    L = L.reshape((dims[0], dims[1], L.shape[1]))
    return L, R.T


def svd_right(T, nkeep=None, tol=1e-12):
    assert len(T.shape) == 3
    dims = T.shape
    A = T.reshape((dims[0], dims[1] * dims[2]))
    L, R = svd(A, nkeep, "right", tol)
    assert R.shape[0] == dims[1] * dims[2]
    R = R.T.reshape((R.shape[1], dims[1], dims[2]))
    return L, R


def svd_two(T, nkeep=None, canonical="center", tol=1e-12):
    assert len(T.shape) == 4
    dims = T.shape
    A = T.reshape((dims[0] * dims[1], dims[2] * dims[3]))
    L, R = svd(A, nkeep, canonical, tol)
    assert L.shape[0] == dims[0] * dims[1] and R.shape[0] == dims[2] * dims[3]
    L = L.reshape((dims[0], dims[1], L.shape[1]))
    R = R.T.reshape((R.shape[1], dims[2], dims[3]))
    return L, R
