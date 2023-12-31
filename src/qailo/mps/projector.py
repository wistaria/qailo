import numpy as np

from ..mps.svd import compact_svd


def normalize_ss(ss0, ss1):
    """
    Order the subscripts that are not contracted from 0.
    Subscripts that will be contracted are assigned a larger value.
    """
    ss0_out = ss0.copy()
    ss1_out = ss1.copy()
    i = 0
    for k in range(len(ss0)):
        if ss0[k] not in ss1:
            ss0_out[k] = i
            i += 1
    for k in range(len(ss1)):
        if ss1[k] not in ss0:
            ss1_out[k] = i
            i += 1
    for k in range(len(ss0)):
        if ss0[k] in ss1:
            ss0_out[k] += i
    for k in range(len(ss1)):
        if ss1[k] in ss0:
            ss1_out[k] += i
    return ss0_out, ss1_out


def collect_legs(ss0, ss1):
    """
    legs0L: list of legs in T0 that are NOT contracted
    legs0R: list of legs in T0 that will be contracted
    legs1L: list of legs in T1 that will be contracted
    legs1R: list of legs in T1 that are NOT contracted
    """
    legs0L = []
    legs0R = []
    legs1L = []
    legs1R = []
    for k in range(len(ss0)):
        if ss0[k] not in ss1:
            legs0L.append(k)
        else:
            legs0R.append(k)
    for k in range(len(ss1)):
        if ss1[k] not in ss0:
            legs1R.append(k)
        else:
            legs1L.append(k)
    return legs0L, legs0R, legs1L, legs1R


def projector(T0, ss0_in, T1, ss1_in, nkeep=None, tol=1e-12):
    ss0, ss1 = normalize_ss(ss0_in, ss1_in)
    legs0L, legs0R, legs1L, legs1R = collect_legs(ss0, ss1)
    dim0L = np.prod([T0.shape[i] for i in legs0L])
    dims0R = [T0.shape[i] for i in legs0R]
    dims1L = [T1.shape[i] for i in legs1L]
    dim1R = np.prod([T1.shape[i] for i in legs1R])
    assert len(dims0R) == len(dims1L)
    TT0 = np.einsum(T0, ss0).reshape([dim0L] + dims0R)
    TT1 = np.einsum(T1, ss1).reshape([dim1R] + dims0R)
    ss_sum = list(range(2, len(dims0R) + 2))
    A = np.einsum(TT0, [0] + ss_sum, TT1, [1] + ss_sum)
    S, U, V = compact_svd(A, nkeep=nkeep, tol=tol)
    U = np.einsum(U, [0, 1], np.sqrt(1 / S), [1], [0, 1])
    V = np.einsum(V, [0, 1], np.sqrt(1 / S), [1], [0, 1])
    PL = np.einsum(TT0.conj(), [0] + ss_sum, U, [0, max(ss_sum) + 1])
    PR = np.einsum(TT1, [0] + ss_sum, V, [0, max(ss_sum) + 1])
    return S, PL.conj(), PR
