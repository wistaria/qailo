import numpy as np

from .svd import compact_svd


def normalize_s(s0, s1):
    ss0 = s0.copy()
    ss1 = s1.copy()
    i = 0
    for k in range(len(s0)):
        if s0[k] not in s1:
            ss0[k] = i
            i += 1
    print(s0, s1, ss0, ss1)
    for k in range(len(s1)):
        if s1[k] not in s0:
            ss1[k] = i
            i += 1
    print(s0, s1, ss0, ss1)
    for k in range(len(s0)):
        if s0[k] in s1:
            ss0[k] += i
    print(s0, s1, ss0, ss1)
    for k in range(len(s1)):
        if s1[k] in s0:
            ss1[k] += i
    print(s0, s1, ss0, ss1)
    return ss0, ss1


def collect_legs(s0, s1):
    legs0L = []
    legs0R = []
    legs1L = []
    legs1R = []
    for k in range(len(s0)):
        if s0[k] not in s1:
            legs0L.append(k)
        else:
            legs0R.append(k)
    for k in range(len(s1)):
        if s1[k] not in s0:
            legs1R.append(k)
        else:
            legs1L.append(k)
    return legs0L, legs0R, legs1L, legs1R


def projector(T0, s0, T1, s1, nkeep=None, tol=1e-12):
    ss0, ss1 = normalize_s(s0, s1)
    legs0L, legs0R, legs1L, legs1R = collect_legs(ss0, ss1)
    dim0L = np.prod([T0.shape[i] for i in legs0L])
    dims0R = [T0.shape[i] for i in legs0R]
    dims1L = [T1.shape[i] for i in legs1L]
    dim1R = np.prod([T1.shape[i] for i in legs1R])
    assert len(dims0R) == len(dims1L)
    TT0 = np.einsum(T0, ss0).reshape([dim0L] + dims0R)
    TT1 = np.einsum(T1, ss1).reshape([dim1R] + dims1L)
    ss_sum = list(range(2, len(dims0R) + 2))
    A = np.einsum(TT0, [0] + ss_sum, TT1, [1] + ss_sum)
    S, U, V = compact_svd(A, nkeep=nkeep, tol=tol)
    U = np.einsum(U, [0, 1], np.sqrt(1 / S), [1], [0, 1])
    V = np.einsum(V, [0, 1], np.sqrt(1 / S), [1], [0, 1])
    WL = np.einsum(TT0, [0] + ss_sum, U, [0, max(ss_sum) + 1])
    WR = np.einsum(TT1, [0] + ss_sum, V, [0, max(ss_sum) + 1])
    return WL, WR
