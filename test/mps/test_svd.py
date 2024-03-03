import numpy as np
from qailo.mps.svd import LegPartition, compact_svd, tensor_svd


def test_compact_svd():
    maxn = 16
    nt = 16
    for _ in range(nt):
        m, n, d = np.random.randint(2, maxn, size=(3,))
        A = np.random.random((m, n))
        S, U, V = compact_svd(A)
        assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))
        S, U, V = compact_svd(A, nkeep=d)
        assert S.shape[0] == U.shape[1]
        assert S.shape[0] == V.shape[1]
        assert U.shape[0] == m
        assert V.shape[0] == n
        if d >= min(m, n):
            assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))
        else:
            assert not np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))

    for _ in range(nt):
        m, n, d = np.random.randint(2, maxn, size=(3,))
        A = np.random.random((m, n)) + 1j * np.random.random((m, n))
        S, U, V = compact_svd(A)
        assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))
        S, U, V = compact_svd(A, nkeep=d)
        assert S.shape[0] == U.shape[1]
        assert S.shape[0] == V.shape[1]
        assert U.shape[0] == m
        assert V.shape[0] == n
        if d >= min(m, n):
            assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))
        else:
            assert not np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V.conj()))


def test_svd_left():
    maxn = 16
    nt = 16
    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p))
        L, R = tensor_svd(T, LegPartition([0, 1], [2]), "left")
        assert len(L.shape) == 3
        assert len(R.shape) == 2
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert np.allclose(T, np.einsum("ijl,lk->ijk", L, R))
        L, R = tensor_svd(T, LegPartition([0, 1], [2]), "left", nkeep=d)
        assert len(L.shape) == 3
        assert len(R.shape) == 2
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        if d >= min(m * n, p):
            assert np.allclose(T, np.einsum("ijl,lk->ijk", L, R))
        else:
            assert not np.allclose(T, np.einsum("ijl,lk->ijk", L, R))

    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p)) + 1j * np.random.random((m, n, p))
        L, R = tensor_svd(T, LegPartition([0, 1], [2]), "left")
        assert len(L.shape) == 3
        assert len(R.shape) == 2
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert np.allclose(T, np.einsum("ijl,lk->ijk", L, R))
        L, R = tensor_svd(T, LegPartition([0, 1], [2]), "left", nkeep=d)
        assert len(L.shape) == 3
        assert len(R.shape) == 2
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        if d >= min(m * n, p):
            assert np.allclose(T, np.einsum("ijl,lk->ijk", L, R))
        else:
            assert not np.allclose(T, np.einsum("ijl,lk->ijk", L, R))


def test_svd_right():
    maxn = 16
    nt = 16
    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p))
        L, R = tensor_svd(T, LegPartition([0], [1, 2]), "right")
        assert len(L.shape) == 2
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == R.shape[0]
        assert R.shape[1] == n
        assert R.shape[2] == p
        assert np.allclose(T, np.einsum("il,ljk->ijk", L, R))
        L, R = tensor_svd(T, LegPartition([0], [1, 2]), "right", nkeep=d)
        assert len(L.shape) == 2
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == R.shape[0]
        assert R.shape[1] == n
        assert R.shape[2] == p
        if d >= min(m, n * p):
            assert np.allclose(T, np.einsum("il,ljk->ijk", L, R))
        else:
            assert not np.allclose(T, np.einsum("il,ljk->ijk", L, R))

    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p)) + 1j * np.random.random((m, n, p))
        L, R = tensor_svd(T, LegPartition([0], [1, 2]), "right")
        assert len(L.shape) == 2
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == R.shape[0]
        assert R.shape[1] == n
        assert R.shape[2] == p
        assert np.allclose(T, np.einsum("il,ljk->ijk", L, R))
        L, R = tensor_svd(T, LegPartition([0], [1, 2]), "right", nkeep=d)
        assert len(L.shape) == 2
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == R.shape[0]
        assert R.shape[1] == n
        assert R.shape[2] == p
        if d >= min(m, n * p):
            assert np.allclose(T, np.einsum("il,ljk->ijk", L, R))
        else:
            assert not np.allclose(T, np.einsum("il,ljk->ijk", L, R))


def test_svd_two():
    maxn = 8
    nt = 16
    for _ in range(nt):
        m, n, p, r, d = np.random.randint(2, maxn, size=(5,))
        T = np.random.random((m, n, p, r))
        L, R = tensor_svd(T, LegPartition([0, 1], [2, 3]))
        assert len(L.shape) == 3
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert R.shape[2] == r
        assert np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))
        L, R = tensor_svd(T, LegPartition([0, 1], [2, 3]), nkeep=d)
        assert len(L.shape) == 3
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert R.shape[2] == r
        if d >= min(m * n, p * r):
            assert np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))
        else:
            assert not np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))

    for _ in range(nt):
        m, n, p, r, d = np.random.randint(2, maxn, size=(5,))
        T = np.random.random((m, n, p, r)) + 1j * np.random.random((m, n, p, r))
        L, R = tensor_svd(T, LegPartition([0, 1], [2, 3]))
        assert len(L.shape) == 3
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert R.shape[2] == r
        assert np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))
        L, R = tensor_svd(T, LegPartition([0, 1], [2, 3]), nkeep=d)
        assert len(L.shape) == 3
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert R.shape[2] == r
        if d >= min(m * n, p * r):
            assert np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))
        else:
            assert not np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))


if __name__ == "__main__":
    test_compact_svd()
    test_svd_left()
    test_svd_right()
    test_svd_two()
