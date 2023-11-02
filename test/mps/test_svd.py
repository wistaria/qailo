import numpy as np
import qailo as q


def test_compact_svd():
    maxn = 64
    nt = 64
    for _ in range(nt):
        m, n, d = np.random.randint(2, maxn, size=(3,))
        A = np.random.random((m, n))
        S, U, V = q.mps.compact_svd(A)
        assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V))
        for c in "center", "left", "right":
            L, R = q.mps.compact_svd(A, canonical=c)
            assert np.allclose(A, np.einsum("ik,jk->ij", L, R))
        S, U, V = q.mps.compact_svd(A, nkeep=d)
        assert S.shape[0] == U.shape[1]
        assert S.shape[0] == V.shape[1]
        assert U.shape[0] == m
        assert V.shape[0] == n
        if d >= min(m, n):
            assert np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V))
        else:
            assert not np.allclose(A, np.einsum("k,ik,jk->ij", S, U, V))


def test_svd_left():
    maxn = 32
    nt = 64
    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p))
        L, R = q.mps.svd_left(T)
        assert len(L.shape) == 3
        assert len(R.shape) == 2
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert np.allclose(T, np.einsum("ijl,lk->ijk", L, R))
        L, R = q.mps.svd_left(T, nkeep=d)
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
    maxn = 32
    nt = 64
    for _ in range(nt):
        m, n, p, d = np.random.randint(2, maxn, size=(4,))
        T = np.random.random((m, n, p))
        L, R = q.mps.svd_right(T)
        assert len(L.shape) == 2
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == R.shape[0]
        assert R.shape[1] == n
        assert R.shape[2] == p
        assert np.allclose(T, np.einsum("il,ljk->ijk", L, R))
        L, R = q.mps.svd_right(T, nkeep=d)
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
    nt = 64
    for _ in range(nt):
        m, n, p, r, d = np.random.randint(2, maxn, size=(5,))
        T = np.random.random((m, n, p, r))
        L, R = q.mps.svd_two(T)
        assert len(L.shape) == 3
        assert len(R.shape) == 3
        assert L.shape[0] == m
        assert L.shape[1] == n
        assert L.shape[2] == R.shape[0]
        assert R.shape[1] == p
        assert R.shape[2] == r
        assert np.allclose(T, np.einsum("ijm,mkl->ijkl", L, R))
        L, R = q.mps.svd_two(T, nkeep=d)
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
