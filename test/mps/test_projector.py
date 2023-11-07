import numpy as np
import qailo as q
from pytest import approx
from qailo.mps.projector import _full_projector


def test_projector():
    maxn = 4
    nt = 16
    for _ in range(nt):
        n0, n1, n2, d = np.random.randint(2, maxn, size=(4,))
        T0 = np.random.random((n0, n1))
        T1 = np.random.random((n1, n2))
        _, WL, WR = q.mps.projector(T0, [0, 2], T1, [2, 1], nkeep=d)
        assert WL.shape[1] <= d and WR.shape[1] <= d
        assert WL.shape[0] == n1 and WR.shape[0] == n1
        A = np.einsum(T0, [0, 2], WR, [2, 3], WL.conj().T, [3, 4], T1, [4, 1])

        S, U, V = q.mps.compact_svd(np.einsum(T0, [0, 2], T1, [2, 1]), nkeep=d)
        B = np.einsum("k,ik,jk->ij", S, U, V)
        assert np.allclose(A, B)

    for _ in range(nt):
        n0, n1, n2, n3, n4, n5, d = np.random.randint(2, maxn, size=(7,))
        T0 = np.random.random((n0, n1, n2, n3))
        T1 = np.random.random((n3, n2, n4, n5))
        _, WL, WR = q.mps.projector(T0, [0, 1, 4, 5], T1, [5, 4, 2, 3], nkeep=d)
        assert WL.shape[2] <= d and WR.shape[2] <= d
        assert WL.shape[0] == n2 and WL.shape[1] == n3
        assert WR.shape[0] == n2 and WR.shape[1] == n3
        A = np.einsum(
            T0, [0, 1, 4, 5], WR, [4, 5, 6], WL.conj(), [7, 8, 6], T1, [8, 7, 2, 3]
        )

        L, R = q.mps.tensor_svd(
            np.einsum(T0, [0, 1, 4, 5], T1, [5, 4, 2, 3]), [[0, 1], [2, 3]], nkeep=d
        )
        B = np.einsum(L, [0, 1, 4], R, [4, 2, 3])
        assert np.allclose(A, B)

    for _ in range(nt):
        w0 = np.random.random((4, 4)) + 1.0j * np.random.random((4, 4))
        t0 = np.random.random((4, 2, 3)) + 1.0j * np.random.random((4, 2, 3))
        t1 = np.random.random((3, 2, 2)) + 1.0j * np.random.random((3, 2, 2))
        w2 = np.random.random((2, 2)) + 1.0j * np.random.random((2, 2))
        p = np.random.random((2, 2, 2, 2)) + 1.0j * np.random.random((2, 2, 2, 2))
        B = np.einsum(t0, [0, 4, 6], t1, [6, 5, 3], p, [1, 2, 4, 5])
        p0, p1 = q.mps.svd.tensor_svd(p, [[0, 2], [1, 3]])
        assert np.allclose(np.einsum(p0, [0, 2, 4], p1, [4, 1, 3]), p)
        t0 = np.einsum(t0, [0, 4, 3], p0, [1, 4, 2])
        t1 = np.einsum(t1, [0, 4, 3], p1, [1, 2, 4])
        assert np.allclose(np.einsum(t0, [0, 1, 4, 5], t1, [5, 4, 2, 3]), B)
        tt0 = np.einsum(w0, [0, 4], t0, [4, 1, 2, 3])
        tt1 = np.einsum(t1, [0, 1, 2, 4], w2, [4, 3])
        _, WL, WR = q.mps.projector(tt0, [0, 1, 4, 5], tt1, [5, 4, 2, 3])
        assert np.allclose(
            np.einsum(WL.conj(), [2, 3, 0], WR, [2, 3, 1]), np.identity(WL.shape[2])
        )
        tt0 = np.einsum(t0, [0, 1, 3, 4], WR, [3, 4, 2])
        tt1 = np.einsum(WL.conj(), [3, 4, 0], t1, [4, 3, 1, 2])
        A = np.einsum(tt0, [0, 1, 4], tt1, [4, 2, 3])
        print(np.linalg.norm(A - B))
        assert np.allclose(A, B)

    for _ in range(nt):
        w0 = np.random.random((4, 4)) + 1.0j * np.random.random((4, 4))
        t0 = np.random.random((4, 2, 3)) + 1.0j * np.random.random((4, 2, 3))
        t1 = np.random.random((3, 2, 2)) + 1.0j * np.random.random((3, 2, 2))
        w2 = np.random.random((2, 2)) + 1.0j * np.random.random((2, 2))
        p = np.random.random((2, 2, 2, 2)) + 1.0j * np.random.random((2, 2, 2, 2))
        B = np.einsum(t0, [0, 4, 6], t1, [6, 5, 3], p, [1, 2, 4, 5])
        p0, p1 = q.mps.svd.tensor_svd(p, [[0, 2], [1, 3]])
        assert np.allclose(np.einsum(p0, [0, 2, 4], p1, [4, 1, 3]), p)
        t0 = np.einsum(t0, [0, 4, 3], p0, [1, 4, 2])
        t1 = np.einsum(t1, [1, 4, 3], p1, [0, 2, 4])
        assert np.allclose(np.einsum(t0, [0, 1, 4, 5], t1, [4, 5, 2, 3]), B)
        tt0 = np.einsum(w0, [0, 4], t0, [4, 1, 2, 3])
        tt1 = np.einsum(t1, [0, 1, 2, 4], w2, [4, 3])
        _, WL, WR = q.mps.projector(tt0, [0, 1, 4, 5], tt1, [4, 5, 2, 3])
        assert np.allclose(
            np.einsum(WL.conj(), [2, 3, 0], WR, [2, 3, 1]), np.identity(WL.shape[2])
        )
        tt0 = np.einsum(t0, [0, 1, 3, 4], WR, [3, 4, 2])
        tt1 = np.einsum(WL.conj(), [3, 4, 0], t1, [3, 4, 1, 2])
        A = np.einsum(tt0, [0, 1, 4], tt1, [4, 2, 3])
        print(np.linalg.norm(A - B))
        assert np.allclose(A, B)


def test_full_projector():
    maxn = 4
    nt = 16
    for _ in range(nt):
        n0, n1, n2, d = np.random.randint(2, maxn, size=(4,))
        d = min(d, n0, n2)
        T0 = np.random.random((n0, n1))
        T1 = np.random.random((n1, n2))
        S, WL, WR = _full_projector(T0, [0, 2], T1, [2, 1])
        assert WL.shape[0] == n1 and WR.shape[0] == n1
        A = np.einsum(
            T0, [0, 2], WR[:, :d], [2, 3], WL.conj().T[:d, :], [3, 4], T1, [4, 1]
        )

        S, U, V = q.mps.compact_svd(np.einsum(T0, [0, 2], T1, [2, 1]), nkeep=d)
        B = np.einsum("k,ik,jk->ij", S, U, V)
        assert np.allclose(A, B)

    for _ in range(nt):
        n0, n1, n2, n3, n4, n5, d = np.random.randint(2, maxn, size=(7,))
        d = min(d, n0 * n1, n4 * n5)
        T0 = np.random.random((n0, n1, n2, n3))
        T1 = np.random.random((n3, n2, n4, n5))
        S, WL, WR = q.mps.projector(T0, [0, 1, 4, 5], T1, [5, 4, 2, 3], nkeep=d)
        assert WL.shape[0] == n2 and WL.shape[1] == n3
        assert WR.shape[0] == n2 and WR.shape[1] == n3
        A = np.einsum(
            T0,
            [0, 1, 4, 5],
            WR[:, :, :d],
            [4, 5, 6],
            WL.conj()[:, :, :d],
            [7, 8, 6],
            T1,
            [8, 7, 2, 3],
        )

        L, R = q.mps.tensor_svd(
            np.einsum(T0, [0, 1, 4, 5], T1, [5, 4, 2, 3]), [[0, 1], [2, 3]], nkeep=d
        )
        B = np.einsum(L, [0, 1, 4], R, [4, 2, 3])
        assert np.allclose(A, B)

    for _ in range(nt):
        w0 = np.random.random((1, 1)) + 1.0j * np.random.random((1, 1))
        t0 = np.random.random((1, 2, 3)) + 1.0j * np.random.random((1, 2, 3))
        t1 = np.random.random((3, 2, 2)) + 1.0j * np.random.random((3, 2, 2))
        w2 = np.random.random((2, 2)) + 1.0j * np.random.random((2, 2))
        p = np.random.random((2, 2, 2, 2)) + 1.0j * np.random.random((2, 2, 2, 2))
        B = np.einsum(t0, [0, 4, 6], t1, [6, 5, 3], p, [1, 2, 4, 5])
        p0, p1 = q.mps.svd.tensor_svd(p, [[0, 2], [1, 3]])
        assert np.allclose(np.einsum(p0, [0, 2, 4], p1, [4, 1, 3]), p)
        t0 = np.einsum(t0, [0, 4, 3], p0, [1, 4, 2])
        t1 = np.einsum(t1, [0, 4, 3], p1, [1, 2, 4])
        assert np.allclose(np.einsum(t0, [0, 1, 4, 5], t1, [5, 4, 2, 3]), B)
        tt0 = np.einsum(w0, [0, 4], t0, [4, 1, 2, 3])
        tt1 = np.einsum(t1, [0, 1, 2, 4], w2, [4, 3])
        _, WL, WR = _full_projector(tt0, [0, 1, 4, 5], tt1, [5, 4, 2, 3])
        tt0 = np.einsum(t0, [0, 1, 3, 4], WR, [3, 4, 2])
        tt1 = np.einsum(WL.conj(), [3, 4, 0], t1, [4, 3, 1, 2])
        A = np.einsum(tt0, [0, 1, 4], tt1, [4, 2, 3])
        print(np.linalg.norm(A - B))
        assert np.allclose(A, B)

        assert np.allclose(
            np.einsum(WL.conj(), [2, 3, 0], WR, [2, 3, 1]), np.identity(WL.shape[2])
        )
        n = WR.shape[0] * WR.shape[1]
        assert np.linalg.norm(
            np.einsum(WR, [0, 1, 4], WL.conj(), [2, 3, 4]).reshape((n, n))
            - np.identity(n)
        ) == approx(0, abs=1e-8)

    for _ in range(nt):
        w0 = np.random.random((1, 1)) + 1.0j * np.random.random((1, 1))
        t0 = np.random.random((1, 2, 3)) + 1.0j * np.random.random((1, 2, 3))
        t1 = np.random.random((3, 2, 2)) + 1.0j * np.random.random((3, 2, 2))
        w2 = np.random.random((2, 2)) + 1.0j * np.random.random((2, 2))
        p = np.random.random((2, 2, 2, 2)) + 1.0j * np.random.random((2, 2, 2, 2))
        B = np.einsum(t0, [0, 4, 6], t1, [6, 5, 3], p, [1, 2, 4, 5])
        p0, p1 = q.mps.svd.tensor_svd(p, [[0, 2], [1, 3]])
        assert np.allclose(np.einsum(p0, [0, 2, 4], p1, [4, 1, 3]), p)
        t0 = np.einsum(t0, [0, 4, 3], p0, [1, 4, 2])
        t1 = np.einsum(t1, [1, 4, 3], p1, [0, 2, 4])
        assert np.allclose(np.einsum(t0, [0, 1, 4, 5], t1, [4, 5, 2, 3]), B)
        tt0 = np.einsum(w0, [0, 4], t0, [4, 1, 2, 3])
        tt1 = np.einsum(t1, [0, 1, 2, 4], w2, [4, 3])
        _, WL, WR = _full_projector(tt0, [0, 1, 4, 5], tt1, [4, 5, 2, 3])
        tt0 = np.einsum(t0, [0, 1, 3, 4], WR, [3, 4, 2])
        tt1 = np.einsum(WL.conj(), [3, 4, 0], t1, [3, 4, 1, 2])
        A = np.einsum(tt0, [0, 1, 4], tt1, [4, 2, 3])
        print(np.linalg.norm(A - B))
        assert np.allclose(A, B)

        assert np.allclose(
            np.einsum(WL.conj(), [2, 3, 0], WR, [2, 3, 1]), np.identity(WL.shape[2])
        )
        n = WR.shape[0] * WR.shape[1]
        assert np.linalg.norm(
            np.einsum(WR, [0, 1, 4], WL.conj(), [2, 3, 4]).reshape((n, n))
            - np.identity(n)
        ) == approx(0, abs=1e-8)


if __name__ == "__main__":
    test_projector()
