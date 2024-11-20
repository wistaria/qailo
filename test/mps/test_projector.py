import numpy as np

import qailo as q
from qailo.mps.svd import compact_svd, tensor_svd


def test_projector():
    maxn = 4
    nt = 16
    for _ in range(nt):
        n0, n1, n2, d = np.random.randint(2, maxn, size=(4,))
        T0 = np.random.random((n0, n1))
        T1 = np.random.random((n1, n2))
        _, PLh, PR = q.mps.projector(T0, [0, 2], T1, [2, 1], nkeep=d)
        assert PLh.shape[1] <= d and PR.shape[1] <= d
        assert PLh.shape[0] == n1 and PR.shape[0] == n1
        A = np.einsum(T0, [0, 2], PR, [2, 3], PLh.T, [3, 4], T1, [4, 1])

        S, U, V = compact_svd(np.einsum(T0, [0, 2], T1, [2, 1]), nkeep=d)
        B = np.einsum("k,ik,jk->ij", S, U, V)
        assert np.allclose(A, B)

    for _ in range(nt):
        n0, n1, n2, n3, n4, n5, d = np.random.randint(2, maxn, size=(7,))
        T0 = np.random.random((n0, n1, n2, n3))
        T1 = np.random.random((n3, n2, n4, n5))
        _, PLh, PR = q.mps.projector(T0, [0, 1, 4, 5], T1, [5, 4, 2, 3], nkeep=d)
        assert PLh.shape[2] <= d and PR.shape[2] <= d
        assert PLh.shape[0] == n2 and PLh.shape[1] == n3
        assert PR.shape[0] == n2 and PR.shape[1] == n3
        A = np.einsum(T0, [0, 1, 4, 5], PR, [4, 5, 6], PLh, [7, 8, 6], T1, [8, 7, 2, 3])

        L, R = tensor_svd(
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
        _, PLh, PR = q.mps.projector(tt0, [0, 1, 4, 5], tt1, [5, 4, 2, 3])
        assert np.allclose(
            np.einsum(PLh, [2, 3, 0], PR, [2, 3, 1]), np.identity(PLh.shape[2])
        )
        tt0 = np.einsum(t0, [0, 1, 3, 4], PR, [3, 4, 2])
        tt1 = np.einsum(PLh, [3, 4, 0], t1, [4, 3, 1, 2])
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
        _, PLh, PR = q.mps.projector(tt0, [0, 1, 4, 5], tt1, [4, 5, 2, 3])
        assert np.allclose(
            np.einsum(PLh, [2, 3, 0], PR, [2, 3, 1]), np.identity(PLh.shape[2])
        )
        tt0 = np.einsum(t0, [0, 1, 3, 4], PR, [3, 4, 2])
        tt1 = np.einsum(PLh, [3, 4, 0], t1, [3, 4, 1, 2])
        A = np.einsum(tt0, [0, 1, 4], tt1, [4, 2, 3])
        print(np.linalg.norm(A - B))
        assert np.allclose(A, B)


if __name__ == "__main__":
    test_projector()
