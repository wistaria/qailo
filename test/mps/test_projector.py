import numpy as np
import qailo as q


def test_projector():
    maxn = 4
    nt = 16
    for _ in range(nt):
        n0, n1, n2, d = np.random.randint(2, maxn, size=(4,))
        T0 = np.random.random((n0, n1))
        T1 = np.random.random((n1, n2))
        WL, WR = q.mps.projector(T0, [0, 2], T1, [2, 1], nkeep=d)
        assert WL.shape[1] <= d and WR.shape[1] <= d
        assert WL.shape[0] == n1 and WR.shape[0] == n1
        A = np.einsum(T0, [0, 2], WR, [2, 3], WL.conj().T, [3, 4], T1, [4, 1])

        S, U, V = q.mps.compact_svd(np.einsum(T0, [0, 2], T1, [2, 1]), nkeep=d)
        B = np.einsum("k,ik,jk->ij", S, U, V)
        assert np.allclose(A, B)

    for _ in range(nt):
        n0, n1, n2, n3, n4, n5, d = np.random.randint(2, maxn, size=(7,))
        T0 = np.random.random((n0, n1, n2, n3))
        T1 = np.random.random((n2, n3, n4, n5))
        WL, WR = q.mps.projector(T0, [0, 1, 4, 5], T1, [4, 5, 2, 3], nkeep=d)
        assert WL.shape[2] <= d and WR.shape[2] <= d
        assert WL.shape[0] == n2 and WL.shape[1] == n3
        assert WR.shape[0] == n2 and WR.shape[1] == n3
        A = np.einsum(
            T0, [0, 1, 4, 5], WR, [4, 5, 6], WL.conj(), [7, 8, 6], T1, [7, 8, 2, 3]
        )

        L, R = q.mps.tensor_svd(
            np.einsum(T0, [0, 1, 4, 5], T1, [4, 5, 2, 3]), [[0, 1], [2, 3]], nkeep=d
        )
        B = np.einsum(L, [0, 1, 4], R, [4, 2, 3])
        assert np.allclose(A, B)


if __name__ == "__main__":
    test_projector()
