import numpy as np
import qailo as q
from pytest import approx


def test_canonical():
    n = 16
    maxdim = 4
    tensors = []
    d = np.random.randint(2, maxdim)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, maxdim)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    for mps in [q.mps.MPS_C, q.mps_p.MPS_P]:
        m = mps(tensors)
        norm = q.mps.norm(m)
        assert q.mps.norm(m) == approx(norm)
        assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n)
            m._canonicalize(p)
            assert q.mps.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)
            assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n - 1)
            m._canonicalize(p, p + 1)
            assert q.mps.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)

    v = np.random.random(2**n).reshape((2,) * n + (1,))
    v /= np.linalg.norm(v)
    tensors = q.mps.tensor_decomposition(v, maxdim)
    for mps in [q.mps.MPS_C, q.mps_p.MPS_P]:
        m = mps(tensors)
        norm = q.mps.norm(m)

        for _ in range(n):
            p = np.random.randint(n)
            m._canonicalize(p)
            assert q.mps.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n - 1)
            m._canonicalize(p, p + 1)
            assert q.mps.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)


if __name__ == "__main__":
    test_canonical()
