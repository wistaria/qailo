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
    m0 = q.mps.MPS_C(tensors)
    m1 = q.mps_p.MPS_P(tensors)
    norm = q.mps.norm(m0)
    assert q.mps.norm(m0) == approx(norm)
    assert q.mps.norm(m1) == approx(norm)
    assert q.mps.is_canonical(m0)
    assert q.mps.is_canonical(m1)

    for _ in range(n):
        p = np.random.randint(n)
        m0._canonicalize(p)
        m1._canonicalize(p)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)
        assert q.mps.is_canonical(m0)
        assert q.mps.is_canonical(m1)

    for _ in range(n):
        p = np.random.randint(n - 1)
        m0._canonicalize(p, p + 1)
        m1._canonicalize(p, p + 1)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)
        assert q.mps.is_canonical(m0)
        assert q.mps.is_canonical(m1)

    v = np.random.random(2**n).reshape((2,) * n + (1,))
    v /= np.linalg.norm(v)
    tensors = q.mps.tensor_decomposition(v, maxdim)
    m0 = q.mps.MPS_C(tensors)
    m1 = q.mps_p.MPS_P(tensors)
    norm = q.mps.norm(m0)

    for _ in range(n):
        p = np.random.randint(n)
        m0._canonicalize(p)
        m1._canonicalize(p)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)
        assert q.mps.is_canonical(m0)
        assert q.mps.is_canonical(m1)

    for _ in range(n):
        p = np.random.randint(n - 1)
        m0._canonicalize(p, p + 1)
        m1._canonicalize(p, p + 1)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)
        assert q.mps.is_canonical(m0)
        assert q.mps.is_canonical(m1)


if __name__ == "__main__":
    test_canonical()
