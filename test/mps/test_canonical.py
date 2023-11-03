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
    mps = q.mps.MPS(tensors)
    norm = q.mps.norm(mps)
    assert q.mps.check_mps(mps)

    for _ in range(n):
        p = np.random.randint(n)
        mps.canonicalize(p)
        assert q.mps.norm(mps) == approx(norm)
        assert q.mps.is_canonical(mps)

    for _ in range(n):
        p = np.random.randint(n - 1)
        mps.canonicalize(p, p + 1)
        assert q.mps.norm(mps) == approx(norm)
        assert q.mps.is_canonical(mps)


if __name__ == "__main__":
    test_canonical()
