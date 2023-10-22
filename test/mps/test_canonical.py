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
    mps = q.mps.mps(tensors, range(n), range(n))

    # normalize
    norm = q.mps.norm(mps)
    tensors[0] /= norm
    mps = q.mps.mps(tensors, range(n), range(n))

    assert q.mps.norm(mps) == approx(1)

    mps.canonicalize(n - 1)
    assert q.mps.check_canonical(mps)
    mps.canonicalize(3)
    assert q.mps.check_canonical(mps)
    for _ in range(16):
        p = np.random.randint(n)
        mps.canonicalize(p)
        assert q.mps.norm(mps) == approx(1)
        assert q.mps.check_canonical(mps)


if __name__ == "__main__":
    test_canonical()
