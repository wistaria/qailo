import numpy as np
import qailo as q
from pytest import approx


def test_swap():
    n = 12
    maxdim = 4
    tensors = []
    d = np.random.randint(2, maxdim)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, maxdim)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    mps = q.mps.MPS(tensors, normalize=True)
    q.mps.check(mps)
    v0 = q.sv.vector(q.mps.state_vector(mps))

    for _ in range(64):
        s = np.random.randint(n - 1)
        print(f"swap tensors at {s} and {s+1}")
        mps.canonicalize(s)
        mps._swap_tensors(s)
        assert q.mps.norm(mps) == approx(1)
        assert q.mps.is_canonical(mps)

    v1 = q.sv.vector(q.mps.state_vector(mps))
    assert len(v0) == len(v1)
    print(v0)
    print(v1)
    assert q.sv.is_close(v0, v1)


def test_move():
    n = 12
    maxdim = 4
    tensors = []
    d = np.random.randint(2, maxdim)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, maxdim)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    mps = q.mps.MPS(tensors, normalize=True)
    q.mps.check(mps)
    v0 = q.sv.vector(q.mps.state_vector(mps))

    for _ in range(16):
        p = np.random.randint(n)
        s = np.random.randint(n)
        mps._move_qubit(p, s)
        assert q.mps.norm(mps) == approx(1)
        assert q.mps.is_canonical(mps)

    v1 = q.sv.vector(q.mps.state_vector(mps))
    assert len(v0) == len(v1)
    print(v0)
    print(v1)
    assert q.sv.is_close(v0, v1)


if __name__ == "__main__":
    test_swap()
    test_move()
