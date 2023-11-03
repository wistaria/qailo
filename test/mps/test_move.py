import numpy as np
import qailo as q
from pytest import approx
from qailo.mps.apply import _move_qubit, _swap_tensors


def test_swap():
    np.random.seed(1234)
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
    m0 = q.mps.MPS_C(tensors)
    m1 = q.mps.MPS_P(tensors)
    q.mps.is_canonical(m0)
    q.mps.is_canonical(m1)
    norm = q.mps.norm(m0)
    v = q.sv.vector(q.mps.state_vector(m0))

    for _ in range(64):
        s = np.random.randint(n - 1)
        print(f"swap tensors at {s} and {s+1}")
        m0._canonicalize(s)
        m1._canonicalize(s)
        _swap_tensors(m0, s)
        _swap_tensors(m1, s)
        print(q.sv.vector(q.mps.state_vector(m0)))
        print(q.sv.vector(q.mps.state_vector(m1)))
        q.mps.is_canonical(m0)
        q.mps.is_canonical(m1)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)

    v0 = q.sv.vector(q.mps.state_vector(m0))
    v1 = q.sv.vector(q.mps.state_vector(m1))
    assert len(v) == len(v0)
    assert len(v) == len(v1)
    print(v)
    print(v0)
    print(v1)
    assert q.sv.is_close(v, v0)
    assert q.sv.is_close(v, v1)


def test_move():
    np.random.seed(1234)
    # n = 12
    n = 4
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
    m1 = q.mps.MPS_P(tensors)
    q.mps.is_canonical(m0)
    q.mps.is_canonical(m1)
    norm = q.mps.norm(m0)
    v = q.sv.vector(q.mps.state_vector(m0))
    print(q.sv.vector(q.mps.state_vector(m0)))
    print(q.sv.vector(q.mps.state_vector(m1)))

    for _ in range(16):
        p = np.random.randint(n)
        s = np.random.randint(n)
        print(f"move qubit at {p} to {s}")
        q.mps.is_canonical(m0)
        q.mps.is_canonical(m1)
        _move_qubit(m0, p, s)
        _move_qubit(m1, p, s)
        print(q.sv.vector(q.mps.state_vector(m0)))
        print(q.sv.vector(q.mps.state_vector(m1)))
        q.mps.is_canonical(m0)
        q.mps.is_canonical(m1)
        assert q.mps.norm(m0) == approx(norm)
        assert q.mps.norm(m1) == approx(norm)

    v0 = q.sv.vector(q.mps.state_vector(m0))
    v1 = q.sv.vector(q.mps.state_vector(m1))
    assert len(v) == len(v0)
    assert len(v) == len(v1)
    print(v)
    print(v0)
    print(v1)
    assert q.sv.is_close(v, v0)
    assert q.sv.is_close(v, v1)


if __name__ == "__main__":
    test_swap()
    test_move()
