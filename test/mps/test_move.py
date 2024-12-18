import numpy as np
from pytest import approx

import qailo as q
from qailo.mps.apply import _move_qubit, _swap_tensors


def test_swap():
    np.random.seed(1234)
    n = 12
    nkeep = 4
    tensors = []
    d = np.random.randint(2, nkeep)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, nkeep)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    for mps in [q.mps.canonical_mps, q.mps.projector_mps]:
        m = mps(tensors)
        q.mps.is_canonical(m)
        norm = q.norm(m)
        v = q.sv.vector(q.mps.state_vector(m))
        for _ in range(64):
            s = np.random.randint(n - 1)
            print(f"swap tensors at {s} and {s+1}")
            m._canonicalize(s)
            _swap_tensors(m, s)
            print(q.sv.vector(q.mps.state_vector(m)))
            q.mps.is_canonical(m)
            assert q.norm(m) == approx(norm)

        vn = q.sv.vector(q.mps.state_vector(m))
        assert len(v) == len(vn)
        assert q.sv.is_close(v, vn)


def test_move():
    np.random.seed(1234)
    # n = 12
    n = 4
    nkeep = 4
    tensors = []
    d = np.random.randint(2, nkeep)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, nkeep)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    for mps in [q.mps.canonical_mps, q.mps.projector_mps]:
        m = mps(tensors)
        q.mps.is_canonical(m)
        norm = q.norm(m)
        v = q.sv.vector(q.mps.state_vector(m))

        for _ in range(16):
            p = np.random.randint(n)
            s = np.random.randint(n)
            print(f"move qubit at {p} to {s}")
            q.mps.is_canonical(m)
            _move_qubit(m, p, s)
            print(q.sv.vector(q.mps.state_vector(m)))
            q.mps.is_canonical(m)
            assert q.norm(m) == approx(norm)

        vn = q.sv.vector(q.mps.state_vector(m))
        assert len(v) == len(vn)
        assert q.sv.is_close(v, vn)


if __name__ == "__main__":
    test_swap()
    test_move()
