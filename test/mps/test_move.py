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
    for mps in [q.mps.MPS_C, q.mps_p.MPS_P]:
        m = mps(tensors)
        q.mps.is_canonical(m)
        norm = q.mps.norm(m)
        v = q.sv.vector(q.mps.state_vector(m))
        for _ in range(64):
            s = np.random.randint(n - 1)
            print(f"swap tensors at {s} and {s+1}")
            m._canonicalize(s)
            _swap_tensors(m, s)
            print(q.sv.vector(q.mps.state_vector(m)))
            q.mps.is_canonical(m)
            assert q.mps.norm(m) == approx(norm)

        vn = q.sv.vector(q.mps.state_vector(m))
        assert len(v) == len(vn)
        assert q.sv.is_close(v, vn)


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
    for mps in [q.mps.MPS_C, q.mps_p.MPS_P]:
        m = mps(tensors)
        q.mps.is_canonical(m)
        norm = q.mps.norm(m)
        v = q.sv.vector(q.mps.state_vector(m))

        for _ in range(16):
            p = np.random.randint(n)
            s = np.random.randint(n)
            print(f"move qubit at {p} to {s}")
            q.mps.is_canonical(m)
            _move_qubit(m, p, s)
            print(q.sv.vector(q.mps.state_vector(m)))
            q.mps.is_canonical(m)
            assert q.mps.norm(m) == approx(norm)

        vn = q.sv.vector(q.mps.state_vector(m))
        assert len(v) == len(vn)
        assert q.sv.is_close(v, vn)


if __name__ == "__main__":
    test_swap()
    test_move()
