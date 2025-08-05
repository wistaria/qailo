from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pytest import approx

import qailo as q
from qailo.mps.type import mps as mpstype


def test_canonical():
    n = 16
    nkeep = 4
    tensors: list[npt.NDArray] = []
    d = np.random.randint(2, nkeep)
    tensors.append(np.random.random((1, 2, d)))
    for _ in range(n - 2):
        dn = np.random.randint(2, nkeep)
        tensors.append(np.random.random((d, 2, dn)))
        d = dn
    tensors.append(np.random.random((d, 2, 1)))
    for mps in [q.mps.canonical_mps, q.mps.projector_mps]:
        m = mps(tensors)
        assert isinstance(m, mpstype)
        norm = q.norm(m)
        assert q.norm(m) == approx(norm)
        assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n)
            m._canonicalize(p)
            assert q.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)
            assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n - 1)
            m._canonicalize(p, p + 1)
            assert q.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)

    v = np.random.random(2**n).reshape((2,) * n + (1,))
    v /= np.linalg.norm(v)
    tensors = q.mps.tensor_decomposition(v, nkeep)
    for mps in [q.mps.canonical_mps, q.mps.projector_mps]:
        m = mps(tensors)
        assert isinstance(m, mpstype)
        norm = q.norm(m)

        for _ in range(n):
            p = np.random.randint(n)
            m._canonicalize(p)
            assert q.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)

        for _ in range(n):
            p = np.random.randint(n - 1)
            m._canonicalize(p, p + 1)
            assert q.norm(m) == approx(norm)
            assert q.mps.is_canonical(m)


if __name__ == "__main__":
    test_canonical()
