import numpy as np
import qailo as q
from pytest import approx


def test_fidelity():
    for n in range(1, 8):
        sv0 = q.sv.zero(n)
        sv1 = sv0
        for i in range(n):
            sv1 = q.sv.apply(sv1, q.op.h(), [i])
        assert q.sv.fidelity(sv0, sv0) == approx(1)
        assert q.sv.fidelity(sv1, sv1) == approx(1)
        assert q.sv.fidelity(sv0, sv1) == approx(1 / np.sqrt(2**n))
