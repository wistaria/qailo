import numpy as np
import qailo as q
from pytest import approx


def test_fidelity():
    for n in range(8):
        sv0 = q.sv.state_vector(n)
        sv1 = sv0
        for i in range(n):
            sv1 = q.sv.apply(q.op.h(), sv1, [i])
        assert q.sv.fidelity(sv0, sv0) == approx(1)
        assert q.sv.fidelity(sv1, sv1) == approx(1)
        assert q.sv.fidelity(sv0, sv1) == approx(1 / np.sqrt(2**n))
