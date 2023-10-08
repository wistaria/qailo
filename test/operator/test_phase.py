import numpy as np
from pytest import approx

import qailo as q


def test_s():
    assert q.op.is_unitary(q.op.s())
    # S = T^2
    assert q.op.s() == approx(q.op.multiply(q.op.t(), q.op.t(), [0]))


def test_t():
    assert q.op.is_unitary(q.op.t())
    # T ~ Rz(pi/4)
    assert q.op.t() == approx(np.exp(1j * np.pi / 8) * q.op.rz(np.pi / 4))
