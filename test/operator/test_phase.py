import numpy as np
import qailo as q


def test_s():
    assert q.op.is_unitary(q.op.s())
    # S = T^2
    assert q.op.is_close(q.op.s(), q.op.multiply(q.op.t(), q.op.t(), [0]))


def test_t():
    assert q.op.is_unitary(q.op.t())
    # T ~ Rz(pi/4)
    assert q.op.is_close(q.op.t(), np.exp(1j * np.pi / 8) * q.op.rz(np.pi / 4))
