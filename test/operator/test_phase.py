import qailo as q
from numpy import exp, pi


def test_p():
    assert q.op.is_close(q.op.p(pi), q.op.z())
    assert q.op.is_close(q.op.p(pi / 2), q.op.s())
    assert q.op.is_close(q.op.p(pi / 4), q.op.t())


def test_s():
    assert q.op.is_unitary(q.op.s())
    # S = T^2
    assert q.op.is_close(q.op.s(), q.op.multiply(q.op.t(), q.op.t(), [0]))


def test_t():
    assert q.op.is_unitary(q.op.t())
    # T ~ Rz(pi/4)
    assert q.op.is_close(q.op.t(), exp(1j * pi / 8) * q.op.rz(pi / 4))
