import numpy as np
import qailo as q


def test_rx():
    assert q.op.is_identity(q.op.rx(0))
    # Rx(2pi) = -I
    assert q.op.is_identity(-q.op.rx(2 * np.pi))
    assert q.op.is_unitary(q.op.rx(0.1))


def test_ry():
    assert q.op.is_identity(q.op.ry(0))
    assert q.op.is_identity(-q.op.ry(2 * np.pi))
    assert q.op.is_unitary(q.op.ry(0.1))


def test_rz():
    assert q.op.is_identity(q.op.rz(0))
    assert q.op.is_identity(-q.op.rz(2 * np.pi))
    assert q.op.is_unitary(q.op.rz(0.1))
