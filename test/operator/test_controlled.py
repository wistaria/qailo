from pytest import approx

import qailo as q


def test_cx():
    print(q.op.cx())
    print(q.op.controlled(q.op.x()))
    print(q.op.matrix(q.op.controlled(q.op.x())))
    assert q.op.is_hermitian(q.op.cx())
    assert q.op.is_unitary(q.op.cx())
    assert q.op.cx() == approx(q.op.controlled(q.op.x()))


def test_cz():
    assert q.op.is_hermitian(q.op.cz())
    assert q.op.is_unitary(q.op.cz())
    # Cz01 = Cz10
    assert q.op.cz() == approx(q.op.multiply(q.op.cz(), q.op.identity(2), [1, 0]))
