import qailo as q

def test_cx():
    assert q.op.is_hermitian(q.op.cx())
    assert q.op.is_unitary(q.op.cx())

def test_cz():
    assert q.op.is_hermitian(q.op.cz())
    assert q.op.is_unitary(q.op.cz())
    # Cz01 = Cz10
    assert q.is_equal(q.op.cz(), q.op.multiply(q.op.cz(), q.op.identity(2), [1, 0]))
