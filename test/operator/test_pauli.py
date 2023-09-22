import qailo as q

def test_x():
    op = q.op.x()
    assert q.op.is_hermitian(op)
    assert q.op.is_unitary(op)
    assert q.op.is_identity(q.op.multiply(op, op, [0]))

def test_y():
    op = q.op.y()
    assert q.op.is_hermitian(op)
    assert q.op.is_unitary(op)
    assert q.op.is_identity(q.op.multiply(op, op, [0]))

def test_z():
    op = q.op.z()
    assert q.op.is_hermitian(op)
    assert q.op.is_unitary(op)
    assert q.op.is_identity(q.op.multiply(op, op, [0]))

def test_xyz():
    # XY = -YX, etc
    assert q.is_equal(q.op.multiply(q.op.x(), q.op.y(), [0]), -1 * q.op.multiply(q.op.y(), q.op.x(), [0]))
    assert q.is_equal(q.op.multiply(q.op.y(), q.op.z(), [0]), -1 * q.op.multiply(q.op.z(), q.op.y(), [0]))
    assert q.is_equal(q.op.multiply(q.op.z(), q.op.x(), [0]), -1 * q.op.multiply(q.op.x(), q.op.z(), [0]))
