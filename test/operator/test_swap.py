import qailo as q


def test_swap():
    op = q.op.swap()
    assert q.op.is_hermitian(op)
    assert q.op.is_unitary(op)
    assert q.op.is_identity(q.op.multiply(op, op, [0, 1]))

    op = q.op.cx()
    op = q.op.multiply(q.op.cx(), op, [1, 0])
    op = q.op.multiply(q.op.cx(), op, [0, 1])
    assert q.is_equal(op, q.op.swap())

    v = q.sv.zero(2)
    v = q.sv.apply(q.op.h(), v, [0])
    v = q.sv.apply(q.op.swap(), v, [0, 1])
    v = q.sv.apply(q.op.h(), v, [1])
    v = q.sv.apply(q.op.swap(), v, [0, 1])
    assert q.is_equal(v, q.sv.zero(2))
