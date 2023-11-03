import qailo as q


def test_swap():
    p = q.op.swap()
    assert q.op.is_hermitian(p)
    assert q.op.is_unitary(p)
    assert q.op.is_identity(q.op.multiply(p, p, [0, 1]))

    p = q.op.cx()
    p = q.op.multiply(p, q.op.cx(), [1, 0])
    p = q.op.multiply(p, q.op.cx(), [0, 1])
    assert q.op.is_close(p, q.op.swap())

    v = q.sv.state_vector(2)
    v = q.sv.apply(v, q.op.h(), [0])
    v = q.sv.apply(v, q.op.swap(), [0, 1])
    v = q.sv.apply(v, q.op.h(), [1])
    v = q.sv.apply(v, q.op.swap(), [0, 1])
    assert q.op.is_close(v, q.sv.state_vector(2))
