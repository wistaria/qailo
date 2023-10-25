import qailo as q


def test_cx():
    print(q.op.cx())
    print(q.op.controlled(q.op.x()))
    print(q.op.matrix(q.op.controlled(q.op.x())))
    assert q.op.is_hermitian(q.op.cx())
    assert q.op.is_unitary(q.op.cx())
    assert q.op.is_close(q.op.cx(), q.op.controlled(q.op.x()))


def test_cz():
    assert q.op.is_hermitian(q.op.cz())
    assert q.op.is_unitary(q.op.cz())
    # Cz01 = Cz10
    assert q.op.is_close(q.op.cz(), q.op.multiply(q.op.cz(), q.op.identity(2), [1, 0]))


def test_ccccx():
    for n in range(3, 8):
        op = q.op.identity(n)
        op = q.op.multiply(q.op.control_begin(), op, [0, 1])
        for i in range(1, n - 2):
            op = q.op.multiply(q.op.control_propagate(), op, [i, i + 1])
        op = q.op.multiply(q.op.control_end(q.op.x()), op, [n - 2, n - 1])
        assert q.op.is_close(op, q.op.cx(n))
