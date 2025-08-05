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
    assert q.op.is_close(q.op.cz(), q.op.multiply(q.op.identity(2), q.op.cz(), [1, 0]))


def test_ccccx():
    for n in range(3, 8):
        p = q.op.identity(n)
        p = q.op.multiply(p, q.op.control_begin(), [0, 1])
        for i in range(1, n - 2):
            p = q.op.multiply(p, q.op.control_propagate(), [i, i + 1])
        p = q.op.multiply(p, q.op.control_end(q.op.x()), [n - 2, n - 1])
        assert q.op.is_close(p, q.op.cx(n))


def test_toffoli():
    for n in range(3, 8):
        p = q.op.identity(n)
        seq = q.op.toffoli_seq(range(n))
        for op, pos in seq:
            p = q.op.multiply(p, op, pos)
        assert q.op.is_close(p, q.op.cx(n))
