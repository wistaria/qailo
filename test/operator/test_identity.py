import qailo as q

def test_identity():
    for n in range(6):
        op = q.op.identity(n)
        assert q.op.is_hermitian(op)
        assert q.op.is_unitary(op)
        assert q.op.is_identity(q.op.multiply(op, op, range(n)))
        assert q.op.is_identity(op)
