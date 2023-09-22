import qailo as q

def test_identity():
    for n in range(6):
        op = q.operator.identity(n)
        assert q.operator.is_hermitian(op)
        assert q.operator.is_unitary(op)
        assert q.operator.is_identity(q.operator.multiply(op, op, range(n)))
        assert q.operator.is_identity(op)
