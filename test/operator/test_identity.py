import qailo as q
from pytest import approx


def test_identity():
    for n in range(1, 6):
        op = q.op.identity(n)
        assert q.op.is_hermitian(op)
        assert q.op.is_unitary(op)
        assert q.op.is_identity(q.op.multiply(op, op, range(n)))
        assert q.op.is_identity(op)
        assert q.op.trace(op) == approx(2**n)
