import qailo as q


def test_pure_state():
    n = 2
    sv = q.sv.zero(2)
    dm = q.op.pure_state(sv)
    print(q.op.matrix(dm))
    assert q.is_equal(q.op.trace(dm), 1)

    sv = q.sv.apply(q.op.h(), sv, [0])
    sv = q.sv.apply(q.op.h(), sv, [1])
    print(q.sv.vector(sv))
    dm = q.op.pure_state(sv)
    print(q.op.matrix(dm))
    assert q.is_equal(q.op.trace(dm), 1)
