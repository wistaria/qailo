import qailo as q


def test_pure_state():
    n = 2
    sv = q.sv.state_vector(n)
    dm = q.sv.pure_state(sv)
    assert q.op.is_density_matrix(dm)

    for i in range(n):
        sv = q.sv.apply(sv, q.op.h(), [i])
    print(q.sv.vector(sv))
    dm = q.sv.pure_state(sv)
    print(q.op.matrix(dm))
    assert q.op.is_density_matrix(dm)
