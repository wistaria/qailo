import qailo as q


def test_density_matrix():
    n = 2
    sv = q.sv.product_state([q.sv.zero()] * n)
    dm = q.sv.density_matrix(sv)
    assert q.op.is_density_matrix(dm)

    for i in range(n):
        sv = q.sv.apply(sv, q.op.h(), [i])
    print(q.sv.vector(sv))
    dm = q.sv.density_matrix(sv)
    print(q.op.matrix(dm))
    assert q.op.is_density_matrix(dm)
