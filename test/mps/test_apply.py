import random

import qailo as q
from pytest import approx


def apply(op, m0, m1, m2, m3, v, seq, pos, maxdim=None):
    m0 = q.mps.apply(m0, op, pos)
    m1 = q.mps.apply(m1, op, pos)
    m2 = q.mps.apply(m2, op, pos)
    m3 = q.mps.apply(m3, op, pos, maxdim)
    v = q.sv.apply(v, op, pos)
    seq.append([op, pos])
    return m0, m1, m2, m3, v, seq


def test_apply():
    n = 8
    p = 64
    maxdim = 2

    m0 = q.mps.zero(n, q.mps.MPS_C)
    m1 = q.mps.zero(n, q.mps_p.MPS_P)
    m2 = q.mps.zero(n, q.mps.MPS_C)
    m3 = q.mps.zero(n, q.mps_p.MPS_P)
    v = q.sv.zero(n)
    seq = []

    i = 4
    j = 0
    print("apply cz on {} and {}".format(i, j))
    m0, m1, m2, m3, v, seq = apply(q.op.cz(), m0, m1, m2, m3, v, seq, [i, j], maxdim)

    for _ in range(p):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            t = random.randrange(3)
            if t == 0:
                print("apply h on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.h(), m0, m1, m2, m3, v, seq, [i], maxdim
                )
            elif t == 1:
                print("apply x on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.s(), m0, m1, m2, m3, v, seq, [i], maxdim
                )
            elif t == 2:
                print("apply s on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.t(), m0, m1, m2, m3, v, seq, [i], maxdim
                )
        else:
            t = random.randrange(2)
            if t == 0:
                print("apply cx on {} and {}".format(i, j))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.cx(), m0, m1, m2, m3, v, seq, [i, j], maxdim
                )
            elif t == 1:
                print("apply cz on {} and {}".format(i, j))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.cz(), m0, m1, m2, m3, v, seq, [i, j], maxdim
                )
        m0._is_canonical()
        m1._is_canonical()
        m2._is_canonical()
        m3._is_canonical()
        f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
        f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
        f2 = q.sv.fidelity(q.mps.state_vector(m2), v)
        f3 = q.sv.fidelity(q.mps.state_vector(m3), v)
        print("fidelity = {} {} {} {}".format(f0, f1, f2, f3))
        assert f0 == approx(1)
        assert f1 == approx(1)
        assert f2 == approx(1)

    f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
    f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
    f2 = q.sv.fidelity(q.mps.state_vector(m2), v)
    f3 = q.sv.fidelity(q.mps.state_vector(m3), v)
    print("final fidelity = {} {} {} {}".format(f0, f1, f2, f3))
    assert f0 == approx(1)
    assert f1 == approx(1)
    assert f2 == approx(1)
    # assert f2 == approx(f3)

    m4 = q.mps.zero(n, q.mps_p.MPS_P)
    f4 = q.sv.fidelity(q.mps.state_vector(q.mps.apply_seq(m4, seq)), v)
    assert f4 == approx(1)


if __name__ == "__main__":
    test_apply()
