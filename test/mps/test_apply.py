import random

import qailo as q
from pytest import approx


def apply(op, m0, m1, v, pos, maxdim=None):
    m0 = q.mps.apply(m0, op, pos)
    m1 = q.mps.apply(m1, op, pos, maxdim)
    v = q.sv.apply(op, v, pos)
    return m0, m1, v


def test_apply():
    n = 8
    p = 64
    maxdim = 2

    m0 = q.mps.MPS(q.mps.product_state(n))
    m1 = q.mps.MPS(q.mps.product_state(n))
    v = q.sv.state_vector(n)

    i = 4
    j = 0
    print("apply cz on {} and {}".format(i, j))
    m0, m1, v = apply(q.op.cz(), m0, m1, v, [i, j], maxdim)

    for _ in range(p):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            t = random.randrange(3)
            if t == 0:
                print("apply h on {}".format(i))
                m0, m1, v = apply(q.op.h(), m0, m1, v, [i], maxdim)
            elif t == 1:
                print("apply x on {}".format(i))
                m0, m1, v = apply(q.op.s(), m0, m1, v, [i], maxdim)
            elif t == 2:
                print("apply s on {}".format(i))
                m0, m1, v = apply(q.op.t(), m0, m1, v, [i], maxdim)
        else:
            t = random.randrange(2)
            if t == 0:
                print("apply cx on {} and {}".format(i, j))
                m0, m1, v = apply(q.op.cx(), m0, m1, v, [i, j], maxdim)
            elif t == 1:
                print("apply cz on {} and {}".format(i, j))
                m0, m1, v = apply(q.op.cz(), m0, m1, v, [i, j], maxdim)

    f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
    f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
    print("final fidelity = {} and {}".format(f0, f1))
    assert f0 == approx(1)


if __name__ == "__main__":
    test_apply()
