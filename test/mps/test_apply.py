from pytest import approx
import random

import qailo as q


def apply(op, m0, m1, v, pos, maxdim=None):
    m0 = q.mps.apply(op, m0, pos)
    m1 = q.mps.apply(op, m1, pos, maxdim)
    v = q.sv.apply(op, v, pos)
    return m0, m1, v


def test_apply():
    n = 8
    p = 100
    maxdim = 4

    m0 = q.mps.mps(n)
    m1 = q.mps.mps(n)
    v = q.sv.state_vector(n)

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
            t = random.randrange(3)
            if t == 0:
                print("apply cx on {} and {}".format(i, j))
                m0, m1, v = apply(q.op.cx(), m0, m1, v, [i, j], maxdim)
            elif t == 1:
                print("apply cz on {} and {}".format(i, j))
                m0, m1, v = apply(q.op.cz(), m0, m1, v, [i, j], maxdim)
    print(q.sv.vector(q.mps.state_vector(m0)))
    print(q.sv.vector(q.mps.state_vector(m1)))
    print(q.sv.vector(v))

    for i in range(n):
        print("shape {}: {}".format(i, m0[3][i].shape))
    for i in range(n):
        print("shape {}: {}".format(i, m1[3][i].shape))

    f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
    f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
    print("fidelity = {} and {}".format(f0, f1))
    assert f0 == approx(1)


if __name__ == "__main__":
    test_apply()