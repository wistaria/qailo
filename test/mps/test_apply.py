from pytest import approx
import random

import qailo as q


def apply(op, m, v, pos):
    m = q.mps.apply(op, m, pos)
    v = q.sv.apply(op, v, pos)
    return m, v


def test_apply():
    n = 8
    p = 100

    m = q.mps.mps(n)
    v = q.sv.state_vector(n)

    for _ in range(p):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            t = random.randrange(3)
            if t == 0:
                print("apply h on {}".format(i))
                m, v = apply(q.op.h(), m, v, [i])
            elif t == 1:
                print("apply x on {}".format(i))
                m, v = apply(q.op.s(), m, v, [i])
            elif t == 2:
                print("apply s on {}".format(i))
                m, v = apply(q.op.t(), m, v, [i])
        else:
            t = random.randrange(3)
            if t == 0:
                print("apply cx on {} and {}".format(i, j))
                m, v = apply(q.op.cx(), m, v, [i, j])
            elif t == 1:
                print("apply cz on {} and {}".format(i, j))
                m, v = apply(q.op.cz(), m, v, [i, j])
    print(q.sv.vector(q.mps.state_vector(m)))
    print(q.sv.vector(v))

    for i in range(n):
        print("shape {}: {}".format(i, m[1][i].shape))

    assert q.sv.fidelity(q.mps.state_vector(m), v) == approx(1)


if __name__ == "__main__":
    test_apply()
