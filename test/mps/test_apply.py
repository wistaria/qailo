import random

import numpy as np
from pytest import approx

import qailo as q


def apply(op, m0, m1, m2, m3, v, seq, pos):
    m0 = q.mps.apply(m0, op, pos)
    m1 = q.mps.apply(m1, op, pos)
    m2 = q.mps.apply(m2, op, pos)
    m3 = q.mps.apply(m3, op, pos)
    v = q.sv.apply(v, op, pos)
    seq.append([op, pos])
    return m0, m1, m2, m3, v, seq


def test_apply():
    n = 8
    p = 64
    nkeep = 2

    m0 = q.mps.zero(n, mps=q.mps.canonical_mps)
    m1 = q.mps.zero(n, mps=q.mps.projector_mps)
    m2 = q.mps.zero(n, nkeep=nkeep, mps=q.mps.canonical_mps)
    m3 = q.mps.zero(n, nkeep=nkeep, mps=q.mps.projector_mps)
    v = q.sv.zero(n)
    seq = []

    i = 4
    j = 0
    print("apply cz on {} and {}".format(i, j))
    m0, m1, m2, m3, v, seq = apply(q.op.cz(), m0, m1, m2, m3, v, seq, [i, j])

    for _ in range(p):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            t = random.randrange(3)
            if t == 0:
                print("apply h on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(q.op.h(), m0, m1, m2, m3, v, seq, [i])
            elif t == 1:
                print("apply x on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(q.op.s(), m0, m1, m2, m3, v, seq, [i])
            elif t == 2:
                print("apply s on {}".format(i))
                m0, m1, m2, m3, v, seq = apply(q.op.t(), m0, m1, m2, m3, v, seq, [i])
        else:
            t = random.randrange(2)
            if t == 0:
                print("apply cx on {} and {}".format(i, j))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.cx(), m0, m1, m2, m3, v, seq, [i, j]
                )
            elif t == 1:
                print("apply cz on {} and {}".format(i, j))
                m0, m1, m2, m3, v, seq = apply(
                    q.op.cz(), m0, m1, m2, m3, v, seq, [i, j]
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

    f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
    f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
    f2 = q.sv.fidelity(q.mps.state_vector(m2), v)
    f3 = q.sv.fidelity(q.mps.state_vector(m3), v)
    print("final fidelity = {} {} {} {}".format(f0, f1, f2, f3))
    assert f0 == approx(1)
    assert f1 == approx(1)

    m4 = q.mps.zero(n, mps=q.mps.projector_mps)
    f4 = q.sv.fidelity(q.mps.state_vector(q.mps.apply_seq(m4, seq)), v)
    assert f4 == approx(1)


def test_toffoli():
    n = 8
    p = 32
    nkeep = 2

    m0 = q.mps.zero(n, mps=q.mps.canonical_mps)
    m1 = q.mps.zero(n, mps=q.mps.projector_mps)
    m2 = q.mps.zero(n, nkeep=nkeep, mps=q.mps.canonical_mps)
    m3 = q.mps.zero(n, nkeep=nkeep, mps=q.mps.projector_mps)
    v = q.sv.zero(n)
    seq = []

    for i in range(n):
        m0, m1, m2, m3, v, seq = apply(q.op.h(), m0, m1, m2, m3, v, seq, [i])

    for _ in range(p):
        k = random.randrange(3, n)
        pos = np.random.permutation(n)[:k].tolist()
        print("apply toffoli on {}".format(pos))
        toffoli = q.op.toffoli_seq(pos)
        m0 = q.mps.apply_seq(m0, toffoli)
        m1 = q.mps.apply_seq(m1, toffoli)
        m2 = q.mps.apply_seq(m2, toffoli)
        m3 = q.mps.apply_seq(m3, toffoli)
        v = q.sv.apply(v, q.op.cx(k), pos)
        seq += toffoli
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

    f0 = q.sv.fidelity(q.mps.state_vector(m0), v)
    f1 = q.sv.fidelity(q.mps.state_vector(m1), v)
    f2 = q.sv.fidelity(q.mps.state_vector(m2), v)
    f3 = q.sv.fidelity(q.mps.state_vector(m3), v)
    print("final fidelity = {} {} {} {}".format(f0, f1, f2, f3))
    assert f0 == approx(1)
    assert f1 == approx(1)

    m4 = q.mps.zero(n, mps=q.mps.projector_mps)
    f4 = q.sv.fidelity(q.mps.state_vector(q.mps.apply_seq(m4, seq)), v)
    assert f4 == approx(1)


if __name__ == "__main__":
    test_apply()
    test_toffoli()
