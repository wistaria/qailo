import numpy as np
from pytest import approx

import qailo as q


def test_qpe():
    n = 3
    phi = 2 * np.pi * 0.7
    u = q.op.p(phi)

    v = q.sv.zero(n + 1)
    v = q.apply(v, q.op.x(), [n])
    v = q.alg.qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[0] == approx(0.02159321892578291)
    assert prob[3] == approx(0.5775210180698607)

    v = q.mps.zero(n + 1)
    v = q.apply(v, q.op.x(), [n])
    v = q.alg.qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[0] == approx(0.02159321892578291)
    assert prob[3] == approx(0.5775210180698607)

    v = q.mps.zero(n + 1, mps=q.mps.projector_mps)
    v = q.apply(v, q.op.x(), [n])
    v = q.alg.qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[0] == approx(0.02159321892578291)
    assert prob[3] == approx(0.5775210180698607)


if __name__ == "__main__":
    test_qpe()
