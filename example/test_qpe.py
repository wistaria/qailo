import numpy as np
import qailo as q
from pytest import approx
from qpe import qpe


def test_qpe():
    n = 3
    phi = 2 * np.pi * (1 / 8)
    u = q.op.p(phi)

    v = q.sv.zero()
    v = q.apply(v, q.op.x())
    v = q.sv.product_state([q.sv.zero(n), v])
    v = qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[4] == approx(1)
    assert prob[0] == approx(0)

    mps = q.mps.MPS_C
    v = q.mps.zero(1, mps)
    v = q.apply(v, q.op.x())
    v = q.mps.product_state([q.mps.zero(n, mps), v], mps)
    v = qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[4] == approx(1)
    assert prob[0] == approx(0)

    mps = q.mps.MPS_P
    v = q.mps.zero(1, mps)
    v = q.apply(v, q.op.x())
    v = q.mps.product_state([q.mps.zero(n, mps), v], mps)
    v = qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    assert prob[4] == approx(1)
    assert prob[0] == approx(0)


if __name__ == "__main__":
    test_qpe()
