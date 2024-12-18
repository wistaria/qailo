from grover import grover
from pytest import approx

import qailo as q


def test_grover():
    n = 4
    target = q.util.str2binary("0000")
    iter = 2 ** (n // 2)
    v = q.sv.zero(n)
    prob = q.probability(grover(v, target, iter))
    assert prob[0] == approx(0.581704139709473)
    assert prob[1] == approx(0.027886390686035)

    v = q.mps.zero(n)
    prob = q.probability(grover(v, target, iter))
    assert prob[0] == approx(0.581704139709473)
    assert prob[1] == approx(0.027886390686035)

    v = q.mps.zero(n, mps=q.mps.projector_mps)
    prob = q.probability(grover(v, target, iter))
    assert prob[0] == approx(0.581704139709473)
    assert prob[1] == approx(0.027886390686035)
