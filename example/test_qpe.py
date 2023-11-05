import numpy as np
import qailo as q
import qpe
from pytest import approx


def test_qpe():
    n = 3
    phi = 2 * np.pi * (1 / 8)
    u = q.op.p(phi)
    ev = q.sv.state_vector(n + 1)
    ev = q.apply(ev, q.op.x(), [n])
    v = qpe.qpe(n, u, ev)
    prob = np.diag(q.op.matrix(q.op.trace(q.sv.pure_state(v), [n])).real)
    assert prob[4] == approx(1)
    assert prob[0] == approx(0)
