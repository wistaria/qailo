import numpy as np
import qailo as q
from pytest import approx


def test_zero_one():
    assert q.sv.zero()[0] == approx(1)
    assert q.sv.zero()[1] == approx(0)
    assert q.sv.one()[0] == approx(0)
    assert q.sv.one()[1] == approx(1)
    assert np.allclose(q.sv.apply(q.sv.zero(), q.op.x()), q.sv.one())


def test_product_state():
    n = 8
    v0 = q.sv.product_state([q.sv.zero()] * n)
    v1 = q.sv.product_state([q.sv.one()] * n)
    assert np.allclose(v0, q.sv.zero(n))
    assert np.allclose(v1, q.sv.one(n))
    for p in range(n):
        v1 = q.sv.apply(v1, q.op.x(), [p])
    assert np.allclose(v0, v1)
