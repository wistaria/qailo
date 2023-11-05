import numpy as np
import qailo as q
import simple


def test_simple():
    v = q.sv.zero(3)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))

    v = q.mps.zero(3, q.mps.MPS_C)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))

    v = q.mps.zero(3, q.mps.MPS_P)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))
