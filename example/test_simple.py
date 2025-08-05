import numpy as np
import simple

import qailo as q


def test_simple():
    v = q.sv.zero(3)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))

    v = q.mps.zero(3)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))

    v = q.mps.zero(3, mps=q.mps.projector_mps)
    v = simple.main(v)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))
