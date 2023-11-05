import numpy as np
import qailo as q
import simple


def test_simple():
    v = simple.main(use_mps=False)
    assert np.allclose(q.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    assert np.allclose(q.vector(v, [1, 1, 1]), 1 / np.sqrt(2))

    v = simple.main(use_mps=True)
    # assert np.allclose(v.vector(v, [0, 0, 0]), 1 / np.sqrt(2))
    # assert np.allclose(v.vector(v, [1, 1, 1]), 1 / np.sqrt(2))
