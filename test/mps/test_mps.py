import numpy as np
import qailo as q


def test_mps():
    n = 4
    c = q.util.str2binary("1100")
    m0 = q.mps.MPS_C(q.mps.product_state(n, c))
    m1 = q.mps.MPS_P(q.mps.product_state(n, c))
    v0 = q.mps.state_vector(m0)
    v1 = q.mps.state_vector(m1)
    assert np.allclose(v0, q.sv.state_vector(n, c))
    assert np.allclose(v1, q.sv.state_vector(n, c))


if __name__ == "__main__":
    test_mps()
