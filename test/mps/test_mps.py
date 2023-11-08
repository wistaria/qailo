import numpy as np
import qailo as q


def test_mps():
    states = [q.sv.one(), q.sv.one(), q.sv.zero(), q.sv.zero()]
    v = q.sv.product_state(states)
    m0 = q.mps.product_state(states)
    m1 = q.mps_p.product_state(states)
    m2 = q.mps_t.product_state(states)
    v0 = q.mps.state_vector(m0)
    v1 = q.mps.state_vector(m1)
    v2 = q.mps.state_vector(m2)
    assert np.allclose(v0, v)
    assert np.allclose(v1, v)
    assert np.allclose(v2, v)


if __name__ == "__main__":
    test_mps()
