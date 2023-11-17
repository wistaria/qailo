import numpy as np
import qailo as q


def test_mps():
    states = [q.sv.one(), q.sv.one(), q.sv.zero(), q.sv.zero()]
    v = q.sv.product_state(states)
    m0 = q.mps.product_state(states)
    m1 = q.mps.product_state(states, mps=q.mps.projector_mps)
    v0 = q.mps.state_vector(m0)
    v1 = q.mps.state_vector(m1)
    assert np.allclose(v0, v)
    assert np.allclose(v1, v)


if __name__ == "__main__":
    test_mps()
