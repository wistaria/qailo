import numpy as np
import qailo as q
from qailo.mps_p.mps_p import MPS_P


def test_mps():
    states = [q.sv.one(), q.sv.one(), q.sv.zero(), q.sv.zero()]
    v = q.sv.product_state(states)
    m0 = q.mps.product_state(states, MPS_P)
    v0 = q.mps.state_vector(m0)
    assert np.allclose(v0, v)


if __name__ == "__main__":
    test_mps()
