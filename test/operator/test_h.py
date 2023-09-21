import numpy as np
import qailo as q

def test_h():
    assert q.op.is_unitary(q.op.h())
    assert q.is_equal(q.op.h(), (q.op.x() + q.op.z()) / np.sqrt(2))
